import argparse
import time
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from LVLMInterface.minigpt4_interface.minigpt4.common.registry import registry


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle
    sep: str = "###"
    sep2: str = None
    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id,
        )

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


CONV_VISION = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
    "Describe this image in detail.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_VISION2 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


class Chat:
    def __init__(self, model, vis_processor, device="cuda:0"):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        stop_words_ids = [
            torch.tensor([835]).to(self.device),
            torch.tensor([2277, 29937]).to(self.device),
        ]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )

    def ask(self, text, conv):
        if (
            len(conv.messages) > 0
            and conv.messages[-1][0] == conv.roles[0]
            and conv.messages[-1][1][-6:] == "</Img>"
        ):  # last message is image.
            conv.messages[-1][1] = " ".join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(
        self,
        conv,
        img_list,
        max_new_tokens=300,
        num_beams=3,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1,
        temperature=1.0,
        max_length=2000,
    ):
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print(
                "Warning: The number of tokens in current conversation exceeds the max length. "
                "The model will not see the contexts outside the range."
            )
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            no_repeat_ngram_size=3,
        )
        output_token = outputs[0]
        if (
            output_token[0] == 0
        ):  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if (
            output_token[0] == 1
        ):  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(
            output_token, add_special_tokens=False
        )
        output_text = output_text.split("###")[0]  # remove the stop sign '###'
        output_text = output_text.split("Assistant:")[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def upload_img(self, image, conv, img_list):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert("RGB")
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        image_emb, _ = self.model.encode_img(image)
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        return msg

    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        prompt_segs = prompt.split("<ImageHere>")
        assert (
            len(prompt_segs) == len(img_list) + 1
        ), "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0
            )
            .to(self.device)
            .input_ids
            for i, seg in enumerate(prompt_segs)
        ]
        # print("seg_tokens",seg_tokens)
        seg_embs = [
            self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens
        ]
        # print("seg_embs",seg_embs)
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [
            seg_embs[-1]
        ]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        # print("mixed_embs",mixed_embs)
        return mixed_embs

    def get_language_emb(self, conv):
        prompt = conv.get_prompt()
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0
            )
            .to(self.device)
            .input_ids
            for i, seg in enumerate(prompt)
        ]
        seg_embs = [
            self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens
        ]
        return seg_embs

    def language_answer(
        self,
        conv,
        max_new_tokens=300,
        num_beams=3,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1,
        temperature=0.95,
        max_length=2000,
    ):
        conv.append_message(conv.roles[1], None)
        embs = self.get_language_emb(conv)
        outputs = self.model.llama_model.generate(
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if (
            output_token[0] == 0
        ):  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if (
            output_token[0] == 1
        ):  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(
            output_token, add_special_tokens=False
        )
        print("output_text:", output_text)
        output_text = output_text.split("###")[0]
        output_text = output_text.split("Assistant:")[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def language_answer2(
        self,
        prompt,
        max_new_tokens=300,
        num_beams=5,
        min_length=1,
        top_p=0.85,
        repetition_penalty=1.0,
        length_penalty=1,
        temperature=0.9,
    ):
        tokenizer = LlamaTokenizer.from_pretrained(
            "/data/share/pyz/ModaFew/checkpoint/vicuna-7b"
        )
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids = torch.as_tensor(input_ids)
        input_ids = input_ids.to(self.device)
        outputs = self.model.llama_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            no_repeat_ngram_size=3,
        )
        print(outputs)
        print(tokenizer.decode(outputs[0]))