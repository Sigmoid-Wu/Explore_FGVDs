import torch
import numpy as np
import copy

from typing import Union, List
from PIL.Image import Image

from .minigpt4.common.config import Config
from .minigpt4.common.registry import registry
from .minigpt4.conversation.conversation import (
    Chat,
    CONV_VISION,
    Conversation,
    SeparatorStyle,
    CONV_VISION2,
)

CONV_VISION = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
    "Describe this image in detail.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


class MiniGPT4Interface:
    def __init__(self, config_path, gpu_id, **kwargs):
        print("Initializing Chat")
        gpu_id = int(gpu_id)
        self.cfg = Config(config_path, **kwargs)
        model_config = self.cfg.model_cfg
        model_config.device_8bit = gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        self.device = "cuda:{}".format(gpu_id)
        model = model_cls.from_config(model_config).to("cuda:{}".format(gpu_id))

        vis_processor_cfg = self.cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(
            vis_processor_cfg.name
        ).from_config(vis_processor_cfg)
        self.chat = Chat(model, vis_processor, device="cuda:{}".format(gpu_id))
        self.conversation_history = CONV_VISION.copy()
        self.image_list = []
        print("Initialization Finished")

    def reset(self):
        self.conversation_history = CONV_VISION.copy()
        self.image_list = []
        # print(self.conversation_history)
        # print("reset success")

    def reset2(self):
        self.conversation_history = CONV_VISION2.copy()
        # print(self.conversation_history)
        # print("reset success")

    def _chat_one_time(self, image, query, **kwargs):
        if image:
            self.chat.upload_img(image, self.conversation_history, self.image_list)
        self.chat.ask(query, self.conversation_history)
        answer = self.chat.answer(self.conversation_history, self.image_list, **kwargs)[
            0
        ]
        return answer

    def zero_shot_generation(
        self, image: Union[Image, str, torch.Tensor], query: str = "", **kwargs
    ):
        assert (
            image
        ), f"In zero_shot_generation function, the image should be Union[Image, str, torch.Tensor], but got {type(image)}"
        answer = self._chat_one_time(image, query, **kwargs)
        self.reset()
        return answer

    def few_shot_generation(
        self,
        example_images: List[Union[Image, str, torch.Tensor]],
        example_texts: List[str],
        input_images: Union[
            List[Union[Image, str, torch.Tensor]], Image, str, torch.Tensor
        ],
        query: str = "",
        **kwargs,
    ):
        assert len(example_images) == len(
            example_texts
        ), f"The few-shot image should num should be the same as the num of example_texts"
        few_shot_num = len(example_texts)
        for i in range(few_shot_num):
            self.chat.upload_img(
                example_images[i], self.conversation_history, self.image_list
            )
            self.chat.ask(query, self.conversation_history)
            self.conversation_history.append_message(
                self.conversation_history.roles[1], example_texts[i]
            )
        if not isinstance(input_images, List):
            input_images = [input_images]
        assert len(input_images) == 1, f"Now only support one image as input"
        for input_image in input_images:
            self.chat.upload_img(
                input_image, self.conversation_history, self.image_list
            )
            self.chat.ask(query, self.conversation_history)
            output_text = self.chat.answer(
                self.conversation_history, self.image_list, **kwargs
            )[0]
        self.reset()
        return output_text

    def language_generation(self, query, **kwargs):
        self.chat.ask(query, self.conversation_history)
        answer = self.chat.language_answer2(self.conversation_history, **kwargs)[0]
        return answer

    def batch_description_generation(
        self,
        batch_images,
        prompts,
        max_new_tokens=60,
        num_beams=3,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1,
        temperature=1.0,
        max_length=2000,
    ):
        batch_size = len(batch_images)
        images = [
            self.chat.vis_processor(raw_image).unsqueeze(0).to(self.device)
            for raw_image in batch_images
        ]
        image_embs = [self.chat.model.encode_img(img)[0] for img in images]
        image_embs = torch.cat(image_embs, dim=0).to(self.device)
        self.chat.upload_img(
            batch_images[0], self.conversation_history, self.image_list
        )
        self.chat.ask(prompts[0], self.conversation_history)
        prompt_segs = self.conversation_history.get_prompt().split("<ImageHere>")
        text_1_seg = (
            self.chat.model.llama_tokenizer(
                [prompt_segs[0]] * batch_size,
                return_tensors="pt",
                add_special_tokens=True,
            )
            .to(self.device)
            .input_ids
        )
        text_1_embs = self.chat.model.llama_model.model.embed_tokens(text_1_seg)
        text_2_seg = (
            self.chat.model.llama_tokenizer(prompts, return_tensors="pt", padding=True)
            .to(self.device)
            .input_ids
        )
        text_2_embs = self.chat.model.llama_model.model.embed_tokens(text_2_seg)

        inputs_embeds = torch.cat([text_1_embs, image_embs, text_2_embs], dim=1)
        # embed = self.chat.get_context_emb(
        #     conv=self.conversation_history, img_list=self.image_list, return_list=True
        # )
        # # embeds = [embed] * batch_size
        # inputs_embeds = []
        # for i, img in enumerate(images):
        #     embs = embed[:]
        #     embs[1] = self.chat.model.encode_img(img)[0]
        #     temp_emb = torch.cat(embs, dim=1)
        #     inputs_embeds.append(temp_emb.squeeze(0))
        # inputs_embeds = torch.stack(inputs_embeds, dim=0).to(self.device)

        current_max_len = inputs_embeds.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print(
                "Warning: The number of tokens in current conversation exceeds the max length. "
                "The model will not see the contexts outside the range."
            )
        begin_idx = max(0, current_max_len - max_length)

        inputs_embeds = inputs_embeds[:, begin_idx:]

        outputs = self.chat.model.llama_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.chat.stopping_criteria,
            num_beams=3,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            no_repeat_ngram_size=3,
        )
        output_token = outputs[0]
        # the model might output a unknow token <unk> at the beginning. remove it
        output_token = outputs[:, 1:]
        output_text = self.chat.model.llama_tokenizer.batch_decode(
            output_token, add_special_tokens=False
        )
        # print("output_text:", output_text)
        # output_text = output_text.split("###")[0]
        # output_text = output_text.split("Assistant:")[-1].strip()
        self.reset()
        return output_text

    def batch_generation(
        self,
        batch_images,
        query,
        max_new_tokens=60,
        num_beams=3,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1,
        temperature=1.0,
        max_length=2000,
    ):
        batch_size = len(batch_images)
        images = [
            self.chat.vis_processor(raw_image).unsqueeze(0).to(self.device)
            for raw_image in batch_images
        ]
        self.chat.upload_img(
            batch_images[0], self.conversation_history, self.image_list
        )
        self.chat.ask(query, self.conversation_history)
        embed = self.chat.get_context_emb(
            conv=self.conversation_history, img_list=self.image_list, return_list=True
        )
        # embeds = [embed] * batch_size
        inputs_embeds = []
        for i, img in enumerate(images):
            embs = embed[:]
            embs[1] = self.chat.model.encode_img(img)[0]
            temp_emb = torch.cat(embs, dim=1)
            inputs_embeds.append(temp_emb.squeeze(0))
        inputs_embeds = torch.stack(inputs_embeds, dim=0).to(self.device)

        current_max_len = inputs_embeds.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print(
                "Warning: The number of tokens in current conversation exceeds the max length. "
                "The model will not see the contexts outside the range."
            )
        begin_idx = max(0, current_max_len - max_length)

        inputs_embeds = inputs_embeds[:, begin_idx:]

        outputs = self.chat.model.llama_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.chat.stopping_criteria,
            num_beams=3,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            no_repeat_ngram_size=3,
        )
        output_token = outputs[0]
        # the model might output a unknow token <unk> at the beginning. remove it
        output_token = outputs[:, 1:]
        output_text = self.chat.model.llama_tokenizer.batch_decode(
            output_token, add_special_tokens=False
        )
        # print("output_text:", output_text)
        # output_text = output_text.split("###")[0]
        # output_text = output_text.split("Assistant:")[-1].strip()
        self.reset()
        return output_text


"""
Conversation.messages:
1) upload image
[["Human", "<Img><ImageHere></Img>"]]

2) ask
[["Human", "<Img><ImageHere></Img> Describe the image" ]]

3) answer
[["Human", "<Img><ImageHere></Img> Describe the image" ], ["Assistant", None]]

Prompt = Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img><ImageHere></Img> Describe the image###Assistant:

get_context_emb
Prompt = Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img><ImageHere></Img> Describe the image###Assistant:
prompt_segs = ["Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img>", "</Img> Describe the image###Assistant:"]

after get the answer 
[["Human", "<Img><ImageHere></Img> Describe the image" ], ["Assistant", "This is a cat!"]]

4) ask again
[["Human", "<Img><ImageHere></Img> Describe the image" ], ["Assistant", "This is a cat!"], ["Human", "Describe the image again"]]

"""
