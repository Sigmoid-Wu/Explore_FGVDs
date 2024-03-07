import argparse
import importlib
import json
import os
import random
from collections import defaultdict
import sys

sys.path.append("/home/hyh30/descriptionRAG")
import numpy as np
import torch
import utils
import math
from tqdm import tqdm
import transformers
from omegaconf import OmegaConf

transformers.logging.set_verbosity_error()
from transformers import IdeficsForVisionText2Text, AutoProcessor, AutoTokenizer
from generate_description.eval.eval_model import BaseEvalModel
from PIL import Image


from generate_description.eval.eval_datasets import *
from generate_description.eval.classification_utils import *
import copy

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` and `Idefics` are supported.",
    default="idefics",
)
parser.add_argument("--shots", nargs="+", default=[0, 1, 2, 4, 8], type=int)
parser.add_argument("--clip", action="store_true")
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)
parser.add_argument("--others", type=str, help="type something to make a difference.")
parser.add_argument("--iterative", action="store_true")
# Trial arguments

parser.add_argument(
    "--dataset_type",
    type=str,
    default="train",
    help="generate description for train or test dataset",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    type=int,
    default=[42],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Number of samples to evaluate on. -1 for all samples.",
)
parser.add_argument("--instruction", action="store_true")
parser.add_argument("--max_length", type=int, help="generate length")
parser.add_argument("--device", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--dataset_name", type=str, default="imagenet")
parser.add_argument("--dataset_root", type=str, default="/tmp")
parser.add_argument("--describe", action="store_true")
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="/data1/share/idefics/idefics",
    help="model path",
)


def main():
    args, leftovers = parser.parse_known_args()
    for shot in args.shots:
        for seed in args.trial_seeds:
            descriptions, ids, train_data = generate_description(
                args,
                seed=seed,
                num_shot=shot,
                max_generation_length=args.max_length,  # original 50
                num_beams=3,
                length_penalty=-2.0,
                temperature=0.2,
                top_k=20,
            )

        results = [
            {
                "document": {
                    "image_id": int(idx),
                    "class_id": train_data[idx]["class_id"],
                    "class_name": train_data[idx]["class_name"],
                    "class_description": des,
                }
            }
            for i, (idx, des) in enumerate(zip(ids, descriptions))
        ]
        if args.dataset_type == "test":
            results_file_path = os.path.join(
                f"/home/hyh30/descriptionRAG/data/idefics/{args.dataset_name}",
                f"{args.dataset_name}_description_{args.dataset_type}_{'RS' if not args.clip else 'SIIR'}_{shot}-shot_{args.others}_instruction{args.instruction}.json",
            )
        else:
            results_file_path = os.path.join(
                f"/home/hyh30/descriptionRAG/data/idefics/{args.dataset_name}",
                f"{args.dataset_name}_description_{args.dataset_type}_{args.others}_instruction{args.instruction}.json",
            )

        with open(results_file_path, "w") as f:
            json.dump(results, f, indent=4)


# def generate_description_prompt(
#     data,
#     dataset_name,
#     dataset_type,
#     batch_demo_samples,
#     description_file,
#     describe=False,
#     instruction=False,
# ):
#     # print(data)
#     class_name = data[0]
#     image = data[1]
#     if dataset_name == "cub200":
#         if describe:
#             question = f'Describe this image for classifying the {class_name if dataset_type == "train" else "bird"}'
#         # question = f'Which detailed visual features should be used to identify the {class_name if class_name is not None else "bird"} in this image?'
#         # question = f'What are the main elements in this image for a bird, including their colors, shapes, and positions, and how do these elements interact or relate to each other?'
#         else:
#             question = f'What are the main elements in this image for {class_name if dataset_type == "train" else " bird"}, and how do these elements interact or relate to each other?'
#         # question = f'Describe this image for classifying the {class_name if class_name is not None else " bird"}'
#         # answer = ''
#     elif dataset_name == "imagenet":
#         if describe:
#             question = f'Describe this image for classifying the {class_name if dataset_type == "train" else "this object"}'
#         # question = f'Which detailed visual features should be used to identify the {class_name if class_name is not None else "bird"} in this image?'
#         # question = f'What are the main elements in this image for a bird, including their colors, shapes, and positions, and how do these elements interact or relate to each other?'
#         else:
#             question = f'What are the main elements in this image for {class_name if dataset_type == "train" else "this object"}, and how do these elements interact or relate to each other?'
#         # question = f'Describe this image for classifying the {class_name if class_name is not None else " bird"}'
#         # answer = ''
#     elif dataset_name == "stanford_dog":
#         if describe:
#             question = f'Describe this image for classifying the {class_name if dataset_type == "train" else "dog"}'
#         # question = f'Which detailed visual features should be used to identify the {class_name if class_name is not None else "bird"} in this image?'
#         # question = f'What are the main elements in this image for a bird, including their colors, shapes, and positions, and how do these elements interact or relate to each other?'
#         else:
#             question = f'What are the main elements in this image for {class_name if dataset_type == "train" else "dog"}, and how do these elements interact or relate to each other?'
#         # question = f'Describe this image for classifying the {class_name if class_name is not None else " bird"}'
#         # answer = ''
#     elif dataset_name == "stanford_car":
#         if describe:
#             question = f'Describe this image for classifying the {class_name if dataset_type == "train" else "car"}'
#         # question = f'Which detailed visual features should be used to identify the {class_name if class_name is not None else "bird"} in this image?'
#         # question = f'What are the main elements in this image for a bird, including their colors, shapes, and positions, and how do these elements interact or relate to each other?'
#         else:
#             question = f'Which detailed visual features should be used to identify the {class_name if dataset_type == "train" else "car"} in this image?'
#             # question = f'What are the main elements in this image for {class_name if class_name is not None else "car"}, and how do these elements interact or relate to each other?'
#         # question = f'Describe this image for classifying the {class_name if class_name is not None else " bird"}'
#         # answer = ''
#     # answer = ""
#     contentvqa = "provide an answer to the question. Use the image to answer."
#     contentcap = "provide an description and use the image to answer."
#     prompt = []
#     if instruction:
#         prompt.append(f"Instruction: {contentvqa if not describe else contentcap}\n")
#     prompt.extend(
#         [
#             "User: ",
#             image,
#             f"{question}\nAssistant:",
#         ]
#     )
#     return prompt
def generate_description_prompt(
    data,
    dataset_name,
    dataset_type,
    batch_demo_samples,
    descriptions,  # 这里传入的是image_id到描述的映射
    name_map,
    describe=False,
    instruction=False,
):
    class_name, image = data[0], data[1]
    prompt = []

    if instruction:
        contentvqa = "Provide an answer to the question. Use the image to answer."
        contentcap = "Provide a description and use the image to describe."
        prompt.append(f"Instruction: {contentvqa if not describe else contentcap}\n")
    if describe:
        question = f'Describe the main elements in this image for classifying {class_name if dataset_type == "train" else name_map[dataset_name]}, and how do these elements interact or relate to each other?'
    else:
        question = f'What are the main elements in this image for classifying {class_name if dataset_type == "train" else name_map[dataset_name]}, and how do these elements interact or relate to each other?'
    if dataset_type == "test":
        for demo_image in batch_demo_samples:
            demo_image_id = demo_image["id"]
            # 从descriptions中获取对应的描述信息
            description_info = descriptions.get(demo_image_id)
            description = (
                description_info["class_description"]
                if description_info
                else "Description not available."
            )

            prompt.extend(
                [
                    "User: ",
                    demo_image["image"],
                    f"{question}\nAssistant: {description}\n",
                ]
            )

    prompt.extend(["User: ", image, f"{question}\nAssistant: "])
    return prompt


# def generate_description_prompt_iterative(data, dataset_name, misclassified, dataset):
#     class_name = data["class_name"]
#     image = data["image"]
#     actual_class_id = data["class_id"]
#     misclassified_image = None  # 初始化为 None 或者默认值
#     misclassified_class_name = ""  # 同样初始化
#     if str(actual_class_id) in misclassified:
#         most_predicted_id = misclassified[str(actual_class_id)]
#         # 找到数据集中属于most_predicted_id类别的所有图片索引
#         misclassified_image_indexes = [
#             i for i, x in enumerate(dataset.labels) if x == most_predicted_id
#         ]
#         # 如果确实有属于这个类别的图片
#         if misclassified_image_indexes:
#             random_index = random.choice(misclassified_image_indexes)
#             # 假设dataset可以直接用索引访问，获取被错误分类的类别名称
#             misclassified_data = dataset[random_index]
#             misclassified_image = misclassified_data["image"]
#             misclassified_class_name = misclassified_data["class_name"]
#             question = f""
#         else:
#             # 如果找不到任何属于most_predicted_id的图片，回退到默认问题
#             question = f'Describe this image for classifying the {class_name if class_name is not None else " bird"}'
#     else:
#         question = f'Describe this image for classifying the {class_name if class_name is not None else " bird"}'
#     prompt = [
#         "Instruction: Analyze the following images.\n",
#         "User: ",
#         image,
#         class_name,
#         misclassified_image,
#         misclassified_class_name,
#         "Describe distinguishing features present in the First image like shape, color or size which the second doesn't have.\nAssistant:",
#     ]

#     return prompt


def postprocess_description_generation_ide(prediction):
    if "Assistant" in prediction:
        extracted_text = prediction.split("Assistant:")[-1].strip()
        extracted_text = extracted_text.split("\n")[0].strip()
    else:
        extracted_text = prediction.split("Answer:")[1]
    return extracted_text


def custom_collate_fn(batch):
    """
    Collate function for DataLoader that collates a list of dicts into a dict of lists.
    """
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def prepare_eval_samples(dataset, num_samples, batch_size, seed):
    np.random.seed(seed)
    if num_samples == -1:
        num_samples = len(dataset)
    random_indices = np.random.choice(len(dataset), num_samples, replace=False)
    t_dataset = torch.utils.data.Subset(dataset, random_indices)
    loader = torch.utils.data.DataLoader(
        t_dataset,
        batch_size=batch_size,
        shuffle=True,  # Enable shuffling for non-distributed loading
        collate_fn=custom_collate_fn,
        num_workers=4,
    )
    return loader


def sample_batch_demos_from_query_set(query_set, num_shots, batch, clip=False):
    if not clip:
        return [
            [query_set[i] for i in random.sample(range(len(query_set)), num_shots)]
            for _ in range(len(batch))
        ]
    else:
        output = []
        for i in range(len(batch["id"])):
            o = []
            for _, id in enumerate(batch["clip_similar_ids"][i][:num_shots]):
                x = copy.deepcopy(query_set.id2item(id))
                o.append(x)
            output.append(o)
        return output


def generate_description(
    args: argparse.Namespace,
    seed: int = 42,
    num_shot: int = 1,
    max_generation_length: int = 50,
    num_beams: int = 3,
    length_penalty: float = -2.0,
    no_repeat_ngram_size: int = 2,
    temperature: float = 0.2,
    top_k: int = 10,
):
    """
    Evaluate a model on classification dataset.

    Args:
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        num_shots (int, optional): number of shots to use. Defaults to 1.
        dataset_name (str, optional): dataset name. Defaults to "imagenet".
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.

    Returns:
        float: accuracy score
    """
    np.random.seed(seed)
    cfg = OmegaConf.load(args.model_cfg)

    if args.dataset_name == "imagenet_subset":
        train_dataset = ImageNetDataset(os.path.join(args.dataset_root, "train"))
        test_dataset = ImageNetDataset(os.path.join(args.dataset_root, "val"))
    elif args.dataset_name == "cub200":
        train_dataset = CUB200Dataset(root=args.dataset_root)
        test_dataset = CUB200Dataset(root=args.dataset_root, train=False)
    elif args.dataset_name == "stanford_car":
        train_dataset = StanfordCarDataset(
            root=(os.path.join(args.dataset_root, "train")), train=True
        )
        test_dataset = StanfordCarDataset(
            root=(os.path.join(args.dataset_root, "test")), train=False
        )
    elif args.dataset_name == "stanford_dog":
        train_dataset = StanfordDogDataset(root=args.dataset_root)
        test_dataset = StanfordDogDataset(root=args.dataset_root, train=False)
    else:
        raise ValueError(f"Unsupported dataset {args.dataset_name}")

    if args.dataset_type == "train":
        dataloader = prepare_eval_samples(
            train_dataset,
            args.num_samples,
            args.batch_size,
            seed=seed,
        )
    else:
        dataloader = prepare_eval_samples(
            test_dataset, args.num_samples, args.batch_size, seed=seed
        )

    np.random.seed(seed)
    # model = model.to_bettertransformer()
    pretrained_model_path = cfg.pretrained_model_path
    model = IdeficsForVisionText2Text.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        cache_dir=pretrained_model_path,
    ).to(args.device)
    processor = AutoProcessor.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        cache_dir=cfg.processor_path,
    )
    all_outputs = []
    all_ids = []
    cnt = 0
    name_map = {
        "imagenet_subset": "this thing",
        "cub200": "bird",
        "stanford_dog": "dog",
        "stanford_car": "car",
    }
    import time

    batch_inference_times = []
    if args.dataset_type == "test":
        file_path = f"/home/hyh30/descriptionRAG/data/{args.model}/{args.dataset_name}/{args.dataset_name}_description_train_{args.others}.json"
        with open(file_path, "r") as file:
            description_file = json.load(file)
        descriptions = {
            item["document"]["image_id"]: item["document"] for item in description_file
        }
    else:
        descriptions = None
        batch_demo_samples = None
    np.random.seed(seed)
    for _, batch in tqdm(
        enumerate(dataloader),
        desc=f"Generating description on {args.dataset_name}",
        total=len(dataloader),
    ):
        if args.dataset_type == "test":
            batch_demo_samples = sample_batch_demos_from_query_set(
                train_dataset, num_shot, batch, args.clip
            )
        # print(batch)
        # Prepare text and images for the batch
        batch_text = [
            generate_description_prompt(
                x,
                args.dataset_name,
                args.dataset_type,
                batch_demo_samples[index] if args.dataset_type == "test" else None,
                descriptions,
                name_map,
                args.describe,
                args.instruction,
            )
            for index, x in enumerate(
                zip(
                    batch["class_name"],
                    batch["image"],
                )
            )
        ]
        # If there is image data in the batch, send it to the model
        if batch_text:
            if cnt == 0:
                print(batch_text[0])
                cnt += 1
            # Call the model's method to get outputs
            # (modify this according to your model and needs)
            inputs = processor(batch_text, return_tensors="pt").to(args.device)
            exit_condition = processor.tokenizer(
                "<end_of_utterance>", add_special_tokens=False
            ).input_ids
            bad_words_ids = processor.tokenizer(
                ["<image>", "<fake_token_around_image>"], add_special_tokens=False
            ).input_ids
            start_time = time.time()
            generated_outputs = model.generate(
                **inputs,
                max_new_tokens=max_generation_length,
                eos_token_id=exit_condition,
                bad_words_ids=bad_words_ids,
                output_scores=True,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                temperature=temperature,
                top_k=top_k,
                return_dict_in_generate=True,
            )
            generated_ids = generated_outputs.sequences
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            print(generated_text[0])
            # break
            end_time = time.time()
            batch_inference_times.append((end_time - start_time) * 1000)
            new_predictions = [
                postprocess_description_generation_ide(out).replace('"', "")
                for out in generated_text
            ]
            # print(new_predictions[0])
            # print(batch['id'])
            all_outputs.extend(new_predictions)
            all_ids.extend(batch["id"])
        # ensemble logprobs together
    average_inference_time = sum(batch_inference_times) / len(batch_inference_times)
    print(f"Average Inference Time per Batch: {average_inference_time} seconds")
    return (
        all_outputs,
        all_ids,
        train_dataset if args.dataset_type == "train" else test_dataset,
    )


if __name__ == "__main__":
    main()
