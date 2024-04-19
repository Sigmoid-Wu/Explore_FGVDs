import argparse
import importlib
import json
import os
from typing import List, Tuple, Dict, Union, Callable
import random
import copy
from collections import defaultdict
from omegaconf import OmegaConf
import numpy as np
import torch
import generate_description.eval.utils as utils
import math
from tqdm import tqdm
import transformers

transformers.logging.set_verbosity_error()

from generate_description.eval.models.open_flamingo import EvalModel as OpenFlamingo

from generate_description.eval.eval_datasets import *
from generate_description.eval.classification_utils import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` is supported.",
    default="open_flamingo",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Number of samples to evaluate on. -1 for all samples.",
)
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 1, 2, 4, 8], type=int)
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
    "--device", type=int, default=0, help="Device to use for evaluation"
)

parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--describe", action="store_true")
parser.add_argument("--instruction", action="store_true")
parser.add_argument(
    "--classification_prompt_ensembling",
    action="store_true",
    help="Whether to use prompt ensembling (average log-likelihoods over permutations of in-context examples)",
)
parser.add_argument(
    "--rices_type",
    default=None,
    help="Type to use RICES for generate description in this demo.",
)

parser.add_argument(
    "--rices_vision_encoder_path",
    default="ViT-L-14",
    type=str,
    help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--rices_vision_encoder_pretrained",
    default="openai",
    type=str,
    help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--cached_demonstration_features",
    default=None,
    help="Directory where rices features for all choices of in-context examples are stored as a pkl file with the dataset name. If None, features are re-computed by script.",
)
parser.add_argument("--dataset_name", type=str, default="imagenet")
parser.add_argument("--dataset_root", type=str, default="/data/")
parser.add_argument("--max_length", type=int, default=50)
parser.add_argument("--model_cfg", type=str, default="of9b.yaml")


parser.add_argument(
    "--others",
    type=str,
    default="",
    help="Other information to add to the results file name.",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():

    args = parser.parse_args()

    for seed in args.trial_seeds:
        descriptions, ids, train_data = generate_description(
            args,
            batch_size=args.batch_size,
            seed=seed,
            dataset_name=args.dataset_name,
            max_generation_length=args.max_length,
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

    results_file_path = os.path.join(
        f"data/openflamingo9B/",
        f"{args.dataset_name}_description_{args.dataset_type}_{args.others}.json",
    )

    with open(results_file_path, "w") as f:
        json.dump(results, f, indent=4)


caption_instruction = f"<image>Describe this image in detail: The image shows"


def sample_batch_demos_from_query_set(
    query_set: torch.utils.data.Dataset,
    num_shots: int,
    batch: Dict[str, List],
    sim_text: bool = False,
) -> List[List]:
    batch_size = len(batch["id"])

    if sim_text:
        outputs = [
            [
                # copy.deepcopy(query_set.id2item(id))
                query_set[id]
                for id in batch["similar_text_ids"][i][:num_shots]
            ]
            for i in range(len(batch["class_name"]))
        ]
        return outputs

    return [
        [query_set[i] for i in random.sample(range(len(query_set)), num_shots)]
        for _ in range(batch_size)
    ]


def sample_batch_txt_sim_demos(query_set, num_shots, batch):

    output = []
    for i in range(len(batch["class_name"])):
        o = []
        for _, id in enumerate(batch["similar_text_ids"][i][:num_shots]):
            x = copy.deepcopy(query_set.id2item(id))
            o.append(x)
        output.append(o)
    return output


def generate_k_shot_template(k: int, prompts: str, assit_answers: List[str]) -> str:
    """ """
    assert k > 0, "k must be greater than 0"
    template = "".join(
        [
            f"<image>Question:{prompts[i]} Answer:{assit_answers[i]}<|endofchunk|>"
            for i in range(k)
        ]
    )
    template += f"<image>Question:{prompts[-1]} Answer:"
    return template


def prompt_fn(
    dataset_name: str,
    dataset_type: str,
    class_name: str,
    name_map: Dict,
    describe: bool = False,
) -> str:

    if describe:
        return f"What distinctive visual features can be used to describe {name_map[dataset_name]} in this image?"
    return f"What are the main elements in this image, and how do they interact or relate to each other?"


def generate_batch_prompts(
    args: argparse.Namespace,
    name_map: Dict,
    k: int,
    batch: Dict[str, List],
    batch_demo_samples: Union[None, List[List]],
    descriptions: Dict[int, Dict],
) -> List[str]:
    question = (
        f"What distinctive visual features can be used to describe {name_map[args.dataset_name]} in this image?"
        if args.describe
        else f"What are the main elements in this image, and how do they interact or relate to each other?"
    )
    if batch_demo_samples is None:
        # batch_prompts = [
        #     question
        #     # prompt_fn(args.dataset_name, args.dataset_type, x, name_map, args.describe)
        #     for _ in batch["class_name"]
        # ]
        return [
            f"<image>Question:{question} Answer:"
            for _ in range(len(batch["class_name"]))
        ]
    return [
        "".join(
            [
                f"<image>Question:{question} Answer:{descriptions[item['id']]['class_description']}<|endofchunk|>"
                for item in demo_samples
            ]
        )
        + f"<image>Question:{question} Answer:"
        for demo_samples in batch_demo_samples
    ]


def generate_description(
    args: argparse.Namespace,
    seed: int = 42,
    num_shot: int = 1,
    max_generation_length: int = 50,
    num_beams: int = 3,
    length_penalty: float = 2.0,
    no_repeat_ngram_size: int = 2,
) -> Tuple[List, List[int], torch.utils.data.Dataset]:
    """
    Evaluate a model on classification dataset.

    Args:
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (str, optional): dataset name. Defaults to "imagenet".
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.

    Returns:
        float: accuracy score
    """

    model_name = "openflamingo9B" if "9b" in args.model_cfg else args.model

    cfg = OmegaConf.load(args.model_cfg)
    eval_model = OpenFlamingo(cfg)

    eval_model.set_device(args.device)

    if args.dataset_name == "imagenet_subset":
        train_dataset = ImageNetDataset(os.path.join(args.dataset_root, "train"))
        # test_dataset = ImageNetDataset(
        #     os.path.join(args.dataset_root, "val"),
        #     similar_images_path=similar_text_path,
        # )
        test_dataset = ImageNetDataset(os.path.join(args.dataset_root, "val"))
    elif args.dataset_name == "cub200":
        train_dataset = CUB200Dataset(root=args.dataset_root)
        # test_dataset = CUB200Dataset(
        #     root=args.dataset_root, train=False, similar_text_path=similar_text_path
        # )
        test_dataset = CUB200Dataset(root=args.dataset_root, train=False)
    elif args.dataset_name == "stanford_car":
        train_dataset = StanfordCarDataset(
            root=(os.path.join(args.dataset_root, "train"))
        )
        test_dataset = StanfordCarDataset(
            root=(os.path.join(args.dataset_root, "test")),
            # similar_text_path=similar_text_path,
        )
    elif args.dataset_name == "stanford_dog":
        train_dataset = StanfordDogDataset(root=args.dataset_root)
        test_dataset = StanfordDogDataset(
            root=args.dataset_root,
            train=False,
            #   similar_text_path=similar_text_path
        )
    elif args.dataset_name == "flower":
        train_dataset = Flowers102Dataset(root=args.dataset_root, train=True)
        test_dataset = Flowers102Dataset(root=args.dataset_root, train=False)
    else:
        raise ValueError(f"Unsupported dataset {args.dataset_name}")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if args.dataset_type == "train":
        dataloader = utils.prepare_eval_samples(
            train_dataset,
            args.num_samples,
            args.batch_size,
            seed=seed,
        )
    else:
        dataloader = utils.prepare_eval_samples(
            test_dataset, args.num_samples, args.batch_size, seed=seed
        )
    name_map = {
        "imagenet_subset": "this thing",
        "cub200": "bird",
        "stanford_dog": "dog",
        "stanford_car": "car",
        "flower": "flower",
    }
    all_outputs = []
    all_ids = []
    cnt = 0
    import time

    batch_inference_times = []
    descriptions = None

    if args.dataset_type == "test" and num_shot > 0:
        file_path = f"/descriptionRAG/data/{model_name}/{args.dataset_name}/{args.dataset_name}_description_train_{args.others}_nolabel.json"
        assert os.path.exists(
            file_path
        ), f"Training description file {file_path} does not exist"
        with open(file_path, "r") as file:
            description_file = json.load(file)
        descriptions = {
            item["document"]["image_id"]: item["document"] for item in description_file
        }

    is_icl = False if num_shot == 0 else True
    for _, batch in tqdm(
        enumerate(dataloader),
        desc=f"device:{args.device} Running {args.sim_text}-{args.others}-{args.dataset_name} {num_shot}-shot",
        total=len(dataloader),
    ):

        batch_demo_samples = (
            sample_batch_demos_from_query_set(
                query_set=train_dataset,
                num_shots=num_shot,
                batch=batch,
                sim_text=args.sim_text,
            )
            if is_icl
            else None
        )
        batch_text = generate_batch_prompts(
            args=args,
            name_map=name_map,
            k=num_shot,
            batch=batch,
            batch_demo_samples=batch_demo_samples,
            descriptions=descriptions,
        )

        batch_images = (
            [[image_data] for image_data in batch["image"]]
            if not is_icl
            else [
                [item["image"] for item in demo_samples] + [batch["image"][i]]
                for i, demo_samples in enumerate(batch_demo_samples)
            ]
        )

        # If there is image data in the batch, send it to the model
        if batch_text:
            if cnt == 0:
                print(batch_text[0])
                cnt += 1

            outputs = eval_model.get_outputs(
                batch_text=batch_text,
                batch_images=batch_images,
                min_generation_length=15,
                max_generation_length=max_generation_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                # temperature=temperature,
                # top_k=top_k,
            )

            all_outputs.extend(outputs)
            all_ids.extend(batch["id"])

    return (
        all_outputs,
        all_ids,
        train_dataset if args.dataset_type == "train" else test_dataset,
    )


if __name__ == "__main__":
    main()
