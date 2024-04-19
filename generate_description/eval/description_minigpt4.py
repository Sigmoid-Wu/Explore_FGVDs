import sys

dir_path = "YOUR PATH"
sys.path.append(dir_path)
from LVLMInterface.minigpt4_interface import MiniGPT4Interface
from prompt_utils import *
import argparse
import importlib
import json
import os
import uuid
import random
from collections import defaultdict

import numpy as np
import torch
import generate_description.eval.utils as utils
import math
from tqdm import tqdm
import transformers

transformers.logging.set_verbosity_error()


from generate_description.eval.eval_datasets import *
from generate_description.eval.classification_utils import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` is supported.",
    default="open_flamingo",
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

parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--dataset_name", type=str, default="cub200")
parser.add_argument("--dataset_root", type=str, default="")
parser.add_argument("--device", type=str, default="2")
parser.add_argument("--describe", action="store_true")
parser.add_argument("--others", type=str, default="alllen60")
parser.add_argument("--max_new_tokens", type=int, default=60)
parser.add_argument("--cfg_path", type=str, default="")
args = parser.parse_args()
if __name__ == "__main__":
    import sys

    set_seed(42)

    import time
    import torch
    from PIL import Image

    gpu_id = "4"
    time_begin = time.time()
    interface = MiniGPT4Interface(
        config_path=args.cfg_path,
        gpu_id=args.device,
    )
    if args.dataset_name == "imagenet_subset":
        train_dataset = ImageNetDataset(os.path.join(args.dataset_root, "train"))
        test_dataset = ImageNetDataset(os.path.join(args.dataset_root, "val"))
        all_class_names = IMAGENET_CLASSNAMES
    elif args.dataset_name == "cub200":
        train_dataset = CUB200Dataset(root=args.dataset_root)
        test_dataset = CUB200Dataset(root=args.dataset_root, train=False)
        all_class_names = CUB_CLASSNAMES
    elif args.dataset_name == "stanford_car":
        train_dataset = StanfordCarDataset(
            root=(os.path.join(args.dataset_root, "train"))
        )
        test_dataset = StanfordCarDataset(
            root=(os.path.join(args.dataset_root, "test"))
        )
        all_class_names = STANFORD_CAR_CLASSNAMES
    elif args.dataset_name == "stanford_dog":
        train_dataset = StanfordDogDataset(root=args.dataset_root)
        test_dataset = StanfordDogDataset(root=args.dataset_root, train=False)

        all_class_names = STANFORD_DOG_CLASSNAMES
    elif args.dataset_name == "flower":
        train_dataset = Flowers102Dataset(root=args.dataset_root, train=True)
        test_dataset = Flowers102Dataset(root=args.dataset_root, train=False)
    else:
        raise ValueError(f"Unsupported dataset {args.dataset_name}")

    name_map = {
        "imagenet_subset": "this thing",
        "cub200": "bird",
        "stanford_dog": "dog",
        "stanford_car": "car",
        "flower": "flower",
    }
    data_set = train_dataset if args.dataset_type == "train" else test_dataset
    question = (
        f"What distinctive visual features can be used to describe {name_map[args.dataset_name]} in this image?"
        if args.describe
        else f"What are the main elements in this image, and how do they interact or relate to each other?"
    )
    # prompt_fn = generate_description_prompt(args.dataset_name)
    result = []
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=utils.custom_collate_fn,
    )
    for i, batch in tqdm(
        enumerate(data_set),
        desc=f"Running {args.others}-{args.dataset_type} on {args.dataset_name}",
        total=len(data_set),
    ):
        if i == 0:
            print(question)

        desc = interface.zero_shot_generation(
            batch["image"],
            query=f"{question}",
            max_new_tokens=args.max_new_tokens,
        )
        # print(desc)
        # desc = interface.batch_description_generation(batch["image"], prompts=prompts)
        # desc = interface.batch_generation(batch["image"], query=prompts[0])
        res = {
            "document": {
                "image_id": batch["id"],
                "class_id": batch["class_id"],
                "class_name": batch["class_name"],
                "class_description": desc,
            }
        }

        result.append(res)

    results_file_path = (
        f"{dir_path}/data/minigpt4/"
        + f"{args.dataset_name}_description_{args.dataset_type}_0-shot_{args.others}.json"
        if args.dataset_type == "test"
        else f"{dir_path}/data/minigpt4/"
        + f"{args.dataset_name}_description_{args.dataset_type}_{args.others}.json"
    )
    with open(results_file_path, "w") as f:
        json.dump(result, f, indent=4)
    # print(f'The zero-shot answer: {desc}')
