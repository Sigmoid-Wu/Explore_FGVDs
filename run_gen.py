import importlib
import os
import sys
import argparse
import json
from typing import List, Tuple, Callable, Dict, Any

dir_path = "YOUR PATH"
sys.path.append(dir_path)
parser = argparse.ArgumentParser()

# Model arguments
parser.add_argument(
    "--model",
    type=str,
    help="Model name. `openflamingo` or `idefics` is supported.",
    default="idefics",
)
parser.add_argument("--model_cfg", type=str, default="/ide.yaml")
parser.add_argument("--device", type=int, default=1)

# results arguments
parser.add_argument(
    "--results_path", type=str, default=None, help="JSON file to save results"
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 1, 2], type=int)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    type=int,
    default=[42],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
# dataset arguments
parser.add_argument(
    "--dataset_type",
    type=str,
    default="test",
    help="generate description for train or test dataset",
)
parser.add_argument("--dataset_name", type=str, default="cub200")
parser.add_argument("--dataset_root", type=str, default="")
parser.add_argument("--sim_text", action="store_true")
# generation arguments
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--describe", action="store_true")
parser.add_argument("--instruction", action="store_true")
parser.add_argument("--clip", action="store_true")
parser.add_argument("--high_des", action="store_true")
parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Number of samples to evaluate on. -1 for all samples.",
)
parser.add_argument("--max_length", type=int, default=50)
parser.add_argument(
    "--others", type=str, default="len50", help="type specific experiments"
)
parser.add_argument("--length_penalty", type=float, default=-2.0)


def main():

    args = parser.parse_args()
    assert os.path.exists(
        args.model_cfg
    ), f"Model config file {args.model_cfg} not found"
    module = importlib.import_module(
        f"generate_description.eval.description_{args.model}"
    )
    generate_fn: Callable[..., Any] = getattr(module, "generate_description")
    model_name = "openflamingo9B" if "9b" in args.model_cfg else args.model
    print(
        f"Generating descriptions for {model_name} model on {args.dataset_name} dataset."
    )
    for shot in args.shots:
        for seed in args.trial_seeds:
            descriptions, ids, train_data = generate_fn(
                args,
                seed=seed,
                num_shot=shot,
                max_generation_length=args.max_length,  # original 50
                num_beams=3,
                length_penalty=args.length_penalty,
                # temperature=0.2,
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
            for _, (idx, des) in enumerate(zip(ids, descriptions))
        ]
        if args.dataset_type == "test":
            if shot == 0:
                results_file_path = os.path.join(
                    f"{dir_path}/data/{model_name}/{args.dataset_name}",
                    f"{args.dataset_name}_description_{args.dataset_type}_{shot}-shot_{args.others}_nolabel.json",
                )
            else:
                if args.clip:
                    results_file_path = os.path.join(
                        f"{dir_path}/data/{model_name}/{args.dataset_name}",
                        f"{args.dataset_name}_description_{args.dataset_type}_{'SIIR'}_{shot}-shot_{args.others}_nolabel.json",
                    )
                else:
                    results_file_path = os.path.join(
                        f"{dir_path}/data/{model_name}/{args.dataset_name}",
                        f"{args.dataset_name}_description_{args.dataset_type}_{'RS' if not args.sim_text else 'STTR'}_{shot}-shot_{args.others}_nolabel.json",
                    )
        else:
            if args.high_des:
                results_file_path = os.path.join(
                    f"{dir_path}/data/{model_name}/{args.dataset_name}",
                    f"{args.dataset_name}_description_{args.dataset_type}_{shot}-shot_{args.others}_nolabel.json",
                )
            else:
                results_file_path = os.path.join(
                    f"{dir_path}/data/{model_name}/{args.dataset_name}",
                    f"{args.dataset_name}_description_{args.dataset_type}_{args.others}_nolabel.json",
                )

        with open(results_file_path, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
