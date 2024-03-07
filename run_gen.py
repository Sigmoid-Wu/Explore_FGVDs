import importlib
import os
import sys
import argparse
import json

sys.path.append("/home/hyh30/descriptionRAG/")
parser = argparse.ArgumentParser()

# Model arguments
parser.add_argument(
    "--model",
    type=str,
    help="Model name. `openflamingo` or `idefics` is supported.",
    default="idefics",
)
parser.add_argument(
    "--model_cfg", type=str, default="/home/hyh30/descriptionRAG/ide.yaml"
)
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
parser.add_argument("--dataset_root", type=str, default="/data/hyh/CUB_200_2011/")

# generation arguments
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--describe", action="store_true")
parser.add_argument("--instruction", action="store_true")
parser.add_argument("--clip", action="store_true")
# parser.add_argument(
#     "--num_samples",
#     type=int,
#     default=-1,
#     help="Number of samples to evaluate on. -1 for all samples.",
# )
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


def main():

    args = parser.parse_args()
    assert os.path.exists(
        args.model_cfg
    ), f"Model config file {args.model_cfg} not found"
    module = importlib.import_module(
        f"generate_description.eval.description_{args.model}"
    )
    generate_fn = getattr(module, "generate_description")
    for shot in args.shots:
        for seed in args.trial_seeds:
            descriptions, ids, train_data = generate_fn(
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
            for _, (idx, des) in enumerate(zip(ids, descriptions))
        ]
        if args.dataset_type == "test":
            results_file_path = os.path.join(
                f"/home/hyh30/descriptionRAG/data/{args.model}/{args.dataset_name}",
                f"{args.dataset_name}_description_{args.dataset_type}_{'RS' if not args.clip else 'SIIR'}_{shot}-shot_{args.others}_instruction{args.instruction}.json",
            )
        else:
            results_file_path = os.path.join(
                f"/home/hyh30/descriptionRAG/data/{args.model}/{args.dataset_name}",
                f"{args.dataset_name}_description_{args.dataset_type}_{args.others}_instruction{args.instruction}.json",
            )

        with open(results_file_path, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
