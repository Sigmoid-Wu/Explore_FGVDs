import json
import argparse
import torch
from torch.nn.functional import cosine_similarity
import random
import numpy as np
from tqdm import tqdm
import open_clip
import pickle
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np


# Setting CLIP
def initialize_clip_model(gpu):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
    )
    model.to(gpu)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    return tokenizer, model


# text embedding
def encode_text(text, tokenizer, model, device):
    inputs = tokenizer(text).to(device)
    outputs = model.encode_text(inputs)
    outputs /= outputs.norm(dim=-1, keepdim=True)
    return outputs


# calculate similarity
def calculate_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)


# process batch data
def process_average_batch(
    batch, class_embeddings, class_index, tokenizer, model, device
):
    batch_results = []
    correct_predictions = 0

    # 批量将查询转换为嵌入

    # 将类别嵌入转换为张量
    class_embeddings_tensor = torch.stack(class_embeddings)

    with torch.no_grad():
        queries = [item["document"]["class_description"] for item in batch]
        query_embeddings = encode_text(queries, tokenizer, model, device)
        for query_idx, query_embedding in enumerate(query_embeddings):
            # 计算与每个类别嵌入的相似度
            similarities = cosine_similarity(
                query_embedding.unsqueeze(0), class_embeddings_tensor
            )
            most_similar_index = torch.argmax(similarities)
            predicted_class_id = class_index[most_similar_index.item()]

            result = {
                "image_id": batch[query_idx]["document"]["image_id"],
                "actual_class_id": batch[query_idx]["document"]["class_id"],
                "predicted_class_id": predicted_class_id,
            }
            if predicted_class_id == result["actual_class_id"]:
                correct_predictions += 1
            batch_results.append(result)

    return batch_results, correct_predictions


def process_divide_batch(
    batch, class_embeddings, class_index, tokenizer, model, device
):
    batch_results = []
    correct_predictions = 0

    class_embeddings_tensor = torch.stack(class_embeddings)
    with torch.no_grad():
        queries = [item["document"]["class_description"] for item in batch]
        query_embeddings = encode_text(queries, tokenizer, model, device)
        # 对于batch中的每个查询计算与类别嵌入的相似度
        similarities_batch = torch.stack(
            [
                torch.max(
                    cosine_similarity(
                        query_embedding.unsqueeze(0), class_embeddings_tensor
                    ),
                    dim=0,
                )[1]
                for query_embedding in query_embeddings
            ]
        )
        for i, query_embedding in enumerate(query_embeddings):
            most_similar_index = similarities_batch[i].item()
            predicted_class_id = -1
            for class_id, indices in class_index.items():
                if most_similar_index in indices:
                    predicted_class_id = class_id
                    break

            result = {
                "image_id": batch[i]["document"]["image_id"],
                "actual_class_id": batch[i]["document"]["class_id"],
                "predicted_class_id": predicted_class_id,
            }
            if predicted_class_id == result["actual_class_id"]:
                correct_predictions += 1
            batch_results.append(result)

    return batch_results, correct_predictions


def process_majority_voting_batch(
    batch, class_embeddings, class_index, tokenizer, model, device, top_n
):
    batch_results = []
    correct_predictions = 0

    # 批量将查询转换为嵌入

    # 将类别嵌入转换为张量
    class_embeddings_tensor = torch.stack(class_embeddings)

    with torch.no_grad():
        queries = [item["document"]["class_description"] for item in batch]
        query_embeddings = encode_text(queries, tokenizer, model, device)
        for query_idx, query_embedding in enumerate(query_embeddings):
            # 计算相似度
            similarities = cosine_similarity(
                query_embedding.unsqueeze(0), class_embeddings_tensor
            )
            top_indices = torch.topk(similarities, top_n).indices.squeeze()

            class_votes = {}
            for idx in top_indices:
                for class_id, indices in class_index.items():
                    if idx.item() in indices:
                        class_votes[class_id] = class_votes.get(class_id, 0) + 1

            predicted_class_id = max(class_votes, key=class_votes.get)

            result = {
                "image_id": batch[query_idx]["document"]["image_id"],
                "actual_class_id": batch[query_idx]["document"]["class_id"],
                "predicted_class_id": predicted_class_id,
            }
            if predicted_class_id == result["actual_class_id"]:
                correct_predictions += 1
            batch_results.append(result)

    return batch_results, correct_predictions


def create_mapping_from_average_json(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    return {
        item["document"]["class_id"]: item["document"]["class_description"]
        for item in data
    }


def create_mapping_from_divide_json(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    divide_mapping = {}
    for item in data:
        class_id = item["document"]["class_id"]
        description = item["document"]["class_description"]
        if class_id not in divide_mapping:
            divide_mapping[class_id] = [description]
        else:
            divide_mapping[class_id].append(description)
    return divide_mapping


def create_description_to_embedding_index_map(divide_json):
    desc_to_index_map = {}
    for index, item in enumerate(divide_json):
        desc = item["document"]["class_description"]
        desc_to_index_map[desc] = index
    return desc_to_index_map


def process_combined_batch(
    batch,
    average_embeddings,
    divide_embeddings,
    average_json_mapping,
    divide_json_mapping,
    desc_to_index_map,
    tokenizer,
    model,
    device,
    top_k,
    top_n,
):
    batch_results = []
    correct_predictions = 0
    # 转换嵌入为张量
    average_embeddings_tensor = torch.stack(average_embeddings)
    divide_embeddings_tensor = torch.stack(divide_embeddings)

    with torch.no_grad():
        queries = [item["document"]["class_description"] for item in batch]
        query_embeddings = encode_text(queries, tokenizer, model, device)
        for query_idx, query_embedding in enumerate(query_embeddings):
            avg_similarities = cosine_similarity(
                query_embedding.unsqueeze(0), average_embeddings_tensor
            )
            top_indices = torch.topk(avg_similarities, top_k).indices.squeeze()

            all_votes = []
        for idx in top_indices:
            class_id = idx.item()  # 使用索引作为类别 ID
            divide_descriptions = divide_json_mapping.get(
                class_id, []
            )  # 根据类别 ID 获取 divide 描述

            for desc in divide_descriptions:
                if desc in desc_to_index_map:
                    desc_idx = desc_to_index_map[desc]
                    desc_embedding = divide_embeddings_tensor[desc_idx]
                    similarity = calculate_similarity(
                        query_embedding.unsqueeze(0), desc_embedding.unsqueeze(0)
                    )
                    all_votes.append(
                        (class_id, similarity)
                    )  # 使用类别 ID 而非描述作为投票的键

            all_votes.sort(key=lambda x: x[1], reverse=True)
            top_votes = all_votes[:top_n]

            if top_votes:
                predicted_class_id = max(top_votes, key=lambda x: x[1])[0]
            else:
                predicted_class_id = -1  # 或选择一个默认的类别

            result = {
                "image_id": batch[query_idx]["document"]["image_id"],
                "actual_class_id": batch[query_idx]["document"]["class_id"],
                "predicted_class_id": predicted_class_id,
            }
            if predicted_class_id == result["actual_class_id"]:
                correct_predictions += 1
            batch_results.append(result)

    return batch_results, correct_predictions


def create_class_index(class_data_mean, class_data_divide, method):
    if method == "mean":
        print(
            {
                i: class_item["document"]["class_id"]
                for i, class_item in enumerate(class_data_mean)
            }
        )
        class_index = {
            i: class_item["document"]["class_id"]
            for i, class_item in enumerate(class_data_mean)
        }
        return class_index
    if method == "divide" or method == "vote":
        class_index = {}
        for i, class_item in enumerate(class_data_divide):
            class_id = class_item["document"]["class_id"]
            if class_id not in class_index:
                class_index[class_id] = []
            class_index[class_id].append(i)
        return class_index


def main():

    parser = argparse.ArgumentParser(
        description="Arguments for CLIP based classification."
    )
    # add needed arguments
    parser.add_argument("--test_data_file", type=str, help="Path to the test data file")
    parser.add_argument(
        "--clip_mean_embeddings", type=str, help="Choose clip_embeddings.[mean]"
    )
    parser.add_argument(
        "--clip_divide_embeddings", type=str, help="Choose clip_embeddings.[divide]"
    )
    parser.add_argument("--method", type=str, help="Evaluate through different methods")
    parser.add_argument(
        "--class_description_mean_file",
        type=str,
        help="Path to the class mean description file",
    )
    parser.add_argument(
        "--class_description_divide_file",
        type=str,
        help="Path to the class divide description file",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--batch_size", type=int, help="Batch Size")
    parser.add_argument("--model", type=str, default="idefics")
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--others", type=str, help="something to add.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")

    # Initialize clip
    tokenizer, model = initialize_clip_model(device)

    # load data
    with open(args.class_description_mean_file, "r") as file:
        class_data_mean = json.load(file)
    with open(args.class_description_divide_file, "r") as file:
        class_data_divide = json.load(file)
    if args.method != "combine":
        class_index = create_class_index(
            class_data_mean, class_data_divide, args.method
        )
    else:
        average_json_mapping = create_mapping_from_average_json(
            args.class_description_mean_file
        )
        divide_json_mapping = create_mapping_from_divide_json(
            args.class_description_divide_file
        )
        desc_to_index_map = create_description_to_embedding_index_map(class_data_divide)
    with open(args.test_data_file, "r") as file:
        test_data = json.load(file)

    # 加载 clip_embeddings
    with open(args.clip_mean_embeddings, "rb") as f:
        average_embeddings = pickle.load(f)

    with open(args.clip_divide_embeddings, "rb") as f:
        divide_embeddings = pickle.load(f)
    print(average_embeddings.shape, divide_embeddings.shape)
    # # 将每个嵌入移动到指定的设备
    average_embeddings = [emb.to(device) for emb in average_embeddings]
    divide_embeddings = [emb.to(device) for emb in divide_embeddings]
    total_correct = 0
    total_samples = 0
    results = []

    if args.method in ["mean", "divide"]:
        batch_inference_times = []
        for i in tqdm(range(0, len(test_data), args.batch_size)):
            batch = test_data[i : i + args.batch_size]
            total_samples += len(batch)
            start_time = time.time()
            if args.method == "mean":
                batch_results, correct_preds = process_average_batch(
                    batch, average_embeddings, class_index, tokenizer, model, device
                )
                total_correct += correct_preds
                results.extend(batch_results)
            elif args.method == "divide":
                batch_results, correct_preds = process_divide_batch(
                    batch, divide_embeddings, class_index, tokenizer, model, device
                )
                total_correct += correct_preds
                results.extend(batch_results)
            end_time = time.time()
            batch_inference_time_ms = (end_time - start_time) * 1000
            batch_inference_times.append(batch_inference_time_ms)
        accuracy = total_correct / total_samples
        print(f"Accuracy: {accuracy}")
        average_inference_time_ms = sum(batch_inference_times) / len(
            batch_inference_times
        )
        print(f"Average Inference Time per Batch: {average_inference_time_ms} ms")
        output_data = {
            "accuracy": accuracy,
            "times": average_inference_time_ms,
            "results": results,
        }

        output_file = f"/home/hyh30/descriptionRAG/results/{args.dataset_name}/clip_embeddings/{args.model}_classification_results_{args.method}_clip_embddings_{args.others}.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Results and accuracy saved to {output_file}")
    elif args.method == "vote":
        final_output = []
        for topk in range(3, 11):
            for i in tqdm(range(0, len(test_data), args.batch_size)):
                batch = test_data[i : i + args.batch_size]
                total_samples += len(batch)
                batch_results, correct_preds = process_majority_voting_batch(
                    batch,
                    divide_embeddings,
                    class_index,
                    tokenizer,
                    model,
                    device,
                    topk,
                )
                total_correct += correct_preds
                results.extend(batch_results)
            accuracy = total_correct / total_samples
            print(f"Accuracy: {accuracy}")

            output_data = {"topk": topk, "accuracy": accuracy, "results": results}
            final_output.append(output_data)
        output_file = f"/home/hyh30/descriptionRAG/results/{args.dataset_name}/clip_embeddings/{args.model}_classification_results_{args.method}_clip_embddings_{args.others}.json"
        with open(output_file, "w") as f:
            json.dump(final_output, f, indent=4)
        print(f"Results and accuracy saved to {output_file}")
    elif args.method == "combine":
        output_data_examples = []
        for topk in range(90, 100):
            for topn in range(5, 6):
                for i in tqdm(range(0, len(test_data), args.batch_size)):
                    batch = test_data[i : i + args.batch_size]
                    total_samples += len(batch)
                    batch_results, correct_preds = process_combined_batch(
                        batch,
                        average_embeddings,
                        divide_embeddings,
                        average_json_mapping,
                        divide_json_mapping,
                        desc_to_index_map,
                        tokenizer,
                        model,
                        args.gpu,
                        topk,
                        topn,
                    )
                    total_correct += correct_preds
                    results.extend(batch_results)
                accuracy = total_correct / total_samples
                print(f"Accuracy for topk={topk}, topn={topn}: {accuracy}")

                output_data = {
                    "top_k": topk,
                    "top_n": topn,
                    "accuracy": accuracy,
                    "results": results,
                }
                output_data_examples.append(output_data)
        topk_values, topn_values, accuracies = [], [], []
        for data_point in output_data_examples:
            topk_values.append(data_point["top_k"])
            topn_values.append(data_point["top_n"])
            accuracies.append(data_point["accuracy"])

        # Convert lists to numpy arrays for easier handling
        topk_values = np.array(topk_values)
        topn_values = np.array(topn_values)
        accuracies = np.array(accuracies)

        # Plotting
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(topk_values, topn_values, c=accuracies, cmap="viridis")
        plt.colorbar(scatter, label="Accuracy")
        plt.xlabel("Top K")
        plt.ylabel("Top N")
        plt.title("Accuracy Improvement with Different Top K and Top N")

        # Highlight the point with the highest accuracy
        max_accuracy_idx = np.argmax(accuracies)
        max_accuracy = accuracies[max_accuracy_idx]
        max_topk = topk_values[max_accuracy_idx]
        max_topn = topn_values[max_accuracy_idx]
        plt.scatter(max_topk, max_topn, color="red")  # Mark the point
        plt.text(
            max_topk, max_topn, f"{max_accuracy:.2f}", color="red", fontsize=12
        )  # Show the accuracy
        plt.grid(True)

        # Save the plot as an image file
        plt.savefig(f"accuracy_plot{args.dataset_name}_{args.model}.png")
        output_file = f"/home/hyh30/descriptionRAG/results/{args.dataset_name}/clip_embeddings/{args.model}_classification_results_{args.method}_clip_embddings_classes_descriptions.json"
        with open(output_file, "w") as f:
            json.dump(output_data_examples, f, indent=4)
        print(f"Results and accuracy saved to {output_file}")


if __name__ == "__main__":
    main()
