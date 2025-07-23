#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Majority Vote Script for ALQAC 2025
Converted from majority_vote.ipynb
This script combines results from multiple models using a majority voting strategy.
"""

import json
import argparse
import os
from collections import Counter
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("majority_vote.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Default paths
DATA_DIR = Path("../ALQAC_2025_data")
OUTPUT_DIR = Path("../evaluation_task2")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def read_json_file(path):
    """
    Read and load JSON file

    Args:
        path: Path to the JSON file

    Returns:
        Loaded JSON data as Python object
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {path}: {e}")
        return None


def combine_results(result_list, best_model_index, json_data):
    """
    Combine results from multiple models using majority voting strategy
    For essay questions, use the prediction from the best model
    For other questions, use majority vote

    Args:
        result_list: List of result lists from different models
        best_model_index: Index of the best model to use for essay questions
        json_data: Original data with question types

    Returns:
        List of combined results
    """
    final_results = []

    # Check if all lists have the same length
    lengths = [len(results) for results in result_list]
    if len(set(lengths)) != 1:
        logger.error("Error: Result lists have different lengths.")
        return []

    for i in range(lengths[0]):
        question_type = json_data[i]["question_type"]
        final_predict = None

        if question_type == "Tự luận":
            # For essay questions, take prediction from best model
            final_predict = result_list[best_model_index][i]["predict"]
        else:
            # For other types, get majority vote from all models
            predicts = [results[i]["predict"] for results in result_list]
            counts = Counter(predicts)
            final_predict = counts.most_common(1)[0][0]

        result_item = {
            "question_id": json_data[i]["question_id"],
            "answer": final_predict,
            "question_type": question_type,
        }
        final_results.append(result_item)

    return final_results


def acc_eval_from_list(prediction_list, gold_list):
    """
    Evaluate accuracy from prediction and gold answer lists.

    Args:
        prediction_list: List of prediction dictionaries
        gold_list: List of gold answer dictionaries

    Returns:
        Accuracy score as a float
    """
    gold_dict = {item["question_id"]: item["answer"] for item in gold_list}
    pred_dict = {item["question_id"]: item["answer"] for item in prediction_list}

    count_true = 0
    total = len(gold_dict)

    for qid, gold_ans in gold_dict.items():
        pred_ans = pred_dict.get(qid)
        if pred_ans is not None and pred_ans.lower() == gold_ans.lower():
            count_true += 1

    acc = count_true / total if total else 0
    logger.info(f"Accuracy = {acc}")

    return acc


def main(args):
    """
    Main execution function

    Args:
        args: Command line arguments
    """
    # Load ground truth data
    logger.info(f"Loading ground truth data from {args.data_path}")
    json_data = read_json_file(args.data_path)
    if not json_data:
        logger.error("Failed to load ground truth data. Exiting.")
        return

    # Load model results
    model_results = []
    model_names = []

    for model_path in args.model_paths:
        logger.info(f"Loading results from {model_path}")
        model_name = os.path.basename(model_path).replace(".json", "")
        model_names.append(model_name)

        model_data = read_json_file(model_path)
        if model_data and "results" in model_data:
            model_results.append(model_data["results"])
        else:
            logger.error(f"Invalid or missing results in {model_path}")
            return

    # Validate data
    if not model_results:
        logger.error("No model results loaded. Exiting.")
        return

    # Combine results
    logger.info(
        f"Combining results using majority voting (best model index: {args.best_model_index})"
    )
    final_results = combine_results(model_results, args.best_model_index, json_data)

    # Evaluate results
    logger.info("Evaluating combined results")
    acc = acc_eval_from_list(final_results, json_data)

    eval_metrics = {
        "accuracy": acc,
        "total_questions": len(json_data),
        "evaluated_questions": len(final_results),
        "models_used": model_names,
        "best_model_for_essays": model_names[args.best_model_index]
        if args.best_model_index < len(model_names)
        else "unknown",
    }

    # Save results
    output_path = args.output_path
    if not output_path:
        output_path = OUTPUT_DIR / "majority_vote_results.json"

    logger.info(f"Saving results to {output_path}")

    output_data = {"results": final_results, "metrics": eval_metrics}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Majority vote complete. Final accuracy: {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine results from multiple models using majority voting"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DATA_DIR / "alqac25_train.json"),
        help="Path to the ground truth data file",
    )

    parser.add_argument(
        "--model-paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to model result JSON files",
    )

    parser.add_argument(
        "--best-model-index",
        type=int,
        default=0,
        help="Index of the best model to use for essay questions",
    )

    parser.add_argument(
        "--output-path", type=str, help="Path to save the combined results"
    )

    args = parser.parse_args()
    main(args)
