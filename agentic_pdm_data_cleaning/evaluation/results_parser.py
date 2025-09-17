import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def load_evaluation_results(path: Path) -> Dict[str, Any]:
    """
    Load evaluation results from a JSON file.
    Converts lists back to NumPy arrays where appropriate.

    Args:
        path: Path to the evaluation results JSON file.

    Returns:
        A dictionary containing the evaluation data.
    """
    with open(path, 'r') as f:
        data = json.load(f)

    # Convert matrices back to NumPy arrays for analysis
    if "classification_confusion_matrix" in data:
        data["classification_confusion_matrix"] = np.array(
            data["classification_confusion_matrix"])
    if "total_tokens_confusion_matrix" in data:
        data["total_tokens_confusion_matrix"] = np.array(
            data["total_tokens_confusion_matrix"])
    if "request_tokens_confusion_matrix" in data:
        data["request_tokens_confusion_matrix"] = np.array(
            data["request_tokens_confusion_matrix"])
    if "response_tokens_confusion_matrix" in data:
        data["response_tokens_confusion_matrix"] = np.array(
            data["response_tokens_confusion_matrix"])

    return data


def summarize_evaluation(data: Dict[str, Any]) -> None:
    """
    Print a summary of the evaluation results.

    Args:
        data: The evaluation results dictionary.
    """
    print(f"Model: {data['model_name']}")
    print(f"Benchmark: {data['benchmark_name']}")
    print(f"Experiment: {data['experiment_name']} ({data['experiment_type']})")
    print(f"Timestamp: {data['timestamp']}")
    print(f"Total Tokens: {data['total_tokens']}")
    print(f"Total Request Tokens: {data['total_request_tokens']}")
    print(f"Total Response Tokens: {data['total_response_tokens']}")
    print("Classification Confusion Matrix:")
    print(data['classification_confusion_matrix'])
    print("Total Tokens Confusion Matrix:")
    print(data['total_tokens_confusion_matrix'])
    print("Request Tokens Confusion Matrix:")
    print(data['request_tokens_confusion_matrix'])
    print("Response Tokens Confusion Matrix:")
    print(data['response_tokens_confusion_matrix'])
    print(f"Number of diffs found: {len(data.get('diffs', {}))}")
    print(f"Repair Coverage: {data.get('repair_coverage', 0):.2%}")
    print(f"Average Tokens per Task: {data.get('avg_tokens_per_task', 0):.2f}")


if __name__ == "__main__":
    # Example usage
    eval_path = Path(
        "results/evaluations/benchmark_mixed_experiments/exp1_qwen3:8b_evaluation.json")
    evaluation_data = load_evaluation_results(eval_path)
    summarize_evaluation(evaluation_data)
