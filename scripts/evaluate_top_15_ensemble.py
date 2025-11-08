"""Evaluate the Top 15 Ensemble Model.

This script evaluates the performance of the top 15 ensemble model using
the same metrics as the individual model evaluator.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from scripts.evaluate_models import NewModelEvaluator
    from scripts.predict_top_15_ensemble import Top15EnsemblePredictor
except ImportError as e:
    print(f"Could not import required classes: {e}")
    sys.exit(1)


def load_test_data(human_file: str, ai_file: str) -> tuple[list[str], np.ndarray]:
    """Load test data from JSONL files, handling multiple formats."""
    texts, labels = [], []

    # Load human texts
    if os.path.exists(human_file):
        with open(human_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = None
                    if "original_content" in data and isinstance(
                        data["original_content"], dict
                    ):
                        if "cleaned_text" in data["original_content"]:
                            text = data["original_content"]["cleaned_text"].strip()
                        elif "cleaned_selftext" in data["original_content"]:
                            text = data["original_content"]["cleaned_selftext"].strip()
                    elif "cleaned_selftext" in data:
                        text = data["cleaned_selftext"].strip()

                    if text and len(text) > 10:
                        texts.append(text)
                        labels.append(0)
                except (json.JSONDecodeError, KeyError):
                    continue

    # Load AI texts
    if os.path.exists(ai_file):
        with open(ai_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = None
                    if "llm_transformation" in data and isinstance(
                        data["llm_transformation"], dict
                    ):
                        if "rewritten_text" in data["llm_transformation"]:
                            text = data["llm_transformation"]["rewritten_text"].strip()
                    elif "original_content" in data and isinstance(
                        data["original_content"], dict
                    ):
                        for field in ["cleaned_text", "content", "raw_text"]:
                            if field in data["original_content"]:
                                text = data["original_content"][field].strip()
                                break
                    elif "rewritten_text" in data:
                        text = data["rewritten_text"].strip()

                    if text and len(text) > 10:
                        texts.append(text)
                        labels.append(1)
                except (json.JSONDecodeError, KeyError):
                    continue

    return texts, np.array(labels)


def main():
    """Run the main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate the Top 15 Ensemble Model")
    parser.add_argument(
        "--human-file",
        type=str,
        default="test_data_human.jsonl",
        help="Human test data file",
    )
    parser.add_argument(
        "--ai-file", type=str, default="test_data_ai.jsonl", help="AI test data file"
    )
    args = parser.parse_args()

    print("--- Evaluating Top 15 Ensemble Model ---")

    # Load test data
    try:
        texts, y_true = load_test_data(args.human_file, args.ai_file)
        print(f"Loaded {len(texts)} test samples.")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # Initialize predictor and load models
    predictor = Top15EnsemblePredictor()
    predictor.load_top_models()

    # Get predictions
    y_pred, y_prob = [], []
    for text in texts:
        pred, prob = predictor.predict(text)
        y_pred.append(pred)
        y_prob.append(prob)

    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Calculate metrics
    evaluator = NewModelEvaluator()
    metrics = evaluator.calculate_advanced_metrics(y_true, y_pred, y_prob)

    print("\n--- Performance Metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=["Human", "AI"]))

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Generate and save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Human", "AI"],
        yticklabels=["Human", "AI"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Top 15 Ensemble Confusion Matrix")
    plot_path = "plots/top_15_ensemble_confusion_matrix.png"
    plt.savefig(plot_path)
    print(f"\nConfusion matrix plot saved to {plot_path}")


if __name__ == "__main__":
    main()
