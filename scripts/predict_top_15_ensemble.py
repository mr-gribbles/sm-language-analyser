"""Top 15 Ensemble Prediction Script.

This script loads the top 15 models based on accuracy and performs an
ensemble prediction by averaging their confidence scores.
"""

import argparse
import os
import sys

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import from predict_text
try:
    from scripts.predict_text import ModelPredictor
except ImportError as e:
    print(f"Could not import from predict_text: {e}")
    sys.exit(1)

# Top 15 models based on combined metrics from plots/detailed_model_metrics.csv
TOP_15_MODELS = [
    "svm_rbf",
    "calibrated_svm",
    "stacking_lr",
    "svm_sigmoid",
    "mlp_small",
    "nu_svc",
    "neural_network_default",
    "calibrated_sgd",
    "neural_network_dropout_low",
    "neural_network_leaky",
    "stacking_rf",
    "neural_network_dropout_high",
    "neural_network_best_ensemble",
    "neural_network_ensemble_optimized",
    "mlp_relu",
]


class Top15EnsemblePredictor(ModelPredictor):
    """Predictor for the top 15 ensemble."""

    def __init__(self, models_dir: str = "models"):
        """Initialize the predictor."""
        super().__init__(models_dir)
        self.top_models = {}

    def interactive_mode(self):
        """Interactive prediction mode for the top 15 ensemble."""
        print("\n--- Top 15 Ensemble Interactive Mode ---")
        self.load_top_models()

        while True:
            try:
                text = input("\nEnter text (or 'quit' to exit): ").strip()
                if text.lower() == "quit":
                    break
                if not text:
                    continue

                prediction, confidence, individual_scores = self.predict(
                    text, verbose=True
                )
                label = "AI-Generated" if prediction == 1 else "Human-Written"

                print(f"\nPrediction: {label}")
                print(f"Mean Confidence: {confidence:.4f}")

            except KeyboardInterrupt:
                print("\nExiting interactive mode.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

    def load_top_models(self):
        """Load the top 15 models."""
        print("Loading top 15 models...")
        for model_name in TOP_15_MODELS:
            try:
                self.top_models[model_name] = self.load_model(model_name)
            except Exception as e:
                print(f"Warning: Could not load model {model_name}: {e}")

        if not self.top_models:
            raise RuntimeError("No models could be loaded for the ensemble.")

        print(
            f"Successfully loaded {len(self.top_models)} out of {len(TOP_15_MODELS)} models."
        )

    def predict(self, text: str, verbose: bool = False):
        """Perform ensemble prediction."""
        if not self.top_models:
            self.load_top_models()

        predictions = []
        for model_name, model_data in self.top_models.items():
            try:
                _, probability = self.predict_single_model(model_data, text, model_name)
                predictions.append(probability)
            except Exception as e:
                print(f"Warning: Prediction failed for model {model_name}: {e}")

        if not predictions:
            raise RuntimeError("All models failed to make a prediction.")

        # Separate predictions by class
        ai_probs = [p for p in predictions if p > 0.5]
        human_probs = [1 - p for p in predictions if p <= 0.5]

        # Voting-based prediction
        if len(ai_probs) > len(human_probs):
            final_prediction = 1
            mean_confidence = np.mean(ai_probs) if ai_probs else 0.5
        else:
            final_prediction = 0
            mean_confidence = np.mean(human_probs) if human_probs else 0.5

        if verbose:
            return final_prediction, mean_confidence, predictions

        return final_prediction, mean_confidence


def main():
    """Run the main function."""
    parser = argparse.ArgumentParser(description="Top 15 Ensemble Prediction")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    args = parser.parse_args()

    predictor = Top15EnsemblePredictor()

    if args.interactive:
        predictor.interactive_mode()
    elif args.text:
        try:
            prediction, confidence = predictor.predict(args.text)
            label = "AI-Generated" if prediction == 1 else "Human-Written"

            print("\n--- Top 15 Ensemble Prediction ---")
            print(f"Text: {args.text[:100]}...")
            print(f"Prediction: {label}")
            print(f"Mean Confidence: {confidence:.4f}")

        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print(
            "Please provide a text to classify with --text or use --interactive mode."
        )


if __name__ == "__main__":
    main()
