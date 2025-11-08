"""Comprehensive Model Interpretability Analyzer for Text Classification.

This script provides detailed interpretability analysis for trained text classification models,
showing which features and linguistic patterns distinguish AI-generated from human text.
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from wordcloud import WordCloud

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import from training script
try:
    from train_models import FeatureExtractor

    print("Successfully imported FeatureExtractor from train_models")
except ImportError as e:
    print(f"Warning: Could not import from train_models: {e}")

    class FeatureExtractor:
        """Dummy FeatureExtractor for compatibility."""

        pass


warnings.filterwarnings("ignore")


class ModelInterpretabilityAnalyzer:
    """Comprehensive interpretability analysis for text classification models."""

    def __init__(
        self, models_dir: str = "models", output_dir: str = "interpretability_results"
    ):
        """Initialize the analyzer."""
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model storage
        self.interpretable_models = {}
        self.feature_extractor = None
        self.feature_names = []
        self.word_features = []
        self.char_features = []
        self.linguistic_features = []

        # Test data
        self.X_test = None
        self.y_test = None
        self.test_texts = None

    def discover_interpretable_models(self) -> Dict[str, Tuple[Any, str, Any]]:
        """Discover and load interpretable models."""
        print("Discovering interpretable models...")

        interpretable_types = {
            # Tree-based models (highly interpretable)
            "random_forest",
            "decision_tree",
            "extra_trees",
            "gradient_boosting",
            "adaboost",
            "hist_gradient_boosting",
            # Linear models (highly interpretable)
            "logistic_regression",
            "ridge",
            "linear_svc",
            "sgd",
            "perceptron",
            "passive_aggressive",
            "calibrated",
            # Distance-based (somewhat interpretable)
            "naive_bayes",
            "linear_discriminant",
            "nearest_centroid",
            # External libraries (if tree-based)
            "xgboost",
            "lightgbm",
            "catboost",
        }

        discovered_models = {}

        if not self.models_dir.exists():
            print(f"Models directory {self.models_dir} does not exist!")
            return discovered_models

        for model_file in self.models_dir.glob("*.pkl"):
            model_name = model_file.stem

            # Check if model type is interpretable
            if any(
                interpretable_type in model_name.lower()
                for interpretable_type in interpretable_types
            ):
                try:
                    model, model_type, feature_extractor = self.load_model_safely(
                        model_file
                    )
                    if model is not None:
                        discovered_models[model_name] = (
                            model,
                            model_type,
                            feature_extractor,
                        )
                        print(f"  ✓ Found interpretable model: {model_name}")
                except Exception as e:
                    print(f"  ✗ Failed to load {model_name}: {e}")

        print(f"Discovered {len(discovered_models)} interpretable models")
        return discovered_models

    def load_model_safely(self, model_path: Path) -> Tuple[Any, str, Any]:
        """Safely load a model file."""
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            if not isinstance(model_data, dict):
                return None, "unknown", None

            model_type = model_data.get("model_type", "sklearn")
            feature_extractor = model_data.get("feature_extractor", None)

            # Handle sklearn models (both old 'sklearn' and new specific types)
            sklearn_types = {
                "sklearn",
                "adaboost",
                "decision_tree",
                "random_forest",
                "gradient_boosting",
                "extra_trees",
                "svm",
                "logistic_regression",
                "ridge",
                "linear_svc",
                "sgd",
                "perceptron",
                "passive_aggressive",
                "naive_bayes",
                "discriminant_analysis",
                "nearest_centroid",
                "calibrated",
                "xgboost",
                "lightgbm",
                "catboost",
                "bagging",
                "knn",
                "mlp",
                "voting",
                "stacking",
            }

            if model_type in sklearn_types:
                model = model_data.get("model")
                return model, model_type, feature_extractor
            else:
                # Skip neural networks for interpretability
                return None, model_type, feature_extractor

        except Exception as e:
            print(f"Error loading {model_path.name}: {e}")
            return None, "unknown", None

    def load_test_data(
        self,
        human_file: str = "test_data_human.jsonl",
        ai_file: str = "test_data_ai.jsonl",
    ) -> bool:
        """Load test data for interpretability analysis."""
        try:
            print(f"Loading test data from {human_file} and {ai_file}")

            texts = []
            labels = []

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
                                    text = data["original_content"][
                                        "cleaned_text"
                                    ].strip()
                                elif "cleaned_selftext" in data["original_content"]:
                                    text = data["original_content"][
                                        "cleaned_selftext"
                                    ].strip()
                            elif "cleaned_selftext" in data:
                                text = data["cleaned_selftext"].strip()

                            if text and len(text) > 10:
                                texts.append(text)
                                labels.append(0)  # Human

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
                                    text = data["llm_transformation"][
                                        "rewritten_text"
                                    ].strip()
                            # Handle LLM-generated format in AI test file
                            elif "original_content" in data and isinstance(
                                data["original_content"], dict
                            ):
                                # Try different text fields from LLM-generated format
                                for field in ["cleaned_text", "content", "raw_text"]:
                                    if field in data["original_content"]:
                                        text = data["original_content"][field].strip()
                                        break
                            elif "rewritten_text" in data:
                                text = data["rewritten_text"].strip()

                            if text and len(text) > 10:
                                texts.append(text)
                                labels.append(1)  # AI

                        except (json.JSONDecodeError, KeyError):
                            continue

            if len(texts) == 0:
                print("No test data found, creating synthetic data")
                texts = ["Sample human text", "Sample AI generated text"]
                labels = [0, 1]

            print(f"Loaded {len(texts)} test samples")
            self.test_texts = texts
            self.y_test = np.array(labels)
            return True

        except Exception as e:
            print(f"Error loading test data: {e}")
            return False

    def extract_feature_names(self, feature_extractor: FeatureExtractor) -> List[str]:
        """Extract meaningful feature names from the feature extractor."""
        feature_names = []

        if feature_extractor is None:
            return []

        try:
            print("Extracting feature names from feature extractor...")
            print(
                "Feature extractor attributes: "
                f"{[attr for attr in dir(feature_extractor) if not attr.startswith('_')]}"
            )

            # Try different possible vectorizer attribute names
            word_vectorizer = None
            char_vectorizer = None

            # Check if there's an enhanced_extractor that contains the actual vectorizers
            enhanced_extractor = None
            if hasattr(feature_extractor, "enhanced_extractor"):
                enhanced_extractor = feature_extractor.enhanced_extractor
                print(
                    "Found enhanced_extractor: "
                    f"{[attr for attr in dir(enhanced_extractor) if not attr.startswith('_')]}"
                )

            # Check for vectorizers in the main extractor first, then in enhanced_extractor
            extractors_to_check = [feature_extractor]
            if enhanced_extractor is not None:
                extractors_to_check.append(enhanced_extractor)

            for extractor in extractors_to_check:
                # Check for various possible word vectorizer attribute names
                for attr_name in [
                    "word_vectorizer",
                    "word_tfidf",
                    "tfidf_word",
                    "vectorizer_word",
                    "word_vec",
                    "tfidf_vectorizer",
                ]:
                    if hasattr(extractor, attr_name):
                        word_vectorizer = getattr(extractor, attr_name)
                        if word_vectorizer is not None:
                            print(
                                f"Found word vectorizer: {attr_name} in {type(extractor).__name__}"
                            )
                            break
                if word_vectorizer is not None:
                    break

            for extractor in extractors_to_check:
                # Check for various possible char vectorizer attribute names
                for attr_name in [
                    "char_vectorizer",
                    "char_tfidf",
                    "tfidf_char",
                    "vectorizer_char",
                    "char_vec",
                    "char_ngram_vectorizer",
                ]:
                    if hasattr(extractor, attr_name):
                        char_vectorizer = getattr(extractor, attr_name)
                        if char_vectorizer is not None:
                            print(
                                f"Found char vectorizer: {attr_name} in {type(extractor).__name__}"
                            )
                            break
                if char_vectorizer is not None:
                    break

            # First check if vocabularies are stored directly in the enhanced extractor
            if enhanced_extractor and hasattr(enhanced_extractor, "_word_vocabulary"):
                try:
                    word_vocab_dict = enhanced_extractor._word_vocabulary
                    if word_vocab_dict:
                        # Sort by index to get ordered feature names
                        word_vocab = [""] * len(word_vocab_dict)
                        for word, idx in word_vocab_dict.items():
                            if idx < len(word_vocab):
                                word_vocab[idx] = word
                        word_vocab = [w for w in word_vocab if w]
                        word_features = [f"word_{word}" for word in word_vocab]
                        feature_names.extend(word_features)
                        self.word_features = word_features
                        print(
                            f"Extracted {len(word_features)} word features from stored vocabulary"
                        )
                except Exception as e:
                    print(f"Error accessing stored word vocabulary: {e}")

            elif word_vectorizer is not None:
                try:
                    # Try multiple methods to get vocabulary
                    word_vocab = None
                    if hasattr(word_vectorizer, "get_feature_names_out"):
                        try:
                            word_vocab = word_vectorizer.get_feature_names_out()
                        except Exception as e1:
                            print(f"get_feature_names_out failed: {e1}")

                    if word_vocab is None and hasattr(word_vectorizer, "vocabulary_"):
                        try:
                            # Get vocabulary from vocabulary_ attribute
                            vocab_dict = word_vectorizer.vocabulary_
                            if vocab_dict:
                                # Sort by index to get ordered feature names
                                word_vocab = [None] * len(vocab_dict)
                                for word, idx in vocab_dict.items():
                                    if idx < len(word_vocab):
                                        word_vocab[idx] = word
                                # Remove any None values
                                word_vocab = [w for w in word_vocab if w is not None]
                        except Exception as e2:
                            print(f"vocabulary_ access failed: {e2}")

                    if word_vocab is not None and len(word_vocab) > 0:
                        word_features = [f"word_{word}" for word in word_vocab]
                        feature_names.extend(word_features)
                        self.word_features = word_features
                        print(f"Extracted {len(word_features)} word features")
                    else:
                        print("Could not extract word vocabulary")

                except Exception as e:
                    print(f"Warning: Could not get word features: {e}")
            else:
                print("No word vectorizer found")

            # First check if char vocabularies are stored directly in the enhanced extractor
            if enhanced_extractor and hasattr(enhanced_extractor, "_char_vocabulary"):
                try:
                    char_vocab_dict = enhanced_extractor._char_vocabulary
                    if char_vocab_dict:
                        # Sort by index to get ordered feature names
                        char_vocab = [""] * len(char_vocab_dict)
                        for char, idx in char_vocab_dict.items():
                            if idx < len(char_vocab):
                                char_vocab[idx] = char
                        char_vocab = [c for c in char_vocab if c]
                        char_features = [f"char_{char}" for char in char_vocab]
                        feature_names.extend(char_features)
                        self.char_features = char_features
                        print(
                            f"Extracted {len(char_features)} character features from stored vocabulary"
                        )
                except Exception as e:
                    print(f"Error accessing stored char vocabulary: {e}")

            elif char_vectorizer is not None:
                try:
                    # Try multiple methods to get vocabulary
                    char_vocab = None
                    if hasattr(char_vectorizer, "get_feature_names_out"):
                        try:
                            char_vocab = char_vectorizer.get_feature_names_out()
                        except Exception as e1:
                            print(f"get_feature_names_out failed: {e1}")

                    if char_vocab is None and hasattr(char_vectorizer, "vocabulary_"):
                        try:
                            # Get vocabulary from vocabulary_ attribute
                            vocab_dict = char_vectorizer.vocabulary_
                            if vocab_dict:
                                # Sort by index to get ordered feature names
                                char_vocab = [None] * len(vocab_dict)
                                for char, idx in vocab_dict.items():
                                    if idx < len(char_vocab):
                                        char_vocab[idx] = char
                                # Remove any None values
                                char_vocab = [c for c in char_vocab if c is not None]
                        except Exception as e2:
                            print(f"vocabulary_ access failed: {e2}")

                    if char_vocab is not None and len(char_vocab) > 0:
                        char_features = [f"char_{char}" for char in char_vocab]
                        feature_names.extend(char_features)
                        self.char_features = char_features
                        print(f"Extracted {len(char_features)} character features")
                    else:
                        print("Could not extract char vocabulary")

                except Exception as e:
                    print(f"Warning: Could not get char features: {e}")
            else:
                print("No char vectorizer found")

            # Get actual linguistic feature names from the enhanced extractor
            linguistic_names = []
            if enhanced_extractor and hasattr(
                enhanced_extractor, "linguistic_extractor"
            ):
                try:
                    ling_extractor = enhanced_extractor.linguistic_extractor
                    if hasattr(ling_extractor, "get_feature_names"):
                        linguistic_names = ling_extractor.get_feature_names()
                        print(
                            f"Got {len(linguistic_names)} linguistic features from enhanced extractor"
                        )
                    else:
                        print(
                            "Linguistic extractor found but no get_feature_names method"
                        )
                except Exception as e:
                    print(f"Error getting linguistic features: {e}")

            # Fallback to hardcoded list if we couldn't get them from extractor
            if not linguistic_names:
                linguistic_names = [
                    "text_length",
                    "word_count",
                    "word_density",
                    "sentence_count",
                    "avg_words_per_sentence",
                    "upper_ratio",
                    "lower_ratio",
                    "digit_ratio",
                    "punct_ratio",
                    "lexical_diversity",
                    "avg_word_length",
                    "flesch_readability",
                ]
                print(
                    f"Using fallback linguistic features list: {len(linguistic_names)}"
                )

            self.linguistic_features = linguistic_names
            feature_names.extend(linguistic_names)
            print(f"Added {len(linguistic_names)} linguistic features")

            print(f"Total feature names extracted: {len(feature_names)}")

            # If we have a mismatch between expected features and actual feature names,
            # create placeholder names for the missing features
            if hasattr(self, "X_test") and self.X_test is not None:
                expected_features = self.X_test.shape[1]
            else:
                # Try to get expected feature count from a test transform
                if self.test_texts and len(self.test_texts) > 0:
                    try:
                        test_features = feature_extractor.transform(self.test_texts[:1])
                        expected_features = test_features.shape[1]
                    except Exception:
                        expected_features = len(feature_names)
                else:
                    expected_features = len(feature_names)

            if len(feature_names) < expected_features:
                missing_count = expected_features - len(feature_names)
                print(
                    f"Creating {missing_count} placeholder feature names for missing TF-IDF features"
                )

                # Add placeholder names for missing features
                # Assume most missing features are word features (since that's typically the largest)
                word_placeholders = [
                    f"word_feature_{i}" for i in range(missing_count - 100)
                ]  # Reserve 100 for char features
                char_placeholders = [
                    f"char_feature_{i}" for i in range(100)
                ]  # Assume 100 char features

                feature_names.extend(word_placeholders)
                feature_names.extend(char_placeholders)

                # Update the word and char features lists
                if not self.word_features:  # If we couldn't get real word features
                    self.word_features = word_placeholders
                if not self.char_features:  # If we couldn't get real char features
                    self.char_features = char_placeholders

                print(f"Total feature names after placeholders: {len(feature_names)}")

        except Exception as e:
            print(f"Warning: Could not extract feature names: {e}")

        return feature_names

    def analyze_linear_model(self, model, model_name: str) -> Dict[str, Any]:
        """Analyze linear model coefficients for interpretability."""
        results = {
            "model_name": model_name,
            "model_type": "linear",
            "interpretability": {},
        }

        try:
            # Get coefficients
            if hasattr(model, "coef_"):
                coefficients = model.coef_
                if coefficients.ndim > 1:
                    coefficients = coefficients[0]  # Binary classification

                # Create feature importance dataframe
                importance_data = []
                for i, coef in enumerate(coefficients):
                    if i < len(self.feature_names):
                        importance_data.append(
                            {
                                "feature": self.feature_names[i],
                                "coefficient": coef,
                                "abs_coefficient": abs(coef),
                                "feature_type": self.get_feature_type(
                                    self.feature_names[i]
                                ),
                            }
                        )

                importance_df = pd.DataFrame(importance_data)
                importance_df = importance_df.sort_values(
                    "abs_coefficient", ascending=False
                )

                results["interpretability"] = {
                    "top_positive_features": importance_df[
                        importance_df["coefficient"] > 0
                    ]
                    .head(20)
                    .to_dict("records"),
                    "top_negative_features": importance_df[
                        importance_df["coefficient"] < 0
                    ]
                    .head(20)
                    .to_dict("records"),
                    "top_word_features": importance_df[
                        importance_df["feature_type"] == "word"
                    ]
                    .head(20)
                    .to_dict("records"),
                    "top_linguistic_features": importance_df[
                        importance_df["feature_type"] == "linguistic"
                    ].to_dict("records"),
                    "coefficient_stats": {
                        "mean_abs_coef": importance_df["abs_coefficient"].mean(),
                        "max_abs_coef": importance_df["abs_coefficient"].max(),
                        "min_abs_coef": importance_df["abs_coefficient"].min(),
                        "num_positive": len(
                            importance_df[importance_df["coefficient"] > 0]
                        ),
                        "num_negative": len(
                            importance_df[importance_df["coefficient"] < 0]
                        ),
                    },
                }

                print(f"  ✓ Analyzed {model_name}: {len(coefficients)} features")

        except Exception as e:
            print(f"  ✗ Failed to analyze {model_name}: {e}")

        return results

    def analyze_tree_model(self, model, model_name: str) -> Dict[str, Any]:
        """Analyze tree-based model for interpretability."""
        results = {
            "model_name": model_name,
            "model_type": "tree",
            "interpretability": {},
        }

        try:
            # Get feature importances
            if (
                hasattr(model, "feature_importances_")
                and model.feature_importances_ is not None
            ):
                importances = model.feature_importances_

                # Check if importances is valid
                if len(importances) == 0:
                    print(f"  ⚠ {model_name}: Empty feature importances")
                    results["interpretability"] = self._get_empty_tree_results()
                    return results

                # Check if we have feature names
                if len(self.feature_names) == 0:
                    print(f"  ⚠ {model_name}: No feature names available")
                    results["interpretability"] = self._get_empty_tree_results()
                    return results

                # Create feature importance dataframe with extra safety
                importance_data = []
                for i, importance in enumerate(importances):
                    # Multiple safety checks - LOWERED THRESHOLD to include more features
                    if (
                        i < len(self.feature_names)
                        and importance is not None
                        and not np.isnan(importance)
                        and importance > 1e-8
                    ):  # Much lower threshold to include more features
                        try:
                            importance_data.append(
                                {
                                    "feature": str(self.feature_names[i]),
                                    "importance": float(importance),
                                    "feature_type": self.get_feature_type(
                                        self.feature_names[i]
                                    ),
                                }
                            )
                        except Exception:
                            # Skip problematic items
                            continue

                # Handle empty dataframe case
                if not importance_data:
                    print(f"  ⚠ {model_name}: No valid features with importance > 0")
                    results["interpretability"] = self._get_empty_tree_results()
                else:
                    try:
                        # Create DataFrame with explicit error handling
                        importance_df = pd.DataFrame(importance_data)

                        # Verify DataFrame has required columns
                        if "importance" not in importance_df.columns:
                            print(
                                f"  ⚠ {model_name}: DataFrame missing importance column"
                            )
                            results["interpretability"] = (
                                self._get_fallback_tree_results(importance_data)
                            )
                            return results

                        # Sort and filter safely
                        importance_df = importance_df.sort_values(
                            "importance", ascending=False
                        )

                        # Filter by type safely
                        word_features = importance_df[
                            importance_df["feature_type"] == "word"
                        ].head(20)
                        linguistic_features = importance_df[
                            importance_df["feature_type"] == "linguistic"
                        ]

                        # Calculate stats safely
                        importance_values = importance_df["importance"].values

                        results["interpretability"] = {
                            "top_features": importance_df.head(30).to_dict("records"),
                            "top_word_features": word_features.to_dict("records"),
                            "top_linguistic_features": linguistic_features.to_dict(
                                "records"
                            ),
                            "importance_stats": {
                                "mean_importance": float(np.mean(importance_values)),
                                "max_importance": float(np.max(importance_values)),
                                "num_nonzero_features": len(importance_df),
                            },
                        }

                    except Exception as df_error:
                        print(
                            f"  ⚠ {model_name}: DataFrame processing error: {df_error}"
                        )
                        results["interpretability"] = self._get_fallback_tree_results(
                            importance_data
                        )

                # Add tree structure info for decision trees
                if isinstance(model, DecisionTreeClassifier):
                    try:
                        if hasattr(model, "tree_") and model.tree_ is not None:
                            results["interpretability"]["tree_info"] = {
                                "n_nodes": int(model.tree_.node_count),
                                "n_leaves": int(model.tree_.n_leaves),
                                "max_depth": int(model.tree_.max_depth),
                            }
                    except Exception as tree_error:
                        print(
                            f"  ⚠ {model_name}: Could not extract tree info: {tree_error}"
                        )

                print(
                    f"  ✓ Analyzed {model_name}: {len(importances)} features, depth: {getattr(model, 'max_depth', 'N/A')}"
                )

            else:
                print(
                    f"  ⚠ {model_name}: No feature_importances_ attribute or it's None"
                )
                results["interpretability"] = self._get_empty_tree_results()

        except Exception as e:
            print(f"  ✗ Failed to analyze {model_name}: {str(e)}")
            results["interpretability"] = self._get_empty_tree_results()

        return results

    def analyze_ensemble_model(self, model, model_name: str) -> Dict[str, Any]:
        """Analyze ensemble model for interpretability."""
        results = {
            "model_name": model_name,
            "model_type": "ensemble",
            "interpretability": {},
        }

        try:
            # Get feature importances (most ensemble methods have this)
            if (
                hasattr(model, "feature_importances_")
                and model.feature_importances_ is not None
            ):
                importances = model.feature_importances_

                # Check if importances is valid
                if len(importances) == 0:
                    print(f"  ⚠ {model_name}: Empty feature importances")
                    results["interpretability"] = self._get_empty_ensemble_results(
                        model
                    )
                    return results

                # Check if we have feature names
                if len(self.feature_names) == 0:
                    print(f"  ⚠ {model_name}: No feature names available")
                    results["interpretability"] = self._get_empty_ensemble_results(
                        model
                    )
                    return results

                # Create feature importance dataframe with extra safety
                importance_data = []
                for i, importance in enumerate(importances):
                    # Multiple safety checks - LOWERED THRESHOLD to include more features
                    if (
                        i < len(self.feature_names)
                        and importance is not None
                        and not np.isnan(importance)
                        and importance > 1e-8
                    ):  # Much lower threshold to include more features
                        try:
                            importance_data.append(
                                {
                                    "feature": str(self.feature_names[i]),
                                    "importance": float(importance),
                                    "feature_type": self.get_feature_type(
                                        self.feature_names[i]
                                    ),
                                }
                            )
                        except Exception:
                            # Skip problematic items
                            continue

                # Handle empty dataframe case
                if not importance_data:
                    print(f"  ⚠ {model_name}: No valid features with importance > 0")
                    results["interpretability"] = self._get_empty_ensemble_results(
                        model
                    )
                else:
                    try:
                        # Create DataFrame with explicit error handling
                        importance_df = pd.DataFrame(importance_data)

                        # Verify DataFrame has required columns
                        if "importance" not in importance_df.columns:
                            print(
                                f"  ⚠ {model_name}: DataFrame missing importance column"
                            )
                            results["interpretability"] = (
                                self._get_fallback_ensemble_results(
                                    importance_data, model
                                )
                            )
                            return results

                        # Sort and filter safely
                        importance_df = importance_df.sort_values(
                            "importance", ascending=False
                        )

                        # Filter by type safely
                        word_features = importance_df[
                            importance_df["feature_type"] == "word"
                        ].head(20)
                        linguistic_features = importance_df[
                            importance_df["feature_type"] == "linguistic"
                        ]

                        # Calculate stats safely
                        importance_values = importance_df["importance"].values

                        results["interpretability"] = {
                            "top_features": importance_df.head(30).to_dict("records"),
                            "top_word_features": word_features.to_dict("records"),
                            "top_linguistic_features": linguistic_features.to_dict(
                                "records"
                            ),
                            "importance_stats": {
                                "mean_importance": float(np.mean(importance_values)),
                                "max_importance": float(np.max(importance_values)),
                                "num_nonzero_features": len(importance_df),
                            },
                            "ensemble_info": {
                                "n_estimators": getattr(model, "n_estimators", "N/A"),
                                "base_estimator": str(
                                    type(getattr(model, "base_estimator", model))
                                ),
                            },
                        }

                    except Exception as df_error:
                        print(
                            f"  ⚠ {model_name}: DataFrame processing error: {df_error}"
                        )
                        results["interpretability"] = (
                            self._get_fallback_ensemble_results(importance_data, model)
                        )

                print(
                    f"  ✓ Analyzed {model_name}: {getattr(model, 'n_estimators', 'N/A')} estimators"
                )

            else:
                print(
                    f"  ⚠ {model_name}: No feature_importances_ attribute or it's None"
                )
                results["interpretability"] = self._get_empty_ensemble_results(model)

        except Exception as e:
            print(f"  ✗ Failed to analyze {model_name}: {str(e)}")
            results["interpretability"] = self._get_empty_ensemble_results(model)

        return results

    def get_feature_type(self, feature_name: str) -> str:
        """Determine the type of feature."""
        if feature_name.startswith("word_"):
            return "word"
        elif feature_name.startswith("char_"):
            return "char"
        elif feature_name in self.linguistic_features:
            return "linguistic"
        else:
            return "unknown"

    def _get_empty_tree_results(self) -> Dict[str, Any]:
        """Get empty results structure for tree models."""
        return {
            "top_features": [],
            "top_word_features": [],
            "top_linguistic_features": [],
            "importance_stats": {
                "mean_importance": 0.0,
                "max_importance": 0.0,
                "num_nonzero_features": 0,
            },
        }

    def _get_fallback_tree_results(
        self, importance_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get fallback results structure for tree models when DataFrame fails."""
        try:
            word_features = [
                f for f in importance_data if f.get("feature_type") == "word"
            ][:20]
            linguistic_features = [
                f for f in importance_data if f.get("feature_type") == "linguistic"
            ]
            importance_values = [f.get("importance", 0) for f in importance_data]

            return {
                "top_features": importance_data[:30],
                "top_word_features": word_features,
                "top_linguistic_features": linguistic_features,
                "importance_stats": {
                    "mean_importance": (
                        float(np.mean(importance_values)) if importance_values else 0.0
                    ),
                    "max_importance": (
                        float(max(importance_values)) if importance_values else 0.0
                    ),
                    "num_nonzero_features": len(importance_data),
                },
            }
        except Exception:
            # Ultimate fallback
            return self._get_empty_tree_results()

    def _get_empty_ensemble_results(self, model) -> Dict[str, Any]:
        """Get empty results structure for ensemble models."""
        return {
            "top_features": [],
            "top_word_features": [],
            "top_linguistic_features": [],
            "importance_stats": {
                "mean_importance": 0.0,
                "max_importance": 0.0,
                "num_nonzero_features": 0,
            },
            "ensemble_info": {
                "n_estimators": getattr(model, "n_estimators", "N/A"),
                "base_estimator": str(type(getattr(model, "base_estimator", model))),
            },
        }

    def _get_fallback_ensemble_results(
        self, importance_data: List[Dict[str, Any]], model
    ) -> Dict[str, Any]:
        """Get fallback results structure for ensemble models when DataFrame fails."""
        try:
            word_features = [
                f for f in importance_data if f.get("feature_type") == "word"
            ][:20]
            linguistic_features = [
                f for f in importance_data if f.get("feature_type") == "linguistic"
            ]
            importance_values = [f.get("importance", 0) for f in importance_data]

            return {
                "top_features": importance_data[:30],
                "top_word_features": word_features,
                "top_linguistic_features": linguistic_features,
                "importance_stats": {
                    "mean_importance": (
                        float(np.mean(importance_values)) if importance_values else 0.0
                    ),
                    "max_importance": (
                        float(max(importance_values)) if importance_values else 0.0
                    ),
                    "num_nonzero_features": len(importance_data),
                },
                "ensemble_info": {
                    "n_estimators": getattr(model, "n_estimators", "N/A"),
                    "base_estimator": str(
                        type(getattr(model, "base_estimator", model))
                    ),
                },
            }
        except Exception:
            # Ultimate fallback
            return self._get_empty_ensemble_results(model)

    def create_feature_importance_plots(
        self, interpretability_results: List[Dict]
    ) -> None:
        """Create separate PNG files for each visualization type."""
        print("Creating individual interpretability visualizations...")

        # Filter out models without interpretability data
        valid_results = [r for r in interpretability_results if r["interpretability"]]

        if not valid_results:
            print("No interpretable models found for plotting")
            return

        # 1. Top word features across models - SEPARATE PNG
        print("  Creating word features plot...")
        plt.figure(figsize=(12, 8))
        self.plot_top_word_features(valid_results)
        plot_file = self.output_dir / "word_features_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"    Saved to {plot_file}")

        # 2. Model comparison heatmap - SEPARATE PNG
        print("  Creating model comparison plot...")
        plt.figure(figsize=(14, 8))
        self.plot_model_comparison_heatmap(valid_results)
        plot_file = self.output_dir / "model_comparison_heatmap.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"    Saved to {plot_file}")

        # 3. Feature type distribution - SEPARATE PNG
        print("  Creating feature distribution plot...")
        plt.figure(figsize=(8, 8))
        self.plot_feature_type_distribution(valid_results)
        plot_file = self.output_dir / "feature_type_distribution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"    Saved to {plot_file}")

        # 4. Coefficient comparison for linear models - SEPARATE PNG
        print("  Creating coefficient comparison plot...")
        plt.figure(figsize=(16, 10))
        self.plot_coefficient_comparison(valid_results)
        plot_file = self.output_dir / "coefficient_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"    Saved to {plot_file}")

        # 5. Top features word cloud - SEPARATE PNG
        print("  Creating word cloud...")
        plt.figure(figsize=(10, 8))
        self.create_importance_wordcloud(valid_results)
        plot_file = self.output_dir / "important_words_wordcloud.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"    Saved to {plot_file}")

        print("All individual plots created successfully!")

    def plot_top_word_features(self, results: List[Dict]) -> None:
        """Plot top word features across models with improved scaling and filtering."""
        word_importance_data = []

        for result in results:
            interp = result["interpretability"]
            model_name = result["model_name"]
            model_type = result["model_type"]

            # Get word features based on model type
            if "top_word_features" in interp:
                features = interp["top_word_features"][
                    :15
                ]  # Get more features to choose from
                for feature in features:
                    word = feature["feature"].replace("word_", "")

                    # Handle different scales for different model types with better scaling
                    if model_type == "linear":
                        raw_value = abs(
                            feature.get(
                                "coefficient", feature.get("abs_coefficient", 0)
                            )
                        )
                        # Improved scaling for better visibility
                        if raw_value > 0.001:  # Filter out very small coefficients
                            importance = min(
                                raw_value * 50, 1.0
                            )  # Better scaling factor
                        else:
                            importance = 0
                    else:
                        importance = feature.get("importance", 0)

                    # Only include features with meaningful values
                    if importance > 0.001:  # Filter threshold
                        word_importance_data.append(
                            {
                                "word": word,
                                "importance": importance,
                                "model": model_name,
                                "model_type": model_type,
                            }
                        )

        if word_importance_data:
            df = pd.DataFrame(word_importance_data)

            # Filter out words that appear in very few models or have low importance
            word_stats = df.groupby("word")["importance"].agg(["mean", "count"])
            significant_words = word_stats[
                (word_stats["mean"] > 0.01)  # High importance
                | (word_stats["count"] >= 2)  # Appears in multiple models
            ].index

            # Get top words for display
            top_words = word_stats.loc[significant_words]["mean"].nlargest(12).index
            df_filtered = df[df["word"].isin(top_words)]

            if not df_filtered.empty:
                # Create pivot for heatmap
                pivot_df = df_filtered.pivot_table(
                    index="word", columns="model", values="importance", fill_value=0
                )

                # Remove rows and columns that are mostly zeros
                pivot_df = pivot_df.loc[
                    (pivot_df > 0.001).any(axis=1)
                ]  # Remove zero rows
                pivot_df = pivot_df.loc[
                    :, (pivot_df > 0.001).any(axis=0)
                ]  # Remove zero columns

                if not pivot_df.empty:
                    # Normalize each column for fair comparison
                    pivot_df_norm = pivot_df.div(pivot_df.max(axis=0), axis=1).fillna(0)

                    # Create heatmap with better formatting
                    ax = sns.heatmap(
                        pivot_df_norm,
                        annot=False,
                        cmap="viridis",
                        cbar_kws={"label": "Normalized Importance"},
                        fmt=".3f",
                        square=False,
                    )

                    plt.title(
                        "Top Word Features Across Models",
                        fontsize=12,
                        fontweight="bold",
                    )
                    plt.ylabel("Words", fontsize=10)
                    plt.xlabel("Models", fontsize=10)
                    plt.xticks(rotation=45, ha="right", fontsize=8)
                    plt.yticks(rotation=0, fontsize=8)

                    return

        # Fallback if no data
        plt.text(
            0.5,
            0.5,
            "No significant word features found",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=12,
        )
        plt.title("Top Word Features (No Data)", fontsize=12)

    def plot_linguistic_features(self, results: List[Dict]) -> None:
        """Plot linguistic features comparison with improved scaling and filtering."""
        linguistic_data = []

        for result in results:
            interp = result["interpretability"]
            model_name = result["model_name"]
            model_type = result["model_type"]

            if "top_linguistic_features" in interp:
                features = interp["top_linguistic_features"]
                for feature in features:
                    # Handle different scales for different model types with better scaling
                    if model_type == "linear":
                        raw_value = abs(
                            feature.get(
                                "coefficient", feature.get("abs_coefficient", 0)
                            )
                        )
                        # Improved scaling for better visibility
                        if raw_value > 0.001:  # Filter out very small coefficients
                            importance = min(
                                raw_value * 50, 1.0
                            )  # Better scaling factor
                        else:
                            importance = 0
                    else:
                        importance = feature.get("importance", 0)

                    # Only include features with meaningful values
                    if importance > 0.001:  # Filter threshold
                        linguistic_data.append(
                            {
                                "feature": feature["feature"],
                                "importance": importance,
                                "model": model_name,
                                "model_type": model_type,
                            }
                        )

        if linguistic_data:
            df = pd.DataFrame(linguistic_data)

            # Filter out features that appear in very few models or have low importance
            feature_stats = df.groupby("feature")["importance"].agg(["mean", "count"])
            significant_features = feature_stats[
                (feature_stats["mean"] > 0.01)  # High importance
                | (feature_stats["count"] >= 2)  # Appears in multiple models
            ].index

            df_filtered = df[df["feature"].isin(significant_features)]

            if not df_filtered.empty:
                # Create pivot for heatmap
                pivot_df = df_filtered.pivot_table(
                    index="feature", columns="model", values="importance", fill_value=0
                )

                # Remove rows and columns that are mostly zeros
                pivot_df = pivot_df.loc[
                    (pivot_df > 0.001).any(axis=1)
                ]  # Remove zero rows
                pivot_df = pivot_df.loc[
                    :, (pivot_df > 0.001).any(axis=0)
                ]  # Remove zero columns

                if not pivot_df.empty:
                    # Normalize each column to prevent linear models from dominating
                    pivot_df_norm = pivot_df.div(pivot_df.max(axis=0), axis=1).fillna(0)

                    # Create heatmap with better formatting
                    ax = sns.heatmap(
                        pivot_df_norm,
                        annot=True,
                        fmt=".3f",
                        cmap="RdBu_r",
                        center=0.5,
                        cbar_kws={"label": "Normalized Importance"},
                        square=False,
                    )

                    plt.title(
                        "Linguistic Features Importance", fontsize=12, fontweight="bold"
                    )
                    plt.ylabel("Features", fontsize=10)
                    plt.xlabel("Models", fontsize=10)
                    plt.xticks(rotation=45, ha="right", fontsize=8)
                    plt.yticks(rotation=0, fontsize=8)

                    return

        # Fallback if no data
        plt.text(
            0.5,
            0.5,
            "No significant linguistic features found",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=12,
        )
        plt.title("Linguistic Features (No Data)", fontsize=12)

    def plot_model_comparison_heatmap(self, results: List[Dict]) -> None:
        """Create model comparison heatmap with proper scaling."""
        model_stats = []

        for result in results:
            stats = {}
            stats["model"] = result["model_name"]
            stats["model_type"] = result["model_type"]

            interp = result["interpretability"]

            if "importance_stats" in interp:
                stats.update(interp["importance_stats"])
            elif "coefficient_stats" in interp:
                coef_stats = interp["coefficient_stats"]
                # Scale coefficient stats to be comparable with importance stats
                stats["mean_importance"] = min(
                    coef_stats.get("mean_abs_coef", 0) * 10, 1.0
                )
                stats["max_importance"] = min(
                    coef_stats.get("max_abs_coef", 0) * 10, 1.0
                )
                stats["num_nonzero_features"] = coef_stats.get(
                    "num_positive", 0
                ) + coef_stats.get("num_negative", 0)

            model_stats.append(stats)

        if model_stats:
            df = pd.DataFrame(model_stats)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df_numeric = df[numeric_cols].fillna(0)

                # Normalize columns to 0-1 range for fair comparison
                df_normalized = df_numeric.div(df_numeric.max(axis=0), axis=1).fillna(0)

                sns.heatmap(
                    df_normalized.T,
                    annot=True,
                    fmt=".3f",
                    cmap="viridis",
                    xticklabels=df["model"],
                    cbar_kws={"label": "Normalized Value"},
                )
                plt.title("Model Statistics Comparison (Normalized)")
                plt.xlabel("Models")
                plt.xticks(rotation=45)
        else:
            plt.text(
                0.5,
                0.5,
                "No model statistics available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Model Comparison (No Data)")

    def plot_feature_type_distribution(self, results: List[Dict]) -> None:
        """Plot distribution of feature types."""
        feature_type_counts = {"word": 0, "char": 0, "linguistic": 0}

        for result in results:
            interp = result["interpretability"]

            # Count features by type
            if "top_features" in interp:
                for feature in interp["top_features"]:
                    feature_type = feature.get("feature_type", "unknown")
                    if feature_type in feature_type_counts:
                        feature_type_counts[feature_type] += 1

        if sum(feature_type_counts.values()) > 0:
            plt.pie(
                feature_type_counts.values(),
                labels=feature_type_counts.keys(),
                autopct="%1.1f%%",
                startangle=90,
            )
            plt.title("Distribution of Important Feature Types")
        else:
            plt.text(
                0.5,
                0.5,
                "No feature type data available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Feature Type Distribution (No Data)")

    def plot_coefficient_comparison(self, results: List[Dict]) -> None:
        """Plot coefficient comparison for linear models with improved filtering."""
        linear_results = [r for r in results if r["model_type"] == "linear"]

        if linear_results:
            coef_data = []

            for result in linear_results:
                model_name = result["model_name"]
                interp = result["interpretability"]

                # Get significant positive features only
                if "top_positive_features" in interp:
                    for feature in interp["top_positive_features"]:
                        coef_value = feature["coefficient"]
                        # Only include coefficients with meaningful magnitude
                        if abs(coef_value) > 0.01:  # Much stricter threshold
                            coef_data.append(
                                {
                                    "model": model_name,
                                    "feature": feature["feature"].replace("word_", ""),
                                    "coefficient": coef_value,
                                    "type": "positive",
                                }
                            )
                        # Stop after getting enough significant features
                        if (
                            len(
                                [
                                    d
                                    for d in coef_data
                                    if d["model"] == model_name
                                    and d["type"] == "positive"
                                ]
                            )
                            >= 8
                        ):
                            break

                # Get significant negative features only
                if "top_negative_features" in interp:
                    for feature in interp["top_negative_features"]:
                        coef_value = feature["coefficient"]
                        # Only include coefficients with meaningful magnitude
                        if abs(coef_value) > 0.01:  # Much stricter threshold
                            coef_data.append(
                                {
                                    "model": model_name,
                                    "feature": feature["feature"].replace("word_", ""),
                                    "coefficient": coef_value,
                                    "type": "negative",
                                }
                            )
                        # Stop after getting enough significant features
                        if (
                            len(
                                [
                                    d
                                    for d in coef_data
                                    if d["model"] == model_name
                                    and d["type"] == "negative"
                                ]
                            )
                            >= 8
                        ):
                            break

            if coef_data:
                df = pd.DataFrame(coef_data)

                # Filter out features that appear in very few models
                feature_counts = df["feature"].value_counts()
                significant_features = feature_counts[
                    feature_counts >= 2
                ].index  # Must appear in at least 2 models
                df_filtered = df[df["feature"].isin(significant_features)]

                if not df_filtered.empty:
                    # Take only top features by absolute coefficient value
                    df_top = df_filtered.nlargest(
                        40, "coefficient", keep="all"
                    ).nsmallest(40, "coefficient", keep="all")

                    # Create pivot for plotting
                    pivot_df = df_top.pivot_table(
                        index="feature",
                        columns="model",
                        values="coefficient",
                        fill_value=0,
                    )

                    # Remove any remaining near-zero rows
                    pivot_df = pivot_df.loc[abs(pivot_df).max(axis=1) > 0.01]

                    if not pivot_df.empty:
                        # Create improved plot
                        ax = pivot_df.plot(kind="bar", figsize=(14, 8), width=0.8)
                        plt.title(
                            "Feature Coefficients Across Linear Models (Filtered)",
                            fontsize=14,
                            fontweight="bold",
                        )
                        plt.xlabel("Features", fontsize=12)
                        plt.ylabel("Coefficient Value", fontsize=12)
                        plt.xticks(rotation=45, ha="right", fontsize=10)
                        plt.legend(
                            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10
                        )
                        plt.grid(axis="y", alpha=0.3)
                        plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)

                        return

                # Fallback if filtering removes everything
                plt.text(
                    0.5,
                    0.5,
                    "No significant coefficients found\n(threshold: |coef| > 0.01)",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                    fontsize=12,
                )
                plt.title("Coefficient Comparison (All Values Too Small)", fontsize=12)

            else:
                plt.text(
                    0.5,
                    0.5,
                    "No coefficient data available",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                    fontsize=12,
                )
                plt.title("Coefficient Comparison (No Data)", fontsize=12)
        else:
            plt.text(
                0.5,
                0.5,
                "No linear models found",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
                fontsize=12,
            )
            plt.title("Linear Model Coefficients (No Models)", fontsize=12)

    def create_importance_wordcloud(self, results: List[Dict]) -> None:
        """Create word cloud of most important features."""
        try:
            word_importances = {}

            for result in results:
                interp = result["interpretability"]

                if "top_word_features" in interp:
                    for feature in interp["top_word_features"]:
                        word = feature["feature"].replace("word_", "")
                        importance = feature.get(
                            "importance", feature.get("abs_coefficient", 0)
                        )

                        if word in word_importances:
                            word_importances[word] += importance
                        else:
                            word_importances[word] = importance

            if word_importances:
                # Create word cloud
                wordcloud = WordCloud(
                    width=400,
                    height=300,
                    background_color="white",
                    max_words=50,
                    colormap="viridis",
                ).generate_from_frequencies(word_importances)

                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.title("Most Important Words (Word Cloud)")
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No word importance data available",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                )
                plt.title("Word Cloud (No Data)")
                plt.axis("off")

        except ImportError:
            plt.text(
                0.5,
                0.5,
                "WordCloud not available\npip install wordcloud",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Word Cloud (WordCloud not installed)")
            plt.axis("off")
        except Exception as e:
            plt.text(
                0.5,
                0.5,
                f"Error creating word cloud:\n{str(e)}",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Word Cloud (Error)")
            plt.axis("off")

    def generate_interpretability_report(
        self, interpretability_results: List[Dict]
    ) -> None:
        """Generate comprehensive interpretability report."""
        print("Generating interpretability report...")

        report_path = self.output_dir / "model_interpretability_report.md"

        with open(report_path, "w") as f:
            f.write("# Model Interpretability Analysis Report\n\n")
            f.write(
                "This report provides detailed interpretability analysis for text classification models,\n"
            )
            f.write(
                "showing which linguistic features and patterns distinguish AI-generated from human text.\n\n"
            )

            # Executive summary
            f.write("## Executive Summary\n\n")
            valid_results = [
                r for r in interpretability_results if r["interpretability"]
            ]
            f.write(f"- **Total Models Analyzed**: {len(valid_results)}\n")

            model_types = {}
            for result in valid_results:
                model_type = result["model_type"]
                model_types[model_type] = model_types.get(model_type, 0) + 1

            f.write(
                f"- **Model Types**: {', '.join([f'{k} ({v})' for k, v in model_types.items()])}\n"
            )
            f.write(f"- **Total Features Analyzed**: {len(self.feature_names)}\n\n")

            # Top insights across all models
            f.write("## Key Interpretability Insights\n\n")

            # Aggregate most important word features
            all_word_features = {}
            all_linguistic_features = {}

            for result in valid_results:
                interp = result["interpretability"]

                if "top_word_features" in interp:
                    for feature in interp["top_word_features"][:10]:
                        word = feature["feature"].replace("word_", "")
                        importance = feature.get(
                            "importance", feature.get("abs_coefficient", 0)
                        )
                        if word in all_word_features:
                            all_word_features[word] += importance
                        else:
                            all_word_features[word] = importance

                if "top_linguistic_features" in interp:
                    for feature in interp["top_linguistic_features"]:
                        feature_name = feature["feature"]
                        importance = feature.get(
                            "importance", feature.get("abs_coefficient", 0)
                        )
                        if feature_name in all_linguistic_features:
                            all_linguistic_features[feature_name] += importance
                        else:
                            all_linguistic_features[feature_name] = importance

            # Top words that distinguish AI vs Human
            f.write("### Most Discriminative Words\n\n")
            if all_word_features:
                sorted_words = sorted(
                    all_word_features.items(), key=lambda x: x[1], reverse=True
                )[:20]
                f.write("| Rank | Word | Aggregate Importance |\n")
                f.write("|------|------|---------------------|\n")
                for i, (word, importance) in enumerate(sorted_words, 1):
                    f.write(f"| {i:2d} | {word:<20} | {importance:.4f} |\n")

            # Top linguistic features
            f.write("\n### Most Important Linguistic Features\n\n")
            if all_linguistic_features:
                sorted_linguistic = sorted(
                    all_linguistic_features.items(), key=lambda x: x[1], reverse=True
                )
                f.write("| Feature | Aggregate Importance | Interpretation |\n")
                f.write("|---------|---------------------|----------------|\n")

                feature_interpretations = {
                    "text_length": "Overall text length in characters",
                    "word_count": "Total number of words",
                    "word_density": "Words per character ratio",
                    "sentence_count": "Number of sentences",
                    "avg_words_per_sentence": "Average words per sentence",
                    "upper_ratio": "Ratio of uppercase characters",
                    "lower_ratio": "Ratio of lowercase characters",
                    "digit_ratio": "Ratio of numeric characters",
                    "punct_ratio": "Ratio of punctuation marks",
                    "lexical_diversity": "Unique words / total words",
                    "avg_word_length": "Average character length per word",
                    "flesch_readability": "Text readability score",
                }

                for feature, importance in sorted_linguistic:
                    interpretation = feature_interpretations.get(
                        feature, "Custom linguistic feature"
                    )
                    f.write(f"| {feature} | {importance:.4f} | {interpretation} |\n")

            # Model-specific analysis
            f.write("\n## Model-Specific Analysis\n\n")

            for result in valid_results:
                model_name = result["model_name"]
                model_type = result["model_type"]
                interp = result["interpretability"]

                f.write(f"### {model_name}\n")
                f.write(f"**Type**: {model_type.title()}\n\n")

                if model_type == "linear":
                    # Linear model analysis
                    if "coefficient_stats" in interp:
                        stats = interp["coefficient_stats"]
                        f.write(
                            f"- **Total Features**: {stats.get('num_positive', 0) + stats.get('num_negative', 0)}\n"
                        )
                        f.write(
                            f"- **Positive Coefficients**: {stats.get('num_positive', 0)}\n"
                        )
                        f.write(
                            f"- **Negative Coefficients**: {stats.get('num_negative', 0)}\n"
                        )
                        f.write(
                            f"- **Max Absolute Coefficient**: {stats.get('max_abs_coef', 0):.4f}\n\n"
                        )

                    # Top positive features (indicate AI)
                    if (
                        "top_positive_features" in interp
                        and interp["top_positive_features"]
                    ):
                        f.write("**Top AI Indicators** (positive coefficients):\n")
                        for i, feature in enumerate(
                            interp["top_positive_features"][:10], 1
                        ):
                            word = feature["feature"].replace("word_", "")
                            f.write(f"{i}. {word} ({feature['coefficient']:.4f})\n")
                        f.write("\n")

                    # Top negative features (indicate Human)
                    if (
                        "top_negative_features" in interp
                        and interp["top_negative_features"]
                    ):
                        f.write("**Top Human Indicators** (negative coefficients):\n")
                        for i, feature in enumerate(
                            interp["top_negative_features"][:10], 1
                        ):
                            word = feature["feature"].replace("word_", "")
                            f.write(f"{i}. {word} ({feature['coefficient']:.4f})\n")
                        f.write("\n")

                elif model_type in ["tree", "ensemble"]:
                    # Tree/ensemble model analysis
                    if "importance_stats" in interp:
                        stats = interp["importance_stats"]
                        f.write(
                            f"- **Mean Feature Importance**: {stats.get('mean_importance', 0):.4f}\n"
                        )
                        f.write(
                            f"- **Max Feature Importance**: {stats.get('max_importance', 0):.4f}\n"
                        )
                        f.write(
                            f"- **Active Features**: {stats.get('num_nonzero_features', 0)}\n\n"
                        )

                    # Top important features
                    if "top_word_features" in interp and interp["top_word_features"]:
                        f.write("**Most Important Words**:\n")
                        for i, feature in enumerate(
                            interp["top_word_features"][:10], 1
                        ):
                            word = feature["feature"].replace("word_", "")
                            f.write(f"{i}. {word} ({feature['importance']:.4f})\n")
                        f.write("\n")

                f.write("---\n\n")

            # Methodology and technical notes
            f.write("## Methodology\n\n")
            f.write("### Feature Types Analyzed\n\n")
            f.write("1. **Word Features**: TF-IDF weighted word occurrences\n")
            f.write(
                "2. **Character Features**: Character n-gram patterns (2-4 chars)\n"
            )
            f.write(
                "3. **Linguistic Features**: Hand-crafted readability and style metrics\n\n"
            )

            f.write("### Model Types\n\n")
            f.write(
                "- **Linear Models**: Feature coefficients show direct word impact\n"
            )
            f.write("- **Tree Models**: Feature importance from decision splits\n")
            f.write(
                "- **Ensemble Models**: Aggregated importance across multiple trees\n\n"
            )

            f.write("### Interpretation Guidelines\n\n")
            f.write(
                "- **Positive coefficients/high importance**: Features that indicate AI-generated text\n"
            )
            f.write(
                "- **Negative coefficients**: Features that indicate human-written text\n"
            )
            f.write("- **High absolute values**: Strong discriminative features\n")
            f.write("- **Consistent across models**: Robust patterns\n\n")

            f.write(
                "Generated using the Comprehensive Model Interpretability Analyzer\n"
            )

        print(f"Interpretability report saved to {report_path}")

    def analyze_individual_prediction(
        self, model, text: str, model_name: str
    ) -> Dict[str, Any]:
        """Analyze what drove a specific prediction."""
        if not self.feature_extractor:
            return {}

        try:
            # Extract features for the single text
            features = self.feature_extractor.transform([text])

            # Get prediction and probability
            prediction = model.predict(features)[0]

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(features)[0]
                confidence = max(prob)
                predicted_class = "AI" if prediction == 1 else "Human"
            else:
                confidence = 0.5
                predicted_class = "AI" if prediction == 1 else "Human"

            explanation = {
                "text": text[:200] + "..." if len(text) > 200 else text,
                "prediction": predicted_class,
                "confidence": confidence,
                "model": model_name,
            }

            # Get feature contributions for interpretable models
            if hasattr(model, "coef_"):
                # Linear model - show top contributing features
                coefficients = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                feature_values = features[0]
                contributions = coefficients * feature_values

                # Get top positive and negative contributions
                contrib_data = []
                for i, contrib in enumerate(contributions):
                    if abs(contrib) > 0.001 and i < len(self.feature_names):
                        contrib_data.append(
                            {
                                "feature": self.feature_names[i],
                                "contribution": contrib,
                                "feature_value": feature_values[i],
                                "coefficient": coefficients[i],
                            }
                        )

                contrib_df = pd.DataFrame(contrib_data)
                contrib_df = contrib_df.sort_values(
                    "contribution", key=abs, ascending=False
                )

                explanation["top_contributions"] = contrib_df.head(10).to_dict(
                    "records"
                )

            return explanation

        except Exception as e:
            print(f"Error analyzing individual prediction: {e}")
            return {}

    def run_comprehensive_analysis(self) -> None:
        """Run the complete interpretability analysis pipeline."""
        print("=" * 80)
        print("COMPREHENSIVE MODEL INTERPRETABILITY ANALYSIS")
        print("Analyzing which features distinguish AI-generated from human text")
        print("=" * 80)

        # Load test data
        print("Loading test data...")
        if not self.load_test_data():
            print("Failed to load test data!")
            return

        # Discover interpretable models
        print("\nDiscovering interpretable models...")
        self.interpretable_models = self.discover_interpretable_models()

        if not self.interpretable_models:
            print("No interpretable models found!")
            return

        # Get feature names from first model
        first_model_data = list(self.interpretable_models.values())[0]
        feature_extractor = first_model_data[2]

        if feature_extractor:
            self.feature_extractor = feature_extractor
            self.feature_names = self.extract_feature_names(feature_extractor)

            # Extract features for analysis
            if self.test_texts:
                try:
                    self.X_test = feature_extractor.transform(self.test_texts)
                    print(
                        f"Extracted {self.X_test.shape[1]} features from {self.X_test.shape[0]} test samples"
                    )
                except Exception as e:
                    print(f"Warning: Could not extract test features: {e}")

        # Analyze each model
        print(f"\nAnalyzing {len(self.interpretable_models)} interpretable models...")
        interpretability_results = []

        for model_name, (model, model_type, _) in self.interpretable_models.items():
            print(f"\nAnalyzing {model_name}...")

            # Determine analysis method based on model characteristics
            if hasattr(model, "coef_"):
                # Linear model
                result = self.analyze_linear_model(model, model_name)
            elif hasattr(model, "feature_importances_"):
                # Tree-based or ensemble model
                if hasattr(model, "estimators_") or hasattr(model, "n_estimators"):
                    result = self.analyze_ensemble_model(model, model_name)
                else:
                    result = self.analyze_tree_model(model, model_name)
            else:
                # Other interpretable models
                print(f"  ⚠ {model_name}: Limited interpretability available")
                result = {
                    "model_name": model_name,
                    "model_type": "other",
                    "interpretability": {},
                }

            interpretability_results.append(result)

        # Create visualizations
        print("\nCreating interpretability visualizations...")
        self.create_feature_importance_plots(interpretability_results)

        # Generate comprehensive report
        print("Generating interpretability report...")
        self.generate_interpretability_report(interpretability_results)

        # Save detailed results as JSON
        results_path = self.output_dir / "detailed_interpretability_results.json"
        with open(results_path, "w") as f:
            json.dump(interpretability_results, f, indent=2, default=str)
        print(f"Detailed results saved to {results_path}")

        # Individual prediction examples
        if self.test_texts and len(self.test_texts) > 0:
            print("\nAnalyzing individual prediction examples...")

            # Get a few example predictions
            example_indices = (
                [0, len(self.test_texts) // 2, -1] if len(self.test_texts) >= 3 else [0]
            )
            examples = []

            for idx in example_indices:
                if idx < len(self.test_texts):
                    text = self.test_texts[idx]

                    # Analyze with first interpretable linear model
                    for model_name, (model, _, _) in self.interpretable_models.items():
                        if hasattr(model, "coef_"):
                            explanation = self.analyze_individual_prediction(
                                model, text, model_name
                            )
                            if explanation:
                                examples.append(explanation)
                                break

            if examples:
                examples_path = self.output_dir / "prediction_examples.json"
                with open(examples_path, "w") as f:
                    json.dump(examples, f, indent=2, default=str)
                print(f"Prediction examples saved to {examples_path}")

        print("\n" + "=" * 80)
        print("INTERPRETABILITY ANALYSIS COMPLETE!")
        print(f"Results saved to: {self.output_dir}/")
        print("Individual visualization files generated:")
        print("  • word_features_analysis.png")
        print("  • model_comparison_heatmap.png")
        print("  • feature_type_distribution.png")
        print("  • coefficient_comparison.png")
        print("  • important_words_wordcloud.png")
        print("Analysis files:")
        print("  • model_interpretability_report.md")
        print("  • detailed_interpretability_results.json")
        if self.test_texts:
            print("  • prediction_examples.json")
        print("=" * 80)


def main():
    """Run the main function to run model interpretability analysis."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Interpretability Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script analyzes interpretable text classification models to reveal:

• Most discriminative words and phrases
• Linguistic patterns that distinguish AI from human text
• Feature importance rankings across different model types
• Individual prediction explanations
• Comprehensive visualizations and reports

Supported interpretable models:
  - Linear models (Logistic Regression, SVM, etc.)
  - Tree-based models (Random Forest, Decision Trees, etc.)
  - Ensemble models (Gradient Boosting, AdaBoost, etc.)

Example usage:
  python interpret_models.py --models-dir models --output-dir interpretability_results
        """,
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained models (default: models)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="interpretability_results",
        help="Directory to save interpretability results (default: interpretability_results)",
    )
    parser.add_argument(
        "--human-file",
        type=str,
        default="test_data_human.jsonl",
        help="Human test data file (default: test_data_human.jsonl)",
    )
    parser.add_argument(
        "--ai-file",
        type=str,
        default="test_data_ai.jsonl",
        help="AI test data file (default: test_data_ai.jsonl)",
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = ModelInterpretabilityAnalyzer(
        models_dir=args.models_dir, output_dir=args.output_dir
    )

    # Run comprehensive analysis
    analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    main()
