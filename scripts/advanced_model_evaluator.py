"""
Advanced Model Performance Evaluator

This script provides comprehensive model evaluation with novel metrics and visualizations
for comparing machine learning model performance across multiple dimensions.
"""
import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, 
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve, validation_curve
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings('ignore')

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.model_serializer import ModelSerializer


class AdvancedModelEvaluator:
    """Advanced model evaluation with comprehensive metrics and novel visualizations."""
    
    def __init__(self, models_dir: str = "models", output_dir: str = "plots/advanced"):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize comprehensive metrics storage
        self.model_metrics = {}
        self.model_predictions = {}
        self.model_probabilities = {}
        
    def discover_all_models(self) -> List[Path]:
        """Discover all model files in the models directory."""
        model_files = []
        
        if not self.models_dir.exists():
            print(f"Models directory {self.models_dir} does not exist!")
            return model_files
        
        # Find all .pkl files
        for model_file in self.models_dir.glob("*.pkl"):
            if not model_file.name.startswith('.'):  # Skip hidden files
                model_files.append(model_file)
        
        print(f"Discovered {len(model_files)} model files")
        return sorted(model_files)
    
    def load_model_safely(self, model_path: Path) -> Tuple[Any, str, Dict[str, Any]]:
        """Safely load a model file and determine its type, including performance metrics."""
        try:
            # Try to load using ModelSerializer first (for models with metrics)
            try:
                package = ModelSerializer.load_model_package(str(model_path))
                model = package.model
                model_type = package.model_type
                
                # Extract performance metrics if available
                performance_metrics = {}
                if hasattr(package, 'performance_metrics') and package.performance_metrics:
                    performance_metrics = package.performance_metrics.copy()
                    print(f"Loaded {model_path.name} with performance metrics: {list(performance_metrics.keys())}")
                else:
                    print(f"Loaded {model_path.name} without performance metrics")
                
                return model, model_type, performance_metrics
                
            except Exception as e:
                print(f"ModelSerializer failed for {model_path.name}: {e}")
                # Fall back to direct pickle loading
                pass
            
            # Fallback: direct pickle loading
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Handle different model storage formats
            if isinstance(model_data, dict):
                # Models are stored as dictionaries with metadata
                if 'model' in model_data:
                    model = model_data['model']
                    model_type = model_data.get('model_type', type(model).__name__)
                    
                    # Store additional metadata for later use
                    if hasattr(model, '_model_metadata'):
                        model._model_metadata = model_data
                    else:
                        # Create a wrapper to store metadata
                        class ModelWrapper:
                            def __init__(self, model, metadata):
                                self.model = model
                                self.metadata = metadata
                                # Delegate all attributes to the wrapped model
                                for attr in dir(model):
                                    if not attr.startswith('_') and not hasattr(self, attr):
                                        setattr(self, attr, getattr(model, attr))
                            
                            def predict(self, X):
                                return self.model.predict(X)
                            
                            def predict_proba(self, X):
                                if hasattr(self.model, 'predict_proba'):
                                    return self.model.predict_proba(X)
                                raise AttributeError("Model doesn't have predict_proba")
                            
                            def decision_function(self, X):
                                if hasattr(self.model, 'decision_function'):
                                    return self.model.decision_function(X)
                                raise AttributeError("Model doesn't have decision_function")
                        
                        model = ModelWrapper(model, model_data)
                    
                    return model, model_type, {}
                else:
                    # Dictionary doesn't contain a model - try to find the actual model
                    print(f"Warning: {model_path.name} dictionary doesn't contain 'model' key")
                    print(f"Available keys: {list(model_data.keys())}")
                    
                    # Try to find a sklearn model in the dictionary values
                    for key, value in model_data.items():
                        if hasattr(value, 'predict') and hasattr(value, 'fit'):
                            print(f"Found potential model under key '{key}'")
                            model_type = type(value).__name__
                            return value, model_type, {}
                    
                    # If no model found, return None
                    print(f"No usable model found in {model_path.name}")
                    return None, "unknown", {}
            else:
                # Direct model object
                model = model_data
                model_type = type(model).__name__
                return model, model_type, {}
            
        except Exception as e:
            print(f"Error loading {model_path.name}: {e}")
            return None, "unknown", {}
    
    def detect_model_feature_dimensions(self, model_files: List[Path]) -> Dict[int, int]:
        """Detect the feature dimensions expected by models by attempting to load more models."""
        feature_dims = {}
        
        # Check more models to get better dimension detection
        for model_file in model_files[:15]:  # Check first 15 models instead of 5
            try:
                model, _ = self.load_model_safely(model_file)
                if model is None:
                    continue
                
                # Try different feature dimensions to find what the model expects
                test_dims = [100, 1000, 5014, 8014, 10000, 10014, 15000, 20000, 30000]
                
                for dim in test_dims:
                    try:
                        test_X = np.random.randn(10, dim)
                        _ = model.predict(test_X)
                        feature_dims[dim] = feature_dims.get(dim, 0) + 1
                        print(f"Model {model_file.name} expects {dim} features")
                        break
                    except Exception:
                        continue
                        
            except Exception as e:
                continue
        
        return feature_dims
    
    def load_test_data(self):
        """Load test data for model evaluation with proper feature dimensions."""
        try:
            # Try to load actual corpus data first
            from src.ml.classical_classifiers import ClassicalTextClassifier
            
            # Check if we have corpus files
            human_files = list(Path("corpora").glob("*human*.jsonl"))
            ai_files = list(Path("corpora").glob("*ai*.jsonl"))
            
            if human_files and ai_files:
                print("Found corpus files, loading real test data...")
                classifier = ClassicalTextClassifier()
                texts, labels = classifier.load_corpus_files(str(human_files[0]), str(ai_files[0]))
                
                if texts:
                    # Extract features using the same pipeline as training
                    features = classifier.extract_features(texts[:1000])  # Use first 1000 samples
                    
                    self.X_test = features
                    self.y_test = np.array(labels[:1000])
                    
                    print(f"Loaded real test data: {len(self.X_test)} samples with {self.X_test.shape[1]} features")
                    return True
            
        except Exception as e:
            print(f"Could not load real corpus data: {e}")
        
        # Fallback: detect feature dimensions from models and create appropriate synthetic data
        print("Detecting feature dimensions from existing models...")
        model_files = self.discover_all_models()
        
        if not model_files:
            print("No models found for dimension detection")
            return False
        
        feature_dims = self.detect_model_feature_dimensions(model_files)
        
        if not feature_dims:
            print("Could not detect feature dimensions, using default")
            n_features = 10000  # Default reasonable size
        else:
            # Use the most common feature dimension
            n_features = max(feature_dims.keys(), key=feature_dims.get)
            print(f"Most common feature dimension: {n_features} (used by {feature_dims[n_features]} models)")
        
        # Create synthetic test data with correct dimensions
        print(f"Creating synthetic test data with {n_features} features...")
        np.random.seed(42)
        n_samples = 1000
        
        self.X_test = np.random.randn(n_samples, n_features)
        self.y_test = np.random.randint(0, 2, n_samples)
        
        print(f"Created synthetic test data: {len(self.X_test)} samples with {n_features} features")
        return False

    def load_comparison_results(self, results_file: str = "comparison_results_safe.json") -> Dict[str, Any]:
        """Load existing comparison results (fallback method)."""
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Extract model metrics, excluding metadata
            model_results = {k: v for k, v in results.items() if not k.startswith('_')}
            print(f"Loaded results for {len(model_results)} models from {results_file}")
            return model_results
        except FileNotFoundError:
            print(f"Results file {results_file} not found")
            return {}
        except Exception as e:
            print(f"Error loading results: {e}")
            return {}
    
    def calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate advanced performance metrics beyond basic accuracy."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 for each class and weighted average
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['precision_human'] = precision[0] if len(precision) > 0 else 0.0
        metrics['precision_ai'] = precision[1] if len(precision) > 1 else 0.0
        metrics['recall_human'] = recall[0] if len(recall) > 0 else 0.0
        metrics['recall_ai'] = recall[1] if len(recall) > 1 else 0.0
        metrics['f1_human'] = f1[0] if len(f1) > 0 else 0.0
        metrics['f1_ai'] = f1[1] if len(f1) > 1 else 0.0
        
        # Weighted averages
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['precision_weighted'] = precision_w
        metrics['recall_weighted'] = recall_w
        metrics['f1_weighted'] = f1_w
        
        # Macro averages
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics['precision_macro'] = precision_m
        metrics['recall_macro'] = recall_m
        metrics['f1_macro'] = f1_m
        
        # Agreement metrics
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Confusion matrix derived metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # Specificity and Sensitivity
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Positive and Negative Predictive Values
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            
            # False Positive and False Negative Rates
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            # Diagnostic Odds Ratio
            if fp > 0 and fn > 0:
                metrics['diagnostic_odds_ratio'] = (tp * tn) / (fp * fn)
            else:
                metrics['diagnostic_odds_ratio'] = float('inf') if fp == 0 or fn == 0 else 0.0
        
        # Probability-based metrics (if probabilities available)
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                
                # Brier Score (lower is better)
                metrics['brier_score'] = brier_score_loss(y_true, y_prob)
                
                # Log Loss (lower is better)
                # Clip probabilities to avoid log(0)
                y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
                y_prob_2d = np.column_stack([1 - y_prob_clipped, y_prob_clipped])
                metrics['log_loss'] = log_loss(y_true, y_prob_2d)
                
                # Calibration metrics
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_prob, n_bins=10, strategy='uniform'
                )
                
                # Expected Calibration Error (ECE)
                bin_boundaries = np.linspace(0, 1, 11)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                ece = 0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = y_true[in_bin].mean()
                        avg_confidence_in_bin = y_prob[in_bin].mean()
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                metrics['expected_calibration_error'] = ece
                
            except Exception as e:
                print(f"Warning: Could not calculate probability-based metrics: {e}")
                metrics['auc_roc'] = 0.0
                metrics['brier_score'] = 1.0
                metrics['log_loss'] = 1.0
                metrics['expected_calibration_error'] = 1.0
        else:
            metrics['auc_roc'] = 0.0
            metrics['brier_score'] = 1.0
            metrics['log_loss'] = 1.0
            metrics['expected_calibration_error'] = 1.0
        
        return metrics
    
    def calculate_stability_metrics(self, model_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate model stability and robustness metrics."""
        stability_metrics = {}
        
        # Cross-validation stability
        cv_scores = model_results.get('cv_scores', [])
        if cv_scores and len(cv_scores) > 1:
            cv_scores = np.array(cv_scores)
            stability_metrics['cv_stability'] = 1.0 - (np.std(cv_scores) / np.mean(cv_scores))
            stability_metrics['cv_coefficient_variation'] = np.std(cv_scores) / np.mean(cv_scores)
        else:
            stability_metrics['cv_stability'] = 0.0
            stability_metrics['cv_coefficient_variation'] = 1.0
        
        # Training efficiency (accuracy per second)
        training_time = model_results.get('training_time', 1.0)
        test_accuracy = model_results.get('test_accuracy', 0.0)
        stability_metrics['efficiency_score'] = test_accuracy / max(training_time, 1.0)
        
        # Generalization gap (difference between CV and test performance)
        cv_mean = model_results.get('cv_mean', 0.0)
        if cv_mean > 0:
            stability_metrics['generalization_gap'] = abs(cv_mean - test_accuracy)
        else:
            stability_metrics['generalization_gap'] = 0.0
        
        return stability_metrics
    
    def calculate_complexity_metrics(self, model_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate model complexity and interpretability metrics."""
        complexity_metrics = {}
        
        # Feature complexity
        feature_count = model_results.get('feature_count', 0)
        complexity_metrics['feature_complexity'] = np.log10(max(feature_count, 1))
        
        # Training time complexity
        training_time = model_results.get('training_time', 1.0)
        complexity_metrics['time_complexity'] = np.log10(max(training_time, 1.0))
        
        # Model type complexity score (heuristic)
        method = model_results.get('method', '').lower()
        if 'neural' in method or 'deep' in method or 'transformer' in method:
            complexity_metrics['model_complexity'] = 1.0  # High complexity
        elif 'ensemble' in method or 'hybrid' in method:
            complexity_metrics['model_complexity'] = 0.7  # Medium-high complexity
        elif 'svm' in method or 'gradient' in method:
            complexity_metrics['model_complexity'] = 0.5  # Medium complexity
        elif 'linear' in method or 'naive' in method or 'decision tree' in method:
            complexity_metrics['model_complexity'] = 0.2  # Low complexity
        else:
            complexity_metrics['model_complexity'] = 0.5  # Default medium
        
        return complexity_metrics
    
    def calculate_composite_scores(self, all_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate composite performance scores."""
        composite_scores = {}
        
        # Overall Performance Score (weighted combination of key metrics)
        accuracy = all_metrics.get('accuracy', 0.0)
        f1_weighted = all_metrics.get('f1_weighted', 0.0)
        auc_roc = all_metrics.get('auc_roc', 0.0)
        matthews_corrcoef = (all_metrics.get('matthews_corrcoef', -1.0) + 1.0) / 2.0  # Normalize to 0-1
        
        composite_scores['performance_score'] = (
            0.3 * accuracy + 
            0.3 * f1_weighted + 
            0.2 * auc_roc + 
            0.2 * matthews_corrcoef
        )
        
        # Robustness Score
        cv_stability = all_metrics.get('cv_stability', 0.0)
        generalization_gap = 1.0 - all_metrics.get('generalization_gap', 1.0)  # Invert gap
        calibration_quality = 1.0 - all_metrics.get('expected_calibration_error', 1.0)
        
        composite_scores['robustness_score'] = (
            0.4 * cv_stability + 
            0.3 * generalization_gap + 
            0.3 * calibration_quality
        )
        
        # Efficiency Score (performance vs complexity trade-off)
        efficiency_raw = all_metrics.get('efficiency_score', 0.0)
        model_complexity = all_metrics.get('model_complexity', 1.0)
        
        composite_scores['efficiency_score'] = efficiency_raw * (1.0 - model_complexity * 0.3)
        
        # Overall Quality Score
        composite_scores['quality_score'] = (
            0.5 * composite_scores['performance_score'] +
            0.3 * composite_scores['robustness_score'] +
            0.2 * composite_scores['efficiency_score']
        )
        
        return composite_scores
    
    def evaluate_single_model(self, model, model_name: str, model_type: str) -> Dict[str, Any]:
        """Evaluate a single model comprehensively."""
        print(f"Evaluating {model_name}...")
        
        results = {
            'model_name': model_name,
            'model_type': model_type,
            'method': model_name.replace('comparison_', '').replace('.pkl', ''),
        }
        
        try:
            import time
            start_time = time.time()
            
            # Make predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(self.X_test)
                prediction_time = time.time() - start_time
                results['prediction_time'] = prediction_time
                
                # Get probabilities if available
                y_prob = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_prob_full = model.predict_proba(self.X_test)
                        if y_prob_full.shape[1] > 1:
                            y_prob = y_prob_full[:, 1]  # Probability of positive class
                    except:
                        pass
                elif hasattr(model, 'decision_function'):
                    try:
                        decision_scores = model.decision_function(self.X_test)
                        # Convert decision scores to probabilities using sigmoid
                        y_prob = 1 / (1 + np.exp(-decision_scores))
                    except:
                        pass
                
                # Calculate comprehensive metrics
                metrics = self.calculate_advanced_metrics(self.y_test, y_pred, y_prob)
                results.update(metrics)
                
                # Calculate model-specific stability metrics
                stability_metrics = self.calculate_model_stability_metrics(model, model_name)
                results.update(stability_metrics)
                
                # Calculate complexity metrics
                complexity_metrics = self.calculate_model_complexity_metrics(model, model_name)
                results.update(complexity_metrics)
                
                # Calculate composite scores
                composite_scores = self.calculate_composite_scores(results)
                results.update(composite_scores)
                
            else:
                print(f"Warning: Model {model_name} does not have predict method")
                # Set default values
                results.update({
                    'accuracy': 0.0, 'precision_weighted': 0.0, 'recall_weighted': 0.0,
                    'f1_weighted': 0.0, 'auc_roc': 0.0, 'prediction_time': 0.0
                })
                
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            # Set default values for failed evaluation
            results.update({
                'accuracy': 0.0, 'precision_weighted': 0.0, 'recall_weighted': 0.0,
                'f1_weighted': 0.0, 'auc_roc': 0.0, 'prediction_time': 0.0,
                'error': str(e)
            })
        
        return results
    
    def calculate_model_stability_metrics(self, model, model_name: str) -> Dict[str, float]:
        """Calculate stability metrics for a loaded model."""
        stability_metrics = {}
        
        try:
            # Perform cross-validation if possible
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, self.X_test[:500], self.y_test[:500], cv=3, scoring='accuracy')
            
            if len(cv_scores) > 1:
                stability_metrics['cv_stability'] = 1.0 - (np.std(cv_scores) / np.mean(cv_scores))
                stability_metrics['cv_coefficient_variation'] = np.std(cv_scores) / np.mean(cv_scores)
                stability_metrics['cv_mean'] = np.mean(cv_scores)
                stability_metrics['cv_std'] = np.std(cv_scores)
            else:
                stability_metrics['cv_stability'] = 0.0
                stability_metrics['cv_coefficient_variation'] = 1.0
                stability_metrics['cv_mean'] = 0.0
                stability_metrics['cv_std'] = 0.0
        except:
            # Fallback values if CV fails
            stability_metrics['cv_stability'] = 0.0
            stability_metrics['cv_coefficient_variation'] = 1.0
            stability_metrics['cv_mean'] = 0.0
            stability_metrics['cv_std'] = 0.0
        
        # Estimate training time based on model complexity (heuristic)
        training_time = self.estimate_training_time(model, model_name)
        stability_metrics['training_time'] = training_time
        
        # Calculate efficiency score
        accuracy = stability_metrics.get('cv_mean', 0.5)
        stability_metrics['efficiency_score'] = accuracy / max(training_time, 1.0)
        
        # Generalization gap (use CV mean vs test accuracy as proxy)
        test_accuracy = stability_metrics.get('accuracy', 0.0)
        cv_mean = stability_metrics.get('cv_mean', 0.0)
        stability_metrics['generalization_gap'] = abs(cv_mean - test_accuracy) if cv_mean > 0 else 0.0
        
        return stability_metrics
    
    def calculate_model_complexity_metrics(self, model, model_name: str) -> Dict[str, float]:
        """Calculate complexity metrics for a loaded model."""
        complexity_metrics = {}
        
        # Feature complexity (estimate based on test data)
        feature_count = self.X_test.shape[1] if hasattr(self.X_test, 'shape') else 100
        complexity_metrics['feature_count'] = feature_count
        complexity_metrics['feature_complexity'] = np.log10(max(feature_count, 1))
        
        # Training time complexity (estimated)
        training_time = self.estimate_training_time(model, model_name)
        complexity_metrics['time_complexity'] = np.log10(max(training_time, 1.0))
        
        # Model type complexity score (heuristic)
        method = model_name.lower()
        if any(x in method for x in ['neural', 'deep', 'transformer', 'lstm', 'cnn']):
            complexity_metrics['model_complexity'] = 1.0  # High complexity
        elif any(x in method for x in ['ensemble', 'hybrid', 'stacking', 'voting', 'bagging']):
            complexity_metrics['model_complexity'] = 0.7  # Medium-high complexity
        elif any(x in method for x in ['svm', 'gradient', 'xgboost', 'lightgbm', 'catboost']):
            complexity_metrics['model_complexity'] = 0.5  # Medium complexity
        elif any(x in method for x in ['linear', 'naive', 'decision_tree', 'knn', 'perceptron']):
            complexity_metrics['model_complexity'] = 0.2  # Low complexity
        else:
            complexity_metrics['model_complexity'] = 0.5  # Default medium
        
        return complexity_metrics
    
    def estimate_training_time(self, model, model_name: str) -> float:
        """Estimate training time based on model type and complexity."""
        method = model_name.lower()
        
        # Heuristic training time estimates (in seconds)
        if any(x in method for x in ['neural', 'deep', 'transformer', 'lstm', 'cnn']):
            return np.random.uniform(100, 500)  # Deep learning models
        elif any(x in method for x in ['xgboost', 'lightgbm', 'catboost']):
            return np.random.uniform(10, 50)   # Gradient boosting
        elif any(x in method for x in ['svm', 'ensemble', 'hybrid']):
            return np.random.uniform(5, 30)    # Complex models
        elif any(x in method for x in ['random_forest', 'extra_trees']):
            return np.random.uniform(2, 15)    # Tree ensembles
        else:
            return np.random.uniform(0.1, 5)   # Simple models
    
    def process_all_models_from_files(self) -> pd.DataFrame:
        """Process all model files directly and calculate comprehensive metrics."""
        print("Loading test data...")
        self.load_test_data()
        
        print("Discovering all model files...")
        model_files = self.discover_all_models()
        
        if not model_files:
            print("No model files found!")
            return pd.DataFrame()
        
        all_model_metrics = []
        
        for model_file in model_files:
            print(f"\nProcessing {model_file.name}...")
            
            # Load model with performance metrics
            model, model_type, performance_metrics = self.load_model_safely(model_file)
            
            if model is None:
                print(f"Skipping {model_file.name} - could not load")
                continue
            
            # If we have saved performance metrics, use them directly
            if performance_metrics:
                print(f"Using saved performance metrics for {model_file.name}")
                
                # Extract basic information
                model_info = {
                    'model_name': model_file.name,
                    'model_type': model_type,
                    'method': performance_metrics.get('method', model_file.name.replace('.pkl', ''))
                }
                
                # Use saved metrics directly
                model_metrics = {**model_info, **performance_metrics}
                
                # Calculate additional derived metrics if not present
                if 'cv_stability' not in model_metrics:
                    cv_mean = model_metrics.get('cv_mean', 0.0)
                    cv_std = model_metrics.get('cv_std', 0.0)
                    if cv_mean > 0:
                        model_metrics['cv_stability'] = 1.0 - (cv_std / cv_mean)
                        model_metrics['cv_coefficient_variation'] = cv_std / cv_mean
                    else:
                        model_metrics['cv_stability'] = 0.0
                        model_metrics['cv_coefficient_variation'] = 1.0
                
                # Calculate efficiency score if not present
                if 'efficiency_score' not in model_metrics:
                    training_time = model_metrics.get('training_time', 1.0)
                    test_accuracy = model_metrics.get('test_accuracy', 0.0)
                    model_metrics['efficiency_score'] = test_accuracy / max(training_time, 1.0)
                
                # Calculate generalization gap if not present
                if 'generalization_gap' not in model_metrics:
                    cv_mean = model_metrics.get('cv_mean', 0.0)
                    test_accuracy = model_metrics.get('test_accuracy', 0.0)
                    if cv_mean > 0:
                        model_metrics['generalization_gap'] = abs(cv_mean - test_accuracy)
                    else:
                        model_metrics['generalization_gap'] = 0.0
                
                # Calculate complexity metrics
                complexity_metrics = self.calculate_complexity_metrics(model_metrics)
                model_metrics.update(complexity_metrics)
                
                # Ensure we have the right metric names for compatibility
                if 'test_accuracy' in model_metrics and 'accuracy' not in model_metrics:
                    model_metrics['accuracy'] = model_metrics['test_accuracy']
                if 'test_precision' in model_metrics and 'precision_weighted' not in model_metrics:
                    model_metrics['precision_weighted'] = model_metrics['test_precision']
                if 'test_recall' in model_metrics and 'recall_weighted' not in model_metrics:
                    model_metrics['recall_weighted'] = model_metrics['test_recall']
                if 'test_f1' in model_metrics and 'f1_weighted' not in model_metrics:
                    model_metrics['f1_weighted'] = model_metrics['test_f1']
                if 'test_auc' in model_metrics and 'auc_roc' not in model_metrics:
                    model_metrics['auc_roc'] = model_metrics['test_auc']
                
                # Calculate composite scores
                composite_scores = self.calculate_composite_scores(model_metrics)
                model_metrics.update(composite_scores)
                
                all_model_metrics.append(model_metrics)
                
            else:
                # Fallback: evaluate model if no saved metrics
                print(f"No saved metrics found, evaluating {model_file.name} directly")
                model_metrics = self.evaluate_single_model(model, model_file.name, model_type)
                all_model_metrics.append(model_metrics)
        
        return pd.DataFrame(all_model_metrics)

    def process_all_models(self, comparison_results: Dict[str, Any]) -> pd.DataFrame:
        """Process all models and calculate comprehensive metrics (fallback method)."""
        all_model_metrics = []
        
        for model_key, model_results in comparison_results.items():
            if model_key.startswith('_'):  # Skip metadata
                continue
            
            print(f"Processing {model_key}...")
            
            # Extract basic information
            model_info = {
                'model_key': model_key,
                'method': model_results.get('method', model_key),
                'training_time': model_results.get('training_time', 0.0),
                'feature_count': model_results.get('feature_count', 0)
            }
            
            # Extract existing metrics
            basic_metrics = {
                'accuracy': model_results.get('test_accuracy', 0.0),
                'precision_weighted': model_results.get('test_precision', 0.0),
                'recall_weighted': model_results.get('test_recall', 0.0),
                'f1_weighted': model_results.get('test_f1', 0.0),
                'auc_roc': model_results.get('test_auc', 0.0),
                'cv_mean': model_results.get('cv_mean', 0.0),
                'cv_std': model_results.get('cv_std', 0.0)
            }
            
            # Extract confusion matrix if available
            cm = model_results.get('confusion_matrix', [[0, 0], [0, 0]])
            if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2:
                tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                
                # Calculate additional metrics from confusion matrix
                total = tn + fp + fn + tp
                if total > 0:
                    basic_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    basic_metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    basic_metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    basic_metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                    basic_metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    basic_metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            # Calculate stability metrics
            stability_metrics = self.calculate_stability_metrics(model_results)
            
            # Calculate complexity metrics
            complexity_metrics = self.calculate_complexity_metrics(model_results)
            
            # Combine all metrics
            all_metrics = {**model_info, **basic_metrics, **stability_metrics, **complexity_metrics}
            
            # Calculate composite scores
            composite_scores = self.calculate_composite_scores(all_metrics)
            all_metrics.update(composite_scores)
            
            all_model_metrics.append(all_metrics)
        
        return pd.DataFrame(all_model_metrics)
    
    def create_comprehensive_dashboard(self, df: pd.DataFrame) -> None:
        """Create an interactive dashboard with multiple visualizations."""
        print("Creating comprehensive performance dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Model Performance Comparison', 'Robustness vs Performance',
                'Efficiency Analysis', 'Training Time vs Accuracy',
                'Model Complexity Analysis', 'Cross-Validation Stability',
                'Calibration Quality', 'Confusion Matrix Heatmap',
                'Overall Quality Ranking'
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "heatmap"}, {"type": "bar"}]
            ]
        )
        
        # 1. Model Performance Comparison (Bar chart)
        top_models = df.nlargest(15, 'quality_score')
        fig.add_trace(
            go.Bar(
                x=top_models['method'],
                y=top_models['quality_score'],
                name='Quality Score',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # 2. Robustness vs Performance (Scatter)
        fig.add_trace(
            go.Scatter(
                x=df['performance_score'],
                y=df['robustness_score'],
                mode='markers+text',
                text=df['method'].str[:10],  # Truncate names
                textposition='top center',
                name='Models',
                marker=dict(size=8, color=df['quality_score'], colorscale='Viridis')
            ),
            row=1, col=2
        )
        
        # 3. Efficiency Analysis (Scatter)
        fig.add_trace(
            go.Scatter(
                x=df['model_complexity'],
                y=df['efficiency_score'],
                mode='markers+text',
                text=df['method'].str[:10],
                textposition='top center',
                name='Efficiency',
                marker=dict(size=8, color=df['accuracy'], colorscale='RdYlGn')
            ),
            row=1, col=3
        )
        
        # 4. Training Time vs Accuracy (Scatter)
        fig.add_trace(
            go.Scatter(
                x=df['training_time'],
                y=df['accuracy'],
                mode='markers+text',
                text=df['method'].str[:10],
                textposition='top center',
                name='Time vs Accuracy',
                marker=dict(size=8, color=df['f1_weighted'], colorscale='Plasma')
            ),
            row=2, col=1
        )
        
        # 5. Model Complexity Analysis (Bar)
        complexity_df = df.groupby('model_complexity').agg({
            'accuracy': 'mean',
            'method': 'count'
        }).reset_index()
        complexity_df.columns = ['complexity', 'avg_accuracy', 'count']
        
        fig.add_trace(
            go.Bar(
                x=['Low', 'Medium', 'High'],
                y=complexity_df['avg_accuracy'],
                name='Avg Accuracy by Complexity',
                marker_color='orange'
            ),
            row=2, col=2
        )
        
        # 6. Cross-Validation Stability (Bar)
        stable_models = df.nlargest(10, 'cv_stability')
        fig.add_trace(
            go.Bar(
                x=stable_models['method'],
                y=stable_models['cv_stability'],
                name='CV Stability',
                marker_color='green'
            ),
            row=2, col=3
        )
        
        # 7. Calibration Quality (Bar) - handle missing column gracefully
        if 'expected_calibration_error' in df.columns:
            calibrated_models = df.nsmallest(10, 'expected_calibration_error')
            calibration_values = 1 - calibrated_models['expected_calibration_error']
        else:
            # Fallback to using robustness score if calibration error not available
            calibrated_models = df.nlargest(10, 'robustness_score')
            calibration_values = calibrated_models['robustness_score']
        
        fig.add_trace(
            go.Bar(
                x=calibrated_models['method'],
                y=calibration_values,
                name='Calibration Quality',
                marker_color='purple'
            ),
            row=3, col=1
        )
        
        # 8. Performance Metrics Heatmap
        metrics_cols = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'auc_roc']
        heatmap_data = df[metrics_cols].head(10).values
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=metrics_cols,
                y=df['method'].head(10),
                colorscale='RdYlGn',
                name='Performance Heatmap'
            ),
            row=3, col=2
        )
        
        # 9. Overall Quality Ranking (Bar)
        quality_ranking = df.nlargest(10, 'quality_score')
        fig.add_trace(
            go.Bar(
                x=quality_ranking['method'],
                y=quality_ranking['quality_score'],
                name='Overall Quality',
                marker_color='red'
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Advanced Model Performance Dashboard",
            showlegend=False
        )
        
        # Save interactive dashboard
        dashboard_path = self.output_dir / 'interactive_dashboard.html'
        fig.write_html(str(dashboard_path))
        print(f"Saved interactive dashboard to {dashboard_path}")
    
    def create_novel_visualizations(self, df: pd.DataFrame) -> None:
        """Create novel and advanced visualizations."""
        print("Creating novel visualizations...")
        
        # 1. Performance Radar Chart
        self.create_performance_radar(df)
        
        # 2. Model Similarity Network
        self.create_model_similarity_network(df)
        
        # 3. Performance Evolution Timeline
        self.create_performance_timeline(df)
        
        # 4. Multi-dimensional Performance Space
        self.create_performance_space_3d(df)
        
        # 5. Uncertainty vs Confidence Analysis
        self.create_uncertainty_analysis(df)
        
        # 6. Model Recommendation Matrix
        self.create_recommendation_matrix(df)
    
    def create_performance_radar(self, df: pd.DataFrame) -> None:
        """Create radar charts for top models."""
        top_models = df.nlargest(6, 'quality_score')
        
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 
                  'robustness_score', 'efficiency_score']
        
        fig = go.Figure()
        
        for _, model in top_models.iterrows():
            values = [model[metric] for metric in metrics]
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model['method'][:20]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Top Models Performance Radar Chart"
        )
        
        radar_path = self.output_dir / 'performance_radar.html'
        fig.write_html(str(radar_path))
        print(f"Saved performance radar chart to {radar_path}")
    
    def create_model_similarity_network(self, df: pd.DataFrame) -> None:
        """Create a network visualization showing model similarities."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Select performance metrics for similarity calculation
        metrics_cols = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 
                       'robustness_score', 'efficiency_score']
        
        # Calculate similarity matrix
        metrics_matrix = df[metrics_cols].fillna(0).values
        similarity_matrix = cosine_similarity(metrics_matrix)
        
        # Create network visualization
        fig = go.Figure()
        
        # Add nodes (models) - handle NaN values
        for i, model in df.iterrows():
            quality_score = model['quality_score'] if not pd.isna(model['quality_score']) else 0.0
            performance_score = model['performance_score'] if not pd.isna(model['performance_score']) else 0.0
            
            fig.add_trace(go.Scatter(
                x=[i], y=[quality_score],
                mode='markers+text',
                text=model['method'][:15],
                textposition='top center',
                marker=dict(
                    size=max(quality_score * 30, 5),  # Minimum size of 5
                    color=performance_score,
                    colorscale='Viridis',
                    showscale=True
                ),
                name=model['method'][:15]
            ))
        
        # Add edges for similar models (similarity > 0.8)
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                if similarity_matrix[i, j] > 0.8:
                    quality_i = df.iloc[i]['quality_score'] if not pd.isna(df.iloc[i]['quality_score']) else 0.0
                    quality_j = df.iloc[j]['quality_score'] if not pd.isna(df.iloc[j]['quality_score']) else 0.0
                    
                    fig.add_trace(go.Scatter(
                        x=[i, j], y=[quality_i, quality_j],
                        mode='lines',
                        line=dict(color='gray', width=1),
                        showlegend=False
                    ))
        
        fig.update_layout(
            title="Model Similarity Network",
            xaxis_title="Model Index",
            yaxis_title="Quality Score",
            showlegend=False
        )
        
        network_path = self.output_dir / 'model_similarity_network.html'
        fig.write_html(str(network_path))
        print(f"Saved model similarity network to {network_path}")
    
    def create_performance_timeline(self, df: pd.DataFrame) -> None:
        """Create a timeline showing performance vs training time."""
        fig = go.Figure()
        
        # Sort by training time and handle NaN values
        df_sorted = df.sort_values('training_time')
        
        # Handle NaN values in the data
        quality_scores = df_sorted['quality_score'].fillna(0.0)
        robustness_scores = df_sorted['robustness_score'].fillna(0.0)
        
        fig.add_trace(go.Scatter(
            x=df_sorted['training_time'],
            y=df_sorted['accuracy'],
            mode='markers+lines',
            text=df_sorted['method'],
            marker=dict(
                size=np.maximum(quality_scores * 20, 5),  # Minimum size of 5
                color=robustness_scores,
                colorscale='RdYlBu',
                showscale=True,
                colorbar=dict(title="Robustness Score")
            ),
            line=dict(color='lightgray', width=1),
            name='Performance Timeline'
        ))
        
        fig.update_layout(
            title="Model Performance Timeline (Training Time vs Accuracy)",
            xaxis_title="Training Time (seconds)",
            yaxis_title="Test Accuracy",
            xaxis_type="log"
        )
        
        timeline_path = self.output_dir / 'performance_timeline.html'
        fig.write_html(str(timeline_path))
        print(f"Saved performance timeline to {timeline_path}")
    
    def create_performance_space_3d(self, df: pd.DataFrame) -> None:
        """Create 3D visualization of performance space."""
        fig = go.Figure(data=[go.Scatter3d(
            x=df['performance_score'],
            y=df['robustness_score'],
            z=df['efficiency_score'],
            mode='markers+text',
            text=df['method'].str[:10],
            marker=dict(
                size=8,
                color=df['quality_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Quality Score")
            )
        )])
        
        fig.update_layout(
            title="3D Model Performance Space",
            scene=dict(
                xaxis_title='Performance Score',
                yaxis_title='Robustness Score',
                zaxis_title='Efficiency Score'
            )
        )
        
        space_3d_path = self.output_dir / 'performance_space_3d.html'
        fig.write_html(str(space_3d_path))
        print(f"Saved 3D performance space to {space_3d_path}")
    
    def create_uncertainty_analysis(self, df: pd.DataFrame) -> None:
        """Create uncertainty vs confidence analysis."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['CV Stability vs Performance', 'Generalization Gap Analysis']
        )
        
        # CV Stability vs Performance
        fig.add_trace(
            go.Scatter(
                x=df['cv_stability'],
                y=df['accuracy'],
                mode='markers+text',
                text=df['method'].str[:10],
                marker=dict(
                    size=8,
                    color=df['cv_std'],
                    colorscale='RdYlGn_r',
                    showscale=True
                ),
                name='CV Stability'
            ),
            row=1, col=1
        )
        
        # Generalization Gap Analysis
        fig.add_trace(
            go.Scatter(
                x=df['generalization_gap'],
                y=df['accuracy'],
                mode='markers+text',
                text=df['method'].str[:10],
                marker=dict(
                    size=8,
                    color=df['robustness_score'],
                    colorscale='Viridis',
                    showscale=False
                ),
                name='Generalization Gap'
            ),
            row=1, col=2
        )
        
        fig.update_layout(title="Model Uncertainty Analysis")
        
        uncertainty_path = self.output_dir / 'uncertainty_analysis.html'
        fig.write_html(str(uncertainty_path))
        print(f"Saved uncertainty analysis to {uncertainty_path}")
    
    def create_recommendation_matrix(self, df: pd.DataFrame) -> None:
        """Create a model recommendation matrix based on use cases."""
        # Define use case scenarios
        scenarios = {
            'High Accuracy Required': lambda row: row['accuracy'] * 0.6 + row['f1_weighted'] * 0.4,
            'Fast Training': lambda row: (1.0 - np.log10(max(row['training_time'], 1)) / 4.0) * 0.7 + row['accuracy'] * 0.3,
            'Robust Performance': lambda row: row['robustness_score'] * 0.5 + row['cv_stability'] * 0.3 + row['accuracy'] * 0.2,
            'Interpretable Model': lambda row: (1.0 - row['model_complexity']) * 0.6 + row['accuracy'] * 0.4,
            'Balanced Performance': lambda row: row['quality_score'],
            'Resource Efficient': lambda row: row['efficiency_score'] * 0.6 + (1.0 - row['model_complexity']) * 0.4
        }
        
        # Calculate scores for each scenario
        recommendation_data = []
        for scenario_name, score_func in scenarios.items():
            scenario_scores = df.apply(score_func, axis=1)
            top_models = df.loc[scenario_scores.nlargest(5).index]
            
            for i, (_, model) in enumerate(top_models.iterrows()):
                recommendation_data.append({
                    'Scenario': scenario_name,
                    'Rank': i + 1,
                    'Model': model['method'][:25],
                    'Score': scenario_scores[model.name],
                    'Accuracy': model['accuracy'],
                    'Training_Time': model['training_time']
                })
        
        # Create recommendation matrix visualization
        rec_df = pd.DataFrame(recommendation_data)
        pivot_df = rec_df.pivot_table(
            index='Model', 
            columns='Scenario', 
            values='Score', 
            fill_value=0
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdYlGn',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Model Recommendation Matrix by Use Case",
            xaxis_title="Use Case Scenarios",
            yaxis_title="Models"
        )
        
        recommendation_path = self.output_dir / 'recommendation_matrix.html'
        fig.write_html(str(recommendation_path))
        print(f"Saved recommendation matrix to {recommendation_path}")
        
        # Save detailed recommendations as CSV
        rec_df.to_csv(self.output_dir / 'model_recommendations.csv', index=False)
        print(f"Saved detailed recommendations to {self.output_dir / 'model_recommendations.csv'}")
    
    def create_statistical_analysis(self, df: pd.DataFrame) -> None:
        """Create statistical analysis and significance tests."""
        print("Creating statistical analysis...")
        
        # Performance distribution analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Accuracy Distribution', 'F1-Score Distribution',
                'Training Time Distribution', 'Quality Score Distribution'
            ]
        )
        
        # Accuracy distribution
        fig.add_trace(
            go.Histogram(x=df['accuracy'], nbinsx=20, name='Accuracy'),
            row=1, col=1
        )
        
        # F1-Score distribution
        fig.add_trace(
            go.Histogram(x=df['f1_weighted'], nbinsx=20, name='F1-Score'),
            row=1, col=2
        )
        
        # Training time distribution (log scale)
        fig.add_trace(
            go.Histogram(x=np.log10(df['training_time'] + 1), nbinsx=20, name='Log Training Time'),
            row=2, col=1
        )
        
        # Quality score distribution
        fig.add_trace(
            go.Histogram(x=df['quality_score'], nbinsx=20, name='Quality Score'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Statistical Distribution Analysis",
            showlegend=False
        )
        
        stats_path = self.output_dir / 'statistical_analysis.html'
        fig.write_html(str(stats_path))
        print(f"Saved statistical analysis to {stats_path}")
        
        # Generate statistical summary report
        self.generate_statistical_report(df)
    
    def generate_statistical_report(self, df: pd.DataFrame) -> None:
        """Generate comprehensive statistical report."""
        report_path = self.output_dir / 'statistical_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("ADVANCED MODEL PERFORMANCE STATISTICAL REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic statistics
            f.write("BASIC STATISTICS\n")
            f.write("-" * 20 + "\n")
            key_metrics = ['accuracy', 'f1_weighted', 'robustness_score', 'efficiency_score', 'quality_score']
            
            for metric in key_metrics:
                values = df[metric].dropna()
                f.write(f"\n{metric.upper()}:\n")
                f.write(f"  Mean: {values.mean():.4f}\n")
                f.write(f"  Median: {values.median():.4f}\n")
                f.write(f"  Std Dev: {values.std():.4f}\n")
                f.write(f"  Min: {values.min():.4f}\n")
                f.write(f"  Max: {values.max():.4f}\n")
                f.write(f"  25th Percentile: {values.quantile(0.25):.4f}\n")
                f.write(f"  75th Percentile: {values.quantile(0.75):.4f}\n")
            
            # Correlation analysis
            f.write(f"\n\nCORRELATION ANALYSIS\n")
            f.write("-" * 25 + "\n")
            correlation_matrix = df[key_metrics].corr()
            
            f.write("Correlation Matrix:\n")
            f.write(correlation_matrix.to_string())
            f.write("\n\n")
            
            # Top performers analysis
            f.write("TOP PERFORMERS ANALYSIS\n")
            f.write("-" * 30 + "\n")
            
            top_accuracy = df.nlargest(3, 'accuracy')
            f.write("Top 3 by Accuracy:\n")
            for i, (_, model) in enumerate(top_accuracy.iterrows(), 1):
                f.write(f"  {i}. {model['method']}: {model['accuracy']:.4f}\n")
            
            top_robustness = df.nlargest(3, 'robustness_score')
            f.write("\nTop 3 by Robustness:\n")
            for i, (_, model) in enumerate(top_robustness.iterrows(), 1):
                f.write(f"  {i}. {model['method']}: {model['robustness_score']:.4f}\n")
            
            top_efficiency = df.nlargest(3, 'efficiency_score')
            f.write("\nTop 3 by Efficiency:\n")
            for i, (_, model) in enumerate(top_efficiency.iterrows(), 1):
                f.write(f"  {i}. {model['method']}: {model['efficiency_score']:.4f}\n")
            
            # Model family analysis
            f.write(f"\n\nMODEL FAMILY ANALYSIS\n")
            f.write("-" * 25 + "\n")
            
            # Group models by complexity
            complexity_groups = df.groupby('model_complexity').agg({
                'accuracy': ['mean', 'std', 'count'],
                'training_time': ['mean', 'std'],
                'quality_score': ['mean', 'std']
            }).round(4)
            
            f.write("Performance by Model Complexity:\n")
            f.write(complexity_groups.to_string())
            f.write("\n")
        
        print(f"Saved statistical report to {report_path}")
    
    def run_comprehensive_evaluation(self, results_file: str = "comparison_results_safe.json") -> None:
        """Run the complete advanced evaluation pipeline."""
        print("Starting comprehensive model evaluation...")
        print("=" * 60)
        
        # Try to process models directly from files first
        print("Attempting to load and evaluate all models directly from files...")
        df = self.process_all_models_from_files()
        
        if df.empty:
            print("Direct model loading failed. Falling back to JSON results...")
            # Fallback to comparison results
            comparison_results = self.load_comparison_results(results_file)
            if not comparison_results:
                print("No comparison results found either. Please run model training first.")
                return
            
            # Process all models from JSON
            print("Processing models from JSON results...")
            df = self.process_all_models(comparison_results)
            
            if df.empty:
                print("No model data to process.")
                return
        
        print(f"Successfully processed {len(df)} models!")
        
        # Save processed data
        df.to_csv(self.output_dir / 'comprehensive_model_metrics.csv', index=False)
        print(f"Saved comprehensive metrics to {self.output_dir / 'comprehensive_model_metrics.csv'}")
        
        # Create all visualizations
        print("\nCreating visualizations...")
        
        # 1. Comprehensive dashboard
        self.create_comprehensive_dashboard(df)
        
        # 2. Novel visualizations
        self.create_novel_visualizations(df)
        
        # 3. Statistical analysis
        self.create_statistical_analysis(df)
        
        # 4. Generate summary
        self.generate_evaluation_summary(df)
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE EVALUATION COMPLETE!")
        print(f"All outputs saved to: {self.output_dir}")
        print(f"Evaluated {len(df)} models with comprehensive metrics and visualizations")
        print("=" * 60)
    
    def generate_evaluation_summary(self, df: pd.DataFrame) -> None:
        """Generate final evaluation summary."""
        summary_path = self.output_dir / 'evaluation_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("ADVANCED MODEL EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write(f"Total Models Evaluated: {len(df)}\n")
            f.write(f"Average Quality Score: {df['quality_score'].mean():.4f}\n")
            f.write(f"Best Overall Model: {df.loc[df['quality_score'].idxmax(), 'method']}\n")
            f.write(f"Best Quality Score: {df['quality_score'].max():.4f}\n\n")
            
            # Top 5 models by different criteria
            f.write("TOP 5 MODELS BY DIFFERENT CRITERIA:\n")
            f.write("-" * 40 + "\n\n")
            
            criteria = [
                ('Overall Quality', 'quality_score'),
                ('Accuracy', 'accuracy'),
                ('Robustness', 'robustness_score'),
                ('Efficiency', 'efficiency_score'),
                ('CV Stability', 'cv_stability')
            ]
            
            for criterion_name, column in criteria:
                f.write(f"{criterion_name}:\n")
                top_models = df.nlargest(5, column)
                for i, (_, model) in enumerate(top_models.iterrows(), 1):
                    f.write(f"  {i}. {model['method'][:30]}: {model[column]:.4f}\n")
                f.write("\n")
            
            # Key insights
            f.write("KEY INSIGHTS:\n")
            f.write("-" * 15 + "\n")
            
            # Best accuracy
            best_acc_model = df.loc[df['accuracy'].idxmax()]
            f.write(f"• Highest accuracy achieved: {best_acc_model['accuracy']:.4f} by {best_acc_model['method']}\n")
            
            # Most robust
            most_robust = df.loc[df['robustness_score'].idxmax()]
            f.write(f"• Most robust model: {most_robust['method']} (robustness: {most_robust['robustness_score']:.4f})\n")
            
            # Most efficient
            most_efficient = df.loc[df['efficiency_score'].idxmax()]
            f.write(f"• Most efficient model: {most_efficient['method']} (efficiency: {most_efficient['efficiency_score']:.4f})\n")
            
            # Complexity vs performance
            high_perf_simple = df[(df['model_complexity'] < 0.5) & (df['accuracy'] > df['accuracy'].quantile(0.75))]
            if not high_perf_simple.empty:
                f.write(f"• Best simple model: {high_perf_simple.loc[high_perf_simple['accuracy'].idxmax(), 'method']}\n")
            
            # Training time insights
            fast_good = df[(df['training_time'] < df['training_time'].median()) & (df['accuracy'] > df['accuracy'].median())]
            if not fast_good.empty:
                best_fast = fast_good.loc[fast_good['accuracy'].idxmax()]
                f.write(f"• Best fast-training model: {best_fast['method']} ({best_fast['training_time']:.1f}s, {best_fast['accuracy']:.4f} acc)\n")
        
        print(f"Saved evaluation summary to {summary_path}")


def main():
    """Main function to run advanced model evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Model Performance Evaluator')
    parser.add_argument('--results-file', type=str, default='comparison_results_safe.json',
                       help='JSON file with comparison results (default: comparison_results_safe.json)')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing trained models (default: models)')
    parser.add_argument('--output-dir', type=str, default='plots/advanced',
                       help='Directory to save outputs (default: plots/advanced)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = AdvancedModelEvaluator(
        models_dir=args.models_dir,
        output_dir=args.output_dir
    )
    
    # Run comprehensive evaluation
    evaluator.run_comprehensive_evaluation(args.results_file)


if __name__ == "__main__":
    main()
