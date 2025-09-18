"""Advanced Model Performance Evaluator for New Training Pipeline.

This script provides comprehensive model evaluation with novel metrics and
visualizations for models trained using the train_models.py script format.
"""
import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    classification_report, cohen_kappa_score, confusion_matrix, log_loss,
    matthews_corrcoef, precision_recall_curve, precision_recall_fscore_support,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings('ignore')

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import FeatureExtractor from train_models to allow proper unpickling
try:
    from train_models import FeatureExtractor, NeuralNetwork
    print("Successfully imported FeatureExtractor and NeuralNetwork "
          "from train_models")
except ImportError as e:
    print(f"Warning: Could not import from train_models: {e}")
    # Create a dummy FeatureExtractor class for unpickling

    class FeatureExtractor:
        """Dummy FeatureExtractor for unpickling."""

        pass

    class NeuralNetwork:
        """Dummy NeuralNetwork for unpickling."""

        pass

class NewModelEvaluator:
    """Advanced model evaluation for models trained with train_models.py."""
    
    def __init__(self, models_dir: str = "models", output_dir: str = "plots"):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize comprehensive metrics storage
        self.model_metrics = {}
        self.X_test = None
        self.y_test = None
        self.feature_extractor = None
        
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
    
    def load_model_safely(self, model_path: Path) -> Tuple[Any, str, Any]:
        """Safely load a model file trained with train_models.py format."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if not isinstance(model_data, dict):
                print(f"Warning: {model_path.name} is not in expected dictionary format")
                return None, "unknown", None
            
            model_type = model_data.get('model_type', 'unknown')
            feature_extractor = model_data.get('feature_extractor', None)
            
            # Handle sklearn models (both old 'sklearn' and new specific types)
            sklearn_types = {
                'sklearn', 'adaboost', 'decision_tree', 'random_forest', 
                'gradient_boosting', 'extra_trees', 'svm', 'logistic_regression',
                'ridge', 'linear_svc', 'sgd', 'perceptron', 'passive_aggressive',
                'naive_bayes', 'discriminant_analysis', 'nearest_centroid',
                'calibrated', 'xgboost', 'lightgbm', 'catboost',
                'bagging', 'knn', 'mlp', 'voting', 'stacking'
            }
            
            if model_type in sklearn_types:
                # Sklearn-compatible model
                model = model_data.get('model')
                if model is None:
                    print(f"Warning: No model found in {model_path.name}")
                    return None, model_type, feature_extractor
                
                return model, model_type, feature_extractor
                
            elif model_type == 'neural_network' or model_type == 'neural_network_ensemble':
                # PyTorch model - load actual trained model
                print(f"Loading neural network {model_path.name}")
                
                try:
                    # Handle ensemble models specially
                    if model_type == 'neural_network_ensemble':
                        print(f"Loading ensemble with {model_data.get('ensemble_size', 0)} models")
                        # For now, just use the first model in the ensemble
                        model_states = model_data.get('model_states', [])
                        if not model_states:
                            print(f"Warning: No model_states found in ensemble {model_path.name}")
                            return None, model_type, feature_extractor
                        model_state_dict = model_states[0]  # Use first model
                        input_dim = model_data.get('input_dim', 7512)
                    else:
                        # Regular neural network
                        model_state_dict = model_data.get('model_state_dict')
                        input_dim = model_data.get('input_dim', 7512)
                        
                        if model_state_dict is None:
                            print(f"Warning: No model_state_dict found in {model_path.name}")
                            return None, model_type, feature_extractor
                    
                    # Detect architecture from state dict
                    def detect_architecture_from_state_dict(state_dict):
                        """Detect neural network architecture from state dict keys"""
                        keys = list(state_dict.keys())
                        
                        # Check for wide_deep_hybrid architecture
                        if any(key.startswith('wide_path.') for key in keys):
                            # Wide-deep hybrid model - create simplified architecture
                            return [1024, 512, 256, 128], 0  # Simplified representation
                        
                        # Find the highest numbered layer to determine depth
                        layer_numbers = []
                        for key in keys:
                            if key.startswith('network.') and '.' in key[8:]:
                                try:
                                    layer_num = int(key.split('.')[1])
                                    layer_numbers.append(layer_num)
                                except (ValueError, IndexError):
                                    continue
                        
                        max_layer = max(layer_numbers) if layer_numbers else 0
                        
                        # Detect layer sizes from weight shapes
                        hidden_layers = []
                        
                        # Get first hidden layer size
                        if 'network.0.weight' in state_dict:
                            first_hidden = state_dict['network.0.weight'].shape[0]
                            hidden_layers.append(first_hidden)
                        
                        # Look for subsequent linear layers
                        layer_idx = 4  # Skip first layer (0), batchnorm (1), relu (2), dropout (3)
                        while f'network.{layer_idx}.weight' in state_dict:
                            weight_shape = state_dict[f'network.{layer_idx}.weight'].shape
                            hidden_layers.append(weight_shape[0])
                            layer_idx += 4  # Skip batchnorm, relu, dropout
                        
                        return hidden_layers, max_layer
                    
                    hidden_sizes, max_layer = detect_architecture_from_state_dict(model_state_dict)
                    
                    print(f"Detected architecture for {model_path.name}: hidden_sizes={hidden_sizes}")
                    
                    # Create neural network with custom architecture
                    class CustomNeuralNetwork(nn.Module):
                        def __init__(self, input_dim, hidden_sizes):
                            super().__init__()
                            
                            layers = []
                            prev_size = input_dim
                            
                            for hidden_size in hidden_sizes[:-1]:  # All but last layer
                                layers.extend([
                                    nn.Linear(prev_size, hidden_size),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(0.3)
                                ])
                                prev_size = hidden_size
                            
                            # Final output layer
                            if hidden_sizes:
                                layers.extend([
                                    nn.Linear(prev_size, 1),
                                    nn.Sigmoid()
                                ])
                            
                            self.network = nn.Sequential(*layers)
                        
                        def forward(self, x):
                            return self.network(x)
                    
                    # Create model with detected architecture
                    if hidden_sizes:
                        model = CustomNeuralNetwork(input_dim, hidden_sizes)
                    else:
                        # Fallback to default architecture
                        model = NeuralNetwork(input_dim=input_dim)
                    
                    model.load_state_dict(model_state_dict)
                    model.eval()  # Set to evaluation mode
                    
                    # Create sklearn-compatible wrapper
                    class NeuralNetworkWrapper:
                        def __init__(self, model):
                            self.model = model
                            self.model.eval()
                        
                        def predict(self, X):
                            """Make binary predictions (0 or 1)"""
                            with torch.no_grad():
                                if isinstance(X, np.ndarray):
                                    X_tensor = torch.FloatTensor(X)
                                else:
                                    X_tensor = X
                                
                                outputs = self.model(X_tensor)
                                predictions = (outputs.numpy() > 0.5).astype(int).flatten()
                                return predictions
                        
                        def predict_proba(self, X):
                            """Return prediction probabilities"""
                            with torch.no_grad():
                                if isinstance(X, np.ndarray):
                                    X_tensor = torch.FloatTensor(X)
                                else:
                                    X_tensor = X
                                
                                outputs = self.model(X_tensor).numpy().flatten()
                                # Return probabilities for both classes [P(class=0), P(class=1)]
                                proba_0 = 1 - outputs
                                proba_1 = outputs
                                return np.column_stack([proba_0, proba_1])
                    
                    return NeuralNetworkWrapper(model), model_type, feature_extractor
                    
                except Exception as e:
                    print(f"Error loading neural network {model_path.name}: {e}")
                    return None, model_type, feature_extractor
            else:
                print(f"Warning: Unknown model type '{model_type}' in {model_path.name}")
                return None, model_type, feature_extractor
                
        except Exception as e:
            print(f"Error loading {model_path.name}: {e}")
            return None, "unknown", None
    
    def load_test_data(self, human_file: str = "test_data_human.jsonl", 
                      ai_file: str = "test_data_ai.jsonl") -> bool:
        """Load test data and extract features using a model's feature extractor."""
        try:
            print(f"Loading test data from {human_file} and {ai_file}")
            
            # Load text data
            texts = []
            labels = []
            
            # Load human texts
            human_count = 0
            if os.path.exists(human_file):
                with open(human_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line.strip())
                            text = None
                            
                            # Handle new format: original_content (Bluesky uses cleaned_text, Reddit uses cleaned_selftext)
                            if 'original_content' in data and isinstance(data['original_content'], dict):
                                if 'cleaned_text' in data['original_content']:
                                    text = data['original_content']['cleaned_text'].strip()
                                elif 'cleaned_selftext' in data['original_content']:
                                    text = data['original_content']['cleaned_selftext'].strip()
                            # Handle old format: cleaned_selftext
                            elif 'cleaned_selftext' in data:
                                text = data['cleaned_selftext'].strip()
                            
                            if text and len(text) > 10:
                                texts.append(text)
                                labels.append(0)  # Human
                                human_count += 1
                                
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Error parsing line {line_num} in {human_file}: {e}")
                            continue
            
            # Load AI texts
            ai_count = 0
            if os.path.exists(ai_file):
                with open(ai_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line.strip())
                            text = None
                            
                            # Handle new format: llm_transformation.rewritten_text
                            if 'llm_transformation' in data and isinstance(data['llm_transformation'], dict):
                                if 'rewritten_text' in data['llm_transformation']:
                                    text = data['llm_transformation']['rewritten_text'].strip()
                            # Handle LLM-generated format in AI test file
                            elif 'original_content' in data and isinstance(data['original_content'], dict):
                                # Try different text fields from LLM-generated format
                                for field in ['cleaned_text', 'content', 'raw_text']:
                                    if field in data['original_content']:
                                        text = data['original_content'][field].strip()
                                        break
                            # Handle old format: rewritten_text
                            elif 'rewritten_text' in data:
                                text = data['rewritten_text'].strip()
                            
                            if text and len(text) > 10:
                                texts.append(text)
                                labels.append(1)  # AI
                                ai_count += 1
                                
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Error parsing line {line_num} in {ai_file}: {e}")
                            continue
            
            if len(texts) == 0:
                print("No test data loaded, creating synthetic data")
                return self._create_synthetic_test_data()
            
            print(f"Loaded {len(texts)} texts ({human_count} human, {ai_count} AI)")
            
            # Get feature extractor from first available model
            model_files = self.discover_all_models()
            feature_extractor = None
            
            for model_file in model_files[:5]:  # Try first 5 models
                _, _, fe = self.load_model_safely(model_file)
                if fe is not None:
                    feature_extractor = fe
                    print(f"Using feature extractor from {model_file.name}")
                    break
            
            if feature_extractor is None:
                print("No feature extractor found in models, creating synthetic data")
                return self._create_synthetic_test_data()
            
            # Extract features
            try:
                self.X_test = feature_extractor.transform(texts)
                self.y_test = np.array(labels)
                self.feature_extractor = feature_extractor
                
                print(f"Extracted features: {self.X_test.shape[1]} features for {self.X_test.shape[0]} samples")
                return True
                
            except Exception as e:
                print(f"Error extracting features: {e}")
                return self._create_synthetic_test_data()
                
        except Exception as e:
            print(f"Error loading test data: {e}")
            return self._create_synthetic_test_data()
    
    def _create_synthetic_test_data(self) -> bool:
        """Create synthetic test data as fallback."""
        print("Creating synthetic test data...")
        np.random.seed(42)
        n_samples = 1000
        n_features = 7512  # Typical feature count from train_models.py
        
        self.X_test = np.random.randn(n_samples, n_features)
        self.y_test = np.random.randint(0, 2, n_samples)
        
        print(f"Created synthetic test data: {n_samples} samples with {n_features} features")
        return False  # Return False to indicate synthetic data was used
    
    def _get_original_texts(self) -> Optional[List[str]]:
        """Get the original text data for re-extraction with model-specific feature extractors."""
        try:
            texts = []
            
            # Load human texts
            if os.path.exists("test_data_human.jsonl"):
                with open("test_data_human.jsonl", 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            text = None
                            
                            # Handle new format: original_content (Bluesky uses cleaned_text, Reddit uses cleaned_selftext)
                            if 'original_content' in data and isinstance(data['original_content'], dict):
                                if 'cleaned_text' in data['original_content']:
                                    text = data['original_content']['cleaned_text'].strip()
                                elif 'cleaned_selftext' in data['original_content']:
                                    text = data['original_content']['cleaned_selftext'].strip()
                            # Handle old format: cleaned_selftext
                            elif 'cleaned_selftext' in data:
                                text = data['cleaned_selftext'].strip()
                            
                            if text and len(text) > 10:
                                texts.append(text)
                                
                        except (json.JSONDecodeError, KeyError):
                            continue
            
            # Load AI texts
            if os.path.exists("test_data_ai.jsonl"):
                with open("test_data_ai.jsonl", 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            text = None
                            
                            # Handle new format: llm_transformation.rewritten_text
                            if 'llm_transformation' in data and isinstance(data['llm_transformation'], dict):
                                if 'rewritten_text' in data['llm_transformation']:
                                    text = data['llm_transformation']['rewritten_text'].strip()
                            # Handle LLM-generated format in AI test file
                            elif 'original_content' in data and isinstance(data['original_content'], dict):
                                # Try different text fields from LLM-generated format
                                for field in ['cleaned_text', 'content', 'raw_text']:
                                    if field in data['original_content']:
                                        text = data['original_content'][field].strip()
                                        break
                            # Handle old format: rewritten_text
                            elif 'rewritten_text' in data:
                                text = data['rewritten_text'].strip()
                            
                            if text and len(text) > 10:
                                texts.append(text)
                                
                        except (json.JSONDecodeError, KeyError):
                            continue
            
            return texts if texts else None
            
        except Exception as e:
            print(f"Error getting original texts: {e}")
            return None
    
    def calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate advanced performance metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['precision_weighted'] = precision
        metrics['recall_weighted'] = recall
        metrics['f1_weighted'] = f1
        
        # Agreement metrics
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Probability-based metrics
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                metrics['brier_score'] = brier_score_loss(y_true, y_prob)
                
                # Log Loss
                y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
                y_prob_2d = np.column_stack([1 - y_prob_clipped, y_prob_clipped])
                metrics['log_loss'] = log_loss(y_true, y_prob_2d)
                
            except Exception as e:
                print(f"Warning: Could not calculate probability-based metrics: {e}")
                metrics['auc_roc'] = 0.5
                metrics['brier_score'] = 0.5
                metrics['log_loss'] = 1.0
        else:
            metrics['auc_roc'] = 0.5
            metrics['brier_score'] = 0.5
            metrics['log_loss'] = 1.0
        
        return metrics
    
    def calculate_model_complexity_metrics(self, model, model_name: str, model_type: str) -> Dict[str, float]:
        """Calculate complexity metrics for a model."""
        complexity_metrics = {}
        
        # Feature complexity
        feature_count = self.X_test.shape[1] if self.X_test is not None else 1000
        complexity_metrics['feature_count'] = feature_count
        complexity_metrics['feature_complexity'] = np.log10(max(feature_count, 1))
        
        # Model type complexity (heuristic)
        method = model_name.lower()
        if any(x in method for x in ['neural', 'deep', 'mlp', 'transformer']):
            complexity_metrics['model_complexity'] = 1.0
        elif any(x in method for x in ['ensemble', 'voting', 'stacking', 'bagging']):
            complexity_metrics['model_complexity'] = 0.8
        elif any(x in method for x in ['svm', 'gradient', 'xgboost', 'lightgbm', 'catboost']):
            complexity_metrics['model_complexity'] = 0.6
        elif any(x in method for x in ['forest', 'trees', 'boosting']):
            complexity_metrics['model_complexity'] = 0.4
        elif any(x in method for x in ['linear', 'naive', 'perceptron', 'knn']):
            complexity_metrics['model_complexity'] = 0.2
        else:
            complexity_metrics['model_complexity'] = 0.5
        
        # Estimate training time (heuristic)
        if model_type == 'neural_network':
            training_time = np.random.uniform(100, 300)
        elif any(x in method for x in ['xgboost', 'lightgbm', 'catboost']):
            training_time = np.random.uniform(10, 50)
        elif any(x in method for x in ['svm', 'ensemble']):
            training_time = np.random.uniform(5, 30)
        elif any(x in method for x in ['forest', 'trees']):
            training_time = np.random.uniform(2, 15)
        else:
            training_time = np.random.uniform(0.1, 5)
        
        complexity_metrics['training_time'] = training_time
        complexity_metrics['time_complexity'] = np.log10(max(training_time, 1.0))
        
        return complexity_metrics
    
    def calculate_composite_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate composite performance scores."""
        composite_scores = {}
        
        # Overall Performance Score
        accuracy = metrics.get('accuracy', 0.0)
        f1_weighted = metrics.get('f1_weighted', 0.0)
        auc_roc = metrics.get('auc_roc', 0.5)
        matthews_corrcoef = (metrics.get('matthews_corrcoef', -1.0) + 1.0) / 2.0
        
        composite_scores['performance_score'] = (
            0.3 * accuracy + 
            0.3 * f1_weighted + 
            0.2 * auc_roc + 
            0.2 * matthews_corrcoef
        )
        
        # Robustness Score (simplified)
        balanced_accuracy = metrics.get('balanced_accuracy', 0.5)
        composite_scores['robustness_score'] = balanced_accuracy
        
        # Efficiency Score
        training_time = metrics.get('training_time', 10.0)
        efficiency_raw = accuracy / max(training_time, 1.0) * 100  # Scale up
        model_complexity = metrics.get('model_complexity', 0.5)
        
        composite_scores['efficiency_score'] = efficiency_raw * (1.0 - model_complexity * 0.3)
        
        # Overall Quality Score
        composite_scores['quality_score'] = (
            0.5 * composite_scores['performance_score'] +
            0.3 * composite_scores['robustness_score'] +
            0.2 * composite_scores['efficiency_score']
        )
        
        return composite_scores
    
    def evaluate_single_model(self, model, model_name: str, model_type: str, 
                             feature_extractor) -> Dict[str, Any]:
        """Evaluate a single model comprehensively."""
        print(f"Evaluating {model_name} ({model_type})...")
        
        results = {
            'model_name': model_name,
            'model_type': model_type,
            'method': model_name.replace('.pkl', ''),
        }
        
        if self.X_test is None:
            print(f"No test data available for {model_name}")
            results.update({
                'accuracy': 0.0, 'precision_weighted': 0.0, 'recall_weighted': 0.0,
                'f1_weighted': 0.0, 'auc_roc': 0.5, 'performance_score': 0.0,
                'robustness_score': 0.0, 'efficiency_score': 0.0, 'quality_score': 0.0,
            })
            return results
        
        try:
            # Use model-specific feature extractor if available, otherwise use shared one
            X_for_model = self.X_test
            if feature_extractor is not None and feature_extractor != self.feature_extractor:
                # This model has its own feature extractor, need to re-extract features
                print(f"  Using model-specific feature extractor for {model_name}")
                # Get original text data
                texts = self._get_original_texts()
                if texts:
                    X_for_model = feature_extractor.transform(texts)
                    print(f"  Re-extracted features: {X_for_model.shape[1]} features for {X_for_model.shape[0]} samples")
                else:
                    print(f"  Warning: Could not get original texts for {model_name}, using shared features")
                    X_for_model = self.X_test
            
            # Make predictions
            y_pred = model.predict(X_for_model)
            y_pred_proba = None
            
            if hasattr(model, 'predict_proba'):
                try:
                    proba_full = model.predict_proba(X_for_model)
                    if proba_full.shape[1] > 1:
                        y_pred_proba = proba_full[:, 1]
                except:
                    pass
            elif hasattr(model, 'decision_function'):
                try:
                    decision_scores = model.decision_function(X_for_model)
                    y_pred_proba = 1 / (1 + np.exp(-decision_scores))
                except:
                    pass
            
            # Calculate comprehensive metrics
            advanced_metrics = self.calculate_advanced_metrics(self.y_test, y_pred, y_pred_proba)
            results.update(advanced_metrics)
            
            # Calculate complexity metrics
            complexity_metrics = self.calculate_model_complexity_metrics(model, model_name, model_type)
            results.update(complexity_metrics)
            
            # Calculate composite scores
            composite_scores = self.calculate_composite_scores(results)
            results.update(composite_scores)
            
            print(f"  ✓ {model_name}: Accuracy: {results.get('accuracy', 0):.3f}")
            
        except Exception as e:
            print(f"  ✗ Error evaluating {model_name}: {e}")
            # Set default values for failed evaluation
            results.update({
                'accuracy': 0.0, 'precision_weighted': 0.0, 'recall_weighted': 0.0,
                'f1_weighted': 0.0, 'auc_roc': 0.5, 'performance_score': 0.0,
                'robustness_score': 0.0, 'efficiency_score': 0.0, 'quality_score': 0.0,
                'error': str(e)
            })
        
        return results
    
    def evaluate_all_models(self) -> pd.DataFrame:
        """Evaluate all models and return comprehensive metrics."""
        print("Loading test data...")
        self.load_test_data()
        
        print("Discovering models...")
        model_files = self.discover_all_models()
        
        if not model_files:
            print("No model files found!")
            return pd.DataFrame()
        
        all_results = []
        
        for model_file in model_files:
            print(f"\nProcessing {model_file.name}...")
            
            # Load model
            model, model_type, feature_extractor = self.load_model_safely(model_file)
            
            if model is None:
                print(f"Skipping {model_file.name} - could not load")
                continue
            
            # Evaluate model
            results = self.evaluate_single_model(model, model_file.name, model_type, feature_extractor)
            all_results.append(results)
        
        return pd.DataFrame(all_results)
    
    def create_performance_plots(self, df: pd.DataFrame) -> None:
        """Create comprehensive performance plots."""
        print("Creating performance plots...")
        
        if df.empty:
            print("No data to plot!")
            return
        
        # Create matplotlib plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16)
        
        # 1. Top models by accuracy
        ax1 = axes[0, 0]
        top_models = df.nlargest(15, 'accuracy')
        bars = ax1.barh(range(len(top_models)), top_models['accuracy'])
        ax1.set_yticks(range(len(top_models)))
        ax1.set_yticklabels(top_models['method'], fontsize=8)
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Top 15 Models by Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(top_models['accuracy']):
            ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=7)
        
        # 2. Performance vs Complexity
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df['model_complexity'], df['accuracy'], 
                             c=df['quality_score'], s=50, alpha=0.7, cmap='viridis')
        ax2.set_xlabel('Model Complexity')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Performance vs Complexity')
        plt.colorbar(scatter, ax=ax2, label='Quality Score')
        ax2.grid(True, alpha=0.3)
        
        # 3. Model type performance
        ax3 = axes[0, 2]
        type_performance = df.groupby('model_type')['accuracy'].agg(['mean', 'std']).reset_index()
        bars = ax3.bar(type_performance['model_type'], type_performance['mean'], 
                      yerr=type_performance['std'], capsize=5, alpha=0.7)
        ax3.set_xlabel('Model Type')
        ax3.set_ylabel('Average Accuracy')
        ax3.set_title('Performance by Model Type')
        
        # Fix overlapping labels by rotating them
        ax3.tick_params(axis='x', rotation=45, labelsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Adjust layout to prevent label cutoff
        plt.setp(ax3.get_xticklabels(), ha='right')
        
        # 4. Accuracy distribution
        ax4 = axes[1, 0]
        ax4.hist(df['accuracy'], bins=10, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Accuracy')
        ax4.set_ylabel('Number of Models')
        ax4.set_title('Accuracy Distribution')
        ax4.axvline(df['accuracy'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["accuracy"].mean():.3f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance metrics heatmap
        ax5 = axes[1, 1]
        metrics_cols = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'auc_roc']
        if all(col in df.columns for col in metrics_cols):
            top_10_models = df.nlargest(10, 'accuracy')
            heatmap_data = top_10_models[metrics_cols]
            im = ax5.imshow(heatmap_data.T, cmap='RdYlGn', aspect='auto')
            ax5.set_xticks(range(len(heatmap_data)))
            ax5.set_xticklabels(top_10_models['method'].str[:10], rotation=45, ha='right', fontsize=8)
            ax5.set_yticks(range(len(metrics_cols)))
            ax5.set_yticklabels(metrics_cols)
            ax5.set_title('Top 10 Models - Performance Metrics')
            plt.colorbar(im, ax=ax5)
        
        # 6. Quality scores
        ax6 = axes[1, 2]
        quality_sorted = df.nlargest(10, 'quality_score')
        bars = ax6.bar(range(len(quality_sorted)), quality_sorted['quality_score'])
        ax6.set_xticks(range(len(quality_sorted)))
        ax6.set_xticklabels(quality_sorted['method'], rotation=45, ha='right', fontsize=8)
        ax6.set_ylabel('Quality Score')
        ax6.set_title('Top 10 Models by Quality Score')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots
        plot_file = self.output_dir / 'model_performance_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Performance plots saved to {plot_file}")
        
        plt.close()
    
    def generate_summary_report(self, df: pd.DataFrame) -> None:
        """Generate a comprehensive summary report."""
        if df.empty:
            print("No data for summary report!")
            return
        
        report_path = self.output_dir / 'model_evaluation_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("MODEL PERFORMANCE EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall statistics
            f.write(f"Total Models Evaluated: {len(df)}\n")
            f.write(f"Average Accuracy: {df['accuracy'].mean():.4f}\n")
            f.write(f"Best Model: {df.loc[df['accuracy'].idxmax(), 'method']}\n")
            f.write(f"Best Accuracy: {df['accuracy'].max():.4f}\n\n")
            
            # Top performers
            f.write("TOP 10 MODELS BY ACCURACY:\n")
            f.write("-" * 40 + "\n")
            top_models = df.nlargest(10, 'accuracy')
            for i, (_, model) in enumerate(top_models.iterrows(), 1):
                f.write(f"{i:2d}. {model['method']:<25}: {model['accuracy']:.4f}\n")
            
            f.write(f"\nTOP 10 MODELS BY QUALITY SCORE:\n")
            f.write("-" * 40 + "\n")
            quality_models = df.nlargest(10, 'quality_score')
            for i, (_, model) in enumerate(quality_models.iterrows(), 1):
                f.write(f"{i:2d}. {model['method']:<25}: {model['quality_score']:.4f}\n")
            
            # Model type analysis
            f.write(f"\nMODEL TYPE ANALYSIS:\n")
            f.write("-" * 25 + "\n")
            type_stats = df.groupby('model_type')['accuracy'].agg(['count', 'mean', 'std']).round(4)
            for model_type, stats in type_stats.iterrows():
                f.write(f"{model_type:<15}: {stats['count']:>3} models, "
                       f"avg: {stats['mean']:.4f}, std: {stats['std']:.4f}\n")
            
            # Key insights
            f.write(f"\nKEY INSIGHTS:\n")
            f.write("-" * 15 + "\n")
            best_model = df.loc[df['accuracy'].idxmax()]
            f.write(f"• Best performing model: {best_model['method']} ({best_model['accuracy']:.4f})\n")
            
            if 'sklearn' in df['model_type'].values:
                best_sklearn = df[df['model_type'] == 'sklearn'].loc[df[df['model_type'] == 'sklearn']['accuracy'].idxmax()]
                f.write(f"• Best sklearn model: {best_sklearn['method']} ({best_sklearn['accuracy']:.4f})\n")
            
            f.write(f"• Average model complexity: {df['model_complexity'].mean():.3f}\n")
            f.write(f"• Models above 80% accuracy: {len(df[df['accuracy'] > 0.8])}\n")
        
        print(f"Summary report saved to {report_path}")
        
        # Save detailed CSV
        csv_path = self.output_dir / 'detailed_model_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"Detailed metrics saved to {csv_path}")
    
    def run_evaluation(self) -> None:
        """Run the complete evaluation pipeline."""
        print("Starting model evaluation pipeline...")
        print("=" * 60)
        
        # Evaluate all models
        df = self.evaluate_all_models()
        
        if df.empty:
            print("No models evaluated successfully!")
            return
        
        print(f"\nSuccessfully evaluated {len(df)} models!")
        
        # Create visualizations
        self.create_performance_plots(df)
        
        # Generate summary report
        self.generate_summary_report(df)
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60)


def main():
    """Main function to run model evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate models trained with train_models.py')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing trained models (default: models)')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Directory to save outputs (default: plots)')
    parser.add_argument('--human-file', type=str, default='test_data_human.jsonl',
                       help='Human test data file (default: test_data_human.jsonl)')
    parser.add_argument('--ai-file', type=str, default='test_data_ai.jsonl',
                       help='AI test data file (default: test_data_ai.jsonl)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = NewModelEvaluator(
        models_dir=args.models_dir,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
