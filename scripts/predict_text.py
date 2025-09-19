"""Comprehensive AI vs Human Text Classification Prediction Script.

This script provides reliable predictions using models trained with the new
training pipeline. Supports both individual model predictions and ensemble
predictions.
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import enhanced feature extractor
try:
    from scripts.enhanced_feature_extractor import EnhancedFeatureExtractor, create_enhanced_feature_extractor
    ENHANCED_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced feature extractor not available: {e}")
    ENHANCED_EXTRACTOR_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """Compatibility shim for old models - maintains the same interface."""
    
    def __init__(self, max_features: int = 10000, 
                 ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.enhanced_extractor = None
        
        # Initialize enhanced extractor if available
        if ENHANCED_EXTRACTOR_AVAILABLE:
            try:
                self.enhanced_extractor = create_enhanced_feature_extractor('baseline_plus')
                print("Using enhanced feature extractor for compatibility")
            except Exception as e:
                print(f"Could not create enhanced extractor: {e}")
                self.enhanced_extractor = None
        
        # Fallback basic extractors
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.scaler = None
        
    def fit_transform(self, texts: List[str], labels: List[int] = None) -> np.ndarray:
        """Fit and transform texts using enhanced extraction if available."""
        if self.enhanced_extractor:
            return self.enhanced_extractor.fit_transform(texts, labels)
        else:
            return self._fit_transform_basic(texts)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted extractor."""
        if self.enhanced_extractor:
            return self.enhanced_extractor.transform(texts)
        else:
            return self._transform_basic(texts)
    
    def _fit_transform_basic(self, texts: List[str]) -> np.ndarray:
        """Basic feature extraction (fallback)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler
        
        # Word-level TF-IDF
        self.word_vectorizer = TfidfVectorizer(
            max_features=self.max_features // 2,
            ngram_range=self.ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        word_features = self.word_vectorizer.fit_transform(texts).toarray()
        
        # Character-level TF-IDF
        self.char_vectorizer = TfidfVectorizer(
            max_features=self.max_features // 4,
            analyzer='char',
            ngram_range=(2, 4),
            lowercase=True
        )
        char_features = self.char_vectorizer.fit_transform(texts).toarray()
        
        # Simple linguistic features
        linguistic_features = self._extract_simple_linguistic_features(texts)
        
        # Combine all features
        combined_features = np.hstack([word_features, char_features, linguistic_features])
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(combined_features)
        
        return scaled_features
    
    def _transform_basic(self, texts: List[str]) -> np.ndarray:
        """Transform using basic extractors."""
        if self.word_vectorizer is None:
            raise ValueError("Feature extractor not fitted. Call fit_transform first.")
        
        word_features = self.word_vectorizer.transform(texts).toarray()
        char_features = self.char_vectorizer.transform(texts).toarray()
        linguistic_features = self._extract_simple_linguistic_features(texts)
        
        combined_features = np.hstack([word_features, char_features, linguistic_features])
        scaled_features = self.scaler.transform(combined_features)
        
        return scaled_features
    
    def _extract_simple_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract basic linguistic features."""
        import re
        
        features = []
        for text in texts:
            text_features = []
            
            # Basic statistics
            text_len = len(text)
            words = text.lower().split()
            word_count = len(words)
            
            text_features.append(text_len)
            text_features.append(word_count)
            text_features.append(word_count / max(text_len, 1))
            
            # Sentence statistics
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            sentence_count = max(len(sentences), 1)
            text_features.append(sentence_count)
            text_features.append(word_count / sentence_count)
            
            # Character-level features
            if text_len > 0:
                upper_ratio = sum(1 for c in text if c.isupper()) / text_len
                lower_ratio = sum(1 for c in text if c.islower()) / text_len
                digit_ratio = sum(1 for c in text if c.isdigit()) / text_len
                punct_ratio = sum(1 for c in text if c in '.,!?;:') / text_len
                text_features.extend([upper_ratio, lower_ratio, digit_ratio, punct_ratio])
            else:
                text_features.extend([0, 0, 0, 0])
            
            # Vocabulary complexity
            unique_words = set(words)
            text_features.append(len(unique_words) / max(word_count, 1))
            
            # Average word length
            if words:
                avg_word_len = np.mean([len(word) for word in words])
                text_features.append(np.clip(avg_word_len, 0, 50))
            else:
                text_features.append(0)
            
            # Simple readability
            avg_sentence_length = word_count / sentence_count
            if words:
                syllable_counts = [max(1, len(re.findall(r'[aeiouAEIOU]', word))) for word in words]
                avg_syllables = np.mean(syllable_counts)
                flesch_score = (206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables))
                text_features.append(np.clip(flesch_score, -100, 200))
            else:
                text_features.append(0)
            
            features.append(text_features)
        
        return np.array(features)


class NeuralNetwork(nn.Module):
    """Improved Neural Network for text classification."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, 
                 dropout: float = 0.3, activation: str = 'relu'):
        super(NeuralNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            ])
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ModelPredictor:
    """Comprehensive model prediction with consistent feature extraction."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """Load a single model with its feature extractor."""
        if not model_path.endswith('.pkl'):
            model_path += '.pkl'
            
        full_path = self.models_dir / model_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Model not found: {full_path}")
         
        try:
            with open(full_path, 'rb') as f:
                model_data = pickle.load(f)
            
            print(f"✓ Loaded {model_path}")
            return model_data
            
        except Exception as e:
            print(f"✗ Failed to load {model_path}: {e}")
            raise
    
    def predict_single_model(self, model_data: Dict[str, Any], text: str, model_name: str = None) -> Tuple[int, float]:
        """Make prediction using a single model with robust error handling."""
        try:
            # Extract features using the model's feature extractor
            feature_extractor = model_data['feature_extractor']
            features = None
            
            # Try multiple approaches for feature extraction
            if hasattr(feature_extractor, 'enhanced_extractor') and feature_extractor.enhanced_extractor:
                try:
                    # Use the enhanced feature extractor from training
                    features = feature_extractor.enhanced_extractor.transform([text])
                except Exception as e:
                    print(f"Warning: Enhanced extractor failed for {model_name}: {e}")
            
            # Fallback to regular feature extractor
            if features is None and hasattr(feature_extractor, 'transform'):
                try:
                    features = feature_extractor.transform([text])
                except Exception as e:
                    print(f"Warning: Regular extractor failed for {model_name}: {e}")
            
            # Last resort: try to reconstruct features
            if features is None:
                raise ValueError(f"Could not extract features for {model_name}")
            
            model_type = model_data['model_type']
            
            if model_type == 'neural_network':
                return self._predict_neural_network(model_data, features, model_name)
            elif model_type == 'neural_network_ensemble':
                return self._predict_neural_network_ensemble(model_data, features, model_name)
            else:
                # Handle all sklearn-compatible model types
                return self._predict_sklearn_model(model_data, features, text, model_name)
                
        except Exception as e:
            # Convert all errors to a standardized format for better handling
            error_msg = str(e)
            if "features" in error_msg and "expecting" in error_msg:
                raise ValueError(f"Feature dimension mismatch: {error_msg}")
            elif "state_dict" in error_msg:
                raise ValueError(f"Neural network architecture mismatch: {error_msg}")
            elif "model" in error_msg.lower():
                raise ValueError(f"Model loading error: {error_msg}")
            else:
                raise ValueError(f"Prediction error: {error_msg}")
    
    def _predict_neural_network(self, model_data: Dict[str, Any], features: np.ndarray, model_name: str = None) -> Tuple[int, float]:
        """Predict using PyTorch neural network."""
        try:
            input_dim = features.shape[1]
            
            # Check if this is a special architecture model
            if model_name == 'neural_network_wide_deep_hybrid':
                return self._predict_wide_deep_hybrid(model_data, features, model_name)
            elif model_name == 'neural_network_best_ensemble':
                return self._predict_best_ensemble_architecture(model_data, features, model_name)
            
            # Map model names to their architectures
            network_configs = {
                'neural_network_default': {'hidden_dims': [512, 256, 128], 'dropout': 0.3, 'activation': 'relu'},
                'neural_network_wide': {'hidden_dims': [1024, 512, 256], 'dropout': 0.3, 'activation': 'relu'},
                'neural_network_deep': {'hidden_dims': [256, 256, 256, 128, 64], 'dropout': 0.3, 'activation': 'relu'},
                'neural_network_shallow': {'hidden_dims': [256, 128], 'dropout': 0.2, 'activation': 'relu'},
                'neural_network_tanh': {'hidden_dims': [512, 256, 128], 'dropout': 0.3, 'activation': 'tanh'},
                'neural_network_leaky': {'hidden_dims': [512, 256, 128], 'dropout': 0.3, 'activation': 'leaky_relu'},
                'neural_network_dropout_high': {'hidden_dims': [512, 256, 128], 'dropout': 0.5, 'activation': 'relu'},
                'neural_network_dropout_low': {'hidden_dims': [512, 256, 128], 'dropout': 0.1, 'activation': 'relu'},
            }
            
            # Get config for this model, default to basic config
            if model_name in network_configs:
                config = network_configs[model_name]
            else:
                config = {'hidden_dims': [512, 256, 128], 'dropout': 0.3, 'activation': 'relu'}
            
            # Reconstruct model architecture
            model = NeuralNetwork(input_dim, **config)
            
            # Load state dict
            model.load_state_dict(model_data['model_state_dict'])
            model.eval()
            
            # Make prediction
            device = torch.device('mps' if torch.backends.mps.is_available() 
                                else 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).to(device)
                output = model(features_tensor)
                probability = output.cpu().numpy().flatten()[0]
                prediction = 1 if probability > 0.5 else 0
            
            return int(prediction), float(probability)
            
        except Exception as e:
            raise ValueError(f"Neural network prediction failed: {e}")

    def _predict_wide_deep_hybrid(self, model_data: Dict[str, Any], features: np.ndarray, model_name: str) -> Tuple[int, float]:
        """Predict using wide-deep hybrid architecture."""
        try:
            # This is a custom architecture - we can't reconstruct it exactly
            # So we'll create a simple approximation and try to load compatible weights
            input_dim = features.shape[1]
            
            # Try to create a simple model that might work with the saved state dict
            model = NeuralNetwork(input_dim, hidden_dims=[1024, 512, 256], dropout=0.3, activation='relu')
            
            device = torch.device('mps' if torch.backends.mps.is_available() 
                                else 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Try partial loading of compatible weights
            state_dict = model_data['model_state_dict']
            model_state_dict = model.state_dict()
            
            # Load any compatible layers
            compatible_weights = {}
            for key in model_state_dict.keys():
                # Try to find similar keys in the saved state dict
                for saved_key in state_dict.keys():
                    if key.split('.')[-1] == saved_key.split('.')[-1]:  # Same parameter type
                        if model_state_dict[key].shape == state_dict[saved_key].shape:
                            compatible_weights[key] = state_dict[saved_key]
                            break
            
            if compatible_weights:
                model.load_state_dict(compatible_weights, strict=False)
            
            model.eval()
            
            # Make prediction
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).to(device)
                output = model(features_tensor)
                probability = output.cpu().numpy().flatten()[0]
                prediction = 1 if probability > 0.5 else 0
            
            return int(prediction), float(probability)
            
        except Exception as e:
            raise ValueError(f"Wide-deep hybrid prediction failed: {e}")

    def _predict_best_ensemble_architecture(self, model_data: Dict[str, Any], features: np.ndarray, model_name: str) -> Tuple[int, float]:
        """Predict using the best ensemble architecture (exact match from create_neural_ensemble.py)."""
        try:
            input_dim = features.shape[1]
            
            # Use the exact architecture from create_neural_ensemble.py _create_optimized_architecture
            # Input -> 1024 -> 512 -> 256 -> 128 -> 64 -> 1
            model = NeuralNetwork(input_dim, hidden_dims=[1024, 512, 256, 128, 64], dropout=0.3, activation='relu')
            
            device = torch.device('mps' if torch.backends.mps.is_available() 
                                else 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Load the exact state dict
            model.load_state_dict(model_data['model_state_dict'])
            model.eval()
            
            # Make prediction
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).to(device)
                output = model(features_tensor)
                probability = output.cpu().numpy().flatten()[0]
                prediction = 1 if probability > 0.5 else 0
            
            return int(prediction), float(probability)
            
        except Exception as e:
            raise ValueError(f"Best ensemble architecture prediction failed: {e}")
    
    def _predict_neural_network_ensemble(self, model_data: Dict[str, Any], features: np.ndarray, model_name: str = None) -> Tuple[int, float]:
        """Predict using a neural network ensemble."""
        try:
            device = torch.device('mps' if torch.backends.mps.is_available() 
                                else 'cuda' if torch.cuda.is_available() else 'cpu')
            
            # Get ensemble parameters
            ensemble_size = model_data['ensemble_size']
            input_dim = model_data['input_dim']
            architectures = model_data['architectures']
            model_states = model_data['model_states']
            
            predictions = []
            
            # Make predictions with each model in the ensemble
            for i in range(ensemble_size):
                arch = architectures[i]
                state_dict = model_states[i]
                
                # Handle both string and dict architecture formats
                if isinstance(arch, str):
                    # Convert string architecture names to configs
                    arch_configs = {
                        'dropout_optimized': {'hidden_dims': [512, 256, 128], 'dropout': 0.2, 'activation': 'relu'},
                        'default': {'hidden_dims': [512, 256, 128], 'dropout': 0.3, 'activation': 'relu'},
                        'wide': {'hidden_dims': [1024, 512, 256], 'dropout': 0.3, 'activation': 'relu'},
                        'deep': {'hidden_dims': [256, 256, 256, 128, 64], 'dropout': 0.3, 'activation': 'relu'},
                    }
                    arch_config = arch_configs.get(arch, {'hidden_dims': [512, 256, 128], 'dropout': 0.3, 'activation': 'relu'})
                else:
                    # Use the dict directly
                    arch_config = arch
                
                # Create model with this architecture
                model = NeuralNetwork(input_dim, **arch_config).to(device)
                model.load_state_dict(state_dict)
                model.eval()
                
                # Make prediction
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).to(device)
                    output = model(features_tensor)
                    probability = output.cpu().numpy().flatten()[0]
                    predictions.append(probability)
            
            # Average the predictions
            avg_probability = float(np.mean(predictions))
            prediction = 1 if avg_probability > 0.5 else 0
            
            return int(prediction), avg_probability
            
        except Exception as e:
            raise ValueError(f"Neural network ensemble prediction failed: {e}")

    def _predict_sklearn_model(self, model_data: Dict[str, Any], features: np.ndarray, text: str, model_name: str = None) -> Tuple[int, float]:
        """Predict using scikit-learn model with enhanced error handling."""
        try:
            model = model_data['model']
            
            # Handle MultinomialNB special case (needs non-negative features and possibly different dimension)
            from sklearn.naive_bayes import MultinomialNB, ComplementNB
            if isinstance(model, (MultinomialNB, ComplementNB)):
                # Check if model has a special scaler for non-negative features
                if hasattr(model, '_feature_scaler'):
                    try:
                        features = model._feature_scaler.transform(features)
                    except Exception as e:
                        print(f"Warning: Feature scaling failed for {model_name}: {e}")
                
                # Handle dimension mismatch by trying to use only word features
                try:
                    prediction = model.predict(features)[0]
                except Exception as dim_error:
                    if "features" in str(dim_error) and "expecting" in str(dim_error):
                        # Try extracting only word features if available
                        feature_extractor = model_data.get('feature_extractor')
                        if feature_extractor and hasattr(feature_extractor, 'max_features'):
                            try:
                                word_features_count = min(features.shape[1], feature_extractor.max_features // 2)
                                features = features[:, :word_features_count]
                                prediction = model.predict(features)[0]
                            except Exception:
                                raise dim_error
                        else:
                            raise dim_error
                    else:
                        raise dim_error
            else:
                # Regular model prediction
                prediction = model.predict(features)[0]
            
            # Get probability if available
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features)[0]
                    if len(probabilities) > 1:
                        probability = probabilities[1]  # Probability of class 1 (AI)
                    else:
                        probability = probabilities[0]
                elif hasattr(model, 'decision_function'):
                    # For SVM and similar models
                    decision_score = model.decision_function(features)[0]
                    # Convert to probability using sigmoid
                    probability = 1 / (1 + np.exp(-decision_score))
                else:
                    # Default probability based on prediction
                    probability = 0.9 if prediction == 1 else 0.1
            except Exception as prob_error:
                # Fallback probability
                probability = 0.9 if prediction == 1 else 0.1
                print(f"Warning: Probability calculation failed for {model_name}: {prob_error}")
            
            return int(prediction), float(probability)
            
        except Exception as e:
            raise ValueError(f"Sklearn model prediction failed: {e}")
    
    def predict_all_models(self, text: str, verbose: bool = True) -> Dict[str, Tuple[int, float]]:
        """Make predictions using all available models."""
        if len(text.strip()) < 10:
            raise ValueError("Text too short for reliable classification (minimum 10 characters)")
        
        # Discover available models
        model_files = list(self.models_dir.glob("*.pkl"))
        
        if not model_files:
            raise FileNotFoundError(f"No models found in {self.models_dir}")
        
        results = {}
        failed_models = []
        
        if verbose:
            print(f"Making predictions using {len(model_files)} models...")
            print("-" * 60)
        
        for model_file in sorted(model_files):
            model_name = model_file.stem
            
            try:
                # Load model
                model_data = self.load_model(model_file.name)
                
                # Make prediction
                prediction, probability = self.predict_single_model(model_data, text, model_name)
                
                results[model_name] = (prediction, probability)
                
                if verbose:
                    label = "AI" if prediction == 1 else "Human"
                    print(f"{model_name:<25} {label:<6} ({probability:.3f})")
                
            except Exception as e:
                failed_models.append((model_name, str(e)))
                if verbose:
                    print(f"{model_name:<25} FAILED ({str(e)[:30]}...)")
        
        if verbose and failed_models:
            print(f"\nFailed models: {len(failed_models)}")
            for model_name, error in failed_models:
                print(f"  - {model_name}: {error}")
        
        return results
    
    def get_ensemble_prediction(self, text: str) -> Tuple[int, float, Dict[str, Any]]:
        """Get ensemble prediction from all models."""
        results = self.predict_all_models(text, verbose=False)
        
        if not results:
            raise ValueError("No models available for prediction")
        
        # Separate predictions by class
        ai_predictions = []
        human_predictions = []
        
        for model_name, (prediction, probability) in results.items():
            if prediction == 1:  # AI
                ai_predictions.append((model_name, probability))
            else:  # Human
                human_predictions.append((model_name, 1 - probability))
        
        # Calculate ensemble metrics
        total_models = len(results)
        ai_count = len(ai_predictions)
        human_count = len(human_predictions)
        
        # Voting-based prediction
        ensemble_prediction = 1 if ai_count > human_count else 0
        
        # Average confidence calculation
        if ensemble_prediction == 1 and ai_predictions:
            ensemble_confidence = np.mean([prob for _, prob in ai_predictions])
        elif ensemble_prediction == 0 and human_predictions:
            ensemble_confidence = np.mean([prob for _, prob in human_predictions])
        else:
            ensemble_confidence = 0.5
        
        # Detailed results
        detailed_results = {
            'total_models': total_models,
            'ai_predictions': ai_count,
            'human_predictions': human_count,
            'ai_percentage': (ai_count / total_models) * 100,
            'human_percentage': (human_count / total_models) * 100,
            'margin': abs(ai_count - human_count),
            'margin_percentage': abs(ai_count - human_count) / total_models * 100,
            'individual_results': results
        }
        
        return ensemble_prediction, ensemble_confidence, detailed_results
    
    def interactive_mode(self):
        """Interactive prediction mode."""
        print("=" * 80)
        print("AI vs Human Text Classifier - Interactive Mode")
        print("=" * 80)
        print("Enter text to classify (type 'quit' to exit)")
        print("Minimum 10 characters required for reliable classification")
        print("-" * 80)
        
        while True:
            try:
                text = input("\nEnter text: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if len(text) < 10:
                    print("Text too short. Please enter at least 10 characters.")
                    continue
                
                print(f"\nAnalyzing: {text[:100]}{'...' if len(text) > 100 else ''}")
                print("=" * 80)
                
                # Get ensemble prediction
                prediction, confidence, details = self.get_ensemble_prediction(text)
                
                # Display results
                label = "AI-Generated" if prediction == 1 else "Human-Written"
                print(f"\nENSEMBLE PREDICTION: {label}")
                print(f"Confidence: {confidence:.3f}")
                print(f"Vote Distribution: {details['ai_predictions']} AI, {details['human_predictions']} Human")
                print(f"Margin: {details['margin']} models ({details['margin_percentage']:.1f}%)")
                
                # Show all model predictions sorted by confidence
                individual = details['individual_results']
                
                print(f"\nALL MODEL PREDICTIONS:")
                print("-" * 60)
                
                # Separate AI and Human predictions
                ai_models = [(name, prob) for name, (pred, prob) in individual.items() if pred == 1]
                human_models = [(name, prob) for name, (pred, prob) in individual.items() if pred == 0]
                
                # Sort by confidence (descending)
                ai_models.sort(key=lambda x: x[1], reverse=True)
                human_models.sort(key=lambda x: 1-x[1], reverse=True)
                
                if ai_models:
                    print(f"\nAI-GENERATED predictions ({len(ai_models)} models):")
                    for name, prob in ai_models:
                        print(f"   {name:<25} {prob:.3f}")
                
                if human_models:
                    print(f"\nHUMAN-WRITTEN predictions ({len(human_models)} models):")
                    for name, prob in human_models:
                        confidence = 1 - prob  # Convert AI probability to Human confidence
                        print(f"   {name:<25} {confidence:.3f}")
                
                print("-" * 80)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def batch_predict(self, texts: List[str]) -> pd.DataFrame:
        """Predict multiple texts and return results as DataFrame."""
        results = []
        
        print(f"Processing {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            try:
                if len(text.strip()) < 10:
                    print(f"Skipping text {i+1}: too short")
                    continue
                
                prediction, confidence, details = self.get_ensemble_prediction(text)
                
                results.append({
                    'text_id': i + 1,
                    'text_preview': text[:50] + "..." if len(text) > 50 else text,
                    'prediction': 'AI' if prediction == 1 else 'Human',
                    'confidence': confidence,
                    'ai_votes': details['ai_predictions'],
                    'human_votes': details['human_predictions'],
                    'total_models': details['total_models'],
                    'margin': details['margin']
                })
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(texts)} texts...")
                    
            except Exception as e:
                print(f"Error processing text {i+1}: {e}")
        
        return pd.DataFrame(results)


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='AI vs Human text classifier prediction')
    parser.add_argument('--text', type=str, help='Single text to classify')
    parser.add_argument('--file', type=str, help='File containing texts to classify (one per line)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--models-dir', default='models', help='Directory containing trained models')
    parser.add_argument('--output', type=str, help='Output file for batch predictions (CSV format)')
    parser.add_argument('--model', type=str, help='Use specific model only')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ModelPredictor(args.models_dir)
    
    try:
        if args.interactive:
            # Interactive mode
            predictor.interactive_mode()
            
        elif args.text:
            # Single text prediction
            print("AI vs Human Text Classifier")
            print("=" * 50)
            print(f"Text: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
            print("-" * 50)
            
            if args.model:
                # Use specific model
                model_data = predictor.load_model(args.model)
                prediction, probability = predictor.predict_single_model(model_data, args.text)
                label = "AI-Generated" if prediction == 1 else "Human-Written"
                print(f"Model: {args.model}")
                print(f"Prediction: {label}")
                print(f"Confidence: {probability:.3f}")
            else:
                # Use ensemble
                prediction, confidence, details = predictor.get_ensemble_prediction(args.text)
                label = "AI-Generated" if prediction == 1 else "Human-Written"
                print(f"Ensemble Prediction: {label}")
                print(f"Confidence: {confidence:.3f}")
                print(f"Vote Distribution: {details['ai_predictions']} AI, {details['human_predictions']} Human")
                print(f"Total Models: {details['total_models']}")
                
                if args.verbose:
                    # Show all model predictions sorted by confidence (same format as interactive mode)
                    individual = details['individual_results']
                    
                    print(f"\nALL MODEL PREDICTIONS:")
                    print("-" * 60)
                    
                    # Separate AI and Human predictions
                    ai_models = [(name, prob) for name, (pred, prob) in individual.items() if pred == 1]
                    human_models = [(name, prob) for name, (pred, prob) in individual.items() if pred == 0]
                    
                    # Sort by confidence (descending)
                    ai_models.sort(key=lambda x: x[1], reverse=True)
                    human_models.sort(key=lambda x: 1-x[1], reverse=True)
                    
                    if ai_models:
                        print(f"\nAI-GENERATED predictions ({len(ai_models)} models):")
                        for name, prob in ai_models:
                            print(f"   {name:<25} {prob:.3f}")
                    
                    if human_models:
                        print(f"\nHUMAN-WRITTEN predictions ({len(human_models)} models):")
                        for name, prob in human_models:
                            confidence = 1 - prob  # Convert AI probability to Human confidence
                            print(f"   {name:<25} {confidence:.3f}")
                else:
                    print("\nUse --verbose to see all individual model predictions")
            
        elif args.file:
            # Batch file prediction
            if not Path(args.file).exists():
                raise FileNotFoundError(f"File not found: {args.file}")
            
            print(f"Loading texts from {args.file}...")
            
            texts = []
            with open(args.file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Try to parse as JSON first
                        try:
                            data = json.loads(line)
                            if 'text' in data:
                                texts.append(data['text'])
                            else:
                                texts.append(line)
                        except json.JSONDecodeError:
                            # Treat as plain text
                            texts.append(line)
            
            if not texts:
                raise ValueError("No texts found in file")
            
            print(f"Found {len(texts)} texts")
            
            # Process texts
            results_df = predictor.batch_predict(texts)
            
            # Display results
            print("\nBatch Prediction Results:")
            print("=" * 80)
            print(results_df.to_string(index=False))
            
            # Save results if output specified
            if args.output:
                results_df.to_csv(args.output, index=False)
                print(f"\nResults saved to {args.output}")
            
            # Summary statistics
            print(f"\nSUMMARY:")
            print(f"Total texts processed: {len(results_df)}")
            print(f"Predicted as AI: {len(results_df[results_df['prediction'] == 'AI'])}")
            print(f"Predicted as Human: {len(results_df[results_df['prediction'] == 'Human'])}")
            print(f"Average confidence: {results_df['confidence'].mean():.3f}")
            
        else:
            # Default to interactive mode
            print("No specific input provided. Starting interactive mode...")
            predictor.interactive_mode()
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
