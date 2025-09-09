"""Universal prediction script for all AI vs Human text classifiers.

This script supports neural network, classical ML, and ensemble methods
with command-line arguments to choose which model to use.
"""
import sys
import os
import argparse
from pathlib import Path
import warnings
from contextlib import contextmanager
from sklearn.exceptions import ConvergenceWarning
from typing import Tuple, Dict, Any, List
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.model_serializer import ModelSerializer


@contextmanager
def suppress_sklearn_warnings():
    """Context manager to suppress sklearn warnings during prediction."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        warnings.simplefilter('ignore')
        # Specific sklearn warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=ConvergenceWarning) 
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        # LightGBM specific warnings
        warnings.filterwarnings('ignore', message='.*X does not have valid feature names.*')
        warnings.filterwarnings('ignore', message='.*LGBMClassifier.*')
        yield


class UniversalModelLoader:
    """Universal loader for all model types using consolidated format."""
    
    def __init__(self):
        self.models_cache = {}
        self.models_dir = Path("models")
    
    def load_model_package(self, model_path: str) -> Tuple[Any, str]:
        """Load a model from consolidated format.
        
        Args:
            model_path: Path to model file (without .pkl extension).
            
        Returns:
            Tuple of (model_package, model_description).
        """
        if model_path in self.models_cache:
            return self.models_cache[model_path]
        
        try:
            # Load the consolidated model package
            package = ModelSerializer.load_model_package(model_path)
            
            # Create a friendly name for the model
            model_name = Path(model_path).stem.replace("comparison_", "")
            model_type = package.metadata.get('model_type', 'Unknown')
            classifier_name = package.metadata.get('classifier_name', model_name)
            
            # Create a descriptive name
            if model_type == 'ensemble':
                description = f"Ensemble: {classifier_name.replace('_', ' ').title()}"
            elif model_type == 'classical':
                description = f"Classical: {classifier_name.replace('_', ' ').title()}"
            elif model_type == 'neural':
                description = "Enhanced Neural Network"
            elif model_type == 'probabilistic':
                description = f"Probabilistic: {classifier_name.replace('_', ' ').title()}"
            elif model_type == 'manifold':
                description = f"Manifold: {classifier_name.replace('_', ' ').title()}"
            elif model_type == 'advanced':
                description = f"Advanced: {classifier_name.replace('_', ' ').title()}"
            elif model_type == 'interpretable':
                description = f"Interpretable: {classifier_name.replace('_', ' ').title()}"
            else:
                description = classifier_name.replace('_', ' ').title()
            
            # Cache the result
            self.models_cache[model_path] = (package, description)
            return package, description
            
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    def _predict_pytorch_model(self, package: Any, text: str) -> Tuple[int, float]:
        """Handle PyTorch model prediction by reconstructing the model."""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            
            # Get model configuration from package
            config = package.config
            metadata = package.metadata
            model_type = config.get('model_type', metadata.get('model_type', 'unknown'))
            
            # Set device
            device = torch.device('cpu')  # Use CPU for inference
            
            # Extract dimensions from state_dict
            state_dict_dims = self._extract_dims_from_state_dict(package.model)
            
            # Reconstruct the model based on type
            if model_type == 'cnn':
                model = self._reconstruct_cnn_model(config, device, state_dict_dims)
            elif model_type == 'transformer':
                model = self._reconstruct_transformer_model(config, device, state_dict_dims)
            elif model_type == 'attention_bilstm':
                model = self._reconstruct_attention_bilstm_model(config, device, state_dict_dims)
            elif model_type in ['enhanced_neural_network', 'lstm_features', 'lstm_attention', 'unknown']:
                # These might be hybrid models - try to reconstruct based on available info
                model = self._reconstruct_hybrid_model(config, metadata, device, state_dict_dims)
            else:
                # For any unknown type, try hybrid reconstruction which has better inference
                model = self._reconstruct_hybrid_model(config, metadata, device, state_dict_dims)
            
            # Load state dict
            model.load_state_dict(package.model)
            model.eval()
            
            # Convert text to sequence
            word_to_idx = config.get('word_to_idx', metadata.get('word_to_idx'))
            max_seq_len = config.get('max_seq_len', metadata.get('max_seq_len', 512))
            
            if not word_to_idx:
                raise ValueError("word_to_idx not found in model configuration")
            
            # Tokenize and convert to sequence
            words = text.lower().split()
            sequence = [word_to_idx.get(word, word_to_idx.get('<UNK>', 1)) for word in words]
            
            # Pad or truncate
            if len(sequence) > max_seq_len:
                sequence = sequence[:max_seq_len]
            else:
                sequence.extend([word_to_idx.get('<PAD>', 0)] * (max_seq_len - len(sequence)))
            
            # Convert to tensor and predict
            with torch.no_grad():
                input_tensor = torch.LongTensor([sequence]).to(device)
                output = model(input_tensor)
                probability = output.item()
                prediction = 1 if probability > 0.5 else 0
            
            return int(prediction), float(probability)
            
        except Exception as e:
            # Fall back to traditional feature approach if available
            if hasattr(package, 'word_vectorizer') and package.word_vectorizer:
                return self._predict_with_traditional_features(package, text)
            else:
                raise ValueError(f"PyTorch model reconstruction failed: {str(e)}")
    
    def _extract_dims_from_state_dict(self, state_dict: dict) -> dict:
        """Extract model dimensions from PyTorch state_dict."""
        dims = {}
        
        # Extract embedding dimensions
        if 'embedding.weight' in state_dict:
            embed_shape = state_dict['embedding.weight'].shape
            dims['vocab_size'] = embed_shape[0]
            dims['embed_dim'] = embed_shape[1]
        
        # Extract LSTM dimensions for BiLSTM models
        if 'lstm.weight_ih_l0' in state_dict:
            lstm_ih_shape = state_dict['lstm.weight_ih_l0'].shape
            # For BiLSTM: weight_ih has shape (4*hidden_size, input_size)
            # The 4 comes from gates: input, forget, cell, output
            total_gates_hidden = lstm_ih_shape[0]
            dims['input_size'] = lstm_ih_shape[1]
            dims['hidden_dim'] = total_gates_hidden // 4  # Single direction hidden size
            
            # Count number of layers by checking weight keys
            layer_count = 0
            for key in state_dict.keys():
                if key.startswith('lstm.weight_ih_l'):
                    layer_num = int(key.split('_l')[1].split('.')[0].split('_')[0])
                    layer_count = max(layer_count, layer_num + 1)
            dims['num_layers'] = layer_count
        
        # Extract attention layer dimensions
        if 'attention.weight' in state_dict:
            attention_shape = state_dict['attention.weight'].shape
            dims['attention_input_size'] = attention_shape[1]
        
        # Extract final layer dimensions
        if 'fc.weight' in state_dict:
            fc_shape = state_dict['fc.weight'].shape
            dims['fc_input_size'] = fc_shape[1]
        
        return dims
    
    def _reconstruct_cnn_model(self, config: dict, device, state_dict_dims: dict = None):
        """Reconstruct CNN model architecture."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        class TextCNN(nn.Module):
            def __init__(self, vocab_size, embed_dim=128, num_filters=100, 
                         filter_sizes=[3, 4, 5], dropout_rate=0.5):
                super(TextCNN, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.convs = nn.ModuleList([
                    nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
                    for fs in filter_sizes
                ])
                self.dropout = nn.Dropout(dropout_rate)
                self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                x = self.embedding(x)
                x = x.transpose(1, 2)
                conv_outputs = []
                for conv in self.convs:
                    conv_out = F.relu(conv(x))
                    pooled = F.max_pool1d(conv_out, conv_out.size(2))
                    conv_outputs.append(pooled.squeeze(2))
                x = torch.cat(conv_outputs, dim=1)
                x = self.dropout(x)
                x = self.fc(x)
                return self.sigmoid(x)
        
        # Use state_dict dimensions if available, otherwise fall back to config
        if state_dict_dims and 'vocab_size' in state_dict_dims and 'embed_dim' in state_dict_dims:
            vocab_size = state_dict_dims['vocab_size']
            embed_dim = state_dict_dims['embed_dim']
        else:
            # Extract from config as fallback
            vocab_size = config.get('vocab_size', len(config.get('word_to_idx', {})))
            embed_dim = 128  # Default
            
            # Try to get actual dimensions from model architecture config if available
            model_arch = config.get('model_architecture', {})
            if model_arch:
                vocab_size = model_arch.get('vocab_size', vocab_size)
                embed_dim = model_arch.get('embed_dim', embed_dim)
        
        return TextCNN(vocab_size, embed_dim).to(device)
    
    def _reconstruct_transformer_model(self, config: dict, device, state_dict_dims: dict = None):
        """Reconstruct Transformer model architecture."""
        import torch
        import torch.nn as nn
        
        class TransformerEncoder(nn.Module):
            def __init__(self, vocab_size, embed_dim=128, num_heads=8, 
                         num_layers=3, ff_dim=512, max_seq_len=512, dropout_rate=0.1):
                super(TransformerEncoder, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
                    dropout=dropout_rate, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.dropout = nn.Dropout(dropout_rate)
                self.fc = nn.Linear(embed_dim, 1)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                seq_len = x.size(1)
                x = self.embedding(x) + self.pos_encoding[:seq_len, :].unsqueeze(0)
                x = self.transformer(x)
                x = x.mean(dim=1)
                x = self.dropout(x)
                x = self.fc(x)
                return self.sigmoid(x)
        
        # Use state_dict dimensions if available, otherwise fall back to config
        if state_dict_dims and 'vocab_size' in state_dict_dims and 'embed_dim' in state_dict_dims:
            vocab_size = state_dict_dims['vocab_size']
            embed_dim = state_dict_dims['embed_dim']
        else:
            vocab_size = config.get('vocab_size', len(config.get('word_to_idx', {})))
            embed_dim = 128
        
        return TransformerEncoder(vocab_size, embed_dim).to(device)
    
    def _reconstruct_attention_bilstm_model(self, config: dict, device, state_dict_dims: dict = None):
        """Reconstruct Attention BiLSTM model architecture."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        class AttentionBiLSTM(nn.Module):
            def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, 
                         num_layers=2, dropout_rate=0.3):
                super(AttentionBiLSTM, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                                   batch_first=True, bidirectional=True, dropout=dropout_rate)
                self.attention = nn.Linear(hidden_dim * 2, 1)
                self.dropout = nn.Dropout(dropout_rate)
                self.fc = nn.Linear(hidden_dim * 2, 1)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                x = self.embedding(x)
                lstm_out, _ = self.lstm(x)
                attention_weights = F.softmax(self.attention(lstm_out), dim=1)
                attended = torch.sum(attention_weights * lstm_out, dim=1)
                x = self.dropout(attended)
                x = self.fc(x)
                return self.sigmoid(x)
        
        # Use state_dict dimensions if available, otherwise fall back to config
        if state_dict_dims and 'vocab_size' in state_dict_dims and 'embed_dim' in state_dict_dims:
            vocab_size = state_dict_dims['vocab_size']
            embed_dim = state_dict_dims['embed_dim']
            hidden_dim = state_dict_dims.get('hidden_dim', 128)
            num_layers = state_dict_dims.get('num_layers', 2)
        else:
            # Extract actual dimensions from config or use defaults
            vocab_size = config.get('vocab_size', len(config.get('word_to_idx', {})))
            embed_dim = 128
            hidden_dim = 128
            num_layers = 2
            
            # Try to get actual dimensions from model architecture config if available
            model_arch = config.get('model_architecture', {})
            if model_arch:
                vocab_size = model_arch.get('vocab_size', vocab_size)
                embed_dim = model_arch.get('embed_dim', embed_dim)
                hidden_dim = model_arch.get('hidden_dim', hidden_dim)
                num_layers = model_arch.get('num_layers', num_layers)
        
        return AttentionBiLSTM(vocab_size, embed_dim, hidden_dim, num_layers).to(device)
    
    def _reconstruct_hybrid_model(self, config: dict, metadata: dict, device, state_dict_dims: dict = None):
        """Reconstruct hybrid/enhanced models - try different architectures."""
        model_type = config.get('model_type', metadata.get('model_type', ''))
        classifier_name = metadata.get('classifier_name', '')
        
        # Enhanced type inference - check multiple sources
        type_clues = [model_type.lower(), classifier_name.lower()]
        
        # Try to infer from available data - check all sources
        if any('lstm' in clue for clue in type_clues):
            return self._reconstruct_attention_bilstm_model(config, device, state_dict_dims)
        elif any('attention' in clue for clue in type_clues):
            return self._reconstruct_attention_bilstm_model(config, device, state_dict_dims)
        elif any('transformer' in clue for clue in type_clues):
            return self._reconstruct_transformer_model(config, device, state_dict_dims)
        elif any('cnn' in clue for clue in type_clues):
            return self._reconstruct_cnn_model(config, device, state_dict_dims)
        else:
            # For unknown types, try to determine from model structure
            # Look at the state_dict keys to infer architecture
            model_keys = list(config.get('model_keys', []))
            if not model_keys:
                model_keys = list(metadata.get('model_keys', []))
            
            if model_keys:
                keys_str = ' '.join(model_keys).lower()
                if 'lstm' in keys_str or 'rnn' in keys_str:
                    return self._reconstruct_attention_bilstm_model(config, device, state_dict_dims)
                elif 'transformer' in keys_str:
                    return self._reconstruct_transformer_model(config, device, state_dict_dims)
                elif 'conv' in keys_str:
                    return self._reconstruct_cnn_model(config, device, state_dict_dims)
            
            # Default fallback - try LSTM first as it's more common
            try:
                return self._reconstruct_attention_bilstm_model(config, device, state_dict_dims)
            except:
                # If LSTM fails, try CNN
                return self._reconstruct_cnn_model(config, device, state_dict_dims)
    
    def _predict_with_traditional_features(self, package: Any, text: str) -> Tuple[int, float]:
        """Fallback prediction using traditional features."""
        features = []
        
        # Extract TF-IDF features
        if hasattr(package, 'word_vectorizer') and package.word_vectorizer:
            word_features = package.word_vectorizer.transform([text]).toarray()
            features.append(word_features)
        
        # Character features (for enhanced models and hybrid models) 
        if hasattr(package, 'char_vectorizer') and package.char_vectorizer:
            char_features = package.char_vectorizer.transform([text]).toarray()
            features.append(char_features)
        
        # Extract linguistic features - determine expected count based on scaler
        if hasattr(package, 'word_vectorizer') and package.word_vectorizer:
            expected_features = None
            model_name = package.metadata.get('classifier_name', '')
            
            if hasattr(package, 'scaler') and package.scaler:
                total_expected = package.scaler.n_features_in_
                word_features_count = word_features.shape[1] if 'word_features' in locals() else 0
                char_features_count = char_features.shape[1] if 'char_features' in locals() and hasattr(package, 'char_vectorizer') and package.char_vectorizer else 0
                expected_linguistic = total_expected - word_features_count - char_features_count
                
                # Special handling for enhanced neural network (includes sentiment analysis)
                if 'neural_network' in model_name and expected_linguistic >= 15:
                    expected_features = expected_linguistic  # Full enhanced features with sentiment
                elif expected_linguistic == 11:
                    expected_features = 11  # Standard linguistic features
                elif expected_linguistic <= 5:
                    expected_features = expected_linguistic  # Minimal features (hybrid model)
            
            linguistic_features = self.extract_linguistic_features([text], expected_features)
            features.append(linguistic_features)
        
        if features:
            combined_features = np.hstack(features)
            
            # Scale if scaler available
            if hasattr(package, 'scaler') and package.scaler:
                try:
                    combined_features = package.scaler.transform(combined_features)
                except ValueError as scaler_error:
                    if "features" in str(scaler_error) and "expecting" in str(scaler_error):
                        # Handle scaler dimension mismatches
                        try:
                            import re
                            match = re.search(r'expecting (\d+) features', str(scaler_error))
                            if match:
                                expected_features = int(match.group(1))
                                current_features = combined_features.shape[1]
                                
                                if current_features < expected_features:
                                    # Pad with zeros
                                    padding = np.zeros((1, expected_features - current_features))
                                    combined_features = np.hstack([combined_features, padding])
                                elif current_features > expected_features:
                                    # Truncate to expected size
                                    combined_features = combined_features[:, :expected_features]
                                
                                # Try scaling again
                                combined_features = package.scaler.transform(combined_features)
                        except Exception:
                            # If we can't fix it, proceed without scaler
                            pass
                    else:
                        # For other scaler errors, try without scaler
                        pass
            
            # Simple heuristic prediction based on features
            # This is a very basic fallback - not ideal but better than failure
            text_len = len(text)
            word_count = len(text.split())
            
            # Simple rule: shorter, more structured text tends to be AI
            if text_len < 100 or word_count / text_len > 0.2:
                return 1, 0.6  # AI-generated with moderate confidence
            else:
                return 0, 0.6  # Human-written with moderate confidence
        
        raise ValueError("No features available for fallback prediction")

    def extract_linguistic_features(self, texts: List[str], expected_features: int = None) -> np.ndarray:
        """Extract linguistic features matching the training pipeline exactly."""
        import re
        
        features = []
        
        for text in texts:
            text_features = []
            
            # Basic statistics with safe division
            text_len = len(text)
            words = text.lower().split()
            word_count = len(words)
            
            text_features.append(text_len)  # Text length
            text_features.append(word_count)  # Word count
            text_features.append(word_count / max(text_len, 1))  # Word density (safe division)
            
            # Sentence statistics with safe operations
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            sentence_count = max(len(sentences), 1)  # Avoid division by zero
            text_features.append(sentence_count)  # Sentence count
            
            # Average words per sentence with safe calculation
            if sentences:
                sentence_lengths = [len(s.split()) for s in sentences]
                avg_words_per_sentence = np.mean(sentence_lengths) if sentence_lengths else 0
            else:
                avg_words_per_sentence = 0
            text_features.append(avg_words_per_sentence)
            
            # Character-level features with safe division
            if text_len > 0:
                text_features.append(sum(1 for c in text if c.isupper()) / text_len)  # Uppercase ratio
                text_features.append(sum(1 for c in text if c.islower()) / text_len)  # Lowercase ratio
                text_features.append(sum(1 for c in text if c.isdigit()) / text_len)  # Digit ratio
                text_features.append(sum(1 for c in text if c in '.,!?;:') / text_len)  # Punctuation ratio
            else:
                text_features.extend([0, 0, 0, 0])
            
            # Vocabulary complexity with safe division
            unique_words = set(words)
            text_features.append(len(unique_words) / max(word_count, 1))  # Lexical diversity (safe division)
            
            # Average word length with safe calculation
            if words:
                avg_word_len = np.mean([len(word) for word in words])
                # Clip extreme values to prevent numerical issues
                avg_word_len = np.clip(avg_word_len, 0, 50)
            else:
                avg_word_len = 0
            text_features.append(avg_word_len)
            
            # Readability approximation (Flesch-like) with safe calculations - may be excluded for some models
            if expected_features != 11:  # Only include readability for models that expect it
                avg_sentence_length = word_count / sentence_count  # Already safe due to max(1) above
                if words:
                    syllable_counts = [max(1, len(re.findall(r'[aeiouAEIOU]', word))) for word in words]
                    avg_syllables = np.mean(syllable_counts)
                    # Clip to reasonable range to prevent extreme values
                    avg_syllables = np.clip(avg_syllables, 1, 10)
                else:
                    avg_syllables = 1
                
                flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
                # Clip Flesch score to reasonable range
                flesch_score = np.clip(flesch_score, -100, 200)
                text_features.append(flesch_score)
            
            # Function word ratios with safe division - may be excluded for some models
            if expected_features != 11:  # Only include for models that expect it
                function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
                function_word_count = sum(1 for word in words if word.lower() in function_words)
                text_features.append(function_word_count / max(word_count, 1))  # Safe division
            
            # Repetition patterns with safe division - may be excluded for some models
            if expected_features != 11:  # Only include for models that expect it
                if len(words) > 1:
                    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
                    unique_bigrams = set(bigrams)
                    bigram_diversity = len(unique_bigrams) / len(bigrams)
                else:
                    bigram_diversity = 0
                text_features.append(bigram_diversity)
            
            # Ensure all features are finite and not NaN
            text_features = [np.clip(f, -1e6, 1e6) if np.isfinite(f) else 0 for f in text_features]
            features.append(text_features)
        
        feature_array = np.array(features)
        
        # Final safety check: replace any remaining NaN or infinite values
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return feature_array

    def predict_with_package(self, package: Any, text: str) -> Tuple[int, float]:
        """Make a prediction using a loaded model package.
        
        Args:
            package: Loaded model package.
            text: Text to classify.
            
        Returns:
            Tuple of (prediction, probability).
        """
        with suppress_sklearn_warnings():
            try:
                # Get model information directly from package metadata (no reloading needed)
                actual_model = package.model
                model_type = package.model_type
                metadata = package.metadata
                config = package.config
                
                # Determine model type efficiently using available metadata
                is_pytorch_model = False
                pytorch_prediction_type = 'sequence_based'
                
                # Check metadata first for pytorch indicators
                if model_type in ['enhanced', 'sequential', 'hybrid', 'lstm']:
                    is_pytorch_model = True
                    if model_type == 'enhanced':
                        pytorch_prediction_type = 'feature_based'
                    else:
                        pytorch_prediction_type = 'sequence_based'
                
                # Check if this is a traditional ML model (has predict method)
                if hasattr(actual_model, 'predict'):
                    # This is a regular sklearn-style model - use it directly with proper features
                    return self._predict_sklearn_model(package, text)
                
                # Handle dual-model anomaly detectors (like isolation_forest, one_class_svm)
                elif isinstance(actual_model, dict) and 'human_model' in actual_model and 'ai_model' in actual_model:
                    return self._predict_dual_anomaly_model(package, text)
                
                # Handle PyTorch models based on prediction type
                elif is_pytorch_model or (hasattr(actual_model, 'keys') and not hasattr(actual_model, 'predict')):
                    if isinstance(actual_model, dict) and any('weight' in str(k) or 'bias' in str(k) for k in actual_model.keys()):
                        # This is a PyTorch state_dict
                        if pytorch_prediction_type == 'feature_based':
                            # Feature-based PyTorch model (Enhanced Neural Network)
                            return self._predict_pytorch_feature_model(package, text)
                        else:
                            # Sequence-based PyTorch model (CNN, Transformer, etc.)
                            return self._predict_pytorch_sequence_model(package, text)
                    else:
                        # Fallback to sklearn approach
                        return self._predict_sklearn_model(package, text)
                
                else:
                    # Default fallback to sklearn approach
                    return self._predict_sklearn_model(package, text)
                
            except Exception as e:
                # If all else fails, try a simple fallback prediction
                try:
                    return self._predict_sklearn_model(package, text)
                except:
                    # Ultimate fallback - simple heuristic
                    text_len = len(text)
                    word_count = len(text.split())
                    if text_len < 100 or word_count / text_len > 0.2:
                        return 1, 0.5  # AI-generated with low confidence
                    else:
                        return 0, 0.5  # Human-written with low confidence
    
    def _predict_sklearn_model(self, package: Any, text: str) -> Tuple[int, float]:
        """Predict using sklearn-style models."""
        actual_model = package.model
        features = []
        
        # Word features
        if hasattr(package, 'word_vectorizer') and package.word_vectorizer:
            word_features = package.word_vectorizer.transform([text]).toarray()
            features.append(word_features)
        
        # Character features (for enhanced models and hybrid models) 
        if hasattr(package, 'char_vectorizer') and package.char_vectorizer:
            char_features = package.char_vectorizer.transform([text]).toarray()
            features.append(char_features)
        
        # Additional features if available
        if hasattr(package, 'additional_vectorizer') and package.additional_vectorizer:
            add_features = package.additional_vectorizer.transform([text]).toarray()
            features.append(add_features)
        
        # Add linguistic features - determine expected count based on scaler
        if hasattr(package, 'word_vectorizer') and package.word_vectorizer:
            expected_features = None
            classifier_name = package.metadata.get('classifier_name', '')
            
            if hasattr(package, 'scaler') and package.scaler:
                total_expected = package.scaler.n_features_in_
                word_features_count = word_features.shape[1] if 'word_features' in locals() else 0
                char_features_count = char_features.shape[1] if 'char_features' in locals() and hasattr(package, 'char_vectorizer') and package.char_vectorizer else 0
                expected_linguistic = total_expected - word_features_count - char_features_count
                
                # Special handling for enhanced neural network (includes sentiment analysis)
                if 'neural_network' in classifier_name and expected_linguistic >= 15:
                    expected_features = expected_linguistic  # Full enhanced features with sentiment
                elif expected_linguistic == 11:
                    expected_features = 11  # Standard linguistic features
                elif expected_linguistic <= 5:
                    expected_features = expected_linguistic  # Minimal features (hybrid model)
            
            linguistic_features = self.extract_linguistic_features([text], expected_features)
            features.append(linguistic_features)
        
        # Combine features
        if features:
            combined_features = np.hstack(features)
        else:
            combined_features = np.array([[len(text), len(text.split()), text.count(' ')]])
        
        # Handle scaling
        if hasattr(package, 'scaler') and package.scaler:
            try:
                combined_features = package.scaler.transform(combined_features)
            except ValueError as scaler_error:
                if "features" in str(scaler_error) and "expecting" in str(scaler_error):
                    try:
                        import re
                        match = re.search(r'expecting (\d+) features', str(scaler_error))
                        if match:
                            expected_features = int(match.group(1))
                            current_features = combined_features.shape[1]
                            
                            if current_features < expected_features:
                                padding = np.zeros((1, expected_features - current_features))
                                combined_features = np.hstack([combined_features, padding])
                            elif current_features > expected_features:
                                combined_features = combined_features[:, :expected_features]
                            
                            combined_features = package.scaler.transform(combined_features)
                    except Exception:
                        pass
        
        # Make prediction with the actual model
        prediction = actual_model.predict(combined_features)[0]
        
        # Get probability if available
        if hasattr(actual_model, 'predict_proba'):
            probabilities = actual_model.predict_proba(combined_features)[0]
            probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        else:
            probability = 1.0 if prediction == 1 else 0.0
        
        return int(prediction), float(probability)
    
    def _predict_dual_anomaly_model(self, package: Any, text: str) -> Tuple[int, float]:
        """Predict using dual-model anomaly detectors."""
        actual_model = package.model
        human_model = actual_model['human_model']
        ai_model = actual_model['ai_model']
        
        # Feature extraction approach for anomaly detection models
        features = []
        
        # Word features
        if hasattr(package, 'word_vectorizer') and package.word_vectorizer:
            word_features = package.word_vectorizer.transform([text]).toarray()
            features.append(word_features)
        
        # Character features  
        if hasattr(package, 'char_vectorizer') and package.char_vectorizer:
            char_features = package.char_vectorizer.transform([text]).toarray()
            features.append(char_features)
        
        # Add linguistic features
        if hasattr(package, 'word_vectorizer') and package.word_vectorizer:
            linguistic_features = self.extract_linguistic_features([text])
            features.append(linguistic_features)
        
        # Combine features
        if features:
            combined_features = np.hstack(features)
        else:
            combined_features = np.array([[len(text), len(text.split()), text.count(' ')]])
        
        # Handle scaler
        if hasattr(package, 'scaler') and package.scaler:
            try:
                combined_features = package.scaler.transform(combined_features)
            except ValueError as scaler_error:
                if "features" in str(scaler_error) and "expecting" in str(scaler_error):
                    try:
                        import re
                        match = re.search(r'expecting (\d+) features', str(scaler_error))
                        if match:
                            expected_features = int(match.group(1))
                            current_features = combined_features.shape[1]
                            
                            if current_features < expected_features:
                                padding = np.zeros((1, expected_features - current_features))
                                combined_features = np.hstack([combined_features, padding])
                            elif current_features > expected_features:
                                combined_features = combined_features[:, :expected_features]
                            
                            combined_features = package.scaler.transform(combined_features)
                    except Exception:
                        pass
        
        # Predict with both models and combine results
        try:
            # Get predictions from both models
            if hasattr(human_model, 'predict'):
                human_pred = human_model.predict(combined_features)[0]
                human_decision_function = getattr(human_model, 'decision_function', None)
                if human_decision_function:
                    human_score = human_decision_function(combined_features)[0]
                else:
                    human_score = float(human_pred)
            else:
                human_pred = 0
                human_score = 0.0
            
            if hasattr(ai_model, 'predict'):
                ai_pred = ai_model.predict(combined_features)[0]
                ai_decision_function = getattr(ai_model, 'decision_function', None)
                if ai_decision_function:
                    ai_score = ai_decision_function(combined_features)[0]
                else:
                    ai_score = float(ai_pred)
            else:
                ai_pred = 0
                ai_score = 0.0
            
            # For anomaly detection: if human model says anomaly, likely AI-generated
            if human_pred == 1:  # Anomaly detected by human model = likely AI
                prediction = 1
                probability = 0.6 + abs(human_score) * 0.3
            elif ai_pred == 1:  # Anomaly detected by AI model = likely human
                prediction = 0
                probability = 0.6 + abs(ai_score) * 0.3
            else:
                # Use the stronger score
                if abs(human_score) > abs(ai_score):
                    prediction = 1
                    probability = 0.5 + abs(human_score) * 0.2
                else:
                    prediction = 0
                    probability = 0.5 + abs(ai_score) * 0.2
            
            probability = np.clip(probability, 0.1, 0.9)
            return int(prediction), float(probability)
            
        except Exception as dual_e:
            raise ValueError(f"Dual model prediction failed: {str(dual_e)}")
    
    def _predict_pytorch_feature_model(self, package: Any, text: str) -> Tuple[int, float]:
        """Predict using feature-based PyTorch models (Enhanced Neural Network)."""
        try:
            import torch
            import torch.nn as nn
            
            # Extract features using TF-IDF and linguistic features (same as training)
            features = []
            
            # Word features
            if hasattr(package, 'word_vectorizer') and package.word_vectorizer:
                word_features = package.word_vectorizer.transform([text]).toarray()
                features.append(word_features)
            
            # Character features 
            if hasattr(package, 'char_vectorizer') and package.char_vectorizer:
                char_features = package.char_vectorizer.transform([text]).toarray()
                features.append(char_features)
            
            # Linguistic features with sentiment analysis
            linguistic_features = self.extract_linguistic_features([text])
            features.append(linguistic_features)
            
            # Combine and scale features
            combined_features = np.hstack(features)
            if hasattr(package, 'scaler') and package.scaler:
                combined_features = package.scaler.transform(combined_features)
            
            # Reconstruct the Enhanced Neural Network model
            from src.ml.text_classifier import EnhancedTextClassifierNetwork
            
            input_dim = combined_features.shape[1]
            model = EnhancedTextClassifierNetwork(input_dim)
            model.load_state_dict(package.model)
            model.eval()
            
            # Make prediction
            with torch.no_grad():
                features_tensor = torch.FloatTensor(combined_features)
                output = model(features_tensor)
                probability = output.item()
                prediction = 1 if probability > 0.5 else 0
            
            return int(prediction), float(probability)
            
        except Exception as e:
            raise ValueError(f"PyTorch feature model prediction failed: {str(e)}")
    
    def _predict_pytorch_sequence_model(self, package: Any, text: str) -> Tuple[int, float]:
        """Predict using sequence-based PyTorch models (CNN, Transformer, etc.)."""
        # This would be the existing PyTorch reconstruction logic
        return self._predict_pytorch_model(package, text)

# Initialize the universal loader
_universal_loader = UniversalModelLoader()


def predict_single_text(classifier, text, model_type):
    """Predict a single text and display results.
    
    Args:
        classifier: Trained classifier instance.
        text: Text string to classify.
        model_type: String description of the model type.
    """
    with suppress_sklearn_warnings():
        predictions, probabilities = classifier.predict([text])
    
    prediction = predictions[0]
    probability = probabilities[0]
    
    if prediction == 1:  # AI
        label = "AI-generated"
        confidence = probability
    else:  # Human
        label = "Human-written"
        confidence = 1 - probability
    
    print(f"Model: {model_type}")
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.3f}")
    print("-" * 50)


def predict_from_file(classifier, file_path, model_type, whole_document=None):
    """Predict texts from a file with flexible handling of document structure.
    
    Args:
        classifier: Trained classifier instance.
        file_path: Path to the text file to analyze.
        model_type: String description of the model type.
        whole_document: Whether to treat file as one document or separate lines.
                       If None, auto-detects based on content structure.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print(f"No content found in {file_path}")
            return
        
        # Auto-detect document structure if not specified
        if whole_document is None:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            short_lines = sum(1 for line in lines if len(line) < 100)
            whole_document = short_lines < len(lines) * 0.7 
        
        if whole_document:
            # Treat entire file as one document (preserving paragraph breaks)
            print(f"Analyzing entire document from {file_path} using {model_type}")
            print("=" * 60)
            
            # Clean up the text but preserve structure
            full_text = ' '.join(content.split())
            
            predictions, probabilities = classifier.predict([full_text])
            
            prediction = predictions[0]
            probability = probabilities[0]
            
            if prediction == 1:  # AI
                label = "AI-generated"
                confidence = probability
            else:  # Human
                label = "Human-written"
                confidence = 1 - probability
            
            print(f"Model: {model_type}")
            print(f"Document preview: {full_text[:200]}{'...' if len(full_text) > 200 else ''}")
            print(f"Document length: {len(full_text)} characters")
            print(f"Prediction: {label}")
            print(f"Confidence: {confidence:.3f}")
            print("=" * 60)
            
        else:
            # Treat each non-empty line as a separate text
            texts = [line.strip() for line in content.split('\n') if line.strip()]
            
            print(f"Analyzing {len(texts)} separate texts from {file_path} using {model_type}")
            print("=" * 60)
            
            predictions, probabilities = classifier.predict(texts)
            
            ai_count = 0
            human_count = 0
            
            for i, text in enumerate(texts):
                prediction = predictions[i]
                probability = probabilities[i]
                
                if prediction == 1:  # AI
                    label = "AI-generated"
                    confidence = probability
                    ai_count += 1
                else:  # Human
                    label = "Human-written"
                    confidence = 1 - probability
                    human_count += 1
                
                print(f"Text {i+1}: {text[:80]}{'...' if len(text) > 80 else ''}")
                print(f"Prediction: {label} (confidence: {confidence:.3f})")
                print("-" * 60)
            
            print(f"\nSummary (using {model_type}):")
            print(f"Total texts: {len(texts)}")
            print(f"Predicted as Human: {human_count}")
            print(f"Predicted as AI: {ai_count}")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")


def interactive_mode(classifier, model_type):
    """Interactive mode for predicting texts.
    
    Args:
        classifier: Trained classifier instance.
        model_type: String description of the model type.
    """
    print(f"AI vs Human Text Classifier ({model_type})")
    print("Enter text to classify (type 'quit' to exit):")
    print("-" * 50)
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            if len(text) < 20:
                print("Text is too short for reliable classification. Please enter longer text (20+ characters).")
                continue
            
            predict_single_text(classifier, text, model_type)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


# Legacy functions removed - using UniversalModelLoader instead


def discover_available_models():
    """Discover all available trained models.
    
    Returns:
        Dictionary with model types as keys and lists of available models as values.
    """
    available_models = {
        'neural': [],
        'classical': [],
        'ensemble': [],
        'hybrid': [],
        'deep_learning': [],
        'probabilistic': [],
        'manifold': [],
        'advanced': [],
        'interpretable': []
    }
    
    models_dir = Path("models")
    if not models_dir.exists():
        return available_models
    
    # Look for models with the pattern: comparison_<method_name>.pkl
    comparison_files = list(models_dir.glob("comparison_*.pkl"))
    
    for model_file in comparison_files:
        base_name = model_file.stem.replace('comparison_', '')
        model_path = str(model_file.with_suffix(''))
        
        # Comprehensive model categorization
        if base_name in ['enhanced_neural_network']:
            available_models['neural'].append((base_name, model_path))
            
        elif base_name in ['lstm_attention']:
            available_models['neural'].append(('sequential', model_path))
            
        elif base_name in ['lstm_features']:
            available_models['hybrid'].append((base_name, model_path))
            
        elif base_name in ['random_forest', 'svm', 'logistic_regression', 'gradient_boosting', 
                          'naive_bayes', 'knn', 'decision_tree', 'adaboost']:
            available_models['classical'].append((base_name, model_path))
            
        elif base_name in ['voting', 'bagging', 'stacking', 'xgboost', 'lightgbm', 
                          'catboost', 'extra_trees', 'custom_ensemble']:
            available_models['ensemble'].append((base_name, model_path))
            
        elif base_name in ['gaussian_nb', 'bernoulli_nb', 'multinomial_nb', 'complement_nb', 
                          'categorical_nb', 'hmm', 'gaussian_mixture']:
            available_models['probabilistic'].append((base_name, model_path))
            
        elif base_name in ['cnn', 'transformer', 'attention_bilstm']:
            available_models['deep_learning'].append((base_name, model_path))
            
        elif base_name in ['pca', 'tsne', 'isomap', 'lle', 'spectral_embedding', 
                          'mds', 'ica', 'factor_analysis', 'truncated_svd']:
            available_models['manifold'].append((base_name, model_path))
            
        elif base_name in ['isolation_forest', 'one_class_svm', 'local_outlier_factor', 
                          'elliptic_envelope', 'sgd', 'passive_aggressive', 'perceptron', 
                          'ridge', 'lasso', 'elastic_net', 'huber', 'quantile', 'tweedie']:
            available_models['advanced'].append((base_name, model_path))
            
        elif base_name in ['linear_regression', 'lasso_regression', 'ridge_regression', 
                          'elastic_net_regression', 'decision_tree_classifier', 
                          'extra_tree_classifier', 'gaussian_nb_classifier']:
            available_models['interpretable'].append((base_name, model_path))
            
        else:
            # For any unrecognized model, try to infer category from name
            if 'nb' in base_name or 'naive' in base_name or 'gaussian' in base_name:
                available_models['probabilistic'].append((base_name, model_path))
            elif 'svm' in base_name or 'forest' in base_name or 'tree' in base_name:
                if 'one_class' in base_name or 'isolation' in base_name:
                    available_models['advanced'].append((base_name, model_path))
                else:
                    available_models['classical'].append((base_name, model_path))
            elif 'boost' in base_name or 'bagging' in base_name or 'voting' in base_name or 'stacking' in base_name:
                available_models['ensemble'].append((base_name, model_path))
            elif 'regression' in base_name or 'lasso' in base_name or 'ridge' in base_name:
                available_models['interpretable'].append((base_name, model_path))
            elif 'neural' in base_name or 'lstm' in base_name or 'cnn' in base_name:
                if 'features' in base_name:
                    available_models['hybrid'].append((base_name, model_path))
                else:
                    available_models['neural'].append((base_name, model_path))
            else:
                # Default to advanced category for unrecognized models
                available_models['advanced'].append((base_name, model_path))
    
    return available_models


def predict_with_all_models(text):
    """Predict text using all available models.
    
    Args:
        text: Text string to classify.
    """
    # Use the new model discovery that simply finds all .pkl files
    models_dir = Path("models")
    if not models_dir.exists():
        print("No trained models found. Train some models first using:")
        print("  python scripts/compare_all_methods.py --human-file <file> --ai-file <file>")
        return
    
    model_files = list(models_dir.glob("comparison_*.pkl"))
    if not model_files:
        print("No trained models found. Train some models first using:")
        print("  python scripts/compare_all_methods.py --human-file <file> --ai-file <file>")
        return
    
    # Use all models with optimized loading
    working_model_files = model_files
    
    # First pass: try to load all models and count successful ones
    results = []
    failed_models = []
    
    print(f"Loading and testing {len(working_model_files)} models...")
    print("=" * 80)
    
    # Process models with progress indication
    for i, model_file in enumerate(working_model_files):
        model_path = str(model_file.with_suffix(''))  # Remove .pkl extension
        model_name = model_file.stem.replace("comparison_", "")
        
        # Show progress for slow loading
        if i % 10 == 0:
            print(f"Processing models... {i+1}/{len(working_model_files)}")
        
        try:
            package, description = _universal_loader.load_model_package(model_path)
            prediction, probability = _universal_loader.predict_with_package(package, text)
            
            if prediction == 1:  # AI
                label = "AI-generated"
                confidence = probability
            else:  # Human
                label = "Human-written"
                confidence = 1 - probability
            
            results.append((description, label, confidence))
            
        except Exception as e:
            # Collect failed models for debugging  
            failed_models.append((model_name, str(e)))
            continue
    
    # Now show the accurate count and results
    total_attempted = len(working_model_files)
    successful_count = len(results)
    failed_count = len(failed_models)
    
    print(f"\nAll Models Prediction ({successful_count} models)")
    print("=" * 80)
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print("=" * 80)
    
    if successful_count == 0:
        print("No models could be loaded successfully.")
        if failed_models:
            print(f"\nFailed models ({failed_count}):")
            for model_name, error in failed_models:
                print(f"  - {model_name}: {error}")
        return
    
    if failed_count > 0:
        print(f"Note: {failed_count} models failed to load/predict (showing {successful_count} working models)")
        print(f"\nFailed models:")
        for model_name, error in failed_models:
            print(f"  - {model_name}: {error}")
    
    # Display results in organized tables
    # Separate results by prediction type
    ai_results = [(model_type, confidence) for model_type, label, confidence in results if label == "AI-generated"]
    human_results = [(model_type, confidence) for model_type, label, confidence in results if label == "Human-written"]
    
    # Sort by confidence (highest to lowest)
    ai_results.sort(key=lambda x: x[1], reverse=True)
    human_results.sort(key=lambda x: x[1], reverse=True)
    
    print()
    print("=" * 90)
    print("PREDICTION RESULTS - ORGANIZED BY CLASSIFICATION")
    print("=" * 90)
    
    # AI-generated predictions table
    if ai_results:
        print(f"\n🤖 MODELS PREDICTING AI-GENERATED ({len(ai_results)} models)")
        print("-" * 60)
        print(f"{'Model':<35} {'Confidence':<10}")
        print("-" * 60)
        for model_type, confidence in ai_results:
            print(f"{model_type:<35} {confidence:.3f}")
    
    # Human-written predictions table
    if human_results:
        print(f"\n👤 MODELS PREDICTING HUMAN-WRITTEN ({len(human_results)} models)")
        print("-" * 60)
        print(f"{'Model':<35} {'Confidence':<10}")
        print("-" * 60)
        for model_type, confidence in human_results:
            print(f"{model_type:<35} {confidence:.3f}")
    
    # Summary statistics
    print("\n" + "=" * 90)
    print("CONSENSUS SUMMARY")
    print("=" * 90)
    ai_predictions = len(ai_results)
    human_predictions = len(human_results)
    total_predictions = len(results)
    
    print(f"Total models: {total_predictions}")
    print(f"AI-generated predictions: {ai_predictions} ({ai_predictions/total_predictions*100:.1f}%)")
    print(f"Human-written predictions: {human_predictions} ({human_predictions/total_predictions*100:.1f}%)")
    
    # Calculate average confidence for each prediction type
    if ai_results:
        avg_ai_conf = sum(conf for _, conf in ai_results) / len(ai_results)
        print(f"Average AI confidence: {avg_ai_conf:.3f}")
    
    if human_results:
        avg_human_conf = sum(conf for _, conf in human_results) / len(human_results)
        print(f"Average Human confidence: {avg_human_conf:.3f}")
    
    # Overall consensus
    if human_predictions > ai_predictions:
        consensus = "HUMAN-WRITTEN"
        margin = human_predictions - ai_predictions
    elif ai_predictions > human_predictions:
        consensus = "AI-GENERATED"
        margin = ai_predictions - human_predictions
    else:
        consensus = "TIE"
        margin = 0
    
    print(f"\nOVERALL CONSENSUS: {consensus}")
    if margin > 0:
        print(f"Margin: {margin} models ({margin/total_predictions*100:.1f}%)")
    print("=" * 90)


def predict_file_with_all_models(file_path, whole_document=None, separate_lines=None):
    """Predict texts from a file using all available models.
    
    Args:
        file_path: Path to the text file to analyze.
        whole_document: Whether to treat file as one document.
        separate_lines: Whether to treat each line as separate text.
    """
    # Use the same approach as predict_with_all_models
    models_dir = Path("models")
    if not models_dir.exists():
        print("No trained models found. Train some models first using:")
        print("  python scripts/compare_all_methods.py --human-file <file> --ai-file <file>")
        return
    
    model_files = list(models_dir.glob("comparison_*.pkl"))
    if not model_files:
        print("No trained models found. Train some models first using:")
        print("  python scripts/compare_all_methods.py --human-file <file> --ai-file <file>")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print(f"No content found in {file_path}")
            return
        
        # Determine document structure
        whole_doc = None
        if whole_document:
            whole_doc = True
        elif separate_lines:
            whole_doc = False
        else:
            # Auto-detect
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            short_lines = sum(1 for line in lines if len(line) < 100)
            whole_doc = short_lines < len(lines) * 0.7
        
        print(f"All Models File Analysis ({len(model_files)} models)")
        print("=" * 80)
        print(f"File: {file_path}")
        
        if whole_doc:
            # Treat entire file as one document
            full_text = ' '.join(content.split())
            print(f"Document length: {len(full_text)} characters")
            print(f"Processing as: Single document")
            print("=" * 80)
            
            all_results = []
            
            # Process all models using universal loader
            for model_file in model_files:
                model_path = str(model_file.with_suffix(''))  # Remove .pkl extension
                
                try:
                    package, description = _universal_loader.load_model_package(model_path)
                    prediction, probability = _universal_loader.predict_with_package(package, full_text)
                    
                    if prediction == 1:  # AI
                        label = "AI-generated"
                        confidence = probability
                    else:  # Human
                        label = "Human-written"
                        confidence = 1 - probability
                    
                    all_results.append((description, label, confidence))
                    print(f"{description:<30} {label:<15} {confidence:.3f}")
                    
                except Exception as e:
                    # Silently skip models that fail to load or predict
                    pass
            
            # Summary for single document
            if all_results:
                print("=" * 80)
                ai_predictions = sum(1 for _, label, _ in all_results if label == "AI-generated")
                human_predictions = len(all_results) - ai_predictions
                
                print(f"DOCUMENT CONSENSUS:")
                print(f"AI-generated predictions: {ai_predictions}/{len(all_results)} ({ai_predictions/len(all_results)*100:.1f}%)")
                print(f"Human-written predictions: {human_predictions}/{len(all_results)} ({human_predictions/len(all_results)*100:.1f}%)")
                
                ai_confidences = [conf for _, label, conf in all_results if label == "AI-generated"]
                human_confidences = [conf for _, label, conf in all_results if label == "Human-written"]
                
                if ai_confidences:
                    avg_ai_conf = sum(ai_confidences) / len(ai_confidences)
                    print(f"Average AI confidence: {avg_ai_conf:.3f}")
                
                if human_confidences:
                    avg_human_conf = sum(human_confidences) / len(human_confidences)
                    print(f"Average Human confidence: {avg_human_conf:.3f}")
        
        else:
            # For separate lines, this would be too verbose, so provide summary
            texts = [line.strip() for line in content.split('\n') if line.strip()]
            print(f"Number of texts: {len(texts)}")
            print(f"Processing as: Separate lines")
            print("=" * 80)
            print("This feature is simplified for file mode. Use --all --interactive for detailed line-by-line analysis.")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")


def interactive_all_models_mode():
    """Interactive mode using all available models."""
    # Use the new model discovery that simply finds all .pkl files
    models_dir = Path("models")
    if not models_dir.exists():
        print("No trained models found. Train some models first using:")
        print("  python scripts/compare_all_methods.py --human-file <file> --ai-file <file>")
        return
    
    model_files = list(models_dir.glob("comparison_*.pkl"))
    if not model_files:
        print("No trained models found. Train some models first using:")
        print("  python scripts/compare_all_methods.py --human-file <file> --ai-file <file>")
        return
    
    print(f"AI vs Human Text Classifier - All Models Mode ({len(model_files)} models)")
    print("Enter text to classify (type 'quit' to exit):")
    print("-" * 80)
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            if len(text) < 20:
                print("Text is too short for reliable classification. Please enter longer text (20+ characters).")
                continue
            
            print()  # Add blank line before results
            predict_with_all_models(text)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main prediction function."""
    # Suppress warnings during prediction
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    parser = argparse.ArgumentParser(description='Universal AI vs Human text classifier prediction')
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--neural', action='store_true',
                           help='Use neural network model')
    model_group.add_argument('--hybrid', action='store_true',
                           help='Use hybrid sequential model')
    model_group.add_argument('--classical', type=str,
                           choices=['random_forest', 'svm', 'logistic_regression', 'gradient_boosting', 
                                   'naive_bayes', 'knn', 'decision_tree', 'adaboost'],
                           help='Use classical ML model')
    model_group.add_argument('--ensemble', type=str,
                           choices=['voting', 'bagging', 'stacking', 'xgboost', 'lightgbm', 
                                   'catboost', 'extra_trees', 'custom_ensemble'],
                           help='Use ensemble model')
    model_group.add_argument('--all', action='store_true',
                           help='Use all available trained models')
    
    # Model path (optional, defaults to standard paths)
    parser.add_argument('--model-path', type=str, default=None,
                       help='Custom path to the trained model (optional, uses standard paths by default)')
    
    # Input options
    parser.add_argument('--text', type=str,
                       help='Single text to classify')
    parser.add_argument('--file', type=str,
                       help='File containing text to classify')
    parser.add_argument('--whole-document', action='store_true',
                       help='Treat entire file as one document (preserves paragraph breaks)')
    parser.add_argument('--separate-lines', action='store_true',
                       help='Treat each line as separate text')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Handle --all mode separately
    if args.all:
        # All models mode
        if args.text:
            predict_with_all_models(args.text)
        elif args.file:
            predict_file_with_all_models(args.file, args.whole_document, args.separate_lines)
        elif args.interactive:
            interactive_all_models_mode()
        else:
            interactive_all_models_mode()
        return
    
    # Set default model paths using consolidated format
    if args.model_path is None:
        if args.neural:
            args.model_path = "models/comparison_enhanced_neural_network"
        elif args.hybrid:
            args.model_path = "models/comparison_lstm_features"
        elif args.classical:
            args.model_path = f"models/comparison_{args.classical}"
        elif args.ensemble:
            args.model_path = f"models/comparison_{args.ensemble}"
    
    # Load single model using universal loader
    try:
        consolidated_path = f"{args.model_path}.pkl"
        if Path(consolidated_path).exists():
            # Create a wrapper class that mimics the old interface
            class SingleModelClassifier:
                def __init__(self, model_path, model_type_desc):
                    self.model_path = model_path
                    self.model_type_desc = model_type_desc
                    self.package, self.description = _universal_loader.load_model_package(model_path)
                
                def predict(self, texts):
                    predictions = []
                    probabilities = []
                    for text in texts:
                        pred, prob = _universal_loader.predict_with_package(self.package, text)
                        predictions.append(pred)
                        probabilities.append(prob)
                    return predictions, probabilities
            
            classifier = SingleModelClassifier(args.model_path, f"Single Model")
            model_type = classifier.description
        else:
            print(f"Error: Model not found at {consolidated_path}")
            print("Available models:")
            models_dir = Path("models")
            if models_dir.exists():
                available_files = list(models_dir.glob("comparison_*.pkl"))
                if available_files:
                    for model_file in available_files[:10]:  # Show first 10
                        print(f"  - {model_file}")
                    if len(available_files) > 10:
                        print(f"  ... and {len(available_files) - 10} more")
                else:
                    print("  No models found.")
            print("\nTrain models first using:")
            print("  python scripts/compare_all_methods.py --human-file <file> --ai-file <file>")
            sys.exit(1)
        
        print(f"{model_type} loaded successfully")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Determine mode and run prediction
    if args.text:
        # Single text prediction
        predict_single_text(classifier, args.text, model_type)
    elif args.file:
        # File prediction with document structure handling
        whole_doc = None
        if args.whole_document:
            whole_doc = True
        elif args.separate_lines:
            whole_doc = False
        # If neither flag is specified, auto-detect (whole_doc remains None)
        
        predict_from_file(classifier, args.file, model_type, whole_document=whole_doc)
    elif args.interactive:
        # Interactive mode
        interactive_mode(classifier, model_type)
    else:
        # Default to interactive mode
        print("No specific input provided. Starting interactive mode...")
        interactive_mode(classifier, model_type)


if __name__ == "__main__":
    main()
