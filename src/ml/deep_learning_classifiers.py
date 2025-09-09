"""Deep learning classifiers for AI vs Human text detection.

This module implements advanced deep learning architectures including CNN,
Transformer-based models, and attention mechanisms using the same pipeline
as other classifiers.
"""
import json
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from .model_serializer import ModelPackage, ModelSerializer


class TextCNN(nn.Module):
    """Convolutional Neural Network for text classification."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_filters: int = 100, 
                 filter_sizes: List[int] = [3, 4, 5], dropout_rate: float = 0.5):
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
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, num_filters, conv_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
        
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for text classification."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_heads: int = 8, 
                 num_layers: int = 3, ff_dim: int = 512, max_seq_len: int = 512, 
                 dropout_rate: float = 0.1):
        super(TransformerEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:seq_len, :].unsqueeze(0)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x)


class AttentionBiLSTM(nn.Module):
    """Bidirectional LSTM with attention mechanism."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout_rate: float = 0.3):
        super(AttentionBiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Attention weights
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch_size, seq_len, 1)
        
        # Weighted sum
        attended = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_dim * 2)
        
        x = self.dropout(attended)
        x = self.fc(x)
        return self.sigmoid(x)


class DeepLearningTextClassifier:
    """Deep learning classifier using various architectures."""
    
    def __init__(self, model_type: str = 'cnn', max_features: int = 10000, 
                 max_seq_len: int = 512, ngram_range: Tuple[int, int] = (1, 2)):
        """Initialize the deep learning classifier.
        
        Args:
            model_type: Type of model ('cnn', 'transformer', 'attention_bilstm')
            max_features: Maximum vocabulary size
            max_seq_len: Maximum sequence length
            ngram_range: Range of n-grams for fallback TF-IDF features
        """
        self.model_type = model_type
        self.max_features = max_features
        self.max_seq_len = max_seq_len
        self.ngram_range = ngram_range
        self.model = None
        self.tokenizer = None
        self.word_to_idx = None
        self.tfidf_vectorizer = None
        self.scaler = None
        
        # Device selection
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def _build_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """Build vocabulary from texts."""
        word_freq = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top max_features
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        for word, _ in sorted_words[:self.max_features - 2]:
            vocab[word] = len(vocab)
        
        return vocab
    
    def _texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        """Convert texts to sequences of token indices."""
        sequences = []
        for text in texts:
            words = text.lower().split()
            sequence = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
            
            # Pad or truncate to max_seq_len
            if len(sequence) > self.max_seq_len:
                sequence = sequence[:self.max_seq_len]
            else:
                sequence.extend([self.word_to_idx['<PAD>']] * (self.max_seq_len - len(sequence)))
            
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract linguistic features as fallback."""
        features = []
        
        for text in texts:
            text_features = []
            
            # Basic statistics
            text_features.append(len(text))
            text_features.append(len(text.split()))
            text_features.append(len(text.split()) / len(text) if len(text) > 0 else 0)
            
            # Sentence statistics
            sentences = text.split('.')
            text_features.append(len(sentences))
            text_features.append(np.mean([len(s.split()) for s in sentences if s.strip()]))
            
            # Character-level features
            text_features.append(sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0)
            text_features.append(sum(1 for c in text if c.islower()) / len(text) if len(text) > 0 else 0)
            text_features.append(sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0)
            text_features.append(sum(1 for c in text if c in '.,!?;:') / len(text) if len(text) > 0 else 0)
            
            # Vocabulary complexity
            words = text.lower().split()
            unique_words = set(words)
            text_features.append(len(unique_words) / len(words) if len(words) > 0 else 0)
            
            # Average word length
            text_features.append(np.mean([len(word) for word in words]) if words else 0)
            
            features.append(text_features)
        
        return np.array(features)
    
    def load_corpus_files(self, human_file: str, ai_file: str) -> Tuple[List[str], List[int]]:
        """Load and prepare training data."""
        texts = []
        labels = []
        
        # Load human-written texts (label = 0)
        print(f"Loading human texts from: {human_file}")
        with open(human_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if 'cleaned_text' in record['original_content']:
                        text = record['original_content']['cleaned_text']
                    elif 'cleaned_selftext' in record['original_content']:
                        text = record['original_content']['cleaned_selftext']
                    else:
                        text = record['original_content'].get('raw_text', 
                              record['original_content'].get('raw_selftext', ''))
                    
                    if text and len(text.strip()) > 20:
                        texts.append(text.strip())
                        labels.append(0)
                except (json.JSONDecodeError, KeyError) as e:
                    continue
        
        # Load AI-rewritten texts (label = 1)
        print(f"Loading AI texts from: {ai_file}")
        with open(ai_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if record.get('llm_transformation') and record['llm_transformation'].get('rewritten_text'):
                        text = record['llm_transformation']['rewritten_text']
                        if text and len(text.strip()) > 20:
                            texts.append(text.strip())
                            labels.append(1)
                except (json.JSONDecodeError, KeyError) as e:
                    continue
        
        print(f"Loaded {len(texts)} texts total:")
        print(f"  Human texts: {labels.count(0)}")
        print(f"  AI texts: {labels.count(1)}")
        
        return texts, labels
    
    def _get_model(self, vocab_size: int):
        """Get the specified model architecture."""
        if self.model_type == 'cnn':
            return TextCNN(vocab_size, embed_dim=128, num_filters=100, 
                          filter_sizes=[3, 4, 5], dropout_rate=0.5)
        elif self.model_type == 'transformer':
            return TransformerEncoder(vocab_size, embed_dim=128, num_heads=8, 
                                    num_layers=3, ff_dim=512, max_seq_len=self.max_seq_len)
        elif self.model_type == 'attention_bilstm':
            return AttentionBiLSTM(vocab_size, embed_dim=128, hidden_dim=128, 
                                 num_layers=2, dropout_rate=0.3)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train_from_files(self, human_file: str, ai_file: str, test_size: float = 0.2, 
                        validation_size: float = 0.15, epochs: int = 50, batch_size: int = 32, 
                        learning_rate: float = 0.001, patience: int = 10) -> Dict[str, Any]:
        """Train the deep learning classifier."""
        # Load data
        texts, labels = self.load_corpus_files(human_file, ai_file)
        
        if len(texts) == 0:
            raise ValueError("No texts provided for training")
        
        # Build vocabulary and convert texts to sequences
        print("Building vocabulary and converting texts to sequences...")
        self.word_to_idx = self._build_vocabulary(texts)
        sequences = self._texts_to_sequences(texts)
        
        print(f"Vocabulary size: {len(self.word_to_idx)}")
        print(f"Sequence shape: {sequences.shape}")
        
        # Also extract TF-IDF features as fallback
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=self.ngram_range,
                stop_words='english',
                lowercase=True
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()
        
        # Extract linguistic features
        linguistic_features = self.extract_linguistic_features(texts)
        
        # Combine traditional features
        traditional_features = np.hstack([tfidf_features, linguistic_features])
        
        # Scale traditional features
        self.scaler = StandardScaler()
        traditional_features_scaled = self.scaler.fit_transform(traditional_features)
        
        # Split data
        X_seq_train, X_seq_test, X_trad_train, X_trad_test, y_train, y_test = train_test_split(
            sequences, traditional_features_scaled, labels, test_size=test_size, 
            random_state=42, stratify=labels
        )
        
        X_seq_train, X_seq_val, X_trad_train, X_trad_val, y_train, y_val = train_test_split(
            X_seq_train, X_trad_train, y_train, test_size=validation_size, 
            random_state=42, stratify=y_train
        )
        
        print(f"Training set: {len(X_seq_train)} samples")
        print(f"Validation set: {len(X_seq_val)} samples")
        print(f"Test set: {len(X_seq_test)} samples")
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.LongTensor(X_seq_train),
            torch.FloatTensor(X_trad_train),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.LongTensor(X_seq_val),
            torch.FloatTensor(X_trad_val),
            torch.FloatTensor(y_val).unsqueeze(1)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = self._get_model(len(self.word_to_idx)).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        print(f"Training {self.model_type} model on {self.device}")
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for seq_batch, trad_batch, label_batch in train_loader:
                seq_batch = seq_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(seq_batch)
                loss = criterion(outputs, label_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += label_batch.size(0)
                train_correct += (predicted == label_batch).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for seq_batch, trad_batch, label_batch in val_loader:
                    seq_batch = seq_batch.to(self.device)
                    label_batch = label_batch.to(self.device)
                    
                    outputs = self.model(seq_batch)
                    loss = criterion(outputs, label_batch)
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += label_batch.size(0)
                    val_correct += (predicted == label_batch).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_dataset = TensorDataset(
            torch.LongTensor(X_seq_test),
            torch.FloatTensor(X_trad_test),
            torch.FloatTensor(y_test).unsqueeze(1)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        test_predictions = []
        test_probabilities = []
        
        with torch.no_grad():
            for seq_batch, trad_batch, label_batch in test_loader:
                seq_batch = seq_batch.to(self.device)
                outputs = self.model(seq_batch)
                probabilities = outputs.cpu().numpy().flatten()
                predictions = (outputs > 0.5).float().cpu().numpy().flatten()
                
                test_predictions.extend(predictions)
                test_probabilities.extend(probabilities)
        
        test_predictions = np.array(test_predictions, dtype=int)
        test_probabilities = np.array(test_probabilities)
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, test_predictions)
        class_report = classification_report(y_test, test_predictions, 
                                           target_names=['Human', 'AI'], output_dict=True)
        
        results = {
            'model_type': self.model_type,
            'history': history,
            'test_accuracy': test_accuracy,
            'test_precision': class_report['weighted avg']['precision'],
            'test_recall': class_report['weighted avg']['recall'],
            'test_f1': class_report['weighted avg']['f1-score'],
            'test_auc': roc_auc_score(y_test, test_probabilities),
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, test_predictions),
            'vocab_size': len(self.word_to_idx)
        }
        
        print(f"\n{self.model_type.title()} Model Test Results:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {results['test_precision']:.4f}")
        print(f"Recall: {results['test_recall']:.4f}")
        print(f"F1-score: {results['test_f1']:.4f}")
        print(f"AUC: {results['test_auc']:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, test_predictions, target_names=['Human', 'AI']))
        
        return results
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict whether texts are AI-generated or human-written."""
        if self.model is None or self.word_to_idx is None:
            raise ValueError("Model not trained. Call train_from_files() first.")
        
        # Convert texts to sequences
        sequences = self._texts_to_sequences(texts)
        
        # Convert to tensor and predict
        sequences_tensor = torch.LongTensor(sequences).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(sequences_tensor)
            probabilities = outputs.cpu().numpy().flatten()
            predictions = (outputs > 0.5).float().cpu().numpy().flatten().astype(int)
        
        return predictions, probabilities
    
    def save_model(self, model_path: str):
        """Save the trained model and preprocessing components."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        package = ModelPackage('deep_learning')
        package.add_model(self.model.state_dict())  # Save state_dict instead of full model
        
        # Add preprocessing components
        package.metadata.update({
            'model_type': self.model_type,
            'max_features': self.max_features,
            'max_seq_len': self.max_seq_len,
            'ngram_range': self.ngram_range,
            'word_to_idx': self.word_to_idx,
            'vocab_size': len(self.word_to_idx) if self.word_to_idx else 0,
            'model_architecture': 'DeepLearningTextClassifier'
        })
        
        # Add traditional feature components
        if self.tfidf_vectorizer is not None:
            package.add_word_vectorizer(self.tfidf_vectorizer)
        if self.scaler is not None:
            package.add_scaler(self.scaler)
        
        return ModelSerializer.save_model_package(package, model_path)
    
    def save_model_with_metrics(self, model_path: str, performance_metrics: Dict[str, Any]):
        """Save the trained model with comprehensive performance metrics."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create model package
        package = ModelPackage('deep_learning')
        package.add_model(self.model.state_dict())  # Save state_dict instead of full model
        
        # Add traditional feature components
        if self.tfidf_vectorizer is not None:
            package.add_word_vectorizer(self.tfidf_vectorizer)
        if self.scaler is not None:
            package.add_scaler(self.scaler)
        
        # Add configuration
        config = {
            'model_type': self.model_type,
            'max_features': self.max_features,
            'max_seq_len': self.max_seq_len,
            'ngram_range': self.ngram_range,
            'word_to_idx': self.word_to_idx,
            'vocab_size': len(self.word_to_idx) if self.word_to_idx else 0,
            'model_architecture': 'DeepLearningTextClassifier'
        }
        package.add_config(config)
        
        # Add performance metrics
        package.add_performance_metrics(performance_metrics)
        
        # Save consolidated package
        return ModelSerializer.save_model_package(package, model_path)
    
    def load_model(self, model_path: str):
        """Load a trained model and preprocessing components."""
        package = ModelSerializer.load_model_package(model_path, self.device)
        
        # Update instance variables from saved metadata
        self.model_type = package.metadata['model_type']
        self.max_features = package.metadata['max_features']
        self.max_seq_len = package.metadata['max_seq_len']
        self.ngram_range = package.metadata['ngram_range']
        self.word_to_idx = package.metadata['word_to_idx']
        
        # Initialize model
        vocab_size = package.metadata['vocab_size']
        self.model = self._get_model(vocab_size).to(self.device)
        
        # Handle both old format (full model object) and new format (state_dict)
        if isinstance(package.model, dict):
            # New format: state_dict
            self.model.load_state_dict(package.model)
        else:
            # Old format: full model object - extract state_dict
            if hasattr(package.model, 'state_dict'):
                self.model.load_state_dict(package.model.state_dict())
            else:
                # Fallback: try to use the model directly if it's already the right type
                if type(package.model).__name__ == type(self.model).__name__:
                    self.model = package.model.to(self.device)
                else:
                    raise ValueError(f"Cannot load model: expected state_dict or compatible model object, got {type(package.model)}")
        
        # Load preprocessing components
        self.tfidf_vectorizer = package.word_vectorizer
        self.scaler = package.scaler
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title(f'{self.model_type.title()} Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Training Accuracy')
        axes[1].plot(history['val_acc'], label='Validation Accuracy')
        axes[1].set_title(f'{self.model_type.title()} Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{self.model_type.title()} training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
        plt.title(f'{self.model_type.title()} Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{self.model_type.title()} confusion matrix plot saved to {save_path}")
        
        plt.show()


def compare_deep_learning_models(human_file: str, ai_file: str, models: List[str] = None, 
                               test_size: float = 0.2, validation_size: float = 0.15) -> Dict[str, Dict[str, Any]]:
    """Compare multiple deep learning models using the same data split."""
    if models is None:
        models = ['cnn', 'transformer', 'attention_bilstm']
    
    results = {}
    
    print("=== Comparing Deep Learning Models ===")
    print(f"Human file: {human_file}")
    print(f"AI file: {ai_file}")
    print(f"Models to test: {models}")
    print("=" * 60)
    
    for model_type in models:
        print(f"\nTraining {model_type}...")
        try:
            classifier = DeepLearningTextClassifier(
                model_type=model_type,
                max_features=10000,
                max_seq_len=512
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size,
                epochs=30,
                batch_size=32
            )
            
            results[model_type] = result
            
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("DEEP LEARNING MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Test Acc':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 80)
    
    for model_type, result in results.items():
        if 'error' not in result:
            test_acc = result['test_accuracy']
            precision = result['test_precision']
            recall = result['test_recall']
            f1 = result['test_f1']
            auc = result['test_auc']
            
            print(f"{model_type:<20} {test_acc:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {auc:<10.4f}")
        else:
            print(f"{model_type:<20} ERROR: {result['error']}")
    
    return results
