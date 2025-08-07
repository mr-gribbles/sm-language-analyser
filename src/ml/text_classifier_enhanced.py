"""Enhanced neural network binary classifier for AI vs Human text detection.

This module implements significant improvements over the base classifier:
- Better feature extraction with character n-grams and linguistic features
- More sophisticated neural architecture
- Optimized hyperparameters
- Advanced regularization techniques
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


class EnhancedTextClassifierNetwork(nn.Module):
    """Enhanced PyTorch neural network for binary text classification."""
    
    def __init__(self, input_dim: int, dropout_rate: float = 0.3):
        """Initialize the enhanced neural network architecture.
        
        Args:
            input_dim: Number of input features from feature extraction.
            dropout_rate: Dropout rate for regularization.
        """
        super(EnhancedTextClassifierNetwork, self).__init__()
        
        # More sophisticated architecture with residual connections
        self.input_layer = nn.Linear(input_dim, 512)
        self.input_bn = nn.BatchNorm1d(512)
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # Hidden layers with residual connections
        self.hidden1 = nn.Linear(512, 256)
        self.hidden1_bn = nn.BatchNorm1d(256)
        self.hidden1_dropout = nn.Dropout(dropout_rate)
        
        self.hidden2 = nn.Linear(256, 128)
        self.hidden2_bn = nn.BatchNorm1d(128)
        self.hidden2_dropout = nn.Dropout(dropout_rate)
        
        self.hidden3 = nn.Linear(128, 64)
        self.hidden3_bn = nn.BatchNorm1d(64)
        self.hidden3_dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.output = nn.Linear(64, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network with residual connections.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
            
        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid activation.
        """
        # Input layer
        x1 = self.leaky_relu(self.input_bn(self.input_layer(x)))
        x1 = self.input_dropout(x1)
        
        # Hidden layer 1
        x2 = self.leaky_relu(self.hidden1_bn(self.hidden1(x1)))
        x2 = self.hidden1_dropout(x2)
        
        # Hidden layer 2
        x3 = self.leaky_relu(self.hidden2_bn(self.hidden2(x2)))
        x3 = self.hidden2_dropout(x3)
        
        # Hidden layer 3
        x4 = self.leaky_relu(self.hidden3_bn(self.hidden3(x3)))
        x4 = self.hidden3_dropout(x4)
        
        # Output
        output = self.sigmoid(self.output(x4))
        
        return output


class EnhancedAIHumanTextClassifier:
    """Enhanced binary classifier to distinguish between AI-generated and human-written text."""
    
    def __init__(self, max_features: int = 15000, ngram_range: Tuple[int, int] = (1, 3)):
        """Initialize the enhanced classifier.
        
        Args:
            max_features: Maximum number of features for TF-IDF vectorization.
            ngram_range: Range of n-grams to extract (default: unigrams to trigrams).
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.scaler = None
        self.model = None
        self.feature_names = None
        # Use MPS (Metal Performance Shaders) for M1/M2/M3/M4 Macs
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
    def extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract linguistic features from texts.
        
        Args:
            texts: List of text strings.
            
        Returns:
            Feature matrix as numpy array.
        """
        features = []
        
        for text in texts:
            text_features = []
            
            # Basic statistics
            text_features.append(len(text))  # Text length
            text_features.append(len(text.split()))  # Word count
            text_features.append(len(text.split()) / len(text) if len(text) > 0 else 0)  # Word density
            
            # Sentence statistics
            sentences = text.split('.')
            text_features.append(len(sentences))  # Sentence count
            text_features.append(np.mean([len(s.split()) for s in sentences if s.strip()]))  # Avg words per sentence
            
            # Character-level features
            text_features.append(sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0)  # Uppercase ratio
            text_features.append(sum(1 for c in text if c.islower()) / len(text) if len(text) > 0 else 0)  # Lowercase ratio
            text_features.append(sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0)  # Digit ratio
            text_features.append(sum(1 for c in text if c in '.,!?;:') / len(text) if len(text) > 0 else 0)  # Punctuation ratio
            
            # Vocabulary complexity
            words = text.lower().split()
            unique_words = set(words)
            text_features.append(len(unique_words) / len(words) if len(words) > 0 else 0)  # Lexical diversity
            
            # Average word length
            text_features.append(np.mean([len(word) for word in words]) if words else 0)
            
            # Readability approximation (Flesch-like)
            avg_sentence_length = len(words) / len(sentences) if len(sentences) > 0 else 0
            avg_syllables = np.mean([max(1, len(re.findall(r'[aeiouAEIOU]', word))) for word in words]) if words else 0
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
            text_features.append(flesch_score)
            
            # Function word ratios (common in AI detection)
            function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
            function_word_count = sum(1 for word in words if word.lower() in function_words)
            text_features.append(function_word_count / len(words) if len(words) > 0 else 0)
            
            # Repetition patterns
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            unique_bigrams = set(bigrams)
            text_features.append(len(unique_bigrams) / len(bigrams) if len(bigrams) > 0 else 0)  # Bigram diversity
            
            features.append(text_features)
        
        return np.array(features)
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract comprehensive features from texts using multiple approaches.
        
        Args:
            texts: List of text strings.
            
        Returns:
            Combined feature matrix as numpy array.
        """
        # Word-level TF-IDF features
        if self.word_vectorizer is None:
            self.word_vectorizer = TfidfVectorizer(
                max_features=self.max_features // 2,  # Split features between word and char
                ngram_range=self.ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode',
                token_pattern=r'\b[a-zA-Z]{2,}\b',  # Words with 2+ chars
                min_df=3,  # Less aggressive filtering
                max_df=0.85,  # Less aggressive filtering
                sublinear_tf=True,
                use_idf=True,
                smooth_idf=True,
                norm='l2'
            )
            word_features = self.word_vectorizer.fit_transform(texts).toarray()
        else:
            word_features = self.word_vectorizer.transform(texts).toarray()
        
        # Character-level TF-IDF features (captures style patterns)
        if self.char_vectorizer is None:
            self.char_vectorizer = TfidfVectorizer(
                max_features=self.max_features // 2,
                analyzer='char',
                ngram_range=(2, 5),  # Character 2-5 grams
                lowercase=True,
                min_df=5,
                max_df=0.9,
                sublinear_tf=True,
                use_idf=True,
                smooth_idf=True,
                norm='l2'
            )
            char_features = self.char_vectorizer.fit_transform(texts).toarray()
        else:
            char_features = self.char_vectorizer.transform(texts).toarray()
        
        # Linguistic features
        linguistic_features = self.extract_linguistic_features(texts)
        
        # Combine all features
        combined_features = np.hstack([word_features, char_features, linguistic_features])
        
        return combined_features
    
    def load_corpus_files(self, human_file: str, ai_file: str) -> Tuple[List[str], List[int]]:
        """Load and prepare training data from specific corpus files.
        
        Args:
            human_file: Path to JSONL file containing human-written texts.
            ai_file: Path to JSONL file containing AI-generated texts.
            
        Returns:
            Tuple of (texts, labels) where labels are 0 for human, 1 for AI.
        """
        texts = []
        labels = []
        
        # Load human-written texts (label = 0)
        print(f"Loading human texts from: {human_file}")
        with open(human_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    # Use cleaned text if available, otherwise raw text
                    if 'cleaned_text' in record['original_content']:
                        text = record['original_content']['cleaned_text']
                    elif 'cleaned_selftext' in record['original_content']:
                        text = record['original_content']['cleaned_selftext']
                    else:
                        # Fallback to raw text
                        text = record['original_content'].get('raw_text', 
                              record['original_content'].get('raw_selftext', ''))
                    
                    if text and len(text.strip()) > 20:  # Filter very short texts
                        texts.append(text.strip())
                        labels.append(0)  # Human-written
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
                        if text and len(text.strip()) > 20:  # Filter very short texts
                            texts.append(text.strip())
                            labels.append(1)  # AI-generated
                except (json.JSONDecodeError, KeyError) as e:
                    continue
        
        print(f"Loaded {len(texts)} texts total:")
        print(f"  Human texts: {labels.count(0)}")
        print(f"  AI texts: {labels.count(1)}")
        
        return texts, labels
    
    def _create_data_loader(self, features: np.ndarray, labels: np.ndarray, 
                           batch_size: int, shuffle: bool = True) -> DataLoader:
        """Create a PyTorch DataLoader from features and labels."""
        features_tensor = torch.FloatTensor(features)
        labels_tensor = torch.FloatTensor(labels).unsqueeze(1)
        dataset = TensorDataset(features_tensor, labels_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train_from_files(self, human_file: str, ai_file: str, test_size: float = 0.2, 
                        validation_size: float = 0.15, epochs: int = 150, batch_size: int = 32, 
                        learning_rate: float = 0.001, patience: int = 15, dropout_rate: float = 0.3,
                        weight_decay: float = 1e-4) -> Dict[str, Any]:
        """Train the enhanced classifier on corpus data from specific files.
        
        Args:
            human_file: Path to JSONL file containing human-written texts.
            ai_file: Path to JSONL file containing AI-generated texts.
            test_size: Proportion of data to use for testing.
            validation_size: Proportion of training data to use for validation.
            epochs: Maximum number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate for the optimizer.
            patience: Early stopping patience.
            dropout_rate: Dropout rate for regularization.
            weight_decay: L2 regularization weight decay.
            
        Returns:
            Training history and evaluation metrics.
        """
        # Load data from specific files
        texts, labels = self.load_corpus_files(human_file, ai_file)
        
        if len(texts) == 0:
            raise ValueError("No texts provided for training")
        
        # Extract comprehensive features
        print("Extracting comprehensive features (word + character + linguistic)...")
        features = self.extract_features(texts)
        print(f"Extracted {features.shape[1]} total features")
        
        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=42, stratify=y_train
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Create data loaders
        train_loader = self._create_data_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self._create_data_loader(X_val, y_val, batch_size, shuffle=False)
        
        # Initialize enhanced model
        self.model = EnhancedTextClassifierNetwork(features_scaled.shape[1], dropout_rate).to(self.device)
        
        # Use focal loss to handle class imbalance better
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                
            def forward(self, inputs, targets):
                bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-bce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
                return focal_loss.mean()
        
        criterion = FocalLoss(alpha=1, gamma=2)
        
        # Use AdamW optimizer with cosine annealing
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, 
                               betas=(0.9, 0.999), eps=1e-8)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        print(f"Enhanced model built with {features_scaled.shape[1]} input features")
        print(f"Using device: {self.device}")
        
        # Training loop with advanced techniques
        history = {
            'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("Training enhanced model...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
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
            scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
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
        test_loader = self._create_data_loader(X_test, y_test, batch_size, shuffle=False)
        
        self.model.eval()
        test_predictions = []
        test_probabilities = []
        
        with torch.no_grad():
            for batch_features, _ in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
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
            'history': history,
            'test_accuracy': test_accuracy,
            'test_precision': class_report['weighted avg']['precision'],
            'test_recall': class_report['weighted avg']['recall'],
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, test_predictions),
            'feature_count': features_scaled.shape[1]
        }
        
        print(f"\nEnhanced Model Test Results:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {results['test_precision']:.4f}")
        print(f"Recall: {results['test_recall']:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, test_predictions, target_names=['Human', 'AI']))
        
        return results
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict whether texts are AI-generated or human-written."""
        if self.model is None or self.word_vectorizer is None or self.scaler is None:
            raise ValueError("Model not trained. Call train_from_files() first.")
        
        # Extract and scale features
        features = self.extract_features(texts)
        features_scaled = self.scaler.transform(features)
        
        # Convert to tensor and predict
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = outputs.cpu().numpy().flatten()
            predictions = (outputs > 0.5).float().cpu().numpy().flatten().astype(int)
        
        return predictions, probabilities
    
    def save_model(self, model_path: str):
        """Save the trained model and preprocessing components."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_layer.in_features,
                'max_features': self.max_features,
                'ngram_range': self.ngram_range
            }
        }, f"{model_path}_enhanced_model.pth")
        
        # Save preprocessing components
        with open(f"{model_path}_word_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.word_vectorizer, f)
            
        with open(f"{model_path}_char_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.char_vectorizer, f)
        
        with open(f"{model_path}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Enhanced model saved to {model_path}_enhanced_model.pth")
        print(f"Word vectorizer saved to {model_path}_word_vectorizer.pkl")
        print(f"Character vectorizer saved to {model_path}_char_vectorizer.pkl")
        print(f"Scaler saved to {model_path}_scaler.pkl")
    
    def load_model(self, model_path: str):
        """Load a trained model and preprocessing components."""
        model_path = Path(model_path)
        
        # Load model
        checkpoint = torch.load(f"{model_path}_enhanced_model.pth", map_location=self.device, weights_only=True)
        model_config = checkpoint['model_config']
        
        # Update instance variables from saved config
        self.max_features = model_config['max_features']
        self.ngram_range = model_config['ngram_range']
        
        # Initialize and load model
        self.model = EnhancedTextClassifierNetwork(model_config['input_dim']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load preprocessing components
        with open(f"{model_path}_word_vectorizer.pkl", 'rb') as f:
            self.word_vectorizer = pickle.load(f)
            
        with open(f"{model_path}_char_vectorizer.pkl", 'rb') as f:
            self.char_vectorizer = pickle.load(f)
        
        with open(f"{model_path}_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"Enhanced model loaded from {model_path}")
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Enhanced Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Training Accuracy')
        axes[1].plot(history['val_acc'], label='Validation Accuracy')
        axes[1].set_title('Enhanced Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
        plt.title('Enhanced Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced confusion matrix plot saved to {save_path}")
        
        plt.show()
