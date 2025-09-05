"""Sequential text classifier using LSTM with trainable embeddings.
"""
import json
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from .model_serializer import ModelPackage, ModelSerializer

class SequentialClassifierNetwork(nn.Module):
    """LSTM with Attention for text classification using word embeddings."""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, n_layers=2, dropout=0.6):
        super(SequentialClassifierNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        out = self.dropout(context_vector)
        return self.fc(out)

class SequentialTextClassifier:
    """Sequential text classifier using an LSTM with Attention architecture."""
    def __init__(self, vocab_size=20000, max_len=512):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word_to_idx = {}
        self.model = None
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from a list of texts."""
        word_counts = Counter(word for text in texts for word in text.split())
        self.word_to_idx = {word: i + 1 for i, (word, _) in enumerate(word_counts.most_common(self.vocab_size - 1))}
        self.word_to_idx['<pad>'] = 0
        self.word_to_idx['<unk>'] = len(self.word_to_idx)

    def texts_to_sequences(self, texts: List[str]) -> List[torch.Tensor]:
        """Convert texts to sequences of indices."""
        return [torch.tensor([self.word_to_idx.get(word, self.word_to_idx['<unk>']) for word in text.split()]) for text in texts]

    def load_corpus_files(self, human_file: str, ai_file: str) -> Tuple[List[str], List[int]]:
        """Load and prepare training data from specific corpus files."""
        texts, labels = [], []
        with open(human_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    text = record['original_content'].get('cleaned_text') or record['original_content'].get('raw_text', '')
                    if text and len(text.strip().split()) > 10:
                        texts.append(text.strip())
                        labels.append(0)
                except (json.JSONDecodeError, KeyError):
                    continue
        with open(ai_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if record:
                        llm_transformation = record.get('llm_transformation')
                        if llm_transformation and llm_transformation.get('rewritten_text'):
                            text = llm_transformation['rewritten_text']
                            if text and len(text.strip().split()) > 10:
                                texts.append(text.strip())
                                labels.append(1)
                except (json.JSONDecodeError, KeyError) as e:
                    continue
        return texts, labels

    def train_from_files(self, human_file: str, ai_file: str, test_size: float = 0.2, validation_size: float = 0.15, epochs: int = 50):
        """Train the sequential classifier."""
        texts, labels = self.load_corpus_files(human_file, ai_file)
        self.build_vocab(texts)
        sequences = self.texts_to_sequences(texts)
        
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=self.word_to_idx['<pad>'])[:, :self.max_len]
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        dataset = TensorDataset(padded_sequences, labels)
        
        test_len = int(len(dataset) * test_size)
        val_len = int((len(dataset) - test_len) * validation_size)
        train_len = len(dataset) - test_len - val_len
        
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        self.model = SequentialClassifierNetwork(len(self.word_to_idx)).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=1e-4)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            self.model.train()
            for seq, label in train_loader:
                seq, label = seq.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(seq)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for seq, label in val_loader:
                    seq, label = seq.to(self.device), label.to(self.device)
                    outputs = self.model(seq)
                    val_loss += criterion(outputs, label).item()
            
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping.")
                    break
        
        self.model.load_state_dict(best_model_state)

        self.model.eval()
        test_preds, y_test = [], []
        with torch.no_grad():
            for seq, label in test_loader:
                seq = seq.to(self.device)
                outputs = self.model(seq)
                test_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                y_test.extend(label.cpu().numpy())
        
        test_predictions = (np.array(test_preds) > 0.5).astype(int)
        
        # Handle edge case where predictions might be perfect
        try:
            class_report = classification_report(y_test, test_predictions, target_names=['Human', 'AI'], output_dict=True)
        except ValueError:
            # Fallback for perfect predictions
            class_report = classification_report(y_test, test_predictions, output_dict=True)
        
        return {
            'test_accuracy': accuracy_score(y_test, test_predictions),
            'test_precision': class_report['weighted avg']['precision'],
            'test_recall': class_report['weighted avg']['recall'],
            'test_f1': class_report['weighted avg']['f1-score'],
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, test_predictions),
        }

    def save_model(self, model_path: str):
        """Save the trained model and preprocessing components using consolidated format."""
        package = ModelPackage('sequential')
        
        # Add the PyTorch model (ModelSerializer will handle state_dict extraction)
        package.add_model(self.model)
        
        # Add additional components as metadata
        package.metadata.update({
            'word_to_idx': self.word_to_idx,
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'model_architecture': 'SequentialClassifierNetwork'
        })
        
        return ModelSerializer.save_model_package(package, model_path)
    
    def save_model_with_metrics(self, model_path: str, performance_metrics: Dict[str, Any]):
        """Save the trained model with comprehensive performance metrics."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create model package
        package = ModelPackage('sequential')
        package.add_model(self.model)
        
        # Add configuration
        config = {
            'word_to_idx': self.word_to_idx,
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'model_architecture': 'SequentialClassifierNetwork'
        }
        package.add_config(config)
        
        # Add performance metrics
        package.add_performance_metrics(performance_metrics)
        
        # Save consolidated package
        return ModelSerializer.save_model_package(package, model_path)

    def load_model(self, model_path: str):
        """Load a trained model and preprocessing components."""
        try:
            # Try consolidated format first
            package = ModelSerializer.load_model_package(model_path)
            
            self.word_to_idx = package.metadata['word_to_idx']
            self.vocab_size = package.metadata['vocab_size']
            self.max_len = package.metadata['max_len']
            
            self.model = SequentialClassifierNetwork(len(self.word_to_idx)).to(self.device)
            self.model.load_state_dict(package.model)
            
        except (FileNotFoundError, KeyError):
            # Fallback to legacy format
            model_path = Path(model_path)
            checkpoint = torch.load(f"{model_path}_sequential_model.pth", map_location=self.device)
            
            self.word_to_idx = checkpoint['word_to_idx']
            self.vocab_size = checkpoint['vocab_size']
            self.max_len = checkpoint['max_len']
            
            self.model = SequentialClassifierNetwork(len(self.word_to_idx)).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict whether texts are AI-generated or human-written.
        
        Args:
            texts: List of text strings to classify.
            
        Returns:
            Tuple of (predictions, probabilities) where:
            - predictions: numpy array of 0s (human) and 1s (AI)
            - probabilities: numpy array of prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_from_files() first.")
        
        sequences = self.texts_to_sequences(texts)
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=self.word_to_idx['<pad>'])[:, :self.max_len]
        
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            seq = padded_sequences.to(self.device)
            outputs = self.model(seq)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            predictions.extend(preds)
            probabilities.extend(probs)
        
        return np.array(predictions), np.array(probabilities)

    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
        plt.title('Sequential Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sequential confusion matrix plot saved to {save_path}")
        
        plt.show()
