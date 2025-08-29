"""Hybrid text classifier using both sequential and feature-based inputs.
"""
import json
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from .model_serializer import ModelPackage, ModelSerializer

class HybridClassifierNetwork(nn.Module):
    """Hybrid model with LSTM for sequence and a feed-forward network for features."""
    def __init__(self, vocab_size, n_features, embedding_dim=128, hidden_dim=256, n_layers=2, dropout=0.5):
        super(HybridClassifierNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            bidirectional=True, dropout=dropout, batch_first=True)
        
        self.feature_processor = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.fc = nn.Linear(hidden_dim * 2 + 128, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, features):
        embedded = self.embedding(seq)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use the concatenated final hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        processed_features = self.feature_processor(features)
        
        combined = torch.cat((hidden, processed_features), dim=1)
        
        return self.fc(combined)

class HybridTextClassifier:
    """Hybrid text classifier using both sequential and feature-based inputs."""
    def __init__(self, vocab_size=20000, max_len=512, max_features=15000, ngram_range=(1, 3)):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.word_to_idx = {}
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.scaler = None
        self.model = None
        self.n_features = None
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def build_vocab(self, texts: List[str]):
        word_counts = Counter(word for text in texts for word in text.split())
        self.word_to_idx = {word: i + 1 for i, (word, _) in enumerate(word_counts.most_common(self.vocab_size - 1))}
        self.word_to_idx['<pad>'] = 0
        self.word_to_idx['<unk>'] = len(self.word_to_idx)

    def texts_to_sequences(self, texts: List[str]) -> List[torch.Tensor]:
        return [torch.tensor([self.word_to_idx.get(word, self.word_to_idx['<unk>']) for word in text.split()]) for text in texts]

    def extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        features = []
        for text in texts:
            text_features = []
            text_features.append(len(text))
            text_features.append(len(text.split()))
            words = text.lower().split()
            unique_words = set(words)
            text_features.append(len(unique_words) / len(words) if len(words) > 0 else 0)
            features.append(text_features)
        return np.array(features)

    def extract_features(self, texts: List[str]) -> np.ndarray:
        if self.word_vectorizer is None:
            self.word_vectorizer = TfidfVectorizer(max_features=self.max_features // 2, ngram_range=self.ngram_range, stop_words='english')
            word_features = self.word_vectorizer.fit_transform(texts).toarray()
        else:
            word_features = self.word_vectorizer.transform(texts).toarray()
        
        if self.char_vectorizer is None:
            self.char_vectorizer = TfidfVectorizer(max_features=self.max_features // 2, analyzer='char', ngram_range=(2, 5))
            char_features = self.char_vectorizer.fit_transform(texts).toarray()
        else:
            char_features = self.char_vectorizer.transform(texts).toarray()
        
        linguistic_features = self.extract_linguistic_features(texts)
        return np.hstack([word_features, char_features, linguistic_features])

    def load_corpus_files(self, human_file: str, ai_file: str) -> Tuple[List[str], List[int]]:
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
                except (json.JSONDecodeError, KeyError):
                    continue
        return texts, labels

    def train_from_files(self, human_file: str, ai_file: str, test_size: float = 0.2, validation_size: float = 0.15, epochs: int = 50):
        texts, labels = self.load_corpus_files(human_file, ai_file)
        self.build_vocab(texts)
        sequences = self.texts_to_sequences(texts)
        
        features = self.extract_features(texts)
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        self.n_features = features_scaled.shape[1]
        
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=self.word_to_idx['<pad>'])
        if padded_sequences.shape[1] > self.max_len:
            padded_sequences = padded_sequences[:, :self.max_len]
        else:
            padded_sequences = torch.nn.functional.pad(padded_sequences, (0, self.max_len - padded_sequences.shape[1]))
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        
        dataset = TensorDataset(padded_sequences, features_tensor, labels)
        
        test_len = int(len(dataset) * test_size)
        val_len = int((len(dataset) - test_len) * validation_size)
        train_len = len(dataset) - test_len - val_len
        
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        self.model = HybridClassifierNetwork(len(self.word_to_idx), self.n_features).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            self.model.train()
            for seq, feats, label in train_loader:
                seq, feats, label = seq.to(self.device), feats.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(seq, feats)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for seq, feats, label in val_loader:
                    seq, feats, label = seq.to(self.device), feats.to(self.device), label.to(self.device)
                    outputs = self.model(seq, feats)
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
            for seq, feats, label in test_loader:
                seq, feats = seq.to(self.device), feats.to(self.device)
                outputs = self.model(seq, feats)
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
        package = ModelPackage('hybrid')
        
        # Add the PyTorch model (ModelSerializer will handle state_dict extraction)
        package.add_model(self.model)
        
        # Add vectorizers and scaler
        package.add_word_vectorizer(self.word_vectorizer)
        package.add_char_vectorizer(self.char_vectorizer)
        package.add_scaler(self.scaler)
        
        # Add additional components as metadata
        package.metadata.update({
            'word_to_idx': self.word_to_idx,
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'n_features': self.n_features,
            'model_architecture': 'HybridClassifierNetwork'
        })
        
        return ModelSerializer.save_model_package(package, model_path)

    def load_model(self, model_path: str):
        """Load a trained model and preprocessing components."""
        try:
            # Try consolidated format first
            package = ModelSerializer.load_model_package(model_path)
            
            self.word_to_idx = package.metadata['word_to_idx']
            self.vocab_size = package.metadata['vocab_size']
            self.max_len = package.metadata['max_len']
            self.n_features = package.metadata['n_features']
            
            self.model = HybridClassifierNetwork(len(self.word_to_idx), self.n_features).to(self.device)
            self.model.load_state_dict(package.model)
            
            self.word_vectorizer = package.word_vectorizer
            self.char_vectorizer = package.char_vectorizer
            self.scaler = package.scaler
            
        except (FileNotFoundError, KeyError):
            # Fallback to legacy format
            model_path = Path(model_path)
            checkpoint = torch.load(f"{model_path}_hybrid_model.pth", map_location=self.device, weights_only=True)
            
            self.word_to_idx = checkpoint['word_to_idx']
            self.vocab_size = checkpoint['vocab_size']
            self.max_len = checkpoint['max_len']
            self.n_features = checkpoint['n_features']
            
            self.model = HybridClassifierNetwork(len(self.word_to_idx), self.n_features).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            with open(f"{model_path}_hybrid_word_vectorizer.pkl", 'rb') as f:
                self.word_vectorizer = pickle.load(f)
            with open(f"{model_path}_hybrid_char_vectorizer.pkl", 'rb') as f:
                self.char_vectorizer = pickle.load(f)
            with open(f"{model_path}_hybrid_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)

    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict whether texts are AI-generated or human-written."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_from_files() first.")

        sequences = self.texts_to_sequences(texts)
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=self.word_to_idx['<pad>'])
        if padded_sequences.shape[1] > self.max_len:
            padded_sequences = padded_sequences[:, :self.max_len]
        else:
            padded_sequences = torch.nn.functional.pad(padded_sequences, (0, self.max_len - padded_sequences.shape[1]))

        features = self.extract_features(texts)
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            seq = padded_sequences.to(self.device)
            feats = features_tensor.to(self.device)
            outputs = self.model(seq, feats)
            probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
            predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities

    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, save_path: Optional[str] = None):
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
        plt.title('Hybrid Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Hybrid confusion matrix plot saved to {save_path}")
        
        plt.show()
