"""Create an Enhanced Neural Network Ensemble for AI vs Human Text Classification.

This script analyzes the best-performing neural network models and creates an
optimized ensemble that combines their strengths for maximum performance.

Based on evaluation results:
- neural_network_dropout_high: 91.7% accuracy  
- neural_network_dropout_low: 91.7% accuracy
- mlp_small: 91.7% accuracy
- neural_network_default: 91.0% accuracy
- neural_network_wide: 90.7% accuracy
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
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Suppress warnings
warnings.filterwarnings('ignore')

# Import from train_models
try:
    from train_models import FeatureExtractor
    print("Successfully imported FeatureExtractor from train_models")
except ImportError as e:
    print(f"Warning: Could not import from train_models: {e}")
    FeatureExtractor = None


class OptimizedNeuralNetwork(nn.Module):
    """Optimized neural network combining best features from top performers."""
    
    def __init__(self, input_dim: int, architecture_type: str = 'best_ensemble'):
        super(OptimizedNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.architecture_type = architecture_type
        
        if architecture_type == 'best_ensemble':
            # Combine features from top performers
            self.network = self._create_optimized_architecture(input_dim)
        elif architecture_type == 'dropout_optimized':
            # Based on dropout_high/low success (91.7%)
            self.network = self._create_dropout_optimized_architecture(input_dim)
        elif architecture_type == 'wide_deep_hybrid':
            # Combine wide (90.7%) and deep (90.3%) approaches
            self.network = self._create_wide_deep_hybrid_architecture(input_dim)
        else:
            # Default optimized architecture
            self.network = self._create_optimized_architecture(input_dim)
    
    def _create_optimized_architecture(self, input_dim: int):
        """Create optimized architecture combining best features."""
        return nn.Sequential(
            # First wide layer (from neural_network_wide success)
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),  # Higher dropout from successful models
            
            # Deep processing layers (from neural_network_deep)
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Final layers (from mlp_small success)
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output layer
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _create_dropout_optimized_architecture(self, input_dim: int):
        """Create architecture optimized for dropout (91.7% accuracy pattern)."""
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # High dropout like successful models
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def _create_wide_deep_hybrid_architecture(self, input_dim: int):
        """Create hybrid wide and deep architecture."""
        # Wide component
        self.wide_path = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256)
        )
        
        # Deep component
        self.deep_path = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64)
        )
        
        # Combine wide and deep
        self.combiner = nn.Sequential(
            nn.Linear(256 + 64, 128),  # wide_path output + deep_path output
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        return None  # Custom forward method handles this
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture_type == 'wide_deep_hybrid':
            wide_out = self.wide_path(x)
            deep_out = self.deep_path(x)
            combined = torch.cat([wide_out, deep_out], dim=1)
            return self.combiner(combined)
        else:
            return self.network(x)


class NeuralNetworkEnsemble:
    """Ensemble of multiple optimized neural networks."""
    
    def __init__(self, input_dim: int, ensemble_size: int = 5):
        self.input_dim = input_dim
        self.ensemble_size = ensemble_size
        self.models = []
        self.architectures = [
            'best_ensemble',
            'dropout_optimized', 
            'wide_deep_hybrid',
            'best_ensemble',  # Include best twice with different random seeds
            'dropout_optimized'
        ]
    
    def create_models(self):
        """Create ensemble of optimized models."""
        self.models = []
        for i in range(self.ensemble_size):
            arch_type = self.architectures[i % len(self.architectures)]
            model = OptimizedNeuralNetwork(self.input_dim, arch_type)
            self.models.append(model)
        print(f"Created ensemble of {len(self.models)} optimized neural networks")
    
    def train_ensemble(self, X_train: torch.Tensor, y_train: torch.Tensor,
                      X_val: torch.Tensor, y_val: torch.Tensor,
                      epochs: int = 200, patience: int = 20):
        """Train all models in the ensemble."""
        device = torch.device('mps' if torch.backends.mps.is_available() 
                             else 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training ensemble on device: {device}")
        
        trained_models = []
        
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{len(self.models)} ({self.architectures[i % len(self.architectures)]})...")
            
            model = model.to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            # Move data to device
            X_train_device = X_train.to(device)
            y_train_device = y_train.to(device)
            X_val_device = X_val.to(device)
            y_val_device = y_val.to(device)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                model.train()
                optimizer.zero_grad()
                
                outputs = model(X_train_device)
                loss = criterion(outputs, y_train_device)
                loss.backward()
                optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_device)
                    val_loss = criterion(val_outputs, y_val_device)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 50 == 0:
                    print(f"    Epoch {epoch+1}: train_loss={loss:.4f}, val_loss={val_loss:.4f}")
            
            # Load best model state
            model.load_state_dict(best_model_state)
            model.eval()
            trained_models.append(model.cpu())  # Move back to CPU for storage
            
            print(f"    ✓ Model {i+1} training completed")
        
        self.models = trained_models
        return trained_models
    
    def predict_ensemble(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions."""
        device = torch.device('mps' if torch.backends.mps.is_available() 
                             else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for model in self.models:
                model = model.to(device)
                model.eval()
                X_device = X.to(device)
                
                outputs = model(X_device).cpu().numpy().flatten()
                predictions = (outputs > 0.5).astype(int)
                
                all_predictions.append(predictions)
                all_probabilities.append(outputs)
        
        # Ensemble averaging
        ensemble_probabilities = np.mean(all_probabilities, axis=0)
        ensemble_predictions = (ensemble_probabilities > 0.5).astype(int)
        
        return ensemble_predictions, ensemble_probabilities
    
    def save_ensemble(self, save_path: str, feature_extractor=None):
        """Save the entire ensemble."""
        ensemble_data = {
            'model_type': 'neural_network_ensemble',
            'ensemble_size': len(self.models),
            'input_dim': self.input_dim,
            'architectures': self.architectures,
            'model_states': [model.state_dict() for model in self.models],
            'feature_extractor': feature_extractor
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        print(f"Ensemble saved to {save_path}")


def load_training_data(human_file: str, ai_file: str) -> Tuple[List[str], List[int]]:
    """Load training data from files."""
    print(f"Loading training data from {human_file} and {ai_file}")
    
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
                    
                    # Skip AI-generated data when looking for human text
                    if data.get('source') == 'llm_generated':
                        continue
                        
                    if 'original_content' in data:
                        if 'cleaned_text' in data['original_content']:
                            text = data['original_content']['cleaned_text'].strip()
                        elif 'cleaned_selftext' in data['original_content']:
                            title = data['original_content'].get('title', '').strip()
                            selftext = data['original_content']['cleaned_selftext'].strip()
                            if title and selftext:
                                text = f"{title}. {selftext}"
                            elif title:
                                text = title
                            elif selftext:
                                text = selftext
                    
                    if text and len(text) > 10:
                        texts.append(text)
                        labels.append(0)
                        human_count += 1
                        
                except (json.JSONDecodeError, KeyError) as e:
                    if line_num <= 10:
                        print(f"  Warning: Skipping malformed human text at line {line_num}: {e}")
                    continue
    
    # Load AI texts
    ai_count = 0
    if os.path.exists(ai_file):
        with open(ai_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    text = None
                    
                    # Extract AI text from rewritten_pairs format
                    if ('llm_transformation' in data and 
                        data['llm_transformation'] is not None and 
                        'rewritten_text' in data['llm_transformation']):
                        text = data['llm_transformation']['rewritten_text'].strip()
                    
                    # Extract AI text from llm_generated_social_media format
                    elif (data.get('source') == 'llm_generated' and 
                          'original_content' in data and 
                          'cleaned_text' in data['original_content']):
                        text = data['original_content']['cleaned_text'].strip()

                    if text and len(text) > 10:
                        texts.append(text)
                        labels.append(1)
                        ai_count += 1
                        
                except (json.JSONDecodeError, KeyError) as e:
                    if line_num <= 10:
                        print(f"  Warning: Skipping malformed AI text at line {line_num}: {e}")
                    continue
    
    print(f"Loaded {len(texts)} texts total ({human_count} human, {ai_count} AI)")
    return texts, labels


def create_and_train_ensemble():
    """Main function to create and train neural network ensemble."""
    print("=" * 80)
    print("NEURAL NETWORK ENSEMBLE CREATOR")
    print("Creating optimized ensemble from best performing neural networks")
    print("=" * 80)
    
    # Load training data
    human_file = "corpora/original_only/combined_original_only_20250827_100900.jsonl"
    ai_file = "corpora/rewritten_pairs/combined_rewritten_pairs_20250807_133424.jsonl"
    
    if not os.path.exists(human_file) or not os.path.exists(ai_file):
        print(f"Error: Training data files not found!")
        print(f"  Human file: {human_file}")
        print(f"  AI file: {ai_file}")
        return
    
    texts, labels = load_training_data(human_file, ai_file)
    
    if len(texts) == 0:
        print("No training data loaded!")
        return
    
    # Split data
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train_texts, X_val_texts, y_train, y_val = train_test_split(
        X_train_texts, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {len(X_train_texts)} samples")
    print(f"Validation set: {len(X_val_texts)} samples") 
    print(f"Test set: {len(X_test_texts)} samples")
    
    # Create feature extractor
    try:
        from scripts.enhanced_feature_extractor import create_enhanced_feature_extractor
        feature_extractor = create_enhanced_feature_extractor('default')
        print("Using enhanced feature extraction")
    except ImportError:
        print("Enhanced feature extractor not available, using basic extraction")
        if FeatureExtractor:
            feature_extractor = FeatureExtractor()
        else:
            print("No feature extractor available!")
            return
    
    # Extract features
    print("Extracting features...")
    X_train = feature_extractor.fit_transform(X_train_texts, y_train)
    X_val = feature_extractor.transform(X_val_texts)
    X_test = feature_extractor.transform(X_test_texts)
    
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Create and train ensemble
    print("\nCreating Neural Network Ensemble...")
    ensemble = NeuralNetworkEnsemble(input_dim=X_train.shape[1], ensemble_size=5)
    ensemble.create_models()
    
    print("\nTraining Ensemble...")
    trained_models = ensemble.train_ensemble(
        X_train_tensor, y_train_tensor, 
        X_val_tensor, y_val_tensor,
        epochs=200, patience=20
    )
    
    # Evaluate ensemble
    print("\nEvaluating Ensemble...")
    ensemble_pred, ensemble_prob = ensemble.predict_ensemble(X_test_tensor)
    
    accuracy = accuracy_score(y_test, ensemble_pred)
    print(f"\nEnsemble Test Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=['Human', 'AI']))
    
    # Save ensemble
    save_path = "models/neural_network_ensemble_optimized.pkl"
    ensemble.save_ensemble(save_path, feature_extractor)
    
    # Save individual best models with different configurations
    print("\nSaving individual optimized models...")
    
    architectures = ['best_ensemble', 'dropout_optimized', 'wide_deep_hybrid']
    for i, (model, arch_type) in enumerate(zip(trained_models[:3], architectures)):
        model_data = {
            'model_type': 'neural_network',
            'model_state_dict': model.state_dict(),
            'architecture_type': arch_type,
            'input_dim': X_train.shape[1],
            'feature_extractor': feature_extractor
        }
        
        model_path = f"models/neural_network_{arch_type}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"  Saved {arch_type} model to {model_path}")
    
    print("\n" + "=" * 80)
    print("NEURAL NETWORK ENSEMBLE CREATION COMPLETE!")
    print(f"Ensemble Accuracy: {accuracy:.4f}")
    print(f"Models saved to models/ directory")
    print("=" * 80)
    
    return ensemble, accuracy


if __name__ == "__main__":
    create_and_train_ensemble()
