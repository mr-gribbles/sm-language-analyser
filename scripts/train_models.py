"""Ultra-Comprehensive AI vs Human Text Classification Training Pipeline.

This script trains 50+ machine learning models suitable for binary text
classification, with consistent feature extraction, proper model saving, and
comprehensive performance evaluation.

PEP 8 compliant and production-ready.
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier, StackingClassifier, VotingClassifier
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import (
    ElasticNet, Lasso, LogisticRegression, PassiveAggressiveClassifier,
    Perceptron, Ridge, RidgeClassifier, SGDClassifier
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.naive_bayes import (
    BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
)
from sklearn.neighbors import (
    KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from torch.utils.data import DataLoader, TensorDataset


# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')


class FeatureExtractor:
    """Consistent feature extraction for all models."""
    
    def __init__(self, max_features: int = 10000, 
                 ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.scaler = None
        
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts to feature vectors."""
        print(f"Extracting features from {len(texts)} texts...")
        
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
        
        # Linguistic features
        linguistic_features = self._extract_linguistic_features(texts)
        
        # Combine all features
        combined_features = np.hstack([word_features, char_features, linguistic_features])
        
        # Scale features for neural networks and SVM
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(combined_features)
        
        print(f"Extracted {scaled_features.shape[1]} total features")
        return scaled_features
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted extractors."""
        if self.word_vectorizer is None:
            raise ValueError("Feature extractor not fitted. Call fit_transform first.")
        
        word_features = self.word_vectorizer.transform(texts).toarray()
        char_features = self.char_vectorizer.transform(texts).toarray()
        linguistic_features = self._extract_linguistic_features(texts)
        
        combined_features = np.hstack([word_features, char_features, linguistic_features])
        scaled_features = self.scaler.transform(combined_features)
        
        return scaled_features
    
    def _extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract linguistic features from texts."""
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
            text_features.append(word_count / max(text_len, 1))  # Word density
            
            # Sentence statistics
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            sentence_count = max(len(sentences), 1)
            text_features.append(sentence_count)
            # Avg words per sentence
            text_features.append(word_count / sentence_count)
            
            # Character-level features
            if text_len > 0:
                upper_ratio = sum(1 for c in text if c.isupper()) / text_len
                lower_ratio = sum(1 for c in text if c.islower()) / text_len
                digit_ratio = sum(1 for c in text if c.isdigit()) / text_len
                punct_ratio = sum(1 for c in text if c in '.,!?;:') / text_len
                text_features.extend([upper_ratio, lower_ratio, digit_ratio,
                                    punct_ratio])
            else:
                text_features.extend([0, 0, 0, 0])
            
            # Vocabulary complexity
            unique_words = set(words)
            # Lexical diversity
            text_features.append(len(unique_words) / max(word_count, 1))
            
            # Average word length
            if words:
                avg_word_len = np.mean([len(word) for word in words])
                text_features.append(np.clip(avg_word_len, 0, 50))
            else:
                text_features.append(0)
            
            # Readability approximation
            avg_sentence_length = word_count / sentence_count
            if words:
                syllable_counts = [
                    max(1, len(re.findall(r'[aeiouAEIOU]', word))) 
                    for word in words
                ]
                avg_syllables = np.mean(syllable_counts)
                flesch_score = (206.835 - (1.015 * avg_sentence_length) - 
                               (84.6 * avg_syllables))
                text_features.append(np.clip(flesch_score, -100, 200))
            else:
                text_features.append(0)
            
            features.append(text_features)
        
        return np.array(features)


class NeuralNetwork(nn.Module):
    """Base Neural Network for text classification."""
    
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


class ComprehensiveModelTrainer:
    """Ultra-comprehensive model training and evaluation."""
    
    def __init__(self, output_dir: str = "models", results_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        self.feature_extractor = FeatureExtractor()
        self.models = {}
        self.results = {}
        
    def load_data(self, human_file: str, ai_file: str) -> Tuple[List[str], List[int]]:
        """Load and prepare training data from the actual JSONL format."""
        print(f"Loading data from {human_file} and {ai_file}")
        
        texts = []
        labels = []
        
        # Load human texts (label 0) from both Bluesky and Reddit formats
        human_count = 0
        with open(human_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    text = None
                    
                    if 'original_content' in data:
                        # Bluesky format: use cleaned_text
                        if 'cleaned_text' in data['original_content']:
                            text = data['original_content']['cleaned_text'].strip()
                        # Reddit format: combine title and cleaned_selftext
                        elif 'cleaned_selftext' in data['original_content']:
                            title = data['original_content'].get('title', '').strip()
                            selftext = data['original_content']['cleaned_selftext'].strip()
                            # Combine title and selftext
                            if title and selftext:
                                text = f"{title}. {selftext}"
                            elif title:
                                text = title
                            elif selftext:
                                text = selftext
                    
                    if text and len(text) > 10:  # Filter out very short texts
                        texts.append(text)
                        labels.append(0)
                        human_count += 1
                        
                except (json.JSONDecodeError, KeyError) as e:
                    if line_num <= 10:  # Only show first 10 errors
                        print(f"  Warning: Skipping malformed human text at line {line_num}: {e}")
                    continue
        
        # Load AI texts (label 1) from llm_transformation.rewritten_text
        ai_count = 0
        with open(ai_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    # Extract AI text from llm_transformation.rewritten_text
                    if ('llm_transformation' in data and data['llm_transformation'] is not None 
                        and 'rewritten_text' in data['llm_transformation']):
                        text = data['llm_transformation']['rewritten_text'].strip()
                        if text and len(text) > 10:  # Filter out very short texts
                            texts.append(text)
                            labels.append(1)
                            ai_count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    if line_num <= 10:  # Only show first 10 errors
                        print(f"  Warning: Skipping malformed AI text at line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(texts)} texts total ({human_count} human, {ai_count} AI)")
        
        if len(texts) == 0:
            raise ValueError("No valid texts found! Check your data files and format.")
        
        if human_count == 0:
            raise ValueError(f"No human texts loaded from {human_file}! Check the file format.")
        
        if ai_count == 0:
            raise ValueError(f"No AI texts loaded from {ai_file}! Check the file format.")
        
        return texts, labels
    
    def prepare_all_models(self) -> Dict[str, Any]:
        """Prepare all 50+ models for training."""
        models = {}
        
        # ========== TREE-BASED MODELS ==========
        print("Preparing tree-based models...")
        
        # Random Forest variants
        models['random_forest_default'] = RandomForestClassifier(n_estimators=100, random_state=42)
        models['random_forest_large'] = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        models['random_forest_small'] = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        models['random_forest_balanced'] = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        
        # Extra Trees variants
        models['extra_trees_default'] = ExtraTreesClassifier(n_estimators=100, random_state=42)
        models['extra_trees_large'] = ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42)
        models['extra_tree_single'] = ExtraTreeClassifier(random_state=42)
        
        # Decision Tree variants
        models['decision_tree_default'] = DecisionTreeClassifier(random_state=42, max_depth=10)
        models['decision_tree_deep'] = DecisionTreeClassifier(random_state=42, max_depth=20)
        models['decision_tree_gini'] = DecisionTreeClassifier(random_state=42, criterion='gini')
        models['decision_tree_entropy'] = DecisionTreeClassifier(random_state=42, criterion='entropy')
        models['decision_tree_balanced'] = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        
        # Gradient Boosting variants
        models['gradient_boosting_default'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        models['gradient_boosting_large'] = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
        models['gradient_boosting_fast'] = GradientBoostingClassifier(n_estimators=50, learning_rate=0.2, random_state=42)
        models['hist_gradient_boosting'] = HistGradientBoostingClassifier(random_state=42)
        
        # AdaBoost variants
        models['adaboost_default'] = AdaBoostClassifier(n_estimators=50, random_state=42)
        models['adaboost_large'] = AdaBoostClassifier(n_estimators=100, learning_rate=0.8, random_state=42)
        models['adaboost_tree'] = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
            n_estimators=50, random_state=42
        )
        
        # ========== SVM VARIANTS ==========
        print("Preparing SVM models...")
        models['svm_rbf'] = SVC(probability=True, kernel='rbf', random_state=42)
        models['svm_linear'] = SVC(probability=True, kernel='linear', random_state=42)
        models['svm_poly'] = SVC(probability=True, kernel='poly', degree=3, random_state=42)
        models['svm_sigmoid'] = SVC(probability=True, kernel='sigmoid', random_state=42)
        models['linear_svc'] = LinearSVC(random_state=42, max_iter=2000)
        models['nu_svc'] = NuSVC(probability=True, random_state=42)
        
        # ========== LINEAR MODELS ==========
        print("Preparing linear models...")
        
        # Logistic Regression variants
        models['logistic_regression_l1'] = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
        models['logistic_regression_l2'] = LogisticRegression(penalty='l2', random_state=42, max_iter=1000)
        models['logistic_regression_elastic'] = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=42, max_iter=1000)
        models['logistic_regression_balanced'] = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        
        # Ridge variants
        models['ridge_classifier'] = RidgeClassifier(random_state=42)
        models['ridge_strong'] = RidgeClassifier(alpha=10.0, random_state=42)
        models['ridge_weak'] = RidgeClassifier(alpha=0.1, random_state=42)
        
        # SGD variants
        models['sgd_hinge'] = SGDClassifier(loss='hinge', random_state=42, max_iter=1000)
        models['sgd_log'] = SGDClassifier(loss='log_loss', random_state=42, max_iter=1000)
        models['sgd_perceptron'] = SGDClassifier(loss='perceptron', random_state=42, max_iter=1000)
        models['sgd_elastic'] = SGDClassifier(penalty='elasticnet', l1_ratio=0.15, random_state=42, max_iter=1000)
        
        # Other linear models
        models['perceptron'] = Perceptron(random_state=42, max_iter=1000)
        models['passive_aggressive'] = PassiveAggressiveClassifier(random_state=42, max_iter=1000)
        models['passive_aggressive_balanced'] = PassiveAggressiveClassifier(class_weight='balanced', random_state=42, max_iter=1000)
        
        # ========== NAIVE BAYES VARIANTS ==========
        print("Preparing Naive Bayes models...")
        models['naive_bayes_multinomial'] = MultinomialNB()
        models['naive_bayes_gaussian'] = GaussianNB()
        models['naive_bayes_bernoulli'] = BernoulliNB()
        models['naive_bayes_complement'] = ComplementNB()
        
        # Different smoothing parameters
        models['naive_bayes_multinomial_smooth'] = MultinomialNB(alpha=0.1)
        models['naive_bayes_bernoulli_smooth'] = BernoulliNB(alpha=0.1)
        
        # ========== DISCRIMINANT ANALYSIS ==========
        print("Preparing discriminant analysis models...")
        models['linear_discriminant'] = LinearDiscriminantAnalysis()
        models['quadratic_discriminant'] = QuadraticDiscriminantAnalysis()
        
        # ========== DISTANCE-BASED MODELS ==========
        print("Preparing distance-based models...")
        models['knn_5'] = KNeighborsClassifier(n_neighbors=5)
        models['knn_10'] = KNeighborsClassifier(n_neighbors=10)
        models['knn_weighted'] = KNeighborsClassifier(n_neighbors=5, weights='distance')
        models['knn_manhattan'] = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
        models['nearest_centroid'] = NearestCentroid()
        
        # ========== NEURAL NETWORKS ==========
        print("Preparing neural network models...")
        models['mlp_small'] = MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=500)
        models['mlp_medium'] = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        models['mlp_large'] = MLPClassifier(hidden_layer_sizes=(200, 100, 50), random_state=42, max_iter=500)
        models['mlp_relu'] = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', random_state=42, max_iter=500)
        models['mlp_tanh'] = MLPClassifier(hidden_layer_sizes=(100, 50), activation='tanh', random_state=42, max_iter=500)
        
        # ========== ENSEMBLE MODELS ==========
        print("Preparing ensemble models...")
        models['bagging_default'] = BaggingClassifier(n_estimators=10, random_state=42)
        models['bagging_large'] = BaggingClassifier(n_estimators=20, random_state=42)
        
        # Voting classifiers
        base_estimators_1 = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=500))
        ]
        models['voting_soft'] = VotingClassifier(base_estimators_1, voting='soft')
        models['voting_hard'] = VotingClassifier(base_estimators_1, voting='hard')
        
        # Stacking classifiers
        models['stacking_lr'] = StackingClassifier(
            base_estimators_1,
            final_estimator=LogisticRegression(random_state=42),
            cv=3
        )
        models['stacking_rf'] = StackingClassifier(
            base_estimators_1,
            final_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            cv=3
        )
        
        # ========== EXTERNAL LIBRARIES (with graceful handling) ==========
        print("Preparing external library models...")
        
        # XGBoost
        try:
            from xgboost import XGBClassifier
            models['xgboost_default'] = XGBClassifier(random_state=42, eval_metric='logloss')
            models['xgboost_conservative'] = XGBClassifier(learning_rate=0.05, n_estimators=200, random_state=42, eval_metric='logloss')
            print("XGBoost models added")
        except ImportError:
            print("XGBoost not available, skipping...")
        
        # LightGBM
        try:
            from lightgbm import LGBMClassifier
            models['lightgbm_default'] = LGBMClassifier(random_state=42, verbose=-1)
            models['lightgbm_balanced'] = LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)
            print("LightGBM models added")
        except ImportError:
            print("LightGBM not available, skipping...")
        
        # CatBoost
        try:
            from catboost import CatBoostClassifier
            models['catboost_default'] = CatBoostClassifier(random_state=42, verbose=False)
            models['catboost_conservative'] = CatBoostClassifier(learning_rate=0.05, iterations=200, random_state=42, verbose=False)
            print("CatBoost models added")
        except ImportError:
            print("CatBoost not available, skipping...")
        
        # ========== CALIBRATED MODELS ==========
        print("Preparing calibrated models...")
        # Add calibrated versions of key models
        models['calibrated_svm'] = CalibratedClassifierCV(SVC(random_state=42), cv=3)
        models['calibrated_ridge'] = CalibratedClassifierCV(RidgeClassifier(random_state=42), cv=3)
        models['calibrated_sgd'] = CalibratedClassifierCV(SGDClassifier(random_state=42, max_iter=1000), cv=3)
        
        print(f"Prepared {len(models)} models total")
        return models
    
    def train_pytorch_networks(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, nn.Module]:
        """Train PyTorch neural network variants."""
        print("Training PyTorch neural networks...")
        
        device = torch.device('mps' if torch.backends.mps.is_available() 
                            else 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        nn_models = {}
        
        # Different network architectures
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
        
        for nn_name, config in network_configs.items():
            try:
                print(f"Training {nn_name}...")
                
                # Prepare data
                X_train_tensor = torch.FloatTensor(X_train).to(device)
                y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
                
                # Model
                model = NeuralNetwork(X_train.shape[1], **config).to(device)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
                
                # Training loop
                best_val_loss = float('inf')
                patience = 20
                patience_counter = 0
                
                for epoch in range(200):
                    model.train()
                    optimizer.zero_grad()
                    
                    outputs = model(X_train_tensor)
                    loss = criterion(outputs, y_train_tensor)
                    loss.backward()
                    optimizer.step()
                    
                    # Validation
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor)
                    
                    scheduler.step(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model state
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        break
                
                # Load best model state
                model.load_state_dict(best_model_state)
                nn_models[nn_name] = model
                
                print(f"  ✓ {nn_name}: Training completed")
                
            except Exception as e:
                print(f"  ✗ Failed to train {nn_name}: {e}")
        
        return nn_models
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, float]:
        """Evaluate a single model."""
        if isinstance(model, nn.Module):
            # Neural network evaluation
            device = next(model.parameters()).device
            model.eval()
            
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                outputs = model(X_test_tensor)
                y_pred_proba = outputs.cpu().numpy().flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            # Scikit-learn model evaluation
            try:
                y_pred = model.predict(X_test)
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    decision_scores = model.decision_function(X_test)
                    y_pred_proba = 1 / (1 + np.exp(-decision_scores))
                else:
                    y_pred_proba = y_pred.astype(float)
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                return {}
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                           model_name: str, cv: int = 3) -> Dict[str, float]:
        """Perform cross-validation."""
        try:
            if isinstance(model, nn.Module):
                # Skip CV for neural networks (too time consuming)
                return {'cv_accuracy_mean': 0, 'cv_accuracy_std': 0}
            
            cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(cv), scoring='accuracy')
            return {
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std()
            }
        except Exception as e:
            print(f"CV failed for {model_name}: {e}")
            return {'cv_accuracy_mean': 0, 'cv_accuracy_std': 0}
    
    def save_model(self, model, model_name: str, feature_extractor: FeatureExtractor):
        """Save model with consistent format."""
        model_path = self.output_dir / f"{model_name}.pkl"
        
        if isinstance(model, nn.Module):
            # Save neural network
            model_data = {
                'model_type': 'neural_network',
                'model_state_dict': model.state_dict(),
                'model_architecture': str(model),
                'feature_extractor': feature_extractor,
                'input_dim': next(model.parameters()).shape[1] if list(model.parameters()) else 0
            }
        else:
            # Save scikit-learn model
            model_data = {
                'model_type': 'sklearn',
                'model': model,
                'feature_extractor': feature_extractor
            }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Saved {model_name} to {model_path}")
    
    def train_all_models(self, human_file: str, ai_file: str, test_size: float = 0.2):
        """Train all models and evaluate performance."""
        print("=" * 80)
        print("Ultra-Comprehensive AI vs Human Text Classification Training Pipeline")
        print(f"Training 50+ machine learning models...")
        print("=" * 80)
        
        # Load data
        texts, labels = self.load_data(human_file, ai_file)
        
        # Split data
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Further split training into train/validation
        X_train_texts, X_val_texts, y_train, y_val = train_test_split(
            X_train_texts, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Extract features
        X_train = self.feature_extractor.fit_transform(X_train_texts)
        X_val = self.feature_extractor.transform(X_val_texts)
        X_test = self.feature_extractor.transform(X_test_texts)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples") 
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Features: {X_train.shape[1]}")
        
        # Prepare all sklearn models
        sklearn_models = self.prepare_all_models()
        
        # Train sklearn models
        print(f"\nTraining {len(sklearn_models)} sklearn models...")
        print("-" * 60)
        
        for model_name, model in sklearn_models.items():
            try:
                print(f"Training {model_name}...")
                
                # Handle special cases for models that need non-negative features
                if isinstance(model, (MultinomialNB, ComplementNB)):
                    # Use only word features (TF-IDF) which are non-negative
                    word_features = X_train[:, :self.feature_extractor.max_features//2]
                    model.fit(word_features, y_train)
                else:
                    model.fit(X_train, y_train)
                
                # Cross-validation
                cv_results = self.cross_validate_model(model, X_train, y_train, model_name)
                
                # Test evaluation
                if isinstance(model, (MultinomialNB, ComplementNB)):
                    word_features_test = X_test[:, :self.feature_extractor.max_features//2]
                    test_results = self.evaluate_model(model, word_features_test, y_test, model_name)
                else:
                    test_results = self.evaluate_model(model, X_test, y_test, model_name)
                
                # Combine results
                self.results[model_name] = {**cv_results, **test_results}
                self.models[model_name] = model
                
                # Save model
                self.save_model(model, model_name, self.feature_extractor)
                
                print(f"  ✓ {model_name}: Test Accuracy: {test_results.get('accuracy', 0):.3f}")
                
            except Exception as e:
                print(f"  ✗ Failed to train {model_name}: {e}")
        
        # Train PyTorch neural networks
        print(f"\nTraining PyTorch neural networks...")
        print("-" * 60)
        
        pytorch_models = self.train_pytorch_networks(X_train, np.array(y_train), X_val, np.array(y_val))
        
        for nn_name, nn_model in pytorch_models.items():
            try:
                test_results = self.evaluate_model(nn_model, X_test, y_test, nn_name)
                
                self.results[nn_name] = test_results
                self.models[nn_name] = nn_model
                self.save_model(nn_model, nn_name, self.feature_extractor)
                
                print(f"  ✓ {nn_name}: Test Accuracy: {test_results.get('accuracy', 0):.3f}")
                
            except Exception as e:
                print(f"  ✗ Failed to evaluate {nn_name}: {e}")
        
        # Generate comprehensive results
        self.generate_results_report()
        self.create_performance_plots()
        
        print("\n" + "=" * 80)
        print("Ultra-comprehensive training completed successfully!")
        print(f"Total models trained: {len(self.results)}")
        print(f"Models saved to: {self.output_dir}")
        print(f"Results saved to: {self.results_dir}")
        print("=" * 80)
    
    def generate_results_report(self):
        """Generate comprehensive results report."""
        if not self.results:
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.results).T
        df = df.sort_values('accuracy', ascending=False)
        
        # Save detailed results
        results_file = self.results_dir / 'comprehensive_training_results.csv'
        df.to_csv(results_file)
        
        # Print summary
        print("\n" + "=" * 100)
        print("COMPREHENSIVE MODEL PERFORMANCE SUMMARY")
        print("=" * 100)
        print(f"{'Model':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
        print("-" * 100)
        
        for model_name, results in df.iterrows():
            print(f"{model_name:<30} {results['accuracy']:<10.3f} {results['precision']:<10.3f} "
                  f"{results['recall']:<10.3f} {results['f1']:<10.3f} {results['auc']:<10.3f}")
        
        # Save summary JSON
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(self.results),
            'best_model': df.index[0] if len(df) > 0 else None,
            'best_accuracy': df['accuracy'].max() if len(df) > 0 else 0,
            'top_10_models': df.index[:10].tolist(),
            'results': self.results
        }
        
        with open(self.results_dir / 'comprehensive_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Model category analysis
        print(f"\n" + "=" * 50)
        print("MODEL CATEGORY BREAKDOWN")
        print("=" * 50)
        
        categories = {
            'Tree-based': [name for name in df.index if any(x in name for x in ['forest', 'tree', 'boosting', 'adaboost'])],
            'SVM variants': [name for name in df.index if 'svm' in name],
            'Linear models': [name for name in df.index if any(x in name for x in ['logistic', 'ridge', 'sgd', 'perceptron', 'passive'])],
            'Naive Bayes': [name for name in df.index if 'naive_bayes' in name],
            'Neural Networks': [name for name in df.index if any(x in name for x in ['neural_network', 'mlp'])],
            'Ensemble': [name for name in df.index if any(x in name for x in ['voting', 'stacking', 'bagging'])],
            'Distance-based': [name for name in df.index if any(x in name for x in ['knn', 'nearest', 'centroid'])],
            'External libs': [name for name in df.index if any(x in name for x in ['xgboost', 'lightgbm', 'catboost'])],
        }
        
        for category, models in categories.items():
            if models:
                avg_accuracy = df.loc[models, 'accuracy'].mean()
                best_model = df.loc[models, 'accuracy'].idxmax()
                print(f"{category:<15}: {len(models):>2} models, avg: {avg_accuracy:.3f}, best: {best_model} ({df.loc[best_model, 'accuracy']:.3f})")
    
    def create_performance_plots(self):
        """Create comprehensive performance visualization plots."""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results).T
        
        # Create large subplot figure for comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Model Performance Analysis (50+ Models)', fontsize=16)
        
        # 1. Top 20 Models by Accuracy
        ax1 = axes[0, 0]
        top_20 = df.nlargest(20, 'accuracy')
        ax1.barh(range(len(top_20)), top_20['accuracy'])
        ax1.set_yticks(range(len(top_20)))
        ax1.set_yticklabels(top_20.index, fontsize=6)
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Top 20 Models by Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision vs Recall scatter (all models)
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df['precision'], df['recall'], c=df['accuracy'], 
                            cmap='viridis', s=50, alpha=0.7)
        ax2.set_xlabel('Precision')
        ax2.set_ylabel('Recall')
        ax2.set_title('Precision vs Recall (All Models)')
        plt.colorbar(scatter, ax=ax2, label='Accuracy')
        ax2.grid(True, alpha=0.3)
        
        # 3. F1 Score distribution
        ax3 = axes[0, 2]
        ax3.hist(df['f1'], bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('F1 Score')
        ax3.set_ylabel('Number of Models')
        ax3.set_title('F1 Score Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. AUC Score distribution
        ax4 = axes[1, 0]
        ax4.hist(df['auc'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_xlabel('AUC Score')
        ax4.set_ylabel('Number of Models')
        ax4.set_title('AUC Score Distribution')
        ax4.grid(True, alpha=0.3)
        
        # 5. Model category performance comparison
        ax5 = axes[1, 1]
        categories = {
            'Tree': [name for name in df.index if any(x in name for x in ['forest', 'tree', 'boosting'])],
            'SVM': [name for name in df.index if 'svm' in name],
            'Linear': [name for name in df.index if any(x in name for x in ['logistic', 'ridge', 'sgd'])],
            'NB': [name for name in df.index if 'naive_bayes' in name],
            'NN': [name for name in df.index if any(x in name for x in ['neural', 'mlp'])],
            'Ensemble': [name for name in df.index if any(x in name for x in ['voting', 'stacking'])],
        }
        
        category_scores = []
        category_names = []
        for cat_name, models in categories.items():
            if models:
                category_scores.append(df.loc[models, 'accuracy'].mean())
                category_names.append(f"{cat_name}\n(n={len(models)})")
        
        ax5.bar(category_names, category_scores, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.set_ylabel('Average Accuracy')
        ax5.set_title('Performance by Model Category')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # 6. Top vs Bottom models comparison
        ax6 = axes[1, 2]
        top_10_acc = df.nlargest(10, 'accuracy')['accuracy']
        bottom_10_acc = df.nsmallest(10, 'accuracy')['accuracy']
        
        ax6.boxplot([top_10_acc, bottom_10_acc], labels=['Top 10', 'Bottom 10'])
        ax6.set_ylabel('Accuracy')
        ax6.set_title('Top 10 vs Bottom 10 Models')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots
        plot_file = self.results_dir / 'comprehensive_performance_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Comprehensive performance plots saved to {plot_file}")
        
        # Create detailed correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[['accuracy', 'precision', 'recall', 'f1', 'auc']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, fmt='.3f')
        plt.title('Comprehensive Model Metrics Correlation Analysis')
        plt.tight_layout()
        
        heatmap_file = self.results_dir / 'comprehensive_metrics_correlation.png'
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to {heatmap_file}")
        
        plt.close('all')


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train comprehensive AI vs Human text classifiers')
    parser.add_argument('--human-file', required=True, help='Path to human texts JSONL file')
    parser.add_argument('--ai-file', required=True, help='Path to AI texts JSONL file')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--output-dir', default='models', help='Output directory for models')
    parser.add_argument('--results-dir', default='results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ComprehensiveModelTrainer(args.output_dir, args.results_dir)
    
    # Train all models
    trainer.train_all_models(args.human_file, args.ai_file, args.test_size)


if __name__ == "__main__":
    main()
