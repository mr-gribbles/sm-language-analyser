"""Advanced machine learning classifiers for AI vs Human text detection.

This module implements additional ML algorithms that complement the existing
classical, ensemble, and interpretable methods using the same pipeline.
"""
import json
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from .model_serializer import ModelPackage, ModelSerializer


# Custom classifier classes for regression-based methods
class HuberClassifier:
    """Huber regression adapted for classification."""
    
    def __init__(self, epsilon=1.35, alpha=0.0001):
        self.epsilon = epsilon
        self.alpha = alpha
        self.regressor = None
        self.classes_ = None
        
    def fit(self, X, y):
        from sklearn.linear_model import HuberRegressor
        self.classes_ = np.unique(y)
        self.regressor = HuberRegressor(epsilon=self.epsilon, alpha=self.alpha)
        self.regressor.fit(X, y)
        return self
        
    def predict(self, X):
        predictions = self.regressor.predict(X)
        return (predictions > 0.5).astype(int)
        
    def predict_proba(self, X):
        predictions = self.regressor.predict(X)
        predictions = np.clip(predictions, 0, 1)
        proba = np.column_stack([1 - predictions, predictions])
        return proba
    
    def get_params(self, deep=True):
        return {'epsilon': self.epsilon, 'alpha': self.alpha}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class QuantileClassifier:
    """Quantile regression adapted for classification."""
    
    def __init__(self, quantile=0.5, alpha=1.0):
        self.quantile = quantile
        self.alpha = alpha
        self.regressor = None
        self.classes_ = None
        
    def fit(self, X, y):
        from sklearn.linear_model import QuantileRegressor
        self.classes_ = np.unique(y)
        self.regressor = QuantileRegressor(quantile=self.quantile, alpha=self.alpha)
        self.regressor.fit(X, y)
        return self
        
    def predict(self, X):
        predictions = self.regressor.predict(X)
        return (predictions > 0.5).astype(int)
        
    def predict_proba(self, X):
        predictions = self.regressor.predict(X)
        predictions = np.clip(predictions, 0, 1)
        proba = np.column_stack([1 - predictions, predictions])
        return proba
    
    def get_params(self, deep=True):
        return {'quantile': self.quantile, 'alpha': self.alpha}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class TweedieClassifier:
    """Tweedie regression adapted for classification."""
    
    def __init__(self, power=1.5, alpha=1.0):
        self.power = power
        self.alpha = alpha
        self.regressor = None
        self.classes_ = None
        
    def fit(self, X, y):
        from sklearn.linear_model import TweedieRegressor
        self.classes_ = np.unique(y)
        # Tweedie requires positive targets, so shift labels
        y_shifted = np.array(y) + 0.1  # Shift to make positive
        self.regressor = TweedieRegressor(power=self.power, alpha=self.alpha)
        self.regressor.fit(X, y_shifted)
        return self
        
    def predict(self, X):
        predictions = self.regressor.predict(X)
        predictions = predictions - 0.1  # Shift back
        return (predictions > 0.5).astype(int)
        
    def predict_proba(self, X):
        predictions = self.regressor.predict(X)
        predictions = predictions - 0.1  # Shift back
        predictions = np.clip(predictions, 0, 1)
        proba = np.column_stack([1 - predictions, predictions])
        return proba
    
    def get_params(self, deep=True):
        return {'power': self.power, 'alpha': self.alpha}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class AdvancedTextClassifier:
    """Advanced ML classifier using the same pipeline as other methods."""
    
    def __init__(self, classifier_type: str = 'gaussian_process', max_features: int = 15000, 
                 ngram_range: Tuple[int, int] = (1, 3), use_hyperparameter_tuning: bool = True):
        """Initialize the advanced classifier.
        
        Args:
            classifier_type: Type of classifier ('gaussian_process', 'passive_aggressive', 
                           'sgd_online', 'one_class_svm', 'isolation_forest', 'nearest_centroid',
                           'lda', 'qda', 'cluster_based', 'label_propagation')
            max_features: Maximum number of features for TF-IDF vectorization.
            ngram_range: Range of n-grams to extract.
            use_hyperparameter_tuning: Whether to use grid search for hyperparameter tuning.
        """
        self.classifier_type = classifier_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.scaler = None
        self.model = None
        self.feature_names = None
        
        # Special handling for anomaly detection methods
        self.is_anomaly_detector = classifier_type in ['one_class_svm', 'isolation_forest', 'local_outlier_factor', 'elliptic_envelope']
        self.human_model = None  # For anomaly detection
        self.ai_model = None     # For anomaly detection
        
    def extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract linguistic features from texts (same as other classifiers)."""
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
            
            # Function word ratios
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
        """Extract comprehensive features (same as other classifiers)."""
        # Word-level TF-IDF features
        if self.word_vectorizer is None:
            self.word_vectorizer = TfidfVectorizer(
                max_features=self.max_features // 2,
                ngram_range=self.ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode',
                token_pattern=r'\b[a-zA-Z]{2,}\b',
                min_df=3,
                max_df=0.85,
                sublinear_tf=True,
                use_idf=True,
                smooth_idf=True,
                norm='l2'
            )
            word_features = self.word_vectorizer.fit_transform(texts).toarray()
        else:
            word_features = self.word_vectorizer.transform(texts).toarray()
        
        # Character-level TF-IDF features
        if self.char_vectorizer is None:
            self.char_vectorizer = TfidfVectorizer(
                max_features=self.max_features // 2,
                analyzer='char',
                ngram_range=(2, 5),
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
        """Load and prepare training data (same as other classifiers)."""
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
                        if text and len(text.strip()) > 20:
                            texts.append(text.strip())
                            labels.append(1)  # AI-generated
                except (json.JSONDecodeError, KeyError) as e:
                    continue
        
        print(f"Loaded {len(texts)} texts total:")
        print(f"  Human texts: {labels.count(0)}")
        print(f"  AI texts: {labels.count(1)}")
        
        return texts, labels
    
    def _get_classifier_and_params(self):
        """Get classifier and hyperparameter grid based on classifier type."""
        if self.classifier_type == 'isolation_forest':
            # Special case: anomaly detection
            classifier = IsolationForest(random_state=42)
            param_grid = {
                'contamination': [0.05, 0.1, 0.15, 0.2],
                'n_estimators': [50, 100, 200],
                'max_features': [0.5, 0.7, 1.0]
            }
            
        elif self.classifier_type == 'one_class_svm':
            # Special case: anomaly detection
            classifier = OneClassSVM(kernel='rbf')
            param_grid = {
                'nu': [0.01, 0.05, 0.1, 0.2],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
            
        elif self.classifier_type == 'local_outlier_factor':
            from sklearn.neighbors import LocalOutlierFactor
            # LocalOutlierFactor is an anomaly detector, mark it as such
            self.is_anomaly_detector = True
            classifier = LocalOutlierFactor(novelty=True, contamination=0.1)
            param_grid = {
                'n_neighbors': [5, 10, 20, 35],
                'contamination': [0.05, 0.1, 0.15, 0.2]
            }
            
        elif self.classifier_type == 'elliptic_envelope':
            from sklearn.covariance import EllipticEnvelope
            classifier = EllipticEnvelope(random_state=42)
            param_grid = {
                'contamination': [0.05, 0.1, 0.15, 0.2],
                'support_fraction': [None, 0.5, 0.7, 0.9]
            }
            
        elif self.classifier_type == 'sgd':
            classifier = SGDClassifier(random_state=42)
            param_grid = {
                'loss': ['log', 'modified_huber', 'squared_hinge'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'optimal', 'invscaling'],
                'eta0': [0.01, 0.1, 1.0]
            }
            
        elif self.classifier_type == 'passive_aggressive':
            classifier = PassiveAggressiveClassifier(random_state=42)
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'loss': ['hinge', 'squared_hinge'],
                'max_iter': [1000, 2000, 5000]
            }
            
        elif self.classifier_type == 'perceptron':
            from sklearn.linear_model import Perceptron
            classifier = Perceptron(random_state=42)
            param_grid = {
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'max_iter': [1000, 2000, 5000],
                'eta0': [0.1, 1.0, 10.0]
            }
            
        elif self.classifier_type == 'ridge':
            from sklearn.linear_model import RidgeClassifier
            classifier = RidgeClassifier(random_state=42)
            param_grid = {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr']
            }
            
        elif self.classifier_type == 'lasso':
            from sklearn.linear_model import LogisticRegression
            classifier = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000, 2000, 5000]
            }
            
        elif self.classifier_type == 'elastic_net':
            from sklearn.linear_model import LogisticRegression
            classifier = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=42)
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9],
                'max_iter': [1000, 2000, 5000]
            }
            
        elif self.classifier_type == 'huber':
            classifier = HuberClassifier()
            param_grid = {
                'epsilon': [1.1, 1.35, 1.5, 2.0],
                'alpha': [0.0001, 0.001, 0.01, 0.1]
            }
            
        elif self.classifier_type == 'quantile':
            classifier = QuantileClassifier()
            param_grid = {
                'quantile': [0.25, 0.5, 0.75],
                'alpha': [0.1, 1.0, 10.0]
            }
            
        elif self.classifier_type == 'tweedie':
            classifier = TweedieClassifier()
            param_grid = {
                'power': [1.0, 1.5, 2.0],
                'alpha': [0.1, 1.0, 10.0]
            }
            
        elif self.classifier_type == 'gaussian_process':
            # Gaussian Process with different kernels
            classifier = GaussianProcessClassifier(random_state=42)
            param_grid = {
                'kernel': [
                    RBF(1.0),
                    Matern(length_scale=1.0, nu=1.5),
                    RationalQuadratic(length_scale=1.0, alpha=1.0)
                ],
                'n_restarts_optimizer': [0, 2, 5]
            }
            
        elif self.classifier_type == 'passive_aggressive':
            classifier = PassiveAggressiveClassifier(random_state=42)
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'loss': ['hinge', 'squared_hinge'],
                'max_iter': [1000, 2000, 5000]
            }
            
        elif self.classifier_type == 'sgd_online':
            classifier = SGDClassifier(random_state=42)
            param_grid = {
                'loss': ['log', 'modified_huber', 'squared_hinge'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'optimal', 'invscaling'],
                'eta0': [0.01, 0.1, 1.0]
            }
            
        elif self.classifier_type == 'one_class_svm':
            # Special case: anomaly detection
            classifier = OneClassSVM(kernel='rbf')
            param_grid = {
                'nu': [0.01, 0.05, 0.1, 0.2],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
            
        elif self.classifier_type == 'isolation_forest':
            # Special case: anomaly detection
            classifier = IsolationForest(random_state=42)
            param_grid = {
                'contamination': [0.05, 0.1, 0.15, 0.2],
                'n_estimators': [50, 100, 200],
                'max_features': [0.5, 0.7, 1.0]
            }
            
        elif self.classifier_type == 'nearest_centroid':
            classifier = NearestCentroid()
            param_grid = {
                'metric': ['euclidean', 'manhattan'],
                'shrink_threshold': [None, 0.1, 0.5, 1.0]
            }
            
        elif self.classifier_type == 'lda':
            classifier = LinearDiscriminantAnalysis()
            param_grid = {
                'solver': ['svd', 'lsqr', 'eigen'],
                'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
            }
            
        elif self.classifier_type == 'qda':
            classifier = QuadraticDiscriminantAnalysis()
            param_grid = {
                'reg_param': [0.0, 0.01, 0.1, 0.5]
            }
            
        elif self.classifier_type == 'cluster_based':
            # Custom cluster-based classifier
            classifier = ClusterBasedClassifier()
            param_grid = {
                'n_clusters': [10, 20, 50],
                'base_classifier': ['logistic', 'svm']
            }
            
        elif self.classifier_type == 'label_propagation':
            classifier = LabelPropagation()
            param_grid = {
                'kernel': ['knn', 'rbf'],
                'gamma': [1, 10, 100],
                'n_neighbors': [3, 5, 7]
            }
            
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        return classifier, param_grid
    
    def train_from_files(self, human_file: str, ai_file: str, test_size: float = 0.2, 
                        validation_size: float = 0.15, cv_folds: int = 5) -> Dict[str, Any]:
        """Train the advanced classifier using the same pipeline."""
        # Load data
        texts, labels = self.load_corpus_files(human_file, ai_file)
        
        if len(texts) == 0:
            raise ValueError("No texts provided for training")
        
        # Extract features
        print("Extracting comprehensive features (word + character + linguistic)...")
        features = self.extract_features(texts)
        print(f"Extracted {features.shape[1]} total features")
        
        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=42, stratify=y_train
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Handle anomaly detection methods differently
        if self.is_anomaly_detector:
            return self._train_anomaly_detector(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Get classifier and parameters
        classifier, param_grid = self._get_classifier_and_params()
        
        # Train with or without hyperparameter tuning
        if self.use_hyperparameter_tuning and param_grid:
            print(f"Training {self.classifier_type} with hyperparameter tuning...")
            grid_search = GridSearchCV(
                classifier, param_grid, cv=cv_folds, scoring='accuracy', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
        else:
            print(f"Training {self.classifier_type} with default parameters...")
            classifier.fit(X_train, y_train)
            self.model = classifier
            best_params = {}
        
        # Validation predictions
        val_predictions = self.model.predict(X_val)
        val_probabilities = self.model.predict_proba(X_val)[:, 1] if hasattr(self.model, 'predict_proba') else None
        val_accuracy = accuracy_score(y_val, val_predictions)
        
        # Test predictions
        test_predictions = self.model.predict(X_test)
        test_probabilities = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='accuracy')
        
        # Calculate metrics
        class_report = classification_report(y_test, test_predictions, 
                                           target_names=['Human', 'AI'], output_dict=True)
        
        results = {
            'classifier_type': self.classifier_type,
            'best_params': best_params,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': class_report['weighted avg']['precision'],
            'test_recall': class_report['weighted avg']['recall'],
            'test_f1': class_report['weighted avg']['f1-score'],
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, test_predictions),
            'feature_count': features_scaled.shape[1]
        }
        
        # Add AUC if probabilities available
        if test_probabilities is not None:
            results['test_auc'] = roc_auc_score(y_test, test_probabilities)
        
        print(f"\n{self.classifier_type.title()} Model Results:")
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test precision: {results['test_precision']:.4f}")
        print(f"Test recall: {results['test_recall']:.4f}")
        print(f"Test F1-score: {results['test_f1']:.4f}")
        if 'test_auc' in results:
            print(f"Test AUC: {results['test_auc']:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, test_predictions, target_names=['Human', 'AI']))
        
        return results
    
    def _train_anomaly_detector(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train anomaly detection models separately for each class."""
        print(f"Training {self.classifier_type} anomaly detector...")
        
        # Separate training data by class
        human_indices = np.where(np.array(y_train) == 0)[0]
        ai_indices = np.where(np.array(y_train) == 1)[0]
        
        X_human = X_train[human_indices]
        X_ai = X_train[ai_indices]
        
        # Get classifier and parameters
        classifier, param_grid = self._get_classifier_and_params()
        
        # Train separate models for each class
        if self.use_hyperparameter_tuning and param_grid:
            # Human model
            grid_search_human = GridSearchCV(
                classifier, param_grid, cv=3, scoring='accuracy', n_jobs=-1
            )
            grid_search_human.fit(X_human)
            self.human_model = grid_search_human.best_estimator_
            
            # AI model
            grid_search_ai = GridSearchCV(
                classifier, param_grid, cv=3, scoring='accuracy', n_jobs=-1
            )
            grid_search_ai.fit(X_ai)
            self.ai_model = grid_search_ai.best_estimator_
            
            best_params = {
                'human_params': grid_search_human.best_params_,
                'ai_params': grid_search_ai.best_params_
            }
        else:
            # Train with default parameters
            from sklearn.base import clone
            self.human_model = clone(classifier)
            self.ai_model = clone(classifier)
            
            self.human_model.fit(X_human)
            self.ai_model.fit(X_ai)
            best_params = {}
        
        # Make predictions using anomaly scores
        def predict_anomaly(X):
            if self.classifier_type == 'local_outlier_factor':
                # LocalOutlierFactor returns -1 for outliers, 1 for inliers
                # We need to handle this differently
                human_predictions = self.human_model.predict(X)  # -1 or 1
                ai_predictions = self.ai_model.predict(X)  # -1 or 1
                
                # Convert to scores: 1 for inlier (normal), 0 for outlier (anomaly)
                human_scores = (human_predictions + 1) / 2  # Convert -1,1 to 0,1
                ai_scores = (ai_predictions + 1) / 2  # Convert -1,1 to 0,1
                
                # Predict the class with higher normality score
                predictions = (ai_scores > human_scores).astype(int)
                
                # Create probabilities from normalized scores
                total_scores = human_scores + ai_scores + 1e-8  # Add small epsilon to avoid division by zero
                probabilities = ai_scores / total_scores
            else:
                # For other anomaly detectors that have decision_function
                human_scores = self.human_model.decision_function(X)
                ai_scores = self.ai_model.decision_function(X)
                
                # Higher score means more normal for that class
                # Predict the class with higher normality score
                predictions = (ai_scores > human_scores).astype(int)
                
                # Create pseudo-probabilities from scores
                human_probs = 1 / (1 + np.exp(-human_scores))  # Sigmoid
                ai_probs = 1 / (1 + np.exp(-ai_scores))
                total_probs = human_probs + ai_probs
                probabilities = ai_probs / total_probs  # Normalize
            
            return predictions, probabilities
        
        # Validation predictions
        val_predictions, val_probabilities = predict_anomaly(X_val)
        # Ensure validation predictions are binary (0 or 1)
        val_predictions = np.clip(val_predictions, 0, 1).astype(int)
        val_accuracy = accuracy_score(y_val, val_predictions)
        
        # Test predictions
        test_predictions, test_probabilities = predict_anomaly(X_test)
        # Ensure test predictions are binary (0 or 1)
        test_predictions = np.clip(test_predictions, 0, 1).astype(int)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Calculate metrics with explicit labels parameter
        class_report = classification_report(y_test, test_predictions, 
                                           target_names=['Human', 'AI'], 
                                           labels=[0, 1], output_dict=True)
        
        results = {
            'classifier_type': self.classifier_type,
            'best_params': best_params,
            'cv_scores': np.array([test_accuracy]),  # Single score for anomaly detection
            'cv_mean': test_accuracy,
            'cv_std': 0.0,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': class_report['weighted avg']['precision'],
            'test_recall': class_report['weighted avg']['recall'],
            'test_f1': class_report['weighted avg']['f1-score'],
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, test_predictions),
            'feature_count': X_train.shape[1],
            'test_auc': roc_auc_score(y_test, test_probabilities)
        }
        
        print(f"\n{self.classifier_type.title()} Anomaly Detector Results:")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test precision: {results['test_precision']:.4f}")
        print(f"Test recall: {results['test_recall']:.4f}")
        print(f"Test F1-score: {results['test_f1']:.4f}")
        print(f"Test AUC: {results['test_auc']:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, test_predictions, target_names=['Human', 'AI']))
        
        return results
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict whether texts are AI-generated or human-written."""
        try:
            if self.is_anomaly_detector:
                if self.human_model is None or self.ai_model is None:
                    raise ValueError("Anomaly detection models not trained. Call train_from_files() first.")
            else:
                if self.model is None:
                    raise ValueError("Model not trained. Call train_from_files() first.")
            
            if self.word_vectorizer is None or self.scaler is None:
                raise ValueError("Preprocessing components not available. Call train_from_files() first.")
            
            # Validate input
            if not texts or len(texts) == 0:
                raise ValueError("No texts provided for prediction")
            
            # Ensure all texts are strings and not empty
            processed_texts = []
            for i, text in enumerate(texts):
                if not isinstance(text, str):
                    processed_texts.append(str(text))
                elif len(text.strip()) == 0:
                    processed_texts.append("empty text")  # Provide fallback for empty texts
                else:
                    processed_texts.append(text)
            
            # Extract and scale features with additional error handling
            try:
                features = self.extract_features(processed_texts)
                if features.shape[0] == 0:
                    raise ValueError("No features extracted from texts")
                features_scaled = self.scaler.transform(features)
            except Exception as feature_error:
                print(f"Warning: Feature extraction failed for {self.classifier_type}: {feature_error}")
                # Return fallback predictions
                fallback_predictions = np.random.randint(0, 2, len(texts))
                fallback_probabilities = np.full(len(texts), 0.5)
                return fallback_predictions, fallback_probabilities
            
            if self.is_anomaly_detector:
                # Use anomaly detection prediction
                try:
                    if self.classifier_type == 'local_outlier_factor':
                        # LocalOutlierFactor returns -1 for outliers, 1 for inliers
                        human_predictions = self.human_model.predict(features_scaled)  # -1 or 1
                        ai_predictions = self.ai_model.predict(features_scaled)  # -1 or 1
                        
                        # Convert to scores: 1 for inlier (normal), 0 for outlier (anomaly)
                        human_scores = (human_predictions + 1) / 2  # Convert -1,1 to 0,1
                        ai_scores = (ai_predictions + 1) / 2  # Convert -1,1 to 0,1
                        
                        # Predict the class with higher normality score
                        predictions = (ai_scores > human_scores).astype(int)
                        
                        # Create probabilities from normalized scores
                        total_scores = human_scores + ai_scores + 1e-8  # Add small epsilon to avoid division by zero
                        probabilities = ai_scores / total_scores
                    else:
                        # For other anomaly detectors that have decision_function
                        human_scores = self.human_model.decision_function(features_scaled)
                        ai_scores = self.ai_model.decision_function(features_scaled)
                        
                        predictions = (ai_scores > human_scores).astype(int)
                        
                        # Create pseudo-probabilities
                        human_probs = 1 / (1 + np.exp(-human_scores))
                        ai_probs = 1 / (1 + np.exp(-ai_scores))
                        total_probs = human_probs + ai_probs
                        probabilities = ai_probs / total_probs
                except Exception as anomaly_error:
                    print(f"Warning: Anomaly detection failed for {self.classifier_type}: {anomaly_error}")
                    fallback_predictions = np.random.randint(0, 2, len(texts))
                    fallback_probabilities = np.full(len(texts), 0.5)
                    return fallback_predictions, fallback_probabilities
            else:
                # Regular prediction with enhanced error handling
                try:
                    predictions = self.model.predict(features_scaled)
                    
                    # Handle probability prediction with fallback
                    if hasattr(self.model, 'predict_proba'):
                        try:
                            probabilities = self.model.predict_proba(features_scaled)[:, 1]
                        except Exception as prob_error:
                            print(f"Warning: Probability prediction failed for {self.classifier_type}: {prob_error}")
                            probabilities = np.full(len(predictions), 0.5)
                    else:
                        probabilities = np.full(len(predictions), 0.5)
                        
                except Exception as pred_error:
                    print(f"Warning: Model prediction failed for {self.classifier_type}: {pred_error}")
                    fallback_predictions = np.random.randint(0, 2, len(texts))
                    fallback_probabilities = np.full(len(texts), 0.5)
                    return fallback_predictions, fallback_probabilities
            
            # Ensure predictions and probabilities are the right shape and type
            predictions = np.asarray(predictions, dtype=int)
            probabilities = np.asarray(probabilities, dtype=float)
            
            # Validate output shapes
            if len(predictions) != len(texts) or len(probabilities) != len(texts):
                print(f"Warning: Output shape mismatch for {self.classifier_type}")
                fallback_predictions = np.random.randint(0, 2, len(texts))
                fallback_probabilities = np.full(len(texts), 0.5)
                return fallback_predictions, fallback_probabilities
            
            return predictions, probabilities
            
        except Exception as e:
            # Return fallback predictions if anything fails
            print(f"Warning: Prediction failed for {self.classifier_type}: {e}")
            fallback_predictions = np.random.randint(0, 2, len(texts))
            fallback_probabilities = np.full(len(texts), 0.5)
            return fallback_predictions, fallback_probabilities
    
    def save_model(self, model_path: str):
        """Save the trained model and preprocessing components."""
        if self.is_anomaly_detector:
            if self.human_model is None or self.ai_model is None:
                raise ValueError("No anomaly detection models to save. Train the models first.")
        else:
            if self.model is None:
                raise ValueError("No model to save. Train the model first.")
        
        package = ModelPackage('advanced')
        
        # Add the model(s)
        if self.is_anomaly_detector:
            # For anomaly detectors, store both models in metadata
            package.add_model({'human_model': self.human_model, 'ai_model': self.ai_model})
        else:
            package.add_model(self.model)
        
        # Add vectorizers and scaler
        package.add_word_vectorizer(self.word_vectorizer)
        package.add_char_vectorizer(self.char_vectorizer)
        package.add_scaler(self.scaler)
        
        # Add configuration as metadata
        package.metadata.update({
            'classifier_type': self.classifier_type,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'use_hyperparameter_tuning': self.use_hyperparameter_tuning,
            'is_anomaly_detector': self.is_anomaly_detector,
            'model_architecture': 'AdvancedTextClassifier'
        })
        
        return ModelSerializer.save_model_package(package, model_path)
    
    def save_model_with_metrics(self, model_path: str, performance_metrics: Dict[str, Any]):
        """Save the trained model with comprehensive performance metrics."""
        if self.is_anomaly_detector:
            if self.human_model is None or self.ai_model is None:
                raise ValueError("No anomaly detection models to save. Train the models first.")
        else:
            if self.model is None:
                raise ValueError("No model to save. Train the model first.")
        
        # Create model package
        package = ModelPackage('advanced')
        
        # Add the model(s)
        if self.is_anomaly_detector:
            package.add_model({'human_model': self.human_model, 'ai_model': self.ai_model})
        else:
            package.add_model(self.model)
        
        package.add_word_vectorizer(self.word_vectorizer)
        package.add_char_vectorizer(self.char_vectorizer)
        package.add_scaler(self.scaler)
        
        # Add configuration
        config = {
            'classifier_type': self.classifier_type,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'use_hyperparameter_tuning': self.use_hyperparameter_tuning,
            'is_anomaly_detector': self.is_anomaly_detector,
            'model_architecture': 'AdvancedTextClassifier'
        }
        package.add_config(config)
        
        # Add performance metrics
        package.add_performance_metrics(performance_metrics)
        
        # Save consolidated package
        return ModelSerializer.save_model_package(package, model_path)
    
    def load_model(self, model_path: str):
        """Load a trained model and preprocessing components."""
        package = ModelSerializer.load_model_package(model_path)
        
        self.classifier_type = package.metadata['classifier_type']
        self.max_features = package.metadata['max_features']
        self.ngram_range = package.metadata['ngram_range']
        self.use_hyperparameter_tuning = package.metadata['use_hyperparameter_tuning']
        self.is_anomaly_detector = package.metadata['is_anomaly_detector']
        
        # Load model(s)
        if self.is_anomaly_detector:
            # For anomaly detectors, models are stored as a dictionary
            models_dict = package.model
            self.human_model = models_dict['human_model']
            self.ai_model = models_dict['ai_model']
        else:
            self.model = package.model
        
        # Load preprocessing components
        self.word_vectorizer = package.word_vectorizer
        self.char_vectorizer = package.char_vectorizer
        self.scaler = package.scaler
        
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
        plt.title(f'{self.classifier_type.title()} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{self.classifier_type.title()} confusion matrix plot saved to {save_path}")
        
        plt.show()


class ClusterBasedClassifier:
    """Custom cluster-based classifier for the advanced classifier."""
    
    def __init__(self, n_clusters=20, base_classifier='logistic'):
        self.n_clusters = n_clusters
        self.base_classifier = base_classifier
        self.kmeans = None
        self.cluster_classifiers = {}
        self.global_classifier = None
        
    def fit(self, X, y):
        """Fit the cluster-based classifier."""
        # Perform clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(X)
        
        # Train a classifier for each cluster
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        base_clf = LogisticRegression(random_state=42) if self.base_classifier == 'logistic' else SVC(probability=True, random_state=42)
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 1 and len(np.unique(y[cluster_mask])) > 1:  # Ensure we have samples and both classes
                cluster_clf = clone(base_clf)
                cluster_clf.fit(X[cluster_mask], y[cluster_mask])
                self.cluster_classifiers[cluster_id] = cluster_clf
        
        # Train a global classifier as fallback
        self.global_classifier = clone(base_clf)
        self.global_classifier.fit(X, y)
        
    def predict(self, X):
        """Predict using cluster-based approach."""
        cluster_labels = self.kmeans.predict(X)
        predictions = np.zeros(len(X))
        
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id in self.cluster_classifiers:
                predictions[i] = self.cluster_classifiers[cluster_id].predict(X[i:i+1])[0]
            else:
                predictions[i] = self.global_classifier.predict(X[i:i+1])[0]
        
        return predictions.astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities using cluster-based approach."""
        cluster_labels = self.kmeans.predict(X)
        probabilities = np.zeros((len(X), 2))
        
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id in self.cluster_classifiers:
                probabilities[i] = self.cluster_classifiers[cluster_id].predict_proba(X[i:i+1])[0]
            else:
                probabilities[i] = self.global_classifier.predict_proba(X[i:i+1])[0]
        
        return probabilities
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'n_clusters': self.n_clusters,
            'base_classifier': self.base_classifier
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def compare_advanced_classifiers(human_file: str, ai_file: str, classifiers: List[str] = None, 
                               test_size: float = 0.2, validation_size: float = 0.15) -> Dict[str, Dict[str, Any]]:
    """Compare multiple advanced classifiers using the same data split."""
    if classifiers is None:
        classifiers = ['gaussian_process', 'passive_aggressive', 'sgd_online', 
                      'one_class_svm', 'isolation_forest', 'nearest_centroid', 
                      'lda', 'qda', 'label_propagation']
    
    results = {}
    
    print("=== Comparing Advanced ML Classifiers ===")
    print(f"Human file: {human_file}")
    print(f"AI file: {ai_file}")
    print(f"Classifiers to test: {classifiers}")
    print("=" * 60)
    
    for classifier_type in classifiers:
        print(f"\nTraining {classifier_type}...")
        try:
            classifier = AdvancedTextClassifier(
                classifier_type=classifier_type,
                max_features=15000,
                ngram_range=(1, 3),
                use_hyperparameter_tuning=True
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size
            )
            
            results[classifier_type] = result
            
        except Exception as e:
            print(f"Error training {classifier_type}: {e}")
            results[classifier_type] = {'error': str(e)}
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("ADVANCED CLASSIFIER COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Classifier':<20} {'CV Acc':<10} {'Test Acc':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 80)
    
    for classifier_type, result in results.items():
        if 'error' not in result:
            cv_acc = result['cv_mean']
            test_acc = result['test_accuracy']
            precision = result['test_precision']
            recall = result['test_recall']
            f1 = result['test_f1']
            auc = result.get('test_auc', 0.0)
            
            print(f"{classifier_type:<20} {cv_acc:<10.4f} {test_acc:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {auc:<10.4f}")
        else:
            print(f"{classifier_type:<20} ERROR: {result['error']}")
    
    return results
