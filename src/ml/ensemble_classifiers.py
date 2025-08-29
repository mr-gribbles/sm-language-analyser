"""Ensemble machine learning classifiers for AI vs Human text detection.

This module implements advanced ensemble methods using the same
feature extraction and training pipeline as the neural network approach.
"""
import json
import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import re

import numpy as np
import pandas as pd
import warnings
from contextlib import contextmanager
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.ensemble import (VotingClassifier, BaggingClassifier, ExtraTreesClassifier,
                             RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
                             StackingClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from .model_serializer import ModelPackage, ModelSerializer


@contextmanager
def suppress_sklearn_warnings():
    """Context manager to suppress specific sklearn numerical warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, 
                              message='.*divide by zero encountered in matmul.*')
        warnings.filterwarnings('ignore', category=RuntimeWarning, 
                              message='.*overflow encountered in matmul.*')
        warnings.filterwarnings('ignore', category=RuntimeWarning, 
                              message='.*invalid value encountered in matmul.*')
        warnings.filterwarnings('ignore', category=RuntimeWarning, 
                              message='.*divide by zero encountered.*')
        warnings.filterwarnings('ignore', category=RuntimeWarning, 
                              message='.*overflow encountered.*')
        warnings.filterwarnings('ignore', category=RuntimeWarning, 
                              message='.*invalid value encountered.*')
        yield


try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class EnsembleTextClassifier:
    """Ensemble ML classifier using the same pipeline as the neural network."""
    
    def __init__(self, ensemble_type: str = 'voting', max_features: int = 15000, 
                 ngram_range: Tuple[int, int] = (1, 3), use_hyperparameter_tuning: bool = True):
        """Initialize the ensemble classifier.
        
        Args:
            ensemble_type: Type of ensemble ('voting', 'bagging', 'stacking', 'xgboost', 
                         'lightgbm', 'catboost', 'extra_trees', 'custom_ensemble')
            max_features: Maximum number of features for TF-IDF vectorization.
            ngram_range: Range of n-grams to extract.
            use_hyperparameter_tuning: Whether to use grid search for hyperparameter tuning.
        """
        self.ensemble_type = ensemble_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.scaler = None
        self.model = None
        self.feature_names = None
        
    def extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract linguistic features from texts with numerical stability fixes."""
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
            
            # Readability approximation (Flesch-like) with safe calculations
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
            
            # Function word ratios with safe division
            function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
            function_word_count = sum(1 for word in words if word.lower() in function_words)
            text_features.append(function_word_count / max(word_count, 1))  # Safe division
            
            # Repetition patterns with safe division
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
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract comprehensive features (same as neural network version)."""
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
        """Load and prepare training data (same as neural network version)."""
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
    
    def _get_ensemble_and_params(self):
        """Get ensemble classifier and hyperparameter grid based on ensemble type."""
        if self.ensemble_type == 'voting':
            # Voting classifier with diverse base estimators
            # Note: Removed ComplementNB because it can't handle negative values from StandardScaler
            base_estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                ('svm', SVC(probability=True, random_state=42)),
                ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                ('gb', GradientBoostingClassifier(random_state=42))
            ]
            
            classifier = VotingClassifier(estimators=base_estimators, voting='soft')
            param_grid = {
                'rf__n_estimators': [100, 200],
                'rf__max_depth': [10, 20],
                'svm__C': [1, 10],
                'lr__C': [1, 10],
                'gb__n_estimators': [100, 200],
                'gb__learning_rate': [0.1, 0.2]
            }
            
        elif self.ensemble_type == 'bagging':
            classifier = BaggingClassifier(
                base_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
                random_state=42, n_jobs=-1
            )
            param_grid = {
                'n_estimators': [10, 20, 30],
                'max_samples': [0.7, 0.8, 0.9],
                'max_features': [0.7, 0.8, 0.9]
            }
            
        elif self.ensemble_type == 'stacking':
            # Stacking classifier with diverse base estimators
            base_estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                ('svm', SVC(probability=True, random_state=42)),
                ('gb', GradientBoostingClassifier(random_state=42)),
                ('nb', ComplementNB())
            ]
            
            classifier = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(random_state=42),
                cv=5
            )
            param_grid = {
                'rf__n_estimators': [100, 200],
                'rf__max_depth': [10, 20],
                'svm__C': [1, 10],
                'gb__n_estimators': [100, 200],
                'final_estimator__C': [1, 10]
            }
            
        elif self.ensemble_type == 'extra_trees':
            classifier = ExtraTreesClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
        elif self.ensemble_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost not available. Install with: pip install xgboost")
            
            classifier = XGBClassifier(random_state=42, eval_metric='logloss')
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
        elif self.ensemble_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ValueError("LightGBM not available. Install with: pip install lightgbm")
            
            classifier = LGBMClassifier(random_state=42, verbose=-1)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'num_leaves': [31, 50, 100]
            }
            
        elif self.ensemble_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                raise ValueError("CatBoost not available. Install with: pip install catboost")
            
            classifier = CatBoostClassifier(random_state=42, verbose=False)
            param_grid = {
                'iterations': [100, 200, 300],
                'depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5]
            }
            
        elif self.ensemble_type == 'custom_ensemble':
            # Custom ensemble combining multiple boosting methods
            base_estimators = []
            
            # Add available boosting methods
            base_estimators.append(('gb', GradientBoostingClassifier(random_state=42)))
            base_estimators.append(('ada', AdaBoostClassifier(random_state=42)))
            
            if XGBOOST_AVAILABLE:
                base_estimators.append(('xgb', XGBClassifier(random_state=42, eval_metric='logloss')))
            
            if LIGHTGBM_AVAILABLE:
                base_estimators.append(('lgb', LGBMClassifier(random_state=42, verbose=-1)))
            
            # Add tree-based methods
            base_estimators.append(('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)))
            base_estimators.append(('et', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)))
            
            classifier = VotingClassifier(estimators=base_estimators, voting='soft')
            param_grid = {
                'gb__n_estimators': [100, 200],
                'ada__n_estimators': [50, 100],
                'rf__max_depth': [10, 20],
                'et__max_depth': [10, 20]
            }
            
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
        
        return classifier, param_grid
    
    def train_from_files(self, human_file: str, ai_file: str, test_size: float = 0.2, 
                        validation_size: float = 0.15, cv_folds: int = 5) -> Dict[str, Any]:
        """Train the ensemble classifier using the same pipeline as neural network."""
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
        
        # Get ensemble and parameters
        classifier, param_grid = self._get_ensemble_and_params()
        
        # Train with or without hyperparameter tuning (with warning suppression)
        with suppress_sklearn_warnings():
            if self.use_hyperparameter_tuning and param_grid:
                print(f"Training {self.ensemble_type} ensemble with hyperparameter tuning...")
                # Use fewer CV folds for ensemble methods to reduce training time
                grid_search = GridSearchCV(
                    classifier, param_grid, cv=max(3, cv_folds-2), scoring='accuracy', 
                    n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"Best parameters: {best_params}")
            else:
                print(f"Training {self.ensemble_type} ensemble with default parameters...")
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
            
            # Cross-validation score (use fewer folds for ensemble methods)
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=max(3, cv_folds-2), scoring='accuracy')
        
        # Calculate metrics
        class_report = classification_report(y_test, test_predictions, 
                                           target_names=['Human', 'AI'], output_dict=True)
        
        results = {
            'ensemble_type': self.ensemble_type,
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
        
        print(f"\n{self.ensemble_type.title()} Ensemble Results:")
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
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict whether texts are AI-generated or human-written."""
        if self.model is None or self.word_vectorizer is None or self.scaler is None:
            raise ValueError("Model not trained. Call train_from_files() first.")
        
        # Extract and scale features
        features = self.extract_features(texts)
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else np.zeros_like(predictions)
        
        return predictions, probabilities
    
    def save_model(self, model_path: str):
        """Save the trained model and preprocessing components using consolidated format."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        package = ModelPackage('ensemble')
        
        # Add the sklearn ensemble model
        package.add_model(self.model)
        
        # Add vectorizers and scaler
        package.add_word_vectorizer(self.word_vectorizer)
        package.add_char_vectorizer(self.char_vectorizer)
        package.add_scaler(self.scaler)
        
        # Add configuration as metadata
        package.metadata.update({
            'ensemble_type': self.ensemble_type,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'use_hyperparameter_tuning': self.use_hyperparameter_tuning,
            'model_architecture': 'EnsembleTextClassifier'
        })
        
        return ModelSerializer.save_model_package(package, model_path)
    
    def load_model(self, model_path: str):
        """Load a trained model and preprocessing components."""
        try:
            # Try consolidated format first
            package = ModelSerializer.load_model_package(model_path)
            
            self.ensemble_type = package.metadata['ensemble_type']
            self.max_features = package.metadata['max_features']
            self.ngram_range = package.metadata['ngram_range']
            self.use_hyperparameter_tuning = package.metadata['use_hyperparameter_tuning']
            
            self.model = package.model
            self.word_vectorizer = package.word_vectorizer
            self.char_vectorizer = package.char_vectorizer
            self.scaler = package.scaler
            
        except (FileNotFoundError, KeyError):
            # Fallback to legacy format
            model_path = Path(model_path)
            
            # Load configuration
            with open(f"{model_path}_{self.ensemble_type}_config.json", 'r') as f:
                config = json.load(f)
            
            self.ensemble_type = config['ensemble_type']
            self.max_features = config['max_features']
            self.ngram_range = config['ngram_range']
            self.use_hyperparameter_tuning = config['use_hyperparameter_tuning']
            
            # Load model
            self.model = joblib.load(f"{model_path}_{self.ensemble_type}_ensemble_model.pkl")
            
            # Load preprocessing components
            with open(f"{model_path}_{self.ensemble_type}_word_vectorizer.pkl", 'rb') as f:
                self.word_vectorizer = pickle.load(f)
                
            with open(f"{model_path}_{self.ensemble_type}_char_vectorizer.pkl", 'rb') as f:
                self.char_vectorizer = pickle.load(f)
            
            with open(f"{model_path}_{self.ensemble_type}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
        plt.title(f'{self.ensemble_type.title()} Ensemble Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{self.ensemble_type.title()} ensemble confusion matrix plot saved to {save_path}")
        
        plt.show()


def compare_ensembles(human_file: str, ai_file: str, ensembles: List[str] = None, 
                     test_size: float = 0.2, validation_size: float = 0.15) -> Dict[str, Dict[str, Any]]:
    """Compare multiple ensemble classifiers using the same data split."""
    if ensembles is None:
        ensembles = ['voting', 'bagging', 'stacking', 'extra_trees']
        
        # Add gradient boosting methods if available
        if XGBOOST_AVAILABLE:
            ensembles.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            ensembles.append('lightgbm')
        if CATBOOST_AVAILABLE:
            ensembles.append('catboost')
            
        ensembles.append('custom_ensemble')
    
    results = {}
    
    print("=== Comparing Ensemble ML Classifiers ===")
    print(f"Human file: {human_file}")
    print(f"AI file: {ai_file}")
    print(f"Ensembles to test: {ensembles}")
    print("=" * 60)
    
    for ensemble_type in ensembles:
        print(f"\nTraining {ensemble_type} ensemble...")
        try:
            classifier = EnsembleTextClassifier(
                ensemble_type=ensemble_type,
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
            
            results[ensemble_type] = result
            
        except Exception as e:
            print(f"Error training {ensemble_type}: {e}")
            results[ensemble_type] = {'error': str(e)}
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("ENSEMBLE COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Ensemble':<20} {'CV Acc':<10} {'Test Acc':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 80)
    
    for ensemble_type, result in results.items():
        if 'error' not in result:
            cv_acc = result['cv_mean']
            test_acc = result['test_accuracy']
            precision = result['test_precision']
            recall = result['test_recall']
            f1 = result['test_f1']
            auc = result.get('test_auc', 0.0)
            
            print(f"{ensemble_type:<20} {cv_acc:<10.4f} {test_acc:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {auc:<10.4f}")
        else:
            print(f"{ensemble_type:<20} ERROR: {result['error']}")
    
    return results
