"""Optimized comprehensive comparison script for all ML methods.

This script trains and compares neural networks, classical ML, and ensemble methods
for AI vs Human text detection using shared feature extraction for efficiency.
Features are extracted once and reused across all compatible models.
"""
import sys
import os
import argparse
import json
import time
import gc
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings

# Suppress all sklearn numerical warnings globally
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*divide by zero encountered.*')
warnings.filterwarnings('ignore', message='.*overflow encountered.*')
warnings.filterwarnings('ignore', message='.*invalid value encountered.*')
warnings.filterwarnings('ignore', message='.*divide by zero encountered in matmul.*')
warnings.filterwarnings('ignore', message='.*overflow encountered in matmul.*')
warnings.filterwarnings('ignore', message='.*invalid value encountered in matmul.*')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.text_classifier import EnhancedAIHumanTextClassifier
from src.ml.classical_classifiers import ClassicalTextClassifier
from src.ml.ensemble_classifiers import EnsembleTextClassifier
from src.ml.sequential_classifier import SequentialTextClassifier
from src.ml.hybrid_classifier import HybridTextClassifier
from src.ml.deep_learning_classifiers import DeepLearningTextClassifier
from src.ml.probabilistic_classifiers import ProbabilisticTextClassifier
from src.ml.manifold_classifiers import ManifoldTextClassifier
from src.ml.advanced_classifiers import AdvancedTextClassifier
from src.ml.interpretable_classifiers import InterpretableTextClassifier


class SharedFeatureExtractor:
    """Shared feature extraction for all compatible models."""
    
    def __init__(self, max_features: int = 15000, ngram_range: Tuple[int, int] = (1, 3)):
        """Initialize the shared feature extractor."""
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.standard_scaler = None
        self.minmax_scaler = None
        
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
    
    def extract_features(self, texts: List[str], reduced_features: bool = True) -> np.ndarray:
        """Extract comprehensive features with numerical stability improvements."""
        max_features = self.max_features // 2 if reduced_features else self.max_features
        ngram_range = (1, 2) if reduced_features else self.ngram_range
        
        # Word-level TF-IDF features
        if self.word_vectorizer is None:
            self.word_vectorizer = TfidfVectorizer(
                max_features=max_features // 2,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode',
                token_pattern=r'\b[a-zA-Z]{2,}\b',
                min_df=5,
                max_df=0.8,
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
                max_features=max_features // 2,
                analyzer='char',
                ngram_range=(2, 4),
                lowercase=True,
                min_df=10,
                max_df=0.85,
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
        
        # Ensure all feature matrices are clean
        word_features = np.nan_to_num(word_features, nan=0.0, posinf=1e6, neginf=-1e6)
        char_features = np.nan_to_num(char_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Combine all features
        combined_features = np.hstack([word_features, char_features, linguistic_features])
        
        # Final safety check for the combined features
        combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return combined_features
    
    def get_scaled_features(self, features: np.ndarray, scaler_type: str = 'standard') -> np.ndarray:
        """Get scaled features using the appropriate scaler with numerical stability."""
        # Ensure features are clean before scaling
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if scaler_type == 'minmax':
            if self.minmax_scaler is None:
                self.minmax_scaler = MinMaxScaler(feature_range=(0.01, 0.99))  # Avoid exact 0/1 for stability
                scaled_features = self.minmax_scaler.fit_transform(features)
            else:
                scaled_features = self.minmax_scaler.transform(features)
        else:  # standard
            if self.standard_scaler is None:
                self.standard_scaler = StandardScaler(with_mean=True, with_std=True)
                scaled_features = self.standard_scaler.fit_transform(features)
            else:
                scaled_features = self.standard_scaler.transform(features)
        
        # Final cleanup of scaled features
        scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # For naive_bayes, ensure all values are positive and non-zero
        if scaler_type == 'minmax':
            scaled_features = np.clip(scaled_features, 1e-10, 1.0)
        
        return scaled_features


def train_feature_based_method(classifier_class, method_type: str, method_name: str,
                              X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                              reduced_features: bool, reduced_cv: bool,
                              feature_extractor: SharedFeatureExtractor,
                              save_model: bool, model_save_path: str,
                              start_time: float) -> Dict[str, Any]:
    """Helper function to train feature-based methods with shared features."""
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
    
    # Initialize classifier with appropriate parameters
    if method_type == 'classical':
        classifier = classifier_class(
            classifier_type=method_name,
            max_features=10000 if reduced_features else 15000,
            ngram_range=(1, 2) if reduced_features else (1, 3),
            use_hyperparameter_tuning=not reduced_features
        )
    elif method_type == 'ensemble':
        classifier = classifier_class(
            ensemble_type=method_name,
            max_features=8000 if reduced_features else 15000,
            ngram_range=(1, 2) if reduced_features else (1, 3),
            use_hyperparameter_tuning=not reduced_features
        )
    else:  # probabilistic, manifold, advanced, interpretable
        classifier = classifier_class(
            classifier_type=method_name,
            max_features=10000 if reduced_features else 15000,
            ngram_range=(1, 2) if reduced_features else (1, 3),
            use_hyperparameter_tuning=not reduced_features
        )
    
    # Set the feature extractor components from shared extractor
    if feature_extractor:
        classifier.word_vectorizer = feature_extractor.word_vectorizer
        classifier.char_vectorizer = feature_extractor.char_vectorizer
        if method_name in ['naive_bayes', 'gaussian_nb', 'bernoulli_nb', 'multinomial_nb', 'complement_nb', 'categorical_nb', 'gaussian_nb_classifier']:
            classifier.scaler = feature_extractor.minmax_scaler
        else:
            classifier.scaler = feature_extractor.standard_scaler
    
    # Get model and parameters
    if method_type == 'classical':
        model, param_grid = classifier._get_classifier_and_params()
    elif method_type == 'ensemble':
        model, param_grid = classifier._get_ensemble_and_params()
    else:
        model, param_grid = classifier._get_classifier_and_params()
    
    # Special handling for gaussian_mixture to apply numerical stability fixes
    if method_name == 'gaussian_mixture':
        from src.ml.probabilistic_classifiers import GaussianMixtureClassifier
        model = GaussianMixtureClassifier()
        param_grid = {
            'n_components': [2, 3, 5, 10],
            'covariance_type': ['full', 'tied', 'diag', 'spherical']
        }
    
    # Handle anomaly detection methods differently
    if hasattr(classifier, 'is_anomaly_detector') and classifier.is_anomaly_detector:
        # For anomaly detectors, we need to use the special training method
        # Separate training data by class
        human_indices = np.where(np.array(y_train) == 0)[0]
        ai_indices = np.where(np.array(y_train) == 1)[0]
        
        X_human = X_train[human_indices]
        X_ai = X_train[ai_indices]
        
        # Train separate models for each class
        if classifier.use_hyperparameter_tuning and param_grid:
            # Human model
            from sklearn.model_selection import GridSearchCV
            grid_search_human = GridSearchCV(
                model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
            )
            grid_search_human.fit(X_human)
            classifier.human_model = grid_search_human.best_estimator_
            
            # AI model
            from sklearn.base import clone
            grid_search_ai = GridSearchCV(
                clone(model), param_grid, cv=3, scoring='accuracy', n_jobs=-1
            )
            grid_search_ai.fit(X_ai)
            classifier.ai_model = grid_search_ai.best_estimator_
            
            best_params = {
                'human_params': grid_search_human.best_params_,
                'ai_params': grid_search_ai.best_params_
            }
        else:
            # Train with default parameters
            from sklearn.base import clone
            classifier.human_model = clone(model)
            classifier.ai_model = clone(model)
            
            classifier.human_model.fit(X_human)
            classifier.ai_model.fit(X_ai)
            best_params = {}
        
        # Set model to None since we use human_model and ai_model for anomaly detection
        classifier.model = None
    else:
        # Train with or without hyperparameter tuning
        if classifier.use_hyperparameter_tuning:
            from sklearn.model_selection import GridSearchCV
            grid_search = GridSearchCV(
                model, param_grid, cv=3 if reduced_cv else 5, scoring='accuracy', 
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            classifier.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            model.fit(X_train, y_train)
            classifier.model = model
            best_params = {}
    
    # Validation and test predictions
    if hasattr(classifier, 'is_anomaly_detector') and classifier.is_anomaly_detector:
        # For anomaly detectors, we need to use the anomaly detection prediction logic directly
        if classifier.classifier_type == 'local_outlier_factor':
            # LocalOutlierFactor returns -1 for outliers, 1 for inliers
            human_predictions = classifier.human_model.predict(X_val)  # -1 or 1
            ai_predictions = classifier.ai_model.predict(X_val)  # -1 or 1
            
            # Convert to scores: 1 for inlier (normal), 0 for outlier (anomaly)
            human_scores = (human_predictions + 1) / 2  # Convert -1,1 to 0,1
            ai_scores = (ai_predictions + 1) / 2  # Convert -1,1 to 0,1
            
            # Predict the class with higher normality score
            val_predictions = (ai_scores > human_scores).astype(int)
            
            # Create probabilities from normalized scores
            total_scores = human_scores + ai_scores + 1e-8  # Add small epsilon to avoid division by zero
            val_probabilities = ai_scores / total_scores
        else:
            # For other anomaly detectors that have decision_function
            human_scores = classifier.human_model.decision_function(X_val)
            ai_scores = classifier.ai_model.decision_function(X_val)
            
            # Higher score means more normal for that class
            # Predict the class with higher normality score
            val_predictions = (ai_scores > human_scores).astype(int)
            
            # Create pseudo-probabilities from scores
            human_probs = 1 / (1 + np.exp(-human_scores))  # Sigmoid
            ai_probs = 1 / (1 + np.exp(-ai_scores))
            total_probs = human_probs + ai_probs
            val_probabilities = ai_probs / total_probs  # Normalize
        
        # Ensure validation predictions are binary (0 or 1)
        val_predictions = np.clip(val_predictions, 0, 1).astype(int)
        val_accuracy = accuracy_score(y_val, val_predictions)
        
        # Test predictions using the same logic
        if classifier.classifier_type == 'local_outlier_factor':
            human_predictions = classifier.human_model.predict(X_test)  # -1 or 1
            ai_predictions = classifier.ai_model.predict(X_test)  # -1 or 1
            
            human_scores = (human_predictions + 1) / 2  # Convert -1,1 to 0,1
            ai_scores = (ai_predictions + 1) / 2  # Convert -1,1 to 0,1
            
            test_predictions = (ai_scores > human_scores).astype(int)
            
            total_scores = human_scores + ai_scores + 1e-8
            test_probabilities = ai_scores / total_scores
        else:
            human_scores = classifier.human_model.decision_function(X_test)
            ai_scores = classifier.ai_model.decision_function(X_test)
            
            test_predictions = (ai_scores > human_scores).astype(int)
            
            human_probs = 1 / (1 + np.exp(-human_scores))
            ai_probs = 1 / (1 + np.exp(-ai_scores))
            total_probs = human_probs + ai_probs
            test_probabilities = ai_probs / total_probs
        
        # Ensure test predictions are binary (0 or 1)
        test_predictions = np.clip(test_predictions, 0, 1).astype(int)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # For anomaly detection, use a simple accuracy score as CV score
        cv_scores = np.array([test_accuracy])
    else:
        # Regular model predictions (classifier.model is not None)
        val_predictions = classifier.model.predict(X_val)
        val_probabilities = classifier.model.predict_proba(X_val)[:, 1] if hasattr(classifier.model, 'predict_proba') else None
        val_accuracy = accuracy_score(y_val, val_predictions)
        
        test_predictions = classifier.model.predict(X_test)
        test_probabilities = classifier.model.predict_proba(X_test)[:, 1] if hasattr(classifier.model, 'predict_proba') else None
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Cross-validation score
        cv_scores = cross_val_score(classifier.model, X_train, y_train, cv=3 if reduced_cv else 5, scoring='accuracy')
    
    # Calculate metrics
    class_report = classification_report(y_test, test_predictions, 
                                       target_names=['Human', 'AI'], 
                                       labels=[0, 1], output_dict=True)
    
    result = {
        f'{method_type}_type': method_name,
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
        'feature_count': X_train.shape[1]
    }
    
    # Add AUC if probabilities available
    if test_probabilities is not None:
        result['test_auc'] = roc_auc_score(y_test, test_probabilities)
    
    training_time = time.time() - start_time
    result['method'] = f'{method_type.title()}: {method_name}'
    result['training_time'] = training_time
    
    # Save model automatically
    if save_model and model_save_path and classifier:
        try:
            model_path = f"{model_save_path}_{method_name}"
            classifier.save_model(model_path)
            print(f"Saved {method_type} model to {model_path}")
        except Exception as e:
            print(f"Failed to save {method_type} model: {e}")
    
    return result


def train_single_method(method_type: str, method_name: str, 
                       X_train: np.ndarray = None, X_val: np.ndarray = None, X_test: np.ndarray = None,
                       y_train: np.ndarray = None, y_val: np.ndarray = None, y_test: np.ndarray = None,
                       texts: List[str] = None, labels: List[int] = None,
                       human_file: str = None, ai_file: str = None,
                       test_size: float = 0.2, validation_size: float = 0.15,
                       reduced_features: bool = True, reduced_cv: bool = True,
                       save_model: bool = False, model_save_path: str = None,
                       feature_extractor: SharedFeatureExtractor = None) -> Dict[str, Any]:
    """Train a single method with pre-extracted features or file-based training.
    
    Args:
        method_type: Type of method ('neural', 'classical', 'ensemble', etc.).
        method_name: Specific method name within the type.
        X_train, X_val, X_test: Pre-extracted feature matrices (optional).
        y_train, y_val, y_test: Pre-split labels (optional).
        texts: Raw texts for methods that need them (optional).
        labels: Labels corresponding to texts (optional).
        human_file: Path to JSONL file with human-written texts (fallback).
        ai_file: Path to JSONL file with AI-generated texts (fallback).
        test_size: Proportion of data for testing.
        validation_size: Proportion of training data for validation.
        reduced_features: Whether to use reduced feature set for memory.
        reduced_cv: Whether to use reduced cross-validation folds.
        save_model: Whether to save the trained model.
        model_save_path: Base path for saving models.
        feature_extractor: Shared feature extractor instance.
        
    Returns:
        Dictionary containing training results and metrics.
    """
    print(f"Training {method_type}: {method_name}")
    start_time = time.time()
    
    classifier = None  # Keep reference for saving
    
    try:
        # Force garbage collection before training
        gc.collect()
        
        if method_type == 'neural':
            classifier = EnhancedAIHumanTextClassifier(
                max_features=10000 if reduced_features else 15000,  # Reduce features to save memory
                ngram_range=(1, 2) if reduced_features else (1, 3)  # Reduce n-gram complexity
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size,
                epochs=100 if reduced_features else 150,  # Reduce epochs for faster training
                batch_size=16 if reduced_features else 32,  # Smaller batch size
                learning_rate=0.001,
                patience=10 if reduced_features else 15,  # Reduce patience
                dropout_rate=0.3,
                weight_decay=1e-4
            )
            
            training_time = time.time() - start_time
            
            # Calculate F1 score properly
            test_f1 = result.get('test_f1')
            if test_f1 is None:
                # Calculate F1 from precision and recall if not available
                precision = result.get('test_precision', 0.0)
                recall = result.get('test_recall', 0.0)
                if precision + recall > 0:
                    test_f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    test_f1 = 0.0
            
            # Save model automatically with comprehensive metrics
            if save_model and model_save_path and classifier:
                try:
                    model_path = f"{model_save_path}_{method_name}"
                    
                    # Create comprehensive metrics for saving
                    comprehensive_metrics = {
                        'test_accuracy': result['test_accuracy'],
                        'test_precision': result['test_precision'],
                        'test_recall': result['test_recall'],
                        'test_f1': test_f1,
                        'test_auc': result.get('test_auc', 0.0),
                        'feature_count': result['feature_count'],
                        'training_time': training_time,
                        'cv_mean': 0.0,  # Neural network doesn't use CV
                        'cv_std': 0.0,
                        'confusion_matrix': result['confusion_matrix'].tolist() if hasattr(result['confusion_matrix'], 'tolist') else result['confusion_matrix'],
                        'method': f'{method_type.title()}: {method_name}',
                        'model_type': method_type,
                        'classifier_name': method_name
                    }
                    
                    # Save model with metrics
                    classifier.save_model_with_metrics(model_path, comprehensive_metrics)
                    print(f"Saved neural network model with metrics to {model_path}")
                except Exception as e:
                    print(f"Failed to save neural network model: {e}")
            
            return {
                'method': f'{method_type.title()}: {method_name}',
                'test_accuracy': result['test_accuracy'],
                'test_precision': result['test_precision'],
                'test_recall': result['test_recall'],
                'test_f1': test_f1,
                'test_auc': result.get('test_auc', 0.0),
                'feature_count': result['feature_count'],
                'training_time': training_time,
                'cv_mean': 0.0,  # Neural network doesn't use CV
                'cv_std': 0.0,
                'best_params': {},
                'confusion_matrix': result['confusion_matrix'],
                'error': None
            }
            
        elif method_type == 'classical':
            # Use pre-extracted features if available
            if X_train is not None and y_train is not None:
                from sklearn.model_selection import cross_val_score
                from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
                
                classifier = ClassicalTextClassifier(
                    classifier_type=method_name,
                    max_features=10000 if reduced_features else 15000,
                    ngram_range=(1, 2) if reduced_features else (1, 3),
                    use_hyperparameter_tuning=not reduced_features
                )
                
                # Set the feature extractor components from shared extractor
                if feature_extractor:
                    classifier.word_vectorizer = feature_extractor.word_vectorizer
                    classifier.char_vectorizer = feature_extractor.char_vectorizer
                    if method_name == 'naive_bayes':
                        classifier.scaler = feature_extractor.minmax_scaler
                    else:
                        classifier.scaler = feature_extractor.standard_scaler
                
                # Get classifier and parameters
                model, param_grid = classifier._get_classifier_and_params()
                
                # Train with or without hyperparameter tuning
                if classifier.use_hyperparameter_tuning:
                    from sklearn.model_selection import GridSearchCV
                    grid_search = GridSearchCV(
                        model, param_grid, cv=3 if reduced_cv else 5, scoring='accuracy', 
                        n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    classifier.model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                else:
                    model.fit(X_train, y_train)
                    classifier.model = model
                    best_params = {}
                
                # Validation and test predictions
                val_predictions = classifier.model.predict(X_val)
                val_probabilities = classifier.model.predict_proba(X_val)[:, 1] if hasattr(classifier.model, 'predict_proba') else None
                val_accuracy = accuracy_score(y_val, val_predictions)
                
                test_predictions = classifier.model.predict(X_test)
                test_probabilities = classifier.model.predict_proba(X_test)[:, 1] if hasattr(classifier.model, 'predict_proba') else None
                test_accuracy = accuracy_score(y_test, test_predictions)
                
                # Cross-validation score
                cv_scores = cross_val_score(classifier.model, X_train, y_train, cv=3 if reduced_cv else 5, scoring='accuracy')
                
                # Calculate metrics
                class_report = classification_report(y_test, test_predictions, 
                                                   target_names=['Human', 'AI'], output_dict=True)
                
                result = {
                    'classifier_type': method_name,
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
                    'feature_count': X_train.shape[1]
                }
                
                # Add AUC if probabilities available
                if test_probabilities is not None:
                    result['test_auc'] = roc_auc_score(y_test, test_probabilities)
                
                training_time = time.time() - start_time
                result['method'] = f'{method_type.title()}: {method_name}'
                result['training_time'] = training_time
                
                # Save model automatically
                if save_model and model_save_path and classifier:
                    try:
                        model_path = f"{model_save_path}_{method_name}"
                        classifier.save_model(model_path)
                        print(f"Saved classical model to {model_path}")
                    except Exception as e:
                        print(f"Failed to save classical model: {e}")
                
                return result
            else:
                # Fallback to file-based training
                classifier = ClassicalTextClassifier(
                    classifier_type=method_name,
                    max_features=10000 if reduced_features else 15000,
                    ngram_range=(1, 2) if reduced_features else (1, 3),
                    use_hyperparameter_tuning=not reduced_features
                )
                
                result = classifier.train_from_files(
                    human_file=human_file,
                    ai_file=ai_file,
                    test_size=test_size,
                    validation_size=validation_size,
                    cv_folds=3 if reduced_cv else 5
                )
                
                training_time = time.time() - start_time
                result['method'] = f'{method_type.title()}: {method_name}'
                result['training_time'] = training_time
                
                # Save model automatically
                if save_model and model_save_path and classifier:
                    try:
                        model_path = f"{model_save_path}_{method_name}"
                        classifier.save_model(model_path)
                        print(f"Saved classical model to {model_path}")
                    except Exception as e:
                        print(f"Failed to save classical model: {e}")
                
                return result
            
        elif method_type == 'ensemble':
            # Use pre-extracted features if available
            if X_train is not None and y_train is not None:
                from sklearn.model_selection import cross_val_score
                from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
                
                classifier = EnsembleTextClassifier(
                    ensemble_type=method_name,
                    max_features=8000 if reduced_features else 15000,
                    ngram_range=(1, 2) if reduced_features else (1, 3),
                    use_hyperparameter_tuning=not reduced_features
                )
                
                # Set the feature extractor components from shared extractor
                if feature_extractor:
                    classifier.word_vectorizer = feature_extractor.word_vectorizer
                    classifier.char_vectorizer = feature_extractor.char_vectorizer
                    classifier.scaler = feature_extractor.standard_scaler
                
                # Get ensemble and parameters
                model, param_grid = classifier._get_ensemble_and_params()
                
                # Train with or without hyperparameter tuning
                if classifier.use_hyperparameter_tuning:
                    from sklearn.model_selection import GridSearchCV
                    grid_search = GridSearchCV(
                        model, param_grid, cv=3 if reduced_cv else 5, scoring='accuracy', 
                        n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    classifier.model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                else:
                    model.fit(X_train, y_train)
                    classifier.model = model
                    best_params = {}
                
                # Validation and test predictions
                val_predictions = classifier.model.predict(X_val)
                val_probabilities = classifier.model.predict_proba(X_val)[:, 1] if hasattr(classifier.model, 'predict_proba') else None
                val_accuracy = accuracy_score(y_val, val_predictions)
                
                test_predictions = classifier.model.predict(X_test)
                test_probabilities = classifier.model.predict_proba(X_test)[:, 1] if hasattr(classifier.model, 'predict_proba') else None
                test_accuracy = accuracy_score(y_test, test_predictions)
                
                # Cross-validation score
                cv_scores = cross_val_score(classifier.model, X_train, y_train, cv=3 if reduced_cv else 5, scoring='accuracy')
                
                # Calculate metrics
                class_report = classification_report(y_test, test_predictions, 
                                                   target_names=['Human', 'AI'], output_dict=True)
                
                result = {
                    'ensemble_type': method_name,
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
                    'feature_count': X_train.shape[1]
                }
                
                # Add AUC if probabilities available
                if test_probabilities is not None:
                    result['test_auc'] = roc_auc_score(y_test, test_probabilities)
                
                training_time = time.time() - start_time
                result['method'] = f'{method_type.title()}: {method_name}'
                result['training_time'] = training_time
                
                # Save model automatically
                if save_model and model_save_path and classifier:
                    try:
                        model_path = f"{model_save_path}_{method_name}"
                        classifier.save_model(model_path)
                        print(f"Saved ensemble model to {model_path}")
                    except Exception as e:
                        print(f"Failed to save ensemble model: {e}")
                
                return result
            else:
                # Fallback to file-based training
                classifier = EnsembleTextClassifier(
                    ensemble_type=method_name,
                    max_features=8000 if reduced_features else 15000,
                    ngram_range=(1, 2) if reduced_features else (1, 3),
                    use_hyperparameter_tuning=not reduced_features
                )
                
                result = classifier.train_from_files(
                    human_file=human_file,
                    ai_file=ai_file,
                    test_size=test_size,
                    validation_size=validation_size,
                    cv_folds=3 if reduced_cv else 5
                )
                
                training_time = time.time() - start_time
                result['method'] = f'{method_type.title()}: {method_name}'
                result['training_time'] = training_time
                
                # Save model automatically
                if save_model and model_save_path and classifier:
                    try:
                        model_path = f"{model_save_path}_{method_name}"
                        classifier.save_model(model_path)
                        print(f"Saved ensemble model to {model_path}")
                    except Exception as e:
                        print(f"Failed to save ensemble model: {e}")
                
                return result
            
        elif method_type == 'sequential':
            classifier = SequentialTextClassifier(
                vocab_size=15000 if reduced_features else 20000,
                max_len=256 if reduced_features else 512
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size,
                epochs=30 if reduced_features else 50
            )
            
            training_time = time.time() - start_time
            
            # Save model automatically
            if save_model and model_save_path and classifier:
                try:
                    model_path = f"{model_save_path}_{method_name}"
                    classifier.save_model(model_path)
                    print(f"Saved sequential model to {model_path}")
                except Exception as e:
                    print(f"Failed to save sequential model: {e}")
            
            return {
                'method': f'{method_type.title()}: {method_name}',
                'test_accuracy': result['test_accuracy'],
                'test_precision': result['test_precision'],
                'test_recall': result['test_recall'],
                'test_f1': result['test_f1'],
                'test_auc': 0.0,  # Sequential classifier doesn't compute AUC
                'feature_count': 0,  # Sequential uses embeddings, not traditional features
                'training_time': training_time,
                'cv_mean': 0.0,  # Sequential doesn't use CV
                'cv_std': 0.0,
                'best_params': {},
                'confusion_matrix': result['confusion_matrix'],
                'error': None
            }
            
        elif method_type == 'hybrid':
            classifier = HybridTextClassifier(
                vocab_size=15000 if reduced_features else 20000,
                max_len=256 if reduced_features else 512,
                max_features=8000 if reduced_features else 15000,
                ngram_range=(1, 2) if reduced_features else (1, 3)
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size,
                epochs=30 if reduced_features else 50
            )
            
            training_time = time.time() - start_time
            
            # Save model automatically
            if save_model and model_save_path and classifier:
                try:
                    model_path = f"{model_save_path}_{method_name}"
                    classifier.save_model(model_path)
                    print(f"Saved hybrid model to {model_path}")
                except Exception as e:
                    print(f"Failed to save hybrid model: {e}")
            
            return {
                'method': f'{method_type.title()}: {method_name}',
                'test_accuracy': result['test_accuracy'],
                'test_precision': result['test_precision'],
                'test_recall': result['test_recall'],
                'test_f1': result['test_f1'],
                'test_auc': 0.0,  # Hybrid classifier doesn't compute AUC
                'feature_count': 0,  # Hybrid uses both embeddings and features
                'training_time': training_time,
                'cv_mean': 0.0,  # Hybrid doesn't use CV
                'cv_std': 0.0,
                'best_params': {},
                'confusion_matrix': result['confusion_matrix'],
                'error': None
            }
            
        elif method_type == 'deep_learning':
            classifier = DeepLearningTextClassifier(
                model_type=method_name,
                max_features=10000 if reduced_features else 15000,
                ngram_range=(1, 2) if reduced_features else (1, 3)
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size,
                epochs=50 if reduced_features else 100,
                batch_size=16 if reduced_features else 32
            )
            
            training_time = time.time() - start_time
            
            # Save model automatically
            if save_model and model_save_path and classifier:
                try:
                    model_path = f"{model_save_path}_{method_name}"
                    classifier.save_model(model_path)
                    print(f"Saved deep learning model to {model_path}")
                except Exception as e:
                    print(f"Failed to save deep learning model: {e}")
            
            return {
                'method': f'{method_type.title()}: {method_name}',
                'test_accuracy': result['test_accuracy'],
                'test_precision': result['test_precision'],
                'test_recall': result['test_recall'],
                'test_f1': result['test_f1'],
                'test_auc': 0.0,  # Deep learning classifier doesn't compute AUC
                'feature_count': 0,  # Deep learning uses embeddings
                'training_time': training_time,
                'cv_mean': 0.0,  # Deep learning doesn't use CV
                'cv_std': 0.0,
                'best_params': {},
                'confusion_matrix': result['confusion_matrix'],
                'error': None
            }
            
        elif method_type in ['probabilistic', 'manifold', 'advanced', 'interpretable']:
            # Use pre-extracted features if available
            if X_train is not None and y_train is not None:
                # Map method types to their classifier classes
                classifier_classes = {
                    'probabilistic': ProbabilisticTextClassifier,
                    'manifold': ManifoldTextClassifier,
                    'advanced': AdvancedTextClassifier,
                    'interpretable': InterpretableTextClassifier
                }
                
                return train_feature_based_method(
                    classifier_classes[method_type], method_type, method_name,
                    X_train, X_val, X_test, y_train, y_val, y_test,
                    reduced_features, reduced_cv, feature_extractor,
                    save_model, model_save_path, start_time
                )
            else:
                # Fallback to file-based training
                classifier_classes = {
                    'probabilistic': ProbabilisticTextClassifier,
                    'manifold': ManifoldTextClassifier,
                    'advanced': AdvancedTextClassifier,
                    'interpretable': InterpretableTextClassifier
                }
                
                max_features_map = {
                    'probabilistic': 10000 if reduced_features else 15000,
                    'manifold': 8000 if reduced_features else 15000,  # Reduced for manifold methods
                    'advanced': 10000 if reduced_features else 15000,
                    'interpretable': 10000 if reduced_features else 15000
                }
                
                classifier = classifier_classes[method_type](
                    classifier_type=method_name,
                    max_features=max_features_map[method_type],
                    ngram_range=(1, 2) if reduced_features else (1, 3),
                    use_hyperparameter_tuning=not reduced_features
                )
                
                result = classifier.train_from_files(
                    human_file=human_file,
                    ai_file=ai_file,
                    test_size=test_size,
                    validation_size=validation_size,
                    cv_folds=3 if reduced_cv else 5
                )
                
                training_time = time.time() - start_time
                result['method'] = f'{method_type.title()}: {method_name}'
                result['training_time'] = training_time
                
                # Save model automatically
                if save_model and model_save_path and classifier:
                    try:
                        model_path = f"{model_save_path}_{method_name}"
                        classifier.save_model(model_path)
                        print(f"Saved {method_type} model to {model_path}")
                    except Exception as e:
                        print(f"Failed to save {method_type} model: {e}")
                
                return result
            
    except Exception as e:
        training_time = time.time() - start_time
        return {
            'method': f'{method_type.title()}: {method_name}',
            'training_time': training_time,
            'error': str(e)
        }
    finally:
        # Force garbage collection after each method
        gc.collect()


def main():
    """Main comparison function with memory-safe processing."""
    parser = argparse.ArgumentParser(description='Memory-safe comparison of all ML methods for AI vs Human text detection')
    
    # File-based arguments
    parser.add_argument('--human-file', type=str, required=True,
                       help='Path to JSONL file containing human-written texts')
    parser.add_argument('--ai-file', type=str, required=True,
                       help='Path to JSONL file containing AI-generated texts')
    
    # Training configuration
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for testing (default: 0.2)')
    parser.add_argument('--validation-size', type=float, default=0.15,
                       help='Proportion of training data for validation (default: 0.15)')
    
    # Memory optimization options
    parser.add_argument('--full-features', action='store_true',
                       help='Use full feature set (may cause memory issues)')
    parser.add_argument('--full-cv', action='store_true',
                       help='Use full cross-validation (may cause memory issues)')
    parser.add_argument('--max-methods', type=int, default=None,
                       help='Maximum number of methods to test (for memory constraints)')
    
    # Method selection
    parser.add_argument('--skip-neural', action='store_true',
                       help='Skip neural network training')
    parser.add_argument('--skip-classical', action='store_true',
                       help='Skip classical ML methods')
    parser.add_argument('--skip-ensemble', action='store_true',
                       help='Skip ensemble methods')
    parser.add_argument('--skip-sequential', action='store_true',
                       help='Skip sequential (LSTM) classifier training')
    parser.add_argument('--skip-hybrid', action='store_true',
                       help='Skip hybrid classifier training')
    parser.add_argument('--skip-deep-learning', action='store_true',
                       help='Skip deep learning classifier training')
    parser.add_argument('--skip-probabilistic', action='store_true',
                       help='Skip probabilistic classifier training')
    parser.add_argument('--skip-manifold', action='store_true',
                       help='Skip manifold learning classifier training')
    parser.add_argument('--skip-advanced', action='store_true',
                       help='Skip advanced classifier training')
    parser.add_argument('--skip-interpretable', action='store_true',
                       help='Skip interpretable classifier training')
    
    # Classical ML methods to test
    parser.add_argument('--classical-methods', nargs='+', 
                       choices=['random_forest', 'svm', 'logistic_regression', 'gradient_boosting', 
                               'naive_bayes', 'knn', 'decision_tree', 'adaboost'],
                       default=['random_forest', 'svm', 'logistic_regression', 'gradient_boosting', 
                               'naive_bayes', 'knn', 'decision_tree', 'adaboost'],
                       help='Classical ML methods to test (default: all methods)')
    
    # Ensemble methods to test
    parser.add_argument('--ensemble-methods', nargs='+',
                       choices=['voting', 'bagging', 'stacking', 'xgboost', 'lightgbm', 
                               'catboost', 'extra_trees', 'custom_ensemble'],
                       default=['voting', 'bagging', 'stacking', 'xgboost', 'lightgbm', 
                               'catboost', 'extra_trees', 'custom_ensemble'],
                       help='Ensemble methods to test (default: all methods)')
    
    # Deep learning methods to test
    parser.add_argument('--deep-learning-methods', nargs='+',
                       choices=['cnn', 'transformer', 'attention_bilstm'],
                       default=['cnn', 'transformer', 'attention_bilstm'],
                       help='Deep learning methods to test (default: all methods)')
    
    # Probabilistic methods to test
    parser.add_argument('--probabilistic-methods', nargs='+',
                       choices=['gaussian_nb', 'bernoulli_nb', 'multinomial_nb', 'complement_nb', 
                               'categorical_nb', 'hmm', 'gaussian_mixture'],
                       default=['gaussian_nb', 'bernoulli_nb', 'multinomial_nb', 'complement_nb', 
                               'categorical_nb', 'hmm', 'gaussian_mixture'],
                       help='Probabilistic methods to test (default: all methods)')
    
    # Manifold learning methods to test
    parser.add_argument('--manifold-methods', nargs='+',
                       choices=['pca', 'tsne', 'isomap', 'lle', 'spectral_embedding', 
                               'mds', 'ica', 'factor_analysis', 'truncated_svd'],
                       default=['pca', 'tsne', 'isomap', 'lle', 'spectral_embedding', 
                               'mds', 'ica', 'factor_analysis', 'truncated_svd'],
                       help='Manifold learning methods to test (default: all methods)')
    
    # Advanced methods to test
    parser.add_argument('--advanced-methods', nargs='+',
                       choices=['isolation_forest', 'one_class_svm', 'local_outlier_factor', 
                               'elliptic_envelope', 'sgd', 'passive_aggressive', 'perceptron', 
                               'ridge', 'lasso', 'elastic_net', 'huber', 'quantile', 'tweedie'],
                       default=['isolation_forest', 'one_class_svm', 'local_outlier_factor', 
                               'elliptic_envelope', 'sgd', 'passive_aggressive', 'perceptron', 
                               'ridge', 'lasso', 'elastic_net', 'huber', 'quantile', 'tweedie'],
                       help='Advanced methods to test (default: all methods)')
    
    # Interpretable methods to test
    parser.add_argument('--interpretable-methods', nargs='+',
                       choices=['linear_regression', 'lasso_regression', 'ridge_regression', 
                               'elastic_net_regression', 'decision_tree_classifier', 
                               'extra_tree_classifier', 'gaussian_nb_classifier'],
                       default=['linear_regression', 'lasso_regression', 'ridge_regression', 
                               'elastic_net_regression', 'decision_tree_classifier', 
                               'extra_tree_classifier', 'gaussian_nb_classifier'],
                       help='Interpretable methods to test (default: all methods)')
    
    # Output options
    parser.add_argument('--output-file', type=str, default='comparison_results_safe.json',
                       help='File to save detailed results (default: comparison_results_safe.json)')
    parser.add_argument('--no-save-models', action='store_true',
                       help='Do NOT save trained models to disk (saves storage space)')
    parser.add_argument('--model-save-path', type=str, default='models/comparison',
                       help='Base path for saving models (default: models/comparison)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information during training')
    
    args = parser.parse_args()
    
    # Validate files
    if not os.path.exists(args.human_file):
        print(f"Error: Human text file '{args.human_file}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(args.ai_file):
        print(f"Error: AI text file '{args.ai_file}' does not exist.")
        sys.exit(1)
    
    # Memory optimization settings
    reduced_features = not args.full_features
    reduced_cv = not args.full_cv
    save_models = not args.no_save_models
    
    print("ML Methods Comparison")
    print("="*50)
    print(f"Human text file: {args.human_file}")
    print(f"AI text file: {args.ai_file}")
    print(f"Test size: {args.test_size}")
    print(f"Validation size: {args.validation_size}")
    print(f"Memory optimization: {'Disabled' if args.full_features else 'Enabled'}")
    print(f"Reduced CV folds: {'No' if args.full_cv else 'Yes (3 instead of 5)'}")
    print(f"Save models: {'No' if args.no_save_models else 'Yes'} (to {args.model_save_path})")
    
    # Build list of methods to test
    methods_to_test = []
    
    if not args.skip_neural:
        methods_to_test.append(('neural', 'enhanced_neural_network'))
    
    if not args.skip_classical:
        for method in args.classical_methods:
            methods_to_test.append(('classical', method))
    
    if not args.skip_ensemble:
        for method in args.ensemble_methods:
            methods_to_test.append(('ensemble', method))
    
    if not args.skip_sequential:
        methods_to_test.append(('sequential', 'lstm_attention'))
    
    if not args.skip_hybrid:
        methods_to_test.append(('hybrid', 'lstm_features'))
    
    if not args.skip_deep_learning:
        for method in args.deep_learning_methods:
            methods_to_test.append(('deep_learning', method))
    
    if not args.skip_probabilistic:
        for method in args.probabilistic_methods:
            methods_to_test.append(('probabilistic', method))
    
    if not args.skip_manifold:
        for method in args.manifold_methods:
            methods_to_test.append(('manifold', method))
    
    if not args.skip_advanced:
        for method in args.advanced_methods:
            methods_to_test.append(('advanced', method))
    
    if not args.skip_interpretable:
        for method in args.interpretable_methods:
            methods_to_test.append(('interpretable', method))
    
    # Limit methods if specified
    if args.max_methods and len(methods_to_test) > args.max_methods:
        methods_to_test = methods_to_test[:args.max_methods]
        print(f"Limited to {args.max_methods} methods due to --max-methods constraint")
    
    print(f"Methods to test: {len(methods_to_test)}")
    if reduced_features:
        print("Using reduced feature set for memory efficiency")
    if reduced_cv:
        print("Using reduced cross-validation for memory efficiency")
    print("="*50)
    
    all_results = {}
    total_start_time = time.time()
    
    try:
        # Separate methods that can use shared features from those that need raw text
        feature_based_methods = []
        text_based_methods = []
        
        for method_type, method_name in methods_to_test:
            if method_type in ['classical', 'ensemble', 'probabilistic', 'manifold', 'advanced', 'interpretable']:
                feature_based_methods.append((method_type, method_name))
            else:
                # Neural, sequential, hybrid, deep_learning need raw text
                text_based_methods.append((method_type, method_name))
        
        print(f"Feature-based methods: {len(feature_based_methods)}")
        print(f"Text-based methods: {len(text_based_methods)}")
        print("="*50)
        
        # Extract features once for all feature-based methods
        shared_features = None
        X_train_std = X_val_std = X_test_std = None
        X_train_mm = X_val_mm = X_test_mm = None
        y_train = y_val = y_test = None
        texts = labels = None
        
        if feature_based_methods:
            print("Extracting shared features for feature-based methods...")
            feature_extractor = SharedFeatureExtractor(
                max_features=10000 if reduced_features else 15000,
                ngram_range=(1, 2) if reduced_features else (1, 3)
            )
            
            # Load data once
            texts, labels = feature_extractor.load_corpus_files(args.human_file, args.ai_file)
            
            # Extract features once
            shared_features = feature_extractor.extract_features(texts, reduced_features)
            print(f"Extracted {shared_features.shape[1]} shared features")
            
            # Split data once
            X_temp, X_test, y_temp, y_test = train_test_split(
                shared_features, labels, test_size=args.test_size, random_state=42, stratify=labels
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=args.validation_size, random_state=42, stratify=y_temp
            )
            
            # Pre-scale features with both scalers
            X_train_std = feature_extractor.get_scaled_features(X_train, 'standard')
            X_val_std = feature_extractor.get_scaled_features(X_val, 'standard')
            X_test_std = feature_extractor.get_scaled_features(X_test, 'standard')
            
            X_train_mm = feature_extractor.get_scaled_features(X_train, 'minmax')
            X_val_mm = feature_extractor.get_scaled_features(X_val, 'minmax')
            X_test_mm = feature_extractor.get_scaled_features(X_test, 'minmax')
            
            print(f"Training set: {len(X_train)} samples")
            print(f"Validation set: {len(X_val)} samples")
            print(f"Test set: {len(X_test)} samples")
            print("="*50)
        
        # Train feature-based methods with shared features
        for i, (method_type, method_name) in enumerate(feature_based_methods, 1):
            print(f"Progress (Feature-based): {i}/{len(feature_based_methods)} methods")
            
            # Choose appropriate scaling for the method
            if method_name in ['naive_bayes', 'gaussian_nb', 'bernoulli_nb', 'multinomial_nb', 'complement_nb', 'categorical_nb', 'gaussian_nb_classifier']:
                X_train_scaled = X_train_mm
                X_val_scaled = X_val_mm
                X_test_scaled = X_test_mm
            else:
                X_train_scaled = X_train_std
                X_val_scaled = X_val_std
                X_test_scaled = X_test_std
            
            result = train_single_method(
                method_type=method_type,
                method_name=method_name,
                X_train=X_train_scaled,
                X_val=X_val_scaled,
                X_test=X_test_scaled,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                texts=texts,
                labels=labels,
                human_file=args.human_file,
                ai_file=args.ai_file,
                test_size=args.test_size,
                validation_size=args.validation_size,
                reduced_features=reduced_features,
                reduced_cv=reduced_cv,
                save_model=save_models,
                model_save_path=args.model_save_path,
                feature_extractor=feature_extractor
            )
            
            result_key = f"{method_type}_{method_name}"
            all_results[result_key] = result
            
            # Print immediate results
            if 'error' not in result or result.get('error') is None:
                print(f"Completed {result['method']}: {result['test_accuracy']:.4f} accuracy")
            else:
                print(f"Failed {result['method']}: {result['error']}")
            
            # Force garbage collection between methods
            gc.collect()
        
        # Train text-based methods individually (they need raw text)
        for i, (method_type, method_name) in enumerate(text_based_methods, 1):
            print(f"Progress (Text-based): {i}/{len(text_based_methods)} methods")
            
            result = train_single_method(
                method_type=method_type,
                method_name=method_name,
                human_file=args.human_file,
                ai_file=args.ai_file,
                test_size=args.test_size,
                validation_size=args.validation_size,
                reduced_features=reduced_features,
                reduced_cv=reduced_cv,
                save_model=save_models,
                model_save_path=args.model_save_path
            )
            
            result_key = f"{method_type}_{method_name}"
            all_results[result_key] = result
            
            # Print immediate results
            if 'error' not in result or result.get('error') is None:
                print(f"Completed {result['method']}: {result['test_accuracy']:.4f} accuracy")
            else:
                print(f"Failed {result['method']}: {result['error']}")
            
            # Force garbage collection between methods
            gc.collect()
        
        total_time = time.time() - total_start_time
        
        # Generate comparison report
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        
        # Create results table
        successful_results = {k: v for k, v in all_results.items() if 'error' not in v or v.get('error') is None}
        
        if successful_results:
            print(f"{'Method':<35} {'Test Acc':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10} {'CV Acc':<10} {'Time(s)':<10}")
            print("-"*80)
            
            # Sort by test accuracy
            sorted_results = sorted(successful_results.items(), 
                                  key=lambda x: x[1].get('test_accuracy', 0), reverse=True)
            
            for method_key, result in sorted_results:
                method_name = result.get('method', method_key)
                test_acc = result.get('test_accuracy', 0.0)
                precision = result.get('test_precision', 0.0)
                recall = result.get('test_recall', 0.0)
                f1 = result.get('test_f1', 0.0)
                auc = result.get('test_auc', 0.0)
                cv_acc = result.get('cv_mean', 0.0)
                train_time = result.get('training_time', 0.0)
                
                print(f"{method_name:<35} {test_acc:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {auc:<10.4f} {cv_acc:<10.4f} {train_time:<10.0f}")
            
            # Identify best performing method
            best_method_key, best_result = sorted_results[0]
            best_method_name = best_result.get('method', best_method_key)
            best_accuracy = best_result.get('test_accuracy', 0.0)
            
            print("\n" + "="*80)
            print("BEST PERFORMING METHOD")
            print("="*80)
            print(f"Method: {best_method_name}")
            print(f"Test Accuracy: {best_accuracy:.4f}")
            print(f"Test Precision: {best_result.get('test_precision', 0.0):.4f}")
            print(f"Test Recall: {best_result.get('test_recall', 0.0):.4f}")
            print(f"Test F1-Score: {best_result.get('test_f1', 0.0):.4f}")
            if best_result.get('test_auc', 0.0) > 0:
                print(f"Test AUC: {best_result.get('test_auc', 0.0):.4f}")
            print(f"Training Time: {best_result.get('training_time', 0.0):.0f} seconds")
            
            # Performance categories
            print("\nPERFORMANCE ANALYSIS:")
            excellent = [k for k, v in successful_results.items() if v.get('test_accuracy', 0) >= 0.90]
            good = [k for k, v in successful_results.items() if 0.85 <= v.get('test_accuracy', 0) < 0.90]
            fair = [k for k, v in successful_results.items() if 0.80 <= v.get('test_accuracy', 0) < 0.85]
            poor = [k for k, v in successful_results.items() if v.get('test_accuracy', 0) < 0.80]
            
            if excellent:
                print(f"Excellent (>=90%): {len(excellent)} methods")
            if good:
                print(f"Good (85-90%): {len(good)} methods")
            if fair:
                print(f"Fair (80-85%): {len(fair)} methods")
            if poor:
                print(f"Poor (<80%): {len(poor)} methods")
            
            # Method family analysis
            neural_results = [v for k, v in successful_results.items() if 'neural' in k.lower()]
            classical_results = [v for k, v in successful_results.items() if 'classical' in k.lower()]
            ensemble_results = [v for k, v in successful_results.items() if 'ensemble' in k.lower()]
            sequential_results = [v for k, v in successful_results.items() if 'sequential' in k.lower()]
            hybrid_results = [v for k, v in successful_results.items() if 'hybrid' in k.lower()]
            deep_learning_results = [v for k, v in successful_results.items() if 'deep_learning' in k.lower()]
            probabilistic_results = [v for k, v in successful_results.items() if 'probabilistic' in k.lower()]
            manifold_results = [v for k, v in successful_results.items() if 'manifold' in k.lower()]
            advanced_results = [v for k, v in successful_results.items() if 'advanced' in k.lower()]
            interpretable_results = [v for k, v in successful_results.items() if 'interpretable' in k.lower()]
            
            print("\nMETHOD FAMILY ANALYSIS:")
            if neural_results:
                avg_neural = sum(r.get('test_accuracy', 0) for r in neural_results) / len(neural_results)
                print(f"Neural Networks: {len(neural_results)} methods, avg accuracy: {avg_neural:.4f}")
            
            if classical_results:
                avg_classical = sum(r.get('test_accuracy', 0) for r in classical_results) / len(classical_results)
                print(f"Classical ML: {len(classical_results)} methods, avg accuracy: {avg_classical:.4f}")
            
            if ensemble_results:
                avg_ensemble = sum(r.get('test_accuracy', 0) for r in ensemble_results) / len(ensemble_results)
                print(f"Ensemble Methods: {len(ensemble_results)} methods, avg accuracy: {avg_ensemble:.4f}")
            
            if sequential_results:
                avg_sequential = sum(r.get('test_accuracy', 0) for r in sequential_results) / len(sequential_results)
                print(f"Sequential (LSTM): {len(sequential_results)} methods, avg accuracy: {avg_sequential:.4f}")
            
            if hybrid_results:
                avg_hybrid = sum(r.get('test_accuracy', 0) for r in hybrid_results) / len(hybrid_results)
                print(f"Hybrid Models: {len(hybrid_results)} methods, avg accuracy: {avg_hybrid:.4f}")
            
            if deep_learning_results:
                avg_deep_learning = sum(r.get('test_accuracy', 0) for r in deep_learning_results) / len(deep_learning_results)
                print(f"Deep Learning: {len(deep_learning_results)} methods, avg accuracy: {avg_deep_learning:.4f}")
            
            if probabilistic_results:
                avg_probabilistic = sum(r.get('test_accuracy', 0) for r in probabilistic_results) / len(probabilistic_results)
                print(f"Probabilistic: {len(probabilistic_results)} methods, avg accuracy: {avg_probabilistic:.4f}")
            
            if manifold_results:
                avg_manifold = sum(r.get('test_accuracy', 0) for r in manifold_results) / len(manifold_results)
                print(f"Manifold Learning: {len(manifold_results)} methods, avg accuracy: {avg_manifold:.4f}")
            
            if advanced_results:
                avg_advanced = sum(r.get('test_accuracy', 0) for r in advanced_results) / len(advanced_results)
                print(f"Advanced Methods: {len(advanced_results)} methods, avg accuracy: {avg_advanced:.4f}")
            
            if interpretable_results:
                avg_interpretable = sum(r.get('test_accuracy', 0) for r in interpretable_results) / len(interpretable_results)
                print(f"Interpretable: {len(interpretable_results)} methods, avg accuracy: {avg_interpretable:.4f}")
        
        else:
            print("No methods completed successfully")
        
        # Report failed methods
        failed_results = {k: v for k, v in all_results.items() if 'error' in v and v.get('error') is not None}
        if failed_results:
            print(f"\nFAILED METHODS ({len(failed_results)}):")
            for method_key, result in failed_results.items():
                method_name = result.get('method', method_key)
                error = result.get('error', 'Unknown error')
                print(f"   {method_name}: {error}")
        
        # Save detailed results
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        json_results = {}
        for key, result in all_results.items():
            json_result = result.copy()
            # Convert numpy arrays to lists for JSON serialization
            if 'confusion_matrix' in json_result:
                json_result['confusion_matrix'] = json_result['confusion_matrix'].tolist()
            if 'cv_scores' in json_result:
                json_result['cv_scores'] = json_result['cv_scores'].tolist()
            json_results[key] = json_result
        
        # Add metadata
        json_results['_metadata'] = {
            'human_file': args.human_file,
            'ai_file': args.ai_file,
            'test_size': args.test_size,
            'validation_size': args.validation_size,
            'total_training_time': total_time,
            'methods_tested': len(all_results),
            'successful_methods': len(successful_results),
            'failed_methods': len(failed_results),
            'best_method': best_method_name if successful_results else None,
            'best_accuracy': best_accuracy if successful_results else None,
            'memory_optimized': reduced_features,
            'reduced_cv': reduced_cv
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_path}")
        print(f"Total comparison time: {total_time:.0f} seconds ({total_time/60:.1f} minutes)")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
