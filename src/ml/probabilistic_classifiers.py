"""Probabilistic machine learning classifiers for AI vs Human text detection.

This module implements probabilistic approaches including Bayesian methods,
Hidden Markov Models, and other probabilistic classifiers using the same
pipeline as other classifiers.
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
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, MultinomialNB, ComplementNB
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from .model_serializer import ModelPackage, ModelSerializer


class BayesianNetworkClassifier:
    """Simple Bayesian Network classifier for text classification."""
    
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.feature_probs = {}
        self.class_probs = {}
        self.feature_names = []
        
    def fit(self, X, y):
        """Fit the Bayesian Network classifier."""
        n_samples, n_features = X.shape
        classes = np.unique(y)
        
        # Calculate class probabilities
        for cls in classes:
            self.class_probs[cls] = np.sum(y == cls) / n_samples
        
        # Calculate feature probabilities for each class
        for cls in classes:
            class_mask = y == cls
            class_data = X[class_mask]
            
            self.feature_probs[cls] = {}
            for feature_idx in range(n_features):
                feature_values = class_data[:, feature_idx]
                
                # For continuous features, use Gaussian assumption
                mean = np.mean(feature_values)
                std = np.std(feature_values) + 1e-9  # Add small epsilon
                self.feature_probs[cls][feature_idx] = {'mean': mean, 'std': std}
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        n_samples = X.shape[0]
        classes = list(self.class_probs.keys())
        probabilities = np.zeros((n_samples, len(classes)))
        
        for i, sample in enumerate(X):
            for j, cls in enumerate(classes):
                # Start with class prior
                log_prob = np.log(self.class_probs[cls])
                
                # Add feature likelihoods
                for feature_idx, feature_value in enumerate(sample):
                    mean = self.feature_probs[cls][feature_idx]['mean']
                    std = self.feature_probs[cls][feature_idx]['std']
                    
                    # Gaussian likelihood
                    likelihood = (1 / (std * np.sqrt(2 * np.pi))) * \
                               np.exp(-0.5 * ((feature_value - mean) / std) ** 2)
                    log_prob += np.log(likelihood + 1e-9)
                
                probabilities[i, j] = log_prob
        
        # Convert log probabilities to probabilities
        probabilities = np.exp(probabilities)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X):
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


class HiddenMarkovModelClassifier:
    """Hidden Markov Model classifier for sequential text features."""
    
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.models = {}
        
    def _extract_sequences(self, texts: List[str]) -> List[List[int]]:
        """Extract character-level sequences from texts."""
        sequences = []
        for text in texts:
            # Convert to character codes and normalize
            char_sequence = [min(ord(c), 127) for c in text.lower()[:100]]  # Limit length
            sequences.append(char_sequence)
        return sequences
    
    def fit(self, texts: List[str], labels: List[int]):
        """Fit HMM models for each class."""
        sequences = self._extract_sequences(texts)
        classes = np.unique(labels)
        
        for cls in classes:
            class_sequences = [seq for i, seq in enumerate(sequences) if labels[i] == cls]
            
            # Simple HMM implementation
            self.models[cls] = self._fit_hmm(class_sequences)
    
    def _fit_hmm(self, sequences: List[List[int]]) -> Dict[str, Any]:
        """Fit a simple HMM to sequences."""
        # Transition probabilities
        transitions = {}
        emissions = {}
        
        for seq in sequences:
            for i in range(len(seq) - 1):
                current_state = seq[i] % self.n_states
                next_state = seq[i + 1] % self.n_states
                
                if current_state not in transitions:
                    transitions[current_state] = {}
                if next_state not in transitions[current_state]:
                    transitions[current_state][next_state] = 0
                transitions[current_state][next_state] += 1
                
                # Emission probabilities
                if current_state not in emissions:
                    emissions[current_state] = {}
                if seq[i] not in emissions[current_state]:
                    emissions[current_state][seq[i]] = 0
                emissions[current_state][seq[i]] += 1
        
        # Normalize probabilities
        for state in transitions:
            total = sum(transitions[state].values())
            for next_state in transitions[state]:
                transitions[state][next_state] /= total
        
        for state in emissions:
            total = sum(emissions[state].values())
            for emission in emissions[state]:
                emissions[state][emission] /= total
        
        return {'transitions': transitions, 'emissions': emissions}
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities using HMM models."""
        sequences = self._extract_sequences(texts)
        classes = list(self.models.keys())
        probabilities = np.zeros((len(texts), len(classes)))
        
        for i, seq in enumerate(sequences):
            for j, cls in enumerate(classes):
                prob = self._sequence_probability(seq, self.models[cls])
                probabilities[i, j] = prob
        
        # Normalize probabilities
        probabilities = probabilities / (probabilities.sum(axis=1, keepdims=True) + 1e-9)
        return probabilities
    
    def _sequence_probability(self, sequence: List[int], model: Dict[str, Any]) -> float:
        """Calculate probability of sequence given HMM model."""
        if len(sequence) == 0:
            return 1e-9
        
        log_prob = 0.0
        transitions = model['transitions']
        emissions = model['emissions']
        
        for i in range(len(sequence) - 1):
            current_state = sequence[i] % self.n_states
            next_state = sequence[i + 1] % self.n_states
            
            # Transition probability
            if current_state in transitions and next_state in transitions[current_state]:
                log_prob += np.log(transitions[current_state][next_state] + 1e-9)
            else:
                log_prob += np.log(1e-9)
            
            # Emission probability
            if current_state in emissions and sequence[i] in emissions[current_state]:
                log_prob += np.log(emissions[current_state][sequence[i]] + 1e-9)
            else:
                log_prob += np.log(1e-9)
        
        return np.exp(log_prob)
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict class labels."""
        probabilities = self.predict_proba(texts)
        return np.argmax(probabilities, axis=1)


class ProbabilisticTextClassifier:
    """Probabilistic ML classifier using various probabilistic approaches."""
    
    def __init__(self, classifier_type: str = 'gaussian_nb', max_features: int = 15000, 
                 ngram_range: Tuple[int, int] = (1, 3), use_hyperparameter_tuning: bool = True):
        """Initialize the probabilistic classifier.
        
        Args:
            classifier_type: Type of classifier ('gaussian_nb', 'bernoulli_nb', 'categorical_nb',
                           'gaussian_mixture', 'lda', 'qda', 'bayesian_network', 'hmm')
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
        
    def extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract linguistic features from texts."""
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
            
            # Readability approximation
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
            text_features.append(len(unique_bigrams) / len(bigrams) if len(bigrams) > 0 else 0)
            
            features.append(text_features)
        
        return np.array(features)
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract comprehensive features."""
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
    
    def _get_classifier_and_params(self):
        """Get classifier and hyperparameter grid based on classifier type."""
        if self.classifier_type == 'gaussian_nb':
            classifier = GaussianNB()
            param_grid = {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
            
        elif self.classifier_type == 'bernoulli_nb':
            classifier = BernoulliNB()
            param_grid = {
                'alpha': [0.1, 0.5, 1.0, 2.0],
                'binarize': [0.0, 0.1, 0.5]
            }
            
        elif self.classifier_type == 'multinomial_nb':
            classifier = MultinomialNB()
            param_grid = {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            }
            
        elif self.classifier_type == 'complement_nb':
            classifier = ComplementNB()
            param_grid = {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            }
            
        elif self.classifier_type == 'categorical_nb':
            # CategoricalNB requires discrete features, so we'll use a wrapper
            classifier = CategoricalNBWrapper()
            param_grid = {
                'alpha': [0.1, 0.5, 1.0, 2.0],
                'n_bins': [5, 10, 20, 50]
            }
            
        elif self.classifier_type == 'gaussian_mixture':
            classifier = GaussianMixtureClassifier()
            param_grid = {
                'n_components': [2, 3, 5, 10],
                'covariance_type': ['full', 'tied', 'diag', 'spherical']
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
            
        elif self.classifier_type == 'bayesian_network':
            classifier = BayesianNetworkClassifier()
            param_grid = {
                'smoothing': [0.1, 0.5, 1.0, 2.0]
            }
            
        elif self.classifier_type == 'hmm':
            classifier = HiddenMarkovModelClassifier()
            param_grid = {
                'n_states': [2, 3, 5, 10]
            }
            
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        return classifier, param_grid
    
    def train_from_files(self, human_file: str, ai_file: str, test_size: float = 0.2, 
                        validation_size: float = 0.15, cv_folds: int = 5) -> Dict[str, Any]:
        """Train the probabilistic classifier."""
        # Load data
        texts, labels = self.load_corpus_files(human_file, ai_file)
        
        if len(texts) == 0:
            raise ValueError("No texts provided for training")
        
        # Handle HMM separately as it works with raw texts
        if self.classifier_type == 'hmm':
            return self._train_hmm(texts, labels, test_size, validation_size)
        
        # Extract features
        print("Extracting comprehensive features (word + character + linguistic)...")
        features = self.extract_features(texts)
        print(f"Extracted {features.shape[1]} total features")
        
        # Scale features for some classifiers
        if self.classifier_type in ['gaussian_nb', 'lda', 'qda', 'gaussian_mixture', 'bayesian_network']:
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = features
            self.scaler = None
        
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
    
    def _train_hmm(self, texts: List[str], labels: List[int], test_size: float, validation_size: float) -> Dict[str, Any]:
        """Train HMM classifier separately."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=42, stratify=y_train
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train HMM
        self.model = HiddenMarkovModelClassifier(n_states=3)
        self.model.fit(X_train, y_train)
        
        # Validation predictions
        val_predictions = self.model.predict(X_val)
        val_probabilities = self.model.predict_proba(X_val)[:, 1]
        val_accuracy = accuracy_score(y_val, val_predictions)
        
        # Test predictions
        test_predictions = self.model.predict(X_test)
        test_probabilities = self.model.predict_proba(X_test)[:, 1]
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Calculate metrics
        class_report = classification_report(y_test, test_predictions, 
                                           target_names=['Human', 'AI'], output_dict=True)
        
        results = {
            'classifier_type': self.classifier_type,
            'best_params': {},
            'cv_scores': np.array([test_accuracy]),
            'cv_mean': test_accuracy,
            'cv_std': 0.0,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': class_report['weighted avg']['precision'],
            'test_recall': class_report['weighted avg']['recall'],
            'test_f1': class_report['weighted avg']['f1-score'],
            'test_auc': roc_auc_score(y_test, test_probabilities),
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, test_predictions),
            'feature_count': 0
        }
        
        print(f"\nHMM Model Results:")
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
        if self.model is None:
            raise ValueError("Model not trained. Call train_from_files() first.")
        
        if self.classifier_type == 'hmm':
            # HMM works with raw texts
            predictions = self.model.predict(texts)
            probabilities = self.model.predict_proba(texts)[:, 1]
        else:
            # Extract and scale features
            features = self.extract_features(texts)
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Predict
            predictions = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else np.zeros_like(predictions)
        
        return predictions, probabilities
    
    def save_model(self, model_path: str):
        """Save the trained model and preprocessing components."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        package = ModelPackage('probabilistic')
        
        # Add the model
        package.add_model(self.model)
        
        # Add vectorizers and scaler (if not HMM)
        if self.classifier_type != 'hmm':
            package.add_word_vectorizer(self.word_vectorizer)
            package.add_char_vectorizer(self.char_vectorizer)
            if self.scaler is not None:
                package.add_scaler(self.scaler)
        
        # Add configuration as metadata
        package.metadata.update({
            'classifier_type': self.classifier_type,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'use_hyperparameter_tuning': self.use_hyperparameter_tuning,
            'model_architecture': 'ProbabilisticTextClassifier'
        })
        
        return ModelSerializer.save_model_package(package, model_path)
    
    def load_model(self, model_path: str):
        """Load a trained model and preprocessing components."""
        package = ModelSerializer.load_model_package(model_path)
        
        self.classifier_type = package.metadata['classifier_type']
        self.max_features = package.metadata['max_features']
        self.ngram_range = package.metadata['ngram_range']
        self.use_hyperparameter_tuning = package.metadata['use_hyperparameter_tuning']
        
        self.model = package.model
        
        if self.classifier_type != 'hmm':
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


class CategoricalNBWrapper:
    """Wrapper for CategoricalNB that discretizes continuous features."""
    
    def __init__(self, alpha: float = 1.0, n_bins: int = 10):
        self.alpha = alpha
        self.n_bins = n_bins
        self.model = None
        self.bin_edges = None
        
    def _discretize_features(self, X):
        """Discretize continuous features into categorical bins."""
        if self.bin_edges is None:
            # Fit bin edges during training
            self.bin_edges = []
            X_discretized = np.zeros_like(X, dtype=int)
            
            for feature_idx in range(X.shape[1]):
                feature_values = X[:, feature_idx]
                # Use quantile-based binning to ensure balanced bins
                try:
                    _, bin_edges = np.histogram(feature_values, bins=self.n_bins)
                    self.bin_edges.append(bin_edges)
                    # Discretize this feature
                    X_discretized[:, feature_idx] = np.digitize(feature_values, bin_edges[1:-1])
                except Exception:
                    # Fallback for constant features
                    self.bin_edges.append(np.array([feature_values.min(), feature_values.max()]))
                    X_discretized[:, feature_idx] = 0
            
            return X_discretized
        else:
            # Transform using existing bin edges
            X_discretized = np.zeros_like(X, dtype=int)
            
            for feature_idx in range(X.shape[1]):
                feature_values = X[:, feature_idx]
                bin_edges = self.bin_edges[feature_idx]
                if len(bin_edges) > 2:
                    X_discretized[:, feature_idx] = np.digitize(feature_values, bin_edges[1:-1])
                else:
                    X_discretized[:, feature_idx] = 0
            
            return X_discretized
    
    def fit(self, X, y):
        """Fit the CategoricalNB model with discretized features."""
        X_discretized = self._discretize_features(X)
        self.model = CategoricalNB(alpha=self.alpha)
        self.model.fit(X_discretized, y)
        return self
    
    def predict(self, X):
        """Predict class labels."""
        X_discretized = self._discretize_features(X)
        return self.model.predict(X_discretized)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X_discretized = self._discretize_features(X)
        return self.model.predict_proba(X_discretized)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'alpha': self.alpha,
            'n_bins': self.n_bins
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class GaussianMixtureClassifier:
    """Gaussian Mixture Model classifier wrapper for sklearn compatibility."""
    
    def __init__(self, n_components: int = 2, covariance_type: str = 'full'):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.models = {}
        self.classes_ = None
        
    def fit(self, X, y):
        """Fit Gaussian Mixture Models for each class."""
        self.classes_ = np.unique(y)
        
        for cls in self.classes_:
            class_mask = y == cls
            class_data = X[class_mask]
            
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=42
            )
            gmm.fit(class_data)
            self.models[cls] = gmm
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities using GMM likelihoods."""
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, len(self.classes_)))
        
        for i, cls in enumerate(self.classes_):
            log_probs = self.models[cls].score_samples(X)
            # Handle potential numerical issues
            log_probs = np.nan_to_num(log_probs, nan=-1e10, posinf=-1e10, neginf=-1e10)
            probabilities[:, i] = log_probs
        
        # Convert log-likelihoods to probabilities with numerical stability
        # Subtract max for numerical stability
        max_log_probs = np.max(probabilities, axis=1, keepdims=True)
        probabilities = probabilities - max_log_probs
        probabilities = np.exp(probabilities)
        
        # Normalize probabilities
        prob_sums = probabilities.sum(axis=1, keepdims=True)
        prob_sums = np.where(prob_sums == 0, 1e-10, prob_sums)  # Avoid division by zero
        probabilities = probabilities / prob_sums
        
        # Final cleanup for any remaining NaN/inf values
        probabilities = np.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0)
        
        return probabilities
    
    def predict(self, X):
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'n_components': self.n_components,
            'covariance_type': self.covariance_type
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def compare_probabilistic_classifiers(human_file: str, ai_file: str, classifiers: List[str] = None, 
                                    test_size: float = 0.2, validation_size: float = 0.15) -> Dict[str, Dict[str, Any]]:
    """Compare multiple probabilistic classifiers using the same data split."""
    if classifiers is None:
        classifiers = ['gaussian_nb', 'bernoulli_nb', 'gaussian_mixture', 'lda', 'qda', 
                      'bayesian_network', 'hmm']
    
    results = {}
    
    print("=== Comparing Probabilistic Classifiers ===")
    print(f"Human file: {human_file}")
    print(f"AI file: {ai_file}")
    print(f"Classifiers to test: {classifiers}")
    print("=" * 60)
    
    for classifier_type in classifiers:
        print(f"\nTraining {classifier_type}...")
        try:
            classifier = ProbabilisticTextClassifier(
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
    print("PROBABILISTIC CLASSIFIER COMPARISON SUMMARY")
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
