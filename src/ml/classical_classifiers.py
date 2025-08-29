"""Classical machine learning classifiers for AI vs Human text detection.

This module implements various classical ML algorithms using the same
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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from .model_serializer import ModelPackage, ModelSerializer


class ClassicalTextClassifier:
    """Classical ML classifier using the same pipeline as the neural network."""
    
    def __init__(self, classifier_type: str = 'random_forest', max_features: int = 15000, 
                 ngram_range: Tuple[int, int] = (1, 3), use_hyperparameter_tuning: bool = True):
        """Initialize the classical classifier.
        
        Args:
            classifier_type: Type of classifier ('random_forest', 'svm', 'logistic_regression', 
                           'gradient_boosting', 'naive_bayes', 'knn', 'decision_tree', 'adaboost')
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
        """Extract linguistic features from texts (same as neural network version)."""
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
    
    def _get_classifier_and_params(self):
        """Get classifier and hyperparameter grid based on classifier type."""
        if self.classifier_type == 'random_forest':
            classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        elif self.classifier_type == 'svm':
            classifier = SVC(random_state=42, probability=True)
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
        elif self.classifier_type == 'logistic_regression':
            classifier = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'l1_ratio': [0.1, 0.5, 0.9]  # Only used with elasticnet
            }
        elif self.classifier_type == 'gradient_boosting':
            # Add better regularization to prevent overfitting
            classifier = GradientBoostingClassifier(
                random_state=42,
                subsample=0.8,  # Use 80% of samples for each tree
                max_features='sqrt',  # Use sqrt of features for each split
                validation_fraction=0.1,  # Use 10% for early stopping
                n_iter_no_change=10,  # Stop if no improvement for 10 iterations
                tol=1e-4  # Tolerance for early stopping
            )
            param_grid = {
                'n_estimators': [50, 100, 150],  # Reduced to prevent overfitting
                'learning_rate': [0.05, 0.1, 0.15],  # More conservative learning rates
                'max_depth': [3, 4, 5],  # Shallower trees to prevent overfitting
                'min_samples_split': [10, 20, 50],  # Higher minimum samples
                'min_samples_leaf': [5, 10, 20]  # Higher minimum leaf samples
            }
        elif self.classifier_type == 'naive_bayes':
            classifier = ComplementNB()  # Better for imbalanced text data
            param_grid = {
                'alpha': [0.1, 0.5, 1.0, 2.0],
                'norm': [True, False]
            }
        elif self.classifier_type == 'knn':
            classifier = KNeighborsClassifier(n_jobs=-1)
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'cosine']
            }
        elif self.classifier_type == 'decision_tree':
            classifier = DecisionTreeClassifier(random_state=42)
            param_grid = {
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        elif self.classifier_type == 'adaboost':
            classifier = AdaBoostClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0, 2.0],
                'algorithm': ['SAMME', 'SAMME.R']
            }
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        return classifier, param_grid
    
    def train_from_files(self, human_file: str, ai_file: str, test_size: float = 0.2, 
                        validation_size: float = 0.15, cv_folds: int = 5) -> Dict[str, Any]:
        """Train the classifier using the same pipeline as neural network."""
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
        
        # Get classifier and parameters
        classifier, param_grid = self._get_classifier_and_params()
        
        # Train with or without hyperparameter tuning
        if self.use_hyperparameter_tuning:
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
        
        package = ModelPackage('classical')
        
        # Add the sklearn model
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
            'model_architecture': 'ClassicalTextClassifier'
        })
        
        return ModelSerializer.save_model_package(package, model_path)
    
    def load_model(self, model_path: str):
        """Load a trained model and preprocessing components."""
        try:
            # Try consolidated format first
            package = ModelSerializer.load_model_package(model_path)
            
            self.classifier_type = package.metadata['classifier_type']
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
            with open(f"{model_path}_{self.classifier_type}_config.json", 'r') as f:
                config = json.load(f)
            
            self.classifier_type = config['classifier_type']
            self.max_features = config['max_features']
            self.ngram_range = config['ngram_range']
            self.use_hyperparameter_tuning = config['use_hyperparameter_tuning']
            
            # Load model
            self.model = joblib.load(f"{model_path}_{self.classifier_type}_model.pkl")
            
            # Load preprocessing components
            with open(f"{model_path}_{self.classifier_type}_word_vectorizer.pkl", 'rb') as f:
                self.word_vectorizer = pickle.load(f)
                
            with open(f"{model_path}_{self.classifier_type}_char_vectorizer.pkl", 'rb') as f:
                self.char_vectorizer = pickle.load(f)
            
            with open(f"{model_path}_{self.classifier_type}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
    
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


def compare_classifiers(human_file: str, ai_file: str, classifiers: List[str] = None, 
                       test_size: float = 0.2, validation_size: float = 0.15) -> Dict[str, Dict[str, Any]]:
    """Compare multiple classical classifiers using the same data split."""
    if classifiers is None:
        classifiers = ['random_forest', 'svm', 'logistic_regression', 'gradient_boosting', 
                      'naive_bayes', 'knn', 'decision_tree', 'adaboost']
    
    results = {}
    
    print("=== Comparing Classical ML Classifiers ===")
    print(f"Human file: {human_file}")
    print(f"AI file: {ai_file}")
    print(f"Classifiers to test: {classifiers}")
    print("=" * 60)
    
    for classifier_type in classifiers:
        print(f"\nTraining {classifier_type}...")
        try:
            classifier = ClassicalTextClassifier(
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
    print("CLASSIFIER COMPARISON SUMMARY")
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
