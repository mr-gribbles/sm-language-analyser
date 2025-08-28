"""
Interpretable machine learning classifiers for AI vs Human text detection.

This rewritten module provides a stable and robust implementation for training
interpretable models, focusing on numerical stability and clear feature analysis.
"""
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib

# Suppress warnings to keep output clean, as we are handling numerical issues.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class InterpretableTextClassifier:
    """
    A robust interpretable ML classifier for text analysis, rewritten for stability.
    """
    def __init__(self, classifier_type: str = 'logistic_regression', max_features: int = 4000):
        self.classifier_type = classifier_type
        self.max_features = max_features
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.feature_names_ = None

    def _load_texts_from_file(self, file_path: str, text_type: str) -> List[str]:
        """Loads texts from a JSONL file."""
        texts = []
        print(f"Loading {text_type} texts from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if text_type == 'human':
                        original_content = record.get('original_content') or {}
                        text = original_content.get('cleaned_text') or original_content.get('cleaned_selftext', '')
                    else:  # AI
                        llm_transformation = record.get('llm_transformation') or {}
                        text = llm_transformation.get('rewritten_text', '')
                    
                    if text and len(text.strip()) > 20:
                        texts.append(text.strip())
                except (json.JSONDecodeError, KeyError):
                    continue
        return texts

    def _compute_linguistic_features(self, text: str) -> Dict[str, float]:
        """Computes a set of linguistic features for a single text with robust error handling."""
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        sentences = len(re.split(r'[.!?]+', text))

        # Safe division helper
        def safe_div(numerator, denominator):
            return numerator / denominator if denominator > 0 else 0.0

        features = {
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': safe_div(sum(len(w) for w in words), word_count),
            'sentence_count': sentences,
            'avg_sentence_length': safe_div(word_count, sentences),
            'lexical_diversity': safe_div(len(set(words)), word_count),
            'punctuation_ratio': safe_div(sum(1 for char in text if char in '.,!?;:'), char_count),
            'uppercase_ratio': safe_div(sum(1 for char in text if char.isupper()), char_count),
        }
        return features

    def _extract_features(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """Extracts combined n-gram and linguistic features."""
        # 1. N-gram features (using CountVectorizer for stability)
        if fit:
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                stop_words='english',
                binary=True # Use binary features for stability
            )
            ngram_features = self.vectorizer.fit_transform(texts).toarray()
        else:
            ngram_features = self.vectorizer.transform(texts).toarray()

        # 2. Linguistic features
        linguistic_features = np.array([list(self._compute_linguistic_features(text).values()) for text in texts])
        
        # 3. Combine features
        combined_features = np.hstack([ngram_features, linguistic_features])
        
        # 4. Store feature names
        if fit:
            ngram_names = self.vectorizer.get_feature_names_out()
            linguistic_names = list(self._compute_linguistic_features("sample text").keys())
            self.feature_names_ = [f"ngram_{n}" for n in ngram_names] + [f"ling_{n}" for n in linguistic_names]
            
        return combined_features

    def _get_model(self):
        """Returns the classifier instance and its hyperparameter grid."""
        models = {
            'logistic_regression': (
                LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'),
                {'C': [0.1, 1, 10], 'penalty': ['l2']}
            ),
            'decision_tree': (
                DecisionTreeClassifier(random_state=42),
                {'max_depth': [10, 20, None], 'min_samples_split': [10, 50]}
            ),
            'random_forest': (
                RandomForestClassifier(random_state=42, n_jobs=-1),
                {'n_estimators': [100, 200], 'max_depth': [10, 20]}
            ),
            'naive_bayes': (
                MultinomialNB(),
                {'alpha': [0.5, 1.0, 2.0]}
            )
        }
        if self.classifier_type not in models:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
        return models[self.classifier_type]

    def train(self, human_file: str, ai_file: str, test_size: float = 0.2):
        """Main training pipeline."""
        # 1. Load data
        human_texts = self._load_texts_from_file(human_file, 'human')
        ai_texts = self._load_texts_from_file(ai_file, 'ai')
        texts = human_texts + ai_texts
        labels = np.array([0] * len(human_texts) + [1] * len(ai_texts))
        print(f"Loaded {len(texts)} total texts ({len(human_texts)} human, {len(ai_texts)} AI).")

        # 2. Extract and clean features
        print("Extracting and cleaning features...")
        features = self._extract_features(texts, fit=True)
        features = np.nan_to_num(features, nan=0.0, posinf=1e5, neginf=-1e5) # Clean features

        # 3. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # 4. Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Clean scaled features to be absolutely sure
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

        # 5. Train model with hyperparameter tuning
        print(f"Training {self.classifier_type} with hyperparameter tuning...")
        classifier, param_grid = self._get_model()
        
        # Naive Bayes cannot handle negative values, so use unscaled data
        if self.classifier_type == 'naive_bayes':
            # Shift data to be non-negative
            min_val = X_train.min()
            X_train_nb = X_train - min_val
            X_test_nb = X_test - min_val
            grid_search = GridSearchCV(classifier, param_grid, cv=3, scoring='accuracy', n_jobs=1)
            grid_search.fit(X_train_nb, y_train)
        else:
            grid_search = GridSearchCV(classifier, param_grid, cv=3, scoring='accuracy', n_jobs=1)
            grid_search.fit(X_train_scaled, y_train)

        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")

        # 6. Evaluate model
        if self.classifier_type == 'naive_bayes':
            y_pred = self.model.predict(X_test_nb)
        else:
            y_pred = self.model.predict(X_test_scaled)
            
        accuracy = accuracy_score(y_test, y_pred)
        print("\n--- Model Evaluation ---")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
        return accuracy

    def get_influential_features(self, n_features: int = 20) -> pd.DataFrame:
        """Returns the most influential features for the trained model."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        if hasattr(self.model, 'coef_'): # Logistic Regression
            importances = self.model.coef_[0]
        elif hasattr(self.model, 'feature_importances_'): # Tree-based models
            importances = self.model.feature_importances_
        else: # Naive Bayes
            # Use log probabilities as a proxy for importance
            importances = self.model.feature_log_prob_[1] - self.model.feature_log_prob_[0]

        feature_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importances,
            'abs_importance': np.abs(importances)
        }).sort_values(by='abs_importance', ascending=False)
        
        # Separate feature type for better readability
        feature_df['type'] = feature_df['feature'].apply(lambda x: 'linguistic' if x.startswith('ling_') else 'n-gram')
        feature_df['feature'] = feature_df['feature'].str.replace(r'^(ngram_|ling_)', '', regex=True)

        return feature_df.head(n_features)

    def save(self, path: str):
        """Saves the trained model and its components."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'scaler': self.scaler,
                'feature_names': self.feature_names_,
                'classifier_type': self.classifier_type
            }, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Loads a trained model from a file."""
        with open(path, 'rb') as f:
            components = pickle.load(f)
        
        instance = cls(classifier_type=components['classifier_type'])
        instance.model = components['model']
        instance.vectorizer = components['vectorizer']
        instance.scaler = components['scaler']
        instance.feature_names_ = components['feature_names']
        print(f"Model loaded from {path}")
        return instance

def compare_and_analyze(human_file: str, ai_file: str):
    """
    Compares different interpretable models and prints the most influential
    features from the best-performing one.
    """
    models_to_compare = ['logistic_regression', 'decision_tree', 'random_forest', 'naive_bayes']
    results = {}

    print("=== Comparing Interpretable Models ===")
    for model_type in models_to_compare:
        print(f"\n--- Training {model_type} ---")
        try:
            classifier = InterpretableTextClassifier(classifier_type=model_type)
            classifier.train(human_file, ai_file)
            
            accuracy = classifier.train(human_file, ai_file)
            
            # Store classifier and its accuracy
            results[model_type] = {
                'classifier': classifier,
                'accuracy': accuracy
            }
        except Exception as e:
            print(f"Error training {model_type}: {e}")

    if not results:
        print("No models were trained successfully.")
        return

    # Find the best model based on accuracy
    best_model_type = max(results, key=lambda m: results[m]['accuracy'])
    best_classifier = results[best_model_type]['classifier']
    
    print(f"\n=== Analysis of Best Model: {best_model_type.replace('_', ' ').title()} ===")
    
    # Get and display influential features
    influential_features = best_classifier.get_influential_features()
    print("\nTop 20 Most Influential Language Features:")
    print(influential_features.to_string(index=False))

    # Save the best model for future analysis
    model_save_path = Path('models') / 'best_model.pkl'
    best_classifier.save(str(model_save_path))
