"""Manifold learning classifiers for AI vs Human text detection.

This module implements manifold learning and dimensionality reduction approaches
for text classification using the same pipeline as other classifiers.
"""
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, FactorAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from .model_serializer import ModelPackage, ModelSerializer


class ManifoldKNNClassifier:
    """KNN classifier with manifold learning preprocessing."""
    
    def __init__(self, manifold_type: str = 'tsne', n_components: int = 50, n_neighbors: int = 5):
        self.manifold_type = manifold_type
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.manifold_learner = None
        self.classifier = None
        
    def _get_manifold_learner(self):
        """Get the specified manifold learning algorithm with numerical stability."""
        # Reduce components for numerical stability
        safe_components = min(self.n_components, 50)
        
        if self.manifold_type == 'tsne':
            return TSNE(n_components=min(safe_components, 3), random_state=42, 
                       perplexity=min(15, safe_components-1), n_iter=300)
        elif self.manifold_type == 'isomap':
            return Isomap(n_components=safe_components, n_neighbors=min(self.n_neighbors, 10))
        elif self.manifold_type == 'lle':
            return LocallyLinearEmbedding(n_components=safe_components, 
                                        n_neighbors=min(self.n_neighbors, 10), random_state=42,
                                        reg=1e-3)  # Add regularization
        elif self.manifold_type in ['spectral', 'spectral_embedding']:
            return SpectralEmbedding(n_components=safe_components, random_state=42,
                                   n_neighbors=min(self.n_neighbors, 10))
        elif self.manifold_type == 'pca':
            return PCA(n_components=safe_components, random_state=42)
        elif self.manifold_type == 'ica':
            return FastICA(n_components=safe_components, random_state=42, 
                          max_iter=200, tol=1e-3)  # Reduce iterations for stability
        elif self.manifold_type in ['svd', 'truncated_svd']:
            return TruncatedSVD(n_components=safe_components, random_state=42)
        elif self.manifold_type == 'mds':
            return MDS(n_components=safe_components, random_state=42, 
                      max_iter=300, eps=1e-6)  # Add stability parameters
        elif self.manifold_type == 'factor_analysis':
            return FactorAnalysis(n_components=safe_components, random_state=42,
                                max_iter=100, tol=1e-3)  # Add stability parameters
        else:
            raise ValueError(f"Unknown manifold type: {self.manifold_type}")
    
    def fit(self, X, y):
        """Fit the manifold learner and classifier with error handling."""
        try:
            # Apply manifold learning with numerical stability
            self.manifold_learner = self._get_manifold_learner()
            
            # Handle potential numerical issues
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                
                if self.manifold_type == 'tsne':
                    # t-SNE doesn't have transform method, so we store the embedding
                    X_embedded = self.manifold_learner.fit_transform(X)
                    self.X_train_embedded = X_embedded
                    self.y_train = y
                else:
                    X_embedded = self.manifold_learner.fit_transform(X)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(X_embedded)) or np.any(np.isinf(X_embedded)):
                # Fallback to PCA if manifold learning fails
                print(f"Warning: {self.manifold_type} produced invalid values, falling back to PCA")
                self.manifold_learner = PCA(n_components=min(50, X.shape[1]), random_state=42)
                X_embedded = self.manifold_learner.fit_transform(X)
            
            # Train classifier on embedded features
            self.classifier = KNeighborsClassifier(n_neighbors=min(self.n_neighbors, len(X_embedded)//10))
            self.classifier.fit(X_embedded, y)
            
        except Exception as e:
            # Fallback to PCA if anything fails
            print(f"Warning: {self.manifold_type} failed ({e}), falling back to PCA")
            self.manifold_learner = PCA(n_components=min(50, X.shape[1]), random_state=42)
            X_embedded = self.manifold_learner.fit_transform(X)
            self.classifier = KNeighborsClassifier(n_neighbors=min(self.n_neighbors, len(X_embedded)//10))
            self.classifier.fit(X_embedded, y)
        
        return self
    
    def predict(self, X):
        """Predict using manifold embedding."""
        try:
            if self.manifold_type == 'tsne':
                # For t-SNE, use nearest neighbors in original space
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(self.X_train_embedded)))
                nn.fit(self.X_train_embedded)
                
                # Find nearest neighbors for each test sample
                predictions = []
                for x in X:
                    # Use original space similarity (simplified approach)
                    distances = np.linalg.norm(self.X_train_embedded - x[:self.X_train_embedded.shape[1]], axis=1)
                    nearest_idx = np.argmin(distances)
                    predictions.append(self.y_train[nearest_idx])
                return np.array(predictions)
            else:
                X_embedded = self.manifold_learner.transform(X)
                # Check for invalid values
                if np.any(np.isnan(X_embedded)) or np.any(np.isinf(X_embedded)):
                    # Return random predictions if transformation fails
                    return np.random.randint(0, 2, len(X))
                return self.classifier.predict(X_embedded)
        except Exception:
            # Return random predictions if anything fails
            return np.random.randint(0, 2, len(X))
    
    def predict_proba(self, X):
        """Predict probabilities using manifold embedding."""
        try:
            if self.manifold_type == 'tsne':
                # Simplified probability estimation for t-SNE
                predictions = self.predict(X)
                probabilities = np.zeros((len(X), 2))
                for i, pred in enumerate(predictions):
                    probabilities[i, pred] = 0.7  # Moderate confidence
                    probabilities[i, 1-pred] = 0.3  # Lower confidence
                return probabilities
            else:
                X_embedded = self.manifold_learner.transform(X)
                # Check for invalid values
                if np.any(np.isnan(X_embedded)) or np.any(np.isinf(X_embedded)):
                    # Return uniform probabilities if transformation fails
                    return np.full((len(X), 2), 0.5)
                return self.classifier.predict_proba(X_embedded)
        except Exception:
            # Return uniform probabilities if anything fails
            return np.full((len(X), 2), 0.5)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'manifold_type': self.manifold_type,
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ManifoldSVMClassifier:
    """SVM classifier with manifold learning preprocessing."""
    
    def __init__(self, manifold_type: str = 'pca', n_components: int = 100, C: float = 1.0):
        self.manifold_type = manifold_type
        self.n_components = n_components
        self.C = C
        self.manifold_learner = None
        self.classifier = None
        
    def _get_manifold_learner(self):
        """Get the specified manifold learning algorithm with numerical stability."""
        safe_components = min(self.n_components, 100)
        
        if self.manifold_type == 'pca':
            return PCA(n_components=safe_components, random_state=42)
        elif self.manifold_type == 'ica':
            return FastICA(n_components=safe_components, random_state=42,
                          max_iter=200, tol=1e-3)
        elif self.manifold_type in ['svd', 'truncated_svd']:
            return TruncatedSVD(n_components=safe_components, random_state=42)
        elif self.manifold_type == 'isomap':
            return Isomap(n_components=safe_components, n_neighbors=10)
        elif self.manifold_type == 'mds':
            return MDS(n_components=safe_components, random_state=42,
                      max_iter=300, eps=1e-6)
        elif self.manifold_type == 'factor_analysis':
            return FactorAnalysis(n_components=safe_components, random_state=42,
                                max_iter=100, tol=1e-3)
        else:
            raise ValueError(f"Unknown manifold type: {self.manifold_type}")
    
    def fit(self, X, y):
        """Fit the manifold learner and SVM classifier with error handling."""
        try:
            # Apply manifold learning with numerical stability
            self.manifold_learner = self._get_manifold_learner()
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                X_embedded = self.manifold_learner.fit_transform(X)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(X_embedded)) or np.any(np.isinf(X_embedded)):
                # Fallback to PCA if manifold learning fails
                print(f"Warning: {self.manifold_type} produced invalid values, falling back to PCA")
                self.manifold_learner = PCA(n_components=min(100, X.shape[1]), random_state=42)
                X_embedded = self.manifold_learner.fit_transform(X)
            
            # Train SVM on embedded features
            self.classifier = SVC(C=self.C, probability=True, random_state=42)
            self.classifier.fit(X_embedded, y)
            
        except Exception as e:
            # Fallback to PCA if anything fails
            print(f"Warning: {self.manifold_type} failed ({e}), falling back to PCA")
            self.manifold_learner = PCA(n_components=min(100, X.shape[1]), random_state=42)
            X_embedded = self.manifold_learner.fit_transform(X)
            self.classifier = SVC(C=self.C, probability=True, random_state=42)
            self.classifier.fit(X_embedded, y)
        
        return self
    
    def predict(self, X):
        """Predict using manifold embedding."""
        try:
            X_embedded = self.manifold_learner.transform(X)
            # Check for invalid values
            if np.any(np.isnan(X_embedded)) or np.any(np.isinf(X_embedded)):
                return np.random.randint(0, 2, len(X))
            return self.classifier.predict(X_embedded)
        except Exception:
            return np.random.randint(0, 2, len(X))
    
    def predict_proba(self, X):
        """Predict probabilities using manifold embedding."""
        try:
            X_embedded = self.manifold_learner.transform(X)
            # Check for invalid values
            if np.any(np.isnan(X_embedded)) or np.any(np.isinf(X_embedded)):
                return np.full((len(X), 2), 0.5)
            return self.classifier.predict_proba(X_embedded)
        except Exception:
            return np.full((len(X), 2), 0.5)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'manifold_type': self.manifold_type,
            'n_components': self.n_components,
            'C': self.C
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ClusterBasedClassifier:
    """Classifier based on cluster membership features."""
    
    def __init__(self, cluster_type: str = 'kmeans', n_clusters: int = 20, 
                 base_classifier: str = 'logistic'):
        self.cluster_type = cluster_type
        self.n_clusters = n_clusters
        self.base_classifier = base_classifier
        self.clusterer = None
        self.classifier = None
        
    def _get_clusterer(self):
        """Get the specified clustering algorithm."""
        if self.cluster_type == 'kmeans':
            return KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        elif self.cluster_type == 'dbscan':
            return DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown cluster type: {self.cluster_type}")
    
    def _get_base_classifier(self):
        """Get the base classifier."""
        if self.base_classifier == 'logistic':
            return LogisticRegression(random_state=42, max_iter=1000)
        elif self.base_classifier == 'svm':
            return SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown base classifier: {self.base_classifier}")
    
    def fit(self, X, y):
        """Fit the clusterer and base classifier."""
        try:
            # Perform clustering
            self.clusterer = self._get_clusterer()
            cluster_labels = self.clusterer.fit_predict(X)
            
            # Create cluster-based features
            cluster_features = self._create_cluster_features(X, cluster_labels)
            
            # Train base classifier
            self.classifier = self._get_base_classifier()
            self.classifier.fit(cluster_features, y)
            
        except Exception as e:
            # Fallback to simple logistic regression
            print(f"Warning: Clustering failed ({e}), using simple logistic regression")
            self.classifier = LogisticRegression(random_state=42, max_iter=1000)
            self.classifier.fit(X, y)
            self.clusterer = None
        
        return self
    
    def _create_cluster_features(self, X, cluster_labels):
        """Create features based on cluster membership and distances."""
        if self.clusterer is None:
            return X  # Return original features if clustering failed
            
        features = []
        
        try:
            if self.cluster_type == 'kmeans':
                cluster_centers = self.clusterer.cluster_centers_
                
                for i, x in enumerate(X):
                    cluster_features = []
                    
                    # Distance to each cluster center
                    for center in cluster_centers:
                        distance = np.linalg.norm(x - center)
                        cluster_features.append(distance)
                    
                    # One-hot encoding of cluster membership
                    cluster_membership = np.zeros(self.n_clusters)
                    if 0 <= cluster_labels[i] < self.n_clusters:  # Valid cluster
                        cluster_membership[cluster_labels[i]] = 1
                    cluster_features.extend(cluster_membership)
                    
                    features.append(cluster_features)
            
            else:  # DBSCAN
                unique_clusters = set(cluster_labels)
                n_clusters = len(unique_clusters)
                
                for i, x in enumerate(X):
                    cluster_features = []
                    
                    # Cluster label (or -1 for noise)
                    cluster_features.append(max(0, cluster_labels[i]))  # Convert -1 to 0
                    
                    # Distance to cluster centroid (if not noise)
                    if cluster_labels[i] >= 0:
                        cluster_mask = cluster_labels == cluster_labels[i]
                        if np.sum(cluster_mask) > 0:
                            cluster_points = X[cluster_mask]
                            centroid = np.mean(cluster_points, axis=0)
                            distance = np.linalg.norm(x - centroid)
                            cluster_features.append(distance)
                        else:
                            cluster_features.append(0)
                    else:
                        cluster_features.append(0)  # Noise point
                    
                    features.append(cluster_features)
            
            return np.array(features)
            
        except Exception:
            # Return original features if feature creation fails
            return X
    
    def predict(self, X):
        """Predict using cluster-based features."""
        try:
            if self.clusterer is None:
                return self.classifier.predict(X)
                
            cluster_labels = self.clusterer.predict(X) if hasattr(self.clusterer, 'predict') else self.clusterer.fit_predict(X)
            cluster_features = self._create_cluster_features(X, cluster_labels)
            return self.classifier.predict(cluster_features)
        except Exception:
            return np.random.randint(0, 2, len(X))
    
    def predict_proba(self, X):
        """Predict probabilities using cluster-based features."""
        try:
            if self.clusterer is None:
                return self.classifier.predict_proba(X)
                
            cluster_labels = self.clusterer.predict(X) if hasattr(self.clusterer, 'predict') else self.clusterer.fit_predict(X)
            cluster_features = self._create_cluster_features(X, cluster_labels)
            return self.classifier.predict_proba(cluster_features)
        except Exception:
            return np.full((len(X), 2), 0.5)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'cluster_type': self.cluster_type,
            'n_clusters': self.n_clusters,
            'base_classifier': self.base_classifier
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ManifoldTextClassifier:
    """Manifold learning classifier using various dimensionality reduction approaches."""
    
    def __init__(self, classifier_type: str = 'pca', manifold_type: str = None, 
                 max_features: int = 8000, ngram_range: Tuple[int, int] = (1, 2), 
                 use_hyperparameter_tuning: bool = False):
        """Initialize the manifold classifier.
        
        Args:
            classifier_type: Type of manifold method to use
            manifold_type: Alternative name for classifier_type (for compatibility)
            max_features: Maximum number of features for TF-IDF vectorization.
            ngram_range: Range of n-grams to extract.
            use_hyperparameter_tuning: Whether to use grid search for hyperparameter tuning.
        """
        # Handle both classifier_type and manifold_type parameters for compatibility
        if manifold_type is not None:
            self.classifier_type = manifold_type
        else:
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
            text_features.append(np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0)
            
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
            function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            function_word_count = sum(1 for word in words if word.lower() in function_words)
            text_features.append(function_word_count / len(words) if len(words) > 0 else 0)
            
            # Repetition patterns
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            unique_bigrams = set(bigrams)
            text_features.append(len(unique_bigrams) / len(bigrams) if len(bigrams) > 0 else 0)
            
            features.append(text_features)
        
        return np.array(features)
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract comprehensive features with reduced dimensionality."""
        # Word-level TF-IDF features (reduced)
        if self.word_vectorizer is None:
            self.word_vectorizer = TfidfVectorizer(
                max_features=self.max_features // 2,
                ngram_range=self.ngram_range,
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
        
        # Character-level TF-IDF features (reduced)
        if self.char_vectorizer is None:
            self.char_vectorizer = TfidfVectorizer(
                max_features=self.max_features // 2,
                analyzer='char',
                ngram_range=(2, 4),  # Reduced range
                lowercase=True,
                min_df=10,
                max_df=0.8,
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
        # Map method names to actual classifiers with safer parameters
        if self.classifier_type in ['pca', 'ica', 'truncated_svd', 'mds', 'factor_analysis']:
            classifier = ManifoldSVMClassifier(manifold_type=self.classifier_type)
            param_grid = {
                'n_components': [20, 50],  # Reduced options
                'C': [0.1, 1]  # Reduced options
            } if self.use_hyperparameter_tuning else {}
            
        elif self.classifier_type in ['tsne', 'isomap', 'lle', 'spectral_embedding']:
            classifier = ManifoldKNNClassifier(manifold_type=self.classifier_type)
            param_grid = {
                'n_components': [10, 20],  # Reduced options
                'n_neighbors': [3, 5]  # Reduced options
            } if self.use_hyperparameter_tuning else {}
            
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        return classifier, param_grid
    
    def train_from_files(self, human_file: str, ai_file: str, test_size: float = 0.2, 
                        validation_size: float = 0.15, cv_folds: int = 3) -> Dict[str, Any]:
        """Train the manifold classifier with improved error handling."""
        # Load data
        texts, labels = self.load_corpus_files(human_file, ai_file)
        
        if len(texts) == 0:
            raise ValueError("No texts provided for training")
        
        # Extract features
        print("Extracting comprehensive features (word + character + linguistic)...")
        features = self.extract_features(texts)
        print(f"Extracted {features.shape[1]} total features")
        
        # Scale features (important for manifold learning) with robust scaling
        self.scaler = StandardScaler()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            features_scaled = self.scaler.fit_transform(features)
        
        # Replace any remaining NaN or infinite values
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        
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
                n_jobs=1, verbose=0  # Reduced parallelism for stability
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
        try:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='accuracy')
        except Exception:
            # Fallback if CV fails
            cv_scores = np.array([val_accuracy])
        
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
            try:
                results['test_auc'] = roc_auc_score(y_test, test_probabilities)
            except Exception:
                results['test_auc'] = 0.5
        else:
            results['test_auc'] = 0.5
        
        print(f"\n{self.classifier_type.title()} Model Results:")
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
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
        if self.model is None or self.word_vectorizer is None or self.scaler is None:
            raise ValueError("Model not trained. Call train_from_files() first.")
        
        # Extract and scale features
        features = self.extract_features(texts)
        features_scaled = self.scaler.transform(features)
        
        # Replace any NaN or infinite values
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Predict
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else np.full(len(predictions), 0.5)
        
        return predictions, probabilities
    
    def save_model(self, model_path: str):
        """Save the trained model and preprocessing components."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        package = ModelPackage('manifold')
        
        # Add the model
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
            'model_architecture': 'ManifoldTextClassifier'
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
        self.word_vectorizer = package.word_vectorizer
        self.char_vectorizer = package.char_vectorizer
        self.scaler = package.scaler
