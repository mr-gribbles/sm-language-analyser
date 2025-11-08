"""Enhanced Feature Extraction for AI vs Human Text Classification.

This module provides advanced feature extraction techniques including:
- Advanced TF-IDF with better preprocessing
- Word embeddings (Word2Vec, FastText)
- Sentence embeddings (SentenceTransformers)
- Advanced linguistic features
- Feature selection techniques
- Ensemble feature combinations
"""

import re
import string
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import (SelectKBest, VarianceThreshold, chi2,
                                       f_classif, mutual_info_classif)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class SparseToDenseTransformer(BaseEstimator, TransformerMixin):
    """Convert sparse matrices to dense arrays (module-level class for pickling)."""

    def fit(self, X, y=None):
        """Fit the transformer."""
        return self

    def transform(self, X):
        """Transform a sparse matrix to a dense matrix."""
        if hasattr(X, "toarray"):
            return X.toarray()
        return X


class AdvancedLinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract comprehensive linguistic features from text."""

    def __init__(
        self,
        include_readability=True,
        include_pos_tags=True,
        include_sentiment=True,
        include_stylometric=True,
    ):
        """Initialize the feature extractor."""
        self.include_readability = include_readability
        self.include_pos_tags = include_pos_tags
        self.include_sentiment = include_sentiment
        self.include_stylometric = include_stylometric

        # Pre-compile regex patterns for efficiency
        self.patterns = {
            "urls": re.compile(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            ),
            "emails": re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            ),
            "mentions": re.compile(r"@\w+"),
            "hashtags": re.compile(r"#\w+"),
            "numbers": re.compile(r"\d+"),
            "repeated_chars": re.compile(r"(.)\1{2,}"),
            "repeated_words": re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE),
            "all_caps": re.compile(r"\b[A-Z]{2,}\b"),
            "punctuation_clusters": re.compile(
                r"[{}]{{2,}}".format(re.escape(string.punctuation))
            ),
        }

    def fit(self, X, y=None):
        """Fit the transformer."""
        return self

    def transform(self, X):
        """Transform texts into linguistic features."""
        features = []

        for text in X:
            text_features = []

            # Basic text statistics
            text_features.extend(self._extract_basic_stats(text))

            # Character-level features
            text_features.extend(self._extract_character_features(text))

            # Word-level features
            text_features.extend(self._extract_word_features(text))

            # Sentence-level features
            text_features.extend(self._extract_sentence_features(text))

            # Social media specific features
            text_features.extend(self._extract_social_media_features(text))

            if self.include_readability:
                text_features.extend(self._extract_readability_features(text))

            if self.include_stylometric:
                text_features.extend(self._extract_stylometric_features(text))

            # Ensure all features are numeric and finite
            text_features = [
                float(f) if np.isfinite(float(f)) else 0.0 for f in text_features
            ]
            features.append(text_features)

        return np.array(features)

    def _extract_basic_stats(self, text: str) -> List[float]:
        """Extract basic text statistics."""
        if not text:
            return [0] * 6

        text_len = len(text)
        char_count = len(text.replace(" ", ""))
        word_count = len(text.split())

        return [
            text_len,
            char_count,
            word_count,
            char_count / max(text_len, 1),  # Character density
            word_count / max(text_len, 1),  # Word density
            text_len / max(word_count, 1),  # Average word length (chars)
        ]

    def _extract_character_features(self, text: str) -> List[float]:
        """Extract character-level features."""
        if not text:
            return [0] * 12

        text_len = len(text)

        # Character type ratios
        upper_ratio = sum(1 for c in text if c.isupper()) / max(text_len, 1)
        lower_ratio = sum(1 for c in text if c.islower()) / max(text_len, 1)
        digit_ratio = sum(1 for c in text if c.isdigit()) / max(text_len, 1)
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(text_len, 1)
        space_ratio = sum(1 for c in text if c.isspace()) / max(text_len, 1)
        punct_ratio = sum(1 for c in text if c in string.punctuation) / max(text_len, 1)

        # Specific punctuation ratios
        question_ratio = text.count("?") / max(text_len, 1)
        exclaim_ratio = text.count("!") / max(text_len, 1)
        period_ratio = text.count(".") / max(text_len, 1)
        comma_ratio = text.count(",") / max(text_len, 1)
        quote_ratio = (text.count('"') + text.count("'")) / max(text_len, 1)

        # Character diversity
        unique_chars = len(set(text.lower()))
        char_diversity = unique_chars / max(text_len, 1)

        return [
            upper_ratio,
            lower_ratio,
            digit_ratio,
            alpha_ratio,
            space_ratio,
            punct_ratio,
            question_ratio,
            exclaim_ratio,
            period_ratio,
            comma_ratio,
            quote_ratio,
            char_diversity,
        ]

    def _extract_word_features(self, text: str) -> List[float]:
        """Extract word-level features."""
        words = text.split()
        if not words:
            return [0] * 10

        word_count = len(words)

        # Word length statistics
        word_lengths = [len(word.strip(string.punctuation)) for word in words]
        avg_word_len = np.mean(word_lengths) if word_lengths else 0
        std_word_len = np.std(word_lengths) if word_lengths else 0
        max_word_len = max(word_lengths) if word_lengths else 0
        min_word_len = min(word_lengths) if word_lengths else 0

        # Vocabulary complexity
        unique_words = set(word.lower().strip(string.punctuation) for word in words)
        lexical_diversity = len(unique_words) / max(word_count, 1)

        # Long words ratio (>6 characters)
        long_words = sum(1 for length in word_lengths if length > 6)
        long_word_ratio = long_words / max(word_count, 1)

        # Short words ratio (<=3 characters)
        short_words = sum(1 for length in word_lengths if length <= 3)
        short_word_ratio = short_words / max(word_count, 1)

        # Function words (common words that carry little semantic content)
        function_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
        }
        function_word_count = sum(
            1
            for word in words
            if word.lower().strip(string.punctuation) in function_words
        )
        function_word_ratio = function_word_count / max(word_count, 1)

        # Capitalized words ratio
        capitalized_words = sum(1 for word in words if word and word[0].isupper())
        capitalized_ratio = capitalized_words / max(word_count, 1)

        return [
            avg_word_len,
            std_word_len,
            max_word_len,
            min_word_len,
            lexical_diversity,
            long_word_ratio,
            short_word_ratio,
            function_word_ratio,
            capitalized_ratio,
            unique_words.__len__(),
        ]

    def _extract_sentence_features(self, text: str) -> List[float]:
        """Extract sentence-level features."""
        # Split by sentence-ending punctuation
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sentences:
            return [0] * 8

        sentence_count = len(sentences)
        words = text.split()
        word_count = len(words)

        # Sentence length statistics
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_len = np.mean(sentence_lengths) if sentence_lengths else 0
        std_sentence_len = np.std(sentence_lengths) if sentence_lengths else 0
        max_sentence_len = max(sentence_lengths) if sentence_lengths else 0
        min_sentence_len = min(sentence_lengths) if sentence_lengths else 0

        # Sentence complexity
        avg_words_per_sentence = word_count / max(sentence_count, 1)

        # Question and exclamation ratios
        question_sentences = text.count("?")
        exclaim_sentences = text.count("!")
        question_ratio = question_sentences / max(sentence_count, 1)
        exclaim_ratio = exclaim_sentences / max(sentence_count, 1)

        return [
            sentence_count,
            avg_sentence_len,
            std_sentence_len,
            max_sentence_len,
            min_sentence_len,
            avg_words_per_sentence,
            question_ratio,
            exclaim_ratio,
        ]

    def _extract_social_media_features(self, text: str) -> List[float]:
        """Extract social media specific features."""
        text_len = max(len(text), 1)

        # Count social media elements
        urls = len(self.patterns["urls"].findall(text))
        emails = len(self.patterns["emails"].findall(text))
        mentions = len(self.patterns["mentions"].findall(text))
        hashtags = len(self.patterns["hashtags"].findall(text))

        # Repeated patterns (potential indicators of artificial text)
        repeated_chars = len(self.patterns["repeated_chars"].findall(text))
        repeated_words = len(self.patterns["repeated_words"].findall(text))
        all_caps_words = len(self.patterns["all_caps"].findall(text))
        punct_clusters = len(self.patterns["punctuation_clusters"].findall(text))

        # Normalize by text length
        return [
            urls / text_len * 1000,  # per 1000 characters
            emails / text_len * 1000,
            mentions / text_len * 1000,
            hashtags / text_len * 1000,
            repeated_chars / text_len * 1000,
            repeated_words / text_len * 1000,
            all_caps_words / text_len * 1000,
            punct_clusters / text_len * 1000,
        ]

    def _extract_readability_features(self, text: str) -> List[float]:
        """Extract readability scores."""
        words = text.split()
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        if not words or not sentences:
            return [0] * 4

        word_count = len(words)
        sentence_count = len(sentences)

        # Average sentence length
        avg_sentence_length = word_count / sentence_count

        # Estimate syllable count (simplified)
        total_syllables = 0
        for word in words:
            # Remove punctuation and convert to lowercase
            clean_word = word.strip(string.punctuation).lower()
            if clean_word:
                # Count vowel groups as syllables
                syllable_count = len(re.findall(r"[aeiouy]+", clean_word))
                # At least 1 syllable per word
                syllable_count = max(1, syllable_count)
                total_syllables += syllable_count

        avg_syllables_per_word = total_syllables / max(word_count, 1)

        # Flesch Reading Ease Score
        flesch_score = (
            206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        )

        # Flesch-Kincaid Grade Level
        grade_level = (
            (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        )

        # Automated Readability Index (ARI)
        char_count = sum(len(word.strip(string.punctuation)) for word in words)
        ari_score = (
            4.71 * (char_count / word_count)
            + 0.5 * (word_count / sentence_count)
            - 21.43
        )

        return [
            np.clip(flesch_score, -100, 200),  # Flesch Reading Ease
            np.clip(grade_level, 0, 20),  # Grade Level
            np.clip(ari_score, 0, 20),  # ARI Score
            avg_syllables_per_word,  # Avg syllables per word
        ]

    def _extract_stylometric_features(self, text: str) -> List[float]:
        """Extract stylometric features that might distinguish AI vs human text."""
        words = text.split()
        if not words:
            return [0] * 12

        word_count = len(words)

        # Most frequent words analysis
        word_freq = Counter(word.lower().strip(string.punctuation) for word in words)
        most_common = word_freq.most_common(10)

        # Frequency of most common word
        most_freq_ratio = most_common[0][1] / word_count if most_common else 0

        # Hapax legomena (words appearing exactly once)
        hapax_count = sum(1 for count in word_freq.values() if count == 1)
        hapax_ratio = hapax_count / max(word_count, 1)

        # Dis legomena (words appearing exactly twice)
        dis_count = sum(1 for count in word_freq.values() if count == 2)
        dis_ratio = dis_count / max(word_count, 1)

        # Yule's K (measure of vocabulary richness)
        N = word_count
        V1 = hapax_count
        if N > 0 and V1 > 0:
            yules_k = (
                10000
                * (sum(count * count for count in word_freq.values()) - N)
                / (N**2)
            )
        else:
            yules_k = 0

        # Simpson's D (vocabulary diversity)
        if N > 1:
            simpsons_d = sum(count * (count - 1) for count in word_freq.values()) / (
                N * (N - 1)
            )
        else:
            simpsons_d = 0

        # Honoré's R (vocabulary richness)
        V = len(word_freq)  # Vocabulary size
        if V > 0:
            honores_r = 100 * np.log(N) / max(1 - V1 / V, 0.01)
        else:
            honores_r = 0

        # Brunét's W (vocabulary richness)
        if V > 0:
            brunets_w = N ** (V**-0.165)
        else:
            brunets_w = 0

        # Type-Token Ratio variations
        ttr = V / max(N, 1)  # Basic TTR
        rttr = V / np.sqrt(N) if N > 0 else 0  # Root TTR
        cttr = V / np.sqrt(2 * N) if N > 0 else 0  # Corrected TTR

        # Measure of Textual Lexical Diversity (MTLD) approximation
        # Simplified version - count how many times TTR drops below 0.72
        mtld_segments = 0
        current_segment_words = []
        current_vocab = set()

        for word in words:
            clean_word = word.lower().strip(string.punctuation)
            if clean_word:
                current_segment_words.append(clean_word)
                current_vocab.add(clean_word)

                if len(current_segment_words) > 10:  # Minimum segment length
                    segment_ttr = len(current_vocab) / len(current_segment_words)
                    if segment_ttr < 0.72:
                        mtld_segments += 1
                        current_segment_words = []
                        current_vocab = set()

        mtld_approx = word_count / max(mtld_segments, 1)

        # Average word frequency
        avg_word_freq = np.mean(list(word_freq.values())) if word_freq else 0

        return [
            most_freq_ratio,
            hapax_ratio,
            dis_ratio,
            yules_k,
            simpsons_d,
            honores_r,
            brunets_w,
            ttr,
            rttr,
            cttr,
            mtld_approx,
            avg_word_freq,
        ]

    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        names = []

        # Basic stats
        names.extend(
            [
                "text_len",
                "char_count",
                "word_count",
                "char_density",
                "word_density",
                "avg_word_len_chars",
            ]
        )

        # Character features
        names.extend(
            [
                "upper_ratio",
                "lower_ratio",
                "digit_ratio",
                "alpha_ratio",
                "space_ratio",
                "punct_ratio",
                "question_ratio",
                "exclaim_ratio",
                "period_ratio",
                "comma_ratio",
                "quote_ratio",
                "char_diversity",
            ]
        )

        # Word features
        names.extend(
            [
                "word_avg_len",
                "word_std_len",
                "word_max_len",
                "word_min_len",
                "lexical_diversity",
                "long_word_ratio",
                "short_word_ratio",
                "function_word_ratio",
                "capitalized_ratio",
                "unique_word_count",
            ]
        )

        # Sentence features
        names.extend(
            [
                "sentence_count",
                "sentence_avg_len",
                "sentence_std_len",
                "sentence_max_len",
                "sentence_min_len",
                "avg_words_per_sentence",
                "question_sentence_ratio",
                "exclaim_sentence_ratio",
            ]
        )

        # Social media features
        names.extend(
            [
                "urls_per_1k",
                "emails_per_1k",
                "mentions_per_1k",
                "hashtags_per_1k",
                "repeated_chars_per_1k",
                "repeated_words_per_1k",
                "all_caps_per_1k",
                "punct_clusters_per_1k",
            ]
        )

        if self.include_readability:
            names.extend(
                ["flesch_score", "grade_level", "ari_score", "avg_syllables_per_word"]
            )

        if self.include_stylometric:
            names.extend(
                [
                    "most_freq_ratio",
                    "hapax_ratio",
                    "dis_ratio",
                    "yules_k",
                    "simpsons_d",
                    "honores_r",
                    "brunets_w",
                    "ttr",
                    "rttr",
                    "cttr",
                    "mtld_approx",
                    "avg_word_freq",
                ]
            )

        return names


class EnhancedFeatureExtractor(BaseEstimator, TransformerMixin):
    """Enhanced feature extraction pipeline with multiple techniques and feature selection."""

    def __init__(
        self,
        # TF-IDF parameters
        max_word_features: int = 8000,
        max_char_features: int = 4000,
        word_ngram_range: Tuple[int, int] = (1, 3),
        char_ngram_range: Tuple[int, int] = (2, 5),
        min_df: int = 2,
        max_df: float = 0.95,
        # Feature selection parameters
        feature_selection_method: str = "f_classif",  # 'chi2', 'f_classif', 'mutual_info'
        select_k_best: Optional[int] = 3000,  # None to disable
        variance_threshold: float = 0.0001,  # Much lower threshold to preserve TF-IDF features
        # Dimensionality reduction
        use_pca: bool = False,
        pca_components: Optional[int] = None,
        use_svd: bool = True,
        svd_components: int = 300,
        # Scaling
        scaler_type: str = "standard",  # 'standard', 'minmax', 'robust', 'none'
        # Other features
        include_pos_tags: bool = False,  # Requires nltk
        include_embeddings: bool = False,  # Requires sentence-transformers
    ):
        """Initialize the enhanced feature extractor."""
        self.max_word_features = max_word_features
        self.max_char_features = max_char_features
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.min_df = min_df
        self.max_df = max_df

        self.feature_selection_method = feature_selection_method
        self.select_k_best = select_k_best
        self.variance_threshold = variance_threshold

        self.use_pca = use_pca
        self.pca_components = pca_components
        self.use_svd = use_svd
        self.svd_components = svd_components

        self.scaler_type = scaler_type
        self.include_pos_tags = include_pos_tags
        self.include_embeddings = include_embeddings

        # Initialize components
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.linguistic_extractor = None
        self.feature_selector = None
        self.variance_selector = None
        self.dimensionality_reducer = None
        self.scaler = None
        self.pipeline = None

    def fit(self, X, y=None):
        """Fit the feature extraction pipeline."""
        print(f"Fitting enhanced feature extractor on {len(X)} texts...")

        # Build the pipeline
        steps = []

        # 1. Extract multiple types of features
        feature_extractors = []

        # Word-level TF-IDF
        self.word_vectorizer = TfidfVectorizer(
            max_features=self.max_word_features,
            ngram_range=self.word_ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words="english",
            lowercase=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b",  # Default pattern that handles contractions properly
            sublinear_tf=True,  # Use log-scaled frequencies
        )
        feature_extractors.append(("word_tfidf", self.word_vectorizer))

        # Character-level TF-IDF
        self.char_vectorizer = TfidfVectorizer(
            max_features=self.max_char_features,
            analyzer="char",
            ngram_range=self.char_ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=True,
            sublinear_tf=True,
        )
        feature_extractors.append(("char_tfidf", self.char_vectorizer))

        # Advanced linguistic features
        self.linguistic_extractor = AdvancedLinguisticFeatureExtractor(
            include_pos_tags=self.include_pos_tags
        )
        feature_extractors.append(("linguistic", self.linguistic_extractor))

        # Combine all feature extractors
        steps.append(("features", FeatureUnion(feature_extractors, n_jobs=-1)))

        # 2. Variance-based feature selection (remove low-variance features)
        if self.variance_threshold > 0:
            self.variance_selector = VarianceThreshold(
                threshold=self.variance_threshold
            )
            steps.append(("variance_selection", self.variance_selector))

        # 3. Statistical feature selection
        if self.select_k_best is not None:
            if self.feature_selection_method == "chi2":
                selector_func = chi2
            elif self.feature_selection_method == "f_classif":
                selector_func = f_classif
            elif self.feature_selection_method == "mutual_info":
                selector_func = mutual_info_classif
            else:
                raise ValueError(
                    f"Unknown feature selection method: {self.feature_selection_method}"
                )

            self.feature_selector = SelectKBest(
                score_func=selector_func,
                k=min(
                    self.select_k_best, self.max_word_features + self.max_char_features
                ),
            )
            steps.append(("feature_selection", self.feature_selector))

        # Note: Dimensionality reduction will be added after we know the actual feature count
        # This is handled in the fit process below

        # 5. Convert to dense and apply scaling (use module-level class for pickling)
        steps.append(("to_dense", SparseToDenseTransformer()))

        # Apply scaling after converting to dense
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.scaler_type != "none":
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        if self.scaler is not None:
            steps.append(("scaling", self.scaler))

        # Create initial pipeline without dimensionality reduction
        initial_pipeline = Pipeline(steps)

        # Fit the initial pipeline to determine feature count after selection
        if y is not None and self.select_k_best is not None:
            initial_pipeline.fit(X, y)
        else:
            initial_pipeline.fit(X)

        # Check how many features we have after selection
        temp_features = initial_pipeline.transform(
            X[:10]
        )  # Small sample to check dimensions
        n_features_after_selection = temp_features.shape[1]

        # Add dimensionality reduction if requested and we have enough features
        if (self.use_svd or self.use_pca) and n_features_after_selection > 1:
            if self.use_svd:
                # Use at most n_features - 1 for SVD, and at most the requested components
                n_components = min(
                    self.svd_components,
                    n_features_after_selection - 1,
                    max(
                        1, n_features_after_selection // 2
                    ),  # Use at most half the features
                )
                if n_components > 0:
                    self.dimensionality_reducer = TruncatedSVD(
                        n_components=n_components, random_state=42
                    )
                    steps.append(
                        ("dimensionality_reduction", self.dimensionality_reducer)
                    )
            elif self.use_pca:
                n_components = min(
                    self.pca_components or min(100, len(X) // 2),
                    n_features_after_selection - 1,
                    max(1, n_features_after_selection // 2),
                )
                if n_components > 0:
                    self.dimensionality_reducer = PCA(
                        n_components=n_components, random_state=42
                    )
                    steps.append(
                        ("dimensionality_reduction", self.dimensionality_reducer)
                    )

        # Rebuild and fit final pipeline with dimensionality reduction if added
        self.pipeline = Pipeline(steps)
        if y is not None and self.select_k_best is not None:
            self.pipeline.fit(X, y)
        else:
            self.pipeline.fit(X)

        # CRITICAL: Store vocabularies directly as attributes for interpretability
        self._store_vocabularies()

        # Print information about the feature extraction
        self._print_feature_info(X)

        return self

    def _store_vocabularies(self):
        """Store vocabularies directly as attributes for interpretability."""
        try:
            # Access the vectorizers through the fitted pipeline
            feature_union = self.pipeline.named_steps.get("features")

            # Get word vectorizer from the feature union
            word_vectorizer = None
            char_vectorizer = None

            if feature_union and hasattr(feature_union, "transformer_list"):
                for name, transformer in feature_union.transformer_list:
                    if name == "word_tfidf":
                        word_vectorizer = transformer
                    elif name == "char_tfidf":
                        char_vectorizer = transformer

            # Store word vocabulary
            if (
                word_vectorizer is not None
                and hasattr(word_vectorizer, "vocabulary_")
                and word_vectorizer.vocabulary_
            ):
                self._word_vocabulary = dict(word_vectorizer.vocabulary_)
                print(f"Stored word vocabulary with {len(self._word_vocabulary)} terms")
            else:
                self._word_vocabulary = {}
                print("No word vocabulary to store")

            # Store character vocabulary
            if (
                char_vectorizer is not None
                and hasattr(char_vectorizer, "vocabulary_")
                and char_vectorizer.vocabulary_
            ):
                self._char_vocabulary = dict(char_vectorizer.vocabulary_)
                print(f"Stored char vocabulary with {len(self._char_vocabulary)} terms")
            else:
                self._char_vocabulary = {}
                print("No char vocabulary to store")

        except Exception as e:
            print(f"Warning: Could not store vocabularies: {e}")
            self._word_vocabulary = {}
            self._char_vocabulary = {}

    def transform(self, X):
        """Transform texts using the fitted pipeline."""
        if self.pipeline is None:
            raise ValueError("Feature extractor not fitted. Call fit() first.")

        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def _print_feature_info(self, X):
        """Print information about the extracted features."""
        if self.pipeline is None:
            return

        # Get feature counts at each stage
        feature_counts = []

        # Initial features
        temp_features = self.pipeline.named_steps["features"].transform(
            X[:100]
        )  # Small sample
        initial_features = temp_features.shape[1]
        feature_counts.append(f"Initial features: {initial_features}")

        # After variance selection
        if "variance_selection" in self.pipeline.named_steps:
            temp_features = self.pipeline.named_steps["variance_selection"].transform(
                temp_features
            )
            feature_counts.append(f"After variance selection: {temp_features.shape[1]}")

        # After feature selection
        if "feature_selection" in self.pipeline.named_steps:
            temp_features = self.pipeline.named_steps["feature_selection"].transform(
                temp_features
            )
            feature_counts.append(f"After feature selection: {temp_features.shape[1]}")

        # After dimensionality reduction
        if "dimensionality_reduction" in self.pipeline.named_steps:
            temp_features = self.pipeline.named_steps[
                "dimensionality_reduction"
            ].transform(temp_features)
            feature_counts.append(
                f"After dimensionality reduction: {temp_features.shape[1]}"
            )

        print("Enhanced feature extraction pipeline:")
        for count in feature_counts:
            print(f"  {count}")

    def get_feature_names_out(self):
        """Get feature names after transformation, including actual words and char n-grams."""
        if self.pipeline is None:
            return []

        try:
            # Get feature names from the feature union (before selection/reduction)
            all_feature_names = []

            # Get word features
            if hasattr(self.word_vectorizer, "get_feature_names_out"):
                try:
                    word_features = self.word_vectorizer.get_feature_names_out()
                    word_feature_names = [f"word_{word}" for word in word_features]
                    all_feature_names.extend(word_feature_names)
                except Exception:
                    pass
            elif (
                hasattr(self.word_vectorizer, "vocabulary_")
                and self.word_vectorizer.vocabulary_
            ):
                try:
                    vocab = self.word_vectorizer.vocabulary_
                    word_features = [""] * len(vocab)
                    for word, idx in vocab.items():
                        if idx < len(word_features):
                            word_features[idx] = word
                    word_feature_names = [
                        f"word_{word}" for word in word_features if word
                    ]
                    all_feature_names.extend(word_feature_names)
                except Exception:
                    pass

            # Get character features
            if hasattr(self.char_vectorizer, "get_feature_names_out"):
                try:
                    char_features = self.char_vectorizer.get_feature_names_out()
                    char_feature_names = [f"char_{char}" for char in char_features]
                    all_feature_names.extend(char_feature_names)
                except Exception:
                    pass
            elif (
                hasattr(self.char_vectorizer, "vocabulary_")
                and self.char_vectorizer.vocabulary_
            ):
                try:
                    vocab = self.char_vectorizer.vocabulary_
                    char_features = [""] * len(vocab)
                    for char, idx in vocab.items():
                        if idx < len(char_features):
                            char_features[idx] = char
                    char_feature_names = [
                        f"char_{char}" for char in char_features if char
                    ]
                    all_feature_names.extend(char_feature_names)
                except Exception:
                    pass

            # Get linguistic feature names
            if hasattr(self.linguistic_extractor, "get_feature_names"):
                try:
                    linguistic_names = self.linguistic_extractor.get_feature_names()
                    all_feature_names.extend(linguistic_names)
                except Exception:
                    pass

            # If we got feature names, handle feature selection and dimensionality reduction
            if all_feature_names:
                current_features = all_feature_names

                # Apply variance selection if present
                if (
                    hasattr(self, "variance_selector")
                    and self.variance_selector is not None
                ):
                    try:
                        if hasattr(self.variance_selector, "get_support"):
                            support_mask = self.variance_selector.get_support()
                            current_features = [
                                name
                                for name, keep in zip(current_features, support_mask)
                                if keep
                            ]
                    except Exception:
                        pass

                # Apply feature selection if present
                if (
                    hasattr(self, "feature_selector")
                    and self.feature_selector is not None
                ):
                    try:
                        if hasattr(self.feature_selector, "get_support"):
                            support_mask = self.feature_selector.get_support()
                            current_features = [
                                name
                                for name, keep in zip(current_features, support_mask)
                                if keep
                            ]
                    except Exception:
                        pass

                # If dimensionality reduction was applied, we lose individual feature names
                if (
                    "dimensionality_reduction" in self.pipeline.named_steps
                    and self.dimensionality_reducer is not None
                ):
                    n_components = len(current_features)
                    if hasattr(self.dimensionality_reducer, "n_components"):
                        n_components = self.dimensionality_reducer.n_components
                    current_features = [f"component_{i}" for i in range(n_components)]

                return current_features

            # Fallback to generic names
            final_features = self.transform(["sample text"]).shape[1]
            return [f"feature_{i}" for i in range(final_features)]

        except Exception as e:
            print(f"Warning: Could not extract feature names: {e}")
            # Final fallback
            try:
                final_features = self.transform(["sample text"]).shape[1]
                return [f"feature_{i}" for i in range(final_features)]
            except Exception:
                return []

    def get_word_feature_names(self):
        """Get just the word feature names with their vocabularies."""
        if self.word_vectorizer is None:
            return []

        try:
            if hasattr(self.word_vectorizer, "get_feature_names_out"):
                return list(self.word_vectorizer.get_feature_names_out())
            elif (
                hasattr(self.word_vectorizer, "vocabulary_")
                and self.word_vectorizer.vocabulary_
            ):
                vocab = self.word_vectorizer.vocabulary_
                word_features = [""] * len(vocab)
                for word, idx in vocab.items():
                    if idx < len(word_features):
                        word_features[idx] = word
                return [word for word in word_features if word]
        except Exception as e:
            print(f"Warning: Could not extract word features: {e}")

        return []

    def get_char_feature_names(self):
        """Get just the character n-gram feature names with their vocabularies."""
        if self.char_vectorizer is None:
            return []

        try:
            if hasattr(self.char_vectorizer, "get_feature_names_out"):
                return list(self.char_vectorizer.get_feature_names_out())
            elif (
                hasattr(self.char_vectorizer, "vocabulary_")
                and self.char_vectorizer.vocabulary_
            ):
                vocab = self.char_vectorizer.vocabulary_
                char_features = [""] * len(vocab)
                for char, idx in vocab.items():
                    if idx < len(char_features):
                        char_features[idx] = char
                return [char for char in char_features if char]
        except Exception as e:
            print(f"Warning: Could not extract char features: {e}")

        return []


def create_enhanced_feature_extractor(
    config: str = "default",
) -> EnhancedFeatureExtractor:
    """Create enhanced feature extractor with different configurations."""
    configs = {
        "default": {
            "max_word_features": 8000,
            "max_char_features": 4000,
            "word_ngram_range": (1, 3),
            "char_ngram_range": (2, 5),
            "select_k_best": 8000,  # Keep more features
            "variance_threshold": 0.0001,
            "use_svd": False,  # Don't reduce further for default
        },
        "baseline_plus": {
            "max_word_features": 5000,  # Match your current system
            "max_char_features": 2500,  # Match your current system
            "word_ngram_range": (1, 2),  # Match your current system
            "char_ngram_range": (2, 4),  # Match your current system
            "select_k_best": None,  # No feature selection - keep all like current system
            "variance_threshold": 0,  # No variance filtering
            "use_svd": False,  # No dimensionality reduction
            "scaler_type": "standard",  # Match your current system
        },
        "high_dim": {
            "max_word_features": 15000,
            "max_char_features": 8000,
            "word_ngram_range": (1, 4),
            "char_ngram_range": (2, 6),
            "select_k_best": 10000,
            "use_svd": True,
            "svd_components": 500,
        },
        "compact": {
            "max_word_features": 5000,
            "max_char_features": 2000,
            "word_ngram_range": (1, 2),
            "char_ngram_range": (2, 4),
            "select_k_best": 3000,
            "use_svd": True,
            "svd_components": 200,
        },
        "stylometric_focus": {
            "max_word_features": 6000,
            "max_char_features": 3000,
            "word_ngram_range": (1, 3),
            "char_ngram_range": (3, 6),  # Longer char n-grams for style
            "select_k_best": 4000,
            "use_svd": False,
            "use_pca": True,
            "pca_components": 250,
            "feature_selection_method": "mutual_info",  # Better for stylometric features
        },
    }

    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")

    return EnhancedFeatureExtractor(**configs[config])


# Example usage and testing functions
def compare_feature_extractors(texts, labels, test_size=0.2):
    """Compare different feature extraction approaches."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # Test different configurations
    configs = ["default", "high_dim", "compact", "stylometric_focus"]
    results = {}

    for config in configs:
        print(f"\nTesting {config} configuration...")

        try:
            # Create and fit feature extractor
            extractor = create_enhanced_feature_extractor(config)
            X_train_features = extractor.fit_transform(X_train, y_train)
            X_test_features = extractor.transform(X_test)

            # Train simple classifier
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(X_train_features, y_train)

            # Evaluate
            y_pred = clf.predict(X_test_features)
            accuracy = accuracy_score(y_test, y_pred)

            results[config] = {
                "accuracy": accuracy,
                "n_features": X_train_features.shape[1],
                "feature_extractor": extractor,
            }

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Features: {X_train_features.shape[1]}")

        except Exception as e:
            print(f"  Failed: {e}")
            results[config] = {"error": str(e)}

    return results


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This is a sample human text with natural language patterns.",
        "Generated text may exhibit different stylistic patterns and vocabulary usage.",
        "Human writing often contains informal elements, contractions, "
        "and varied sentence structures.",
    ]

    sample_labels = [1, 0, 1]  # 1 for human, 0 for AI

    # Create and test feature extractor
    extractor = create_enhanced_feature_extractor("default")
    features = extractor.fit_transform(sample_texts, sample_labels)

    print(f"Extracted {features.shape[1]} features from {len(sample_texts)} texts")
    print(f"Feature shape: {features.shape}")
