"""Analysis module for text data.

This module provides functions to analyze text data for readability,
lexical diversity, and sentiment. It uses the `textstat` library for
readability scores, NLTK for lexical diversity and sentiment analysis,
and handles potential issues with NLTK data downloads gracefully.
It also includes error handling for SSL certificate issues that can
occur on certain systems, particularly macOS.

It ensures that all necessary NLTK resources are available before
performing any analysis, and provides clear instructions for users
if initialization fails.
"""

import textstat
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

# Initialize the sentiment analyzer with error handling
# Download NLTK data if not available
def get_sentiment_analyzer():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        # Download required NLTK data if not found
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            return SentimentIntensityAnalyzer()
        except Exception as e:
            print(f"Warning: Could not initialize sentiment analyzer: {e}")
            return None

# Initialize the sentiment analyzer
sid = get_sentiment_analyzer()

def analyze_readability(text: str) -> dict:
    """Calculates readability scores for a given text.
    
    Keyword arguments:
    text -- The text to analyze (should be a string with at least 10 words).

    Returns:
    A dictionary containing the Flesch Reading Ease and Flesch-Kincaid Grade Level scores.
    If the text is too short or not a string, returns None for both scores.
    """
    if not text or not isinstance(text, str) or len(text.split()) < 10:
        return {
            "flesch_reading_ease": None,
            "flesch_kincaid_grade": None,
        }
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
    }

def analyze_lexical_diversity(text: str) -> dict:
    """Calculates the Type-Token Ratio (TTR) for a text.
    
    Keyword arguments:
    text -- The text to analyze (should be a string).
    
    Returns:
    A dictionary containing the TTR score.
    If the text is empty or not a string, returns TTR as None.
    """
    if not text or not isinstance(text, str):
        return {"ttr": None}
    
    try:
        tokens = word_tokenize(text.lower())
        if not tokens:
            return {"ttr": 0}
            
        unique_tokens = set(tokens)
        ttr = len(unique_tokens) / len(tokens) if len(tokens) > 0 else 0
        return {"ttr": ttr}
    except LookupError:
        # Download punkt tokenizer if not available
        try:
            nltk.download('punkt', quiet=True)
            tokens = word_tokenize(text.lower())
            if not tokens:
                return {"ttr": 0}
            unique_tokens = set(tokens)
            ttr = len(unique_tokens) / len(tokens) if len(tokens) > 0 else 0
            return {"ttr": ttr}
        except Exception as e:
            print(f"Warning: Lexical diversity analysis failed: {e}")
            # Fallback to simple word splitting
            tokens = text.lower().split()
            if not tokens:
                return {"ttr": 0}
            unique_tokens = set(tokens)
            ttr = len(unique_tokens) / len(tokens) if len(tokens) > 0 else 0
            return {"ttr": ttr}
    except Exception as e:
        print(f"Warning: Lexical diversity analysis failed: {e}")
        return {"ttr": None}

def analyze_sentiment(text: str) -> dict:
    """Performs VADER sentiment analysis on a text.
    
    Keyword arguments:
    text -- The text to analyze (should be a string).
    
    Returns:
    A dictionary containing positive, negative, neutral, and compound sentiment scores.
    If the text is empty or not a string, returns None for all scores.
    """
    if not text or not isinstance(text, str) or sid is None:
        return {
            "sentiment_positive": None,
            "sentiment_negative": None,
            "sentiment_neutral": None,
            "sentiment_compound": None
        }
    
    try:
        scores = sid.polarity_scores(text)
        return {
            "sentiment_positive": scores['pos'],
            "sentiment_negative": scores['neg'],
            "sentiment_neutral": scores['neu'],
            "sentiment_compound": scores['compound']
        }
    except Exception as e:
        print(f"Warning: Sentiment analysis failed: {e}")
        return {
            "sentiment_positive": None,
            "sentiment_negative": None,
            "sentiment_neutral": None,
            "sentiment_compound": None
        }
def run_full_analysis(text: str) -> dict:
    """Runs all text analyses and combines them into a single dictionary.
    
    Keyword arguments:
    text -- The text to analyze (should be a string).
    
    Returns:
    A dictionary containing all analysis results: readability, lexical diversity, and sentiment.
    If the text is empty or not a string, returns an empty dictionary.
    """
    if not text:
        return {}
        
    readability = analyze_readability(text)
    lexical = analyze_lexical_diversity(text)
    sentiment = analyze_sentiment(text)
    
    # Combine all dictionaries into one
    full_report = {**readability, **lexical, **sentiment}
    return full_report
