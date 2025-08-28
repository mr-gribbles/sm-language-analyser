"""Text analysis module for linguistic metrics.

This module provides functions to analyze text data for readability,
lexical diversity, and sentiment. It uses the textstat library for
readability scores, NLTK for lexical diversity and sentiment analysis,
and handles potential issues with NLTK data downloads gracefully.
It includes error handling for SSL certificate issues that can occur
on certain systems, particularly macOS.
"""
import ssl
import sys
import nltk
import textstat
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize


def initialize_nltk():
    """Handle downloading of required NLTK data with error handling.

    Provides clear error messages if downloads fail due to SSL or other issues.
    Ensures all necessary NLTK resources are available before performing analysis.
    """
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass  # For systems where this is not needed

    required_resources = ["vader_lexicon", "punkt"]
    for resource in required_resources:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

# Run the initialization and catch potential failures.
try:
    initialize_nltk()
    sid = SentimentIntensityAnalyzer()
except Exception as e:
    print("="*80)
    print("ERROR: A critical error occurred while initializing the NLTK analyzer.")
    print("This is likely because the required NLTK data could not be downloaded.")
    print(f"The specific error was: {e}")
    print("\nTo fix this, please try the following steps:")
    print("1. Run the manual SSL certificate installer for Python (especially on macOS):")
    print("   - Open Finder -> Applications -> Python 3.11 (or your version)")
    print("   - Double-click on 'Install Certificates.command'")
    print("\n2. If that doesn't work, download all required data manually in your terminal:")
    print("   - Make sure your virtual environment is active.")
    print("   - Run the command: python -m nltk.downloader vader_lexicon punkt punkt_tab")
    print("\nAfter running these steps, try the analysis script again.")
    print("="*80)
    sys.exit(1) # Exit the script gracefully.

def analyze_readability(text: str) -> dict:
    """Calculate readability scores for a given text.

    Args:
        text: The text to analyze (should be a string with at least 10 words).

    Returns:
        A dictionary containing the Flesch Reading Ease and Flesch-Kincaid
        Grade Level scores. If the text is too short or not a string,
        returns None for both scores.
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
    """Calculate the Type-Token Ratio (TTR) for a text.

    Args:
        text: The text to analyze (should be a string).

    Returns:
        A dictionary containing the TTR score. If the text is empty
        or not a string, returns TTR as None.
    """
    if not text or not isinstance(text, str):
        return {"ttr": None}
    
    tokens = word_tokenize(text.lower())
    if not tokens:
        return {"ttr": 0}
        
    unique_tokens = set(tokens)
    ttr = len(unique_tokens) / len(tokens) if len(tokens) > 0 else 0
    return {"ttr": ttr}

def analyze_sentiment(text: str) -> dict:
    """Perform VADER sentiment analysis on a text.

    Args:
        text: The text to analyze (should be a string).

    Returns:
        A dictionary containing positive, negative, neutral, and compound
        sentiment scores. If the text is empty or not a string, returns
        None for all scores.
    """
    if not text or not isinstance(text, str):
        return {
            "sentiment_positive": None,
            "sentiment_negative": None,
            "sentiment_neutral": None,
            "sentiment_compound": None
        }
    
    scores = sid.polarity_scores(text)
    return {
        "sentiment_positive": scores['pos'],
        "sentiment_negative": scores['neg'],
        "sentiment_neutral": scores['neu'],
        "sentiment_compound": scores['compound']
    }
def run_full_analysis(text: str) -> dict:
    """Run all text analyses and combine them into a single dictionary.

    Args:
        text: The text to analyze (should be a string).

    Returns:
        A dictionary containing all analysis results: readability, lexical
        diversity, and sentiment. If the text is empty or not a string,
        returns an empty dictionary.
    """
    if not text:
        return {}
        
    readability = analyze_readability(text)
    lexical = analyze_lexical_diversity(text)
    sentiment = analyze_sentiment(text)
    
    # Combine all dictionaries into one
    full_report = {**readability, **lexical, **sentiment}
    return full_report
