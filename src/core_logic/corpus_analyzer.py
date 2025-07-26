import textstat
import nltk
import ssl
import sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

def _initialize_nltk_data():
    """
    Handles downloading of required NLTK data and provides a clear
    error message if downloads fail due to SSL or other issues.
    """
    try:
        # Attempt to create an unverified SSL context for systems with certificate issues.
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Define the resources that need to be checked and downloaded.
    required_resources = {
        'tokenizers/punkt': 'punkt',
        'sentiment/vader_lexicon.zip': 'vader_lexicon',
        'tokenizers/punkt_tab': 'punkt_tab' # Added to handle the specific error
    }

    # Try to download the necessary NLTK resources.
    for path, resource_id in required_resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"NLTK resource '{resource_id}' not found. Attempting to download...")
            nltk.download(resource_id, quiet=True)

# Run the initialization and catch potential failures.
try:
    _initialize_nltk_data()
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
    """Calculates readability scores for a given text."""
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
    """Calculates the Type-Token Ratio (TTR) for a text."""
    if not text or not isinstance(text, str):
        return {"ttr": None}
    
    tokens = word_tokenize(text.lower())
    if not tokens:
        return {"ttr": 0}
        
    unique_tokens = set(tokens)
    ttr = len(unique_tokens) / len(tokens) if len(tokens) > 0 else 0
    return {"ttr": ttr}

def analyze_sentiment(text: str) -> dict:
    """Performs VADER sentiment analysis on a text."""
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
    """Runs all text analyses and combines them into a single dictionary."""
    if not text:
        return {}
        
    readability = analyze_readability(text)
    lexical = analyze_lexical_diversity(text)
    sentiment = analyze_sentiment(text)
    
    # Combine all dictionaries into one
    full_report = {**readability, **lexical, **sentiment}
    return full_report
