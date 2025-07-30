"""This module provides functions to clean raw text data for further processing.

It includes robust cleaning steps such as removing HTML tags, normalizing punctuation,
stripping invisible characters, removing emojis, normalizing whitespace, and converting
the text to lowercase. The `clean_text` function is designed to handle various types of
raw text inputs, ensuring that the output is a clean and standardized string suitable for
text analysis or storage.
"""
import re
from bs4 import BeautifulSoup

def clean_text(raw_text: str) -> str:
    """Performs robust cleaning on raw text data.
    
    Keyword arguments:
    raw_text -- The raw text to be cleaned (should be a string).

    Returns:
    A cleaned version of the text, with HTML tags removed, punctuation normalized,
    invisible characters stripped, emojis removed, whitespace normalized, and converted to lowercase.
    If the input is not a string, raises a TypeError.
    """
    if not isinstance(raw_text, str):
        # Raise an exception if the input is not a string
        raise TypeError(f"Invalid input to clean_text: expected a string, but got {type(raw_text).__name__}.")
        
    # 1. Use BeautifulSoup to remove any potential HTML tags
    soup = BeautifulSoup(raw_text, "html.parser")
    text = soup.get_text()
    
    # 2. Normalize common non-ASCII punctuation
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u2026', '...')
    text = text.replace('\u2014', '--')
    text = text.replace('\u2013', '-')
    
    # 3. Remove invisible characters
    text = text.replace('\u200d', '').replace('\u200b', '')

    # 4. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 5. Remove emojis and other pictographic symbols
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251"
        u"\U0001f900-\U0001f9ff" u"\u2600-\u26FF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r'', text)

    # 6. Remove special characters, keeping only alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s\'\.\?,!:]', '', text)
    
    # 7. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 8. Convert to lowercase
    text = text.lower()
    
    return text
