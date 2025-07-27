import re
from bs4 import BeautifulSoup

def clean_text(raw_text: str) -> str:
    """
    Performs robust cleaning on raw text data.
    
    Args:
        raw_text: The input string to be cleaned.

    Returns:
        A cleaned version of the string.
        
    Raises:
        TypeError: If the input is not a string.
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

    # 4. Remove emojis and other pictographic symbols
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251"
        u"\U0001f900-\U0001f9ff" u"\u2600-\u26FF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r'', text)
    
    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Convert to lowercase
    text = text.lower()
    
    return text
