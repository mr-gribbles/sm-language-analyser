import re
from bs4 import BeautifulSoup

def clean_text(raw_text: str) -> str:
    """
    Performs robust cleaning on raw text data.
    - Removes HTML tags.
    - Normalizes smart quotes and other common punctuation.
    - Removes emojis and other pictographic symbols.
    - Removes zero-width characters.
    - Normalizes whitespace.
    - Converts to lowercase.
    
    Args:
        raw_text: The input string to be cleaned.

    Returns:
        A cleaned version of the string, or an empty string if input is invalid.
    """
    if not isinstance(raw_text, str):
        return ""
        
    # 1. Use BeautifulSoup to remove any potential HTML tags
    soup = BeautifulSoup(raw_text, "html.parser")
    text = soup.get_text()
    
    # 2. Normalize common non-ASCII punctuation to their ASCII equivalents
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Smart double quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # Smart single quotes
    text = text.replace('\u2026', '...')  # Ellipsis
    text = text.replace('\u2014', '--')   # Em dash
    text = text.replace('\u2013', '-')    # En dash
    
    # 3. Remove invisible characters like the Zero Width Joiner
    text = text.replace('\u200d', '')  # Zero Width Joiner
    text = text.replace('\u200b', '')  # Zero Width Space

    # 4. Remove emojis and other pictographic symbols using a comprehensive regex
    # This pattern covers a very wide range of Unicode blocks for symbols and emojis.
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
        u"\u2600-\u26FF"          # Miscellaneous Symbols
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r'', text)
    
    # 5. Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Convert to lowercase for consistency
    text = text.lower()
    
    return text