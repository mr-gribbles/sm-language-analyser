import re
from bs4 import BeautifulSoup

def clean_text(raw_text: str) -> str:
    """
    Performs basic cleaning on raw text data.
    - Removes HTML tags.
    - Normalizes whitespace to a single space.
    - Converts text to lowercase.
    
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
    
    # 2. Replace multiple whitespace characters (including newlines, tabs) with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Convert to lowercase for consistency
    text = text.lower()
    
    return text
