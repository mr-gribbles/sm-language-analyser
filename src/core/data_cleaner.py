"""Text cleaning and preprocessing utilities.

This module provides robust text cleaning functionality including HTML tag removal,
punctuation normalization, invisible character stripping, emoji removal, whitespace
normalization, and case conversion. The clean_text function is designed to handle
various types of raw text inputs and produce standardized output suitable for
text analysis or storage.
"""

import re

from bs4 import BeautifulSoup


def clean_text(raw_text: str) -> str:
    """Perform light cleaning on raw text data, preserving linguistic features.

    This version of the cleaner is designed to keep important signals for
    AI vs. human text detection, such as capitalization, punctuation, and emojis.

    Args:
        raw_text: The raw text to be cleaned (must be a string).

    Returns:
        str: A cleaned version of the text with HTML tags removed, URLs removed,
            and whitespace normalized. Capitalization, punctuation, and emojis
            are preserved.

    Raises:
        TypeError: If the input is not a string.
    """
    if not isinstance(raw_text, str):
        # Raise an exception if the input is not a string
        raise TypeError(
            f"Invalid input to clean_text: expected a string, "
            f"but got {type(raw_text).__name__}."
        )

    # 1. Use BeautifulSoup to remove any potential HTML tags
    soup = BeautifulSoup(raw_text, "html.parser")
    text = soup.get_text()

    # 2. Normalize common non-ASCII punctuation
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2026", "...")
    text = text.replace("\u2014", "--")
    text = text.replace("\u2013", "-")

    # 3. Remove invisible characters
    text = text.replace("\u200d", "").replace("\u200b", "")

    # 4. Comprehensive URL removal
    # Remove full URLs (http/https)
    text = re.sub(r"https?://\S+", "", text, flags=re.MULTILINE)

    # Remove www URLs
    text = re.sub(r"www\.\S+", "", text, flags=re.MULTILINE)

    # Remove standalone "www" that might remain
    text = re.sub(r"\bwww\b", "", text, flags=re.MULTILINE)

    # Remove domain URLs (like domain.com, subdomain.domain.co.uk, etc.)
    # This pattern matches common domain patterns without being too aggressive
    text = re.sub(
        r"\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}"
        r"[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}(/\S*)?\b",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Remove any remaining URL-like patterns (belt and suspenders approach)
    text = re.sub(
        r"\S+\.(com|org|net|edu|gov|io|co|uk|de|fr|jp|au|ca|in|it|ru|br|mx|es|nl|se|no|dk"
        r"|fi|pl|be|at|ch|cz|ie|pt|gr|hu|ro|bg|hr|sk|si|lt|lv|ee|lu|mt|cy)\S*",
        "",
        text,
        flags=re.MULTILINE,
    )

    # 5. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
