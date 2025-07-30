import sys
import os
import pytest

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core_logic.data_cleaner import clean_text

def test_clean_text_removes_urls():
    """Ensures that URLs are removed from the text."""
    text = "This is a text with a url: https://example.com"
    expected = "this is a text with a url:"
    assert clean_text(text).strip() == expected

def test_clean_text_removes_special_chars():
    """Ensures that special characters are removed."""
    text = "This is a text with special characters: !@#$%^&*()"
    expected = "this is a text with special characters: !"
    assert clean_text(text) == expected

def test_clean_text_normalizes_whitespace():
    """Ensures that multiple whitespace characters are normalized to a single space."""
    text = "This is a text with    multiple   spaces."
    expected = "this is a text with multiple spaces."
    assert clean_text(text) == expected

def test_clean_text_removes_html_tags():
    """Ensures that HTML tags are removed."""
    text = "<p>This is a <b>paragraph</b> with <a href='#'>a link</a>.</p>"
    expected = "this is a paragraph with a link."
    assert clean_text(text) == expected

def test_clean_text_removes_emojis():
    """Ensures that emojis are removed."""
    text = "This is a text with emojis 👍🎉"
    expected = "this is a text with emojis"
    assert clean_text(text) == expected

def test_clean_text_handles_empty_string():
    """Ensures that an empty string is handled correctly."""
    text = ""
    expected = ""
    assert clean_text(text) == expected

def test_clean_text_handles_none_input():
    """Ensures that a None input is handled gracefully."""
    with pytest.raises(TypeError):
        clean_text(None)
