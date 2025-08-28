"""Tests for the data cleaner module.

This module contains unit tests for the data_cleaner functionality,
including tests for URL removal, special character handling, whitespace
normalization, HTML tag removal, emoji removal, and error handling.
"""
import os
import sys

import pytest

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.data_cleaner import clean_text


def test_clean_text_removes_urls():
    """Test that URLs are removed from the text."""
    text = "This is a text with a url: https://example.com"
    expected = "This is a text with a url:"
    assert clean_text(text).strip() == expected

def test_clean_text_preserves_special_chars():
    """Test that special characters are preserved."""
    text = "This is a text with special characters: !@#$%^&*()"
    expected = "This is a text with special characters: !@#$%^&*()"
    assert clean_text(text) == expected


def test_clean_text_normalizes_whitespace():
    """Test that multiple whitespace characters are normalized to a single space."""
    text = "This is a text with    multiple   spaces."
    expected = "This is a text with multiple spaces."
    assert clean_text(text) == expected


def test_clean_text_removes_html_tags():
    """Test that HTML tags are removed."""
    text = "<p>This is a <b>paragraph</b> with <a href='#'>a link</a>.</p>"
    expected = "This is a paragraph with a link."
    assert clean_text(text) == expected


def test_clean_text_preserves_emojis():
    """Test that emojis are preserved."""
    text = "This is a text with emojis 👍🎉"
    expected = "This is a text with emojis 👍🎉"
    assert clean_text(text) == expected


def test_clean_text_handles_empty_string():
    """Test that an empty string is handled correctly."""
    text = ""
    expected = ""
    assert clean_text(text) == expected


def test_clean_text_handles_none_input():
    """Test that a None input is handled gracefully."""
    with pytest.raises(TypeError):
        clean_text(None)
