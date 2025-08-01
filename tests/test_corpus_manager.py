"""Tests for the corpus manager module.

This module contains unit tests for the corpus_manager functionality,
including tests for creating corpus records from Reddit and Bluesky posts.
"""
import os
import sys
from unittest.mock import MagicMock

import pytest

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core_logic.corpus_manager import create_corpus_record


@pytest.fixture
def reddit_post():
    """Provide a mock Reddit post object for testing.
    
    Returns:
        MagicMock: A mock Reddit post object with test data.
    """
    post = MagicMock()
    post.id = "test_id"
    post.permalink = "/r/test/comments/test_id"
    post.subreddit.display_name = "test"
    post.title = "Test Title"
    post.selftext = "This is the original post text."
    return post

@pytest.fixture
def bluesky_post():
    """Provide a mock Bluesky post object for testing.
    
    Returns:
        MagicMock: A mock Bluesky post object with test data.
    """
    post = MagicMock()
    post.uri = "at://did:plc:test/app.bsky.feed.post/test_rkey"
    post.cid = "test_cid"
    post.author.did = "did:plc:test"
    post.author.handle = "test.bsky.social"
    post.record.text = "This is the original bluesky post text."
    return post

def test_create_corpus_record_reddit(reddit_post):
    """Test that a Reddit corpus record is created correctly.
    
    Args:
        reddit_post: Mock Reddit post fixture.
    """
    source_details = {
        "platform": "Reddit",
        "post_id": reddit_post.id,
        "post_url": f"https://www.reddit.com{reddit_post.permalink}",
        "subreddit": reddit_post.subreddit.display_name.lower(),
    }
    original_content = {
        "title": reddit_post.title,
        "raw_selftext": reddit_post.selftext,
        "cleaned_selftext": "This is the cleaned post text.",
    }
    record = create_corpus_record(source_details, original_content)
    
    assert record["source_details"]["platform"] == "Reddit"
    assert record["original_content"]["title"] == "Test Title"
    assert record["llm_transformation"] is None

def test_create_corpus_record_bluesky_with_rewrite(bluesky_post):
    """Test that a Bluesky corpus record with a rewrite is created correctly.
    
    Args:
        bluesky_post: Mock Bluesky post fixture.
    """
    source_details = {
        "platform": "Bluesky",
        "post_uri": bluesky_post.uri,
        "post_cid": str(bluesky_post.cid),
        "author_did": bluesky_post.author.did,
        "author_handle": bluesky_post.author.handle,
    }
    original_content = {
        "title": "This is the original bluesky post text.",
        "raw_text": bluesky_post.record.text,
        "cleaned_text": "This is the cleaned bluesky post text.",
    }
    record = create_corpus_record(
        source_details,
        original_content,
        rewritten_text="This is the rewritten text.",
        llm_model="test_model",
        prompt_template="test_prompt",
    )

    assert record["source_details"]["platform"] == "Bluesky"
    assert record["llm_transformation"] is not None
    assert record["llm_transformation"]["model_used"] == "test_model"
