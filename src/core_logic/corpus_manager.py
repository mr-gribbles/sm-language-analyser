"""Corpus record creation and management for social media posts.

This module provides functions to create structured records for posts collected
from Reddit and Bluesky, and save them in the official corpus JSON format.
It handles both original and rewritten content, including metadata about the
source, content, and any transformations applied by LLMs.
"""
import json
import os
import uuid
from datetime import datetime, timezone


def create_corpus_record(source_details, original_content, rewritten_text=None,
                         llm_model=None, prompt_template=None):
    """Structure collected data into the official corpus JSON format.

    Args:
        source_details: Dictionary containing platform-specific source metadata.
        original_content: Dictionary containing the original post content.
        rewritten_text: Optional rewritten text from LLM transformation.
        llm_model: Optional name of the LLM model used for rewriting.
        prompt_template: Optional prompt template used for rewriting.

    Returns:
        A structured corpus record dictionary in the official format.
    """
    record = {
        "corpus_item_id": str(uuid.uuid4()),
        "version": "1.1",
        "source_details": source_details,
        "original_content": original_content,
        "llm_transformation": None,
    }

    if rewritten_text and llm_model and prompt_template:
        record["llm_transformation"] = {
            "model_used": llm_model,
            "prompt_template": prompt_template,
            "rewritten_text": rewritten_text,
            "transformation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
    return record


def save_record_to_corpus(record, directory, filename):
    """Save a corpus record to a specified directory in JSON Lines format.

    Args:
        record: The corpus record dictionary to save.
        directory: The directory where the record should be saved.
        filename: The name of the file to save the record in.
    """
    try:
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    except IOError as e:
        print(f"Error: Could not write to file {filepath}. Details: {e}")
