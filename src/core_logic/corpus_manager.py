"""Manages the creation and saving of corpus records for Reddit and Bluesky posts.

This module provides functions to create structured records for posts collected from Reddit and Bluesky,
and save them in the official corpus JSON format. It handles both original and rewritten content,
including metadata about the source, content, and any transformations applied by LLMs.
"""

import os
import json
import uuid
from datetime import datetime, timezone


def create_reddit_corpus_record(post, cleaned_text, rewritten_text=None, llm_model=None, prompt_template=None):
    """Structures collected Reddit data into the official corpus JSON format.
    
    Keyword arguments:
    post -- The Reddit post object containing details like ID, title, and selftext.
    cleaned_text -- The cleaned version of the post's selftext.
    rewritten_text -- Optional; the text rewritten by an LLM.
    llm_model -- Optional; the name of the LLM model used for rewriting.
    prompt_template -- Optional; the prompt template used for rewriting.
    
    Returns:
    A dictionary representing the corpus record, including source details, original content, and LLM transformation
    if applicable.
    """
    record = {
        "corpus_item_id": str(uuid.uuid4()),
        "version": "1.1",
        "source_details": {
            "platform": "Reddit",
            "post_id": post.id,
            "post_url": f"https://www.reddit.com{post.permalink}",
            "subreddit": post.subreddit.display_name.lower()
        },
        "original_content": {
            "title": post.title,
            "raw_selftext": post.selftext,
            "cleaned_selftext": cleaned_text
        },
        "llm_transformation": None
    }
    if rewritten_text and llm_model and prompt_template:
        record["llm_transformation"] = {
            "model_used": llm_model,
            "prompt_template": prompt_template,
            "rewritten_text": rewritten_text,
            "transformation_timestamp_utc": datetime.now(timezone.utc).isoformat()
        }
    return record

def create_bluesky_corpus_record(post, cleaned_text, rewritten_text=None, llm_model=None, prompt_template=None):
    """Structures collected Bluesky data into the official corpus JSON format.
   
     Keyword arguments:
    post -- The Bluesky post object containing details like URI, CID, and author.
    cleaned_text -- The cleaned version of the post's text.
    rewritten_text -- Optional; the text rewritten by an LLM.
    llm_model -- Optional; the name of the LLM model used for rewriting.
    prompt_template -- Optional; the prompt template used for rewriting.   
    
    Returns:
    A dictionary representing the corpus record, including source details, original content, and LLM transformation
    if applicable.
    """
    # Extract the post's unique record key (rkey) from its URI
    post_rkey = post.uri.split('/')[-1]
    
    record = {
        "corpus_item_id": str(uuid.uuid4()),
        "version": "1.1",
        "source_details": {
            "platform": "Bluesky",
            "post_uri": post.uri,
            "post_cid": str(post.cid),
            "author_did": post.author.did,
            "author_handle": post.author.handle
        },
        "original_content": {
            # Bluesky posts don't have titles, so we use the first 70 chars as a pseudo-title
            "title": (post.record.text[:70] + '...') if len(post.record.text) > 70 else post.record.text,
            "raw_text": post.record.text,
            "cleaned_text": cleaned_text
        },
        "llm_transformation": None
    }
    if rewritten_text and llm_model and prompt_template:
        record["llm_transformation"] = {
            "model_used": llm_model,
            "prompt_template": prompt_template,
            "rewritten_text": rewritten_text,
            "transformation_timestamp_utc": datetime.now(timezone.utc).isoformat()
        }
    return record

def save_record_to_corpus(record, directory, filename):
    """Saves a corpus record to a specified directory in JSON Lines format.
    
    Keyword arguments:
    record -- The corpus record dictionary to save.
    directory -- The directory where the record should be saved.
    filename -- The name of the file to save the record in.
    """
    try:
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    except IOError as e:
        print(f"Error: Could not write to file {filepath}. Details: {e}")
