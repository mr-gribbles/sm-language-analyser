import os
import json
import uuid
from datetime import datetime, timezone

# --- Reddit Record Creator (No Changes) ---
def create_reddit_corpus_record(post, cleaned_text, rewritten_text=None, llm_model=None, prompt_template=None):
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

# --- Bluesky Record Creator ---
def create_bluesky_corpus_record(post, cleaned_text, rewritten_text=None, llm_model=None, prompt_template=None):
    """
    Structures collected Bluesky data into the official corpus JSON format.
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

# --- Save Function ---
def save_record_to_corpus(record, directory, filename):
    try:
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    except IOError as e:
        print(f"Error: Could not write to file {filepath}. Details: {e}")
