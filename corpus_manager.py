# corpus_manager.py
import os
import json
import uuid

def create_corpus_record(post, cleaned_text):
    """
    Structures the collected data into the official corpus JSON format.
    
    Args:
        post: The PRAW Submission object from the scraper.
        cleaned_text: The cleaned version of the post's selftext.

    Returns:
        A dictionary representing the structured corpus record.
    """
    return {
        "corpus_item_id": str(uuid.uuid4()),
        "version": "1.0",
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

def save_record_to_corpus(record, directory, filename):
    """
    Appends a single JSON record to the specified JSONL file.
    Creates the directory and file if they don't exist.
    
    Args:
        record: The dictionary to save.
        directory: The folder to save the file in.
        filename: The name of the file.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        
        # Open the file in append mode and write the record as a new line
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
            
    except IOError as e:
        print(f"Error: Could not write to file {filepath}. Details: {e}")
