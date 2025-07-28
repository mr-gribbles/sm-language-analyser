"""Reddit Unedited Collection Pipeline

This script orchestrates the collection of original posts from Reddit.
It fetches posts from the user's timeline, cleans the text, and saves the results in a structured format.
It handles pagination, ensures unique posts are processed, and manages API rate limits.
It is designed to be efficient and robust, with error handling for common issues.
"""
import time
from datetime import datetime, timezone
import config

from src.clients.reddit_client import reddit
from src.scrapers.reddit_scraper import get_random_text_post
from src.core_logic.data_cleaner import clean_text
from src.core_logic.corpus_manager import create_reddit_corpus_record, save_record_to_corpus

def main():
    """Main function to run the Reddit Unedited Collection Pipeline."""
    # Prepare the output filename with a timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    output_filename = f"reddit_original_{timestamp}.jsonl"
    
    print("--- Starting Reddit Unedited Collection Pipeline ---")
    print(f"Posts to Collect: {config.NUM_POSTS_TO_COLLECT}")
    print(f"Output File: {config.ORIGINAL_ONLY_DIR}/{output_filename}")
    print("-------------------------------------------")

    collected_ids = set()
    # Collect posts until the desired number is reached
    while len(collected_ids) < config.NUM_POSTS_TO_COLLECT:
        target_subreddit = config.get_target_subreddit()
        post = get_random_text_post(target_subreddit, limit=config.REDDIT_SAMPLE_LIMIT)
        # If post is valid and not already collected
        if post and post.id not in collected_ids:
            try:
                cleaned_text = clean_text(post.selftext)
            except TypeError as e:
                print(f"Warning: Skipping post ID {post.id} due to invalid text content. Error: {e}")
                continue
            # Create a corpus record and save it
            record = create_reddit_corpus_record(post, cleaned_text)
            save_record_to_corpus(record, config.ORIGINAL_ONLY_DIR, output_filename)
            
            collected_ids.add(post.id)
            print(f"Collected post {len(collected_ids)}/{config.NUM_POSTS_TO_COLLECT}. ID: {post.id} from r/{target_subreddit}")

        time.sleep(config.SLEEP_TIMER)

    print("\n--- Unedited Pipeline Complete ---")
    print(f"{len(collected_ids)} unique posts have been saved.")

if __name__ == "__main__":
    main()
