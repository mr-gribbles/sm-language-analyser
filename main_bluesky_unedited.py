"""Bluesky Unedited Post Collection Pipeline

This script orchestrates the collection of unedited posts from Bluesky.
It fetches posts from the user's timeline, cleans the text, and saves the results in a structured format.
It handles pagination, ensures unique posts are processed, and manages API rate limits.
It is designed to be efficient and robust, with error handling for common issues.
"""
import time
from datetime import datetime, timezone
import config
from src.core_logic.data_cleaner import clean_text
from src.core_logic.corpus_manager import create_bluesky_corpus_record, save_record_to_corpus
from src.scrapers.bluesky_scraper import fetch_bluesky_timeline_page

def main():
    """Main function to run the Bluesky Unedited Collection Pipeline."""
    # Prepare the output filename with a timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    output_filename = f"bluesky_original_{timestamp}.jsonl"
    
    print("--- Starting Bluesky Unedited Collection Pipeline ---")
    print(f"Posts to Collect: {config.NUM_POSTS_TO_COLLECT}")
    print(f"Output Directory: {config.ORIGINAL_ONLY_DIR}")
    print("-------------------------------------------")

    collected_uris = set()
    post_buffer = []
    cursor = None
    # Collect posts until the desired number is reached
    # This loop will continue until we have collected the required number of unique posts
    while len(collected_uris) < config.NUM_POSTS_TO_COLLECT:
        if not post_buffer:
            print("Post buffer is empty. Fetching a new page from the timeline...")
            new_posts, cursor = fetch_bluesky_timeline_page(limit=config.BLUESKY_SAMPLE_LIMIT, cursor=cursor)
            if not new_posts:
                print("Failed to fetch new posts or reached the end of the timeline. Stopping.")
                break
            post_buffer.extend([p for p in new_posts if p.uri not in collected_uris])
        # If no new posts were found, wait before retrying
        if not post_buffer:
            print("No new unique posts found on this page. Waiting before retrying...")
            time.sleep(20)
            continue
        post = post_buffer.pop(0)
        if post.uri in collected_uris:
            continue
        try:
            cleaned_text = clean_text(post.record.text)
        except TypeError as e:
            print(f"Warning: Skipping post URI {post.uri} due to invalid text content. Error: {e}")
            continue
        # Create a corpus record and save it
        record = create_bluesky_corpus_record(
            post=post,
            cleaned_text=cleaned_text
        )
        save_record_to_corpus(record, config.ORIGINAL_ONLY_DIR, output_filename)
        collected_uris.add(post.uri)
        print(f"Collected Post {len(collected_uris)}/{config.NUM_POSTS_TO_COLLECT}. URI: {post.uri}")
        time.sleep(config.SLEEP_TIMER)

    print("\n--- Bluesky Unedited Pipeline Complete ---")
    print(f"{len(collected_uris)} unique posts have been saved.")

if __name__ == "__main__":
    main()
