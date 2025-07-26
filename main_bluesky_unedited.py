import time
from datetime import datetime, timezone
import config
from data_cleaner import clean_text
from corpus_manager import create_bluesky_corpus_record, save_record_to_corpus
from bluesky_scraper import get_random_bluesky_post

def main():
    """
    Main orchestration function to run the full Bluesky unedited post collection pipeline.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    output_filename = f"bluesky_original_{timestamp}.jsonl"
    
    print("--- Starting Bluesky Unedited Collection Pipeline ---")
    print(f"Posts to Collect: {config.NUM_POSTS_TO_COLLECT}")
    print(f"Output Directory: {config.ORIGINAL_ONLY_DIR}")
    print("-------------------------------------------")

    collected_uris = set()

    while len(collected_uris) < config.NUM_POSTS_TO_COLLECT:
        # 1. Scrape: Get a random post from Bluesky
        post = get_random_bluesky_post(limit=config.SAMPLE_LIMIT)

        if post and post.uri not in collected_uris:
            # 2. Clean: Process the raw text
            cleaned_text = clean_text(post.record.text)
            
            # 3. Structure & Save: Create the record without any rewrite data
            record = create_bluesky_corpus_record(
                post=post,
                cleaned_text=cleaned_text
                # Note: We do not pass any llm_* arguments here
            )
            save_record_to_corpus(record, config.ORIGINAL_ONLY_DIR, output_filename)
            
            collected_uris.add(post.uri)
            print(f"Collected Post {len(collected_uris)}/{config.NUM_POSTS_TO_COLLECT}. URI: {post.uri}")

        # Be a polite API user
        time.sleep(config.SLEEP_TIMER)

    print("\n--- Bluesky Unedited Pipeline Complete ---")
    print(f"{len(collected_uris)} unique posts have been saved.")

if __name__ == "__main__":
    main()
