import time
from datetime import datetime, timezone
import config
from src.scrapers.bluesky_scraper import get_random_bluesky_post
from src.core_logic.data_cleaner import clean_text
from src.core_logic.corpus_manager import create_bluesky_corpus_record, save_record_to_corpus

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
        post = get_random_bluesky_post(limit=config.SAMPLE_LIMIT)

        if post and post.uri not in collected_uris:
            cleaned_text = clean_text(post.record.text)
            
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
