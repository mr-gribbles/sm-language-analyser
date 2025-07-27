import time
from datetime import datetime, timezone
import config
from src.core_logic.data_cleaner import clean_text
from src.core_logic.corpus_manager import create_bluesky_corpus_record, save_record_to_corpus
from src.scrapers.bluesky_scraper import fetch_bluesky_timeline_page

def main():
    """
    Main orchestration function to run the full, efficient Bluesky unedited post collection pipeline.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    output_filename = f"bluesky_original_{timestamp}.jsonl"
    
    print("--- Starting Bluesky Unedited Collection Pipeline ---")
    print(f"Posts to Collect: {config.NUM_POSTS_TO_COLLECT}")
    print(f"Output Directory: {config.ORIGINAL_ONLY_DIR}")
    print("-------------------------------------------")

    collected_uris = set()
    post_buffer = []
    cursor = None

    while len(collected_uris) < config.NUM_POSTS_TO_COLLECT:
        if not post_buffer:
            print("Post buffer is empty. Fetching a new page from the timeline...")
            new_posts, cursor = fetch_bluesky_timeline_page(limit=config.SAMPLE_LIMIT, cursor=cursor)
            
            if not new_posts:
                print("Failed to fetch new posts or reached the end of the timeline. Stopping.")
                break
            
            post_buffer.extend([p for p in new_posts if p.uri not in collected_uris])

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
