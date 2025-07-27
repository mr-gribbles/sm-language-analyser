import time
from datetime import datetime, timezone
import config

from src.core_logic.data_cleaner import clean_text
from src.core_logic.llm_rewriter import rewrite_text_with_gemini
from src.core_logic.corpus_manager import create_bluesky_corpus_record, save_record_to_corpus
from src.scrapers.bluesky_scraper import fetch_bluesky_timeline_page

def main():
    """
    Main orchestration function to run the full, efficient Bluesky rewriting pipeline.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    output_filename = f"bluesky_rewritten_{timestamp}.jsonl"
    
    print("--- Starting Bluesky Rewriter Collection Pipeline ---")
    print(f"Posts to Collect: {config.NUM_POSTS_TO_COLLECT}")
    print(f"Output Directory: {config.REWRITTEN_PAIRS_DIR}")
    print("-------------------------------------------")

    collected_uris = set()
    post_buffer = []
    cursor = None
    empty_fetch_attempts = 0

    while len(collected_uris) < config.NUM_POSTS_TO_COLLECT:
        if not post_buffer:
            print("Post buffer is empty. Fetching a new page from the timeline...")
            new_posts, cursor = fetch_bluesky_timeline_page(limit=config.SAMPLE_LIMIT, cursor=cursor)
            
            if not new_posts and cursor is None:
                print("Failed to fetch new posts and no cursor returned. Stopping.")
                break
            
            unique_new_posts = [p for p in new_posts if p.uri not in collected_uris]
            post_buffer.extend(unique_new_posts)

        if not post_buffer:
            empty_fetch_attempts += 1
            print(f"No new unique posts found on this page. Attempt {empty_fetch_attempts}/3.")
            if empty_fetch_attempts >= 3:
                print("Failed to find new posts after 3 attempts. Stopping pipeline.")
                break
            time.sleep(10) 
            continue
        
        empty_fetch_attempts = 0
        post = post_buffer.pop(0)
        
        if post.uri in collected_uris:
            continue

        try:
            cleaned_text = clean_text(post.record.text)
        except TypeError as e:
            print(f"Warning: Skipping post URI {post.uri} due to invalid text content. Error: {e}")
            continue

        rewritten_text = rewrite_text_with_gemini(
            text_to_rewrite=cleaned_text,
            model_name=config.LLM_MODEL,
            prompt_template=config.REWRITE_PROMPT_TEMPLATE
        )

        if rewritten_text:
            record = create_bluesky_corpus_record(
                post=post,
                cleaned_text=cleaned_text,
                rewritten_text=rewritten_text,
                llm_model=config.LLM_MODEL,
                prompt_template=config.REWRITE_PROMPT_TEMPLATE
            )
            save_record_to_corpus(record, config.REWRITTEN_PAIRS_DIR, output_filename)
            
            collected_uris.add(post.uri)
            print(f"Collected & Rewrote Post {len(collected_uris)}/{config.NUM_POSTS_TO_COLLECT}. URI: {post.uri}")

        time.sleep(config.SLEEP_TIMER)

    print("\n--- Bluesky Rewriter Pipeline Complete ---")
    print(f"{len(collected_uris)} unique posts have been saved.")

if __name__ == "__main__":
    main()
