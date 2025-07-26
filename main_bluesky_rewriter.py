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

    while len(collected_uris) < config.NUM_POSTS_TO_COLLECT:
        # If the buffer is empty, fetch a new page of posts from Bluesky
        if not post_buffer:
            print("Post buffer is empty. Fetching a new page from the timeline...")
            new_posts, cursor = fetch_bluesky_timeline_page(limit=config.SAMPLE_LIMIT, cursor=cursor)
            
            if not new_posts:
                print("Failed to fetch new posts or reached the end of the timeline. Stopping.")
                break
            
            # Add only new, unseen posts to our buffer
            post_buffer.extend([p for p in new_posts if p.uri not in collected_uris])

        # If the buffer is still empty after a fetch, wait and try again
        if not post_buffer:
            print("No new unique posts found on this page. Waiting before retrying...")
            time.sleep(20) # Wait longer if a page yields no new content
            continue

        # Process the next post from the front of the buffer
        post = post_buffer.pop(0)
        
        # This check is redundant but safe
        if post.uri in collected_uris:
            continue

        cleaned_text = clean_text(post.record.text)
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
