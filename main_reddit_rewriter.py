import time
from datetime import datetime, timezone
import config

# Import modular functions
from reddit_client import reddit
from reddit_scraper import get_random_text_post
from data_cleaner import clean_text
from llm_rewriter import rewrite_text_with_gemini
from corpus_manager import create_corpus_record, save_record_to_corpus

def main():
    """
    Main orchestration function to run the full rewriting pipeline.
    """
    # Generate a unique filename for this specific run
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    # Note: We get the subreddit for the filename AFTER we successfully fetch a post
    
    print("--- Starting Rewriter Collection Pipeline ---")
    print(f"Posts to Collect: {config.NUM_POSTS_TO_COLLECT}")
    print(f"Output Directory: {config.REWRITTEN_PAIRS_DIR}")
    print("-------------------------------------------")

    collected_ids = set()
    output_filename = None # To be defined after the first post is fetched

    while len(collected_ids) < config.NUM_POSTS_TO_COLLECT:
        # 1. Scrape: Get a random subreddit and then a random post
        target_subreddit = config.get_target_subreddit()
        post = get_random_text_post(target_subreddit, limit=config.SAMPLE_LIMIT)

        if post and post.id not in collected_ids:
            # 2. Clean: Process the raw text
            cleaned_text = clean_text(post.selftext)
            
            # 3. Rewrite: Send the cleaned text to the LLM
            rewritten_text = rewrite_text_with_gemini(
                text_to_rewrite=cleaned_text,
                model_name=config.LLM_MODEL,
                prompt_template=config.REWRITE_PROMPT_TEMPLATE
            )

            # Only proceed if the rewrite was successful
            if rewritten_text:
                # Set the filename on the first successful collection
                if not output_filename:
                    output_filename = f"reddit_rewritten_{timestamp}.jsonl"

                # 4. Structure & Save: Create the record with all data
                record = create_corpus_record(
                    post=post,
                    cleaned_text=cleaned_text,
                    rewritten_text=rewritten_text,
                    llm_model=config.LLM_MODEL,
                    prompt_template=config.REWRITE_PROMPT_TEMPLATE
                )
                save_record_to_corpus(record, config.REWRITTEN_PAIRS_DIR, output_filename)
                
                # Update progress
                collected_ids.add(post.id)
                print(f"Collected & Rewrote Post {len(collected_ids)}/{config.NUM_POSTS_TO_COLLECT}. ID: {post.id}")

        # Be a polite API user
        time.sleep(config.SLEEP_TIMER)

    print("\n--- Rewriter Pipeline Complete ---")
    print(f"{len(collected_ids)} unique posts have been saved to {config.REWRITTEN_PAIRS_DIR}/{output_filename}.")

if __name__ == "__main__":
    main()
