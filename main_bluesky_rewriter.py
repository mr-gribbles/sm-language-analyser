import time
from datetime import datetime, timezone
import config
from data_cleaner import clean_text
from llm_rewriter import rewrite_text_with_gemini
from corpus_manager import create_bluesky_corpus_record, save_record_to_corpus
from bluesky_scraper import get_random_bluesky_post

def main():
    """
    Main orchestration function to run the full Bluesky rewriting pipeline.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    output_filename = f"bluesky_rewritten_{timestamp}.jsonl"
    
    print("--- Starting Bluesky Rewriter Collection Pipeline ---")
    print(f"Posts to Collect: {config.NUM_POSTS_TO_COLLECT}")
    print(f"Output Directory: {config.REWRITTEN_PAIRS_DIR}")
    print("-------------------------------------------")

    collected_uris = set()

    while len(collected_uris) < config.NUM_POSTS_TO_COLLECT:
        # 1. Scrape: Get a random post from Bluesky
        post = get_random_bluesky_post(limit=config.SAMPLE_LIMIT) # Using same sample limit config

        if post and post.uri not in collected_uris:
            # 2. Clean: Process the raw text
            cleaned_text = clean_text(post.record.text)
            
            # 3. Rewrite: Send the cleaned text to the LLM
            rewritten_text = rewrite_text_with_gemini(
                text_to_rewrite=cleaned_text,
                model_name=config.LLM_MODEL,
                prompt_template=config.REWRITE_PROMPT_TEMPLATE
            )

            if rewritten_text:
                # 4. Structure & Save
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
