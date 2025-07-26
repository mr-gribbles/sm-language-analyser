import time
from datetime import datetime, timezone
import config
from src.clients.reddit_client import reddit
from src.scrapers.reddit_scraper import get_random_text_post
from src.core_logic.data_cleaner import clean_text
from src.core_logic.llm_rewriter import rewrite_text_with_gemini
from src.core_logic.corpus_manager import create_reddit_corpus_record, save_record_to_corpus

def main():
    """
    Main orchestration function to run the full rewriting pipeline.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    output_filename = f"reddit_rewritten_{timestamp}.jsonl"
    
    print("--- Starting Reddit Rewriter Collection Pipeline ---")
    print(f"Posts to Collect: {config.NUM_POSTS_TO_COLLECT}")
    print(f"Output Directory: {config.REWRITTEN_PAIRS_DIR}")
    print("-------------------------------------------")

    collected_ids = set()

    while len(collected_ids) < config.NUM_POSTS_TO_COLLECT:
        target_subreddit = config.get_target_subreddit()
        post = get_random_text_post(target_subreddit, limit=config.SAMPLE_LIMIT)

        if post and post.id not in collected_ids:
            cleaned_text = clean_text(post.selftext)
            
            rewritten_text = rewrite_text_with_gemini(
                text_to_rewrite=cleaned_text,
                model_name=config.LLM_MODEL,
                prompt_template=config.REWRITE_PROMPT_TEMPLATE
            )

            if rewritten_text:
                record = create_reddit_corpus_record(
                    post=post,
                    cleaned_text=cleaned_text,
                    rewritten_text=rewritten_text,
                    llm_model=config.LLM_MODEL,
                    prompt_template=config.REWRITE_PROMPT_TEMPLATE
                )
                save_record_to_corpus(record, config.REWRITTEN_PAIRS_DIR, output_filename)
                
                collected_ids.add(post.id)
                print(f"Collected & Rewrote Post {len(collected_ids)}/{config.NUM_POSTS_TO_COLLECT}. ID: {post.id}")

        time.sleep(config.SLEEP_TIMER)

    print(f"\n--- Rewriter Pipeline Complete ---")
    print(f"{len(collected_ids)} unique posts have been saved.")

if __name__ == "__main__":
    main()
