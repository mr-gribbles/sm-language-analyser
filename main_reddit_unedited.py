import time
from datetime import datetime, timezone
import config
from src.clients.reddit_client import reddit
from src.scrapers.reddit_scraper import get_random_text_post
from src.core_logic.data_cleaner import clean_text
from src.core_logic.corpus_manager import create_reddit_corpus_record, save_record_to_corpus

def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    output_filename = f"reddit_original_{timestamp}.jsonl"
    
    print("--- Starting Reddit Unedited Collection Pipeline ---")
    print(f"Posts to Collect: {config.NUM_POSTS_TO_COLLECT}")
    print(f"Output File: {config.ORIGINAL_ONLY_DIR}/{output_filename}")
    print("-------------------------------------------")

    collected_ids = set()

    while len(collected_ids) < config.NUM_POSTS_TO_COLLECT:
        target_subreddit = config.get_target_subreddit()
        post = get_random_text_post(target_subreddit, limit=config.SAMPLE_LIMIT)

        if post and post.id not in collected_ids:
            cleaned_text = clean_text(post.selftext)
            record = create_reddit_corpus_record(post, cleaned_text)
            save_record_to_corpus(record, config.ORIGINAL_ONLY_DIR, output_filename)
            
            collected_ids.add(post.id)
            print(f"Collected post {len(collected_ids)}/{config.NUM_POSTS_TO_COLLECT}. ID: {post.id} from r/{target_subreddit}")

        time.sleep(config.SLEEP_TIMER)

    print("\n--- Unedited Pipeline Complete ---")
    print(f"{len(collected_ids)} unique posts have been saved.")

if __name__ == "__main__":
    main()
