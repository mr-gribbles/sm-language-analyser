"""Data collection and processing pipeline for social media platforms.

This module orchestrates the entire data collection process, including scraping,
cleaning, optional LLM rewriting, and corpus storage for both Reddit and
Bluesky platforms.
"""
import time
from datetime import datetime, timezone
from src import config
from src.core_logic.data_cleaner import clean_text
from src.core_logic.llm_rewriter import rewrite_text_with_gemini
from src.core_logic.corpus_manager import (
    create_corpus_record,
    save_record_to_corpus,
)
from src.scrapers.reddit_scraper import get_random_text_post
from src.scrapers.bluesky_scraper import fetch_bluesky_timeline_page


def run_pipeline(platform: str, rewrite: bool, num_posts: int = None,
                 reddit_limit: int = None, bluesky_limit: int = None):
    """Run the data collection and processing pipeline for the specified platform.

    Args:
        platform: The platform to scrape data from ('reddit' or 'bluesky').
        rewrite: Whether to rewrite the posts using an LLM.
        num_posts: The number of posts to collect. Defaults to config value.
        reddit_limit: The sample limit for Reddit. Defaults to config value.
        bluesky_limit: The sample limit for Bluesky. Defaults to config value.
    """
    num_posts_to_collect = num_posts or config.NUM_POSTS_TO_COLLECT
    if rewrite:
        output_dir = config.REWRITTEN_PAIRS_DIR
        pipeline_type = "Rewriter"
    else:
        output_dir = config.ORIGINAL_ONLY_DIR
        pipeline_type = "Unedited"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    output_filename = f"{platform}_{'rewritten' if rewrite else 'original'}_{timestamp}.jsonl"

    print(f"--- Starting {platform.capitalize()} {pipeline_type} Collection Pipeline ---")
    print(f"Posts to Collect: {num_posts_to_collect}")
    print(f"Output Directory: {output_dir}")
    print("-------------------------------------------")

    if platform == "reddit":
        sample_limit = reddit_limit or config.REDDIT_SAMPLE_LIMIT
        _run_reddit_pipeline(rewrite, output_dir, output_filename, num_posts_to_collect, sample_limit)
    elif platform == "bluesky":
        sample_limit = bluesky_limit or config.BLUESKY_SAMPLE_LIMIT
        _run_bluesky_pipeline(rewrite, output_dir, output_filename, num_posts_to_collect, sample_limit)

    print(f"\n--- {platform.capitalize()} {pipeline_type} Pipeline Complete ---")

def _run_reddit_pipeline(rewrite: bool, output_dir: str, output_filename: str,
                         num_posts_to_collect: int, sample_limit: int):
    """Run the Reddit-specific data collection pipeline.

    Args:
        rewrite: Whether to rewrite posts using LLM.
        output_dir: Directory to save the collected data.
        output_filename: Name of the output file.
        num_posts_to_collect: Number of posts to collect.
        sample_limit: Maximum number of posts to sample from each subreddit.
    """
    collected_ids = set()
    while len(collected_ids) < num_posts_to_collect:
        target_subreddit = config.get_target_subreddit()
        post = get_random_text_post(target_subreddit, limit=sample_limit)
        if post and post.id not in collected_ids:
            try:
                cleaned_text = clean_text(post.selftext)
            except TypeError as e:
                print(f"Warning: Skipping post ID {post.id} due to invalid text content. Error: {e}")
                continue

            rewritten_text = None
            if rewrite:
                rewritten_text = rewrite_text_with_gemini(
                    text_to_rewrite=cleaned_text,
                    model_name=config.LLM_MODEL,
                    prompt_template=config.REWRITE_PROMPT_TEMPLATE,
                )
            
            source_details = {
                "platform": "Reddit",
                "post_id": post.id,
                "post_url": f"https://www.reddit.com{post.permalink}",
                "subreddit": post.subreddit.display_name.lower(),
            }
            original_content = {
                "title": post.title,
                "raw_selftext": post.selftext,
                "cleaned_selftext": cleaned_text,
            }
            record = create_corpus_record(
                source_details=source_details,
                original_content=original_content,
                rewritten_text=rewritten_text,
                llm_model=config.LLM_MODEL if rewrite else None,
                prompt_template=config.REWRITE_PROMPT_TEMPLATE if rewrite else None,
            )
            save_record_to_corpus(record, output_dir, output_filename)
            collected_ids.add(post.id)
            print(f"Collected Post {len(collected_ids)}/{num_posts_to_collect}. ID: {post.id}")
        time.sleep(config.SLEEP_TIMER)

def _run_bluesky_pipeline(rewrite: bool, output_dir: str, output_filename: str,
                          num_posts_to_collect: int, sample_limit: int):
    """Run the Bluesky-specific data collection pipeline.

    Args:
        rewrite: Whether to rewrite posts using LLM.
        output_dir: Directory to save the collected data.
        output_filename: Name of the output file.
        num_posts_to_collect: Number of posts to collect.
        sample_limit: Maximum number of posts to sample from timeline.
    """
    collected_uris = set()
    post_buffer = []
    cursor = None
    empty_fetch_attempts = 0
    last_cursor = None
    while len(collected_uris) < num_posts_to_collect:
        if not post_buffer:
            new_posts, cursor = fetch_bluesky_timeline_page(limit=sample_limit, cursor=cursor)

            if not new_posts and cursor == last_cursor:
                empty_fetch_attempts += 1
                if empty_fetch_attempts >= 3:
                    print("No new posts found after multiple attempts with the same cursor. Stopping.")
                    break
            elif new_posts:
                empty_fetch_attempts = 0

            last_cursor = cursor

            if not new_posts and cursor is None:
                print("Failed to fetch new posts and no cursor returned. Stopping.")
                break
            
            unique_new_posts = [p for p in new_posts if p.uri not in collected_uris]
            post_buffer.extend(unique_new_posts)

        if not post_buffer:
            print("Post buffer is empty, fetching more...")
            time.sleep(config.SLEEP_TIMER)
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

        rewritten_text = None
        if rewrite:
            rewritten_text = rewrite_text_with_gemini(
                text_to_rewrite=cleaned_text,
                model_name=config.LLM_MODEL,
                prompt_template=config.REWRITE_PROMPT_TEMPLATE,
            )

        source_details = {
            "platform": "Bluesky",
            "post_uri": post.uri,
            "post_cid": str(post.cid),
            "author_did": post.author.did,
            "author_handle": post.author.handle,
        }
        original_content = {
            "title": (post.record.text[:70] + '...') if len(post.record.text) > 70 else post.record.text,
            "raw_text": post.record.text,
            "cleaned_text": cleaned_text,
        }
        record = create_corpus_record(
            source_details=source_details,
            original_content=original_content,
            rewritten_text=rewritten_text,
            llm_model=config.LLM_MODEL if rewrite else None,
            prompt_template=config.REWRITE_PROMPT_TEMPLATE if rewrite else None,
        )
        save_record_to_corpus(record, output_dir, output_filename)
        collected_uris.add(post.uri)
        print(f"Collected Post {len(collected_uris)}/{num_posts_to_collect}. URI: {post.uri}")
        time.sleep(config.SLEEP_TIMER)
