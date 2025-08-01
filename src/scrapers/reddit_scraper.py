"""Reddit scraper for collecting pure text posts.

This module provides functionality to scrape Reddit for high-quality text posts,
filtering out posts that contain images and returning random posts from
specified subreddits.
"""
import random
from src.clients.reddit_client import reddit_client

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp']


def get_random_text_post(subreddit_name, limit=100):
    """Fetch a random, pure text post from a given subreddit.

    Filters out posts that contain image links in their body and ensures
    the post has substantial text content.

    Args:
        subreddit_name: The name of the subreddit to scrape.
        limit: The maximum number of posts to fetch (default is 100).

    Returns:
        A random pure text post object if available, otherwise None.
        Returns None if no pure text posts are found or if the Reddit
        client is not initialized.
    """
    if not reddit_client:
        print("Reddit instance not available.")
        return None
    
    try:
        subreddit = reddit_client.subreddit(subreddit_name)
        print(f"Searching for pure text posts in r/{subreddit_name}...")
        hot_posts = subreddit.hot(limit=limit)
        
        # Filter out posts that contain images in their body
        # and ensure the post is a self post with non-empty selftext.
        # This will also exclude posts with very short text.
        pure_text_posts = [
            post for post in hot_posts 
            if post.is_self and post.selftext and not any(ext in post.selftext.lower() for ext in IMAGE_EXTENSIONS)
        ]
        
        if pure_text_posts:
            print(f"Found {len(pure_text_posts)} pure text posts. Selecting one at random.")
            return random.choice(pure_text_posts)
        else:
            print(f"No pure text posts found in the top {limit} posts of r/{subreddit_name}.")
            return None

    except Exception as e:
        print(f"An error occurred during scraping: {e}")
        return None
