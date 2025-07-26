import random
from src.clients.reddit_client import reddit

def get_random_text_post(subreddit_name, limit=100):
    """
    Fetches a random, non-empty, pure text post from a given subreddit,
    filtering out posts that contain image links in their body.
    """
    if not reddit:
        print("Reddit instance not available.")
        return None

    try:
        subreddit = reddit.subreddit(subreddit_name)
        print(f"Searching for pure text posts in r/{subreddit_name}...")
        hot_posts = subreddit.hot(limit=limit)
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        pure_text_posts = [
            post for post in hot_posts 
            if post.is_self and post.selftext and not any(ext in post.selftext.lower() for ext in image_extensions)
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
