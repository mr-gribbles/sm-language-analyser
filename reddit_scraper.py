from reddit_client import reddit
import random

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

        # Fetch the hot posts from the subreddit
        hot_posts = subreddit.hot(limit=limit)
    
        # Define common image file extensions to check for in the post body
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']

        # Filter posts:
        # 1. Must be a text post (is_self).
        # 2. Must have content in its body (post.selftext).
        # 3. Must NOT contain any of the image extensions in its body text.
        pure_text_posts = [
            post for post in hot_posts 
            if post.is_self and post.selftext and not any(ext in post.selftext.lower() for ext in image_extensions)
        ]
        
        # Select a random post from the filtered list
        if pure_text_posts:
            print(f"Found {len(pure_text_posts)} pure text posts. Selecting one at random.")
            return random.choice(pure_text_posts)
        else:
            print(f"No pure text posts found in the top {limit} posts of r/{subreddit_name}.")
            return None

    except Exception as e:
        print(f"An error occurred during scraping: {e}")
        return None

# This block allows you to test this file directly
if __name__ == "__main__":
    # A good subreddit for finding text posts
    target_subreddit = "newzealand" 
    random_post = get_random_text_post(target_subreddit)
    
    if random_post:
        print("\n--- Test Fetch Successful ---")
        print(f"Title: {random_post.title}")
        print(f"Subreddit: r/{random_post.subreddit}")
        print(f"URL: {random_post.url}")