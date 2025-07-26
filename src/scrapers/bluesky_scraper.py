import random
from atproto_client.models.app.bsky.feed.post import Record

# Import the authenticated client instance
from src.clients.bluesky_client import bluesky_client

def get_random_bluesky_post(limit=100):
    """
    Fetches a list of recent text posts from the Bluesky "what's hot" feed
    and returns a random one.
    """
    if not bluesky_client:
        print("Bluesky client is not available.")
        return None
    
    try:
        # Fetch the "what's hot" timeline.
        # For atproto v0.0.61+, parameters must be passed in a `params` dictionary.
        response = bluesky_client.app.bsky.feed.get_timeline(
            params={'algorithm': 'whats-hot', 'limit': limit}
        )
        
        if not response or not response.feed:
            print("Could not fetch the Bluesky timeline.")
            return None
        
        # Filter for posts that are in English and have text content
        text_posts = [
            item.post for item in response.feed
            if isinstance(item.post.record, Record) and item.post.record.text and 'en' in (item.post.record.langs or [])
        ]

        if text_posts:
            return random.choice(text_posts)
        else:
            print(f"No English text posts found in the last {limit} timeline items.")
            return None

    except Exception as e:
        print(f"An error occurred while scraping Bluesky: {e}")
        return None
