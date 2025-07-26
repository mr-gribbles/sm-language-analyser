import random
from atproto_client.models.app.bsky.feed.post import Record
from atproto_client.models.app.bsky.embed.images import Main as EmbedImagesMain
from src.clients.bluesky_client import bluesky_client

WHATS_HOT_FEED_URI = "at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/whats-hot"

def fetch_bluesky_timeline_page(limit=100, cursor=None):
    """
    Fetches a single page of the "What's Hot" feed and applies advanced
    filtering to return only high-quality text posts.

    Args:
        limit: The number of posts to fetch per page (max 100).
        cursor: The cursor from the previous page fetch to get the next set of posts.

    Returns:
        A tuple containing (list_of_posts, next_cursor).
    """
    if not bluesky_client:
        print("Bluesky client is not available.")
        return [], None
    
    try:
        params = {'feed': WHATS_HOT_FEED_URI, 'limit': limit}
        if cursor:
            params['cursor'] = cursor
            
        response = bluesky_client.app.bsky.feed.get_feed(params=params)
        
        if not response or not response.feed:
            print("Could not fetch the Bluesky feed or reached the end.")
            return [], None
        
        high_quality_text_posts = []
        for item in response.feed:
            post = item.post
            
            # Basic check for a valid text post in English
            if not (isinstance(post.record, Record) and post.record.text and 'en' in (post.record.langs or [])):
                continue

            # Advanced filtering for higher quality
            text = post.record.text
            words = text.split()

            # 1. Filter out posts that have image embeds
            if post.embed and isinstance(post.embed, EmbedImagesMain):
                continue
            
            # 2. Filter out very short posts (e.g., less than 5 words)
            if len(words) < 5:
                continue
            
            # 3. Filter out posts that are mostly hashtags
            hashtags = [word for word in words if word.startswith('#')]
            # Avoid division by zero for posts with no words after splitting
            if len(words) > 0 and (len(hashtags) / len(words) > 0.5):
                continue
            
            # If all checks pass, it's a good candidate for our corpus
            high_quality_text_posts.append(post)
        
        return high_quality_text_posts, response.cursor

    except Exception as e:
        print(f"An error occurred while scraping Bluesky: {e}")
        return [], None