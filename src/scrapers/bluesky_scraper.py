"""This module provides a function to scrape the "What's Hot" feed from Bluesky.

It filters the posts to return only high-quality text posts, excluding those with images or low word counts.
The function `fetch_bluesky_timeline_page` fetches a single page of the feed and applies advanced filtering.
"""
import time
from atproto import exceptions
from atproto_client.models.app.bsky.feed.post import Record
from atproto_client.models.app.bsky.embed.images import Main as EmbedImagesMain

# Import the authenticated client instance
from src.clients.bluesky_client import bluesky_client

# Define the unique URI for the "What's Hot" feed.
WHATS_HOT_FEED_URI = "at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/whats-hot"

def fetch_bluesky_timeline_page(limit=100, cursor=None):
    """
    Fetches a single page of the "What's Hot" feed and applies advanced
    filtering to return only high-quality text posts.

    Keyword arguments:
    limit -- The maximum number of posts to fetch (default is 100).
    cursor -- Optional; used for pagination to fetch the next page of results.  

    Returns:
    A list of high-quality text posts that meet the filtering criteria.
    If no posts are found or an error occurs, returns an empty list and None for the cursor.    
    """
    if not bluesky_client:
        print("Bluesky client is not available.")
        return [], None
    
    max_retries = 3
    for attempt in range(max_retries):
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
                if not (isinstance(post.record, Record) and post.record.text and 'en' in (post.record.langs or [])):
                    continue
                if post.embed and isinstance(post.embed, EmbedImagesMain):
                    continue
                
                text = post.record.text
                words = text.split()
                if len(words) < 5:
                    continue
                
                hashtags = [word for word in words if word.startswith('#')]
                if len(words) > 0 and (len(hashtags) / len(words) > 0.5):
                    continue
                
                high_quality_text_posts.append(post)
            
            return high_quality_text_posts, response.cursor

        except exceptions.AtProtocolError as e:
            if e.response and e.response.status_code == 400:
                print("Fatal: Received a 400 Bad Request error from Bluesky. This indicates an issue with the request itself (e.g., invalid cursor).")
                print(f"Error details: {e.response.content}")
                return [], None  # Stop immediately, no retry
            
            # For other network/API errors, proceed with retry logic
            print(f"Warning: An error occurred while fetching from Bluesky. Attempt {attempt + 1}/{max_retries}.")
            print(f"Error details: {e}")
            if attempt + 1 < max_retries:
                wait_time = 30 * (attempt + 1)
                print(f"Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Failed to fetch from Bluesky.")
                return [], None
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred: {e}")
            return [], None
            
    return [], None
