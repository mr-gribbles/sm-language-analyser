import praw
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

def initialize_reddit_client():
    """
    Initializes and returns an authenticated PRAW Reddit client.
    """
    global reddit
    if not all([CLIENT_ID, CLIENT_SECRET, USER_AGENT]):
        print("ERROR: Reddit credentials not found in .env file.")
        return None
    
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
            check_for_async=False 
        )
        print("PRAW client initialized successfully.")
        return reddit
    except Exception as e:
        print(f"Failed to initialize Reddit client: {e}")
        return None

# Create a single, reusable client instance for other modules to import
reddit_client = initialize_reddit_client()