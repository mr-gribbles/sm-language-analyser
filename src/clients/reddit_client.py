"""Reddit client initialization and authentication.

This module handles the connection and authentication to the Reddit API using
PRAW (Python Reddit API Wrapper). It reads credentials from environment
variables and creates a single, shared client instance that can be imported
by other parts of the application.
"""
import os
import praw
from dotenv import load_dotenv

# Load environment variables from .env file at the module level
load_dotenv()

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")


def initialize_reddit_client():
    """Initialize and return an authenticated PRAW Reddit client.

    This function reads the client ID, client secret, and user agent from
    the environment variables and attempts to create an authenticated
    Reddit API client.

    Returns:
        An authenticated PRAW client instance if initialization is
        successful, otherwise None.
    """
    if not all([CLIENT_ID, CLIENT_SECRET, USER_AGENT]):
        print("ERROR: Reddit credentials not found in .env file.")
        return None
    try:
        client = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
            check_for_async=False
        )
        print("PRAW client initialized successfully.")
        return client
    except Exception as e:
        print(f"Failed to initialize Reddit client: {e}")
        return None

# Create a single, reusable client instance for other modules to import
reddit_client = initialize_reddit_client()
