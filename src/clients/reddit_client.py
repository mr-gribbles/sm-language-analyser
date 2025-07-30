"""Initializes and provides a reusable, authenticated Reddit client.

This module is responsible for handling the connection and authentication to
the Reddit API using PRAW (Python Reddit API Wrapper). It reads credentials
from environment variables, logs in, and creates a single, shared client
instance that can be imported by other parts of the application, such as the
scraper module.

"""

import praw
import os
from dotenv import load_dotenv

# Load environment variables from .env file at the module level
load_dotenv()

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

def initialize_reddit_client():
    """
    Initializes and returns an authenticated PRAW Reddit client.
    This function reads the client ID, client secret, and user agent from
    the environment variables. It then attempts to log in to the Reddit API.
    If successful, it returns an authenticated client instance.
    Returns:
        praw.Reddit | None: An authenticated PRAW client instance if login is
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
