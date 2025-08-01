"""Bluesky client initialization and authentication.

This module handles the connection and authentication to the Bluesky API using
atproto. It reads credentials from environment variables, logs in, and creates
a single, shared client instance that can be imported by other parts of the
application.
"""
import os
from atproto import Client
from dotenv import load_dotenv

# Load environment variables from .env file at the module level
load_dotenv()

BLUESKY_USERNAME = os.getenv("BLUESKY_USERNAME")
BLUESKY_PASSWORD = os.getenv("BLUESKY_PASSWORD")


def initialize_bluesky_client():
    """Initialize and return an authenticated atproto Client for Bluesky.

    Returns:
        An authenticated client instance if login is successful,
        otherwise None.
    """

    if not all([BLUESKY_USERNAME, BLUESKY_PASSWORD]):
        print("ERROR: Bluesky credentials not found in .env file.")
        return None
    
    try:
        client = Client()
        client.login(BLUESKY_USERNAME, BLUESKY_PASSWORD)
        print(f"Bluesky client initialized successfully. Authenticated as: {client.me.handle}")
        return client
    except Exception as e:
        print(f"Failed to initialize Bluesky client due to an error: {e}")
        return None

# Create a single, reusable client instance for other modules to import
bluesky_client = initialize_bluesky_client()
