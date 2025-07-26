import os
from atproto import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE")
BLUESKY_APP_PASSWORD = os.getenv("BLUESKY_APP_PASSWORD")

def initialize_bluesky_client():
    """
    Initializes and returns an authenticated atproto Client for Bluesky.
    """
    if not all([BLUESKY_HANDLE, BLUESKY_APP_PASSWORD]):
        print("ERROR: Bluesky credentials not found in .env file.")
        return None
    
    try:
        client = Client()
        client.login(BLUESKY_HANDLE, BLUESKY_APP_PASSWORD)
        print(f"Bluesky client initialized successfully. Authenticated as: {client.me.handle}")
        return client
    except Exception as e:
        print(f"Failed to initialize Bluesky client: {e}")
        return None

# Create a single, reusable client instance for other modules to import
bluesky_client = initialize_bluesky_client()
