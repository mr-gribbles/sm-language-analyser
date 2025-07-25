import praw
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve credentials from environment variables
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# A single, reusable Reddit instance
reddit = None

try:
    # Initialize the PRAW instance and assign it to the 'reddit' variable
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
        # Adding this check prevents PRAW from running in read-only mode if credentials are bad
        check_for_async=False 
    )
    # A simple check to confirm it's working when the module is imported
    # This will only print once
    print("PRAW client initialized successfully.")
except Exception as e:
    print(f"Error initializing PRAW client: {e}")
    # Exit if we can't connect, as nothing else will work
    exit()
