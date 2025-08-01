"""Configuration settings for the Social Media Language Analysis Pipeline.

This module contains all configuration variables and settings used throughout
the application, including API limits, directory paths, and LLM settings.
All settings can be overridden using environment variables.
"""
import os
import random
from dotenv import load_dotenv

load_dotenv()

# --- SUBREDDIT SETTINGS ---
SUBREDDIT_LIST = [
    "TrueOffMyChest", "explainlikeimfive", "NoStupidQuestions",
    "CasualConversation", "self", "tifu", "AmItheAsshole",
    "LifeProTips", "TwoXChromosomes", "AskScience"
]


def get_target_subreddit():
    """Return a randomly selected subreddit from the configured list.
    
    Returns:
        str: A randomly chosen subreddit name from SUBREDDIT_LIST.
    """
    return random.choice(SUBREDDIT_LIST)

# --- GENERAL PIPELINE SETTINGS ---
NUM_POSTS_TO_COLLECT = int(os.getenv("NUM_POSTS_TO_COLLECT", 100))

# --- API & CLIENT SETTINGS ---
REDDIT_SAMPLE_LIMIT = int(os.getenv("REDDIT_SAMPLE_LIMIT", 300))
BLUESKY_SAMPLE_LIMIT = int(os.getenv("BLUESKY_SAMPLE_LIMIT", 100))
SLEEP_TIMER = float(os.getenv("SLEEP_TIMER", 4.5))

# --- LLM REWRITER SETTINGS ---
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
REWRITE_PROMPT_TEMPLATE = os.getenv(
    "REWRITE_PROMPT_TEMPLATE",
    "Please rewrite the following text. Do not add any commentary, just provide the rewritten text:\n\n---\n\n{text_to_rewrite}"
)

# --- DIRECTORY SETTINGS ---
ORIGINAL_ONLY_DIR = os.getenv("ORIGINAL_ONLY_DIR", "corpora/original_only")
REWRITTEN_PAIRS_DIR = os.getenv("REWRITTEN_PAIRS_DIR", "corpora/rewritten_pairs")
