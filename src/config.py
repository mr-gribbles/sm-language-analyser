"""Configuration settings for the Academic Paper Analysis Pipeline.

This module contains all configuration variables and settings used throughout
the application, including API limits, directory paths, and search settings.
All settings can be overridden using environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- GENERAL PIPELINE SETTINGS ---
NUM_POSTS_TO_COLLECT = int(os.getenv("NUM_POSTS_TO_COLLECT", 100))
SLEEP_TIMER = float(os.getenv("SLEEP_TIMER", 3.0))  # arXiv recommends 3+ seconds

# --- ARXIV SETTINGS ---
ARXIV_DELAY = float(os.getenv("ARXIV_DELAY", 3.0))
ARXIV_MAX_RESULTS_PER_REQUEST = int(os.getenv("ARXIV_MAX_RESULTS_PER_REQUEST", 100))

# --- DIRECTORY SETTINGS ---
ORIGINAL_ONLY_DIR = os.getenv("ORIGINAL_ONLY_DIR", "corpora/original_only")

# --- DEFAULT SEARCH SETTINGS ---
DEFAULT_ARXIV_QUERY = os.getenv("DEFAULT_ARXIV_QUERY", "machine learning")
DEFAULT_ARXIV_CATEGORY = os.getenv("DEFAULT_ARXIV_CATEGORY", None)
