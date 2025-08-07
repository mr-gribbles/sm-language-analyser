"""Configuration settings for the Academic Paper Analysis Pipeline.

This module contains all configuration variables and settings used throughout
the application, including API limits, directory paths, and LLM settings.
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

# --- LLM REWRITER SETTINGS ---
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
REWRITE_PROMPT_TEMPLATE = os.getenv(
    "REWRITE_PROMPT_TEMPLATE",
    "Please rewrite the following academic text in a different style while maintaining the technical accuracy and meaning. Do not add any commentary, just provide the rewritten text:\n\n---\n\n{text_to_rewrite}"
)

# --- DIRECTORY SETTINGS ---
ORIGINAL_ONLY_DIR = os.getenv("ORIGINAL_ONLY_DIR", "corpora/original_only")
REWRITTEN_PAIRS_DIR = os.getenv("REWRITTEN_PAIRS_DIR", "corpora/rewritten_pairs")

# --- DEFAULT SEARCH SETTINGS ---
DEFAULT_ARXIV_QUERY = os.getenv("DEFAULT_ARXIV_QUERY", "machine learning")
DEFAULT_ARXIV_CATEGORY = os.getenv("DEFAULT_ARXIV_CATEGORY", None)
