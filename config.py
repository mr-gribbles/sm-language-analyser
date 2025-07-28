import random

# --- SUBREDDIT SETTINGS ---
# A list of subreddits to randomly choose from for each post collection.
SUBREDDIT_LIST = [
    "TrueOffMyChest",
    "explainlikeimfive",
    "NoStupidQuestions",
    "CasualConversation",
    "self",
    "tifu",
    "AmItheAsshole",
    "TodayILearned"
]

def get_target_subreddit():
    """Returns a randomly selected subreddit from the list."""
    return random.choice(SUBREDDIT_LIST)

# --- GENERAL PIPELINE SETTINGS ---
# The total number of unique posts you want to collect in a single run.
NUM_POSTS_TO_COLLECT = 1000

# --- API & CLIENT SETTINGS ---
# The number of posts to fetch from Reddit in a single API call for sampling.
REDDIT_SAMPLE_LIMIT = 1000
BLUESKY_SAMPLE_LIMIT = 100
# A polite delay (in seconds) between API calls.
SLEEP_TIMER = 4.5

# --- LLM REWRITER SETTINGS ---
# The model to use for rewriting.
LLM_MODEL = "gemini-2.5-flash-lite"

# The prompt template that instructs the LLM on how to perform the rewrite.
REWRITE_PROMPT_TEMPLATE = (
    "Please rewrite the following text."
    "Do not add any commentary, just provide the rewritten text:\n\n---\n\n{text_to_rewrite}"
)

# --- DIRECTORY SETTINGS ---
# Directory for posts that are collected but NOT rewritten.
ORIGINAL_ONLY_DIR = "corpora/original_only"

# Directory for posts that ARE rewritten by the LLM.
REWRITTEN_PAIRS_DIR = "corpora/rewritten_pairs"
