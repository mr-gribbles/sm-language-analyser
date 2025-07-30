import argparse
from src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Social Media Language Analysis Pipeline")
    parser.add_argument("platform", choices=["reddit", "bluesky"], help="The platform to scrape data from.")
    parser.add_argument("--rewrite", action="store_true", help="Enable LLM rewriting of posts.")
    args = parser.parse_args()

    run_pipeline(platform=args.platform, rewrite=args.rewrite)

if __name__ == "__main__":
    main()
