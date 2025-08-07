"""Main entry point for the Academic Paper Analysis Pipeline.

This module provides the command-line interface for running the data collection
and processing pipeline for academic papers from arXiv.
"""
import argparse
from src.pipeline import run_pipeline


def main():
    """Parse command-line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Academic Paper Analysis Pipeline"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Search query for papers (e.g., 'machine learning', 'neural networks')"
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        help="arXiv category (e.g., 'cs.AI', 'cs.CL', 'stat.ML')"
    )
    parser.add_argument(
        "--max-papers", "-n",
        type=int,
        default=50,
        help="Maximum number of papers to collect (default: 50)"
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Enable LLM rewriting of paper abstracts for AI vs Human classification."
    )
    args = parser.parse_args()

    run_pipeline(
        query=args.query,
        category=args.category,
        max_papers=args.max_papers,
        rewrite=args.rewrite
    )


if __name__ == "__main__":
    main()
