#!/usr/bin/env python3
"""
Training script for interpretable AI vs Human text classifiers.

This script uses the rewritten, stable InterpretableTextClassifier to compare
models and analyze the most influential language features.
"""
import sys
import os
import argparse

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.interpretable_classifiers import compare_and_analyze

def main():
    """Main function to run the comparison and analysis."""
    parser = argparse.ArgumentParser(
        description='Train and compare interpretable AI vs Human text classifiers.'
    )
    parser.add_argument(
        '--human-file', 
        type=str, 
        required=True,
        help='Path to JSONL file containing human-written texts.'
    )
    parser.add_argument(
        '--ai-file', 
        type=str, 
        required=True,
        help='Path to JSONL file containing AI-generated texts.'
    )
    args = parser.parse_args()

    # Validate file paths
    if not os.path.exists(args.human_file):
        print(f"Error: Human text file not found at '{args.human_file}'")
        sys.exit(1)
    if not os.path.exists(args.ai_file):
        print(f"Error: AI text file not found at '{args.ai_file}'")
        sys.exit(1)

    # Run the comparison and analysis
    compare_and_analyze(args.human_file, args.ai_file)

if __name__ == "__main__":
    main()
