"""Analyze and save a text corpus file for linguistic metrics.

This script intelligently handles both original and rewritten text from Reddit and Bluesky corpora.
It performs the following:
- Loads a JSONL corpus file.
- Analyzes the text for readability, lexical diversity, and sentiment.
- Saves the results to a CSV file.
- Provides a summary of the analysis results.
- Handles errors gracefully and logs warnings for malformed lines.
- Uses argparse for command-line interface.
It ensures that the analysis is performed on the correct text field, whether it is original or rewritten text.
It also includes error handling for file not found and unexpected errors.
"""
import json
import pandas as pd
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core_logic.corpus_analyzer import run_full_analysis

def analyze_corpus_file(filepath: str):
    """
    Analyzes a corpus file for linguistic metrics and saves the results to a CSV file.

    Keyword arguments:
    filepath -- The path to the .jsonl corpus file to analyze.  

    Returns:
    None. The results are saved to a CSV file with the same name as the input file
    """
    if not filepath.endswith('.jsonl'):
        print("Error: Please provide a valid .jsonl file.")
        return

    results = []
    analysis_type = "Unknown"
    print(f"Analyzing corpus file: {filepath}...")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    record = json.loads(line)
                    
                    text_to_analyze = None
                    if record.get("llm_transformation") and record["llm_transformation"].get("rewritten_text"):
                        text_to_analyze = record["llm_transformation"]["rewritten_text"]
                        if i == 0: analysis_type = "Rewritten Text"
                    else:
                        original_content = record.get("original_content", {})
                        text_to_analyze = original_content.get("cleaned_selftext") or original_content.get("cleaned_text")
                        if i == 0: analysis_type = "Original Text"

                    analysis_results = run_full_analysis(text_to_analyze) if text_to_analyze else {}

                    row = {
                        "corpus_item_id": record.get("corpus_item_id"),
                        **analysis_results
                    }
                    results.append(row)
                
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line in {filepath}")
                    continue
        
        df = pd.DataFrame(results)
        
        print(f"\n--- Corpus Analysis Summary ({analysis_type}) ---")
        numeric_df = df.select_dtypes(include='number')
        summary = numeric_df.mean().round(3)
        print(summary.to_string())
        
        # Create the analysis_results directory if it doesn't exist
        output_dir = 'analysis_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a clean filename and save the CSV in the new directory
        base_filename = os.path.basename(filepath).replace('.jsonl', '_analysis.csv')
        output_csv_path = os.path.join(output_dir, base_filename)
        
        df.to_csv(output_csv_path, index=False)
        print(f"\nDetailed analysis saved to: {output_csv_path}")

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a text corpus file for linguistic metrics.")
    parser.add_argument("filepath", type=str, help="The path to the .jsonl corpus file to analyze.")
    
    args = parser.parse_args()
    
    analyze_corpus_file(args.filepath)
