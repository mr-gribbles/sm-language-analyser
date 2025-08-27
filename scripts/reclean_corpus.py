#!/usr/bin/env python3
"""
Re-cleans an existing JSONL corpus file using the updated data cleaner.

This script is designed to process a corpus file that has already been cleaned
once, applying the new, less aggressive cleaning logic to produce a new file
that preserves capitalization, punctuation, and emojis.
"""
import sys
import os
import json
import argparse
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core_logic.data_cleaner import clean_text

def reclean_corpus(input_file: str, output_file: str):
    """
    Reads a JSONL file, re-cleans the text, and writes to a new file.

    Args:
        input_file: Path to the input JSONL file.
        output_file: Path to the output JSONL file.
    """
    print(f"Starting re-cleaning process for '{input_file}'...")
    
    record_count = 0
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                record = json.loads(line.strip())
                
                # Determine which field to clean
                if 'original_content' in record and 'raw_text' in record['original_content']:
                    raw_text = record['original_content']['raw_text']
                    cleaned_text = clean_text(raw_text)
                    record['original_content']['cleaned_text'] = cleaned_text
                elif 'original_content' in record and 'raw_selftext' in record['original_content']:
                    raw_text = record['original_content']['raw_selftext']
                    cleaned_text = clean_text(raw_text)
                    record['original_content']['cleaned_selftext'] = cleaned_text
                
                outfile.write(json.dumps(record) + '\n')
                record_count += 1
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping malformed record: {e}")
                continue
    
    print(f"Successfully re-cleaned {record_count} records.")
    print(f"Output saved to '{output_file}'")

def main():
    """Main function to parse arguments and run the re-cleaning script."""
    parser = argparse.ArgumentParser(
        description='Re-clean a JSONL corpus file with updated cleaning logic.'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Path to the input JSONL corpus file.'
    )
    
    args = parser.parse_args()
    
    # Generate a new output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"combined_original_only_{timestamp}.jsonl"
    output_path = os.path.join(os.path.dirname(args.input_file), output_filename)
    
    reclean_corpus(args.input_file, output_path)

if __name__ == "__main__":
    main()
