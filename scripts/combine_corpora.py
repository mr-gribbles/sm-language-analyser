"""Combine multiple .jsonl files into a single file and optionally delete the originals.

This script searches for all .jsonl files in a specified directory, combines them into a single output file,
and then deletes the original source files if the user confirms.
It provides feedback on the number of files processed and the total number of records in the combined file.
It uses argparse for command-line interface.
It ensures that the output file is named based on the directory and current timestamp.
"""
import os
import glob
import json
import argparse
from datetime import datetime

def combine_jsonl_files(directory: str, delete_originals: bool = False):
    """Combines all .jsonl files in the specified directory into a single file.

    Keyword arguments:
    directory -- The path to the directory containing the .jsonl files to combine.
    delete_originals -- Whether to delete the original source files after combining.
    
    Returns:
    None. The combined results are saved to a new .jsonl file in the same directory
    """
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # Prepare the output filename based on the directory name and current timestamp
    dir_name = os.path.basename(os.path.normpath(directory))
    timestamp = datetime.now().strftime("%Y%m%d")
    output_filename = f"combined_{dir_name}_{timestamp}.jsonl"
    output_filepath = os.path.join(directory, output_filename)

    # Use glob to find all files ending with .jsonl in the target directory
    source_files = glob.glob(os.path.join(directory, '*.jsonl'))

    # Exclude the output file from the list of source files to prevent it from combining with itself
    if output_filepath in source_files:
        source_files.remove(output_filepath)

    if not source_files:
        print(f"No .jsonl files found in '{directory}'. Nothing to combine.")
        return

    print(f"Found {len(source_files)} files to combine in '{directory}'.")
    print(f"Output will be saved to: {output_filepath}")

    total_lines = 0
    seen_post_ids = set()
    
    try:
        # Open the single output file in write mode
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            # Iterate through each source file
            for filename in source_files:
                print(f"  -> Processing {os.path.basename(filename)}...")
                with open(filename, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        try:
                            post = json.loads(line)
                            post_id = post.get('corpus_item_id')

                            if post_id and post_id not in seen_post_ids:
                                outfile.write(line)
                                seen_post_ids.add(post_id)
                                total_lines += 1
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from line in {filename}: {line.strip()}")

        print("\n--- Combination Complete ---")
        print(f"Successfully combined {len(source_files)} files into '{output_filename}'.")
        print(f"The combined file contains {total_lines} records.")

        if delete_originals:
            print("Deleting original source files...")
            for filename in source_files:
                os.remove(filename)
                print(f"  -> Deleted {os.path.basename(filename)}")
            print("Original files deleted.")

    except IOError as e:
        print(f"\nAn error occurred during file operations: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    # Set up command-line argument parsing to get the target directory
    parser = argparse.ArgumentParser(description="Combine multiple .jsonl corpus files and delete the originals.")
    parser.add_argument("directory", type=str, help="The path to the directory containing the .jsonl files to combine.")
    parser.add_argument("--delete-originals", action="store_true", help="Delete the original source files after combining.")
    
    args = parser.parse_args()
    
    combine_jsonl_files(args.directory, args.delete_originals)
