import os
import glob
import argparse
from datetime import datetime

def combine_jsonl_files(directory: str):
    """
    Finds all .jsonl files in a specified directory, combines them
    into a single output file inside that same directory, and then
    deletes the original source files.

    Args:
        directory: The path to the directory containing the .jsonl files.
    """
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # Use glob to find all files ending with .jsonl in the target directory
    source_files = glob.glob(os.path.join(directory, '*.jsonl'))

    if not source_files:
        print(f"No .jsonl files found in '{directory}'. Nothing to combine.")
        return
    # Prepare the output filename based on the directory name and current timestamp
    dir_name = os.path.basename(os.path.normpath(directory))
    timestamp = datetime.now().strftime("%Y%m%d")
    output_filename = f"combined_{dir_name}_{timestamp}.jsonl"
    output_filepath = os.path.join(directory, output_filename)

    print(f"Found {len(source_files)} files to combine in '{directory}'.")
    print(f"Output will be saved to: {output_filepath}")

    total_lines = 0
    try:
        # Open the single output file in write mode
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            # Iterate through each source file
            for filename in source_files:
                print(f"  -> Processing {os.path.basename(filename)}...")
                with open(filename, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        # Write each line from the source file to the output file
                        outfile.write(line)
                        total_lines += 1
        
        print("\n--- Combination Complete ---")
        print(f"Successfully combined {len(source_files)} files into '{output_filename}'.")
        print(f"The combined file contains {total_lines} records.")

        # Optionally delete the original source files
        delete_originals = input("Do you want to delete the original source files? (y/n): ").strip().lower()
        if delete_originals == 'y':
            delete_originals = input("Are you sure? This cannot be undone. (y/n): ").strip().lower()
            if delete_originals == 'y':
                print("Deleting original source files...")
                for filename in source_files:
                    os.remove(filename)
                    print(f"  -> Deleted {os.path.basename(filename)}")
                print("Original files deleted.")
            else:
                print("Original files were not deleted.")
        else:
            print("Original files were not deleted.")

    except IOError as e:
        print(f"\nAn error occurred during file operations: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    # Set up command-line argument parsing to get the target directory
    parser = argparse.ArgumentParser(description="Combine multiple .jsonl corpus files and delete the originals.")
    parser.add_argument("directory", type=str, help="The path to the directory containing the .jsonl files to combine.")
    
    args = parser.parse_args()
    
    combine_jsonl_files(args.directory)
