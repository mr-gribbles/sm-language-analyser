import json
import pandas as pd
import argparse
from src.core_logic.corpus_analyzer import run_full_analysis

def analyze_corpus_file(filepath: str):
    """
    Loads a JSONL corpus file and runs analysis.
    - Intelligently finds the correct text field for both Reddit and Bluesky corpora.
    - If the file contains rewritten text, ONLY the rewritten text is analyzed.
    - If the file contains only original text, the original text is analyzed.
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
                    
                    rewritten_text = record.get("llm_transformation", {}).get("rewritten_text") if record.get("llm_transformation") else None
                    
                    if rewritten_text:
                        if i == 0: analysis_type = "Rewritten Text"
                        analysis_results = run_full_analysis(rewritten_text)
                    else:
                        if i == 0: analysis_type = "Original Text"
                        original_content = record.get("original_content", {})
                        
                        # --- FIX: Check for both possible key names ---
                        # This makes the script compatible with both Reddit and Bluesky files.
                        text_to_analyze = original_content.get("cleaned_selftext") or original_content.get("cleaned_text", "")
                        
                        analysis_results = run_full_analysis(text_to_analyze)

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
        
        output_csv_path = filepath.replace('.jsonl', '_analysis.csv')
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
