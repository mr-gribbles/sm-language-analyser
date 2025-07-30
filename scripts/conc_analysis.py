"""Performs advanced corpus analysis using the 'conc' package.

This script loads an original corpus and a rewritten corpus, then performs
a keyness analysis to identify which words are statistically more frequent
(i.e., "key") in the rewritten text. It provides a powerful way to
quantify the stylistic changes introduced by the LLM.
"""
import json
import argparse
import os
import tempfile
import shutil
from conc.corpus import Corpus
from conc.conc import Conc
from conc.core import get_stop_words
from conc.keyness import Keyness

save_path = f'corpora/'
stop_words = get_stop_words(save_path = save_path)

def load_corpus_texts(filepath: str) -> list:
    """Loads the cleaned text from a .jsonl corpus file into a list."""
    texts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                text = record.get("original_content", {}).get("cleaned_selftext") or \
                       record.get("original_content", {}).get("cleaned_text", "")
                
                if record.get("llm_transformation"):
                    text = record.get("llm_transformation", {}).get("rewritten_text", "")
                
                if text:
                    texts.append(text)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An error occurred while loading {filepath}: {e}")
        return []
    return texts

def build_corpus_from_texts(name: str, texts: list) -> Corpus:
    """Builds a conc.Corpus object from a list of strings."""
    # Create a temporary directory to hold the text files
    temp_dir = tempfile.mkdtemp()
    
    print(f"Writing texts for '{name}' corpus to temporary files...")
    for i, text in enumerate(texts):
        with open(os.path.join(temp_dir, f"doc_{i}.txt"), 'w', encoding='utf-8') as f:
            f.write(text)
            
    # Create a new Corpus object and build it from the files
    corpus = Corpus(name)
    # The build method needs a path to save the final corpus. We'll make another temp dir for that.
    save_path = tempfile.mkdtemp()
    corpus.build_from_files(source_path=temp_dir, save_path=save_path)
    
    # Clean up the temporary directories
    shutil.rmtree(temp_dir)
    # The 'save_path' directory is kept as it contains the built corpus data
    print(f"Corpus '{name}' built successfully.")
    
    return corpus

def main(original_corpus_path: str, rewritten_corpus_path: str):
    """Main function to run the advanced analysis pipeline."""
    print("--- Starting Advanced Corpus Analysis using 'conc' ---")
    
    # 1. Load the text data from your corpus files
    print("Loading original corpus texts...")
    original_texts = load_corpus_texts(original_corpus_path)
    if not original_texts:
        return

    print("Loading rewritten corpus texts...")
    rewritten_texts = load_corpus_texts(rewritten_corpus_path)
    if not rewritten_texts:
        return

    # 2. Build the 'conc' Corpus objects from the text lists
    print("\nBuilding corpus objects...")
    original_corpus = build_corpus_from_texts('Original', original_texts)
    rewritten_corpus = build_corpus_from_texts('Rewritten', rewritten_texts)
    conc_original = Conc(original_corpus)
    conc_rewritten = Conc(rewritten_corpus)
    conc_original.frequencies(exclude_punctuation = True, page_current = 1, normalize_by=1000, exclude_tokens = stop_words).display()
    conc_rewritten.frequencies(exclude_punctuation = True, page_current = 1, normalize_by=1000, exclude_tokens = stop_words).display()
    # 3. Perform Keyness Analysis
    print("\nPerforming keyness analysis...")
    keyness = Keyness(rewritten_corpus, original_corpus)

    # 4. Display the results
    print("\n--- Top 20 Keywords for Rewritten Corpus (vs. Original) ---")
    keyness.keywords(show_document_frequency = True, statistical_significance_cut = 0.0001, apply_bonferroni = True, order_descending = True, min_frequency_reference = 1, page_current = 1).display()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform advanced analysis on two corpora using the 'conc' package.")
    parser.add_argument("original_corpus", type=str, help="Path to the original (unedited) .jsonl corpus file.")
    parser.add_argument("rewritten_corpus", type=str, help="Path to the rewritten .jsonl corpus file.")
    
    args = parser.parse_args()
    
    main(args.original_corpus, args.rewritten_corpus)