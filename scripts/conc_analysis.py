"""Advanced corpus analysis using the 'conc' package.

This script loads an original corpus and a rewritten corpus, then performs
a keyness analysis to identify which words are statistically more frequent
(i.e., "key") in the rewritten text. It provides a powerful way to quantify
the stylistic changes introduced by the LLM.
"""
import argparse
import json
import os
import shutil
import tempfile

import spacy
from conc.conc import Conc
from conc.core import get_stop_words
from conc.corpus import Corpus
from conc.keyness import Keyness
from spacy.cli import download as spacy_download


def ensure_spacy_model_installed(model="en_core_web_sm"):
    """Check if a spaCy model is installed and download it if not.
    
    Args:
        model: The name of the spaCy model to check/install.
    """
    try:
        spacy.load(model)
        print(f"spaCy model '{model}' already installed.")
    except OSError:
        print(f"spaCy model '{model}' not found. Downloading...")
        spacy_download(model)
        print(f"'{model}' downloaded successfully.")

# Ensure the required spaCy model is installed before proceeding
ensure_spacy_model_installed()

save_path = f'corpora/'
stop_words = get_stop_words(save_path = save_path)

def load_corpus_texts(filepath: str) -> list:
    """Load the cleaned text from a .jsonl corpus file into a list.
    
    Args:
        filepath: Path to the .jsonl corpus file.
        
    Returns:
        List of text strings extracted from the corpus file.
    """
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
    """Build a conc.Corpus object from a list of strings.
    
    Args:
        name: Name for the corpus.
        texts: List of text strings to include in the corpus.
        
    Returns:
        A conc.Corpus object built from the provided texts.
    """
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
    """Run the advanced analysis pipeline.
    
    Args:
        original_corpus_path: Path to the original corpus file.
        rewritten_corpus_path: Path to the rewritten corpus file.
    """
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
    
    print("\nGenerating frequency analysis...")
    original_freq_result = conc_original.frequencies(exclude_punctuation=True, page_current=1, normalize_by=1000, exclude_tokens=stop_words)
    original_freq_df = original_freq_result.to_frame().to_pandas()

    rewritten_freq_result = conc_rewritten.frequencies(exclude_punctuation=True, page_current=1, normalize_by=1000, exclude_tokens=stop_words)
    rewritten_freq_df = rewritten_freq_result.to_frame().to_pandas()

    # 3. Perform Keyness Analysis
    print("Performing keyness analysis...")
    keyness = Keyness(rewritten_corpus, original_corpus)
    keyness_result = keyness.keywords(show_document_frequency=True, statistical_significance_cut=0.0001, apply_bonferroni=True, order_descending=True, min_frequency_reference=1, page_current=1)
    keyness_df = keyness_result.to_frame().to_pandas()

    # 4. Generate HTML report
    print("Generating HTML report...")
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Concordance Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .section {{ margin: 30px 0; }}
        </style>
    </head>
    <body>
        <h1>Concordance Analysis Report</h1>
        
        <div class="section">
            <h2>Original Corpus Frequencies</h2>
            {original_freq_df.to_html(index=False, classes='table')}
        </div>
        
        <div class="section">
            <h2>Rewritten Corpus Frequencies</h2>
            {rewritten_freq_df.to_html(index=False, classes='table')}
        </div>
        
        <div class="section">
            <h2>Keyness Analysis - Top Keywords for Rewritten Corpus (vs. Original)</h2>
            {keyness_df.to_html(index=False, classes='table')}
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename based on input files
    original_name = os.path.basename(original_corpus_path).replace('.jsonl', '')
    rewritten_name = os.path.basename(rewritten_corpus_path).replace('.jsonl', '')
    html_filename = f"{original_name}_vs_{rewritten_name}_concordance_report.html"
    html_path = os.path.join(output_dir, html_filename)
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report saved to: {html_path}")
    print(f"📊 Report contains {len(original_freq_df)} original corpus frequencies")
    print(f"📊 Report contains {len(rewritten_freq_df)} rewritten corpus frequencies") 
    print(f"📊 Report contains {len(keyness_df)} keyness analysis results")
    print("🎉 Concordance analysis completed successfully!")
    print(f"💡 To view the report, download the file: {html_filename}")
def run_conc_analysis(original_corpus_path: str, rewritten_corpus_path: str):
    """Run the main conc analysis logic.
    
    Args:
        original_corpus_path: Path to the original corpus file.
        rewritten_corpus_path: Path to the rewritten corpus file.
    """
    main(original_corpus_path, rewritten_corpus_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform advanced analysis on two corpora using the 'conc' package.")
    parser.add_argument("original_corpus", type=str, help="Path to the original (unedited) .jsonl corpus file.")
    parser.add_argument("rewritten_corpus", type=str, help="Path to the rewritten .jsonl corpus file.")
    
    args = parser.parse_args()
    
    run_conc_analysis(args.original_corpus, args.rewritten_corpus)
