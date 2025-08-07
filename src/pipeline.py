"""Data collection and processing pipeline for academic papers.

This module orchestrates the entire data collection process for academic papers
from arXiv, including scraping, cleaning, optional LLM rewriting, and corpus storage.
"""
import time
from datetime import datetime, timezone
from src import config
from src.core_logic.data_cleaner import clean_text
from src.core_logic.llm_rewriter import rewrite_text_with_gemini
from src.core_logic.corpus_manager import (
    create_corpus_record,
    save_record_to_corpus,
)
from src.clients.arxiv_client import ArXivClient


def run_pipeline(query: str = None, category: str = None, max_papers: int = 50, rewrite: bool = False):
    """Run the academic paper collection and processing pipeline.

    Args:
        query: Search query for arXiv papers.
        category: arXiv category to search in.
        max_papers: Maximum number of papers to collect.
        rewrite: Whether to rewrite the papers using an LLM.
    """
    if rewrite:
        output_dir = config.REWRITTEN_PAIRS_DIR
        pipeline_type = "Rewriter"
    else:
        output_dir = config.ORIGINAL_ONLY_DIR
        pipeline_type = "Original"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    
    # Create descriptive filename
    if category:
        search_term = category.replace(".", "_")
    elif query:
        search_term = query.replace(" ", "_")
    else:
        search_term = "machine_learning"  # default
    
    output_filename = f"arxiv_{search_term}_{'rewritten' if rewrite else 'original'}_{timestamp}.jsonl"

    print(f"--- Starting arXiv {pipeline_type} Collection Pipeline ---")
    print(f"Query: {query}")
    print(f"Category: {category}")
    print(f"Papers to Collect: {max_papers}")
    print(f"Output Directory: {output_dir}")
    print(f"Output File: {output_filename}")
    print("-------------------------------------------")

    _run_arxiv_pipeline(rewrite, output_dir, output_filename, max_papers, query, category)

    print(f"\n--- arXiv {pipeline_type} Pipeline Complete ---")


def _run_arxiv_pipeline(rewrite: bool, output_dir: str, output_filename: str,
                       max_papers: int, query: str = None, category: str = None):
    """Run the arXiv-specific data collection pipeline.

    Args:
        rewrite: Whether to rewrite papers using LLM.
        output_dir: Directory to save the collected data.
        output_filename: Name of the output file.
        max_papers: Number of papers to collect.
        query: Search query for papers.
        category: arXiv category to search in.
    """
    client = ArXivClient(delay=3.0)  # Respect arXiv's rate limits
    collected_ids = set()
    
    # Default to machine learning papers if no query/category specified
    if not query and not category:
        query = "machine learning"
        print(f"No query or category specified, defaulting to: '{query}'")
    
    print(f"Searching arXiv with query='{query}', category='{category}'")
    
    try:
        for paper_data in client.collect_papers(
            query=query,
            category=category,
            max_papers=max_papers * 2  # Get extra in case some are filtered
        ):
            paper_id = paper_data['id']
            
            if paper_id in collected_ids:
                continue
                
            # Extract the main text content
            main_text = paper_data['original_content']['cleaned_text']
            
            try:
                # Clean the text (though it's already cleaned by the scraper)
                cleaned_text = clean_text(main_text)
            except TypeError as e:
                print(f"Warning: Skipping paper {paper_id} due to invalid text content. Error: {e}")
                continue

            rewritten_text = None
            if rewrite:
                rewritten_text = rewrite_text_with_gemini(
                    text_to_rewrite=cleaned_text,
                    model_name=config.LLM_MODEL,
                    prompt_template=config.REWRITE_PROMPT_TEMPLATE,
                )
            
            # Convert to the expected corpus format
            source_details = {
                "platform": "arXiv",
                "paper_id": paper_data['original_content']['paper_id'],
                "paper_url": paper_data['original_content']['url'],
                "categories": paper_data['original_content']['categories'],
                "authors": paper_data['original_content']['authors'],
                "published_date": paper_data['original_content']['published_date'],
            }
            
            original_content = {
                "title": paper_data['original_content']['title'],
                "raw_text": paper_data['original_content']['raw_text'],
                "cleaned_text": cleaned_text,
                "abstract": paper_data['original_content']['abstract'],
            }
            
            record = create_corpus_record(
                source_details=source_details,
                original_content=original_content,
                rewritten_text=rewritten_text,
                llm_model=config.LLM_MODEL if rewrite else None,
                prompt_template=config.REWRITE_PROMPT_TEMPLATE if rewrite else None,
            )
            
            save_record_to_corpus(record, output_dir, output_filename)
            collected_ids.add(paper_id)
            
            print(f"Collected Paper {len(collected_ids)}/{max_papers}. "
                  f"ID: {paper_data['original_content']['paper_id']}")
            
            if len(collected_ids) >= max_papers:
                break
                
            time.sleep(config.SLEEP_TIMER)
            
    except Exception as e:
        print(f"Error in arXiv pipeline: {e}")
        raise
    
    if len(collected_ids) == 0:
        print("Warning: No papers were collected. Try adjusting your search query or category.")
