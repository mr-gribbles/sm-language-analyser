"""Academic paper collection pipeline for arXiv.

This module orchestrates the entire data collection process for academic papers
from arXiv, including scraping, cleaning, and corpus storage. It provides a
clean interface for collecting papers by query, category, or other criteria.
"""
import time
from datetime import datetime, timezone
from typing import Optional

from src import config
from src.core_logic.data_cleaner import clean_text
from src.core_logic.corpus_manager import (
    create_corpus_record,
    save_record_to_corpus,
)
from src.clients.arxiv_client import ArXivClient


def run_paper_collection_pipeline(
    query: Optional[str] = None,
    category: Optional[str] = None,
    max_papers: int = 50
) -> None:
    """Run the academic paper collection and processing pipeline.

    Args:
        query: Search query for arXiv papers.
        category: arXiv category to search in.
        max_papers: Maximum number of papers to collect.
    """
    output_dir = config.ORIGINAL_ONLY_DIR
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    
    # Create descriptive filename
    search_term = _get_search_term(category, query)
    output_filename = f"arxiv_{search_term}_original_{timestamp}.jsonl"

    print(f"Starting arXiv paper collection")
    print(f"Query: {query}")
    print(f"Category: {category}")
    print(f"Papers to collect: {max_papers}")
    print(f"Output: {output_dir}/{output_filename}")

    _collect_arxiv_papers(output_dir, output_filename, max_papers, query, category)

    print(f"Collection complete")


def _get_search_term(category: Optional[str], query: Optional[str]) -> str:
    """Generate a search term for the output filename.
    
    Args:
        category: arXiv category code.
        query: Search query string.
        
    Returns:
        Sanitized search term for filename.
    """
    if category:
        return category.replace(".", "_")
    elif query:
        return query.replace(" ", "_")
    else:
        return "machine_learning"  # default


def _collect_arxiv_papers(
    output_dir: str,
    output_filename: str,
    max_papers: int,
    query: Optional[str] = None,
    category: Optional[str] = None
) -> None:
    """Collect papers from arXiv and save to corpus.

    Args:
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
                
            # Extract and clean the main text content
            main_text = paper_data['original_content']['cleaned_text']
            
            try:
                cleaned_text = clean_text(main_text)
            except TypeError as e:
                print(f"Warning: Skipping paper {paper_id} due to invalid text content. Error: {e}")
                continue
            
            # Create corpus record
            record = _create_paper_corpus_record(paper_data, cleaned_text)
            
            # Save to corpus
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


def _create_paper_corpus_record(paper_data: dict, cleaned_text: str) -> dict:
    """Create a corpus record from paper data.
    
    Args:
        paper_data: Paper data from arXiv client.
        cleaned_text: Cleaned text content.
        
    Returns:
        Corpus record dictionary.
    """
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
    
    return create_corpus_record(
        source_details=source_details,
        original_content=original_content,
        rewritten_text=None,  # No rewriting in main pipeline
        llm_model=None,
        prompt_template=None,
    )
