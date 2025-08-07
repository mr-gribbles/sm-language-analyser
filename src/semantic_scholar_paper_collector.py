"""Academic paper collection pipeline for Semantic Scholar.

This module orchestrates the entire data collection process for academic papers
from Semantic Scholar, including scraping, cleaning, and corpus storage. It provides a
clean interface for collecting papers by field, query, or other criteria.
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
from src.clients.semantic_scholar_client import SemanticScholarClient


def run_semantic_scholar_collection_pipeline(
    field: Optional[str] = None,
    query: Optional[str] = None,
    max_papers: int = 50
) -> None:
    """Run the Semantic Scholar paper collection and processing pipeline.

    Args:
        field: Semantic Scholar field to search in (e.g., 'Psychology', 'Philosophy').
        query: Search query for papers.
        max_papers: Maximum number of papers to collect.
    """
    output_dir = config.ORIGINAL_ONLY_DIR
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    
    # Create descriptive filename
    search_term = _get_search_term(field, query)
    output_filename = f"semantic_scholar_{search_term}_original_{timestamp}.jsonl"

    print("Starting Semantic Scholar paper collection")
    if field:
        print(f"Field: {field}")
    if query:
        print(f"Query: {query}")
    print(f"Papers to collect: {max_papers}")

    _collect_semantic_scholar_papers(output_dir, output_filename, max_papers, field, query)

    print("Collection complete")


def _get_search_term(field: Optional[str], query: Optional[str]) -> str:
    """Generate a search term for the output filename.
    
    Args:
        field: Semantic Scholar field name.
        query: Search query string.
        
    Returns:
        Sanitized search term for filename.
    """
    if field:
        return field.lower().replace(" ", "_")
    elif query:
        return query.replace(" ", "_")
    else:
        return "psychology"  # default


def _collect_semantic_scholar_papers(
    output_dir: str,
    output_filename: str,
    max_papers: int,
    field: Optional[str] = None,
    query: Optional[str] = None
) -> None:
    """Collect papers from Semantic Scholar and save to corpus.

    Args:
        output_dir: Directory to save the collected data.
        output_filename: Name of the output file.
        max_papers: Number of papers to collect.
        field: Semantic Scholar field to search in.
        query: Search query for papers.
    """
    client = SemanticScholarClient(
        delay=config.SEMANTIC_SCHOLAR_DELAY,
        api_key=config.SEMANTIC_SCHOLAR_API_KEY
    )
    collected_ids = set()
    
    # Default to psychology papers if no field/query specified
    if not field and not query:
        field = "Psychology"
        print(f"No field or query specified, defaulting to field: '{field}'")
    
    print(f"Searching Semantic Scholar with field='{field}', query='{query}'")
    
    try:
        # Collect papers based on field or query
        if field and not query:
            papers_generator = client.collect_papers_by_field(
                field=field,
                max_papers=max_papers * 2  # Get extra in case some are filtered
            )
        elif query:
            papers_list = client.collect_papers_by_topic(
                topic=query,
                fields=[field] if field else None,
                max_papers=max_papers * 2
            )
            papers_generator = iter(papers_list)
        else:
            papers_generator = client.collect_papers_by_field(
                field="Psychology",
                max_papers=max_papers * 2
            )
        
        for paper_data in papers_generator:
            paper_id = paper_data['paper_id']
            
            if paper_id in collected_ids:
                continue
            
            # Extract and clean the main text content (abstract for Semantic Scholar)
            main_text = paper_data['abstract']
            
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
                  f"ID: {paper_data['paper_id']}")
            
            if len(collected_ids) >= max_papers:
                break
                
            time.sleep(config.SEMANTIC_SCHOLAR_DELAY)
            
    except Exception as e:
        print(f"Error in Semantic Scholar pipeline: {e}")
        raise
    
    if len(collected_ids) == 0:
        print("Warning: No papers were collected. Try adjusting your search field or query.")


def _create_paper_corpus_record(paper_data: dict, cleaned_text: str) -> dict:
    """Create a corpus record from Semantic Scholar paper data.
    
    Args:
        paper_data: Paper data from Semantic Scholar client.
        cleaned_text: Cleaned text content.
        
    Returns:
        Corpus record dictionary.
    """
    source_details = {
        "platform": "semantic_scholar",
        "paper_id": paper_data['paper_id'],
        "paper_url": paper_data['url'],
        "categories": paper_data['categories'],
        "fields_of_study": paper_data['fields_of_study'],
        "authors": paper_data['authors'],
        "published_year": paper_data['year'],
        "venue": paper_data['venue'],
        "citation_count": paper_data['citation_count'],
        "is_open_access": paper_data['is_open_access'],
    }
    
    original_content = {
        "title": paper_data['title'],
        "raw_text": paper_data['abstract'],  # Use abstract as main text
        "cleaned_text": cleaned_text,
        "abstract": paper_data['abstract'],
    }
    
    return create_corpus_record(
        source_details=source_details,
        original_content=original_content,
        rewritten_text=None,  # No rewriting in main pipeline
        llm_model=None,
        prompt_template=None,
    )
