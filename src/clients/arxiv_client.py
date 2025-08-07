"""ArXiv client for academic paper data collection.

This module provides a client interface for collecting academic papers from arXiv
that integrates with the existing corpus management system.
"""

import logging
from typing import Dict, List, Optional, Generator
from datetime import datetime

from ..scrapers.arxiv_scraper import ArXivScraper

logger = logging.getLogger(__name__)


class ArXivClient:
    """Client for collecting academic papers from arXiv."""
    
    def __init__(self, delay: float = 3.0):
        """Initialize the ArXiv client.
        
        Args:
            delay: Delay between API requests in seconds.
        """
        self.scraper = ArXivScraper(delay=delay)
        self.source_name = "arxiv"
    
    def collect_papers(self, query: str = None, category: str = None,
                      max_papers: int = 100, start_date: str = None,
                      end_date: str = None) -> Generator[Dict, None, None]:
        """Collect academic papers from arXiv.
        
        Args:
            query: Search query for papers.
            category: arXiv category to search in.
            max_papers: Maximum number of papers to collect.
            start_date: Start date for paper search (YYYY-MM-DD).
            end_date: End date for paper search (YYYY-MM-DD).
            
        Yields:
            Dictionary containing paper data in corpus format.
        """
        logger.info(f"Starting arXiv paper collection")
        logger.info(f"Query: {query}, Category: {category}, Max papers: {max_papers}")
        
        collected_count = 0
        
        try:
            for paper in self.scraper.search_papers(
                query=query,
                category=category,
                max_results=max_papers,
                start_date=start_date,
                end_date=end_date
            ):
                # Convert to corpus format
                corpus_entry = self._convert_to_corpus_format(paper)
                
                if corpus_entry:
                    collected_count += 1
                    logger.debug(f"Collected paper {collected_count}: {paper['title'][:50]}...")
                    yield corpus_entry
                
                if collected_count >= max_papers:
                    break
                    
        except Exception as e:
            logger.error(f"Error during paper collection: {e}")
            raise
        
        logger.info(f"Completed arXiv collection: {collected_count} papers collected")
    
    def collect_by_category(self, category: str, max_papers: int = 100) -> Generator[Dict, None, None]:
        """Collect papers from a specific arXiv category.
        
        Args:
            category: arXiv category code (e.g., "cs.AI", "cs.CL").
            max_papers: Maximum number of papers to collect.
            
        Yields:
            Dictionary containing paper data in corpus format.
        """
        yield from self.collect_papers(category=category, max_papers=max_papers)
    
    def collect_recent_papers(self, days: int = 7, max_papers: int = 100) -> Generator[Dict, None, None]:
        """Collect recent papers from the last N days.
        
        Args:
            days: Number of days to look back.
            max_papers: Maximum number of papers to collect.
            
        Yields:
            Dictionary containing paper data in corpus format.
        """
        yield from self.scraper.get_recent_papers(days=days, max_results=max_papers)
    
    def _convert_to_corpus_format(self, paper: Dict) -> Optional[Dict]:
        """Convert arXiv paper data to corpus format.
        
        Args:
            paper: Paper data from arXiv scraper.
            
        Returns:
            Dictionary in corpus format or None if conversion fails.
        """
        try:
            # Create the main text content from title and abstract
            main_text = f"{paper['title']}\n\n{paper['abstract']}"
            
            # Create corpus entry
            corpus_entry = {
                'id': f"arxiv_{paper['paper_id']}",
                'source': 'arxiv',
                'original_content': {
                    'raw_text': main_text,
                    'cleaned_text': main_text,  # Already cleaned by scraper
                    'title': paper['title'],
                    'abstract': paper['abstract'],
                    'authors': paper['authors'],
                    'categories': paper['categories'],
                    'paper_id': paper['paper_id'],
                    'url': paper['url'],
                    'published_date': paper['published_date'],
                    'doi': paper.get('doi'),
                    'journal_reference': paper.get('journal_reference')
                },
                'metadata': {
                    'scraped_at': paper['scraped_at'],
                    'source_type': 'academic_paper',
                    'language': 'en',
                    'word_count': len(main_text.split()),
                    'char_count': len(main_text),
                    'primary_category': paper['categories'][0] if paper['categories'] else None,
                    'all_categories': paper['categories'],
                    'author_count': len(paper['authors'])
                },
                'processing_info': {
                    'collected_at': datetime.now().isoformat(),
                    'client_version': '1.0',
                    'data_quality': self._assess_data_quality(paper)
                }
            }
            
            return corpus_entry
            
        except Exception as e:
            logger.error(f"Error converting paper to corpus format: {e}")
            return None
    
    def _assess_data_quality(self, paper: Dict) -> str:
        """Assess the quality of paper data.
        
        Args:
            paper: Paper data dictionary.
            
        Returns:
            Quality assessment string.
        """
        score = 0
        max_score = 5
        
        # Check if title exists and is reasonable length
        if paper.get('title') and len(paper['title'].split()) >= 3:
            score += 1
        
        # Check if abstract exists and is substantial
        if paper.get('abstract') and len(paper['abstract'].split()) >= 50:
            score += 1
        
        # Check if authors are present
        if paper.get('authors') and len(paper['authors']) > 0:
            score += 1
        
        # Check if categories are present
        if paper.get('categories') and len(paper['categories']) > 0:
            score += 1
        
        # Check if publication date is present
        if paper.get('published_date'):
            score += 1
        
        # Convert score to quality rating
        if score >= 5:
            return 'high'
        elif score >= 3:
            return 'medium'
        else:
            return 'low'


def main():
    """Example usage of the ArXiv client."""
    client = ArXivClient()
    
    # Example 1: Collect machine learning papers
    print("Collecting machine learning papers...")
    ml_papers = list(client.collect_papers(
        query="machine learning",
        max_papers=5
    ))
    
    for paper in ml_papers:
        print(f"ID: {paper['id']}")
        print(f"Title: {paper['original_content']['title']}")
        print(f"Authors: {', '.join(paper['original_content']['authors'])}")
        print(f"Quality: {paper['processing_info']['data_quality']}")
        print(f"Word count: {paper['metadata']['word_count']}")
        print("-" * 80)
    
    # Example 2: Collect from AI category
    print("\nCollecting from cs.AI category...")
    ai_papers = list(client.collect_by_category("cs.AI", max_papers=3))
    
    for paper in ai_papers:
        print(f"Title: {paper['original_content']['title']}")
        print(f"Categories: {', '.join(paper['original_content']['categories'])}")
        print()


if __name__ == "__main__":
    main()
