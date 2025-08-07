"""Semantic Scholar client for academic paper collection.

This module provides a high-level client interface for collecting academic papers
from Semantic Scholar across multiple disciplines including psychology, philosophy,
business, and economics.
"""

import logging
from typing import Dict, List, Optional, Generator
from datetime import datetime

try:
    from ..scrapers.semantic_scholar_scraper import SemanticScholarScraper
except ImportError:
    from scrapers.semantic_scholar_scraper import SemanticScholarScraper

logger = logging.getLogger(__name__)


class SemanticScholarClient:
    """High-level client for Semantic Scholar paper collection."""
    
    def __init__(self, delay: float = 1.0, api_key: Optional[str] = None):
        """Initialize the Semantic Scholar client.
        
        Args:
            delay: Delay between API requests in seconds.
            api_key: Optional API key for higher rate limits.
        """
        self.scraper = SemanticScholarScraper(delay=delay, api_key=api_key)
        
        # Predefined field collections for easy access
        self.field_collections = {
            'social_sciences': ['Psychology', 'Sociology', 'Political Science', 'Anthropology'],
            'humanities': ['Philosophy', 'History', 'Linguistics', 'Literature'],
            'business_economics': ['Business', 'Economics', 'Management', 'Finance'],
            'interdisciplinary': ['Psychology', 'Philosophy', 'Business', 'Economics', 'Political Science']
        }
    
    def collect_papers_by_field(self, field: str, max_papers: int = 100,
                               year_range: tuple = None) -> Generator[Dict, None, None]:
        """Collect papers from a specific academic field.
        
        Args:
            field: Academic field name (e.g., 'Psychology', 'Philosophy', 'Business').
            max_papers: Maximum number of papers to collect.
            year_range: Optional tuple of (start_year, end_year).
            
        Yields:
            Paper dictionaries.
        """
        try:
            yield from self.scraper.get_papers_by_field(
                field_of_study=field,
                max_results=max_papers,
                year_range=year_range
            )
                    
        except Exception as e:
            logger.error(f"Error collecting papers from {field}: {e}")
    
    def collect_papers_by_topic(self, topic: str, fields: List[str] = None,
                               max_papers: int = 100, min_citations: int = 0) -> List[Dict]:
        """Collect papers on a specific topic across multiple fields.
        
        Args:
            topic: Research topic or keywords.
            fields: List of fields to search in (optional).
            max_papers: Maximum number of papers to collect.
            min_citations: Minimum citation count filter.
            
        Returns:
            List of paper dictionaries.
        """
        logger.info(f"Collecting papers on topic: {topic}")
        
        papers = []
        try:
            for paper in self.scraper.search_papers(
                query=topic,
                fields_of_study=fields,
                max_results=max_papers,
                min_citation_count=min_citations
            ):
                papers.append(paper)
                
                if len(papers) % 10 == 0:
                    logger.info(f"Collected {len(papers)} papers so far...")
                    
        except Exception as e:
            logger.error(f"Error collecting papers on topic {topic}: {e}")
        
        logger.info(f"Successfully collected {len(papers)} papers on {topic}")
        return papers
    
    def collect_interdisciplinary_papers(self, max_papers: int = 100,
                                       year_range: tuple = None) -> List[Dict]:
        """Collect papers from multiple disciplines for diverse corpus.
        
        Args:
            max_papers: Maximum total number of papers to collect.
            year_range: Optional tuple of (start_year, end_year).
            
        Returns:
            List of paper dictionaries from various fields.
        """
        logger.info(f"Collecting interdisciplinary papers (max: {max_papers})")
        
        fields = self.field_collections['interdisciplinary']
        papers_per_field = max_papers // len(fields)
        
        all_papers = []
        
        for field in fields:
            logger.info(f"Collecting {papers_per_field} papers from {field}")
            
            try:
                field_papers = list(self.scraper.get_papers_by_field(
                    field_of_study=field,
                    max_results=papers_per_field,
                    year_range=year_range
                ))
                
                all_papers.extend(field_papers)
                logger.info(f"Added {len(field_papers)} papers from {field}")
                
            except Exception as e:
                logger.error(f"Error collecting papers from {field}: {e}")
        
        # If we have fewer papers than requested, try to fill the gap
        if len(all_papers) < max_papers:
            remaining = max_papers - len(all_papers)
            logger.info(f"Collecting {remaining} additional papers")
            
            try:
                additional_papers = list(self.scraper.search_papers(
                    query="research",
                    max_results=remaining,
                    year_range=year_range
                ))
                all_papers.extend(additional_papers)
                
            except Exception as e:
                logger.error(f"Error collecting additional papers: {e}")
        
        logger.info(f"Successfully collected {len(all_papers)} interdisciplinary papers")
        return all_papers
    
    def collect_highly_cited_papers(self, field: str, min_citations: int = 50,
                                   max_papers: int = 100) -> List[Dict]:
        """Collect highly cited papers from a specific field.
        
        Args:
            field: Academic field name.
            min_citations: Minimum citation count.
            max_papers: Maximum number of papers to collect.
            
        Returns:
            List of highly cited paper dictionaries.
        """
        logger.info(f"Collecting highly cited papers from {field} (min citations: {min_citations})")
        
        papers = []
        try:
            for paper in self.scraper.get_highly_cited_papers(
                field_of_study=field,
                min_citations=min_citations,
                max_results=max_papers
            ):
                papers.append(paper)
                
                if len(papers) % 10 == 0:
                    logger.info(f"Collected {len(papers)} highly cited papers so far...")
                    
        except Exception as e:
            logger.error(f"Error collecting highly cited papers from {field}: {e}")
        
        logger.info(f"Successfully collected {len(papers)} highly cited papers from {field}")
        return papers
    
    def collect_recent_papers(self, field: str = None, days: int = 365,
                             max_papers: int = 100) -> List[Dict]:
        """Collect recent papers from the specified time period.
        
        Args:
            field: Optional field to filter by.
            days: Number of days to look back.
            max_papers: Maximum number of papers to collect.
            
        Returns:
            List of recent paper dictionaries.
        """
        field_desc = f" from {field}" if field else ""
        logger.info(f"Collecting recent papers{field_desc} (last {days} days)")
        
        papers = []
        try:
            for paper in self.scraper.get_recent_papers(
                field_of_study=field,
                days=days,
                max_results=max_papers
            ):
                papers.append(paper)
                
                if len(papers) % 10 == 0:
                    logger.info(f"Collected {len(papers)} recent papers so far...")
                    
        except Exception as e:
            logger.error(f"Error collecting recent papers{field_desc}: {e}")
        
        logger.info(f"Successfully collected {len(papers)} recent papers{field_desc}")
        return papers
    
    def get_available_fields(self) -> Dict[str, List[str]]:
        """Get available field collections.
        
        Returns:
            Dictionary of field collection names and their constituent fields.
        """
        return self.field_collections.copy()
    
    def collect_balanced_corpus(self, fields: List[str], papers_per_field: int = 50,
                               year_range: tuple = None) -> List[Dict]:
        """Collect a balanced corpus with equal representation from each field.
        
        Args:
            fields: List of academic fields to include.
            papers_per_field: Number of papers to collect from each field.
            year_range: Optional tuple of (start_year, end_year).
            
        Returns:
            List of paper dictionaries with balanced representation.
        """
        logger.info(f"Collecting balanced corpus from {len(fields)} fields ({papers_per_field} papers each)")
        
        all_papers = []
        
        for field in fields:
            logger.info(f"Collecting {papers_per_field} papers from {field}")
            
            try:
                field_papers = list(self.scraper.get_papers_by_field(
                    field_of_study=field,
                    max_results=papers_per_field,
                    year_range=year_range
                ))
                
                all_papers.extend(field_papers)
                logger.info(f"Added {len(field_papers)} papers from {field}")
                
            except Exception as e:
                logger.error(f"Error collecting papers from {field}: {e}")
        
        logger.info(f"Successfully collected balanced corpus with {len(all_papers)} total papers")
        return all_papers


def main():
    """Example usage of the Semantic Scholar client."""
    client = SemanticScholarClient()
    
    # Example 1: Collect psychology papers
    psychology_papers = client.collect_papers_by_field("Psychology", max_papers=10)
    print(f"Collected {len(psychology_papers)} psychology papers")
    
    # Example 2: Collect papers on a specific topic
    behavioral_papers = client.collect_papers_by_topic(
        "behavioral economics",
        fields=["Economics", "Psychology"],
        max_papers=5
    )
    print(f"Collected {len(behavioral_papers)} behavioral economics papers")
    
    # Example 3: Collect interdisciplinary papers
    interdisciplinary_papers = client.collect_interdisciplinary_papers(max_papers=20)
    print(f"Collected {len(interdisciplinary_papers)} interdisciplinary papers")
    
    # Show available field collections
    print("\nAvailable field collections:")
    for collection_name, fields in client.get_available_fields().items():
        print(f"  {collection_name}: {', '.join(fields)}")


if __name__ == "__main__":
    main()
