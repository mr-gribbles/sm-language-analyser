"""ArXiv academic paper scraper for corpus generation.

This module provides functionality to scrape academic papers from arXiv.org
using their OAI-PMH API. It extracts paper metadata, abstracts, and full text
when available, creating structured data suitable for NLP training.
"""

import json
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Generator, Tuple
from urllib.parse import urlencode
import re
import logging

logger = logging.getLogger(__name__)


class ArXivScraper:
    """Scraper for academic papers from arXiv.org."""
    
    def __init__(self, delay: float = 3.0):
        """Initialize the ArXiv scraper.
        
        Args:
            delay: Delay between API requests in seconds (arXiv recommends 3+ seconds).
        """
        self.base_url = "http://export.arxiv.org/api/query"
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Academic-Text-Corpus-Builder/1.0 (Research Use)'
        })
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content.
        
        Args:
            text: Raw text to clean.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove LaTeX commands (basic cleanup)
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Remove mathematical expressions in $ delimiters
        text = re.sub(r'\$[^$]*\$', '[MATH]', text)
        
        # Clean up common artifacts
        text = re.sub(r'\s*\[MATH\]\s*', ' [MATH] ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _parse_paper_entry(self, entry: ET.Element) -> Optional[Dict]:
        """Parse a single paper entry from arXiv API response.
        
        Args:
            entry: XML element containing paper data.
            
        Returns:
            Dictionary with paper information or None if parsing fails.
        """
        try:
            # Extract basic information
            paper_id = entry.find('{http://www.w3.org/2005/Atom}id').text
            paper_id = paper_id.split('/')[-1]  # Extract just the arXiv ID
            
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            title = self._clean_text(title)
            
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
            summary = self._clean_text(summary)
            
            # Extract authors
            authors = []
            for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                if name_elem is not None:
                    authors.append(name_elem.text)
            
            # Extract categories
            categories = []
            for category in entry.findall('{http://www.w3.org/2005/Atom}category'):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Extract publication date
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            published_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
            
            # Extract updated date
            updated_elem = entry.find('{http://www.w3.org/2005/Atom}updated')
            updated_date = None
            if updated_elem is not None:
                updated_date = datetime.fromisoformat(updated_elem.text.replace('Z', '+00:00'))
            
            # Extract DOI if available
            doi = None
            doi_elem = entry.find('{http://arxiv.org/schemas/atom}doi')
            if doi_elem is not None:
                doi = doi_elem.text
            
            # Extract journal reference if available
            journal_ref = None
            journal_elem = entry.find('{http://arxiv.org/schemas/atom}journal_ref')
            if journal_elem is not None:
                journal_ref = journal_elem.text
            
            # Skip papers with very short abstracts (likely incomplete)
            if len(summary.split()) < 50:
                logger.debug(f"Skipping paper {paper_id} - abstract too short")
                return None
            
            # Skip papers with non-English content (basic heuristic)
            if not self._is_likely_english(title + " " + summary):
                logger.debug(f"Skipping paper {paper_id} - likely non-English")
                return None
            
            return {
                'paper_id': paper_id,
                'title': title,
                'abstract': summary,
                'authors': authors,
                'categories': categories,
                'published_date': published_date.isoformat(),
                'updated_date': updated_date.isoformat() if updated_date else None,
                'doi': doi,
                'journal_reference': journal_ref,
                'source': 'arxiv',
                'url': f"https://arxiv.org/abs/{paper_id}",
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error parsing paper entry: {e}")
            return None
    
    def _is_likely_english(self, text: str) -> bool:
        """Basic heuristic to check if text is likely in English.
        
        Args:
            text: Text to check.
            
        Returns:
            True if text appears to be English.
        """
        # Simple heuristic: check for common English words
        english_indicators = [
            'the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'you', 'that',
            'we', 'for', 'are', 'with', 'as', 'this', 'have', 'from', 'or',
            'one', 'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were'
        ]
        
        text_lower = text.lower()
        english_word_count = sum(1 for word in english_indicators if word in text_lower)
        
        # If we find at least 5 common English words, assume it's English
        return english_word_count >= 5
    
    def search_papers(self, query: str = None, category: str = None, 
                     max_results: int = 100, start_date: str = None,
                     end_date: str = None) -> Generator[Dict, None, None]:
        """Search for papers on arXiv.
        
        Args:
            query: Search query (e.g., "machine learning", "neural networks").
            category: arXiv category (e.g., "cs.AI", "cs.CL", "stat.ML").
            max_results: Maximum number of papers to retrieve.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            
        Yields:
            Dictionary containing paper information.
        """
        # Build search query
        search_terms = []
        
        if query:
            # Search in title and abstract
            search_terms.append(f'(ti:"{query}" OR abs:"{query}")')
        
        if category:
            search_terms.append(f'cat:{category}')
        
        if start_date or end_date:
            date_range = "submittedDate:["
            date_range += start_date if start_date else "1991-01-01"
            date_range += " TO "
            date_range += end_date if end_date else "2030-12-31"
            date_range += "]"
            search_terms.append(date_range)
        
        # If no specific search terms, get recent papers
        if not search_terms:
            # Get papers from the last 30 days
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            search_terms.append(f"submittedDate:[{thirty_days_ago} TO 2030-12-31]")
        
        search_query = " AND ".join(search_terms)
        
        # Fetch papers in batches
        batch_size = min(100, max_results)  # arXiv API limit is 100 per request
        start_index = 0
        total_fetched = 0
        
        logger.info(f"Searching arXiv with query: {search_query}")
        
        while total_fetched < max_results:
            # Calculate how many to fetch in this batch
            current_batch_size = min(batch_size, max_results - total_fetched)
            
            # Build API request
            params = {
                'search_query': search_query,
                'start': start_index,
                'max_results': current_batch_size,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            try:
                logger.debug(f"Fetching batch: start={start_index}, size={current_batch_size}")
                response = self.session.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                entries = root.findall('{http://www.w3.org/2005/Atom}entry')
                
                if not entries:
                    logger.info("No more papers found")
                    break
                
                # Process each paper
                batch_count = 0
                for entry in entries:
                    paper_data = self._parse_paper_entry(entry)
                    if paper_data:
                        yield paper_data
                        batch_count += 1
                        total_fetched += 1
                
                logger.info(f"Processed {batch_count} papers from batch (total: {total_fetched})")
                
                # If we got fewer papers than requested, we've reached the end
                if len(entries) < current_batch_size:
                    break
                
                start_index += current_batch_size
                
                # Rate limiting
                time.sleep(self.delay)
                
            except requests.RequestException as e:
                logger.error(f"Error fetching papers: {e}")
                break
            except ET.ParseError as e:
                logger.error(f"Error parsing XML response: {e}")
                break
    
    def get_papers_by_category(self, category: str, max_results: int = 100) -> Generator[Dict, None, None]:
        """Get papers from a specific arXiv category.
        
        Args:
            category: arXiv category code (e.g., "cs.AI", "cs.CL", "stat.ML").
            max_results: Maximum number of papers to retrieve.
            
        Yields:
            Dictionary containing paper information.
        """
        yield from self.search_papers(category=category, max_results=max_results)
    
    def get_recent_papers(self, days: int = 7, max_results: int = 100) -> Generator[Dict, None, None]:
        """Get recent papers from the last N days.
        
        Args:
            days: Number of days to look back.
            max_results: Maximum number of papers to retrieve.
            
        Yields:
            Dictionary containing paper information.
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        yield from self.search_papers(start_date=start_date, end_date=end_date, max_results=max_results)


def main():
    """Example usage of the ArXiv scraper."""
    scraper = ArXivScraper()
    
    # Example 1: Get recent machine learning papers
    print("Fetching recent machine learning papers...")
    ml_papers = list(scraper.search_papers(
        query="machine learning",
        max_results=10
    ))
    
    for paper in ml_papers:
        print(f"Title: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"Categories: {', '.join(paper['categories'])}")
        print(f"Abstract: {paper['abstract'][:200]}...")
        print("-" * 80)
    
    # Example 2: Get papers from computer science AI category
    print("\nFetching papers from cs.AI category...")
    ai_papers = list(scraper.get_papers_by_category("cs.AI", max_results=5))
    
    for paper in ai_papers:
        print(f"Title: {paper['title']}")
        print(f"URL: {paper['url']}")
        print()


if __name__ == "__main__":
    main()
