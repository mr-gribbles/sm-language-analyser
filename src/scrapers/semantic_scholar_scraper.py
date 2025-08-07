"""Semantic Scholar academic paper scraper for corpus generation.

This module provides functionality to scrape academic papers from Semantic Scholar
using their Academic Graph API. It extracts paper metadata, abstracts, and full text
when available, creating structured data suitable for NLP training across multiple
academic disciplines including psychology, philosophy, business, and economics.
"""

import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Generator, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class SemanticScholarScraper:
    """Scraper for academic papers from Semantic Scholar."""
    
    def __init__(self, delay: float = 1.0, api_key: Optional[str] = None):
        """Initialize the Semantic Scholar scraper.
        
        Args:
            delay: Delay between API requests in seconds (1 second is recommended).
            api_key: Optional API key for higher rate limits.
        """
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.delay = delay
        self.session = requests.Session()
        
        headers = {
            'User-Agent': 'Academic-Text-Corpus-Builder/1.0 (Research Use)'
        }
        
        if api_key:
            headers['x-api-key'] = api_key
            
        self.session.headers.update(headers)
        
        # Field of study mappings for better categorization
        self.field_mappings = {
            'Psychology': ['Psychology', 'Cognitive Science', 'Behavioral Science'],
            'Philosophy': ['Philosophy', 'Ethics', 'Logic'],
            'Business': ['Business', 'Management', 'Marketing', 'Finance'],
            'Economics': ['Economics', 'Econometrics', 'Economic Policy'],
            'Political Science': ['Political Science', 'Public Policy', 'International Relations'],
            'Sociology': ['Sociology', 'Social Science', 'Anthropology'],
            'Education': ['Education', 'Educational Technology', 'Pedagogy'],
            'Law': ['Law', 'Legal Studies', 'Jurisprudence'],
            'History': ['History', 'Historical Studies'],
            'Linguistics': ['Linguistics', 'Language', 'Computational Linguistics']
        }
    
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
        
        # Remove common artifacts from academic papers
        text = re.sub(r'\b(Abstract|ABSTRACT)\b:?\s*', '', text)
        text = re.sub(r'\b(Keywords|KEYWORDS)\b:?\s*.*?(?=\.|$)', '', text)
        
        # Clean up mathematical expressions and special characters
        text = re.sub(r'[^\w\s\.,;:!?\-\(\)\[\]\'\"]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _is_likely_english(self, text: str) -> bool:
        """Basic heuristic to check if text is likely in English.
        
        Args:
            text: Text to check.
            
        Returns:
            True if text appears to be English.
        """
        if not text:
            return False
            
        # Simple heuristic: check for common English words
        english_indicators = [
            'the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'that',
            'we', 'for', 'are', 'with', 'as', 'this', 'have', 'from',
            'study', 'research', 'analysis', 'results', 'method', 'data'
        ]
        
        text_lower = text.lower()
        english_word_count = sum(1 for word in english_indicators if word in text_lower)
        
        # If we find at least 4 common English words, assume it's English
        return english_word_count >= 4
    
    def _parse_paper_data(self, paper: Dict) -> Optional[Dict]:
        """Parse paper data from Semantic Scholar API response.
        
        Args:
            paper: Paper data from API response.
            
        Returns:
            Dictionary with standardized paper information or None if parsing fails.
        """
        try:
            # Extract basic information
            paper_id = paper.get('paperId')
            if not paper_id:
                return None
            
            title = paper.get('title', '')
            title = self._clean_text(title)
            
            abstract = paper.get('abstract', '')
            abstract = self._clean_text(abstract)
            
            # Skip papers without abstracts or with very short abstracts
            if not abstract or len(abstract.split()) < 30:
                logger.debug(f"Skipping paper {paper_id} - abstract too short or missing")
                return None
            
            # Skip non-English papers
            if not self._is_likely_english(title + " " + abstract):
                logger.debug(f"Skipping paper {paper_id} - likely non-English")
                return None
            
            # Extract authors
            authors = []
            if paper.get('authors'):
                for author in paper['authors']:
                    if author.get('name'):
                        authors.append(author['name'])
            
            # Extract publication info
            year = paper.get('year')
            venue = paper.get('venue', '')
            
            # Extract fields of study
            fields_of_study = []
            if paper.get('fieldsOfStudy'):
                fields_of_study = paper['fieldsOfStudy']
            
            # Map to broader categories
            categories = self._map_fields_to_categories(fields_of_study)
            
            # Extract citation count
            citation_count = paper.get('citationCount', 0)
            
            # Extract reference count
            reference_count = paper.get('referenceCount', 0)
            
            # Extract influential citation count
            influential_citation_count = paper.get('influentialCitationCount', 0)
            
            # Extract external IDs
            external_ids = paper.get('externalIds', {})
            doi = external_ids.get('DOI')
            arxiv_id = external_ids.get('ArXiv')
            pubmed_id = external_ids.get('PubMed')
            
            # Extract URL
            url = paper.get('url', f"https://www.semanticscholar.org/paper/{paper_id}")
            
            # Extract open access info
            is_open_access = paper.get('isOpenAccess', False)
            open_access_pdf = None
            if paper.get('openAccessPdf'):
                open_access_pdf = paper['openAccessPdf'].get('url')
            
            return {
                'paper_id': paper_id,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'categories': categories,
                'fields_of_study': fields_of_study,
                'year': year,
                'venue': venue,
                'citation_count': citation_count,
                'reference_count': reference_count,
                'influential_citation_count': influential_citation_count,
                'doi': doi,
                'arxiv_id': arxiv_id,
                'pubmed_id': pubmed_id,
                'is_open_access': is_open_access,
                'open_access_pdf': open_access_pdf,
                'source': 'semantic_scholar',
                'url': url,
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error parsing paper data: {e}")
            return None
    
    def _map_fields_to_categories(self, fields_of_study: List[str]) -> List[str]:
        """Map Semantic Scholar fields of study to broader categories.
        
        Args:
            fields_of_study: List of field names from Semantic Scholar.
            
        Returns:
            List of broader category names.
        """
        categories = set()
        
        for field in fields_of_study:
            for category, field_list in self.field_mappings.items():
                if any(mapped_field.lower() in field.lower() for mapped_field in field_list):
                    categories.add(category)
        
        # If no mapping found, use the original fields
        if not categories:
            categories = set(fields_of_study[:3])  # Take first 3 fields
        
        return list(categories)
    
    def search_papers(self, query: str, fields_of_study: List[str] = None,
                     year_range: Tuple[int, int] = None, min_citation_count: int = 0,
                     max_results: int = 100, open_access_only: bool = False) -> Generator[Dict, None, None]:
        """Search for papers on Semantic Scholar.
        
        Args:
            query: Search query (keywords, topics, etc.).
            fields_of_study: List of fields to filter by (e.g., ["Psychology", "Philosophy"]).
            year_range: Tuple of (start_year, end_year) to filter by publication year.
            min_citation_count: Minimum number of citations required.
            max_results: Maximum number of papers to retrieve.
            open_access_only: If True, only return open access papers.
            
        Yields:
            Dictionary containing paper information.
        """
        # Build search parameters
        params = {
            'query': query,
            'limit': min(100, max_results),  # API limit is 100 per request
            'fields': 'paperId,title,abstract,authors,year,venue,fieldsOfStudy,citationCount,referenceCount,influentialCitationCount,externalIds,url,isOpenAccess,openAccessPdf'
        }
        
        if fields_of_study:
            params['fieldsOfStudy'] = ','.join(fields_of_study)
        
        if year_range:
            params['year'] = f"{year_range[0]}-{year_range[1]}"
        
        if min_citation_count > 0:
            params['minCitationCount'] = min_citation_count
        
        if open_access_only:
            params['openAccessPdf'] = 'true'
        
        offset = 0
        total_fetched = 0
        
        logger.info(f"Searching Semantic Scholar with query: {query}")
        
        while total_fetched < max_results:
            params['offset'] = offset
            current_limit = min(100, max_results - total_fetched)
            params['limit'] = current_limit
            
            retry_count = 0
            max_retries = 3
            base_delay = self.delay
            
            while retry_count <= max_retries:
                try:
                    logger.debug(f"Fetching batch: offset={offset}, limit={current_limit}")
                    response = self.session.get(f"{self.base_url}/paper/search", params=params, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    papers = data.get('data', [])
                    
                    if not papers:
                        logger.info("No more papers found")
                        return
                    
                    # Process each paper
                    batch_count = 0
                    for paper in papers:
                        paper_data = self._parse_paper_data(paper)
                        if paper_data:
                            yield paper_data
                            batch_count += 1
                            total_fetched += 1
                    
                    logger.info(f"Processed {batch_count} papers from batch (total: {total_fetched})")
                    
                    # Check if we've reached the end
                    if len(papers) < current_limit:
                        return
                    
                    offset += len(papers)
                    
                    # Rate limiting - successful request, reset retry count
                    retry_count = 0
                    time.sleep(base_delay)
                    break
                    
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limit error
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.warning(f"Max retries exceeded for rate limiting. Stopping collection.")
                            return
                        
                        # Exponential backoff for rate limiting
                        wait_time = base_delay * (2 ** retry_count) + (retry_count * 5)
                        logger.warning(f"Rate limited. Waiting {wait_time:.1f} seconds before retry {retry_count}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"HTTP error fetching papers: {e}")
                        return
                        
                except requests.RequestException as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error(f"Max retries exceeded for request error: {e}")
                        return
                    
                    wait_time = base_delay * retry_count
                    logger.warning(f"Request error. Waiting {wait_time:.1f} seconds before retry {retry_count}/{max_retries}: {e}")
                    time.sleep(wait_time)
                    continue
                    
                except (KeyError, ValueError) as e:
                    logger.error(f"Error parsing API response: {e}")
                    return
    
    def get_papers_by_field(self, field_of_study: str, max_results: int = 100,
                           year_range: Tuple[int, int] = None) -> Generator[Dict, None, None]:
        """Get papers from a specific field of study.
        
        Args:
            field_of_study: Field name (e.g., "Psychology", "Philosophy", "Business").
            max_results: Maximum number of papers to retrieve.
            year_range: Optional year range filter.
            
        Yields:
            Dictionary containing paper information.
        """
        # Use a broad query for the field
        query = field_of_study.lower()
        yield from self.search_papers(
            query=query,
            fields_of_study=[field_of_study],
            year_range=year_range,
            max_results=max_results
        )
    
    def get_recent_papers(self, field_of_study: str = None, days: int = 365,
                         max_results: int = 100) -> Generator[Dict, None, None]:
        """Get recent papers from the last N days.
        
        Args:
            field_of_study: Optional field to filter by.
            days: Number of days to look back (default: 365 for recent papers).
            max_results: Maximum number of papers to retrieve.
            
        Yields:
            Dictionary containing paper information.
        """
        current_year = datetime.now().year
        start_year = max(2000, current_year - (days // 365) - 1)
        
        query = field_of_study.lower() if field_of_study else "research"
        fields = [field_of_study] if field_of_study else None
        
        yield from self.search_papers(
            query=query,
            fields_of_study=fields,
            year_range=(start_year, current_year),
            max_results=max_results
        )
    
    def get_highly_cited_papers(self, field_of_study: str, min_citations: int = 50,
                               max_results: int = 100) -> Generator[Dict, None, None]:
        """Get highly cited papers from a specific field.
        
        Args:
            field_of_study: Field name to search in.
            min_citations: Minimum citation count.
            max_results: Maximum number of papers to retrieve.
            
        Yields:
            Dictionary containing paper information.
        """
        query = field_of_study.lower()
        yield from self.search_papers(
            query=query,
            fields_of_study=[field_of_study],
            min_citation_count=min_citations,
            max_results=max_results
        )


def main():
    """Example usage of the Semantic Scholar scraper."""
    scraper = SemanticScholarScraper()
    
    # Example 1: Get psychology papers
    print("Fetching psychology papers...")
    psychology_papers = list(scraper.get_papers_by_field("Psychology", max_results=5))
    
    for paper in psychology_papers:
        print(f"Title: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"Categories: {', '.join(paper['categories'])}")
        print(f"Year: {paper['year']}")
        print(f"Citations: {paper['citation_count']}")
        print(f"Abstract: {paper['abstract'][:200]}...")
        print("-" * 80)
    
    # Example 2: Search for specific topic
    print("\nFetching papers on 'behavioral economics'...")
    behavioral_econ_papers = list(scraper.search_papers(
        query="behavioral economics",
        fields_of_study=["Economics", "Psychology"],
        max_results=3
    ))
    
    for paper in behavioral_econ_papers:
        print(f"Title: {paper['title']}")
        print(f"Venue: {paper['venue']}")
        print(f"URL: {paper['url']}")
        print()


if __name__ == "__main__":
    main()
