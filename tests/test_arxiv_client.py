"""Tests for the arXiv client functionality."""

import pytest
from unittest.mock import Mock, patch
from src.clients.arxiv_client import ArXivClient


class TestArXivClient:
    """Test cases for ArXivClient."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = ArXivClient(delay=0.1)  # Faster for testing
    
    def test_client_initialization(self):
        """Test that the client initializes correctly."""
        assert self.client.source_name == "arxiv"
        assert self.client.scraper is not None
    
    def test_convert_to_corpus_format(self):
        """Test conversion of arXiv paper data to corpus format."""
        # Mock paper data
        paper_data = {
            'paper_id': '2024.01234',
            'title': 'Test Paper Title',
            'abstract': 'This is a test abstract for the paper.',
            'authors': ['John Doe', 'Jane Smith'],
            'categories': ['cs.AI', 'cs.LG'],
            'published_date': '2024-01-01T00:00:00+00:00',
            'url': 'https://arxiv.org/abs/2024.01234',
            'scraped_at': '2024-01-01T12:00:00+00:00',
            'doi': None,
            'journal_reference': None
        }
        
        # Convert to corpus format
        corpus_entry = self.client._convert_to_corpus_format(paper_data)
        
        # Verify the conversion
        assert corpus_entry is not None
        assert corpus_entry['id'] == 'arxiv_2024.01234'
        assert corpus_entry['source'] == 'arxiv'
        assert 'original_content' in corpus_entry
        assert 'metadata' in corpus_entry
        assert 'processing_info' in corpus_entry
        
        # Check original content
        original = corpus_entry['original_content']
        assert original['title'] == 'Test Paper Title'
        assert original['abstract'] == 'This is a test abstract for the paper.'
        assert original['authors'] == ['John Doe', 'Jane Smith']
        assert original['categories'] == ['cs.AI', 'cs.LG']
        
        # Check metadata
        metadata = corpus_entry['metadata']
        assert metadata['source_type'] == 'academic_paper'
        assert metadata['language'] == 'en'
        assert metadata['primary_category'] == 'cs.AI'
        assert metadata['author_count'] == 2
    
    def test_assess_data_quality(self):
        """Test data quality assessment."""
        # High quality paper
        high_quality_paper = {
            'title': 'A Comprehensive Study of Machine Learning',
            'abstract': 'This paper presents a comprehensive analysis of machine learning techniques and their applications in various domains. ' * 10,  # Long abstract
            'authors': ['Author One', 'Author Two'],
            'categories': ['cs.AI'],
            'published_date': '2024-01-01T00:00:00+00:00'
        }
        
        quality = self.client._assess_data_quality(high_quality_paper)
        assert quality == 'high'
        
        # Low quality paper
        low_quality_paper = {
            'title': 'Test',
            'abstract': 'Short abstract.',
            'authors': [],
            'categories': [],
            'published_date': None
        }
        
        quality = self.client._assess_data_quality(low_quality_paper)
        assert quality == 'low'
    
    @patch('src.clients.arxiv_client.ArXivScraper')
    def test_collect_papers_with_query(self, mock_scraper_class):
        """Test paper collection with a query."""
        # Mock the scraper
        mock_scraper = Mock()
        mock_scraper_class.return_value = mock_scraper
        
        # Mock paper data
        mock_paper = {
            'paper_id': '2024.01234',
            'title': 'Test Paper',
            'abstract': 'Test abstract with sufficient length to pass quality checks.',
            'authors': ['Test Author'],
            'categories': ['cs.AI'],
            'published_date': '2024-01-01T00:00:00+00:00',
            'url': 'https://arxiv.org/abs/2024.01234',
            'scraped_at': '2024-01-01T12:00:00+00:00',
            'doi': None,
            'journal_reference': None
        }
        
        mock_scraper.search_papers.return_value = [mock_paper]
        
        # Create new client to use mocked scraper
        client = ArXivClient()
        
        # Collect papers
        papers = list(client.collect_papers(query="machine learning", max_papers=1))
        
        # Verify results
        assert len(papers) == 1
        assert papers[0]['id'] == 'arxiv_2024.01234'
        
        # Verify scraper was called correctly
        mock_scraper.search_papers.assert_called_once_with(
            query="machine learning",
            category=None,
            max_results=1,
            start_date=None,
            end_date=None
        )


if __name__ == "__main__":
    pytest.main([__file__])
