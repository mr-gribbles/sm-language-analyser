"""Script for collecting academic papers from arXiv for corpus generation.

This script provides a command-line interface for collecting academic papers
from arXiv with various search options and categories.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.clients.arxiv_client import ArXivClient
from src.core_logic.corpus_manager import save_record_to_corpus
from src import config
from datetime import datetime, timezone
import json


def main():
    """Main function for the arXiv paper collection script."""
    parser = argparse.ArgumentParser(
        description="Collect academic papers from arXiv for corpus generation"
    )
    
    # Search options
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Search query (e.g., 'machine learning', 'neural networks')"
    )
    
    parser.add_argument(
        "--category", "-c",
        type=str,
        help="arXiv category (e.g., 'cs.AI', 'cs.CL', 'stat.ML')"
    )
    
    parser.add_argument(
        "--max-papers", "-n",
        type=int,
        default=50,
        help="Maximum number of papers to collect (default: 50)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for paper search (YYYY-MM-DD format)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for paper search (YYYY-MM-DD format)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="corpora/original_only",
        help="Output directory for collected papers (default: corpora/original_only)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output filename (default: auto-generated with timestamp)"
    )
    
    # Processing options
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Enable LLM rewriting of paper abstracts"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay between API requests in seconds (default: 3.0)"
    )
    
    # Predefined categories
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List popular arXiv categories and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_categories:
        print_popular_categories()
        return
    
    # Validate arguments
    if not args.query and not args.category:
        print("Warning: No query or category specified. Using default query 'machine learning'")
        args.query = "machine learning"
    
    # Generate output filename if not provided
    if not args.output_file:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        search_term = args.category or args.query.replace(" ", "_")
        args.output_file = f"arxiv_{search_term}_{timestamp}.jsonl"
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=== arXiv Paper Collection ===")
    print(f"Query: {args.query}")
    print(f"Category: {args.category}")
    print(f"Max papers: {args.max_papers}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Output: {args.output_dir}/{args.output_file}")
    print(f"Rewrite: {args.rewrite}")
    print("=" * 50)
    
    # Initialize client
    client = ArXivClient(delay=args.delay)
    
    # Collect papers
    collected_count = 0
    try:
        for paper_data in client.collect_papers(
            query=args.query,
            category=args.category,
            max_papers=args.max_papers,
            start_date=args.start_date,
            end_date=args.end_date
        ):
            # Save to corpus
            save_record_to_corpus(paper_data, args.output_dir, args.output_file)
            collected_count += 1
            
            print(f"Collected {collected_count}/{args.max_papers}: {paper_data['original_content']['title'][:60]}...")
            
            if collected_count >= args.max_papers:
                break
    
    except KeyboardInterrupt:
        print(f"\nCollection interrupted by user. Collected {collected_count} papers.")
    except Exception as e:
        print(f"Error during collection: {e}")
        return 1
    
    print(f"\nCollection complete! Collected {collected_count} papers.")
    print(f"Output saved to: {args.output_dir}/{args.output_file}")
    
    # Print some statistics
    if collected_count > 0:
        print_collection_stats(args.output_dir, args.output_file)
    
    return 0


def print_popular_categories():
    """Print popular arXiv categories."""
    categories = {
        "Computer Science": {
            "cs.AI": "Artificial Intelligence",
            "cs.CL": "Computation and Language (NLP)",
            "cs.CV": "Computer Vision and Pattern Recognition",
            "cs.LG": "Machine Learning",
            "cs.NE": "Neural and Evolutionary Computing",
            "cs.RO": "Robotics",
            "cs.IR": "Information Retrieval",
            "cs.HC": "Human-Computer Interaction",
            "cs.CR": "Cryptography and Security",
            "cs.DS": "Data Structures and Algorithms"
        },
        "Statistics": {
            "stat.ML": "Machine Learning",
            "stat.AP": "Applications",
            "stat.CO": "Computation",
            "stat.ME": "Methodology",
            "stat.TH": "Statistics Theory"
        },
        "Mathematics": {
            "math.ST": "Statistics Theory",
            "math.PR": "Probability",
            "math.OC": "Optimization and Control",
            "math.NA": "Numerical Analysis",
            "math.IT": "Information Theory"
        },
        "Physics": {
            "physics.data-an": "Data Analysis, Statistics and Probability",
            "physics.comp-ph": "Computational Physics",
            "physics.soc-ph": "Physics and Society"
        },
        "Quantitative Biology": {
            "q-bio.QM": "Quantitative Methods",
            "q-bio.GN": "Genomics",
            "q-bio.BM": "Biomolecules",
            "q-bio.NC": "Neurons and Cognition"
        },
        "Economics": {
            "econ.EM": "Econometrics",
            "econ.TH": "Theoretical Economics"
        }
    }
    
    print("Popular arXiv Categories:")
    print("=" * 50)
    
    for field, cats in categories.items():
        print(f"\n{field}:")
        for code, name in cats.items():
            print(f"  {code:<12} - {name}")
    
    print("\nExample usage:")
    print("  python scripts/collect_arxiv_papers.py --category cs.AI --max-papers 100")
    print("  python scripts/collect_arxiv_papers.py --query 'deep learning' --max-papers 50")
    print("  python scripts/collect_arxiv_papers.py --category stat.ML --start-date 2024-01-01")


def print_collection_stats(output_dir: str, output_file: str):
    """Print statistics about the collected papers."""
    file_path = Path(output_dir) / output_file
    
    if not file_path.exists():
        return
    
    try:
        papers = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                papers.append(json.loads(line.strip()))
        
        if not papers:
            return
        
        print("\nCollection Statistics:")
        print("-" * 30)
        
        # Count by category
        categories = {}
        authors = set()
        word_counts = []
        
        for paper in papers:
            # Categories
            paper_cats = paper['original_content'].get('categories', [])
            for cat in paper_cats:
                categories[cat] = categories.get(cat, 0) + 1
            
            # Authors
            paper_authors = paper['original_content'].get('authors', [])
            authors.update(paper_authors)
            
            # Word count
            text = paper['original_content'].get('cleaned_text', '')
            word_counts.append(len(text.split()))
        
        print(f"Total papers: {len(papers)}")
        print(f"Unique authors: {len(authors)}")
        
        if word_counts:
            avg_words = sum(word_counts) / len(word_counts)
            print(f"Average words per paper: {avg_words:.0f}")
            print(f"Total words: {sum(word_counts):,}")
        
        # Top categories
        if categories:
            print(f"\nTop categories:")
            sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
            for cat, count in sorted_cats[:5]:
                print(f"  {cat}: {count} papers")
    
    except Exception as e:
        print(f"Error calculating statistics: {e}")


if __name__ == "__main__":
    sys.exit(main())
