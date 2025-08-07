"""Main entry point for the Academic Paper Analysis Pipeline.

This module provides the command-line interface for running the data collection
and processing pipeline for academic papers from arXiv. It supports various
search methods including specific categories, queries, and random selection.
"""
import argparse
import random
from typing import List, Tuple

from src.arxiv_paper_collector import run_paper_collection_pipeline


def get_popular_arxiv_categories() -> List[Tuple[str, str]]:
    """Get a list of popular arXiv categories for random selection.
    
    Returns:
        List of tuples containing (category_code, description) for popular
        arXiv categories across multiple academic disciplines.
    """
    categories = [
        # Computer Science
        ("cs.AI", "Artificial Intelligence"),
        ("cs.CL", "Computation and Language (NLP)"),
        ("cs.CV", "Computer Vision and Pattern Recognition"),
        ("cs.LG", "Machine Learning"),
        ("cs.NE", "Neural and Evolutionary Computing"),
        ("cs.RO", "Robotics"),
        ("cs.IR", "Information Retrieval"),
        ("cs.HC", "Human-Computer Interaction"),
        ("cs.CR", "Cryptography and Security"),
        ("cs.DS", "Data Structures and Algorithms"),
        
        # Statistics
        ("stat.ML", "Machine Learning (Statistics)"),
        ("stat.AP", "Applications"),
        ("stat.CO", "Computation"),
        ("stat.ME", "Methodology"),
        ("stat.TH", "Statistics Theory"),
        
        # Mathematics
        ("math.ST", "Statistics Theory"),
        ("math.PR", "Probability"),
        ("math.OC", "Optimization and Control"),
        ("math.NA", "Numerical Analysis"),
        ("math.IT", "Information Theory"),
        
        # Physics
        ("physics.data-an", "Data Analysis, Statistics and Probability"),
        ("physics.comp-ph", "Computational Physics"),
        ("physics.soc-ph", "Physics and Society"),
        
        # Quantitative Biology
        ("q-bio.QM", "Quantitative Methods"),
        ("q-bio.GN", "Genomics"),
        ("q-bio.BM", "Biomolecules"),
        ("q-bio.NC", "Neurons and Cognition"),
        
        # Economics
        ("econ.EM", "Econometrics"),
        ("econ.TH", "Theoretical Economics")
    ]
    return categories


def select_random_category() -> Tuple[str, str]:
    """Select a random category from the popular categories list.
    
    Returns:
        Tuple containing (category_code, description) for a randomly
        selected arXiv category.
    """
    categories = get_popular_arxiv_categories()
    return random.choice(categories)


def print_available_categories() -> None:
    """Print all available arXiv categories organized by academic field."""
    categories = get_popular_arxiv_categories()
    
    print("Available arXiv Categories")
    print("=" * 50)
    
    # Group categories by academic field
    fields = {
        "Computer Science": [],
        "Statistics": [],
        "Mathematics": [],
        "Physics": [],
        "Quantitative Biology": [],
        "Economics": []
    }
    
    for code, description in categories:
        if code.startswith("cs."):
            fields["Computer Science"].append((code, description))
        elif code.startswith("stat."):
            fields["Statistics"].append((code, description))
        elif code.startswith("math."):
            fields["Mathematics"].append((code, description))
        elif code.startswith("physics."):
            fields["Physics"].append((code, description))
        elif code.startswith("q-bio."):
            fields["Quantitative Biology"].append((code, description))
        elif code.startswith("econ."):
            fields["Economics"].append((code, description))
    
    # Print categories organized by field
    for field, field_categories in fields.items():
        if field_categories:
            print(f"\n{field}:")
            for code, description in field_categories:
                print(f"   {code:<20} - {description}")
    
    print(f"\nTotal categories available: {len(categories)}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Academic Paper Analysis Pipeline - Collect and analyze "
                   "academic papers from arXiv for NLP research and AI detection."
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Search query for papers (e.g., 'machine learning', 'neural networks')"
    )
    
    parser.add_argument(
        "--category", "-c",
        type=str,
        help="arXiv category code (e.g., 'cs.AI', 'cs.CL', 'stat.ML')"
    )
    
    parser.add_argument(
        "--random", "-r",
        action="store_true",
        help="Randomly select a category from popular arXiv categories"
    )
    
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available arXiv categories and exit"
    )
    
    parser.add_argument(
        "--max-papers", "-n",
        type=int,
        default=50,
        help="Maximum number of papers to collect (default: 50)"
    )
    
    return parser


def main() -> None:
    """Parse command-line arguments and run the paper collection pipeline."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle list categories option
    if args.list_categories:
        print_available_categories()
        return

    # Handle random category selection
    category = args.category
    if args.random:
        if args.category:
            print("Warning: Both --random and --category specified. "
                  "Using random selection.")
        
        selected_category, description = select_random_category()
        category = selected_category
        print(f"Randomly selected category: {category} ({description})")
    
    # Provide helpful message if no search criteria specified
    if not category and not args.query and not args.random:
        print("No search criteria specified. Using default query 'machine learning'")
        print("Tip: Use --random to randomly select a category, or specify "
              "--category or --query")

    # Run the paper collection pipeline
    run_paper_collection_pipeline(
        query=args.query,
        category=category,
        max_papers=args.max_papers
    )


if __name__ == "__main__":
    main()
