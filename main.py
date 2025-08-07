"""Main entry point for the Academic Paper Analysis Pipeline.

This module provides the command-line interface for running the data collection
and processing pipeline for academic papers from both arXiv and Semantic Scholar.
It automatically selects the appropriate scraper based on the category chosen.
"""
import argparse
import random
from typing import List, Tuple

from src.arxiv_paper_collector import run_paper_collection_pipeline
from src.semantic_scholar_paper_collector import run_semantic_scholar_collection_pipeline


def get_all_available_categories() -> List[Tuple[str, str, str]]:
    """Get a list of all available categories from both ArXiv and Semantic Scholar.
    
    Returns:
        List of tuples containing (category_code, description, source) for all
        available categories across both platforms.
    """
    categories = []
    
    # ArXiv categories
    arxiv_categories = [
        # Computer Science
        ("cs.AI", "Artificial Intelligence", "arxiv"),
        ("cs.CL", "Computation and Language (NLP)", "arxiv"),
        ("cs.CV", "Computer Vision and Pattern Recognition", "arxiv"),
        ("cs.LG", "Machine Learning", "arxiv"),
        ("cs.NE", "Neural and Evolutionary Computing", "arxiv"),
        ("cs.RO", "Robotics", "arxiv"),
        ("cs.IR", "Information Retrieval", "arxiv"),
        ("cs.HC", "Human-Computer Interaction", "arxiv"),
        ("cs.CR", "Cryptography and Security", "arxiv"),
        ("cs.DS", "Data Structures and Algorithms", "arxiv"),
        
        # Statistics
        ("stat.ML", "Machine Learning (Statistics)", "arxiv"),
        ("stat.AP", "Applications", "arxiv"),
        ("stat.CO", "Computation", "arxiv"),
        ("stat.ME", "Methodology", "arxiv"),
        ("stat.TH", "Statistics Theory", "arxiv"),
        
        # Mathematics
        ("math.ST", "Statistics Theory", "arxiv"),
        ("math.PR", "Probability", "arxiv"),
        ("math.OC", "Optimization and Control", "arxiv"),
        ("math.NA", "Numerical Analysis", "arxiv"),
        ("math.IT", "Information Theory", "arxiv"),
        
        # Physics
        ("physics.data-an", "Data Analysis, Statistics and Probability", "arxiv"),
        ("physics.comp-ph", "Computational Physics", "arxiv"),
        ("physics.soc-ph", "Physics and Society", "arxiv"),
        
        # Quantitative Biology
        ("q-bio.QM", "Quantitative Methods", "arxiv"),
        ("q-bio.GN", "Genomics", "arxiv"),
        ("q-bio.BM", "Biomolecules", "arxiv"),
        ("q-bio.NC", "Neurons and Cognition", "arxiv"),
        
        # Economics
        ("econ.EM", "Econometrics", "arxiv"),
        ("econ.TH", "Theoretical Economics", "arxiv")
    ]
    
    # Semantic Scholar categories
    semantic_scholar_categories = [
        # Social Sciences
        ("Psychology", "Psychology and Cognitive Science", "semantic_scholar"),
        ("Sociology", "Sociology and Social Science", "semantic_scholar"),
        ("Political Science", "Political Science and Public Policy", "semantic_scholar"),
        ("Anthropology", "Anthropology and Cultural Studies", "semantic_scholar"),
        
        # Humanities
        ("Philosophy", "Philosophy and Ethics", "semantic_scholar"),
        ("History", "History and Historical Studies", "semantic_scholar"),
        ("Linguistics", "Linguistics and Language Studies", "semantic_scholar"),
        ("Literature", "Literature and Literary Studies", "semantic_scholar"),
        
        # Business and Economics
        ("Business", "Business and Management", "semantic_scholar"),
        ("Economics", "Economics and Economic Policy", "semantic_scholar"),
        ("Management", "Management and Organizational Behavior", "semantic_scholar"),
        ("Finance", "Finance and Financial Studies", "semantic_scholar"),
        
        # Other Fields
        ("Education", "Education and Pedagogy", "semantic_scholar"),
        ("Law", "Law and Legal Studies", "semantic_scholar"),
        ("Cognitive Science", "Cognitive Science and Neuroscience", "semantic_scholar"),
    ]
    
    categories.extend(arxiv_categories)
    categories.extend(semantic_scholar_categories)
    return categories


def get_popular_arxiv_categories() -> List[Tuple[str, str]]:
    """Get a list of popular arXiv categories for random selection.
    
    Returns:
        List of tuples containing (category_code, description) for popular
        arXiv categories across multiple academic disciplines.
    """
    all_categories = get_all_available_categories()
    return [(code, desc) for code, desc, source in all_categories if source == "arxiv"]


def get_semantic_scholar_categories() -> List[Tuple[str, str]]:
    """Get a list of Semantic Scholar categories.
    
    Returns:
        List of tuples containing (category_code, description) for
        Semantic Scholar categories.
    """
    all_categories = get_all_available_categories()
    return [(code, desc) for code, desc, source in all_categories if source == "semantic_scholar"]


def is_arxiv_category(category: str) -> bool:
    """Check if a category belongs to ArXiv.
    
    Args:
        category: Category code to check.
        
    Returns:
        True if category is an ArXiv category, False otherwise.
    """
    arxiv_categories = get_popular_arxiv_categories()
    return any(category == code for code, _ in arxiv_categories)


def is_semantic_scholar_category(category: str) -> bool:
    """Check if a category belongs to Semantic Scholar.
    
    Args:
        category: Category code to check.
        
    Returns:
        True if category is a Semantic Scholar category, False otherwise.
    """
    semantic_categories = get_semantic_scholar_categories()
    return any(category == code for code, _ in semantic_categories)


def select_random_category() -> Tuple[str, str]:
    """Select a random category from the popular categories list.
    
    Returns:
        Tuple containing (category_code, description) for a randomly
        selected arXiv category.
    """
    categories = get_popular_arxiv_categories()
    return random.choice(categories)


def print_available_categories() -> None:
    """Print all available categories from both ArXiv and Semantic Scholar."""
    all_categories = get_all_available_categories()
    
    print("Available Categories")
    print("=" * 60)
    
    # Group ArXiv categories by academic field
    arxiv_fields = {
        "Computer Science": [],
        "Statistics": [],
        "Mathematics": [],
        "Physics": [],
        "Quantitative Biology": [],
        "Economics (ArXiv)": []
    }
    
    # Group Semantic Scholar categories
    semantic_fields = {
        "Social Sciences": [],
        "Humanities": [],
        "Business & Economics": [],
        "Other Fields": []
    }
    
    for code, description, source in all_categories:
        if source == "arxiv":
            if code.startswith("cs."):
                arxiv_fields["Computer Science"].append((code, description))
            elif code.startswith("stat."):
                arxiv_fields["Statistics"].append((code, description))
            elif code.startswith("math."):
                arxiv_fields["Mathematics"].append((code, description))
            elif code.startswith("physics."):
                arxiv_fields["Physics"].append((code, description))
            elif code.startswith("q-bio."):
                arxiv_fields["Quantitative Biology"].append((code, description))
            elif code.startswith("econ."):
                arxiv_fields["Economics (ArXiv)"].append((code, description))
        else:  # semantic_scholar
            if code in ["Psychology", "Sociology", "Political Science", "Anthropology"]:
                semantic_fields["Social Sciences"].append((code, description))
            elif code in ["Philosophy", "History", "Linguistics", "Literature"]:
                semantic_fields["Humanities"].append((code, description))
            elif code in ["Business", "Economics", "Management", "Finance"]:
                semantic_fields["Business & Economics"].append((code, description))
            else:
                semantic_fields["Other Fields"].append((code, description))
    
    # Print ArXiv categories
    print("\nArXiv Categories:")
    print("-" * 40)
    for field, field_categories in arxiv_fields.items():
        if field_categories:
            print(f"\n{field}:")
            for code, description in field_categories:
                print(f"   {code:<20} - {description}")
    
    # Print Semantic Scholar categories
    print("\n\nSemantic Scholar Categories:")
    print("-" * 40)
    for field, field_categories in semantic_fields.items():
        if field_categories:
            print(f"\n{field}:")
            for code, description in field_categories:
                print(f"   {code:<20} - {description}")
    
    arxiv_count = len([c for c in all_categories if c[2] == "arxiv"])
    semantic_count = len([c for c in all_categories if c[2] == "semantic_scholar"])
    print(f"\nTotal categories: {len(all_categories)} (ArXiv: {arxiv_count}, Semantic Scholar: {semantic_count})")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Academic Paper Analysis Pipeline - Collect and analyze "
                   "academic papers from ArXiv and Semantic Scholar. "
                   "Automatically selects the appropriate scraper based on category.",
        epilog="""
Examples:
  # ArXiv categories (technical/scientific)
  python main.py --category cs.AI --max-papers 100
  python main.py --category Psychology --max-papers 50
  
  # Query-based (auto-selects scraper)
  python main.py --query "machine learning" --max-papers 75
  python main.py --query "behavioral economics" --max-papers 50
  
  # List all available categories
  python main.py --list-categories
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Search query for papers (auto-selects ArXiv or Semantic Scholar based on content)"
    )
    
    parser.add_argument(
        "--category", "-c",
        type=str,
        help="Category/field code (ArXiv: cs.AI, cs.LG, etc. | Semantic Scholar: Psychology, Philosophy, etc.)"
    )
    
    parser.add_argument(
        "--random", "-r",
        action="store_true",
        help="Randomly select a category from popular ArXiv categories"
    )
    
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available categories from both ArXiv and Semantic Scholar"
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
    
    # Determine which scraper to use based on category
    if category:
        if is_arxiv_category(category):
            run_paper_collection_pipeline(
                query=args.query,
                category=category,
                max_papers=args.max_papers
            )
        elif is_semantic_scholar_category(category):
            run_semantic_scholar_collection_pipeline(
                field=category,
                query=args.query,
                max_papers=args.max_papers
            )
        else:
            print(f"Unknown category '{category}'. Use --list-categories to see available options.")
            return
    elif args.query:
        # Auto-select scraper based on query content
        query_lower = args.query.lower()
        social_science_keywords = [
            'psychology', 'philosophy', 'business', 'economics', 'sociology',
            'political', 'anthropology', 'education', 'law', 'history',
            'linguistics', 'literature', 'cognitive science', 'behavioral',
            'social', 'ethics', 'management', 'finance'
        ]
        
        if any(keyword in query_lower for keyword in social_science_keywords):
            run_semantic_scholar_collection_pipeline(
                field=None,
                query=args.query,
                max_papers=args.max_papers
            )
        else:
            run_paper_collection_pipeline(
                query=args.query,
                category=None,
                max_papers=args.max_papers
            )
    else:
        # Default to ArXiv machine learning
        run_paper_collection_pipeline(
            query="machine learning",
            category=None,
            max_papers=args.max_papers
        )


if __name__ == "__main__":
    main()
