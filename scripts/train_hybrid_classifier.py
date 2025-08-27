"""Training script for the Hybrid text classifier.
"""
import sys
import os
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.hybrid_classifier import HybridTextClassifier


def main():
    """Main training function with Hybrid classifier."""
    parser = argparse.ArgumentParser(description='Train AI vs Human text classifier (Hybrid version)')
    
    # File-based arguments
    parser.add_argument('--human-file', type=str,
                       help='Path to JSONL file containing human-written texts')
    parser.add_argument('--ai-file', type=str,
                       help='Path to JSONL file containing AI-generated texts')
    
    # Model configuration
    parser.add_argument('--model-path', type=str, default='models/ai_human_classifier_hybrid',
                       help='Path to save the trained model (default: models/ai_human_classifier_hybrid)')
    
    # Output options
    parser.add_argument('--save-plots', action='store_true',
                       help='Save training plots to files')
    parser.add_argument('--plot-dir', type=str, default='plots',
                       help='Directory to save plots (default: plots)')
    
    args = parser.parse_args()
    
    # Validate required files
    if not args.human_file or not args.ai_file:
        print("Error: Both --human-file and --ai-file are required.")
        sys.exit(1)
    
    if not os.path.exists(args.human_file):
        print(f"Error: Human text file '{args.human_file}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(args.ai_file):
        print(f"Error: AI text file '{args.ai_file}' does not exist.")
        sys.exit(1)
    
    print("=== Hybrid AI vs Human Text Classifier Training ===")
    print(f"Human text file: {args.human_file}")
    print(f"AI text file: {args.ai_file}")
    print(f"Model save path: {args.model_path}")
    print("=" * 60)
    
    # Initialize classifier
    classifier = HybridTextClassifier()
    
    try:
        # Train the model
        results = classifier.train_from_files(
            human_file=args.human_file,
            ai_file=args.ai_file,
        )
        
        # Save the model
        print(f"\nSaving model to {args.model_path}...")
        classifier.save_model(args.model_path)
        
        # Save plots if requested
        if args.save_plots:
            plot_dir = Path(args.plot_dir)
            plot_dir.mkdir(exist_ok=True)
            
            confusion_path = plot_dir / "confusion_matrix_hybrid.png"
            
            print(f"Saving plots to {plot_dir}...")
            classifier.plot_confusion_matrix(results['confusion_matrix'], str(confusion_path))
        
        # Print final summary
        print("\n" + "=" * 60)
        print("HYBRID TRAINING COMPLETE")
        print("=" * 60)
        print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Final Test Precision: {results['test_precision']:.4f}")
        print(f"Final Test Recall: {results['test_recall']:.4f}")
        print(f"Model saved to: {args.model_path}")
        
        if args.save_plots:
            print(f"Plots saved to: {args.plot_dir}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
