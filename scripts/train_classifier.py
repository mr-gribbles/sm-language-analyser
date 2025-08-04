#!/usr/bin/env python3
"""Improved training script for the AI vs Human text classifier.

This script includes better hyperparameters, regularization techniques,
and options to address overfitting and improve accuracy.
"""
import sys
import os
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.text_classifier import AIHumanTextClassifier


def main():
    """Main training function with improved defaults."""
    parser = argparse.ArgumentParser(description='Train AI vs Human text classifier (improved version)')
    
    # File-based arguments
    parser.add_argument('--human-file', type=str,
                       help='Path to JSONL file containing human-written texts')
    parser.add_argument('--ai-file', type=str,
                       help='Path to JSONL file containing AI-generated texts')
    
    # Directory-based arguments (legacy)
    parser.add_argument('--corpus-dir', type=str, default='corpora',
                       help='Directory containing corpus files (default: corpora)')
    
    # Model configuration (improved defaults)
    parser.add_argument('--model-path', type=str, default='models/ai_human_classifier_improved',
                       help='Path to save the trained model (default: models/ai_human_classifier_improved)')
    parser.add_argument('--max-features', type=int, default=12000,
                       help='Maximum number of TF-IDF features (default: 12000)')
    parser.add_argument('--ngram-min', type=int, default=1,
                       help='Minimum n-gram size (default: 1)')
    parser.add_argument('--ngram-max', type=int, default=3,
                       help='Maximum n-gram size (default: 3)')
    
    # Training configuration (improved defaults)
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                       help='Learning rate (default: 0.0005)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (default: 20)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for testing (default: 0.2)')
    parser.add_argument('--validation-size', type=float, default=0.15,
                       help='Proportion of training data for validation (default: 0.15)')
    
    # Regularization options
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for regularization (default: 1e-4)')
    
    # Output options
    parser.add_argument('--save-plots', action='store_true',
                       help='Save training plots to files')
    parser.add_argument('--plot-dir', type=str, default='plots',
                       help='Directory to save plots (default: plots)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed training information')
    
    args = parser.parse_args()
    
    # Determine which approach to use
    use_specific_files = args.human_file and args.ai_file
    
    if use_specific_files:
        # Validate specific files
        if not os.path.exists(args.human_file):
            print(f"Error: Human text file '{args.human_file}' does not exist.")
            sys.exit(1)
        
        if not os.path.exists(args.ai_file):
            print(f"Error: AI text file '{args.ai_file}' does not exist.")
            sys.exit(1)
        
        print("=== AI vs Human Text Classifier Training (Improved) ===")
        print(f"Human text file: {args.human_file}")
        print(f"AI text file: {args.ai_file}")
        
    else:
        # Validate directory approach
        if not os.path.exists(args.corpus_dir):
            print(f"Error: Corpus directory '{args.corpus_dir}' does not exist.")
            print("Make sure you have generated corpus data using the pipeline first.")
            sys.exit(1)
        
        # Check if corpus has both human and AI data
        original_dir = Path(args.corpus_dir) / "original_only"
        rewritten_dir = Path(args.corpus_dir) / "rewritten_pairs"
        
        if not original_dir.exists() or not any(original_dir.glob("*.jsonl")):
            print(f"Error: No human text data found in {original_dir}")
            print("Run the pipeline with rewrite=False to generate human text data.")
            sys.exit(1)
        
        if not rewritten_dir.exists() or not any(rewritten_dir.glob("*.jsonl")):
            print(f"Error: No AI text data found in {rewritten_dir}")
            print("Run the pipeline with rewrite=True to generate AI text data.")
            sys.exit(1)
        
        print("=== AI vs Human Text Classifier Training (Improved) ===")
        print(f"Corpus directory: {args.corpus_dir}")
    
    print(f"Model save path: {args.model_path}")
    print(f"Max features: {args.max_features}")
    print(f"N-gram range: ({args.ngram_min}, {args.ngram_max})")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Patience: {args.patience}")
    print(f"Dropout: {args.dropout}")
    print(f"Weight decay: {args.weight_decay}")
    print("=" * 60)
    
    # Initialize classifier with improved parameters
    classifier = AIHumanTextClassifier(
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max)
    )
    
    try:
        # Train the model using appropriate method
        if use_specific_files:
            results = classifier.train_from_files(
                human_file=args.human_file,
                ai_file=args.ai_file,
                test_size=args.test_size,
                validation_size=args.validation_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience
            )
        else:
            results = classifier.train(
                corpus_dir=args.corpus_dir,
                test_size=args.test_size,
                validation_size=args.validation_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience
            )
        
        # Save the model
        print(f"\nSaving model to {args.model_path}...")
        classifier.save_model(args.model_path)
        
        # Save plots if requested
        if args.save_plots:
            plot_dir = Path(args.plot_dir)
            plot_dir.mkdir(exist_ok=True)
            
            history_path = plot_dir / "training_history_improved.png"
            confusion_path = plot_dir / "confusion_matrix_improved.png"
            
            print(f"Saving plots to {plot_dir}...")
            classifier.plot_training_history(results['history'], str(history_path))
            classifier.plot_confusion_matrix(results['confusion_matrix'], str(confusion_path))
        
        # Print final summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Final Test Precision: {results['test_precision']:.4f}")
        print(f"Final Test Recall: {results['test_recall']:.4f}")
        print(f"Model saved to: {args.model_path}")
        
        if args.save_plots:
            print(f"Plots saved to: {args.plot_dir}")
        
        # Performance analysis
        train_acc = results['history']['train_acc'][-1] if results['history']['train_acc'] else 0
        val_acc = results['history']['val_acc'][-1] if results['history']['val_acc'] else 0
        
        print(f"\nPerformance Analysis:")
        print(f"Final Training Accuracy: {train_acc:.4f}")
        print(f"Final Validation Accuracy: {val_acc:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        
        if train_acc - val_acc > 0.1:
            print("⚠️  WARNING: Model may be overfitting (train >> validation accuracy)")
            print("   Consider: reducing model complexity, increasing dropout, or collecting more data")
        elif results['test_accuracy'] < 0.85:
            print("⚠️  WARNING: Test accuracy is below 85%")
            print("   Consider: analyzing data quality, adjusting hyperparameters, or collecting more data")
        else:
            print("✅ Model performance looks good!")
        
        print("\nNext steps:")
        print("1. Test the model: python scripts/predict_text.py --interactive")
        print("2. Analyze data quality: python scripts/analyze_training_data.py --human-file <file> --ai-file <file>")
        print("3. If accuracy is low, try collecting more diverse training data")
        
    except Exception as e:
        print(f"Error during training: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
