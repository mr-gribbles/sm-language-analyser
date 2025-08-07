"""Enhanced training script for the AI vs Human text classifier.

This script uses the enhanced classifier with improved feature extraction,
better neural architecture, and optimized hyperparameters to achieve
higher accuracy (target: 85%+).
"""
import sys
import os
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.text_classifier_enhanced import EnhancedAIHumanTextClassifier


def main():
    """Main training function with enhanced classifier."""
    parser = argparse.ArgumentParser(description='Train AI vs Human text classifier (enhanced version)')
    
    # File-based arguments
    parser.add_argument('--human-file', type=str,
                       help='Path to JSONL file containing human-written texts')
    parser.add_argument('--ai-file', type=str,
                       help='Path to JSONL file containing AI-generated texts')
    
    # Model configuration (enhanced defaults)
    parser.add_argument('--model-path', type=str, default='models/ai_human_classifier_enhanced',
                       help='Path to save the trained model (default: models/ai_human_classifier_enhanced)')
    parser.add_argument('--max-features', type=int, default=15000,
                       help='Maximum number of TF-IDF features (default: 15000)')
    parser.add_argument('--ngram-min', type=int, default=1,
                       help='Minimum n-gram size (default: 1)')
    parser.add_argument('--ngram-max', type=int, default=3,
                       help='Maximum n-gram size (default: 3)')
    
    # Training configuration (enhanced defaults)
    parser.add_argument('--epochs', type=int, default=150,
                       help='Maximum number of training epochs (default: 150)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for testing (default: 0.2)')
    parser.add_argument('--validation-size', type=float, default=0.15,
                       help='Proportion of training data for validation (default: 0.15)')
    
    # Enhanced regularization options
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
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
    
    # Validate required files
    if not args.human_file or not args.ai_file:
        print("Error: Both --human-file and --ai-file are required.")
        print("Example usage:")
        print("  python scripts/train_classifier_enhanced.py \\")
        print("    --human-file corpora/original_only/combined_original_only_20250807_083322.jsonl \\")
        print("    --ai-file corpora/rewritten_pairs/combined_rewritten_pairs_20250807_133558.jsonl \\")
        print("    --save-plots")
        sys.exit(1)
    
    if not os.path.exists(args.human_file):
        print(f"Error: Human text file '{args.human_file}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(args.ai_file):
        print(f"Error: AI text file '{args.ai_file}' does not exist.")
        sys.exit(1)
    
    print("=== Enhanced AI vs Human Text Classifier Training ===")
    print(f"Human text file: {args.human_file}")
    print(f"AI text file: {args.ai_file}")
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
    
    # Initialize enhanced classifier
    classifier = EnhancedAIHumanTextClassifier(
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max)
    )
    
    try:
        # Train the enhanced model
        results = classifier.train_from_files(
            human_file=args.human_file,
            ai_file=args.ai_file,
            test_size=args.test_size,
            validation_size=args.validation_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            dropout_rate=args.dropout,
            weight_decay=args.weight_decay
        )
        
        # Save the model
        print(f"\nSaving enhanced model to {args.model_path}...")
        classifier.save_model(args.model_path)
        
        # Save plots if requested
        if args.save_plots:
            plot_dir = Path(args.plot_dir)
            plot_dir.mkdir(exist_ok=True)
            
            history_path = plot_dir / "training_history_enhanced.png"
            confusion_path = plot_dir / "confusion_matrix_enhanced.png"
            
            print(f"Saving plots to {plot_dir}...")
            classifier.plot_training_history(results['history'], str(history_path))
            classifier.plot_confusion_matrix(results['confusion_matrix'], str(confusion_path))
        
        # Print final summary
        print("\n" + "=" * 60)
        print("ENHANCED TRAINING COMPLETE")
        print("=" * 60)
        print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Final Test Precision: {results['test_precision']:.4f}")
        print(f"Final Test Recall: {results['test_recall']:.4f}")
        print(f"Total Features Used: {results['feature_count']}")
        print(f"Model saved to: {args.model_path}")
        
        if args.save_plots:
            print(f"Plots saved to: {args.plot_dir}")
        
        # Performance analysis
        train_acc = results['history']['train_acc'][-1] if results['history']['train_acc'] else 0
        val_acc = results['history']['val_acc'][-1] if results['history']['val_acc'] else 0
        
        print(f"\nEnhanced Model Performance Analysis:")
        print(f"Final Training Accuracy: {train_acc:.4f}")
        print(f"Final Validation Accuracy: {val_acc:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        
        # Detailed analysis
        if train_acc - val_acc > 0.1:
            print("⚠️  WARNING: Model may be overfitting (train >> validation accuracy)")
            print("   Consider: reducing model complexity, increasing dropout, or collecting more data")
        elif results['test_accuracy'] < 0.85:
            print("⚠️  WARNING: Test accuracy is below 85%")
            print("   Consider: analyzing data quality, collecting more diverse training data")
            print("   The enhanced model should perform better than the basic version")
        elif results['test_accuracy'] >= 0.90:
            print("🎉 EXCELLENT: Model achieved 90%+ accuracy!")
            print("   This is excellent performance for AI vs Human text classification")
        elif results['test_accuracy'] >= 0.85:
            print("✅ GOOD: Model achieved target accuracy of 85%+")
            print("   This is solid performance for AI vs Human text classification")
        
        # Feature breakdown
        print(f"\nFeature Breakdown:")
        print(f"- Word-level TF-IDF features: ~{args.max_features // 2}")
        print(f"- Character-level TF-IDF features: ~{args.max_features // 2}")
        print(f"- Linguistic features: ~15")
        print(f"- Total features: {results['feature_count']}")
        
        print("\nKey Improvements in Enhanced Model:")
        print("✓ Multi-level feature extraction (word + character + linguistic)")
        print("✓ Deeper neural architecture (4 hidden layers)")
        print("✓ Advanced regularization techniques")
        print("✓ Focal loss for better class balance handling")
        print("✓ Cosine annealing learning rate scheduling")
        print("✓ Better hyperparameter defaults")
        
        print("\nNext steps:")
        print("1. Test the enhanced model: python scripts/predict_text.py --model-path models/ai_human_classifier_enhanced --interactive")
        print("2. Compare with basic model performance")
        print("3. If accuracy is still low, consider collecting more diverse training data")
        
    except Exception as e:
        print(f"Error during enhanced training: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
