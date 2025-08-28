"""Training script for classical machine learning classifiers.

This script trains various classical ML algorithms for AI vs Human text detection
using the same pipeline as the neural network approach.
"""
import sys
import os
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.classical_classifiers import ClassicalTextClassifier, compare_classifiers


def main():
    """Main training function for classical classifiers."""
    parser = argparse.ArgumentParser(description='Train classical ML classifiers for AI vs Human text detection')
    
    # File-based arguments
    parser.add_argument('--human-file', type=str, required=True,
                       help='Path to JSONL file containing human-written texts')
    parser.add_argument('--ai-file', type=str, required=True,
                       help='Path to JSONL file containing AI-generated texts')
    
    # Classifier selection
    parser.add_argument('--classifier', type=str, 
                       choices=['random_forest', 'svm', 'logistic_regression', 'gradient_boosting', 
                               'naive_bayes', 'knn', 'decision_tree', 'adaboost', 'all'],
                       default='random_forest',
                       help='Type of classifier to train (default: random_forest)')
    
    # Model configuration
    parser.add_argument('--model-path', type=str, default='models/classical_classifier',
                       help='Path to save the trained model (default: models/classical_classifier)')
    parser.add_argument('--max-features', type=int, default=15000,
                       help='Maximum number of TF-IDF features (default: 15000)')
    parser.add_argument('--ngram-min', type=int, default=1,
                       help='Minimum n-gram size (default: 1)')
    parser.add_argument('--ngram-max', type=int, default=3,
                       help='Maximum n-gram size (default: 3)')
    
    # Training configuration
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for testing (default: 0.2)')
    parser.add_argument('--validation-size', type=float, default=0.15,
                       help='Proportion of training data for validation (default: 0.15)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    
    # Hyperparameter tuning
    parser.add_argument('--no-hyperparameter-tuning', action='store_true',
                       help='Disable hyperparameter tuning (use default parameters)')
    
    # Output options
    parser.add_argument('--save-plots', action='store_true',
                       help='Save confusion matrix plots to files')
    parser.add_argument('--plot-dir', type=str, default='plots',
                       help='Directory to save plots (default: plots)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed training information')
    
    args = parser.parse_args()
    
    # Validate files
    if not os.path.exists(args.human_file):
        print(f"Error: Human text file '{args.human_file}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(args.ai_file):
        print(f"Error: AI text file '{args.ai_file}' does not exist.")
        sys.exit(1)
    
    print(f"Training {args.classifier} classifier")
    print(f"Human file: {args.human_file}")
    print(f"AI file: {args.ai_file}")
    
    try:
        if args.classifier == 'all':
            # Compare all classifiers
            print("Comparing all classifiers")
            results = compare_classifiers(
                human_file=args.human_file,
                ai_file=args.ai_file,
                test_size=args.test_size,
                validation_size=args.validation_size
            )
            
            # Save best performing model
            best_classifier = None
            best_accuracy = 0.0
            
            for classifier_type, result in results.items():
                if 'error' not in result and result['test_accuracy'] > best_accuracy:
                    best_accuracy = result['test_accuracy']
                    best_classifier = classifier_type
            
            if best_classifier:
                print(f"Best classifier: {best_classifier} (accuracy: {best_accuracy:.4f})")
                print("Training best classifier")
                
                classifier = ClassicalTextClassifier(
                    classifier_type=best_classifier,
                    max_features=args.max_features,
                    ngram_range=(args.ngram_min, args.ngram_max),
                    use_hyperparameter_tuning=not args.no_hyperparameter_tuning
                )
                
                result = classifier.train_from_files(
                    human_file=args.human_file,
                    ai_file=args.ai_file,
                    test_size=args.test_size,
                    validation_size=args.validation_size,
                    cv_folds=args.cv_folds
                )
                
                classifier.save_model(args.model_path)
                
                if args.save_plots:
                    plot_dir = Path(args.plot_dir)
                    plot_dir.mkdir(exist_ok=True)
                    confusion_path = plot_dir / f"confusion_matrix_{best_classifier}.png"
                    classifier.plot_confusion_matrix(result['confusion_matrix'], str(confusion_path))
        
        else:
            # Train single classifier
            classifier = ClassicalTextClassifier(
                classifier_type=args.classifier,
                max_features=args.max_features,
                ngram_range=(args.ngram_min, args.ngram_max),
                use_hyperparameter_tuning=not args.no_hyperparameter_tuning
            )
            
            result = classifier.train_from_files(
                human_file=args.human_file,
                ai_file=args.ai_file,
                test_size=args.test_size,
                validation_size=args.validation_size,
                cv_folds=args.cv_folds
            )
            
            # Save the model
            print(f"Saving model to {args.model_path}")
            classifier.save_model(args.model_path)
            
            # Save plots if requested
            if args.save_plots:
                plot_dir = Path(args.plot_dir)
                plot_dir.mkdir(exist_ok=True)
                confusion_path = plot_dir / f"confusion_matrix_{args.classifier}.png"
                classifier.plot_confusion_matrix(result['confusion_matrix'], str(confusion_path))
            
            # Print results
            print("Training complete")
            print(f"Classifier: {args.classifier}")
            print(f"Cross-validation accuracy: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")
            print(f"Test accuracy: {result['test_accuracy']:.4f}")
            print(f"Test precision: {result['test_precision']:.4f}")
            print(f"Test recall: {result['test_recall']:.4f}")
            print(f"Test F1-score: {result['test_f1']:.4f}")
            if 'test_auc' in result:
                print(f"Test AUC: {result['test_auc']:.4f}")
            print(f"Total features used: {result['feature_count']}")
            print(f"Model saved to: {args.model_path}")
            
            if args.save_plots:
                print(f"Plots saved to: {args.plot_dir}")
    
    except Exception as e:
        print(f"Error during training: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
