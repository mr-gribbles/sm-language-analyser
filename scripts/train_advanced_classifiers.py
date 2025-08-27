#!/usr/bin/env python3
"""
Training script for advanced machine learning classifiers.

This script trains and evaluates advanced ML methods including Gaussian Process,
anomaly detection methods, discriminant analysis, and other specialized approaches.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.ml.advanced_classifiers import AdvancedTextClassifier, compare_advanced_classifiers


def main():
    parser = argparse.ArgumentParser(description='Train advanced ML classifiers for AI vs Human text detection')
    
    # Data files
    parser.add_argument('--human-file', type=str, required=True,
                       help='Path to JSONL file containing human-written texts')
    parser.add_argument('--ai-file', type=str, required=True,
                       help='Path to JSONL file containing AI-generated texts')
    
    # Model selection
    parser.add_argument('--classifier', type=str, default='gaussian_process',
                       choices=['gaussian_process', 'passive_aggressive', 'sgd_online', 
                               'one_class_svm', 'isolation_forest', 'nearest_centroid',
                               'lda', 'qda', 'cluster_based', 'label_propagation'],
                       help='Type of advanced classifier to train')
    
    # Training parameters
    parser.add_argument('--max-features', type=int, default=15000,
                       help='Maximum number of features for TF-IDF vectorization')
    parser.add_argument('--ngram-range', type=str, default='1,3',
                       help='N-gram range for TF-IDF (e.g., "1,3" for unigrams to trigrams)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--validation-size', type=float, default=0.15,
                       help='Proportion of training data to use for validation')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    # Hyperparameter tuning
    parser.add_argument('--no-hyperparameter-tuning', action='store_true',
                       help='Disable hyperparameter tuning (use default parameters)')
    
    # Model saving
    parser.add_argument('--save-model', type=str, default=None,
                       help='Path to save the trained model (without extension)')
    
    # Comparison mode
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all available advanced classifiers')
    parser.add_argument('--compare-classifiers', type=str, nargs='+', default=None,
                       help='List of specific classifiers to compare')
    
    # Output options
    parser.add_argument('--save-results', type=str, default=None,
                       help='Path to save results as JSON file')
    parser.add_argument('--plot-confusion-matrix', action='store_true',
                       help='Plot and display confusion matrix')
    parser.add_argument('--save-plots', type=str, default=None,
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Parse n-gram range
    ngram_range = tuple(map(int, args.ngram_range.split(',')))
    
    # Validate input files
    if not os.path.exists(args.human_file):
        print(f"Error: Human file not found: {args.human_file}")
        sys.exit(1)
    
    if not os.path.exists(args.ai_file):
        print(f"Error: AI file not found: {args.ai_file}")
        sys.exit(1)
    
    # Create output directories if needed
    if args.save_plots:
        Path(args.save_plots).mkdir(parents=True, exist_ok=True)
    
    if args.save_model:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
    
    # Comparison mode
    if args.compare_all or args.compare_classifiers:
        print("=== Advanced Classifier Comparison Mode ===")
        
        classifiers_to_compare = args.compare_classifiers if args.compare_classifiers else None
        
        results = compare_advanced_classifiers(
            human_file=args.human_file,
            ai_file=args.ai_file,
            classifiers=classifiers_to_compare,
            test_size=args.test_size,
            validation_size=args.validation_size
        )
        
        # Save comparison results
        if args.save_results:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for classifier_name, result in results.items():
                if 'error' not in result:
                    json_result = result.copy()
                    # Convert numpy arrays to lists
                    if 'cv_scores' in json_result:
                        json_result['cv_scores'] = json_result['cv_scores'].tolist()
                    if 'confusion_matrix' in json_result:
                        json_result['confusion_matrix'] = json_result['confusion_matrix'].tolist()
                    # Remove non-serializable items
                    json_result.pop('classification_report', None)
                    json_results[classifier_name] = json_result
                else:
                    json_results[classifier_name] = result
            
            with open(args.save_results, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\nComparison results saved to: {args.save_results}")
        
        return
    
    # Single classifier training mode
    print(f"=== Training {args.classifier.title()} Advanced Classifier ===")
    print(f"Human file: {args.human_file}")
    print(f"AI file: {args.ai_file}")
    print(f"Max features: {args.max_features}")
    print(f"N-gram range: {ngram_range}")
    print(f"Test size: {args.test_size}")
    print(f"Validation size: {args.validation_size}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Hyperparameter tuning: {not args.no_hyperparameter_tuning}")
    print("=" * 60)
    
    # Initialize classifier
    classifier = AdvancedTextClassifier(
        classifier_type=args.classifier,
        max_features=args.max_features,
        ngram_range=ngram_range,
        use_hyperparameter_tuning=not args.no_hyperparameter_tuning
    )
    
    # Train the classifier
    try:
        results = classifier.train_from_files(
            human_file=args.human_file,
            ai_file=args.ai_file,
            test_size=args.test_size,
            validation_size=args.validation_size,
            cv_folds=args.cv_folds
        )
        
        print(f"\n=== Training Complete ===")
        print(f"Final test accuracy: {results['test_accuracy']:.4f}")
        print(f"Final test F1-score: {results['test_f1']:.4f}")
        if 'test_auc' in results:
            print(f"Final test AUC: {results['test_auc']:.4f}")
        
        # Save model
        if args.save_model:
            classifier.save_model(args.save_model)
            print(f"Model saved to: {args.save_model}")
        
        # Plot confusion matrix
        if args.plot_confusion_matrix:
            save_path = None
            if args.save_plots:
                save_path = os.path.join(args.save_plots, f"{args.classifier}_confusion_matrix.png")
            
            classifier.plot_confusion_matrix(results['confusion_matrix'], save_path)
        
        # Save results
        if args.save_results:
            # Convert numpy arrays to lists for JSON serialization
            json_results = results.copy()
            if 'cv_scores' in json_results:
                json_results['cv_scores'] = json_results['cv_scores'].tolist()
            if 'confusion_matrix' in json_results:
                json_results['confusion_matrix'] = json_results['confusion_matrix'].tolist()
            # Remove non-serializable items
            json_results.pop('classification_report', None)
            
            with open(args.save_results, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"Results saved to: {args.save_results}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
