"""Training script for ensemble machine learning classifiers.

This script trains various ensemble ML algorithms for AI vs Human text detection
using the same pipeline as the neural network approach.
"""
import sys
import os
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.ensemble_classifiers import EnsembleTextClassifier, compare_ensembles


def main():
    """Main training function for ensemble classifiers."""
    parser = argparse.ArgumentParser(description='Train ensemble ML classifiers for AI vs Human text detection')
    
    # File-based arguments
    parser.add_argument('--human-file', type=str, required=True,
                       help='Path to JSONL file containing human-written texts')
    parser.add_argument('--ai-file', type=str, required=True,
                       help='Path to JSONL file containing AI-generated texts')
    
    # Ensemble selection
    parser.add_argument('--ensemble', type=str, 
                       choices=['voting', 'bagging', 'stacking', 'xgboost', 'lightgbm', 
                               'catboost', 'extra_trees', 'custom_ensemble', 'all'],
                       default='voting',
                       help='Type of ensemble to train (default: voting)')
    
    # Model configuration
    parser.add_argument('--model-path', type=str, default='models/ensemble_classifier',
                       help='Path to save the trained model (default: models/ensemble_classifier)')
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
    
    print("=== Ensemble ML Classifiers Training ===")
    print(f"Human text file: {args.human_file}")
    print(f"AI text file: {args.ai_file}")
    print(f"Ensemble: {args.ensemble}")
    print(f"Model save path: {args.model_path}")
    print(f"Max features: {args.max_features}")
    print(f"N-gram range: ({args.ngram_min}, {args.ngram_max})")
    print(f"Hyperparameter tuning: {not args.no_hyperparameter_tuning}")
    print("=" * 60)
    
    try:
        if args.ensemble == 'all':
            # Compare all ensemble methods
            print("Training and comparing all ensemble classifiers...")
            results = compare_ensembles(
                human_file=args.human_file,
                ai_file=args.ai_file,
                test_size=args.test_size,
                validation_size=args.validation_size
            )
            
            # Save best performing model
            best_ensemble = None
            best_accuracy = 0.0
            
            for ensemble_type, result in results.items():
                if 'error' not in result and result['test_accuracy'] > best_accuracy:
                    best_accuracy = result['test_accuracy']
                    best_ensemble = ensemble_type
            
            if best_ensemble:
                print(f"\nBest performing ensemble: {best_ensemble} (accuracy: {best_accuracy:.4f})")
                print(f"Training and saving best ensemble...")
                
                classifier = EnsembleTextClassifier(
                    ensemble_type=best_ensemble,
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
                    confusion_path = plot_dir / f"confusion_matrix_{best_ensemble}_ensemble.png"
                    classifier.plot_confusion_matrix(result['confusion_matrix'], str(confusion_path))
        
        else:
            # Train single ensemble
            classifier = EnsembleTextClassifier(
                ensemble_type=args.ensemble,
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
            print(f"\nSaving {args.ensemble} ensemble model to {args.model_path}...")
            classifier.save_model(args.model_path)
            
            # Save plots if requested
            if args.save_plots:
                plot_dir = Path(args.plot_dir)
                plot_dir.mkdir(exist_ok=True)
                confusion_path = plot_dir / f"confusion_matrix_{args.ensemble}_ensemble.png"
                classifier.plot_confusion_matrix(result['confusion_matrix'], str(confusion_path))
            
            # Print final summary
            print("\n" + "=" * 60)
            print("ENSEMBLE TRAINING COMPLETE")
            print("=" * 60)
            print(f"Ensemble: {args.ensemble}")
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
            
            # Performance analysis
            if result['test_accuracy'] >= 0.90:
                print("🎉 EXCELLENT: Ensemble achieved 90%+ accuracy!")
                print("   Ensemble methods often provide the best performance!")
            elif result['test_accuracy'] >= 0.85:
                print("✅ GOOD: Ensemble achieved target accuracy of 85%+")
            elif result['test_accuracy'] >= 0.80:
                print("⚠️  FAIR: Ensemble achieved 80%+ accuracy")
            else:
                print("❌ POOR: Ensemble accuracy is below 80%")
                print("   Consider: collecting more diverse data, feature engineering, or different ensemble combinations")
            
            # Ensemble-specific insights
            if args.ensemble == 'voting':
                print("\n📊 Voting Ensemble Insights:")
                print("   - Combines predictions from multiple diverse classifiers")
                print("   - Generally robust and stable performance")
                print("   - Good baseline ensemble method")
            elif args.ensemble == 'stacking':
                print("\n📊 Stacking Ensemble Insights:")
                print("   - Uses meta-learner to combine base classifier predictions")
                print("   - Often achieves highest accuracy but more complex")
                print("   - May be prone to overfitting with small datasets")
            elif args.ensemble in ['xgboost', 'lightgbm', 'catboost']:
                print(f"\n📊 {args.ensemble.upper()} Insights:")
                print("   - Gradient boosting method, excellent for structured data")
                print("   - Built-in regularization and feature importance")
                print("   - Often top performer in ML competitions")
            elif args.ensemble == 'custom_ensemble':
                print("\n📊 Custom Ensemble Insights:")
                print("   - Combines multiple boosting and tree methods")
                print("   - Leverages strengths of different algorithm families")
                print("   - Often provides best overall performance")
            
            print(f"\nNext steps:")
            print(f"1. Test the ensemble: python scripts/predict_text.py --model-path {args.model_path} --ensemble-type {args.ensemble} --interactive")
            print(f"2. Compare with neural network and classical ML performance")
            print(f"3. Consider hyperparameter tuning if not already enabled")
            print(f"4. Analyze feature importance if supported by the ensemble")
    
    except Exception as e:
        print(f"Error during ensemble training: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
