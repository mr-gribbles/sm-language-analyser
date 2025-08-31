"""Memory-safe comprehensive comparison script for all ML methods.

This script trains and compares neural networks, classical ML, and ensemble methods
for AI vs Human text detection using the same training/testing pipeline, but with
memory-efficient processing to prevent system crashes.
"""
import sys
import os
import argparse
import json
import time
import gc
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.text_classifier import EnhancedAIHumanTextClassifier
from src.ml.classical_classifiers import ClassicalTextClassifier
from src.ml.ensemble_classifiers import EnsembleTextClassifier
from src.ml.sequential_classifier import SequentialTextClassifier
from src.ml.hybrid_classifier import HybridTextClassifier
from src.ml.deep_learning_classifiers import DeepLearningTextClassifier
from src.ml.probabilistic_classifiers import ProbabilisticTextClassifier
from src.ml.manifold_classifiers import ManifoldTextClassifier
from src.ml.advanced_classifiers import AdvancedTextClassifier
from src.ml.interpretable_classifiers import InterpretableTextClassifier


def train_single_method(method_type: str, method_name: str, human_file: str,
                       ai_file: str, test_size: float = 0.2,
                       validation_size: float = 0.15,
                       reduced_features: bool = True, reduced_cv: bool = True,
                       save_model: bool = False,
                       model_save_path: str = None) -> Dict[str, Any]:
    """Train a single method with memory-safe settings.
    
    Args:
        method_type: Type of method ('neural', 'classical', 'ensemble').
        method_name: Specific method name within the type.
        human_file: Path to JSONL file with human-written texts.
        ai_file: Path to JSONL file with AI-generated texts.
        test_size: Proportion of data for testing.
        validation_size: Proportion of training data for validation.
        reduced_features: Whether to use reduced feature set for memory.
        reduced_cv: Whether to use reduced cross-validation folds.
        save_model: Whether to save the trained model.
        model_save_path: Base path for saving models.
        
    Returns:
        Dictionary containing training results and metrics.
    """
    print(f"Training {method_type}: {method_name}")
    start_time = time.time()
    
    classifier = None  # Keep reference for saving
    
    try:
        # Force garbage collection before training
        gc.collect()
        
        if method_type == 'neural':
            classifier = EnhancedAIHumanTextClassifier(
                max_features=10000 if reduced_features else 15000,  # Reduce features to save memory
                ngram_range=(1, 2) if reduced_features else (1, 3)  # Reduce n-gram complexity
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size,
                epochs=100 if reduced_features else 150,  # Reduce epochs for faster training
                batch_size=16 if reduced_features else 32,  # Smaller batch size
                learning_rate=0.001,
                patience=10 if reduced_features else 15,  # Reduce patience
                dropout_rate=0.3,
                weight_decay=1e-4
            )
            
            training_time = time.time() - start_time
            
            # Calculate F1 score properly
            test_f1 = result.get('test_f1')
            if test_f1 is None:
                # Calculate F1 from precision and recall if not available
                precision = result.get('test_precision', 0.0)
                recall = result.get('test_recall', 0.0)
                if precision + recall > 0:
                    test_f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    test_f1 = 0.0
            
            # Save model automatically
            if save_model and model_save_path and classifier:
                try:
                    model_path = f"{model_save_path}_{method_name}"
                    classifier.save_model(model_path)
                    print(f"Saved neural network model to {model_path}")
                except Exception as e:
                    print(f"Failed to save neural network model: {e}")
            
            return {
                'method': f'{method_type.title()}: {method_name}',
                'test_accuracy': result['test_accuracy'],
                'test_precision': result['test_precision'],
                'test_recall': result['test_recall'],
                'test_f1': test_f1,
                'test_auc': result.get('test_auc', 0.0),
                'feature_count': result['feature_count'],
                'training_time': training_time,
                'cv_mean': 0.0,  # Neural network doesn't use CV
                'cv_std': 0.0,
                'best_params': {},
                'confusion_matrix': result['confusion_matrix'],
                'error': None
            }
            
        elif method_type == 'classical':
            classifier = ClassicalTextClassifier(
                classifier_type=method_name,
                max_features=10000 if reduced_features else 15000,
                ngram_range=(1, 2) if reduced_features else (1, 3),
                use_hyperparameter_tuning=not reduced_features  # Skip hyperparameter tuning for speed
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size,
                cv_folds=3 if reduced_cv else 5  # Reduce CV folds to save memory
            )
            
            training_time = time.time() - start_time
            result['method'] = f'{method_type.title()}: {method_name}'
            result['training_time'] = training_time
            
            # Save model automatically
            if save_model and model_save_path and classifier:
                try:
                    model_path = f"{model_save_path}_{method_name}"
                    classifier.save_model(model_path)
                    print(f"Saved classical model to {model_path}")
                except Exception as e:
                    print(f"Failed to save classical model: {e}")
            
            return result
            
        elif method_type == 'ensemble':
            classifier = EnsembleTextClassifier(
                ensemble_type=method_name,
                max_features=8000 if reduced_features else 15000,  # Even more reduced for ensembles
                ngram_range=(1, 2) if reduced_features else (1, 3),
                use_hyperparameter_tuning=not reduced_features  # Skip hyperparameter tuning for speed
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size,
                cv_folds=3 if reduced_cv else 5  # Reduce CV folds to save memory
            )
            
            training_time = time.time() - start_time
            result['method'] = f'{method_type.title()}: {method_name}'
            result['training_time'] = training_time
            
            # Save model automatically
            if save_model and model_save_path and classifier:
                try:
                    model_path = f"{model_save_path}_{method_name}"
                    classifier.save_model(model_path)
                    print(f"Saved ensemble model to {model_path}")
                except Exception as e:
                    print(f"Failed to save ensemble model: {e}")
            
            return result
            
        elif method_type == 'sequential':
            classifier = SequentialTextClassifier(
                vocab_size=15000 if reduced_features else 20000,
                max_len=256 if reduced_features else 512
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size,
                epochs=30 if reduced_features else 50
            )
            
            training_time = time.time() - start_time
            
            # Save model automatically
            if save_model and model_save_path and classifier:
                try:
                    model_path = f"{model_save_path}_{method_name}"
                    classifier.save_model(model_path)
                    print(f"Saved sequential model to {model_path}")
                except Exception as e:
                    print(f"Failed to save sequential model: {e}")
            
            return {
                'method': f'{method_type.title()}: {method_name}',
                'test_accuracy': result['test_accuracy'],
                'test_precision': result['test_precision'],
                'test_recall': result['test_recall'],
                'test_f1': result['test_f1'],
                'test_auc': 0.0,  # Sequential classifier doesn't compute AUC
                'feature_count': 0,  # Sequential uses embeddings, not traditional features
                'training_time': training_time,
                'cv_mean': 0.0,  # Sequential doesn't use CV
                'cv_std': 0.0,
                'best_params': {},
                'confusion_matrix': result['confusion_matrix'],
                'error': None
            }
            
        elif method_type == 'hybrid':
            classifier = HybridTextClassifier(
                vocab_size=15000 if reduced_features else 20000,
                max_len=256 if reduced_features else 512,
                max_features=8000 if reduced_features else 15000,
                ngram_range=(1, 2) if reduced_features else (1, 3)
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size,
                epochs=30 if reduced_features else 50
            )
            
            training_time = time.time() - start_time
            
            # Save model automatically
            if save_model and model_save_path and classifier:
                try:
                    model_path = f"{model_save_path}_{method_name}"
                    classifier.save_model(model_path)
                    print(f"Saved hybrid model to {model_path}")
                except Exception as e:
                    print(f"Failed to save hybrid model: {e}")
            
            return {
                'method': f'{method_type.title()}: {method_name}',
                'test_accuracy': result['test_accuracy'],
                'test_precision': result['test_precision'],
                'test_recall': result['test_recall'],
                'test_f1': result['test_f1'],
                'test_auc': 0.0,  # Hybrid classifier doesn't compute AUC
                'feature_count': 0,  # Hybrid uses both embeddings and features
                'training_time': training_time,
                'cv_mean': 0.0,  # Hybrid doesn't use CV
                'cv_std': 0.0,
                'best_params': {},
                'confusion_matrix': result['confusion_matrix'],
                'error': None
            }
            
        elif method_type == 'deep_learning':
            classifier = DeepLearningTextClassifier(
                model_type=method_name,
                max_features=10000 if reduced_features else 15000,
                ngram_range=(1, 2) if reduced_features else (1, 3)
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size,
                epochs=50 if reduced_features else 100,
                batch_size=16 if reduced_features else 32
            )
            
            training_time = time.time() - start_time
            
            # Save model automatically
            if save_model and model_save_path and classifier:
                try:
                    model_path = f"{model_save_path}_{method_name}"
                    classifier.save_model(model_path)
                    print(f"Saved deep learning model to {model_path}")
                except Exception as e:
                    print(f"Failed to save deep learning model: {e}")
            
            return {
                'method': f'{method_type.title()}: {method_name}',
                'test_accuracy': result['test_accuracy'],
                'test_precision': result['test_precision'],
                'test_recall': result['test_recall'],
                'test_f1': result['test_f1'],
                'test_auc': 0.0,  # Deep learning classifier doesn't compute AUC
                'feature_count': 0,  # Deep learning uses embeddings
                'training_time': training_time,
                'cv_mean': 0.0,  # Deep learning doesn't use CV
                'cv_std': 0.0,
                'best_params': {},
                'confusion_matrix': result['confusion_matrix'],
                'error': None
            }
            
        elif method_type == 'probabilistic':
            classifier = ProbabilisticTextClassifier(
                classifier_type=method_name,
                max_features=10000 if reduced_features else 15000,
                ngram_range=(1, 2) if reduced_features else (1, 3),
                use_hyperparameter_tuning=not reduced_features
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size,
                cv_folds=3 if reduced_cv else 5
            )
            
            training_time = time.time() - start_time
            result['method'] = f'{method_type.title()}: {method_name}'
            result['training_time'] = training_time
            
            # Save model automatically
            if save_model and model_save_path and classifier:
                try:
                    model_path = f"{model_save_path}_{method_name}"
                    classifier.save_model(model_path)
                    print(f"Saved probabilistic model to {model_path}")
                except Exception as e:
                    print(f"Failed to save probabilistic model: {e}")
            
            return result
            
        elif method_type == 'manifold':
            classifier = ManifoldTextClassifier(
                manifold_type=method_name,
                max_features=8000 if reduced_features else 15000,  # Reduced for manifold methods
                ngram_range=(1, 2) if reduced_features else (1, 3),
                use_hyperparameter_tuning=not reduced_features
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size,
                cv_folds=3 if reduced_cv else 5
            )
            
            training_time = time.time() - start_time
            result['method'] = f'{method_type.title()}: {method_name}'
            result['training_time'] = training_time
            
            # Save model automatically
            if save_model and model_save_path and classifier:
                try:
                    model_path = f"{model_save_path}_{method_name}"
                    classifier.save_model(model_path)
                    print(f"Saved manifold model to {model_path}")
                except Exception as e:
                    print(f"Failed to save manifold model: {e}")
            
            return result
            
        elif method_type == 'advanced':
            classifier = AdvancedTextClassifier(
                classifier_type=method_name,
                max_features=10000 if reduced_features else 15000,
                ngram_range=(1, 2) if reduced_features else (1, 3),
                use_hyperparameter_tuning=not reduced_features
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size,
                cv_folds=3 if reduced_cv else 5
            )
            
            training_time = time.time() - start_time
            result['method'] = f'{method_type.title()}: {method_name}'
            result['training_time'] = training_time
            
            # Save model automatically
            if save_model and model_save_path and classifier:
                try:
                    model_path = f"{model_save_path}_{method_name}"
                    classifier.save_model(model_path)
                    print(f"Saved advanced model to {model_path}")
                except Exception as e:
                    print(f"Failed to save advanced model: {e}")
            
            return result
            
        elif method_type == 'interpretable':
            classifier = InterpretableTextClassifier(
                classifier_type=method_name,
                max_features=10000 if reduced_features else 15000,
                ngram_range=(1, 2) if reduced_features else (1, 3),
                use_hyperparameter_tuning=not reduced_features
            )
            
            result = classifier.train_from_files(
                human_file=human_file,
                ai_file=ai_file,
                test_size=test_size,
                validation_size=validation_size,
                cv_folds=3 if reduced_cv else 5
            )
            
            training_time = time.time() - start_time
            result['method'] = f'{method_type.title()}: {method_name}'
            result['training_time'] = training_time
            
            # Save model automatically
            if save_model and model_save_path and classifier:
                try:
                    model_path = f"{model_save_path}_{method_name}"
                    classifier.save_model(model_path)
                    print(f"Saved interpretable model to {model_path}")
                except Exception as e:
                    print(f"Failed to save interpretable model: {e}")
            
            return result
            
    except Exception as e:
        training_time = time.time() - start_time
        return {
            'method': f'{method_type.title()}: {method_name}',
            'training_time': training_time,
            'error': str(e)
        }
    finally:
        # Force garbage collection after each method
        gc.collect()


def main():
    """Main comparison function with memory-safe processing."""
    parser = argparse.ArgumentParser(description='Memory-safe comparison of all ML methods for AI vs Human text detection')
    
    # File-based arguments
    parser.add_argument('--human-file', type=str, required=True,
                       help='Path to JSONL file containing human-written texts')
    parser.add_argument('--ai-file', type=str, required=True,
                       help='Path to JSONL file containing AI-generated texts')
    
    # Training configuration
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for testing (default: 0.2)')
    parser.add_argument('--validation-size', type=float, default=0.15,
                       help='Proportion of training data for validation (default: 0.15)')
    
    # Memory optimization options
    parser.add_argument('--full-features', action='store_true',
                       help='Use full feature set (may cause memory issues)')
    parser.add_argument('--full-cv', action='store_true',
                       help='Use full cross-validation (may cause memory issues)')
    parser.add_argument('--max-methods', type=int, default=None,
                       help='Maximum number of methods to test (for memory constraints)')
    
    # Method selection
    parser.add_argument('--skip-neural', action='store_true',
                       help='Skip neural network training')
    parser.add_argument('--skip-classical', action='store_true',
                       help='Skip classical ML methods')
    parser.add_argument('--skip-ensemble', action='store_true',
                       help='Skip ensemble methods')
    parser.add_argument('--skip-sequential', action='store_true',
                       help='Skip sequential (LSTM) classifier training')
    parser.add_argument('--skip-hybrid', action='store_true',
                       help='Skip hybrid classifier training')
    parser.add_argument('--skip-deep-learning', action='store_true',
                       help='Skip deep learning classifier training')
    parser.add_argument('--skip-probabilistic', action='store_true',
                       help='Skip probabilistic classifier training')
    parser.add_argument('--skip-manifold', action='store_true',
                       help='Skip manifold learning classifier training')
    parser.add_argument('--skip-advanced', action='store_true',
                       help='Skip advanced classifier training')
    parser.add_argument('--skip-interpretable', action='store_true',
                       help='Skip interpretable classifier training')
    
    # Classical ML methods to test
    parser.add_argument('--classical-methods', nargs='+', 
                       choices=['random_forest', 'svm', 'logistic_regression', 'gradient_boosting', 
                               'naive_bayes', 'knn', 'decision_tree', 'adaboost'],
                       default=['random_forest', 'logistic_regression', 'gradient_boosting'],
                       help='Classical ML methods to test')
    
    # Ensemble methods to test
    parser.add_argument('--ensemble-methods', nargs='+',
                       choices=['voting', 'bagging', 'stacking', 'xgboost', 'lightgbm', 
                               'catboost', 'extra_trees', 'custom_ensemble'],
                       default=['voting', 'extra_trees'],
                       help='Ensemble methods to test')
    
    # Deep learning methods to test
    parser.add_argument('--deep-learning-methods', nargs='+',
                       choices=['cnn', 'transformer', 'attention_bilstm'],
                       default=['cnn'],
                       help='Deep learning methods to test')
    
    # Probabilistic methods to test
    parser.add_argument('--probabilistic-methods', nargs='+',
                       choices=['gaussian_nb', 'bernoulli_nb', 'multinomial_nb', 'complement_nb', 
                               'categorical_nb', 'hmm', 'gaussian_mixture'],
                       default=['gaussian_nb', 'bernoulli_nb'],
                       help='Probabilistic methods to test')
    
    # Manifold learning methods to test
    parser.add_argument('--manifold-methods', nargs='+',
                       choices=['pca', 'tsne', 'isomap', 'lle', 'spectral_embedding', 
                               'mds', 'ica', 'factor_analysis', 'truncated_svd'],
                       default=['pca', 'tsne'],
                       help='Manifold learning methods to test')
    
    # Advanced methods to test
    parser.add_argument('--advanced-methods', nargs='+',
                       choices=['isolation_forest', 'one_class_svm', 'local_outlier_factor', 
                               'elliptic_envelope', 'sgd', 'passive_aggressive', 'perceptron', 
                               'ridge', 'lasso', 'elastic_net', 'huber', 'quantile', 'tweedie'],
                       default=['isolation_forest', 'sgd'],
                       help='Advanced methods to test')
    
    # Interpretable methods to test
    parser.add_argument('--interpretable-methods', nargs='+',
                       choices=['linear_regression', 'lasso_regression', 'ridge_regression', 
                               'elastic_net_regression', 'decision_tree_classifier', 
                               'extra_tree_classifier', 'gaussian_nb_classifier'],
                       default=['linear_regression', 'decision_tree_classifier'],
                       help='Interpretable methods to test')
    
    # Output options
    parser.add_argument('--output-file', type=str, default='comparison_results_safe.json',
                       help='File to save detailed results (default: comparison_results_safe.json)')
    parser.add_argument('--no-save-models', action='store_true',
                       help='Do NOT save trained models to disk (saves storage space)')
    parser.add_argument('--model-save-path', type=str, default='models/comparison',
                       help='Base path for saving models (default: models/comparison)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information during training')
    
    args = parser.parse_args()
    
    # Validate files
    if not os.path.exists(args.human_file):
        print(f"Error: Human text file '{args.human_file}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(args.ai_file):
        print(f"Error: AI text file '{args.ai_file}' does not exist.")
        sys.exit(1)
    
    # Memory optimization settings
    reduced_features = not args.full_features
    reduced_cv = not args.full_cv
    save_models = not args.no_save_models
    
    print("ML Methods Comparison")
    print("="*50)
    print(f"Human text file: {args.human_file}")
    print(f"AI text file: {args.ai_file}")
    print(f"Test size: {args.test_size}")
    print(f"Validation size: {args.validation_size}")
    print(f"Memory optimization: {'Disabled' if args.full_features else 'Enabled'}")
    print(f"Reduced CV folds: {'No' if args.full_cv else 'Yes (3 instead of 5)'}")
    print(f"Save models: {'No' if args.no_save_models else 'Yes'} (to {args.model_save_path})")
    
    # Build list of methods to test
    methods_to_test = []
    
    if not args.skip_neural:
        methods_to_test.append(('neural', 'enhanced_neural_network'))
    
    if not args.skip_classical:
        for method in args.classical_methods:
            methods_to_test.append(('classical', method))
    
    if not args.skip_ensemble:
        for method in args.ensemble_methods:
            methods_to_test.append(('ensemble', method))
    
    if not args.skip_sequential:
        methods_to_test.append(('sequential', 'lstm_attention'))
    
    if not args.skip_hybrid:
        methods_to_test.append(('hybrid', 'lstm_features'))
    
    if not args.skip_deep_learning:
        for method in args.deep_learning_methods:
            methods_to_test.append(('deep_learning', method))
    
    if not args.skip_probabilistic:
        for method in args.probabilistic_methods:
            methods_to_test.append(('probabilistic', method))
    
    if not args.skip_manifold:
        for method in args.manifold_methods:
            methods_to_test.append(('manifold', method))
    
    if not args.skip_advanced:
        for method in args.advanced_methods:
            methods_to_test.append(('advanced', method))
    
    if not args.skip_interpretable:
        for method in args.interpretable_methods:
            methods_to_test.append(('interpretable', method))
    
    # Limit methods if specified
    if args.max_methods and len(methods_to_test) > args.max_methods:
        methods_to_test = methods_to_test[:args.max_methods]
        print(f"Limited to {args.max_methods} methods due to --max-methods constraint")
    
    print(f"Methods to test: {len(methods_to_test)}")
    if reduced_features:
        print("Using reduced feature set for memory efficiency")
    if reduced_cv:
        print("Using reduced cross-validation for memory efficiency")
    print("="*50)
    
    all_results = {}
    total_start_time = time.time()
    
    try:
        # Train methods one by one to avoid memory issues
        for i, (method_type, method_name) in enumerate(methods_to_test, 1):
            print(f"Progress: {i}/{len(methods_to_test)} methods")
            
            result = train_single_method(
                method_type=method_type,
                method_name=method_name,
                human_file=args.human_file,
                ai_file=args.ai_file,
                test_size=args.test_size,
                validation_size=args.validation_size,
                reduced_features=reduced_features,
                reduced_cv=reduced_cv,
                save_model=save_models,
                model_save_path=args.model_save_path
            )
            
            result_key = f"{method_type}_{method_name}"
            all_results[result_key] = result
            
            # Print immediate results
            if 'error' not in result or result.get('error') is None:
                print(f"Completed {result['method']}: {result['test_accuracy']:.4f} accuracy")
            else:
                print(f"Failed {result['method']}: {result['error']}")
            
            # Force garbage collection between methods
            gc.collect()
        
        total_time = time.time() - total_start_time
        
        # Generate comparison report
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        
        # Create results table
        successful_results = {k: v for k, v in all_results.items() if 'error' not in v or v.get('error') is None}
        
        if successful_results:
            print(f"{'Method':<35} {'Test Acc':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10} {'CV Acc':<10} {'Time(s)':<10}")
            print("-"*80)
            
            # Sort by test accuracy
            sorted_results = sorted(successful_results.items(), 
                                  key=lambda x: x[1].get('test_accuracy', 0), reverse=True)
            
            for method_key, result in sorted_results:
                method_name = result.get('method', method_key)
                test_acc = result.get('test_accuracy', 0.0)
                precision = result.get('test_precision', 0.0)
                recall = result.get('test_recall', 0.0)
                f1 = result.get('test_f1', 0.0)
                auc = result.get('test_auc', 0.0)
                cv_acc = result.get('cv_mean', 0.0)
                train_time = result.get('training_time', 0.0)
                
                print(f"{method_name:<35} {test_acc:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {auc:<10.4f} {cv_acc:<10.4f} {train_time:<10.0f}")
            
            # Identify best performing method
            best_method_key, best_result = sorted_results[0]
            best_method_name = best_result.get('method', best_method_key)
            best_accuracy = best_result.get('test_accuracy', 0.0)
            
            print("\n" + "="*80)
            print("BEST PERFORMING METHOD")
            print("="*80)
            print(f"Method: {best_method_name}")
            print(f"Test Accuracy: {best_accuracy:.4f}")
            print(f"Test Precision: {best_result.get('test_precision', 0.0):.4f}")
            print(f"Test Recall: {best_result.get('test_recall', 0.0):.4f}")
            print(f"Test F1-Score: {best_result.get('test_f1', 0.0):.4f}")
            if best_result.get('test_auc', 0.0) > 0:
                print(f"Test AUC: {best_result.get('test_auc', 0.0):.4f}")
            print(f"Training Time: {best_result.get('training_time', 0.0):.0f} seconds")
            
            # Performance categories
            print("\nPERFORMANCE ANALYSIS:")
            excellent = [k for k, v in successful_results.items() if v.get('test_accuracy', 0) >= 0.90]
            good = [k for k, v in successful_results.items() if 0.85 <= v.get('test_accuracy', 0) < 0.90]
            fair = [k for k, v in successful_results.items() if 0.80 <= v.get('test_accuracy', 0) < 0.85]
            poor = [k for k, v in successful_results.items() if v.get('test_accuracy', 0) < 0.80]
            
            if excellent:
                print(f"Excellent (>=90%): {len(excellent)} methods")
            if good:
                print(f"Good (85-90%): {len(good)} methods")
            if fair:
                print(f"Fair (80-85%): {len(fair)} methods")
            if poor:
                print(f"Poor (<80%): {len(poor)} methods")
            
            # Method family analysis
            neural_results = [v for k, v in successful_results.items() if 'neural' in k.lower()]
            classical_results = [v for k, v in successful_results.items() if 'classical' in k.lower()]
            ensemble_results = [v for k, v in successful_results.items() if 'ensemble' in k.lower()]
            sequential_results = [v for k, v in successful_results.items() if 'sequential' in k.lower()]
            hybrid_results = [v for k, v in successful_results.items() if 'hybrid' in k.lower()]
            deep_learning_results = [v for k, v in successful_results.items() if 'deep_learning' in k.lower()]
            probabilistic_results = [v for k, v in successful_results.items() if 'probabilistic' in k.lower()]
            manifold_results = [v for k, v in successful_results.items() if 'manifold' in k.lower()]
            advanced_results = [v for k, v in successful_results.items() if 'advanced' in k.lower()]
            interpretable_results = [v for k, v in successful_results.items() if 'interpretable' in k.lower()]
            
            print("\nMETHOD FAMILY ANALYSIS:")
            if neural_results:
                avg_neural = sum(r.get('test_accuracy', 0) for r in neural_results) / len(neural_results)
                print(f"Neural Networks: {len(neural_results)} methods, avg accuracy: {avg_neural:.4f}")
            
            if classical_results:
                avg_classical = sum(r.get('test_accuracy', 0) for r in classical_results) / len(classical_results)
                print(f"Classical ML: {len(classical_results)} methods, avg accuracy: {avg_classical:.4f}")
            
            if ensemble_results:
                avg_ensemble = sum(r.get('test_accuracy', 0) for r in ensemble_results) / len(ensemble_results)
                print(f"Ensemble Methods: {len(ensemble_results)} methods, avg accuracy: {avg_ensemble:.4f}")
            
            if sequential_results:
                avg_sequential = sum(r.get('test_accuracy', 0) for r in sequential_results) / len(sequential_results)
                print(f"Sequential (LSTM): {len(sequential_results)} methods, avg accuracy: {avg_sequential:.4f}")
            
            if hybrid_results:
                avg_hybrid = sum(r.get('test_accuracy', 0) for r in hybrid_results) / len(hybrid_results)
                print(f"Hybrid Models: {len(hybrid_results)} methods, avg accuracy: {avg_hybrid:.4f}")
            
            if deep_learning_results:
                avg_deep_learning = sum(r.get('test_accuracy', 0) for r in deep_learning_results) / len(deep_learning_results)
                print(f"Deep Learning: {len(deep_learning_results)} methods, avg accuracy: {avg_deep_learning:.4f}")
            
            if probabilistic_results:
                avg_probabilistic = sum(r.get('test_accuracy', 0) for r in probabilistic_results) / len(probabilistic_results)
                print(f"Probabilistic: {len(probabilistic_results)} methods, avg accuracy: {avg_probabilistic:.4f}")
            
            if manifold_results:
                avg_manifold = sum(r.get('test_accuracy', 0) for r in manifold_results) / len(manifold_results)
                print(f"Manifold Learning: {len(manifold_results)} methods, avg accuracy: {avg_manifold:.4f}")
            
            if advanced_results:
                avg_advanced = sum(r.get('test_accuracy', 0) for r in advanced_results) / len(advanced_results)
                print(f"Advanced Methods: {len(advanced_results)} methods, avg accuracy: {avg_advanced:.4f}")
            
            if interpretable_results:
                avg_interpretable = sum(r.get('test_accuracy', 0) for r in interpretable_results) / len(interpretable_results)
                print(f"Interpretable: {len(interpretable_results)} methods, avg accuracy: {avg_interpretable:.4f}")
        
        else:
            print("No methods completed successfully")
        
        # Report failed methods
        failed_results = {k: v for k, v in all_results.items() if 'error' in v and v.get('error') is not None}
        if failed_results:
            print(f"\nFAILED METHODS ({len(failed_results)}):")
            for method_key, result in failed_results.items():
                method_name = result.get('method', method_key)
                error = result.get('error', 'Unknown error')
                print(f"   {method_name}: {error}")
        
        # Save detailed results
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        json_results = {}
        for key, result in all_results.items():
            json_result = result.copy()
            # Convert numpy arrays to lists for JSON serialization
            if 'confusion_matrix' in json_result:
                json_result['confusion_matrix'] = json_result['confusion_matrix'].tolist()
            if 'cv_scores' in json_result:
                json_result['cv_scores'] = json_result['cv_scores'].tolist()
            json_results[key] = json_result
        
        # Add metadata
        json_results['_metadata'] = {
            'human_file': args.human_file,
            'ai_file': args.ai_file,
            'test_size': args.test_size,
            'validation_size': args.validation_size,
            'total_training_time': total_time,
            'methods_tested': len(all_results),
            'successful_methods': len(successful_results),
            'failed_methods': len(failed_results),
            'best_method': best_method_name if successful_results else None,
            'best_accuracy': best_accuracy if successful_results else None,
            'memory_optimized': reduced_features,
            'reduced_cv': reduced_cv
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_path}")
        print(f"Total comparison time: {total_time:.0f} seconds ({total_time/60:.1f} minutes)")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
