#!/usr/bin/env python3
"""Script to train all available ML methods for AI vs Human text classification."""

import subprocess
import sys
import os

def main():
    """Generate and optionally run the command to train all methods."""
    
    # All available methods
    classical_methods = ['random_forest', 'svm', 'logistic_regression', 'gradient_boosting', 
                        'naive_bayes', 'knn', 'decision_tree', 'adaboost']
    
    ensemble_methods = ['voting', 'bagging', 'stacking', 'xgboost', 'lightgbm', 
                       'catboost', 'extra_trees', 'custom_ensemble']
    
    deep_learning_methods = ['cnn', 'transformer', 'attention_bilstm']
    
    probabilistic_methods = ['gaussian_nb', 'bernoulli_nb', 'multinomial_nb', 'complement_nb', 
                            'categorical_nb', 'hmm', 'gaussian_mixture']
    
    manifold_methods = ['pca', 'tsne', 'isomap', 'lle', 'spectral_embedding', 
                       'mds', 'ica', 'factor_analysis', 'truncated_svd']
    
    advanced_methods = ['isolation_forest', 'one_class_svm', 'local_outlier_factor', 
                       'elliptic_envelope', 'sgd', 'passive_aggressive', 'perceptron', 
                       'ridge', 'lasso', 'elastic_net', 'huber', 'quantile', 'tweedie']
    
    interpretable_methods = ['linear_regression', 'lasso_regression', 'ridge_regression', 
                            'elastic_net_regression', 'decision_tree_classifier', 
                            'extra_tree_classifier', 'gaussian_nb_classifier']
    
    # Count total methods
    total_methods = (len(classical_methods) + len(ensemble_methods) + len(deep_learning_methods) + 
                    len(probabilistic_methods) + len(manifold_methods) + len(advanced_methods) + 
                    len(interpretable_methods) + 3)  # +3 for neural, sequential, hybrid
    
    print(f"Total methods available: {total_methods}")
    print("\nBreakdown:")
    print(f"  Neural Network: 1")
    print(f"  Classical ML: {len(classical_methods)}")
    print(f"  Ensemble: {len(ensemble_methods)}")
    print(f"  Sequential (LSTM): 1")
    print(f"  Hybrid: 1")
    print(f"  Deep Learning: {len(deep_learning_methods)}")
    print(f"  Probabilistic: {len(probabilistic_methods)}")
    print(f"  Manifold Learning: {len(manifold_methods)}")
    print(f"  Advanced: {len(advanced_methods)}")
    print(f"  Interpretable: {len(interpretable_methods)}")
    
    # Build the command
    cmd = [
        'python', 'scripts/compare_all_methods.py',
        '--human-file', 'REPLACE_WITH_HUMAN_FILE',
        '--ai-file', 'REPLACE_WITH_AI_FILE',
        '--classical-methods'] + classical_methods + [
        '--ensemble-methods'] + ensemble_methods + [
        '--deep-learning-methods'] + deep_learning_methods + [
        '--probabilistic-methods'] + probabilistic_methods + [
        '--manifold-methods'] + manifold_methods + [
        '--advanced-methods'] + advanced_methods + [
        '--interpretable-methods'] + interpretable_methods
    
    print(f"\nTo train all {total_methods} methods, run this command:")
    print("=" * 80)
    
    # Print command in a readable format
    cmd_str = ' '.join(cmd)
    cmd_str = cmd_str.replace('REPLACE_WITH_HUMAN_FILE', '<path_to_human_texts.jsonl>')
    cmd_str = cmd_str.replace('REPLACE_WITH_AI_FILE', '<path_to_ai_texts.jsonl>')
    
    # Break long command into multiple lines for readability
    parts = cmd_str.split(' --')
    print(parts[0] + ' \\')
    for part in parts[1:]:
        if part.endswith('-methods'):
            print(f'  --{part} \\')
        else:
            methods_part = part.split(' ', 1)
            if len(methods_part) == 2:
                method_type, methods_list = methods_part
                methods = methods_list.split(' ')
                print(f'    {" ".join(methods)} \\')
            else:
                print(f'    {part} \\')
    
    print("\nOr for memory-constrained systems, add --full-features for better accuracy")
    print("but longer training time, or keep default settings for faster training.")
    
    # Ask if user wants to run it
    if len(sys.argv) > 2:
        human_file = sys.argv[1]
        ai_file = sys.argv[2]
        
        if os.path.exists(human_file) and os.path.exists(ai_file):
            response = input(f"\nFound files {human_file} and {ai_file}. Train all {total_methods} methods? (y/N): ")
            if response.lower() == 'y':
                # Replace placeholders with actual files
                final_cmd = []
                for arg in cmd:
                    if arg == 'REPLACE_WITH_HUMAN_FILE':
                        final_cmd.append(human_file)
                    elif arg == 'REPLACE_WITH_AI_FILE':
                        final_cmd.append(ai_file)
                    else:
                        final_cmd.append(arg)
                
                print(f"Starting training of all {total_methods} methods...")
                print("This may take several hours depending on your system.")
                
                try:
                    subprocess.run(final_cmd, check=True)
                    print(f"Successfully trained all {total_methods} methods!")
                except subprocess.CalledProcessError as e:
                    print(f"Training failed with error: {e}")
                    sys.exit(1)
            else:
                print("Training cancelled.")
        else:
            print(f"Error: Could not find files {human_file} and/or {ai_file}")
    else:
        print(f"\nTo run this script with your data files:")
        print(f"  python scripts/train_all_methods.py <human_file.jsonl> <ai_file.jsonl>")


if __name__ == "__main__":
    main()
