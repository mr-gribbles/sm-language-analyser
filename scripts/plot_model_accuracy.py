"""
Model Accuracy Plotting Script

This script loads all trained models from the /models folder and creates
comprehensive plots comparing their accuracy and performance metrics.
"""
import sys
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.model_serializer import ModelSerializer


def load_model_metrics(model_path: str) -> Dict[str, Any]:
    """Load metrics from a trained model file."""
    try:
        # Try to load using ModelSerializer first
        serializer = ModelSerializer()
        package = serializer.load_model_package(model_path)
        
        if package and hasattr(package, 'metadata') and package.metadata:
            metrics = package.metadata.get('training_metrics', {})
            if metrics:
                return {
                    'test_accuracy': metrics.get('test_accuracy', 0.0),
                    'test_precision': metrics.get('test_precision', 0.0),
                    'test_recall': metrics.get('test_recall', 0.0),
                    'test_f1': metrics.get('test_f1', 0.0),
                    'test_auc': metrics.get('test_auc', 0.0),
                    'cv_mean': metrics.get('cv_mean', 0.0),
                    'cv_std': metrics.get('cv_std', 0.0),
                    'training_time': metrics.get('training_time', 0.0),
                    'feature_count': metrics.get('feature_count', 0),
                    'model_type': metrics.get('model_type', 'Unknown')
                }
        
        # Fallback: try to load as pickle and extract metrics
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, dict):
            # Check if it's a results dictionary
            if 'test_accuracy' in data:
                return {
                    'test_accuracy': data.get('test_accuracy', 0.0),
                    'test_precision': data.get('test_precision', 0.0),
                    'test_recall': data.get('test_recall', 0.0),
                    'test_f1': data.get('test_f1', 0.0),
                    'test_auc': data.get('test_auc', 0.0),
                    'cv_mean': data.get('cv_mean', 0.0),
                    'cv_std': data.get('cv_std', 0.0),
                    'training_time': data.get('training_time', 0.0),
                    'feature_count': data.get('feature_count', 0),
                    'model_type': data.get('model_type', 'Unknown')
                }
            
            # Check if it has a metadata or metrics attribute
            if 'metadata' in data and isinstance(data['metadata'], dict):
                metrics = data['metadata'].get('training_metrics', {})
                if metrics:
                    return {
                        'test_accuracy': metrics.get('test_accuracy', 0.0),
                        'test_precision': metrics.get('test_precision', 0.0),
                        'test_recall': metrics.get('test_recall', 0.0),
                        'test_f1': metrics.get('test_f1', 0.0),
                        'test_auc': metrics.get('test_auc', 0.0),
                        'cv_mean': metrics.get('cv_mean', 0.0),
                        'cv_std': metrics.get('cv_std', 0.0),
                        'training_time': metrics.get('training_time', 0.0),
                        'feature_count': metrics.get('feature_count', 0),
                        'model_type': metrics.get('model_type', 'Unknown')
                    }
        
        # If no metrics found, return default values
        return {
            'test_accuracy': 0.0,
            'test_precision': 0.0,
            'test_recall': 0.0,
            'test_f1': 0.0,
            'test_auc': 0.0,
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'training_time': 0.0,
            'feature_count': 0,
            'model_type': 'Unknown'
        }
        
    except Exception as e:
        print(f"Warning: Could not load metrics from {model_path}: {e}")
        return None


def evaluate_models_on_test_data(human_file: str, ai_file: str) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all models on a test dataset to get fresh accuracy metrics.
    This is useful when stored metrics are not available.
    """
    print("Evaluating models on test data...")
    
    # Import prediction functionality
    from scripts.predict_text import discover_available_models, load_neural_network, load_classical_model
    from scripts.predict_text import load_ensemble_model, load_sequential_model, load_hybrid_model
    from scripts.predict_text import load_deep_learning_model, load_probabilistic_model, load_manifold_model
    from scripts.predict_text import load_advanced_model, load_interpretable_model
    
    # Load test data
    texts = []
    labels = []
    
    # Load human texts (label = 0)
    try:
        with open(human_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if 'cleaned_text' in record.get('original_content', {}):
                        text = record['original_content']['cleaned_text']
                    elif 'cleaned_selftext' in record.get('original_content', {}):
                        text = record['original_content']['cleaned_selftext']
                    else:
                        text = record.get('original_content', {}).get('raw_text', 
                              record.get('original_content', {}).get('raw_selftext', ''))
                    
                    if text and len(text.strip()) > 20:
                        texts.append(text.strip())
                        labels.append(0)  # Human
                except (json.JSONDecodeError, KeyError):
                    continue
    except FileNotFoundError:
        print(f"Warning: Human file {human_file} not found")
    
    # Load AI texts (label = 1)
    try:
        with open(ai_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record and record.get('llm_transformation') and record['llm_transformation'] and record['llm_transformation'].get('rewritten_text'):
                        text = record['llm_transformation']['rewritten_text']
                        if text and len(text.strip()) > 20:
                            texts.append(text.strip())
                            labels.append(1)  # AI
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    except FileNotFoundError:
        print(f"Warning: AI file {ai_file} not found")
    
    if not texts:
        print("No test data found. Using dummy metrics.")
        return {}
    
    print(f"Loaded {len(texts)} test samples ({labels.count(0)} human, {labels.count(1)} AI)")
    
    # Discover available models
    available_models = discover_available_models()
    
    model_metrics = {}
    
    # Evaluate each model
    for category, models in available_models.items():
        for model_name, model_path in models:
            try:
                print(f"Evaluating {category}: {model_name}")
                
                # Load the appropriate model
                if category == 'neural':
                    if model_name == 'sequential':
                        classifier, model_type = load_sequential_model(model_path)
                    else:
                        classifier, model_type = load_neural_network(model_path)
                elif category == 'classical':
                    classifier, model_type = load_classical_model(model_path, model_name)
                elif category == 'ensemble':
                    classifier, model_type = load_ensemble_model(model_path, model_name)
                elif category == 'hybrid':
                    classifier, model_type = load_hybrid_model(model_path)
                elif category == 'deep_learning':
                    classifier, model_type = load_deep_learning_model(model_path, model_name)
                elif category == 'probabilistic':
                    classifier, model_type = load_probabilistic_model(model_path, model_name)
                elif category == 'manifold':
                    classifier, model_type = load_manifold_model(model_path, model_name)
                elif category == 'advanced':
                    classifier, model_type = load_advanced_model(model_path, model_name)
                elif category == 'interpretable':
                    classifier, model_type = load_interpretable_model(model_path, model_name)
                else:
                    continue
                
                # Make predictions
                predictions, probabilities = classifier.predict(texts)
                
                # Calculate accuracy
                accuracy = np.mean(np.array(predictions) == np.array(labels))
                
                # Calculate precision, recall, F1
                from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, predictions, average='weighted', zero_division=0
                )
                
                # Calculate AUC if probabilities available
                auc = 0.0
                if probabilities is not None and len(set(labels)) > 1:
                    try:
                        auc = roc_auc_score(labels, probabilities)
                    except:
                        auc = 0.0
                
                model_metrics[model_type] = {
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'test_auc': auc,
                    'cv_mean': 0.0,  # Not available from evaluation
                    'cv_std': 0.0,
                    'training_time': 0.0,  # Not available from evaluation
                    'feature_count': 0,
                    'model_type': model_type
                }
                
            except Exception as e:
                print(f"Failed to evaluate {category}: {model_name} - {e}")
                continue
    
    return model_metrics


def discover_and_load_model_metrics(models_dir: str = "models", 
                                   human_file: str = None, 
                                   ai_file: str = None) -> Dict[str, Dict[str, float]]:
    """Discover all model files and load their metrics."""
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"Models directory {models_dir} does not exist")
        return {}
    
    model_files = list(models_path.glob("comparison_*.pkl"))
    print(f"Found {len(model_files)} model files")
    
    model_metrics = {}
    models_without_metrics = []
    
    for model_file in model_files:
        model_name = model_file.stem.replace('comparison_', '')
        print(f"Loading metrics for: {model_name}")
        
        metrics = load_model_metrics(str(model_file))
        if metrics and metrics.get('test_accuracy', 0) > 0:
            # Create a display name
            display_name = model_name.replace('_', ' ').title()
            model_metrics[display_name] = metrics
        else:
            models_without_metrics.append(model_name)
    
    print(f"Loaded metrics for {len(model_metrics)} models")
    if models_without_metrics:
        print(f"Models without stored metrics: {len(models_without_metrics)}")
        
        # If we have test data files, evaluate models without metrics
        if human_file and ai_file and os.path.exists(human_file) and os.path.exists(ai_file):
            print("Evaluating models without stored metrics on test data...")
            fresh_metrics = evaluate_models_on_test_data(human_file, ai_file)
            model_metrics.update(fresh_metrics)
    
    return model_metrics


def create_accuracy_comparison_plot(model_metrics: Dict[str, Dict[str, float]], 
                                   output_dir: str = "plots"):
    """Create a horizontal bar plot comparing test accuracy of all models."""
    if not model_metrics:
        print("No model metrics available for plotting")
        return
    
    # Prepare data
    models = list(model_metrics.keys())
    accuracies = [model_metrics[model]['test_accuracy'] for model in models]
    
    # Sort by accuracy
    sorted_data = sorted(zip(models, accuracies), key=lambda x: x[1], reverse=True)
    models, accuracies = zip(*sorted_data)
    
    # Create plot
    plt.figure(figsize=(12, max(8, len(models) * 0.4)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    bars = plt.barh(range(len(models)), accuracies, color=colors)
    
    # Customize plot
    plt.yticks(range(len(models)), models)
    plt.xlabel('Test Accuracy')
    plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(acc + 0.005, i, f'{acc:.3f}', va='center', fontweight='bold')
    
    # Color code by performance
    for i, acc in enumerate(accuracies):
        if acc >= 0.95:
            bars[i].set_color('darkgreen')
        elif acc >= 0.90:
            bars[i].set_color('green')
        elif acc >= 0.85:
            bars[i].set_color('orange')
        else:
            bars[i].set_color('red')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'model_accuracy_comparison.pdf', bbox_inches='tight')
    print(f"Saved accuracy comparison plot to {output_path}")
    plt.show()


def create_metrics_heatmap(model_metrics: Dict[str, Dict[str, float]], 
                          output_dir: str = "plots"):
    """Create a heatmap showing all metrics for all models."""
    if not model_metrics:
        print("No model metrics available for plotting")
        return
    
    # Prepare data
    models = list(model_metrics.keys())
    metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
    
    # Create matrix
    data_matrix = []
    for model in models:
        row = [model_metrics[model].get(metric, 0.0) for metric in metrics]
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Create heatmap
    plt.figure(figsize=(10, max(8, len(models) * 0.3)))
    
    # Create heatmap with custom colormap
    sns.heatmap(data_matrix, 
                xticklabels=[m.replace('test_', '').title() for m in metrics],
                yticklabels=models,
                annot=True, 
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0, vmax=1,
                cbar_kws={'label': 'Score'})
    
    plt.title('Model Performance Metrics Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics')
    plt.ylabel('Models')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'model_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'model_metrics_heatmap.pdf', bbox_inches='tight')
    print(f"Saved metrics heatmap to {output_path}")
    plt.show()


def create_model_family_comparison(model_metrics: Dict[str, Dict[str, float]], 
                                  output_dir: str = "plots"):
    """Create a plot comparing different model families."""
    if not model_metrics:
        print("No model metrics available for plotting")
        return
    
    # Categorize models by family
    families = {
        'Neural Networks': [],
        'Classical ML': [],
        'Ensemble Methods': [],
        'Sequential/LSTM': [],
        'Hybrid Models': [],
        'Deep Learning': [],
        'Probabilistic': [],
        'Advanced': [],
        'Interpretable': []
    }
    
    for model_name, metrics in model_metrics.items():
        model_lower = model_name.lower()
        if 'neural' in model_lower or 'enhanced' in model_lower:
            families['Neural Networks'].append(metrics['test_accuracy'])
        elif any(word in model_lower for word in ['classical', 'random forest', 'svm', 'logistic', 'gradient', 'naive bayes', 'knn', 'decision tree', 'adaboost']):
            families['Classical ML'].append(metrics['test_accuracy'])
        elif any(word in model_lower for word in ['ensemble', 'voting', 'bagging', 'stacking', 'xgboost', 'lightgbm', 'catboost', 'extra trees']):
            families['Ensemble Methods'].append(metrics['test_accuracy'])
        elif 'sequential' in model_lower or 'lstm' in model_lower:
            families['Sequential/LSTM'].append(metrics['test_accuracy'])
        elif 'hybrid' in model_lower:
            families['Hybrid Models'].append(metrics['test_accuracy'])
        elif any(word in model_lower for word in ['cnn', 'transformer', 'attention']):
            families['Deep Learning'].append(metrics['test_accuracy'])
        elif any(word in model_lower for word in ['probabilistic', 'gaussian', 'bernoulli', 'multinomial', 'complement', 'categorical', 'hmm', 'mixture']):
            families['Probabilistic'].append(metrics['test_accuracy'])
        elif any(word in model_lower for word in ['advanced', 'isolation', 'one class', 'sgd', 'passive', 'perceptron', 'ridge', 'lasso', 'elastic']):
            families['Advanced'].append(metrics['test_accuracy'])
        elif 'interpretable' in model_lower:
            families['Interpretable'].append(metrics['test_accuracy'])
    
    # Filter out empty families
    families = {k: v for k, v in families.items() if v}
    
    if not families:
        print("No model families found for comparison")
        return
    
    # Create box plot
    plt.figure(figsize=(12, 8))
    
    family_names = list(families.keys())
    family_data = [families[name] for name in family_names]
    
    box_plot = plt.boxplot(family_data, labels=family_names, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(family_names)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Test Accuracy')
    plt.title('Model Family Performance Comparison', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add mean values as text
    for i, (name, data) in enumerate(families.items(), 1):
        mean_acc = np.mean(data)
        plt.text(i, mean_acc + 0.01, f'μ={mean_acc:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'model_family_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'model_family_comparison.pdf', bbox_inches='tight')
    print(f"Saved family comparison plot to {output_path}")
    plt.show()


def create_performance_distribution(model_metrics: Dict[str, Dict[str, float]], 
                                   output_dir: str = "plots"):
    """Create a histogram showing the distribution of model performance."""
    if not model_metrics:
        print("No model metrics available for plotting")
        return
    
    accuracies = [metrics['test_accuracy'] for metrics in model_metrics.values()]
    
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    n, bins, patches = plt.hist(accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Color code bins by performance level
    for i, (patch, bin_start) in enumerate(zip(patches, bins[:-1])):
        if bin_start >= 0.95:
            patch.set_facecolor('darkgreen')
        elif bin_start >= 0.90:
            patch.set_facecolor('green')
        elif bin_start >= 0.85:
            patch.set_facecolor('orange')
        else:
            patch.set_facecolor('red')
    
    # Add statistics
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    median_acc = np.median(accuracies)
    
    plt.axvline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.3f}')
    plt.axvline(median_acc, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_acc:.3f}')
    
    plt.xlabel('Test Accuracy')
    plt.ylabel('Number of Models')
    plt.title('Distribution of Model Performance', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add text box with statistics
    stats_text = f'Models: {len(accuracies)}\nMean: {mean_acc:.3f}\nStd: {std_acc:.3f}\nMin: {min(accuracies):.3f}\nMax: {max(accuracies):.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'performance_distribution.pdf', bbox_inches='tight')
    print(f"Saved performance distribution plot to {output_path}")
    plt.show()


def create_top_models_detailed_plot(model_metrics: Dict[str, Dict[str, float]], 
                                   top_n: int = 10, output_dir: str = "plots"):
    """Create a detailed plot of the top N performing models."""
    if not model_metrics:
        print("No model metrics available for plotting")
        return
    
    # Sort models by accuracy and take top N
    sorted_models = sorted(model_metrics.items(), 
                          key=lambda x: x[1]['test_accuracy'], reverse=True)[:top_n]
    
    models = [item[0] for item in sorted_models]
    metrics_data = [item[1] for item in sorted_models]
    
    # Prepare data for plotting
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    metrics_keys = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
    
    # Create subplot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.15
    
    # Create bars for each metric
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (metric_name, metric_key, color) in enumerate(zip(metrics_names, metrics_keys, colors)):
        values = [data.get(metric_key, 0.0) for data in metrics_data]
        bars = ax.bar(x + i * width, values, width, label=metric_name, color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if value > 0:  # Only show non-zero values
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Customize plot
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title(f'Top {top_n} Models - Detailed Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f'top_{top_n}_models_detailed.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / f'top_{top_n}_models_detailed.pdf', bbox_inches='tight')
    print(f"Saved top {top_n} models detailed plot to {output_path}")
    plt.show()


def create_summary_report(model_metrics: Dict[str, Dict[str, float]], 
                         output_dir: str = "plots"):
    """Create a text summary report of model performance."""
    if not model_metrics:
        print("No model metrics available for report")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_path = output_path / 'model_performance_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("MODEL PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        accuracies = [metrics['test_accuracy'] for metrics in model_metrics.values()]
        f.write(f"Total Models Analyzed: {len(model_metrics)}\n")
        f.write(f"Average Accuracy: {np.mean(accuracies):.4f}\n")
        f.write(f"Standard Deviation: {np.std(accuracies):.4f}\n")
        f.write(f"Best Accuracy: {max(accuracies):.4f}\n")
        f.write(f"Worst Accuracy: {min(accuracies):.4f}\n\n")
        
        # Performance categories
        excellent = [name for name, metrics in model_metrics.items() if metrics['test_accuracy'] >= 0.95]
        good = [name for name, metrics in model_metrics.items() if 0.90 <= metrics['test_accuracy'] < 0.95]
        fair = [name for name, metrics in model_metrics.items() if 0.85 <= metrics['test_accuracy'] < 0.90]
        poor = [name for name, metrics in model_metrics.items() if metrics['test_accuracy'] < 0.85]
        
        f.write("PERFORMANCE CATEGORIES:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Excellent (≥95%): {len(excellent)} models\n")
        f.write(f"Good (90-95%): {len(good)} models\n")
        f.write(f"Fair (85-90%): {len(fair)} models\n")
        f.write(f"Poor (<85%): {len(poor)} models\n\n")
        
        # Top 10 models
        sorted_models = sorted(model_metrics.items(), 
                              key=lambda x: x[1]['test_accuracy'], reverse=True)
        
        f.write("TOP 10 PERFORMING MODELS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Rank':<4} {'Model':<35} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
        f.write("-" * 80 + "\n")
        
        for i, (model_name, metrics) in enumerate(sorted_models[:10], 1):
            f.write(f"{i:<4} {model_name:<35} {metrics['test_accuracy']:<10.4f} "
                   f"{metrics['test_precision']:<10.4f} {metrics['test_recall']:<10.4f} "
                   f"{metrics['test_f1']:<10.4f}\n")
        
        f.write("\n")
        
        # Bottom 5 models
        f.write("BOTTOM 5 PERFORMING MODELS:\n")
        f.write("-" * 32 + "\n")
        f.write(f"{'Rank':<4} {'Model':<35} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
        f.write("-" * 80 + "\n")
        
        for i, (model_name, metrics) in enumerate(sorted_models[-5:], len(sorted_models)-4):
            f.write(f"{i:<4} {model_name:<35} {metrics['test_accuracy']:<10.4f} "
                   f"{metrics['test_precision']:<10.4f} {metrics['test_recall']:<10.4f} "
                   f"{metrics['test_f1']:<10.4f}\n")
    
    print(f"Saved performance report to {report_path}")


def main():
    """Main function to create all plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create accuracy comparison plots for all trained models')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing trained models (default: models)')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Directory to save plots (default: plots)')
    parser.add_argument('--human-file', type=str, default=None,
                       help='JSONL file with human texts for evaluation (optional)')
    parser.add_argument('--ai-file', type=str, default=None,
                       help='JSONL file with AI texts for evaluation (optional)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top models to show in detailed plot (default: 10)')
    parser.add_argument('--evaluate-fresh', action='store_true',
                       help='Evaluate all models on test data instead of using stored metrics')
    
    args = parser.parse_args()
    
    print("Model Accuracy Plotting Script")
    print("=" * 40)
    print(f"Models directory: {args.models_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Load model metrics
    if args.evaluate_fresh and args.human_file and args.ai_file:
        print("Evaluating models on fresh test data...")
        model_metrics = evaluate_models_on_test_data(args.human_file, args.ai_file)
    else:
        print("Loading stored model metrics...")
        model_metrics = discover_and_load_model_metrics(
            models_dir=args.models_dir,
            human_file=args.human_file,
            ai_file=args.ai_file
        )
    
    if not model_metrics:
        print("No model metrics found. Please ensure:")
        print("1. Models directory contains trained models")
        print("2. Models have stored metrics, or")
        print("3. Provide --human-file and --ai-file for fresh evaluation")
        return
    
    print(f"Found metrics for {len(model_metrics)} models")
    print("=" * 40)
    
    # Create all plots
    print("Creating accuracy comparison plot...")
    create_accuracy_comparison_plot(model_metrics, args.output_dir)
    
    print("Creating metrics heatmap...")
    create_metrics_heatmap(model_metrics, args.output_dir)
    
    print("Creating model family comparison...")
    create_model_family_comparison(model_metrics, args.output_dir)
    
    print("Creating performance distribution plot...")
    create_performance_distribution(model_metrics, args.output_dir)
    
    print(f"Creating top {args.top_n} models detailed plot...")
    create_top_models_detailed_plot(model_metrics, args.top_n, args.output_dir)
    
    print("Creating summary report...")
    create_summary_report(model_metrics, args.output_dir)
    
    print("=" * 40)
    print("All plots and reports created successfully!")
    print(f"Check the '{args.output_dir}' directory for output files.")


if __name__ == "__main__":
    main()
