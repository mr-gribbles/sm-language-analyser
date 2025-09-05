"""
Simple Model Accuracy Plotting Script

This script uses the prediction functionality to evaluate all models
and create accuracy comparison plots.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.predict_text import discover_available_models, predict_with_all_models


def create_sample_texts_for_evaluation():
    """Create a diverse set of sample texts for model evaluation."""
    # Sample human-written texts (various styles and topics)
    human_texts = [
        "I've been thinking about this problem for weeks now, and I finally think I have a solution. The key insight came to me while I was walking my dog yesterday morning.",
        "The weather today is absolutely gorgeous! I decided to take a long walk through the park and grab some coffee from that new place downtown.",
        "My grandmother used to tell me stories about her childhood during the war. She would describe how they had to ration food and make do with very little.",
        "I'm really struggling with this math homework. The equations just don't make sense to me, no matter how many times I read the textbook.",
        "Last weekend we went camping in the mountains. The view from our campsite was breathtaking, especially during sunrise.",
        "I can't believe how fast this year has gone by. It feels like just yesterday we were celebrating New Year's, and now it's already December.",
        "My cat has this weird habit of sitting in boxes that are way too small for her. She'll squeeze herself in there and look completely content.",
        "The concert last night was incredible. The band played all their classic hits, and the crowd was singing along to every song.",
        "I've been learning to cook during quarantine, and let me tell you, it's been a journey of many burnt meals and kitchen disasters.",
        "There's something magical about reading a good book on a rainy day. I get completely lost in the story and forget about everything else."
    ]
    
    # Sample AI-generated texts (more formal, structured patterns)
    ai_texts = [
        "The implementation of advanced machine learning algorithms has revolutionized the field of data analysis, enabling researchers to extract meaningful insights from complex datasets.",
        "In order to optimize performance metrics, it is essential to consider multiple variables and their interdependent relationships within the system architecture.",
        "The comprehensive analysis reveals significant correlations between various factors, suggesting that a multifaceted approach would be most beneficial for achieving desired outcomes.",
        "Recent developments in artificial intelligence have demonstrated remarkable capabilities in natural language processing, computer vision, and predictive analytics applications.",
        "The systematic evaluation of different methodologies indicates that hybrid approaches tend to outperform traditional single-method implementations across various benchmarks.",
        "To ensure optimal results, it is recommended to implement a robust framework that incorporates best practices and industry standards for quality assurance.",
        "The integration of multiple data sources provides enhanced visibility into operational processes, facilitating more informed decision-making capabilities.",
        "Advanced algorithms utilize sophisticated mathematical models to process information efficiently and generate accurate predictions based on historical patterns.",
        "The scalable architecture enables seamless integration with existing systems while maintaining high performance standards and reliability metrics.",
        "Comprehensive testing protocols ensure that all components function correctly within the specified parameters and meet established quality benchmarks."
    ]
    
    # Combine and label
    texts = human_texts + ai_texts
    labels = [0] * len(human_texts) + [1] * len(ai_texts)  # 0 = Human, 1 = AI
    
    return texts, labels


def evaluate_all_models():
    """Evaluate all available models and return accuracy metrics."""
    print("Creating sample texts for evaluation...")
    texts, true_labels = create_sample_texts_for_evaluation()
    
    print(f"Evaluating models on {len(texts)} sample texts ({true_labels.count(0)} human, {true_labels.count(1)} AI)")
    
    # Discover available models
    available_models = discover_available_models()
    
    model_results = {}
    
    # Import model loading functions
    from scripts.predict_text import (load_neural_network, load_classical_model, load_ensemble_model,
                                     load_sequential_model, load_hybrid_model, load_deep_learning_model,
                                     load_probabilistic_model, load_manifold_model, load_advanced_model,
                                     load_interpretable_model)
    
    total_models = sum(len(models) for models in available_models.values())
    current_model = 0
    
    # Evaluate each model category
    for category, models in available_models.items():
        for model_name, model_path in models:
            current_model += 1
            try:
                print(f"[{current_model}/{total_models}] Evaluating {category}: {model_name}")
                
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
                accuracy = np.mean(np.array(predictions) == np.array(true_labels))
                
                # Calculate other metrics
                from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_labels, predictions, average='weighted', zero_division=0
                )
                
                # Calculate AUC if probabilities available
                auc = 0.0
                if probabilities is not None and len(set(true_labels)) > 1:
                    try:
                        auc = roc_auc_score(true_labels, probabilities)
                    except:
                        auc = 0.0
                
                model_results[model_type] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'category': category
                }
                
                print(f"  → Accuracy: {accuracy:.3f}")
                
            except Exception as e:
                print(f"  → Failed: {str(e)}")
                continue
    
    return model_results


def create_accuracy_bar_plot(model_results: Dict, output_dir: str = "plots"):
    """Create a horizontal bar plot of model accuracies."""
    if not model_results:
        print("No model results to plot")
        return
    
    # Prepare data
    models = list(model_results.keys())
    accuracies = [model_results[model]['accuracy'] for model in models]
    
    # Sort by accuracy
    sorted_data = sorted(zip(models, accuracies), key=lambda x: x[1], reverse=True)
    models, accuracies = zip(*sorted_data)
    
    # Create plot
    plt.figure(figsize=(12, max(8, len(models) * 0.4)))
    
    # Color code by performance
    colors = []
    for acc in accuracies:
        if acc >= 0.95:
            colors.append('darkgreen')
        elif acc >= 0.90:
            colors.append('green')
        elif acc >= 0.85:
            colors.append('orange')
        elif acc >= 0.80:
            colors.append('gold')
        else:
            colors.append('red')
    
    bars = plt.barh(range(len(models)), accuracies, color=colors, alpha=0.8)
    
    # Customize plot
    plt.yticks(range(len(models)), models)
    plt.xlabel('Accuracy')
    plt.title('Model Accuracy Comparison\n(Evaluated on Sample Texts)', fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(acc + 0.005, i, f'{acc:.3f}', va='center', fontweight='bold', fontsize=10)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkgreen', label='Excellent (≥95%)'),
        Patch(facecolor='green', label='Good (90-95%)'),
        Patch(facecolor='orange', label='Fair (85-90%)'),
        Patch(facecolor='gold', label='Okay (80-85%)'),
        Patch(facecolor='red', label='Poor (<80%)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'model_accuracy_comparison.pdf', bbox_inches='tight')
    print(f"Saved accuracy comparison plot to {output_path}")
    plt.show()


def create_metrics_comparison(model_results: Dict, output_dir: str = "plots"):
    """Create a comparison of multiple metrics."""
    if not model_results:
        print("No model results to plot")
        return
    
    # Get top 15 models by accuracy
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:15]
    
    models = [item[0] for item in sorted_models]
    
    # Prepare data
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.15
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (metric, metric_name, color) in enumerate(zip(metrics, metric_names, colors)):
        values = [sorted_models[j][1][metric] for j in range(len(models))]
        bars = ax.bar(x + i * width, values, width, label=metric_name, color=color, alpha=0.8)
        
        # Add value labels on bars (only for accuracy to avoid clutter)
        if metric == 'accuracy':
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Top 15 Models - Performance Metrics Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'top_models_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'top_models_metrics_comparison.pdf', bbox_inches='tight')
    print(f"Saved metrics comparison plot to {output_path}")
    plt.show()


def create_category_comparison(model_results: Dict, output_dir: str = "plots"):
    """Create a box plot comparing model categories."""
    if not model_results:
        print("No model results to plot")
        return
    
    # Group by category
    categories = {}
    for model_name, results in model_results.items():
        category = results['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(results['accuracy'])
    
    # Filter out categories with too few models
    categories = {k: v for k, v in categories.items() if len(v) >= 2}
    
    if not categories:
        print("Not enough models per category for comparison")
        return
    
    plt.figure(figsize=(12, 8))
    
    category_names = list(categories.keys())
    category_data = [categories[name] for name in category_names]
    
    box_plot = plt.boxplot(category_data, labels=category_names, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(category_names)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Accuracy')
    plt.title('Model Category Performance Comparison', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add mean values
    for i, (name, data) in enumerate(categories.items(), 1):
        mean_acc = np.mean(data)
        plt.text(i, mean_acc + 0.02, f'μ={mean_acc:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'category_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'category_performance_comparison.pdf', bbox_inches='tight')
    print(f"Saved category comparison plot to {output_path}")
    plt.show()


def create_summary_report(model_results: Dict, output_dir: str = "plots"):
    """Create a text summary report."""
    if not model_results:
        print("No model results to report")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_path = output_path / 'model_accuracy_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("MODEL ACCURACY ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        accuracies = [results['accuracy'] for results in model_results.values()]
        f.write(f"Total Models Evaluated: {len(model_results)}\n")
        f.write(f"Average Accuracy: {np.mean(accuracies):.4f}\n")
        f.write(f"Standard Deviation: {np.std(accuracies):.4f}\n")
        f.write(f"Best Accuracy: {max(accuracies):.4f}\n")
        f.write(f"Worst Accuracy: {min(accuracies):.4f}\n\n")
        
        # Performance categories
        excellent = [name for name, results in model_results.items() if results['accuracy'] >= 0.95]
        good = [name for name, results in model_results.items() if 0.90 <= results['accuracy'] < 0.95]
