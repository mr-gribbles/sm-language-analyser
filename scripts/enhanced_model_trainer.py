"""
Enhanced Model Trainer with Comprehensive Metrics

This script enhances the existing training pipeline to include additional performance
metrics and novel evaluation approaches for better model comparison.
"""
import sys
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    log_loss, brier_score_loss, average_precision_score
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import learning_curve, validation_curve, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.model_serializer import ModelSerializer


class EnhancedMetricsCalculator:
    """Calculate comprehensive performance metrics for model evaluation."""
    
    @staticmethod
    def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_prob: Optional[np.ndarray] = None,
                                      cv_scores: Optional[List[float]] = None,
                                      training_time: float = 0.0,
                                      model_name: str = "Unknown") -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Class-specific metrics
        if len(precision) >= 2:
            metrics['precision_human'] = precision[0]
            metrics['precision_ai'] = precision[1]
            metrics['recall_human'] = recall[0]
            metrics['recall_ai'] = recall[1]
            metrics['f1_human'] = f1[0]
            metrics['f1_ai'] = f1[1]
            metrics['support_human'] = support[0]
            metrics['support_ai'] = support[1]
        
        # Weighted and macro averages
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        metrics['precision_weighted'] = precision_w
        metrics['recall_weighted'] = recall_w
        metrics['f1_weighted'] = f1_w
        metrics['precision_macro'] = precision_m
        metrics['recall_macro'] = recall_m
        metrics['f1_macro'] = f1_m
        
        # Agreement and correlation metrics
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Confusion matrix derived metrics
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # Sensitivity and Specificity
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Positive and Negative Predictive Values
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            
            # False rates
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            # Likelihood ratios
            metrics['lr_positive'] = metrics['sensitivity'] / metrics['fpr'] if metrics['fpr'] > 0 else float('inf')
            metrics['lr_negative'] = metrics['fnr'] / metrics['specificity'] if metrics['specificity'] > 0 else float('inf')
            
            # Diagnostic odds ratio
            if fp > 0 and fn > 0:
                metrics['diagnostic_odds_ratio'] = (tp * tn) / (fp * fn)
            else:
                metrics['diagnostic_odds_ratio'] = float('inf')
            
            # Youden's J statistic
            metrics['youden_j'] = metrics['sensitivity'] + metrics['specificity'] - 1
        
        # Probability-based metrics
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                # ROC AUC
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                
                # Precision-Recall AUC
                metrics['auc_pr'] = average_precision_score(y_true, y_prob)
                
                # Brier Score (lower is better)
                metrics['brier_score'] = brier_score_loss(y_true, y_prob)
                
                # Log Loss (lower is better)
                y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
                y_prob_2d = np.column_stack([1 - y_prob_clipped, y_prob_clipped])
                metrics['log_loss'] = log_loss(y_true, y_prob_2d)
                
                # Calibration metrics
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_prob, n_bins=10, strategy='uniform'
                )
                
                # Expected Calibration Error (ECE)
                bin_boundaries = np.linspace(0, 1, 11)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                ece = 0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = y_true[in_bin].mean()
                        avg_confidence_in_bin = y_prob[in_bin].mean()
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                metrics['expected_calibration_error'] = ece
                
                # Maximum Calibration Error (MCE)
                mce = 0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = y_true[in_bin].mean()
                        avg_confidence_in_bin = y_prob[in_bin].mean()
                        mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
                
                metrics['max_calibration_error'] = mce
                
                # Reliability diagram data
                metrics['calibration_curve'] = {
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist()
                }
                
            except Exception as e:
                print(f"Warning: Could not calculate probability-based metrics: {e}")
                metrics.update({
                    'auc_roc': 0.0, 'auc_pr': 0.0, 'brier_score': 1.0,
                    'log_loss': 1.0, 'expected_calibration_error': 1.0,
                    'max_calibration_error': 1.0, 'calibration_curve': None
                })
        else:
            metrics.update({
                'auc_roc': 0.0, 'auc_pr': 0.0, 'brier_score': 1.0,
                'log_loss': 1.0, 'expected_calibration_error': 1.0,
                'max_calibration_error': 1.0, 'calibration_curve': None
            })
        
        # Cross-validation metrics
        if cv_scores:
            cv_scores = np.array(cv_scores)
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            metrics['cv_min'] = cv_scores.min()
            metrics['cv_max'] = cv_scores.max()
            metrics['cv_scores'] = cv_scores.tolist()
            
            # CV stability metrics
            metrics['cv_stability'] = 1.0 - (cv_scores.std() / cv_scores.mean()) if cv_scores.mean() > 0 else 0.0
            metrics['cv_coefficient_variation'] = cv_scores.std() / cv_scores.mean() if cv_scores.mean() > 0 else 1.0
        else:
            metrics.update({
                'cv_mean': 0.0, 'cv_std': 0.0, 'cv_min': 0.0, 'cv_max': 0.0,
                'cv_scores': [], 'cv_stability': 0.0, 'cv_coefficient_variation': 1.0
            })
        
        # Training efficiency metrics
        metrics['training_time'] = training_time
        metrics['efficiency_score'] = metrics['accuracy'] / max(training_time, 1.0)
        
        # Model metadata
        metrics['model_name'] = model_name
        metrics['timestamp'] = time.time()
        
        return metrics
    
    @staticmethod
    def calculate_learning_curves(estimator, X, y, cv=5, train_sizes=None):
        """Calculate learning curves for the model."""
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                estimator, X, y, cv=cv, train_sizes=train_sizes,
                scoring='accuracy', n_jobs=-1, random_state=42
            )
            
            return {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': val_scores.mean(axis=1).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist()
            }
        except Exception as e:
            print(f"Warning: Could not calculate learning curves: {e}")
            return None
    
    @staticmethod
    def calculate_validation_curves(estimator, X, y, param_name, param_range, cv=5):
        """Calculate validation curves for hyperparameter analysis."""
        try:
            train_scores, val_scores = validation_curve(
                estimator, X, y, param_name=param_name, param_range=param_range,
                cv=cv, scoring='accuracy', n_jobs=-1
            )
            
            return {
                'param_name': param_name,
                'param_range': param_range.tolist() if hasattr(param_range, 'tolist') else list(param_range),
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': val_scores.mean(axis=1).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist()
            }
        except Exception as e:
            print(f"Warning: Could not calculate validation curves: {e}")
            return None


class ModelPerformanceAnalyzer:
    """Analyze and compare model performance across multiple dimensions."""
    
    def __init__(self, output_dir: str = "analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_model_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis of model performance."""
        analysis = {}
        
        # Extract performance metrics
        models_data = []
        for model_key, model_results in results.items():
            if model_key.startswith('_'):  # Skip metadata
                continue
                
            model_data = {
                'model_key': model_key,
                'method': model_results.get('method', model_key),
                'accuracy': model_results.get('test_accuracy', 0.0),
                'f1_weighted': model_results.get('test_f1', 0.0),
                'precision_weighted': model_results.get('test_precision', 0.0),
                'recall_weighted': model_results.get('test_recall', 0.0),
                'auc_roc': model_results.get('test_auc', 0.0),
                'training_time': model_results.get('training_time', 0.0),
                'cv_mean': model_results.get('cv_mean', 0.0),
                'cv_std': model_results.get('cv_std', 0.0),
                'feature_count': model_results.get('feature_count', 0)
            }
            models_data.append(model_data)
        
        df = pd.DataFrame(models_data)
        
        if df.empty:
            return {'error': 'No model data available for analysis'}
        
        # Performance statistics
        analysis['performance_stats'] = {
            'total_models': len(df),
            'accuracy_stats': {
                'mean': df['accuracy'].mean(),
                'std': df['accuracy'].std(),
                'min': df['accuracy'].min(),
                'max': df['accuracy'].max(),
                'median': df['accuracy'].median()
            },
            'f1_stats': {
                'mean': df['f1_weighted'].mean(),
                'std': df['f1_weighted'].std(),
                'min': df['f1_weighted'].min(),
                'max': df['f1_weighted'].max(),
                'median': df['f1_weighted'].median()
            },
            'training_time_stats': {
                'mean': df['training_time'].mean(),
                'std': df['training_time'].std(),
                'min': df['training_time'].min(),
                'max': df['training_time'].max(),
                'median': df['training_time'].median()
            }
        }
        
        # Top performers
        analysis['top_performers'] = {
            'by_accuracy': df.nlargest(5, 'accuracy')[['method', 'accuracy']].to_dict('records'),
            'by_f1': df.nlargest(5, 'f1_weighted')[['method', 'f1_weighted']].to_dict('records'),
            'by_auc': df.nlargest(5, 'auc_roc')[['method', 'auc_roc']].to_dict('records'),
            'by_efficiency': df.nlargest(5, 'accuracy')[['method', 'accuracy', 'training_time']].to_dict('records')
        }
        
        # Model family analysis
        analysis['family_analysis'] = self._analyze_model_families(df)
        
        # Performance correlations
        numeric_cols = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 
                       'auc_roc', 'training_time', 'cv_mean', 'feature_count']
        correlation_matrix = df[numeric_cols].corr()
        analysis['correlations'] = correlation_matrix.to_dict()
        
        # Efficiency analysis
        df['efficiency'] = df['accuracy'] / (df['training_time'] + 1)  # Add 1 to avoid division by zero
        analysis['efficiency_analysis'] = {
            'most_efficient': df.nlargest(5, 'efficiency')[['method', 'accuracy', 'training_time', 'efficiency']].to_dict('records'),
            'efficiency_vs_accuracy_correlation': df['efficiency'].corr(df['accuracy'])
        }
        
        # Stability analysis
        df['stability'] = 1 - (df['cv_std'] / (df['cv_mean'] + 1e-8))  # Avoid division by zero
        analysis['stability_analysis'] = {
            'most_stable': df.nlargest(5, 'stability')[['method', 'cv_mean', 'cv_std', 'stability']].to_dict('records'),
            'stability_vs_accuracy_correlation': df['stability'].corr(df['accuracy'])
        }
        
        return analysis
    
    def _analyze_model_families(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by model family."""
        families = {
            'Neural Networks': df[df['method'].str.contains('neural|Neural', case=False, na=False)],
            'Classical ML': df[df['method'].str.contains('classical|Classical|Random|SVM|Logistic|Gradient|Naive|KNN|Decision|AdaBoost', case=False, na=False)],
            'Ensemble Methods': df[df['method'].str.contains('ensemble|Ensemble|Voting|Bagging|Stacking|XGBoost|LightGBM|CatBoost|Extra', case=False, na=False)],
            'Deep Learning': df[df['method'].str.contains('deep|Deep|CNN|Transformer|LSTM|Attention', case=False, na=False)],
            'Probabilistic': df[df['method'].str.contains('probabilistic|Probabilistic|Gaussian|Bernoulli|Multinomial|HMM', case=False, na=False)],
            'Advanced': df[df['method'].str.contains('advanced|Advanced|Isolation|SGD|Passive|Perceptron|Ridge|Lasso|Elastic', case=False, na=False)]
        }
        
        family_analysis = {}
        for family_name, family_df in families.items():
            if not family_df.empty:
                family_analysis[family_name] = {
                    'count': len(family_df),
                    'avg_accuracy': family_df['accuracy'].mean(),
                    'avg_f1': family_df['f1_weighted'].mean(),
                    'avg_training_time': family_df['training_time'].mean(),
                    'best_model': family_df.loc[family_df['accuracy'].idxmax(), 'method'] if len(family_df) > 0 else None,
                    'best_accuracy': family_df['accuracy'].max() if len(family_df) > 0 else 0.0
                }
        
        return family_analysis
    
    def generate_performance_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive performance report."""
        report_path = self.output_dir / 'performance_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Performance statistics
            stats = analysis['performance_stats']
            f.write(f"OVERALL STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Models Analyzed: {stats['total_models']}\n\n")
            
            f.write("ACCURACY STATISTICS:\n")
            acc_stats = stats['accuracy_stats']
            f.write(f"  Mean: {acc_stats['mean']:.4f}\n")
            f.write(f"  Std Dev: {acc_stats['std']:.4f}\n")
            f.write(f"  Min: {acc_stats['min']:.4f}\n")
            f.write(f"  Max: {acc_stats['max']:.4f}\n")
            f.write(f"  Median: {acc_stats['median']:.4f}\n\n")
            
            f.write("F1-SCORE STATISTICS:\n")
            f1_stats = stats['f1_stats']
            f.write(f"  Mean: {f1_stats['mean']:.4f}\n")
            f.write(f"  Std Dev: {f1_stats['std']:.4f}\n")
            f.write(f"  Min: {f1_stats['min']:.4f}\n")
            f.write(f"  Max: {f1_stats['max']:.4f}\n")
            f.write(f"  Median: {f1_stats['median']:.4f}\n\n")
            
            f.write("TRAINING TIME STATISTICS:\n")
            time_stats = stats['training_time_stats']
            f.write(f"  Mean: {time_stats['mean']:.2f} seconds\n")
            f.write(f"  Std Dev: {time_stats['std']:.2f} seconds\n")
            f.write(f"  Min: {time_stats['min']:.2f} seconds\n")
            f.write(f"  Max: {time_stats['max']:.2f} seconds\n")
            f.write(f"  Median: {time_stats['median']:.2f} seconds\n\n")
            
            # Top performers
            f.write("TOP PERFORMERS\n")
            f.write("-" * 15 + "\n")
            
            top_perf = analysis['top_performers']
            f.write("Top 5 by Accuracy:\n")
            for i, model in enumerate(top_perf['by_accuracy'], 1):
                f.write(f"  {i}. {model['method']}: {model['accuracy']:.4f}\n")
            
            f.write("\nTop 5 by F1-Score:\n")
            for i, model in enumerate(top_perf['by_f1'], 1):
                f.write(f"  {i}. {model['method']}: {model['f1_weighted']:.4f}\n")
            
            f.write("\nTop 5 by AUC:\n")
            for i, model in enumerate(top_perf['by_auc'], 1):
                f.write(f"  {i}. {model['method']}: {model['auc_roc']:.4f}\n")
            
            # Model family analysis
            if 'family_analysis' in analysis:
                f.write(f"\n\nMODEL FAMILY ANALYSIS\n")
                f.write("-" * 25 + "\n")
                
                for family_name, family_stats in analysis['family_analysis'].items():
                    f.write(f"\n{family_name}:\n")
                    f.write(f"  Count: {family_stats['count']}\n")
                    f.write(f"  Avg Accuracy: {family_stats['avg_accuracy']:.4f}\n")
                    f.write(f"  Avg F1: {family_stats['avg_f1']:.4f}\n")
                    f.write(f"  Avg Training Time: {family_stats['avg_training_time']:.2f}s\n")
                    f.write(f"  Best Model: {family_stats['best_model']}\n")
                    f.write(f"  Best Accuracy: {family_stats['best_accuracy']:.4f}\n")
            
            # Efficiency analysis
            if 'efficiency_analysis' in analysis:
                f.write(f"\n\nEFFICIENCY ANALYSIS\n")
                f.write("-" * 20 + "\n")
                
                eff_analysis = analysis['efficiency_analysis']
                f.write("Most Efficient Models (Accuracy/Time):\n")
                for i, model in enumerate(eff_analysis['most_efficient'], 1):
                    f.write(f"  {i}. {model['method']}: {model['efficiency']:.6f} "
                           f"(Acc: {model['accuracy']:.4f}, Time: {model['training_time']:.2f}s)\n")
                
                f.write(f"\nEfficiency vs Accuracy Correlation: {eff_analysis['efficiency_vs_accuracy_correlation']:.4f}\n")
            
            # Stability analysis
            if 'stability_analysis' in analysis:
                f.write(f"\n\nSTABILITY ANALYSIS\n")
                f.write("-" * 18 + "\n")
                
                stab_analysis = analysis['stability_analysis']
                f.write("Most Stable Models (Low CV Variation):\n")
                for i, model in enumerate(stab_analysis['most_stable'], 1):
                    f.write(f"  {i}. {model['method']}: {model['stability']:.4f} "
                           f"(CV: {model['cv_mean']:.4f}±{model['cv_std']:.4f})\n")
                
                f.write(f"\nStability vs Accuracy Correlation: {stab_analysis['stability_vs_accuracy_correlation']:.4f}\n")
        
        print(f"Saved performance analysis report to {report_path}")
        return str(report_path)


def enhance_training_script_metrics(script_path: str, backup: bool = True) -> bool:
    """Enhance existing training scripts to include comprehensive metrics."""
    script_path = Path(script_path)
    
    if not script_path.exists():
        print(f"Script {script_path} does not exist")
        return False
    
    # Create backup if requested
    if backup:
        backup_path = script_path.with_suffix(script_path.suffix + '.backup')
        backup_path.write_text(script_path.read_text())
        print(f"Created backup at {backup_path}")
    
    # Read the original script
    original_content = script_path.read_text()
    
    # Add enhanced metrics import at the top
    enhanced_import = """
# Enhanced metrics calculation
from scripts.enhanced_model_trainer import EnhancedMetricsCalculator

"""
    
    # Add enhanced metrics calculation in the training function
    enhanced_metrics_code = """
    # Calculate comprehensive metrics
    metrics_calculator = EnhancedMetricsCalculator()
    comprehensive_metrics = metrics_calculator.calculate_comprehensive_metrics(
        y_true=y_test,
        y_pred=test_predictions,
        y_prob=test_probabilities if 'test_probabilities' in locals() else None,
        cv_scores=cv_scores if 'cv_scores' in locals() else None,
        training_time=training_time if 'training_time' in locals() else 0.0,
        model_name=method_name if 'method_name' in locals() else 'Unknown'
    )
    
    # Merge with existing results
    if 'result' in locals():
        result.update(comprehensive_metrics)
    else:
        result = comprehensive_metrics
"""
    
    # Insert the import after existing imports
    if "import" in original_content:
        lines = original_content.split('\n')
        import_end = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import') or line.strip().startswith('from'):
                import_end = i
        
        lines.insert(import_end + 1, enhanced_import)
        modified_content = '\n'.join(lines)
    else:
        modified_content = enhanced_import + original_content
    
    # Add enhanced metrics calculation before return statements
    # This is a simplified approach - in practice, you'd want more sophisticated parsing
    modified_content = modified_content.replace(
        "return result",
        enhanced_metrics_code + "\n    return result"
    )
    
    # Write the modified script
    script_path.write_text(modified_content)
    print(f"Enhanced {script_path} with comprehensive metrics")
    
    return True


def create_unified_evaluation_pipeline():
    """Create a unified evaluation pipeline script."""
    pipeline_script = """#!/usr/bin/env python3
'''
Unified Model Evaluation Pipeline

This script runs the complete evaluation pipeline including training,
comprehensive metrics calculation, and advanced visualization.
'''
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    '''Run a command and handle errors.'''
    print(f"\\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("SUCCESS!")
        if result.stdout:
            print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    '''Run the complete evaluation pipeline.'''
    print("UNIFIED MODEL EVALUATION PIPELINE")
    print("="*60)
    
    # Define pipeline steps
    steps = [
        {
            'command': 'python scripts/compare_all_methods.py --human-file corpora/original_only/combined_original_only_20250827_100900.jsonl --ai-file corpora/rewritten_pairs/combined_rewritten_pairs_20250807_133424.jsonl --output-file comparison_results_enhanced.json',
            'description': 'Train and compare all models with enhanced metrics'
        },
        {
            'command': 'python scripts/advanced_model_evaluator.py --results-file comparison_results_enhanced.json --output-dir plots/comprehensive',
            'description': 'Run advanced model evaluation and create visualizations'
        },
        {
            'command': 'python scripts/plot_model_accuracy.py --models-dir models --output-dir plots/standard',
            'description': 'Create standard accuracy plots'
        }
    ]
    
    # Execute pipeline steps
    success_count = 0
    for i, step in enumerate(steps, 1):
        print(f"\\nSTEP {i}/{len(steps)}: {step['description']}")
        
        if run_command(step['command'], step['description']):
            success_count += 1
        else:
            print(f"Step {i} failed. Continuing with remaining steps...")
    
    # Summary
    print(f"\\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Completed {success_count}/{len(steps)} steps successfully")
    
    if success_count == len(steps):
        print("\\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\\nGenerated outputs:")
        print("  - comparison_results_enhanced.json: Enhanced model comparison results")
        print("  - plots/comprehensive/: Advanced visualizations and analysis")
        print("  - plots/standard/: Standard accuracy plots")
    else:
        print(f"\\n⚠️  {len(steps) - success_count} steps failed")
        print("Check the error messages above for details")
    
    print(f"\\n{'='*60}")

if __name__ == "__main__":
    main()
"""
    
    # Write the pipeline script
    pipeline_path = Path("scripts/run_evaluation_pipeline.py")
    pipeline_path.write_text(pipeline_script)
    print(f"Created unified evaluation pipeline at {pipeline_path}")


def main():
    """Main function for enhanced model trainer utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Model Trainer Utilities')
    parser.add_argument('--action', type=str, choices=['analyze', 'enhance', 'pipeline'], 
                       default='analyze', help='Action to perform')
    parser.add_argument('--results-file', type=str, default='comparison_results_safe.json',
                       help='Results file to analyze')
    parser.add_argument('--script-path', type=str, help='Script path to enhance')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                       help='Output directory for analysis')
    
    args = parser.parse_args()
    
    if args.action == 'analyze':
        # Analyze model performance
        analyzer = ModelPerformanceAnalyzer(args.output_dir)
        
        try:
            with open(args.results_file, 'r') as f:
                results = json.load(f)
            
            analysis = analyzer.analyze_model_performance(results)
            report_path = analyzer.generate_performance_report(analysis)
            
            print(f"Analysis complete! Report saved to: {report_path}")
            
        except FileNotFoundError:
            print(f"Results file {args.results_file} not found")
        except Exception as e:
            print(f"Error during analysis: {e}")
    
    elif args.action == 'enhance':
        # Enhance training script
        if not args.script_path:
            print("--script-path required for enhance action")
            return
        
        success = enhance_training_script_metrics(args.script_path)
        if success:
            print(f"Successfully enhanced {args.script_path}")
        else:
            print(f"Failed to enhance {args.script_path}")
    
    elif args.action == 'pipeline':
        # Create evaluation pipeline
        create_unified_evaluation_pipeline()
        print("Created unified evaluation pipeline script")


if __name__ == "__main__":
    main()
