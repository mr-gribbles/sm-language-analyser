#!/usr/bin/env python3
"""Language pattern analysis script for AI vs Human text detection.

This script analyzes the most influential language patterns that distinguish
AI-generated from human-written social media posts using interpretable ML models.
"""
import sys
import os
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.interpretable_classifiers import InterpretableTextClassifier


def analyze_feature_categories(classifier, save_path=None):
    """Analyze features by category and create comprehensive visualizations."""
    if classifier.feature_names_ is None:
        raise ValueError("Model not trained yet")
    
    # Get all features with importance scores
    top_features = classifier.get_influential_features(n_features=50)
    
    # Separate linguistic and word features
    linguistic_features = top_features[top_features['type'] == 'linguistic'].copy()
    word_features = top_features[top_features['type'] == 'n-gram'].copy()
    
    # Clean feature names
    linguistic_features['clean_name'] = linguistic_features['feature'].str.replace('ling_', '')
    word_features['clean_name'] = word_features['feature'].str.replace('word_', '')
    
    # Create comprehensive analysis
    analysis = {
        'linguistic_features': {
            'total_count': len(linguistic_features),
            'top_10': linguistic_features.head(10),
            'categories': {}
        },
        'word_features': {
            'total_count': len(word_features),
            'top_10': word_features.head(10),
            'most_important_words': word_features.head(20)['clean_name'].tolist()
        }
    }
    
    # Categorize linguistic features
    categories = {
        'Readability': ['flesch_readability', 'avg_words_per_sentence', 'avg_sentence_length_chars'],
        'Vocabulary Complexity': ['lexical_diversity', 'bigram_diversity', 'avg_word_length', 'short_word_ratio', 'long_word_ratio'],
        'Writing Style': ['function_word_ratio', 'capitalized_word_ratio', 'repeated_char_ratio'],
        'Punctuation Patterns': ['punctuation_ratio', 'exclamation_ratio', 'question_ratio', 'comma_ratio', 'period_ratio'],
        'Text Structure': ['sentence_count', 'word_count', 'text_length', 'word_density'],
        'Character Patterns': ['uppercase_ratio', 'lowercase_ratio', 'digit_ratio', 'vowel_ratio', 'consonant_ratio']
    }
    
    for category, features in categories.items():
        category_features = linguistic_features[linguistic_features['clean_name'].isin(features)]
        if not category_features.empty:
            analysis['linguistic_features']['categories'][category] = {
                'features': category_features.to_dict('records'),
                'avg_importance': category_features['abs_importance'].mean(),
                'max_importance': category_features['abs_importance'].max()
            }
    
    # Create visualizations
    if save_path:
        create_pattern_visualizations(analysis, save_path)
    
    return analysis


def create_pattern_visualizations(analysis, save_dir):
    """Create comprehensive visualizations of language patterns."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Linguistic features by category
    plt.figure(figsize=(14, 8))
    
    categories = []
    importances = []
    feature_names = []
    
    for category, data in analysis['linguistic_features']['categories'].items():
        for feature in data['features']:
            categories.append(category)
            importances.append(feature['abs_importance'])
            feature_names.append(feature['clean_name'])
    
    if categories:
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Category': categories,
            'Importance': importances,
            'Feature': feature_names
        })
        
        # Sort by importance within each category
        df = df.sort_values(['Category', 'Importance'], ascending=[True, False])
        
        # Create grouped bar plot
        plt.figure(figsize=(16, 10))
        
        # Use different colors for each category
        colors = plt.cm.Set3(range(len(df['Category'].unique())))
        category_colors = dict(zip(df['Category'].unique(), colors))
        
        bars = plt.bar(range(len(df)), df['Importance'], 
                      color=[category_colors[cat] for cat in df['Category']])
        
        plt.xticks(range(len(df)), df['Feature'], rotation=45, ha='right')
        plt.xlabel('Linguistic Features')
        plt.ylabel('Importance Score')
        plt.title('Most Important Linguistic Features for AI vs Human Detection\n(Grouped by Category)')
        
        # Add category labels
        current_category = None
        category_positions = []
        category_labels = []
        
        for i, category in enumerate(df['Category']):
            if category != current_category:
                category_positions.append(i)
                category_labels.append(category)
                current_category = category
        
        # Add vertical lines to separate categories
        for pos in category_positions[1:]:
            plt.axvline(x=pos-0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Create legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=category_colors[cat], label=cat) 
                          for cat in df['Category'].unique()]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'linguistic_features_by_category.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Category importance comparison
    if analysis['linguistic_features']['categories']:
        plt.figure(figsize=(12, 8))
        
        category_names = list(analysis['linguistic_features']['categories'].keys())
        avg_importances = [data['avg_importance'] for data in analysis['linguistic_features']['categories'].values()]
        max_importances = [data['max_importance'] for data in analysis['linguistic_features']['categories'].values()]
        
        x = range(len(category_names))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], avg_importances, width, label='Average Importance', alpha=0.8)
        plt.bar([i + width/2 for i in x], max_importances, width, label='Maximum Importance', alpha=0.8)
        
        plt.xlabel('Feature Categories')
        plt.ylabel('Importance Score')
        plt.title('Feature Category Importance Comparison')
        plt.xticks(x, category_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / 'category_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Word cloud of most important words
    try:
        from wordcloud import WordCloud
        
        if analysis['word_features']['most_important_words']:
            # Create word cloud with importance weights
            word_freq = {}
            for i, word in enumerate(analysis['word_features']['most_important_words'][:30]):
                # Weight by inverse rank (higher rank = higher weight)
                word_freq[word] = 30 - i
            
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Most Important Words/N-grams for AI Detection')
            plt.tight_layout()
            plt.savefig(save_dir / 'important_words_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
    except ImportError:
        print("WordCloud not available. Skipping word cloud visualization.")
    
    print(f"Visualizations saved to {save_dir}")


def generate_pattern_report(analysis, classifier_type, save_path=None):
    """Generate a comprehensive text report of language patterns."""
    report = []
    report.append("=" * 80)
    report.append(f"LANGUAGE PATTERN ANALYSIS REPORT")
    report.append(f"Classifier: {classifier_type.title()}")
    report.append("=" * 80)
    
    # Executive Summary
    report.append("\nEXECUTIVE SUMMARY:")
    report.append("-" * 40)
    
    total_linguistic = analysis['linguistic_features']['total_count']
    total_word = analysis['word_features']['total_count']
    
    report.append(f"• Total linguistic features analyzed: {total_linguistic}")
    report.append(f"• Total word/n-gram features analyzed: {total_word}")
    report.append(f"• Feature categories identified: {len(analysis['linguistic_features']['categories'])}")
    
    # Most important linguistic features
    report.append("\nTOP 10 MOST IMPORTANT LINGUISTIC FEATURES:")
    report.append("-" * 50)
    
    for idx, row in analysis['linguistic_features']['top_10'].iterrows():
        feature_name = row['feature'].replace('ling_', '')
        report.append(f"{idx+1:2d}. {feature_name:<25} (Importance: {row['abs_importance']:.4f})")
    
    # Category analysis
    report.append("\nFEATURE CATEGORY ANALYSIS:")
    report.append("-" * 40)
    
    # Sort categories by average importance
    sorted_categories = sorted(
        analysis['linguistic_features']['categories'].items(),
        key=lambda x: x[1]['avg_importance'],
        reverse=True
    )
    
    for category, data in sorted_categories:
        report.append(f"\n{category.upper()}:")
        report.append(f"  Average Importance: {data['avg_importance']:.4f}")
        report.append(f"  Maximum Importance: {data['max_importance']:.4f}")
        report.append(f"  Key Features:")
        
        # Sort features within category by importance
        sorted_features = sorted(data['features'], key=lambda x: x['abs_importance'], reverse=True)
        for feature in sorted_features[:3]:  # Top 3 in each category
            report.append(f"    • {feature['clean_name']}: {feature['abs_importance']:.4f}")
    
    # Most important words
    report.append("\nTOP 20 MOST IMPORTANT WORDS/N-GRAMS:")
    report.append("-" * 45)
    
    for i, word in enumerate(analysis['word_features']['most_important_words'][:20], 1):
        report.append(f"{i:2d}. {word}")
    
    # Interpretation
    report.append("\nINTERPRETATION AND INSIGHTS:")
    report.append("-" * 40)
    
    # Analyze patterns
    top_category = sorted_categories[0] if sorted_categories else None
    if top_category:
        report.append(f"• The most influential category is '{top_category[0]}' with an average importance of {top_category[1]['avg_importance']:.4f}")
    
    # Look for specific patterns
    linguistic_df = analysis['linguistic_features']['top_10']
    
    if 'flesch_readability' in linguistic_df['clean_name'].values:
        report.append("• Readability scores are highly important, suggesting AI and human texts differ in complexity")
    
    if 'lexical_diversity' in linguistic_df['clean_name'].values:
        report.append("• Lexical diversity is a key differentiator, indicating vocabulary usage patterns differ")
    
    if 'function_word_ratio' in linguistic_df['clean_name'].values:
        report.append("• Function word usage patterns help distinguish AI from human writing")
    
    if any('punctuation' in name for name in linguistic_df['clean_name'].values):
        report.append("• Punctuation patterns are significant indicators of AI vs human authorship")
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Report saved to {save_path}")
    
    return report_text


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze language patterns in AI vs Human text detection')
    
    # Model loading
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the saved .pkl model file.')
    
    # Analysis options
    parser.add_argument('--top-features', type=int, default=50,
                       help='Number of top features to analyze')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate comprehensive text report')
    parser.add_argument('--create-visualizations', action='store_true',
                       help='Create pattern visualizations')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                       help='Directory to save analysis results')
    parser.add_argument('--report-file', type=str, default='language_patterns_report.txt',
                       help='Filename for the text report')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the trained model
    print(f"Loading model from {args.model_path}...")
    
    try:
        classifier = InterpretableTextClassifier.load(args.model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Perform analysis
    print("Analyzing language patterns...")
    
    viz_dir = output_dir / 'visualizations' if args.create_visualizations else None
    analysis = analyze_feature_categories(classifier, save_path=viz_dir)
    
    # Generate report
    if args.generate_report:
        print("Generating comprehensive report...")
        report_path = output_dir / args.report_file
        report = generate_pattern_report(analysis, classifier.classifier_type, save_path=report_path)
        print("\n" + report)
    
    # Print summary
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved to: {output_dir}")
    
    # Quick summary
    print(f"\nQuick Summary:")
    print(f"• Most important linguistic feature: {analysis['linguistic_features']['top_10'].iloc[0]['feature'].replace('ling_', '')}")
    print(f"• Most important word/n-gram: {analysis['word_features']['most_important_words'][0]}")
    print(f"• Total features analyzed: {analysis['linguistic_features']['total_count'] + analysis['word_features']['total_count']}")


if __name__ == "__main__":
    main()
