#!/usr/bin/env python3
"""Enhanced prediction script for the AI vs Human text classifier.

This script uses a trained enhanced classifier to predict whether input text
is AI-generated or human-written.
"""
import sys
import os
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.text_classifier_enhanced import EnhancedAIHumanTextClassifier


def predict_single_text(classifier, text):
    """Predict a single text and display results."""
    predictions, probabilities = classifier.predict([text])
    
    prediction = predictions[0]
    probability = probabilities[0]
    
    if prediction == 1:  # AI
        label = "AI-generated"
        confidence = probability
    else:  # Human
        label = "Human-written"
        confidence = 1 - probability
    
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.3f}")
    print("-" * 50)


def predict_from_file(classifier, file_path, whole_document=None):
    """Predict texts from a file with flexible handling of document structure.
    
    Args:
        classifier: The trained classifier
        file_path: Path to the text file
        whole_document: If True, treat entire file as one document. If False, treat each line as separate text.
                       If None (default), auto-detect based on content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print(f"No content found in {file_path}")
            return
        
        # Auto-detect document structure if not specified
        if whole_document is None:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            # If most lines are short (< 100 chars), treat as separate texts
            # If most lines are long, treat as one document with paragraph breaks
            short_lines = sum(1 for line in lines if len(line) < 100)
            whole_document = short_lines < len(lines) * 0.7  # Less than 70% are short lines
        
        if whole_document:
            # Treat entire file as one document (preserving paragraph breaks)
            print(f"Analyzing entire document from {file_path}...")
            print("=" * 60)
            
            # Clean up the text but preserve structure
            full_text = ' '.join(content.split())  # Normalize whitespace but keep content together
            
            predictions, probabilities = classifier.predict([full_text])
            
            prediction = predictions[0]
            probability = probabilities[0]
            
            if prediction == 1:  # AI
                label = "AI-generated"
                confidence = probability
            else:  # Human
                label = "Human-written"
                confidence = 1 - probability
            
            print(f"Document preview: {full_text[:200]}{'...' if len(full_text) > 200 else ''}")
            print(f"Document length: {len(full_text)} characters")
            print(f"Prediction: {label}")
            print(f"Confidence: {confidence:.3f}")
            print("=" * 60)
            
        else:
            # Treat each non-empty line as a separate text
            texts = [line.strip() for line in content.split('\n') if line.strip()]
            
            print(f"Analyzing {len(texts)} separate texts from {file_path}...")
            print("=" * 60)
            
            predictions, probabilities = classifier.predict(texts)
            
            ai_count = 0
            human_count = 0
            
            for i, text in enumerate(texts):
                prediction = predictions[i]
                probability = probabilities[i]
                
                if prediction == 1:  # AI
                    label = "AI-generated"
                    confidence = probability
                    ai_count += 1
                else:  # Human
                    label = "Human-written"
                    confidence = 1 - probability
                    human_count += 1
                
                print(f"Text {i+1}: {text[:80]}{'...' if len(text) > 80 else ''}")
                print(f"Prediction: {label} (confidence: {confidence:.3f})")
                print("-" * 60)
            
            print(f"\nSummary:")
            print(f"Total texts: {len(texts)}")
            print(f"Predicted as Human: {human_count}")
            print(f"Predicted as AI: {ai_count}")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")


def interactive_mode(classifier):
    """Interactive mode for predicting texts."""
    print("=== Enhanced Interactive AI vs Human Text Classifier ===")
    print("Enter text to classify (type 'quit' to exit):")
    print("=" * 50)
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            if len(text) < 20:
                print("Text is too short for reliable classification. Please enter longer text (20+ characters).")
                continue
            
            predict_single_text(classifier, text)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict AI vs Human text classification (Enhanced)')
    parser.add_argument('--model-path', type=str, default='models/ai_human_classifier_enhanced',
                       help='Path to the trained enhanced model (default: models/ai_human_classifier_enhanced)')
    parser.add_argument('--text', type=str,
                       help='Single text to classify')
    parser.add_argument('--file', type=str,
                       help='File containing text to classify')
    parser.add_argument('--whole-document', action='store_true',
                       help='Treat entire file as one document (preserves paragraph breaks)')
    parser.add_argument('--separate-lines', action='store_true',
                       help='Treat each line as separate text')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Check if enhanced model exists
    model_path = Path(args.model_path)
    if not (Path(f"{model_path}_enhanced_model.pth").exists() and 
            Path(f"{model_path}_word_vectorizer.pkl").exists() and 
            Path(f"{model_path}_char_vectorizer.pkl").exists() and
            Path(f"{model_path}_scaler.pkl").exists()):
        print(f"Error: Enhanced model files not found at '{args.model_path}'")
        print("Expected files:")
        print(f"  - {model_path}_enhanced_model.pth")
        print(f"  - {model_path}_word_vectorizer.pkl")
        print(f"  - {model_path}_char_vectorizer.pkl")
        print(f"  - {model_path}_scaler.pkl")
        print("\nMake sure you have trained the enhanced model first using:")
        print("  python scripts/train_classifier_enhanced.py --human-file <file> --ai-file <file>")
        sys.exit(1)
    
    # Load the enhanced model
    print(f"Loading enhanced model from {args.model_path}...")
    classifier = EnhancedAIHumanTextClassifier()
    
    try:
        classifier.load_model(args.model_path)
        print("Enhanced model loaded successfully!")
        print("=" * 50)
    except Exception as e:
        print(f"Error loading enhanced model: {e}")
        sys.exit(1)
    
    # Determine mode
    if args.text:
        # Single text prediction
        predict_single_text(classifier, args.text)
    elif args.file:
        # File prediction with document structure handling
        whole_doc = None
        if args.whole_document:
            whole_doc = True
        elif args.separate_lines:
            whole_doc = False
        # If neither flag is specified, auto-detect (whole_doc remains None)
        
        predict_from_file(classifier, args.file, whole_document=whole_doc)
    elif args.interactive:
        # Interactive mode
        interactive_mode(classifier)
    else:
        # Default to interactive mode
        print("No specific input provided. Starting interactive mode...")
        interactive_mode(classifier)


if __name__ == "__main__":
    main()
