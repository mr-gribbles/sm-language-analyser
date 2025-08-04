#!/usr/bin/env python3
"""Prediction script for the AI vs Human text classifier.

This script uses a trained classifier to predict whether input text
is AI-generated or human-written.
"""
import sys
import os
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.text_classifier import AIHumanTextClassifier


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


def predict_from_file(classifier, file_path):
    """Predict texts from a file (one text per line)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        if not texts:
            print(f"No texts found in {file_path}")
            return
        
        print(f"Predicting {len(texts)} texts from {file_path}...")
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
    print("=== Interactive AI vs Human Text Classifier ===")
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
            
            if len(text) < 10:
                print("Text is too short for reliable classification. Please enter longer text.")
                continue
            
            predict_single_text(classifier, text)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict AI vs Human text classification')
    parser.add_argument('--model-path', type=str, default='models/ai_human_classifier',
                       help='Path to the trained model (default: models/ai_human_classifier)')
    parser.add_argument('--text', type=str,
                       help='Single text to classify')
    parser.add_argument('--file', type=str,
                       help='File containing texts to classify (one per line)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not (Path(f"{model_path}_model.pth").exists() and 
            Path(f"{model_path}_vectorizer.pkl").exists() and 
            Path(f"{model_path}_scaler.pkl").exists()):
        print(f"Error: Model files not found at '{args.model_path}'")
        print("Make sure you have trained the model first using train_classifier.py")
        sys.exit(1)
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    classifier = AIHumanTextClassifier()
    
    try:
        classifier.load_model(args.model_path)
        print("Model loaded successfully!")
        print("=" * 50)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Determine mode
    if args.text:
        # Single text prediction
        predict_single_text(classifier, args.text)
    elif args.file:
        # File prediction
        predict_from_file(classifier, args.file)
    elif args.interactive:
        # Interactive mode
        interactive_mode(classifier)
    else:
        # Default to interactive mode
        print("No specific input provided. Starting interactive mode...")
        interactive_mode(classifier)


if __name__ == "__main__":
    main()
