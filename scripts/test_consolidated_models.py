#!/usr/bin/env python3
"""Test script to demonstrate the new consolidated model functionality."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.ml.text_classifier import EnhancedAIHumanTextClassifier
from src.ml.model_serializer import ModelSerializer


def test_consolidated_loading():
    """Test loading one of the consolidated models."""
    print("=" * 60)
    print("TESTING CONSOLIDATED MODEL LOADING")
    print("=" * 60)
    
    # Check if we have any consolidated models
    models_dir = Path("models")
    consolidated_models = list(models_dir.glob("*_consolidated.pkl"))
    
    if not consolidated_models:
        print("No consolidated models found. Run the migration script first.")
        return
    
    print(f"Found {len(consolidated_models)} consolidated models:")
    for model_path in consolidated_models:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  {model_path.name} ({size_mb:.1f} MB)")
    
    # Test loading the enhanced model if available
    enhanced_model_path = models_dir / "ai_human_classifier_enhanced_consolidated.pkl"
    if enhanced_model_path.exists():
        print(f"\nTesting loading of: {enhanced_model_path.name}")
        
        try:
            # Create classifier instance
            classifier = EnhancedAIHumanTextClassifier()
            
            # Load the consolidated model
            classifier.load_model(str(enhanced_model_path.with_suffix('')))
            
            print("✅ Model loaded successfully!")
            
            # Test prediction on sample texts
            test_texts = [
                "This is a sample human-written text with natural language patterns.",
                "The following text demonstrates various linguistic characteristics and patterns."
            ]
            
            print("\nTesting predictions on sample texts...")
            predictions, probabilities = classifier.predict(test_texts)
            
            for i, (text, pred, prob) in enumerate(zip(test_texts, predictions, probabilities)):
                label = "AI-generated" if pred == 1 else "Human-written"
                confidence = prob if pred == 1 else (1 - prob)
                print(f"Text {i+1}: {label} (confidence: {confidence:.3f})")
                print(f"  Text: {text[:50]}...")
            
            print("✅ Predictions completed successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    else:
        print(f"\nEnhanced model not found at: {enhanced_model_path}")
        print("Testing with first available consolidated model...")
        
        first_model = consolidated_models[0]
        print(f"Testing: {first_model.name}")
        
        try:
            # Load using the generic model serializer
            package = ModelSerializer.load_model_package(str(first_model.with_suffix('')))
            print(f"✅ Successfully loaded model package of type: {package.model_type}")
            print(f"   Config keys: {list(package.config.keys())}")
            print(f"   Has word vectorizer: {package.word_vectorizer is not None}")
            print(f"   Has char vectorizer: {package.char_vectorizer is not None}")
            print(f"   Has scaler: {package.scaler is not None}")
            
        except Exception as e:
            print(f"❌ Error loading model package: {e}")


def demonstrate_space_savings():
    """Demonstrate the space savings from consolidation."""
    print("\n" + "=" * 60)
    print("SPACE SAVINGS ANALYSIS")
    print("=" * 60)
    
    models_dir = Path("models")
    
    # Count old files
    old_patterns = ["*_word_vectorizer.pkl", "*_char_vectorizer.pkl", "*_scaler.pkl", 
                   "*_model.pkl", "*_ensemble_model.pkl", "*_enhanced_model.pth", 
                   "*_hybrid_model.pth", "*_sequential_model.pth", "*_config.json"]
    
    old_files = []
    for pattern in old_patterns:
        old_files.extend(models_dir.glob(pattern))
    
    # Filter out consolidated files
    old_files = [f for f in old_files if not f.stem.endswith('_consolidated')]
    
    # Count consolidated files
    consolidated_files = list(models_dir.glob("*_consolidated.pkl"))
    
    # Calculate sizes
    old_total_size = sum(f.stat().st_size for f in old_files) / (1024 * 1024)
    consolidated_total_size = sum(f.stat().st_size for f in consolidated_files) / (1024 * 1024)
    
    print(f"Old multi-file approach:")
    print(f"  Files: {len(old_files)}")
    print(f"  Total size: {old_total_size:.1f} MB")
    
    print(f"\nNew consolidated approach:")
    print(f"  Files: {len(consolidated_files)}")
    print(f"  Total size: {consolidated_total_size:.1f} MB")
    
    if old_files:
        file_reduction = ((len(old_files) - len(consolidated_files)) / len(old_files)) * 100
        print(f"\nFile count reduction: {file_reduction:.1f}%")
        print(f"From {len(old_files)} files to {len(consolidated_files)} files")
    
    print(f"\nBenefits of consolidation:")
    print(f"  ✅ Easier model management (single file per model)")
    print(f"  ✅ Reduced file system clutter")
    print(f"  ✅ Atomic model operations (all components in one file)")
    print(f"  ✅ Simplified deployment and sharing")


if __name__ == "__main__":
    test_consolidated_loading()
    demonstrate_space_savings()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. All new models will automatically use the consolidated format")
    print("2. Update your existing scripts to use the new loading methods")
    print("3. Run cleanup when ready: python scripts/migrate_models.py --cleanup")
    print("4. The consolidated models are fully compatible with existing prediction workflows")
