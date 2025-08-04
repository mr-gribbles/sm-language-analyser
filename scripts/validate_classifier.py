#!/usr/bin/env python3
"""Validation script to check classifier implementation and dependencies.

This script validates that the AI vs Human text classifier is properly
implemented and ready for use.
"""
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def check_dependencies():
    """Check if required dependencies are installed."""
    print("=== Checking Dependencies ===")
    
    missing_deps = []
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Device: {device}")
    except ImportError:
        missing_deps.append("torch")
        print("❌ PyTorch not found")
    
    try:
        import sklearn
        print(f"✅ scikit-learn: {sklearn.__version__}")
    except ImportError:
        missing_deps.append("scikit-learn")
        print("❌ scikit-learn not found")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        missing_deps.append("numpy")
        print("❌ NumPy not found")
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError:
        missing_deps.append("pandas")
        print("❌ Pandas not found")
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        missing_deps.append("matplotlib")
        print("❌ Matplotlib not found")
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install -r requirements_ml.txt")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True


def check_file_structure():
    """Check if all required files are present."""
    print("\n=== Checking File Structure ===")
    
    required_files = [
        "src/ml/text_classifier.py",
        "scripts/train_classifier.py",
        "scripts/predict_text.py",
        "scripts/demo_classifier.py",
        "tests/test_text_classifier.py",
        "requirements_ml.txt",
        "ML_README.md",
        "CLASSIFIER_SUMMARY.md"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    print("\n✅ All required files are present!")
    return True


def check_corpus_data():
    """Check if corpus data is available."""
    print("\n=== Checking Corpus Data ===")
    
    corpus_dir = Path("corpora")
    original_dir = corpus_dir / "original_only"
    rewritten_dir = corpus_dir / "rewritten_pairs"
    
    if not corpus_dir.exists():
        print("❌ No corpus directory found")
        return False
    
    human_files = list(original_dir.glob("*.jsonl")) if original_dir.exists() else []
    ai_files = list(rewritten_dir.glob("*.jsonl")) if rewritten_dir.exists() else []
    
    if not human_files:
        print("❌ No human text data found in corpora/original_only/")
        print("   Run: python main.py --platform reddit --rewrite false --num-posts 500")
    else:
        print(f"✅ Found {len(human_files)} human text file(s)")
    
    if not ai_files:
        print("❌ No AI text data found in corpora/rewritten_pairs/")
        print("   Run: python main.py --platform reddit --rewrite true --num-posts 500")
    else:
        print(f"✅ Found {len(ai_files)} AI text file(s)")
    
    return bool(human_files and ai_files)


def test_classifier_import():
    """Test importing the classifier."""
    print("\n=== Testing Classifier Import ===")
    
    try:
        from src.ml.text_classifier import AIHumanTextClassifier, TextClassifierNetwork
        print("✅ Successfully imported AIHumanTextClassifier")
        print("✅ Successfully imported TextClassifierNetwork")
        
        # Test basic initialization
        classifier = AIHumanTextClassifier(max_features=100)
        print("✅ Successfully initialized classifier")
        print(f"✅ Device detected: {classifier.device}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to import classifier: {e}")
        return False


def test_corpus_loading():
    """Test loading corpus data if available."""
    print("\n=== Testing Corpus Loading ===")
    
    try:
        from src.ml.text_classifier import AIHumanTextClassifier
        
        classifier = AIHumanTextClassifier(max_features=100)
        texts, labels = classifier.load_corpus_data("corpora")
        
        if len(texts) == 0:
            print("⚠️  No texts loaded - corpus data not available")
            return True  # Not an error, just no data
        
        print(f"✅ Successfully loaded {len(texts)} texts:")
        print(f"   Human texts: {labels.count(0)}")
        print(f"   AI texts: {labels.count(1)}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load corpus data: {e}")
        return False


def main():
    """Main validation function."""
    print("🔍 AI vs Human Text Classifier Validation")
    print("=" * 50)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Dependencies", check_dependencies),
        ("Classifier Import", test_classifier_import),
        ("Corpus Data", check_corpus_data),
        ("Corpus Loading", test_corpus_loading),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            results[check_name] = False
    
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! The classifier is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements_ml.txt")
        print("2. Generate training data if needed")
        print("3. Train your first model: python scripts/demo_classifier.py")
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")
        print("\nCommon solutions:")
        print("- Install dependencies: pip install -r requirements_ml.txt")
        print("- Generate corpus data using the main pipeline")
        print("- Check file permissions and paths")


if __name__ == "__main__":
    main()
