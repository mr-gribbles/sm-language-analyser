"""Tests for the AI vs Human text classifier."""
import sys
import os
import unittest
import tempfile
import shutil
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.text_classifier import AIHumanTextClassifier


class TestAIHumanTextClassifier(unittest.TestCase):
    """Test cases for the AIHumanTextClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = AIHumanTextClassifier(max_features=1000, ngram_range=(1, 2))
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_corpus(self):
        """Create a minimal test corpus for testing."""
        # Create directory structure
        original_dir = Path(self.temp_dir) / "original_only"
        rewritten_dir = Path(self.temp_dir) / "rewritten_pairs"
        original_dir.mkdir(parents=True)
        rewritten_dir.mkdir(parents=True)
        
        # Sample human texts (original)
        human_data = [
            {
                "corpus_item_id": "test1",
                "version": "1.1",
                "source_details": {"platform": "Test"},
                "original_content": {
                    "cleaned_text": "this is a human written text with natural language patterns"
                },
                "llm_transformation": None
            },
            {
                "corpus_item_id": "test2", 
                "version": "1.1",
                "source_details": {"platform": "Test"},
                "original_content": {
                    "cleaned_text": "hey everyone hope you're having a great day today lol"
                },
                "llm_transformation": None
            },
            {
                "corpus_item_id": "test3",
                "version": "1.1", 
                "source_details": {"platform": "Test"},
                "original_content": {
                    "cleaned_text": "just wanted to share my thoughts on this topic because i think its important"
                },
                "llm_transformation": None
            }
        ]
        
        # Sample AI texts (rewritten)
        ai_data = [
            {
                "corpus_item_id": "test4",
                "version": "1.1",
                "source_details": {"platform": "Test"},
                "original_content": {"cleaned_text": "original text"},
                "llm_transformation": {
                    "model_used": "test-model",
                    "prompt_template": "test prompt",
                    "rewritten_text": "This represents a human-authored text that exhibits natural linguistic characteristics.",
                    "transformation_timestamp_utc": "2025-01-01T00:00:00Z"
                }
            },
            {
                "corpus_item_id": "test5",
                "version": "1.1",
                "source_details": {"platform": "Test"},
                "original_content": {"cleaned_text": "original text"},
                "llm_transformation": {
                    "model_used": "test-model",
                    "prompt_template": "test prompt", 
                    "rewritten_text": "Greetings to all individuals, I trust you are experiencing a pleasant day today.",
                    "transformation_timestamp_utc": "2025-01-01T00:00:00Z"
                }
            },
            {
                "corpus_item_id": "test6",
                "version": "1.1",
                "source_details": {"platform": "Test"},
                "original_content": {"cleaned_text": "original text"},
                "llm_transformation": {
                    "model_used": "test-model",
                    "prompt_template": "test prompt",
                    "rewritten_text": "I wished to articulate my perspectives regarding this subject matter as I believe it holds significance.",
                    "transformation_timestamp_utc": "2025-01-01T00:00:00Z"
                }
            }
        ]
        
        # Write test data to files
        import json
        
        with open(original_dir / "test_human.jsonl", 'w') as f:
            for item in human_data:
                f.write(json.dumps(item) + '\n')
        
        with open(rewritten_dir / "test_ai.jsonl", 'w') as f:
            for item in ai_data:
                f.write(json.dumps(item) + '\n')
    
    def test_load_corpus_data(self):
        """Test loading corpus data."""
        self.create_test_corpus()
        
        texts, labels = self.classifier.load_corpus_data(self.temp_dir)
        
        # Should have 6 texts total (3 human + 3 AI)
        self.assertEqual(len(texts), 6)
        self.assertEqual(len(labels), 6)
        
        # Should have 3 human (0) and 3 AI (1) labels
        self.assertEqual(labels.count(0), 3)
        self.assertEqual(labels.count(1), 3)
    
    def test_extract_features(self):
        """Test feature extraction."""
        test_texts = [
            "this is a test sentence",
            "another test sentence here",
            "final test sentence"
        ]
        
        features = self.classifier.extract_features(test_texts)
        
        # Should return a 2D array
        self.assertEqual(len(features.shape), 2)
        self.assertEqual(features.shape[0], 3)  # 3 texts
        self.assertGreater(features.shape[1], 0)  # Some features
        
        # Vectorizer should be fitted
        self.assertIsNotNone(self.classifier.vectorizer)
        self.assertIsNotNone(self.classifier.feature_names)
    
    def test_build_model(self):
        """Test model building."""
        from src.ml.text_classifier import TextClassifierNetwork
        
        input_dim = 100
        model = TextClassifierNetwork(input_dim)
        
        # Check model structure - should have the sequential network
        self.assertIsNotNone(model.network)
        
        # Test forward pass with dummy data
        import torch
        dummy_input = torch.randn(5, input_dim)  # batch_size=5
        output = model(dummy_input)
        
        # Check output shape
        self.assertEqual(output.shape, (5, 1))
        
        # Check output is in valid range (sigmoid output)
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
    
    def test_training_with_minimal_data(self):
        """Test training with minimal data (may not converge well but should run)."""
        self.create_test_corpus()
        
        # Use very small model for quick testing
        self.classifier.max_features = 50
        
        try:
            results = self.classifier.train(
                corpus_dir=self.temp_dir,
                epochs=2,  # Very few epochs for testing
                batch_size=2,
                test_size=0.3,
                validation_size=0.2
            )
            
            # Should return results dictionary
            self.assertIn('history', results)
            self.assertIn('test_accuracy', results)
            self.assertIn('classification_report', results)
            
            # Model should be trained
            self.assertIsNotNone(self.classifier.model)
            self.assertIsNotNone(self.classifier.vectorizer)
            self.assertIsNotNone(self.classifier.scaler)
            
        except Exception as e:
            # Training might fail with very small data, but should not crash
            self.fail(f"Training crashed unexpectedly: {e}")
    
    def test_prediction_without_training(self):
        """Test that prediction fails without training."""
        test_texts = ["This is a test sentence."]
        
        with self.assertRaises(ValueError):
            self.classifier.predict(test_texts)
    
    def test_save_load_model(self):
        """Test saving and loading model (requires training first)."""
        self.create_test_corpus()
        
        # Train a minimal model
        self.classifier.max_features = 50
        self.classifier.train(
            corpus_dir=self.temp_dir,
            epochs=1,
            batch_size=2
        )
        
        # Save model
        model_path = Path(self.temp_dir) / "test_model"
        self.classifier.save_model(str(model_path))
        
        # Check files exist
        self.assertTrue(Path(f"{model_path}_model.pth").exists())
        self.assertTrue(Path(f"{model_path}_vectorizer.pkl").exists())
        self.assertTrue(Path(f"{model_path}_scaler.pkl").exists())
        
        # Load model in new classifier
        new_classifier = AIHumanTextClassifier()
        new_classifier.load_model(str(model_path))
        
        # Should have loaded components
        self.assertIsNotNone(new_classifier.model)
        self.assertIsNotNone(new_classifier.vectorizer)
        self.assertIsNotNone(new_classifier.scaler)
        
        # Should be able to make predictions
        test_texts = ["This is a test sentence."]
        predictions, probabilities = new_classifier.predict(test_texts)
        
        self.assertEqual(len(predictions), 1)
        self.assertEqual(len(probabilities), 1)
        self.assertIn(predictions[0], [0, 1])
        self.assertGreaterEqual(probabilities[0], 0.0)
        self.assertLessEqual(probabilities[0], 1.0)


class TestClassifierIntegration(unittest.TestCase):
    """Integration tests using actual corpus data if available."""
    
    def test_with_real_corpus_data(self):
        """Test with real corpus data if available."""
        corpus_dir = "corpora"
        
        if not os.path.exists(corpus_dir):
            self.skipTest("No corpus directory found - run data collection first")
        
        original_dir = Path(corpus_dir) / "original_only"
        rewritten_dir = Path(corpus_dir) / "rewritten_pairs"
        
        if not (original_dir.exists() and rewritten_dir.exists()):
            self.skipTest("Corpus directories not found - run data collection first")
        
        if not (any(original_dir.glob("*.jsonl")) and any(rewritten_dir.glob("*.jsonl"))):
            self.skipTest("No corpus files found - run data collection first")
        
        # Test loading real data
        classifier = AIHumanTextClassifier(max_features=1000)
        texts, labels = classifier.load_corpus_data(corpus_dir)
        
        self.assertGreater(len(texts), 0)
        self.assertEqual(len(texts), len(labels))
        self.assertIn(0, labels)  # Should have human texts
        self.assertIn(1, labels)  # Should have AI texts
        
        print(f"Loaded {len(texts)} texts from real corpus:")
        print(f"  Human texts: {labels.count(0)}")
        print(f"  AI texts: {labels.count(1)}")


if __name__ == '__main__':
    unittest.main()
