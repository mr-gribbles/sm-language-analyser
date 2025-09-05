"""Universal prediction script for all AI vs Human text classifiers.

This script supports neural network, classical ML, and ensemble methods
with command-line arguments to choose which model to use.
"""
import sys
import os
import argparse
from pathlib import Path
import warnings
from sklearn.exceptions import ConvergenceWarning

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.text_classifier import EnhancedAIHumanTextClassifier
from src.ml.classical_classifiers import ClassicalTextClassifier
from src.ml.ensemble_classifiers import EnsembleTextClassifier
from src.ml.hybrid_classifier import HybridTextClassifier
from src.ml.sequential_classifier import SequentialTextClassifier
from src.ml.deep_learning_classifiers import DeepLearningTextClassifier
from src.ml.probabilistic_classifiers import ProbabilisticTextClassifier
from src.ml.manifold_classifiers import ManifoldTextClassifier
from src.ml.advanced_classifiers import AdvancedTextClassifier
from src.ml.interpretable_classifiers import InterpretableTextClassifier


def predict_single_text(classifier, text, model_type):
    """Predict a single text and display results.
    
    Args:
        classifier: Trained classifier instance.
        text: Text string to classify.
        model_type: String description of the model type.
    """
    predictions, probabilities = classifier.predict([text])
    
    prediction = predictions[0]
    probability = probabilities[0]
    
    if prediction == 1:  # AI
        label = "AI-generated"
        confidence = probability
    else:  # Human
        label = "Human-written"
        confidence = 1 - probability
    
    print(f"Model: {model_type}")
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.3f}")
    print("-" * 50)


def predict_from_file(classifier, file_path, model_type, whole_document=None):
    """Predict texts from a file with flexible handling of document structure.
    
    Args:
        classifier: Trained classifier instance.
        file_path: Path to the text file to analyze.
        model_type: String description of the model type.
        whole_document: Whether to treat file as one document or separate lines.
                       If None, auto-detects based on content structure.
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
            short_lines = sum(1 for line in lines if len(line) < 100)
            whole_document = short_lines < len(lines) * 0.7 
        
        if whole_document:
            # Treat entire file as one document (preserving paragraph breaks)
            print(f"Analyzing entire document from {file_path} using {model_type}")
            print("=" * 60)
            
            # Clean up the text but preserve structure
            full_text = ' '.join(content.split())
            
            predictions, probabilities = classifier.predict([full_text])
            
            prediction = predictions[0]
            probability = probabilities[0]
            
            if prediction == 1:  # AI
                label = "AI-generated"
                confidence = probability
            else:  # Human
                label = "Human-written"
                confidence = 1 - probability
            
            print(f"Model: {model_type}")
            print(f"Document preview: {full_text[:200]}{'...' if len(full_text) > 200 else ''}")
            print(f"Document length: {len(full_text)} characters")
            print(f"Prediction: {label}")
            print(f"Confidence: {confidence:.3f}")
            print("=" * 60)
            
        else:
            # Treat each non-empty line as a separate text
            texts = [line.strip() for line in content.split('\n') if line.strip()]
            
            print(f"Analyzing {len(texts)} separate texts from {file_path} using {model_type}")
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
            
            print(f"\nSummary (using {model_type}):")
            print(f"Total texts: {len(texts)}")
            print(f"Predicted as Human: {human_count}")
            print(f"Predicted as AI: {ai_count}")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")


def interactive_mode(classifier, model_type):
    """Interactive mode for predicting texts.
    
    Args:
        classifier: Trained classifier instance.
        model_type: String description of the model type.
    """
    print(f"AI vs Human Text Classifier ({model_type})")
    print("Enter text to classify (type 'quit' to exit):")
    print("-" * 50)
    
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
            
            predict_single_text(classifier, text, model_type)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def load_neural_network(model_path):
    """Load neural network model.
    
    Args:
        model_path: Base path to the neural network model files.
        
    Returns:
        Tuple of (classifier instance, model type description).
    """
    classifier = EnhancedAIHumanTextClassifier()
    classifier.load_model(model_path)
    return classifier, "Enhanced Neural Network"


def load_classical_model(model_path, classifier_type):
    """Load classical ML model.
    
    Args:
        model_path: Base path to the classical model files.
        classifier_type: Type of classical classifier to load.
        
    Returns:
        Tuple of (classifier instance, model type description).
    """
    classifier = ClassicalTextClassifier(classifier_type=classifier_type)
    classifier.load_model(model_path)
    return classifier, f"Classical: {classifier_type.title()}"


def load_ensemble_model(model_path, ensemble_type):
    """Load ensemble model.
    
    Args:
        model_path: Base path to the ensemble model files.
        ensemble_type: Type of ensemble classifier to load.
        
    Returns:
        Tuple of (classifier instance, model type description).
    """
    classifier = EnsembleTextClassifier(ensemble_type=ensemble_type)
    classifier.load_model(model_path)
    return classifier, f"Ensemble: {ensemble_type.title()}"


def load_sequential_model(model_path):
    """Load sequential model.
    
    Args:
        model_path: Base path to the sequential model files.
        
    Returns:
        Tuple of (classifier instance, model type description).
    """
    classifier = SequentialTextClassifier()
    classifier.load_model(model_path)
    return classifier, "Sequential LSTM"


def load_hybrid_model(model_path):
    """Load hybrid model.
    
    Args:
        model_path: Base path to the hybrid model files.
        
    Returns:
        Tuple of (classifier instance, model type description).
    """
    classifier = HybridTextClassifier()
    classifier.load_model(model_path)
    return classifier, "Hybrid Sequential"


def load_deep_learning_model(model_path, model_type):
    """Load deep learning model.
    
    Args:
        model_path: Base path to the deep learning model files.
        model_type: Type of deep learning model to load.
        
    Returns:
        Tuple of (classifier instance, model type description).
    """
    classifier = DeepLearningTextClassifier(model_type=model_type)
    classifier.load_model(model_path)
    return classifier, f"Deep Learning: {model_type.upper()}"


def load_probabilistic_model(model_path, classifier_type):
    """Load probabilistic model.
    
    Args:
        model_path: Base path to the probabilistic model files.
        classifier_type: Type of probabilistic classifier to load.
        
    Returns:
        Tuple of (classifier instance, model type description).
    """
    classifier = ProbabilisticTextClassifier(classifier_type=classifier_type)
    classifier.load_model(model_path)
    return classifier, f"Probabilistic: {classifier_type.title()}"


def load_manifold_model(model_path, manifold_type):
    """Load manifold learning model.
    
    Args:
        model_path: Base path to the manifold model files.
        manifold_type: Type of manifold learning method to load.
        
    Returns:
        Tuple of (classifier instance, model type description).
    """
    try:
        classifier = ManifoldTextClassifier(manifold_type=manifold_type)
        classifier.load_model(model_path)
        return classifier, f"Probabilistic: {manifold_type.title()}"
    except Exception as e:
        raise ValueError(f"Failed to load manifold model {manifold_type}: {e}")


def load_advanced_model(model_path, classifier_type):
    """Load advanced model.
    
    Args:
        model_path: Base path to the advanced model files.
        classifier_type: Type of advanced classifier to load.
        
    Returns:
        Tuple of (classifier instance, model type description).
    """
    classifier = AdvancedTextClassifier(classifier_type=classifier_type)
    classifier.load_model(model_path)
    return classifier, f"Advanced: {classifier_type.title()}"


def load_interpretable_model(model_path, classifier_type):
    """Load interpretable model.
    
    Args:
        model_path: Base path to the interpretable model files.
        classifier_type: Type of interpretable classifier to load.
        
    Returns:
        Tuple of (classifier instance, model type description).
    """
    classifier = InterpretableTextClassifier(classifier_type=classifier_type)
    classifier.load_model(model_path)
    return classifier, f"Interpretable: {classifier_type.title()}"


def discover_available_models():
    """Discover all available trained models.
    
    Returns:
        Dictionary with model types as keys and lists of available models as values.
    """
    available_models = {
        'neural': [],
        'classical': [],
        'ensemble': [],
        'hybrid': [],
        'deep_learning': [],
        'probabilistic': [],
        'manifold': [],
        'advanced': [],
        'interpretable': []
    }
    
    models_dir = Path("models")
    if not models_dir.exists():
        return available_models
    
    # Global set to track all processed models to avoid duplicates across categories
    global_processed_models = set()
    
    # Look for models with the pattern: comparison_<method_name>.pkl (current format)
    comparison_files = list(models_dir.glob("comparison_*.pkl"))
    
    for model_file in comparison_files:
        base_name = model_file.stem.replace('comparison_', '')
        model_path = str(model_file.with_suffix(''))
        
        # Skip if already processed globally
        if model_path in global_processed_models:
            continue
        
        # Define model categories with exact matches to avoid duplicates
        model_categories = {
            # Classical models
            'classical': ['random_forest', 'svm', 'logistic_regression', 'gradient_boosting', 
                         'naive_bayes', 'knn', 'decision_tree', 'adaboost'],
            
            # Ensemble models  
            'ensemble': ['voting', 'bagging', 'stacking', 'xgboost', 'lightgbm', 
                        'catboost', 'extra_trees', 'custom_ensemble'],
            
            # Neural network models
            'neural': ['enhanced_neural_network'],
            
            # Sequential/LSTM models
            'sequential': ['lstm_attention'],
            
            # Hybrid models
            'hybrid': ['lstm_features'],
            
            # Probabilistic models
            'probabilistic': ['gaussian_nb', 'bernoulli_nb', 'multinomial_nb', 'complement_nb', 
                             'categorical_nb', 'hmm', 'gaussian_mixture'],
            
            # Deep learning models
            'deep_learning': ['cnn', 'transformer', 'attention_bilstm'],
            
            # Manifold learning models
            'manifold': ['pca', 'tsne', 'isomap', 'lle', 'spectral_embedding', 
                        'mds', 'ica', 'factor_analysis', 'truncated_svd'],
            
            # Advanced models
            'advanced': ['isolation_forest', 'one_class_svm', 'local_outlier_factor', 
                        'elliptic_envelope', 'sgd', 'passive_aggressive', 'perceptron', 
                        'ridge', 'lasso', 'elastic_net', 'huber', 'quantile', 'tweedie'],
            
            # Interpretable models
            'interpretable': ['linear_regression', 'lasso_regression', 'ridge_regression', 
                             'elastic_net_regression', 'decision_tree_classifier', 
                             'extra_tree_classifier', 'gaussian_nb_classifier']
        }
        
        # Find which category this model belongs to
        model_added = False
        for category, model_types in model_categories.items():
            if base_name in model_types:
                if category == 'sequential':
                    # Sequential models go into neural category
                    available_models['neural'].append(('sequential', model_path))
                elif category == 'manifold':
                    # Manifold models go into manifold category (but display as Probabilistic for compatibility)
                    available_models['manifold'].append((base_name, model_path))
                else:
                    available_models[category].append((base_name, model_path))
                
                global_processed_models.add(model_path)
                model_added = True
                break
        
        # If model wasn't categorized, skip it to avoid duplicates
        if not model_added:
            print(f"Warning: Unknown model type '{base_name}' - skipping")
    
    # Fallback logic removed to prevent potential duplication issues
    # The consolidated model format should be sufficient
    
    return available_models


def predict_with_all_models(text):
    """Predict text using all available models.
    
    Args:
        text: Text string to classify.
    """
    available_models = discover_available_models()
    
    total_models = (len(available_models['neural']) +
                   len(available_models['classical']) +
                   len(available_models['ensemble']) +
                   len(available_models['hybrid']) +
                   len(available_models['deep_learning']) +
                   len(available_models['probabilistic']) +
                   len(available_models['manifold']) +
                   len(available_models['advanced']) +
                   len(available_models['interpretable']))
    
    if total_models == 0:
        print("No trained models found. Train some models first using:")
        print("  python scripts/compare_all_methods.py --human-file <file> --ai-file <file>")
        return
    
    print(f"All Models Prediction ({total_models} models)")
    print("=" * 80)
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print("=" * 80)
    
    results = []
    
    # Neural network predictions
    for model_name, model_path in available_models['neural']:
        try:
            if model_name == 'sequential':
                classifier, model_type = load_sequential_model(model_path)
            else:
                classifier, model_type = load_neural_network(model_path)
            
            predictions, probabilities = classifier.predict([text])
            
            prediction = predictions[0]
            probability = probabilities[0]
            
            if prediction == 1:  # AI
                label = "AI-generated"
                confidence = probability
            else:  # Human
                label = "Human-written"
                confidence = 1 - probability
            
            results.append((model_type, label, confidence))
            
        except Exception as e:
            # Silently skip models that fail to load or predict
            pass
    
    # Classical model predictions
    for model_name, model_path in available_models['classical']:
        try:
            classifier, model_type = load_classical_model(model_path, model_name)
            predictions, probabilities = classifier.predict([text])
            
            prediction = predictions[0]
            probability = probabilities[0]
            
            if prediction == 1:  # AI
                label = "AI-generated"
                confidence = probability
            else:  # Human
                label = "Human-written"
                confidence = 1 - probability
            
            results.append((model_type, label, confidence))
            
        except Exception as e:
            # Silently skip models that fail to load or predict
            pass
    
    # Ensemble model predictions
    for model_name, model_path in available_models['ensemble']:
        try:
            classifier, model_type = load_ensemble_model(model_path, model_name)
            predictions, probabilities = classifier.predict([text])
            
            prediction = predictions[0]
            probability = probabilities[0]
            
            if prediction == 1:  # AI
                label = "AI-generated"
                confidence = probability
            else:  # Human
                label = "Human-written"
                confidence = 1 - probability
            
            results.append((model_type, label, confidence))
            
        except Exception as e:
            # Silently skip models that fail to load or predict
            pass

    # Hybrid model predictions
    for model_name, model_path in available_models['hybrid']:
        try:
            classifier, model_type = load_hybrid_model(model_path)
            predictions, probabilities = classifier.predict([text])
            
            prediction = predictions[0]
            probability = probabilities[0]
            
            if prediction == 1:  # AI
                label = "AI-generated"
                confidence = probability
            else:  # Human
                label = "Human-written"
                confidence = 1 - probability
            
            results.append((model_type, label, confidence))
            
        except Exception as e:
            # Silently skip models that fail to load or predict
            pass
    
    # Deep learning model predictions
    for model_name, model_path in available_models['deep_learning']:
        try:
            classifier, model_type = load_deep_learning_model(model_path, model_name)
            predictions, probabilities = classifier.predict([text])
            
            prediction = predictions[0]
            probability = probabilities[0]
            
            if prediction == 1:  # AI
                label = "AI-generated"
                confidence = probability
            else:  # Human
                label = "Human-written"
                confidence = 1 - probability
            
            results.append((model_type, label, confidence))
            
        except Exception as e:
            # Silently skip models that fail to load or predict
            pass
    
    # Probabilistic model predictions
    for model_name, model_path in available_models['probabilistic']:
        try:
            classifier, model_type = load_probabilistic_model(model_path, model_name)
            predictions, probabilities = classifier.predict([text])
            
            prediction = predictions[0]
            probability = probabilities[0]
            
            if prediction == 1:  # AI
                label = "AI-generated"
                confidence = probability
            else:  # Human
                label = "Human-written"
                confidence = 1 - probability
            
            results.append((model_type, label, confidence))
            
        except Exception as e:
            # Silently skip models that fail to load or predict
            pass
    
    # Manifold learning model predictions
    for model_name, model_path in available_models['manifold']:
        try:
            classifier, model_type = load_manifold_model(model_path, model_name)
            predictions, probabilities = classifier.predict([text])
            
            prediction = predictions[0]
            probability = probabilities[0]
            
            if prediction == 1:  # AI
                label = "AI-generated"
                confidence = probability
            else:  # Human
                label = "Human-written"
                confidence = 1 - probability
            
            results.append((model_type, label, confidence))
            
        except Exception as e:
            # Silently skip models that fail to load or predict
            pass
    
    # Advanced model predictions
    for model_name, model_path in available_models['advanced']:
        try:
            classifier, model_type = load_advanced_model(model_path, model_name)
            predictions, probabilities = classifier.predict([text])
            
            prediction = predictions[0]
            probability = probabilities[0]
            
            if prediction == 1:  # AI
                label = "AI-generated"
                confidence = probability
            else:  # Human
                label = "Human-written"
                confidence = 1 - probability
            
            results.append((model_type, label, confidence))
            
        except Exception as e:
            # Silently skip models that fail to load or predict
            pass
    
    # Interpretable model predictions
    for model_name, model_path in available_models['interpretable']:
        try:
            classifier, model_type = load_interpretable_model(model_path, model_name)
            predictions, probabilities = classifier.predict([text])
            
            prediction = predictions[0]
            probability = probabilities[0]
            
            if prediction == 1:  # AI
                label = "AI-generated"
                confidence = probability
            else:  # Human
                label = "Human-written"
                confidence = 1 - probability
            
            results.append((model_type, label, confidence))
            
        except Exception as e:
            # Silently skip models that fail to load or predict
            pass
    
    # Display results in organized tables
    if results:
        # Separate results by prediction type
        ai_results = [(model_type, confidence) for model_type, label, confidence in results if label == "AI-generated"]
        human_results = [(model_type, confidence) for model_type, label, confidence in results if label == "Human-written"]
        
        # Sort by confidence (highest to lowest)
        ai_results.sort(key=lambda x: x[1], reverse=True)
        human_results.sort(key=lambda x: x[1], reverse=True)
        
        print()
        print("=" * 90)
        print("PREDICTION RESULTS - ORGANIZED BY CLASSIFICATION")
        print("=" * 90)
        
        # AI-generated predictions table
        if ai_results:
            print(f"\n🤖 MODELS PREDICTING AI-GENERATED ({len(ai_results)} models)")
            print("-" * 60)
            print(f"{'Model':<35} {'Confidence':<10}")
            print("-" * 60)
            for model_type, confidence in ai_results:
                print(f"{model_type:<35} {confidence:.3f}")
        
        # Human-written predictions table
        if human_results:
            print(f"\n👤 MODELS PREDICTING HUMAN-WRITTEN ({len(human_results)} models)")
            print("-" * 60)
            print(f"{'Model':<35} {'Confidence':<10}")
            print("-" * 60)
            for model_type, confidence in human_results:
                print(f"{model_type:<35} {confidence:.3f}")
        
        # Summary statistics
        print("\n" + "=" * 90)
        print("CONSENSUS SUMMARY")
        print("=" * 90)
        ai_predictions = len(ai_results)
        human_predictions = len(human_results)
        total_predictions = len(results)
        
        print(f"Total models: {total_predictions}")
        print(f"AI-generated predictions: {ai_predictions} ({ai_predictions/total_predictions*100:.1f}%)")
        print(f"Human-written predictions: {human_predictions} ({human_predictions/total_predictions*100:.1f}%)")
        
        # Calculate average confidence for each prediction type
        if ai_results:
            avg_ai_conf = sum(conf for _, conf in ai_results) / len(ai_results)
            print(f"Average AI confidence: {avg_ai_conf:.3f}")
        
        if human_results:
            avg_human_conf = sum(conf for _, conf in human_results) / len(human_results)
            print(f"Average Human confidence: {avg_human_conf:.3f}")
        
        # Overall consensus
        if human_predictions > ai_predictions:
            consensus = "HUMAN-WRITTEN"
            margin = human_predictions - ai_predictions
        elif ai_predictions > human_predictions:
            consensus = "AI-GENERATED"
            margin = ai_predictions - human_predictions
        else:
            consensus = "TIE"
            margin = 0
        
        print(f"\nOVERALL CONSENSUS: {consensus}")
        if margin > 0:
            print(f"Margin: {margin} models ({margin/total_predictions*100:.1f}%)")
        print("=" * 90)


def predict_file_with_all_models(file_path, whole_document=None, separate_lines=None):
    """Predict texts from a file using all available models.
    
    Args:
        file_path: Path to the text file to analyze.
        whole_document: Whether to treat file as one document.
        separate_lines: Whether to treat each line as separate text.
    """
    available_models = discover_available_models()
    
    total_models = (len(available_models['neural']) +
                   len(available_models['classical']) +
                   len(available_models['ensemble']) +
                   len(available_models['hybrid']) +
                   len(available_models['deep_learning']) +
                   len(available_models['probabilistic']) +
                   len(available_models['manifold']) +
                   len(available_models['advanced']) +
                   len(available_models['interpretable']))
    
    if total_models == 0:
        print("No trained models found. Train some models first using:")
        print("  python scripts/compare_all_methods.py --human-file <file> --ai-file <file>")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print(f"No content found in {file_path}")
            return
        
        # Determine document structure
        whole_doc = None
        if whole_document:
            whole_doc = True
        elif separate_lines:
            whole_doc = False
        else:
            # Auto-detect
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            short_lines = sum(1 for line in lines if len(line) < 100)
            whole_doc = short_lines < len(lines) * 0.7
        
        print(f"All Models File Analysis ({total_models} models)")
        print("=" * 80)
        print(f"File: {file_path}")
        
        if whole_doc:
            # Treat entire file as one document
            full_text = ' '.join(content.split())
            print(f"Document length: {len(full_text)} characters")
            print(f"Processing as: Single document")
            print("=" * 80)
            
            all_results = []
            
            # Get predictions from all models
            for model_name, model_path in available_models['neural']:
                try:
                    classifier, model_type = load_neural_network(model_path)
                    predictions, probabilities = classifier.predict([full_text])
                    
                    prediction = predictions[0]
                    probability = probabilities[0]
                    
                    if prediction == 1:  # AI
                        label = "AI-generated"
                        confidence = probability
                    else:  # Human
                        label = "Human-written"
                        confidence = 1 - probability
                    
                    all_results.append((model_type, label, confidence))
                    print(f"{model_type:<30} {label:<15} {confidence:.3f}")
                    
                except Exception as e:
                    # Silently skip models that fail to load or predict
                    pass
            
            for model_name, model_path in available_models['classical']:
                try:
                    classifier, model_type = load_classical_model(model_path, model_name)
                    predictions, probabilities = classifier.predict([full_text])
                    
                    prediction = predictions[0]
                    probability = probabilities[0]
                    
                    if prediction == 1:  # AI
                        label = "AI-generated"
                        confidence = probability
                    else:  # Human
                        label = "Human-written"
                        confidence = 1 - probability
                    
                    all_results.append((model_type, label, confidence))
                    print(f"{model_type:<30} {label:<15} {confidence:.3f}")
                    
                except Exception as e:
                    # Silently skip models that fail to load or predict
                    pass
            
            for model_name, model_path in available_models['ensemble']:
                try:
                    classifier, model_type = load_ensemble_model(model_path, model_name)
                    predictions, probabilities = classifier.predict([full_text])
                    
                    prediction = predictions[0]
                    probability = probabilities[0]
                    
                    if prediction == 1:  # AI
                        label = "AI-generated"
                        confidence = probability
                    else:  # Human
                        label = "Human-written"
                        confidence = 1 - probability
                    
                    all_results.append((model_type, label, confidence))
                    print(f"{model_type:<30} {label:<15} {confidence:.3f}")
                    
                except Exception as e:
                    # Silently skip models that fail to load or predict
                    pass
            
            # Summary for single document
            if all_results:
                print("=" * 80)
                ai_predictions = sum(1 for _, label, _ in all_results if label == "AI-generated")
                human_predictions = len(all_results) - ai_predictions
                
                print(f"DOCUMENT CONSENSUS:")
                print(f"AI-generated predictions: {ai_predictions}/{len(all_results)} ({ai_predictions/len(all_results)*100:.1f}%)")
                print(f"Human-written predictions: {human_predictions}/{len(all_results)} ({human_predictions/len(all_results)*100:.1f}%)")
                
                ai_confidences = [conf for _, label, conf in all_results if label == "AI-generated"]
                human_confidences = [conf for _, label, conf in all_results if label == "Human-written"]
                
                if ai_confidences:
                    avg_ai_conf = sum(ai_confidences) / len(ai_confidences)
                    print(f"Average AI confidence: {avg_ai_conf:.3f}")
                
                if human_confidences:
                    avg_human_conf = sum(human_confidences) / len(human_confidences)
                    print(f"Average Human confidence: {avg_human_conf:.3f}")
        
        else:
            # Treat each line as separate text
            texts = [line.strip() for line in content.split('\n') if line.strip()]
            print(f"Number of texts: {len(texts)}")
            print(f"Processing as: Separate lines")
            print("=" * 80)
            
            # This would be very verbose, so provide summary instead
            print("Processing each line with all models...")
            
            model_summaries = {}
            
            # Process with each model
            for model_name, model_path in available_models['neural']:
                try:
                    classifier, model_type = load_neural_network(model_path)
                    predictions, probabilities = classifier.predict(texts)
                    
                    ai_count = sum(predictions)
                    human_count = len(predictions) - ai_count
                    avg_confidence = sum(probabilities) / len(probabilities)
                    
                    model_summaries[model_type] = {
                        'ai_count': ai_count,
                        'human_count': human_count,
                        'avg_confidence': avg_confidence
                    }
                    
                except Exception as e:
                    model_summaries[model_type] = {'error': str(e)}
            
            for model_name, model_path in available_models['classical']:
                try:
                    classifier, model_type = load_classical_model(model_path, model_name)
                    predictions, probabilities = classifier.predict(texts)
                    
                    ai_count = sum(predictions)
                    human_count = len(predictions) - ai_count
                    avg_confidence = sum(probabilities) / len(probabilities)
                    
                    model_summaries[model_type] = {
                        'ai_count': ai_count,
                        'human_count': human_count,
                        'avg_confidence': avg_confidence
                    }
                    
                except Exception as e:
                    model_summaries[model_type] = {'error': str(e)}
            
            for model_name, model_path in available_models['ensemble']:
                try:
                    classifier, model_type = load_ensemble_model(model_path, model_name)
                    predictions, probabilities = classifier.predict(texts)
                    
                    ai_count = sum(predictions)
                    human_count = len(predictions) - ai_count
                    avg_confidence = sum(probabilities) / len(probabilities)
                    
                    model_summaries[model_type] = {
                        'ai_count': ai_count,
                        'human_count': human_count,
                        'avg_confidence': avg_confidence
                    }
                    
                except Exception as e:
                    model_summaries[model_type] = {'error': str(e)}
            
            # Print summary table
            print(f"{'Model':<30} {'AI Count':<10} {'Human Count':<12} {'Avg Conf':<10}")
            print("-" * 70)
            
            for model_type, summary in model_summaries.items():
                if 'error' in summary:
                    print(f"{model_type:<30} ERROR: {summary['error']}")
                else:
                    ai_count = summary['ai_count']
                    human_count = summary['human_count']
                    avg_conf = summary['avg_confidence']
                    print(f"{model_type:<30} {ai_count:<10} {human_count:<12} {avg_conf:<10.3f}")
            
            print("=" * 80)
            print("OVERALL CONSENSUS:")
            
            # Calculate overall consensus
            total_ai = sum(s.get('ai_count', 0) for s in model_summaries.values() if 'error' not in s)
            total_human = sum(s.get('human_count', 0) for s in model_summaries.values() if 'error' not in s)
            total_predictions = total_ai + total_human
            
            if total_predictions > 0:
                print(f"Total AI predictions: {total_ai}/{total_predictions} ({total_ai/total_predictions*100:.1f}%)")
                print(f"Total Human predictions: {total_human}/{total_predictions} ({total_human/total_predictions*100:.1f}%)")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")


def interactive_all_models_mode():
    """Interactive mode using all available models."""
    available_models = discover_available_models()
    
    total_models = (len(available_models['neural']) + 
                   len(available_models['classical']) + 
                   len(available_models['ensemble']) +
                   len(available_models['hybrid']) +
                   len(available_models['deep_learning']) +
                   len(available_models['probabilistic']) +
                   len(available_models['manifold']) +
                   len(available_models['advanced']) +
                   len(available_models['interpretable']))
    
    if total_models == 0:
        print("No trained models found. Train some models first using:")
        print("  python scripts/compare_all_methods.py --human-file <file> --ai-file <file>")
        return
    
    print(f"AI vs Human Text Classifier - All Models Mode ({total_models} models)")
    print("Enter text to classify (type 'quit' to exit):")
    print("-" * 80)
    
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
            
            print()  # Add blank line before results
            predict_with_all_models(text)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main prediction function."""
    # Suppress warnings during prediction
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    parser = argparse.ArgumentParser(description='Universal AI vs Human text classifier prediction')
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--neural', action='store_true',
                           help='Use neural network model')
    model_group.add_argument('--hybrid', action='store_true',
                           help='Use hybrid sequential model')
    model_group.add_argument('--classical', type=str,
                           choices=['random_forest', 'svm', 'logistic_regression', 'gradient_boosting', 
                                   'naive_bayes', 'knn', 'decision_tree', 'adaboost'],
                           help='Use classical ML model')
    model_group.add_argument('--ensemble', type=str,
                           choices=['voting', 'bagging', 'stacking', 'xgboost', 'lightgbm', 
                                   'catboost', 'extra_trees', 'custom_ensemble'],
                           help='Use ensemble model')
    model_group.add_argument('--all', action='store_true',
                           help='Use all available trained models')
    
    # Model path (optional, defaults to standard paths)
    parser.add_argument('--model-path', type=str, default=None,
                       help='Custom path to the trained model (optional, uses standard paths by default)')
    
    # Input options
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
    
    # Handle --all mode separately
    if args.all:
        # All models mode
        if args.text:
            predict_with_all_models(args.text)
        elif args.file:
            predict_file_with_all_models(args.file, args.whole_document, args.separate_lines)
        elif args.interactive:
            interactive_all_models_mode()
        else:
            interactive_all_models_mode()
        return
    
    # Set default model paths if not provided
    if args.model_path is None:
        if args.neural:
            # Check for consolidated model first
            consolidated_path = "models/ai_human_classifier_enhanced_consolidated"
            if Path(f"{consolidated_path}.pkl").exists():
                args.model_path = consolidated_path
            else:
                args.model_path = "models/ai_human_classifier_enhanced"
        elif args.hybrid:
            consolidated_path = "models/ai_human_classifier_hybrid_consolidated"
            if Path(f"{consolidated_path}.pkl").exists():
                args.model_path = consolidated_path
            else:
                args.model_path = "models/ai_human_classifier_hybrid"
        elif args.classical:
            consolidated_path = f"models/comparison_{args.classical}_{args.classical}_consolidated"
            if Path(f"{consolidated_path}.pkl").exists():
                args.model_path = consolidated_path
            else:
                args.model_path = f"models/comparison_{args.classical}"
        elif args.ensemble:
            consolidated_path = f"models/comparison_{args.ensemble}_{args.ensemble}_consolidated"
            if Path(f"{consolidated_path}.pkl").exists():
                args.model_path = consolidated_path
            else:
                args.model_path = f"models/comparison_{args.ensemble}"
    
    # Determine model type and load appropriate classifier
    try:
        if args.neural:
            # Check for consolidated model first, then fallback to multi-file
            consolidated_path = f"{args.model_path}.pkl"
            if Path(consolidated_path).exists():
                classifier, model_type = load_neural_network(args.model_path)
            else:
                # Check neural network files
                model_path = Path(args.model_path)
                required_files = [
                    f"{model_path}_enhanced_model.pth",
                    f"{model_path}_word_vectorizer.pkl",
                    f"{model_path}_char_vectorizer.pkl",
                    f"{model_path}_scaler.pkl"
                ]
                
                missing_files = [f for f in required_files if not Path(f).exists()]
                if missing_files:
                    print("Error: Neural network model files not found:")
                    for f in missing_files:
                        print(f"  - {f}")
                    print(f"\nTrain a neural network model first using:")
                    print(f"  python scripts/train_classifier.py --human-file <file> --ai-file <file> --model-path {args.model_path}")
                    sys.exit(1)
                
                classifier, model_type = load_neural_network(args.model_path)

        elif args.hybrid:
            # Check hybrid model files
            model_path = Path(args.model_path)
            required_files = [
                f"{model_path}_hybrid_model.pth",
                f"{model_path}_hybrid_word_vectorizer.pkl",
                f"{model_path}_hybrid_char_vectorizer.pkl",
                f"{model_path}_hybrid_scaler.pkl"
            ]
            
            missing_files = [f for f in required_files if not Path(f).exists()]
            if missing_files:
                print("Error: Hybrid model files not found:")
                for f in missing_files:
                    print(f"  - {f}")
                print(f"\nTrain a hybrid model first using:")
                print(f"  python scripts/train_hybrid_classifier.py --human-file <file> --ai-file <file> --model-path {args.model_path}")
                sys.exit(1)
            
            classifier, model_type = load_hybrid_model(args.model_path)
            
        elif args.classical:
            # Check classical model files
            model_path = Path(args.model_path)
            required_files = [
                f"{model_path}_{args.classical}_model.pkl",
                f"{model_path}_{args.classical}_word_vectorizer.pkl",
                f"{model_path}_{args.classical}_char_vectorizer.pkl",
                f"{model_path}_{args.classical}_scaler.pkl",
                f"{model_path}_{args.classical}_config.json"
            ]
            
            missing_files = [f for f in required_files if not Path(f).exists()]
            if missing_files:
                print("Error: Classical model files not found:")
                for f in missing_files:
                    print(f"  - {f}")
                print(f"\nTrain a classical model first using:")
                print(f"  python scripts/train_classical_classifiers.py --human-file <file> --ai-file <file> --classifier {args.classical} --model-path {args.model_path}")
                sys.exit(1)
            
            classifier, model_type = load_classical_model(args.model_path, args.classical)
            
        elif args.ensemble:
            # Check ensemble model files
            model_path = Path(args.model_path)
            required_files = [
                f"{model_path}_{args.ensemble}_ensemble_model.pkl",
                f"{model_path}_{args.ensemble}_word_vectorizer.pkl",
                f"{model_path}_{args.ensemble}_char_vectorizer.pkl",
                f"{model_path}_{args.ensemble}_scaler.pkl",
                f"{model_path}_{args.ensemble}_config.json"
            ]
            
            missing_files = [f for f in required_files if not Path(f).exists()]
            if missing_files:
                print("Error: Ensemble model files not found:")
                for f in missing_files:
                    print(f"  - {f}")
                print(f"\nTrain an ensemble model first using:")
                print(f"  python scripts/train_ensemble_classifiers.py --human-file <file> --ai-file <file> --ensemble {args.ensemble} --model-path {args.model_path}")
                sys.exit(1)
            
            classifier, model_type = load_ensemble_model(args.model_path, args.ensemble)
        
        print(f"{model_type} loaded successfully")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Determine mode and run prediction
    if args.text:
        # Single text prediction
        predict_single_text(classifier, args.text, model_type)
    elif args.file:
        # File prediction with document structure handling
        whole_doc = None
        if args.whole_document:
            whole_doc = True
        elif args.separate_lines:
            whole_doc = False
        # If neither flag is specified, auto-detect (whole_doc remains None)
        
        predict_from_file(classifier, args.file, model_type, whole_document=whole_doc)
    elif args.interactive:
        # Interactive mode
        interactive_mode(classifier, model_type)
    else:
        # Default to interactive mode
        print("No specific input provided. Starting interactive mode...")
        interactive_mode(classifier, model_type)


if __name__ == "__main__":
    main()
