"""Unified model serialization utility for consolidating model components into single files."""
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class ModelPackage:
    """Container for all model components."""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.scaler = None
        self.config = {}
        self.metadata = {}
        self.performance_metrics = {}
        self.training_history = {}
    
    def add_model(self, model):
        """Add the main model."""
        self.model = model
    
    def add_word_vectorizer(self, vectorizer):
        """Add word-level TF-IDF vectorizer."""
        self.word_vectorizer = vectorizer
    
    def add_char_vectorizer(self, vectorizer):
        """Add character-level TF-IDF vectorizer."""
        self.char_vectorizer = vectorizer
    
    def add_scaler(self, scaler):
        """Add feature scaler."""
        self.scaler = scaler
    
    def add_config(self, config: Dict[str, Any]):
        """Add configuration dictionary."""
        self.config = config
    
    def add_metadata(self, metadata: Dict[str, Any]):
        """Add metadata dictionary."""
        self.metadata = metadata
    
    def add_performance_metrics(self, metrics: Dict[str, Any]):
        """Add comprehensive performance metrics."""
        self.performance_metrics = metrics
    
    def add_training_history(self, history: Dict[str, Any]):
        """Add training history and additional data."""
        self.training_history = history


class ModelSerializer:
    """Unified model serializer for all classifier types."""
    
    @staticmethod
    def save_model_package(package: ModelPackage, filepath: str):
        """Save a complete model package to a single file.
        
        Args:
            package: ModelPackage containing all components
            filepath: Path to save the model package (without extension)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        package_data = {
            'model_type': package.model_type,
            'config': package.config,
            'metadata': package.metadata,
            'performance_metrics': package.performance_metrics,
            'training_history': package.training_history,
            'word_vectorizer': package.word_vectorizer,
            'char_vectorizer': package.char_vectorizer,
            'scaler': package.scaler
        }
        
        # Handle PyTorch models by extracting state_dict and saving proper metadata
        if package.model is not None:
            if hasattr(package.model, 'state_dict'):
                # This is a PyTorch model, save its state_dict with reconstruction info
                package_data['model_state_dict'] = package.model.state_dict()
                package_data['model_architecture'] = package.model.__class__.__name__
                package_data['is_pytorch_model'] = True
                
                # For Enhanced models, they use feature-based prediction, not sequence-based
                if package.model_type == 'enhanced':
                    package_data['pytorch_prediction_type'] = 'feature_based'
                else:
                    package_data['pytorch_prediction_type'] = 'sequence_based'
                    
                # Ensure config has the necessary reconstruction info
                if hasattr(package.model, 'input_layer') and hasattr(package.model.input_layer, 'in_features'):
                    package.config['input_dim'] = package.model.input_layer.in_features
            else:
                # This is a regular model (sklearn, etc.)
                package_data['model'] = package.model
                package_data['is_pytorch_model'] = False
        else:
            package_data['model'] = None
            package_data['is_pytorch_model'] = False
        
        save_path = f"{filepath}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(package_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return save_path
    
    @staticmethod
    def load_model_package(filepath: str, device = None) -> ModelPackage:
        """Load a complete model package from a single file.
        
        Args:
            filepath: Path to the model package file
            device: PyTorch device for loading models (if applicable)
            
        Returns:
            ModelPackage with all components loaded
        """
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.pkl')
        
        # Suppress the torch.load warning by setting weights_only appropriately
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch.storage")
            with open(filepath, 'rb') as f:
                package_data = pickle.load(f)
        
        package = ModelPackage(package_data['model_type'])
        package.config = package_data.get('config', {})
        package.metadata = package_data.get('metadata', {})
        package.performance_metrics = package_data.get('performance_metrics', {})
        package.training_history = package_data.get('training_history', {})
        package.word_vectorizer = package_data.get('word_vectorizer')
        package.char_vectorizer = package_data.get('char_vectorizer')
        package.scaler = package_data.get('scaler')
        
        # Handle PyTorch models properly
        if package.model_type in ['enhanced', 'hybrid', 'sequential', 'lstm']:
            if 'model_state_dict' in package_data:
                package.model = package_data['model_state_dict']
            elif 'model' in package_data:
                package.model = package_data['model']
            else:
                package.model = None
        else:
            package.model = package_data.get('model')
        
        return package
