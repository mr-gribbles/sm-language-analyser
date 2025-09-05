"""Unified model serialization utility for consolidating model components into single files."""
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import torch


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
            'word_vectorizer': package.word_vectorizer,
            'char_vectorizer': package.char_vectorizer,
            'scaler': package.scaler
        }
        
        # Handle PyTorch models by extracting state_dict
        if package.model is not None:
            if hasattr(package.model, 'state_dict'):
                # This is a PyTorch model, save its state_dict
                package_data['model_state_dict'] = package.model.state_dict()
                package_data['model_config'] = package.config
            else:
                # This is a regular model (sklearn, etc.)
                package_data['model'] = package.model
        else:
            package_data['model'] = None
        
        save_path = f"{filepath}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(package_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return save_path
    
    @staticmethod
    def load_model_package(filepath: str, device: Optional[torch.device] = None) -> ModelPackage:
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
