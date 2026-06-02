"""
Configuration settings for the Genomic Epidemiology Pipeline.

This module contains all configuration parameters for data processing,
model training, and simulation parameters.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Configuration for data ingestion and preprocessing."""
    
    # Sequence data parameters
    min_sequence_length: int = 25000  # Minimum viral genome length
    max_sequence_length: int = 35000  # Maximum viral genome length
    sequence_identity_threshold: float = 0.95  # For alignment filtering
    
    # Environmental data parameters
    temperature_range: tuple = (-20.0, 50.0)  # Celsius
    humidity_range: tuple = (0.0, 100.0)  # Percentage
    
    # Temporal parameters
    min_date: str = "2019-12-01"  # Start of data collection
    max_date: str = "2024-01-01"  # End of data collection
    time_window_days: int = 30  # Temporal aggregation window
    
    # Geographic parameters
    geographic_resolution: str = "country"  # country, state, city
    
    # Host species categories
    host_species: List[str] = field(default_factory=lambda: [
        "human", "bat", "pangolin", "mink", "cat", "dog", "other"
    ])


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Sequence encoding parameters
    encoding_method: str = "one_hot"  # one_hot, kmer, embedding
    kmer_size: int = 3  # For k-mer encoding
    sliding_window: int = 100  # For sequence windowing
    overlap: int = 50  # Overlap between windows
    
    # Environmental feature normalization
    temperature_normalization: str = "standard"  # standard, minmax, robust
    categorical_encoding: str = "one_hot"  # one_hot, label, target
    
    # Temporal features
    include_seasonal_features: bool = True
    include_trend_features: bool = True
    lag_features: List[int] = field(default_factory=lambda: [1, 7, 14, 30])
    
    # Feature selection
    max_features: Optional[int] = 10000
    feature_selection_method: str = "variance"  # variance, mutual_info, chi2


@dataclass
class ModelConfig:
    """Configuration for the probabilistic model."""
    
    # Model architecture
    model_type: str = "transformer"  # transformer, bayesian_nn, lstm
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8  # For transformer
    dropout_rate: float = 0.1
    
    # Bayesian parameters
    prior_scale: float = 1.0
    posterior_samples: int = 100
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Regularization
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Loss function weights
    mutation_prediction_weight: float = 1.0
    fitness_prediction_weight: float = 0.5
    temporal_consistency_weight: float = 0.3


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation and hindcasting."""
    
    # Monte Carlo parameters
    num_simulations: int = 10000
    simulation_steps: int = 180  # Days to simulate forward
    confidence_levels: List[float] = field(default_factory=lambda: [0.68, 0.95, 0.99])
    
    # Hindcasting parameters
    hindcast_cutoff_date: str = "2020-12-01"
    hindcast_duration_days: int = 180
    
    # Prediction parameters
    prediction_horizon_days: int = 90
    ensemble_size: int = 10
    
    # Validation parameters
    cross_validation_folds: int = 5
    test_split_ratio: float = 0.2


@dataclass
class ComputeConfig:
    """Configuration for computational resources."""
    
    # Device settings
    device: str = "auto"  # auto, cpu, cuda, mps
    num_workers: int = 4  # For data loading
    pin_memory: bool = True
    
    # Memory management
    max_memory_gb: float = 16.0
    batch_accumulation_steps: int = 1
    
    # Parallel processing
    use_multiprocessing: bool = True
    max_processes: int = 8
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10  # Every N epochs


class Config:
    """Main configuration class that combines all sub-configurations."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration with optional YAML override."""
        
        # Default configurations
        self.data = DataConfig()
        self.features = FeatureConfig()
        self.model = ModelConfig()
        self.simulation = SimulationConfig()
        self.compute = ComputeConfig()
        
        # Project paths
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        self.logs_dir = self.project_root / "logs"
        
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.models_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Load YAML config if provided
        if config_path:
            self.load_yaml_config(config_path)
    
    def load_yaml_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file."""
        import yaml
        
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Update configurations with YAML values
        for section, values in yaml_config.items():
            if hasattr(self, section):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'data': self.data.__dict__,
            'features': self.features.__dict__,
            'model': self.model.__dict__,
            'simulation': self.simulation.__dict__,
            'compute': self.compute.__dict__
        }
    
    def save_yaml(self, output_path: Union[str, Path]) -> None:
        """Save current configuration to YAML file."""
        import yaml
        
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)


# Global configuration instance
config = Config()