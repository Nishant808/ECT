"""
Feature Engineering Pipeline.

This module orchestrates the complete feature engineering process,
integrating sequence encoding, environmental normalization, and temporal structuring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

from .sequence_encoder import SequenceEncoder
from .environmental_normalizer import EnvironmentalNormalizer
from .temporal_structurer import TemporalStructurer


class FeaturePipeline:
    """
    Complete feature engineering pipeline for genomic epidemiology.
    
    Orchestrates sequence encoding, environmental normalization,
    and temporal structuring to create ML-ready datasets.
    """
    
    def __init__(self,
                 sequence_config: Optional[Dict] = None,
                 environmental_config: Optional[Dict] = None,
                 temporal_config: Optional[Dict] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize the feature pipeline.
        
        Args:
            sequence_config: Configuration for sequence encoder
            environmental_config: Configuration for environmental normalizer
            temporal_config: Configuration for temporal structurer
            cache_dir: Directory for caching intermediate results
        """
        # Default configurations
        self.sequence_config = sequence_config or {
            'encoding_method': 'one_hot',
            'kmer_size': 3,
            'window_size': 100,
            'overlap': 50
        }
        
        self.environmental_config = environmental_config or {
            'continuous_method': 'standard',
            'categorical_method': 'one_hot',
            'imputation_strategy': 'median',
            'seasonal_features': True
        }
        
        self.temporal_config = temporal_config or {
            'time_window_days': 30,
            'prediction_horizon_days': 90,
            'min_sequences_per_window': 5,
            'overlap_ratio': 0.5
        }
        
        # Initialize components
        # Only pass kwargs that SequenceEncoder.__init__ accepts
        _seq_valid_keys = {'encoding_method', 'kmer_size', 'window_size', 'overlap', 'vocab_size'}
        _seq_kwargs = {k: v for k, v in self.sequence_config.items() if k in _seq_valid_keys}
        self.sequence_encoder = SequenceEncoder(**_seq_kwargs)
        _env_valid_keys = {'continuous_method', 'categorical_method', 'imputation_strategy', 'seasonal_features'}
        _env_kwargs = {k: v for k, v in self.environmental_config.items() if k in _env_valid_keys}
        self.environmental_normalizer = EnvironmentalNormalizer(**_env_kwargs)
        _tmp_valid_keys = {'time_window_days', 'prediction_horizon_days', 'min_sequences_per_window', 'overlap_ratio'}
        _tmp_kwargs = {k: v for k, v in self.temporal_config.items() if k in _tmp_valid_keys}
        self.temporal_structurer = TemporalStructurer(**_tmp_kwargs)
        
        # Cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline state
        self.fitted = False
        self.feature_metadata = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _cache_data(self, data: Union[np.ndarray, pd.DataFrame], 
                   cache_key: str) -> None:
        """Cache intermediate data to disk."""
        if self.cache_dir is None:
            return
        
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            import pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Cached data to {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cache data: {e}")
    
    def _load_cached_data(self, cache_key: str) -> Optional[Union[np.ndarray, pd.DataFrame]]:
        """Load cached data from disk."""
        if self.cache_dir is None:
            return None
        
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_path.exists():
            return None
        
        try:
            import pickle
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self.logger.info(f"Loaded cached data from {cache_path}")
            return data
        except Exception as e:
            self.logger.warning(f"Failed to load cached data: {e}")
            return None
    
    def _validate_input_data(self,
                           sequence_data: pd.DataFrame,
                           environmental_data: pd.DataFrame) -> None:
        """Validate input data format and content."""
        # Check required columns in sequence data
        required_seq_cols = ['sequence_id', 'sequence', 'date', 'location']
        missing_seq_cols = [col for col in required_seq_cols 
                           if col not in sequence_data.columns]
        if missing_seq_cols:
            raise ValueError(f"Missing columns in sequence_data: {missing_seq_cols}")
        
        # Check required columns in environmental data
        required_env_cols = ['date', 'location']
        missing_env_cols = [col for col in required_env_cols 
                           if col not in environmental_data.columns]
        if missing_env_cols:
            raise ValueError(f"Missing columns in environmental_data: {missing_env_cols}")
        
        # Check for at least one environmental variable
        env_vars = [col for col in environmental_data.columns 
                   if col not in ['date', 'location']]
        if not env_vars:
            raise ValueError("No environmental variables found in environmental_data")
        
        # Validate data types
        sequence_data['date'] = pd.to_datetime(sequence_data['date'])
        environmental_data['date'] = pd.to_datetime(environmental_data['date'])
        
        # Check for overlapping date ranges
        seq_date_range = (sequence_data['date'].min(), sequence_data['date'].max())
        env_date_range = (environmental_data['date'].min(), environmental_data['date'].max())
        
        if seq_date_range[1] < env_date_range[0] or env_date_range[1] < seq_date_range[0]:
            raise ValueError("No temporal overlap between sequence and environmental data")
        
        self.logger.info(f"Validated input data: {len(sequence_data)} sequences, "
                        f"{len(environmental_data)} environmental records")
    
    def _merge_sequence_environmental_data(self,
                                         sequence_data: pd.DataFrame,
                                         environmental_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge sequence and environmental data by date and location.
        
        Args:
            sequence_data: DataFrame with sequence information
            environmental_data: DataFrame with environmental variables
            
        Returns:
            Merged DataFrame
        """
        # Create date bins for merging (daily resolution)
        sequence_data = sequence_data.copy()
        environmental_data = environmental_data.copy()
        
        # Extract date components for merging
        sequence_data['merge_date'] = sequence_data['date'].dt.date
        environmental_data['merge_date'] = environmental_data['date'].dt.date
        
        # Merge on date and location
        merged_data = pd.merge(
            sequence_data,
            environmental_data,
            on=['merge_date', 'location'],
            how='inner',
            suffixes=('_seq', '_env')
        )
        
        # Use sequence date as primary date
        merged_data['date'] = merged_data['date_seq']
        merged_data = merged_data.drop(columns=['date_seq', 'date_env', 'merge_date'])
        
        self.logger.info(f"Merged data: {len(merged_data)} records after joining")
        
        return merged_data
    
    def _encode_sequences(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """
        Encode viral sequences using the sequence encoder.
        
        Args:
            sequences: List of viral genome sequences
            
        Returns:
            Dictionary with encoded sequences and metadata
        """
        cache_key = f"encoded_sequences_{hash(tuple(sequences))}"
        cached_result = self._load_cached_data(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        self.logger.info(f"Encoding {len(sequences)} sequences...")
        
        # Fit encoder if not already fitted
        if not self.sequence_encoder.fitted:
            self.sequence_encoder.fit(sequences)
        
        # Encode sequences in batches
        batch_size = 100
        encoded_results = {
            'encoded_sequences': [],
            'sequence_features': [],
            'sequence_indices': [],
            'num_windows': []
        }
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch_result = self.sequence_encoder.encode_batch(batch)
            
            for key in encoded_results:
                if key in batch_result and batch_result[key]:
                    if isinstance(batch_result[key], list):
                        encoded_results[key].extend(batch_result[key])
                    else:
                        encoded_results[key].append(batch_result[key])
        
        # Convert lists to arrays
        for key in encoded_results:
            if encoded_results[key] and isinstance(encoded_results[key][0], np.ndarray):
                encoded_results[key] = np.array(encoded_results[key])
        
        # Cache results
        self._cache_data(encoded_results, cache_key)
        
        self.logger.info("Sequence encoding completed")
        return encoded_results
    
    def _normalize_environmental_data(self,
                                    env_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize environmental data using the environmental normalizer.
        
        Args:
            env_data: DataFrame with environmental variables
            
        Returns:
            Normalized environmental DataFrame
        """
        cache_key = f"normalized_env_{hash(tuple(env_data.columns))}"
        cached_result = self._load_cached_data(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        self.logger.info("Normalizing environmental data...")
        
        # Separate date and location columns
        feature_cols = [col for col in env_data.columns 
                       if col not in ['date', 'location', 'sequence_id', 'sequence']]
        
        env_features = env_data[feature_cols]
        
        # Fit and transform environmental data
        normalized_env = self.environmental_normalizer.fit_transform(env_features)
        
        # Add back date and location
        normalized_env['date'] = env_data['date'].values
        normalized_env['location'] = env_data['location'].values
        
        # Cache results
        self._cache_data(normalized_env, cache_key)
        
        self.logger.info("Environmental normalization completed")
        return normalized_env
    
    def _create_temporal_structure(self,
                                 sequence_data: pd.DataFrame,
                                 env_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create temporal structure for sequence-to-sequence learning.
        
        Args:
            sequence_data: DataFrame with encoded sequences and dates
            env_data: DataFrame with normalized environmental data
            
        Returns:
            Tuple of (input_sequences, target_sequences, environmental_features)
        """
        cache_key = "temporal_structure"
        cached_result = self._load_cached_data(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        self.logger.info("Creating temporal structure...")
        
        # Fit temporal structurer
        self.temporal_structurer.fit(sequence_data, env_data)
        
        # Get training data
        input_sequences, target_sequences, env_features = (
            self.temporal_structurer.get_training_data()
        )
        
        result = (input_sequences, target_sequences, env_features)
        
        # Cache results
        self._cache_data(result, cache_key)
        
        self.logger.info(f"Temporal structure created: {len(input_sequences)} sequence pairs")
        return result
    
    def fit_transform(self,
                     sequence_data: pd.DataFrame,
                     environmental_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Fit the pipeline and transform the data.
        
        Args:
            sequence_data: DataFrame with viral sequences and metadata
            environmental_data: DataFrame with environmental variables
            
        Returns:
            Dictionary with transformed data ready for ML
        """
        self.logger.info("Starting feature pipeline fit_transform...")
        
        # Validate input data
        self._validate_input_data(sequence_data, environmental_data)
        
        # Merge sequence and environmental data
        merged_data = self._merge_sequence_environmental_data(
            sequence_data, environmental_data
        )
        
        # Encode sequences
        encoded_sequences = self._encode_sequences(merged_data['sequence'].tolist())
        
        # Add encoded sequences to merged data
        merged_data['encoded_sequence'] = list(encoded_sequences['encoded_sequences'])
        
        # Normalize environmental data
        normalized_env = self._normalize_environmental_data(merged_data)
        
        # Create temporal structure
        input_sequences, target_sequences, env_features = self._create_temporal_structure(
            merged_data[['date', 'encoded_sequence']], normalized_env
        )
        
        # Store feature metadata
        self.feature_metadata = {
            'sequence_encoder_features': self.sequence_encoder.get_feature_names(),
            'environmental_features': self.environmental_normalizer.get_feature_names(),
            'temporal_features': self.temporal_structurer.get_feature_names(),
            'input_sequence_shape': input_sequences.shape,
            'target_sequence_shape': target_sequences.shape,
            'environmental_features_shape': env_features.shape,
            'num_temporal_windows': len(input_sequences),
            'window_metadata': self.temporal_structurer.window_metadata
        }
        
        self.fitted = True
        
        # Prepare final output
        result = {
            'input_sequences': input_sequences,
            'target_sequences': target_sequences,
            'environmental_features': env_features,
            'sequence_features': encoded_sequences.get('sequence_features', None),
            'metadata': self.feature_metadata
        }
        
        self.logger.info("Feature pipeline completed successfully")
        return result
    
    def transform(self,
                 sequence_data: pd.DataFrame,
                 environmental_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Transform new data using fitted pipeline.
        
        Args:
            sequence_data: DataFrame with viral sequences and metadata
            environmental_data: DataFrame with environmental variables
            
        Returns:
            Dictionary with transformed data
        """
        if not self.fitted:
            raise ValueError("Pipeline not fitted. Call fit_transform() first.")
        
        self.logger.info("Transforming new data...")
        
        # Validate and merge data
        self._validate_input_data(sequence_data, environmental_data)
        merged_data = self._merge_sequence_environmental_data(
            sequence_data, environmental_data
        )
        
        # Encode sequences
        encoded_sequences = self._encode_sequences(merged_data['sequence'].tolist())
        merged_data['encoded_sequence'] = list(encoded_sequences['encoded_sequences'])
        
        # Normalize environmental data
        normalized_env = self.environmental_normalizer.transform(
            merged_data[[col for col in merged_data.columns 
                        if col not in ['date', 'location', 'sequence_id', 'sequence', 'encoded_sequence']]]
        )
        normalized_env['date'] = merged_data['date'].values
        normalized_env['location'] = merged_data['location'].values
        
        # Create prediction windows
        current_date = merged_data['date'].max()
        prediction_windows = self.temporal_structurer.create_prediction_windows(
            current_date, merged_data[['date', 'encoded_sequence']], normalized_env
        )
        
        if not prediction_windows:
            raise ValueError("No valid prediction windows could be created")
        
        # Extract features for prediction
        input_sequences = [window['input_sequence'] for window in prediction_windows]
        env_features = [window['input_env'] for window in prediction_windows]
        
        result = {
            'input_sequences': np.array(input_sequences),
            'environmental_features': pd.DataFrame(env_features).fillna(0).values,
            'prediction_windows': prediction_windows,
            'metadata': self.feature_metadata
        }
        
        self.logger.info("Data transformation completed")
        return result
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """Get suggested importance weights for all features."""
        if not self.fitted:
            return {}
        
        weights = {}
        
        # Environmental feature weights
        env_weights = self.environmental_normalizer.get_feature_importance_weights()
        weights.update(env_weights)
        
        # Sequence feature weights (uniform for now)
        seq_features = self.sequence_encoder.get_feature_names()
        for feature in seq_features:
            weights[f'sequence_{feature}'] = 1.0
        
        # Temporal feature weights
        temporal_features = self.temporal_structurer.get_feature_names()
        for feature in temporal_features:
            if 'diversity' in feature.lower():
                weights[feature] = 0.8
            elif 'time_delta' in feature.lower():
                weights[feature] = 0.6
            else:
                weights[feature] = 0.7
        
        return weights
    
    def save_pipeline(self, filepath: str) -> None:
        """Save the complete fitted pipeline to disk."""
        import pickle
        
        pipeline_data = {
            'sequence_config': self.sequence_config,
            'environmental_config': self.environmental_config,
            'temporal_config': self.temporal_config,
            'sequence_encoder': self.sequence_encoder,
            'environmental_normalizer': self.environmental_normalizer,
            'temporal_structurer': self.temporal_structurer,
            'feature_metadata': self.feature_metadata,
            'fitted': self.fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        self.logger.info(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath: str) -> 'FeaturePipeline':
        """Load a fitted pipeline from disk."""
        import pickle
        
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        pipeline = cls(
            sequence_config=pipeline_data['sequence_config'],
            environmental_config=pipeline_data['environmental_config'],
            temporal_config=pipeline_data['temporal_config']
        )
        
        # Restore fitted components
        pipeline.sequence_encoder = pipeline_data['sequence_encoder']
        pipeline.environmental_normalizer = pipeline_data['environmental_normalizer']
        pipeline.temporal_structurer = pipeline_data['temporal_structurer']
        pipeline.feature_metadata = pipeline_data['feature_metadata']
        pipeline.fitted = pipeline_data['fitted']
        
        return pipeline
    
    def get_summary(self) -> Dict:
        """Get a summary of the pipeline configuration and results."""
        summary = {
            'pipeline_config': {
                'sequence_encoding': self.sequence_config,
                'environmental_normalization': self.environmental_config,
                'temporal_structuring': self.temporal_config
            },
            'fitted': self.fitted
        }
        
        if self.fitted:
            summary['feature_metadata'] = self.feature_metadata
            summary['component_status'] = {
                'sequence_encoder_fitted': self.sequence_encoder.fitted,
                'environmental_normalizer_fitted': self.environmental_normalizer.fitted,
                'temporal_structurer_fitted': self.temporal_structurer.fitted
            }
        
        return summary