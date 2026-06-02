"""
Temporal Data Structurer.

This module structures viral sequence and environmental data temporally
to enable learning of sequence transitions over time given environmental conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
from collections import defaultdict


class TemporalStructurer:
    """
    Structures data temporally for sequence-to-sequence learning.
    
    Creates time-ordered datasets that capture viral evolution patterns
    under varying environmental conditions.
    """
    
    def __init__(self,
                 time_window_days: int = 30,
                 prediction_horizon_days: int = 90,
                 min_sequences_per_window: int = 5,
                 overlap_ratio: float = 0.5,
                 aggregation_method: str = "mean"):
        """
        Initialize the temporal structurer.
        
        Args:
            time_window_days: Size of temporal windows in days
            prediction_horizon_days: How far ahead to predict
            min_sequences_per_window: Minimum sequences required per window
            overlap_ratio: Overlap between consecutive windows (0-1)
            aggregation_method: Method for aggregating multiple sequences
                               in same window ('mean', 'median', 'concat')
        """
        self.time_window_days = time_window_days
        self.prediction_horizon_days = prediction_horizon_days
        self.min_sequences_per_window = min_sequences_per_window
        self.overlap_ratio = overlap_ratio
        self.aggregation_method = aggregation_method
        
        # Computed parameters
        self.step_size_days = int(time_window_days * (1 - overlap_ratio))
        
        # Storage for structured data
        self.temporal_windows = []
        self.window_metadata = []
        self.fitted = False
    
    def _create_time_windows(self, 
                           dates: pd.Series) -> List[Tuple[datetime, datetime]]:
        """
        Create overlapping time windows from date range.
        
        Args:
            dates: Series of datetime objects
            
        Returns:
            List of (start_date, end_date) tuples
        """
        min_date = dates.min()
        max_date = dates.max()
        
        windows = []
        current_start = min_date
        
        while current_start + timedelta(days=self.time_window_days) <= max_date:
            window_end = current_start + timedelta(days=self.time_window_days)
            windows.append((current_start, window_end))
            current_start += timedelta(days=self.step_size_days)
        
        return windows
    
    def _aggregate_sequences_in_window(self,
                                     sequences: List[np.ndarray],
                                     method: str = "mean") -> np.ndarray:
        """
        Aggregate multiple sequences within a time window.
        
        Args:
            sequences: List of encoded sequence arrays
            method: Aggregation method ('mean', 'median', 'concat')
            
        Returns:
            Aggregated sequence representation
        """
        if len(sequences) == 0:
            return None
        
        if len(sequences) == 1:
            return sequences[0]
        
        if method == "mean":
            # Average across sequences
            return np.mean(sequences, axis=0)
        
        elif method == "median":
            # Median across sequences
            return np.median(sequences, axis=0)
        
        elif method == "concat":
            # Concatenate sequences
            return np.concatenate(sequences, axis=0)
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def _calculate_window_statistics(self,
                                   sequences: List[np.ndarray],
                                   env_data: pd.DataFrame) -> Dict:
        """
        Calculate statistics for a temporal window.
        
        Args:
            sequences: List of sequences in the window
            env_data: Environmental data for the window
            
        Returns:
            Dictionary of window statistics
        """
        stats = {
            'num_sequences': len(sequences),
            'sequence_diversity': 0.0,
            'env_mean': {},
            'env_std': {},
            'env_range': {}
        }
        
        # Calculate sequence diversity (if multiple sequences)
        if len(sequences) > 1:
            # Calculate pairwise distances between sequences
            distances = []
            for i in range(len(sequences)):
                for j in range(i + 1, len(sequences)):
                    # Simple Hamming distance for encoded sequences
                    if sequences[i].shape == sequences[j].shape:
                        dist = np.mean(sequences[i] != sequences[j])
                        distances.append(dist)
            
            if distances:
                stats['sequence_diversity'] = np.mean(distances)
        
        # Calculate environmental statistics
        numeric_cols = env_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if not env_data[col].empty:
                stats['env_mean'][col] = env_data[col].mean()
                stats['env_std'][col] = env_data[col].std()
                stats['env_range'][col] = env_data[col].max() - env_data[col].min()
        
        return stats
    
    def _create_sequence_pairs(self,
                             windows: List[Tuple[datetime, datetime]],
                             sequence_data: pd.DataFrame,
                             env_data: pd.DataFrame) -> List[Dict]:
        """
        Create input-target sequence pairs for training.
        
        Args:
            windows: List of time windows
            sequence_data: DataFrame with sequences and dates
            env_data: DataFrame with environmental data and dates
            
        Returns:
            List of sequence pair dictionaries
        """
        sequence_pairs = []
        
        for i, (start_date, end_date) in enumerate(windows):
            # Find sequences in current window (input)
            input_mask = ((sequence_data['date'] >= start_date) & 
                         (sequence_data['date'] < end_date))
            input_sequences = sequence_data[input_mask]
            
            if len(input_sequences) < self.min_sequences_per_window:
                continue
            
            # Find target window
            target_start = end_date
            target_end = target_start + timedelta(days=self.prediction_horizon_days)
            
            # Check if target window exists in data
            target_mask = ((sequence_data['date'] >= target_start) & 
                          (sequence_data['date'] < target_end))
            target_sequences = sequence_data[target_mask]
            
            if len(target_sequences) < self.min_sequences_per_window:
                continue
            
            # Get environmental data for both windows
            input_env_mask = ((env_data['date'] >= start_date) & 
                             (env_data['date'] < end_date))
            target_env_mask = ((env_data['date'] >= target_start) & 
                              (env_data['date'] < target_end))
            
            input_env = env_data[input_env_mask]
            target_env = env_data[target_env_mask]
            
            if input_env.empty or target_env.empty:
                continue
            
            # Aggregate sequences in each window
            input_seq_list = input_sequences['encoded_sequence'].tolist()
            target_seq_list = target_sequences['encoded_sequence'].tolist()
            
            aggregated_input = self._aggregate_sequences_in_window(
                input_seq_list, self.aggregation_method
            )
            aggregated_target = self._aggregate_sequences_in_window(
                target_seq_list, self.aggregation_method
            )
            
            if aggregated_input is None or aggregated_target is None:
                continue
            
            # Calculate window statistics
            input_stats = self._calculate_window_statistics(input_seq_list, input_env)
            target_stats = self._calculate_window_statistics(target_seq_list, target_env)
            
            # Create sequence pair
            pair = {
                'window_id': i,
                'input_sequence': aggregated_input,
                'target_sequence': aggregated_target,
                'input_env': input_env.drop(columns=['date']).mean().to_dict(),
                'target_env': target_env.drop(columns=['date']).mean().to_dict(),
                'input_date_range': (start_date, end_date),
                'target_date_range': (target_start, target_end),
                'input_stats': input_stats,
                'target_stats': target_stats,
                'time_delta_days': self.prediction_horizon_days
            }
            
            sequence_pairs.append(pair)
        
        return sequence_pairs
    
    def _create_lag_features(self,
                           env_data: pd.DataFrame,
                           lag_days: List[int]) -> pd.DataFrame:
        """
        Create lagged environmental features.
        
        Args:
            env_data: Environmental data with date column
            lag_days: List of lag periods in days
            
        Returns:
            DataFrame with lagged features
        """
        env_with_lags = env_data.copy()
        
        # Sort by date
        env_with_lags = env_with_lags.sort_values('date')
        
        # Create lagged features
        numeric_cols = env_with_lags.select_dtypes(include=[np.number]).columns
        
        for lag in lag_days:
            for col in numeric_cols:
                lag_col_name = f'{col}_lag_{lag}d'
                env_with_lags[lag_col_name] = env_with_lags[col].shift(lag)
        
        return env_with_lags
    
    def _calculate_environmental_trends(self,
                                      env_data: pd.DataFrame,
                                      window_size: int = 7) -> pd.DataFrame:
        """
        Calculate environmental trends and derivatives.
        
        Args:
            env_data: Environmental data with date column
            window_size: Window size for trend calculation
            
        Returns:
            DataFrame with trend features
        """
        env_with_trends = env_data.copy()
        
        # Sort by date
        env_with_trends = env_with_trends.sort_values('date')
        
        numeric_cols = env_with_trends.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Rolling mean (smoothed trend)
            trend_col = f'{col}_trend_{window_size}d'
            env_with_trends[trend_col] = (
                env_with_trends[col].rolling(window=window_size, center=True).mean()
            )
            
            # First derivative (rate of change)
            derivative_col = f'{col}_derivative'
            env_with_trends[derivative_col] = env_with_trends[col].diff()
            
            # Second derivative (acceleration)
            accel_col = f'{col}_acceleration'
            env_with_trends[accel_col] = env_with_trends[derivative_col].diff()
        
        return env_with_trends
    
    def fit(self,
            sequence_data: pd.DataFrame,
            environmental_data: pd.DataFrame,
            lag_days: Optional[List[int]] = None) -> 'TemporalStructurer':
        """
        Fit the temporal structurer on sequence and environmental data.
        
        Args:
            sequence_data: DataFrame with columns ['date', 'encoded_sequence', ...]
            environmental_data: DataFrame with environmental variables and 'date'
            lag_days: Optional list of lag periods for environmental features
            
        Returns:
            Self for method chaining
        """
        # Validate input data
        required_seq_cols = ['date', 'encoded_sequence']
        for col in required_seq_cols:
            if col not in sequence_data.columns:
                raise ValueError(f"Missing required column in sequence_data: {col}")
        
        if 'date' not in environmental_data.columns:
            raise ValueError("Missing 'date' column in environmental_data")
        
        # Convert dates to datetime
        sequence_data = sequence_data.copy()
        environmental_data = environmental_data.copy()
        
        sequence_data['date'] = pd.to_datetime(sequence_data['date'])
        environmental_data['date'] = pd.to_datetime(environmental_data['date'])
        
        # Create lag features if specified
        if lag_days:
            environmental_data = self._create_lag_features(environmental_data, lag_days)
        
        # Calculate environmental trends
        environmental_data = self._calculate_environmental_trends(environmental_data)
        
        # Create time windows
        all_dates = pd.concat([sequence_data['date'], environmental_data['date']])
        windows = self._create_time_windows(all_dates)
        
        # Create sequence pairs
        self.temporal_windows = self._create_sequence_pairs(
            windows, sequence_data, environmental_data
        )
        
        # Store metadata
        self.window_metadata = {
            'total_windows': len(windows),
            'valid_pairs': len(self.temporal_windows),
            'date_range': (all_dates.min(), all_dates.max()),
            'avg_sequences_per_window': np.mean([
                pair['input_stats']['num_sequences'] 
                for pair in self.temporal_windows
            ]) if self.temporal_windows else 0
        }
        
        self.fitted = True
        return self
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get structured training data for model training.
        
        Returns:
            Tuple of (input_sequences, target_sequences, environmental_features)
        """
        if not self.fitted:
            raise ValueError("TemporalStructurer not fitted. Call fit() first.")
        
        if not self.temporal_windows:
            raise ValueError("No valid temporal windows found.")
        
        # Extract sequences and environmental features
        input_sequences = []
        target_sequences = []
        env_features = []
        
        for pair in self.temporal_windows:
            input_sequences.append(pair['input_sequence'])
            target_sequences.append(pair['target_sequence'])
            
            # Combine input and target environmental features
            combined_env = {}
            
            # Add input environmental features with prefix
            for key, value in pair['input_env'].items():
                combined_env[f'input_{key}'] = value
            
            # Add target environmental features with prefix
            for key, value in pair['target_env'].items():
                combined_env[f'target_{key}'] = value
            
            # Add temporal features
            combined_env['time_delta_days'] = pair['time_delta_days']
            combined_env['input_sequence_diversity'] = pair['input_stats']['sequence_diversity']
            combined_env['target_sequence_diversity'] = pair['target_stats']['sequence_diversity']
            
            env_features.append(combined_env)
        
        # Convert to numpy arrays
        input_sequences = np.array(input_sequences)
        target_sequences = np.array(target_sequences)
        
        # Convert environmental features to structured array
        env_df = pd.DataFrame(env_features)
        env_features = env_df.fillna(0).values
        
        return input_sequences, target_sequences, env_features
    
    def get_feature_names(self) -> List[str]:
        """Get names of environmental features."""
        if not self.fitted or not self.temporal_windows:
            return []
        
        # Get feature names from first window
        sample_pair = self.temporal_windows[0]
        feature_names = []
        
        # Input environmental features
        for key in sample_pair['input_env'].keys():
            feature_names.append(f'input_{key}')
        
        # Target environmental features
        for key in sample_pair['target_env'].keys():
            feature_names.append(f'target_{key}')
        
        # Additional features
        feature_names.extend([
            'time_delta_days',
            'input_sequence_diversity',
            'target_sequence_diversity'
        ])
        
        return feature_names
    
    def get_window_info(self) -> pd.DataFrame:
        """
        Get information about all temporal windows.
        
        Returns:
            DataFrame with window metadata
        """
        if not self.fitted:
            return pd.DataFrame()
        
        window_info = []
        
        for pair in self.temporal_windows:
            info = {
                'window_id': pair['window_id'],
                'input_start': pair['input_date_range'][0],
                'input_end': pair['input_date_range'][1],
                'target_start': pair['target_date_range'][0],
                'target_end': pair['target_date_range'][1],
                'input_num_sequences': pair['input_stats']['num_sequences'],
                'target_num_sequences': pair['target_stats']['num_sequences'],
                'input_diversity': pair['input_stats']['sequence_diversity'],
                'target_diversity': pair['target_stats']['sequence_diversity']
            }
            
            # Add environmental means
            for key, value in pair['input_env'].items():
                info[f'input_{key}_mean'] = value
            
            for key, value in pair['target_env'].items():
                info[f'target_{key}_mean'] = value
            
            window_info.append(info)
        
        return pd.DataFrame(window_info)
    
    def create_prediction_windows(self,
                                current_date: datetime,
                                sequence_data: pd.DataFrame,
                                environmental_data: pd.DataFrame) -> List[Dict]:
        """
        Create windows for making future predictions.
        
        Args:
            current_date: Current date for prediction
            sequence_data: Recent sequence data
            environmental_data: Recent environmental data
            
        Returns:
            List of prediction windows
        """
        # Create input window ending at current_date
        input_start = current_date - timedelta(days=self.time_window_days)
        input_end = current_date
        
        # Filter data for input window
        input_seq_mask = ((sequence_data['date'] >= input_start) & 
                         (sequence_data['date'] < input_end))
        input_sequences = sequence_data[input_seq_mask]
        
        input_env_mask = ((environmental_data['date'] >= input_start) & 
                         (environmental_data['date'] < input_end))
        input_env = environmental_data[input_env_mask]
        
        if len(input_sequences) < self.min_sequences_per_window or input_env.empty:
            return []
        
        # Aggregate input sequences
        input_seq_list = input_sequences['encoded_sequence'].tolist()
        aggregated_input = self._aggregate_sequences_in_window(
            input_seq_list, self.aggregation_method
        )
        
        # Calculate input statistics
        input_stats = self._calculate_window_statistics(input_seq_list, input_env)
        
        # Create prediction window
        prediction_window = {
            'input_sequence': aggregated_input,
            'input_env': input_env.drop(columns=['date']).mean().to_dict(),
            'input_date_range': (input_start, input_end),
            'input_stats': input_stats,
            'prediction_date': current_date,
            'prediction_horizon_days': self.prediction_horizon_days
        }
        
        return [prediction_window]
    
    def save_structurer(self, filepath: str) -> None:
        """Save the fitted structurer to disk."""
        import pickle
        
        structurer_data = {
            'time_window_days': self.time_window_days,
            'prediction_horizon_days': self.prediction_horizon_days,
            'min_sequences_per_window': self.min_sequences_per_window,
            'overlap_ratio': self.overlap_ratio,
            'aggregation_method': self.aggregation_method,
            'step_size_days': self.step_size_days,
            'temporal_windows': self.temporal_windows,
            'window_metadata': self.window_metadata,
            'fitted': self.fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(structurer_data, f)
    
    @classmethod
    def load_structurer(cls, filepath: str) -> 'TemporalStructurer':
        """Load a fitted structurer from disk."""
        import pickle
        
        with open(filepath, 'rb') as f:
            structurer_data = pickle.load(f)
        
        structurer = cls(
            time_window_days=structurer_data['time_window_days'],
            prediction_horizon_days=structurer_data['prediction_horizon_days'],
            min_sequences_per_window=structurer_data['min_sequences_per_window'],
            overlap_ratio=structurer_data['overlap_ratio'],
            aggregation_method=structurer_data['aggregation_method']
        )
        
        # Restore fitted data
        structurer.step_size_days = structurer_data['step_size_days']
        structurer.temporal_windows = structurer_data['temporal_windows']
        structurer.window_metadata = structurer_data['window_metadata']
        structurer.fitted = structurer_data['fitted']
        
        return structurer