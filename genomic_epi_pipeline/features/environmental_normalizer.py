"""
Environmental Data Normalizer.

This module handles the normalization and encoding of environmental factors
that serve as selective pressures in viral evolution prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, LabelEncoder, TargetEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from datetime import datetime, timedelta
import warnings


class EnvironmentalNormalizer:
    """
    Normalizes and encodes environmental data for viral evolution modeling.
    
    Handles continuous variables (temperature, humidity) and categorical
    variables (host species, geographic location) with appropriate
    normalization and encoding strategies.
    """
    
    def __init__(self,
                 continuous_method: str = "standard",
                 categorical_method: str = "one_hot",
                 imputation_strategy: str = "median",
                 seasonal_features: bool = True,
                 geographic_encoding: str = "one_hot"):
        """
        Initialize the environmental normalizer.
        
        Args:
            continuous_method: Normalization method for continuous variables
                              ('standard', 'minmax', 'robust')
            categorical_method: Encoding method for categorical variables
                               ('one_hot', 'label', 'target')
            imputation_strategy: Strategy for missing value imputation
                                ('mean', 'median', 'mode', 'knn')
            seasonal_features: Whether to extract seasonal features from dates
            geographic_encoding: Method for encoding geographic information
        """
        self.continuous_method = continuous_method
        self.categorical_method = categorical_method
        self.imputation_strategy = imputation_strategy
        self.seasonal_features = seasonal_features
        self.geographic_encoding = geographic_encoding
        
        # Initialize scalers and encoders
        self.continuous_scaler = self._get_continuous_scaler()
        self.categorical_encoder = self._get_categorical_encoder()
        self.imputer = self._get_imputer()
        
        # Feature names and fitted status
        self.continuous_features = []
        self.categorical_features = []
        self.feature_names = []
        self.fitted = False
        
        # Store fit parameters
        self.fit_params = {}
    
    def _get_continuous_scaler(self):
        """Get the appropriate scaler for continuous variables."""
        if self.continuous_method == "standard":
            return StandardScaler()
        elif self.continuous_method == "minmax":
            return MinMaxScaler()
        elif self.continuous_method == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown continuous method: {self.continuous_method}")
    
    def _get_categorical_encoder(self):
        """Get the appropriate encoder for categorical variables."""
        if self.categorical_method == "one_hot":
            return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        elif self.categorical_method == "label":
            return LabelEncoder()
        elif self.categorical_method == "target":
            return TargetEncoder()
        else:
            raise ValueError(f"Unknown categorical method: {self.categorical_method}")
    
    def _get_imputer(self):
        """Get the appropriate imputer for missing values."""
        if self.imputation_strategy in ["mean", "median"]:
            return SimpleImputer(strategy=self.imputation_strategy)
        elif self.imputation_strategy == "mode":
            return SimpleImputer(strategy="most_frequent")
        elif self.imputation_strategy == "knn":
            return KNNImputer(n_neighbors=5)
        else:
            raise ValueError(f"Unknown imputation strategy: {self.imputation_strategy}")
    
    def _extract_temporal_features(self, dates: pd.Series) -> pd.DataFrame:
        """
        Extract temporal features from date information.
        
        Args:
            dates: Series of datetime objects
            
        Returns:
            DataFrame with temporal features
        """
        temporal_features = pd.DataFrame(index=dates.index)
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(dates):
            dates = pd.to_datetime(dates)
        
        # Basic temporal features
        temporal_features['year'] = dates.dt.year
        temporal_features['month'] = dates.dt.month
        temporal_features['day_of_year'] = dates.dt.dayofyear
        temporal_features['week_of_year'] = dates.dt.isocalendar().week
        temporal_features['day_of_week'] = dates.dt.dayofweek
        
        if self.seasonal_features:
            # Seasonal features using sine/cosine encoding
            temporal_features['month_sin'] = np.sin(2 * np.pi * dates.dt.month / 12)
            temporal_features['month_cos'] = np.cos(2 * np.pi * dates.dt.month / 12)
            temporal_features['day_sin'] = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
            temporal_features['day_cos'] = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25)
            
            # Season categories
            seasons = []
            for month in dates.dt.month:
                if month in [12, 1, 2]:
                    seasons.append('winter')
                elif month in [3, 4, 5]:
                    seasons.append('spring')
                elif month in [6, 7, 8]:
                    seasons.append('summer')
                else:
                    seasons.append('autumn')
            
            temporal_features['season'] = seasons
        
        return temporal_features
    
    def _encode_geographic_features(self, locations: pd.Series) -> pd.DataFrame:
        """
        Encode geographic location information.
        
        Args:
            locations: Series of location strings
            
        Returns:
            DataFrame with encoded geographic features
        """
        geo_features = pd.DataFrame(index=locations.index)
        
        if self.geographic_encoding == "one_hot":
            # One-hot encode locations
            unique_locations = locations.unique()
            for location in unique_locations:
                if pd.notna(location):
                    geo_features[f'location_{location}'] = (locations == location).astype(int)
        
        elif self.geographic_encoding == "label":
            # Label encode locations
            geo_features['location_encoded'] = LabelEncoder().fit_transform(
                locations.fillna('unknown')
            )
        
        return geo_features
    
    def _calculate_climate_indices(self, 
                                  temperature: pd.Series,
                                  humidity: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Calculate derived climate indices.
        
        Args:
            temperature: Temperature values in Celsius
            humidity: Optional humidity values in percentage
            
        Returns:
            DataFrame with climate indices
        """
        climate_features = pd.DataFrame(index=temperature.index)
        
        # Temperature-based features
        climate_features['temp_celsius'] = temperature
        climate_features['temp_kelvin'] = temperature + 273.15
        
        # Temperature categories
        temp_categories = []
        for temp in temperature:
            if pd.isna(temp):
                temp_categories.append('unknown')
            elif temp < 0:
                temp_categories.append('freezing')
            elif temp < 10:
                temp_categories.append('cold')
            elif temp < 25:
                temp_categories.append('moderate')
            elif temp < 35:
                temp_categories.append('warm')
            else:
                temp_categories.append('hot')
        
        climate_features['temp_category'] = temp_categories
        
        if humidity is not None:
            climate_features['humidity'] = humidity
            
            # Calculate heat index (apparent temperature)
            # Simplified heat index calculation
            heat_index = []
            for temp, hum in zip(temperature, humidity):
                if pd.isna(temp) or pd.isna(hum):
                    heat_index.append(np.nan)
                elif temp >= 27:  # Heat index only meaningful at high temps
                    hi = (temp * 1.8 + 32)  # Convert to Fahrenheit for calculation
                    hi = (-42.379 + 2.04901523 * hi + 10.14333127 * hum
                          - 0.22475541 * hi * hum - 6.83783e-3 * hi**2
                          - 5.481717e-2 * hum**2 + 1.22874e-3 * hi**2 * hum
                          + 8.5282e-4 * hi * hum**2 - 1.99e-6 * hi**2 * hum**2)
                    heat_index.append((hi - 32) / 1.8)  # Convert back to Celsius
                else:
                    heat_index.append(temp)
            
            climate_features['heat_index'] = heat_index
        
        return climate_features
    
    def fit(self, 
            data: pd.DataFrame,
            target: Optional[pd.Series] = None) -> 'EnvironmentalNormalizer':
        """
        Fit the normalizer on environmental data.
        
        Args:
            data: DataFrame with environmental variables
            target: Optional target variable for target encoding
            
        Returns:
            Self for method chaining
        """
        # Identify feature types
        self.continuous_features = []
        self.categorical_features = []
        
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                if col not in ['year', 'month', 'day', 'week']:  # Exclude temporal integers
                    self.continuous_features.append(col)
            else:
                self.categorical_features.append(col)
        
        # Store original data info
        self.fit_params['data_shape'] = data.shape
        self.fit_params['feature_types'] = {
            'continuous': self.continuous_features,
            'categorical': self.categorical_features
        }
        
        # Fit imputer on continuous features
        if self.continuous_features:
            continuous_data = data[self.continuous_features]
            self.imputer.fit(continuous_data)
        
        # Fit scaler on continuous features
        if self.continuous_features:
            continuous_data = data[self.continuous_features]
            # Impute first, then scale
            imputed_data = self.imputer.transform(continuous_data)
            self.continuous_scaler.fit(imputed_data)
        
        # Fit categorical encoder
        if self.categorical_features:
            categorical_data = data[self.categorical_features]
            
            if self.categorical_method == "target" and target is not None:
                self.categorical_encoder.fit(categorical_data, target)
            else:
                if self.categorical_method == "one_hot":
                    self.categorical_encoder.fit(categorical_data)
                elif self.categorical_method == "label":
                    # For label encoding, fit each column separately
                    self.categorical_encoder = {}
                    for col in self.categorical_features:
                        encoder = LabelEncoder()
                        encoder.fit(categorical_data[col].fillna('unknown'))
                        self.categorical_encoder[col] = encoder
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform environmental data using fitted normalizers.
        
        Args:
            data: DataFrame with environmental variables
            
        Returns:
            Normalized and encoded DataFrame
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        transformed_data = pd.DataFrame(index=data.index)
        
        # Transform continuous features
        if self.continuous_features:
            continuous_data = data[self.continuous_features]
            
            # Impute missing values
            imputed_data = self.imputer.transform(continuous_data)
            
            # Scale the data
            scaled_data = self.continuous_scaler.transform(imputed_data)
            
            # Add to transformed data
            for i, col in enumerate(self.continuous_features):
                transformed_data[f'{col}_normalized'] = scaled_data[:, i]
        
        # Transform categorical features
        if self.categorical_features:
            categorical_data = data[self.categorical_features]
            
            if self.categorical_method == "one_hot":
                encoded_data = self.categorical_encoder.transform(categorical_data)
                feature_names = self.categorical_encoder.get_feature_names_out(
                    self.categorical_features
                )
                
                for i, name in enumerate(feature_names):
                    transformed_data[name] = encoded_data[:, i]
            
            elif self.categorical_method == "label":
                for col in self.categorical_features:
                    encoder = self.categorical_encoder[col]
                    # Handle unknown categories
                    col_data = categorical_data[col].fillna('unknown')
                    
                    # Transform known categories, assign -1 to unknown
                    encoded_values = []
                    for value in col_data:
                        try:
                            encoded_values.append(encoder.transform([value])[0])
                        except ValueError:
                            encoded_values.append(-1)  # Unknown category
                    
                    transformed_data[f'{col}_encoded'] = encoded_values
            
            elif self.categorical_method == "target":
                encoded_data = self.categorical_encoder.transform(categorical_data)
                for i, col in enumerate(self.categorical_features):
                    transformed_data[f'{col}_target_encoded'] = encoded_data[:, i]
        
        # Extract temporal features if date column exists
        date_columns = [col for col in data.columns 
                       if 'date' in col.lower() or 'time' in col.lower()]
        
        for date_col in date_columns:
            if date_col in data.columns:
                temporal_features = self._extract_temporal_features(data[date_col])
                transformed_data = pd.concat([transformed_data, temporal_features], axis=1)
        
        # Extract geographic features if location column exists
        location_columns = [col for col in data.columns 
                           if any(geo_term in col.lower() 
                                 for geo_term in ['location', 'country', 'region', 'city'])]
        
        for loc_col in location_columns:
            if loc_col in data.columns:
                geo_features = self._encode_geographic_features(data[loc_col])
                transformed_data = pd.concat([transformed_data, geo_features], axis=1)
        
        # Calculate climate indices if temperature is available
        if 'temperature' in data.columns:
            humidity_col = 'humidity' if 'humidity' in data.columns else None
            humidity_data = data[humidity_col] if humidity_col else None
            
            climate_features = self._calculate_climate_indices(
                data['temperature'], humidity_data
            )
            transformed_data = pd.concat([transformed_data, climate_features], axis=1)
        
        # Store feature names
        self.feature_names = transformed_data.columns.tolist()
        
        return transformed_data
    
    def fit_transform(self, 
                     data: pd.DataFrame,
                     target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit the normalizer and transform the data in one step.
        
        Args:
            data: DataFrame with environmental variables
            target: Optional target variable for target encoding
            
        Returns:
            Normalized and encoded DataFrame
        """
        return self.fit(data, target).transform(data)
    
    def inverse_transform_continuous(self, 
                                   data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform continuous features back to original scale.
        
        Args:
            data: DataFrame with normalized continuous features
            
        Returns:
            DataFrame with original scale continuous features
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        inverse_data = data.copy()
        
        # Find normalized continuous features
        normalized_cols = [col for col in data.columns 
                          if col.endswith('_normalized')]
        
        if normalized_cols:
            # Extract the normalized values
            normalized_values = data[normalized_cols].values
            
            # Inverse transform
            original_values = self.continuous_scaler.inverse_transform(normalized_values)
            
            # Replace in dataframe
            for i, col in enumerate(normalized_cols):
                original_col = col.replace('_normalized', '')
                inverse_data[original_col] = original_values[:, i]
                # Remove normalized column
                inverse_data = inverse_data.drop(columns=[col])
        
        return inverse_data
    
    def get_feature_names(self) -> List[str]:
        """Get the names of all transformed features."""
        return self.feature_names if self.fitted else []
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """
        Get suggested importance weights for different feature types.
        
        Returns:
            Dictionary mapping feature names to importance weights
        """
        weights = {}
        
        for feature in self.feature_names:
            if any(term in feature.lower() for term in ['temp', 'climate', 'heat']):
                weights[feature] = 1.0  # High importance for temperature
            elif any(term in feature.lower() for term in ['humidity', 'season']):
                weights[feature] = 0.8  # Medium-high importance
            elif any(term in feature.lower() for term in ['location', 'country']):
                weights[feature] = 0.6  # Medium importance for geography
            elif any(term in feature.lower() for term in ['host', 'species']):
                weights[feature] = 0.9  # High importance for host
            else:
                weights[feature] = 0.5  # Default weight
        
        return weights
    
    def save_normalizer(self, filepath: str) -> None:
        """Save the fitted normalizer to disk."""
        import pickle
        
        normalizer_data = {
            'continuous_method': self.continuous_method,
            'categorical_method': self.categorical_method,
            'imputation_strategy': self.imputation_strategy,
            'seasonal_features': self.seasonal_features,
            'geographic_encoding': self.geographic_encoding,
            'continuous_scaler': self.continuous_scaler,
            'categorical_encoder': self.categorical_encoder,
            'imputer': self.imputer,
            'continuous_features': self.continuous_features,
            'categorical_features': self.categorical_features,
            'feature_names': self.feature_names,
            'fitted': self.fitted,
            'fit_params': self.fit_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(normalizer_data, f)
    
    @classmethod
    def load_normalizer(cls, filepath: str) -> 'EnvironmentalNormalizer':
        """Load a fitted normalizer from disk."""
        import pickle
        
        with open(filepath, 'rb') as f:
            normalizer_data = pickle.load(f)
        
        normalizer = cls(
            continuous_method=normalizer_data['continuous_method'],
            categorical_method=normalizer_data['categorical_method'],
            imputation_strategy=normalizer_data['imputation_strategy'],
            seasonal_features=normalizer_data['seasonal_features'],
            geographic_encoding=normalizer_data['geographic_encoding']
        )
        
        # Restore fitted components
        normalizer.continuous_scaler = normalizer_data['continuous_scaler']
        normalizer.categorical_encoder = normalizer_data['categorical_encoder']
        normalizer.imputer = normalizer_data['imputer']
        normalizer.continuous_features = normalizer_data['continuous_features']
        normalizer.categorical_features = normalizer_data['categorical_features']
        normalizer.feature_names = normalizer_data['feature_names']
        normalizer.fitted = normalizer_data['fitted']
        normalizer.fit_params = normalizer_data['fit_params']
        
        return normalizer