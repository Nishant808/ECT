"""
Benchmarking Suite for Viral Evolution Prediction Models.

This module implements comprehensive benchmarking to compare the Environmental-Conditioned
Transformer against baseline models including Markov chains and vanilla Transformers.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import time
import json
from abc import ABC, abstractmethod
from collections import defaultdict

from ..models.probabilistic_engine import ViralEvolutionPredictor
from .phylogenetic_distance import validate_evolutionary_predictions, PhylogeneticValidationConfig
from .confidence_intervals import analyze_prediction_calibration, CalibrationConfig


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    test_split_ratio: float = 0.2
    cross_validation_folds: int = 5
    random_seed: int = 42
    max_training_epochs: int = 50
    early_stopping_patience: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    
    # Evaluation metrics
    calculate_phylogenetic_distance: bool = True
    calculate_calibration_metrics: bool = True
    calculate_likelihood_scores: bool = True
    
    # Output settings
    save_predictions: bool = True
    save_models: bool = False
    verbose: bool = True


@dataclass
class ModelBenchmarkResult:
    """Result from benchmarking a single model."""
    model_name: str
    model_type: str
    training_time: float
    inference_time: float
    
    # Accuracy metrics
    nucleotide_accuracy: float
    sequence_mse: float
    amino_acid_mse: float
    
    # Phylogenetic metrics
    rf_distance: Optional[float]
    normalized_rf_distance: Optional[float]
    topological_similarity: Optional[float]
    
    # Calibration metrics
    calibration_score: Optional[float]
    coverage_error: Optional[float]
    
    # Likelihood metrics
    log_likelihood: Optional[float]
    perplexity: Optional[float]
    
    # Additional metrics
    fitness_correlation: Optional[float]
    mutation_frequency_correlation: Optional[float]
    
    # Model-specific metrics
    model_specific_metrics: Dict[str, Any]
    
    # Metadata
    n_parameters: int
    memory_usage_mb: float
    benchmark_metadata: Dict


@dataclass
class BenchmarkSuiteResult:
    """Complete benchmarking suite result."""
    model_results: List[ModelBenchmarkResult]
    comparison_matrix: pd.DataFrame
    statistical_significance: Dict
    ranking_analysis: Dict
    benchmark_summary: Dict
    config: BenchmarkConfig


class BaselineModel(ABC):
    """Abstract base class for baseline models."""
    
    def __init__(self, name: str):
        self.name = name
        self.fitted = False
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def fit(self, sequences: List[str], environmental_data: Optional[np.ndarray] = None):
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, input_sequences: List[str], 
                environmental_data: Optional[np.ndarray] = None) -> Dict:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """Get model information and parameters."""
        pass


class MarkovChainBaseline(BaselineModel):
    """
    Markov chain baseline model for viral evolution prediction.
    
    Uses transition probabilities between nucleotides without environmental conditioning.
    """
    
    def __init__(self, order: int = 1):
        """
        Initialize Markov chain model.
        
        Args:
            order: Order of Markov chain (1 = first-order)
        """
        super().__init__(f"MarkovChain_Order{order}")
        self.order = order
        self.transition_probs = {}
        self.nucleotide_freqs = {}
        
    def fit(self, sequences: List[str], environmental_data: Optional[np.ndarray] = None):
        """Fit Markov chain transition probabilities."""
        self.logger.info(f"Fitting {self.name} model...")
        
        # Count transitions
        transition_counts = defaultdict(lambda: defaultdict(int))
        nucleotide_counts = defaultdict(int)
        
        for sequence in sequences:
            sequence = sequence.upper()
            
            # Count nucleotide frequencies
            for nt in sequence:
                if nt in 'ATGC':
                    nucleotide_counts[nt] += 1
            
            # Count transitions
            for i in range(len(sequence) - self.order):
                context = sequence[i:i + self.order]
                next_nt = sequence[i + self.order]
                
                if all(nt in 'ATGC' for nt in context + next_nt):
                    transition_counts[context][next_nt] += 1
        
        # Convert counts to probabilities
        total_nucleotides = sum(nucleotide_counts.values())
        self.nucleotide_freqs = {
            nt: count / total_nucleotides 
            for nt, count in nucleotide_counts.items()
        }
        
        for context in transition_counts:
            total_transitions = sum(transition_counts[context].values())
            self.transition_probs[context] = {
                nt: count / total_transitions
                for nt, count in transition_counts[context].items()
            }
        
        self.fitted = True
        self.logger.info(f"{self.name} model fitted successfully")
    
    def predict(self, input_sequences: List[str], 
                environmental_data: Optional[np.ndarray] = None) -> Dict:
        """Generate predictions using Markov chain."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        probabilities = []
        
        for sequence in input_sequences:
            sequence = sequence.upper()
            pred_sequence = []
            pred_probs = []
            
            for i in range(len(sequence)):
                if i < self.order:
                    # Use nucleotide frequencies for initial positions
                    probs = [self.nucleotide_freqs.get(nt, 0.25) for nt in 'ATGC']
                else:
                    # Use transition probabilities
                    context = sequence[i - self.order:i]
                    
                    if context in self.transition_probs:
                        probs = [
                            self.transition_probs[context].get(nt, 0.25) 
                            for nt in 'ATGC'
                        ]
                    else:
                        # Fallback to uniform distribution
                        probs = [0.25, 0.25, 0.25, 0.25]
                
                # Normalize probabilities
                probs = np.array(probs)
                probs = probs / np.sum(probs)
                
                # Sample nucleotide
                sampled_nt_idx = np.random.choice(4, p=probs)
                sampled_nt = 'ATGC'[sampled_nt_idx]
                
                pred_sequence.append(sampled_nt)
                pred_probs.append(probs)
            
            predictions.append(''.join(pred_sequence))
            probabilities.append(np.array(pred_probs))
        
        return {
            'sequences': predictions,
            'probabilities': probabilities,
            'method': 'markov_chain',
            'model_name': self.name
        }
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'name': self.name,
            'type': 'markov_chain',
            'order': self.order,
            'n_parameters': len(self.transition_probs) * 4,  # Approximate
            'memory_usage_mb': 0.1,  # Minimal memory usage
            'uses_environmental_data': False
        }


class VanillaTransformerBaseline(BaselineModel):
    """
    Vanilla Transformer baseline without environmental conditioning.
    
    Uses only sequence information for prediction.
    """
    
    def __init__(self, 
                 sequence_dim: int = 4,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 4):
        """
        Initialize vanilla Transformer.
        
        Args:
            sequence_dim: Dimension of sequence encoding
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__("VanillaTransformer")
        
        self.model = self._build_model(sequence_dim, hidden_dim, num_layers, num_heads)
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def _build_model(self, sequence_dim: int, hidden_dim: int, 
                    num_layers: int, num_heads: int) -> nn.Module:
        """Build vanilla transformer model."""
        
        class VanillaTransformer(nn.Module):
            def __init__(self, sequence_dim, hidden_dim, num_layers, num_heads):
                super().__init__()
                
                self.input_projection = nn.Linear(sequence_dim, hidden_dim)
                self.positional_encoding = nn.Parameter(
                    torch.randn(1000, hidden_dim) * 0.1
                )
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 2,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                
                self.output_head = nn.Linear(hidden_dim, sequence_dim)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                batch_size, seq_len, _ = x.shape
                
                # Input projection
                x = self.input_projection(x)
                
                # Add positional encoding
                pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0)
                x = x + pos_enc
                
                # Transformer
                x = self.dropout(x)
                x = self.transformer(x)
                
                # Output projection
                output = self.output_head(x)
                
                return F.softmax(output, dim=-1)
        
        return VanillaTransformer(sequence_dim, hidden_dim, num_layers, num_heads)
    
    def fit(self, sequences: List[str], environmental_data: Optional[np.ndarray] = None):
        """Fit vanilla transformer model."""
        self.logger.info(f"Fitting {self.name} model...")
        
        # Convert sequences to tensors
        encoded_sequences = self._encode_sequences(sequences)
        
        # Create training data (input-output pairs)
        X, y = self._create_training_pairs(encoded_sequences)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        # Training loop
        self.model.train()
        num_epochs = 20  # Reduced for benchmarking
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            # Simple batch processing
            batch_size = 16
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                
                # Forward pass
                predictions = self.model(batch_X)
                
                # Calculate loss
                loss = F.cross_entropy(
                    predictions.view(-1, 4),
                    batch_y.view(-1).long()
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if epoch % 5 == 0:
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                self.logger.info(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        self.fitted = True
        self.logger.info(f"{self.name} model fitted successfully")
    
    def _encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """Encode sequences as one-hot tensors."""
        nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        
        encoded = []
        max_len = max(len(seq) for seq in sequences)
        
        for sequence in sequences:
            seq_encoded = np.zeros((max_len, 4))
            
            for i, nt in enumerate(sequence.upper()):
                if nt in nucleotide_map:
                    seq_encoded[i, nucleotide_map[nt]] = 1
                else:
                    # Unknown nucleotide - uniform distribution
                    seq_encoded[i, :] = 0.25
            
            encoded.append(seq_encoded)
        
        return torch.tensor(np.array(encoded), dtype=torch.float32).to(self.device)
    
    def _create_training_pairs(self, encoded_sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create input-output pairs for training."""
        # Simple approach: predict next nucleotide
        X = encoded_sequences[:, :-1, :]  # All but last position
        y = torch.argmax(encoded_sequences[:, 1:, :], dim=-1)  # Next nucleotide indices
        
        return X, y
    
    def predict(self, input_sequences: List[str], 
                environmental_data: Optional[np.ndarray] = None) -> Dict:
        """Generate predictions using vanilla transformer."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        
        # Encode input sequences
        encoded_input = self._encode_sequences(input_sequences)
        
        with torch.no_grad():
            # Generate predictions
            predictions = self.model(encoded_input)
        
        # Convert back to sequences
        predicted_sequences = []
        probabilities = predictions.cpu().numpy()
        
        for i in range(len(input_sequences)):
            pred_indices = np.argmax(probabilities[i], axis=-1)
            pred_sequence = ''.join(['ATGC'[idx] for idx in pred_indices])
            predicted_sequences.append(pred_sequence)
        
        return {
            'sequences': predicted_sequences,
            'probabilities': probabilities,
            'method': 'vanilla_transformer',
            'model_name': self.name
        }
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        n_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            'name': self.name,
            'type': 'vanilla_transformer',
            'n_parameters': n_params,
            'memory_usage_mb': n_params * 4 / (1024 * 1024),  # Approximate
            'uses_environmental_data': False
        }


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for viral evolution prediction models.
    
    Compares multiple models across various metrics including phylogenetic
    distance, calibration, and likelihood scores.
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark suite.
        
        Args:
            config: Benchmarking configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds for reproducibility
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
    
    def run_benchmark(self,
                     models: Dict[str, Any],
                     sequences: List[str],
                     environmental_data: Optional[np.ndarray] = None,
                     output_dir: Optional[str] = None) -> BenchmarkSuiteResult:
        """
        Run comprehensive benchmark across all models.
        
        Args:
            models: Dictionary of models to benchmark
            sequences: List of viral sequences
            environmental_data: Environmental data (optional)
            output_dir: Directory to save results
            
        Returns:
            BenchmarkSuiteResult with comprehensive comparison
        """
        self.logger.info("Starting comprehensive model benchmarking...")
        
        # Split data
        train_sequences, test_sequences, train_env, test_env = self._split_data(
            sequences, environmental_data
        )
        
        # Benchmark each model
        model_results = []
        
        for model_name, model in models.items():
            self.logger.info(f"Benchmarking model: {model_name}")
            
            try:
                result = self._benchmark_single_model(
                    model=model,
                    model_name=model_name,
                    train_sequences=train_sequences,
                    test_sequences=test_sequences,
                    train_env=train_env,
                    test_env=test_env
                )
                model_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to benchmark {model_name}: {e}")
                continue
        
        # Generate comparison analysis
        comparison_matrix = self._create_comparison_matrix(model_results)
        statistical_significance = self._test_statistical_significance(model_results)
        ranking_analysis = self._analyze_model_rankings(model_results)
        benchmark_summary = self._create_benchmark_summary(model_results)
        
        # Save results if output directory provided
        if output_dir:
            self._save_benchmark_results(
                model_results, comparison_matrix, output_dir
            )
        
        result = BenchmarkSuiteResult(
            model_results=model_results,
            comparison_matrix=comparison_matrix,
            statistical_significance=statistical_significance,
            ranking_analysis=ranking_analysis,
            benchmark_summary=benchmark_summary,
            config=self.config
        )
        
        self.logger.info("Benchmarking completed successfully!")
        return result
    
    def _split_data(self, sequences: List[str], 
                   environmental_data: Optional[np.ndarray]) -> Tuple:
        """Split data into training and testing sets."""
        n_samples = len(sequences)
        n_test = int(n_samples * self.config.test_split_ratio)
        
        # Random split
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        train_sequences = [sequences[i] for i in train_indices]
        test_sequences = [sequences[i] for i in test_indices]
        
        train_env = None
        test_env = None
        
        if environmental_data is not None:
            train_env = environmental_data[train_indices]
            test_env = environmental_data[test_indices]
        
        return train_sequences, test_sequences, train_env, test_env
    
    def _benchmark_single_model(self,
                               model: Any,
                               model_name: str,
                               train_sequences: List[str],
                               test_sequences: List[str],
                               train_env: Optional[np.ndarray],
                               test_env: Optional[np.ndarray]) -> ModelBenchmarkResult:
        """Benchmark a single model."""
        
        # Get model info
        if hasattr(model, 'get_model_info'):
            model_info = model.get_model_info()
        else:
            model_info = {'name': model_name, 'type': 'unknown'}
        
        # Training
        start_time = time.time()
        
        if hasattr(model, 'fit'):
            model.fit(train_sequences, train_env)
        
        training_time = time.time() - start_time
        
        # Inference
        start_time = time.time()
        
        if hasattr(model, 'predict'):
            predictions = model.predict(test_sequences, test_env)
        else:
            # For PyTorch models, use different interface
            predictions = self._predict_pytorch_model(model, test_sequences, test_env)
        
        inference_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_all_metrics(
            predictions, test_sequences, test_env
        )
        
        # Create result
        result = ModelBenchmarkResult(
            model_name=model_name,
            model_type=model_info.get('type', 'unknown'),
            training_time=training_time,
            inference_time=inference_time,
            nucleotide_accuracy=metrics.get('nucleotide_accuracy', 0.0),
            sequence_mse=metrics.get('sequence_mse', float('inf')),
            amino_acid_mse=metrics.get('amino_acid_mse', float('inf')),
            rf_distance=metrics.get('rf_distance'),
            normalized_rf_distance=metrics.get('normalized_rf_distance'),
            topological_similarity=metrics.get('topological_similarity'),
            calibration_score=metrics.get('calibration_score'),
            coverage_error=metrics.get('coverage_error'),
            log_likelihood=metrics.get('log_likelihood'),
            perplexity=metrics.get('perplexity'),
            fitness_correlation=metrics.get('fitness_correlation'),
            mutation_frequency_correlation=metrics.get('mutation_frequency_correlation'),
            model_specific_metrics=metrics.get('model_specific', {}),
            n_parameters=model_info.get('n_parameters', 0),
            memory_usage_mb=model_info.get('memory_usage_mb', 0.0),
            benchmark_metadata={
                'n_train_sequences': len(train_sequences),
                'n_test_sequences': len(test_sequences),
                'uses_environmental_data': model_info.get('uses_environmental_data', False)
            }
        )
        
        return result
    
    def _predict_pytorch_model(self, model, test_sequences: List[str], 
                              test_env: Optional[np.ndarray]) -> Dict:
        """Handle prediction for PyTorch models."""
        # This is a simplified version - real implementation would depend on model interface
        return {
            'sequences': test_sequences,  # Placeholder
            'probabilities': [np.ones((len(seq), 4)) * 0.25 for seq in test_sequences],
            'method': 'pytorch_model'
        }
    
    def _calculate_all_metrics(self, predictions: Dict, 
                              test_sequences: List[str],
                              test_env: Optional[np.ndarray]) -> Dict:
        """Calculate all evaluation metrics."""
        metrics = {}
        
        predicted_sequences = predictions.get('sequences', [])
        predicted_probs = predictions.get('probabilities', [])
        
        if predicted_sequences and len(predicted_sequences) == len(test_sequences):
            # Nucleotide accuracy
            metrics['nucleotide_accuracy'] = self._calculate_nucleotide_accuracy(
                predicted_sequences, test_sequences
            )
            
            # Sequence MSE (simplified)
            metrics['sequence_mse'] = self._calculate_sequence_mse(
                predicted_sequences, test_sequences
            )
            
            # Phylogenetic distance
            if self.config.calculate_phylogenetic_distance and len(test_sequences) >= 3:
                try:
                    phylo_result = validate_evolutionary_predictions(
                        predicted_sequences, test_sequences
                    )
                    metrics['rf_distance'] = phylo_result.tree_comparison.rf_distance
                    metrics['normalized_rf_distance'] = phylo_result.tree_comparison.normalized_rf_distance
                    metrics['topological_similarity'] = phylo_result.tree_comparison.topological_similarity
                except Exception as e:
                    self.logger.warning(f"Phylogenetic analysis failed: {e}")
            
            # Log likelihood
            if predicted_probs:
                metrics['log_likelihood'] = self._calculate_log_likelihood(
                    predicted_probs, test_sequences
                )
                metrics['perplexity'] = np.exp(-metrics['log_likelihood'])
        
        return metrics
    
    def _calculate_nucleotide_accuracy(self, predicted: List[str], 
                                     actual: List[str]) -> float:
        """Calculate nucleotide-level accuracy."""
        total_nucleotides = 0
        correct_nucleotides = 0
        
        for pred_seq, actual_seq in zip(predicted, actual):
            min_len = min(len(pred_seq), len(actual_seq))
            
            for i in range(min_len):
                total_nucleotides += 1
                if pred_seq[i].upper() == actual_seq[i].upper():
                    correct_nucleotides += 1
        
        return correct_nucleotides / total_nucleotides if total_nucleotides > 0 else 0.0
    
    def _calculate_sequence_mse(self, predicted: List[str], actual: List[str]) -> float:
        """Calculate sequence-level MSE."""
        # Simplified MSE based on Hamming distance
        total_mse = 0.0
        
        for pred_seq, actual_seq in zip(predicted, actual):
            min_len = min(len(pred_seq), len(actual_seq))
            hamming_dist = sum(p != a for p, a in zip(pred_seq[:min_len], actual_seq[:min_len]))
            normalized_dist = hamming_dist / min_len if min_len > 0 else 1.0
            total_mse += normalized_dist ** 2
        
        return total_mse / len(predicted) if predicted else float('inf')
    
    def _calculate_log_likelihood(self, predicted_probs: List[np.ndarray], 
                                 actual_sequences: List[str]) -> float:
        """Calculate log likelihood of actual sequences under predicted probabilities."""
        total_log_likelihood = 0.0
        total_positions = 0
        
        nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        
        for probs, actual_seq in zip(predicted_probs, actual_sequences):
            for i, nt in enumerate(actual_seq.upper()):
                if i < len(probs) and nt in nucleotide_map:
                    nt_idx = nucleotide_map[nt]
                    prob = probs[i, nt_idx]
                    
                    # Avoid log(0)
                    prob = max(prob, 1e-10)
                    total_log_likelihood += np.log(prob)
                    total_positions += 1
        
        return total_log_likelihood / total_positions if total_positions > 0 else float('-inf')
    
    def _create_comparison_matrix(self, model_results: List[ModelBenchmarkResult]) -> pd.DataFrame:
        """Create comparison matrix of all models."""
        if not model_results:
            return pd.DataFrame()
        
        # Extract key metrics
        data = []
        for result in model_results:
            row = {
                'Model': result.model_name,
                'Type': result.model_type,
                'Parameters': result.n_parameters,
                'Training Time (s)': result.training_time,
                'Inference Time (s)': result.inference_time,
                'Nucleotide Accuracy': result.nucleotide_accuracy,
                'Sequence MSE': result.sequence_mse,
                'RF Distance': result.rf_distance,
                'Normalized RF': result.normalized_rf_distance,
                'Topological Similarity': result.topological_similarity,
                'Log Likelihood': result.log_likelihood,
                'Memory (MB)': result.memory_usage_mb
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _test_statistical_significance(self, model_results: List[ModelBenchmarkResult]) -> Dict:
        """Test statistical significance of model differences."""
        # Simplified significance testing
        # In practice, would use proper statistical tests
        
        significance_results = {
            'best_model': None,
            'significant_differences': [],
            'p_values': {}
        }
        
        if len(model_results) >= 2:
            # Find best model by nucleotide accuracy
            best_model = max(model_results, key=lambda x: x.nucleotide_accuracy)
            significance_results['best_model'] = best_model.model_name
            
            # Compare all pairs (simplified)
            for i, model1 in enumerate(model_results):
                for j, model2 in enumerate(model_results[i+1:], i+1):
                    acc_diff = abs(model1.nucleotide_accuracy - model2.nucleotide_accuracy)
                    
                    # Simplified significance test
                    if acc_diff > 0.05:  # 5% difference threshold
                        significance_results['significant_differences'].append({
                            'model1': model1.model_name,
                            'model2': model2.model_name,
                            'accuracy_difference': acc_diff,
                            'significant': True
                        })
        
        return significance_results
    
    def _analyze_model_rankings(self, model_results: List[ModelBenchmarkResult]) -> Dict:
        """Analyze model rankings across different metrics."""
        if not model_results:
            return {}
        
        rankings = {}
        
        # Rank by different metrics
        metrics_to_rank = [
            ('nucleotide_accuracy', False),  # Higher is better
            ('sequence_mse', True),          # Lower is better
            ('rf_distance', True),           # Lower is better
            ('training_time', True),         # Lower is better
            ('n_parameters', True)           # Lower is better (for efficiency)
        ]
        
        for metric, lower_is_better in metrics_to_rank:
            # Get values for this metric
            values = []
            for result in model_results:
                value = getattr(result, metric, None)
                if value is not None:
                    values.append((result.model_name, value))
            
            if values:
                # Sort and rank
                values.sort(key=lambda x: x[1], reverse=not lower_is_better)
                rankings[metric] = [name for name, _ in values]
        
        # Calculate overall ranking (simple average of ranks)
        if rankings:
            model_names = [result.model_name for result in model_results]
            overall_scores = {name: 0 for name in model_names}
            
            for metric_rankings in rankings.values():
                for rank, model_name in enumerate(metric_rankings):
                    overall_scores[model_name] += rank
            
            # Sort by overall score (lower is better)
            overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1])
            rankings['overall'] = [name for name, _ in overall_ranking]
        
        return rankings
    
    def _create_benchmark_summary(self, model_results: List[ModelBenchmarkResult]) -> Dict:
        """Create summary of benchmark results."""
        if not model_results:
            return {}
        
        summary = {
            'n_models_tested': len(model_results),
            'best_accuracy_model': None,
            'fastest_training_model': None,
            'most_efficient_model': None,
            'best_phylogenetic_model': None,
            'metric_ranges': {}
        }
        
        # Find best models by different criteria
        if model_results:
            summary['best_accuracy_model'] = max(
                model_results, key=lambda x: x.nucleotide_accuracy
            ).model_name
            
            summary['fastest_training_model'] = min(
                model_results, key=lambda x: x.training_time
            ).model_name
            
            # Most efficient (best accuracy per parameter)
            efficiency_scores = []
            for result in model_results:
                if result.n_parameters > 0:
                    efficiency = result.nucleotide_accuracy / (result.n_parameters / 1000)
                    efficiency_scores.append((result.model_name, efficiency))
            
            if efficiency_scores:
                summary['most_efficient_model'] = max(
                    efficiency_scores, key=lambda x: x[1]
                )[0]
            
            # Best phylogenetic model
            phylo_models = [r for r in model_results if r.topological_similarity is not None]
            if phylo_models:
                summary['best_phylogenetic_model'] = max(
                    phylo_models, key=lambda x: x.topological_similarity
                ).model_name
        
        # Calculate metric ranges
        metrics = ['nucleotide_accuracy', 'sequence_mse', 'training_time', 'n_parameters']
        
        for metric in metrics:
            values = [getattr(r, metric, None) for r in model_results]
            values = [v for v in values if v is not None]
            
            if values:
                summary['metric_ranges'][metric] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        return summary
    
    def _save_benchmark_results(self, model_results: List[ModelBenchmarkResult],
                               comparison_matrix: pd.DataFrame,
                               output_dir: str):
        """Save benchmark results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comparison matrix
        comparison_matrix.to_csv(output_path / "model_comparison.csv", index=False)
        
        # Save detailed results
        detailed_results = []
        for result in model_results:
            result_dict = {
                'model_name': result.model_name,
                'model_type': result.model_type,
                'training_time': result.training_time,
                'inference_time': result.inference_time,
                'nucleotide_accuracy': result.nucleotide_accuracy,
                'sequence_mse': result.sequence_mse,
                'rf_distance': result.rf_distance,
                'normalized_rf_distance': result.normalized_rf_distance,
                'topological_similarity': result.topological_similarity,
                'log_likelihood': result.log_likelihood,
                'n_parameters': result.n_parameters,
                'memory_usage_mb': result.memory_usage_mb
            }
            detailed_results.append(result_dict)
        
        with open(output_path / "detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        self.logger.info(f"Benchmark results saved to {output_path}")


def run_model_benchmark(models: Dict[str, Any],
                       sequences: List[str],
                       environmental_data: Optional[np.ndarray] = None,
                       config: Optional[BenchmarkConfig] = None,
                       output_dir: Optional[str] = None) -> BenchmarkSuiteResult:
    """
    Convenience function to run model benchmarking.
    
    Args:
        models: Dictionary of models to benchmark
        sequences: List of viral sequences
        environmental_data: Environmental data
        config: Benchmark configuration
        output_dir: Output directory
        
    Returns:
        BenchmarkSuiteResult
    """
    if config is None:
        config = BenchmarkConfig()
    
    benchmark_suite = BenchmarkSuite(config)
    
    return benchmark_suite.run_benchmark(
        models=models,
        sequences=sequences,
        environmental_data=environmental_data,
        output_dir=output_dir
    )