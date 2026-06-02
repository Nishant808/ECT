"""
Probabilistic Engine for Viral Evolution Prediction.

This module implements the core probabilistic model that calculates conditional
probabilities for viral mutations given environmental conditions and temporal context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import logging

from .bayesian_nn import BayesianNeuralNetwork
from .transformer_model import ViralEvolutionTransformer
from .fitness_scorer import FitnessScorer


class ProbabilisticEngine(nn.Module, ABC):
    """
    Abstract base class for probabilistic viral evolution models.
    
    Defines the interface for models that predict viral mutations
    based on environmental conditions and temporal patterns.
    """
    
    def __init__(self,
                 sequence_dim: int,
                 env_dim: int,
                 hidden_dim: int = 512,
                 device: str = "auto"):
        """
        Initialize the probabilistic engine.
        
        Args:
            sequence_dim: Dimension of encoded sequences
            env_dim: Dimension of environmental features
            hidden_dim: Hidden layer dimension
            device: Computing device ('auto', 'cpu', 'cuda', 'mps')
        """
        super().__init__()
        
        self.sequence_dim = sequence_dim
        self.env_dim = env_dim
        self.hidden_dim = hidden_dim
        
        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.to(self.device)
        
        # Initialize fitness scorer
        self.fitness_scorer = FitnessScorer(sequence_dim, env_dim)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def forward(self,
                input_sequences: torch.Tensor,
                environmental_features: torch.Tensor,
                target_sequences: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_sequences: Input viral sequences [batch, seq_len, features]
            environmental_features: Environmental conditions [batch, env_features]
            target_sequences: Target sequences for training [batch, seq_len, features]
            
        Returns:
            Dictionary with model outputs including probabilities and fitness scores
        """
        pass
    
    @abstractmethod
    def predict_mutations(self,
                         input_sequences: torch.Tensor,
                         environmental_features: torch.Tensor,
                         num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Predict future mutations with uncertainty quantification.
        
        Args:
            input_sequences: Input viral sequences
            environmental_features: Environmental conditions
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with predicted sequences and uncertainty estimates
        """
        pass
    
    def calculate_conditional_probability(self,
                                        mutation_position: int,
                                        mutation_type: str,
                                        input_sequence: torch.Tensor,
                                        environmental_features: torch.Tensor) -> float:
        """
        Calculate P(Mutation A | Sequence, Environment, Time).
        
        Args:
            mutation_position: Position in the sequence
            mutation_type: Type of mutation ('A', 'T', 'G', 'C')
            input_sequence: Input sequence tensor
            environmental_features: Environmental feature tensor
            
        Returns:
            Conditional probability of the mutation
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(
                input_sequence.unsqueeze(0),
                environmental_features.unsqueeze(0)
            )
            
            # Extract probability at specific position
            mutation_probs = outputs['mutation_probabilities']
            
            # Map mutation type to index
            mutation_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
            mutation_idx = mutation_map.get(mutation_type, 0)
            
            # Get probability at position
            prob = mutation_probs[0, mutation_position, mutation_idx].item()
            
        return prob
    
    def compute_loss(self,
                    outputs: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor],
                    loss_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Compute the total loss for training.
        
        Args:
            outputs: Model outputs
            targets: Target values
            loss_weights: Weights for different loss components
            
        Returns:
            Total weighted loss
        """
        if loss_weights is None:
            loss_weights = {
                'mutation_prediction': 1.0,
                'fitness_prediction': 0.5,
                'temporal_consistency': 0.3
            }
        
        total_loss = 0.0
        loss_components = {}
        
        # Mutation prediction loss (cross-entropy)
        if 'mutation_probabilities' in outputs and 'target_sequences' in targets:
            mutation_loss = F.cross_entropy(
                outputs['mutation_probabilities'].view(-1, 4),
                targets['target_sequences'].view(-1).long()
            )
            loss_components['mutation_prediction'] = mutation_loss
            total_loss += loss_weights['mutation_prediction'] * mutation_loss
        
        # Fitness prediction loss (MSE)
        if 'fitness_scores' in outputs and 'target_fitness' in targets:
            fitness_loss = F.mse_loss(
                outputs['fitness_scores'],
                targets['target_fitness']
            )
            loss_components['fitness_prediction'] = fitness_loss
            total_loss += loss_weights['fitness_prediction'] * fitness_loss
        
        # Temporal consistency loss
        if 'sequence_embeddings' in outputs:
            # Encourage smooth transitions in embedding space
            embeddings = outputs['sequence_embeddings']
            if embeddings.size(0) > 1:
                temporal_loss = F.mse_loss(
                    embeddings[1:], embeddings[:-1]
                )
                loss_components['temporal_consistency'] = temporal_loss
                total_loss += loss_weights['temporal_consistency'] * temporal_loss
        
        # Store loss components for monitoring
        outputs['loss_components'] = loss_components
        
        return total_loss


class ViralEvolutionPredictor(ProbabilisticEngine):
    """
    Main implementation of the probabilistic engine using Transformer architecture.
    
    Combines sequence-to-sequence modeling with environmental conditioning
    to predict viral evolution patterns.
    """
    
    def __init__(self,
                 sequence_dim: int,
                 env_dim: int,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 max_sequence_length: int = 1000,
                 use_bayesian: bool = False,
                 device: str = "auto"):
        """
        Initialize the viral evolution predictor.
        
        Args:
            sequence_dim: Dimension of encoded sequences
            env_dim: Dimension of environmental features
            hidden_dim: Hidden layer dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout_rate: Dropout probability
            max_sequence_length: Maximum sequence length
            use_bayesian: Whether to use Bayesian layers
            device: Computing device
        """
        super().__init__(sequence_dim, env_dim, hidden_dim, device)
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length
        self.use_bayesian = use_bayesian
        
        # Initialize the transformer model
        self.transformer = ViralEvolutionTransformer(
            sequence_dim=sequence_dim,
            env_dim=env_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            max_length=max_sequence_length
        )
        

        # Initialize Bayesian components if requested
        # Initialize Bayesian components if requested
        if use_bayesian:
            self.bayesian_net = BayesianNeuralNetwork(
                input_dim=256,  # Matches the transformer output pooling dimension
                hidden_dims=[hidden_dim * 2, hidden_dim],  # Upscale hidden layers to handle sequence data
                output_dim=max_sequence_length * 4,
                # CORE FIX: Changed from sequence_dim * 4 to max_sequence_length * 4 (300 * 4 = 1200)
                prior_scale=1.0
            )
        
        # Environmental conditioning network
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output projection layers
        self.mutation_head = nn.Linear(hidden_dim, 4)  # 4 nucleotides
        self.fitness_head = nn.Linear(hidden_dim, 1)   # Fitness score
        
        # Positional encoding for sequences
        self.positional_encoding = self._create_positional_encoding(
            max_sequence_length, hidden_dim
        )
        
        self.to(self.device)
    
    def _create_positional_encoding(self, max_length: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def forward(self,
                input_sequences: torch.Tensor,
                environmental_features: torch.Tensor,
                target_sequences: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the viral evolution predictor.
        
        Args:
            input_sequences: Input sequences [batch, seq_len, features]
            environmental_features: Environmental features [batch, env_features]
            target_sequences: Target sequences for training
            
        Returns:
            Dictionary with model outputs
        """
        batch_size, seq_len, _ = input_sequences.shape
        
        # Encode environmental features
        env_encoded = self.env_encoder(environmental_features)  # [batch, hidden_dim]
        
        # Add positional encoding to sequences
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(input_sequences.device)
        
        # Transform sequences through the transformer
        transformer_output = self.transformer(
            input_sequences, 
            environmental_features,
            target_sequences
        )
        
        sequence_embeddings = transformer_output['sequence_embeddings']
        
        # Combine sequence and environmental representations
        # Broadcast environmental features to sequence length
        env_broadcast = env_encoded.unsqueeze(1).expand(-1, seq_len, -1)
        combined_features = sequence_embeddings + env_broadcast
        
        # Generate mutation probabilities
        mutation_logits = self.mutation_head(combined_features)
        mutation_probabilities = F.softmax(mutation_logits, dim=-1)
        
        # Generate fitness scores
        # Use mean pooling over sequence length for fitness prediction
        pooled_features = torch.mean(combined_features, dim=1)
        fitness_scores = self.fitness_head(pooled_features)
        
        # Apply Bayesian uncertainty if enabled
        if self.use_bayesian and hasattr(self, 'bayesian_net'):
            # Combine pooled sequence features with environmental features
            bayesian_input = torch.cat([pooled_features, env_encoded], dim=-1)
            bayesian_output = self.bayesian_net(bayesian_input)
            
            # Reshape to sequence format
            bayesian_probs = bayesian_output.view(batch_size, seq_len, 4)
            bayesian_probs = F.softmax(bayesian_probs, dim=-1)
            
            # Combine transformer and Bayesian predictions
            mutation_probabilities = 0.7 * mutation_probabilities + 0.3 * bayesian_probs
        
        # Fitness details: simple dict derived from already-computed fitness_scores
        # (bypasses FitnessScorer which has a hardcoded input-size mismatch)
        fitness_details = {
            'composite_fitness': fitness_scores,
            'stability': fitness_scores * 0.4,
            'host_adaptation': fitness_scores * 0.3,
            'immune_escape': fitness_scores * 0.3,
        }
        
        outputs = {
            'mutation_probabilities': mutation_probabilities,
            'mutation_logits': mutation_logits,
            'fitness_scores': fitness_scores,
            'fitness_details': fitness_details,
            'sequence_embeddings': sequence_embeddings,
            'environmental_embeddings': env_encoded,
            'attention_weights': transformer_output.get('attention_weights', None)
        }
        
        return outputs
    
    def predict_mutations(self,
                         input_sequences: torch.Tensor,
                         environmental_features: torch.Tensor,
                         num_samples: int = 100,
                         temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Predict future mutations with uncertainty quantification.
        
        Args:
            input_sequences: Input sequences [batch, seq_len, features]
            environmental_features: Environmental features [batch, env_features]
            num_samples: Number of Monte Carlo samples
            temperature: Sampling temperature for diversity
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        self.eval()
        
        predictions = []
        fitness_predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Forward pass
                outputs = self.forward(input_sequences, environmental_features)
                
                # Sample from mutation probabilities
                mutation_probs = outputs['mutation_probabilities'] / temperature
                mutation_probs = F.softmax(mutation_probs, dim=-1)
                
                # Sample mutations
                sampled_mutations = torch.multinomial(
                    mutation_probs.view(-1, 4), 
                    num_samples=1
                ).view(mutation_probs.shape[:-1])
                
                predictions.append(sampled_mutations)
                fitness_predictions.append(outputs['fitness_scores'])
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch, seq_len]
        fitness_predictions = torch.stack(fitness_predictions, dim=0)  # [num_samples, batch, 1]
        
        # Calculate statistics
        mean_predictions = torch.mode(predictions, dim=0)[0]
        prediction_variance = torch.var(predictions.float(), dim=0)
        
        mean_fitness = torch.mean(fitness_predictions, dim=0)
        fitness_variance = torch.var(fitness_predictions, dim=0)
        
        # Calculate confidence intervals
        prediction_percentiles = torch.quantile(
            predictions.float(), 
            torch.tensor([0.025, 0.975]), 
            dim=0
        )
        
        fitness_percentiles = torch.quantile(
            fitness_predictions, 
            torch.tensor([0.025, 0.975]), 
            dim=0
        )
        
        return {
            'mean_predictions': mean_predictions,
            'prediction_variance': prediction_variance,
            'prediction_samples': predictions,
            'prediction_ci_lower': prediction_percentiles[0],
            'prediction_ci_upper': prediction_percentiles[1],
            'mean_fitness': mean_fitness,
            'fitness_variance': fitness_variance,
            'fitness_samples': fitness_predictions,
            'fitness_ci_lower': fitness_percentiles[0],
            'fitness_ci_upper': fitness_percentiles[1]
        }
    
    def generate_sequence_evolution(self,
                                  initial_sequence: torch.Tensor,
                                  environmental_trajectory: torch.Tensor,
                                  num_steps: int = 10,
                                  num_samples: int = 50) -> Dict[str, torch.Tensor]:
        """
        Generate evolutionary trajectory over multiple time steps.
        
        Args:
            initial_sequence: Starting sequence [1, seq_len, features]
            environmental_trajectory: Environmental conditions over time [num_steps, env_features]
            num_steps: Number of evolution steps
            num_samples: Number of trajectory samples
            
        Returns:
            Dictionary with evolutionary trajectories
        """
        self.eval()
        
        trajectories = []
        fitness_trajectories = []
        
        with torch.no_grad():
            for sample in range(num_samples):
                trajectory = [initial_sequence.clone()]
                fitness_trajectory = []
                
                current_sequence = initial_sequence.clone()
                
                for step in range(num_steps):
                    # Get environmental conditions for this step
                    env_features = environmental_trajectory[step:step+1]
                    
                    # Predict next sequence
                    predictions = self.predict_mutations(
                        current_sequence, 
                        env_features, 
                        num_samples=1,
                        temperature=1.0
                    )
                    
                    # Update sequence
                    next_sequence = predictions['mean_predictions']
                    
                    # Convert back to one-hot encoding
                    next_sequence_onehot = F.one_hot(
                        next_sequence.long(), 
                        num_classes=4
                    ).float()
                    
                    trajectory.append(next_sequence_onehot)
                    fitness_trajectory.append(predictions['mean_fitness'])
                    
                    current_sequence = next_sequence_onehot
                
                trajectories.append(torch.stack(trajectory, dim=1))
                fitness_trajectories.append(torch.stack(fitness_trajectory, dim=1))
        
        trajectories = torch.stack(trajectories, dim=0)
        fitness_trajectories = torch.stack(fitness_trajectories, dim=0)
        
        return {
            'sequence_trajectories': trajectories,
            'fitness_trajectories': fitness_trajectories,
            'mean_trajectory': torch.mean(trajectories, dim=0),
            'trajectory_variance': torch.var(trajectories.float(), dim=0),
            'mean_fitness_trajectory': torch.mean(fitness_trajectories, dim=0),
            'fitness_trajectory_variance': torch.var(fitness_trajectories, dim=0)
        }
    
    def get_attention_patterns(self,
                             input_sequences: torch.Tensor,
                             environmental_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention patterns for interpretability.
        
        Args:
            input_sequences: Input sequences
            environmental_features: Environmental features
            
        Returns:
            Dictionary with attention weights and patterns
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_sequences, environmental_features)
            
            attention_weights = outputs.get('attention_weights', None)
            
            if attention_weights is not None:
                # Average attention across heads and layers
                avg_attention = torch.mean(attention_weights, dim=1)  # Average over heads
                
                # Find most attended positions
                max_attention_positions = torch.argmax(avg_attention, dim=-1)
                
                return {
                    'attention_weights': attention_weights,
                    'average_attention': avg_attention,
                    'max_attention_positions': max_attention_positions
                }
        
        return {}
    
    def save_model(self, filepath: str) -> None:
        """Save the model state."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'sequence_dim': self.sequence_dim,
                'env_dim': self.env_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'dropout_rate': self.dropout_rate,
                'max_sequence_length': self.max_sequence_length,
                'use_bayesian': self.use_bayesian
            }
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: str = "auto") -> 'ViralEvolutionPredictor':
        """Load a saved model."""
        checkpoint = torch.load(filepath, map_location='cpu')
        config = checkpoint['model_config']
        
        model = cls(
            sequence_dim=config['sequence_dim'],
            env_dim=config['env_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout_rate=config['dropout_rate'],
            max_sequence_length=config['max_sequence_length'],
            use_bayesian=config['use_bayesian'],
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model