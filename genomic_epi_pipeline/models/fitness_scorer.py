"""
Fitness Scorer for Viral Sequences.

This module implements fitness scoring that evaluates thermodynamic stability
and host-adaptation probability for predicted viral mutations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class FitnessScorer(nn.Module):
    """
    Fitness scorer that evaluates viral sequence fitness based on
    thermodynamic stability and environmental adaptation.
    """
    
    def __init__(self,
                 sequence_dim: int,
                 env_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout_rate: float = 0.1):
        """
        Initialize the fitness scorer.
        
        Args:
            sequence_dim: Dimension of sequence representation
            env_dim: Dimension of environmental features
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.sequence_dim = sequence_dim
        self.env_dim = env_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Thermodynamic stability predictor
        self.stability_network = self._build_stability_network()
        
        # Host adaptation predictor
        self.adaptation_network = self._build_adaptation_network()
        
        # Environmental fitness predictor
        self.env_fitness_network = self._build_env_fitness_network()
        
        # Sequence complexity analyzer
        self.complexity_analyzer = self._build_complexity_analyzer()
        
        # Final fitness aggregator
        self.fitness_aggregator = nn.Sequential(
            nn.Linear(4, hidden_dim),  # 4 fitness components
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Fitness score between 0 and 1
        )
        
        # Learnable weights for fitness components
        self.component_weights = nn.Parameter(torch.ones(4))
        
    def _build_stability_network(self) -> nn.Module:
        """Build thermodynamic stability prediction network."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.sequence_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # Output layer (stability score)
        layers.append(nn.Linear(self.hidden_dim, 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def _build_adaptation_network(self) -> nn.Module:
        """Build host adaptation prediction network."""
        layers = []
        
        # Input layer (sequence + simplified host features)
        input_dim = self.sequence_dim + 10  # Assume 10 host-related features
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # Output layer (adaptation score)
        layers.append(nn.Linear(self.hidden_dim, 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def _build_env_fitness_network(self) -> nn.Module:
        """Build environmental fitness prediction network."""
        layers = []
        
        # Input layer (sequence + environmental features)
        input_dim = self.sequence_dim + self.env_dim
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        # Hidden layers with environmental gating
        for _ in range(self.num_layers - 1):
            layers.append(EnvironmentalGatedLayer(self.hidden_dim, self.env_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # Output layer (environmental fitness score)
        layers.append(nn.Linear(self.hidden_dim, 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def _build_complexity_analyzer(self) -> nn.Module:
        """Build sequence complexity analyzer."""
        return nn.Sequential(
            nn.Linear(self.sequence_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def _calculate_gc_content_fitness(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Calculate fitness based on GC content.
        
        Args:
            sequences: Sequence probabilities [batch, seq_len, 4] (A, T, G, C)
            
        Returns:
            GC content fitness scores [batch, 1]
        """
        # Extract G and C probabilities
        gc_probs = sequences[:, :, [2, 3]]  # G and C are indices 2 and 3
        gc_content = torch.sum(gc_probs, dim=-1).mean(dim=-1, keepdim=True)  # [batch, 1]
        
        # Optimal GC content is around 40-60% for many viruses
        optimal_gc = 0.5
        gc_deviation = torch.abs(gc_content - optimal_gc)
        
        # Convert to fitness score (higher is better)
        gc_fitness = torch.exp(-5 * gc_deviation)  # Exponential decay from optimal
        
        return gc_fitness
    
    def _calculate_codon_usage_fitness(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Calculate fitness based on codon usage bias.
        
        Args:
            sequences: Sequence probabilities [batch, seq_len, 4]
            
        Returns:
            Codon usage fitness scores [batch, 1]
        """
        batch_size, seq_len, _ = sequences.shape
        
        # Simplified codon usage analysis
        # Group sequences into codons (triplets)
        if seq_len % 3 != 0:
            # Pad to make divisible by 3
            padding = 3 - (seq_len % 3)
            sequences = F.pad(sequences, (0, 0, 0, padding))
            seq_len = sequences.shape[1]
        
        # Reshape to codons
        codons = sequences.view(batch_size, seq_len // 3, 3, 4)
        
        # Calculate codon probabilities (simplified)
        # For each codon position, find most likely nucleotide
        codon_indices = torch.argmax(codons, dim=-1)  # [batch, num_codons, 3]
        
        # Convert to codon codes (0-63)
        codon_codes = (codon_indices[:, :, 0] * 16 + 
                      codon_indices[:, :, 1] * 4 + 
                      codon_indices[:, :, 2])  # [batch, num_codons]
        
        # Calculate codon diversity (higher diversity = higher fitness)
        codon_diversity = []
        for i in range(batch_size):
            unique_codons = torch.unique(codon_codes[i])
            diversity = len(unique_codons) / 64.0  # Normalize by total possible codons
            codon_diversity.append(diversity)
        
        codon_fitness = torch.tensor(codon_diversity, device=sequences.device).unsqueeze(1)
        
        return codon_fitness
    
    def _calculate_secondary_structure_fitness(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Calculate fitness based on predicted secondary structure stability.
        
        Args:
            sequences: Sequence probabilities [batch, seq_len, 4]
            
        Returns:
            Secondary structure fitness scores [batch, 1]
        """
        # Simplified secondary structure prediction
        # Based on base pairing probabilities
        
        batch_size, seq_len, _ = sequences.shape
        
        # Calculate base pairing potential
        # A-T and G-C pairs are more stable
        a_probs = sequences[:, :, 0]  # A
        t_probs = sequences[:, :, 1]  # T
        g_probs = sequences[:, :, 2]  # G
        c_probs = sequences[:, :, 3]  # C
        
        # Calculate pairing potential for each position
        at_pairing = torch.minimum(a_probs, t_probs)
        gc_pairing = torch.minimum(g_probs, c_probs)
        
        # GC pairs are more stable (weight = 3), AT pairs less stable (weight = 2)
        pairing_stability = 3 * gc_pairing + 2 * at_pairing
        
        # Average stability across sequence
        structure_fitness = pairing_stability.mean(dim=-1, keepdim=True)
        
        return structure_fitness
    
    def forward(self,
                sequence_probs: torch.Tensor,
                environmental_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate comprehensive fitness scores.
        
        Args:
            sequence_probs: Sequence probabilities [batch, seq_len, 4]
            environmental_features: Environmental features [batch, env_dim]
            
        Returns:
            Dictionary with fitness scores and components
        """
        batch_size, seq_len, _ = sequence_probs.shape
        
        # Flatten sequence for network input
        sequence_flat = sequence_probs.view(batch_size, -1)
        
        # 1. Thermodynamic stability
        stability_score = self.stability_network(sequence_flat)
        
        # 2. Host adaptation (simplified - use subset of env features as host proxy)
        host_features = environmental_features[:, :10] if environmental_features.shape[1] >= 10 else environmental_features
        if host_features.shape[1] < 10:
            # Pad with zeros if not enough features
            padding = torch.zeros(batch_size, 10 - host_features.shape[1], device=host_features.device)
            host_features = torch.cat([host_features, padding], dim=1)
        
        adaptation_input = torch.cat([sequence_flat, host_features], dim=1)
        adaptation_score = self.adaptation_network(adaptation_input)
        
        # 3. Environmental fitness
        env_input = torch.cat([sequence_flat, environmental_features], dim=1)
        env_fitness_score = self.env_fitness_network(env_input)
        
        # 4. Sequence complexity
        complexity_score = self.complexity_analyzer(sequence_flat)
        
        # Additional biologically-motivated fitness components
        gc_fitness = self._calculate_gc_content_fitness(sequence_probs)
        codon_fitness = self._calculate_codon_usage_fitness(sequence_probs)
        structure_fitness = self._calculate_secondary_structure_fitness(sequence_probs)
        
        # Combine all fitness components
        fitness_components = torch.cat([
            stability_score,
            adaptation_score,
            env_fitness_score,
            complexity_score
        ], dim=1)
        
        # Apply learnable weights
        weighted_components = fitness_components * F.softmax(self.component_weights, dim=0)
        
        # Final fitness score
        final_fitness = self.fitness_aggregator(weighted_components)
        
        # Detailed fitness breakdown
        fitness_details = {
            'total_fitness': final_fitness,
            'stability_score': stability_score,
            'adaptation_score': adaptation_score,
            'environmental_fitness': env_fitness_score,
            'complexity_score': complexity_score,
            'gc_content_fitness': gc_fitness,
            'codon_usage_fitness': codon_fitness,
            'structure_fitness': structure_fitness,
            'component_weights': F.softmax(self.component_weights, dim=0)
        }
        
        return fitness_details
    
    def predict_fitness_change(self,
                             original_sequence: torch.Tensor,
                             mutated_sequence: torch.Tensor,
                             environmental_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict fitness change due to mutations.
        
        Args:
            original_sequence: Original sequence probabilities
            mutated_sequence: Mutated sequence probabilities
            environmental_features: Environmental features
            
        Returns:
            Dictionary with fitness changes
        """
        # Calculate fitness for both sequences
        original_fitness = self.forward(original_sequence, environmental_features)
        mutated_fitness = self.forward(mutated_sequence, environmental_features)
        
        # Calculate changes
        fitness_change = {
            'total_fitness_change': mutated_fitness['total_fitness'] - original_fitness['total_fitness'],
            'stability_change': mutated_fitness['stability_score'] - original_fitness['stability_score'],
            'adaptation_change': mutated_fitness['adaptation_score'] - original_fitness['adaptation_score'],
            'environmental_change': mutated_fitness['environmental_fitness'] - original_fitness['environmental_fitness'],
            'complexity_change': mutated_fitness['complexity_score'] - original_fitness['complexity_score']
        }
        
        # Add original and mutated fitness for reference
        fitness_change['original_fitness'] = original_fitness
        fitness_change['mutated_fitness'] = mutated_fitness
        
        return fitness_change
    
    def get_fitness_landscape(self,
                            sequence_template: torch.Tensor,
                            environmental_features: torch.Tensor,
                            mutation_positions: List[int],
                            mutation_types: List[str]) -> torch.Tensor:
        """
        Generate fitness landscape for specific mutations.
        
        Args:
            sequence_template: Template sequence [1, seq_len, 4]
            environmental_features: Environmental features [1, env_dim]
            mutation_positions: List of positions to mutate
            mutation_types: List of nucleotides to try ('A', 'T', 'G', 'C')
            
        Returns:
            Fitness landscape tensor [num_mutations, 1]
        """
        nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        fitness_scores = []
        
        for pos in mutation_positions:
            for nt in mutation_types:
                # Create mutated sequence
                mutated_seq = sequence_template.clone()
                
                # Set mutation (one-hot encoding)
                mutated_seq[0, pos, :] = 0
                mutated_seq[0, pos, nucleotide_map[nt]] = 1
                
                # Calculate fitness
                fitness = self.forward(mutated_seq, environmental_features)
                fitness_scores.append(fitness['total_fitness'])
        
        return torch.cat(fitness_scores, dim=0)


class EnvironmentalGatedLayer(nn.Module):
    """
    Layer with environmental gating for context-dependent processing.
    """
    
    def __init__(self, hidden_dim: int, env_dim: int):
        """
        Initialize environmental gated layer.
        
        Args:
            hidden_dim: Hidden layer dimension
            env_dim: Environmental feature dimension
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.env_dim = env_dim
        
        # Main transformation
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Environmental gate
        self.env_gate = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Environmental modulation
        self.env_modulation = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor, env_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with environmental gating.
        
        Args:
            x: Input tensor [batch, hidden_dim]
            env_features: Environmental features [batch, env_dim]
            
        Returns:
            Gated output tensor [batch, hidden_dim]
        """
        # Main transformation
        transformed = self.linear(x)
        
        # Environmental gating
        gate = self.env_gate(env_features)
        modulation = self.env_modulation(env_features)
        
        # Apply gating and modulation
        output = transformed * gate + x * (1 - gate) + modulation
        
        return output


class FitnessLandscapeAnalyzer:
    """
    Analyzer for fitness landscapes and evolutionary trajectories.
    """
    
    def __init__(self, fitness_scorer: FitnessScorer):
        """
        Initialize fitness landscape analyzer.
        
        Args:
            fitness_scorer: Trained fitness scorer model
        """
        self.fitness_scorer = fitness_scorer
    
    def analyze_mutational_robustness(self,
                                    sequence: torch.Tensor,
                                    environmental_features: torch.Tensor,
                                    num_mutations: int = 100) -> Dict[str, float]:
        """
        Analyze mutational robustness of a sequence.
        
        Args:
            sequence: Input sequence [1, seq_len, 4]
            environmental_features: Environmental features [1, env_dim]
            num_mutations: Number of random mutations to test
            
        Returns:
            Dictionary with robustness metrics
        """
        original_fitness = self.fitness_scorer(sequence, environmental_features)
        original_score = original_fitness['total_fitness'].item()
        
        fitness_changes = []
        seq_len = sequence.shape[1]
        
        for _ in range(num_mutations):
            # Random mutation
            mutated_seq = sequence.clone()
            pos = torch.randint(0, seq_len, (1,)).item()
            new_nt = torch.randint(0, 4, (1,)).item()
            
            # Apply mutation
            mutated_seq[0, pos, :] = 0
            mutated_seq[0, pos, new_nt] = 1
            
            # Calculate fitness change
            mutated_fitness = self.fitness_scorer(mutated_seq, environmental_features)
            fitness_change = mutated_fitness['total_fitness'].item() - original_score
            fitness_changes.append(fitness_change)
        
        fitness_changes = np.array(fitness_changes)
        
        return {
            'mean_fitness_change': float(np.mean(fitness_changes)),
            'std_fitness_change': float(np.std(fitness_changes)),
            'fraction_beneficial': float(np.mean(fitness_changes > 0)),
            'fraction_neutral': float(np.mean(np.abs(fitness_changes) < 0.01)),
            'fraction_deleterious': float(np.mean(fitness_changes < -0.01)),
            'max_beneficial_change': float(np.max(fitness_changes)),
            'max_deleterious_change': float(np.min(fitness_changes))
        }
    
    def find_fitness_peaks(self,
                          sequence_template: torch.Tensor,
                          environmental_features: torch.Tensor,
                          search_radius: int = 5) -> List[Tuple[int, str, float]]:
        """
        Find local fitness peaks through systematic search.
        
        Args:
            sequence_template: Template sequence
            environmental_features: Environmental features
            search_radius: Number of positions around each site to search
            
        Returns:
            List of (position, nucleotide, fitness_score) tuples
        """
        seq_len = sequence_template.shape[1]
        nucleotides = ['A', 'T', 'G', 'C']
        peaks = []
        
        for pos in range(seq_len):
            best_fitness = -float('inf')
            best_nt = None
            
            for nt in nucleotides:
                # Create mutation
                mutated_seq = sequence_template.clone()
                nt_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3}[nt]
                mutated_seq[0, pos, :] = 0
                mutated_seq[0, pos, nt_idx] = 1
                
                # Calculate fitness
                fitness = self.fitness_scorer(mutated_seq, environmental_features)
                fitness_score = fitness['total_fitness'].item()
                
                if fitness_score > best_fitness:
                    best_fitness = fitness_score
                    best_nt = nt
            
            peaks.append((pos, best_nt, best_fitness))
        
        # Sort by fitness score
        peaks.sort(key=lambda x: x[2], reverse=True)
        
        return peaks