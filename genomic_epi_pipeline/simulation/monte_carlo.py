"""
Monte Carlo Simulation Framework for Viral Evolution Prediction.

This module implements parallelized Monte Carlo simulations to generate
distributions of predicted evolutionary trajectories with uncertainty quantification.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import logging
import pickle
import time
from collections import defaultdict

from ..models.probabilistic_engine import ViralEvolutionPredictor


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    num_simulations: int = 10000
    simulation_steps: int = 180  # Days to simulate forward
    step_size_days: int = 7     # Time step resolution
    confidence_levels: List[float] = None
    parallel_workers: int = None
    batch_size: int = 100       # Simulations per batch
    temperature: float = 1.0    # Sampling temperature
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.68, 0.95, 0.99]
        if self.parallel_workers is None:
            self.parallel_workers = min(mp.cpu_count(), 8)


@dataclass
class SimulationTrajectory:
    """Single simulation trajectory result."""
    trajectory_id: int
    initial_sequence: np.ndarray
    sequence_trajectory: np.ndarray  # [time_steps, seq_len, 4]
    fitness_trajectory: np.ndarray   # [time_steps]
    environmental_conditions: np.ndarray  # [time_steps, env_dim]
    mutation_events: List[Dict]      # List of mutation events
    final_fitness: float
    trajectory_metadata: Dict


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation ensemble."""
    config: MonteCarloConfig
    trajectories: List[SimulationTrajectory]
    consensus_sequences: np.ndarray      # [time_steps, seq_len, 4]
    sequence_variance: np.ndarray        # [time_steps, seq_len, 4]
    fitness_statistics: Dict
    mutation_frequencies: Dict
    confidence_intervals: Dict
    convergence_metrics: Dict
    simulation_metadata: Dict


class EvolutionarySimulator:
    """
    Core evolutionary simulator for single trajectory generation.
    
    Simulates viral evolution step-by-step using model predictions
    and environmental conditions.
    """
    
    def __init__(self, 
                 model: ViralEvolutionPredictor,
                 config: MonteCarloConfig):
        """
        Initialize evolutionary simulator.
        
        Args:
            model: Trained viral evolution prediction model
            config: Monte Carlo configuration
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def simulate_trajectory(self,
                          initial_sequence: torch.Tensor,
                          environmental_trajectory: torch.Tensor,
                          trajectory_id: int = 0) -> SimulationTrajectory:
        """
        Simulate a single evolutionary trajectory.
        
        Args:
            initial_sequence: Starting sequence [seq_len, 4]
            environmental_trajectory: Environmental conditions [time_steps, env_dim]
            trajectory_id: Unique identifier for this trajectory
            
        Returns:
            SimulationTrajectory object
        """
        num_steps = len(environmental_trajectory)
        seq_len, num_nucleotides = initial_sequence.shape
        
        # Initialize trajectory storage
        sequence_trajectory = np.zeros((num_steps + 1, seq_len, num_nucleotides))
        fitness_trajectory = np.zeros(num_steps + 1)
        mutation_events = []
        
        # Set initial conditions
        current_sequence = initial_sequence.clone()
        sequence_trajectory[0] = current_sequence.numpy()
        
        # Calculate initial fitness
        with torch.no_grad():
            initial_outputs = self.model(
                current_sequence.unsqueeze(0),
                environmental_trajectory[0:1]
            )
            fitness_trajectory[0] = initial_outputs['fitness_scores'].item()
        
        # Simulate evolution step by step
        for step in range(num_steps):
            env_conditions = environmental_trajectory[step:step+1]
            
            # Generate next sequence
            next_sequence, mutations, fitness = self._simulate_evolution_step(
                current_sequence, env_conditions, step
            )
            
            # Store results
            sequence_trajectory[step + 1] = next_sequence.numpy()
            fitness_trajectory[step + 1] = fitness
            
            # Record mutation events
            for mutation in mutations:
                mutation_events.append({
                    'step': step,
                    'position': mutation['position'],
                    'from_nucleotide': mutation['from'],
                    'to_nucleotide': mutation['to'],
                    'probability': mutation['probability'],
                    'fitness_change': mutation.get('fitness_change', 0.0)
                })
            
            # Update current sequence
            current_sequence = next_sequence
        
        # Create trajectory object
        trajectory = SimulationTrajectory(
            trajectory_id=trajectory_id,
            initial_sequence=initial_sequence.numpy(),
            sequence_trajectory=sequence_trajectory,
            fitness_trajectory=fitness_trajectory,
            environmental_conditions=environmental_trajectory.numpy(),
            mutation_events=mutation_events,
            final_fitness=fitness_trajectory[-1],
            trajectory_metadata={
                'num_mutations': len(mutation_events),
                'fitness_change': fitness_trajectory[-1] - fitness_trajectory[0],
                'simulation_time': num_steps * self.config.step_size_days
            }
        )
        
        return trajectory
    
    def _simulate_evolution_step(self,
                               current_sequence: torch.Tensor,
                               env_conditions: torch.Tensor,
                               step: int) -> Tuple[torch.Tensor, List[Dict], float]:
        """
        Simulate one evolution step.
        
        Args:
            current_sequence: Current sequence state
            env_conditions: Environmental conditions for this step
            step: Current time step
            
        Returns:
            Tuple of (next_sequence, mutation_events, fitness_score)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get model predictions
            outputs = self.model(
                current_sequence.unsqueeze(0),
                env_conditions
            )
            
            mutation_probs = outputs['mutation_probabilities'][0]  # [seq_len, 4]
            fitness_score = outputs['fitness_scores'][0].item()
            
            # Apply temperature scaling
            if self.config.temperature != 1.0:
                mutation_probs = mutation_probs / self.config.temperature
                mutation_probs = F.softmax(mutation_probs, dim=-1)
            
            # Sample mutations
            next_sequence, mutations = self._sample_mutations(
                current_sequence, mutation_probs
            )
        
        return next_sequence, mutations, fitness_score
    
    def _sample_mutations(self,
                         current_sequence: torch.Tensor,
                         mutation_probs: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Sample mutations based on predicted probabilities.
        
        Args:
            current_sequence: Current sequence [seq_len, 4]
            mutation_probs: Mutation probabilities [seq_len, 4]
            
        Returns:
            Tuple of (new_sequence, mutation_events)
        """
        seq_len, _ = current_sequence.shape
        new_sequence = current_sequence.clone()
        mutations = []
        
        # Sample mutations for each position
        for pos in range(seq_len):
            # Get current nucleotide
            current_nt_idx = torch.argmax(current_sequence[pos]).item()
            
            # Sample new nucleotide
            new_nt_idx = torch.multinomial(mutation_probs[pos], 1).item()
            
            # Check if mutation occurred
            if new_nt_idx != current_nt_idx:
                # Record mutation
                nucleotide_map = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
                mutations.append({
                    'position': pos,
                    'from': nucleotide_map[current_nt_idx],
                    'to': nucleotide_map[new_nt_idx],
                    'probability': mutation_probs[pos, new_nt_idx].item()
                })
                
                # Apply mutation
                new_sequence[pos] = 0
                new_sequence[pos, new_nt_idx] = 1
        
        return new_sequence, mutations


class ParallelMonteCarloSimulator:
    """
    Parallelized Monte Carlo simulator for large-scale trajectory generation.
    
    Coordinates multiple evolutionary simulators to generate thousands of
    trajectories efficiently using multiprocessing.
    """
    
    def __init__(self, 
                 model: ViralEvolutionPredictor,
                 config: MonteCarloConfig):
        """
        Initialize parallel Monte Carlo simulator.
        
        Args:
            model: Trained viral evolution prediction model
            config: Monte Carlo configuration
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set random seed for reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
    
    def run_simulation(self,
                      initial_sequence: torch.Tensor,
                      environmental_trajectory: torch.Tensor,
                      save_trajectories: bool = True,
                      output_dir: Optional[str] = None) -> MonteCarloResults:
        """
        Run complete Monte Carlo simulation ensemble.
        
        Args:
            initial_sequence: Starting sequence for all trajectories
            environmental_trajectory: Environmental conditions over time
            save_trajectories: Whether to save individual trajectories
            output_dir: Directory to save results
            
        Returns:
            MonteCarloResults object with ensemble statistics
        """
        self.logger.info(f"Starting Monte Carlo simulation with {self.config.num_simulations} trajectories")
        start_time = time.time()
        
        # Generate all trajectories
        trajectories = self._generate_trajectories_parallel(
            initial_sequence, environmental_trajectory
        )
        
        # Analyze ensemble results
        self.logger.info("Analyzing ensemble results...")
        results = self._analyze_ensemble(trajectories, environmental_trajectory)
        
        # Save results if requested
        if save_trajectories and output_dir:
            self._save_results(results, output_dir)
        
        simulation_time = time.time() - start_time
        self.logger.info(f"Monte Carlo simulation completed in {simulation_time:.2f} seconds")
        
        results.simulation_metadata['total_simulation_time'] = simulation_time
        results.simulation_metadata['trajectories_per_second'] = self.config.num_simulations / simulation_time
        
        return results
    
    def _generate_trajectories_parallel(self,
                                      initial_sequence: torch.Tensor,
                                      environmental_trajectory: torch.Tensor) -> List[SimulationTrajectory]:
        """Generate trajectories using parallel processing."""
        trajectories = []
        
        # Create batches for parallel processing
        batch_size = self.config.batch_size
        num_batches = (self.config.num_simulations + batch_size - 1) // batch_size
        
        self.logger.info(f"Processing {num_batches} batches with {self.config.parallel_workers} workers")
        
        # Use ThreadPoolExecutor for I/O bound tasks or ProcessPoolExecutor for CPU bound
        # Using ThreadPoolExecutor here since we're working with PyTorch tensors
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit batches
            future_to_batch = {}
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, self.config.num_simulations)
                batch_size_actual = end_idx - start_idx
                
                future = executor.submit(
                    self._simulate_batch,
                    initial_sequence,
                    environmental_trajectory,
                    start_idx,
                    batch_size_actual
                )
                future_to_batch[future] = batch_idx
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_trajectories = future.result()
                    trajectories.extend(batch_trajectories)
                    
                    if (batch_idx + 1) % 10 == 0:
                        self.logger.info(f"Completed batch {batch_idx + 1}/{num_batches}")
                        
                except Exception as e:
                    self.logger.error(f"Batch {batch_idx} failed: {e}")
        
        return trajectories
    
    def _simulate_batch(self,
                       initial_sequence: torch.Tensor,
                       environmental_trajectory: torch.Tensor,
                       start_idx: int,
                       batch_size: int) -> List[SimulationTrajectory]:
        """Simulate a batch of trajectories."""
        simulator = EvolutionarySimulator(self.model, self.config)
        batch_trajectories = []
        
        for i in range(batch_size):
            trajectory_id = start_idx + i
            
            # Add small random perturbations to initial sequence for diversity
            perturbed_sequence = initial_sequence.clone()
            if trajectory_id > 0:  # Keep first trajectory unchanged
                # Add small random noise
                noise = torch.randn_like(perturbed_sequence) * 0.01
                perturbed_sequence = F.softmax(perturbed_sequence + noise, dim=-1)
            
            trajectory = simulator.simulate_trajectory(
                perturbed_sequence,
                environmental_trajectory,
                trajectory_id
            )
            batch_trajectories.append(trajectory)
        
        return batch_trajectories
    
    def _analyze_ensemble(self,
                         trajectories: List[SimulationTrajectory],
                         environmental_trajectory: torch.Tensor) -> MonteCarloResults:
        """Analyze ensemble of trajectories to compute statistics."""
        if not trajectories:
            raise ValueError("No trajectories to analyze")
        
        # Extract trajectory data
        num_trajectories = len(trajectories)
        num_steps = trajectories[0].sequence_trajectory.shape[0]
        seq_len = trajectories[0].sequence_trajectory.shape[1]
        
        # Stack all sequence trajectories
        all_sequences = np.stack([t.sequence_trajectory for t in trajectories])
        all_fitness = np.stack([t.fitness_trajectory for t in trajectories])
        
        # Calculate consensus sequences (mean across trajectories)
        consensus_sequences = np.mean(all_sequences, axis=0)
        
        # Calculate sequence variance
        sequence_variance = np.var(all_sequences, axis=0)
        
        # Fitness statistics
        fitness_statistics = {
            'mean_trajectory': np.mean(all_fitness, axis=0),
            'std_trajectory': np.std(all_fitness, axis=0),
            'median_trajectory': np.median(all_fitness, axis=0),
            'final_fitness_mean': np.mean([t.final_fitness for t in trajectories]),
            'final_fitness_std': np.std([t.final_fitness for t in trajectories]),
            'fitness_change_mean': np.mean([t.trajectory_metadata['fitness_change'] for t in trajectories]),
            'fitness_change_std': np.std([t.trajectory_metadata['fitness_change'] for t in trajectories])
        }
        
        # Mutation frequency analysis
        mutation_frequencies = self._analyze_mutation_frequencies(trajectories)
        
        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            all_sequences, all_fitness
        )
        
        # Convergence metrics
        convergence_metrics = self._assess_convergence(trajectories)
        
        # Simulation metadata
        simulation_metadata = {
            'num_trajectories': num_trajectories,
            'num_steps': num_steps,
            'sequence_length': seq_len,
            'total_mutations': sum(len(t.mutation_events) for t in trajectories),
            'avg_mutations_per_trajectory': np.mean([len(t.mutation_events) for t in trajectories]),
            'config': self.config
        }
        
        return MonteCarloResults(
            config=self.config,
            trajectories=trajectories,
            consensus_sequences=consensus_sequences,
            sequence_variance=sequence_variance,
            fitness_statistics=fitness_statistics,
            mutation_frequencies=mutation_frequencies,
            confidence_intervals=confidence_intervals,
            convergence_metrics=convergence_metrics,
            simulation_metadata=simulation_metadata
        )
    
    def _analyze_mutation_frequencies(self, trajectories: List[SimulationTrajectory]) -> Dict:
        """Analyze mutation frequencies across all trajectories."""
        position_mutations = defaultdict(lambda: defaultdict(int))
        nucleotide_transitions = defaultdict(int)
        
        for trajectory in trajectories:
            for mutation in trajectory.mutation_events:
                pos = mutation['position']
                from_nt = mutation['from_nucleotide']
                to_nt = mutation['to_nucleotide']
                
                # Count mutations at each position
                position_mutations[pos][f"{from_nt}>{to_nt}"] += 1
                
                # Count nucleotide transitions
                transition = f"{from_nt}>{to_nt}"
                nucleotide_transitions[transition] += 1
        
        # Convert to frequencies
        total_trajectories = len(trajectories)
        
        mutation_frequencies = {
            'position_frequencies': {
                pos: {mut: count / total_trajectories 
                     for mut, count in mutations.items()}
                for pos, mutations in position_mutations.items()
            },
            'transition_frequencies': {
                transition: count / total_trajectories
                for transition, count in nucleotide_transitions.items()
            },
            'hotspot_positions': self._identify_hotspots(position_mutations, total_trajectories)
        }
        
        return mutation_frequencies
    
    def _identify_hotspots(self, position_mutations: Dict, total_trajectories: int) -> List[Dict]:
        """Identify mutation hotspots."""
        hotspots = []
        
        for pos, mutations in position_mutations.items():
            total_mutations_at_pos = sum(mutations.values())
            mutation_frequency = total_mutations_at_pos / total_trajectories
            
            if mutation_frequency > 0.1:  # Threshold for hotspot
                hotspots.append({
                    'position': pos,
                    'mutation_frequency': mutation_frequency,
                    'most_common_mutation': max(mutations.items(), key=lambda x: x[1]),
                    'total_mutations': total_mutations_at_pos
                })
        
        # Sort by mutation frequency
        hotspots.sort(key=lambda x: x['mutation_frequency'], reverse=True)
        
        return hotspots
    
    def _calculate_confidence_intervals(self,
                                      all_sequences: np.ndarray,
                                      all_fitness: np.ndarray) -> Dict:
        """Calculate confidence intervals for predictions."""
        confidence_intervals = {}
        
        for confidence_level in self.config.confidence_levels:
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            # Sequence confidence intervals
            seq_ci_lower = np.percentile(all_sequences, lower_percentile, axis=0)
            seq_ci_upper = np.percentile(all_sequences, upper_percentile, axis=0)
            
            # Fitness confidence intervals
            fitness_ci_lower = np.percentile(all_fitness, lower_percentile, axis=0)
            fitness_ci_upper = np.percentile(all_fitness, upper_percentile, axis=0)
            
            confidence_intervals[f'{confidence_level:.0%}'] = {
                'sequence_lower': seq_ci_lower,
                'sequence_upper': seq_ci_upper,
                'fitness_lower': fitness_ci_lower,
                'fitness_upper': fitness_ci_upper
            }
        
        return confidence_intervals
    
    def _assess_convergence(self, trajectories: List[SimulationTrajectory]) -> Dict:
        """Assess convergence of Monte Carlo simulation."""
        # Simple convergence assessment based on running statistics
        num_trajectories = len(trajectories)
        
        # Calculate running mean of final fitness
        final_fitness_values = [t.final_fitness for t in trajectories]
        running_means = []
        
        for i in range(1, num_trajectories + 1):
            running_mean = np.mean(final_fitness_values[:i])
            running_means.append(running_mean)
        
        # Calculate convergence metrics
        final_mean = running_means[-1]
        convergence_threshold = 0.01  # 1% change threshold
        
        # Find when running mean stabilized
        converged_at = num_trajectories
        for i in range(100, num_trajectories):  # Start checking after 100 trajectories
            if abs(running_means[i] - final_mean) / abs(final_mean) < convergence_threshold:
                converged_at = i
                break
        
        convergence_metrics = {
            'converged_at_trajectory': converged_at,
            'final_mean_fitness': final_mean,
            'convergence_achieved': converged_at < num_trajectories * 0.8,
            'running_means': running_means,
            'effective_sample_size': min(converged_at * 2, num_trajectories)
        }
        
        return convergence_metrics
    
    def _save_results(self, results: MonteCarloResults, output_dir: str):
        """Save Monte Carlo results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete results object
        with open(output_path / "monte_carlo_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary statistics as CSV
        summary_stats = pd.DataFrame([{
            'num_trajectories': results.simulation_metadata['num_trajectories'],
            'final_fitness_mean': results.fitness_statistics['final_fitness_mean'],
            'final_fitness_std': results.fitness_statistics['final_fitness_std'],
            'total_mutations': results.simulation_metadata['total_mutations'],
            'avg_mutations_per_trajectory': results.simulation_metadata['avg_mutations_per_trajectory'],
            'simulation_time': results.simulation_metadata.get('total_simulation_time', 0)
        }])
        summary_stats.to_csv(output_path / "simulation_summary.csv", index=False)
        
        # Save consensus sequences
        np.save(output_path / "consensus_sequences.npy", results.consensus_sequences)
        np.save(output_path / "sequence_variance.npy", results.sequence_variance)
        
        # Save mutation frequencies
        with open(output_path / "mutation_frequencies.pkl", 'wb') as f:
            pickle.dump(results.mutation_frequencies, f)
        
        self.logger.info(f"Monte Carlo results saved to {output_path}")


# Convenience function for running Monte Carlo simulations
def run_monte_carlo_simulation(model: ViralEvolutionPredictor,
                             initial_sequence: torch.Tensor,
                             environmental_trajectory: torch.Tensor,
                             config: Optional[MonteCarloConfig] = None,
                             output_dir: Optional[str] = None) -> MonteCarloResults:
    """
    Convenience function to run Monte Carlo simulation.
    
    Args:
        model: Trained viral evolution prediction model
        initial_sequence: Starting sequence for simulation
        environmental_trajectory: Environmental conditions over time
        config: Monte Carlo configuration
        output_dir: Directory to save results
        
    Returns:
        MonteCarloResults object
    """
    if config is None:
        config = MonteCarloConfig()
    
    simulator = ParallelMonteCarloSimulator(model, config)
    
    return simulator.run_simulation(
        initial_sequence=initial_sequence,
        environmental_trajectory=environmental_trajectory,
        save_trajectories=True,
        output_dir=output_dir
    )