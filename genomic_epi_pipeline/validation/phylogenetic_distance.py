"""
Phylogenetic Validation Engine.

This module implements phylogenetic tree reconstruction and validation
using Robinson-Foulds distance metrics for evolutionary prediction accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass
from pathlib import Path
import logging
from collections import defaultdict
import itertools
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, to_tree
import warnings

# Try to import Bio modules, with fallbacks
try:
    from Bio import Phylo
    from Bio.Phylo import TreeConstruction
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
    from Bio import AlignIO
    from Bio.Align import MultipleSeqAlignment
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    warnings.warn("Biopython not available. Using simplified tree construction.")


@dataclass
class PhylogeneticValidationConfig:
    """Configuration for phylogenetic validation."""
    tree_method: str = "nj"  # neighbor_joining, upgma, maximum_likelihood
    distance_metric: str = "hamming"  # hamming, jukes_cantor, kimura
    bootstrap_replicates: int = 100
    confidence_threshold: float = 0.7
    normalize_rf_distance: bool = True
    save_trees: bool = True
    tree_format: str = "newick"


@dataclass
class TreeComparisonResult:
    """Result of comparing two phylogenetic trees."""
    rf_distance: float
    normalized_rf_distance: float
    max_possible_rf: int
    symmetric_difference: int
    common_splits: int
    total_splits_tree1: int
    total_splits_tree2: int
    topological_similarity: float
    comparison_metadata: Dict


@dataclass
class PhylogeneticValidationResult:
    """Complete phylogenetic validation result."""
    predicted_tree: Optional[object]
    actual_tree: Optional[object]
    tree_comparison: TreeComparisonResult
    sequence_distances: np.ndarray
    bootstrap_support: Optional[Dict]
    validation_metrics: Dict
    tree_files: Dict[str, str]


class SequenceDistanceCalculator:
    """
    Calculate pairwise distances between sequences for tree construction.
    
    Supports multiple distance metrics commonly used in phylogenetics.
    """
    
    def __init__(self, metric: str = "hamming"):
        """
        Initialize distance calculator.
        
        Args:
            metric: Distance metric ('hamming', 'jukes_cantor', 'kimura')
        """
        self.metric = metric
        self.logger = logging.getLogger(__name__)
    
    def calculate_distance_matrix(self, sequences: List[str]) -> np.ndarray:
        """
        Calculate pairwise distance matrix between sequences.
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            Symmetric distance matrix
        """
        n_sequences = len(sequences)
        distance_matrix = np.zeros((n_sequences, n_sequences))
        
        for i in range(n_sequences):
            for j in range(i + 1, n_sequences):
                dist = self._calculate_pairwise_distance(sequences[i], sequences[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def _calculate_pairwise_distance(self, seq1: str, seq2: str) -> float:
        """Calculate distance between two sequences."""
        if len(seq1) != len(seq2):
            raise ValueError("Sequences must have equal length")
        
        if self.metric == "hamming":
            return self._hamming_distance(seq1, seq2)
        elif self.metric == "jukes_cantor":
            return self._jukes_cantor_distance(seq1, seq2)
        elif self.metric == "kimura":
            return self._kimura_distance(seq1, seq2)
        else:
            raise ValueError(f"Unknown distance metric: {self.metric}")
    
    def _hamming_distance(self, seq1: str, seq2: str) -> float:
        """Calculate Hamming distance (proportion of differing sites)."""
        differences = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
        return differences / len(seq1)
    
    def _jukes_cantor_distance(self, seq1: str, seq2: str) -> float:
        """Calculate Jukes-Cantor corrected distance."""
        p = self._hamming_distance(seq1, seq2)
        
        # Avoid division by zero and log of negative numbers
        if p >= 0.75:
            return float('inf')  # Sequences too divergent
        
        try:
            return -0.75 * np.log(1 - (4/3) * p)
        except (ValueError, ZeroDivisionError):
            return float('inf')
    
    def _kimura_distance(self, seq1: str, seq2: str) -> float:
        """Calculate Kimura 2-parameter distance."""
        transitions = 0  # A<->G, C<->T
        transversions = 0  # All other changes
        
        transition_pairs = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        
        for c1, c2 in zip(seq1, seq2):
            if c1 != c2:
                if (c1, c2) in transition_pairs:
                    transitions += 1
                else:
                    transversions += 1
        
        length = len(seq1)
        P = transitions / length  # Transition proportion
        Q = transversions / length  # Transversion proportion
        
        # Avoid mathematical errors
        if 1 - 2*P - Q <= 0 or 1 - 2*Q <= 0:
            return float('inf')
        
        try:
            return -0.5 * np.log((1 - 2*P - Q) * np.sqrt(1 - 2*Q))
        except (ValueError, ZeroDivisionError):
            return float('inf')


class SimpleTreeConstructor:
    """
    Simple tree constructor using hierarchical clustering.
    
    Fallback implementation when Biopython is not available.
    """
    
    def __init__(self, method: str = "nj"):
        """
        Initialize tree constructor.
        
        Args:
            method: Tree construction method ('nj', 'upgma')
        """
        self.method = method
        self.logger = logging.getLogger(__name__)
    
    def construct_tree(self, distance_matrix: np.ndarray, 
                      sequence_names: List[str]) -> Dict:
        """
        Construct phylogenetic tree from distance matrix.
        
        Args:
            distance_matrix: Pairwise distance matrix
            sequence_names: Names of sequences
            
        Returns:
            Dictionary representing the tree structure
        """
        if self.method == "nj":
            return self._neighbor_joining(distance_matrix, sequence_names)
        elif self.method == "upgma":
            return self._upgma(distance_matrix, sequence_names)
        else:
            raise ValueError(f"Unknown tree construction method: {self.method}")
    
    def _neighbor_joining(self, distance_matrix: np.ndarray, 
                         sequence_names: List[str]) -> Dict:
        """Simplified neighbor-joining algorithm."""
        n = len(sequence_names)
        
        if n < 3:
            raise ValueError("Need at least 3 sequences for tree construction")
        
        # Use scipy's linkage for simplicity (not true NJ but similar)
        # Convert distance matrix to condensed form
        condensed_distances = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distances, method='average')
        
        # Convert to tree structure
        tree_dict = self._linkage_to_tree_dict(linkage_matrix, sequence_names)
        
        return tree_dict
    
    def _upgma(self, distance_matrix: np.ndarray, 
              sequence_names: List[str]) -> Dict:
        """UPGMA tree construction using hierarchical clustering."""
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method='average')
        tree_dict = self._linkage_to_tree_dict(linkage_matrix, sequence_names)
        return tree_dict
    
    def _linkage_to_tree_dict(self, linkage_matrix: np.ndarray, 
                             sequence_names: List[str]) -> Dict:
        """Convert scipy linkage matrix to tree dictionary."""
        n = len(sequence_names)
        
        # Create tree structure
        tree = {
            'method': self.method,
            'n_sequences': n,
            'sequence_names': sequence_names,
            'linkage_matrix': linkage_matrix.tolist(),
            'splits': self._extract_splits_from_linkage(linkage_matrix, sequence_names)
        }
        
        return tree
    
    def _extract_splits_from_linkage(self, linkage_matrix: np.ndarray, 
                                   sequence_names: List[str]) -> List[Tuple[Set[str], Set[str]]]:
        """Extract splits (bipartitions) from linkage matrix."""
        n = len(sequence_names)
        splits = []
        
        # Each row in linkage matrix represents a merge
        for i, row in enumerate(linkage_matrix):
            left_idx, right_idx = int(row[0]), int(row[1])
            
            # Get the sequences in each cluster
            left_seqs = self._get_cluster_sequences(left_idx, i, sequence_names, linkage_matrix)
            right_seqs = self._get_cluster_sequences(right_idx, i, sequence_names, linkage_matrix)
            
            # Create split (bipartition)
            all_seqs = set(sequence_names)
            split = (left_seqs, all_seqs - left_seqs)
            splits.append(split)
        
        return splits
    
    def _get_cluster_sequences(self, cluster_idx: int, merge_step: int,
                              sequence_names: List[str], 
                              linkage_matrix: np.ndarray) -> Set[str]:
        """Get sequences in a cluster at a given merge step."""
        n = len(sequence_names)
        
        if cluster_idx < n:
            # Original sequence
            return {sequence_names[cluster_idx]}
        else:
            # Merged cluster - recursively get sequences
            merge_idx = cluster_idx - n
            if merge_idx >= merge_step:
                # This shouldn't happen in proper linkage
                return set()
            
            left_idx, right_idx = int(linkage_matrix[merge_idx][0]), int(linkage_matrix[merge_idx][1])
            left_seqs = self._get_cluster_sequences(left_idx, merge_idx, sequence_names, linkage_matrix)
            right_seqs = self._get_cluster_sequences(right_idx, merge_idx, sequence_names, linkage_matrix)
            
            return left_seqs | right_seqs


class RobinsonFouldsCalculator:
    """
    Calculate Robinson-Foulds distance between phylogenetic trees.
    
    The RF distance measures the topological difference between trees
    by counting the number of splits (bipartitions) that differ.
    """
    
    def __init__(self, normalize: bool = True):
        """
        Initialize RF calculator.
        
        Args:
            normalize: Whether to normalize RF distance by maximum possible
        """
        self.normalize = normalize
        self.logger = logging.getLogger(__name__)
    
    def calculate_rf_distance(self, tree1: Dict, tree2: Dict) -> TreeComparisonResult:
        """
        Calculate Robinson-Foulds distance between two trees.
        
        Args:
            tree1: First phylogenetic tree
            tree2: Second phylogenetic tree
            
        Returns:
            TreeComparisonResult with distance metrics
        """
        # Extract splits from both trees
        splits1 = self._extract_splits(tree1)
        splits2 = self._extract_splits(tree2)
        
        # Calculate RF distance
        rf_distance, comparison_stats = self._compute_rf_distance(splits1, splits2)
        
        # Calculate maximum possible RF distance
        n_taxa = len(tree1.get('sequence_names', []))
        max_rf = 2 * (n_taxa - 3) if n_taxa >= 3 else 0
        
        # Normalize if requested
        normalized_rf = rf_distance / max_rf if max_rf > 0 and self.normalize else rf_distance
        
        # Calculate topological similarity
        topological_similarity = 1 - normalized_rf if self.normalize else None
        
        result = TreeComparisonResult(
            rf_distance=rf_distance,
            normalized_rf_distance=normalized_rf,
            max_possible_rf=max_rf,
            symmetric_difference=comparison_stats['symmetric_difference'],
            common_splits=comparison_stats['common_splits'],
            total_splits_tree1=len(splits1),
            total_splits_tree2=len(splits2),
            topological_similarity=topological_similarity,
            comparison_metadata={
                'n_taxa': n_taxa,
                'normalization_applied': self.normalize,
                'tree1_method': tree1.get('method', 'unknown'),
                'tree2_method': tree2.get('method', 'unknown')
            }
        )
        
        return result
    
    def _extract_splits(self, tree: Dict) -> Set[Tuple[frozenset, frozenset]]:
        """Extract splits (bipartitions) from tree."""
        if 'splits' in tree:
            # Splits already computed
            splits = set()
            for split in tree['splits']:
                if len(split) == 2:
                    # Convert to frozensets for hashing
                    part1, part2 = frozenset(split[0]), frozenset(split[1])
                    # Ensure consistent ordering (smaller set first)
                    if len(part1) <= len(part2):
                        splits.add((part1, part2))
                    else:
                        splits.add((part2, part1))
            return splits
        else:
            # Need to extract splits from tree structure
            # This is a simplified version - real implementation would traverse tree
            return set()
    
    def _compute_rf_distance(self, splits1: Set, splits2: Set) -> Tuple[int, Dict]:
        """Compute RF distance between two sets of splits."""
        # Find splits unique to each tree
        unique_to_tree1 = splits1 - splits2
        unique_to_tree2 = splits2 - splits1
        
        # RF distance is the number of splits that differ
        rf_distance = len(unique_to_tree1) + len(unique_to_tree2)
        
        # Common splits
        common_splits = len(splits1 & splits2)
        
        comparison_stats = {
            'symmetric_difference': rf_distance,
            'common_splits': common_splits,
            'unique_to_tree1': len(unique_to_tree1),
            'unique_to_tree2': len(unique_to_tree2)
        }
        
        return rf_distance, comparison_stats


class PhylogeneticValidator:
    """
    Main phylogenetic validation engine.
    
    Coordinates tree construction, comparison, and validation metrics.
    """
    
    def __init__(self, config: PhylogeneticValidationConfig):
        """
        Initialize phylogenetic validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.distance_calculator = SequenceDistanceCalculator(config.distance_metric)
        self.tree_constructor = SimpleTreeConstructor(config.tree_method)
        self.rf_calculator = RobinsonFouldsCalculator(config.normalize_rf_distance)
        
        self.logger = logging.getLogger(__name__)
    
    def validate_predictions(self,
                           predicted_sequences: List[str],
                           actual_sequences: List[str],
                           sequence_names: Optional[List[str]] = None,
                           output_dir: Optional[str] = None) -> PhylogeneticValidationResult:
        """
        Validate predicted sequences against actual sequences using phylogenetic analysis.
        
        Args:
            predicted_sequences: List of predicted DNA sequences
            actual_sequences: List of actual DNA sequences
            sequence_names: Names for sequences (optional)
            output_dir: Directory to save results
            
        Returns:
            PhylogeneticValidationResult object
        """
        self.logger.info("Starting phylogenetic validation...")
        
        # Validate inputs
        if len(predicted_sequences) != len(actual_sequences):
            raise ValueError("Number of predicted and actual sequences must match")
        
        if len(predicted_sequences) < 3:
            raise ValueError("Need at least 3 sequences for phylogenetic analysis")
        
        # Generate sequence names if not provided
        if sequence_names is None:
            sequence_names = [f"seq_{i}" for i in range(len(predicted_sequences))]
        
        # Construct phylogenetic trees
        self.logger.info("Constructing phylogenetic trees...")
        predicted_tree = self._construct_tree(predicted_sequences, 
                                            [f"pred_{name}" for name in sequence_names])
        actual_tree = self._construct_tree(actual_sequences,
                                         [f"actual_{name}" for name in sequence_names])
        
        # Calculate tree comparison
        self.logger.info("Calculating Robinson-Foulds distance...")
        tree_comparison = self.rf_calculator.calculate_rf_distance(predicted_tree, actual_tree)
        
        # Calculate sequence distance matrices
        pred_distances = self.distance_calculator.calculate_distance_matrix(predicted_sequences)
        actual_distances = self.distance_calculator.calculate_distance_matrix(actual_sequences)
        
        # Additional validation metrics
        validation_metrics = self._calculate_validation_metrics(
            pred_distances, actual_distances, tree_comparison
        )
        
        # Save trees if requested
        tree_files = {}
        if self.config.save_trees and output_dir:
            tree_files = self._save_trees(predicted_tree, actual_tree, output_dir)
        
        result = PhylogeneticValidationResult(
            predicted_tree=predicted_tree,
            actual_tree=actual_tree,
            tree_comparison=tree_comparison,
            sequence_distances=np.stack([pred_distances, actual_distances]),
            bootstrap_support=None,  # Could be implemented
            validation_metrics=validation_metrics,
            tree_files=tree_files
        )
        
        self.logger.info(f"Phylogenetic validation completed. RF distance: {tree_comparison.rf_distance}")
        
        return result
    
    def _construct_tree(self, sequences: List[str], names: List[str]) -> Dict:
        """Construct phylogenetic tree from sequences."""
        # Calculate distance matrix
        distance_matrix = self.distance_calculator.calculate_distance_matrix(sequences)
        
        # Construct tree
        tree = self.tree_constructor.construct_tree(distance_matrix, names)
        
        return tree
    
    def _calculate_validation_metrics(self,
                                    pred_distances: np.ndarray,
                                    actual_distances: np.ndarray,
                                    tree_comparison: TreeComparisonResult) -> Dict:
        """Calculate additional validation metrics."""
        # Distance matrix correlation
        pred_flat = pred_distances[np.triu_indices_from(pred_distances, k=1)]
        actual_flat = actual_distances[np.triu_indices_from(actual_distances, k=1)]
        
        distance_correlation = np.corrcoef(pred_flat, actual_flat)[0, 1]
        
        # Distance matrix MSE
        distance_mse = np.mean((pred_distances - actual_distances) ** 2)
        
        # Topological accuracy metrics
        topological_accuracy = tree_comparison.topological_similarity or 0
        
        validation_metrics = {
            'rf_distance': tree_comparison.rf_distance,
            'normalized_rf_distance': tree_comparison.normalized_rf_distance,
            'topological_accuracy': topological_accuracy,
            'distance_correlation': distance_correlation,
            'distance_mse': distance_mse,
            'common_splits_ratio': (tree_comparison.common_splits / 
                                  max(tree_comparison.total_splits_tree1, 1)),
            'tree_similarity_score': 1 - tree_comparison.normalized_rf_distance
        }
        
        return validation_metrics
    
    def _save_trees(self, predicted_tree: Dict, actual_tree: Dict, 
                   output_dir: str) -> Dict[str, str]:
        """Save trees to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        tree_files = {}
        
        # Save tree structures as JSON (since we're using dict representation)
        import json
        
        pred_file = output_path / "predicted_tree.json"
        with open(pred_file, 'w') as f:
            json.dump(predicted_tree, f, indent=2, default=str)
        tree_files['predicted_tree'] = str(pred_file)
        
        actual_file = output_path / "actual_tree.json"
        with open(actual_file, 'w') as f:
            json.dump(actual_tree, f, indent=2, default=str)
        tree_files['actual_tree'] = str(actual_file)
        
        return tree_files


def validate_evolutionary_predictions(predicted_sequences: List[str],
                                    actual_sequences: List[str],
                                    config: Optional[PhylogeneticValidationConfig] = None,
                                    sequence_names: Optional[List[str]] = None,
                                    output_dir: Optional[str] = None) -> PhylogeneticValidationResult:
    """
    Convenience function for phylogenetic validation.
    
    Args:
        predicted_sequences: Predicted DNA sequences
        actual_sequences: Actual DNA sequences
        config: Validation configuration
        sequence_names: Sequence names
        output_dir: Output directory
        
    Returns:
        PhylogeneticValidationResult
    """
    if config is None:
        config = PhylogeneticValidationConfig()
    
    validator = PhylogeneticValidator(config)
    
    return validator.validate_predictions(
        predicted_sequences=predicted_sequences,
        actual_sequences=actual_sequences,
        sequence_names=sequence_names,
        output_dir=output_dir
    )


# Example usage
if __name__ == "__main__":
    # Test with mock sequences
    predicted_seqs = [
        "ATCGATCGATCGATCG",
        "ATCGATCGATCGATCG",
        "ATCGATCGATCGATCG",
        "ATCGATCGATCGATCG"
    ]
    
    actual_seqs = [
        "ATCGATCGATCGATCG",
        "ATCGATCGATCGATCG", 
        "ATCGATCGATCGATCG",
        "ATCGATCGATCGATCG"
    ]
    
    # Add some mutations to make sequences different
    predicted_seqs[1] = "ATCGATCGATCGATCG".replace("A", "G", 1)
    predicted_seqs[2] = "ATCGATCGATCGATCG".replace("T", "C", 2)
    
    actual_seqs[1] = "ATCGATCGATCGATCG".replace("A", "T", 1)
    actual_seqs[2] = "ATCGATCGATCGATCG".replace("G", "A", 1)
    
    print("Testing phylogenetic validation with mock sequences...")
    
    try:
        result = validate_evolutionary_predictions(
            predicted_sequences=predicted_seqs,
            actual_sequences=actual_seqs,
            sequence_names=["seq1", "seq2", "seq3", "seq4"]
        )
        
        print(f"RF Distance: {result.tree_comparison.rf_distance}")
        print(f"Normalized RF Distance: {result.tree_comparison.normalized_rf_distance:.3f}")
        print(f"Topological Similarity: {result.tree_comparison.topological_similarity:.3f}")
        print("Phylogenetic validation test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")