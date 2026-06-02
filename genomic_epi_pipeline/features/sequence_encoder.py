"""
Sequence Encoder for Viral Genomic Data.

This module provides various encoding methods to convert viral sequences
into numerical representations suitable for machine learning models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from Bio.Seq import Seq
import torch
import torch.nn as nn


class SequenceEncoder:
    """
    Encodes viral sequences into numerical matrices using various methods.
    
    Supports one-hot encoding, k-mer counting, and learned embeddings.
    """
    
    def __init__(self, 
                 encoding_method: str = "one_hot",
                 kmer_size: int = 3,
                 window_size: int = 100,
                 overlap: int = 50,
                 vocab_size: Optional[int] = None):
        """
        Initialize the sequence encoder.
        
        Args:
            encoding_method: Method for encoding ('one_hot', 'kmer', 'embedding')
            kmer_size: Size of k-mers for k-mer encoding
            window_size: Size of sliding windows
            overlap: Overlap between consecutive windows
            vocab_size: Vocabulary size for embedding methods
        """
        self.encoding_method = encoding_method
        self.kmer_size = kmer_size
        self.window_size = window_size
        self.overlap = overlap
        self.vocab_size = vocab_size
        
        # Nucleotide mapping
        self.nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4, '-': 5}
        self.reverse_map = {v: k for k, v in self.nucleotide_map.items()}
        
        # Initialize encoding components
        self.kmer_vectorizer = None
        self.embedding_layer = None
        self.fitted = False
    
    def _create_sliding_windows(self, sequence: str) -> List[str]:
        """
        Create sliding windows from a sequence.
        
        Args:
            sequence: Input DNA sequence
            
        Returns:
            List of sequence windows
        """
        windows = []
        step = self.window_size - self.overlap
        
        for i in range(0, len(sequence) - self.window_size + 1, step):
            window = sequence[i:i + self.window_size]
            windows.append(window)
        
        return windows
    
    def _one_hot_encode_sequence(self, sequence: str) -> np.ndarray:
        """
        One-hot encode a single sequence.
        
        Args:
            sequence: Input DNA sequence
            
        Returns:
            One-hot encoded matrix (length x 6)
        """
        # Convert to uppercase and handle unknown nucleotides
        sequence = sequence.upper()
        sequence = ''.join([nt if nt in self.nucleotide_map else 'N' for nt in sequence])
        
        # Create one-hot matrix
        encoded = np.zeros((len(sequence), len(self.nucleotide_map)))
        
        for i, nt in enumerate(sequence):
            encoded[i, self.nucleotide_map[nt]] = 1
        
        return encoded
    
    def _generate_kmers(self, sequence: str) -> List[str]:
        """
        Generate k-mers from a sequence.
        
        Args:
            sequence: Input DNA sequence
            
        Returns:
            List of k-mers
        """
        sequence = sequence.upper()
        kmers = []
        
        for i in range(len(sequence) - self.kmer_size + 1):
            kmer = sequence[i:i + self.kmer_size]
            # Only include k-mers without ambiguous nucleotides
            if all(nt in ['A', 'T', 'G', 'C'] for nt in kmer):
                kmers.append(kmer)
        
        return kmers
    
    def _kmer_encode_sequence(self, sequence: str) -> np.ndarray:
        """
        K-mer encode a single sequence.
        
        Args:
            sequence: Input DNA sequence
            
        Returns:
            K-mer count vector
        """
        if self.kmer_vectorizer is None:
            raise ValueError("K-mer vectorizer not fitted. Call fit() first.")
        
        kmers = self._generate_kmers(sequence)
        kmer_string = ' '.join(kmers)
        
        return self.kmer_vectorizer.transform([kmer_string]).toarray()[0]
    
    def _calculate_sequence_features(self, sequence: str) -> Dict[str, float]:
        """
        Calculate additional sequence features.
        
        Args:
            sequence: Input DNA sequence
            
        Returns:
            Dictionary of sequence features
        """
        sequence = sequence.upper().replace('-', '').replace('N', '')
        
        if len(sequence) == 0:
            return {
                'gc_content': 0.0,
                'length': 0,
                'purine_content': 0.0,
                'pyrimidine_content': 0.0,
                'complexity': 0.0
            }
        
        # Basic composition features
        gc_content = (sequence.count("G") + sequence.count("C")) / len(sequence) if len(sequence) > 0 else 0.0
        length = len(sequence)
        
        # Purine (A, G) and Pyrimidine (T, C) content
        purine_count = sequence.count('A') + sequence.count('G')
        pyrimidine_count = sequence.count('T') + sequence.count('C')
        purine_content = purine_count / length if length > 0 else 0
        pyrimidine_content = pyrimidine_count / length if length > 0 else 0
        
        # Sequence complexity (Shannon entropy)
        nt_counts = Counter(sequence)
        total = sum(nt_counts.values())
        complexity = -sum((count/total) * np.log2(count/total) 
                         for count in nt_counts.values() if count > 0)
        
        return {
            'gc_content': gc_content,
            'length': length,
            'purine_content': purine_content,
            'pyrimidine_content': pyrimidine_content,
            'complexity': complexity
        }
    
    def fit(self, sequences: List[str]) -> 'SequenceEncoder':
        """
        Fit the encoder on a collection of sequences.
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            Self for method chaining
        """
        if self.encoding_method == "kmer":
            # Fit k-mer vectorizer
            all_kmers = []
            for seq in sequences:
                kmers = self._generate_kmers(seq)
                all_kmers.append(' '.join(kmers))
            
            self.kmer_vectorizer = CountVectorizer(
                token_pattern=r'\b\w+\b',
                max_features=self.vocab_size
            )
            self.kmer_vectorizer.fit(all_kmers)
        
        elif self.encoding_method == "embedding":
            # Initialize embedding layer
            vocab_size = self.vocab_size or len(self.nucleotide_map)
            self.embedding_layer = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=64
            )
        
        self.fitted = True
        return self
    
    def encode_sequence(self, sequence: str) -> Dict[str, np.ndarray]:
        """
        Encode a single sequence using the specified method.
        
        Args:
            sequence: Input DNA sequence
            
        Returns:
            Dictionary containing encoded sequence and features
        """
        if not self.fitted and self.encoding_method in ["kmer", "embedding"]:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        result = {}
        
        # Create sliding windows
        windows = self._create_sliding_windows(sequence)
        
        if self.encoding_method == "one_hot":
            # One-hot encode each window
            encoded_windows = []
            for window in windows:
                encoded = self._one_hot_encode_sequence(window)
                encoded_windows.append(encoded)
            
            result['encoded_sequence'] = np.array(encoded_windows)
        
        elif self.encoding_method == "kmer":
            # K-mer encode each window
            encoded_windows = []
            for window in windows:
                encoded = self._kmer_encode_sequence(window)
                encoded_windows.append(encoded)
            
            result['encoded_sequence'] = np.array(encoded_windows)
        
        elif self.encoding_method == "embedding":
            # Convert to indices for embedding
            sequence_indices = []
            for nt in sequence.upper():
                idx = self.nucleotide_map.get(nt, self.nucleotide_map['N'])
                sequence_indices.append(idx)
            
            result['sequence_indices'] = np.array(sequence_indices)
        
        # Calculate additional features
        result['sequence_features'] = self._calculate_sequence_features(sequence)
        result['num_windows'] = len(windows)
        
        return result
    
    def encode_batch(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """
        Encode a batch of sequences.
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            Dictionary containing batch-encoded sequences and features
        """
        batch_results = {
            'encoded_sequences': [],
            'sequence_features': [],
            'sequence_indices': [],
            'num_windows': []
        }
        
        for sequence in sequences:
            result = self.encode_sequence(sequence)
            
            if 'encoded_sequence' in result:
                batch_results['encoded_sequences'].append(result['encoded_sequence'])
            
            if 'sequence_indices' in result:
                batch_results['sequence_indices'].append(result['sequence_indices'])
            
            batch_results['sequence_features'].append(result['sequence_features'])
            batch_results['num_windows'].append(result['num_windows'])
        
        # Convert to numpy arrays where appropriate
        if batch_results['encoded_sequences']:
            # Pad sequences to same length for batching
            max_windows = max(len(seq) for seq in batch_results['encoded_sequences'])
            max_length = max(seq.shape[1] for seq in batch_results['encoded_sequences'])
            
            padded_sequences = []
            for seq in batch_results['encoded_sequences']:
                # Pad windows dimension
                if len(seq) < max_windows:
                    padding = np.zeros((max_windows - len(seq), seq.shape[1], seq.shape[2]))
                    seq = np.concatenate([seq, padding], axis=0)
                
                padded_sequences.append(seq)
            
            batch_results['encoded_sequences'] = np.array(padded_sequences)
        
        # Convert sequence features to structured array
        if batch_results['sequence_features']:
            feature_df = pd.DataFrame(batch_results['sequence_features'])
            batch_results['sequence_features'] = feature_df.values
        
        return batch_results
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names for the encoded representation.
        
        Returns:
            List of feature names
        """
        if self.encoding_method == "one_hot":
            return [f"nt_{nt}" for nt in self.nucleotide_map.keys()]
        
        elif self.encoding_method == "kmer" and self.kmer_vectorizer:
            return self.kmer_vectorizer.get_feature_names_out().tolist()
        
        elif self.encoding_method == "embedding":
            return [f"emb_dim_{i}" for i in range(64)]  # Default embedding dim
        
        return []
    
    def save_encoder(self, filepath: str) -> None:
        """Save the fitted encoder to disk."""
        import pickle
        
        encoder_data = {
            'encoding_method': self.encoding_method,
            'kmer_size': self.kmer_size,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'vocab_size': self.vocab_size,
            'nucleotide_map': self.nucleotide_map,
            'kmer_vectorizer': self.kmer_vectorizer,
            'fitted': self.fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(encoder_data, f)
    
    @classmethod
    def load_encoder(cls, filepath: str) -> 'SequenceEncoder':
        """Load a fitted encoder from disk."""
        import pickle
        
        with open(filepath, 'rb') as f:
            encoder_data = pickle.load(f)
        
        encoder = cls(
            encoding_method=encoder_data['encoding_method'],
            kmer_size=encoder_data['kmer_size'],
            window_size=encoder_data['window_size'],
            overlap=encoder_data['overlap'],
            vocab_size=encoder_data['vocab_size']
        )
        
        encoder.nucleotide_map = encoder_data['nucleotide_map']
        encoder.kmer_vectorizer = encoder_data['kmer_vectorizer']
        encoder.fitted = encoder_data['fitted']
        
        return encoder