"""
Transformer Model for Viral Evolution Prediction.

This module implements a specialized Transformer architecture for modeling
viral sequence evolution with environmental conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with environmental conditioning."""
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 dropout: float = 0.1,
                 env_dim: int = 0):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            env_dim: Environmental feature dimension for conditioning
        """
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.env_dim = env_dim
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Environmental conditioning
        if env_dim > 0:
            self.env_projection = nn.Linear(env_dim, d_model)
            self.env_gate = nn.Linear(d_model + env_dim, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                env_features: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [batch, seq_len, d_model]
            key: Key tensor [batch, seq_len, d_model]
            value: Value tensor [batch, seq_len, d_model]
            env_features: Environmental features [batch, env_dim]
            mask: Attention mask [batch, seq_len, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = query.shape
        
        # Apply environmental conditioning if available
        if env_features is not None and self.env_dim > 0:
            # Project environmental features
            env_proj = self.env_projection(env_features)  # [batch, d_model]
            env_proj = env_proj.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, d_model]
            
            # Gate the environmental influence
            env_broadcast = env_features.unsqueeze(1).expand(-1, seq_len, -1)
            gate_input = torch.cat([query, env_broadcast], dim=-1)
            gate = torch.sigmoid(self.env_gate(gate_input))
            
            # Apply environmental conditioning to query
            query = query + gate * env_proj
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # mask arrives as [B, 1, 1, L] or [B, 1, L, L] — expand to [B, heads, L, L]
            while mask.dim() < 4:
                mask = mask.unsqueeze(1)
            mask = mask.expand(-1, self.num_heads, attention_scores.size(-2), attention_scores.size(-1))
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.w_o(context)
        
        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network with environmental conditioning."""
    
    def __init__(self, 
                 d_model: int, 
                 d_ff: int, 
                 dropout: float = 0.1,
                 env_dim: int = 0):
        """
        Initialize position-wise feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            env_dim: Environmental feature dimension
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.env_dim = env_dim
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Environmental conditioning
        if env_dim > 0:
            self.env_linear = nn.Linear(env_dim, d_ff)
            self.env_gate = nn.Linear(d_ff + env_dim, d_ff)
    
    def forward(self, 
                x: torch.Tensor,
                env_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of position-wise feed-forward network.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            env_features: Environmental features [batch, env_dim]
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # First linear transformation
        ff_output = F.relu(self.linear1(x))
        
        # Apply environmental conditioning if available
        if env_features is not None and self.env_dim > 0:
            batch_size, seq_len, _ = x.shape
            
            # Project environmental features
            env_proj = F.relu(self.env_linear(env_features))  # [batch, d_ff]
            env_proj = env_proj.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, d_ff]
            
            # Gate the environmental influence
            env_broadcast = env_features.unsqueeze(1).expand(-1, seq_len, -1)
            gate_input = torch.cat([ff_output, env_broadcast], dim=-1)
            gate = torch.sigmoid(self.env_gate(gate_input))
            
            # Apply environmental conditioning
            ff_output = ff_output + gate * env_proj
        
        ff_output = self.dropout(ff_output)
        
        # Second linear transformation
        output = self.linear2(ff_output)
        
        return output


class TransformerLayer(nn.Module):
    """Single transformer layer with environmental conditioning."""
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int,
                 dropout: float = 0.1,
                 env_dim: int = 0):
        """
        Initialize transformer layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            env_dim: Environmental feature dimension
        """
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout, env_dim)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, env_dim)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor,
                env_features: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of transformer layer.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            env_features: Environmental features [batch, env_dim]
            mask: Attention mask [batch, seq_len, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.self_attention(
            x, x, x, env_features, mask
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x, env_features)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class ViralEvolutionTransformer(nn.Module):
    """
    Transformer model specialized for viral evolution prediction.
    
    Incorporates environmental conditioning and temporal awareness
    for predicting viral sequence evolution.
    """
    
    def __init__(self,
                 sequence_dim: int,
                 env_dim: int,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 max_length: int = 1000):
        """
        Initialize the viral evolution transformer.
        
        Args:
            sequence_dim: Dimension of input sequences
            env_dim: Dimension of environmental features
            hidden_dim: Hidden dimension of the model
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout_rate: Dropout probability
            max_length: Maximum sequence length
        """
        super().__init__()
        
        self.sequence_dim = sequence_dim
        self.env_dim = env_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        
        # Input projection
        self.input_projection = nn.Linear(sequence_dim, hidden_dim)
        
        # Environmental encoder
        self.env_encoder = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_length, hidden_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                d_model=hidden_dim,
                num_heads=num_heads,
                d_ff=hidden_dim * 4,
                dropout=dropout_rate,
                env_dim=env_dim
            )
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Temporal consistency module (hidden_size=hidden_dim for residual add)
        self.temporal_consistency = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
    
    def _create_positional_encoding(self, max_length: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0  # True for allowed positions, False for masked
    
    def _create_padding_mask(self, sequences: torch.Tensor) -> torch.Tensor:
        """Create padding mask for variable-length sequences."""
        # Assume padding is represented by all-zero vectors
        padding_mask = torch.any(sequences != 0, dim=-1)  # [batch, seq_len]
        return padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
    
    def forward(self,
                input_sequences: torch.Tensor,
                environmental_features: torch.Tensor,
                target_sequences: Optional[torch.Tensor] = None,
                use_causal_mask: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the transformer model.
        
        Args:
            input_sequences: Input sequences [batch, seq_len, sequence_dim]
            environmental_features: Environmental features [batch, env_dim]
            target_sequences: Target sequences for teacher forcing [batch, seq_len, sequence_dim]
            use_causal_mask: Whether to use causal masking for autoregressive generation
            
        Returns:
            Dictionary with model outputs
        """
        batch_size, seq_len, _ = input_sequences.shape
        device = input_sequences.device
        
        # Project input sequences to hidden dimension
        x = self.input_projection(input_sequences)  # [batch, seq_len, hidden_dim]
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(device)
        x = x + pos_encoding
        
        # Apply dropout
        x = self.dropout(x)
        
        # Encode environmental features
        env_encoded = self.env_encoder(environmental_features)  # [batch, hidden_dim]
        
        # Create attention masks
        padding_mask = self._create_padding_mask(input_sequences)
        
        if use_causal_mask:
            causal_mask = self._create_causal_mask(seq_len, device)
            attention_mask = padding_mask & causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            attention_mask = padding_mask
        
        # Pass through transformer layers
        attention_weights_list = []
        
        for layer in self.transformer_layers:
            x, attention_weights = layer(x, environmental_features, attention_mask)
            attention_weights_list.append(attention_weights)
        
        # Apply final normalization
        x = self.output_norm(x)
        
        # Apply temporal consistency module
        temporal_output, (hidden_state, cell_state) = self.temporal_consistency(x)
        
        # Combine transformer output with temporal consistency
        sequence_embeddings = x + temporal_output
        
        # Stack attention weights from all layers
        all_attention_weights = torch.stack(attention_weights_list, dim=1)  # [batch, num_layers, num_heads, seq_len, seq_len]
        
        outputs = {
            'sequence_embeddings': sequence_embeddings,
            'transformer_output': x,
            'temporal_output': temporal_output,
            'environmental_embeddings': env_encoded,
            'attention_weights': all_attention_weights,
            'hidden_state': hidden_state,
            'cell_state': cell_state
        }
        
        return outputs
    
    def generate_sequence(self,
                         initial_sequence: torch.Tensor,
                         environmental_features: torch.Tensor,
                         max_length: int,
                         temperature: float = 1.0,
                         top_k: Optional[int] = None,
                         top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generate sequences autoregressively.
        
        Args:
            initial_sequence: Starting sequence [1, init_len, sequence_dim]
            environmental_features: Environmental features [1, env_dim]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated sequence [1, max_length, sequence_dim]
        """
        self.eval()
        
        with torch.no_grad():
            generated = initial_sequence.clone()
            
            for _ in range(max_length - initial_sequence.size(1)):
                # Forward pass with causal masking
                outputs = self.forward(
                    generated, 
                    environmental_features, 
                    use_causal_mask=True
                )
                
                # Get logits for the last position
                last_embeddings = outputs['sequence_embeddings'][:, -1:, :]  # [1, 1, hidden_dim]
                
                # Convert embeddings to sequence probabilities (this would need an output head)
                # For now, we'll use a simple linear projection
                if not hasattr(self, 'generation_head'):
                    self.generation_head = nn.Linear(self.hidden_dim, self.sequence_dim).to(generated.device)
                
                logits = self.generation_head(last_embeddings)  # [1, 1, sequence_dim]
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
                next_token = next_token.view(1, 1, -1)
                
                # Convert to one-hot if needed
                if self.sequence_dim == 4:  # Nucleotide sequences
                    next_token_onehot = F.one_hot(next_token.squeeze(-1), num_classes=4).float()
                    generated = torch.cat([generated, next_token_onehot], dim=1)
                else:
                    generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def get_attention_maps(self,
                          input_sequences: torch.Tensor,
                          environmental_features: torch.Tensor,
                          layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Extract attention maps for visualization.
        
        Args:
            input_sequences: Input sequences
            environmental_features: Environmental features
            layer_idx: Specific layer index (None for all layers)
            
        Returns:
            Attention weights tensor
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_sequences, environmental_features)
            attention_weights = outputs['attention_weights']
            
            if layer_idx is not None:
                return attention_weights[:, layer_idx, :, :, :]
            else:
                return attention_weights
    
    def compute_sequence_similarity(self,
                                  seq1: torch.Tensor,
                                  seq2: torch.Tensor,
                                  environmental_features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between sequences in the learned embedding space.
        
        Args:
            seq1: First sequence [1, seq_len, sequence_dim]
            seq2: Second sequence [1, seq_len, sequence_dim]
            environmental_features: Environmental features [1, env_dim]
            
        Returns:
            Similarity score
        """
        self.eval()
        
        with torch.no_grad():
            # Get embeddings for both sequences
            outputs1 = self.forward(seq1, environmental_features)
            outputs2 = self.forward(seq2, environmental_features)
            
            emb1 = outputs1['sequence_embeddings'].mean(dim=1)  # [1, hidden_dim]
            emb2 = outputs2['sequence_embeddings'].mean(dim=1)  # [1, hidden_dim]
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(emb1, emb2, dim=-1)
            
            return similarity