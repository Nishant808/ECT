"""
Bayesian Neural Network for Uncertainty Quantification.

This module implements a Bayesian neural network using variational inference
to provide uncertainty estimates for viral evolution predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
import math


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.
    
    Uses variational inference to learn distributions over weights
    instead of point estimates.
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 prior_scale: float = 1.0,
                 posterior_scale_init: float = -3.0):
        """
        Initialize Bayesian linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            prior_scale: Scale of the prior distribution
            posterior_scale_init: Initial value for posterior scale (log scale)
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_scale = prior_scale
        
        # Weight parameters (mean and log scale)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters (mean and log scale)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters(posterior_scale_init)
        
        # Prior distributions
        self.weight_prior = torch.distributions.Normal(0, prior_scale)
        self.bias_prior = torch.distributions.Normal(0, prior_scale)
    
    def reset_parameters(self, posterior_scale_init: float) -> None:
        """Initialize parameters."""
        # Initialize means
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        
        # Initialize log scales
        nn.init.constant_(self.weight_log_sigma, posterior_scale_init)
        nn.init.constant_(self.bias_log_sigma, posterior_scale_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weight sampling.
        
        Args:
            x: Input tensor [batch_size, in_features]
            
        Returns:
            Output tensor [batch_size, out_features]
        """
        # Sample weights and biases
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)
        
        # Reparameterization trick
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)
        
        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior.
        
        Returns:
            KL divergence value
        """
        # Weight KL divergence
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight_posterior = torch.distributions.Normal(self.weight_mu, weight_sigma)
        weight_kl = torch.distributions.kl_divergence(
            weight_posterior, 
            self.weight_prior.expand(self.weight_mu.shape)
        ).sum()
        
        # Bias KL divergence
        bias_sigma = torch.exp(self.bias_log_sigma)
        bias_posterior = torch.distributions.Normal(self.bias_mu, bias_sigma)
        bias_kl = torch.distributions.kl_divergence(
            bias_posterior,
            self.bias_prior.expand(self.bias_mu.shape)
        ).sum()
        
        return weight_kl + bias_kl


class BayesianNeuralNetwork(nn.Module):
    """
    Bayesian neural network for uncertainty quantification in viral evolution.
    
    Provides both aleatoric (data) and epistemic (model) uncertainty estimates.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 prior_scale: float = 1.0,
                 posterior_scale_init: float = -3.0,
                 activation: str = "relu",
                 dropout_rate: float = 0.1):
        """
        Initialize Bayesian neural network.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            prior_scale: Scale of prior distributions
            posterior_scale_init: Initial posterior scale
            activation: Activation function ('relu', 'tanh', 'gelu')
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.prior_scale = prior_scale
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(BayesianLinear(
                prev_dim, hidden_dim, prior_scale, posterior_scale_init
            ))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = BayesianLinear(
            prev_dim, output_dim, prior_scale, posterior_scale_init
        )
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Aleatoric uncertainty parameter (learnable noise)
        self.log_noise = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Bayesian network.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        # Pass through hidden layers
        for layer, dropout in zip(self.layers, self.dropouts):
            x = layer(x)
            x = self.activation(x)
            x = dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def predict_with_uncertainty(self,
                               x: torch.Tensor,
                               num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty quantification.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        self.train()  # Enable dropout and weight sampling
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, output_dim]
        
        # Calculate statistics
        mean_prediction = torch.mean(predictions, dim=0)
        epistemic_uncertainty = torch.var(predictions, dim=0)
        
        # Aleatoric uncertainty (learned noise)
        aleatoric_uncertainty = torch.exp(self.log_noise) * torch.ones_like(mean_prediction)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Confidence intervals
        std_prediction = torch.sqrt(total_uncertainty)
        ci_lower = mean_prediction - 1.96 * std_prediction
        ci_upper = mean_prediction + 1.96 * std_prediction
        
        return {
            'mean': mean_prediction,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'std': std_prediction,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'samples': predictions
        }
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute total KL divergence for all Bayesian layers.
        
        Returns:
            Total KL divergence
        """
        kl_total = 0.0
        
        for layer in self.layers:
            kl_total += layer.kl_divergence()
        
        kl_total += self.output_layer.kl_divergence()
        
        return kl_total
    
    def elbo_loss(self,
                  x: torch.Tensor,
                  y: torch.Tensor,
                  num_batches: int,
                  beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute Evidence Lower BOund (ELBO) loss.
        
        Args:
            x: Input tensor
            y: Target tensor
            num_batches: Number of batches in the dataset
            beta: KL divergence weight (beta-VAE style)
            
        Returns:
            Dictionary with loss components
        """
        # Forward pass
        predictions = self.forward(x)
        
        # Likelihood term (reconstruction loss)
        likelihood = -F.mse_loss(predictions, y, reduction='sum')
        
        # KL divergence term
        kl_div = self.kl_divergence()
        
        # ELBO = likelihood - KL divergence
        # Scale KL by number of batches (standard practice)
        elbo = likelihood - beta * kl_div / num_batches
        
        # Loss is negative ELBO
        loss = -elbo
        
        return {
            'loss': loss,
            'likelihood': likelihood,
            'kl_divergence': kl_div,
            'elbo': elbo
        }
    
    def calibration_error(self,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         num_samples: int = 100,
                         num_bins: int = 10) -> float:
        """
        Compute calibration error for uncertainty estimates.
        
        Args:
            x: Input tensor
            y: True targets
            num_samples: Number of Monte Carlo samples
            num_bins: Number of calibration bins
            
        Returns:
            Expected Calibration Error (ECE)
        """
        predictions = self.predict_with_uncertainty(x, num_samples)
        
        mean_pred = predictions['mean']
        std_pred = predictions['std']
        
        # Calculate prediction intervals
        errors = torch.abs(mean_pred - y)
        
        # Bin by confidence (inverse of uncertainty)
        confidence = 1.0 / (1.0 + std_pred)
        
        # Calculate ECE
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        ece = 0.0
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in this bin
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            
            if in_bin.sum() > 0:
                # Average confidence in bin
                bin_confidence = confidence[in_bin].mean()
                
                # Average accuracy in bin (for regression, use threshold)
                threshold = std_pred[in_bin].mean()
                bin_accuracy = (errors[in_bin] <= threshold).float().mean()
                
                # Weighted contribution to ECE
                bin_weight = in_bin.sum().float() / len(confidence)
                ece += bin_weight * torch.abs(bin_confidence - bin_accuracy)
        
        return ece.item()


class VariationalDropout(nn.Module):
    """
    Variational dropout for additional uncertainty quantification.
    
    Learns dropout probabilities instead of using fixed values.
    """
    
    def __init__(self, input_dim: int, alpha_init: float = -3.0):
        """
        Initialize variational dropout.
        
        Args:
            input_dim: Input dimension
            alpha_init: Initial log alpha value
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # Learnable dropout parameter (log alpha)
        self.log_alpha = nn.Parameter(torch.full((input_dim,), alpha_init))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply variational dropout.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with variational dropout applied
        """
        if not self.training:
            return x
        
        # Calculate dropout probability
        alpha = torch.exp(self.log_alpha)
        
        # Sample dropout mask
        eps = torch.randn_like(x)
        
        # Variational dropout
        return x * (1 + alpha * eps)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence for variational dropout.
        
        Returns:
            KL divergence value
        """
        alpha = torch.exp(self.log_alpha)
        
        # KL divergence between log-normal and standard normal
        kl = 0.5 * torch.log(alpha + 1e-8) + 0.5 * (1 + alpha).log() - 0.5
        
        return kl.sum()


class EnsembleBayesianNetwork(nn.Module):
    """
    Ensemble of Bayesian neural networks for improved uncertainty estimation.
    
    Combines multiple Bayesian networks to capture both epistemic and
    aleatoric uncertainty more effectively.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 num_models: int = 5,
                 prior_scale: float = 1.0,
                 **kwargs):
        """
        Initialize ensemble of Bayesian networks.
        
        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
            num_models: Number of models in ensemble
            prior_scale: Prior scale for individual models
            **kwargs: Additional arguments for BayesianNeuralNetwork
        """
        super().__init__()
        
        self.num_models = num_models
        
        # Create ensemble of Bayesian networks
        self.models = nn.ModuleList([
            BayesianNeuralNetwork(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                prior_scale=prior_scale,
                **kwargs
            )
            for _ in range(num_models)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble (returns mean prediction).
        
        Args:
            x: Input tensor
            
        Returns:
            Mean prediction across ensemble
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        return torch.mean(torch.stack(predictions, dim=0), dim=0)
    
    def predict_with_uncertainty(self,
                               x: torch.Tensor,
                               num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Predict with ensemble uncertainty quantification.
        
        Args:
            x: Input tensor
            num_samples: Number of samples per model
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        all_predictions = []
        
        for model in self.models:
            model_predictions = model.predict_with_uncertainty(x, num_samples)
            all_predictions.append(model_predictions['samples'])
        
        # Combine predictions from all models
        all_predictions = torch.cat(all_predictions, dim=0)  # [num_models * num_samples, batch, output]
        
        # Calculate ensemble statistics
        mean_prediction = torch.mean(all_predictions, dim=0)
        total_uncertainty = torch.var(all_predictions, dim=0)
        
        # Decompose uncertainty
        # Model uncertainty (between models)
        model_means = []
        for model in self.models:
            model_pred = model.predict_with_uncertainty(x, num_samples)
            model_means.append(model_pred['mean'])
        
        model_means = torch.stack(model_means, dim=0)
        model_uncertainty = torch.var(model_means, dim=0)
        
        # Data uncertainty (within models)
        data_uncertainty = total_uncertainty - model_uncertainty
        
        # Confidence intervals
        std_prediction = torch.sqrt(total_uncertainty)
        ci_lower = mean_prediction - 1.96 * std_prediction
        ci_upper = mean_prediction + 1.96 * std_prediction
        
        return {
            'mean': mean_prediction,
            'model_uncertainty': model_uncertainty,
            'data_uncertainty': data_uncertainty,
            'total_uncertainty': total_uncertainty,
            'std': std_prediction,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'samples': all_predictions
        }
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute total KL divergence for ensemble.
        
        Returns:
            Sum of KL divergences from all models
        """
        total_kl = 0.0
        
        for model in self.models:
            total_kl += model.kl_divergence()
        
        return total_kl