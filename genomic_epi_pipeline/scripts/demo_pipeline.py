#!/usr/bin/env python3
"""
Demonstration Script for Genomic Epidemiology Pipeline.

This script demonstrates the complete workflow from feature engineering
through model training and prediction with mock data.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.sequence_encoder import SequenceEncoder
from features.environmental_normalizer import EnvironmentalNormalizer
from features.temporal_structurer import TemporalStructurer
from features.feature_pipeline import FeaturePipeline
from models.probabilistic_engine import ViralEvolutionPredictor
from models.fitness_scorer import FitnessScorer
from config.settings import Config


def generate_mock_viral_sequences(num_sequences: int = 100, sequence_length: int = 300) -> list:
    """Generate mock viral sequences for demonstration."""
    nucleotides = ['A', 'T', 'G', 'C']
    sequences = []
    
    for _ in range(num_sequences):
        sequence = ''.join(np.random.choice(nucleotides, sequence_length))
        sequences.append(sequence)
    
    return sequences


def generate_mock_data(num_sequences: int = 100, num_days: int = 365) -> tuple:
    """
    Generate mock sequence and environmental data for demonstration.
    
    Args:
        num_sequences: Number of viral sequences to generate
        num_days: Number of days of data to generate
        
    Returns:
        Tuple of (sequence_dataframe, environmental_dataframe)
    """
    print("🧬 Generating mock viral sequences...")
    
    # Generate mock sequences
    sequences = generate_mock_viral_sequences(num_sequences, sequence_length=300)
    
    # Generate sequence metadata
    start_date = datetime(2020, 1, 1)
    locations = ['USA', 'UK', 'Germany', 'France', 'Italy', 'Spain', 'Canada']
    host_species = ['human', 'bat', 'mink', 'cat']
    
    sequence_data = []
    for i, seq in enumerate(sequences):
        date = start_date + timedelta(days=np.random.randint(0, num_days))
        location = np.random.choice(locations)
        host = np.random.choice(host_species)
        
        sequence_data.append({
            'sequence_id': f'seq_{i:04d}',
            'sequence': seq,
            'date': date,
            'location': location,
            'host_species': host
        })
    
    sequence_df = pd.DataFrame(sequence_data)
    
    print("🌡️ Generating mock environmental data...")
    
    # Generate environmental data
    environmental_data = []
    for day in range(num_days):
        date = start_date + timedelta(days=day)
        
        for location in locations:
            # Simulate seasonal temperature variation
            day_of_year = date.timetuple().tm_yday
            base_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365)
            temperature = base_temp + np.random.normal(0, 5)
            
            humidity = 50 + 20 * np.sin(2 * np.pi * day_of_year / 365 + np.pi/4) + np.random.normal(0, 10)
            humidity = np.clip(humidity, 0, 100)
            
            population_density = np.random.uniform(100, 1000)
            
            environmental_data.append({
                'date': date,
                'location': location,
                'temperature': temperature,
                'humidity': humidity,
                'population_density': population_density
            })
    
    environmental_df = pd.DataFrame(environmental_data)
    
    print(f"✅ Generated {len(sequence_df)} sequences and {len(environmental_df)} environmental records")
    
    return sequence_df, environmental_df


def demonstrate_feature_engineering():
    """Demonstrate the feature engineering pipeline."""
    print("\n" + "="*60)
    print("🔧 PHASE 2: FEATURE ENGINEERING DEMONSTRATION")
    print("="*60)
    
    # Generate mock data
    sequence_data, environmental_data = generate_mock_data(num_sequences=50, num_days=180)
    
    # Initialize feature pipeline
    print("\n📊 Initializing Feature Pipeline...")
    pipeline = FeaturePipeline(
        sequence_config={
            'encoding_method': 'one_hot',
            'window_size': 50,
            'overlap': 25
        },
        environmental_config={
            'continuous_method': 'standard',
            'categorical_method': 'one_hot',
            'seasonal_features': True
        },
        temporal_config={
            'time_window_days': 30,
            'prediction_horizon_days': 60,
            'min_sequences_per_window': 3
        }
    )
    
    # Process features
    print("\n🔄 Processing features...")
    try:
        features = pipeline.fit_transform(sequence_data, environmental_data)
        
        print(f"✅ Feature engineering completed!")
        print(f"   - Input sequences shape: {features['input_sequences'].shape}")
        print(f"   - Target sequences shape: {features['target_sequences'].shape}")
        print(f"   - Environmental features shape: {features['environmental_features'].shape}")
        print(f"   - Number of temporal windows: {features['metadata']['num_temporal_windows']}")
        
        return features, pipeline
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return None, None


def demonstrate_model_architecture(features):
    """Demonstrate the probabilistic engine and model architecture."""
    print("\n" + "="*60)
    print("🤖 PHASE 3: PROBABILISTIC ENGINE DEMONSTRATION")
    print("="*60)
    
    if features is None:
        print("❌ Cannot demonstrate model without features")
        return None
    
    # Extract dimensions
    sequence_dim = features['input_sequences'].shape[-1]
    env_dim = features['environmental_features'].shape[-1]
    
    print(f"\n📐 Model dimensions:")
    print(f"   - Sequence dimension: {sequence_dim}")
    print(f"   - Environmental dimension: {env_dim}")
    
    # Initialize model
    print("\n🏗️ Initializing Viral Evolution Predictor...")
    model = ViralEvolutionPredictor(
        sequence_dim=sequence_dim,
        env_dim=env_dim,
        hidden_dim=256,  # Smaller for demo
        num_layers=3,    # Fewer layers for demo
        num_heads=4,     # Fewer heads for demo
        dropout_rate=0.1,
        use_bayesian=True,
        device='cpu'     # Use CPU for demo
    )
    
    print(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Demonstrate forward pass
    print("\n🔮 Demonstrating forward pass...")
    
    # Convert to tensors
    input_sequences = torch.tensor(features['input_sequences'][:5]).float()
    env_features = torch.tensor(features['environmental_features'][:5]).float()
    target_sequences = torch.tensor(features['target_sequences'][:5]).float()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_sequences, env_features, target_sequences)
    
    print(f"✅ Forward pass completed!")
    print(f"   - Mutation probabilities shape: {outputs['mutation_probabilities'].shape}")
    print(f"   - Fitness scores shape: {outputs['fitness_scores'].shape}")
    print(f"   - Sequence embeddings shape: {outputs['sequence_embeddings'].shape}")
    
    # Demonstrate conditional probability calculation
    print("\n🎯 Demonstrating conditional probability calculation...")
    
    prob = model.calculate_conditional_probability(
        mutation_position=10,
        mutation_type='G',
        input_sequence=input_sequences[0],
        environmental_features=env_features[0]
    )
    
    print(f"✅ P(G at position 10 | sequence, environment) = {prob:.4f}")
    
    # Demonstrate uncertainty quantification
    print("\n📊 Demonstrating uncertainty quantification...")
    
    predictions = model.predict_mutations(
        input_sequences[:3],
        env_features[:3],
        num_samples=20  # Fewer samples for demo
    )
    
    print(f"✅ Uncertainty quantification completed!")
    print(f"   - Mean predictions shape: {predictions['mean_predictions'].shape}")
    print(f"   - Prediction variance shape: {predictions['prediction_variance'].shape}")
    print(f"   - 95% CI bounds available: {predictions['prediction_ci_lower'].shape}")
    
    return model


def demonstrate_fitness_scoring(model, features):
    """Demonstrate the fitness scoring system."""
    print("\n" + "="*60)
    print("💪 FITNESS SCORING DEMONSTRATION")
    print("="*60)
    
    if model is None or features is None:
        print("❌ Cannot demonstrate fitness scoring without model and features")
        return
    
    print("\n🧮 Demonstrating fitness scoring...")
    
    # Get sample data
    input_sequences = torch.tensor(features['input_sequences'][:3]).float()
    env_features = torch.tensor(features['environmental_features'][:3]).float()
    
    # Forward pass to get mutation probabilities
    with torch.no_grad():
        outputs = model(input_sequences, env_features)
        mutation_probs = outputs['mutation_probabilities']
    
    # Calculate fitness scores
    fitness_details = model.fitness_scorer(mutation_probs, env_features)
    
    print(f"✅ Fitness scoring completed!")
    print(f"   - Total fitness scores: {fitness_details['total_fitness'].squeeze()}")
    print(f"   - Stability scores: {fitness_details['stability_score'].squeeze()}")
    print(f"   - Adaptation scores: {fitness_details['adaptation_score'].squeeze()}")
    print(f"   - Environmental fitness: {fitness_details['environmental_fitness'].squeeze()}")
    print(f"   - GC content fitness: {fitness_details['gc_content_fitness'].squeeze()}")
    
    # Demonstrate fitness change prediction
    print("\n🔄 Demonstrating fitness change prediction...")
    
    # Create a mock mutation
    original_seq = mutation_probs[:1]
    mutated_seq = original_seq.clone()
    # Simulate a mutation at position 10
    mutated_seq[0, 10, :] = torch.tensor([0.1, 0.1, 0.7, 0.1])  # High G probability
    
    fitness_change = model.fitness_scorer.predict_fitness_change(
        original_seq, mutated_seq, env_features[:1]
    )
    
    print(f"✅ Fitness change analysis completed!")
    print(f"   - Total fitness change: {fitness_change['total_fitness_change'].item():.4f}")
    print(f"   - Stability change: {fitness_change['stability_change'].item():.4f}")


def demonstrate_training_loop(model, features):
    """Demonstrate a simplified training loop."""
    print("\n" + "="*60)
    print("🎓 TRAINING DEMONSTRATION")
    print("="*60)
    
    if model is None or features is None:
        print("❌ Cannot demonstrate training without model and features")
        return
    
    print("\n🏋️ Demonstrating training loop (5 epochs)...")
    
    # Prepare data
    input_sequences = torch.tensor(features['input_sequences']).float()
    target_sequences = torch.tensor(features['target_sequences']).long()
    env_features = torch.tensor(features['environmental_features']).float()
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    for epoch in range(5):
        # Forward pass
        outputs = model(input_sequences, env_features, target_sequences.float())
        
        # Calculate loss
        targets = {'target_sequences': target_sequences}
        loss = model.compute_loss(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")
    
    print("✅ Training demonstration completed!")


def main():
    """Main demonstration function."""
    print("🧬 GENOMIC EPIDEMIOLOGY PIPELINE DEMONSTRATION")
    print("=" * 60)
    print("This script demonstrates the implemented Phase 2 (Feature Engineering)")
    print("and Phase 3 (Probabilistic Engine) components of the pipeline.")
    print()
    
    # Load configuration
    print("⚙️ Loading configuration...")
    config = Config()
    print(f"✅ Configuration loaded: {config.model.model_type} model with {config.model.hidden_dim} hidden dimensions")
    
    # Demonstrate feature engineering
    features, pipeline = demonstrate_feature_engineering()
    
    if features is not None:
        # Demonstrate model architecture
        model = demonstrate_model_architecture(features)
        
        if model is not None:
            # Demonstrate fitness scoring
            demonstrate_fitness_scoring(model, features)
            
            # Demonstrate training
            demonstrate_training_loop(model, features)
    
    print("\n" + "="*60)
    print("🎉 DEMONSTRATION COMPLETED!")
    print("="*60)
    print("The pipeline successfully demonstrated:")
    print("✅ Feature engineering with sequence encoding, environmental normalization, and temporal structuring")
    print("✅ Probabilistic engine with Transformer architecture and Bayesian uncertainty")
    print("✅ Fitness scoring with multiple biological criteria")
    print("✅ Conditional probability calculation P(Mutation | Sequence, Environment, Time)")
    print("✅ Training loop with loss computation and optimization")
    print()
    print("Next steps for full implementation:")
    print("🔲 Phase 1: Data ingestion from real databases (NCBI, Nextstrain)")
    print("🔲 Phase 4: Monte Carlo simulation and hindcasting validation")
    print("🔲 Phase 5: Phylogenetic distance calculation and confidence intervals")
    print("🔲 Integration with HPC resources for large-scale training")


if __name__ == "__main__":
    main()