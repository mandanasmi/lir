#!/usr/bin/env python3
"""
Simplified Attention Strategy Experiments for LIR

This script tests different attention strategies in a simplified setting
without requiring the full PDG framework. It demonstrates how different
attention mechanisms affect learning performance.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import time
import json
from pathlib import Path
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

@dataclass
class ExperimentConfig:
    """Configuration for attention experiments."""
    # Model parameters
    input_size: int = 10
    hidden_size: int = 20
    output_size: int = 5
    num_layers: int = 3
    
    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 0.01
    batch_size: int = 32
    num_samples: int = 1000
    
    # Attention strategy
    attention_strategy: str = "uniform"
    
    # Experiment parameters
    num_runs: int = 5
    seed: int = 42


class AttentionStrategy:
    """Base class for attention strategies."""
    
    def compute_attention(self, model, inputs, targets, loss_fn) -> torch.Tensor:
        """Compute attention weights for the current batch."""
        raise NotImplementedError


class UniformAttention(AttentionStrategy):
    """Uniform attention - equal weights for all parameters."""
    
    def compute_attention(self, model, inputs, targets, loss_fn) -> torch.Tensor:
        total_params = sum(p.numel() for p in model.parameters())
        return torch.ones(total_params, dtype=torch.float32)


class GradientMagnitudeAttention(AttentionStrategy):
    """Attention based on gradient magnitudes."""
    
    def compute_attention(self, model, inputs, targets, loss_fn) -> torch.Tensor:
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Compute gradients
        gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        
        # Compute attention based on gradient magnitudes
        attention_weights = []
        for grad in gradients:
            grad_mag = torch.abs(grad).flatten()
            attention_weights.append(grad_mag)
        
        attention = torch.cat(attention_weights)
        # Normalize to sum to 1
        if attention.sum() > 0:
            attention = attention / attention.sum()
        else:
            attention = torch.ones_like(attention) / len(attention)
        
        return attention


class EntropyBasedAttention(AttentionStrategy):
    """Attention based on output entropy."""
    
    def compute_attention(self, model, inputs, targets, loss_fn) -> torch.Tensor:
        with torch.no_grad():
            outputs = model(inputs)
            # Compute entropy of outputs
            probs = torch.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            avg_entropy = entropy.mean()
            
            # Use entropy to determine attention (higher entropy = more attention)
            total_params = sum(p.numel() for p in model.parameters())
            attention = torch.full((total_params,), avg_entropy.item(), dtype=torch.float32)
            
            # Normalize
            if attention.sum() > 0:
                attention = attention / attention.sum()
            else:
                attention = torch.ones_like(attention) / len(attention)
        
        return attention


class LossBasedAttention(AttentionStrategy):
    """Attention based on current loss value."""
    
    def compute_attention(self, model, inputs, targets, loss_fn) -> torch.Tensor:
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Use loss value to determine attention
            total_params = sum(p.numel() for p in model.parameters())
            attention = torch.full((total_params,), loss.item(), dtype=torch.float32)
            
            # Normalize
            if attention.sum() > 0:
                attention = attention / attention.sum()
            else:
                attention = torch.ones_like(attention) / len(attention)
        
        return attention


class AdaptiveAttention(AttentionStrategy):
    """Adaptive attention that changes over time."""
    
    def __init__(self):
        self.iteration = 0
    
    def compute_attention(self, model, inputs, targets, loss_fn) -> torch.Tensor:
        # Start with uniform attention
        total_params = sum(p.numel() for p in model.parameters())
        base_attention = torch.ones(total_params, dtype=torch.float32)
        
        # Add adaptive component based on iteration
        adaptive_factor = 1.0 + 0.1 * np.sin(self.iteration * 0.1)
        attention = base_attention * adaptive_factor
        
        self.iteration += 1
        
        # Normalize
        if attention.sum() > 0:
            attention = attention / attention.sum()
        else:
            attention = torch.ones_like(attention) / len(attention)
        
        return attention


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for testing."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def generate_synthetic_data(num_samples: int, input_size: int, output_size: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data for training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random inputs
    X = torch.randn(num_samples, input_size)
    
    # Generate targets using a simple linear relationship with noise
    true_weights = torch.randn(input_size, output_size)
    y = torch.matmul(X, true_weights) + 0.1 * torch.randn(num_samples, output_size)
    
    return X, y


def train_with_attention(model, train_data, test_data, attention_strategy: AttentionStrategy, 
                        config: ExperimentConfig) -> Dict[str, Any]:
    """Train model with specified attention strategy."""
    
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    
    # Training history
    train_losses = []
    test_losses = []
    attention_history = []
    
    for epoch in range(config.num_epochs):
        epoch_train_loss = 0.0
        epoch_attention = []
        
        for batch_X, batch_y in train_loader:
            # Compute attention weights
            attention = attention_strategy.compute_attention(model, batch_X, batch_y, loss_fn)
            epoch_attention.append(attention.mean().item())
            
            # Forward pass
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            
            # Apply attention to gradients (simplified approach)
            # In practice, this would modify the gradient computation more directly
            optimizer.zero_grad()
            loss.backward()
            
            # Apply attention-based scaling to gradients
            param_idx = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_size = param.numel()
                    param_attention = attention[param_idx:param_idx + param_size].view(param.shape)
                    param.grad = param.grad * param_attention.mean()
                    param_idx += param_size
            
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Evaluate on test set
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = loss_fn(test_outputs, y_test).item()
        
        train_losses.append(epoch_train_loss / len(train_loader))
        test_losses.append(test_loss)
        attention_history.append(np.mean(epoch_attention))
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'attention_history': attention_history,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'convergence_epoch': find_convergence_epoch(test_losses)
    }


def find_convergence_epoch(losses: List[float], tolerance: float = 1e-4) -> int:
    """Find the epoch where convergence occurs."""
    if len(losses) < 2:
        return len(losses)
    
    for i in range(1, len(losses)):
        if abs(losses[i] - losses[i-1]) < tolerance:
            return i
    
    return len(losses)


def run_experiment_suite(config: ExperimentConfig) -> Dict[str, Any]:
    """Run experiments with different attention strategies."""
    
    # Define attention strategies
    strategies = {
        'uniform': UniformAttention(),
        'gradient_magnitude': GradientMagnitudeAttention(),
        'entropy_based': EntropyBasedAttention(),
        'loss_based': LossBasedAttention(),
        'adaptive': AdaptiveAttention()
    }
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"Running experiment with {strategy_name} attention...")
        
        strategy_results = []
        
        for run in range(config.num_runs):
            # Generate data
            train_data = generate_synthetic_data(config.num_samples, config.input_size, config.output_size, config.seed + run)
            test_data = generate_synthetic_data(config.num_samples // 4, config.input_size, config.output_size, config.seed + run + 1000)
            
            # Create model
            model = SimpleMLP(config.input_size, config.hidden_size, config.output_size, config.num_layers)
            
            # Train with attention
            run_result = train_with_attention(model, train_data, test_data, strategy, config)
            strategy_results.append(run_result)
        
        # Aggregate results
        results[strategy_name] = {
            'mean_final_test_loss': np.mean([r['final_test_loss'] for r in strategy_results]),
            'std_final_test_loss': np.std([r['final_test_loss'] for r in strategy_results]),
            'mean_convergence_epoch': np.mean([r['convergence_epoch'] for r in strategy_results]),
            'std_convergence_epoch': np.std([r['convergence_epoch'] for r in strategy_results]),
            'mean_final_train_loss': np.mean([r['final_train_loss'] for r in strategy_results]),
            'std_final_train_loss': np.std([r['final_train_loss'] for r in strategy_results]),
            'all_runs': strategy_results
        }
    
    return results


def plot_results(results: Dict[str, Any], output_dir: str = "attention_results"):
    """Create visualization plots for the results."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Plot 1: Final test loss comparison
    plt.figure(figsize=(12, 8))
    strategies = list(results.keys())
    mean_losses = [results[s]['mean_final_test_loss'] for s in strategies]
    std_losses = [results[s]['std_final_test_loss'] for s in strategies]
    
    plt.bar(strategies, mean_losses, yerr=std_losses, capsize=5, alpha=0.7)
    plt.title("Final Test Loss Comparison Across Attention Strategies", fontsize=14)
    plt.xlabel("Attention Strategy", fontsize=12)
    plt.ylabel("Final Test Loss", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "final_loss_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Convergence epoch comparison
    plt.figure(figsize=(12, 8))
    mean_epochs = [results[s]['mean_convergence_epoch'] for s in strategies]
    std_epochs = [results[s]['std_convergence_epoch'] for s in strategies]
    
    plt.bar(strategies, mean_epochs, yerr=std_epochs, capsize=5, alpha=0.7, color='orange')
    plt.title("Convergence Epoch Comparison", fontsize=14)
    plt.xlabel("Attention Strategy", fontsize=12)
    plt.ylabel("Convergence Epoch", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "convergence_epoch_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Training curves for best strategies
    plt.figure(figsize=(12, 8))
    
    # Sort strategies by performance
    sorted_strategies = sorted(strategies, key=lambda x: results[x]['mean_final_test_loss'])
    
    for strategy in sorted_strategies[:3]:  # Plot top 3 strategies
        all_runs = results[strategy]['all_runs']
        # Average across runs
        avg_test_losses = np.mean([run['test_losses'] for run in all_runs], axis=0)
        std_test_losses = np.std([run['test_losses'] for run in all_runs], axis=0)
        
        epochs = range(len(avg_test_losses))
        plt.plot(epochs, avg_test_losses, label=f"{strategy}", linewidth=2)
        plt.fill_between(epochs, avg_test_losses - std_test_losses, 
                        avg_test_losses + std_test_losses, alpha=0.2)
    
    plt.title("Training Curves for Top 3 Attention Strategies", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Test Loss", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_path}")


def save_results(results: Dict[str, Any], output_dir: str = "attention_results"):
    """Save results to JSON file."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Convert results to serializable format
    serializable_results = {}
    for strategy, data in results.items():
        serializable_results[strategy] = {
            'mean_final_test_loss': float(data['mean_final_test_loss']),
            'std_final_test_loss': float(data['std_final_test_loss']),
            'mean_convergence_epoch': float(data['mean_convergence_epoch']),
            'std_convergence_epoch': float(data['std_convergence_epoch']),
            'mean_final_train_loss': float(data['mean_final_train_loss']),
            'std_final_train_loss': float(data['std_final_train_loss'])
        }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"attention_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {results_file}")


def print_summary(results: Dict[str, Any]):
    """Print a formatted summary of the results."""
    
    print("\n" + "="*80)
    print("🎯 ATTENTION STRATEGY EXPERIMENT RESULTS")
    print("="*80)
    
    # Sort strategies by performance
    sorted_strategies = sorted(results.items(), key=lambda x: x[1]['mean_final_test_loss'])
    
    print(f"{'Strategy':<20} {'Final Loss':<15} {'Convergence':<12} {'Train Loss':<15}")
    print("-" * 80)
    
    for strategy, data in sorted_strategies:
        print(f"{strategy:<20} {data['mean_final_test_loss']:.6f}±{data['std_final_test_loss']:.6f} "
              f"{data['mean_convergence_epoch']:.1f}±{data['std_convergence_epoch']:.1f} "
              f"{data['mean_final_train_loss']:.6f}±{data['std_final_train_loss']:.6f}")
    
    print("\n🏆 BEST PERFORMING STRATEGY:", sorted_strategies[0][0])
    print(f"   Final Test Loss: {sorted_strategies[0][1]['mean_final_test_loss']:.6f} ± {sorted_strategies[0][1]['std_final_test_loss']:.6f}")
    print(f"   Convergence Epoch: {sorted_strategies[0][1]['mean_convergence_epoch']:.1f} ± {sorted_strategies[0][1]['std_convergence_epoch']:.1f}")


def main():
    """Main function to run attention strategy experiments."""
    
    print("🚀 Starting Attention Strategy Experiments...")
    
    # Configuration
    config = ExperimentConfig(
        input_size=10,
        hidden_size=20,
        output_size=5,
        num_layers=3,
        num_epochs=50,
        learning_rate=0.01,
        batch_size=32,
        num_samples=1000,
        num_runs=3
    )
    
    print(f"Configuration: {config.num_epochs} epochs, {config.num_runs} runs per strategy")
    
    # Run experiments
    start_time = time.time()
    results = run_experiment_suite(config)
    total_time = time.time() - start_time
    
    # Analyze and save results
    print_summary(results)
    save_results(results)
    plot_results(results)
    
    print(f"\n⏱️  Total experiment time: {total_time:.2f} seconds")
    print("✅ Experiments completed successfully!")


if __name__ == "__main__":
    main()
