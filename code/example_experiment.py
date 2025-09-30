#!/usr/bin/env python3
"""
Example script demonstrating the LIR attention and control experimental framework.

This script shows how to:
1. Set up and run experiments
2. Compare different attention and control strategies
3. Analyze and visualize results
4. Extend the framework with custom strategies

Run this script to see the framework in action:
    python example_experiment.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from experimental_design import (
    ExperimentRunner, ExperimentConfig,
    AttentionStrategy, ControlStrategy,
    UniformAttention, EntropyBasedAttention,
    UniformControl, GradientBasedControl,
    MetricsCalculator
)


def demonstrate_basic_usage():
    """Demonstrate basic usage of the experimental framework."""
    print("🔬 Demonstrating Basic Experimental Framework Usage")
    print("=" * 60)
    
    # Create experiment runner
    runner = ExperimentRunner(output_dir="example_results")
    
    # Create a simple configuration
    config = ExperimentConfig(
        num_vars=3,
        num_edges=3,
        gamma=1.0,
        T=15,  # Fewer iterations for demonstration
        inner_iters=10,
        lr=1e-2,
        num_runs=2,  # Fewer runs for demonstration
        eval_iters=100
    )
    
    print(f"Configuration: {config.num_vars} variables, {config.num_edges} edges")
    print(f"Training: {config.T} outer steps, {config.inner_iters} inner steps")
    print(f"Runs: {config.num_runs} per strategy combination")
    
    # Run experiments with a few strategy combinations
    print("\n🚀 Running experiments...")
    results = runner.run_experiment_suite(
        base_config=config,
        attention_strategies=["uniform", "entropy_based"],
        control_strategies=["uniform", "gradient_based"],
        pdg_types=["random"]
    )
    
    print(f"✅ Completed {len(results)} experiment runs")
    
    # Analyze results
    print("\n📊 Analyzing results...")
    analysis = runner.analyze_results(results)
    
    # Print summary
    print("\n📈 Results Summary:")
    print("-" * 50)
    for strategy, stats in analysis.items():
        print(f"{strategy}:")
        print(f"  Final Loss: {stats['mean_final_loss']:.6f} ± {stats['std_final_loss']:.6f}")
        print(f"  Convergence Rate: {stats['convergence_rate']:.2f}")
        print(f"  Training Time: {stats['mean_training_time']:.2f}s")
        print()
    
    # Save results
    runner.save_results(results, analysis)
    print("💾 Results saved to example_results/")
    
    return results, analysis


def demonstrate_custom_strategies():
    """Demonstrate how to create and use custom attention and control strategies."""
    print("\n🛠️  Demonstrating Custom Strategy Implementation")
    print("=" * 60)
    
    # Custom attention strategy: Focus on edges with high variance
    class VarianceBasedAttention(AttentionStrategy):
        """Attention strategy based on variance in conditional distributions."""
        
        def compute_attention(self, pdg, mu_star, gamma):
            attention_weights = []
            
            for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
                if hasattr(P, 'probs'):
                    probs = P.probs()
                    # Compute variance across all conditional distributions
                    variance = torch.var(probs).item()
                    attention_weights.append(variance)
                else:
                    attention_weights.append(0.0)
            
            attention = torch.tensor(attention_weights, dtype=torch.double)
            # Normalize to sum to 1
            if attention.sum() > 0:
                attention = attention / attention.sum()
            else:
                attention = torch.ones_like(attention) / len(attention)
            
            return attention
    
    # Custom control strategy: Adaptive learning rate per parameter
    class AdaptiveLearningRateControl(ControlStrategy):
        """Control strategy with adaptive learning rates per parameter."""
        
        def __init__(self):
            self.parameter_history = {}
            self.iteration = 0
        
        def compute_control(self, pdg, gradients, mu_star):
            control_weights = []
            
            for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
                if hasattr(P, 'logits') and P.logits.requires_grad:
                    param_id = str(L)
                    
                    # Track parameter update history
                    if param_id not in self.parameter_history:
                        self.parameter_history[param_id] = []
                    
                    # Compute adaptive learning rate based on gradient history
                    if len(self.parameter_history[param_id]) > 0:
                        # Use exponential moving average of gradient magnitudes
                        avg_grad_mag = np.mean(self.parameter_history[param_id][-5:])  # Last 5 iterations
                        adaptive_lr = 1.0 / (1.0 + avg_grad_mag)
                    else:
                        adaptive_lr = 1.0
                    
                    control_weights.append(adaptive_lr)
                    
                    # Update history
                    if P.logits.grad is not None:
                        grad_mag = torch.norm(P.logits.grad).item()
                        self.parameter_history[param_id].append(grad_mag)
                else:
                    control_weights.append(1.0)
            
            self.iteration += 1
            control = torch.tensor(control_weights, dtype=torch.double)
            
            # Normalize
            if control.sum() > 0:
                control = control / control.sum() * len(control)
            
            return control
    
    print("✅ Custom strategies implemented:")
    print("  - VarianceBasedAttention: Focus on high-variance edges")
    print("  - AdaptiveLearningRateControl: Adaptive learning rates per parameter")
    
    # Test custom strategies
    runner = ExperimentRunner(output_dir="custom_strategy_results")
    
    # Add custom strategies to runner
    runner.attention_strategies["variance_based"] = VarianceBasedAttention()
    runner.control_strategies["adaptive_lr"] = AdaptiveLearningRateControl()
    
    config = ExperimentConfig(
        num_vars=3,
        num_edges=3,
        gamma=1.0,
        T=10,
        inner_iters=8,
        lr=1e-2,
        num_runs=2,
        eval_iters=50
    )
    
    print("\n🧪 Testing custom strategies...")
    results = runner.run_experiment_suite(
        base_config=config,
        attention_strategies=["uniform", "variance_based"],
        control_strategies=["uniform", "adaptive_lr"],
        pdg_types=["random"]
    )
    
    analysis = runner.analyze_results(results)
    
    print("\n📊 Custom Strategy Results:")
    print("-" * 40)
    for strategy, stats in analysis.items():
        print(f"{strategy}: Loss = {stats['mean_final_loss']:.6f}, "
              f"Conv Rate = {stats['convergence_rate']:.2f}")
    
    return results, analysis


def demonstrate_metrics_analysis():
    """Demonstrate advanced metrics analysis."""
    print("\n📊 Demonstrating Advanced Metrics Analysis")
    print("=" * 60)
    
    # Simulate some loss history data
    loss_histories = {
        "strategy_1": [1.0, 0.8, 0.6, 0.5, 0.45, 0.44, 0.43, 0.42, 0.41, 0.40],
        "strategy_2": [1.0, 0.9, 0.7, 0.6, 0.55, 0.52, 0.50, 0.49, 0.48, 0.47],
        "strategy_3": [1.0, 0.7, 0.5, 0.4, 0.35, 0.33, 0.32, 0.31, 0.30, 0.29]
    }
    
    print("Analyzing convergence and stability metrics...")
    
    for strategy, history in loss_histories.items():
        # Compute convergence metrics
        conv_metrics = MetricsCalculator.compute_convergence_metrics(history)
        
        # Compute stability metrics
        stability_metrics = MetricsCalculator.compute_stability_metrics(history)
        
        print(f"\n{strategy}:")
        print(f"  Convergence: {conv_metrics['converged']} (iter {conv_metrics['iterations']})")
        print(f"  Final Loss: {conv_metrics['final_loss']:.4f}")
        print(f"  Loss Reduction: {conv_metrics['loss_reduction']:.4f}")
        print(f"  Stability: {stability_metrics['stability']:.4f}")
        print(f"  Variance: {stability_metrics['variance']:.6f}")
    
    # Create a simple visualization
    plt.figure(figsize=(10, 6))
    for strategy, history in loss_histories.items():
        plt.plot(history, label=strategy, marker='o')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Trajectories Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('example_results/loss_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n📈 Loss trajectory plot saved to example_results/loss_trajectories.png")


def demonstrate_pdg_variations():
    """Demonstrate different PDG topologies."""
    print("\n🕸️  Demonstrating Different PDG Topologies")
    print("=" * 60)
    
    from experimental_design import PDGGenerator
    
    # Test different PDG topologies
    topologies = {
        "Chain": PDGGenerator.generate_chain_pdg,
        "Tree": PDGGenerator.generate_tree_pdg,
        "Fully Connected": PDGGenerator.generate_fully_connected_pdg,
        "Sparse": PDGGenerator.generate_sparse_pdg
    }
    
    print("Generating PDGs with different topologies...")
    
    for name, generator in topologies.items():
        try:
            pdg = generator(num_vars=4, val_range=(2, 3), seed=42)
            num_edges = len(list(pdg.edges("l,X,Y,α,β,P")))
            print(f"  {name}: {num_edges} edges")
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    print("\n✅ PDG topology generation demonstrated")


def main():
    """Main demonstration function."""
    print("🎯 LIR Attention and Control Experimental Framework Demo")
    print("=" * 70)
    
    try:
        # Run demonstrations
        print("\n1️⃣  Basic Framework Usage")
        results1, analysis1 = demonstrate_basic_usage()
        
        print("\n2️⃣  Custom Strategy Implementation")
        results2, analysis2 = demonstrate_custom_strategies()
        
        print("\n3️⃣  Advanced Metrics Analysis")
        demonstrate_metrics_analysis()
        
        print("\n4️⃣  PDG Topology Variations")
        demonstrate_pdg_variations()
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("  - Run 'python run_experiments.py --help' for command-line usage")
        print("  - Check example_results/ for generated files")
        print("  - Modify strategies in experimental_design.py for your research")
        print("  - Read EXPERIMENTAL_DESIGN_README.md for detailed documentation")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

