#!/usr/bin/env python3
"""
Executable script for running LIR attention and control experiments.

This script provides a command-line interface for running experiments
to test different attention and control selection strategies in the LIR framework.

Usage:
    python run_experiments.py --help
    python run_experiments.py --quick  # Run a quick test
    python run_experiments.py --full   # Run full experimental suite
    python run_experiments.py --custom --attention uniform,entropy_based --control uniform,gradient_based
"""

import argparse
import sys
from pathlib import Path
import json
import time

# Add the current directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

from experimental_design import (
    ExperimentRunner, ExperimentConfig, 
    AttentionStrategy, ControlStrategy,
    UniformAttention, GradientMagnitudeAttention, EntropyBasedAttention,
    InconsistencyBasedAttention, AdaptiveAttention,
    UniformControl, GradientBasedControl, ParameterSensitivityControl, AdaptiveControl
)


def create_quick_config() -> ExperimentConfig:
    """Create a configuration for quick testing."""
    return ExperimentConfig(
        num_vars=3,
        num_edges=3,
        gamma=1.0,
        T=10,  # Fewer iterations for quick testing
        inner_iters=10,
        lr=1e-2,
        num_runs=2,  # Fewer runs for quick testing
        eval_iters=50
    )


def create_full_config() -> ExperimentConfig:
    """Create a configuration for comprehensive testing."""
    return ExperimentConfig(
        num_vars=4,
        num_edges=4,
        gamma=1.0,
        T=30,
        inner_iters=20,
        lr=1e-2,
        num_runs=5,
        eval_iters=350
    )


def run_quick_experiment():
    """Run a quick experiment to test the framework."""
    print("🚀 Running quick experiment...")
    
    runner = ExperimentRunner(output_dir="quick_results")
    config = create_quick_config()
    
    # Test just a few strategy combinations
    results = runner.run_experiment_suite(
        base_config=config,
        attention_strategies=["uniform", "entropy_based"],
        control_strategies=["uniform", "gradient_based"],
        pdg_types=["random"]
    )
    
    # Analyze and save results
    analysis = runner.analyze_results(results)
    runner.save_results(results, analysis)
    runner.plot_results(results, analysis)
    
    print("✅ Quick experiment completed!")
    return results, analysis


def run_full_experiment():
    """Run the full experimental suite."""
    print("🔬 Running full experimental suite...")
    
    runner = ExperimentRunner(output_dir="full_results")
    config = create_full_config()
    
    # Test all strategy combinations
    results = runner.run_experiment_suite(
        base_config=config,
        attention_strategies=["uniform", "gradient_magnitude", "entropy_based", "inconsistency_based"],
        control_strategies=["uniform", "gradient_based", "parameter_sensitivity"],
        pdg_types=["random"]
    )
    
    # Analyze and save results
    analysis = runner.analyze_results(results)
    runner.save_results(results, analysis)
    runner.plot_results(results, analysis)
    
    print("✅ Full experiment completed!")
    return results, analysis


def run_custom_experiment(attention_strategies: list, control_strategies: list, 
                         num_vars: int = 4, num_edges: int = 4, num_runs: int = 3):
    """Run a custom experiment with specified strategies."""
    print(f"🎯 Running custom experiment with {attention_strategies} attention and {control_strategies} control...")
    
    runner = ExperimentRunner(output_dir="custom_results")
    config = ExperimentConfig(
        num_vars=num_vars,
        num_edges=num_edges,
        gamma=1.0,
        T=20,
        inner_iters=15,
        lr=1e-2,
        num_runs=num_runs,
        eval_iters=200
    )
    
    results = runner.run_experiment_suite(
        base_config=config,
        attention_strategies=attention_strategies,
        control_strategies=control_strategies,
        pdg_types=["random"]
    )
    
    # Analyze and save results
    analysis = runner.analyze_results(results)
    runner.save_results(results, analysis)
    runner.plot_results(results, analysis)
    
    print("✅ Custom experiment completed!")
    return results, analysis


def print_results_summary(analysis: dict):
    """Print a formatted summary of the results."""
    print("\n" + "="*60)
    print("📊 EXPERIMENTAL RESULTS SUMMARY")
    print("="*60)
    
    # Sort strategies by mean final loss
    sorted_strategies = sorted(analysis.items(), key=lambda x: x[1]['mean_final_loss'])
    
    print(f"{'Strategy':<30} {'Final Loss':<15} {'Conv Rate':<10} {'Time (s)':<10}")
    print("-" * 65)
    
    for strategy, stats in sorted_strategies:
        print(f"{strategy:<30} {stats['mean_final_loss']:.6f}±{stats['std_final_loss']:.6f} "
              f"{stats['convergence_rate']:.2f}      {stats['mean_training_time']:.2f}")
    
    print("\n🏆 Best performing strategy:", sorted_strategies[0][0])
    print(f"   Final Loss: {sorted_strategies[0][1]['mean_final_loss']:.6f}")
    print(f"   Convergence Rate: {sorted_strategies[0][1]['convergence_rate']:.2f}")


def main():
    """Main function to handle command line arguments and run experiments."""
    parser = argparse.ArgumentParser(
        description="Run LIR attention and control experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --quick
  python run_experiments.py --full
  python run_experiments.py --custom --attention uniform,entropy_based --control uniform,gradient_based
  python run_experiments.py --custom --attention uniform --control uniform,gradient_based --vars 5 --edges 6 --runs 10
        """
    )
    
    # Experiment type selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--quick', action='store_true', 
                      help='Run a quick experiment with minimal parameters')
    group.add_argument('--full', action='store_true', 
                      help='Run the full experimental suite with all strategies')
    group.add_argument('--custom', action='store_true', 
                      help='Run a custom experiment with specified strategies')
    
    # Custom experiment parameters
    parser.add_argument('--attention', type=str, 
                       help='Comma-separated list of attention strategies (uniform,gradient_magnitude,entropy_based,inconsistency_based,adaptive)')
    parser.add_argument('--control', type=str, 
                       help='Comma-separated list of control strategies (uniform,gradient_based,parameter_sensitivity,adaptive)')
    parser.add_argument('--vars', type=int, default=4, 
                       help='Number of variables in PDG (default: 4)')
    parser.add_argument('--edges', type=int, default=4, 
                       help='Number of edges in PDG (default: 4)')
    parser.add_argument('--runs', type=int, default=3, 
                       help='Number of runs per strategy combination (default: 3)')
    
    # Output options
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip generating plots')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Validate custom experiment parameters
    if args.custom:
        if not args.attention or not args.control:
            print("❌ Error: --custom requires both --attention and --control to be specified")
            sys.exit(1)
        
        # Parse strategy lists
        attention_strategies = [s.strip() for s in args.attention.split(',')]
        control_strategies = [s.strip() for s in args.control.split(',')]
        
        # Validate strategies
        valid_attention = ["uniform", "gradient_magnitude", "entropy_based", "inconsistency_based", "adaptive"]
        valid_control = ["uniform", "gradient_based", "parameter_sensitivity", "adaptive"]
        
        for strategy in attention_strategies:
            if strategy not in valid_attention:
                print(f"❌ Error: Invalid attention strategy '{strategy}'. Valid options: {valid_attention}")
                sys.exit(1)
        
        for strategy in control_strategies:
            if strategy not in valid_control:
                print(f"❌ Error: Invalid control strategy '{strategy}'. Valid options: {valid_control}")
                sys.exit(1)
    
    # Run experiments
    try:
        start_time = time.time()
        
        if args.quick:
            results, analysis = run_quick_experiment()
        elif args.full:
            results, analysis = run_full_experiment()
        elif args.custom:
            results, analysis = run_custom_experiment(
                attention_strategies=attention_strategies,
                control_strategies=control_strategies,
                num_vars=args.vars,
                num_edges=args.edges,
                num_runs=args.runs
            )
        
        total_time = time.time() - start_time
        
        # Print summary
        print_results_summary(analysis)
        print(f"\n⏱️  Total experiment time: {total_time:.2f} seconds")
        print(f"📁 Results saved to: {Path.cwd()}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

