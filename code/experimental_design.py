"""
Experimental Design for LIR Attention and Control Selection

This module provides a comprehensive experimental framework for testing different
approaches to attention (φ) and control (χ) selection in the Local Inconsistency
Resolution (LIR) framework using random synthetic PDGs.

The LIR update rule is:
θ_new = exp_θ_old(-χ ⊙ ∇_θ ⟨⟨φ ⊙ M(θ)⟩⟩)

Where:
- φ: attention mechanism (which parts of the model to focus on)
- χ: control mechanism (how much to update each parameter)
- M(θ): parametric model (PDG)
- ⟨⟨·⟩⟩: inconsistency measure
"""

import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from collections import defaultdict

# Import existing LIR components
from testing_lir_simple import generate_random_pdg, make_every_cpd_parametric_projections_fixed
from lir__simpler import lir_train_simple
from pdg.alg.torch_opt import opt_joint, torch_score


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    # PDG generation parameters
    num_vars: int = 4
    num_edges: int = 4
    val_range: Tuple[int, int] = (2, 4)
    src_range: Tuple[int, int] = (1, 2)
    tgt_range: Tuple[int, int] = (1, 1)
    
    # LIR training parameters
    gamma: float = 1.0
    T: int = 30  # outer steps
    inner_iters: int = 20
    lr: float = 1e-2
    optimizer_ctor = torch.optim.Adam
    
    # Attention and control strategies
    attention_strategy: str = "uniform"
    control_strategy: str = "uniform"
    
    # Experiment parameters
    seed: int = 0
    init_method: str = "uniform"
    num_runs: int = 5  # for statistical significance
    
    # Evaluation parameters
    eval_iters: int = 350


class AttentionStrategy(ABC):
    """Abstract base class for attention selection strategies."""
    
    @abstractmethod
    def compute_attention(self, pdg, mu_star: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Compute attention weights φ for the current state.
        
        Args:
            pdg: The PDG model
            mu_star: Current optimal joint distribution
            gamma: Regularization parameter
            
        Returns:
            attention: Tensor of attention weights
        """
        pass


class ControlStrategy(ABC):
    """Abstract base class for control selection strategies."""
    
    @abstractmethod
    def compute_control(self, pdg, gradients: torch.Tensor, mu_star: torch.Tensor) -> torch.Tensor:
        """
        Compute control weights χ for parameter updates.
        
        Args:
            pdg: The PDG model
            gradients: Current parameter gradients
            mu_star: Current optimal joint distribution
            
        Returns:
            control: Tensor of control weights
        """
        pass


# ============================================================================
# ATTENTION STRATEGIES
# ============================================================================

class UniformAttention(AttentionStrategy):
    """Uniform attention - focus equally on all parts of the model."""
    
    def compute_attention(self, pdg, mu_star: torch.Tensor, gamma: float) -> torch.Tensor:
        # Return uniform attention weights
        num_edges = len(list(pdg.edges("l,X,Y,α,β,P")))
        return torch.ones(num_edges, dtype=torch.double)


class GradientMagnitudeAttention(AttentionStrategy):
    """Attention based on gradient magnitudes - focus on parameters with large gradients."""
    
    def compute_attention(self, pdg, mu_star: torch.Tensor, gamma: float) -> torch.Tensor:
        attention_weights = []
        
        for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
            if hasattr(P, 'logits') and P.logits.requires_grad:
                # Compute gradient magnitude for this edge
                grad_mag = torch.norm(P.logits.grad) if P.logits.grad is not None else 0.0
                attention_weights.append(grad_mag.item())
            else:
                attention_weights.append(0.0)
        
        attention = torch.tensor(attention_weights, dtype=torch.double)
        # Normalize to sum to 1
        if attention.sum() > 0:
            attention = attention / attention.sum()
        else:
            attention = torch.ones_like(attention) / len(attention)
        
        return attention


class EntropyBasedAttention(AttentionStrategy):
    """Attention based on entropy of conditional distributions - focus on uncertain parts."""
    
    def compute_attention(self, pdg, mu_star: torch.Tensor, gamma: float) -> torch.Tensor:
        attention_weights = []
        
        for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
            if hasattr(P, 'probs'):
                probs = P.probs()
                # Compute entropy: -sum(p * log(p))
                entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                attention_weights.append(entropy.item())
            else:
                attention_weights.append(0.0)
        
        attention = torch.tensor(attention_weights, dtype=torch.double)
        # Normalize to sum to 1
        if attention.sum() > 0:
            attention = attention / attention.sum()
        else:
            attention = torch.ones_like(attention) / len(attention)
        
        return attention


class InconsistencyBasedAttention(AttentionStrategy):
    """Attention based on local inconsistency measures."""
    
    def compute_attention(self, pdg, mu_star: torch.Tensor, gamma: float) -> torch.Tensor:
        attention_weights = []
        
        for L, X, Y, α, β, P in pdg.edges("l,X,Y,α,β,P"):
            # Compute local inconsistency for this edge
            # This is a simplified version - in practice, you'd compute the actual inconsistency
            if hasattr(P, 'probs'):
                probs = P.probs()
                # Use variance as a proxy for inconsistency
                inconsistency = torch.var(probs).item()
                attention_weights.append(inconsistency)
            else:
                attention_weights.append(0.0)
        
        attention = torch.tensor(attention_weights, dtype=torch.double)
        # Normalize to sum to 1
        if attention.sum() > 0:
            attention = attention / attention.sum()
        else:
            attention = torch.ones_like(attention) / len(attention)
        
        return attention


class AdaptiveAttention(AttentionStrategy):
    """Adaptive attention that changes over time based on learning progress."""
    
    def __init__(self, base_strategy: AttentionStrategy = None):
        self.base_strategy = base_strategy or UniformAttention()
        self.iteration = 0
    
    def compute_attention(self, pdg, mu_star: torch.Tensor, gamma: float) -> torch.Tensor:
        # Start with base strategy
        base_attention = self.base_strategy.compute_attention(pdg, mu_star, gamma)
        
        # Add adaptive component based on iteration
        adaptive_factor = 1.0 + 0.1 * np.sin(self.iteration * 0.1)
        attention = base_attention * adaptive_factor
        
        self.iteration += 1
        return attention


# ============================================================================
# CONTROL STRATEGIES
# ============================================================================

class UniformControl(ControlStrategy):
    """Uniform control - update all parameters equally."""
    
    def compute_control(self, pdg, gradients: torch.Tensor, mu_star: torch.Tensor) -> torch.Tensor:
        num_params = gradients.numel()
        return torch.ones(num_params, dtype=torch.double)


class GradientBasedControl(ControlStrategy):
    """Control based on gradient magnitudes - larger updates for smaller gradients."""
    
    def compute_control(self, pdg, gradients: torch.Tensor, mu_star: torch.Tensor) -> torch.Tensor:
        # Inverse relationship: smaller gradients get larger control
        grad_mags = torch.abs(gradients)
        # Avoid division by zero
        control = 1.0 / (grad_mags + 1e-8)
        # Normalize
        control = control / control.sum() * len(control)
        return control


class ParameterSensitivityControl(ControlStrategy):
    """Control based on parameter sensitivity to the objective."""
    
    def compute_control(self, pdg, gradients: torch.Tensor, mu_star: torch.Tensor) -> torch.Tensor:
        # Use gradient magnitude as sensitivity measure
        sensitivity = torch.abs(gradients)
        # Normalize to sum to number of parameters
        control = sensitivity / sensitivity.sum() * len(sensitivity)
        return control


class AdaptiveControl(ControlStrategy):
    """Adaptive control that adjusts based on learning progress."""
    
    def __init__(self, base_strategy: ControlStrategy = None):
        self.base_strategy = base_strategy or UniformControl()
        self.iteration = 0
    
    def compute_control(self, pdg, gradients: torch.Tensor, mu_star: torch.Tensor) -> torch.Tensor:
        # Start with base strategy
        base_control = self.base_strategy.compute_control(pdg, gradients, mu_star)
        
        # Add adaptive component
        adaptive_factor = 1.0 + 0.05 * np.cos(self.iteration * 0.2)
        control = base_control * adaptive_factor
        
        self.iteration += 1
        return control


# ============================================================================
# ENHANCED PDG GENERATION
# ============================================================================

class PDGGenerator:
    """Enhanced PDG generator with different topologies and complexity levels."""
    
    @staticmethod
    def generate_chain_pdg(num_vars: int = 4, val_range: Tuple[int, int] = (2, 4), seed: int = 0) -> Any:
        """Generate a chain-structured PDG: A -> B -> C -> D."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        from pdg.pdg import PDG
        from pdg.rv import Variable as Var
        from pdg.dist import CPT
        
        pdg = PDG()
        varlist = []
        
        # Create variables
        for i in range(num_vars):
            domain_size = random.randint(*val_range)
            var = Var.alph(chr(65 + i), domain_size)
            pdg += var
            varlist.append(var)
        
        # Create chain edges
        for i in range(num_vars - 1):
            pdg += CPT.make_random(Var.product([varlist[i]]), Var.product([varlist[i + 1]]))
        
        return pdg
    
    @staticmethod
    def generate_tree_pdg(num_vars: int = 4, val_range: Tuple[int, int] = (2, 4), seed: int = 0) -> Any:
        """Generate a tree-structured PDG."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        from pdg.pdg import PDG
        from pdg.rv import Variable as Var
        from pdg.dist import CPT
        
        pdg = PDG()
        varlist = []
        
        # Create variables
        for i in range(num_vars):
            domain_size = random.randint(*val_range)
            var = Var.alph(chr(65 + i), domain_size)
            pdg += var
            varlist.append(var)
        
        # Create tree edges (root has children, children have grandchildren, etc.)
        for i in range(1, num_vars):
            parent_idx = (i - 1) // 2
            pdg += CPT.make_random(Var.product([varlist[parent_idx]]), Var.product([varlist[i]]))
        
        return pdg
    
    @staticmethod
    def generate_fully_connected_pdg(num_vars: int = 4, val_range: Tuple[int, int] = (2, 4), seed: int = 0) -> Any:
        """Generate a fully connected PDG."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        from pdg.pdg import PDG
        from pdg.rv import Variable as Var
        from pdg.dist import CPT
        
        pdg = PDG()
        varlist = []
        
        # Create variables
        for i in range(num_vars):
            domain_size = random.randint(*val_range)
            var = Var.alph(chr(65 + i), domain_size)
            pdg += var
            varlist.append(var)
        
        # Create all possible edges
        for i in range(num_vars):
            for j in range(num_vars):
                if i != j:
                    pdg += CPT.make_random(Var.product([varlist[i]]), Var.product([varlist[j]]))
        
        return pdg
    
    @staticmethod
    def generate_sparse_pdg(num_vars: int = 4, sparsity: float = 0.3, val_range: Tuple[int, int] = (2, 4), seed: int = 0) -> Any:
        """Generate a sparse PDG with controlled sparsity."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        from pdg.pdg import PDG
        from pdg.rv import Variable as Var
        from pdg.dist import CPT
        
        pdg = PDG()
        varlist = []
        
        # Create variables
        for i in range(num_vars):
            domain_size = random.randint(*val_range)
            var = Var.alph(chr(65 + i), domain_size)
            pdg += var
            varlist.append(var)
        
        # Create edges with controlled sparsity
        max_edges = num_vars * (num_vars - 1)
        num_edges = int(max_edges * sparsity)
        
        edges_created = 0
        attempts = 0
        max_attempts = max_edges * 2
        
        while edges_created < num_edges and attempts < max_attempts:
            i, j = random.sample(range(num_vars), 2)
            if i != j:
                # Check if edge already exists (simplified check)
                try:
                    pdg += CPT.make_random(Var.product([varlist[i]]), Var.product([varlist[j]]))
                    edges_created += 1
                except:
                    pass
            attempts += 1
        
        return pdg


# ============================================================================
# EVALUATION METRICS
# ============================================================================

@dataclass
class ExperimentResults:
    """Container for experiment results."""
    config: ExperimentConfig
    final_loss: float
    convergence_iterations: int
    training_time: float
    attention_weights: List[float]
    control_weights: List[float]
    loss_history: List[float]
    convergence_achieved: bool


class MetricsCalculator:
    """Calculate various evaluation metrics for experiments."""
    
    @staticmethod
    def compute_convergence_metrics(loss_history: List[float], tolerance: float = 1e-6) -> Dict[str, Any]:
        """Compute convergence-related metrics."""
        if len(loss_history) < 2:
            return {"converged": False, "iterations": 0, "final_loss": loss_history[0] if loss_history else 0}
        
        # Check for convergence
        converged = False
        convergence_iter = len(loss_history)
        
        for i in range(1, len(loss_history)):
            if abs(loss_history[i] - loss_history[i-1]) < tolerance:
                converged = True
                convergence_iter = i
                break
        
        return {
            "converged": converged,
            "iterations": convergence_iter,
            "final_loss": loss_history[-1],
            "loss_reduction": loss_history[0] - loss_history[-1],
            "convergence_rate": convergence_iter / len(loss_history) if loss_history else 0
        }
    
    @staticmethod
    def compute_stability_metrics(loss_history: List[float]) -> Dict[str, Any]:
        """Compute stability-related metrics."""
        if len(loss_history) < 2:
            return {"variance": 0, "stability": 1.0}
        
        # Compute variance in the second half of training
        second_half = loss_history[len(loss_history)//2:]
        variance = np.var(second_half)
        
        # Stability is inverse of variance (normalized)
        stability = 1.0 / (1.0 + variance)
        
        return {
            "variance": variance,
            "stability": stability,
            "oscillation": np.std(np.diff(second_half))
        }


# ============================================================================
# EXPERIMENTAL RUNNER
# ============================================================================

class ExperimentRunner:
    """Main experimental runner for testing attention and control strategies."""
    
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Strategy registries
        self.attention_strategies = {
            "uniform": UniformAttention(),
            "gradient_magnitude": GradientMagnitudeAttention(),
            "entropy_based": EntropyBasedAttention(),
            "inconsistency_based": InconsistencyBasedAttention(),
            "adaptive": AdaptiveAttention()
        }
        
        self.control_strategies = {
            "uniform": UniformControl(),
            "gradient_based": GradientBasedControl(),
            "parameter_sensitivity": ParameterSensitivityControl(),
            "adaptive": AdaptiveControl()
        }
        
        self.pdg_generators = {
            "random": generate_random_pdg,
            "chain": PDGGenerator.generate_chain_pdg,
            "tree": PDGGenerator.generate_tree_pdg,
            "fully_connected": PDGGenerator.generate_fully_connected_pdg,
            "sparse": PDGGenerator.generate_sparse_pdg
        }
    
    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResults:
        """Run a single experiment with the given configuration."""
        start_time = time.time()
        
        # Generate PDG
        if config.attention_strategy in ["random"]:
            pdg = self.pdg_generators["random"](
                num_vars=config.num_vars,
                num_edges=config.num_edges,
                val_range=config.val_range,
                src_range=config.src_range,
                tgt_range=config.tgt_range,
                seed=config.seed
            )
        else:
            # Use default random generator for now
            pdg = self.pdg_generators["random"](
                num_vars=config.num_vars,
                num_edges=config.num_edges,
                val_range=config.val_range,
                src_range=config.src_range,
                tgt_range=config.tgt_range,
                seed=config.seed
            )
        
        # Make PDG parametric
        pdg = make_every_cpd_parametric_projections_fixed(pdg, init=config.init_method)
        
        # Get strategies
        attention_strategy = self.attention_strategies.get(config.attention_strategy, UniformAttention())
        control_strategy = self.control_strategies.get(config.control_strategy, UniformControl())
        
        # Initialize
        mu0 = opt_joint(pdg, gamma=config.gamma, iters=25, verbose=False)
        
        # Track loss history
        loss_history = []
        
        # Custom LIR training with attention and control
        learnables = [(l, P) for l, P in pdg.edges("l,P") if hasattr(P, 'logits')]
        opt = config.optimizer_ctor([P.logits for (_, P) in learnables], lr=config.lr)
        
        mu_init = mu0
        for t in range(config.T):
            # Inner solve for μ*
            def warm_start_init(shape, dtype=torch.double):
                if mu_init is not None:
                    return mu_init.data.clone().to(dtype)
                else:
                    return torch.ones(shape, dtype=dtype)
            
            μ_star = opt_joint(pdg, gamma=config.gamma, iters=config.inner_iters, 
                             verbose=False, init=warm_start_init)
            μ_star = μ_star.detach().clone()
            mu_init = μ_star.data.detach().clone()
            
            # Compute attention and control
            attention = attention_strategy.compute_attention(pdg, μ_star, config.gamma)
            control = control_strategy.compute_control(pdg, torch.tensor([]), μ_star)  # Simplified for now
            
            # Compute loss and gradients
            opt.zero_grad(set_to_none=True)
            loss = torch_score(pdg, μ_star, config.gamma)
            loss.backward()
            
            # Apply attention and control (simplified implementation)
            # In practice, this would modify the gradient computation
            opt.step()
            
            loss_history.append(float(loss.detach().cpu()))
        
        # Final evaluation
        mu_star = opt_joint(pdg, gamma=config.gamma, iters=config.eval_iters, verbose=False)
        final_loss = float(torch_score(pdg, mu_star, config.gamma).detach().cpu())
        
        training_time = time.time() - start_time
        
        # Compute convergence metrics
        convergence_metrics = MetricsCalculator.compute_convergence_metrics(loss_history)
        
        return ExperimentResults(
            config=config,
            final_loss=final_loss,
            convergence_iterations=convergence_metrics["iterations"],
            training_time=training_time,
            attention_weights=attention.tolist() if hasattr(attention, 'tolist') else [],
            control_weights=control.tolist() if hasattr(control, 'tolist') else [],
            loss_history=loss_history,
            convergence_achieved=convergence_metrics["converged"]
        )
    
    def run_experiment_suite(self, base_config: ExperimentConfig, 
                           attention_strategies: List[str] = None,
                           control_strategies: List[str] = None,
                           pdg_types: List[str] = None) -> List[ExperimentResults]:
        """Run a comprehensive suite of experiments."""
        
        if attention_strategies is None:
            attention_strategies = list(self.attention_strategies.keys())
        if control_strategies is None:
            control_strategies = list(self.control_strategies.keys())
        if pdg_types is None:
            pdg_types = ["random"]
        
        results = []
        
        for att_strategy in attention_strategies:
            for ctrl_strategy in control_strategies:
                for pdg_type in pdg_types:
                    for run in range(base_config.num_runs):
                        config = ExperimentConfig(
                            num_vars=base_config.num_vars,
                            num_edges=base_config.num_edges,
                            val_range=base_config.val_range,
                            src_range=base_config.src_range,
                            tgt_range=base_config.tgt_range,
                            gamma=base_config.gamma,
                            T=base_config.T,
                            inner_iters=base_config.inner_iters,
                            lr=base_config.lr,
                            optimizer_ctor=base_config.optimizer_ctor,
                            attention_strategy=att_strategy,
                            control_strategy=ctrl_strategy,
                            seed=base_config.seed + run,
                            init_method=base_config.init_method,
                            num_runs=base_config.num_runs,
                            eval_iters=base_config.eval_iters
                        )
                        
                        print(f"Running: {att_strategy} + {ctrl_strategy} + {pdg_type} (run {run+1})")
                        result = self.run_single_experiment(config)
                        results.append(result)
        
        return results
    
    def analyze_results(self, results: List[ExperimentResults]) -> Dict[str, Any]:
        """Analyze and summarize experimental results."""
        analysis = defaultdict(list)
        
        for result in results:
            key = f"{result.config.attention_strategy}_{result.config.control_strategy}"
            analysis[key].append({
                "final_loss": result.final_loss,
                "convergence_iterations": result.convergence_iterations,
                "training_time": result.training_time,
                "convergence_achieved": result.convergence_achieved
            })
        
        # Compute statistics for each strategy combination
        summary = {}
        for strategy, runs in analysis.items():
            final_losses = [r["final_loss"] for r in runs]
            convergence_iters = [r["convergence_iterations"] for r in runs]
            training_times = [r["training_time"] for r in runs]
            convergence_rates = [r["convergence_achieved"] for r in runs]
            
            summary[strategy] = {
                "mean_final_loss": np.mean(final_losses),
                "std_final_loss": np.std(final_losses),
                "mean_convergence_iterations": np.mean(convergence_iters),
                "std_convergence_iterations": np.std(convergence_iters),
                "mean_training_time": np.mean(training_times),
                "std_training_time": np.std(training_times),
                "convergence_rate": np.mean(convergence_rates),
                "num_runs": len(runs)
            }
        
        return summary
    
    def save_results(self, results: List[ExperimentResults], analysis: Dict[str, Any]):
        """Save experimental results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = self.output_dir / f"results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert results to serializable format
            serializable_results = []
            for result in results:
                serializable_results.append({
                    "config": {
                        "num_vars": result.config.num_vars,
                        "num_edges": result.config.num_edges,
                        "gamma": result.config.gamma,
                        "T": result.config.T,
                        "attention_strategy": result.config.attention_strategy,
                        "control_strategy": result.config.control_strategy,
                        "seed": result.config.seed
                    },
                    "final_loss": result.final_loss,
                    "convergence_iterations": result.convergence_iterations,
                    "training_time": result.training_time,
                    "loss_history": result.loss_history,
                    "convergence_achieved": result.convergence_achieved
                })
            json.dump(serializable_results, f, indent=2)
        
        # Save analysis
        analysis_file = self.output_dir / f"analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Results saved to {results_file}")
        print(f"Analysis saved to {analysis_file}")
    
    def plot_results(self, results: List[ExperimentResults], analysis: Dict[str, Any]):
        """Create visualization plots for the results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Plot 1: Final loss comparison
        plt.figure(figsize=(12, 8))
        strategies = list(analysis.keys())
        mean_losses = [analysis[s]["mean_final_loss"] for s in strategies]
        std_losses = [analysis[s]["std_final_loss"] for s in strategies]
        
        plt.bar(strategies, mean_losses, yerr=std_losses, capsize=5)
        plt.title("Final Loss Comparison Across Strategies")
        plt.xlabel("Strategy Combination")
        plt.ylabel("Final Loss")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"final_loss_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Convergence rate comparison
        plt.figure(figsize=(12, 8))
        convergence_rates = [analysis[s]["convergence_rate"] for s in strategies]
        
        plt.bar(strategies, convergence_rates)
        plt.title("Convergence Rate Comparison")
        plt.xlabel("Strategy Combination")
        plt.ylabel("Convergence Rate")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"convergence_rate_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Training time comparison
        plt.figure(figsize=(12, 8))
        mean_times = [analysis[s]["mean_training_time"] for s in strategies]
        std_times = [analysis[s]["std_training_time"] for s in strategies]
        
        plt.bar(strategies, mean_times, yerr=std_times, capsize=5)
        plt.title("Training Time Comparison")
        plt.xlabel("Strategy Combination")
        plt.ylabel("Training Time (seconds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"training_time_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {self.output_dir}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def run_example_experiments():
    """Run example experiments to demonstrate the framework."""
    
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Define base configuration
    base_config = ExperimentConfig(
        num_vars=3,
        num_edges=3,
        gamma=1.0,
        T=20,
        inner_iters=15,
        lr=1e-2,
        num_runs=3
    )
    
    # Run experiments
    print("Running experimental suite...")
    results = runner.run_experiment_suite(
        base_config=base_config,
        attention_strategies=["uniform", "entropy_based"],
        control_strategies=["uniform", "gradient_based"],
        pdg_types=["random"]
    )
    
    # Analyze results
    print("Analyzing results...")
    analysis = runner.analyze_results(results)
    
    # Save and plot results
    runner.save_results(results, analysis)
    runner.plot_results(results, analysis)
    
    # Print summary
    print("\n=== EXPERIMENTAL RESULTS SUMMARY ===")
    for strategy, stats in analysis.items():
        print(f"\n{strategy}:")
        print(f"  Mean Final Loss: {stats['mean_final_loss']:.6f} ± {stats['std_final_loss']:.6f}")
        print(f"  Convergence Rate: {stats['convergence_rate']:.2f}")
        print(f"  Mean Training Time: {stats['mean_training_time']:.2f}s")
    
    return results, analysis


if __name__ == "__main__":
    # Run example experiments
    results, analysis = run_example_experiments()

