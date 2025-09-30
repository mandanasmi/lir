# LIR Attention and Control Experimental Design

This document describes a comprehensive experimental framework for testing different approaches to attention (φ) and control (χ) selection in the Local Inconsistency Resolution (LIR) framework using random synthetic Probabilistic Dependency Graphs (PDGs).

## Overview

The LIR update rule is:
```
θ_new = exp_θ_old(-χ ⊙ ∇_θ ⟨⟨φ ⊙ M(θ)⟩⟩)
```

Where:
- **φ (attention)**: Determines which parts of the model to focus on during learning
- **χ (control)**: Determines how much to update each parameter
- **M(θ)**: The parametric model (PDG)
- **⟨⟨·⟩⟩**: The inconsistency measure

## Experimental Framework Components

### 1. Attention Strategies

The framework implements several attention selection strategies:

#### **Uniform Attention**
- Focuses equally on all parts of the model
- Baseline strategy for comparison
- `φ = uniform weights`

#### **Gradient Magnitude Attention**
- Focuses on parameters with large gradients
- Assumes parameters with large gradients need more attention
- `φ ∝ |∇_θ loss|`

#### **Entropy-Based Attention**
- Focuses on uncertain parts of the model
- Uses entropy of conditional distributions as attention weights
- `φ ∝ entropy(P(Y|X))`

#### **Inconsistency-Based Attention**
- Focuses on parts of the model with high local inconsistency
- Uses variance in conditional distributions as a proxy
- `φ ∝ var(P(Y|X))`

#### **Adaptive Attention**
- Dynamically adjusts attention based on learning progress
- Combines base strategy with time-varying component
- `φ = base_strategy × adaptive_factor(t)`

### 2. Control Strategies

The framework implements several control selection strategies:

#### **Uniform Control**
- Updates all parameters equally
- Baseline strategy for comparison
- `χ = uniform weights`

#### **Gradient-Based Control**
- Larger updates for parameters with smaller gradients
- Inverse relationship with gradient magnitude
- `χ ∝ 1/(|∇_θ loss| + ε)`

#### **Parameter Sensitivity Control**
- Updates based on parameter sensitivity to the objective
- Uses gradient magnitude as sensitivity measure
- `χ ∝ |∇_θ loss|`

#### **Adaptive Control**
- Dynamically adjusts control based on learning progress
- Combines base strategy with time-varying component
- `χ = base_strategy × adaptive_factor(t)`

### 3. PDG Generation

The framework supports multiple PDG topologies:

#### **Random PDG**
- Randomly generated edges between variables
- Configurable number of variables, edges, and domain sizes
- Default topology for most experiments

#### **Chain PDG**
- Linear chain structure: A → B → C → D
- Tests sequential dependencies

#### **Tree PDG**
- Tree structure with root and children
- Tests hierarchical dependencies

#### **Fully Connected PDG**
- All variables connected to all other variables
- Tests complex interdependencies

#### **Sparse PDG**
- Controlled sparsity with configurable edge density
- Tests sparse dependency structures

### 4. Evaluation Metrics

The framework tracks multiple evaluation metrics:

#### **Convergence Metrics**
- Final loss value
- Number of iterations to convergence
- Convergence rate (percentage of runs that converged)
- Loss reduction from initial to final

#### **Stability Metrics**
- Variance in loss during training
- Oscillation in loss trajectory
- Training stability score

#### **Efficiency Metrics**
- Training time per run
- Computational efficiency
- Memory usage (if available)

## Usage

### Quick Start

Run a quick experiment to test the framework:

```bash
python run_experiments.py --quick
```

### Full Experimental Suite

Run comprehensive experiments with all strategy combinations:

```bash
python run_experiments.py --full
```

### Custom Experiments

Run experiments with specific strategy combinations:

```bash
# Test specific attention and control strategies
python run_experiments.py --custom \
    --attention uniform,entropy_based \
    --control uniform,gradient_based

# Customize PDG parameters
python run_experiments.py --custom \
    --attention uniform,gradient_magnitude \
    --control uniform,parameter_sensitivity \
    --vars 5 --edges 6 --runs 10
```

### Available Strategies

**Attention Strategies:**
- `uniform`: Uniform attention weights
- `gradient_magnitude`: Based on gradient magnitudes
- `entropy_based`: Based on entropy of distributions
- `inconsistency_based`: Based on local inconsistency
- `adaptive`: Adaptive attention over time

**Control Strategies:**
- `uniform`: Uniform control weights
- `gradient_based`: Based on gradient magnitudes
- `parameter_sensitivity`: Based on parameter sensitivity
- `adaptive`: Adaptive control over time

## Experimental Design

### Hypothesis Testing

The experiments are designed to test several hypotheses:

1. **Attention Hypothesis**: Different attention strategies will lead to different convergence properties and final performance.

2. **Control Hypothesis**: Different control strategies will affect the stability and efficiency of learning.

3. **Interaction Hypothesis**: The combination of attention and control strategies will have non-additive effects.

4. **Topology Hypothesis**: Different PDG topologies will respond differently to attention and control strategies.

### Experimental Variables

#### **Independent Variables**
- Attention strategy (5 levels)
- Control strategy (4 levels)
- PDG topology (5 types)
- PDG complexity (number of variables, edges)
- Learning parameters (learning rate, iterations)

#### **Dependent Variables**
- Final loss value
- Convergence time
- Training stability
- Computational efficiency

#### **Control Variables**
- Random seed (for reproducibility)
- Initialization method
- Optimizer choice
- Hardware/software environment

### Statistical Analysis

The framework provides:

1. **Descriptive Statistics**: Mean, standard deviation, confidence intervals
2. **Comparative Analysis**: Pairwise comparisons between strategies
3. **Significance Testing**: Statistical tests for differences between strategies
4. **Effect Size Analysis**: Magnitude of differences between strategies

## Output and Results

### Generated Files

Each experiment run generates:

1. **Raw Results** (`results_TIMESTAMP.json`): Complete experimental data
2. **Analysis** (`analysis_TIMESTAMP.json`): Statistical analysis summary
3. **Plots**:
   - Final loss comparison
   - Convergence rate comparison
   - Training time comparison
   - Loss trajectories (if enabled)

### Result Interpretation

#### **Performance Ranking**
Strategies are ranked by:
1. Mean final loss (lower is better)
2. Convergence rate (higher is better)
3. Training time (lower is better)

#### **Statistical Significance**
Results include:
- Confidence intervals for all metrics
- Statistical significance tests
- Effect size calculations

## Extending the Framework

### Adding New Attention Strategies

1. Inherit from `AttentionStrategy` class
2. Implement `compute_attention()` method
3. Register in `ExperimentRunner.attention_strategies`

```python
class MyAttentionStrategy(AttentionStrategy):
    def compute_attention(self, pdg, mu_star, gamma):
        # Your attention computation logic
        return attention_weights
```

### Adding New Control Strategies

1. Inherit from `ControlStrategy` class
2. Implement `compute_control()` method
3. Register in `ExperimentRunner.control_strategies`

```python
class MyControlStrategy(ControlStrategy):
    def compute_control(self, pdg, gradients, mu_star):
        # Your control computation logic
        return control_weights
```

### Adding New PDG Topologies

1. Add generator method to `PDGGenerator` class
2. Register in `ExperimentRunner.pdg_generators`

```python
@staticmethod
def generate_my_topology(**kwargs):
    # Your PDG generation logic
    return pdg
```

## Example Results

### Typical Output

```
📊 EXPERIMENTAL RESULTS SUMMARY
============================================================
Strategy                         Final Loss      Conv Rate  Time (s)
-----------------------------------------------------------------
uniform_uniform                  0.123456±0.001  0.80      12.34
entropy_based_gradient_based     0.098765±0.002  0.90      15.67
gradient_magnitude_uniform       0.112345±0.003  0.75      11.23
...

🏆 Best performing strategy: entropy_based_gradient_based
   Final Loss: 0.098765
   Convergence Rate: 0.90
```

### Key Findings (Example)

1. **Entropy-based attention** often performs best for complex PDGs
2. **Gradient-based control** provides good stability
3. **Adaptive strategies** show promise but need tuning
4. **PDG topology** significantly affects strategy performance

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PDG submodule is properly initialized
2. **Memory Issues**: Reduce number of variables or runs
3. **Convergence Issues**: Adjust learning rate or iterations
4. **Plot Generation**: Install matplotlib and seaborn

### Performance Tips

1. Use `--quick` for initial testing
2. Start with fewer runs (`--runs 2`) for development
3. Use smaller PDGs (`--vars 3 --edges 3`) for faster iteration
4. Enable parallel processing for large experiments (future enhancement)

## Future Enhancements

### Planned Features

1. **Parallel Execution**: Run multiple experiments in parallel
2. **Advanced Visualizations**: Interactive plots and dashboards
3. **Hyperparameter Optimization**: Automatic tuning of learning parameters
4. **Real-world Datasets**: Integration with actual PDG datasets
5. **Advanced Metrics**: More sophisticated evaluation criteria

### Research Directions

1. **Theoretical Analysis**: Mathematical analysis of attention/control strategies
2. **Adaptive Strategies**: More sophisticated adaptive mechanisms
3. **Multi-objective Optimization**: Balancing multiple performance criteria
4. **Transfer Learning**: How strategies transfer across different PDG types

## References

1. Richardson, O. (2020). Probabilistic Dependency Graphs. arXiv:2012.10800
2. LIR Project Documentation
3. Local Inconsistency Resolution Theory

## Contact

For questions about this experimental framework, please refer to the main LIR project documentation or create an issue in the project repository.

