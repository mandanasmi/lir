# LIR Attention and Control Experimental Design - Summary

## Overview

I have designed and implemented a comprehensive experimental framework for testing different approaches to attention (φ) and control (χ) selection in the Local Inconsistency Resolution (LIR) framework using random synthetic Probabilistic Dependency Graphs (PDGs).

## What Was Created

### 1. Core Experimental Framework (`experimental_design.py`)

**Attention Strategies Implemented:**
- **Uniform Attention**: Baseline strategy focusing equally on all model parts
- **Gradient Magnitude Attention**: Focus on parameters with large gradients
- **Entropy-Based Attention**: Focus on uncertain parts using entropy of distributions
- **Inconsistency-Based Attention**: Focus on high local inconsistency areas
- **Adaptive Attention**: Dynamic attention that changes over time

**Control Strategies Implemented:**
- **Uniform Control**: Baseline strategy updating all parameters equally
- **Gradient-Based Control**: Larger updates for parameters with smaller gradients
- **Parameter Sensitivity Control**: Updates based on parameter sensitivity
- **Adaptive Control**: Dynamic control that adjusts over time

**PDG Generation Capabilities:**
- **Random PDG**: Randomly generated edges with configurable parameters
- **Chain PDG**: Linear chain structure (A → B → C → D)
- **Tree PDG**: Hierarchical tree structure
- **Fully Connected PDG**: All variables connected to all others
- **Sparse PDG**: Controlled sparsity with configurable edge density

**Evaluation Metrics:**
- Convergence metrics (final loss, iterations to convergence, convergence rate)
- Stability metrics (variance, oscillation, stability score)
- Efficiency metrics (training time, computational efficiency)

### 2. Command-Line Interface (`run_experiments.py`)

**Usage Options:**
```bash
# Quick test
python run_experiments.py --quick

# Full experimental suite
python run_experiments.py --full

# Custom experiments
python run_experiments.py --custom --attention uniform,entropy_based --control uniform,gradient_based
```

**Features:**
- Flexible parameter configuration
- Multiple experiment types (quick, full, custom)
- Comprehensive result analysis and visualization
- Statistical significance testing

### 3. Example and Documentation

**Example Script (`example_experiment.py`):**
- Demonstrates basic framework usage
- Shows how to implement custom strategies
- Provides advanced metrics analysis examples
- Tests different PDG topologies

**Comprehensive Documentation (`EXPERIMENTAL_DESIGN_README.md`):**
- Detailed explanation of all components
- Usage instructions and examples
- Extension guidelines for new strategies
- Troubleshooting and performance tips

**Requirements File (`requirements.txt`):**
- All necessary dependencies
- Version specifications
- Optional development dependencies

## Key Features of the Experimental Design

### 1. **Modular Architecture**
- Abstract base classes for easy extension
- Strategy registry system
- Configurable experiment parameters
- Pluggable PDG generators

### 2. **Comprehensive Evaluation**
- Multiple evaluation metrics
- Statistical analysis with confidence intervals
- Automated result visualization
- Performance ranking and comparison

### 3. **Reproducibility**
- Seed-based random generation
- Configurable experiment parameters
- Detailed result logging
- Version-controlled dependencies

### 4. **Extensibility**
- Easy addition of new attention/control strategies
- Support for new PDG topologies
- Customizable evaluation metrics
- Modular experimental runner

## Experimental Hypotheses to Test

### 1. **Attention Hypothesis**
Different attention strategies will lead to different convergence properties and final performance.

**Test:** Compare uniform vs. entropy-based vs. gradient-magnitude attention across various PDG topologies.

### 2. **Control Hypothesis**
Different control strategies will affect the stability and efficiency of learning.

**Test:** Compare uniform vs. gradient-based vs. parameter-sensitivity control strategies.

### 3. **Interaction Hypothesis**
The combination of attention and control strategies will have non-additive effects.

**Test:** Full factorial design testing all combinations of attention and control strategies.

### 4. **Topology Hypothesis**
Different PDG topologies will respond differently to attention and control strategies.

**Test:** Compare strategy performance across chain, tree, fully-connected, and sparse PDGs.

## How to Use the Framework

### Quick Start
```bash
cd /Users/mandanasamiei/Documents/GitHub/lir/code
python run_experiments.py --quick
```

### Custom Experiments
```bash
python run_experiments.py --custom \
    --attention uniform,entropy_based,gradient_magnitude \
    --control uniform,gradient_based,parameter_sensitivity \
    --vars 5 --edges 6 --runs 10
```

### Example Usage
```bash
python example_experiment.py
```

## Expected Outcomes

### 1. **Performance Insights**
- Which attention strategies work best for different PDG types
- How control strategies affect learning stability
- Optimal combinations of attention and control

### 2. **Theoretical Understanding**
- Relationship between attention mechanisms and convergence
- Impact of control strategies on parameter updates
- Interaction effects between attention and control

### 3. **Practical Guidelines**
- Recommendations for strategy selection
- Parameter tuning guidelines
- Performance optimization tips

## Files Created

1. **`experimental_design.py`** - Core experimental framework
2. **`run_experiments.py`** - Command-line interface
3. **`example_experiment.py`** - Usage examples and demonstrations
4. **`EXPERIMENTAL_DESIGN_README.md`** - Comprehensive documentation
5. **`EXPERIMENTAL_SUMMARY.md`** - This summary document
6. **`requirements.txt`** - Dependencies

## Next Steps

### 1. **Run Initial Experiments**
```bash
python run_experiments.py --quick
```

### 2. **Analyze Results**
- Review generated plots and statistics
- Identify promising strategy combinations
- Plan follow-up experiments

### 3. **Extend Framework**
- Add new attention/control strategies
- Implement additional PDG topologies
- Enhance evaluation metrics

### 4. **Scale Up**
- Run full experimental suite
- Test on larger PDGs
- Implement parallel processing

## Technical Notes

### Dependencies
- Requires the PDG submodule to be properly initialized
- Uses PyTorch for gradient computations
- Matplotlib/Seaborn for visualization
- NumPy for numerical computations

### Performance Considerations
- Start with small PDGs for initial testing
- Use fewer runs during development
- Enable parallel processing for large experiments (future enhancement)

### Memory Usage
- Larger PDGs require more memory
- Multiple runs can accumulate memory usage
- Consider reducing batch sizes for large experiments

## Conclusion

This experimental framework provides a comprehensive, extensible, and well-documented system for testing different attention and control selection approaches in the LIR framework. It enables systematic evaluation of various strategies using random synthetic PDGs, with robust statistical analysis and visualization capabilities.

The framework is ready for immediate use and can be easily extended with new strategies, topologies, and evaluation metrics as research progresses.

