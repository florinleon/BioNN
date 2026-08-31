# BioNN Model

BioNN is a neural network model designed to study how biologically inspired regulatory mechanisms shape hidden representations and learning behavior. Each hidden layer uses a standard weighted transformation followed by ReLU activation, with optional mechanisms that modify how signals are processed. 

These mechanisms include input gating, gain modulation, threshold modulation, lateral inhibition, homeostatic threshold adaptation, structural plasticity, and activation decorrelation. Input gating changes the contribution of individual input coordinates, gain modulation adjusts unit responsiveness, and threshold modulation shifts activation thresholds. Lateral inhibition introduces competition between hidden units, while homeostasis adapts slow thresholds according to recent activity. Structural plasticity masks a configurable fraction of connections and can update that mask from learned weight magnitudes. Activation decorrelation adds a penalty that discourages redundant hidden-unit activity. A shared context pathway drives the modulation mechanisms, while fixed initialization streams keep comparable parameters consistent across configurations. This design makes each switch interpretable as a distinct computational contribution within the same underlying network.

The repository includes XOR-cluster and sparse-parity experiments, repeated seeded runs, accuracy and sparsity measurements, hidden-representation metrics, and batch ablation utilities. Experiments compare all-on, all-off, one-on, one-off, two-on, two-off, and selected higher-order combinations, with centralized CSV summaries for systematic analysis and downstream comparative evaluation.

## Citation

A detailed description of the architecture and examples can be found in this paper:

> Florin Leon, *Biologically Inspired Mechanisms for Facilitating Grokking in Multilayer Perceptrons*, 2026, https://arxiv.org/abs/2608.28184 .

## Note

The implementations are intended as reference programs rather than optimized systems. The programs are distributed in the hope that they will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose.
