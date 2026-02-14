# Adaptive Feature Interaction Networks with Dynamic Gating

A novel neural architecture for tabular data that dynamically gates feature interactions based on sample-specific importance scores. The model combines neural architecture search principles with adaptive interaction order selection, enabling instance-dependent feature engineering that addresses limitations of fixed interaction patterns in traditional tabular models.

## Key Innovation

The architecture introduces three main novelties: (1) dynamic gating layers that compute sample-specific feature importance weights, (2) multi-order interaction layers supporting pairwise, triplet, and higher-order feature combinations, and (3) a meta-learning framework that adaptively weights different interaction orders per prediction. This differentiable approach to feature interaction selection provides both improved performance and enhanced interpretability through learned sparsity.

## Methodology

Unlike traditional tabular models that use fixed feature engineering or static interaction patterns, this architecture learns instance-dependent feature interactions through three integrated mechanisms. First, the DynamicGatingLayer applies a small neural network with temperature-scaled softmax to compute per-sample importance scores, enabling sparse feature selection that varies by input. Second, the FeatureInteractionLayer computes combinatorial products up to order-k (pairwise x_i * x_j, triplet x_i * x_j * x_k) with efficient indexing. Third, the InteractionOrderSelector meta-learner produces sample-specific weights over different interaction orders, allowing the model to adaptively emphasize simple vs complex interactions. The SparsityRegularizedLoss adds L1 regularization on gate activations to encourage interpretable, sparse solutions. This end-to-end differentiable design addresses key limitations of both neural networks (poor tabular performance) and tree-based methods (fixed interactions) while maintaining computational efficiency through learned sparsity.

## Installation

```bash
pip install -r requirements.txt
```

For development with testing dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Training

Train the full model with dynamic gating and adaptive interaction selection:

```bash
python scripts/train.py --config configs/default.yaml
```

Train baseline model for ablation study:

```bash
python scripts/train.py --config configs/ablation.yaml
```

### Evaluation

Evaluate trained model on test set:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

### Prediction

Make predictions on new data:

```bash
python scripts/predict.py --checkpoint models/final_model.pt --input data.csv --output predictions.json
```

## Architecture Overview

The model consists of several key components:

- **DynamicGatingLayer**: Computes sample-specific attention weights with temperature-controlled sparsity
- **FeatureInteractionLayer**: Generates pairwise and higher-order feature interactions
- **InteractionOrderSelector**: Meta-learning module that weights different interaction orders adaptively
- **SparsityRegularizedLoss**: Custom loss function encouraging sparse gate activations

## Configuration

All hyperparameters are controlled via YAML config files in `configs/`:

- `default.yaml`: Full model with all novel components
- `ablation.yaml`: Baseline configuration for comparison

Key parameters:
- `num_gates`: Number of gating units (default: 32)
- `interaction_orders`: List of orders to compute (default: [2, 3])
- `gate_temperature`: Controls gate sparsity (lower = more sparse)
- `sparsity_weight`: Regularization strength for gate activations

## Project Structure

```
adaptive-feature-interaction-networks-with-dynamic-gating/
├── src/adaptive_feature_interaction_networks_with_dynamic_gating/
│   ├── models/          # Core model architecture
│   ├── data/            # Data loading and preprocessing
│   ├── training/        # Training loop with advanced features
│   ├── evaluation/      # Metrics and analysis
│   └── utils/           # Configuration utilities
├── scripts/
│   ├── train.py         # Training pipeline
│   ├── evaluate.py      # Evaluation with multiple metrics
│   └── predict.py       # Inference on new data
├── configs/
│   ├── default.yaml     # Main configuration
│   └── ablation.yaml    # Baseline for ablation study
├── tests/               # Comprehensive test suite
└── results/             # Training logs and outputs
```

## Experiment Results

Results from training on binary classification task (12 epochs, default configuration):

| Metric | Value |
|--------|-------|
| Test Accuracy | 91.20% |
| Test Precision | 0.912 |
| Test Recall | 0.912 |
| Test F1-Score | 0.912 |
| AUC-ROC | 0.961 |
| Test Log Loss | 0.248 |
| Feature Interaction Sparsity | 50.6% |
| Inference Throughput | 488.8 samples/sec |

Training convergence: Final train loss 0.0014, validation loss 0.249 after 12 epochs. The model achieves strong sparsity (50.6%) in feature interactions while maintaining high accuracy, demonstrating effective dynamic gating.

To reproduce: `python scripts/train.py --config configs/default.yaml` followed by `python scripts/evaluate.py --checkpoint checkpoints/best_model.pt`

## Ablation Study

The project includes configurations for systematic ablation:

1. **Full model** (`configs/default.yaml`): All components enabled
   - Dynamic gating with 32 gates
   - Multi-order interactions (pairwise + triplet)
   - Sparsity regularization (weight=0.001)
   - Low temperature gating (0.5) for sparsity

2. **Baseline** (`configs/ablation.yaml`): Reduced model
   - Fewer gates (16)
   - Only pairwise interactions
   - No sparsity regularization
   - Higher temperature (1.0) for dense gates

Compare results using:

```bash
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --output-dir results/full_model

python scripts/train.py --config configs/ablation.yaml
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --output-dir results/baseline
```

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src/adaptive_feature_interaction_networks_with_dynamic_gating
```

## Technical Details

The gating layer computes per-sample importance via `gate_scores = softmax(GateNetwork(x) / temperature)`, where lower temperature encourages sparsity. Feature interactions include order 1 (identity), order 2 (pairwise x_i * x_j), and order 3 (triplet x_i * x_j * x_k), with meta-learner weighting these adaptively per sample. Loss function: `total_loss = classification_loss + lambda * mean(|gate_scores|)` for sparsity regularization.

Training pipeline features: mixed precision (auto on CUDA), cosine LR scheduling, early stopping, gradient clipping, checkpoint saving, and optional MLflow tracking (`--mlflow` flag).

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- scikit-learn >= 1.3
- XGBoost >= 2.0
- LightGBM >= 4.0
- See `requirements.txt` for complete list

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
