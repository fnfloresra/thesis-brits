# BRITS: Bidirectional Recurrent Imputation for Time Series

A PyTorch implementation of the BRITS model for multivariate time series imputation.

## Overview

BRITS (Bidirectional Recurrent Imputation for Time Series) is a deep learning method for imputing missing values in multivariate time series data. The model uses bidirectional recurrent neural networks with the following key components:

- **Temporal Decay**: Models how the relevance of past observations decreases over time
- **Feature Regression**: Estimates missing values based on correlations with other features
- **Bidirectional Processing**: Leverages information from both past and future observations
- **Consistency Loss**: Encourages agreement between forward and backward imputations

## Installation

```bash
# Clone the repository
git clone https://github.com/fnfloresra/thesis-brits.git
cd thesis-brits

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training with Synthetic Data

```bash
# Quick test with synthetic data
python train.py --synthetic --epochs 10 --batch_size 32

# Full training with synthetic data
python train.py --synthetic --epochs 100 --batch_size 64 --hid_size 108
```

### Using the Model in Python

```python
import torch
from brits import BRITS

# Create model
model = BRITS(
    input_size=35,      # Number of features
    rnn_hid_size=64,    # RNN hidden size
    impute_weight=1.0,  # Weight for imputation loss
    seq_len=48,         # Sequence length
)

# Prepare data (batch_size, seq_len, n_features)
values = torch.randn(32, 48, 35)  # Your time series data
masks = torch.ones(32, 48, 35)    # 1 for observed, 0 for missing
deltas = torch.ones(32, 48, 35)   # Time gaps

# Forward pass
result = model(values=values, masks=masks, deltas=deltas)

# Get imputed values
imputations = result['imputations']
```

### Custom Dataset

```python
import numpy as np
from brits.data import TimeSeriesDataset, get_data_loader

# Your data with missing values as NaN
data = np.random.randn(1000, 48, 35).astype(np.float32)
data[np.random.rand(*data.shape) < 0.2] = np.nan  # 20% missing

# Create dataset and data loader
dataset = TimeSeriesDataset(data)
data_loader = get_data_loader(dataset, batch_size=64)
```

## Project Structure

```
thesis-brits/
├── brits/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── brits.py      # BRITS model (bidirectional)
│   │   └── rits.py       # RITS model (unidirectional)
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py     # Dataset and data loader
│   └── utils.py          # Utility functions
├── tests/
│   └── test_model.py     # Unit tests
├── train.py              # Training script
├── requirements.txt      # Dependencies
└── README.md
```

## Model Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | brits | Model type: brits or rits |
| `--hid_size` | 64 | RNN hidden size |
| `--impute_weight` | 1.0 | Weight for imputation loss |
| `--label_weight` | 0.0 | Weight for classification loss |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--patience` | 20 | Early stopping patience |

## Reference

This implementation is based on the paper:

> Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018).
> **BRITS: Bidirectional Recurrent Imputation for Time Series.**
> Advances in Neural Information Processing Systems, 31.
> [Paper Link](https://papers.nips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html)

## License

MIT License
