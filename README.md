# FakeNewsNet

A graph-based fake news detection model using PyTorch Geometric, with support for both traditional graph-based and BERT-enhanced approaches.

## Features

- Dual model architecture:
  - Graph-based approach using GCN
  - Hybrid BERT-Graph approach combining TinyBERT with GCN
- Efficient sparse graph structure with configurable connections
- Caching system for faster data loading
- Comprehensive analysis tools
- Class imbalance handling
- Advanced regularization techniques

## Installation

This project uses Poetry for dependency management. To install:

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/mowhammadrezaa/fakenewsnet.git
cd fakenewsnet

# Install dependencies
poetry install
```

## Dataset Structure

The dataset should be organized as follows:

```
dataset/
├── gossipcop_fake.csv
├── gossipcop_real.csv
├── politifact_fake.csv
├── politifact_real.csv
```

## Training

### Graph-based Model

Train the traditional graph-based model:

```bash
poetry run python -m fakenewsnet.train \
    --data_dir dataset \
    --cache_dir cache \
    --model_type graph \
    --save_model \
    --hidden_channels 64 \
    --learning_rate 0.01 \
    --epochs 100 \
    --patience 10 \
    --dropout 0.3 \
    --weight_decay 1e-5 \
    --grad_clip 1.0
```

### BERT-Graph Model

Train the hybrid BERT-Graph model:

```bash
poetry run python -m fakenewsnet.train \
    --data_dir dataset \
    --cache_dir cache \
    --model_type bert_graph \
    --save_model \
    --hidden_channels 64 \
    --learning_rate 0.01 \
    --epochs 100 \
    --patience 10 \
    --dropout 0.3 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --bert_model prajjwal1/bert-tiny \
    --freeze_bert
```

### Training Parameters

Common parameters for both models:
- `--data_dir`: Path to dataset directory (default: 'dataset')
- `--cache_dir`: Directory for cached data (default: 'cache')
- `--model_type`: Type of model to use ('graph' or 'bert_graph')
- `--hidden_channels`: Number of hidden channels (default: 64)
- `--learning_rate`: Learning rate (default: 0.01)
- `--epochs`: Number of training epochs (default: 100)
- `--patience`: Early stopping patience (default: -1, disabled)
- `--dropout`: Dropout rate (default: 0.3)
- `--weight_decay`: L2 regularization coefficient (default: 1e-5)
- `--l1_lambda`: L1 regularization coefficient (default: 0.0)
- `--grad_clip`: Gradient clipping threshold (default: 1.0)
- `--max_connections`: Maximum connections per node (default: 10)
- `--debug`: Enable debug mode with small dataset
- `--debug_size`: Samples per class in debug mode (default: 100)

BERT-Graph specific parameters:
- `--bert_model`: BERT model to use (default: 'prajjwal1/bert-tiny')
- `--freeze_bert`: Freeze BERT parameters during training

## Analysis

Generate analysis report and visualizations:

```bash
poetry run python -m fakenewsnet.analysis.analyze \
    --data_dir dataset \
    --cache_dir cache \
    --model_path models/model_graph.pt \  # or model_bert_graph.pt
    --output_dir analysis_output
```

## Model Architecture

### Graph-based Model
- Two GCN layers with batch normalization
- Dropout for regularization
- Class-weighted loss function for handling imbalance
- Early stopping based on validation loss
- Gradient clipping to prevent exploding gradients

### BERT-Graph Model
- TinyBERT for text feature extraction
- Feature projection layer to match GCN dimensions
- Two GCN layers with batch normalization
- Optional BERT parameter freezing
- Combined text and graph features for classification

### Class Imbalance Handling

Both models implement several techniques to handle class imbalance:

1. Class-weighted loss function
2. Batch normalization layers
3. Balanced regularization parameters
4. Early stopping based on validation loss

### Regularization Techniques

Multiple regularization methods are employed to prevent overfitting:

1. Dropout (default: 0.3)
2. L2 regularization via weight decay (default: 1e-5)
3. Optional L1 regularization
4. Gradient clipping (default: 1.0)
5. Batch normalization after each GCN layer

## Project Structure

```
fakenewsnet/
├── data/
│   └── loader.py
├── models/
│   ├── graph_model.py
│   ├── bert_graph_model.py
│   └── model_factory.py
├── analysis/
│   └── analyze.py
├── utils/
│   └── metrics.py
├── train.py
└── __init__.py
```

## Output Directories

- `cache/`: Cached processed data
- `models/`: Saved model checkpoints
  - `model_graph.pt`: Graph-based model
  - `model_bert_graph.pt`: BERT-Graph model
- `analysis_output/`: Analysis reports and visualizations

## License

MIT License 