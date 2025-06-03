# Fake News Detection Using Propagation Graphs

This project implements a graph-based approach for fake news detection using the FakeNewsNet dataset. The model utilizes propagation graphs constructed from retweet and reply relationships to classify news articles as either real or fake.

## Project Structure

```
fakenewsnet/
├── src/
│   └── fakenewsnet/
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── graph_model.py
│       ├── utils/
│       │   ├── __init__.py
│       │   └── metrics.py
│       └── __init__.py
├── tests/
└── dataset/
```

## Setup

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Download the FakeNewsNet dataset and place it in the `dataset` directory.

## Usage

[To be added]

## Features

- Data loading and preprocessing for FakeNewsNet dataset
- Propagation graph construction from retweet and reply relationships
- Graph-based model for fake news detection
- Evaluation metrics calculation (accuracy, precision, recall, F1-score)

## Dependencies

- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

## License

[To be added] 