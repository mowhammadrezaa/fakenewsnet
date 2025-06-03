"""
Data loader for FakeNewsNet dataset.
"""
import os
from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from torch.serialization import add_safe_globals

# Add PyTorch Geometric types to safe globals
add_safe_globals([
    'torch_geometric.data.data.Data',
    'torch_geometric.data.data.DataEdgeAttr',
    'torch_geometric.data.data.DataNodeAttr'
])

class FakeNewsNetLoader:
    def __init__(self, data_dir: str, cache_dir: str = None, debug: bool = False, debug_size: int = 100, max_connections: int = 10):
        """
        Initialize the FakeNewsNet data loader.
        
        Args:
            data_dir (str): Path to the FakeNewsNet dataset directory
            cache_dir (str, optional): Directory to cache processed data. If None, uses data_dir/cache
            debug (bool): If True, use a small subset of data for faster development
            debug_size (int): Number of samples to use in debug mode (per class)
            max_connections (int): Maximum number of connections per node
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.label_encoder = LabelEncoder()
        self.debug = debug
        self.debug_size = debug_size
        self.max_connections = max_connections
        
    def _create_sparse_edges(self, num_nodes: int) -> torch.Tensor:
        """
        Create a sparse edge index tensor with limited connections per node.
        
        Args:
            num_nodes (int): Number of nodes in the graph
            
        Returns:
            torch.Tensor: Edge index tensor of shape [2, num_edges]
        """
        # Initialize lists to store edges
        rows = []
        cols = []
        
        # For each node, create connections to max_connections other nodes
        for i in tqdm(range(num_nodes), desc="Creating edges"):
            # Get random indices for connections
            possible_connections = list(range(num_nodes))
            possible_connections.remove(i)  # Remove self-connection
            np.random.seed(i)  # For reproducibility
            selected_connections = np.random.choice(
                possible_connections,
                size=min(self.max_connections, len(possible_connections)),
                replace=False
            )
            
            # Add edges
            for j in selected_connections:
                rows.append(i)
                cols.append(j)
        
        # Convert to tensor
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        return edge_index
        
    def _get_cache_path(self) -> Path:
        """Get the path for cached data."""
        suffix = '_debug' if self.debug else ''
        return self.cache_dir / f'processed_data{suffix}.pt'
        
    def _save_to_cache(self, data: Data, metadata: Dict) -> None:
        """Save processed data and metadata to cache."""
        cache_data = {
            'data': data,
            'metadata': metadata,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder
        }
        torch.save(cache_data, self._get_cache_path(), _use_new_zipfile_serialization=False)
        print(f"Saved processed data to {self._get_cache_path()}")
        
    def _load_from_cache(self) -> Tuple[Data, Dict]:
        """Load processed data and metadata from cache."""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return None, None
            
        print(f"Loading cached data from {cache_path}")
        try:
            cache_data = torch.load(cache_path, weights_only=False)
        except Exception as e:
            print(f"Error loading cache: {e}")
            print("Attempting to load with legacy format...")
            cache_data = torch.load(cache_path, pickle_module=torch.serialization.pickle)
        
        # Restore vectorizer and label encoder state
        self.vectorizer = cache_data['vectorizer']
        self.label_encoder = cache_data['label_encoder']
        
        return cache_data['data'], cache_data['metadata']
        
    def load_data(self) -> Tuple[Data, Dict]:
        """
        Load and preprocess the FakeNewsNet dataset.
        Will use cached data if available, otherwise process from scratch.
        
        Returns:
            Tuple[Data, Dict]: Graph data and metadata
        """
        # Try to load from cache first
        data, metadata = self._load_from_cache()
        if data is not None:
            print("Using cached data")
            return data, metadata
            
        print("No cache found. Processing data from scratch...")
        # Load datasets
        gossipcop_real_df = pd.read_csv(self.data_dir / 'gossipcop_real.csv')
        gossipcop_fake_df = pd.read_csv(self.data_dir / 'gossipcop_fake.csv')
        politifact_real_df = pd.read_csv(self.data_dir / 'politifact_real.csv')
        politifact_fake_df = pd.read_csv(self.data_dir / 'politifact_fake.csv')
        
        # Add labels to each dataset
        gossipcop_real_df['label'] = 'real'
        gossipcop_fake_df['label'] = 'fake'
        politifact_real_df['label'] = 'real'
        politifact_fake_df['label'] = 'fake'
        
        if self.debug:
            print(f"Debug mode: Using {self.debug_size} samples per class")
            # Sample a small portion of data for each class
            gossipcop_real_df = gossipcop_real_df.sample(n=min(self.debug_size, len(gossipcop_real_df)), random_state=42)
            gossipcop_fake_df = gossipcop_fake_df.sample(n=min(self.debug_size, len(gossipcop_fake_df)), random_state=42)
            politifact_real_df = politifact_real_df.sample(n=min(self.debug_size, len(politifact_real_df)), random_state=42)
            politifact_fake_df = politifact_fake_df.sample(n=min(self.debug_size, len(politifact_fake_df)), random_state=42)
        
        # Combine datasets
        df = pd.concat([gossipcop_real_df, gossipcop_fake_df, politifact_real_df, politifact_fake_df], ignore_index=True)
        
        print("Extracting features...")
        # Extract features
        text_features = self.vectorizer.fit_transform(df['title']).toarray()
        
        # Encode labels
        labels = self.label_encoder.fit_transform(df['label'])
        
        # Create node features
        x = torch.FloatTensor(text_features)
        y = torch.LongTensor(labels)
        
        print("Creating graph structure...")
        # Create sparse edge index
        num_nodes = len(df)
        edge_index = self._create_sparse_edges(num_nodes)
        
        # Create batch (all nodes in one batch for now)
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y, batch=batch)
        
        # Create metadata
        metadata = {
            'num_nodes': num_nodes,
            'num_features': text_features.shape[1],
            'num_classes': len(self.label_encoder.classes_),
            'feature_names': self.vectorizer.get_feature_names_out().tolist(),
            'class_names': self.label_encoder.classes_.tolist(),
            'debug_mode': self.debug,
            'debug_size': self.debug_size if self.debug else None,
            'max_connections': self.max_connections
        }
        
        # Save to cache
        self._save_to_cache(data, metadata)
        
        return data, metadata
    
    def construct_propagation_graph(self) -> Data:
        """
        Construct propagation graph from retweet and reply relationships.
        
        Returns:
            Data: PyTorch Geometric Data object containing the graph
        """
        # For now, we'll use the same graph as in load_data
        # In a real implementation, this would use the actual retweet/reply relationships
        data, _ = self.load_data()
        return data 