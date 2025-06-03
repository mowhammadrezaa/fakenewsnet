"""
Graph-based model for fake news detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class FakeNewsDetector(nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int = 64):
        """
        Initialize the fake news detection model.
        
        Args:
            num_node_features (int): Number of input node features
            hidden_channels (int): Number of hidden channels in the model
        """
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 2)  # Binary classification
        
    def forward(self, x, edge_index, batch):
        """
        Forward pass of the model.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            batch: Batch assignment for each node
            
        Returns:
            torch.Tensor: Model predictions
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1) 