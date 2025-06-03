"""
Graph-based model for fake news detection.
"""
import torch
from torch_geometric.nn import GCNConv

class FakeNewsDetector(torch.nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int, dropout: float = 0.5):
        """
        Initialize the model.
        
        Args:
            num_node_features: Number of input features per node
            hidden_channels: Number of hidden channels
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.classifier = torch.nn.Linear(hidden_channels, 2)  # 2 classes: real/fake
        
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            Log probabilities for each class
        """
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # Classification layer
        x = self.classifier(x)
        
        return torch.log_softmax(x, dim=1) 