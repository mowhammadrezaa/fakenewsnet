"""
Analysis module for fake news detection model.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch_geometric.data import Data
import networkx as nx
from tqdm import tqdm

from fakenewsnet.data.loader import FakeNewsNetLoader
from fakenewsnet.models.graph_model import FakeNewsDetector

class ModelAnalyzer:
    def __init__(self, model_path: str, data_dir: str, cache_dir: str, device: str = 'cuda'):
        """
        Initialize the model analyzer.
        
        Args:
            model_path: Path to the trained model
            data_dir: Path to the dataset directory
            cache_dir: Path to the cache directory
            device: Device to run the model on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        
        # Load data
        self.loader = FakeNewsNetLoader(data_dir=data_dir, cache_dir=cache_dir)
        self.graph_data, self.metadata = self.loader.load_data()
        
        # Split data into train/val/test
        self._split_data()
        
        # Load model
        self.model = self._load_model()
        
    def _split_data(self):
        """Split data into train/val/test sets."""
        num_nodes = self.graph_data.num_nodes
        indices = torch.randperm(num_nodes)
        
        # 60% train, 20% val, 20% test
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        self.train_mask[indices[:train_size]] = True
        self.val_mask[indices[train_size:train_size + val_size]] = True
        self.test_mask[indices[train_size + val_size:]] = True
        
        # Move data to device
        self.graph_data = self.graph_data.to(self.device)
        self.train_mask = self.train_mask.to(self.device)
        self.val_mask = self.val_mask.to(self.device)
        self.test_mask = self.test_mask.to(self.device)
        
    def _load_model(self) -> FakeNewsDetector:
        """Load the trained model."""
        model = FakeNewsDetector(
            num_node_features=self.graph_data.num_node_features,
            hidden_channels=64  # Should match training
        ).to(self.device)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model
    
    def get_predictions(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get model predictions and true labels for all splits."""
        with torch.no_grad():
            pred = self.model(self.graph_data.x, self.graph_data.edge_index)
            pred = pred.argmax(dim=1)
            
            predictions = {
                'train': (
                    pred[self.train_mask].cpu().numpy(),
                    self.graph_data.y[self.train_mask].cpu().numpy()
                ),
                'val': (
                    pred[self.val_mask].cpu().numpy(),
                    self.graph_data.y[self.val_mask].cpu().numpy()
                ),
                'test': (
                    pred[self.test_mask].cpu().numpy(),
                    self.graph_data.y[self.test_mask].cpu().numpy()
                )
            }
            return predictions
    
    def analyze_performance(self, output_dir: str) -> Dict:
        """
        Analyze model performance and generate visualizations.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            Dict containing performance metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get predictions for all splits
        predictions = self.get_predictions()
        
        # Calculate metrics for each split
        metrics = {}
        for split_name, (y_pred, y_true) in predictions.items():
            metrics[split_name] = {
                'confusion_matrix': confusion_matrix(y_true, y_pred),
                'classification_report': classification_report(y_true, y_pred, output_dict=True)
            }
            
            # Generate visualizations for each split
            self._plot_confusion_matrix(
                metrics[split_name]['confusion_matrix'],
                output_dir,
                split_name
            )
        
        # Generate other visualizations
        self._plot_class_distribution(
            self.graph_data.y.cpu().numpy(),
            output_dir
        )
        self._plot_feature_importance(output_dir)
        self._plot_graph_structure(output_dir)
        
        return metrics
    
    def _plot_confusion_matrix(self, cm: np.ndarray, output_dir: Path, split_name: str):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {split_name.capitalize()} Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(output_dir / f'confusion_matrix_{split_name}.png')
        plt.close()
    
    def _plot_class_distribution(self, y_true: np.ndarray, output_dir: Path):
        """Plot class distribution."""
        plt.figure(figsize=(8, 6))
        sns.countplot(x=y_true)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Real', 'Fake'])
        plt.savefig(output_dir / 'class_distribution.png')
        plt.close()
    
    def _plot_feature_importance(self, output_dir: Path):
        """Plot feature importance based on model weights."""
        # Get feature importance from the first layer
        weights = self.model.conv1.lin.weight.data.cpu().numpy()
        importance = np.abs(weights).mean(axis=0)
        
        # Get top features
        top_n = 20
        top_indices = np.argsort(importance)[-top_n:]
        top_features = [self.metadata['feature_names'][i] for i in top_indices]
        top_importance = importance[top_indices]
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_importance, y=top_features)
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Average Weight Magnitude')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png')
        plt.close()
    
    def _plot_graph_structure(self, output_dir: Path):
        """Plot graph structure and node embeddings."""
        # Convert to NetworkX graph
        edge_index = self.graph_data.edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_edges_from(edge_index.T)
        
        # Get node embeddings from the last layer
        with torch.no_grad():
            x = self.model.conv1(self.graph_data.x, self.graph_data.edge_index)
            x = torch.relu(x)
            x = self.model.conv2(x, self.graph_data.edge_index)
            embeddings = x.cpu().numpy()
        
        # Plot graph structure
        plt.figure(figsize=(12, 8))
        
        # Sample the graph if it's too large
        max_nodes = 1000
        if G.number_of_nodes() > max_nodes:
            # Sample nodes while preserving class balance
            real_nodes = [n for n in G.nodes() if self.graph_data.y[n].item() == 0]
            fake_nodes = [n for n in G.nodes() if self.graph_data.y[n].item() == 1]
            
            sample_size = max_nodes // 2
            sampled_real = np.random.choice(real_nodes, min(sample_size, len(real_nodes)), replace=False)
            sampled_fake = np.random.choice(fake_nodes, min(sample_size, len(fake_nodes)), replace=False)
            
            sampled_nodes = np.concatenate([sampled_real, sampled_fake])
            G = G.subgraph(sampled_nodes)
        
        # Use a faster layout algorithm
        try:
            # Try spectral layout first (faster for large graphs)
            pos = nx.spectral_layout(G)
        except:
            # Fall back to random layout if spectral fails
            pos = nx.random_layout(G)
        
        # Draw the graph
        node_colors = [self.graph_data.y[n].item() for n in G.nodes()]
        nx.draw(G, pos, node_size=50, alpha=0.6, 
                node_color=node_colors,
                cmap='coolwarm')
        plt.title('Graph Structure (Node Colors: Real/Fake)')
        plt.savefig(output_dir / 'graph_structure.png')
        plt.close()
        
        # Plot node embeddings
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings[:, 0], embeddings[:, 1],
                   c=self.graph_data.y.cpu().numpy(),
                   cmap='coolwarm', alpha=0.6)
        plt.title('Node Embeddings (t-SNE)')
        plt.colorbar(label='Class (0: Real, 1: Fake)')
        plt.savefig(output_dir / 'node_embeddings.png')
        plt.close()
    
    def generate_report(self, output_dir: str) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            output_dir: Directory to save the report and visualizations
            
        Returns:
            Path to the generated report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get performance metrics
        metrics = self.analyze_performance(output_dir)
        
        # Generate report
        report = []
        report.append("# Fake News Detection Model Analysis Report\n")
        
        # Model Performance for each split
        for split_name in ['train', 'val', 'test']:
            report.append(f"## {split_name.capitalize()} Set Performance\n")
            report.append("### Classification Report\n")
            report.append("```")
            report.append(classification_report(
                self.graph_data.y[self._get_mask(split_name)].cpu().numpy(),
                self.get_predictions()[split_name][0],
                target_names=['Real', 'Fake']
            ))
            report.append("```\n")
        
        # Graph Structure Analysis
        report.append("## Graph Structure Analysis\n")
        report.append(f"- Total number of nodes: {self.metadata['num_nodes']}")
        report.append(f"- Number of edges: {self.graph_data.edge_index.size(1)}")
        report.append(f"- Average node degree: {self.graph_data.edge_index.size(1) / self.metadata['num_nodes']:.2f}\n")
        
        # Data Split Information
        report.append("## Data Split Information\n")
        report.append(f"- Training set size: {self.train_mask.sum().item()}")
        report.append(f"- Validation set size: {self.val_mask.sum().item()}")
        report.append(f"- Test set size: {self.test_mask.sum().item()}\n")
        
        # Feature Importance
        report.append("## Feature Importance Analysis\n")
        report.append("The top 20 most important features for classification are shown in the feature_importance.png visualization.\n")
        
        # Save report
        report_path = output_dir / 'analysis_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        return report_path
    
    def _get_mask(self, split_name: str) -> torch.Tensor:
        """Get the mask for a specific split."""
        return getattr(self, f'{split_name}_mask')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze fake news detection model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='dataset',
                      help='Path to the dataset directory')
    parser.add_argument('--cache_dir', type=str, default='cache',
                      help='Path to the cache directory')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                      help='Directory to save analysis results')
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ModelAnalyzer(
        model_path=args.model_path,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir
    )
    
    # Generate report
    report_path = analyzer.generate_report(args.output_dir)
    print(f"Analysis report generated at: {report_path}")

if __name__ == '__main__':
    main()