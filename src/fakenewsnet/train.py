"""
Training script for the fake news detection model.
"""
import argparse
import os
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import numpy as np

from fakenewsnet.data.loader import FakeNewsNetLoader
from fakenewsnet.models.model_factory import ModelFactory
from fakenewsnet.utils.metrics import calculate_metrics

def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize data loader
    loader = FakeNewsNetLoader(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        debug=args.debug,
        debug_size=args.debug_size,
        max_connections=args.max_connections
    )
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    graph_data, metadata = loader.load_data()
    
    # Split data into train/val/test
    num_nodes = graph_data.num_nodes
    indices = torch.randperm(num_nodes)
    
    # 60% train, 20% val, 20% test
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Calculate class weights for balanced training
    train_labels = graph_data.y[train_mask]
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    
    print(f"Class distribution in training set: {class_counts.tolist()}")
    print(f"Class weights: {class_weights.tolist()}")
    
    # Move data to device
    graph_data = graph_data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    
    # Initialize model using factory
    model = ModelFactory.create_model(
        model_type=args.model_type,
        num_node_features=graph_data.num_node_features,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        bert_model_name=args.bert_model if args.model_type == 'bert_graph' else None,
        freeze_bert=args.freeze_bert if args.model_type == 'bert_graph' else None
    ).to(device)
    
    # Initialize optimizer with reduced weight decay
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Use weighted loss for class imbalance
    criterion = torch.nn.NLLLoss(weight=class_weights)
    
    # Training loop with early stopping
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass (with BERT processing if using BERT-Graph model)
        if args.model_type == 'bert_graph':
            out = model(graph_data.x, graph_data.edge_index, graph_data.texts)
        else:
            out = model(graph_data.x, graph_data.edge_index)
        
        # Calculate training loss
        train_loss = criterion(out[train_mask], graph_data.y[train_mask])
        
        # Add L1 regularization if specified (with reduced strength)
        if args.l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            train_loss = train_loss + args.l1_lambda * l1_norm
        
        # Backward pass
        train_loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            if args.model_type == 'bert_graph':
                val_out = model(graph_data.x, graph_data.edge_index, graph_data.texts)
            else:
                val_out = model(graph_data.x, graph_data.edge_index)
                
            val_loss = criterion(val_out[val_mask], graph_data.y[val_mask])
            
            # Calculate validation metrics
            val_pred = val_out[val_mask].argmax(dim=1)
            val_true = graph_data.y[val_mask]
            val_acc = (val_pred == val_true).float().mean()
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if args.patience != -1 and patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Print progress
        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"Training Loss: {train_loss.item():.4f}")
            print(f"Validation Loss: {val_loss.item():.4f}")
            print(f"Validation Accuracy: {val_acc.item():.4f}")
            
            # Print class distribution in predictions
            train_pred = out[train_mask].argmax(dim=1)
            train_dist = torch.bincount(train_pred, minlength=2)
            print(f"Training predictions distribution: {train_dist.tolist()}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    print("\nEvaluating model on test set...")
    model.eval()
    with torch.no_grad():
        if args.model_type == 'bert_graph':
            test_out = model(graph_data.x, graph_data.edge_index, graph_data.texts)
        else:
            test_out = model(graph_data.x, graph_data.edge_index)
            
        test_pred = test_out[test_mask].argmax(dim=1)
        test_true = graph_data.y[test_mask]
        
        metrics = calculate_metrics(
            y_true=test_true.cpu().numpy(),
            y_pred=test_pred.cpu().numpy()
        )
        
        print("\nTest Set Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        
        # Print class distribution in test predictions
        test_dist = torch.bincount(test_pred, minlength=2)
        print(f"Test predictions distribution: {test_dist.tolist()}")
    
    # Save model
    if args.save_model:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / f'model_{args.model_type}.pt')
        print(f"\nModel saved to {save_dir / f'model_{args.model_type}.pt'}")

def main():
    parser = argparse.ArgumentParser(description='Train fake news detection model')
    parser.add_argument('--data_dir', type=str, default='dataset',
                      help='Path to the dataset directory')
    parser.add_argument('--model_type', type=str, default='graph',
                      choices=['graph', 'bert_graph'],
                      help='Type of model to use')
    parser.add_argument('--hidden_channels', type=int, default=64,
                      help='Number of hidden channels in the model')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=-1,
                      help='Number of epochs to wait for improvement before early stopping')
    parser.add_argument('--print_every', type=int, default=10,
                      help='Print training progress every N epochs')
    parser.add_argument('--save_model', action='store_true',
                      help='Save the trained model')
    parser.add_argument('--save_dir', type=str, default='models',
                      help='Directory to save the model')
    parser.add_argument('--cache_dir', type=str, default='cache',
                      help='Directory to cache the processed data')
    parser.add_argument('--debug', action='store_true',
                      help='Use debug mode with a small subset of data')
    parser.add_argument('--debug_size', type=int, default=100,
                      help='Number of samples per class to use in debug mode')
    parser.add_argument('--max_connections', type=int, default=10,
                      help='Maximum number of connections per node in the graph')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout rate for regularization')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay (L2 regularization) coefficient')
    parser.add_argument('--l1_lambda', type=float, default=0.0,
                      help='L1 regularization coefficient')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                      help='Gradient clipping threshold')
    parser.add_argument('--bert_model', type=str, default='prajjwal1/bert-tiny',
                      help='BERT model to use for text feature extraction (only for bert_graph model)')
    parser.add_argument('--freeze_bert', action='store_true',
                      help='Freeze BERT parameters during training (only for bert_graph model)')
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main() 