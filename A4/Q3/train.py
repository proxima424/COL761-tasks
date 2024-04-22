import sys
import torch
from argparse import ArgumentParser
import numpy as np
import torch 
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import roc_auc_score, accuracy_score
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import global_mean_pool
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import os
from torch_geometric.nn import GraphSAGE

def prepare_data(data):
    # Convert positive and negative edges to numpy arrays
    pos_edges = data.positive_edges.t().numpy()
    neg_edges = data.negative_edges.t().numpy()

    # Split positive edges into train and validation sets
    train_pos_edges, val_pos_edges = train_test_split(pos_edges, test_size=0.2, random_state=42)

    # Split negative edges into train and validation sets
    train_neg_edges, val_neg_edges = train_test_split(neg_edges, test_size=0.2, random_state=42)

    # Convert back to tensor format
    train_pos_edge_index = torch.tensor(train_pos_edges).t().contiguous()
    val_pos_edge_index = torch.tensor(val_pos_edges).t().contiguous()
    train_neg_edge_index = torch.tensor(train_neg_edges).t().contiguous()
    val_neg_edge_index = torch.tensor(val_neg_edges).t().contiguous()

    data.train_pos_edge_index = train_pos_edge_index
    data.val_pos_edge_index=val_pos_edge_index
    data.train_neg_edge_index=train_neg_edge_index
    data.val_neg_edge_index=val_neg_edge_index


class GNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.conv1 = GraphSAGE(in_dim, hidden_dim,num_layers=1)
        self.conv2 = GraphSAGE(hidden_dim, out_dim,num_layers=1)
        self.lin = nn.Linear(out_dim, 1)
    def encode(self, x, edge_index) :
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
      
        return x
    
    def decode(self, z, pos_edge_index, neg_edge_index):
        pos_scores = self.lin(z[pos_edge_index[0]] * z[pos_edge_index[1]]).squeeze()
        neg_scores = self.lin(z[neg_edge_index[0]] * z[neg_edge_index[1]]).squeeze()
        return pos_scores, neg_scores
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z
    
def train(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)

    pos_scores, neg_scores = model.decode(z, data.train_pos_edge_index, data.train_neg_edge_index)
    
    labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))], dim=0)
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    # print(scores)
    train_loss = criterion(scores, labels)
    train_loss.backward()
    optimizer.step()
  
    return train_loss.item()

def evaluate_val(model,criterion,data):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        # val_pos_edge_index=data.mask1
        # val_neg_edge_index=data.mask2
        val_pos_scores, val_neg_scores = model.decode(z, data.val_pos_edge_index, data.val_neg_edge_index)
        # print(val_neg_scores.shape)
        val_pos_labels = torch.ones(val_pos_scores.size(0))
        val_neg_labels = torch.zeros(val_neg_scores.size(0))
        scores = torch.cat([val_pos_scores, val_neg_scores], dim=0)
        probabilities = torch.sigmoid(scores)

        labels = torch.cat([torch.ones(val_pos_scores.size(0)), torch.zeros(val_neg_scores.size(0))], dim=0)
        val_loss = criterion(torch.cat([val_pos_scores, val_neg_scores], dim=0), torch.cat([val_pos_labels, val_neg_labels], dim=0))
        auc_roc = roc_auc_score(labels, probabilities)
        pred_labels = (probabilities > 0.5).int()
        accuracy = accuracy_score(labels, pred_labels)

    return accuracy,auc_roc

def evaluate_train(model,criterion, data):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        # val_pos_edge_index=data.mask1
        # val_neg_edge_index=data.mask2
        val_pos_scores, val_neg_scores = model.decode(z, data.train_pos_edge_index, data.train_neg_edge_index)
        # print(val_neg_scores.shape)
        val_pos_labels = torch.ones(val_pos_scores.size(0))
        val_neg_labels = torch.zeros(val_neg_scores.size(0))
        scores = torch.cat([val_pos_scores, val_neg_scores], dim=0)
        probabilities = torch.sigmoid(scores)

        labels = torch.cat([torch.ones(val_pos_scores.size(0)), torch.zeros(val_neg_scores.size(0))], dim=0)
        val_loss = criterion(torch.cat([val_pos_scores, val_neg_scores], dim=0), torch.cat([val_pos_labels, val_neg_labels], dim=0))
        auc_roc = roc_auc_score(labels, probabilities)
        pred_labels = (probabilities > 0.5).int()
        accuracy = accuracy_score(labels, pred_labels)

    return accuracy,auc_roc

parser = ArgumentParser()
# parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument("-d", "--data_path", type=str, required=True)
parser.add_argument("-m", "--model_path", type=str, required=True)

args = parser.parse_args()

if __name__ == "__main__":

    dataset_path = args.data_path
    model_path = args.model_path
    data=torch.load(dataset_path)
    prepare_data(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(in_dim=data.x.size(1), hidden_dim=64, out_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    val_accuracies = []
    roc_scores = []
    best_loss = float('inf')
    best_accuracy = 0
    patience_counter = 0
    num_epochs=100
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    early_stopping_rounds = 10  # Number of consecutive epochs with decreasing validation accuracy to trigger early stopping

    best_val_accuracy = 0.0
    no_improvement_count = 0

    for epoch in range(num_epochs):
        loss = train(model, optimizer, criterion, data)
        train_acc,train_auc = evaluate_train(model,criterion, data)
        val_acc,val_auc = evaluate_val(model, criterion,data)
        scheduler.step()
        if val_acc <= best_val_accuracy:
            no_improvement_count += 1
            if no_improvement_count >= early_stopping_rounds:
                print(f"Validation accuracy did not improve for {early_stopping_rounds} consecutive epochs. Stopping training.")
                break
        else:
            best_val_accuracy = val_acc
            no_improvement_count = 0
    
        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Train Acc:{train_acc:.4f},Train AUC: {train_auc:.4f},Val Acc:{val_acc:.4f}, Val AUC: {val_auc:.4f}')
        losses.append(loss)
        val_accuracies.append(val_acc)
        roc_scores.append(val_auc)
    
    torch.save(model.state_dict(), model_path)
