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


parser = ArgumentParser()
parser.add_argument("-m", "--model_path", type=str, required=True)
parser.add_argument("-d", "--data_path", type=str, required=True)
parser.add_argument("-o", "--output_path", type=str, required=True)
args = parser.parse_args()

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

def evaluate(data,model,file_path):
    model.eval()
    
    # Compute node embeddings
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index) ## node embeddinsg
    
    # Predict scores for test edges
    with torch.no_grad():
        test_scores=model.lin(z[data.test_edges[0]] * z[data.test_edges[1]]).squeeze()
       
        test_probs=torch.sigmoid(test_scores)
        test_probabilities = test_probs.numpy()
        with open(file_path, 'w') as file:
            for prob in test_probabilities:
                file.write(f"{prob}\n")



if __name__ == "__main__":
    dataset_path = args.data_path
    model_path = args.model_path
    output_path=args.output_path

    data=torch.load(dataset_path)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## Load model
    model = GNN(in_dim=data.x.size(1), hidden_dim=64, out_dim=32)
    model.load_state_dict(torch.load(model_path))

    ## Evaluate on test_edges
    evaluate(data,model,output_path)



