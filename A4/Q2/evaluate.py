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
import pandas as pd
import csv 

parser = ArgumentParser()
# parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument("-m", "--model_path", type=str, required=True)
parser.add_argument("-d", "--data_path", type=str, required=True)
parser.add_argument("-o", "--output_path", type=str, required=True)
args = parser.parse_args()

class GraphSageClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels1,hidden_channels2, hidden_channels3,out_channels):
        super(GraphSageClassifier, self).__init__()
        self.bn=nn.BatchNorm1d(in_channels)
        self.conv1 = GraphSAGE(in_channels, hidden_channels1,num_layers=2)
        self.bn1 = nn.BatchNorm1d(hidden_channels1)  # Batch normalization layer
        self.conv2=GraphSAGE(hidden_channels1,hidden_channels2,num_layers=2)
        self.bn2 = nn.BatchNorm1d(hidden_channels2)
        self.conv3=GraphSAGE(hidden_channels2, hidden_channels3,num_layers=2)
        self.bn3 = nn.BatchNorm1d(hidden_channels3)
        self.conv4 = GraphSAGE(hidden_channels3, out_channels,num_layers=1)

    def forward(self, x, edge_index):
        # x=self.bn(x)
        x = self.bn1(F.relu(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.bn2(F.relu(self.conv2(x, edge_index)))
        x=self.bn3(F.relu(self.conv3(x,edge_index)))
        x=self.conv4(x,edge_index)
        return F.log_softmax(x, dim=1)
    

def evaluate(model, data,file_path):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.max(1)[1]
        pred_prob = F.softmax(logits, dim=1)
    predictions_np = pred_prob.cpu().numpy()

 
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(predictions_np)
 


if __name__ == "__main__":
    dataset_path = args.data_path
    model_path = args.model_path
    output_path=args.output_path

    data=torch.load(dataset_path)
    directory=torch.load(model_path)
    num_classes = directory['num_classes']
    # data.num_classes=data.y.max()+1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model=GraphSageClassifier(in_channels=data.num_features, hidden_channels1=256,hidden_channels2=128,hidden_channels3=64, out_channels=num_classes).to(device)
    model.load_state_dict(directory['model_state_dict'])
    evaluate(model,data,output_path)