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

parser = ArgumentParser()
# parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument("-d", "--data_path", type=str, required=True)
parser.add_argument("-m", "--model_path", type=str, required=True)

def prepare_data(data):
    test_indices = torch.where(data.y == -1)[0]
    
    # Remove test indices from consideration
    remaining_indices = torch.tensor(list(set(range(data.num_nodes)) - set(test_indices.numpy())))
    
    # Split the remaining indices into train and validation sets
    train_indices, val_indices = train_test_split(remaining_indices, test_size=0.2, random_state=42)
    
    # Initialize train and validation masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    # Set the train and validation masks
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    
    # Assign masks to the data object
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)  # Reset test mask
    data.test_mask[test_indices] = True 
    data.num_classes=data.y.max()+1

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
    

def train(model, optimizer, criterion, data, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits[mask].max(1)[1]
        pred_prob = F.softmax(logits[mask], dim=1)
        labels = data.y[mask]
        accuracy = accuracy_score(labels, pred)
        auc = roc_auc_score(labels.cpu().numpy(), pred_prob.cpu().numpy(), multi_class='ovo', average='macro')
    return accuracy, auc

args = parser.parse_args()

if __name__ == "__main__":
    epochs=1000
    patience=5

    dataset_path = args.data_path
    model_path = args.model_path
    data=torch.load(dataset_path)
    prepare_data(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSageClassifier(in_channels=data.num_features, hidden_channels1=256,hidden_channels2=128,hidden_channels3=64, out_channels=data.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.NLLLoss()
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Adjust learning rate every 20 epochs by multiplying it by 0.5
    # scheduler1 = ExponentialLR(optimizer, gamma=0.9)
    # scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    losses = []
    val_accuracies = []
    roc_scores = []
    best_loss = float('inf')
    best_accuracy = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        loss = train(model, optimizer, criterion, data, data.train_mask)
        train_acc, train_auc = evaluate(model, data, data.train_mask)
        val_acc, val_auc = evaluate(model, data, data.val_mask)
        scheduler.step()
        # scheduler2.step()
        if loss < best_loss or train_acc > best_accuracy:
            best_loss = min(best_loss, loss)
            best_accuracy = max(best_accuracy, train_acc)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping: Training stopped due to loss stagnation or decreasing accuracy after {patience} epochs.")
            break
        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')
        losses.append(loss)
        val_accuracies.append(val_acc)
        roc_scores.append(val_auc)
    checkpoint={
        'model_state_dict': model.state_dict(),
        'num_classes': data.num_classes
    }
    torch.save(checkpoint,
 model_path)
    # torch.save(model.state_dict(), model_path)