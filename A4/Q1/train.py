import torch
from argparse import ArgumentParser
import numpy as np
import torch 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
import torch
import torch_geometric
import torch
from torch_geometric.nn import GINConv, global_max_pool
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(42)

parser = ArgumentParser()
# parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument("-d", "--data_path", type=str, required=True)
parser.add_argument("-m", "--model_path", type=str, required=True)

def prepare_data(dataset):
    num_classes = max([data.y.item() for data in dataset]) + 1

    has_edge_attr = dataset[0].edge_attr != None
    print("Edge attributes available: ", has_edge_attr)

    # Number of node features
    if has_edge_attr:
        num_node_features = dataset[0].x.size(1)
            # Number of edge features
    else:
        num_node_features = 1
        # Check if the node features tensor is not 2D
        if len(dataset[0].x.size()) != 2:
            # Reshape the node features tensor to make it 2D with one dimension of size 1
            for i in dataset:
                i.x = i.x.view(-1, 1)
            # Number of edge features
    num_edge_features = dataset[0].edge_index.size(0)

    

    # Extract labels from the dataset
    labels = [data.y.item() for data in dataset]

    # Create masks for each class
    class_masks = [np.array(labels) == i for i in range(num_classes)]

    # Initialize empty lists for training and validation data
    train_data = []
    val_data = []

    # Split data for each class using masks
    for class_mask in class_masks:
        class_data = [data for i, data in enumerate(dataset) if class_mask[i]]
        train_class_data, val_class_data = train_test_split(class_data, test_size=0.2, random_state=42)
        train_data.extend(train_class_data)
        val_data.extend(val_class_data)

    # Shuffle the training and validation data
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)

    # Print the number of data points in the training and validation sets
    print("Number of training data points:", len(train_data))
    print("Number of validation data points:", len(val_data))
    print("Number of classes:", num_classes)
    print("Number of node features:", num_node_features)
    print("Number of edge features:", num_edge_features)
    print("Edge attributes available: ", has_edge_attr)

    return train_data, val_data, num_classes, num_node_features, num_edge_features, has_edge_attr


class GINClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, in_channels, num_layers, use_norm_layers):
        super().__init__()
        self.use_norm_layers = use_norm_layers
        self.layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()

        self.linear = torch.nn.Linear(in_channels, hidden_channels)

        for _ in range(num_layers):
            conv = GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU()
            ))
            self.layers.append(conv)
            if use_norm_layers:
                self.norm_layers.append(torch.nn.LayerNorm(hidden_channels))

        self.fc1 = torch.nn.Linear(hidden_channels, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.fc3 = torch.nn.Linear(32, out_channels)

        init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, edge_index, batch):
        x = self.linear(x)
        x = torch.relu(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index).relu()
            if self.use_norm_layers:
                x = self.norm_layers[i](x)
        x = global_max_pool(x, batch)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


def evaluate(data_loader, model):
    
    # Initialize lists to store predictions and ground truth labels
    all_predictions = []
    all_ground_truth = []
    total_correct_train = 0
    total_samples_train = 0

    for i, batch in enumerate(data_loader):
            out = model(batch.x.float(), batch.edge_index.to(torch.long), batch.batch)

            # Calculate accuracy
            predictions = (out > 0.5).float()
            total_correct_train += (predictions.squeeze().int() == batch.y.int()).sum().item()
            total_samples_train += batch.y.size(0)

            # Collect predictions and ground truth labels
            all_predictions.extend(predictions.squeeze().cpu().numpy())
            all_ground_truth.extend(batch.y.cpu().numpy())

    # Calculate training accuracy
    train_accuracy = total_correct_train / total_samples_train

    # Calculate ROC-AUC score on training data
    train_roc_auc = roc_auc_score(all_ground_truth, all_predictions)

    # # Print training logs
    # print(f"Train Accuracy: {train_accuracy:.4f}, Train ROC-AUC: {train_roc_auc:.4f}")

    return train_accuracy, train_roc_auc    


args = parser.parse_args()

if __name__ == "__main__":

    epochs = 50

    dataset_path = args.data_path
    model_path = args.model_path
    data = torch.load(dataset_path)
    train_data, val_data, num_classes, num_node_features, num_edge_features, has_edge_attr = prepare_data(data)
    model = GINClassifier(128, 1, num_node_features, num_node_features % 6, True)


    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)


    optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    optimizer.zero_grad()
    
    best_validation_roc_auc = 0.0
    best_model_state = None
        # Training Loop
    for epoch in range(epochs):
        
        model.train()
        total_loss = 0.0
        total_correct_train = 0
        total_samples_train = 0

        # Initialize lists to store predictions and ground truth labels
        all_predictions = []
        all_ground_truth = []

        for i, batch in enumerate(train_loader):

            # Move batch to device
            weights = torch.ones_like(batch.y.reshape(1,-1))
            weights[batch.y.reshape(1,-1) == 1] = 2.0
            loss_fun = torch.nn.BCELoss(weight=weights)

            out = model(batch.x.float(), batch.edge_index.to(torch.long), batch.batch)
            
            # Compute loss
            optimizer.zero_grad()
            loss = loss_fun(out.reshape(1, -1), batch.y.float().reshape(1, -1))
            loss.mean().backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy
            predictions = (out > 0.5).float()
            total_correct_train += (predictions.squeeze().int() == batch.y.int()).sum().item()
            total_samples_train += batch.y.size(0)

            # Collect predictions and ground truth labels
            all_predictions.extend(predictions.squeeze().cpu().numpy())
            all_ground_truth.extend(batch.y.cpu().numpy())
        # Calculate average loss
    
        # Calculate training accuracy
        train_accuracy = total_correct_train / total_samples_train
        # Calculate ROC-AUC score on training data
        train_roc_auc = roc_auc_score(all_ground_truth, all_predictions)

        average_loss = total_loss / len(train_loader)
        # Print training logs
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {average_loss:.4f},Train Accuracy: {train_accuracy:.4f}, Train ROC-AUC: {train_roc_auc:.4f}")

        scheduler.step()

        if epoch >= 20:
            validation_accuracy, validation_roc_auc = evaluate(val_loader, model)
            print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {validation_accuracy:.4f}, Validation ROC-AUC: {validation_roc_auc:.4f}")
            
            # Check if current model performs better on validation data
            if validation_roc_auc > best_validation_roc_auc:
                best_validation_roc_auc = validation_roc_auc
                best_model_state = model.state_dict()

    # Save the best model
    print(f"Best Validation ROC-AUC: {best_validation_roc_auc:.4f}")
    if best_model_state is not None:
        torch.save(best_model_state, model_path)