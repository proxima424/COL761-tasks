import torch
from argparse import ArgumentParser
import torch 
import torch
import torch
from torch_geometric.nn import GINConv, global_max_pool
import torch.nn.init as init
import csv 

parser = ArgumentParser()
# parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument("-m", "--model_path", type=str, required=True)
parser.add_argument("-d", "--data_path", type=str, required=True)
parser.add_argument("-o", "--output_path", type=str, required=True)
args = parser.parse_args()

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

def evaluate(model, data, file_path):
    predictions = []
    # Initialize lists to store predictions and ground truth labels
    model.eval()
    for i, batch in enumerate(data):
        out = model(batch.x.float(), batch.edge_index.to(torch.long), batch.batch)
        predictions.append(out.detach().cpu().numpy()[0])

    
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(predictions)
    


 


if __name__ == "__main__":
    dataset_path = args.data_path
    model_path = args.model_path
    output_path=args.output_path

    data=torch.load(dataset_path)
    directory=torch.load(model_path)

    has_edge_attr = data[0].edge_attr != None
    print("Edge attributes available: ", has_edge_attr)

    # Number of node features
    if has_edge_attr:
        num_node_features = data[0].x.size(1)
    else:
        num_node_features = 1
        if len(data[0].x.size()) != 2:
            # Reshape the node features tensor to make it 2D with one dimension of size 1
            for i in data:
                i.x = i.x.view(-1, 1)

    model=GINClassifier(128, 1, num_node_features, num_node_features % 6, True)

    model.load_state_dict(directory)
    evaluate(model,data,output_path)