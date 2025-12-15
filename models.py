import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import BatchNorm

class GNNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, hidden1=128, hidden2=64, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden1)
        self.bn1 = BatchNorm(hidden1)

        self.conv2 = GCNConv(hidden1, hidden2)
        self.bn2 = BatchNorm(hidden2)

        # optional third conv for capacity
        self.conv3 = GCNConv(hidden2, hidden2)
        self.bn3 = BatchNorm(hidden2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden2, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # graph-level pooling
        x = global_mean_pool(x, batch)  # [num_graphs, hidden2]

        return self.mlp(x)
