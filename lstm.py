import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=260, hidden_dim=128, num_layers=2, num_classes=16):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch, 30, 260]
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]                # last layer hidden state
        return self.fc(h)
