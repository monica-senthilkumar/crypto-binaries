import torch
from torch_geometric.data import Data
import random

NUM_CLASSES = 16
SAMPLES_PER_CLASS = 200
NUM_NODES = 30
FEATURE_DIM = 260


def generate_one_graph(class_id):
    # Node features shaped by class
    x = torch.randn(NUM_NODES, FEATURE_DIM) * 0.3 + class_id * 0.8

    # Simple chain graph 0-1-2-3...
    edges = []
    for i in range(NUM_NODES - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    edge_index = torch.tensor(edges, dtype=torch.long).t()

    y = torch.tensor([class_id], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


def generate_dataset():
    dataset = []
    for c in range(NUM_CLASSES):
        for _ in range(SAMPLES_PER_CLASS):
            dataset.append(generate_one_graph(c))

    random.shuffle(dataset)
    torch.save(dataset, "synthetic_dataset.pt")
    print("Dataset saved: 3200 samples")


if __name__ == "__main__":
    generate_dataset()
