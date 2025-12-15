import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from dataset_loader import load_dataset


# --------------------------
# GNN MODEL
# --------------------------
class HighAccGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.4):
        super().__init__()

        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)

        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# --------------------------
# TRAINING FUNCTION
# --------------------------
def train_model(dataset):
    print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = [data.y.item() for data in dataset]
    num_classes = len(set(labels))
    print("Detected number of classes =", num_classes)

    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    train_split = int(0.70 * len(dataset))
    val_split = int(0.15 * len(dataset))

    train_idx = idx[:train_split]
    val_idx = idx[train_split:train_split + val_split]
    test_idx = idx[train_split + val_split:]

    train_ds = [dataset[i] for i in train_idx]
    val_ds = [dataset[i] for i in val_idx]
    test_ds = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    feat_dim = dataset[0].x.shape[1]
    print("Input dim:", feat_dim)

    model = HighAccGNN(feat_dim, hidden_dim=128, num_classes=num_classes).to(device)

    class_counts = torch.bincount(torch.tensor(labels))
    weights = 1.0 / class_counts.float()
    weights = weights / weights.mean()

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    best_acc = 0
    patience_counter = 0
    PATIENCE = 8

    for epoch in range(40):

        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)

            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1} | Train Loss {total_loss:.4f} | Val Acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_gnn.pth")
            print("🔥 Model improved and saved!")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break

    print("Training finished. Loading best model...")
    model.load_state_dict(torch.load("best_gnn.pth"))

    print("\nTesting...")
    evaluate(model, test_loader, device, detailed=True)


# --------------------------
# EVALUATION FUNCTION
# --------------------------
def evaluate(model, loader, device, detailed=False):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            pred = logits.argmax(dim=1).cpu()
            preds.extend(pred.numpy())
            labels.extend(batch.y.cpu().numpy())

    acc = accuracy_score(labels, preds)

    if detailed:
        print("Accuracy =", acc)
        print("\nClass Report:\n", classification_report(labels, preds))

    return acc


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    from dataset_loader import load_dataset   # <-- your loader function
    dataset = load_dataset()
    train_model(dataset)
