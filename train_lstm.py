#!/usr/bin/env python3
# train_lstmnew.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import random

# ------------------- CONFIG -------------------
DATA_PATH = "synthetic_dataset.pt"   # path to your dataset file
BATCH_SIZE = 32
EPOCHS = 40
LR = 0.001
PATIENCE = 8

SEQ_LEN = 30
FEATURE_DIM = 260
NUM_CLASSES = 16

# ------------------- DATASET WRAPPER -------------------
class LSTMDataset(Dataset):
    def __init__(self, graph_list):
        super().__init__()
        self.data = graph_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        g = self.data[idx]
        x = g.x          # shape [30,260]
        y = g.y
        # ensure y is scalar Long tensor
        if isinstance(y, torch.Tensor):
            y = y.long().view(-1)[0]
        else:
            y = torch.tensor(y).long()
        return x, y

# ------------------- LSTM MODEL -----------------------
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=FEATURE_DIM,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        # x: [batch, 30, 260]
        _, (hn, _) = self.lstm(x)
        last = hn[-1]     # [batch, 128]
        out = self.fc(last)
        return out

# ------------------- LOADING + SPLIT -------------------
def load_dataset():
    dataset = torch.load(DATA_PATH, weights_only=False)
    print("Total samples:", len(dataset))

    random.shuffle(dataset)

    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    return (
        dataset[:train_size],
        dataset[train_size:train_size + val_size],
        dataset[-test_size:]
    )

# ------------------- EVALUATION -----------------------
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()

            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total

# ------------------- TRAINING LOOP -----------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_list, val_list, test_list = load_dataset()

    train_loader = DataLoader(LSTMDataset(train_list), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(LSTMDataset(val_list), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(LSTMDataset(test_list), batch_size=BATCH_SIZE)

    model = LSTMClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    patience = 0

    print("\nStarting LSTM training...\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), "lstm_best.pt")
            print(f" -> Saved best model (epoch {epoch})")
        else:
            patience += 1

        if patience >= PATIENCE:
            print("\nEarly stopping.\n")
            break

    print("Best val acc:", best_acc)

    # -------- Load best model --------
    model.load_state_dict(torch.load("lstm_best.pt", map_location=device))
    print("Loaded best LSTM model.\n")

    # -------- Final Test --------
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"Test loss {test_loss:.4f} | Test acc {test_acc:.4f}\n")

    # -------- Sample Predictions --------
    print("--- 5 Sample Predictions ---")
    model.eval()
    with torch.no_grad():
        for i in range(5):
            x, y = LSTMDataset(test_list)[i]
            x = x.unsqueeze(0).to(device)

            out = model(x)
            pred = out.argmax().item()

            print(f"Sample {i+1}: True = {y.item()} | Pred = {pred}")

    print("\nDone.")

if __name__ == "__main__":
    train()

