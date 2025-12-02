# train_gnn.py
import torch
from torch_geometric.loader import DataLoader 
from sklearn.metrics import classification_report
from models import GNNClassifier
import random

DATA_PATH = "synthetic_dataset.pt"
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
PATIENCE = 8

def load_dataset():
    dataset = torch.load(DATA_PATH , weights_only=False)
    print("Total samples:", len(dataset))

    random.shuffle(dataset)

    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    train_list = dataset[:train_size]
    val_list   = dataset[train_size:train_size + val_size]
    test_list  = dataset[-test_size:]

    print(f"Splits -> train {len(train_list)}, val {len(val_list)}, test {len(test_list)}")
    return train_list, val_list, test_list

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            loss_sum += loss.item()

            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

    return loss_sum / len(loader), correct / total

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_list, val_list, test_list = load_dataset()

    train_loader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_list, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_list, batch_size=BATCH_SIZE, shuffle=False)

    model = GNNClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0
    patience = 0

    print("Starting GNN training...\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(dim=1) == batch.y).sum().item()
            total += batch.y.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), "gnn_best.pt")
            print(f"  -> Saved new best model (epoch {epoch} | val_acc {val_acc:.4f})")
        else:
            patience += 1

        if patience >= PATIENCE:
            print("\nEarly stopping triggered.\n")
            break

    print("Training finished. Best val acc:", best_acc)

    # ----------- Load best model ----------------
    model.load_state_dict(torch.load("gnn_best.pt"))
    print("Loaded best GNN model for final testing...\n")

    # ----------- Final Test ---------------------
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test loss {test_loss:.4f} | Test acc {test_acc:.4f}")

    # ----------- 5 Prediction Examples ----------
    print("\n--- SAMPLE PREDICTIONS (5 graphs) ---")
    model.eval()
    with torch.no_grad():
        for i in range(5):
            sample = test_list[i].to(device)
            out = model(sample)
            pred = out.argmax().item()
            true = sample.y.item()
            print(f"Sample {i+1}: True = {true}, Predicted = {pred}")

    print("\nDone.")

if __name__ == "__main__":
    train()
