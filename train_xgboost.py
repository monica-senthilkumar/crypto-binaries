# train_xgboostnew.py
import torch
import numpy as np
import xgb_utils as xgb
from sklearn.metrics import accuracy_score, classification_report
import random

DATA_PATH = "synthetic_dataset.pt"
NUM_CLASSES = 16
SEQ_LEN = 30
FEATURE_DIM = 260

# ------------ Convert Graph → Flat Vector ------------

def graph_to_flat(graph):
    """
    graph.x shape = [30,260]
    Return flattened vector shape = [7800]
    """
    return graph.x.reshape(-1).numpy()

# ------------ Load + Split Dataset ------------

def load_dataset():
    dataset = torch.load(DATA_PATH , weights_only = False)
    random.shuffle(dataset)
    print("Total samples:", len(dataset))

    # 70/15/15 split
    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    return (
        dataset[:train_size],
        dataset[train_size:train_size + val_size],
        dataset[-test_size:]
    )

# ------------ Prepare Data for XGBoost ------------

def prepare_data(graph_list):
    X = []
    y = []
    for g in graph_list:
        X.append(graph_to_flat(g))
        y.append(int(g.y.item()))

    X = np.array(X)
    y = np.array(y)
    return X, y

# ------------ Train XGBoost Model ------------
from xgboost import XGBClassifier

def train_xgboost():
    print("Loading dataset...")
    train_list, val_list, test_list = load_dataset()

    print("Preparing data...")
    X_train, y_train = prepare_data(train_list)
    X_val, y_val     = prepare_data(val_list)
    X_test, y_test   = prepare_data(test_list)

    print("Training shapes:", X_train.shape, y_train.shape)

    model = XGBClassifier(
        objective="multi:softmax",
        num_class=NUM_CLASSES,
        eval_metric="mlogloss",
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        n_estimators=50
    )

    print("\nTraining XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    model.save_model("xgb_model.json")
    print("\nSaved XGBoost model → xgb_model.json")

    # ----------- Evaluation -----------
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, preds))

    # ----------- 5 Sample Predictions -----------
    print("\n--- 5 Sample Predictions ---")
    for i in range(5):
        pred = model.predict(X_test[i:i+1])[0]
        print(f"Sample {i+1}: True = {y_test[i]} | Pred = {pred}")

    # ----------- 5 Sample Predictions -----------
    
if __name__ == "__main__":
    train_xgboost()




