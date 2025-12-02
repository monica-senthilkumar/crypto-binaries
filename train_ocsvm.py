import torch
import numpy as np
from sklearn.svm import OneClassSVM
import random
import joblib

DATASET_PATH = "synthetic_dataset.pt"
NUM_CLASSES = 16
NUM_NODES = 30
FEATURE_DIM = 260
TRAIN_SPLIT = 0.8

# ---------------------------------------------------------
# Convert graph → 1D vector for OC-SVM
# ---------------------------------------------------------
def graph_to_vector(graph):
    """
    Flatten node features: (30, 260) -> (7800,)
    """
    return graph.x.reshape(-1).numpy()

# ---------------------------------------------------------
# Main OCSVM Training
# ---------------------------------------------------------
def train_ocsvm_per_class(dataset):
    # Separate dataset by class
    class_data = {c: [] for c in range(NUM_CLASSES)}

    for g in dataset:
        label = g.y.item()
        class_data[label].append(graph_to_vector(g))

    # Model dict
    models = {}

    # Train one model per class
    for c in range(NUM_CLASSES):
        print(f"\nTraining OC-SVM for class {c} ...")

        samples = np.array(class_data[c])
        split = int(TRAIN_SPLIT * len(samples))

        train_samples = samples[:split]

        print(f"  Training samples: {train_samples.shape}")

        # OC-SVM model
        oc = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=0.1
        )
        oc.fit(train_samples)

        # Save model
        joblib.dump(oc, f"ocsvm_class_{c}.pkl")
        models[c] = oc

        print(f"  Saved: ocsvm_class_{c}.pkl")

    return models

# ---------------------------------------------------------
# Prediction using all 16 OC-SVM models
# ---------------------------------------------------------
def predict(models, graph):
    x = graph_to_vector(graph).reshape(1, -1)

    scores = []
    for c in range(NUM_CLASSES):
        score = models[c].score_samples(x)[0]  # higher = more inlier
        scores.append(score)

    return int(np.argmax(scores))

# ---------------------------------------------------------
# Main Driver
# ---------------------------------------------------------
def main():
    print("Loading dataset...")
    dataset = torch.load(DATASET_PATH , weights_only=False)
    random.shuffle(dataset)

    # Train all 16 OC-SVMs
    models = train_ocsvm_per_class(dataset)

    # ------------------------------
    # Evaluate using last 20% samples
    # ------------------------------
    print("\nEvaluating on held-out data...")

    split = int(TRAIN_SPLIT * len(dataset))
    val_data = dataset[split:]

    total = len(val_data)
    correct = 0

    for g in val_data:
        pred = predict(models, g)
        true = g.y.item()
        if pred == true:
            correct += 1

    acc = correct / total
    print(f"\nOC-SVM Validation Accuracy = {acc:.4f}")

    # ------------------------------
    # Show 5 test case predictions
    # ------------------------------
    print("\n---- 5 TEST CASE PREDICTIONS ----")
    for i in range(5):
        g = val_data[i]
        pred = predict(models, g)
        print(f"Sample {i+1}:   True = {g.y.item()}   Pred = {pred}")

if __name__ == "__main__":
    main()













