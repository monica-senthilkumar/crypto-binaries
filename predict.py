
import torch
import numpy as np
from torch_geometric.data import Data
import joblib   # to load .pkl files
import os 
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from gnn import GNNClassifier
from lstm import LSTMClassifier
from xgb_utils import XGBoostWrapper
from blockchain import Blockchain









# -----------------------------------------------------------
# Load ML Models
# -----------------------------------------------------------

gnn_model = GNNClassifier()
gnn_model.load_state_dict(torch.load("gnn_best.pt", map_location="cpu"))
gnn_model.eval()

lstm_model = LSTMClassifier()
lstm_model.load_state_dict(torch.load("lstm_best.pt", map_location="cpu"))
lstm_model.eval()

xgb_wrapper = XGBoostWrapper()
xgb_wrapper.model.load_model("xgb_model.json")

# Load OCSVM models
NUM_CLASSES = 16
ocsvm_models = []
for i in range(NUM_CLASSES):
    fname = f"ocsvm_class_{i}.pkl"
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing: {fname}")
    ocsvm_models.append(joblib.load(fname))
print(f"Loaded {len(ocsvm_models)} OCSVM models.")

# -----------------------------------------------------------
# LABEL MAPS
# -----------------------------------------------------------
LABEL_MAP = {
    0: "AES", 1: "DES", 2: "SIMON", 3: "RSA",
    4: "SHA-256", 5: "PBKDF2", 6: "ChaCha20", 7: "Blowfish",
    8: "Twofish", 9: "Camellia", 10: "MD5", 11: "Keccak",
    12: "HMAC", 13: "ECDSA", 14: "Ed25519", 15: "Argon2"
}

STANDARD_MAP = {
    "AES": "Standard", "DES": "Standard", "SIMON": "Proprietary", "RSA": "Standard",
    "SHA-256": "Standard", "PBKDF2": "Standard", "ChaCha20": "Standard", "Blowfish": "Standard",
    "Twofish": "Standard", "Camellia": "Standard", "MD5": "Standard (Deprecated)", "Keccak": "Standard",
    "HMAC": "Standard", "ECDSA": "Standard", "Ed25519": "Standard", "Argon2": "Standard"
}

PROTOCOL_SEQUENCE = {
    "AES": "Key expansion → SubBytes → ShiftRows → MixColumns → AddRoundKey",
    "DES": "Expansion → XOR → S-box → Permutation → 16 Rounds",
    "SIMON": "Feistel Network → Bitwise Rotation → XOR",
    "RSA": "KeyGen → Encryption c=m^e mod n → Decryption m=c^d mod n",
    "SHA-256": "Padding → Parsing → Compression → Digest",
    "PBKDF2": "Salt → HMAC Looping → Key Derivation",
    "ChaCha20": "Key setup → Add, XOR, Rotate → 20 rounds",
    "Blowfish": "Feistel → Subkeys → 16 rounds",
    "Twofish": "Whitening → Key-dependent S-Boxes → Feistel",
    "Camellia": "FL/FL⁻¹ → Feistel → Key Whitening",
    "MD5": "Padding → Append length → 64-step loop",
    "Keccak": "Sponge construction → Absorb → Squeeze",
    "HMAC": "ipad/opad XOR → Hashing",
    "ECDSA": "KeyGen → Sign (r,s) → Verification",
    "Ed25519": "Curve25519 → SHA-512 → Signature",
    "Argon2": "Memory-fill → Compression → Final hash"
}

ALGO_FUNCTION = {
    "AES": "Symmetric block cipher used for secure data encryption.",
    "DES": "Symmetric block cipher, older encryption standard.",
    "SIMON": "Lightweight block cipher for constrained devices.",
    "RSA": "Asymmetric cipher used for key exchange and digital signatures.",
    "SHA-256": "Hash function producing 256-bit digest for data integrity.",
    "PBKDF2": "Key derivation function using password and salt.",
    "ChaCha20": "Stream cipher for high-speed encryption.",
    "Blowfish": "Symmetric block cipher, fast and secure.",
    "Twofish": "Symmetric block cipher, alternative to AES.",
    "Camellia": "Symmetric cipher, compatible with AES security.",
    "MD5": "Hash function, now considered insecure.",
    "Keccak": "Hash function underlying SHA-3 standard.",
    "HMAC": "Message authentication using hash functions.",
    "ECDSA": "Elliptic Curve Digital Signature Algorithm.",
    "Ed25519": "Fast elliptic curve signature scheme.",
    "Argon2": "Memory-hard key derivation for password hashing."
}

# -----------------------------------------------------------
# FEATURE GENERATION FROM HEX   (converts the given hex input to each model features)
# -----------------------------------------------------------
def hex_to_features(hex_text):
    """
    Convert single-line hexadecimal input into features for each model
    """
    # Remove spaces, newlines, and make lowercase
    hex_text = ''.join(hex_text.split()).lower()

    # Remove any non-hex characters (just in case)
    hex_text = ''.join(c for c in hex_text if c in '0123456789abcdef')

    # If length is odd, pad with a leading zero
    if len(hex_text) % 2 != 0:
        hex_text = '0' + hex_text

    raw_bytes = bytes.fromhex(hex_text)
    arr = np.array(list(raw_bytes), dtype=np.float32)

    # Pad/repeat for each model
    def pad_to_length(a, target):
        if len(a) >= target:
            return a[:target]
        repeat_factor = (target // len(a)) + 1
        return np.tile(a, repeat_factor)[:target]

    node_flat = pad_to_length(arr, 40 * 260)
    lstm_flat = pad_to_length(arr, 30 * 260)
    xgb_flat  = pad_to_length(arr, 7800)

    node_features = node_flat.reshape(40, 260)
    lstm_sequence = lstm_flat.reshape(30, 260)

    # Random edges for GNN (fixed seed for reproducibility)
    np.random.seed(42)
    edge_index = np.array([
        np.random.randint(0, 40, 80),
        np.random.randint(0, 40, 80)
    ], dtype=np.int64)

    return node_features, edge_index, lstm_sequence, xgb_flat


# -----------------------------------------------------------
# MODEL PREDICTION FUNCTIONS
# -----------------------------------------------------------
def prepare_gnn_graph(node_features, edge_index):
    x = torch.tensor(node_features, dtype=torch.float)
    e = torch.tensor(edge_index, dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    return Data(x=x, edge_index=e, batch=batch)

def prepare_lstm_input(sequence):
    return torch.tensor(sequence, dtype=torch.float).unsqueeze(0)

def run_models(node_features, edge_index, lstm_sequence, flat_features):   # runs all the models for given input and do predictions
    # GNN
    g_input = prepare_gnn_graph(node_features, edge_index)
    with torch.no_grad():
        g_logits = gnn_model(g_input)
        g_pred = int(torch.argmax(g_logits, 1).item())

    # LSTM
    l_input = prepare_lstm_input(lstm_sequence)
    with torch.no_grad():
        l_logits = lstm_model(l_input)
        l_pred = int(torch.argmax(l_logits, 1).item())

    # XGBoost
    x_pred = int(xgb_wrapper.model.predict(np.array([flat_features]))[0])

    # OCSVM
    vec = flat_features.reshape(1, -1)
    scores = [m.score_samples(vec)[0] for m in ocsvm_models]
    o_pred = int(np.argmax(scores))

    return {"gnn": g_pred, "lstm": l_pred, "xgb": x_pred, "ocsvm": o_pred}

# -----------------------------------------------------------
# FINAL OUTPUT
# -----------------------------------------------------------
def display_results(pred):
    # Majority voting
    votes = [pred["gnn"], pred["lstm"], pred["xgb"], pred["ocsvm"]]
    final_class = max(set(votes), key=votes.count)
    name = LABEL_MAP[final_class]

    print("\n================== RESULT ==================")
    print(f"GNN Prediction     : {LABEL_MAP[pred['gnn']]}")
    print(f"LSTM Prediction    : {LABEL_MAP[pred['lstm']]}")
    print(f"XGBoost Prediction : {LABEL_MAP[pred['xgb']]}")
    print(f"OCSVM Prediction   : {LABEL_MAP[pred['ocsvm']]}")
    print("--------------------------------------------")
    print(f"FINAL ALGORITHM    : {name}")
    print(f"STANDARD/PROPRIETARY : {STANDARD_MAP[name]}")
    print(f"PROTOCOL/SEQUENCE     : {PROTOCOL_SEQUENCE[name]}")
    print(f"FUNCTION              : {ALGO_FUNCTION[name]}")
    print("============================================\n")
    
    accuracy, precision, recall, false_pos = compute_model_metrics(
    predictions=votes,
    final_algo=final_class
)

    print("\n--------- MODEL METRICS ---------")
    print(f"Accuracy  : {accuracy:.3f}")
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"False Positives : {false_pos}")
    print("--------------------------------")
    
    
    
    
    
    return final_class

# -----------------------------------------------------------
# MODEL METRICS (Dummy example: Replace with real validation)
# -----------------------------------------------------------
def print_metrics(true_labels, predicted_labels):
    acc = accuracy_score(true_labels, predicted_labels)
    prec = precision_score(true_labels, predicted_labels, average="macro", zero_division=0)
    rec = recall_score(true_labels, predicted_labels, average="macro", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels, labels=list(range(NUM_CLASSES))).ravel() if len(set(true_labels))==2 else (0,0,0,0)
    print("--------- MODEL METRICS ---------")
    print(f"Accuracy  : {acc:.3f}")
    print(f"Precision : {prec:.3f}")
    print(f"Recall    : {rec:.3f}")
    print(f"False Positives : {fp}")
    print("--------------------------------\n")
    
    
def compute_model_metrics(predictions, final_algo):
    total = len(predictions)
    correct = sum(1 for p in predictions if p == final_algo)
    wrong = total - correct

    accuracy = correct / total
    precision = accuracy
    recall = accuracy

    return accuracy, precision, recall, wrong







# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    print("\nEnter crypto algorithm as single-line HEX:\n")
    hex_input = input("> ")

    node, edge, lstm_seq, flat = hex_to_features(hex_input)
    predictions = run_models(node, edge, lstm_seq, flat)
    final_class = display_results(predictions)

    # Blockchain
    chain = Blockchain()
    block_data = {
        "gnn_prediction": predictions["gnn"],
        "lstm_prediction": predictions["lstm"],
        "xgb_prediction": predictions["xgb"],
        "ocsvm_prediction": predictions["ocsvm"],
        "final_result": final_class
    }
    block = chain.add_block(block_data)
    print("=== BLOCK ADDED TO BLOCKCHAIN ===")
    print(json.dumps(block.__dict__, indent=4))
    print("=================================\n")
    
    

    # Example metrics (replace with validation set for real values)
    #true_labels = [final_class]  # dummy true label
    #predicted_labels = [final_class]  # dummy predicted label
    #print_metrics([final_class], [final_class])