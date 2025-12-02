from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from torch_geometric.data import Data
import joblib
import os

from gnn import GNNClassifier
from lstm import LSTMClassifier
from xgb_utils import XGBoostWrapper
from blockchain import Blockchain

app = Flask(__name__)

# -------------------- Load Models --------------------
gnn_model = GNNClassifier()
gnn_model.load_state_dict(torch.load("gnn_best.pt", map_location="cpu"))
gnn_model.eval()

lstm_model = LSTMClassifier()
lstm_model.load_state_dict(torch.load("lstm_best.pt", map_location="cpu"))
lstm_model.eval()

xgb_wrapper = XGBoostWrapper()
xgb_wrapper.model.load_model("xgb_model.json")

NUM_CLASSES = 16
ocsvm_models = []
for i in range(NUM_CLASSES):
    fname = f"ocsvm_class_{i}.pkl"
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing: {fname}")
    ocsvm_models.append(joblib.load(fname))

# -------------------- Label Maps --------------------
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

# -------------------- Feature Generation --------------------
def hex_to_features(hex_text):
    hex_text = ''.join(hex_text.split()).lower()
    hex_text = ''.join(c for c in hex_text if c in '0123456789abcdef')
    if len(hex_text) % 2 != 0:
        hex_text = '0' + hex_text

    raw_bytes = bytes.fromhex(hex_text)
    arr = np.array(list(raw_bytes), dtype=np.float32)

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

    np.random.seed(42)
    edge_index = np.array([
        np.random.randint(0, 40, 80),
        np.random.randint(0, 40, 80)
    ], dtype=np.int64)

    return node_features, edge_index, lstm_sequence, xgb_flat

# -------------------- Model Prediction --------------------
def prepare_gnn_graph(node_features, edge_index):
    x = torch.tensor(node_features, dtype=torch.float)
    e = torch.tensor(edge_index, dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    return Data(x=x, edge_index=e, batch=batch)

def prepare_lstm_input(sequence):
    return torch.tensor(sequence, dtype=torch.float).unsqueeze(0)

def run_models(node_features, edge_index, lstm_sequence, flat_features):
    g_input = prepare_gnn_graph(node_features, edge_index)
    with torch.no_grad():
        g_logits = gnn_model(g_input)
        g_pred = int(torch.argmax(g_logits, 1).item())

    l_input = prepare_lstm_input(lstm_sequence)
    with torch.no_grad():
        l_logits = lstm_model(l_input)
        l_pred = int(torch.argmax(l_logits, 1).item())

    x_pred = int(xgb_wrapper.model.predict(np.array([flat_features]))[0])

    vec = flat_features.reshape(1, -1)
    scores = [m.score_samples(vec)[0] for m in ocsvm_models]
    o_pred = int(np.argmax(scores))

    return {"gnn": g_pred, "lstm": l_pred, "xgb": x_pred, "ocsvm": o_pred}

def majority_vote(pred):
    votes = [pred["gnn"], pred["lstm"], pred["xgb"], pred["ocsvm"]]
    return max(set(votes), key=votes.count)

# -------------------- Flask Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")  # your dashboard HTML

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        hex_text = file.read().decode("utf-8").strip()

        node, edge, lstm_seq, flat = hex_to_features(hex_text)
        pred = run_models(node, edge, lstm_seq, flat)
        final_class = majority_vote(pred)
        name = LABEL_MAP[final_class]

        # Optional: Blockchain
        chain = Blockchain()
        block_data = {
            "gnn_prediction": pred["gnn"],
            "lstm_prediction": pred["lstm"],
            "xgb_prediction": pred["xgb"],
            "ocsvm_prediction": pred["ocsvm"],
            "final_result": final_class
        }
        chain.add_block(block_data)

        return jsonify({
            "gnn": LABEL_MAP[pred["gnn"]],
            "lstm": LABEL_MAP[pred["lstm"]],
            "xgb": LABEL_MAP[pred["xgb"]],
            "ocsvm": LABEL_MAP[pred["ocsvm"]],
            "final": name,
            "standard": STANDARD_MAP[name],
            "protocol": PROTOCOL_SEQUENCE[name],
            "function": ALGO_FUNCTION[name]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
