from flask import Flask, request, jsonify, render_template
import torch
from torch_geometric.data import Data
from predict import run_prediction as predict_all_models, LABEL_MAP
import threading
import webbrowser
import time
import os

app = Flask(__name__)

# ------------------ GRAPH GENERATOR ------------------
def generate_graph_from_file(file_content=None):
    # Placeholder: generate random graph
    x = torch.rand(30, 260)
    edges = [[i, i+1] for i in range(29)] + [[i+1, i] for i in range(29)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

# ------------------ ROUTES ------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    file_content = file.read() if file else None

    graph = generate_graph_from_file(file_content)

    node_features = graph.x.numpy()
    edge_index = graph.edge_index.numpy()
    lstm_sequence = graph.x.numpy()
    flat_features = graph.x.reshape(-1).numpy()

    output = predict_all_models(node_features, edge_index, lstm_sequence, flat_features)

    return jsonify({
        "gnn": LABEL_MAP.get(output["gnn_prediction"], "Unknown"),
        "lstm": LABEL_MAP.get(output["lstm_prediction"], "Unknown"),
        "xgb": LABEL_MAP.get(output["xgb_prediction"], "Unknown"),
        "ocsvm": LABEL_MAP.get(output["ocsvm_prediction"], "Unknown")
    })

# ------------------ AUTO OPEN BROWSER ------------------
def open_browser():
    time.sleep(1)  # Wait for server to start
    webbrowser.open("http://127.0.0.1:5000")

# ------------------ MAIN ------------------
if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    app.run(debug=True)
