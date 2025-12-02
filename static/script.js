const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const results = document.getElementById("results");
const uploadLabelSpan = document.querySelector(".upload-label span");
const clearBtn = document.querySelector(".clear-btn");

uploadBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) {
        results.textContent = "Please select a file first.";
        return;
    }

    // Display filename
    uploadLabelSpan.textContent = file.name;

    results.textContent = "Running model...";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            results.textContent = "Error: " + data.error;
            return;
        }

        results.innerHTML = `
            <strong>GNN Prediction:</strong> ${data.gnn} <br>
            <strong>LSTM Prediction:</strong> ${data.lstm} <br>
            <strong>XGBoost Prediction:</strong> ${data.xgb} <br>
            <strong>OCSVM Prediction:</strong> ${data.ocsvm} <br>
            <strong>FINAL ALGORITHM:</strong> ${data.final} <br>
            <strong>STANDARD/PROPRIETARY:</strong> ${data.standard} <br>
            <strong>PROTOCOL/SEQUENCE:</strong> ${data.protocol} <br>
            <strong>FUNCTION:</strong> ${data.function}
        `;
    } catch (err) {
        results.textContent = "Error: " + err;
    }
});

clearBtn.addEventListener("click", () => {
    fileInput.value = "";
    uploadLabelSpan.textContent = "Drop a file or click to select";
    results.textContent = "No results yet.";
});
