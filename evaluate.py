import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from types import SimpleNamespace

from data.generate_data import generate_data
from models.vqc_model import create_vqc

def evaluate():
    # Recreate model and load parameters
    vqc = create_vqc(num_qubits=6)
    with open("saved_models/vqc_params.pkl", "rb") as f:
        weights = pickle.load(f)
    vqc._fit_result = SimpleNamespace(x=weights)

    with open("saved_models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load data
    X, y = generate_data()

    # Apply same scaling
    X = scaler.transform(X)

    # Split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Predict
    preds = vqc.predict(X_test)
    if preds.ndim > 1 and preds.shape[1] > 1:
        preds = np.argmax(preds, axis=1)

    # Accuracy
    acc = np.mean(preds == y_test)
    print("Test Accuracy:", acc)

    # Plot
    plt.figure()
    plt.plot(y_test[:20], label="True")
    plt.plot(preds[:20], label="Predicted")
    plt.legend()
    plt.title("Quantum VQC Predictions")
    plt.show()

if __name__ == "__main__":
    evaluate()