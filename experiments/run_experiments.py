import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.generate_data import generate_data
from models.vqc_model import create_vqc
from models.classical_model import create_classical_model

def run():

    np.random.seed(42)

    print("\nGenerating dataset...")

    X, y = generate_data(samples=200, features=6)

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 3.14))
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining Quantum VQC...")

    vqc = create_vqc(num_qubits=6)

    vqc.fit(X_train, y_train)

    q_acc = vqc.score(X_test, y_test)

    print(f"Quantum Accuracy: {q_acc:.4f}")

    print("\nTraining Classical SVM...")

    clf = create_classical_model()

    clf.fit(X_train, y_train)

    c_acc = clf.score(X_test, y_test)

    print(f"Classical Accuracy: {c_acc:.4f}")

    os.makedirs("results", exist_ok=True)

    results = pd.DataFrame({
        "Model": ["Quantum VQC", "Classical SVM"],
        "Accuracy": [q_acc, c_acc]
    })

    results.to_csv("results/results.csv", index=False)

    print("\n===== FINAL RESULTS =====")
    print(results)

if __name__ == "__main__":
    run()