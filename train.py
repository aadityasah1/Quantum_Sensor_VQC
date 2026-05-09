import os
import time
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data.generate_data import generate_data
from models.vqc_model import create_vqc

def train():

    np.random.seed(42)

    print("\nGenerating dataset...")

    # Larger dataset for stability
    X, y = generate_data(samples=200, features=6)

    # Normalize for quantum encoding
    scaler = MinMaxScaler(feature_range=(0, 3.14))
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nCreating VQC model...")

    vqc = create_vqc(num_qubits=6)

    print("\nTraining started...")

    start_time = time.time()

    vqc.fit(X_train, y_train)

    end_time = time.time()

    print("\nTraining completed!")

    # Accuracy
    train_acc = vqc.score(X_train, y_train)
    test_acc = vqc.score(X_test, y_test)

    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    print(f"\nTraining Time: {end_time - start_time:.2f} seconds")

    # Save models
    os.makedirs("saved_models", exist_ok=True)

    with open("saved_models/vqc_params.pkl", "wb") as f:
        pickle.dump(vqc.weights, f)

    with open("saved_models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save experiment metadata
    metadata = {
        "samples": 200,
        "features": 4,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "training_time": end_time - start_time
    }

    with open("saved_models/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("\nModel parameters saved!")

if __name__ == "__main__":
    train()