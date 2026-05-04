import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data.generate_data import generate_data
from models.vqc_model import create_vqc

def train():
    np.random.seed(42)

    # Load data
    X, y = generate_data(samples=50)  # Increased for better training

    # Normalize (important for quantum circuits)
    scaler = MinMaxScaler(feature_range=(0, 3.14))
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create model
    vqc = create_vqc(num_qubits=4)

    print("Training started...")
    vqc.fit(X_train, y_train)
    print("Training completed!")

    # Save model + scaler
    os.makedirs("saved_models", exist_ok=True)

    # Save parameters instead of the whole model (pickle issue with VQC)
    with open("saved_models/vqc_params.pkl", "wb") as f:
        pickle.dump(vqc.weights, f)

    with open("saved_models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Model parameters saved!")

if __name__ == "__main__":
    train()