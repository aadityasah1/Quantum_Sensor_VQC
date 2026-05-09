import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.generate_data import generate_data
from models.vqc_model import create_vqc
from models.classical_model import create_classical_model

def add_classical_noise(X, noise_level):

    noise = np.random.normal(
        0,
        noise_level,
        X.shape
    )

    return X + noise

def run_noise_analysis():

    np.random.seed(42)

    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]

    trials = 5

    quantum_accuracies = []
    classical_accuracies = []

    for noise_level in noise_levels:

        print(f"\nRunning Noise Level: {noise_level}")

        q_trial_acc = []
        c_trial_acc = []

        for trial in range(trials):

            print(f"  Trial {trial+1}")

            # Generate clean data
            X, y = generate_data(
                samples=200,
                features=6
            )

            # Add noise
            X = add_classical_noise(X, noise_level)

            # Normalize
            scaler = MinMaxScaler(
                feature_range=(0, 3.14)
            )

            X = scaler.fit_transform(X)

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            # ------------------------
            # Quantum Model
            # ------------------------

            vqc = create_vqc(num_qubits=6)

            vqc.fit(X_train, y_train)

            q_acc = vqc.score(X_test, y_test)

            q_trial_acc.append(q_acc)

            # ------------------------
            # Classical Model
            # ------------------------

            clf = create_classical_model()

            clf.fit(X_train, y_train)

            c_acc = clf.score(X_test, y_test)

            c_trial_acc.append(c_acc)

        # Average over trials
        quantum_avg = np.mean(q_trial_acc)
        classical_avg = np.mean(c_trial_acc)

        quantum_accuracies.append(quantum_avg)
        classical_accuracies.append(classical_avg)

        print(f"Quantum Avg Accuracy: {quantum_avg:.4f}")
        print(f"Classical Avg Accuracy: {classical_avg:.4f}")

    # Save Results
    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame({
        "Noise Level": noise_levels,
        "Quantum Accuracy": quantum_accuracies,
        "Classical Accuracy": classical_accuracies
    })

    df.to_csv(
        "results/noise_results.csv",
        index=False
    )

    # ------------------------
    # Plot
    # ------------------------

    plt.figure(figsize=(8,5))

    plt.plot(
        noise_levels,
        quantum_accuracies,
        marker='o',
        label='Quantum VQC'
    )

    plt.plot(
        noise_levels,
        classical_accuracies,
        marker='s',
        label='Classical SVM'
    )

    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")

    plt.title(
        "Quantum vs Classical Under Noise"
    )

    plt.legend()

    plt.grid(True)

    plt.savefig(
        "results/noise_comparison.png"
    )

    plt.show()

    print("\nNoise analysis completed!")

if __name__ == "__main__":
    run_noise_analysis()