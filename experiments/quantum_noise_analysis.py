import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from qiskit_aer.primitives import SamplerV2
from qiskit_aer import AerSimulator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.generate_data import generate_data
from models.vqc_model import create_vqc
from models.classical_model import create_classical_model
from noise.quantum_noise import create_quantum_noise_model


def run_quantum_noise_analysis():

    np.random.seed(42)

    noise_levels = [0.0, 0.02, 0.05, 0.1, 0.15]

    trials = 3

    ideal_quantum_acc = []
    noisy_quantum_acc = []
    classical_acc = []

    for noise_level in noise_levels:

        print(f"\n==============================")
        print(f"Noise Level: {noise_level}")
        print(f"==============================")

        ideal_trial = []
        noisy_trial = []
        classical_trial = []

        for trial in range(trials):

            print(f"Trial {trial+1}/{trials}")

            # --------------------------------
            # Dataset
            # --------------------------------

            X, y = generate_data(
                samples=200,
                features=6
            )

            # --------------------------------
            # PCA Compression
            # --------------------------------

            pca = PCA(n_components=4)
            X = pca.fit_transform(X)

            # --------------------------------
            # Scaling
            # --------------------------------

            scaler = MinMaxScaler(
                feature_range=(0, 3.14)
            )

            X = scaler.fit_transform(X)

            # --------------------------------
            # Split
            # --------------------------------

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            # ==========================================
            # IDEAL QUANTUM
            # ==========================================

            ideal_vqc = create_vqc(
                num_qubits=4
            )

            ideal_vqc.fit(X_train, y_train)

            ideal_score = ideal_vqc.score(
                X_test,
                y_test
            )

            ideal_trial.append(ideal_score)

            # ==========================================
            # NOISY QUANTUM
            # ==========================================

            noise_model = create_quantum_noise_model(
                noise_level=noise_level
            )

            backend = AerSimulator(
                noise_model=noise_model
            )

            noisy_sampler = SamplerV2.from_backend(backend)

            noisy_vqc = create_vqc(
                num_qubits=4,
                sampler=noisy_sampler,
                backend=backend
            )

            noisy_vqc.fit(X_train, y_train)

            noisy_score = noisy_vqc.score(
                X_test,
                y_test
            )

            noisy_trial.append(noisy_score)

            # ==========================================
            # CLASSICAL MODEL
            # ==========================================

            clf = create_classical_model()

            clf.fit(X_train, y_train)

            classical_score = clf.score(
                X_test,
                y_test
            )

            classical_trial.append(classical_score)

        # ---------------------------------
        # Average Results
        # ---------------------------------

        ideal_quantum_acc.append(
            np.mean(ideal_trial)
        )

        noisy_quantum_acc.append(
            np.mean(noisy_trial)
        )

        classical_acc.append(
            np.mean(classical_trial)
        )

        print(f"Ideal Quantum: {np.mean(ideal_trial):.4f}")
        print(f"Noisy Quantum: {np.mean(noisy_trial):.4f}")
        print(f"Classical SVM: {np.mean(classical_trial):.4f}")

    # ======================================
    # Save Results
    # ======================================

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame({
        "Noise Level": noise_levels,
        "Ideal Quantum": ideal_quantum_acc,
        "Noisy Quantum": noisy_quantum_acc,
        "Classical SVM": classical_acc
    })

    df.to_csv(
        "results/quantum_noise_results.csv",
        index=False
    )

    # ======================================
    # Plot Results
    # ======================================

    plt.figure(figsize=(10,6))

    plt.plot(
        noise_levels,
        ideal_quantum_acc,
        marker='o',
        linewidth=2,
        label='Ideal Quantum VQC'
    )

    plt.plot(
        noise_levels,
        noisy_quantum_acc,
        marker='s',
        linewidth=2,
        label='Noisy Quantum VQC'
    )

    plt.plot(
        noise_levels,
        classical_acc,
        marker='^',
        linewidth=2,
        label='Classical SVM'
    )

    plt.xlabel("Quantum Noise Level")
    plt.ylabel("Classification Accuracy")

    plt.title(
        "Ideal vs Noisy Quantum Classification"
    )

    plt.legend()

    plt.grid(True)

    plt.savefig(
        "results/quantum_noise_comparison.png"
    )

    plt.show()

    print("\nQuantum noise analysis completed!")


if __name__ == "__main__":
    run_quantum_noise_analysis()
