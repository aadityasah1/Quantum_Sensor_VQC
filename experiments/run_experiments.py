import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.generate_data import generate_data
from models.vqc_model import create_vqc
from models.classical_model import create_classical_model

def run():
    X, y = generate_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Quantum
    vqc = create_vqc(num_qubits=4)
    vqc.fit(X_train, y_train)
    q_acc = vqc.score(X_test, y_test)

    # Classical
    clf = create_classical_model()
    clf.fit(X_train, y_train)
    c_acc = clf.score(X_test, y_test)

    results = pd.DataFrame({
        "Model": ["Quantum VQC", "Classical SVM"],
        "Accuracy": [q_acc, c_acc]
    })

    results.to_csv("results/results.csv", index=False)
    print(results)

if __name__ == "__main__":
    run()