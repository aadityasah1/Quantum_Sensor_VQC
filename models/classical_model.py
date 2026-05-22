"""
Classical baseline models for comparison against VQC.
"""

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier


def create_classical_model(name: str = "svm_rbf", seed: int = 42):
    """
    Returns a fitted-ready sklearn estimator.

    Available models
    ----------------
    'svm_rbf'            : RBF-kernel SVC (default, strong baseline)
    'svm_linear'         : Linear SVC
    'logistic_regression': Logistic regression
    'mlp'                : 2-hidden-layer MLP (64, 32)
    'random'             : Stratified dummy (lower bound)
    """
    models = {
        "svm_rbf": SVC(
            kernel="rbf", C=10.0, gamma="scale",
            probability=True, random_state=seed
        ),
        "svm_linear": SVC(
            kernel="linear", C=1.0,
            probability=True, random_state=seed
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=seed
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=20,
        ),
        "random": DummyClassifier(strategy="stratified", random_state=seed),
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(models)}")
    return models[name]
