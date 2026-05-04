import numpy as np

def generate_data(samples=200, features=4):
    np.random.seed(42)

    X = []
    y = []

    for i in range(samples):
        if i < samples // 2:
            signal = np.sin(np.linspace(0, 2*np.pi, features))
            label = 0
        else:
            signal = np.cos(np.linspace(0, 2*np.pi, features))
            label = 1

        noise = np.random.normal(0, 0.1, features)
        signal = signal + noise

        X.append(signal)
        y.append(label)

    return np.array(X), np.array(y)