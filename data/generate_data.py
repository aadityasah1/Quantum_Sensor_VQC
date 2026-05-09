import numpy as np

def generate_data(samples=200, features=6):

    np.random.seed(42)

    X = []
    y = []

    t = np.linspace(0, 2*np.pi, features)

    for i in range(samples):

        if i < samples // 2:

            # Class 0
            signal = (
                np.sin(t)
                + 0.3*np.cos(2*t)
            )

            label = 0

        else:

            # Class 1
            signal = (
                np.sin(t + 0.5)
                + 0.3*np.cos(2*t + 0.2)
            )

            label = 1

        # Stronger intrinsic noise
        intrinsic_noise = np.random.normal(
            0,
            0.25,
            features
        )

        signal = signal + intrinsic_noise

        X.append(signal)
        y.append(label)

    return np.array(X), np.array(y)