from sklearn.svm import SVC

def create_classical_model():
    return SVC(kernel='rbf')