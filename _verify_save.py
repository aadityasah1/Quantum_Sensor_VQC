import sys, pickle, cloudpickle, numpy as np
sys.path.insert(0, ".")

with open("saved_models/vqc_model.pkl", "rb") as f:
    vqc = cloudpickle.load(f)
print("Model loaded:", type(vqc).__name__)
print("Weights shape:", vqc._fit_result.x.shape)

with open("saved_models/metadata.pkl", "rb") as f:
    meta = pickle.load(f)
print("CV accuracy:", round(meta["cv_mean_accuracy"], 4), "+-", round(meta["cv_std_accuracy"], 4))
print("CV AUC-ROC :", round(meta["cv_mean_auc"], 4))
print("Best fold  :", meta["best_fold"], "(fold", meta["best_fold"]+1, "/5)")
print("All artifacts verified OK")
