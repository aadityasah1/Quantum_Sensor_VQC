"""
Generate the extra figures for the LaTeX research paper.

Outputs (PNG + PDF, 300 DPI) into results/:
  fig_pipeline.png      block diagram of the end-to-end pipeline
  fig_signals.png       Normal vs Fault sensor waveforms (physics of the data)
  fig_pca.png           PCA 2-D projection + explained-variance spectrum
  fig_featuremap.png    ZZ feature-map circuit (data encoding)
  fig_ansatz.png        EfficientSU2 ansatz circuit (trainable block)
  fig_circuit_full.png  full VQC circuit (feature map + ansatz)
  fig_zne.png           Zero-Noise Extrapolation fit at the device-noise point
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import zz_feature_map, efficient_su2

# this file lives in scripts/ ; put the project root on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from data.generate_data import generate_sensor_data

RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
SEED = 42
ACCENT = "#1F4E79"
QBLUE = "#3B6EA5"
GREEN = "#4C9F70"
RED = "#C44E52"
ORANGE = "#DD8452"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "figure.dpi": 150, "savefig.dpi": 300,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RES, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# -----------------------------------------------------------------------------
# 1. Sensor signal waveforms (the physics behind the data)
# -----------------------------------------------------------------------------
def fig_signals():
    n_features = 8
    t_dense = np.linspace(0, 0.02, 500)
    t_samp = np.linspace(0, 0.02, n_features)
    omega0, omega1, zeta, A = 2*np.pi*50, 2*np.pi*180, 80.0, 1.0
    snr = 3.0
    rng = np.random.default_rng(SEED)

    def normal(t):
        return A*np.exp(-zeta*t)*np.cos(omega0*t)

    def fault(t):
        return A*np.exp(-zeta*t)*np.cos(omega1*t) + 0.4*np.cos(3*omega1*t)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, fn, col) in zip(
            axes,
            [("Class 0 — Normal bearing (50 Hz)", normal, GREEN),
             ("Class 1 — Fault bearing (180 Hz + 3rd harmonic)", fault, RED)]):
        s_dense = fn(t_dense)
        s_samp = fn(t_samp)
        noise_std = np.sqrt(np.mean(s_samp**2)/snr)
        s_noisy = s_samp + rng.normal(0, noise_std, n_features)
        ax.plot(t_dense*1e3, s_dense, color=col, lw=2, label="underlying signal")
        ax.plot(t_samp*1e3, s_noisy, "o", color="black", ms=7,
                label="8 sampled features (noisy)")
        ax.vlines(t_samp*1e3, 0, s_noisy, color="gray", lw=0.8, alpha=0.5)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("time (ms)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")
    axes[0].set_ylabel("amplitude (a.u.)")
    fig.suptitle("Physics-inspired sensor signals (SNR = 3)", fontsize=13)
    fig.tight_layout()
    save(fig, "fig_signals")


# -----------------------------------------------------------------------------
# 2. PCA projection + explained-variance spectrum
# -----------------------------------------------------------------------------
def fig_pca():
    X, y = generate_sensor_data(n_samples=240, n_features=8, snr=3.0, seed=SEED)
    pca = PCA(n_components=4, random_state=SEED)
    Xp = pca.fit_transform(X)
    Xs = MinMaxScaler((0, 1)).fit_transform(Xp)
    evr = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for cls, col, lab in [(0, GREEN, "Normal"), (1, RED, "Fault")]:
        m = y == cls
        ax.scatter(Xs[m, 0], Xs[m, 1], c=col, label=lab, alpha=0.7,
                   edgecolors="k", linewidths=0.3, s=40)
    ax.set_xlabel("PC 1 (scaled to [0,1])")
    ax.set_ylabel("PC 2 (scaled to [0,1])")
    ax.set_title("PCA projection of sensor data (2 of 4 components)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    comps = [f"PC{i+1}" for i in range(4)]
    ax2.bar(comps, evr*100, color=QBLUE, alpha=0.9)
    ax2.plot(comps, np.cumsum(evr)*100, "o-", color=ORANGE,
             label="cumulative")
    for i, v in enumerate(evr):
        ax2.text(i, v*100+1, f"{v*100:.0f}%", ha="center", fontsize=9)
    ax2.set_ylabel("variance explained (%)")
    ax2.set_title(f"Explained variance (4 qubits = {evr.sum()*100:.0f}%)")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save(fig, "fig_pca")


# -----------------------------------------------------------------------------
# 3 & 4 & 5. Circuit diagrams
# -----------------------------------------------------------------------------
def fig_circuits():
    # one repetition shown for readability (experiment uses reps=2)
    fm1 = zz_feature_map(feature_dimension=4, reps=1)
    # one trainable block (16 params); the VQC re-uploads data and stacks 3 of
    # these for 48 parameters total
    ans = efficient_su2(num_qubits=4, reps=1, entanglement="linear")

    f1 = fm1.draw("mpl", fold=-1, style={"name": "iqp"})
    save(f1, "fig_featuremap")

    f2 = ans.draw("mpl", fold=-1)
    save(f2, "fig_ansatz")


# -----------------------------------------------------------------------------
# 6. ZNE extrapolation fit
# -----------------------------------------------------------------------------
def fig_zne():
    import csv
    scales, accs = [], []
    with open(os.path.join(RES, "month1_zne.csv")) as f:
        for row in csv.DictReader(f):
            if row["scale_factor"] == "extrapolated":
                continue
            scales.append(float(row["scale_factor"]))
            accs.append(float(row["accuracy"]))
    scales, accs = np.array(scales, dtype=float), np.array(accs)
    coeffs = np.polyfit(scales, accs, 1)
    x_line = np.linspace(0, 5.3, 100)
    y_line = np.polyval(coeffs, x_line)
    extrap = float(np.polyval(coeffs, 0.0))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(x_line, y_line, "--", color="gray", lw=1.6,
            label="linear (Richardson) fit")
    ax.scatter(scales, accs, color=QBLUE, s=90, zorder=5,
               label="measured (noisy) accuracy")
    ax.scatter([0], [extrap], color=ORANGE, marker="^", s=160, zorder=6,
               edgecolors="k", linewidths=0.6,
               label=f"ZNE estimate = {extrap:.3f}")
    ax.scatter([1], [accs[0]], facecolors="none", edgecolors=RED, s=220,
               linewidths=1.8, zorder=4, label=f"raw noisy = {accs[0]:.3f}")
    ax.annotate(f"+{(extrap-accs[0])*100:.1f} pts",
                xy=(0, extrap), xytext=(1.3, extrap+0.02),
                fontsize=10, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.set_xlabel("noise scale factor  (unitary gate folding)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Zero-Noise Extrapolation at device noise (2q depol = 0.02)")
    ax.set_xticks([0, 1, 3, 5])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_zne")


# -----------------------------------------------------------------------------
# 7. Pipeline block diagram
# -----------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc, tc="white", fs=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                         linewidth=1.2, edgecolor="#333333", facecolor=fc)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center",
            color=tc, fontsize=fs, weight="bold", wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=14, lw=1.4, color="#555555"))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")

    _box(ax, 0.1, 1.8, 1.9, 1.0, "Sensor signals\nNormal / Fault", GREEN, fs=9.5)
    _box(ax, 2.3, 1.8, 1.6, 1.0, "PCA\n8 -> 4", "#6C8EBF", fs=10)
    _box(ax, 4.2, 1.8, 1.7, 1.0, "MinMax\nscale [0,1]", "#6C8EBF", fs=10)
    _box(ax, 6.2, 1.8, 2.0, 1.0, "ZZ feature map\n(data encoding)", QBLUE, fs=9.5)

    # two quantum branches
    _box(ax, 8.6, 3.0, 3.2, 0.95, "Fidelity kernel  ->  QSVC", ACCENT, fs=9.5)
    _box(ax, 8.6, 1.85, 3.2, 0.95, "EfficientSU2 ansatz  ->  VQC", ACCENT, fs=9.5)
    _box(ax, 8.6, 0.55, 3.2, 0.95, "Noise model + ZNE\n(resilience study)", RED, fs=9)

    _arrow(ax, 2.0, 2.3, 2.3, 2.3)
    _arrow(ax, 3.9, 2.3, 4.2, 2.3)
    _arrow(ax, 5.9, 2.3, 6.2, 2.3)
    _arrow(ax, 8.2, 2.5, 8.6, 3.45)   # to QSVC
    _arrow(ax, 8.2, 2.3, 8.6, 2.32)   # to VQC
    _arrow(ax, 10.2, 1.85, 10.2, 1.5) # VQC -> noise study

    ax.text(6, 4.25, "End-to-end quantum sensor classification pipeline",
            ha="center", fontsize=13, weight="bold", color=ACCENT)
    fig.tight_layout()
    save(fig, "fig_pipeline")


if __name__ == "__main__":
    fig_signals()
    fig_pca()
    fig_circuits()
    fig_zne()
    fig_pipeline()
    print("ALL FIGURES DONE")
