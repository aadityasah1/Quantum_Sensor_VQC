"""
Generate all publication-quality plots from saved CSV results.
Run after experiments have completed and CSVs are in results/.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from evaluation.visualization import (
    plot_model_comparison,
    plot_noise_sweep,
    plot_noise_type_ablation,
)


def generate_all_plots(results_dir: str = "results") -> None:
    print(f"\nGenerating plots from {results_dir}/...")

    # Model comparison bar chart
    comp_path = os.path.join(results_dir, "comparison_results.csv")
    if os.path.exists(comp_path):
        df = pd.read_csv(comp_path)
        fig = plot_model_comparison(df, output_dir=results_dir)
        plt.close(fig)
        print("  Saved model_comparison.png/pdf")
    else:
        print(f"  Skipping model comparison (no {comp_path})")

    # Quantum noise sweep
    sweep_path = os.path.join(results_dir, "quantum_noise_sweep.csv")
    if os.path.exists(sweep_path):
        df = pd.read_csv(sweep_path)
        fig = plot_noise_sweep(df, output_dir=results_dir)
        plt.close(fig)
        print("  Saved noise_sweep.png/pdf")
    else:
        print(f"  Skipping noise sweep (no {sweep_path})")

    # Noise type ablation
    ablation_path = os.path.join(results_dir, "noise_type_ablation.csv")
    if os.path.exists(ablation_path):
        df = pd.read_csv(ablation_path)
        fig = plot_noise_type_ablation(df, output_dir=results_dir)
        plt.close(fig)
        print("  Saved noise_type_ablation.png/pdf")
    else:
        print(f"  Skipping noise ablation (no {ablation_path})")

    print("\nAll available plots generated.")


if __name__ == "__main__":
    generate_all_plots()
