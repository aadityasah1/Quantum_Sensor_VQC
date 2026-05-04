import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/results.csv")

plt.bar(df["Model"], df["Accuracy"])
plt.title("Quantum vs Classical Performance")
plt.ylabel("Accuracy")
plt.savefig("results/performance_plot.png")
print("Plot saved to results/performance_plot.png")
print(df)