import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("simulation/summary.json") as f:
    summary = json.load(f)

agg = summary["aggregate"]
methods = ["Always-Send", "Always-Retry", "Always-Drop", "Threshold-Rule", "Q-learning", "DQN"]
labels = ["Always-\nSend", "Always-\nRetry", "Always-\nDrop", "Threshold-\nRule", "Q-\nlearning", "Proposed\nDQN"]
colors = ["#E69F00", "#9C27B0", "#8895A7", "#2CA02C", "#D62728", "#1F77B4"]

metrics = [("MSR", "Message Success Rate (MSR)"),
           ("KUE", "Key Usage Efficiency (KUE)"),
           ("SBAR", "Security Breach Avoidance Rate (SBAR)"),
           ("ALD", "Average Latency per Decision (ALD, s)")]

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle("Proposed DQN policy vs. tabular Q-learning, threshold rule, and static baselines\n(aggregated across scenarios, mean $\\pm$ std over 5 seeds)",
              fontsize=13, fontweight="bold")

for ax, (key, title) in zip(axes.flat, metrics):
    means = [agg[m][key][0] for m in methods]
    stds = [agg[m][key][1] for m in methods]
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.tick_params(axis="y", labelsize=9.5)
    top = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(0, top * 1.22 + 0.05)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + top * 0.03, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("figures/fig7_dqn_vs_baselines_v2.png", dpi=200)
fig.savefig("figures/fig7_dqn_vs_baselines_v2.pdf")
print("Saved figures/fig7_dqn_vs_baselines_v2.png and .pdf")
