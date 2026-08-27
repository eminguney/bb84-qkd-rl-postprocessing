import json
import numpy as np
from scipy import stats

with open("simulation/results.json") as f:
    results = json.load(f)

SCENARIOS = ["clean", "noisy", "attacked", "sparse"]
METHODS = ["DQN", "Q-learning", "Threshold-Rule", "Always-Send", "Always-Retry", "Always-Drop"]
METRICS = ["MSR", "KUE", "SBAR", "ALD"]

# ---- Table 2 style: DQN per-scenario mean over 5 seeds ----
print("=== Per-scenario DQN results (mean over 5 seeds) ===")
for sc in SCENARIOS:
    vals = {m: np.mean([results["DQN"][str(s)][sc][m] for s in range(5)]) for m in METRICS}
    print(f"{sc:10s} MSR={vals['MSR']:.3f} KUE={vals['KUE']:.3f} SBAR={vals['SBAR']:.3f} ALD={vals['ALD']:.2f}")

# ---- Table 3 style: aggregate across scenarios, mean+-std over 5 seeds ----
print()
print("=== Aggregate across 4 scenarios, mean +/- std over 5 seeds ===")
agg = {}
for method in METHODS:
    per_seed_agg = {m: [] for m in METRICS}
    for s in range(5):
        for m in METRICS:
            vals = [results[method][str(s)][sc][m] for sc in SCENARIOS]
            per_seed_agg[m].append(np.mean(vals))
    agg[method] = per_seed_agg
    line = f"{method:15s}"
    for m in METRICS:
        arr = np.array(per_seed_agg[m])
        line += f" {m}={arr.mean():.3f}+/-{arr.std(ddof=1):.3f}"
    print(line)

# ---- Paired t-tests across the 5 seeds (DQN vs others) on MSR and SBAR ----
print()
print("=== Paired t-tests (5 seeds), DQN vs other methods ===")
for method in METHODS:
    if method == "DQN":
        continue
    for m in ["MSR", "SBAR", "KUE"]:
        a = np.array(agg["DQN"][m])
        b = np.array(agg[method][m])
        if np.allclose(a, b):
            print(f"DQN vs {method:15s} [{m}]: identical values, t-test undefined")
            continue
        t, p = stats.ttest_rel(a, b)
        print(f"DQN vs {method:15s} [{m}]: t={t:.4f} p={p:.6f}")

with open("simulation/summary.json", "w") as f:
    json.dump({
        "per_scenario_dqn": {sc: {m: float(np.mean([results["DQN"][str(s)][sc][m] for s in range(5)])) for m in METRICS} for sc in SCENARIOS},
        "aggregate": {method: {m: [float(np.mean(agg[method][m])), float(np.std(agg[method][m], ddof=1))] for m in METRICS} for method in METHODS},
    }, f, indent=2)
print("\nSaved simulation/summary.json")
