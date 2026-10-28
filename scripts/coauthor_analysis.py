"""Preliminary analysis of CoAuthor temporal-signature features.

Paired within-session comparisons (each writer is their own control):
  1. AI insertion size vs the writer's own human burst size
  2. Deletion rate of AI-inserted chars vs self-authored chars (afterlife)
  3. Survival of AI vs human chars in the final document
Effect sizes: paired Cohen's d (on logs where heavy-tailed) + Wilcoxon.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("results/coauthor_sessions.csv")
df = df[df["n_ai_inserts"] >= 3].copy()  # sessions with meaningful AI use
print(f"sessions with >=3 AI insertions: {len(df)} / 1447")


def paired_report(name, a, b, log=False):
    m = a.notna() & b.notna()
    x, y = a[m].to_numpy(float), b[m].to_numpy(float)
    if log:
        x, y = np.log1p(x), np.log1p(y)
    diff = x - y
    d = diff.mean() / diff.std(ddof=1)
    w = stats.wilcoxon(x, y)
    print(
        f"{name}: n={len(x)}  mean_a={np.expm1(x.mean()):.1f}  mean_b={np.expm1(y.mean()):.1f}  "
        f"paired_d={d:.2f}  wilcoxon_p={w.pvalue:.2e}"
        if log
        else f"{name}: n={len(x)}  mean_a={x.mean():.3f}  mean_b={y.mean():.3f}  "
        f"paired_d={d:.2f}  wilcoxon_p={w.pvalue:.2e}"
    )
    return d


print("\n== RQ1: AI insertion size vs writer's own burst size (chars, log-scale) ==")
d1 = paired_report("ai_size vs human_burst_mean", df["ai_insert_size_mean"], df["human_burst_mean"], log=True)
print("median z of AI size vs own burst distribution:", round(df["ai_size_z_vs_own_bursts"].median(), 2))
print("share of sessions where mean AI insert > writer's p95 burst:",
      round((df["ai_insert_size_mean"] > df["human_burst_p95"]).mean(), 3))

print("\n== RQ4: afterlife — deletion rate of AI vs self-authored chars ==")
d2 = paired_report("del_rate_A vs del_rate_H", df["del_rate_A"], df["del_rate_H"])
print("share of sessions where AI text deleted MORE than own text:",
      round((df["del_rate_A"] > df["del_rate_H"]).mean(), 3))

print("\n== survival in final doc ==")
paired_report("survival_A vs survival_H", df["survival_A"], df["survival_H"])

print("\n== timing ==")
print("median human IKI (ms):", df["iki_median_ms"].median())
print("median post-AI-accept latency (ms):", df["post_ai_latency_median_ms"].median())

# figure: burst-size separation
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
bins = np.logspace(0, 3.2, 40)
ax.hist(df["human_burst_mean"].dropna(), bins=bins, alpha=0.6, label="Human burst size (session mean)")
ax.hist(df["ai_insert_size_mean"].dropna(), bins=bins, alpha=0.6, label="AI insertion size (session mean)")
ax.set_xscale("log")
ax.set_xlabel("characters per burst/insertion (log)")
ax.set_ylabel("sessions")
ax.set_title("Insertion burst size: human vs AI")
ax.legend()

ax = axes[1]
m = df["del_rate_A"].notna() & df["del_rate_H"].notna()
ax.scatter(df.loc[m, "del_rate_H"], df.loc[m, "del_rate_A"], s=8, alpha=0.35)
lim = max(df.loc[m, "del_rate_H"].max(), df.loc[m, "del_rate_A"].max())
ax.plot([0, lim], [0, lim], "k--", lw=1)
ax.set_xlabel("deletion rate of self-authored chars")
ax.set_ylabel("deletion rate of AI-inserted chars")
ax.set_title("Afterlife: revision of AI vs own text (per session)")

fig.tight_layout()
Path("results").mkdir(exist_ok=True)
fig.savefig("results/coauthor_prelim.png", dpi=150)
print("\nwrote results/coauthor_prelim.png")
