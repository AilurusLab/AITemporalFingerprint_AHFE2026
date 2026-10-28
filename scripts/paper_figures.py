"""Publication figures for the AHFE paper."""

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams.update({"font.size": 15, "axes.titlesize": 15,
                            "axes.labelsize": 15, "xtick.labelsize": 15,
                            "ytick.labelsize": 15})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C = {"human_only": "#4878CF", "collaborative": "#6ACC65", "overreliance_sim": "#D65F5F"}
L = {"human_only": "Human-only", "collaborative": "Collaborative AI",
     "overreliance_sim": "Delegation (sim.)"}
LW = dict(L, human_only="Minimal-AI")  # writing panel: condition allows <=2 AI insertions

sess = pd.read_csv("results/coauthor_sessions.csv")
wsp = pd.read_csv("results/writing_spectrum.csv")
csp = pd.read_csv("results/code_spectrum.csv")
rhe = pd.read_csv("results/realhumaneval_participants.csv")

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))

# (a) writing: human bursts vs AI insertion sizes
ax = axes[0, 0]
bins = np.logspace(0.4, 3.2, 36)
d = sess[sess["n_ai_inserts"] >= 3]
ax.hist(d["human_burst_mean"], bins=bins, alpha=0.65, color="#4878CF", label="Human burst (session mean)")
ax.hist(d["ai_insert_size_mean"], bins=bins, alpha=0.65, color="#D65F5F", label="AI insertion (session mean)")
ax.set_xscale("log")
ax.set_xlabel("characters per burst / insertion")
ax.set_ylabel("sessions")
ax.set_title("(a) Writing: insertion size, human vs AI (CoAuthor)")
ax.legend(fontsize=12, loc="upper left")

# (b) code: human vs AI-interval snapshot deltas
ax = axes[0, 1]
bins = np.logspace(0.8, 3.6, 30)
ax.hist(rhe["human_delta_mean"], bins=bins, alpha=0.65, color="#4878CF", label="Human-only interval (mean)")
ax.hist(rhe["ai_delta_mean"], bins=bins, alpha=0.65, color="#D65F5F", label="AI-containing interval (mean)")
ax.set_xscale("log")
ax.set_xlabel("characters inserted per snapshot interval")
ax.set_ylabel("participants")
ax.set_title("(b) Programming: snapshot deltas (RealHumanEval)")
ax.legend(fontsize=12)

M = {"human_only": "o", "collaborative": "^", "overreliance_sim": "s"}

# (c) writing spectrum
ax = axes[1, 0]
for cond, g in wsp.groupby("condition"):
    ax.scatter(g["burst_max"], g["pause_rate_per_100"], s=11, alpha=0.45, c=C[cond],
               marker=M[cond], label=LW[cond])
ax.set_xscale("log")
ax.set_xlabel("largest single burst (chars)")
ax.set_ylabel("pauses ≥ 2 s per 100 insertion events")
ax.set_title("(c) Writing reliance spectrum (observable features)")
ax.legend(fontsize=12)

# (d) code spectrum
ax = axes[1, 1]
for cond, g in csp.groupby("condition"):
    ax.scatter(g["largest_delta_share"], g["delta_cv"], s=16, alpha=0.55, c=C[cond],
               marker=M[cond], label=L[cond])
ax.set_xlabel("share of all insertions in single largest delta")
ax.set_ylabel("coefficient of variation of deltas")
ax.set_title("(d) Programming reliance spectrum (observable features)")
ax.legend(fontsize=12)

fig.tight_layout()
fig.savefig("results/fig1_signatures_spectrum.png", dpi=600)

# figure 2: afterlife
fig2, ax = plt.subplots(figsize=(7.2, 4.6))
cats = ["Writing:\nown text", "Writing:\nAI text", "Code:\nautocomplete", "Code:\nchat-copied"]
retained = [90.4, 92.8, 13.7, 0.0]
partial = [0, 0, 24.1, 18.9]
gone = [9.6, 7.2, 62.2, 81.1]
x = np.arange(len(cats))
ax.bar(x, retained, 0.6, label="retained (>90%)", color="#6ACC65")
ax.bar(x, partial, 0.6, bottom=retained, label="partially edited", color="#EBB434")
ax.bar(x, gone, 0.6, bottom=np.array(retained) + np.array(partial), label="deleted / discarded", color="#D65F5F")
ax.set_xticks(x, cats)
ax.set_ylabel("% of inserted content")
ax.set_title("The afterlife of inserted content: retention in the final artifact")
ax.legend(fontsize=13.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False)
fig2.tight_layout()
fig2.savefig("results/fig2_afterlife.png", dpi=600, bbox_inches="tight")
print("wrote results/fig1_signatures_spectrum.png, results/fig2_afterlife.png")
