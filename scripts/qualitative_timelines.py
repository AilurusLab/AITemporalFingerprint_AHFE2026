"""Address the 'too good to be true' concern about F1 >= 0.997.

1. Triviality checks: is the sim/real separation carried by a single trivial
   threshold? Report (a) per-feature range overlap between simulated delegation
   and all real sessions, (b) accuracy of a one-feature, one-threshold rule.
2. Qualitative timelines (Fig. 3): cumulative changed characters over time for
   a representative session from each writing condition (minimal-AI,
   collaborative with AI-acceptance marks, simulated delegation), so the
   simulation's realism can be inspected visually.
"""

import json
from pathlib import Path

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

rng = np.random.default_rng(7)
DATA = Path("data/coauthor/coauthor-v1.0")

# ---------- 1. triviality checks ----------
w = pd.read_csv("results/writing_spectrum.csv")
real = w[w["condition"] != "overreliance_sim"]
sim = w[w["condition"] == "overreliance_sim"]

print("== per-feature range overlap (simulated delegation vs ALL real sessions) ==")
for f in ["burst_max", "burst_cv", "pause_rate_per_100", "burst_p95"]:
    lo_s, hi_s = sim[f].min(), sim[f].max()
    share_real_in_sim_range = ((real[f] >= lo_s) & (real[f] <= hi_s)).mean()
    print(f"{f:20s} sim range [{lo_s:8.1f}, {hi_s:8.1f}]  "
          f"real sessions inside sim range: {100*share_real_in_sim_range:5.1f}%")

print("\n== single-feature threshold rule ==")
for f in ["burst_max", "burst_cv"]:
    # best threshold: midpoint between real max-quantiles and sim min
    thr = (real[f].quantile(0.999) + sim[f].min()) / 2
    tp = (sim[f] > thr).mean()
    fp = (real[f] > thr).mean()
    print(f"rule '{f} > {thr:.0f}': catches {100*tp:.1f}% of sim, "
          f"false-flags {100*fp:.2f}% of real ({int((real[f] > thr).sum())} sessions)")

# ---------- 2. qualitative timelines ----------


def stream_of(path):
    events = [json.loads(l) for l in open(path)]
    out = []
    for e in events:
        d = e.get("textDelta")
        if e["eventName"] not in ("text-insert", "text-delete") or not isinstance(d, dict):
            continue
        ins = sum(len(o["insert"]) if isinstance(o.get("insert"), str) else 1
                  for o in d.get("ops", []) if "insert" in o)
        dele = sum(o.get("delete", 0) for o in d.get("ops", []))
        out.append((e["eventTimestamp"], ins + dele, e.get("eventSource") == "api" and ins > 0, ins))
    return out


def pick_session(min_api, max_api):
    """Deterministically pick a representative session: 10-25 min duration,
    total changed chars closest to 2,000."""
    best, best_score = None, None
    for f in sorted(DATA.glob("*.jsonl")):
        s = stream_of(f)
        if len(s) < 100:
            continue
        n_api = sum(1 for _, _, a, _ in s if a)
        if not (min_api <= n_api <= max_api):
            continue
        dur_min = (s[-1][0] - s[0][0]) / 60000
        total = sum(c for _, c, _, _ in s)
        if not (10 <= dur_min <= 25):
            continue
        score = abs(total - 2000)
        if best_score is None or score < best_score:
            best, best_score = (f, s), score
    return best


def simulate_delegation_stream(target_len=2000, iki_median=140.0):
    """Matches the v2 timing model in spectrum_writing.simulate_overreliance."""
    t, out = 0.0, [(0.0, 0, False, 0)]  # anchor at session start
    n_paste = 2
    sizes = rng.dirichlet(np.ones(n_paste)) * target_len * 0.94
    t += rng.uniform(20_000, 90_000)
    for s in sizes:
        out.append((t, int(s), True, int(s)))
        t += rng.uniform(15_000, 60_000)
    t += 1000 * target_len / rng.uniform(12, 30)  # full read-through
    n_chars = int(target_len * 0.08)
    while n_chars > 0:
        edit = min(n_chars, int(rng.integers(3, 20)))
        for _ in range(edit):
            t += float(rng.lognormal(np.log(iki_median), 0.45))
            out.append((t, 1, False, 1))
        if rng.random() < 0.05:
            t += rng.uniform(30_000, 90_000)
        else:
            t += rng.uniform(2_000, 25_000)
        n_chars -= edit
    return out


minimal = pick_session(0, 0)
collab = pick_session(8, 40)
simst = simulate_delegation_stream()

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=False)
panels = [
    ("(a) Minimal-AI (real, zero AI insertions)", minimal[1], "#4878CF"),
    ("(b) Collaborative AI (real)", collab[1], "#6ACC65"),
    ("(c) Simulated wholesale delegation", simst, "#D65F5F"),
]
for ax, (title, s, color) in zip(axes, panels):
    t0 = s[0][0]
    mins = [(t - t0) / 60000 for t, _, _, _ in s]
    cum = np.cumsum([c for _, c, _, _ in s])
    ax.plot(mins, cum, color=color, lw=1.4)
    ai_x = [(t - t0) / 60000 for t, _, a, _ in s if a]
    ai_y = [cum[i] for i, (_, _, a, _) in enumerate(s) if a]
    if ai_x:
        ax.scatter(ai_x, ai_y, s=26, marker="v", color="#333333", zorder=3,
                   label="AI insertion accepted")
        ax.legend(fontsize=12, loc="lower right")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("minutes from session start")
axes[0].set_ylabel("cumulative changed characters")
fig.tight_layout()
fig.savefig("results/fig3_example_timelines.png", dpi=600)
print(f"\npicked minimal-AI session: {minimal[0].name}")
print(f"picked collaborative session: {collab[0].name}")
print("wrote results/fig3_example_timelines.png")
