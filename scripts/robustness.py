"""Robustness analyses for the quality pass.

1. Pause-threshold sensitivity: recompute the writing burst-size contrast
   (AI insertion vs writer's own burst) at 0.5 / 1 / 2 / 3 s thresholds.
2. Feature ablation: 3-class spectrum RF with the top feature removed
   (writing: pause_rate_per_100; code: largest_delta_share).
3. Pairwise Cliff's delta between conditions on the dominant features.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ---- 1. pause-threshold sensitivity (CoAuthor) ----
DATA = Path("data/coauthor/coauthor-v1.0")
THRESHOLDS = [500, 1000, 2000, 3000]

rows = []
for f in sorted(DATA.glob("*.jsonl")):
    events = [json.loads(l) for l in open(f)]
    user_ts, user_sz, ai_sz = [], [], []
    for e in events:
        d = e.get("textDelta")
        if e["eventName"] != "text-insert" or not isinstance(d, dict):
            continue
        ins = sum(len(o["insert"]) if isinstance(o.get("insert"), str) else 1
                  for o in d.get("ops", []) if "insert" in o)
        if ins == 0:
            continue
        if e.get("eventSource") == "api":
            ai_sz.append(ins)
        else:
            user_ts.append(e["eventTimestamp"])
            user_sz.append(ins)
    if len(ai_sz) < 3 or len(user_ts) < 30:
        continue
    ts, sz = np.array(user_ts, float), np.array(user_sz, float)
    gaps = np.diff(ts)
    row = {"ai_mean": np.mean(ai_sz)}
    for th in THRESHOLDS:
        bursts, cur = [], sz[0]
        for g, s in zip(gaps, sz[1:]):
            if g >= th:
                bursts.append(cur)
                cur = 0
            cur += s
        bursts.append(cur)
        row[f"burst_mean_{th}"] = float(np.mean(bursts))
    rows.append(row)

sens = pd.DataFrame(rows)
print(f"== 1. Pause-threshold sensitivity (n={len(sens)} sessions) ==")
for th in THRESHOLDS:
    x = np.log1p(sens["ai_mean"])
    y = np.log1p(sens[f"burst_mean_{th}"])
    d = (x - y).mean() / (x - y).std(ddof=1)
    print(f"threshold {th/1000:.1f}s: human burst mean={sens[f'burst_mean_{th}'].mean():.1f}  "
          f"AI mean={sens['ai_mean'].mean():.1f}  paired_d={d:.2f}")

# ---- 2. feature ablation ----
def ablate(csv, features, drop, label):
    df = pd.read_csv(csv)
    X = df[features].fillna(df[features].median())
    y = df["condition"]
    for name, feats in [("full", features), (f"minus {drop}", [f for f in features if f != drop])]:
        clf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        pred = cross_val_predict(clf, X[feats], y, cv=cv)
        proba = cross_val_predict(clf, X[feats], y, cv=cv, method="predict_proba")
        f1o = f1_score(y, pred, labels=["overreliance_sim"], average=None)[0]
        auc = roc_auc_score(y, proba, multi_class="ovr")
        print(f"{label} [{name}]: overreliance F1={f1o:.3f}  macro AUC={auc:.3f}")

W_FEATS = ["burst_max", "burst_p95", "burst_median", "burst_cv", "iki_median",
           "pause_rate_per_100", "del_ratio", "product_process"]
C_FEATS = ["delta_max", "delta_p95", "delta_median", "delta_cv",
           "largest_delta_share", "del_ratio", "gap_mean_s"]
print("\n== 2. Feature ablation ==")
ablate("results/writing_spectrum.csv", W_FEATS, "pause_rate_per_100", "writing")
ablate("results/code_spectrum.csv", C_FEATS, "largest_delta_share", "code")


# ---- 3. pairwise Cliff's delta ----
def cliffs(a, b):
    a, b = np.asarray(a.dropna()), np.asarray(b.dropna())
    n = 0
    for x in a:
        n += (x > b).sum() - (x < b).sum()
    return n / (len(a) * len(b))


print("\n== 3. Pairwise Cliff's delta ==")
for csv, feat in [("results/writing_spectrum.csv", "burst_max"),
                  ("results/code_spectrum.csv", "largest_delta_share")]:
    df = pd.read_csv(csv)
    g = dict(list(df.groupby("condition")))
    print(f"{csv.split('/')[-1]} / {feat}:")
    for a, b in [("overreliance_sim", "collaborative"), ("overreliance_sim", "human_only"),
                 ("collaborative", "human_only")]:
        print(f"  {a} vs {b}: {cliffs(g[a][feat], g[b][feat]):+.2f}")
