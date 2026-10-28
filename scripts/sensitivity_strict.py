"""Round-1 review fixes:
1. Strict zero-AI writing subset: do the minimal-AI findings hold when the
   human condition contains no API insertions at all?
2. Kruskal-Wallis restricted to the two REAL conditions per domain (inference
   against the simulated group is not meaningful).
3. Leakage-free imputation (SimpleImputer inside a Pipeline, fitted per fold):
   confirm headline classification numbers are unchanged.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

W_FEATS = ["burst_max", "burst_p95", "burst_median", "burst_cv", "iki_median",
           "pause_rate_per_100", "del_ratio", "product_process"]
C_FEATS = ["delta_max", "delta_p95", "delta_median", "delta_cv",
           "largest_delta_share", "del_ratio", "gap_mean_s"]

w = pd.read_csv("results/writing_spectrum.csv")
c = pd.read_csv("results/code_spectrum.csv")

# ---- 1. strict zero-AI subset ----
strict = w[(w["condition"] == "human_only") & (w["n_api_inserts"] == 0)]
minimal = w[w["condition"] == "human_only"]
collab = w[w["condition"] == "collaborative"]
print(f"strict zero-AI sessions: {len(strict)} of {len(minimal)} minimal-AI")
print(f"burst_max median: strict={strict['burst_max'].median():.0f}  "
      f"minimal-AI={minimal['burst_max'].median():.0f}  collaborative={collab['burst_max'].median():.0f}")
u = stats.mannwhitneyu(strict["burst_max"].dropna(), collab["burst_max"].dropna())
print(f"strict vs collaborative burst_max: Mann-Whitney p={u.pvalue:.2e}")

w_strict = pd.concat([
    strict.assign(condition="human_strict"),
    collab,
    w[w["condition"] == "overreliance_sim"],
])
pipe = make_pipeline(SimpleImputer(strategy="median"),
                     RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced"))
cv = StratifiedKFold(5, shuffle=True, random_state=0)
pred = cross_val_predict(pipe, w_strict[W_FEATS], w_strict["condition"], cv=cv)
print("\n== strict-subset 3-class RF (writing, leakage-free pipeline) ==")
print(classification_report(w_strict["condition"], pred, digits=3))

# ---- 2. KW restricted to real conditions ----
print("== Kruskal-Wallis, real conditions only ==")
for name, df, feats in [("writing", w, W_FEATS), ("code", c, C_FEATS)]:
    real = df[df["condition"] != "overreliance_sim"]
    groups = dict(list(real.groupby("condition")))
    ps = {}
    for f in feats:
        a, b = [g[f].dropna() for g in groups.values()]
        ps[f] = stats.mannwhitneyu(a, b).pvalue
    sig = {k: f"{v:.1e}" for k, v in ps.items()}
    m = len(feats)
    print(f"{name}: {sig}")
    print(f"  surviving Bonferroni (x{m}): {[k for k, v in ps.items() if v * m < .05]}")

# ---- 3. leakage-free pipeline on original conditions ----
print("\n== original 3-class results with leakage-free imputation ==")
for name, df, feats in [("writing", w, W_FEATS), ("code", c, C_FEATS)]:
    pred = cross_val_predict(pipe, df[feats], df["condition"], cv=cv)
    proba = cross_val_predict(pipe, df[feats], df["condition"], cv=cv, method="predict_proba")
    rep = classification_report(df["condition"], pred, digits=3, output_dict=True)
    print(f"{name}: overreliance F1={rep['overreliance_sim']['f1-score']:.3f}  "
          f"acc={rep['accuracy']:.3f}  macro AUC={roc_auc_score(df['condition'], proba, multi_class='ovr'):.3f}")
