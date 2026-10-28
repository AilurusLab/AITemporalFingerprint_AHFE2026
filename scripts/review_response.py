"""Analyses responding to mock-review round 1.

1. Confusion matrices for both spectrum classifiers, with the fairness-critical
   number: how many REAL sessions/traces are misclassified INTO the simulated
   delegation class.
2. Evasion variant (writing): chunked pasting: 6-12 medium pastes with paced
   gaps instead of 1-3 large pastes, same polish phase. Train on the original
   three conditions, test recall on the evasion variant.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

rng = np.random.default_rng(23)

W_FEATS = ["burst_max", "burst_p95", "burst_median", "burst_cv", "iki_median",
           "pause_rate_per_100", "del_ratio", "product_process"]
C_FEATS = ["delta_max", "delta_p95", "delta_median", "delta_cv",
           "largest_delta_share", "del_ratio", "gap_mean_s"]

for name, csv, feats in [("writing", "results/writing_spectrum.csv", W_FEATS),
                         ("code", "results/code_spectrum.csv", C_FEATS)]:
    df = pd.read_csv(csv)
    X = df[feats].fillna(df[feats].median())
    y = df["condition"]
    clf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")
    pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    labels = ["human_only", "collaborative", "overreliance_sim"]
    cm = confusion_matrix(y, pred, labels=labels)
    print(f"== {name} confusion matrix (rows=true, cols=pred) {labels} ==")
    print(cm)
    real_into_sim = cm[0, 2] + cm[1, 2]
    n_real = cm[0].sum() + cm[1].sum()
    print(f"real sessions misclassified as delegation: {real_into_sim}/{n_real} "
          f"({100*real_into_sim/n_real:.2f}%)\n")

# ---- evasion variant (writing) ----
from spectrum_writing import observables  # reuse the exact feature code

wsp = pd.read_csv("results/writing_spectrum.csv")
ikis = wsp["iki_median"].dropna().to_numpy()


def simulate_chunked(target_len, iki_median):
    t = 0.0
    stream, total_ins, total_del = [], 0, 0
    n_paste = rng.integers(6, 13)
    sizes = rng.dirichlet(np.ones(n_paste)) * target_len * rng.uniform(0.90, 0.98)
    t += rng.uniform(20_000, 90_000)
    for s in sizes:
        stream.append((t, int(max(1, s))))
        total_ins += int(max(1, s))
        t += rng.uniform(20_000, 120_000)  # paced: read/scroll between chunks
    frac = rng.uniform(0.03, 0.12)
    n_chars = int(target_len * frac)
    while n_chars > 0:
        edit = min(n_chars, int(rng.integers(3, 20)))
        total_del += int(rng.integers(0, edit + 1))
        for _ in range(edit):
            t += float(rng.lognormal(np.log(max(60, iki_median)), 0.45))
            stream.append((t, 1))
            total_ins += 1
        t += rng.uniform(1_000, 8_000)
        n_chars -= edit
    return observables(stream, total_ins - total_del, total_ins, total_del)


ev = []
for i in range(200):
    r = simulate_chunked(rng.uniform(800, 3500), float(rng.choice(ikis)))
    ev.append(r)
evdf = pd.DataFrame(ev)

clf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")
clf.fit(wsp[W_FEATS].fillna(wsp[W_FEATS].median()), wsp["condition"])
pred = clf.predict(evdf[W_FEATS].fillna(wsp[W_FEATS].median()))
counts = pd.Series(pred).value_counts()
print("== evasion variant (6-12 medium pastes, paced): classified as ==")
print(counts.to_string())
print(f"caught as delegation: {100*counts.get('overreliance_sim', 0)/len(evdf):.1f}%")
print("evasion medians:", evdf[["burst_max", "burst_cv", "pause_rate_per_100"]].median().round(1).to_dict())
