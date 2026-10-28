"""Programming-side AI-reliance spectrum at snapshot granularity.

Conditions (per participant-task trace):
  human_only       — RealHumanEval participants with zero AI events in the trace
  collaborative    — participants with >= 2 AI events (accepted suggestion / chat copy)
  overreliance_sim — SIMULATED traces: one or two large pastes early, then small
                     scattered edits; snapshot cadence and edit sizes grounded in
                     the real human-only distributions.

Observable features per trace (no AI labels used): max/p95/median snapshot
insertion delta, delta CV, share of insertions arriving in the single largest
delta, deletion ratio, mean inter-snapshot gap.
"""

import ast
import json
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

rng = np.random.default_rng(11)

study = pd.read_csv("data/realhumaneval/study_data.csv")
ac = pd.read_csv("data/realhumaneval/autocomplete_data.csv")
chat = pd.read_csv("data/realhumaneval/chat_data.csv")


def deltas(a, b):
    sm = SequenceMatcher(None, a, b, autojunk=False)
    ins = dele = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            ins += j2 - j1
        if tag in ("delete", "replace"):
            dele += i2 - i1
    return ins, dele


def observables(ins_list, del_list, gap_list):
    ins = np.array(ins_list, float)
    if len(ins) < 5 or ins.sum() < 100:
        return None
    return {
        "delta_max": float(ins.max()),
        "delta_p95": float(np.percentile(ins, 95)),
        "delta_median": float(np.median(ins[ins > 0])) if (ins > 0).any() else np.nan,
        "delta_cv": float(ins.std() / ins.mean()) if ins.mean() > 0 else np.nan,
        "largest_delta_share": float(ins.max() / ins.sum()),
        "del_ratio": float(np.sum(del_list) / ins.sum()),
        "gap_mean_s": float(np.mean(gap_list) / 1000),
        "n_snapshots": len(ins),
    }


rows = []
human_gap_pool, human_ins_pool = [], []
for _, p in study.iterrows():
    pid = p["programmer_id"]
    try:
        h = json.loads(p["code_history"])
    except (TypeError, json.JSONDecodeError):
        continue
    idx = sorted(h["code"].keys(), key=int)
    snaps = [(h["times"][k], h["code"][k] or "", h["task_name"][k]) for k in idx]

    n_ai = int(((ac["programmer_id"] == pid) & (ac["accepted"] == 1)).sum())
    my_chat = chat[chat["programmer_id"] == pid]
    for _, c in my_chat.iterrows():
        try:
            evs = ast.literal_eval(c["copy_events"]) if isinstance(c["copy_events"], str) else []
        except (ValueError, SyntaxError):
            evs = []
        n_ai += sum(1 for e in evs if e.get("copied"))

    ins_l, del_l, gap_l = [], [], []
    for (t0, c0, k0), (t1, c1, k1) in zip(snaps, snaps[1:]):
        if k0 != k1:
            continue
        i, d = deltas(c0, c1)
        ins_l.append(i)
        del_l.append(d)
        gap_l.append(t1 - t0)
    r = observables(ins_l, del_l, gap_l)
    if not r:
        continue
    if n_ai == 0:
        r["condition"] = "human_only"
        human_gap_pool.extend(gap_l)
        human_ins_pool.extend([x for x in ins_l if x > 0])
    elif n_ai >= 2:
        r["condition"] = "collaborative"
    else:
        continue
    r["pid"] = pid
    rows.append(r)

real = pd.DataFrame(rows)
print(real["condition"].value_counts().to_string())

# simulated overreliance traces grounded in human-only pools
gap_pool = np.array(human_gap_pool, float)
gap_pool = gap_pool[(gap_pool > 0) & (gap_pool < 600_000)]
ins_pool = np.array(human_ins_pool, float)
sim = []
for i in range(150):
    total = rng.uniform(300, 1500)  # final solution size (chars)
    n_paste = rng.integers(1, 3)
    paste_sizes = rng.dirichlet(np.ones(n_paste)) * total * rng.uniform(0.85, 0.97)
    n_snap = int(rng.integers(8, 30))
    ins_l = [0.0] * n_snap
    del_l = [0.0] * n_snap
    pos = sorted(rng.choice(range(0, max(2, n_snap // 3)), size=n_paste, replace=False))
    for ppos, s in zip(pos, paste_sizes):
        ins_l[ppos] = float(s)
    # polish: small edits in later snapshots, sized like real small human deltas
    small = ins_pool[ins_pool < np.percentile(ins_pool, 60)]
    for j in range(max(pos) + 1, n_snap):
        if rng.random() < 0.6:
            ins_l[j] = float(rng.choice(small))
            del_l[j] = float(rng.choice(small)) * rng.uniform(0, 0.8)
    gap_l = list(rng.choice(gap_pool, size=n_snap))
    r = observables(ins_l, del_l, gap_l)
    if r:
        r["condition"] = "overreliance_sim"
        r["pid"] = f"sim{i}"
        sim.append(r)

df = pd.concat([real, pd.DataFrame(sim)], ignore_index=True)
df.to_csv("results/code_spectrum.csv", index=False)

FEATURES = ["delta_max", "delta_p95", "delta_median", "delta_cv",
            "largest_delta_share", "del_ratio", "gap_mean_s"]

print("\n== condition medians ==")
print(df.groupby("condition")[FEATURES].median().round(2).to_string())

print("\n== Kruskal-Wallis ==")
for feat in FEATURES:
    groups = [g[feat].dropna() for _, g in df.groupby("condition")]
    h, p = stats.kruskal(*groups)
    print(f"{feat:20s} H={h:7.1f}  p={p:.2e}")

X = df[FEATURES].fillna(df[FEATURES].median())
y = df["condition"]
clf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")
pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
proba = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")
print("\n== 3-class RF (5-fold CV) ==")
print(classification_report(y, pred, digits=3))
print("macro OVR AUC:", round(roc_auc_score(y, proba, multi_class="ovr"), 3))
clf.fit(X, y)
print("importances:\n", pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False).round(3).to_string())
