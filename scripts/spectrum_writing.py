"""Writing-side AI-reliance spectrum: human-only vs collaborative vs simulated overreliance.

Conditions:
  human-only     — CoAuthor sessions with < 3 AI insertions (same population/interface)
  collaborative  — CoAuthor sessions with >= 3 AI insertions
  overreliance   — SIMULATED sessions: 1-3 large pastes of AI text followed by
                   minor edits (3-12% of chars touched) executed at an author-
                   realistic IKI sampled from real sessions. Clearly labeled as
                   simulation; parameters grounded in observed distributions.

All conditions are scored with the SAME observable session-level features
(no provenance labels used), so the analysis applies to unlabeled logs:
  max/p95/median burst, burst CV, median IKI, pause rate, deletion ratio,
  product/process ratio.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PAUSE = 2000.0
rng = np.random.default_rng(7)

DATA = Path("data/coauthor/coauthor-v1.0")


def observables(stream, final_len, total_ins, total_del):
    """stream: list of (ts, ins_chars) insertion events (any origin)."""
    ts = np.array([t for t, _ in stream], float)
    sz = np.array([s for _, s in stream], float)
    gaps = np.diff(ts)
    bursts, cur = [], sz[0]
    for g, s in zip(gaps, sz[1:]):
        if g >= PAUSE:
            bursts.append(cur)
            cur = 0
        cur += s
    bursts.append(cur)
    b = np.array(bursts, float)
    ikis = gaps[(gaps > 0) & (gaps < PAUSE)]
    dur_min = (ts[-1] - ts[0]) / 60000 if len(ts) > 1 else np.nan
    return {
        "burst_max": b.max(),
        "burst_p95": float(np.percentile(b, 95)),
        "burst_median": float(np.median(b)),
        "burst_cv": float(b.std() / b.mean()) if b.mean() > 0 else np.nan,
        "iki_median": float(np.median(ikis)) if len(ikis) else np.nan,
        "pause_rate_per_100": 100 * float((gaps >= PAUSE).sum() / max(1, len(sz))),
        "del_ratio": total_del / total_ins if total_ins else np.nan,
        "product_process": final_len / total_ins if total_ins else np.nan,
        "duration_min": dur_min,
    }


def parse_session(path):
    events = [json.loads(l) for l in open(path)]
    stream, total_ins, total_del, doc_len, n_api = [], 0, 0, 0, 0
    for e in events:
        d = e.get("textDelta")
        if e["eventName"] not in ("text-insert", "text-delete") or not isinstance(d, dict):
            continue
        ins = sum(len(o["insert"]) if isinstance(o.get("insert"), str) else (1 if "insert" in o else 0)
                  for o in d.get("ops", []))
        dele = sum(o.get("delete", 0) for o in d.get("ops", []))
        total_ins += ins
        total_del += dele
        doc_len += ins - dele
        if ins > 0:
            stream.append((e["eventTimestamp"], ins))
            if e.get("eventSource") == "api":
                n_api += 1
    if total_ins < 200 or len(stream) < 30:
        return None
    row = observables(stream, doc_len, total_ins, total_del)
    row["n_api_inserts"] = n_api
    return row


def simulate_overreliance(target_len, iki_median):
    """One simulated wholesale-paste-then-polish session.

    Timing model (v2, after qualitative-timeline inspection): the polish phase
    now includes a full read-through of the pasted text at 12-30 chars/s and
    reading-scale pauses between point edits (2-25 s, with occasional longer
    re-reads), so simulated sessions have realistic durations instead of the
    v1 model's compressed 1-8 s scan gaps.
    """
    t = 0.0
    stream, total_ins, total_del = [], 0, 0
    sigma = 0.45
    # paste phase: 1-3 pastes with read pauses between
    n_paste = rng.integers(1, 4)
    sizes = rng.dirichlet(np.ones(n_paste)) * target_len * rng.uniform(0.90, 0.98)
    t += rng.uniform(20_000, 90_000)  # initial prompt-reading / AI-querying time
    for s in sizes:
        stream.append((t, int(max(1, s))))
        total_ins += int(max(1, s))
        t += rng.uniform(15_000, 60_000)  # skim what was pasted
    # full read-through of the pasted document before editing
    t += 1000 * target_len / rng.uniform(12, 30)
    # polish phase: touch 3-12% of chars in small point edits
    frac = rng.uniform(0.03, 0.12)
    n_chars = int(target_len * frac)
    while n_chars > 0:
        edit = min(n_chars, int(rng.integers(3, 20)))
        n_dele = int(rng.integers(0, edit + 1))
        total_del += n_dele
        for _ in range(edit):  # typed at the author's own IKI
            t += float(rng.lognormal(np.log(max(60, iki_median)), sigma))
            stream.append((t, 1))
            total_ins += 1
        if rng.random() < 0.05:
            t += rng.uniform(30_000, 90_000)  # occasional longer re-read
        else:
            t += rng.uniform(2_000, 25_000)  # read to next edit point
        n_chars -= edit
    return observables(stream, total_ins - total_del, total_ins, total_del)


rows = []
for f in sorted(DATA.glob("*.jsonl")):
    r = parse_session(f)
    if r:
        r["condition"] = "human_only" if r["n_api_inserts"] < 3 else "collaborative"
        r["session"] = f.stem
        rows.append(r)

real = pd.DataFrame(rows)
print(real["condition"].value_counts().to_string())

# ground simulation in real session parameters: document lengths in the CoAuthor
# essay range (~418 words avg) and IKIs resampled from real writers
sim_rows = []
targets = rng.uniform(800, 3500, size=400)
ikis = real["iki_median"].dropna().to_numpy()
for i in range(400):
    r = simulate_overreliance(targets[i], float(rng.choice(ikis)))
    r["condition"] = "overreliance_sim"
    r["session"] = f"sim{i}"
    r["n_api_inserts"] = np.nan
    sim_rows.append(r)

df = pd.concat([real, pd.DataFrame(sim_rows)], ignore_index=True)
df.to_csv("results/writing_spectrum.csv", index=False)

FEATURES = ["burst_max", "burst_p95", "burst_median", "burst_cv", "iki_median",
            "pause_rate_per_100", "del_ratio", "product_process"]

print("\n== condition medians ==")
print(df.groupby("condition")[FEATURES].median().round(2).to_string())

print("\n== Kruskal-Wallis across three conditions ==")
for feat in FEATURES:
    groups = [g[feat].dropna() for _, g in df.groupby("condition")]
    h, p = stats.kruskal(*groups)
    print(f"{feat:22s} H={h:8.1f}  p={p:.2e}")

# 3-class classifier on observable features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score

X = df[FEATURES].fillna(df[FEATURES].median())
y = df["condition"]
clf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")
pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
proba = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")
print("\n== 3-class RF (5-fold CV, observable features only) ==")
print(classification_report(y, pred, digits=3))
print("macro OVR AUC:", round(roc_auc_score(y, proba, multi_class="ovr"), 3))

clf.fit(X, y)
imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("feature importances:\n", imp.round(3).to_string())
imp.to_csv("results/writing_spectrum_importances.csv")
