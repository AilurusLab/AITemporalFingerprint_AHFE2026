"""External human-only programming baseline from the 2019 CS1 keystroke corpus
(Edwards & Hart, JEDM 2023). No AI condition existed in 2019; every event is
human. Computes per-subject IKI and burst statistics with the same definitions
used for CoAuthor, giving keystroke-level context for what human code
production looks like without AI.
"""

import numpy as np
import pandas as pd

PAUSE = 2000.0

cols = ["SubjectID", "AssignmentID", "ClientTimestamp", "EditType", "InsertText"]
df = pd.read_csv("data/cs1_2019_keystrokes.csv", usecols=cols)
print(f"events: {len(df):,}  subjects: {df['SubjectID'].nunique()}")

ins = df[df["EditType"] == "Insert"].copy()
ins["n"] = ins["InsertText"].astype(str).str.len().clip(lower=1)
ins = ins.sort_values(["SubjectID", "AssignmentID", "ClientTimestamp"])

rows = []
for (subj, _), g in ins.groupby(["SubjectID", "AssignmentID"]):
    ts = g["ClientTimestamp"].to_numpy(float)
    sz = g["n"].to_numpy(float)
    if len(ts) < 200:
        continue
    gaps = np.diff(ts)
    ikis = gaps[(gaps > 0) & (gaps < PAUSE)]
    bursts, cur = [], sz[0]
    for gp, s in zip(gaps, sz[1:]):
        if gp >= PAUSE:
            bursts.append(cur)
            cur = 0
        cur += s
    bursts.append(cur)
    b = np.array(bursts)
    rows.append({
        "subject": subj,
        "iki_median": float(np.median(ikis)) if len(ikis) else np.nan,
        "burst_median": float(np.median(b)),
        "burst_p95": float(np.percentile(b, 95)),
        "burst_max": float(b.max()),
        "single_event_max": float(sz.max()),
    })

r = pd.DataFrame(rows)
r.to_csv("results/cs1_baseline.csv", index=False)
print(f"subject-assignment traces: {len(r)}")
print(r[["iki_median", "burst_median", "burst_p95", "burst_max", "single_event_max"]]
      .median().round(1).to_string())
print("\nshare of traces whose largest SINGLE event exceeds 100 chars (paste-like):",
      round((r["single_event_max"] > 100).mean(), 3))
print("share of traces whose largest 2s-burst exceeds 500 chars:",
      round((r["burst_max"] > 500).mean(), 3))
