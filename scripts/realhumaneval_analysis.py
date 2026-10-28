"""Paired analysis for RealHumanEval + graded survival of AI code spans.

Graded survival: fraction of an AI span's 20-char normalized shingles present in
the final snapshot of its task (1.0 = fully retained, 0 = gone). Distinguishes
'edited but retained' from 'discarded', which verbatim matching cannot.
"""

import ast
import json
import re

import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("results/realhumaneval_participants.csv")

print(f"n participants (paired): {len(df)}")
print("\n== AI vs human snapshot insertion deltas (chars, log paired) ==")
x = np.log1p(df["ai_delta_mean"])
y = np.log1p(df["human_delta_mean"])
d = (x - y).mean() / (x - y).std(ddof=1)
w = stats.wilcoxon(x, y)
print(f"ai_delta mean={df['ai_delta_mean'].mean():.0f}  human_delta mean={df['human_delta_mean'].mean():.0f}  "
      f"paired_d={d:.2f}  p={w.pvalue:.2e}")
print("share where AI-interval delta > participant p95 human delta:",
      round((df["ai_delta_mean"] > df["human_delta_p95"]).mean(), 3))

print("\n== churn after AI intervals vs after human intervals ==")
m = df["post_ai_del_mean"].notna() & df["post_h_del_mean"].notna()
x = np.log1p(df.loc[m, "post_ai_del_mean"]); y = np.log1p(df.loc[m, "post_h_del_mean"])
dd = (x - y).mean() / (x - y).std(ddof=1)
w2 = stats.wilcoxon(x, y)
print(f"n={m.sum()}  post_ai_del mean={df.loc[m,'post_ai_del_mean'].mean():.0f}  "
      f"post_h_del mean={df.loc[m,'post_h_del_mean'].mean():.0f}  paired_d={dd:.2f}  p={w2.pvalue:.2e}")

# graded survival
study = pd.read_csv("data/realhumaneval/study_data.csv")
ac = pd.read_csv("data/realhumaneval/autocomplete_data.csv")
chat = pd.read_csv("data/realhumaneval/chat_data.csv")


def norm(s):
    return re.sub(r"\s+", "", s or "")


def shingles(s, k=20):
    return {s[i : i + k] for i in range(0, max(1, len(s) - k + 1))}


vals = {"autocomplete": [], "chat_copy": []}
for _, p in study.iterrows():
    pid = p["programmer_id"]
    try:
        h = json.loads(p["code_history"])
    except (TypeError, json.JSONDecodeError):
        continue
    idx = sorted(h["code"].keys(), key=int)
    finals = {}
    for k in idx:
        finals[h["task_name"][k]] = h["code"][k] or ""

    spans = []
    my_ac = ac[(ac["programmer_id"] == pid) & (ac["accepted"] == 1)]
    spans += [("autocomplete", s or "", tn) for s, tn in zip(my_ac["suggestion"], my_ac["task_name"])]
    my_chat = chat[chat["programmer_id"] == pid]
    for _, c in my_chat.iterrows():
        try:
            evs = ast.literal_eval(c["copy_events"]) if isinstance(c["copy_events"], str) else []
        except (ValueError, SyntaxError):
            evs = []
        spans += [("chat_copy", e.get("copied", ""), c["task_name"]) for e in evs if e.get("copied")]

    for kind, text, tn in spans:
        f = finals.get(tn)
        nt = norm(text)
        if f is None or len(nt) < 40:
            continue
        sh = shingles(nt)
        nf = norm(f)
        frac = np.mean([s in nf for s in sh])
        vals[kind].append(frac)

for kind, v in vals.items():
    v = np.array(v)
    if len(v):
        print(f"\n== graded survival ({kind}): n={len(v)} spans ==")
        print(f"mean shingle survival: {v.mean():.3f}  |  fully retained (>0.9): {(v>0.9).mean():.3f}  "
              f"|  discarded (<0.1): {(v<0.1).mean():.3f}  |  partially edited (0.1-0.9): {((v>=0.1)&(v<=0.9)).mean():.3f}")
