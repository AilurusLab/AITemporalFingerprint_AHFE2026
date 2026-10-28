"""Temporal-signature features from RealHumanEval (programming, event/snapshot level).

Per participant:
  - snapshot deltas (inserted/deleted chars between consecutive editor snapshots,
    via SequenceMatcher) within the same task
  - intervals classified AI-containing (accepted autocomplete or chat-copy event
    falls inside) vs human-only
  - paired comparison of insertion delta sizes; post-interval churn
  - afterlife: survival of AI-inserted code (whitespace-normalized substring
    match against the participant's final snapshot for that task)
"""

import ast
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

study = pd.read_csv("data/realhumaneval/study_data.csv")
ac = pd.read_csv("data/realhumaneval/autocomplete_data.csv")
chat = pd.read_csv("data/realhumaneval/chat_data.csv")

id_col = next(c for c in study.columns if "id" in c.lower() and "task" not in c.lower())
print("participant id column:", id_col)


def norm(s):
    return re.sub(r"\s+", "", s or "")


def deltas(a, b):
    sm = SequenceMatcher(None, a, b, autojunk=False)
    ins = dele = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            ins += j2 - j1
        if tag in ("delete", "replace"):
            dele += i2 - i1
    return ins, dele


rows = []
surv_rows = []
for _, p in study.iterrows():
    pid = p[id_col]
    try:
        h = json.loads(p["code_history"])
    except (TypeError, json.JSONDecodeError):
        continue
    idx = sorted(h["code"].keys(), key=int)
    snaps = [(h["times"][k], h["code"][k] or "", h["task_name"][k]) for k in idx]

    my_ac = ac[(ac[  # accepted suggestions for this participant
        "programmer_id"] == pid) & (ac["accepted"] == 1)]
    ai_events = [(t, len(s or "")) for t, s in zip(my_ac["timestamp"], my_ac["suggestion"])]
    ai_texts = [(t, s or "", tn) for t, s, tn in zip(my_ac["timestamp"], my_ac["suggestion"], my_ac["task_name"])]

    my_chat = chat[chat["programmer_id"] == pid] if "programmer_id" in chat.columns else chat.iloc[0:0]
    for _, c in my_chat.iterrows():
        try:
            evs = ast.literal_eval(c["copy_events"]) if isinstance(c["copy_events"], str) else []
        except (ValueError, SyntaxError):
            evs = []
        for e in evs:
            copied = e.get("copied", "")
            ts = e.get("timestamp", c["timestamp"])
            if copied:
                ai_events.append((ts, len(copied)))
                ai_texts.append((ts, copied, c.get("task_name", "")))

    ai_ts = np.array([t for t, _ in ai_events], dtype=float)

    human_ins, ai_ins, post_ai_del, post_h_del = [], [], [], []
    prev_ai = False
    for (t0, c0, task0), (t1, c1, task1) in zip(snaps, snaps[1:]):
        if task0 != task1:
            prev_ai = False
            continue
        ins, dele = deltas(c0, c1)
        has_ai = bool(len(ai_ts) and ((ai_ts > t0) & (ai_ts <= t1)).any())
        if ins > 0:
            (ai_ins if has_ai else human_ins).append(ins)
        if prev_ai:
            post_ai_del.append(dele)
        else:
            post_h_del.append(dele)
        prev_ai = has_ai

    # afterlife: survival of AI text in final snapshot of its task
    finals = {}
    for t, c, tn in snaps:
        finals[tn] = c
    for t, text, tn in ai_texts:
        f = finals.get(tn)
        if f is not None and len(norm(text)) >= 20:
            surv_rows.append({"pid": pid, "kind": "ai", "survived": norm(text) in norm(f)})

    if len(human_ins) >= 5 and len(ai_ins) >= 2:
        rows.append({
            "pid": pid,
            "n_snapshots": len(snaps),
            "n_ai_events": len(ai_events),
            "human_delta_mean": float(np.mean(human_ins)),
            "human_delta_p95": float(np.percentile(human_ins, 95)),
            "ai_delta_mean": float(np.mean(ai_ins)),
            "post_ai_del_mean": float(np.mean(post_ai_del)) if post_ai_del else np.nan,
            "post_h_del_mean": float(np.mean(post_h_del)) if post_h_del else np.nan,
        })

df = pd.DataFrame(rows)
sv = pd.DataFrame(surv_rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/realhumaneval_participants.csv", index=False)
sv.to_csv("results/realhumaneval_survival.csv", index=False)
print(f"participants with paired data: {len(df)}")
print(f"AI text spans checked for survival: {len(sv)}, survival rate: {sv['survived'].mean():.3f}" if len(sv) else "no survival rows")
