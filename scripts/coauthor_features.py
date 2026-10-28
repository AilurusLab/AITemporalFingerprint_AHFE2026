"""Extract temporal-signature features from CoAuthor sessions.

Maintains a char-level provenance buffer ('H' human / 'A' AI) per session by
replaying Quill deltas, then computes per-session features:
  - human IKI distribution (within user text-insert stream)
  - human P-burst sizes (pause threshold segmentation, 2 s default)
  - AI insertion sizes (chars per api text-insert)
  - post-acceptance latency (api insert -> next user text event)
  - afterlife: survival of H vs A chars in the final document, and
    deletion events that strike A spans vs H spans
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAUSE_THRESHOLD_MS = 2000

DATA_DIR = Path("data/coauthor/coauthor-v1.0")
OUT = Path("results/coauthor_sessions.csv")


def process_session(path):
    events = [json.loads(l) for l in open(path)]
    buf = []  # provenance per char
    ins_H = ins_A = del_H = del_A = 0

    user_insert_ts = []          # timestamps of user text-inserts
    user_insert_sizes = []       # chars per user insert event
    ai_insert_sizes = []         # chars per api insert event
    ai_accept_ts = []            # api insert timestamps
    post_ai_latency = []         # api insert -> next user text event

    # initialize with prompt text (origin 'P' = prompt, excluded from analysis)
    init = next((e for e in events if e["eventName"] == "system-initialize"), None)
    if init and init.get("currentDoc"):
        buf = ["P"] * len(init["currentDoc"])

    pending_ai_ts = None
    for e in events:
        name, src, ts = e["eventName"], e.get("eventSource"), e["eventTimestamp"]
        delta = e.get("textDelta")
        if name not in ("text-insert", "text-delete") or not isinstance(delta, dict):
            continue
        ops = delta.get("ops", [])
        origin = "A" if src == "api" else "H"

        # walk ops with provenance accounting
        pos = 0
        ev_ins = ev_del = 0
        for op in ops:
            if "retain" in op:
                pos += op["retain"]
            elif "insert" in op:
                text = op["insert"]
                n = len(text) if isinstance(text, str) else 1
                buf[pos:pos] = [origin] * n
                pos += n
                ev_ins += n
            elif "delete" in op:
                n = op["delete"]
                seg = buf[pos : pos + n]
                del_A += sum(1 for c in seg if c == "A")
                del_H += sum(1 for c in seg if c == "H")
                del buf[pos : pos + n]
                ev_del += n

        if origin == "A" and ev_ins > 0:
            ins_A += ev_ins
            ai_insert_sizes.append(ev_ins)
            ai_accept_ts.append(ts)
            pending_ai_ts = ts
        elif origin == "H":
            if ev_ins > 0:
                ins_H += ev_ins
                user_insert_ts.append(ts)
                user_insert_sizes.append(ev_ins)
            if pending_ai_ts is not None and (ev_ins > 0 or ev_del > 0):
                post_ai_latency.append(ts - pending_ai_ts)
                pending_ai_ts = None

    if ins_H < 50:  # skip degenerate sessions
        return None

    # human IKIs and P-bursts
    ts_arr = np.array(user_insert_ts, dtype=float)
    sizes = np.array(user_insert_sizes, dtype=float)
    ikis = np.diff(ts_arr)
    ikis_typing = ikis[(ikis > 0) & (ikis < PAUSE_THRESHOLD_MS)]

    bursts = []
    cur = sizes[0] if len(sizes) else 0
    for gap, size in zip(ikis, sizes[1:]):
        if gap >= PAUSE_THRESHOLD_MS:
            bursts.append(cur)
            cur = 0
        cur += size
    bursts.append(cur)
    bursts = np.array(bursts, dtype=float)

    surv_H = sum(1 for c in buf if c == "H")
    surv_A = sum(1 for c in buf if c == "A")

    return {
        "session": path.stem,
        "n_events": len(events),
        "ins_H": ins_H,
        "ins_A": ins_A,
        "n_ai_inserts": len(ai_insert_sizes),
        "ai_insert_size_mean": float(np.mean(ai_insert_sizes)) if ai_insert_sizes else np.nan,
        "human_burst_mean": float(np.mean(bursts)),
        "human_burst_median": float(np.median(bursts)),
        "human_burst_p95": float(np.percentile(bursts, 95)),
        "human_burst_max": float(np.max(bursts)),
        "iki_median_ms": float(np.median(ikis_typing)) if len(ikis_typing) else np.nan,
        "post_ai_latency_median_ms": float(np.median(post_ai_latency)) if post_ai_latency else np.nan,
        "del_rate_H": del_H / ins_H if ins_H else np.nan,
        "del_rate_A": del_A / ins_A if ins_A else np.nan,
        "survival_H": surv_H / ins_H if ins_H else np.nan,
        "survival_A": surv_A / ins_A if ins_A else np.nan,
        # per-session z-score of AI insert size vs the author's own burst distribution
        "ai_size_z_vs_own_bursts": (
            float((np.mean(ai_insert_sizes) - np.mean(bursts)) / np.std(bursts))
            if ai_insert_sizes and np.std(bursts) > 0
            else np.nan
        ),
    }


def main():
    rows = []
    files = sorted(DATA_DIR.glob("*.jsonl"))
    for i, f in enumerate(files):
        try:
            r = process_session(f)
            if r:
                rows.append(r)
        except Exception as ex:  # noqa: BLE001
            print(f"skip {f.name}: {ex}", file=sys.stderr)
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(files)}")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} sessions -> {OUT}")


if __name__ == "__main__":
    main()
