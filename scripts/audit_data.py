"""Audit every Nova-77 JSONL file: schema, coverage, gaps, view. Writes audit.csv (one row per video)."""
import csv, json, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

KEYS = {"tracking_id", "bbox", "keypoints_2d", "keypoints_confidence", "keypoints_3d", "rest_pose", "joint_rotations", "root_pose"}


def one(args):
    path, root = args
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    has = np.array([bool(r["detections"]) for r in rows])
    multi = max((len(r["detections"]) for r in rows), default=0)
    ids = {d["tracking_id"] for r in rows for d in r["detections"]}
    miss, bad = set(), 0
    hips, conf = [], []
    for r in rows:
        for d in r["detections"][:1]:
            miss |= KEYS - set(d)
            k3 = np.asarray(d["keypoints_3d"], float)
            if k3.shape != (77, 3) or not np.isfinite(k3).all():
                bad += 1
            hips.append(k3[0])
            conf.append(np.mean(d["keypoints_confidence"]))
    t = np.array([r["time_ms"] for r in rows], float)
    idx = np.flatnonzero(has)
    gaps = np.diff(idx) - 1
    gaps = gaps[gaps > 0]

    def longest(maxgap):
        if not len(idx):
            return 0
        best, s, p = 1, idx[0], idx[0]
        for i in idx[1:]:
            if i - p - 1 > maxgap:
                s = i
            best = max(best, i - s + 1)
            p = i
        return int(best)
    if len(hips) >= 10:
        h = np.array(hips)
        dx, dz = float(h[-1, 0] - h[0, 0]), float(h[-1, 2] - h[0, 2])
        view = "Side" if abs(dx) >= 2 * abs(dz) else "Front" if abs(dz) >= 2 * abs(dx) else "Oblique"
        direction = {"Side": "LR" if dx > 0 else "RL", "Front": "Away" if dz > 0 else "Toward"}.get(view, "Unknown")
    else:
        dx = dz = float("nan"); view = direction = "Unknown"
    return dict(file=os.path.relpath(path, root), frames=len(rows), detected=int(has.sum()), max_people=multi,
                track_ids=len(ids), missing_fields="|".join(sorted(miss)), bad_frames=bad,
                fps=round(1000 / np.median(np.diff(t)), 3) if len(t) > 1 else float("nan"),
                n_gaps=len(gaps), gaps_le5=int((gaps <= 5).sum()), max_gap=int(gaps.max()) if len(gaps) else 0,
                run0=longest(0), run5=longest(5), conf=round(float(np.mean(conf)), 3) if conf else float("nan"),
                dx=round(dx, 2), dz=round(dz, 2), view=view, direction=direction)


if __name__ == "__main__":
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    files = sorted(str(p) for p in (root / "frame").rglob("*.jsonl"))
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(one, [(f, str(root / "frame")) for f in files], chunksize=4))
    meta = {r["output"]: r for r in csv.DictReader(open(root / "frame_reports/video_metadata_v0_1.csv", encoding="utf-8-sig"))}
    for l in open(root / "frame_reports/batch_status_v0_1.jsonl", encoding="utf-8"):
        s = json.loads(l)
        if s.get("output") in meta:
            meta[s["output"]].update(width=s.get("width"), height=s.get("height"))
    for r in rows:
        m = meta.get(r["file"], {})
        for k in ("dataset", "subject_id", "group", "severity", "direction", "session", "width", "height"):
            r["label_" + k] = m.get(k, "")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("files", len(rows), "| without metadata", sum(not r["label_dataset"] for r in rows))
    print("missing fields:", dict(Counter(r["missing_fields"] for r in rows)), "| bad frames:", sum(r["bad_frames"] for r in rows),
          "| max people per frame:", dict(Counter(r["max_people"] for r in rows)), "| >1 track id:", sum(r["track_ids"] > 1 for r in rows),
          "| fps:", dict(Counter(round(r["fps"], 1) for r in rows)))
    for ds in ("KOA-PD-NM", "Hip_OA"):
        for g in sorted({r["label_group"] for r in rows if r["label_dataset"] == ds}):
            s = [r for r in rows if r["label_dataset"] == ds and r["label_group"] == g]
            subj = {(r["label_group"], r["label_severity"], r["label_subject_id"]) for r in s}
            print(f"{ds:10s} {g:4s} videos {len(s):3d} subjects {len(subj):3d} severity {dict(Counter(r['label_severity'] for r in s))} "
                  f"session {dict(Counter(r['label_session'] for r in s))} view {dict(Counter(r['view'] for r in s))} "
                  f"| run0 median {int(np.median([r['run0'] for r in s]))} run5 median {int(np.median([r['run5'] for r in s]))} "
                  f"| run5>=120: {sum(r['run5'] >= 120 for r in s)}/{len(s)} | detected median {np.median([r['detected']/max(r['frames'],1) for r in s]):.2f}"
                  f" | size {dict(Counter((r['label_width'], r['label_height']) for r in s).most_common(2))}")
    k = [r for r in rows if r["label_dataset"] == "KOA-PD-NM" and r["view"] == "Side"]
    ok = sum((r["direction"] == "LR") == (r["label_direction"] == "left to right") for r in k)
    print(f"KOA-PD-NM side direction agrees with label: {ok}/{len(k)}")
