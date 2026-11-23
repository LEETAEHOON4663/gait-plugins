"""Front-view (Hip_OA) part of evaluate_new.py, one recording at a time. Usage: evaluate_hip_front.py <nova77_full> <out>"""
import csv, json, sys, warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from core.plugin import discover, run_one
from scripts.dataset import _load, frame_sizes
warnings.filterwarnings("ignore"); torch.set_num_threads(1)
data, out = Path(sys.argv[1]), Path(sys.argv[2])
IDS = ["pd_stgcn_sabo2022", "pd_stgcn_sabo2022_ap2d", "pd_stgcn_sabo2022_dt2d"]
BY = None


def score(args):
    global BY
    BY = BY or {p.manifest.id: p for p in discover()[0]}
    path, size = args
    p, q = _load(path, size), _load(path)                          # with and without the image size
    if p.axes.direction != "AwayFromCamera": return p.axes.direction, None, len(q.walk), len(p.walk)
    return p.axes.direction, [(lambda x: x.score if x.status == "ok" else float("nan"))(run_one(BY[i], p)) for i in IDS], len(q.walk), len(p.walk)


if __name__ == "__main__":
    meta = [m for m in csv.DictReader(open(data / "frame_reports/video_metadata_v0_1.csv", encoding="utf-8-sig")) if m["dataset"] == "Hip_OA" and (data / "frame" / m["output"]).is_file()]
    rows, sizes = [], frame_sizes(data)
    with ProcessPoolExecutor(2) as ex:
        for n, (m, (d, sc, n0, n1)) in enumerate(zip(meta, ex.map(score, [(data / "frame" / m["output"], sizes.get(m["output"])) for m in meta], chunksize=2))):
            rows.append({"subject": m["subject_id"], "session": m["session"], "direction": d, "scores": sc, "walk_before": n0, "walk_visible": n1})
            if n % 100 == 0: print(f"{n}/{len(meta)}", flush=True)
    (out / "hip_front.json").write_text(json.dumps(rows))
    w0, w1 = np.array([r["walk_before"] for r in rows]), np.array([r["walk_visible"] for r in rows])
    print(f"walking frames: {w0.sum()} -> {w1.sum()} after keeping only frames with the whole body visible; recordings with less than 60 frames left: {np.sum(w1 < 60)} of {len(rows)}", flush=True)
    for j, i in enumerate(IDS):
        h = defaultdict(list)
        for r in rows:
            if r["scores"] is not None and np.isfinite(r["scores"][j]): h[r["subject"], r["session"]].append(r["scores"][j])
        v = [x for x in h.values() if len(x) >= 2]; d = np.mean([np.mean([abs(a - b) for q, a in enumerate(x) for b in x[q + 1:]]) for x in v])
        vv = [(np.mean(h[s, "M0"]), np.mean(h[s, "M6"])) for s in {k[0] for k in h} if h.get((s, "M0")) and h.get((s, "M6"))]
        print(f"{i:26s} front (away): repeat walks in one session differ by {d:.2f} points (n={len(v)} sessions); before/after surgery {np.mean([x[0] for x in vv]):.2f} -> {np.mean([x[1] for x in vv]):.2f} (n={len(vv)}), lower in {sum(b < a for a, b in vv)}", flush=True)
