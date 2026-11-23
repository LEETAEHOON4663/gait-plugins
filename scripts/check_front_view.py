"""How does the ST-GCN plugin behave on front-view hip OA videos (no parkinsonism expected)? Usage: check_front_view.py <nova77_full>"""
import csv, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from scipy.stats import wilcoxon
from core.plugin import discover, run_one
from scripts.dataset import _load

data = Path(sys.argv[1])
meta = [r for r in csv.DictReader(open(data / "frame_reports/video_metadata_v0_1.csv", encoding="utf-8-sig")) if r["dataset"] == "Hip_OA" and (data / "frame" / r["output"]).is_file()]
pl = next(p for p in discover()[0] if p.manifest.id == "pd_stgcn_sabo2022")
with ProcessPoolExecutor(6) as ex:
    prep = list(ex.map(_load, [data / "frame" / r["output"] for r in meta], chunksize=8))
rows = []
for r, p in zip(meta, prep):
    res = run_one(pl, p)
    if res.status == "ok":
        rows.append((r["subject_id"], r["session"], p.axes.direction, res.score, p.axes.distance_m / len(p.walk) * p.raw.fps, p.axes.left_right_ok))
print("rejected for left/right contradiction, by direction:", dict(Counter(p.axes.direction for p in prep if p.axes.view != "Unknown" and not p.axes.left_right_ok)))
print("model-estimated path length, median m:", round(float(np.median([p.axes.distance_m for p in prep if np.isfinite(p.axes.distance_m)])), 2))
print("videos", len(meta), "scored", len(rows), "| directions", dict(Counter(r[2] for r in rows)), "| left/right inconsistent", sum(not r[5] for r in rows))
for d in sorted({r[2] for r in rows}):
    s = np.array([r[3] for r in rows if r[2] == d]); v = np.array([r[4] for r in rows if r[2] == d])
    print(f"{d:15s} n={len(s):3d} score mean {s.mean():.2f} sd {s.std():.2f} | speed (model m/s) {v.mean():.2f} | class {dict(sorted(Counter(np.clip(np.round(s), 0, 2).astype(int).tolist()).items()))}")
by = defaultdict(lambda: defaultdict(list))
for r in rows: by[r[0]][r[1]].append(r[3])
pairs = [(np.mean(v["M0"]), np.mean(v["M6"])) for v in by.values() if v["M0"] and v["M6"]]
a = np.array(pairs)
print(f"before surgery M0 {a[:,0].mean():.2f} vs 6 months after M6 {a[:,1].mean():.2f}, n={len(a)} patients, Wilcoxon p={wilcoxon(a[:,0], a[:,1]).pvalue:.1e}; lower after surgery in {int((a[:,1] < a[:,0]).sum())}")
