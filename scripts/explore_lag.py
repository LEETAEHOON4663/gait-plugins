"""Is the 2D-3D disagreement a time lag, a joint-definition offset, or noise? Usage: explore_lag.py <nova77_full>"""
import csv, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from core import nova77
from core.common import prepare
warnings.filterwarnings("ignore")
data = Path(sys.argv[1])
BODY = ["Hips", "Neck1", "Head", "LeftArm", "RightArm", "LeftLeg", "RightLeg", "LeftShin", "RightShin", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase", "LeftHand", "RightHand"]
def one(a):
    path, f, cx, cy = a
    w = prepare(nova77.load(path)).walk
    if len(w) < 60: return None
    c = [w.index(b) for b in BODY]; X, u = w.xyz[:, c], w.xy[:, c]
    proj = np.stack([f * X[..., 0] / X[..., 2] + cx, f * X[..., 1] / X[..., 2] + cy], -1)
    h = np.median(np.linalg.norm(u[:, 2] - (u[:, 9] + u[:, 10]) / 2, axis=1))         # head to ankles in pixels
    lags = {}
    for k in range(-6, 7):
        a_, b_ = (proj[k:], u[:len(u) - k]) if k >= 0 else (proj[:k], u[-k:])
        lags[k] = float(np.median(np.linalg.norm(a_ - b_, axis=-1)))
    d = proj - u
    return lags, np.median(np.abs(d), axis=(0, 1)), np.median(d, axis=0), h, float(np.median(np.linalg.norm(d - np.median(d, axis=0), axis=-1)))
meta = [r for r in csv.DictReader(open(data / "frame_reports/video_metadata_v0_1.csv", encoding="utf-8-sig")) if (data / "frame" / r["output"]).is_file()]
for ds, f, cx, cy in (("KOA-PD-NM", 1450, 640, 360), ("Hip_OA", 677, 322, 243)):
    with ProcessPoolExecutor(6) as ex: R = [r for r in ex.map(one, [(data / "frame" / m["output"], f, cx, cy) for m in meta if m["dataset"] == ds][:200], chunksize=4) if r]
    L = {k: np.median([r[0][k] for r in R]) for k in R[0][0]}
    print(f"{ds} n={len(R)} body height {np.median([r[3] for r in R]):.0f} px | error by lag (3D shifted by k frames):", {k: round(v, 1) for k, v in L.items()}, "| best lag per clip:", np.bincount([min(r[0], key=r[0].get) + 6 for r in R], minlength=13).tolist())
    print("   median |dx|,|dy| px:", np.round(np.median([r[1] for r in R], 0), 1), "| after removing each joint's constant offset:", round(np.median([r[4] for r in R]), 1), "px")
    off = np.median([r[2] for r in R], 0); print("   constant offset per joint (dx,dy):", {b: tuple(np.round(o, 0).astype(int).tolist()) for b, o in zip(BODY, off)})
