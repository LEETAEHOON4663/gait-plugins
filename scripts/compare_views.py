"""The question "is flipping enough, or do we need 3D?", answered with people filmed twice: walking left-to-right and
right-to-left. The same person's average gait cycle should come out the same. No disease labels are used for the choice.
Usage: compare_views.py <nova77_full>"""
import csv, json, sys, warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from core import jointsets, nova77
from core.common import prepare
from core.view import Camera, build_world, look
from scripts.dataset import ROOT
warnings.filterwarnings("ignore")
data = Path(sys.argv[1])
JS = "kinect13"; NAMES = list(jointsets.JOINT_SETS[JS]); SW = [0] + [k + 1 if k % 2 else k - 1 for k in range(1, 13)]
BASE, FLOOR, FULL = dict(refine=False, use_floor=False), dict(refine=False, use_floor=True), dict(refine=True, use_floor=True)

def joints(s, xyz):
    cols = []
    for src in jointsets.JOINT_SETS[JS].values():
        parts = (src,) if isinstance(src, str) else src
        cols.append(np.mean([xyz[:, s.index(p)] for p in parts], axis=0))
    return np.stack(cols, 1)

def one(path):
    p = prepare(nova77.load(path))
    if p.axes.view != "Side" or not p.axes.left_right_ok or len(p.walk) < 45: return None
    w0, w1, w2 = (build_world(p.walk, **k) for k in (BASE, FLOOR, FULL))
    cam = joints(p.walk, p.walk.xyz) * np.array([1, -1, 1])                          # raw camera axes, y up
    hs = {}
    a = w0.xyz; fwd = lambda n: a[:, p.walk.index(n), 0] - a[:, p.walk.index("Hips"), 0]
    for s in ("Left", "Right"):
        q = savgol_filter(fwd(s + "Foot"), 9, 2); hs[s] = find_peaks(q, distance=int(.6 * p.raw.fps), prominence=.25 * (np.percentile(q, 95) - np.percentile(q, 5)))[0]
    out = {"cam": cam, "w0": joints(p.walk, w0.xyz), "w1": joints(p.walk, w1.xyz), "w2": joints(w2.body, w2.xyz), "hs": hs,
           "dir": p.axes.direction, "err": w2.info["reprojection_px"], "scale": w2.info["scale"], "slope": w2.floor_slope, "kept": w2.info["kept_original"]}
    return out

def cycle(x, hs):
    """Strides, hip-centred, 51 phases, divided by the recording's own body size so that a method cannot win by shrinking."""
    seg = lambda i, j: np.median(np.linalg.norm(x[:, i] - x[:, j], axis=1))
    x = (x - (x[:, [7]] + x[:, [8]]) / 2) * 1.25 / (seg(0, 1) + (seg(7, 9) + seg(8, 10)) / 2 + (seg(9, 11) + seg(10, 12)) / 2)
    st = []
    for a, b in zip(hs[:-1], hs[1:]):
        t = np.linspace(a, b, 51); i = np.clip(t.astype(int), 0, len(x) - 2); f = (t - i)[:, None, None]
        st.append(x[i] * (1 - f) + x[i + 1] * f)
    return st

meta = [r for r in csv.DictReader(open(data / "frame_reports/video_metadata_v0_1.csv", encoding="utf-8-sig")) if r["dataset"] == "KOA-PD-NM" and (data / "frame" / r["output"]).is_file()]
with ProcessPoolExecutor(6) as ex: R = list(ex.map(one, [data / "frame" / r["output"] for r in meta], chunksize=4))
by = defaultdict(dict)
for m, o in zip(meta, R):
    if o: by[f'{m["group"]}_{m["severity"]}_{m["subject_id"]}'][m["direction"]] = o
pairs = [(k, d["left to right"], d["right to left"]) for k, d in by.items() if len(d) == 2]
mean = lambda st: np.mean(st, 0) if st else None
dist = lambda a, b: np.linalg.norm(a - b, axis=-1)                                    # (51, 13)

def compare(fa, fb):
    """fa / fb give (joint array, heel-strike side) for the left-to-right and right-to-left recording."""
    res = []
    for k, A, B in pairs:
        (xa, sa), (xb, sb) = fa(A), fb(B)
        ca, cb = mean(cycle(xa, A["hs"][sa])), mean(cycle(xb, B["hs"][sb]))
        if ca is not None and cb is not None: res.append((dist(ca, cb), np.abs(ca - cb)))
    return np.mean([r[0] for r in res], (0, 1)) * 100, np.mean([r[1] for r in res], (0, 1)) * 100, len(res)

flip = lambda x: x * np.array([-1, 1, 1])
front = lambda x: look(x, Camera(azimuth=0, elevation=0))
methods = {
    "A  camera axes as they are": (lambda A: (A["cam"], "Left"), lambda B: (B["cam"], "Left")),
    "B  flip the second video": (lambda A: (A["cam"], "Left"), lambda B: (flip(B["cam"]), "Left")),
    "C  flip and swap left/right labels": (lambda A: (A["cam"], "Left"), lambda B: (flip(B["cam"])[:, SW], "Right")),
    "D  turn in 3D, level camera assumed": (lambda A: (front(A["w0"]), "Left"), lambda B: (front(B["w0"]), "Left")),
    "E  turn in 3D + floor": (lambda A: (front(A["w1"]), "Left"), lambda B: (front(B["w1"]), "Left")),
    "F  turn in 3D + floor + 2D correction": (lambda A: (front(A["w2"]), "Left"), lambda B: (front(B["w2"]), "Left")),
}
print(f"{len(pairs)} people filmed in both directions. Distance between their two average gait cycles, hip-centred, cm")
print(f"{'method':42s}  n   all joints  legs  arms   | per axis: left-right  up  depth")
for name, (fa, fb) in methods.items():
    d, ax, n = compare(fa, fb)
    a3 = ax.reshape(-1, 13, 3).mean((0, 1)) if ax.ndim == 3 else ax
    legs, arms = d[[9, 10, 11, 12]].mean(), d[[3, 4, 5, 6]].mean()
    # per-axis means need the raw differences again
    acc = []
    for k, A, B in pairs:
        (xa, sa), (xb, sb) = fa(A), fb(B); ca, cb = mean(cycle(xa, A["hs"][sa])), mean(cycle(xb, B["hs"][sb]))
        if ca is not None and cb is not None: acc.append(np.abs(ca - cb).mean((0, 1)))
    acc = np.mean(acc, 0) * 100
    print(f"{name:42s} {n:3d}   {d[[i for i in range(13) if i not in (7, 8)]].mean():6.1f}     {legs:5.1f}  {arms:5.1f}   |            {acc[0]:5.1f}  {acc[1]:5.1f}  {acc[2]:5.1f}")
# repeatability floor: odd vs even strides of the same recording (method F)
fl = []
for k, A, B in pairs:
    for o in (A, B):
        st = cycle(front(o["w2"]), o["hs"]["Left"])
        if len(st) >= 2: fl.append(dist(np.mean(st[::2], 0), np.mean(st[1::2], 0)))
print(f"floor: two strides of the SAME recording differ by {np.mean([f.mean() for f in fl])*100:.1f} cm (n={len(fl)} recordings)")
print("2D correction kept the original proportions in", sum(o["kept"] for o in R if o), "of", sum(1 for o in R if o), "clips")
print("2D correction: reprojection error median %.1f -> %.1f px | bone scales: " % tuple(np.median([o["err"] for o in R if o], 0)), {k: (round(np.median([o["scale"][k] for o in R if o]), 3), round(float(np.std([o["scale"][k] for o in R if o])), 3)) for k in R[0]["scale"]} if R[0] else "")
print("floor slope: LR %.1f deg, RL %.1f deg" % tuple(np.mean([o["slope"] for o in R if o and o["dir"] == d]) for d in ("LeftToRight", "RightToLeft")))
