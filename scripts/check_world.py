"""Effect of the 3D layer on the ST-GCN plugin. Label-free measures first. Usage: check_world.py <nova77_full>"""
import csv, sys, warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from core import jointsets
from core.plugin import discover
from core.view import Camera, build_world, look
from scripts.dataset import _load, koa_pd_nm
warnings.filterwarnings("ignore")
data = Path(sys.argv[1])
pl = next(p for p in discover()[0] if p.manifest.id == "pd_stgcn_sabo2022"); P = sys.modules[type(pl).__module__]
N = list(jointsets.JOINT_SETS["kinect13"])
net = lambda x: float(pl.model(torch.from_numpy(x[None].astype(np.float32).copy())).item())
vids = [v for v in koa_pd_nm(data) if not pl.check(v["prepared"])]
meta = [r for r in csv.DictReader(open(data / "frame_reports/video_metadata_v0_1.csv", encoding="utf-8-sig")) if r["dataset"] == "Hip_OA" and (data / "frame" / r["output"]).is_file()]
with ProcessPoolExecutor(6) as ex: prep = list(ex.map(_load, [data / "frame" / r["output"] for r in meta], chunksize=8))
hip = [(r["subject_id"], r["session"], p) for r, p in zip(meta, prep) if not pl.check(p) and p.axes.direction == "AwayFromCamera"]

def raw_flip(p, swap):                      # no 3D turn: camera axes, mirrored so everyone walks left-to-right
    n, c = jointsets.select(p.walk, "kinect13"); c = c * np.array([1, -1, 1])
    if p.axes.direction == "RightToLeft":
        c = c * np.array([-1, 1, 1])
        if swap: c = c[:, [0] + [k + 1 if k % 2 else k - 1 for k in range(1, 13)]]
    return c
W = {}
def world(p, **kw):
    key = (id(p), tuple(kw.items()))
    if key not in W: W[key] = build_world(p.walk, **kw)
    return W[key]
def turned(p, el=30.0, **kw):
    w = world(p, **kw); n, x = jointsets.select(w.body, "kinect13", w.xyz); return look(x, Camera(azimuth=0, elevation=el))
def icc(groups):
    g = [np.array(v) for v in groups if len(v) >= 2]; k = np.mean([len(v) for v in g]); n = len(g); a = np.concatenate(g)
    msb = sum(len(v) * (v.mean() - a.mean()) ** 2 for v in g) / (n - 1); msw = sum(((v - v.mean()) ** 2).sum() for v in g) / (len(a) - n)
    return (msb - msw) / (msb + (k - 1) * msw), np.mean([np.mean([abs(x - y) for i, x in enumerate(v) for y in v[i + 1:]]) for v in g])
print(f"{'input to the model':44s} side LR vs RL: ICC |diff| bias | front repeat: ICC |diff| | hip OA M0 M6 (n, p) | AUC PD-NM PD-KOA")
for name, fn, isfront in (("flip only", lambda p: raw_flip(p, False), False), ("flip and swap sides", lambda p: raw_flip(p, True), False),
                          ("3D turn, level camera (before)", lambda p: turned(p, use_floor=False), True), ("3D turn + floor (now)", lambda p: turned(p), True),
                          ):
    sc = {v["file"]: net(P.study_preprocess(fn(v["prepared"]), N)) for v in vids}
    by, dirs = defaultdict(list), defaultdict(dict)
    for v in vids: by[v["subject"]].append(sc[v["file"]]); dirs[v["subject"]][v["direction"]] = sc[v["file"]]
    i1, d1 = icc(by.values()); bias = np.mean([d["left to right"] - d["right to left"] for d in dirs.values() if len(d) == 2])
    sub = {k: np.mean(v) for k, v in by.items()}; g = np.array([k.split("_")[0] for k in sub]); x = np.array(list(sub.values()))
    auc = lambda a, b: roc_auc_score(g[(g == a) | (g == b)] == a, x[(g == a) | (g == b)])
    s2 = ""
    if isfront:
        hb, hs = defaultdict(list), defaultdict(lambda: defaultdict(list))
        for a, b, p in hip:
            z = net(P.study_preprocess(fn(p), N)); hb[a, b].append(z); hs[a][b].append(z)
        i2, d2 = icc(hb.values()); pr = np.array([(np.mean(v["M0"]), np.mean(v["M6"])) for v in hs.values() if v["M0"] and v["M6"]])
        s2 = f"{i2:.2f} {d2:.2f}             | {pr[:, 0].mean():.2f} {pr[:, 1].mean():.2f} (n={len(pr)}, p={wilcoxon(pr[:, 0], pr[:, 1]).pvalue:.0e})"
    print(f"{name:44s}                {i1:.2f}  {d1:.2f} {bias:+.2f} | {s2:58s} | {auc('PD', 'NM'):.2f}  {auc('PD', 'KOA'):.2f}")
# camera is fixed within a session: how constant is the estimated tilt, and does one tilt per session help?
tilt = defaultdict(list)
for a, b, p in hip: tilt[a, b].append(world(p).floor_slope)
print(f"front tilt estimate: mean {np.mean(sum(tilt.values(), [])):.1f} deg, sd between clips of one session {np.mean([np.std(v) for v in tilt.values() if len(v) > 1]):.1f} deg, sd between sessions {np.std([np.median(v) for v in tilt.values()]):.1f} deg; depth scale median {np.median([world(p).info['depth_scale'] for _, _, p in hip]):.2f}")
hb, hs = defaultdict(list), defaultdict(lambda: defaultdict(list))
for a, b, p in hip:
    z = net(P.study_preprocess(turned(p, tilt_deg=float(np.median(tilt[a, b]))), N)); hb[a, b].append(z); hs[a][b].append(z)
i2, d2 = icc(hb.values()); pr = np.array([(np.mean(v["M0"]), np.mean(v["M6"])) for v in hs.values() if v["M0"] and v["M6"]])
print(f"one tilt per session: front repeat ICC {i2:.2f} |diff| {d2:.2f} | hip OA M0 {pr[:, 0].mean():.2f} M6 {pr[:, 1].mean():.2f} (n={len(pr)}, p={wilcoxon(pr[:, 0], pr[:, 1]).pvalue:.0e}, lower in {int((pr[:, 1] < pr[:, 0]).sum())})")
cl = lambda x: np.clip(np.round(x), 0, 2)
for el in (20, 25, 35, 40):
    a = np.array([net(P.study_preprocess(turned(v["prepared"], 30.0), N)) for v in vids]); b = np.array([net(P.study_preprocess(turned(v["prepared"], float(el)), N)) for v in vids])
    print(f"tilt {el} instead of 30 deg: mean |change| {np.abs(a - b).mean():.2f}, class changes {np.mean(cl(a) != cl(b)):.0%}")
