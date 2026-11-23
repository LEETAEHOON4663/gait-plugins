"""Is the score repeatable, and how sensitive is it to choices the study does not determine? (Input shift: check_shift.py.)
Usage: check_fidelity.py <nova77_full> <stgcn repo>"""
import csv, glob, sys, warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from core import jointsets
from core.view import Camera, look
from core.plugin import discover, run_one
from scripts.dataset import _load, koa_pd_nm
warnings.filterwarnings("ignore")

data, repo = Path(sys.argv[1]), Path(sys.argv[2])
pl = next(p for p in discover()[0] if p.manifest.id == "pd_stgcn_sabo2022"); P = sys.modules[type(pl).__module__]
NAMES = list(jointsets.JOINT_SETS["kinect13"])
net = lambda x: float(pl.model(torch.from_numpy(x[None].astype(np.float32).copy())).item())
bn = pl.model.encoder.data_bn
mu, sd = bn.running_mean.numpy().reshape(13, 3), np.sqrt(bn.running_var.numpy()).reshape(13, 3)   # V*C order

vids = [v for v in koa_pd_nm(data) if not pl.check(v["prepared"])]
ours = [pl.adapt(v["prepared"]) for v in vids]
# ---- 2. repeatability: same person, different recordings
def icc(groups):
    """ICC(1,1) and mean absolute difference between recordings of the same person."""
    g = [np.array(v) for v in groups if len(v) >= 2]
    k = np.mean([len(v) for v in g]); n = len(g); allv = np.concatenate(g); gm = allv.mean()
    msb = sum(len(v) * (v.mean() - gm) ** 2 for v in g) / (n - 1)
    msw = sum(((v - v.mean()) ** 2).sum() for v in g) / (len(allv) - n)
    mad = np.mean([np.mean([abs(a - b) for i, a in enumerate(v) for b in v[i + 1:]]) for v in g])
    return (msb - msw) / (msb + (k - 1) * msw), mad, n
by, grp = defaultdict(list), {}
for v, x in zip(vids, ours):
    by[v["subject"]].append(net(x)); grp[v["subject"]] = v["group"]
print("2. repeatability")
for g in ("NM", "PD", "KOA", "all"):
    i, d, n = icc([v for s, v in by.items() if g == "all" or grp[s] == g])
    print(f"   KOA-PD-NM {g:4s} walking left-to-right vs right-to-left (camera sees the other side): ICC {i:.2f}, mean |difference| {d:.2f} points, n={n}")
meta = [r for r in csv.DictReader(open(data / "frame_reports/video_metadata_v0_1.csv", encoding="utf-8-sig")) if r["dataset"] == "Hip_OA" and (data / "frame" / r["output"]).is_file()]
with ProcessPoolExecutor(6) as ex: prep = list(ex.map(_load, [data / "frame" / r["output"] for r in meta], chunksize=8))
hb = defaultdict(list)
for m_, p in zip(meta, prep):
    res = run_one(pl, p)
    if res.status == "ok" and p.axes.direction == "AwayFromCamera": hb[m_["subject_id"], m_["session"]].append(res.score)
i, d, n = icc(list(hb.values()))
print(f"   Hip_OA front view, repeated walks in one session: ICC {i:.2f}, mean |difference| {d:.2f} points, n={n} sessions")

# ---- 3. sensitivity: how much the score moves for choices the study does not determine
base = np.array([net(x) for x in ours])
def vary(fn):
    return np.array([net(fn(v["prepared"])) for v in vids])
def build(p, head=None, tilt=P.TILT_DEG, noise=0.0, scale=1.0, hipw=1.0):
    w = p.world; n, c = jointsets.select(w.body, "kinect13", w.xyz); c = c.copy()
    if head: c[:, 0] = w.xyz[:, w.body.index(head)]
    if hipw != 1.0:
        mid = (c[:, 7] + c[:, 8]) / 2
        for j in (7, 8): c[:, j] = mid + (c[:, j] - mid) * hipw
    if noise: c = c + np.random.default_rng(0).normal(0, noise, c.shape)
    return P.study_preprocess(look(c, Camera(azimuth=0, elevation=tilt)) * scale, n)
print("3. mean |score change| when one unavoidable choice changes (score range 0-2)")
for name, kw in (("head joint: skull base instead of head centre", dict(head="Head")), ("head joint: top of head", dict(head="HeadEnd")), ("tilt 25 instead of 30 deg", dict(tilt=25)),
                 ("tilt 35 instead of 30 deg", dict(tilt=35)), ("body 10% larger", dict(scale=1.1)), ("body 10% smaller", dict(scale=0.9)), ("hips 30% narrower", dict(hipw=0.7)),
                 ("joint noise 1 cm", dict(noise=0.01)), ("joint noise 3 cm", dict(noise=0.03))):
    s = vary(lambda p: build(p, **kw)); d = s - base
    cls = np.mean(np.clip(np.round(s), 0, 2) != np.clip(np.round(base), 0, 2))
    print(f"   {name:48s} mean change {d.mean():+.2f}, mean |change| {np.abs(d).mean():.2f}, class changes in {cls:.0%} of clips, rank correlation {np.corrcoef(np.argsort(np.argsort(s)), np.argsort(np.argsort(base)))[0, 1]:.2f}")
