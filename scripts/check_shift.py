"""Distribution shift with the zero padding removed. In the study's input every real frame has its hip centre at exactly -0.5
and every padded frame is 0, so the padded share q follows from the stored mean of the two hips, and the mean and spread
of the real frames can be recovered. Usage: check_shift.py <nova77_full> <stgcn repo>"""
import csv, glob, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from core import jointsets
from core.common import in_walk_frame
from core.view import Camera, look
from core.plugin import discover
from scripts.dataset import koa_pd_nm
warnings.filterwarnings("ignore")
data, repo = Path(sys.argv[1]), Path(sys.argv[2])
pl = next(p for p in discover()[0] if p.manifest.id == "pd_stgcn_sabo2022"); P = sys.modules[type(pl).__module__]
N = list(jointsets.JOINT_SETS["kinect13"]); L, R = N.index("LHip"), N.index("RHip")
bn = pl.model.encoder.data_bn
mu, var = bn.running_mean.numpy().reshape(13, 3).astype(float), bn.running_var.numpy().reshape(13, 3).astype(float)
q = 1 + (mu[L] + mu[R]).mean() / 2 / 0.5
print("padded share of the study's training frames, per axis:", np.round(1 + (mu[L] + mu[R]) / 2 / 0.5, 3), "-> q =", round(q, 3))
m_real = mu / (1 - q); sd_real = np.sqrt(np.maximum((var + mu ** 2) / (1 - q) - m_real ** 2, 1e-6))

def frames(k):                                  # hip-centred metres, no padding, no crop
    h = (k[:, L] + k[:, R]) / 2
    return k - h[:, None] - 0.5
sample = [frames(np.array([[[float(r[n + a]) for a in ("_x", "_y", "_z")] for n in N] for r in csv.DictReader(open(f))]))
          for f in sorted(glob.glob(str(repo / "sample_data/data/kinect_skel_data_3d/kinect_all/*.csv")))]
vids = [v for v in koa_pd_nm(data) if not pl.check(v["prepared"])]
def ours(v, tilt=P.TILT_DEG):
    w = v["prepared"].world; n, c = jointsets.select(w.body, "kinect13", w.xyz)
    return frames(look(c, Camera(azimuth=0, elevation=tilt)))
O = {v["file"]: ours(v) for v in vids}
# The study trained on every walk plus its flip, and the flip only swaps the left-right coordinate between sides. So the
# left-right axis must be compared after the same flip is added to our frames; up and depth are unaffected.
def body(a):
    d = lambda i, j: float(np.median(np.linalg.norm(a[:, N.index(i)] - a[:, N.index(j)], axis=1)))
    return {"shoulder width": d("LShoulder", "RShoulder"), "hip width": d("LHip", "RHip"), "thigh": d("LHip", "LKnee"), "shank": d("LKnee", "LAnkle"), "upper arm": d("LShoulder", "LElbow")}
SW = [0] + [k + 1 if k % 2 else k - 1 for k in range(1, 13)]
def with_flip(a):
    b = a.copy(); b[..., 0] = a[:, SW, 0]
    return np.concatenate([a, b])
def rep(name, X):
    a = with_flip(np.concatenate(X)); s = (a.mean(0) - m_real) / sd_real; r = a.std(0) / sd_real
    print(f"{name:38s} |shift| median {np.median(np.abs(s)):.2f} SD; by axis lr {np.median(np.abs(s[:, 0])):.2f} up {np.median(np.abs(s[:, 1])):.2f} depth {np.median(np.abs(s[:, 2])):.2f} | spread ratio lr {np.median(r[:, 0]):.2f} up {np.median(r[:, 1]):.2f} depth {np.median(r[:, 2]):.2f}")
    return s
rep("study's own 3 sample walks (1 person)", sample)
for g in ("NM", "PD", "KOA"): rep(f"ours {g}", [O[v["file"]] for v in vids if v["group"] == g])
rep("ours all", list(O.values()))
A = np.concatenate(list(O.values()))
print("\nmean position relative to the hip centre, metres (study training / ours)")
for ax, c in (("up", 1), ("depth", 2)):
    print(f"  {ax:5s}", "  ".join(f"{n} {m_real[k, c] + .5:+.2f}/{A[:, k, c].mean() + .5:+.2f}" for k, n in enumerate(N) if n in ("Head", "LShoulder", "RShoulder", "LWrist", "LHip", "RHip", "LKnee", "LAnkle", "RAnkle")))
print(f"  hip-height spread in the study (SD): {sd_real[L, 1] * 1000:.1f} mm, ours {A[:, L, 1].std() * 1000:.1f} mm")
print("\npelvis direction against the walking direction (0 deg = the hip line is square to the path)")
def yaw(v):
    p = v["prepared"]; w = in_walk_frame(p.walk.xyz, p.axes); d = (w[:, p.walk.index("LeftLeg")] - w[:, p.walk.index("RightLeg")]).mean(0)
    return np.degrees(np.arctan2(d[0], d[1])), np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1])))
for g in ("NM", "PD", "KOA"):
    for d in ("left to right", "right to left"):
        y = np.array([yaw(v) for v in vids if v["group"] == g and v["direction"] == d])
        print(f"  {g:4s} {d:14s} yaw {y[:, 0].mean():+6.1f} deg (sd {y[:, 0].std():.1f})  roll {y[:, 1].mean():+5.1f} deg (sd {y[:, 1].std():.1f})  n={len(y)}")
print("\nsegment lengths (m) study samples / ours:", {k: (round(v, 2), round(body(A)[k], 2)) for k, v in body(np.concatenate(sample)).items()})
