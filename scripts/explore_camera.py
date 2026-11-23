"""How consistent is Nova-77's 3D with its own 2D, and how is the camera really oriented? Usage: explore_camera.py <nova77_full>"""
import csv, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from core import nova77
from core.common import CAMERA_UP, prepare
warnings.filterwarnings("ignore")
data = Path(sys.argv[1])
BODY = ["Hips", "Neck1", "Head", "LeftArm", "RightArm", "LeftHand", "RightHand", "LeftLeg", "RightLeg", "LeftShin", "RightShin", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"]

def one(path):
    p = prepare(nova77.load(path)); s = p.raw; ok = s.status[:, 0] == 1
    cols = [s.index(b) for b in BODY]
    X, u = s.xyz[ok][:, cols].reshape(-1, 3), s.xy[ok][:, cols].reshape(-1, 2)
    if len(X) < 200: return None
    a, b = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]                         # pinhole fit: u = f a + cx, v = f b + cy
    A = np.block([[a[:, None], np.ones((len(a), 1)), np.zeros((len(a), 1))], [b[:, None], np.zeros((len(a), 1)), np.ones((len(a), 1))]])
    sol, *_ = np.linalg.lstsq(A, np.concatenate([u[:, 0], u[:, 1]]), rcond=None)
    r = np.linalg.norm(np.stack([sol[0] * a + sol[1] - u[:, 0], sol[0] * b + sol[2] - u[:, 1]], 1), axis=1)
    o = dict(f=sol[0], cx=sol[1], cy=sol[2], err=float(np.median(r)), err95=float(np.percentile(r, 95)), view=p.axes.view, direction=p.axes.direction, frames=len(p.walk))
    w, ax = p.walk, p.axes
    if ax.view in ("Side", "Front") and len(w) >= 45 and ax.left_right_ok:
        F = lambda n: w.point(n)
        up = CAMERA_UP
        # floor points: the lower toe base in every frame, as seen along the assumed up axis
        tl, tr = F("LeftToeBase"), F("RightToeBase")
        hl, hr = tl @ up, tr @ up
        low = np.where((hl < hr)[:, None], tl, tr)
        s_, h_ = low @ ax.forward, low @ up
        k = h_ <= np.percentile(h_, 60)
        o["floor_slope"] = float(np.degrees(np.arctan(np.polyfit(s_, h_, 1)[0])))                   # along the walking direction
        d = F("LeftLeg") - F("RightLeg"); o["hip_roll"] = float(np.degrees(np.arctan2((d @ up).mean(), (d @ ax.left).mean())))
        d = F("LeftArm") - F("RightArm"); o["shoulder_roll"] = float(np.degrees(np.arctan2((d @ up).mean(), (d @ ax.left).mean())))
        b_ = F("Neck1") - (F("LeftFoot") + F("RightFoot")) / 2; o["body_lat"] = float(np.degrees(np.arctan2((b_ @ ax.left).mean(), (b_ @ up).mean())))
        o["body_fwd"] = float(np.degrees(np.arctan2((b_ @ ax.forward).mean(), (b_ @ up).mean())))
        o["hip_h"] = float(np.ptp(np.polyval(np.polyfit(np.arange(len(w)), F("Hips") @ up, 1), [0, len(w) - 1])))
    return o

meta = [r for r in csv.DictReader(open(data / "frame_reports/video_metadata_v0_1.csv", encoding="utf-8-sig")) if (data / "frame" / r["output"]).is_file()]
with ProcessPoolExecutor(6) as ex: res = list(ex.map(one, [data / "frame" / r["output"] for r in meta], chunksize=4))
for ds in ("KOA-PD-NM", "Hip_OA"):
    R = [(m, o) for m, o in zip(meta, res) if m["dataset"] == ds and o]
    q = lambda k: np.array([o[k] for _, o in R])
    print(f"\n{ds}: n={len(R)}  focal px median {np.median(q('f')):.0f} (IQR {np.percentile(q('f'), 25):.0f}-{np.percentile(q('f'), 75):.0f})  centre ({np.median(q('cx')):.0f}, {np.median(q('cy')):.0f})  reprojection error median {np.median(q('err')):.2f} px, 95% {np.median(q('err95')):.2f} px")
    for d in sorted({o["direction"] for _, o in R}):
        S = [o for _, o in R if o["direction"] == d and "hip_roll" in o]
        if len(S) < 10: continue
        g = lambda k: f"{np.mean([o[k] for o in S]):+5.1f} (sd {np.std([o[k] for o in S]):.1f})"
        print(f"  {d:15s} n={len(S):3d}  floor slope {g('floor_slope')}  hip roll {g('hip_roll')}  shoulder roll {g('shoulder_roll')}  body sideways {g('body_lat')}  body forward {g('body_fwd')}")
