"""Is the walked distance right? A foot on the ground must not slide: relative to the pelvis it moves backwards exactly as fast as
the pelvis moves forwards. k = (that speed) / (speed of the pelvis in the 3D estimate). k = 1 means a consistent depth scale.
Also: did the camera tilt differ between the two hip-OA sessions? Usage: explore_depth_scale.py <nova77_full>"""
import csv, sys, warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from scipy.signal import savgol_filter
from core import nova77
from core.common import CAMERA_UP, prepare
from core.view import floor_slope_deg
warnings.filterwarnings("ignore")
data = Path(sys.argv[1])

def skate_scale(w, ax):
    """Stance = the foot that is lower and moving backwards relative to the pelvis."""
    fps = w.fps; hips = w.point("Hips") @ ax.forward
    v_root = np.gradient(savgol_filter(hips, 15, 2)) * fps
    v = []
    for s in ("Left", "Right"):
        o = "Right" if s == "Left" else "Left"
        rel = savgol_filter(w.point(s + "Foot") @ ax.forward - hips, 9, 2); vr = np.gradient(rel) * fps
        lower = (w.point(s + "ToeBase") @ ax.up) < (w.point(o + "ToeBase") @ ax.up)
        st = lower & (vr < 0) & (np.abs(rel) < 0.6 * np.ptp(rel))
        v += (-vr[st]).tolist()
    return (np.median(v), float(np.median(v_root)), len(v)) if len(v) >= 20 else (np.nan, np.nan, 0)

def one(path):
    p = prepare(nova77.load(path)); w, ax = p.walk, p.axes
    if ax.view not in ("Side", "Front") or len(w) < 60 or not ax.left_right_ok: return None
    vb, vr, n = skate_scale(w, ax)
    px = lambda a, b: np.linalg.norm(w.xy[a:b, w.index("Head")] - (w.xy[a:b, w.index("LeftFoot")] + w.xy[a:b, w.index("RightFoot")]) / 2, axis=1).mean()
    k = max(5, len(w) // 8); z = w.point("Hips")[:, 2]
    return dict(view=ax.view, dir=ax.direction, k=vb / vr if vr and vr > 0.02 else np.nan, v_body=vb, v_root=vr, slope=floor_slope_deg(w, ax),
                z0=float(z[:k].mean()), z1=float(z[-k:].mean()), size_ratio=float(px(0, k) / px(-k - 1, -1)), z_ratio=float(z[-k:].mean() / z[:k].mean()))
meta = [r for r in csv.DictReader(open(data / "frame_reports/video_metadata_v0_1.csv", encoding="utf-8-sig")) if (data / "frame" / r["output"]).is_file()]
with ProcessPoolExecutor(6) as ex: R = list(ex.map(one, [data / "frame" / r["output"] for r in meta], chunksize=4))
for ds, vw in (("KOA-PD-NM", "Side"), ("Hip_OA", "Front")):
    S = [(m, o) for m, o in zip(meta, R) if o and m["dataset"] == ds and o["view"] == vw]
    q = lambda c: np.array([o[c] for _, o in S], float)
    k = q("k"); k = k[np.isfinite(k)]
    print(f"{ds} {vw}: n={len(S)}  k median {np.median(k):.2f} (IQR {np.percentile(k,25):.2f}-{np.percentile(k,75):.2f}) | speed from the stance foot {np.nanmedian(q('v_body')):.2f} m/s, from the pelvis path {np.nanmedian(q('v_root')):.2f} m/s")
    if ds == "KOA-PD-NM":
        for g in ("NM", "PD", "KOA"):
            kk = np.array([o["k"] for m, o in S if m["group"] == g]); print(f"   {g:4s} k median {np.nanmedian(kk):.2f} (IQR {np.nanpercentile(kk,25):.2f}-{np.nanpercentile(kk,75):.2f}) n={len(kk)}")
    else:
        print(f"   distance to camera in the 3D estimate: {np.median(q('z0')):.1f} -> {np.median(q('z1')):.1f} m (ratio {np.median(q('z_ratio')):.2f}); ratio the image size implies {np.median(q('size_ratio')):.2f}")
        for ses in ("M0", "M6"):
            o_ = [o for m, o in S if m["session"] == ses]; print(f"   {ses}: floor slope {np.mean([o['slope'] for o in o_]):.1f} deg (sd {np.std([o['slope'] for o in o_]):.1f}), k median {np.nanmedian([o['k'] for o in o_]):.2f}, speed (stance foot) {np.nanmedian([o['v_body'] for o in o_]):.2f} m/s, n={len(o_)}")
        by = defaultdict(lambda: defaultdict(list))
        for m, o in S: by[m["subject_id"]][m["session"]].append(o["slope"])
        d = np.array([np.mean(v["M6"]) - np.mean(v["M0"]) for v in by.values() if v["M0"] and v["M6"]]); print(f"   camera tilt change M0->M6 in the same patient: mean {d.mean():+.1f} deg, sd {d.std():.1f}, larger than 2 deg in {np.mean(np.abs(d) > 2):.0%} of {len(d)}")
