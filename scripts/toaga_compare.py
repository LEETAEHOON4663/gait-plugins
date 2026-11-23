"""Nova-77 on TOAGA against the study, judged by the criteria fixed in toaga_criteria.py.

1. Find where each TOAGA walk sits in the long video (the walk files have their own clock) by matching 2D positions.
2. Feed Nova-77's own 2D joints to the study's AlphaPose and Detectron models; compare with the study's pipelines (no view change).
3. View change on real data: 3D from the bottom camera shown from a virtual top camera, scored, and compared with the real top camera.
Usage: toaga_compare.py <folder with OAW01-top.jsonl ...> <PoseTracking> <stgcn repo> <out>"""
import json, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from core import jointsets, nova77
from core.common import fill_short_gaps, prepare
from core.view import Camera, build_world, fit_pinhole, look
from scripts.study_code import JOINTS, StudyModel
from scripts.toaga import STUDY_RES, align_pair, at_30fps, centre, fill, index, read, smooth
from scripts.toaga_criteria import summarise, verdict
warnings.filterwarnings("ignore"); torch.set_num_threads(4)
USE = [JOINTS.index(n) for n in ("LShoulder", "RShoulder", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle")]


def align(long_xy, walk_xy, max_err=40.0):
    """Offset of walk_xy (T, 13, 2) inside long_xy (N, 13, 2), scale between the two pixel grids, and the error in walk pixels."""
    w = walk_xy[:, USE].reshape(len(walk_xy), -1)
    best = (None, np.nan, np.inf)
    for o in range(0, len(long_xy) - len(walk_xy) + 1):
        a = long_xy[o:o + len(walk_xy)][:, USE].reshape(len(walk_xy), -1)
        ok = np.isfinite(a)
        if ok.mean() < 0.9: continue
        s = (a[ok] * w[ok]).sum() / (a[ok] ** 2).sum()
        d = np.abs(a * s - w).reshape(len(walk_xy), -1, 2)            # per axis: a distortion along one axis must not hide
        e = max(np.nanmedian(d[..., 0]), np.nanmedian(d[..., 1]))
        if e < best[2]: best = (o, s, e)
    return best if best[2] < max_err else (None, *best[1:])


def main(nim_dir, tracks, repo, out, only_front=True, view_change=True):
    nim_dir, out = Path(nim_dir), Path(out)
    idx, models = index(tracks), {e: StudyModel(repo, e) for e in ("alphapose", "detectron")}
    res, notes = {}, []
    files = {(int(p.stem[3:5]), p.stem.split("-")[1]): p for p in sorted(nim_dir.glob("OAW*.jsonl"))}
    print(f"Nova-77 files found: {len(files)} of 28")
    for (person, cam), path in files.items():
        s = nova77.load(path); n2d = jointsets.select_xy(fill_short_gaps(s), "coco13_nose")[1]
        for k, eng in idx.items():
            if k[:2] != (person, cam) or (only_front and k[2] != "front") or "alphapose" not in eng: continue
            t, xy, _ = read(eng["alphapose"]); ap = at_30fps(t, fill(xy))[1]
            o, sc, err = align(n2d, ap)
            if o is None:
                notes.append(f"{k}: not found in the video (best {err:.0f} px)"); continue
            seg = n2d[o:o + len(ap)] * sc
            if not np.isfinite(seg).all(): notes.append(f"{k}: gaps in the Nova-77 track"); continue
            r = {"offset": int(o), "scale": float(sc), "align_px": float(err)}
            ref = {"alphapose": ap}
            if "detectron" in eng:                                       # only the frames both extractors cover
                td, xd, _ = read(eng["detectron"]); dt = at_30fps(td, fill(xd))[1]; pos = align_pair(ap, dt)
                if pos is not None:
                    i, j, n = pos; ap_c, seg, ref = ap[i:i + n], seg[i:i + n], {"alphapose": ap[i:i + n], "detectron": dt[j:j + n]}
                    r["offset"] += i
            for m, sm in models.items():
                f = lambda z: float(np.mean(sm.scores(centre(smooth(z)), np.ones(z.shape[:2]), STUDY_RES)))
                r[f"{m}|nova2d"] = f(seg)
                for e, z in ref.items():
                    r[f"{m}|{e}"] = f(z) if np.isfinite(z).all() else np.nan
            res[k] = r
    print(f"walks matched: {len(res)}, not matched or not used: {len(notes)}"); [print("   ", n) for n in notes[:8]]
    if res: print(f"image scale between the files: {np.median([r['scale'] for r in res.values()]):.3f} (should be the same everywhere; spread {np.ptp([r['scale'] for r in res.values()]):.3f}), alignment error {np.median([r['align_px'] for r in res.values()]):.1f} px")
    get = lambda n: [r.get(n, np.nan) for r in res.values()]; people = [k[0] for k in res]
    usable = sum(1 for k, e in idx.items() if (k[:2] in files) and (k[2] == "front" or not only_front) and "alphapose" in e)
    coverage = len(res) / max(usable, 1)
    report = {"matched": len(res), "usable": usable}
    print("\n1. study model fed with Nova-77's own 2D joints (no view change), judged against swapping AlphaPose and Detectron")
    for m, other in (("alphapose", "detectron"), ("detectron", "alphapose")):
        yard = summarise(get(f"{m}|{m}"), get(f"{m}|{other}"), people); new = summarise(get(f"{m}|{m}"), get(f"{m}|nova2d"), people)
        ok, why = verdict(new, yard, coverage); report[f"{m}_2d"] = dict(new=new, yard=yard, passed=ok)
        print(f"   {m} model: {'PASS' if ok else 'NO VERDICT' if ok is None else 'FAIL'} (bias {new['bias']:+.2f}, n={new['n']})"); [print("       ", w) for w in why]
    cross = cross_camera(files, idx, models, res) if view_change else []
    if cross:
        print("\n2. view change on real data: bottom camera's 3D shown from a virtual top camera, against the real top camera")
        for m in models:
            pp = [c["person"] for c in cross]; col = lambda n: [c[n] for c in cross]
            yard = summarise(col(f"{m}|top_ap"), col(f"{m}|top_dt"), pp); new = summarise(col(f"{m}|top_ap"), col(f"{m}|virtual"), pp)
            nov = summarise(col(f"{m}|top_nova2d"), col(f"{m}|virtual"), pp)
            ok, why = verdict(new, yard); report[f"{m}_view"] = dict(new=new, yard=yard, passed=ok, against_nova2d=nov)
            print(f"   {m} model: {'PASS' if ok else 'NO VERDICT' if ok is None else 'FAIL'} (bias {new['bias']:+.2f}, n={new['n']}); same test against Nova-77's own top 2D: mean |diff| {nov['diff']:.2f}, rank corr {nov['rho']:.2f}"); [print("       ", w) for w in why]
    out.mkdir(parents=True, exist_ok=True); (out / "toaga_compare.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def cross_camera(files, idx, models, res):
    out = []
    for (p, c, d, w), r in list(res.items()):
        if c != "bottom" or (p, "top", d, w) not in res or (p, "top") not in files or (p, "bottom") not in files: continue
        def seg(cam):
            s = nova77.load(files[p, cam]); k = (p, cam, d, w); n = len(at_30fps(read(idx[k]["alphapose"])[0], fill(read(idx[k]["alphapose"])[1]))[0])
            return s.slice(res[k]["offset"], res[k]["offset"] + n)
        try:
            b, t = (prepare(seg(c_)) for c_ in ("bottom", "top"))
            if b.world is None or t.world is None or len(b.walk) < 45: continue
            f = fit_pinhole(t.walk)[0] * res[p, "top", d, w]["scale"]; z = np.median(t.walk.point("Hips")[:, 2])
            toward = 0.0 if d == "front" else 180.0
            cam = Camera(azimuth=toward, elevation=abs(t.world.floor_slope), output="2d", distance_m=float(z), focal_px=float(f), size=(1080, 1920))
            view = look(jointsets.select(b.world.body, "coco13_nose", b.world.xyz)[1], cam)
            row = {"person": p}
            for m, sm in models.items():
                row[f"{m}|virtual"] = float(np.mean(sm.scores(centre(smooth(fill(view))), np.ones(view.shape[:2]), STUDY_RES)))
                q = res[p, "top", d, w]; row[f"{m}|top_ap"], row[f"{m}|top_dt"], row[f"{m}|top_nova2d"] = q[f"{m}|alphapose"], q.get(f"{m}|detectron", np.nan), q[f"{m}|nova2d"]
            out.append(row)
        except Exception as e:
            print("    cross-camera skipped", (p, d, w), type(e).__name__, e)
    return out


if __name__ == "__main__":
    main(*sys.argv[1:5])
