"""Run the real pipeline on a few recordings and write everything the demo page shows. Usage: export_demo.py <nova77_full> <out.json>"""
import csv, json, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from core import jointsets, nova77
from core.plugin import discover, run
from core.common import in_walk_frame
warnings.filterwarnings("ignore")
data, out = Path(sys.argv[1]), Path(sys.argv[2])
BODY = {"pelvis": "Hips", "chest": "Chest", "neck": "Neck1", "head": ("Head", "HeadEnd"), "lsho": "LeftArm", "lelb": "LeftForeArm", "lwri": "LeftHand", "rsho": "RightArm",
        "relb": "RightForeArm", "rwri": "RightHand", "lhip": "LeftLeg", "lkne": "LeftShin", "lank": "LeftFoot", "ltoe": "LeftToeBase", "ltip": "LeftToeEnd",
        "rhip": "RightLeg", "rkne": "RightShin", "rank": "RightFoot", "rtoe": "RightToeBase", "rtip": "RightToeEnd"}
every = discover()[0]
plugins = [p for p in every if p.manifest.kind in ("pretrained", "measurement")]   # published studies and published measurement rules; our own baseline is left out
from shared import gait_features as koa                                  # only its step counter is used, as an independent check
meta = {r["output"]: r for r in csv.DictReader(open(data / "frame_reports/video_metadata_v0_1.csv", encoding="utf-8-sig"))}
size = {}
for l in open(data / "frame_reports/batch_status_v0_1.jsonl", encoding="utf-8"):
    s = json.loads(l); size[s.get("output")] = (s.get("width"), s.get("height"))
K = "KOA-PD-NM/KOA-PD-NM/"


def pick(s, arr, names=BODY):
    return np.stack([np.mean([arr[:, s.index(p)] for p in ((v,) if isinstance(v, str) else v)], axis=0) for v in names.values()], axis=1)


def case(rel, title):
    p, results = run(nova77.load(data / "frame" / rel, size=size.get(rel)), plugins)
    m, n = meta[rel], len(p.walk)
    w = p.walk if n else p.clean
    cap = 260                                                                  # at most about 8.7 s in the page
    w = w.slice(0, min(len(w), cap))
    c = {"file": Path(rel).name, "title": title, "dataset": m["dataset"], "group": m["group"], "severity": m["severity"], "size": size.get(rel),
         "frames": len(p.raw), "walk": n, "fps": p.raw.fps, "detected": int((p.raw.status[:, 0] == 1).sum()), "filled": round(float((p.walk.status == 2).mean()), 3) if n else None,
         "view": p.axes.view, "direction": p.axes.direction, "lr_ok": bool(p.axes.left_right_ok)}
    c["xy"] = np.round(pick(w, w.xy), 1).tolist() if len(w) and np.isfinite(pick(w, w.xy)).all() else None
    if p.world is not None and np.isfinite(p.world.xyz[:len(w)]).all():
        x = pick(p.world.body, p.world.xyz)[:len(w)]
        c["world"] = np.round((x - x[0, 0]) * 1000).astype(int).tolist()        # mm, (forward, left, up)
        c["tilt"], c["depth"] = round(p.world.floor_slope, 1), round(p.world.info.get("depth_scale", 1.0), 2)
    if p.axes.view == "Side" and n >= 60:
        names, cam = jointsets.select(p.walk, "lower_limb_arms")
        st = koa.compute(names, in_walk_frame(cam, p.axes), p.raw.fps)["stride_time_s"]
        c["cadence_counted"] = round(1 / st, 3) if np.isfinite(st) and st > 0 else None
    c["results"] = [{"id": r.plugin, "condition": r.condition, "status": r.status, "label": r.label, "score": r.score, "detail": r.detail, "reasons": r.reasons[:1], "warnings": r.warnings} for r in results]
    return c


def first(prefix, group_ok):
    for rel in sorted(k for k in meta if k.startswith(prefix) and (data / "frame" / k).is_file()):
        p, results = run(nova77.load(data / "frame" / rel, size=size.get(rel)), plugins)
        if group_ok(p, results): return rel
    raise SystemExit("no example for " + prefix)


ok_all = lambda p, r: all(x.status == "ok" for x in r)
cases = [case(first(K + "NM/", ok_all), "정상군"), case(K + "PD/PD_MD/001_PD_01_MD.jsonl", "파킨슨병 (중등도)"), case(K + "KOA/KOA_SV/001_KOA_01_SV.jsonl", "무릎 관절염 (중증)"),
         case(first("Hip_OA/Patients/HOA001/M0/", lambda p, r: p.axes.direction == "AwayFromCamera" and p.axes.left_right_ok), "고관절 관절염, 수술 전 (카메라 반대쪽으로)"),
         case(first(K + "KOA/KOA_EL/", lambda p, r: p.axes.view == "Oblique"), "비스듬히 찍힌 영상")]
man = {p.manifest.id: {"study": p.manifest.study, "source": p.manifest.source, "output": p.manifest.output, "license": p.manifest.license, "joint_set": p.manifest.requires.joint_set,
                       "joints": list(jointsets.JOINT_SETS[p.manifest.requires.joint_set]), "views": p.manifest.requires.views, "min_frames": p.manifest.requires.min_frames, "fidelity": p.manifest.fidelity} for p in plugins}
out.write_text(json.dumps({"body": list(BODY), "plugins": man, "cases": cases}, ensure_ascii=False, separators=(",", ":")))
for c in cases:
    print(c["title"], "|", c["file"], "|", c["view"], c["direction"], "| walk", c["walk"], "of", c["frames"], "| tilt", c.get("tilt"), "depth", c.get("depth"), "| xy", c["xy"] is not None, "world", "world" in c)
    for r in c["results"]: print("     ", r["id"], r["status"], r["label"] or r["reasons"], r["warnings"])
print("bytes", out.stat().st_size)
