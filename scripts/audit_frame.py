"""How often does the body leave the image? One recording at a time. Usage: audit_frame.py <nova77_full> <out>"""
import csv, json, sys, warnings
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from core import jointsets, nova77
from core.common import prepare
from core.plugin import outside_share
from scripts.dataset import frame_sizes
warnings.filterwarnings("ignore")
data, out = Path(sys.argv[1]), Path(sys.argv[2]); sizes = frame_sizes(data)
SETS = {"kinect13 (ST-GCN 3D)": "kinect13", "gaitlab_lower": "gaitlab_lower", "lower_limb_arms (Zeni)": "lower_limb_arms"}
rows, n = [], 0
for m in csv.DictReader(open(data / "frame_reports/video_metadata_v0_1.csv", encoding="utf-8-sig")):
    f = data / "frame" / m["output"]
    if not f.is_file() or m["output"] not in sizes: continue
    p = prepare(nova77.load(f, size=sizes[m["output"]])); n += 1
    r = {"file": m["output"], "dataset": m["dataset"], "group": m["group"], "view": p.axes.view, "frames": len(p.walk)}
    for name, js in SETS.items():
        cols = [p.walk.index(j) for j in jointsets.canonical_joints(js)]
        r[name] = outside_share(p.walk.xy[:, cols], sizes[m["output"]]) if len(p.walk) else float("nan")
    rows.append(r)
    if n % 100 == 0: print(n, flush=True)
(out / "frame_audit.json").write_text(json.dumps(rows))
for ds in ("KOA-PD-NM", "Hip_OA"):
    s = [r for r in rows if r["dataset"] == ds and r["frames"] >= 45]
    print(ds, len(s), "recordings; share of walking frames with a joint outside the image:")
    for name in SETS:
        v = np.array([r[name] for r in s])
        print(f"   {name:24s} median {np.median(v):.0%}, over 20%: {np.sum(v > .2)} ({np.mean(v > .2):.0%}), over 50%: {np.mean(v > .5):.0%}")
