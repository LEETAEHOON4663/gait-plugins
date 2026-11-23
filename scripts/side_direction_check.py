"""Side view only: do the new Sabo models agree between the two walking directions, and do they separate groups? One recording at a
time, one process, so it stays light on a shared server. Usage: side_direction_check.py <nova77_full> <out>"""
import csv, json, sys, time, warnings
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from core import nova77
from core.common import prepare
from core.plugin import discover, run_one
from scripts.dataset import frame_sizes
warnings.filterwarnings("ignore"); torch.set_num_threads(1)
data, out = Path(sys.argv[1]), Path(sys.argv[2])
BY = {p.manifest.id: p for p in discover()[0]}
IDS = ["pd_stgcn_sabo2022", "pd_stgcn_sabo2022_ap2d", "pd_stgcn_sabo2022_dt2d", "gait_events_zeni2008", "gait_gaitlab2020"]
meta = [m for m in csv.DictReader(open(data / "frame_reports/video_metadata_v0_1.csv", encoding="utf-8-sig")) if m["dataset"] == "KOA-PD-NM" and (data / "frame" / m["output"]).is_file()]
sizes, rows, t0 = frame_sizes(data), [], time.time()
for n, m in enumerate(meta):
    p = prepare(nova77.load(data / "frame" / m["output"], size=sizes.get(m["output"])))
    r = {"subject": f'{m["group"]}_{m["severity"]}_{m["subject_id"]}', "group": m["group"], "direction": m["direction"], "view": p.axes.view}
    for i in IDS:
        x = run_one(BY[i], p); r[i] = x.score if x.status == "ok" else float("nan")
    rows.append(r)
    if n % 50 == 0: print(n, f"{time.time() - t0:.0f}s", flush=True)
(out / "side_direction.json").write_text(json.dumps(rows))
for i in IDS:
    by = defaultdict(dict)
    for r in rows:
        if np.isfinite(r[i]) and r["view"] == "Side": by[r["subject"]][r["direction"]] = r[i]
    d = [abs(x["left to right"] - x["right to left"]) for x in by.values() if len(x) == 2]
    scale = np.nanstd([r[i] for r in rows if np.isfinite(r[i])])
    print(f"{i:26s} left-to-right vs right-to-left: mean |difference| {np.mean(d):.2f} (spread of all scores SD {scale:.2f}), n={len(d)}")
