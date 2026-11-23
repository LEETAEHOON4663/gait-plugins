"""KOA-PD-NM videos with labels and demographics. Subject numbers restart in each severity folder, so the subject key
includes group and severity."""
import csv, json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from core import nova77
from core.common import prepare

ROOT = Path(__file__).resolve().parent.parent


def _load(path, size=None):
    return prepare(nova77.load(path, size=size))


def frame_sizes(data):
    """{jsonl path under frame/: (width, height)} from the export's status file."""
    out = {}
    for l in open(Path(data) / "frame_reports/batch_status_v0_1.jsonl", encoding="utf-8"):
        s = json.loads(l)
        if s.get("width"): out[s["output"]] = (s["width"], s["height"])
    return out


def koa_pd_nm(data, workers=6):
    data = Path(data)
    demo = {(r["group"], r["severity"], r["subject_id"]): r for r in csv.DictReader(open(ROOT / "data/koa_pd_nm_demographics.csv"))}
    meta = [r for r in csv.DictReader(open(data / "frame_reports/video_metadata_v0_1.csv", encoding="utf-8-sig"))
            if r["dataset"] == "KOA-PD-NM" and (data / "frame" / r["output"]).is_file()]
    sizes = frame_sizes(data)
    with ProcessPoolExecutor(workers) as ex:
        prepared = list(ex.map(_load, [data / "frame" / r["output"] for r in meta], [sizes.get(r["output"]) for r in meta], chunksize=4))
    out = []
    for r, p in zip(meta, prepared):
        d = demo[(r["group"], r["severity"], r["subject_id"])]
        out.append({"file": r["output"], "subject": f'{r["group"]}_{r["severity"]}_{r["subject_id"]}', "group": r["group"],
                    "severity": r["severity"], "direction": r["direction"], "age": float(d["age"]), "sex": d["sex"], "prepared": p})
    return out


def subject_mean(rows):
    """One row per subject: numeric fields averaged over that subject's videos (NaN ignored)."""
    by = defaultdict(list)
    for r in rows:
        by[r["subject"]].append(r)
    out = []
    for s, rs in sorted(by.items()):
        o = {}
        for k, v in rs[0].items():
            if isinstance(v, str):
                o[k] = v
            else:
                a = np.array([r[k] for r in rs], float)
                o[k] = float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
        o["n_videos"] = len(rs)
        out.append(o)
    return out
