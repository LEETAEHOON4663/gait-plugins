"""KOA-PD-NM should not change when the image size is known (nobody leaves the image). One process. Usage: check_koa_unchanged.py <nova77_full>"""
import csv, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from core import nova77
from core.common import prepare
from scripts.dataset import frame_sizes
warnings.filterwarnings("ignore")
data = Path(sys.argv[1]); sizes = frame_sizes(data); changed = same = 0
for m in csv.DictReader(open(data / "frame_reports/video_metadata_v0_1.csv", encoding="utf-8-sig")):
    f = data / "frame" / m["output"]
    if m["dataset"] != "KOA-PD-NM" or not f.is_file(): continue
    a, b = prepare(nova77.load(f)), prepare(nova77.load(f, size=sizes[m["output"]]))
    if len(a.walk) != len(b.walk): changed += 1; print("changed", m["output"], len(a.walk), "->", len(b.walk))
    else: same += 1
print("same", same, "changed", changed)
