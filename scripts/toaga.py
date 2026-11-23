"""TOAGA: file index, 2D tracks, and the study-style input. Used by toaga_baseline.py and toaga_compare.py."""
import csv, re
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from scripts.study_code import JOINTS, centre

NAME = re.compile(r"OAW(\d+)-(\w+)-(top|bottom)-(front|back)-(\d+)\.csv$", re.I)


SIZE = (1080, 1920)      # the video is portrait; used only to drop joints outside the image
# Normalisation: as the study's landscape data. toaga_norm_check.py decided this from the checkpoints' input statistics: body height
# 296 px here vs 291 px in training, while the portrait rule squashes the body and the model answers 1.50 +- 0.03 for every walk.
STUDY_RES = (1920, 1080)


def index(root):
    """{(person, camera, direction, walk): {engine: path}}"""
    out = {}
    for p in sorted(Path(root).rglob("*.csv")):
        m = NAME.search(p.name)
        if m:
            person, eng, cam, d, w = m.groups()
            out.setdefault((int(person), cam.lower(), d.lower(), int(w)), {})[eng.lower()] = p
    return out


def read(path):
    """time (T,), xy (T, 13, 2) px with NaN where missing, conf (T, 13). OpenPose writes (0, 0) for joints it did not find, and a
    few tracks leave the 1080 x 1920 image; both count as missing."""
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    num = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "NaN", "nan") else np.nan
    a = np.array([[[num(r, j + s) for s in ("_x", "_y", "_conf")] for j in JOINTS] for r in rows])
    xy = a[..., :2]
    bad = (xy == 0).all(-1) | (xy < -1).any(-1) | (xy[..., 0] > SIZE[0] + 1) | (xy[..., 1] > SIZE[1] + 1)
    xy[bad] = np.nan
    return np.array([num(r, "time") for r in rows]), xy, a[..., 2]


def at_30fps(t, xy):
    """Videos have 24-30 fps; the study works at 30. Linear resampling on the real time stamps."""
    n = int(round((t[-1] - t[0]) * 30)) + 1
    new = t[0] + np.arange(n) / 30
    return new, np.stack([[np.interp(new, t, xy[:, j, c]) for c in range(2)] for j in range(xy.shape[1])]).transpose(2, 0, 1)


def fill(xy):
    """Linear interpolation per joint; a joint never seen stays NaN."""
    xy = xy.copy(); t = np.arange(len(xy))
    for j in range(xy.shape[1]):
        ok = np.isfinite(xy[:, j]).all(1)
        if ok.sum() >= 2:
            for c in range(2):
                xy[:, j, c] = np.interp(t, t[ok], xy[ok, j, c])
    return xy


def smooth(xy, fps=30.0, cut=8.0):
    """Zero-lag 2nd-order Butterworth, 8 Hz, as TOAGA describes for its trajectories."""
    if len(xy) < 15 or not np.isfinite(xy).all():
        return xy
    b, a = butter(2, cut / (fps / 2)); return filtfilt(b, a, xy, axis=0)


def study_input(xy, conf=None, time=None):
    """The study's 2D input: gaps filled, hip centre at (100, 100). conf is only used by the study to blank unseen joints."""
    xy = fill(xy)
    if time is not None and np.isfinite(xy).all():
        xy, conf = at_30fps(time, xy)[1], None
    xy = smooth(xy)
    return centre(xy), np.ones(xy.shape[:2]) if conf is None else np.where(np.isfinite(conf), np.maximum(conf, 1e-3), 1e-3)


ALIGN = [JOINTS.index(n) for n in ("LShoulder", "RShoulder", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle")]


def align_pair(a, b, min_len=90, max_err=25.0):
    """Two tracks of the same walk, cut at different places: (start in a, start in b, length) of the common frames, or None.
    The extractors' files start at different times, so only frames that show the same moment may be compared."""
    ai, bi = a[:, ALIGN], b[:, ALIGN]; best = (np.inf, None)
    for o in range(-(len(b) - min_len), len(a) - min_len + 1):           # frame 0 of b is frame o of a
        s0, e0 = max(0, o), min(len(a), o + len(b))
        if e0 - s0 < min_len: continue
        d = np.abs(ai[s0:e0] - bi[s0 - o:e0 - o])
        e = max(np.nanmedian(d[..., 0]), np.nanmedian(d[..., 1]))
        if e < best[0]: best = (e, (s0, s0 - o, e0 - s0))
    return best[1] if best[0] < max_err else None


def seconds_ok(xy):
    return np.isfinite(xy).all()
