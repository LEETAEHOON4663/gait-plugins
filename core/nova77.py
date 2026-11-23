"""Extractor adapter: NVIDIA NIM 3D Body Pose (Nova-77) JSON Lines -> canonical Skeleton.

To support another extractor, write one more function that returns a Skeleton. Plugins do not change.
Joint order follows nova77_joints_v0_1.csv / NVIDIA SOMA-77 (sides are the subject's own left and right).
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .skeleton import MISSING, OBSERVED, InputError, Skeleton

_FINGERS = [f"{f}{i}" for f in ("Index", "Middle", "Ring", "Pinky") for i in (1, 2, 3, 4, "End")]
_ARM = ["Shoulder", "Arm", "ForeArm", "Hand"] + [f"HandThumb{i}" for i in (1, 2, 3, "End")] + [f"Hand{f}" for f in _FINGERS]
_LEG = ["Leg", "Shin", "Foot", "ToeBase", "ToeEnd"]
JOINTS = tuple(["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw", "LeftEye", "RightEye"]
               + ["Left" + n for n in _ARM] + ["Right" + n for n in _ARM]
               + ["Left" + n for n in _LEG] + ["Right" + n for n in _LEG])
assert len(JOINTS) == 77

# `bbox` is in NVIDIA's official client output but not in the team's export, and no plugin needs it.
REQUIRED = {"tracking_id": None, "keypoints_2d": (77, 2), "keypoints_3d": (77, 3), "keypoints_confidence": (77,)}


def load(path, fps: float | None = None, size=None) -> Skeleton:
    """`size` is (width, height) of the frames the 2D joints refer to. With it, plugins can tell when the body leaves the image."""
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".gz":                                      # the bundled samples are gzipped; the hash below is of the JSONL itself
        raw = gzip.decompress(raw)
    rows, prev = [], -1
    for n, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise InputError(f"line {n}: invalid JSON") from e
        if not isinstance(r, dict) or not {"frame_id", "detections"} <= r.keys():
            raise InputError(f"line {n}: expected one frame object with frame_id and detections")
        if r["frame_id"] <= prev:
            raise InputError(f"line {n}: frame_id must increase")
        prev = r["frame_id"]
        rows.append(r)
    if not rows:
        raise InputError("no frames")

    warnings = []
    tracks = Counter(d["tracking_id"] for r in rows for d in r["detections"] if "tracking_id" in d)
    if len(tracks) > 1:
        warnings.append(f"several_people_kept_longest_track:{len(tracks)}")
    keep = tracks.most_common(1)[0][0] if tracks else None

    t = len(rows)
    xyz, xy = np.full((t, 77, 3), np.nan), np.full((t, 77, 2), np.nan)
    conf, status = np.full((t, 77), np.nan), np.full((t, 77), MISSING, np.uint8)
    for i, r in enumerate(rows):
        for d in r["detections"]:
            lacking = [k for k in REQUIRED if k not in d]
            if lacking:
                raise InputError(f"frame {r['frame_id']}: missing {lacking}")
            if d["tracking_id"] != keep:
                continue
            arr = {}
            for k, shape in REQUIRED.items():
                if shape is None:
                    continue
                a = np.asarray(d[k], float)
                if a.shape != shape or not np.isfinite(a).all():
                    raise InputError(f"frame {r['frame_id']}: {k} must be finite with shape {shape}")
                arr[k] = a
            xyz[i], xy[i], conf[i] = arr["keypoints_3d"], arr["keypoints_2d"], arr["keypoints_confidence"]
            status[i] = OBSERVED

    ids = np.array([r["frame_id"] for r in rows])
    if fps is None:
        # time_ms is rounded to whole ms, so use the full span, not the median step (33 ms would give 30.3 fps)
        ms = [r.get("time_ms") for r in rows]
        if len(rows) > 1 and None not in (ms[0], ms[-1]) and ms[-1] > ms[0]:
            fps = round((ids[-1] - ids[0]) * 1000.0 / (ms[-1] - ms[0]), 2)
        else:
            raise InputError("fps is unknown: pass fps or include time_ms")
    return Skeleton(JOINTS, xyz, xy, conf, status, float(fps), ids,
                    {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "extractor": "nvidia-nim-body-pose/nova77", **({"size": (int(size[0]), int(size[1]))} if size else {})},
                    tuple(warnings))
