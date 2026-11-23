"""Common layer: gap filling, main walking segment, camera view and walking axes.

Every step only adds information (status flags, axes). The raw skeleton stays available to every plugin.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .skeleton import INTERPOLATED, MISSING, Skeleton

MAX_GAP = 5                              # frames; longer gaps split the walk instead of being invented
CAMERA_UP = np.array([0.0, -1.0, 0.0])   # assumes a level camera (image y points down)


def fill_short_gaps(s: Skeleton, max_gap: int = MAX_GAP) -> Skeleton:
    """Linear interpolation inside gaps of at most max_gap frames. Filled joints are marked INTERPOLATED."""
    xyz, xy, status = s.xyz.copy(), s.xy.copy(), s.status.copy()
    for j in range(len(s.joints)):
        seen = np.flatnonzero(s.status[:, j] != MISSING)
        for a, b in zip(seen[:-1], seen[1:]):
            if 1 < b - a <= max_gap + 1:
                w = (np.arange(a + 1, b) - a)[:, None] / (b - a)
                xyz[a + 1:b, j] = (1 - w) * s.xyz[a, j] + w * s.xyz[b, j]
                xy[a + 1:b, j] = (1 - w) * s.xy[a, j] + w * s.xy[b, j]
                status[a + 1:b, j] = INTERPOLATED
    return replace(s, xyz=xyz, xy=xy, status=status)


BODY_JOINTS = ["Hips", "Neck1", "Head", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand", "LeftLeg", "RightLeg",
               "LeftShin", "RightShin", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase", "LeftToeEnd", "RightToeEnd"]


def inside_image(s: Skeleton, joints, size, margin: float = 0.01) -> np.ndarray:
    """(T,) frames in which every given joint lies inside the image. A joint outside is only the extractor's guess."""
    w, h = size
    xy = s.xy[:, [s.index(j) for j in joints]]
    return ((xy[..., 0] >= -margin * w) & (xy[..., 0] <= (1 + margin) * w) & (xy[..., 1] >= -margin * h) & (xy[..., 1] <= (1 + margin) * h)).all(axis=1)


def main_segment(s: Skeleton, joints=None, size=None) -> Skeleton:
    """Longest run of frames in which all (or the given) joints are observed or interpolated and, if the image size is known, the
    whole body is inside the image."""
    cols = slice(None) if joints is None else [s.index(j) for j in joints]
    ok = (s.status[:, cols] != MISSING).all(axis=1)
    if size is not None and len(s):
        ok &= inside_image(s, BODY_JOINTS, size)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
    runs = list(zip(edges[::2], edges[1::2]))
    if not runs:
        return s.slice(0, 0)
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return s.slice(int(a), int(b))


@dataclass(frozen=True)
class WalkAxes:
    forward: np.ndarray
    left: np.ndarray
    up: np.ndarray
    view: str                 # Side / Front / Oblique / Unknown
    direction: str            # LeftToRight / RightToLeft / TowardCamera / AwayFromCamera / Unknown
    distance_m: float         # model-estimated, uncalibrated
    left_right_ok: bool       # the subject's left hip really lies on the left of the walking direction


def walking_axes(s: Skeleton) -> WalkAxes:
    unknown = WalkAxes(*[np.full(3, np.nan)] * 3, "Unknown", "Unknown", float("nan"), False)
    if len(s) < 10:
        return unknown
    k = max(1, len(s) // 10)
    hips = s.point("Hips")
    d = hips[-k:].mean(0) - hips[:k].mean(0)
    d = d - d.dot(CAMERA_UP) * CAMERA_UP
    dist = float(np.linalg.norm(d))
    if not np.isfinite(dist) or dist < 0.3:       # standing still or turning on the spot
        return unknown
    f = d / dist
    left = np.cross(CAMERA_UP, f)
    dx, dz = abs(d[0]), abs(d[2])
    if dx >= 2 * dz:
        view, direction = "Side", "LeftToRight" if d[0] > 0 else "RightToLeft"
    elif dz >= 2 * dx:
        view, direction = "Front", "AwayFromCamera" if d[2] > 0 else "TowardCamera"
    else:
        view, direction = "Oblique", "Unknown"
    lr = float(np.mean((s.point("LeftLeg") - s.point("RightLeg")) @ left))
    return WalkAxes(f, left, CAMERA_UP.copy(), view, direction, dist, lr > 0)


def in_walk_frame(xyz: np.ndarray, ax: WalkAxes) -> np.ndarray:
    """Rotate camera coordinates to (forward, left, up). Rotation only: no centring, no scaling."""
    return xyz @ np.stack([ax.forward, ax.left, ax.up], axis=1)


@dataclass(frozen=True)
class Prepared:
    raw: Skeleton        # exactly what the extractor produced
    clean: Skeleton      # short gaps filled and flagged
    walk: Skeleton       # longest continuous walking segment of clean
    axes: WalkAxes       # level-camera axes; used for view and direction
    world: object = None  # core.view.World: the walk in (forward, left, up) with the vertical taken from the floor


def prepare(s: Skeleton) -> Prepared:
    clean = fill_short_gaps(s)
    walk = main_segment(clean, size=s.source.get("size"))
    from .view import build_world                    # here to avoid a circular import
    ax = walking_axes(walk)
    return Prepared(s, clean, walk, ax, build_world(walk) if ax.view != "Unknown" else None)
