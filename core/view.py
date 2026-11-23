"""3D layer: true vertical from the floor, and a virtual camera that shows the same walk from any direction, tilt and lens a
study used.

    camera 3D --floor--> (forward, left, up) --Camera--> the study's view

Rotating the view is exact. What it cannot fix is depth error of the monocular 3D estimate: turning the view turns that error
into something visible. Use scripts/compare_views.py to see how large it is.

refine_with_2d (bone lengths from the image) is off by default: in compare_views.py it made two recordings of the same person
agree less (7.3 cm instead of 6.3 cm), although it lowers the reprojection error.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .common import CAMERA_UP, WalkAxes, walking_axes
from .skeleton import Skeleton

# ---------------------------------------------------------------- bone tree (Nova-77, body only; fingers and face follow)

_TRUNK = ["Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd"]
_GROUP = {**{n: "trunk" for n in _TRUNK}}
_PARENT = {"Spine1": "Hips", "Spine2": "Spine1", "Chest": "Spine2", "Neck1": "Chest", "Neck2": "Neck1", "Head": "Neck2",
           "HeadEnd": "Head", "Jaw": "Head", "LeftEye": "Head", "RightEye": "Head"}
for _s in ("Left", "Right"):
    _PARENT.update({f"{_s}Shoulder": "Chest", f"{_s}Arm": f"{_s}Shoulder", f"{_s}ForeArm": f"{_s}Arm", f"{_s}Hand": f"{_s}ForeArm",
                    f"{_s}Leg": "Hips", f"{_s}Shin": f"{_s}Leg", f"{_s}Foot": f"{_s}Shin", f"{_s}ToeBase": f"{_s}Foot", f"{_s}ToeEnd": f"{_s}ToeBase"})
    _GROUP.update({f"{_s}Shoulder": "shoulder", f"{_s}Arm": "shoulder", f"{_s}ForeArm": "upper_arm", f"{_s}Hand": "forearm", f"{_s}Leg": "hip",
                   f"{_s}Shin": "thigh", f"{_s}Foot": "shank", f"{_s}ToeBase": "foot", f"{_s}ToeEnd": "foot"})
GROUPS = ["trunk", "shoulder", "upper_arm", "forearm", "hip", "thigh", "shank", "foot"]
_USE = ["Hips", "Chest", "Neck1", "Head", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand", "LeftLeg", "RightLeg",
        "LeftShin", "RightShin", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"]


def fit_pinhole(s: Skeleton) -> tuple[float, float, float]:
    """Focal length and principal point that the extractor's 3D and 2D imply (u = f X/Z + cx, v = f Y/Z + cy)."""
    cols = [s.index(n) for n in _USE]
    X, u = s.xyz[:, cols].reshape(-1, 3), s.xy[:, cols].reshape(-1, 2)
    ok = np.isfinite(X).all(1) & np.isfinite(u).all(1)
    X, u = X[ok], u[ok]
    a, b, o, z = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2], np.ones(len(X)), np.zeros(len(X))
    A = np.concatenate([np.stack([a, o, z], 1), np.stack([b, z, o], 1)])
    f, cx, cy = np.linalg.lstsq(A, np.concatenate([u[:, 0], u[:, 1]]), rcond=None)[0]
    return float(f), float(cx), float(cy)


def project(xyz: np.ndarray, cam) -> np.ndarray:
    f, cx, cy = cam
    return np.stack([f * xyz[..., 0] / xyz[..., 2] + cx, f * xyz[..., 1] / xyz[..., 2] + cy], -1)


def reprojection_error(s: Skeleton, cam) -> float:
    cols = [s.index(n) for n in _USE]
    return float(np.nanmedian(np.linalg.norm(project(s.xyz[:, cols], cam) - s.xy[:, cols], axis=-1)))


def refine_with_2d(s: Skeleton, cam=None, prior_sd: float = 0.15, pixel_sd: float = 8.0) -> tuple[Skeleton, dict]:
    """Keep every bone direction from the 3D estimate; change one length scale per bone group (left = right, constant over the
    clip) and the body position per frame so that the skeleton projects onto the 2D joints. Linear least squares; groups the
    view cannot see (hip and shoulder width from the side) stay near 1 because of the prior."""
    cam = cam or fit_pinhole(s)
    f, cx, cy = cam
    root = s.point("Hips")
    # offset of each used joint from the hips, split by bone group: X_j = root + sum_g scale_g * D[j, g]
    D = {"Hips": np.zeros((len(s), len(GROUPS), 3))}
    for n in _PARENT:                                                   # parents come before children
        par = _PARENT[n]
        step = s.xyz[:, s.index(n)] - s.xyz[:, s.index(par)]
        d = D[par].copy()
        if n in _GROUP:
            d[:, GROUPS.index(_GROUP[n])] += step
        D[n] = d
    M = np.zeros((len(GROUPS), len(GROUPS)))
    frames = []
    for t in range(len(s)):
        A, B = [], []
        for n in _USE:
            u, v = s.xy[t, s.index(n)]
            if not np.isfinite([u, v]).all() or not np.isfinite(D[n][t]).all():
                continue
            zest = max(root[t, 2] + D[n][t, :, 2].sum(), 0.3)
            for a, k in ((u - cx, 0), (v - cy, 1)):                      # (u - cx) Z - f X = 0, divided by Z to be in pixels
                row = np.zeros(3); row[2] = a; row[k] = -f
                A.append(row / zest)
                B.append((a * D[n][t, :, 2] - f * D[n][t, :, k]) / zest)
        A, B = np.array(A), np.array(B)
        if len(A) < 12:
            frames.append(None); continue
        Q, _ = np.linalg.qr(A)
        PB = B - Q @ (Q.T @ B)
        M += PB.T @ PB
        frames.append((A, B))
    # One camera cannot see absolute size: shrinking body and distance together fits equally well (and noise then favours
    # shrinking). So trunk + thigh + shank length is held fixed and only the proportions change.
    lam, one = (pixel_sd / prior_sd) ** 2, np.ones(len(GROUPS))
    length = lambda a, b: float(np.nanmedian(np.linalg.norm(s.xyz[:, s.index(a)] - s.xyz[:, s.index(b)], axis=1)))
    w = np.zeros(len(GROUPS))
    for g, a, b in (("trunk", "Hips", "Head"), ("thigh", "LeftLeg", "LeftShin"), ("shank", "LeftShin", "LeftFoot")):
        w[GROUPS.index(g)] = length(a, b)
    K = np.block([[M + lam * np.eye(len(GROUPS)), w[:, None]], [w[None], np.zeros((1, 1))]])
    d = np.linalg.solve(K, np.concatenate([-M @ one, [0.0]]))[:len(GROUPS)]
    scale = one + d
    if scale.min() < 0.7 or scale.max() > 1.4:                       # not a plausible body: keep the extractor's proportions
        scale = one
    new_root = root.copy()
    for t, fr in enumerate(frames):
        if fr is not None:
            new_root[t] = np.linalg.lstsq(fr[0], -fr[1] @ scale, rcond=None)[0]
    xyz = s.xyz.copy()
    body = {"Hips": new_root}
    for n in _PARENT:
        par = _PARENT[n]
        k = scale[GROUPS.index(_GROUP[n])] if n in _GROUP else 1.0
        body[n] = body[par] + k * (s.xyz[:, s.index(n)] - s.xyz[:, s.index(par)])
    for n, v in body.items():
        xyz[:, s.index(n)] = v
    for side in ("Left", "Right"):                                        # fingers ride on the hand
        shift = body[f"{side}Hand"] - s.xyz[:, s.index(f"{side}Hand")]
        for i, n in enumerate(s.joints):
            if n.startswith(f"{side}Hand") and n != f"{side}Hand":
                xyz[:, i] = s.xyz[:, i] + shift
    out = replace(s, xyz=xyz)
    return out, {"camera": cam, "kept_original": bool((scale == 1).all()), "scale": dict(zip(GROUPS, np.round(scale, 3).tolist())),
                 "reprojection_px": (reprojection_error(s, cam), reprojection_error(out, cam))}


# ---------------------------------------------------------------- vertical and walking axes

def _rot(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    a = np.radians(deg)
    return v * np.cos(a) + np.cross(axis, v) * np.sin(a) + axis * axis.dot(v) * (1 - np.cos(a))


def floor_slope_deg(w: Skeleton, ax: WalkAxes) -> float:
    """Slope of the floor along the walking direction, from the lower foot in every frame (Theil-Sen line, robust)."""
    tl, tr = w.point("LeftToeBase"), w.point("RightToeBase")
    low = np.where((tl @ ax.up < tr @ ax.up)[:, None], tl, tr)
    x, h = low @ ax.forward, low @ ax.up
    i = np.arange(0, len(x), max(1, len(x) // 120))
    dx, dh = x[i][:, None] - x[i][None], h[i][:, None] - h[i][None]
    m = np.abs(dx) > 0.25 * np.ptp(x)
    return float(np.degrees(np.arctan(np.median(dh[m] / dx[m])))) if m.any() else 0.0


# A planted foot does not slide, so relative to the pelvis it moves back exactly as fast as the pelvis moves forward.
# In side views, where distance is reliable, this ratio came out at 0.93 (182 clips, the same for NM, PD and KOA). In the front
# views it was 1.38, so there the walked distance is too short by about 1.38 / 0.93 (scripts/explore_depth_scale.py).
SIDE_SKATE = 0.93


def depth_scale(w: Skeleton, ax: WalkAxes) -> float:
    """How much longer the walked distance really is than in the 3D estimate (1 in side views)."""
    if ax.view != "Front" or len(w) < 45:
        return 1.0
    from scipy.signal import savgol_filter
    hips = w.point("Hips") @ ax.forward
    v_root = float(np.median(np.gradient(savgol_filter(hips, 15, 2))))
    back = []
    for s, o in (("Left", "Right"), ("Right", "Left")):
        rel = savgol_filter(w.point(s + "Foot") @ ax.forward - hips, 9, 2)
        v = np.gradient(rel)
        lower = (w.point(s + "ToeBase") @ ax.up) < (w.point(o + "ToeBase") @ ax.up)
        back += (-v[lower & (v < 0) & (np.abs(rel) < 0.6 * np.ptp(rel))]).tolist()
    if len(back) < 20 or v_root <= 0:
        return 1.0
    return float(np.clip(np.median(back) / v_root / SIDE_SKATE, 0.7, 3.0))


def world_axes(w: Skeleton, use_floor: bool = True, lateral_deg: float = 0.0, tilt_deg: float | None = None) -> tuple[WalkAxes, float]:
    """Walking axes with the true vertical. `lateral_deg` corrects camera pitch in side views (or roll in front views) when the
    recording set-up is calibrated; it cannot be seen from the floor line of a single walk, so it is 0 unless supplied.
    `tilt_deg` replaces the estimated floor slope when the camera is known (it is fixed within a recording session)."""
    ax = walking_axes(w)
    if ax.view == "Unknown":
        return ax, 0.0
    slope = 0.0
    if tilt_deg is not None:
        slope = tilt_deg
    elif use_floor:                                  # a walk that is too short in depth makes the floor look too steep
        slope = float(np.degrees(np.arctan(np.tan(np.radians(floor_slope_deg(w, ax))) / depth_scale(w, ax))))
    slope = float(np.clip(slope, -25, 25))
    up = _rot(ax.up, ax.left, -slope) if slope else ax.up
    fwd = np.cross(ax.left, up)
    if lateral_deg:
        up = _rot(up, fwd, lateral_deg)
    left = np.cross(up, fwd)
    return replace(ax, forward=fwd / np.linalg.norm(fwd), left=left / np.linalg.norm(left), up=up / np.linalg.norm(up)), slope


# ---------------------------------------------------------------- virtual camera

@dataclass(frozen=True)
class Camera:
    """Where a study's camera stood, relative to the walker.

    azimuth 0 = in front (walker comes closer), 180 = behind, +90 = at the walker's left side, -90 = at the right side.
    elevation = downward tilt in degrees. `output` "3d" returns camera axes (x image-right, y up, z away) in metres;
    "2d" returns pixels of a pinhole camera that stands `distance_m` from the middle of the walk at `height_m` above the hips.
    """
    azimuth: float = 0.0
    elevation: float = 0.0
    output: str = "3d"
    distance_m: float = 4.0
    height_m: float = 0.0
    focal_px: float = 1000.0
    size: tuple = (1280, 720)
    mirror: bool = False            # flip image left-right (studies that used mirrored video)

    def axes(self):
        a, e = np.radians(self.azimuth), np.radians(self.elevation)
        toward = np.array([np.cos(a), np.sin(a), 0.0])                    # from the walker to the camera, in (forward, left, up)
        right = np.array([-np.sin(a), np.cos(a), 0.0])
        look = -np.cos(e) * toward - np.sin(e) * np.array([0, 0, 1.0])
        return right, np.cross(right, look), look, toward


def look(world: np.ndarray, cam: Camera) -> np.ndarray:
    """(..., 3) in (forward, left, up) -> the camera's view of it."""
    right, up, fwd, toward = cam.axes()
    p = world
    if cam.output == "2d":
        centre = np.nanmean(world.reshape(-1, 3), axis=0)
        pos = centre + toward * cam.distance_m + np.array([0, 0, 1.0]) * cam.height_m
        p = world - pos
    x, y, z = p @ right, p @ up, p @ fwd
    if cam.mirror:
        x = -x
    if cam.output == "3d":
        return np.stack([x, y, z], -1)
    return np.stack([cam.size[0] / 2 + cam.focal_px * x / z, cam.size[1] / 2 - cam.focal_px * y / z], -1)


@dataclass(frozen=True)
class World:
    body: Skeleton              # walking segment after the 2D correction (camera coordinates)
    xyz: np.ndarray             # (T, 77, 3) in (forward, left, up)
    axes: WalkAxes
    floor_slope: float
    info: dict


def build_world(walk: Skeleton, refine: bool = False, use_floor: bool = True, lateral_deg: float = 0.0, tilt_deg: float | None = None) -> World:
    body, info = refine_with_2d(walk) if refine and len(walk) >= 30 else (walk, {})
    ax, slope = world_axes(body, use_floor, lateral_deg, tilt_deg)
    info = {**info, "depth_scale": depth_scale(body, walking_axes(body))}
    xyz = body.xyz @ np.stack([ax.forward, ax.left, ax.up], axis=1)
    return World(body, xyz, ax, slope, info)
