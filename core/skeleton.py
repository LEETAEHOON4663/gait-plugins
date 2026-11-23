"""Canonical skeleton shared by every plugin.

Nothing is thrown away here: all joints, 2D + 3D, confidence and a per-joint status are kept.
Arrays are read-only, so one plugin cannot change the input another plugin receives.
Study-specific normalisation (centring, scaling, cropping, ...) belongs in the plugin, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

MISSING, OBSERVED, INTERPOLATED = 0, 1, 2


class InputError(ValueError):
    """The input file does not follow the expected format."""


@dataclass(frozen=True)
class Skeleton:
    joints: tuple                 # joint names, length J
    xyz: np.ndarray               # (T, J, 3) camera axes: x image-right, y down, z away; NaN where missing
    xy: np.ndarray                # (T, J, 2) image pixels; NaN where missing
    conf: np.ndarray              # (T, J) extractor score (not a probability); NaN where not observed
    status: np.ndarray            # (T, J) MISSING / OBSERVED / INTERPOLATED
    fps: float
    frame_ids: np.ndarray         # (T,)
    source: dict = field(default_factory=dict)   # file, sha256, extractor
    warnings: tuple = ()

    def __post_init__(self):
        t, j = len(self.frame_ids), len(self.joints)
        for name, shape in (("xyz", (t, j, 3)), ("xy", (t, j, 2)), ("conf", (t, j)), ("status", (t, j))):
            a = getattr(self, name)
            if a.shape != shape:
                raise InputError(f"{name} must have shape {shape}, got {a.shape}")
            a = a.copy()
            a.setflags(write=False)
            object.__setattr__(self, name, a)

    def __len__(self):
        return len(self.frame_ids)

    def index(self, name: str) -> int:
        return self.joints.index(name)

    def point(self, name: str) -> np.ndarray:
        """(T, 3) trajectory of one joint (a writable copy)."""
        return self.xyz[:, self.index(name)].copy()

    def slice(self, start: int, end: int) -> "Skeleton":
        return replace(self, xyz=self.xyz[start:end], xy=self.xy[start:end], conf=self.conf[start:end],
                       status=self.status[start:end], frame_ids=self.frame_ids[start:end])
