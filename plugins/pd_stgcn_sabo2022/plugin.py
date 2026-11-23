"""Parkinsonism severity from 13 joints: Sabo, Mehdizadeh, Iaboni, Taati (IEEE JBHI 2022), Kinect3D model.

Takes only 13 of the 77 joints and applies the study's own preprocessing:
Kinect-like camera axes -> hip centring -> -0.5 offset -> zero padding at the start -> middle 120 frames.
"""
from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path

import numpy as np

from core import jointsets
from core.common import Prepared
from core.plugin import Manifest, Plugin, Requires
from core.view import Camera, look


WINDOW = 120
# The study's camera hung from the ceiling, looking down. The angle is measured on the study's own sample walks
# (scripts/verify_stgcn_port.py prints it), because a hip-centred skeleton still contains this tilt.
TILT_DEG = 30.0
CHECKPOINT = Path(os.environ.get("STGCN_WEIGHTS", "")) if os.environ.get("STGCN_WEIGHTS") else Path(__file__).parent / "weights" / "checkpoint_final.pt"


# In front of the walker, looking down; x = image-right = the walker's left, y up, z away. These are the study's axes.
STUDY_CAMERA = Camera(azimuth=0, elevation=TILT_DEG, output="3d")


def study_preprocess(kinect: np.ndarray, names: list, window: int = WINDOW) -> np.ndarray:
    """(T, 13, 3) metres -> (3, window, 13). Mirrors kinect3D_centre_hip, normalize_by_resolution with resolution 1,
    pad_zero_beginning and crop_middle from the original repository, in the same order."""
    x = kinect.astype(np.float32)
    hip = (x[:, names.index("LHip")] + x[:, names.index("RHip")]) / 2
    x = x - hip[:, None] - 0.5
    if len(x) < window:
        x = np.concatenate([np.zeros((window - len(x), *x.shape[1:]), np.float32), x])
    elif len(x) > window:
        b = int((len(x) - window) / 2)
        x = x[b:b + window]
    return x.transpose(2, 0, 1)


class SaboSTGCN(Plugin):
    manifest = Manifest(
        id="pd_stgcn_sabo2022", version="0.2.0", condition="parkinsonism severity",
        output="predicted MDS-UPDRS gait item, 0 (none) to 2; a severity estimate, not a diagnosis",
        study="Sabo et al., Estimating Parkinsonism Severity in Natural Gait Videos of Older Adults with Dementia, JBHI 2022",
        source="https://github.com/TaatiTeam/stgcn_parkinsonism_prediction (Kinect3D, no gait features)",
        license="UHN, academic non-commercial use only",
        requires=Requires(joint_set="kinect13", views=("Side", "Front"), min_frames=60, min_observed=0.5),
        kind="pretrained",
        fidelity="port equals the original code on the study's sample data; accuracy on Nova-77 input not established; class changes in about a quarter of clips if the study camera angle is off by 5 degrees")

    @cached_property
    def model(self):
        import torch
        import importlib.util
        spec = importlib.util.spec_from_file_location("gait_plugin_stgcn_model", Path(__file__).parent / "model.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        torch.set_num_threads(2)
        return mod.load(CHECKPOINT)

    def adapt(self, p: Prepared) -> np.ndarray:
        names, world = jointsets.select(p.world.body, "kinect13", p.world.xyz)
        return study_preprocess(look(world, STUDY_CAMERA), names)

    def predict(self, x: np.ndarray) -> dict:
        import torch
        with torch.no_grad():
            s = float(self.model(torch.from_numpy(x[None].copy())).item())
        padded = int((x == 0).all(axis=(0, 2)).sum())
        return {"label": f"UPDRS-gait {int(np.clip(round(s), 0, 2))}", "score": s,
                "detail": {"padded_frames": padded, "window": WINDOW},
                "warnings": [f"clip_shorter_than_{WINDOW}_frames_zero_padded"] if padded else []}


PLUGIN = SaboSTGCN
