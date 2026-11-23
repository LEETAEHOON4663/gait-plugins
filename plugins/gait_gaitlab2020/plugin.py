"""Gait parameters from a side view: Kidzinski et al., Nature Communications 2020 (mobile-gaitlab), weights as published.

Detects what the paper detects - speed, cadence, knee flexion and the gait deviation index - not a disease. The model saw only 2D side
views of one walking person, so the 3D skeleton is turned to the side of the walker. The authors' demo feeds the frames of a 29.97
frames/s video as they are (their FPS = 25 constant is unused), so our 30 frames/s input is used without resampling.
"""
from __future__ import annotations

import importlib.util
import os
from functools import cached_property
from pathlib import Path

import numpy as np

from core import jointsets
from core.common import Prepared
from core.plugin import Manifest, Plugin, Requires
from core.view import Camera, look

HERE = Path(__file__).parent
WEIGHTS = Path(os.environ.get("GAITLAB_WEIGHTS") or HERE / "weights" / "gaitlab.npz")
_spec = importlib.util.spec_from_file_location("gait_plugin_gaitlab_model", HERE / "model.py")
model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model)

ORDER = ["RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "LBigToe", "RBigToe"]
SLOT = {"RHip": model.RHIP, "RKnee": model.RKNE, "RAnkle": model.RANK, "LHip": model.LHIP, "LKnee": model.LKNE, "LAnkle": model.LANK, "LBigToe": model.LBTO, "RBigToe": model.RBTO}
SIDE_CAMERA = Camera(azimuth=90, output="3d")              # image x = along the path, y up; the paper's model is blind to mirroring


def to_body25(world: np.ndarray, names: list, fps: float) -> np.ndarray:
    """(T, K, 3) in (forward, left, up) at the recording's fps -> (T, 25, 2) image-like pixels."""
    v = look(world, SIDE_CAMERA)[..., :2] * np.array([1.0, -1.0])            # image y points down
    out = np.full((len(v), 25, 2), np.nan)
    for k, name in enumerate(names):
        out[:, SLOT[name]] = v[:, k]
    # Arbitrary pixel scale; the model normalises position and size away. The offset keeps every coordinate away from exactly 0,
    # which the authors' code treats as a missing joint.
    return out * 1000.0 + 10000.0


class GaitLab2020(Plugin):
    manifest = Manifest(
        id="gait_gaitlab2020", version="0.2.0", condition="gait parameters (speed, cadence, knee flexion, GDI)",
        output="GDI (about 0-100; 100 = typical), walking speed, cadence, knee flexion; estimates, not a diagnosis",
        study="Kidzinski et al., Deep neural networks enable quantitative movement analysis using single-camera videos, Nat Commun 2020",
        source="https://github.com/stanfordnmbl/mobile-gaitlab (demo models)", license="Apache-2.0 code; training data CC BY-NC; a patent is pending (commercial use)",
        requires=Requires(joint_set="gaitlab_lower", views=("Side",), min_frames=126, min_observed=0.5, fps=(29.0, 31.0)),
        kind="pretrained",
        fidelity="port equals the authors' published demo output (7 values, 2.5e-6); trained mostly on children with cerebral palsy; cadence agreement on our data: see results/gaitlab.json")

    @cached_property
    def net(self):
        return model.GaitLab(WEIGHTS)

    def adapt(self, p: Prepared) -> np.ndarray:
        names, w = jointsets.select(p.world.body, "gaitlab_lower", p.world.xyz)
        return to_body25(w, names, p.raw.fps)

    def predict(self, xy: np.ndarray) -> dict:
        out = model.predict_all(self.net, xy)
        lab = f"GDI {out['GDI']:.0f}, speed {out['speed']:.2f} m/s, cadence {out['cadence']:.2f}, knee flexion L/R {out['KneeFlex_maxExtension_L']:.0f}/{out['KneeFlex_maxExtension_R']:.0f} deg"
        return {"label": lab, "score": out["GDI"], "detail": {"outputs": {k: round(v, 4) for k, v in out.items()}}, "warnings": []}


PLUGIN = GaitLab2020
