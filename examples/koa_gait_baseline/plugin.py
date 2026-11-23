"""Knee osteoarthritis vs (normal + Parkinson's), logistic regression on gait features.

No published skeleton model for knee OA with public weights was found, so this one is trained here on KOA-PD-NM
(train.py). It is our own baseline, not a reproduction of a published study.
"""
from __future__ import annotations

import importlib.util
import json
from functools import cached_property
from pathlib import Path

import numpy as np

from core import jointsets
from core.common import Prepared, in_walk_frame
from core.plugin import Manifest, Plugin, Requires

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("gait_plugin_koa_features", HERE.parent.parent / "shared" / "gait_features.py")
features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(features)


class KOABaseline(Plugin):
    manifest = Manifest(
        id="koa_gait_baseline", version="0.1.0", condition="knee osteoarthritis",
        output="score 0-1 for knee-OA-like gait against normal and Parkinson's gait; screening demo, not a diagnosis",
        study="own baseline on KOA-PD-NM (Mendeley 44pfnysy89), not a published model",
        source="examples/koa_gait_baseline/train.py", license="project code; dataset license not checked, see its Mendeley page",
        requires=Requires(joint_set="lower_limb_arms", views=("Side",), min_frames=45, min_observed=0.5),
        kind="retrained", fidelity="see model.json: subject-level cross-validation")

    @cached_property
    def model(self) -> dict:
        return json.loads((HERE / "model.json").read_text())

    def adapt(self, p: Prepared) -> dict:
        names, cam = jointsets.select(p.walk, self.manifest.requires.joint_set)
        return features.compute(names, in_walk_frame(cam, p.axes), p.raw.fps)

    def check(self, p: Prepared) -> list:
        reasons = super().check(p)
        if not reasons:
            f = self.adapt(p)
            if min(f["strides_L"], f["strides_R"]) < 1:
                reasons.append("needs_one_full_stride_per_side")
        return reasons

    def predict(self, f: dict) -> dict:
        m = self.model
        raw = np.array([f[k] for k in m["features"]], float)
        x = np.where(np.isfinite(raw), raw, m["median"])
        z = (x - np.array(m["mean"])) / np.array(m["scale"])
        c = z * np.array(m["coef"])
        s = float(1 / (1 + np.exp(-(c.sum() + m["intercept"]))))
        top = sorted(zip(m["features"], raw, c), key=lambda t: -abs(t[2]))[:3]
        imputed = [k for k, v in zip(m["features"], raw) if not np.isfinite(v)]
        return {"label": "positive" if s >= m["positive"] else "negative" if s <= m["negative"] else "indeterminate", "score": s,
                "detail": {"evidence": [{"feature": k, "value": round(float(v), 2), "push": round(float(p), 2)} for k, v, p in top]},
                "warnings": [f"imputed:{','.join(imputed)}"] if imputed else []}


PLUGIN = KOABaseline
