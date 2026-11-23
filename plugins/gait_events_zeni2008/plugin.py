"""Gait events and timing from foot positions: Zeni, Richards, Higginson, Gait & Posture 2008 (coordinate-based heel strike and toe off).

No weights: a published calculation rule. Heel strike is the instant the foot is furthest ahead of the pelvis, toe off when it is
furthest behind. Nova-77 has no heel, so the ankle is used for heel strike and the toe base for toe off. Reports timing only."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from core import jointsets
from core.common import Prepared, in_walk_frame
from core.plugin import Manifest, Plugin, Requires

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("gait_plugin_gait_features_zeni", ROOT / "shared" / "gait_features.py")
features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(features)


class Zeni2008(Plugin):
    manifest = Manifest(
        id="gait_events_zeni2008", version="0.1.0", condition="gait timing (cadence, stride and stance time, left-right asymmetry)",
        output="steps per minute, stride time (s), stance share (%), step-time asymmetry (%); a measurement, not a diagnosis",
        study="Zeni et al., Gait & Posture 2008: coordinate-based gait events", source="calculation rule in shared/gait_features.py (no code from the paper)", license="formula only",
        requires=Requires(joint_set="lower_limb_arms", views=("Side", "Front"), min_frames=45, min_observed=0.5), kind="measurement",
        fidelity="published rule, ankle instead of heel; not validated against force plates on these recordings")

    def adapt(self, p: Prepared):
        names, cam = jointsets.select(p.walk, "lower_limb_arms")
        return features.compute(names, in_walk_frame(cam, p.axes), p.raw.fps), p.axes.view

    def check(self, p: Prepared) -> list:
        reasons = super().check(p)
        if not reasons:
            f, _ = self.adapt(p)
            if min(f["strides_L"], f["strides_R"]) < 1:
                reasons.append("needs_one_full_stride_per_side")
        return reasons

    def predict(self, x) -> dict:
        f, view = x
        num = lambda v, n=2: None if not np.isfinite(v) else round(float(v), n)
        detail = {"cadence_steps_min": num(f["cadence"], 1), "stride_time_s": num(f["stride_time_s"]), "stance_pct": num(f["stance_pct_mean"], 1), "step_time_asym_pct": num(f["step_time_asym"], 1),
                  "strides_measured": int(f["strides_L"] + f["strides_R"])}
        c = detail["cadence_steps_min"]
        return {"label": f"{c:.0f} steps/min" if c is not None else "cadence not measurable", "score": float(f["cadence"]) if np.isfinite(f["cadence"]) else float("nan"), "detail": detail,
                "warnings": [] if view == "Side" else ["front_view_distances_not_used"]}


PLUGIN = Zeni2008
