"""Plugin contract, discovery and runner.

A plugin is a folder plugins/<name>/ with a plugin.py that defines PLUGIN, a Plugin subclass.
Adding a disease = adding a folder. Removing it = deleting the folder. The core never names a disease.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import jointsets
from .common import Prepared, prepare
from .skeleton import OBSERVED, Skeleton

API_VERSION = 1
PLUGIN_DIR = Path(__file__).resolve().parent.parent / "plugins"


@dataclass(frozen=True)
class Requires:
    """What the study needs. The router checks this before the plugin ever runs."""
    joint_set: str
    views: tuple = ("Side", "Front")
    min_frames: int = 30
    min_observed: float = 0.5      # share of required-joint samples that are observed, not interpolated
    fps: tuple = (25.0, 35.0)
    max_outside: float = 0.2       # share of walking frames with a required joint outside the image (only checked if the size is known)


@dataclass(frozen=True)
class Manifest:
    id: str
    version: str
    condition: str
    output: str                    # what the number means, in words
    study: str
    source: str
    license: str
    requires: Requires
    kind: str                      # "pretrained" (study weights) or "retrained" (trained here on public data)
    fidelity: str                  # evidence that the study's performance survives; shown with every result
    api: int = API_VERSION


@dataclass
class Result:
    plugin: str
    condition: str
    status: str                    # ok / ineligible / error
    label: str = ""
    score: float | None = None
    detail: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    fidelity: str = ""

    def to_dict(self):
        return asdict(self)


def outside_share(xy: np.ndarray, size, margin: float = 0.01) -> float:
    """Share of frames in which any of the given joints lies outside the image (1% margin)."""
    w, h = size
    inside = (xy[..., 0] >= -margin * w) & (xy[..., 0] <= (1 + margin) * w) & (xy[..., 1] >= -margin * h) & (xy[..., 1] <= (1 + margin) * h)
    return float((~inside).any(axis=1).mean())


class Plugin(ABC):
    manifest: Manifest

    def check(self, p: Prepared) -> list:
        """Reason codes; an empty list means the plugin can run on this input."""
        r, reasons = self.manifest.requires, []
        missing = jointsets.resolve(r.joint_set, p.raw.joints)["unavailable"]
        if missing:
            reasons.append("joints_unavailable:" + ",".join(missing))
        if p.axes.view not in r.views:
            reasons.append(f"view:{p.axes.view}_needs_{'/'.join(r.views)}")
        if len(p.walk) < r.min_frames:
            reasons.append(f"too_short:{len(p.walk)}<{r.min_frames}_frames")
        elif not missing:
            cols = [p.walk.index(j) for j in jointsets.canonical_joints(r.joint_set)]
            seen = float((p.walk.status[:, cols] == OBSERVED).mean())
            if seen < r.min_observed:
                reasons.append(f"too_few_observed:{seen:.2f}<{r.min_observed}")
            size = p.raw.source.get("size")
            if size:
                out = outside_share(p.walk.xy[:, cols], size)
                if out > r.max_outside:
                    reasons.append(f"body_out_of_frame:{out:.0%}>{r.max_outside:.0%}")
        if p.axes.view != "Unknown" and not p.axes.left_right_ok:
            reasons.append("left_right_contradicts_walking_direction")   # walking axes are wrong, or the sides are swapped
        if not r.fps[0] <= p.raw.fps <= r.fps[1]:
            reasons.append(f"fps:{p.raw.fps}")
        return reasons

    @abstractmethod
    def adapt(self, p: Prepared):
        """Take only what the study needs and apply the study's own normalisation."""

    @abstractmethod
    def predict(self, model_input) -> dict:
        """Run the study's model. Return {"label", "score", "detail", "warnings"}."""


def discover(folder: Path | str = PLUGIN_DIR) -> tuple[list, dict]:
    """Load every plugin folder. A broken plugin is reported and skipped; it never stops the others."""
    plugins, failures = [], {}
    for d in sorted(Path(folder).iterdir()) if Path(folder).is_dir() else []:
        f = d / "plugin.py"
        if not f.is_file():
            continue
        name = f"gait_plugin_{d.name}"
        try:
            spec = importlib.util.spec_from_file_location(name, f, submodule_search_locations=[str(d)])
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            plugin = mod.PLUGIN()
            if plugin.manifest.api != API_VERSION:
                raise RuntimeError(f"plugin API {plugin.manifest.api}, core API {API_VERSION}")
            if plugin.manifest.id in {p.manifest.id for p in plugins}:
                raise RuntimeError(f"duplicate id {plugin.manifest.id}")
            plugins.append(plugin)
        except Exception as e:
            sys.modules.pop(name, None)
            failures[d.name] = f"{type(e).__name__}: {e}"
    return plugins, failures


def run_one(plugin: Plugin, p: Prepared) -> Result:
    m = plugin.manifest
    res = Result(m.id, m.condition, "ok", fidelity=m.fidelity, warnings=list(p.raw.warnings))
    try:
        res.reasons = plugin.check(p)
        if res.reasons:
            res.status = "ineligible"
            return res
        out = plugin.predict(plugin.adapt(p))
        res.label, res.score, res.detail = out["label"], float(out["score"]), out.get("detail", {})
        res.warnings += out.get("warnings", [])
        if not np.isfinite(res.score):
            raise ValueError("score is not finite")
    except Exception as e:                           # isolate: one plugin must not stop the others
        res.status, res.label, res.score = "error", "", None
        res.reasons = [f"{type(e).__name__}: {e}", traceback.format_exc(limit=2)]
    return res


def run(skeleton: Skeleton, plugins) -> tuple[Prepared, list]:
    p = prepare(skeleton)
    return p, [run_one(pl, p) for pl in plugins]
