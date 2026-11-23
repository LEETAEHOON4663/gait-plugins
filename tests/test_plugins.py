import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
import pytest
from core import nova77
from core.common import prepare
from core.plugin import discover, run
from test_core import synth

PLUGINS, FAILURES = discover()
BY = {p.manifest.id: p for p in PLUGINS}
need_weights = pytest.mark.skipif(not (ROOT / "plugins/pd_stgcn_sabo2022/weights/checkpoint_final.pt").exists(), reason="weights not downloaded")


def test_shipped_plugins_load():
    assert not FAILURES and {"pd_stgcn_sabo2022", "gait_gaitlab2020", "gait_events_zeni2008"} <= set(BY) and "koa_gait_baseline" not in BY      # our own baseline is an example, not a study


def test_stgcn_input_is_hip_centred_with_the_study_offset(tmp_path):
    x = BY["pd_stgcn_sabo2022"].adapt(prepare(nova77.load(synth(tmp_path / "a.jsonl"))))
    assert x.shape == (3, 120, 13)
    assert np.allclose((x[:, :, 7] + x[:, :, 8]) / 2, -0.5, atol=1e-5)


def test_stgcn_input_does_not_depend_on_camera_view(tmp_path):
    a, b = (BY["pd_stgcn_sabo2022"].adapt(prepare(nova77.load(synth(tmp_path / f"{i}.jsonl", direction=d)))) for i, d in enumerate([(1, 0), (0, -1)]))
    assert np.allclose(a, b, atol=1e-4)


def test_stgcn_short_clip_is_zero_padded_at_the_start(tmp_path):
    x = BY["pd_stgcn_sabo2022"].adapt(prepare(nova77.load(synth(tmp_path / "a.jsonl", t=80))))
    assert (x[:, :40] == 0).all() and (x[:, 40:] != 0).any()


def test_gait_features_on_a_known_gait(tmp_path):
    from core import jointsets
    from core.common import in_walk_frame
    from shared import gait_features
    p = prepare(nova77.load(synth(tmp_path / "a.jsonl", t=240)))
    names, cam = jointsets.select(p.walk, "lower_limb_arms")
    f = gait_features.compute(names, in_walk_frame(cam, p.axes), p.raw.fps)
    assert f["stride_time_s"] == pytest.approx(1.1, abs=0.04) and f["cadence"] == pytest.approx(109, abs=4)
    swing = np.sin(2 * np.pi * np.arange(240) / 30 / 1.1)                # the synthetic leg stretches a little while swinging
    leg = np.median(np.hypot(0.42, 0.15 * swing) + np.hypot(0.42, 0.17 * swing))
    assert f["speed_leg_s"] == pytest.approx(1 / leg, rel=0.02)
    assert f["knee_rom_asym"] < 5 and f["step_time_asym"] < 10


@need_weights
def test_plugins_run_on_one_skeleton_and_are_deterministic(tmp_path):
    s = nova77.load(synth(tmp_path / "a.jsonl", t=240))
    r1, r2 = (run(s, PLUGINS)[1] for _ in range(2))
    assert {r.plugin: r.status for r in r1}["pd_stgcn_sabo2022"] == "ok", [(r.plugin, r.reasons) for r in r1]
    assert [r.score for r in r1] == [r.score for r in r2]


@need_weights
def test_front_view_runs_only_the_plugin_that_allows_it(tmp_path):
    r = {x.plugin: x.status for x in run(nova77.load(synth(tmp_path / "a.jsonl", t=240, direction=(0, -1))), PLUGINS)[1]}
    assert r["pd_stgcn_sabo2022"] == "ok" and r["gait_gaitlab2020"] == "ineligible"                  # a side-only plugin is refused in front view
