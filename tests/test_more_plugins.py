import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
import pytest
from core import nova77
from core.common import prepare
from core.plugin import discover
from test_core import synth

PLUGINS, FAILURES = discover()
BY = {p.manifest.id: p for p in PLUGINS}
need2d = pytest.mark.skipif(not (ROOT / "plugins/pd_stgcn_sabo2022_dt2d/weights/1.pt").exists(), reason="2D weights not copied")


def test_new_plugins_load():
    assert not FAILURES and {"pd_stgcn_sabo2022_ap2d", "pd_stgcn_sabo2022_dt2d", "gait_events_zeni2008"} <= set(BY)


def test_zeni_counts_steps_of_a_known_gait(tmp_path):
    p = prepare(nova77.load(synth(tmp_path / "a.jsonl", t=300, planted=True)))
    r = BY["gait_events_zeni2008"].predict(BY["gait_events_zeni2008"].adapt(p))
    assert r["detail"]["stride_time_s"] == pytest.approx(1.1, abs=0.06)
    assert r["detail"]["cadence_steps_min"] == pytest.approx(60 * 2 / 1.1, abs=8)
    assert 45 < r["detail"]["stance_pct"] < 75


def test_zeni_asks_for_a_full_stride(tmp_path):
    p = prepare(nova77.load(synth(tmp_path / "a.jsonl", t=50, planted=True)))
    assert any(x.startswith("needs_one_full_stride") or x.startswith("too_short") for x in BY["gait_events_zeni2008"].check(p))


@need2d
@pytest.mark.parametrize("pid", ["pd_stgcn_sabo2022_ap2d", "pd_stgcn_sabo2022_dt2d"])
def test_2d_models_run_and_do_not_depend_on_camera_direction(tmp_path, pid):
    f = lambda d: BY[pid].predict(BY[pid].adapt(prepare(nova77.load(synth(tmp_path / "a.jsonl", t=240, direction=d, planted=True)))))["score"]
    a = f((1, 0))
    assert 0 - 1 < a < 3
    for d in ((-1, 0), (0, -1), (0, 1)):
        assert f(d) == pytest.approx(a, abs=0.02)


@need2d
def test_2d_body_is_scaled_to_the_study_size(tmp_path):
    from shared.sabo2d import TRAIN_BODY_PX, to_study_pixels
    from core import jointsets
    p = prepare(nova77.load(synth(tmp_path / "a.jsonl", t=200, planted=True)))
    px = to_study_pixels(jointsets.select(p.world.body, "coco13_nose", p.world.xyz)[1])
    hip = (px[:, 7] + px[:, 8]) / 2
    assert np.allclose(hip, 100, atol=0.01)
    assert np.median((px[:, 11, 1] + px[:, 12, 1]) / 2 - px[:, 0, 1]) == pytest.approx(TRAIN_BODY_PX, rel=0.02)
