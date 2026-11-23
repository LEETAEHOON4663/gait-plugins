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
need = pytest.mark.skipif(not (ROOT / "plugins/gait_gaitlab2020/weights/gaitlab.npz").exists(), reason="weights not converted")


def test_gaitlab_loads():
    assert not FAILURES and "gait_gaitlab2020" in BY


@need
def test_side_view_gives_all_seven_published_outputs(tmp_path):
    r = {x.plugin: x for x in run(nova77.load(synth(tmp_path / "a.jsonl", t=300, planted=True)), PLUGINS)[1]}["gait_gaitlab2020"]
    assert r.status == "ok", r.reasons
    assert set(r.detail["outputs"]) == {"GDI", "gmfcs", "speed", "cadence", "SEMLS_dev_residual", "KneeFlex_maxExtension_L", "KneeFlex_maxExtension_R"}
    assert np.isfinite(list(r.detail["outputs"].values())).all()


@need
@pytest.mark.parametrize("kw,reason", [(dict(direction=(0, -1), t=300), "view:Front"), (dict(direction=(1, 0), t=100), "too_short:100<126_frames")])
def test_front_view_and_short_walks_are_refused(tmp_path, kw, reason):
    p = prepare(nova77.load(synth(tmp_path / "a.jsonl", planted=True, **kw)))
    assert any(x.startswith(reason) for x in BY["gait_gaitlab2020"].check(p))


@need
def test_direction_of_walking_does_not_matter(tmp_path):
    """The study normalises the facing direction, so left-to-right and right-to-left of the same walk must agree."""
    g = lambda d: BY["gait_gaitlab2020"].predict(BY["gait_gaitlab2020"].adapt(prepare(nova77.load(synth(tmp_path / "x.jsonl", t=300, direction=d, planted=True)))))["detail"]["outputs"]
    a, b = g((1, 0)), g((-1, 0))
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-3), k


@need
def test_deterministic(tmp_path):
    s = nova77.load(synth(tmp_path / "a.jsonl", t=300, planted=True))
    x = [r.score for r in run(s, PLUGINS)[1]]
    y = [r.score for r in run(s, PLUGINS)[1]]
    assert x == y
