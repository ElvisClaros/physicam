"""Contract tests for task 04 integration fixes."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from physicam.calibrate import Calibration
from physicam.physics import fit_model
from physicam.meta_client import MetaClient
from physicam.sam import Track

FIXTURE = Path(__file__).parent / "fixtures" / "bounce240_red_ball_f0-199.sam.txt.gz"


def make_track(t, x, y, fps=60.0):
    t = np.asarray(t, float)
    n = len(t)
    area = np.full(n, 2000.0)
    return Track(id="t0", label="ball", frames=np.round(t * fps).astype(int), t=t,
                 x=np.asarray(x, float), y=np.asarray(y, float), area=area,
                 bbox=np.zeros((n, 4)), conf=np.full(n, 0.9),
                 eq_diameter=np.sqrt(4 * area / np.pi), contours={}, fps=fps)


def test_pendulum_curve_both_axes_info_and_no_duplicate_period():
    fps, T_true = 60.0, 1.6
    omega = 2 * np.pi / T_true
    t = np.arange(0, 600) / fps
    x = 500 + 200 * np.exp(-0.02 * t) * np.cos(omega * t)
    y = 400 + 20 * np.exp(-0.02 * t) * np.cos(omega * t)
    fit = fit_model(make_track(t, x, y, fps), "pendulum", Calibration(0.001, 0.0, "reference", ""))
    d = fit.to_dict()
    assert "period" not in d["params"] and "T" in d["params"]
    assert "axis" not in d["params"]
    assert d["info"]["axis"] in ("x", "y")
    assert d["curve"]["axis"] == d["info"]["axis"]
    assert set(("t", "x", "y")) <= set(d["curve"])
    assert len(d["curve"]["x"]) == len(d["curve"]["y"]) == len(d["curve"]["t"])
    assert d["params"]["T"]["stderr"] > 0
    assert abs(d["params"]["T"]["value"] - T_true) / T_true < 0.01


def test_bounce_impacts_shape_and_segment_curves():
    from test_physics import _bounce_trajectory
    t, x, y = _bounce_trajectory()
    fit = fit_model(make_track(t, x, y, fps=240.0), "bounce",
                    Calibration(9.81 / 8000.0, 0.0, "reference", ""))
    d = fit.to_dict()
    assert d["impacts"], "bounce must emit impacts"
    for im in d["impacts"]:
        assert set(im) == {"t", "x", "y", "e", "e_stderr", "e_height", "valid"}
        assert isinstance(im["valid"], bool)
    valid = [im for im in d["impacts"] if im["valid"]]
    assert valid and all(abs(im["e"] - 0.8) < 0.08 for im in valid)
    for seg in d["segments"]:
        if "curve" in seg:
            assert set(("t", "x", "y")) <= set(seg["curve"])


def test_simplify_sam_prompt():
    from physicam.agent import simplify_sam_prompt
    assert simplify_sam_prompt("red ball") == ("red ball", False)
    clean, changed = simplify_sam_prompt(
        "small shiny spherical metal weight bob at free end of coil spring")
    assert changed and len(clean.split()) <= 3


def test_make_trace_title_examples():
    from physicam.agent import make_trace_title
    title = make_trace_title("track_objects", {"prompt": "metal ball"},
                             {"tracks": [{"label": "metal ball", "n_frames": 491, "t_range": [0.43, 16.8]}]})
    assert "metal ball" in title and "491 frames" in title
    assert make_trace_title("track_objects", {"prompt": "silver metal weight"}, {"tracks": []}).startswith(
        "SAM 3.1 found nothing")
    fit_title = make_trace_title("fit_model", {"model": "pendulum"},
                                 {"params": {"T": {"value": 0.9633, "stderr": 0.0003, "unit": "s"}}, "r2": 0.965})
    assert "T = 0.9633" in fit_title and "R" in fit_title
    assert make_trace_title("inspect_frames", {"times_s": [0, 1, 2]}, {"text": "A classroom scene"}).startswith(
        "Looked at 3 frames:")
    assert make_trace_title("run_python", {}, {"result": {"mean_T": 0.963}}).startswith("Python:")


class _CountingSamClient:
    def __init__(self, data_dir):
        self.settings = SimpleNamespace(sam="sam-3.1", spark="muse-spark-1.3", data_dir=str(data_dir))
        self.prompts: list[str] = []

    def upload_file(self, path):
        return "file-fake"

    def responses(self, body):
        text_in = body["input"][0]["content"][0]["text"]
        self.prompts.append(text_in)
        text = gzip.open(FIXTURE, "rt").read()
        return {"id": "r", "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
                "usage": {"video_frames_processed": 200}}

    output_text = staticmethod(MetaClient.output_text)


def test_concurrent_multi_prompt_tracking(tmp_path):
    from physicam.agent import AgentSession
    from physicam.plan import Plan
    client = _CountingSamClient(tmp_path)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    session = AgentSession(client, video, Plan(), "q", {}, fps=240.0)
    out = session.tool_track_objects(prompts=["red ball", "metal ball"])
    assert sorted(client.prompts) == ["metal ball", "red ball"]
    assert out["tracks"], out
    # Same fixture for both prompts = same physical object: cross-prompt
    # dedup keeps one track and records the other prompt as an alias.
    assert len(out["tracks"]) == 1, out["tracks"]
    assert len(session.tracks) == 1
    assert out["tracks"][0]["aliases"], out["tracks"]


def test_run_python_parses_json_stdout(tmp_path):
    from physicam.agent import AgentSession
    from physicam.plan import Plan
    session = AgentSession(_CountingSamClient(tmp_path), tmp_path / "c.mp4", Plan(), "q", {}, fps=30.0)
    out = session.tool_run_python("import json; print(json.dumps({'mean_T': 0.963}))")
    assert out["result"] == {"mean_T": 0.963}
