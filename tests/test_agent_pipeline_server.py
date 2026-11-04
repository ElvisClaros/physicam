"""Agent, fallback, sandbox, pipeline and server tests (no network)."""

from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from physicam.meta_client import MetaClient

FIXTURE = Path(__file__).parent / "fixtures" / "bounce240_red_ball_f0-199.sam.txt.gz"

CANNED_PLAN = {
    "experiment": "Red ball bouncing; measure restitution and g",
    "phenomenon": "bounce",
    "moving_objects": [{"sam_prompt": "red ball", "role": "moving",
                        "description": "red bouncing ball", "standard_size_m": 0.08}],
    "reference_candidates": [],
    "spoken_facts": "",
    "time_window_s": None,
    "slow_motion": False,
    "capture_fps_guess": None,
    "camera_static": True,
    "quality_issues": [],
    "recommended_model": "bounce",
    "target_quantity": {"quantity": "coefficient of restitution", "unit": ""},
}

REPORT_MD = "## Question\n\nq\n\n## Results\n\n| Quantity | Value |\n|---|---|\n| e | 0.8 |"


def _text_resp(text: str, rid: str = "resp-1") -> dict:
    return {"id": rid, "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}


def _call_resp(name: str, args: dict, rid: str) -> dict:
    return {"id": rid, "output": [{"type": "function_call", "name": name,
             "arguments": json.dumps(args), "call_id": f"call-{rid}"}],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}


class FakeClient:
    """Canned plan + SAM fixture + tool-call sequence track->calibrate->fit->finish."""

    def __init__(self, data_dir: str):
        self.settings = SimpleNamespace(sam="sam-3.1", spark="muse-spark-1.3",
                                        wonder="muse-wonder-1.0", asr="muse-voice-transcribe-1.0",
                                        data_dir=data_dir)
        from physicam.meta_client import CostAccumulator
        self.cost = CostAccumulator()
        self._agent_step = 0
        self.seen_tools: list[str] = []

    def upload_file(self, path):
        return "file-fake"

    def responses(self, body: dict) -> dict:
        model = body.get("model", "")
        if model == "sam-3.1":
            content = body["input"][0]["content"]
            if any(c.get("type") == "input_image" for c in content):
                return _text_resp("")
            text = gzip.open(FIXTURE, "rt").read()
            out = _text_resp(text)
            out["usage"] = {"video_frames_processed": 200}
            self.cost.add_sam_frames(200)
            return out
        if (body.get("text") or {}).get("format", {}).get("type") == "json_schema":
            return _text_resp(json.dumps(CANNED_PLAN))
        if body.get("tools"):
            steps = [
                ("track_objects", {"prompt": "red ball"}),
                ("calibrate_from_track", {"track_id": "t0", "real_diameter_m": 0.08}),
                ("fit_model", {"track_id": "t0", "model": "bounce"}),
                ("finish", {"headline": "e = 0.80 ± 0.05", "value": 0.8, "unit": "",
                            "uncertainty": 0.05, "summary": "From fitted bounce segments.",
                            "fit_ids": ["f0"]}),
            ]
            name, args = steps[min(self._agent_step, len(steps) - 1)]
            self.seen_tools.append(name)
            self._agent_step += 1
            return _call_resp(name, args, f"resp-agent-{self._agent_step}")
        # Report (Wonder/Spark without tools).
        return _text_resp(REPORT_MD)

    output_text = staticmethod(MetaClient.output_text)
    function_calls = staticmethod(MetaClient.function_calls)


class GarbageClient(FakeClient):
    """Plan/SAM work, but the agent model returns garbage (no tool calls)."""

    def responses(self, body: dict) -> dict:
        if body.get("tools"):
            return _text_resp("I cannot help with that.")
        return super().responses(body)


# -- agent -------------------------------------------------------------------

def test_agent_loop_executes_tools_and_answers(tmp_path):
    from physicam.agent import run_agent
    from physicam.plan import Plan
    plan = Plan.from_dict(CANNED_PLAN)
    client = FakeClient(str(tmp_path))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    result = run_agent(client, video, plan, "What is the coefficient of restitution?",
                       {}, fps=240.0, width=1280, height=720, emit=lambda ev: None)
    assert result.answer is not None
    assert result.answer["value"] == 0.8
    assert client.seen_tools == ["track_objects", "calibrate_from_track", "fit_model", "finish"]
    assert result.tracks and result.fits
    assert result.calibration is not None and result.calibration.method == "track_size"


def test_fallback_when_model_returns_garbage(tmp_path):
    from physicam.agent import run_agent
    from physicam.plan import Plan
    plan = Plan.from_dict(CANNED_PLAN)
    client = GarbageClient(str(tmp_path))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    result = run_agent(client, video, plan, "What is the coefficient of restitution?",
                       {}, fps=240.0, width=1280, height=720, emit=lambda ev: None)
    assert result.used_fallback
    assert result.answer is not None and result.answer["headline"]
    assert result.fits, "fallback must fit the recommended model"


# -- sandbox -----------------------------------------------------------------

def test_sandbox_blocks_env_keys_and_returns_json(monkeypatch):
    from physicam.sandbox import run_python
    monkeypatch.setenv("MODEL_API_KEY", "should-not-leak")
    monkeypatch.setenv("META_API_KEY", "should-not-leak")
    tracks = {"t0": {"t": np.arange(3, dtype=float), "x": np.ones(3), "y": np.ones(3) * 2, "area": np.ones(3)}}
    out = run_python("import os, json; print(json.dumps({'leak': os.environ.get('MODEL_API_KEY'), 'n': len(tracks['t0']['t']), 'm': m_per_px}))",
                     tracks, m_per_px=0.001)
    assert out["ok"], out
    payload = json.loads(out["stdout"])
    assert payload["leak"] is None
    assert payload["n"] == 3 and payload["m"] == 0.001


def test_sandbox_times_out():
    from physicam.sandbox import run_python
    out = run_python("import time; time.sleep(30)", {}, timeout=2.0)
    assert not out["ok"]
    assert "timeout" in out["error"]


# -- per-frame calibration -----------------------------------------------------

def test_per_frame_calibration_correction():
    from physicam.calibrate import apparent_size_variation, from_track_size
    from physicam.physics import fit_model
    from physicam.sam import Track
    fps = 240.0
    n = 200
    t = np.arange(n) / fps
    # Apparent diameter grows 40 -> 50 px (approaching camera).
    d = np.linspace(40, 50, n)
    area = np.pi * (d / 2) ** 2
    # True metric motion: Y_m = 0.5*g*t^2 about cy, X constant offset.
    cx, cy = 640.0, 360.0
    D = 0.08
    s_true = D / d
    Y_m = 0.1 + 0.5 * 9.81 * t ** 2 * 0.1  # scaled-down fall in metres (short clip)
    X_m = np.full(n, 0.05)
    x = cx + X_m / s_true
    y = cy + Y_m / s_true
    bbox = np.column_stack([x - d / 2, y - d / 2, x + d / 2, y + d / 2])
    tr = Track(id="t0", label="ball", frames=np.arange(n), t=t, x=x, y=y, area=area,
               bbox=bbox, conf=np.ones(n), eq_diameter=d.copy(), contours={}, fps=fps)
    assert apparent_size_variation(tr) > 0.03
    cal_pf = from_track_size(tr, D, per_frame=True, frame_size=(1280, 720))
    assert cal_pf.kind == "per_frame"
    assert cal_pf.cx == 640.0 and cal_pf.cy == 360.0
    dr = cal_pf.depth_ratio(t)
    assert dr[0] > dr[-1]  # approaching camera: depth decreases
    cal_const = from_track_size(tr, D, per_frame=False)
    assert cal_const.kind == "constant"
    fit_pf = fit_model(tr, "free_fall", cal_pf)
    fit_c = fit_model(tr, "free_fall", cal_const)
    g_pf = fit_pf.params["g"]["value"]
    # Per-frame recovers the true (scaled) acceleration better than constant.
    g_true = 9.81 * 0.1
    assert abs(g_pf - g_true) < abs(fit_c.params["g"]["value"] - g_true)


# -- pipeline ------------------------------------------------------------------

@pytest.mark.skipif(not Path("web/mock/clip.mp4").exists(), reason="demo clip missing")
def test_pipeline_writes_contract_result(tmp_path):
    from physicam.pipeline import run_job
    client = FakeClient(str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    job_dir = tmp_path / "job1"
    events: list[dict] = []
    result = run_job(job_dir, "web/mock/clip.mp4",
                     "What is the coefficient of restitution?", {}, emit=events.append, client=client)
    assert result["status"] == "done", result.get("error")
    on_disk = json.loads((job_dir / "result.json").read_text())
    for key in ("id", "status", "question", "video", "plan", "calibration", "tracks", "fits",
                "answer", "report_markdown", "trace", "cost"):
        assert key in on_disk, key
    assert on_disk["tracks"] and on_disk["tracks"][0]["color"]
    assert on_disk["tracks"][0]["contours"] is not None
    assert on_disk["fits"] and on_disk["answer"]
    assert on_disk["video"]["fps"] == 240.0
    types = [e["type"] for e in events]
    assert "stage" in types and "done" in types
    assert (job_dir / "events.jsonl").exists()
    assert '"done"' in (job_dir / "events.jsonl").read_text()


# -- server --------------------------------------------------------------------

def _seed_finished_job(tmp_path, monkeypatch):
    """Create a finished job under a temp data dir and point the app at it."""
    jobs = tmp_path / "jobs"
    job_dir = jobs / "testjob1"
    job_dir.mkdir(parents=True)
    src = Path("web/mock/clip.mp4")
    (job_dir / "video.mp4").write_bytes(src.read_bytes() if src.exists() else b"\x00" * 1000)
    result = {"id": "testjob1", "status": "done", "question": "q", "video": {"url": "/media/testjob1/video.mp4"},
              "plan": None, "calibration": None, "tracks": [], "fits": [], "answer": None,
              "report_markdown": "", "trace": [], "cost": {}}
    (job_dir / "result.json").write_text(json.dumps(result))
    (job_dir / "events.jsonl").write_text(
        json.dumps({"type": "stage", "stage": "ingest"}) + "\n" + json.dumps({"type": "done"}) + "\n")
    monkeypatch.setenv("PHYSICAM_DATA_DIR", str(tmp_path))
    return job_dir


def test_server_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    _seed_finished_job(tmp_path, monkeypatch)
    # examples file
    ex_dir = tmp_path / "exseed"
    # Server reads examples/examples.json relative to cwd; only assert shape when present.
    from physicam.server import create_app
    tc = TestClient(create_app())
    assert tc.get("/api/health").json() == {"status": "ok"}
    r = tc.get("/api/jobs/testjob1")
    assert r.status_code == 200 and r.json()["status"] == "done"
    r = tc.get("/api/jobs/nope")
    assert r.status_code == 404
    # SSE replay of a finished job must terminate.
    r = tc.get("/api/jobs/testjob1/events")
    assert r.status_code == 200
    assert '"done"' in r.text
    # Range support.
    r = tc.get("/media/testjob1/video.mp4", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206, r.status_code
    assert len(r.content) == 100
    assert r.headers.get("content-range", "").startswith("bytes 0-99/")
    # Examples endpoint returns a list.
    r = tc.get("/api/examples")
    assert r.status_code == 200 and isinstance(r.json(), list)
    # Static index.
    r = tc.get("/")
    assert r.status_code == 200 and "Physicam" in r.text
