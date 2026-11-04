"""Report number-rounding and rereport CLI tests."""

from __future__ import annotations

import json

from physicam.report import _round_long_decimals, compact_result, write_report


def _result():
    return {
        "question": "Period?",
        "tracks": [{"id": "t1", "label": "metal ball", "frames": [0, 1],
                    "t": [0.43333333333333335, 16.7666666667]}],
        "fits": [{"id": "f0", "model": "pendulum", "track_id": "t1",
                  "t_range": [0.43333333333333335, 16.7666666667],
                  "params": {"T": {"value": 0.9648424115937253,
                                   "stderr": 0.005818123, "unit": "s"}},
                  "info": {"T_halves": [0.9588938250023455, 0.9696460330697484],
                           "chi2_red": 5.421296486791168},
                  "r2": 0.99912345, "rmse_px": 0.7123456, "n_points": 10,
                  "warnings": []}],
        "answer": {"value": 0.9648424115937253, "uncertainty": 0.005818123,
                   "unit": "s", "headline": "T"},
        "video": {"fps": 30.0, "duration_s": 16.8000001},
    }


def test_compact_result_rounds_to_uncertainty_and_times():
    compact = compact_result(_result())
    fit = compact["fits"][0]
    assert fit["params"]["T"] == {"value": 0.9648, "stderr": 0.0058, "unit": "s"}
    assert fit["t_range"] == [0.433, 16.767]
    assert fit["info"]["T_halves"] == [0.9589, 0.9696]
    assert fit["info"]["chi2_red"] == 5.421
    assert compact["tracks"][0]["t_range"] == [0.433, 16.767]
    assert compact["answer"]["value"] == 0.9648
    assert compact["answer"]["uncertainty"] == 0.0058
    dumped = json.dumps(compact)
    assert "0.433333" not in dumped
    assert "0.964842" not in dumped


def test_round_long_decimals_postprocess():
    text = "from 0.43333333333333335 s, T = 0.9648424115937253 ± 0.0058181234, at 16.766667 s"
    out = _round_long_decimals(text)
    assert "0.4333" in out
    assert "0.9648" in out
    assert "0.005818" in out
    assert "16.766667" in out  # exactly 6 decimal digits: untouched


def test_write_report_prompt_and_postprocess_with_fake_client():
    captured = {}

    class FakeClient:
        settings = None

        def responses(self, body):
            captured["body"] = body
            return {"output": []}

        @staticmethod
        def output_text(resp):
            return "T = 0.9648424115937253 s from 0.43333333333333335 s"

    md = write_report(FakeClient(), _result())
    prompt = captured["body"]["input"][0]["content"][0]["text"]
    assert "never more than 4 significant digits" in prompt
    assert "no raw floats" in prompt
    assert "0.964842" not in prompt
    assert "0.9648" in md
    assert "0.4333" in md


def test_rereport_cli_rewrites_result_and_report(tmp_path, monkeypatch):
    import physicam.meta_client as meta_client_mod
    from physicam.cli import main

    job = tmp_path / "job1"
    job.mkdir()
    result = _result()
    result.update({"id": "job1", "status": "done", "report_markdown": "old"})
    (job / "result.json").write_text(json.dumps(result))

    def _no_client(*a, **k):
        raise RuntimeError("no key in test")

    monkeypatch.setattr(meta_client_mod, "MetaClient", _no_client)
    assert main(["rereport", str(job)]) == 0
    rewritten = json.loads((job / "result.json").read_text())
    assert rewritten["report_markdown"]
    assert rewritten["report_markdown"] != "old"
    assert (job / "report.md").read_text() == rewritten["report_markdown"]
