"""Muse Wonder lab report from a compact result summary."""

from __future__ import annotations

import json


def compact_result(result: dict) -> dict:
    """Strip arrays; keep summaries, fits and calibration for the report model."""
    tracks = []
    for tr in result.get("tracks", []) or []:
        t = tr.get("t") or []
        tracks.append({"id": tr.get("id"), "label": tr.get("label"),
                       "n_frames": len(tr.get("frames") or t),
                       "t_range": [t[0], t[-1]] if t else None})
    fits = []
    for f in result.get("fits", []) or []:
        fits.append({"id": f.get("id"), "model": f.get("model"), "track_id": f.get("track_id"),
                     "t_range": f.get("t_range"), "params": f.get("params"),
                     "r2": f.get("r2"), "rmse_px": f.get("rmse_px"),
                     "n_points": f.get("n_points"), "warnings": f.get("warnings"),
                     "segments": [{"index": s.get("index"), "g": s.get("g"), "r2": s.get("r2"),
                                   "excluded_from_pool": s.get("excluded_from_pool")}
                                  for s in (f.get("segments") or [])],
                     "restitution": f.get("restitution")})
    return {"question": result.get("question"), "plan": result.get("plan"),
            "calibration": result.get("calibration"), "tracks": tracks,
            "fits": fits, "answer": result.get("answer"),
            "video": {k: (result.get("video") or {}).get(k) for k in ("fps", "width", "height", "n_frames", "duration_s")}}


def _template_report(result: dict) -> str:
    ans = result.get("answer") or {}
    cal = result.get("calibration") or {}
    lines = [f"## Question\n\n{result.get('question','')}", "## What Physicam saw\n"]
    plan = result.get("plan") or {}
    lines.append(str(plan.get("experiment", "A moving object was tracked through the video.")))
    lines.append("\n## Method\n")
    lines.append(f"Tracked {(result.get('tracks') or [{}])[0].get('label','object')} with SAM 3.1; "
                 f"calibration: {cal.get('method','none')} ({cal.get('details','')}).")
    lines.append("\n## Results\n")
    if ans:
        lines.append(f"| Quantity | Value |\n|---|---|\n| {ans.get('unit','result')} | {ans.get('headline','')} |")
    lines.append("\n## Discussion\n\nSee fit warnings and residuals in the result data.")
    lines.append("\n## How to measure better next time\n\nUse a static camera, a known-size reference in the plane of motion, and higher frame rate.")
    return "\n".join(lines)


def write_report(client, result: dict) -> str:
    """Write a ~300-word lab report with Muse Wonder (fallback: Spark / template)."""
    compact = compact_result(result)
    prompt = (
        "Write a short lab report (~300 words) in markdown with exactly these sections: "
        "Question; What Physicam saw; Method; Results (a markdown table); "
        "Discussion (compare with accepted values where they exist, dominant error sources, "
        "what the residuals say); How to measure better next time. "
        "Use only the numbers in this JSON; never invent numbers.\n\n"
        + json.dumps(compact, indent=1)[:8000])
    models: list[tuple[str, str]] = []
    settings = getattr(client, "settings", None) if client is not None else None
    if settings is not None:
        models = [(getattr(settings, "wonder", "muse-wonder-1.0"), "Wonder"),
                  (getattr(settings, "spark", "muse-spark-1.3"), "Spark")]
    else:
        models = [("muse-wonder-1.0", "Wonder"), ("muse-spark-1.3", "Spark")]
    if client is None:
        return _template_report(result)
    for model, _label in models:
        try:
            body = {"model": model, "max_output_tokens": 2500,
                    "reasoning": {"effort": "low"},
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]}
            # Wonder timeout is handled by the caller's client timeout; keep call simple.
            resp = client.responses(body)
            text = client.output_text(resp) if hasattr(client, "output_text") else ""
            if not text:
                from physicam.meta_client import MetaClient
                text = MetaClient.output_text(resp)
            if text and text.strip():
                return text.strip()
        except Exception:
            continue
    return _template_report(result)
