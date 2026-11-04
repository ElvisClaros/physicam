"""Muse Wonder lab report from a compact result summary."""

from __future__ import annotations

import json
import math
import re

_TIME_KEYS = {"t", "t_range", "t0", "t1", "time", "time_s", "duration_s",
              "t_start", "t_end", "start", "end", "timestamp"}


def _round_sig(value: float, sig: int = 4) -> float:
    """Round *value* to *sig* significant digits."""
    if value == 0 or not math.isfinite(value):
        return float(value)
    exp = math.floor(math.log10(abs(value)))
    factor = 10 ** (sig - 1 - exp)
    return round(value * factor) / factor


def _round_pair(value, stderr) -> tuple[float, float]:
    """Round (value, stderr): 2 sig digits on stderr, value to its decimal place."""
    try:
        v, u = float(value), float(stderr)
    except (TypeError, ValueError):
        return value, stderr
    if not math.isfinite(u) or u == 0:
        return _round_sig(v, 4), stderr
    if not math.isfinite(v):
        return v, _round_sig(u, 2)
    places = 1 - math.floor(math.log10(abs(u)))  # decimal places of 2nd sig digit
    return round(v, places), round(u, places)


def _round_floats(obj, key: str | None = None):
    """Recursively round floats: value/stderr pairs jointly, times to 3 dp, else 4 sig digits."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, dict):
        out = dict(obj)
        # value ± stderr / uncertainty pairs are rounded together.
        unc_key = "stderr" if "stderr" in out else ("uncertainty" if "uncertainty" in out else None)
        if unc_key and isinstance(out.get("value"), (int, float)) and isinstance(out.get(unc_key), (int, float)):
            rv, ru = _round_pair(out["value"], out[unc_key])
            out["value"], out[unc_key] = rv, ru
            for k, v in list(out.items()):
                if k not in ("value", unc_key):
                    out[k] = _round_floats(v, key=k)
            return out
        return {k: _round_floats(v, key=k) for k, v in out.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, key=key) for v in obj]
    if isinstance(obj, float):
        if key in _TIME_KEYS:
            return round(obj, 3)
        return _round_sig(obj, 4)
    return obj


_LONG_DECIMAL_RE = re.compile(r"-?\d+\.\d{7,}(?:[eE][+-]?\d+)?")


def _round_long_decimals(text: str) -> str:
    """Round any number with >6 decimal digits to 4 significant digits."""
    def _sub(m: re.Match) -> str:
        try:
            v = float(m.group(0))
        except ValueError:
            return m.group(0)
        if v == 0:
            return "0"
        return format(_round_sig(v, 4), ".4g")

    return _LONG_DECIMAL_RE.sub(_sub, text)


# Backwards-friendly alias.
postprocess_report_numbers = _round_long_decimals


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
                     "info": f.get("info"),
                     "r2": f.get("r2"), "rmse_px": f.get("rmse_px"),
                     "n_points": f.get("n_points"), "warnings": f.get("warnings"),
                     "segments": [{"index": s.get("index"), "g": s.get("g"), "r2": s.get("r2"),
                                   "excluded_from_pool": s.get("excluded_from_pool")}
                                  for s in (f.get("segments") or [])],
                     "restitution": f.get("restitution")})
    compact = {"question": result.get("question"), "plan": result.get("plan"),
               "calibration": result.get("calibration"), "tracks": tracks,
               "fits": fits, "answer": result.get("answer"),
               "video": {k: (result.get("video") or {}).get(k) for k in ("fps", "width", "height", "n_frames", "duration_s")}}
    return _round_floats(compact)


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
        "Use only the numbers in this JSON; never invent numbers. "
        "Use proper significant figures: never more than 4 significant digits, "
        "no raw floats (do not copy long decimal expansions such as 0.43333333333333335). "
        "Round a value to the decimal place of its uncertainty, giving the uncertainty "
        "to 2 significant digits; when there is no uncertainty use 4 significant digits. "
        "Give times to 3 decimal places.\n\n"
        + json.dumps(compact, indent=1)[:8000])
    models: list[tuple[str, str]] = []
    settings = getattr(client, "settings", None) if client is not None else None
    if settings is not None:
        models = [(getattr(settings, "wonder", "muse-wonder-1.0"), "Wonder"),
                  (getattr(settings, "spark", "muse-spark-1.3"), "Spark")]
    else:
        models = [("muse-wonder-1.0", "Wonder"), ("muse-spark-1.3", "Spark")]
    if client is None:
        return _round_long_decimals(_template_report(result))
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
                return _round_long_decimals(text.strip())
        except Exception:
            continue
    return _round_long_decimals(_template_report(result))
