"""Muse Spark tool-calling agent plus deterministic fallback."""

from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from physicam import calibrate as calib_mod
from physicam.calibrate import Calibration
from physicam.physics import fit_model as phys_fit
from physicam.plan import Plan, normalize_facts
from physicam.sandbox import run_python as sandbox_run

SYSTEM_PROMPT = """You are a careful experimental physicist measuring physics from video.
Rules:
- Never invent numbers: every number in your answer must come from a tool output.
- Always report uncertainty. Inspect fit quality (R^2, residual RMS, warnings) and refit with a different window or model when a fit is poor (R^2 < 0.98 or large warnings).
- Calibration preference, in order: (1) user facts or spoken facts, (2) a static reference object of standard size, (3) the tracked object's standard size, (4) gravity as the ruler (only when the question is NOT about g itself), (5) pixels (say so honestly).
- Use track_objects first, then calibrate, then fit_model, inspect the fit, and finally call finish.
- Prefer per-frame perspective calibration (calibrate_from_track per_frame) when the apparent size varies.
- Keep steps efficient: at most a handful of tool calls."""

TOOLS: list[dict] = [
    {"type": "function", "name": "track_objects", "description": "Track an object with SAM 3.1. Returns stitched track summaries with ids.",
     "parameters": {"type": "object", "properties": {
         "prompt": {"type": "string"}, "t_start": {"type": ["number", "null"]}, "t_end": {"type": ["number", "null"]}},
         "required": ["prompt"], "additionalProperties": False}},
    {"type": "function", "name": "calibrate_reference", "description": "Calibrate from a static reference object of known size.",
     "parameters": {"type": "object", "properties": {
         "prompt": {"type": "string"}, "real_size_m": {"type": "number"}, "dimension": {"type": "string"}},
         "required": ["prompt", "real_size_m"], "additionalProperties": False}},
    {"type": "function", "name": "calibrate_from_track", "description": "Calibrate from a tracked object's known diameter. per_frame enables perspective correction (default true when apparent size varies >3%).",
     "parameters": {"type": "object", "properties": {
         "track_id": {"type": "string"}, "real_diameter_m": {"type": "number"}, "per_frame": {"type": ["boolean", "null"]}},
         "required": ["track_id", "real_diameter_m"], "additionalProperties": False}},
    {"type": "function", "name": "calibrate_from_gravity", "description": "Set scale so fitted acceleration equals g=9.81. Not valid when measuring g itself.",
     "parameters": {"type": "object", "properties": {
         "track_id": {"type": "string"}, "t_start": {"type": ["number", "null"]}, "t_end": {"type": ["number", "null"]}},
         "required": ["track_id"], "additionalProperties": False}},
    {"type": "function", "name": "fit_model", "description": "Fit a physics model (free_fall, projectile, bounce, pendulum, oscillator, constant_velocity, polynomial). Returns params, r2, rmse_px, warnings, segments.",
     "parameters": {"type": "object", "properties": {
         "track_id": {"type": "string"}, "model": {"type": "string"},
         "t_start": {"type": ["number", "null"]}, "t_end": {"type": ["number", "null"]},
         "options": {"type": ["object", "null"], "additionalProperties": True}},
         "required": ["track_id", "model"], "additionalProperties": False}},
    {"type": "function", "name": "inspect_frames", "description": "Send up to 4 extracted frames to Spark vision to disambiguate tracks or read a ruler.",
     "parameters": {"type": "object", "properties": {
         "times_s": {"type": "array", "items": {"type": "number"}}, "question": {"type": "string"}},
         "required": ["times_s", "question"], "additionalProperties": False}},
    {"type": "function", "name": "run_python", "description": "Run numpy/scipy code in a sandbox with `tracks` and `m_per_px` available; print JSON.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                    "required": ["code"], "additionalProperties": False}},
    {"type": "function", "name": "finish", "description": "Finish with the final answer. Every number must come from tool outputs.",
     "parameters": {"type": "object", "properties": {
         "headline": {"type": "string"}, "value": {"type": ["number", "null"]},
         "unit": {"type": "string"}, "uncertainty": {"type": ["number", "null"]},
         "summary": {"type": "string"}, "fit_ids": {"type": "array", "items": {"type": "string"}}},
         "required": ["headline", "summary"], "additionalProperties": False}},
]

PHEN_TO_MODEL = {"free_fall": "free_fall", "projectile": "projectile", "bounce": "bounce",
                 "pendulum": "pendulum", "oscillator": "oscillator",
                 "constant_velocity": "constant_velocity", "rolling": "constant_velocity",
                 "other": "projectile"}


@dataclass
class AgentResult:
    answer: dict | None = None
    tracks: list = field(default_factory=list)
    fits: list = field(default_factory=list)
    calibration: Calibration | None = None
    trace: list = field(default_factory=list)
    used_fallback: bool = False


def _trange(a, b):
    if a is None and b is None:
        return None
    return [float(a) if a is not None else -1e18, float(b) if b is not None else 1e18]


class AgentSession:
    """Holds per-job state and implements the agent tools."""

    def __init__(self, client, video_path, plan: Plan, question: str, facts: dict | None,
                 fps: float, width: int = 0, height: int = 0, emit=None):
        self.client = client
        self.video_path = str(video_path)
        self.plan = plan
        self.question = question
        self.facts = normalize_facts(facts)
        self.fps = float(fps)
        self.width, self.height = int(width), int(height)
        self.emit = emit or (lambda ev: None)
        self.tracks: dict[str, object] = {}
        self._label_cache: dict[str, list] = {}
        self.calibration: Calibration | None = None
        self.fits: list = []
        self._fit_counter = 0
        self.answer: dict | None = None
        self.trace: list[dict] = []

    # -- emit helpers -------------------------------------------------------
    def _trace(self, step: int, name: str, args: dict, summary: str, ms: int):
        entry = {"step": step, "kind": "tool", "name": name, "args": args,
                 "summary": summary[:500], "ms": int(ms)}
        self.trace.append(entry)
        try:
            self.emit({"type": "trace", **entry})
        except Exception:
            pass

    def _emit_track(self, tr):
        try:
            self.emit({"type": "track", "track": tr.to_json()})
        except Exception:
            pass

    def _emit_fit(self, fit):
        try:
            self.emit({"type": "fit", "fit": fit.to_dict()})
        except Exception:
            pass

    def get_track(self, track_id: str):
        if track_id not in self.tracks:
            raise ValueError(f"unknown track_id {track_id!r}; available: {sorted(self.tracks)}")
        return self.tracks[track_id]

    # -- tools ---------------------------------------------------------------
    def tool_track_objects(self, prompt: str, t_start=None, t_end=None) -> dict:
        from physicam.sam import segment_video
        key = prompt.strip().lower()
        if key in self._label_cache:
            return {"tracks": [t.summary() for t in self._label_cache[key]], "cached": True}
        tracks = segment_video(self.client, self.video_path, prompt, self.fps)
        # Optional time-window slice (keep ids from segment_video).
        if t_start is not None or t_end is not None:
            from physicam.sam import Track
            sliced = []
            for tr in tracks:
                mask = np.ones(tr.n, dtype=bool)
                if t_start is not None:
                    mask &= tr.t >= float(t_start)
                if t_end is not None:
                    mask &= tr.t <= float(t_end)
                if mask.sum() < 2:
                    continue
                idx = np.where(mask)[0]
                sliced.append(Track(id=tr.id, label=tr.label, frames=tr.frames[idx], t=tr.t[idx],
                                    x=tr.x[idx], y=tr.y[idx], area=tr.area[idx], bbox=tr.bbox[idx],
                                    conf=tr.conf[idx], eq_diameter=tr.eq_diameter[idx],
                                    contours={int(tr.frames[i]): tr.contours[int(tr.frames[i])]
                                              for i in idx if int(tr.frames[i]) in tr.contours},
                                    fps=tr.fps))
            tracks = sliced
        self._label_cache[key] = tracks
        for tr in tracks:
            self.tracks[tr.id] = tr
            self._emit_track(tr)
        return {"tracks": [t.summary() for t in tracks]}

    def tool_calibrate_reference(self, prompt: str, real_size_m: float, dimension: str = "longest") -> dict:
        mid = 0.0
        try:
            from physicam.video import probe
            mid = probe(self.video_path).duration_s / 2.0
        except Exception:
            pass
        cal = calib_mod.from_reference(self.client, self.video_path, mid, prompt,
                                       float(real_size_m), dimension or "longest")
        self.calibration = cal
        return cal.to_dict()

    def tool_calibrate_from_track(self, track_id: str, real_diameter_m: float, per_frame=None) -> dict:
        tr = self.get_track(track_id)
        if per_frame is None:
            per_frame = calib_mod.apparent_size_variation(tr) > 0.03
        fs = (self.width, self.height) if self.width and self.height else None
        cal = calib_mod.from_track_size(tr, float(real_diameter_m), per_frame=bool(per_frame),
                                        frame_size=fs)
        self.calibration = cal
        out = cal.to_dict()
        out["apparent_size_variation"] = round(calib_mod.apparent_size_variation(tr), 4)
        return out

    def tool_calibrate_from_gravity(self, track_id: str, t_start=None, t_end=None) -> dict:
        tr = self.get_track(track_id)
        none_cal = calib_mod.none()
        # Prefer bounce pooled g_px when the plan is a bounce, else projectile accel.
        model = "bounce" if self.plan.recommended_model == "bounce" else "projectile"
        fit = phys_fit(tr, model, none_cal, t_range=_trange(t_start, t_end) if (t_start is not None or t_end is not None) else None)
        a_px = float(fit.params.get("g_px", fit.params.get("g", {})).get("value", 0.0)) \
            if isinstance(fit.params.get("g_px", fit.params.get("g")), dict) else 0.0
        if not a_px:
            a_px = float(fit.params["g"]["value"])
        cal = calib_mod.from_gravity(a_px)
        self.calibration = cal
        return {**cal.to_dict(), "fitted_accel_px_s2": a_px}

    def tool_fit_model(self, track_id: str, model: str, t_start=None, t_end=None, options=None) -> dict:
        tr = self.get_track(track_id)
        cal = self.calibration or calib_mod.none()
        tr_range = None
        if t_start is not None or t_end is not None:
            lo = float(t_start) if t_start is not None else float(tr.t[0])
            hi = float(t_end) if t_end is not None else float(tr.t[-1])
            tr_range = [lo, hi]
        fit = phys_fit(tr, model, cal, t_range=tr_range, options=options or None)
        fit.id = f"f{self._fit_counter}"
        self._fit_counter += 1
        self.fits.append(fit)
        self._emit_fit(fit)
        out = fit.to_dict()
        # Compact: drop long arrays for the model.
        out.pop("residuals", None)
        if out.get("curve"):
            c = out["curve"]
            out["curve"] = {k: (v[:: max(1, len(v) // 20)] if isinstance(v, list) and len(v) > 20 else v)
                            for k, v in c.items()}
        return out

    def tool_inspect_frames(self, times_s: list, question: str) -> dict:
        from physicam.video import extract_frame
        content: list[dict] = [{"type": "input_text", "text": str(question)}]
        for t in list(times_s or [])[:4]:
            img = extract_frame(self.video_path, float(t))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})
        body = {"model": getattr(getattr(self.client, "settings", None), "spark", "muse-spark-1.3"),
                "max_output_tokens": 2000, "reasoning": {"effort": "low"},
                "input": [{"role": "user", "content": content}]}
        resp = self.client.responses(body)
        text = self.client.output_text(resp) if hasattr(self.client, "output_text") else ""
        if not text:
            from physicam.meta_client import MetaClient
            text = MetaClient.output_text(resp)
        return {"text": text}

    def tool_run_python(self, code: str) -> dict:
        tracks = {tid: {"t": tr.t, "x": tr.x, "y": tr.y, "area": tr.area}
                  for tid, tr in self.tracks.items()}
        mpp = self.calibration.m_per_px if self.calibration else 1.0
        return sandbox_run(code, tracks, mpp)

    def dispatch(self, name: str, args: dict) -> dict:
        """Execute one tool by name; returns a JSON-serialisable dict."""
        args = _sanitize_args(name, dict(args or {}))
        if name == "track_objects":
            return self.tool_track_objects(args["prompt"], args.get("t_start"), args.get("t_end"))
        if name == "calibrate_reference":
            return self.tool_calibrate_reference(args["prompt"], float(args["real_size_m"]),
                                                 args.get("dimension") or "longest")
        if name == "calibrate_from_track":
            return self.tool_calibrate_from_track(args["track_id"], float(args["real_diameter_m"]),
                                                  args.get("per_frame"))
        if name == "calibrate_from_gravity":
            return self.tool_calibrate_from_gravity(args["track_id"], args.get("t_start"), args.get("t_end"))
        if name == "fit_model":
            return self.tool_fit_model(args["track_id"], args["model"], args.get("t_start"),
                                       args.get("t_end"), args.get("options"))
        if name == "inspect_frames":
            return self.tool_inspect_frames(args.get("times_s") or [], args.get("question", ""))
        if name == "run_python":
            return self.tool_run_python(args.get("code", ""))
        raise ValueError(f"unknown tool {name!r}")


def _sanitize_args(name: str, args: dict) -> dict:
    """Unwrap double-encoded JSON string arguments (observed from Spark live).

    Spark sometimes sends ``prompt='{"prompt": "red ball", ...}'``; unwrap
    one level when the parsed dict contains tool-argument keys.
    """
    out = dict(args)
    if "prompt" not in out and out.get("sam_prompt"):
        out["prompt"] = out["sam_prompt"]
    for key in ("prompt", "code", "question", "model", "track_id"):
        v = out.get(key)
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                inner = json.loads(v)
            except Exception:
                continue
            if isinstance(inner, dict) and inner:
                merged = dict(inner)
                for k, val in out.items():
                    if k != key and k not in merged:
                        merged[k] = val
                # The nested dict replaces the stringified value entirely.
                return merged
    return out


def _fcalls(client, resp: dict) -> list[dict]:
    if hasattr(client, "function_calls"):
        try:
            return client.function_calls(resp)
        except Exception:
            pass
    from physicam.meta_client import MetaClient
    return MetaClient.function_calls(resp)


def _otext(client, resp: dict) -> str:
    if hasattr(client, "output_text"):
        try:
            return client.output_text(resp)
        except Exception:
            pass
    from physicam.meta_client import MetaClient
    return MetaClient.output_text(resp)


def run_agent_loop(session: AgentSession, max_steps: int = 12, timeout_s: float = 480.0) -> AgentResult:
    """Spark tool-calling loop with previous_response_id chaining."""
    client = session.client
    start = time.time()
    context = {
        "question": session.question, "facts": session.facts,
        "plan": session.plan.to_dict(),
        "fps": session.fps, "video": session.video_path,
        "instructions": "Track, calibrate, fit, check fit quality, then finish. "
                        "Calibration preference and never-invent-numbers rules are in the system prompt.",
    }
    body: dict = {
        "model": getattr(getattr(client, "settings", None), "spark", "muse-spark-1.3"),
        "max_output_tokens": 4000, "reasoning": {"effort": "low"},
        "instructions": SYSTEM_PROMPT,
        "tools": TOOLS,
        "input": [{"role": "user", "content": [{"type": "input_text",
                    "text": json.dumps(context, indent=1)}]}],
    }
    step = 0
    while step < max_steps:
        if time.time() - start > timeout_s:
            raise TimeoutError(f"agent exceeded {timeout_s}s")
        step += 1
        t0 = time.time()
        resp = client.responses(body)
        calls = _fcalls(client, resp)
        if not calls:
            text = _otext(client, resp)
            raise RuntimeError(f"model returned no tool calls: {text[:300]}")
        outputs: list[dict] = []
        finished = False
        for call in calls:
            name = call.get("name", "")
            try:
                args = json.loads(call.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            args = _sanitize_args(name, args)
            ct0 = time.time()
            if name == "finish" and not session.fits:
                # A finish with no fit is not a measurement; send the model back.
                out = {"error": "finish requires at least one fit_model result first; "
                                "track an object (try the plan sam_prompt without a time window), calibrate, fit, then finish"}
                session._trace(step, "finish_rejected", args, out["error"], 0)
                outputs.append({"type": "function_call_output", "call_id": call.get("call_id", ""),
                                "output": json.dumps(out)})
                continue
            if name == "finish":
                session.answer = {
                    "headline": str(args.get("headline", "")),
                    "value": args.get("value"), "unit": str(args.get("unit", "") or ""),
                    "uncertainty": args.get("uncertainty"),
                    "summary": str(args.get("summary", "")),
                    "fit_ids": list(args.get("fit_ids") or []),
                }
                session._trace(step, "finish", args, session.answer["headline"], int((time.time() - ct0) * 1000))
                finished = True
                break
            try:
                out = session.dispatch(name, args)
                summary = json.dumps(out)[:400]
            except Exception as exc:  # tool errors go back to the model
                out = {"error": f"{type(exc).__name__}: {exc}"}
                summary = out["error"]
            session._trace(step, name, args, summary, int((time.time() - ct0) * 1000))
            outputs.append({"type": "function_call_output", "call_id": call.get("call_id", ""),
                            "output": json.dumps(out)[:12000]})
        if finished:
            break
        resp_id = resp.get("id")
        if resp_id:
            body = {"model": getattr(getattr(client, "settings", None), "spark", "muse-spark-1.3"),
                    "max_output_tokens": 4000, "reasoning": {"effort": "low"},
                    "instructions": SYSTEM_PROMPT, "tools": TOOLS,
                    "previous_response_id": resp_id, "input": outputs}
        else:
            # Fallback chaining for clients without response ids: resend outputs as user text.
            body = {"model": getattr(getattr(client, "settings", None), "spark", "muse-spark-1.3"),
                    "max_output_tokens": 4000, "reasoning": {"effort": "low"},
                    "instructions": SYSTEM_PROMPT, "tools": TOOLS,
                    "input": [{"role": "user", "content": [{"type": "input_text",
                               "text": "Tool results: " + json.dumps(outputs)}]}]}
    if session.answer is None:
        raise RuntimeError("agent ended without finish")
    return AgentResult(answer=session.answer, tracks=list(session.tracks.values()),
                       fits=list(session.fits), calibration=session.calibration,
                       trace=list(session.trace))


def _primary_answer(fit, plan: Plan, question: str, calib: Calibration | None) -> dict:
    """Build a templated answer from a fit (fallback path)."""
    params = fit.params if fit else {}
    q = (question or "").lower()
    # Restitution first for bounce questions that ask for it.
    if fit is not None and fit.model == "bounce" and ("restitution" in q or "bounc" in q):
        valid = [r for r in fit.restitution if "e" in r and r.get("valid", True)]
        if valid:
            e = float(np.mean([r["e"] for r in valid]))
            e_err = float(np.sqrt(np.sum([r.get("e_stderr", 0.0) ** 2 for r in valid])) / len(valid)) if valid else 0.0
            g = params.get("g", {})
            headline = f"e = {e:.2f} ± {e_err:.2f}"
            if g:
                headline += f"; g = {g.get('value', 0):.2f} ± {g.get('stderr', 0):.2f} {g.get('unit','')}"
            return {"headline": headline, "value": round(e, 4), "unit": "",
                    "uncertainty": round(e_err, 4),
                    "summary": f"Coefficient of restitution from {len(valid)} impact(s) "
                               f"(velocity ratio of fitted flight parabolas); pooled g from the same fit. "
                               f"Fit R^2={fit.r2:.3f}, RMSE={fit.rmse_px:.2f}.",
                    "fit_ids": [fit.id]}
    # Pick the most relevant scalar param.
    priority = ["g", "period", "T", "speed", "e_0"]
    for key in priority:
        if key in params and isinstance(params[key], dict) and "value" in params[key]:
            p = params[key]
            unit = p.get("unit", "")
            headline = f"{key} = {p['value']:.3g} ± {p.get('stderr', 0):.2g} {unit}".strip()
            return {"headline": headline, "value": p["value"], "unit": unit,
                    "uncertainty": p.get("stderr", 0.0),
                    "summary": f"{key} from a {fit.model} fit (R^2={fit.r2:.3f}, RMSE={fit.rmse_px:.2f}, "
                               f"n={fit.n_points}). Calibration: {(calib.method if calib else 'none')}.",
                    "fit_ids": [fit.id]}
    return {"headline": "Measurement complete (see fits)", "value": None, "unit": "",
            "uncertainty": None, "summary": "Fit completed; see fitted parameters.",
            "fit_ids": [fit.id] if fit else []}


def run_fallback(session: AgentSession) -> AgentResult:
    """Deterministic recipe: track first object, best calibration, recommended fit."""
    plan = session.plan
    step = 100
    prompt = plan.moving_objects[0].sam_prompt if plan.moving_objects else "red ball"
    t0 = time.time()
    out = session.tool_track_objects(prompt)
    session._trace(step, "track_objects", {"prompt": prompt}, json.dumps(out)[:400],
                   int((time.time() - t0) * 1000))
    if not session.tracks:
        raise RuntimeError(f"fallback: no tracks for prompt {prompt!r}")
    track = list(session.tracks.values())[0]
    # Calibration by best available method.
    nf = session.facts
    calibrated = False
    try:
        if nf.get("reference_object") and nf.get("reference_size_m"):
            t0 = time.time()
            o = session.tool_calibrate_reference(nf["reference_object"], float(nf["reference_size_m"]),
                                                 nf.get("dimension", "longest"))
            session._trace(step, "calibrate_reference", {"prompt": nf["reference_object"]}, json.dumps(o)[:400],
                           int((time.time() - t0) * 1000))
            calibrated = True
        elif nf.get("diameter_m"):
            t0 = time.time()
            o = session.tool_calibrate_from_track(track.id, float(nf["diameter_m"]), None)
            session._trace(step, "calibrate_from_track", {"track_id": track.id}, json.dumps(o)[:400],
                           int((time.time() - t0) * 1000))
            calibrated = True
        elif plan.reference_candidates:
            r = sorted(plan.reference_candidates, key=lambda c: -float(c.confidence or 0))[0]
            if float(r.confidence or 0) >= 0.5:
                t0 = time.time()
                o = session.tool_calibrate_reference(r.sam_prompt, float(r.real_size_m), r.dimension or "longest")
                session._trace(step, "calibrate_reference", {"prompt": r.sam_prompt}, json.dumps(o)[:400],
                               int((time.time() - t0) * 1000))
                calibrated = True
        if not calibrated and plan.moving_objects and plan.moving_objects[0].standard_size_m:
            t0 = time.time()
            o = session.tool_calibrate_from_track(track.id, float(plan.moving_objects[0].standard_size_m), None)
            session._trace(step, "calibrate_from_track", {"track_id": track.id}, json.dumps(o)[:400],
                           int((time.time() - t0) * 1000))
            calibrated = True
    except Exception:
        calibrated = session.calibration is not None
    model = plan.recommended_model or PHEN_TO_MODEL.get(plan.phenomenon, "projectile")
    if model not in ("free_fall", "projectile", "bounce", "pendulum", "oscillator",
                     "constant_velocity", "polynomial"):
        model = PHEN_TO_MODEL.get(plan.phenomenon, "projectile")
    # Gravity-as-ruler only when not measuring g and still uncalibrated.
    target_q = str((plan.target_quantity or {}).get("quantity", "")).lower()
    wants_g = "g" == target_q or target_q.startswith("grav") or "what is g" in session.question.lower()
    if not calibrated and not wants_g and model in ("free_fall", "projectile", "bounce"):
        try:
            t0 = time.time()
            o = session.tool_calibrate_from_gravity(track.id)
            session._trace(step, "calibrate_from_gravity", {"track_id": track.id}, json.dumps(o)[:400],
                           int((time.time() - t0) * 1000))
            calibrated = True
        except Exception:
            pass
    if session.calibration is None:
        session.calibration = calib_mod.none()
    t0 = time.time()
    # Pendulum length option from spoken/facts is not parsed; keep model defaults.
    fit_out = session.tool_fit_model(track.id, model)
    session._trace(step, "fit_model", {"track_id": track.id, "model": model}, json.dumps(fit_out)[:400],
                   int((time.time() - t0) * 1000))
    fit = session.fits[-1] if session.fits else None
    answer = _primary_answer(fit, plan, session.question, session.calibration)
    session.answer = answer
    session._trace(step, "finish", {}, answer["headline"], 0)
    try:
        session.emit({"type": "answer", "answer": answer})
    except Exception:
        pass
    return AgentResult(answer=answer, tracks=list(session.tracks.values()), fits=list(session.fits),
                       calibration=session.calibration, trace=list(session.trace), used_fallback=True)


def run_agent(client, video_path, plan: Plan, question: str, facts: dict | None,
              fps: float, width: int = 0, height: int = 0, emit=None) -> AgentResult:
    """Run the Spark agent; on any failure (or offline) run the deterministic fallback."""
    session = AgentSession(client, video_path, plan, question, facts, fps, width, height, emit)
    if os.environ.get("PHYSICAM_OFFLINE") == "1":
        result = run_fallback(session)
        result.used_fallback = True
        return result
    try:
        result = run_agent_loop(session)
        # The loop already emitted track/fit events; ensure answer event fires.
        try:
            session.emit({"type": "answer", "answer": result.answer})
        except Exception:
            pass
        return result
    except Exception:
        # Fall back, but keep any tracks/fits/calibration the loop produced.
        try:
            result = run_fallback(session)
        except Exception:
            # If fallback also fails but the loop had produced fits, surface its error.
            raise
        result.used_fallback = True
        return result
