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
from physicam.plan import Plan, extract_facts_from_text, normalize_facts
from physicam.sandbox import run_python as sandbox_run

SYSTEM_PROMPT = """You are a careful experimental physicist measuring physics from video.
Rules:
- Never invent numbers: every number in your answer must come from a tool output.
- Always report uncertainty. Inspect fit quality (R^2, residual RMS, warnings) and refit with a different window or model when a fit is poor (R^2 < 0.98 or large warnings).
- Calibration preference, in order: (1) user facts or spoken facts, (2) a static reference object of standard size, (3) the tracked object's standard size, (4) gravity as the ruler (only when the question is NOT about g itself), (5) pixels (say so honestly).
- Use track_objects first, then calibrate, then fit_model, inspect the fit, and finally call finish.
- SAM prompts must be 1-3 words: colour/material + noun ("metal ball", "red ball", "spring"). Long descriptive prompts fail. Use track_objects with prompts=[...] to try 2-4 short candidates concurrently.
- Prefer per-frame perspective calibration (calibrate_from_track per_frame) when the apparent size varies.
- Keep steps efficient: at most a handful of tool calls.
Pendulum measuring g:
- fit_model(model="pendulum", options={"length_m": L}) computes g = 4*pi^2*L/T0^2, where T0 is the measured period T with the finite-amplitude correction, from a circle fit to the bob path plus a damped-sinusoid fit on theta(t). A known length therefore gives g directly, without needing pixel calibration.
- Spoken or user facts must be passed through to the fit: look in facts, facts_extracted and plan.facts_extracted / plan.spoken_facts for entries like {quantity: "length", value_m: 0.80, refers_to: "string"}. Pass that value_m as options.length_m. If a diameter fact for the bob is also present, also pass options={"bob_diameter_m": <value_m>} so a string length can be corrected to pivot-to-bob-centre.
- Worked example: the user said "the string is 80 cm", so facts_extracted contains a length of 0.80 m referring to the string; call fit_model(track_id="t0", model="pendulum", options={"length_m": 0.80}) and report the returned g (with its uncertainty, usually dominated by the length uncertainty).
- If no length fact and no calibration is available, the pendulum fit returns T, T0 and theta0 plus L_predicted_m, the length implied by assuming g=9.81. That predicted length is a derived fact, never a measurement of g: do not finish with it as if g had been measured."""

TOOLS: list[dict] = [
    {"type": "function", "name": "track_objects", "description": "Track objects with SAM 3.1. Each prompt must be 1-3 words: colour/material + noun (\"metal ball\", \"red ball\", \"spring\"); long descriptive prompts fail and will be simplified. Pass prompts=[...] with 2-4 short candidates to run them concurrently. Returns stitched track summaries with ids.",
     "parameters": {"type": "object", "properties": {
         "prompt": {"type": ["string", "null"]}, "prompts": {"type": ["array", "null"], "items": {"type": "string"}},
         "t_start": {"type": ["number", "null"]}, "t_end": {"type": ["number", "null"]}},
         "required": [], "additionalProperties": False}},
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
    {"type": "function", "name": "fit_model", "description": "Fit a physics model (free_fall, projectile, bounce, pendulum, oscillator, constant_velocity, polynomial). Returns params, r2, rmse_px, warnings, segments. For pendulum: options={\"length_m\": L} (metres, from a spoken/user length fact in facts_extracted) computes g=4*pi^2*L/T0^2 with the finite-amplitude correction, so pass spoken/user length facts through as length_m (e.g. user said 80 cm -> options={\"length_m\": 0.80}); also pass options={\"bob_diameter_m\": D} when the bob diameter is known. With a calibration instead, g comes from the circle-fit L_px*m_per_px. With neither, pendulum returns T, T0, theta0_deg and L_predicted_m (length implied by g=9.81), and no g.",
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


_SAM_FILLER = {"small", "shiny", "spherical", "round", "swinging", "hanging", "free", "at", "the",
               "end", "of", "a", "an", "near", "on", "in", "with", "attached", "coil", "tiny", "large",
               "big", "little", "object", "moving"}


def simplify_sam_prompt(prompt: str) -> tuple[str, bool]:
    """Simplify a SAM prompt to 1-3 words when it exceeds 4 words."""
    import re as _re
    raw = str(prompt or "").strip()
    words = _re.findall(r"[A-Za-z0-9'-]+", raw.lower())
    if len(words) <= 4:
        return raw, False
    kept = [w for w in words if w not in _SAM_FILLER]
    if not kept:
        kept = words
    simplified = " ".join(kept[:3])
    return simplified, simplified.lower() != raw.lower()


def _fmt_num(v) -> str:
    try:
        return f"{float(v):.4g}"
    except Exception:
        return str(v)


def make_trace_title(name: str, args: dict, result) -> str:
    """Human one-line title for a trace event from its tool result."""
    args = args or {}
    if isinstance(result, dict) and result.get("error"):
        return f"{name}: {str(result['error'])[:80]}"
    if name == "track_objects":
        tracks = (result or {}).get("tracks", []) if isinstance(result, dict) else []
        prompts = args.get("prompts") or ([args.get("prompt")] if args.get("prompt") else [])
        simplified = (result or {}).get("simplified") if isinstance(result, dict) else None
        if not tracks:
            label = ", ".join(str(p) for p in prompts) or "object"
            return f"SAM 3.1 found nothing for \u201c{label}\u201d"
        parts = []
        for tr in tracks[:3]:
            tr_range = tr.get("t_range") or [0, 0]
            alias = f" (aliases: {', '.join(tr['aliases'])})" if tr.get('aliases') else ""
            parts.append(f"SAM 3.1 tracked \u201c{tr.get('label', '')}\u201d{alias}: {tr.get('n_frames', 0)} frames "
                         f"({_fmt_num(tr_range[0])}\u2013{_fmt_num(tr_range[1])} s)")
        title = "; ".join(parts)
        if simplified:
            title += f" (simplified prompt: {simplified[0].get('from', '')} \u2192 {simplified[0].get('to', '')})"
        return title
    if name == "fit_model":
        model = args.get("model", "")
        if isinstance(result, dict) and result.get("params"):
            params = result["params"]
            for key in ("T", "g", "speed", "e_0"):
                if key in params and isinstance(params[key], dict):
                    p = params[key]
                    unit = p.get("unit", "")
                    return (f"Fit {model} \u2192 {key} = {_fmt_num(p.get('value'))} \u00b1 "
                            f"{_fmt_num(p.get('stderr', 0))} {unit} (R\u00b2 {_fmt_num(result.get('r2', 0))})").strip()
            return f"Fit {model} (R\u00b2 {_fmt_num(result.get('r2', 0))})"
        return f"Fit {model}"
    if name == "inspect_frames":
        text = str((result or {}).get("text", "")) if isinstance(result, dict) else ""
        n = len(args.get("times_s") or [])
        return f"Looked at {n} frames: {text[:80]}"
    if name == "run_python":
        parsed = (result or {}).get("result") if isinstance(result, dict) else None
        if isinstance(parsed, dict) and parsed:
            inner = ", ".join(f"{k} = {_fmt_num(v)}" for k, v in list(parsed.items())[:4]
                              if isinstance(v, (int, float)))
            return f"Python: {inner}" if inner else f"Python: {str(parsed)[:80]}"
        if parsed is not None:
            return f"Python: {str(parsed)[:80]}"
        stdout = str((result or {}).get("stdout", "")) if isinstance(result, dict) else ""
        return f"Python: {stdout.strip().splitlines()[0][:80] if stdout.strip() else 'ran'}"
    if name.startswith("calibrate"):
        if isinstance(result, dict) and result.get("m_per_px"):
            return f"Calibrated: {_fmt_num(float(result['m_per_px']) * 1000)} mm/px ({result.get('method', '')})"
        return "Calibrated"
    if name == "finish":
        return str(args.get("headline") or (result or {}).get("headline") or "Finished")
    if name == "finish_rejected":
        return "Finish rejected: no fit yet"
    return str(name)


def _trange(a, b):
    if a is None and b is None:
        return None
    return [float(a) if a is not None else -1e18, float(b) if b is not None else 1e18]


def _all_facts_extracted(plan, facts=None) -> list[dict]:
    """All structured length/height/diameter/distance facts for a job.

    Merges user facts (normalised) with the plan's facts_extracted and
    anything extractable from plan.spoken_facts. User facts come first.
    """
    out: list[dict] = []

    def _add(fact):
        if isinstance(fact, dict) and fact.get("value_m") is not None:
            key = (str(fact.get("quantity", "")), float(fact["value_m"]),
                   str(fact.get("refers_to", "")))
            if key not in [(str(f.get("quantity", "")), float(f["value_m"]),
                            str(f.get("refers_to", ""))) for f in out]:
                out.append({"quantity": str(fact.get("quantity", "length")),
                            "value_m": float(fact["value_m"]),
                            "refers_to": str(fact.get("refers_to", "") or ""),
                            "quote": str(fact.get("quote", "") or "")})

    nf = normalize_facts(facts) if facts else {}
    for fact in nf.get("facts_extracted") or []:
        _add(fact)
    if plan is not None:
        for fact in getattr(plan, "facts_extracted", None) or []:
            _add(fact)
        spoken = getattr(plan, "spoken_facts", "") or ""
        if spoken:
            for fact in extract_facts_from_text(str(spoken)):
                _add(fact)
    if nf.get("length_m") is not None and not any(f["quantity"] == "length" for f in out):
        _add({"quantity": "length", "value_m": float(nf["length_m"]), "refers_to": "", "quote": ""})
    return out


def _pendulum_fit_options(plan, facts=None) -> dict | None:
    """fit_model options for a pendulum from spoken/user length facts.

    Returns {"length_m": L, ...} or None when no length fact exists.
    A length referring to the string is flagged, and a known bob
    diameter is passed through so physics can add the bob radius.
    """
    nf = normalize_facts(facts) if facts else {}
    extracted = _all_facts_extracted(plan, facts)
    length_facts = [f for f in extracted if f.get("quantity") == "length"]
    length_fact = None
    for fact in length_facts:
        blob = f"{fact.get('refers_to', '')} {fact.get('quote', '')}".lower()
        if any(word in blob for word in ("string", "pendulum", "cord")):
            length_fact = fact
            break
    if length_fact is None and length_facts:
        length_fact = length_facts[0]
    length_m = nf.get("length_m")
    if length_m is None and length_fact is not None:
        length_m = length_fact.get("value_m")
    if length_m is None and nf.get("string_length_m") is not None:
        length_m = nf["string_length_m"]
    if length_m is None:
        return None
    options: dict = {"length_m": float(length_m)}
    blob = ""
    if length_fact is not None:
        blob = f"{length_fact.get('refers_to', '')} {length_fact.get('quote', '')}".lower()
        if length_fact.get("refers_to"):
            options["length_refers_to"] = str(length_fact["refers_to"])
        if length_fact.get("quote"):
            options["length_quote"] = str(length_fact["quote"])
    if nf.get("string_length_m") is not None or "string" in blob or "cord" in blob:
        options["length_is_string"] = True
    if nf.get("length_m_stderr") is not None:
        options["length_m_stderr"] = float(nf["length_m_stderr"])
    bob_d = nf.get("bob_diameter_m", nf.get("diameter_m"))
    if bob_d is None:
        for fact in extracted:
            if fact.get("quantity") == "diameter":
                bob_d = fact.get("value_m")
                break
    if bob_d is not None:
        options["bob_diameter_m"] = float(bob_d)
    return options


class AgentSession:
    """Holds per-job state and implements the agent tools."""

    def __init__(self, client, video_path, plan: Plan, question: str, facts: dict | None,
                 fps: float, width: int = 0, height: int = 0, emit=None,
                 pre_tracked: dict | None = None, pre_summary: str = ""):
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
        self.pre_summary = pre_summary or ""
        # Cross-prompt dedup: several candidate prompts may segment the
        # same physical object; keep one track (aliases carry the rest).
        _flat: list = []
        for _tracks in (pre_tracked or {}).values():
            for _tr in _tracks:
                if _tr.id in {t.id for t in _flat}:
                    _tr.id = f"t{len(_flat) + len(self.tracks)}"
                    while _tr.id in {t.id for t in _flat}:
                        _tr.id = f"t{int(_tr.id[1:]) + 1}" if _tr.id[1:].isdigit() else f"t{len(_flat)}"
                _flat.append(_tr)
        if _flat:
            from physicam.sam import deduplicate_tracks

            _flat = deduplicate_tracks(_flat)
        for _tr in _flat:
            if _tr.id in self.tracks:
                _tr.id = f"t{len(self.tracks)}"
            self.tracks[_tr.id] = _tr
        for _prompt in (pre_tracked or {}):
            kept = [t for t in _flat if t.label == str(_prompt) or str(_prompt) in (t.aliases or [])]
            self._label_cache[str(_prompt).strip().lower()] = kept or list(_flat)

    # -- emit helpers -------------------------------------------------------
    def _trace(self, step: int, name: str, args: dict, summary: str, ms: int, title: str | None = None, result=None):
        if title is None:
            parsed = result
            if parsed is None:
                try:
                    parsed = json.loads(summary) if summary and summary.strip().startswith("{") else summary
                except Exception:
                    parsed = summary
            try:
                title = make_trace_title(name, args, parsed)
            except Exception:
                title = summary[:120]
        entry = {"step": step, "kind": "tool", "name": name, "args": args,
                 "summary": summary[:2000], "title": str(title)[:300], "ms": int(ms)}
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
    def tool_track_objects(self, prompt: str | None = None, t_start=None, t_end=None, prompts=None) -> dict:
        from concurrent.futures import ThreadPoolExecutor

        from physicam.sam import segment_video
        raw_prompts: list[str] = []
        if prompts:
            raw_prompts = [str(p) for p in prompts if str(p).strip()]
        if prompt:
            raw_prompts.append(str(prompt))
        if not raw_prompts:
            raise ValueError("track_objects requires prompt or prompts")
        simplified: list[dict] = []
        clean_prompts: list[str] = []
        for p in raw_prompts:
            clean, changed = simplify_sam_prompt(p)
            if changed:
                simplified.append({"from": p, "to": clean})
            clean_prompts.append(clean)
        # Dedupe preserving order.
        seen: set[str] = set()
        clean_prompts = [p for p in clean_prompts if not (p.lower() in seen or seen.add(p.lower()))]  # type: ignore[func-returns-value]

        def _slice(tracks):
            if t_start is None and t_end is None:
                return tracks
            from physicam.sam import Track
            out = []
            for tr in tracks:
                mask = np.ones(tr.n, dtype=bool)
                if t_start is not None:
                    mask &= tr.t >= float(t_start)
                if t_end is not None:
                    mask &= tr.t <= float(t_end)
                if mask.sum() < 2:
                    continue
                idx = np.where(mask)[0]
                out.append(Track(id=tr.id, label=tr.label, frames=tr.frames[idx], t=tr.t[idx],
                                 x=tr.x[idx], y=tr.y[idx], area=tr.area[idx], bbox=tr.bbox[idx],
                                 conf=tr.conf[idx], eq_diameter=tr.eq_diameter[idx],
                                 contours={int(tr.frames[i]): tr.contours[int(tr.frames[i])]
                                           for i in idx if int(tr.frames[i]) in tr.contours},
                                 fps=tr.fps, aliases=list(getattr(tr, "aliases", []) or [])))
            return out

        def _one(p: str):
            key = p.strip().lower()
            if key in self._label_cache:
                return p, list(self._label_cache[key]), True
            return p, _slice(segment_video(self.client, self.video_path, p, self.fps)), False

        results: list[tuple[str, list, bool]] = []
        if len(clean_prompts) == 1:
            results = [_one(clean_prompts[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(4, len(clean_prompts))) as pool:
                results = list(pool.map(_one, clean_prompts))
        cached_any = False
        raw_tracks: list = []
        for p, tracks, was_cached in results:
            cached_any = cached_any or was_cached
            for tr in tracks:
                taken = {t.id for t in raw_tracks} | {tid for tid, t in self.tracks.items() if t is not tr}
                while tr.id in taken:
                    tr.id = f"t{int(tr.id[1:]) + 1}" if tr.id[1:].isdigit() else f"t{len(taken)}"
                raw_tracks.append(tr)
        # Cross-prompt dedup: candidate prompts may all segment one object.
        from physicam.sam import deduplicate_tracks

        all_tracks = deduplicate_tracks(raw_tracks) if raw_tracks else []
        kept_ids = {t.id for t in all_tracks}
        for tr in raw_tracks:
            if tr.id not in kept_ids and self.tracks.get(tr.id) is tr:
                del self.tracks[tr.id]
        for p, tracks, was_cached in results:
            matched = [t for t in all_tracks if t.label == p or p in (t.aliases or [])]
            self._label_cache[p.strip().lower()] = matched if matched else list(tracks)
        for tr in all_tracks:
            is_new = self.tracks.get(tr.id) is not tr
            self.tracks[tr.id] = tr
            if is_new:
                self._emit_track(tr)
        out: dict = {"tracks": [t.summary() for t in all_tracks], "prompts": clean_prompts}
        if simplified:
            out["simplified"] = simplified
            out["note"] = "; ".join(f"prompt simplified: {s['from']!r} -> {s['to']!r}" for s in simplified)
        if cached_any:
            out["cached"] = True
        return out

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
        out = sandbox_run(code, tracks, mpp)
        stdout = str(out.get("stdout", "") or "").strip()
        if stdout:
            try:
                out["result"] = json.loads(stdout)
            except Exception:
                # Last non-empty line may be JSON.
                for line in reversed(stdout.splitlines()):
                    try:
                        out["result"] = json.loads(line.strip())
                        break
                    except Exception:
                        continue
        return out

    def dispatch(self, name: str, args: dict) -> dict:
        """Execute one tool by name; returns a JSON-serialisable dict."""
        args = _sanitize_args(name, dict(args or {}))
        if name == "track_objects":
            return self.tool_track_objects(args.get("prompt"), args.get("t_start"), args.get("t_end"),
                                           prompts=args.get("prompts"))
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
        "facts_extracted": _all_facts_extracted(session.plan, session.facts),
        "plan": session.plan.to_dict(),
        "fps": session.fps, "video": session.video_path,
        "instructions": "Track, calibrate, fit, check fit quality, then finish. "
                        "Calibration preference and never-invent-numbers rules are in the system prompt. "
                        "For a pendulum, pass any length in facts_extracted through as "
                        "fit_model options.length_m (spoken example: \"the string is 80 cm\" -> "
                        "options={\"length_m\": 0.80}) to compute g.",
        "pre_tracked": session.pre_summary,
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
                session._trace(step, "finish", args, session.answer["headline"], int((time.time() - ct0) * 1000), result=session.answer)
                finished = True
                break
            try:
                out = session.dispatch(name, args)
                summary = json.dumps(out)[:400]
            except Exception as exc:  # tool errors go back to the model
                out = {"error": f"{type(exc).__name__}: {exc}"}
                summary = out["error"]
            session._trace(step, name, args, summary, int((time.time() - ct0) * 1000), result=out)
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
    candidates = (plan.moving_objects[0].sam_prompts if plan.moving_objects and plan.moving_objects[0].sam_prompts else [prompt])
    t0 = time.time()
    if session.tracks:
        out = {"tracks": [t.summary() for t in session.tracks.values()], "pre_tracked": True}
    else:
        out = session.tool_track_objects(prompts=candidates)
    session._trace(step, "track_objects", {"prompt": prompt, "prompts": candidates}, json.dumps(out)[:2000],
                   int((time.time() - t0) * 1000), result=out)
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
            session._trace(step, "calibrate_reference", {"prompt": nf["reference_object"]}, json.dumps(o)[:2000],
                           int((time.time() - t0) * 1000), result=o)
            calibrated = True
        elif nf.get("diameter_m"):
            t0 = time.time()
            o = session.tool_calibrate_from_track(track.id, float(nf["diameter_m"]), None)
            session._trace(step, "calibrate_from_track", {"track_id": track.id}, json.dumps(o)[:2000],
                           int((time.time() - t0) * 1000), result=o)
            calibrated = True
        elif plan.reference_candidates:
            r = sorted(plan.reference_candidates, key=lambda c: -float(c.confidence or 0))[0]
            if float(r.confidence or 0) >= 0.5:
                t0 = time.time()
                o = session.tool_calibrate_reference(r.sam_prompt, float(r.real_size_m), r.dimension or "longest")
                session._trace(step, "calibrate_reference", {"prompt": r.sam_prompt}, json.dumps(o)[:2000],
                               int((time.time() - t0) * 1000), result=o)
                calibrated = True
        if not calibrated and plan.moving_objects and plan.moving_objects[0].standard_size_m:
            t0 = time.time()
            o = session.tool_calibrate_from_track(track.id, float(plan.moving_objects[0].standard_size_m), None)
            session._trace(step, "calibrate_from_track", {"track_id": track.id}, json.dumps(o)[:2000],
                           int((time.time() - t0) * 1000), result=o)
            calibrated = True
    except Exception:
        calibrated = session.calibration is not None
    model = plan.recommended_model or PHEN_TO_MODEL.get(plan.phenomenon, "projectile")
    if model not in ("free_fall", "projectile", "bounce", "pendulum", "oscillator",
                     "constant_velocity", "polynomial"):
        model = PHEN_TO_MODEL.get(plan.phenomenon, "projectile")
    # Gravity-as-ruler only when not measuring g and still uncalibrated.
    target_q = str((plan.target_quantity or {}).get("quantity", "")).lower()
    import re as _re
    wants_g = ("g" == target_q or target_q.startswith("grav") or "gravity" in session.question.lower()
               or bool(_re.search(r"\bg\b", session.question.lower())))
    if not calibrated and not wants_g and model in ("free_fall", "projectile", "bounce"):
        try:
            t0 = time.time()
            o = session.tool_calibrate_from_gravity(track.id)
            session._trace(step, "calibrate_from_gravity", {"track_id": track.id}, json.dumps(o)[:2000],
                           int((time.time() - t0) * 1000), result=o)
            calibrated = True
        except Exception:
            pass
    if session.calibration is None:
        session.calibration = calib_mod.none()
    t0 = time.time()
    # Pendulum: pass spoken/user length facts through as length_m.
    fit_options = _pendulum_fit_options(plan, session.facts) if model == "pendulum" else None
    tw = plan.time_window_s if getattr(plan, "time_window_s", None) else None
    fit_out = (session.tool_fit_model(track.id, model, float(tw[0]), float(tw[1]), options=fit_options)
               if tw and len(tw) == 2 else session.tool_fit_model(track.id, model, options=fit_options))
    session._trace(step, "fit_model", {"track_id": track.id, "model": model,
                                       **({"options": fit_options} if fit_options else {})},
                   json.dumps(fit_out)[:2000],
                   int((time.time() - t0) * 1000), result=fit_out)
    fit = session.fits[-1] if session.fits else None
    answer = _primary_answer(fit, plan, session.question, session.calibration)
    session.answer = answer
    session._trace(step, "finish", {}, answer["headline"], 0, result=answer)
    try:
        session.emit({"type": "answer", "answer": answer})
    except Exception:
        pass
    return AgentResult(answer=answer, tracks=list(session.tracks.values()), fits=list(session.fits),
                       calibration=session.calibration, trace=list(session.trace), used_fallback=True)


def run_agent(client, video_path, plan: Plan, question: str, facts: dict | None,
              fps: float, width: int = 0, height: int = 0, emit=None,
              pre_tracked: dict | None = None, pre_summary: str = "") -> AgentResult:
    """Run the Spark agent; on any failure (or offline) run the deterministic fallback."""
    session = AgentSession(client, video_path, plan, question, facts, fps, width, height, emit,
                           pre_tracked=pre_tracked, pre_summary=pre_summary)
    for _tr in list(session.tracks.values()):
        session._emit_track(_tr)
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
