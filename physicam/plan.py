"""Muse Spark video-understanding experiment planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

KNOWN_SIZES = """Known standard sizes (use only when the object is clearly visible):
- A4 sheet: 0.297 m (long) x 0.210 m; US Letter: 0.279 x 0.216 m
- Credit card: 0.0856 m (long); tennis ball diameter 0.067 m; ping-pong ball 0.040 m
- Soda can: 0.122 m tall, 0.066 m diameter; basketball 0.24 m; soccer ball 0.22 m
- Ruler / tape measure: read the marks if legible; a standard door is ~2.0 m tall
- Smartphone: ~0.147 m long (varies; low confidence unless model is identifiable)"""

PHENOMENA = ["free_fall", "projectile", "bounce", "pendulum", "oscillator",
             "rolling", "constant_velocity", "other"]
MODELS = ["free_fall", "projectile", "bounce", "pendulum", "oscillator",
          "constant_velocity", "polynomial"]


@dataclass
class MovingObject:
    sam_prompt: str
    role: str = "moving"
    description: str = ""
    standard_size_m: float | None = None
    sam_prompts: list[str] = field(default_factory=list)


@dataclass
class ReferenceCandidate:
    sam_prompt: str
    real_size_m: float
    dimension: str = "longest"
    confidence: float = 0.5
    why: str = ""


@dataclass
class Plan:
    """Structured experiment plan from Spark (see AGENTS.md contract)."""

    experiment: str = ""
    phenomenon: str = "other"
    moving_objects: list[MovingObject] = field(default_factory=list)
    reference_candidates: list[ReferenceCandidate] = field(default_factory=list)
    spoken_facts: str = ""
    time_window_s: list[float] | None = None
    slow_motion: bool = False
    capture_fps_guess: float | None = None
    camera_static: bool = True
    quality_issues: list[str] = field(default_factory=list)
    recommended_model: str = "projectile"
    target_quantity: dict = field(default_factory=lambda: {"quantity": "", "unit": ""})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        d = dict(d or {})
        return cls(
            experiment=str(d.get("experiment", "")),
            phenomenon=str(d.get("phenomenon", "other")),
            moving_objects=[MovingObject(sam_prompt=str(m.get("sam_prompt", "")),
                                         role=str(m.get("role", "moving")),
                                         description=str(m.get("description", "")),
                                         standard_size_m=m.get("standard_size_m"),
                                         sam_prompts=([str(p) for p in m["sam_prompts"] if p] if len(m.get("sam_prompts") or []) >= 2
                                                      else default_sam_prompts(str(m.get("sam_prompt", "")))))
                            for m in (d.get("moving_objects") or []) if isinstance(m, dict) and m.get("sam_prompt")],
            reference_candidates=[ReferenceCandidate(**{k: r.get(k) for k in ("sam_prompt", "real_size_m", "dimension", "confidence", "why") if k in r})
                                  for r in (d.get("reference_candidates") or []) if isinstance(r, dict) and r.get("sam_prompt")],
            spoken_facts=str(d.get("spoken_facts", "") or ""),
            time_window_s=list(d["time_window_s"]) if d.get("time_window_s") else None,
            slow_motion=bool(d.get("slow_motion", False)),
            capture_fps_guess=d.get("capture_fps_guess"),
            camera_static=bool(d.get("camera_static", True)),
            quality_issues=list(d.get("quality_issues") or []),
            recommended_model=str(d.get("recommended_model", "projectile")),
            target_quantity=dict(d.get("target_quantity") or {"quantity": "", "unit": ""}),
        )

    def contract_dict(self) -> dict:
        """Plan shape stored in result.json per AGENTS.md."""
        ref = None
        if self.reference_candidates:
            r = self.reference_candidates[0]
            ref = {"sam_prompt": r.sam_prompt, "real_size_m": r.real_size_m,
                   "dimension": r.dimension, "confidence": r.confidence, "why": r.why}
        return {
            "experiment": self.experiment,
            "phenomenon": self.phenomenon,
            "objects": [{"sam_prompt": o.sam_prompt, "sam_prompts": (o.sam_prompts or [o.sam_prompt]), "role": o.role} for o in self.moving_objects],
            "moving_objects": [asdict(o) for o in self.moving_objects],
            "reference_candidates": [asdict(r) for r in self.reference_candidates],
            "reference": ref,
            "model": self.recommended_model,
            "recommended_model": self.recommended_model,
            "spoken_facts": self.spoken_facts,
            "time_window_s": self.time_window_s,
            "slow_motion": self.slow_motion,
            "capture_fps_guess": self.capture_fps_guess,
            "camera_static": self.camera_static,
            "quality_issues": self.quality_issues,
            "target_quantity": self.target_quantity,
            "notes": self.spoken_facts,
        }


def default_sam_prompts(prompt: str) -> list[str]:
    """Generate 2-4 short candidate SAM prompts from one prompt."""
    words = str(prompt or "").split()
    out: list[str] = []
    for cand in [" ".join(words), " ".join(words[1:]), " ".join(words[-2:]), words[-1] if words else ""]:
        cand = cand.strip()
        if cand and len(cand.split()) <= 3 and cand not in out:
            out.append(cand)
    return out[:4] or [str(prompt)]


def normalize_facts(facts: dict | None) -> dict:
    """Normalise user facts from the UI / CLI into canonical keys."""
    f = dict(facts or {})
    out: dict[str, Any] = {}
    # reference
    ref_obj = f.get("reference_object") or f.get("reference") or f.get("ref_prompt") or f.get("reference_prompt")
    ref_size = f.get("reference_size_m", f.get("real_size_m", f.get("reference_size")))
    if ref_obj:
        out["reference_object"] = str(ref_obj)
    if ref_size is not None:
        try:
            out["reference_size_m"] = float(ref_size)
        except (TypeError, ValueError):
            pass
    fps = f.get("fps_override", f.get("capture_fps", f.get("capture_fps_override")))
    if fps is not None:
        try:
            out["fps_override"] = float(fps)
        except (TypeError, ValueError):
            pass
    diam = f.get("diameter_m", f.get("object_diameter_m", f.get("diameter")))
    if diam is not None:
        try:
            out["diameter_m"] = float(diam)
        except (TypeError, ValueError):
            pass
    if f.get("dimension"):
        out["dimension"] = str(f["dimension"])
    return out


def apply_facts(plan: Plan, facts: dict | None) -> Plan:
    """User explicit facts override everything in the plan."""
    nf = normalize_facts(facts)
    if nf.get("fps_override"):
        plan.capture_fps_guess = float(nf["fps_override"])
        plan.slow_motion = True
    if nf.get("diameter_m") and plan.moving_objects:
        plan.moving_objects[0].standard_size_m = float(nf["diameter_m"])
    if nf.get("reference_object") and nf.get("reference_size_m"):
        cand = ReferenceCandidate(sam_prompt=str(nf["reference_object"]),
                                  real_size_m=float(nf["reference_size_m"]),
                                  dimension=str(nf.get("dimension", "longest")),
                                  confidence=1.0, why="user-provided fact")
        plan.reference_candidates = [cand] + [c for c in plan.reference_candidates
                                              if c.sam_prompt != cand.sam_prompt]
    return plan


def _schema() -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "experiment": {"type": "string"},
            "phenomenon": {"type": "string", "enum": PHENOMENA},
            "moving_objects": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "sam_prompt": {"type": "string"}, "role": {"type": "string"},
                    "description": {"type": "string"},
                    "standard_size_m": {"type": ["number", "null"]},
                    "sam_prompts": {"type": "array", "items": {"type": "string"}}},
                "required": ["sam_prompt", "role", "description", "standard_size_m", "sam_prompts"]}},
            "reference_candidates": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "sam_prompt": {"type": "string"}, "real_size_m": {"type": "number"},
                    "dimension": {"type": "string"}, "confidence": {"type": "number"},
                    "why": {"type": "string"}},
                "required": ["sam_prompt", "real_size_m", "dimension", "confidence", "why"]}},
            "spoken_facts": {"type": "string"},
            "time_window_s": {"type": ["array", "null"], "items": {"type": "number"}},
            "slow_motion": {"type": "boolean"},
            "capture_fps_guess": {"type": ["number", "null"]},
            "camera_static": {"type": "boolean"},
            "quality_issues": {"type": "array", "items": {"type": "string"}},
            "recommended_model": {"type": "string", "enum": MODELS},
            "target_quantity": {"type": "object", "additionalProperties": False,
                                "properties": {"quantity": {"type": "string"}, "unit": {"type": "string"}},
                                "required": ["quantity", "unit"]},
        },
        "required": ["experiment", "phenomenon", "moving_objects", "reference_candidates",
                     "spoken_facts", "time_window_s", "slow_motion", "capture_fps_guess",
                     "camera_static", "quality_issues", "recommended_model", "target_quantity"],
    }


def heuristic_plan(question: str, facts: dict | None = None) -> Plan:
    """Offline / fallback plan from the question text alone."""
    q = (question or "").lower()
    nf = normalize_facts(facts)
    if "pendulum" in q or "period" in q and "swing" in q:
        phen, model = "pendulum", "pendulum"
    elif "restitution" in q or "bounce" in q or "bounc" in q:
        phen, model = "bounce", "bounce"
    elif "spring" in q or "oscillat" in q:
        phen, model = "oscillator", "oscillator"
    elif "how fast" in q or "velocity" in q or "speed" in q:
        phen, model = "constant_velocity", "constant_velocity"
    elif "projectile" in q:
        phen, model = "projectile", "projectile"
    else:
        phen, model = "free_fall", "projectile"
    prompt = "red ball" if "ball" in q else "moving object"
    candidates = [prompt] + (["ball"] if "ball" in q else ["object"])
    target = {"quantity": "coefficient of restitution", "unit": ""} if phen == "bounce" and "restitution" in q else \
        {"quantity": "g", "unit": "m/s^2"} if "g" in q.split() or "gravity" in q else {"quantity": "", "unit": ""}
    plan = Plan(experiment=f"Measure {target['quantity'] or phen} from video",
                phenomenon=phen, moving_objects=[MovingObject(sam_prompt=prompt, role="moving",
                                                              description="primary moving object",
                                                              sam_prompts=candidates[:4])],
                recommended_model=model, target_quantity=target)
    return apply_facts(plan, nf)


def plan_experiment(client, proxy_video_path, question: str, facts: dict | None = None) -> Plan:
    """Ask Muse Spark to watch the proxy video and plan the experiment."""
    nf = normalize_facts(facts)
    # Disk cache keyed by proxy sha + question + facts (replays offline).
    cache_path: Path | None = None
    try:
        from physicam.video import sha256_file
        data_dir = getattr(getattr(client, "settings", None), "data_dir", "data")
        key_src = sha256_file(proxy_video_path) + "|" + question + "|" + json.dumps(nf, sort_keys=True)
        key = hashlib.sha256(key_src.encode()).hexdigest()
        cache_path = Path(data_dir) / "cache" / "plan" / f"{key}.json"
        if cache_path.exists():
            return apply_facts(Plan.from_dict(json.loads(cache_path.read_text())), nf)
    except Exception:
        cache_path = None

    file_id = client.upload_file(proxy_video_path)
    prompt = (
        "You are planning a quantitative physics measurement from this video.\n"
        f"User question: {question}\n"
        f"User-provided facts (override everything else): {json.dumps(nf)}\n\n"
        "Watch and listen: narration may state lengths, fps or other facts - put them in spoken_facts.\n"
        "For moving_objects give sam_prompt as ONE short phrase of 1-3 words (colour/material + noun, "
        "e.g. \"red ball\", \"metal ball\", \"spring\") and sam_prompts with 2-4 candidate prompts per object, "
        "each also 1-3 words. Long descriptive prompts fail. "
        "Avoid commands, lists and negation. Set standard_size_m only for objects of truly standard size.\n"
        "For reference_candidates list roughly static visible objects of standard known size.\n"
        f"{KNOWN_SIZES}\n"
        "recommended_model must be one of: " + ", ".join(MODELS) + ".\n"
        "time_window_s is [start, end] in proxy-video seconds where the relevant motion happens, or null.\n"
        "If the clip looks like slow motion, set slow_motion and guess the true capture fps."
    )
    body = {
        "model": getattr(getattr(client, "settings", None), "spark", "muse-spark-1.3"),
        "max_output_tokens": 4000,
        "reasoning": {"effort": "low"},
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_video", "file_id": file_id}]}],
        "text": {"format": {"type": "json_schema", "name": "plan", "schema": _schema(), "strict": True}},
    }
    resp = client.responses(body)
    text = client.output_text(resp) if hasattr(client, "output_text") else ""
    if not text:
        # Tolerate clients that only expose raw output.
        from physicam.meta_client import MetaClient
        text = MetaClient.output_text(resp)
    plan = Plan.from_dict(json.loads(text))
    plan = apply_facts(plan, nf)
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(plan.to_dict(), indent=1))
        except Exception:
            pass
    return plan
