"""Muse Spark video-understanding experiment planning."""

from __future__ import annotations

import hashlib
import json
import re
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
             "rolling", "constant_velocity", "draining", "other"]
MODELS = ["free_fall", "projectile", "bounce", "pendulum", "oscillator",
          "constant_velocity", "polynomial", "draining"]


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
    facts_extracted: list[dict] = field(default_factory=list)
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
            facts_extracted=_normalize_fact_list(d.get("facts_extracted"))
            or extract_facts_from_text(str(d.get("spoken_facts", "") or "")),
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
            "facts_extracted": [dict(f) for f in self.facts_extracted],
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


_UNIT_TO_M = {
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "centimetre": 0.01, "centimetres": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "millimetre": 0.001, "millimetres": 0.001,
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
}
_MEASURE_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?|\.\d+)\s*"
    r"(?P<unit>centimetres?|centimeters?|cm|millimetres?|millimeters?|mm|metres?|meters?|m|inches|inch|in|feet|foot|ft)\b",
    re.IGNORECASE)
_REFERS_CANDIDATES = ["hole", "orifice", "jar", "container", "tank",
                      "door frame", "string", "cord", "thread", "pendulum", "bob",
                      "ball", "bottle", "keys", "key", "ruler", "tape measure",
                      "table", "floor", "ramp", "cart", "spring"]


def _normalize_fact_list(raw) -> list[dict]:
    """Normalise a raw facts_extracted list to the canonical shape."""
    out: list[dict] = []
    if not isinstance(raw, (list, tuple)):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        quantity = str(item.get("quantity", "length") or "length").lower().strip()
        if quantity not in ("length", "height", "diameter", "distance"):
            quantity = "length"
        value = item.get("value_m", item.get("value", item.get("size_m")))
        try:
            value_m = float(value)
        except (TypeError, ValueError):
            continue
        out.append({"quantity": quantity, "value_m": value_m,
                    "refers_to": str(item.get("refers_to", "") or ""),
                    "quote": str(item.get("quote", "") or "")})
    return out


def extract_facts_from_text(text: str) -> list[dict]:
    """Extract lengths/heights/diameters/distances with units from free text.

    Returns ``[{quantity, value_m, refers_to, quote}]`` with values in
    metres. Deterministic regex extraction, used for spoken_facts and
    for user question/fact text.
    """
    text = str(text or "")
    out: list[dict] = []
    seen: set[tuple] = set()
    for match in _MEASURE_RE.finditer(text):
        try:
            value_m = float(match.group("value")) * _UNIT_TO_M[match.group("unit").lower()]
        except (KeyError, ValueError):
            continue
        lo = max(0, match.start() - 60)
        hi = min(len(text), match.end() + 30)
        window = text[lo:hi].lower()
        if "diameter" in window or "across" in window:
            quantity = "diameter"
        elif "height" in window or "tall" in window or "drop" in window:
            quantity = "height"
        elif "distance" in window or "away from" in window:
            quantity = "distance"
        else:
            quantity = "length"
        refers_to = ""
        for cand in _REFERS_CANDIDATES:
            if cand in window:
                refers_to = cand
                break
        if quantity == "diameter" and not refers_to:
            refers_to = "bob"
        # Quote: the containing sentence/fragment.
        start = max(text.rfind(sep, 0, match.start()) for sep in ".\n;") + 1
        ends = [p for p in (text.find(sep, match.end()) for sep in ".\n;") if p >= 0]
        end = min(ends) if ends else len(text)
        quote = text[start:end].strip() or text[lo:hi].strip()
        key = (quantity, round(value_m, 9), refers_to)
        if key in seen:
            continue
        seen.add(key)
        out.append({"quantity": quantity, "value_m": float(value_m),
                    "refers_to": refers_to, "quote": quote})
    return out


# Backwards-friendly alias.
parse_spoken_facts = extract_facts_from_text


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
    length = f.get("length_m", f.get("pendulum_length_m", f.get("length")))
    if length is not None:
        try:
            out["length_m"] = float(length)
        except (TypeError, ValueError):
            pass
    string_length = f.get("string_length_m", f.get("string_length"))
    if string_length is not None:
        try:
            out["string_length_m"] = float(string_length)
            out.setdefault("length_m", float(string_length))
        except (TypeError, ValueError):
            pass
    bob_diam = f.get("bob_diameter_m")
    if bob_diam is not None:
        try:
            out["bob_diameter_m"] = float(bob_diam)
        except (TypeError, ValueError):
            pass
    # Draining facts (container/hole/h0/hole level + liquid + reference csv).
    for key in ("container_diameter_m", "hole_diameter_m", "h0_m", "hole_height_m",
                "hole_diameter_stderr_m", "container_diameter_stderr_m"):
        v = f.get(key, f.get(key.replace("_m", "")))
        if v is not None:
            try:
                out[key] = float(v)
            except (TypeError, ValueError):
                pass
    # Common aliases.
    if "container_diameter_m" not in out and f.get("jar_diameter_m") is not None:
        try:
            out["container_diameter_m"] = float(f["jar_diameter_m"])
        except (TypeError, ValueError):
            pass
    if f.get("hole_y_px") is not None:
        try:
            out["hole_y_px"] = float(f["hole_y_px"])
        except (TypeError, ValueError):
            pass
    if f.get("liquid"):
        out["liquid"] = str(f["liquid"])
    if f.get("reference_csv"):
        out["reference_csv"] = str(f["reference_csv"])
    if f.get("container_diameter_m") is not None and "diameter_m" not in out:
        pass  # keep separate from moving-object diameter
    if f.get("length_m_stderr") is not None:
        try:
            out["length_m_stderr"] = float(f["length_m_stderr"])
        except (TypeError, ValueError):
            pass
    extracted = _normalize_fact_list(f.get("facts_extracted"))
    spoken = f.get("spoken_facts") or f.get("spoken") or f.get("narration")
    if spoken:
        out["spoken_facts"] = str(spoken)
        if not extracted:
            extracted = extract_facts_from_text(str(spoken))
    if extracted:
        out["facts_extracted"] = extracted
        for fact in extracted:
            if fact["quantity"] == "length" and "length_m" not in out:
                out["length_m"] = float(fact["value_m"])
            if fact["quantity"] == "diameter" and "diameter_m" not in out:
                out["diameter_m"] = float(fact["value_m"])
    elif out.get("length_m") is not None:
        out["facts_extracted"] = [{"quantity": "length", "value_m": float(out["length_m"]),
                                    "refers_to": str(f.get("length_refers_to", "") or ""),
                                    "quote": str(f.get("quote", "") or "")}]
    return out


def apply_facts(plan: Plan, facts: dict | None) -> Plan:
    """User explicit facts override everything in the plan."""
    nf = normalize_facts(facts)
    if nf.get("facts_extracted"):
        merged = [dict(f) for f in nf["facts_extracted"]]
        for fact in plan.facts_extracted:
            if fact not in merged:
                merged.append(fact)
        plan.facts_extracted = merged
    elif not plan.facts_extracted and plan.spoken_facts:
        plan.facts_extracted = extract_facts_from_text(plan.spoken_facts)
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
            "facts_extracted": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "quantity": {"type": "string", "enum": ["length", "height", "diameter", "distance"]},
                    "value_m": {"type": "number"}, "refers_to": {"type": "string"},
                    "quote": {"type": "string"}},
                "required": ["quantity", "value_m", "refers_to", "quote"]}},
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
                     "spoken_facts", "facts_extracted", "time_window_s", "slow_motion", "capture_fps_guess",
                     "camera_static", "quality_issues", "recommended_model", "target_quantity"],
    }


def heuristic_plan(question: str, facts: dict | None = None) -> Plan:
    """Offline / fallback plan from the question text alone."""
    q = (question or "").lower()
    nf = normalize_facts(facts)
    draining_words = ("drain", "torricelli", "liquid level", "water level", "emptying",
                      "discharge coefficient", "hole", "orifice", "hourglass")
    if any(w in q for w in draining_words) or nf.get("hole_diameter_m") or nf.get("container_diameter_m"):
        phen, model = "draining", "draining"
        if "oil" in q:
            prompt, candidates = "yellow oil", ["yellow oil", "oil", "liquid"]
        elif "brown" in q or "dyed" in q or "agua" in q:
            prompt, candidates = "brown liquid", ["brown liquid", "water", "liquid"]
        else:
            prompt, candidates = "brown liquid", ["brown liquid", "water", "liquid"]
        extracted_d = extract_facts_from_text(question or "")
        # Map diameter facts referring to hole/container into nf keys.
        for fact in extracted_d:
            blob = f"{fact.get('refers_to','')} {fact.get('quote','')}".lower()
            if fact.get("quantity") == "diameter" and "hole" in blob and "hole_diameter_m" not in nf:
                nf["hole_diameter_m"] = float(fact["value_m"])
            if fact.get("quantity") == "diameter" and any(w in blob for w in ("jar", "container", "tank")) and "container_diameter_m" not in nf:
                nf["container_diameter_m"] = float(fact["value_m"])
        plan_d = Plan(experiment="Measure liquid draining (Torricelli): discharge coefficient and drain time",
                      phenomenon=phen, moving_objects=[MovingObject(sam_prompt=prompt, role="liquid",
                                                                    description="liquid body whose surface level falls",
                                                                    sam_prompts=candidates[:4])],
                      spoken_facts=(question if extracted_d else ""),
                      facts_extracted=extracted_d,
                      recommended_model=model,
                      target_quantity={"quantity": "discharge coefficient", "unit": ""})
        return apply_facts(plan_d, {**nf, **({"facts_extracted": extracted_d} if extracted_d else {})})
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
    extracted = extract_facts_from_text(question or "")
    plan = Plan(experiment=f"Measure {target['quantity'] or phen} from video",
                phenomenon=phen, moving_objects=[MovingObject(sam_prompt=prompt, role="moving",
                                                              description="primary moving object",
                                                              sam_prompts=candidates[:4])],
                spoken_facts=(question if extracted else ""),
                facts_extracted=extracted,
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
        "Explicitly capture every spoken length, height, diameter or distance with its unit in "
        "facts_extracted as {quantity: \"length\"|\"height\"|\"diameter\"|\"distance\", value_m: <metres, "
        "converted from cm/mm/in/ft>, refers_to: <what it describes, e.g. \"string\">, quote: <exact spoken "
        "words>}. Example: \"the string is 80 cm\" -> {quantity: \"length\", value_m: 0.80, refers_to: "
        "\"string\", quote: \"the string is 80 cm\"}. Also keep the full spoken text in spoken_facts.\n"
        "For moving_objects give sam_prompt as ONE short phrase of 1-3 words (colour/material + noun, "
        "e.g. \"red ball\", \"metal ball\", \"spring\") and sam_prompts with 2-4 candidate prompts per object, "
        "each also 1-3 words. Long descriptive prompts fail. "
        "Avoid commands, lists and negation. Set standard_size_m only for objects of truly standard size.\n"
        "For reference_candidates list roughly static visible objects of standard known size.\n"
        f"{KNOWN_SIZES}\n"
        "recommended_model must be one of: " + ", ".join(MODELS) + ".\n"
        "If a liquid level falls (emptying jar/tank/bottle, draining, hourglass sand), set phenomenon=\"draining\" "
        "and recommended_model=\"draining\"; the moving object is the liquid itself with sam_prompt/sam_prompts as "
        "colour+liquid candidates (\"brown liquid\", \"water\", \"yellow oil\"), and extract hole diameter, container "
        "diameter and liquid type as facts (diameter facts referring to the hole or the jar/container).\n"
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
