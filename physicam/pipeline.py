"""Job orchestration: ingest -> plan -> agent -> report -> done."""

from __future__ import annotations

import json
import time
from pathlib import Path

from physicam.calibrate import Calibration
from physicam.plan import Plan, apply_facts, heuristic_plan, normalize_facts, plan_experiment

PALETTE = ["#ff4d4f", "#4da3ff", "#ffd24d", "#52e0a0", "#c792ff", "#ff9f43"]


def _cost_registry_path(data_dir) -> Path:
    return Path(data_dir) / "cache" / "cost_original.json"


def _cost_key(video_sha: str, question: str, facts) -> str:
    import hashlib

    src = f"{video_sha}|{question}|{json.dumps(facts or {}, sort_keys=True, default=str)}"
    return hashlib.sha256(src.encode()).hexdigest()


def _load_cost_registry(data_dir) -> dict:
    try:
        p = _cost_registry_path(data_dir)
        if p.exists():
            data = json.loads(p.read_text())
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _save_cost_registry(data_dir, registry: dict) -> None:
    try:
        p = _cost_registry_path(data_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(registry, indent=1))
        tmp.replace(p)
    except Exception:
        pass


def _is_live_cost(cost: dict) -> bool:
    try:
        return bool((cost.get("sam_frames") or 0) or (cost.get("spark_tokens") or 0)
                    or (cost.get("total_tokens") or 0) or (cost.get("usd_estimate") or 0))
    except Exception:
        return False


def _write_result(job_dir: Path, result: dict) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    tmp = job_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(job_dir / "result.json")


def _track_json(tr, color: str) -> dict:
    d = tr.to_json()
    d["color"] = color
    # Contract keys first per AGENTS.md; extra keys (conf, eq_diameter, fps) are tolerated.
    return d


def _fit_json(fit, idx: int) -> dict:
    d = fit.to_dict()
    if not d.get("id"):
        d["id"] = f"f{idx}"
    d.setdefault("impacts", [])
    d.setdefault("info", {})
    return d


def _pretrack(client, video_path, plan, fps: float):
    """Run SAM concurrently on plan candidate prompts; keep tracks found.

    Returns (pre_tracked dict prompt->tracks, summary string). Offline, only
    prompts with a cached SAM file are tried so no API call is made.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path as _P

    from physicam.agent import simplify_sam_prompt
    from physicam.sam import _sam_cache_path, segment_video
    from physicam.video import sha256_file

    if client is None:
        return {}, ""
    candidates: list[str] = []
    for obj in (plan.moving_objects or []):
        for p in (obj.sam_prompts or [obj.sam_prompt]):
            clean, _ = simplify_sam_prompt(str(p))
            if clean and clean not in candidates:
                candidates.append(clean)
    # Cache-discovered prompts for this exact video (free offline replays).
    try:
        data_dir = getattr(getattr(client, "settings", None), "data_dir", "data")
        sha = sha256_file(video_path)
        for path in sorted((_P(data_dir) / "cache" / "sam").glob(f"{sha}_*.txt")):
            slug = path.name[len(sha) + 1:-4]
            if slug.startswith("prompt_"):
                continue
            clean, _ = simplify_sam_prompt(slug.replace("_", " "))
            if clean and clean not in candidates:
                candidates.append(clean)
        if os.environ.get("PHYSICAM_OFFLINE") == "1":
            candidates = [c for c in candidates if _sam_cache_path(data_dir, sha, c).exists()]
    except Exception:
        pass
    if not candidates:
        return {}, ""

    def _one(p):
        try:
            return p, segment_video(client, video_path, p, fps)
        except Exception:
            return p, []

    if len(candidates) == 1:
        results = [_one(candidates[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(4, len(candidates))) as pool:
            results = list(pool.map(_one, candidates))
    # Cross-prompt dedup: candidate prompts may segment the same object.
    from physicam.sam import deduplicate_tracks

    flat: list = []
    for prompt, tracks in results:
        for tr in [t for t in (tracks or []) if t.n > 0]:
            taken = {t.id for t in flat}
            while tr.id in taken:
                tr.id = f"t{int(tr.id[1:]) + 1}" if tr.id[1:].isdigit() else f"t{len(taken)}"
            flat.append(tr)
    deduped = deduplicate_tracks(flat) if flat else []
    pre: dict = {}
    parts: list[str] = []
    for tr in deduped:
        pre.setdefault(tr.label, []).append(tr)
        alias = f" (aliases: {', '.join(tr.aliases)})" if tr.aliases else ""
        parts.append(f"\u201c{tr.label}\u201d{alias}: {tr.n} frames ({float(tr.t[0]):.2f}\u2013{float(tr.t[-1]):.2f} s)")
    summary = ("pre-tracked: " + "; ".join(parts)) if parts else "pre-tracked: none"
    return pre, summary


def run_job(job_dir, video_path, question: str, facts: dict | None = None,
            emit=None, client=None) -> dict:
    """Run the full pipeline for one job and return the result dict.

    ``emit(event_dict)`` is called for every SSE event. ``client`` may be
    injected (tests / FakeClient); otherwise a MetaClient is created.
    """
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    _user_emit = emit or (lambda ev: None)
    try:
        (job_dir / "events.jsonl").unlink(missing_ok=True)
    except Exception:
        pass

    def emit(ev: dict) -> None:  # persist every event for SSE replay after restart
        try:
            with open(job_dir / "events.jsonl", "a") as fh:
                fh.write(json.dumps(ev) + "\n")
        except Exception:
            pass
        _user_emit(ev)
    job_id = job_dir.name
    nf = normalize_facts(facts)
    owned_client = False
    if client is None:
        import os
        if os.environ.get("PHYSICAM_OFFLINE") == "1":
            # Offline still needs a client object for SAM cache reads; build one
            # without requiring a key for cached paths. segment_video only calls
            # the API on a cache miss, where it will fail honestly.
            from physicam.meta_client import MetaClient
            try:
                client = MetaClient()
            except Exception:
                client = None
        else:
            from physicam.meta_client import MetaClient
            client = MetaClient()
            owned_client = True

    result: dict = {
        "id": job_id, "status": "running", "question": question,
        "video": {}, "plan": None, "calibration": None,
        "tracks": [], "fits": [], "answer": None, "strobe_url": None,
        "report_markdown": "", "trace": [],
        "cost": {"sam_frames": 0, "spark_tokens": 0, "usd_estimate": 0.0},
        "error": None,
    }

    def safe_emit(ev: dict) -> None:
        try:
            emit(ev)
        except Exception:
            pass

    def stage(name: str) -> None:
        safe_emit({"type": "stage", "stage": name})
        _write_result(job_dir, result)

    # Original live cost for this video+question (cached replays carry it over).
    _cost_data_dir = getattr(getattr(client, "settings", None), "data_dir", "data") if client is not None else "data"
    _cost_key_val: str | None = None
    _prior_original: dict | None = None
    try:
        from physicam.video import sha256_file as _sha

        _cost_key_val = _cost_key(_sha(Path(video_path)), question, nf)
        _prior = _load_cost_registry(_cost_data_dir).get(_cost_key_val)
        if isinstance(_prior, dict) and _prior:
            _prior_original = {k: v for k, v in _prior.items() if k != "original"}
    except Exception:
        _cost_key_val, _prior_original = None, None

    def _finalize_cost() -> dict:
        cost = cost_now()
        if not isinstance(cost, dict):
            cost = {"sam_frames": 0, "spark_tokens": 0, "usd_estimate": 0.0}
        cost = {k: v for k, v in cost.items() if k != "original"}
        if _prior_original:
            cost["original"] = dict(_prior_original)
        elif _is_live_cost(cost) and _cost_key_val:
            snapshot = dict(cost)
            registry = _load_cost_registry(_cost_data_dir)
            registry[_cost_key_val] = snapshot
            _save_cost_registry(_cost_data_dir, registry)
            cost["original"] = dict(snapshot)
        return cost

    def cost_now() -> dict:
        try:
            if client is not None and hasattr(client, "cost"):
                return client.cost.to_dict()
        except Exception:
            pass
        return result["cost"]

    try:
        from physicam.video import normalize, make_proxy, probe

        # 1. ingest -----------------------------------------------------
        stage("ingest")
        src = Path(video_path)
        info_src = probe(src)
        normalized = job_dir / "video.mp4"
        if src.resolve() != normalized.resolve():
            info = normalize(src, normalized)
        else:
            info = info_src
        proxy = job_dir / "proxy.mp4"
        try:
            make_proxy(normalized, proxy)
        except Exception:
            proxy = normalized  # audio-less fallback: plan on the normalized file
        capture_fps = float(nf["fps_override"]) if nf.get("fps_override") else float(info.fps)
        duration_capture = (info.n_frames / capture_fps) if capture_fps else info.duration_s
        result["video"] = {
            "url": f"/media/{job_id}/video.mp4", "fps": capture_fps,
            "container_fps": float(info.fps), "width": int(info.width), "height": int(info.height),
            "n_frames": int(info.n_frames), "duration_s": round(float(duration_capture), 4),
        }
        result["cost"] = cost_now()
        _write_result(job_dir, result)

        # 2. plan ---------------------------------------------------------
        stage("plan")
        import os
        plan: Plan
        if os.environ.get("PHYSICAM_OFFLINE") == "1":
            try:
                if client is not None:
                    plan = plan_experiment(client, proxy, question, nf)
                else:
                    raise RuntimeError("offline without client")
            except Exception:
                plan = heuristic_plan(question, nf)
        elif client is None:
            plan = heuristic_plan(question, nf)
        else:
            plan = plan_experiment(client, proxy, question, nf)
        plan = apply_facts(plan, nf)
        if nf.get("fps_override"):
            capture_fps = float(nf["fps_override"])
        elif plan.capture_fps_guess:
            capture_fps = float(plan.capture_fps_guess)
            result["video"]["fps"] = capture_fps
            result["video"]["duration_s"] = round(info.n_frames / capture_fps, 4) if capture_fps else info.duration_s
        result["plan"] = plan.contract_dict()
        result["cost"] = cost_now()
        _write_result(job_dir, result)
        safe_emit({"type": "plan", "plan": result["plan"]})

        # 3. agent --------------------------------------------------------
        stage("agent")
        from physicam.agent import run_agent

        track_colors: dict[str, str] = {}

        def agent_emit(ev: dict) -> None:
            etype = ev.get("type")
            if etype == "trace":
                result["trace"].append({k: ev.get(k) for k in ("step", "kind", "name", "args", "summary", "title", "ms")})
                _write_result(job_dir, result)
                safe_emit(ev)
            elif etype == "track" and ev.get("track"):
                tid = ev["track"].get("id", "")
                if tid not in track_colors:
                    track_colors[tid] = PALETTE[len(track_colors) % len(PALETTE)]
                tj = dict(ev["track"])
                tj["color"] = track_colors[tid]
                ids = [t["id"] for t in result["tracks"]]
                if tid in ids:
                    result["tracks"][ids.index(tid)] = tj
                else:
                    result["tracks"].append(tj)
                result["cost"] = cost_now()
                _write_result(job_dir, result)
                safe_emit({"type": "track", "track": tj})
            elif etype == "fit" and ev.get("fit"):
                fj = ev["fit"]
                ids = [f["id"] for f in result["fits"]]
                if fj.get("id") in ids:
                    result["fits"][ids.index(fj["id"])] = fj
                else:
                    result["fits"].append(fj)
                _write_result(job_dir, result)
                safe_emit(ev)
            elif etype == "answer":
                result["answer"] = ev.get("answer")
                _write_result(job_dir, result)
                safe_emit(ev)
            else:
                safe_emit(ev)

        pre_tracked, pre_summary = _pretrack(client, normalized, plan, capture_fps)
        if pre_summary:
            safe_emit({"type": "trace", "step": 0, "kind": "tool", "name": "pre_track",
                       "args": {}, "summary": pre_summary, "title": pre_summary, "ms": 0})
        agent_result = run_agent(client, normalized, plan, question, nf,
                                 fps=capture_fps, width=int(info.width), height=int(info.height),
                                 emit=agent_emit, pre_tracked=pre_tracked, pre_summary=pre_summary)
        # Ensure tracks/fits/calibration/answer are in the result even if the
        # agent emitted nothing (fallback emits too, but be safe).
        if agent_result.tracks and not result["tracks"]:
            for i, tr in enumerate(agent_result.tracks):
                tj = _track_json(tr, PALETTE[i % len(PALETTE)])
                result["tracks"].append(tj)
                safe_emit({"type": "track", "track": tj})
        if agent_result.fits and not result["fits"]:
            for i, fit in enumerate(agent_result.fits):
                fj = _fit_json(fit, i)
                result["fits"].append(fj)
                safe_emit({"type": "fit", "fit": fj})
        if agent_result.calibration is not None:
            result["calibration"] = agent_result.calibration.to_dict()
            safe_emit({"type": "calibration", "calibration": result["calibration"]})
        elif result["calibration"] is None:
            result["calibration"] = Calibration(1.0, 0.0, "none", "").to_dict()
        result["trace"] = agent_result.trace or result["trace"]
        result["answer"] = agent_result.answer or result["answer"]
        if result["answer"]:
            safe_emit({"type": "answer", "answer": result["answer"]})
        result["cost"] = cost_now()
        _write_result(job_dir, result)

        # 3b. strobe composite (from cached SAM masks; never calls the API)
        try:
            from physicam.strobe import make_strobe
            data_dir = getattr(getattr(client, "settings", None), "data_dir", "data") if client is not None else "data"
            all_tracks = list(agent_result.tracks) if agent_result.tracks else []
            fit_ids = {getattr(f, "track_id", None) for f in (agent_result.fits or [])}
            strobe_tracks = [t for t in all_tracks if t.id in fit_ids][:2] or all_tracks[:1]
            if strobe_tracks:
                spath = make_strobe(normalized, strobe_tracks, job_dir, str(data_dir),
                                    fits=[f.to_dict() if hasattr(f, "to_dict") else f for f in (agent_result.fits or [])],
                                    width=int(info.width), height=int(info.height))
                if spath is not None:
                    result["strobe_url"] = f"/media/{job_id}/strobe.jpg"
                    _write_result(job_dir, result)
                    safe_emit({"type": "strobe", "url": result["strobe_url"],
                               "strobe_url": result["strobe_url"]})
        except Exception:
            pass

        # 4. report --------------------------------------------------------
        stage("report")
        from physicam.report import write_report
        try:
            report_md = write_report(client, result) if client is not None else ""
        except Exception:
            report_md = ""
        if not report_md:
            from physicam.report import _template_report
            report_md = _template_report(result)
        result["report_markdown"] = report_md
        (job_dir / "report.md").write_text(report_md)
        result["cost"] = cost_now()
        _write_result(job_dir, result)
        safe_emit({"type": "report", "report_markdown": report_md})

        # 5. done -----------------------------------------------------------
        result["status"] = "done"
        result["cost"] = _finalize_cost()
        _write_result(job_dir, result)
        safe_emit({"type": "done"})
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        try:
            result["cost"] = cost_now()
        except Exception:
            pass
        _write_result(job_dir, result)
        safe_emit({"type": "error", "error": result["error"]})
        return result
    finally:
        if owned_client is not None and owned_client:
            try:
                client.close()  # type: ignore[union-attr]
            except Exception:
                pass
