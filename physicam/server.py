"""FastAPI app: REST + SSE + static web UI."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from physicam.config import get_settings

_jobs_lock = threading.Lock()
_subscribers: dict[str, list[queue.Queue]] = {}
_runners: dict[str, threading.Thread] = {}


def jobs_root() -> Path:
    return Path(get_settings().data_dir) / "jobs"


def examples_root() -> Path:
    return Path("examples")


def find_job_dir(job_id: str) -> Path | None:
    """Search data/jobs then examples/jobs for *job_id*."""
    for base in (jobs_root(), examples_root() / "jobs"):
        p = base / job_id
        if p.exists():
            return p
    return None


def _read_result(job_dir: Path, job_id: str) -> dict:
    rp = job_dir / "result.json"
    if rp.exists():
        try:
            return json.loads(rp.read_text())
        except Exception:
            pass
    return {"id": job_id, "status": "running", "question": "", "video": {}, "plan": None,
            "calibration": None, "tracks": [], "fits": [], "answer": None,
            "report_markdown": "", "trace": [], "cost": {}, "error": None}


def make_emit(job_id: str, job_dir: Path):
    """Emit function that persists events.jsonl and broadcasts to subscribers."""
    def emit(ev: dict) -> None:
        # Persistence to events.jsonl is owned by pipeline.run_job.
        with _jobs_lock:
            subs = list(_subscribers.get(job_id, []))
        for q in subs:
            try:
                q.put(ev)
            except Exception:
                pass
    return emit


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(title="Physicam")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/jobs")
    async def create_job(video: UploadFile = File(...), question: str = Form("What is g?"),
                         facts: str | None = Form(None)):
        job_id = uuid.uuid4().hex[:12]
        job_dir = jobs_root() / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
        src = job_dir / f"source{suffix}"
        data = await video.read()
        src.write_bytes(data)
        facts_dict: dict = {}
        if facts:
            try:
                facts_dict = json.loads(facts)
            except Exception:
                facts_dict = {}
        initial = {"id": job_id, "status": "running", "question": question, "video": {},
                   "plan": None, "calibration": None, "tracks": [], "fits": [], "answer": None,
                   "report_markdown": "", "trace": [], "cost": {}, "error": None}
        (job_dir / "result.json").write_text(json.dumps(initial, indent=1))
        emit = make_emit(job_id, job_dir)

        def run() -> None:
            from physicam.pipeline import run_job
            run_job(job_dir, src, question, facts_dict, emit=emit)

        thread = threading.Thread(target=run, daemon=True)
        with _jobs_lock:
            _runners[job_id] = thread
        thread.start()
        return {"id": job_id}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job_dir = find_job_dir(job_id)
        if job_dir is None:
            raise HTTPException(404, "job not found")
        return _read_result(job_dir, job_id)

    @app.get("/api/jobs/{job_id}/level.csv")
    def get_level_csv(job_id: str):
        job_dir = find_job_dir(job_id)
        if job_dir is None:
            raise HTTPException(404, "job not found")
        path = job_dir / "level.csv"
        if path.exists():
            return FileResponse(path, media_type="text/csv",
                                headers={"Content-Disposition": f'attachment; filename="{job_id}-level.csv"'})
        result = _read_result(job_dir, job_id)
        level = result.get("level")
        if not level:
            raise HTTPException(404, "no level data for this job")
        import csv as _csv
        import io as _io
        buf = _io.StringIO()
        w = _csv.writer(buf)
        w.writerow(["t_s", "h_px", "h_m", "surface_y_px"])
        ts, hs = level.get("t") or [], level.get("h") or level.get("h_px") or []
        hms = level.get("h_m") or [None] * len(ts)
        sys_ = level.get("surface_y") or [None] * len(ts)
        cal = result.get("calibration") or {}
        mpp = cal.get("m_per_px") if cal.get("method") not in (None, "none") else None
        unit = level.get("unit")
        for i in range(min(len(ts), len(hs))):
            h = hs[i]
            if unit == "m":
                h_px = (h / mpp) if mpp else ""
                h_m = h
            else:
                h_px = h
                h_m = (h * mpp) if mpp else (hms[i] if i < len(hms) else "")
            w.writerow([ts[i], h_px, h_m if h_m is not None else "", sys_[i] if i < len(sys_) else ""])
        from fastapi.responses import Response as _Resp
        return _Resp(content=buf.getvalue(), media_type="text/csv",
                     headers={"Content-Disposition": f'attachment; filename="{job_id}-level.csv"'})

    @app.get("/api/jobs/{job_id}/events")
    def get_events(job_id: str):
        job_dir = find_job_dir(job_id)
        if job_dir is None:
            raise HTTPException(404, "job not found")
        q: queue.Queue = queue.Queue()
        with _jobs_lock:
            _subscribers.setdefault(job_id, []).append(q)

        def gen():
            try:
                # Replay persisted events first.
                past: list[dict] = []
                ef = job_dir / "events.jsonl"
                if ef.exists():
                    for line in ef.read_text().splitlines():
                        if line.strip():
                            try:
                                past.append(json.loads(line))
                            except Exception:
                                pass
                finished = any(e.get("type") in ("done", "error") for e in past)
                for ev in past:
                    yield f"data: {json.dumps(ev)}\n\n"
                if finished:
                    return
                # If the job already finished but events file was just written, re-check.
                result = _read_result(job_dir, job_id)
                if result.get("status") in ("done", "error"):
                    # Replay once more in case events landed after first read.
                    return
                last_heartbeat = time.time()
                while True:
                    try:
                        ev = q.get(timeout=1.0)
                        yield f"data: {json.dumps(ev)}\n\n"
                        if ev.get("type") in ("done", "error"):
                            return
                    except queue.Empty:
                        if time.time() - last_heartbeat >= 15:
                            yield ": heartbeat\n\n"
                            last_heartbeat = time.time()
                        # Exit if job finished without us seeing the event (restart case).
                        r = _read_result(job_dir, job_id)
                        if r.get("status") in ("done", "error"):
                            # Drain any queued events before closing.
                            while not q.empty():
                                ev = q.get()
                                yield f"data: {json.dumps(ev)}\n\n"
                            return
            finally:
                with _jobs_lock:
                    lst = _subscribers.get(job_id, [])
                    if q in lst:
                        lst.remove(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/media/{job_id}/video.mp4")
    def media_video(job_id: str, request: Request):
        job_dir = find_job_dir(job_id)
        if job_dir is None:
            raise HTTPException(404, "job not found")
        path = job_dir / "video.mp4"
        if not path.exists():
            # Example jobs may store the source under a different name.
            cands = sorted(job_dir.glob("*.mp4"))
            if not cands:
                raise HTTPException(404, "video not found")
            path = cands[0]
        return _range_response(path, request, "video/mp4")

    @app.get("/media/{job_id}/strobe.jpg")
    def media_strobe(job_id: str):
        job_dir = find_job_dir(job_id)
        if job_dir is None:
            raise HTTPException(404, "job not found")
        path = job_dir / "strobe.jpg"
        if not path.exists():
            raise HTTPException(404, "strobe not found")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/jobs/{job_id}/thumb.jpg")
    def thumb(job_id: str):
        job_dir = find_job_dir(job_id)
        if job_dir is None:
            raise HTTPException(404, "job not found")
        thumb_path = job_dir / "thumb.jpg"
        if thumb_path.exists():
            return FileResponse(thumb_path, media_type="image/jpeg")
        video = job_dir / "video.mp4"
        if not video.exists():
            cands = sorted(job_dir.glob("*.mp4"))
            if not cands:
                raise HTTPException(404, "video not found")
            video = cands[0]
        try:
            from physicam.video import extract_frame
            img = extract_frame(video, 0.0)
            img.save(thumb_path, format="JPEG", quality=85)
        except Exception as exc:
            raise HTTPException(500, f"thumbnail failed: {exc}") from exc
        return FileResponse(thumb_path, media_type="image/jpeg")

    @app.get("/api/examples")
    def examples():
        p = examples_root() / "examples.json"
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        return []

    @app.post("/api/transcribe")
    async def transcribe(audio: UploadFile = File(...)):
        import tempfile
        from physicam.meta_client import MetaClient
        from physicam.video import ffmpeg_exe

        data = await audio.read()
        suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / f"in{suffix}"
            src.write_bytes(data)
            wav = Path(td) / "out.wav"
            proc = subprocess.run(
                [ffmpeg_exe(), "-y", "-i", str(src), "-ar", "16000", "-ac", "1",
                 "-c:a", "pcm_s16le", str(wav)],
                capture_output=True, text=True, timeout=120)
            if proc.returncode != 0 or not wav.exists():
                raise HTTPException(500, f"audio conversion failed: {(proc.stderr or '')[-500:]}")
            client = getattr(app.state, "meta_client", None)
            if client is None:
                client = MetaClient()
            out = client.transcribe_wav(wav)
            transcript = out.get("transcript", "") if isinstance(out, dict) else ""
            return {"transcript": transcript}

    # Static web UI last so API routes win.
    web_dir = Path("web")
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    return app


def _range_response(path: Path, request: Request, media_type: str):
    """File response with HTTP Range (206) support."""
    size = path.stat().st_size
    range_header = request.headers.get("range") or request.headers.get("Range")
    if not range_header:
        return FileResponse(path, media_type=media_type,
                            headers={"Accept-Ranges": "bytes", "Content-Length": str(size)})
    # Parse single range: bytes=start-end
    try:
        unit, rng = range_header.split("=", 1)
        start_s, _, end_s = rng.partition("-")
        if start_s == "":
            length = int(end_s)
            start, end = max(0, size - length), size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            raise ValueError("bad range")
    except Exception:
        return FileResponse(path, media_type=media_type,
                            headers={"Accept-Ranges": "bytes", "Content-Length": str(size)})
    length = end - start + 1

    def iter_file():
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(iter_file(), status_code=206, media_type=media_type, headers={
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Accept-Ranges": "bytes", "Content-Length": str(length)})


app = create_app()
