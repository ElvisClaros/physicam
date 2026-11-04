"""Command-line interface: probe / track / fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_probe(args) -> int:
    from physicam.video import probe

    info = probe(args.video)
    print(json.dumps(info.to_dict(), indent=2))
    return 0


def _cmd_track(args) -> int:
    from physicam.config import get_settings
    from physicam.meta_client import MetaClient
    from physicam.sam import segment_video
    from physicam.video import probe, normalize

    settings = get_settings()
    video_path = Path(args.video)
    info = probe(video_path)
    src = video_path
    tmp_norm: Path | None = None
    if args.t0 is not None or args.t1 is not None:
        tmp_norm = Path(settings.data_dir) / "cache" / f"clip_{video_path.stem}.mp4"
        normalize(video_path, tmp_norm, t_start=args.t0, t_end=args.t1)
        src = tmp_norm
        info = probe(src)
    with MetaClient(settings) as client:
        tracks = segment_video(client, src, args.prompt, info.fps)
        cost = client.cost.to_dict()
    for tr in tracks:
        print(json.dumps(tr.summary()))
    out_dir = Path(settings.data_dir) / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{video_path.stem}_{args.prompt.replace(' ', '_')}"
    out_path = out_dir / f"{name}.json"
    payload = {"video": str(video_path), "prompt": args.prompt, "fps": info.fps,
               "cost": cost, "tracks": [tr.to_json() for tr in tracks]}
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out_path} ({len(tracks)} track(s))", file=sys.stderr)
    return 0


def _load_tracks(path: str) -> tuple[list, dict]:
    from physicam.sam import Track

    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "tracks" in data:
        return [Track.from_json(d) for d in data["tracks"]], data
    if isinstance(data, list):
        return [Track.from_json(d) for d in data], {}
    return [Track.from_json(data)], {}


def _cmd_fit(args) -> int:
    from physicam.calibrate import Calibration
    from physicam.physics import fit_model

    tracks, _meta = _load_tracks(args.tracks)
    by_id = {tr.id: tr for tr in tracks}
    if args.track_id not in by_id:
        print(f"track {args.track_id!r} not found; available: {sorted(by_id)}", file=sys.stderr)
        return 2
    tr = by_id[args.track_id]
    calib = Calibration(args.m_per_px, 0.05 if args.m_per_px != 1.0 else 0.0,
                         "reference" if args.m_per_px != 1.0 else "none",
                         "from --m-per-px" if args.m_per_px != 1.0 else "")
    if args.m_per_px == 1.0:
        calib = Calibration(1.0, 0.0, "none", "")
    t_range = None
    if args.t0 is not None or args.t1 is not None:
        t_range = [args.t0 if args.t0 is not None else float(tr.t[0]),
                   args.t1 if args.t1 is not None else float(tr.t[-1])]
    fit = fit_model(tr, args.model, calib, t_range=t_range)
    print(json.dumps(fit.to_dict(), indent=2))
    return 0


def _cmd_analyze(args) -> int:
    import hashlib
    from physicam.pipeline import run_job

    facts = json.loads(args.facts) if args.facts else {}
    if args.job_dir:
        job_dir = Path(args.job_dir)
    else:
        digest = hashlib.sha256((str(Path(args.video).resolve()) + args.question).encode()).hexdigest()[:12]
        job_dir = Path("data/jobs") / digest

    def emit(ev: dict) -> None:
        print(json.dumps(ev)[:2000], flush=True)

    result = run_job(job_dir, args.video, args.question, facts, emit=emit)
    ans = result.get("answer") or {}
    print(f"answer: {ans.get('headline', '(none)')}")
    print(f"report: {job_dir / 'report.md'}")
    print(f"result: {job_dir / 'result.json'}")
    return 0 if result.get("status") == "done" else 1


def _cmd_serve(args) -> int:
    import uvicorn
    uvicorn.run("physicam.server:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="physicam", description="Physics measurement from phone video")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("probe", help="probe a video file")
    pp.add_argument("video")
    pp.set_defaults(func=_cmd_probe)

    pt = sub.add_parser("track", help="track an object with SAM 3.1")
    pt.add_argument("video")
    pt.add_argument("prompt")
    pt.add_argument("--t0", type=float, default=None)
    pt.add_argument("--t1", type=float, default=None)
    pt.set_defaults(func=_cmd_track)

    pf = sub.add_parser("fit", help="fit a physics model to a tracked trajectory")
    pf.add_argument("tracks")
    pf.add_argument("track_id")
    pf.add_argument("model")
    pf.add_argument("--t0", type=float, default=None)
    pf.add_argument("--t1", type=float, default=None)
    pf.add_argument("--m-per-px", type=float, default=1.0)
    pf.set_defaults(func=_cmd_fit)

    pa = sub.add_parser("analyze", help="run the full pipeline on a video")
    pa.add_argument("video")
    pa.add_argument("question")
    pa.add_argument("--facts", default=None, help="JSON string of facts")
    pa.add_argument("--job-dir", default=None)
    pa.set_defaults(func=_cmd_analyze)

    ps = sub.add_parser("serve", help="serve the web UI + API")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8000)
    ps.set_defaults(func=_cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
