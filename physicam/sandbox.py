"""Run model-written numpy/scipy code in a stripped subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_PRELUDE = '''
import json
import numpy as np
import scipy
import scipy.optimize as _opt

class _ArrDict(dict):
    __getattr__ = dict.get

_data = np.load("tracks.npz")
with open("meta.json") as _fh:
    _meta = json.load(_fh)
tracks = {}
for _tid in _meta["track_ids"]:
    tracks[_tid] = _ArrDict({k: _data[f"{_tid}__{k}"] for k in ("t", "x", "y", "area") if f"{_tid}__{k}" in _data})
m_per_px = _meta["m_per_px"]
'''


def run_python(code: str, tracks: dict | None = None, m_per_px: float = 1.0,
               timeout: float = 20.0) -> dict:
    """Run *code* in a subprocess; return ``{ok, stdout, error}``.

    ``tracks`` maps id -> mapping with ``t, x, y, area`` arrays (or a Track).
    The environment is stripped of API keys; cwd is a temp directory.
    """
    tracks = tracks or {}
    env = {k: v for k, v in os.environ.items()
           if k not in ("MODEL_API_KEY", "META_API_KEY") and "API_KEY" not in k
           and not k.endswith("_TOKEN") and "SECRET" not in k}
    env.pop("MODEL_API_KEY", None)
    env.pop("META_API_KEY", None)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        arrays: dict[str, np.ndarray] = {}
        ids: list[str] = []
        for tid, tr in tracks.items():
            ids.append(str(tid))
            if hasattr(tr, "t"):
                vals = {"t": tr.t, "x": tr.x, "y": tr.y, "area": tr.area}
            else:
                vals = dict(tr)  # type: ignore[arg-type]
            for k in ("t", "x", "y", "area"):
                if k in vals:
                    arrays[f"{tid}__{k}"] = np.asarray(vals[k], dtype=float)
        if arrays:
            np.savez(td_path / "tracks.npz", **arrays)
        else:
            np.savez(td_path / "tracks.npz")
        (td_path / "meta.json").write_text(json.dumps({"track_ids": ids, "m_per_px": float(m_per_px)}))
        script = td_path / "script.py"
        script.write_text(_PRELUDE + "\n" + (code or "") + "\n")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=td, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")) if exc.stdout else ""
            return {"ok": False, "stdout": out[:8192], "error": f"timeout after {timeout}s"}
        stdout = (proc.stdout or "")[:8192]
        if proc.returncode != 0:
            err = (proc.stderr or "")[-4000:]
            return {"ok": False, "stdout": stdout, "error": err}
        return {"ok": True, "stdout": stdout, "error": ""}
