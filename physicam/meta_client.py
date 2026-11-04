"""Sync httpx wrapper for the Meta Model API.

Every network call to Meta goes through this module (see AGENTS.md).
The API key is read from Settings and never logged.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

from physicam.config import Settings, get_settings


class CostAccumulator:
    """Simple usage/cost accumulator that pipeline code can read."""

    def __init__(self) -> None:
        self.sam_frames: int = 0
        self.tokens: dict[str, dict[str, int]] = {}
        # SAM pricing: $0.20 per 1000 frames, minimum 50 per request (already in billed frames).
        self._sam_usd_per_frame = 0.20 / 1000.0
        self._token_prices: dict[str, tuple[float, float]] = {
            # (input, output) USD per million tokens; only Wonder priced in AGENTS.md.
            "muse-wonder-1.0": (4.0, 20.0),
        }

    def add_sam_frames(self, n: int) -> None:
        self.sam_frames += int(n)

    def add_tokens(self, model: str, usage: dict | None) -> None:
        if not usage:
            return
        slot = self.tokens.setdefault(model, {"input": 0, "output": 0, "total": 0})
        slot["input"] += int(usage.get("input_tokens", 0) or 0)
        slot["output"] += int(usage.get("output_tokens", 0) or 0)
        slot["total"] += int(usage.get("total_tokens", 0) or 0)

    def usd_estimate(self) -> float:
        usd = self.sam_frames * self._sam_usd_per_frame
        for model, t in self.tokens.items():
            if model in self._token_prices:
                pin, pout = self._token_prices[model]
                usd += t["input"] / 1e6 * pin + t["output"] / 1e6 * pout
        return round(usd, 4)

    def to_dict(self) -> dict:
        total_tokens = sum(t["total"] for t in self.tokens.values())
        return {
            "sam_frames": self.sam_frames,
            "spark_tokens": self.tokens.get("muse-spark-1.3", {}).get("total", 0),
            "tokens_by_model": dict(self.tokens),
            "total_tokens": total_tokens,
            "usd_estimate": self.usd_estimate(),
        }


class MetaClient:
    """Synchronous Meta API client with retry/backoff on 429/5xx/timeouts."""

    def __init__(self, settings: Settings | None = None, *, transport: httpx.BaseTransport | None = None,
                 timeout: float = 600.0) -> None:
        self.settings = settings or get_settings()
        self.cost = CostAccumulator()
        self._client = httpx.Client(base_url=self.settings.base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MetaClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.require_key()}"}

    def _post_with_retry(self, path: str, **kwargs: Any) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                resp = self._client.post(path, headers=self._headers(), **kwargs)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt == 3:
                    raise
                time.sleep(min(2 ** attempt * 1.0, 30.0))
                continue
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                wait = min(2 ** attempt * 1.0, 30.0)
                ra = resp.headers.get("retry-after")
                if ra:
                    try:
                        wait = max(wait, float(ra))
                    except ValueError:
                        pass
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("request failed after retries")  # pragma: no cover

    # -- Files ---------------------------------------------------------------
    def _files_cache_path(self) -> Path:
        p = Path(self.settings.data_dir) / "cache" / "files.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def upload_file(self, path: str | Path) -> str:
        """Upload *path* and return its file id, cached by sha256 (ids do not expire)."""
        from physicam.video import sha256_file

        p = Path(path)
        digest = sha256_file(p)
        cache_path = self._files_cache_path()
        cache: dict[str, str] = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
            except (json.JSONDecodeError, OSError):
                cache = {}
        if digest in cache:
            return cache[digest]
        data = p.read_bytes()
        mime = "video/mp4" if p.suffix.lower() in (".mp4", ".mov", ".m4v") else "application/octet-stream"
        resp = self._post_with_retry("/files", files={"file": (p.name, data, mime)},
                                     data={"purpose": "user_data"})
        file_id = resp.json()["id"]
        cache[digest] = file_id
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1))
        tmp.replace(cache_path)
        return file_id

    # -- Responses -----------------------------------------------------------
    def responses(self, body: dict) -> dict:
        """POST /responses with retries; accumulates token/frame usage."""
        resp = self._post_with_retry("/responses", json=body)
        out = resp.json()
        model = str(body.get("model", ""))
        u = out.get("usage") or {}
        if u:
            self.cost.add_tokens(model, u)
            vf = u.get("video_frames_processed")
            if vf:
                self.cost.add_sam_frames(int(vf))
        return out

    @staticmethod
    def output_text(resp: dict) -> str:
        """Concatenate all output_text parts of a Responses result."""
        parts: list[str] = []
        for item in resp.get("output", []) or []:
            for content in item.get("content", []) or []:
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(content["text"])
        return "".join(parts)

    @staticmethod
    def function_calls(resp: dict) -> list[dict]:
        """Return function-call items: [{name, arguments, call_id}]."""
        calls: list[dict] = []
        for item in resp.get("output", []) or []:
            if item.get("type") == "function_call":
                calls.append({"name": item.get("name", ""), "arguments": item.get("arguments", "{}"),
                              "call_id": item.get("call_id", "")})
        return calls

    @staticmethod
    def usage(resp: dict) -> dict:
        """Return the usage dict of a Responses result (or {})."""
        return dict(resp.get("usage") or {})

    # -- Voice ---------------------------------------------------------------
    def transcribe_wav(self, path: str | Path, mode: str = "PUSH_TO_TALK") -> dict:
        """Transcribe a mono 16-bit PCM 16 kHz WAV via POST /asr/transcribe."""
        p = Path(path)
        request = {"mode": mode, "model": self.settings.asr, "audioEncoding": "WAV"}
        resp = self._post_with_retry(
            "/asr/transcribe",
            files={
                "request": ("request.json", json.dumps(request).encode(), "application/json"),
                "audio": (p.name, p.read_bytes(), "audio/wav"),
            },
        )
        return resp.json()
