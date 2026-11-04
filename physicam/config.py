"""Settings loaded from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings. The API key never leaves this object except as a header."""

    api_key: str = ""
    base_url: str = "https://api.meta.ai/v1"
    spark: str = "muse-spark-1.3"
    wonder: str = "muse-wonder-1.0"
    sam: str = "sam-3.1"
    asr: str = "muse-voice-transcribe-1.0"
    data_dir: str = "data"

    @classmethod
    def from_env(cls, env: dict | None = None) -> "Settings":
        """Load settings from *env* (defaults to ``os.environ``)."""
        e = os.environ if env is None else env
        key = e.get("MODEL_API_KEY") or e.get("META_API_KEY") or ""
        return cls(
            api_key=key,
            base_url=e.get("META_BASE_URL", "https://api.meta.ai/v1").rstrip("/"),
            data_dir=e.get("PHYSICAM_DATA_DIR", "data"),
        )

    def require_key(self) -> str:
        """Return the API key or raise a clear error."""
        if not self.api_key:
            raise RuntimeError("no API key: set MODEL_API_KEY (or META_API_KEY) in the environment")
        return self.api_key


def get_settings() -> Settings:
    """Convenience accessor used across the package."""
    return Settings.from_env()
