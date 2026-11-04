"""Paths and options. Everything is overridable so tests and deployments can point elsewhere."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from upliftiq.data.registry import PROJECT_ROOT

DEFAULT_CORS = ("http://localhost:3000", "http://127.0.0.1:3000")


def _cors_from_env() -> tuple[str, ...]:
    raw = os.environ.get("UPLIFTIQ_CORS_ORIGINS")
    return tuple(o.strip() for o in raw.split(",") if o.strip()) if raw else DEFAULT_CORS


@dataclass(frozen=True)
class Settings:
    root: Path = PROJECT_ROOT
    model_dir: Path | None = None  # package directory; default models/criteo_s_learner/1.0.0
    mlflow_dir: Path | None = None  # default <root>/mlruns
    cors_origins: tuple[str, ...] = field(default_factory=_cors_from_env)
    max_batch: int = 10_000  # public-demo cap on records per request

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def package_dir(self) -> Path:
        return self.model_dir or self.root / "models" / "criteo_s_learner" / "1.0.0"

    @property
    def tracking_dir(self) -> Path:
        return self.mlflow_dir or self.root / "mlruns"
