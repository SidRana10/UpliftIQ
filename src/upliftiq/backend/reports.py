"""Read-only access to committed reports. Nothing is recomputed; missing files raise clearly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class ReportUnavailable(RuntimeError):
    """A report file the endpoint depends on is missing or unreadable."""

    def __init__(self, relative: str):
        super().__init__(f"report not available: {relative}")
        self.relative = relative


class Reports:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, relative: str) -> Path:
        p = (self.root / relative).resolve()
        if not p.is_relative_to(self.root.resolve()) or not p.is_file():
            raise ReportUnavailable(relative)
        return p

    def json(self, relative: str) -> Any:
        try:
            return json.loads(self.path(relative).read_text())
        except (OSError, ValueError):
            raise ReportUnavailable(relative) from None

    def csv(self, relative: str) -> list[dict]:
        try:
            df = pd.read_csv(self.path(relative))
        except (OSError, ValueError, pd.errors.ParserError):
            raise ReportUnavailable(relative) from None
        return json.loads(df.to_json(orient="records"))  # NaN -> null, numpy -> python

    def exists(self, relative: str) -> bool:
        try:
            self.path(relative)
            return True
        except ReportUnavailable:
            return False
