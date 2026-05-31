"""Append-only experiment log (DESIGN s7). Each iteration records hypothesis, the
edited files' hashes, full metrics, and whether the edit was kept or reverted. This
is the audit trail and lets a run be replayed/resumed.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


@dataclass
class LogEntry:
    iteration: int
    hypothesis: str
    file_hashes: dict[str, str]
    metrics: dict
    kept: bool
    best_profit: float
    wall_time: float
    note: str = ""


class ExperimentLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: LogEntry) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    @staticmethod
    def file_hashes(files: dict[str, str]) -> dict[str, str]:
        return {k: _hash(v) for k, v in files.items()}

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in open(self.path) if l.strip()]
