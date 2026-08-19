from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ModuleResult:
    report_dir: Path
    artifacts: list[Path]


class ProfilingModule(Protocol):
    name: str

    def run(
        self,
        feature_csv: Path,
        label_csv: Path,
        output_dir: Path,
        join_key: str | None = None,
        targets: list[str] | None = None,
    ) -> ModuleResult:
        """Run a feature/label profiling module."""
