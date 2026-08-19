from __future__ import annotations

import sys
from types import TracebackType

from tqdm.auto import tqdm


class ModuleProgress:
    def __init__(
        self,
        module_name: str,
        total: int,
        enabled: bool | None = None,
    ) -> None:
        self.enabled = sys.stderr.isatty() if enabled is None else enabled
        self._bar = tqdm(
            total=total,
            desc=module_name,
            unit="step",
            disable=not self.enabled,
        )

    def __enter__(self) -> ModuleProgress:
        self._bar.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self._bar.__exit__(exc_type, exc, traceback)

    def step(self, label: str) -> None:
        self._bar.set_postfix_str(label)
        self._bar.update(1)
