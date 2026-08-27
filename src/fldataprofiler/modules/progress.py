from __future__ import annotations

import sys
import threading
import time
from types import TracebackType

from tqdm.auto import tqdm


class ModuleProgress:
    def __init__(
        self,
        module_name: str,
        total: int,
        unit: str = "step",
        enabled: bool | None = None,
    ) -> None:
        self.enabled = sys.stderr.isatty() if enabled is None else enabled
        self._bar = tqdm(
            total=total,
            desc=module_name,
            unit=unit,
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

    def set_total(self, total: int) -> None:
        if hasattr(self._bar, "total"):
            self._bar.total = total
        if hasattr(self._bar, "kwargs") and isinstance(self._bar.kwargs, dict):
            self._bar.kwargs["total"] = total
        if hasattr(self._bar, "refresh"):
            self._bar.refresh()

    def step(self, label: str | None = None, n: int = 1) -> None:
        if label and hasattr(self._bar, "set_postfix_str"):
            display_label = label if len(label) <= 35 else label[:32] + "..."
            self._bar.set_postfix_str(display_label)
        if hasattr(self._bar, "update"):
            self._bar.update(n)


class StatusTimer:
    """A lightweight context manager that displays a message with a live ticking seconds counter in a background thread."""

    def __init__(self, message: str, enabled: bool | None = None) -> None:
        self.message = message
        self.enabled = sys.stderr.isatty() if enabled is None else enabled
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    def __enter__(self) -> StatusTimer:
        if not self.enabled:
            return self
        self._start_time = time.perf_counter()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            sys.stderr.write(f"\r{self.message}... (0s)")
            sys.stderr.flush()
            while not self._stop_event.wait(0.25):
                elapsed = int(time.perf_counter() - self._start_time)
                mins, secs = divmod(elapsed, 60)
                timer_str = f"{mins:02d}:{secs:02d}" if mins > 0 else f"{secs}s"
                sys.stderr.write(f"\r{self.message}... ({timer_str})")
                sys.stderr.flush()
        except Exception:
            pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if not self.enabled or self._thread is None:
            return None
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        try:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
        except Exception:
            pass
        return None
