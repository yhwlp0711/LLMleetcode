"""Load user submissions safely with timeout and isolated module namespace."""

from __future__ import annotations

import importlib.util
import signal
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType


class SubmissionLoadError(Exception):
    pass


class TimeoutError_(Exception):
    pass


def load_module_from_path(path: Path, module_name: str | None = None) -> ModuleType:
    """Load a Python file as a fresh module (does not pollute sys.modules permanently)."""
    name = module_name or f"_mlleetcode_user_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SubmissionLoadError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise SubmissionLoadError(f"Error while importing submission: {exc!r}") from exc
    finally:
        # The returned module object stays alive via the caller's reference; we
        # drop it from sys.modules so repeated loads (e.g. the long-running web
        # server) don't leak a module per submission.
        sys.modules.pop(name, None)
    return module


@contextmanager
def time_limit(seconds: float):
    """SIGALRM-based timeout. Unix only; on unsupported platforms or non-main threads this is a no-op."""
    import threading

    if (
        not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError_(f"Execution exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
