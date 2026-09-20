"""A lock on an output directory, so two runs cannot overwrite each other's results.

Two screening runs once overlapped on `outputs/pupil/classical/`. Both wrote the
same per-video CSVs and the same summary, and the summary was read between the
two — producing a ranking that mixed a pre-fix run with a post-fix one. The
aggregates happened to survive; the ranking did not.

The guard is a lock file created atomically with `O_CREAT | O_EXCL`, carrying the
holder's pid, start time and command line. A holder touches it periodically; a
lock whose heartbeat has gone quiet for longer than `stale_after_s` is treated as
abandoned and taken over, with a warning, so a crashed run cannot block the
directory for ever.

Liveness is judged by heartbeat rather than by probing the pid, because pid
checks are not portable and a recycled pid would be worse than a stale timestamp.

Pure stdlib, no network (CLAUDE.md rule 2).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_STALE_AFTER_S = 300.0
LOCK_NAME = "run.lock"


class RunLockBusy(RuntimeError):
    """Raised when another live run holds the output directory."""

    def __init__(self, path: Path, holder: dict, age_s: float):
        self.path = path
        self.holder = holder
        self.age_s = age_s
        started = holder.get("started_at", "unknown")
        pid = holder.get("pid", "?")
        cmd = " ".join(holder.get("argv", [])) or "unknown command"
        super().__init__(
            f"{path.parent} is already in use by another run.\n"
            f"  pid        : {pid}\n"
            f"  started    : {started}\n"
            f"  command    : {cmd}\n"
            f"  last active: {age_s:.0f}s ago\n"
            f"Wait for it to finish, or write elsewhere with --out-dir."
        )


class RunLock:
    """Exclusive claim on a directory for the lifetime of a run."""

    def __init__(self, directory: Path | str, stale_after_s: float = DEFAULT_STALE_AFTER_S,
                 name: str = LOCK_NAME):
        self.directory = Path(directory)
        self.path = self.directory / name
        self.stale_after_s = stale_after_s
        self._held = False
        self.took_over_stale: dict | None = None

    def acquire(self) -> "RunLock":
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            self._create()
            return self
        except FileExistsError:
            pass

        holder, age = self._holder()
        if age is not None and age <= self.stale_after_s:
            raise RunLockBusy(self.path, holder, age)

        # Abandoned: the previous holder stopped touching the lock.
        self.took_over_stale = holder
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        try:
            self._create()
        except FileExistsError as exc:                      # another run won the race
            holder, age = self._holder()
            raise RunLockBusy(self.path, holder, age if age is not None else 0.0) from exc
        return self

    def heartbeat(self) -> None:
        """Mark the run as still alive. Cheap enough to call after every item."""
        if not self._held:
            return
        try:
            os.utime(self.path, None)
        except FileNotFoundError:
            pass

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            if self._is_ours():
                self.path.unlink()
        except FileNotFoundError:
            pass

    def _create(self) -> None:
        payload = json.dumps({
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "argv": sys.argv,
        }, indent=2).encode()
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        self._held = True

    def _holder(self) -> tuple[dict, float | None]:
        try:
            holder = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            holder = {}
        try:
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            age = None
        return holder, age

    def _is_ours(self) -> bool:
        holder, _ = self._holder()
        return holder.get("pid") == os.getpid()

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
