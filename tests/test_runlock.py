"""Tests for the output-directory lock."""
from __future__ import annotations

import json
import os
import time

import pytest

from phacoguard.runlock import RunLock, RunLockBusy


def test_acquire_creates_a_lock_file(tmp_path):
    with RunLock(tmp_path) as lock:
        assert lock.path.exists()
        holder = json.loads(lock.path.read_text(encoding="utf-8"))
        assert holder["pid"] == os.getpid()
    assert not lock.path.exists(), "the lock is released on exit"


def test_second_run_is_refused(tmp_path):
    with RunLock(tmp_path):
        with pytest.raises(RunLockBusy) as exc:
            RunLock(tmp_path).acquire()
    assert "already in use" in str(exc.value)
    assert "--out-dir" in str(exc.value), "the message should say how to proceed"


def test_release_allows_a_later_run(tmp_path):
    RunLock(tmp_path).acquire().release()
    with RunLock(tmp_path):
        pass
    assert not (tmp_path / "run.lock").exists()


def test_lock_is_released_when_the_run_raises(tmp_path):
    with pytest.raises(ValueError):
        with RunLock(tmp_path):
            raise ValueError("boom")
    RunLock(tmp_path).acquire().release()   # would raise RunLockBusy if still held


def test_abandoned_lock_is_taken_over(tmp_path):
    stale = RunLock(tmp_path)
    stale.acquire()
    old = time.time() - 10_000
    os.utime(stale.path, (old, old))
    taken = RunLock(tmp_path, stale_after_s=300).acquire()
    assert taken.took_over_stale is not None
    assert taken.took_over_stale["pid"] == os.getpid()
    taken.release()


def test_fresh_heartbeat_keeps_the_lock_held(tmp_path):
    held = RunLock(tmp_path)
    held.acquire()
    old = time.time() - 10_000
    os.utime(held.path, (old, old))
    held.heartbeat()                        # holder is still alive
    with pytest.raises(RunLockBusy):
        RunLock(tmp_path, stale_after_s=300).acquire()
    held.release()


def test_a_released_lock_does_not_delete_someone_elses(tmp_path):
    first = RunLock(tmp_path)
    first.acquire()
    first._held = False                     # simulate a lost handle
    second_holder = {"pid": os.getpid() + 1, "started_at": "x", "argv": []}
    first.path.write_text(json.dumps(second_holder), encoding="utf-8")
    first.release()
    assert first.path.exists(), "release must not remove another run's lock"


def test_corrupt_lock_file_is_survivable(tmp_path):
    lock = RunLock(tmp_path)
    lock.directory.mkdir(parents=True, exist_ok=True)
    lock.path.write_text("not json", encoding="utf-8")
    with pytest.raises(RunLockBusy):
        RunLock(tmp_path, stale_after_s=300).acquire()
