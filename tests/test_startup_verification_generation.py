"""BIBLE P1 for the server generation, and a cheap worker boot (owner, 14 Oct 23:35: Q1=A).

Every worker inherits the server's custody session id. The first process of a generation
(and of each checked-out code version) runs ``verify_restart`` and ``verify_system_state``
and stamps ``state/startup_verification.json``; a pool re-grow honours that stamp instead
of repeating the generation checks (2-20 s per boot in production on 13-14 Oct). Each
process still writes its own ``worker_boot`` row.

A worker also caches its bytecode, but only into a prefix inside its data dir (the
launcher's ``state/pycache.noindex``): never beside a source, so never into a signed bundle (WA6),
and never into a prefix someone else set up (a developer's, this test harness's).
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import agent as agent_module

REPO = pathlib.Path(__file__).resolve().parents[1]


class _Env:
    def __init__(self, root):
        self.repo_dir = self.drive_root = root

    def drive_path(self, name):
        path = self.repo_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path


@pytest.fixture
def boot(tmp_path, monkeypatch):
    """Start one fresh process of a generation (its own once-per-process flag) and record the checks."""
    env, checks = _Env(tmp_path), []
    monkeypatch.setattr(agent_module, "verify_restart", lambda _env, sha: checks.append(("restart", sha)))
    monkeypatch.setattr(agent_module, "verify_system_state", lambda _env, sha: checks.append(("system", sha)))

    (tmp_path / "state").mkdir()

    def start(generation="generation-1", sha="a" * 40):
        monkeypatch.setattr(agent_module, "_worker_boot_logged", False)
        monkeypatch.setattr(agent_module, "current_custody_session_id", lambda: generation)
        monkeypatch.setattr(agent_module, "get_git_info", lambda _repo: ("main", sha))
        agent_module.OuroborosAgent._log_worker_boot_once(SimpleNamespace(env=env))

    def boots():
        path = tmp_path / "logs" / "events.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        return [row for row in rows if row.get("type") == "worker_boot"]

    return SimpleNamespace(start=start, checks=checks, boots=boots,
                           receipt=tmp_path / "state" / "startup_verification.json")


def test_the_first_process_of_a_generation_verifies_and_stamps(boot):
    boot.start()
    assert boot.checks == [("restart", "a" * 40), ("system", "a" * 40)]
    stamp = json.loads(boot.receipt.read_text(encoding="utf-8"))
    assert stamp["generation"] == "generation-1" and stamp["git_sha"] == "a" * 40


def test_a_regrown_worker_of_the_same_generation_and_code_honours_the_stamp(boot):
    boot.start()
    del boot.checks[:]
    for _ in range(3):
        boot.start()
    assert boot.checks == [], "the generation's checks are not repeated per process"
    assert len(boot.boots()) == 4, "every process still records its own worker_boot row"


@pytest.mark.parametrize("change", ["generation", "code_version"])
def test_a_new_generation_or_code_version_verifies_again(boot, change):
    boot.start()
    del boot.checks[:]
    boot.start(**({"generation": "generation-2"} if change == "generation" else {"sha": "b" * 40}))
    assert [name for name, _sha in boot.checks] == ["restart", "system"]


def test_a_failed_verification_stamps_nothing_and_the_next_process_verifies(boot, monkeypatch):
    def broken(_env, _sha):
        raise OSError("state unreadable")

    monkeypatch.setattr(agent_module, "verify_system_state", broken)
    boot.start()
    assert not boot.receipt.exists()
    monkeypatch.setattr(agent_module, "verify_system_state", lambda _env, sha: boot.checks.append(("system", sha)))
    boot.start()
    assert ("system", "a" * 40) in boot.checks and boot.receipt.exists()


def test_an_unreadable_or_unknown_stamp_verifies_again(boot):
    boot.receipt.parent.mkdir(parents=True, exist_ok=True)
    boot.receipt.write_text("{not json", encoding="utf-8")
    boot.start()
    assert [name for name, _sha in boot.checks] == ["restart", "system"]
    del boot.checks[:]
    boot.start(generation="")  # no known generation: nothing can be honoured
    assert [name for name, _sha in boot.checks] == ["restart", "system"]


@pytest.mark.parametrize("prefix", ["data_dir", "elsewhere", "none"])
def test_a_worker_writes_bytecode_only_into_a_prefix_inside_its_data_dir(tmp_path, prefix):
    """A real interpreter started with -B (as spawn starts every worker under the launcher's
    PYTHONDONTWRITEBYTECODE): with the launcher's prefix (inside the data dir) the worker's own
    later imports are cached there; a prefix elsewhere (a developer's, this harness's) or none
    writes nothing anywhere."""
    data = tmp_path / "data"
    pycache = {"data_dir": data / "state" / "pycache.noindex", "elsewhere": tmp_path / "pycache"}.get(prefix)
    module = REPO / "ouroboros" / "preflight_receipts.py"
    beside = sorted((module.parent / "__pycache__").glob("preflight_receipts.*.pyc"))
    # An empty value is "no prefix" to Python (the test harness refills an absent one).
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPO),
           "PYTHONPYCACHEPREFIX": str(pycache or "")}
    probe = ("import sys; import supervisor.worker_process as w; "
             f"w.write_bytecode_to_prefix_only({str(data)!r}); "
             "print(int(sys.dont_write_bytecode)); import ouroboros.preflight_receipts")
    done = subprocess.run([sys.executable, "-B", "-c", probe], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == ("0" if prefix == "data_dir" else "1")
    cached = sorted(tmp_path.rglob("preflight_receipts.*.pyc"))
    assert bool(cached) is (prefix == "data_dir")
    assert all(path.is_relative_to(data / "state" / "pycache.noindex") for path in cached)
    assert sorted((module.parent / "__pycache__").glob("preflight_receipts.*.pyc")) == beside, \
        "nothing is ever written beside the source"


def test_a_worker_handed_an_older_launchers_prefix_moves_it_before_writing(tmp_path):
    """A worker under a legacy ``state/pycache`` (its own and the one its children inherit) writes only unindexed."""
    data = tmp_path / "data"
    legacy, new = data / "state" / "pycache", data / "state" / "pycache.noindex"
    legacy.mkdir(parents=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPO), "PYTHONPYCACHEPREFIX": str(legacy)}
    probe = ("import os, sys; import supervisor.worker_process as w; "
             f"w.write_bytecode_to_prefix_only({str(data)!r}); "
             "print(sys.pycache_prefix); print(os.environ['PYTHONPYCACHEPREFIX']); import ouroboros.preflight_receipts")
    done = subprocess.run([sys.executable, "-B", "-c", probe], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[-2:] == [str(new), str(new)], "the old worker wrote into the legacy tree"
    cached = sorted(tmp_path.rglob("preflight_receipts.*.pyc"))
    assert cached and all(path.is_relative_to(new) for path in cached) and not legacy.exists()
