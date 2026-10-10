"""F1: the session history audit's real child, its stop races and its silence.

The audit exists to take an O(history) diagnostic OFF the readiness path.  What
must therefore hold is: a real separate interpreter does the work, the parent's
record is a closed bounded schema, stop wins from either side of handle
publication, an unreadable history stays UNKNOWN instead of becoming an
accusation, and the money the pass reads (the usage store and the retained
journal evidence) is not touched by it.
"""
from __future__ import annotations

import json
import os
import pathlib
import threading
import time

import pytest

from ouroboros import usage_store
from ouroboros.startup_historical_audit import HistoricalAudit, _report_fields
from tests._usage_store_testing import ledger_rows
from tests.fixtures_usage_store import (  # noqa: F401  (pytest fixtures)
    ARCHIVE_SEGMENT_REL,
    _seed_mixed_ledger,
    data_root as data_root,              # re-exported for pytest, not called here
    fold_into_archive,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
GATE = 120.0


def _records(root: pathlib.Path) -> list[dict]:
    path = root / "logs" / "supervisor.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("type") == "startup_historical_audit":
            rows.append(row)
    return rows


def _await_terminal(root: pathlib.Path, timeout: float = GATE) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for row in _records(root):
            if row.get("phase") in {"completed", "unknown", "failed", "stopped"}:
                return row
        time.sleep(0.05)
    raise AssertionError(f"no terminal audit record within {timeout}s: {_records(root)}")


def _seal_manifest(root: pathlib.Path, attempt_id: str, task_id: str = "t") -> None:
    directory = root / "observability" / "calls" / task_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{attempt_id}.json").write_text(json.dumps({
        "call_id": attempt_id,
        "task_id": task_id,
        "model_send_seal": {"attempt_id": attempt_id, "canonical_basis": "model_send_candidate_v1",
                            "pre_redaction_sha256": "0" * 64, "size_bytes": 1},
    }), encoding="utf-8")


@pytest.fixture()
def archived_root(data_root, monkeypatch):
    """An install upgraded from a compacted journal: folded attempts live only
    in a retained archive segment, the store holds their aggregate, plus seals
    over an archived identity, a live identity and an absent one."""
    (data_root / "logs").mkdir(parents=True, exist_ok=True)
    _seed_mixed_ledger(data_root)
    rows_before = ledger_rows(data_root)
    archived_ids = [str(row["attempt_id"]) for row in rows_before
                    if row.get("state") == "settled" and str(row.get("kind") or "attempt") == "attempt"
                    and not row.get("review_skill")]
    assert archived_ids
    fold_into_archive(data_root, archived_ids)
    live_ids = [str(row["attempt_id"]) for row in ledger_rows(data_root)  # the import runs here, as at boot
                if str(row.get("kind") or "attempt") == "attempt"]
    assert live_ids, "open attempts are never folded"
    folded = set(archived_ids) - set(live_ids)
    assert folded, "at least one identity must live only in the archive"
    _seal_manifest(data_root, sorted(folded)[0])
    _seal_manifest(data_root, live_ids[0])
    _seal_manifest(data_root, "absent-attempt-id")
    return data_root


def _money_bytes(root: pathlib.Path) -> list:
    paths = [root / usage_store.STORE_REL, root / usage_store.LEDGER_REL, root / ARCHIVE_SEGMENT_REL]
    return [(path.name, path.read_bytes()) for path in paths]


# --------------------------------------------------------------------------
# The real child
# --------------------------------------------------------------------------

def test_real_child_runs_the_history_pass_in_its_own_interpreter(archived_root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(archived_root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(archived_root / "settings.json"))
    money_before = _money_bytes(archived_root)

    audit = HistoricalAudit()
    audit.start(archived_root, REPO)
    terminal = _await_terminal(archived_root)

    started = next(row for row in _records(archived_root) if row["phase"] == "started")
    assert started["pid"] != os.getpid(), "the audit must not run inside the server process"
    assert terminal["phase"] == "completed", terminal
    assert terminal["exit_code"] == 0
    assert terminal["manifests_checked"] == 3
    # Exactly one seal names an identity that is in neither the store nor the
    # retained evidence; the archived one must NOT be accused.
    assert terminal["facts_written"] == 1, terminal
    assert terminal["cpu_seconds"] >= 0 and terminal["wall_seconds"] >= 0

    # The money the pass reads is untouched by it.
    assert _money_bytes(archived_root) == money_before


def test_recorded_facts_carry_no_paths_identities_or_messages(archived_root, monkeypatch):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(archived_root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(archived_root / "settings.json"))
    audit = HistoricalAudit()
    audit.start(archived_root, REPO)
    _await_terminal(archived_root)
    allowed = {"ts", "type", "phase", "pid", "exit_code", "duration_seconds",
               "facts_written", "manifests_checked", "wall_seconds", "cpu_seconds",
               "exception_class"}
    for row in _records(archived_root):
        assert set(row) <= allowed, row
        blob = json.dumps(row)
        assert str(archived_root) not in blob
        assert "attempt" not in blob and "sha256" not in blob


def test_unreadable_history_stays_unknown_and_accuses_nobody(data_root, monkeypatch):
    (data_root / "logs").mkdir(parents=True, exist_ok=True)
    (data_root / usage_store.STORE_REL).write_bytes(b"not a database\n" * 512)  # a damaged store
    _seal_manifest(data_root, "absent-attempt-id")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data_root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(data_root / "settings.json"))

    audit = HistoricalAudit()
    audit.start(data_root, REPO)
    terminal = _await_terminal(data_root)
    assert terminal["phase"] == "unknown", terminal
    assert terminal["facts_written"] == 0, "an unknown ledger must not produce accusations"


# --------------------------------------------------------------------------
# Stop races: both sides of handle publication
# --------------------------------------------------------------------------

class _FakeChild:
    def __init__(self, payload: bytes = b""):
        import io

        self.pid = 987654
        self.stdout = io.BytesIO(payload)
        self.killed = threading.Event()
        self.waited = threading.Event()

    def kill(self):
        self.killed.set()

    def wait(self):
        self.waited.set()
        return -9 if self.killed.is_set() else 0


def _patch_spawn(monkeypatch, child, *, gate: threading.Event | None = None, seen=None):
    from ouroboros import process_custody

    def fake_spawn(cmd, **kwargs):
        if seen is not None:
            seen.append((cmd, kwargs))
        if gate is not None:
            assert gate.wait(GATE)
        return child

    monkeypatch.setattr(process_custody, "spawn_supervised", fake_spawn)


def test_the_audit_child_inherits_the_containment_token(tmp_path, monkeypatch):
    """The reap that proves a data root quiet is env-token membership: a child
    spawned without the token is a live writer the container cannot see."""
    from ouroboros.process_containment import CONTAINMENT_ENV_PREFIX

    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(CONTAINMENT_ENV_PREFIX + "DEADBEEF", "1")  # upper: Windows folds env keys
    monkeypatch.setenv("OUROBOROS_UNRELATED_SECRET", "no")
    seen: list = []
    _patch_spawn(monkeypatch, _FakeChild(), seen=seen)

    audit = HistoricalAudit()
    audit.start(tmp_path, REPO)
    _await_terminal(tmp_path, timeout=10)
    assert len(seen) == 1
    env = seen[0][1]["env"]
    assert env[CONTAINMENT_ENV_PREFIX + "DEADBEEF"] == "1"
    assert "OUROBOROS_UNRELATED_SECRET" not in env  # the allowlist still holds


def test_stop_between_spawn_and_publication_still_kills_the_child(tmp_path, monkeypatch):
    """Stop wins even when it lands while the spawner is inside Popen."""
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    child = _FakeChild()
    gate = threading.Event()
    _patch_spawn(monkeypatch, child, gate=gate)

    audit = HistoricalAudit()
    audit.start(tmp_path, REPO)
    time.sleep(0.05)              # the spawner thread is blocked inside the fake spawn
    assert audit._process is None  # nothing is published yet
    audit.stop()                   # stop cannot see a handle; it must latch
    gate.set()                     # the spawn now returns and publishes
    assert child.killed.wait(GATE), "the spawner did not re-check stop after publishing"
    terminal = _await_terminal(tmp_path, timeout=10)
    assert terminal["phase"] == "stopped", terminal


def test_stop_after_publication_signals_the_published_child(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    child = _FakeChild()
    published = threading.Event()
    hold = threading.Event()

    from ouroboros import process_custody

    def fake_spawn(cmd, **kwargs):
        return child

    monkeypatch.setattr(process_custody, "spawn_supervised", fake_spawn)

    original_read = child.stdout.read

    def blocking_read(size=-1):
        published.set()
        assert hold.wait(GATE)
        return original_read(size)

    child.stdout.read = blocking_read  # type: ignore[assignment]
    audit = HistoricalAudit()
    audit.start(tmp_path, REPO)
    assert published.wait(GATE)
    assert audit._process is child
    audit.stop()
    assert child.killed.is_set()
    hold.set()
    terminal = _await_terminal(tmp_path, timeout=10)
    assert terminal["phase"] == "stopped", terminal


def test_stop_before_start_never_spawns(tmp_path, monkeypatch):
    seen: list = []
    _patch_spawn(monkeypatch, _FakeChild(), seen=seen)
    audit = HistoricalAudit()
    audit.stop()
    audit.start(tmp_path, REPO)
    time.sleep(0.05)
    assert seen == []
    assert audit._launched is False


def test_one_launch_per_generation_even_under_concurrent_starts(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    seen: list = []
    _patch_spawn(monkeypatch, _FakeChild(), seen=seen)
    audit = HistoricalAudit()
    ready = threading.Barrier(4)

    def start():
        ready.wait(GATE)
        audit.start(tmp_path, REPO)

    threads = [threading.Thread(target=start) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=GATE)
    _await_terminal(tmp_path, timeout=10)
    assert len(seen) == 1, seen


def test_the_child_gets_a_dedicated_process_group(tmp_path, monkeypatch):
    """Own group, no Job breakaway: a custody-write failure inside
    spawn_supervised kills the child's whole POSIX group, which must never be
    the server's own group (independent Fable finding on the first WIP)."""
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    seen: list = []
    _patch_spawn(monkeypatch, _FakeChild(), seen=seen)
    audit = HistoricalAudit()
    audit.start(tmp_path, REPO)
    _await_terminal(tmp_path, timeout=10)
    assert len(seen) == 1
    cmd, kwargs = seen[0]
    assert kwargs.get("new_process_group", True) is True
    assert kwargs["purpose"] == "startup_historical_audit"
    assert kwargs["scope"] == "session"
    assert cmd[1:3] == ["-m", "ouroboros.startup_historical_audit"]
    # Only ordinary runtime environment, explicit roots and the inherited
    # containment token reach the child.
    from ouroboros.process_containment import CONTAINMENT_ENV_PREFIX

    assert {key for key in kwargs["env"] if not key.startswith(CONTAINMENT_ENV_PREFIX)} <= {
        "PATH", "HOME", "USERPROFILE", "SystemRoot", "WINDIR", "TEMP", "TMP", "TMPDIR",
        "LANG", "LC_ALL", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH",
        "OUROBOROS_DATA_DIR", "OUROBOROS_REPO_DIR", "OUROBOROS_SETTINGS_PATH",
    }, kwargs["env"]


def test_an_oversized_child_report_is_refused_and_the_child_killed(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    child = _FakeChild(b"x" * 9000)
    _patch_spawn(monkeypatch, child)
    audit = HistoricalAudit()
    audit.start(tmp_path, REPO)
    terminal = _await_terminal(tmp_path, timeout=10)
    assert terminal["phase"] == "unknown", terminal
    assert child.killed.is_set()


def test_a_spawn_failure_records_only_the_exception_class(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    from ouroboros import process_custody

    def explode(cmd, **kwargs):
        raise PermissionError(f"/secret/path/{cmd}")

    monkeypatch.setattr(process_custody, "spawn_supervised", explode)
    audit = HistoricalAudit()
    audit.start(tmp_path, REPO)
    terminal = _await_terminal(tmp_path, timeout=10)
    assert terminal["phase"] == "failed"
    assert terminal["exception_class"] == "PermissionError"
    assert "/secret/path" not in json.dumps(terminal)


# --------------------------------------------------------------------------
# Report schema
# --------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    b"", b"not json", b"[]", b'{"status":"weird","facts_written":1,"manifests_checked":1,'
    b'"wall_seconds":1,"cpu_seconds":1}',
    b'{"status":"completed","facts_written":-1,"manifests_checked":1,"wall_seconds":1,"cpu_seconds":1}',
    b'{"status":"completed","facts_written":true,"manifests_checked":1,"wall_seconds":1,"cpu_seconds":1}',
    b'{"status":"completed","facts_written":1,"manifests_checked":1,"wall_seconds":Infinity,"cpu_seconds":1}',
])
def test_report_fields_refuses_anything_outside_the_closed_schema(payload):
    assert _report_fields(payload) == {}


def test_report_fields_drops_unknown_keys_and_clamps_exception_class():
    fields = _report_fields(
        b'{"status":"failed","facts_written":0,"manifests_checked":0,"wall_seconds":0,'
        b'"cpu_seconds":0,"exception_class":"ValueError","path":"/secret","note":"x"}'
    )
    assert fields == {"status": "failed", "facts_written": 0, "manifests_checked": 0,
                      "wall_seconds": 0, "cpu_seconds": 0, "exception_class": "ValueError"}
    hostile = _report_fields(
        b'{"status":"failed","facts_written":0,"manifests_checked":0,"wall_seconds":0,'
        b'"cpu_seconds":0,"exception_class":"/etc/passwd not found"}'
    )
    assert "exception_class" not in hostile
