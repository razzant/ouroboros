"""Shared fixtures and helpers for the CPL4-C6 compaction pins.

Imported by ``tests/test_usage_compaction.py`` (the compaction pass) and
``tests/test_usage_compaction_archive.py`` (the archive reader / CPL-5 join
surface). Module, not conftest, following the ``tests/fixtures_*.py``
convention already used by ``tests/fixtures_e2e_cancellation.py``.
"""

from __future__ import annotations

import json

import pytest

from ouroboros import usage_accounting as ua
from ouroboros import usage_compaction as uc


@pytest.fixture
def data_root_any_tier(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    (root / "state").mkdir(parents=True)
    (root / ua.IMPORT_REL).parent.mkdir(parents=True, exist_ok=True)
    (root / ua.IMPORT_REL).write_text(
        json.dumps({"completed": True}), encoding="utf-8"
    )
    return root


@pytest.fixture
def data_root(data_root_any_tier):
    return data_root_any_tier  # the pass runs on every OS since the 7.0 Windows kernel tier


def _request(data_root, **overrides):
    values = {"model": "openai/gpt-5.2", "provider": "openai", "reservation_usd": 1.0,
              "drive_root": data_root, "task_id": "child", "root_task_id": "root", "source": "test"}
    values.update(overrides)
    return ua.AttemptRequest(**values)


def _ledger_lines(data_root):
    path = data_root / ua.LEDGER_REL
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ledger_rows(data_root):
    return [json.loads(line) for line in _ledger_lines(data_root)]


def _settle(data_root, *, cost=None, usage=None, cost_final=False, **request_overrides):
    reservation = ua.reserve_attempt(_request(data_root, **request_overrides))
    ua.mark_dispatched(reservation)
    ua.settle_attempt(reservation, usage or {"prompt_tokens": 10, "completion_tokens": 5},
                      cost_usd=cost, cost_final=cost_final)
    return reservation


def _seed_mixed_ledger(data_root):
    """A realistic ledger: settled (weird floats, unknown costs), unresolved,
    released, in-flight, sessions, external, review-attributed."""
    _settle(data_root, cost=0.123456789012345, cost_final=True)
    _settle(data_root, cost=1.1, task_id="t2", root_task_id="root2", root_limit_usd=50.0)
    _settle(data_root, cost=2.2, task_id="t2", root_task_id="root2", root_limit_usd=40.0)
    _settle(data_root, cost=None, usage={}, model="openai/gpt-5.2-mini")
    reservation = ua.reserve_attempt(_request(data_root, task_id="t3"))
    ua.mark_dispatched(reservation)
    ua.mark_unresolved(reservation, "provider went dark")
    reservation = ua.reserve_attempt(_request(data_root, task_id="t4"))
    ua.release_attempt(reservation, "not_dispatched")
    ua.record_subscription_session("sess-1", drive_root=data_root, route="claudexor:claude", model="fable",
                                   task_id="t5", root_task_id="root", spend_usd=0.5, reset_at="2026-09-02T00:00:00Z")
    ua.record_unmetered_external_dispatch("ext-1", drive_root=data_root, model="ext-model", task_id="t6",
                                          prompt_tokens=7, completion_tokens=3)
    with ua.usage_scope(ua.UsageScope(
        drive_root=data_root, task_id="rv", root_task_id="root",
        review_skill="skill-x", review_wave_id="w1", review_slot_id="s1",
    )):
        _settle(data_root, cost=3.5, cost_final=True)
    # In-flight chains that MUST survive: one reserved, one dispatched.
    reserved = ua.reserve_attempt(_request(data_root, task_id="open-r"))
    dispatched = ua.reserve_attempt(_request(data_root, task_id="open-d"))
    ua.mark_dispatched(dispatched)
    return reserved, dispatched


def _compact(data_root):
    with ua._locked(data_root) as heartbeat:
        return uc.compact_usage_ledger_locked(data_root, heartbeat=heartbeat)


@pytest.fixture
def compacted(data_root):
    """A seeded, compacted ledger: its live header and the segment that header names."""
    _seed_mixed_ledger(data_root)
    assert _compact(data_root) is not None
    header = _ledger_rows(data_root)[0]
    return header, data_root / header["archive_rel"]


def _append_raw_row(data_root, row):
    """Append one already-legal row straight to the live ledger bytes."""
    path = data_root / ua.LEDGER_REL
    row = {**row, "seq": len(_ledger_lines(data_root)) + 1}
    with path.open("ab") as handle:
        handle.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
    return row


def _raced_row(attempt_id):
    """A settled charge a concurrent holder lands: the row a stale swap erases."""
    return {"kind": "subscription_session", "attempt_id": attempt_id, "state": "settled",
            "ts": "2026-09-01T00:00:00+00:00", "cost_usd": 0.25, "cost_final": True, "model": "fable",
            "provider": "claudexor", "category": "task", "source": "subscription", "task_id": "t",
            "root_task_id": "root", "parent_task_id": ""}
