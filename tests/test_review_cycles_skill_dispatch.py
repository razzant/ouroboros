"""Skill-review dispatch accounting and append-only marker tests.

Covers one-paid-unit lifecycle merging, panel-contract-scoped rebuttal replay,
per-wave marker concurrency and migration, and durable API dispatch binding.
"""

from __future__ import annotations

import json
import pathlib
import types

import pytest


KEY = "OUROBOROS_REVIEW_MAX_CYCLES"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.delenv("OUROBOROS_REVIEW_ENFORCEMENT", raising=False)
    yield


# ---------------------------------------------------------------------------
# F3 — the skill dispatch marker: four wave outcomes, one paid unit each


def _fake_manifest():
    return types.SimpleNamespace(
        name="demo", description="d", version="1", type="tool", runtime="python",
        timeout_sec=30, permissions=[], conflicts=[], env_from_settings=[],
        requires=[], scripts=[], scheduled_tasks=[], entry="main.py",
        is_extension=lambda: False,
    )


def _wire_skill_wave(monkeypatch, tmp_path, *, content_hash, passes):
    import ouroboros.skill_review_passes as passes_mod
    from ouroboros import config as cfg
    from ouroboros import skill_review

    drive = pathlib.Path(tmp_path)
    skill = types.SimpleNamespace(
        name="demo", manifest=_fake_manifest(), skill_dir=drive / "skill",
        load_error="", review=None, source="",
    )
    binding = types.SimpleNamespace(state_drive_root=drive)
    monkeypatch.setattr(skill_review, "build_resolved_resource_binding", lambda *a, **k: binding)
    monkeypatch.setattr(skill_review, "load_bound_skill", lambda b: skill)
    monkeypatch.setattr(skill_review, "compute_content_hash", lambda *a, **k: content_hash)
    monkeypatch.setattr(skill_review, "_run_deterministic_preflight", lambda *a, **k: None)
    monkeypatch.setattr(skill_review, "reviewer_slot_config_error", lambda: "")
    monkeypatch.setattr(skill_review, "_build_skill_file_packs", lambda *a, **k: ["pack"])
    monkeypatch.setattr(skill_review, "_official_hub_review_profile", lambda s: "")
    monkeypatch.setattr(skill_review, "_review_wave_budget_block", lambda *a, **k: None)
    monkeypatch.setattr(skill_review, "emit_review_model_error_events", lambda *a, **k: None)
    monkeypatch.setattr(skill_review, "save_review_state", lambda *a, **k: None)
    monkeypatch.setattr(
        skill_review, "auto_grant_if_enabled",
        lambda *a, **k: types.SimpleNamespace(
            requested_keys=[], granted_keys=[],
            requested_permissions=[], granted_permissions=[]),
    )
    monkeypatch.setattr(cfg, "get_review_models", lambda: ["m1", "m2"])
    monkeypatch.setattr(passes_mod, "run_skill_review_passes", passes)
    return skill


def _panel_result(models, *, verdict="PASS"):
    """A minimal parseable two-model panel body for the REAL parser: each actor
    returns a JSON array answering every required checklist item."""
    from ouroboros.skill_review import _SKILL_REVIEW_ITEMS

    items = [
        {"item": item, "verdict": verdict, "severity": "advisory", "reason": "checked"}
        for item in _SKILL_REVIEW_ITEMS
    ]
    return json.dumps({
        "model_count": len(models),
        "results": [
            {"model": model, "text": json.dumps(items), "slot_id": f"slot_{idx + 1}"}
            for idx, model in enumerate(models)
        ],
    })


def _dispatching(passes_body):
    from ouroboros.review_dispatch import stamp_review_paid_on_dispatch

    def _wave(ctx, *a, **k):
        stamp_review_paid_on_dispatch(ctx)
        return passes_body(ctx)

    return _wave


def _paid_units(tmp_path, content_hash):
    from ouroboros.skill_review_cycles import count_paid_skill_review_cycles

    return count_paid_skill_review_cycles(
        pathlib.Path(tmp_path), "demo", "manual:demo", content_hash=content_hash,
    )


def _history_rows(tmp_path):
    from ouroboros.skill_review_history import review_history_path

    path = review_history_path(pathlib.Path(tmp_path), "demo")
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_skill_wave_outcomes_yield_exactly_one_paid_unit_each(tmp_path, monkeypatch):
    """F3 on a REAL jsonl (persist=True): substantive verdict, quorum failure,
    transport failure, and exception-after-dispatch each leave exactly ONE
    paid unit in the derived count — via the terminal row when one lands, via
    the unmerged write-ahead dispatch marker when none does."""
    from ouroboros import skill_review
    from ouroboros.skill_review_history import load_dispatch_markers

    ctx = types.SimpleNamespace(task_id="", task_metadata={}, event_queue=None)

    # (1) substantive verdict: terminal row carries the paid facts, marker merged.
    _wire_skill_wave(
        monkeypatch, tmp_path, content_hash="h-verdict",
        passes=_dispatching(lambda c: ("prompt", {}, _panel_result(["m1", "m2"]), "")),
    )
    outcome = skill_review.review_skill(ctx, "demo", persist=True)
    assert outcome.status in ("clean", "warnings", "blockers")
    assert outcome.paid is True and outcome.wave_id
    assert _paid_units(tmp_path, "h-verdict") == 1
    assert load_dispatch_markers(pathlib.Path(tmp_path), "demo") == []  # merged+cleared
    row = _history_rows(tmp_path)[-1]
    assert row["paid"] is True and row["wave_id"] == outcome.wave_id
    assert row["review_contract_fingerprint"] == outcome.review_contract_fingerprint

    # (2) quorum failure: the internal history append carries the paid facts.
    _wire_skill_wave(
        monkeypatch, tmp_path, content_hash="h-quorum",
        passes=_dispatching(lambda c: ("prompt", {}, _panel_result(["m1"]), "")),
    )
    outcome = skill_review.review_skill(ctx, "demo", persist=True)
    assert outcome.status == "pending" and outcome.paid is True
    assert _paid_units(tmp_path, "h-quorum") == 1
    assert load_dispatch_markers(pathlib.Path(tmp_path), "demo") == []
    row = _history_rows(tmp_path)[-1]
    assert row["paid"] is True and row["content_hash"] == "h-quorum"

    # (3) transport failure: no history row lands, the unmerged marker counts.
    _wire_skill_wave(
        monkeypatch, tmp_path, content_hash="h-transport",
        passes=_dispatching(lambda c: ("prompt", {}, "", "provider exploded")),
    )
    outcome = skill_review.review_skill(ctx, "demo", persist=True)
    assert outcome.status == "pending" and outcome.paid is True
    assert _paid_units(tmp_path, "h-transport") == 1
    markers = load_dispatch_markers(pathlib.Path(tmp_path), "demo")
    assert len(markers) == 1
    assert markers[0].get("content_hash") == "h-transport" and markers[0].get("paid") is True

    # (4) exception after dispatch: the write-ahead marker survives the crash.
    def _boom(ctx_arg):
        raise RuntimeError("wave crashed after dispatch")

    _wire_skill_wave(
        monkeypatch, tmp_path, content_hash="h-crash", passes=_dispatching(_boom),
    )
    with pytest.raises(RuntimeError):
        skill_review.review_skill(ctx, "demo", persist=True)
    assert _paid_units(tmp_path, "h-crash") == 1
    # Per-wave markers are APPEND-ONLY: wave (4)'s write did NOT displace the
    # orphaned marker (3) — both coexist and each spend is still exactly one
    # unit derived from its own unmerged marker.
    assert _paid_units(tmp_path, "h-transport") == 1
    unmerged = load_dispatch_markers(pathlib.Path(tmp_path), "demo")
    assert {m.get("content_hash") for m in unmerged} == {"h-transport", "h-crash"}
    # The seam is restored after every wave.
    assert getattr(ctx, "_review_paid_stamp", None) is None


def test_lifecycle_timeout_terminal_merges_the_dispatch_marker(tmp_path, monkeypatch):
    """F3(b): a lifecycle timeout finalizes with NO result object — the
    terminal history row still carries the paid facts, merged from the
    write-ahead dispatch marker by job id."""
    from ouroboros.skill_review_history import load_dispatch_markers, write_dispatch_marker
    from ouroboros.skill_review_runner import _mark_review_job_timeout, review_job_state_path
    from ouroboros.utils import atomic_write_json

    drive = pathlib.Path(tmp_path)
    (drive / "logs").mkdir(parents=True, exist_ok=True)
    job_path = review_job_state_path(drive, "demo")
    job_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(job_path, {
        "job_id": "job-77", "status": "running", "skill": "demo",
        "content_hash": "h-t", "group_id": "task:root-5:demo", "root_task_id": "root-5",
    }, trailing_newline=True)
    write_dispatch_marker(
        drive, "demo", wave_id="job-77", group_id="task:root-5:demo",
        content_hash="h-t", root_task_id="root-5",
        review_contract_fingerprint="cf-9", rebuttal_sha256="reb-9",
    )
    _mark_review_job_timeout(drive, "demo", "h-t", reason="lifecycle_timeout")
    rows = _history_rows(tmp_path)
    assert rows and rows[-1]["status"] == "timeout"
    assert rows[-1]["paid"] is True  # merged from the marker, result was None
    assert rows[-1]["usage_attribution_schema"] == "physical_attempt_v1"
    assert rows[-1]["review_contract_fingerprint"] == "cf-9"
    assert rows[-1]["rebuttal_sha256"] == "reb-9"
    assert load_dispatch_markers(drive, "demo") == []  # merge cleared it
    from ouroboros.skill_review_cycles import count_paid_skill_review_cycles

    assert count_paid_skill_review_cycles(drive, "demo", "task:root-5:demo") == 1


def test_terminal_history_append_failure_is_a_loud_typed_event(tmp_path, monkeypatch):
    """F3(c): a swallowed terminal-history append surfaces as a typed event."""
    import ouroboros.skill_review_runner as runner
    from ouroboros import skill_review_history

    drive = pathlib.Path(tmp_path)
    (drive / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(skill_review_history, "append_history_once", lambda *a, **k: False)
    ok = runner._append_terminal_history(
        drive, "demo", {"job_id": "j-1"}, status="failed",
        terminal_reason="boom", result=None, ts="t1",
    )
    assert ok is False
    events = [json.loads(line) for line in
              (drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    failed = [e for e in events if e["type"] == "skill_review_history_append_failed"]
    assert failed and failed[0]["skill"] == "demo" and failed[0]["job_id"] == "j-1"


# ---------------------------------------------------------------------------
# F4a — spent-rebuttal memory is scoped to the CURRENT panel contract


def test_spent_rebuttal_memory_lapses_with_the_panel_contract(tmp_path):
    from ouroboros.skill_review_history import review_history_path
    from ouroboros.skill_review_cycles import find_free_replay_row

    path = review_history_path(pathlib.Path(tmp_path), "demo")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        # reb-1 was answered by a substantive verdict — under the RETIRED contract.
        {"ts": "t1", "status": "warnings", "content_hash": "h1", "paid": True,
         "group_id": "manual:demo", "review_contract_fingerprint": "cf-old",
         "rebuttal_sha256": "reb-1", "job_id": "j1"},
        # The current contract has its own substantive verdict for the snapshot.
        {"ts": "t2", "status": "warnings", "content_hash": "h1", "paid": True,
         "group_id": "manual:demo", "review_contract_fingerprint": "cf-new",
         "job_id": "j2"},
    ]
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    # Under the current contract reb-1 has never been adjudicated: it buys the
    # paid rerun (no replay) instead of being refused as already spent.
    assert find_free_replay_row(
        pathlib.Path(tmp_path), "demo", group_id="manual:demo", content_hash="h1",
        contract_fingerprint="cf-new", rebuttal_sha256="reb-1",
    ) is None
    # Once the CURRENT contract has answered it, the same hash replays free.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(
            {"ts": "t3", "status": "warnings", "content_hash": "h1", "paid": True,
             "group_id": "manual:demo", "review_contract_fingerprint": "cf-new",
             "rebuttal_sha256": "reb-1", "job_id": "j3"}) + "\n")
    assert find_free_replay_row(
        pathlib.Path(tmp_path), "demo", group_id="manual:demo", content_hash="h1",
        contract_fingerprint="cf-new", rebuttal_sha256="reb-1",
    ) is not None


# ---------------------------------------------------------------------------
# C4 — append-only per-wave dispatch markers (concurrency + legacy migration)


def test_concurrent_wave_markers_are_append_only_and_each_merge_clears_its_own(tmp_path):
    """Two interleaved waves on ONE skill both keep their write-ahead paid
    fact (per-wave marker files — no single-file overwrite), both count
    toward the ceiling, and each terminal-row merge clears exactly its own
    marker without minting spurious infra rows for the live sibling."""
    from ouroboros.skill_review_cycles import count_paid_skill_review_cycles
    from ouroboros.skill_review_history import (
        append_history_once,
        load_dispatch_markers,
        write_dispatch_marker,
    )
    from ouroboros.utils import utc_now_iso

    drive = pathlib.Path(tmp_path)
    (drive / "logs").mkdir(parents=True, exist_ok=True)
    write_dispatch_marker(
        drive, "demo", wave_id="wave-A", group_id="manual:demo", content_hash="h-1",
    )
    write_dispatch_marker(
        drive, "demo", wave_id="wave-B", group_id="manual:demo", content_hash="h-1",
    )

    # Append-only: writing B neither displaced A's marker nor flushed the LIVE
    # wave A into the history as a fake infra terminal.
    markers = load_dispatch_markers(drive, "demo")
    assert {m["wave_id"] for m in markers} == {"wave-A", "wave-B"}
    assert {m["usage_attribution_schema"] for m in markers} == {"physical_attempt_v1"}
    assert _history_rows(tmp_path) == []
    assert count_paid_skill_review_cycles(
        drive, "demo", "manual:demo", content_hash="h-1",
    ) == 2

    # Wave A's REAL terminal row lands (idempotent merge by wave id): the paid
    # fact merges from A's own marker and ONLY A's marker is cleared.
    assert append_history_once(drive, "demo", {
        "ts": utc_now_iso(), "status": "clean", "content_hash": "h-1",
        "group_id": "manual:demo", "job_id": "wave-A", "wave_id": "wave-A",
        "failure_signature": [], "fail_findings": [],
    })
    rows = _history_rows(tmp_path)
    assert [row["status"] for row in rows] == ["clean"]  # the verdict, not "interrupted"
    assert rows[0]["usage_attribution_schema"] == "physical_attempt_v1"
    assert rows[-1]["paid"] is True
    assert {m["wave_id"] for m in load_dispatch_markers(drive, "demo")} == {"wave-B"}
    assert count_paid_skill_review_cycles(
        drive, "demo", "manual:demo", content_hash="h-1",
    ) == 2  # one landed row + one still-unmerged marker

    # Wave B merges too: no markers left, the count is stable.
    assert append_history_once(drive, "demo", {
        "ts": utc_now_iso(), "status": "warnings", "content_hash": "h-1",
        "group_id": "manual:demo", "job_id": "wave-B", "wave_id": "wave-B",
        "failure_signature": [], "fail_findings": [],
    })
    assert load_dispatch_markers(drive, "demo") == []
    assert count_paid_skill_review_cycles(
        drive, "demo", "manual:demo", content_hash="h-1",
    ) == 2


def test_legacy_single_file_marker_is_read_and_flushed_on_the_next_write(tmp_path):
    """Migration: a pre-upgrade SINGLE-file marker is tolerated read-side
    (listed + counted) and the next wave's write flushes it into the history
    as a paid infra terminal and removes the file — the spend is never
    forgotten and never double-counted."""
    from ouroboros.skill_review_cycles import count_paid_skill_review_cycles
    from ouroboros.skill_review_history import (
        legacy_dispatch_marker_path,
        load_dispatch_markers,
        write_dispatch_marker,
    )

    drive = pathlib.Path(tmp_path)
    (drive / "logs").mkdir(parents=True, exist_ok=True)
    legacy = legacy_dispatch_marker_path(drive, "demo")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({
        "ts": "2026-01-01T00:00:00Z", "wave_id": "wave-legacy",
        "group_id": "manual:demo", "content_hash": "h-1",
        "root_task_id": "", "paid": True,
        "review_contract_fingerprint": "cf-old", "rebuttal_sha256": "",
    }), encoding="utf-8")

    assert any(
        m["wave_id"] == "wave-legacy" for m in load_dispatch_markers(drive, "demo")
    )
    assert count_paid_skill_review_cycles(
        drive, "demo", "manual:demo", content_hash="h-1",
    ) == 1

    write_dispatch_marker(
        drive, "demo", wave_id="wave-new", group_id="manual:demo", content_hash="h-1",
    )
    assert not legacy.exists()
    flushed = [
        row for row in _history_rows(tmp_path)
        if row.get("terminal_reason") == "dispatched_wave_never_finalized"
    ]
    assert flushed and flushed[-1]["wave_id"] == "wave-legacy"
    assert flushed[-1]["paid"] is True
    assert flushed[-1]["review_contract_fingerprint"] == "cf-old"
    assert {m["wave_id"] for m in load_dispatch_markers(drive, "demo")} == {"wave-new"}
    assert count_paid_skill_review_cycles(
        drive, "demo", "manual:demo", content_hash="h-1",
    ) == 2  # the flushed legacy row + the new unmerged marker


def test_late_marker_overlays_unpaid_terminal_without_rewriting_or_double_counting(tmp_path):
    from ouroboros.skill_review_cycles import count_paid_skill_review_cycles
    from ouroboros.skill_review_history import (
        append_history_once,
        load_dispatch_markers,
        load_history,
        review_history_path,
        write_dispatch_marker,
    )

    drive = pathlib.Path(tmp_path)
    (drive / "logs").mkdir(parents=True, exist_ok=True)
    terminal = {
        "ts": "t1", "status": "timeout", "terminal_reason": "slot timeout",
        "content_hash": "h-late", "group_id": "task:root-late:demo",
        "root_task_id": "root-late", "job_id": "wave-late",
        "failure_signature": [], "fail_findings": [],
    }
    assert append_history_once(drive, "demo", terminal)
    path = review_history_path(drive, "demo")
    raw_before = path.read_bytes()

    write_dispatch_marker(
        drive, "demo", wave_id="wave-late", group_id="task:root-late:demo",
        content_hash="h-late", root_task_id="root-late",
        review_contract_fingerprint="cf-late", rebuttal_sha256="reb-late",
    )
    effective = load_history(drive, "demo", limit=0)[0]
    assert effective["paid"] is True
    assert effective["wave_id"] == "wave-late"
    assert effective["usage_attribution_schema"] == "physical_attempt_v1"
    assert effective["review_contract_fingerprint"] == "cf-late"
    assert path.read_bytes() == raw_before  # overlay is read-only
    assert count_paid_skill_review_cycles(drive, "demo", "task:root-late:demo") == 1

    # An idempotent terminal retry must not erase the only late-dispatch fact.
    assert append_history_once(drive, "demo", terminal)
    assert path.read_bytes() == raw_before
    assert [row["wave_id"] for row in load_dispatch_markers(drive, "demo")] == ["wave-late"]
    assert count_paid_skill_review_cycles(drive, "demo", "task:root-late:demo") == 1


def test_duplicate_terminal_clears_only_a_marker_already_present_in_raw_row(tmp_path):
    from ouroboros.skill_review_cycles import count_paid_skill_review_cycles
    from ouroboros.skill_review_history import (
        append_history_once,
        load_dispatch_markers,
        write_dispatch_marker,
    )

    drive = pathlib.Path(tmp_path)
    (drive / "logs").mkdir(parents=True, exist_ok=True)
    terminal = {
        "ts": "t1", "status": "clean", "content_hash": "h-stale",
        "group_id": "manual:demo", "root_task_id": "", "job_id": "wave-stale",
        "wave_id": "wave-stale", "paid": True,
        "usage_attribution_schema": "physical_attempt_v1",
        "review_contract_fingerprint": "cf-stale", "rebuttal_sha256": "reb-stale",
        "failure_signature": [], "fail_findings": [],
    }
    assert append_history_once(drive, "demo", terminal)
    write_dispatch_marker(
        drive, "demo", wave_id="wave-stale", group_id="manual:demo",
        content_hash="h-stale", review_contract_fingerprint="cf-stale",
        rebuttal_sha256="reb-stale",
    )
    assert len(load_dispatch_markers(drive, "demo")) == 1
    assert append_history_once(drive, "demo", terminal)
    assert load_dispatch_markers(drive, "demo") == []
    assert count_paid_skill_review_cycles(
        drive, "demo", "manual:demo", content_hash="h-stale",
    ) == 1


def test_direct_quorum_terminal_keeps_wave_identity_for_a_late_dispatch(tmp_path, monkeypatch):
    from ouroboros import skill_review
    from ouroboros.skill_review_history import load_history

    captured = {}

    def _terminal_before_worker_dispatch(ctx, *_args, **_kwargs):
        captured["stamp"] = ctx._review_paid_stamp
        return "prompt", {}, _panel_result(["m1"]), ""

    _wire_skill_wave(
        monkeypatch, tmp_path, content_hash="h-direct-late",
        passes=_terminal_before_worker_dispatch,
    )
    ctx = types.SimpleNamespace(task_id="", task_metadata={}, event_queue=None)
    outcome = skill_review.review_skill(ctx, "demo", persist=True)
    assert outcome.status == "pending" and outcome.paid is False and outcome.wave_id
    raw = _history_rows(tmp_path)[-1]
    assert raw.get("paid") is not True and raw["wave_id"] == outcome.wave_id

    captured["stamp"]()
    effective = load_history(pathlib.Path(tmp_path), "demo", limit=0)[-1]
    assert effective["paid"] is True and effective["wave_id"] == outcome.wave_id
    assert effective["usage_attribution_schema"] == "physical_attempt_v1"
    assert _paid_units(tmp_path, "h-direct-late") == 1


def test_bound_api_paid_stamp_waits_for_durable_sync_and_async_dispatch(tmp_path):
    import asyncio

    from ouroboros import usage_accounting as ua
    from ouroboros.review_dispatch import ReviewPaidStamp, bind_api_review_paid_stamp

    drive = pathlib.Path(tmp_path)
    request = ua.AttemptRequest(
        model="local-test", provider="local", reservation_usd=0.0,
        drive_root=drive, task_id="review", root_task_id="review",
    )
    writes = []

    def _write_paid():
        assert _ledger_state(drive) == "dispatched"
        writes.append("paid")

    def _ledger_state(root):
        rows = [json.loads(line) for line in (root / ua.LEDGER_REL).read_text().splitlines()]
        return rows[-1]["state"]

    stamp = ReviewPaidStamp(_write_paid)
    with bind_api_review_paid_stamp(stamp):
        with pytest.raises(RuntimeError, match="candidate refused"):
            ua.execute_physical_attempt(
                request, lambda: None,
                before_dispatch=lambda _reservation: (_ for _ in ()).throw(RuntimeError("candidate refused")),
            )
    assert writes == [] and stamp.fired is False

    with bind_api_review_paid_stamp(stamp):
        with pytest.raises(RuntimeError, match="wire failed"):
            ua.execute_physical_attempt(
                request, lambda: (_ for _ in ()).throw(RuntimeError("wire failed")),
            )
    assert writes == ["paid"] and stamp.fired is True

    async_stamp = ReviewPaidStamp(lambda: writes.append("async"))

    async def _send():
        assert async_stamp.fired is True
        return {"usage": {"prompt_tokens": 0, "completion_tokens": 0}}

    async def _run():
        with bind_api_review_paid_stamp(async_stamp):
            return await ua.execute_physical_attempt_async(request, _send)

    assert asyncio.run(_run())["usage"]["prompt_tokens"] == 0
    assert writes == ["paid", "async"]


def test_the_cross_skill_paid_count_is_byte_bounded(tmp_path, monkeypatch):
    """Audit #14-6b: ``load_history`` claims "every reader is byte-bounded",
    but the task-scoped count still scanned EVERY installed skill's
    review_history.jsonl WHOLE — the one read that multiplies the cost by the
    number of skills. It now uses the same tail window as the rest of the
    family, with the family's disclosed residual: a row aged past the window
    under-counts, it never over-blocks."""
    from ouroboros import skill_review_history
    from ouroboros.skill_review_cycles import count_paid_skill_review_cycles
    from ouroboros.skill_review_history import append_history_once

    drive = pathlib.Path(tmp_path)
    (drive / "logs").mkdir(parents=True, exist_ok=True)

    def _row(skill, wave, ancient=False):
        return {
            "ts": "t", "status": "clean", "paid": True,
            "content_hash": "h", "group_id": "task:root-b:" + skill,
            "root_task_id": "root-b", "job_id": wave, "wave_id": wave,
            "usage_attribution_schema": "physical_attempt_v1",
            "padding": "x" * 4000 if ancient else "",
        }

    for skill in ("alpha", "beta"):
        assert append_history_once(drive, skill, _row(skill, f"{skill}-ancient", ancient=True))
        assert append_history_once(drive, skill, _row(skill, f"{skill}-recent"))

    assert count_paid_skill_review_cycles(drive, "alpha", "task:root-b:alpha") == 4
    # Shrink the window below the ancient rows: they fall out of the read for
    # BOTH skills, proving the count no longer scans the files whole.
    monkeypatch.setattr(skill_review_history, "_DETAIL_LOOKUP_MAX_BYTES", 1024)
    assert count_paid_skill_review_cycles(drive, "alpha", "task:root-b:alpha") == 2
