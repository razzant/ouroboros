"""S6 C1/C2 — what a CORRUPT cancel-intent projection does to the claim fence.

The mint (``request_cancel``) reads the projection strictly: a malformed file
refuses the mutation with a typed ``CancelIntentProjectionCorrupt`` and keeps
the bytes (pinned by ``tests/test_gate_round3_fixes.py``). This module pins the
five NON-minting mutators — ``mark_finalize_control_drained``,
``mark_intent_scope``, ``claim_intent``, ``release_claim``, ``settle_intent`` —
and the watchdog's enforcement read over the same corrupt file.

The distinction being characterized is the one the module already draws in
prose: reading-for-behaviour (fail-soft with disclosure) versus
authoring-a-record (fail-closed). A mutator that reads softly cannot tell
"nobody minted an intent" from "the projection is unreadable", and the second
answer silently removes the claim-first exclusion that ``cancel_task_custody``
relies on before it tears a task down.
"""

from __future__ import annotations

import json
import pathlib
import types

import pytest

from ouroboros import cancel_intents as ci


CORRUPT_CONTAINER = '"not an object"'
CORRUPT_INTENTS = '{"schema_version": 1, "intents": "not an object"}'


def _store(drive_root) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / "cancel_intents.json"


def _trail(drive_root):
    path = pathlib.Path(drive_root) / "logs" / "supervisor.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _corrupt_after_mint(tmp_path, payload: str) -> dict:
    """One live intent, then the projection is replaced by ``payload``."""
    intent = ci.request_cancel(tmp_path, "victim", reason="stop it", source="http_single")
    _store(tmp_path).write_text(payload, encoding="utf-8")
    return intent


# ---------------------------------------------------------------------------
# C1 — the five non-minting mutators over a corrupt projection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", [CORRUPT_CONTAINER, CORRUPT_INTENTS])
def test_c1_non_minting_mutators_fail_closed_on_a_corrupt_projection(tmp_path, payload):
    """C1/O1: every mutator that AUTHORS a record refuses a corrupt projection.

    Pre-fix these five returned the absent-intent answer (``None``/``False``)
    without raising, which is indistinguishable from "no cancel was ever
    requested" — the shape that drops the claim-first fence. They now raise the
    same typed error the mint raises, and the bytes are still never rewritten.
    """
    intent = _corrupt_after_mint(tmp_path, payload)

    with pytest.raises(ci.CancelIntentProjectionCorrupt):
        ci.claim_intent(tmp_path, "victim", owner="cancel_task_custody")
    with pytest.raises(ci.CancelIntentProjectionCorrupt):
        ci.settle_intent(
            tmp_path, "victim", outcome="cancelled",
            expected_generation=intent.get("generation"),
            request_id=str(intent.get("request_id") or ""),
        )
    with pytest.raises(ci.CancelIntentProjectionCorrupt):
        ci.release_claim(
            tmp_path, "victim", error="teardown failed",
            expected_generation=intent.get("generation"),
            request_id=str(intent.get("request_id") or ""),
        )
    with pytest.raises(ci.CancelIntentProjectionCorrupt):
        ci.mark_intent_scope(tmp_path, "victim", ci.SCOPE_CASCADE)
    with pytest.raises(ci.CancelIntentProjectionCorrupt):
        ci.mark_finalize_control_drained(tmp_path, "victim")

    assert _store(tmp_path).read_text(encoding="utf-8") == payload, "bytes are kept"
    refusals = [
        row for row in _trail(tmp_path)
        if row.get("event") == "projection_corrupt_refused"
    ]
    assert {row.get("op") for row in refusals} == {
        "claim_intent", "settle_intent", "release_claim", "mark_intent_scope",
        "mark_finalize_control_drained",
    }, "each refusal is disclosed with the operation that was refused"


def test_c1_custody_treats_a_corrupt_projection_as_a_refused_claim(tmp_path):
    """C1/O1 consequence at the caller: the claim-first fence is NOT dropped.

    ``task_lifecycle._claim_intent`` already documents the two distinct shapes:
    ``{}`` means "no active intent exists, custody may proceed on the legacy
    path", a RAISED claim means "cannot tell whether a live owner exists, treat
    as refused". Pre-fix a corrupt projection produced the FIRST shape; it now
    produces the second, which is what the module's own prose promises.
    """
    from supervisor.task_lifecycle import _claim_intent

    _corrupt_after_mint(tmp_path, CORRUPT_CONTAINER)
    q = types.SimpleNamespace(DRIVE_ROOT=tmp_path)

    claim = _claim_intent(q, "victim")

    assert claim.get("claim_refused") is True, (
        "a claim that cannot be proven exclusive is refused, never silently "
        "downgraded to the no-intent legacy path"
    )
    assert claim.get("claim_error") == "claim_read_failed"


def test_c1_absent_projection_is_still_an_ordinary_empty_read(tmp_path):
    """The strictness must separate ABSENT from MALFORMED: a never-written
    projection is the ordinary first-write case, not corruption."""
    assert ci.claim_intent(tmp_path, "nobody", owner="custody") is None
    assert ci.settle_intent(tmp_path, "nobody", outcome="cancelled") is None
    # upstream hardening gave release_claim a bool contract (fence proof);
    # absent projection is the ordinary False, not an error
    assert ci.release_claim(tmp_path, "nobody", error="") is False
    assert ci.mark_intent_scope(tmp_path, "nobody", ci.SCOPE_CASCADE) is False
    assert ci.mark_finalize_control_drained(tmp_path, "nobody") is False
    assert not _store(tmp_path).exists()


def test_c1_a_healthy_projection_without_the_row_is_not_corruption(tmp_path):
    """A valid projection that simply holds no row for this task keeps the
    absent-intent answer — the fix must not turn "no intent" into an error."""
    ci.request_cancel(tmp_path, "other", reason="unrelated")

    assert ci.claim_intent(tmp_path, "victim", owner="custody") is None
    assert ci.settle_intent(tmp_path, "victim", outcome="cancelled") is None
    assert ci.release_claim(tmp_path, "victim", error="") is False
    assert ci.mark_intent_scope(tmp_path, "victim", ci.SCOPE_CASCADE) is False
    assert ci.mark_finalize_control_drained(tmp_path, "victim") is False
    assert list(ci.active_intents(tmp_path)) == ["other"], "the live row is untouched"


# ---------------------------------------------------------------------------
# Disclosure — the projection's own version field
# ---------------------------------------------------------------------------


def test_schema_version_is_written_but_nothing_dispatches_on_it(tmp_path):
    """Disclosure (MIGRATION_v7.md): the envelope carries ``schema_version``
    and NO reader branches on it, so a future bump would be read as if it were
    version 1. The next format change must add the reader first; this test
    holds the evidence for that claim instead of leaving it in prose.
    """
    ci.request_cancel(tmp_path, "v1", reason="envelope shape")

    envelope = json.loads(_store(tmp_path).read_text(encoding="utf-8"))
    assert envelope["schema_version"] == ci._SCHEMA_VERSION == 1
    assert set(envelope) == {"schema_version", "intents"}

    source = pathlib.Path(ci.__file__).read_text(encoding="utf-8")
    reads = [
        line.strip() for line in source.splitlines()
        if "schema_version" in line and "_SCHEMA_VERSION" not in line.split("#")[0]
    ]
    assert reads == [], f"a reader appeared without a migration decision: {reads}"


# ---------------------------------------------------------------------------
# C2 — the watchdog's enforcement read stays fail-soft-but-loud (unchanged)
# ---------------------------------------------------------------------------


def test_c2_a_corrupt_projection_blinds_the_watchdog_loudly(tmp_path, monkeypatch, caplog):
    """C2: enforcement DEGRADES to "no intents" — deliberately, and loudly.

    ``active_intents(..., disclose_corruption=True)`` is a READ for behaviour,
    not an authored record: it keeps its fail-soft contract so one unreadable
    file cannot wedge the supervisor tick, and pays for it with a ``log.error``
    plus a typed forensic row. This test exists so the O1 write-side strictness
    cannot be mistaken for a licence to change the read side too: the observable
    below is the state of the art, not a defect being fixed.
    """
    from supervisor import queue as q
    from supervisor import task_lifecycle as tl

    _corrupt_after_mint(tmp_path, CORRUPT_CONTAINER)
    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    fed: list[str] = []
    monkeypatch.setattr(tl, "cancel_task_custody", lambda tid, **_kw: fed.append(tid))
    monkeypatch.setattr(tl, "cancel_task_by_id", lambda tid, **_kw: fed.append(tid))

    with caplog.at_level("ERROR"):
        outcomes = tl.sweep_cancel_intents()

    assert outcomes == {}, "the sweep sees no intents at all"
    assert fed == [], "no task is fed into custody even though an intent exists"
    assert any(
        "cancel-intent projection is unreadable/malformed" in record.getMessage()
        for record in caplog.records
    ), "the degrade is loud"
    assert any(
        row.get("event") == "projection_corrupt_refused"
        and row.get("op") == "active_intents"
        for row in _trail(tmp_path)
    ), "the enforcement read discloses the degrade in the durable trail"
