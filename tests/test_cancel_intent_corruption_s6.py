"""S6 C1/C2 — what a CORRUPT cancel-intent projection does to the claim fence.

The mint (``request_cancel``) reads the projection strictly: a malformed file
refuses the mutation with a typed ``CancelIntentProjectionCorrupt`` and keeps
the bytes (pinned by ``tests/test_gate_round3_fixes.py``). This module pins the
six NON-minting mutators — ``mark_finalize_control_drained``,
``mark_intent_scope``, ``claim_intent``, ``release_claim``, ``settle_intent``,
``withdraw_intent`` — and the watchdog's enforcement read over the same corrupt file.

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
# C1 — the six non-minting mutators over a corrupt projection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", [CORRUPT_CONTAINER, CORRUPT_INTENTS])
def test_c1_non_minting_mutators_fail_closed_on_a_corrupt_projection(tmp_path, payload):
    """C1/O1: every mutator that AUTHORS a record refuses a corrupt projection.

    Pre-fix the first five returned the absent-intent answer (``None``/``False``)
    without raising, which is indistinguishable from "no cancel was ever
    requested" — the shape that drops the claim-first fence. All six now raise the
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
    with pytest.raises(ci.CancelIntentProjectionCorrupt):
        ci.withdraw_intent(
            tmp_path, "victim", withdrawable=lambda row: True,
            expected_generation=intent.get("generation"),
            request_id=str(intent.get("request_id") or ""),
        )

    assert _store(tmp_path).read_text(encoding="utf-8") == payload, "bytes are kept"
    refusals = [
        row for row in _trail(tmp_path)
        if row.get("event") == "projection_corrupt_refused"
    ]
    assert {row.get("op") for row in refusals} == {
        "claim_intent", "settle_intent", "release_claim", "mark_intent_scope",
        "mark_finalize_control_drained", "withdraw_intent",
    }, "each refusal is disclosed with the operation that was refused"


def test_c1_a_healthy_projection_without_the_row_is_not_corruption(tmp_path):
    """A valid projection that simply holds no row for this task keeps the
    absent-intent answer — the fix must not turn "no intent" into an error."""
    ci.request_cancel(tmp_path, "other", reason="unrelated")

    assert ci.claim_intent(tmp_path, "victim", owner="custody") is None
    assert ci.settle_intent(tmp_path, "victim", outcome="cancelled") is None
    assert ci.release_claim(tmp_path, "victim", error="") is False
    assert ci.mark_intent_scope(tmp_path, "victim", ci.SCOPE_CASCADE) is False
    assert ci.mark_finalize_control_drained(tmp_path, "victim") is False
    assert ci.withdraw_intent(tmp_path, "victim", withdrawable=lambda row: True) is None
    assert list(ci.active_intents(tmp_path)) == ["other"], "the live row is untouched"


# ---------------------------------------------------------------------------
# Disclosure — the projection's own version field
# ---------------------------------------------------------------------------
# C2 — the watchdog's enforcement read stays fail-soft-but-loud (unchanged)
# ---------------------------------------------------------------------------
