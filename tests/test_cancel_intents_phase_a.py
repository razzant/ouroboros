"""Poltergeist phase A — the durable cancel-intent store itself (owner batch-2 4=A, batch-4 1=A).

This module owns the ``ouroboros.cancel_intents`` contract: minting a request idempotently
and forensically, the claim/settle/release lifecycle, the stored fields and their
migration, the effective read that projects a pending cancel over an intent or the legacy
latch, widen-only scope, and the generation fencing that keeps a live claim from being
stolen while an abandoned one is taken over.

Its consumers were split verbatim into ``tests/test_cancel_custody.py``,
``tests/test_cancel_task_done_validation.py``, ``tests/test_cancel_queue_integration.py``,
``tests/test_cancel_terminal_delivery.py``, ``tests/test_cancel_pending_outbox.py``,
``tests/test_cancel_cascade_and_disclosure.py`` and
``tests/test_cancel_live_kill_path.py``; the queue environment, the capture queue and the
live-process scaffolding they share live in ``tests/_cancel_intents_shared.py``.
"""

from __future__ import annotations

import json

from ouroboros import cancel_intents as ci
from ouroboros.task_results import (
    STATUS_CANCEL_REQUESTED,
    STATUS_COMPLETED,
    STATUS_RUNNING,
    load_task_result,
    write_task_result,
)


def test_request_cancel_is_idempotent_and_forensically_logged(tmp_path):
    first = ci.request_cancel(tmp_path, "t1", reason="stop it", source="agent_tool",
                              requested_by="parent1")
    assert first["state"] == ci.INTENT_REQUESTED
    assert first["already_requested"] is False
    assert first["request_id"].startswith("ci_")

    second = ci.request_cancel(tmp_path, "t1", reason="again")
    assert second["already_requested"] is True
    assert second["request_id"] == first["request_id"]
    # The projection stays compact: one active row.
    assert list(ci.active_intents(tmp_path)) == ["t1"]

    trail = (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in trail.splitlines() if line.strip()]
    requested = [r for r in rows if r.get("type") == "cancel_intent" and r.get("event") == "requested"]
    assert len(requested) == 1 and requested[0]["task_id"] == "t1"


def test_claim_settle_and_release_lifecycle(tmp_path):
    ci.request_cancel(tmp_path, "t2", source="http_single")
    claimed = ci.claim_intent(tmp_path, "t2", owner="cancel_task_custody")
    assert claimed["state"] == ci.INTENT_CLAIMED
    assert claimed["generation"] == 1

    # A failed custody attempt releases the claim; the watchdog can re-feed it.
    ci.release_claim(tmp_path, "t2", error="worker refused to die")
    row = ci.active_intent(tmp_path, "t2")
    assert row["state"] == ci.INTENT_REQUESTED
    assert row["last_error"] == "worker refused to die"

    reclaimed = ci.claim_intent(tmp_path, "t2", owner="cancel_task_custody")
    assert reclaimed["generation"] == 2

    settled = ci.settle_intent(tmp_path, "t2", outcome="cancelled", detail="teardown ok")
    assert settled["request_id"] == row["request_id"]
    # Settled rows LEAVE the projection (compactness is the design).
    assert ci.active_intent(tmp_path, "t2") is None
    assert ci.settle_intent(tmp_path, "t2", outcome="cancelled") is None  # idempotent

    # Claim staleness: a fresh claim is respected; unreadable provenance is stale.
    ci.request_cancel(tmp_path, "t3")
    fresh = ci.claim_intent(tmp_path, "t3", owner="x")
    assert ci.claim_is_stale(fresh) is False
    assert ci.claim_is_stale({**fresh, "claimed_at": "not-a-time"}) is True


def test_cancel_state_fields_and_migration(tmp_path):
    assert ci.cancel_state_fields(tmp_path, "none") == {}
    ci.request_cancel(tmp_path, "t4", reason="why not")
    fields = ci.cancel_state_fields(tmp_path, "t4")
    assert fields == {"cancel_state": "pending", "cancel_reason": "why not"}

    # Boot migration: a legacy latch file becomes a synthetic active intent;
    # the file itself is untouched (legacy read-path).
    write_task_result(tmp_path, "legacy1", STATUS_CANCEL_REQUESTED, result="wedged")
    migrated = ci.migrate_legacy_cancel_latches(tmp_path)
    assert migrated == ["legacy1"]
    intent = ci.active_intent(tmp_path, "legacy1")
    assert intent["source"] == "boot_migration"
    assert load_task_result(tmp_path, "legacy1")["status"] == STATUS_CANCEL_REQUESTED
    # Idempotent at the next boot.
    assert ci.migrate_legacy_cancel_latches(tmp_path) == []


def test_effective_read_projects_pending_for_intent_and_legacy_latch(tmp_path):
    from ouroboros.task_status import load_effective_task_result

    write_task_result(tmp_path, "run1", STATUS_RUNNING, result="working")
    ci.request_cancel(tmp_path, "run1")
    eff = load_effective_task_result(tmp_path, "run1")
    assert eff["status"] == STATUS_RUNNING and eff["cancel_state"] == "pending"

    write_task_result(tmp_path, "legacy2", STATUS_CANCEL_REQUESTED)
    eff = load_effective_task_result(tmp_path, "legacy2")
    assert eff["cancel_state"] == "pending"

    # A settled task never carries the pending projection.
    write_task_result(tmp_path, "done1", STATUS_COMPLETED, result="ok")
    ci.request_cancel(tmp_path, "done1")
    assert "cancel_state" not in load_effective_task_result(tmp_path, "done1")


def test_request_cancel_refuses_to_mint_an_intent_for_a_settled_task(tmp_path):
    """A-F8: a settled task would otherwise wear a false 'Cancelling…' badge."""
    write_task_result(tmp_path, "done2", STATUS_COMPLETED, result="finished on its own")
    outcome = ci.request_cancel(tmp_path, "done2", reason="too late", source="agent_tool")
    assert outcome["already_settled"] is True
    assert outcome["status"] == STATUS_COMPLETED
    assert ci.active_intent(tmp_path, "done2") is None
    assert ci.cancel_state_fields(tmp_path, "done2") == {}


def test_a_live_claim_is_never_stolen_and_an_abandoned_one_is(tmp_path):
    """A-F11 + A-F1c: exclusive while alive, taken over once abandoned."""
    ci.request_cancel(tmp_path, "excl1")
    first = ci.claim_intent(tmp_path, "excl1", owner="custody-1")
    assert first["generation"] == 1 and not first.get("claim_refused")

    refused = ci.claim_intent(tmp_path, "excl1", owner="custody-2")
    assert refused["claim_refused"] is True
    assert ci.active_intent(tmp_path, "excl1")["generation"] == 1

    # GR3-2: age ALONE is not abandonment while the claimant pid (this test
    # process) probes ALIVE — the stale claim is still refused.
    from datetime import datetime, timezone
    store = tmp_path / "state" / "cancel_intents.json"
    data = json.loads(store.read_text(encoding="utf-8"))
    data["intents"]["excl1"]["claimed_at"] = datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() - ci.CLAIM_STALE_SEC - 10, tz=timezone.utc,
    ).isoformat()
    store.write_text(json.dumps(data), encoding="utf-8")
    still_refused = ci.claim_intent(tmp_path, "excl1", owner="custody-2")
    assert still_refused["claim_refused"] is True
    assert ci.claim_is_abandoned(ci.active_intent(tmp_path, "excl1")) is False
    # A provably DEAD claiming process makes the takeover legitimate — even
    # with a fresh claim timestamp (no three-minute wait on a dead owner).
    data = json.loads(store.read_text(encoding="utf-8"))
    data["intents"]["excl1"]["claimed_at"] = ci.utc_now_iso()
    data["intents"]["excl1"]["claim_pid"] = 2 ** 22  # never a live pid
    store.write_text(json.dumps(data), encoding="utf-8")
    assert ci.claim_is_abandoned(ci.active_intent(tmp_path, "excl1")) is True
    taken = ci.claim_intent(tmp_path, "excl1", owner="custody-2")
    assert not taken.get("claim_refused") and taken["generation"] == 2
    # Stale with liveness UNKNOWN (pid missing) is also recoverable.
    data = json.loads(store.read_text(encoding="utf-8"))
    data["intents"]["excl1"]["claimed_at"] = datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() - ci.CLAIM_STALE_SEC - 10, tz=timezone.utc,
    ).isoformat()
    data["intents"]["excl1"].pop("claim_pid", None)
    store.write_text(json.dumps(data), encoding="utf-8")
    assert ci.claim_is_abandoned(ci.active_intent(tmp_path, "excl1")) is True


def test_stale_claimants_release_and_settle_are_fenced_by_generation(tmp_path):
    """A-F2: ``generation`` is a FENCE, not forensics."""
    ci.request_cancel(tmp_path, "fence1")
    stale = ci.claim_intent(tmp_path, "fence1", owner="custody-1")
    # Custody-1 is taken over (its claiming process is provably DEAD — GR3-2:
    # age alone never abandons a live claimant); custody-2 owns generation 2.
    store = tmp_path / "state" / "cancel_intents.json"
    data = json.loads(store.read_text(encoding="utf-8"))
    data["intents"]["fence1"]["claim_pid"] = 2 ** 22  # never a live pid
    store.write_text(json.dumps(data), encoding="utf-8")
    fresh = ci.claim_intent(tmp_path, "fence1", owner="custody-2")
    assert fresh["generation"] == stale["generation"] + 1

    # The stale claimant's release must NOT revert the newer claim.
    ci.release_claim(
        tmp_path, "fence1", error="stale", expected_generation=stale["generation"],
        request_id=stale["request_id"],
    )
    current = ci.active_intent(tmp_path, "fence1")
    assert current["state"] == ci.INTENT_CLAIMED
    assert current["claim_owner"] == "custody-2"

    # Nor may its settle delete the intent the new owner is still working.
    assert ci.settle_intent(
        tmp_path, "fence1", outcome="cancelled",
        expected_generation=stale["generation"], request_id=stale["request_id"],
    ) is None
    assert ci.active_intent(tmp_path, "fence1") is not None

    # The real owner settles it.
    assert ci.settle_intent(
        tmp_path, "fence1", outcome="cancelled",
        expected_generation=fresh["generation"], request_id=fresh["request_id"],
    ) is not None
    assert ci.active_intent(tmp_path, "fence1") is None
    trail = (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
    assert "claim_release_refused" in trail and "settle_refused" in trail


def test_mark_intent_scope_is_widen_only(tmp_path):
    """GR2-1d: single→cascade widens; cascade→single is refused as a no-op plus
    a forensic row (a narrowed record would replay the root alone)."""
    ci.request_cancel(tmp_path, "w1")
    assert ci.active_intent(tmp_path, "w1")["scope"] == ci.SCOPE_SINGLE
    assert ci.mark_intent_scope(tmp_path, "w1", ci.SCOPE_CASCADE) is True
    assert ci.active_intent(tmp_path, "w1")["scope"] == ci.SCOPE_CASCADE

    assert ci.mark_intent_scope(tmp_path, "w1", ci.SCOPE_SINGLE) is False
    assert ci.active_intent(tmp_path, "w1")["scope"] == ci.SCOPE_CASCADE
    trail = (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
    assert "scope_narrow_refused" in trail

    # The re-request path is widen-only too: an explicit single re-request over
    # a cascade intent must not narrow the recorded shape.
    again = ci.request_cancel(tmp_path, "w1", scope=ci.SCOPE_SINGLE)
    assert again["already_requested"] is True
    assert ci.active_intent(tmp_path, "w1")["scope"] == ci.SCOPE_CASCADE


def test_request_cancel_mints_a_cascade_coordination_intent_over_a_settled_target(tmp_path):
    """GR2-1b (store half): the cascade ingress may mint over a SETTLED root —
    that intent is the watchdog's replay trigger for the live descendants."""
    write_task_result(tmp_path, "sr0", "failed", result="died on budget")
    refused = ci.request_cancel(tmp_path, "sr0", scope=ci.SCOPE_CASCADE)
    assert refused["already_settled"] is True and ci.active_intent(tmp_path, "sr0") is None

    minted = ci.request_cancel(
        tmp_path, "sr0", scope=ci.SCOPE_CASCADE, allow_settled_target=True,
    )
    assert minted["already_requested"] is False
    row = ci.active_intent(tmp_path, "sr0")
    assert row is not None and row["scope"] == ci.SCOPE_CASCADE
    # The effective-status read never projects a false "Cancelling…" badge onto
    # the settled card (the projection only rides non-settled results).
    from ouroboros.task_status import load_effective_task_result

    assert "cancel_state" not in load_effective_task_result(tmp_path, "sr0")
