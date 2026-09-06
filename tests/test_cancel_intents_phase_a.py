"""Poltergeist phase A — durable cancel-intent lifecycle (owner batch-2 4=A, batch-4 1=A).

Closes the incident classes with tests:
- the wedged ``cancel_requested`` latch (intent survives a lost event; the
  supervisor watchdog feeds custody, the ONE settle owner);
- completed-result erasure by a late cancel (natural completion WINS, E2E
  through the real kill path with a live split-drive worker);
- the fabricated final-$0 cancel accounting;
- the undelivered salvaged answer (durable outbox seam, honest omitted counts);
- nonterminal ``task_done`` publication (durable lifecycle fault, no release).
"""

from __future__ import annotations
import json
import pytest
from ouroboros import cancel_intents as ci
from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
from ouroboros.task_results import (
    STATUS_CANCEL_REQUESTED,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_RUNNING,
    TASK_RESULT_SCHEMA_VERSION,
    load_task_result,
    write_task_result,
)

from tests._cancel_intents_shared import _write_root_retry_pair


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

def test_request_cancel_canonicalizes_a_valid_new_id_retry(tmp_path):
    _write_root_retry_pair(tmp_path, "old-root", "new-root")

    intent = ci.request_cancel(tmp_path, "old-root", reason="stop retry")

    assert intent["task_id"] == "new-root"
    assert intent["requested_task_id"] == "old-root"
    assert ci.active_intent(tmp_path, "old-root") is None
    assert ci.active_intent(tmp_path, "new-root")["request_id"] == intent["request_id"]

def test_cascade_rekeys_an_existing_single_retry_intent_to_the_logical_root(tmp_path):
    _write_root_retry_pair(tmp_path, "rekey-old", "rekey-new")
    single = ci.request_cancel(tmp_path, "rekey-old", reason="stop retry")

    cascade = ci.request_cancel(
        tmp_path,
        "rekey-old",
        reason="stop the whole tree",
        scope=ci.SCOPE_CASCADE,
    )

    assert cascade["request_id"] == single["request_id"]
    assert cascade["task_id"] == "rekey-old"
    assert cascade["scope"] == ci.SCOPE_CASCADE
    assert ci.active_intent(tmp_path, "rekey-new") is None
    assert ci.active_intent(tmp_path, "rekey-old")["request_id"] == single["request_id"]

def test_stop_now_on_retry_leaf_hardens_the_logical_root_cascade_intent(tmp_path):
    """A physical retry is an alias of an existing logical cascade owner.

    Stop-now must harden that one durable episode; minting a leaf-scoped single
    intent would create two cancellation owners and let the HTTP path narrow
    the already-authoritative cascade.
    """
    root_id, leaf_id = "cascade-stop-root", "cascade-stop-leaf"
    _write_root_retry_pair(tmp_path, root_id, leaf_id, new_status=STATUS_RUNNING)
    graceful = ci.request_cancel(
        tmp_path,
        root_id,
        reason="finish before stopping the tree",
        scope=ci.SCOPE_CASCADE,
        requested_stop_policy=ci.STOP_POLICY_FINALIZE,
    )

    hardened = ci.request_cancel(
        tmp_path,
        leaf_id,
        reason="stop now",
        requested_stop_policy=ci.STOP_POLICY_IMMEDIATE,
    )

    assert hardened["task_id"] == root_id
    assert hardened["request_id"] == graceful["request_id"]
    assert hardened["scope"] == ci.SCOPE_CASCADE
    assert ci.stop_policy(hardened) == ci.STOP_POLICY_IMMEDIATE
    assert ci.active_intent(tmp_path, leaf_id) is None
    assert set(ci.active_intents(tmp_path)) == {root_id}

def test_stable_retry_root_status_keeps_logical_cascade_cancel_state(tmp_path):
    """Following a stable root handle to its live retry keeps owner-stop state."""
    from ouroboros.task_status import load_effective_task_result

    root_id, leaf_id = "status-cascade-root", "status-cascade-leaf"
    _write_root_retry_pair(tmp_path, root_id, leaf_id, new_status=STATUS_RUNNING)
    ci.request_cancel(
        tmp_path,
        root_id,
        reason="wrap up the whole tree",
        scope=ci.SCOPE_CASCADE,
        requested_stop_policy=ci.STOP_POLICY_FINALIZE,
    )

    effective = load_effective_task_result(tmp_path, root_id)

    assert effective["task_id"] == leaf_id
    assert effective["status"] == STATUS_RUNNING
    assert effective["cancel_state"] == "pending"
    assert effective["cancel_reason"] == "wrap up the whole tree"
    assert effective["stop_policy"] == ci.STOP_POLICY_FINALIZE
    assert ci.active_intent(tmp_path, leaf_id) is None

def test_request_cancel_same_id_retry_stays_exact(tmp_path):
    write_task_result(
        tmp_path,
        "same-id-child",
        "interrupted",
        root_task_id="root",
        parent_task_id="root",
        delegation_role="subagent",
        retry_task_id="same-id-child",
    )

    intent = ci.request_cancel(tmp_path, "same-id-child")

    assert intent["task_id"] == "same-id-child"
    assert "requested_task_id" not in intent

def test_request_cancel_refuses_partial_retry_lineage(tmp_path):
    write_task_result(
        tmp_path,
        "broken-old",
        "interrupted",
        superseded_by="broken-new",
        retry_task_id="different-new",
    )

    with pytest.raises(ci.CancelIntentLineageIndeterminate):
        ci.request_cancel(tmp_path, "broken-old")

    assert ci.active_intents(tmp_path) == {}

def test_request_cancel_refuses_a_retry_leaf_bound_to_a_foreign_root(tmp_path):
    write_task_result(
        tmp_path,
        "lineage-old",
        "interrupted",
        root_task_id="lineage-old",
        delegation_role="root",
        superseded_by="lineage-new",
        retry_task_id="lineage-new",
    )
    write_task_result(
        tmp_path,
        "lineage-new",
        "scheduled",
        root_task_id="foreign-root",
        parent_task_id="",
        delegation_role="root",
        supersedes_task_id="lineage-old",
        original_task_id="lineage-old",
        timeout_retry_from="lineage-old",
    )

    with pytest.raises(ci.CancelIntentLineageIndeterminate):
        ci.request_cancel(tmp_path, "lineage-old")

    assert ci.active_intents(tmp_path) == {}

def test_request_cancel_reports_a_terminal_retry_leaf_as_already_settled(tmp_path):
    _write_root_retry_pair(
        tmp_path, "finished-old", "finished-new", new_status=STATUS_COMPLETED,
    )

    result = ci.request_cancel(tmp_path, "finished-old")

    assert result["already_settled"] is True
    assert result["task_id"] == "finished-new"
    assert result["requested_task_id"] == "finished-old"
    assert result["status"] == STATUS_COMPLETED
    assert ci.active_intents(tmp_path) == {}

def test_claim_settle_and_release_lifecycle(tmp_path):
    ci.request_cancel(tmp_path, "t2", source="http_single")
    claimed = ci.claim_intent(tmp_path, "t2", owner="cancel_task_custody")
    assert claimed["state"] == ci.INTENT_CLAIMED
    assert claimed["generation"] == 1

    # A failed custody attempt releases the claim; the watchdog can re-feed it.
    assert ci.release_claim(tmp_path, "t2", error="worker refused to die") is True
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

    assert ci.release_claim(tmp_path, "t2", error="already released") is False

    # Claim staleness: a fresh claim is respected; unreadable provenance is stale.
    ci.request_cancel(tmp_path, "t3")
    fresh = ci.claim_intent(tmp_path, "t3", owner="x")
    assert ci.claim_is_stale(fresh) is False
    assert ci.claim_is_stale({**fresh, "claimed_at": "not-a-time"}) is True

@pytest.mark.parametrize(
    "corrupt_bytes",
    [
        b'{"intents": [broken',
        b'{"schema_version": 1, "intents": []}',
        b'{"schema_version": 1, "intents": {"corrupt-claim": []}}',
        b'{"schema_version": 1, "intents": {"unrelated": []}}',
    ],
)
def test_claim_intent_refuses_an_existing_corrupt_projection(
    tmp_path, corrupt_bytes,
):
    """A claim mutator cannot collapse corrupt authority into an empty store."""
    projection = tmp_path / "state" / "cancel_intents.json"
    projection.parent.mkdir(parents=True)
    projection.write_bytes(corrupt_bytes)

    with pytest.raises(ci.CancelIntentProjectionCorrupt):
        ci.claim_intent(tmp_path, "corrupt-claim", owner="pending_drop")

    assert projection.read_bytes() == corrupt_bytes

def test_claim_intent_absent_projection_is_a_read_only_miss(tmp_path):
    projection = tmp_path / "state" / "cancel_intents.json"

    assert ci.claim_intent(tmp_path, "no-intent", owner="pending_drop") is None
    assert not projection.exists()

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

def _durable_events(root, event_type: str):
    path = root / "logs" / "events.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return [row for row in rows if row.get("type") == event_type]

def _write_pre_7_0_row(root, task_id: str, status: str, **fields):
    """A genuine pre-7.0 file: no ``_schema_version`` stamp anywhere."""
    results = root / "task_results"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{task_id}.json").write_text(
        json.dumps({"task_id": task_id, "status": status, **fields}), encoding="utf-8")

def test_boot_migration_admits_the_unstamped_latch_and_quarantines_the_rest(tmp_path):
    """ABI-2 carve-out (owner 4A) — the one exception to the Q8=B wholesale
    quarantine. A pre-7.0 result file is UNSTAMPED, so the first ordinary read
    moves it into ``task_results/quarantine/``; this scan therefore stopped
    seeing tasks wedged in the legacy ``cancel_requested`` latch and they
    reached no terminal at all. Boot now performs, for the LATCHED rows only,
    the same stamp-on-write a live pre-upgrade task performs on its next
    lifecycle write. Every other unstamped row still quarantines, and applying
    the carve-out is one typed durable fact."""
    _write_pre_7_0_row(tmp_path, "latched", STATUS_CANCEL_REQUESTED, note="pre-7.0")
    _write_pre_7_0_row(tmp_path, "finished", STATUS_COMPLETED, result="pre-7.0")

    assert ci.migrate_legacy_cancel_latches(tmp_path) == ["latched"]

    stored = load_task_result(tmp_path, "latched")
    assert stored["status"] == STATUS_CANCEL_REQUESTED and stored["note"] == "pre-7.0"
    assert stored[SCHEMA_VERSION_KEY] == TASK_RESULT_SCHEMA_VERSION
    assert ci.active_intent(tmp_path, "latched")["source"] == "boot_migration"
    # No latch, no carve-out: ordinary pre-7.0 history is quarantined as before.
    assert load_task_result(tmp_path, "finished") is None
    quarantined = (tmp_path / "task_results" / "quarantine").glob("*.json")
    assert [path.name for path in quarantined] == ["finished.json"]

    admitted = _durable_events(tmp_path, "task_result_cancel_latch_admitted")
    assert len(admitted) == 1
    assert admitted[0]["count"] == 1 and admitted[0]["task_ids"] == ["latched"]
    assert admitted[0]["reason"] == "unstamped_pre_7_0"
    # The next boot admits nothing and records no second fact.
    assert ci.migrate_legacy_cancel_latches(tmp_path) == []
    assert len(_durable_events(tmp_path, "task_result_cancel_latch_admitted")) == 1

def test_boot_migrates_the_latch_before_any_quarantining_read():
    """The carve-out is an ORDER as much as a write: whichever durable
    task-result read reaches a pre-7.0 latch first quarantines it, so the
    migration has to get there before them. The orphan reconcile used to win
    that race — it is the first such read of the boot — and the migration ran
    only later, in the custody sweep."""
    import inspect

    from ouroboros import server_maintenance

    recovery = inspect.getsource(server_maintenance._run_startup_task_recovery)
    assert recovery.index("migrate_legacy_cancel_latches") < recovery.index(
        "reconcile_orphaned_running_tasks"), recovery
    assert "migrate_legacy_cancel_latches" not in inspect.getsource(
        server_maintenance._startup_custody_sweep), (
        "the migration must not ALSO run from the later custody sweep, whose "
        "own reads would already have quarantined the latch"
    )

def test_the_admitted_latch_reaches_the_cancelled_terminal(tmp_path, monkeypatch):
    """The carve-out's whole point. Once admitted, the latch is an ORDINARY
    intent, so the existing custody miss lane (neither queued nor running)
    writes the durable ``cancelled`` terminal and settles the intent. Before
    it, that lane's own fail-soft read quarantined the row, found no durable
    result, and settled ``not_found`` — the task vanished without a terminal."""
    import supervisor.queue as q
    from supervisor import cancel_publication, task_lifecycle, terminal_delivery, workers

    _write_pre_7_0_row(tmp_path, "wedged", STATUS_CANCEL_REQUESTED,
                       description="wedged in the pre-redesign cancel latch")
    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "RUNNING", {}, raising=False)
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(q, "_emit_cancel_task_done", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(terminal_delivery, "deliver_miss_lane_outcome",
                        lambda *a, **kw: True, raising=False)
    monkeypatch.setattr(task_lifecycle, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(task_lifecycle, "_ACTIVE_CASCADE_FENCES", {}, raising=False)

    assert ci.migrate_legacy_cancel_latches(tmp_path) == ["wedged"]
    intent = ci.active_intent(tmp_path, "wedged")

    outcome = cancel_publication._finalize_cancel_intent_on_miss(q, "wedged", intent=intent)

    assert outcome == cancel_publication.CANCEL_CANCELLED
    assert load_task_result(tmp_path, "wedged")["status"] == STATUS_CANCELLED
    assert ci.active_intent(tmp_path, "wedged") is None

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
