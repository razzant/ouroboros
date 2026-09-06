from __future__ import annotations

import time
from types import SimpleNamespace

from tests._delivery_candidate_shared import write_child as _write_child


def _parent_ctx(tmp_path, task_id: str = "parent1") -> SimpleNamespace:
    return SimpleNamespace(
        drive_root=str(tmp_path),
        budget_drive_root=str(tmp_path),
        task_metadata={"budget_drive_root": str(tmp_path), "root_task_id": task_id},
        task_id=task_id,
        role="orchestrator",
    )


def _payload(child_id: str, disposition: str, result_sha256: str) -> dict:
    return {
        "type": "child_result_disposition",
        "child_task_id": child_id,
        "disposition": disposition,
        "child_result_sha256": result_sha256,
    }


def test_child_result_hash_has_exact_semantic_boundary():
    from ouroboros.tools.join_ledger import _child_result_sha256

    base = {
        "status": "completed",
        "result": "answer",
        "trace_summary": "trace",
        "artifact_status": "ready",
        "artifacts": [
            {
                "kind": "report",
                "name": "a.md",
                "sha256": "a" * 64,
                "path": "/tmp/one",
            }
        ],
    }
    reference = _child_result_sha256(base)
    assert _child_result_sha256(
        {
            **base,
            "cost_usd": 9.9,
            "updated_at": "tomorrow",
            "queue_reconciliation_warning": "diagnostic",
            "parent_decision": "cancelled",
            "child_result_disposition": "deferred",
            "child_result_disposition_sha256": "0" * 64,
        }
    ) == reference
    for field, value in (
        ("result", "changed"),
        ("status", "failed"),
        ("trace_summary", "changed"),
    ):
        assert _child_result_sha256({**base, field: value}) != reference
    assert _child_result_sha256(
        {
            **base,
            "artifacts": [
                {"kind": "report", "name": "a.md", "sha256": "b" * 64}
            ],
        }
    ) != reference


def test_task_tree_row_is_sole_authority_and_raw_result_is_unchanged(tmp_path):
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.task_tree_ledger import tree_ledger_rows
    from ouroboros.tools.join_ledger import (
        _child_result_sha256,
        _current_child_result_disposition,
    )
    from ouroboros.tools.task_tree import _tree_note

    _write_child(tmp_path)
    raw_before = load_task_result(tmp_path, "child1")
    shown_hash = _child_result_sha256(load_effective_task_result(tmp_path, "child1"))
    payload = _payload("child1", "integrated", shown_hash)

    result = _tree_note(
        _parent_ctx(tmp_path),
        "decision",
        "used the complete child analysis",
        payload=payload,
    )
    assert result.startswith("OK:")
    assert load_task_result(tmp_path, "child1") == raw_before
    rows = tree_ledger_rows("parent1", data_root=tmp_path)
    assert len(rows) == 1
    assert rows[0]["payload"] == payload

    effective = load_effective_task_result(tmp_path, "child1")
    assert effective["child_result_disposition"] == "integrated"
    assert effective["child_result_disposition_sha256"] == shown_hash
    assert effective["child_result_disposition_source"] == "task_tree_ledger"
    assert _current_child_result_disposition(effective) == "integrated"

    retry = _tree_note(
        _parent_ctx(tmp_path),
        "decision",
        "used the complete child analysis",
        payload=payload,
    )
    assert "idempotent" in retry
    assert tree_ledger_rows("parent1", data_root=tmp_path) == rows


def test_changed_child_result_reopens_and_old_hash_is_stale(tmp_path):
    from ouroboros.task_results import write_task_result
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.task_tree_ledger import tree_ledger_rows
    from ouroboros.tools.join_ledger import (
        _child_result_sha256,
        _current_child_result_disposition,
    )
    from ouroboros.tools.task_tree import _tree_note

    _write_child(tmp_path)
    before = load_effective_task_result(tmp_path, "child1")
    old_hash = _child_result_sha256(before)
    payload = _payload("child1", "integrated", old_hash)
    assert _tree_note(
        _parent_ctx(tmp_path), "decision", "integrated", payload=payload
    ).startswith("OK:")
    rows_before = tree_ledger_rows("parent1", data_root=tmp_path)

    write_task_result(tmp_path, "child1", "completed", result="new child result")
    changed = load_effective_task_result(tmp_path, "child1")
    assert _child_result_sha256(changed) != old_hash
    assert _current_child_result_disposition(changed) == ""
    assert "child_result_disposition" not in changed
    stale = _tree_note(
        _parent_ctx(tmp_path), "decision", "still integrated", payload=payload
    )
    assert "CHILD_RESULT_STALE" in stale
    assert tree_ledger_rows("parent1", data_root=tmp_path) == rows_before


def test_artifact_change_reopens_exact_hash_disposition(tmp_path):
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.task_results import write_task_result
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import (
        _child_result_sha256,
        _current_child_result_disposition,
    )
    from ouroboros.tools.task_tree import _tree_note

    child_id = "artifact-child"
    write_task_result(
        tmp_path,
        child_id,
        "completed",
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        result="artifact-backed result",
    )
    artifact_dir = task_artifact_dir_path(tmp_path, child_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "report.md"
    artifact_path.write_text("version one\n", encoding="utf-8")
    shown_hash = _child_result_sha256(load_effective_task_result(tmp_path, child_id))
    assert _tree_note(
        _parent_ctx(tmp_path),
        "decision",
        "integrated artifact",
        payload=_payload(child_id, "integrated", shown_hash),
    ).startswith("OK:")
    assert _current_child_result_disposition(
        load_effective_task_result(tmp_path, child_id)
    ) == "integrated"

    artifact_path.write_text("version two\n", encoding="utf-8")
    changed = load_effective_task_result(tmp_path, child_id)
    assert _child_result_sha256(changed) != shown_hash
    assert _current_child_result_disposition(changed) == ""


def test_malformed_tagged_payloads_have_zero_mutation(tmp_path):
    from ouroboros.task_results import load_task_result
    from ouroboros.task_tree_ledger import tree_ledger_rows
    from ouroboros.tools.join_ledger import _child_result_sha256
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.task_tree import _tree_note

    _write_child(tmp_path)
    raw_before = load_task_result(tmp_path, "child1")
    digest = _child_result_sha256(load_effective_task_result(tmp_path, "child1"))
    malformed = [
        {**_payload("child1", "integrated", digest), "extra": True},
        _payload("child1", "unknown", digest),
        _payload("child1", "integrated", "0" * 63),
        {"type": "child_result_disposition", "child_task_id": "child1"},
    ]
    for payload in malformed:
        result = _tree_note(
            _parent_ctx(tmp_path), "decision", "rationale", payload=payload
        )
        assert "INVALID" in result
    wrong_kind = _tree_note(
        _parent_ctx(tmp_path),
        "note",
        "rationale",
        payload=_payload("child1", "integrated", digest),
    )
    assert "INVALID" in wrong_kind
    assert load_task_result(tmp_path, "child1") == raw_before
    assert tree_ledger_rows("parent1", data_root=tmp_path) == []


def test_malformed_disposition_names_every_violation_in_one_reply(tmp_path):
    """W2: aggregated diagnostics — the old one-error-per-round shape cost a live
    parent 9 paid rounds discovering the constraints serially. One malformed call
    now returns EVERY violated constraint plus a correct example, and stays an
    atomic no-op (no truncation, no superset-key acceptance)."""
    from ouroboros.task_tree_ledger import tree_ledger_rows
    from ouroboros.tools.task_tree import _tree_note

    _write_child(tmp_path)
    bad = {
        "type": "child_result_disposition",
        "child_task_id": "child1",
        "disposition": "absorbed",          # not in the enum
        "child_result_sha256": "0" * 63,     # not 64-hex
        "supports_claims": ["claim_1"],      # unknown key: rejected, never ignored
    }
    result = _tree_note(_parent_ctx(tmp_path), "decision", "x" * 501, payload=bad)

    assert result.count("CHILD_RESULT_DISPOSITION_INVALID") == 1
    for fragment in (
        "unknown key(s) supports_claims",
        "disposition must be one of",
        "child_result_sha256 must be the 64-char hex sha",
        "at most 500 characters",
        "Correct example",
        "atomic no-op",
    ):
        assert fragment in result, fragment
    assert tree_ledger_rows("parent1", data_root=tmp_path) == []

    # A valid payload with a MISSING rationale also gets the aggregated shape.
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256

    digest = _child_result_sha256(load_effective_task_result(tmp_path, "child1"))
    no_reason = _tree_note(
        _parent_ctx(tmp_path), "decision", "  ", payload=_payload("child1", "integrated", digest),
    )
    assert "tree_note text is required as the rationale" in no_reason
    assert "Correct example" in no_reason


def test_ledger_append_renders_the_same_aggregated_violations(tmp_path):
    """One contract, ONE diagnostic authority: the ledger append path renders the
    aggregated violations too, instead of keeping a second, weaker one-line
    message for the same closed key set (it is unreachable through join_ledger
    today, so the two could drift apart unnoticed)."""
    from ouroboros.task_tree_ledger import CHILD_RESULT_DISPOSITION_TYPE, tree_ledger_append

    out = tree_ledger_append(
        "root1", "decision", "why",
        task_id="parent1",
        payload={
            "type": CHILD_RESULT_DISPOSITION_TYPE,
            "child_task_id": "child1",
            "disposition": "absorbed",       # not in the enum
            "child_result_sha256": "0" * 63,  # not 64-hex
            "supports_claims": ["claim_1"],   # unknown key
        },
        allow_child_result_disposition=True,
        data_root=tmp_path,
    )

    assert out.startswith("⚠️ CHILD_RESULT_DISPOSITION_INVALID:")
    for fragment in (
        "unknown key(s) supports_claims",
        "disposition must be one of",
        "child_result_sha256 must be the 64-char hex sha",
    ):
        assert fragment in out, fragment
    # The superseded single-line message is gone, not merely shadowed.
    assert "payload must contain exactly" not in out


def test_disposition_violations_helper_is_the_normalizer_authority():
    from ouroboros.task_tree_ledger import (
        child_result_disposition_violations,
        normalize_child_result_disposition_payload,
    )

    good = _payload("child1", "integrated", "a" * 64)
    assert child_result_disposition_violations(good) == []
    assert normalize_child_result_disposition_payload(good) is not None
    for bad in (
        None,
        "text",
        {**good, "extra": 1},
        {**good, "disposition": "unknown"},
        {**good, "child_result_sha256": "xyz"},
        {"type": "child_result_disposition"},
    ):
        assert child_result_disposition_violations(bad), bad
        assert normalize_child_result_disposition_payload(bad) is None


def test_batch_disposition_records_one_authoritative_row_per_child(tmp_path):
    """Q2A (slime saga): a fan-out parent needed one bureaucratic tree_note per
    child (six calls in the incident). One call with a children array now
    expands into the SAME per-child authoritative rows as the single form, so
    every existing reader (projection, absorption gate) is unchanged."""
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.task_tree_ledger import tree_ledger_rows
    from ouroboros.tools.join_ledger import _child_result_sha256
    from ouroboros.tools.task_tree import _tree_note

    entries = []
    for child_id, disposition in (
        ("child1", "integrated"), ("child2", "irrelevant"), ("child3", "deferred"),
    ):
        _write_child(tmp_path, child_id=child_id)
        entries.append({
            "child_task_id": child_id,
            "disposition": disposition,
            "child_result_sha256": _child_result_sha256(
                load_effective_task_result(tmp_path, child_id)
            ),
        })

    result = _tree_note(
        _parent_ctx(tmp_path),
        "decision",
        "batch disposition after absorbing all three",
        payload={"type": "child_result_disposition", "children": entries},
    )

    assert result.startswith("OK: batch child disposition recorded for 3 child(ren).")
    rows = tree_ledger_rows("parent1", data_root=tmp_path)
    assert len(rows) == 3
    recorded = {row["payload"]["child_task_id"]: row["payload"] for row in rows}
    for entry in entries:
        payload = recorded[entry["child_task_id"]]
        assert payload["disposition"] == entry["disposition"]
        assert payload["child_result_sha256"] == entry["child_result_sha256"]
        effective = load_effective_task_result(tmp_path, entry["child_task_id"])
        assert effective["child_result_disposition"] == entry["disposition"]
        assert effective["child_result_disposition_source"] == "task_tree_ledger"


def test_batch_disposition_rejects_invalid_entries_individually(tmp_path):
    """Exact-hash binding is preserved PER CHILD: a stale hash or a foreign task
    rejects only its own entry — the clear error names which entries failed —
    while valid entries still record."""
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.task_tree_ledger import tree_ledger_rows
    from ouroboros.tools.join_ledger import _child_result_sha256
    from ouroboros.tools.task_tree import _tree_note

    _write_child(tmp_path)
    _write_child(tmp_path, child_id="child2")
    good_hash = _child_result_sha256(load_effective_task_result(tmp_path, "child1"))

    result = _tree_note(
        _parent_ctx(tmp_path),
        "decision",
        "partial batch",
        payload={
            "type": "child_result_disposition",
            "children": [
                {"child_task_id": "child1", "disposition": "integrated",
                 "child_result_sha256": good_hash},
                {"child_task_id": "child2", "disposition": "integrated",
                 "child_result_sha256": "0" * 64},          # stale hash
                {"child_task_id": "stranger9", "disposition": "irrelevant",
                 "child_result_sha256": "1" * 64},          # not our child
                {"child_task_id": "child2", "disposition": "absorbed",
                 "child_result_sha256": "bad"},             # enum + sha violations
                "not-an-object",
            ],
        },
    )

    assert result.startswith("⚠️ CHILD_RESULT_DISPOSITION_PARTIAL: 1/5")
    assert "[child1] OK:" in result
    assert "[child2] ⚠️ CHILD_RESULT_STALE" in result
    assert "[stranger9] ⚠️ CHILD_RESULT_LINEAGE_FORBIDDEN" in result
    assert "disposition must be one of" in result
    assert "[entry 4] ⚠️ CHILD_RESULT_DISPOSITION_INVALID: entry must be a JSON object." in result
    rows = tree_ledger_rows("parent1", data_root=tmp_path)
    assert [row["payload"]["child_task_id"] for row in rows] == ["child1"]


def test_batch_disposition_envelope_is_validated_atomically(tmp_path):
    """A malformed batch ENVELOPE (empty/non-array children, stray keys mixing
    the single and batch forms) records nothing."""
    from ouroboros.task_tree_ledger import tree_ledger_rows
    from ouroboros.tools.task_tree import _tree_note

    _write_child(tmp_path)
    for payload in (
        {"type": "child_result_disposition", "children": []},
        {"type": "child_result_disposition", "children": "child1"},
        {"type": "child_result_disposition", "children": [], "child_task_id": "child1"},
    ):
        result = _tree_note(_parent_ctx(tmp_path), "decision", "why", payload=payload)
        assert "CHILD_RESULT_DISPOSITION_INVALID" in result
        assert "atomic no-op" in result
    wrong_kind = _tree_note(
        _parent_ctx(tmp_path),
        "note",
        "why",
        payload={"type": "child_result_disposition", "children": [
            {"child_task_id": "child1", "disposition": "integrated",
             "child_result_sha256": "a" * 64},
        ]},
    )
    assert "require kind='decision'" in wrong_kind
    assert tree_ledger_rows("parent1", data_root=tmp_path) == []


def test_orphan_note_claim_detail_is_scoped_to_undecided_children(monkeypatch):
    """The blackboard-derived claim detail belongs ONLY to children the exact-hash
    disposition projection left UNDECIDED. A deferred child IS carried by that
    projection (that is why it lands on the deferred list) and its row DOES bind,
    so "not carried by this round's disposition projection — re-submit to close it"
    would be a provably false owner-visible instruction."""
    import ouroboros.loop as loop
    from ouroboros.tools.join_ledger import _child_result_sha256

    child = {"task_id": "child1", "status": "completed", "result": "child work"}
    digest = _child_result_sha256(child)
    # The real projection fields: disposition rows are excluded from the hash, so
    # this is a genuinely CARRIED, exactly-bound deferral (no monkeypatched state).
    deferred_child = {
        **child,
        "child_result_disposition_source": "task_tree_ledger",
        "child_result_disposition": "deferred",
        "child_result_disposition_sha256": digest,
    }
    monkeypatch.setattr(loop, "_direct_child_results", lambda _ctx: [dict(deferred_child)])
    monkeypatch.setattr(
        loop,
        "_claimed_child_dispositions",
        lambda _ctx: {"child1": ("deferred", digest)},
    )

    note = loop._forced_orphan_note(SimpleNamespace())

    assert "DEFERRED CHILD RESULTS: child1 [completed]" in note
    assert "re-submit" not in note
    assert "disposition projection" not in note

    # The undecided child the detail was written for still gets it.
    monkeypatch.setattr(loop, "_direct_child_results", lambda _ctx: [dict(child)])
    undecided_note = loop._forced_orphan_note(SimpleNamespace())
    assert "recorded for this exact result hash" in undecided_note
    assert "re-submit to close it" in undecided_note


def test_non_child_lineage_is_rejected_without_a_row(tmp_path):
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.task_tree_ledger import tree_ledger_rows
    from ouroboros.tools.join_ledger import _child_result_sha256
    from ouroboros.tools.task_tree import _tree_note

    write_task_result(
        tmp_path,
        "stranger",
        "completed",
        parent_task_id="other-parent",
        root_task_id="other-parent",
        delegation_role="subagent",
        result="not yours",
    )
    raw_before = load_task_result(tmp_path, "stranger")
    digest = _child_result_sha256(load_effective_task_result(tmp_path, "stranger"))
    result = _tree_note(
        _parent_ctx(tmp_path),
        "decision",
        "invalid lineage",
        payload=_payload("stranger", "irrelevant", digest),
    )
    assert "LINEAGE_FORBIDDEN" in result
    assert load_task_result(tmp_path, "stranger") == raw_before
    assert tree_ledger_rows("parent1", data_root=tmp_path) == []


def test_append_failure_has_zero_task_result_mutation_and_retry_works(
    tmp_path, monkeypatch
):
    import ouroboros.tools.join_ledger as join_ledger
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.task_tree_ledger import tree_ledger_rows
    from ouroboros.tools.task_tree import _tree_note

    _write_child(tmp_path)
    raw_before = load_task_result(tmp_path, "child1")
    digest = join_ledger._child_result_sha256(
        load_effective_task_result(tmp_path, "child1")
    )
    payload = _payload("child1", "deferred", digest)
    real_append = join_ledger.tree_ledger_append
    monkeypatch.setattr(
        join_ledger,
        "tree_ledger_append",
        lambda *args, **kwargs: "⚠️ TREE_LEDGER_WRITE_FAILED",
    )
    failed = _tree_note(
        _parent_ctx(tmp_path), "decision", "later", payload=payload
    )
    assert "WRITE_FAILED" in failed
    assert load_task_result(tmp_path, "child1") == raw_before
    assert tree_ledger_rows("parent1", data_root=tmp_path) == []

    monkeypatch.setattr(join_ledger, "tree_ledger_append", real_append)
    assert _tree_note(
        _parent_ctx(tmp_path), "decision", "later", payload=payload
    ).startswith("OK:")


def test_latest_valid_row_wins_for_same_exact_hash(tmp_path):
    from ouroboros.loop import _child_disposition_state
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.task_tree_ledger import tree_ledger_rows
    from ouroboros.tools.join_ledger import _child_result_sha256
    from ouroboros.tools.task_tree import _tree_note

    _write_child(tmp_path)
    digest = _child_result_sha256(load_effective_task_result(tmp_path, "child1"))
    ctx = _parent_ctx(tmp_path)
    assert _tree_note(
        ctx,
        "decision",
        "defer until synthesis",
        payload=_payload("child1", "deferred", digest),
    ).startswith("OK:")
    assert _tree_note(
        ctx,
        "decision",
        "now integrated",
        payload=_payload("child1", "integrated", digest),
    ).startswith("OK:")
    assert len(tree_ledger_rows("parent1", data_root=tmp_path)) == 2
    effective = load_effective_task_result(tmp_path, "child1")
    assert _child_disposition_state(effective) == "integrated"
    assert effective["child_result_disposition_reason"] == "now integrated"


def test_legacy_task_result_disposition_fields_are_not_authority(tmp_path):
    from ouroboros.loop import _child_disposition_state
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import load_effective_task_result

    _write_child(
        tmp_path,
        child_result_disposition="integrated",
        child_result_disposition_sha256="a" * 64,
        child_result_disposition_reason="legacy mirror",
        child_result_disposition_source="old_writer",
        child_result_disposition_beacon_state="confirmed",
        child_result_disposition_beacon_sha256="b" * 64,
        parent_decision="discarded",
        parent_decision_child_result_sha256="a" * 64,
    )
    assert "child_result_disposition" in load_task_result(tmp_path, "child1")
    effective = load_effective_task_result(tmp_path, "child1")
    assert "child_result_disposition" not in effective
    assert "child_result_disposition_beacon_state" not in effective
    assert _child_disposition_state(effective) == ""


def test_tree_gc_removes_ephemeral_disposition_authority(tmp_path):
    from ouroboros.headless import prune_task_trees
    from ouroboros.task_results import write_task_result
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256
    from ouroboros.tools.task_tree import _tree_note

    _write_child(tmp_path)
    digest = _child_result_sha256(load_effective_task_result(tmp_path, "child1"))
    assert _tree_note(
        _parent_ctx(tmp_path),
        "decision",
        "integrated before root completion",
        payload=_payload("child1", "integrated", digest),
    ).startswith("OK:")
    assert load_effective_task_result(tmp_path, "child1")[
        "child_result_disposition"
    ] == "integrated"
    write_task_result(tmp_path, "parent1", "completed", result="root done")
    report = prune_task_trees(
        tmp_path,
        retention_days=1,
        now=time.time() + 3 * 86400,
    )
    assert report["pruned"]
    assert "child_result_disposition" not in load_effective_task_result(
        tmp_path, "child1"
    )


def test_cancellation_wins_and_late_scratch_result_is_deleted(tmp_path):
    from ouroboros.headless import HEADLESS_TASKS_DIR, remove_subagent_task_drive
    from ouroboros.loop import _child_disposition_state
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result, write_task_result
    from ouroboros.task_status import load_effective_task_result

    child_id = "cancel-race"
    write_task_result(
        tmp_path,
        child_id,
        STATUS_CANCELLED,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        parent_decision="cancelled",
        result="Task cancelled.",
    )
    scratch = tmp_path / HEADLESS_TASKS_DIR / child_id / "data"
    write_task_result(
        scratch,
        child_id,
        "completed",
        result="late result that must be ignored",
        trace_summary="late trace",
    )

    assert remove_subagent_task_drive(tmp_path, child_id) is True
    assert not scratch.parent.exists()
    raw = load_task_result(tmp_path, child_id) or {}
    assert raw["status"] == STATUS_CANCELLED
    assert "terminal_child_result_snapshot" not in raw
    assert "late result that must be ignored" not in str(raw)
    assert _child_disposition_state(
        load_effective_task_result(tmp_path, child_id)
    ) == "cancelled"


def test_legacy_cancel_requested_latch_is_pending_not_handled(tmp_path):
    """GR2-8c: only a SETTLED ``cancelled`` counts as a handled disposition.

    The legacy ``cancel_requested`` STATUS is an unsettled latch — intent, not
    outcome (phase A moved intent to the durable cancel_state projection).
    Treating it as "cancelled" suppressed the parent's pending-child handoff
    reminder for a child the supervisor was still tearing down; such a child
    must stay visible as cancel-pending until custody settles it.
    """
    from ouroboros.loop import _child_disposition_state
    from ouroboros.task_results import STATUS_CANCEL_REQUESTED, write_task_result
    from ouroboros.task_status import load_effective_task_result

    child_id = "legacy-latch-child"
    write_task_result(
        tmp_path,
        child_id,
        STATUS_CANCEL_REQUESTED,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        parent_decision="cancelled",
        result="wedged in the pre-redesign latch",
    )

    effective = load_effective_task_result(tmp_path, child_id)
    assert _child_disposition_state(effective) != "cancelled", (
        "an unsettled latch must not read as a handled cancellation"
    )
    # ...and the same row with a SETTLED cancelled status IS handled.
    write_task_result(tmp_path, child_id, "cancelled", parent_decision="cancelled")
    assert _child_disposition_state(
        load_effective_task_result(tmp_path, child_id)
    ) == "cancelled"


def test_settled_cancel_blocks_custom_child_read_and_artifact_copy(
    tmp_path, monkeypatch
):
    """Only a SUPERVISOR-SETTLED ``cancelled`` blocks child-drive promotion.

    Re-pinned for the phase-A cancel redesign: the blocking authority used to be
    the cancel-INTENT latch, which is exactly what erased a child that had
    already finished. The settled outcome still blocks (the task's terminal truth
    is decided); the legacy latch no longer does — see the test below.
    """
    import pathlib

    import ouroboros.headless as headless
    import ouroboros.task_status as task_status
    from ouroboros.task_results import (
        STATUS_CANCELLED,
        load_task_result,
        write_task_result,
    )

    parent = tmp_path / "parent"
    custom_child = tmp_path / "surviving-custom-child"
    child_id = "cancel-copy-race"
    late_artifact = custom_child / "late.txt"
    late_artifact.parent.mkdir(parents=True)
    late_artifact.write_text("late artifact", encoding="utf-8")
    write_task_result(
        custom_child,
        child_id,
        "completed",
        result="late completed result",
        trace_summary="late completed trace",
        artifacts=[{
            "kind": "report",
            "name": late_artifact.name,
            "path": str(late_artifact),
        }],
    )
    write_task_result(
        parent,
        child_id,
        STATUS_CANCELLED,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        parent_decision="cancelled",
        result="canonical settled cancellation",
        trace_summary="canonical cancellation trace",
        child_drive_root=str(custom_child),
    )
    canonical = load_task_result(parent, child_id) or {}

    # A surviving custom child root models failed scratch cleanup. Neither the
    # effective reader nor copy-back may even read it once cancellation SETTLED.
    real_load = load_task_result
    observed_roots: list[pathlib.Path] = []

    def guarded_load(root, task_id):
        resolved = pathlib.Path(root).resolve(strict=False)
        observed_roots.append(resolved)
        assert resolved != custom_child.resolve(strict=False)
        return real_load(root, task_id)

    monkeypatch.setattr(task_status, "load_task_result", guarded_load)
    monkeypatch.setattr(headless, "load_task_result", guarded_load)

    effective = task_status.load_effective_task_result(parent, child_id)
    assert effective["status"] == STATUS_CANCELLED
    assert effective["result"] == "canonical settled cancellation"
    assert effective["trace_summary"] == "canonical cancellation trace"
    assert "child_status" not in effective
    assert not effective.get("artifacts")

    copied = headless.copy_child_task_result(
        parent,
        {
            "id": child_id,
            "drive_root": str(custom_child),
            "delegation_role": "subagent",
        },
    )
    assert copied == canonical
    assert observed_roots and all(root == parent.resolve() for root in observed_roots)
    assert not list(parent.rglob(late_artifact.name))
    assert load_task_result(parent, child_id) == canonical


def test_legacy_cancel_latch_no_longer_blocks_child_promotion(tmp_path):
    """A pre-redesign ``cancel_requested`` file must NOT bury a finished child.

    This is the incident itself: the latch counted as terminal, so the post-kill
    re-check read it back and the child's completed answer was deleted with its
    drive. Completion wins now (owner 4=A) — an old latch file is a cancel
    REQUEST, and a child that finished first keeps its result.
    """
    import ouroboros.headless as headless
    import ouroboros.task_status as task_status
    from ouroboros.task_results import (
        STATUS_CANCEL_REQUESTED,
        STATUS_COMPLETED,
        load_task_result,
        write_task_result,
    )

    parent = tmp_path / "parent"
    custom_child = tmp_path / "child-drive"
    child_id = "legacy-latch-child"
    write_task_result(
        custom_child,
        child_id,
        STATUS_COMPLETED,
        result="the finished child answer",
        trace_summary="did the work",
    )
    write_task_result(
        parent,
        child_id,
        STATUS_CANCEL_REQUESTED,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        result="legacy cancellation request",
        child_drive_root=str(custom_child),
    )

    copied = headless.copy_child_task_result(
        parent,
        {
            "id": child_id,
            "drive_root": str(custom_child),
            "delegation_role": "subagent",
        },
    )
    assert copied is not None and copied["status"] == STATUS_COMPLETED
    assert copied["result"] == "the finished child answer"
    stored = load_task_result(parent, child_id) or {}
    assert stored["status"] == STATUS_COMPLETED
    # The effective read surfaces the promoted completion, not the old latch.
    effective = task_status.load_effective_task_result(parent, child_id)
    assert effective["status"] == STATUS_COMPLETED


def test_settled_cancel_blocks_finalizer_before_workspace_or_child_reads(
    tmp_path, monkeypatch
):
    """Same re-pin for artifact finalization: SETTLED cancelled blocks, latch does not."""
    import ouroboros.headless as headless
    from ouroboros.task_results import (
        STATUS_CANCELLED,
        load_task_result,
        write_task_result,
    )

    parent = tmp_path / "parent"
    workspace = tmp_path / "late-workspace"
    custom_child = tmp_path / "surviving-custom-child"
    workspace.mkdir()
    custom_child.mkdir()
    (workspace / "late.txt").write_text("late workspace change", encoding="utf-8")
    (custom_child / "late-memory.txt").write_text("late child data", encoding="utf-8")
    child_id = "cancel-finalize-race"
    write_task_result(
        parent,
        child_id,
        STATUS_CANCELLED,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        parent_decision="cancelled",
        result="canonical settled cancellation",
        trace_summary="canonical cancellation trace",
        child_drive_root=str(custom_child),
        workspace_root=str(workspace),
    )
    canonical = load_task_result(parent, child_id) or {}

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled task touched a late artifact surface")

    monkeypatch.setattr(headless, "task_artifacts_dir", forbidden)
    monkeypatch.setattr(headless, "_workspace_root_from_task", forbidden)
    monkeypatch.setattr(headless, "_child_drive_from_task", forbidden)
    monkeypatch.setattr(headless, "write_workspace_patch_artifacts", forbidden)
    monkeypatch.setattr(headless, "build_memory_export", forbidden)

    artifacts = headless.finalize_task_artifacts(
        parent,
        {
            "id": child_id,
            "workspace_root": str(workspace),
            "drive_root": str(custom_child),
            "delegation_role": "subagent",
        },
    )
    assert artifacts == []
    assert not (parent / "task_results" / "artifacts" / child_id).exists()
    assert load_task_result(parent, child_id) == canonical


def test_full_blackboard_never_locks_out_validated_dispositions(tmp_path, monkeypatch):
    """A chatty swarm filling the ledger must not block the disposition authority."""
    from ouroboros import task_tree_ledger as ttl

    monkeypatch.setattr(ttl, "_MAX_LEDGER_BYTES", 64)
    root_id = "root-full-ledger"
    path = ttl.tree_ledger_path(root_id, data_root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x" * 200 + "\n", encoding="utf-8")

    # Ordinary notes are refused once the blackboard is full...
    refused = ttl.tree_ledger_append(
        root_id, "note", "just chatter", task_id="t1", data_root=tmp_path,
    )
    assert "ledger is full" in refused

    # ...but a validated child-result disposition still lands.
    accepted = ttl.tree_ledger_append(
        root_id,
        "decision",
        "reject: superseded",
        task_id="t1",
        payload={
            "type": ttl.CHILD_RESULT_DISPOSITION_TYPE,
            "child_task_id": "c1",
            "disposition": "irrelevant",
            "child_result_sha256": "a" * 64,
        },
        allow_child_result_disposition=True,
        data_root=tmp_path,
    )
    assert "ledger is full" not in accepted
    assert not accepted.startswith("⚠️")


def test_the_disposition_enum_and_its_cost_come_from_one_place():
    """The enum was hand-written twice with no per-value meaning, although the
    consequence of ``deferred`` is real and host-enforced. Both payload sites
    now read the validator's own set, in a stable order, and the schema states
    what each value costs — the schema is where the model actually reads it."""
    from ouroboros.task_tree_ledger import CHILD_RESULT_DISPOSITIONS
    from ouroboros.tools import task_tree

    entry = next(tool for tool in task_tree.get_tools() if tool.name == "tree_note")
    payload = entry.schema["parameters"]["properties"]["payload"]["properties"]
    single = payload["disposition"]
    batch = payload["children"]["items"]["properties"]["disposition"]

    assert single is batch or single == batch
    for schema in (single, batch):
        assert schema["enum"] == sorted(CHILD_RESULT_DISPOSITIONS)
        assert schema["enum"] == ["deferred", "integrated", "irrelevant"]
        for value in CHILD_RESULT_DISPOSITIONS:
            assert value in schema["description"], value
        assert "degraded/best_effort" in schema["description"]
