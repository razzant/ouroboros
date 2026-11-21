"""Pending-invocation recovery must preserve one exact physical intention."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros.delegate_shared import _fail, delegate_result


def _session_snapshot() -> dict:
    from ouroboros.subagent_runtime import select_subagent_snapshot

    settings = {
        "OUROBOROS_SUBAGENTS": json.dumps({
            "enabled": True,
            "items": [{
                "subagent_id": "session-builder",
                "name": "Session builder",
                "recommended_use": "Subscription-backed implementation.",
                "route": {
                    "kind": "agent_session",
                    "target_id": "codex=gpt-5.6-sol",
                    "credential_profile_id": "profile-1",
                },
                "effort": "high",
            }],
        }),
    }
    return select_subagent_snapshot(settings, subagent_id="session-builder")[0]


def _pending_handoff(tmp_path, task_id: str, *, input_files=None):
    import ouroboros.delegate_recovery as recovery
    from ouroboros import delegate_custody as custody
    from ouroboros.subagent_work_order import work_order_fingerprint

    custody._CUSTODY.clear()
    snapshot = _session_snapshot()
    task = {
        "id": task_id,
        "_attempt": 2,
        "configured_subagent": snapshot,
        "drive_root": str(tmp_path),
        "task_constraint": {},
        "task_contract": {"objective": "Build", "expected_output": "Patch"},
    }
    if input_files is not None:
        task["prepared_input_files"] = input_files
    authority = recovery.authority_fingerprint_from_task(task)
    fingerprint = work_order_fingerprint(task)
    invocation_id = f"inv-{task_id}"
    assert custody.record_start_requested(
        tmp_path,
        run_id="",
        task_id=task_id,
        invocation_id=invocation_id,
        idempotency_key=invocation_id,
        max_seconds=60,
        request={"prompt": "stored canonical request"},
        project_id="",
        project_owned=False,
        route="codex",
        selected_subagent_id=snapshot["selected_subagent_id"],
        config_fingerprint=snapshot["config_fingerprint"],
        work_order_fingerprint=fingerprint,
        authority_fingerprint=authority,
    )
    handoff = recovery.prepare_handoff(
        tmp_path,
        task,
        cause=recovery.CAUSE_WORKER_CRASH,
        old_attempt=1,
        new_attempt=2,
        worker_id=1,
        exitcode=1,
    )
    assert handoff["pending_invocation_id"] == invocation_id
    return custody, recovery, task, snapshot, invocation_id, fingerprint, authority


def test_exact_pending_retry_bypasses_fresh_zero_run_unknown_fence(
    monkeypatch, tmp_path,
):
    import ouroboros.subagent_runtime as runtime
    import ouroboros.tools.delegate as delegate
    from ouroboros import delegate_custody as custody

    calls = []
    monkeypatch.setattr(
        delegate,
        "_delegate_start",
        lambda _ctx, prompt, _max_seconds, retry_of, **_kwargs: (
            calls.append((prompt, retry_of))
            or delegate_result({"status": "started", "run_id": "run-recovered"})
        ),
    )
    ctx = SimpleNamespace(
        task_id="retry-with-zero-run-gap",
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
        _configured_actor_bootstrap={
            "zero_run_evidence_status": "unknown",
            "zero_run_evidence_gaps": ["verification_receipts_incomplete"],
            "exact_start_pending": False,
        },
    )
    assert custody.record_start_requested(
        tmp_path,
        run_id="",
        task_id=ctx.task_id,
        invocation_id="inv-recovery",
        idempotency_key="inv-recovery",
        max_seconds=60,
        request={"prompt": "stored canonical request"},
        project_id="",
        project_owned=False,
        route="codex",
    )

    result = json.loads(runtime.exact_start(
        ctx, "stored canonical request", {"retry_of": "inv-recovery"},
    ).text)

    assert result["status"] == "started"
    assert calls == [("stored canonical request", "inv-recovery")]


def test_unknown_retry_token_cannot_bypass_zero_run_unknown_fence(
    monkeypatch, tmp_path,
):
    import ouroboros.subagent_runtime as runtime
    import ouroboros.tools.delegate as delegate

    monkeypatch.setattr(
        delegate,
        "_delegate_start",
        lambda *_args, **_kwargs: pytest.fail(
            "an unowned or missing retry token must not bypass the zero-run fence"
        ),
    )
    ctx = SimpleNamespace(
        task_id="retry-without-pending-invocation",
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
        _configured_actor_bootstrap={
            "zero_run_evidence_status": "unknown",
            "zero_run_evidence_gaps": ["verification_receipts_incomplete"],
            "exact_start_pending": False,
        },
    )

    result = json.loads(runtime.exact_start(
        ctx, "stored canonical request", {"retry_of": "unknown-invocation"},
    ).text)

    assert result["status"] == "refused"
    assert result["reason"] == "zero_run_evidence_unavailable"


def test_retry_refusal_preserves_live_owner_handoff_and_pending_invocation(
    monkeypatch, tmp_path,
):
    import ouroboros.tools.delegate as delegate
    from ouroboros.tools.registry import ToolContext

    custody, recovery, task, _snapshot, invocation_id, _fingerprint, _authority = (
        _pending_handoff(tmp_path, "recovery-deferred")
    )
    monkeypatch.setattr(
        delegate,
        "exact_start",
        lambda *_args, **_kwargs: _fail(
            "delegate_start", "subscription_window_exhausted",
            "the engine subscription window is spent",
        ),
    )
    monkeypatch.setattr(
        custody,
        "reconcile_task_runs",
        lambda *_args, **_kwargs: pytest.fail(
            "a live recovery owner must not enter owner-gone reconciliation"
        ),
    )
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task["id"])
    ctx.budget_drive_root = str(tmp_path)

    result = recovery.adopt_handoff(ctx, task)

    assert result["reason"] == "pending_recovery_deferred"
    assert result["pending_invocation_id"] == invocation_id
    assert recovery._read(tmp_path, task["id"])["holders"][0]["status"] == "deferred"
    assert task["id"] in recovery.recoverable_task_ids(tmp_path)
    assert [row["invocation_id"] for row in custody.pending_invocations(tmp_path)] == [
        invocation_id
    ]

    replayed = []

    class OrphanGateway:
        def handshake(self):
            return {}

        def start_run(self, _body, *, idempotency_key):
            replayed.append(idempotency_key)
            return {}

        def close(self):
            pass

    custody.reconcile_orphaned_runs(
        tmp_path,
        running_task_ids=set(),
        recoverable_task_ids=recovery.recoverable_task_ids(tmp_path),
        gateway_factory=lambda: OrphanGateway(),
        replay_pending=lambda: True,  # the admitted sweep is the one surface that replays
    )
    assert replayed == []  # the exact deferred handle remains reserved
    custody._CUSTODY.clear()


def test_definite_retry_refusal_retires_handoff_without_false_pending_claim(
    monkeypatch, tmp_path,
):
    import ouroboros.tools.delegate as delegate
    from ouroboros.tools.registry import ToolContext

    custody, recovery, task, _snapshot, invocation_id, _fingerprint, _authority = (
        _pending_handoff(tmp_path, "recovery-definite")
    )

    def refuse_definitely(*_args, **_kwargs):
        assert custody.emit(tmp_path, custody.START_FAILED, {
            "task_id": task["id"],
            "invocation_id": invocation_id,
            "definite": True,
        })
        return _fail("delegate_start", "route_refused", "the engine refused this route")

    monkeypatch.setattr(delegate, "exact_start", refuse_definitely)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task["id"])
    ctx.budget_drive_root = str(tmp_path)

    result = recovery.adopt_handoff(ctx, task)

    assert result["reason"] == "pending_recovery_definitely_refused"
    assert "pending_invocation_id" not in result
    assert custody.invocation_record(tmp_path, invocation_id)["state"] == "failed_definite"
    assert recovery._read(tmp_path, task["id"])["status"] == "vetoed"
    assert task["id"] not in recovery.recoverable_task_ids(tmp_path)
    custody._CUSTODY.clear()


def test_a_pending_body_that_cannot_be_located_now_defers_the_holder(monkeypatch, tmp_path):
    """The holder's replay reads its pending invocation's stored body. While that body, held only by another task's
    pack, cannot be located (the packs catalog is held), the holder is deferred and kept for the next pass, never
    refused as a missing prompt; once the body reads, the next pass replays exactly that body."""
    import threading

    import ouroboros.tools.delegate as delegate
    from ouroboros import evidence_packs, store
    from ouroboros.tools.registry import ToolContext

    custody, recovery, task, _snapshot, invocation_id, _fingerprint, _authority = (
        _pending_handoff(tmp_path, "recovery-unlocated")
    )
    [ref] = [row["request_ref"] for row in custody._iter_rows(custody.event_log_path(tmp_path))
             if row["type"] == custody.START_REQUESTED]
    evidence_packs.write_pack(tmp_path, "other", 1, [ref])
    evidence_packs.ensure_index(tmp_path)
    (tmp_path / "observability" / "blobs" / f"{ref['sha256']}.{ref['kind']}.gz").unlink()
    starts = []
    monkeypatch.setattr(delegate, "exact_start", lambda _ctx, prompt, retry: starts.append(
        (prompt, retry["retry_of"])) or _fail("delegate_start", "subscription_window_exhausted", "spent"))
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task["id"])
    ctx.budget_drive_root = str(tmp_path)
    held, release = threading.Event(), threading.Event()

    def hold():
        with store.name_lock(tmp_path, evidence_packs.INDEX_LOCK_NAME, timeout_sec=5):
            held.set()
            release.wait(600)

    holder = threading.Thread(target=hold, name="packer-holds-index")
    holder.start()
    try:
        assert held.wait(5)
        with store.acquisition_budget(0):
            result = recovery.adopt_handoff(ctx, task)
    finally:
        release.set()
        holder.join(10)
    assert result["status"] == "recovery_required" and result["reason"] == "evidence_unavailable", result
    assert recovery._read(tmp_path, task["id"])["holders"][0]["status"] == "deferred"
    assert starts == [] and task["id"] in recovery.recoverable_task_ids(tmp_path)
    assert [row["invocation_id"] for row in custody.pending_invocations(tmp_path)] == [invocation_id]
    assert recovery.adopt_handoff(ctx, task)["reason"] == "pending_recovery_deferred"
    assert starts == [("stored canonical request", invocation_id)]
    custody._CUSTODY.clear()


def test_started_retry_race_adopts_same_durable_invocation(monkeypatch, tmp_path):
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.tools.delegate as delegate
    from ouroboros.tools.registry import ToolContext

    custody, recovery, task, snapshot, invocation_id, fingerprint, authority = (
        _pending_handoff(tmp_path, "recovery-started-race")
    )

    def expose_started_race(*_args, **_kwargs):
        assert custody.record_started(tmp_path, custody.RunCustody(
            run_id="run-started-race",
            task_id=task["id"],
            route_id="codex",
            invocation_id=invocation_id,
            selected_subagent_id=snapshot["selected_subagent_id"],
            config_fingerprint=snapshot["config_fingerprint"],
            work_order_fingerprint=fingerprint,
            authority_fingerprint=authority,
        ))
        return _fail(
            "delegate_start", "invocation_already_started",
            "that invocation already bound a run", run_id="run-started-race",
        )

    class Gateway:
        def get_run(self, run_id):
            assert run_id == "run-started-race"
            return {"id": run_id, "state": "running"}

        def close(self):
            pass

    monkeypatch.setattr(delegate, "exact_start", expose_started_race)
    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task["id"])
    ctx.budget_drive_root = str(tmp_path)

    result = recovery.adopt_handoff(ctx, task)

    assert {key: result[key] for key in ("status", "run_id", "cause")} == {
        "status": "adopted",
        "run_id": "run-started-race",
        "cause": recovery.CAUSE_WORKER_CRASH,
    }
    assert recovery._read(tmp_path, task["id"])["status"] == "adopted"
    custody._CUSTODY.clear()


def test_a_review_panels_work_never_displaces_the_actors_own_handoff_holder(
    monkeypatch, tmp_path,
):
    """Issue #1006: the handoff holder is the task's OWN delegation. An open
    review run and a pending review invocation registered under the same task id
    belong to their panel: they must neither become the holder nor veto the
    actor's exact binding by counting as a second candidate."""
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.tools.delegate as delegate
    from ouroboros import delegate_custody as dc
    from ouroboros.tools.registry import ToolContext

    task_id = "recovery-beside-review"
    dc._CUSTODY.clear()
    assert dc.record_started(tmp_path, dc.RunCustody(
        run_id="run-panel", task_id=task_id, route_id="codex",
        source="review_substrate", category="task_acceptance_review"))
    assert dc.record_start_requested(
        tmp_path, run_id="", task_id=task_id, invocation_id="inv-panel",
        idempotency_key="inv-panel", request={"prompt": "review packet"},
        route="codex", source="review_substrate.extraction")

    custody, recovery, task, snapshot, invocation_id, fingerprint, authority = (
        _pending_handoff(tmp_path, task_id)
    )

    def expose_started_race(*_args, **_kwargs):
        assert custody.record_started(tmp_path, custody.RunCustody(
            run_id="run-leaf",
            task_id=task["id"],
            route_id="codex",
            invocation_id=invocation_id,
            selected_subagent_id=snapshot["selected_subagent_id"],
            config_fingerprint=snapshot["config_fingerprint"],
            work_order_fingerprint=fingerprint,
            authority_fingerprint=authority,
        ))
        return _fail(
            "delegate_start", "invocation_already_started",
            "that invocation already bound a run", run_id="run-leaf",
        )

    class Gateway:
        def get_run(self, run_id):
            assert run_id == "run-leaf"
            return {"id": run_id, "state": "running"}

        def close(self):
            pass

    monkeypatch.setattr(delegate, "exact_start", expose_started_race)
    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task["id"])
    ctx.budget_drive_root = str(tmp_path)

    adopted = recovery.adopt_handoff(ctx, task)
    assert {key: adopted[key] for key in ("status", "run_id", "cause")} == {
        "status": "adopted",
        "run_id": "run-leaf",
        "cause": recovery.CAUSE_WORKER_CRASH,
    }
    custody._CUSTODY.clear()


def test_review_substrate_work_does_not_hold_the_actors_zero_run_fence(tmp_path):
    """A pending review invocation and an open review run leave the slot free."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_recovery import unsettled_start_ids
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.verify import _verify_and_record

    custody._CUSTODY.clear()
    (tmp_path / "repo").mkdir(parents=True, exist_ok=True)
    (tmp_path / "drive").mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(
        repo_dir=str(tmp_path / "repo"), drive_root=str(tmp_path / "drive"),
    )
    ctx.task_id = "actor-with-review"
    ctx._configured_actor_bootstrap = {
        "route_id": "session-a",
        "work_order_fingerprint": "a" * 64,
        "physical_started": False,
    }
    drive = custody.custody_root(ctx)
    assert custody.record_start_requested(
        drive,
        run_id="",
        task_id=ctx.task_id,
        invocation_id="inv-review",
        idempotency_key="inv-review",
        request={"prompt": "review packet"},
        route="codex",
        source="review_substrate.extraction",
    )
    custody.record_started(drive, custody.RunCustody(
        run_id="run-review", task_id=ctx.task_id, route_id="codex",
        source="review_substrate",
    ))
    custody._CUSTODY.clear()

    assert unsettled_start_ids(drive, ctx.task_id) == {
        "open_run_ids": [],
        "pending_invocation_ids": [],
        "undisposed_patch_run_ids": [],
    }

    written = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="incomplete",
        zero_run_basis="only read-only review runs remain open",
    )

    assert "zero_run_requires_settlement" not in written
    assert [
        row.get("contract_kind")
        for row in read_verification_receipts(drive, ctx.task_id)
    ] == ["delegation_zero_run"]
    custody._CUSTODY.clear()


def test_a_session_start_keeps_its_held_heads_by_address_and_revision_and_replays_their_exact_text(tmp_path):
    """Both directions of a start's held heads in custody. Without heads the stored envelope is the request's own
    bytes and its row names none. With heads the stored envelope names each head by address, revision and digest
    instead of its text, the text is one blob any start sending it shares, the row and the run's terminal receipt
    name the heads (topic, revision, characters) that reach the parent's outcome, and every replay reads the exact
    request back; a head whose blob is gone leaves the request unknown, never a body with a token in it."""
    import hashlib

    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_invocation import invocation_record
    from ouroboros.delegate_terminal import terminal_custody_disclosure
    from ouroboros.observability import _json_bytes, read_blob_ref
    from ouroboros.tools.control_task_results import _subtask_outcome_summary

    custody._CUSTODY.clear()
    head = "Nova's working head.\nA second line, kept whole.\n"
    revision = "a" * 64
    body = {"prompt": "Build", "instructions": "OWNER RULES\n\n## Held knowledge notes\n\n### Nova\n\n" + head + "\n\n[Host: ...]"}
    heads = [{"topic": "circle/nova", "revision": revision, "chars": len(head), "text": head}]
    facts = [{"topic": "circle/nova", "revision": revision, "chars": len(head)}]
    for invocation, held in (("inv-plain", None), ("inv-held", heads), ("inv-again", heads)):
        assert custody.record_start_requested(
            tmp_path, run_id="", task_id="nanny", invocation_id=invocation, idempotency_key=invocation, max_seconds=60,
            request=body, project_id="", project_owned=False, route="codex", **({"held_heads": held} if held else {}))
        assert custody.record_started(tmp_path, custody.RunCustody(run_id=f"run-{invocation}", task_id="nanny",
                                                                   route_id="codex", invocation_id=invocation))
        assert custody.emit(tmp_path, custody.SETTLED, {"run_id": f"run-{invocation}", "task_id": "nanny", "state": "succeeded"})
    rows = {row["invocation_id"]: row for row in custody.custody_rows(tmp_path) if row.get("type") == custody.START_REQUESTED}
    stored = {key: read_blob_ref(tmp_path, row["request_ref"], expected_kind="json") for key, row in rows.items()}
    assert rows["inv-plain"]["request_ref"]["sha256"] == hashlib.sha256(_json_bytes(body)).hexdigest()
    assert stored["inv-plain"] == body and "held_heads" not in rows["inv-plain"]
    for key in ("inv-held", "inv-again"):
        assert head not in stored[key]["instructions"] and "\x00held head circle/nova@" + revision in stored[key]["instructions"]
        assert rows[key]["held_heads"] == facts and invocation_record(tmp_path, key)["request"] == body
    assert stored["inv-held"] == stored["inv-again"]  # one blob of the head's text, shared
    assert invocation_record(tmp_path, "inv-plain")["request"] == body

    disclosure = terminal_custody_disclosure(tmp_path, "nanny")
    receipts = {row["run_id"]: row for row in disclosure["terminal_runs"]}
    assert "held_heads" not in receipts["run-inv-plain"] and receipts["run-inv-held"]["held_heads"] == facts
    outcome = json.loads(_subtask_outcome_summary({"task_id": "nanny", "delegate_terminal_reconciliation": disclosure}))
    assert {row["run_id"]: row.get("held_heads") for row in outcome["delegated_custody"]["terminal_runs"]}["run-inv-held"] == facts

    (blob,) = (tmp_path / "observability" / "blobs").glob("*.txt.gz")
    blob.unlink()
    assert invocation_record(tmp_path, "inv-held")["request"] is None
    custody._CUSTODY.clear()


def test_a_stored_session_request_always_splices_back_to_its_original_envelope(tmp_path):
    """A start's stored envelope names its held heads by reference only where putting them back gives the original
    envelope. A head whose text stands inside an earlier head's reference (``size=``), or prose shaped like a
    reference (its size past what ``int()`` reads included), stores the envelope whole and the row says so; the
    start records, and every replay reads the original request either way."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_invocation import invocation_record
    from ouroboros.delegate_pending import pending_invocations, request_body
    from ouroboros.observability import read_blob_ref

    custody._CUSTODY.clear()

    def start(*texts, prose=""):
        notes = "".join(f"### Note {number}\n\n{text}\n\n" for number, text in enumerate(texts))
        heads = [{"topic": f"note-{number}", "revision": "a" * 64, "chars": len(text), "text": text}
                 for number, text in enumerate(texts)]
        return {"prompt": "Build", "instructions": prose + "## Held knowledge notes\n\n" + notes + "[Host: ...]"}, heads

    manifest, april = "Use the manifest for the current format.\n", "Keep the April figures.\n"
    quoted = "A mark quotes \x00held head x@r sha256=" + "b" * 64 + " size={}\x00.\n\n"
    starts = {"inv-ordinary": start(manifest, april), "inv-inside": start(manifest, "size="),
              "inv-prose": start(april, prose=quoted.format("5")),
              "inv-digits": start(april, prose=quoted.format("0" * 4_300 + "5"))}  # 4,301 digits
    for invocation, (body, heads) in starts.items():
        assert custody.record_start_requested(
            tmp_path, run_id="", task_id="nanny", invocation_id=invocation, idempotency_key=invocation,
            max_seconds=60, request=body, project_id="", project_owned=False, route="codex", held_heads=heads)
    rows = {row["invocation_id"]: row for row in custody.custody_rows(tmp_path) if row.get("type") == custody.START_REQUESTED}
    stored = {key: read_blob_ref(tmp_path, row["request_ref"], expected_kind="json") for key, row in rows.items()}
    assert manifest not in stored["inv-ordinary"]["instructions"] and "held_heads_inline" not in rows["inv-ordinary"]
    inline = ("inv-inside", "inv-prose", "inv-digits")
    assert [(stored[key], rows[key]["held_heads_inline"]) for key in inline] == [(starts[key][0], True) for key in inline]
    assert {key: row["held_heads"] for key, row in rows.items()} == {key: [
        {name: head[name] for name in ("topic", "revision", "chars")} for head in heads] for key, (_, heads) in starts.items()}
    original = {key: body for key, (body, _heads) in starts.items()}
    assert {row["invocation_id"]: request_body(tmp_path, row) for row in pending_invocations(tmp_path)} == original
    assert {row["invocation_id"]: row["request"] for row in pending_invocations(tmp_path, custody.custody_rows(tmp_path))} == original
    assert {key: invocation_record(tmp_path, key)["request"] for key in starts} == original
    custody._CUSTODY.clear()


def test_only_a_start_whose_row_lists_held_heads_reads_its_envelope_with_head_references(tmp_path):
    """A mark quoting the reference delimiter, in the envelope of a start that sent no held heads (or ran before
    them), replays as written; a start whose row lists its heads still reads them back, also through an open-request
    projection the previous build cached without that list."""
    from ouroboros import delegate_custody as custody
    from ouroboros import store
    from ouroboros.delegate_invocation import invocation_record
    from ouroboros.delegate_pending import pending_invocations, request_body

    custody._CUSTODY.clear()
    example ="\x00held head x@r sha256=" + "b" * 64 + " size=5\x00"
    head = "Nova's working head.\n"
    plain = {"prompt": "Build", "instructions": "A mark quotes the delimiter " + example + " as text."}
    held = {"prompt": "Build", "instructions": "## Held knowledge notes\n\n### Nova\n\n" + head + "\n[Host: ...]"}
    heads = [{"topic": "circle/nova", "revision": "a" * 64, "chars": len(head), "text": head}]
    for invocation, body, extra in (("inv-plain", plain, {}), ("inv-held", held, {"held_heads": heads})):
        assert custody.record_start_requested(
            tmp_path, run_id="", task_id="nanny", invocation_id=invocation, idempotency_key=invocation,
            max_seconds=60, request=body, project_id="", project_owned=False, route="codex", **extra)
    with store.open_store(tmp_path, write=True) as txn:  # the held start's projection as the previous build cached it
        facts = txn.get_row("custody_open", "invocation:inv-held")
        txn.put_row("custody_open", "invocation:inv-held", {**facts, "pending": [
            {key: value for key, value in record.items() if key != "held_heads"} for record in facts["pending"]]})
    assert {row["invocation_id"]: request_body(tmp_path, row) for row in pending_invocations(tmp_path)} == {
        "inv-plain": plain, "inv-held": held}
    chain = pending_invocations(tmp_path, custody.custody_rows(tmp_path))  # a recovery sweep's own projection
    assert {row["invocation_id"]: row["request"] for row in chain} == {"inv-plain": plain, "inv-held": held}
    assert [invocation_record(tmp_path, key)["request"] for key in ("inv-plain", "inv-held")] == [plain, held]
    custody._CUSTODY.clear()
