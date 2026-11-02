"""Late retry staging must re-owe the discharged physical owner's receipt (#1573).

The custody fact, its closing receipt and the outbox row commit in ONE store transaction
(``delegate_custody_current.publish``); the addressed refresh runs after that COMMIT. Once
the store accepted a retained output B, B is either published through the physical owner's
``get_task_result`` digest selector or still named by that owner's closing receipt, whatever
fails after the COMMIT; a rolled-back transaction claims nothing.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current, delegate_terminal, obligations, store
from ouroboros import usage_accounting as usage
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.registry import ToolContext
from tests.test_delegate_retry_output import (
    TerminalGateway,
    bind_gateway,
    call,
    durable_run,
    read_whole,
    receipts,
    retry_context,
    wait_result,
)
from tests.test_delegate_retry_output import (
    isolated_retry_state as isolated_retry_state,
)
from tests.test_delegate_retry_output import (
    reader_coverage as reader_coverage,
)

pytestmark = pytest.mark.serial


def refresh(data, monkeypatch):
    """The maintenance path must not recover this receipt by folding history."""
    def no_history(*args, **kwargs):
        raise AssertionError("addressed refresh must use current custody only")

    with monkeypatch.context() as patch:
        patch.setattr(custody, "replay", no_history)
        patch.setattr(custody, "custody_rows", no_history)
        return delegate_terminal.backfill_terminal_reconciliations(data)


def discharged_retry(tmp_path, monkeypatch, state="succeeded", *, acknowledged=True,
                     stored_usd=1.75, ledger_usd=1.75):
    ctx = retry_context(tmp_path, monkeypatch, split=True)
    data = custody.custody_root(ctx)
    held = durable_run(ctx, state=state)
    # Multiple historical receipts must survive refreshing only the late run.
    for rid in ("old-failed", "old-cancelled"):
        durable_run(ctx, run_id=rid, state=rid.removeprefix("old-"))
    if ledger_usd is not None:
        # The task's money is the ledger's: the held run's session receipt. The older runs stay
        # unpriced, so the known subtotal is partial (cost_final False), never a result-only literal.
        usage.record_subscription_session(held.run_id, drive_root=data, route="test-route", model="test-model",
                                          task_id=held.task_id, root_task_id=held.task_id, spend_usd=ledger_usd)
    write_task_result(data, held.task_id, "failed", reason_code="idle_timeout",
                      accounted_upper_bound_usd=stored_usd, cost_final=False,
                      delegated_runs_started=3, delegated_runs_settled=0,
                      outcome_axes={"execution": {"status": "infra_failed"}})
    if acknowledged:
        assert refresh(data, monkeypatch) == [held.task_id]
        assert obligations.members(data, "custody_open") == {}
        assert obligations.members(data, "delegated_runs") == {}
    else:  # R0: the settling transaction's receipt is still owed, no refresh has run yet
        assert set(obligations.members(data, "delegated_runs")[held.task_id]["closed_runs"]) == {
            held.run_id, "old-failed", "old-cancelled"}
    before = load_task_result(data, held.task_id)
    if acknowledged:
        assert len(before["delegate_terminal_reconciliation"]["terminal_runs"]) == 3
    bind_gateway(monkeypatch, TerminalGateway(state=state, output="late retained work " * 8000))
    return ctx, held, before


def source_page(ctx, task_id, artifact):
    result = call(ctx, "get_task_result", task_id=task_id,
                  delegated_output_sha256=artifact["sha256"],
                  source_start_char=0, source_end_char=80)
    assert result.status == "ok", result.text
    return json.loads(result.text)["delegated_output_source"]


def owed_source(data, task_id, run_id):
    """What the physical owner's closing receipt names, if a receipt is owed at all."""
    duty = obligations.members(data, "delegated_runs").get(task_id, {}).get("closed_runs", {}).get(run_id)
    return (duty or {}).get("output_source") or {}


def abort_spill_transaction(patch):
    """Fail the custody transaction of the OUTPUT_SPILLED event itself: rollback, nothing committed."""
    real = store.Txn.outbox_put

    def outbox_put(self, event):
        if event.get("type") == custody.OUTPUT_SPILLED:
            raise sqlite3.OperationalError("injected: the spill's custody transaction aborted")
        return real(self, event)

    patch.setattr(store.Txn, "outbox_put", outbox_put)


@pytest.mark.parametrize("state", ["succeeded", "failed", "cancelled"])
def test_late_retry_wait_republishes_discharged_owner_receipt(tmp_path, monkeypatch, state):
    ctx, held, before = discharged_retry(tmp_path, monkeypatch, state)
    data = custody.custody_root(ctx)
    settled = [row for row in custody.custody_rows(data) if row["type"] == custody.SETTLED]
    original_refresh = delegate_terminal.refresh_terminal_reconciliation
    addressed = []

    def observed_refresh(root, task_id, **kwargs):
        addressed.append((task_id, kwargs.get("trigger")))
        with monkeypatch.context() as patch:
            patch.setattr(custody, "replay", lambda *a, **k: pytest.fail("history replay"))
            return original_refresh(root, task_id, **kwargs)

    monkeypatch.setattr(delegate_terminal, "refresh_terminal_reconciliation", observed_refresh)
    artifact = wait_result(ctx, held.run_id)["output_delivery"]["artifact"]
    assert artifact["source"]["task_id"] == ctx.task_id != held.task_id
    # A detached physical-task reader survives removal of the staging child's drive.
    reader = ToolContext(repo_dir=ctx.repo_dir, drive_root=data, task_id="physical-reader")
    shutil.rmtree(ctx.drive_root)
    custody._CUSTODY.clear()
    page = source_page(reader, held.task_id, artifact)
    assert page["status"] == "available", page
    assert page["text"] == Path(artifact["abs_path"]).read_text(encoding="utf-8")[:80]
    assert page["staging_task_id"] == ctx.task_id
    assert addressed == [(held.task_id, "output_spilled")]
    after = load_task_result(data, held.task_id)
    for key in ("status", "reason_code", "outcome_axes", "accounted_upper_bound_usd",
                "cost_final", "delegated_runs_started", "delegated_runs_settled"):
        assert after[key] == before[key]
    old = {row["run_id"]: row for row in before["delegate_terminal_reconciliation"]["terminal_runs"]}
    new = {row["run_id"]: row for row in after["delegate_terminal_reconciliation"]["terminal_runs"]}
    assert set(new) == set(old)
    assert new["old-failed"] == old["old-failed"]
    assert new["old-cancelled"] == old["old-cancelled"]
    assert {k: v for k, v in new[held.run_id].items()
            if k not in {"staged_output", "terminal_observation"}} == old[held.run_id]
    assert new[held.run_id]["staged_output"]["sha256"] == artifact["sha256"]
    # The late wait also retains the exact engine observation on the physical
    # owner's canonical drive; both published sources survive the retry drive.
    observed = new[held.run_id]["terminal_observation"]
    source = delegate_custody_current.run(data, held.run_id).terminal_source
    assert observed == {
        **{key: value for key, value in source.items() if key != "read"},
        "complete": True, "content_kind": "terminal_observation",
        "source_read": {"tool": "get_task_result", "arguments": {
            "task_id": held.task_id, "delegated_output_sha256": source["sha256"]}},
    }
    observation_page = source_page(reader, held.task_id, observed)
    assert observation_page["status"] == "available"
    assert observation_page["content_kind"] == "terminal_observation"
    assert observation_page["staging_task_id"] == held.task_id
    assert observation_page["source_complete"] is True
    assert [row for row in custody.custody_rows(data) if row["type"] == custody.SETTLED] == settled
    assert not receipts(ctx), "digest publication/read grants no D7 receipt"
    assert obligations.members(data, "custody_open") == {}
    assert obligations.members(data, "delegated_runs") == {}
    assert refresh(data, monkeypatch) == []
    # Only the authorized retry's complete file read earns the existing D7 receipt.
    read_whole(ctx, artifact)
    assert [(row["task_id"], row["reader_task_id"]) for row in receipts(ctx)] == [
        (held.task_id, ctx.task_id)]


def test_rejected_result_write_keeps_late_receipt_owed(tmp_path, monkeypatch):
    from ouroboros import task_results

    ctx, held, before = discharged_retry(tmp_path, monkeypatch)
    data = custody.custody_root(ctx)
    write = task_results.write_task_result

    def reject(root, task_id, *args, **kwargs):
        if task_id == held.task_id and "delegate_terminal_reconciliation" in kwargs:
            return load_task_result(root, task_id)
        return write(root, task_id, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(task_results, "write_task_result", reject)
        artifact = wait_result(ctx, held.run_id)["output_delivery"]["artifact"]
        owed = obligations.members(data, "delegated_runs")
        assert set(owed) == {held.task_id}
        assert owed[held.task_id]["closed_runs"][held.run_id]["output_source"] == artifact["source"]
        assert refresh(data, patch) == []
    assert obligations.members(data, "delegated_runs") == owed
    assert load_task_result(data, held.task_id) == before
    assert source_page(ctx, held.task_id, artifact)["reason"] == "source_not_published"
    assert refresh(data, monkeypatch) == [held.task_id]
    assert source_page(ctx, held.task_id, artifact)["status"] == "available"
    assert obligations.members(data, "delegated_runs") == {}


@pytest.mark.parametrize("interleave_refresh", [False, True], ids=["rolled-back", "stale-ack-first"])
def test_spill_custody_must_commit_before_publication(tmp_path, monkeypatch, interleave_refresh):
    """Rolled back: the transaction that would carry B commits nothing and claims nothing;
    an ordinary later wait lands the same retained bytes. Stale ACK first: a refresh that
    acknowledged the old receipt just before B's transaction cannot leave B unowed."""
    ctx, held, before = discharged_retry(tmp_path, monkeypatch, acknowledged=False)  # R0 still owed
    data = custody.custody_root(ctx)
    publish, seen = delegate_custody_current.publish, []

    def publish_spill(root, event):
        if event["type"] == custody.OUTPUT_SPILLED:
            seen.append(event)
            if interleave_refresh:  # the sweep publishes and acknowledges R0 just before B's transaction
                assert refresh(data, monkeypatch) == [held.task_id]
                assert obligations.members(data, "delegated_runs") == {}
        return publish(root, event)

    with monkeypatch.context() as patch:
        patch.setattr(delegate_custody_current, "publish", publish_spill)
        if not interleave_refresh:
            abort_spill_transaction(patch)
        artifact = wait_result(ctx, held.run_id)["output_delivery"]["artifact"]
    assert len(seen) == 1
    if not interleave_refresh:
        assert custody.replay(data)[held.run_id].output_source == {}
        assert owed_source(data, held.task_id, held.run_id) == {}, "a rolled-back spill owes nothing new"
        assert not [row for row in custody.custody_rows(data) if row["type"] == custody.OUTPUT_SPILLED]
        assert refresh(data, monkeypatch) == [held.task_id]  # the settling transaction's R0 receipt
        assert source_page(ctx, held.task_id, artifact)["reason"] == "source_not_published"
        assert not receipts(ctx)
        # A later ordinary wait can land the same retained bytes.
        again = wait_result(ctx, held.run_id)["output_delivery"]["artifact"]
        assert again["sha256"] == artifact["sha256"]
    assert refresh(data, monkeypatch) == []
    assert source_page(ctx, held.task_id, artifact)["status"] == "available"
    assert obligations.members(data, "delegated_runs") == {}


def test_running_owner_keeps_receipt_until_terminal_without_lifecycle_change(tmp_path, monkeypatch):
    ctx = retry_context(tmp_path, monkeypatch, chain=("running-owner",), split=True)
    held = durable_run(ctx, owner=ctx.task_id)
    data = custody.custody_root(ctx)
    before = load_task_result(data, ctx.task_id)
    bind_gateway(monkeypatch, TerminalGateway(output="late work " * 16000))
    artifact = wait_result(ctx, held.run_id)["output_delivery"]["artifact"]
    after_wait = load_task_result(data, ctx.task_id)
    assert after_wait["status"] == before["status"] == "running"
    assert refresh(data, monkeypatch) == []
    assert load_task_result(data, ctx.task_id) == after_wait
    assert held.run_id in obligations.members(data, "delegated_runs")[held.task_id]["closed_runs"]
    assert source_page(ctx, held.task_id, artifact)["reason"] == "source_not_published"
    write_task_result(data, ctx.task_id, "completed")
    assert refresh(data, monkeypatch) == [ctx.task_id]
    assert source_page(ctx, held.task_id, artifact)["status"] == "available"
    assert obligations.members(data, "delegated_runs") == {}


def test_post_commit_refresh_error_preserves_spill_success(tmp_path, monkeypatch, caplog):
    ctx, held, before = discharged_retry(tmp_path, monkeypatch)
    data = custody.custody_root(ctx)
    emit, outcomes = custody.emit, []

    def observe_emit(root, kind, payload, **kwargs):
        landed = emit(root, kind, payload, **kwargs)
        if kind == custody.OUTPUT_SPILLED:
            outcomes.append(landed)
        return landed

    def broken_refresh(*args, **kwargs):
        raise ValueError("injected post-commit bookkeeping failure")

    with monkeypatch.context() as patch:
        patch.setattr(custody, "emit", observe_emit)
        patch.setattr(delegate_terminal, "refresh_terminal_reconciliation", broken_refresh)
        artifact = wait_result(ctx, held.run_id)["output_delivery"]["artifact"]
    spills = [row for row in custody.custody_rows(data) if row["type"] == custody.OUTPUT_SPILLED]
    assert len(spills) == 1 and spills[0]["sha256"] == artifact["sha256"]
    assert outcomes == [True], "post-commit bookkeeping cannot turn a committed custody fact into failure"
    resolved = custody.replay(data)[held.run_id]
    assert resolved.output_source == artifact["source"]
    assert resolved.output_sha == artifact["sha256"] and resolved.output_complete
    assert any("post-commit bookkeeping" in record.message and record.exc_info
               for record in caplog.records)
    assert load_task_result(data, held.task_id) == before
    assert owed_source(data, held.task_id, held.run_id) == artifact["source"]
    assert source_page(ctx, held.task_id, artifact)["reason"] == "source_not_published"
    assert refresh(data, monkeypatch) == [held.task_id]
    assert source_page(ctx, held.task_id, artifact)["status"] == "available"
    assert obligations.members(data, "delegated_runs") == {}


def test_failed_addressed_audit_keeps_unpublished_spill_owed(tmp_path, monkeypatch):
    ctx, held, before = discharged_retry(tmp_path, monkeypatch)
    data = custody.custody_root(ctx)

    def failed_patch_audit(*args, **kwargs):
        raise ValueError("injected addressed audit failure")

    with monkeypatch.context() as patch:
        patch.setattr(custody, "undisposed_patches", failed_patch_audit)
        artifact = wait_result(ctx, held.run_id)["output_delivery"]["artifact"]
        failed = load_task_result(data, held.task_id)
        assert failed["delegate_terminal_reconciliation"]["audit_status"] == "failed"
        assert source_page(ctx, held.task_id, artifact)["reason"] == "source_not_published"
        owed = obligations.members(data, "delegated_runs")
        assert held.run_id in owed[held.task_id]["closed_runs"], "failed audit discharged unpublished output"
        assert owed[held.task_id]["closed_runs"][held.run_id]["output_source"] == artifact["source"]
        # The failed-disclosure no-op path must preserve the same duty too.
        assert refresh(data, patch) == []
        assert obligations.members(data, "delegated_runs") == owed
    assert refresh(data, monkeypatch) == [held.task_id]
    assert source_page(ctx, held.task_id, artifact)["status"] == "available"
    assert obligations.members(data, "delegated_runs") == {}
    after = load_task_result(data, held.task_id)
    for key in ("status", "reason_code", "outcome_axes", "accounted_upper_bound_usd",
                "cost_final", "delegated_runs_started", "delegated_runs_settled"):
        assert after[key] == before[key]


@pytest.mark.parametrize("continued", [False, True], ids=["same-digest", "superseded-payload"])
def test_failed_spill_acknowledgement_names_only_the_returned_digest(tmp_path, monkeypatch, continued):
    """A read receipt cannot transfer to B when its spill transaction rolls back."""
    ctx, held, _before = discharged_retry(tmp_path, monkeypatch)
    data = custody.custody_root(ctx)
    held = durable_run(ctx, owner=held.task_id, run_id=held.run_id,
                       snapshot_id="retained-snapshot", capture_id="original-capture")
    first = wait_result(ctx, held.run_id)["output_delivery"]["artifact"]
    read_whole(ctx, first)
    assert wait_result(ctx, held.run_id)["output_delivery"]["consumed"]
    if continued:
        durable_run(ctx, owner=held.task_id, run_id="same-snapshot-continuation", state=None,
                    snapshot_id="retained-snapshot", capture_id="continuation-capture",
                    continuation_of=held.run_id)
        assert custody.replay(data)[held.run_id].superseded_by == "same-snapshot-continuation"

    with monkeypatch.context() as patch:
        abort_spill_transaction(patch)
        delivery = wait_result(ctx, held.run_id)["output_delivery"]
    returned = delivery["artifact"]
    assert (returned["sha256"] != first["sha256"]) is continued
    assert delivery["consumed"] is (not continued), "acknowledged A cannot certify returned B"
    assert custody.replay(data)[held.run_id].output_sha == first["sha256"]
    assert [row["sha256"] for row in receipts(ctx)] == [first["sha256"]]


# --- NEW-B: a retained output the store accepted is published or durably owed -------------
# The b87 red (OUTPUT_SPILLED appended, closing bookkeeping lost, R0 acknowledged, digest
# ``source_not_published`` with no duty) translated to the SQL commit boundaries: the
# receipt was acknowledged before B's COMMIT; the chain append after COMMIT fails; the
# process dies right after COMMIT; a stale R0 audit's write and ACK span B's COMMIT.

NEW_B_FAULTS = ["acked-before-commit", "flush-fails-after-commit", "dies-after-commit",
                "stale-audit-spans-commit"]


@pytest.mark.parametrize("fault", NEW_B_FAULTS)
def test_a_committed_late_output_is_published_or_durably_owed(tmp_path, monkeypatch, fault):
    from ouroboros import task_results

    ctx, held, before = discharged_retry(tmp_path, monkeypatch,
                                         acknowledged=fault != "stale-audit-spans-commit")
    data = custody.custody_root(ctx)
    gateway = TerminalGateway(output="late retained work " * 8000)
    bind_gateway(monkeypatch, gateway)
    started = [row for row in custody.custody_rows(data) if row["type"] == custody.STARTED]
    late = []
    with monkeypatch.context() as patch:
        if fault == "flush-fails-after-commit":
            patch.setattr(delegate_custody_current, "_append_durable", lambda *_a, **_k: False)
        elif fault == "dies-after-commit":
            # Nothing after the COMMIT runs: no chain append, no refresh.
            patch.setattr(delegate_custody_current, "flush_outbox", lambda _root: 0)
            patch.setattr(delegate_custody_current, "_refresh_owed_receipt", lambda *_a: None)
        if fault == "stale-audit-spans-commit":
            # The sweep audits the owed R0 receipt; B's wait then commits and refreshes in its
            # own execution context before the sweep's write and acknowledgement land.
            write, armed = task_results.write_task_result, [True]

            def late_wait_first(root, task_id, *args, **kwargs):
                if armed and task_id == held.task_id and "delegate_terminal_reconciliation" in kwargs:
                    armed.clear()
                    late.append(wait_result(ctx, held.run_id)["output_delivery"]["artifact"])
                return write(root, task_id, *args, **kwargs)

            patch.setattr(task_results, "write_task_result", late_wait_first)
            delegate_terminal.refresh_recently_settled_terminals(data)
            artifact = late[0]
        else:
            artifact = wait_result(ctx, held.run_id)["output_delivery"]["artifact"]
    spilled = [row for row in custody.custody_rows(data)
               if row["type"] == custody.OUTPUT_SPILLED and row["sha256"] == artifact["sha256"]]
    committed = custody.replay(data)[held.run_id]
    assert committed.output_source == artifact["source"], "the store accepted B"
    first = source_page(ctx, held.task_id, artifact)
    duty = owed_source(data, held.task_id, held.run_id)
    print("NEW_B_FACTS " + json.dumps({"fault": fault, "first": {k: first.get(k) for k in ("status", "reason")},
                                       "duty_names_b": duty.get("sha256") == artifact["sha256"],
                                       "chain_rows": len(spilled),
                                       "outbox": delegate_custody_current.outbox_depth(data)}))
    assert first["status"] == "available" or duty == artifact["source"], "new B neither published nor durably owed"
    assert not receipts(ctx), "neither publication nor duty acknowledges"
    # Ordinary recovery owns the rest: the sweep and the next publication, no wait, no new run.
    custody._CUSTODY.clear()
    delegate_terminal.refresh_recently_settled_terminals(data)
    delegate_custody_current.flush_outbox(data)
    recovered = source_page(ctx, held.task_id, artifact)
    assert recovered["status"] == "available" and recovered["sha256"] == artifact["sha256"]
    assert obligations.members(data, "delegated_runs") == {}
    assert delegate_custody_current.outbox_depth(data) == 0
    assert len([row for row in custody.custody_rows(data) if row["type"] == custody.OUTPUT_SPILLED
                and row["sha256"] == artifact["sha256"]]) == 1, "the parked event reached the chain once"
    assert [row for row in custody.custody_rows(data) if row["type"] == custody.STARTED] == started
    after = load_task_result(data, held.task_id)
    for key in ("status", "reason_code", "outcome_axes", "accounted_upper_bound_usd",
                "cost_final", "delegated_runs_started", "delegated_runs_settled"):
        assert after[key] == before[key]
    assert (after["accounted_upper_bound_usd"], after["cost_final"]) == (1.75, False), "the ledger's partial subtotal"
    receipts_by_run = {row["run_id"]: row for row in after["delegate_terminal_reconciliation"]["terminal_runs"]}
    assert set(receipts_by_run) == {held.run_id, "old-failed", "old-cancelled"}
    assert after["delegated_runs_unreconciled"] == []
    # Repeated ordinary waits are idempotent: same digest, same published row, nothing owed.
    published = after
    again = wait_result(ctx, held.run_id)["output_delivery"]["artifact"]
    assert again["sha256"] == artifact["sha256"]
    assert load_task_result(data, held.task_id) == published
    assert obligations.members(data, "delegated_runs") == {}


@pytest.mark.parametrize("money", ["unbacked", "ledger-unavailable", "corrected"])
def test_the_owed_refresh_publishes_only_ledger_money(tmp_path, monkeypatch, money):
    """The amount a custody refresh publishes is the ledger's, whatever the result stored: a figure
    with no ledger money behind it becomes unknown (never $0), an unreadable ledger rewrites no
    money and keeps the debt, and a stored figure the ledger contradicts is corrected."""
    from ouroboros import usage_store

    ctx, held, before = discharged_retry(tmp_path, monkeypatch, acknowledged=False,
                                         stored_usd=2.0 if money == "corrected" else 1.75,
                                         ledger_usd=None if money == "unbacked" else 1.75)
    data = custody.custody_root(ctx)
    keys = ("accounted_upper_bound_usd", "cost_final", "cost_accounting_status", "cost_custody")
    with monkeypatch.context() as patch:
        if money == "ledger-unavailable":
            def busy(*_args, **_kwargs):
                raise usage_store.UsageLockUnavailable("injected: usage store busy", reason="contention")

            patch.setattr(usage_store, "read", busy)
        delegate_terminal.refresh_recently_settled_terminals(data)
    after = load_task_result(data, held.task_id)
    owed = obligations.members(data, "delegated_runs").get(held.task_id, {})
    if money == "ledger-unavailable":
        assert {key: after.get(key) for key in keys} == {key: before.get(key) for key in keys}
        assert owed.get("cost_revision"), "an unread ledger acknowledges no cost debt"
        delegate_terminal.refresh_recently_settled_terminals(data)
        after = load_task_result(data, held.task_id)
    assert "cost_revision" not in obligations.members(data, "delegated_runs").get(held.task_id, {})
    expected = None if money == "unbacked" else 1.75
    assert after["accounted_upper_bound_usd"] == expected and after["cost_final"] is False
    assert after["cost_custody"]["unrecorded_run_ids"] == (
        sorted([held.run_id, "old-cancelled", "old-failed"]) if money == "unbacked" else ["old-cancelled", "old-failed"])
