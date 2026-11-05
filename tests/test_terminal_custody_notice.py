"""Final delivery and continuation show current custody beside unchanged prose."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody, delegate_terminal
from ouroboros.agent_startup_checks import task_result_authority_projection
from ouroboros.outcomes import public_task_result
from ouroboros.task_finalization import prepare_terminal_send_event, provider_terminal_body, terminal_host_notice_text
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.join_ledger import _child_result_sha256


def _cancelled_with_patch(tmp_path):
    text = "The last observed delegated run was still working; its outcome is unconfirmed."
    write_task_result(tmp_path, "root", "completed", result=text,
                      reason_code="budget_exhausted", delegated_runs_unreconciled=["run-one"],
                      continuation_narrative={"text": text})
    assert custody.emit(tmp_path, custody.STARTED, {
        "run_id": "run-one", "task_id": "root", "route": "fixture",
        "model": "fixture-model", "profile_id": "fixture-profile",
        "selected_subagent_id": "fixture-actor",
        "snapshot_id": "snapshot-one", "shape": {},
    })
    assert custody.emit(tmp_path, custody.SETTLED, {
        "run_id": "run-one", "task_id": "root", "route": "fixture", "state": "cancelled",
        "cost_usd": 0.0, "cost_final": True, "spend_disclosed": True,
    })
    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, "root")
    return text, load_task_result(tmp_path, "root")


def test_cleanup_receipt_is_carried_before_final_delivery_without_rewriting_answer(tmp_path):
    text, stored = _cancelled_with_patch(tmp_path)
    audit = stored["delegate_terminal_reconciliation"]
    assert audit["open_run_ids"] == audit["pending_invocation_ids"] == []
    assert audit["undisposed_patch_run_ids"] == ["run-one"]
    assert audit["terminal_runs"] == [{
        "run_id": "run-one", "state": "cancelled", "model": "fixture-model",
        "profile_id": "fixture-profile", "selected_subagent_id": "fixture-actor",
    }]
    before = copy.deepcopy(stored)
    usage = {"terminal_origin": "model_final", "terminal_host_notice": "Budget stop retained."}
    event = prepare_terminal_send_event(
        tmp_path, {"id": "root", "chat_id": 1}, text, usage,
        {"type": "send_message", "task_id": "root", "chat_id": 1, "text": text},
        presence=False,
    )
    assert event["text"] == text
    # One voice: the base host notice stays a field of the RESULT and never rides
    # the send event; custody travels on its own field (#1006). Single-body
    # transports still read the stored join below.
    assert "terminal_host_notice" not in event
    # The leaf's own model rides the replayed row, so the nanny's terminal is
    # not read as a verdict about the role the host played (I9).
    # Every reader sees only the categories holding something (#1403).
    custody_text = ("Confirmed delegated terminal receipts: run-one: cancelled on fixture-model.\n"
                    "Pending patch decisions: run-one.")
    assert event["terminal_custody_notice"] == custody_text
    joined = public_task_result({**stored, **usage})["terminal_host_notice"]
    assert joined == "Budget stop retained.\n\n" + custody_text
    assert load_task_result(tmp_path, "root") == before


def test_late_settlement_and_disposition_refresh_inspection_and_continuation_only(tmp_path):
    text, stored = _cancelled_with_patch(tmp_path)
    first_hash = _child_result_sha256(stored)
    public = public_task_result(stored)
    authority = task_result_authority_projection(stored, drive_root=tmp_path)
    assert "run-one: cancelled" in public["terminal_host_notice"]
    assert authority["terminal_host_notice"] == public["terminal_host_notice"]
    assert authority["result"] == public["result"] == text
    assert authority["continuation_narrative"]["text"] == text
    assert public["reason_code"] == "budget_exhausted"
    assert custody.emit(tmp_path, custody.PATCH_DISPOSED, {
        "run_id": "run-one", "task_id": "root", "disposition": "rejected",
    })
    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, "root")
    refreshed = load_task_result(tmp_path, "root")
    notice = terminal_host_notice_text(refreshed)
    assert "run-one: cancelled" in notice
    assert "Pending patch decisions" not in notice  # the decided patch leaves the category empty
    assert refreshed["result"] == text
    assert refreshed["continuation_narrative"]["text"] == text
    assert refreshed["reason_code"] == "budget_exhausted"
    assert _child_result_sha256(refreshed) != first_hash
    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, "root") is False


def test_absent_execution_is_not_a_cancelled_receipt_and_unreadable_audit_stays_unknown():
    row = {"delegate_terminal_reconciliation": {
        "audit_status": "ok", "open_run_ids": [], "pending_invocation_ids": [],
        "undisposed_patch_run_ids": ["run-unknown"], "terminal_runs": [],
    }}
    notice = terminal_host_notice_text(row)
    assert "cancelled" not in notice
    assert "Pending patch decisions: run-unknown" in notice
    row["delegate_terminal_reconciliation"]["audit_status"] = "failed"
    notice = terminal_host_notice_text(row)
    assert "could not be verified" in notice
    assert "none recorded" not in notice


def test_custody_notice_is_bounded_and_does_not_duplicate_on_public_projection():
    row = {"task_id": "root", "result": "answer", "terminal_host_notice": "Other host fact.",
           "delegate_terminal_reconciliation": {
               "audit_status": "ok", "open_run_ids": [], "pending_invocation_ids": [],
               "undisposed_patch_run_ids": [f"run-{i}" for i in range(12)],
               "terminal_runs": [{"run_id": f"run-{i}", "state": "cancelled",
                                  "model": "leaf-model"} for i in range(12)],
           }}
    public = public_task_result(row)
    assert "+2 more in task details" in public["terminal_host_notice"]
    assert "run-11: cancelled" not in public["terminal_host_notice"]
    assert public_task_result(public)["terminal_host_notice"] == public["terminal_host_notice"]
    assert row["terminal_host_notice"] == "Other host fact."


def test_successful_settled_runs_add_no_unrelated_terminal_notice():
    row = {"delegate_terminal_reconciliation": {
        "audit_status": "ok", "terminal_runs": [{"run_id": "run-ok", "state": "succeeded"}],
        "open_run_ids": [], "pending_invocation_ids": [], "undisposed_patch_run_ids": [],
    }}
    assert terminal_host_notice_text(row) == ""


def test_a_tasks_own_reviewers_never_become_its_unreconciled_custody(tmp_path):
    """Issue #1006: a task whose only open rows are its acceptance-review slots
    audits CLEAN. Its terminal was «Done with warnings» over a run the review
    panel owns, so the audit must not name it, and the notice stays empty."""
    write_task_result(tmp_path, "root", "completed", result="verdict",
                      delegated_runs_unreconciled=["run-review"])
    assert custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-review", task_id="root", route_id="codex", model="review-model",
        source="review_substrate", category="task_acceptance_review",
        review_slot_id="triad_286lhb"))
    assert custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-review-done", task_id="root", route_id="codex",
        source="review_substrate:task_acceptance"))
    assert custody.emit(tmp_path, custody.SETTLED, {
        "run_id": "run-review-done", "task_id": "root", "route": "codex",
        "state": "failed", "source": "review_substrate"})
    assert custody.record_start_requested(
        tmp_path, run_id="", task_id="root", invocation_id="inv-review",
        idempotency_key="inv-review", request={"prompt": "packet"},
        route="codex", source="review_substrate.extraction")
    custody._CUSTODY.clear()

    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, "root")
    audit = load_task_result(tmp_path, "root")["delegate_terminal_reconciliation"]
    assert audit["audit_status"] == "ok"
    assert audit["open_run_ids"] == []
    assert audit["pending_invocation_ids"] == []
    assert audit["terminal_runs"] == []
    assert audit["unreconciled"] == []
    assert terminal_host_notice_text(load_task_result(tmp_path, "root")) == ""
    custody._CUSTODY.clear()


def test_the_tasks_own_open_run_still_shows_beside_its_reviewers(tmp_path):
    """The regression pin for the exclusion above: only the REVIEW rows leave."""
    write_task_result(tmp_path, "root", "completed", result="verdict",
                      delegated_runs_unreconciled=["stale"])
    assert custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-review", task_id="root", route_id="codex",
        source="review_substrate"))
    assert custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-leaf", task_id="root", route_id="codex", model="leaf-model"))
    custody._CUSTODY.clear()

    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, "root")
    audit = load_task_result(tmp_path, "root")["delegate_terminal_reconciliation"]
    assert audit["open_run_ids"] == audit["unreconciled"] == ["run-leaf"]
    assert "Open delegated execution: run-leaf" in terminal_host_notice_text(
        load_task_result(tmp_path, "root"))
    custody._CUSTODY.clear()


def _audit(**facts):
    return {"task_id": "root", "result": "answer", "status": "completed",
            "delegate_terminal_reconciliation": {"audit_status": "ok", "open_run_ids": [],
                                                 "pending_invocation_ids": [], "undisposed_patch_run_ids": [],
                                                 "terminal_runs": [], **facts}}


# The pre-#1403 rendering of ``_FAILED``, and the disposition identity computed over it on b6772cce.
_FAILED_RUN = {"run_id": "run-7c1", "state": "failed", "model": "leaf-m"}
_FAILED_NOTICE = "Confirmed delegated terminal receipts: run-7c1: failed on leaf-m."
_FAILED_FULL = (_FAILED_NOTICE + "\n"
                "Open delegated execution: none recorded by the current audit.\n"
                "Pending delegated invocations: none recorded by the current audit.\n"
                "Pending patch decisions: none recorded by the current audit.")
_FAILED_HASH = "ea29f0d212c55c6df1fcb7d1db93ed6040e307c70712560b1eef0f9c5991786f"


def test_empty_categories_leave_every_notice_and_the_child_hash_keeps_its_bytes():
    """#1403: no reader sees an empty audit category; non-success, nonempty debt and an
    unavailable audit stay. The child-result hash alone still reads the old text, so its
    values are pinned to their pre-#1403 bytes (computed on b6772cce)."""
    from ouroboros.task_finalization import terminal_custody_notice_text, terminal_notice_text

    failed = _audit(terminal_runs=[_FAILED_RUN])
    assert terminal_custody_notice_text(failed) == terminal_host_notice_text(failed) == _FAILED_NOTICE
    assert terminal_notice_text(failed) == public_task_result(failed)["terminal_host_notice"] == _FAILED_NOTICE
    assert terminal_host_notice_text(failed, include_empty=True) == _FAILED_FULL
    assert terminal_notice_text(failed, include_empty=True) == _FAILED_FULL
    assert _child_result_sha256(failed) == _FAILED_HASH

    debt = _audit(open_run_ids=["run-a"], pending_invocation_ids=["inv-b"])
    both = "Open delegated execution: run-a.\nPending delegated invocations: inv-b."
    assert terminal_custody_notice_text(debt) == terminal_host_notice_text(debt) == both
    assert terminal_host_notice_text(debt, include_empty=True) == (
        both + "\nPending patch decisions: none recorded by the current audit.")

    patch = _audit(undisposed_patch_run_ids=["run-one"], terminal_runs=[{"run_id": "run-one", "state": "cancelled"}])
    patch["terminal_host_notice"] = "Budget stop retained."
    assert terminal_host_notice_text(patch) == (
        "Budget stop retained.\n\nConfirmed delegated terminal receipts: run-one: cancelled.\n"
        "Pending patch decisions: run-one.")
    assert _child_result_sha256(patch) == "ddfff4f813d13a518cdeafb2be9ffabec50e09f28c768cc2784d7270f6144f34"

    unknown = {"task_id": "root", "result": "answer", "status": "completed",
               "delegate_terminal_reconciliation": {"audit_status": "failed"}}
    unavailable = "Current delegated execution could not be verified; custody remains unresolved in task details."
    assert terminal_custody_notice_text(unknown) == terminal_host_notice_text(unknown) == unavailable
    assert _child_result_sha256(unknown) == "dd261cdf1be1f0fe34839dc19abafe911b37989e7bbd006a59dec16608b1574a"
    assert terminal_custody_notice_text(_audit(terminal_runs=[{"run_id": "ok", "state": "succeeded"}])) == ""


def _stored_failed_child(tmp_path, **fields):
    audit = {**_audit(terminal_runs=[_FAILED_RUN])["delegate_terminal_reconciliation"]}
    write_task_result(tmp_path, "child", "completed", result="Answer.", parent_task_id="parent1",
                      terminal_origin="model_final", terminal_host_notice="Budget stop retained.",
                      delegate_terminal_reconciliation=audit, **fields)
    return load_task_result(tmp_path, "child")


def test_every_model_api_and_system_reader_shows_the_failed_run_without_empty_categories(tmp_path):
    from ouroboros.task_finalization import build_sealed_final_package, sealed_final_prompt_section
    from ouroboros.task_status import format_subagent_absorption_message
    from ouroboros.tools.control_task_results import _compact_child_projection, _get_task_result
    from tests.test_child_result_disposition import _parent_ctx

    stored = _stored_failed_child(tmp_path)
    expected = "Budget stop retained.\n\n" + _FAILED_NOTICE
    parent = _parent_ctx(tmp_path)
    rendered = {
        "public": public_task_result(stored)["terminal_host_notice"],
        "authority": task_result_authority_projection(stored, drive_root=tmp_path)["terminal_host_notice"],
        "compact": _compact_child_projection("child", stored, None)["terminal_host_notice"],
        "answer_view": json.loads(_get_task_result(parent, "child", view="answer"))["nonclean_terminal_host_notice"],
        "sealed": build_sealed_final_package(stored, "Answer.")["terminal_host_notice"],
    }
    assert rendered == dict.fromkeys(rendered, expected)
    bodies = {
        "full": _get_task_result(parent, "child"),
        "absorption": format_subagent_absorption_message([stored], parent_task_id="parent1"),
        "synthesis": sealed_final_prompt_section(build_sealed_final_package(stored, "Answer.")),
    }
    for reader, body in bodies.items():
        assert expected in body and "none recorded" not in body, reader
    # The identity these readers publish beside the text is the one the old text produced.
    assert json.loads(_get_task_result(parent, "child", view="answer"))["child_result_sha256"] == (
        _child_result_sha256(stored))


@pytest.mark.parametrize("jsonl", [False, True])
def test_cli_prints_the_api_notice_once_without_empty_categories(tmp_path, monkeypatch, capsys, jsonl):
    from ouroboros import cli
    from ouroboros.gateway.tasks import api_task_get

    _stored_failed_child(tmp_path)
    response = asyncio.run(api_task_get(SimpleNamespace(
        path_params={"task_id": "child"}, app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)),
    )))
    assert response.status_code == 200
    api_row = {**json.loads(response.body), "cost_final": True, "cost_with_children_partial": False}
    expected = "Budget stop retained.\n\n" + _FAILED_NOTICE
    assert api_row["terminal_host_notice"] == expected
    monkeypatch.setattr(cli, "_client", lambda *_a, **_kw: SimpleNamespace(
        request=lambda *_a, **_kw: {"task_id": "child"},
    ))
    monkeypatch.setattr(cli, "_wait_task", lambda *_a, **_kw: api_row)
    assert cli.main(["run", "--no-stream", *(["--jsonl"] if jsonl else []), "Produce the report."]) == 0
    captured = capsys.readouterr()
    if jsonl:
        assert json.loads(captured.out.splitlines()[-1])["result"]["terminal_host_notice"] == expected
    else:
        assert captured.err == "[Host status]\n" + expected + "\n"


@pytest.mark.parametrize("base,expected", [
    ("", _FAILED_NOTICE),
    (_FAILED_FULL, _FAILED_NOTICE),  # an older row already holding the full text
    ("Other host fact.\n\n" + _FAILED_FULL, "Other host fact.\n\n" + _FAILED_NOTICE),
], ids=["raw", "old-joined", "old-joined-base"])
def test_any_earlier_projection_converges_without_doubling_or_moving_the_hash(base, expected):
    row = _audit(terminal_runs=[_FAILED_RUN])
    if base:
        row["terminal_host_notice"] = base
    # Every form of one row is one identity: the hash of the raw row with the same host base.
    raw = _audit(terminal_runs=[_FAILED_RUN])
    if base.startswith("Other host fact."):
        raw["terminal_host_notice"] = "Other host fact."
    identity = _child_result_sha256(raw)
    assert terminal_host_notice_text(row) == expected
    assert terminal_host_notice_text(row, include_empty=True) == expected.replace(_FAILED_NOTICE, _FAILED_FULL)
    once = public_task_result(row)
    twice = public_task_result(once)
    assert once["terminal_host_notice"] == twice["terminal_host_notice"] == expected
    assert row.get("terminal_host_notice", "") == base  # a read never rewrites the stored row
    assert _child_result_sha256(row) == _child_result_sha256(once) == _child_result_sha256(twice) == identity
    if not base.startswith("Other host fact."):
        assert identity == _FAILED_HASH


@pytest.mark.parametrize("foreign", [
    "Pending patch decisions: none recorded by the current audit.",  # the label, not this audit's rendering
    _FAILED_FULL + "\n\nA later host fact.",
    _FAILED_NOTICE + " Not the same sentence.",
], ids=["empty-label", "not-a-suffix", "extended-sentence"])
def test_other_host_text_mentioning_custody_words_is_never_stripped(foreign):
    row = _audit(terminal_runs=[_FAILED_RUN])
    row["terminal_host_notice"] = foreign
    assert terminal_host_notice_text(row) == foreign + "\n\n" + _FAILED_NOTICE
    assert terminal_host_notice_text(row, include_empty=True) == foreign + "\n\n" + _FAILED_FULL
    clean = _audit(terminal_runs=[{"run_id": "ok", "state": "succeeded"}])
    clean["terminal_host_notice"] = foreign
    assert terminal_host_notice_text(clean) == terminal_host_notice_text(clean, include_empty=True) == foreign


def test_wrap_up_receipt_stays_for_every_reader_but_its_own_row():
    from ouroboros.owner_stop_receipt import FIELD, receipt_text
    from ouroboros.task_finalization import terminal_custody_notice_text

    row = _audit(terminal_runs=[_FAILED_RUN], **{FIELD: {"task_id": "root", "gap": "fixture gap"}})
    receipt = receipt_text(row["delegate_terminal_reconciliation"])
    assert "Descendant membership unknown: fixture gap" in receipt
    assert terminal_host_notice_text(row) == _FAILED_NOTICE + "\n\n" + receipt
    assert public_task_result(row)["terminal_host_notice"] == _FAILED_NOTICE + "\n\n" + receipt
    assert terminal_host_notice_text(row, include_empty=True) == _FAILED_FULL + "\n\n" + receipt
    assert terminal_custody_notice_text(row) == _FAILED_NOTICE
    only_receipt = _audit(**{FIELD: {"task_id": "root", "gap": "fixture gap"}})
    assert terminal_host_notice_text(only_receipt) == terminal_host_notice_text(only_receipt, include_empty=True) == receipt
    assert terminal_custody_notice_text(only_receipt) == ""


def test_a_silent_finish_stays_exactly_as_quiet_as_before():
    from ouroboros.task_finalization import quiet_finish

    silent = {"wake_silent": True}
    assert quiet_finish({**silent, **_audit(terminal_runs=[{"run_id": "ok", "state": "succeeded"}])}) is True
    assert quiet_finish({**silent, **_audit(terminal_runs=[_FAILED_RUN])}) is False
    # A category that is present but unreadable is not an empty one: it keeps its line, so it never goes quiet.
    malformed = _audit(pending_invocation_ids="not-a-list")
    assert quiet_finish({**silent, **malformed}) is False
    line = "Pending delegated invocations: none recorded by the current audit."
    assert terminal_host_notice_text(malformed) == line


@pytest.mark.parametrize("frozen,expected", [
    (provider_terminal_body("Answer.", "Budget stop retained.\n\n" + _FAILED_FULL), "Answer."),
    (provider_terminal_body("Older explicit reply", "Budget stop retained.\n\n" + _FAILED_FULL),
     provider_terminal_body("Older explicit reply", "Budget stop retained.\n\n" + _FAILED_FULL)),
    ("Exact authored reply", "Exact authored reply"),
    ("", ""),
], ids=["old-composition", "different-reply", "authored", "deliberately-empty"])
def test_presence_replay_undoes_only_a_recorded_host_composition(tmp_path, frozen, expected):
    from ouroboros.presence_runner import _cached_result

    _stored_failed_child(tmp_path, metadata={"presence": {}, "presence_outcome": "message",
                                             "presence_result_text": frozen})
    assert _cached_result(tmp_path, "child").text == expected


def test_delivery_snapshot_keeps_unknown_custody_without_global_fallback(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_current as current, store

    reads = []
    write_task_result(tmp_path, "own", "running")

    def unavailable(root, task_id):
        reads.append(task_id)
        raise store.StoreBusy("fixture contention")

    monkeypatch.setattr(current, "task_snapshot", unavailable)
    monkeypatch.setattr(current, "snapshot", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("delivery fell back to unrelated custody")))
    usage = {}
    event = prepare_terminal_send_event(
        tmp_path, {"id": "own", "chat_id": 1}, "Answer.", usage,
        {"type": "send_message", "text": "Answer."}, presence=False,
    )
    assert reads == ["own"]
    assert usage["delegate_terminal_reconciliation"]["audit_status"] == "failed"
    assert usage["delegate_terminal_reconciliation"]["unreconciled"] == [
        "delegated_run_state_unknown:snapshot_unavailable"]
    assert "could not be verified" in event["terminal_custody_notice"]
    assert event["text"] == "Answer."


def test_loop_exit_defers_only_annotation_and_resets_its_scope_on_failure(tmp_path, monkeypatch):
    from ouroboros.delegate_custody_reconcile import defer_loop_exit_audit

    calls, usage = [], {}

    def reconcile(root, task_id, **kwargs):
        calls.append(("release", task_id))
        return [{"run_id": "released", "action": "settled"}]

    monkeypatch.setattr(custody, "reconcile_task_runs", reconcile)
    monkeypatch.setattr(delegate_terminal, "_audit_task_custody",
                        lambda root, task_id, result, **kw: calls.append(("audit", task_id)))
    monkeypatch.setattr(delegate_terminal, "record_terminal_reconciliation",
                        lambda root, task_id, result, **kw: calls.append(("record", task_id)))
    with pytest.raises(RuntimeError, match="before delivery"):
        with defer_loop_exit_audit(usage):
            assert custody.release_task_runs(tmp_path, "first") == [{"run_id": "released", "action": "settled"}]
            assert calls == [("release", "first")]
            raise RuntimeError("before delivery")
    assert usage["delegate_terminal_reconciliation"]["outcomes"] == [{"run_id": "released", "action": "settled"}]
    assert usage["delegate_terminal_reconciliation"]["audit_status"] == "pending"
    custody.release_task_runs(tmp_path, "second")
    assert calls == [("release", "first"), ("release", "second"), ("audit", "second"), ("record", "second")]
