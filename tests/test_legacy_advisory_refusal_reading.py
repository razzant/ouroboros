"""An old Advisory review refusal over a delivered answer reads Done with warnings, on read only.

Records from the retired task-acceptance machine could end with the host declining a
second paid panel over an unchanged answer (``identical_acceptance_refused``). That
decision folded the objective to ``fail`` whatever the enforcement, so a delivered
answer under explicit Advisory enforcement, whose panel only failed to reach a verdict,
read Failed with no cause. The readers now derive ``best_effort`` for exactly that
shape, keeping the stored status beside it (``ignored_status``); the stored record,
its review, its verdict, its money and its dates are never rewritten, the published
Main row keeps its words, and every other enforcement, verdict or execution keeps
the recorded fail. The fixture is synthetic, with the field shape of a real record.
"""
from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros.outcomes import normalize_outcome_axes, public_task_result
from ouroboros.project_dialogue import _completion_verdict, completion_status_label, outcome_phase

SENTENCE = "Nothing had changed since the last review, so the recorded verdict stands."


def _record(*, enforcement="advisory", review="degraded", signals=("DEGRADED",), execution="ok",
            reason="identical_acceptance_refused", status="completed", result="Done and merged: the reviewed change is in."):
    decision = {"status": "finalized_unaccepted", "reason": reason, "source": "task_acceptance_review",
                "rationale": "No new material since the paid panel; quoting the recorded verdict.",
                **({"enforcement": enforcement} if enforcement is not None else {})}
    review_axis = {"status": review, "trigger": "auto_effect_after_agent_advisory", "eligibility": "eligible",
                   "run_count": 1, "aggregate_signals": list(signals), "outcome_tier": "best_effort",
                   "enforcement_impact": "degrades_completion", "acceptance_decision": decision}
    return {
        "task_id": "legacy-review", "status": status, "reason_code": "final_message", "terminal_origin": "model_final",
        "result": result, "chat_id": 7, "project_id": "skills-page", "delegation_role": "root",
        "started_at": "2026-09-19T22:10:00+00:00", "ts": "2026-09-20T00:41:02+00:00",
        "accounted_upper_bound_usd": 41.61, "cost_accounting_status": "available", "cost_final": True,
        "artifact_status": "ready_no_changes", "review_status": copy.deepcopy(review_axis),
        "outcome_axes": {
            "schema_version": 1, "lifecycle": {"status": status},
            "execution": {"status": execution, "reason_code": "final_message" if execution == "ok" else "empty_final_text"},
            "artifacts": {"status": "ready_no_changes"},
            "objective": {"status": "fail", "source": "task_acceptance_review", "review_status": review,
                          "outcome_tier": "blocked_with_evidence", "reason": reason},
            "review": review_axis,
        },
    }


def test_the_refusal_reads_best_effort_beside_its_stored_fail_and_nothing_else_moves():
    record = _record()
    before = copy.deepcopy(record)
    axes = normalize_outcome_axes(record)
    assert axes["objective"] == {**record["outcome_axes"]["objective"], "status": "best_effort", "ignored_status": "fail"}
    assert axes["review"] == record["outcome_axes"]["review"]  # the verdict and the decision as recorded
    assert axes["execution"]["status"] == "ok" and axes["lifecycle"]["status"] == "completed"
    assert record == before, "the reader never writes the record it reads"
    assert normalize_outcome_axes({**record, "outcome_axes": axes}) == axes, "a second read changes nothing"
    assert outcome_phase(record, {}) == outcome_phase({}, record) == "warn"
    assert completion_status_label(record, {}) == "Done with warnings"
    assert _completion_verdict(record, {}) == SENTENCE


@pytest.mark.parametrize("variant, change", [
    ("blocking enforcement", {"enforcement": "blocking"}),
    ("no recorded enforcement", {"enforcement": None}),
    ("unknown enforcement", {"enforcement": "auto"}),
    ("a reviewer's FAIL", {"review": "fail", "signals": ("FAIL",)}),
    ("a FAIL among the signals", {"signals": ("DEGRADED", "FAIL")}),
    ("a failed execution", {"execution": "failed", "status": "failed"}),
    ("no delivered answer", {"execution": "failed", "result": ""}),
    ("execution ok with an empty result", {"result": ""}),
    ("execution ok with an unknown result", {"result": None}),
    ("an explicit unfinished stop", {"reason": "author_stop"}),
    ("spent review rounds", {"reason": "review_cycles_exhausted"}),
])
def test_anything_but_that_exact_shape_keeps_the_recorded_fail(variant, change):
    record = _record(**change)
    assert normalize_outcome_axes(record)["objective"]["status"] == "fail", variant
    assert completion_status_label(record, {}) == "Failed", variant


@pytest.mark.parametrize("change,sentence", [
    ({"enforcement": "blocking"}, SENTENCE),
    ({"enforcement": None}, SENTENCE),
    ({"enforcement": "unknown"}, SENTENCE),
    ({"review": "fail", "signals": ("FAIL",)}, SENTENCE),
    ({"enforcement": "blocking", "reason": "review_degraded"},
     "The reviewers did not reach a verdict on this answer."),
    ({"reason": "review_cycles_exhausted"},
     "The task used up its review rounds before the answer was signed off."),
])
def test_red_decision_without_execution_cause_explains_itself_without_rewriting(tmp_path, change, sentence):
    from ouroboros import store
    from ouroboros.task_results import load_task_result
    from tests._store_shared import seed_result, result_signature

    record = _record(**change)
    seed_result(tmp_path, record["task_id"], record)
    with store.open_store(tmp_path, write=False) as txn:
        original_bytes = txn.body_text(record["task_id"])
    signature = result_signature(tmp_path, record["task_id"])
    stored = load_task_result(tmp_path, record["task_id"])
    for row in (stored, public_task_result(stored)):
        assert completion_status_label(row, {}) == "Failed"
        assert _completion_verdict(row, {}) == _completion_verdict({}, row) == sentence
        assert row["outcome_axes"]["objective"]["status"] == "fail"
        assert row["review_status"] == record["review_status"]
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.body_text(record["task_id"]) == original_bytes
    assert result_signature(tmp_path, record["task_id"]) == signature


def test_execution_failure_keeps_priority_over_the_red_decision():
    record = _record(enforcement="blocking", execution="failed", status="failed")
    assert _completion_verdict(record, {}) == "The task ended without a final answer."
    record["outcome_axes"]["execution"]["reason_code"] = "future_execution_failure"
    assert _completion_verdict(record, {}) == "future_execution_failure."
    record["outcome_axes"]["execution"]["reason_code"] = "final_message"
    record["outcome_axes"]["review"]["acceptance_decision"]["status"] = "accepted"
    record["review_status"]["acceptance_decision"]["status"] = "accepted"
    assert _completion_verdict(record, {}) == "", "a successful decision cannot explain Failed"
    debt = _record(enforcement="blocking", status="failed")
    debt.update(reason_code="delegated_custody_unreconciled", delegated_runs_unreconciled=["run-a1"])
    assert _completion_verdict(debt, {}) == "Some delegated work was never reconciled."


def test_only_affirmative_answer_evidence_survives_the_compact_projection():
    from ouroboros.terminal_projection import _project_row
    record = _record()
    record.pop("terminal_origin")  # old answers predate the typed provenance
    before = copy.deepcopy(record)
    projected = _project_row(record["task_id"], record, {}, {"chat_id": 7})
    assert "result" not in projected  # the current compact carrier needs no second answer copy
    assert completion_status_label(projected, {}) == "Done with warnings"
    assert projected["outcome_axes"] == record["outcome_axes"]
    assert projected["_result_text_present"] is True
    assert record == before
    for change in ({"result": ""}, {"result": None}, {"terminal_origin": "host_notice"},
                   {"terminal_origin": "host_salvage"}, {"terminal_origin": "unknown"}):
        assert completion_status_label({**record, **change}, {}) == "Failed"
    record.pop("result")
    assert completion_status_label(record, {}) == "Failed"


@pytest.mark.parametrize("change,label", [
    ({}, "Done with warnings"),
    ({"result": ""}, "Failed"),
    ({"result": None}, "Failed"),
    ({"terminal_origin": "host_notice"}, "Failed"),
    ({"terminal_origin": "host_salvage"}, "Failed"),
    ({"terminal_origin": "unknown"}, "Failed"),
    ({"terminal_origin": ""}, "Done with warnings"),
    ({"status": "failed", "outcome_axes": {"objective": {"status": "fail", "source": "acceptance"}}}, "Failed"),
])
def test_stamp_sources_keep_raw_objective_while_compact_history_interprets_it(tmp_path, monkeypatch, change, label):
    from pathlib import Path
    import subprocess

    from ouroboros import store, task_store
    from ouroboros.preflight_node import resolve_node
    from ouroboros.task_results import load_task_result
    from ouroboros.terminal_projection import _project_row, stamp_facts
    from tests._store_shared import result_signature, seed_result

    record = {**_record(), **change}
    seed_result(tmp_path, record["task_id"], record)
    before = load_task_result(tmp_path, record["task_id"])
    signature = result_signature(tmp_path, record["task_id"])
    with store.open_store(tmp_path, write=False) as txn:
        body = txn.body_text(record["task_id"])
    projected = _project_row(record["task_id"], before, {}, {"chat_id": 7})
    assert projected["outcome_axes"] == record["outcome_axes"]
    assert projected["terminal_origin"] == record["terminal_origin"]
    assert projected["_result_text_present"] is bool(record["result"])
    assert "result" not in projected
    assert completion_status_label(projected, {}) == label
    for rows, source in (([({}, projected)], "terminal_root_projection"), ([], "task_results")):
        fact = stamp_facts(tmp_path, [record["task_id"]], rows=rows)[record["task_id"]]
        assert fact["source"] == source
        assert fact["status"] == record["status"]
        assert fact["objective_status"] == "fail", "a display interpretation is not a historical fact"
        assert "outcome_axes" not in fact

    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "chat.jsonl").write_text(json.dumps(projected) + "\n", encoding="utf-8")
    (logs / "progress.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(task_store, "documents_by_ids", lambda *_: pytest.fail("history fetched full bodies"))
    # Exercise both current-result enrichment and the compact row alone after result pruning.
    views = [projected]
    for result_available in (True, False):
        from ouroboros.gateway.history import make_chat_history_endpoint
        if not result_available:
            monkeypatch.setattr(task_store, "display_documents", lambda *_a, **_kw: {})
        response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"chat_id": "7"})))
        rows = json.loads(response.body)["messages"]
        replayed = next(row for row in rows if row.get("task_id") == record["task_id"])
        assert {"done": "Done", "warn": "Done with warnings", "error": "Failed"}[replayed["outcome_phase"]] == label
        views.append(replayed)
    node = resolve_node()
    assert node, "the compact-row/history consumer proof requires Node"
    script = """import {taskOutcomeSeverity, taskPresentation} from './web/modules/log_events.js';
let text = ''; for await (const chunk of process.stdin) text += chunk;
console.log(JSON.stringify(JSON.parse(text).map(row => taskPresentation(taskOutcomeSeverity(row)).headline)));"""
    checked = subprocess.run([node, "--input-type=module", "-e", script], input=json.dumps(views),
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    assert json.loads(checked.stdout) == [label] * len(views)
    assert load_task_result(tmp_path, record["task_id"]) == before
    assert result_signature(tmp_path, record["task_id"]) == signature
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.body_text(record["task_id"]) == body


@pytest.mark.parametrize("answer", ["Kept answer.", "", None])
def test_display_view_carries_answer_presence_without_reading_full_bodies(tmp_path, monkeypatch, answer):
    from ouroboros import task_store
    from ouroboros.task_reads import TaskReads
    from ouroboros.task_results import load_task_result, write_task_result

    record = _record(result=answer)
    write_task_result(tmp_path, record["task_id"], record["status"],
                      **{k: v for k, v in record.items() if k not in {"task_id", "status"}})
    before = load_task_result(tmp_path, record["task_id"])
    monkeypatch.setattr(task_store, "documents_by_ids", lambda *_: pytest.fail("display read fetched full bodies"))
    view = TaskReads(tmp_path).effective(record["task_id"])
    assert "result" not in view
    assert view["_result_text_present"] is bool(answer)
    assert completion_status_label(view, {}) == ("Done with warnings" if answer else "Failed")
    assert load_task_result(tmp_path, record["task_id"]) == before
    assert "_result_text_present" not in before, "the existence bit is a read projection, not storage"


def test_task_detail_and_history_read_it_the_same_way_and_the_published_main_row_keeps_its_words(tmp_path):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.gateway.tasks import api_task_get
    from ouroboros.task_results import load_task_result, write_task_result

    record = _record()
    fields = {key: value for key, value in record.items() if key not in ("task_id", "status")}
    write_task_result(tmp_path, "legacy-review", "completed", **fields)
    stored_before = json.dumps(load_task_result(tmp_path, "legacy-review"), sort_keys=True)
    assert public_task_result(load_task_result(tmp_path, "legacy-review"))["outcome_axes"]["objective"]["status"] == "best_effort"

    app = Starlette(routes=[Route("/api/tasks/{task_id}", endpoint=api_task_get, methods=["GET"])])
    app.state.drive_root = tmp_path
    detail = TestClient(app).get("/api/tasks/legacy-review").json()
    assert detail["outcome_axes"]["objective"]["status"] == "best_effort"
    assert detail["outcome_axes"]["objective"]["ignored_status"] == "fail"
    assert detail["review_status"]["acceptance_decision"]["reason"] == "identical_acceptance_refused"
    assert detail["accounted_upper_bound_usd"] == 41.61 and detail["status"] == "completed"

    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    main_row = {"ts": "2026-09-20T00:43:23Z", "direction": "system", "chat_id": 1, "type": "project_completion_summary",
                "task_id": "legacy-review", "project_id": "skills-page", "project_name": "Skills page",
                "target_label": "Skills page", "status": "completed", "text": "Skills page · Failed\nOpen the Project for details."}
    room_final = {"ts": "2026-09-20T00:41:01Z", "direction": "out", "chat_id": 7, "task_id": "legacy-review",
                  "format": "markdown", "text": record["result"], "task_terminal_status": "completed"}
    (logs / "chat.jsonl").write_text("\n".join(json.dumps(row) for row in (room_final, main_row)) + "\n", encoding="utf-8")
    (logs / "progress.jsonl").write_text("", encoding="utf-8")
    endpoint = make_chat_history_endpoint(tmp_path)
    room = json.loads(asyncio.run(endpoint(SimpleNamespace(query_params={"chat_id": "7"}))).body)["messages"]
    final = next(row for row in room if row.get("task_id") == "legacy-review")
    assert final["outcome_phase"] == "warn"
    assert final["outcome_axes"]["objective"]["status"] == "best_effort"
    main = json.loads(asyncio.run(endpoint(SimpleNamespace(query_params={"chat_id": "1"}))).body)["messages"]
    published = next(row for row in main if row.get("system_type") == "project_completion_summary")
    assert published["text"] == "Skills page · Failed\nOpen the Project for details.", "a published row is history"
    assert json.dumps(load_task_result(tmp_path, "legacy-review"), sort_keys=True) == stored_before


def test_a_current_advisory_author_finish_keeps_its_own_reading():
    """The result review's informed author finish is unchanged: Done, its critic's FAIL kept beside it."""
    record = {"status": "completed", "reason_code": "final_message", "outcome_axes": {
        "execution": {"status": "ok"},
        "objective": {"status": "pass", "source": "author_acceptance", "reason": "author_finish"},
        "review": {"status": "fail", "source": "result_review", "author_finish": True, "aggregate_signals": ["FAIL"],
                   "acceptance_decision": {"status": "finalized_unaccepted", "source": "result_review",
                                           "reason": "author_finish", "enforcement": "advisory"}}}}
    assert normalize_outcome_axes(record)["objective"]["status"] == "pass"
    assert completion_status_label(record, {}) == "Done"
