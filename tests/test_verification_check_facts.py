"""#1585: a check that ran is not a criterion that was met.

``verify_and_record`` keeps its string return and its durable receipt; after the
receipt write it publishes the recorded verdict (``verification``) beside the call's
own classification, which stays the legacy adapter's (a handled FAIL is a completed,
``ok`` call). These tests drive the real registry and the real loop wrapper and read
every consumer: the durable tools.jsonl row and trace payload, the live frames, the
model's compact facts, the llm trace, history replay, and the reflection/error
readers that must NOT treat a failed criterion as a failed call.
"""
from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.test_process_environment import process_context as _process_context

process_context = _process_context

pytestmark = pytest.mark.serial  # real child processes (CI serial lane)

_KEYS = {"status", "contract_kind", "expected_match", "check", "check_rendering",
         "criterion_source", "receipt_ts"}


def _receipts(data, task_id):
    from ouroboros.outcomes import verification_receipts_path

    path = verification_receipts_path(data, task_id)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _printing(text):
    return [sys.executable, "-c", f"print({text!r})"]


def _verify(registry, workspace, **args):
    return registry.execute_result("verify_and_record", {
        "contract_kind": "explicit_command", "cwd": str(workspace), **args})


def _legacy(text):
    from ouroboros.tools.tool_result import LegacyTextResultAdapter

    typed = LegacyTextResultAdapter.from_text("verify_and_record", text)
    return typed.status, typed.code


# ---------------------------------------------------------------------------
# Producer: the verdict is published beside an unchanged call classification
# ---------------------------------------------------------------------------


def test_text_mismatch_is_a_completed_call_with_a_failed_verdict(process_context):
    registry, ctx, workspace, data = process_context
    result = _verify(registry, workspace, check=_printing("hello"), expected="goodbye",
                     criterion_id="C1", criterion_source="task_stated")
    [receipt] = _receipts(data, ctx.task_id)
    fact = dict(result.meta["verification"])
    assert (result.status, result.code) == ("ok", "OK") == _legacy(result.text)
    assert result.text.startswith("verify_and_record [explicit_command] FAIL: exit=0 expected='goodbye'")
    assert set(fact) == _KEYS | {"criterion_id", "matched"}
    assert fact == {
        "status": "fail", "contract_kind": "explicit_command", "expected_match": "substring",
        "matched": False, "check": receipt["check"], "check_rendering": receipt["check_rendering"],
        "criterion_id": "C1", "criterion_source": "task_stated", "receipt_ts": receipt["ts"],
    }
    assert receipt["status"] == "fail" and receipt["matched"] is False and receipt["returncode"] == 0


@pytest.mark.parametrize("mode,expected,output,matches", [
    ("exact", "hello", "hello world", False),
    ("exact", "hello world", "hello world", True),
    ("exact_line", "two", "one\ntwo three", False),
    ("exact_line", "two", "one\ntwo", True),
    ("json_equals", '{"a": 1}', '{"a": 2}', False),
    ("json_equals", '{"b": 2, "a": 1}', '{"a": 1, "b": 2}', True),
])
def test_each_match_mode_reports_its_own_evaluated_match(process_context, mode, expected, output, matches):
    registry, ctx, workspace, data = process_context
    result = _verify(registry, workspace, check=_printing(output), expected=expected, expected_match=mode)
    fact = result.meta["verification"]
    assert (fact["expected_match"], fact["matched"]) == (mode, matches)
    assert fact["status"] == ("pass" if matches else "fail") == _receipts(data, ctx.task_id)[-1]["status"]
    assert (result.status, result.code) == ("ok", "OK")


def test_no_expected_text_means_no_comparison_ran(process_context):
    from ouroboros.tools.process_facts import consume_last_process_facts

    registry, ctx, workspace, data = process_context
    consume_last_process_facts()
    result = _verify(registry, workspace, check=[sys.executable, "-c", "raise SystemExit(3)"])
    fact = result.meta["verification"]
    assert fact["status"] == "fail" and "matched" not in fact
    # The exit stays a process fact on its own channel, never a second schema here.
    assert "exit_code" not in fact and "returncode" not in fact
    assert consume_last_process_facts()["exit_code"] == 3
    assert _receipts(data, ctx.task_id)[-1]["matched"] is None
    assert (result.status, result.code) == ("ok", "OK")


def test_timeout_records_no_comparison_in_receipt_and_published_fact(process_context, monkeypatch):
    registry, ctx, workspace, data = process_context

    def timeout(cmd, **_kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", timeout)
    result = _verify(registry, workspace, check=_printing("x"), expected="x")
    receipt = _receipts(data, ctx.task_id)[-1]
    fact = result.meta["verification"]
    assert receipt["matched"] is None and receipt["returncode"] is None
    from ouroboros._outcome_receipts import verification_receipt_ledger_row
    assert verification_receipt_ledger_row(receipt)["matched"] is None
    # Historical receipts are not rewritten by this new producer convention.
    assert verification_receipt_ledger_row(dict(receipt, matched=False))["matched"] is False
    assert fact["status"] == "fail" and "matched" not in fact
    assert fact["detail"].startswith("check timed out after")
    assert (result.status, result.code) == ("ok", "OK") == _legacy(result.text)


@pytest.mark.parametrize("case,prefix,status", [
    ("differ", "bytes differ at offset 3", "fail"),
    ("missing", "bytes_equal: file not found: b.bin", "fail"),
    ("equal", "bytes_equal: a.bin == b.bin", "pass"),
])
def test_bytes_equal_keeps_the_producers_own_detail(process_context, case, prefix, status):
    registry, ctx, workspace, data = process_context
    (workspace / "a.bin").write_bytes(b"abcdef")
    if case != "missing":
        (workspace / "b.bin").write_bytes(b"abcdef" if case == "equal" else b"abcXef")
    result = _verify(registry, workspace, check=["true"], expected_match="bytes_equal",
                     artifact_paths=["a.bin", "b.bin"])
    fact = result.meta["verification"]
    # A missing file is not "differing bytes": no byte verdict, so no `matched`, and the
    # recorded sentence and the durable nullable match both say which it was.
    assert fact["detail"].startswith(prefix) and fact["status"] == status
    assert fact.get("matched", "absent") == {"differ": False, "missing": "absent", "equal": True}[case]
    assert fact["expected_match"] == "bytes_equal"
    from ouroboros._outcome_receipts import verification_receipt_ledger_row
    receipt = _receipts(data, ctx.task_id)[-1]
    match = {"differ": False, "missing": None, "equal": True}[case]
    assert receipt["matched"] is match
    assert verification_receipt_ledger_row(receipt)["matched"] is match


def test_receipt_custody_failure_stays_an_error_without_a_recorded_verdict(process_context, monkeypatch):
    registry, ctx, workspace, data = process_context
    monkeypatch.setattr("ouroboros.tools.verify.append_verification_receipt", lambda *_a, **_k: False)
    result = _verify(registry, workspace, check=_printing("hello"), expected="goodbye")
    assert (result.status, result.code) == ("error", "TOOL_ERROR")
    assert "receipt_custody_failed" in result.text and "verification" not in result.meta
    assert _receipts(data, ctx.task_id) == []


@pytest.mark.parametrize("secret", ["1", "a"])
def test_selected_secret_cannot_rewrite_the_verdict_vocabulary(process_context, monkeypatch, secret):
    registry, ctx, workspace, data = process_context
    monkeypatch.setattr("ouroboros.config.load_settings", lambda: {"CUSTOM_KEY": secret})
    result = _verify(registry, workspace, check=[sys.executable, "-c", "import os; print(os.environ['TOKEN'])"],
                     expected="zzz", env_from_settings={"TOKEN": "CUSTOM_KEY"})
    receipt = _receipts(data, ctx.task_id)[-1]
    fact = result.meta["verification"]
    assert (fact["status"], fact["contract_kind"], fact["expected_match"], fact["criterion_source"]) == (
        "fail", "explicit_command", "substring", "agent_defined")
    assert fact["receipt_ts"] == receipt["ts"] and fact["check_rendering"] == receipt["check_rendering"]
    assert fact["check"] == receipt["check"]  # the identity text is masked exactly as the receipt's


def test_wide_characters_stay_inside_the_result_meta_bound(process_context):
    registry, _ctx, workspace, _data = process_context
    wide = "\U0001F600" * 400
    result = _verify(registry, workspace, check=["sh", "-c", f"echo {wide}"], expected="absent",
                     criterion_id=wide)
    fact = result.meta["verification"]
    assert fact["status"] == "fail" and len(fact["criterion_id"]) == 120
    assert "truncated" in fact["check"]


# ---------------------------------------------------------------------------
# The real loop wrapper and every downstream consumer
# ---------------------------------------------------------------------------


def _loop(process_context):
    registry, ctx, workspace, data = process_context
    frames = []
    ctx.event_queue = SimpleNamespace(put_nowait=lambda env: frames.append(env.get("data", env)), put=None)
    logs = data / "logs"
    logs.mkdir(exist_ok=True)
    counter = iter(range(1, 100))

    def call(**args):
        from ouroboros.loop_tool_execution import _execute_with_timeout

        tc = {"id": f"call-{next(counter)}", "function": {"name": "verify_and_record", "arguments": json.dumps(
            {"contract_kind": "explicit_command", "cwd": str(workspace), **args})}}
        return _execute_with_timeout(registry, tc, logs, 120, ctx.task_id)

    return call, frames, logs, ctx, data


def _tool_rows(logs):
    return [json.loads(line) for line in (logs / "tools.jsonl").read_text(encoding="utf-8").splitlines()]


def test_failed_check_reaches_every_consumer_as_one_completed_call(process_context):
    from ouroboros.loop_tool_execution import process_tool_results
    from ouroboros.observability import read_call_payload
    from ouroboros.outcomes import _OK_TOOL_STATUSES, _classify_tool_errors
    from ouroboros.post_task_synthesis import task_tool_metrics
    from ouroboros.tool_call_log import replay_evidence
    from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY

    call, frames, logs, ctx, data = _loop(process_context)
    exec_result = call(check=_printing("hello"), expected="goodbye", criterion_id="C1")
    verdict = exec_result["result_meta"]["verification"]
    assert exec_result["is_error"] is False and exec_result["result_meta"]["status"] == "ok"
    assert verdict["status"] == "fail" and verdict["matched"] is False
    # Durable record: the tools.jsonl settlement and the trace payload.
    settled = _tool_rows(logs)[-1]
    assert (settled["type"], settled["status"], settled["is_error"]) == ("tool_call", "ok", False)
    assert settled["tool_result_meta"]["verification"] == verdict and settled["exit_code"] == 0
    _manifest, payload, _ = read_call_payload(data, task_id=ctx.task_id, call_id=exec_result["trace_ref"]["call_id"])
    assert payload["semantic_ok"] is True and payload["result_meta"]["verification"] == verdict
    # Live: the settled frame and the finished frame both carry the verdict, once each.
    live = {frame.get("type"): frame for frame in frames}
    assert live["tool_call"]["tool_result_meta"]["verification"] == verdict
    finished = live["tool_call_finished"]
    assert finished["verification"] == verdict and finished["is_error"] is False and finished["exit_code"] == 0
    assert finished["invocation_id"] == settled["invocation_id"]
    # Model view and trace: compact facts on the canonical row, the full verdict on the trace row.
    messages, llm_trace = [], {"tool_calls": []}
    assert process_tool_results([exec_result], messages, llm_trace, lambda _m: None) == 0
    facts = messages[-1][TOOL_RESULT_RECORD_KEY]["facts"]
    assert facts["verification"] == {"status": "fail", "contract_kind": "explicit_command",
                                     "expected_match": "substring", "matched": False, "criterion_id": "C1"}
    assert (facts["status"], facts["is_error"], facts["exit_code"]) == ("ok", False, 0)
    [row] = llm_trace["tool_calls"]
    assert row["verification"] == verdict and row["status"] == "ok"
    # No error bucket and no error in the task-facts census, for a handled criterion failure.
    assert not row["is_error"] and row["status"] in _OK_TOOL_STATUSES
    assert task_tool_metrics(llm_trace)["tool_errors"] == 0
    assert not any(_classify_tool_errors(llm_trace).values())
    # History replay: the settled observation carries the verdict and the live preview.
    observation = next(o for o in replay_evidence(data, ctx.task_id)["observations"] if o["fact"] == "settled")
    assert (observation["status"], observation["verification"]) == ("ok", verdict)
    assert observation["result_preview"] == finished["result_preview"]
    assert observation["key"] == f"tool:{ctx.task_id}:{finished['invocation_id']}"
    assert observation["ts"] == settled["ts"] and observation["exit_code"] == 0


def test_failed_then_passing_check_keeps_both_verdicts_and_counts_no_error(process_context):
    from ouroboros.loop_tool_execution import process_tool_results
    from ouroboros.tool_call_log import replay_evidence

    call, _frames, _logs, ctx, data = _loop(process_context)
    target = ctx.workspace_root / "out.txt"
    target.write_text("draft\n", encoding="utf-8")
    check = [sys.executable, "-c", f"print(open({str(target)!r}).read())"]
    first = call(check=check, expected="final")
    target.write_text("final\n", encoding="utf-8")
    second = call(check=check, expected="final")
    llm_trace = {"tool_calls": []}
    assert process_tool_results([first, second], [], llm_trace, lambda _m: None) == 0
    assert [row["verification"]["status"] for row in llm_trace["tool_calls"]] == ["fail", "pass"]
    settled = [o for o in replay_evidence(data, ctx.task_id)["observations"] if o["fact"] == "settled"]
    assert [(o["status"], o["verification"]["status"]) for o in settled] == [("ok", "fail"), ("ok", "pass")]
    assert "result_preview" in settled[0] and "result_preview" not in settled[1]
    assert [r["status"] for r in _receipts(data, ctx.task_id)] == ["fail", "pass"]


def test_custody_failure_through_the_loop_stays_one_error_without_a_verdict(process_context, monkeypatch):
    from ouroboros.tool_call_log import replay_evidence

    monkeypatch.setattr("ouroboros.tools.verify.append_verification_receipt", lambda *_a, **_k: False)
    call, frames, _logs, ctx, data = _loop(process_context)
    exec_result = call(check=_printing("hello"), expected="goodbye")
    assert exec_result["is_error"] is True and "verification" not in exec_result["result_meta"]
    finished = next(frame for frame in frames if frame.get("type") == "tool_call_finished")
    assert finished["is_error"] is True and "verification" not in finished
    settled = next(o for o in replay_evidence(data, ctx.task_id)["observations"] if o["fact"] == "settled")
    assert settled["status"] == "error" and "verification" not in settled


def test_replay_ignores_rows_without_a_typed_verdict():
    from ouroboros.tool_call_log import _check_verdict

    assert _check_verdict({"tool_result_meta": None}) == {}
    assert _check_verdict({"tool_result_meta": {"verification": "fail"}}) == {}
    assert _check_verdict({"result_preview": "verify_and_record [explicit_command] FAIL"}) == {}
