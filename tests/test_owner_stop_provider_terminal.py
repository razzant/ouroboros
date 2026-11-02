"""An accepted Stop in the unknown-provider wait survives the real terminal writer."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import agent_task_pipeline as pipeline, loop, loop_transport
from ouroboros import usage_accounting as accounting
from ouroboros.project_dialogue import _completion_verdict, completion_status_label
from ouroboros.task_results import load_task_result, write_task_result
from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN, owner_stop_control_id
from supervisor.queue import _request_finalization_grace as request_finalization_grace
from tests.test_owner_stop_terminal_command import _press
from tests.test_transport_death_retry import _LedgerLLM, _ledger, _loop_kwargs, _no_chain
from tests._shared import saved_settings


def _stop_during_unknown_wait(root, monkeypatch, button):
    """Stub only model/network/timing seams; publish real intent and mailbox control."""
    write_task_result(root, "t-death", "running", chat_id=1, _is_direct_chat=True)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    saved_settings(monkeypatch, root, OUROBOROS_MODEL_FALLBACKS="")  # a saved blank: no chain
    monkeypatch.setattr(loop, "_run_cross_model_fallback_chain", _no_chain)
    llm = _LedgerLLM(root, lambda: httpx.ReadError("controlled post-dispatch failure"))
    kwargs = _loop_kwargs(root, llm, [])
    kwargs["drive_logs"] = root / "logs"
    ctx = kwargs["tools"]._ctx
    ctx.task_id, ctx.task_attempt, ctx.budget_drive_root = "t-death", 1, root
    ctx.is_direct_chat, ctx.route_wait_on_primary = True, True
    ctx.task_metadata = {"root_task_id": "t-death"}
    accepted = {}
    clock = [loop_transport.time.monotonic()]
    monkeypatch.setattr(loop_transport, "time", SimpleNamespace(
        monotonic=lambda: clock[0], time=loop_transport.time.time))

    def press():
        assert not accepted
        intent = _press(root, "t-death", button)
        reason = REASON_OWNER_STOPPED_DIRECT_TURN if button == "stop_now" else "owner_requested_finalization"
        mid = request_finalization_grace(root, "t-death", reason, chat_id=1, stamp=1,
                                        control_msg_id=owner_stop_control_id(intent), quiet=True)
        assert mid
        accepted.update(intent=intent, reason=reason, message_id=mid)

    def sleep(_seconds, check):
        if button == "wrap_up":
            press()  # ordinary round-top drain: Wrap up during the outage
            assert check()
            return True
        return False  # reach the recovery probe without another provider call

    def recovered(*_args, **_kwargs):
        assert button == "stop_now"
        press()  # after the first peek, before the physical continuation is granted
        clock[0] += 3600000  # the real wait bound expires before its next iteration
        return {"kind": "upstream_http", "status_code": 200}

    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", sleep)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable", recovered)
    with accounting.usage_scope(accounting.UsageScope(drive_root=root, task_id="t-death",
                                                    root_task_id="t-death", global_limit_usd=100.0)):
        text, usage, trace = loop.run_llm_loop(**kwargs)
    assert llm.calls == 1 and accepted
    assert usage["reason_code"] == "provider_unavailable"
    assert usage["execution_status"] == "infra_failed"
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
    ledger = _ledger(root)
    assert len(ledger) == 1 and ledger[0]["state"] == "unresolved"
    assert accounting.usage_projection(root)["unresolved_upper_bound_usd"] == 1.0
    if button == "stop_now":
        assert ctx._transport_repeat_control_reason == REASON_OWNER_STOPPED_DIRECT_TURN
        assert ctx._skip_post_task_synthesis is True
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *_a, **_kw: None)
    events = []
    pipeline.emit_task_results(SimpleNamespace(drive_root=root, repo_dir=root), None, None, events,
        {"id": "t-death", "type": "task", "chat_id": 1, "_is_direct_chat": True},
        text, usage, trace, 0.0, root / "logs", ctx=ctx)
    stored = load_task_result(root, "t-death")
    if stored.get("root_phase_checkpoint"):
        write_task_result(root, "t-death", stored["status"], root_phase_checkpoint={"post_task_synthesis": "completed"})
        stored = load_task_result(root, "t-death")
    assert stored["result"] == text
    assert stored["status"] == "failed"
    assert stored["outcome_axes"]["execution"]["status"] == "infra_failed"
    assert _ledger(root) == ledger, "publishing Stop does not settle the unknown provider attempt"
    return stored, next(e for e in events if e["type"] == "send_message"), accepted


@pytest.mark.parametrize("button,word", [("stop_now", "Stopped"), ("wrap_up", "Failed")])
def test_accepted_provider_wait_control_survives_storage_and_both_presenters(tmp_path, monkeypatch, button, word):
    from ouroboros.preflight_node import resolve_node
    from ouroboros.test_evidence import diagnostic_directory, write_record
    from ouroboros.tools.control_task_results import _get_task_result

    stored, final, accepted = _stop_during_unknown_wait(tmp_path, monkeypatch, button)
    if button == "stop_now":
        origin = stored["cancel_origin"]
        assert origin["control_reason"] == accepted["reason"]
        assert origin["request_id"] == accepted["intent"]["request_id"]
        assert _completion_verdict(stored, {}) == "Stopped from the app (Stop now)"
        reader = _get_task_result(SimpleNamespace(drive_root=tmp_path, task_metadata={}), "t-death")
        assert "[OWNER_CONTROL] command=stop_now" in reader and "[CANCELLED_BY]" not in reader
        assert "no terminal provider outcome" in reader
    rows = [stored, final["progress_meta"]]
    assert all(completion_status_label(row, {}) == word for row in rows)
    controls = []
    if button == "stop_now":
        for change in ({"cancel_origin": {**stored["cancel_origin"], "control_reason": "deadline_local"}},
                       {"cancel_origin": {**stored["cancel_origin"], "control_reason": ""}},
                       {"reason_code": "worker_crash_signal"}):
            control = {**stored, **change}
            assert completion_status_label(control, {}) == "Failed"
            controls.append(control)
    # Do not invent fixture-only origins: feed the actual saved/live records to JS.
    node = resolve_node()
    assert node, "this consumer proof requires Node"
    script = """import {taskOutcomeSeverity, taskReasonDetail} from './web/modules/log_events.js';
let text = ''; for await (const chunk of process.stdin) text += chunk;
console.log(JSON.stringify(JSON.parse(text).map(row => ({
  severity:taskOutcomeSeverity(row), cause:taskReasonDetail(row)
}))));"""
    checked = subprocess.run([node, "--input-type=module", "-e", script], input=json.dumps(rows + controls),
                             cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    views = json.loads(checked.stdout)
    assert [v["severity"] for v in views] == ["cancelled" if button == "stop_now" else "error"] * len(rows) + ["error"] * len(controls)
    assert all(v["cause"].rstrip(".") == _completion_verdict(row, {}).rstrip(".") for v, row in zip(views, rows + controls))
    write_record(diagnostic_directory(tmp_path / "proof", "owner-provider-stop") / f"{button}.json",
                 {"stored": stored, "final": final, "js": views, "accepted": accepted})
