"""A source address and a chosen view survive every batch delivery policy."""
import json

import pytest

from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.context_fit import estimate_context_prompt_tokens
from ouroboros.tool_result_delivery import project_tool_result_batch


def test_consolidator_receives_the_source_in_its_actual_message_text(tmp_path):
    original = "Retained source. " * 1000
    rows, receipt = project_tool_result_batch(
        [{"tool_call_id": "read", "result": original}], [], [], drive_root=tmp_path,
        task_id="t", fit_candidate=lambda messages, tools: {
            "accepted": len(json.dumps(messages)) < 2400})
    assert receipt["status"] == "projected"
    # The consolidator forwards only row['result'], not the row's host metadata.
    shown = rows[0]["result"]
    marker = json.loads(shown.split("\n[Tool result source view]\n", 1)[1])
    assert marker["source_status"] == "ready"
    assert read_actor_source_bytes(tmp_path, "t", marker["source_ref"]).decode() == original


def test_unknown_window_keeps_an_explicit_head_tail_request(tmp_path):
    original = "HEAD " + "middle " * 5000 + " TAIL"
    rows, receipt = project_tool_result_batch(
        [{"tool_call_id": "run", "result": original, "fn_name": "run_command",
          "tool_args": {"view_head_chars": 20, "view_tail_chars": 20}}], [], [],
        drive_root=tmp_path, task_id="t", policy="measured_frame",
        fit_candidate=lambda messages, tools: {"accepted": True, "capacity_total_tokens": None})
    assert receipt["status"] == "capacity_unknown"
    assert rows[0]["result_partial"]
    assert rows[0]["result_source_view"]["shown_ranges"] == [[0, 20], [len(original) - 20, len(original)]]
    assert read_actor_source_bytes(tmp_path, "t", rows[0]["result_source_ref"]).decode() == original


def test_measured_eighth_holds_with_different_json_escape_densities(tmp_path):
    def fit(messages, tools):
        return {"accepted": True, "estimated_input_tokens": estimate_context_prompt_tokens(messages, tools),
                "capacity_total_tokens": 18000, "response_reserve_tokens": 1024}

    rows, receipt = project_tool_result_batch(
        [{"tool_call_id": "plain", "result": "a" * 100000},
         {"tool_call_id": "escaped", "result": '"\n' * 6000}], [], [],
        drive_root=tmp_path, task_id="t", policy="measured_frame", fit_candidate=fit)
    assert receipt["status"] == "projected"
    assert all(row["result_partial"] for row in rows)
    assert 0 < receipt["unsolicited_body_tokens"] <= receipt["unsolicited_allowance_tokens"]
    assert receipt["fit"]["estimated_input_tokens"] + receipt["reserve_tokens"] <= 18000


def test_producer_footer_and_notes_are_in_the_measured_candidate(tmp_path):
    from ouroboros.tools.tool_result import ToolResult
    from tests.test_tool_result_delivery import _ctx, _deliver, _row

    measured_texts = []

    def fit(messages, tools):
        measured_texts.append(messages[-1]["content"])
        return {"accepted": True, "estimated_input_tokens": len(json.dumps(messages)),
                "capacity_total_tokens": 14000, "response_reserve_tokens": 1000}

    body = "Source body " * 10000
    note = "Recorded host warning: " + "w" * 2000
    typed = ToolResult(status="error", code="TOOL_REPORTED_FAILURE", text=body + note,
                       producer_text=body, host_annotations=(note,))
    messages, trace = _deliver(_ctx(tmp_path), [_row("one", "run_command", typed.text,
        is_error=True, typed=typed)], fit)
    delivered = messages[-1]["content"]
    assert delivered in measured_texts
    assert note in delivered and "PRODUCER_RESULT_SOURCE_JSON=" in delivered
    assert trace["tool_result_delivery"][-1]["status"] == "projected"
    assert len(json.dumps(messages)) + 1000 <= 14000


@pytest.mark.parametrize("image_cost", [0, 16000, 19500])
def test_completed_batch_measurement_includes_auto_image(tmp_path, monkeypatch, image_cost):
    from copy import deepcopy
    from types import SimpleNamespace
    from ouroboros.tools import vision
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.tool_result import ToolResult
    from ouroboros.loop_tool_execution import process_tool_results

    repo = tmp_path / "repo"
    repo.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path, task_id="review-image-budget")
    messages = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "body", "type": "function", "function": {"name": "fixture_read", "arguments": "{}"}},
        {"id": "picture", "type": "function", "function": {"name": "ext_fixture_screenshot", "arguments": "{}"}},
    ]}]
    ctx.messages = messages
    attached = []

    def attach(inner, path):
        attached.append(path)
        inner.messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,SYNTHETIC_ONLY"}}]})
        return True, "synthetic attachment; no image file was opened"

    monkeypatch.setattr(vision, "attach_local_image_to_context", attach)
    measurements = []
    window, reserve = 20000, 1000
    auto_image = bool(image_cost)

    def facts(candidate, schemas):
        # Controlled whole-request oracle: one token per delivered text char;
        # a native block has an explicit known cost. The real allocator runs.
        text = sum(len(row["content"]) for row in candidate if row["role"] == "tool")
        images = sum(isinstance(row.get("content"), list) for row in candidate)
        return {"accepted": True, "estimated_input_tokens": 500 + text + image_cost * images,
                "capacity_total_tokens": window, "target_total_tokens": None,
                "response_reserve_tokens": reserve, "measurement_basis": "synthetic_complete_candidate",
                "measurement_density": 1.0}

    def measure(candidate, schemas):
        measurements.append(deepcopy(candidate))
        return facts(candidate, schemas)

    def result(ident, name, body):
        return {"tool_call_id": ident, "fn_name": name, "result": body,
                "tool_args": {}, "args_for_log": {}, "is_error": False,
                "result_meta": {}, "tool_result": ToolResult(status="ok", code="OK", text=body)}

    picture = json.dumps({"auto_attach_image": "synthetic.png"} if auto_image else {"status": "ok"})
    rows = [result("body", "fixture_read", "HEAD" + "x" * 100000 + "TAIL"),
            result("picture", "ext_fixture_screenshot", picture)]
    trace = {"tool_calls": []}
    assert process_tool_results(rows, messages, trace, lambda *a, **k: None,
        SimpleNamespace(_ctx=ctx), fit_candidate=measure, tool_schemas=[]) == 0
    assert [row["role"] for row in messages][:3] == ["assistant", "tool", "tool"]
    assert len(attached) == int(auto_image)
    report = {"auto_image": auto_image, "batch": trace["tool_result_delivery"][0],
              "actual_input_tokens": facts(messages, [])["estimated_input_tokens"],
              "reserve": reserve, "window": window,
              "measured_image_counts": [sum(isinstance(row.get("content"), list) for row in batch)
                                        for batch in measurements]}
    print(json.dumps(report, sort_keys=True))
    # The producer's image must remain attached after the contiguous tool block.
    # This fails on the frozen candidate: every allocator measurement omitted it.
    if auto_image:
        assert all(any(isinstance(row.get("content"), list) for row in batch)
                   for batch in measurements), report
    if image_cost + 500 + reserve > window:
        assert report["batch"]["status"] == "minimum_view_unfit"
        assert report["actual_input_tokens"] + reserve > window
    else:
        assert report["actual_input_tokens"] + reserve <= window
    assert ctx.messages is messages
    assert trace["tool_calls"][1].get("image_attachment") == ({"status": "attached"} if auto_image else None)
    assert all(len(batch) >= 3 for batch in measurements)  # complete tool block in every fit
