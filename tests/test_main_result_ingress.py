"""Main sends the measured batch view, retaining exact producer sources.

These tests need the root loop callback wiring and ingress-lane keyword API.
They deliberately exercise the real loop/registry/tool executor, not a direct
call to the projector. The shared provider fixture makes no external calls.
"""
from copy import deepcopy
from dataclasses import replace
import math
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import context, loop, loop_tool_execution
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.tools.registry import ToolEntry
from ouroboros.tools.tool_result import ToolResult
from tests.test_main_authored_context import call
from tests.test_main_authored_context import main_loop as _main_loop

main_loop = _main_loop


pytestmark = pytest.mark.serial
TOOL = "fixture_ingress_result"


def _register_results(fixture, monkeypatch, outputs):
    calls = []

    def result(_ctx, name):
        calls.append(name)
        return outputs[name]

    fixture.registry.register(ToolEntry(
        name=TOOL,
        schema={"name": TOOL, "description": "Return an already prepared local fixture.",
                "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                               "required": ["name"]}},
        handler=result,
    ))
    # The fixture is pure/read-only; exercise the existing parallel batch path.
    monkeypatch.setattr(loop_tool_execution, "_PARALLEL_SAFE_TOOLS",
                        loop_tool_execution._PARALLEL_SAFE_TOOLS | {TOOL})
    return calls


def _observe_measurements(monkeypatch, *, minimum_unfit=False):
    observations = []

    def measure(plan, messages, schemas, mode, effort, round_id):
        # A deterministic complete-candidate measurement oracle: non-result
        # input has a fixed measured cost; each character of a result envelope
        # or body costs one synthetic token. No allocator/persistence is mocked.
        # accepted=True has the authored-view meaning, NOT physical-fit proof.
        window = plan.window_tokens
        reserve = plan.output_reserve_tokens
        base = window if minimum_unfit else 1000
        measured = base + sum(len(str(row.get("content") or "")) for row in messages
                              if row.get("role") == "tool")
        facts = {"accepted": True, "strict_bound_proven": False,
                 "estimated_input_tokens": measured, "response_reserve_tokens": reserve,
                 "capacity_total_tokens": window, "target_total_tokens": None,
                 "measurement_basis": "cold_estimate", "measurement_density": 1.0,
                 "predicted_capacity_miss": measured + reserve > window,
                 "route_fp": plan.route_fp, "round_id": round_id}
        observations.append({"plan": plan, "messages": deepcopy(messages), "schemas": deepcopy(schemas),
                             "mode": mode, "effort": effort, "facts": facts})
        return facts

    monkeypatch.setattr(loop, "_measure_main_context_view", measure)
    return observations


def _tool_messages(request):
    return {row["tool_call_id"]: row["content"] for row in request["messages"]
            if row.get("role") == "tool"}


def test_main_delivers_complete_small_result_without_rewriting_it(main_loop, monkeypatch):
    f = main_loop
    original = "Exact small producer output.\nFinal status: complete.\n"
    executed = _register_results(f, monkeypatch, {"small": ToolResult(status="ok", code="OK", text=original)})
    measurements = _observe_measurements(monkeypatch)
    answer, _, trace = f.run([
        call(TOOL, {"name": "small"}, "small-result"), {"content": "done"},
    ])

    assert answer == "done" and executed == ["small"]
    assert _tool_messages(f.inputs[-1])["small-result"] == original
    assert measurements, "Main must pass its real ingress measurement callback"
    assert all(row["schemas"] == f.inputs[0]["tools"] for row in measurements)
    assert next(row for row in trace["tool_calls"] if row["tool_call_id"] == "small-result")["is_error"] is False


@pytest.mark.parametrize("minimum_unfit", [False, True], ids=["shared-eighth", "required-envelopes-unfit"])
def test_main_parallel_results_share_room_and_keep_sources_and_outcomes(main_loop, monkeypatch, minimum_unfit):
    f = main_loop
    f.ctx.context_fit_plan = replace(f.ctx.context_fit_plan, window_tokens=200_000)
    originals = {"ok": "RESULT_OK\n" + "Ω" * 80_000 + "\nPROCESS_EXIT=0",
                 "error": "RESULT_ERROR\n" + "Ж" * 80_000 + "\nPROCESS_EXIT=7"}
    executed = _register_results(f, monkeypatch, {
        "ok": ToolResult(status="ok", code="OK", text=originals["ok"], meta={"exit_code": 0}),
        "error": ToolResult(status="error", code="TOOL_REPORTED_FAILURE",
                            text=originals["error"], meta={"exit_code": 7}),
    })
    measurements = _observe_measurements(monkeypatch, minimum_unfit=minimum_unfit)
    request = call(TOOL, {"name": "ok"}, "large-ok")
    request["tool_calls"].extend(call(TOOL, {"name": "error"}, "large-error")["tool_calls"])
    assert loop_tool_execution.tool_calls_can_run_parallel(request["tool_calls"])

    answer, _, trace = f.run([request, {"content": "done"}])

    assert answer == "done" and sorted(executed) == ["error", "ok"]
    batches = trace["tool_result_delivery"]
    assert len(batches) == 1, "Both completed results must reach one shared allocation"
    assert measurements[0]["facts"]["accepted"] is True
    assert measurements[0]["facts"]["predicted_capacity_miss"] is True
    batch = batches[0]
    assert batch["results"] == 2
    assert batch["status"] == ("minimum_view_unfit" if minimum_unfit else "projected")
    visible = _tool_messages(f.inputs[-1])
    assert set(visible) == {"large-ok", "large-error"}
    assert all(isinstance(text, str) and text for text in visible.values())
    outcomes = {row["tool_call_id"]: row for row in trace["tool_calls"]}
    for row in outcomes.values():
        name = "ok" if row["tool_call_id"] == "large-ok" else "error"
        source = row["result_source_ref"]
        retained = read_actor_source_bytes(f.ctx.drive_root, "authored-main", source)
        assert retained == originals[name].encode("utf-8")
        assert source["sha256"] in visible[row["tool_call_id"]]
        assert row["result_partial"] is True
    assert outcomes["large-ok"]["is_error"] is False
    assert outcomes["large-error"]["is_error"] is True
    assert "TOOL_REPORTED_FAILURE" in visible["large-error"]
    displayed_body = sum(text.count("Ω") + text.count("Ж") for text in visible.values())
    assert displayed_body <= math.ceil(200_000 / 8), "One eighth for the batch, not for each result"
    if not minimum_unfit:
        assert displayed_body > 0
        assert "PROCESS_EXIT=0" in visible["large-ok"]
        assert "PROCESS_EXIT=7" in visible["large-error"]
        final = measurements[-1]["facts"]
        assert final["estimated_input_tokens"] + final["response_reserve_tokens"] <= final["capacity_total_tokens"]
    else:
        # Honest minimum-view failure must not become an empty second result or
        # a false success merely because the authored-view callback accepts it.
        assert measurements[-1]["facts"]["predicted_capacity_miss"] is True


def test_ingress_after_real_fallback_uses_adopted_plan_and_sent_envelope(main_loop, monkeypatch):
    f = main_loop
    original_plan = f.ctx.context_fit_plan
    f.ctx.active_effort_override = "max"
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai/alternate")
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **kw: (
        {"model": task["model"], "provider": "openai"},
        SimpleNamespace(route_fp="adopted-alternate", status="confirmed", stale=False, window_tokens=200_000)))
    executed = _register_results(f, monkeypatch, {
        "small": ToolResult(status="ok", code="OK", text="Fallback result stays exact."),
    })
    measurements = _observe_measurements(monkeypatch)
    refused = httpx.HTTPStatusError(
        "fixture refusal", request=httpx.Request("POST", "https://fixture.invalid"),
        response=httpx.Response(400, json={"error": {"message": "fixture refusal"}}),
    )

    answer, _, _ = f.run([refused, call(TOOL, {"name": "small"}, "fallback-result"), {"content": "done"}])

    assert answer == "done" and executed == ["small"]
    assert [request["model"] for request in f.inputs] == ["openai/test-model", "openai/alternate", "openai/alternate"]
    assert measurements, "Fallback tool results must still use Main's allocation callback"
    for measured in measurements:
        assert measured["plan"] is not original_plan
        assert measured["plan"].model == "openai/alternate"
        assert measured["facts"]["route_fp"] == "adopted-alternate"
        assert measured["facts"]["capacity_total_tokens"] == 200_000
        assert measured["schemas"] == f.inputs[1]["tools"]
        assert measured["effort"] == f.inputs[1]["reasoning_effort"] == "max"
        assert measured["mode"] == f.ctx.active_context_mode
    assert _tool_messages(f.inputs[-1])["fallback-result"] == "Fallback result stays exact."
