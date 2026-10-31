"""A solve response, requested route and post-task authorship are distinct facts."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros import loop
from ouroboros.loop_llm_call import call_llm_with_retry
from ouroboros.outcomes import collect_trace_refs
from ouroboros.task_finalization import model_execution_projection
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_loop_compaction import _ctx
from tests._store_shared import read_result, result_signature


class Model:
    def __init__(self, message, usage):
        self.message, self.usage = message, usage

    def chat(self, **kwargs):
        return self.message, {"cost": 0.0, "prompt_tokens": 1, "completion_tokens": 1,
                              "provider": "openrouter", **self.usage}


def dispatch(ctx, model, *, message=None, reported=None):
    ctx.active_model = model
    ctx.llm = Model(message or {"role": "assistant", "content": "useful answer"},
                    {"resolved_model": reported} if reported is not None else {})
    return loop._dispatch_round_model(ctx, None, attempt_cap=1)


def test_real_dispatch_preserves_last_solve_through_empty_and_forced_calls(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.accumulated_usage["initial_model_request"] = {"model": "primary", "use_local": False}
    dispatch(ctx, "fallback", reported="Provider display alias")
    observed = model_execution_projection(ctx.accumulated_usage)
    assert observed == {
        "requested_model": "primary", "requested_use_local": False,
        "used_model": "fallback", "reported_model": "Provider display alias",
        "used_local": False, "provider": "openrouter",
        "llm_call_id": ctx.accumulated_usage["llm_call_refs"][-1]["llm_call_id"],
        "last_llm_error_kind": None,
        "source": "usable_solve_response",
    }
    ctx.round_idx += 1
    failed = dispatch(ctx, "empty", message={"role": "assistant", "content": "", "tool_calls": []})
    assert failed[0] is None
    after_empty = model_execution_projection(ctx.accumulated_usage)
    # The solve half is preserved. The host's OWN last typed failure is a
    # separate fact on the same projection and does move (I9): a nanny that
    # died on its own lane must be readable without guessing at the leaf.
    assert {k: v for k, v in after_empty.items() if k != "last_llm_error_kind"} == {
        k: v for k, v in observed.items() if k != "last_llm_error_kind"}
    assert after_empty["last_llm_error_kind"] == "provider_incomplete_response"
    assert observed["last_llm_error_kind"] is None
    assert not ctx.accumulated_usage["llm_call_refs"][-1].get("usable_solve_response")
    # Forced/post-task calls use the same call recorder but not ordinary dispatch.
    call_llm_with_retry(Model({"role": "assistant", "content": "wrap up"}, {}),
                       ctx.messages, "forced", [], "high", 1, ctx.drive_logs,
                       ctx.task_id, 3, None, ctx.accumulated_usage, attempt_cap=1)
    # ...and a successful send clears the stale typed error, so the projection
    # returns to the solve half alone.
    assert model_execution_projection(ctx.accumulated_usage) == observed
    refs = collect_trace_refs(ctx.accumulated_usage, {})["llm_call_refs"]
    assert len(refs) == 3
    assert refs[0]["usable_solve_response"] is True
    assert refs[-1]["model"] == "forced" and refs[-1]["reported_model"] is None
    assert refs[-1]["response_ref"]


def test_tool_response_and_local_route_are_usable_without_fabricated_reported_name(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.active_use_local = True
    ctx.accumulated_usage["initial_model_request"] = {"model": "remote", "use_local": False}
    dispatch(ctx, "local", message={"role": "assistant", "content": "",
             "tool_calls": [{"id": "t", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]})
    observed = model_execution_projection(ctx.accumulated_usage)
    assert observed["used_model"] == "local" and observed["used_local"] is True
    assert observed["reported_model"] is None
    assert ctx.accumulated_usage["llm_call_refs"][-1]["resolved_model"] == "local (local)"


def test_stale_call_cannot_be_marked_by_a_response_without_its_own_record(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    dispatch(ctx, "first")
    previous = ctx.accumulated_usage["llm_call_refs"][-1]
    previous.pop("usable_solve_response")
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *a, **k: ({"content": "new"}, 0.0))
    loop._dispatch_round_model(ctx, None, attempt_cap=1)
    assert "usable_solve_response" not in previous


def test_new_unobserved_attempt_and_legacy_are_different():
    assert model_execution_projection({"llm_call_refs": [{"model": "legacy"}]}) is None
    projected = model_execution_projection({"initial_model_request": {"model": "primary", "use_local": False}})
    assert projected["source"] == "not_observed"
    assert projected["used_model"] is None and projected["reported_model"] is None
    projected = model_execution_projection({"llm_call_refs": [{"model": "seen", "llm_call_id": "id", "usable_solve_response": True}]})
    assert projected["requested_model"] is None and projected["requested_use_local"] is None


def test_terminal_store_and_event_preserve_the_same_projection(tmp_path):
    from ouroboros.agent_task_pipeline import emit_task_results
    from ouroboros.post_task_checkpoint import project_replica_task_result_fields
    from ouroboros.outcomes import public_task_result

    ctx = _ctx(tmp_path)
    ctx.accumulated_usage["initial_model_request"] = {"model": "primary", "use_local": False}
    dispatch(ctx, "fallback")
    task = {"id": "task", "type": "task", "chat_id": 1, "text": "Produce a report",
            "_skip_post_task_synthesis": True, "model": "primary"}
    pending = []
    emit_task_results(SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None,
                      pending, task, "answer", ctx.accumulated_usage,
                      {"tool_calls": [], "reasoning_notes": []}, 0.0, tmp_path / "logs")
    stored = load_task_result(tmp_path, "task")
    expected = model_execution_projection(ctx.accumulated_usage)
    assert stored["model_execution"] == expected and stored["model"] == "primary"
    assert public_task_result(stored)["model_execution"] == expected
    assert next(e for e in pending if e["type"] == "send_message")["progress_meta"]["model_execution"] == expected
    assert next(e for e in pending if e["type"] == "task_done")["model_execution"] == expected
    canonical = {"root_phase_checkpoint": {"post_task_synthesis": "completed"},
                 "model_execution": {"used_model": "post-task"}, "prompt_tokens": 900}
    overlay = project_replica_task_result_fields(canonical, stored)
    assert overlay["model_execution"] == expected
    assert overlay["trace_refs"] == stored["trace_refs"]
    assert "prompt_tokens" not in overlay


@pytest.mark.parametrize("new", [0, None, 7])
def test_old_fanout_keys_normalize_on_read_only_and_new_values_win(tmp_path, new):
    write_task_result(tmp_path, "task", "completed", swarm_efficiency={
        "wave_count": 9, "inter_wave_latency_sec_total": 40,
        "fanout_count": new, "fanout_interval_sec_total": new, "kept": "evidence"})
    before = result_signature(tmp_path, "task")
    result = load_task_result(tmp_path, "task", strict=True)["swarm_efficiency"]
    assert result == {"fanout_count": new, "fanout_interval_sec_total": new, "kept": "evidence"}
    assert result_signature(tmp_path, "task") == before
    write_task_result(tmp_path, "task", "completed", note="unrelated")
    assert read_result(tmp_path, "task")["swarm_efficiency"]["wave_count"] == 9


def test_old_only_fanout_keys_keep_unknown_siblings(tmp_path):
    write_task_result(tmp_path, "task", "completed", swarm_efficiency={"wave_count": 6, "inter_wave_latency_sec_total": 4.4, "future": [1]})
    assert load_task_result(tmp_path, "task")["swarm_efficiency"] == {"fanout_count": 6, "fanout_interval_sec_total": 4.4, "future": [1]}


def test_real_loop_records_initial_request_before_the_first_fallback(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_loop_transport_wait import _loop_kwargs

    class FallbackModel(Model):
        def default_model(self):
            return "primary"

        def chat(self, **kwargs):
            if kwargs["model"] == "primary":
                raise RuntimeError("HTTP 401 invalid API key")
            return super().chat(**kwargs)

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "fallback")
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "false")
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda *a, **k: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    noted = []
    monkeypatch.setattr(loop, "_note_low_start", lambda _ctx, _plan, mode, task_id, _queue: noted.append((task_id, mode)))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    kwargs = _loop_kwargs(tmp_path, registry, [], FallbackModel({"role": "assistant", "content": "Fallback result"}, {}))
    kwargs["drive_logs"] = tmp_path / "logs"
    text, usage, trace = loop.run_llm_loop(**kwargs)
    assert text == "Fallback result"
    assert noted == [(kwargs["task_id"], "max")]  # a fresh start is weighed for its Low-start line once
    assert usage["initial_model_request"] == {"model": "primary", "use_local": False}
    assert model_execution_projection(usage)["used_model"] == "fallback"
    assert registry._ctx.active_model == "fallback"
    assert collect_trace_refs(usage, trace)["llm_call_refs"][-1]["usable_solve_response"] is True


def test_owner_wait_source_retains_initial_route_and_marked_calls(tmp_path, monkeypatch):
    from ouroboros.owner_wait import checkpoint_owner_wait, load_owner_wait, resume_native_loop, set_owner_wait
    from ouroboros.task_pacing import CostCeiling
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt = "wait-model", 1
    ctx.active_model, ctx.active_effort = "fallback", "high"
    ctx.active_use_local, ctx.active_context_mode = False, "max"
    ctx._owner_wait_requested = "quiz"
    ctx._cost_ceiling = CostCeiling(state="disabled")
    ctx.context_fit_plan = None
    usage = {"initial_model_request": {"model": "primary", "use_local": False},
             "llm_call_refs": [{"model": "fallback", "use_local": False, "llm_call_id": "solve-call",
                                "usable_solve_response": True}]}
    expected = model_execution_projection(usage)
    write_task_result(tmp_path, ctx.task_id, "running")
    wait = checkpoint_owner_wait(ctx, [{"role": "user", "content": "go"}], {}, usage, 1, [], set())
    set_owner_wait(tmp_path, ctx.task_id, {**wait, "state": "waiting"})
    ctx.owner_wait_resume = {**wait, "restart_transaction_id": "confirmed"}
    ctx.owner_wait_callback = lambda *_: None
    saved = load_owner_wait(ctx)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    restored = {}
    resume_native_loop(registry, saved, [], {}, restored, set())
    assert model_execution_projection(restored) == expected
    assert ctx.active_model == "fallback" and ctx.owner_wait_resume is None


def test_dead_host_lane_renders_its_own_half_without_inventing_the_leaf():
    """I9: the host's typed failure and the leaf's identity are SEPARATE facts.

    A nanny that died on its own Codex lane used to be reported by the reviewer
    role it played, so the owner asked why the leaf's model was broken while
    that leaf was alive. The host half now rides model_execution beside the
    model it was running; when the delegated reconciliation was never persisted
    (it was null on the live row) the custody notice stays silent rather than
    guessing at the leaf.
    """
    from ouroboros.task_finalization import terminal_host_notice_text

    usage = {
        "initial_model_request": {"model": "primary", "use_local": False},
        "llm_call_refs": [{"model": "host-lane-model", "llm_call_id": "call-1",
                           "provider": "claudexor", "usable_solve_response": True}],
        "_last_llm_error_kind": "provider_outcome_unknown",
    }
    projected = model_execution_projection(usage)
    assert projected["used_model"] == "host-lane-model"
    assert projected["provider"] == "claudexor"
    assert projected["last_llm_error_kind"] == "provider_outcome_unknown"
    assert terminal_host_notice_text({"model_execution": projected}) == ""
