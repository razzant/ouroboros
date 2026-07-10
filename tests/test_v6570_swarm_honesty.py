"""v6.57.0 (Phase 1) — swarm/outcome honesty + settings regression tests.

Covers: find_child_tasks scope=direct (no false grandchild absorption), the
policy_denials outcome bucket, verify_and_record refused_out_of_scope, the
recursive cost_usd_with_children rollup, the subagent profile summary, the
protected-artifact glob carve-out, and the EFFORT_SCALE SSOT.
"""
from __future__ import annotations

import pathlib

import pytest

from ouroboros.config import (
    EFFORT_SCALE,
    clamp_effort_to,
    effort_one_step_down,
    effort_rank,
    resolve_effort,
)


# --- 3.1 find_child_tasks scope=direct --------------------------------------

def _write_result(root: pathlib.Path, tid: str, **fields) -> None:
    from ouroboros.task_results import write_task_result

    write_task_result(root, tid, fields.pop("status", "completed"), **fields)


def test_find_child_tasks_scope_direct_excludes_siblings_and_parent(tmp_path):
    """A leaf grandchild must NOT see its parent/sibling as children (the false
    children_unabsorbed incident). scope=direct returns only its OWN children."""
    from ouroboros.task_status import find_child_tasks

    root = tmp_path
    # Tree: root -> builder(b), polisher(p) -> grandchild(g). g is a childless leaf.
    _write_result(root, "b", delegation_role="subagent", parent_task_id="root", root_task_id="root")
    _write_result(root, "p", delegation_role="subagent", parent_task_id="root", root_task_id="root")
    _write_result(root, "g", delegation_role="subagent", parent_task_id="p", root_task_id="root")

    # The grandchild finalizing (exclude_task_id=g, as the real consumers pass):
    # scope="direct" => no children at all.
    direct = find_child_tasks(root, parent_task_id="g", root_task_id="root", exclude_task_id="g", scope="direct")
    assert direct == []

    # Legacy subtree scope would (wrongly, for absorption) surface the parent + sibling.
    subtree = find_child_tasks(root, parent_task_id="g", root_task_id="root", exclude_task_id="g", scope="subtree")
    assert {r["task_id"] for r in subtree} == {"b", "p"}

    # The polisher's direct children are exactly [g].
    p_direct = find_child_tasks(root, parent_task_id="p", root_task_id="root", scope="direct")
    assert {r["task_id"] for r in p_direct} == {"g"}


# --- 1.5d cost_usd_with_children rollup -------------------------------------

def test_compute_cost_with_children_rolls_up_direct_children(tmp_path):
    from ouroboros.task_status import compute_cost_with_children

    root = tmp_path
    _write_result(root, "child1", delegation_role="subagent", parent_task_id="parent",
                  root_task_id="parent", cost_usd=1.5, status="completed")
    # A grandchild already rolled up into child2's own with-children total.
    _write_result(root, "child2", delegation_role="subagent", parent_task_id="parent",
                  root_task_id="parent", cost_usd=2.0, cost_usd_with_children=5.0, status="completed")

    total, partial = compute_cost_with_children(root, "parent", own_cost_usd=0.5)
    # own(0.5) + child1(1.5) + child2 rolled-up(5.0) = 7.0
    assert total == 7.0
    assert partial is False


def test_compute_cost_with_children_marks_partial_for_running_child(tmp_path):
    from ouroboros.task_status import compute_cost_with_children

    root = tmp_path
    _write_result(root, "c1", delegation_role="subagent", parent_task_id="p",
                  root_task_id="p", cost_usd=1.0, status="running")
    total, partial = compute_cost_with_children(root, "p", own_cost_usd=0.25)
    assert total == 1.25
    assert partial is True


# --- 1.3 policy_denials bucket ----------------------------------------------

def test_policy_denial_does_not_degrade_or_headline_tool_failure():
    from ouroboros.outcomes import EXECUTION_OK, derive_loop_outcome

    for status in ("integration_blocked", "workspace_blocked", "light_mode_blocked",
                   "resource_policy_blocked", "protected_blocked"):
        out = derive_loop_outcome(
            "Site is built.",
            {"rounds": 3},
            {"tool_calls": [{"tool": "run_command", "is_error": True, "status": status,
                             "result": f"⚠️ {status.upper()}: blocked"}]},
        )
        ex = out["outcome_axes"]["execution"]
        assert ex["status"] == EXECUTION_OK, status
        assert out["reason_code"] == "final_message", status
        assert ex["policy_denials"][0]["status"] == status, status


def test_genuine_error_still_headlines_tool_failure():
    from ouroboros.outcomes import EXECUTION_DEGRADED, derive_loop_outcome

    out = derive_loop_outcome(
        "done",
        {"rounds": 1},
        {"tool_calls": [{"tool": "run_command", "is_error": True, "status": "error",
                         "result": "⚠️ boom"}]},
    )
    assert out["outcome_axes"]["execution"]["status"] == EXECUTION_DEGRADED
    assert out["reason_code"] == "tool_failure"


# --- EFFORT_SCALE SSOT (1.8) -------------------------------------------------

def test_effort_scale_ordered_and_includes_xhigh_max():
    assert EFFORT_SCALE == ("none", "minimal", "low", "medium", "high", "xhigh", "max")
    assert effort_rank("xhigh") > effort_rank("high")
    assert effort_rank("max") == len(EFFORT_SCALE) - 1
    assert effort_rank("bogus") == -1


def test_clamp_effort_to_ceiling():
    assert clamp_effort_to("xhigh", "high") == "high"   # clamped down
    assert clamp_effort_to("low", "high") == "low"      # already under ceiling
    assert clamp_effort_to("max", "max") == "max"       # equal
    assert clamp_effort_to("high", "") == "high"        # unknown ceiling: pass-through


def test_effort_one_step_down():
    assert effort_one_step_down("max") == "xhigh"
    assert effort_one_step_down("xhigh") == "high"
    assert effort_one_step_down("none") == "none"       # floor


def test_resolve_effort_accepts_xhigh_and_max(monkeypatch):
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "xhigh")
    assert resolve_effort("task") == "xhigh"
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "max")
    assert resolve_effort("task") == "max"


def test_normalize_reasoning_effort_uses_scale():
    from ouroboros.llm import normalize_reasoning_effort

    assert normalize_reasoning_effort("max") == "max"
    assert normalize_reasoning_effort("xhigh") == "xhigh"
    assert normalize_reasoning_effort("bogus") == "medium"


# --- 1.8 learned effort ceiling: clamp + DISCLOSURE (adversarial r1) ----------

@pytest.fixture
def _clean_effort_ceiling_cache():
    from ouroboros.llm import LLMClient

    LLMClient._EFFORT_CEILING_CACHE.clear()
    LLMClient._EFFORT_CEILING_LOADED.clear()
    yield
    LLMClient._EFFORT_CEILING_CACHE.clear()
    LLMClient._EFFORT_CEILING_LOADED.clear()


def test_effort_clamp_records_disclosure(_clean_effort_ceiling_cache):
    """A learned-ceiling clamp is NEVER silent: the client records the requested→
    applied pair, and popping it twice yields it exactly once (per-call semantics)."""
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test")
    LLMClient._EFFORT_CEILING_CACHE["openai/gpt-test"] = "high"
    assert client._clamp_effort_for_model("openai/gpt-test", "xhigh") == "high"
    note = client._pop_effort_clamp_disclosure()
    assert note == {
        "requested": "xhigh", "applied": "high",
        "reason": "learned_ceiling", "model": "openai/gpt-test",
    }
    assert client._pop_effort_clamp_disclosure() is None


def test_effort_clamp_no_disclosure_without_actual_clamp(_clean_effort_ceiling_cache):
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test")
    LLMClient._EFFORT_CEILING_CACHE["openai/gpt-test"] = "high"
    # Under the ceiling: no clamp, no note.
    assert client._clamp_effort_for_model("openai/gpt-test", "medium") == "medium"
    assert client._pop_effort_clamp_disclosure() is None
    # A clamped attempt followed by an unclamped one must NOT leak the stale note
    # (the pending record resets at every payload build).
    client._clamp_effort_for_model("openai/gpt-test", "max")
    client._clamp_effort_for_model("openai/gpt-test", "low")
    assert client._pop_effort_clamp_disclosure() is None


def test_record_effort_ceiling_floors_at_low(_clean_effort_ceiling_cache):
    """Rejections of the LOWEST thinking tiers must not poison the route: learning
    a ceiling below "low" (i.e. minimal→none) would permanently disable thinking
    off one bad request — the drop-param retry already covers unsupported carriers."""
    from ouroboros.llm import LLMClient

    LLMClient._record_effort_ceiling("anthropic/claude-test", "minimal")
    assert "anthropic/claude-test" not in LLMClient._EFFORT_CEILING_CACHE
    LLMClient._record_effort_ceiling("anthropic/claude-test", "low")
    assert "anthropic/claude-test" not in LLMClient._EFFORT_CEILING_CACHE
    # A genuine high-tier rejection still learns (and a LOWER ceiling wins).
    LLMClient._record_effort_ceiling("anthropic/claude-test", "xhigh")
    assert LLMClient._EFFORT_CEILING_CACHE["anthropic/claude-test"] == "high"
    LLMClient._record_effort_ceiling("anthropic/claude-test", "high")
    assert LLMClient._EFFORT_CEILING_CACHE["anthropic/claude-test"] == "medium"
    LLMClient._record_effort_ceiling("anthropic/claude-test", "xhigh")  # never regain
    assert LLMClient._EFFORT_CEILING_CACHE["anthropic/claude-test"] == "medium"


def test_generic_param_rejection_learns_no_effort_ceiling(_clean_effort_ceiling_cache):
    """A temperature-only 400 must not teach a phantom effort ceiling."""
    from ouroboros.llm import LLMClient

    payload = {"model": "m", "temperature": 0.2, "reasoning_effort": "xhigh"}
    retry = LLMClient._retry_without_optional_sampling(
        payload, "openai/gpt-test", RuntimeError("400: temperature is not supported"),
    )
    assert retry is not None and "temperature" not in retry
    assert "openai/gpt-test" not in LLMClient._EFFORT_CEILING_CACHE
    # An effort-implicating rejection DOES learn.
    retry2 = LLMClient._retry_without_optional_sampling(
        payload, "openai/gpt-test", RuntimeError("400: reasoning_effort value not supported"),
    )
    assert retry2 is not None
    assert LLMClient._EFFORT_CEILING_CACHE.get("openai/gpt-test") == "high"


def test_anthropic_direct_effort_rejection_degrades_gracefully(
    _clean_effort_ceiling_cache, monkeypatch
):
    """Scope r7 advisory asked for proof: an older direct-Anthropic endpoint that
    400s the thinking/output_config carriers routes through the SAME
    _retry_without_optional_sampling as other lanes — the call degrades to
    'no forced thinking' instead of hard-failing."""
    import json as _json

    import requests as _requests

    from ouroboros.llm import LLMClient

    calls: list = []

    class _Resp:
        def __init__(self, status, body):
            self.status_code = status
            self._body = body
            self.text = _json.dumps(body)
            self.reason = "Bad Request" if status >= 400 else "OK"
            self.url = "https://api.anthropic.test/messages"

        def json(self):
            return self._body

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        if "output_config" in json:
            return _Resp(400, {"error": {"message": "output_config: Extra inputs are not permitted"}})
        return _Resp(200, {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 3, "output_tokens": 1},
            "stop_reason": "end_turn",
        })

    monkeypatch.setattr(_requests, "post", fake_post)
    client = LLMClient(api_key="test")
    LLMClient._REJECTED_PARAMS_CACHE.clear()
    target = {"resolved_model": "claude-old", "usage_model": "anthropic/claude-old",
              "base_url": "https://api.anthropic.test", "api_key": "k"}
    msg, _ = client._chat_anthropic(target, [{"role": "user", "content": "hi"}], None, "high", 128, "auto")
    assert msg["content"] == "ok"
    assert len(calls) == 2  # 400 with the carrier, then the degraded retry
    assert "output_config" in calls[0] and "output_config" not in calls[1]
    assert "thinking" not in calls[1]
    LLMClient._REJECTED_PARAMS_CACHE.clear()


def test_openrouter_nested_reasoning_rejection_retries_and_learns(_clean_effort_ceiling_cache):
    """Triad r6: the OpenRouter lane carries effort NESTED as
    extra_body.reasoning.effort — an effort rejection there must (a) retry with
    ONLY the nested carrier removed (provider routing survives), (b) learn the
    ceiling, and (c) the rejected-params cache must strip the nested slot on the
    NEXT payload build."""
    from ouroboros.llm import LLMClient

    payload = {
        "model": "vendor/model-x",
        "messages": [],
        "extra_body": {
            "reasoning": {"effort": "xhigh", "exclude": False},
            "provider": {"require_parameters": True},
        },
    }
    retry = LLMClient._retry_without_optional_sampling(
        payload, "vendor/model-x", RuntimeError("400: reasoning effort not supported on this endpoint"),
    )
    assert retry is not None
    assert "reasoning" not in retry["extra_body"]
    assert retry["extra_body"]["provider"] == {"require_parameters": True}
    assert LLMClient._EFFORT_CEILING_CACHE.get("vendor/model-x") == "high"
    # The learned rejection strips the nested carrier on subsequent builds.
    fresh = {"extra_body": {"reasoning": {"effort": "high"}, "provider": {"order": ["a"]}}}
    LLMClient._apply_rejected_param_cache(fresh, "vendor/model-x")
    assert "reasoning" not in fresh["extra_body"]
    assert fresh["extra_body"]["provider"] == {"order": ["a"]}
    # A NON-effort error with only the nested carrier present does not retry.
    payload2 = {"model": "m", "extra_body": {"reasoning": {"effort": "low"}}}
    assert LLMClient._retry_without_optional_sampling(
        payload2, "vendor/model-y", RuntimeError("400: bad request shape"),
    ) is None


def test_anthropic_direct_effort_mapping_and_clamp_disclosure(
    _clean_effort_ceiling_cache, monkeypatch
):
    """Anthropic-direct: (a) our `minimal` maps to the provider floor `low` (its
    documented set has no minimal — an out-of-range value would 400 and poison the
    learned ceiling); (b) a learned-ceiling clamp lands in the RETURNED usage as
    reasoning_effort_clamped (the durable llm_usage disclosure, adversarial r1)."""
    import requests as _requests

    from ouroboros.llm import LLMClient

    captured: list = []

    class _Resp:
        status_code = 200
        def json(self):
            return {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 3, "output_tokens": 1},
                "stop_reason": "end_turn",
            }

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.append(json)
        return _Resp()

    monkeypatch.setattr(_requests, "post", fake_post)
    client = LLMClient(api_key="test")
    target = {"resolved_model": "claude-test-5", "usage_model": "anthropic/claude-test-5",
              "base_url": "https://api.anthropic.test", "api_key": "k"}
    msgs = [{"role": "user", "content": "hi"}]

    _, usage = client._chat_anthropic(target, msgs, None, "minimal", 128, "auto")
    assert captured[-1]["output_config"] == {"effort": "low"}
    assert "reasoning_effort_clamped" not in usage

    LLMClient._EFFORT_CEILING_CACHE["anthropic/claude-test-5"] = "high"
    _, usage2 = client._chat_anthropic(target, msgs, None, "xhigh", 128, "auto")
    assert captured[-1]["output_config"] == {"effort": "high"}
    assert usage2["reasoning_effort_clamped"] == {
        "requested": "xhigh", "applied": "high",
        "reason": "learned_ceiling", "model": "anthropic/claude-test-5",
    }


def test_remote_lane_usage_carries_clamp_disclosure(_clean_effort_ceiling_cache):
    """The OpenRouter/OpenAI-compatible lane's usage normalizer merges the pending
    clamp note recorded at payload build (the r1 fix initially landed the pop in the
    GigaChat normalizer — a lane that never clamps; pinned here on the REAL lane)."""
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test")
    LLMClient._EFFORT_CEILING_CACHE["openai/gpt-test"] = "medium"
    assert client._clamp_effort_for_model("openai/gpt-test", "high") == "medium"
    _, usage = client._normalize_remote_response(
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
         "usage": {"prompt_tokens": 2, "completion_tokens": 1}},
        {"provider": "openai", "resolved_model": "gpt-test", "usage_model": "openai/gpt-test"},
        skip_cost_fetch=True,
    )
    assert usage["reasoning_effort_clamped"]["requested"] == "high"
    assert usage["reasoning_effort_clamped"]["applied"] == "medium"
    # Popped exactly once — the next normalize carries nothing.
    _, usage_next = client._normalize_remote_response(
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
         "usage": {"prompt_tokens": 2, "completion_tokens": 1}},
        {"provider": "openai", "resolved_model": "gpt-test", "usage_model": "openai/gpt-test"},
        skip_cost_fetch=True,
    )
    assert "reasoning_effort_clamped" not in usage_next


# --- 1.5 subagent profile summary -------------------------------------------

def test_summarize_subagent_profile_readonly_vs_acting():
    from ouroboros.tool_access import (
        predicted_subagent_profile,
        summarize_subagent_profile,
    )

    ro = predicted_subagent_profile(write_surface="")
    assert ro == "local_readonly_subagent"
    ro_summary = summarize_subagent_profile(ro, effective_lane="light")
    assert "shell=no" in ro_summary
    assert "read-only" in ro_summary
    assert "model_lane=light" in ro_summary

    acting = predicted_subagent_profile(write_surface="self_worktree")
    assert acting == "acting_subagent"
    acting_summary = summarize_subagent_profile(acting, effective_lane="heavy")
    assert "shell=yes" in acting_summary
    assert "active_workspace" in acting_summary
