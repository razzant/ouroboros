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


def test_genuine_error_is_recorded_without_headlining_a_tool_failure():
    from ouroboros.outcomes import EXECUTION_OK, derive_loop_outcome

    out = derive_loop_outcome(
        "done",
        {"rounds": 1},
        {"tool_calls": [{"tool": "run_command", "is_error": True, "status": "error",
                         "result": "⚠️ boom"}]},
    )
    assert out["outcome_axes"]["execution"]["status"] == EXECUTION_OK
    assert out["reason_code"] == "final_message"
    assert out["outcome_axes"]["execution"]["unresolved_tool_errors"][0]["status"] == "error"
    assert out["outcome_axes"]["objective"]["warning"] == "residual_tool_errors_without_review"


# --- EFFORT_SCALE SSOT (1.8) -------------------------------------------------

def test_effort_scale_ordered_and_top_tier_is_last():
    assert EFFORT_SCALE == ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
    assert effort_rank("xhigh") > effort_rank("high")
    assert effort_rank("ultra") > effort_rank("max")
    assert effort_rank("ultra") == len(EFFORT_SCALE) - 1
    assert effort_rank("bogus") == -1


def test_clamp_effort_to_ceiling():
    assert clamp_effort_to("xhigh", "high") == "high"   # clamped down
    assert clamp_effort_to("low", "high") == "low"      # already under ceiling
    assert clamp_effort_to("max", "max") == "max"       # equal
    assert clamp_effort_to("high", "") == "high"        # unknown ceiling: pass-through


def test_effort_one_step_down():
    assert effort_one_step_down("ultra") == "max"
    assert effort_one_step_down("max") == "xhigh"
    assert effort_one_step_down("xhigh") == "high"
    assert effort_one_step_down("none") == "none"       # floor


def test_resolve_effort_accepts_the_top_tiers(monkeypatch):
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "xhigh")
    assert resolve_effort("task") == "xhigh"
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "max")
    assert resolve_effort("task") == "max"
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "ultra")
    assert resolve_effort("task") == "ultra"


def test_normalize_reasoning_effort_uses_scale():
    from ouroboros.llm import normalize_reasoning_effort

    assert normalize_reasoning_effort("max") == "max"
    assert normalize_reasoning_effort("xhigh") == "xhigh"
    assert normalize_reasoning_effort("ultra") == "ultra"
    assert normalize_reasoning_effort("bogus") == "medium"


# --- 1.8 effort tier notes: clamp DISCLOSURE (adversarial r1) ----------------

@pytest.fixture
def _clean_effort_clamp_note():
    """The per-call tier note lives in the test's context: a payload build that staged one
    without a send (the DeepSeek case of test_provider_key_test) must not reach these calls."""
    from ouroboros.llm_capability_policy import _EFFORT_CLAMP_CVAR

    _EFFORT_CLAMP_CVAR.set(None)
    yield
    _EFFORT_CLAMP_CVAR.set(None)


def test_anthropic_direct_effort_rejection_degrades_gracefully(
    _clean_effort_clamp_note, monkeypatch
):
    """Scope r7 advisory asked for proof: an older direct-Anthropic endpoint that
    400s the thinking/output_config carriers routes through the SAME
    request-wire recovery as other lanes — the call degrades to
    'no forced thinking' instead of hard-failing."""
    import json as _json

    import requests as _requests

    from ouroboros.llm import LLMClient

    monkeypatch.setenv("TOTAL_BUDGET", "0")

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
    target = {"provider": "anthropic", "resolved_model": "claude-old", "usage_model": "anthropic/claude-old",
              "base_url": "https://api.anthropic.test", "api_key": "k"}
    msg, _ = client._chat_anthropic(target, [{"role": "user", "content": "hi"}], None, "high", 128, "auto")
    assert msg["content"] == "ok"
    assert len(calls) == 2  # 400 with the carrier, then the degraded retry
    assert "output_config" in calls[0] and "output_config" not in calls[1]
    assert "thinking" not in calls[1]


def test_anthropic_direct_effort_mapping_and_clamp_disclosure(
    _clean_effort_clamp_note, monkeypatch, tmp_path
):
    """Anthropic maps minimal to low; a stored legacy ceiling never clamps the sent effort."""
    import requests as _requests

    from ouroboros.llm import LLMClient
    from tests._shared import write_legacy_capability_rows

    monkeypatch.setenv("TOTAL_BUDGET", "0")

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
    target = {"provider": "anthropic", "resolved_model": "claude-test-5", "usage_model": "anthropic/claude-test-5",
              "base_url": "https://api.anthropic.test", "api_key": "k"}
    msgs = [{"role": "user", "content": "hi"}]

    _, usage = client._chat_anthropic(target, msgs, None, "minimal", 128, "auto")
    assert captured[-1]["output_config"] == {"effort": "low"}
    assert "reasoning_effort_clamped" not in usage

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    write_legacy_capability_rows(tmp_path, effort_ceilings={"anthropic/claude-test-5": "high"})
    _, usage2 = client._chat_anthropic(target, msgs, None, "xhigh", 128, "auto")
    assert captured[-1]["output_config"] == {"effort": "xhigh"}
    assert "reasoning_effort_clamped" not in usage2


def test_remote_lane_usage_carries_clamp_disclosure(_clean_effort_clamp_note):
    """The OpenRouter/OpenAI-compatible lane's usage normalizer merges the pending
    clamp note recorded at payload build (the r1 fix initially landed the pop in the
    GigaChat normalizer — a lane that never clamps; pinned here on the REAL lane)."""
    from ouroboros.llm import LLMClient
    from ouroboros.llm_capability_policy import _EFFORT_CLAMP_CVAR

    client = LLMClient(api_key="test")
    # Staged the way the DeepSeek/Z.ai payload builders stage a tier note.
    _EFFORT_CLAMP_CVAR.set({"requested": "high", "applied": "medium",
                            "reason": "provider_wire_mapping", "model": "openai/gpt-test"})
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
