"""The discriminated route of running work (``route_choice``): following its role, or the task's own.

Provider-free: the loop's round boundary, ``switch_model``, the continuation serializer/restorer and the
nanny's inherited cognition run on real contexts; only the context-fit rebind is stubbed. r2_astra's
"most likely one-PR failure": Main=A, Light=B; an explicit switch to B; save Main=B; checkpoint and
resume; save Main=C: the explicit choice stays B for the parent and its inherited nanny, the following
case moves to C, a configured API child keeps its row, and exactly one notice names each actual change.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import ouroboros.loop as loop_mod
from ouroboros import config, owner_wait, route_choice
from ouroboros.loop_model_call import _apply_round_route_overrides
from ouroboros.tools.control_runtime import _switch_model
from ouroboros.tools.registry import ToolRegistry
from tests.test_task_settings_snapshot import isolated_runtime, save_owner  # noqa: F401 -- the shared isolated, sealed root

A, B, C, D = "openai::a", "openai::b", "openai::c", "openai::d"


def _document(main, light=B, **extra):
    return {"OUROBOROS_MODEL": main, "OUROBOROS_MODEL_LIGHT": light, "OUROBOROS_MODEL_FALLBACKS": "",
            "OPENAI_API_KEY": "key", "OUROBOROS_MODEL_ACCOUNTS": json.dumps({"main": ""}), **extra}


@pytest.fixture
def stubbed_rebind(monkeypatch):
    monkeypatch.setattr(loop_mod, "_rebind_context_fit_plan", lambda plan, *_a, **_kw: (plan, "max"))


LLM = SimpleNamespace(default_model=lambda: config.runtime_setting("OUROBOROS_MODEL"))


def _turn(tmp_path, metadata=None):
    """A fresh turn as the loop opens it: real ToolContext, the turn's primary and route choice recorded."""
    tools = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = tools._ctx
    ctx.task_id, ctx.task_attempt, ctx.task_metadata = "t-1", 1, dict(metadata or {})
    route_choice.bind_task_route(ctx, ctx.task_metadata, presence=False)
    loop_mod._reset_turn_state(ctx)
    model, _effort, use_local, *_rest = loop_mod._initial_round_route(ctx, LLM, "medium")
    ctx.active_model, ctx.active_use_local, ctx.active_effort = model, use_local, "medium"
    return tools, ctx


def _round(tools, ctx, messages):
    """One round boundary; returns ``(acting binding, the settings notice this round carries or "")``."""
    model, use_local, _effort, _plan, _mode = _apply_round_route_overrides(
        ctx, tools, messages, (ctx.active_model, ctx.active_use_local, ctx.active_effort), "plan", "max", "max", [], llm=LLM)
    ctx.active_model, ctx.active_use_local = model, use_local
    notice, ctx._route_facts_pending = str(getattr(ctx, "_route_facts_pending", "") or ""), ""  # consumed by the call
    return (model, use_local), notice


def _resume(tmp_path, ctx, messages, metadata=None):
    """Checkpoint through the one serializer, JSON round-trip, restore into a fresh turn."""
    state = json.loads(json.dumps(owner_wait.continuation_state(ctx, messages, {}, {}, 3, [], set())))
    tools, fresh = _turn(tmp_path, metadata)
    owner_wait.restore_continuation_state(tools, state, messages, {}, {}, set())
    return tools, fresh, state


def _nanny(parent_ctx, leaf_id="session-leaf"):
    """A nanny scheduled by ``parent_ctx`` for an agent-session leaf: its frozen cognition and task metadata."""
    from ouroboros.supervisor_route import _inherited_cognition

    cognitive = _inherited_cognition(parent_ctx, config.load_settings())
    metadata = {"delegation_role": "subagent", "model": cognitive["model"], "use_local_model": cognitive["use_local_model"],
                "configured_subagent": {"selected_subagent_id": leaf_id, "route": {"kind": "agent_session", "target_id": "claude"}},
                "parent_cognitive_route": cognitive}
    return cognitive, metadata


def test_explicit_choices_stay_while_following_work_moves_with_settings(tmp_path, stubbed_rebind):
    save_owner(_document(A))
    messages = [{"role": "user", "content": "go"}]
    tools, ctx = _turn(tmp_path)
    assert (ctx.active_model, ctx.route_choice, ctx.primary_route["model"]) == (A, route_choice.following("main"), A)
    assert _round(tools, ctx, messages) == ((A, False), "")

    assert _switch_model(ctx, model=B).startswith("OK:")  # the mind's own departure to Light's model
    assert _round(tools, ctx, messages) == ((B, False), "") and ctx.route_choice == route_choice.explicit("switch_model")
    save_owner(_document(B))  # Main becomes the model the mind chose: still the mind's choice
    assert _round(tools, ctx, messages) == ((B, False), "")

    tools2, ctx2, state = _resume(tmp_path, ctx, messages)
    assert state["route"]["route_choice"] == route_choice.explicit("switch_model") and ctx2.active_model == B
    save_owner(_document(C))
    binding, notice = _round(tools2, ctx2, messages)
    assert binding == (B, False) and notice == "" and ctx2.route_choice == route_choice.explicit("switch_model")
    assert ctx2.primary_route["model"] == C  # the primary is a reference to Main, read at the boundary

    nanny_cognition, nanny_metadata = _nanny(ctx2)
    assert nanny_cognition["chosen_by"] == "parent_switch" and nanny_cognition["model"] == B
    _tools, nanny = _turn(tmp_path, nanny_metadata)
    assert nanny.task_model_override == B and nanny.route_choice == route_choice.explicit("switch_model")
    save_owner(_document(D))
    assert nanny.active_model == B and _round(_tools, nanny, messages) == ((B, False), "")

    assert "primary route" in _switch_model(ctx2, primary="return")  # back to the CURRENT Main
    assert ctx2.route_choice == route_choice.following("main")
    assert _round(tools2, ctx2, messages) == ((D, False), "")  # the mind's own act names its reason: no settings notice
    save_owner(_document(C))
    binding, notice = _round(tools2, ctx2, messages)
    assert binding == (C, False) and notice.startswith("[ROUTE FACTS] Main changed in Settings: openai::d → openai::c")


def test_following_work_and_its_nanny_move_once_per_change_while_an_api_child_keeps_its_row(tmp_path, stubbed_rebind):
    save_owner(_document(A))
    messages = [{"role": "user", "content": "go"}]
    tools, ctx = _turn(tmp_path)
    save_owner(_document(C))
    binding, notice = _round(tools, ctx, messages)
    assert binding == (C, False) and "Main changed in Settings: openai::a → openai::c" in notice
    assert ctx.primary_route["model"] == C
    assert _round(tools, ctx, messages) == ((C, False), "")  # one notice per actual change
    handovers = ctx._accumulated_usage.get("authoring_handovers") if isinstance(getattr(ctx, "_accumulated_usage", None), dict) else None
    assert handovers is None or [row["reason"] for row in handovers] == ["settings_change"]

    nanny_cognition, nanny_metadata = _nanny(ctx)
    assert (nanny_cognition["chosen_by"], nanny_cognition["follows_role"]) == ("inherited", "main")
    nanny_tools, nanny = _turn(tmp_path, nanny_metadata)
    assert nanny.task_model_override is None and nanny.route_choice == route_choice.following("main")
    assert nanny.active_model == C and nanny.primary_route["role"] == "subagent:session-leaf"
    save_owner(_document(D, OUROBOROS_MODEL_ACCOUNTS=json.dumps({"main": "pinned-account"})))
    binding, notice = _round(nanny_tools, nanny, messages)
    assert binding == (D, False) and "Main changed in Settings" in notice
    from ouroboros.model_slots import task_model_binding
    assert task_model_binding({"task_metadata": nanny.task_metadata}) == ("subagent:session-leaf", "pinned-account")

    tools3, ctx3, _state = _resume(tmp_path, ctx, messages)  # a cold resume rejoins the current Main with one notice
    assert ctx3.active_model == C and ctx3.route_choice == route_choice.following("main")
    binding, notice = _round(tools3, ctx3, messages)
    assert binding == (D, False) and "openai::c → openai::d" in notice

    api_metadata = {"delegation_role": "subagent", "model": "openai::row", "use_local_model": False,
                    "configured_subagent": {"selected_subagent_id": "row", "route": {"kind": "api_model", "target_id": "openai::row"}}}
    api_tools, api_child = _turn(tmp_path, api_metadata)
    assert api_child.route_choice == route_choice.explicit("configured_row") and api_child.active_model == "openai::row"
    save_owner(_document(A))
    assert _round(api_tools, api_child, messages) == (("openai::row", False), "")


def test_a_legacy_checkpoint_is_classified_by_its_explicit_carriers(tmp_path):
    def classify(route=None, *, context_model_role="", override=None, metadata=None, waiter_overrides=None):
        ctx = SimpleNamespace(task_metadata=metadata or {}, primary_route=(route or {}).get("primary_route"),
                              model_wait_context=SimpleNamespace(overrides=waiter_overrides or {}),
                              task_model_override=override)
        return route_choice.classify_legacy(ctx, {"route": route or {}, "context_model_role": context_model_role})

    primary = {"model": A, "use_local": False, "role": "main"}
    assert classify() == route_choice.following("main")  # pre-primary_route checkpoint, no carrier
    assert classify({"primary_route": primary, "active_model": A, "active_use_local": False}) == route_choice.following("main")
    assert classify({"primary_route": primary, "active_model": B, "active_use_local": False}) == route_choice.explicit("switch_model")
    assert classify({"primary_route": primary, "active_model": A, "active_use_local": True}) == route_choice.explicit("switch_model")
    assert classify({"primary_route": primary, "active_model": A, "active_use_local": False,
                     "active_model_override": B}) == route_choice.explicit("switch_model")
    assert classify({"primary_route": primary, "active_model": A, "active_use_local": False},
                    context_model_role="fallback:0") == route_choice.explicit("fallback")
    assert classify({"primary_route": primary, "active_model": A, "active_use_local": False},
                    waiter_overrides={"main": {"model_account_override": "acct"}}) == route_choice.explicit("wait_card")
    assert classify({"primary_route": primary, "active_model": A, "active_use_local": False},
                    override="openai::row") == route_choice.explicit("task_argument")
    wake = {"model": "consc/model", "use_local": False, "role": "consciousness"}
    assert classify({"primary_route": wake, "active_model": "consc/model", "active_use_local": False}) == route_choice.following("consciousness")
    saved = {"route": {"route_choice": route_choice.explicit("fallback")}}
    assert route_choice.restored_choice(SimpleNamespace(), saved) == route_choice.explicit("fallback")


def test_a_consciousness_wake_resolves_its_slot_per_call_instead_of_copying_it(tmp_path, stubbed_rebind):
    save_owner(_document(A, OUROBOROS_MODEL_CONSCIOUSNESS="openai::consc"))
    messages = [{"role": "user", "content": "wake"}]
    tools, ctx = _turn(tmp_path, {"model_role": "consciousness"})
    assert ctx.task_model_override is None and ctx.route_choice == route_choice.following("consciousness")
    assert (ctx.active_model, ctx.primary_route) == ("openai::consc", {"model": "openai::consc", "use_local": False, "role": "consciousness"})
    save_owner(_document(A, OUROBOROS_MODEL_CONSCIOUSNESS="openai::deeper"))
    binding, notice = _round(tools, ctx, messages)
    assert binding == ("openai::deeper", False) and notice.startswith("[ROUTE FACTS] Consciousness changed in Settings")
    save_owner(_document(C, OUROBOROS_MODEL_CONSCIOUSNESS=""))  # an empty slot is Main (В25=B), read when the call opens
    assert _round(tools, ctx, messages)[0] == (C, False)
    save_owner(_document(C, OUROBOROS_MODEL_CONSCIOUSNESS="", USE_LOCAL_CONSCIOUSNESS=True))
    assert _round(tools, ctx, messages)[0] == (C, True)  # the slot's own local flag still decides
    assert ctx.primary_route == {"model": C, "use_local": True, "role": "consciousness"}
