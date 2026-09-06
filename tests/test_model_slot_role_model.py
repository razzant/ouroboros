"""F1 (v6.39): model-slot role-model + 429-aware fallback chain + cooldown.

Covers the empty->Main accessors, the new comma-separated fallback chain
(dedup / drop-active / benchmark no-op / legacy-singular env), the stored-key
rename migration, the process-local cooldown, and the subagent lane resolver
(mutating-child -> heavy, read-only -> light, explicit honored, depth-cap note).
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

import ouroboros.config as config
from ouroboros import fallback_cooldown as fcd
from ouroboros import subagents


@pytest.fixture(autouse=True)
def _owned_gateway_uses_each_test_transport(monkeypatch):
    from ouroboros import claudexor_daemon

    class _Gateway:
        def close(self):
            pass

    monkeypatch.setattr(
        claudexor_daemon,
        "ensure_owned_gateway",
        _Gateway,
    )


# ---------------------------------------------------------------- accessors

def test_heavy_and_light_empty_fall_back_to_main(monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main-x")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "")
    assert config.get_heavy_model() == "provider::main-x"
    assert config.get_light_model() == "provider::main-x"


def test_heavy_and_light_explicit_values_are_honored(monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main-x")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    assert config.get_heavy_model() == "provider::strong"
    assert config.get_light_model() == "provider::cheap"


# ----------------------------------------------------------- fallback chain

def test_fallback_chain_dedups_and_drops_active(monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "a, b , a, c")
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACK", raising=False)
    assert config.get_fallback_models("b") == ["a", "c"]
    # No active model -> full deduped chain in order.
    assert config.get_fallback_models("") == ["a", "b", "c"]


def test_fallback_chain_benchmark_dedupes_to_no_op(monkeypatch):
    # Benchmark sets every slot to one model; the active model is dropped, so the
    # chain collapses to empty -> no cross-model fallback happens.
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "same::model")
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACK", raising=False)
    assert config.get_fallback_models("same::model") == []


def test_fallback_chain_reads_legacy_singular_env(monkeypatch):
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACKS", raising=False)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACK", "legacy::single")
    assert config.get_fallback_models("primary") == ["legacy::single"]


def test_fallback_chain_empty_means_no_fallback(monkeypatch):
    # An explicitly empty/unset Fallbacks slot must NOT silently fall back to the shipped
    # Anthropic default (which would cross an OpenAI-compatible/local owner into an
    # unconfigured provider). The default reaches a default install via the env instead.
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACKS", raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACK", raising=False)
    assert config.get_fallback_models("primary") == []


def test_advisory_fallback_model_uses_main_when_light_empty(monkeypatch):
    from ouroboros.tools.claude_advisory_review import _resolve_fallback_model
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main-x")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "")
    # Empty Light must resolve to Main, never "" (which would call chat with no model id).
    assert _resolve_fallback_model() == "provider::main-x"


def test_parse_fallback_chain_ssot(monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "a, b , a")
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACK", raising=False)
    # Raw chain: parsed, whitespace-trimmed, NO dedup, NO active-drop (those belong to
    # get_fallback_models on top).
    assert config.parse_fallback_chain() == ["a", "b", "a"]
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACKS", raising=False)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACK", "legacy")
    assert config.parse_fallback_chain() == ["legacy"]


def test_infer_model_category_recognizes_chain_link(monkeypatch):
    from ouroboros.pricing import infer_model_category
    monkeypatch.setenv("OUROBOROS_MODEL", "main/x")
    monkeypatch.delenv("OUROBOROS_MODEL_HEAVY", raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_LIGHT", raising=False)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "fb/one, fb/two")
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACK", raising=False)
    # A model that is a LINK of the chain is categorized "fallback", not "other".
    assert infer_model_category("fb/two") == "fallback"
    assert infer_model_category("main/x") == "main"
    assert infer_model_category("unrelated/z") == "other"


# -------------------------------------------------------- stored migration

def test_stored_slot_keys_migrate_on_load(monkeypatch, tmp_path):
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({
        "OUROBOROS_MODEL": "provider::main",
        "OUROBOROS_MODEL_CODE": "provider::legacy-code",
        "USE_LOCAL_CODE": True,
        "OUROBOROS_MODEL_FALLBACK": "provider::legacy-fb",
    }), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", pathlib.Path(settings_file))
    for key in ("OUROBOROS_MODEL_HEAVY", "USE_LOCAL_HEAVY", "OUROBOROS_MODEL_FALLBACKS",
                "OUROBOROS_MODEL_CODE", "USE_LOCAL_CODE", "OUROBOROS_MODEL_FALLBACK"):
        monkeypatch.delenv(key, raising=False)

    loaded = config.load_settings()

    assert loaded.get("OUROBOROS_MODEL_HEAVY") == "provider::legacy-code"
    assert loaded.get("USE_LOCAL_HEAVY") is True
    assert loaded.get("OUROBOROS_MODEL_FALLBACKS") == "provider::legacy-fb"
    # Legacy keys are dropped, not left to linger.
    assert "OUROBOROS_MODEL_CODE" not in loaded
    assert "USE_LOCAL_CODE" not in loaded
    assert "OUROBOROS_MODEL_FALLBACK" not in loaded


def test_migrate_legacy_slot_keys_ssot():
    # The shared SSOT helper preserves a stored value, drops the legacy key, and never
    # clobbers an already-set new key.
    s = {"OUROBOROS_MODEL_CODE": "x", "USE_LOCAL_CODE": True, "OUROBOROS_MODEL_FALLBACK": "y"}
    config.migrate_legacy_slot_keys(s)
    assert s == {"OUROBOROS_MODEL_HEAVY": "x", "USE_LOCAL_HEAVY": True, "OUROBOROS_MODEL_FALLBACKS": "y"}
    # An already-set new key wins; the legacy key is still dropped.
    s2 = {"OUROBOROS_MODEL_CODE": "old", "OUROBOROS_MODEL_HEAVY": "new"}
    config.migrate_legacy_slot_keys(s2)
    assert s2 == {"OUROBOROS_MODEL_HEAVY": "new"}


def test_colab_settings_migrate_legacy_drive_keys():
    # A Colab re-run with legacy Drive settings.json must keep the owner's prior
    # code/heavy + fallback customizations (not silently drop them).
    from ouroboros.colab_bootstrap import build_colab_settings
    existing = {
        "OUROBOROS_MODEL": "openai::gpt-5.5",
        "OUROBOROS_MODEL_CODE": "openai::gpt-5.5-custom-heavy",
        "OUROBOROS_MODEL_FALLBACK": "openai::gpt-5.5-mini",
        "OPENAI_API_KEY": "sk-openai-existing",
    }
    out = build_colab_settings({}, models=None, existing=existing)
    assert out.get("OUROBOROS_MODEL_HEAVY") == "openai::gpt-5.5-custom-heavy"
    assert out.get("OUROBOROS_MODEL_FALLBACKS") == "openai::gpt-5.5-mini"
    assert "OUROBOROS_MODEL_CODE" not in out
    assert "OUROBOROS_MODEL_FALLBACK" not in out


# ---------------------------------------------------------------- cooldown

def test_cooldown_marks_and_heals(monkeypatch):
    fcd.reset_for_tests()
    monkeypatch.delenv("OUROBOROS_FALLBACK_COOLDOWN_ENABLED", raising=False)
    monkeypatch.setenv("OUROBOROS_FALLBACK_COOLDOWN_SEC", "120")
    assert fcd.is_cooling_down("m1") is False
    fcd.mark_cooldown("m1")
    assert fcd.is_cooling_down("m1") is True
    # A zero-length window heals immediately on the next read (passive heal).
    monkeypatch.setenv("OUROBOROS_FALLBACK_COOLDOWN_SEC", "0")
    fcd.mark_cooldown("m2")
    assert fcd.is_cooling_down("m2") is False


def test_cooldown_disabled_is_noop(monkeypatch):
    fcd.reset_for_tests()
    monkeypatch.setenv("OUROBOROS_FALLBACK_COOLDOWN_ENABLED", "false")
    fcd.mark_cooldown("m1")
    assert fcd.is_cooling_down("m1") is False


def test_cooldown_local_and_remote_are_distinct(monkeypatch):
    fcd.reset_for_tests()
    monkeypatch.delenv("OUROBOROS_FALLBACK_COOLDOWN_ENABLED", raising=False)
    monkeypatch.setenv("OUROBOROS_FALLBACK_COOLDOWN_SEC", "120")
    fcd.mark_cooldown("m1", use_local=True)
    assert fcd.is_cooling_down("m1", use_local=True) is True
    assert fcd.is_cooling_down("m1", use_local=False) is False


def test_attempts_per_model_is_bounded(monkeypatch):
    monkeypatch.setenv("OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL", "9")
    assert fcd.attempts_per_model() == 2
    monkeypatch.setenv("OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL", "0")
    assert fcd.attempts_per_model() == 1
    monkeypatch.setenv("OUROBOROS_FALLBACK_ATTEMPTS_PER_MODEL", "nonsense")
    assert fcd.attempts_per_model() == 1


# ------------------------------------------------------------ lane resolver

def test_omitted_lane_inherits_the_parents_lane(monkeypatch):
    """An omitted lane INHERITS the parent's effective lane (v6.87.26).

    It used to collapse to `light` whatever the parent was running, so a Heavy
    parent handing a child a slice of its own job got a Light child and no signal.
    Cheap work is still Light — but the parent has to SAY so, which is the whole
    point: the declaration is visible instead of implied by silence. The child's
    write authority still does not enter the choice (the v6.87.7 decoupling)."""
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")

    res = subagents.resolve_subagent_lane("auto", parent_lane="heavy")
    assert res.effective_lane == "heavy"
    assert res.model == "provider::strong"

    # A parent that judges the work cheap names it, and is obeyed.
    named = subagents.resolve_subagent_lane("light", parent_lane="heavy")
    assert named.effective_lane == "light"
    assert named.model == "provider::cheap"

    # No lane on record is the root agent, and the root runs Main.
    root_child = subagents.resolve_subagent_lane("auto")
    assert root_child.effective_lane == "main"
    assert root_child.model == "provider::main"


def test_resolver_takes_no_authority_argument():
    """The resolver must not regrow an authority input. If a future change adds one,
    this fails and the reviewer sees the coupling coming back."""
    import inspect

    params = set(inspect.signature(subagents.resolve_subagent_lane).parameters)
    assert "mutating" not in params
    assert not (params & {"mutating", "may_mutate", "write_surface", "surface"})
    assert "requested_lane" in params
    # Deliberately NOT an equality assertion: what this test is FOR is the absence of
    # an authority input, and pinning the exact signature would mean a future cleanup
    # has to delete a test in order to delete dead code (v6.87.28 deleted the
    # slot_index/slot_count fan-out parameters exactly that way).


def test_explicit_lane_is_honored_at_any_depth(monkeypatch):
    """Depth bounds how DEEP delegation goes, never how strong a descendant is —
    structurally: depth is not a resolver input at all (the dead parameter was
    removed in XG-2R.4; the signature guard below watches for regrowth)."""
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    assert "depth" not in __import__("inspect").signature(subagents.resolve_subagent_lane).parameters
    res = subagents.resolve_subagent_lane("heavy")
    assert res.effective_lane == "heavy"
    assert res.model == "provider::strong"


def test_explicit_main_honored(monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    res = subagents.resolve_subagent_lane("main")
    assert res.effective_lane == "main"


def test_code_lane_is_rejected_no_legacy_alias():
    with pytest.raises(ValueError):
        subagents.normalize_subagent_model_lane("code")


def test_build_envelope_tolerates_legacy_stored_lane():
    # The PUBLIC schema rejects "code", but an envelope built from an already-ran task's
    # durable record (which may carry a pre-v6.39 "code" lane) must NOT crash — it coerces
    # the unknown stored lane to a safe default (not a "code"->"heavy" alias).
    env = subagents.build_subagent_envelope(
        task_id="t1", parent_task_id="p1", root_task_id="r1", task_group_id="",
        depth=1, role="builder", requested_lane="code", effective_lane="code",
        model="m", status="completed", usage={},
    )
    assert env["requested_lane"] == "auto"
    # The coercion target is LANE_OF_RECORD — the same answer the resolver gives for
    # "no lane on record" — not a second hardcoded default that outlives the first.
    assert env["effective_lane"] == subagents.LANE_OF_RECORD == "main"


def test_string_false_may_mutate_stays_falsey(monkeypatch):
    # A tool-call payload may carry may_mutate as the STRING "false"; the SSOT
    # normalize_bool must treat it as falsey (regression: bool("false") was truthy).
    # Since v6.87.7 may_mutate governs AUTHORITY only — it no longer reaches the lane
    # resolver at all — so this pins the primitive rather than a routing side effect.
    from ouroboros.contracts.task_contract import normalize_bool
    assert normalize_bool("false") is False
    assert normalize_bool("true") is True


def test_use_local_empty_heavy_follows_main_flag(monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "")
    monkeypatch.setenv("USE_LOCAL_MAIN", "true")
    monkeypatch.delenv("USE_LOCAL_HEAVY", raising=False)
    res = subagents.resolve_subagent_lane("heavy")
    # The lane REPORTS the slot the model actually came from: with no Heavy slot
    # configured this child runs Main, and calling it "heavy" claimed a strength
    # nobody configured (v6.87.26 — the reduction capability_delta announces).
    assert res.effective_lane == "main"
    assert res.model == "provider::main"
    # Empty Heavy -> Main, so the Main local flag governs (not silently ignored).
    assert res.use_local_model is True


# ------------------------------------------------- cooldown trigger SSOT (C1)

def test_cooldown_error_kinds_include_rate_limit_but_not_in_retry_kinds():
    from ouroboros.loop_llm_call import _COOLDOWN_ERROR_KINDS, _TRANSIENT_RETRY_KINDS
    # A body-error 429 is classified "rate_limit" -> it MUST trigger cooldown.
    assert "rate_limit" in _COOLDOWN_ERROR_KINDS
    assert _TRANSIENT_RETRY_KINDS <= _COOLDOWN_ERROR_KINDS
    # ...but the same-model transient-retry budget must NOT be widened by it.
    assert "rate_limit" not in _TRANSIENT_RETRY_KINDS


# ------------------------------------ credentialed-model resolver parses chain (C2)

def test_resolve_credentialed_model_parses_fallbacks_chain(monkeypatch):
    from ouroboros.provider_models import resolve_credentialed_model
    # Only OpenRouter is credentialed in this environment.
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    for k in ("GIGACHAT_CREDENTIALS", "GIGACHAT_USER", "GIGACHAT_PASSWORD",
              "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CLOUDRU_FOUNDATION_MODELS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OUROBOROS_MODEL", "gigachat::GigaChat")  # uncredentialed
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "")
    # First chain entry uncredentialed (gigachat), second routes via OpenRouter.
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "gigachat::nocreds, anthropic/claude-sonnet-4.6")
    # The resolver must parse the chain and return the credentialed SECOND entry — not
    # test the raw comma-string as one (broken) model id, nor skip past it.
    assert resolve_credentialed_model("gigachat::GigaChat") == "anthropic/claude-sonnet-4.6"


def test_empty_light_slot_inherits_main_routing_even_when_models_match(monkeypatch):
    """ENV PRESENCE decides the inherit-from-Main case, not string equality. A
    local-only install whose Main happens to equal the shipped Light default must
    still route the Light lane locally — it is running Main, not the Light slot."""
    from ouroboros.config import SETTINGS_DEFAULTS
    from ouroboros.subagents import _use_local_for_lane

    shared = SETTINGS_DEFAULTS["OUROBOROS_MODEL_LIGHT"]
    monkeypatch.setenv("OUROBOROS_MODEL", shared)
    monkeypatch.delenv("OUROBOROS_MODEL_LIGHT", raising=False)
    monkeypatch.setenv("USE_LOCAL_MAIN", "true")
    monkeypatch.delenv("USE_LOCAL_LIGHT", raising=False)

    assert _use_local_for_lane("light", shared) is True
    # A slot the owner really configured still governs itself.
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", shared)
    assert _use_local_for_lane("light", shared) is False


def _scheduling_ctx(tmp_path, *, parent_deadline: str = "", parent_lane: str = ""):
    import queue

    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent1"
    ctx.task_depth = 0
    ctx.current_chat_id = 1
    ctx.event_queue = queue.Queue()
    ctx.active_model = "provider::parent"
    ctx.active_effort = "high"
    ctx.active_use_local = False
    ctx.task_metadata = {"root_task_id": "root1", "session_id": "sess1"}
    if parent_lane:
        ctx.task_metadata["effective_model_lane"] = parent_lane
    if parent_deadline:
        ctx.task_metadata["task_contract"] = {"deadline_at": parent_deadline}
    return ctx


def test_subagent_id_replaces_the_public_executor_axis(tmp_path, monkeypatch):
    """The selected row, not a second public executor argument, chooses substrate."""
    from ouroboros.tools.control import _schedule_task
    from tests._shared import configure_test_subagent

    api_id = configure_test_subagent(monkeypatch, subagent_id="api", kind="api_model")
    api_ctx = _scheduling_ctx(tmp_path / "api")
    assert "TOOL_ARG_ERROR" not in _schedule_task(
        api_ctx, subagent_id=api_id, objective="o", expected_output="e",
    )
    assert api_ctx.event_queue.get_nowait()["requested_executor"] == "native"

    session_id = configure_test_subagent(
        monkeypatch, subagent_id="session", kind="agent_session",
        target="claude=claude-fable-5",
    )
    session_ctx = _scheduling_ctx(tmp_path / "session")
    assert "TOOL_ARG_ERROR" not in _schedule_task(
        session_ctx, subagent_id=session_id, objective="o", expected_output="e",
    )
    assert session_ctx.event_queue.get_nowait()["requested_executor"] == "harness"

    conflict = _schedule_task(
        _scheduling_ctx(tmp_path / "conflict"), subagent_id=session_id,
        objective="o", expected_output="e", executor="harness",
    )
    assert "subagent_selector_conflict" in conflict


def test_effort_is_not_an_owner_facing_axis(tmp_path, monkeypatch):
    """There are THREE owner-facing axes and effort is not one of them (v6.87.28).

    A parent declares the WORK: write_surface (what the child may do), model_lane
    (how good the answer must be), executor (where it runs). A public `effort` broke
    that twice — it was a second knob for the question `model_lane` already answers,
    so `model_lane=light` with `effort=max` pinned the cheapest model to the
    strongest reasoning with no rule to reconcile them; and a harness route carries
    its own effort, so a parent asking `low` against a route pinned to `xhigh` had no
    rule for who wins. The refusal names the withdrawal instead of calling a
    parameter that was real for four releases 'unsupported'."""
    from ouroboros.tools.control import _schedule_task, schedule_subagent_properties
    from tests._shared import configure_test_subagent

    subagent_id = configure_test_subagent(monkeypatch)

    assert "effort" not in schedule_subagent_properties()

    ctx = _scheduling_ctx(tmp_path / "named")
    out = _schedule_task(ctx, objective="o", expected_output="e", effort="xhigh")
    assert "TOOL_ARG_ERROR" in out and "effort was withdrawn" in out
    assert "model_lane" in out
    assert ctx.event_queue.empty()

    # The combination that had no answer is refused at the door, not ranked.
    ctx = _scheduling_ctx(tmp_path / "conflict")
    out = _schedule_task(ctx, objective="o", expected_output="e",
                         model_lane="light", effort="max")
    assert "TOOL_ARG_ERROR" in out
    assert ctx.event_queue.empty()

    # Scheduling states intent; nothing about effort is recorded there at all.
    ctx = _scheduling_ctx(tmp_path / "omitted")
    assert "TOOL_ARG_ERROR" not in _schedule_task(
        ctx, subagent_id=subagent_id, objective="o", expected_output="e",
    )
    assert "reasoning_effort" not in ctx.event_queue.get_nowait()


def test_effort_is_derived_from_the_owner_setting_at_dispatch(tmp_path, monkeypatch):
    """Removing the knob did not remove the capability: the owner still controls
    effort through `config.resolve_effort(task_type)`, exactly as they did whenever
    the parameter was omitted — which was the normal case."""
    from ouroboros.agent import resolve_dispatch_axes

    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "xhigh")
    task = {"id": "c1", "type": "task", "delegation_role": "subagent"}
    dispatch = resolve_dispatch_axes(task)
    assert dispatch.effort == "xhigh"
    assert task["reasoning_effort"] == "xhigh"
    assert task["capability_delta"]["derived_effort"] == "xhigh"

    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "low")
    assert resolve_dispatch_axes({"id": "c2", "type": "task",
                                  "delegation_role": "subagent"}).effort == "low"


def test_a_stored_legacy_effort_is_ignored_with_the_reason_stated(tmp_path):
    """`effort` was model-visible and sits on durable records written before it was
    withdrawn. Loading one must not crash, must not obey it, and must not drop it in
    silence — a value that quietly stops meaning anything is the same class of defect
    as a reduction nobody announces."""
    from ouroboros.agent import capability_delta_prompt_block, resolve_dispatch_axes
    from ouroboros.subagents import LEGACY_SUBAGENT_FIELDS

    assert "reasoning_effort" in LEGACY_SUBAGENT_FIELDS

    task = {"id": "c1", "type": "task", "delegation_role": "subagent",
            "reasoning_effort": "max"}
    dispatch = resolve_dispatch_axes(task)
    # Not obeyed: the derived effort wins, whatever the record said.
    assert dispatch.effort == config.resolve_effort("task") != "max"
    assert task["reasoning_effort"] == dispatch.effort
    # ...and not dropped in silence.
    note = task["capability_delta"]["legacy_note"]
    assert "reasoning_effort='max'" in note and "derived" in note
    assert "Ignored on your record" in capability_delta_prompt_block(dispatch)
    # An ignored field is not a REDUCTION — nothing was taken away.
    assert task["capability_delta"]["reduced"] is False

    # A stray `effort` inside a stored task contract is dropped by the contract
    # builder rather than raising: contracts outlive the schema that wrote them.
    from ouroboros.contracts.task_contract import build_task_contract

    contract = build_task_contract({"id": "c1", "task_contract": {"effort": "max"}})
    assert "effort" not in contract


def _enqueue_through_supervisor(tmp_path, monkeypatch, *, parent_lane: str = "", **schedule_kwargs):
    """Drive the REAL path: tool call -> event -> supervisor -> the task a worker is handed."""
    from types import SimpleNamespace

    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.tools.control import _schedule_task
    from tests._shared import configure_test_subagent

    ctx = _scheduling_ctx(tmp_path, parent_lane=parent_lane)
    legacy_executor = str(schedule_kwargs.pop("executor", "") or "").strip()
    legacy_lane = str(schedule_kwargs.pop("model_lane", "") or "").strip()
    actor_effort = str(schedule_kwargs.pop("_actor_effort", "high") or "")
    if not schedule_kwargs.get("subagent_id"):
        is_session = legacy_executor == "harness"
        if is_session:
            target = "claude=route-a"
            monkeypatch.setattr(
                "ouroboros.subagents.route_health",
                lambda *_args, **_kwargs: ("configured_session_route_unavailable", ""),
            )
        else:
            monkeypatch.setattr(
                "ouroboros.provider_models.model_has_credentials", lambda _model: True,
            )
            if legacy_lane == "light":
                target = str(config.get_light_model() or config.get_main_model())
            elif legacy_lane == "heavy":
                target = str(config.get_heavy_model() or config.get_main_model())
            else:
                target = str(os.environ.get("OUROBOROS_MODEL") or "provider::main")
        schedule_kwargs["subagent_id"] = configure_test_subagent(
            monkeypatch,
            subagent_id="session-actor" if is_session else "api-actor",
            kind="agent_session" if is_session else "api_model",
            target=target,
            effort=actor_effort,
        )
    out = _schedule_task(ctx, objective="o", expected_output="e", **schedule_kwargs)
    assert "TOOL_ARG_ERROR" not in out, out
    event = ctx.event_queue.get_nowait()
    event["type"] = "schedule_subagent"
    event["depth"] = 0
    event["delegation_role"] = ""

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *a, **k: None)
    enqueued = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            pass

        def enqueue_task(self, task):
            enqueued.append(task)

        def persist_queue_snapshot(self, reason=""):
            pass

    ev_module._handle_schedule_task(event, FakeCtx())
    assert enqueued, "supervisor did not enqueue the task"
    return enqueued[0]


def test_the_request_reaches_the_worker_and_only_the_request(tmp_path, monkeypatch):
    """The parent's INTENT must reach the task the WORKER is handed — and nothing else.

    This asserts on the task the supervisor actually enqueues, not on the event and not on
    a re-implementation of the agent's fallback. An earlier version of this test built the
    payload itself from the event, which meant it supplied the very keys under test and
    could not fail — and a version before THAT re-implemented the agent's three lines in
    the test body. Both passed while the supervisor was silently dropping the keys on the
    floor. The loss is destructive, not merely inert: the worker writes its own view back
    over the durable record, so a drop here also erases the evidence of what was asked.

    The second half is the v6.87.28 invariant: what the child GETS is not on this task,
    because it has not been resolved. A schedule-time answer about live availability is
    an answer about a moment that has passed by the time the child starts."""
    task = _enqueue_through_supervisor(
        tmp_path, monkeypatch, parent_lane="heavy", executor="harness")
    assert task["requested_executor"] == "harness"
    assert task["requested_model_lane"] == "auto"
    assert task["parent_model_lane"] == "heavy"
    assert task["metadata"]["requested_executor"] == "harness"
    assert task["metadata"]["parent_model_lane"] == "heavy"
    for derived in ("effective_model_lane", "model", "use_local_model",
                    "reasoning_effort", "effective_executor", "capability_delta"):
        assert derived not in task, derived
        assert derived not in task["metadata"], derived


def test_availability_is_a_dispatch_fact_not_a_schedule_fact(tmp_path, monkeypatch):
    """The reason there is exactly one resolution and it runs at dispatch.

    A child scheduled while no harness route exists can wait out the whole outage in
    the queue. Resolving at schedule time froze the answer onto the record forever;
    resolving again at dispatch produced a SECOND record that disagreed with the first
    about the same child. With the D28 correction the down state is a typed BLOCK, so
    freezing it at schedule time would have refused a child whose route came back
    while it sat in the queue."""
    from ouroboros.agent import resolve_dispatch_axes
    import ouroboros.subagents as subagent_module

    task = _enqueue_through_supervisor(tmp_path, monkeypatch, executor="harness")

    state = {"reason": "subscription_window_exhausted"}
    monkeypatch.setattr(
        subagent_module, "route_health",
        lambda *_args, **_kwargs: (state["reason"], "2030-01-01T00:00:00Z"),
    )
    down = resolve_dispatch_axes(dict(task))
    assert (down.executor, down.route) == ("blocked", "")
    assert down.blocked is True and down.delta.reduced is True

    # The exact configured route comes back while the same immutable task waits.
    state["reason"] = ""
    up = resolve_dispatch_axes(dict(task))
    assert (up.executor, up.route) == ("harness", "claude=route-a")
    assert up.delta.reduced is False


def test_deadline_at_narrows_but_never_extends(tmp_path, monkeypatch):
    """`deadline_at` is public as of v6.87.7, and narrowing-only: a child may be bound
    tighter than its parent, never looser."""
    from ouroboros.tools.control import _INTERNAL_SCHEDULE_OPTIONS, _schedule_task
    from tests._shared import configure_test_subagent

    subagent_id = configure_test_subagent(monkeypatch)

    assert _INTERNAL_SCHEDULE_OPTIONS == frozenset()

    # Relative to now, not hardcoded: `deadline_at` must be a FUTURE instant, so fixed
    # calendar dates in this test would silently turn into rejections as time passes.
    from datetime import timedelta

    from ouroboros.deadline_utils import utc_now

    def stamp(hours):
        return (utc_now() + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

    parent, tighter, looser = stamp(12), stamp(9), stamp(23)

    ctx = _scheduling_ctx(tmp_path / "tighter", parent_deadline=parent)
    _schedule_task(
        ctx, subagent_id=subagent_id, objective="o", expected_output="e",
        deadline_at=tighter,
    )
    evt = ctx.event_queue.get_nowait()
    assert evt["task_contract"]["deadline_at"] == tighter

    ctx = _scheduling_ctx(tmp_path / "looser", parent_deadline=parent)
    _schedule_task(
        ctx, subagent_id=subagent_id, objective="o", expected_output="e",
        deadline_at=looser,
    )
    evt = ctx.event_queue.get_nowait()
    assert evt["task_contract"]["deadline_at"] == parent

    # A model-authored deadline is validated, because both failures are otherwise silent.
    ctx = _scheduling_ctx(tmp_path / "garbage")
    out = _schedule_task(
        ctx, subagent_id=subagent_id, objective="o", expected_output="e",
        deadline_at="in 2 hours",
    )
    assert "TOOL_ARG_ERROR" in out and "ISO-8601" in out
    assert ctx.event_queue.empty()

    ctx = _scheduling_ctx(tmp_path / "past")
    out = _schedule_task(
        ctx, subagent_id=subagent_id, objective="o", expected_output="e",
        deadline_at=stamp(-1),
    )
    assert "TOOL_ARG_ERROR" in out and "already in the past" in out
    assert ctx.event_queue.empty()


def test_the_envelope_states_the_request_until_dispatch_fills_it_in(tmp_path, monkeypatch):
    """The envelope is the subagent's public description, and until the child is
    dispatched the honest description has an intent and NO answer. `effective_lane`
    used to default to `light`, so a queued child's envelope named a lane, a slot and
    a strength that no resolution had produced — a claim, not a record."""
    from ouroboros.agent import resolve_dispatch_axes
    from ouroboros.tools.control import _schedule_task
    from tests._shared import configure_test_subagent

    ctx = _scheduling_ctx(tmp_path / "asked")
    subagent_id = configure_test_subagent(
        monkeypatch, subagent_id="session-actor", kind="agent_session",
        target="claude=route-a",
    )
    _schedule_task(
        ctx, subagent_id=subagent_id, objective="o", expected_output="e",
    )
    envelope = ctx.event_queue.get_nowait()["subagent_envelope"]
    assert envelope["executor"] == "harness"          # the request
    assert envelope["effective_lane"] == ""           # nothing resolved yet
    assert envelope["reasoning_effort"] == ""
    assert envelope["effective_executor"] == ""
    assert envelope["capability_delta"] == {}

    task = _enqueue_through_supervisor(tmp_path / "ran", monkeypatch, executor="harness")
    resolve_dispatch_axes(task)
    filled = task["subagent_envelope"]
    assert filled["effective_lane"] == "main"
    assert filled["model"] and filled["reasoning_effort"]
    # The pin no route can honor is a typed block, never a silent re-route to paid
    # native execution (D28).
    assert filled["effective_executor"] == "blocked"
    assert filled["tool_profile"] == "local_readonly_subagent"
    assert filled["capability_delta"]["reduced"] is True


def test_the_scheduling_intent_survives_a_queue_snapshot(tmp_path, monkeypatch):
    """A pending child that waits through a restart must come back holding what its
    parent asked for, INCLUDING the parent's own lane: an omitted lane inherits it and
    only the parent knew it, so a resumed child without it would resolve `auto`
    against the lane of record and silently come back weaker. The intent lives at the
    task TOP LEVEL because that is where the resolution reads it — restoring only the
    copies nested in `metadata` would leave a resumed child resolving from nothing."""
    import supervisor.queue as q

    task = _enqueue_through_supervisor(
        tmp_path, monkeypatch, parent_lane="heavy", executor="harness")

    import json as _json

    captured = {}
    monkeypatch.setattr(q, "atomic_write_text",
                        lambda path, text: captured.update(_json.loads(text)))
    monkeypatch.setattr(q, "PENDING", [task], raising=False)
    monkeypatch.setattr(q, "RUNNING", {}, raising=False)
    assert q.persist_queue_snapshot(reason="test") is True

    rows = captured.get("pending") or []
    assert rows, captured
    restored = rows[0]["task"]
    assert restored["requested_executor"] == "harness"
    assert restored["parent_model_lane"] == "heavy"


def test_a_dispatched_childs_delta_survives_a_restart(tmp_path, monkeypatch):
    """The other half: once a child HAS been dispatched, its resolution must not be
    re-derived by a replay. A RUNNING row that came back through a snapshot without
    the delta would leave the child believing its pin had been honored.

    This pins the SERIALIZATION half only (the snapshot's field list) by injecting
    an already-resolved task into RUNNING; how the resolution REACHES the
    supervisor's RUNNING copy across the process boundary is pinned by
    test_the_workers_resolution_crosses_the_process_boundary_to_the_snapshot —
    without that merge, this test alone passed while real snapshots stayed
    unresolved (XG-2R.1)."""
    import supervisor.queue as q

    from ouroboros.agent import resolve_dispatch_axes

    task = _enqueue_through_supervisor(tmp_path, monkeypatch, executor="harness")
    resolve_dispatch_axes(task)

    import json as _json

    captured = {}
    monkeypatch.setattr(q, "atomic_write_text",
                        lambda path, text: captured.update(_json.loads(text)))
    monkeypatch.setattr(q, "PENDING", [], raising=False)
    monkeypatch.setattr(q, "RUNNING", {"c1": {"task": task, "worker_id": 0,
                                              "started_at": 0.0, "attempt": 1}}, raising=False)
    assert q.persist_queue_snapshot(reason="test") is True

    restored = (captured.get("running") or [])[0]["task"]
    assert restored["effective_executor"] == "blocked"
    assert restored["capability_delta"]["reduced"] is True
    assert restored["reasoning_effort"] == "high"


def test_the_workers_resolution_crosses_the_process_boundary_to_the_snapshot(tmp_path, monkeypatch):
    """XG-2R.1 (three reviewers converged): `resolve_dispatch_axes` stamps the WORKER
    process's clone of the task, `assign_tasks` holds its own `dict(task)` in RUNNING,
    and `persist_queue_snapshot` serializes the supervisor's copy — so without a
    worker->supervisor merge the real snapshot carried the UNRESOLVED intent and a
    restart lost the resolved axes and `capability_delta`.

    This test crosses the REAL seam instead of hand-injecting a resolved task:
    the worker's copy is a serialized clone (as pickling across the process
    boundary makes it), the resolution travels ONLY as the JSON-serializable
    `task_dispatch_resolved` event through the REAL registered handler
    (`dispatch_event`), the handler itself takes the snapshot, and the restored
    row must carry the resolved axes + delta."""
    import json as _json
    import queue as queue_mod

    import supervisor.queue as q
    from supervisor import events as ev_module
    from ouroboros.agent import emit_dispatch_resolution, resolve_dispatch_axes
    from ouroboros.subagents import SUBAGENT_RESOLUTION_FIELDS

    supervisor_task = _enqueue_through_supervisor(tmp_path, monkeypatch, executor="harness")
    # The supervisor's RUNNING copy at assignment — intent only, exactly as
    # assign_tasks stores it BEFORE the worker resolves anything.
    running = {"c1": {"task": dict(supervisor_task), "worker_id": 0,
                      "started_at": 0.0, "attempt": 1}}

    # The worker receives a SERIALIZED CLONE: its mutations cannot alias into the
    # supervisor's dict.
    worker_task = _json.loads(_json.dumps(supervisor_task))
    worker_task["id"] = "c1"
    out_q = queue_mod.Queue()
    dispatch = resolve_dispatch_axes(worker_task)
    # The merge set is pinned to the one writer: record_fields() + the envelope.
    assert set(SUBAGENT_RESOLUTION_FIELDS) == set(dispatch.record_fields()) | {"subagent_envelope"}
    emit_dispatch_resolution(out_q, worker_task, dispatch)

    # The masked defect, stated: the supervisor's copy is still unresolved.
    assert "effective_executor" not in running["c1"]["task"]

    captured = {}
    monkeypatch.setattr(q, "atomic_write_text",
                        lambda path, text: captured.update(_json.loads(text)))
    monkeypatch.setattr(q, "PENDING", [], raising=False)
    monkeypatch.setattr(q, "RUNNING", running, raising=False)

    class Ctx:
        RUNNING = running

        @staticmethod
        def persist_queue_snapshot(reason=""):
            return q.persist_queue_snapshot(reason=reason)

    # Only the event crosses — a JSON round-trip proves nothing shared rides along.
    evt = _json.loads(_json.dumps(out_q.get_nowait()))
    assert evt["type"] == "task_dispatch_resolved"
    ev_module.dispatch_event(evt, Ctx())

    # Restart: what a restore reads back is the snapshot the HANDLER persisted.
    rows = captured.get("running") or []
    assert rows and rows[0]["id"] == "c1", captured.get("reason")
    restored = rows[0]["task"]
    assert restored["effective_executor"] == "blocked"
    assert restored["capability_delta"]["reduced"] is True
    assert restored["effective_model_lane"] == "main"
    assert restored["model"]
    assert restored["reasoning_effort"] == "high"
    assert restored["subagent_envelope"]["effective_executor"] == "blocked"
    # Intent was merged INTO, not replaced: the request the parent stated survives.
    assert restored["requested_executor"] == "harness"


def test_a_prior_resolutions_residue_is_not_a_legacy_request(tmp_path):
    """Consequence of the resolution surviving the snapshot (XG-2R.1, fable's
    self_consistency half): a crash-requeued child's record now carries
    `reasoning_effort` BECAUSE record_fields() wrote it. Re-dispatching that record
    must not disclose a false 'reasoning_effort=... ignored' legacy note to the
    child prompt and parent readback — LEGACY_SUBAGENT_FIELDS names fields from
    RETIRED SCHEMAS, and a record carrying its own capability_delta proves the
    value is the resolver's residue. A genuinely legacy record (no delta) keeps
    the note."""
    from ouroboros.agent import resolve_dispatch_axes

    task = {"id": "c1", "type": "task", "delegation_role": "subagent"}
    first = resolve_dispatch_axes(task)
    assert first.delta.legacy_note == ""
    assert task["reasoning_effort"]  # the residue the snapshot now preserves

    # The requeue replay: same record, resolution already on it.
    replay = resolve_dispatch_axes(dict(task))
    assert replay.legacy_ignored == {}
    assert replay.delta.legacy_note == ""

    # The genuine legacy case is unchanged: stored effort, no prior resolution.
    legacy = resolve_dispatch_axes({"id": "c2", "type": "task",
                                    "delegation_role": "subagent",
                                    "reasoning_effort": "max"})
    assert "reasoning_effort" in legacy.delta.legacy_note


# ------------------------------------------------------- capability_delta (v6.87.26)

def _light_lane_ctx(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    return _scheduling_ctx(tmp_path, **kwargs)


def _dispatched(tmp_path, monkeypatch, **schedule_kwargs):
    """Drive the WHOLE path: tool call -> event -> supervisor -> the worker's dispatch.

    Everything a child GETS is decided in the last step, so a test that stops at the
    event asserts on intent and calls it a resolution."""
    from ouroboros.agent import resolve_dispatch_axes

    task = _enqueue_through_supervisor(tmp_path, monkeypatch, **schedule_kwargs)
    return task, resolve_dispatch_axes(task)


def test_configured_api_actor_uses_its_exact_model_not_the_legacy_parent_lane(tmp_path, monkeypatch):
    """The stable row is now the model authority; parent lanes are history only."""
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")

    task, _ = _dispatched(tmp_path / "main", monkeypatch, parent_lane="heavy")
    assert task["effective_model_lane"] == "main"
    assert task["model"] == "provider::main"
    assert task["requested_model_lane"] == "auto"
    assert task["capability_delta"]["reduced"] is False

    named, _ = _dispatched(tmp_path / "named", monkeypatch, model_lane="light")
    assert named["effective_model_lane"] == "main"
    assert named["model"] == "provider::cheap"


def test_a_reduction_reaches_the_record_the_child_and_the_parents_readback(tmp_path, monkeypatch):
    """The invariant: a child landing below what was asked for is LOUD in all THREE
    places named by the owner — the durable record/envelope, the child's own prompt,
    and the TERMINAL parent-facing result.

    The executor pin is the reduction that had no reporting at all — `harness` was
    recorded on the event, the task and the envelope (under the key `executor`, which
    reads as who RAN it) and then no code ever resolved it, so a child that ran
    natively left a durable record claiming a harness had run it.

    Since the D28 correction the unhonored EXPLICIT pin resolves to `blocked` rather
    than to paid `native` (see test_an_explicit_harness_pin_is_a_typed_blocker), so
    what the three surfaces must carry is the BLOCK. The disclosure duty is the same;
    only the honest answer changed.

    None of the three can be reached at SCHEDULE time any more, which is the point:
    all three read a fact that does not exist until the child starts."""
    from ouroboros.agent import capability_delta_prompt_block
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control import _get_task_result

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    task, dispatch = _dispatched(tmp_path / "three", monkeypatch, executor="harness")

    # 1) the durable record + its envelope
    assert task["effective_executor"] == "blocked"
    delta = task["capability_delta"]
    assert delta["reduced"] is True
    assert delta["reason"] == "configured_session_route_unavailable"
    assert task["subagent_envelope"]["capability_delta"] == delta
    assert task["subagent_envelope"]["executor"] == "harness"

    # 2) the child's own prompt
    block = capability_delta_prompt_block(dispatch)
    assert "[CAPABILITY DELTA]" in block
    assert "executor harness->blocked" in block

    # 3) the parent, when it READS the answer
    ctx = _scheduling_ctx(tmp_path / "readback")
    write_task_result(tmp_path / "readback", "child1", "completed",
                      result="done", capability_delta=delta)
    out = _get_task_result(ctx, "child1")
    assert "capability_delta" in out and "configured_session_route_unavailable" in out

    # ...and the scheduling result no longer pretends to know: it states the request.
    from ouroboros.tools.control import _schedule_task
    from tests._shared import configure_test_subagent

    sched_ctx = _scheduling_ctx(tmp_path / "sched")
    subagent_id = configure_test_subagent(
        monkeypatch, subagent_id="session-actor", kind="agent_session",
        target="claude=route-a",
    )
    scheduled = _schedule_task(
        sched_ctx, subagent_id=subagent_id, objective="o", expected_output="e",
    )
    assert "CAPABILITY_DELTA" not in scheduled
    assert "subagent_id=session-actor" in scheduled
    assert "route=agent_session" in scheduled


def test_a_child_that_got_what_was_asked_stays_quiet(tmp_path, monkeypatch):
    """A warning that always fires is not a warning. Nothing was taken away here, so
    no block reaches the child and no delta reaches the parent's readback — `auto`
    resolving to a concrete executor is the absence of a preference, not a loss."""
    from ouroboros.agent import capability_delta_prompt_block

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    task, dispatch = _dispatched(tmp_path / "quiet", monkeypatch, parent_lane="light")
    assert task["capability_delta"]["reduced"] is False
    assert task["capability_delta"]["effective_executor"] == "native"
    assert capability_delta_prompt_block(dispatch) == ""


def test_an_explicit_harness_pin_is_a_typed_blocker_not_a_paid_reroute(tmp_path, monkeypatch):
    """D28, the owner's words: at an EXPLICIT `executor: harness` an unavailable route
    stays a TYPED BLOCKER — «деньги API не тратятся без явного выбора». The §8 ban #12
    exception (fall back to another route) is AUTO-ONLY.

    This resolved to `native` with a loud `capability_delta`, which discloses the wrong
    thing: however loudly it is announced, re-routing the pin to native execution
    spends exactly the metered money the parent refused. The reason string matches
    `cxi/p34-converged`'s rule table so synthesis adopts that table without a
    behavioural diff (synthesis hazard H1)."""
    from ouroboros import subagents as sub

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")

    # EXPLICIT harness, no route: blocked, and the block is one predicate.
    task, dispatch = _dispatched(tmp_path / "pinned", monkeypatch, executor="harness")
    assert dispatch.executor == "blocked" and dispatch.blocked is True
    assert dispatch.route == ""
    assert task["effective_executor"] == "blocked"
    assert dispatch.delta.reason == "configured_session_route_unavailable"
    assert dispatch.delta.reduced is True

    # AUTO with no route: native, and quiet — nothing was asked for.
    auto_task, auto_dispatch = _dispatched(tmp_path / "auto", monkeypatch, executor="auto")
    assert auto_dispatch.executor == "native" and auto_dispatch.blocked is False
    assert auto_task["capability_delta"]["reduced"] is False

    # The whole rule table, at the SURVIVING resolver (p34's typed table — H1):
    # `route` is a DelegationRoute or None, and the outcome is a typed record.
    route_a = sub.DelegationRoute(route_id="route-a")

    def row(requested, route):
        res = sub.resolve_subagent_executor(requested, route=route)
        return res.executor, res.reason

    assert row("harness", None) == ("blocked", "harness_not_configured")
    assert row("harness", route_a) == ("harness", "harness_ready")
    assert row("auto", None) == ("native", "harness_not_configured")
    assert row("auto", route_a) == ("harness", "harness_ready")
    assert row("native", None) == ("native", "requested_native")
    # An exhausted subscription window blocks a PIN and falls auto back, loudly.
    spent = sub.resolve_subagent_executor("harness", route=route_a, reset_at="2030-01-01T00:00:00Z")
    assert (spent.executor, spent.reason) == ("blocked", "subscription_window_exhausted")
    assert sub.resolve_subagent_executor("auto", route=route_a, reset_at="X").executor == "native"
    # `blocked` is a resolution OUTCOME, never a request a parent may make.
    assert "blocked" not in sub.SUBAGENT_EXECUTORS


def test_blocked_configured_session_terminals_unrun_without_a_model_round(tmp_path, monkeypatch):
    """Charter D2 (owner 2026-08-28, N2=A): a blocked configured session ends the
    task UNRUN and typed at $0 — never a model episode, never host API fallback.
    The dc4c0204 wake-the-nanny behavior this test used to pin is retired."""
    from ouroboros import agent as agent_module
    from ouroboros.agent import Env, OuroborosAgent

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr("ouroboros.agent.build_llm_messages", lambda **kwargs: ([], {}))

    calls: list = []

    def _never(**kwargs):
        calls.append(kwargs)
        return "the model was called", {}, {"reasoning_notes": [], "tool_calls": []}

    monkeypatch.setattr(agent_module, "run_llm_loop", _never)

    repo = tmp_path / "repo"
    repo.mkdir()
    drive = tmp_path / "drive"
    drive.mkdir()

    pinned = _enqueue_through_supervisor(tmp_path / "sched", monkeypatch, executor="harness")
    pinned.update({"id": "pinned1", "chat_id": 1, "drive_root": str(drive)})

    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=drive))
    agent._handle_task_scoped(dict(pinned))

    assert calls == [], "a blocked pin must terminal unrun with zero model rounds"
    from ouroboros import delegate_custody as custody

    custody_events = [
        json.loads(line)
        for line in custody.event_log_path(
            pathlib.Path(pinned["budget_drive_root"])
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        evt.get("type") == "delegate_run_configured_startup_fault"
        and evt.get("host_fallback") is False
        for evt in custody_events
    )

    # The durable record keeps the selected session intent and typed failure.
    result_path = drive / "task_results" / "pinned1.json"
    record = json.loads(result_path.read_text(encoding="utf-8"))
    assert record["effective_executor"] == "blocked"
    assert record["model"] == "provider::parent"
    assert record["configured_subagent"]["selected_subagent_id"] == "session-actor"
    assert record["capability_delta"]["reason"] == "configured_session_route_unavailable"
    assert float(record.get("cost_usd") or 0.0) == 0.0
    assert "NOT run on metered API tokens" in str(record.get("result") or "")


def test_legacy_heavy_migration_selects_an_exact_api_model(tmp_path, monkeypatch):
    """The compatibility fixture becomes an exact actor, never a live lane fallback."""
    from ouroboros.agent import capability_delta_prompt_block

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "")
    task, dispatch = _dispatched(tmp_path / "noheavy", monkeypatch, model_lane="heavy")
    assert task["effective_model_lane"] == "main"
    assert task["model"] == "provider::main"
    assert task["capability_delta"]["reason"] == ""
    assert capability_delta_prompt_block(dispatch) == ""

    # ...and a configured Heavy slot is honored silently.
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    task, dispatch = _dispatched(tmp_path / "heavy", monkeypatch, model_lane="heavy")
    assert task["effective_model_lane"] == "main"
    assert task["model"] == "provider::strong"
    assert capability_delta_prompt_block(dispatch) == ""


def test_legacy_model_global_ceiling_is_not_dispatch_authority(tmp_path, monkeypatch):
    """Scheduling reports requested effort; exact wire disclosure owns adaptation."""
    from ouroboros.agent import capability_delta_prompt_block
    from ouroboros.llm import LLMClient

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "max")
    monkeypatch.setitem(LLMClient._EFFORT_CEILING_CACHE, "provider::cheap", "low")

    task, dispatch = _dispatched(
        tmp_path / "ceiling", monkeypatch, model_lane="light", _actor_effort="max",
    )
    delta = task["capability_delta"]
    assert (delta["derived_effort"], delta["effective_effort"]) == ("max", "max")
    assert delta["reason"] == ""
    assert "effort" not in capability_delta_prompt_block(dispatch)
    assert task["reasoning_effort"] == "max"

    # An effort inside the band is not a delta.
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "low")
    _task, dispatch = _dispatched(
        tmp_path / "inband", monkeypatch, model_lane="light", _actor_effort="low",
    )
    assert capability_delta_prompt_block(dispatch) == ""


def test_a_legacy_parent_lane_does_not_break_scheduling_or_dispatch(tmp_path, monkeypatch):
    """Inheritance reads the PARENT's stored lane, and durable data outlives the schema
    that wrote it. A pre-v6.39 `code` on the parent's record must not turn every child
    it spawns into an uncaught ValueError — the public schema stays strict about what
    a CALLER may ask for, which is a different question."""
    from ouroboros.tools.control import _schedule_task

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    task, _ = _dispatched(tmp_path / "legacy", monkeypatch, parent_lane="code")
    assert task["effective_model_lane"] == "main"

    # A caller asking for it directly is still refused.
    ctx = _light_lane_ctx(tmp_path / "asked", monkeypatch)
    assert "subagent_selection_required" in _schedule_task(
        ctx, objective="o", expected_output="e", model_lane="code")


def test_the_completion_envelope_is_built_from_the_same_mapping_as_the_scheduler():
    """The envelope a RAN child publishes and the one the scheduler wrote are twins,
    and they were two field-by-field mappings in two modules. They had already
    drifted: the completion side re-derived the effective-lane fallback as a
    hardcoded `light`, so a record missing that field came back describing a lane
    the resolver would never produce — and the delta axes had to be added twice."""
    from ouroboros.subagents import envelope_from_task

    delta = {"requested_executor": "harness", "effective_executor": "native", "reduced": True}
    env = envelope_from_task(
        {"id": "c1", "model_lane": "", "requested_executor": "harness",
         "effective_model_lane": "heavy", "effective_executor": "native",
         "executor_route": "", "tool_profile": "acting_subagent",
         "capability_delta": delta},
        status="completed", usage={"rounds": 3},
    )
    assert env["effective_lane"] == "heavy"
    assert env["executor"] == "harness"
    assert env["effective_executor"] == "native"
    assert env["tool_profile"] == "acting_subagent"
    assert env["capability_delta"]["reduced"] is True
    assert env["usage"]["rounds"] == 3

    # A record with NO resolution on it describes no lane, rather than substituting
    # one: "not dispatched" and "ran on the lane of record" are different facts.
    assert envelope_from_task({"id": "c2"}, status="requested")["effective_lane"] == ""


def test_lane_rank_is_the_only_lane_ordering(tmp_path):
    """One comparison decides "weaker than what was asked" for every axis. Effort
    already had `config.effort_rank`; the lane had nothing, so the question was
    simply never asked. `auto` has no rank — it is a request to inherit, not a
    strength, so the thing an effective lane is measured against is the lane the
    request RESOLVED FROM, never the literal `auto`."""
    from ouroboros.subagents import LANE_STRENGTH, lane_is_weaker, lane_rank

    assert LANE_STRENGTH == ("light", "main", "heavy")
    assert lane_rank("light") < lane_rank("main") < lane_rank("heavy")
    assert lane_rank("auto") == -1
    assert lane_rank("code") == -1
    assert lane_is_weaker("main", "heavy") is True
    assert lane_is_weaker("heavy", "main") is False
    # Nothing can rank below `auto`, which is why comparing against it was a
    # disclosure that could never fire.
    assert lane_is_weaker("light", "auto") is False


def test_intended_lane_is_the_one_owner_of_what_a_request_means(tmp_path):
    """`auto` means "the parent's lane". Two places need that answer and neither may
    own it: the resolution measures the effective lane against it, and the ADMISSION
    gate for a `require_lane` constraint runs before the child is dispatched, so it
    cannot ask what lane the child ended up on. One predicate, two readers."""
    from ouroboros.subagents import LANE_OF_RECORD, intended_lane, resolve_subagent_lane

    assert intended_lane("auto", "heavy") == "heavy"
    assert intended_lane("light", "heavy") == "light"
    assert intended_lane("auto", "") == LANE_OF_RECORD
    # Stored garbage on either side must not make a child unschedulable.
    assert intended_lane("auto", "code") == LANE_OF_RECORD
    assert intended_lane("code", "heavy") == "heavy"
    # The resolution asks this predicate rather than re-deriving it.
    assert resolve_subagent_lane("auto", parent_lane="heavy").resolved_from == "heavy"


# ------------------------------------------------- v6.87.27: the twins that were missed

def test_configured_actor_does_not_inherit_a_stale_heavy_parent(tmp_path, monkeypatch):
    """A historical parent lane cannot silently rewrite an explicit actor snapshot."""
    from ouroboros.agent import capability_delta_prompt_block

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "")

    task, dispatch = _dispatched(tmp_path / "inherited-noheavy", monkeypatch, parent_lane="heavy")
    delta = task["capability_delta"]
    assert (delta["requested_lane"], delta["resolved_lane"], delta["effective_lane"]) == (
        "auto", "main", "main")
    assert delta["reduced"] is False
    assert delta["reason"] == ""
    assert task["effective_model_lane"] == "main"
    assert task["model"] == "provider::main"
    assert capability_delta_prompt_block(dispatch) == ""

    # An inherited lane the install CAN provide still takes nothing away.
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    _ok, quiet = _dispatched(
        tmp_path / "inherited-ok", monkeypatch, parent_lane="heavy", model_lane="heavy",
    )
    assert capability_delta_prompt_block(quiet) == ""


def test_scheduler_effort_stays_pre_wire_until_exact_route_disclosure(tmp_path, monkeypatch):
    """Legacy floors stay diagnostic; physical request usage reports any adjustment."""
    from ouroboros.agent import capability_delta_prompt_block
    from ouroboros.llm import LLMClient

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "none")
    monkeypatch.setitem(LLMClient._EFFORT_FLOOR_CACHE, "provider::cheap", "low")
    monkeypatch.setitem(LLMClient._EFFORT_FLOOR_LOADED, "provider::cheap", float("inf"))

    task, dispatch = _dispatched(
        tmp_path / "floor", monkeypatch, model_lane="light", _actor_effort="none",
    )
    delta = task["capability_delta"]
    assert LLMClient.clamp_effort_for_route("provider::cheap", "none") == "low"
    assert delta["effective_effort"] == "none"
    assert delta["reduced"] is False
    assert capability_delta_prompt_block(dispatch) == ""

    # ...and it must not become a false alarm when ANOTHER axis opens the block: a
    # raised effort inside a real reduction still is not something taken away.
    _both, both_dispatch = _dispatched(
        tmp_path / "floor-and-pin", monkeypatch, model_lane="light", executor="harness",
        _actor_effort="none",
    )
    block = capability_delta_prompt_block(both_dispatch)
    assert "executor harness->blocked" in block
    assert "effort none->low" not in block


def test_a_require_lane_refusal_states_the_facts_not_the_lane_default(tmp_path):
    """The refusal is read by the model at the exact moment it is deciding how to fix
    a rejected spawn, and it restated a default owned three modules away in
    `subagents`. That copy went stale in v6.87.7, was corrected in v6.87.14 and went
    stale AGAIN in v6.87.26 — it told the model an omitted lane resolves to `light`
    while the code inherits the parent's. It now states only what the reducer holds,
    and it is measured against the INTENDED lane, because admission runs before the
    child is dispatched and the effective lane does not exist yet."""
    from ouroboros.tools.control_delegation import effective_delegation_budget

    row = {"payload": {"constraint_id": "c1", "directive": "require_lane",
                       "scope": {"lane": "heavy"}}}
    refusal = effective_delegation_budget(
        {}, unresolved_constraints=[row], role="critic",
        requested_lane="auto", intended_lane="main")
    assert refusal.ok is False
    assert refusal.reason_code == "delegation_constraint_require_lane"
    # No claim about what an omitted lane means — that rule is not this module's.
    assert "v6.87" not in refusal.detail
    # The facts it does hold, and a REACHABLE remedy: "ask for the lane explicitly"
    # is not one when the install has no such slot, so the constraint has to give.
    assert "'heavy'" in refusal.detail and "'auto'" in refusal.detail and "'main'" in refusal.detail
    assert "override_delegation_constraint('c1')" in refusal.detail

    # An omitted lane that INHERITS the required one is admitted: the gate reads the
    # same predicate the resolution does, so it cannot disagree with it about `auto`.
    from ouroboros.subagents import intended_lane

    ok = effective_delegation_budget(
        {}, unresolved_constraints=[row], role="critic", requested_lane="auto",
        intended_lane=intended_lane("auto", "heavy"))
    assert ok.ok is True


def test_the_admission_gate_asks_the_predicate_rather_than_the_raw_request(tmp_path, monkeypatch):
    """The WIRING, not the reducer. The reducer above is pure and can be handed
    anything; what decides whether a real spawn is admitted is what the SUPERVISOR
    passes it. Handing it the raw request means `auto` is compared verbatim against a
    required lane, so a Heavy parent whose omitted-lane child INHERITS Heavy — the
    v6.87.26 default, and the common case — is rejected for asking for the very lane
    the constraint demands."""
    import ouroboros.task_tree_ledger as ledger

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    monkeypatch.setattr(
        ledger, "open_delegation_constraints",
        lambda _root: [{"payload": {"constraint_id": "c1", "directive": "require_lane",
                                    "scope": {"lane": "heavy"}}}])

    task = _enqueue_through_supervisor(tmp_path, monkeypatch, parent_lane="heavy")
    assert task["parent_model_lane"] == "heavy"
    assert task["requested_model_lane"] == "auto"


def test_the_parent_sees_the_reduction_when_it_reads_the_childs_answer(tmp_path):
    """The TERMINAL parent-facing disclosure, and since v6.87.28 the only one: the
    reduction is not known until the child is dispatched, so a scheduling result
    cannot carry it. This is also the moment the parent cares most — it is reading
    the ANSWER that decides whether to trust a weaker result."""
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control import _get_task_result

    ctx = _scheduling_ctx(tmp_path / "readback")
    reduced = {"requested_lane": "heavy", "resolved_lane": "heavy", "effective_lane": "main",
               "requested_executor": "harness", "effective_executor": "native",
               "reason": "lane_slot_unavailable=heavy", "reduced": True}
    write_task_result(tmp_path / "readback", "child1", "completed",
                      result="done", capability_delta=reduced)
    out = _get_task_result(ctx, "child1")
    assert "capability_delta" in out
    assert "lane_slot_unavailable=heavy" in out

    # A delta that took nothing away and ignored nothing is noise in every payload.
    write_task_result(tmp_path / "readback", "child2", "completed",
                      result="done", capability_delta={**reduced, "reduced": False})
    assert "capability_delta" not in _get_task_result(ctx, "child2")

    # ...but an IGNORED legacy field is something to say, even without a reduction.
    write_task_result(tmp_path / "readback", "child3", "completed", result="done",
                      capability_delta={"reduced": False, "legacy_note": "reasoning_effort='max' ignored"})
    assert "legacy_note" in _get_task_result(ctx, "child3")


def test_the_batch_absorb_discloses_the_reduction_too(tmp_path):
    """The TWIN of the single-child read, and the one a fan-out parent actually uses.

    A parent absorbs children through two surfaces: `get_task_result`/`wait_task` read
    one child in full, and `wait_tasks` projects a batch compactly — which is the
    right tool for "five independent children scheduled in one burst" by its own tool
    description. The delta reached the first and not the second, so the parent most
    likely to have several weakened children was the one told about none of them. The
    compact projection is a DISCLOSED omission of forensics; a capability reduction is
    not forensics, it is what decides how far to trust the answer."""
    import json as _json

    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control import _get_task_result, _wait_for_tasks

    ctx = _scheduling_ctx(tmp_path / "batch")
    reduced = {"requested_lane": "heavy", "resolved_lane": "heavy", "effective_lane": "main",
               "requested_executor": "harness", "effective_executor": "native",
               "reason": "lane_slot_unavailable=heavy", "reduced": True}
    root = tmp_path / "batch"
    write_task_result(root, "c1", "completed", result="done", capability_delta=reduced)
    write_task_result(root, "c2", "completed", result="done",
                      capability_delta={**reduced, "reduced": False, "legacy_note": ""})

    batch = _json.loads(_wait_for_tasks(ctx, ["c1", "c2"], timeout_sec=1))["tasks"]
    assert batch["c1"]["capability_delta"]["reason"] == "lane_slot_unavailable=heavy"
    # The same predicate decides both surfaces, so they cannot disagree about which
    # deltas are worth saying.
    assert "capability_delta" not in batch["c2"]
    assert ("capability_delta" in _get_task_result(ctx, "c1")) is True
    assert ("capability_delta" in _get_task_result(ctx, "c2")) is False


def test_one_resolution_writes_every_derived_field(tmp_path, monkeypatch):
    """"Not two resolvers, not two records." Every derived field on a child's record
    comes from `SubagentDispatch.record_fields()`, so an added axis is one edit rather
    than a field-by-field mapping repeated in four modules that drift apart a release
    later — and no OTHER surface may mint one."""
    from ouroboros.agent import resolve_dispatch_axes
    from ouroboros.subagents import SUBAGENT_INTENT_FIELDS, resolve_subagent_dispatch

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    task = {"id": "c1", "type": "task", "delegation_role": "subagent",
            "requested_model_lane": "auto", "parent_model_lane": "heavy",
            "requested_executor": "auto"}
    before = dict(task)
    dispatch = resolve_dispatch_axes(task)

    derived = dispatch.record_fields()
    assert set(derived) == {
        "effective_model_lane", "model", "use_local_model", "reasoning_effort",
        "effective_executor", "executor_route", "tool_profile", "capability_delta"}
    # Nothing derived leaks into the intent half, and nothing intended is rewritten.
    assert not set(derived) & set(SUBAGENT_INTENT_FIELDS)
    for key in SUBAGENT_INTENT_FIELDS:
        assert task.get(key) == before.get(key)

    # The resolution is a pure function of the record: asking twice answers twice.
    assert resolve_subagent_dispatch(before, task_type="task").record_fields() == derived

    # A task that is not a delegated child is not resolved at all.
    root = {"id": "r1", "type": "task"}
    assert resolve_dispatch_axes(root) is None
    assert "capability_delta" not in root


def test_queue_snapshot_projects_every_scheduling_intent_field(monkeypatch, tmp_path):
    """R2-3 (F9 delta): a PENDING child's queue-snapshot row is all a restarted
    supervisor has, so an intent field missing from the projection is silently
    dropped across restart — `required_model_lane` was, re-opening the
    auto+harness⇒light default over a gate-verified lane. Walk
    SUBAGENT_INTENT_FIELDS against the REAL projection so no future intent
    field can be dropped the same way."""
    import json as _json

    from ouroboros.subagents import SUBAGENT_INTENT_FIELDS
    from supervisor import queue as queue_mod

    pending: list = []
    running: dict = {}
    queue_mod.init_queue_refs(pending, running, {"value": 0})
    monkeypatch.setattr(queue_mod, "QUEUE_SNAPSHOT_PATH",
                        tmp_path / "queue_snapshot.json")
    task = {"id": "t-intent-pin", "type": "task"}
    sentinels = {name: f"sentinel-{i}" for i, name in enumerate(SUBAGENT_INTENT_FIELDS)}
    task.update(sentinels)
    pending.append(task)
    assert queue_mod.persist_queue_snapshot(reason="intent-field-pin") is True
    snapshot = _json.loads((tmp_path / "queue_snapshot.json").read_text(encoding="utf-8"))
    row = snapshot["pending"][0]["task"]
    for name, value in sentinels.items():
        assert row.get(name) == value, (
            f"scheduling intent field {name!r} is missing from the pending "
            "queue-snapshot projection (supervisor/queue.py) — a restart would "
            "silently drop it")

def test_a_stored_auto_parent_lane_is_the_lane_of_record_not_the_cheapest(monkeypatch):
    """A task record can legitimately carry the literal `auto` as its effective lane —
    the supervisor falls that field back to the REQUESTED lane, which is `auto`
    whenever a task was queued without a resolved one. Its children read it as
    `parent_lane`, and `auto` is not a strength: unhandled it reached `_lane_model`
    as an unknown lane, whose fall-through was the LIGHT model. The child dropped to
    the cheapest route on this install, silently, and called the lane `auto`."""
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")

    res = subagents.resolve_subagent_lane("auto", parent_lane="auto")
    assert res.effective_lane == subagents.LANE_OF_RECORD == "main"
    assert res.model == "provider::main"
    # The fall-through itself: an unknown lane is "no lane on record", not Light.
    assert subagents._lane_model("code") == "provider::main"


def test_prompt_block_omits_the_broken_below_phrase_on_an_executor_only_delta():
    """reduced=True with NO disclosable axis is the auto-fallback case (the axis
    renderer deliberately keeps a non-pinned executor out of the list): the block
    used to render "You are running BELOW what your parent asked for: " over an
    empty list — a broken sentence duplicating dispatch_executor_note's job."""
    from types import SimpleNamespace

    from ouroboros.agent import capability_delta_prompt_block

    class _Delta:
        def as_dict(self):
            return {
                "requested_lane": "auto", "resolved_lane": "main",
                "effective_lane": "main", "derived_effort": "",
                "effective_effort": "", "requested_executor": "auto",
                "effective_executor": "native",
                "reason": "subscription_window_exhausted",
                "reduced": True, "legacy_note": "",
            }

    block = capability_delta_prompt_block(
        SimpleNamespace(delta=_Delta(), executor_resolution=None))
    assert "BELOW what your parent asked" not in block
    assert block == ""  # nothing else to say either: the executor note owns it


# ---------------------------------------------------------------- B2: light-lane nanny policy


def _harness_ready_dispatch(monkeypatch):
    """Force the executor axis to a healthy harness route without a live daemon."""
    route = subagents.DelegationRoute(route_id="codex")
    monkeypatch.setattr(
        subagents, "dispatch_executor_resolution",
        lambda task: subagents.resolve_subagent_executor("auto", route=route),
    )


def test_auto_lane_on_harness_executor_defaults_to_light_by_policy(monkeypatch):
    """B2 (poltergeist phase B): a harness-dispatched child whose request said
    `auto` is a NANNY — its own rounds are custody chores around a $0 delegated
    run, so the dispatch policy resolves it to the LIGHT lane instead of the
    parent's expensive lane, and the provenance says the POLICY answered."""
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    _harness_ready_dispatch(monkeypatch)

    dispatch = subagents.resolve_subagent_dispatch(
        {"id": "c1", "type": "task", "requested_model_lane": "auto",
         "parent_model_lane": "main"},
        task_type="task",
    )
    assert dispatch.executor == "harness"
    assert dispatch.lane.effective_lane == "light"
    assert dispatch.lane.model == "provider::cheap"
    assert dispatch.lane.provenance == "policy"
    assert dispatch.delta.as_dict()["lane_provenance"] == "policy"
    # Not a reduction relative to itself: the policy IS the resolved baseline.
    assert dispatch.lane.reduced is False


def test_explicit_lane_always_wins_over_the_harness_policy(monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    _harness_ready_dispatch(monkeypatch)

    dispatch = subagents.resolve_subagent_dispatch(
        {"id": "c2", "type": "task", "requested_model_lane": "heavy"},
        task_type="task",
    )
    assert dispatch.executor == "harness"
    assert dispatch.lane.effective_lane == "heavy"
    assert dispatch.lane.model == "provider::strong"
    assert dispatch.lane.provenance == "requested"


def test_a_required_lane_wins_over_the_harness_policy_default(monkeypatch):
    """F9 (sol #1) admission→dispatch consistency: a child ADMITTED under a
    satisfied `require_lane` constraint (auto request, parent on the required
    lane) carries `required_model_lane` on its record — and the dispatch policy
    default (auto+harness ⇒ light) must NOT apply over it. With the policy
    suppressed, `auto` inherits the parent's lane, which is exactly the lane the
    gate verified; the provenance honestly says "inherited"."""
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    _harness_ready_dispatch(monkeypatch)

    dispatch = subagents.resolve_subagent_dispatch(
        {"id": "c-req", "type": "task", "requested_model_lane": "auto",
         "parent_model_lane": "heavy", "required_model_lane": "heavy"},
        task_type="task",
    )
    assert dispatch.executor == "harness"
    assert dispatch.lane.effective_lane == "heavy"
    assert dispatch.lane.model == "provider::strong"
    assert dispatch.lane.provenance == "inherited"

    # Stored garbage in the field is ignored — the policy applies as usual.
    garbage = subagents.resolve_subagent_dispatch(
        {"id": "c-junk", "type": "task", "requested_model_lane": "auto",
         "parent_model_lane": "heavy", "required_model_lane": "warp-lane"},
        task_type="task",
    )
    assert garbage.lane.effective_lane == "light"
    assert garbage.lane.provenance == "policy"


def test_preflight_native_fallback_reresolves_without_the_harness_policy(monkeypatch):
    """F10 (sol #2, probe `native light policy`): a harness dispatch falsified at
    the toolset preflight falls back to NATIVE — and must not stay on the
    policy-light lane/cheap model the harness resolution chose. The fallback
    re-resolves lane/model/effort as a native dispatch would (parent
    inheritance), and the record, delta and envelope all describe it."""
    from types import SimpleNamespace

    from ouroboros.agent import preflight_delegate_visibility, resolve_dispatch_axes

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    _harness_ready_dispatch(monkeypatch)

    task = {"id": "c-fb", "type": "task", "delegation_role": "subagent",
            "requested_model_lane": "auto", "parent_model_lane": "heavy",
            "requested_executor": "auto"}
    dispatch = resolve_dispatch_axes(task)
    assert dispatch.lane.effective_lane == "light"  # the harness policy, pre-preflight
    assert task["model"] == "provider::cheap"

    tools = SimpleNamespace(available_tools=lambda: ["read_file", "web_search"])
    amended, changed = preflight_delegate_visibility(tools, task, dispatch)
    assert changed is True
    assert amended.executor == "native"
    # Lane and model re-resolved WITHOUT the harness policy: parent inheritance.
    assert amended.lane.effective_lane == "heavy"
    assert amended.lane.model == "provider::strong"
    assert amended.lane.provenance == "inherited"
    # Every stamped surface tells the re-resolved story.
    assert task["effective_model_lane"] == "heavy"
    assert task["model"] == "provider::strong"
    assert task["effective_executor"] == "native"
    assert task["capability_delta"]["effective_lane"] == "heavy"
    assert task["capability_delta"]["lane_provenance"] == "inherited"
    assert "delegate_tools_invisible" in task["capability_delta"]["reason"]
    assert task["capability_delta"]["reduced"] is True
    assert task["subagent_envelope"]["effective_lane"] == "heavy"
    assert task["subagent_envelope"]["model"] == "provider::strong"
    assert task["subagent_envelope"]["effective_executor"] == "native"


def test_native_child_keeps_plain_inheritance(monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "provider::strong")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "")

    dispatch = subagents.resolve_subagent_dispatch(
        {"id": "c3", "type": "task", "requested_model_lane": "auto",
         "parent_model_lane": "heavy"},
        task_type="task",
    )
    assert dispatch.executor == "native"
    assert dispatch.lane.effective_lane == "heavy"
    assert dispatch.lane.provenance == "inherited"


def test_policy_light_with_an_empty_light_slot_lands_main_and_says_so(monkeypatch):
    """The provenance names the DECISION source even when the slot outcome moves
    the effective lane: policy said light, no light slot exists, the model is
    Main — and the record must carry both facts, not blend them."""
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.delenv("OUROBOROS_MODEL_LIGHT", raising=False)
    _harness_ready_dispatch(monkeypatch)

    dispatch = subagents.resolve_subagent_dispatch(
        {"id": "c4", "type": "task", "requested_model_lane": "auto"},
        task_type="task",
    )
    assert dispatch.lane.provenance == "policy"
    assert dispatch.lane.resolved_from == "light"
    assert dispatch.lane.effective_lane == "main"
    assert dispatch.lane.model == "provider::main"


def test_switch_model_never_rewrites_the_dispatch_lane_record(monkeypatch, tmp_path):
    """B2 acceptance-model provenance: the nanny raising itself for an acceptance
    round is a ToolContext override (visible per-round in llm_usage rows), never a
    rewrite of the durable dispatch resolution — the record keeps saying which
    lane the child was DISPATCHED on."""
    from ouroboros.tools.control import _switch_model
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "provider::cheap")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    record = {"effective_model_lane": "light", "model": "provider::cheap",
              "capability_delta": {"lane_provenance": "policy"}}
    ctx.task_metadata = dict(record)

    out = _switch_model(ctx, model="provider::main")
    assert "OK: switching" in out
    assert ctx.active_model_override == "provider::main"
    # The durable dispatch record is untouched — acceptance-round provenance is
    # read from llm_usage (each round carries the REAL model), not from here.
    assert {k: ctx.task_metadata[k] for k in record} == record


def test_switch_model_refuses_an_unknown_effort_instead_of_coercing(monkeypatch, tmp_path):
    """`effort` is validated like `model`: an unknown tier is refused rather than
    silently coerced to `medium`, and the refusal takes the whole call with it —
    a model switch requested in the same call is NOT applied."""
    from ouroboros.config import EFFORT_SCALE
    from ouroboros.tools.control import _switch_model
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setattr(
        "ouroboros.llm.LLMClient.available_models",
        lambda self: ["provider::main"],
    )
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)

    out = _switch_model(ctx, model="provider::main", effort="enormous")
    assert out.startswith("⚠️ Unknown effort: enormous")
    assert ", ".join(EFFORT_SCALE) in out
    assert ctx.active_effort_override is None
    assert ctx.active_model_override is None

    # The new top tier is accepted, case/whitespace normalized like before.
    assert "OK: switching" in _switch_model(ctx, effort=" ULTRA ")
    assert ctx.active_effort_override == "ultra"

    # Whitespace-only effort is neither refused nor applied: it stays an empty
    # request and falls through to the listing, leaving no override behind.
    blank = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    assert _switch_model(blank, effort="   ").startswith("Current available models:")
    assert blank.active_effort_override is None
    assert blank.active_model_override is None
