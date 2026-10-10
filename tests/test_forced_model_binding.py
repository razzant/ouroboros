"""Acting role, exact account and active turn survive prospective and forced sends."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ouroboros import llm_claudexor, llm_substitution, loop_forced_finalization as forced, task_pacing
from ouroboros.contracts.task_contract import normalize_budget_profile
from ouroboros.loop_model_call import _RoundModelCallContext, _adopt_fallback_route, _call_round_model
from ouroboros.llm import LLMClient
from ouroboros.llm_claudexor import ModelTurnState
from ouroboros.loop_llm_call import call_llm_with_retry
from ouroboros.loop_round_limits import _RoundLimitContext
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, task_model_binding
from ouroboros.model_wait import task_model_wait_scope
from ouroboros.tools.registry import ToolRegistry
from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed
from ouroboros.send_clock import CLOCK_NOTE_PREFIX
from tests.test_llm_claudexor import MODEL, ROUTE, result, setup as setup


def _unclocked(messages):
    """A Main send ends with exactly one host clock line (``send_clock``); the rest is canonical."""
    assert str(messages[-1].get("content") or "").startswith(CLOCK_NOTE_PREFIX), messages[-1]
    assert not any(str(row.get("content") or "").startswith(CLOCK_NOTE_PREFIX) for row in messages[:-1])
    return messages[:-1]


@pytest.fixture
def acting(setup, monkeypatch):
    root, gateway, client = setup
    from ouroboros.task_results import write_task_result
    write_task_result(root, "task-one", "running", root_task_id="task-one", _attempt=1)
    gateway.results *= 4
    gateway.dispatch *= 4
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "main-only", "fallback": ["fallback-only"]}))
    # Auto is the mode that asks the route's catalog (Inline sends without asking),
    # so these sends show which account binding that read used.
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    catalogs = []

    def catalog(source, profile=None, *, requested_model=None):
        catalogs.append((source, profile, requested_model))
        return {"source": source, "credentialProfileId": profile, "models": [
            {"id": requested_model, "inputModalities": ["text"] if profile == "main-only" else ["text", "image"]}]}

    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(catalog))
    messages = [{"role": "system", "content": "Own SYSTEM"}, {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    tools = ToolRegistry(repo_dir=root.parent, drive_root=root)
    tools._ctx.task_metadata = {"configured_subagent": {"selected_subagent_id": "visual-actor",
        "route": {"kind": "api_model", "target_id": MODEL, "credential_profile_id": "actor-only"}}}
    tools._ctx.active_model = MODEL
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ctx = _RoundLimitContext(messages=deepcopy(messages), llm=client, active_model=MODEL,
        active_effort="high", max_retries=1, drive_logs=logs, task_id="task-one", round_idx=2,
        event_queue=None, accumulated_usage={}, task_type="task", active_use_local=False,
        max_rounds=2, drive_root=root, tools=tools, tool_schemas=[])
    return SimpleNamespace(root=root, gateway=gateway, client=client, messages=messages,
                           tools=tools, ctx=ctx, catalogs=catalogs)


def test_forced_call_keeps_configured_actor_account_and_image(acting):
    role, pin = task_model_binding({"task_metadata": acting.tools._ctx.task_metadata})
    call_llm_with_retry(acting.client, deepcopy(acting.messages), MODEL, [], "high", 1,
        acting.ctx.drive_logs, "task-one", 1, None, {}, model_role=role, model_account_override=pin)
    assert forced._call_forced_model_once(acting.ctx) == "Ответ 🐍"
    normal, final = [payload for payload, _ in acting.gateway.uploads]
    assert normal["account"] == {"mode": "pin", "profileId": "actor-only"}
    assert final["account"] == normal["account"]
    # The direct call binds no Main clock; the forced final is a Main send and does.
    assert normal["messages"] == _unclocked(final["messages"]) == acting.messages


def test_prospective_image_preparation_keeps_actor_pin(acting):
    _, prepared = task_pacing.prepared_wrapup_candidate(
        acting.ctx, deepcopy(acting.messages), allow_server_web_search=False)
    assert prepared == acting.messages
    assert acting.catalogs[-1][1] == "actor-only"


def test_subscription_prepared_candidate_admits_the_actual_forced_send(acting):
    acting.ctx.messages = [{"role": "user", "content": "Please finish"}]
    request, prepared = task_pacing.prepared_wrapup_candidate(
        acting.ctx, deepcopy(acting.ctx.messages), allow_server_web_search=False)
    assert request.provider == "claudexor"
    text = forced._call_forced_model_once(acting.ctx, initial_messages=prepared, admitted_request=request)
    assert text == "Ответ 🐍", acting.ctx.accumulated_usage
    assert len(acting.gateway.creates) == 1


def test_subscription_prospective_and_send_share_the_install_cache_key(acting, monkeypatch):
    """The admitted candidate declares the same cache affinity as the real send."""
    from ouroboros import llm_claudexor

    acting.ctx.messages = [{"role": "user", "content": "Please finish"}]
    built = []
    build = llm_claudexor._request

    def record(target, messages, tools, parameters):
        payload = build(target, messages, tools, parameters)
        built.append(deepcopy(payload["options"]))
        return payload

    monkeypatch.setattr(llm_claudexor, "_request", record)
    request, prepared = task_pacing.prepared_wrapup_candidate(
        acting.ctx, deepcopy(acting.ctx.messages), allow_server_web_search=False)
    assert forced._call_forced_model_once(acting.ctx, initial_messages=prepared,
                                          admitted_request=request) == "Ответ 🐍"
    # One install-scoped key per model (not the execution id): the priced
    # candidate and the dispatched payload must still carry the same one.
    shared_key = llm_claudexor.cache_key_for_model(acting.ctx.active_model)
    assert shared_key and acting.ctx.accumulated_usage["execution_id"] not in shared_key
    assert len(built) == 2 and built[0] == built[1] == {"reasoningEffort": "high", "cacheKey": shared_key}
    assert acting.gateway.uploads[0][0]["options"] == built[1]


def test_prospective_build_reads_the_failed_profile_without_spending_it(acting):
    """Only the dispatch consumes the one-shot fact, so its candidate still admits it."""
    acting.ctx.accumulated_usage["execution_id"] = "execution-refusal"
    acting.ctx.messages = [result()["message"], {"role": "user", "content": "Please finish"}]
    # The fact is keyed by the affinity the dispatch declares: the install-scoped
    # Codex key, no longer the execution id.
    shared_key = llm_claudexor.cache_key_for_model(MODEL)
    token = llm_substitution._FAILED_PROFILE.set((shared_key, "codex", "exact-model", "account-a"))
    try:
        with task_model_wait_scope(task={"id": "task-one", "_attempt": 1}, drive_root=acting.root,
                                   event_queue=None, worker_slot_held=True) as wait:
            wait.overrides["subagent:visual-actor"] = {
                "model": MODEL, "use_local": False, "model_account_override": ""}
            request, prepared = task_pacing.prepared_wrapup_candidate(
                acting.ctx, deepcopy(acting.ctx.messages), allow_server_web_search=False)
            assert llm_substitution._FAILED_PROFILE.get()[3] == "account-a"  # observed, not spent
            assert forced._call_forced_model_once(acting.ctx, initial_messages=prepared,
                                                  admitted_request=request) == "Ответ 🐍"
        assert llm_substitution._FAILED_PROFILE.get() == ()  # the dispatch spent it, exactly once
    finally:
        llm_substitution._FAILED_PROFILE.reset(token)
    payload = acting.gateway.uploads[0][0]
    assert payload["account"] == {"mode": "auto"}  # the refused account is not preferred back
    assert payload["options"] == {"reasoningEffort": "high", "cacheKey": shared_key}


@pytest.mark.parametrize("shape", ["mid_round_image", "late_system_notice"])
def test_subscription_prospective_and_send_share_transcript_normalization(acting, shape):
    if shape == "mid_round_image":
        assistant = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "shot", "type": "function", "function": {"name": "browser", "arguments": "{}"}}],
            "nativeContinuation": {"format": "codex.responses.v1", "payload": [
                {"type": "reasoning", "encrypted_content": "opaque+==\r\n"}]}}
        messages = [{"role": "system", "content": "Own SYSTEM"},
            {"role": "user", "content": "Open the page"}, assistant,
            deepcopy(acting.messages[-1]),
            {"role": "tool", "tool_call_id": "shot", "content": "Screenshot attached"}]
    else:
        messages = [{"role": "system", "content": "Own SYSTEM"},
            {"role": "user", "content": "Start"}, {"role": "assistant", "content": "Started"},
            {"role": "system", "content": "Runtime notice"}]
    messages.append({"role": "user", "content": "Finish from the current evidence"})
    original = deepcopy(messages)
    normalized = LLMClient._normalize_system_message_placement(messages)
    assert normalized != original
    acting.ctx.messages = messages
    request, prepared = task_pacing.prepared_wrapup_candidate(
        acting.ctx, messages, allow_server_web_search=False)
    assert forced._call_forced_model_once(acting.ctx, initial_messages=prepared,
                                          admitted_request=request) == "Ответ 🐍"
    sent = acting.gateway.uploads[0][0]["messages"]
    # The answered send's clock line joins the canonical transcript; nothing else changes.
    assert _unclocked(sent) == normalized and _unclocked(messages) == original
    assert len(acting.gateway.creates) == 1
    if shape == "mid_round_image":
        assert sent[3]["role"] == "tool" and sent[3]["tool_call_id"] == "shot"
        assert sent[4]["content"][0]["type"] == "image_url"
        assert sent[2]["nativeContinuation"] == assistant["nativeContinuation"]


@pytest.mark.parametrize("pin", ["temporary-actor", ""])
def test_forced_and_prospective_preserve_task_actor_override_including_auto(acting, pin):
    task = {"id": "task-one", "_attempt": 1}
    with task_model_wait_scope(task=task, drive_root=acting.root,
                              event_queue=None, worker_slot_held=True) as wait:
        wait.overrides["subagent:visual-actor"] = {
            "model": MODEL, "use_local": False, "model_account_override": pin}
        request, prepared = task_pacing.prepared_wrapup_candidate(
            acting.ctx, deepcopy(acting.messages), allow_server_web_search=False)
        assert prepared == acting.messages
        assert forced._call_forced_model_once(acting.ctx, initial_messages=prepared,
                                              admitted_request=request) == "Ответ 🐍"
    payload = acting.gateway.uploads[0][0]
    assert payload["account"] == ({"mode": "pin", "profileId": pin} if pin else {"mode": "auto"})
    assert _unclocked(payload["messages"]) == acting.messages
    assert acting.catalogs[-1][1] == (pin or None)


def test_adopted_fallback_plan_keeps_its_role_not_actor_or_main(acting):
    plan = SimpleNamespace(model_role="fallback:0", model_route={"credentialProfileId": "observed-not-a-pin"})
    _adopt_fallback_route(acting.ctx, acting.tools, MODEL, False, acting.ctx.messages,
        deepcopy(acting.messages), plan, "max", [], acting.ctx.accumulated_usage)
    assert acting.tools._ctx.context_fit_plan is plan
    request, prepared = task_pacing.prepared_wrapup_candidate(
        acting.ctx, deepcopy(acting.messages), allow_server_web_search=False)
    assert forced._call_forced_model_once(acting.ctx, initial_messages=prepared,
                                          admitted_request=request) == "Ответ 🐍"
    assert acting.gateway.uploads[0][0]["account"] == {"mode": "pin", "profileId": "fallback-only"}
    assert prepared == acting.messages and acting.catalogs[-1][1] == "fallback-only"


def test_unconfigured_task_still_binds_main(acting):
    acting.tools._ctx.task_metadata = {}
    acting.ctx.messages = [{"role": "user", "content": "finish"}]
    assert forced._call_forced_model_once(acting.ctx) == "Ответ 🐍"
    assert acting.gateway.uploads[0][0]["account"] == {"mode": "pin", "profileId": "main-only"}


def test_task_binding_priority_is_role_then_plan_then_frozen_actor(acting):
    task = {"task_metadata": acting.tools._ctx.task_metadata}
    plan = SimpleNamespace(model_role="fallback:0", model_route={"credentialProfileId": "observed-only"})
    assert task_model_binding(task) == ("subagent:visual-actor", "actor-only")
    assert task_model_binding(task, context_fit_plan=plan) == ("fallback:0", None)
    assert task_model_binding({**task, "model_role": "main"}, context_fit_plan=plan) == ("main", None)
    assert task_model_binding(task, overrides={"subagent:visual-actor": {
        "model_account_override": ""}}) == ("subagent:visual-actor", "")


@pytest.mark.parametrize("fallback", [False, True])
def test_browser_attachment_is_canonical_and_asks_no_catalog(acting, fallback):
    """The screenshot joins the canonical turn on any binding; which account's catalog
    decides what a route receives is the send's question (pinned by the ordinary
    and prospective sends above), never the attach step's."""
    from ouroboros.tools.browser import _inject_native_screenshot

    ctx = acting.tools._ctx
    ctx.messages = []
    ctx.task_metadata["configured_subagent"]["route"]["credential_profile_id"] = "main-only"
    with task_model_wait_scope(task={"id": "task-one", "_attempt": 1}, drive_root=acting.root,
                              event_queue=None, worker_slot_held=True) as wait:
        if fallback:
            ctx.context_fit_plan = SimpleNamespace(model_role="fallback:0")
        else:
            wait.overrides["subagent:visual-actor"] = {
                "model": MODEL, "use_local": False, "model_account_override": "temporary-actor"}
        _inject_native_screenshot(ctx, "QUFBQQ==")
    assert ctx.messages and ctx.messages[-1]["content"][-1]["type"] == "image_url"
    assert acting.catalogs == []


@pytest.mark.parametrize("fallback", [False, True])
def test_ordinary_round_uses_the_same_actor_or_active_plan_binding(acting, fallback):
    if fallback:
        acting.tools._ctx.context_fit_plan = SimpleNamespace(model_role="fallback:0",
            model_route=deepcopy(acting.gateway.results[0]["route"]))
    ctx = _RoundModelCallContext(llm=acting.client, messages=deepcopy(acting.messages),
        tools=acting.tools, context_fit_plan=None, active_model=MODEL, tool_schemas=[],
        active_effort="high", max_retries=1, drive_logs=acting.ctx.drive_logs, task_id="task-one",
        round_idx=1, event_queue=None, accumulated_usage={}, task_type="task",
        active_use_local=False, active_context_mode="max", drive_root=acting.root, attempt_cap=1)
    message, _, _ = _call_round_model(ctx)
    assert message["content"] == "Ответ 🐍"
    payload = acting.gateway.uploads[0][0]
    assert payload["account"] == {"mode": "pin", "profileId": "fallback-only" if fallback else "actor-only"}
    from tests.test_subscription_main_wait import _without_context_facts

    assert _without_context_facts(_unclocked(payload["messages"]), physical=True) == acting.messages


TURN = {"route": ROUTE, "format": "codex.turn.v1", "payload": {"turnState": "the-running-turn"}}
LANDED = {"route": ROUTE, "format": "codex.turn.v1", "payload": {"turnState": "after-the-forced-send"}}


@pytest.fixture
def turn_engine(monkeypatch):
    """A serving engine whose strict request schema accepts the active-turn field."""
    from ouroboros import config, llm_claudexor

    monkeypatch.setattr(llm_claudexor, "owned_engine_version",
                        lambda: config.CLAUDEXOR_MODEL_TURN_STATE_MIN_VERSION)


def _admitted_wrapup(acting, envelope=None, *, opted=True):
    """Arm the loop's turn slot, then price the candidate the forced send must match."""
    acting.ctx.messages = [{"role": "user", "content": "Please finish"}]
    slot = ModelTurnState(deepcopy(envelope)) if opted else None
    acting.tools._ctx.model_turn_state = slot
    request, prepared = task_pacing.prepared_wrapup_candidate(
        acting.ctx, deepcopy(acting.ctx.messages), allow_server_web_search=False)
    return slot, request, prepared


@pytest.mark.parametrize("envelope", [None, TURN])
def test_the_admitted_forced_send_carries_the_priced_active_turn(acting, turn_engine, envelope):
    """An empty opted-in slot and a live one both reach the wire as priced."""
    slot, request, prepared = _admitted_wrapup(acting, envelope)
    assert forced._call_forced_model_once(
        acting.ctx, initial_messages=prepared, admitted_request=request) == "Ответ 🐍"
    payload = acting.gateway.uploads[0][0]
    assert payload["nativeContinuation"] == envelope and len(acting.gateway.creates) == 1
    # A result without an envelope leaves the turn stateless rather than stale.
    assert acting.tools._ctx.model_turn_state is slot and slot.envelope is None


def test_the_dispatched_forced_result_replaces_the_loop_slot(acting, turn_engine):
    acting.gateway.results = [{**row, "nativeContinuation": deepcopy(LANDED)}
                              for row in acting.gateway.results]
    slot, request, prepared = _admitted_wrapup(acting, TURN)
    assert forced._call_forced_model_once(
        acting.ctx, initial_messages=prepared, admitted_request=request) == "Ответ 🐍"
    assert acting.tools._ctx.model_turn_state is slot and slot.envelope == LANDED


@pytest.mark.parametrize("version", ["3.10.3", ""])
def test_an_older_or_unobserved_engine_keeps_the_legacy_candidate_and_send(acting, monkeypatch, version):
    """No field at all, and the same bytes a slotless caller would have priced."""
    from ouroboros import llm_claudexor, send_clock

    monkeypatch.setattr(llm_claudexor, "owned_engine_version", lambda: version)
    # Each candidate carries the host clock line to the second; price both at one instant.
    instant = send_clock._now()
    monkeypatch.setattr(send_clock, "_now", lambda: instant)
    _none, legacy, _messages = _admitted_wrapup(acting, opted=False)
    _slot, request, prepared = _admitted_wrapup(acting, TURN)
    assert request.candidate_raw_sha256 == legacy.candidate_raw_sha256
    assert request.candidate_raw_size_bytes == legacy.candidate_raw_size_bytes
    assert forced._call_forced_model_once(
        acting.ctx, initial_messages=prepared, admitted_request=request) == "Ответ 🐍"
    assert "nativeContinuation" not in acting.gateway.uploads[0][0]


def test_a_candidate_priced_on_another_turn_is_still_refused_before_dispatch(acting, turn_engine):
    """The identity fence stays a fence; the turn slot is not exempt from it."""
    slot, request, prepared = _admitted_wrapup(acting, TURN)
    slot.envelope = deepcopy(LANDED)
    with pytest.raises(PhysicalAttemptPreconditionFailed):
        forced._call_forced_model_once(
            acting.ctx, initial_messages=prepared, admitted_request=request)
    assert not acting.gateway.creates


def test_the_budget_soft_landing_wraps_up_with_the_model_not_the_host_text(acting, turn_engine):
    """loop_budget's exhausted-ceiling rail reaches synthesis with a turn armed."""
    from ouroboros import loop as loop_module

    acting.ctx.messages = [{"role": "user", "content": "Please finish"}]
    acting.ctx.llm_trace = {}
    acting.tools._ctx.model_turn_state = ModelTurnState(deepcopy(TURN))
    ceiling = task_pacing.resolve_cost_ceiling(100.0, normalize_budget_profile({"cost_hard_stop_pct": 50}), root_cap_usd=0.5)
    text, _usage, _trace = loop_module._soft_land_exhausted_ceiling(acting.ctx, ceiling)
    assert "Ответ 🐍" in text and "no working room" not in text
    assert acting.gateway.uploads[0][0]["nativeContinuation"] == TURN


def test_a_failed_probe_between_the_candidate_and_the_send_keeps_the_priced_bytes(acting, monkeypatch):
    """The identity mismatch that sent a forced final to host text.

    ``prepared_wrapup_candidate`` prices the forced final and the real
    ``_request`` builds the send bytes a moment later. Both read the serving
    engine version off the ONE owned-daemon singleton the Accounts panel polls
    concurrently in the same process, and a failed probe used to blank it — so
    the admitted candidate carried ``nativeContinuation``, the send dropped it,
    the pre-dispatch identity fence refused the send and the task lost its
    model wrap-up. The PROVEN version survives the probe, so both reads agree.
    """
    from ouroboros import claudexor_daemon, config

    manager = claudexor_daemon.OwnedClaudexorDaemon()
    manager._engine_version = config.CLAUDEXOR_MODEL_TURN_STATE_MIN_VERSION
    manager._proven_engine_version = config.CLAUDEXOR_MODEL_TURN_STATE_MIN_VERSION
    monkeypatch.setattr(claudexor_daemon, "get_owned_daemon", lambda: manager)
    slot, request, prepared = _admitted_wrapup(acting, TURN)

    # The real failure branch a concurrent poll of an unprovisioned or
    # unreachable home takes: it blanks the liveness field and nothing else.
    monkeypatch.setattr(claudexor_daemon, "owned_daemon_provisioned", lambda: False)
    assert manager._classify_liveness() == (None, "not_provisioned", "")
    assert manager._engine_version == ""

    assert forced._call_forced_model_once(
        acting.ctx, initial_messages=prepared, admitted_request=request) == "Ответ 🐍"
    payload = acting.gateway.uploads[0][0]
    assert payload["nativeContinuation"] == TURN and len(acting.gateway.creates) == 1
    assert acting.tools._ctx.model_turn_state is slot and slot.envelope is None
