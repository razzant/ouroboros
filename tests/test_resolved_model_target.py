"""ABI-4 ``ResolvedModelTarget`` — the typed resolved-model destination.

Pins immutability, value identity and construction at each existing resolution seam
(the cross-model fallback ladder, the reviewer model lists, the delegated
route), and the consumer-sweep grep pins (no comma/at re-parsing beside a seam
that already yields the dataclass).
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

from ouroboros.config import (
    ResolvedModelTarget,
    fallback_candidate_targets,
    get_fallback_models,
    resolve_model_target,
    resolved_review_model_target,
)
from ouroboros.subagents import DelegationRoute, parse_subagent_harness

REPO = pathlib.Path(__file__).resolve().parent.parent

_PROVIDER_CREDENTIAL_ENV = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MINIMAX_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY", "GIGACHAT_CREDENTIALS", "GIGACHAT_USER",
    "GIGACHAT_PASSWORD", "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL",
    "OPENAI_BASE_URL",
)


def _clear_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _PROVIDER_CREDENTIAL_ENV:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Contract: frozen + slots, value identity, typed sentinels.
# ---------------------------------------------------------------------------


def test_frozen_with_slots():
    target = ResolvedModelTarget(model_id="m", provider_route="openrouter")
    with pytest.raises(dataclasses.FrozenInstanceError):
        target.model_id = "other"  # type: ignore[misc]
    assert hasattr(ResolvedModelTarget, "__slots__")
    assert not hasattr(target, "__dict__")


def test_value_identity_equality_and_hash():
    a = ResolvedModelTarget("m", "openrouter", "cred", "high", 128000)
    b = ResolvedModelTarget("m", "openrouter", "cred", "high", 128000)
    c = ResolvedModelTarget("m", "openrouter", "cred", "low", 128000)
    assert a == b and hash(a) == hash(b)
    assert a != c
    assert len({a, b, c}) == 2


def test_sentinels_are_typed_never_none():
    """Absent facts are ""/0, and no field defaults to None (design rule)."""
    target = ResolvedModelTarget(model_id="m", provider_route="openrouter")
    assert target.credential_ref == "" and target.effort == "" and target.context_window == 0
    for field in dataclasses.fields(ResolvedModelTarget):
        assert field.default is not None, field.name


def test_constructor_normalizes_at_the_seam():
    target = resolve_model_target("  openai::gpt-x  ", effort=" high ", context_window=-5)
    assert target == ResolvedModelTarget(
        model_id="openai::gpt-x", provider_route="openai",
        credential_ref="", effort="high", context_window=0,
    )
    assert resolve_model_target("plain-model").provider_route == "openrouter"
    assert resolve_model_target("mymodel (local)").provider_route == "local"


# ---------------------------------------------------------------------------
# Seam 1: the cross-model fallback candidate ladder.
# ---------------------------------------------------------------------------


def test_fallback_ladder_is_a_typed_view_of_the_chain_ssot(monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai::a, b ,mymodel (local),b")
    candidates = fallback_candidate_targets("")
    assert isinstance(candidates, tuple)
    assert [c.model_id for c in candidates] == get_fallback_models("")
    # provider_route stays the "" sentinel DELIBERATELY: the chain's dispatch
    # lane is the loop's single global USE_LOCAL_FALLBACK flag (pre-existing
    # contract), so a per-candidate route would be a fabricated fact no
    # dispatcher consumes (adversarial finding 7 disposition).
    assert [c.provider_route for c in candidates] == ["", "", ""]
    # The active model collapses out of the ladder exactly as in the SSOT list.
    assert [c.model_id for c in fallback_candidate_targets("b")] == get_fallback_models("b")
    # Ladder targets keep the "" effort sentinel: the round owns active effort.
    assert all(c.effort == "" and c.context_window == 0 for c in candidates)


@pytest.mark.parametrize("captured_local", [False, True])
def test_fallback_dispatch_lane_stays_the_global_flag(tmp_path, monkeypatch, captured_local):
    """Every candidate uses the task's shared fallback flag, including after a save."""
    from types import SimpleNamespace
    from ouroboros import fallback_cooldown, loop, loop_model_call
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot

    settings = {"USE_LOCAL_FALLBACK": str(captured_local).lower(),
                "OUROBOROS_MODEL_FALLBACKS": "remote-model,other (local)"}
    snapshot = task_settings_snapshot(settings, settings)
    monkeypatch.setenv("USE_LOCAL_FALLBACK", str(not captured_local).lower())
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    dispatched = []

    def call(ctx):
        dispatched.append((ctx.active_model, ctx.active_use_local))
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", call)
    with task_settings_scope(snapshot):
        loop_model_call._run_cross_model_fallback_chain(
            llm=None, ctx=SimpleNamespace(), tools=SimpleNamespace(_ctx=SimpleNamespace()),
            messages=[], active_model="primary", active_use_local=False, tool_schemas=[],
            active_effort="high", max_retries=1, drive_logs=tmp_path / "logs", task_id="t",
            round_idx=1, event_queue=None, accumulated_usage={}, task_type="task",
            emit_progress=lambda _text, *, incident=None: None,
            context_fit_plan=None, active_context_mode="max",
        )
    assert dispatched == [("remote-model", captured_local), ("other (local)", captured_local)]


def test_an_attempt_carries_its_incident_reason_and_pin_and_only_adoption_names_the_active_route(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import fallback_cooldown, loop, loop_model_call

    fallback = "claudexor::codex=fallback"
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", fallback)
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", '{"fallback":["account-a"]}')
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))

    def answered(call):  # the send records its request's account in the call meta, as the real producer does
        call.accumulated_usage["_last_llm_call_meta"] = {"model": call.active_model, "requested_profile": "account-a"}
        return {"role": "assistant"}, 0, "max"

    monkeypatch.setattr(loop, "_call_round_model", answered)
    progress = []
    ctx = SimpleNamespace(active_model="primary", active_use_local=False)
    tools = SimpleNamespace(_ctx=SimpleNamespace())

    loop_model_call._run_cross_model_fallback_chain(
        llm=None, ctx=ctx, tools=tools, messages=[], active_model="primary",
        active_use_local=False, tool_schemas=[], active_effort="high", max_retries=1,
        drive_logs=tmp_path / "logs", task_id="task-7", round_idx=3, event_queue=None,
        accumulated_usage={"_last_llm_error_kind": "provider_transient"}, task_type="task",
        emit_progress=lambda text, *, incident=None: progress.append((text, incident)),
        context_fit_plan=None, active_context_mode="max",
    )

    assert progress == [
        (f"⚡ Fallback: trying {fallback} (account: account-a; pinned account: siblings were not tried) "
         "after a temporary provider failure on primary.",
         {"task_incident": "model_lane_attempt", "toast_once": f"task-7:model_lane_attempt:3:{fallback}"}),
        (f"⚡ Fallback: {fallback} (account: account-a) is now the active route; it answered.", None),
    ]


def test_each_attempt_names_the_candidate_that_just_failed_in_its_own_words(tmp_path, monkeypatch):
    """Two refused candidates, then an answer. Each attempt names the model actually tried, that
    model's own failure class in words and only the sentence its own provider left (stamped by the
    real owner, as the call owner stamps it beside the attempt's row); every attempt is announced
    before its dispatch and keeps its own toast key. Only the answering route is adopted, once."""
    from types import SimpleNamespace
    from ouroboros import fallback_cooldown, loop, loop_model_call
    from ouroboros.loop_transport import stamp_owner_provider_message

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "alt-a,alt-b,alt-c")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(fallback_cooldown, "mark_cooldown", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    cut = ("The provider kept only the start of this sentence about the refused tool payload. " * 9)[:300]
    failures = {  # each failed call's typed kind and failure, as the call owner stamps them
        "alt-a": ("auth_error", RuntimeError("AuthenticationError('401 invalid_api_key')")),
        "alt-b": ("provider_body_error", {"message": cut, "message_truncated": True}),
    }
    usage = {"_last_llm_error_kind": "bad_request", "_last_llm_provider_message": "Upstream request failed.",
             "_last_llm_failure": {"model": "primary", "failure_code": "bad_request"}}
    progress, dispatched = [], []

    def call(ctx):
        dispatched.append((ctx.active_model, len(progress)))
        if ctx.active_model not in failures:
            return {"role": "assistant", "content": "done"}, 0.0, "max"
        usage["_last_llm_error_kind"], failure = failures[ctx.active_model]
        usage["_last_llm_failure"] = {"model": ctx.active_model, "failure_code": usage["_last_llm_error_kind"]}
        stamp_owner_provider_message(usage, failure)
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", call)
    ctx = SimpleNamespace(active_model="primary", active_use_local=False)
    msg, model, *_rest = loop_model_call._run_cross_model_fallback_chain(
        llm=None, ctx=ctx, tools=SimpleNamespace(_ctx=SimpleNamespace()), messages=[], active_model="primary",
        active_use_local=False, tool_schemas=[], active_effort="high", max_retries=1,
        drive_logs=tmp_path / "logs", task_id="task-7", round_idx=3, event_queue=None,
        accumulated_usage=usage, task_type="task",
        emit_progress=lambda text, *, incident=None: progress.append((text, incident)),
        context_fit_plan=None, active_context_mode="max",
    )

    assert [text.removeprefix("⚡ ") for text, _incident in progress] == [
        'Fallback: trying alt-a after a rejected request on primary. The provider said: "Upstream request failed."',
        "Fallback: trying alt-b after an authentication or authorization refusal on alt-a; "
        "sign-in is still needed there.",
        "Fallback: trying alt-c after an error in the provider's response on alt-b. "
        f'The provider said: "{" ".join(cut.split())}…" (shortened). '
        "The task's Logs keep the host's record of this failure.",
        "Fallback: alt-c is now the active route; it answered.",
    ]
    assert [incident for _text, incident in progress] == [
        {"task_incident": "model_lane_attempt", "toast_once": f"task-7:model_lane_attempt:3:{name}"}
        for name in ("alt-a", "alt-b", "alt-c")] + [None], "an outcome row is ordinary progress: no incident, no toast"
    assert dispatched == [("alt-a", 1), ("alt-b", 2), ("alt-c", 3)], "each attempt is announced before its dispatch"
    assert msg["content"] == "done" and model == ctx.active_model == "alt-c"


def _walk(tmp_path, usage, progress, *, active_model="primary", tool_ctx=None, messages=None):
    from types import SimpleNamespace

    from ouroboros import loop_model_call

    ctx = SimpleNamespace(active_model=active_model, active_use_local=False)
    tools = SimpleNamespace(_ctx=tool_ctx or SimpleNamespace())
    result = loop_model_call._run_cross_model_fallback_chain(
        llm=None, ctx=ctx, tools=tools, messages=[] if messages is None else messages, active_model=active_model, active_use_local=False,
        tool_schemas=[], active_effort="high", max_retries=1, drive_logs=tmp_path, task_id="task-7", round_idx=3,
        event_queue=None, accumulated_usage=usage, task_type="task",
        emit_progress=lambda text, *, incident=None: progress.append((text.removeprefix("⚡ "), incident)),
        context_fit_plan=None, active_context_mode="max")
    return result, ctx, tools


@pytest.mark.parametrize("sentence", ["The route refused this.", ""])
def test_a_walk_nothing_adopted_ends_in_one_plain_row_on_its_last_attempt(tmp_path, monkeypatch, sentence):
    """Each failure opens the next attempt's row, so only the last one needs the end row. It reports that attempt
    alone, in its own words: no route claimed or denied, no reason the walk stopped, no incident or toast."""
    from ouroboros import fallback_cooldown, loop
    from ouroboros.loop_transport import stamp_owner_provider_message

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "alt-a,alt-b")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    usage, progress = {"_last_llm_error_kind": "rate_limit"}, []

    def refused(ctx):
        usage.update(_last_llm_error_kind="bad_request",
                     _last_llm_failure={"model": ctx.active_model, "failure_code": "bad_request"})
        stamp_owner_provider_message(usage, {"message": f"{ctx.active_model}: {sentence}"} if sentence else None)
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", refused)
    (msg, model, *_rest), ctx, tools = _walk(tmp_path, usage, progress)
    assert msg is None and model == "primary" and ctx.active_model == "primary"
    assert [incident is not None for _text, incident in progress] == [True, True, False]
    quote = f' The provider said: "alt-a: {sentence}"' if sentence else ""
    last = f' The provider said: "alt-b: {sentence}"' if sentence else ""
    assert [text for text, _incident in progress] == [
        "Fallback: trying alt-a after a rate limit on primary.",
        f"Fallback: trying alt-b after a rejected request on alt-a.{quote}",
        f"Fallback: the last attempt, on alt-b, ended without a usable reply after a rejected request.{last}",
    ]
    assert not getattr(tools._ctx, "_route_facts_pending", "")


def test_a_clock_spent_before_the_next_route_is_no_cause_of_the_last_attempt(tmp_path, monkeypatch):
    """The walk's own pre-call clock stops it with nothing stamped: the row keeps to the last attempt's own failure."""
    import time

    from ouroboros import fallback_cooldown, loop

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "alt-a,alt-b")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    usage, progress, dialed = {"_last_llm_error_kind": "rate_limit"}, [], []
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: time.time() - 1 if dialed else None)

    def refused(ctx):
        dialed.append(ctx.active_model)
        usage.update(_last_llm_error_kind="bad_request",
                     _last_llm_failure={"model": ctx.active_model, "failure_code": "bad_request"})
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", refused)
    _walk(tmp_path, usage, progress)
    assert dialed == ["alt-a"]
    assert [text for text, _incident in progress] == [
        "Fallback: trying alt-a after a rate limit on primary.",
        "Fallback: the last attempt, on alt-a, ended without a usable reply after a rejected request."]


def test_the_walks_end_never_lends_the_last_route_an_earlier_routes_sentence(tmp_path, monkeypatch):
    """The real deadline refusal stamps its kind, writes no failure row and leaves the sentence of the attempt
    before it: that sentence stays with its own route, and the refused route gets the minimal observation."""
    import time

    from ouroboros import fallback_cooldown, loop
    from ouroboros.loop_llm_call import _deadline_not_dispatched

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "alt-a,alt-b")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    usage, progress, dispatched = {"_last_llm_error_kind": "bad_request", "_last_llm_provider_message": "Only primary said this.",
                                   "_last_llm_failure": {"model": "primary", "failure_code": "bad_request"}}, [], []

    def out_of_time(ctx):
        dispatched.append(ctx.active_model)
        usage.pop("_last_llm_failure", None)  # as the call owner clears it before admission
        assert _deadline_not_dispatched(time.time() - 1, usage, tmp_path, task_id="task-7", model=ctx.active_model, round_idx=3)
        return None, None, "max"

    monkeypatch.setattr(loop, "_call_round_model", out_of_time)
    _walk(tmp_path, usage, progress)
    assert dispatched == ["alt-a"], "the walk stops at a spent deadline"
    assert usage["_last_llm_provider_message"] == "Only primary said this.", "the stale sentence this row must not borrow"
    assert [text for text, _incident in progress] == [
        'Fallback: trying alt-a after a rejected request on primary. The provider said: "Only primary said this."',
        "Fallback: the last attempt, on alt-a, ended without a usable reply; no cause of its own is known here.",
    ]


@pytest.mark.parametrize("eligible", [False, True], ids=["fenced", "eligible"])
def test_the_walks_end_keeps_an_unknown_outcome_with_its_own_attempt(tmp_path, monkeypatch, eligible):
    """Each route's call ends with an outcome nobody knows. Fenced, the walk stops there and that operation may still
    be readable; eligible, the next attempt's row names the earlier attempt. The end row speaks of the last attempt
    alone: no ending, no earlier attempt, nothing unknown lent to the primary's known quota refusal."""
    from ouroboros import fallback_cooldown, loop, loop_transport

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "alt-a,alt-b")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(fallback_cooldown, "mark_cooldown", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    monkeypatch.setattr(loop_transport, "new_generation_after_unknown", lambda *_a, **_k: eligible)
    usage = {"_last_llm_error_kind": "quota_exhausted", "_last_llm_failure": {"model": "primary", "failure_code": "quota_exhausted"}}
    progress = []

    def unknown(ctx):
        usage.update(_last_llm_error_kind="provider_outcome_unknown",
                     _last_llm_failure={"model": ctx.active_model, "failure_code": "provider_outcome_unknown"},
                     _pending_transport_outcome={"physical_attempt_id": f"attempt-{ctx.active_model}"})
        return None, None, "max"

    monkeypatch.setattr(loop, "_call_round_model", unknown)
    (msg, model, *_rest), _ctx, _tools = _walk(tmp_path, usage, progress)
    assert (msg, model, [text for text, _incident in progress]) == (None, "primary", [
        "Fallback: trying alt-a after a quota or billing refusal on primary.",
        *(["Fallback: trying alt-b after an attempt with no known outcome on alt-a; the earlier attempt’s outcome and cost "
           "stay unknown."] if eligible else []),
        f"Fallback: the last attempt, on {'alt-b' if eligible else 'alt-a'}, has no known outcome; its outcome and cost stay unknown.",
    ])


@pytest.mark.parametrize("answered", [True, False], ids=["answered", "reply_allowance_spent"])
def test_a_return_to_the_primary_and_a_spent_reply_say_so(tmp_path, monkeypatch, answered):
    """A walk from an acting fallback dials the primary first. Adoption is the host's choice of route; a
    reply that spent its allowance is adopted too and is reported as no usable answer, never as an answer."""
    from types import SimpleNamespace

    from ouroboros import fallback_cooldown, loop
    from ouroboros import loop_output_exhaustion as recovery

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "alt-a,alt-b")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    usage, progress = {"_last_llm_error_kind": "rate_limit"}, []

    def primary_replies(ctx):
        if answered:
            return {"role": "assistant", "content": "back"}, 0.0, "max"
        usage.update(_last_llm_error_kind=recovery.KIND, _last_llm_failure={"model": "primary", "failure_code": recovery.KIND})
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", primary_replies)
    (msg, model, *_rest), _ctx, tools = _walk(
        tmp_path, usage, progress, active_model="alt-a",
        tool_ctx=SimpleNamespace(primary_route={"model": "primary", "use_local": False, "role": "main"}))
    assert (msg is not None) is answered and model == "primary"
    assert [incident is not None for _text, incident in progress] == [True, False]
    assert progress[0][0] == "Fallback: trying primary (primary) after a rate limit on alt-a."
    assert progress[1][0] == ("Fallback: back on the primary route primary; it answered." if answered else
                              "Fallback: back on the primary route primary, with no usable answer after a reply "
                              "that reached its length limit before any visible output.")
    assert tools._ctx._route_facts_pending == "", "the primary needs no switch facts"


def test_a_route_already_spent_is_skipped_before_any_preparation(tmp_path, monkeypatch):
    """Output recovery's own binding check, before and during the walk (a send claiming a later row's binding).
    A skipped route is neither announced nor prepared: the next attempt still follows the route really tried,
    and nothing of the skipped one reaches the transcript, the view, the cooldown or the usage record."""
    from ouroboros import fallback_cooldown, loop
    from ouroboros import loop_output_exhaustion as recovery

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "alt-a,alt-b,alt-c,alt-d")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    cooled = []
    monkeypatch.setattr(fallback_cooldown, "mark_cooldown", lambda *binding, **_k: cooled.append(binding[0]))
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    prepared, dialed = [], []

    def rebind(_plan, _tools, _messages, *, model, **_kwargs):
        prepared.append(model)
        return None, "max"

    def call(ctx):
        dialed.append(ctx.active_model)
        if ctx.active_model == "alt-d":
            return {"role": "assistant", "content": "done"}, 0.0, "max"
        usage["_last_llm_error_kind"] = "provider_transient"
        usage["_last_llm_failure"] = {"model": ctx.active_model, "failure_code": "provider_transient"}
        usage[recovery.KEY]["tried"].append(["alt-c", False, ""])
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_rebind_context_fit_plan", rebind)
    monkeypatch.setattr(loop, "_call_round_model", call)
    usage = {"_last_llm_error_kind": "provider_outcome_unknown",
             recovery.KEY: {"stage": "fallback", "owner_revision": 0, "tried": [["alt-a", False, ""]]}}
    progress, messages = [], []
    (msg, model, *_rest), _ctx, _tools = _walk(tmp_path, usage, progress, messages=messages)
    assert msg["content"] == "done" and model == "alt-d"
    assert prepared == dialed == ["alt-b", "alt-d"]
    assert [text for text, _incident in progress] == [
        "Fallback: trying alt-b after an attempt with no known outcome on primary; the earlier attempt’s outcome and cost stay unknown.",
        "Fallback: trying alt-d after a temporary provider failure on alt-b.",
        "Fallback: alt-d is now the active route; it answered.",
    ]
    assert not {"alt-a", "alt-c"} & set(cooled)
    [notice] = [row["content"] for row in messages if "[Transport recovery]" in row["content"]]
    assert "the host is trying a configured alternative." in notice and not any(
        name in notice for name in ("alt-a", "alt-b", "alt-c", "alt-d")), "no route, skipped or tried, is named to the model"
    assert usage[recovery.KEY]["tried"] == [["alt-a", False, ""], ["alt-c", False, ""]], "no skipped route is consumed"


@pytest.mark.parametrize("pin_changes", [False, True])
def test_preparation_cannot_reenable_a_spent_fallback_binding(tmp_path, monkeypatch, pin_changes):
    """A live account edit during context preparation keeps the old recovery guard's semantics."""
    from types import SimpleNamespace

    from ouroboros import fallback_cooldown, loop, loop_model_call
    from ouroboros import loop_output_exhaustion as recovery

    model = "claudexor::codex=fallback"
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", model)
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", '{"main":"account-b","fallback":["account-a"]}')
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(fallback_cooldown, "mark_cooldown", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: None)
    monkeypatch.setattr(recovery, "_save", lambda _ctx: None)
    monkeypatch.setattr(recovery, "_owner_revision", lambda _ctx: 0)

    def rebind(*_args, **_kwargs):
        if pin_changes:
            monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", '{"main":"account-b","fallback":["account-b"]}')
        return None, "max"

    monkeypatch.setattr(loop, "_rebind_context_fit_plan", rebind)
    sent = []

    def physical_send(ctx):
        ctx.physical_send_claim(SimpleNamespace(attempt_id=f"attempt-{len(sent)}"))
        sent.append(recovery.binding(ctx))
        ctx.accumulated_usage.update(
            _last_llm_error_kind="rate_limit",
            _last_llm_failure={"model": model, "failure_code": "rate_limit"})
        return None, 0.0, "max"

    monkeypatch.setattr(loop_model_call, "_call_round_model_once", physical_send)
    usage = {"_last_llm_error_kind": recovery.KIND,
             "_last_llm_failure": {"model": model, "failure_code": recovery.KIND},
             recovery.KEY: {"stage": "fallback", "owner_revision": 0,
                            "tried": [[model, False, "account-b"]], "first": {}}}
    _walk(tmp_path, usage, [], active_model=model)
    assert sent == ([] if pin_changes else [[model, False, "account-a"]])


@pytest.mark.parametrize("second", ["same_model_other_account", "other_model"])
def test_a_route_output_recovery_never_sent_borrows_no_earlier_row(tmp_path, monkeypatch, second):
    """The real output-recovery wrapper and claim. The walk opens on a request that never reached the provider while
    the informed chance is unspent; the first route spends it and fails, which moves recovery to its fallback stage,
    and that stage's gate then returns the next route without any send: it stamps its own kind and leaves the earlier
    route's row and sentence. Adopting that route is the walk's unchanged policy; the row claims no reply, no
    provider sentence and no account for it (no call of its own recorded one), also when both routes carry one
    model name on different accounts."""
    from types import SimpleNamespace

    from ouroboros import fallback_cooldown, loop, loop_model_call
    from ouroboros import loop_output_exhaustion as recovery
    from ouroboros.loop_transport import stamp_owner_provider_message

    first = "claudexor::codex=fallback"
    other = first if second == "same_model_other_account" else "claudexor::glm=fallback"
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", f"{first},{other}")
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", '{"fallback":["account-a","account-b"]}')
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(fallback_cooldown, "mark_cooldown", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: None)  # the claimed send was not refunded
    monkeypatch.setattr(recovery, "_save", lambda _ctx: None)
    monkeypatch.setattr(recovery, "_owner_revision", lambda _ctx: 0)
    sent = []

    def physical_send(ctx):  # below the real wrapper: the one send it claims, refused by the provider
        ctx.physical_send_claim(SimpleNamespace(attempt_id=f"attempt-{len(sent)}"))
        sent.append(recovery.binding(ctx))
        usage.update(_last_llm_error_kind="rate_limit", _last_llm_failure={"model": ctx.active_model, "failure_code": "rate_limit"})
        stamp_owner_provider_message(usage, {"message": "account-a is rate limited."})
        return None, 0.0, "max"

    monkeypatch.setattr(loop_model_call, "_call_round_model_once", physical_send)
    usage = {"_last_llm_error_kind": "transport_unavailable",
             "_last_llm_failure": {"model": "primary", "failure_code": "transport_unavailable"},
             recovery.KEY: {"stage": "informed", "owner_revision": 0, "tried": [["primary", False, ""]], "first": {}}}
    progress = []
    (msg, model, *_rest), ctx, _tools = _walk(tmp_path, usage, progress)
    assert sent == [[first, False, "account-a"]], "the gate returned the second route before any send"
    assert usage[recovery.KEY]["stage"] == "fallback" and usage["_last_llm_error_kind"] == recovery.KIND
    assert msg is None and model == ctx.active_model == other, "adoption policy is unchanged"
    pin = "; pinned account: siblings were not tried"
    assert [text for text, _incident in progress] == [
        f"Fallback: trying {first} (account: account-a{pin}) after a request that never reached the provider on primary.",
        f"Fallback: trying {other} (account: account-b{pin}) after a rate limit on {first}. "
        'The provider said: "account-a is rate limited."',
        f"Fallback: {other} is now the active route, with no usable answer; no cause of its own is known here.",
    ]


def test_a_spent_reply_is_adopted_under_the_account_its_own_request_carried(tmp_path, monkeypatch):
    """A reply that spent its allowance is adopted with no usable answer. Its row names the account its request
    carried (the empty-response producer's row), never the pin the owner set while that reply was generated."""
    from ouroboros import fallback_cooldown, loop
    from ouroboros import loop_output_exhaustion as recovery

    fallback = "claudexor::codex=fallback"
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", fallback)
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", '{"fallback":["account-a"]}')
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    usage, progress = {"_last_llm_error_kind": "rate_limit"}, []

    def spent(call):  # loop_llm_call's empty-response row: this call's meta, its failure code set
        monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", '{"fallback":["account-b"]}')
        row = {"model": call.active_model, "requested_profile": "account-a", "failure_code": recovery.KIND}
        usage.update(_last_llm_call_meta=row, _last_llm_failure=row, _last_llm_error_kind=recovery.KIND)
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", spent)
    (msg, model, *_rest), _ctx, _tools = _walk(tmp_path, usage, progress)
    assert msg is None and model == fallback
    assert progress[-1] == (f"Fallback: {fallback} (account: account-a) is now the active route, with no usable answer "
                            "after a reply that reached its length limit before any visible output.", None)


@pytest.mark.parametrize("usage,reason", [
    ({}, " after no usable answer on primary"),
    # An opaque 400 is a rejected request, never a guessed context overflow.
    ({"_last_llm_error_kind": "bad_request"}, " after a rejected request on primary"),
    ({"_last_llm_error_kind": "provider_outcome_unknown"},
     " after an attempt with no known outcome on primary; the earlier attempt’s outcome and cost stay unknown"),
    ({"_last_llm_error_kind": "a_future_kind"}, " after a provider failure on primary"),
])
def test_an_attempt_says_no_more_than_the_record_knows(tmp_path, monkeypatch, usage, reason):
    """The class the table has no words for reads as the owner's generic words; the
    exception text and a typed code stay in the record, never on the notice."""
    from types import SimpleNamespace
    from ouroboros import fallback_cooldown, loop, loop_model_call

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "alt-a")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    monkeypatch.setattr(loop, "_call_round_model", lambda _ctx: ({"role": "assistant"}, 0, "max"))
    usage = {**usage, "_last_llm_error": "BadRequestError('Error code: 400 - {}')"} if usage else usage
    progress = []

    loop_model_call._run_cross_model_fallback_chain(
        llm=None, ctx=SimpleNamespace(active_model="primary", active_use_local=False),
        tools=SimpleNamespace(_ctx=SimpleNamespace()), messages=[], active_model="primary",
        active_use_local=False, tool_schemas=[], active_effort="high", max_retries=1,
        drive_logs=tmp_path / "logs", task_id="task-7", round_idx=3, event_queue=None,
        accumulated_usage=usage, task_type="task",
        emit_progress=lambda text, *, incident=None: progress.append(text),
        context_fit_plan=None, active_context_mode="max",
    )

    assert progress[0] == f"⚡ Fallback: trying alt-a{reason}."


@pytest.mark.parametrize("override,expected_account,pinned", [
    ("", "Auto", False), ("account-b", "account-b", True),
])
def test_fallback_notice_names_the_account_the_task_override_binds(
    tmp_path, monkeypatch, override, expected_account, pinned,
):
    """A task-local wait override, not the configured value, is what the send uses."""
    from types import SimpleNamespace
    from ouroboros import fallback_cooldown, loop, loop_model_call
    from ouroboros.model_wait import task_model_wait_scope

    fallback = "claudexor::codex=fallback"
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", fallback)
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", '{"fallback":["account-a"]}')
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))

    def answered(call):  # the request carried the override's account, which the call meta records
        call.accumulated_usage["_last_llm_call_meta"] = {"model": call.active_model, "requested_profile": override}
        return {"role": "assistant"}, 0, "max"

    monkeypatch.setattr(loop, "_call_round_model", answered)
    progress = []

    with task_model_wait_scope(task={"id": "task-7", "_attempt": 1}, drive_root=tmp_path,
                               event_queue=None, worker_slot_held=True) as wait:
        wait.overrides["fallback:0"] = {"model": fallback, "use_local": False,
                                        "model_account_override": override}
        loop_model_call._run_cross_model_fallback_chain(
            llm=None, ctx=SimpleNamespace(active_model="primary", active_use_local=False),
            tools=SimpleNamespace(_ctx=SimpleNamespace()), messages=[], active_model="primary",
            active_use_local=False, tool_schemas=[], active_effort="high", max_retries=1,
            drive_logs=tmp_path / "logs", task_id="task-7", round_idx=3, event_queue=None,
            accumulated_usage={}, task_type="task",
            emit_progress=lambda text, *, incident=None: progress.append(text),
            context_fit_plan=None, active_context_mode="max",
        )

    attempt, adopted = progress
    assert f"(account: {expected_account}" in attempt and f"(account: {expected_account})" in adopted
    assert "account-a" not in attempt + adopted
    assert ("pinned account: siblings were not tried" in attempt) is pinned
    assert "pinned account" not in adopted


def test_api_fallback_notice_omits_inapplicable_account_clause(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import fallback_cooldown, loop, loop_model_call

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai::alternate")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    monkeypatch.setattr(loop, "_call_round_model", lambda _ctx: ({"role": "assistant"}, 0, "max"))
    progress = []

    loop_model_call._run_cross_model_fallback_chain(
        llm=None, ctx=SimpleNamespace(active_model="primary", active_use_local=False),
        tools=SimpleNamespace(_ctx=SimpleNamespace()), messages=[], active_model="primary",
        active_use_local=False, tool_schemas=[], active_effort="high", max_retries=1,
        drive_logs=tmp_path / "logs", task_id="task-7", round_idx=3, event_queue=None,
        accumulated_usage={}, task_type="task",
        emit_progress=lambda text, *, incident=None: progress.append(text),
        context_fit_plan=None, active_context_mode="max",
    )

    assert len(progress) == 2 and "account:" not in "".join(progress)


# ---------------------------------------------------------------------------
# Seam 2: the reviewer model lists (review_model_routes / reviewer slots).
# ---------------------------------------------------------------------------


def test_review_target_pins_local_route_when_review_predicate_says_so(monkeypatch):
    _clear_provider_credentials(monkeypatch)
    monkeypatch.setenv("USE_LOCAL_MAIN", "1")
    from ouroboros.provider_models import review_model_uses_local

    assert review_model_uses_local("vendor/m1") is True
    assert resolved_review_model_target("vendor/m1").provider_route == "local"


def test_pool_slots_consume_the_typed_local_route(monkeypatch):
    _clear_provider_credentials(monkeypatch)
    monkeypatch.delenv("USE_LOCAL_MAIN", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from ouroboros.reviewer_slot_config import review_pool_slots
    from tests.review_pool_rosters import set_review_pool

    set_review_pool(monkeypatch, ["vendor/m1"])
    slots = review_pool_slots(default_effort="high")
    assert [(s.model, s.use_local) for s in slots] == [("vendor/m1", False)]
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("USE_LOCAL_MAIN", "1")
    assert [s.use_local for s in review_pool_slots()] == [True]


# ---------------------------------------------------------------------------
# Seam 3: the delegated route (delegate/claudexor pinning).
# ---------------------------------------------------------------------------


def test_delegation_route_bridges_to_the_typed_target():
    route = parse_subagent_harness("codex=gpt-5.5:high")
    assert route == DelegationRoute(route_id="codex", model="gpt-5.5", effort="high")
    assert route.resolved_target() == ResolvedModelTarget(
        model_id="gpt-5.5", provider_route="codex",
        credential_ref="", effort="high", context_window=0,
    )
    pinned = DelegationRoute(route_id="claude", model="", effort="", profile_id="acct-1")
    target = pinned.resolved_target()
    assert (target.model_id, target.provider_route, target.credential_ref) == ("", "claude", "acct-1")


# ---------------------------------------------------------------------------
# Consumer-sweep pins (grep-level): downstream takes the dataclass; no new
# comma/at parsing beside a seam that already yields it.
# ---------------------------------------------------------------------------


def test_fallback_chain_consumer_takes_the_dataclass():
    source = (REPO / "ouroboros" / "loop_model_call.py").read_text(encoding="utf-8")
    assert "fallback_candidate_targets(" in source
    assert "get_fallback_models(" not in source
    assert 'split(","' not in source and 'partition("=")' not in source


def test_reviewer_slot_builders_take_the_dataclass():
    source = (REPO / "ouroboros" / "reviewer_slot_config.py").read_text(encoding="utf-8")
    # ONE builder (the pool's delivery slot) resolves the typed target; the lane
    # era's second builder left with the lanes.
    assert source.count("resolved_review_model_target(") == 1
    assert "use_local=review_model_uses_local(" not in source
    assert 'split(","' not in source


def test_delegate_run_request_takes_the_dataclass():
    """Behavioural, not textual: the wire body must carry exactly the typed
    target's fields, so a route the dataclass resolves differently (an account
    pin, a route-carried effort) reaches Claudexor through that one read."""
    from types import SimpleNamespace

    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tools.delegate import _start_request

    route = DelegationRoute(route_id="codex", model="gpt-5.5", effort="high", profile_id="acct-1")
    target = route.resolved_target()
    request = _start_request(
        SimpleNamespace(), route, delegated_run_shape(False),
        "/tmp/project", "do the work", 300, "host instructions",
    )
    assert request["harnesses"] == [target.provider_route]
    assert request["primaryHarness"] == target.provider_route
    assert request["model"] == target.model_id
    assert request["effort"] == target.effort
    assert request["credentialProfileId"] == target.credential_ref

    # A route with nothing pinned sends no empty wire keys: "" means the
    # engine's own default, and the body must not claim one.
    bare = _start_request(
        SimpleNamespace(), DelegationRoute(route_id="claude"), delegated_run_shape(False),
        "/tmp/project", "do the work", 300, "host instructions",
    )
    assert not {"model", "effort", "credentialProfileId"} & set(bare)

    source = (REPO / "ouroboros" / "tools" / "delegate.py").read_text(encoding="utf-8")
    assert 'split(","' not in source and 'partition("=")' not in source
