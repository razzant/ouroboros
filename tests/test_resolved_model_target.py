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


def test_fallback_notice_carries_lane_switch_incident_reason_and_pin(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import fallback_cooldown, loop, loop_model_call

    fallback = "claudexor::codex=fallback"
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", fallback)
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", '{"fallback":["account-a"]}')
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    monkeypatch.setattr(loop, "_call_round_model", lambda _ctx: ({"role": "assistant"}, 0, "max"))
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

    assert len(progress) == 1
    text, incident = progress[0]
    assert "account: account-a" in text and "reason: provider_transient" in text
    assert "pinned account: siblings were not tried" in text
    assert incident == {
        "task_incident": "model_lane_switch",
        "toast_once": f"task-7:model_lane_switch:3:{fallback}",
    }


def test_second_lane_switch_names_the_candidate_that_just_failed(tmp_path, monkeypatch):
    """Each notice names the model actually tried, beside that model's own reason."""
    from types import SimpleNamespace
    from ouroboros import fallback_cooldown, loop, loop_model_call

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "alt-a,alt-b")
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_: False)
    monkeypatch.setattr(fallback_cooldown, "mark_cooldown", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "_task_deadline_epoch", lambda _: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **k: (None, "max"))
    usage = {"_last_llm_error_kind": "primary_kind"}
    progress = []

    def call(ctx):
        # Each candidate fails and stamps its own typed kind on the record.
        usage["_last_llm_error_kind"] = f"{ctx.active_model}_kind"
        return None, 0.0, "max"

    monkeypatch.setattr(loop, "_call_round_model", call)
    loop_model_call._run_cross_model_fallback_chain(
        llm=None, ctx=SimpleNamespace(active_model="primary", active_use_local=False),
        tools=SimpleNamespace(_ctx=SimpleNamespace()), messages=[], active_model="primary",
        active_use_local=False, tool_schemas=[], active_effort="high", max_retries=1,
        drive_logs=tmp_path / "logs", task_id="task-7", round_idx=3, event_queue=None,
        accumulated_usage=usage, task_type="task",
        emit_progress=lambda text, *, incident=None: progress.append(text),
        context_fit_plan=None, active_context_mode="max",
    )

    assert progress == [
        "⚡ Fallback: primary → alt-a; reason: primary_kind",
        "⚡ Fallback: alt-a → alt-b; reason: alt-a_kind",
    ]


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
    monkeypatch.setattr(loop, "_call_round_model", lambda _ctx: ({"role": "assistant"}, 0, "max"))
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

    assert len(progress) == 1 and f"account: {expected_account}" in progress[0]
    assert "account-a" not in progress[0]
    assert ("pinned account: siblings were not tried" in progress[0]) is pinned


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

    assert len(progress) == 1
    assert "account:" not in progress[0]


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
