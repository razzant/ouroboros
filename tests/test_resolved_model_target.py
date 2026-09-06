"""ABI-4 ``ResolvedModelTarget`` — the typed resolved-model destination.

The suite name is fixed by docs/v7next/DESIGN_RESOLVED_MODEL_TARGET.md: it pins
frozen-ness, value identity, construction at each existing resolution seam
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
    get_review_models,
    get_review_targets,
    get_scope_review_models,
    get_scope_review_targets,
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


def test_fallback_dispatch_lane_stays_the_global_flag():
    """Equivalence pin for the sweep's byte-identical contract: the loop's
    local-vs-remote lane still comes from the one global USE_LOCAL_FALLBACK
    read, never from a per-candidate route field."""
    source = (REPO / "ouroboros" / "loop_model_call.py").read_text(encoding="utf-8")
    assert 'os.environ.get("USE_LOCAL_FALLBACK", "")' in source
    assert "candidate.provider_route" not in source


# ---------------------------------------------------------------------------
# Seam 2: the reviewer model lists (review_model_routes / reviewer slots).
# ---------------------------------------------------------------------------


def test_review_targets_match_effective_lists(monkeypatch):
    _clear_provider_credentials(monkeypatch)
    monkeypatch.delenv("USE_LOCAL_MAIN", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "vendor/m1,vendor/m2")
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_MODELS", "vendor/s1,vendor/s1")
    triad = get_review_targets()
    scope = get_scope_review_targets()
    assert [t.model_id for t in triad] == get_review_models() == ["vendor/m1", "vendor/m2"]
    assert [t.model_id for t in scope] == get_scope_review_models() == ["vendor/s1", "vendor/s1"]
    assert {t.provider_route for t in triad} == {"openrouter"}


def test_review_targets_pin_local_route_when_review_predicate_says_so(monkeypatch):
    _clear_provider_credentials(monkeypatch)
    monkeypatch.setenv("USE_LOCAL_MAIN", "1")
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "vendor/m1,vendor/m2")
    from ouroboros.provider_models import review_model_uses_local

    assert review_model_uses_local("vendor/m1") is True
    assert all(t.provider_route == "local" for t in get_review_targets())
    assert resolved_review_model_target("vendor/m1").provider_route == "local"


def test_reviewer_slots_consume_the_typed_local_route(monkeypatch):
    _clear_provider_credentials(monkeypatch)
    monkeypatch.delenv("USE_LOCAL_MAIN", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from ouroboros.reviewer_slot_config import reviewer_slots

    slots = reviewer_slots(models=["vendor/m1"], effort="high")
    assert [(s.model, s.use_local) for s in slots] == [("vendor/m1", False)]
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("USE_LOCAL_MAIN", "1")
    assert [s.use_local for s in reviewer_slots(models=["vendor/m1"])] == [True]


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
    assert source.count("resolved_review_model_target(") >= 2
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
