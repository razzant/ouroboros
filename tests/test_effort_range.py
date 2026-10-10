"""The owner's effort range: the tolerant read, the one effort decision, the model-named
level detector, the retired role keys, and the reviewer rule that rides them.

Every rule is pinned in both directions: a request inside and outside the range, a pin
with and without a request, binding and Cyber Pro, a name beside each of them."""

from __future__ import annotations

import itertools
import json

import pytest

from ouroboros import settings_scales as scales
from ouroboros.settings_scales import (
    EFFORT_SCALE, OWNER_EFFORT_TIERS, choose_effort, clamp_effort_into, effort_fact,
    effort_fact_phrase, effort_range,
)


def _env(monkeypatch, **values):
    for key in ("OUROBOROS_EFFORT_MIN", "OUROBOROS_EFFORT_TASK", "OUROBOROS_EFFORT_MAX"):
        monkeypatch.delenv(key, raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key, value)


# --- the tolerant read ----------------------------------------------------------------


def test_fresh_install_reads_low_medium_high(monkeypatch):
    _env(monkeypatch)
    assert effort_range() == {"min": "low", "recommended": "medium", "max": "high"}
    assert OWNER_EFFORT_TIERS == ("none", "low", "medium", "high", "xhigh", "max", "ultra")


def test_a_document_with_only_task_widens_the_bounds_around_it(monkeypatch):
    """The owner's own install: TASK=high, no MIN/MAX -> low / high / high."""
    _env(monkeypatch, OUROBOROS_EFFORT_TASK="high")
    assert effort_range() == {"min": "low", "recommended": "high", "max": "high"}
    _env(monkeypatch, OUROBOROS_EFFORT_TASK="ultra")
    assert effort_range() == {"min": "low", "recommended": "ultra", "max": "ultra"}
    _env(monkeypatch, OUROBOROS_EFFORT_TASK="none")
    assert effort_range() == {"min": "none", "recommended": "none", "max": "high"}


def test_unknown_values_take_their_key_default_and_minimal_round_trips(monkeypatch):
    _env(monkeypatch, OUROBOROS_EFFORT_MIN="bogus", OUROBOROS_EFFORT_TASK="", OUROBOROS_EFFORT_MAX="HIGH")
    assert effort_range() == {"min": "low", "recommended": "medium", "max": "high"}
    _env(monkeypatch, OUROBOROS_EFFORT_MIN="minimal", OUROBOROS_EFFORT_TASK="minimal", OUROBOROS_EFFORT_MAX="none")
    assert effort_range() == {"min": "minimal", "recommended": "minimal", "max": "minimal"}
    # A document read bypasses the environment; a live read opens the owner's current document.
    assert effort_range({"OUROBOROS_EFFORT_MAX": "ultra"}) == {"min": "low", "recommended": "medium", "max": "ultra"}


def test_a_live_read_sees_the_current_document_while_the_task_scope_keeps_its_own(monkeypatch, tmp_path):
    from ouroboros import config
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot

    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"OUROBOROS_EFFORT_TASK": "medium", "OUROBOROS_EFFORT_MAX": "ultra"}), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", path, raising=True)
    _env(monkeypatch, OUROBOROS_EFFORT_TASK="low", OUROBOROS_EFFORT_MAX="low")
    old = task_settings_snapshot({"OUROBOROS_EFFORT_TASK": "low", "OUROBOROS_EFFORT_MAX": "low"},
                                 {"OUROBOROS_EFFORT_TASK": "low", "OUROBOROS_EFFORT_MAX": "low"})
    from ouroboros.settings_integrity import live_effort_range

    with task_settings_scope(old):
        assert effort_range()["max"] == "low"  # the running task keeps the range it started with
        assert live_effort_range()["max"] == "ultra"  # a participant starting now reads the owner's document


def test_clamp_moves_to_the_nearest_bound_only(monkeypatch):
    rng = {"min": "low", "recommended": "medium", "max": "high"}
    assert clamp_effort_into("none", rng) == "low"
    assert clamp_effort_into("ultra", rng) == "high"
    assert clamp_effort_into("medium", rng) == "medium"
    assert clamp_effort_into("", rng) == "" and clamp_effort_into("bogus", rng) == "bogus"


# --- the one decision ------------------------------------------------------------------


def _triples():
    """Every ordered min <= recommended <= max over the seven owner tiers."""
    for low, rec, high in itertools.product(OWNER_EFFORT_TIERS, repeat=3):
        if scales.effort_rank(low) <= scales.effort_rank(rec) <= scales.effort_rank(high):
            yield {"min": low, "recommended": rec, "max": high}


@pytest.mark.parametrize("binds", [True, False])
def test_choose_effort_over_every_ordered_triple(binds):
    for rng in _triples():
        lo, rec, hi = scales.effort_rank(rng["min"]), scales.effort_rank(rng["recommended"]), scales.effort_rank(rng["max"])
        # Silence: the role default — recommended, or the top for default_top roles — in every mode.
        assert choose_effort("", binds=binds, rng=rng) == (rng["recommended"], "auto")
        assert choose_effort("", default_top=True, binds=binds, rng=rng) == (rng["max"], "auto")
        for tier in EFFORT_SCALE:
            rank = scales.effort_rank(tier)
            level, source = choose_effort(tier, binds=binds, rng=rng)
            if binds:  # a request clamps to the nearest bound, never refused, source auto
                assert source == "auto" and scales.effort_rank(level) == min(max(rank, lo), hi)
            else:  # Cyber Pro: the request applies as asked
                assert (level, source) == (tier, "cyber")
            # A pin: wins while the range binds; in Cyber Pro the request beats it, silence sits on it.
            assert choose_effort("", pin=tier, binds=binds, rng=rng) == (tier, "pin")
            assert choose_effort("ultra", pin=tier, binds=binds, rng=rng) == ((tier, "pin") if binds else ("ultra", "cyber"))
            # A level in the model name wins over everything, in every mode, even outside the range.
            assert choose_effort("none", pin="max", model_named=tier, binds=binds, rng=rng) == (tier, "model_name")
        assert lo <= rec <= hi


def test_unknown_tiers_read_as_absent_and_the_fact_records_the_request(monkeypatch):
    rng = {"min": "low", "recommended": "medium", "max": "high"}
    assert choose_effort("bogus", binds=True, rng=rng) == ("medium", "auto")
    assert choose_effort("", pin="BOGUS", binds=True, rng=rng) == ("medium", "auto")
    assert choose_effort(" XHIGH ", binds=True, rng=rng) == ("high", "auto")
    assert effort_fact(" Ultra", "high", "auto") == {"requested": "ultra", "applied": "high", "source": "auto"}
    assert effort_fact("", "medium", "auto") == {"requested": "", "applied": "medium", "source": "auto"}


def test_the_decision_binds_by_the_runtime_mode(monkeypatch):
    from ouroboros import config
    from ouroboros.runtime_mode_policy import effort_range_binds

    _env(monkeypatch, OUROBOROS_EFFORT_MAX="high")
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "pro")
    assert effort_range_binds() is True
    assert choose_effort("ultra") == ("high", "auto")
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "cyber_pro")
    assert effort_range_binds() is False
    assert choose_effort("ultra") == ("ultra", "cyber")
    # A consciousness-origin task capped to light binds even on a Cyber Pro install.
    assert effort_range_binds({"initiator": "consciousness", "runtime_mode_cap": "light"}) is True


def test_the_phrase_speaks_only_when_a_request_was_moved_or_set_aside():
    rng = {"min": "low", "recommended": "medium", "max": "high"}
    assert effort_fact_phrase(effort_fact("", "medium", "auto"), rng) == ""
    assert effort_fact_phrase(effort_fact("high", "high", "auto"), rng) == ""
    assert effort_fact_phrase(effort_fact("", "xhigh", "pin"), rng) == ""  # a pin deciding alone is the row's own
    assert effort_fact_phrase(effort_fact("", "max", "model_name"), rng) == ""
    assert effort_fact_phrase(effort_fact("ultra", "ultra", "cyber"), rng) == ""
    assert effort_fact_phrase(effort_fact("ultra", "high", "auto"), rng) == (
        "effort high: your request ultra moved into my human's range low..high")
    assert effort_fact_phrase(effort_fact("none", "low", "auto"), rng) == (
        "effort low: your request none moved into my human's range low..high")
    assert effort_fact_phrase(effort_fact("low", "xhigh", "pin"), rng) == (
        "effort xhigh: pinned by my human; requested low not applied")
    assert effort_fact_phrase(effort_fact("low", "max", "model_name"), rng) == (
        "effort max: the level in the model name; requested low not applied")


def test_resolve_effort_keeps_its_signature_over_the_range(monkeypatch):
    _env(monkeypatch, OUROBOROS_EFFORT_TASK="medium", OUROBOROS_EFFORT_MAX="ultra")
    assert scales.resolve_effort("task") == scales.resolve_effort("presence") == scales.resolve_effort("") == "medium"
    assert scales.resolve_effort("evolution") == scales.resolve_effort("consciousness") == "ultra"


# --- the model-named level detector -----------------------------------------------------


@pytest.mark.parametrize("target, expected", [
    ("cursor=grok-4.7-xhigh-fast", "xhigh"), ("cursor=cursor-grok-4.6-high", "high"), ("agy=gemini-3.1-pro-max-fast", "max"),
    ("agy=gemini-3.8-flash-high", "high"), ("codex=gpt-5.6-sol", ""), ("claude=claude-opus-5", ""),
    ("cursor=composer-2.5", ""), ("cursor=auto", ""), ("codex=gpt-5.6-sol-high", ""), ("cursor", ""),
])
def test_session_targets_name_their_level_only_on_slug_harnesses(target, expected):
    from ouroboros.route_spec import ROUTE_KIND_AGENT_SESSION, RouteSpec, compound_session_effort, model_named_effort

    route = RouteSpec(ROUTE_KIND_AGENT_SESSION, target)
    assert compound_session_effort(route) == model_named_effort(route) == expected


@pytest.mark.parametrize("model, expected", [
    ("claudexor::cursor=grok-4.7-xhigh-fast", "xhigh"), ("claudexor::agy=gemini-3.1-pro-low", "low"),
    ("claudexor::codex=gpt-5.6-sol-high", ""), ("claudexor::claude=claude-opus-5", ""), ("claudexor::cursor", ""),
    ("openai/gpt-5.5-high", ""), ("openai::gpt-5.5", ""), ("cursor-grok-4.6-high", ""), ("", ""),
])
def test_api_wrapped_claudexor_models_name_their_level_the_same_way(model, expected):
    from ouroboros.route_spec import ROUTE_KIND_API_MODEL, RouteSpec, api_model_named_effort, model_named_effort

    assert api_model_named_effort(model) == expected
    if model:
        assert model_named_effort(RouteSpec(ROUTE_KIND_API_MODEL, model)) == expected


def test_a_stored_api_row_beside_a_named_model_still_loads_and_the_name_wins_at_execution(monkeypatch):
    """The shared parser keeps the session-only conflict rule: a stored API row pinned beside a
    named model never turns the catalog SOURCE_INVALID; the name decides at execution."""
    from ouroboros.configured_subagents import parse_configured_subagents
    from ouroboros.route_spec import ROUTE_KIND_API_MODEL, RouteSpec, validate_compound_session_effort

    rows = parse_configured_subagents(json.dumps({"enabled": True, "items": [
        {"subagent_id": "named", "recommended_use": "x", "effort": "low",
         "route": {"kind": "api_model", "target_id": "claudexor::cursor=grok-4.7-xhigh-fast"}},
    ]}))
    assert rows.items[0].effort == "low"
    validate_compound_session_effort(RouteSpec(ROUTE_KIND_API_MODEL, "claudexor::cursor=grok-4.7-xhigh-fast"), "low",
                                     setting="s", where="w")  # never an error for an API row
    with pytest.raises(ValueError, match="conflicts with compound route effort"):
        validate_compound_session_effort(RouteSpec("agent_session", "cursor=grok-4.7-xhigh-fast"), "low",
                                         setting="s", where="w")
    assert choose_effort("medium", pin="low", model_named="xhigh", binds=True) == ("xhigh", "model_name")


def test_the_claudexor_model_transport_never_sends_a_contradicting_effort():
    from ouroboros.llm_claudexor import _request

    named = {"source": "cursor", "resolved_model": "grok-4.7-xhigh-fast", "processing_preferences": []}
    body = _request(named, [{"role": "user", "content": "hi"}], None, {"reasoning_effort": "low"})
    assert body["options"]["reasoningEffort"] == "xhigh" and named["requested_reasoning_effort"] == "low"
    plain = {"source": "codex", "resolved_model": "gpt-5.6-sol", "processing_preferences": []}
    body = _request(plain, [{"role": "user", "content": "hi"}], None, {"reasoning_effort": "low"})
    assert body["options"]["reasoningEffort"] == "low"


# --- the retired role keys ---------------------------------------------------------------


def test_the_role_keys_are_dropped_with_the_notice_naming_the_range_top(monkeypatch, tmp_path):
    from ouroboros import config
    from ouroboros.settings_defaults import retired_setting_keys_notice

    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"OUROBOROS_EFFORT_EVOLUTION": "xhigh", "OUROBOROS_EFFORT_CONSCIOUSNESS": "low",
                                "OUROBOROS_EFFORT_TASK": "high"}), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", path, raising=True)
    loaded = config.load_settings()
    assert "OUROBOROS_EFFORT_EVOLUTION" not in loaded and "OUROBOROS_EFFORT_CONSCIOUSNESS" not in loaded
    assert loaded["OUROBOROS_EFFORT_TASK"] == "high" and loaded["OUROBOROS_EFFORT_MAX"] == "high"
    notice = retired_setting_keys_notice(("OUROBOROS_EFFORT_EVOLUTION", "OUROBOROS_EFFORT_CONSCIOUSNESS"))
    assert "NOT honored" in notice and "OUROBOROS_EFFORT_EVOLUTION -> OUROBOROS_EFFORT_MAX" in notice


def test_the_rc_auditor_reports_a_stored_role_key_as_a_note_naming_the_range_top(tmp_path):
    import os
    import pathlib

    from tests.test_rc_audit_fixture_suite import _build_clean_70_install, _load_module, _run

    module = _load_module()
    checks = {c["key"]: c for c in module.build_scope()["checks"] if c["id"] == "retired-setting"}
    for key in ("OUROBOROS_EFFORT_EVOLUTION", "OUROBOROS_EFFORT_CONSCIOUSNESS"):
        assert "OUROBOROS_EFFORT_MAX" in checks[key]["migration"] and "effort range" in checks[key]["behavior"]
    data = _build_clean_70_install(tmp_path / "install")
    document = json.loads((data / "settings.json").read_text(encoding="utf-8"))
    document.update({"OUROBOROS_EFFORT_EVOLUTION": "xhigh", "OUROBOROS_EFFORT_CONSCIOUSNESS": "high"})
    (data / "settings.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    result = _run(data, "--json", str(tmp_path / "report.json"), isolated_root=tmp_path / "isol")
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(pathlib.Path(tmp_path / "report.json").read_text(encoding="utf-8"))
    notes = [f for f in report["findings"] if f["check_id"] == "retired-setting"]
    assert {f["subject"] for f in notes} == {"settings.json:OUROBOROS_EFFORT_EVOLUTION",
                                            "settings.json:OUROBOROS_EFFORT_CONSCIOUSNESS"}
    assert all(f["severity"] == "note" and "OUROBOROS_EFFORT_MAX" in f["detail"] for f in notes)
    assert report["summary"]["incompatible"] == 0
    assert os.environ.get("OUROBOROS_EFFORT_EVOLUTION") is None


# --- reviewers ---------------------------------------------------------------------------


def _reviewer(target="openai/gpt-5.6-terra", *, effort="", kind="api_chat", session_target=""):
    from ouroboros.reviewer_slot_config import ConfiguredReviewerSlot

    return ConfiguredReviewerSlot(slot_id="r", kind=kind, target_id=target, effort=effort, session_target=session_target)


def test_reviewer_rule_name_then_pin_then_order_clamped_then_the_range_top(monkeypatch):
    from ouroboros import config
    from ouroboros.reviewer_slot_config import row_at_effort_order, row_effort, row_effort_source

    _env(monkeypatch, OUROBOROS_EFFORT_MIN="low", OUROBOROS_EFFORT_TASK="medium", OUROBOROS_EFFORT_MAX="high")
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "pro")
    auto, pinned = _reviewer(), _reviewer(effort="xhigh")
    named = _reviewer("cursor=grok-4.7-max-fast", kind="agent_session", session_target="cursor=grok-4.7-max-fast")
    named_api = _reviewer("claudexor::agy=gemini-3.1-pro-low")
    assert row_effort(auto) == "high" and row_effort_source(auto) == "auto"
    assert row_effort(auto, default="ultra") == "high" and row_effort(auto, default="none") == "low"
    assert row_effort(pinned) == "xhigh" and row_effort(pinned, default="low") == "xhigh"
    assert row_effort_source(pinned) == "pin"
    assert row_effort(named, default="low") == "max" and row_effort(named_api, default="ultra") == "low"
    assert row_effort_source(named) == row_effort_source(named_api) == "model_name"
    assert row_at_effort_order(auto, "ultra").effort == "high"
    assert row_at_effort_order(pinned, "low") is None and row_at_effort_order(named, "low") is None
    # The deep review's Main row: Auto -> the range's top; a named Main -> its level.
    monkeypatch.setenv("OUROBOROS_MODEL", "openai/gpt-5.6-sol")
    from ouroboros.deep_self_review import main_review_row

    assert row_effort(main_review_row()) == "high"
    monkeypatch.setenv("OUROBOROS_EFFORT_MAX", "ultra")
    assert row_effort(main_review_row()) == "ultra"
    monkeypatch.setenv("OUROBOROS_MODEL", "claudexor::cursor=grok-4.7-xhigh-fast")
    assert row_effort(main_review_row()) == "xhigh"
    # Cyber Pro: the order beats the pin, unclamped, never the name.
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "cyber_pro")
    assert row_effort(pinned, default="low") == "low" and row_effort(auto, default="ultra") == "ultra"
    assert row_effort(named, default="low") == "max" and row_at_effort_order(pinned, "low").effort == "low"
    assert row_effort(pinned) == "xhigh" and row_effort(auto) == "ultra"


def test_a_wave_order_discloses_the_effective_weaker_level_not_the_raw_order(monkeypatch):
    from types import SimpleNamespace

    from ouroboros.tools.review_change import _effort_facts

    seats = [(SimpleNamespace(slot_id="auto", effort="low", declared_effort="none"), (), False),
             (SimpleNamespace(slot_id="pinned", effort="xhigh", declared_effort=""), (), False)]
    facts = _effort_facts("none", seats, {"auto": "high", "pinned": "xhigh"})
    assert facts == {"order": "none", "applied": ["auto"], "not_applied": ["pinned"], "weaker_than_configured": ["auto"]}
    seats = [(SimpleNamespace(slot_id="auto", effort="high", declared_effort="ultra"), (), False)]
    assert _effort_facts("ultra", seats, {"auto": "high"})["weaker_than_configured"] == []


# --- children: the request, the decision, the carrier to the first start, the disclosure -----


def _session_snapshot(target="codex=gpt-5.6-sol", effort=""):
    return {"schema": 1, "selected_subagent_id": "builder", "config_fingerprint": "fp", "effort": effort,
            "route": {"kind": "agent_session", "target_id": target, "credential_profile_id": ""},
            "processing_preference": "", "access": "full", "selected_at": "2026-10-10T00:00:00Z"}


def _api_snapshot(target="openai::gpt-5.6-sol", effort=""):
    return {"schema": 1, "selected_subagent_id": "builder", "config_fingerprint": "fp", "effort": effort,
            "route": {"kind": "api_model", "target_id": target, "credential_profile_id": ""},
            "processing_preference": "", "selected_at": "2026-10-10T00:00:00Z"}


def _stub_session_route(monkeypatch):
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.subagents as subagents

    class Gateway:
        def close(self):
            pass

    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
    monkeypatch.setattr(subagents, "route_health", lambda *_a, **_k: ("", ""))
    monkeypatch.setattr("ouroboros.provider_models.model_has_credentials", lambda _model: True)


@pytest.mark.parametrize("cyber", [False, True])
def test_dispatch_decides_a_childs_effort_from_the_row_and_the_request(monkeypatch, cyber):
    """Auto row: the request clamped into the range (unclamped in Cyber Pro), recommended with
    none; pinned row: the pin, which a request beats only in Cyber Pro; a level in the model
    name (session slug or API-wrapped) always wins. The decision is written as three scalars;
    a session row's decision is its LEAF's while the nanny inherits the parent's effort."""
    from ouroboros.subagent_runtime import resolve_configured_actor_dispatch

    _env(monkeypatch, OUROBOROS_EFFORT_MIN="low", OUROBOROS_EFFORT_TASK="medium", OUROBOROS_EFFORT_MAX="high")
    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "cyber_pro" if cyber else "pro")
    _stub_session_route(monkeypatch)
    nanny = {"model": "openai::parent", "effort": "ultra"}

    def dispatch(snapshot, requested=""):
        task = {"id": "c", "configured_subagent": snapshot, "requested_effort": requested,
                "parent_cognitive_route": nanny, "task_constraint": {}}
        d = resolve_configured_actor_dispatch(task, task_type="task")
        return d, d.record_fields()

    d, rec = dispatch(_api_snapshot())
    assert (rec["effort_level"], rec["effort_requested"], rec["effort_source"]) == ("medium", "", "auto") == (d.effort, "", "auto")
    d, rec = dispatch(_api_snapshot(), "high")
    assert (d.effort, rec["effort_source"]) == ("high", "cyber" if cyber else "auto")
    d, rec = dispatch(_api_snapshot(), "ultra")
    assert (d.effort, rec["effort_requested"], rec["effort_source"]) == (("ultra", "ultra", "cyber") if cyber else ("high", "ultra", "auto"))
    d, rec = dispatch(_api_snapshot(), "none")
    assert d.effort == ("none" if cyber else "low")
    d, rec = dispatch(_api_snapshot(effort="xhigh"), "low")
    assert (d.effort, rec["effort_source"]) == (("low", "cyber") if cyber else ("xhigh", "pin"))
    d, rec = dispatch(_api_snapshot(effort="xhigh"))
    assert (d.effort, rec["effort_source"]) == ("xhigh", "pin")
    d, rec = dispatch(_api_snapshot("claudexor::cursor=grok-4.7-max-fast", effort="low"), "none")
    assert (d.effort, rec["effort_source"], rec["effort_requested"]) == ("max", "model_name", "none")
    # A session row: the leaf's decision rides the exact route; the nanny keeps the parent's level.
    d, rec = dispatch(_session_snapshot(), "ultra")
    leaf = d.executor_resolution.route
    assert leaf.effort == rec["effort_level"] == ("ultra" if cyber else "high") and rec["effort_requested"] == "ultra"
    assert d.effort == d.delta.derived_effort == d.delta.effective_effort == rec["reasoning_effort"] == "ultra"
    d, rec = dispatch(_session_snapshot("cursor=grok-4.7-xhigh-fast"), "low")
    assert d.executor_resolution.route.effort == rec["effort_level"] == "xhigh" and rec["effort_source"] == "model_name"
    d, rec = dispatch(_session_snapshot(effort="medium"), "ultra")
    assert d.executor_resolution.route.effort == ("ultra" if cyber else "medium")


def test_the_leaf_effort_rides_one_carrier_from_the_record_to_the_start_body(monkeypatch, tmp_path):
    """Record -> bootstrap -> bound start -> exact start -> the route the start body is built from,
    before the nanny's first model call; the row's pin stays beside it for the receipt identity."""
    import ouroboros.subagent_runtime as runtime
    import ouroboros.tools.delegate as delegate
    from ouroboros.delegate_shared import delegate_result
    from ouroboros.subagent_bootstrap import _prepare_actor_first_bootstrap
    from ouroboros.subagents import DelegationRoute, delegated_run_shape
    from ouroboros.tools.registry import ToolContext

    task = {"id": "child", "configured_subagent": _session_snapshot(), "task_contract": {"objective": "Build"},
            "effort_level": "high", "effort_requested": "ultra", "effort_source": "auto"}
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "child"
    _prepare_actor_first_bootstrap(ctx, task, None)
    assert ctx._configured_actor_bootstrap["effort_fact"] == {"requested": "ultra", "applied": "high", "source": "auto"}
    seen = {}

    def _start(_ctx, prompt, *_a, **_k):
        actor, refusal = runtime.prepare_delegate_start_actor(
            _ctx, tmp_path, recovering=False, invocation_id="", work_order_fingerprint="w", authority_fingerprint="a")
        assert refusal is None
        seen.update(actor)
        body = delegate._start_request(_ctx, actor["route"], delegated_run_shape(False), str(tmp_path), prompt, 60, "")
        seen["body"] = body
        return delegate_result({"status": "started", "run_id": "run-1"})

    monkeypatch.setattr(delegate, "_delegate_start", _start)
    out = json.loads(runtime.delegate_start_entry(ctx, "").text)
    assert out["status"] == "started"
    assert seen["route"].effort == seen["body"]["effort"] == "high" and seen["row_effort"] == ""
    assert seen["effort_fact"] == {"requested": "ultra", "applied": "high", "source": "auto"}
    # The wire never contradicts a level the model slug carries.
    named = DelegationRoute(route_id="cursor", model="grok-4.7-xhigh-fast", effort="low")
    assert delegate._start_request(ctx, named, delegated_run_shape(False), str(tmp_path), "p", 60, "")["effort"] == "xhigh"


def test_a_direct_delegate_start_decides_now_against_the_current_range(monkeypatch, tmp_path):
    """``delegate_start(effort=…)``: a pinned row keeps its pin outside Cyber Pro (the request
    set aside and said so), in Cyber Pro the request wins; an unknown tier is a typed refusal;
    ``auto`` and omission ask nothing."""
    import ouroboros.subagent_runtime as runtime
    import ouroboros.tools.delegate as delegate
    from ouroboros.tools.tool_result import ToolResult
    from tests.test_delegate_start_root_selector import _registry
    from tests.test_delegated_skill_payload import _payload_ctx

    _env(monkeypatch, OUROBOROS_EFFORT_MAX="high")
    ctx = _payload_ctx(tmp_path, monkeypatch)  # the payload-session row is pinned low
    seen = []
    monkeypatch.setattr(delegate, "_delegate_start", lambda *_a, **_k: (
        seen.append(runtime._EXACT_START_SELECTION.get()) or ToolResult(status="ok", code="OK", text=json.dumps({"status": "started"}))))
    registry = _registry(tmp_path, monkeypatch, ctx)
    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "pro")
    assert json.loads(registry.execute("delegate_start", {"subagent_id": "payload-session", "prompt": "p", "effort": "ultra"}))["status"] == "started"
    assert seen[-1]["effort_fact"] == {"requested": "ultra", "applied": "low", "source": "pin"}
    assert json.loads(registry.execute("delegate_start", {"subagent_id": "payload-session", "prompt": "p", "effort": "auto"}))["status"] == "started"
    assert seen[-1]["effort_fact"] == {"requested": "", "applied": "low", "source": "pin"}
    refused = json.loads(registry.execute("delegate_start", {"subagent_id": "payload-session", "prompt": "p", "effort": "turbo"}))
    assert refused["status"] == "refused" and refused["reason"] == "effort_invalid" and len(seen) == 2
    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "cyber_pro")
    assert json.loads(registry.execute("delegate_start", {"subagent_id": "payload-session", "prompt": "p", "effort": "ultra"}))["status"] == "started"
    assert seen[-1]["effort_fact"] == {"requested": "ultra", "applied": "ultra", "source": "cyber"}


def test_a_retry_replays_its_recorded_effort(monkeypatch, tmp_path):
    import ouroboros.subagent_runtime as runtime
    from ouroboros import delegate_custody as custody
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "t"
    monkeypatch.setattr(custody, "invocation_record", lambda _root, _token: {"request": {"effort": "high"}})
    assert runtime._leaf_effort_choice(ctx, None, "auto", None, "inv-1") == {}
    assert runtime._leaf_effort_choice(ctx, None, "high", None, "inv-1") == {}
    with pytest.raises(runtime.SubagentSelectionError, match="retry_selector_conflict"):
        runtime._leaf_effort_choice(ctx, None, "low", None, "inv-1")


def test_the_effort_fact_reaches_the_child_prompt_the_parent_and_the_chat_frames(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from ouroboros.agent_dispatch import capability_delta_prompt_block
    from ouroboros.gateway.contracts import ChatOutbound
    from ouroboros.gateway.history import _PROGRESS_META_FIELDS
    from ouroboros.subagent_messages import SUBAGENT_MESSAGE_FIELDS, subagent_message_meta
    from ouroboros.subagents import CapabilityDelta
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.control import _get_task_result, _wait_for_tasks
    from tests.test_model_slot_role_model import _scheduling_ctx

    _env(monkeypatch, OUROBOROS_EFFORT_MAX="high")
    moved = {"requested": "ultra", "applied": "high", "source": "auto"}
    native = SimpleNamespace(delta=CapabilityDelta(), executor_resolution=None, effort_fact=moved)
    assert capability_delta_prompt_block(native) == (
        "[CAPABILITY DELTA]\nYour effort high: your request ultra moved into my human's range low..high.")
    harness = SimpleNamespace(delta=CapabilityDelta(effective_executor="harness"), executor_resolution=None,
                              effort_fact={"requested": "low", "applied": "xhigh", "source": "pin"})
    assert "Your delegated run's effort xhigh: pinned by my human; requested low not applied." in capability_delta_prompt_block(harness)
    quiet = SimpleNamespace(delta=CapabilityDelta(), executor_resolution=None,
                            effort_fact={"requested": "", "applied": "xhigh", "source": "pin"})
    assert capability_delta_prompt_block(quiet) == ""

    ctx = _scheduling_ctx(tmp_path / "parent")
    write_task_result(tmp_path / "parent", "moved", "completed", result="done",
                      effort_level="high", effort_requested="ultra", effort_source="auto")
    write_task_result(tmp_path / "parent", "plain", "completed", result="done",
                      effort_level="high", effort_requested="high", effort_source="auto")
    full = _get_task_result(ctx, "moved")
    assert '"effort": {\n    "level": "high",\n    "requested": "ultra",\n    "source": "auto"\n  }' in full
    assert '"effort"' not in _get_task_result(ctx, "plain")
    batch = json.loads(_wait_for_tasks(ctx, ["moved", "plain"], timeout_sec=1))["tasks"]
    assert batch["moved"]["effort"] == {"level": "high", "requested": "ultra", "source": "auto"} and "effort" not in batch["plain"]

    task = {"id": "c", "delegation_role": "subagent", "effort_level": "high", "effort_requested": "ultra", "effort_source": "auto"}
    meta = subagent_message_meta(task, task_id="c", event="running")
    assert (meta["effort_level"], meta["effort_requested"], meta["effort_source"]) == ("high", "ultra", "auto")
    assert subagent_message_meta({"id": "old", "delegation_role": "subagent"}, task_id="old")["effort_level"] == ""
    for key in ("effort_level", "effort_requested", "effort_source"):
        assert key in SUBAGENT_MESSAGE_FIELDS and key in _PROGRESS_META_FIELDS and key in ChatOutbound.__annotations__


def test_the_dispatch_effort_fact_reaches_the_durable_result_through_the_real_writers(tmp_path, monkeypatch):
    """The completion write and the exception write carry the dispatch's effort decision, so
    the parent's projections and the terminal frame read it from the result file; a task that
    was never dispatched carries no effort keys (unknown stays unknown)."""
    from types import SimpleNamespace

    import ouroboros.agent_task_pipeline as pipeline
    from ouroboros.agent import _task_exception_terminal
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control import _get_task_result
    from tests.test_model_slot_role_model import _scheduling_ctx

    _env(monkeypatch, OUROBOROS_EFFORT_MAX="high")
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    (tmp_path / "logs").mkdir()
    fact = {"effort_level": "high", "effort_requested": "ultra", "effort_source": "auto"}
    child = {"id": "child-done", "type": "task", "chat_id": 1, "text": "inspect", "delegation_role": "subagent",
             "parent_task_id": "parent1", "root_task_id": "parent1", **fact}
    pipeline.emit_task_results(env, None, None, [], child, "Findings.", {"rounds": 1},
                               {"tool_calls": [], "reasoning_notes": []}, 0.0, tmp_path / "logs")
    assert {key: load_task_result(tmp_path, "child-done").get(key) for key in fact} == fact
    assert '"requested": "ultra"' in _get_task_result(_scheduling_ctx(tmp_path), "child-done")

    _task_exception_terminal(env, {**child, "id": "child-crash"}, RuntimeError("boom"), tmp_path / "logs")
    assert {key: load_task_result(tmp_path, "child-crash").get(key) for key in fact} == fact

    root = {"id": "root-done", "type": "task", "chat_id": 1, "text": "x"}
    pipeline.emit_task_results(env, None, None, [], root, "Done.", {"rounds": 1},
                               {"tool_calls": [], "reasoning_notes": []}, 0.0, tmp_path / "logs")
    assert not set(fact) & set(load_task_result(tmp_path, "root-done"))


def test_session_receipt_identity_keeps_the_rows_pin_not_the_leaf_level(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as custody
    from ouroboros.subagent_history import record_session_execution, subagent_last_delegation

    def settle(**fields):
        monkeypatch.setattr(custody, "_CUSTODY", {})
        entry = custody.RunCustody(run_id=fields.pop("run_id"), selected_subagent_id="worker", task_id="task",
                                   route_id="codex", model="m", **fields)
        record_session_execution(tmp_path, entry, {"summary": {"state": "succeeded", "finishedAt": "2099-01-01T00:00:00Z"}}, {})
        return subagent_last_delegation(tmp_path)["identity"]

    assert settle(run_id="auto", effort="high", row_effort="")["effort"] == ""  # Auto row, leaf high
    assert settle(run_id="cyber", effort="ultra", row_effort="high")["effort"] == "high"  # Cyber: pin high, leaf ultra
    assert settle(run_id="legacy", effort="high")["effort"] == "high"  # no pin recorded: today's copy


# --- Main and the roots Ouroboros creates itself -------------------------------------------


def _main_ctx(tmp_path, monkeypatch, **metadata):
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setattr("ouroboros.llm.LLMClient.available_models", lambda self: ["provider::main"])
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_metadata = dict(metadata)
    return ctx


def test_switch_model_keeps_main_at_the_level_it_works_at_outside_cyber_pro(tmp_path, monkeypatch):
    from ouroboros.tools.control_runtime import _switch_model

    _env(monkeypatch, OUROBOROS_EFFORT_TASK="medium", OUROBOROS_EFFORT_MAX="high")
    monkeypatch.setenv("OUROBOROS_MODEL", "openai::gpt-5.6-sol")
    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "pro")
    ctx = _main_ctx(tmp_path, monkeypatch)
    ctx.active_effort = "medium"
    out = _switch_model(ctx, effort="ultra")
    assert out == ("effort=ultra not applied: Main works at my human's recommended level (medium); "
                   "deeper thinking is delegated — schedule_subagent(effort=…).")
    assert ctx.active_effort_override is None
    # Model switching keeps working beside the kept level.
    out = _switch_model(ctx, model="provider::main", effort="low")
    assert out.startswith("OK: switching to model=provider::main on next round. effort=low not applied")
    assert ctx.active_model_override == "provider::main" and ctx.active_effort_override is None
    # A root the owner pinned works at the pin; a model-named Main at the name's level, in every mode.
    pinned = _main_ctx(tmp_path, monkeypatch, reasoning_effort="xhigh")
    pinned.active_effort = "xhigh"
    assert "the level my human pinned for this task (xhigh)" in _switch_model(pinned, effort="low")
    monkeypatch.setenv("OUROBOROS_MODEL", "claudexor::cursor=grok-4.7-xhigh-fast")
    assert "the level in the model name (xhigh) holds in every mode" in _switch_model(_main_ctx(tmp_path, monkeypatch), effort="low")
    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "cyber_pro")
    assert "not applied" in _switch_model(_main_ctx(tmp_path, monkeypatch), effort="low")
    monkeypatch.setenv("OUROBOROS_MODEL", "openai::gpt-5.6-sol")
    cyber = _main_ctx(tmp_path, monkeypatch)
    assert _switch_model(cyber, effort="ultra") == "OK: switching to effort=ultra on next round."
    assert cyber.active_effort_override == "ultra"


def test_switch_model_moves_children_evolution_and_wakes_inside_the_range(tmp_path, monkeypatch):
    from ouroboros.tools.control_runtime import _switch_model

    _env(monkeypatch, OUROBOROS_EFFORT_MIN="low", OUROBOROS_EFFORT_TASK="medium", OUROBOROS_EFFORT_MAX="high")
    monkeypatch.setenv("OUROBOROS_MODEL", "openai::gpt-5.6-sol")
    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "pro")
    auto = _main_ctx(tmp_path, monkeypatch, delegation_role="subagent", configured_subagent=_api_snapshot())
    assert _switch_model(auto, effort="ultra") == (
        "OK: switching to effort=high on next round. (requested ultra, moved into my human's range low..high)")
    assert auto.active_effort_override == "high"
    assert _switch_model(auto, effort="low") == "OK: switching to effort=low on next round."
    pinned = _main_ctx(tmp_path, monkeypatch, delegation_role="subagent", configured_subagent=_api_snapshot(effort="xhigh"))
    assert _switch_model(pinned, effort="low") == (
        "effort=low not applied: my human pinned your row at xhigh; it holds outside Cyber Pro.")
    assert pinned.active_effort_override is None
    named = _main_ctx(tmp_path, monkeypatch, delegation_role="subagent",
                      configured_subagent=_session_snapshot("cursor=grok-4.7-max-fast"), effective_executor="harness")
    out = _switch_model(named, effort="low")
    assert "your row's model name carries max, which holds in every mode" in out and "delegated run keeps" in out
    nanny = _main_ctx(tmp_path, monkeypatch, delegation_role="subagent", configured_subagent=_session_snapshot(),
                      effective_executor="harness")
    out = _switch_model(nanny, effort="ultra")
    assert out.startswith("OK: switching to effort=high on next round.") and "Your delegated run keeps the level it started at" in out
    evolution = _main_ctx(tmp_path, monkeypatch)
    evolution.current_task_type = "evolution"
    assert _switch_model(evolution, effort="none") == (
        "OK: switching to effort=low on next round. (requested none, moved into my human's range low..high)")
    wake = _main_ctx(tmp_path, monkeypatch, model_role="consciousness")
    assert _switch_model(wake, effort="medium") == "OK: switching to effort=medium on next round."
    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "cyber_pro")
    assert _switch_model(pinned, effort="low") == "OK: switching to effort=low on next round."
    assert _switch_model(evolution, effort="ultra") == "OK: switching to effort=ultra on next round."


def test_roots_ouroboros_creates_itself_start_at_recommended_outside_cyber_pro(tmp_path, monkeypatch):
    from ouroboros.tools.control_routing import _root_effort_arg

    _env(monkeypatch, OUROBOROS_EFFORT_TASK="medium")
    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "pro")
    fields, error, note = _root_effort_arg("ultra", "promote_chat_to_task")
    assert fields == {} and error == "" and note == (
        "reasoning_effort='ultra' ignored: outside Cyber Pro a root I create starts at my human's recommended "
        "level (medium); deeper thinking is delegated with schedule_subagent(effort=...)")
    assert _root_effort_arg(None, "promote_chat_to_task") == ({}, "", "")
    _fields, error, _note = _root_effort_arg("turbo", "route_to_project")
    assert error.startswith("⚠️ TOOL_ARG_ERROR (route_to_project): reasoning_effort must be one of")
    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "cyber_pro")
    assert _root_effort_arg("ultra", "promote_chat_to_task") == ({"reasoning_effort": "ultra"}, "", "")


def test_a_followup_outside_cyber_pro_starts_at_recommended_and_says_so(tmp_path, monkeypatch):
    import types

    from ouroboros.tools import followup
    from supervisor import queue_schedules
    from tests.test_root_effort_ingress import _install_queue, _pool_ready

    _env(monkeypatch, OUROBOROS_EFFORT_TASK="medium")
    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "pro")
    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_ready(monkeypatch, workers)
    monkeypatch.setattr(followup, "_is_delegated_subagent", lambda _ctx: False)
    ctx = types.SimpleNamespace(task_id="origin-root", root_task_id="origin-root", current_chat_id=7,
                                drive_root=tmp_path, budget_drive_root=tmp_path, project_id="",
                                task_metadata={"root_task_id": "origin-root"},
                                task_contract={}, is_direct_chat=False, workspace_root=None)
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "origin-root", "running", root_task_id="origin-root", chat_id=7)
    out = followup._handle_schedule_followup(ctx, relation="independent", run_at="2099-01-01T00:00:00Z",
                                             objective="look again", reasoning_effort="xhigh")
    assert out.startswith("FOLLOWUP_SCHEDULED") and "reasoning_effort='xhigh' ignored: outside Cyber Pro" in out
    [row] = queue_schedules.load_schedule_store(tmp_path)["tasks"]
    assert "reasoning_effort" not in row["task"]
    refused = followup._handle_schedule_followup(ctx, relation="independent", run_at="2099-01-01T00:00:00Z",
                                                 objective="again", reasoning_effort="turbo")
    assert "FOLLOWUP_EFFORT_INVALID" in refused


def test_main_starts_at_the_named_level_or_recommended_and_strong_roles_at_the_top(monkeypatch):
    from ouroboros.agent_dispatch import _initial_effort_for

    _env(monkeypatch, OUROBOROS_EFFORT_TASK="medium", OUROBOROS_EFFORT_MAX="ultra")
    monkeypatch.setenv("OUROBOROS_MODEL", "openai::gpt-5.6-sol")
    monkeypatch.delenv("OUROBOROS_MODEL_CONSCIOUSNESS", raising=False)
    assert _initial_effort_for({"id": "r"}, "task") == "medium"
    assert _initial_effort_for({"id": "r", "reasoning_effort": "xhigh"}, "task") == "xhigh"  # the owner's pin
    assert _initial_effort_for({"id": "e"}, "evolution") == "ultra"
    assert _initial_effort_for({"id": "w", "metadata": {"model_role": "consciousness"}}, "task") == "ultra"
    monkeypatch.setenv("OUROBOROS_MODEL", "claudexor::agy=gemini-3.1-pro-low")
    assert _initial_effort_for({"id": "r", "reasoning_effort": "xhigh"}, "task") == "low"  # the name wins
    # A child reads back what its dispatch decided, never Main's name.
    assert _initial_effort_for({"id": "c", "delegation_role": "subagent", "reasoning_effort": "high"}, "task") == "high"


# --- presets, the mind's facts, onboarding -------------------------------------------------


def test_presets_mint_auto_actors_and_keep_reviewer_seats_pinned():
    from ouroboros.reviewer_slot_config import review_pool_rows
    from ouroboros.subscription_install_presets import HarnessDiscovery, compile_install_preset
    from tests.test_subscription_install_presets import LIVE_MODELS

    discoveries = [HarnessDiscovery(h, tuple(LIVE_MODELS[h])) for h in ("claude", "cursor")]
    preset = compile_install_preset(discoveries, settings={"OPENROUTER_API_KEY": "configured",
                                                           "OUROBOROS_MODEL": "openai/gpt-5.6-sol",
                                                           "OUROBOROS_MODEL_LIGHT": "openai/gpt-5.6-luna"})
    assert preset.ok, preset.refusal
    items = {row["subagent_id"]: row for row in json.loads(preset.available_subagents)["items"]}
    assert "effort" not in items["primary-builder"] and items["primary-builder"]["route"]["target_id"] == "claude=claude-opus-5"
    assert items["independent-perspective"]["route"]["target_id"] == "cursor=cursor-grok-4.6-high"
    assert items["independent-perspective"]["effort"] == "high"  # the level rides in the model id
    assert "effort" not in items["fast-scout"]
    pool = review_pool_rows({"OUROBOROS_SUBAGENTS": preset.available_subagents})
    assert pool and all(row.effort for row in pool)  # reviewer seats keep their explicit levels
    assert not any(row.slot_id == "primary-builder" for row in pool)


def test_legacy_fast_scout_is_an_auto_row():
    from ouroboros.configured_subagents import resolve_settings_subagent_candidate

    resolution, _diagnostics = resolve_settings_subagent_candidate({
        "OUROBOROS_MODEL": "openai/gpt-5.6-sol", "OUROBOROS_MODEL_LIGHT": "openai/gpt-5.6-luna",
        "OPENROUTER_API_KEY": "configured"})
    rows = {row.subagent_id: row for row in resolution.config.items}
    assert rows["fast-scout"].effort == ""


def test_the_runtime_block_names_the_range_and_whether_it_binds(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from ouroboros.context import build_runtime_section

    _env(monkeypatch, OUROBOROS_EFFORT_TASK="medium", OUROBOROS_EFFORT_MAX="xhigh")
    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "pro")
    env = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, budget_drive_root=tmp_path)
    section = build_runtime_section(env, {"id": "t", "type": "task"})
    data = json.loads(section.split("## Runtime context\n\n", 1)[1])
    assert data["effort_range"] == {"min": "low", "recommended": "medium", "max": "xhigh", "binds": True}
    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "cyber_pro")
    section = build_runtime_section(env, {"id": "t", "type": "task"})
    assert json.loads(section.split("## Runtime context\n\n", 1)[1])["effort_range"]["binds"] is False


def test_review_rows_on_a_named_main_keep_the_names_level(monkeypatch):
    from ouroboros.gateway.onboarding import review_rows_on_main

    catalog = {"enabled": True, "items": [
        {"subagent_id": "r1", "recommended_use": "x", "review_eligible": True, "effort": "medium",
         "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-sol"}},
        {"subagent_id": "r2", "recommended_use": "y", "review_eligible": True,
         "route": {"kind": "agent_session", "target_id": "cursor=grok-4.7-xhigh-fast"}}]}
    plain = review_rows_on_main(catalog, {"OPENROUTER_API_KEY": "configured", "OUROBOROS_MODEL": "openai/gpt-5.6-sol"})
    assert [row.get("effort") for row in plain["items"]] == ["medium", "xhigh"]
    named = review_rows_on_main(catalog, {"CLAUDEXOR_MODELS_ENABLED": "true",
                                          "OUROBOROS_MODEL": "claudexor::cursor=grok-4.7-max-fast"})
    assert [row.get("effort") for row in named["items"]] == [None, None]


# --- the gateway: the owner endpoint, the state field, the generic save ----------------------


def _effort_app(isolated_settings):
    from starlette.applications import Starlette
    from starlette.routing import Route

    from ouroboros.gateway.owner_effort import api_owner_effort_range

    app = Starlette(routes=[Route("/api/owner/effort-range", endpoint=api_owner_effort_range, methods=["POST"])])
    app.state.drive_root = isolated_settings.parent
    return app


def test_the_owner_endpoint_writes_the_triple_atomically_projects_it_and_audits(monkeypatch, tmp_path):
    from starlette.testclient import TestClient

    from ouroboros import config as cfg
    from ouroboros.gateway.state import effort_range as state_effort_range
    from tests.test_owner_settings_write_seam import isolated_settings as _fixture  # noqa: F401

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings_path = data_dir / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)
    settings_path.write_text(json.dumps({"OUROBOROS_EFFORT_TASK": "high", "OUROBOROS_MODEL": "openai/x"}), encoding="utf-8")
    _env(monkeypatch)
    client = TestClient(_effort_app(settings_path))

    ok = client.post("/api/owner/effort-range", json={"min": "none", "recommended": "medium", "max": "ultra"})
    assert ok.status_code == 200, ok.text
    assert ok.json() == {"ok": True, "effort_range": {"min": "none", "recommended": "medium", "max": "ultra"}}
    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert (stored["OUROBOROS_EFFORT_MIN"], stored["OUROBOROS_EFFORT_TASK"], stored["OUROBOROS_EFFORT_MAX"]) == ("none", "medium", "ultra")
    assert stored["OUROBOROS_MODEL"] == "openai/x"  # the rest of the document is untouched
    import os

    assert (os.environ["OUROBOROS_EFFORT_MIN"], os.environ["OUROBOROS_EFFORT_MAX"]) == ("none", "ultra")  # same-lock projection
    assert state_effort_range() == {"min": "none", "recommended": "medium", "max": "ultra"}  # GET /api/state reads it
    events = [json.loads(line) for line in (data_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    audit = [e for e in events if e.get("type") == "owner_api_action" and e.get("action") == "effort_range"]
    assert audit and audit[-1]["effort_range"]["max"] == "ultra" and audit[-1]["previous_effort_range"]["recommended"] == "high"
    # Every tier is accepted, minimal included; the owner's TASK stays the recommended key.
    assert client.post("/api/owner/effort-range", json={"min": "minimal", "recommended": "minimal", "max": "low"}).status_code == 200
    assert json.loads(settings_path.read_text(encoding="utf-8"))["OUROBOROS_EFFORT_TASK"] == "minimal"


@pytest.mark.parametrize("body, fragment", [
    ({"min": "low", "recommended": "turbo", "max": "high"}, "'recommended' must be one of"),
    ({"min": "high", "recommended": "medium", "max": "ultra"}, "ordered min"),
    ({"min": "low", "recommended": "high", "max": "medium"}, "ordered min"),
    ({"recommended": "medium"}, "'min' must be one of"),
    (["not", "an", "object"], "JSON body must be an object"),
])
def test_the_owner_endpoint_refuses_an_incomplete_or_unordered_triple(monkeypatch, tmp_path, body, fragment):
    from starlette.testclient import TestClient

    from ouroboros import config as cfg

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings_path = data_dir / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)
    resp = TestClient(_effort_app(settings_path)).post("/api/owner/effort-range", json=body)
    assert resp.status_code == 400, resp.text
    assert resp.json()["saved"] is False and resp.json()["code"] == "effort_range_invalid" and fragment in resp.json()["error"]
    assert not settings_path.exists()


def test_the_owner_endpoint_shares_the_write_seam_refusals(monkeypatch, tmp_path):
    from starlette.testclient import TestClient

    from ouroboros import config as cfg
    from tests.test_owner_settings_write_seam import _foreign_lock

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings_path = data_dir / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)
    with _foreign_lock(settings_path):
        resp = TestClient(_effort_app(settings_path)).post(
            "/api/owner/effort-range", json={"min": "low", "recommended": "medium", "max": "high"})
    assert resp.status_code == 503 and resp.json()["code"] == "settings_locked" and resp.json()["saved"] is False
    assert not settings_path.exists()


def test_the_generic_save_accepts_a_tier_per_key_and_refuses_anything_else(monkeypatch, tmp_path):
    from starlette.testclient import TestClient

    from ouroboros import config as cfg
    from tests.test_owner_settings_write_seam import _settings_app

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings_path = data_dir / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)
    client = TestClient(_settings_app(monkeypatch, settings_path))
    refused = client.post("/api/settings", json={"OUROBOROS_EFFORT_TASK": "turbo"})
    assert refused.status_code == 400 and "OUROBOROS_EFFORT_TASK must be one of" in refused.json()["error"]
    assert not settings_path.exists()
    # The order is not judged here (the read is tolerant): TASK above MAX saves, and reads as a widened top.
    saved = client.post("/api/settings", json={"OUROBOROS_EFFORT_TASK": " ULTRA ", "OUROBOROS_EFFORT_MAX": "high"})
    assert saved.status_code == 200, saved.text
    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert (stored["OUROBOROS_EFFORT_TASK"], stored["OUROBOROS_EFFORT_MAX"]) == ("ultra", "high")
    assert effort_range(stored) == {"min": "low", "recommended": "ultra", "max": "ultra"}


def test_the_endpoint_is_indexed_mirrored_and_owner_only_for_the_browser():
    from types import SimpleNamespace

    from ouroboros.browser_policy import _is_effort_range_owner_post
    from ouroboros.gateway.contracts import EffortRange, OwnerEffortRangeResponse, StateResponse
    from ouroboros.gateway.endpoint_index import HTTP_ENDPOINTS

    assert "POST /api/owner/effort-range" in HTTP_ENDPOINTS
    from typing import get_type_hints

    assert set(EffortRange.__annotations__) == {"min", "recommended", "max"}
    assert get_type_hints(OwnerEffortRangeResponse)["effort_range"] is EffortRange
    assert get_type_hints(StateResponse)["effort_range"] is EffortRange
    assert _is_effort_range_owner_post(SimpleNamespace(method="POST", url="http://127.0.0.1:8765/api/owner/effort-range"))
    assert _is_effort_range_owner_post(SimpleNamespace(method="POST", url="http://127.0.0.1:8765/api/owner/effort%2Drange/"))
    assert not _is_effort_range_owner_post(SimpleNamespace(method="GET", url="http://127.0.0.1:8765/api/owner/effort-range"))


@pytest.mark.parametrize("role", ["evolution", "consciousness"])
@pytest.mark.parametrize("mode", ["pro", "cyber_pro"])
def test_an_owner_task_pin_holds_against_an_evolution_or_wake_self_switch(tmp_path, monkeypatch, role, mode):
    """Evolution tasks and wakes may move their own level inside the range, but an explicit
    effort the owner gave the task (API, CLI, schedule) is a pin outside Cyber Pro."""
    from ouroboros.tools.control_runtime import _switch_model

    _env(monkeypatch, OUROBOROS_EFFORT_MIN="low", OUROBOROS_EFFORT_TASK="medium", OUROBOROS_EFFORT_MAX="high")
    monkeypatch.setenv("OUROBOROS_MODEL", "openai::gpt-5.6-sol")
    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", mode)
    metadata = {"model_role": "consciousness"} if role == "consciousness" else {}
    pinned = _main_ctx(tmp_path, monkeypatch, **metadata, reasoning_effort="xhigh")
    pinned.current_task_type = "evolution" if role == "evolution" else "task"
    out = _switch_model(pinned, effort="low")
    if mode == "cyber_pro":
        assert pinned.active_effort_override == "low"
    else:
        assert pinned.active_effort_override is None
        assert "pinned this task at xhigh" in out
    free = _main_ctx(tmp_path, monkeypatch, **metadata)
    free.current_task_type = pinned.current_task_type
    _switch_model(free, effort="low")
    assert free.active_effort_override == "low", "without a pin the role still moves inside the range"


def test_a_session_start_failure_keeps_the_rows_configured_effort_as_identity(tmp_path):
    """An Auto row whose leaf was started at high and refused must not read as "Earlier
    settings": the receipt identity is the row's pin ('' = Auto) on both terminal paths."""
    from ouroboros import delegate_custody as custody
    from ouroboros.subagent_history import session_request_facts, subagent_last_delegation

    start = {"model": "m", "effort": "high", "access": "full"}
    facts = session_request_facts(start, selected_subagent_id="worker", task_id="task", route="codex",
                                  processing={"requested": ""}, row_effort="")
    assert custody.emit(tmp_path, custody.START_FAILED, {
        **facts, "invocation_id": "failed-attempt", "definite": True, "reason": "start_refused"})
    assert subagent_last_delegation(tmp_path)["identity"]["effort"] == ""
    # A recovery fact without the row's pin keeps the level it always recorded.
    legacy = session_request_facts(start, selected_subagent_id="worker", task_id="task", route="codex",
                                   processing={"requested": ""})
    assert custody.emit(tmp_path, custody.START_FAILED, {
        **legacy, "invocation_id": "legacy-attempt", "definite": True, "reason": "start_refused"})
    assert subagent_last_delegation(tmp_path)["identity"]["effort"] == "high"
