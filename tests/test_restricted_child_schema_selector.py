"""#1860: a restricted child whose permitted catalog exceeds its route's schema ceiling.

Read-only and acting children carry the existing selector (``enable_tools``) in every
context mode. It loads a schema the child may already call and grants nothing; the fit
notice and the discovery header name only the selectors the actor can use. Real registry,
real fitting and real dispatch on the direct OpenAI route's declared ceiling (128), with
synthetic extension descriptors standing in for a large skill/MCP catalog; no provider send.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import extension_loader, loop, loop_model_call
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.extension_surface_names import extension_surface_name
from ouroboros.provider_models import tool_schema_limit
from ouroboros.tool_policy import initial_tool_schemas
from ouroboros.tools.registry import ToolContext, ToolRegistry

OPENAI = "openai::gpt-5.6-terra"
PROFILES = ("local_readonly_subagent", "acting_subagent")
EXTRA = 100  # synthetic extension tools: both profiles exceed 128 schemas


def _names(schemas):
    return [schema["function"]["name"] for schema in schemas]


@pytest.fixture
def extensions(monkeypatch):
    names = [extension_surface_name("synthetic", f"lookup_{index:03d}") for index in range(EXTRA)]
    monkeypatch.setattr(extension_loader, "is_extension_live", lambda *_a, **_k: True)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    with extension_loader._lock:
        for name in names:
            extension_loader._tools[name] = {
                "name": name, "description": f"Synthetic lookup {name}", "timeout_sec": 5, "skill": "synthetic",
                "schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                "handler": lambda ctx, query="", _name=name: f"{_name}:{query}",
            }
    yield names
    with extension_loader._lock:
        for name in names:
            extension_loader._tools.pop(name, None)


def _child(tmp_path, monkeypatch, profile, mode, *, grants=None, contract=None, metadata=None):
    repo, data, workspace = (tmp_path / name for name in ("repo", "data", "workspace"))
    for path in (repo, data, workspace):
        path.mkdir(parents=True, exist_ok=True)
    for key, value in (("OUROBOROS_RUNTIME_MODE", "advanced"), ("OUROBOROS_SAFETY_MODE", "off"),
                       ("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")):
        monkeypatch.setenv(key, value)
    constraint = (TaskConstraint(mode=profile, allow_enable=False) if profile == "local_readonly_subagent" else
                  TaskConstraint(mode=profile, surface="external_workspace", write_root=str(workspace),
                                 external_tool_grants=tuple(grants or ())))
    ctx = ToolContext(repo_dir=repo, drive_root=data, workspace_root=workspace, workspace_mode="external",
                      task_id=f"selector-{profile}-{mode}", active_context_mode=mode, task_constraint=constraint,
                      task_contract=dict(contract or {}), task_metadata=dict(metadata or {}))
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ctx)
    return registry


def _round(registry, resident):
    """The Main round's (and a wait's reprepare's) fitting context around the real registry."""
    return SimpleNamespace(tool_schemas=resident, active_model=OPENAI, active_use_local=False,
                           task_id=registry._ctx.task_id, messages=[{"role": "user", "content": "go"}],
                           tools=registry, defer_resource_wait=None, attempt_cap=None, accumulated_usage={},
                           active_context_mode=registry._ctx.active_context_mode)


@pytest.mark.parametrize("mode", ["low", "max"])
@pytest.mark.parametrize("profile", PROFILES)
def test_an_omitted_permitted_schema_loads_stays_through_the_next_fit_and_dispatches(
        tmp_path, monkeypatch, extensions, profile, mode):
    registry = _child(tmp_path, monkeypatch, profile, mode, grants=extensions)
    resident = initial_tool_schemas(registry, context_mode=mode)
    assert tool_schema_limit(OPENAI) == 128 < len(resident)
    assert "enable_tools" in _names(resident) and "compact_context" not in _names(resident)
    loop._setup_dynamic_tools(registry, resident, [], context_mode=mode)
    first = _round(registry, resident)
    assert loop_model_call._fit_route_tool_ceiling(first)
    left_out = set(registry._ctx._route_left_out_tool_names)
    assert len(resident) == 128 and left_out and left_out <= set(extensions)
    notice = first.messages[-1]["content"]
    assert "list_available_tools marks them not loaded, and enable_tools loads one" in notice
    chosen = extensions[-1]  # the late-position tail: what a plain truncation would drop first
    assert chosen in left_out
    listing = registry.execute("list_available_tools", {"namespace": chosen.rsplit("_lookup", 1)[0]})
    assert "enable_tools loads a not-loaded one" in listing and chosen in listing

    loaded = registry.execute("enable_tools", {"tools": chosen})
    assert "Call schemas loaded" in loaded and chosen in loaded
    assert _names(resident)[-1] == chosen and len(resident) == 129
    assert chosen in registry._ctx._actor_loaded_tool_names
    second = _round(registry, resident)  # the next request fits the same list again
    assert loop_model_call._fit_route_tool_ceiling(second)
    assert chosen in _names(resident) and len(resident) == 128
    assert "enable_tools" in _names(resident) and "list_available_tools" in _names(resident)
    assert registry.execute(chosen, {"query": "q"}) == f"{chosen}:q"
    assert loop_model_call._fit_route_tool_ceiling(_round(registry, resident)) is False  # within the ceiling


@pytest.mark.parametrize("mode", ["low", "max", "nano"])
@pytest.mark.parametrize("profile", PROFILES)
def test_a_fitting_catalog_keeps_its_envelope_and_the_selector_grants_nothing(tmp_path, monkeypatch, profile, mode):
    registry = _child(tmp_path, monkeypatch, profile, mode)
    resident = initial_tool_schemas(registry, context_mode=mode)
    assert len(resident) <= 128 and "enable_tools" in _names(resident)
    assert ("compact_context" in _names(resident)) is (mode == "nano")
    before = list(resident)
    loop._setup_dynamic_tools(registry, resident, [], context_mode=mode)
    assert loop_model_call._fit_route_tool_ceiling(_round(registry, resident)) is False
    assert resident == before
    forbidden = ("write_file", "commit_reviewed", "run_command") if profile == "local_readonly_subagent" else (
        "commit_reviewed", "request_restart", "update_identity")
    for name in forbidden:
        result = registry.execute_result("enable_tools", {"tools": name})
        assert result.status == "blocked" and "Hidden by policy" in result.text, result.text
        assert registry.execute_result(name, {}).status == "blocked"
    assert resident == before


def test_the_acting_work_order_forbids_tool_grants_not_the_selector_its_registry_carries(tmp_path, monkeypatch):
    """The [WRITE SURFACE] text frozen into the queued task, and the registry built from the same constraint."""
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from supervisor.events import _compose_subagent_text

    order = {"mode": "acting_subagent", "surface": "external_workspace", "write_root": str(tmp_path / "workspace")}
    surface = _compose_subagent_text("obj", role="builder", expected_output="out", constraints="", context="",
                                     task_constraint=order).split("[WRITE SURFACE]", 1)[1]
    forbidden = surface.split("Do NOT ", 1)[1].split(". ", 1)[0]  # the one sentence of prohibitions
    assert "grant yourself tools" in forbidden and "enable" not in forbidden, forbidden
    registry = _child(tmp_path, monkeypatch, "acting_subagent", "max")
    registry._ctx.task_constraint = normalize_task_constraint(order)
    resident = initial_tool_schemas(registry, context_mode="max")
    assert "enable_tools" in _names(resident)
    loop._setup_dynamic_tools(registry, resident, [], context_mode="max")
    before = list(resident)
    refused = registry.execute_result("enable_tools", {"tools": "commit_reviewed"})  # a grant it may not give itself
    assert refused.status == "blocked" and "Hidden by policy" in refused.text and resident == before


def test_an_acting_child_cannot_load_or_call_an_ungranted_extension(tmp_path, monkeypatch, extensions):
    granted, ungranted = extensions[:-1], extensions[-1]
    registry = _child(tmp_path, monkeypatch, "acting_subagent", "max", grants=granted)
    resident = initial_tool_schemas(registry, context_mode="max")
    assert ungranted not in _names(resident) and granted[-1] in _names(resident)
    loop._setup_dynamic_tools(registry, resident, [], context_mode="max")
    refused = registry.execute_result("enable_tools", {"tools": ungranted})
    assert refused.status != "ok" and ungranted not in _names(resident)
    assert "ACTING_SUBAGENT_TOOL_NOT_GRANTED" in registry.execute(ungranted, {"query": "q"})


@pytest.mark.parametrize("disabled", [("enable_tools",), ("list_available_tools",),
                                      ("enable_tools", "list_available_tools")])
@pytest.mark.parametrize("dispatch_only", [False, True])
def test_a_selector_the_actor_cannot_call_is_never_recommended(tmp_path, monkeypatch, extensions,
                                                               disabled, dispatch_only):
    """A contract's disabled selector, including a consciousness-origin task whose disabled schema stays visible."""
    from ouroboros.tools import tool_discovery

    registry = _child(tmp_path, monkeypatch, "local_readonly_subagent", "max",
                      contract={"disabled_tools": list(disabled)},
                      metadata={"initiator": "consciousness"} if dispatch_only else None)
    if dispatch_only:
        registry._ctx.task_constraint = None  # a root task the consciousness started
    resident = initial_tool_schemas(registry, context_mode="max")
    if dispatch_only:
        assert set(disabled) <= set(_names(resident))  # dispatch-only policy keeps the schema visible
    else:
        assert not set(disabled) & set(_names(resident))
    loop._setup_dynamic_tools(registry, resident, [], context_mode="max")
    fitted = _round(registry, resident)
    assert loop_model_call._fit_route_tool_ceiling(fitted)
    notice = fitted.messages[-1]["content"]
    for name in ("enable_tools", "list_available_tools"):
        assert (name in notice) is (name not in disabled), notice
    assert notice.rstrip().endswith(".") and "They stay callable" in notice
    listing = tool_discovery.list_available_tools(registry, registry._ctx, resident)
    assert ("enable_tools loads" in listing) is ("enable_tools" not in disabled)
    assert registry.execute_result("enable_tools", {"tools": "read_file"}).status == (
        "blocked" if "enable_tools" in disabled else "ok")
    miss = registry.execute("definitely_not_a_tool", {})
    assert ("list_available_tools" in miss) is ("list_available_tools" not in disabled)


def test_a_low_or_max_diagnostic_keeps_its_role_and_is_not_told_to_call_the_selector(tmp_path, monkeypatch):
    from ouroboros import provider_models
    from ouroboros.tool_capabilities import TASK_DIAGNOSTIC_TOOL_NAMES

    monkeypatch.setattr("ouroboros.work_diagnostics.context_ref", lambda ctx: {"task_id": "observed"})
    registry = _child(tmp_path, monkeypatch, "local_readonly_subagent", "max")
    resident = initial_tool_schemas(registry, context_mode="max")
    assert set(_names(resident)) <= TASK_DIAGNOSTIC_TOOL_NAMES and "enable_tools" not in _names(resident)
    monkeypatch.setitem(provider_models.PROVIDER_TOOL_SCHEMA_LIMITS, "openai", len(resident) - 2)
    loop._setup_dynamic_tools(registry, resident, [], context_mode="max")
    fitted = _round(registry, resident)
    assert loop_model_call._fit_route_tool_ceiling(fitted)
    notice = fitted.messages[-1]["content"]
    assert "enable_tools" not in notice and "list_available_tools marks them not loaded." in notice
    assert registry.execute_result("enable_tools", {"tools": "read_file"}).status == "blocked"


@pytest.mark.parametrize("profile", PROFILES)
def test_a_cold_continuation_keeps_the_loaded_late_schema_and_a_callable_selector(
        tmp_path, monkeypatch, extensions, profile):
    """Real registry through the existing cold-resume rail (owner_wait restore + reconcile)."""
    from tests import test_continuation_tools as harness

    registry = _child(tmp_path, monkeypatch, profile, "low", grants=extensions)
    first = initial_tool_schemas(registry, context_mode="low")
    loop._setup_dynamic_tools(registry, first, [], context_mode="low")
    loop_model_call._fit_route_tool_ceiling(_round(registry, first))
    chosen = extensions[-1]
    assert chosen not in _names(first)
    registry.execute("enable_tools", {"tools": chosen})
    loop_model_call._fit_route_tool_ceiling(_round(registry, first))
    assert chosen in _names(first) and len(first) == 128
    frozen = json.loads(json.dumps(first))  # what the saved continuation carries
    # The new process finds one earlier extension gone, so the fitted list differs and reconciliation
    # adopts current residency (an unchanged policy reproduces the frozen list and stays quiet).
    with extension_loader._lock:
        extension_loader._tools.pop(extensions[0])

    cold = _child(tmp_path / "cold", monkeypatch, profile, "low", grants=extensions)  # a new process
    ctx = cold._ctx
    for key, value in dict(task_attempt=1, context_fit_plan=None, active_model=OPENAI, active_use_local=False,
                           active_effort="high", budget_drive_root=ctx.drive_root, owner_wait_resume={},
                           drive_logs=lambda: ctx.drive_root / "logs", event_queue=None,
                           owner_wait_callback=lambda *_: "owner_input", model_wait_context=None).items():
        setattr(ctx, key, value)
    monkeypatch.setattr(harness, "Registry", lambda *_args: cold)
    prepared = harness.prepare(tmp_path, monkeypatch, mode="low", frozen=frozen, loaded=[chosen],
                               history=[{"role": "assistant", "tool_calls": [{"function": {"name": chosen}}]}])
    _tools, state, messages, _trace, enabled, _projections = prepared
    state["route"]["active_model"] = OPENAI
    resident = harness.restore(prepared)
    assert chosen in _names(resident) and "enable_tools" in _names(resident) and len(resident) == 128
    assert cold._ctx._actor_loaded_tool_names is enabled and chosen in enabled
    assert extensions[0] not in _names(resident) and any("tool set changed" in str(row.get("content"))
                                                          for row in messages)
    notices = [row["content"] for row in messages if "tool schemas in one request" in str(row.get("content"))]
    assert notices and "list_available_tools marks them not loaded, and enable_tools loads one" in notices[-1]
    another = sorted(set(extensions[1:]) - set(_names(resident)))[-1]
    receipt = cold.execute("enable_tools", {"tools": another})
    assert "Call schemas loaded" in receipt and another in receipt
    assert another in enabled and cold.execute(chosen, {"query": "c"}) == f"{chosen}:c"
