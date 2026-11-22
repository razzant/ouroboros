"""What the model is shown and may call, pinned per actor context (п.84).

One catalog state (this tree's built-ins, one live extension tool, one pinned and
one unpinned MCP server) is read through every visibility reader of the registry
in a matrix of contexts. Per context the record keeps the decisions a request is
built from, never schema bytes: the canonical array (``schemas()``, names in
order) with its omission rows; the initial selection the loop really starts from
(``tool_policy.initial_tool_schemas``) for Nano, Low and Max; and what each
transport is passed after its route restriction (``fit_tool_schemas_to_limit``
under the route's own ``tool_schema_limit``, then the transport's own order:
OpenAI-compatible and Anthropic sort by name, local and Claudexor keep the
registry order) with what it left out. Beside them: ``available_tools``,
``policy_hidden_reason`` of every name it answers for, which names
``get_schema_by_name`` does not answer or answers otherwise than the canonical
array, ``callable_rows``, the parameter names and narrowed enums of every tool
whose call schema differs from Main's (narrowing in ``_schema_for_entry`` is a
visibility decision), and whether a child's substituted description replaced the
entry's. A consciousness wake also pins equality with an owner turn and the
dispatch refusal of a withheld name (owner decision В31=B). A description edit
does not move the record.

Byte identity was proven once, for the change that unified the five readers:
the record of that time held every schema's bytes and was written green on the
unchanged base (b9513746d); after one deliberate move (985e90ccd: the network
gate in available_tools) the unification itself (165089dc6) did not change its
parent's fixture and stayed green. What stays pinned is the decisions.

A deliberate change of what any context sees or may call regenerates the record
with ``OURO_TOOL_VISIBILITY_GOLDEN=write``; name it in the PR. An unintended one
fails here with the arrays that moved.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import pathlib

import pytest

from ouroboros import consciousness_authority as ca
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools.registry import ToolContext, ToolRegistry

FIXTURE = pathlib.Path(__file__).with_name("fixtures") / "tool_visibility_golden.json"
WRITE = os.environ.get("OURO_TOOL_VISIBILITY_GOLDEN", "") == "write"

EXT_SKILL = "weather"
MCP_PINNED = "mcp_pinned__lookup"
MCP_LAZY = "mcp_lazy__search"
LEGACY_SKILL = "legacy"
# The prose of the dynamic surfaces, kept here so a test can reword it as a catalog update would.
DYNAMIC_PROSE = {MCP_PINNED: "Lookup records.", MCP_LAZY: "Search the archive.", EXT_SKILL: "Forecast weather."}
# A transport per route family; each one's own ceiling and projection is read from the code.
ROUTES = {
    "openai_direct": ("openai::gpt-5.6-terra", False),  # OpenAI's own API: 128 schemas at most
    "anthropic": ("anthropic/claude-opus-4.5", False),
    "local": ("local-model", True),
    "claudexor": ("claudexor::codex=gpt-6.2-astra", False),
}
# Names discovery is asked about besides the registered built-ins.
EXTRA_NAMES = ("definitely_not_a_tool", "calendar")
# Tools whose description a child profile replaces (``ToolRegistry._schema_for_entry``).
CHILD_DESCRIBED = ("verify_and_record", "browse_page", "browser_action", "review_change")


def _h(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode("utf-8")).hexdigest()


def _ext_name() -> str:
    from ouroboros import extension_loader

    return extension_loader.extension_surface_name(EXT_SKILL, "forecast")


def _legacy_name() -> str:
    from ouroboros import extension_loader

    return extension_loader.extension_surface_name(LEGACY_SKILL, "echo")


def _wake(level: str) -> dict:
    task = {"id": "wake-1", "type": "task", "text": "wake", "_is_direct_chat": True, "chat_id": 1,
            "metadata": {"initiator": "consciousness", "usage_category": "consciousness",
                         "wake_reason": "heartbeat", "consciousness_autonomy": level,
                         "model_role": "consciousness"}}
    return ca.apply_consciousness_authority(task)["metadata"]


def _presence_contract() -> dict:
    from ouroboros.presence_authority import (
        PresenceCapabilityCeiling, PresenceToolGrant, presence_ceiling_payload,
    )

    grants = tuple(PresenceToolGrant(name) for name in (
        "read_file", "send_user_message", "vcs_commit_reviewed", MCP_PINNED, _ext_name()))
    ceiling = PresenceCapabilityCeiling(
        skill_name="behavior", skill_content_hash="a" * 64, profile_fingerprint="b" * 64,
        state_fingerprint="c" * 64, selection_fingerprint="d" * 64, model_slot="main",
        inline_max_rounds=3, tool_grants=grants, resource_grants=(), digest="0" * 64)
    return {"capability_ceiling": presence_ceiling_payload(ceiling)}


def _contexts(root: pathlib.Path) -> dict:
    worktree = root / "wt"
    worktree.mkdir(parents=True, exist_ok=True)
    main = {"task_id": "turn-1", "is_direct_chat": True, "task_metadata": {"client_message_id": "cm-1"}}
    readonly = {"task_id": "child-r", "task_metadata": {"delegation_role": "subagent"},
                "task_constraint": TaskConstraint(mode="local_readonly_subagent")}
    acting = {"task_id": "child-a", "task_metadata": {"delegation_role": "subagent"},
              "workspace_root": worktree, "workspace_mode": "self_worktree",
              "task_constraint": TaskConstraint(mode="acting_subagent", surface="self_worktree",
                                                write_root=str(worktree), external_tool_grants=(MCP_PINNED,))}
    return {
        "main": main,
        "root_work": {"task_id": "work-1", "task_metadata": {"task_contract": {"goal": "x"}}},
        "consciousness_act": {**main, "task_metadata": _wake("act")},
        "consciousness_observe": {**main, "task_metadata": _wake("observe")},
        "consciousness_full": {**main, "task_metadata": _wake("full")},
        "readonly_helper": readonly,
        # A configured actor-first session before any leaf: its one receipt, verify_and_record, is shown narrowed.
        "readonly_actor_bootstrap": {**readonly, "_actor_bootstrap": {"physical_started": False,
                                                                       "zero_run_receipt_recorded": False}},
        "acting_helper": acting,
        "acting_helper_cyber_pro": {**acting, "_runtime_mode": "cyber_pro"},
        "presence": {"task_id": "presence-1", "task_contract": _presence_contract()},
        "nano_main": {**main, "active_context_mode": "nano"},
        "nano_readonly_helper": {**readonly, "active_context_mode": "nano"},
        "low_main": {**main, "active_context_mode": "low"},
        "network_false": {"task_id": "work-n", "task_contract": {"allowed_resources": {"network": False}}},
        "web_false_network_true": {"task_id": "work-w",
                                   "task_contract": {"allowed_resources": {"web": False, "network": True}}},
        "disabled_by_contract": {"task_id": "work-d", "task_contract": {"disabled_tools": [
            "toggle_evolution", "web_search", "advisory_review", _ext_name(), MCP_LAZY]}},
        "unavailable_key": {**main, "_unavailable": True},
        "extension_generation_change": {**main, "_stale_extension": True},
        "bare_registry": {},
    }


class _Manager:
    """One pinned and one unpinned MCP server, as ``mcp_client`` publishes them."""

    def __init__(self):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        self.rows = [
            {"name": MCP_PINNED, "raw_name": "lookup", "description": DYNAMIC_PROSE[MCP_PINNED],
             "raw_description": DYNAMIC_PROSE[MCP_PINNED], "schema": schema, "server_id": "pinned",
             "schemas_at_start": True, "metadata": {"annotations": {"readOnlyHint": True}}},
            {"name": MCP_LAZY, "raw_name": "search", "description": DYNAMIC_PROSE[MCP_LAZY],
             "raw_description": DYNAMIC_PROSE[MCP_LAZY], "schema": schema, "server_id": "lazy",
             "schemas_at_start": False, "metadata": {}},
        ]

    def list_tools_for_registry(self):
        return [dict(row) for row in self.rows]

    def tool_name_collisions(self):
        return []

    def enabled_servers_without_tools(self):
        return []

    def catalog_notices(self):
        return []

    def get_tool(self, name):
        return next((dict(row) for row in self.rows if row["name"] == name), None)


@contextlib.contextmanager
def catalog(name: str, spec: dict, root: pathlib.Path):
    """The one catalog state a context reads, undone after it; credentials and backends present unless it says not."""
    from ouroboros import extension_loader
    from ouroboros import extension_registry_state as state

    # The process-global extension registry is this context's own: whatever an earlier test in this
    # worker left there is set aside first and put back afterwards (in place: modules hold the dicts).
    assert extension_loader._tools is state._tools and extension_loader._extensions is state._extensions
    with state._lock:
        held = dict(state._tools), dict(state._extensions)
        state._tools.clear()
        state._extensions.clear()
    try:
        with _catalog_patches(name, dict(spec), root) as registry:
            yield registry
    finally:
        with state._lock:
            state._tools.clear()
            state._tools.update(held[0])
            state._extensions.clear()
            state._extensions.update(held[1])


@contextlib.contextmanager
def _catalog_patches(name: str, spec: dict, root: pathlib.Path):
    from ouroboros import extension_loader, mcp_client
    from ouroboros import extension_registry_state as state
    from ouroboros.tools import extension_dispatch, github, search

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
        mp.setenv("OUROBOROS_CONTEXT_MODE", "max")
        mp.setenv("GITHUB_TOKEN", "synthetic-token")
        mp.setattr(search, "_available_web_search_backends", lambda: ["synthetic"])
        mp.setattr(github, "github_cli_configured", lambda: True)
        mp.setattr(mcp_client, "ensure_configured_from_settings", lambda **_kw: None)
        manager = _Manager()
        mp.setattr(mcp_client, "get_manager", lambda: manager)
        mp.setitem(extension_loader._tools, _ext_name(), {
            "name": _ext_name(), "handler": lambda ctx: "ok", "description": DYNAMIC_PROSE[EXT_SKILL],
            "schema": {"type": "object", "properties": {}}, "timeout_sec": 5, "skill": EXT_SKILL,
        })
        live = {EXT_SKILL: True}
        mp.setattr(extension_loader, "is_extension_live", lambda skill, *_a, **_k: live.get(skill, False))
        mp.setattr(extension_dispatch, "_adopt_extension_generation", lambda _ctx: {"action": "in_sync"})
        if spec.pop("_runtime_mode", "") == "cyber_pro":
            from ouroboros.config import get_runtime_mode

            # The product's own scale must hold the mode; the row never adds it for itself.
            mp.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
            assert get_runtime_mode() == "cyber_pro", "the effective runtime mode is not Cyber Pro"
        if spec.pop("_unavailable", False):
            mp.delenv("GITHUB_TOKEN", raising=False)
            mp.setattr(github, "github_cli_configured", lambda: False)
            mp.setattr(search, "_available_web_search_backends", lambda: [])
            mp.setattr(search, "_web_search_unavailability",
                       lambda: ("missing_credential", "web_search_backend: synthetic"))
        if spec.pop("_stale_extension", False):
            # #2933: a descriptor of a previous generation is not live until the registry adopts the new one.
            surface = _legacy_name()
            mp.setitem(state._tools, surface, {"name": surface, "skill": LEGACY_SKILL,
                                               "description": "legacy echo", "schema": {}})
            mp.setitem(state._extensions, LEGACY_SKILL, state._ExtensionRegistrations(generation_digest="old"))
            live[LEGACY_SKILL] = False

            def adopt(_ctx):
                state._extensions[LEGACY_SKILL] = state._ExtensionRegistrations(generation_digest="new")
                live[LEGACY_SKILL] = True
                return {"action": "reloaded"}

            mp.setattr(extension_dispatch, "_adopt_extension_generation", adopt)
        base = root / name
        repo, drive = base / "repo", base / "drive"
        repo.mkdir(parents=True)
        drive.mkdir()
        (repo / "README.md").write_text("ok\n", encoding="utf-8")
        bootstrap = spec.pop("_actor_bootstrap", None)
        registry = ToolRegistry(repo_dir=repo, drive_root=drive)
        if spec:
            registry.set_context(ToolContext(repo_dir=repo, drive_root=drive, **spec))
        if bootstrap is not None:
            registry._ctx._configured_actor_bootstrap = dict(bootstrap)
        yield registry


def _names(schemas) -> list:
    return [s["function"]["name"] for s in schemas]


def _transport_order(transport: str, schemas: list) -> list:
    """The tool names in the order each transport sends them, from the transport's own projection."""
    from ouroboros.llm import LLMClient

    if transport == "openai_direct":
        sent = LLMClient._sanitize_chat_completion_tools(schemas)
    elif transport == "anthropic":
        sent = LLMClient._build_anthropic_tools(schemas)
    elif transport == "local":  # llm_local: only cache marks are dropped, the registry order is sent
        sent = [{k: v for k, v in tool.items() if k != "cache_control"} for tool in schemas]
    else:
        sent = copy.deepcopy(schemas)  # llm_claudexor deep-copies the array as built
    return [tool.get("name") or tool["function"]["name"] for tool in sent]


def _shape(schema: dict) -> dict:
    """What a call schema lets the model ask for: its parameter names and every enum, never its prose."""
    properties = (schema["function"].get("parameters") or {}).get("properties") or {}
    return {"params": sorted(properties),
            "enums": {key: list(value["enum"]) for key, value in sorted(properties.items())
                      if isinstance(value, dict) and isinstance(value.get("enum"), list)}}


def snapshot(registry: ToolRegistry) -> dict:
    """Every visibility answer of one registry; ``shapes`` stay whole here (the record keeps Main's differences)."""
    from ouroboros.provider_models import tool_schema_limit
    from ouroboros.tool_policy import fit_tool_schemas_to_limit, initial_tool_schemas

    canonical = registry.schemas()
    omissions = registry.capability_omissions()
    initial = {mode: initial_tool_schemas(registry, context_mode=mode) for mode in ("nano", "low", "max")}
    for selected in initial.values():  # every selection is a subsequence of the canonical array
        assert [s for s in canonical if s in selected] == selected
    own = initial[getattr(registry._ctx, "active_context_mode", "") or "max"]
    passed = {}
    for transport, (model, use_local) in ROUTES.items():
        fitted, left_out = fit_tool_schemas_to_limit(own, tool_schema_limit(model, use_local=use_local))
        passed[transport] = {"names": _transport_order(transport, fitted), "left_out": list(left_out)}
    by_canonical = {s["function"]["name"]: s for s in canonical}
    asked = [*registry._entries, _ext_name(), _legacy_name(), MCP_PINNED, MCP_LAZY, *EXTRA_NAMES]
    answers = {name: registry.get_schema_by_name(name) for name in asked}
    hidden = {name: reason for name in asked if (reason := registry.policy_hidden_reason(name))}
    rows = registry.callable_rows()
    return {
        "canonical": _names(canonical),
        "omissions": omissions,
        "initial": {mode: _names(schemas) for mode, schemas in initial.items()},
        "passed": passed,
        "available_tools": list(registry.available_tools()),
        "policy_hidden_reason": hidden,
        # Named discovery agrees with the canonical array: shown names answer the same schema, others none.
        "get_schema_by_name_absent": sorted(name for name, schema in answers.items()
                                            if schema is None and name not in by_canonical),
        "get_schema_by_name_differs": sorted(name for name, schema in answers.items()
                                             if name in by_canonical and schema != by_canonical[name]
                                             or name not in by_canonical and schema is not None),
        # A built-in row is its name; an MCP row keeps its raw name and marks, never the server's prose.
        "callable_rows": [row["name"] if set(row) == {"name"} else
                          {key: row[key] for key in ("name", "raw_name", "marks") if key in row} for row in rows],
        "shapes": {name: _shape(schema) for name, schema in by_canonical.items()},
        "child_described": {name: registry._entries[name].schema.get("description")
                            != by_canonical[name]["function"].get("description")
                            for name in CHILD_DESCRIBED if name in by_canonical},
    }


def _against_main(records: dict) -> dict:
    """The record form: each tool's call schema only where it differs from Main's (params, changed enums)."""
    main = records["main"]["shapes"]
    result = {}
    for name, record in records.items():
        record = dict(record)
        shapes = record.pop("shapes")
        narrowed = {}
        for tool, shape in shapes.items():
            if shape == main.get(tool):
                continue
            usual = (main.get(tool) or {}).get("enums", {})
            changed = {key: values for key, values in shape["enums"].items() if usual.get(key) != values}
            narrowed[tool] = {"params": shape["params"], **({"enums": changed} if changed else {})}
        record["narrowed_against_main"] = narrowed
        result[name] = record
    return result


def _diff(expected: dict, actual: dict) -> str:
    lines = []
    for key in sorted(set(expected) | set(actual)):
        if expected.get(key) == actual.get(key):
            continue
        old, new = expected.get(key), actual.get(key)
        if isinstance(old, list) and isinstance(new, list):
            gone = [item for item in old if item not in new][:12]
            came = [item for item in new if item not in old][:12]
            lines.append(f"  {key}: removed {gone} added {came}" if gone or came else f"  {key}: order changed")
        else:
            lines.append(f"  {key}: {json.dumps(old, ensure_ascii=False)[:400]} -> {json.dumps(new, ensure_ascii=False)[:400]}")
    return "\n".join(lines)


def _flatten(record: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in record.items():
        if isinstance(value, dict) and key not in ("policy_hidden_reason", "narrowed_against_main", "child_described"):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _snapshots(root: pathlib.Path, names=None) -> dict:
    result = {}
    for name, spec in _contexts(root).items():
        if names is None or name in names:
            with catalog(name, spec, root) as registry:
                result[name] = snapshot(registry)
    return result


def _pool(records: dict) -> dict:
    """The stored form: every array stored once by the digest of its content; a record refers to it as ``@digest``."""
    arrays: dict = {}

    def put(value):
        if isinstance(value, (list, dict)) and len(json.dumps(value)) > 160:
            key = _h(value)[:12]
            arrays[key] = value
            return f"@{key}"
        return value

    def walk(node, depth):
        if isinstance(node, dict) and depth < 3:
            return {key: walk(value, depth + 1) for key, value in node.items()}
        return put(node)

    contexts = {name: {key: walk(value, 1) for key, value in record.items()} for name, record in records.items()}
    return {"arrays": arrays, "contexts": contexts}


def _expand(pooled: dict) -> dict:
    arrays = pooled["arrays"]

    def walk(node):
        if isinstance(node, str) and node.startswith("@") and node[1:] in arrays:
            return arrays[node[1:]]
        if isinstance(node, dict):
            return {key: walk(value) for key, value in node.items()}
        return node

    return {name: walk(record) for name, record in pooled["contexts"].items()}


def test_what_every_context_sees_and_may_call_is_pinned(tmp_path):
    actual = _against_main(_snapshots(tmp_path))
    if WRITE:
        FIXTURE.write_text(json.dumps(_pool(actual), ensure_ascii=False, indent=0, sort_keys=True) + "\n",
                           encoding="utf-8")
    expected = _expand(json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert sorted(expected) == sorted(actual), "the context matrix itself changed"
    moved = {name: _diff(_flatten(expected[name]), _flatten(actual[name]))
             for name in actual if expected[name] != actual[name]}
    assert not moved, ("what the model sees moved (OURO_TOOL_VISIBILITY_GOLDEN=write regenerates a deliberate change):\n"
                       + "\n".join(f"{name}:\n{text}" for name, text in moved.items()))


@pytest.mark.parametrize("level", ["act", "observe", "full"])
def test_a_wake_sees_an_owner_turns_tools_and_is_refused_at_the_call(tmp_path, level):
    """В31=B on the real arrays: the wake's request carries Main's schemas and selection; its level binds at dispatch."""
    main = _snapshots(tmp_path, {"main"})["main"]
    name = f"consciousness_{level}"
    with catalog(name, _contexts(tmp_path)[name], tmp_path) as wake_registry:
        wake = snapshot(wake_registry)
        for key in ("canonical", "omissions", "initial", "passed", "available_tools", "get_schema_by_name_absent",
                    "get_schema_by_name_differs", "policy_hidden_reason", "shapes", "child_described"):
            assert wake[key] == main[key], key
        withheld = set(ca.disabled_tools_for(level))

        def callable_names(record):
            return {row if isinstance(row, str) else row["name"] for row in record["callable_rows"]}

        assert callable_names(wake) == callable_names(main) - withheld
        if level == "full":
            assert not withheld
            return
        assert "toggle_evolution" in withheld and "toggle_evolution" in wake["available_tools"]
        refusal = wake_registry.execute("toggle_evolution", {"enabled": True, "objective": "x"})
        assert "RESOURCE_CONSTRAINT_BLOCKED" in refusal and "toggle_evolution" in refusal


def test_the_record_holds_decisions_not_bytes():
    """Names, order, reasons and parameter shapes: no schema prose and no byte digests, so a description edit
    does not move it."""
    text = FIXTURE.read_text(encoding="utf-8")
    assert FIXTURE.stat().st_size < 200_000
    assert "\"description\"" not in text and "schema_hashes" not in text and "\"bytes\"" not in text



def test_a_description_edit_does_not_move_the_record(tmp_path, monkeypatch):
    """The record pins decisions: rewording every tool's description, as most tool changes do, leaves it as it was,
    child-profile substitutions included, and so does new prose from an MCP server or an extension."""
    from dataclasses import replace

    from ouroboros.tools.tool_catalog import ToolCatalog

    names = {"main", "readonly_helper", "acting_helper"}
    before = _against_main(_snapshots(tmp_path / "before", names))
    real = ToolRegistry._load_modules

    def reworded(self):
        loaded = real(self)
        return ToolCatalog((loaded.origin_for(name), replace(entry, schema={
            **entry.schema, "description": f"{entry.schema.get('description', '')} Reworded."}))
            for name, entry in loaded.entries.items())

    monkeypatch.setattr(ToolRegistry, "_load_modules", reworded)
    for name, prose in list(DYNAMIC_PROSE.items()):
        monkeypatch.setitem(DYNAMIC_PROSE, name, f"{prose} Reworded.")
    assert _against_main(_snapshots(tmp_path / "after", names)) == before


def test_a_descriptor_another_test_left_behind_stays_out_of_the_record(tmp_path):
    """The record reads only its own catalog: a live-looking descriptor an earlier test in this worker left in the
    process-global extension registry (here a second tool of the record's own skill) neither enters it nor is lost."""
    from ouroboros import extension_loader
    from ouroboros import extension_registry_state as state

    clean = _snapshots(tmp_path / "clean", {"main"})["main"]
    stray = extension_loader.extension_surface_name(EXT_SKILL, "leaked")
    with state._lock:
        state._tools[stray] = {"name": stray, "skill": EXT_SKILL, "description": "left by another test",
                               "schema": {"type": "object", "properties": {}}}
    try:
        assert _snapshots(tmp_path / "dirty", {"main"})["main"] == clean
        assert stray in state._tools, "the other test's state is put back"
    finally:
        with state._lock:
            state._tools.pop(stray, None)
