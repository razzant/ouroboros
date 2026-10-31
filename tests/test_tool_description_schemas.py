"""Representative calls retain the executable contracts of the edited descriptions.

Six of the seven remain; the seventh tool, task_acceptance_review, left with the review fold.
query_data, delegate_start, delegate_wait, wait_tasks (#1801, #502) and schedule_subagent (#502,
#1175) changed prose only: their parameter structures are pinned with the prose removed, and the
conditions a caller needs before a call stay in the tool text that every profile and the custom
codec deliver.
"""

import hashlib
import json
import pathlib

import pytest

from ouroboros.tools.registry import ToolRegistry

# Conditions the custom projection clipped, wholly or partly, from the schedule_subagent lead before
# #502 (full and acting kept 2310 of its 3392 characters, readonly 2552 and so lost fewer), in the
# wording that now reaches that consumer.
_SCHEDULE_TAIL = (
    "no child commits, grants itself tools or writes identity/scratchpad",
    "Cyber-effective children inherit selected review/skill/runtime tools within explicit task restrictions",
    "Nesting obeys configured depth/cap limits",
    "Children scheduled in one round run concurrently",
    "each sibling started before the first responds pays a full prefix write",
    "Native children can address you, siblings or any task in their tree",
    "need objective/constraints stating what is interim, whom to address and what ends participation",
    "delegate_start(subagent_id=id, continue_from=run_id, prompt=''): new run, "
    "same session if possible, else a new one with retained evidence",
    "Retrieve the handoff with get_task_result, wait_task or wait_tasks before relying on it",
)


@pytest.mark.parametrize("constraint", [
    {}, {"mode": "local_readonly_subagent"},
    {"mode": "acting_subagent", "surface": "self_worktree", "root_task_id": "projection"},
], ids=["full", "readonly", "acting"])
def test_schedule_readonly_choice_survives_custom_projection(tmp_path, constraint):
    from ouroboros.openai_chat_custom import project_function_tools_to_openai_custom

    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry._ctx.task_constraint = constraint
    schema = next(tool for tool in registry.schemas() if tool["function"]["name"] == "schedule_subagent")
    function = schema["function"]
    # Normal/pruned function schemas carry parameter prose; the custom codec
    # strips it and keeps the tool prose only as far as its description bound allows.
    assert "Select readonly for read-only session execution" in function["parameters"]["properties"]["access"]["description"]
    catalog = project_function_tools_to_openai_custom([schema])
    description = catalog.wire_tools()[0]["custom"]["description"]
    assert description.startswith(" ".join(function["description"].split()) + "\n")
    assert "access='readonly' selects read-only session execution" in description
    assert "evidence-only private checking copy" in description
    for condition in _SCHEDULE_TAIL:
        assert condition in description, condition
    assert catalog.schema_binding("schedule_subagent").schema() == function["parameters"]


@pytest.mark.serial
@pytest.mark.parametrize("name,arguments,invalid_field,invalid_value", [
    ("schedule_subagent", {"subagent_id": "actor-1", "objective": "Inspect interfaces",
                          "expected_output": "Findings", "role": "", "write_surface": "read_only"},
     "write_surface", "invented_surface"),
    ("plan_task", {"goal": "Repair interfaces", "plan": "Inspect and verify", "affected_paths": [],
                   "done_when": [{"id": "c1", "claim": "Interfaces remain compatible", "support": "run the suite"}],
                   "ask": "council", "message": "First look."},
     "ask", "everyone"),
    ("delegate_start", {"prompt": "New evidence", "subagent_id": "actor-1",
                        "continue_from": "settled-run", "continue_carrier": "packet"},
     "continue_carrier", "restart"),
    ("verify_and_record", {"contract_kind": "explicit_command", "check": ["python", "check.py"],
                           "expected_match": "bytes_equal", "artifact_paths": ["actual", "expected"]},
     "expected_match", "approximate"),
    ("start_task", {"objective": "Inspect supplied project", "title": "Supplied project inspection",
                              "workspace": "none", "context_requires_self_body_docs": False},
     "context_requires_self_body_docs", "false"),
    ("preflight_review", {"commit_message": "Repair interfaces", "deterministic_only": True,
                          "source": "index"},
     "source", "head"),
    ("query_data", {"op": "sql", "db": "money", "query": "SELECT 1", "max_rows": 5}, "db", "ledger"),
    ("query_data", {"op": "jsonl", "kind": "events_log", "tail": 20}, "max_rows", 0),
    ("delegate_wait", {"run_id": "run-1", "checkpoint_after_sec": 60, "checkpoint_reason": "Inspect the diff"},
     "checkpoint_after_sec", "soon"),
    ("wait_tasks", {"task_ids": ["child-1"], "mode": "any_terminal", "timeout_sec": 60}, "mode", "first_terminal"),
])
def test_edited_description_schemas_keep_representative_call_shapes(
        tmp_path, name, arguments, invalid_field, invalid_value):
    """Exercise the exported parameter contracts, including continuation aliases."""
    from jsonschema import Draft7Validator

    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    parameters = registry.get_schema_by_name(name)["function"]["parameters"]
    Draft7Validator.check_schema(parameters)
    validator = Draft7Validator(parameters)
    assert not list(validator.iter_errors(arguments))
    assert list(validator.iter_errors({**arguments, invalid_field: invalid_value}))


_PROFILES = {
    "full": {}, "readonly": {"mode": "local_readonly_subagent"},
    "acting": {"mode": "acting_subagent", "surface": "self_worktree", "root_task_id": "projection"},
}
_SQL_ONLY = "db, query, max_rows, snapshot and timeout_sec are op=sql only"
# Parameters with every string description removed (required, defaults, enums and bounds kept),
# as emitted before these prose-only edits. A deliberate parameter change updates its pin.
# delegate_start: + attachment_paths (string array) and attachment_import (reuse|new), C505;
# without those two the structure is the earlier c4263da7232644b1 / 1aa885fc9a0c37ff.
# query_data: + kind "update_material" (the offered update range); before it f790f964764d75b7.
_STRUCTURE = {
    "query_data": {"full": "1408c97acc31b0f7", "readonly": "1408c97acc31b0f7", "acting": "1408c97acc31b0f7"},
    "delegate_start": {"full": "5e25716838978044", "readonly": "1326e91ef42b3cea", "acting": "1326e91ef42b3cea"},
    "delegate_wait": {"full": "60ab2c0f421da812", "readonly": "60ab2c0f421da812", "acting": "60ab2c0f421da812"},
    "wait_tasks": {"full": "c806a86311a0a965", "readonly": "c806a86311a0a965", "acting": "c806a86311a0a965"},
    # readonly prunes write_surface, write_root, protected_paths_grant and external_tool_grants.
    "schedule_subagent": {"full": "e17e74235ea913fd", "readonly": "ec6b3c08c33ac3fa", "acting": "e17e74235ea913fd"},
}


def _without_prose(value):
    if isinstance(value, dict):
        return {key: _without_prose(item) for key, item in value.items()
                if not (key == "description" and isinstance(item, str))}
    return [_without_prose(item) for item in value] if isinstance(value, list) else value


@pytest.mark.parametrize("profile", sorted(_PROFILES))
def test_prose_edits_keep_structure_and_deliver_conditions_in_every_profile(tmp_path, profile):
    from ouroboros.openai_chat_custom import project_function_tools_to_openai_custom

    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry._ctx.task_constraint = _PROFILES[profile]
    schemas = {tool["function"]["name"]: tool for tool in registry.schemas()}
    for name in _STRUCTURE:
        function = schemas[name]["function"]
        structure = json.dumps(_without_prose(function["parameters"]), sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256(structure.encode()).hexdigest()[:16] == _STRUCTURE[name][profile], name
        catalog = project_function_tools_to_openai_custom([schemas[name]])
        custom = catalog.wire_tools()[0]["custom"]["description"]
        # The whole tool text survives the codec's bound; only parameter prose and defaults are dropped.
        assert custom.startswith(" ".join(function["description"].split()) + "\n"), name
        assert catalog.schema_binding(name).schema() == function["parameters"]
        if name == "query_data":
            assert _SQL_ONLY in function["description"] and _SQL_ONLY in custom
    query = schemas["query_data"]["function"]
    assert query["parameters"]["required"] == ["op"]
    assert {key: value.get("default") for key, value in query["parameters"]["properties"].items()
            if key in {"db", "query", "max_rows", "snapshot", "timeout_sec", "tail"}} == {
        "db": "current", "query": None, "max_rows": 100, "snapshot": False, "timeout_sec": 1, "tail": 100}
    for field in ("db", "max_rows"):
        assert query["parameters"]["properties"][field]["description"].startswith("op=sql only")
    assert "accepts only the default current, as if omitted" in query["parameters"]["properties"]["db"]["description"]
    assert "accepts only the default 100" in query["parameters"]["properties"]["max_rows"]["description"]


_EVERY = frozenset(_PROFILES)
# Clauses #502/#1175 removed from the schedule_subagent lead that another lead of a COMPLETE Low/Max
# catalog carries, whole also in custom: (tool, its words, profiles where the clause can apply). A
# Nano selection or a route's schema cap can omit those tools; see the Nano test below.
_SCHEDULE_REMOVALS_CARRIED_BY = (
    ("wait_tasks", "With mode=any_terminal it returns as soon as the FIRST child finishes", _EVERY),
    ("forward_to_worker", "To wait for a reply without spending model rounds, call await_messages", _EVERY),
    ("delegate_answer", "Answer a delegated run's pending interactive question", _EVERY),
    ("chronicle_write", "publishes pages and parts only as drafts signed in its own name", _EVERY),
    ("delegate_start", "A TOP-LEVEL task may select one existing user-managed skill via root='skill_payload', "
                       "bucket and skill_name", _EVERY),
    # Only a caller that can request write_surface can schedule these children.
    ("integrate_subagent_patch", "Genesis is a standalone directory, not a repo patch", {"full", "acting"}),
    ("integrate_subagent_patch", "This never commits", {"full", "acting"}),
    ("integrate_subagent_patch", "For best-of-N pick a child and integrate it", {"full", "acting"}),
)


@pytest.mark.parametrize("profile", sorted(_PROFILES))
def test_schedule_lead_removals_stay_in_co_present_custom_leads(tmp_path, profile):
    from ouroboros.openai_chat_custom import project_function_tools_to_openai_custom

    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry._ctx.task_constraint = _PROFILES[profile]
    catalog = project_function_tools_to_openai_custom(registry.schemas())
    leads = {tool["custom"]["name"]: tool["custom"]["description"].rpartition("\nInput must be one JSON object")[0]
             for tool in catalog.wire_tools()}
    assert "schedule_subagent" in leads
    for name, words, profiles in _SCHEDULE_REMOVALS_CARRIED_BY:
        if profile in profiles:
            assert words in leads.get(name, ""), (name, words)


# SYSTEM is delivered whole in every mode, Nano included: its own words for the removed clauses that a
# Nano selection or a capped catalog may not show through any other lead.
_SCHEDULE_REMOVALS_IN_SYSTEM = (
    "Several builders on ONE new deliverable each get `write_surface=external_workspace` with `write_root` omitted",
    "one cooperative tree I integrate as sole committer",
    "`genesis` is a standalone per-child repo",
    "an installed skill payload changes only through MY OWN lane — a top-level `delegate_start` with "
    "`root=\"skill_payload\"`",
    "a child cannot open the payload lane",
    "chronicle page and part drafts in their own name",
)


@pytest.mark.parametrize("profile", sorted(_PROFILES))
def test_nano_selected_schedule_lead_stands_without_other_delegation_leads(tmp_path, profile):
    """A Nano actor may load schedule_subagent alone, so no other delegation lead is in its request:
    the lead, SYSTEM and the Connections names must then carry what a caller needs before the call."""
    from ouroboros.openai_chat_custom import project_function_tools_to_openai_custom
    from ouroboros.tool_connections import connection_views
    from ouroboros.tool_policy import NANO_SCHEMA_META_NAMES, initial_tool_schemas

    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry._ctx.task_constraint = _PROFILES[profile]
    schemas = initial_tool_schemas(registry, context_mode="nano", schema_names=["schedule_subagent"])
    names = {schema["function"]["name"] for schema in schemas}
    assert "schedule_subagent" in names and names <= {"schedule_subagent", *NANO_SCHEMA_META_NAMES}, names
    function = next(schema["function"] for schema in schemas if schema["function"]["name"] == "schedule_subagent")
    custom = next(tool["custom"]["description"] for tool in project_function_tools_to_openai_custom(schemas).wire_tools()
                  if tool["custom"]["name"] == "schedule_subagent")
    assert custom.startswith(" ".join(function["description"].split()) + "\n")
    for condition in (*_SCHEDULE_TAIL, "write_root, else the selected/inherited workspace_root, else one new Git tree",
                      "Runtime data, including installed skill payloads, is never a write_surface",
                      "access='readonly' selects read-only session execution", "evidence-only private checking copy"):
        assert condition in custom, condition
    system = " ".join((pathlib.Path(__file__).resolve().parents[1] / "prompts" / "SYSTEM.md")
                      .read_text(encoding="utf-8").split())
    for words in _SCHEDULE_REMOVALS_IN_SYSTEM:
        assert words in system, words
    # The other carriers are absent here, yet named in the Nano map and loadable on demand.
    _, nano_map = connection_views(registry, helper=profile != "full")
    for name, _words, profiles in _SCHEDULE_REMOVALS_CARRIED_BY:
        if profile in profiles:
            assert name not in names and name in nano_map, name
