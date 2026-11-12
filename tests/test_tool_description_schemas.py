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
from tests._cases import case_folders


@pytest.mark.parametrize("dialect", ["openai_function", "openai_custom", "anthropic", "gigachat", "external"])
def test_finish_gist_rule_reaches_every_real_tool_projection(tmp_path, dialect):
    from ouroboros.llm import LLMClient
    from ouroboros.llm_claudexor import _request
    from ouroboros.openai_chat_custom import project_function_tools_to_openai_custom
    from ouroboros.tools.control import GIST_RULE

    expected = (
        "On a root task or self-started run, `summary` is my gist of the run for my human and my later self, "
        "in my human's language; in a tight window it stands for this work's replies. "
        "Its first sentence says what the run did, found or decided: a clean finish shows it on the card, "
        "room row and portal, and that sentence represents this work in my other conversations. "
        "Further sentences, when the work needs them, keep what a later reader needs without the rows: "
        "numbers, paths, decisions, promises, what remains and whose step it is."
    )
    assert GIST_RULE == expected  # shrinking the constant cannot weaken this delivery check
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    schema = registry.get_schema_by_name("finish_task")
    assert schema["function"]["description"].startswith(expected + "\n\n")
    field = schema["function"]["parameters"]["properties"]["summary"]
    assert field["description"].startswith("The gist, as this tool's description says; a held cause takes precedence.")
    if dialect == "openai_function":
        projected = LLMClient._sanitize_chat_completion_tools([schema])[0]["function"]
    elif dialect == "openai_custom":
        projected = project_function_tools_to_openai_custom([schema]).wire_tools()[0]["custom"]
        assert field["description"] not in projected["description"]  # parameter prose really is stripped
    elif dialect == "anthropic":
        projected = LLMClient._build_anthropic_tools([schema])[0]
    elif dialect == "gigachat":
        projected = LLMClient._gigachat_functions([schema])[0]
    else:
        projected = _request({"source": "codex", "resolved_model": "test-model"}, [], [schema],
                             {"model_account_override": ""})["tools"][0]["function"]
    assert projected["description"].startswith(expected)

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
    ("query_data", {"op": "jsonl", "kind": "events_log", "tail": 20}, "max_rows", 0),
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
# delegate_start and schedule_subagent: + input_files (the same typed {path, root} items; schedule_subagent's
# lead pays its custom-tool budget), #2593; before them
# delegate_start 5e25716838978044 / 1326e91ef42b3cea, schedule_subagent e17e74235ea913fd / ec6b3c08c33ac3fa.
# query_data: + kind "update_material" (the offered update range); before it f790f964764d75b7.
# query_data: - kind "chat_log" (retired at dialogue-store cutover), #2998: 3b64a3753e957063.
# + plane (canonical: a fork explicitly reads the installation's logs), #1701.
# wait_tasks: + view, sleep_mode, wake_after_sec, wake_at and no timeout_sec default (an omitted
# window stays distinguishable, PR1696 native task wait); before it c806a86311a0a965.
# schedule_subagent: + memory_view (the fields a parent may widen a child's view by), #2924; its lead pays the
# custom-tool budget again; before it 3d5b8e7b8e317503 / 2e04edea50ceb1b6.
_STRUCTURE = {
    "query_data": {"full": "d7100565e686c951", "readonly": "d7100565e686c951", "acting": "d7100565e686c951"},
    "delegate_start": {"full": "a2f6fdb7a8bf6822", "readonly": "e7dcf92431326bc5", "acting": "e7dcf92431326bc5"},
    "delegate_wait": {"full": "60ab2c0f421da812", "readonly": "60ab2c0f421da812", "acting": "60ab2c0f421da812"},
    "wait_tasks": {"full": "50667317e4bef21f", "readonly": "50667317e4bef21f", "acting": "50667317e4bef21f"},
    # readonly prunes write_surface, write_root, protected_paths_grant and external_tool_grants.
    "schedule_subagent": {"full": "f04afa9698352e6a", "readonly": "69f811a4b8ca2ba2", "acting": "f04afa9698352e6a"},
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
    ("integrate_subagent_patch", "native external_workspace verifies files already in the shared tree WITHOUT "
                                 "reapplying", {"full", "acting"}),  # #2593 paid input_files with these words
)


def test_schedule_lead_removals_stay_in_co_present_custom_leads(subtests, tmp_path):
    case_paths = case_folders(tmp_path)
    for profile in sorted(_PROFILES):
        tmp_path = next(case_paths)
        with subtests.test(profile=profile):
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
    # #2593 paid schedule_subagent's typed input_files with these three lead clauses.
    "`start_task` begins only separate new work",
    "without candidate write authority its changes remain evidence and create no apply duty",
    "I verify the files and its recorded verdict instead of re-applying its patch",
)


def test_nano_selected_schedule_lead_stands_without_other_delegation_leads(subtests, tmp_path):
    """A Nano actor may load schedule_subagent alone, so no other delegation lead is in its request:
    the lead, SYSTEM and the Connections names must then carry what a caller needs before the call."""
    case_paths = case_folders(tmp_path)
    for profile in sorted(_PROFILES):
        tmp_path = next(case_paths)
        with subtests.test(profile=profile):
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
                              "Runtime data (installed skill payloads too) is never a write_surface",
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
