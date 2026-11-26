"""Unified Body Map residency and explicit whole-handbook opt-in (BIBLE P1).

Under BIBLE P1, every mode and focus carries the resident Body Map + Reference Atlas and
draws deep engineering manuals on demand; ``DEVELOPMENT.md`` is never preloaded by mode,
workspace binding, or task kind. Only a task's explicit ``context_requires_development=true``
adds the whole development handbook in every mode.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros.context_input_selection import development_required

KEY = "context_requires_development"


def test_only_explicit_contract_flag_requires_whole_development_handbook():
    assert development_required({"type": "task"}) is False
    assert development_required({"type": "evolution"}) is False
    assert development_required({"type": "task", KEY: True}) is False
    assert development_required({"type": "task", "task_contract": {KEY: False}}) is False
    assert development_required({"type": "task", "task_contract": {"context_requires_self_body_docs": True}}) is False
    assert development_required({"type": "task", "task_contract": {KEY: True}}) is True


def test_unified_body_map_has_no_ordinary_development_block_until_explicitly_required():
    from ouroboros.context_layout import reference_doc_sections

    development = "# DEVELOPMENT.md\n\n## Review\n\nRead every source.\n\n## Naming\n\nNames say what they are.\n"
    env = SimpleNamespace(repo_dir=None)
    ordinary = reference_doc_sections(env, context_mode="max", include_development=False,
                                      architecture_text="# ARCH\n\nbody", development_text=development)
    required = reference_doc_sections(env, context_mode="max", include_development=True, required_book=True,
                                      architecture_text="# ARCH\n\nbody", development_text=development)
    assert required[0] == ordinary[0]  # the common sections stay shared across tasks (one cached prefix)
    assert ordinary[1] == ""
    assert required[1] == "## DEVELOPMENT.md\n\n" + development


@pytest.mark.parametrize("flag", [True, False, None])
def test_start_task_carries_an_explicit_answer_to_the_admitted_task_and_absence_stays_absent(tmp_path, monkeypatch, flag):
    from tests.test_root_effort_ingress import _promoted

    task, _row, _workers = _promoted(tmp_path, monkeypatch, **({} if flag is None else {"context_requires_development": flag}))
    assert task.get("context_requires_development", "absent") == ("absent" if flag is None else flag)


# --- What the caller reads ------------------------------------------------------------------------------------------

def test_the_public_schema_says_what_true_costs_and_a_subject_caller_omits_it():
    from ouroboros.openai_chat_custom import project_function_tools_to_openai_custom
    from ouroboros.tools.control import get_tools

    schema = next(tool for tool in get_tools() if tool.name == "start_task").schema
    description = schema["parameters"]["properties"][KEY]["description"]
    assert "every mode" in description and "Body Map" in description
    # start_task reaches the custom codec without parameter descriptions: the lead it sends carries the conditions, once.
    wire = project_function_tools_to_openai_custom([{"type": "function", "function": schema}]).wire_tools()[0]
    lead = (wire.get("custom") or wire)["description"].split("\n")[0]
    clause = ("Set context_requires_development=true only when the task itself needs the whole development handbook in "
              "every mode, never for its subject; a route too small for it refuses the task.")
    assert clause in lead and lead.count("context_requires_development") == 1
    assert "default" not in schema["parameters"]["properties"][KEY] and KEY not in schema["parameters"]["required"]
    self_body = schema["parameters"]["properties"]["context_requires_self_body_docs"]["description"]
    assert "resident Body Map + Reference Atlas" in self_body
    assert "the design bible of docs/DESIGN.md from any folder" in self_body  # item 97, RV1 (Carbon 1.2)


def test_no_public_door_gained_the_requirement():
    from ouroboros.gateway import tasks as api
    from ouroboros.tools import control_subagent_spec

    assert "task_contract" in api._RESERVED_METADATA_KEYS
    assert KEY not in control_subagent_spec.schedule_subagent_properties()
