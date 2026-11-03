"""Static selection suite for the GitHub tools: owner-style request → expected tool, method and key arguments.

No model call. Every expected call in ``tests/fixtures/github_tool_selection.jsonl`` must validate
against the registered schema (unknown arguments refused), satisfy the handler's own argument rules
without launching gh, and together the cases must cover every method of both tools and pr_merge.
"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from ouroboros.tools import github, github_checks, github_read, github_write, github_pr_write, github_workflows

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "github_tool_selection.jsonl"
CASES = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
SCHEMAS = {entry.name: entry.schema["parameters"] for entry in github.get_tools()}


def _validator(tool: str) -> Draft202012Validator:
    schema = {**SCHEMAS[tool], "additionalProperties": False}
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_the_suite_has_distinct_cases_over_every_method():
    assert len(CASES) >= 40 and len({c["id"] for c in CASES}) == len(CASES)
    assert {c["tool"] for c in CASES} == {"github_read", "github_write", "pr_merge"}
    read_methods = {c["args"]["method"] for c in CASES if c["tool"] == "github_read"}
    write_methods = {c["args"]["method"] for c in CASES if c["tool"] == "github_write"}
    assert read_methods == set(github_read.METHODS) and write_methods == set(github_write.METHODS)
    kinds = {c["args"].get("kind") for c in CASES if c["tool"] == "github_read" and c["args"]["method"] == "list"}
    assert kinds >= {"issue", "pr", "comment", "review_comment", "review", "discussion", "notification", "release", "label", "artifact"}
    assert all(c["request"].strip() for c in CASES)


@pytest.mark.parametrize("case", CASES, ids=[f"{c['id']}-{c['tool']}-{c['args'].get('method', 'merge')}" for c in CASES])
def test_every_expected_call_validates_against_the_registered_schema(case):
    errors = sorted(_validator(case["tool"]).iter_errors(case["args"]), key=lambda e: list(e.path))
    assert not errors, [f"{list(e.path)}: {e.message}" for e in errors]
    assert set(case["args"]) <= set(SCHEMAS[case["tool"]]["properties"])
    for name in SCHEMAS[case["tool"]]["required"]:
        assert name in case["args"], f"{case['id']}: {name} is required"


@pytest.mark.parametrize("case", [c for c in CASES if c["tool"] != "pr_merge"], ids=[str(c["id"]) for c in CASES if c["tool"] != "pr_merge"])
def test_every_expected_call_passes_the_handlers_argument_rules_before_any_gh_launch(case, monkeypatch):
    """The handler's own validation admits the call: the first thing it does is launch gh (which this fake records
    and refuses), never a typed argument refusal."""
    launched = []

    def no_gh(args, ctx, *a, **kw):
        launched.append(list(args))
        return github.GhResult(False, "⚠️ GH_ERROR: fixture transport", 1, 503, "exit")

    for module in (github, github_read, github_checks, github_pr_write, github_workflows):  # every name the transport is bound to; no real gh, no network
        monkeypatch.setattr(module, "_gh_run", no_gh)
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "ghp_fixture")
    ctx = SimpleNamespace(repo_dir=pathlib.Path("."), drive_root=pathlib.Path("."), task_id="t", _active_builtin_tool_result=None,
                          task_metadata={}, workspace_root=None, project_id="")
    run = github_read.run_read if case["tool"] == "github_read" else github_write.run_write
    text = run(ctx, **case["args"])
    assert launched, f"{case['id']}: the call was refused before gh: {text}"
    assert "TOOL_ARG_ERROR" not in text.splitlines()[0], text
