"""#1725: fields whose words reach the owner say so; their bytes and handlers are unchanged.

The stop rationale changed only its description (the claim field left with the
task-acceptance tool). These tests pin, through the real handlers, that the answer
the author submits is the exact text a finish stages (no rewrite, no spacing repair),
and that the rationale's description tells the author who reads it. Model promotes
and routes to existing projects must name the root they create (owner answer B,
13 Oct), refused before launch effects, while the owner, API and already-queued
entrances keep the lexical first-line name.
"""

import json
from types import SimpleNamespace

import pytest

# Samples in the shapes the issue measured: markdown, a list, a run-together
# sentence exactly as an author wrote it, non-ASCII. None is normalized.
CLAIMS = [
    "## Done\n\nThe export now handles empty input.\n\n- tests: 14 passed\n- docs updated",
    "Fixed export.Ran tests;all green.Docs updated",
    "Готово: экспорт исправлен, тесты проходят.  Two  spaces stay.",
]


def _root(tmp_path):
    return SimpleNamespace(drive_root=str(tmp_path), task_id="root", root_task_id="root",
                           task_metadata={"root_task_id": "root"}, task_contract={})


def _tool_schema(module, name):
    return next(entry.schema for entry in module.get_tools() if entry.name == name)


@pytest.mark.parametrize("claim", CLAIMS)
def test_an_explicit_finish_stages_the_claim_bytes_as_the_answer(tmp_path, claim):
    from ouroboros.tools.control_runtime import _finish_task

    ctx = _root(tmp_path)
    reply = json.loads(_finish_task(ctx, action="finish", answer=claim, rationale="Complete."))
    assert reply["status"] == "completion_requested"
    assert ctx._completion_request["answer"] == claim


def test_owner_and_api_admission_keep_the_first_line_name():
    """The shared fallback for entrances the owner or the API drives, and queued events, is unchanged."""
    from ouroboros.project_naming import admission_names

    objective = "Work order: audit chat rows.\nRead every producer first."
    assert admission_names({"title": "Chat row audit"}, objective) == ("Chat row audit", "Chat row audit")
    assert admission_names({}, objective) == ("", "Work order: audit chat rows")
    # Lexical only: spacing the author wrote is neither repaired nor removed.
    assert admission_names({}, "Fixexport.Ran tests") == ("", "Fixexport.Ran tests")


def _no_effects(monkeypatch):
    """Every effect the promote could start after its argument checks fails the test if reached."""
    from ouroboros.tools import control_routing

    def reached(*_args, **_kwargs):
        raise AssertionError("an untitled model root must be refused before this effect")

    monkeypatch.setattr(control_routing, "uuid", SimpleNamespace(uuid4=reached))
    monkeypatch.setattr(control_routing, "_promotion_pool_disabled_from_snapshot", reached)
    monkeypatch.setattr(control_routing, "_emit_and_wait_for_routing", reached)


UNNAMED = [{}, {"title": ""}, {"title": "   \n\t"}, {"title": "**"}, {"title": '""'}]


@pytest.mark.parametrize("named", UNNAMED, ids=["omitted", "empty", "whitespace", "markup-only", "quotes-only"])
def test_a_model_promote_without_a_name_is_refused_before_any_effect(tmp_path, monkeypatch, named):
    from ouroboros.tools.control import _start_task

    _no_effects(monkeypatch)
    ctx = SimpleNamespace(pending_events=[], event_queue=None, current_chat_id=1, drive_root=tmp_path)
    out = _start_task(ctx, "Work order: audit chat rows.",  **named)
    assert out.startswith("⚠️ TOOL_ARG_ERROR (start_task): title is required"), out
    assert "Nothing was scheduled" in out
    assert ctx.pending_events == [] and list(tmp_path.iterdir()) == []


def test_the_registered_tool_requires_a_name_and_refuses_an_unnamed_call(tmp_path, monkeypatch):
    from jsonschema import Draft7Validator

    import ouroboros.safety as safety
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    parameters = registry.get_schema_by_name("start_task")["function"]["parameters"]
    assert parameters["required"] == ["objective", "title"]
    title = parameters["properties"]["title"]
    assert "default" not in title and "If omitted" not in title["description"]
    assert "owner sees on its card" in title["description"]
    args = {"objective": "Work order: audit chat rows."}
    assert list(Draft7Validator(parameters).iter_errors(args))

    _no_effects(monkeypatch)
    for unnamed in ({}, {"title": "  "}):
        result = registry.execute_result("start_task", {**args, **unnamed})
        assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR"), result
        assert "title is required" in result.text


def test_a_named_promote_is_admitted_under_its_name_and_a_queued_untitled_event_still_is(tmp_path, monkeypatch):
    """The refusal lives in the model-facing tool, not in admission: replay keeps working."""
    import supervisor.workers as workers
    from ouroboros.tools.control import _start_task

    monkeypatch.setattr("ouroboros.tools.control_events._wait_for_promotion_admission",
                        lambda *_a, **_k: {"status": "scheduled", "effective_project_id": ""})
    ctx = SimpleNamespace(pending_events=[], event_queue=None, current_chat_id=1, drive_root=tmp_path)
    objective = "Work order: audit chat rows.\nRead every producer first."
    assert _start_task(ctx, objective, title="Chat row audit", workspace="none").startswith("OK: task")
    [event] = ctx.pending_events
    assert event["title"] == "Chat row audit" and event["objective"] == objective

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    admitted = []
    supervisor_ctx = SimpleNamespace(enqueue_task=admitted.append, persist_queue_snapshot=lambda **_k: True,
                                     load_state=lambda: {"owner_chat_id": 1})
    legacy = {key: value for key, value in event.items() if key != "title"}
    legacy.update(task_id="legacy" + event["task_id"][6:], routing_token="legacy-" + event["routing_token"])
    for queued in (event, legacy):
        assert workers.promote_chat_to_task(queued, supervisor_ctx)["status"] == "scheduled"
    named, untitled = admitted
    assert (named["title"], named["suggested_name"]) == ("Chat row audit", "Chat row audit")
    assert (untitled["title"], untitled["suggested_name"]) == ("", "Work order: audit chat rows")


def test_the_descriptions_name_the_reader_without_changing_the_shapes():
    from ouroboros.tools import control

    finish = _tool_schema(control, "finish_task")["parameters"]
    assert finish["required"] == ["action"]
    assert "owner reads it as the stop's cause on the task row" in finish["properties"]["rationale"]["description"]
