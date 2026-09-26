"""An admitted presence conversation keeps its own memory, and nothing more.

The ceiling is compiled from the reviewed profile plus one constant set: the
tools by which this mind reads and revises what it knows
(`tool_capabilities.COGNITIVE_MEMORY_TOOL_NAMES`). The baseline adds no
authority to act outside the mind, it is covered by the ceiling digest, and it
never displaces a selection the profile authored with argument bindings.
"""

from __future__ import annotations

import json

import pytest

from ouroboros.presence_authority import (
    PresenceAuthorityError,
    apply_presence_argument_bindings,
    build_presence_capability_ceiling,
    presence_ceiling_allows_tool,
    presence_ceiling_from_payload,
    presence_ceiling_payload,
)
from ouroboros.presence_capabilities import (
    PresenceArgumentBinding,
    PresenceProfileResolution,
    PresenceSelection,
    PresenceToolTarget,
)
from ouroboros.presence_runtime import ResolvedPresenceRuntime
from ouroboros.tool_capabilities import COGNITIVE_MEMORY_TOOL_NAMES
from ouroboros.tools.registry import ToolContext


def _resolution(*selections):
    return PresenceProfileResolution(
        active=tuple(selections),
        missing_required=(),
        missing_optional=(),
        orphaned=(),
        runtime=ResolvedPresenceRuntime("main", 10, 10, False),
        profile_fingerprint="a" * 64,
        selection_fingerprint="b" * 64,
        required_selections_present=True,
    )


def _ceiling(*selections):
    return build_presence_capability_ceiling(
        skill_name="community-helper",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=_resolution(*selections),
    )


_OWN_WORK = {"get_task_result", "recent_tasks", "steer_task"}


def test_profile_without_tool_selections_still_carries_its_own_memory():
    ceiling = _ceiling()

    assert [grant.name for grant in ceiling.tool_grants] == sorted(COGNITIVE_MEMORY_TOOL_NAMES | _OWN_WORK)
    assert all(grant.bindings == () for grant in ceiling.tool_grants if grant.name in COGNITIVE_MEMORY_TOOL_NAMES)
    for name in COGNITIVE_MEMORY_TOOL_NAMES:
        assert presence_ceiling_allows_tool(ceiling, name)


def test_the_baseline_grants_no_authority_outside_the_mind():
    ceiling = _ceiling()

    for name in ("write_file", "edit_text", "run_command", "send_user_message", "skill_exec"):
        assert not presence_ceiling_allows_tool(ceiling, name)


def test_a_selected_baseline_tool_keeps_the_profile_authored_bindings():
    # Deduplication is by name and the profile wins: its bindings are exact
    # host facts, and losing them would hand the argument back to the model.
    selection = PresenceSelection(
        "1" * 64,
        PresenceToolTarget("builtin", "knowledge_write"),
        (PresenceArgumentBinding(("scope",), "static", static_value="global"),),
    )
    ceiling = _ceiling(selection)

    grant = next(item for item in ceiling.tool_grants if item.name == "knowledge_write")
    assert [(item.argument_path, item.static_value) for item in grant.bindings] == [
        (("scope",), "global"),
    ]
    assert [item.name for item in ceiling.tool_grants] == sorted(COGNITIVE_MEMORY_TOOL_NAMES | _OWN_WORK)

    ctx = ToolContext(
        repo_dir=None,
        drive_root=None,
        task_contract={"capability_ceiling": presence_ceiling_payload(ceiling)},
    )
    bound = apply_presence_argument_bindings(
        ctx, "knowledge_write", {"topic": "note", "scope": "project:sneaky"},
    )
    assert bound == {"topic": "note", "scope": "global"}
    # A baseline name the profile did not select arrives unbound: the model
    # supplies its own arguments, exactly as in any other room.
    assert apply_presence_argument_bindings(
        ctx, "knowledge_read", {"topic": "note"},
    ) == {"topic": "note"}


def test_the_digest_covers_the_baseline(tmp_path):
    ceiling = _ceiling()
    payload = presence_ceiling_payload(ceiling)

    assert presence_ceiling_from_payload(payload) == ceiling

    stripped = json.loads(json.dumps(payload))
    stripped["tools"] = [tool for tool in stripped["tools"] if tool["name"] != "update_identity"]
    with pytest.raises(PresenceAuthorityError) as caught:
        presence_ceiling_from_payload(stripped)
    assert caught.value.code == "presence_authority_digest_mismatch"


def test_admitted_external_turn_writes_global_knowledge_and_nothing_else(tmp_path):
    """The whole path, no model: admission → runner → registry → the note on disk.

    A real presence admission (reviewed skill, saved selection, bound room) runs
    one bounded turn whose agent writes what it learned about a person. The note
    lands on the canonical global shelf, the six memory tools are offered, and
    every tool that would act outside this mind is absent and refused.
    """
    from tests.test_presence_admission import _admit, _binding, _install_behavior, _select_history
    from tests.test_presence_runner import _event

    from ouroboros.presence_runner import PresenceTurnGate, run_presence_turn
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    skill_dir = _install_behavior(data)
    _select_history(data, skill_dir)
    admission = _admit(data, _binding(data))
    assert [grant.name for grant in admission.capability_ceiling.tool_grants] == sorted(
        COGNITIVE_MEMORY_TOOL_NAMES | _OWN_WORK
    )
    seen: dict[str, object] = {}

    class Agent:
        def __init__(self, repo_dir, drive_root, **_kwargs):
            self.repo_dir = repo_dir
            self.drive_root = drive_root

        def handle_task(self, task):
            ctx = ToolContext(
                repo_dir=self.repo_dir,
                drive_root=self.drive_root,
                task_id=str(task.get("id") or "presence-turn"),
                task_contract=task.get("task_contract") or {},
                task_metadata=task.get("metadata") or {},
                current_chat_id=task.get("chat_id"),
            )
            registry = ToolRegistry(repo_dir=self.repo_dir, drive_root=self.drive_root)
            registry.set_context(ctx)
            seen["schemas"] = {schema["function"]["name"] for schema in registry.schemas()}
            seen["write"] = registry.execute(
                "knowledge_write",
                {
                    "topic": "people/alex",
                    "scope": "global",
                    "content": "Alex asked for short answers today; I read it as a preference to test.",
                },
            )
            seen["scratchpad"] = registry.execute(
                "update_scratchpad", {"content": "Alex's room: brevity today, worth testing next time."})
            seen["identity"] = registry.execute(
                "update_identity", {"content": "I am Ouroboros. " + "I keep one memory in every room. " * 4})
            seen["refused"] = {
                name: registry.execute(name, args)
                for name, args in (
                    ("write_file", {"root": "runtime_data", "path": "memory/identity.md", "content": "no"}),
                    ("send_user_message", {"text": "no"}),
                    ("run_command", {"cmd": ["true"]}),
                )
            }
            from ouroboros.task_results import write_task_result

            write_task_result(data, task["id"], "completed", metadata=task["metadata"], result="Noted.")
            return [{"type": "presence_result", "outcome": "message", "text": "Noted.", "work_ref": ""}]

    result = run_presence_turn(
        admission=admission,
        event=_event(),
        repo_dir=repo,
        drive_root=data,
        agent_factory=lambda repo_dir, drive_root, **kwargs: Agent(repo_dir, drive_root, **kwargs),
        gate=PresenceTurnGate(2),
    )

    assert result.outcome == "message"
    note = (data / "memory" / "knowledge" / "people" / "alex.md").read_text(encoding="utf-8")
    assert "short answers" in note
    assert "PRESENCE_CAPABILITY_BLOCKED" not in str(seen["write"])
    assert str(seen["scratchpad"]).startswith("OK")
    assert str(seen["identity"]).startswith("OK")
    assert "brevity today" in (data / "memory" / "scratchpad_blocks.json").read_text(encoding="utf-8")
    assert "one memory in every room" in (data / "memory" / "identity.md").read_text(encoding="utf-8")
    assert COGNITIVE_MEMORY_TOOL_NAMES <= seen["schemas"]
    for name, refusal in seen["refused"].items():
        assert name not in seen["schemas"]
        assert "PRESENCE_CAPABILITY_BLOCKED" in refusal
