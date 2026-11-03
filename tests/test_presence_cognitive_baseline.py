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

from ouroboros.presence_authority import PresenceAuthorityError, apply_presence_argument_bindings, presence_ceiling_allows_tool, presence_ceiling_from_payload, presence_ceiling_payload
from tests.frozen_presence_authority import build_presence_capability_ceiling
from tests.frozen_presence_capabilities import (
    PresenceArgumentBinding,
    PresenceProfileResolution,
    PresenceSelection,
    PresenceToolTarget,
)
from tests.frozen_presence_runtime import ResolvedPresenceRuntime
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


_OWN_WORK = {"get_task_result", "recent_tasks", "tell_task"}


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


def test_place_admission_has_no_cognitive_allowlist(tmp_path):
    from tests.test_presence_admission import _admit, _binding, _install_behavior
    _install_behavior(tmp_path)
    admitted = _admit(tmp_path, _binding(tmp_path))
    assert admitted.capability_ceiling is None
    assert admitted.place.context_topics == ("community-guidelines",)
