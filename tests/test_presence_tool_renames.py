"""A saved Presence spelling of a renamed builtin is its successor: a rename keeps saved choices
working and never makes the old name callable.

Both directions per carrier: a required selection saved under the old name admits and its ceiling
grants the successor; a frozen pre-update ceiling naming the old spelling still allows the successor;
an unknown old name tells the caller the rename; an unrelated unknown name gets no rename line; a
genuinely missing builtin is still refused.
"""
from __future__ import annotations

import pytest

from ouroboros.contracts.task_contract import TOOL_NAME_ALIASES
from tests.frozen_presence_capabilities import (
    PresenceSelection, PresenceState, PresenceToolTarget, presence_state_fingerprint, save_presence_state,
)
from tests.frozen_presence_profile import parse_presence_profile, presence_request_fingerprint
from ouroboros.skill_loader import load_skill
from tests.test_presence_admission import _admit, _binding, _install_behavior

PAIRS = sorted(TOOL_NAME_ALIASES.items())


def _select(drive_root, skill_dir, name):
    loaded = load_skill(skill_dir, drive_root)
    profile = parse_presence_profile(loaded.manifest, skill_dir)
    state = PresenceState((PresenceSelection(
        presence_request_fingerprint(profile.capability_requests[0]), PresenceToolTarget("builtin", name)),))
    save_presence_state(drive_root, loaded.name, state,
                        expected_state_fingerprint=presence_state_fingerprint(PresenceState()))


def test_legacy_selections_do_not_constrain_new_focus_admission(tmp_path):
    skill_dir = _install_behavior(tmp_path)
    _select(tmp_path, skill_dir, "does_not_exist")
    admission = _admit(tmp_path, _binding(tmp_path))
    assert admission.capability_ceiling is None


@pytest.mark.parametrize("old,new", PAIRS)
def test_a_frozen_ceiling_with_the_old_spelling_allows_the_successor_only(old, new):
    from ouroboros.presence_authority import PresenceToolGrant, presence_ceiling_allows_tool

    ceiling = type("Ceiling", (), {"tool_grants": (PresenceToolGrant(old),)})()
    assert presence_ceiling_allows_tool(ceiling, new)
    assert not presence_ceiling_allows_tool(ceiling, "write_file")  # quiet: nothing else widens


@pytest.mark.parametrize("old,new", PAIRS)
def test_an_unknown_old_name_names_its_successor(old, new):
    from ouroboros.tool_policy import name_miss_guidance

    lines = name_miss_guidance(old, "builtin", [{"name": new, "description": "x"}], discovery=True)
    assert lines[0] == f"{old!r} was renamed {new!r} (not called)."
    quiet = name_miss_guidance("no_such_tool", "builtin", [{"name": new, "description": "x"}], discovery=True)
    assert not any("renamed" in line for line in quiet)
    elsewhere = name_miss_guidance(old, "mcp_srv", [], discovery=True)  # an MCP tool may own that name
    assert not any("renamed" in line for line in elsewhere)


def _bound_ceiling_ctx(grant_name):
    from types import SimpleNamespace

    from ouroboros.presence_authority import PresenceCapabilityCeiling, PresenceToolGrant, presence_ceiling_payload
    from tests.frozen_presence_capabilities import PresenceArgumentBinding

    binding = PresenceArgumentBinding(("project_id",), "static", static_value="host-bound")
    ceiling = PresenceCapabilityCeiling(
        skill_name="behavior", skill_content_hash="a" * 64, profile_fingerprint="b" * 64,
        state_fingerprint="c" * 64, selection_fingerprint="d" * 64, model_slot="main", inline_max_rounds=3,
        tool_grants=(PresenceToolGrant(grant_name, (binding,)),), resource_grants=(), digest="0" * 64)
    return SimpleNamespace(task_contract={"capability_ceiling": presence_ceiling_payload(ceiling)}, task_metadata={})


@pytest.mark.parametrize("old,new", PAIRS)
def test_a_frozen_old_spelling_grant_binds_the_successors_arguments(old, new):
    """The host binding follows the grant to the successor; the model's value never wins."""
    from ouroboros.presence_authority import apply_presence_argument_bindings
    from ouroboros.tools.registry_core import _presence_bound_args, _presence_tool_allowed

    ctx = _bound_ceiling_ctx(old)
    assert _presence_tool_allowed(ctx, new) and not _presence_tool_allowed(ctx, old)
    assert apply_presence_argument_bindings(ctx, new, {"project_id": "model-chosen"})["project_id"] == "host-bound"
    bound, refusal = _presence_bound_args(ctx, new, {"project_id": "model-chosen"})
    assert refusal == "" and bound["project_id"] == "host-bound"
    unrelated = apply_presence_argument_bindings(ctx, "write_file", {"project_id": "model-chosen"})
    assert unrelated["project_id"] == "model-chosen"  # quiet: the binding belongs to that grant only


def test_a_saved_builtin_name_resolves_and_a_corrupt_one_fails_typed():
    from tests.frozen_presence_capabilities import PresenceStateError, _target_from_payload

    old, new = PAIRS[0]
    target = _target_from_payload({"type": "tool", "kind": "builtin", "name": old, "provider": ""}, field_name="t")
    assert target.name == new
    with pytest.raises(PresenceStateError):
        _target_from_payload({"type": "tool", "kind": "builtin", "name": [old], "provider": ""}, field_name="t")
