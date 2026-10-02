from __future__ import annotations

import pytest

from ouroboros.presence_admission import PresenceAdmissionError, admit_presence_turn
from ouroboros.presence_bindings import (
    PresenceBinding,
    PresenceEndpoint,
    new_presence_binding_id,
    save_presence_binding,
)
from ouroboros.presence_capabilities import (
    PresenceSelection,
    PresenceState,
    PresenceToolTarget,
    presence_state_fingerprint,
    save_presence_state,
)
from ouroboros.presence_profile import parse_presence_profile, presence_request_fingerprint
from ouroboros.skill_loader import (
    SkillReviewState,
    load_skill,
    save_enabled,
    save_review_state,
)


def _install_behavior(drive_root, *, enabled: bool = True, required: bool = True):
    skill_dir = drive_root / "skills" / "external" / "community-helper"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: community-helper\n"
        "description: Neutral presence fixture.\n"
        "version: 0.1.0\n"
        "type: instruction\n"
        "presence:\n"
        "  instructions: Participate helpfully in the selected room.\n"
        "  context_topics: [community-guidelines]\n"
        "  runtime_defaults:\n"
        "    model_slot: light\n"
        "    inline_max_rounds: 10\n"
        "  capability_requests:\n"
        "    - id: history\n"
        "      kind: tool\n"
        f"      required: {'true' if required else 'false'}\n"
        "      purpose: Read relevant room history.\n"
        "---\n"
        "# Community helper\n",
        encoding="utf-8",
    )
    loaded = load_skill(skill_dir, drive_root)
    assert loaded is not None
    save_enabled(drive_root, loaded.name, enabled)
    save_review_state(
        drive_root,
        loaded.name,
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    return skill_dir


def _select_history(drive_root, skill_dir) -> None:
    loaded = load_skill(skill_dir, drive_root)
    assert loaded is not None
    profile = parse_presence_profile(loaded.manifest, skill_dir)
    assert profile is not None
    request_fingerprint = presence_request_fingerprint(profile.capability_requests[0])
    state = PresenceState(
        (PresenceSelection(request_fingerprint, PresenceToolTarget("builtin", "chat_history")),)
    )
    save_presence_state(
        drive_root,
        loaded.name,
        state,
        expected_state_fingerprint=presence_state_fingerprint(PresenceState()),
    )


def _select_missing_builtin(drive_root, skill_dir) -> None:
    loaded = load_skill(skill_dir, drive_root)
    assert loaded is not None
    profile = parse_presence_profile(loaded.manifest, skill_dir)
    assert profile is not None
    state = PresenceState((PresenceSelection(
        presence_request_fingerprint(profile.capability_requests[0]),
        PresenceToolTarget("builtin", "does_not_exist"),
    ),))
    save_presence_state(
        drive_root,
        loaded.name,
        state,
        expected_state_fingerprint=presence_state_fingerprint(PresenceState()),
    )


def _binding(drive_root, *, transport_skill: str = "telegram-bot") -> PresenceBinding:
    origin = PresenceEndpoint("telegram", "public-bot", "room-42", "thread-7")
    destination = PresenceEndpoint("telegram", "public-bot", "room-42", "thread-7")
    return save_presence_binding(
        drive_root,
        PresenceBinding(
            new_presence_binding_id(),
            transport_skill,
            "community-helper",
            origin,
            destination,
        ),
    )


def _admit(drive_root, binding):
    return admit_presence_turn(
        drive_root=drive_root,
        authenticated_transport_skill=binding.transport_skill,
        binding_id=binding.binding_id,
        global_max_rounds=8,
        repo_path="",
    )


def test_admission_freezes_reviewed_behavior_runtime_digests_and_authority(tmp_path):
    drive_root = tmp_path / "data"
    skill_dir = _install_behavior(drive_root)
    _select_history(drive_root, skill_dir)
    binding = _binding(drive_root)

    admission = _admit(drive_root, binding)

    assert admission.instructions == "Participate helpfully in the selected room."
    assert admission.context_topics == ("community-guidelines",)
    assert admission.model_slot == "light"
    assert admission.inline_max_rounds == 8
    assert admission.origin == binding.origin
    assert admission.destination == binding.destination
    assert admission.capability_ceiling.skill_name == "community-helper"
    # The reviewed profile selected chat_history; the rest is the constant
    # cognitive baseline plus the own-work baseline every new ceiling carries.
    assert [grant.name for grant in admission.capability_ceiling.tool_grants] == [
        "chat_history",
        "chronicle_write",
        "get_task_result",
        "knowledge_list",
        "knowledge_read",
        "knowledge_write",
        "memory_mark",
        "memory_read",
        "recent_tasks",
        "steer_task",
        "update_identity",
        "update_scratchpad",
    ]
    assert admission.capability_ceiling.skill_content_hash == admission.skill_content_hash
    assert admission.capability_ceiling.profile_fingerprint == admission.profile_fingerprint
    assert admission.capability_ceiling.state_fingerprint == admission.state_fingerprint
    assert admission.capability_ceiling.selection_fingerprint == admission.selection_fingerprint


def test_admission_rejects_binding_owned_by_another_transport(tmp_path):
    drive_root = tmp_path / "data"
    _install_behavior(drive_root)
    binding = _binding(drive_root)

    with pytest.raises(PresenceAdmissionError) as caught:
        admit_presence_turn(
            drive_root=drive_root,
            authenticated_transport_skill="slack-bridge",
            binding_id=binding.binding_id,
            global_max_rounds=8,
            repo_path="",
        )

    assert caught.value.code == "presence_binding_wrong_transport"


@pytest.mark.parametrize(
    ("state", "expected"),
    [("disabled", "presence_behavior_skill_disabled"), ("stale", "presence_behavior_review_stale")],
)
def test_admission_rejects_disabled_or_stale_behavior(tmp_path, state, expected):
    drive_root = tmp_path / "data"
    _install_behavior(drive_root, enabled=state != "disabled")
    if state == "stale":
        save_review_state(
            drive_root,
            "community-helper",
            SkillReviewState(status="pass", content_hash="0" * 64),
        )
    binding = _binding(drive_root)

    with pytest.raises(PresenceAdmissionError) as caught:
        _admit(drive_root, binding)

    assert caught.value.code == expected


def test_admission_rejects_missing_required_selection(tmp_path):
    drive_root = tmp_path / "data"
    _install_behavior(drive_root)
    binding = _binding(drive_root)

    with pytest.raises(PresenceAdmissionError) as caught:
        _admit(drive_root, binding)

    assert caught.value.code == "presence_authority_missing_required"


def test_admission_rejects_selected_but_unavailable_required_target(tmp_path):
    drive_root = tmp_path / "data"
    skill_dir = _install_behavior(drive_root)
    _select_missing_builtin(drive_root, skill_dir)
    binding = _binding(drive_root)

    with pytest.raises(PresenceAdmissionError) as caught:
        _admit(drive_root, binding)

    assert caught.value.code == "presence_required_capability_unavailable"


def _select_mcp(drive_root, skill_dir, *, name: str, provider: str) -> None:
    loaded = load_skill(skill_dir, drive_root)
    profile = parse_presence_profile(loaded.manifest, skill_dir)
    state = PresenceState((PresenceSelection(
        presence_request_fingerprint(profile.capability_requests[0]),
        PresenceToolTarget("mcp", name, provider),
    ),))
    save_presence_state(drive_root, loaded.name, state,
                        expected_state_fingerprint=presence_state_fingerprint(PresenceState()))


def _serve_mcp_tools(monkeypatch, tools: dict) -> None:
    from types import SimpleNamespace

    from ouroboros import mcp_client

    monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", lambda **_kw: None)
    monkeypatch.setattr(mcp_client, "get_manager", lambda: SimpleNamespace(
        get_tool=lambda name: {"name": name, "server_id": tools[name]} if name in tools else None))


WIRE = "mcp_my_company_b762e68c1ef1__list_files"


@pytest.mark.parametrize("provider", ["my_company_b762e68c1ef1", "my_company__b762e68c1ef1"])
def test_mcp_grant_is_ready_for_its_exact_or_pre_fixed_point_provider(tmp_path, monkeypatch, provider):
    # #1328: a grant stored before the fixed-point id kept the unconverged
    # provider; the same wire tool on the one server it now names is that grant.
    drive_root = tmp_path / "data"
    skill_dir = _install_behavior(drive_root)
    _select_mcp(drive_root, skill_dir, name=WIRE, provider=provider)
    _serve_mcp_tools(monkeypatch, {WIRE: "my_company_b762e68c1ef1"})
    admission = _admit(drive_root, _binding(drive_root))
    assert WIRE in [grant.name for grant in admission.capability_ceiling.tool_grants]


@pytest.mark.parametrize("tools", [
    {},  # an ambiguous id serves no tools at all
    {WIRE: "another_server"},  # the wire name now belongs to a different server
])
def test_mcp_grant_is_refused_when_its_server_is_not_the_named_one(tmp_path, monkeypatch, tools):
    drive_root = tmp_path / "data"
    skill_dir = _install_behavior(drive_root)
    _select_mcp(drive_root, skill_dir, name=WIRE, provider="my_company__b762e68c1ef1")
    _serve_mcp_tools(monkeypatch, tools)
    with pytest.raises(PresenceAdmissionError) as caught:
        _admit(drive_root, _binding(drive_root))
    assert caught.value.code == "presence_required_capability_unavailable"
