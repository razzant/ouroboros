from __future__ import annotations

import pytest

from ouroboros.presence_admission import PresenceAdmissionError, admit_presence_turn
from ouroboros.presence_bindings import PresenceBinding, PresenceEndpoint
from tests.frozen_presence_bindings import new_presence_binding_id, save_presence_binding
from tests.frozen_presence_capabilities import (
    PresenceSelection,
    PresenceState,
    PresenceToolTarget,
    presence_state_fingerprint,
    save_presence_state,
)
from tests.frozen_presence_profile import parse_presence_profile, presence_request_fingerprint
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


def test_admission_migrates_without_behavior_gates_or_ceiling(tmp_path):
    skill = _install_behavior(tmp_path, enabled=False)
    _select_missing_builtin(tmp_path, skill)
    binding = _binding(tmp_path)
    admitted = _admit(tmp_path, binding)
    assert admitted.place and admitted.binding_id == binding.binding_id
    assert admitted.instructions == "Participate helpfully in the selected room."
    assert admitted.model_slot == "light" and admitted.inline_max_rounds is None
    assert admitted.capability_ceiling is None
    assert admitted.origin == binding.origin and admitted.destination == binding.destination


def test_admission_requires_matching_authenticated_connector(tmp_path):
    _install_behavior(tmp_path)
    binding = _binding(tmp_path)
    assert _admit(tmp_path, binding).place
    with pytest.raises(PresenceAdmissionError) as caught:
        admit_presence_turn(drive_root=tmp_path, authenticated_transport_skill="unrelated", binding_id=binding.binding_id)
    assert caught.value.code == "presence_binding_wrong_transport"


def test_new_admission_uses_scope_without_binding_id(tmp_path):
    from ouroboros.places import connect_place
    place = connect_place(tmp_path, connector="bridge", provider="chat", account_id="a", conversation_id="r")
    admitted = admit_presence_turn(drive_root=tmp_path, authenticated_transport_skill="bridge",
                                  provider="chat", account_id="a", conversation_id="r")
    assert admitted.binding_id == place.place_id
    with pytest.raises(PresenceAdmissionError):
        admit_presence_turn(drive_root=tmp_path, authenticated_transport_skill="bridge",
                            provider="chat", account_id="other", conversation_id="r")
