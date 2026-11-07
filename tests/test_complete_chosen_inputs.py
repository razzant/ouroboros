"""Complete chosen assignments and operative plans at their actual consumer seams."""

import json
from types import SimpleNamespace

import pytest

from ouroboros import plan_record
from ouroboros.tools import plan_packet

_PREFIX = "яё𐍈🚀\n" * 500


def _revision_fields(field: str, tail: str) -> dict:
    """A plan whose decisive text sits at the END of a long field; list fields carry 40
    ordinary items before the decisive one (the retired spec capped lists at 40)."""
    fields = {
        "goal": "Full plan", "plan": "Chosen plan",
        "done_when": [{"id": f"c{i}", "claim": f"first {i}"} for i in range(40)]
        + [{"id": "decisive", "claim": "the decisive claim", "support": "check it"}],
        "affected_paths": [f"src/first_{i}.py" for i in range(40)] + ["src/decisive.py"],
    }
    if field in ("goal", "plan"):
        fields[field] = _PREFIX + tail
    elif field == "affected_paths":
        fields["affected_paths"][-1] = _PREFIX + tail
    else:
        fields["done_when"][-1][field.split(".", 1)[1]] = _PREFIX + tail
    return fields


def _participant_content(revision: dict) -> str:
    return plan_packet.build_participant_content(
        objective="Parent goal", owner_words="", dialogue_section="", revision=revision, delta=None, cross_talk="",
        own_objections=[], message="", evidence_section=plan_packet.render_evidence({}), first_time=True)


@pytest.mark.parametrize("field", ["goal", "done_when.support", "affected_paths"])
def test_operative_tail_changes_revision_identity_and_current_delivery(field):
    before = plan_record.compose_revision(None, **_revision_fields(field, "TAIL_A"))
    after = plan_record.compose_revision(None, **_revision_fields(field, "TAIL_B"))
    assert len(after["done_when"]) == 41 and len(after["affected_paths"]) == 41  # nothing past an item cap is lost
    assert before["content_sha256"] != after["content_sha256"]
    delta = plan_record.revision_delta({**before, "n": 1}, {**after, "n": 2})
    assert any(delta[key] for key in ("goal_changed", "plan_changed", "done_when_reworded", "affected_paths_added"))
    serialized = json.dumps(after, ensure_ascii=False)
    assert "TAIL_B" in serialized and "TAIL_A" not in serialized
    packet = _participant_content({**after, "n": 2, "body": False})
    assert "TAIL_B" in packet and "TAIL_A" not in packet and "OMISSION NOTE" not in packet


def test_large_direct_request_keeps_chosen_prompt_and_host_instruction_roles(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon
    from ouroboros.gateways import claudexor
    from tests.test_nanny_economics import _start_with_contract

    # Importing the helper does not activate its module's autouse fixture.
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: claudexor.ClaudexorGateway())
    prompt = " \n" + "яё𐍈🚀\n" * 55_000 + "CHOSEN_ASSIGNMENT_TAIL\n "
    objective = "HOST_OBJECTIVE:" + "О" * 250_001
    expected = "HOST_EXPECTED:" + "Е" * 250_001
    request = _start_with_contract(tmp_path, monkeypatch, {
        "objective": objective, "expected_output": expected,
        "context": " \nHOST_REFERENCE_CONTEXT\n ",
    }, prompt=prompt)
    assert request["prompt"] == prompt
    instructions = request["instructions"]
    assert instructions.count(objective) == 1 and instructions.count(expected) == 1
    marker = "HOST TASK CONTRACT AUTHORITY (normalized JSON; predecessor is a brief):\n"
    # The contract JSON is followed by my human's words for the work, so decode just the JSON.
    host, _end = json.JSONDecoder().raw_decode(instructions.split(marker, 1)[1])
    assert host["objective"] == objective and host["expected_output"] == expected
    assert host["context"] == " \nHOST_REFERENCE_CONTEXT\n "
    assert "CHOSEN_ASSIGNMENT_TAIL" not in instructions


def test_current_delivery_keeps_full_parent_objective_and_plan_prose():
    from ouroboros.tools import plan_ask

    objective = " \n" + "О" * 8_001 + "OBJECTIVE_TAIL\n "
    prose = " \n" + "П" * 40_001 + "PROSE_TAIL\n "
    revision = plan_record.compose_revision(None, goal="g", plan=prose, done_when=[], affected_paths=[])
    assert revision["plan"] == prose  # the revision keeps the chosen prose byte for byte
    # What a participant receives, assembled by the organ from the task contract and the revision.
    packet = plan_ask.participant_content(
        SimpleNamespace(task_contract={"objective": objective}, task_metadata={}), plan_record.empty_record(),
        {**revision, "n": 1, "body": False}, plan_ask.Participant(SimpleNamespace(slot_id="s1"), owner_marked=True),
        message="", manifest={}, dialogue_section="")
    # The rendering trims only the prose's outer whitespace; every character of the text stays.
    assert objective in packet and prose.strip() in packet
    assert "OMISSION NOTE" not in packet
