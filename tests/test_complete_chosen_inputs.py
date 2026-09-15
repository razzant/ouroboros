"""Complete chosen assignments and operative plans at their actual consumer seams."""

from copy import deepcopy
import json

import pytest

from ouroboros.tools import plan_packet, plan_spec


@pytest.mark.parametrize("field", ["goal", "in_scope", "non_goals", "invariants",
                                   "affected_paths", "affected_resources", "evidence",
                                   "acceptance_claims", "decisions", "deferred"])
def test_operative_tail_changes_normalized_identity_and_current_packet(field):
    prefix = "яё𐍈🚀\n" * 500
    raw = {"goal": "Full plan", field: prefix + "TAIL_A"}
    if field != "goal":
        # Also cross the former 40-item list cap; the decisive item is last.
        raw[field] = [f"first {i}" for i in range(40)] + [prefix + "TAIL_A"]
    before, errors = plan_spec.normalize_spec(raw)
    assert not errors
    changed = deepcopy(raw)
    if field == "goal":
        changed[field] = prefix + "TAIL_B"
    else:
        changed[field][-1] = prefix + "TAIL_B"
    after, errors = plan_spec.normalize_spec(changed)
    assert not errors
    assert plan_spec.spec_hash(before) != plan_spec.spec_hash(after)
    assert plan_spec.spec_delta(before, after)["changed"]
    serialized = json.dumps(after, ensure_ascii=False)
    assert "TAIL_B" in serialized and "TAIL_A" not in serialized
    packet = plan_packet.build_plan_review_user_content(
        objective="Parent goal", goal=after["goal"], plan_prose="Chosen plan",
        spec=after, manifest={}, prior_cycles=[], dispositions=[],
        spec_delta=None, root_exploration_log=None,
    )
    assert "TAIL_B" in packet and "OMISSION NOTE" not in packet


@pytest.mark.parametrize("collection,key", [("decisions", "choice"), ("decisions", "why"),
                                           ("decisions", "rejected"), ("deferred", "what"),
                                           ("deferred", "why_safe_to_defer")])
def test_nested_operative_tails_survive(collection, key):
    tail = "строка\n" * 200 + "DECISIVE_NESTED_TAIL"
    item = {"choice": "choice"} if collection == "decisions" else {"what": "later"}
    item[key] = [f"rejected {i}" for i in range(8)] + [tail] if key == "rejected" else tail
    raw = {"goal": "Full plan", collection: [item]}
    spec, errors = plan_spec.normalize_spec(raw)
    assert not errors
    assert spec[collection][0][key] == item[key]
    changed = deepcopy(raw)
    changed[collection][0][key] = item[key] + ["next"] if key == "rejected" else tail + "next"
    assert plan_spec.spec_hash(spec) != plan_spec.spec_hash(plan_spec.normalize_spec(changed)[0])


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
    marker = "HOST TASK CONTRACT AUTHORITY (complete normalized JSON; exact strings are authority):\n"
    host = json.loads(instructions.split(marker, 1)[1])
    assert host["objective"] == objective and host["expected_output"] == expected
    assert host["context"] == " \nHOST_REFERENCE_CONTEXT\n "
    assert "CHOSEN_ASSIGNMENT_TAIL" not in instructions


def test_current_packet_keeps_full_parent_objective_and_plan_prose():
    objective = " \n" + "О" * 8_001 + "OBJECTIVE_TAIL\n "
    prose = " \n" + "П" * 40_001 + "PROSE_TAIL\n "
    spec, errors = plan_spec.normalize_spec({"goal": "g"})
    assert not errors
    packet = plan_packet.build_plan_review_user_content(
        objective=objective, goal="g", plan_prose=prose, spec=spec, manifest={},
        prior_cycles=[], dispositions=[], spec_delta=None, root_exploration_log=None,
    )
    assert objective in packet and prose in packet
    assert "OMISSION NOTE" not in packet
