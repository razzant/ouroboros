"""The paused-work display reads only the model requests still awaiting an answer (#1528).

``owner_pause.trees_model_requests`` answers every paused root in one read of the usage store's open-set index,
narrowed to dispatched and unresolved rows: the only states ``tree_census.model_handoffs`` names. Admission's
``tree_model_requests`` keeps the whole open set, whose reserved rows tie damaged members to their tree
(``tests/test_tree_census_damage.py``). Attempts are written by the store's own writer, cloned from rows the real
accounting API produced; the display's consumer (``work_status.model_tails``) is read beside each answer.
"""
from __future__ import annotations

import pytest

from tests._cost_meter import CostMeter, tagged

ROOTS = ("paused-a", "paused-b", "paused-c")
_DROP = {"revision", "seq", "ts"}


def _templates(root):
    """One row chain per state, as the real accounting API writes it (read back as the store holds it)."""
    from ouroboros import usage_accounting as ua, usage_store

    def request(task):
        return ua.AttemptRequest(model="vendor/model", provider="test", reservation_usd=.2, drive_root=root,
                                 task_id=task, root_task_id="template-root", category="task", source="llm.chat")

    def row(reservation):
        with usage_store.read(root) as txn:
            return txn.attempt(reservation.attempt_id)

    chains = {}
    for state in ("abandoned", "unresolved", "settled"):
        reserved = ua.reserve_attempt(request(f"template-{state}"))
        chain = [row(reserved)]
        ua.mark_dispatched(reserved)
        chain.append(row(reserved))
        if state == "abandoned":
            ua.terminalize_abandoned_attempt(reserved, reason="owner_task_terminal")
        elif state == "unresolved":
            ua.mark_unresolved(reserved, "provider_outcome_unknown")
        else:
            ua.settle_attempt(reserved, {"prompt_tokens": 10, "completion_tokens": 5}, cost_usd=0.001,
                              cost_final=True)
        chains[state] = [*chain, row(reserved)]
    chains["dispatched"] = chains["settled"][:2]
    chains["reserved"] = chains["settled"][:1]
    return chains


def _seed(root, chains, plan):
    """``plan``: (state, count, root_task_id, task_id) rows, each walking its template chain."""
    from ouroboros import usage_store

    counter = 0
    with usage_store.hold(root) as txn:
        for state, count, root_task_id, task_id in plan:
            for _ in range(count):
                counter += 1
                previous = None
                for template in chains[state]:
                    row = {key: value for key, value in template.items() if key not in _DROP}
                    row.update(attempt_id=f"att-{state}-{counter:06d}", task_id=task_id, root_task_id=root_task_id,
                               parent_task_id=root_task_id)
                    if row.get("billing_group_id"):
                        row["billing_group_id"] = root_task_id
                    previous = txn.write(row, previous)


def _trees(root):
    from ouroboros.task_results import write_task_result

    for rid in ROOTS:
        write_task_result(root, rid, "running", root_task_id=rid, text="paused root")
        write_task_result(root, f"{rid}-c0", "completed", parent_task_id=rid, root_task_id=rid,
                          delegation_role="subagent", result="member")


def _answers(root):
    from ouroboros.gateway.work_status import model_tails
    from supervisor import tree_census

    tree_census._MEMO.clear()
    plural = tree_census.trees_model_handoffs(root, ROOTS)
    tree_census._MEMO.clear()
    single = {rid: tree_census.model_handoffs(root, rid) for rid in ROOTS}
    return plural, single, {rid: model_tails(plural[rid]) for rid in ROOTS}, {rid: model_tails(single[rid])
                                                                               for rid in ROOTS}


def test_the_display_read_addresses_the_open_set_index(tmp_path, monkeypatch):
    from ouroboros import owner_pause, usage_store

    _templates(tmp_path)
    statements, real = [], usage_store.capture
    monkeypatch.setattr(usage_store, "capture",
                        lambda conn, sql, params, *a, **k: statements.append((conn, sql, params)) or real(
                            conn, sql, params, *a, **k))
    assert owner_pause.trees_model_requests(tmp_path, ROOTS) == {rid: [] for rid in ROOTS}
    (conn, sql, params), = statements
    with usage_store.read(tmp_path) as txn:
        plan = [str(row[-1]) for row in txn.conn.execute(f"EXPLAIN QUERY PLAN {sql}", params)]
    assert any("attempts_open" in step for step in plan), plan
    assert not any(step.startswith("SCAN attempts") for step in plan), plan


@pytest.mark.parametrize("state", ["none", "reserved", "abandoned", "settled", "dispatched", "unresolved",
                                   "review_dispatched"])
def test_the_display_answers_what_admission_answers_for_every_state(tmp_path, state):
    _trees(tmp_path)
    chains = _templates(tmp_path)
    if state == "review_dispatched":
        chains["review_dispatched"] = [{**row, "review_slot_id": "slot", "review_skill": "review"}
                                       for row in chains["dispatched"]]
    if state != "none":
        _seed(tmp_path, chains, [(state, 2, "paused-a", "paused-a-c0")])
    plural, single, plural_tails, single_tails = _answers(tmp_path)
    assert plural == single and plural_tails == single_tails
    named = [row["kind"] for row in plural["paused-a"]]
    assert named == (["model_handoff"] * 2 if state in {"dispatched", "unresolved", "review_dispatched"} else [])
    assert plural["paused-b"] == plural["paused-c"] == []


def test_a_tree_without_a_sent_request_reads_no_member(tmp_path, monkeypatch):
    from supervisor import tree_census

    _trees(tmp_path)
    chains = _templates(tmp_path)
    _seed(tmp_path, chains, [("reserved", 2, "paused-a", "paused-a-c0"), ("abandoned", 5, "paused-a", "paused-a-c0"),
                             ("abandoned", 5, "paused-b", "paused-b-c0"), ("dispatched", 1, "paused-c", "paused-c-c0")])
    censuses, real = [], tree_census.writer_facts
    monkeypatch.setattr(tree_census, "writer_facts", lambda root, tree, **k: censuses.append(tree) or real(root, tree, **k))
    tree_census._MEMO.clear()
    answered = tree_census.trees_model_handoffs(tmp_path, ROOTS)
    assert censuses == ["paused-c"], "only the tree with a sent request reads its members"
    assert answered["paused-a"] == answered["paused-b"] == []
    assert [row["kind"] for row in answered["paused-c"]] == ["model_handoff"]


@pytest.mark.parametrize("shape", ["inactive_only", "inactive_plus_dispatched"])
def test_a_broken_census_shows_the_owner_the_same_display(tmp_path, monkeypatch, shape):
    """Only the intermediate row differs: no member read, so no unreadable census beside unsent requests."""
    from ouroboros import owner_pause

    _trees(tmp_path)
    chains = _templates(tmp_path)
    plan = [("abandoned", 3, "paused-a", "paused-a-c0"), ("reserved", 2, "paused-a", "paused-a-c0")]
    if shape == "inactive_plus_dispatched":
        plan.append(("dispatched", 1, "paused-a", "paused-a-c0"))
    _seed(tmp_path, chains, plan)

    def broken(*_args, **_kwargs):
        raise ValueError("tree_membership_unreadable")

    monkeypatch.setattr(owner_pause, "tree_member_results", broken)
    plural, single, plural_tails, single_tails = _answers(tmp_path)
    assert [row["kind"] for row in single["paused-a"]] == ["attempt_custody_unreadable"]
    assert [row["kind"] for row in plural["paused-a"]] == (
        [] if shape == "inactive_only" else ["attempt_custody_unreadable"])
    assert plural_tails == single_tails == {rid: [] for rid in ROOTS}


def test_the_display_read_is_flat_against_history_and_unsent_rows(tmp_path):
    from ouroboros import owner_pause

    def measured(name, inactive):
        root = tmp_path / name
        root.mkdir()
        chains = _templates(root)
        plan = [("dispatched", 1, rid, f"{rid}-c0") for rid in ROOTS]
        plan += [("unresolved", 1, rid, f"{rid}-c0") for rid in ROOTS]
        plan += [(state, inactive, rid, f"{rid}-c0") for rid in ROOTS for state in ("settled", "abandoned", "reserved")]
        plan.append(("abandoned", inactive, "unpaused-root", "unpaused-root-t"))
        _seed(root, chains, plan)
        with CostMeter(root) as meter:
            rows = tagged(name, lambda: owner_pause.trees_model_requests(root, ROOTS))
            counts = meter.snapshot(name)["counts"]
        assert {rid: sorted(row["state"] for row in rows[rid]) for rid in ROOTS} == {
            rid: ["dispatched", "unresolved"] for rid in ROOTS}
        return {key: counts.get(key, 0) for key in ("statements", "rows_crossed", "json_loads", "json_chars")}

    small, large = measured("small", 2), measured("large", 200)
    assert small == large, (small, large)
