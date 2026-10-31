"""#1567: what a Continue successor's FIRST model request actually carries.

Driven through the real consumers: ``admit_continuation`` on an installed queue,
the successor's startup authority materialization
(``validate_task_authority_sources``) and ``build_llm_messages`` in Max. The
predecessor's compact brief (status and cause, owner words, omitted fields with sizes) is in that first request beside the exact reader of
the complete record, so the work order points to it instead of ordering it read;
the request names no actor (a nonce proves the action, not who pressed it); the
predecessor's stored self-body handbook choice survives admission, its restored
contract-only copy, replay and a second Continue after pre-start shutdown; the system-checkout
SHA is labelled with its own surface capture time, never as the task's start; a named
predecessor that reads back empty at start refuses it instead of an empty brief.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros.task_store import delete_document
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_doc_context import _make_env_and_memory, foreign_tree
from tests.test_owner_continue import NONCE, _interrupted

FIELD = "context_requires_self_body_docs"
SYSTEM_ROOT = "/checkouts/ouroboros"
BASELINE_START = "2026-10-01T08:00:00+00:00"
SURFACE_ADVANCED = "2026-10-01T09:30:00+00:00"
TERMINAL_AT = "2026-10-01T10:00:00+00:00"


def _mutation_evidence() -> dict:
    """An advanced baseline: the task's own commit re-anchored the system surface
    (its SHA and captured_at), the top-level baseline time stayed at the start."""
    return {
        "version": 1,
        "effect_state": "quiescent",
        "baseline": {
            "captured_at": BASELINE_START,
            "surfaces": [
                {"surface_type": "external_workspace", "canonical_root": "/work/copy",
                 "captured_at": BASELINE_START, "known_paths": [], "git": {"base_commit": "e" * 40}},
                {"surface_type": "system_repo", "canonical_root": SYSTEM_ROOT, "captured_at": SURFACE_ADVANCED,
                 "known_paths": [], "git": {"base_commit": "b" * 40, "base_tree": "f" * 40, "dirty_paths": []}},
            ],
            "epochs": [{"advanced_at": SURFACE_ADVANCED, "reason": "attributed_commit"}],
        },
        "terminal_candidate_snapshot": {
            "captured_at": TERMINAL_AT,
            "surfaces": [
                {"surface_type": "external_workspace", "canonical_root": "/work/copy", "base_commit": "d" * 40},
                {"surface_type": "system_repo", "canonical_root": SYSTEM_ROOT, "base_commit": "c" * 40},
            ],
        },
    }


def _first_request(env, memory, task):
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.context import build_llm_messages

    task = copy.deepcopy(task)
    assert validate_task_authority_sources(env, task) == {}
    messages, _cap = build_llm_messages(env=env, memory=memory, task=task)
    return task, messages


@pytest.mark.parametrize("flag", [True, False, None])
def test_successor_first_request_carries_the_predecessor_once_with_its_flag_and_dated_checkout(
        tmp_path, monkeypatch, flag):
    from ouroboros.agent import Env
    from ouroboros.memory import Memory
    from ouroboros.server_restart import _shutdown_task_cleanup_args
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setattr("ouroboros.context.resolve_context_fit_route", lambda *_a, **_kw: (
        {"model": "fixture", "provider": "fixture"},
        SimpleNamespace(route_fp="fixture", status="asserted", stale=False, window_tokens=1_000_000),
    ))
    books, _books_memory = _make_env_and_memory(tmp_path / "books")
    for sub in ("state", "memory/knowledge", "logs"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory" / "identity.md").write_text("I am Ouroboros.", encoding="utf-8")
    env = Env(repo_dir=books.repo_dir, drive_root=tmp_path, budget_drive_root=tmp_path)
    memory = Memory(tmp_path, repo_dir=books.repo_dir)
    workspace = foreign_tree(tmp_path / "working-copy")  # proven foreign: only the flag restores the handbook
    contract = {"context": "PREDECESSOR-CONTRACT-SENTINEL", **({FIELD: flag} if flag is not None else {})}
    _interrupted(tmp_path, workspace_root=str(workspace), workspace_mode="external",
                 task_contract=contract, mutation_evidence=_mutation_evidence())
    before = load_task_result(tmp_path, "pred-1")

    ack = admit_continuation("pred-1", action_nonce=NONCE)

    assert ack["ok"] is True and ack["held"] is False, ack
    successor = ack["successor_task_id"]
    row = next(task for task in workers.PENDING if task["id"] == successor)
    after = load_task_result(tmp_path, "pred-1")
    for key in ("task_contract", "mutation_evidence"):
        assert after[key] == before[key], "the predecessor's read sources are left exactly as recorded"

    # The stored choice rides on the successor itself, false included; absent stays absent.
    if flag is None:
        assert FIELD not in row
    else:
        assert row[FIELD] is flag
    assert row["task_contract"][FIELD] is bool(flag)
    assert load_task_result(tmp_path, successor)["metadata"]["task_contract"][FIELD] is bool(flag)
    expected_binding = {"workspace_root": str(workspace), "workspace_mode": "external",
                        "deadline_at": before["deadline_at"]}
    assert all(load_task_result(tmp_path, successor).get(key) == value
               for key, value in expected_binding.items()), "initial publication preserves the binding"

    text = row["text"]
    assert text.startswith("[CONTINUE] Continue was requested for task pred-1, which was interrupted")
    assert "owner pressed" not in text.lower(), "a nonce proves the action, not its actor"
    assert "Read its saved results" not in text and "already in your context" in text
    source = row["predecessor_authority_source"]
    assert source["arguments"] == {"task_id": "pred-1", "include_authority": True}
    assert text.count("get_task_result(task_id='pred-1', include_authority=True)") == 1
    # Existing authority lines are unchanged: delegated runs, schedules, money, deadline, owner words.
    for kept in ("continue_from=<run_id>", "manage_schedules(action='list')",
                 "Original money and hard deadlines still apply to related work.",
                 "An explicit hard deadline still applies to this work: 2099-01-01T00:00:00+00:00.",
                 "Owner's original instruction (verbatim):\nWrite the Friday report"):
        assert kept in text
    # The checkout line: the system surface's own dated observations only.
    assert f"HEAD {'b' * 40} at {SURFACE_ADVANCED}" in text
    assert f"its terminal snapshot saw HEAD {'c' * 40} at {TERMINAL_AT}" in text
    assert "git_head in your Runtime context" in text
    for absent in ("e" * 40, "d" * 40, BASELINE_START, "started on", "ran on"):
        assert absent not in text

    # The fresh first request in Max, and the restored queue row (the snapshot keeps
    # task_contract, not the top-level flag), make the same handbook decision.
    restored = copy.deepcopy(row)
    restored.pop(FIELD, None)
    handbooks = []
    for candidate in (row, restored):
        task, messages = _first_request(env, memory, candidate)
        system = messages[0]["content"]
        handbooks.append(sum(block["text"].startswith("## DEVELOPMENT.md\n") for block in system))
        system_text = "\n".join(block["text"] for block in system)
        rendered = json.dumps(messages, ensure_ascii=False)
        assert '"predecessor_authority"' in system_text and "PREDECESSOR-CONTRACT-SENTINEL" not in rendered
        brief = task["predecessor_authority"]
        # The brief carries status and the exact reader; the predecessor contract and
        # mutation evidence remain at that source, with their omission sizes named.
        assert "task_contract" not in brief and brief["omitted_fields"]["task_contract"] > 0
        assert load_task_result(tmp_path, "pred-1")["task_contract"]["context"] == "PREDECESSOR-CONTRACT-SENTINEL"
        assert brief["source"]["arguments"] == {"task_id": "pred-1", "include_authority": True}
        assert "mutation_evidence" not in brief and brief["omitted_fields"]["mutation_evidence"] > 0
        assert json.dumps(before["mutation_evidence"]) not in rendered
        users = json.dumps([m for m in messages if m.get("role") == "user"], ensure_ascii=False)
        assert json.dumps(text, ensure_ascii=False)[1:-1] in users, "the work order is the first user turn"
    assert handbooks == [int(flag is True)] * 2, "only an explicit true restores the handbook in an external folder"

    # A recovered admission (queue row kept, result row lost) keeps the same contract.
    assert delete_document(tmp_path, successor)
    recovered = admit_continuation("pred-1", action_nonce=NONCE)
    assert recovered["replay"] is True and recovered["recovered"] is True
    assert load_task_result(tmp_path, successor)["metadata"]["task_contract"][FIELD] is bool(flag)
    assert all(load_task_result(tmp_path, successor).get(key) == value
               for key, value in expected_binding.items()), "replay preserves folder AND deadline"

    # B never reaches a worker: the ordinary shutdown writer preserves the
    # admission/replay contract ONLY under metadata. Continue B -> C reads it.
    events = []
    monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", False)
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))
    status, reason = _shutdown_task_cleanup_args(restart_requested=False)
    assert not workers.RUNNING and not workers.WORKERS
    assert workers.kill_workers(terminal_status=status, result_reason=reason, stop_source="server_shutdown")
    stopped = load_task_result(tmp_path, successor)
    assert not workers.PENDING
    assert stopped["status"] == "cancelled" and stopped["cancel_origin"]["source"] == "server_shutdown"
    assert "task_contract" not in stopped and FIELD not in stopped
    assert stopped["deadline_at"] == expected_binding["deadline_at"]
    assert (stopped["workspace_root"], stopped["workspace_mode"]) == (str(workspace), "external")
    assert stopped["metadata"]["task_contract"][FIELD] is bool(flag)
    assert any(event.get("task_id") == successor and event["type"] == "task_done" for event in events)

    second = admit_continuation(successor, action_nonce=NONCE)
    assert second["ok"] is True and second["held"] is False, second
    second_id = second["successor_task_id"]
    second_row = next(task for task in workers.PENDING if task["id"] == second_id)
    # Restore the actual persisted queue source, which omits the top-level flag.
    workers.PENDING.clear()
    assert queue.restore_pending_from_snapshot() == 1
    restored = workers.PENDING[0]
    assert restored["id"] == second_id and FIELD not in restored
    for candidate in (second_row, restored):
        # Real dispatch: B's pre-start result row handed its folder on to C.
        assert all(candidate.get(key) == value for key, value in expected_binding.items())
        assert (candidate["workspace_root"], candidate["workspace_mode"]) == (str(workspace), "external")
        task, messages = _first_request(env, memory, candidate)
        handbooks = [block for block in messages[0]["content"]
                     if block["text"].startswith("## DEVELOPMENT.md\n")]
        assert len(handbooks) == int(flag is True), "second Continue must carry the stored handbook choice"
        assert "task_contract" not in task["predecessor_authority"]
        assert task["predecessor_authority"]["omitted_fields"]["task_contract"] > 0
        assert load_task_result(tmp_path, successor)["metadata"]["task_contract"] == stopped["metadata"]["task_contract"]
        assert task["predecessor_authority_source"]["arguments"] == {
            "task_id": successor, "include_authority": True}
        users = json.dumps([m for m in messages if m.get("role") == "user"], ensure_ascii=False)
        assert json.dumps(second_row["text"], ensure_ascii=False)[1:-1] in users
        assert "Owner's original instruction (verbatim):\nWrite the Friday report" in second_row["text"]
    # A's absent flag was normalized by B's ordinary admission; C copies that
    # STORED false. Admission above still proves absence was not copied from A.
    assert second_row[FIELD] is bool(flag)
    assert restored["task_contract"][FIELD] is bool(flag)
    assert load_task_result(tmp_path, successor)["metadata"] == stopped["metadata"]
    assert delete_document(tmp_path, second_id)
    replay = admit_continuation(successor, action_nonce=NONCE)
    assert replay["replay"] is True and replay["recovered"] is True
    assert load_task_result(tmp_path, second_id)["metadata"]["task_contract"][FIELD] is bool(flag)
    assert all(load_task_result(tmp_path, second_id).get(key) == value
               for key, value in expected_binding.items()), "second-hop replay retains the same binding"
    _, messages = _first_request(env, memory, workers.PENDING[0])
    assert sum(block["text"].startswith("## DEVELOPMENT.md\n")
               for block in messages[0]["content"]) == int(flag is True)


@pytest.mark.parametrize("canonical,metadata,expected", [
    pytest.param({FIELD: False}, {FIELD: True}, False, id="canonical-false-wins"),
    pytest.param({FIELD: True}, {FIELD: False}, True, id="canonical-true-wins"),
    pytest.param({"context": "partial canonical contract"}, {FIELD: True}, True, id="per-field-fallback"),
    pytest.param(None, {FIELD: True}, True, id="metadata-only-true"),
    pytest.param(None, {FIELD: False}, False, id="metadata-only-false"),
    pytest.param({}, {}, None, id="absent-stays-absent"),
    pytest.param({FIELD: None}, {FIELD: True}, None, id="canonical-nonbool-wins"),
    pytest.param({}, {FIELD: "true"}, None, id="metadata-nonbool-not-copied"),
    pytest.param({}, "invalid", None, id="non-dict-carrier-ignored"),
])
def test_admission_copies_only_stored_booleans_with_canonical_field_precedence(
        tmp_path, monkeypatch, canonical, metadata, expected):
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, task_contract=canonical, metadata={"task_contract": metadata})
    before = load_task_result(tmp_path, "pred-1")
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"] is True and ack["held"] is False, ack
    row = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    if expected is None:
        assert FIELD not in row, "absence and non-booleans must not invent a top-level false"
    else:
        assert row[FIELD] is expected
    after = load_task_result(tmp_path, "pred-1")
    for key in ("task_contract", "metadata"):
        assert after.get(key) == before.get(key)


@pytest.mark.parametrize("damage", ["missing", "quarantined"])
def test_a_named_predecessor_that_reads_back_empty_refuses_the_start(tmp_path, monkeypatch, damage):
    """The work order says the saved brief is already in context. A record gone
    (or quarantined as unreadable) by the successor's start is the typed startup
    refusal, never an empty envelope standing in for it."""
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    task = copy.deepcopy(next(row for row in workers.PENDING if row["id"] == ack["successor_task_id"]))
    assert "already in your context" in task["text"]
    if damage == "missing":
        assert delete_document(tmp_path, "pred-1")
    else:  # what the import leaves for an unreadable file: no row, the identity occupied
        from ouroboros import store
        from ouroboros.store_import import OCCUPIED
        from ouroboros.task_result_schema import QUARANTINED_SCHEMA_REASON

        with store.open_store(tmp_path, write=True) as txn:
            txn.delete_task("pred-1")
            txn.set_meta(OCCUPIED, {**(txn.meta(OCCUPIED) or {}), "pred-1": QUARANTINED_SCHEMA_REASON})

    refusal = validate_task_authority_sources(SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path), task)

    assert refusal["reason_code"] == "authority_source_unavailable"
    assert refusal["source"] == task["predecessor_authority_source"]
    assert "predecessor_authority" not in task


def test_without_a_materializable_authority_the_work_order_keeps_the_only_route(tmp_path, monkeypatch):
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)

    def unavailable(_row):
        raise ValueError("authority projection unavailable")

    monkeypatch.setattr("ouroboros.server_routing_context._task_result_ground_truth", unavailable)
    ack = admit_continuation("pred-1", action_nonce=NONCE)

    row = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    assert "predecessor_authority_source" not in row and "predecessor_task_id" not in row
    assert "already in your context" not in row["text"]
    assert ("Its saved result is not in your context: get_task_result(task_id='pred-1', "
            "include_authority=True) returns it") in row["text"]
    assert "Recorded observation of that task's system checkout" not in row["text"], "nothing recorded, no line"


def test_peer_mail_rides_as_a_count_and_reader_not_bodies(tmp_path, monkeypatch):
    """Messages other tasks addressed to the predecessor are not concatenated into the
    assignment: the work order carries their count, size and senders plus the exact
    reader of the successor's own stored ``metadata.continuation.peer_context``, which
    the real store resolves; a predecessor without peer mail gets no such line."""
    from ouroboros.owner_mailbox import write_task_message
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.query_data import _query_data
    from ouroboros.tools.registry import ToolContext
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    bodies = [f"peer note {index}: " + "x" * 600 + f" END{index}" for index in range(40)]
    for index, body in enumerate(bodies):
        assert write_task_message(tmp_path, body, "pred-1", source_task_id=f"peer-task-{index % 3}",
                                  msg_id=f"peer-{index}")
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"] is True, ack
    successor = ack["successor_task_id"]
    row = next(task for task in workers.PENDING if task["id"] == successor)
    text = row["text"]
    reader = (f"query_data(op='json', kind='task_result', ids=[{successor!r}], "
              "select='metadata.continuation.peer_context')")
    assert f"40 messages, {sum(len(b) for b in bodies)} characters, from peer-task-0, peer-task-1, peer-task-2." in text
    assert reader in text and text.count("Messages from other tasks") == 1
    assert not any(body in text for body in bodies), "no peer body rides in the assignment"
    assert "Owner's original instruction (verbatim):\nWrite the Friday report" in text
    carried = row["metadata"]["continuation"]["peer_context"]
    assert [item["text"] for item in carried] == bodies
    assert [item["text"] for item in load_task_result(tmp_path, successor)["metadata"]["continuation"]["peer_context"]] == bodies
    # The named reader resolves the exact rows from the successor's own record, in order.
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, budget_drive_root=str(tmp_path), task_id=successor)
    payload = _query_data(ctx, op="json", kind="task_result", ids=[successor],
                          select="metadata.continuation.peer_context")
    assert all(body in payload for body in bodies) and payload.index(bodies[0]) < payload.index(bodies[-1])

    _interrupted(tmp_path, "pred-2")
    quiet = admit_continuation("pred-2", action_nonce=NONCE)
    assert quiet["ok"] is True, quiet
    quiet_row = next(task for task in workers.PENDING if task["id"] == quiet["successor_task_id"])
    assert "Messages from other tasks" not in quiet_row["text"] and "query_data" not in quiet_row["text"]
    assert quiet_row["metadata"]["continuation"]["peer_context"] == []


@pytest.mark.parametrize("stored,carried", [
    pytest.param({"model_role": "consciousness"}, "consciousness", id="role-carried"),
    pytest.param({"model_role": "consciousness", "model": "claudexor::codex=gpt-6-astra"}, "consciousness",
                 id="role-carried-route-dropped"),
    pytest.param({"model": "claudexor::codex=gpt-6-astra"}, "", id="route-alone-not-a-choice"),
    pytest.param({}, "", id="install-default-stays-default"),
])
def test_continue_carries_the_stored_model_role_never_the_resolved_route(tmp_path, monkeypatch, stored, carried):
    """The successor keeps the predecessor's explicit per-task model choice the way it
    keeps effort: the stored role (a slot name the dispatcher resolves at ITS start),
    never the model string the predecessor happened to run on; a task on the install
    default carries nothing and follows the current default."""
    from ouroboros.agent_dispatch import model_role_slot_override
    from ouroboros.model_slots import MODEL_ROLE_SETTINGS
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, metadata=stored, reasoning_effort="high")
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"] is True, ack
    successor = ack["successor_task_id"]
    row = next(task for task in workers.PENDING if task["id"] == successor)
    assert row["reasoning_effort"] == "high"
    assert row["metadata"].get("model_role", "") == carried
    assert "model" not in row and "model" not in row["metadata"], "a resolved route is never copied"
    assert load_task_result(tmp_path, successor)["metadata"].get("model_role", "") == carried
    replay = admit_continuation("pred-1", action_nonce=NONCE)
    assert replay["replay"] is True
    assert load_task_result(tmp_path, successor)["metadata"].get("model_role", "") == carried
    # The dispatcher resolves the carried role against the CURRENT slot setting at start.
    slot = {MODEL_ROLE_SETTINGS["consciousness"]: "provider/current-slot-model"}
    monkeypatch.setattr("ouroboros.route_choice.runtime_setting", lambda key, default="": slot.get(key, default))
    assert model_role_slot_override(row["metadata"]) == (("provider/current-slot-model", False) if carried else None)


@pytest.mark.parametrize("evidence,expected", [
    ({}, {}),
    ({"baseline": {"surfaces": [{"surface_type": "external_workspace", "canonical_root": "/w",
                                 "captured_at": BASELINE_START, "git": {"base_commit": "e" * 40}}]}}, {}),
    ({"baseline": {"surfaces": [{"surface_type": "system_repo", "canonical_root": SYSTEM_ROOT,
                                 "git": {"base_commit": "b" * 40}}]}}, {}),
    ({"baseline": {"surfaces": [{"surface_type": "system_repo", "canonical_root": SYSTEM_ROOT,
                                 "captured_at": SURFACE_ADVANCED, "git": {"base_commit": "b" * 40}}]},
      "terminal_candidate_snapshot": {"captured_at": TERMINAL_AT, "surfaces": [
          {"surface_type": "system_repo", "canonical_root": "/another/checkout", "base_commit": "c" * 40}]}},
     {"commit": "b" * 40, "captured_at": SURFACE_ADVANCED}),
])
def test_only_a_dated_system_surface_is_a_checkout_observation(evidence, expected):
    from ouroboros.owner_continue import recorded_system_checkout

    assert recorded_system_checkout({"mutation_evidence": evidence}) == expected
