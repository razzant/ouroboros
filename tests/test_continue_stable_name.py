"""A continued work keeps one short subject name; its current focus and past sentences live beside it (#3047, owner 10.11).

Real admission, the real ``update_focus`` handler, the real result store and the work projection's own name walk.
"""
from __future__ import annotations

import pathlib
import queue as queue_module
from types import SimpleNamespace

import pytest

from ouroboros.projects_registry import _load_bindings, _save_bindings, create_project
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.project_journal import _update_focus, may_publish_focus
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_owner_continue import NONCE

SENTENCE = "Partial totals sent; the full check went to a successor."
SUBJECT = "Why the weekly totals differ"
RECENT = {"reader": "recent_tasks"}
BINDING = {"binding_id": "binding-1"}


def _legacy_origin(root, task_id="origin-1", **extra):
    """A root the old terminal write named after its own closing sentence, its Presence binding inherited by work."""
    write_task_result(root, task_id, "running", chat_id=7, project_id="", root_task_id=task_id, delegation_role="root",
                      objective="Look into why totals differ\nand report", origin_message_absent=True,
                      metadata={"presence_binding_authority": dict(BINDING), "origin_message_absent": True,
                                "objective_author": {"kind": "task", "task_id": "presence-turn"}}, **extra)
    write_task_result(root, task_id, "cancelled", wake_summary=SENTENCE, suggested_name=SENTENCE,
                      cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    return load_task_result(root, task_id)


@pytest.fixture
def env(tmp_path, monkeypatch):
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    return SimpleNamespace(root=tmp_path, q=queue, workers=workers)


def _continue(env, predecessor, nonce=NONCE):
    from supervisor.continuation_admission import admit_continuation

    ack = admit_continuation(predecessor, action_nonce=nonce)
    assert ack["ok"] is True, ack
    return next(task for task in env.workers.PENDING if task["id"] == ack["successor_task_id"])


def _ctx(env, task, *, bind=True):
    """The live pooled root a worker would hold: its host metadata, RUNNING with this attempt, bound to a Project."""
    if bind:
        project = create_project(env.root, "report-research", name="Report research")
        bindings = _load_bindings(env.root, fresh=True)
        bindings["bindings"][task["id"]] = {"project_id": "report-research", "project_chat_id": project["chat_id"], "chat_id": project["chat_id"]}
        _save_bindings(env.root, bindings)
    env.workers.PENDING[:] = [row for row in env.workers.PENDING if row["id"] != task["id"]]
    owner = {"kind": "pooled", "task_id": task["id"], "attempt": 1}
    write_task_result(env.root, task["id"], "running", task_attempt=1, execution_owner=owner)
    return SimpleNamespace(task_id=task["id"], drive_root=env.root, project_id="", is_direct_chat=False,
                           task_metadata={**task["metadata"], "root_task_id": task["id"], "delegation_role": "root",
                                          "_execution_owner": owner},
                           event_queue=queue_module.Queue())


def _events(ctx):
    drained = []
    while not ctx.event_queue.empty():
        drained.append(ctx.event_queue.get_nowait())
    return drained


def test_the_whole_flow_keeps_one_name_a_fresh_line_and_the_old_sentence(env):
    from ouroboros.gateway.work_projection import _Observations, _work_of  # the card's own name walk

    before = _legacy_origin(env.root)
    first = _continue(env, "origin-1")
    # A closing sentence is not a name: the Continue starts from the host's subject and asks its author for one.
    assert SENTENCE not in first["title"] and first["metadata"]["continuation"]["work_name"] != SENTENCE
    assert "With your first update_focus, also pass work_name" in first["text"]

    ctx = _ctx(env, first)
    assert may_publish_focus(ctx)
    receipt = _update_focus(ctx, "Reading the latest dialogues", RECENT, work_name=f'"{SUBJECT}."')
    assert receipt.startswith("OK:") and f"Work name set: {SUBJECT!r}" in receipt
    origin = load_task_result(env.root, "origin-1")
    assert origin["title"] == origin["suggested_name"] == SUBJECT
    for field in ("status", "wake_summary", "cancel_origin", "task_attempt", "terminal_time", "ts"):
        assert origin.get(field) == before.get(field), field
    assert origin["wake_summary"] == SENTENCE
    own = load_task_result(env.root, first["id"])
    assert own["suggested_name"] == f"Continue: {SUBJECT}" and own["focus"]["text"] == "Reading the latest dialogues"
    assert own["title"] == first["title"]  # the record of how it was admitted
    named = [(e["task_id"], e["suggested_name"]) for e in _events(ctx) if e["type"] == "task_named"]
    assert named == [("origin-1", SUBJECT), (first["id"], f"Continue: {SUBJECT}")]

    # Reload: the canonical name outranks the admission's stamped copy.
    assert first["metadata"]["continuation"]["work_name"] != SUBJECT
    assert _work_of(first["id"], {first["id"]: own}, _Observations(env.root), {})[1] == SUBJECT

    # It ends; the next Continue is the same work under the same name and is not asked again.
    write_task_result(env.root, first["id"], "cancelled", cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    second = _continue(env, first["id"], nonce="press-0002-abcdef")
    assert second["metadata"]["continuation"]["work_id"] == "origin-1"
    assert second["metadata"]["continuation"]["work_name"] == SUBJECT and second["title"] == f"Continue: {SUBJECT}"
    assert "also pass work_name" not in second["text"]
    next_ctx = _ctx(env, second)
    receipt = _update_focus(next_ctx, "Grouping the causes", RECENT, work_name="A different name")
    assert receipt.startswith("OK:") and "work_name ignored" in receipt and SUBJECT in receipt
    assert load_task_result(env.root, second["id"])["focus"]["text"] == "Grouping the causes"
    assert load_task_result(env.root, "origin-1")["title"] == SUBJECT
    assert load_task_result(env.root, "origin-1")["wake_summary"] == SENTENCE


def test_a_gist_of_several_sentences_leaves_its_first_as_a_name_a_continue_replaces(env):
    """rev_fable REV2 F3: the closing write names a run by its gist's first sentence, but that name was proven the
    run's own sentence only by equality with the whole gist, so a gist of two sentences kept its first sentence as a
    name nobody could replace. The first sentence or the whole gist stands as the run's own sentence; any other name,
    and an author's title, stay."""
    from ouroboros.project_naming import summary_stands_as_name, work_name_open, work_shown_name
    gist = SENTENCE + " March and April match; May waits on the bank's export, my step next."
    _legacy_origin(env.root)
    write_task_result(env.root, "origin-1", "cancelled", wake_summary=gist, suggested_name=SENTENCE,
                      cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    origin = load_task_result(env.root, "origin-1")
    assert summary_stands_as_name(origin) and work_name_open(origin) and work_shown_name(origin) != SENTENCE
    first = _continue(env, "origin-1")
    assert first["metadata"]["continuation"]["work_name"] != SENTENCE and "also pass work_name" in first["text"]
    receipt = _update_focus(_ctx(env, first), "Reading the bank's export", RECENT, work_name=SUBJECT)
    assert f"Work name set: {SUBJECT!r}" in receipt and load_task_result(env.root, "origin-1")["wake_summary"] == gist
    for record in ({"suggested_name": "Mail check", "wake_summary": gist}, {"suggested_name": "March and April match;",
                   "wake_summary": gist}, {"suggested_name": SENTENCE, "wake_summary": gist, "title": "Bank export"}):
        assert not summary_stands_as_name(record), record
    assert summary_stands_as_name({"suggested_name": SENTENCE, "wake_summary": SENTENCE})  # a record from before


def test_a_retried_name_is_the_name_standing(env):
    _legacy_origin(env.root)
    ctx = _ctx(env, _continue(env, "origin-1"))
    assert "Work name set" in _update_focus(ctx, "One", RECENT, work_name=SUBJECT)
    assert "Work name set" in _update_focus(ctx, "Two", RECENT, work_name=SUBJECT)


@pytest.mark.parametrize("names", [
    {"title": "Friday report", "suggested_name": "Weekly numbers"},
    {"suggested_name": "A coined name nobody can prove host-made"},
], ids=["authored", "coined"])
def test_an_explicit_or_ambiguous_name_is_kept_and_the_focus_still_publishes(env, names):
    write_task_result(env.root, "origin-1", "running", chat_id=7, root_task_id="origin-1", origin_message_text="x",
                      origin_message_ref={"chat_id": 7, "client_message_id": "m-1"}, **names)
    write_task_result(env.root, "origin-1", "cancelled", cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    task = _continue(env, "origin-1")
    assert task["metadata"]["continuation"]["work_name"] == names["suggested_name"]
    assert "also pass work_name" not in task["text"]
    ctx = _ctx(env, task, bind=False)
    receipt = _update_focus(ctx, "Going on", RECENT, work_name="Another")
    assert receipt.startswith("OK:") and "already named" in receipt
    stored = load_task_result(env.root, "origin-1")
    assert stored["suggested_name"] == names["suggested_name"] and stored.get("title", "") == names.get("title", "")
    assert load_task_result(env.root, task["id"])["focus"]["text"] == "Going on"


def test_a_derived_request_line_may_be_replaced_by_the_subject(env):
    write_task_result(env.root, "origin-1", "running", chat_id=7, root_task_id="origin-1", description="Look at the logs\nand more",
                      suggested_name="Look at the logs", origin_message_text="x", origin_message_ref={"chat_id": 7, "client_message_id": "m-1"})
    write_task_result(env.root, "origin-1", "cancelled", cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    task = _continue(env, "origin-1")
    assert "also pass work_name" in task["text"]
    assert "Work name set" in _update_focus(_ctx(env, task, bind=False), "Going on", RECENT, work_name=SUBJECT)
    assert load_task_result(env.root, "origin-1")["title"] == SUBJECT


def test_a_missing_first_record_stays_an_honest_fallback(env):
    from tests._store_shared import delete_result

    _legacy_origin(env.root)
    task = _continue(env, "origin-1")
    ctx = _ctx(env, task)
    assert delete_result(env.root, "origin-1")
    receipt = _update_focus(ctx, "Going on", RECENT, work_name=SUBJECT)
    assert receipt.startswith("OK:") and "work_name ignored" in receipt and "missing" in receipt
    assert load_task_result(env.root, "origin-1") in (None, {})
    assert load_task_result(env.root, task["id"])["suggested_name"] == task["suggested_name"]


def test_the_target_is_the_hosts_own_lineage(env):
    """The tool has no target argument; a root that is no Continue names nothing and still publishes."""
    write_task_result(env.root, "plain", "running", root_task_id="plain", task_attempt=1)
    ctx = SimpleNamespace(task_id="plain", drive_root=env.root, project_id="", is_direct_chat=False,
                          task_metadata={"root_task_id": "plain", "delegation_role": "root"})
    receipt = _update_focus(ctx, "Plain", RECENT, work_name=SUBJECT)
    assert receipt.startswith("OK:") and "only a Continue" in receipt
    assert not load_task_result(env.root, "plain").get("title")


@pytest.mark.parametrize("change", ["no_project", "child", "speaker", "no_admission", "no_work_id", "forged_author"])
def test_only_a_project_bound_admitted_continue_extends_focus(env, change):
    _legacy_origin(env.root)
    task = _continue(env, "origin-1")
    ctx = _ctx(env, task, bind=change != "no_project")
    continuation = ctx.task_metadata["continuation"]
    if change == "child":
        ctx.task_metadata.update(parent_task_id="p", delegation_role="subagent")
    elif change == "speaker":
        ctx.task_metadata["presence"] = dict(BINDING)
    elif change == "no_admission":
        ctx.task_metadata["continuation"] = {k: v for k, v in continuation.items() if k != "admission"}
    elif change == "no_work_id":
        ctx.task_metadata["continuation"] = {k: v for k, v in continuation.items() if k != "work_id"}
    else:
        ctx.task_metadata["objective_author"] = {"kind": "owner"}
    assert not may_publish_focus(ctx)
    assert "TOOL_FORBIDDEN" in _update_focus(ctx, "Not mine", RECENT, work_name=SUBJECT)
    assert load_task_result(env.root, task["id"]).get("focus") is None
    assert not load_task_result(env.root, "origin-1").get("title")


def test_a_clean_finish_wake_keeps_its_sentence_as_its_name(env):
    """Nothing continued it, so nothing about its card changes."""
    from ouroboros.gateway.work_projection import _Observations, _work_of

    wake = {"task_id": "wake-1", "status": "completed", "suggested_name": SENTENCE, "wake_summary": SENTENCE}
    write_task_result(env.root, "wake-1", "completed", suggested_name=SENTENCE, wake_summary=SENTENCE)
    assert _work_of("wake-1", {"wake-1": wake}, _Observations(env.root), {}) == ("wake-1", SENTENCE)
    assert load_task_result(env.root, "wake-1").get("title") in (None, "")


def test_the_tool_schema_keeps_the_name_optional_and_the_handler_takes_both_shapes(env):
    from ouroboros.tools.project_journal import get_tools

    entry = next(tool for tool in get_tools() if tool.name == "update_focus")
    parameters = entry.schema["parameters"]
    assert "work_name" in parameters["properties"] and parameters["required"] == ["text", "source_ref"]
    _legacy_origin(env.root)
    ctx = _ctx(env, _continue(env, "origin-1"))
    assert entry.handler(ctx, text="One", source_ref=RECENT).startswith("OK:")
    assert "Work name set" in entry.handler(ctx, text="Two", source_ref=RECENT, work_name=SUBJECT)


def _at_the_write(monkeypatch, action):
    """Run ``action`` once just as the guarded focus write starts: the handler's earlier liveness checks already passed."""
    import ouroboros.task_results as results

    real, fired = results.write_task_result, []

    def write(root, task_id, status, **fields):
        if "focus" in fields and not fired:
            fired.append(True)
            action(real)
        return real(root, task_id, status, **fields)

    monkeypatch.setattr(results, "write_task_result", write)
    return fired


@pytest.mark.parametrize("change", ["finalized", "new_attempt"])
def test_a_late_handler_of_a_dead_attempt_leaves_the_name_open(env, monkeypatch, change):
    _legacy_origin(env.root)
    ctx = _ctx(env, _continue(env, "origin-1"))
    successor = ctx.task_id
    before = load_task_result(env.root, "origin-1")
    actions = {
        "finalized": lambda real: real(env.root, successor, "completed", result="ended meanwhile"),
        "new_attempt": lambda real: real(env.root, successor, "running", task_attempt=2,
                                         execution_owner={"kind": "pooled", "task_id": successor, "attempt": 2}),
    }
    fired = _at_the_write(monkeypatch, actions[change])
    receipt = _update_focus(ctx, "Too late", RECENT, work_name=SUBJECT)
    assert fired and receipt.startswith("⚠️ FOCUS_TASK_NOT_LIVE"), receipt
    assert load_task_result(env.root, "origin-1") == before  # not one byte of the first record moved
    own = load_task_result(env.root, successor)
    assert own.get("focus") is None and not own["suggested_name"].endswith(SUBJECT)
    assert not [event for event in _events(ctx) if event["type"] == "task_named"]


def test_a_stale_focus_claims_no_name(env):
    _legacy_origin(env.root)
    ctx = _ctx(env, _continue(env, "origin-1"))
    before = load_task_result(env.root, "origin-1")
    assert _update_focus(ctx, "Current line", RECENT).startswith("OK:")
    stored = load_task_result(env.root, ctx.task_id)["focus"]
    write_task_result(env.root, ctx.task_id, "running", focus={**stored, "authored_at": "2999-01-01T00:00:00+00:00"})
    receipt = _update_focus(ctx, "Older line", RECENT, work_name=SUBJECT)
    assert receipt.startswith("⚠️ FOCUS_STALE"), receipt
    assert load_task_result(env.root, "origin-1") == before


def test_the_claim_and_the_focus_are_one_transaction(env, monkeypatch):
    from ouroboros import store

    _legacy_origin(env.root)
    ctx = _ctx(env, _continue(env, "origin-1"))
    before = load_task_result(env.root, "origin-1")
    real = store.Txn.store_task

    def refuse_the_focus_row(self, document, **kwargs):
        if document["task_id"] == ctx.task_id:
            raise OSError("disk refused the focus row")
        return real(self, document, **kwargs)

    monkeypatch.setattr(store.Txn, "store_task", refuse_the_focus_row)
    receipt = _update_focus(ctx, "Going on", RECENT, work_name=SUBJECT)
    assert receipt.startswith("⚠️ TOOL_ERROR"), receipt
    assert load_task_result(env.root, "origin-1") == before  # the origin's write rolled back with it
    assert load_task_result(env.root, ctx.task_id).get("focus") is None
    monkeypatch.setattr(store.Txn, "store_task", real)
    assert "Work name set" in _update_focus(ctx, "Going on", RECENT, work_name=SUBJECT)  # a valid attempt keeps access
    assert load_task_result(env.root, "origin-1")["title"] == SUBJECT


def test_a_valid_current_attempt_names_with_every_old_fact_kept(env):
    _legacy_origin(env.root)
    ctx = _ctx(env, _continue(env, "origin-1"))
    before = load_task_result(env.root, "origin-1")
    assert "Work name set" in _update_focus(ctx, "Going on", RECENT, work_name=SUBJECT)
    after = load_task_result(env.root, "origin-1")
    changed = {key for key in {*before, *after} if before.get(key) != after.get(key)}
    assert changed == {"title", "suggested_name", "updated_at"}, changed


def _late_namer(env, monkeypatch, task_id, before_write):
    """A turn namer whose model call settled, paused at its row read while ``before_write`` lands."""
    import threading

    import ouroboros.task_results as results
    from ouroboros import project_naming

    waiting, release, settled, broadcasts = threading.Event(), threading.Event(), threading.Event(), []
    original = results.load_task_result

    def wait_first(*args, **kwargs):
        if threading.current_thread().name == f"namer-{task_id}":
            waiting.set()
            assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(results, "load_task_result", wait_first)
    monkeypatch.setattr(project_naming, "llm_project_name", lambda *a, **k: "A different delayed title")
    monkeypatch.setattr(project_naming, "_refresh_root_cost_after_naming", lambda *a: settled.set())
    project_naming.spawn_turn_namer(env.root, task_id, "Look into the weekly totals", broadcast=broadcasts.append)
    assert waiting.wait(5)
    before_write()
    release.set()
    assert settled.wait(5)
    return broadcasts


def test_a_late_turn_namer_leaves_an_adopted_name_and_says_nothing(env, monkeypatch):
    from ouroboros.gateway.work_projection import _Observations, _work_of
    from tests.test_continue_author_stop import GROUP, STOP_AXES

    origin = "origin-1"
    write_task_result(env.root, origin, "running", chat_id=7, root_task_id=origin, delegation_role="root",
                      objective="Look into the weekly totals", _is_direct_chat=True, origin_message_text="Look into the weekly totals",
                      origin_message_ref={"chat_id": 7, "client_message_id": "m-1"},
                      billing_group={**GROUP, "billing_group_id": origin})
    made = {}

    def continue_and_adopt():
        write_task_result(env.root, origin, "completed", outcome_axes=STOP_AXES)
        made["task"] = _continue(env, origin)
        assert "Work name set" in _update_focus(_ctx(env, made["task"], bind=False), "Investigating", RECENT, work_name=SUBJECT)

    broadcasts = _late_namer(env, monkeypatch, origin, continue_and_adopt)
    assert broadcasts == []
    assert load_task_result(env.root, origin)["suggested_name"] == SUBJECT
    own = load_task_result(env.root, made["task"]["id"])
    assert _work_of(made["task"]["id"], {made["task"]["id"]: own}, _Observations(env.root), {})[1] == SUBJECT


def test_a_late_turn_namer_still_names_a_turn_nobody_adopted(env, monkeypatch):
    write_task_result(env.root, "plain-1", "running", chat_id=7, root_task_id="plain-1", delegation_role="root", title="Owner's title")
    broadcasts = _late_namer(env, monkeypatch, "plain-1", lambda: None)
    assert load_task_result(env.root, "plain-1")["suggested_name"] == "A different delayed title"
    assert broadcasts == [{"type": "task_named", "task_id": "plain-1", "suggested_name": "A different delayed title"}]


# A split root's execution replica (its child drive) predates the canonical name: the shared replica reducer
# serves both the effective read (``load_effective_task_result``, the card's ``TaskReads``) and the terminal
# copyback (``copy_child_task_result``), and a titled canonical record owns its name slots there (#3047).
def _replica(env, task_id, label):
    """The child drive a split root's worker writes: a copy of the canonical row as admitted, before any naming."""
    from copy import deepcopy

    from tests._store_shared import seed_result

    child = env.root / f"child-{label}"
    seed_result(child, task_id, deepcopy(load_task_result(env.root, task_id)))
    write_task_result(env.root, task_id, load_task_result(env.root, task_id)["status"], child_drive_root=str(child))
    return child


def _effective(env, task_id):
    from ouroboros.task_status import load_effective_task_result

    return load_effective_task_result(env.root, task_id, materialize_artifacts=False)


def test_an_adopted_work_name_outranks_its_origins_execution_replica(env):
    from ouroboros.gateway.work_projection import _Observations, _work_of
    from ouroboros.headless import copy_child_task_result
    from tests._store_shared import seed_result

    prior = _legacy_origin(env.root)  # it ran out of context rather than being stopped: its replica is its outcome
    prior = {key: value for key, value in prior.items() if key != "cancel_origin"}
    seed_result(env.root, "origin-1", {**prior, "status": "failed", "reason_code": "context_capacity"})
    child = _replica(env, "origin-1", "origin")
    seed_result(child, "origin-1", {**load_task_result(child, "origin-1"), "title": ""})  # the worker's empty slot
    task = _continue(env, "origin-1")
    assert "Work name set" in _update_focus(_ctx(env, task), "Reading the latest dialogues", RECENT, work_name=SUBJECT)
    assert load_task_result(child, "origin-1")["suggested_name"] == SENTENCE  # the replica still has the old name
    effective = _effective(env, "origin-1")
    assert (effective["title"], effective["suggested_name"]) == (SUBJECT, SUBJECT)
    assert effective["wake_summary"] == SENTENCE  # the closing sentence stays the episode's history
    assert _work_of("origin-1", {"origin-1": effective}, _Observations(env.root), {})[1] == SUBJECT

    copy_child_task_result(env.root, {"id": "origin-1", "drive_root": str(child)})
    copied = load_task_result(env.root, "origin-1")
    assert (copied["title"], copied["suggested_name"], copied["wake_summary"]) == (SUBJECT, SUBJECT, SENTENCE)


def test_a_successors_claimed_name_outranks_its_replica_on_read_and_copyback(env):
    from ouroboros.headless import copy_child_task_result

    _legacy_origin(env.root)
    task = _continue(env, "origin-1")
    ctx = _ctx(env, task)
    child = _replica(env, task["id"], "successor")
    admitted = load_task_result(child, task["id"])["suggested_name"]
    assert admitted == task["suggested_name"] != f"Continue: {SUBJECT}"
    assert "Work name set" in _update_focus(ctx, "Reading the latest dialogues", RECENT, work_name=SUBJECT)
    assert _effective(env, task["id"])["suggested_name"] == f"Continue: {SUBJECT}"

    write_task_result(child, task["id"], "completed", result="done")  # the worker finishes on its own drive
    copy_child_task_result(env.root, {**task, "drive_root": str(child)})
    copied = load_task_result(env.root, task["id"])
    assert copied["status"] == "completed" and copied["result"] == "done"  # the replica still owns its outcome
    assert copied["suggested_name"] == f"Continue: {SUBJECT}" and copied["title"] == task["title"]
    assert _effective(env, task["id"])["suggested_name"] == f"Continue: {SUBJECT}"


@pytest.mark.parametrize("consumer", ["read", "copyback"])
def test_an_untitled_run_still_takes_its_executions_closing_name(env, consumer):
    """No author named it, so the closing sentence its worker wrote is its name, over a canonical stand-in."""
    from ouroboros.headless import copy_child_task_result

    write_task_result(env.root, "wake-1", "running", root_task_id="wake-1", suggested_name="Wake: scheduled check")
    child = _replica(env, "wake-1", "wake")
    write_task_result(child, "wake-1", "completed", result="ok", wake_summary=SENTENCE, suggested_name=SENTENCE)
    if consumer == "copyback":
        copy_child_task_result(env.root, {"id": "wake-1", "drive_root": str(child)})
        stored = load_task_result(env.root, "wake-1")
    else:
        stored = _effective(env, "wake-1")
    assert stored["suggested_name"] == SENTENCE and not stored.get("title")


@pytest.mark.parametrize("consumer", ["read", "copyback"])
def test_an_explicit_title_stays_whatever_its_replica_carries(env, consumer):
    from ouroboros.headless import copy_child_task_result

    write_task_result(env.root, "authored-1", "running", root_task_id="authored-1",
                      title="Friday report", suggested_name="Friday report")
    child = _replica(env, "authored-1", "authored")
    write_task_result(child, "authored-1", "completed", result="ok", title="", suggested_name="Some other words")
    if consumer == "copyback":
        copy_child_task_result(env.root, {"id": "authored-1", "drive_root": str(child)})
        stored = load_task_result(env.root, "authored-1")
    else:
        stored = _effective(env, "authored-1")
    assert (stored["title"], stored["suggested_name"]) == ("Friday report", "Friday report")
    assert stored["status"] == "completed"


# Neutral names for the split-drive fixtures below.
WEEKLY = "Weekly report"
TOTALS = "Totals sent; the rest went to a successor."


def _split_continue(env):
    """A Continue admitted onto its own drive (its origin's split mode): the queue row names its execution replica
    (``drive_root``) beside the canonical record's root; the replica is the canonical row as its worker began."""
    from copy import deepcopy

    from ouroboros.headless import task_state_dir
    from tests._store_shared import seed_result

    write_task_result(env.root, "origin-1", "running", chat_id=7, project_id="", root_task_id="origin-1",
                      delegation_role="root", objective="Prepare the weekly totals\nand send them",
                      origin_message_absent=True, memory_mode="empty",
                      child_drive_root=str(task_state_dir(env.root, "origin-1") / "data"),
                      metadata={"presence_binding_authority": dict(BINDING), "origin_message_absent": True,
                                "objective_author": {"kind": "task", "task_id": "presence-turn"}})
    write_task_result(env.root, "origin-1", "cancelled", wake_summary=TOTALS, suggested_name=TOTALS,
                      cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    task = _continue(env, "origin-1")
    child, canonical = pathlib.Path(task["drive_root"]), pathlib.Path(task["budget_drive_root"])
    assert child != canonical == env.root
    ctx = _ctx(env, task)
    seed_result(child, task["id"], deepcopy(load_task_result(env.root, task["id"])))
    env.workers.RUNNING[task["id"]] = {"task": task}
    return task, ctx, child


def _live(env):
    """The server's own consumer of a carrier's ``task_named`` frames, over the live queue maps."""
    from supervisor.conversation_lane import handle_task_named

    broadcasts = []
    consumer = SimpleNamespace(RUNNING=env.workers.RUNNING, PENDING=env.workers.PENDING,
                               bridge=SimpleNamespace(broadcast=broadcasts.append))
    return (lambda event: handle_task_named(event, consumer)), broadcasts


def test_a_split_continues_claimed_name_reaches_its_live_card_history_and_copyback(env):
    """#3047: the live consumer reads the record owning the name, as the effective read and the copyback do: the real
    ``update_focus`` frames of a split Continue whose replica still carries its admission name."""
    from ouroboros.gateway.work_projection import _Observations, _work_of
    from ouroboros.headless import copy_child_task_result

    task, ctx, child = _split_continue(env)
    admitted, claimed = task["suggested_name"], f"Continue: {WEEKLY}"
    handle, broadcasts = _live(env)
    assert "Work name set" in _update_focus(ctx, "Checking this week's totals", RECENT, work_name=WEEKLY)
    for event in _events(ctx):
        if event["type"] == "task_named":
            handle(event)
    assert load_task_result(child, task["id"])["suggested_name"] == admitted != claimed  # the replica predates it
    assert [(row["task_id"], row["suggested_name"]) for row in broadcasts] == [("origin-1", WEEKLY), (task["id"], claimed)]
    assert env.workers.RUNNING[task["id"]]["task"]["suggested_name"] == claimed  # the queue row follows by value
    own = _effective(env, task["id"])
    assert own["suggested_name"] == claimed
    assert _work_of(task["id"], {task["id"]: own}, _Observations(env.root), {})[1] == WEEKLY
    write_task_result(child, task["id"], "completed", result="done")  # the worker finishes on its own drive
    copy_child_task_result(env.root, {**task, "drive_root": str(child)})
    assert load_task_result(env.root, task["id"])["suggested_name"] == claimed
    handle({"type": "task_named", "task_id": task["id"], "suggested_name": admitted})  # an older frame, late
    assert broadcasts[-1]["suggested_name"] == claimed == task["suggested_name"]


def test_an_untitled_split_run_still_takes_its_executions_name_live(env):
    """No author titled it, so its worker's own name on its drive stands live, as on read and copyback."""
    from tests._store_shared import seed_result

    child = env.root / "child-wake"
    write_task_result(env.root, "wake-1", "running", root_task_id="wake-1", suggested_name="Wake: scheduled check",
                      child_drive_root=str(child))
    seed_result(child, "wake-1", {**load_task_result(env.root, "wake-1"), "suggested_name": TOTALS})
    row = {"id": "wake-1", "suggested_name": "Wake: scheduled check", "drive_root": str(child),
           "budget_drive_root": str(env.root)}
    env.workers.RUNNING["wake-1"] = {"task": row}
    handle, broadcasts = _live(env)
    handle({"type": "task_named", "task_id": "wake-1", "suggested_name": TOTALS})
    assert row["suggested_name"] == TOTALS and broadcasts == [
        {"type": "task_named", "task_id": "wake-1", "suggested_name": TOTALS}]


def test_a_split_rows_busy_canonical_record_leaves_its_name_standing(env, monkeypatch):
    """The live handler never waits on a busy store and never takes the older replica in place of the canonical
    record it could not read: the row keeps its own name until a frame reads that record."""
    import ouroboros.task_results as task_results
    from ouroboros.store import StoreBusy, bounded_wait
    from supervisor.queue import _queue_lock

    task, ctx, child = _split_continue(env)
    admitted, claimed = task["suggested_name"], f"Continue: {WEEKLY}"
    assert "Work name set" in _update_focus(ctx, "Checking this week's totals", RECENT, work_name=WEEKLY)
    handle, broadcasts = _live(env)
    read, reads = task_results.load_task_result, []

    def busy(root, task_id, **options):
        if task_id == task["id"]:
            reads.append((pathlib.Path(root), _queue_lock._is_owned(), bounded_wait(10.0)))
            if pathlib.Path(root) == env.root:
                raise StoreBusy("database is locked", reason="contention")
        return read(root, task_id, **options)

    with monkeypatch.context() as patched:
        patched.setattr(task_results, "load_task_result", busy)
        handle({"type": "task_named", "task_id": task["id"], "suggested_name": claimed})
    assert reads == [(env.root, False, 0.0)]  # off Q, zero wait, and the replica left unread
    assert task["suggested_name"] == admitted and broadcasts[-1]["suggested_name"] == admitted
    handle({"type": "task_named", "task_id": task["id"], "suggested_name": claimed})
    assert task["suggested_name"] == claimed and broadcasts[-1]["suggested_name"] == claimed


def test_the_reducer_leaves_a_slot_the_canonical_record_lacks_to_its_replica():
    from ouroboros.post_task_checkpoint import project_replica_task_result_fields

    overlay = project_replica_task_result_fields({"title": "Friday report"}, {"title": "", "suggested_name": "Coined"})
    assert "title" not in overlay and overlay["suggested_name"] == "Coined"
    assert project_replica_task_result_fields({"title": " "}, {"title": "T", "suggested_name": "S"}) == {
        "title": "T", "suggested_name": "S"}


def test_the_card_strips_the_one_continue_prefix_the_host_writes():
    """One Python stripper serves the name's owner and the works list; the episode card strips the same prefix in
    JavaScript, its pattern pinned to the Python constant."""
    import pathlib
    import re

    from ouroboros.gateway import work_projection
    from ouroboros.project_naming import CONTINUE_TITLE_PREFIX, without_continue

    assert without_continue(f"{CONTINUE_TITLE_PREFIX}{CONTINUE_TITLE_PREFIX}Totals") == "Totals"
    assert work_projection._first_name({"suggested_name": f" {CONTINUE_TITLE_PREFIX}Totals "}) == "Totals"
    source = (pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "task_episodes.js").read_text("utf-8")
    pattern = re.search(r"text\.replace\(/\^\(\?:(.+?)\)\+/", source).group(1)
    assert pattern == f"{CONTINUE_TITLE_PREFIX.rstrip()} *"
