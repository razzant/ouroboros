"""#2482: new work whose recorded source is an old occurrence that already ran under the place's carrier.

The follow-up row names that occurrence; its host receipt (or a Continue's kept schedule basis) names the origin it
retained; that origin decides as a producer would, or, being such a run itself, names the next. One keyed read per
generation, through the real scheduler tick, recovery paths and Continue admission. What is running keeps what it holds,
and an unproved, foreign, child, speaker or ceilinged record stops the proof wherever it stands, as does a repeated id.
"""

from __future__ import annotations

import contextlib
import copy

import pytest

from ouroboros.dialogue_provenance import presence_caller_binding, presence_task_binding
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.registry import ToolContext
from tests.test_presence_own_work import BINDING, OTHER, _ceiling, _presence, _registry, presence_ceiling_payload
from tests.test_presence_related_owner_work import (
    AUTHORITY,
    _accepted_before_the_relation,
    _interrupted_after_it_ran,
    _old_followup_row,
    _owner_row,
    _pooled,
    _read_back,
    _worker_reads_the_owners_table,
    _write_origin,
)
from tests.test_schedule_occurrence import q as _q

q = _q  # the real scheduler's queue fixture, requested by name; re-bound so a parameter named q is no redefinition


def _old_run(q, upstream="old-root"):
    """An old occurrence that already ran: a real accepted receipt naming ``upstream``, its record under the old carrier."""
    _write_origin(q.root, upstream)
    return _interrupted_after_it_ran(q, _accepted_before_the_relation(q, upstream))


def _generations(q, count, upstream="old-root"):
    """``count`` old occurrences that already ran, oldest first: the first names ``upstream``, each later one the one before."""
    chain = [_old_run(q, upstream)]
    for generation in range(1, count):
        chain.append(_interrupted_after_it_ran(q, _accepted_before_the_relation(q, chain[-1], f"g{generation}")))
    return chain


@contextlib.contextmanager
def _record_reads(q, *, busy=""):
    """Every record the proof loads, in order and with whether Q was held. Loading ``busy`` is contended. The bound only
    ends a walk that would never end by itself: an exception here stops it, and the extra reads then fail the test."""
    from ouroboros import task_status
    from ouroboros.store import StoreBusy

    reads, real_read = [], task_status.load_effective_task_result

    def spied(root, task_id, **kw):
        reads.append((task_id, q.queue._queue_lock._is_owned()))
        if task_id == busy or len(reads) > 1000:
            raise StoreBusy("record busy", reason="contention")
        return real_read(root, task_id, **kw)

    with pytest.MonkeyPatch.context() as spy:
        spy.setattr(task_status, "load_effective_task_result", spied)
        yield reads


def _assert_one_walk_each(reads, walk):
    """Every decision read exactly ``walk``, in that order, off Q: nothing beyond the record that ended it, nothing twice."""
    assert reads and not any(held for _task_id, held in reads), reads
    assert [task_id for task_id, _held in reads] == walk * (len(reads) // len(walk)), reads


def _assert_consumers(q, task, converted):
    """What the row's own worker and its place then see, through the real tools."""
    assert _worker_reads_the_owners_table(q, task) is converted
    place, _place = _registry(q.root)
    stranger, _stranger = _registry(q.root, binding=OTHER)
    assert "PRESENCE_CAPABILITY_BLOCKED" not in place.execute("get_task_result", {"task_id": task["id"]})
    assert "PRESENCE_CAPABILITY_BLOCKED" in stranger.execute("get_task_result", {"task_id": task["id"]})


def _corrupt_origin(q, task_id, names):
    """The host receipt of ``task_id``, otherwise intact, names ``names`` as its origin (a hand-corrupted record)."""
    receipt = copy.deepcopy(load_task_result(q.root, task_id)["schedule_admission"])
    receipt["task"]["metadata"]["schedule_basis"]["origin"]["task_id"] = names
    write_task_result(q.root, task_id, "failed", schedule_admission=receipt)


def _assert_carrier(task, converted):
    assert ("presence_binding_relation" in task["metadata"]) is converted
    assert ("presence_binding_authority" in task["metadata"]) is not converted
    assert presence_task_binding(task) == (None if converted else BINDING)


def _occurrence_of_a_follow_up_of(q, source, schedule_id="s2"):
    """The new occurrence of a saved old follow-up that names ``source``, through the real scheduler tick."""
    _old_followup_row(q, source, schedule_id)
    q.queue.check_scheduled_tasks()
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    return task


@pytest.mark.parametrize("upstream,converted", [
    ("old-root", True),  # the door's stamp beside the old carrier, on a root
    ("owner-turn", True),  # the answering owner turn, by its accepted late answer
    ("acting-root", False),  # the place's own delegated root: no owner door anywhere
    ("missing", False),  # the retained origin cannot be read: unproved
    ("other-binding", False),  # the owner's related work of another place
    ("child-origin", False),  # a delegated child: the door's stamp beside it relates nothing
    ("origin-ceiling", False),  # a run under a ceiling hands on no relation
])
def test_a_new_occurrence_reads_its_old_source_and_the_origin_that_source_retained(q, upstream, converted):
    _owner_row(q)
    source = _old_run(q, upstream)

    task = _occurrence_of_a_follow_up_of(q, source)

    _assert_carrier(task, converted)
    _assert_consumers(q, task, converted)
    ran = load_task_result(q.root, source)["metadata"]
    assert "presence_binding_authority" in ran and "presence_binding_relation" not in ran  # what ran stays as recorded


@pytest.mark.parametrize("change", ["foreign-receipt", "ceiling", "parent", "speaker", "foreign-binding"])
def test_the_old_source_must_itself_be_a_root_of_this_binding_with_a_host_receipt(q, change):
    """A carrier on the source is not a guess to read past: its own receipt, role, ceiling and binding decide."""
    source = _old_run(q)
    receipt = copy.deepcopy(load_task_result(q.root, source)["schedule_admission"])
    fields = {
        "foreign-receipt": {"schedule_admission": {**receipt, "token": "a-foreign-token"}},
        "ceiling": {"task_contract": {"capability_ceiling": presence_ceiling_payload(_ceiling())}},
        "parent": {"parent_task_id": "a-parent"},
        "speaker": {"metadata": {"presence": _presence(), **AUTHORITY}},  # a speaker stays one, whatever else it holds
        "foreign-binding": {"metadata": {"presence_binding_authority": {"binding_id": OTHER}}},
    }[change]
    write_task_result(q.root, source, "failed", **fields)

    _assert_carrier(_occurrence_of_a_follow_up_of(q, source), False)


@pytest.mark.parametrize("generations", [1, 3])
@pytest.mark.parametrize("path", ["republished", "restored"])
def test_an_accepted_unstarted_occurrence_of_such_a_source_converts_at_its_recovery(q, path, generations):
    """Accepted before the upgrade, never started: the same existing recovery publications, the same proof, read off Q."""
    _owner_row(q)
    chain = _generations(q, generations)
    before = _accepted_before_the_relation(q, chain[-1], "s2")
    if path == "republished":
        q.pending.clear()
    else:
        assert q.queue.persist_queue_snapshot(reason="test_saved_before_upgrade") is True
        q.pending.clear()
        assert q.queue.restore_pending_from_snapshot() == 1
    with _record_reads(q) as reads:
        q.queue.check_scheduled_tasks()

    _assert_one_walk_each(reads, [*reversed(chain), "old-root"])
    live = _read_back(q, before["task"]["id"], before, converted=True)
    assert _worker_reads_the_owners_table(q, live) is True


def _still_running(q, before):
    """An accepted old occurrence that reached a worker and keeps running: its record says so, its row is the worker's."""
    task_id = before["task"]["id"]
    q.pending.clear()
    write_task_result(q.root, task_id, "running", schedule_admission={**before, "dispatch": "possible"})
    return ToolContext(repo_dir=q.root, drive_root=q.root, budget_drive_root=q.root, task_id=task_id, current_chat_id=1,
                       task_metadata=_pooled(before["task"]), task_contract=before["task"].get("task_contract") or {})


@pytest.mark.parametrize("upstream,converted", [("old-root", True), ("acting-root", False)])
def test_a_follow_up_the_still_narrowed_run_emits_reads_its_source_and_the_run_keeps_what_it_holds(q, upstream, converted):
    from ouroboros.tools.followup import _handle_schedule_followup
    from tests.test_schedule_occurrence import _rows

    _owner_row(q)
    chain = _generations(q, 2, upstream)  # the proof for the run's follow-up then reads the run, these two and the upstream
    run = _still_running(q, _accepted_before_the_relation(q, chain[-1], "s-run"))
    holds = presence_caller_binding(run), _worker_reads_the_owners_table(q, {"id": run.task_id, "metadata": run.task_metadata})
    assert holds == (BINDING, False)

    written = _handle_schedule_followup(run, objective="Recheck the source", run_at="2000-01-01T00:00:00Z", relation="independent",
                                        destination="place")

    assert written.startswith("FOLLOWUP_SCHEDULED"), written
    [row] = [item for item in _rows(q).values() if (item.get("followup_origin") or {}).get("task_id") == run.task_id]
    assert row["task"]["metadata"]["presence_binding_authority"] == AUTHORITY["presence_binding_authority"]  # the run's writer
    q.queue.check_scheduled_tasks()
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    _assert_carrier(task, converted)
    assert _worker_reads_the_owners_table(q, task) is converted
    assert (presence_caller_binding(run), _worker_reads_the_owners_table(q, {"id": run.task_id, "metadata": run.task_metadata})) == holds


def test_a_root_the_still_narrowed_run_starts_keeps_that_runs_carrier_and_proves_nothing_further(q, monkeypatch):
    """The named boundary: start_task is the run's own held authority acting now, not a recorded source. The root keeps
    the carrier and the contract, has no host schedule fact of its own, and a follow-up it emits stays narrowed."""
    from ouroboros.schedule_continuation import recorded_origin
    from ouroboros.tools.control_routing import _start_task

    _write_origin(q.root, "old-root")
    run = _still_running(q, _accepted_before_the_relation(q, "old-root"))
    started = []
    monkeypatch.setattr("ouroboros.tools.control_routing._promotion_pool_disabled_from_snapshot", lambda _ctx: "")
    monkeypatch.setattr("ouroboros.tools.control_routing._emit_and_wait_for_routing",
                        lambda _ctx, evt: (started.append(evt) or "live", {"status": "scheduled"}))
    assert _start_task(run, "Follow the lead", title="Lead", destination="place").startswith("OK: task")
    assert started[0]["presence_binding_authority"] == AUTHORITY["presence_binding_authority"]
    assert "presence_binding_relation" not in started[0] and "task_contract" in started[0]

    write_task_result(q.root, "started-root", "running", chat_id=1, delegation_role="root", root_task_id="started-root",
                      metadata=dict(AUTHORITY))
    assert recorded_origin(load_task_result(q.root, "started-root")) == ""
    _assert_carrier(_occurrence_of_a_follow_up_of(q, "started-root"), False)


@pytest.mark.parametrize("generations", [1, 3])
@pytest.mark.parametrize("upstream,converted", [("old-root", True), ("acting-root", False)])
def test_a_new_continue_of_an_occurrence_of_such_a_source_reads_the_same_records(q, upstream, converted, generations):
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE

    chain = _generations(q, generations, upstream)
    occurrence = _interrupted_after_it_ran(q, _accepted_before_the_relation(q, chain[-1], "s2"))

    ack = admit_continuation(occurrence, action_nonce=NONCE)

    assert ack["ok"], ack
    [row] = [task for task in q.pending if task["id"] == ack["successor_task_id"]]
    _assert_carrier(row, converted)


@pytest.mark.parametrize("kept_basis_valid", [True, False], ids=["kept-basis", "malformed-kept-basis"])
def test_a_narrowed_continue_as_the_source_proves_by_the_schedule_basis_its_host_admission_kept(q, kept_basis_valid):
    import supervisor.schedule_occurrence as occurrences
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE

    source = _old_run(q)
    with pytest.MonkeyPatch.context() as previous:
        previous.setattr(occurrences, "_origin_relation", lambda *_a, **_k: {})
        ack = admit_continuation(source, action_nonce=NONCE)
    assert ack["ok"], ack
    [narrowed] = list(q.pending)
    q.pending.clear()
    stored = load_task_result(q.root, narrowed["id"])
    assert "presence_binding_authority" in stored["metadata"]
    if not kept_basis_valid:  # the same check Continue's owner sources apply to this input
        metadata = copy.deepcopy(stored["metadata"])
        metadata["continuation"]["owner_sources"]["schedule"] = {"origin": dict(stored["metadata"]["continuation"][
            "owner_sources"]["schedule"]["origin"])}
        write_task_result(q.root, narrowed["id"], "scheduled", metadata=metadata)

    _assert_carrier(_occurrence_of_a_follow_up_of(q, narrowed["id"]), kept_basis_valid)


def test_a_narrowed_continue_inside_a_chain_decides_by_the_same_kept_basis(q):
    """A Continue kept its source's origin by value: the walk goes from it to that origin, past the source it continued."""
    import supervisor.schedule_occurrence as occurrences
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE

    source = _old_run(q)
    with pytest.MonkeyPatch.context() as previous:
        previous.setattr(occurrences, "_origin_relation", lambda *_a, **_k: {})
        ack = admit_continuation(source, action_nonce=NONCE)
    assert ack["ok"], ack
    [narrowed] = list(q.pending)
    q.pending.clear()
    later = _interrupted_after_it_ran(q, _accepted_before_the_relation(q, narrowed["id"], "s-later"))

    with _record_reads(q) as reads:
        task = _occurrence_of_a_follow_up_of(q, later)

    _assert_carrier(task, True)
    _assert_one_walk_each(reads, [later, narrowed["id"], "old-root"])


def test_a_schedule_basis_in_caller_metadata_with_no_host_admission_proves_nothing(q):
    """The kept basis counts only beside the host's validated admission: metadata alone can be anyone's words."""
    _write_origin(q.root, "old-root")
    basis = {"source": "schedule_admission", "schedule_id": "s0", "occurrence_token": "t", "objective": "x",
             "origin": {"task_id": "old-root", "root_task_id": "old-root"}}
    write_task_result(q.root, "api-task", "running", chat_id=1, delegation_role="root", root_task_id="api-task",
                      metadata={**AUTHORITY, "continuation": {"owner_sources": {"schedule": basis}}})

    _assert_carrier(_occurrence_of_a_follow_up_of(q, "api-task"), False)


@pytest.mark.parametrize("generations", [3, 8])
@pytest.mark.parametrize("upstream", ["old-root", "owner-turn"])
def test_a_chain_of_any_length_proves_through_every_generation_by_one_keyed_read_each(q, upstream, generations):
    """Each follow-up of a follow-up ran before the upgrade: no depth limits the proof, and the records stay as they ran."""
    _owner_row(q)
    chain = _generations(q, generations, upstream)
    recorded = [copy.deepcopy(load_task_result(q.root, task_id)) for task_id in chain]

    with _record_reads(q) as reads:
        task = _occurrence_of_a_follow_up_of(q, chain[-1])

    _assert_carrier(task, True)
    _assert_consumers(q, task, True)
    assert [task_id for task_id, _held in reads] == [*reversed(chain), upstream]  # one walk: the row then carries it by value
    _assert_one_walk_each(reads, [*reversed(chain), upstream])
    assert [load_task_result(q.root, task_id) for task_id in chain] == recorded  # the proof wrote nothing on what it read


def test_what_a_chain_proved_is_carried_by_value_so_later_generations_read_no_further(q):
    """The chain is paid once: an old-template row naming the proved run reads that record, and that run's own follow-up
    is written with the relation and reads nothing at all."""
    from ouroboros.tools.followup import _handle_schedule_followup
    from tests.test_schedule_occurrence import _rows

    task = _occurrence_of_a_follow_up_of(q, _generations(q, 3)[-1])
    q.pending.clear()
    write_task_result(q.root, task["id"], "running", schedule_admission={
        **load_task_result(q.root, task["id"])["schedule_admission"], "dispatch": "possible"})
    run = ToolContext(repo_dir=q.root, drive_root=q.root, budget_drive_root=q.root, task_id=task["id"], current_chat_id=1,
                      task_metadata=_pooled(task), task_contract=task.get("task_contract") or {})

    with _record_reads(q) as reads:
        old_template = _occurrence_of_a_follow_up_of(q, task["id"], "s-old-template")
    _assert_carrier(old_template, True)
    assert [task_id for task_id, _held in reads] == [task["id"]]

    q.pending.clear()
    written = _handle_schedule_followup(run, objective="Recheck", run_at="2000-01-01T00:00:00Z", relation="independent")
    assert written.startswith("FOLLOWUP_SCHEDULED"), written
    [row] = [item for item in _rows(q).values() if item["id"] != "s-old-template"
             and (item.get("followup_origin") or {}).get("task_id") == task["id"]]
    assert row["task"]["metadata"]["presence_binding_relation"]["binding_id"] == BINDING
    with _record_reads(q) as reads:
        q.queue.check_scheduled_tasks()
        q.queue.check_scheduled_tasks()
    [next_task] = q.pending
    _assert_carrier(next_task, True)
    assert reads == []


@pytest.mark.parametrize("upstream", ["acting-root", "other-binding", "child-origin", "origin-ceiling", "missing"])
def test_a_chain_that_ends_unproved_stays_narrowed_and_pays_its_whole_length(q, upstream):
    """The disclosed cost: only the last record decides, so a place's own acting chain reads every generation."""
    _owner_row(q)
    chain = _generations(q, 3, upstream)

    with _record_reads(q) as reads:
        task = _occurrence_of_a_follow_up_of(q, chain[-1])

    _assert_carrier(task, False)
    _assert_consumers(q, task, False)
    _assert_one_walk_each(reads, [*reversed(chain), upstream])


@pytest.mark.parametrize("change", ["no-host-receipt", "foreign-receipt", "ceiling", "parent", "child", "no-role", "speaker",
                                    "foreign-binding", "foreign-relation", "no-carrier", "retried"])
def test_an_invalid_intermediate_stops_the_proof_there_and_nothing_past_it_is_read(q, change):
    """Each generation answers the same checks as the first. The proving root past it would have converted the row."""
    chain = _generations(q, 4)
    invalid = chain[1]
    stored = load_task_result(q.root, invalid)
    receipt = copy.deepcopy(stored["schedule_admission"])
    if change == "retried":  # the effective read answers with the retry's record: it keeps no receipt naming itself
        write_task_result(q.root, "its-retry", "failed", chat_id=1, delegation_role="root", root_task_id="its-retry",
                          metadata=dict(AUTHORITY))
    fields = {
        "retried": {"superseded_by": "its-retry", "retry_task_id": "its-retry"},
        "no-host-receipt": {"schedule_admission": {}},
        "foreign-receipt": {"schedule_admission": {**receipt, "token": "a-foreign-token"}},
        "ceiling": {"task_contract": {"capability_ceiling": presence_ceiling_payload(_ceiling())}},
        "parent": {"parent_task_id": "a-parent"},
        "child": {"delegation_role": "subagent"},
        "no-role": {"delegation_role": ""},
        "speaker": {"metadata": {**stored["metadata"], "presence": _presence()}},
        "foreign-binding": {"metadata": {**stored["metadata"], "presence_binding_authority": {"binding_id": OTHER}}},
        "foreign-relation": {"metadata": {"presence_binding_relation": {"binding_id": OTHER}}},
        "no-carrier": {"metadata": {"caller_note": "an ordinary owner root"}},
    }[change]
    write_task_result(q.root, invalid, "failed", **fields)

    with _record_reads(q) as reads:
        task = _occurrence_of_a_follow_up_of(q, chain[-1])

    _assert_carrier(task, False)
    _assert_consumers(q, task, False)
    _assert_one_walk_each(reads, [chain[3], chain[2], invalid])


def test_a_generation_with_no_record_stops_the_proof_there(q):
    chain = _generations(q, 4)
    _corrupt_origin(q, chain[2], "never-written")

    with _record_reads(q) as reads:
        task = _occurrence_of_a_follow_up_of(q, chain[-1])

    _assert_carrier(task, False)
    _assert_one_walk_each(reads, [chain[3], chain[2], "never-written"])


def test_an_unreadable_generation_proves_nothing_and_the_next_pass_reads_it_again(q):
    """Contention on one intermediate is no verdict: the queued occurrence keeps its carrier and converges when it reads."""
    chain = _generations(q, 4)

    with _record_reads(q, busy=chain[1]) as reads:
        task = _occurrence_of_a_follow_up_of(q, chain[-1])

    _assert_carrier(task, False)
    _assert_one_walk_each(reads, [chain[3], chain[2], chain[1]])
    before = load_task_result(q.root, task["id"])["schedule_admission"]
    with _record_reads(q) as reads:
        q.queue.check_scheduled_tasks()
    _assert_one_walk_each(reads, [*reversed(chain), "old-root"])
    _read_back(q, task["id"], before, converted=True)


@pytest.mark.parametrize("names", [0, 1, 2], ids=["self", "two-records", "whole-chain"])
def test_a_corrupt_cycle_ends_at_the_repeated_id_without_another_read(q, names):
    """The oldest generation names itself, the one after it, or the newest: each id is read once and nothing proves."""
    chain = _generations(q, 3)
    _corrupt_origin(q, chain[0], chain[names])

    with _record_reads(q) as reads:
        task = _occurrence_of_a_follow_up_of(q, chain[-1])

    _assert_carrier(task, False)
    _assert_one_walk_each(reads, list(reversed(chain)))
