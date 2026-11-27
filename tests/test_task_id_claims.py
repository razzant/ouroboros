"""A new task id is claimed in the canonical store before anything is staged under it (#4317).

A request never takes, merges into or answers with another record's id: a generated id another record holds
is minted again, a fixed one is refused with that record unchanged, and the request's own record replays.
Every record here is imaginary and lives in ``tmp_path``; a collision is forced through the one mint
(``task_id_claims.mint_task_id``).
"""
from __future__ import annotations

import os
import shutil
import threading
from functools import partial
from types import SimpleNamespace

import pytest

from ouroboros import store, task_id_claims
from ouroboros.task_id_claims import (
    ID_CLAIMS, TaskIdTaken, TaskIdUnconfirmed, abandoned, advance_provisional_claim, claim_id, claim_provisional_id,
    release_provisional_claim, release_task_claim, remove_own_paths, retained_task_paths, same_request,
)
from ouroboros.task_results import load_task_result, write_task_result
from tests._store_shared import result_signature, seed_result, unreadable_result

FOREIGN = {"status": "completed", "result": "an unrelated answer from weeks ago"}


def _foreign(root, task_id):
    seed_result(root, task_id, {"task_id": task_id, **FOREIGN})
    return result_signature(root, task_id)


def _mint(monkeypatch, *ids):
    minted = iter(ids)
    monkeypatch.setattr(task_id_claims, "mint_task_id", lambda width: next(minted))
    return minted


def _claim_row(root, token):
    owns = partial(same_request, token=token)
    write = lambda tid: write_task_result(root, tid, "requested", claim=owns, create_only=True,  # noqa: E731
                                          request_claim={"token": token, "phase": "claimed"})
    return write, (lambda tid: release_task_claim(root, tid, owns))


def _held(root, task_id):
    with store.open_store(root, write=False) as txn:
        return txn.get_row(ID_CLAIMS, task_id)


# ---- the strict row claim --------------------------------------------------------------------------------------

def test_a_strict_claim_never_takes_or_changes_another_records_id(tmp_path):
    before = _foreign(tmp_path, "deadbeef")
    mine = partial(same_request, token="new-request")
    for create_only in (False, True):
        with pytest.raises(TaskIdTaken) as caught:
            write_task_result(tmp_path, "deadbeef", "requested", claim=mine, create_only=create_only,
                              request_claim={"token": "new-request"}, description="a new helper")
        assert caught.value.reason == "another_request"
    assert result_signature(tmp_path, "deadbeef") == before
    # The ordinary create-only write keeps its initialize-if-absent no-op.
    assert write_task_result(tmp_path, "deadbeef", "requested", create_only=True)["status"] == "completed"
    assert result_signature(tmp_path, "deadbeef") == before


def test_the_same_request_replays_its_row_and_only_its_own_row_is_released(tmp_path):
    mine = partial(same_request, token="t1")
    write_task_result(tmp_path, "aaaa0001", "requested", claim=mine, request_claim={"token": "t1", "phase": "claimed"})
    stored = write_task_result(tmp_path, "aaaa0001", "requested", claim=mine, description="the full request",
                               request_claim={"token": "t1", "phase": "requested"})
    assert stored["request_claim"]["phase"] == "requested" and stored["description"] == "the full request"
    seed_result(tmp_path, "aaaa0002", {"task_id": "aaaa0002", "status": "requested", "request_claim": {"token": "t2"}})
    assert release_task_claim(tmp_path, "aaaa0002", mine) is False
    assert load_task_result(tmp_path, "aaaa0002")["request_claim"] == {"token": "t2"}
    assert release_task_claim(tmp_path, "aaaa0001", mine) is True and load_task_result(tmp_path, "aaaa0001") is None
    assert release_task_claim(tmp_path, "aaaa0001", mine) is True  # nothing of this claim remains


def test_a_row_the_owner_kept_or_a_stop_targets_is_never_released_with_its_claim(tmp_path):
    from ouroboros.cancel_intents import has_active_intent, request_cancel

    mine = partial(same_request, token="t1")
    write_task_result(tmp_path, "aaaa0001", "requested", claim=mine, request_claim={"token": "t1"},
                      owner_keep={"at": "2026-11-26T00:00:00+00:00", "scope": "task", "source": "test"})
    assert release_task_claim(tmp_path, "aaaa0001", mine) is False
    assert load_task_result(tmp_path, "aaaa0001")["request_claim"] == {"token": "t1"}
    # A Stop recorded before the deleting transaction keeps the row its custody settles, and the Stop stays open.
    write_task_result(tmp_path, "aaaa0002", "requested", claim=mine, request_claim={"token": "t1"})
    request_cancel(tmp_path, "aaaa0002", source="test")
    assert release_task_claim(tmp_path, "aaaa0002", mine) is False
    assert load_task_result(tmp_path, "aaaa0002")["status"] == "requested" and has_active_intent(tmp_path, "aaaa0002")


def test_a_legacy_request_matches_only_a_legacy_helper_row_never_another_protocol(tmp_path):
    helper = {"delegation_role": "subagent", "parent_task_id": "parent0", "status": "requested"}
    assert same_request(helper, "") and same_request({**helper, "parent_task_id": "another"}, "")
    for other in ({"delegation_role": "root", "status": "requested"}, {"status": "completed"},
                  {"status": "requested", "promotion_admission": {"routing_token": "r"}}):
        assert not same_request(other, "")
    assert not same_request({**helper, "request_claim": {"token": "t"}}, "")  # a tokened row is no legacy request's
    assert not same_request(helper, "t")  # nor is a tokenless row a tokened request's
    assert same_request({**helper, "request_claim": {"token": "t"}}, "t") and not same_request({}, "")


def test_an_identity_the_import_left_occupied_or_a_bodyless_row_is_taken(tmp_path):
    from ouroboros.store_import import OCCUPIED

    write_task_result(tmp_path, "bodyless", "completed", result="a row whose body goes")
    with store.open_store(tmp_path, write=True) as txn:
        txn.set_meta(OCCUPIED, {"refused1": "schema_refused"})
        txn.conn.execute("DELETE FROM task_bodies WHERE task_id='bodyless'")
    mine = partial(same_request, token="t")
    for task_id, reason in (("refused1", "occupied_by_import"), ("bodyless", "row_without_readable_body")):
        with pytest.raises(TaskIdTaken) as caught:
            write_task_result(tmp_path, task_id, "requested", claim=mine, request_claim={"token": "t"})
        assert caught.value.reason == reason
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.row("refused1", "task_id") is None and txn.body_text("bodyless") is None


# ---- the claim of an id with no row yet ------------------------------------------------------------------------

def test_a_provisional_claim_is_no_task_row_yet_holds_the_id_against_every_claimant(tmp_path):
    from ouroboros.task_store import exists

    write_task_result(tmp_path, "seed0000", "requested")  # the store exists
    claim_provisional_id(tmp_path, "held0001", "t-http")
    assert load_task_result(tmp_path, "held0001") is None and not exists(tmp_path, "held0001")
    held = _held(tmp_path, "held0001")
    assert held["phase"] == "claimed" and held["producer"]["pid"] == os.getpid() and held["token"] == "t-http"
    claim_provisional_id(tmp_path, "held0001", "t-http")  # a replay of the same claim is the same claim
    with pytest.raises(TaskIdTaken) as caught:  # no other request, row claimant or provisional claimant takes it
        claim_provisional_id(tmp_path, "held0001", "t-other")
    assert caught.value.reason == "claimed"
    with pytest.raises(TaskIdTaken) as caught:
        write_task_result(tmp_path, "held0001", "requested", claim=partial(same_request, token="t-row"),
                          request_claim={"token": "t-row"})
    assert caught.value.reason == "claimed" and load_task_result(tmp_path, "held0001") is None
    assert release_provisional_claim(tmp_path, "held0001", "t-other") is False and _held(tmp_path, "held0001")
    assert release_provisional_claim(tmp_path, "held0001", "t-http") is True and _held(tmp_path, "held0001") is None
    assert release_provisional_claim(tmp_path, "held0001", "t-http") is True  # nothing of it remains
    assert write_task_result(tmp_path, "held0001", "requested", claim=partial(same_request, token="t-row"),
                             request_claim={"token": "t-row"})["status"] == "requested"


def test_a_provisional_claim_never_takes_an_id_a_row_holds(tmp_path):
    before = _foreign(tmp_path, "deadbeef")
    with pytest.raises(TaskIdTaken) as caught:
        claim_provisional_id(tmp_path, "deadbeef", "t-http")
    assert caught.value.reason == "another_request" and _held(tmp_path, "deadbeef") is None
    assert result_signature(tmp_path, "deadbeef") == before


def test_only_a_claim_abandoned_before_its_hand_off_is_taken_over(tmp_path, monkeypatch):
    write_task_result(tmp_path, "seed0000", "requested")
    claim_provisional_id(tmp_path, "held0001", "t-dead")
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, birth, proof=None: True)
    assert abandoned(_held(tmp_path, "held0001"))
    claim_provisional_id(tmp_path, "held0001", "t-new")  # its producer provably ended before any hand-off
    assert _held(tmp_path, "held0001")["token"] == "t-new"
    assert advance_provisional_claim(tmp_path, "held0001", "t-new", "handoff") == ""
    assert not abandoned(_held(tmp_path, "held0001"))
    with pytest.raises(TaskIdTaken):  # an admission that may be queued stays held, its producer gone or not
        claim_provisional_id(tmp_path, "held0001", "t-later")
    assert advance_provisional_claim(tmp_path, "held0001", "t-other", "handoff") == "lost"
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, birth, proof=None: False)
    claim_provisional_id(tmp_path, "held0002", "t-live")
    with pytest.raises(TaskIdTaken):  # a producer that may live keeps its claim
        claim_provisional_id(tmp_path, "held0002", "t-rival")


def test_two_concurrent_claims_of_one_id_admit_exactly_one_across_producers(tmp_path):
    write_task_result(tmp_path, "seed0000", "requested")  # the store exists before the race
    barrier, outcomes = threading.Barrier(3), {}

    def row_claim(token):
        barrier.wait()
        try:
            write_task_result(tmp_path, "race0001", "requested", claim=partial(same_request, token=token),
                              create_only=True, request_claim={"token": token})
            outcomes[token] = "claimed"
        except TaskIdTaken:
            outcomes[token] = "taken"

    def provisional(token):
        barrier.wait()
        try:
            claim_provisional_id(tmp_path, "race0001", token)
            outcomes[token] = "claimed"
        except TaskIdTaken:
            outcomes[token] = "taken"

    threads = [threading.Thread(target=row_claim, args=("a",)), threading.Thread(target=row_claim, args=("b",)),
               threading.Thread(target=provisional, args=("c",))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert sorted(outcomes.values()) == ["claimed", "taken", "taken"]


def test_a_case_alias_is_taken_only_where_the_volume_folds_case(tmp_path, monkeypatch):
    write_task_result(tmp_path, "AbC12345", "completed", result="a legacy id with capitals")
    claim_provisional_id(tmp_path, "Held0001", "t-http")
    monkeypatch.setattr(task_id_claims, "case_insensitive", lambda _dir: False)
    claim_provisional_id(tmp_path, "abc12345", "t-sensitive")  # a distinct name on a case-sensitive volume
    release_provisional_claim(tmp_path, "abc12345", "t-sensitive")
    monkeypatch.setattr(task_id_claims, "case_insensitive", lambda _dir: True)
    write, _release = _claim_row(tmp_path, "t-row")
    with pytest.raises(TaskIdTaken) as caught:  # a row, and a claim, spelled otherwise are the same file-keyed state
        write("ABC12345")
    assert caught.value.reason == "name_alias"
    with pytest.raises(TaskIdTaken) as caught:
        claim_provisional_id(tmp_path, "HELD0001", "t-alias")
    assert caught.value.reason == "name_alias"
    # Exact legacy reads are never narrowed: each spelling that exists still reads as itself.
    assert load_task_result(tmp_path, "AbC12345")["task_id"] == "AbC12345"
    assert load_task_result(tmp_path, "ABC12345") is None


def test_names_the_file_system_resolves_to_an_occupied_one_are_taken_and_unholdable_new_ones_refused(tmp_path,
                                                                                                  monkeypatch):
    from ouroboros.store_import import OCCUPIED

    write_task_result(tmp_path, "legacy1.", "completed", result="a legacy id ending in a dot")
    with store.open_store(tmp_path, write=True) as txn:
        txn.set_meta(OCCUPIED, {"Refused2.": "schema_refused", "ABC123": "quarantined_before_import"})
    monkeypatch.setattr(task_id_claims, "IS_WINDOWS", False)
    monkeypatch.setattr(task_id_claims, "case_insensitive", lambda _dir: True)
    with pytest.raises(TaskIdTaken) as caught:  # the import's identities too, under another spelling
        claim_provisional_id(tmp_path, "abc123", "t")
    assert caught.value.reason == "occupied_by_import"
    claim_provisional_id(tmp_path, "legacy1", "t")  # off Windows a trailing dot is a distinct name
    release_provisional_claim(tmp_path, "legacy1", "t")
    monkeypatch.setattr(task_id_claims, "IS_WINDOWS", True)
    for taken, reason in (("legacy1", "name_alias"), ("Legacy1", "name_alias"), ("refused2", "occupied_by_import")):
        with pytest.raises(TaskIdTaken) as caught:
            claim_provisional_id(tmp_path, taken, "t")
        assert caught.value.reason == reason, taken
    with pytest.raises(task_id_claims.TaskIdUnrepresentable):  # a new id Windows cannot hold as written
        claim_id("fresh.", *_claim_row(tmp_path, "t"), width=8)
    assert load_task_result(tmp_path, "legacy1.")["task_id"] == "legacy1."  # historical reads are unchanged


def test_the_alias_lookup_seeks_existing_spellings_never_scans_the_history(tmp_path):
    from ouroboros.task_id_claims import taken_in

    def statements() -> list:
        seen: list = []
        with store.open_store(tmp_path, write=True) as txn:
            txn.conn.set_trace_callback(seen.append)
            try:
                taken_in(txn, "abcdef12", folds=True)
            finally:
                txn.conn.set_trace_callback(None)
        return seen

    for index in range(3):
        write_task_result(tmp_path, f"ab{index:06d}", "completed")
    few = statements()
    for index in range(300):
        write_task_result(tmp_path, f"{index:08x}", "completed")
    many = statements()
    assert len(few) == len(many) <= 4 * len("abcdef12") + 10, (len(few), len(many))
    with store.open_store(tmp_path, write=False) as txn:
        plan = [row[-1] for row in txn.conn.execute(
            "EXPLAIN QUERY PLAN SELECT task_id FROM tasks WHERE task_id>=? AND task_id<? ORDER BY task_id LIMIT 1",
            ("ab", "ac"))]
    assert plan and all("SEARCH" in step and "SCAN" not in step for step in plan), plan


def test_a_path_inspection_error_is_never_read_as_absence(tmp_path, monkeypatch):
    real = os.lstat
    held = retained_task_paths(tmp_path, "aaaa0001")[0]

    def denied(path, *args, **kwargs):
        if str(path) == str(held):
            raise PermissionError(13, "denied", str(path))
        return real(path, *args, **kwargs)

    write_task_result(tmp_path, "seed0000", "requested")
    monkeypatch.setattr(task_id_claims.os, "lstat", denied)
    with pytest.raises(PermissionError):
        task_id_claims.present(held)
    minted = _mint(monkeypatch, "aaaa0001", "bbbb0002")
    written: list = []
    with pytest.raises(task_id_claims.TaskIdUninspectable) as caught:  # neither free nor moved to another id
        claim_id("", written.append, lambda _tid: True, width=8,
                 folders=lambda tid: retained_task_paths(tmp_path, tid))
    assert caught.value.task_id == "aaaa0001" and written == [] and next(minted) == "bbbb0002"
    assert remove_own_paths([held]) is False


def test_a_volume_whose_case_folding_cannot_be_probed_is_read_as_folding_and_not_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(task_id_claims, "_FOLDING", {})

    def denied(_path):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(task_id_claims.os, "scandir", denied)
    assert task_id_claims.case_insensitive(tmp_path) is True and task_id_claims._FOLDING == {}


def test_startup_retires_only_the_claims_proof_settles(tmp_path, monkeypatch):
    from ouroboros.task_id_claims import retire_settled_claims

    write_task_result(tmp_path, "receipt1", "scheduled", result="a receipt whose claim was never retired")

    def claim(phase, pid):
        return {"token": phase + str(pid), "phase": phase, "producer": {"pid": pid, "birth": 1}}

    with store.open_store(tmp_path, write=True) as txn:
        for tid, record in (("receipt1", claim("handoff", 12)), ("ended001", claim("claimed", 11)),
                            ("handoff1", claim("handoff", 11)), ("living01", claim("claimed", 12))):
            txn.put_row(ID_CLAIMS, tid, record)
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: pid == 11)
    assert sorted(retire_settled_claims(tmp_path)) == ["ended001", "receipt1"]
    assert _held(tmp_path, "handoff1") and _held(tmp_path, "living01")  # an unread queue proves no hand-off ended
    assert load_task_result(tmp_path, "receipt1")["status"] == "scheduled"
    with store.open_store(tmp_path, write=True) as txn:  # more hand-offs whose producer ended
        for tid in ("queued01", "staged02"):
            txn.put_row(ID_CLAIMS, tid, claim("handoff", 11))
        txn.put_row(ID_CLAIMS, "handoff3", claim("handoff", 12))  # its producer may still live
    (tmp_path / "task_drives" / "staged02").mkdir(parents=True)  # what it staged stays, under its recorded refusal
    # The recovered queue, read whole: a hand-off it holds may still run; one it lacks cannot. With nothing staged its
    # claim goes; what one staged stays keyed by its never-admitted refusal, which then holds the id.
    assert sorted(retire_settled_claims(tmp_path, queued={"queued01"})) == ["handoff1", "staged02"]
    assert all(_held(tmp_path, tid) for tid in ("queued01", "handoff3", "living01"))
    staged = load_task_result(tmp_path, "staged02")
    assert staged["admission_outcome"] == "never_admitted" and staged["admission_cleanup"] == {"staging_retained": True}
    assert (tmp_path / "task_drives" / "staged02").is_dir() and load_task_result(tmp_path, "handoff1") is None


# ---- folders an id keys ----------------------------------------------------------------------------------------

def _staged(root, tid):  # the selection a helper's parent retains for it (``delegate_readonly_inputs``)
    return root / "task_results" / "artifacts" / "parent0" / "source_handles" / "inputs" / f"scheduled-{tid}"


def test_every_producers_retained_paths_are_one_list_and_any_of_them_takes_the_id(tmp_path, monkeypatch):
    paths = retained_task_paths(tmp_path, "aaaa0001", _staged(tmp_path, "aaaa0001"))
    assert [p.relative_to(tmp_path).parts[:2] for p in paths[:3]] == [
        ("state", "headless_tasks"), ("task_results", "artifacts"), ("task_drives", "aaaa0001")]
    assert paths[3] == _staged(tmp_path, "aaaa0001")
    for index, path in enumerate(paths):  # a task's state, its artifacts, a direct task drive, a retained selection
        path.mkdir(parents=True)
        _mint(monkeypatch, "aaaa0001", f"free000{index}")
        write, release = _claim_row(tmp_path, f"t{index}")
        tid, _row = claim_id("", write, release, width=8,
                             folders=lambda tid: retained_task_paths(tmp_path, tid, _staged(tmp_path, tid)))
        assert tid == f"free000{index}" and load_task_result(tmp_path, "aaaa0001") is None
        shutil.rmtree(path)


def test_a_generated_id_is_minted_again_only_on_a_proven_occupancy_never_on_an_unknown_outcome(tmp_path, monkeypatch):
    before = _foreign(tmp_path, "deadbeef")
    _mint(monkeypatch, "deadbeef", "cafef00d")
    write, release = _claim_row(tmp_path, "t")
    tid, stored = claim_id("", write, release, width=8)
    assert tid == "cafef00d" and same_request(stored, "t") and result_signature(tmp_path, "deadbeef") == before
    with pytest.raises(TaskIdTaken):  # a fixed id is refused, never replaced
        claim_id("deadbeef", *_claim_row(tmp_path, "u"), width=8)
    assert result_signature(tmp_path, "deadbeef") == before
    minted = _mint(monkeypatch, "feedface", "0badf00d")

    def unanswered(_tid):
        raise RuntimeError("the store did not answer")

    with pytest.raises(TaskIdUnconfirmed) as caught:  # may be committed: reported by name, never moved to another id
        claim_id("", unanswered, lambda _tid: False, width=8)
    assert caught.value.task_id == "feedface" and next(minted) == "0badf00d"


def test_folders_made_after_the_claim_by_a_writer_that_claims_nothing_take_the_id_when_the_claim_is_released(tmp_path,
                                                                                                         monkeypatch):
    _mint(monkeypatch, "aaaa0001", "aaaa0002")
    write, release = _claim_row(tmp_path, "t")

    def write_then_folder(tid):
        stored = write(tid)
        if tid == "aaaa0001":
            (tmp_path / "task_drives" / tid).mkdir(parents=True)
        return stored

    tid, _row = claim_id("", write_then_folder, release, width=8, folders=lambda tid: retained_task_paths(tmp_path, tid))
    assert tid == "aaaa0002" and load_task_result(tmp_path, "aaaa0001") is None
    _mint(monkeypatch, "bbbb0001", "bbbb0002")

    def write_then_folder_unreleasable(tid):
        stored = write(tid)
        (tmp_path / "task_drives" / tid).mkdir(parents=True)
        return stored

    with pytest.raises(TaskIdUnconfirmed) as caught:  # its claim cannot be proved released: held, named, not moved
        claim_id("", write_then_folder_unreleasable, lambda _tid: False, width=8,
                 folders=lambda tid: retained_task_paths(tmp_path, tid))
    assert caught.value.task_id == "bbbb0001"


def test_only_a_folder_that_holds_nothing_goes_and_one_with_any_file_stays_exactly_as_it_is(tmp_path):
    state, artifacts, direct = retained_task_paths(tmp_path, "aaaa0001")
    (state / "data" / "memory").mkdir(parents=True)  # empty folders alone: nobody's bytes
    (artifacts / "attachments").mkdir(parents=True)
    assert remove_own_paths((state, artifacts, direct)) is True
    assert not any(os.path.lexists(path) for path in (state, artifacts, direct))
    (state / "data" / "memory").mkdir(parents=True)
    (state / "data" / "image.bin").write_bytes(b"a fictional image: whose it is, nothing here proves")
    assert remove_own_paths((state,)) is False and (state / "data" / "memory").is_dir()  # nothing of it is touched
    assert (state / "data" / "image.bin").read_bytes() == b"a fictional image: whose it is, nothing here proves"
    target = tmp_path / "elsewhere"
    (target / "empty").mkdir(parents=True)
    artifacts.parent.mkdir(parents=True, exist_ok=True)
    artifacts.symlink_to(target, target_is_directory=True)  # a link is never followed or removed as its own
    assert remove_own_paths((artifacts,)) is False and os.path.islink(artifacts) and (target / "empty").is_dir()


def test_a_folder_under_another_spelling_is_an_alias_and_is_never_removed_as_ones_own(tmp_path):
    (tmp_path / "state" / "headless_tasks" / "AAAA0001").mkdir(parents=True)
    if not task_id_claims.case_insensitive(tmp_path):
        pytest.skip("needs a volume that folds case")
    alias = retained_task_paths(tmp_path, "aaaa0001")[0]
    assert os.path.lexists(alias)  # the volume answers for the other spelling: the id is taken
    assert remove_own_paths([alias]) is False and (tmp_path / "state" / "headless_tasks" / "AAAA0001").is_dir()


def test_an_attempt_that_loses_the_claim_leaves_the_holders_reservation_and_files_alone(tmp_path, monkeypatch):
    from supervisor import queue, task_api_claim, workers
    from supervisor.task_admission import reserve_task_admission

    monkeypatch.setattr(workers, "WORKERS", {0: SimpleNamespace()})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    queue.ADMISSION_RESERVATIONS.clear()
    write_task_result(tmp_path, "seed0000", "requested")
    claim_provisional_id(tmp_path, "held0001", "winner")
    folder = tmp_path / "state" / "headless_tasks" / "held0001" / "data"
    folder.mkdir(parents=True)
    queue.ADMISSION_RESERVATIONS["held0001"] = "winner"
    # The loser's reservation is refused by memory first, before the store is asked.
    assert reserve_task_admission("held0001", "loser", drive_root=tmp_path)["reason"] == "duplicate_task_id"
    task_id, refusal = task_api_claim.claim_task_admission(tmp_path, "held0001", "loser")
    assert task_id == "" and refusal[0] == 409
    assert queue.ADMISSION_RESERVATIONS["held0001"] == "winner" and folder.is_dir()
    assert _held(tmp_path, "held0001")["token"] == "winner"
    queue.ADMISSION_RESERVATIONS.clear()
    # With memory free, the store's claim refuses it, and the loser never holds a reservation.
    task_id, refusal = task_api_claim.claim_task_admission(tmp_path, "held0002", "w2")
    assert task_id == "held0002" and refusal is None and queue.ADMISSION_RESERVATIONS["held0002"] == "w2"
    queue.ADMISSION_RESERVATIONS.pop("held0002")
    task_id, refusal = task_api_claim.claim_task_admission(tmp_path, "held0002", "loser")
    assert task_id == "" and refusal[0] == 409 and refusal[2]["reason_code"] == "duplicate_task_id"
    assert "held0002" not in queue.ADMISSION_RESERVATIONS and _held(tmp_path, "held0002")["token"] == "w2"


# ---- helpers: schedule_subagent and its admission --------------------------------------------------------------

def _ctx(tmp_path):
    from ouroboros.tools.registry import ToolContext

    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    return ToolContext(repo_dir=repo, drive_root=data, task_id="parent0"), data


def _schedule(tmp_path, monkeypatch, **params):
    from ouroboros.tools import control_scheduling
    from tests._shared import configure_test_subagent

    subagent_id = configure_test_subagent(monkeypatch)
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "3")
    ctx, data = _ctx(tmp_path)
    result = control_scheduling._schedule_task(
        ctx, subagent_id=subagent_id, objective="Inspect files", expected_output="findings", **params)
    return ctx, data, result


def test_schedule_subagent_never_returns_an_id_another_record_holds(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    before = _foreign(data, "deadbeef")
    _mint(monkeypatch, "deadbeef", "cafef00d")
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")
    [event] = ctx.pending_events
    assert event["task_id"] == "cafef00d" and "deadbeef" not in result, result
    assert result_signature(data, "deadbeef") == before
    row = load_task_result(data, "cafef00d")
    assert row["status"] == "requested" and row["request_claim"]["phase"] == "requested"
    assert same_request(row, event["request_token"])
    # Distinct ids answer distinct requests: the waiter's id reads this request, never the old answer.
    assert row["description"] == "Inspect files" and row.get("result") != FOREIGN["result"]


def test_a_retained_selection_of_an_earlier_owner_takes_the_id_as_a_row_does(tmp_path, monkeypatch):
    from ouroboros.delegate_readonly_inputs import scheduled_inputs_root

    ctx, data = _ctx(tmp_path)
    taken = scheduled_inputs_root(ctx, "deadbeef")
    assert taken.name == "scheduled-deadbeef"
    taken.mkdir(parents=True)
    _mint(monkeypatch, "deadbeef", "cafef00d")
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")
    assert ctx.pending_events[0]["task_id"] == "cafef00d" and taken.is_dir()
    assert load_task_result(data, "deadbeef") is None


def _fail_request_write(monkeypatch, after_write=None):
    """The helper's full REQUESTED write fails; its claim write (phase ``claimed``) is the real one."""
    from ouroboros.tools import control_scheduling

    real = control_scheduling.write_task_result

    def write(root, tid, status, **fields):
        if (fields.get("request_claim") or {}).get("phase") == "requested":
            if after_write is not None:
                after_write(root, tid)
            raise OSError("the store refused the request row")
        return real(root, tid, status, **fields)

    monkeypatch.setattr(control_scheduling, "write_task_result", write)


def test_an_unsent_request_keeps_what_it_staged_whole_and_its_record_closed_as_never_sent(tmp_path, monkeypatch):
    from ouroboros.headless import prepared_drive_mode, task_state_dir

    _mint(monkeypatch, "cafef00d")
    _fail_request_write(monkeypatch, after_write=lambda root, tid: (root / "task_drives" / tid).mkdir(parents=True))
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")
    assert "failed to persist requested status for cafef00d" in result and "never reused" not in result, result
    assert "task id cafef00d keeps this request's record, closed as never sent: its staged folder stays" in result
    row = load_task_result(data, "cafef00d")  # another writer may have written there too: the id stays with it
    assert row["status"] == "failed" and row["reason_code"] == "request_unsent", row
    assert row["admission_outcome"] == "never_admitted" and "staged folder stays" in row["result"]
    assert prepared_drive_mode(task_state_dir(data, "cafef00d") / "data", "cafef00d") == "empty"  # untouched
    assert (task_state_dir(data, "cafef00d") / "data" / "memory").is_dir()
    assert (data / "task_drives" / "cafef00d").is_dir()  # a direct drive this request never makes is not its to remove
    assert ctx.pending_events == []


def test_an_unsent_request_that_raises_before_staging_releases_its_claim_too(tmp_path, monkeypatch):
    from ouroboros.tools import control_scheduling

    def refused(*_args, **_kwargs):
        raise ValueError("an inherited declared selection cannot widen")

    monkeypatch.setattr(control_scheduling, "_child_constraint", refused)
    _mint(monkeypatch, "cafef00d")
    with pytest.raises(ValueError):
        _schedule(tmp_path, monkeypatch, memory_mode="empty")
    assert load_task_result(tmp_path / "data", "cafef00d") is None
    assert not (tmp_path / "data" / "state" / "headless_tasks" / "cafef00d").exists()


def test_an_unsent_request_honors_the_owners_keep_and_says_its_id_stays_held(tmp_path, monkeypatch):
    from ouroboros.headless import task_state_dir

    keep = {"at": "2026-11-26T00:00:00+00:00", "scope": "task", "source": "test"}
    _mint(monkeypatch, "cafef00d")
    _fail_request_write(monkeypatch, after_write=lambda root, tid: write_task_result(
        root, tid, "requested", owner_keep=keep))
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")
    assert "SUBTASK_STATUS_ERROR: task id cafef00d was claimed" in result and "release is unconfirmed" in result
    assert "is unknown" in result and "never reused" not in result
    row = load_task_result(data, "cafef00d")
    assert row["owner_keep"] == keep and row["request_claim"]["phase"] == "claimed"
    assert task_state_dir(data, "cafef00d").is_dir()  # the kept materials stay
    assert ctx.pending_events == []


def test_a_record_that_could_not_be_closed_keeps_the_row_and_names_the_attempted_id(tmp_path, monkeypatch):
    from ouroboros.tools import control_scheduling

    _mint(monkeypatch, "cafef00d")
    _fail_request_write(monkeypatch)
    monkeypatch.setattr(control_scheduling, "close_unsent_claim", lambda *_a, **_k: False)
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")
    assert "SUBTASK_STATUS_ERROR: task id cafef00d was claimed" in result, result
    assert "its staged folder stays" in result and "its record could not be closed" in result
    assert load_task_result(data, "cafef00d")["request_claim"]["phase"] == "claimed"  # recovery refuses it as unsent


def test_a_refusal_before_staging_releases_its_row_and_removes_no_folder(tmp_path, monkeypatch):
    from ouroboros.tools import control_scheduling

    def refused_beside_a_foreign_folder(_ctx, _params, _surface, _workspace, _metadata, named):
        (tmp_path / "data" / "task_drives" / named[1][0]).mkdir(parents=True)  # a writer that claims nothing
        return "", "", "refused before staging"

    monkeypatch.setattr(control_scheduling, "_child_constraint", refused_beside_a_foreign_folder)
    _mint(monkeypatch, "cafef00d")
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")
    assert result == "refused before staging" and load_task_result(data, "cafef00d") is None
    assert (data / "task_drives" / "cafef00d").is_dir()  # it staged nothing, so it removes nothing


def test_a_failed_drive_waits_for_the_gated_release_which_the_owners_keep_holds(tmp_path, monkeypatch):
    from ouroboros.tools import control_scheduling

    keep = {"at": "2026-11-26T00:00:00+00:00", "scope": "task", "source": "test"}

    def partial_drive(root, tid, _mode, project_id=""):
        (root / "state" / "headless_tasks" / tid / "data").mkdir(parents=True)
        write_task_result(root, tid, "requested", owner_keep=keep)
        raise OSError("no room")

    monkeypatch.setattr(control_scheduling, "prepare_task_drive", partial_drive)
    _mint(monkeypatch, "cafef00d")
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")
    assert "SUBTASK_DRIVE_ERROR" in result and "it is held (owner_keep)" in result, result
    assert (data / "state" / "headless_tasks" / "cafef00d" / "data").is_dir() and ctx.pending_events == []


def test_a_pending_stop_keeps_an_unsent_claim(tmp_path, monkeypatch):
    from ouroboros.cancel_intents import request_cancel

    _mint(monkeypatch, "cafef00d")
    _fail_request_write(monkeypatch, after_write=lambda root, tid: request_cancel(root, tid, source="test"))
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")
    assert "a Stop of it is pending" in result and ctx.pending_events == [], result
    assert load_task_result(data, "cafef00d")["status"] == "requested"
    assert (data / "state" / "headless_tasks" / "cafef00d").is_dir()


def test_a_stop_landing_while_an_unsent_claim_is_undone_keeps_its_row(tmp_path, monkeypatch):
    from ouroboros.cancel_intents import has_active_intent, request_cancel
    from ouroboros.tools import control_scheduling

    real = control_scheduling.remove_own_paths

    def stop_meanwhile(paths, **kwargs):  # after the rollback's own look for a Stop, before its writing transaction
        removed = real(paths, **kwargs)
        request_cancel(tmp_path / "data", "cafef00d", source="test")
        return removed

    monkeypatch.setattr(control_scheduling, "remove_own_paths", stop_meanwhile)
    _mint(monkeypatch, "cafef00d")
    _fail_request_write(monkeypatch)
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")
    assert "a Stop of it is pending" in result and ctx.pending_events == [], result
    row = load_task_result(data, "cafef00d")  # the row the Stop's custody settles, and the Stop itself, both stand
    assert row["status"] == "requested" and row["request_claim"]["phase"] == "claimed"
    assert has_active_intent(data, "cafef00d")


def test_a_claim_of_unknown_outcome_is_named_never_moved_and_grants_no_release_or_removal(tmp_path, monkeypatch):
    from ouroboros.tools import control_scheduling

    real = control_scheduling.write_task_result

    def commit_then_fail(root, tid, status, **fields):
        stored = real(root, tid, status, **fields)
        if (fields.get("request_claim") or {}).get("phase") == "claimed":
            (root / "task_drives" / tid).mkdir(parents=True)  # whoever made it, an unproved claim removes nothing
            raise OSError("the store answered too late")
        return stored

    monkeypatch.setattr(control_scheduling, "write_task_result", commit_then_fail)
    minted = _mint(monkeypatch, "cafef00d", "0badf00d")
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")
    assert "SUBTASK_STATUS_ERROR: the claim of task id cafef00d is unconfirmed" in result, result
    assert next(minted) == "0badf00d" and ctx.pending_events == []
    assert load_task_result(data, "cafef00d")["request_claim"]["phase"] == "claimed"  # recovery refuses it as unsent
    assert (data / "task_drives" / "cafef00d").is_dir()


def test_an_unproven_release_after_an_error_reports_the_original_cause_and_the_claim(tmp_path, monkeypatch):
    from ouroboros.tools import control_scheduling

    def failing(*_args, **_kwargs):
        raise ValueError("an inherited declared selection cannot widen")

    monkeypatch.setattr(control_scheduling, "_build_child_subagent_contract", failing)
    monkeypatch.setattr(control_scheduling, "close_unsent_claim", lambda *_a, **_k: False)
    _mint(monkeypatch, "cafef00d")
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")
    assert "ValueError: an inherited declared selection cannot widen" in result, result
    assert "task id cafef00d was claimed" in result and "its record could not be closed" in result
    assert "is unknown" in result and ctx.pending_events == []
    assert load_task_result(data, "cafef00d")["request_claim"]["phase"] == "claimed"


def test_a_control_signal_keeps_its_type_while_its_record_keeps_the_id(tmp_path, monkeypatch):
    from ouroboros.owner_pause import OwnerPauseRefused
    from ouroboros.tools import control_scheduling

    def paused(*_args, **_kwargs):
        raise OwnerPauseRefused("owner_launch_authority_unavailable")

    monkeypatch.setattr(control_scheduling, "_build_child_subagent_contract", paused)
    _mint(monkeypatch, "cafef00d")
    with pytest.raises(OwnerPauseRefused, match="owner_launch_authority_unavailable"):
        _schedule(tmp_path, monkeypatch, memory_mode="empty")
    row = load_task_result(tmp_path / "data", "cafef00d")  # what it staged stays, keyed by its closed record
    assert row["status"] == "failed" and row["reason_code"] == "request_unsent", row


@pytest.mark.parametrize("failure", ["refused", "raised"])
def test_a_retained_selection_keeps_its_record_closed_as_never_sent(tmp_path, monkeypatch, failure):
    from ouroboros.delegate_readonly_inputs import scheduled_inputs_root
    from ouroboros.headless import task_state_dir
    from ouroboros.tools import control_scheduling

    def retains_then_stops(ctx, _fields, child_task_id, *_args):
        scheduled_inputs_root(ctx, child_task_id).mkdir(parents=True)  # the selection retained for the parent
        if failure == "raised":
            raise ValueError("the selection could not be bound")
        return "⚠️ input_files_unavailable: one selected file changed. The child was not queued."

    monkeypatch.setattr(control_scheduling, "_prepare_child_input_files", retains_then_stops)
    _mint(monkeypatch, "cafef00d")
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")
    cause = "the selection could not be bound" if failure == "raised" else "one selected file changed"
    assert cause in result and "task id cafef00d keeps this request's record, closed as never sent" in result, result
    row = load_task_result(data, "cafef00d")
    assert row["status"] == "failed" and row["admission_outcome"] == "never_admitted"
    assert row["reason_code"] == "request_unsent" and ctx.pending_events == []
    assert "staged folder stays" in row["result"] and "inputs selected for it stay" in row["result"]
    assert scheduled_inputs_root(ctx, "cafef00d").is_dir() and task_state_dir(data, "cafef00d").is_dir()


def test_helper_and_start_refusals_name_no_generated_id_another_record_holds(tmp_path, monkeypatch):
    from ouroboros.tools.control import _start_task

    data = tmp_path / "data"
    data.mkdir()
    _foreign(data, "deadbeef")
    monkeypatch.setattr(task_id_claims, "mint_task_id", lambda _width: "deadbeef")
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")
    assert "every generated task id is held by another record" in result and "deadbeef" not in result, result
    _foreign(tmp_path, "deadbeefdeadbeef")
    monkeypatch.setattr(task_id_claims, "mint_task_id", lambda _width: "deadbeefdeadbeef")
    _start_ready(monkeypatch)
    start_ctx = _start_ctx(tmp_path)
    out = _start_task(start_ctx, "a separate work", title="Separate work")
    assert out.startswith("⚠️ PROMOTE_REJECTED: the task was not scheduled") and "deadbeef" not in out, out
    assert start_ctx.pending_events == []


# ---- a pending Stop -------------------------------------------------------------------------------------------

def _stop(root, task_id):
    from ouroboros.cancel_intents import request_cancel

    request_cancel(root, task_id, source="test")


def test_a_pending_stop_takes_an_id_from_every_claim_and_never_wedges_a_provisional_one(tmp_path, monkeypatch):
    write_task_result(tmp_path, "seed0000", "requested")
    _stop(tmp_path, "aaaa0001")
    with pytest.raises(TaskIdTaken) as caught:
        claim_id("aaaa0001", *_claim_row(tmp_path, "t"), width=8)
    assert caught.value.reason == "stop_pending" and load_task_result(tmp_path, "aaaa0001") is None
    _mint(monkeypatch, "aaaa0001", "bbbb0002")
    assert claim_id("", *_claim_row(tmp_path, "t"), width=8)[0] == "bbbb0002"
    with pytest.raises(TaskIdTaken):  # the task API's claim is read under the same interlock
        claim_provisional_id(tmp_path, "aaaa0001", "t-http")
    assert _held(tmp_path, "aaaa0001") is None
    claim_provisional_id(tmp_path, "held0001", "t-http")
    _stop(tmp_path, "held0001")
    assert advance_provisional_claim(tmp_path, "held0001", "t-http", "handoff") == "stopped"
    # Its rollback releases the claim: the Stop still takes the id from the next claim, and its custody finds
    # nothing under it to settle.
    assert release_provisional_claim(tmp_path, "held0001", "t-http") is True and _held(tmp_path, "held0001") is None
    with pytest.raises(TaskIdTaken):
        claim_provisional_id(tmp_path, "held0001", "t-retry")
    claim_provisional_id(tmp_path, "held0002", "t-ended")
    _stop(tmp_path, "held0002")
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: True)
    assert task_id_claims.retire_settled_claims(tmp_path) == ["held0002"]  # a Stop wedges no abandoned claim


def test_a_claims_write_and_a_stop_take_one_order_under_the_stop_interlock(tmp_path):
    from ouroboros.cancel_intents import has_active_intent
    from ouroboros.task_id_claims import stop_fenced

    write_task_result(tmp_path, "seed0000", "requested")
    recorded = threading.Event()
    stop = threading.Thread(target=lambda: (_stop(tmp_path, "aaaa0003"), recorded.set()))

    def commit():  # a Stop of the id arrives while the claim's write holds the interlock
        stop.start()
        assert not recorded.wait(0.5)  # it waits for the commit, never landing between the read and the write
        return "committed"

    assert stop_fenced(tmp_path, "aaaa0003", commit) == "committed"
    stop.join(10)
    assert recorded.is_set() and has_active_intent(tmp_path, "aaaa0003")  # after the claim: on its record
    with pytest.raises(TaskIdTaken) as caught:  # and every later claim's write is refused, nothing written
        stop_fenced(tmp_path, "aaaa0003", lambda: pytest.fail("written over a pending Stop"))
    assert caught.value.reason == "stop_pending"


def test_a_store_busy_before_anything_was_written_refuses_the_claim_unheld(tmp_path):
    from ouroboros.store import StoreBusy

    def busy(rolled_back):
        def write(_tid):
            error = StoreBusy("the store is busy", reason="contention")
            error.rollback_confirmed = rolled_back
            raise error
        return write

    with pytest.raises(task_id_claims.TaskIdUninspectable):  # nothing was written: no claim to release or name
        claim_id("aaaa0005", busy(True), lambda _tid: pytest.fail("nothing to release"), width=8)
    with pytest.raises(TaskIdUnconfirmed):  # a commit whose rollback is unknown stays unknown
        claim_id("aaaa0006", busy(False), lambda _tid: True, width=8)


def test_a_claims_store_wait_under_the_stop_lock_is_one_short_slice_and_a_busy_store_mints_no_second_id(
        tmp_path, monkeypatch):
    from ouroboros import store
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
    from ouroboros.cancel_intents import cancellation_projection_lock
    from ouroboros.store import StoreBusy

    budgets = []

    def write():
        budgets.append(store.bounded_wait(10.0))  # what a store acquisition may wait while the Stop lock is held
        return "written"

    assert task_id_claims.stop_fenced(tmp_path, "aaaa0026", write) == "written"
    assert budgets == [ACQUISITION_SLICE_SEC]
    minted = _mint(monkeypatch, "aaaa0027", "aaaa0028")

    def busy(tid):  # a store held by another writer: the claim's acquisition gives up, nothing written
        def write_under_lock():
            raise StoreBusy("the store is busy", reason="contention")
        return task_id_claims.stop_fenced(tmp_path, tid, write_under_lock)

    with pytest.raises(task_id_claims.TaskIdUninspectable):  # refused unheld, never success
        claim_id("", busy, lambda _tid: pytest.fail("nothing to release"), width=8)
    assert next(minted) == "aaaa0028"  # the unknown outcome minted no second id
    with cancellation_projection_lock(tmp_path):  # and the Stop lock was left at once
        pass


def test_recovery_leaves_a_stopped_request_to_the_stops_custody(tmp_path, monkeypatch):
    from ouroboros.promotion_intent import prepare_promotion_intent, reconcile_promotion_intents
    from supervisor import queue, workspace_preparation as prep

    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: True)
    seed_result(tmp_path, "aaaa0017", _unsent("aaaa0017", root_task_id="parent0", role="helper"))
    prepare_promotion_intent(tmp_path, {"task_id": "aaaa0018", "routing_token": "route", "objective": "separate"})
    for task_id in ("aaaa0017", "aaaa0018"):
        _stop(tmp_path, task_id)
    before = {task_id: result_signature(tmp_path, task_id) for task_id in ("aaaa0017", "aaaa0018")}
    report = prep.recover_interrupted_preparations(tmp_path)
    assert "aaaa0017" in report["live"] and report["settled"] == [] and report["errors"] == [], report
    assert reconcile_promotion_intents(tmp_path, protected=set()) == {"not_started": [], "held": []}
    assert {task_id: result_signature(tmp_path, task_id) for task_id in before} == before


def test_startup_keeps_an_ended_claim_whose_staged_folders_remain(tmp_path, monkeypatch):
    write_task_result(tmp_path, "seed0000", "requested")
    claim_provisional_id(tmp_path, "staged01", "t-ended")
    (tmp_path / "state" / "headless_tasks" / "staged01" / "data").mkdir(parents=True)
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: True)
    assert task_id_claims.retire_settled_claims(tmp_path) == [] and _held(tmp_path, "staged01")
    shutil.rmtree(tmp_path / "state" / "headless_tasks" / "staged01")
    assert task_id_claims.retire_settled_claims(tmp_path) == ["staged01"]


def test_a_claimed_helper_receives_no_forwarded_message_until_its_request_is_bound(tmp_path):
    from ouroboros.tools.core import _forward_to_worker

    ctx, data = _ctx(tmp_path)
    write_task_result(data, "aaaa0001", "requested", parent_task_id="parent0", root_task_id="parent0",
                      delegation_role="subagent", request_claim={"token": "t", "phase": "claimed"})
    refused = _forward_to_worker(ctx, "aaaa0001", "hello")  # as get, peek and wait read it: no task yet
    assert "TASK_NOT_FOUND" in refused and "not registered" in refused
    write_task_result(data, "aaaa0001", "requested", request_claim={"token": "t", "phase": "requested"})
    assert _forward_to_worker(ctx, "aaaa0001", "hello").startswith("Message forwarded to task aaaa0001")


def test_a_claim_being_prepared_is_no_task_to_any_public_reader_and_stays_whole_for_custody(tmp_path):
    import json

    from ouroboros import task_store
    from ouroboros.native_task_wait import observe
    from ouroboros.task_status import find_child_tasks
    from ouroboros.tools import control_task_results as results
    from ouroboros.tools.join_ledger import _peek_task
    from ouroboros.tools.recent_tasks import recent_tasks_page
    from tests._shared import write_queue_document

    ctx, data = _ctx(tmp_path)
    ctx.task_metadata = {"budget_drive_root": str(data), "root_task_id": "parent0"}
    write_queue_document(data / "state" / "queue_snapshot.json", {"pending": [], "running": []})
    kin = {"parent_task_id": "parent0", "root_task_id": "parent0", "delegation_role": "subagent"}
    write_task_result(data, "aaaa0001", "requested", **kin, request_claim={"token": "t1", "phase": "claimed"})
    write_task_result(data, "aaaa0002", "requested", **kin, request_claim={"token": "t2", "phase": "requested"})
    write_task_result(data, "aaaa0003", "completed", result="an imaginary answer", **kin)  # a row before tokens
    public = ["aaaa0002", "aaaa0003"]
    # Lineage readers (join, finalization, owner-word carriage, logs) and the repair roster read the sent rows only.
    assert sorted(row["task_id"] for row in find_child_tasks(data, parent_task_id="parent0", scope="direct")) == public
    roster = results._children_roster_projection(ctx, data)
    assert sorted(item["task_id"] for item in roster["children_roster"]) == public
    assert roster["children_roster_omitted"] == 0
    # A page leaves the claim out before its limit: a page of two holds both tasks.
    assert sorted(item["name"] for item in task_store.inventory(data)) == public
    page = recent_tasks_page(data, limit=2)
    assert page["total"] == 2 and sorted(task["task_id"] for task in page["tasks"]) == public
    # get, peek, both waits, the sibling advisory and the unminted probe: an id not yet registered.
    assert "unknown or not yet registered" in results._get_task_result(ctx, "aaaa0001")
    assert "unknown or not yet registered" not in results._get_task_result(ctx, "aaaa0002")
    assert _peek_task(ctx, "aaaa0001").startswith("Task aaaa0001 [unknown]")
    assert _peek_task(ctx, "aaaa0002").startswith("Task aaaa0002 [requested]")
    waited = results._wait_for_tasks(ctx, ["aaaa0001", "aaaa0002"], timeout_sec=0)
    batch = json.loads(waited)
    assert batch["tasks"]["aaaa0001"]["unknown_task_id"] is True and batch["tasks"]["aaaa0002"]["status"] == "requested"
    assert sorted(item["task_id"] for item in batch["children_roster"]) == public and '"t1"' not in waited
    assert results._unminted_wait_ids(ctx, data, ["aaaa0001", "aaaa0002"]) == ["aaaa0001"]
    assert results._count_live_sibling_children(ctx, data, exclude_task_id="aaaa0003") == 1
    rows, unknown, _contended = observe(ctx, ["aaaa0001", "aaaa0002"])
    assert rows["aaaa0001"] == {} and unknown == {"aaaa0001": "source_missing"}
    assert rows["aaaa0002"]["status"] == "requested"
    # Custody and recovery read the stored rows themselves.
    assert load_task_result(data, "aaaa0001")["request_claim"] == {"token": "t1", "phase": "claimed"}
    assert sorted(task_store.task_facts(data, where="parent_task_id=?", params=("parent0",))) == ["aaaa0001", *public]
    # Its hand-off makes it a task to every reader at once.
    write_task_result(data, "aaaa0001", "requested", request_claim={"token": "t1", "phase": "requested"})
    assert len(find_child_tasks(data, parent_task_id="parent0", scope="direct")) == 3
    assert len(task_store.inventory(data)) == 3 and "aaaa0001" not in results._unminted_wait_ids(ctx, data, ["aaaa0001"])
    assert _peek_task(ctx, "aaaa0001").startswith("Task aaaa0001 [requested]")


def _reject_recorder(monkeypatch):
    calls = []
    monkeypatch.setattr("supervisor.events._reject_schedule_task", lambda *a, **k: calls.append(k) or True)
    return calls


def test_admission_refuses_another_requests_row_aloud_and_replays_its_own(tmp_path, monkeypatch):
    from supervisor.task_admission import subagent_schedule_preflight

    calls = _reject_recorder(monkeypatch)
    before = _foreign(tmp_path, "deadbeef")
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, PENDING=[], RUNNING={})
    event = {"task_id": "deadbeef", "request_token": "new-request", "parent_task_id": "parent0", "role": "helper"}
    assert subagent_schedule_preflight(ctx, event, 1) is True
    assert calls[-1]["reason_code"] == "task_id_taken" and calls[-1]["persist_result"] is False
    assert result_signature(tmp_path, "deadbeef") == before
    row = {"task_id": "cafef00d", "status": "requested", "delegation_role": "subagent", "parent_task_id": "parent0",
           "request_claim": {"token": "mine", "phase": "requested"}}
    seed_result(tmp_path, "cafef00d", row)
    assert subagent_schedule_preflight(ctx, {**event, "task_id": "cafef00d", "request_token": "mine"}, 1) is False
    seed_result(tmp_path, "cafef00d", {**row, "status": "scheduled"})
    assert subagent_schedule_preflight(ctx, {**event, "task_id": "cafef00d", "request_token": "mine"}, 1) is True
    assert len(calls) == 1  # an exact replay of an admitted request is ignored, not refused


def test_the_first_step_refuses_another_requests_row_aloud_inline_and_on_the_lane(tmp_path, monkeypatch):
    from supervisor import workspace_preparation as prep
    from supervisor.task_admission import foreign_request_row, subagent_schedule_preflight

    calls = _reject_recorder(monkeypatch)
    monkeypatch.setattr("supervisor.events_schedule_task._reject_schedule_task",
                        lambda *a, **k: calls.append(k) or True)
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, PENDING=[], RUNNING={})
    legacy = {"task_id": "aaaa0001", "parent_task_id": "parent0", "role": "helper"}  # an older helper's: no token
    seed_result(tmp_path, "aaaa0001", {"task_id": "aaaa0001", "status": "scheduled", "delegation_role": "subagent",
                                       "parent_task_id": "parent0"})
    assert subagent_schedule_preflight(ctx, legacy, 1) is True and calls == []  # its own row: a replay, ignored
    others = {"aaaa0002": {"status": "requested", "delegation_role": "root",
                           "promotion_admission": {"status": "emitted", "routing_token": "r"}},
              "aaaa0003": {"status": "requested", "delegation_role": "subagent", "request_claim": {"token": "other"}}}
    for tid, row in others.items():
        seed_result(tmp_path, tid, {"task_id": tid, **row})
        before = result_signature(tmp_path, tid)
        for request in ({}, {"request_token": "mine"}):
            assert subagent_schedule_preflight(ctx, {**legacy, "task_id": tid, **request}, 1) is True
            assert calls[-1]["reason_code"] == "task_id_taken" and calls[-1]["persist_result"] is False
        seen = len(calls)
        prep._run(prep.PreparationClaim(task_id=tid, ctx=ctx, token="prep", generation=None, generation_token="",
                                        owner={}, event={**legacy, "task_id": tid},
                                        route={"delegation_role": "subagent"}))
        assert len(calls) == seen + 1 and calls[-1]["reason_code"] == "task_id_taken"
        assert result_signature(tmp_path, tid) == before  # no marker, receipt or refusal ever lands on it
    seed_result(tmp_path, "aaaa0004", {"task_id": "aaaa0004", "status": "requested", "delegation_role": "subagent",
                                       "request_claim": {"token": "mine", "phase": "requested"}})
    assert foreign_request_row(ctx, "aaaa0004", "mine") is False and foreign_request_row(ctx, "aaaa0005", "") is False


def test_a_lane_refusal_before_its_request_check_never_lands_on_another_requests_row(tmp_path, monkeypatch):
    from supervisor import queue, workspace_preparation as prep

    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    generation = SimpleNamespace(token="g", wait_store=lambda operation, **_kwargs: operation(),
                                 closed=lambda: False, on_executor=lambda: False, submit=lambda *_args: True)
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, send_with_budget=lambda *_args, **_kwargs: None)
    for tid, row, settled in (("aaaa0014", {"delegation_role": "subagent", "request_claim": {"token": "other"}}, False),
                              ("aaaa0015", {"delegation_role": "subagent", "request_claim": {"token": "mine"}}, True)):
        seed_result(tmp_path, tid, {"task_id": tid, "status": "requested", **row})
        queue.ADMISSION_RESERVATIONS[tid] = "prep"  # the lane's own memory claim, before any check or marker
        claim = prep.PreparationClaim(task_id=tid, ctx=ctx, token="prep", generation=generation, generation_token="g",
                                      owner={}, event={"request_token": "mine"}, route={"delegation_role": "subagent"})
        before = result_signature(tmp_path, tid)
        assert claim.refuse("stopped while waiting", "root_cancelled") is settled
        assert (result_signature(tmp_path, tid) == before) is not settled


def test_a_producer_never_sends_over_a_stop_or_recreates_a_released_claim(tmp_path, monkeypatch):
    from ouroboros.task_store import delete_document
    from ouroboros.tools import control_scheduling

    real = control_scheduling._build_child_subagent_contract
    data = tmp_path / "data"
    endings = iter([lambda tid: write_task_result(data, tid, "cancelled", result="Stopped by the owner."),
                    lambda tid: delete_document(data, tid)])

    def ends_meanwhile(spec):
        next(endings)(spec["tid"])
        return real(spec)

    monkeypatch.setattr(control_scheduling, "_build_child_subagent_contract", ends_meanwhile)
    _mint(monkeypatch, "cafef00d")
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")  # a Stop landed meanwhile
    assert ctx.pending_events == [] and "ended or changed" in result and "never reused" not in result, result
    row = load_task_result(data, "cafef00d")
    assert row["status"] == "cancelled" and row["result"] == "Stopped by the owner."
    assert (data / "state" / "headless_tasks" / "cafef00d").is_dir()  # the terminal record keeps its files
    _mint(monkeypatch, "0badf00d")
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")  # the record went meanwhile
    assert ctx.pending_events == [] and load_task_result(data, "0badf00d") is None, result  # never recreated
    assert "status: absent" in result and "its record is gone" in result
    assert (data / "state" / "headless_tasks" / "0badf00d").is_dir()  # unproven: left, and the id stays occupied


def _unsent(task_id, **fields):
    return {"task_id": task_id, "status": "requested", "delegation_role": "subagent", "parent_task_id": "parent0",
            "requested_by": {"pid": 11, "birth": 1, "host_pid": 22, "host_birth": 2},
            "request_claim": {"token": "t", "phase": "claimed"}, **fields}


def test_a_recovery_refuses_only_the_row_it_judged_and_never_recreates_a_released_one(tmp_path, monkeypatch):
    from supervisor import queue, workspace_preparation as prep

    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: True)
    judged = _unsent("aaaa0005")  # what the pass read
    moved = (_unsent("aaaa0005", request_claim={"token": "t", "phase": "requested"}),  # its producer completed it
             _unsent("aaaa0005", request_claim={"token": "other", "phase": "claimed"}),  # another request's row
             _unsent("aaaa0005", workspace_preparation={"token": "prep", "phase": "queued"}))  # a claim marked it
    for current in moved:
        seed_result(tmp_path, "aaaa0005", current)
        before = result_signature(tmp_path, "aaaa0005")
        assert prep._recover_unmarked(tmp_path, judged) == "retained"
        assert result_signature(tmp_path, "aaaa0005") == before
    assert prep._recover_unmarked(tmp_path, _unsent("aaaa0006")) == "retained"  # released meanwhile
    assert load_task_result(tmp_path, "aaaa0006") is None
    seed_result(tmp_path, "aaaa0007", _unsent("aaaa0007"))
    assert prep._recover_unmarked(tmp_path, _unsent("aaaa0007")) == "settled"
    assert load_task_result(tmp_path, "aaaa0007")["admission_outcome"] == "never_admitted"


def test_recovery_announces_no_refusal_for_a_row_retired_or_sent_meanwhile(tmp_path, monkeypatch):
    from supervisor import message_bus, queue, workspace_preparation as prep

    told = []
    monkeypatch.setattr(message_bus, "send_with_budget", lambda chat_id, text, **_kw: told.append((chat_id, text)))
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: True)
    helper = {"chat_id": 1, "root_task_id": "parent0", "role": "helper"}
    # Its producer retired the row this pass judged (its own call already said "not scheduled"): nothing is announced.
    assert prep._recover_unmarked(tmp_path, _unsent("aaaa0014", **helper)) == "retained"
    assert told == [] and load_task_result(tmp_path, "aaaa0014") is None
    moved = _unsent("aaaa0015", **helper, request_claim={"token": "t", "phase": "requested"})  # completed and sent
    seed_result(tmp_path, "aaaa0015", moved)
    assert prep._recover_unmarked(tmp_path, _unsent("aaaa0015", **helper)) == "retained" and told == []
    seed_result(tmp_path, "aaaa0016", _unsent("aaaa0016", **helper))
    assert prep._recover_unmarked(tmp_path, _unsent("aaaa0016", **helper)) == "settled"
    assert [chat for chat, _text in told] == [1] and "not scheduled" in told[0][1], told


def test_a_terminal_parent_settles_an_unsent_claim_and_leaves_its_maybe_live_producers_files(tmp_path, monkeypatch):
    from supervisor import queue, workspace_preparation as prep

    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: False)
    seed_result(tmp_path, "parent0", {"task_id": "parent0", "status": "completed"})
    seed_result(tmp_path, "aaaa0013", _unsent("aaaa0013", root_task_id="parent0", role="helper"))
    drive = tmp_path / "state" / "headless_tasks" / "aaaa0013" / "data"
    drive.mkdir(parents=True)
    assert prep.recover_interrupted_preparations(tmp_path)["settled"] == ["aaaa0013"]
    row = load_task_result(tmp_path, "aaaa0013")
    assert row["status"] == "failed" and row["reason_code"] == prep.REQUEST_ORPHANED and drive.is_dir()


def test_recovery_refuses_an_unsent_claim_once_its_producer_ended_and_never_another_protocols_row(tmp_path, monkeypatch):
    from supervisor import workspace_preparation

    ended = {"producer": True}
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended",
                        lambda pid, _birth, proof=None: ended["producer"] if pid == 11 else False)
    claimed = _unsent("aaaa0006")
    assert workspace_preparation._request_ended(tmp_path, claimed) == workspace_preparation.REQUEST_UNSENT
    # A complete request may still ride its live bus: it waits for both ends, as before.
    assert workspace_preparation._request_ended(tmp_path, {**claimed, "request_claim": {
        "token": "t", "phase": "requested"}}) == ""
    ended["producer"] = False
    assert workspace_preparation._request_ended(tmp_path, claimed) == ""
    seed_result(tmp_path, "parent0", {"task_id": "parent0", "status": "cancelled"})  # an ended parent settles it too
    assert workspace_preparation._request_ended(tmp_path, claimed) == workspace_preparation.REQUEST_ORPHANED
    assert workspace_preparation._request_ended(tmp_path, {**claimed, "request_claim": {
        "token": "t", "phase": "requested"}}) == ""  # a complete request still waits for its bus
    # The recovery pass reads every REQUESTED row; an intent or an API task is never a helper's to settle.
    ended["producer"] = True
    for other in ({"task_id": "aaaa0009", "status": "requested", "delegation_role": "root",
                   "requested_by": claimed["requested_by"], "promotion_admission": {"status": "emitted"}},
                  {"task_id": "aaaa0010", "status": "requested", "requested_by": claimed["requested_by"]}):
        seed_result(tmp_path, other["task_id"], other)
        before = result_signature(tmp_path, other["task_id"])
        assert workspace_preparation._recover_unmarked(tmp_path, other) == "live"
        assert result_signature(tmp_path, other["task_id"]) == before


def test_a_depth_refusal_records_its_attempt_under_its_own_fresh_id(tmp_path, monkeypatch):
    from ouroboros.tools.control_delegation import record_depth_limit_refusal

    data = tmp_path / "data"
    data.mkdir()
    before = _foreign(data, "deadbeef")
    (data / "task_drives" / "0ff1ce00").mkdir(parents=True)  # an earlier owner's direct drive
    _mint(monkeypatch, "deadbeef", "0ff1ce00", "cafef00d")
    ctx = SimpleNamespace(task_id="parent0", task_metadata={}, drive_root=str(data), budget_drive_root="")
    text = record_depth_limit_refusal(ctx, {"objective": "too deep", "role": "helper"}, {}, {},
                                      current_depth=3, new_depth=4, max_depth=3)
    assert "task_id=cafef00d" in text, text
    assert result_signature(data, "deadbeef") == before and load_task_result(data, "0ff1ce00") is None
    assert load_task_result(data, "cafef00d")["reason_code"] == "subtask_depth_limit"
    _mint(monkeypatch, "0badcafe")
    parentless = SimpleNamespace(task_id="", task_metadata={}, drive_root=str(data), budget_drive_root="")
    record_depth_limit_refusal(parentless, {"objective": "too deep", "role": "helper"}, {}, {},
                               current_depth=3, new_depth=4, max_depth=3)
    row = load_task_result(data, "0badcafe")  # a parentless attempt roots at its assigned id, its contract too
    assert row["root_task_id"] == "0badcafe" == row["task_contract"]["lineage"]["root_task_id"]
    _mint(monkeypatch, "feedface", "0badf00d")
    monkeypatch.setattr("ouroboros.task_results.write_task_result", lambda *a, **k: (_ for _ in ()).throw(OSError("late")))
    text = record_depth_limit_refusal(ctx, {"objective": "too deep", "role": "helper"}, {}, {},
                                      current_depth=3, new_depth=4, max_depth=3)
    assert "depth_refusal_evidence_unavailable" in text and "Whether task id feedface holds" in text
    assert "is unknown" in text and "never reused" not in text


def test_recovery_settles_an_interrupted_preparation_of_a_tokened_request(tmp_path, monkeypatch):
    from supervisor import queue, workspace_preparation as prep

    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: True)
    marker = {"token": "prep", "phase": "queued", "generation": "gone", "owner": {"pid": 11, "birth": "b"},
              "surface": "", "claimed_at": "2026-11-26T00:00:00+00:00"}
    seed_result(tmp_path, "aaaa0011", {
        "task_id": "aaaa0011", "status": "requested", "delegation_role": "subagent", "parent_task_id": "parent0",
        "root_task_id": "parent0", "role": "helper", "request_claim": {"token": "t", "phase": "requested"},
        "workspace_preparation": marker})
    unrelated = tmp_path / "state" / "headless_tasks" / "aaaa0011" / "data" / "notes.txt"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"written under this id by a writer that claims nothing")
    # A Stop pending on a marked row is that Stop's to settle: recovery writes nothing over it and moves no file.
    seed_result(tmp_path, "aaaa0021", {**_unsent("aaaa0021", root_task_id="parent0", role="helper"),
                                       "request_claim": {"token": "t2", "phase": "requested"},
                                       "workspace_preparation": {**marker, "token": "prep2"}})
    _stop(tmp_path, "aaaa0021")
    stopped = result_signature(tmp_path, "aaaa0021")
    report = prep.recover_interrupted_preparations(tmp_path)
    assert report["settled"] == ["aaaa0011"], report  # its own marker names its row: the refusal lands there
    assert "aaaa0021" in report["live"] and result_signature(tmp_path, "aaaa0021") == stopped
    row = load_task_result(tmp_path, "aaaa0011")
    assert row["status"] == "failed" and row["admission_outcome"] == "never_admitted"
    assert row["workspace_preparation"]["phase"] == "refused" and row["request_claim"]["token"] == "t"
    # Its id was claimed, which proves nothing about the rest of what lies under it: the folder stays whole.
    assert unrelated.read_bytes() == b"written under this id by a writer that claims nothing"


def test_a_stop_landing_during_a_recovered_refusal_keeps_the_row_its_stop_settles(tmp_path, monkeypatch):
    from supervisor import queue, workspace_preparation as prep

    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    marker = {"token": "prep", "phase": "queued", "generation": "gone", "owner": {"pid": 11, "birth": "b"}}
    row = {**_unsent("aaaa0022", root_task_id="parent0", role="helper"), "workspace_preparation": marker,
           "request_claim": {"token": "t", "phase": "requested"}}
    seed_result(tmp_path, "aaaa0022", row)
    real = prep._stop_interlock

    def stop_first(root, tid, stop):  # a Stop recorded after the pass judged the row, before its refusal reads
        _stop(root, tid)
        return real(root, tid, stop)

    monkeypatch.setattr(prep, "_stop_interlock", stop_first)
    before = result_signature(tmp_path, "aaaa0022")
    assert prep.PreparationClaim.recovered(tmp_path, row, marker).refuse("not scheduled", "test_refusal") is False
    assert result_signature(tmp_path, "aaaa0022") == before  # read under the Stop lock: nothing written over it


class _Request(SimpleNamespace):
    def __getattr__(self, name):  # a composed request's field this branch does not read stays empty
        return None


def test_an_unconfirmed_restart_barrier_is_no_helper_refusal_and_leaves_its_receipt_and_files(tmp_path, monkeypatch):
    from supervisor import events_schedule_task as schedule, task_admission

    receipt = {**_unsent("aaaa0025", root_task_id="parent0", role="helper"), "status": "scheduled",
               "request_claim": {"token": "t", "phase": "requested"},
               "workspace_preparation": {"token": "prep", "phase": "admitting"}}
    seed_result(tmp_path, "aaaa0025", receipt)
    staged = tmp_path / "state" / "headless_tasks" / "aaaa0025" / "data" / "input.txt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"staged input")
    before = result_signature(tmp_path, "aaaa0025")
    for name, value in (("_admitted_cognitive_route", lambda ctx, r: {}), ("_gate_view", lambda ctx, prep: ctx),
                        ("_active_cap_state", lambda *a: (False, "")), ("_named_notes_admission_refusal",
                                                                         lambda ctx, task: ""),
                        ("_build_scheduled_task_payload", lambda fields: {"id": fields["tid"]}),
                        ("_compose_subagent_text", lambda *a, **k: "text")):
        monkeypatch.setattr(schedule, name, value)
    monkeypatch.setattr(task_admission, "enqueue_subagent_with_scheduled_result", lambda ctx, task, **k: (
        {**task, "_admission_blocked": "restart_admission_unconfirmed"}, "", "", False))
    refused = []
    request = _Request(tid="aaaa0025", delegation_role="subagent", root_task_id="parent0", parent_id="parent0",
                       role="helper", desc="d", evt={}, intent={}, task_contract={}, depth=1, max_depth=3,
                       result_fields={}, direct_child_count=0, reject=lambda *a, **k: refused.append(k))
    schedule._admit_scheduled_payload(SimpleNamespace(DRIVE_ROOT=tmp_path, PENDING=[], RUNNING={}), None, request)
    assert refused == []  # unknown: no refusal row and no cleanup; the queue's recovery decides after the restart
    assert result_signature(tmp_path, "aaaa0025") == before and staged.read_bytes() == b"staged input"


def test_a_refused_request_keeps_what_lies_under_its_claimed_id_and_a_legacy_row_its_old_rollback(tmp_path,
                                                                                                 monkeypatch):
    from supervisor import events_schedule_task as schedule

    removed = []
    monkeypatch.setattr(schedule, "_remove_refused_child_drive", lambda _ctx, tid: removed.append(tid))
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, send_with_budget=lambda *a, **k: None)
    fields = {"parent_task_id": "parent0", "root_task_id": "parent0", "delegation_role": "subagent"}
    for tid, claim in (("aaaa0023", {"request_claim": {"token": "t", "phase": "requested"}}), ("aaaa0024", {})):
        seed_result(tmp_path, tid, {"task_id": tid, "status": "requested", **fields, **claim})
        assert schedule._reject_schedule_task(
            ctx, tid=tid, chat_id=None, delegation_role="subagent", parent_id="parent0", root_task_id="parent0",
            role="helper", result_fields=fields, detail="Subagent not scheduled: a test refusal.",
            reason_code="test_refusal", extra_fields={"admission_outcome": "never_admitted"})
        assert load_task_result(tmp_path, tid)["admission_outcome"] == "never_admitted"
    assert removed == ["aaaa0024"]  # by its id only for a row written before claims, as before


def test_recovery_settles_an_unsent_tokened_claim_as_unsent(tmp_path, monkeypatch):
    from supervisor import queue, workspace_preparation as prep

    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: pid == 11)
    seed_result(tmp_path, "aaaa0012", _unsent("aaaa0012", root_task_id="parent0", role="helper"))
    folder = tmp_path / "state" / "headless_tasks" / "aaaa0012" / "data" / "image.bin"
    folder.parent.mkdir(parents=True)
    folder.write_bytes(b"made by whoever drew this id")  # an unsent claim records no drive: nothing proves whose
    assert prep.recover_interrupted_preparations(tmp_path)["settled"] == ["aaaa0012"]
    row = load_task_result(tmp_path, "aaaa0012")
    assert row["status"] == "failed" and row["reason_code"] == prep.REQUEST_UNSENT
    assert folder.read_bytes() == b"made by whoever drew this id"  # recovery removes none of it


def test_a_requests_proof_token_never_reaches_the_public_or_the_authority_projection(tmp_path):
    import json

    from ouroboros.agent_startup_checks import task_result_authority_projection
    from ouroboros.outcomes import public_task_result
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    row = {"task_id": "aaaa0001", "status": "requested", "delegation_role": "subagent",
           "request_claim": {"token": "proof-token", "phase": "requested"}}
    public, authority = public_task_result(row), task_result_authority_projection(row)
    for projection in (public, authority):
        assert "request_claim" not in projection and "proof-token" not in json.dumps(projection)
        assert projection["task_id"] == "aaaa0001" and projection["status"] == "requested"
    # The tool reads through the same projection, every field selection included; the producer reads its row raw.
    (tmp_path / "repo").mkdir()
    write_task_result(tmp_path, "aaaa0001", "completed", result="an imaginary answer", parent_task_id="parent0",
                      root_task_id="parent0", delegation_role="subagent", terminal_origin="model_final",
                      request_claim={"token": "proof-token", "phase": "requested"})
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path, task_id="parent0")
    ctx.task_metadata = {"budget_drive_root": str(tmp_path), "root_task_id": "parent0"}
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    for args in ({"include_authority": True}, {"include_authority": True, "authority_fields": ["request_claim"]},
                 {"include_authority": True, "authority_fields": []}):
        text = registry.execute_result("get_task_result", {"task_id": "aaaa0001", **args}).text
        assert "proof-token" not in text, (args, text)  # a field asked for by name is echoed, never its value
        assert "request_claim" not in text or args.get("authority_fields"), (args, text)
    assert load_task_result(tmp_path, "aaaa0001")["request_claim"]["token"] == "proof-token"


def test_an_api_receipt_lands_only_on_its_own_record_outside_q_and_leaves_any_unproven_one_unchanged(
        tmp_path, monkeypatch):
    from ouroboros import task_results
    from supervisor import queue
    from supervisor.task_api_claim import land_api_receipt

    under_q = []
    real_load, real_write = task_results.load_task_result, task_results.write_task_result

    def load(*args, **kwargs):
        under_q.append(queue._queue_lock._is_owned())
        return real_load(*args, **kwargs)

    def write(*args, **kwargs):
        under_q.append(queue._queue_lock._is_owned())
        return real_write(*args, **kwargs)

    monkeypatch.setattr(task_results, "load_task_result", load)

    def receipt(token):
        return {"description": "x", "memory_mode": "empty", "result": "Task accepted and durably scheduled.",
                "api_admission": {"token": token, "status": "accepted"}}

    # No row yet: the receipt creates one only while this admission's own claim holds the id; its replay lands too.
    claim_provisional_id(tmp_path, "aaaa0030", "mine")
    for _replay in range(2):
        land_api_receipt(tmp_path, "aaaa0030", "mine", receipt("mine"), write)
    assert real_load(tmp_path, "aaaa0030")["status"] == "scheduled"
    claim_provisional_id(tmp_path, "aaaa0031", "theirs")
    for tid in ("aaaa0031", "aaaa0032"):  # another admission's claim, or none at all: nothing is created
        with pytest.raises(RuntimeError):
            land_api_receipt(tmp_path, tid, "mine", receipt("mine"), write)
        assert real_load(tmp_path, tid) is None
    # Its own task's first Stop wrote first, carrying its token: its words stand, the receipt adds what it lacks.
    real_write(tmp_path, "aaaa0033", "cancelled", result="Task cancelled by user/agent request.",
               _admission_owner_token="mine")
    land_api_receipt(tmp_path, "aaaa0033", "mine", receipt("mine"), write)
    row = real_load(tmp_path, "aaaa0033")
    assert row["status"] == "cancelled" and row["result"] == "Task cancelled by user/agent request."
    assert row["memory_mode"] == "empty" and row["api_admission"] == receipt("mine")["api_admission"]
    # A record without its token proves nothing of it, whatever its shape: unknown, and the record left as it is.
    for tid, fields in (("aaaa0034", {}), ("aaaa0035", {"_admission_owner_token": "theirs"}),
                        ("aaaa0036", {"api_admission": {"token": "theirs", "status": "accepted"}})):
        real_write(tmp_path, tid, "running", result="another writer's work", **fields)
        before = result_signature(tmp_path, tid)
        with pytest.raises(RuntimeError):
            land_api_receipt(tmp_path, tid, "mine", receipt("mine"), write)
        assert result_signature(tmp_path, tid) == before
    assert under_q and not any(under_q)  # every read and write of the receipt ran outside Q


# ---- the task API ------------------------------------------------------------------------------------------------

@pytest.fixture()
def api(tmp_path, monkeypatch):
    return _api_client(tmp_path, monkeypatch)


def _api_client(tmp_path, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.tasks import api_tasks_create
    from supervisor import queue, workers

    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    (data / "memory").mkdir(parents=True)
    (data / "memory" / "identity.md").write_text("a fictional identity", encoding="utf-8")
    monkeypatch.setattr(workers, "WORKERS", {0: SimpleNamespace()})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    captured = []
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: captured.append(dict(task)) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])
    monkeypatch.setattr("supervisor.message_bus.try_get_bridge", lambda: None)
    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root, app.state.repo_dir = data, repo
    return TestClient(app), data, captured


def test_the_task_api_mints_again_past_held_ids_and_refuses_a_fixed_one_unchanged(api, monkeypatch):
    from ouroboros.gateway import tasks

    client, data, captured = api
    before = _foreign(data, "deadbeefdeadbeef")
    claim_provisional_id(data, "feedfacefeedface", "another-admission")  # another request's claim, still no row
    _mint(monkeypatch, "deadbeefdeadbeef", "feedfacefeedface", "cafef00dcafef00d")
    response = client.post("/api/tasks", json={"description": "a new root"})
    assert response.status_code == 200, response.text
    assert response.json()["task_id"] == "cafef00dcafef00d" == captured[-1]["id"]
    assert load_task_result(data, "cafef00dcafef00d")["status"] == "scheduled"
    assert _held(data, "cafef00dcafef00d") is None  # its receipt row holds the id now
    assert result_signature(data, "deadbeefdeadbeef") == before
    assert _held(data, "feedfacefeedface")["token"] == "another-admission"
    # A caller's id whose record appears after the early probe is refused, that record unchanged.
    monkeypatch.setattr(tasks, "_task_identity_occupied", lambda *_args: False)
    refused = client.post("/api/tasks", json={"description": "x", "task_id": "deadbeefdeadbeef"})
    assert refused.status_code == 409 and refused.json()["reason_code"] == "duplicate_task_id"
    taken = client.post("/api/tasks", json={"description": "x", "task_id": "feedfacefeedface"})
    assert taken.status_code == 409 and taken.json()["reason_code"] == "duplicate_task_id"
    assert result_signature(data, "deadbeefdeadbeef") == before and len(captured) == 1


def test_a_proved_rollback_frees_the_id_and_an_unproven_one_keeps_it_held(api, monkeypatch):
    from ouroboros.gateway import tasks
    from supervisor import queue

    client, data, captured = api
    prepare = tasks.prepare_task_drive

    def failing(*_args, **_kwargs):
        raise OSError("no room for the drive")

    monkeypatch.setattr(tasks, "prepare_task_drive", failing)
    first = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-1"})
    assert first.status_code == 503 and "status" not in first.json(), first.text
    assert load_task_result(data, "explicit-1") is None and _held(data, "explicit-1") is None
    assert "explicit-1" not in queue.ADMISSION_RESERVATIONS
    monkeypatch.setattr(tasks, "prepare_task_drive", prepare)
    assert client.post("/api/tasks", json={"description": "x", "task_id": "explicit-1"}).status_code == 200
    monkeypatch.setattr(tasks, "prepare_task_drive", failing)
    monkeypatch.setattr(task_id_claims, "release_provisional_claim", lambda *_args, **_kwargs: False)
    unconfirmed = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-2"})
    assert unconfirmed.status_code == 503, unconfirmed.text
    assert unconfirmed.json()["status"] == "unconfirmed" and unconfirmed.json()["task_id"] == "explicit-2"
    monkeypatch.setattr(tasks, "prepare_task_drive", prepare)
    again = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-2"})
    assert again.status_code == 409, again.text  # never silently permitted
    assert [task["id"] for task in captured] == ["explicit-1"]


def test_the_task_api_holds_no_reservation_while_the_store_decides_so_a_helper_winners_event_lands(api,
                                                                                                 monkeypatch):
    from ouroboros.gateway import tasks
    from supervisor import queue
    from supervisor.task_admission import subagent_schedule_preflight

    client, data, captured = api
    seed_result(data, "feedface", {"task_id": "feedface", "status": "requested", "delegation_role": "subagent",
                                   "request_claim": {"token": "t", "phase": "requested"}})
    monkeypatch.setattr(tasks, "_task_identity_occupied", lambda *_args: False)  # the helper won after the probe
    seen, real = [], task_id_claims.claim_provisional_id

    def deciding(root, tid, token):
        ctx = SimpleNamespace(DRIVE_ROOT=data, PENDING=[], RUNNING={})
        seen.append((dict(queue.ADMISSION_RESERVATIONS),
                     subagent_schedule_preflight(ctx, {"task_id": "feedface", "request_token": "t"}, 1)))
        return real(root, tid, token)

    monkeypatch.setattr(task_id_claims, "claim_provisional_id", deciding)
    response = client.post("/api/tasks", json={"description": "x", "task_id": "feedface"})
    assert response.status_code == 409 and response.json()["reason_code"] == "duplicate_task_id", response.text
    assert seen == [({}, False)]  # the winner's only event is not taken for a replay
    assert queue.ADMISSION_RESERVATIONS == {} and captured == []


def test_an_unconfirmed_claim_or_rollback_answers_a_typed_503_naming_the_id(api, monkeypatch):
    from ouroboros.gateway import tasks
    from supervisor import queue

    client, data, captured = api
    real = task_id_claims.claim_provisional_id

    def commit_then_fail(root, tid, token):
        real(root, tid, token)
        raise OSError("the store answered too late")

    with monkeypatch.context() as fault:
        fault.setattr(task_id_claims, "claim_provisional_id", commit_then_fail)
        response = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-3"})
    assert response.status_code == 503, response.text
    assert response.json()["status"] == "unconfirmed" and response.json()["task_id"] == "explicit-3"
    assert load_task_result(data, "explicit-3") is None and _held(data, "explicit-3")

    def loses_ownership(*_args, **_kwargs):
        queue.ADMISSION_RESERVATIONS.clear()  # the rollback cannot be proved its own any more
        raise OSError("no room for the drive")

    def foreign_artifacts(root, tid, *_args, **_kwargs):
        (root / "task_results" / "artifacts" / tid).mkdir(parents=True)  # nothing of this admission staged there
        raise OSError("no room for the drive")

    monkeypatch.setattr(tasks, "prepare_task_drive", loses_ownership)
    response = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-4"})
    assert response.status_code == 503 and response.json()["status"] == "unconfirmed", response.text
    assert _held(data, "explicit-4") and captured == []
    # A folder this admission did not make is never its to remove: its own part undone, it lets the id go and says
    # what holds it now, instead of keeping a provably clean rollback held.
    monkeypatch.setattr(tasks, "prepare_task_drive", foreign_artifacts)
    occupied = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-7"})
    assert occupied.status_code == 409 and occupied.json()["reason_code"] == "task_id_occupied", occupied.text
    assert _held(data, "explicit-7") is None and "explicit-7" not in queue.ADMISSION_RESERVATIONS and captured == []
    assert (data / "task_results" / "artifacts" / "explicit-7").is_dir()


def _miss(root, monkeypatch, task_id):
    """The Stop's custody of an id neither queued nor running (``_finalize_cancel_intent_on_miss``)."""
    from ouroboros.cancel_intents import active_intent
    from supervisor import cancel_publication, queue, terminal_delivery

    monkeypatch.setattr(queue, "DRIVE_ROOT", root)
    monkeypatch.setattr(terminal_delivery, "deliver_miss_lane_outcome", lambda *a, **kw: True, raising=False)
    return cancel_publication._finalize_cancel_intent_on_miss(queue, task_id, intent=active_intent(root, task_id))


def test_a_stop_before_the_hand_off_queues_nothing_and_a_clean_rollback_lets_the_id_go(api, monkeypatch):
    from ouroboros.gateway import tasks
    from supervisor import cancel_publication, queue

    client, data, captured = api
    monkeypatch.setattr(queue, "task_settlement_liveness", lambda _tid: False)  # a server's proven-not-live answer
    attach = tasks.attach_task_contract

    def stopped_meanwhile(task):
        _stop(data, task["id"])
        return attach(task)

    body = {"description": "x", "task_id": "explicit-12"}  # nothing staged: what a drive holds would stay (below)
    with monkeypatch.context() as stopping:
        stopping.setattr(tasks, "attach_task_contract", stopped_meanwhile)
        response = client.post("/api/tasks", json=body)
    assert response.status_code == 409 and response.json()["reason_code"] == "task_id_stop_pending", response.text
    assert captured == [] and _held(data, "explicit-12") is None and "explicit-12" not in queue.ADMISSION_RESERVATIONS
    assert not (data / "state" / "headless_tasks" / "explicit-12").exists()
    again = client.post("/api/tasks", json=body)  # the pending Stop still takes the id
    assert again.status_code == 409 and again.json()["reason_code"] == "task_id_stop_pending", again.text
    assert _miss(data, monkeypatch, "explicit-12") == cancel_publication.CANCEL_NOT_FOUND  # nothing held it any more
    assert client.post("/api/tasks", json=body).status_code == 200 and [t["id"] for t in captured] == ["explicit-12"]


def test_a_stop_that_finds_no_row_fences_an_admission_before_its_hand_off_and_waits_after_it(tmp_path, monkeypatch):
    from ouroboros.cancel_intents import active_intent
    from supervisor import cancel_publication as cp

    write_task_result(tmp_path, "seed0000", "requested")
    claim_provisional_id(tmp_path, "held0001", "t-http")
    _stop(tmp_path, "held0001")
    assert _miss(tmp_path, monkeypatch, "held0001") == cp.CANCEL_CANCELLED  # nothing was queued, nothing will be
    assert load_task_result(tmp_path, "held0001") is None  # and no row was invented for it
    assert _held(tmp_path, "held0001")["phase"] == "stopped" and active_intent(tmp_path, "held0001") is None
    assert advance_provisional_claim(tmp_path, "held0001", "t-http", "handoff") == "stopped"  # it cannot pass
    claim_provisional_id(tmp_path, "held0002", "t-http")
    assert advance_provisional_claim(tmp_path, "held0002", "t-http", "handoff") == ""
    _stop(tmp_path, "held0002")
    assert _miss(tmp_path, monkeypatch, "held0002") == cp.CANCEL_FAILED  # it may be queuing the task: pending
    assert active_intent(tmp_path, "held0002") and _held(tmp_path, "held0002")["phase"] == "handoff"
    _stop(tmp_path, "nothing1")
    assert _miss(tmp_path, monkeypatch, "nothing1") == cp.CANCEL_NOT_FOUND  # a genuinely absent id keeps its 404
    # A receipt lands and retires its claim between the row read and the claim read: the row is read again.
    claim_provisional_id(tmp_path, "held0003", "t-http")
    _stop(tmp_path, "held0003")
    real = task_id_claims.stop_provisional_claim

    def receipted_meanwhile(root, tid):
        write_task_result(root, tid, "scheduled", api_admission={"token": "t-http", "status": "accepted"})
        task_id_claims.retire_provisional_claim(root, tid, "t-http")
        return real(root, tid)

    monkeypatch.setattr(task_id_claims, "stop_provisional_claim", receipted_meanwhile)
    monkeypatch.setattr("supervisor.cancel_publication._audit_delegated_runs_on_kill", lambda *a, **k: {})
    assert _miss(tmp_path, monkeypatch, "held0003") != cp.CANCEL_NOT_FOUND
    assert load_task_result(tmp_path, "held0003")["status"] == "cancelled"


def test_a_helper_request_still_being_prepared_is_not_a_public_task(api, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.task_events import api_task_events
    from ouroboros.gateway.tasks import _tasks_list_payload, api_task_get

    _client, data, _captured = api
    write_task_result(data, "older001", "completed", ts="2026-11-01T00:00:00+00:00")
    write_task_result(data, "older002", "completed", ts="2026-11-02T00:00:00+00:00")
    claim = {"token": "t", "phase": "claimed"}
    write_task_result(data, "prepared", "requested", delegation_role="subagent", request_claim=claim,
                      ts="2026-11-03T00:00:00+00:00")
    app = Starlette(routes=[Route("/api/tasks/{task_id}", endpoint=api_task_get, methods=["GET"]),
                            Route("/api/tasks/{task_id}/events", endpoint=api_task_events, methods=["GET", "POST"])])
    app.state.drive_root = data
    reader = TestClient(app)
    assert reader.get("/api/tasks/prepared").status_code == 404
    assert reader.get("/api/tasks/prepared/events?wait=0").status_code == 404
    assert reader.post("/api/tasks/prepared/events", json={"v": 2, "wait": 0}).status_code == 404

    def listed(wanted=(), limit=2):
        return [row["task_id"] for row in _tasks_list_payload(data, set(wanted), limit, False)["tasks"]]

    assert listed() == ["older002", "older001"]  # the page stays full of public rows
    assert listed(("requested",), None) == [] and listed(("requested", "completed")) == ["older002", "older001"]
    # Internal readers and recovery still see the claim; once the request is completed it is public again.
    assert load_task_result(data, "prepared")["request_claim"] == claim
    write_task_result(data, "prepared", "requested", request_claim={"token": "t", "phase": "requested"})
    assert listed() == ["prepared", "older002"] and reader.get("/api/tasks/prepared").status_code == 200


def test_an_api_claim_is_retired_only_once_a_record_holds_the_id(api, monkeypatch):
    client, data, _captured = api
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda _task: {"_admission_blocked": "root_budget_fence"})
    refused = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-6"})
    assert refused.status_code == 409 and load_task_result(data, "explicit-6")["status"] == "failed", refused.text
    assert _held(data, "explicit-6") is None


@pytest.mark.parametrize("refusal", ["task_id_lookup_failed", "admission_reservation_lost", "unrecorded"])
def test_a_refusal_that_queued_and_recorded_nothing_frees_the_same_id(api, monkeypatch, refusal):
    from supervisor import queue, task_admission

    client, data, captured = api
    task_id, reason = "explicit-5", "root_budget_fence" if refusal == "unrecorded" else refusal

    def blocked(task):
        queue.ADMISSION_RESERVATIONS.pop(task["id"], None)  # the queue releases it with these refusals
        return {**task, "_admission_blocked": reason}

    with monkeypatch.context() as fault:
        fault.setattr("supervisor.queue.enqueue_task", blocked)
        if refusal == "unrecorded":
            fault.setattr(task_admission, "persist_never_admitted_refusal",
                          lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("the refusal could not be saved")))
        response = client.post("/api/tasks", json={"description": "x", "task_id": task_id})
    assert response.status_code == 503 and "status" not in response.json(), response.text
    assert response.json()["reason_code"] == reason and "same task_id may be sent again" in response.json()["error"]
    assert load_task_result(data, task_id) is None and _held(data, task_id) is None
    assert task_id not in queue.ADMISSION_RESERVATIONS
    assert client.post("/api/tasks", json={"description": "x", "task_id": task_id}).status_code == 200
    assert [task["id"] for task in captured] == [task_id]


@pytest.mark.parametrize("ambiguity", ["restart_barrier", "unreadable", "competing"])
def test_an_ambiguous_or_competing_refusal_keeps_everything_held(api, monkeypatch, ambiguity):
    from supervisor import queue, task_admission

    client, data, captured = api
    task_id = "explicit-8"

    def blocked(task):
        if ambiguity == "competing":
            queue.ADMISSION_RESERVATIONS[task["id"]] = "another-admission"
            return {**task, "_admission_blocked": "admission_reservation_owned"}
        if ambiguity == "unreadable":
            queue.ADMISSION_RESERVATIONS.pop(task["id"], None)
            unreadable_result(monkeypatch, task["id"])
            return {**task, "_admission_blocked": "task_id_lookup_failed"}
        return {**task, "_admission_blocked": "restart_admission_unconfirmed"}

    monkeypatch.setattr("supervisor.queue.enqueue_task", blocked)
    written = []
    if ambiguity == "restart_barrier":  # a partial publication may boot the row: no refusal may be written at all
        monkeypatch.setattr(task_admission, "persist_never_admitted_refusal",
                            lambda *_a, **_k: written.append("never_admitted"))
        monkeypatch.setattr("ouroboros.gateway.tasks.write_task_result", lambda *_a, **_k: written.append("failed"))
    response = client.post("/api/tasks", json={"description": "x", "task_id": task_id, "memory_mode": "empty"})
    expected = 409 if ambiguity == "competing" else 503
    assert response.status_code == expected, response.text
    assert ambiguity == "competing" or response.json()["status"] == "unconfirmed"
    assert _held(data, task_id)["phase"] == "handoff" and captured == [] and written == []
    assert (data / "state" / "headless_tasks" / task_id / "data").is_dir()  # what it staged stays for recovery
    holder = queue.ADMISSION_RESERVATIONS.get(task_id)
    assert holder == "another-admission" if ambiguity == "competing" else holder


def test_a_refusal_over_another_records_generated_id_neither_names_it_nor_keeps_it_reserved(api, monkeypatch):
    from supervisor import queue

    client, data, captured = api
    _mint(monkeypatch, "cafef00dcafef00d")

    def occupied_meanwhile(task):
        _foreign(data, task["id"])  # a writer that claims nothing wrote it before the queue refused
        queue.ADMISSION_RESERVATIONS.pop(task["id"], None)
        return {**task, "_admission_blocked": "task_id_lookup_failed"}

    monkeypatch.setattr("supervisor.queue.enqueue_task", occupied_meanwhile)
    response = client.post("/api/tasks", json={"description": "x"})
    assert response.status_code == 409 and response.json()["reason_code"] == "task_id_occupied", response.text
    assert "task_id" not in response.json() and "cafef00d" not in response.text and captured == []
    assert result_signature(data, "cafef00dcafef00d") and "cafef00dcafef00d" not in queue.ADMISSION_RESERVATIONS
    assert _held(data, "cafef00dcafef00d") is None  # the record under it holds the id now


def test_refused_attachments_keep_their_422_and_manifest_and_what_was_staged_stays_with_its_recorded_refusal(
        api, monkeypatch, tmp_path):
    from starlette.routing import Route

    from ouroboros.gateway.tasks import api_task_get
    from ouroboros.headless import prepared_drive_mode, task_state_dir
    from supervisor import queue

    client, data, captured = api
    client.app.router.routes.append(Route("/api/tasks/{task_id}", endpoint=api_task_get, methods=["GET"]))
    body = {"description": "x", "task_id": "explicit-9", "memory_mode": "empty", "allow_partial_attachments": False,
            "attachments": [{"path": str(tmp_path / "absent.txt"), "label": "absent"}]}
    refused = client.post("/api/tasks", json=body)  # another writer may have written into its drive: none of it goes
    answer = refused.json()
    assert refused.status_code == 422 and answer["reason_code"] == "attachment_admission_rejected", refused.text
    assert answer["status"] == "rejected" and answer["attachment_manifest"] == [
        {"ordinal": 0, "status": "rejected", "reason": "source_missing", "label": "absent"}]
    assert answer["staging_retained"] is True and "sending the task again takes another task_id" in answer["error"]
    assert prepared_drive_mode(task_state_dir(data, "explicit-9") / "data", "explicit-9") == "empty"  # untouched
    row = load_task_result(data, "explicit-9")  # its own proven refusal keeps the id, for that row's retention
    assert row["status"] == "failed" and row["admission_outcome"] == "never_admitted"
    assert row["reason_code"] == "attachment_admission_rejected" and row["admission_cleanup"] == {"staging_retained": True}
    assert _held(data, "explicit-9") is None and "explicit-9" not in queue.ADMISSION_RESERVATIONS
    shown = client.get("/api/tasks/explicit-9")
    assert shown.status_code == 200 and shown.json()["status"] == "failed", shown.text
    again = client.post("/api/tasks", json={**body, "attachments": []})  # the id is its refusal's now
    assert again.status_code == 409, again.text
    assert captured == [] and load_task_result(data, "explicit-9")["reason_code"] == "attachment_admission_rejected"


def test_a_preparation_that_stopped_before_its_drive_leaves_no_hold(api, monkeypatch):
    from ouroboros.gateway import tasks
    from ouroboros.headless import task_state_dir

    client, data, _captured = api

    def stops_early(root, task_id, *_args, **_kwargs):
        task_state_dir(root, task_id).mkdir(parents=True, exist_ok=True)  # the id's folder, no drive under it
        raise OSError("no room for the drive")

    monkeypatch.setattr(tasks, "prepare_task_drive", stops_early)
    response = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-10", "memory_mode": "empty"})
    assert response.status_code == 503 and "status" not in response.json(), response.text
    assert not task_state_dir(data, "explicit-10").exists() and _held(data, "explicit-10") is None


def test_a_generated_id_that_is_held_or_cannot_be_checked_is_never_named(api, monkeypatch):
    client, data, captured = api
    _foreign(data, "deadbeefdeadbeef")
    monkeypatch.setattr(task_id_claims, "mint_task_id", lambda _width: "deadbeefdeadbeef")
    exhausted = client.post("/api/tasks", json={"description": "x"})
    assert exhausted.status_code == 503 and exhausted.json()["reason_code"] == "task_id_exhausted", exhausted.text
    assert "task_id" not in exhausted.json() and "deadbeef" not in exhausted.text
    (data / "task_results" / "artifacts" / "feedfacefeedface").mkdir(parents=True)
    monkeypatch.setattr(task_id_claims, "mint_task_id", lambda _width: "feedfacefeedface")
    occupied_by_folders = client.post("/api/tasks", json={"description": "x"})
    assert occupied_by_folders.status_code == 503 and "feedface" not in occupied_by_folders.text
    real = task_id_claims.present

    def denied(path):
        if path.name == "cafef00dcafef00d":
            raise PermissionError(13, "denied", str(path))
        return real(path)

    monkeypatch.setattr(task_id_claims, "present", denied)
    monkeypatch.setattr(task_id_claims, "mint_task_id", lambda _width: "cafef00dcafef00d")
    unchecked = client.post("/api/tasks", json={"description": "x"})
    assert unchecked.status_code == 503 and unchecked.json()["reason_code"] == "task_id_inspection_failed"
    assert "cafef00d" not in unchecked.text and captured == []


def test_a_caller_id_the_store_cannot_answer_for_is_reported_unchecked_not_existing(api, monkeypatch):
    client, data, captured = api
    write_task_result(data, "seed0000", "requested")
    unreadable_result(monkeypatch, "explicit-11")
    response = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-11"})
    # Still the pinned 409 (never read as free), but the answer says it was not checked rather than known to exist.
    assert response.status_code == 409 and response.json()["reason_code"] == "task_id_inspection_failed", response.text
    assert "could not be checked" in response.json()["error"] and captured == []


# ---- start_task --------------------------------------------------------------------------------------------------

def _start_ctx(tmp_path):
    return SimpleNamespace(pending_events=[], event_queue=None, current_chat_id=1, drive_root=tmp_path, project_id="",
                           task_metadata={"force_plan": True, "force_plan_source": "swarm"})


def _start_ready(monkeypatch, receipt=None):
    monkeypatch.setattr("ouroboros.tools.control_routing._promotion_pool_disabled_from_snapshot", lambda _ctx: "")
    monkeypatch.setattr("ouroboros.tools.control_events._wait_for_promotion_admission",
                        lambda *_args, **_kwargs: dict(receipt or {"status": "scheduled"}))


def test_start_task_writes_its_intent_once_under_its_claimed_id_and_emits_exactly_it(tmp_path, monkeypatch):
    from ouroboros import promotion_intent
    from ouroboros.tools.control import _start_task

    _start_ready(monkeypatch)
    before = _foreign(tmp_path, "deadbeefdeadbeef")
    _mint(monkeypatch, "deadbeefdeadbeef", "cafef00dcafef00d")
    writes, real = [], promotion_intent.write_task_result
    monkeypatch.setattr(promotion_intent, "write_task_result",
                        lambda *args, **fields: writes.append(args[1]) or real(*args, **fields))
    ctx = _start_ctx(tmp_path)
    out = _start_task(ctx, "a separate work", title="Separate work")
    assert out.startswith("OK: task cafef00dcafef00d") and "deadbeef" not in out, out
    [event] = ctx.pending_events
    assert event["task_id"] == "cafef00dcafef00d" and result_signature(tmp_path, "deadbeefdeadbeef") == before
    assert writes == ["deadbeefdeadbeef", "cafef00dcafef00d"]  # the taken id refused, then the one intent write
    saved = load_task_result(tmp_path, "cafef00dcafef00d")["promotion_intent"]
    assert {key: event[key] for key in saved} == saved  # what is emitted is exactly what was saved


def test_the_same_start_replays_its_own_intent_without_another_id(tmp_path, monkeypatch):
    from ouroboros.promotion_intent import claim_new_start

    _mint(monkeypatch, "cafef00dcafef00d")  # one id only: a second acquisition would have none to take
    event = {"type": "promote_chat_to_task", "routing_token": "route-1", "objective": "a separate work",
             "ts": "2026-11-26T00:00:00+00:00"}
    first = claim_new_start(tmp_path, event)
    assert event["task_id"] == "cafef00dcafef00d" and first["promotion_intent"] == event
    stored = result_signature(tmp_path, "cafef00dcafef00d")
    replay = {**event, "ts": "2026-11-26T01:00:00+00:00"}
    replayed = claim_new_start(tmp_path, replay)
    assert replayed["promotion_admission"]["routing_token"] == "route-1"
    assert replayed["promotion_intent"] == first["promotion_intent"]  # the saved intent, exactly, with its first time
    assert result_signature(tmp_path, "cafef00dcafef00d") == stored
    with pytest.raises(RuntimeError, match="not persisted"):  # the same token with other words is no replay
        claim_new_start(tmp_path, {**event, "objective": "another work"})
    # Another start naming that id is no replay: it is refused, never moved to a fresh id, the intent unchanged.
    other = {**event, "routing_token": "route-2"}
    with pytest.raises(TaskIdTaken):
        claim_new_start(tmp_path, other)
    assert other["task_id"] == "cafef00dcafef00d" and result_signature(tmp_path, "cafef00dcafef00d") == stored


def test_a_start_whose_claim_outcome_is_unknown_names_its_id_and_emits_nothing(tmp_path, monkeypatch):
    from ouroboros import promotion_intent
    from ouroboros.tools.control import _start_task

    _start_ready(monkeypatch)
    real = promotion_intent.prepare_promotion_intent

    def answered_too_late(root, event, **kwargs):
        real(root, event, **kwargs)
        raise OSError("the store answered too late")

    monkeypatch.setattr(promotion_intent, "prepare_promotion_intent", answered_too_late)
    minted = _mint(monkeypatch, "cafef00dcafef00d", "0badf00d0badf00d")
    ctx = _start_ctx(tmp_path)
    out = _start_task(ctx, "a separate work", title="Separate work")
    assert out.startswith("⚠️ PROMOTE_UNCONFIRMED: task id cafef00dcafef00d"), out
    assert ctx.pending_events == [] and next(minted) == "0badf00d0badf00d"  # never moved to another id


def test_a_start_refused_before_its_claim_names_no_id(tmp_path, monkeypatch):
    from ouroboros.tools.control import _start_task

    monkeypatch.setattr("ouroboros.tools.control_routing._promotion_pool_disabled_from_snapshot", lambda _ctx: "down")
    minted = _mint(monkeypatch, "cafef00dcafef00d")
    ctx = _start_ctx(tmp_path)
    out = _start_task(ctx, "a separate work", title="Separate work")
    assert out.startswith("⚠️ PROMOTE_REJECTED: the task was not scheduled"), out
    assert ctx.pending_events == [] and next(minted) == "cafef00dcafef00d"  # nothing was minted or shown


# ---- wave-4 corrections: unheld refusals, a parent's view, attributable cleanup, owner carriage, recovery ---------

def test_a_caller_id_the_claim_could_not_check_answers_409_and_a_generated_one_is_never_named(api, monkeypatch):
    from ouroboros.store import StoreBusy, Txn

    client, data, captured = api
    real = Txn.get_row

    def busy(self, set_name, key):
        if set_name == ID_CLAIMS:  # the claim's first read, before anything is written or reserved
            raise StoreBusy("the store is busy", reason="contention")
        return real(self, set_name, key)

    monkeypatch.setattr(Txn, "get_row", busy)
    named = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-13"})
    assert named.status_code == 409 and named.json()["reason_code"] == "task_id_inspection_failed", named.text
    assert "status" not in named.json() and "could not be checked" in named.json()["error"]
    _mint(monkeypatch, "cafef00dcafef00d")
    generated = client.post("/api/tasks", json={"description": "x"})
    assert generated.status_code == 503 and generated.json()["reason_code"] == "task_id_inspection_failed"
    assert "cafef00d" not in generated.text and "status" not in generated.json()
    monkeypatch.setattr(Txn, "get_row", real)
    assert _held(data, "explicit-13") is None and _held(data, "cafef00dcafef00d") is None and captured == []


def test_a_helper_still_being_prepared_never_changes_its_parents_event_view(tmp_path):
    from ouroboros.gateway.task_events import _TaskEventCursorFollower
    from ouroboros.task_store import delete_document

    write_task_result(tmp_path, "parent0", "running")
    follower = _TaskEventCursorFollower(tmp_path, "parent0", {"v": 2, "seq": 0, "view": "", "positions": {}})
    follower.refresh_view()
    first = follower.view
    write_task_result(tmp_path, "aaaa0001", "requested", parent_task_id="parent0", root_task_id="parent0",
                      delegation_role="subagent", request_claim={"token": "t", "phase": "claimed"})
    assert follower.refresh_view() is None and follower.view == first  # no replay for a child not yet sent
    delete_document(tmp_path, "aaaa0001")  # a refused request's claim goes, and the view still never moved
    assert follower.refresh_view() is None and follower.view == first
    write_task_result(tmp_path, "aaaa0002", "requested", parent_task_id="parent0", root_task_id="parent0",
                      delegation_role="subagent", request_claim={"token": "t", "phase": "requested"})
    assert follower.refresh_view()["type"] == "cursor_replay" and "aaaa0002" in follower.task_filter_ids


def test_a_forked_helpers_rollback_keeps_its_drive_whole_and_another_writers_bytes_stay(tmp_path, monkeypatch):
    from ouroboros.headless import prepared_drive_mode, task_state_dir
    from ouroboros.tools import control_scheduling

    _mint(monkeypatch, "cafef00d")
    foreign = tmp_path / "data" / "task_results" / "artifacts" / "cafef00d" / "source_handles" / "image.bin"
    real = control_scheduling.prepare_task_drive

    def beside_another_writer(root, tid, mode, project_id=""):  # a writer that claims nothing draws the same id
        foreign.parent.mkdir(parents=True)
        foreign.write_bytes(b"another conversation's image")
        return real(root, tid, mode, project_id=project_id)

    monkeypatch.setattr(control_scheduling, "prepare_task_drive", beside_another_writer)
    _fail_request_write(monkeypatch)
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="forked")
    assert "failed to persist requested status for cafef00d" in result and ctx.pending_events == [], result
    assert prepared_drive_mode(task_state_dir(data, "cafef00d") / "data", "cafef00d") == "forked"  # untouched
    assert load_task_result(data, "cafef00d")["reason_code"] == "request_unsent"  # its closed record keeps the id
    assert foreign.read_bytes() == b"another conversation's image"  # never this request's to remove


def test_owner_rows_are_never_carried_to_a_helper_still_being_prepared(tmp_path, monkeypatch):
    from ouroboros import owner_relay

    members = [{"task_id": "aaaa0001", "status": "requested", "delegation_role": "subagent",
                "request_claim": {"token": "t", "phase": "claimed"}},
               {"task_id": "aaaa0002", "status": "requested", "delegation_role": "subagent",
                "request_claim": {"token": "u", "phase": "requested"}}]
    carried = []
    monkeypatch.setattr("ouroboros.task_status.find_child_tasks", lambda *_a, **_k: [dict(row) for row in members])
    monkeypatch.setattr(owner_relay, "_carry_to_member", lambda _root, _drive, tid, *_a: carried.append(tid))
    monkeypatch.setattr("ouroboros.delegate_interactions.place_at_receipt", lambda *_a, **_k: None)
    owner_relay.carry_at_receipt(tmp_path, "parent0", [])
    assert carried == ["aaaa0002"]


def test_recovery_refuses_no_row_a_stop_reached_after_its_first_look(tmp_path, monkeypatch):
    from supervisor import queue, workspace_preparation as prep

    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: True)
    seed_result(tmp_path, "aaaa0019", _unsent("aaaa0019", root_task_id="parent0", role="helper"))
    before = result_signature(tmp_path, "aaaa0019")
    announced, recovered = [], prep.PreparationClaim.recovered

    def stopped_meanwhile(root, row, *args):  # after the first look for a Stop, before the refusal's write
        _stop(root, row["task_id"])
        return recovered(root, row, *args)

    monkeypatch.setattr(prep.PreparationClaim, "recovered", staticmethod(stopped_meanwhile))
    monkeypatch.setattr("supervisor.events_schedule_task._send_subagent_rejection", lambda *a, **k: announced.append(k))
    report = prep.recover_interrupted_preparations(tmp_path)
    assert "aaaa0019" not in report["settled"] and announced == [], report
    assert result_signature(tmp_path, "aaaa0019") == before  # the row the Stop's custody settles stands


def _tree(root):
    """Every file under ROOT with its bytes: what must not change."""
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _finished(root, task_id, files):
    """A finished task's record and, with FILES, its folders, all imaginary: what a new start must never take or
    touch. Without them the record alone holds the id, so no folder check answers before the store's claim."""
    seed_result(root, task_id, {"task_id": task_id, **FOREIGN})
    for folder in ("state/headless_tasks", "task_results/artifacts") if files else ():
        (root / folder / task_id / "data").mkdir(parents=True)
        (root / folder / task_id / "data" / "answer.txt").write_text("an unrelated answer from weeks ago")
    return result_signature(root, task_id), _tree(root / "state" / "headless_tasks" / task_id), _tree(
        root / "task_results" / "artifacts" / task_id)


@pytest.mark.parametrize("files", [True, False], ids=["record_and_folders", "record_only"])
@pytest.mark.parametrize("entry", ["schedule_subagent", "depth_refusal", "http_start", "start_task"])
def test_a_new_start_forced_onto_a_finished_tasks_id_never_answers_with_it_or_touches_its_files(tmp_path, monkeypatch,
                                                                                               entry, files):
    """#4317 regression: the one mint is forced to draw a finished task's id first (an injected generator, an
    imaginary record, with or without its folders). The start takes a fresh id; the finished record, its answer
    and its files stay byte for byte, nothing under the old id is created, and nothing the start returns is the
    old answer."""
    from ouroboros.tools.control_delegation import record_depth_limit_refusal

    width = 16 if entry in {"http_start", "start_task"} else 8
    taken, fresh = "deadbeef" * (width // 8), "cafef00d" * (width // 8)
    data = tmp_path / ("data" if entry != "start_task" else "")
    data.mkdir(parents=True, exist_ok=True)
    signature, state, artifacts = _finished(data, taken, files)
    _mint(monkeypatch, taken, fresh)
    if entry == "schedule_subagent":
        ctx, _data, answer = _schedule(tmp_path, monkeypatch, memory_mode="empty")
        assert ctx.pending_events[0]["task_id"] == fresh, answer
    elif entry == "depth_refusal":
        ctx = SimpleNamespace(task_id="parent0", task_metadata={}, drive_root=str(data), budget_drive_root="")
        answer = record_depth_limit_refusal(ctx, {"objective": "too deep", "role": "helper"}, {}, {},
                                            current_depth=3, new_depth=4, max_depth=3)
        assert f"task_id={fresh}" in answer, answer
    elif entry == "http_start":
        client, _data, _captured = _api_client(tmp_path, monkeypatch)
        response = client.post("/api/tasks", json={"description": "a new root"})
        answer = response.text
        assert response.status_code == 200 and response.json()["task_id"] == fresh, answer
    else:
        from ouroboros.tools.control import _start_task

        _start_ready(monkeypatch)
        ctx = _start_ctx(tmp_path)
        answer = _start_task(ctx, "a separate work", title="Separate work")
        assert answer.startswith(f"OK: task {fresh}") and ctx.pending_events[0]["task_id"] == fresh, answer
    assert taken not in answer and FOREIGN["result"] not in answer
    assert result_signature(data, taken) == signature and load_task_result(data, taken)["result"] == FOREIGN["result"]
    assert _tree(data / "state" / "headless_tasks" / taken) == state
    assert _tree(data / "task_results" / "artifacts" / taken) == artifacts
    assert files or not task_id_claims.occupied(retained_task_paths(data, taken))  # nothing staged under it
    assert _held(data, taken) is None  # and no claim stands over another record's id
    assert load_task_result(data, fresh) and load_task_result(data, fresh).get("result") != FOREIGN["result"]


# ---- a request's own folders are the ones it made, never the ones its id names ----------------------------------

def test_a_requests_own_folders_are_made_by_exclusive_creation_and_a_standing_one_is_never_taken(tmp_path):
    first, second = tmp_path / "state" / "headless_tasks" / "aaaa0001", tmp_path / "task_results" / "aaaa0001"
    assert task_id_claims.make_own_folders((first, second)) == (first, second) and first.is_dir()
    with pytest.raises(FileExistsError):  # another writer's folder: nothing made is kept, nothing of it touched
        task_id_claims.make_own_folders((tmp_path / "fresh", first))
    assert not (tmp_path / "fresh").exists() and first.is_dir() and second.is_dir()


def test_a_helper_whose_id_meets_another_writers_state_folder_stages_and_removes_nothing(tmp_path, monkeypatch):
    from ouroboros.tools import control_scheduling

    real = control_scheduling._child_constraint
    foreign = tmp_path / "data" / "state" / "headless_tasks" / "cafef00d" / "data" / "image.bin"

    def beside_another_writer(ctx, params, surface, workspace, metadata, named):  # a writer that claims nothing
        foreign.parent.mkdir(parents=True)  # draws the same id after the claim, before this request stages
        foreign.write_bytes(b"another task's drive")
        return real(ctx, params, surface, workspace, metadata, named)

    monkeypatch.setattr(control_scheduling, "_child_constraint", beside_another_writer)
    _mint(monkeypatch, "cafef00d")
    ctx, data, result = _schedule(tmp_path, monkeypatch, memory_mode="empty")
    assert "SUBTASK_DRIVE_ERROR" in result and "another writer" in result and "cafef00d" not in result, result
    assert ctx.pending_events == [] and load_task_result(data, "cafef00d") is None  # its own row is released
    assert foreign.read_bytes() == b"another task's drive"


def test_an_api_admission_whose_id_meets_another_writers_folder_stages_and_removes_nothing(api, monkeypatch):
    from supervisor import queue, task_api_claim

    client, data, captured = api
    real = task_api_claim.claim_task_admission
    foreign = data / "state" / "headless_tasks" / "explicit-14" / "data" / "image.bin"

    def beside_another_writer(*args):  # a writer that claims nothing draws the same id after the claim
        claimed = real(*args)
        foreign.parent.mkdir(parents=True)
        foreign.write_bytes(b"another task's drive")
        return claimed

    monkeypatch.setattr(task_api_claim, "claim_task_admission", beside_another_writer)
    response = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-14", "memory_mode": "empty"})
    assert response.status_code == 409 and response.json()["reason_code"] == "task_id_occupied", response.text
    assert captured == [] and _held(data, "explicit-14") is None and "explicit-14" not in queue.ADMISSION_RESERVATIONS
    assert foreign.read_bytes() == b"another task's drive" and load_task_result(data, "explicit-14") is None


def test_an_api_rollback_keeps_its_staged_drive_and_a_direct_drive_it_never_made(api, monkeypatch):
    from ouroboros.gateway import tasks
    from ouroboros.headless import prepared_drive_mode, task_state_dir
    from supervisor import queue

    client, data, captured = api
    direct = data / "task_drives" / "explicit-15" / "notes.txt"

    def beside_a_direct_drive(task):
        direct.parent.mkdir(parents=True)
        direct.write_text("another writer's direct drive", encoding="utf-8")
        raise OSError("the contract could not be attached")

    monkeypatch.setattr(tasks, "attach_task_contract", beside_a_direct_drive)
    response = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-15", "memory_mode": "empty"})
    assert response.status_code == 503 and response.json()["staging_retained"] is True, response.text
    assert response.json()["error"].startswith("the contract could not be attached. Nothing was queued.")
    assert prepared_drive_mode(task_state_dir(data, "explicit-15") / "data", "explicit-15") == "empty"  # untouched
    assert direct.read_text(encoding="utf-8") == "another writer's direct drive"  # the one it never made stays
    row = load_task_result(data, "explicit-15")  # its cause, recorded as its never-admitted refusal
    assert row["status"] == "failed" and row["reason_code"] == "admission_preparation_failed"
    assert _held(data, "explicit-15") is None and "explicit-15" not in queue.ADMISSION_RESERVATIONS and captured == []


def test_a_recorded_refusal_leaves_what_it_staged_whole_under_its_row(api, monkeypatch):
    from ouroboros.headless import prepared_drive_mode, task_state_dir

    client, data, captured = api
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda _task: {"_admission_blocked": "root_budget_fence"})
    refused = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-16", "memory_mode": "empty"})
    assert refused.status_code == 409 and refused.json()["status"] == "failed", refused.text
    row = load_task_result(data, "explicit-16")  # the refusal row keys what stays, for its retention
    assert row["reason_code"] == "root_budget_fence" and row["admission_cleanup"] == {"child_drive_removed": False}
    assert prepared_drive_mode(task_state_dir(data, "explicit-16") / "data", "explicit-16") == "empty"  # untouched
    assert _held(data, "explicit-16") is None and captured == []  # the row holds the id now


# ---- a hand-off settles whichever writer comes first, and unknown stays unknown ---------------------------------

def test_the_receipt_lands_on_dispatch_marks_its_own_launch_wrote_first(api, monkeypatch):
    """Q is released before the receipt lands, so the tick may claim and launch the task first: its dispatch marks
    (``task_admission.commit_dispatch_marks``) then create the row. That row is this admission's own and carries its
    token, so the receipt lands on it; a 503 here would invite a client to send again work that already started."""
    from supervisor import queue, task_admission

    client, data, captured = api
    monkeypatch.setattr(queue, "DRIVE_ROOT", data)
    marks = []

    def enqueue(task):  # the stamps the queue gives an admitted row (``queue.enqueue_task``)
        row = {**task, "_admission_owner_token": task["_admission_token"], "_attempt": 1, "_project_scope_none": True}
        captured.append(row)
        return row

    def launched_first(reason=""):  # the barrier publishes; meanwhile the tick claimed the row and commits its marks
        if reason == "api_task_create":
            marks.append(task_admission.commit_dispatch_marks(
                captured[-1], [task_admission.project_dispatch_projection(captured[-1])]))
        return True

    monkeypatch.setattr("supervisor.queue.enqueue_task", enqueue)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", launched_first)
    response = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-20"})
    assert marks == [True] and response.status_code == 200, response.text
    row = load_task_result(data, "explicit-20")
    assert row["api_admission"] == {"token": captured[-1]["_admission_token"], "status": "accepted"}
    assert row["status"] == "scheduled" and row["admitted_dispatch"] == "possible" and row["description"] == "x"
    assert _held(data, "explicit-20") is None  # its own row holds the id now


def test_a_rollback_that_meets_another_records_row_lets_only_its_own_hold_go(api, monkeypatch):
    """Before its hand-off an admission's rollback finds a row it never wrote under its id: its own claim and
    reservation go (nothing of it is queued), the row stays exactly as it is, and the id is that row's."""
    from ouroboros.gateway import tasks
    from supervisor import queue

    client, data, captured = api
    written = []

    def taken_meanwhile(root, task_id, *_args, **_kwargs):  # a writer that claims nothing took the id meanwhile
        written.append(_foreign(root, task_id))
        raise OSError("no room for the drive")

    monkeypatch.setattr(tasks, "prepare_task_drive", taken_meanwhile)
    response = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-22"})
    assert response.status_code == 409 and response.json()["reason_code"] == "task_id_occupied", response.text
    assert "explicit-22" not in queue.ADMISSION_RESERVATIONS and _held(data, "explicit-22") is None
    assert result_signature(data, "explicit-22") == written[0] and captured == []


def test_a_refused_hand_off_records_its_refusal_through_a_contended_store_and_lets_its_hold_go(api, monkeypatch):
    """The queue refused the task and what it staged stays, so its never-admitted refusal must take the id. A store
    another writer holds only delays that write (nothing was rolled in), so it is written again within the store's
    usual wait; claim and reservation then go, and what was staged stays under the refusal row."""
    from ouroboros.store import StoreBusy
    from supervisor import queue, task_admission

    client, data, captured = api
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda _task: {"_admission_blocked": "root_budget_fence"})
    real, busy = task_admission.write_task_result, [2]

    def contended(root, task_id, status, **fields):  # the first two refusal writes meet another writer's lock
        if fields.get("admission_outcome") == "never_admitted" and busy[0]:
            busy[0] -= 1
            error = StoreBusy("the store is busy", reason="contention")
            error.rollback_confirmed = True
            raise error
        return real(root, task_id, status, **fields)

    monkeypatch.setattr(task_admission, "write_task_result", contended)
    response = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-21", "memory_mode": "empty"})
    assert busy == [0] and response.json().get("staging_retained") is True, response.text
    row = load_task_result(data, "explicit-21")
    assert row["status"] == "failed" and row["admission_outcome"] == "never_admitted"
    assert row["reason_code"] == "root_budget_fence" and row["admission_cleanup"] == {"staging_retained": True}
    assert _held(data, "explicit-21") is None and "explicit-21" not in queue.ADMISSION_RESERVATIONS and captured == []
    assert (data / "state" / "headless_tasks" / "explicit-21" / "data").is_dir()  # what it staged stays


def test_restart_records_a_hand_off_its_server_left_unsettled_beside_staging_so_its_stop_settles(tmp_path,
                                                                                                monkeypatch):
    """A hand-off whose server ended with an active Stop pending on the id cannot be proven never admitted (the
    saved queue skips rows under an active Stop): recovery treats it as admitted (scheduled, never never_admitted),
    keeps its staging untouched, retires its claim, and lets the Stop path finish it with its usual cancelled outcome."""
    from ouroboros.cancel_intents import active_intent
    from supervisor import cancel_publication as cp, queue

    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    monkeypatch.setattr("supervisor.cancel_publication._audit_delegated_runs_on_kill", lambda *a, **k: {})
    write_task_result(tmp_path, "seed0000", "requested")
    claim_provisional_id(tmp_path, "held0010", "t-http")
    assert advance_provisional_claim(tmp_path, "held0010", "t-http", "handoff") == ""
    staged = tmp_path / "state" / "headless_tasks" / "held0010" / "data" / "input.txt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"staged input")
    _stop(tmp_path, "held0010")
    assert _miss(tmp_path, monkeypatch, "held0010") == cp.CANCEL_FAILED  # it may be queuing the task: pending
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: True)
    assert task_id_claims.retire_settled_claims(tmp_path, queued=None) == []  # an unread queue proves nothing
    assert task_id_claims.retire_settled_claims(tmp_path, queued={"held0010"}) == []  # a queued one may run
    assert load_task_result(tmp_path, "held0010") is None and _held(tmp_path, "held0010")["phase"] == "handoff"
    assert task_id_claims.retire_settled_claims(tmp_path, queued=set()) == ["held0010"]
    row = load_task_result(tmp_path, "held0010")
    assert row["status"] == "scheduled" and row.get("admission_outcome") != "never_admitted", row
    assert _held(tmp_path, "held0010") is None
    assert staged.read_bytes() == b"staged input"  # nothing removed: the task row keys it for its retention
    assert _miss(tmp_path, monkeypatch, "held0010") == cp.CANCEL_CANCELLED
    row = load_task_result(tmp_path, "held0010")
    assert row["status"] == "cancelled" and row.get("admission_outcome") != "never_admitted", row
    assert active_intent(tmp_path, "held0010") is None  # the Stop settled through its usual cancel outcome


def test_a_cascade_or_graceful_stop_racing_an_admissions_receipt_stops_the_queued_task(tmp_path, monkeypatch):
    """A cascade or graceful Stop reads the live queue, then the claim. An admission may complete in between: queue
    its task, land its receipt and only then retire its claim. The Stop reads the live queue again and stops the
    task it finds there as it asked, never answering 404 for work that is about to run."""
    import asyncio

    from ouroboros.cancel_intents import SCOPE_CASCADE, active_intent
    from ouroboros.gateway import tasks
    from supervisor import queue

    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "task_has_live_ownership", lambda _task_id, **_kw: False)  # nothing of it runs yet
    monkeypatch.setattr(queue, "task_subtree_is_live",
                        lambda task_id, **_kw: any(row["id"] == task_id for row in queue.PENDING))
    monkeypatch.setattr("supervisor.owner_stop.begin_graceful_stop", lambda _task_id: None)
    write_task_result(tmp_path, "seed0000", "requested")
    real, admitted = task_id_claims.provisional_claim, set()

    def admitted_meanwhile(root, task_id):  # between the Stop's queue read and its claim read
        if task_id not in admitted:
            admitted.add(task_id)
            queue.PENDING.append({"id": task_id})
            write_task_result(root, task_id, "scheduled", api_admission={"token": "t-http", "status": "accepted"})
            assert task_id_claims.retire_provisional_claim(root, task_id, "t-http")
        return real(root, task_id)

    for task_id in ("held0030", "held0031"):
        claim_provisional_id(tmp_path, task_id, "t-http")
        assert advance_provisional_claim(tmp_path, task_id, "t-http", "handoff") == ""
    monkeypatch.setattr(task_id_claims, "provisional_claim", admitted_meanwhile)
    assert tasks._record_stop("held0030", cascade=True, stop_action_id="") is not None
    assert active_intent(tmp_path, "held0030")["scope"] == SCOPE_CASCADE  # the Stop it asked for
    answer = asyncio.run(tasks._graceful_stop_acknowledgement("held0031", cascade=True))
    assert answer.status_code == 202, bytes(answer.body)
    assert active_intent(tmp_path, "held0031")["scope"] == SCOPE_CASCADE


def test_no_step_under_the_stop_lock_waits_on_the_store_longer_than_one_slice(tmp_path, monkeypatch):
    """A Stop records its intent under one lock and gives up on it after 4 s. Every admission step that commits under
    that lock (a helper claim's release and closing, the task API's hand-off, boot's new-work reconciliation) waits
    on the store there for one short slice at most, never the store's usual 10 s; a busy store releases the lock and
    the step is tried again."""
    import contextlib

    from ouroboros import cancel_intents, promotion_intent
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
    from ouroboros.store import StoreBusy

    write_task_result(tmp_path, "seed0000", "requested")
    for task_id in ("held0020", "held0021"):
        claim_provisional_id(tmp_path, task_id, "t-http")
    for task_id, token in (("aaaa0050", "t1"), ("aaaa0051", "t2")):
        claim_id(task_id, _claim_row(tmp_path, token)[0], lambda _tid: True, width=8)
    write_task_result(tmp_path, "intent01", "requested", promotion_intent={"task_id": "intent01"},
                      promotion_admission={"status": "emitted", "phase": "prepared", "routing_token": "r"})
    held, waits = [0], []
    real_lock, real_connection, real_open = cancel_intents.cancellation_projection_lock, store.connection, store.open_store

    @contextlib.contextmanager
    def stop_lock(root, **kwargs):
        with real_lock(root, **kwargs):
            held[0] += 1
            try:
                yield
            finally:
                held[0] -= 1

    @contextlib.contextmanager
    def connection(*args, **kwargs):
        if held[0]:
            waits.append(store.bounded_wait(store.STORE_WAIT_SEC))
        with real_connection(*args, **kwargs) as conn:
            yield conn

    monkeypatch.setattr(cancel_intents, "cancellation_projection_lock", stop_lock)
    monkeypatch.setattr(store, "connection", connection)
    assert advance_provisional_claim(tmp_path, "held0020", "t-http", "handoff") == ""
    assert release_task_claim(tmp_path, "aaaa0050", partial(same_request, token="t1"))
    assert task_id_claims.close_unsent_claim(tmp_path, "aaaa0051", "t2", result="never sent")
    assert promotion_intent.reconcile_promotion_intents(tmp_path, protected=set())["not_started"] == ["intent01"]
    assert len(waits) >= 4 and set(waits) == {ACQUISITION_SLICE_SEC}, waits
    busy = []

    def busy_once(root, *, write, **kwargs):  # another writer holds the store for the first try
        if write and not busy:
            busy.append(held[0])
            error = StoreBusy("the store is busy", reason="contention")
            error.rollback_confirmed = True
            raise error
        return real_open(root, write=write, **kwargs)

    monkeypatch.setattr(store, "open_store", busy_once)
    assert advance_provisional_claim(tmp_path, "held0021", "t-http", "handoff") == "" and busy == [1]
    assert _held(tmp_path, "held0021")["phase"] == "handoff"


def test_between_the_put_and_the_receipt_the_queued_row_and_the_claim_keep_the_id(tmp_path, monkeypatch):
    """The queue releases an admission's reservation once it takes the row (``queue.enqueue_task``), before the row
    is durable. The id stays held meanwhile by the queued row in memory and by the claim, which goes only once a row
    holds the id: a rival admission or helper claim of it is refused throughout, also after a restart barrier
    withdrew the row from memory."""
    from supervisor import queue, workers
    from supervisor.task_api_claim import claim_task_admission

    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    monkeypatch.setattr(workers, "_worker_pool_execution_state",
                        lambda *_args: {"available": True, "disabled_reason": ""})
    write_task_result(tmp_path, "seed0000", "requested")
    claim_provisional_id(tmp_path, "explicit-30", "first")
    assert advance_provisional_claim(tmp_path, "explicit-30", "first", "handoff") == ""
    queue.PENDING.append({"id": "explicit-30", "_admission_owner_token": "first"})  # taken, its reservation released
    helper_write = _claim_row(tmp_path, "rival")[0]
    for _withdrawn in (False, True):
        refusal = claim_task_admission(tmp_path, "explicit-30", "second")[1]
        assert refusal[0] == 409 and refusal[2]["reason_code"] == "duplicate_task_id", refusal
        with pytest.raises(TaskIdTaken):
            claim_id("explicit-30", helper_write, lambda _tid: True, width=8)
        queue.PENDING.clear()  # the restart barrier's withdrawal: the claim alone holds it
    assert _held(tmp_path, "explicit-30")["token"] == "first" and load_task_result(tmp_path, "explicit-30") is None


def _admitting_receipt(root, task_id, owner):
    """A helper's SCHEDULED receipt still ``admitting`` (``workspace_preparation``): written before its put, which
    nothing proved, beside the prepared copy it records."""
    marker = {"token": "prep", "phase": "admitting", "generation": "gone", "owner": owner, "surface": "self_worktree",
              "claimed_at": "2026-11-26T00:00:00+00:00",
              "resource": {"surface": "self_worktree", "path": str(root / "copies" / task_id)}}
    seed_result(root, task_id, {**_unsent(task_id, root_task_id="parent0", role="helper"), "status": "scheduled",
                                "request_claim": {"token": "t", "phase": "requested"}, "workspace_preparation": marker})


def _saved_queue(root, monkeypatch, *task_ids, text=None):
    """The queue snapshot a restart would boot, naming TASK_IDS (or TEXT as it stands on disk)."""
    import json

    from supervisor import queue

    path = root / "state" / "queue_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text is not None else json.dumps(
        {"pending": [{"id": tid, "task": {"id": tid}} for tid in task_ids], "running": []}), encoding="utf-8")
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", path)
    return path


def _sweep(root, monkeypatch):
    """The periodic recovery sweep with the queue empty in memory; the copies it disposes."""
    from supervisor import queue, workspace_preparation as prep

    monkeypatch.setattr(queue, "DRIVE_ROOT", root)
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    disposed = []
    monkeypatch.setattr(prep, "_dispose_resource", lambda task_id, _resource: disposed.append(task_id) or "removed")
    return disposed


def test_the_next_sweep_keeps_a_receipt_whose_barrier_wrote_the_queue_before_its_confirmation_failed(tmp_path,
                                                                                                    monkeypatch):
    """A helper's restart barrier wrote the snapshot naming its row; its confirmation failed and so did the write that
    withdrew it, so the callback left receipt and copy (above). The periodic sweep after it finds no row in memory
    and a saved queue this process published before, yet the queue on disk names the row, which a restart would
    boot: unknown, never refused, and the copy stays. Once the published queue no longer names it, no put happened."""
    from supervisor import queue, workspace_preparation as prep

    disposed = _sweep(tmp_path, monkeypatch)
    _admitting_receipt(tmp_path, "aaaa0040", prep._process_owner())  # admitted here; its job has ended
    path = _saved_queue(tmp_path, monkeypatch, "aaaa0040")
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", (path, {}, "an earlier publication of this process"))
    report = prep.recover_interrupted_preparations(tmp_path)
    assert report["uncertain"] == ["aaaa0040"] and report["settled"] == [], report
    row = load_task_result(tmp_path, "aaaa0040")
    assert row["status"] == "scheduled" and row["workspace_preparation"]["phase"] == "admitting" and disposed == []
    _saved_queue(tmp_path, monkeypatch)  # a later publication of this process: the queue without it
    assert prep.recover_interrupted_preparations(tmp_path)["settled"] == ["aaaa0040"] and disposed == ["aaaa0040"]
    assert load_task_result(tmp_path, "aaaa0040")["admission_outcome"] == "never_admitted"


def test_an_unreadable_saved_queue_never_proves_an_earlier_servers_receipt_was_not_put(tmp_path, monkeypatch):
    """The queue an earlier server left could not be read at boot: it may hold, even have launched, that server's
    admitted helper. Its receipt stays unknown with its copy, also after this process published a queue of its own
    over it; only a restore that read the saved queue whole, naming no row of it, proves that no put happened."""
    from supervisor import queue, queue_snapshot, workspace_preparation as prep

    disposed = _sweep(tmp_path, monkeypatch)
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: True)
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    _admitting_receipt(tmp_path, "aaaa0041", {"pid": 11, "birth": "b"})
    path = _saved_queue(tmp_path, monkeypatch, text="{ the queue the earlier server left, torn")
    report = prep.recover_interrupted_preparations(tmp_path)
    assert report["uncertain"] == ["aaaa0041"] and report["settled"] == [] and disposed == [], report
    _saved_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", (path, {}, "this process's own queue, written over it"))
    report = prep.recover_interrupted_preparations(tmp_path)
    assert report["uncertain"] == ["aaaa0041"] and disposed == [], report
    monkeypatch.setattr(queue_snapshot, "_RESTORED", {str(path.absolute())})  # a restore that read it whole
    assert prep.recover_interrupted_preparations(tmp_path)["settled"] == ["aaaa0041"] and disposed == ["aaaa0041"]


def _cold_boot(root, monkeypatch):
    """Prepare a fresh supervisor generation on ROOT ready to run restore -> startup publication -> recovery."""
    from ouroboros import server_boot, server_maintenance, store_switch
    from supervisor import active_activity, queue, queue_snapshot, workers

    registry = active_activity.DirectActivityRegistry()
    monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: registry)
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    monkeypatch.setattr(workers, "PENDING", queue.PENDING)
    monkeypatch.setattr(workers, "RUNNING", queue.RUNNING)
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    monkeypatch.setattr(queue_snapshot, "_RESTORING", {})
    monkeypatch.setattr(store_switch, "require_published", lambda _root: None)
    monkeypatch.setattr(server_boot, "resume_peer_replies", lambda _root: None)
    monkeypatch.setattr(server_maintenance, "DATA_DIR", root)
    queue.init(root)
    return queue, queue_snapshot, server_maintenance


def test_startup_treats_stopped_saved_api_handoff_as_admitted_never_never_admitted(tmp_path, monkeypatch):
    """MUST-1: POST /api/tasks with an attachment saved pending in the queue snapshot, Stop recorded an active
    intent, and the server stopped before the POST receipt and before Stop ran. At startup restore rightly skips
    the stopped row, and claim recovery must treat the work as admitted (never never_admitted) so the existing
    Stop path settles it as cancelled with its staged attachment intact."""
    import json

    from ouroboros.cancel_intents import active_intent
    from ouroboros.utils import utc_now_iso
    from supervisor import cancel_publication as cp

    monkeypatch.setattr("supervisor.cancel_publication._audit_delegated_runs_on_kill", lambda *a, **k: {})
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: True)
    write_task_result(tmp_path, "seed0000", "requested")
    tid, tok = "stopped001", "tok-stop-1"
    claim_provisional_id(tmp_path, tid, tok)
    assert advance_provisional_claim(tmp_path, tid, tok, "handoff") == ""
    attachment = tmp_path / "task_results" / "artifacts" / tid / "upload.txt"
    attachment.parent.mkdir(parents=True)
    attachment.write_bytes(b"owner attachment bytes")
    saved_task = {
        "id": tid, "type": "task", "chat_id": 42, "description": "process attachment",
        "title": "Attachment task", "admitted_dispatch": "none", "_admission_owner_token": tok,
        "metadata": {"source": "api_task"},
    }
    snap_path = tmp_path / "state" / "queue_snapshot.json"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(json.dumps({
        "ts": utc_now_iso(), "pending": [{"id": tid, "task": saved_task}], "running": [],
        "acceptance_fences": [], "budget_root_fences": [],
    }), encoding="utf-8")
    _stop(tmp_path, tid)
    queue, _qs, sm = _cold_boot(tmp_path, monkeypatch)
    assert queue.restore_pending_from_snapshot() == 0  # active Stop keeps the row out of PENDING
    assert queue.persist_queue_snapshot(reason="startup") is True
    sm._run_startup_task_recovery(tmp_path, tmp_path, skip_live_data=False, prior_worker_pids=set())
    row = load_task_result(tmp_path, tid)
    assert row is not None and row.get("admission_outcome") != "never_admitted", row
    assert row["status"] == "scheduled" and row.get("chat_id") == 42 and _held(tmp_path, tid) is None
    assert attachment.read_bytes() == b"owner attachment bytes"
    assert _miss(tmp_path, monkeypatch, tid) == cp.CANCEL_CANCELLED
    cancelled = load_task_result(tmp_path, tid)
    assert cancelled["status"] == "cancelled" and cancelled.get("admission_outcome") != "never_admitted", cancelled
    assert cancelled.get("cancel_origin", {}).get("source") == "test"
    assert active_intent(tmp_path, tid) is None and attachment.read_bytes() == b"owner attachment bytes"


def test_startup_records_unknown_outcome_when_original_saved_queue_was_unreadable(tmp_path, monkeypatch):
    """MUST-2: POST saved pending in the snapshot and crashed before the receipt. At cold start the old snapshot is
    unreadable, so restore leaves _RESTORED unset, yet startup publication overwrites it with an empty queue and sets
    QUEUE_PUBLICATION. Claim recovery must still see the original snapshot was unreadable and record an honest unknown
    failed outcome (never never_admitted), keeping staged files and ID ownership."""
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: True)
    write_task_result(tmp_path, "seed0000", "requested")
    for tid, tok in (("unread001", "tok-u1"), ("unread002", "tok-u2")):
        claim_provisional_id(tmp_path, tid, tok)
        assert advance_provisional_claim(tmp_path, tid, tok, "handoff") == ""
    staged = tmp_path / "state" / "headless_tasks" / "unread001" / "data" / "input.txt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"saved staging")
    snap_path = tmp_path / "state" / "queue_snapshot.json"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text("{ torn snapshot written before crash", encoding="utf-8")
    queue, queue_snapshot, sm = _cold_boot(tmp_path, monkeypatch)
    assert queue.restore_pending_from_snapshot() == 0
    assert str(snap_path.absolute()) not in queue_snapshot._RESTORED
    assert queue.persist_queue_snapshot(reason="startup") is True
    assert queue.QUEUE_PUBLICATION is not None and not queue_snapshot.saved_queue_unowned(require_readable=True)
    sm._run_startup_task_recovery(tmp_path, tmp_path, skip_live_data=False, prior_worker_pids=set())
    for tid in ("unread001", "unread002"):
        row = load_task_result(tmp_path, tid)
        assert row is not None and row["status"] == "failed", (tid, row)
        assert row.get("admission_outcome") != "never_admitted", (tid, row)
        assert row["reason_code"] == "saved_queue_unreadable", (tid, row)
        assert "unreadable" in row["result"].lower() and "unknown" in row["result"].lower(), (tid, row)
        assert _held(tmp_path, tid) is None
        with pytest.raises(TaskIdTaken):  # the recorded row keeps the ID owned
            claim_provisional_id(tmp_path, tid, "later-tok")
    assert staged.read_bytes() == b"saved staging"
    assert load_task_result(tmp_path, "unread001").get("admission_cleanup") == {"staging_retained": True}


def test_refusal_retry_after_contended_post_commit_readback_is_idempotent(api, monkeypatch):
    """SHOULD-1: in PERSIST mode a concurrent writer can raise StoreBusy on the post-commit readback inside
    persist_never_admitted_refusal. Retrying must reuse the same refusal token and recognize its own committed row
    instead of minting a fresh token and failing with 503 unconfirmed and a held reservation."""
    from ouroboros.gateway import tasks
    from ouroboros.store import StoreBusy
    from supervisor import queue, task_admission

    client, data, captured = api

    def fail_after_staging(_task):
        raise OSError("the contract could not be attached")

    monkeypatch.setattr(tasks, "attach_task_contract", fail_after_staging)
    real_load, tripped = task_admission.load_task_result, []

    def busy_post_commit_readback(root, task_id, *args, **kwargs):
        row = real_load(root, task_id, *args, **kwargs)
        if task_id in {"explicit-25", "held-retry-1"} and isinstance(row, dict) and row.get("status") == "failed":
            if task_id not in tripped:
                tripped.append(task_id)
                error = StoreBusy("concurrent writer in PERSIST mode", reason="contention")
                error.rollback_confirmed = True
                raise error
        return row

    monkeypatch.setattr(task_admission, "load_task_result", busy_post_commit_readback)
    response = client.post("/api/tasks", json={"description": "x", "task_id": "explicit-25", "memory_mode": "empty"})
    assert "explicit-25" in tripped
    assert response.json().get("staging_retained") is True and response.json().get("status") != "unconfirmed", response.text
    row = load_task_result(data, "explicit-25")
    assert row["status"] == "failed" and row["admission_outcome"] == "never_admitted"
    assert row["reason_code"] == "admission_preparation_failed" and row["admission_cleanup"] == {"staging_retained": True}
    assert _held(data, "explicit-25") is None and "explicit-25" not in queue.ADMISSION_RESERVATIONS and captured == []
    assert (data / "state" / "headless_tasks" / "explicit-25" / "data").is_dir()

    # The same post-commit StoreBusy on startup's unsettled hand-off refusal write also retries idempotently.
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: True)
    claim_provisional_id(data, "held-retry-1", "tok-retry-1")
    assert advance_provisional_claim(data, "held-retry-1", "tok-retry-1", "handoff") == ""
    staged = data / "state" / "headless_tasks" / "held-retry-1" / "data" / "input.txt"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"staged")
    assert task_id_claims.retire_settled_claims(data, queued=set()) == ["held-retry-1"]
    assert "held-retry-1" in tripped and _held(data, "held-retry-1") is None
    assert load_task_result(data, "held-retry-1")["admission_outcome"] == "never_admitted"


def test_periodic_reconcile_retries_claim_recovery_after_startup_store_contention(tmp_path, monkeypatch):
    """SHOULD-2: if startup claim recovery exhausts its CURRENT retry budget, the handoff claim stays held and its
    pending Stop waits. Once the store frees, the existing periodic reconcile sweep retries recovery cheaply so the
    claim and its pending Stop settle without waiting for another restart."""
    import json

    from ouroboros.cancel_intents import active_intent
    from ouroboros.store import StoreBusy
    from ouroboros.utils import utc_now_iso
    from supervisor import cancel_publication as cp

    monkeypatch.setattr("supervisor.cancel_publication._audit_delegated_runs_on_kill", lambda *a, **k: {})
    monkeypatch.setattr("ouroboros.local_custody_repair.owner_ended", lambda pid, _birth, proof=None: True)
    write_task_result(tmp_path, "seed0000", "requested")
    for tid, tok in (("contend001", "tok-c1"), ("contend002", "tok-c2")):
        claim_provisional_id(tmp_path, tid, tok)
        assert advance_provisional_claim(tmp_path, tid, tok, "handoff") == ""
        staged = tmp_path / "state" / "headless_tasks" / tid / "data" / "input.txt"
        staged.parent.mkdir(parents=True)
        staged.write_bytes(b"staged")
    _stop(tmp_path, "contend001")
    snap_path = tmp_path / "state" / "queue_snapshot.json"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(json.dumps({
        "ts": utc_now_iso(), "pending": [], "running": [], "acceptance_fences": [], "budget_root_fences": [],
    }), encoding="utf-8")
    queue, _qs, sm = _cold_boot(tmp_path, monkeypatch)
    assert queue.restore_pending_from_snapshot() == 0
    assert queue.persist_queue_snapshot(reason="startup") is True

    def always_busy(_step, **_kw):
        err = StoreBusy("store busy past startup budget", reason="contention")
        err.rollback_confirmed = True
        raise err

    with monkeypatch.context() as busy:
        busy.setattr(task_id_claims, "while_contended", always_busy)
        sm._run_startup_task_recovery(tmp_path, tmp_path, skip_live_data=False, prior_worker_pids=set())
    assert _held(tmp_path, "contend001")["phase"] == "handoff" and _held(tmp_path, "contend002")["phase"] == "handoff"
    assert _miss(tmp_path, monkeypatch, "contend001") == cp.CANCEL_FAILED  # Stop still blocked on handoff

    # Store is now free: the existing periodic reconcile pass retries claim recovery.
    monkeypatch.setattr(sm, "_periodic_zombie_reconcile", lambda **_kw: None)
    monkeypatch.setattr(sm, "_run_reconcile_prunes", lambda *_a: None)
    monkeypatch.setattr(sm, "_run_drive_custody_pass", lambda *_a: None)
    monkeypatch.setattr(sm, "_tick_maintenance_child", lambda: None)
    monkeypatch.setattr("ouroboros.maintenance.child.request", lambda **_kw: None)
    monkeypatch.setattr("ouroboros.review_operation.retire_dead_review_pointers_softly", lambda *_a, **_kw: None)
    monkeypatch.setattr("ouroboros.late_review_marks.replay_owed_late_reviews", lambda *_a, **_kw: None)
    sm._run_periodic_reconcile_sweep([0.0], latch=SimpleNamespace(release=lambda: None))
    assert _held(tmp_path, "contend001") is None and _held(tmp_path, "contend002") is None
    assert load_task_result(tmp_path, "contend001")["status"] == "scheduled"
    assert load_task_result(tmp_path, "contend002")["admission_outcome"] == "never_admitted"
    assert _miss(tmp_path, monkeypatch, "contend001") == cp.CANCEL_CANCELLED
    assert load_task_result(tmp_path, "contend001")["status"] == "cancelled"
    assert active_intent(tmp_path, "contend001") is None
