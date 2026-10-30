"""The owner's keep, closed: every destructive consumer honours it, the probe never runs under the custody
lock, the inventory says what it could not read, and the model's writer stands only on held owner words."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest

from ouroboros import copy_environments as environments
from ouroboros import owner_keep
from ouroboros import review_checking_prune as rp
from ouroboros import subagent_worktrees as sw
from ouroboros.task_custody import task_custody_lock
from ouroboros.task_results import load_task_result
from tests._store_shared import seed_result
from tests.test_body_candidate_retention import candidate  # noqa: F401 -- the body fixture
from tests.test_review_checking_prune import _custody
from tests.test_review_checkout_lifecycle import TASK, _frozen, _seat
from tests.test_storage_free_safety import _copy, copies  # noqa: F401 -- the plain copies fixture
from tests.test_store import document

KEEP = {"basis": "test"}


# ---- lock order: the probe before the custody lock, proven by a real interleaving ----------------


def _settlement_meanwhile(root, task_id, run):
    """Run ``run(live)`` while a stand-in for drive settlement holds the queue lock the probe needs and
    then asks for the task's custody lock, as Phase B does. Whether settlement got custody is returned."""
    queue_lock, held, probed, outcome = threading.Lock(), threading.Event(), threading.Event(), {}

    def live(_task):
        probed.set()
        with queue_lock:
            return False

    def settlement():
        with queue_lock:
            held.set()
            probed.wait(5)
            with task_custody_lock(root, task_id, timeout_sec=0.5) as locked:
                outcome["custody"] = locked

    thread = threading.Thread(target=settlement)
    thread.start()
    held.wait(5)
    try:
        outcome["result"] = run(live)
    finally:
        thread.join(10)
    return outcome


def test_an_environment_retirer_asks_the_probe_before_the_custody_lock(candidate):  # noqa: F811
    ctx, data, env_root = candidate
    sw._force_rmtree(Path(ctx.repo_dir))  # the copy is gone; its declared environment remains
    outcome = _settlement_meanwhile(data, ctx.task_id, lambda live: environments.retire_environment(
        env_root, data, live, sw._resolve_root()))
    assert outcome["custody"] is True, "settlement waited out a retirer that held custody while probing"
    assert outcome["result"] == "" and not env_root.exists()


def test_a_seat_home_retirer_asks_the_probe_before_the_custody_lock(tmp_path):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-done")
    seed_result(root, TASK, document(TASK, "completed"))
    outcome = _settlement_meanwhile(root, TASK, lambda live: rp.retire_seat_home(root, area.parent, live, {}))
    assert outcome["custody"] is True, "settlement waited out a retirer that held custody while probing"
    assert outcome["result"] == "in_use" and area.is_dir()  # its checking copy is still there


def test_a_task_continued_after_its_proof_keeps_its_seat_home(tmp_path):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-done")
    seed_result(root, TASK, document(TASK, "running"))
    proven = {TASK: True}  # a proof read before the task was continued
    assert rp.retire_seat_home(root, area.parent, lambda _t: False, proven) == "in_use"
    assert area.parent.is_dir()


def test_a_changed_declaration_after_the_verdict_keeps_the_environment(candidate, monkeypatch):  # noqa: F811
    import os

    from ouroboros.test_environment import SCRATCH_MARKER

    ctx, data, env_root = candidate
    sw._force_rmtree(Path(ctx.repo_dir))

    def verdict(*_args, **_kwargs):  # a new process touched the declaration after the verdict was read
        os.utime(env_root / SCRATCH_MARKER, (1, 1))
        return ""

    monkeypatch.setattr(environments, "environment_verdict", verdict)
    assert environments.retire_environment(env_root, data, lambda _t: False, sw._resolve_root()) == "changed"
    assert env_root.is_dir()


# ---- every destructive consumer -----------------------------------------------------------------


def test_a_kept_task_keeps_its_captured_review_copy_until_released(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    done = _seat(root, source, "op-done")
    seed_result(root, TASK, document(TASK, "completed"))
    _custody(monkeypatch, settled=[done])
    assert owner_keep.set_owner_keep(root, TASK, keep=True, source=KEEP)["keep"] is True
    report = rp.prune_review_checkouts(root, set())
    assert report["released"] == 0 and report["kept"] >= 1 and done.is_dir()
    assert owner_keep.set_owner_keep(root, TASK, keep=False, source=KEEP)["status"] == "released"
    assert rp.prune_review_checkouts(root, set())["released"] == 1 and not done.exists()


def test_a_capture_still_runs_for_a_kept_task_and_only_the_release_waits(tmp_path, monkeypatch):
    from ouroboros import review_checking as rc

    root, source = _frozen(tmp_path)
    done = _seat(root, source, "op-done")
    seed_result(root, TASK, document(TASK, "completed"))
    owner_keep.set_owner_keep(root, TASK, keep=True, source=KEEP)
    receipt = rc.capture_seat_evidence(root, TASK, "op-done")
    assert receipt["status"] == "retained" and done.is_dir(), "the receipt is written, the copy stays"


def test_a_rejected_genesis_project_of_a_kept_task_stays(copies, tmp_path):  # noqa: F811
    from ouroboros.genesis_project import remove_genesis_project

    projects = tmp_path / "projects"
    project = projects / "kept"
    project.mkdir(parents=True)
    seed_result(copies[0], "kept", document("kept", "failed"))
    owner_keep.set_owner_keep(copies[0], "kept", keep=True, source=KEEP)
    assert remove_genesis_project(str(project), projects_root=projects, task_id="kept", data_dir=copies[0]) is False
    assert project.is_dir()
    owner_keep.set_owner_keep(copies[0], "kept", keep=False, source=KEEP)
    assert remove_genesis_project(str(project), projects_root=projects, task_id="kept", data_dir=copies[0]) is True
    assert not project.exists()


def _resumable(ctx, data):
    """What a resume reads finds the candidate, and its first write is accepted; its pins and capture events."""
    from ouroboros import body_candidate as body

    row = body.find(ctx.task_id, data)
    assert row is not None and not row.get("retiring") and Path(ctx.repo_dir).is_dir()
    assert body._select(ctx.task_id, data) == row and [fact["owner_task"] for fact in body.context_fact(data_dir=data)] == [ctx.task_id]
    body._update_row(row, lambda _entry: None, data)  # CANDIDATE_MISSING if it were marked retiring
    pins = subprocess.run(["git", "for-each-ref", "--format=%(objectname)", body.PIN_PREFIX], cwd=row["repo_dir"],
                          capture_output=True, text=True, check=True).stdout.split()
    events = data / "logs" / "events.jsonl"
    captured = sum('"body_candidate_preserved"' in line for line in events.read_text(encoding="utf-8").splitlines()) \
        if events.is_file() else 0
    return pins, captured


def test_a_kept_candidate_stays_resumable_through_every_pass_and_goes_once_released(candidate):  # noqa: F811
    ctx, data, _scratch = candidate
    (Path(ctx.repo_dir) / "work.txt").write_text("unique\n", encoding="utf-8")  # an unkept pass would pin this first
    owner_keep.set_owner_keep(data, ctx.task_id, keep=True, source=KEEP)
    for _ in range(2):
        reasons: dict = {}
        assert sw.prune_orphans(retention_days=0, reasons=reasons)["removed"] == 0
        assert list(reasons.values()) == [owner_keep.OWNER_KEEP]
        assert _resumable(ctx, data) == ([], 0), "no pin and no capture event for what stays"
    owner_keep.set_owner_keep(data, ctx.task_id, keep=False, source=KEEP)
    assert sw.prune_orphans(retention_days=0)["removed"] == 1 and not Path(ctx.repo_dir).exists()


@pytest.mark.parametrize("late", ["owner_keep", "busy"])
def test_a_keep_or_a_busy_gate_after_the_proof_leaves_the_candidate_unmarked(candidate, monkeypatch, late):  # noqa: F811
    """The proof is read before the gate; a keep landing after it, or a gate that cannot answer, leaves the
    record as it was: never retiring, so the next pass and a resume both find it."""
    import contextlib

    from ouroboros import body_candidate as body

    ctx, data, _scratch = candidate
    (Path(ctx.repo_dir) / "work.txt").write_text("unique\n", encoding="utf-8")
    real = body.verdict_holds

    def holds_then_late(*args, **kwargs):
        answer = real(*args, **kwargs)
        assert answer, "the proof held: only the gate stands between it and the removal"
        if late == "owner_keep":
            owner_keep.set_owner_keep(data, ctx.task_id, keep=True, source=KEEP)
        return answer

    @contextlib.contextmanager
    def busy(*_args, **_kwargs):
        yield "busy"

    monkeypatch.setattr(body, "verdict_holds", holds_then_late)
    if late == "busy":
        monkeypatch.setattr(owner_keep, "gate", busy)
    reasons: dict = {}
    assert sw.prune_orphans(retention_days=0, reasons=reasons)["removed"] == 0
    assert list(reasons.values()) == [late]
    pins, _captured = _resumable(ctx, data)
    assert len(pins) == 1, "this pass's capture stays on its pin; the copy stays too"


# ---- the inventory and its truthful status -------------------------------------------------------


def test_the_inventory_names_environments_pins_seats_and_checkouts_and_says_what_it_could_not_read(copies):  # noqa: F811
    root, repo, wtroot = copies
    handle = _copy(copies)
    env = Path(handle.path + ".env")
    env.mkdir()
    gone = sw.provision_execution_snapshot(target_root=repo, task_id="finished", snapshot_id="snap",
                                           worktree_root=wtroot, data_dir=root)
    sw.release_snapshot_checkout("snap", worktree_root=wtroot, data_dir=root)  # pins and record stay
    seats = root / "state" / "review_checkouts" / "seats"
    for name, marker in (("mine", {"task_id": "finished"}), ("other", {"task_id": "else"}), ("torn", None)):
        (seats / name).mkdir(parents=True)
        (seats / name / "checking.json").write_text("{torn" if marker is None else json.dumps(marker), encoding="utf-8")
    checkout = root / "state" / "review_checkouts" / "tok"
    checkout.mkdir()
    (checkout / "checkout.json").write_text(json.dumps({"schema_version": 1, "task_id": "finished"}), encoding="utf-8")
    answer = owner_keep.set_owner_keep(root, "finished", keep=True, source=KEEP)
    assert answer["status"] == "partly_retired" and answer["keep"] is True
    assert {handle.path, str(env), "pins:snap", str(seats / "mine"), str(checkout)} <= set(answer["retained"])
    assert gone.path in answer["retired"] and str(seats / "other") not in answer["retained"]
    assert answer["unknown"] == ["review_seats"]


def test_with_a_class_unread_a_keep_is_saved_even_when_all_it_found_is_gone(copies):  # noqa: F811
    root = copies[0]
    seed_result(root, "gone", document("gone", "completed", child_drive_root=str(root / "task_drives" / "gone")))
    seats = root / "state" / "review_checkouts" / "seats" / "torn"
    seats.mkdir(parents=True)
    (seats / "checking.json").write_text("{torn", encoding="utf-8")
    answer = owner_keep.set_owner_keep(root, "gone", keep=True, source=KEEP)
    assert (answer["status"], answer["keep"], answer["unknown"]) == ("partly_retired", True, ["review_seats"])
    assert load_task_result(root, "gone")["owner_keep"]["scope"] == "task"


def test_no_source_writes_nothing_for_a_keep_or_a_release(copies):  # noqa: F811
    _copy(copies)
    assert owner_keep.set_owner_keep(copies[0], "finished", keep=True, source={})["reason"] == "source_missing"
    assert "owner_keep" not in load_task_result(copies[0], "finished")
    owner_keep.set_owner_keep(copies[0], "finished", keep=True, source=KEEP)
    refused = owner_keep.set_owner_keep(copies[0], "finished", keep=False, source={})
    assert (refused["status"], refused["keep"]) == ("refused", True)


# ---- the model's writer: root only, on held owner words ------------------------------------------


def _owner_context(root, *, origin=True, metadata=None):
    from types import SimpleNamespace

    from ouroboros import dialogue_store
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.project_dialogue import build_owner_message_ref

    row = {"chat_id": 1, "direction": "in", "ts": "2026-10-30T11:00:00Z", "text": "keep that task, I need it",
           "client_message_id": "owner-keep-1", "attachments": []}
    dialogue_store.append(root, "chat", row)
    meta = dict(metadata or {})
    if origin:
        meta["origin_message_ref"] = build_owner_message_ref(chat_id=1, client_message_id="owner-keep-1",
                                                             ts=row["ts"], text=row["text"])
    ctx = SimpleNamespace(task_id="main-turn", drive_root=root, budget_drive_root=root, current_chat_id=1,
                          task_metadata=meta, pending_events=[], event_queue=None, drive_logs=lambda: root / "logs")
    return ctx, format_address(row_address(row))


def test_the_tool_keeps_on_the_owner_message_this_turn_answers(copies):  # noqa: F811
    from ouroboros.tools.task_retention import _set_task_retention

    _copy(copies)
    ctx, address = _owner_context(copies[0])
    answer = json.loads(_set_task_retention(ctx, "finished", True, "the owner wants to look at it later"))
    assert answer["status"] == "kept" and answer["keep"] is True
    fact = load_task_result(copies[0], "finished")["owner_keep"]
    assert fact["source"]["basis"] == "owner_message" and fact["source"]["address"] == address
    assert fact["source"]["message_ref"]["client_message_id"] == "owner-keep-1"
    assert fact["note"] == "the owner wants to look at it later" and fact["scope"] == "task"
    released = json.loads(_set_task_retention(ctx, "finished", False, "the owner is done with it"))
    assert released["status"] == "released" and "owner_keep" not in load_task_result(copies[0], "finished")


def test_without_owner_words_the_tool_saves_nothing(copies):  # noqa: F811
    from ouroboros.tools.task_retention import _set_task_retention

    _copy(copies)
    ctx, address = _owner_context(copies[0], origin=False)
    assert "OWNER_SOURCE_REQUIRED" in _set_task_retention(ctx, "finished", True, "I think it is useful")
    assert "OWNER_SOURCE_NOT_HELD" in _set_task_retention(ctx, "finished", True, "why", owner_source=address + "0")
    assert "owner_keep" not in load_task_result(copies[0], "finished")


def test_a_held_owner_message_other_than_the_origin_can_be_named(copies):  # noqa: F811
    from ouroboros.owner_relay import retain_owner_message
    from ouroboros.tools.task_retention import _set_task_retention

    _copy(copies)
    ctx, _address = _owner_context(copies[0], origin=False)
    from ouroboros import dialogue_store
    from ouroboros.chat_chain import format_address, row_address

    row = {"chat_id": 1, "direction": "in", "ts": "2026-10-30T11:05:00Z", "text": "and keep it",
           "client_message_id": "owner-keep-2", "attachments": []}
    dialogue_store.append(copies[0], "chat", row)
    retain_owner_message(ctx, row)
    handle = format_address(row_address(row))
    answer = json.loads(_set_task_retention(ctx, "finished", True, "named", owner_source=handle))
    assert answer["status"] == "kept"
    assert load_task_result(copies[0], "finished")["owner_keep"]["source"]["address"] == handle


def test_a_delegated_or_child_task_cannot_change_what_is_kept(copies):  # noqa: F811
    from ouroboros.tools.task_retention import _set_task_retention

    _copy(copies)
    ctx, _address = _owner_context(copies[0], metadata={"delegation_role": "subagent", "parent_task_id": "main"})
    assert "ROOT_ONLY" in _set_task_retention(ctx, "finished", True, "child says so")
    assert "owner_keep" not in load_task_result(copies[0], "finished")


def test_the_tool_is_registered_root_only_and_withheld_from_observe(tmp_path):
    from ouroboros.safety import TOOL_POLICY
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES, LOCAL_READONLY_SUBAGENT_TOOL_NAMES, OBSERVE_WORLD_MUTATION_TOOLS,
    )
    from ouroboros.tools.registry import ToolRegistry

    names = {schema["function"]["name"] for schema in ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path).schemas()}
    assert "set_task_retention" in names and "task_retention" in ToolRegistry._FROZEN_TOOL_MODULES
    assert "set_task_retention" not in ACTING_SUBAGENT_TOOL_NAMES | LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    assert "set_task_retention" in OBSERVE_WORLD_MUTATION_TOOLS and "set_task_retention" in TOOL_POLICY


@pytest.mark.parametrize("keep", [True, False])
def test_the_route_keeps_its_own_request_source(copies, monkeypatch, keep):  # noqa: F811
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.task_owner_keep import api_task_owner_keep

    _copy(copies)
    monkeypatch.setattr("ouroboros.gateway.task_owner_keep.request_drive_root", lambda _request: copies[0])
    monkeypatch.setattr("ouroboros.gateway.task_owner_keep.migrating_error", lambda _root: None)
    client = TestClient(Starlette(routes=[Route("/api/tasks/{task_id}/owner-keep", endpoint=api_task_owner_keep,
                                                methods=["POST"])]))
    answer = client.post("/api/tasks/finished/owner-keep", json={"keep": keep}).json()
    assert answer["status"] == ("kept" if keep else "released") and "unknown" in answer
    if keep:
        source = load_task_result(copies[0], "finished")["owner_keep"]["source"]
        assert source == {"basis": "authenticated_gateway_request", "route": "POST /api/tasks/{task_id}/owner-keep"}


# ---- a pin is reread under custody: never reported retained once its record let it go ------------


def _released_snapshot(env):
    root, repo, wtroot = env
    handle = sw.provision_execution_snapshot(target_root=repo, task_id="finished", snapshot_id="snap", worktree_root=wtroot, data_dir=root)
    seed_result(root, "finished", document("finished", "completed"))
    assert sw.release_snapshot_checkout("snap", worktree_root=wtroot, data_dir=root)
    return handle


def _keep_with(monkeypatch, root, meanwhile):
    """Keep ``finished`` with ``meanwhile`` running after the unlocked inventory, before the custody lock."""
    listing = owner_keep.inventory

    def listed_then_changed(*args):
        listed = listing(*args)
        assert "pins:snap" in listed[0]
        meanwhile()
        return listed

    monkeypatch.setattr(owner_keep, "inventory", listed_then_changed)
    return owner_keep.set_owner_keep(root, "finished", keep=True, source=KEEP)


def test_a_pin_still_recorded_under_custody_is_retained(copies, monkeypatch):  # noqa: F811
    _released_snapshot(copies)
    answer = _keep_with(monkeypatch, copies[0], lambda: None)
    assert answer["retained"] == ["pins:snap"] and answer["status"] == "partly_retired"
    assert answer["retired"] == [str(sw._resolve_root() / "copies.noindex" / "dlg_finished_snap")], "its checkout was released"
    assert load_task_result(copies[0], "finished")[owner_keep.OWNER_KEEP]


def test_a_pin_retired_after_the_inventory_is_reported_retired_and_no_fact_is_saved(copies, monkeypatch):  # noqa: F811
    root, _repo, wtroot = copies
    handle = _released_snapshot(copies)

    def retire():
        assert sw.remove_execution_snapshot("snap", worktree_root=wtroot, data_dir=root)
        assert sw._git(copies[1], "show-ref", "--verify", "--quiet", handle.baseline_ref, check=False).returncode != 0

    answer = _keep_with(monkeypatch, root, retire)
    assert (answer["status"], answer["retained"]) == ("already_retired", []) and "pins:snap" in answer["retired"]
    assert answer["keep"] is False and owner_keep.OWNER_KEEP not in load_task_result(root, "finished")


def test_pins_released_after_the_inventory_are_retired_though_the_record_stays(copies, monkeypatch):  # noqa: F811
    root = copies[0]
    handle = _released_snapshot(copies)

    def release():
        row = sw._record_at(handle.path, root)
        sw._put_record({key: value for key, value in row.items() if key not in ("baseline_ref", "result_ref")}, root)

    answer = _keep_with(monkeypatch, root, release)
    assert (answer["status"], answer["retained"]) == ("already_retired", []) and "pins:snap" in answer["retired"]


def test_a_record_unreadable_under_custody_claims_nothing_about_its_pin_and_saves_the_fact(copies, monkeypatch):  # noqa: F811
    root = copies[0]
    handle = _released_snapshot(copies)
    answer = _keep_with(monkeypatch, root, lambda: sw._record_file(handle.path, root).write_text("{", encoding="utf-8"))
    assert "pins:snap" not in answer["retained"] + answer["retired"] and "private_copies" in answer["unknown"]
    assert answer["keep"] is True and load_task_result(root, "finished")[owner_keep.OWNER_KEEP]
