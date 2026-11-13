"""Remote reviewer seats (``remote_engines``) at the consumers that actually run them and keep their work.

The production review pool keeps a marked row's remote engine all the way to the seat's gateway: on the slot the pool
builds, in the aligned vectors the commit, skill and result surfaces dispatch from, on the slot their fan-out builds,
at the start that chooses the gateway, and on ``/review``'s deep self-review row; a seat on the owned engine stays
there. A writable remote seat checks a staged copy on the engine's host: its own capture brings that copy's result
and ``.review-evidence/`` home, receipt first, before the local copy goes, and the host copy stays until it has.
Against the stand in ``tests/_remote_engine_stand.py``: no real SSH, host, engine, model or account is used.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import pathlib
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import remote_engines, remote_workspace
from ouroboros import review_checking as rc
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests._remote_engine_stand import STORE
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool

pytest_plugins = ("tests._remote_engine_stand",)


def _remote_row(subagent_id: str = "remote-reviewer", access: str = "readonly") -> dict:
    row = pool_seat(subagent_id, "codex=gpt-6", kind="agent_session", profile_id="remote-main", access=access)
    row["route"]["engine"] = "vps"
    return row


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def _author(tmp_path) -> pathlib.Path:
    repo = tmp_path / "author"
    repo.mkdir()
    for args in (("init", "-q"), ("config", "user.email", "a@b"), ("config", "user.name", "a")):
        _git(repo, *args)
    (repo / ".gitignore").write_text(".review-drive/\nbuild/\n")
    (repo / "f.txt").write_text("v0\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _owned_gateway():
    raise ClaudexorUnavailable("daemon_unreachable", "no owned engine answers in this test")


# --- the pool keeps where a seat runs -----------------------------------------------------------------------

def test_the_pools_remote_seat_keeps_its_engine_through_the_vectors_to_its_gateway(tmp_path, monkeypatch):
    """The pool every review surface reads: a marked remote row's seat carries its engine on the pool's slot, in the
    commit/skill/result aligned vectors (an oversize packet's api drop too), on the slot their fan-out builds and
    into the start that picks the seat's gateway; the local row beside it stays on the owned engine."""
    from ouroboros import execution_domain
    from ouroboros.review_execution import (ReviewRouteUnavailable, SessionInvocation, run_delegated_review_session,
                                            session_route_for_review_slot)
    from ouroboros.reviewer_slot_config import commit_triad_delivery, review_pool_slots
    from ouroboros.tools import review_admission as admission
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_multi_model import _multi_model_review_async

    set_review_pool(monkeypatch, pool_roster(_remote_row(), pool_seat("local-reviewer", "codex=gpt-6",
                                                                      kind="agent_session")))
    slots = review_pool_slots()
    assert [(slot.slot_id, getattr(slot, "engine", "")) for slot in slots] == [("remote-reviewer", "vps"),
                                                                                ("local-reviewer", "")]
    assert [session_route_for_review_slot(slot).engine for slot in slots] == ["vps", ""]
    plan = admission.seat_vectors(commit_triad_delivery())
    assert plan.get("engines") == ["vps", ""] and admission.drop_api_rows(plan).get("engines") == ["vps", ""]

    built: dict = {}

    def _capture(request, *, slots, drive_root, llm, usage_ctx=None):
        built[slots[0].slot_id] = slots[0]
        return SimpleNamespace(actors=[{"slot_id": slots[0].slot_id, "model": slots[0].model, "status": "ok",
                                        "raw_text": "[]", "usage": {}, "prompt_ref": {}, "response_ref": {}}])

    monkeypatch.setattr("ouroboros.review_substrate.run_review_request", _capture)
    (tmp_path / "gov").mkdir()
    (tmp_path / "data").mkdir()
    ctx = ToolContext(repo_dir=tmp_path / "gov", drive_root=tmp_path / "data")
    ctx.task_id = "t-pool"
    asyncio.run(_multi_model_review_async("staged diff", "", list(plan["models"]), ctx, routes=list(plan["routes"]),
                                          row_plan=plan, session_task="BRIEF", session_root=""))
    assert {sid: session_route_for_review_slot(slot).engine for sid, slot in built.items()} == {
        "remote-reviewer": "vps", "local-reviewer": ""}

    asked = []

    def _gateway(record, root, engine="", custody_root=None):
        asked.append(engine)
        raise ReviewRouteUnavailable("stopped at the seat's gateway choice", code="test_stop")

    monkeypatch.setattr(execution_domain, "seat_gateway", _gateway)
    for slot_id in ("remote-reviewer", "local-reviewer"):
        invocation = SessionInvocation(task_id="t-pool", surface="multi_model_review", slot_id=slot_id, timeout_sec=60,
                                       session_route=session_route_for_review_slot(built[slot_id]))
        with pytest.raises(ReviewRouteUnavailable, match="gateway choice"):
            run_delegated_review_session(prompt="review", root=str(tmp_path), custody_drive=tmp_path / "custody",
                                         invocation=invocation)
    assert asked == ["vps", ""]  # the remote seat is never started on the owned engine in its place


def test_a_remote_deep_review_row_is_asked_and_run_on_its_engine(tmp_path, monkeypatch):
    """``/review`` on a remote catalog row: its route health is asked of that engine and the slot it runs carries it;
    the same row on the owned engine stays there."""
    from ouroboros import deep_self_review, execution_domain
    from ouroboros.configured_subagents import parse_configured_subagents
    from ouroboros.review_execution import session_route_for_review_slot
    from ouroboros.reviewer_slot_config import _catalog_row_slot

    remote = _catalog_row_slot(parse_configured_subagents({"enabled": True, "items": [_remote_row()]}).items[0], {})
    local = dataclasses.replace(remote, engine="")
    asked = []

    def _gateway(engine="", **_kwargs):
        asked.append(engine)
        raise RuntimeError("no engine is reached in this test")

    monkeypatch.setattr(execution_domain, "native_gateway", _gateway)
    assert deep_self_review._session_route_reason(remote).startswith("agent_service_unavailable")
    assert deep_self_review._session_route_reason(local).startswith("agent_service_unavailable")
    assert asked == ["vps", ""]

    class _Stop(Exception):
        pass

    built = {}

    def _executor(assignment, llm=None):
        built["slot"] = assignment.slot
        raise _Stop()

    monkeypatch.setattr("ouroboros.review_execution._review_route_executor", _executor)
    repo = pathlib.Path(__file__).resolve().parents[1]
    (tmp_path / "drive").mkdir()
    with pytest.raises(_Stop):
        deep_self_review._run_retrieving_review(repo, tmp_path / "drive", None, lambda _text: None, remote,
                                                task_id="t-deep", deadline_at="")
    assert session_route_for_review_slot(built["slot"]).engine == "vps"


# --- a remote seat's work comes home through its own capture -------------------------------------------------

_VIEW = "a" * 64  # the content address of the brief's paged subject view (``review_brief_coupling``)
_DIFF = b"diff --git a/f.txt b/f.txt\n--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-v0\n+v1 subject\n"


def _subject(tmp_path):
    """The author's staged change frozen as the commit gate freezes it (``SUBJECT_INDEX``), beside the brief's paged
    view of that change the host materialized in the review root: ``(author, data, source)``."""
    author = _author(tmp_path)
    (author / "f.txt").write_text("v1 subject\n")
    _git(author, "add", "f.txt")
    view = author / ".review-drive" / "t-seat" / f"{_VIEW}.txt"
    view.parent.mkdir(parents=True)
    view.write_bytes(_DIFF)
    data = tmp_path / "data"
    return author, data, rc.freeze_checking_source(author, data_root=data, subject=rc.SUBJECT_INDEX)


def _send(data, request, *, operation, invocation, custody_facts, **start):
    """Record a seat's start, then send it through the remote gateway (staging included): ``(handle, engine_ref)``."""
    client = remote_engines.gateway("vps")
    try:
        client.custody_root = data
        assert custody.record_start_requested(
            data, run_id="", task_id="t-seat", invocation_id=invocation, operation_id=operation,
            idempotency_key=invocation, request=request, route="codex", surface="multi_model_review",
            slot_id="remote-reviewer", engine_ref=dict(client.engine_ref), **custody_facts, **start)
        return client.start_run(request, idempotency_key=invocation), dict(client.engine_ref)
    finally:
        client.close()


def _remote_seat(host, tmp_path):
    """A writable seat prepared for the remote engine (``session_checking``), its real wire request
    (``prepare_review_session_request``) recorded and staged on the host."""
    from ouroboros.review_execution import SessionInvocation
    from ouroboros.review_session_preparation import prepare_review_session_request
    from ouroboros.subagents import delegated_run_shape, parse_subagent_harness

    author, data, source = _subject(tmp_path)
    shape = delegated_run_shape(False, "workspace_write")
    route = dataclasses.replace(parse_subagent_harness("codex=gpt-6"), engine="vps")
    client = remote_engines.gateway("vps")
    try:
        checking = rc.session_checking(client, custody_drive=data, task_id="t-seat", operation_id="op-remote",
                                       root=str(author), source=source, shape=shape, route_id="codex")
    finally:
        client.close()
    invocation = SessionInvocation(task_id="t-seat", surface="multi_model_review", slot_id="remote-reviewer",
                                   timeout_sec=600, session_route=route, access="workspace_write")
    request = prepare_review_session_request(invocation, route, prompt="review the staged change", root=str(author),
                                             thread_id="", schema_asked=False, ephemeral=True, shape=shape,
                                             execution=checking["execution"], instructions=checking["instructions"])
    handle, ref = _send(data, request, operation="op-remote", invocation="inv-seat", custody_facts=checking["custody"])
    return SimpleNamespace(author=author, data=data, source=source, checking=checking, request=request, ref=ref,
                           run_id=handle["runId"], work=host.remote_root / "inv-seat" / "work",
                           area=pathlib.Path(checking["execution"]["workspaceRoot"]))


def _review_on_host(work, name="report.md", text="what I checked on the host"):
    """What a reviewer leaves in its host copy: a changed and a new file, an ignored output and its evidence."""
    (work / "f.txt").write_text("v2 by the reviewer\n")
    (work / "new.txt").write_text("a reproduction\n")
    (work / "build").mkdir(exist_ok=True)
    (work / "build" / "cache.bin").write_bytes(b"\0" * 64)  # Git-ignored on the host: named, never sent
    (work / ".review-evidence").mkdir(exist_ok=True)
    (work / ".review-evidence" / name).write_text(text)


def _started(seat, run_id, *, invocation="inv-seat", continuation_of=""):
    entry = custody.RunCustody(run_id=run_id, task_id="t-seat", invocation_id=invocation, route_id="codex",
                               model="gpt-6", access="workspace_write", mode="agent", ledger_root=str(seat.data),
                               engine_ref=dict(seat.ref), daemon_store_id=STORE, continuation_of=continuation_of,
                               **seat.checking["custody"])
    assert custody.record_started(seat.data, entry)
    assert custody.emit(seat.data, custody.SETTLED, {"run_id": run_id, "task_id": "t-seat"})


def test_a_remote_seat_is_told_what_its_host_copy_holds_and_where_the_subject_change_is(host, tmp_path):
    """The engine's actual staged start body: a remote seat is never told that ``git diff --cached``, the frozen
    parent or the original root shows the subject (on that host its copy is one fresh commit, with no parent object
    and no original root); it is pointed at the brief's ``.review-drive/`` view, which travelled byte-exact with its
    copy. A local seat's words are unchanged."""
    seat = _remote_seat(host, tmp_path)
    sent = next(call["body"] for call in host.state["calls"] if call["path"] == "/v2/runs" and call["key"] == "inv-seat")
    words = sent["instructions"]
    for local_only in ("`git diff --cached` is the subject", "always is", "linked worktree", "is read-only reference"):
        assert local_only not in words
    assert "fresh Git repository whose one commit is this copy as the run started" in words
    assert "the `.review-drive/` view it names in this copy" in words and f"{seat.area} -> {seat.work}" in words
    assert (seat.work / ".review-drive" / "t-seat" / f"{_VIEW}.txt").read_bytes() == _DIFF  # the change, readable
    assert _git(seat.work, "rev-list", "--count", "HEAD").strip() == "1" and not _git(seat.work, "diff", "--cached")
    assert _git(seat.work, "show", "HEAD:f.txt") == "v1 subject\n"  # the subject's files as frozen
    assert subprocess.run(["git", "-C", str(seat.work), "cat-file", "-e", f"{seat.source['parent']}^{{commit}}"],
                          capture_output=True).returncode != 0  # the parent the local words name is not there
    local = rc.checking_instructions(str(seat.area), seat.source, str(seat.author), "⟨scratch⟩")
    assert local == rc.checking_instructions(str(seat.area), seat.source, str(seat.author), "⟨scratch⟩", remote=False)
    assert "so `git diff --cached` is the subject until you change it" in local and "linked worktree" in local


def test_a_remote_seats_work_comes_home_through_its_capture_before_either_copy_goes(host, tmp_path, monkeypatch):
    """The terminal capture brings the host copy's changes and ``.review-evidence/`` into the seat's own copy and
    retains them under a receipt naming the run, the patch and the evidence by digest, and the host copy's ignored
    outputs by name only (as a local capture names them); only then does the local copy go, and only after that
    receipt may the reconcile sweep retire the host copy. Nothing reaches the product."""
    from ouroboros import delegate_continuation
    from ouroboros.delegate_custody_reconcile import reconcile_orphaned_runs

    seat = _remote_seat(host, tmp_path)
    _review_on_host(seat.work)
    _started(seat, seat.run_id)
    host.state["run_states"] = {seat.run_id: "succeeded"}
    monkeypatch.setattr(delegate_continuation, "still_continuable", lambda *_a, **_k: False)
    assert remote_workspace.copy_hold(seat.data, current.run(seat.data, seat.run_id)) == "review_capture_owed"
    reconcile_orphaned_runs(seat.data, running_task_ids=set(), gateway_factory=_owned_gateway)
    assert seat.work.is_dir()  # the host copy waits for the seat's receipt

    done = rc.terminal_seat_evidence(seat.data, "t-seat", "op-remote", {"summary": {"state": "succeeded"}})
    assert done["status"] == "retained", done
    returned = done["remote_result"]
    assert (returned["run_id"], returned["state"], returned["evidence"]["state"]) == (seat.run_id, "applied", "returned")
    assert {"M\tf.txt", "A\tnew.txt"} <= set(done["changed"]) and done["ignored_not_retained"] == ["build/"]
    kept = pathlib.Path(done["evidence_dir"])
    assert (kept / "files" / ".review-evidence" / "report.md").read_text() == "what I checked on the host"
    assert not [path for path in kept.rglob("*") if "build" in path.relative_to(kept).parts]  # bytes never sent
    patch = (kept / "remote" / remote_workspace.RESULT_PATCH_NAME).read_bytes()
    assert hashlib.sha256(patch).hexdigest() == returned["sha256"] and b".review-evidence" not in patch
    assert b"cache.bin" not in patch and not seat.area.exists()  # the receipt came first, then the copy went
    assert (seat.author / "f.txt").read_text() == "v1 subject\n" and not (seat.author / "new.txt").exists()
    assert remote_workspace.copy_hold(seat.data, current.run(seat.data, seat.run_id)) == ""
    reconcile_orphaned_runs(seat.data, running_task_ids=set(), gateway_factory=_owned_gateway)
    assert not seat.work.parent.exists() and current.run(seat.data, seat.run_id).remote_copy_retired


def test_a_remote_seat_whose_work_cannot_come_home_keeps_both_copies(host, tmp_path, monkeypatch):
    """A start not yet recorded as started, a host whose evidence cannot be read, or a host copy Git cannot fully
    list never reads as complete: the capture fails closed with the reason, the local copy stays, the host copy is
    held for it; a later capture (the prune pass's retry) brings everything home."""
    from ouroboros import delegate_continuation

    seat = _remote_seat(host, tmp_path)
    _review_on_host(seat.work)
    terminal = {"summary": {"state": "succeeded"}}
    unbound = rc.terminal_seat_evidence(seat.data, "t-seat", "op-remote", terminal)
    assert unbound["status"] == "capture_failed" and "start_unbound" in unbound["error"] and seat.area.is_dir()
    _started(seat, seat.run_id)
    monkeypatch.setattr(delegate_continuation, "still_continuable", lambda *_a, **_k: False)
    with monkeypatch.context() as lost:
        lost.setattr(remote_workspace, "_return_evidence",
                     lambda *_a: {"state": "unreachable", "code": "remote_ssh_timeout"})
        unread = rc.capture_seat_evidence(seat.data, "t-seat", "op-remote")
    assert unread["status"] == "capture_failed" and "evidence_unread" in unread["error"]
    assert seat.area.is_dir() and not (seat.area / "new.txt").exists()  # nothing applied while the evidence is away
    locked = seat.work / "locked"  # a folder the host's Git cannot open: its outputs are unknown, never "none"
    locked.mkdir()
    locked.chmod(0)
    try:
        unlisted = rc.capture_seat_evidence(seat.data, "t-seat", "op-remote")
    finally:
        locked.chmod(0o700)
    assert unlisted["status"] == "capture_failed" and "unlisted" in unlisted["error"] and seat.area.is_dir()
    assert "ignored_not_retained" not in unlisted or unlisted["ignored_not_retained"] == []
    assert remote_workspace.copy_hold(seat.data, current.run(seat.data, seat.run_id)) == "review_capture_owed"
    locked.rmdir()
    done = rc.capture_seat_evidence(seat.data, "t-seat", "op-remote")
    assert done["status"] == "retained" and done["remote_result"]["state"] in ("applied", "already_applied")
    assert done["ignored_not_retained"] == ["build/"] and not seat.area.exists()
    assert remote_workspace.copy_hold(seat.data, current.run(seat.data, seat.run_id)) == ""


def test_an_explicit_successor_holds_the_copy_and_brings_its_own_work_home(host, tmp_path, monkeypatch):
    """Explicit Continue of a stopped remote seat (``review_continuation``): the stopped run's capture brings its work
    into the held copy; the copy passes to the successor (``transfer_checking_copy``), whose recorded start (the
    original request, continued) restages it on the same engine; both host copies stay until the successor's own
    receipt, which brings the successor's run (the chain's tail) home with both turns' evidence."""
    from ouroboros import delegate_continuation
    from ouroboros.review_continuation import transfer_checking_copy

    seat = _remote_seat(host, tmp_path)
    _review_on_host(seat.work, "first.md", "the first turn")
    _started(seat, seat.run_id)
    monkeypatch.setattr(delegate_continuation, "still_continuable", lambda *_a, **_k: False)
    stopped = {"summary": {"state": "interrupted", "resumable": True}}
    held = rc.terminal_seat_evidence(seat.data, "t-seat", "op-remote", stopped, hold=True)
    assert held["status"] == "retained" and held["remote_result"]["run_id"] == seat.run_id
    assert (seat.area / "new.txt").is_file()  # the first turn's work waits in the held copy
    transfer_checking_copy(seat.data, "t-seat", "op-remote", "op-next")
    body = {**seat.request, "continueFrom": seat.run_id}
    handle, _ref = _send(seat.data, body, operation="op-next", invocation="inv-next",
                         custody_facts=seat.checking["custody"], continuation_of=seat.run_id)
    following = host.remote_root / "inv-next" / "work"
    assert (following / "new.txt").read_text() == "a reproduction\n"  # restaged from the held copy
    (following / "next.txt").write_text("the successor's check\n")
    (following / ".review-evidence" / "second.md").write_text("the second turn")
    _started(seat, handle["runId"], invocation="inv-next", continuation_of=seat.run_id)
    for run_id in (seat.run_id, handle["runId"]):  # the copy's current holder owes its receipt
        assert remote_workspace.copy_hold(seat.data, current.run(seat.data, run_id)) == "review_capture_owed"
    done = rc.terminal_seat_evidence(seat.data, "t-seat", "op-next", {"summary": {"state": "succeeded"}})
    assert done["status"] == "retained" and done["remote_result"]["run_id"] == handle["runId"], done
    assert {"A\tnext.txt", "A\tnew.txt", "M\tf.txt"} <= set(done["changed"]) and not seat.area.exists()
    kept = pathlib.Path(done["evidence_dir"]) / "files" / ".review-evidence"
    assert (kept / "first.md").read_text() == "the first turn" and (kept / "second.md").read_text() == "the second turn"
    for run_id in (seat.run_id, handle["runId"]):
        assert remote_workspace.copy_hold(seat.data, current.run(seat.data, run_id)) == ""


def _local_seat(tmp_path, monkeypatch):
    """A writable seat on the owned engine with the reviewer's local edit: ``(data, area)``; a remote return asked for
    on its behalf fails the test."""
    author = _author(tmp_path)
    data = tmp_path / "data"
    source = rc.freeze_checking_source(author, data_root=data)
    area = pathlib.Path(rc.prepare_seat_area(data, "t-local", "op-local", source))

    def _never(*_args, **_kwargs):
        raise AssertionError("a local seat's capture asked for a remote return")

    monkeypatch.setattr(remote_workspace, "mirror_seat_result", _never, raising=False)
    (area / "f.txt").write_text("v1 locally\n")
    return data, area


def test_a_local_seats_capture_never_asks_for_a_remote_return(tmp_path, monkeypatch):
    """A seat on the owned engine is prepared, captured and released exactly as before: no remote mark, no remote
    return asked for, its own delta and the receipt as they were."""
    data, area = _local_seat(tmp_path, monkeypatch)
    assert "remote" not in json.loads((area.parent / "checking.json").read_text())
    done = rc.terminal_seat_evidence(data, "t-local", "op-local", {"summary": {"state": "succeeded"}})
    assert done["status"] == "retained" and "M\tf.txt" in done["changed"] and "remote_result" not in done
    assert not area.exists()


def test_the_host_copy_hold_follows_a_seat_copys_receipt_and_nothing_else(tmp_path, monkeypatch):
    """The hold reads the seat copy a run worked in: owed until that copy's receipt stands; a run in no seat copy (a
    delegated run's own copy, or none) owes it nothing."""
    from ouroboros.review_checking_prune import seat_capture_owed

    data, area = _local_seat(tmp_path, monkeypatch)
    row = SimpleNamespace(execution_root=str(area))
    assert seat_capture_owed(data, row)  # its copy has no receipt yet
    assert rc.terminal_seat_evidence(data, "t-local", "op-local", {"summary": {"state": "succeeded"}})["status"] == \
        "retained"
    assert not seat_capture_owed(data, row)
    assert not seat_capture_owed(data, SimpleNamespace(execution_root=str(tmp_path / "copies" / "a-run")))
    assert not seat_capture_owed(data, SimpleNamespace(execution_root=""))
