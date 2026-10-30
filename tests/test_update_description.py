"""The update description as the mind's own work (Ouroboros#2389).

The one writer (``record_update_description`` -> ``update_letter.save_description``), the owner's
request from the Updates panel (``POST /api/update/description`` -> one ordinary Main task through
the queue's receipt-bearing admission) and the worker-to-window refresh, each through its real
consumer: the tool registry with its existing gates, the gateway routes with the real queue door,
the supervisor dispatch table, real Git and real processes for the lock. Both directions for each
rule: what a call writes AND what it leaves alone.
"""
from __future__ import annotations

import asyncio
import json
import queue as queue_module
import subprocess
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from ouroboros import update_letter as ul
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools.registry import ToolContext, ToolRegistry

BASE, TARGET, NEWER = "a" * 40, "b" * 40, "c" * 40
REPO_ROOT = Path(__file__).resolve().parents[1]


def _key(base=BASE, target=TARGET):
    return {"base_sha": base, "target_sha": target, "update_channel": "stable", "target_ref": "managed/main"}


def _offer(drive, base=BASE, target=TARGET, **over):
    """What a fetching check leaves: the range on offer with its material, nothing written yet."""
    record = {"schema": 1, "state": "none", "text": "", "last_good": None, "key": _key(base, target),
              "checked_key": _key(base, target), "checked_target_version": "6.114.0",
              "material": {"key": _key(base, target), "text_bytes": 4321}, **over}
    (drive / "state").mkdir(parents=True, exist_ok=True)
    ul.atomic_write_json(ul.record_path(drive), record)
    return record


def _versions(argv):
    return (0, "6.114.0", "") if argv[:2] == ["git", "show"] else (1, "", "")


@pytest.fixture(autouse=True)
def _no_git_for_versions(monkeypatch):
    monkeypatch.setattr(ul, "_default_git", lambda: _versions)


def _registry(repo, drive, **fields):
    repo.mkdir(parents=True, exist_ok=True)
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)
    fields.setdefault("task_id", "turn-1")
    reg.set_context(ToolContext(repo_dir=repo, drive_root=drive, **fields))
    return reg


def _save(reg, text, *, base=BASE, target=TARGET, expected=0):
    return reg.execute_result("record_update_description", {
        "text": text, "base_sha": base, "target_sha": target, "expected_revision": expected})


def _ok(result):
    assert result.status == "ok", result.text
    return json.loads(result.text)


# --- the tool: registration and the existing gates ---------------------------------------


def test_the_writer_is_one_tool_outside_core_children_and_observe():
    from ouroboros import consciousness_authority as ca
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        CORE_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
        OBSERVE_WORLD_MUTATION_TOOLS,
    )
    from ouroboros.tools.control import get_tools

    entries = [entry for entry in get_tools() if entry.name == "record_update_description"]
    assert len(entries) == 1
    schema = entries[0].schema["parameters"]
    assert set(schema["required"]) == {"base_sha", "target_sha", "expected_revision"}
    assert {"text", "text_file"} <= set(schema["properties"]) and schema["properties"]["text_file"]["required"] == ["path"]
    assert "record_update_description" in OBSERVE_WORLD_MUTATION_TOOLS
    assert "record_update_description" in ca.disabled_tools_for("observe")
    assert "record_update_description" not in ca.disabled_tools_for("act")
    assert "record_update_description" not in CORE_TOOL_NAMES | LOCAL_READONLY_SUBAGENT_TOOL_NAMES | ACTING_SUBAGENT_TOOL_NAMES


def test_children_and_an_observe_wake_cannot_write_it(tmp_path, monkeypatch):
    from ouroboros import consciousness_authority as ca

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    drive = tmp_path / "data"
    _offer(drive)
    before = ul.record_path(drive).read_bytes()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    readonly = _registry(tmp_path / "repo", drive, task_constraint=TaskConstraint(mode="local_readonly_subagent"))
    acting = _registry(tmp_path / "repo", drive, workspace_root=str(worktree), workspace_mode="self_worktree",
                       task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree",
                                                      write_root=str(worktree)))
    for reg, why in ((readonly, "read-only subagent"), (acting, "acting subagent")):
        assert why in reg.policy_hidden_reason("record_update_description")
        assert _save(reg, "a child's text").status != "ok"
    wake = {"id": "wake-1", "type": "task", "text": "wake", "_is_direct_chat": True, "chat_id": 1, "metadata": {
        "initiator": "consciousness", "usage_category": "consciousness", "wake_reason": "heartbeat",
        "consciousness_autonomy": "observe", "model_role": "consciousness"}}
    observe = _registry(tmp_path / "repo", drive, is_direct_chat=True,
                        task_metadata=ca.apply_consciousness_authority(wake)["metadata"])
    assert "RESOURCE_CONSTRAINT_BLOCKED" in observe.execute("record_update_description", {
        "text": "observe's text", "base_sha": BASE, "target_sha": TARGET, "expected_revision": 0})
    assert ul.record_path(drive).read_bytes() == before


# --- the tool: what a save writes and what it leaves alone -------------------------------


def test_a_save_lands_at_the_canonical_root_with_host_derived_authorship(tmp_path):
    canonical = tmp_path / "data"
    _offer(canonical)
    fork = canonical / "state" / "headless_tasks" / "turn-1" / "data"
    fork.mkdir(parents=True)
    events = queue_module.Queue()
    reg = _registry(tmp_path / "repo", fork, budget_drive_root=str(canonical), event_queue=events)
    reg._ctx.active_model = "anthropic/claude-opus-5.5"
    receipt = _ok(_save(reg, "## What changes\n\n- The Check button reads the channel."))
    assert receipt["revision"] == 1 and receipt["key"]["target_sha"] == TARGET and receipt["replaced"] is None
    assert receipt["window_refresh"] == "queued_for_host" and events.get_nowait() == {"type": "update_progress_changed"}
    record = ul.read_record(canonical)
    assert record["state"] == "ready" and record["text"].startswith("## What changes") and record["revision"] == 1
    assert record["author"] == {"task_id": "turn-1", "model": "anthropic/claude-opus-5.5", "requested": False}
    assert record["target_version"] == "6.114.0" and record["checked_key"] == _key()
    assert not ul.record_path(fork).exists(), "a forked drive never holds the installation's description"
    view = ul.project_letter(record, head_sha=BASE, latest_sha=TARGET)
    assert view["description_current"] is True and view["relation"] == "pending" and view["author"]["task_id"] == "turn-1"


def test_a_stale_author_cannot_erase_a_newer_description_and_the_replaced_one_returns_whole(tmp_path):
    drive = tmp_path / "data"
    _offer(drive)
    first = _registry(tmp_path / "repo", drive, task_id="wake-7")
    second = _registry(tmp_path / "repo", drive, task_id="request-task")
    assert _ok(_save(first, "the wake's text"))["revision"] == 1
    late = _save(second, "the request task's text, written against revision 0", expected=0)
    assert late.status == "blocked" and "revision conflict" in late.text and '"revision": 1' in late.text
    assert "the wake's text" in late.text, "the refusal shows what is saved instead"
    assert ul.read_record(drive)["text"] == "the wake's text"
    replaced = _ok(_save(second, "a rewrite that read revision 1", expected=1))
    assert replaced["revision"] == 2 and replaced["replaced"]["text"] == "the wake's text"
    assert replaced["replaced"]["author"]["task_id"] == "wake-7"


def test_a_file_the_writer_wrote_is_saved_as_its_text_trimmed_like_text(tmp_path):
    """Already-written prose is saved from its file instead of being emitted again: written by the real
    write_file into the task's own drive and addressed as write_file named it, or by the absolute path
    the receipt gives back. The bytes arrive as written (CRLF, a tab, non-ASCII, inner blank lines)
    under the same strip, range, revision and authorship as text=; the receipt names what was read."""
    import hashlib

    drive = tmp_path / "data"
    _offer(drive)
    reg = _registry(tmp_path / "repo", drive, task_id="request-task")
    wrote = reg.execute_result("write_file", {"root": "task_drive", "path": "letter.md", "content": "a draft"})
    assert wrote.status == "ok" and "task_drive:letter.md" in wrote.text, wrote.text
    draft = drive / "task_drives" / "request-task" / "letter.md"
    raw = "\n\n## Что изменится\r\n\r\n- Первое\t«кавычки» ✓\r\n\r\n\r\n- Второе\n\n".encode("utf-8")
    draft.write_bytes(raw)
    receipt = _ok(reg.execute_result("record_update_description", {
        "text_file": {"root": "task_drive", "path": "letter.md"}, "base_sha": BASE, "target_sha": TARGET,
        "expected_revision": 0}))
    record = ul.read_record(drive)
    assert record["text"] == raw.decode("utf-8").strip() and "\r\n\r\n\r\n- Второе" in record["text"]
    assert record["revision"] == receipt["revision"] == 1 and record["author"]["task_id"] == "request-task"
    assert receipt["text_file"] == {"root": "task_drive", "path": str(draft.resolve()), "bytes": len(raw),
                                    "sha256": hashlib.sha256(raw).hexdigest()}
    assert draft.read_bytes() == raw, "the source file is read, never moved or rewritten"
    again = _ok(reg.execute_result("record_update_description", {
        "text_file": {"path": receipt["text_file"]["path"]}, "base_sha": BASE, "target_sha": TARGET,
        "expected_revision": 1}))
    assert again["text_file"]["root"] == "task_drive" and again["replaced"]["text"] == record["text"]
    assert ul.read_record(drive)["revision"] == 2


def test_text_file_is_refused_where_read_file_is_and_never_beside_text(tmp_path):
    drive = tmp_path / "data"
    _offer(drive)
    before = ul.record_path(drive).read_bytes()
    reg = _registry(tmp_path / "repo", drive)
    store = drive / "projects" / "p1"
    store.mkdir(parents=True)
    (store / "facts.md").write_text("another room's facts", encoding="utf-8")
    address = {"root": "runtime_data", "path": "projects/p1/facts.md"}
    assert "ACCESS_DENIED" in reg.execute_result("read_file", address).text
    refused = reg.execute_result("record_update_description", {
        "text_file": address, "base_sha": BASE, "target_sha": TARGET, "expected_revision": 0})
    assert refused.status == "blocked" and "ACCESS_DENIED" in refused.text and "nothing was saved" in refused.text
    scratch = drive / "task_drives" / "turn-1"
    scratch.mkdir(parents=True)
    (scratch / "latin1.md").write_bytes("caf\xe9".encode("latin-1"))
    (scratch / "blank.md").write_text(" \n\n", encoding="utf-8")
    (scratch / "real.md").write_text("the file's text", encoding="utf-8")
    for args, words in (({"text": "inline", "text_file": {"root": "task_drive", "path": "real.md"}}, "not both"),
                        ({"text_file": {"root": "task_drive", "path": "absent.md"}}, "absent.md"),
                        ({"text_file": {"root": "task_drive", "path": "latin1.md"}}, "not UTF-8"),
                        ({"text_file": {"root": "task_drive", "path": "blank.md"}}, "text is empty"),
                        ({"text_file": {"root": "nowhere", "path": "real.md"}}, "nowhere")):
        result = reg.execute_result("record_update_description", {
            **args, "base_sha": BASE, "target_sha": TARGET, "expected_revision": 0})
        assert result.status == "error" and words in result.text, (args, result.text)
    assert ul.record_path(drive).read_bytes() == before


def test_text_file_takes_each_spelling_read_file_takes_in_a_bound_body_candidate(tmp_path):
    """A bound candidate keeps the serving spelling of a body path (``serving/x``, the absolute serving
    path) naming the candidate's own file. text_file is read through read_file's dispatch normalization,
    so every spelling read_file reads saves that same file, and a file read_file cannot read stays unread."""
    drive, serving, candidate = (tmp_path / name for name in ("data", "serving", "candidate"))
    serving.mkdir()
    candidate.mkdir()
    _offer(drive)
    reg = ToolRegistry(repo_dir=candidate, drive_root=drive)
    reg.set_context(ToolContext(repo_dir=candidate, system_repo_dir=candidate, serving_repo_dir=serving, drive_root=drive,
                                task_id="request-task", task_metadata={
                                    "body_candidate": {"path": str(candidate), "repo_dir": str(serving)}}))
    letter = candidate / "letter.md"
    spellings = ("letter.md", "serving/letter.md", str(serving / "letter.md"), str(letter))
    for revision, path in enumerate(spellings):
        letter.write_text(f"The candidate's description, read as {path}\n", encoding="utf-8")
        address = {"root": "active_workspace", "path": path}
        assert reg.execute_result("read_file", address).status == "ok", path
        receipt = _ok(reg.execute_result("record_update_description", {
            "text_file": address, "base_sha": BASE, "target_sha": TARGET, "expected_revision": revision}))
        assert receipt["text_file"]["path"] == str(letter.resolve()), path
        assert ul.read_record(drive)["text"] == f"The candidate's description, read as {path}"
    (serving / "only-serving.md").write_text("the serving checkout's own file", encoding="utf-8")
    address = {"root": "active_workspace", "path": str(serving / "only-serving.md")}
    assert "NOT_FOUND" in reg.execute_result("read_file", address).text, "the serving spelling names the candidate's file"
    unread = reg.execute_result("record_update_description", {
        "text_file": address, "base_sha": BASE, "target_sha": TARGET, "expected_revision": len(spellings)})
    assert unread.status == "error" and "nothing was saved" in unread.text, unread.text
    assert ul.read_record(drive)["revision"] == len(spellings)


def test_only_ranges_someone_offered_can_be_named(tmp_path):
    drive = tmp_path / "data"
    _offer(drive)
    reg = _registry(tmp_path / "repo", drive)
    other = _save(reg, "text about a range nobody offered", target=NEWER)
    assert other.status == "blocked" and "range not offered" in other.text and TARGET in other.text
    assert _save(reg, "short sha", target="b" * 12).status == "error"
    assert _save(reg, "   ").status == "error"
    assert reg.execute_result("record_update_description", {
        "text": "x", "base_sha": BASE, "target_sha": TARGET, "expected_revision": -1}).status == "error"
    assert ul.read_record(drive)["state"] == "none" and "revision" not in ul.read_record(drive)


def test_an_applied_description_can_be_rephrased_for_its_own_range_only(tmp_path):
    # After the update landed the check offers nothing (base == target); the text that described
    # it stays and may be rewritten, but never relabelled onto a range nobody offered.
    drive = tmp_path / "data"
    _offer(drive, state="ready", text="about b", revision=3, author={"task_id": "t", "model": "m", "requested": True},
           checked_key=_key(TARGET, TARGET), target_in_running=True, target_in_for=TARGET)
    reg = _registry(tmp_path / "repo", drive)
    assert ul.legal_ranges(ul.read_record(drive)) == [_key()]
    assert _ok(_save(reg, "about b, said better", expected=3))["revision"] == 4
    assert _save(reg, "about c", base=TARGET, target=NEWER, expected=4).status == "blocked"
    view = ul.project_letter(ul.read_record(drive), head_sha=TARGET, latest_sha=TARGET)
    assert view["relation"] == "applied" and view["text"] == "about b, said better"


_RACER = """
import json, sys
from pathlib import Path
from ouroboros import update_letter as ul
ul._default_git = lambda: (lambda argv: (1, "", ""))
drive, name = Path(sys.argv[1]), sys.argv[2]
out = ul.save_description("text of " + name, sys.argv[3], sys.argv[4], 0, author={"task_id": name, "model": "m"},
                          drive_root=drive)
print(json.dumps({"name": name, "outcome": out["outcome"]}))
"""


def test_concurrent_processes_save_once_per_revision(tmp_path):
    """The lock is a file every process takes: four writers that all read revision 0 race, one wins,
    the rest are refused with the winner's text kept — never a silent overwrite."""
    drive = tmp_path / "data"
    _offer(drive)
    procs = [subprocess.Popen([sys.executable, "-c", _RACER, str(drive), f"writer-{n}", BASE, TARGET],
                              cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for n in range(4)]
    results = []
    for proc in procs:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err
        results.append(json.loads(out.strip().splitlines()[-1]))
    outcomes = sorted(r["outcome"] for r in results)
    assert outcomes == ["revision_conflict"] * 3 + ["saved"], results
    winner = next(r["name"] for r in results if r["outcome"] == "saved")
    record = ul.read_record(drive)
    assert record["revision"] == 1 and record["text"] == f"text of {winner}" and record["author"]["task_id"] == winner


# --- the owner's request: one ordinary Main task through the real queue door -------------


@pytest.fixture()
def door(tmp_path, monkeypatch):
    from supervisor import queue, state

    root = tmp_path / "data"
    root.mkdir()
    state.init(root)
    state.save_state({})
    queue.init(root)
    pending: list = []
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    state.update_state(lambda live: live.update(owner_chat_id=1))
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    pool = {"available": True, "disabled_reason": ""}
    monkeypatch.setattr("supervisor.workers._worker_pool_execution_state", lambda: dict(pool))
    monkeypatch.setattr("ouroboros.config.DATA_DIR", root)
    frames: list = []
    monkeypatch.setattr("ouroboros.gateway.update_progress.notify_changed", lambda: frames.append("update_progress_changed"))
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway.router import collect_routes

    app = Starlette(routes=collect_routes(data_dir=root))
    app.state.drive_root = root
    with TestClient(app) as client:
        yield types.SimpleNamespace(root=root, pending=pending, client=client, pool=pool, frames=frames)


def _request(door, mode="write", base=BASE, target=TARGET):
    return door.client.post("/api/update/description", json={"mode": mode, "base_sha": base, "target_sha": target})


def _finish(root, task_id, status):
    from ouroboros.task_results import write_task_result

    write_task_result(root, task_id, status, result="the task's own end")


def test_write_admits_one_ordinary_main_task_and_records_the_request(door, tmp_path):
    from ouroboros.task_results import load_task_result

    _offer(door.root)
    response = _request(door)
    assert response.status_code == 200, response.text
    body = response.json()
    task_id = body["task_id"]
    assert body["status"] == "queued" and body["key"] == _key() and len(task_id) == 8
    [task] = door.pending
    assert (task["id"], task["type"], task["chat_id"]) == (task_id, "task", 1)
    assert (task["root_task_id"], task["delegation_role"]) == (task_id, "root"), "a typed root, like any host start"
    assert task["metadata"]["update_description"]["key"] == _key()
    assert task["metadata"]["update_description"]["revision_at_request"] == 0
    for part in (f"{BASE}..{TARGET}", 'kind="update_material"', "record_update_description",
                 "expected_revision=0", "text_file=", 'vcs_read(root="system_repo"', "max_bytes=4321)",
                 "No check facts for this range are recorded."):
        assert part in task["text"], part
    for template in ("120 words", "ONE short paragraph", "Name the commits or PRs", "no headings"):
        assert template not in task["text"]
    assert load_task_result(door.root, task_id, strict=True)["status"] == "scheduled"
    assert ul.read_record(door.root)["request"]["task_id"] == task_id and door.frames == ["update_progress_changed"]
    # A second press while the task can still save answers with that task, admitting nothing.
    again = _request(door, mode="rewrite")
    assert again.status_code == 409 and again.json()["task_id"] == task_id and len(door.pending) == 1
    # The requested task's own save is marked as the owner's request.
    reg = _registry(tmp_path / "repo", door.root, task_id=task_id)
    _ok(_save(reg, "what the requested task found"))
    assert ul.read_record(door.root)["author"]["requested"] is True
    _finish(door.root, task_id, "completed")
    view = ul.project_letter_for_panel({"running_sha": BASE, "checked_target_sha": TARGET}, drive_root=door.root)
    assert view["request"] == {"task_id": task_id, "current_task_id": task_id, "mode": "write", "key": _key(),
                               "requested_at": view["request"]["requested_at"], "status": "completed", "saved": True}
    assert view["description_current"] is True
    # Once it ended, the owner may ask again: a new host-minted id, never derived from content.
    later = _request(door, mode="rewrite")
    assert later.status_code == 200 and later.json()["task_id"] != task_id and len(door.pending) == 2
    assert door.pending[1]["metadata"]["update_description"]["revision_at_request"] == 1


@pytest.mark.parametrize(("binding", "chat"), [("known_empty", 1), ("web", 1), ("external", 77), ("unconfirmed", 0)])
def test_write_is_main_work_until_an_owner_message_binds_a_chat(door, binding, chat):
    """Write before any owner message (the binding known empty) is Main's work, where the panel's door
    opens it. A bound chat keeps it (a transport's own chat too); a binding the state cannot vouch for
    is used as read, never taken for empty, so that work stays in the hidden partition as before."""
    from supervisor import state

    def bind(live):
        live["owner_chat_id"] = {"known_empty": None, "web": 1, "external": 77, "unconfirmed": None}[binding]
        if binding == "unconfirmed":
            state.mark_unconfirmed(live, "owner_chat_id")

    state.update_state(bind)
    _offer(door.root)
    assert _request(door).status_code == 200
    [task] = door.pending
    assert task["chat_id"] == chat


def test_a_refused_admission_and_a_range_nobody_offered_record_nothing(door):
    _offer(door.root)
    before = ul.record_path(door.root).read_bytes()
    door.pool.update(available=False, disabled_reason="worker_readiness_exhausted")
    refused = _request(door)
    assert refused.status_code == 503 and refused.json()["reason"] == "worker_readiness_exhausted"
    assert door.pending == [] and ul.record_path(door.root).read_bytes() == before
    door.pool.update(available=True, disabled_reason="")
    stranger = _request(door, target=NEWER)
    assert stranger.status_code == 409 and stranger.json()["legal_ranges"] == [_key()] and door.pending == []
    assert _request(door, mode="publish").status_code == 400 and door.frames == []


def test_a_failed_request_keeps_the_saved_text_and_nothing_repeats(door, tmp_path):
    _offer(door.root, state="ready", text="the saved description", revision=2,
           author={"task_id": "earlier", "model": "m", "requested": False})
    task_id = _request(door, mode="rewrite").json()["task_id"]
    assert "my human asked for a new one" in door.pending[0]["text"]
    _finish(door.root, task_id, "failed")
    status = {"running_sha": BASE, "checked_target_sha": TARGET, "check_ok": None}
    view = ul.project_letter_for_panel(status, drive_root=door.root)
    assert view["text"] == "the saved description" and view["revision"] == 2 and view["description_current"]
    assert view["request"]["status"] == "failed" and view["request"]["saved"] is False
    # Reads and a failed check admit nothing on their own.
    ul.refresh_after_check({**status, "check_ok": False}, drive_root=door.root)
    ul.official_update_projection(BASE, drive_root=door.root, state={})
    assert len(door.pending) == 1


def test_a_stored_host_retry_is_the_request_s_own_work_and_its_end_ends_the_request(door, tmp_path):
    """Read the reciprocal rows older reapers published, without recreating their retired admission.
    The retry's save belongs to the request; later initiative leaves that fact intact."""
    from ouroboros.task_results import load_task_result, write_task_result

    _offer(door.root)
    handle = _request(door).json()["task_id"]
    task = door.pending.pop(0)
    write_task_result(door.root, handle, "running")
    retry_id = f"{handle}-retry"
    retried = {**task, "id": retry_id, "original_task_id": handle, "timeout_retry_from": handle, "_attempt": 2}
    write_task_result(door.root, retry_id, "scheduled", supersedes_task_id=handle,
                      original_task_id=handle, root_task_id=task.get("root_task_id") or handle,
                      parent_task_id=task.get("parent_task_id"), delegation_role=task.get("delegation_role"),
                      timeout_retry_from=handle, metadata=task.get("metadata", {}))
    write_task_result(door.root, handle, "interrupted", reason_code="idle_timeout_retry",
                      superseded_by=retry_id, retry_task_id=retry_id)
    door.pending.append(retried)
    status = {"running_sha": BASE, "checked_target_sha": TARGET}
    assert ul.project_letter_for_panel(status, drive_root=door.root)["request"]["status"] == "scheduled"
    assert _request(door, mode="rewrite").status_code == 409 and [t["id"] for t in door.pending] == [retry_id]
    _ok(_save(_registry(tmp_path / "repo", door.root, task_id=retry_id), "the retry's description"))
    record = ul.read_record(door.root)
    assert (record["author"]["task_id"], record["author"]["requested"], record["request_saved"]) == (retry_id, True, handle)
    _finish(door.root, retry_id, "completed")
    assert load_task_result(door.root, handle, strict=True)["status"] == "interrupted", "the original row stays as written"
    view = ul.project_letter_for_panel(status, drive_root=door.root)
    assert (view["request"]["task_id"], view["request"]["status"], view["request"]["saved"]) == (handle, "completed", True)
    _ok(_save(_registry(tmp_path / "repo", door.root, task_id="chat-turn"), "Main's shorter rewrite", expected=1))
    view = ul.project_letter_for_panel(status, drive_root=door.root)
    assert view["author"] == {"task_id": "chat-turn", "model": "", "requested": False} and view["request"]["saved"] is True
    again = _request(door, mode="rewrite")
    assert again.status_code == 200 and again.json()["task_id"] not in (handle, retry_id)
    assert ul.project_letter_for_panel(status, drive_root=door.root)["request"]["saved"] is False, "a new request starts unsaved"


def test_a_recorded_continue_of_the_request_saves_as_the_request(door, tmp_path):
    """The owner's Continue after a technical interruption is a new root bound to the request's own
    result; its save is the request's, while an unrelated focus's save before it is not."""
    from ouroboros.owner_continue import claim_on_predecessor, successor_id
    from ouroboros.task_results import write_task_result

    _offer(door.root)
    handle = _request(door).json()["task_id"]
    write_task_result(door.root, handle, "running")
    write_task_result(door.root, handle, "cancelled", cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    successor = successor_id(handle, "continue-nonce-1")
    claim_on_predecessor(door.root, handle, "continue-nonce-1", {
        "verb": "continue", "binding_version": 1, "predecessor_task_id": handle,
        "action_nonce": "continue-nonce-1", "successor_task_id": successor, "cause": "snapshot_restore"})
    status = {"running_sha": BASE, "checked_target_sha": TARGET}
    _ok(_save(_registry(tmp_path / "repo", door.root, task_id="wake-3"), "an unrelated focus's text"))
    assert ul.read_record(door.root)["author"]["requested"] is False
    assert ul.project_letter_for_panel(status, drive_root=door.root)["request"]["saved"] is False
    _ok(_save(_registry(tmp_path / "repo", door.root, task_id=successor), "the continued description", expected=1))
    assert ul.read_record(door.root)["author"] == {"task_id": successor, "model": "", "requested": True}
    request = ul.project_letter_for_panel(status, drive_root=door.root)["request"]
    assert (request["task_id"], request["status"], request["saved"]) == (handle, "cancelled", True)


@pytest.mark.parametrize("terminal", ["completed", "cancelled"])
def test_an_admitted_continue_holds_write_and_its_end_releases_it(door, tmp_path, terminal):
    """Both recorded ends of Continue identify the current author; old physical history stays intact."""
    from ouroboros.owner_continue import binding_sha, claim_on_predecessor, mark_claim_admitted, successor_id
    from ouroboros.task_results import load_task_result, write_task_result

    _offer(door.root)
    handle = _request(door).json()["task_id"]
    door.pending.clear()
    write_task_result(door.root, handle, "cancelled", cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    successor = successor_id(handle, "continue-live")
    binding = {"verb": "continue", "binding_version": 1, "predecessor_task_id": handle,
               "action_nonce": "continue-live", "successor_task_id": successor, "cause": "snapshot_restore"}
    claim_on_predecessor(door.root, handle, "continue-live", binding)
    write_task_result(door.root, successor, "running", root_task_id=successor, delegation_role="root",
                      continuation_admission={"binding": binding, "binding_sha256": binding_sha(binding)})
    mark_claim_admitted(door.root, handle, "continue-live")
    status = {"running_sha": BASE, "checked_target_sha": TARGET}
    request = ul.project_letter_for_panel(status, drive_root=door.root)["request"]
    assert _request(door).status_code == 409 and not door.pending
    assert (request["task_id"], request["current_task_id"], request["status"]) == (handle, successor, "running")
    _ok(_save(_registry(tmp_path / "repo", door.root, task_id=successor), "continued prose"))
    _finish(door.root, successor, terminal)
    request = ul.project_letter_for_panel(status, drive_root=door.root)["request"]
    assert (request["status"], request["saved"]) == (terminal, True)
    assert load_task_result(door.root, handle, strict=True)["status"] == "cancelled"
    assert _request(door, mode="rewrite").status_code == 200


def test_a_retained_range_rewrite_is_told_whose_material_and_facts_are_stored(door):
    """A..B was described, then a check replaced the one stored material and check facts with B..C.
    Rewriting A..B names neither as its own and gives A..B's repository recipe; a request for B..C
    gets both, with the reader's own budget."""
    _offer(door.root, base=TARGET, target=NEWER, state="ready", text="about a..b", revision=1, key=_key(),
           author={"task_id": "t", "model": "m", "requested": False},
           qualification={"base_sha": TARGET, "target_sha": NEWER, "commits": []})
    handle = _request(door, mode="rewrite").json()["task_id"]
    text = door.pending[0]["text"]
    for part in (f"The stored update material covers {TARGET}..{NEWER}, not this range",
                 f"git log --format='%H %s%n%n%b' {BASE}..{TARGET}", f'base="{BASE}", head="{TARGET}"',
                 f"No check facts for this range are recorded (the recorded ones cover {TARGET}..{NEWER})."):
        assert part in text, part
    assert 'kind="update_material"' not in text and 'select="qualification"' not in text
    _finish(door.root, handle, "completed")
    _request(door, mode="write", base=TARGET, target=NEWER)
    newer = door.pending[1]["text"]
    assert 'kind="update_material", select="text", max_bytes=4321)' in newer and ul.CHECK_FACTS in newer
    assert "not this range" not in newer and "No check facts" not in newer


# --- the worker's save reaches an open window through the host --------------------------


def test_a_worker_save_reaches_an_open_window_through_the_supervisor(tmp_path, monkeypatch):
    """A worker has no server loop: the tool queues the existing frame, the supervisor's dispatch
    table answers it by broadcasting, the window's socket hears it and its status read shows the
    saved text as current."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    import supervisor.git_ops as git_ops
    from ouroboros.gateway import ws
    from ouroboros.gateway.router import collect_routes
    from supervisor.events import dispatch_event

    drive = tmp_path / "data"
    _offer(drive)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", drive)
    monkeypatch.setattr(ws, "_ws_clients", [])
    monkeypatch.setattr(ws, "_event_loop", None)
    monkeypatch.setattr(git_ops, "compute_managed_update_status", lambda fetch: {
        "managed": True, "available": True, "current_sha": BASE, "latest_sha": TARGET, "checked_target_sha": TARGET,
        "update_channel": "stable", "target_ref": "managed/main", "check_ok": None})
    monkeypatch.setattr("ouroboros.server_process.server_source_baseline", lambda: BASE)
    monkeypatch.setattr(git_ops, "git_capture", lambda *a, **k: (1, "", ""))

    @asynccontextmanager
    async def lifespan(_app):
        ws.set_event_loop(asyncio.get_running_loop())
        yield

    app = Starlette(routes=collect_routes(data_dir=drive), lifespan=lifespan)
    app.state.drive_root = drive
    events = queue_module.Queue()
    host = types.SimpleNamespace(bridge=types.SimpleNamespace(broadcast=ws.broadcast_ws_sync), DRIVE_ROOT=drive,
                                 append_jsonl=lambda *_a: None)
    with TestClient(app) as client, client.websocket_connect("/ws") as socket:
        assert client.get("/api/update/status").json()["letter"]["state"] == "none"
        receipt = _ok(_save(_registry(tmp_path / "repo", drive, event_queue=events), "seen by the window"))
        assert receipt["window_refresh"] == "queued_for_host"
        dispatch_event(events.get_nowait(), host)
        assert socket.receive_json() == {"type": "update_progress_changed"}
        letter = client.get("/api/update/status").json()["letter"]
    assert letter["text"] == "seen by the window" and letter["description_current"] is True and letter["revision"] == 1


# --- the material's advertised budget is the reader's own ------------------------------


def _range_git(commits):
    """Git answers for one range of ``(sha, subject, body)``, served to the real collector."""
    def capture(argv):
        if argv[:2] == ["git", "rev-list"]:
            return 0, "\n".join(sha for sha, _subject, _body in commits), ""
        if argv[:2] == ["git", "show"]:
            return 0, "7.9.0", ""
        if argv[:2] != ["git", "log"]:
            return 1, "", ""
        if argv[-1] != f"{BASE}..{TARGET}":  # README rows and local history: none
            return 0, "", ""
        if "-n" in argv:  # the bodies
            return 0, "".join(f"{sha}\x1f{body}\x1e" for sha, _subject, body in commits), ""
        return 0, "".join(f"{sha}\x1f2026-10-28T08:00:00+00:00\x1f{subject}\x1e" for sha, subject, _body in commits), ""
    return capture


def test_the_advertised_material_budget_is_exactly_what_the_reader_admits(tmp_path, monkeypatch):
    """A long range whose bodies carry newlines, quotes and backslashes: its canonical JSON exceeds its
    raw bytes by more than the former +4096 allowance. The check records the reader's own count; the
    Runtime pointer and the request advertise it; the real query_data returns the whole text at exactly
    that budget and refuses one byte less."""
    import re

    monkeypatch.setattr("ouroboros.update_qualification.post_merge_checks",
                        lambda commits, url: {"status": "unavailable", "reason": "offline", "runs": [], "missing_shas": []})
    commits = [(f"{n:040x}", f"Change {n}", 'A "quoted" step in C:\\path\nand its next line\n' * 30 + "ünïcode")
               for n in range(1, 161)]
    drive = tmp_path / "data"
    record = ul.refresh_after_check({"check_ok": True, "available": True, "running_sha": BASE, "current_sha": BASE,
                                     "latest_sha": TARGET, "update_channel": "stable", "target_ref": "managed/main"},
                                    drive_root=drive, git=_range_git(commits))
    document = json.loads(ul.material_path(drive).read_text(encoding="utf-8"))
    budget, raw = record["material"]["text_bytes"], len(document["text"].encode("utf-8"))
    assert document["text_bytes"] == budget and document["commits"] == 160 and budget > raw + 4096
    assert ul.project_letter(record, head_sha=BASE, latest_sha=TARGET)["material"]["text_bytes"] == budget
    assert "max_bytes=<letter.material.text_bytes>" in ul.DESCRIPTION_SOURCES
    request = ul.author_request_text(record, {"mode": "write", "key": record["checked_key"], "revision_at_request": 0})
    assert re.search(r'kind="update_material", select="text", max_bytes=(\d+)\)', request).group(1) == str(budget)
    reg = _registry(tmp_path / "repo", drive, task_id="main-turn")

    def read(max_bytes):
        return reg.execute_result("query_data", {"op": "json", "kind": "update_material", "select": "text",
                                                 "max_bytes": max_bytes})

    whole = read(budget)
    assert whole.status == "ok", whole.text[:300]
    assert json.loads(whole.text)["data"] == document["text"]
    for short in (budget - 1, raw, raw + 4096):  # the reader's floor, and both former recipes
        refused = read(short)
        assert refused.status == "error" and f"at least {budget}" in refused.text, short


# --- real Git: an ambiguous change that exists only in the official target --------------


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def test_main_can_read_an_ambiguous_target_only_patch_and_save_what_it_found(tmp_path, monkeypatch):
    """The commit message says only "tweak"; the patch says what changed. The check collects the
    range, query_data serves it with its read recipe, vcs_read on the system repository shows the
    patch of a commit the running checkout does not contain, and the description is saved for
    exactly that range."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "ouroboros")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "VERSION").write_text("7.8.1\n")
    (repo / "panel.txt").write_text("label: Refresh\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "official")
    (repo / "panel.txt").write_text("label: Check\nwrite: Write description\n")
    _git(repo, "commit", "-q", "-am", "tweak")
    target = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/managed/ouroboros", target)
    _git(repo, "checkout", "-q", "ouroboros")
    _git(repo, "branch", "-D", "official")
    assert _git(repo, "rev-parse", "HEAD") == base and _git(repo, "branch", "--contains", target) == ""

    def capture(cmd):
        proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

    drive = tmp_path / "data"
    monkeypatch.setattr(ul, "_default_git", lambda: capture)
    monkeypatch.setattr("ouroboros.update_qualification.post_merge_checks",
                        lambda commits, url: {"status": "unavailable", "reason": "offline", "runs": [], "missing_shas": []})
    status = {"check_ok": True, "available": True, "running_sha": base, "current_sha": base, "latest_sha": target,
              "update_channel": "development", "target_ref": "managed/ouroboros"}
    record = ul.refresh_after_check(status, drive_root=drive)
    assert record["checked_key"]["target_sha"] == target and record["state"] == "none"
    reg = _registry(repo, drive, system_repo_dir=repo, task_id="main-turn")
    material = reg.execute_result("query_data", {"op": "json", "kind": "update_material", "select": "text",
                                                 "format": "text"})
    assert material.status == "ok", material.text
    text = json.loads(material.text)["data"]
    assert f"- {target} " in text and "tweak" in text and 'vcs_read(root="system_repo", action="show"' in text
    patch = reg.execute_result("vcs_read", {"action": "show", "root": "system_repo", "ref": target})
    assert patch.status == "ok", patch.text
    assert "-label: Refresh" in patch.text and "+label: Check" in patch.text and "+write: Write description" in patch.text
    receipt = _ok(_save(reg, "The panel's Refresh is now **Check**, and a separate **Write description** button "
                             "asks me to describe the update.", base=base, target=target))
    assert receipt["key"]["target_sha"] == target and receipt["revision"] == 1
    view = ul.project_letter(ul.read_record(drive), head_sha=base, latest_sha=target)
    assert view["description_current"] is True and view["target_version"] == "7.8.1"
    # Once a later check stores another range's material, a rewrite of this retained range is sent to the
    # repository, and that recipe reads exactly this range through the real tool.
    retained = {**ul.read_record(drive), "material": {"key": _key(target, NEWER), "text_bytes": 1}}
    request = ul.author_request_text(retained, {"mode": "rewrite", "key": record["checked_key"], "revision_at_request": 1})
    assert f'vcs_read(root="system_repo", action="diff", base="{base}", head="{target}", stat=true)' in request
    diff = reg.execute_result("vcs_read", {"action": "diff", "root": "system_repo", "base": base, "head": target,
                                           "stat": True})
    assert diff.status == "ok" and "panel.txt" in diff.text, diff.text[:300]
