"""Parent GO 30 Oct 08:52: a Continue successor keeps its predecessor's recorded memory isolation on its OWN
drive, keeps the work's service teardown choice and holds the work's declared inputs in its own store.

The successor was admitted with no drive fields and ran on the canonical drive whatever its predecessor's producer
had prepared, although the API door records ``forked``/``empty`` and refuses shared memory for external workspaces.
Each root here is opened through the real API door and interrupted by a server shutdown before it starts; the real
``admit_continuation`` then admits its successor, and the drives are the ones ``headless.prepare_task_drive`` made.
The successor's drive is fresh from the current canonical root (never the predecessor's), prepared once, and freed
only when its admission is proven absent.
"""

from __future__ import annotations

import base64
import copy
import json
import pathlib
import re
import shutil
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_continue_execution_budget import _continue
from tests.test_owner_continue import NONCE
from tests.test_restart_retention import _pool_events
from tests.test_root_effort_ingress import _pool_ready


@pytest.fixture(autouse=True)
def _isolated_projects_root(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path_factory.mktemp("projects_root")))


@pytest.fixture
def pool(tmp_path, monkeypatch):
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda task: True)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **_k: True)
    (tmp_path / "memory").mkdir(exist_ok=True)
    (tmp_path / "memory" / "identity.md").write_text("IDENTITY v1\n", encoding="utf-8")
    return SimpleNamespace(root=tmp_path, queue=queue, state=state, workers=workers, mp=monkeypatch)


def _interrupted(pool, **body):
    """A root the API door admitted, then interrupted by a server shutdown before it started."""
    from tests.test_continue_api_opening import _door

    _pool_ready(pool.mp, pool.workers)
    response = _door(pool).post("/api/tasks", json={"description": "Fix the importer", **body})
    assert response.status_code == 200, response.text
    _shutdown(pool)
    return response.json()["task_id"]


def _shutdown(pool):
    _pool_events(pool.workers, pool.mp)
    pool.mp.setattr(pool.workers, "WORKERS", {})
    assert pool.workers.kill_workers(terminal_status="cancelled", result_reason="server shutdown",
                                     stop_source="server_shutdown")


def _drive(pool, task_id):
    from ouroboros.headless import task_state_dir

    return task_state_dir(pool.root, task_id) / "data"


def _state(drive):
    return json.loads((drive / "state" / "state.json").read_text(encoding="utf-8"))


def _advertised(text):
    """The predecessor's folders a work order names as readable."""
    found = re.search(r"reaches its folders (.+?), until a prune removes them", text)
    return found[1].split(", ") if found else []


def test_a_forked_root_continues_on_its_own_fresh_fork_never_the_predecessors(pool):
    from ouroboros.task_results import load_task_result

    task_id = _interrupted(pool, memory_mode="forked")
    pred = _drive(pool, task_id)
    assert _state(pred)["memory_mode"] == "forked"
    (pred / "memory" / "identity.md").write_text("FORK-LOCAL EDIT\n", encoding="utf-8")
    (pool.root / "memory" / "identity.md").write_text("IDENTITY v2\n", encoding="utf-8")

    row, binding = _continue(pool, task_id)

    own = _drive(pool, row["id"])
    assert binding["memory_mode"] == "forked"
    assert (_state(own)["headless_task_id"], _state(own)["memory_mode"]) == (row["id"], "forked")
    assert (own / "memory" / "identity.md").read_text(encoding="utf-8") == "IDENTITY v2\n", "fresh from canonical"
    assert (pred / "memory" / "identity.md").read_text(encoding="utf-8") == "FORK-LOCAL EDIT\n", "never touched"
    expected = {"memory_mode": "forked", "drive_root": str(own), "child_drive_root": str(own),
                "budget_drive_root": str(pool.root)}
    assert {key: row.get(key) for key in expected} == expected
    stored = load_task_result(pool.root, row["id"])
    assert {key: stored.get(key) for key in expected} == expected, "a successor that never starts hands them on"
    assert row["metadata"]["child_drive_root"] == str(own) and row["metadata"]["budget_drive_root"] == str(pool.root)
    # Interrupted before it started: its end exported no memory, so none is promised; its own drive is still there.
    assert "own forked memory drive" in row["text"] and "memory_export.json" not in row["text"]
    assert "It recorded no memory export." in row["text"] and "never a drive's memory folder" in row["text"]
    assert all(pathlib.Path(folder).is_dir() for folder in _advertised(row["text"]))


def test_an_empty_root_stays_empty_and_a_shared_one_keeps_todays_binding(pool):
    empty = _interrupted(pool, memory_mode="empty")
    row, binding = _continue(pool, empty)
    assert binding["memory_mode"] == "empty" and _state(_drive(pool, row["id"]))["memory_mode"] == "empty"
    assert not (_drive(pool, row["id"]) / "memory" / "identity.md").exists(), "no canonical memory is seeded"

    shared = _interrupted(pool)
    (pool.root / "task_drives" / shared).mkdir(parents=True)  # an ordinary scratch folder proves no drive
    row, binding = _continue(pool, shared)
    assert "memory_mode" not in binding and not row.get("drive_root") and not row.get("child_drive_root")
    assert not _drive(pool, row["id"]).exists()
    assert "memory drive" not in row["text"]


def test_the_same_nonce_never_prepares_again(pool):
    from supervisor.continuation_admission import admit_continuation

    task_id = _interrupted(pool, memory_mode="forked")
    row, _binding = _continue(pool, task_id)
    own = _drive(pool, row["id"])
    state = (own / "state" / "state.json").read_bytes()
    (own / "memory" / "note.md").write_text("the successor's own work\n", encoding="utf-8")
    (pool.root / "memory" / "identity.md").write_text("IDENTITY v3\n", encoding="utf-8")

    replay = admit_continuation(task_id, action_nonce=NONCE)

    assert replay["ok"] is True and replay["replay"] is True
    assert (own / "state" / "state.json").read_bytes() == state
    assert (own / "memory" / "note.md").exists()
    assert (own / "memory" / "identity.md").read_text(encoding="utf-8") == "IDENTITY v1\n", "never reseeded"


def _fail(pool, failure):
    from ouroboros import task_results

    if failure == "blocked":
        pool.mp.setattr(pool.queue, "enqueue_task", lambda task, **_k: {**task, "_admission_blocked": "test_refusal"})
    elif failure in {"rollback_confirmed", "rollback_unknown"}:
        failing = {"owner_continue_admitted"} | ({"owner_continue_rollback"} if failure == "rollback_unknown" else set())
        pool.mp.setattr(pool.queue, "persist_queue_snapshot", lambda reason="", **_k: reason not in failing)
    else:
        real = task_results.write_task_result

        def flaky(root, task_id, status, **fields):
            if status == "scheduled" and fields.get("continuation_admission") is not None:
                raise OSError("disk")
            return real(root, task_id, status, **fields)

        pool.mp.setattr(task_results, "write_task_result", flaky)


@pytest.mark.parametrize("failure, freed", [
    ("blocked", True),              # the queue refused the successor: proven unadmitted
    ("rollback_confirmed", True),   # its admission snapshot failed and the rollback snapshot recorded the removal
    ("rollback_unknown", False),    # neither snapshot confirmed: the queued row may be durable
    ("publication_unknown", False),  # the queue row and its snapshot are durable; only the result write failed
])
def test_the_successor_drive_is_freed_only_when_its_admission_is_proven_absent(pool, failure, freed):
    from supervisor.continuation_admission import admit_continuation

    task_id = _interrupted(pool, memory_mode="forked")
    _fail(pool, failure)

    ack = admit_continuation(task_id, action_nonce=NONCE)

    assert ack["ok"] is False, ack
    own = _drive(pool, ack["successor_task_id"])
    assert own.exists() is (not freed)
    if failure == "rollback_unknown":
        # The claim stays and the same nonce retries: the prepared drive is reused, never reseeded.
        state = (own / "state" / "state.json").read_bytes()
        (pool.root / "memory" / "identity.md").write_text("IDENTITY v4\n", encoding="utf-8")
        pool.mp.setattr(pool.queue, "persist_queue_snapshot", lambda **_k: True)
        retried = admit_continuation(task_id, action_nonce=NONCE)
        assert retried["ok"] is True and retried["successor_task_id"] == ack["successor_task_id"], retried
        assert (own / "state" / "state.json").read_bytes() == state
        assert (own / "memory" / "identity.md").read_text(encoding="utf-8") == "IDENTITY v1\n"


def test_the_drive_mode_comes_only_from_a_producer_fact(pool):
    from ouroboros.headless import prepare_task_drive
    from supervisor.continuation_admission import _successor_drive_mode

    q = pool.queue

    def drive(task_id):
        return str(_drive(pool, task_id))

    (pool.root / "task_drives" / "plain-1").mkdir(parents=True)
    assert _successor_drive_mode(q, "plain-1", {"child_drive_root": str(pool.root / "task_drives" / "plain-1"),
                                                "memory_mode": "forked"}) == "", "scratch is no drive"
    assert _successor_drive_mode(q, "plain-1", {"drive_root": str(pool.root), "memory_mode": "shared"}) == ""
    assert _successor_drive_mode(q, "rec-1", {"child_drive_root": drive("rec-1"), "memory_mode": "empty"}) == "empty"
    assert _successor_drive_mode(q, "rec-2", {"child_drive_root": drive("rec-2"), "memory_mode": "shared"}) == "forked"
    prepare_task_drive(pool.root, "state-1", "empty")
    assert _successor_drive_mode(q, "state-1", {"child_drive_root": drive("state-1")}) == "empty"
    assert _successor_drive_mode(q, "state-1", {}) == "empty", "the producer's own state is evidence"
    for task_id, row in (("gone-1", {"child_drive_root": drive("gone-1")}),  # missing mode, no state: never invented
                         ("odd-1", {"child_drive_root": drive("odd-1"), "memory_mode": "isolated"}),  # unsupported
                         ("state-1", {"child_drive_root": drive("state-1"), "memory_mode": "forked"})):  # contradicted
        with pytest.raises(ValueError, match="continuation_drive_mode_unknown"):
            _successor_drive_mode(q, task_id, row)


def test_an_unknown_mode_refuses_before_any_claim(pool):
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.continuation_admission import admit_continuation

    task_id = _interrupted(pool, memory_mode="forked")
    write_task_result(pool.root, task_id, "cancelled", memory_mode="isolated")

    ack = admit_continuation(task_id, action_nonce=NONCE)

    assert ack == {"ok": False, "error": "continuation_drive_mode_unknown"}
    assert "continued_by" not in load_task_result(pool.root, task_id) and pool.workers.PENDING == []


def test_a_never_started_successor_hands_its_mode_to_its_own_successor(pool):
    task_id = _interrupted(pool, memory_mode="empty")
    first, _binding = _continue(pool, task_id)
    _shutdown(pool)

    second, binding = _continue(pool, first["id"])

    assert binding["memory_mode"] == "empty"
    assert _state(_drive(pool, second["id"]))["headless_task_id"] == second["id"]
    assert _drive(pool, first["id"]).exists(), "the first successor's drive stays its own until its own prune"


def test_the_teardown_choice_rides_with_the_work(pool):
    from ouroboros.tools.services import task_service_teardown

    task_id = _interrupted(pool, service_teardown="keep")
    row, _binding = _continue(pool, task_id)

    assert row["metadata"]["service_teardown"] == "keep"
    assert task_service_teardown(SimpleNamespace(task_metadata=row["metadata"])) == "keep"


def _tools(pool, monkeypatch, tmp_path_factory):
    """A successor's own read_file and view_image as its worker builds them: its own drive, canonical as the budget
    root of a split one; ``view`` returns the tool's answer and the conversation it attached to."""
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from tests.test_predecessor_lineage_read import predecessor_envelope

    home, repo = tmp_path_factory.mktemp("home"), tmp_path_factory.mktemp("repo")
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    for key, value in (("OUROBOROS_USER_FILES_ROOT", str(home)), ("OUROBOROS_RUNTIME_MODE", "advanced"),
                       ("OUROBOROS_SAFETY_MODE", "off"), ("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")):
        monkeypatch.setenv(key, value)

    def call(row, tool, args, predecessor=""):
        split = {"memory_mode": row["memory_mode"], "budget_drive_root": str(pool.root)} if row.get("child_drive_root") else {}
        ctx = ToolContext(repo_dir=repo, drive_root=pathlib.Path(row.get("drive_root") or pool.root), task_id=row["id"],
                          **split)
        ctx.task_metadata = {"root_task_id": row["id"], **split}
        ctx.task_contract = {"predecessor_authority": predecessor_envelope(predecessor)} if predecessor else {}
        ctx.messages = []
        registry = ToolRegistry(repo_dir=repo, drive_root=ctx.drive_root)
        registry.set_context(ctx)
        return registry.execute(tool, args), ctx.messages
    return SimpleNamespace(
        read=lambda row, path, predecessor="", root="artifact_store": call(row, "read_file", {"root": root, "path": path},
                                                                           predecessor)[0],
        view=lambda row, path: call(row, "view_image", {"path": path}))


def _declared(pool, task_id):
    """The predecessor's own declared input manifest, as its record carries it."""
    from ouroboros.task_results import load_task_result

    stored = load_task_result(pool.root, task_id)
    return (stored.get("task_contract") or stored["metadata"]["task_contract"])["attachment_manifest"]


def _store(pool, row):
    from ouroboros.artifacts import task_artifact_dir_path

    return task_artifact_dir_path(row.get("drive_root") or pool.root, row["id"])


def _holds(pool, row, declared):
    """ROW's own store and row declare every input with its declared identity: staged ones byte-verified there."""
    from ouroboros.artifacts import stream_artifact_file
    from ouroboros.task_results import load_task_result

    own = row["attachments"]
    assert [(r["ordinal"], r["label"], r["status"], r.get("sha256")) for r in own] == [
        (r["ordinal"], r["label"], r["status"], r.get("sha256")) for r in declared]
    for mine in own:
        if mine["status"] == "staged":
            assert stream_artifact_file(_store(pool, row) / mine["relpath"]) == {"size": mine["size"], "sha256": mine["sha256"]}
            assert f"read_file(root='artifact_store', path='{mine['relpath']}')" in row["text"]
    assert [r["relpath"] for r in row["attachment_images"]] == [r["relpath"] for r in own if r.get("is_image")]
    assert row["task_contract"]["attachment_manifest"] == own
    stored = load_task_result(pool.root, row["id"])
    assert stored["metadata"]["task_contract"]["attachment_manifest"] == own, "a successor that never starts hands them on"


@pytest.mark.parametrize("memory_mode", ["shared", "forked"])
def test_declared_inputs_ride_every_hop_in_each_successors_own_store(pool, tmp_path_factory, monkeypatch, memory_mode):
    """Owner A (the saved request and its conditions): each successor holds the work's inputs in its OWN store,
    copied one hop back from its direct predecessor under their recorded identity, so a second Continue reads them as
    the first did after the original task's copies are gone. A rejected declaration rides as the same rejection."""
    from ouroboros.artifacts import task_artifact_dir_path
    from tests.test_continue_quiz_origin import _png

    inputs = tmp_path_factory.mktemp("inputs")
    (inputs / "crash.log").write_text("ATTACHED BYTES\n", encoding="utf-8")
    (inputs / "shot.png").write_bytes(_png())
    task_id = _interrupted(pool, memory_mode=memory_mode, attachments=[
        {"path": str(inputs / "crash.log")}, {"path": str(inputs / "shot.png")}, {"path": str(inputs / "absent.txt")}])
    declared = _declared(pool, task_id)
    assert [(row["status"], row["reason"]) for row in declared] == [
        ("staged", ""), ("staged", ""), ("rejected", "source_missing")]
    tools = _tools(pool, monkeypatch, tmp_path_factory)

    first, _binding = _continue(pool, task_id)

    _holds(pool, first, declared)
    assert [row["is_image"] for row in first["attachments"][:2]] == [False, True]
    assert "ATTACHED BYTES" in tools.read(first, "attachments/crash.log")
    assert f"copied from task {task_id}" in first["text"] and "rejected (reason=source_missing, ordinal=2)" in first["text"]
    for drive in (pool.root, _drive(pool, task_id)):  # the original task's own copies go; the next hop never reads them
        shutil.rmtree(task_artifact_dir_path(drive, task_id) / "attachments", ignore_errors=True)
    _shutdown(pool)

    second, _binding = _continue(pool, first["id"])

    _holds(pool, second, declared)
    assert "ATTACHED BYTES" in tools.read(second, "attachments/crash.log")
    assert f"copied from task {first['id']}" in second["text"]
    assert second["metadata"]["continuation"]["owner_sources"]["api"]["task_id"] == task_id, "the request's own origin"
    _the_inputs_reach_their_consumers(pool, tools, second, declared, tmp_path_factory, monkeypatch)


def _first_request_images(pool, row, tmp_path_factory, monkeypatch):
    """The native image blocks of the real first request of ROW's own agent, on the roots its worker gives it
    (``worker_process``: the row's ``drive_root``, else the pool's canonical root)."""
    from ouroboros.agent import Env
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.context import build_llm_messages
    from ouroboros.memory import Memory
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_doc_context import _make_env_and_memory

    template, _memory = _make_env_and_memory(tmp_path_factory.mktemp("first_request"))
    drive = pathlib.Path(row.get("drive_root") or pool.root)
    env = Env(repo_dir=template.repo_dir, drive_root=drive, budget_drive_root=pool.root)
    for key, value in {"OUROBOROS_CONTEXT_MODE": "max", "OUROBOROS_TASK_REVIEW_MODE": "off",
                       "OUROBOROS_SAFETY_MODE": "off", "MCP_ENABLED": "false"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("ouroboros.context.resolve_context_fit_route", lambda *_a, **_kw: (
        {"model": "openai/test-model", "provider": "openai"},
        SimpleNamespace(route_fp="fixture", status="asserted", stale=False, window_tokens=1_000_000)))
    task = copy.deepcopy(row)
    assert validate_task_authority_sources(env, task) == {}
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
    registry._ctx.task_metadata = task["metadata"]
    messages, _cap = build_llm_messages(env=env, memory=Memory(drive_root=drive, repo_dir=env.repo_dir), task=task,
                                        ctx=registry._ctx)
    return [part for message in messages if message["role"] == "user" and isinstance(message["content"], list)
            for part in message["content"] if part.get("type") == "image_url"]


def _the_inputs_reach_their_consumers(pool, tools, row, declared, tmp_path_factory, monkeypatch):
    """The image's readers and a delegated child: the real first request's native image block, ``view_image`` on the
    successor's own copy, and the copy a child it delegates gets."""
    from ouroboros.tools.control_scheduling import _materialize_child_attachment_manifest

    image = next(entry for entry in row["attachments"] if entry.get("is_image"))
    own_image = _store(pool, row) / image["relpath"]
    [native] = _first_request_images(pool, row, tmp_path_factory, monkeypatch)
    assert pathlib.Path(native["_source_path"]) == own_image.resolve()
    assert base64.b64decode(native["image_url"]["url"].split(",", 1)[1]) == own_image.read_bytes()
    answer, messages = tools.view(row, str(own_image))
    assert "now attached as a local image block" in answer, answer
    parts = [part for message in messages if isinstance(message.get("content"), list) for part in message["content"]]
    assert [part["_source_sha256"] for part in parts if part.get("type") == "image_url"] == [image["sha256"]]
    child = tmp_path_factory.mktemp("child_drive")
    projection, error = _materialize_child_attachment_manifest(
        row["task_contract"], child, "child-of-successor", owner_drive=pathlib.Path(row.get("drive_root") or pool.root),
        owner_task_id=row["id"], custody_drive=pool.root)
    assert error == "", error
    copied = [entry for entry in projection["attachment_manifest"] if entry["status"] == "staged"]
    assert [(entry["label"], entry["sha256"]) for entry in copied] == [
        (entry["label"], entry["sha256"]) for entry in declared if entry["status"] == "staged"]
    assert all(pathlib.Path(entry["abs_path"]).resolve().is_relative_to(child.resolve()) for entry in copied)


def test_a_canonical_successor_gets_its_image_natively_with_no_canonical_memory_export(pool, tmp_path_factory, monkeypatch):
    """The same saved request's image reaches the first request of a successor on the canonical drive, as it reached
    its API root's. The row records no ``drive_root`` (assignment and terminal files read that field as a fork), so its
    end exports no canonical memory: the first request reads the root its agent actually runs on."""
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.headless import prepare_terminal_task_files
    from ouroboros.task_results import load_task_result, write_task_result
    from tests.test_continue_quiz_origin import _png

    source = tmp_path_factory.mktemp("inputs") / "shot.png"
    source.write_bytes(_png())
    task_id = _interrupted(pool, attachments=[{"path": str(source)}])

    row, binding = _continue(pool, task_id)

    assert "memory_mode" not in binding and not row.get("drive_root") and not row.get("child_drive_root")
    [image] = row["attachment_images"]
    [native] = _first_request_images(pool, row, tmp_path_factory, monkeypatch)
    assert pathlib.Path(native["_source_path"]) == (_store(pool, row) / image["relpath"]).resolve()
    assert base64.b64decode(native["image_url"]["url"].split(",", 1)[1]) == source.read_bytes()

    write_task_result(pool.root, row["id"], "failed", reason_code="provider_unavailable", result="The provider failed.")
    assert prepare_terminal_task_files(pool.root, row)["error"] == ""
    assert not (task_artifact_dir_path(pool.root, row["id"]) / "memory_export.json").exists()
    assert "memory_export" not in {item.get("kind") for item in load_task_result(pool.root, row["id"]).get("artifacts") or []}


def test_lost_input_bytes_stay_a_named_gap_on_every_hop(pool, tmp_path_factory):
    """Bytes the predecessor no longer holds (gone, or other bytes under the name) are never copied or replaced: the
    successor's own row keeps the declaration as a rejected source_missing row with its recorded identity, the work
    order names it, and the next hop carries that gap by value."""
    from ouroboros.artifacts import task_artifact_dir_path

    inputs = tmp_path_factory.mktemp("inputs")
    names = ("gone.txt", "changed.txt", "kept.txt")
    for name in names:
        (inputs / name).write_text(f"{name} as declared\n", encoding="utf-8")
    task_id = _interrupted(pool, memory_mode="forked", attachments=[{"path": str(inputs / name)} for name in names])
    declared = _declared(pool, task_id)
    store = task_artifact_dir_path(_drive(pool, task_id), task_id)
    (store / declared[0]["relpath"]).unlink()
    (store / declared[1]["relpath"]).write_text("other bytes\n", encoding="utf-8")
    gaps = [{"ordinal": row["ordinal"], "label": row["label"], "mime": row["mime"], "size": row["size"],
             "sha256": row["sha256"], "status": "rejected", "reason": "source_missing"} for row in declared[:2]]

    first, _binding = _continue(pool, task_id)

    assert first["attachments"][:2] == gaps and first["attachments"][2]["status"] == "staged"
    assert sorted(path.name for path in (_store(pool, first) / "attachments").iterdir()) == ["kept.txt"]
    for row in declared[:2]:
        assert f"{row['label']} ({row['size']} bytes, sha256 {row['sha256']})" in first["text"]
    _shutdown(pool)

    second, _binding = _continue(pool, first["id"])

    assert second["attachments"][:2] == gaps and second["attachments"][2]["status"] == "staged"
    assert f"{declared[0]['label']} ({declared[0]['size']} bytes, sha256 {declared[0]['sha256']})" in second["text"]


def test_a_long_input_manifest_rides_whole_and_an_unreadable_one_refuses_before_the_claim(pool, tmp_path_factory):
    """Past the inline preview the declared set lives behind ``attachment_manifest_ref``: the successor copies all of
    it and keeps its own reference. A manifest no store reads is an unknown input set: refused before the claim,
    with nothing written, and the same press is admitted once it reads again."""
    from ouroboros.artifacts import resolve_attachment_manifest, task_artifact_dir_path
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    inputs = tmp_path_factory.mktemp("inputs")
    for index in range(26):
        (inputs / f"part{index:02d}.txt").write_text(f"part {index}\n", encoding="utf-8")
    task_id = _interrupted(pool, memory_mode="forked",
                           attachments=[{"path": str(inputs / f"part{index:02d}.txt")} for index in range(26)])
    stored = load_task_result(pool.root, task_id)
    ref = (stored.get("task_contract") or stored["metadata"]["task_contract"])["attachment_manifest_ref"]
    source = task_artifact_dir_path(_drive(pool, task_id), task_id) / ref["path"]
    whole = source.read_bytes()
    source.write_text("{torn", encoding="utf-8")

    ack = admit_continuation(task_id, action_nonce=NONCE)

    assert ack == {"ok": False, "error": "continuation_inputs_unavailable"}
    assert "continued_by" not in load_task_result(pool.root, task_id) and pool.workers.PENDING == []
    source.write_bytes(whole)
    row, _binding = _continue(pool, task_id)
    assert row["attachment_manifest_ref"]["count"] == 26 and len(row["attachment_manifest"]) == 25
    own = resolve_attachment_manifest(row["drive_root"], row["id"], row["task_contract"])
    assert [entry["label"] for entry in own] == [f"part{index:02d}.txt" for index in range(26)]
    assert [pathlib.Path(entry["abs_path"]).read_text(encoding="utf-8") for entry in own] == [
        f"part {index}\n" for index in range(26)]
    assert pathlib.Path(own[25]["abs_path"]).resolve().is_relative_to(_store(pool, row).resolve())
    assert "Complete input manifest (26 declarations)" in row["text"]


def test_a_failed_input_copy_admits_nothing_and_the_same_press_recovers(pool, tmp_path_factory):
    """A copy that fails midway (the disk filled) leaves no file, no queue row and no drive behind; the claim stays,
    so the same nonce is admitted once the copy can land."""
    from ouroboros import artifacts
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    source = tmp_path_factory.mktemp("inputs") / "crash.log"
    source.write_text("ATTACHED BYTES\n", encoding="utf-8")
    task_id = _interrupted(pool, memory_mode="forked", attachments=[{"path": str(source)}])
    real = artifacts.copy_artifact_file

    def disk_full(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    pool.mp.setattr(artifacts, "copy_artifact_file", disk_full)
    ack = admit_continuation(task_id, action_nonce=NONCE)

    assert (ack["ok"], ack["error"]) == (False, "continuation_inputs_unavailable"), ack
    successor = ack["successor_task_id"]
    assert load_task_result(pool.root, task_id)["continued_by"]["successor_task_id"] == successor
    assert pool.workers.PENDING == [] and successor not in pool.queue.ADMISSION_RESERVATIONS
    assert not _drive(pool, successor).exists(), "nothing was admitted: its own drive is freed with the copy"
    pool.mp.setattr(artifacts, "copy_artifact_file", real)
    row, _binding = _continue(pool, task_id)
    assert row["id"] == successor and row["attachments"][0]["status"] == "staged"
    assert (_store(pool, row) / row["attachments"][0]["relpath"]).read_text(encoding="utf-8") == "ATTACHED BYTES\n"


def test_a_partial_seed_is_never_reused_as_a_prepared_drive(pool):
    """A preparation that failed mid-seed and could not be freed (its absence unprovable) leaves no producer record,
    so the same nonce prepares the drive again instead of adopting an incomplete one as prepared."""
    from ouroboros import headless
    from supervisor.continuation_admission import admit_continuation

    knowledge = pool.root / "memory" / "knowledge"
    knowledge.mkdir()
    (knowledge / "topic.md").write_text("global note\n", encoding="utf-8")
    task_id = _interrupted(pool, memory_mode="forked")
    seed, liveness = headless._copy_stable_memory, pool.queue.task_settlement_liveness

    def seed_then_fail(parent, child, *, project_id=""):
        (child / "memory").mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / "memory" / "identity.md", child / "memory" / "identity.md")
        raise OSError(28, "No space left on device")

    pool.mp.setattr(headless, "_copy_stable_memory", seed_then_fail)
    pool.mp.setattr(pool.queue, "task_settlement_liveness", lambda _task_id: None)  # absence unprovable: retained
    ack = admit_continuation(task_id, action_nonce=NONCE)

    assert (ack["ok"], ack["error"]) == (False, "continuation_drive_unavailable"), ack
    partial = _drive(pool, ack["successor_task_id"])
    assert (partial / "memory" / "identity.md").exists(), "the incomplete drive stays: its absence is unprovable"
    recorded_after_failure = (partial / "state" / "state.json").exists()
    pool.mp.setattr(headless, "_copy_stable_memory", seed)
    pool.mp.setattr(pool.queue, "task_settlement_liveness", liveness)

    row, _binding = _continue(pool, task_id)

    assert row["id"] == ack["successor_task_id"]
    assert (partial / "memory" / "knowledge" / "topic.md").read_text(encoding="utf-8") == "global note\n", "whole seed"
    assert (_state(partial)["headless_task_id"], _state(partial)["memory_mode"]) == (row["id"], "forked")
    assert recorded_after_failure is False, "the producer's record names only a complete drive"


@pytest.mark.parametrize("restart", ["fresh", "stale", "stale_without_drive"])
def test_a_project_shared_root_continues_on_its_own_project_fork_through_restore(pool, restart):
    """The API door runs a project-scoped "shared" root on a forked drive; its Continue prepares the successor's own
    Project fork (no global knowledge). A real snapshot restore before the successor starts keeps its drive fields. The
    snapshot's age is its publisher's stamp (``state/queue.alive``, ``queue_stamp``): past the restore window the row
    waits under the Project recheck that assignment runs, which admits it to the owner's Resume and a worker only while
    its own prepared drive exists."""
    from ouroboros import projects_registry as registry
    from ouroboros.task_results import load_task_result
    from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact
    from supervisor.queue_snapshot import persist_queue_snapshot, restore_pending_from_snapshot
    from tests.test_project_hold_recovery import worker

    (pool.root / "memory" / "knowledge").mkdir()
    (pool.root / "memory" / "knowledge" / "topic.md").write_text("global note\n", encoding="utf-8")
    registry.create_project(pool.root, "target", name="Target")
    task_id = _interrupted(pool, project_id="target", memory_mode="shared")
    assert load_task_result(pool.root, task_id)["memory_mode"] == "shared"
    assert _state(_drive(pool, task_id))["memory_mode"] == "forked"

    row, binding = _continue(pool, task_id)

    own = _drive(pool, row["id"])
    assert (binding["memory_mode"], row["project_id"], _state(own)["memory_mode"]) == ("forked", "target", "forked")
    assert (own / "memory" / "identity.md").exists() and not (own / "memory" / "knowledge" / "topic.md").exists()
    drive_fields = {key: row[key] for key in ("memory_mode", "drive_root", "child_drive_root", "budget_drive_root")}
    pool.mp.setattr(pool.queue, "persist_queue_snapshot", persist_queue_snapshot)
    assert persist_queue_snapshot(reason="test_before_restart") is True
    if restart != "fresh":
        alive = pathlib.Path(pool.queue.QUEUE_SNAPSHOT_PATH).with_name("queue.alive")
        stamp = json.loads(alive.read_text(encoding="utf-8"))
        alive.write_text(json.dumps({**stamp, "ts": "2000-01-01T00:00:00Z"}), encoding="utf-8")
    if restart == "stale_without_drive":
        shutil.rmtree(own)
    pool.workers.PENDING.clear()

    assert restore_pending_from_snapshot() == 1

    [restored] = pool.workers.PENDING
    assert {key: restored.get(key) for key in drive_fields} == drive_fields
    assert bool(restored.get("_project_admission_restore_hold")) is (restart != "fresh")
    assert budget_hold_fact(restored)["reason"] == HOLD_SAVED_WORK
    sent = worker(pool, pool.mp)
    pool.workers.assign_tasks()
    assert sent == []
    if restart == "stale_without_drive":
        assert restored["_project_admission_restore_hold"]["detail"] == "The original prepared task drive is unavailable."
    else:
        assert not restored.get("_project_admission_restore_hold")

    answer = pool.queue.resume_budget_paused_task(row["id"])
    pool.workers.assign_tasks()
    pool.workers.assign_tasks()

    if restart == "stale_without_drive":
        assert sent == [] and not own.exists(), "never dispatched, and no empty drive made in its place"
        assert restored["_project_admission_restore_hold"]["detail"] == "The original prepared task drive is unavailable."
    else:
        assert answer["ok"], answer
        assert [task["id"] for task in sent] == [row["id"]]
        assert {key: sent[0].get(key) for key in drive_fields} == drive_fields
        assert _state(own)["headless_task_id"] == row["id"]


def _provider_failed(pool, task_id, sent):
    """The worker's own failed terminal on the drive it ran on (``sent``: the row its assignment sent), its terminal
    files and the supervisor's ``task_done``."""
    from ouroboros.headless import prepare_terminal_task_files
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.events import _handle_task_done
    from tests.test_task_done_durable_vs_transport import _mk_ctx

    write_task_result(pathlib.Path(sent.get("drive_root") or pool.root), task_id, "failed",
                      reason_code="provider_unavailable", result="The provider was unavailable.")
    prepared = prepare_terminal_task_files(pool.root, sent)
    assert not prepared["error"], prepared
    ctx = _mk_ctx(pool.root, [], [])
    ctx.RUNNING, ctx.WORKERS, ctx.PENDING = pool.workers.RUNNING, pool.workers.WORKERS, pool.workers.PENDING
    _handle_task_done({"type": "task_done", "task_id": task_id, "worker_id": 7, "task_type": "task",
                       "status": "failed", "_files_prepared_attempt": 1}, ctx)
    assert load_task_result(pool.root, task_id)["status"] == "failed" and task_id not in pool.workers.RUNNING


def test_a_continued_fork_reads_its_predecessors_memory_export_through_its_lineage(pool, tmp_path_factory, monkeypatch):
    """The work order's promise end to end: a forked root that ran and failed leaves ``memory_export.json`` from its
    finalized end; its successor starts on a fresh fork, never that one, and its own read_file reaches the export
    through the one-hop lineage read."""
    from ouroboros.artifacts import task_artifact_dir_path
    from tests.test_continue_api_opening import _door
    from tests.test_continue_execution_budget import _start

    _pool_ready(pool.mp, pool.workers)
    response = _door(pool).post("/api/tasks", json={"description": "Fix the importer", "memory_mode": "forked"})
    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]
    pool.mp.setattr(pool.workers, "WORKERS", {})  # the door's readiness stubs are no worker slots
    sent, _meta = _start(pool, task_id, executed=5)
    (_drive(pool, task_id) / "memory" / "identity.md").write_text("FORK-LOCAL EDIT\n", encoding="utf-8")
    task_file = _drive(pool, task_id) / "task_drives" / task_id / "notes.md"
    task_file.parent.mkdir(parents=True)
    task_file.write_text("A TASK FILE\n", encoding="utf-8")
    _provider_failed(pool, task_id, sent)
    export = task_artifact_dir_path(pool.root, task_id) / "memory_export.json"
    assert json.loads(export.read_text(encoding="utf-8"))["files"]["identity.md"] == "FORK-LOCAL EDIT\n"

    row, _binding = _continue(pool, task_id)

    assert (_drive(pool, row["id"]) / "memory" / "identity.md").read_text(encoding="utf-8") == "IDENTITY v1\n"
    assert f"Its memory export is readable at {export}." in row["text"]
    assert {str(task_file.parent), str(export.parent)} <= set(_advertised(row["text"]))
    assert all(pathlib.Path(folder).is_dir() for folder in _advertised(row["text"]))
    tools = _tools(pool, monkeypatch, tmp_path_factory)
    assert "FORK-LOCAL EDIT" in tools.read(row, str(export), predecessor=task_id)
    # What the sentence names is what the one-hop lineage read gives: the task drive, never the memory folder.
    assert "A TASK FILE" in tools.read(row, str(task_file), predecessor=task_id, root="task_drive")
    for root in ("task_drive", "artifact_store"):
        memory = tools.read(row, str(_drive(pool, task_id) / "memory" / "identity.md"), predecessor=task_id, root=root)
        assert "FORK-LOCAL EDIT" not in memory and memory.startswith("⚠️"), memory


@pytest.mark.parametrize("left", ["scratch-only", "nothing"])
def test_a_removed_drive_is_never_advertised_and_a_scratch_folder_is_named_as_itself(pool, tmp_path_factory, monkeypatch,
                                                                                      left):
    """R1 correction: ``own_child_drives`` also lists a plain ``task_drives/<id>`` scratch folder, and the sentence read
    it as a fork: ``task_drives/<id>/task_drives/<id>`` and an ``artifacts/<id>`` that exist nowhere. With the fork
    removed by a drive prune, the work order names only folders that exist, each by its real path; the real read_file
    reaches what it names and the memory gap stays stated."""
    task_id = _interrupted(pool, memory_mode="forked")
    fork = _drive(pool, task_id)
    shutil.rmtree(fork)
    scratch = pool.root / "task_drives" / task_id
    if left == "scratch-only":
        scratch.mkdir(parents=True)
        (scratch / "notes.md").write_text("A SCRATCH NOTE\n", encoding="utf-8")

    row, binding = _continue(pool, task_id)

    assert binding["memory_mode"] == "forked", "the recorded mode: never guessed from what is left"
    folders = _advertised(row["text"])
    assert all(pathlib.Path(folder).is_dir() for folder in folders), folders
    assert str(fork) not in row["text"] and str(scratch / "task_drives") not in row["text"]
    assert "It recorded no memory export." in row["text"] and "never a drive's memory folder" in row["text"]
    if left == "scratch-only":
        assert str(scratch) in folders
        tools = _tools(pool, monkeypatch, tmp_path_factory)
        assert "A SCRATCH NOTE" in tools.read(row, str(scratch / "notes.md"), predecessor=task_id, root="task_drive")
    else:
        assert folders or "reaches no folder of it now" in row["text"]


def _owner_turn(pool, task_id, words, files, msg_id):
    """One owner follow-up as the server routes it to a live task (``server_owner_routing``): the files staged into the
    task's own store, the row written with their manifest. Returns the row's exact text and the staged manifest."""
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.server_owner_routing import _stage_mailbox_attachments
    from ouroboros.task_results import load_task_result
    from supervisor.queue_timeouts import _task_drive_for_task

    drive = _task_drive_for_task(load_task_result(pool.root, task_id), task_id)
    note, manifest, _report = _stage_mailbox_attachments(SimpleNamespace(DRIVE_ROOT=pool.root), drive, task_id, {
        "chat_attachment_uploads": [{"path": str(path)} for path in files]})
    assert [row["status"] for row in manifest] == ["staged"] * len(files), manifest
    assert write_owner_message(drive, words + note, task_id, msg_id=msg_id, attachment_manifest=list(manifest))
    return words + note, [dict(row) for row in manifest]


def _relayed(pool, task_id, words, data, name):
    """An owner chat message with one stored upload, quoted to the task through the real relay writer
    (``owner_relay.write_relay``) as Main relays it; returns the quoted row's owner text and its upload ref."""
    from ouroboros.chat_uploads import store_upload, stored_refs
    from ouroboros.owner_relay import owner_quote, owner_text, write_relay
    from ouroboros.task_results import load_task_result
    from supervisor.queue_timeouts import _task_drive_for_task

    _stored, ref = store_upload(data, name, data_dir=pool.root)
    row = {"chat_id": 7, "direction": "in", "ts": "2026-10-30T12:00:00+00:00", "text": words,
           "client_message_id": f"chat-{name}", "attachments": stored_refs([ref])}
    drive = _task_drive_for_task(load_task_result(pool.root, task_id), task_id)
    _entry, created = write_relay(drive, task_id, relay_id=f"relay-{name}", note="Main relays the owner's file",
                                  quotes=[owner_quote(pool.root, row)], chat_id=7, teller_task_id="main-turn",
                                  canonical_root=pool.root)
    assert created
    return owner_text(row), stored_refs([ref])[0]


def _owned(pool, row):
    """(label, status, sha256) of ROW's declared inputs; each staged one byte-verified in ROW's OWN store, once."""
    from ouroboros.artifacts import stream_artifact_file

    staged = [entry for entry in row["attachments"] if entry["status"] == "staged"]
    assert len({entry["relpath"] for entry in staged}) == len(staged), "one file each, never one name for two"
    for entry in staged:
        assert stream_artifact_file(_store(pool, row) / entry["relpath"]) == {"size": entry["size"],
                                                                             "sha256": entry["sha256"]}
    return [(entry["label"], entry["status"], entry.get("sha256")) for entry in row["attachments"]]


@pytest.mark.parametrize("memory_mode", ["shared", "forked"])
def test_late_owner_inputs_ride_every_hop_in_each_successors_own_store(pool, tmp_path_factory, monkeypatch, memory_mode):
    """Astra R1: files the owner attached to later messages lived only in the interrupted task's store. After a
    provider failure the successor carried the owner's words but not the files, so read_file failed on the next hop
    and a child it delegated got none. Each successor now holds them in its OWN store beside the declared inputs,
    each once under its own identity (two turns' same-named files stay two files), lost bytes stay a named gap, the
    owner's words stay exact, and the second hop's reader, a delegated child and the first request's native image
    get them after the original copies are gone; a replay copies nothing again."""
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.tools.control_scheduling import _materialize_child_attachment_manifest
    from supervisor.continuation_admission import admit_continuation
    from tests.test_continue_api_opening import _door
    from tests.test_continue_execution_budget import _start
    from tests.test_continue_quiz_origin import _png

    sources = {name: tmp_path_factory.mktemp(name) for name in ("declared", "turn_a", "turn_b", "turn_c")}
    (sources["declared"] / "crash.log").write_text("ATTACHED BYTES\n", encoding="utf-8")
    for turn, text in (("turn_a", "first turn notes\n"), ("turn_b", "second turn notes\n"), ("turn_c", "third turn notes\n")):
        (sources[turn] / "notes.txt").write_text(text, encoding="utf-8")
    (sources["turn_a"] / "late.png").write_bytes(_png())
    (sources["turn_a"] / "gone.txt").write_text("bytes lost before the Continue\n", encoding="utf-8")
    _pool_ready(pool.mp, pool.workers)
    response = _door(pool).post("/api/tasks", json={"description": "Fix the importer", "memory_mode": memory_mode,
                                                    "attachments": [{"path": str(sources["declared"] / "crash.log")}]})
    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]
    pool.mp.setattr(pool.workers, "WORKERS", {})
    sent, _meta = _start(pool, task_id, executed=5)
    words_a, turn_a = _owner_turn(pool, task_id, "Also read my notes and the screenshot",
                                  [sources["turn_a"] / name for name in ("notes.txt", "late.png", "gone.txt")], "owner-a")
    words_b, turn_b = _owner_turn(pool, task_id, "Newer notes", [sources["turn_b"] / "notes.txt"], "owner-b")
    words_r, quoted = _relayed(pool, task_id, "Use these notes too", b"relayed owner notes\n", "notes.txt")
    _provider_failed(pool, task_id, sent)
    for drive in (pool.root, _drive(pool, task_id)):  # bytes the task no longer holds
        (task_artifact_dir_path(drive, task_id) / turn_a[2]["relpath"]).unlink(missing_ok=True)
    tools = _tools(pool, monkeypatch, tmp_path_factory)

    first, _binding = _continue(pool, task_id)

    late = [("notes.txt", "staged", turn_a[0]["sha256"]), ("late.png", "staged", turn_a[1]["sha256"]),
            ("gone.txt", "rejected", turn_a[2]["sha256"]), ("notes.txt", "staged", turn_b[0]["sha256"]),
            ("notes.txt", "staged", quoted["sha256"])]  # a relay's quoted chat upload: a third, from the owner's chat
    hop1 = _owned(pool, first)
    assert hop1 == [("crash.log", "staged", _declared(pool, task_id)[0]["sha256"]), *late]
    assert turn_a[0]["relpath"] != turn_b[0]["relpath"], "the second turn's notes.txt was staged beside the first"
    assert [entry["relpath"] for entry in first["attachment_images"]] == [turn_a[1]["relpath"]]
    gone = turn_a[2]
    assert first["attachments"][3] == {"ordinal": 2, "label": "gone.txt", "mime": gone["mime"], "size": gone["size"],
                                       "sha256": gone["sha256"], "status": "rejected", "reason": "source_missing"}
    assert f"gone.txt ({gone['size']} bytes, sha256 {gone['sha256']})" in first["text"]
    assert [row["content"] for row in first["metadata"]["continuation"]["owner_sources"]["later"]] == [
        words_a, words_b, words_r]
    assert first["attachments"][-1]["relpath"] == f"attachments/{quoted['upload']}", "named by its unique upload"
    assert f"path='{turn_a[0]['relpath']}'" in words_a, "the owner's own words name the file the successor holds"
    assert "first turn notes" in tools.read(first, turn_a[0]["relpath"])
    for drive in (pool.root, _drive(pool, task_id)):  # the original task's own copies go; the next hop never reads them
        shutil.rmtree(task_artifact_dir_path(drive, task_id) / "attachments", ignore_errors=True)
    sent, _meta = _start(pool, first["id"], executed=0)
    words_c, turn_c = _owner_turn(pool, first["id"], "And the latest notes", [sources["turn_c"] / "notes.txt"], "owner-c")
    _provider_failed(pool, first["id"], sent)

    second, _binding = _continue(pool, first["id"])

    held = _owned(pool, second)
    assert held == [*hop1, ("notes.txt", "staged", turn_c[0]["sha256"])], "each input once, in order"
    for relpath, text in ((turn_a[0]["relpath"], "first turn notes"), (turn_b[0]["relpath"], "second turn notes"),
                          (turn_c[0]["relpath"], "third turn notes"), (f"attachments/{quoted['upload']}", "relayed owner")):
        assert text in tools.read(second, relpath)
    later = second["metadata"]["continuation"]["owner_sources"]["later"]
    assert [row["content"] for row in later] == [words_a, words_b, words_r, words_c]
    [native] = _first_request_images(pool, second, tmp_path_factory, monkeypatch)
    assert base64.b64decode(native["image_url"]["url"].split(",", 1)[1]) == (sources["turn_a"] / "late.png").read_bytes()
    projection, error = _materialize_child_attachment_manifest(
        second["task_contract"], tmp_path_factory.mktemp("child_drive"), "child-of-successor",
        owner_drive=pathlib.Path(second.get("drive_root") or pool.root), owner_task_id=second["id"], custody_drive=pool.root)
    assert error == "", error
    assert [(row["label"], row["sha256"]) for row in projection["attachment_manifest"] if row["status"] == "staged"] == [
        (label, sha) for label, status, sha in held if status == "staged"]
    listing = sorted(path.name for path in (_store(pool, second) / "attachments").iterdir())
    assert admit_continuation(first["id"], action_nonce=NONCE)["replay"] is True
    assert sorted(path.name for path in (_store(pool, second) / "attachments").iterdir()) == listing, "nothing copied twice"


def test_a_late_owner_manifest_past_the_preview_rides_whole(pool, tmp_path_factory):
    """A late owner message with more files than the inline preview records them behind its own
    ``attachment_manifest_ref``; the successor reads that reference (``held_attachment_manifest``) and copies every
    one, in order, into its own store under its own reference."""
    from ouroboros.artifacts import resolve_attachment_manifest
    from ouroboros.task_results import load_task_result
    from tests.test_continue_api_opening import _door

    inputs = tmp_path_factory.mktemp("late_parts")
    for index in range(26):
        (inputs / f"late{index:02d}.txt").write_text(f"late part {index}\n", encoding="utf-8")
    _pool_ready(pool.mp, pool.workers)
    response = _door(pool).post("/api/tasks", json={"description": "Fix the importer"})
    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]
    _owner_turn(pool, task_id, "All the parts", [inputs / f"late{index:02d}.txt" for index in range(26)], "owner-parts")
    _shutdown(pool)
    [captured] = [json.loads(raw) for raw in load_task_result(pool.root, task_id)["owner_mailbox"]["rows"]]
    assert captured["attachment_manifest_ref"]["count"] == 26 and len(captured["attachment_manifest"]) == 25

    row, _binding = _continue(pool, task_id)

    assert row["attachment_manifest_ref"]["count"] == 26 and len(row["attachment_manifest"]) == 25
    own = resolve_attachment_manifest(row.get("drive_root") or pool.root, row["id"], row["task_contract"])
    assert [entry["label"] for entry in own] == [f"late{index:02d}.txt" for index in range(26)]
    assert [pathlib.Path(entry["abs_path"]).read_text(encoding="utf-8") for entry in own] == [
        f"late part {index}\n" for index in range(26)]
    assert all(pathlib.Path(entry["abs_path"]).resolve().is_relative_to(_store(pool, row).resolve()) for entry in own)
