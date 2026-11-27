"""Isolated history transport: real Python reader, canonical responses, custody and admission."""
from __future__ import annotations

import asyncio
import io
import json
import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store, skill_lifecycle_queue, utils
from ouroboros.gateway import history, history_live, history_reader
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateway.history_paging import _ANNOTATIONS_NAME, DECODED_CURSOR
from tests._dialogue import confirm_first_use, import_all
from tests._room_archive_world import lost_registry
from tests._task_settings_helpers import isolated_runtime  # noqa: F401


def selectors(thread_id=1):
    return {"thread_id": thread_id, "cursor": None, "around": None}


@pytest.fixture
def reader(monkeypatch):
    instance = history_reader.HistoryReader()
    monkeypatch.setattr(history_reader, "reader", instance)
    yield instance
    instance.stop()
    if instance._thread is not None:
        instance._thread.join(5)
        assert not instance.live_children()


def test_actual_reader_reuses_bytes_and_reports_fallback(isolated_runtime, reader, caplog, monkeypatch):  # noqa: F811
    root = isolated_runtime
    confirm_first_use(root)
    import_all(root)
    dialogue_store.append(root, "chat", {"direction": "in", "text": "вымышленная строка", "chat_id": 1,
                                         "ts": "2026-11-25T19:00:00Z"})
    endpoint = make_chat_history_endpoint(root)

    async def get(**params):
        paths, decoded = [], {}
        path_token = history_reader.READ_PATH.set(paths.append)
        cursor_token = DECODED_CURSOR.set(decoded)
        try:
            response = await endpoint(SimpleNamespace(query_params=params))
        finally:
            DECODED_CURSOR.reset(cursor_token)
            history_reader.READ_PATH.reset(path_token)
        return (response.status_code, response.body, decoded.get("kind")), paths

    async def exercise():
        baseline, paths = await get()
        assert paths == ["server"] and baseline[0] == 200
        page = json.loads(baseline[1])
        row, after = page["messages"][0], page["after_cursor"]
        assert row["text"] == "вымышленная строка" and after
        variants = [({}, 200, None), ({"cursor": after}, 200, "after"),
                    ({"around": ""}, 400, None), ({"around": row["row_address"]}, 200, None)]
        expected = [await get(**params) for params, _, _ in variants]
        reader.start(root, Path(__file__).resolve().parents[1])
        assert await asyncio.to_thread(reader.ready.wait, 8), reader._failure
        pid = reader.pid
        for (params, status, kind), (want, _) in zip(variants, expected):
            assert want[0] == status and want[2] == kind
            assert await get(**params) == (want, ["child"])
            assert reader.pid == pid
        assert await asyncio.gather(get(), get()) == [(baseline, ["child"])] * 2
        # Only the serving process knows the running operation. Pin its row's
        # clock so the same body can be compared after losing the child.
        clock = "2026-11-25T19:00:02Z"
        monkeypatch.setattr(history, "utc_now_iso", lambda: clock)
        monkeypatch.setattr(utils, "utc_now_iso", lambda: clock)
        monkeypatch.setattr(skill_lifecycle_queue, "queue_snapshot", lambda: {"active": {
            "status": "running", "kind": "review", "target": "fixture", "chat_id": 1, "message": "working"}})
        (root / "logs").mkdir(exist_ok=True)
        (root / "logs" / _ANNOTATIONS_NAME).touch()
        child_skill, used = await get()
        assert used == ["child"]
        assert any(row.get("system_type") == "skill_lifecycle" for row in json.loads(child_skill[1])["messages"])
        rejected, used = await get(cursor=after)
        assert (rejected[0], rejected[2], used) == (409, "after", ["child"])
        assert json.loads(rejected[1])["detail"] == "notes_changed"
        # Exact test-owned process, no global stop: service is still available.
        reader._process.kill()
        await asyncio.to_thread(reader._thread.join, 5)
        assert not reader.live_children()
        assert (await get(cursor=after)) == (rejected, ["server"])
        assert await get() == (child_skill, ["server"])
        assert caplog.text.count("room_history_reader_fallback cause=connection_lost") == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("user_site_enabled", [True, False])
def test_direct_reader_uses_selected_userbase_without_inheriting_pythonpath(
    isolated_runtime, reader, monkeypatch, user_site_enabled,  # noqa: F811
):
    import subprocess
    import sys
    import sysconfig

    from ouroboros import process_custody

    root = isolated_runtime
    user_base = root / "state" / "python-userbase"
    packages = Path(sysconfig.get_path("purelib", scheme=sysconfig.get_preferred_scheme("user"),
                                       vars={"userbase": str(user_base)}))
    packages.mkdir(parents=True)
    marker = root / "import-observed.txt"
    unrelated = root / "unrelated"
    unrelated.mkdir()
    (unrelated / "sitecustomize.py").write_text(f"open({str(marker)!r}, 'w').write('wrong path')\n", encoding="utf-8")
    (packages / "history_reader_test_dependency.py").write_text("VALUE = 'selected'\n", encoding="utf-8")
    (packages / "usercustomize.py").write_text(
        "import os\nfrom pathlib import Path\nfrom history_reader_test_dependency import VALUE\n"
        f"Path({str(marker)!r}).write_text(f'{{os.getpid()}}:{{VALUE}}:{{os.getenv(\"READER_UNUSED_TEST\")}}')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONUSERBASE", str(user_base))
    monkeypatch.setenv("PYTHONPATH", str(unrelated))
    monkeypatch.setenv("READER_UNUSED_TEST", "not routed")
    monkeypatch.setenv("PYTHONNOUSERSITE", "" if user_site_enabled else "1")
    repo = Path(__file__).resolve().parents[1]
    spawn = process_custody.spawn_supervised
    environments = []

    def observe_spawn(*args, **kwargs):
        if kwargs.get("purpose") == history_reader.PURPOSE:
            environments.append(dict(kwargs["env"]))
        return spawn(*args, **kwargs)

    monkeypatch.setattr(process_custody, "spawn_supervised", observe_spawn)
    reader.start(root, repo)
    assert reader.ready.wait(8), reader._failure
    env, = environments
    assert env["PYTHONUSERBASE"] == str(user_base)
    assert env["PYTHONNOUSERSITE"] == ("" if user_site_enabled else "1")
    assert env["PYTHONPATH"] == str(repo) and "READER_UNUSED_TEST" not in env
    # The same admitted site settings on the packaged-style base interpreter.
    # A dependency-only venv can launch the real reader without allowing its user site.
    marker.unlink(missing_ok=True)
    env.pop("__PYVENV_LAUNCHER__", None)
    base_python = Path(sys._base_executable).resolve(strict=True)
    probe = subprocess.run([str(base_python), "-c",
                            "import os, sys; assert sys.prefix == sys.base_prefix; print(os.getpid(), flush=True)"],
                           env=env, cwd=root, capture_output=True, check=True, text=True, timeout=10)
    if user_site_enabled:
        assert marker.read_text(encoding="utf-8") == f"{probe.stdout.strip()}:selected:None"
    else:
        assert not marker.exists()


def test_real_503_is_not_retried_in_server(isolated_runtime, reader, caplog):  # noqa: F811
    root = isolated_runtime
    lost_registry(root)
    endpoint = make_chat_history_endpoint(root)
    request = SimpleNamespace(query_params={})

    async def exercise():
        baseline = await endpoint(request)
        assert baseline.status_code == 503
        caplog.clear()
        reader.start(root, Path(__file__).resolve().parents[1])
        assert await asyncio.to_thread(reader.ready.wait, 8), reader._failure
        used = []
        token = history_reader.READ_PATH.set(used.append)
        try:
            response = await endpoint(request)
        finally:
            history_reader.READ_PATH.reset(token)
        assert (response.status_code, response.body, used) == (503, baseline.body, ["child"])
        assert "room_history_reader_fallback" not in caplog.text

    asyncio.run(exercise())


@pytest.mark.parametrize("stop_confirmed", [True, False])
def test_smart_update_retires_reader_before_checkout_without_fencing_history(
    isolated_runtime, reader, monkeypatch, caplog, stop_confirmed,  # noqa: F811
):
    from starlette.responses import JSONResponse

    from ouroboros.gateway import control
    from supervisor import git_ops, update_merge

    root = isolated_runtime
    confirm_first_use(root)
    import_all(root)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", root)
    monkeypatch.setattr(control, "_index_lock_precondition", lambda: None)
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda reason: [])
    plan = {"available": True, "kind": "conflicting", "base_sha": "before", "target_sha": "after"}
    monkeypatch.setattr(update_merge, "plan_managed_update_merge", lambda **kwargs: plan)
    monkeypatch.setattr(update_merge, "acquire_update_lock", object)
    monkeypatch.setattr(update_merge, "release_update_lock", lambda lock: None)

    crossed, respawned = [], []
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: respawned.append(True))

    def before_stash(**kwargs):
        crossed.append(True)
        assert reader._process.poll() is not None and not reader.ready.is_set()
        return None, JSONResponse({"status": "synthetic_checkout_boundary"})

    monkeypatch.setattr(control, "_stash_local_work_fenced", before_stash)
    live_children = reader.live_children
    if not stop_confirmed:
        monkeypatch.setattr(reader, "live_children", lambda *, include_server_reads=True:
                            live_children() if include_server_reads else ["synthetic-unconfirmed"])

    endpoint = make_chat_history_endpoint(root)
    request = SimpleNamespace(query_params={})
    reader.start(root, Path(__file__).resolve().parents[1])
    assert reader.ready.wait(8), reader._failure

    async def exercise():
        paths = []
        token = history_reader.READ_PATH.set(paths.append)
        try:
            before = await endpoint(request)
            update = await asyncio.to_thread(control._apply_smart_update_fenced, object(),
                                            expected_base_sha="before", expected_target_sha="after")
            assert update.status_code == (200 if stop_confirmed else 409)
            assert crossed == ([True] if stop_confirmed else [])
            assert respawned == ([] if stop_confirmed else [True])
            if not stop_confirmed:
                assert "room history reader stop unconfirmed" in json.loads(update.body)["blockers"]
            after = await endpoint(request)
            assert (before.status_code, after.status_code, after.body, paths) == (
                200, 200, before.body, ["child", "server"])
            process = reader._process
            assert process.poll() is not None
            reader.start(root, Path(__file__).resolve().parents[1])
            assert reader._process is process and not reader.ready.is_set()
            assert "room_history_reader_fallback cause=checkout_changing" in caplog.text
        finally:
            history_reader.READ_PATH.reset(token)

    asyncio.run(exercise())


@pytest.mark.parametrize("review_state", ["open", "closed_local", "previous_session"])
def test_real_child_replays_project_evidence_managed_wait_and_paused_review_owner(
    isolated_runtime, reader, monkeypatch, review_state,  # noqa: F811
):
    from ouroboros import deployment_policy, projects_registry, review_operation
    from supervisor import queue
    from tests._store_shared import seed_result

    root = isolated_runtime
    confirm_first_use(root)
    import_all(root)
    room = projects_registry.create_project(root, "fixture-project", name="Fixture Project")
    chat = room["chat_id"]
    task = {"chat_id": chat, "project_id": room["id"], "delegation_role": "root"}
    stamp = "2026-11-25T19:00:01Z"
    monkeypatch.setattr(history, "utc_now_iso", lambda: stamp)
    monkeypatch.setattr(utils, "utc_now_iso", lambda: stamp)
    monkeypatch.setattr(queue, "RUNNING", {"project-live": {"task": {**task, "id": "project-live"}}})
    seed_result(root, "project-live", {**task, "root_task_id": "project-live", "status": "running", "tool_calls": 2})

    role = deployment_policy.CHECKER_ROLE
    monkeypatch.setattr(deployment_policy, "managed_model_roles", lambda: frozenset({role}))
    dialogue_store.append(root, "progress", {"type": "task_model_wait", "task_id": "checker-author",
                                            "chat_id": chat, "ts": stamp, "role": role, "wait_id": "checker-fixture",
                                            "revision": 1, "task_attempt": 1, "state": "waiting", "is_progress": False})

    paused, retry, owner = "previous-author", "queued-retry", "review-controller"
    controller = review_operation.controller_identity()
    if review_state == "previous_session":
        controller = {**controller, "session": "older-custody-session"}
    monkeypatch.setattr(review_operation, "_LIVE", {
        owner: SimpleNamespace(closed=review_state != "open")
    } if review_state != "previous_session" else {})
    monkeypatch.setattr(queue, "PENDING", [{**task, "id": retry, "root_task_id": paused,
                                          "timeout_retry_from": paused, "original_task_id": paused}])
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {paused: {"cause": "owner_pause"}})
    seed_result(root, paused, {**task, "root_task_id": paused, "status": "running", "tool_calls": 3,
                               "owner_pause": {"state": "paused"}, "review_operations": {
                                   owner: {"state": "dispatched", "controller": controller}}})

    endpoint = make_chat_history_endpoint(root)

    async def get(path):
        observed = []
        token = history_reader.READ_PATH.set(observed.append)
        try:
            response = await endpoint(SimpleNamespace(query_params={"chat_id": str(chat)}))
        finally:
            history_reader.READ_PATH.reset(token)
        assert (response.status_code, observed) == (200, [path])
        return response.body

    async def exercise():
        inline = await get("server")
        messages = json.loads(inline)["messages"]
        evidence = {row.get("task_id") for row in messages if row.get("system_type") == "task_evidence"}
        assert "project-live" in evidence
        assert (paused in evidence) is (review_state == "open")
        [wait] = [row for row in messages if row.get("system_type") == "task_model_wait"]
        assert wait["model_waits"]["checker-fixture"]["managed"] is True
        reader.start(root, Path(__file__).resolve().parents[1])
        assert await asyncio.to_thread(reader.ready.wait, 8), reader._failure
        assert await get("child") == inline

        capture = history_live.capture
        for key, replacement in (("managed_roles", []), ("activities", [[], [], {}, [], [], []]),
                                 ("review_operations", []) if review_state == "open" else ("controller", {"pid": 0})):
            # Both witnesses matter. Keep the same IPC path, actor and response assembler.
            monkeypatch.setattr(history_live, "capture", lambda *args: {**capture(*args), key: replacement})
            broken = json.loads(await get("child"))["messages"]
            if key == "managed_roles":
                [wait] = [row for row in broken if row.get("system_type") == "task_model_wait"]
                assert "managed" not in wait["model_waits"]["checker-fixture"]
            elif key == "activities":
                assert not any(row.get("task_id") == "project-live" and row.get("system_type") == "task_evidence"
                               for row in broken)
            else:
                broken_evidence = {row.get("task_id") for row in broken if row.get("system_type") == "task_evidence"}
                assert (paused in broken_evidence) is (review_state != "open")

    asyncio.run(exercise())


def test_ingress_witnesses_are_the_same_in_the_reader_and_server(isolated_runtime, reader, monkeypatch):  # noqa: F811
    from ouroboros.process_custody import current_custody_session_id
    from supervisor import message_bus

    root = isolated_runtime
    confirm_first_use(root)
    import_all(root)
    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    session = current_custody_session_id()
    for index, name in enumerate(("pending", "completed", "previous"), start=1):
        dialogue_store.append(root, "chat", {"direction": "in", "chat_id": 1, "text": name,
                                              "client_message_id": name, "ingress_process": session, "ingress_accepted": True,
                                              "ts": f"2026-11-25T19:00:0{index}Z"}, dispatch={"dispatcher": "web"})
        if name != "pending":
            witness = session if name == "completed" else "previous-generation"
            assert dialogue_store.claim_dispatch(root, 1, name, process=witness)
            assert dialogue_store.complete_dispatch(root, 1, name)
    endpoint = make_chat_history_endpoint(root)
    request = SimpleNamespace(query_params={})

    async def exercise():
        baseline = await endpoint(request)
        assert baseline.status_code == 200
        rows = {row["text"]: row for row in json.loads(baseline.body)["messages"]}
        for key, item in (("ingress_pending", "pending"), ("ingress_dispatched", "completed")):
            assert {name for name, row in rows.items() if row.get(key)} == {item}
        reader.start(root, Path(__file__).resolve().parents[1])
        assert await asyncio.to_thread(reader.ready.wait, 8)
        paths = []
        token = history_reader.READ_PATH.set(paths.append)
        try:
            replay = await endpoint(request)
            assert (replay.status_code, replay.body, paths) == (200, baseline.body, ["child"])
        finally:
            history_reader.READ_PATH.reset(token)

    asyncio.run(exercise())


def test_history_observes_owed_dispatch_before_message_bus_init(isolated_runtime, reader, monkeypatch):  # noqa: F811
    from ouroboros.process_custody import current_custody_session_id
    from supervisor import message_bus, message_ingress

    root = isolated_runtime
    confirm_first_use(root)
    import_all(root)
    monkeypatch.setattr(message_bus, "DATA_DIR", None)
    session = current_custody_session_id()
    names = ("pending", "completed", "previous", "previous-entered")
    for index, name in enumerate(names, start=1):
        dialogue_store.append(root, "chat", {"direction": "in", "chat_id": 1, "text": name,
                                              "client_message_id": name, "ingress_accepted": True,
                                              "ingress_process": session if name == "completed" else "previous-generation",
                                              "ts": f"2026-11-25T19:00:0{index}Z"},
                              dispatch=message_ingress.obligation("bus", name, {"chat_id": 1, "client_message_id": name}))
        if name != "pending":
            witness = session if name == "completed" else "previous-generation"
            assert dialogue_store.claim_dispatch(root, 1, name, process=witness)
            if name != "previous-entered":
                assert dialogue_store.complete_dispatch(root, 1, name)
    endpoint = make_chat_history_endpoint(root)
    request = SimpleNamespace(query_params={})

    async def exercise():
        baseline = await endpoint(request)
        assert baseline.status_code == 200
        rows = {row["text"]: row for row in json.loads(baseline.body)["messages"]}
        assert set(rows) == set(names) and all(row.get("ingress_accepted") is True for row in rows.values())
        for key, item in (("ingress_pending", "pending"), ("ingress_dispatched", "completed")):
            assert {name for name, row in rows.items() if row.get(key)} == {item}
        reader.start(root, Path(__file__).resolve().parents[1])
        assert await asyncio.to_thread(reader.ready.wait, 8)
        paths = []
        token = history_reader.READ_PATH.set(paths.append)
        try:
            replay = await endpoint(request)
            assert (replay.status_code, replay.body, paths) == (200, baseline.body, ["child"])
        finally:
            history_reader.READ_PATH.reset(token)
        assert [dialogue_store.dispatch_state(root, 1, name) for name in names] == [
            "pending", "completed", "completed", "dispatched",
        ], "both GET paths observe without taking or replaying the dispatch"

    asyncio.run(exercise())


def test_two_dispatched_reads_hold_slots_even_after_caller_cancelled(tmp_path, reader, monkeypatch):
    reader.bind(tmp_path)
    reader.ready.set()
    admitted = []

    async def exercise():
        release = [asyncio.Event() for _ in range(3)]
        both = asyncio.Event()

        async def dispatch(root, selected, capture):
            nth = len(admitted)
            admitted.append((selected["thread_id"], capture(root, selected)))
            if nth == 1:
                both.set()
            await release[nth].wait()
            return 200, b"unchanged", None

        value = ["first"]
        monkeypatch.setattr(history_live, "capture", lambda *_: value[0])
        monkeypatch.setattr(reader, "_dispatch", dispatch)
        tasks = [asyncio.create_task(reader.read(tmp_path.resolve(), selectors(i), pytest.fail)) for i in range(3)]
        await asyncio.wait_for(both.wait(), 2)
        assert admitted == [(0, "first"), (1, "first")]
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
        assert len(admitted) == 2
        value[0] = "new fact at admission"
        release[1].set()
        assert await tasks[1] == (200, b"unchanged", None)
        release[0].set()
        release[2].set()
        await tasks[2]
        assert admitted[-1] == (2, "new fact at admission")

    asyncio.run(exercise())


def test_unavailable_child_does_not_reduce_inline_concurrency(tmp_path, reader):
    root = tmp_path.resolve()
    reader.bind(root)
    entered = [threading.Event() for _ in range(3)]
    release = threading.Event()

    def inline(index):
        entered[index].set()
        assert release.wait(5)
        return 200, b"same", None

    async def exercise():
        requests = [asyncio.create_task(reader.read(root, selectors(i), lambda i=i: inline(i))) for i in range(3)]
        try:
            assert all(await asyncio.gather(*(asyncio.to_thread(event.wait, 2) for event in entered)))
        finally:
            release.set()
            assert await asyncio.gather(*requests) == [(200, b"same", None)] * 3

    asyncio.run(exercise())


def test_stop_releases_waiters_without_starting_more_work(tmp_path, reader, monkeypatch):
    root = tmp_path.resolve()
    reader.bind(root)
    reader.ready.set()

    async def exercise():
        active, waiting, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        dispatched = []

        async def dispatch(_root, selected, _capture):
            dispatched.append(selected["thread_id"])
            if len(dispatched) == 2:
                active.set()
            await release.wait()
            raise history_reader.ReaderUnavailable("connection_lost")

        monkeypatch.setattr(reader, "_dispatch", dispatch)

        async def third():
            waiting.set()
            return await reader.read(root, selectors(3), pytest.fail)

        requests = [asyncio.create_task(reader.read(root, selectors(i), pytest.fail)) for i in (1, 2)]
        await asyncio.wait_for(active.wait(), 2)
        requests.append(asyncio.create_task(third()))
        await asyncio.wait_for(waiting.wait(), 2)
        reader.stop()
        release.set()
        responses = await asyncio.wait_for(asyncio.gather(*requests), 2)
        assert [row[0] for row in responses] == [503, 503, 503] and dispatched == [1, 2]

    asyncio.run(exercise())


@pytest.mark.parametrize("truncated", [b"\x00", history_reader._HEADER.pack(2, 1) + b"{}"])
def test_incomplete_reply_settles_every_waiter_even_if_dead_stdin_cannot_flush(tmp_path, reader, monkeypatch, caplog,
                                                                             truncated):
    from ouroboros import startup_historical_audit

    reply = io.BytesIO()
    history_reader._frame(reply, {"ready": 123})
    reply.write(truncated)
    reply.seek(0)
    closed = []

    def close():
        assert reader._write_lock.locked()
        closed.append(True)
        raise BrokenPipeError

    process = SimpleNamespace(pid=123, stdout=reply, stdin=SimpleNamespace(close=close),
                              stderr=io.BytesIO(b"synthetic private body\nFailed to read chat history\n"),
                              poll=lambda: 0, wait=lambda: 0, kill=lambda: None)
    monkeypatch.setattr(startup_historical_audit, "spawn_maintenance_job",
                        lambda *a, on_spawn, **kw: on_spawn(process) or process)
    futures = [Future(), Future()]
    reader._requests.update({i: (future, None) for i, future in enumerate(futures)})
    reader._run(tmp_path, tmp_path)
    assert not reader._requests and reply.closed and closed == [True]
    assert all(isinstance(future.exception(0), history_reader.ReaderUnavailable) for future in futures)
    assert "synthetic private body" not in caplog.text


def test_worker_assembly_exception_still_reaches_the_controller(isolated_runtime, monkeypatch):  # noqa: F811
    from ouroboros.gateway import history_response

    def assemble(_render, _root, *, thread_id, **kwargs):
        if thread_id == 1:
            raise ValueError("synthetic private body")
        return 200, b"{}", None

    monkeypatch.setattr(history_response, "assemble", assemble)
    source, output = io.BytesIO(), io.BytesIO()
    for request_id in (1, 2):
        history_reader._frame(source, {"id": request_id, "selectors": selectors(request_id), "live": {}})
    source.seek(0)
    history_reader._responses(source, output, isolated_runtime)
    output.seek(0)
    ready, _ = history_reader._receive(output)
    assert "ready" in ready
    responses = dict((meta["id"], (meta, body)) for meta, body in
                     (history_reader._receive(output), history_reader._receive(output)))
    assert responses[1] == ({"id": 1, "failed": True}, b"")
    assert responses[2] == ({"id": 2, "status": 200, "cursor_kind": None}, b"{}")


def test_anchored_read_copies_saved_custody_without_touching_unneeded_live_census(isolated_runtime, monkeypatch):  # noqa: F811
    from ouroboros.gateway import state
    from ouroboros.presence_recovery import copied_queue_row
    from supervisor import queue, queue_snapshot

    root = isolated_runtime
    (root / "state").mkdir(exist_ok=True)
    (root / "state" / "queue_snapshot.json").write_text(
        json.dumps({"running": [], "pending": [{"id": "unrestored-presence"}]}), encoding="utf-8")
    monkeypatch.setattr(queue, "DRIVE_ROOT", root)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", root / "state" / "queue_snapshot.json")
    monkeypatch.setattr(queue_snapshot, "_RESTORING", {str(queue.QUEUE_SNAPSHOT_PATH.absolute()): -1})
    monkeypatch.setattr(queue, "_queue_lock", SimpleNamespace(acquire=lambda **_kw: pytest.fail("unneeded lock")))
    monkeypatch.setattr(state, "copy_activity_inputs", lambda *a, **kw: pytest.fail("unneeded live census"))
    live = history_live.capture(root, {**selectors(), "around": "row:synthetic-anchor"})
    assert live["activities"] is None
    assert copied_queue_row(root, "unrestored-presence", live["custody_root"], live["queue_custody"]) == "listed"
    assert copied_queue_row(root, "unlisted", live["custody_root"], live["queue_custody"]) == ""


def test_unavailable_live_custody_is_not_copied_as_an_empty_queue(isolated_runtime, monkeypatch):  # noqa: F811
    from ouroboros.presence_recovery import copy_queue_custody_inputs
    from supervisor import queue, queue_snapshot

    root = isolated_runtime
    monkeypatch.setattr(queue, "DRIVE_ROOT", root)
    monkeypatch.setattr(queue_snapshot, "saved_queue_decision", lambda: False)
    monkeypatch.setattr(queue, "_queue_lock", threading.Lock())
    with queue._queue_lock, pytest.raises(RuntimeError, match="live queue custody unavailable"):
        copy_queue_custody_inputs(root)


def test_handoff_disallows_new_inline_reads_and_refusal_cannot_undo_shutdown(tmp_path, reader, monkeypatch):
    from ouroboros import store_switch

    reader.bind(tmp_path)
    handoff, refusal = reader.prepare_handoff(tmp_path.resolve())
    assert handoff is not None and refusal == ""
    monkeypatch.setattr(store_switch, "inline_history_ready", lambda _root: True)

    async def exercise():
        used = []
        token = history_reader.READ_PATH.set(used.append)
        try:
            async def request(inline):
                return await reader.read(tmp_path.resolve(), selectors(), inline)

            assert (await request(pytest.fail))[0] == 503
            assert used == ["handoff"]
            reader.continue_inline(handoff)
            assert await request(lambda: (200, b"ordinary", None)) == (200, b"ordinary", None)
            reader.stop()
            reader.continue_inline(handoff)
            assert (await request(pytest.fail))[0] == 503
            assert used == ["handoff", "server", "handoff"]
        finally:
            history_reader.READ_PATH.reset(token)

    asyncio.run(exercise())


def test_a_cancelled_waiting_request_never_takes_a_build_slot(tmp_path, reader, monkeypatch):
    reader.bind(tmp_path)
    reader.ready.set()
    entered, calls = [], []

    async def exercise():
        finish, both, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def dispatch(root, selected, capture):
            calls.append(selected["thread_id"])
            if len(calls) == 2:
                both.set()
            await finish.wait()
            return 200, b"same", None

        monkeypatch.setattr(reader, "_dispatch", dispatch)

        async def get(i):
            entered.append(i)
            if i == 2:
                waiting.set()
            return await reader.read(tmp_path.resolve(), selectors(i), pytest.fail)

        taken = [asyncio.create_task(get(i)) for i in range(2)]
        await asyncio.wait_for(both.wait(), 2)
        queued = asyncio.create_task(get(2))
        try:
            await asyncio.wait_for(waiting.wait(), 2)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
        finally:
            finish.set()
            await asyncio.gather(*taken)
        assert await get(3) == (200, b"same", None)
        assert entered == [0, 1, 2, 3] and calls == [0, 1, 3]

    asyncio.run(exercise())


def test_http_disconnect_skips_contended_admission_but_not_dispatched_work(tmp_path, reader, monkeypatch):
    import socket

    from starlette.applications import Starlette
    from starlette.routing import Route
    from uvicorn import Config
    from uvicorn.protocols.http.h11_impl import H11Protocol
    from uvicorn.server import ServerState

    from ouroboros.gateway.history_attribution import drain, recorded

    reader.bind(tmp_path)
    reader.ready.set()
    app = Starlette(routes=[Route("/api/chat/history", recorded(make_chat_history_endpoint(tmp_path)))])
    drain()

    async def exercise():
        running = [asyncio.Event() for _ in range(5)]
        release = [asyncio.Event() for _ in range(5)]
        receiving = [asyncio.Event() for _ in range(5)]
        finished = [asyncio.Event() for _ in range(5)]
        calls, clients, disconnect_events = [], {}, []

        async def dispatch(_root, selected, _capture):
            chat = selected["thread_id"]
            calls.append(chat)
            running[chat].set()
            await release[chat].wait()
            return 200, b"{}", None

        async def observed_app(scope, receive, send):
            chat = int(scope["query_string"].split(b"&", 1)[0].split(b"=")[1])

            async def observe_receive():
                receiving[chat].set()
                message = await receive()
                if message["type"] == "http.disconnect":
                    disconnect_events.append(chat)
                return message

            try:
                await app(scope, observe_receive, send)
            finally:
                finished[chat].set()

        config = Config(observed_app, http="h11", interface="asgi3", lifespan="off", access_log=False,
                        proxy_headers=False, log_config=None)
        config.load()
        server = ServerState()

        async def request(chat):
            # Connected test-owned descriptors: actual Uvicorn/HTTP/EOF, no
            # listener, local port, credentials, browser profile or running app.
            left, right = socket.socketpair()
            try:
                transport, _ = await asyncio.get_running_loop().connect_accepted_socket(
                    lambda: H11Protocol(config, server, {}), left)
                incoming, writer = await asyncio.open_connection(sock=right)
            except Exception:
                left.close()
                right.close()
                raise
            clients[chat] = writer, transport
            writer.write(f"GET /api/chat/history?chat_id={chat}&reason=navigate HTTP/1.1\r\n"
                         "Host: fixture\r\nConnection: close\r\n\r\n".encode())
            await writer.drain()
            return incoming

        async def disconnect(chat):
            writer, _ = clients[chat]
            writer.close()
            await writer.wait_closed()

        async def wait_for(event):
            await asyncio.wait_for(event.wait(), 3)

        monkeypatch.setattr(reader, "_dispatch", dispatch)
        monkeypatch.setattr(history, "_assemble_history_response", pytest.fail)
        try:
            await request(1)
            await request(2)
            await wait_for(running[1])
            await wait_for(running[2])
            assert not receiving[1].is_set() and not receiving[2].is_set()  # free admission doesn't read the socket
            await request(3)
            await wait_for(receiving[3])
            await disconnect(3)
            await wait_for(finished[3])
            assert calls == [1, 2] and disconnect_events == [3]
            [row] = drain()["groups"]
            assert (row["room"], row["reason"], row["status"], row["bytes"], row["n"]) == (
                "other", "navigate", "none", 0, 1)

            await disconnect(1)  # a build already on the pipe still holds its permit
            fourth = await request(4)
            await wait_for(receiving[4])
            assert calls == [1, 2] and not finished[1].is_set()
            release[2].set()
            await wait_for(running[4])
            release[4].set()
            assert (await asyncio.wait_for(fourth.readline(), 3)).startswith(b"HTTP/1.1 200 ")
            assert calls == [1, 2, 4]
        finally:
            for event in release:
                event.set()
            for writer, transport in clients.values():
                writer.close()
                transport.close()
            if server.tasks:
                await asyncio.wait_for(asyncio.gather(*server.tasks), 3)
            drain()

    asyncio.run(exercise())


def test_received_cursor_survives_body_loss_and_abandoned_reply_keeps_next_frame(reader):
    def stream(*frames):
        out = io.BytesIO()
        for request_id, kind in frames:
            history_reader._frame(out, {"id": request_id, "status": 200, "cursor_kind": kind}, b"{}")
        return out.getvalue()

    first, second, decoded = Future(), Future(), {}
    reader._requests[1] = (first, decoded)
    with pytest.raises(EOFError):
        reader._read_reply(io.BytesIO(stream((1, "after"))[:-1]))
    assert decoded == {"kind": "after"} and not first.done()
    reader._requests.pop(1)  # the awaiting dispatch already left its table
    reader._requests[2] = (second, {})
    replies = io.BytesIO(stream((1, "after"), (2, "older")))
    reader._read_reply(replies)
    reader._read_reply(replies)
    assert second.result(0) == (200, b"{}", "older") and not reader._requests


def test_reader_diagnostics_accept_exact_fixed_crlf_and_lf_only(caplog):
    fixed = b"Failed to read chat history"
    history_reader._log_stderr(io.BytesIO(fixed + b"\r\n" + fixed + b"\n" +
                                          fixed + b" PRIVATE\r\n\xff\n"))
    assert caplog.messages == ["room_history_reader: Failed to read chat history"] * 2


def test_child_loss_wakes_queued_get_while_two_server_captures_still_hold_slots(isolated_runtime, reader, monkeypatch):  # noqa: F811
    root = isolated_runtime.resolve()
    confirm_first_use(root)
    import_all(root)
    dialogue_store.append(root, "chat", {"direction": "in", "text": "kept", "chat_id": 1,
                                        "ts": "2026-11-25T19:00:00Z"})
    endpoint = make_chat_history_endpoint(root)
    blocking, release = threading.Barrier(3), threading.Event()
    capture = history_live.capture

    def hold(*args):
        blocking.wait(timeout=5)
        assert release.wait(5)
        return capture(*args)

    async def exercise():
        params = SimpleNamespace(query_params={})
        expected = await endpoint(params)
        assert expected.status_code == 200 and b"kept" in expected.body
        reader.start(root, Path(__file__).resolve().parents[1])
        assert await asyncio.to_thread(reader.ready.wait, 8), reader._failure
        monkeypatch.setattr(history_live, "capture", hold)
        begun = [asyncio.create_task(endpoint(params)) for _ in range(2)]
        receiving = asyncio.Event()

        async def receive():
            receiving.set()
            await asyncio.Future()

        try:
            await asyncio.to_thread(blocking.wait, 5)
            queued = asyncio.create_task(endpoint(SimpleNamespace(query_params={}, receive=receive)))
            await asyncio.wait_for(receiving.wait(), 3)
            reader._process.kill()  # exact test-owned PID; both capture threads stay blocked
            result = await asyncio.wait_for(queued, 3)
            assert (result.status_code, result.body) == (200, expected.body)
            assert reader._server_reads == 2 and not any(task.done() for task in begun)
        finally:
            release.set()
            await asyncio.gather(*begun)

    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["capture", "inline"])
def test_server_read_is_in_the_handoff_and_restart_census(tmp_path, reader, monkeypatch, phase):
    reader.bind(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def hold(result):
        entered.set()
        assert release.wait(5)
        return result

    if phase == "capture":
        reader.ready.set()
        monkeypatch.setattr(history_live, "capture", lambda *_: hold({}))
        inline = pytest.fail
    else:
        def inline():
            return hold((200, b"same", None))

    def read(produce):
        return reader.read(tmp_path.resolve(), selectors(), produce)

    async def exercise():
        active = asyncio.create_task(read(inline))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            monkeypatch.setattr(history_reader, "_HANDOFF_SECONDS", 0)
            handoff, refusal = reader.prepare_handoff(tmp_path.resolve())
            assert handoff is not None and refusal and reader.live_children()
            assert (await read(pytest.fail))[0] == 503
        finally:
            release.set()
            await active
        assert not reader.live_children()
        assert reader.prepare_handoff(tmp_path.resolve())[1] == ""

    asyncio.run(exercise())


def test_cancelled_inline_history_is_counted_with_the_decoded_kind(tmp_path, reader, monkeypatch):
    from ouroboros.gateway.history_attribution import drain, recorded

    reader.bind(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def assemble(*args, **kwargs):
        # The selector records its decoded kind on this thread, then is slow.
        DECODED_CURSOR.get()["kind"] = "after"
        entered.set()
        assert release.wait(5)
        return b"{}"

    monkeypatch.setattr(history, "_assemble_history_response", assemble)
    endpoint = recorded(make_chat_history_endpoint(tmp_path))
    drain()

    async def exercise():
        task = asyncio.create_task(endpoint(SimpleNamespace(query_params={"cursor": "fixture"})))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            [row] = drain()["groups"]
            assert (row["mode"], row["status"], row["n"], row["bytes"]) == ("after", "none", 1, 0)
        finally:
            release.set()
            if not task.done():
                await task

    asyncio.run(exercise())


@pytest.mark.parametrize("where, lost", [("pending", False), ("lane", True), (None, False)])
def test_presence_end_fact_uses_serving_queue_custody_not_child_globals(isolated_runtime, monkeypatch, where, lost):  # noqa: F811
    from ouroboros import presence_carrier, presence_recovery, presence_runner
    from ouroboros.task_reads import TaskReads
    from ouroboros.task_status import _presence_attempt_lost

    root = isolated_runtime
    row = {"task_id": "test-turn", "status": "interrupted", "metadata": {"presence": {}, "source": "presence"},
           "presence_recovery_recorded": {"summary_id": "observed"}}
    monkeypatch.setattr(presence_carrier, "moved_text_turn", lambda *_a, **_kw: True)
    monkeypatch.setattr(presence_runner, "presence_turn_is_live", lambda *_a, **_kw: False)
    monkeypatch.setattr(presence_recovery, "_queue_row", lambda *_: where)
    assert _presence_attempt_lost(root, row, {}) is lost
    from ouroboros.review_operation import controller_identity

    facts = {"custody_root": str(root.resolve()), "queue_custody": (
                 {"live": {"test-turn": where}, "saved_path": ""} if where else None),
             "controller": controller_identity(), "epoch": 1, "direct": {}, "presence": []}
    reads = TaskReads(root, history_live=json.loads(json.dumps(facts)))
    monkeypatch.setattr(presence_runner, "presence_turn_is_live", lambda *_a, **_kw: pytest.fail("child local liveness"))
    monkeypatch.setattr(presence_recovery, "_queue_row", lambda *_a: pytest.fail("child local queue"))
    assert _presence_attempt_lost(root, row, {}, reads=reads) is lost


def test_anchor_capture_defers_saved_queue_read_until_a_selected_turn_needs_it(isolated_runtime, monkeypatch):  # noqa: F811
    from ouroboros import presence_recovery
    from ouroboros.gateway import state
    from supervisor import queue, queue_snapshot

    root = isolated_runtime.resolve()
    path = Path(queue.QUEUE_SNAPSHOT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"running": [], "pending": [{"id": "saved-turn"}]}), encoding="utf-8")
    monkeypatch.setattr(queue_snapshot, "saved_queue_decision", lambda: None)
    monkeypatch.setattr(queue, "PENDING", [{"id": "live-turn"}])
    original = queue_snapshot.snapshot_names_work
    reads = []

    def observe(path, **kwargs):
        reads.append(Path(path))
        return original(path, **kwargs)

    monkeypatch.setattr(queue_snapshot, "snapshot_names_work", observe)
    monkeypatch.setattr(state, "copy_activity_inputs", lambda *_a, **_kw: pytest.fail("anchored work census"))
    monkeypatch.setattr(skill_lifecycle_queue, "queue_snapshot", lambda: pytest.fail("anchored skill census"))
    snapshot = history_live.capture(root, {**selectors(), "around": "fixture"})["queue_custody"]
    assert reads == []
    args = str(root), snapshot
    assert presence_recovery.copied_queue_row(root, "saved-turn", *args) == "listed"
    assert presence_recovery.copied_queue_row(root, "live-turn", *args) == ""
    assert reads == [path, path]
    path.write_text('{"running":[],"pending":[]}', encoding="utf-8")
    assert presence_recovery.copied_queue_row(root, "live-turn", *args) == "pending"


def test_pre_exec_presence_witness_is_compared_to_the_serving_generation(isolated_runtime, monkeypatch):  # noqa: F811
    from ouroboros import review_operation, task_status
    from ouroboros.task_reads import TaskReads

    observer = {"pid": 123, "session": "serving", "birth": "same-birth", "kernel_birth": ""}
    row = {"task_id": "test-turn", "status": "running", "metadata": {"source": "presence", "presence": {}},
           "presence_continuation": {"author_controller": {**observer, "session": "previous"}}}
    root = isolated_runtime.resolve()
    facts = {"custody_root": str(root), "queue_custody": {"live": {}, "saved_path": ""},
             "controller": observer, "epoch": 1, "direct": {},
             "presence": []}
    reads = TaskReads(root, history_live=json.loads(json.dumps(facts)))
    monkeypatch.setattr(review_operation.os, "getpid", lambda: 321)
    monkeypatch.setattr("ouroboros.platform_layer.pid_is_alive", lambda _: True)
    monkeypatch.setattr("ouroboros.process_containment.pid_is_zombie", lambda _: False)
    monkeypatch.setattr("ouroboros.platform_layer.process_start_time", lambda _: "same-birth")
    assert task_status._presence_attempt_lost(root, row, {}, reads=reads)
    row["presence_continuation"]["author_controller"] = observer
    assert not task_status._presence_attempt_lost(root, row, {}, reads=reads)


def test_orphan_observation_respects_serving_presence_direct_and_clock(tmp_path, monkeypatch):
    from ouroboros import task_status
    from ouroboros.task_reads import TaskReads

    root = tmp_path.resolve()
    tid, clock = "turn", 100
    row = {"status": "running", "ts": "1970-01-01T00:00:01Z"}
    snapshot = {"ts": "1970-01-01T00:01:40Z", "running": [], "pending": []}
    events = SimpleNamespace(latest_event_ts=lambda _: 1, latest_worker_boot=lambda: 30)
    monkeypatch.setattr(task_status, "_ORPHAN_RUNNING_GRACE_SECONDS", 10)
    monkeypatch.setattr(task_status.time, "time", lambda: 10000)
    monkeypatch.setattr("ouroboros.presence_runner.presence_turn_is_live", lambda *_a, **_kw: False)
    monkeypatch.setattr(task_status, "queue_snapshot_observation", lambda _s, *, boot=False, now=None: {"fresh": now == clock})
    live = {"epoch": clock, "direct": {tid: str(root)}, "presence": []}
    reads = TaskReads(root, history_live=live)
    assert not task_status._is_stale_orphan_running_task(root, tid, row, events, snapshot, reads=reads)
    live["direct"].clear()
    assert task_status._is_stale_orphan_running_task(root, tid, row, events, snapshot, reads=reads)
    live["presence"] = [[str(root), tid]]
    assert not task_status._is_stale_orphan_running_task(root, tid, row, events, snapshot, reads=reads)
    live["presence"].clear()
    row["ts"] = snapshot["ts"]
    assert not task_status._is_stale_orphan_running_task(root, tid, row, events, snapshot, reads=reads)


def test_history_uses_the_direct_venv_interpreter_without_retargeting_other_jobs(tmp_path, monkeypatch):
    from ouroboros import platform_layer, startup_historical_audit

    launcher = str(tmp_path / "venv" / "Scripts" / "python.exe")
    base = tmp_path / "python.exe"
    base.touch()
    real_command = platform_layer.direct_python_command
    monkeypatch.setattr(startup_historical_audit.sys, "executable", launcher)
    monkeypatch.setattr(platform_layer, "direct_python_command",
                        lambda args: real_command(args, windows=True, executable=launcher, base=str(base)))
    calls = []
    monkeypatch.setattr("ouroboros.process_custody.spawn_supervised", lambda args, **kw: calls.append((args, kw["env"])))
    startup_historical_audit.spawn_maintenance_job(tmp_path, tmp_path, history_reader.PURPOSE, direct_interpreter=True)
    startup_historical_audit.spawn_maintenance_job(tmp_path, tmp_path, "startup_historical_audit")
    (reader_args, reader_env), (audit_args, audit_env) = calls
    assert (reader_args[:3], reader_env["__PYVENV_LAUNCHER__"]) == (
        [str(base), "-m", "ouroboros.gateway.history_reader"], launcher)
    assert audit_args[0] == launcher and "__PYVENV_LAUNCHER__" not in audit_env


def test_each_lease_releases_canonical_and_nested_root_handles(isolated_runtime):  # noqa: F811
    from ouroboros import store
    from ouroboros.store_import import ensure_current_store

    child = isolated_runtime / "retained-child"
    child.mkdir()
    ensure_current_store(isolated_runtime)
    ensure_current_store(child)
    handles = []
    with store.thread_read_connections():
        for root in (isolated_runtime, child):
            with store.open_store(root, write=False) as txn:
                handles.append(txn.conn)
        with dialogue_store.transaction(isolated_runtime, write=True) as conn:
            handles.append(conn)
        with dialogue_store.transaction(isolated_runtime, write=False) as conn:
            handles.append(conn)
    from sqlite3 import ProgrammingError

    for handle in handles:
        with pytest.raises(ProgrammingError, match="closed"):
            handle.execute("SELECT 1")



def test_live_lane_rows_share_one_nonblocking_queue_lease(tmp_path, monkeypatch):
    from ouroboros.gateway import state
    from supervisor import active_activity, conversation_lane, queue

    class QueueLease:
        held = False
        calls = []

        def acquire(self, blocking=True):
            self.calls.append(blocking)
            assert not blocking, "the live lane read must not enter a blocking acquisition"
            if self.held:
                return False
            self.held = True
            return True

        def release(self):
            self.held = False

        def __enter__(self):
            return self.acquire()

    lock = QueueLease()
    for module in (queue, conversation_lane):
        monkeypatch.setattr(module, "_queue_lock", lock)
    rows = {"carried": {"lane": "conversation", "task": {"id": "carried", "chat_id": 1}}}
    monkeypatch.setattr(conversation_lane, "_pool", lambda: SimpleNamespace(RUNNING=rows))
    monkeypatch.setattr(queue, "RUNNING", rows)
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})

    def snapshot(**_kw):
        assert not lock.held
        return [{"activity_id": "inline", "chat_id": 1}]

    def presence(_root, *, availability, listed):
        assert not lock.held
        assert listed == {"inline", "carried"}
        return []

    monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: SimpleNamespace(snapshot=snapshot))
    monkeypatch.setattr(state, "_presence_turns_snapshot_safe", presence)
    availability = {"complete": True}
    direct, _, _, running, _, _ = state.copy_activity_inputs(
        tmp_path, owners_only=True, availability=availability, queue_nowait=True)
    assert [row["activity_id"] for row in direct] == ["inline", "carried"]
    assert [item[0] for item in running] == ["carried"] and availability["complete"]
    assert lock.calls == [False] and not lock.held
    lock.held = True
    with pytest.raises(RuntimeError, match="live activity inputs unavailable"):
        state.copy_activity_inputs(tmp_path, owners_only=True, queue_nowait=True)
    assert lock.calls == [False, False] and lock.held


@pytest.fixture
def continuing_switch(isolated_runtime, reader, monkeypatch):  # noqa: F811
    from ouroboros import store, store_switch

    root = isolated_runtime.resolve()
    confirm_first_use(root)
    import_all(root)
    dialogue_store.append(root, "chat", {"direction": "in", "text": "synthetic transfer", "chat_id": 1})
    reader.bind(root)
    monkeypatch.setattr(store, "_HANDED_OFF", {"synthetic mark"})
    monkeypatch.setattr(store_switch, "switch_needs", lambda *_: {})
    monkeypatch.setattr(store_switch, "lacks_wake_chain_reader", lambda *_: False)
    endpoint = make_chat_history_endpoint(root)

    def switch(reset, **kwargs):
        return store_switch.switch_or_reset(root, root, target_sha="1" * 40, target_ref="fixture", reason="test",
                                          door="ui_rollback", reset=reset, **kwargs)

    async def get_async():
        response = await endpoint(SimpleNamespace(query_params={}))
        return response.status_code, response.body

    return SimpleNamespace(root=root, store=store, switch=switch, get_async=get_async,
                           get=lambda: asyncio.run(get_async()))


@pytest.mark.parametrize("failure", ["busy", "sqlite_error"])
def test_handoff_refusal_does_not_probe_until_an_actual_read_and_retries(continuing_switch, monkeypatch, failure):
    import sqlite3

    world = continuing_switch
    expected = world.get()
    probes = []
    ready = world.store.store_ready

    def published(root):
        probes.append(root)
        if len(probes) == 1:
            raise world.store.StoreBusy("synthetic publication busy")
        return ready(root)

    def refuse(*args):
        if failure == "sqlite_error":
            raise sqlite3.OperationalError("synthetic journal mode change busy")
        raise world.store.StoreBusy("synthetic handoff busy")

    monkeypatch.setattr(world.store, "store_ready", published)
    monkeypatch.setattr(world.store, "hand_off", refuse)
    ok, reason, deferred = world.switch(lambda: pytest.fail("Git must not start"))
    assert not ok and not deferred and "busy" in reason
    assert not probes and world.store._HANDED_OFF == {"synthetic mark"}
    assert world.get()[0] == 503 and world.store._HANDED_OFF == {"synthetic mark"}
    assert world.get() == world.get() == expected
    assert probes == [world.root, world.root] and not world.store._HANDED_OFF


def test_second_same_door_pre_effect_refusal_cannot_release_unknown_transfer(continuing_switch, monkeypatch):
    from ouroboros import store_switch

    world = continuing_switch
    monkeypatch.setattr(store_switch, "_hand_off_stores", lambda _root: "")
    first = store_switch.HistoryHandoff()
    assert world.switch(lambda: (False, "possible Git effects"), handoff=first) == (False, "possible Git effects", False)
    second = store_switch.HistoryHandoff()
    assert not world.switch(lambda: store_switch.UnappliedCheckout(False, "Git index busy"), handoff=second)[0]
    second.resume_inline()  # a late answer for the same door is still this attempt's
    monkeypatch.setattr(world.store, "store_ready", lambda _root: pytest.fail("unknown transfer authorizes no probe"))
    assert world.get()[0] == 503 and world.store._HANDED_OFF == {"synthetic mark"}


def test_delayed_publication_read_cannot_clear_a_new_handoff(continuing_switch, reader, monkeypatch):
    from ouroboros import store_switch

    world = continuing_switch
    expected = world.get()
    monkeypatch.setattr(store_switch, "_hand_off_stores", lambda _root: "")
    first = store_switch.HistoryHandoff()
    assert world.switch(lambda: (True, "validated same runtime"), handoff=first)[0]
    first.resume_inline()
    entered, let_finish = threading.Event(), threading.Event()
    ready = world.store.store_ready

    def rendezvous(root):
        entered.set()
        assert let_finish.wait(4)
        return ready(root)

    async def race():
        monkeypatch.setattr(world.store, "store_ready", rendezvous)
        request = asyncio.create_task(world.get_async())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            monkeypatch.setattr(history_reader, "_HANDOFF_SECONDS", 0)
            ok, message, _ = await asyncio.to_thread(world.switch, lambda: pytest.fail("an active lease blocks Git"))
            assert not ok and "read still in use" in message
        finally:
            let_finish.set()
        assert (await request)[0] == 503 and world.store._HANDED_OFF == {"synthetic mark"}
        assert await world.get_async() == expected
        assert not world.store._HANDED_OFF and not reader.live_children()

    asyncio.run(race())


@pytest.mark.parametrize("outcome", ["declined", "accepted", "rolled_back", "rollback_unknown", "deferred"])
def test_replace_resumes_only_its_own_confirmed_continuation(continuing_switch, monkeypatch, outcome):
    from ouroboros import store_switch
    from ouroboros.gateway import control
    from supervisor import git_ops, update_merge

    world = continuing_switch
    expected = world.get()
    state, restarts = {}, []
    plan = {"available": True, "kind": "conflicting", "base_sha": "before", "target_sha": "after"}
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", world.root)
    monkeypatch.setattr(git_ops, "_write_update_intent", lambda _arg: None)
    monkeypatch.setattr(git_ops, "prepare_managed_update", lambda *a, **kw: (True, {"update_intent": {}}))
    monkeypatch.setattr(git_ops, "checkout_and_reset", lambda *a, **kw: (True, "updated"))
    monkeypatch.setattr(store_switch, "switch_needs", lambda *_: {"facts": outcome == "deferred"})
    monkeypatch.setattr(store_switch, "_hand_off_stores", lambda _root: "")
    monkeypatch.setattr(update_merge, "plan_managed_update_merge", lambda **kw: plan)
    monkeypatch.setattr(update_merge, "acquire_update_lock", object)
    monkeypatch.setattr(update_merge, "release_update_lock", lambda _lock: None)
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: state)
    monkeypatch.setattr(update_merge, "write_update_tx", lambda value: state.update(value))
    monkeypatch.setattr(update_merge, "update_restart_smoke", lambda: {"ok": not outcome.startswith("roll")})
    monkeypatch.setattr(update_merge, "rollback_managed_update", lambda _why: (outcome == "rolled_back", "restored"))
    monkeypatch.setattr(update_merge, "mark_update_tx_gate_blocked", lambda *a: None)
    monkeypatch.setattr(control, "_index_lock_precondition", lambda: None)
    monkeypatch.setattr(control, "_quiesce_repo_writers", lambda _why: [])
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: None)
    monkeypatch.setattr(control, "_request_restart", lambda _req: restarts.append(True) or outcome == "accepted")
    response = control._apply_replace_recovery_fenced(None, expected_base_sha="before", expected_target_sha="after")
    if outcome in {"accepted", "rollback_unknown"}:
        assert response.status_code == (200 if outcome == "accepted" else 500)
        assert world.get()[0] == 503 and world.store._HANDED_OFF
    else:
        assert world.get() == expected
        assert bool(world.store._HANDED_OFF) is (outcome == "deferred")
    assert bool(restarts) is not outcome.startswith("roll")



@pytest.mark.parametrize("outcome", ["confirmed", "not_applied", "unknown", "exit_follows"])
def test_boot_smoke_rollback_pairs_history_with_its_actual_exit(continuing_switch, monkeypatch, outcome):
    import server
    from ouroboros import store_switch
    from supervisor import git_ops, update_merge

    world = continuing_switch
    expected = world.get()
    tx = {"pre_update_sha": "1" * 40}
    exits = []
    results = {"confirmed": (True, "synthetic restored checkout"),
               "not_applied": store_switch.UnappliedCheckout(False, "synthetic pre-effect refusal"),
               "unknown": (False, "synthetic unknown effects"), "exit_follows": (True, "synthetic restored checkout")}

    def finalizer(**_kw):
        ok, _msg, facts = update_merge._boot_smoke_rollback(tx, "fixture")
        return {"rolled_back": ok, **facts}

    monkeypatch.setattr(server, "DATA_DIR", world.root)
    monkeypatch.setattr(server, "_wait_for_supervisor_update_finalize", lambda: True)
    monkeypatch.setattr(server.body_adoption, "settle_on_boot", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_request_restart_exit", lambda: exits.append(True))
    monkeypatch.setattr(server, "broadcast_ws_sync", lambda _row: None)
    from supervisor import workers
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda _why: None)
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: tx if outcome == "exit_follows" else {})
    monkeypatch.setattr(update_merge, "finalize_managed_update_on_boot", finalizer)
    monkeypatch.setattr(update_merge, "rollback_managed_update_on_boot", lambda _why: results[outcome])
    monkeypatch.setattr(update_merge._g, "DRIVE_ROOT", world.root)
    monkeypatch.setattr(update_merge._g, "REPO_DIR", world.root)
    monkeypatch.setattr(git_ops, "advance_recovery_point", lambda _ready: None)
    monkeypatch.setattr(git_ops, "compute_managed_update_status", lambda **_kw: {})
    monkeypatch.setattr(store_switch, "_hand_off_stores", lambda _root: "")
    server._boot_managed_update_tasks()
    assert exits == ([True] if outcome == "exit_follows" else [])
    if outcome in {"unknown", "exit_follows"}:
        assert world.get()[0] == 503 and world.store._HANDED_OFF
    else:
        assert world.get() == expected and not world.store._HANDED_OFF


@pytest.mark.parametrize("outcome", ["known", "unknown", "binding_raised"])
def test_adoption_rejected_after_restart_checkout_settles_only_confirmed_handoff(continuing_switch, monkeypatch, outcome):
    import pathlib

    from ouroboros import body_adoption, server_restart, store_switch

    world = continuing_switch
    expected = world.get()
    abandoned = []
    reason = "test adoption"
    receipt = world.root / "state" / "pending_restart_verify.json"
    receipt.write_text(json.dumps({"reason": reason, "expected_sha": "new"}), encoding="utf-8")
    known = outcome != "unknown"

    def read_adoption(_root):
        if outcome == "binding_raised":
            raise OSError("could not read the adoption binding")
        return {"phase": "authorized", "reason": reason, "cand": "new", "old": "before", "branch": "ouroboros",
                "repo_dir": str(world.root)}

    monkeypatch.setattr(body_adoption, "read", read_adoption)
    monkeypatch.setattr(body_adoption, "_git", lambda *_a, **kw: SimpleNamespace(stdout="verified recovery point"))
    monkeypatch.setattr(body_adoption, "abandon", lambda _root, why: abandoned.append(why))
    monkeypatch.setattr(store_switch, "_hand_off_stores", lambda _root: "")
    monkeypatch.setattr(server_restart, "release_agent_restart", lambda *_a: None)
    from supervisor import evolution_lifecycle
    monkeypatch.setattr(evolution_lifecycle, "disable_evolution_projection", lambda: None)
    monkeypatch.setattr(evolution_lifecycle, "pause_evolution_campaign", lambda _why: None)

    def safe_restart(**_kw):
        ok, message, _ = world.switch(lambda: (known, "synthetic stable fallback"))
        return ok, message

    server = SimpleNamespace(read_json_dict=lambda _path: {}, pathlib=pathlib, body_adoption=body_adoption,
                             _safe_restart_serialized=lambda fn, **kw: fn(**kw),
                             _request_restart_exit=lambda: pytest.fail("unarmed adoption must not exit"))
    ctx = SimpleNamespace(DRIVE_ROOT=world.root, load_state=dict, safe_restart=safe_restart)
    if outcome == "binding_raised":
        with pytest.raises(OSError, match="adoption binding"):
            server_restart._perform_supervisor_restart(server, ctx, restart_reason=reason)
    else:
        server_restart._perform_supervisor_restart(server, ctx, restart_reason=reason)
        [why] = abandoned
        assert ("serving checkout moved" if known else "synthetic stable fallback") in why
    if known:
        assert world.get() == expected and not world.store._HANDED_OFF
    else:
        assert world.get()[0] == 503 and world.store._HANDED_OFF


def test_reader_relays_only_fixed_translation_and_cost_custody_diagnostics(caplog, capsys):
    import logging

    relay = history_reader._ReaderDiagnostics()
    for logger, func in [("ouroboros.i18n_memory", "cached_memory"),
                         ("ouroboros.delegate_custody_usage", "cost_snapshot"),
                         ("ouroboros.unrelated", "cached_memory")]:
        relay.handle(logging.LogRecord(logger, logging.WARNING, "synthetic-private-path", 1,
                                      "synthetic-private-text: %s", ("synthetic-private-id",), None, func))
    stderr = capsys.readouterr().err
    assert stderr == "Malformed translation memory ignored\nCost custody unavailable\n"
    history_reader._log_stderr(io.BytesIO(b"synthetic-private-output\n" + stderr.encode()))
    assert "Malformed translation memory ignored" in caplog.text and "Cost custody unavailable" in caplog.text
    assert "synthetic-private" not in caplog.text
