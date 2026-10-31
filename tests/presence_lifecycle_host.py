"""Child-only TCP fixture; never point its root or control routes at an installation.

The author parks through the kept trigger (``tests/test_presence_continuation.py``):
it reviews its own result with ``review_change(subject="result")``, the held
reviewer writes an addressed message to its mailbox, the author leaves only its
own wait, and its next plain prose parks on that review.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import socket
import sys
import threading


def shutdown_fixture_processes(root, generation):
    """Join this private interpreter's event bus and multiprocessing support."""
    from multiprocessing import active_children, forkserver, resource_tracker
    from supervisor.workers import shutdown_event_q

    children = [child.pid for child in active_children()]
    tracker = resource_tracker._resource_tracker
    tracker_pid = tracker._pid
    server = forkserver._forkserver
    server_pid = server._forkserver_pid
    # Cleanup cases explicitly attach the real supervisor event bus before
    # accepting work. This small Host fixture has no server lifespan to close
    # that owned bus for it; producers never create one while delivering.
    shutdown_event_q()
    remaining = [child.pid for child in active_children()]
    assert not remaining, f"fixture multiprocessing children remain: {remaining}"
    # Linux's forkserver is not an active_children() member. Join the support
    # process while this interpreter owns its pipe/PID, before parent exit can
    # race the strict private-process-group assertion in the TCP consumer.
    server._stop()
    # CPython normally lets the tracker exit on interpreter pipe EOF, AFTER the
    # parent is reaped. Join it here (as CPython's own test cleanup does), while
    # this fixture still owns it. Never kill/ignore arbitrary group members.
    tracker._stop()
    (root / f"process-cleanup-{generation}.json").write_text(json.dumps({
        "event_bus_children": children, "remaining_children": remaining,
        "resource_tracker_pid": tracker_pid, "resource_tracker_joined": tracker._pid is None,
        "forkserver_pid": server_pid, "forkserver_joined": server._forkserver_pid is None,
    }))


def seed(data):
    from ouroboros.presence_bindings import PresenceBinding, PresenceEndpoint
    from tests.frozen_presence_bindings import save_presence_binding
    from tests.frozen_presence_capabilities import (
        PresenceSelection, PresenceState, PresenceToolTarget, presence_state_fingerprint, save_presence_state)
    from tests.frozen_presence_profile import parse_presence_profile, presence_request_fingerprint
    from ouroboros.skill_loader import load_skill, save_enabled, save_review_state, SkillReviewState
    from tests.test_host_service_api import _seed_token
    from tests.test_presence_continuation_process import TOKEN

    marker = data / "fixture-binding.json"
    if marker.exists():
        return json.loads(marker.read_text())["binding"]
    _seed_token(data, skill="telegram-bot", token=TOKEN,
                permissions=["presence"], manifest_permissions=["presence"])
    folder = data / "skills" / "external" / "lifecycle-helper"
    folder.mkdir(parents=True)
    names = ["chat_history", "review_change", "get_task_result"]
    (folder / "SKILL.md").write_text(
        "---\nname: lifecycle-helper\ndescription: Isolated report helper.\nversion: 0.1\n"
        "type: instruction\npresence:\n  instructions: Prepare reports and explicitly review substantive results.\n"
        "  capability_requests:\n" + "".join(
            f"    - id: {name}\n      kind: tool\n      required: true\n      purpose: Isolated test.\n"
            for name in names) + "---\n# Isolated helper\n")
    loaded = load_skill(folder, data)
    save_enabled(data, loaded.name, True)
    save_review_state(data, loaded.name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    profile = parse_presence_profile(loaded.manifest, folder)
    state = PresenceState(tuple(PresenceSelection(presence_request_fingerprint(request),
                                                 PresenceToolTarget("builtin", name))
                                for request, name in zip(profile.capability_requests, names)))
    save_presence_state(data, loaded.name, state, expected_state_fingerprint=presence_state_fingerprint(PresenceState()))
    binding = save_presence_binding(data, PresenceBinding(
        "b" * 32, "telegram-bot", loaded.name, PresenceEndpoint("telegram", "bot-1", "room-1", "topic-1"),
        PresenceEndpoint("telegram", "bot-1", "room-1", "topic-1"))).binding_id
    marker.write_text(json.dumps({"binding": binding}))
    return binding


def _last_review_left(messages):
    """The latest tool result is a review its author left for an addressed message."""
    from tests.test_presence_continuation import LEFT

    tools = [row for row in messages if row.get("role") == "tool"]
    return bool(tools) and LEFT in str(tools[-1].get("content"))


def main(root, generation, mode, scenario="incoming", with_event_bus=False):
    import pytest
    import uvicorn
    from starlette.background import BackgroundTask
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from ouroboros import loop
    from ouroboros.gateway.host_service import create_host_service_app
    from ouroboros.presence_runner import run_presence_turn
    from ouroboros.task_status import reconcile_orphaned_running_tasks
    from supervisor import queue as _task_queue  # noqa: F401 -- normal supervisor import order for Panic
    from tests.test_presence_continuation import review, waits
    from tests.test_presence_continuation_bootstrap import install_bootstrap_harness, finish, tool_call
    from tests.test_presence_continuation_process import ROOT, ANSWER, MANUAL, LATE_CHILD, REVISED_ANSWER, TOOL_NOTE

    assert Path(os.environ["OUROBOROS_DATA_DIR"]).resolve() == (root / "data").resolve()
    assert Path(os.environ["HOME"]).resolve() == (root / "home").resolve()
    patch = pytest.MonkeyPatch()
    h = install_bootstrap_harness(patch, root / "data", repo=ROOT, enforcement=mode)
    patch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    patch.setattr("supervisor.consciousness_carriers.autonomous_work_enabled", lambda *_: True, raising=False)
    patch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {
        "status": "available", "remaining_usd": 100})
    patch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, ""))
    event_bus_children = []
    if with_event_bus:
        from multiprocessing import active_children
        from supervisor.workers import get_event_q

        bus = get_event_q()
        bus.put({"fixture_bus_ready": True})
        assert bus.get(timeout=5) == {"fixture_bus_ready": True}
        event_bus_children = [child.pid for child in active_children()]
    patch.setenv("OUROBOROS_PRESENCE_MAX_ACTIVE", "1")
    binding = seed(h.data)
    reconciled = reconcile_orphaned_running_tasks(h.data)
    inner_inference = loop.call_llm_with_retry
    inference_release = threading.Event()
    close_release = threading.Event()
    if scenario == "late_child_close_barrier":
        from ouroboros.model_wait import TaskModelWait
        from ouroboros.task_results import load_task_result
        from ouroboros.utils import atomic_write_json

        scenario = "late_child"
        original_close = TaskModelWait.close

        def close_after_release(waiter):
            if any(agent.tools._ctx.model_wait_context is waiter for agent in h.agents):
                terminal = load_task_result(h.data, waiter.task_id)
                assert terminal["status"] in {"completed", "cancelled", "failed"}
                # Published whole (temporary file, then rename): the test polls this marker with a strict read.
                atomic_write_json(root / "before-author-close.json", {
                    "task_id": waiter.task_id, "status": terminal["status"], "closed": waiter.closed})
                assert close_release.wait(30), "test failed to release author teardown"
            return original_close(waiter)

        patch.setattr(TaskModelWait, "close", close_after_release)

    def inference(llm, messages, *args, **kwargs):
        tools = args[1] if len(args) > 1 else []
        with (root / "model-inputs.jsonl").open("a") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "messages": messages,
                                     "tools": [tool["function"]["name"] for tool in tools]}) + "\n")
        if scenario == "disconnect" and h.calls == 0:
            (root / "before-first-inference.json").write_text(json.dumps({"pid": os.getpid()}))
            assert inference_release.wait(30), "test failed to release first inference"
        return inner_inference(llm, messages, *args, **kwargs)

    patch.setattr(loop, "call_llm_with_retry", inference)

    def script(messages):
        if generation > 1:
            assert MANUAL in str(messages), "restart attempted inference without a fresh manual event"
            return finish("manual", message="Continued after the explicit request.")
        if h.calls == 1:
            return review(answer=ANSWER)  # the author reviews its own result and waits for it
        if h.calls in (2, 4) and _last_review_left(messages):
            return waits()  # it left that review for the addressed message: plain prose parks on it
        if scenario == "late_child" and h.calls == 3:
            from ouroboros.task_results import write_task_result

            # The original reviewer has settled before the same author resumes.
            # Hold a new, distinct review; never rely on a timing window between
            # releasing the first reviewer and parking for the second.
            assert h.settled.wait(60)
            h.release.clear()
            h.entered.clear()
            h.settled.clear()
            ctx = h.agents[0].tools._ctx
            # Promotion admission/dispatch is outside this transport fixture.
            # Retain its canonical scheduled child and source binding, exactly
            # as a successful independent-work handoff leaves them for polling.
            write_task_result(h.data, LATE_CHILD, "scheduled", delegation_role="root",
                              root_task_id=LATE_CHILD, description="Compile the independent report",
                              metadata={"presence": dict(ctx.task_metadata["presence"])})
            ctx._swarm_handoff_attempt = {"status": "scheduled", "task_id": LATE_CHILD}
            return review(answer=REVISED_ANSWER)
        if scenario == "tool_silent":
            # Its tool delivered the result: the mind chooses silence; a receipt never chooses for it.
            return {"content": None, "tool_calls": [tool_call("finish_task", {
                "action": "finish", "silent": True, "answer": TOOL_NOTE}, "select")]}
        return finish("select", message=REVISED_ANSWER if scenario == "late_child" else ANSWER)

    h.script = script

    def runner(**kwargs):
        return run_presence_turn(repo_dir=ROOT, drive_root=h.data, agent_factory=h.factory, **kwargs)

    # The proactive tool calls the module owner directly; keep its actual execution
    # manager and admission while observing the same production agent constructor.
    if scenario == "proactive":
        import functools
        patch.setattr("ouroboros.presence_runner.run_presence_turn",
                      functools.partial(run_presence_turn, agent_factory=h.factory))

    app = create_host_service_app(h.data, presence_runner=runner)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    patch.setenv("OUROBOROS_HOST_SERVICE_PORT", str(port))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))

    async def stop(request):
        from ouroboros.cancel_intents import request_cancel, STOP_POLICY_IMMEDIATE
        body = await request.json()
        request_cancel(h.data, body["task_id"], source="isolated-owner", reason="Stop parked author",
                       requested_stop_policy=STOP_POLICY_IMMEDIATE)
        return JSONResponse({"requested": True})

    async def panic(_request):
        from ouroboros.server_control import execute_panic_stop, PanicIngress
        ingress = PanicIngress(lambda: execute_panic_stop(None, lambda: None, data_dir=h.data,
            panic_exit_code=73, log=logging.getLogger("isolated-panic")))
        assert ingress.request("/panic", source="web")
        return JSONResponse({"requested": True})

    async def live(_request):
        return JSONResponse([agent.tools._ctx.task_id for agent in h.agents
                             if getattr(agent.tools._ctx, "model_wait_context", None) is not None
                             and not agent.tools._ctx.model_wait_context.closed])

    async def shutdown(_request):
        inference_release.set()
        close_release.set()
        h.release.set()
        server.should_exit = True
        return JSONResponse({"stopping": True})

    async def restart(_request):
        from ouroboros.server_control import restart_current_process

        # The real direct-server transfer replaces this image while the author
        # and review are still parked. Do not release either before exec.
        patch.setenv("OUROBOROS_SERVER_REEXEC_ARGV_JSON", json.dumps([
            "-m", "tests.presence_lifecycle_host", str(root), str(generation + 1), mode, scenario,
            *(["--fixture-event-bus"] if with_event_bus else [])]))
        return JSONResponse({"restarting": True}, background=BackgroundTask(
            restart_current_process, "127.0.0.1", port, repo_dir=ROOT,
            log=logging.getLogger("isolated-restart"), owner_initiated=True))

    async def release_inference(_request):
        inference_release.set()
        return JSONResponse({"released": True})

    async def release_review(_request):
        h.release.set()
        return JSONResponse({"released": True})

    async def release_close(_request):
        close_release.set()
        return JSONResponse({"released": True})

    async def review_state(_request):
        return JSONResponse({"reviews": len(h.reviews), "model_calls": h.calls,
                             "held": h.entered.is_set() and not h.release.is_set()})

    def initiate():
        from tests.presence_legacy_helpers import _initiate_presence
        from ouroboros.tools.registry import ToolContext
        # Same stable caller/dedupe identity across process recycle.
        return json.loads(_initiate_presence(
            ToolContext(repo_dir=ROOT, drive_root=h.data, task_id="isolated-proactive-caller"),
            binding, "Prepare the report.", "same-proactive-request"))

    async def proactive(_request):
        import asyncio
        return JSONResponse(await asyncio.to_thread(initiate))

    app.routes.extend([Route("/fixture/stop", stop, methods=["POST"]),
                       Route("/fixture/panic", panic, methods=["POST"]),
                       Route("/fixture/release-inference", release_inference, methods=["POST"]),
                       Route("/fixture/release-review", release_review, methods=["POST"]),
                       Route("/fixture/release-close", release_close, methods=["POST"]),
                       Route("/fixture/review-state", review_state),
                       Route("/fixture/proactive", proactive, methods=["POST"]),
                       Route("/fixture/restart", restart, methods=["POST"]),
                       Route("/fixture/live", live), Route("/fixture/shutdown", shutdown, methods=["POST"])])
    from ouroboros.review_operation import controller_identity
    from ouroboros.utils import atomic_write_json
    atomic_write_json(root / f"ready-{generation}.json", {
        "url": f"http://127.0.0.1:{port}", "binding": binding, "pid": os.getpid(),
        "controller": controller_identity(), "reconciled": reconciled,
        "event_bus_children": event_bus_children})
    try:
        server.run(sockets=[sock])
    finally:
        inference_release.set()
        close_release.set()
        h.release.set()
        if h.entered.is_set():
            assert h.settled.wait(60)
        shutdown_fixture_processes(root, generation)
        sock.close()
        print(json.dumps({"threads": [thread.name for thread in threading.enumerate()],
                          "model_calls": h.calls, "review_calls": len(h.reviews)}), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "incoming",
         with_event_bus="--fixture-event-bus" in sys.argv[5:])
