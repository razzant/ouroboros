"""The route completion frame carries the same relay aggregate after body delivery."""
import asyncio
import json
from pathlib import Path

import pytest

from ouroboros import extension_process_runner as runner
from ouroboros import extension_loader
from ouroboros.extension_route_stream import RouteStreamResponse
from ouroboros.tools.process_facts import consume_last_process_facts
from tests._extension_loader_shared import _prepare_extension
from tests.test_extension_ws_diagnostics import child_env, relay_endpoint, relay_state  # noqa: F401


@pytest.mark.parametrize(("status", "reason"), [
    (202, None), (429, "rate_limited"), (500, "http_server_error"),
])
def test_real_route_completion_discloses_background_relay_failures(tmp_path, status, reason, caplog):
    from ouroboros.tools import shell_process

    # Ended children an earlier test left tracked (an idle or stop probe retires them) are not this route's.
    already_tracked = set(shell_process._active_subprocesses)
    # Sending after the HTTP body proves that the final X carries the diagnostic;
    # an earlier result snapshot would miss every failure in this fixture.
    plugin = '''from starlette.background import BackgroundTask
from starlette.responses import Response
def register(api):
    def stream(request):
        def after():
            for index in range(5):
                assert api.send_ws_message('progress', {'index': index}) is None
        return Response(b'already delivered', background=BackgroundTask(after))
    api.register_route('stream', stream)
'''
    skill, skills_repo, drive = _prepare_extension(tmp_path, "ws_stream_diagnostic", plugin,
                                                  permissions=["route", "ws_handler"])
    repo = Path(__file__).resolve().parents[1]
    env = child_env(tmp_path, drive, repo)
    assert extension_loader.load_extension(skill, lambda: {}, drive_root=drive,
                                           repo_path=str(skills_repo)) is None
    spec = extension_loader.list_routes()[f"/api/extensions/{skill.name}/stream"]
    with relay_endpoint(status) as (bridge, requests):
        env.update(bridge)

        def child_factory():
            return runner._child_process(
                {"mode": "route", "skill_name": skill.name, "surface": spec["path"],
                 "request": {"method": "GET", "path": spec["path"]},
                 "drive_root": str(drive), "repo_dir": str(repo), "skills_repo_path": str(skills_repo)},
                skill_dir=skill.skill_dir, drive_root=drive, repo_dir=repo, env=env, stream=True)

        response = RouteStreamResponse(spec, child_factory)
        events = []

        async def run():
            delivered = asyncio.Event()

            async def receive():
                await delivered.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                events.append(message)
                if message["type"] == "http.response.body" and not message.get("more_body"):
                    delivered.set()

            await response({"type": "http", "method": "GET"}, receive, send)

        asyncio.run(run())
    assert events[0]["status"] == 200, events
    assert b"".join(row.get("body", b"") for row in events) == b"already delivered"
    assert response.body_complete is True
    facts = consume_last_process_facts()
    assert facts["exit_code"] == 0
    if reason:
        assert facts["ws_relay_failures"] == response.ws_relay_failures == {reason: 5}
        assert len([row for row in caplog.records if "WS relay failures" in row.getMessage()]) == 1
    else:
        assert response.ws_relay_failures is None and "ws_relay_failures" not in facts
        assert not [row for row in caplog.records if "WS relay failures" in row.getMessage()]
    assert len(requests) == (5 if isinstance(status, int) else 0)
    assert not list((drive / "state" / "skills" / skill.name / "extension_calls").glob("*.result.json"))
    assert not shell_process._active_subprocesses - already_tracked


def test_legacy_empty_completion_frame_remains_compatible():
    response = RouteStreamResponse({}, None)
    frames = asyncio.Queue()
    frames.put_nowait((b"S", json.dumps({"status": 200, "headers": []}).encode()))
    frames.put_nowait((b"B", b"\x00complete"))
    frames.put_nowait((b"X", b""))
    events = []

    async def send(message):
        events.append(message)

    asyncio.run(response._pump(frames, send))
    assert events[-1]["body"] == b"complete"
    assert response.ws_relay_failures is None
