"""`browse_page` on a remote placement: the forward, the rewrite, the origin block.

Before this, `tools/browser.py` knew nothing about placement. A remote task asking
for `http://localhost:5173` opened HOME's own loopback: a page rendered, nothing in
the result said whose machine it came from, and the transport that could have made
the target's port reachable (`remote_browser_forward.SSHBrowserForwardManager`) had
no caller outside the broker. That is the same silent-wrong-host class the root
matrix closed for file reads, and worse here, because a screenshot of the wrong
service looks exactly like a screenshot of the right one.

No Playwright and no real `ssh` here: the forward is opened through the broker
facade, so a fake service is the honest seam, and the origin block is a pure
predicate over a URL. The real `ssh -L` child, its custody registration and its
panic teardown are covered by `tests/test_remote_browser_forward.py` (serial lane).
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.tools import browser as browser_tools
from ouroboros.tools.browser import (
    _remote_foreign_origin_blocked,
    _resolve_placement_url,
)
from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY, SshWorkspaceRef

_LOCAL_PORT = 41731
_ORIGIN = f"http://127.0.0.1:{_LOCAL_PORT}"


class _FakeBrowserState:
    """Only what the forward map needs; the real one also holds Playwright handles."""


def _remote_ctx(task_id: str = "task-1"):
    ref = SshWorkspaceRef(
        connection_id="conn-1",
        remote_root="/srv/app",
        workspace_id="ws-1",
    )
    return SimpleNamespace(
        task_id=task_id,
        task_metadata={SEALED_WORKSPACE_REF_KEY: ref.to_payload()},
        browser_state=_FakeBrowserState(),
        workspace_root="",
    )


def _local_ctx(workspace_root: str = "/home/me/project"):
    return SimpleNamespace(
        task_id="task-1",
        task_metadata={},
        browser_state=_FakeBrowserState(),
        workspace_root=workspace_root,
    )


class _FakeService:
    """Records what the broker facade was asked for; hands back a forward record."""

    def __init__(self, *, local_port: int = _LOCAL_PORT, error: Exception | None = None):
        self.calls: list[dict] = []
        self._local_port = local_port
        self._error = error

    def open_browser_forward(self, workspace_ref, *, remote_port, task_id):
        self.calls.append(
            {"workspace_ref": dict(workspace_ref), "remote_port": remote_port, "task_id": task_id}
        )
        if self._error is not None:
            raise self._error
        return {
            "forward_id": "fwd-1",
            "connection_id": "conn-1",
            "task_id": task_id,
            "remote_port": remote_port,
            "local_port": self._local_port,
            "url": f"http://127.0.0.1:{self._local_port}/",
            "origin": f"http://127.0.0.1:{self._local_port}",
            "task_token": "t",
            "config_sha256": "d",
        }


@pytest.fixture
def broker(monkeypatch):
    service = _FakeService()
    monkeypatch.setattr(
        "ouroboros.remote_workspace.get_remote_workspace_service",
        lambda: service,
    )
    return service


# ── the forward and the rewrite ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "asked,expected_port,expected",
    [
        ("http://localhost:5173", 5173, f"{_ORIGIN}/"),
        ("http://127.0.0.1:5173/", 5173, f"{_ORIGIN}/"),
        ("http://localhost:3000/app/page?q=1#frag", 3000, f"{_ORIGIN}/app/page?q=1#frag"),
        # No explicit port is port 80 / 443 on the TARGET's loopback, not on Home's.
        ("http://localhost/", 80, f"{_ORIGIN}/"),
        ("https://localhost/secure", 443, f"{_ORIGIN}/secure"),
        ("http://[::1]:8080/x", 8080, f"{_ORIGIN}/x"),
    ],
)
def test_a_loopback_url_is_forwarded_to_the_target_and_rewritten(
    broker, asked, expected_port, expected
):
    ctx = _remote_ctx()
    url, refusal = _resolve_placement_url(ctx, asked)
    assert refusal == ""
    assert url == expected
    assert len(broker.calls) == 1
    call = broker.calls[0]
    assert call["remote_port"] == expected_port
    assert call["task_id"] == "task-1"
    # Addressed by the SEALED placement descriptor, not by anything re-derived.
    assert call["workspace_ref"]["kind"] == "ssh"
    assert call["workspace_ref"]["connection_id"] == "conn-1"
    assert call["workspace_ref"]["remote_root"] == "/srv/app"


def test_one_forward_per_port_is_reused_across_calls(broker):
    ctx = _remote_ctx()
    first, _ = _resolve_placement_url(ctx, "http://localhost:5173/a")
    second, _ = _resolve_placement_url(ctx, "http://localhost:5173/b")
    third, _ = _resolve_placement_url(ctx, "http://localhost:9999/c")
    assert first.startswith(_ORIGIN) and second.startswith(_ORIGIN)
    assert second.endswith("/b") and third.endswith("/c")
    assert [call["remote_port"] for call in broker.calls] == [5173, 9999]


def test_the_forward_map_survives_a_browser_rebuild(broker):
    """`cleanup_browser` must not drop the map: the forward belongs to the TASK.

    It runs mid-task on a thread switch or an engine change. Clearing the map there
    would leak one `ssh -L` child per rebuild and then hand the model a dead origin.
    """

    ctx = _remote_ctx()
    _resolve_placement_url(ctx, "http://localhost:5173/")
    browser_tools.cleanup_browser(ctx)
    again, refusal = _resolve_placement_url(ctx, "http://localhost:5173/")
    assert refusal == ""
    assert again == f"{_ORIGIN}/"
    assert len(broker.calls) == 1


# ── a local placement is byte-identical to before ───────────────────────────


@pytest.mark.parametrize(
    "asked",
    [
        "http://localhost:5173/",
        "http://127.0.0.1:3000/app",
        "https://example.com/docs",
        "http://192.168.1.10:8080/",
        "file:///tmp/whatever/index.html",
        "http://localhost/",
    ],
)
def test_a_local_placement_is_untouched_and_never_calls_the_broker(monkeypatch, asked):
    called = []
    monkeypatch.setattr(
        "ouroboros.remote_workspace.get_remote_workspace_service",
        lambda: called.append(1),
    )
    url, refusal = _resolve_placement_url(_local_ctx(), asked)
    assert (url, refusal) == (asked, "")
    assert called == []


def test_a_context_with_no_sealed_placement_is_local_not_remote(monkeypatch):
    """A lightweight ctx must never be read as remote by accident."""

    monkeypatch.setattr(
        "ouroboros.remote_workspace.get_remote_workspace_service",
        lambda: pytest.fail("a placement-less context reached the broker"),
    )
    bare = SimpleNamespace()
    assert _resolve_placement_url(bare, "http://localhost:5173/") == (
        "http://localhost:5173/",
        "",
    )
    assert _remote_foreign_origin_blocked("http://127.0.0.1:8765/api", bare) is False


# ── typed refusals, never a fall back to Home's own port ────────────────────


def test_no_transport_in_this_process_is_a_typed_refusal(monkeypatch):
    from ouroboros.workspace_diagnostics import RemoteWorkspaceError

    def _unavailable():
        raise RemoteWorkspaceError(
            "remote_workspace_unavailable",
            "Remote workspace broker is not configured.",
            phase="connect",
        )

    monkeypatch.setattr(
        "ouroboros.remote_workspace.get_remote_workspace_service", _unavailable
    )
    url, refusal = _resolve_placement_url(_remote_ctx(), "http://localhost:5173/")
    assert url == ""
    assert refusal.startswith("⚠️ BROWSER_REMOTE_FORWARD_UNAVAILABLE")
    # The refusal must SAY what the alternative would have been, because "it worked
    # but showed the wrong machine" is the failure this replaces.
    assert "5173" in refusal


def test_a_transport_failure_is_typed_and_names_the_port(monkeypatch):
    service = _FakeService(error=RuntimeError("ssh: connect to host port 22: refused"))
    monkeypatch.setattr(
        "ouroboros.remote_workspace.get_remote_workspace_service", lambda: service
    )
    url, refusal = _resolve_placement_url(_remote_ctx(), "http://localhost:5173/")
    assert url == ""
    assert refusal.startswith("⚠️ BROWSER_REMOTE_FORWARD_FAILED")
    assert "5173" in refusal


def test_a_forward_without_a_local_origin_is_a_refusal(monkeypatch):
    class _Empty(_FakeService):
        def open_browser_forward(self, workspace_ref, *, remote_port, task_id):
            super().open_browser_forward(workspace_ref, remote_port=remote_port, task_id=task_id)
            return {"forward_id": "fwd", "origin": ""}

    monkeypatch.setattr(
        "ouroboros.remote_workspace.get_remote_workspace_service", lambda: _Empty()
    )
    url, refusal = _resolve_placement_url(_remote_ctx(), "http://localhost:5173/")
    assert url == ""
    assert "BROWSER_REMOTE_FORWARD_FAILED" in refusal


def test_a_call_with_no_task_id_cannot_own_a_forward(monkeypatch):
    monkeypatch.setattr(
        "ouroboros.remote_workspace.get_remote_workspace_service",
        lambda: pytest.fail("a task-less call reached the broker"),
    )
    ctx = _remote_ctx(task_id="")
    url, refusal = _resolve_placement_url(ctx, "http://localhost:5173/")
    assert url == ""
    assert "BROWSER_REMOTE_FORWARD_UNAVAILABLE" in refusal


@pytest.mark.parametrize(
    "asked",
    [
        "http://192.168.1.10:8080/",
        "http://10.0.0.5:3000/",
        "http://172.16.4.4/",
        "http://[fd00::1]/",
    ],
)
def test_a_private_non_loopback_host_on_a_remote_task_is_refused_as_ambiguous(
    broker, asked
):
    """Home's LAN and the target's LAN are different networks and the URL says neither.

    The exemption is loopback-only by decision (plan §3.2, RATIFIED Q3а). Resolving
    a private address against Home would be the wrong-host read again, just one hop
    further out, so it is refused with the ambiguity named.
    """

    url, refusal = _resolve_placement_url(_remote_ctx(), asked)
    assert url == ""
    assert refusal.startswith("⚠️ BROWSER_REMOTE_PRIVATE_HOST_AMBIGUOUS")
    assert broker.calls == []


def test_a_public_host_is_the_same_host_from_either_machine(broker):
    for asked in ("https://example.com/docs", "http://93.184.216.34/"):
        assert _resolve_placement_url(_remote_ctx(), asked) == (asked, "")
    assert broker.calls == []


# ── file:// — Home roots still work, the target's files say so ──────────────


def test_a_file_url_that_exists_on_home_stays_allowed_on_a_remote_task(broker, tmp_path):
    """Home roots are Home-native on EVERY placement, so this is correct, not wrong-host."""

    deliverable = tmp_path / "report.html"
    deliverable.write_text("<h1>report</h1>", encoding="utf-8")
    url, refusal = _resolve_placement_url(_remote_ctx(), deliverable.as_uri())
    assert refusal == ""
    assert url == deliverable.as_uri()


def test_a_file_url_missing_on_home_names_the_deferred_bridge(broker, tmp_path):
    """A bare "file not found" would send the owner hunting for a file on their server."""

    missing = (tmp_path / "target-only" / "index.html").as_uri()
    url, refusal = _resolve_placement_url(_remote_ctx(), missing)
    assert url == missing
    assert refusal.startswith("⚠️ BROWSER_REMOTE_FILE_URL_UNSUPPORTED")
    assert "not implemented yet" in refusal
    assert "read_file" in refusal


# ── the promised origin block ───────────────────────────────────────────────


def test_the_bridged_page_may_talk_to_its_own_forward(broker):
    ctx = _remote_ctx()
    _resolve_placement_url(ctx, "http://localhost:5173/")
    for url in (
        f"{_ORIGIN}/",
        f"{_ORIGIN}/assets/app.js",
        f"http://localhost:{_LOCAL_PORT}/api/data",
        f"http://[::1]:{_LOCAL_PORT}/api/data",
    ):
        assert _remote_foreign_origin_blocked(url, ctx) is False, url


@pytest.mark.parametrize(
    "url",
    [
        # Home's control plane — the whole point of the block.
        "http://127.0.0.1:8765/api/settings",
        "http://localhost:8766/v1/models",
        "http://127.0.0.1:8767/",
        # Any OTHER loopback port: Home's unrelated dev servers.
        "http://localhost:3000/",
        "http://127.0.0.1:9999/",
        # Private / link-local on either machine.
        "http://192.168.1.10/",
        "http://10.1.2.3:8080/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[fd00::5]/",
    ],
)
def test_a_foreign_loopback_or_private_origin_is_blocked_for_the_bridged_page(
    broker, url
):
    ctx = _remote_ctx()
    _resolve_placement_url(ctx, "http://localhost:5173/")
    assert _remote_foreign_origin_blocked(url, ctx) is True, url


def test_the_public_internet_stays_reachable_from_the_bridged_page(broker):
    ctx = _remote_ctx()
    _resolve_placement_url(ctx, "http://localhost:5173/")
    for url in ("https://example.com/x", "https://cdn.example.org/app.js"):
        assert _remote_foreign_origin_blocked(url, ctx) is False, url


def test_the_registered_handler_aborts_what_the_predicate_refuses(broker):
    """The HANDLER, not the predicate — that gap is how this shipped untested.

    Everything above asks `_remote_foreign_origin_blocked` directly. The route that
    actually enforces it was an anonymous lambda, and the only assertions reaching its
    registration were a route COUNT and a pattern STRING; the one behavioural check, in
    `test_browser_isolation.py`, drove `routes[-1]` — which this route had displaced.
    Inverting the handler's condition left 323 tests green while Home's control plane
    and its entire private range became reachable from a bridged page. So the branch
    that calls `abort()` is exercised here, over the three answers the predicate gives:
    refuse Home's control plane and LAN, allow the task's own forward and the internet.
    """

    ctx = _remote_ctx()
    _resolve_placement_url(ctx, "http://localhost:5173/")
    events: list[str] = []
    route = SimpleNamespace(
        request=SimpleNamespace(url=""),
        abort=lambda: events.append("abort"),
        continue_=lambda: events.append("continue"),
        fallback=lambda: events.append("fallback"),
    )
    for url in (
        "http://127.0.0.1:8765/api/settings",
        "http://192.168.1.10/admin",
        f"{_ORIGIN}/assets/app.js",
        "https://example.com/",
    ):
        route.request.url = url
        browser_tools._route_remote_origin_block(route, ctx)
    assert events == ["abort", "abort", "fallback", "fallback"]


def test_the_block_applies_before_any_forward_is_open(broker):
    """A remote task with no forward yet has no permitted loopback origin at all."""

    ctx = _remote_ctx()
    assert _remote_foreign_origin_blocked("http://127.0.0.1:8765/api", ctx) is True
    assert _remote_foreign_origin_blocked(f"{_ORIGIN}/", ctx) is True


def test_the_block_does_not_touch_a_local_placement():
    """Byte-identical: a local task keeps reaching its own loopback and LAN."""

    ctx = _local_ctx()
    for url in (
        "http://127.0.0.1:3000/",
        "http://localhost:8765/api/settings",
        "http://192.168.1.10/",
        "https://example.com/",
    ):
        assert _remote_foreign_origin_blocked(url, ctx) is False, url


@pytest.mark.parametrize(
    "url",
    ["ws://localhost:9999/socket", "data:text/html,<h1>x</h1>", "about:blank"],
)
def test_non_http_schemes_are_left_to_the_existing_scheme_guards(broker, url):
    """This predicate answers about ORIGINS; scheme policy is not its job."""

    ctx = _remote_ctx()
    _resolve_placement_url(ctx, "http://localhost:5173/")
    assert _remote_foreign_origin_blocked(url, ctx) is False


# ── the transport-side lifecycle this relies on ─────────────────────────────


def test_the_broker_closes_a_tasks_forwards_at_every_terminal_seam():
    """The custody claim in the docs, read off the broker rather than trusted.

    The forward is task-scoped, so `finish_task`, both cancel paths, project-session
    close, connection retirement, panic and lifespan teardown must each drop it. A
    forward that outlived its task would be a live `ssh -L` into a remote host with
    nothing left to own it.

    BOUNDARY: this reads the broker's SOURCE, one line at a time, and pairs a
    `_browser_forwards` mention with a closer name spelled on that same line. It
    therefore cannot see a close that happens through an indirection — a helper the seam
    delegates to, a `getattr(manager, name)()` dispatch, a mention and its closer split
    across two lines, or a closer reached from a module other than
    `remote_workspace.py`. It also proves only that the CALL is written, never that it
    runs or succeeds; the behavioural half of that claim is the forward-lifecycle tests
    above, which drive the manager directly. What it does close is the omission that
    matters here — a terminal seam added later with no close written at all.
    """

    import ast

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    source = (repo_root / "ouroboros" / "remote_workspace.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner.setdefault(line, node.name)
    closers: dict[str, set[str]] = {}
    for number, line in enumerate(source.splitlines(), 1):
        text = line.strip()
        if "_browser_forwards" not in text:
            continue
        for closer in ("close_task", "close_connection", "close_all", "panic_close_all"):
            if closer in text:
                closers.setdefault(owner.get(number, "<module>"), set()).add(closer)

    assert "close_task" in closers.get("finish_task", set())
    assert "close_task" in closers.get("cancel", set())
    assert "close_task" in closers.get("cancel_admission", set())
    assert "close_task" in closers.get("_close_project_session_on_broker", set())
    assert "close_connection" in closers.get("_cancel_connection_on_broker", set())
    assert "panic_close_all" in closers.get("panic", set())
    assert "close_all" in closers.get("close", set())
