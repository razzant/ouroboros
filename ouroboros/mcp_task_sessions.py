"""Task-owned stdio sessions for a Playwright MCP browser server.

An owner opts a stdio server in with ``browser_bridge: true``. Only a bound task
attempt opens it, every call of that attempt reuses the one connection, and
Settings never launches it.

The page an action targets is read from the upstream server itself, not from an
adapter tool. The pinned ``@playwright/mcp@0.0.82`` (playwright-core
1.64.0-alpha-1789764292000, ``tools/backend/tabs.ts`` and ``response.ts``)
answers ``browser_tabs`` ``{"action": "list"}`` with a ``### Result`` section of
one line per open tab, ``- <index>: (current) [<title>](<url>)``; the url is
``page.url()`` of the current tab, which every page action targets. The listing
takes no aria snapshot, so element refs the model already holds stay valid. The
host reads it on the same connection before browser policy and Safety, again
just before dispatch, and after the call; a missing, ambiguous or moved answer
refuses the action. Nothing makes the last read and the action atomic: a page
that navigates on its own in between is a disclosed race, not a closed one.

That read judges only the selected page. The requests Playwright routes — an
iframe, a fetch, a beacon, a form post, a worker's or service worker's fetch,
a headless popup's document — are judged by the host's
``browser_request_block_reason``, the decision ``tools/browser.py`` routes its
own requests through. The host installs ``_GUARD_JS`` through upstream's own
``browser.initPage`` seam (``PLAYWRIGHT_MCP_INIT_PAGE``): playwright-core
``tools/backend/tab.ts`` runs each init page with the Playwright ``page`` of
every tab before ``ensureTab`` returns, so the guard's ``context.route`` exists
before the first tool touches a page, and the context route also holds every
later page and frame of that context. Each request waits on the host's verdict
over a private socket; only an explicit allowance lets it on. The guard
acknowledges itself on the same tab listing the host reads before an action,
and an action without that acknowledgement is refused: an owner configuration
that replaces the init page leaves no silent unguarded route. A route never
sees a native redirect hop (a 307 carries a form's POST on) or a WebSocket,
and with ``--extension`` a page-opened tab is attached only after Chrome
created it, so its first requests go unjudged. Service-worker coverage is
specific to this pinned core and assumes the owner has not set
``PLAYWRIGHT_DISABLE_SERVICE_WORKER_NETWORK=1`` in the server environment.

Each bridge process inherits a containment marker scoped to this installation
and task before it exists, so task Stop, cancel and Panic find its live members
in the process table without a ledger row having been written first.

A session is published before its worker starts, so Stop can meet it unstarted
or mid-launch. Revocation and the launch decision share ``_lock``: a revoked
session never launches, and a launch already admitted is scanned again once
its worker has unwound. Only that later scan, over a fresh container on the
original marker, may report the closure ``confirmed``; a scan while the worker
still runs is provisional. A call whose connection closes before any answer
was possibly submitted: its effect is unknown, nothing resends it, and the
session closes. A tool error the server answers is an ordinary result over a
live session.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import dataclasses
import hashlib
import inspect
import json
import logging
import os
import pathlib
import tempfile
import threading
import types
from typing import Any

from ouroboros import browser_policy
from ouroboros.process_containment import (
    CONTAINMENT_ENV_PREFIX,
    MARKER_MEMBER,
    ProcessContainer,
    pid_is_zombie,
    pid_marker_state,
    pids_with_env_marker,
)
from ouroboros.process_custody import record_process
from ouroboros.tools.tool_result import ToolResult

log = logging.getLogger(__name__)

_sessions: dict[tuple[str, str], "TaskBrowserSession"] = {}
_lock = threading.RLock()
_ended_attempts: set[tuple[str, str]] = set()
# Every round's tool listing rediscovers a bridge; a refused or failed open of an
# unchanged entry answers from here instead of spawning again in that attempt.
_failed_opens: dict[tuple[str, str, str], tuple[Any, str]] = {}
_STATE_TOOL = "browser_tabs"
_STATE_ARGS = {"action": "list"}
_SCRIPT_TOOLS = frozenset({"browser_evaluate", "browser_run_code_unsafe"})
_CLOSE_GRACE_SEC = 5
_EXIT_GRACE_SEC = 2
_GUARD_ENV = "OUROBOROS_BROWSER_GUARD"
_GUARD_FACTS_LIMIT = 64 * 1024 * 1024  # Larger request facts are refused, not judged partially.
# Loaded by upstream as ``const { default: func } = require(initPage)``. It runs
# in the bridge's Node process; a page cannot reach it or the host's socket.
_GUARD_JS = r"""'use strict';
const net = require('net');
const socketPath = process.env.OUROBOROS_BROWSER_GUARD;
const guarded = new WeakSet();

function ask(facts) {
  return new Promise((resolve, reject) => {
    const socket = net.createConnection(socketPath);
    const chunks = [];
    socket.setTimeout(30000, () => socket.destroy(new Error('the host did not answer')));
    socket.on('connect', () => socket.end(JSON.stringify(facts) + '\n'));
    socket.on('data', chunk => chunks.push(chunk));
    socket.on('end', () => {
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); } catch (error) { reject(error); }
    });
    socket.on('error', reject);
  });
}

async function decide(route) {
  let answer = {};
  try {
    const request = route.request();
    answer = await ask({ url: request.url(), method: request.method(), post_data: request.postData() });
  } catch (error) {
    answer = {};
  }
  // Only the host's explicit allowance lets a request on; anything else aborts it.
  if (answer && answer.block === '')
    await route.fallback();
  else
    await route.abort('blockedbyclient');
}

exports.default = async ({ page }) => {
  if (!socketPath)
    throw new Error('OUROBOROS_BROWSER_GUARD is not set');
  const context = page.context();
  if (!guarded.has(context)) {
    await context.route('**/*', route => decide(route).catch(() => {}));
    guarded.add(context);
  }
  const answer = await ask({ armed: true });
  if (!answer || answer.armed !== true)
    throw new Error('the Ouroboros host did not acknowledge the request guard');
};
"""
# Node-side code runs beside the guard, not behind it: it can remove the
# guard's route or send requests no route sees (``page.request``), and a
# continuing ``browser_route`` handler runs before the guard and sends the
# request on itself. Page JavaScript (``browser_evaluate``) stays behind it.
_GUARD_BYPASS_TOOLS = frozenset({"browser_run_code_unsafe", "browser_route"})


def _private_socket_dir() -> tempfile.TemporaryDirectory:
    """A 0700 directory for the guard and its socket; an AF_UNIX path holds about 100 bytes."""
    directory = tempfile.TemporaryDirectory(prefix="ouroboros-mcpg-")
    if len(directory.name) > 80 and os.path.isdir("/tmp"):
        directory.cleanup()
        directory = tempfile.TemporaryDirectory(prefix="ouroboros-mcpg-", dir="/tmp")
    return directory


class BrowserBridgeRefusal(PermissionError):
    """A host decision that stopped the call before dispatch (a registered blocked code)."""

    def __init__(self, text: str, code: str = "LEGACY_BLOCKED"):
        super().__init__(text)
        self.code = code


class BrowserBridgeTimeout(TimeoutError):
    """A dispatched bridge call timed out; its close receipt must reach the caller."""


class BrowserBridgeBroken(RuntimeError):
    """The connection closed under a call before it answered; the session is closed.

    ``submitted`` is true for the action itself, whose effect is then unknown;
    an observation that broke this way dispatched no action.
    """

    def __init__(self, text: str, *, submitted: bool, closure: str):
        super().__init__(text)
        self.submitted = submitted
        self.closure = closure


def _owner(ctx: Any) -> tuple[str, str]:
    task = str(getattr(ctx, "task_id", "") or "")
    attempt = getattr(ctx, "task_attempt", None)
    if not task or not getattr(ctx, "task_lifecycle_bound", False):
        raise RuntimeError("MCP browser bridge requires a bound task and attempt")
    # Direct turns omit _attempt; the lifecycle's initial attempt is one.
    return task, str(1 if attempt in (None, "") else attempt)


def _data_root(ctx: Any) -> pathlib.Path:
    from ouroboros.tool_access_paths import canonical_data_root

    return canonical_data_root(ctx)


def _scope(data_root: Any, task_id: str = "") -> str:
    """Marker scope of this installation's bridges, or of one task's."""
    def tag(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    scope = f"mcpb{tag(str(pathlib.Path(data_root).resolve(strict=False)))}_"
    return scope + (f"{tag(task_id)}_" if task_id else "")


def _page_state(result: Any) -> dict[str, Any]:
    """The current tab from the upstream ``browser_tabs`` list, or a refusal."""
    texts = [part.text for part in (getattr(result, "content", None) or [])
             if getattr(part, "type", "") == "text"]
    if getattr(result, "isError", False) or len(texts) != 1 or not texts[0].startswith("### Result\n"):
        raise RuntimeError("BROWSER_STATE_UNKNOWN: the tab listing was not one Result answer")
    lines: list[str] = []
    for line in texts[0][len("### Result\n"):].splitlines():
        if line.startswith("### "):
            break  # Page titles hold no newline, so only upstream starts a section.
        lines.append(line)
    if not lines or not all(line.startswith(f"- {index}:") for index, line in enumerate(lines)):
        raise RuntimeError("BROWSER_STATE_UNKNOWN: the tab listing is not the expected format")
    current = [index for index, line in enumerate(lines) if line.startswith(f"- {index}: (current) [")]
    if len(current) != 1:
        raise RuntimeError("BROWSER_STATE_UNKNOWN: the listing does not name exactly one current tab")
    index = current[0]
    body = lines[index][len(f"- {index}: (current) ["):]
    # Title and url may each contain "](", and then the split is ambiguous; a
    # crashed tab ends in " [crashed]" rather than ")".
    if not body.endswith(")") or body.count("](") != 1:
        raise RuntimeError("BROWSER_STATE_UNKNOWN: the current tab's url cannot be read unambiguously")
    url = body[:-1].split("](", 1)[1]
    if not url:
        raise RuntimeError("BROWSER_STATE_UNKNOWN: the current tab reports no url")
    return {"url": url, "tab_index": index, "open_tabs": len(lines)}


def _policy_inputs(ctx: Any) -> tuple[bool, str]:
    from ouroboros.config import get_runtime_mode
    from ouroboros.tools.core import is_restricted_subagent_profile

    return is_restricted_subagent_profile(ctx), get_runtime_mode()


def _block_reason(ctx: Any, url: str) -> str:
    restricted, runtime_mode = _policy_inputs(ctx)
    return browser_policy.browser_url_block_reason(url, ctx, restricted=restricted, runtime_mode=runtime_mode)


def _request_block_reason(ctx: Any, facts: Any) -> str:
    """The host verdict on one request the guard paused, as ``tools/browser.py`` decides its own."""
    try:
        if not isinstance(facts.get("url"), str) or not isinstance(facts.get("method"), str):
            raise ValueError("request url or method missing")
        if facts.get("post_data") is not None and not isinstance(facts["post_data"], str):
            raise ValueError("request body is not text")
        request = types.SimpleNamespace(url=facts["url"], method=facts["method"], post_data=facts.get("post_data"))
        restricted, runtime_mode = _policy_inputs(ctx)
        return browser_policy.browser_request_block_reason(
            request, ctx, restricted=restricted, runtime_mode=runtime_mode)
    except Exception:
        log.warning("Browser bridge request policy could not read the request or its target", exc_info=True)
        return "BROWSER_POLICY_UNAVAILABLE: the request or its target identity could not be read"


def _authority(ctx: Any, name: str, args: dict, state: dict[str, Any]) -> None:
    """Browser policy on the actual page and on the requested action."""
    if name == "browser_navigate":
        # A navigation replaces the page; its destination is the target fact.
        destination = args.get("url")
        if not isinstance(destination, str) or not destination:
            raise BrowserBridgeRefusal("BROWSER_POLICY_UNAVAILABLE: navigation destination missing")
        reason = _block_reason(ctx, destination)
    else:
        reason = _block_reason(ctx, state["url"])
    if reason:
        raise BrowserBridgeRefusal(reason)
    if name == _STATE_TOOL and args.get("action") != "list":
        raise BrowserBridgeRefusal(
            "BROWSER_POLICY_UNAVAILABLE: browser_tabs may only list; a new, closed or "
            "selected tab cannot be verified before the change"
        )
    if name in _SCRIPT_TOOLS:
        from ouroboros.tool_access import active_tool_profile
        from ouroboros.tools.core import is_restricted_subagent_profile

        if is_restricted_subagent_profile(ctx) and active_tool_profile(ctx) != "acting_subagent":
            raise BrowserBridgeRefusal(
                "BROWSER_LOCAL_READONLY_BLOCKED: local-readonly subagents cannot run arbitrary browser JavaScript."
            )
    # A fulfilling route answers from the server itself and sends nothing on.
    fulfils = name == "browser_route" and (args.get("body") is not None or args.get("status") is not None)
    if name in _GUARD_BYPASS_TOOLS and not fulfils:
        from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency

        restricted, runtime_mode = _policy_inputs(ctx)
        if restricted or not mode_has_unrestricted_agency(runtime_mode):
            raise BrowserBridgeRefusal(
                f"BROWSER_REQUEST_GUARD_BYPASS: {name} runs in the browser server beside the request guard, "
                "where it can remove the guard or send requests no guard sees; browser policy binds in this "
                "runtime mode. Page tools and page JavaScript run behind the guard."
            )


class TaskBrowserSession:
    """One task attempt's connection, its marked process tree and its close outcome."""

    def __init__(self, cfg: Any, ctx: Any):
        from ouroboros.platform_layer import IS_WINDOWS

        if IS_WINDOWS:
            raise RuntimeError("MCP browser bridge requires process custody not yet available on Windows")
        task, self.attempt = _owner(ctx)
        self.cfg = cfg
        self.ctx = ctx  # The request guard's policy context; each call refreshes it.
        self.key = (task, cfg.id)
        self.data_root = _data_root(ctx)
        self.container = ProcessContainer(_scope(self.data_root, task))
        self.token = next(iter(self.container.containment_env()))
        self.ready: concurrent.futures.Future = concurrent.futures.Future()
        self.thread = threading.Thread(target=self._run, name=f"mcp-browser-{cfg.id}", daemon=True)
        self.revoked = False
        self.close_outcome = "unconfirmed: not closed"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: Any = None
        self._read_stream: Any = None
        self._closing = asyncio.Event()
        self._call_lock = threading.Lock()
        self._reap_lock = threading.Lock()
        self._started = False
        self._launched = False  # The worker passed its launch admission (under ``_lock``).
        self._served_out = False  # The worker's loop is closed: it can launch nothing more.
        self.scratch: tempfile.TemporaryDirectory | None = None
        self.guard_dir = _private_socket_dir()
        self.guard_armed = False
        self._blocked: collections.deque = collections.deque(maxlen=20)
        self._blocked_omitted = 0

    def start(self, timeout: int) -> list[dict]:
        with _lock:  # Stop may meet the published session before this start.
            if not self._started:
                if self.revoked:
                    raise RuntimeError("MCP browser task session was revoked before it started")
                self._started = True
                self.thread.start()
        return self.ready.result(timeout=timeout)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        except BaseException as exc:
            if not self.ready.done():
                self.ready.set_exception(exc)
        finally:
            self._ended_unexpectedly()
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()  # A caller waiting on one gets CancelledError, not a hang.
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            with _lock:
                self._served_out = True
            self._reap()  # The final scan: nothing of this worker can launch after it.
            if self.scratch is not None:
                self.scratch.cleanup()
            self.guard_dir.cleanup()

    def _ended_unexpectedly(self) -> None:
        """Revoke a session that ended without a host close; its attempt does not respawn it."""
        with _lock:
            if not self.revoked:
                _failed_opens[(self.key[0], self.attempt, self.cfg.id)] = (
                    self.cfg, "session ended after discovery or during startup")
            self.revoked = True

    def _reap(self) -> str:
        """Scan the marked tree; only a scan once the worker can launch nothing more confirms.

        ``ProcessContainer.reap`` consumes the marker it scans, and the same object
        scanned again sees nothing and answers "" — so every later scan gets a fresh
        container over the original marker instead of reading that as a close.
        """
        with self._reap_lock:
            if self.close_outcome != "confirmed":
                with _lock:
                    final = (self._served_out or (self.revoked and not self._launched)
                             or (self._started and not self.thread.is_alive()))
                container = self.container
                self.container = ProcessContainer.for_scope(self.token[len(CONTAINMENT_ENV_PREFIX):])
                outcome = container.reap()
                if outcome:
                    self.close_outcome = f"unconfirmed: {outcome}"
                else:
                    self.close_outcome = "confirmed" if final else "unconfirmed: session worker did not exit"
            return self.close_outcome

    async def _answer_guard(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """One guard question: an acknowledgement, or a paused request to judge."""
        try:
            try:
                facts = json.loads(await reader.readline())
                if not isinstance(facts, dict):
                    raise ValueError("request facts are not an object")
            except (ValueError, UnicodeDecodeError) as exc:  # oversized, truncated or malformed
                facts, reason = {}, f"BROWSER_POLICY_UNAVAILABLE: unreadable request facts ({type(exc).__name__})"
            else:
                reason = ""
            if facts.get("armed") is True and not self.revoked:
                self.guard_armed = True
                answer: dict[str, Any] = {"armed": True}
            else:
                if self.revoked:
                    reason = "BROWSER_SESSION_REVOKED: the task's browser session is closing"
                # DNS/service identity may block; keep the MCP reader and other
                # paused requests live while this exact policy decision runs.
                reason = reason or await asyncio.to_thread(_request_block_reason, self.ctx, facts)
                if self.revoked:
                    reason = "BROWSER_SESSION_REVOKED: the task's browser session is closing"
                if reason:
                    from ouroboros.utils import truncate_within_limit

                    if len(self._blocked) == self._blocked.maxlen:
                        self._blocked_omitted += 1
                    self._blocked.append(truncate_within_limit(
                        f"{reason} — {facts.get('method') or '?'} {facts.get('url') or '?'}", 400))
                answer = {"block": reason}
            writer.write(json.dumps(answer).encode("utf-8") + b"\n")
            await writer.drain()
        except (ConnectionError, OSError):
            pass  # The guard then aborts the request: no answer is no allowance.
        finally:
            writer.close()

    async def _serve(self) -> None:
        from mcp.types import Implementation
        from ouroboros.mcp_client import ClientSession, _transport_factory
        from ouroboros.version import get_version

        if not self.cfg.cwd:
            self.scratch = tempfile.TemporaryDirectory(prefix="ouroboros-mcp-browser-")
        guard = pathlib.Path(self.guard_dir.name)
        (guard / "guard.cjs").write_text(_GUARD_JS, encoding="utf-8")
        server = await asyncio.start_unix_server(
            self._answer_guard, path=str(guard / "g.sock"), limit=_GUARD_FACTS_LIMIT)
        # The SDK starts the server in its own session; the marker is inherited
        # from birth and survives that and a descendant's later setsid. The
        # host's guard replaces any init page the entry's own env names. The
        # marker is the session's own string: a scan cannot consume it.
        cfg = dataclasses.replace(
            self.cfg,
            cwd=self.cfg.cwd or (self.scratch.name if self.scratch is not None else ""),
            env={**self.cfg.env, self.token: "1",
                 "PLAYWRIGHT_MCP_INIT_PAGE": str(guard / "guard.cjs"), _GUARD_ENV: str(guard / "g.sock")},
        )
        with _lock:  # Revocation and this launch decision are one step.
            if self.revoked:
                server.close()
                raise RuntimeError("MCP browser task session was revoked before its server launched")
            self._launched = True
        async with server, _transport_factory(cfg) as streams:
            pids = pids_with_env_marker(self.token)
            if not pids:
                raise RuntimeError("MCP browser process identity could not be observed")
            for pid in pids:  # The generation reaper's durable record.
                record_process(self.data_root, pid=pid, cmd=[cfg.command, *cfg.args],
                               purpose="mcp_browser_bridge", scope="task", owner_task_id=self.key[0])
            client_kwargs = (
                {"client_info": Implementation(name="Ouroboros", version=get_version())}
                if "client_info" in inspect.signature(ClientSession).parameters else {}
            )
            async with ClientSession(streams[0], streams[1], **client_kwargs) as session:
                await session.initialize()
                tools, cursor = [], None
                for _page in range(50):
                    listed = await (session.list_tools(cursor=cursor) if cursor else session.list_tools())
                    tools.extend({"name": tool.name, "description": tool.description or "",
                                  "input_schema": tool.inputSchema or {}} for tool in listed.tools or [])
                    cursor = getattr(listed, "nextCursor", None)
                    if not cursor:
                        break
                if cursor or _STATE_TOOL not in {tool["name"] for tool in tools}:
                    raise RuntimeError(f"MCP browser server does not list its {_STATE_TOOL} tool")
                if self.revoked:
                    raise RuntimeError("MCP browser task session was revoked while starting")
                self._session, self._read_stream = session, streams[0]
                self.ready.set_result(tools)
                while not self._closing.is_set():
                    # The SDK reader exits on stdio EOF without waking our close event.
                    if streams[0].statistics().open_send_streams == 0:
                        raise RuntimeError("MCP browser transport closed")
                    try:
                        await asyncio.wait_for(self._closing.wait(), timeout=0.2)
                    except asyncio.TimeoutError:
                        pass

    def _request(self, name: str, args: dict, timeout: int, *, dispatch: bool = False) -> Any:
        loop, session = self._loop, self._session
        if self.revoked or loop is None or session is None or loop.is_closed():
            raise RuntimeError("MCP browser task session is revoked or closed")

        def submit() -> concurrent.futures.Future:
            call = session.call_tool(name, args)
            try:
                return asyncio.run_coroutine_threadsafe(call, loop)
            except RuntimeError:
                call.close()
                raise RuntimeError("MCP browser task session is closed") from None

        if dispatch:
            from ouroboros.owner_pause import operation_start

            with operation_start():  # Pause admission for the action, never for its wait.
                future = submit()
        else:
            future = submit()
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            outcome = self.stop()
            raise BrowserBridgeTimeout(
                f"MCP browser call {name!r} did not answer in {timeout}s; its effect is unknown. "
                f"The task session was closed: {outcome}"
            ) from exc
        except concurrent.futures.CancelledError as exc:
            raise self._broken(name, dispatch, "the session closed during the call") from exc
        except Exception as exc:
            if not self._connection_closed(exc):
                raise  # The server answered with an error over a live connection.
            raise self._broken(name, dispatch, f"{type(exc).__name__}: {exc}") from exc

    def _connection_closed(self, exc: BaseException) -> bool:
        """Whether a call failed because its connection closed, not on the server's own answer.

        The SDK fails every pending request with ``CONNECTION_CLOSED`` once the
        server's output has ended; a live server may send that code too, so
        the transport's read side must also be closed.
        """
        import anyio
        from mcp.shared.exceptions import McpError
        from mcp.types import CONNECTION_CLOSED

        if isinstance(exc, (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream)):
            return True
        read = self._read_stream
        return (isinstance(exc, McpError) and exc.error.code == CONNECTION_CLOSED
                and read is not None and read.statistics().open_send_streams == 0)

    def _broken(self, name: str, submitted: bool, cause: str) -> BrowserBridgeBroken:
        """Close a session whose connection closed under ``name``; nothing is resent."""
        self._ended_unexpectedly()
        outcome = self.stop()
        if submitted:
            text = (f"MCP browser call {name!r} was submitted, then the connection closed before any "
                    f"answer ({cause}). Whether it took effect is unknown; it was not resent. "
                    f"The task session was closed: {outcome}")
        else:
            text = (f"MCP browser connection closed during {name!r} ({cause}). "
                    f"The task session was closed: {outcome}")
        return BrowserBridgeBroken(text, submitted=submitted, closure=outcome)

    def _observe(self, timeout: int) -> dict[str, Any]:
        return _page_state(self._request(_STATE_TOOL, dict(_STATE_ARGS), timeout))

    def _observe_before(self, timeout: int) -> dict[str, Any]:
        try:
            return self._observe(timeout)
        except RuntimeError as exc:  # Unknown state refuses; nothing was dispatched.
            raise BrowserBridgeRefusal(f"{exc}; the action was not dispatched") from exc

    def call(self, prefixed: str, name: str, args: dict, ctx: Any, timeout: int) -> tuple[ToolResult, str, str]:
        """The result, any Safety advice and the guard's refusals; a pre-dispatch refusal raises."""
        from ouroboros.mcp_client import _tool_result_from_call_result
        from ouroboros.safety import check_safety

        with self._call_lock:  # Observation, assessment and action share one turn.
            if self.revoked or _owner(ctx) != (self.key[0], self.attempt):
                raise BrowserBridgeRefusal("MCP browser session owner has changed or been revoked")
            self.ctx = ctx
            state = self._observe_before(timeout)
            if not self.guard_armed:  # The listing ran upstream's ensureTab, and so every init page.
                raise BrowserBridgeRefusal(
                    "BROWSER_REQUEST_GUARD_UNAVAILABLE: the bridge never acknowledged the host's request guard "
                    "(an init page in the server's own CLI arguments replaces it); the action was "
                    "not dispatched"
                )
            _authority(ctx, name, args, state)
            allowed, advice = check_safety(
                prefixed, args, messages=getattr(ctx, "messages", None), ctx=ctx,
                resolved_binding={"browser_page": {**state, "observed_by": "browser_tabs list, same connection"}},
            )
            if not allowed:
                raise BrowserBridgeRefusal(advice, code="SAFETY_VIOLATION")
            if self._observe_before(timeout) != state:
                raise BrowserBridgeRefusal(
                    "BROWSER_STATE_CHANGED: the current tab or its url changed during the assessment; "
                    "the action was not dispatched"
                )
            self._blocked.clear()  # Earlier refusals were background requests between calls.
            self._blocked_omitted = 0
            result = _tool_result_from_call_result(self._request(name, args, timeout, dispatch=True))
            blocked = [self._blocked.popleft() for _ in range(len(self._blocked))]
            try:
                after = self._observe(timeout)
                reason = _block_reason(ctx, after["url"])
            except RuntimeError as exc:
                reason = str(exc)
            if reason:
                withheld = (f"⚠️ MCP_TOOL_ERROR: BROWSER_POLICY_POST_DISPATCH — {name} was dispatched and its "
                            f"effect stands, but the page afterwards is not confirmed inside browser policy "
                            f"({reason}). Its result is withheld; read the current state before any retry.")
                result = ToolResult(status="error", code="MCP_ERROR", text=withheld, meta={"host_verdict": True})
            note = ("⚠️ BROWSER_REQUEST_BLOCKED: the host request guard aborted these requests during the call; "
                    "the page saw a network failure for each:\n" + "\n".join(f"- {line}" for line in blocked)
                    ) if blocked else ""
            if self._blocked_omitted:
                note += f"\nEarlier blocked-request diagnostics omitted: {self._blocked_omitted}."
            return result, advice, note

    def stop(self) -> str:
        """Revoke new calls and launches, close the connection, then scan the marked tree.

        A scan while the worker still runs is provisional; the worker's own exit
        scans again, so a later call reads the closure it reached.
        """
        with _lock:
            self.revoked = True
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._closing.set)
            except RuntimeError:
                pass  # The loop already ended.
        if self._started:
            self.thread.join(timeout=_CLOSE_GRACE_SEC)
        self._reap()
        if self._started:
            self.thread.join(timeout=_EXIT_GRACE_SEC)
        return self.close_outcome


def _foreign_live_members(data_root: pathlib.Path, task_id: str) -> str:
    """Why a new session of this task may not open: earlier members still live."""
    scope = ProcessContainer.for_scope(_scope(data_root, task_id))
    marker = next(iter(scope.containment_env()))
    pids = pids_with_env_marker(marker)
    if pids is None:
        return "the process table could not be read"
    held = [session.token for session in _sessions.values()
            if session.key[0] == task_id and not session.revoked]
    foreign = [pid for pid in pids if not pid_is_zombie(pid)
               and not any(pid_marker_state(pid, token) == MARKER_MEMBER for token in held)]
    return f"earlier bridge processes are still live: {foreign}" if foreign else ""


def discover(cfg: Any, ctx: Any, timeout: int) -> list[dict]:
    task, attempt = _owner(ctx)
    with _lock:
        if (task, attempt) in _ended_attempts:
            raise RuntimeError("MCP browser task attempt has ended")
        failed = _failed_opens.get((task, attempt, cfg.id))
        if failed is not None and failed[0] == cfg:
            raise RuntimeError(f"MCP browser bridge did not open in this task attempt: {failed[1]}")
        session = _sessions.get((task, cfg.id))
        if session is not None and (session.attempt != attempt or session.revoked or session.cfg != cfg):
            if session.close_outcome != "confirmed":
                raise RuntimeError("MCP browser session cannot be reopened before its earlier "
                                   f"closure is confirmed ({session.close_outcome})")
            _sessions.pop(session.key, None)
            session = None
        if session is None:
            if blocker := _foreign_live_members(_data_root(ctx), task):
                _failed_opens[(task, attempt, cfg.id)] = (cfg, f"cannot reopen: {blocker}")
                raise RuntimeError(f"MCP browser session of this task cannot reopen: {blocker}")
            session = TaskBrowserSession(cfg, ctx)
            _sessions[session.key] = session
    try:
        return session.start(timeout)
    except BaseException as exc:
        with _lock:
            _failed_opens[(task, attempt, cfg.id)] = (cfg, f"{type(exc).__name__}: {exc}")
        _close(session)
        raise


def call(cfg: Any, prefixed: str, name: str, args: dict, ctx: Any, timeout: int) -> tuple[ToolResult, str, str]:
    task, attempt = _owner(ctx)
    with _lock:
        session = _sessions.get((task, cfg.id))
    if session is None or session.attempt != attempt:
        raise BrowserBridgeRefusal("MCP browser session was not discovered by this task attempt")
    if session.cfg != cfg:
        raise BrowserBridgeRefusal("MCP browser configuration changed during this task attempt")
    return session.call(prefixed, name, args, ctx, timeout)


def _close(session: TaskBrowserSession) -> dict:
    outcome = session.stop()
    if outcome == "confirmed":
        with _lock:
            if _sessions.get(session.key) is session:
                _sessions.pop(session.key, None)
    return {"server": session.cfg.id, "closure": outcome}


def stop_task(ctx: Any, *, end_attempt: bool = True) -> list[dict]:
    """Close held sessions; only terminal cleanup ends admission for this attempt.

    A serialized pause releases its worker and may resume on another process.
    Close its process-local connection without ending the retained attempt.
    Warm in-process waits do not call this cleanup.
    """
    try:
        owner = _owner(ctx)
    except RuntimeError:
        return []  # An unbound context cannot have opened a bridge.
    with _lock:
        if end_attempt:
            _ended_attempts.add(owner)
        for key in [key for key in _failed_opens if key[0] == owner[0]]:
            _failed_opens.pop(key, None)
        found = [session for key, session in _sessions.items() if key[0] == owner[0]]
    return [_close(session) for session in found]


def release_stale(cfg: Any, ctx: Any) -> dict | None:
    """Close this attempt's session opened with ``cfg`` once that entry is no longer current.

    A discovery captures its configuration before it opens; a Settings change
    in between finds nothing to revoke yet, so the refresh that sees its own
    listing discarded disposes of exactly the session it opened.
    """
    task, attempt = _owner(ctx)
    with _lock:
        session = _sessions.get((task, cfg.id))
    if session is None or session.attempt != attempt or session.cfg != cfg:
        return None
    return _close(session)


def revoke_changed(configs: list[Any]) -> list[dict]:
    """Close held bridges whose Settings entry was disabled or changed."""
    current = {cfg.id: cfg for cfg in configs if cfg.enabled and cfg.browser_bridge}
    with _lock:
        stale = [session for session in _sessions.values() if current.get(session.cfg.id) != session.cfg]
    return [_close(session) for session in stale]


def stop_scope(data_root: Any, task_id: str = "", *, request_only: bool = False) -> dict:
    """Panic (whole installation) or cancel (one task) over the marked process table.

    A request only signals; settlement is confirmed only by the container's
    quiet live scans, never inferred from a request or an owner's exit.
    """
    from ouroboros import platform_layer

    container = ProcessContainer.for_scope(_scope(data_root, task_id))
    if not request_only:
        outcome = container.reap()
        return {"closure": "confirmed" if not outcome else f"unconfirmed: {outcome}"}
    marker = next(iter(container.containment_env()))
    pids = pids_with_env_marker(marker)
    if pids is None:
        return {"closure": "unconfirmed: process table unreadable"}
    signalled = [pid for pid in pids if pid_marker_state(pid, marker) == MARKER_MEMBER]
    for pid in signalled:
        platform_layer.force_kill_pid(pid)
    return {"requested_pids": signalled, "closure": "unconfirmed: signal request only"}


def settle_dead_task(data_root: Any, task_id: str) -> dict:
    """Close what a task's bridges left once its worker is confirmed dead.

    The dead worker can issue no further call, so the task's cancellation
    stands. An unconfirmed scan is logged and keeps refusing a reopen of this
    task (``discover``); it is never reported as a confirmed close.
    """
    try:
        outcome = stop_scope(data_root, task_id)
    except Exception as exc:  # A scan failure must not undo a confirmed worker death.
        outcome = {"closure": f"unconfirmed: {type(exc).__name__}: {exc}"}
    if outcome["closure"] != "confirmed":
        log.error("Browser bridge closure unconfirmed for task %s: %s", task_id, outcome["closure"])
    return outcome
