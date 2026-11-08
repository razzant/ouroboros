"""The private browser-request policy callback of one ``browser_policy`` MCP task session.

The session (``mcp_task_session``) starts this listener before its server process
exists and gives only that process ``OUROBOROS_BROWSER_GUARD`` (``127.0.0.1:<port>``)
and ``OUROBOROS_BROWSER_GUARD_TOKEN`` (``secrets.token_hex(32)``). The server's own
JavaScript (owned by its skill, as are every browser tool name and the route that
asks) sends one UTF-8 JSON line per TCP connection and gets one line back:

- ``{"armed": true, "token": T}`` → ``{"armed": true}`` while the session is live,
  else ``{"armed": false, "block": reason}``: the guard is acknowledged before any
  page action.
- ``{"action": "evaluate", "url": U?, "token": T}`` → ``{"block": ""}`` or a reason:
  ``browser_policy.browser_script_block_reason``, then the URL decision for a
  non-empty ``U``.
- ``{"url", "method", "post_data", "resource_type"?, "body_present"?, "token"}`` →
  ``{"block": browser_policy.browser_request_block_reason(...)}`` for every routed
  request, the agent's or the human's in the shared tab.

Every decision reads the session's bound ToolContext with the actor facts
``tools/browser.py`` passes for its own route (restricted profile, effective mode);
no predicate is copied here or into JavaScript. The client treats anything but the
exact allowance as a block. Decisions run on this listener's worker threads, never
on the MCP SDK event loop, so DNS inside the policy cannot stall the session's
reader or the server's pings. A line is bounded; tokens and bodies are never logged.
At most ``MCP_BROWSER_GUARD_PENDING`` questions are held, deciding or waiting for a
worker; another is refused at once. Each question's read allowance runs from its
acceptance, so time spent waiting counts. Retiring the guard answers every waiting
question with a block and runs none of their decisions; a decision already running
cannot be interrupted, finishes on its worker and is answered with a block.

Adapted from razzant/ouroboros#1401 ``mcp_task_sessions.py`` (Andrei Kaznacheev with Ouroboros,
head 4384d057): its private callback answered by the host's own request policy and acknowledged
before page actions, here over portable loopback TCP and without its tool table or lease state.
"""

from __future__ import annotations

import contextvars
import hmac
import json
import logging
import secrets
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Dict

from ouroboros.config import (
    MCP_BROWSER_GUARD_LINE_BYTES, MCP_BROWSER_GUARD_PENDING, MCP_BROWSER_GUARD_READ_SEC, MCP_BROWSER_GUARD_WORKERS,
)

log = logging.getLogger(__name__)

ADDRESS_ENV = "OUROBOROS_BROWSER_GUARD"
TOKEN_ENV = "OUROBOROS_BROWSER_GUARD_TOKEN"
GUARD_ENV_NAMES = frozenset({ADDRESS_ENV, TOKEN_ENV})  # host-owned: a saved server config may not set them
_CLOSED = "BROWSER_SESSION_CLOSED: the task's MCP session is closing"
_FULL = "BROWSER_POLICY_UNAVAILABLE: too many browser-policy questions are already waiting"


class BrowserGuard:
    """One session's loopback listener; ``closing()`` refuses, ``close()`` stops it."""

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self.token = secrets.token_hex(32)
        self.armed = False
        self._closing = threading.Event()  # every question is answered with a block
        self._stopped = threading.Event()  # no question is accepted any more
        self._listener = socket.create_server(("127.0.0.1", 0))  # bound before the process can ask
        self._listener.settimeout(0.25)
        self.address = f"127.0.0.1:{self._listener.getsockname()[1]}"
        # A page-load burst queues behind these, within the pending bound; the asker's own deadline fails closed.
        self._pool = ThreadPoolExecutor(max_workers=MCP_BROWSER_GUARD_WORKERS, thread_name_prefix="mcp-browser-guard")
        self._held = threading.BoundedSemaphore(MCP_BROWSER_GUARD_PENDING)  # one place per accepted question
        self._waiting: set = set()  # accepted questions no worker has taken yet
        self._waiting_lock = threading.Lock()
        # A clean context: the listener never carries the invocation that started its session.
        self._thread = threading.Thread(target=contextvars.Context().run, args=(self._accept,),
                                        name="mcp-browser-guard-accept", daemon=True)
        self._thread.start()

    def environment(self) -> Dict[str, str]:
        return {ADDRESS_ENV: self.address, TOKEN_ENV: self.token}

    def closing(self) -> None:
        """From now on every question is answered with a block (the session is ending)."""
        self._closing.set()

    def close(self) -> None:
        """Stop accepting once the session's process is gone; every waiting question gets a block now.

        A decision already running is not interrupted (a policy read cannot be): its worker finishes
        it after this returns, and its answer is a block too."""
        self._closing.set()
        self._stopped.set()
        self._thread.join(timeout=2.0)  # the accept loop closes its own listener
        self._pool.shutdown(wait=False, cancel_futures=True)  # no waiting question starts from here on
        with self._waiting_lock:
            waiting, self._waiting = self._waiting, set()
        for connection in waiting:
            _refuse(connection, _CLOSED)
            self._held.release()

    def _accept(self) -> None:
        try:
            while not self._stopped.is_set():
                try:
                    connection, _ = self._listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    time.sleep(0.05)  # a transient accept failure (descriptor pressure) is retried
                    continue
                deadline = time.monotonic() + MCP_BROWSER_GUARD_READ_SEC  # waiting for a worker counts
                if not self._held.acquire(blocking=False):
                    _refuse(connection, _FULL)
                    continue
                with self._waiting_lock:
                    self._waiting.add(connection)
                try:
                    self._pool.submit(self._answer, connection, deadline)
                except RuntimeError:  # the pool is shut down: nobody can answer
                    if self._taken(connection):
                        _refuse(connection, _CLOSED)
                        self._held.release()
                    return
        finally:
            self._listener.close()

    def _taken(self, connection: socket.socket) -> bool:
        """Claim a waiting question for its one answerer; False once ``close`` has answered it."""
        with self._waiting_lock:
            if connection not in self._waiting:
                return False
            self._waiting.discard(connection)
            return True

    def _answer(self, connection: socket.socket, deadline: float) -> None:
        if not self._taken(connection):
            return
        try:
            with connection:
                try:
                    facts = json.loads(_read_line(connection, deadline).decode("utf-8"))
                    if not isinstance(facts, dict):
                        raise ValueError("request facts are not an object")
                    answer = self._decide(facts)
                except Exception as exc:  # oversized, truncated, slow or malformed: no answer is an allowance
                    answer = {"block": f"BROWSER_POLICY_UNAVAILABLE: unreadable request facts ({type(exc).__name__})"}
                _send(connection, answer, MCP_BROWSER_GUARD_READ_SEC)
        finally:
            self._held.release()

    def _decide(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        arming = facts.get("armed") is True
        token = facts.get("token")
        if not isinstance(token, str) or not hmac.compare_digest(token.encode("utf-8"), self.token.encode("utf-8")):
            reason = "BROWSER_POLICY_UNAVAILABLE: the question did not carry this session's guard token"
        elif self._closing.is_set():
            reason = _CLOSED
        elif arming:
            self.armed = True
            return {"armed": True}
        elif "action" in facts:
            reason = (self._script_reason(facts.get("url")) if facts.get("action") == "evaluate"
                      else "BROWSER_POLICY_UNAVAILABLE: unknown guard action")
        else:
            reason = self._request_reason(facts)
        if self._closing.is_set():  # the session ended while this decision ran
            reason = _CLOSED
        return {"armed": False, "block": reason} if arming else {"block": reason}

    def _actor(self) -> tuple:
        from ouroboros import browser_policy
        from ouroboros.tools.core import is_restricted_subagent_profile

        return is_restricted_subagent_profile(self._ctx), browser_policy.browser_runtime_mode(self._ctx)

    def _script_reason(self, url: Any) -> str:
        from ouroboros import browser_policy

        try:
            if reason := browser_policy.browser_script_block_reason(self._ctx):
                return reason
            if url in (None, ""):
                return ""
            if not isinstance(url, str):
                return "BROWSER_POLICY_UNAVAILABLE: the page url is not text"
            restricted, runtime_mode = self._actor()
            return browser_policy.browser_url_block_reason(url, self._ctx, restricted=restricted,
                                                           runtime_mode=runtime_mode)
        except Exception:
            log.warning("MCP browser guard could not decide a script question", exc_info=True)
            return "BROWSER_POLICY_UNAVAILABLE: the actor's script rights could not be read"

    def _request_reason(self, facts: Dict[str, Any]) -> str:
        from ouroboros import browser_policy

        url, method, post_data = facts.get("url"), facts.get("method"), facts.get("post_data")
        if not isinstance(url, str) or not isinstance(method, str):
            return "BROWSER_POLICY_UNAVAILABLE: the request url or method is missing"
        if post_data is not None and not isinstance(post_data, str):
            return "BROWSER_POLICY_UNAVAILABLE: the request body is not text"
        request = SimpleNamespace(url=url, method=method, post_data=post_data,
                                  resource_type=facts.get("resource_type"), body_present=facts.get("body_present"))
        try:
            restricted, runtime_mode = self._actor()
            return browser_policy.browser_request_block_reason(request, self._ctx, restricted=restricted,
                                                               runtime_mode=runtime_mode)
        except Exception:
            log.warning("MCP browser guard could not read a request's target identity", exc_info=True)
            return "BROWSER_POLICY_UNAVAILABLE: runtime service identity could not be read"


def _send(connection: socket.socket, answer: Dict[str, Any], timeout: float) -> None:
    """Send one answer line. Unread bytes that already arrived are drained first, without waiting, so the
    close does not reset the answer away; an asker cut off anyway reads that as a block, like any answer
    but the exact allowance."""
    try:
        connection.setblocking(False)
        connection.recv(65536)
    except OSError:
        pass
    try:
        connection.settimeout(timeout)
        connection.sendall(json.dumps(answer).encode("utf-8") + b"\n")
    except OSError:
        pass  # the asker then blocks on its own


def _refuse(connection: socket.socket, reason: str) -> None:
    """Answer a question that was never read (the guard is full or retiring) with a block, and close it.

    The block goes out at once; the close then waits, within the same bound, for the asker's one line,
    which may still be on its way: a line landing on a closed socket resets the connection, and a reset
    discards the delivered block on some stacks (Windows)."""
    deadline = time.monotonic() + 1.0
    with connection:
        try:
            connection.settimeout(1.0)
            connection.sendall(json.dumps({"block": reason}).encode("utf-8") + b"\n")
            _read_line(connection, deadline)
        except (OSError, ValueError):
            pass  # an asker cut off reads that as a block, like any answer but the exact allowance


def _read_line(connection: socket.socket, deadline: float) -> bytes:
    """One request line before ``deadline`` (acceptance plus the read allowance) within the size bound."""
    chunks, size = [], 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("request facts did not arrive in time")
        connection.settimeout(remaining)
        chunk = connection.recv(65536)
        end = chunk.find(b"\n") if chunk else 0
        chunks.append(chunk if end < 0 else chunk[:end])
        size += len(chunks[-1])
        if size >= MCP_BROWSER_GUARD_LINE_BYTES:  # the asker sends a larger body as unavailable
            raise ValueError("request facts exceed the line bound")
        if not chunk or end >= 0:
            return b"".join(chunks)
