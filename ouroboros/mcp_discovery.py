"""The MCP manager's single background listing owner, including skill lifecycle reports.

Configuration events coalesce one more pass; skill-only passes list missing current evidence. Passive reads
only adopt; a settings-mtime change or a worker report requests discovery. The same owner settles reports off the supervisor intake and can be joined before its caller's
state is torn down. No periodic retry, extra executor or independent skill-settle daemon is involved.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time

log = logging.getLogger(__name__)


def settings_fingerprint(settings: dict, *, skill_settings: bool = True) -> str:
    from ouroboros.config import get_runtime_mode

    payload = {
        "MCP_ENABLED": settings.get("MCP_ENABLED"),
        "MCP_TOOL_TIMEOUT_SEC": settings.get("MCP_TOOL_TIMEOUT_SEC"),
        "MCP_SERVERS": settings.get("MCP_SERVERS"),
        "effective_runtime_mode": get_runtime_mode(),
        "saved_runtime_mode": settings.get("OUROBOROS_RUNTIME_MODE"),
    }
    references = set()
    servers = settings.get("MCP_SERVERS")
    for server in servers if isinstance(servers, list) else []:
        refs = server.get("env_from_settings") if isinstance(server, dict) else None
        if isinstance(refs, dict):
            references.update(ref for ref in refs.values() if isinstance(ref, str))
    if skill_settings:
        from ouroboros.skill_mcp import settings_references

        references.update(settings_references(settings))
    payload["selected_settings"] = {ref: settings.get(ref) for ref in references}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


class Discovery:
    """Owned by one MCPManager; settle and listing share its one coalescing thread."""

    def __init__(self, manager):
        self.manager = manager
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._settle = ""
        self._refresh = 0  # 0: none, 1: missing skill catalogs, 2: all enabled servers
        self._reason = ""

    def _start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run, name="mcp-discovery", daemon=True)
            try:
                self._thread.start()
            except BaseException:
                self._running = False
                raise

    def settle(self, reason: str):
        with self._lock:
            self._settle = reason
            self._start()

    def refresh(self, reason: str, *, missing_only: bool = False):
        with self.manager._lock:
            if not self.manager._enabled or not any(
                    rt.config.enabled and not rt.config.skill_error for rt in self.manager._servers.values()):
                return
        with self._lock:
            self._refresh = max(self._refresh, 1 if missing_only else 2)
            self._reason = reason
            self._start()

    def _settle_skill(self, reason):
        from ouroboros.config import load_settings

        settings = load_settings()
        if not self.manager.listed(settings):
            self.manager.configuration_changed(settings, reason=reason)
        elif self.manager.rebind():
            self.refresh(reason, missing_only=True)

    def _run(self):
        while True:
            with self._lock:
                settle, refresh, reason = self._settle, self._refresh, self._reason
                self._settle, self._refresh = "", 0
                if not settle and not refresh:
                    self._running = False
                    return
            try:
                if settle:
                    self._settle_skill(settle)
                if refresh == 2:
                    self.manager.refresh_all()
                elif refresh == 1:
                    # Re-read at execution: an earlier in-flight listing may already have paid this debt.
                    for server_id in self.manager.rebind():
                        self.manager.refresh_server(server_id)
            except Exception:
                log.warning("Background MCP discovery failed (%s)", settle or reason, exc_info=True)

    def wait(self, timeout: float = 60) -> bool:
        """Join this owner's work, including a pass coalesced while a previous one was ending."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                thread = self._thread
            if thread is None:
                return True
            thread.join(max(0, deadline - time.monotonic()))
            with self._lock:
                if thread is self._thread:
                    return not thread.is_alive()
            if time.monotonic() >= deadline:
                return False
