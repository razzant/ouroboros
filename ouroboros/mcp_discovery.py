"""The MCP manager's single background listing owner, including skill lifecycle reports.

Configuration events coalesce one more pass; skill-only passes list missing current evidence, and a changed Extra
certificate bundle path lists the servers that verify with it (alone, nothing else). Passive reads only adopt; a
settings-mtime change or a worker report requests discovery. The same owner settles reports off the supervisor intake
and can be joined before its caller's state is torn down. No periodic retry, extra executor or independent skill-settle
daemon is involved.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time

log = logging.getLogger(__name__)


def bundle_path(settings: dict) -> str:
    """The saved Extra certificate bundle path, as the connections read it (``net_transport``)."""
    return str(settings.get("OUROBOROS_EXTRA_CA_BUNDLE") or "").strip()


def settings_fingerprint(settings: dict, *, values: bool = True, path=None) -> str:
    """What decides the servers and their listings; ``values=False`` leaves out the settings whose change relists only
    the servers that read them: a skill's settings and the Extra certificate bundle path. A disabled row's references
    affect neither comparison until it is enabled (which changes ``MCP_SERVERS``). An enabled process's setting counts
    as its literal value, the bundle's included; ``path`` replaces only the normalized path read by TLS clients."""
    from ouroboros.config import get_runtime_mode
    from ouroboros.mcp_client import setting_flag

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
        if isinstance(refs, dict) and setting_flag(server.get("enabled", False)):
            references.update(ref for ref in refs.values() if isinstance(ref, str))
    if values:
        from ouroboros.skill_mcp import settings_references

        references.update(settings_references(settings))
        payload["OUROBOROS_EXTRA_CA_BUNDLE"] = bundle_path(settings) if path is None else path
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
        self._servers = {}  # ids the next pass lists as well (``relist``), in order
        self._reason = ""
        self._connections = ""  # the previous configuration event's ``owed`` inputs
        self._bundle_path = None

    def owed(self, settings: dict, listed: str, configs) -> tuple:
        """What a configuration event owes, decided under the manager's lock against the event before: ``("all", [])``
        when what a connection reads changed; else the enabled ``configs`` a changed Extra certificate bundle path
        reaches (``mcp_headers.reads_trust``), with ``"missing"`` (the missing listings) unless the path alone changed:
        a stdio server or skill given the setting itself sees its own literal change. ``listed`` is the fingerprint the
        event before covered."""
        from ouroboros.mcp_headers import reads_trust

        connections = settings_fingerprint(settings, values=False)
        path = bundle_path(settings)
        last = self._bundle_path
        full = connections != self._connections
        self._connections = connections
        self._bundle_path = path
        if full:
            return "all", []
        path_only = path != last and settings_fingerprint(settings, path=last) == listed
        trusting = [cfg.id for cfg in configs if cfg.enabled and reads_trust(cfg)] if path != last else []
        return ("" if path_only else "missing"), trusting

    def schedule(self, reason: str, owed: tuple) -> None:
        """Start what ``owed`` decided: a full pass, or the missing listings (a skill's settings, or an event that
        changed nothing) and the named servers."""
        kind, servers = owed
        if kind == "all":
            self.manager.refresh_all_background(reason=reason)
            return
        if kind == "missing":
            self.refresh(reason, missing_only=True)
        self.relist(reason, servers)

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

    def relist(self, reason: str, servers):
        """List these servers on the next pass, coalesced with whatever else is owed: a full pass covers them."""
        if servers:
            with self._lock:
                self._servers.update(dict.fromkeys(servers))
                self._reason = reason
                self._start()

    def _settle_skill(self, reason):
        from ouroboros.config import load_settings

        settings = load_settings()
        if not self.manager.listed(settings):
            self.manager.configuration_changed(settings, reason=reason)
        if self.manager.rebind():  # the report's own missing listings, whatever that event scheduled
            self.refresh(reason, missing_only=True)

    def _run(self):
        while True:
            with self._lock:
                settle, refresh, reason, servers = self._settle, self._refresh, self._reason, list(self._servers)
                self._settle, self._refresh, self._servers = "", 0, {}
                if not settle and not refresh and not servers:
                    self._running = False
                    return
            try:
                if settle:
                    self._settle_skill(settle)
                if refresh == 2:
                    self.manager.refresh_all()
                else:
                    # Re-read at execution: an earlier in-flight listing may already have paid this debt.
                    for server_id in dict.fromkeys([*servers, *(self.manager.rebind() if refresh == 1 else [])]):
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
