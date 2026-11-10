"""The bot token this bridge has accepted, and the one observer that keeps it current.

Every Telegram client of the bridge is built from the accepted token held here, never by reading Settings
for a request. The observer reads the saved token every ``WATCH_SEC`` through the host's strict settings
state (PluginAPI 2.4 ``get_settings_state``): a new value is accepted at once and the generation advances;
an empty or absent one is accepted as missing; a settings document that cannot be read keeps the accepted
token in force and is reported as unknown, never as cleared. A client remembers the generation it was
built for and checks it before each request, so after a change nothing more goes out with the old token:
a multipart operation stops with ``TelegramBindingChanged`` after the parts Telegram already answered, and
nothing is resent through another bot.

The observer also keeps the Mini App companion on the saved token: on its first read and after each
accepted change it asks the host to apply the settings to the companion
(``refresh_companion_process``), which respawns it only when its token differs. The request names the token
read; only an answer that the companion carries it settles that token, so a save the observer has not read yet
(``settings_changed``) is asked about again at its next read. Once settled, each read confirms from the host's
memory (``only_if_differs``) that the companion still carries it, so one started elsewhere on other settings
(recovery after exhausted restarts) is brought back. Saving a token grants, enables and resumes nothing: a
disabled skill has no observer, and Off or a revoked grant closes this API before anything here runs again.
A read or Mini App start the operating system refuses is logged once and retried at the next read, so it
never spends the observer's restarts; any other failure still ends it visibly.
"""
from __future__ import annotations

import asyncio
import threading
import weakref

from .telegram_api import TelegramBindingChanged, TelegramTokenMissing

TOKEN_KEY = "TELEGRAM_BOT_TOKEN"
WATCH_SEC = 2.0
# Outcomes after which this generation needs no further refresh (PluginAPI 2.4's closed set); the rest are
# asked again next read: not_started, retained, stop_unconfirmed, settings_unknown, settings_changed.
_SETTLED_REFRESH = frozenset({
    "unchanged", "not_running", "restarted", "started", "access_closed", "panic",
    "not_server", "no_supervisor", "disabled", "missing_grants", "not_desired",
})
# Asked again without a warning: a refused start warns once by itself, and a newer save is the next read's to apply.
_QUIET_RETRY = frozenset({"start_failed", "settings_changed"})
# Settled answers after which the companion carries the token; each later read confirms that from memory.
_APPLIED = frozenset({"unchanged", "not_running", "restarted", "started"})


class TokenBinding:
    """The accepted token and its generation; ``state`` is pending, configured, missing, unknown or closed.

    A closed binding (the API reported access closed, or the skill unloads) keeps no token and has advanced its
    generation, so no client built before keeps working. ``validated`` is the generation the poller's getMe
    proved; readiness is that exact binding, held only in memory, never a saved token or an older record. The
    poller's own record (ready, waiting, refused) is held the same way, for the generation it was made for: after a
    change, or in a new load, none speaks until the poller records one for the token in force.
    """

    def __init__(self, authority=None) -> None:
        # ``authority()``: whether this load's API still holds runtime access. Unload clears it before it stops
        # anything, so a send in flight ends at its next request, not after the companion's stop.
        self._authority = authority
        self._lock = threading.Lock()
        self.token, self.generation, self.state, self.validated = "", 0, "pending", -1
        self.companion = ""  # the last refresh outcome of the Mini App companion, for status
        self._record = (-1, "", "")  # the poller's (generation, state, reason), for status

    def snapshot(self) -> tuple[str, int]:
        with self._lock:
            return self.token, self.generation

    def current(self, generation: int) -> bool:
        with self._lock:
            if generation != self.generation or self.state == "closed":
                return False
        if self._authority is not None and not self._authority():
            self.close()
            return False
        return True

    def observe(self, reply: dict) -> bool:
        """Apply one ``get_settings_state`` reply; True when the accepted token changed."""
        status = str((reply or {}).get("status") or "")
        with self._lock:
            if self.state == "closed":
                return False  # a closed API never reopens; a new load brings a new binding
            if status == "unknown":
                self.state = "unknown"  # the accepted token stays in force
                return False
            if status != "ok":
                self._close_locked()
                return True
            token = str(((reply.get("values") or {}).get(TOKEN_KEY)) or "").strip()
            self.state = "configured" if token else "missing"
            if token == self.token:
                return False
            self.token, self.generation = token, self.generation + 1
            return True

    def _close_locked(self) -> None:
        self.state, self.token, self.generation = "closed", "", self.generation + 1

    def close(self) -> None:
        """Unload, Off or a revoked grant: every client of this binding stops before its next request."""
        with self._lock:
            if self.state != "closed":
                self._close_locked()

    def read_now(self, api) -> bool:
        """One synchronous read (registration, a status request); a failed read says unknown, never raises."""
        try:
            return self.observe(read_token_state(api))
        except Exception:
            with self._lock:
                if self.state != "closed":
                    self.state = "unknown"
            return False

    def mark_validated(self, generation: int) -> None:
        with self._lock:
            if generation == self.generation and self.state != "closed":
                self.validated = generation

    def is_validated(self) -> bool:
        with self._lock:
            return bool(self.token) and self.validated == self.generation and self.state != "closed"

    def note(self, generation: int, state: str, reason: str = "") -> bool:
        """Keep the poller's record for ``generation``; one for a generation no longer in force is dropped (False)."""
        with self._lock:
            if generation != self.generation or self.state == "closed":
                return False
            self._record = (generation, state, reason)
            return True

    def noted(self) -> tuple[str, str]:
        """The poller's (state, reason) for the token in force; empty until it recorded one for this generation."""
        with self._lock:
            generation, state, reason = self._record
            return (state, reason) if generation == self.generation and self.state != "closed" else ("", "")


def accepted_token(api) -> str:
    """The token in force for this API: read once while no observation was made yet, then from memory."""
    binding = binding_for(api)
    if binding.state == "pending":
        binding.read_now(api)
    return binding.snapshot()[0]


def read_token_state(api) -> dict:
    """The saved token's strict state; a host without PluginAPI 2.4 answers through ``get_settings``,
    which cannot tell an unreadable settings document from a missing token."""
    reader = getattr(api, "get_settings_state", None)
    if reader is None:
        return {"status": "ok", "values": dict(api.get_settings([TOKEN_KEY]) or {}), "missing": []}
    return reader([TOKEN_KEY])


_BINDINGS: "weakref.WeakKeyDictionary[object, TokenBinding]" = weakref.WeakKeyDictionary()
_PINNED: dict[int, tuple[object, TokenBinding]] = {}  # an API object without weak references (stand-ins)
_BINDINGS_LOCK = threading.Lock()


def _runtime_authority(api):
    """Whether ``api`` still holds runtime access, asked without reading settings (``get_settings_state([])``);
    a host without that method has no such check. Gone, refused or failed reads as closed."""
    if not callable(getattr(api, "get_settings_state", None)):
        return None
    try:
        ref = weakref.ref(api)
    except TypeError:
        ref = lambda: api  # noqa: E731 — a stand-in without weak references is pinned anyway

    def holds_access() -> bool:
        target = ref()
        if target is None:
            return False
        try:
            reply = target.get_settings_state([])
        except Exception:
            return False
        return str((reply or {}).get("status") or "") != "access_closed"

    return holds_access


def binding_for(api) -> TokenBinding:
    """The binding of this PluginAPI instance: one per loaded generation of the skill."""
    with _BINDINGS_LOCK:
        try:
            binding = _BINDINGS.get(api)
            if binding is None:
                binding = _BINDINGS[api] = TokenBinding(_runtime_authority(api))
            return binding
        except TypeError:
            pinned = _PINNED.get(id(api))
            if pinned is None or pinned[0] is not api:
                pinned = _PINNED[id(api)] = (api, TokenBinding(_runtime_authority(api)))
            return pinned[1]


def _report_failure(api, what: str, exc: Exception) -> None:
    """One handler failure line: no token is a wait the status shows, and a token change stops the rest quietly."""
    if isinstance(exc, TelegramTokenMissing):
        return
    if isinstance(exc, TelegramBindingChanged):
        api.log("info", f"Telegram {what} stopped: the bot token changed before its next part; nothing was resent.")
        return
    api.log("error", f"Telegram {what} error: {exc}")


_WAITING_FOR = {  # why the bridge has no token, by the binding's state
    "missing": ("info", "Telegram bot token is not set; the bridge waits for it (Settings → Secrets)."),
    "unknown": ("warning", "Telegram settings could not be read; the bridge waits until it can read the saved token."),
}


async def _wait_for_bot_token(api, binding, poll_sec: float = 1.0) -> None:
    """No token: wait quietly, spending no restart budget, until a saved one is accepted. The binding's state says
    why (not set, settings unreadable): one line each time that reason changes, none while it stays the same, and
    none for a closing binding."""
    if binding.state == "pending":
        binding.read_now(api)
    said = ""
    while not binding.snapshot()[0]:
        reason = binding.state
        if reason in _WAITING_FOR and reason != said:
            said = reason
            api.log(*_WAITING_FOR[reason])
        await asyncio.sleep(poll_sec)


def _note_os_failure(api, failing: str, what: str, exc: OSError) -> str:
    """One warning when a kind of operating-system failure starts; repeats of it stay quiet until a tick succeeds."""
    kind = f"{what}:{type(exc).__name__}"
    if kind != failing:
        api.log("warning", f"Telegram could not {what} ({type(exc).__name__}); retrying every {WATCH_SEC:g} s.")
    return kind


def make_token_observer(api, companion: str = "", on_change=None):
    """The supervised observer; ``companion`` is the Mini App companion's name when it is registered, and
    ``on_change()`` hears each accepted change (the typing cue's wake: its pacing belongs to one token)."""
    binding = binding_for(api)

    async def observe() -> None:
        settled = -1  # the generation whose companion refresh concluded
        failing = ""  # the operating-system failure already logged, until a tick succeeds again
        while True:
            try:
                reply = await asyncio.to_thread(read_token_state, api)
            except OSError as exc:  # a read the system refused says nothing about the token: unknown
                reply, failing = {"status": "unknown"}, _note_os_failure(api, failing, "read the saved token", exc)
            changed = binding.observe(reply)
            if binding.state == "closed":
                return  # the skill is unloading; a new load brings its own observer
            if changed:
                api.log("info", "Telegram bot token " + ("applied." if binding.token else "cleared; the bridge waits for one."))
                if on_change is not None:
                    on_change()
            token, generation = binding.snapshot()
            confirm = settled == generation and binding.companion in _APPLIED
            if companion and binding.state != "unknown" and (settled != generation or confirm):
                try:
                    outcome = await asyncio.to_thread(api.refresh_companion_process, companion,
                                                      expected={TOKEN_KEY: token}, only_if_differs=confirm)
                except OSError as exc:
                    # A process start the system refused (fork under memory pressure, a missing executable)
                    # is retried at the next tick, never spends this observer's restarts: Off, a revoked
                    # grant, unload and Panic still close the API, and the next refresh answers with them.
                    outcome, failing = "start_failed", _note_os_failure(api, failing, "start the Mini App", exc)
                else:
                    if failing:
                        api.log("info", "Telegram token observer recovered; the saved token is applied again.")
                    failing = ""
                reported, binding.companion = binding.companion, outcome
                settled = generation if outcome in _SETTLED_REFRESH else -1
                if outcome in {"restarted", "started"}:
                    api.log("info", "Telegram Mini App restarted with the saved bot token.")
                elif outcome not in _SETTLED_REFRESH and outcome not in _QUIET_RETRY and outcome != reported:
                    api.log("warning", f"Telegram Mini App could not apply the saved token yet ({outcome}); retrying.")
            elif failing and binding.state != "unknown":
                api.log("info", "Telegram token observer recovered; the saved token is read again.")
                failing = ""
            await asyncio.sleep(WATCH_SEC)

    return observe


__all__ = ["TOKEN_KEY", "TokenBinding", "WATCH_SEC", "accepted_token", "binding_for", "make_token_observer",
           "read_token_state"]
