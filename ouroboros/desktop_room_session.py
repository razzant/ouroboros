"""One macOS shell's room input, in memory. Only the built-in main response gets a room ticket.

A new managed server gets a key. Each main-frame navigation gets a fresh native request
challenge. A document must present the matching response ticket through WebKit's reply
handler on that very view and main frame. No draft travels in an HTTP request or response.
A signed/source exec can inherit an already unlinked POSIX shared memory descriptor. The
kernel releases the bytes on close/exit, including Panic. There is no companion or pathname
by the time any private bytes are written.

A room the served registry positively no longer has (``gone``) keeps its slots until Quit,
handed only to the verified built-in document as before, and stops counting as input the
owner could still clear.

A handoff carries room slots (drafts, quotes, file counts, held sends) only to a successor that
speaks this protocol, and while input is held only the built-in page loads; slots reach a verified
main frame alone. Input waits for an exact room restore or the owner's choice. A late edit keeps
the shell; Quit ends the session. The shell's own notices about input a replacement, or Help's page,
cannot take along live here too: fixed words and editor names, never what was typed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import mmap
import os
import re
import secrets
import stat
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

KEY_ENV = "OUROBOROS_DESKTOP_DOCUMENT_KEY"
FD_ENV = "OUROBOROS_DESKTOP_SESSION_FD"
REQUEST_HEADER = "X-Ouroboros-Desktop-Navigation"
RESPONSE_HEADER = "X-Ouroboros-Desktop-Document"
PRESENTATION_HEADER = "X-Ouroboros-Desktop-Interface"
META = "ouroboros-desktop-document"
PANIC_HANDLER = "ouroborosDesktopPanic"
_ROOM_KEY = re.compile(r"ouro_chat_(?:session_id|input_history|active_project|(?:draft|reply|unconfirmed|files)(?::[1-9][0-9]*)?)")
_INPUT = {"ouro_chat_draft", "ouro_chat_reply", "ouro_chat_files", "ouro_chat_unconfirmed"}
HELD = "ouro:renderer-holds-edits"  # always the shell's own sheet: the window that holds the edits may not paint
ASK = "ouro:renderer-asks-edits"  # that window did not answer for its known edits: only its owner gives them up
WAITS = "ouro:help-waits-for-edits"  # Help's page would take the place of the document holding them: it waits instead
_SECOND_BUTTON = 1001  # NSAlertSecondButtonReturn
NOTICES = {
    HELD: ("Your unsaved edits are still in this window",
           "Save or copy your edits in {editors}, then finish editing: the fresh window opens by itself. "
           "Nothing was saved or discarded for you. Quitting Ouroboros gives these edits up."),
    WAITS: ("Your unsaved edits are still in this window",
            "Save or copy your edits in {editors}, then finish editing: the page you chose in Help opens by itself. "
            "Nothing was saved or discarded for you."),
    ASK: ("Your unsaved edits are still in this window",
          "This window did not confirm your edits in {editors}. Keep editing to save or copy them: the fresh window "
          "opens by itself when you finish. Open the fresh window now only to give these edits up: nothing was saved "
          "for you.", "Keep Editing", "Open Fresh Window"),
    "ouro:renderer-refused": ("The new window could not open", "Keep using this window or open built-in setup in Help."),
    "ouro:renderer-incompatible": ("Room input is still in this window",
                                   "Copy and clear drafts and replies, remove staged files and resolve unconfirmed sends "
                                   "so the new interface can open. Built-in setup is available in Help."),
    "ouro:room-input-unconfirmed": ("Check your last room input",
                                    "The previous window could not confirm its last draft, reply or send. Check other "
                                    "unsaved forms. Room input is kept until Quit. To return to the built-in interface, "
                                    "clear Interface folder in Settings › This computer › Advanced and restart. If Settings "
                                    "is unavailable, copy any input you can before quitting, then set OUROBOROS_UI_DIR "
                                    "to an empty string in {settings_path} while Ouroboros is stopped and reopen it."),
}


def native_notice(window, event, editors="", decide=None):
    """One sheet on ``window`` (AppKit thread). It returns at once: no nested run loop, nothing waits for an answer.
    Further buttons are the owner's choice, and ``decide`` hears whether it was the second."""
    from AppKit import NSAlert

    from ouroboros.config import SETTINGS_PATH

    title, message, *buttons = NOTICES[event]
    alert = NSAlert.alloc().init()
    alert.setMessageText_(title)
    alert.setInformativeText_(message.format(settings_path=SETTINGS_PATH, editors=editors))
    for button in buttons:
        alert.addButtonWithTitle_(button)
    alert.beginSheetModalForWindow_completionHandler_(window, decide and (lambda answer: decide(answer == _SECOND_BUTTON)))


def document_ticket(secret: str, challenge: str, *, source: str = "builtin") -> str:
    if not secret or not re.fullmatch(r"[0-9a-f]{64}", challenge):
        return ""
    # Keep the builtin room protocol readable by existing launchers. A receipt for
    # displaying the selected directory or the no-interface page cannot open a room.
    message = challenge if source == "builtin" else f"{source}:{challenge}"
    return hmac.new(secret.encode("ascii"), message.encode("ascii"), hashlib.sha256).hexdigest()


def root_url(url, port):
    # Socket refresh preserves the literal loopback authority and root, changing only its query.
    return port is not None and str(url).partition("#")[0].partition("?")[0] == f"http://127.0.0.1:{port}/"


def anchor_navigation(current, target, reload):
    if not current or reload:
        return False
    here = str(current).partition("#")[0]
    there, separator, _ = str(target).partition("#")
    return here == there and bool(separator)


def served_document(http, response, port, key, challenge, *, allow_unmarked):
    """Identify a displayable main response; only the stamped built-in response gives a ticket."""
    if (not challenge or not http.URL() or not root_url(http.URL().absoluteString(), port)
            or not hasattr(http, "statusCode") or not response.canShowMIMEType()
            or str(http.MIMEType()).lower() != "text/html"
            or str(http.valueForHTTPHeaderField_("Content-Disposition")).lower().startswith("attachment")):
        return "", ""
    for source, status, header in (("builtin", 200, RESPONSE_HEADER), ("directory", 200, PRESENTATION_HEADER),
                                   ("none", 404, PRESENTATION_HEADER)):
        if http.statusCode() == status:
            expected = document_ticket(key, challenge, source=source)
            value = str(http.valueForHTTPHeaderField_(header) or "")
            if expected and value.isascii() and hmac.compare_digest(value, expected):
                return source, expected if source == "builtin" else ""
    if (allow_unmarked and http.statusCode() in (200, 404)
            and not any(http.valueForHTTPHeaderField_(h) for h in (RESPONSE_HEADER, PRESENTATION_HEADER))):
        # A legacy server may confirm what is on screen separately; this confers no room ticket.
        return "unmarked", ""
    return "", ""


# An older server's unmarked root: read the same core-owned selection used by browser reconnection. An isolated
# content world uses its own fetch, even when the selected page is custom.
OLDER_ROOT_PROBE = """
    const r = await fetch(stateURL, {cache:'no-store', redirect:'error', mode:'same-origin'});
    if (!r.ok) return null;
    const source = (await r.json()).served_ui?.source;
    const core = document.querySelector('meta[name="ouroboros-ui-owner"][content="core"]');
    if ((source === 'builtin' && core) || source === 'directory') return source;
    if (source === 'none' && [...document.querySelectorAll('code')]
        .some(el => el.textContent === 'OUROBOROS_UI_DIR=none')) return source;
    return null;
"""
# An older signed built-in has no mounted callback. Its app module finishes after createComposer has installed the
# input handlers. Import the very same module in its own world; WebKit returns its existing module evaluation.
LEGACY_INPUT_PROBE = """
    const entry = document.querySelector('script[type="module"][src="/static/app.js"]');
    if (entry?.src !== new URL('/static/app.js', location.href).href) return false;
    await import(entry.src);
    const field = document.querySelector('#page-chat.active textarea#chat-input');
    return !!field && field.isConnected && !field.disabled && !field.readOnly;
"""


def main_frame_request(message, port):
    frame = message.frameInfo()
    request = frame.request() if frame and frame.isMainFrame() else None
    url = request.URL() if request else None
    return request if url and root_url(url.absoluteString(), port) else None


def matches_document(ticket, expected):
    return bool(expected and isinstance(ticket, str) and ticket.isascii() and hmac.compare_digest(ticket, expected))


def matches_retired_panic(previous, message, params):
    # Only a distinct, still-identifiable outgoing view can stand in for the posted
    # message's tagged request. The current view may already host a different document.
    if not previous or params not in ([], [previous[2]]):
        return False
    import objc

    issuer = message.webView()
    if issuer is not None and objc.pyobjc_id(issuer) != previous[0]:
        return False
    request = main_frame_request(message, previous[1])
    if not request:
        return False
    return bool(params and issuer is not None and not previous[4]) or document_ticket(
        previous[3], str(request.valueForHTTPHeaderField_(REQUEST_HEADER) or "")) == previous[2]


class SessionRefused(Exception):
    """Public reason only; the caller still owns the previous document and its data."""


class RoomSession:
    """The main document's existing string slots; no file contents, sender or route reconstruction."""

    def __init__(self, root: Path):
        self.root = str(root.resolve())
        self.items: dict[str, str] = {}
        self.edits: set[str] = set()
        self.gone: set[str] = set()  # rooms the registry positively no longer has (``launcher_readiness.absent_rooms``)
        self.restore_failed = False  # tell the first verified room as well as the local launcher log
        self.lock, self.revision = Lock(), 0

    @property
    def rooms(self):
        """Project rooms that still count as holding input; Main has no room number and always counts."""
        with self.lock:
            return {room for base, _, room in (key.partition(":") for key in self.items)
                    if base in _INPUT and room and room not in self.gone}

    @property
    def has_room_input(self):
        """Input its owner could still clear or resolve in a room that exists."""
        with self.lock:
            return any(base in _INPUT and room not in self.gone
                       for base, _, room in (key.partition(":") for key in self.items))

    @property
    def needs_transfer(self):
        with self.lock:
            return any(key != "ouro_chat_session_id" for key in self.items)

    def update(self, key, value):
        self.apply([(key, value)])

    def apply(self, changes):
        for key, value in changes:
            if not isinstance(key, str) or not _ROOM_KEY.fullmatch(key) or (value is not None and not isinstance(value, str)):
                raise ValueError("invalid room slot")
        with self.lock:
            for key, value in changes:
                if value is None:
                    self.items.pop(key, None)
                else:
                    if self.items.get(key) != value:
                        self.gone.discard(key.partition(":")[2])  # a page that edits a room's slot still has that room
                    self.items[key] = value
            self.revision += 1

    @contextmanager
    def prepare_exec(self, *, transfer=True):
        # The lifecycle worker encodes and fills shared memory. The final AppKit turn
        # checks a revision and passes a descriptor; it never waits on the page or a writer.
        with self.lock:
            revision, items = self.revision, self.items.copy()
        fd, size = None, 0

        def validate():
            if self.edits:
                raise SessionRefused("unsaved edits in " + ", ".join(sorted(self.edits)))
            if self.revision != revision:
                raise SessionRefused("room input changed during source handoff")

        @contextmanager
        def inherit(env):
            validate()
            if fd is not None:
                os.set_inheritable(fd, True)
                env[FD_ENV] = f"{fd}:{size}"
            try:
                yield
            finally:
                if fd is not None:
                    env.pop(FD_ENV, None)
                    os.set_inheritable(fd, False)

        validate()  # avoid asking a window with known unsaved edits to leave full screen
        try:
            if items and transfer:
                try:
                    from _posixshmem import shm_open, shm_unlink

                    payload = json.dumps([os.getpid(), self.root, items], separators=(",", ":")).encode("ascii")
                    size = len(payload)
                    name = "/ouro-" + secrets.token_hex(10)
                    fd = shm_open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
                    shm_unlink(name)  # no resource tracker or pathname before private bytes are written
                    os.set_inheritable(fd, False)
                    os.ftruncate(fd, size)
                    with mmap.mmap(fd, size, access=mmap.ACCESS_WRITE) as shared:
                        shared[:] = payload
                except (OSError, ImportError) as exc:
                    raise SessionRefused(f"cannot prepare anonymous room memory ({type(exc).__name__})") from exc
            yield validate, inherit
        finally:
            if fd is not None:
                os.close(fd)

    @classmethod
    def receive(cls, root):
        session = cls(root)
        number = os.environ.pop(FD_ENV, None)
        if number is None:
            return session
        try:
            fd, size = map(int, number.split(":"))
            if fd < 3 or size < 1:
                raise ValueError
            info = os.fstat(fd)
            if stat.S_IFMT(info.st_mode) not in (0, stat.S_IFREG) or size > info.st_size:
                raise ValueError
            with mmap.mmap(fd, size, access=mmap.ACCESS_READ) as shared:
                pid, source, items = json.loads(shared[:])
            if pid != os.getpid() or source != session.root:
                raise ValueError
            # The environment alone does not transfer ownership of an arbitrary open fd.
            try:
                if not isinstance(items, dict):
                    raise ValueError
                session.apply(list(items.items()))
            finally:
                os.close(fd)
        except (ValueError, TypeError, OSError, OverflowError, RecursionError, MemoryError):
            session.restore_failed = True
            logging.getLogger("launcher.webkit").warning(
                "Could not restore room input from the source launcher. Starting with an empty desktop session.")
        return session
