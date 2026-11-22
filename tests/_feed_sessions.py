"""Recorded feed sessions and the stand they are played on (tests/test_feed_session_replay_browser.py).

A session (``tests/fixtures/feed_sessions/*.json``) is what reaches a room's feed over time: the rooms' saved rows,
then its events in order: openings, replies arriving, history reads held and released, photos loading, the window's
width and height, the phone keyboard, wheel, keys and touch. Its rows are synthetic, never live data. ``rooms`` maps
``main`` or a Project's key to its ``rows`` (``Stand.rows``) and ``unread``; each event names one action and its
value: ``open``, ``arrive`` (``words``, ``role``), ``wheel`` / ``swipe`` (px, + toward the end), ``key``, ``image``,
``viewport`` ([w, h]), ``drag`` (a handle, px), ``keyboard`` (visible height, null: down), ``older``, ``quote``,
``hold`` (``latest`` | ``older`` | ``after``), ``release``, ``latest`` (↓), ``pause`` (ms); ``expect`` (``following`` |
``reading``) names what the reader is doing after it when the rules alone do not say.

The stand serves the production page (``web/index.html``, no runtime) and answers each ``/api/chat/history`` read with
the real endpoint over a dialogue store built from the session, so cursors, coverage and the newest arrival are the
server's own; the shell's other reads get quiet answers, the socket is a stand-in the player pushes frames into, and a
photo's bytes wait for the session's ``image`` event. Every read receipt is recorded as it arrives, with the room shown
then. Time is real: no fake clock, no replaced animation frames.

After every event the player reads the feed (``FACTS``) and ``Reader`` checks the rules of docs/DESIGN.md the event
touches: a room first opens at its newest message, an unread one on its newest arrival, and is read once its newest
reply is shown, never before (a room not shown is never read while another is, so a reply there leaves it unread and
marked in the list; a read room loses the mark); a return finds it as left; a quote lands on its row; the live edge
follows replies while a passage being read stays put through replies, older pages, photos, width, height and the
keyboard; a late picture grows away from the place read (up out of view when it starts above the band, so its bottom
and the passage after it stay; down when read); a gesture moves the text by its own way, as far as it went;
↓ reaches the retained end at once. The checks hold for every engine, the WebKit without
scroll anchoring that the macOS window runs too (``overflow-anchor: none``).
"""
from __future__ import annotations

import base64
import hashlib
import json
import struct
import threading
import zlib
from datetime import datetime, timedelta, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web'
SESSIONS = Path(__file__).resolve().parent / 'fixtures' / 'feed_sessions'
START = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)
WORDS = ('the', 'reply', 'reads', 'history', 'passage', 'owner', 'room', 'feed', 'width', 'keeps', 'place', 'newest',
         'message', 'older', 'page', 'lands', 'while', 'reading', 'scrolls', 'wheel', 'photo', 'below', 'above', 'line')
TOLERANCE = 2  # px: a passage that stays may move by subpixel rounding, never by a line
LIVE_EDGE = 48  # px: a gesture toward the end that stops this close to it follows (chat_reading_position.js nearBottom)
PHOTO_HOLD = 120  # s: a held photo's bytes wait at most this long for its event (its picture is checked to wait)


def sessions() -> list[str]:
    """Every recorded session, by its file's stem, in a stable order (shards agree on node ids)."""
    return sorted(path.stem for path in SESSIONS.glob('*.json'))


def load(name: str) -> dict:
    return json.loads((SESSIONS / f'{name}.json').read_text(encoding='utf-8'))


def words(seed: int, count: int) -> str:
    """Deterministic prose of ``count`` words: a row's height is the same in every run and engine."""
    state, out = seed * 2654435761 % 2**32, []
    for _ in range(count):
        state = (state * 1103515245 + 12345) % 2**31
        out.append(WORDS[state % len(WORDS)])
    return ' '.join(out).capitalize() + '.'


def png(width: int, height: int) -> bytes:
    """A solid image of the given size, so loading it changes the layout by exactly its height."""
    row = b'\x00' + b'\x80\x90\xa0' * width
    chunk = lambda kind, data: (struct.pack('>I', len(data)) + kind + data  # noqa: E731
                                + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(row * height)) + chunk(b'IEND', b''))


def photo_url(key: str) -> str:
    return f'/api/tasks/feed-photos/artifacts/chat-media-{hashlib.sha256(key.encode()).hexdigest()}.png'


class Stand:
    """One session's server: the page, the real history endpoint over the session's store, held reads and photos."""

    def __init__(self, session: dict, data_dir: Path):
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from ouroboros.gateway.history import make_chat_history_endpoint

        self.session, self.data = session, data_dir
        self.rooms: dict[str, dict] = {}  # session room key -> {chat_id, project_id}
        self.minute = 0
        self.main_revision = 1
        self.seen: dict[str, int] = {}
        self.shown = ''  # the Project room the page shows or is opening ('' while it shows Main)
        self.allowed: dict[str, int] = {}  # a Project room not shown -> the receipt it may still reach
        self.receipts: list[dict] = []  # every receipt that reads a Project room further, as it arrives
        self.photos: dict[str, threading.Event] = {}  # photo -> set once its bytes may be served
        self.stored: dict[str, list[dict]] = {}  # room -> its rows as stored, oldest first
        self.quotes: dict[str, dict] = {}  # reply -> {'text': its words, 'target': the words of the row it quotes}
        self.holds: list[dict] = []
        self.reads: list[dict] = []
        self.photo_heights: dict[str, int] = {}
        self.lock = threading.Lock()
        self._seed()
        self.history = TestClient(Starlette(routes=[Route('/api/chat/history', make_chat_history_endpoint(data_dir))]))
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), partial(_Handler, stand=self, directory=str(WEB)))
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    # -- the store ----------------------------------------------------------------------------------------------
    def _seed(self):
        from ouroboros.projects_registry import create_project

        for key, room in self.session['rooms'].items():
            if key == 'main':
                self.rooms[key] = {'chat_id': 1, 'project_id': ''}
            else:
                project = create_project(self.data, f'feed-{key}', name=room.get('name') or f'Room {key}')
                self.rooms[key] = {'chat_id': project['chat_id'], 'project_id': project['id']}
            for spec in room.get('rows', []):
                for row in self.rows(key, spec):
                    self.append(key, row)
            revision = self.revise(key)
            # A read room was read to its newest message; an unread one was read before its last arrival.
            self.seen[key] = revision - (1 if room.get('unread') else 0)
            if key != 'main':  # not shown yet: no receipt may read it further
                self.allowed[key] = self.seen[key]

    def rows(self, key: str, spec: dict) -> list[dict]:
        """A row spec's rows: ``count`` replies (or the owner's, ``role``) of ``words`` [low, high]; a ``photo`` of
        ``height`` px whose bytes wait for its ``image`` event; a ``reply`` quoting the room's row number ``to``."""
        if 'photo' in spec:
            self.photos[spec['photo']] = threading.Event()
            self.photo_heights[spec['photo']] = spec.get('height', 360)
            return [{'type': 'photo', 'text': f'Photo {spec["photo"]}', 'caption': f'Photo {spec["photo"]}',
                     'mime': 'image/png', 'task_id': 'feed-photos', 'download_url': photo_url(spec['photo']),
                     'direction': 'out'}]
        if 'reply' in spec:
            from ouroboros.chat_references import reference_from_row

            target = self.stored[key][spec['to'] - 1]
            text = words(20_000 + self.minute, spec.get('words', [30, 30])[0])
            self.quotes[spec['reply']] = {'text': text, 'target': target['text']}
            return [{'direction': 'out', 'text': text, 'reply_to': reference_from_row(target)}]
        low, high = spec.get('words', [24, 80])
        direction = 'in' if spec.get('role') == 'owner' else 'out'
        return [{'direction': direction, 'text': words(self.minute + index, low + (index * 37) % (high - low + 1))}
                for index in range(spec.get('count', 1))]

    def append(self, key: str, row: dict) -> dict:
        from ouroboros import dialogue_store

        self.minute += 1
        stored = {'ts': (START + timedelta(minutes=self.minute)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                  'chat_id': self.rooms[key]['chat_id'], **row}
        dialogue_store.append(self.data, 'chat', stored)
        self.stored.setdefault(key, []).append(stored)
        return stored

    def newest_reply(self, key: str) -> str:
        """The room's newest arrival that is not the owner's: what reading the room means (DESIGN "Unread")."""
        return next((row['text'] for row in reversed(self.stored.get(key, []))
                     if row['direction'] == 'out' and row.get('type') != 'photo'), '')

    def revision(self, key: str) -> int:
        """The room's visible revision now: what its read receipt (``seen``, written by the page) has to reach."""
        if key == 'main':
            return self.main_revision
        from ouroboros.projects_registry import get_project

        return int(get_project(self.data, self.rooms[key]['project_id'])['visible_revision'])

    def revise(self, key: str) -> int:
        """The room's visible revision after its rows changed: what the shell's state read reports."""
        from tests._dialogue import advance_room_revision

        if key == 'main':
            self.main_revision += 1
            return self.main_revision
        return int(advance_room_revision(self.data, project_id=self.rooms[key]['project_id'])['visible_revision'])

    def arrive(self, key: str, *, count: int = 40, role: str = 'assistant') -> dict:
        """A reply stored in the room and its live frame, as the bus stores and broadcasts it: a reply carries no
        client id (the row and the frame share role, words and time); an owner's message from another tab does."""
        owner = role == 'owner'
        identity = {'client_message_id': f'feed-owner-{self.minute + 1}'} if owner else {}
        row = self.append(key, {'direction': 'in' if owner else 'out', 'text': words(10_000 + self.minute, count),
                                **identity})
        if not owner:  # the owner's own message is nothing unread (DESIGN "Unread")
            self.revise(key)
        return {'type': 'chat', 'role': 'user' if owner else 'assistant', 'content': row['text'], 'markdown': False,
                'is_progress': False, 'ts': row['ts'], 'task_id': '', 'chat_id': row['chat_id'], **identity,
                **({'sender_session_id': 'another-tab'} if owner else {})}

    # -- what the shell reads -----------------------------------------------------------------------------------
    def projects(self) -> list[dict]:
        from ouroboros.projects_registry import list_projects

        return list_projects(self.data)

    def state(self) -> dict:
        rows = self.projects()
        return {'supervisor_ready': True, 'active_chat_activities': [], 'active_direct_turns': [], 'projects': rows,
                'project_chat_ids': [row['chat_id'] for row in rows], 'main_visible_revision': self.main_revision}

    def show(self, key: str, keeps: int = 0):
        """The page shows (or starts to open) room ``key`` from now on. The Project room it leaves may still be read as
        far as ``keeps`` (the revision of a newest reply it showed: a receipt in flight), never by what comes after."""
        with self.lock:
            if self.shown and self.shown != key:
                self.allowed[self.shown] = max(self.seen.get(self.shown, 0), keeps)
            self.shown = '' if key == 'main' else key

    def preferences(self, written: dict | None = None) -> dict:
        with self.lock:
            for project_id, value in ((written or {}).get('project_seen_revision') or {}).items():
                key = next((k for k, room in self.rooms.items() if room['project_id'] == project_id), None)
                if key and int(value) > self.seen.get(key, 0):  # with the room shown as it arrives
                    self.receipts.append({'room': key, 'seen': int(value), 'shown': self.shown or 'main',
                                          'allowed': self.allowed.get(key, 0)})
                if key:
                    self.seen[key] = max(self.seen.get(key, 0), int(value))
            if isinstance((written or {}).get('main_seen_revision'), int) and 'main' in self.rooms:
                self.seen['main'] = max(self.seen.get('main', 0), written['main_seen_revision'])
            return {'project_seen_revision': {room['project_id']: self.seen.get(key, 0)
                                              for key, room in self.rooms.items() if room['project_id']},
                    'main_seen_revision': self.seen.get('main', self.main_revision)}

    def read_history(self, query: str) -> tuple[int, bytes]:
        params = parse_qs(query)
        cursor = (params.get('cursor') or [''])[0]
        kind = 'around' if params.get('around') else _cursor_kind(cursor)
        read = {'kind': kind, 'chat_id': int((params.get('chat_id') or ['1'])[0]), 'cursor': cursor}
        with self.lock:
            self.reads.append(read)
            hold = next((h for h in self.holds if not h['caught'] and h['kind'] in (kind, 'any')), None)
            if hold:
                hold['caught'] = read
        if hold:
            hold['event'].wait(30)
        with self.lock:
            response = self.history.get(f'/api/chat/history?{query}')
        return response.status_code, response.content

    def hold(self, kind: str) -> dict:
        hold = {'kind': kind, 'caught': None, 'event': threading.Event()}
        with self.lock:
            self.holds.append(hold)
        return hold

    def release(self):
        with self.lock:
            holds, self.holds = self.holds, []
        for hold in holds:
            hold['event'].set()

    def release_photo(self, name: str):
        self.photos[name].set()

    def close(self):
        self.release()
        for photo in self.photos.values():
            photo.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.history.close()


def _cursor_kind(cursor: str) -> str:
    """What a read asks for, from its cursor (opaque to the client; the server's own JSON, base64url)."""
    if not cursor:
        return 'latest'
    try:
        decoded = json.loads(base64.urlsafe_b64decode(cursor + '=' * (-len(cursor) % 4)))
    except ValueError:
        return 'older'
    if decoded.get('kind') == 'page':
        return 'latest' if decoded.get('recent') else 'older'
    return str(decoded.get('kind') or 'older')


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, stand: Stand, **kwargs):
        self.stand = stand
        super().__init__(*args, **kwargs)

    def log_message(self, *_):
        pass

    def _json(self, body, status=200):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 - the stdlib's handler names
        url = urlsplit(self.path)
        if url.path == '/api/chat/history':
            status, body = self.stand.read_history(url.query)
            return self._json(body, status)
        if url.path.startswith('/api/tasks/feed-photos/artifacts/'):
            name = next((n for n in self.stand.photos if photo_url(n) == url.path), None)
            if name is None:
                return self._json({'error': 'not found'}, 404)
            self.stand.photos[name].wait(PHOTO_HOLD)
            data = png(640, self.stand.photo_heights[name])
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        if url.path == '/api/state':
            return self._json(self.stand.state())
        if url.path == '/api/projects':
            wanted = parse_qs(url.query).get('id')
            return self._json({'projects': [row for row in self.stand.projects() if not wanted or row['id'] == wanted[0]]})
        if url.path == '/api/ui/preferences':
            return self._json(self.stand.preferences())
        if url.path == '/api/onboarding':  # a configured install: no setup overlay
            self.send_response(204)
            return self.end_headers()
        if url.path.startswith('/api/'):
            return self._json({})
        self.path = url.path.removeprefix('/static') or '/'
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        url = urlsplit(self.path)
        length = int(self.headers.get('Content-Length') or 0)
        try:
            body = json.loads(self.rfile.read(length) or b'{}')
        except ValueError:
            body = {}
        if url.path == '/api/ui/preferences':
            return self._json(self.stand.preferences(body if isinstance(body, dict) else {}))
        return self._json({'ok': True})


# The page's socket is this stand-in: it opens, beats like the server's socket, and delivers the frames the player
# pushes (``__feedPush``). History reads in flight are counted for the rest check (``__feedReads``).
INIT = """(() => {
    const sockets = [];
    class FeedSocket extends EventTarget {
        constructor(url) {
            super();
            this.url = String(url); this.readyState = 0; this.onopen = this.onmessage = this.onclose = this.onerror = null;
            sockets.push(this);
            setTimeout(() => { this.readyState = 1; this._fire(new Event('open')); this._beat(); }, 20);
            this._timer = setInterval(() => this._beat(), 10000);
        }
        _fire(event) {
            this['on' + event.type]?.call(this, event);
            this.dispatchEvent(event);
        }
        _beat() { if (this.readyState === 1) this._deliver({type: 'heartbeat'}); }
        _deliver(frame) { this._fire(new MessageEvent('message', {data: JSON.stringify(frame)})); }
        send() {}
        close() { this.readyState = 3; clearInterval(this._timer); this._fire(new Event('close')); }
    }
    Object.assign(FeedSocket, {CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3});
    window.WebSocket = FeedSocket;
    window.__feedPush = frame => { for (const socket of sockets) if (socket.readyState === 1) socket._deliver(frame); };
    window.__feedOpen = () => sockets.some(socket => socket.readyState === 1);
    window.__feedReads = {inflight: 0, done: 0};
    const nativeFetch = window.fetch.bind(window);
    window.fetch = (input, init) => {
        const url = new URL(typeof input === 'string' ? input : input?.url || '', location.href);
        if (url.pathname !== '/api/chat/history') return nativeFetch(input, init);
        window.__feedReads.inflight += 1;
        const settle = () => { window.__feedReads.inflight -= 1; window.__feedReads.done += 1; };
        return nativeFetch(input, init).then(response => {
            const json = response.json.bind(response);
            response.json = () => json().finally(settle);
            return response;
        }, error => { settle(); throw error; });
    };
})()"""

# The macOS window's WebKit (WKWebView, macOS 26) has no scroll anchoring: this engine mode removes it from the feeds.
NO_ANCHOR = """(() => {
    const style = document.createElement('style');
    style.textContent = '#chat-messages, .chat-messages { overflow-anchor: none !important; }';
    (document.head || document.documentElement).append(style);
})()"""

# The room's feed as the reader sees it, in the page's coordinates: the read band between the bar's reach and the
# composer (project_read_state.js readBand), the row at its top, the row read before (``place``, by its identity), a
# message named by its words (``named``: an unread room's newest arrival, a quote's target), a photo's picture named
# by its caption (``picture``: whether it still loads, its row and the row after it), whether the newest reply starts
# in the band, the live edge, the newest row, ↓ and the rows shown twice (a defect). A row is a keyed bubble, or the
# keyed item inside an unkeyed one (a photo's).
FACTS = """([sel, place, words, reply, photo]) => {
    const feed = document.querySelector(sel);
    if (!feed || !feed.getClientRects().length) return null;
    const page = feed.parentElement, rect = feed.getBoundingClientRect();
    const header = page.querySelector('.chat-page-header')
        || page.closest('.project-panel')?.querySelector('.project-panel-bar');
    const composer = page.querySelector('.chat-input-area');
    let top = rect.top, bottom = rect.bottom;
    if (header?.getClientRects().length) {
        const box = header.getBoundingClientRect();
        top = Math.max(top, box.bottom + (parseFloat(getComputedStyle(header).getPropertyValue('--edge-reach')) || 0));
    }
    if (composer?.getClientRects().length) bottom = Math.min(bottom, composer.getBoundingClientRect().top);
    const keyed = '[data-history-id], [data-client-message-id]';
    const rows = [...feed.querySelectorAll(keyed)]
        .filter(node => { const bubble = node.closest('.chat-bubble'); return !bubble || bubble === node || !bubble.matches(keyed); })
        .filter(node => node.getClientRects().length);
    const id = node => node.dataset.historyId || 'client:' + node.dataset.clientMessageId;
    const at = node => {
        const box = node.getBoundingClientRect();
        return {id: id(node), y: box.top, top: box.top - top, bottom: box.bottom - top, height: box.height};
    };
    const first = rows.find(node => node.getBoundingClientRect().bottom > top + 1);
    const held = place && rows.find(node => id(node) === place);
    // A message by its words: its keyed bubble, or the keyed item in an unkeyed one (a photo's), the row a landing aims at.
    const bubble = text => {
        const node = text && [...feed.querySelectorAll('.chat-bubble')].find(item => item.querySelector('.message')?.textContent.includes(text));
        return node && !node.matches(keyed) ? [...node.querySelectorAll(keyed)].find(item => item.textContent.includes(text)) || node : node;
    };
    const named = bubble(words), newestReply = bubble(reply);
    const item = photo ? [...feed.querySelectorAll('.chat-gallery-item')].find(node => node.textContent.includes(photo)) : null;
    const image = item?.querySelector('img'), img = image?.getBoundingClientRect();
    // The picture's own row and the row after it: the passage that follows a picture growing up out of view.
    const own = item && rows.find(node => node === item || node.contains(item));
    const after = own && rows[rows.indexOf(own) + 1];
    const shown = node => Boolean(node) && node.getBoundingClientRect().top >= top - 1
        && node.getBoundingClientRect().top < bottom;
    const newest = rows.at(-1), button = page.querySelector('.chat-scroll-bottom-btn');
    const count = (keys, map = new Map()) => { for (const key of keys) map.set(key, (map.get(key) || 0) + 1); return map; };
    // A reply drawn twice shows its words twice (a photo's menu repeats by design: media bubbles are not compared).
    const texts = [...feed.querySelectorAll('.chat-bubble:not(:has(.chat-photo-open)) > .message')]
        .map(node => node.textContent.trim().slice(0, 60)).filter(Boolean);
    const twice = [...count(rows.map(id)), ...count(texts)].filter(([, n]) => n > 1).map(([key]) => key);
    return {scrollTop: feed.scrollTop, scrollHeight: feed.scrollHeight, clientHeight: feed.clientHeight,
        gap: feed.scrollHeight - feed.scrollTop - feed.clientHeight, band: {top, bottom, height: bottom - top},
        first: first ? at(first) : null, place: held ? at(held) : null, named: named ? at(named) : null,
        picture: img ? {top: img.top - top, bottom: img.bottom - top, height: img.height, loading: !image.complete,
            row: own ? id(own) : null, next: after ? at(after) : null} : null,
        replyShown: shown(newestReply),
        newest: newest ? {...at(newest), shown: newest.getBoundingClientRect().bottom <= bottom + 1} : null,
        rows: rows.length, resizing: feed.hasAttribute('data-resizing'), twice,
        latest: {shown: Boolean(button?.classList.contains('visible')), busy: button?.getAttribute('aria-busy') === 'true'},
        reads: window.__feedReads?.done || 0, inflight: window.__feedReads?.inflight || 0};
}"""

# A feed at rest: its scroll box and size unchanged for eight frames, no history read in flight (unless one is held
# on purpose) and no width still moving. Real frames: the engine's own animation clock.
REST = """([sel, held]) => new Promise((done, fail) => {
    let last = '', still = 0;
    const deadline = performance.now() + 20000;
    const tick = () => {
        const feed = document.querySelector(sel);
        const reads = window.__feedReads?.inflight || 0;
        const now = !feed ? 'none' : [feed.scrollTop, feed.scrollHeight, feed.clientHeight,
            feed.hasAttribute('data-resizing'), held ? 0 : reads].join('|');
        still = now === last ? still + 1 : 0;
        last = now;
        if (feed && still >= 8 && !feed.hasAttribute('data-resizing') && (held || !reads)) return done(still);
        if (performance.now() > deadline) return fail(new Error(`the feed ${sel} never rested: ${now}`));
        requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
})"""


class Reader:
    """The rules of docs/DESIGN.md a feed event touches, checked on the facts before and after it.

    It holds what the reader is doing in each room (following the live edge, or reading a passage) and which rooms
    the page session has opened, so a return is held to "as left" and a first opening to "at its newest message".
    """

    def __init__(self, session: dict):
        self.session = session
        self.mode: dict[str, str] = {}  # room -> 'following' | 'reading'
        self.place: dict[str, dict] = {}  # room -> the row at the band's top when last at rest
        self.room = ''
        self.problems: list[str] = []

    def note(self, rule: str, ok: bool, facts: dict, detail: str = ''):
        if not ok:
            self.problems.append(f'{rule}: {detail} | facts {json.dumps(_brief(facts))}')

    def following(self, after: dict) -> bool:
        return after['gap'] <= TOLERANCE and bool(after['newest'] and after['newest']['shown'])

    def kept(self, room: str, after: dict) -> tuple[bool, str]:
        """The row that was at the read band's top is where it was, by its identity, within subpixel rounding."""
        place, now = self.place.get(room), after.get('place')
        if not place:
            return False, 'no row read was recorded before this event (a stand error)'
        if not now:
            return False, f'the row read ({place["id"]}) is gone from the feed'
        # On the screen: when the bar enters the flow under a phone keyboard the band's top moves, the text must not.
        moved = now['y'] - place['y']
        return abs(moved) <= TOLERANCE, f'the row read ({place["id"]}) moved {moved:+.1f}px on the screen'

    def picture(self, room: str, before: dict, after: dict):
        """A late picture grows away from the place read (DESIGN "History edges"): up out of view when it starts above
        the band's top, so its bottom and the passage after it stay (the row read, or the row after the picture's own
        when that row reaches into the band); down when the reader is on it, so its top and the row read stay."""
        was, now = before.get('picture'), after.get('picture')
        if not (was and now):
            self.note('image: the picture loading is in the feed', False, after, f'picture before {was}, after {now}')
            return
        above, place = was['top'] < 0, self.place.get(room)
        if above and place and place['id'] == was['row']:
            follows, then = was['next'], now['next']
            ok = not follows or bool(then and then['id'] == follows['id']
                                     and abs(then['y'] - follows['y']) <= TOLERANCE)
            detail = f'the row after the picture, {follows and follows["id"]}, is now {then}'
        else:
            ok, detail = self.kept(room, after)
        self.note('image: the passage being read stays where it was', ok, after, detail)
        moved = now['bottom'] - was['bottom'] if above else now['top'] - was['top']
        self.note('image: a picture above the passage grows up out of view' if above else 'image: a picture read grows down',
                  abs(moved) <= TOLERANCE, after, f'its {"bottom" if above else "top"} moved {moved:+.1f}px')

    def receipts(self, receipts: list[dict], facts: dict | None):
        """A Project room not shown is never read while another room is: its receipt reaches no further than the room
        left it (a receipt in flight for the newest reply it showed then), so a reply there leaves it unread."""
        for receipt in receipts:
            self.note('a reply in a room not shown leaves it unread', receipt['room'] == receipt['shown']
                      or receipt['seen'] <= receipt['allowed'], facts or {},
                      f'{receipt["room"]}: its read receipt reached {receipt["seen"]} while {receipt["shown"]} was '
                      f'shown (it was left at {receipt["allowed"]})')

    def landed(self, after: dict) -> bool:
        """The named message shows its start in the read band, clear of the bar and of the composer."""
        named = after.get('named')
        return bool(named) and -TOLERANCE <= named['top'] <= after['band']['height'] - min(24, named['height'])

    def settle(self, room: str, after: dict, edge: float = TOLERANCE):
        """What the reader is doing once an event has settled: following when the feed rests at its live edge (a
        gesture toward the end that stops within the app's live edge, 48 px, follows: wheel, swipe or key)."""
        self.mode[room] = 'following' if after['gap'] <= edge else 'reading'
        self.place[room] = dict(after['first']) if after.get('first') else None

    def check(self, event: dict, before: dict | None, after: dict | None, extra: dict | None = None):
        kind = next(iter(event))
        room = self.room
        if after is None:
            self.problems.append(f'{kind}: the room {room!r} shows no feed')
            return
        self.note('a row shows once', not after['twice'], after, f'rows shown twice: {after["twice"]}')
        self.receipts((extra or {}).get('receipts', []), after)
        if kind == 'image':  # its bytes come with this event: a picture loaded before (a hold run out) compares nothing
            was = (before or {}).get('picture')
            self.note('image: the picture still loads until its event (else a stand error)',
                      bool(was and was['loading']), before or after, f'picture {was}')
        away = (extra or {}).get('away')
        if away:  # a reply in a Project room not shown
            self.note('a room not shown with a new reply is marked unread in the list', away['marked'], after, away['room'])
        if 'marked' in (extra or {}):
            self.note('a room read loses its unread mark in the list, one still unread keeps it', extra['marked'], after,
                      f'read receipt {extra["read"]}')
        if 'read' in (extra or {}):
            # Read only once its newest reply shows in the band, clear of the bar and the composer; never before.
            self.note('the room is read once its newest reply is shown, and only then',
                      extra['read'] == after['replyShown'], after,
                      f'read receipt {extra["read"]}, newest reply shown {after["replyShown"]}')
        toward_end = (kind in ('wheel', 'swipe') and event[kind] > 0
                      or kind == 'key' and event['key'] in ('PageDown', 'End'))
        edge = LIVE_EDGE if toward_end else TOLERANCE
        if 'expect' in event:
            self.note(f'{kind}: the reader is then {event["expect"]}',
                      ('following' if after['gap'] <= edge else 'reading') == event['expect'], after)
        if kind == 'open':
            target = event['open']
            if target in self.mode:  # a return within the page session finds the room as left
                if self.mode[target] == 'following':
                    self.note('a return to a followed room follows the fresh end', self.following(after), after)
                else:
                    ok, detail = self.kept(target, after)
                    self.note('a return to older history keeps that place', ok, after, detail)
                    self.note('a return to older history keeps ↓ shown', after['latest']['shown'], after)
            elif self.session['rooms'][target].get('unread'):
                self.note('an unread room opens on its newest arrival', self.landed(after), after,
                          f'its newest arrival at {after.get("named")}')
            else:
                self.note('a room first opens at its newest message', self.following(after), after)
        elif kind in ('arrive', 'image', 'viewport', 'drag', 'keyboard', 'older', 'release', 'pause'):
            # Whatever changes around the room shown (a reply here or elsewhere, a page, a photo, a size), its
            # reader keeps doing what they did: the live edge follows, a passage being read stays.
            if self.mode.get(room) == 'following' and kind != 'older':
                self.note(f'{kind}: the live edge follows', self.following(after), after)
            elif kind == 'image':
                self.picture(room, before, after)
            else:
                ok, detail = self.kept(room, after)
                self.note(f'{kind}: the passage being read stays where it was', ok, after, detail)
                if kind == 'arrive' and event['arrive'] == room:
                    self.note('arrive: a new reply stays below the passage', after['gap'] > TOLERANCE, after)
                    self.note('arrive: ↓ stays shown while reading', after['latest']['shown'], after)
                if kind == 'older':
                    self.note('older: an older page lands above the passage', after['rows'] > before['rows'], after,
                              f'{before["rows"]} rows before, {after["rows"]} after')
        elif kind in ('wheel', 'swipe', 'key'):
            delta = event.get('wheel', event.get('swipe'))
            if kind == 'key':
                delta = -1 if event['key'] in ('PageUp', 'ArrowUp', 'Home') else 1
            at_edge = before['scrollTop'] <= 0 if delta < 0 else before['gap'] <= TOLERANCE
            moved = after['scrollTop'] - before['scrollTop']
            self.note(f'{kind}: the gesture scrolls the feed', at_edge or (moved < 0 if delta < 0 else moved > 0), after,
                      f'scrollTop {before["scrollTop"]} -> {after["scrollTop"]} for {delta}')
            # The text moves by the gesture's own way, never by a jump the page adds (a page landing above moves
            # scrollTop, not the text): the row read before it travels its way, at most as far as the wheel or the
            # finger went (a swipe ends at rest, no fling) or a page key's screen, give or take a line.
            place, now = self.place.get(room), after.get('place')
            if place and now and event.get('key') not in ('End', 'Home'):
                travelled = place['y'] - now['y']
                reach = (before['clientHeight'] if kind == 'key' else abs(delta)) + 24
                self.note(f'{kind}: the text travels by the gesture, never a jump',
                          travelled * delta >= -TOLERANCE and abs(travelled) <= reach, after,
                          f'the row read travelled {travelled:+.0f}px for a gesture of {delta}')
        elif kind == 'quote':
            self.note('quote: a quote lands on the row it quotes', self.landed(after), after,
                      f'the quoted row at {after.get("named")}')
        elif kind == 'latest':
            self.note('latest: ↓ reaches the retained end at once', bool(extra and extra.get('frames') is not None
                                                                         and extra['frames'] <= 3), after, f'{extra}')
            if not (extra or {}).get('held'):
                self.note('latest: ↓ then follows the present', self.following(after), after)
        if kind != 'hold':
            self.settle(room, after, edge)
        if kind == 'latest' and (extra or {}).get('held'):
            self.mode[room] = 'following'


def _brief(facts: dict | None) -> dict:
    if not facts:
        return {}
    return {key: facts.get(key) for key in ('scrollTop', 'scrollHeight', 'clientHeight', 'gap', 'first', 'place', 'named',
                                            'picture', 'newest', 'rows', 'resizing', 'latest', 'reads', 'inflight')}
