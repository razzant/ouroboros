"""PR #3896 through the owner's own door: what a numberless length refusal teaches the NEXT fresh chat.

A provider answered a Max-sized first request with HTTP 400, code ``"1261"``, message exactly ``"Prompt too long"`` and
no number (Z.ai's wording; #3858 recognizes it). Such a refusal measures nothing about the model's window, yet it
proves that THIS size was not admitted on THIS account. The fresh chat that comes next must not blindly send a request
that large again.

What runs is one real keyless server (``tests/system_e2e``'s clone, settings, worker carriers, loopback model) and the
owner's WebSocket chat frame, the one the SPA sends: ``{"type": "chat", "chat_id": <a room that has never spoken>}``,
each frame under its own unique client message id. That is the stock door: the server's own transaction commits the
canonical inbound row and stamps its ``origin_message_ref`` on the turn before the carrier is dispatched, and the turn
is found by that id (the sibling scenarios' ``_received_turn``). Nothing seeds an inbound row of a new room or touches
its history; the off-wire carrier (``lay_authored_pages``) lays only the twenty story rooms. No planner and no loop is
called directly, no window, refusal or hint is written beforehand and no component under test is patched — including
the size fit itself: every rung below is a request the running server chose and the loopback model really accepted or
refused over HTTP. Per scenario ONE server, ONE model route, ONE base URL and ONE settings file serve the new rooms:

* A1 and A2 on the same synthetic transport key A;
* B1 on key B, reached by the owner's own ``POST /api/settings`` (the same route and URL, the same data root);
* in the negative control only, A3 after the owner's own save moves the key to A and the line moves (``stage R``).

Every row is read from what the server received: status, ``Authorization``, reply cap, size and the markers of the
full books, the books' maps, the BIBLE line, the owner's words, the story's long bodies and its authored leads. The
Development book's own line (``dev``: the witness tests/test_context_refusal_passes.py reads, present only in
docs/development) is recorded and printed too, so a reading of the real Max view is visible in the table; nothing
asserts it, because no measurement in this series has shown whether that book is in an owner turn here.

WHICH POSTS ARE A ROOM'S OWN TURN is decided by what a request is, never by what the assertions are about to say. The
loopback model records every POST in ARRIVAL order the moment its body is read (``WireLedger``), before any reply is
written, so a slow or refused request cannot move behind a later one; the HTTP status it really wrote joins the SAME
record, and a window is read only when every record in it is settled. The model is owned by one scenario and serves one
turn at a time, so a window runs from the previous turn's end to this one's and no POST falls between two turns. Every
POST of a window lands in exactly one place (``partition``): ``light`` — the one documented post-turn memory draft, a
lone tool-less user message that opens with a sentence quoted from ouroboros/memory_fallback.py (lines 65 and 70),
excluded wherever it lands because a turn's late phase is not ordered against its durable result, so the draft can
reach this window or the next; ``owner`` — a user-role input of the request carries the room's short marker
(``room_marker``), the current owner message, so a title cut short, a retry under another reply cap or a send without
tool schemas is still chosen and then fails its named assertion instead of vanishing; ``unowned`` — everything else,
and it fails by name: no request that could be a turn's first or extra send is excused as harmless, a marker-less
tool-less one that leads the first send or a differently capped retry included. A copy of the marker in the system view
or the room's memory is not the owner's message. ``check_wire`` reads every room of every scenario that way, the
negative control and stage R included.

THREE FIXED LINES, each placed from what this stand measures on the wire under the unified resident Body Map +
Reference Atlas (replacing the retired 600 KB whole-books-to-maps rung):

* ``enough`` (E) = 294,000, between the measured refused full-Max first send (~310.7k / ~325.3k) and the hinted
  folded-memory Max first send (~276.8k, 14 bodies + 20/20 leads; cold walk ~172.8k): ~16.7k+ mock tokens of room on
  both sides. The memory carrier's own legal smaller form brings the request under the line while keeping the Body Map
  and full toolset intact; A2 starts folded and passes in one POST.
* ``standing`` (S) = 120,000, between the refused folded-memory Max form (~172k) and Nano's standing toolset (~73k):
  both full Max and folded Max are refused; Nano's standing toolset rescues the turn over HTTP while keeping the
  resident Body Map, BIBLE and owner words intact. A2 starts folded (skipping full Max) and steps to standing tools.
* ``none`` (N) = 40,000, strictly below Nano's smallest legal standing-tool form (~73k, 30k+ clearance): nothing
  legal passes, so nothing mandatory (BIBLE, owner words, Body Map, standing tools) may be dropped to pass. Every room
  still sends, walks down to the standing toolset, is refused and ends its turn ``failed``.

``CLEARANCE`` is asserted from the recorded sizes of every scenario's own run: at a scenario's FIXED line, no request
that belongs to a turn may sit inside 10,000 mock tokens of it (stage R's line is derived from the run, so it is read
from the same recorded sizes rather than guarded by this floor). The same stand measured up to 15,703 mock tokens of
spread between two new chats' first sends in ONE run (558,651 against 574,354), so a larger floor would be a threshold
cut by hope rather than by evidence — and a line that drifts onto a form is exactly what this check refuses to hide.
The line E used to carry (500,000) sat 1,975 under a measured accepted first send, and the single MARGIN of 50,000
claimed a distance this fixture had never demonstrated at E; MARGIN keeps its place only at M, where the refused
smaller Max form is measured 105,000+ over the line. Both pricing errors were fixture defects, and the four baseline
and two candidate test failures they produced say nothing about the feature.

The synthetic carrier is laid here by ``lay_authored_pages``, the way ``tests._stand.lay_rooms`` lays a story (real
chat rows, one published page per closed room, the same needles) with the one thing a modern install's pages carry and
that story does not: ``metadata.lead_rule`` of the delivered procedure, and a short authored first sentence over a long
body. That provenance is what makes a page's lead an ordinary telling (``memory_forms.stage_of`` answers
``story_ordinary`` with it and puts the same form below the telling without it), so the carrier's legal smaller form
gives up the 20 bodies and keeps the 20 leads instead of only moving counters. A2's owner words are as long as A1's in
E, so in E nothing but that ordinary form can make its first send smaller; in M they are a little shorter, so the
smaller Max form A2 could send is a few tokens under A1's refused one and still over the line: only the maps may be
accepted first.

WHICH CLAIM BELONGS TO WHOM, kept apart on purpose:

* ``cold_walk_and_control`` — true on every tree, and about LEGACY recovery only: A1 meets the refusal with nothing
  learned and walks the ladder to a rung that the provider really accepts, B1 on another key of the same runtime does
  the same, the HTTP decisions follow the line, the reply cap never moves, the mandatory text survives, and the
  refused first send carries the whole carrier. It deliberately asks nothing about the authored leads of the send that
  recovery ends with: measured on both trees, the post-refusal emergency walk gives all 20 tellings up, so a lead
  floor there was a claim about recovery that never held — and a cold walk learns nothing, so it could not have said
  anything about the route hint either.
* the ``A2`` tests — the new promise: the FIRST send of a fresh chat on the account that just refused is the shape
  the refusal taught, and before it retries anything it still carries the authored leads and the mandatory text. It
  is the window's only POST besides the documented Light draft (a hidden, leading or extra one fails by name), and the
  turn then ends as an ordinary answer: a completed durable result that holds the known synthetic answer, not just an
  HTTP 200 (``assert_answered``). The baseline's expected failure is reached only after an HTTP 400 first POST and a
  later accepted retry that produced that answer; additional POSTs after an accepted first send are a different failure.
* ``test_none_a3_...`` — a regression guard, true on every tree today and built to fail on a route that re-inflates
  a fresh chat's first send: after N's own line has done its work, one named stage (``stage R``) restores the key to
  A and moves the line to the size this account's route itself last put on the wire plus the measured spread between
  two new chats' first sends. A fresh chat that grows its first send back beyond that is refused again by the same
  HTTP path, and the chat that is admitted ends as the same ordinary answer. Nothing is written beforehand and B1's
  evidence is taken before the line moves.

The shared ``e2e_clone`` runs committed HEAD, so an uncommitted candidate is first committed in an isolated clone by
its operator, as in the other scenarios of this package. The module carries the package's own ``integration`` marker
and the conftest scenario ceiling, so a default local run neither boots this server nor waits without a name.

Run (one scenario, no other lane)::

    OUROBOROS_E2E_DEEP=mock python -m pytest -o addopts="" -p no:cacheprovider \\
        tests/system_e2e/test_new_chat_after_unitless_length_refusal.py -q
"""

from __future__ import annotations

import io
import json
import math
import pathlib
import signal
import threading
import time
import uuid

import pytest

from devtools.benchmarks.common.server_runner import _api_status
from tests._first_contact import ANSWER
from tests._stand import CHARS_PER_TOKEN, FIRST_ROOM, room_needle
from tests.system_e2e.conftest import SCENARIO_TIMEOUT_SEC
from tests.system_e2e.harness import (
    LANE_MOCK,
    MOCK_SLUG,
    ArtifactOracle,
    ScriptedStubModel,
    keyless_settings,
    message_text,
    require_lane,
    start_server,
    wait_durable_result,
    wait_until,
    ws_url,
)

# Server-booting scenarios: out of the default local battery (``integration``) and bounded by the package's own
# scenario ceiling, so a hung scenario fails BY NAME instead of starving the lane (tests/system_e2e/conftest.py).
# The signal-vs-thread method is the same platform choice that conftest and test_stand_server.py make locally; the
# number itself has one home, and this module adds no second copy of it.
_TIMEOUT_METHOD = "signal" if hasattr(signal, "SIGALRM") else "thread"
pytestmark = [pytest.mark.integration, pytest.mark.timeout(SCENARIO_TIMEOUT_SEC, method=_TIMEOUT_METHOD)]

BYTES_PER_TOKEN = 3  # the loopback model's tokenizer: JSON bytes of messages and tools / 3 (as tests/_first_contact_server.py)
MOCK_MODEL, MOCK_CHAT_PATH = MOCK_SLUG.split("::", 1)[1], "/v1/chat/completions"
KEY_A = "synthetic-transport-key-A-3896-not-a-credential"
KEY_B = "synthetic-transport-key-B-3896-not-a-credential"
#: The fixed stock reply allowance every ordinary owner send must carry, on BASE and HEAD alike. This is the test's own
#: expected wire value, not an imported implementation constant: verified as MAIN_LOOP_MAX_TOKENS in
#: ouroboros/loop_llm_call.py at dfce82f2593a5f40ace4c95429f35b6090f47835 and 54f846740b231e0e88d54d5e44622cd24cff85eb.
#: The keyless OpenAI-compatible mock publishes no response maximum, so ResponseLimit.ceiling leaves the requested
#: allowance unchanged and llm_openai_compatible carries it as ``max_tokens``.
EXPECTED_REPLY_CAP = 65_536
#: Z.ai's own numberless body (live refusal of 2026-11-20): HTTP 400, code "1261", exactly this message, no number.
ZAI_REFUSAL = {"error": {"code": "1261", "message": "Prompt too long", "type": "invalid_request_error"}}
#: Rooms that have never spoken: the owner's first frame in each creates the conversation.
CHAT = {"A1": 7101, "A2": 7102, "B1": 7103, "A3": 7104}
#: Which rooms each scenario speaks, in order. A3 is the negative control's recovery stage only, and it is the LAST
#: room of every scenario, so the control key's evidence is always taken before anything moves.
ROOMS = {"enough": ("A1", "A2", "B1"), "standing": ("A1", "A2", "B1"), "none": ("A1", "A2", "B1", "A3")}
#: The synthetic transport key each room's own sends must ride, asserted after ``partition`` chose them.
TRANSPORT_KEY = {"A1": KEY_A, "A2": KEY_A, "B1": KEY_B, "A3": KEY_A}
#: The owner's words: a room's short marker (``room_marker``) opens them and the synthetic title below pads them. The
#: recovery stage speaks them by the same rule, its own line read from what THIS run refused.
BASE_WORDS = "Привет! Ты на связи? Проверь контекст, пожалуйста."
TITLE_UNIT = " Синтетический заголовок комнаты номер 3896."  # 46 bytes of padding per unit, nothing else differs
TITLE_UNITS = {"enough": {"A1": 200, "A2": 200, "B1": 200, "A3": 200},
               "standing": {"A1": 200, "A2": 200, "B1": 200, "A3": 200},
               "none": {"A1": 200, "A2": 10, "B1": 200, "A3": 10}}
BIBLE_KEPT = "owed on body changes"  # a BIBLE.md line: mandatory text, never folded
BODY_MAP_HEADING = "## Body"  # resident Body Map section heading (detailed or compact parts list)
BODY_MAP_MARK = "docs/architecture/05-supervisor-loop.md"  # permanent chapter address carried in the Body Map
#: A line only the Development book carries (docs/development/05) — recorded and printed, never asserted.
DEV_MARK = "Hermetic preflight isolation"
#: The documented post-turn Light draft opens with one of these two sentences, verbatim from ouroboros/memory_fallback.py
#: (``_PAGE``, line 65; ``_PART``, line 70), and is a lone tool-less user message. It is the ONLY kind of request a
#: room's own sends are ever set apart from; everything else unexplained stays in view (``partition``, ``check_wire``).
LIGHT_DRAFT_OPENINGS = ("You are a helper drafting one memory page for Ouroboros from the rows below.",
                        "You are a helper drafting one memory part for Ouroboros:")
#: One sentence saying what the room settled and what it left open: the authored lead a telling keeps. It carries no
#: needle, so a send that stopped above the openings still shows the lead and none of the body.
LEAD_UNIT = "Room {room} closed its arc: the work it shipped is told here, one follow-up is still open."
BODY_FILLER = "What this room settled, and why it matters to the rest of the work. "  # as tests/_stand.lay_rooms
STORY_ROOMS, STORY_TOKENS = 20, 100_000  # synthetic story: its weight is what the memory carrier can give back
#: Fixed lines, priced from the measured forms under the unified resident Body Map + Reference Atlas:
#: full Max ~310.7k, hinted folded Max ~276.8k (cold walk ~172.8k), Nano standing toolset ~74.5k..83.0k.
THRESHOLD = {"enough": 294_000, "standing": 120_000, "none": 40_000}
CLEARANCE = 10_000  # no turn's request may sit inside this of its scenario's line; asserted from this run's sizes
#: Stage R's line: the size this account's own route last put on the wire, plus the spread the same stand measured
#: between two new chats' first sends in ONE run (558,651 against 574,354: 15,703 mock tokens) taken twice over, so
#: the guard cannot cry wolf on noise while a re-inflated story (~133,000 mock tokens here) still cannot pass it.
RECOVERY_SPREAD = 30_000
#: How long a window waits for a POST still in flight before it fails by name. The loopback model answers at once, so
#: this bounds a broken request; it is never a pace.
SETTLE_TIMEOUT_SEC = 60


def room_marker(room: str) -> str:
    """The short, stable head of a room's own owner message. A title cut short or reworded takes the rest of the words
    away, never this: the room's send is still found, and the whole-words check then names what changed."""
    return f"{room}-3896"


def owner_words(scenario: str, room: str) -> str:
    return f"{room_marker(room)} {BASE_WORDS}" + TITLE_UNIT * TITLE_UNITS[scenario][room]


def page_lead(room: int) -> str:
    return LEAD_UNIT.format(room=room)


def page_text(room: int, body_chars: int) -> str:
    """The authored lead first, then the long body: the needle inside the body says whose room a send still holds
    whole, while the lead sentence says whose room is still told."""
    body = room_needle(room) + ": " + (BODY_FILLER * (body_chars // len(BODY_FILLER) + 1))[:body_chars]
    return f"{page_lead(room)} {body}"


def lay_authored_pages(root: pathlib.Path, rooms: int, story_tokens: int) -> dict:
    """Closed arcs of ``rooms`` live rooms, each sealed by ONE page, through the real ChronicleStore.

    Rows and records go in the way ``tests._stand.lay_rooms`` and ``tests._turn_shapes`` lay a story — the store's own
    statements for the rows, ``_turn_shapes._page`` for the record, one publication for the batch — and only the
    record's ``metadata`` is added: ``lead_rule`` at ``LEAD_RULE_VERSION``, the revision of the lead instruction a
    modern page's writer was shown. That provenance is what lets the memory carrier's own legal smaller form (an
    ordinary telling, ``story_ordinary``) yield a body and keep its lead; a page without it cannot be folded inside
    the telling at all, and the rungs below the telling give the lead up too. Nothing is patched: no planner, no
    reader or writer, no memory policy, no harness, and no product or mock target is added for this.
    """
    from ouroboros.chronicle_lead import LEAD_RULE_VERSION
    from ouroboros.chronicle_store import ChronicleStore
    from tests import _turn_shapes as shapes

    body_chars = max(80, story_tokens * CHARS_PER_TOKEN // max(1, rooms))
    open_rows: list[dict] = []
    paged: dict[int, list[dict]] = {}
    for index in range(rooms):
        chat_id = FIRST_ROOM + index
        open_rows += shapes._rows(chat_id, shapes.OPEN_ROWS, "open")
        paged[chat_id] = shapes._rows(chat_id, 2, "paged")
    shapes.append_rows(root, open_rows)
    ids = shapes.append_rows(root, [row for rows in paged.values() for row in rows])
    records = [{**shapes._page(chat_id, ids[index * 2:index * 2 + 2], page_text(chat_id, body_chars)),
                "metadata": {"lead_rule": LEAD_RULE_VERSION}} for index, chat_id in enumerate(paged)]
    published = ChronicleStore(root).publish(records)
    if not published.ok:
        raise RuntimeError(f"the rooms' authored story did not land: {published}")
    return {"rooms": rooms, "pages": len(records), "story_chars": sum(len(record["text"]) for record in records)}


class WireLedger:
    """Every POST the loopback model receives, one record each, in the order the POSTs ARRIVED.

    A record is appended as soon as the request's body is in hand and before any reply is written, so a slow or refused
    request can neither move behind a later one nor fall out of the count a window is cut by; the HTTP status the model
    really wrote joins the SAME record when the reply is done (``settle``). A window is read only when every record in
    it is settled: a request still in flight is waited for, and a wait that expires fails by name instead of handing
    out a half-written record.
    """

    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._changed = threading.Condition()

    def arrive(self, raw: bytes, auth: str = "", path: str = "") -> dict:
        row = wire_row(raw, auth, path)
        with self._changed:
            row.update(seq=len(self._rows), status=None, settled=False)
            self._rows.append(row)
        return row

    def settle(self, row: dict, status: int | None) -> None:
        with self._changed:
            row.update(status=status, settled=True)
            self._changed.notify_all()

    def count(self) -> int:
        with self._changed:
            return len(self._rows)

    def settle_point(self, timeout: float = SETTLE_TIMEOUT_SEC) -> int:
        """The count at a moment when no recorded POST is in flight: every record up to it has its HTTP outcome."""
        deadline = time.monotonic() + timeout
        with self._changed:
            while pending := [row for row in self._rows if not row["settled"]]:
                left = deadline - time.monotonic()
                assert left > 0, f"{len(pending)} POST(s) still in flight after {timeout:g}s: {describe(pending)}"
                self._changed.wait(left)
            return len(self._rows)

    def window(self, start: int, stop: int) -> list[dict]:
        """EVERY POST between two counts, in arrival order, as settled copies: none filtered, none half-written. Which
        of them are a room's own turn is ``partition``'s decision, read from what each request is."""
        with self._changed:
            rows = self._rows[start:stop]
            pending = [row for row in rows if not row["settled"]]
            assert not pending, f"{len(pending)} POST(s) of the window are not settled: {describe(pending)}"
            return [dict(row) for row in rows]


class ZaiLikeModel(ScriptedStubModel):
    """Loopback OpenAI-compatible model that refuses like Z.ai and reports no usage; one record per POST it received.

    ``refuse_at`` is read when each request is answered, so a named later stage can move the line: the value is the
    provider's, never a per-room patch, and the HTTP decision stays the model's own. Every POST is recorded in arrival
    order and settled with the status the model really wrote (``WireLedger``).
    """

    def __init__(self, *, refuse_at: int) -> None:
        super().__init__(final_answer=ANSWER)
        self.refuse_at = refuse_at
        self.ledger = WireLedger()
        base, outer = self._server.RequestHandlerClass, self

        class Handler(base):
            status = 0

            def send_response(self, code, message=None):  # noqa: D102 - stdlib hook: the status really written
                self.status = int(code)
                super().send_response(code, message)

            def do_GET(self):  # noqa: N802
                if self.path.rstrip("/").endswith("/models"):  # lists the model and publishes NO window
                    data = json.dumps({"object": "list", "data": [{"id": MOCK_MODEL, "object": "model"}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_error(404)

            def do_POST(self):  # noqa: N802
                self.status = 0  # this request's own outcome, never an earlier one's on a kept-alive connection
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                row = outer.ledger.arrive(raw, self.headers.get("Authorization") or "", self.path)  # arrival order
                try:
                    if row["n_prompt"] >= outer.refuse_at:  # the line as it stands for THIS request
                        data = json.dumps(ZAI_REFUSAL).encode()
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                    else:
                        original, self.rfile = self.rfile, io.BytesIO(raw)
                        try:
                            super().do_POST()
                        finally:
                            self.rfile = original
                finally:
                    outer.ledger.settle(row, self.status or None)  # the status really written, in the SAME record

        self._server.RequestHandlerClass = Handler

    def _completion(self, body: dict) -> dict:
        completion = super()._completion(body)
        completion["usage"] = None  # a provider that reports nothing: no fake prompt_tokens for the product to anchor on
        return completion

    def count(self) -> int:
        return self.ledger.count()

    def settle_point(self) -> int:
        """The count at a moment when no POST is in flight (see ``WireLedger.settle_point``)."""
        return self.ledger.settle_point()

    def window(self, start: int, stop: int) -> list[dict]:
        """EVERY POST the model received between two counts, in arrival order, settled: no filter of any kind."""
        return self.ledger.window(start, stop)


def wire_row(raw: bytes, auth: str = "", path: str = "") -> dict:
    """One POST as the loopback model received it: size (the line's own measure), reply cap, tool schemas, and the
    message roles with the text of each user-role input. The system view and the memory ride in the system message;
    only a user-role input can be the owner's own message. A body that is no JSON object is still a POST that arrived:
    it is recorded with no messages, and no room owns it."""
    try:
        body = json.loads(raw or b"{}")
    except ValueError:
        body = {}
    body = body if isinstance(body, dict) else {}
    sized = json.dumps({"messages": body.get("messages"), "tools": body.get("tools")},
                       ensure_ascii=False).encode("utf-8")
    messages = [message for message in (body.get("messages") or []) if isinstance(message, dict)]
    return {"auth": auth, "path": path, "model": body.get("model"),
            "n_prompt": math.ceil(len(sized) / BYTES_PER_TOKEN), "raw_bytes": len(raw),
            "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens"),
            "tools": len(body.get("tools") or []), "text": raw.decode("utf-8", "replace"),
            "roles": tuple(str(message.get("role") or "") for message in messages),
            "user_texts": tuple(message_text(message) for message in messages if message.get("role") == "user")}


def describe(rows: list[dict]) -> str:
    """Arrival number, reply cap, tool count, roles, size and HTTP status of each request: a shape for a failure
    message, never a key and never a body."""
    return "; ".join(
        f"#{row['seq']} cap={row['max_tokens']} tools={row['tools']} roles={'+'.join(row['roles']) or '-'}"
        f" mock_tokens={row['n_prompt']} bytes={row['raw_bytes']} http={row['status']} path={row.get('path') or '-'}"
        for row in rows)


def mark(row: dict, words: str) -> dict:
    """The carrier's own states read from the bytes the model was given: bodies still whole (the needle), the authored
    leads a telling keeps, the resident Body Map beside them, and the mandatory text named apart. ``owner_words`` is
    the room's WHOLE words inside a user-role input: a copy elsewhere in the request does not count."""
    text = row["text"]
    return {**{k: v for k, v in row.items() if k not in ("text", "user_texts")},
            "owner_words": any(words in part for part in row["user_texts"]),
            "body_map": BODY_MAP_HEADING in text and BODY_MAP_MARK in text, "bible": BIBLE_KEPT in text,
            "dev": DEV_MARK in text,
            "bodies": sum(room_needle(FIRST_ROOM + i) in text for i in range(STORY_ROOMS)),
            "leads": sum(page_lead(FIRST_ROOM + i) in text for i in range(STORY_ROOMS))}


def is_light_draft(row: dict) -> bool:
    """The one documented auxiliary request of a turn's window: the post-turn Light memory draft. Its whole shape is
    named — a lone user message, no tool schemas, one of the two openings memory_fallback writes first — so a turn send
    that lost its schemas, or whose owner message was cut, is never taken for it, and its own reply cap (16,384 against
    the turn's 65,536, measured here) is not what leaves it out."""
    return row["roles"] == ("user",) and not row["tools"] and row["user_texts"][0].startswith(LIGHT_DRAFT_OPENINGS)


def has_owner_input(row: dict, room: str) -> bool:
    """The room's own message reached the model as an input of the user: a user-role message carrying the room's
    marker. The system view and the memory may hold copies of the owner's words; a copy is not the message."""
    return any(room_marker(room) in part for part in row["user_texts"])


def partition(rows: list[dict], room: str, words: str) -> dict:
    """Every POST of one turn's window, each in exactly one named place and none dropped: ``light`` (the documented
    draft, decided first by what it is), ``owner`` (carries the room's own message as a user-role input — chosen by that
    alone, never by its reply cap, its size, its schemas or whether the whole title survived) and ``unowned`` (everything
    else: no request is excused as harmless, ``check_wire`` fails on any)."""
    parts: dict[str, list[dict]] = {"light": [], "owner": [], "unowned": []}
    for row in rows:
        kind = "light" if is_light_draft(row) else "owner" if has_owner_input(row, room) else "unowned"
        parts[kind].append(mark(row, words))
    return parts


def terminal_class(got: dict) -> str:
    """The class a turn's durable result ended in, read from the stored task record (``status`` and ``result``):
    ``answered`` is a completed turn whose result holds the known synthetic answer; ``failed`` is the host's failed
    terminal (``agent_task_pipeline._durable_terminal_status``: a failed or infra_failed execution axis, where a size
    refusal no recovery undid ends); anything else is named as it is, so a surprise says what it was."""
    status, has_answer = str(got.get("status") or "unknown"), ANSWER in str(got.get("text") or "")
    if status == "completed":
        return "answered" if has_answer else "completed-without-answer"
    return f"{status}-with-answer" if has_answer else status


def table(result: dict) -> str:
    """Every POST of every room's window, in arrival order: the room's own turn sends numbered, the documented Light
    drafts and everything unowned listed apart, plus the model's raw count per window and each turn's terminal class —
    the whole evidence a failing assert carries. HTTP decision, raw byte size, projected size, cap, book/memory markers;
    never a payload."""
    lines = []
    for room, got in result["rooms"].items():
        light, unowned = got.get("light", []), got.get("unowned", [])
        lines.append(f"{room} chat={CHAT[room]} turn={got['status']!r} class={terminal_class(got)} window={got['window']} POSTs"
                     f" (this turn: {len(got['posts'])}) light={len(light)} unowned={len(unowned)}")
        if got.get("recovered"):
            lines.append(f"  stage R: line moved to {result['recovered_line']} after {result['refused_before_recovery']} was refused on key A")
        for n, row in enumerate(got["posts"], 1):
            lines.append(
                f"  #{n} seq={row['seq']} http={row['status']} cap={row['max_tokens']} mock_tokens={row['n_prompt']} bytes={row['raw_bytes']}"
                f" body_map={row['body_map']}"
                f" tools={row['tools']} bible={row['bible']} words={row['owner_words']}"
                f" story_bodies={row['bodies']} story_leads={row['leads']} key={'A' if row['auth'] == 'Bearer ' + KEY_A else 'B' if row['auth'] == 'Bearer ' + KEY_B else '?'}"
                f" dev={row['dev']} path={row['path']!r} model={row['model']!r}")
        for kind, listed in (("light", light), ("unowned", unowned)):
            lines += [f"  {kind} seq={row['seq']} http={row['status']} cap={row['max_tokens']} mock_tokens={row['n_prompt']}"
                      f" bytes={row['raw_bytes']} tools={row['tools']} roles={'+'.join(row['roles']) or '-'}"
                      f" path={row.get('path') or '-'}" for row in listed]
    return "threshold={threshold} mock tokens, {n} POSTs on the wire ({boot} before the first owner frame)\n".format(
        threshold=result["threshold"], n=result["posts_total"], boot=result.get("boot", 0)) + "\n".join(lines)


def run_scenario(clone: pathlib.Path, root: pathlib.Path, scenario: str) -> dict:
    """The scenario's new rooms on one running server; each turn's wire read from the model's own record."""
    refuse_at = THRESHOLD[scenario]  # the scenario's fixed line, one and the same for baseline and candidate
    rooms_names = ROOMS[scenario]
    data = root / "server" / "data"
    data.mkdir(parents=True, exist_ok=True)
    lay_authored_pages(data, STORY_ROOMS, STORY_TOKENS)  # the memory carrier, before the server's first boot
    with ZaiLikeModel(refuse_at=refuse_at) as model:
        settings = keyless_settings(model, OUROBOROS_CONTEXT_MODE="max", OUROBOROS_REASONING_SUMMARY="off",
                                    OUROBOROS_CONSCIOUSNESS_SAFETY_MODE="off", OPENAI_COMPATIBLE_API_KEY=KEY_A,
                                    OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED=model.base_url)
        server = start_server(clone, root / "server", settings)
        recovered_line = refused_before = 0
        try:
            oracle = ArtifactOracle(server.data_root)
            rooms: dict[str, dict] = {}
            from websockets.sync.client import connect

            boot = cut = model.settle_point()  # the first window opens here: nothing the boot sent is a turn's send

            def speak(room: str, ws) -> None:
                nonlocal cut
                words = owner_words(scenario, room)
                message_id = "e2e-3896-" + uuid.uuid4().hex  # one unique client message id per new chat's frame
                ws.send(json.dumps({"type": "chat", "content": words,
                                    "client_message_id": message_id, "chat_id": CHAT[room]}))
                task = wait_until(lambda: _received_turn(oracle, message_id), 120)
                assert task and task.get("_is_direct_chat") is True, f"{room}: no owner turn was received: {task}"
                origin = (task.get("metadata") or {}).get("origin_message_ref") or {}
                assert origin.get("chat_id") == CHAT[room], (  # the stock door admitted the row into THIS new chat
                    f"{room}: the turn did not start from the owner's own WS-admitted row of this chat: {origin.get('chat_id')!r}")
                stored = wait_durable_result(oracle, task["id"], timeout=300)
                stop = model.settle_point()  # no POST in flight: every record up to here has its HTTP outcome
                rows = model.window(cut, stop)  # one turn at a time on this owned model: windows tile the arrivals
                cut = stop
                parts = partition(rows, room, words)  # every POST of the window, each in one named place, none dropped
                rooms[room] = {"status": stored.get("status"), "text": str(stored.get("result") or ""),
                               "posts": parts["owner"], "light": parts["light"], "unowned": parts["unowned"],
                               "window": len(rows), "task_id": task["id"],
                               "recovered": room == "A3"}

            with connect(ws_url(server), open_timeout=30, proxy=None) as ws, _FramePump(ws):
                for room in rooms_names:
                    if room == "B1":  # the owner's own save: same provider, URL and model, another transport key
                        _save_key(server, settings, KEY_B)
                        speak(room, ws)
                        continue
                    if room == "A3":
                        # ---- stage R: the negative control's named threshold change, AFTER N's own line has ----
                        # ---- produced every one of its refusals. The key returns to A through the owner's own    ----
                        # ---- settings route, and the new line is read from the size THIS account's own route    ----
                        # ---- last put on the wire (A2's first send) — not from a rung A1's emergency walk        ----
                        # ---- reached, and not written beforehand; B1's evidence is already taken. A fresh chat    ----
                        # ---- must not inflate its first send past that and be refused again over real HTTP.      ----
                        _save_key(server, settings, KEY_A)
                        a2_sends = rooms["A2"]["posts"]
                        assert a2_sends, "stage R needs A2's own send on this account to read its line from it"
                        refused_before = a2_sends[0]["n_prompt"]
                        recovered_line = refused_before + RECOVERY_SPREAD
                        model.refuse_at = recovered_line
                    speak(room, ws)
            return {"rooms": rooms, "threshold": refuse_at, "posts_total": model.count(), "boot": boot,
                    "base_url": settings["OPENAI_COMPATIBLE_BASE_URL"],
                    "recovered_line": recovered_line, "refused_before_recovery": refused_before}
        finally:
            server.stop()


def _save_key(server, settings: dict, key: str) -> None:
    """The owner's own save of another transport key, through the real route and checked on disk."""
    saved = _api_status(server.base_url, "POST", "/api/settings", {"OPENAI_COMPATIBLE_API_KEY": key}, timeout=120)
    assert saved.get("status") == 200, saved
    on_disk = json.loads(pathlib.Path(server.settings_path).read_text(encoding="utf-8"))
    assert on_disk.get("OPENAI_COMPATIBLE_API_KEY") == key, f"the key change to {key} was not saved"
    assert on_disk.get("OPENAI_COMPATIBLE_BASE_URL") == settings["OPENAI_COMPATIBLE_BASE_URL"]


def _received_turn(oracle: ArtifactOracle, client_message_id: str) -> dict | None:
    """The owner turn's task row, found by the id the owner's own frame carried (as the sibling scenarios read it)."""
    for row in oracle.events("task_received"):
        task = row.get("task") or {}
        origin = (task.get("metadata") or {}).get("origin_message_ref") or {}
        if origin.get("client_message_id") == client_message_id:
            return task
    return None


class _FramePump:
    """Every frame the SAME /ws delivers, drained on its own thread, so the SPA's socket never fills.

    ``websockets.sync`` allows ``send`` from one thread while another blocks in ``recv``. Nothing is asserted about
    these frames here: the turn's outcome is read from the durable record, and the pump only keeps the transport honest.
    """

    def __init__(self, ws) -> None:
        self.ws = ws
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)

    def _pump(self) -> None:
        from websockets.exceptions import ConnectionClosed

        while not self._stop.is_set():
            try:
                self.ws.recv(timeout=1.0)
            except TimeoutError:
                continue
            except ConnectionClosed:
                return

    def __enter__(self) -> "_FramePump":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(5)


# ---- the two scenarios --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scenario(e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    _selector_selfcheck()  # the oracle itself, on hand-made wire bodies, before a server is booted for any scenario
    _ledger_selfcheck()  # the recording and the terminal classes, on hand-made POSTs, likewise
    ran: dict[str, dict] = {}

    def run(name: str) -> dict:
        if name not in ran:
            ran[name] = run_scenario(e2e_clone, tmp_path_factory.mktemp(f"newchat-{name}"), name)
            print(table(ran[name]))  # evidence only: exactly one wire table per scenario (E/M/N); HTTP/bytes/key letters, no payloads
        return ran[name]

    return run


def check_wire(result: dict) -> str:
    """What must hold of EVERY request a scenario's turns put on the wire, for every test that reads it — the negative
    control and stage R included. Which requests are a room's own turn was decided by what they are (``partition``);
    everything about the windows and the chosen requests is asserted here, so no send is excused by being unusual: every
    POST is settled, the windows tile the arrival order without a gap, nothing but the room's own sends and the
    documented Light draft is in a window (a hidden, leading or retried request fails by name instead of counting as
    harmless), and each own send carries the owner's whole words in the user's own message, the exact synthetic
    Authorization, the tool schemas and the fixed stock reply allowance on every send of every room — read off the wire,
    never used to choose a row. Returns the evidence table for the caller's own messages."""
    evidence = table(result)
    caps: set = set()
    follows = None  # the arrival number the next window has to open with
    for room, got in result["rooms"].items():
        rows = sorted((row for kind in ("posts", "light", "unowned") for row in got[kind]), key=lambda row: row["seq"])
        pending = [row for row in rows if not row["settled"]]
        assert not pending, f"{room}: {len(pending)} POST(s) have no HTTP outcome yet: {describe(pending)}\n{evidence}"
        seqs = [row["seq"] for row in rows]
        run = seqs == list(range(seqs[0], seqs[0] + len(seqs))) if seqs else True
        assert run and (follows is None or not seqs or seqs[0] == follows), (
            f"{room}: the window is not one gap-free run of arrivals that follows the previous turn's: {seqs}\n{evidence}")
        follows = seqs[-1] + 1 if seqs else follows
        assert not got["unowned"], (
            f"{room}: {len(got['unowned'])} request(s) in this turn's window carry no owner message in a user-role input and "
            f"are no documented Light draft (a hidden, leading, retried or message-less send): {describe(got['unowned'])}\n{evidence}")
        assert got["posts"], f"{room}: no request carried the room's own owner message (window={got['window']})\n{evidence}"
        for n, row in enumerate(got["posts"], 1):
            assert row["path"] == MOCK_CHAT_PATH and row["model"] == MOCK_MODEL, (
                f"{room} #{n}: wrong HTTP endpoint or resolved model\n{evidence}")
            assert row["owner_words"], f"{room} #{n}: the owner's whole words are not in the user's own message (title cut or reworded)\n{evidence}"
            assert row["tools"], f"{room} #{n}: this send carries no tool schemas\n{evidence}"
            assert row["auth"] == f"Bearer {TRANSPORT_KEY[room]}", (
                f"{room} #{n}: wrong or missing synthetic Authorization\n{evidence}")
            assert row["max_tokens"], f"{room} #{n}: this send names no reply cap\n{evidence}"
        room_caps = {row["max_tokens"] for row in got["posts"]}
        assert len(room_caps) == 1, f"{room}: the reply cap moved inside one turn {sorted(room_caps)}\n{evidence}"
        assert room_caps == {EXPECTED_REPLY_CAP}, (
            f"{room}: the reply cap is not the fixed stock allowance {EXPECTED_REPLY_CAP}: {sorted(room_caps)}\n{evidence}")
        caps |= room_caps
    assert len(caps) == 1, f"the reply cap moved between rooms {sorted(caps)}\n{evidence}"
    return evidence


def assert_answered(result: dict, room: str, evidence: str) -> None:
    """The turn ended as an ordinary answer, not merely with an HTTP 200: its durable result is completed and holds the
    known synthetic answer, and the room's last send, the one that answer came back on, was accepted."""
    got = result["rooms"][room]
    assert terminal_class(got) == "answered", (
        f"{room}: the turn ended {terminal_class(got)!r}, not as the model's own answer\n{evidence}")
    assert got["posts"][-1]["status"] == 200, f"{room}: its last send was not the accepted one\n{evidence}"


def assert_a2_single_send(result: dict, evidence: str, why: str) -> None:
    """Only a refused first POST with an accepted retry earns the baseline's pinned failure signature."""
    posts = result["rooms"]["A2"]["posts"]  # check_wire has already established that these are its ordered own POSTs
    if len(posts) > 1:
        assert posts[0]["status"] == 400, f"A2 sent again but its first POST was not HTTP 400\n{evidence}"
        assert_answered(result, "A2", evidence)
    assert len(posts) == 1, f"A2 re-sent before settling ({why})\n{evidence}"


def _selector_selfcheck() -> None:
    """The selector and ``check_wire`` against hand-made wire bodies — no server, no network. The documented Light draft
    is left out wherever it lands and listed; a title cut short, a retry under another reply cap and a send without
    schemas are each still CHOSEN and fail their named assertion; and every other request that carries no owner message
    fails by name instead of counting as harmless: a turn-shaped one that lost the owner's message (its words survive
    only as a copy in the system view), a marker-less tool-less one that LEADS the first send, a differently capped
    marker-less retry, and a look-alike of the Light draft that is not its shape. A wrong or missing Authorization and a
    uniformly changed reply cap are still CHOSEN and fail their named assertions."""
    room, words = "A1", owner_words("enough", "A1")
    schemas = [{"type": "function", "function": {"name": "probe", "parameters": {"type": "object"}}}]
    view = {"role": "system", "content": f"memory view of the room: {words}"}  # a copy of the words, not the message

    def post(*messages, cap=EXPECTED_REPLY_CAP, tools=schemas, model=MOCK_MODEL, path=MOCK_CHAT_PATH):
        return messages, cap, tools, model, path

    def window(*posts, auth=f"Bearer {KEY_A}"):
        """The hand-made POSTs arrive in the order given, each answered 200, and are read as one window of ``room``."""
        ledger = WireLedger()
        for messages, cap, tools, model, path in posts:
            body = {"model": model, "messages": list(messages), "max_tokens": cap, **({"tools": tools} if tools else {})}
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            ledger.settle(ledger.arrive(raw, auth, path), 200)
        got = partition(ledger.window(0, ledger.settle_point(1)), room, words)
        result = {"rooms": {room: {"status": "completed", "text": ANSWER, "window": len(posts), "posts": got["owner"],
                                   "light": got["light"], "unowned": got["unowned"]}},
                  "threshold": THRESHOLD["enough"], "posts_total": len(posts)}
        return result, got

    own = post(view, {"role": "user", "content": words})
    draft = post({"role": "user", "content": LIGHT_DRAFT_OPENINGS[0] + "\n[row:1] " + words}, cap=16_384, tools=None)
    for order in ((own, draft), (draft, own)):  # a turn's late phase can reach its window before or after its own send
        result, got = window(*order)
        assert [len(got[kind]) for kind in ("owner", "light", "unowned")] == [1, 1, 0], "the draft is left out, the turn kept"
        assert "light=1" in check_wire(result), "the draft is listed in the evidence"
    for label, wrong, expect in (
            ("title cut short", post(view, {"role": "user", "content": words[:len(room_marker(room)) + 60]}), "whole words"),
            ("retry under another cap", post(view, {"role": "user", "content": words}, cap=32_768), "reply cap moved"),
            ("send without schemas", post(view, {"role": "user", "content": words}, tools=None), "no tool schemas"),
            ("other resolved model", post(view, {"role": "user", "content": words}, model="wrong-model"),
             "wrong HTTP endpoint or resolved model"),
            ("other HTTP endpoint", post(view, {"role": "user", "content": words}, path="/v1/responses"),
             "wrong HTTP endpoint or resolved model"),
            ("owner message lost, copy kept", post(view, {"role": "user", "content": "(empty message)"}),
             "carry no owner message")):
        result, got = window(own, wrong)
        assert len(got["owner"]) + len(got["unowned"]) == 2 and not got["light"], f"{label}: a request was dropped"
        with pytest.raises(AssertionError, match=expect):
            check_wire(result)
    for label, auth in (("wrong transport key", "Bearer synthetic-wrong-key-3896"),
                        ("missing transport key", "")):
        result, got = window(own, auth=auth)
        assert len(got["owner"]) == 1 and not got["unowned"] and not got["light"], f"{label}: the send was not chosen"
        with pytest.raises(AssertionError, match="wrong or missing synthetic Authorization"):
            check_wire(result)
    result, got = window(post(view, {"role": "user", "content": words}, cap=EXPECTED_REPLY_CAP - 1))
    assert len(got["owner"]) == 1 and not got["unowned"] and not got["light"], "uniformly changed cap: the send was not chosen"
    with pytest.raises(AssertionError, match="fixed stock allowance"):
        check_wire(result)
    look = LIGHT_DRAFT_OPENINGS[0] + "\n[row:1] other rows"  # the opening alone: no owner marker anywhere in it
    for label, order in (
            ("a marker-less tool-less request leads the first send",
             (post({"role": "user", "content": "Reply OK"}, cap=16, tools=None), own)),
            ("a differently capped marker-less retry follows it",
             (own, post({"role": "user", "content": "(empty message)"}, cap=32_768, tools=None))),
            ("a Light look-alike that carries tool schemas", (own, post({"role": "user", "content": look}, cap=16_384))),
            ("a Light look-alike beside a system message",
             (own, post(view, {"role": "user", "content": look}, cap=16_384, tools=None))),
            ("a Light look-alike that is not a lone message",
             (own, post({"role": "user", "content": look}, {"role": "user", "content": "more"}, cap=16_384, tools=None)))):
        result, got = window(*order)
        assert len(got["owner"]) == 1 and len(got["unowned"]) == 1 and not got["light"], f"{label}: not told apart"
        with pytest.raises(AssertionError, match="carry no owner message"):
            check_wire(result)


def _ledger_selfcheck() -> None:
    """The ledger, the window rules and the terminal classes against hand-made POSTs — no server, no network. Records
    keep ARRIVAL order whatever order the replies come in; a request still in flight is waited for and one that never
    settles fails by name; a window that lost a record in the middle fails as a gap; a send that arrives after its turn's
    window closed lands in the NEXT window, where nobody owns it, instead of vanishing; each terminal class is told
    apart."""

    def raw(room: str) -> bytes:
        return json.dumps({"messages": [{"role": "user", "content": f"{room_marker(room)} words"}], "max_tokens": 1}).encode()

    def result_of(room: str, rows: list[dict]) -> dict:
        got = partition(rows, room, owner_words("enough", room))
        return {"rooms": {room: {"status": "completed", "text": ANSWER, "window": len(rows), "posts": got["owner"],
                                 "light": got["light"], "unowned": got["unowned"]}},
                "threshold": THRESHOLD["enough"], "posts_total": len(rows)}

    ledger = WireLedger()
    first, second = ledger.arrive(raw("A1"), "Bearer k", "/v1"), ledger.arrive(raw("A1"), "Bearer k", "/v1")
    ledger.settle(second, 200)  # the later request is answered first
    ledger.settle(first, 400)
    rows = ledger.window(0, ledger.settle_point(1))
    assert [(row["seq"], row["status"]) for row in rows] == [(0, 400), (1, 200)], "records follow arrival, not the replies"
    slow = ledger.arrive(raw("A1"), "Bearer k", "/v1")
    timer = threading.Timer(0.05, ledger.settle, (slow, 200))
    timer.start()
    assert ledger.window(0, ledger.settle_point(5))[-1]["status"] == 200, "an in-flight POST is waited for, then read settled"
    timer.join()
    stuck = ledger.arrive(raw("A1"), "Bearer k", "/v1")
    with pytest.raises(AssertionError, match="still in flight"):
        ledger.settle_point(0.05)
    with pytest.raises(AssertionError, match="not settled"):
        ledger.window(0, ledger.count())
    ledger.settle(stuck, 200)
    kept = [row for row in ledger.window(0, ledger.count()) if row["seq"] != 1]  # a record gone from the middle
    with pytest.raises(AssertionError, match="gap-free"):
        check_wire(result_of("A1", kept))
    later = WireLedger()  # A1's own second send arrives after A1's window closed at one record
    for room in ("A1", "A1", "B1"):
        later.settle(later.arrive(raw(room), "Bearer k", "/v1"), 200)
    with pytest.raises(AssertionError, match="carry no owner message"):
        check_wire(result_of("B1", later.window(1, 3)))
    for status, text, expect in (("completed", ANSWER, "answered"), ("completed", "ok", "completed-without-answer"),
                                 ("failed", "context overflow", "failed"), ("failed", ANSWER, "failed-with-answer"),
                                 ("cancelled", "", "cancelled")):
        assert terminal_class({"status": status, "text": text}) == expect, (status, expect)
    for status, last, expect in (("failed", 400, "not as the model's own answer"), ("completed", 400, "not the accepted one")):
        with pytest.raises(AssertionError, match=expect):
            assert_answered({"rooms": {"A2": {"status": status, "text": ANSWER if status == "completed" else "",
                                              "posts": [{"status": last}]}}}, "A2", "")


def cold_walk_and_control(result: dict, *, accepted_tools: str) -> None:
    """Cold recovery and unaffected control: A1 meets the refusal with nothing learned, B1 is an unaffected control
    on another key of the same runtime, the upstream HTTP really decided, and mandatory text (BIBLE, owner words,
    resident Body Map) survives every send without changing the reply cap."""
    a1, a2, b1 = (result["rooms"][room]["posts"] for room in ("A1", "A2", "B1"))
    evidence = check_wire(result)  # owner's whole words, schemas and one reply cap on EVERY send of every room
    for room, posts, key in (("A1", a1, KEY_A), ("A2", a2, KEY_A), ("B1", b1, KEY_B)):
        got = result["rooms"][room]
        assert got["status"] == "completed" and ANSWER in got["text"], f"{room}: no usable answer\n{evidence}"
        assert all(row["auth"] == f"Bearer {key}" for row in posts), f"{room}: wrong transport key\n{evidence}"
        assert all(row["bible"] and row["body_map"] for row in posts), f"{room}: mandatory text or Body Map lost\n{evidence}"
        for row in posts:
            over = row["n_prompt"] >= result["threshold"]
            assert row["status"] == (400 if over else 200), f"{room}: HTTP does not follow the line\n{evidence}"
            assert abs(row["n_prompt"] - result["threshold"]) >= CLEARANCE, (
                f"{room}: a turn's request sits within {CLEARANCE} of the line {result['threshold']}; "
                f"the threshold must move with the measured forms\n{evidence}")
    # A1: the very first full Max request of the route is refused, then the walk is accepted in the named form.
    assert a1[0]["status"] == 400 and a1[0]["body_map"], evidence
    assert a1[-1]["status"] == 200 and a1[-1]["n_prompt"] < a1[0]["n_prompt"] and a1[-1]["body_map"], evidence
    if accepted_tools == "full":
        assert a1[-1]["tools"] == a1[0]["tools"] and b1[-1]["tools"] == b1[0]["tools"], evidence
    else:
        assert a1[-1]["tools"] < a1[0]["tools"] and b1[-1]["tools"] < b1[0]["tools"], evidence
    # B1: another account has taught the route nothing: it sends the same full Max first request as A1, then walks.
    assert b1[0]["status"] == 400 and b1[0]["body_map"] and b1[0]["tools"] == a1[0]["tools"], f"B1 changed class\n{evidence}"
    assert b1[0]["bodies"] == a1[0]["bodies"] == STORY_ROOMS, f"B1's first send was already folded\n{evidence}"
    assert a1[0]["raw_bytes"] <= b1[0]["raw_bytes"] < 1.1 * a1[0]["raw_bytes"], f"B1's first send is no full Max form\n{evidence}"
    assert b1[-1]["status"] == 200, evidence
    assert result["base_url"].startswith("http://127.0.0.1:"), evidence
    assert a1[0]["leads"] == a1[0]["bodies"] == STORY_ROOMS, f"A1's refused first send does not carry the whole carrier\n{evidence}"
    if accepted_tools == "full":
        for room, posts in (("A1", a1), ("B1", b1)):
            assert posts[-1]["bodies"] < STORY_ROOMS, f"{room}: no story body left the accepted folded Max send\n{evidence}"


def test_enough_cold_walk_and_control(scenario):
    """E: the smaller memory form (full toolset + Body Map intact) is the first accepted rung of A1 and B1."""
    cold_walk_and_control(scenario("enough"), accepted_tools="full")


def test_standing_cold_walk_and_control(scenario):
    """S: full Max and folded Max are both refused; Nano standing toolset rescues A1 and B1 with Body Map intact."""
    result = scenario("standing")
    cold_walk_and_control(result, accepted_tools="standing")
    a1 = result["rooms"]["A1"]["posts"]
    assert len(a1) >= 3 and a1[1]["status"] == 400 and a1[1]["tools"] == a1[0]["tools"] and a1[1]["n_prompt"] < a1[0]["n_prompt"], table(result)


def test_enough_candidate_a2_first_send_is_the_smaller_max_form(scenario):
    """Candidate, E: A2's first POST is the memory's ordinary telling, accepted at once with full tools and Body Map."""
    result = scenario("enough")
    a1, a2 = result["rooms"]["A1"]["posts"], result["rooms"]["A2"]["posts"]
    evidence = check_wire(result)
    assert_a2_single_send(result, evidence, "the route hint did not shape its first send")
    assert len(owner_words("enough", "A2")) >= len(owner_words("enough", "A1")), (
        f"A2's words must not be the shorter ones in E\n{evidence}")
    first = a2[0]
    assert first["status"] == 200 and first["n_prompt"] <= result["threshold"] - CLEARANCE, (
        f"A2's first send was not admitted with measured room\n{evidence}")
    assert first["raw_bytes"] < a1[0]["raw_bytes"], evidence
    assert a1[0]["bodies"] == a1[0]["leads"] == STORY_ROOMS, f"A1's refused first is not the whole carrier\n{evidence}"
    assert first["bodies"] < a1[0]["bodies"], f"the memory's long bodies are still in A2's first send\n{evidence}"
    assert first["leads"] == STORY_ROOMS, f"A2's first send fell below the authored leads\n{evidence}"
    assert first["body_map"] and first["tools"] == a1[0]["tools"], f"Body Map or full toolset lost when memory folding sufficed\n{evidence}"
    assert first["bible"] and first["owner_words"], evidence
    assert_answered(result, "A2", evidence)


def test_standing_candidate_a2_folds_memory_then_steps_to_standing_tools(scenario):
    """Candidate, S: A2 skips full Max by starting folded (20/20 leads + full tools), then rescues via standing tools."""
    result = scenario("standing")
    a1, a2 = result["rooms"]["A1"]["posts"], result["rooms"]["A2"]["posts"]
    evidence = check_wire(result)
    line = result["threshold"]
    assert a1[1]["status"] == 400 and a1[1]["n_prompt"] > line + CLEARANCE, evidence
    first, last = a2[0], a2[-1]
    assert first["status"] == 400 and first["raw_bytes"] < a1[0]["raw_bytes"], evidence
    assert first["bodies"] < a1[0]["bodies"] and first["leads"] == STORY_ROOMS, f"A2 first send did not fold tellings cleanly\n{evidence}"
    assert first["tools"] == a1[0]["tools"] and first["body_map"], evidence
    assert last["status"] == 200 and last["n_prompt"] < line - CLEARANCE, f"A2 did not recover under standing tools\n{evidence}"
    assert last["tools"] < first["tools"] and last["body_map"] and last["bible"] and last["owner_words"], evidence
    assert_answered(result, "A2", evidence)


def test_below_every_legal_form_nothing_mandatory_is_dropped_or_refused_unsent(scenario):
    """Negative control: below Nano's standing-toolset floor, BIBLE, owner words and Body Map are never dropped,
    and an earlier refusal never makes the next chat skip sending."""
    result = scenario("none")
    a1, a2, b1 = (result["rooms"][room]["posts"] for room in ("A1", "A2", "B1"))
    evidence = check_wire(result)
    assert len(a1) >= 3 and a1[-1]["tools"] < a1[0]["tools"] and a1[-1]["body_map"], f"A1 did not reach Nano standing tools\n{evidence}"
    smallest = min(row["n_prompt"] for row in a1)
    for room, posts in (("A1", a1), ("A2", a2), ("B1", b1)):
        assert posts, f"{room}: a refusal elsewhere made this chat send nothing\n{evidence}"
        assert all(row["status"] == 400 and row["n_prompt"] >= result["threshold"] for row in posts), evidence
        assert all(row["n_prompt"] - result["threshold"] >= CLEARANCE for row in posts), (
            f"{room}: a turn's send sits within {CLEARANCE} of the irreducible line\n{evidence}")
        assert all(row["bible"] and row["owner_words"] and row["tools"] and row["body_map"] for row in posts), (
            f"{room}: mandatory text or Body Map lost\n{evidence}")
        assert min(row["n_prompt"] for row in posts) >= smallest - 15_000, f"{room}: under the smallest legal form\n{evidence}"
        assert posts[0]["leads"] == STORY_ROOMS, f"{room}: its FIRST send fell below the authored leads\n{evidence}"
    for room in ("A1", "A2", "B1"):
        assert terminal_class(result["rooms"][room]) == "failed", (
            f"{room}: the turn ended {terminal_class(result['rooms'][room])!r}, not failed\n{evidence}")


def test_none_a3_after_the_line_recovers_does_not_inflate_its_first_send(scenario):
    """After every legal form was refused at N's line, stage R puts the account back on key A and raises the line to
    A2's refused first send plus RECOVERY_SPREAD. A3 must be admitted by that FIRST send without re-inflating bodies."""
    result = scenario("none")
    evidence = check_wire(result)
    a2 = result["rooms"]["A2"]["posts"]
    a3 = result["rooms"]["A3"]["posts"]
    assert result["recovered_line"] > result["threshold"], f"stage R never moved the line\n{evidence}"
    assert a3, f"A3: the recovery stage sent nothing\n{evidence}"
    assert a2, f"A2 sent nothing, so stage R has no reference to read\n{evidence}"
    assert all(row["auth"] == f"Bearer {KEY_A}" for row in a3), f"A3 did not run on the restored key A\n{evidence}"
    last_sent = a2[0]["n_prompt"]
    assert result["refused_before_recovery"] == last_sent, (
        f"stage R's line is not read from the send it follows\n{evidence}")
    first = a3[0]
    assert first["n_prompt"] <= last_sent + RECOVERY_SPREAD, (
        f"A3's first send re-inflated past the {last_sent} mock tokens this account's route itself just sent\n{evidence}")
    assert first["status"] == 200, f"A3's first send was refused again by the recovered line\n{evidence}"
    assert len(a3) == 1, f"A3 walked instead of starting where the refusal taught it\n{evidence}"
    assert first["bodies"] <= a2[0]["bodies"], (
        f"A3's first send carries more whole story bodies than the send this account's route itself just made "
        f"({a2[0]['bodies']} of {STORY_ROOMS} rooms still whole)\n{evidence}")
    assert first["leads"] == STORY_ROOMS, f"A3's first send fell below the authored leads\n{evidence}"
    assert first["bible"] and first["owner_words"] and first["tools"] and first["body_map"], evidence
    assert_answered(result, "A3", evidence)
    b1 = result["rooms"]["B1"]["posts"]
    assert b1[0]["status"] == 400 and b1[0]["body_map"] and b1[0]["bodies"] == STORY_ROOMS, f"B1's control changed\n{evidence}"
    assert all(row["auth"] == f"Bearer {KEY_B}" for row in b1), f"B1's evidence was not all taken on key B\n{evidence}"
