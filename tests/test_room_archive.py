"""The read-only archive of saved rooms while Main's history refuses for the Project registry (#3294).

Two-sided over one synthetic world staged as an older body left it (``tests._room_archive_world``: the old
zero-byte commit witness, rows through the store's own append, bindings and any registry as bytes, so no modern
registry, lock or backup ever exists to restore from). With a readable registry Main reads and the archive is not
needed; lost (or one row's identity unreadable), Main refuses with the typed field that offers the archive, and
the archive shows only plain rows and fixed notices positive evidence places: the owner's words typed in the room,
fixed notices written there, and in a recognized Project room the plain words of work its own binding places and
the owner message that binding names. Everything else stays unshown: Ouroboros's replies no binding places, typed
rows (questions, answers, deliveries, cards), a room no record names or records contradict, private contact,
place, hidden and A2A rows, rows without an exact stored room. Reads write nothing, never create or plan the
dialogue store, and pages continue honestly, also across a stretch with nothing to show; a continuation whose room's
selecting evidence changed answers 409 ``archive_cursor_changed``. Every word below is fictional.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_history_evidence
from ouroboros import dialogue_store as ds
from ouroboros import room_archive
from ouroboros.contracts.chat_id_policy import project_chat_id
from ouroboros.gateway import history
from ouroboros.gateway.chat_archive import make_chat_archive_endpoint
from ouroboros.gateway.history_contracts import ChatArchiveResponse
from ouroboros.gateway.schema import validate_ingress
from tests import _dialogue as dialogue_test
from tests._room_archive_world import TS, binding, lost_registry, project_row, write_bindings, write_registry
from tests._task_settings_helpers import isolated_runtime  # noqa: F401 -- the sealed install every case runs in

ORIGIN = "Please do the beta work"
REGISTRY = [project_row("alpha", "Alpha room"), project_row("beta", "Beta room"), project_row("gamma", "Gamma room")]


@pytest.fixture
def root(isolated_runtime):  # noqa: F811 -- the sealed install's data root
    return isolated_runtime


def _world(root, *, registry=False):
    """Main, a room a Main task was moved into (``beta``), a folderless room the owner typed in that Main's
    completion row names (``alpha``) and a room no saved record names (``gamma``); ``registry`` writes the
    readable registry an older body kept, else the registry is lost."""
    lost_registry(root)
    alpha, beta, gamma = (project_chat_id(pid) for pid in ("alpha", "beta", "gamma"))
    delta, epsilon = project_chat_id("delta"), project_chat_id("epsilon")  # rooms only a store folder could name
    owner = {"direction": "in", "task_id": "", "ts": TS}
    out = {"direction": "out", "ts": TS}
    host = {"direction": "out", "role": "system", "ts": TS}
    for row in [
        {**owner, "chat_id": 1, "text": "main owner", "client_message_id": "cm-main"},
        {**owner, "chat_id": 1, "text": ORIGIN, "client_message_id": "cm-beta"},
        {**out, "chat_id": 1, "task_id": "direct-1", "text": "main reply"},
        {**owner, "chat_id": alpha, "text": "alpha owner"},
        {**out, "chat_id": alpha, "task_id": "alpha-task", "text": "alpha reply"},
        {**out, "chat_id": 1, "task_id": "beta-task", "text": "beta reply before the move"},
        {**out, "chat_id": beta, "task_id": "beta-task", "text": "beta reply"},
        {**out, "chat_id": 1, "task_id": "beta-task", "type": "main_notice", "text": "beta notice to main"},
        {**host, "chat_id": 1, "task_id": "beta-task", "type": "project_handoff", "system_type": "project_handoff",
         "project_id": "beta", "project_name": "Beta room", "text": "Beta room"},
        {**host, "chat_id": 1, "task_id": "alpha-task", "type": "project_completion_summary",
         "system_type": "project_completion_summary", "project_id": "alpha", "project_name": "Alpha room",
         "text": "**Alpha room** · Done"},
        # Fixed notices keep their written room (positive); a bound work's typed rows are never shown (negative).
        {**host, "chat_id": 1, "task_id": "lost-task", "type": "task_not_started", "system_type": "task_not_started",
         "text": "main notice: not started"},
        {**host, "chat_id": beta, "task_id": "beta-task", "type": "steer_not_delivered",
         "system_type": "steer_not_delivered", "text": "beta notice: steer not delivered"},
        {**out, "chat_id": beta, "task_id": "beta-task", "type": "quiz", "text": "beta question",
         "quiz": {"quiz_id": "q1", "question": "beta question", "options": [{"label": "Yes"}]}},
        {**owner, "chat_id": beta, "task_id": "beta-task", "type": "quiz_answer", "text": "beta answer"},
        {**out, "chat_id": 1, "task_id": "beta-task", "type": "document", "text": "beta document caption"},
        {**out, "chat_id": beta, "task_id": "beta-task", "type": "card", "text": "beta card"},
        {**owner, "chat_id": gamma, "text": "gamma owner"},
        {**owner, "chat_id": delta, "text": "delta owner"},
        {**owner, "chat_id": epsilon, "text": "epsilon owner"},
        # Never an archive row: another audience, the hidden partition, A2A, no exact room, a task's or answer's.
        {**owner, "chat_id": 1, "text": "private contact", "presence_contact": {"place": "x"}},
        {**owner, "chat_id": 1, "text": "place arrival", "source": "presence:telegram"},
        {**owner, "chat_id": 0, "text": "hidden partition"},
        {**owner, "chat_id": -1001, "text": "a2a traffic"},
        {"direction": "in", "ts": TS, "text": "no stored chat"},
        {**owner, "chat_id": "1", "text": "string chat"},
        {**owner, "chat_id": 1, "task_id": "beta-task", "text": "owner steering a task"},
        {**owner, "chat_id": 1, "type": "quiz_answer", "text": "owner answer"},
        {**owner, "chat_id": True, "text": "boolean chat"},
    ]:
        ds.append(root, "chat", row)
    write_bindings(root, [binding("beta-task", "beta", origin=ORIGIN, origin_id="cm-beta")])
    if registry:
        write_registry(root, REGISTRY)
    return {"alpha": alpha, "beta": beta, "gamma": gamma}


def _archive(root, **params):
    response = asyncio.run(make_chat_archive_endpoint(root)(SimpleNamespace(query_params=params)))
    body = json.loads(response.body)
    if response.status_code == 200:
        assert not validate_ingress(body, ChatArchiveResponse), validate_ingress(body, ChatArchiveResponse)
    return response.status_code, body


def _main_history(root, **params):
    response = asyncio.run(history.make_chat_history_endpoint(root)(SimpleNamespace(query_params=params)))
    return response.status_code, json.loads(response.body)


def _encoded(value):
    return base64.urlsafe_b64encode(json.dumps(value).encode("utf-8")).decode("ascii").rstrip("=")


def test_pending_profile_reset_offers_no_previous_history(root):
    from supervisor.state_initialization import mark_pending

    lost_registry(root)
    mark_pending(root, origin="owner_reset")
    status, page = _main_history(root)
    assert status == 503 and page["messages"] == [] and "project_registry" not in page
    assert _archive(root) == (503, {"error": "archive_source_unavailable", "reason_code": "archive_source_unavailable"})


def _texts(payload):
    return [message["text"] for message in payload.get("messages") or []]


def _offered(root):
    status, main = _archive(root)
    assert status == 200 and main["state"] == "available", main
    return {room["project_id"] for room in main["rooms"] if room["room"] != "main"}


def _all_pages(root, room):
    texts, cursor, seen = [], None, set()
    for _ in range(200):
        status, page = _archive(root, room=room, **({"before": cursor} if cursor else {}))
        assert status == 200, page
        texts[:0] = _texts(page)  # each older page goes before the rows read already
        if not page["has_more"]:
            assert page["next_cursor"] is None
            return texts, page
        cursor = page["next_cursor"]
        assert cursor and cursor not in seen
        seen.add(cursor)
    pytest.fail("the archive cursor did not reach its end")


MAIN_ROWS = ["main owner", "Please do the beta work", "beta notice to main", "Beta room", "Alpha room · Done",
             "main notice: not started"]
BETA_ROWS = ["Please do the beta work", "beta reply before the move", "beta reply", "beta notice to main",
             "beta notice: steer not delivered", "owner steering a task"]
NEVER = {"main reply", "alpha reply", "gamma owner", "delta owner", "epsilon owner", "private contact", "place arrival",
         "hidden partition", "a2a traffic", "no stored chat", "string chat", "owner answer", "boolean chat",
         "beta question", "beta answer", "beta document caption", "beta card"}


def test_a_healthy_registry_needs_no_archive_and_main_reads_its_history(root):
    _world(root, registry=True)
    assert room_archive.archive_needed(root) is False
    assert _archive(root) == (200, {"state": "not_needed"})
    assert _archive(root, room="beta") == (200, {"state": "not_needed"})
    status, main = _main_history(root)
    assert status == 200 and "main owner" in _texts(main) and "project_registry" not in main


def test_a_lost_registry_refuses_main_and_the_archive_shows_only_what_evidence_places(root):
    rooms = _world(root)
    status, refused = _main_history(root)
    assert status == 503 and refused["reason_code"] == "history_source_unavailable"
    assert refused["project_registry"] == "unavailable" and refused["detail"] == "Project registry unavailable"
    assert room_archive.archive_needed(root) is True

    status, main = _archive(root)
    assert status == 200 and main["state"] == "available" and main["room"] == "main"
    assert _texts(main) == MAIN_ROWS, "the owner's own words and fixed notices, oldest first"
    assert main["complete"] is False and main["has_more"] is False and main["next_cursor"] is None
    assert main["limits"] == ["placement_unproven"]
    assert [message["role"] for message in main["messages"]] == ["user", "user", "assistant", "system", "system", "system"]
    for message in main["messages"]:  # read-only rows: no address, task, quiz or reply identity
        assert set(message) <= {"id", "role", "text", "ts", "kind", "attachments"} and message["id"].startswith("archive:")
    assert [message.get("kind") for message in main["messages"]] == [
        None, None, None, "project_handoff", "project_completion_summary", None], "a pinned row names its kind, nothing else"

    listed = {room["room"]: room for room in main["rooms"]}
    assert set(listed) == {"main", str(rooms["alpha"]), str(rooms["beta"])}, "gamma: no saved record names it"
    assert listed[str(rooms["beta"])] == {"room": str(rooms["beta"]), "chat_id": rooms["beta"], "project_id": "beta",
                                          "name": "Beta room", "evidence": ["binding", "main_record"]}
    assert listed[str(rooms["alpha"])]["evidence"] == ["main_record"]
    assert listed[str(rooms["alpha"])]["name"] == "Alpha room"

    status, beta = _archive(root, room=str(rooms["beta"]))
    assert status == 200 and _texts(beta) == BETA_ROWS, "its binding places the moved work's words, wherever written"
    status, alpha = _archive(root, room=str(rooms["alpha"]))
    assert status == 200 and _texts(alpha) == ["alpha owner"], "no binding places the room's own work"
    for payload in (main, beta, alpha):
        assert not set(_texts(payload)) & NEVER
    assert "owner steering a task" not in _texts(main), "an owner row of moved work follows that work"


def test_only_mains_refusal_offers_the_archive_and_another_rooms_exact_read_keeps_its_ordinary_one(root):
    rooms = _world(root)
    status, room = _main_history(root, chat_id=rooms["beta"])
    assert status == 200, "a Project room still reads with the rooms the readable rows name"
    address = next(message["row_address"] for message in room["messages"] if message.get("row_address"))
    status, exact = _main_history(root, chat_id=rooms["beta"], around=address)
    assert status == 503 and exact["reason_code"] == "history_source_unavailable"
    assert exact["detail"] == "Project registry unavailable" and "project_registry" not in exact
    status, main = _main_history(root, around=address)
    assert status == 503 and main["project_registry"] == "unavailable", "Main's exact read is Main's refusal"


def test_a_room_no_record_names_cannot_be_read_by_its_chat_id(root):
    rooms = _world(root)
    for room in (str(rooms["gamma"]), "0", "-1001", "1", "beta", "main ", str(rooms["beta"]) + " "):
        status, body = _archive(root, room=room)
        assert status == 404 and body == {"error": "archive_room_unknown", "reason_code": "archive_room_unknown"}, room
    assert _archive(root, room="")[1]["room"] == "main", "no room named reads Main"


def test_without_a_readable_bindings_file_no_work_is_placed_but_the_owners_words_stay(root):
    rooms = _world(root)
    (root / "state" / "project_task_bindings.json").write_text("{broken", encoding="utf-8")
    status, main = _archive(root)
    assert status == 200 and _texts(main) == MAIN_ROWS and main["limits"] == ["placement_unproven", "bindings_unavailable"]
    assert str(rooms["beta"]) in {room["room"] for room in main["rooms"]}, "Main's handoff row still names it"
    own = ["beta notice: steer not delivered"]  # a fixed notice written in the room needs no binding
    status, beta = _archive(root, room=str(rooms["beta"]))
    assert status == 200 and _texts(beta) == own, "a binding that cannot be read places no work"
    (root / "state" / "project_task_bindings.json").unlink()
    status, beta = _archive(root, room=str(rooms["beta"]))
    assert status == 200 and _texts(beta) == own and "bindings_unavailable" in beta["limits"]


def test_pages_continue_honestly_across_empty_stretches_and_reach_every_shown_row(root, monkeypatch):
    rooms = _world(root)
    monkeypatch.setattr(room_archive, "PAGE_ROWS", 2)
    monkeypatch.setattr(room_archive, "_PROBE_ROWS", 2)
    monkeypatch.setattr(room_archive, "_CANDIDATE_CEILING", 2)
    texts, _last = _all_pages(root, "main")
    assert texts == MAIN_ROWS
    empty, cursor = 0, None
    while True:
        status, page = _archive(root, **({"before": cursor} if cursor else {}))
        assert status == 200 and len(page["messages"]) <= 2
        empty += not page["messages"] and page["has_more"]
        if not page["has_more"]:
            break
        cursor = page["next_cursor"]
    assert empty, "a stretch of only unplaced rows answers its continuation, with nothing invented"
    texts, _last = _all_pages(root, str(rooms["beta"]))
    assert texts == BETA_ROWS


@pytest.mark.parametrize("portable_size", [False, True])
def test_saved_bytes_and_discovery_share_the_page_without_losing_the_next_words(root, monkeypatch, portable_size):
    lost_registry(root)
    write_bindings(root, [])
    room = project_chat_id("kiln")
    ds.append(root, "chat", {"chat_id": room, "ts": TS, "direction": "in", "text": "The first clay tile"})
    metadata = {"chat_id": 1, "role": "system", "type": "project_started", "project_id": "kiln",
                "project_name": "The kiln room"}
    notice = ds.append(root, "chat", {**metadata, "ts": TS, "direction": "out", "text": "The kiln room",
                                    "completion_answer": "Blue squares on paper. " * 300})
    owner = {"chat_id": 1, "ts": TS, "direction": "in", "text": "first draft"}
    older, newest = [ds.append(root, "chat", owner) for _ in range(2)]
    older_words, newest_words = "черепица " * 54, "a complete large message about birds " * 70
    revised = ds.revise_text(root, "chat", older.seq, older_words)
    ds.revise_text(root, "chat", newest.seq, newest_words)
    body_bytes = len(ds.encode(revised).encode("utf-8"))
    discovery_bytes = len(json.dumps(metadata, separators=(",", ":")).encode("utf-8"))
    budget = body_bytes + discovery_bytes // 2
    assert older.length < body_bytes < budget < body_bytes + discovery_bytes
    assert len(newest_words) > budget and notice.length > budget
    monkeypatch.setattr(room_archive, "_READ_BYTES", budget)
    if portable_size:
        monkeypatch.setattr(room_archive, "_BODY_SIZE", "length(CAST(body AS BLOB))")

    taken, decoded = [], []
    capture, decode = room_archive.capture, ds.decode

    def observe_capture(conn, sql, params, columns):
        result = capture(conn, sql, params, columns)
        if columns == ("seq", "body"):
            taken.extend(seq for seq, _ in result.tuples())
        return result

    def observe_decode(body):
        row = decode(body)
        if row.get("type") == "project_started":
            decoded.append(row)
        return row

    monkeypatch.setattr(room_archive, "capture", observe_capture)
    monkeypatch.setattr(ds, "decode", observe_decode)
    cursor = None
    for expected, seq in ((newest_words, newest.seq), (older_words, older.seq), ("The kiln room", notice.seq)):
        taken.clear()
        decoded.clear()
        status, page = _archive(root, **({"before": cursor} if cursor else {}))
        assert status == 200 and _texts(page) == [expected]
        assert taken == [seq], "size-only lookahead must not fetch any deferred row body"
        assert decoded[0] == metadata, "room discovery decodes only placement and audience metadata"
        assert page["rooms"][1]["name"] == "The kiln room"
        cursor = page["next_cursor"]
        assert page["has_more"] == (seq != notice.seq) == bool(cursor)


def test_a_cursor_is_this_rooms_own_and_never_reaches_below_the_import_frontier(root, monkeypatch):
    rooms = _world(root)
    monkeypatch.setattr(room_archive, "PAGE_ROWS", 1)
    status, newest = _archive(root)
    assert status == 200 and newest["has_more"] and newest["next_cursor"]
    _, main_view = room_archive._cursor_token(newest["next_cursor"], "main")
    for bad in ("x" * 300, "not-base64!", room_archive._cursor("main", 10**12, main_view),
                room_archive._cursor(str(rooms["beta"]), 5, main_view), "eyJ2IjoxfQ",
                _encoded({"v": 1, "room": "main", "before": 5})):
        status, body = _archive(root, before=bad)
        assert status == 400 and body["reason_code"] == "archive_cursor_invalid", bad
    floor = ds.stream_end(root, "chat") - 1
    with ds.transaction(root, write=True) as conn:
        conn.execute("INSERT INTO import_ledger (stream, name, base, size, state) "
                     "VALUES ('chat', 'fixture-segment', 0, ?, 'pending')", (floor,))
    status, page = _archive(root)
    assert status == 200 and page["has_more"] and "importing" in page["limits"]
    status, refused = _archive(root, before=page["next_cursor"])
    assert status == 503 and refused["reason_code"] == "migrating", "older rows are not readable yet, never absent"


def test_an_archive_read_writes_and_acknowledges_nothing(root):
    rooms = _world(root)
    state = root / "state"
    before = {path.name: path.read_bytes() for path in state.iterdir() if path.is_file() and path.suffix == ".json"}
    rooms_before, counts_before = ds.rooms(root), ds.counts(root)
    meta_before = ds.meta(root, "room_messages_bindings")
    for room in ("main", str(rooms["beta"]), str(rooms["alpha"])):
        assert _archive(root, room=room)[0] == 200
    after = {path.name: path.read_bytes() for path in state.iterdir() if path.is_file() and path.suffix == ".json"}
    assert after == before, "no registry, binding or preference write"
    assert not (state / "projects.json").exists() and not (state / "projects.json.lock").exists()
    assert (state / "projects.json.committed").read_bytes() == b"", "the old witness stays as it was"
    assert ds.rooms(root) == rooms_before and ds.counts(root) == counts_before, "no read credit, no row"
    assert ds.meta(root, "room_messages_bindings") == meta_before


def test_one_registry_row_whose_identity_cannot_be_read_refuses_main_on_every_page(root):
    """A readable room beside a row with no readable identity: Main is every chat no Project reserves, so the
    room that row reserved would reach Main's window. Main refuses on its newest page and on an older cursor,
    and the archive recognizes the readable rows by the registry itself. A malformed routing field is no
    identity gap: Main reads on."""
    rooms = _world(root, registry=True)
    status, healthy = _main_history(root, n_human=1)
    assert status == 200 and healthy["has_more"] and healthy["next_cursor"]
    gamma_row, readable = REGISTRY[2], REGISTRY[:2]
    for gap in ({"name": "Gamma room", "chat_id": gamma_row["chat_id"]}, {"name": "Gamma room"},
                {**gamma_row, "id": "Gamma"}, {**gamma_row, "chat_id": "not a chat"}, "not a row"):
        write_registry(root, [*readable, gap])
        for params in ({}, {"cursor": healthy["next_cursor"]}):
            status, refused = _main_history(root, **params)
            assert status == 503 and refused["project_registry"] == "unavailable", (gap, params)
            assert "gamma owner" not in _texts(refused)
        status, main = _archive(root)
        assert status == 200 and _texts(main) == MAIN_ROWS
        listed = {room["room"]: room for room in main["rooms"]}
        assert listed[str(rooms["beta"])]["evidence"] == ["binding", "main_record", "registry"]
        assert listed[str(rooms["alpha"])]["evidence"] == ["main_record", "registry"]
        assert str(rooms["gamma"]) not in listed, (gap, "the unreadable row names no room")

    write_registry(root, [*readable, {**gamma_row, "lifecycle": "bogus", "routing_generation": "x"}])
    status, main = _main_history(root)
    assert status == 200 and "gamma owner" not in _texts(main), "routing corruption keeps the identity"
    assert _archive(root) == (200, {"state": "not_needed"})


def test_a_room_the_readable_registry_marks_deleted_is_never_recognized(root):
    rooms = _world(root)
    write_registry(root, [REGISTRY[0], {**REGISTRY[1], "lifecycle": "tombstoned"}, REGISTRY[2], {"name": "no identity"}])
    status, main = _archive(root)
    assert status == 200 and str(rooms["beta"]) not in {room["room"] for room in main["rooms"]}
    assert _archive(root, room=str(rooms["beta"]))[0] == 404, "its binding and Main's row do not resurrect it"


def test_a_registry_authored_store_marker_names_a_room_and_a_bare_folder_does_not(root, monkeypatch):
    from ouroboros import utils

    _world(root)
    marked, bare = root / "projects" / "delta", root / "projects" / "epsilon"
    marked.mkdir(parents=True)
    bare.mkdir(parents=True)
    (marked / ".project.json").write_text(json.dumps({"id": "delta", "name": "Delta room"}), encoding="utf-8")
    idle = root / "projects" / "idle"
    idle.mkdir()
    (idle / ".project.json").write_text(json.dumps({"id": "idle", "name": "Idle room"}), encoding="utf-8")
    reads, staged = [], []
    snapshot = utils.json_snapshot
    history_snapshot = dialogue_history_evidence.saved_history_snapshot
    holding = False

    @contextmanager
    def observe_snapshot(path):
        nonlocal holding
        with history_snapshot(path) as conn:
            holding = True
            try:
                yield conn
            finally:
                holding = False

    def observe_staging(reader):
        def observed(*args, **kwargs):
            staged.append((reader.__name__, holding))
            return reader(*args, **kwargs)
        return observed

    def observe(path, **kwargs):
        if path.name == ".project.json":
            reads.append((path.parent.name, holding))
        return snapshot(path, **kwargs)

    monkeypatch.setattr(utils, "json_snapshot", observe)
    monkeypatch.setattr(dialogue_history_evidence, "saved_history_snapshot", observe_snapshot)
    files = ("_bindings", "_registry_evidence", "_store_candidates")
    for name in files:
        monkeypatch.setattr(room_archive, name, observe_staging(getattr(room_archive, name)))
    status, main = _archive(root)
    listed = {room["project_id"]: room for room in main["rooms"] if room["room"] != "main"}
    assert status == 200 and listed["delta"]["evidence"] == ["marker"] and listed["delta"]["name"] == "Delta room"
    assert "epsilon" not in listed, "a folder alone names no room"
    assert sorted(staged) == sorted((name, False) for name in files), "bulk filesystem reads precede the snapshot"
    assert reads == [("delta", True)], "only the marker selected by the snapshot's activity needs its contents"
    status, delta = _archive(root, room=str(project_chat_id("delta")))
    assert status == 200 and _texts(delta) == ["delta owner"]


def test_shelves_an_owner_reset_set_aside_name_no_room_for_any_reader(root):
    from ouroboros import project_registry_recovery as recovery, projects_registry

    _world(root)
    kept = root / "projects" / ".before-reset-20260101-0123abcd"
    for pid in ("delta", "proj_0123456789ab"):  # a named room and a bare workspace shelf, both with a marker
        (kept / pid / "knowledge").mkdir(parents=True)
        (kept / pid / ".project.json").write_text(json.dumps({"id": pid, "name": f"{pid} room"}), encoding="utf-8")
    assert not {"delta", "proj_0123456789ab", kept.name} & _offered(root)
    assert room_archive._store_candidates(root) == ({}, False)
    assert recovery._only_workspace_shelves(root), "the folder is no evidence of an earlier registry either"
    # Beside a readable registry the importer of earlier stores is at work, and still finds nothing there.
    write_registry(root, [])
    assert list(projects_registry.unregistered_project_stores(root, [])) == []
    assert projects_registry.reconcile_projects(root) == 0 and projects_registry.list_projects(root) == []
    assert sorted(entry.name for entry in kept.iterdir()) == ["delta", "proj_0123456789ab"]


def test_a_store_that_cannot_answer_is_a_typed_unavailable_answer(root, monkeypatch):
    from ouroboros.store import StoreError

    _world(root)

    def unavailable(*_args, **_kwargs):
        raise StoreError("store busy")

    monkeypatch.setattr(room_archive, "_probe", unavailable)
    assert _archive(root) == (503, {"error": "archive_source_unavailable", "reason_code": "archive_source_unavailable"})


def test_an_archive_read_never_creates_or_cuts_over_the_dialogue_store(root):
    """The boot and the import owner prepare the dialogue store; a read takes it as it stands: chain files still
    waiting answer "being prepared" and nothing is created, and a healthy registry reads nothing at all."""
    lost_registry(root)
    write_registry(root, REGISTRY[:1])
    dialogue_test.seed(root, live={"chat": [{"direction": "in", "task_id": "", "ts": TS, "chat_id": 1, "text": "waiting"}]})
    assert not ds.present(root) and ds.holds_dialogue_files(root)
    assert _archive(root) == (200, {"state": "not_needed"})
    (root / "state" / "projects.json").unlink()
    for room in ("main", "12345"):
        assert _archive(root, room=room) == (503, {"error": "migrating", "reason_code": "migrating"}), room
    assert not ds.present(root), "a read created or cut over no store"
    (root / "logs" / "chat.jsonl").unlink()
    status, empty = _archive(root)
    assert status == 200 and empty["messages"] == [] and empty["has_more"] is False
    assert not ds.present(root), "a root holding no dialogue reads the empty schema, writing nothing"


def test_unreadable_legacy_source_is_unavailable_and_no_database_is_created(root, monkeypatch):
    lost_registry(root)
    target = root / ds.LIVE_REL["chat"]
    stat = type(target).stat

    def read_stat(path, *args, **kwargs):
        if path == target:
            raise PermissionError("fixture source unavailable")
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(type(target), "stat", read_stat)
    assert _archive(root) == (503, {"error": "archive_source_unavailable", "reason_code": "archive_source_unavailable"})
    assert not ds.present(root)


def test_confirmed_empty_page_keeps_its_snapshot_when_a_legacy_source_arrives(root, monkeypatch):
    lost_registry(root)
    read_page = room_archive._page

    def read_after_arrival(*args):
        dialogue_test.seed(root, live={"chat": [{"chat_id": 1, "ts": TS, "direction": "in", "text": "later"}]})
        return read_page(*args)

    with monkeypatch.context() as change:
        change.setattr(room_archive, "_page", read_after_arrival)
        status, page = _archive(root)
    assert status == 200 and page["messages"] == [] and page["has_more"] is False
    assert _archive(root) == (503, {"error": "migrating", "reason_code": "migrating"})
    assert not ds.present(root)


def test_store_removed_between_observation_and_open_stays_unavailable(root, monkeypatch):
    lost_registry(root)
    dialogue_test.seed(root, live={"chat": [{"chat_id": 1, "ts": TS, "direction": "in", "text": "waiting"}]})
    # The first observation is from before Reset moved the file. The actual open sees the current disk.
    identity = ds.store.file_identity
    observed = [(123, 456)]

    def identity_after_reset(path):
        if path == ds.path(root) and observed:
            return observed.pop()
        return identity(path)

    with monkeypatch.context() as change:
        change.setattr(ds.store, "file_identity", identity_after_reset)
        assert _archive(root) == (503, {"error": "archive_source_unavailable", "reason_code": "archive_source_unavailable"})
    assert not ds.present(root)


def test_a_contradicted_identity_is_never_offered_and_its_neighbours_stay(root):
    """One id in two registry rows, a deleted project reserving another's chat and a binding naming another chat
    are contradictions: no binding, marker or Main row outweighs one, and an uncontradicted room is still offered."""
    rooms = _world(root, registry=True)
    write_registry(root, [*REGISTRY, {**REGISTRY[1], "name": "Beta twin"}])
    assert _offered(root) == {"alpha", "gamma"}, "the same id twice proves neither row, whatever its binding says"
    assert _archive(root, room=str(rooms["beta"]))[0] == 404
    assert _texts(_archive(root, room=str(rooms["alpha"]))[1]) == ["alpha owner"], "a proven neighbour reads"

    write_registry(root, [*REGISTRY, project_row("ghost", "Ghost", chat_id=rooms["alpha"], lifecycle="tombstoned")])
    assert _offered(root) == {"beta", "gamma"}, "a deleted reservation of alpha's chat competes for it"

    (root / "state" / "projects.json").unlink()
    write_bindings(root, [binding("beta-task", "beta", origin=ORIGIN, origin_id="cm-beta"),
                          binding("other-task", "beta", chat_id=rooms["gamma"])])
    assert _offered(root) == {"alpha"}, "a binding naming another chat contradicts beta's own"


def test_a_non_derived_room_a_truncated_window_cannot_verify_stays_withheld(root, monkeypatch):
    """Discovery of notices and shelves is bounded, and a bound is not a proof: a proposed room standing in
    another chat than its project's derived address could be contradicted by a notice or marker of that id
    outside the window (each claims the derived address), so a truncated window withholds it and says so;
    its derived-address neighbours stand. A complete window sees the contradiction itself."""
    lost_registry(root)
    pid = "legacy"
    other_chat = project_chat_id(pid) + 1
    host = {"direction": "out", "role": "system", "ts": TS, "chat_id": 1}
    ds.append(root, "chat", {**host, "type": "project_started", "project_id": pid,
                             "project_name": "Legacy room", "text": "Legacy room"})
    for name in ("uno", "dos", "tres"):
        ds.append(root, "chat", {**host, "type": "project_started", "project_id": name,
                                 "project_name": name, "text": name})
    write_bindings(root, [binding("legacy-task", pid, chat_id=other_chat), binding("uno-task", "uno")])

    monkeypatch.setattr(room_archive, "_EVIDENCE_ROWS", 2)  # the pid's own notice falls outside the window
    status, main = _archive(root)
    assert status == 200 and main["state"] == "available"
    listed = {room["project_id"] for room in main["rooms"] if room["room"] != "main"}
    assert pid not in listed and "uno" in listed
    assert "rooms_withheld_unverified" in main["limits"]
    assert _archive(root, room=str(other_chat))[0] == 404

    monkeypatch.setattr(room_archive, "_EVIDENCE_ROWS", 200)  # a complete window sees the contradiction
    status, main = _archive(root)
    assert pid not in {room["project_id"] for room in main["rooms"] if room["room"] != "main"}
    assert "rooms_withheld_unverified" not in main["limits"]


CHANGED = {"error": "archive_cursor_changed", "reason_code": "archive_cursor_changed"}


def test_a_tombstone_survives_a_malformed_field_of_its_own_row_and_ambiguity_withholds_only_its_identities(root):
    """Each registry row's facts are read one by one: a deleted mark outlives an unreadable address in the same row;
    an identity whose own address or lifecycle cannot be read is withheld; a reservation whose identity cannot be
    read contests its chat. Each withholds only the identities it touches."""
    rooms = _world(root)
    for row in ({**REGISTRY[1], "lifecycle": "tombstoned", "chat_id": "not a chat"},
                {**REGISTRY[1], "lifecycle": "tombstoned", "chat_id": None, "routing_generation": "x"},
                {**REGISTRY[1], "chat_id": [rooms["beta"]]},
                {**REGISTRY[1], "lifecycle": 7},
                {**REGISTRY[1], "lifecycle": ["tombstoned"]},
                {**REGISTRY[1], "lifecycle": {"state": "active"}}):
        write_registry(root, [REGISTRY[0], row, {"name": "no identity"}])
        assert room_archive.archive_needed(root) is True
        assert _offered(root) == {"alpha"}, (row, "its binding and Main's row do not outweigh the row's own facts")
        assert _archive(root, room=str(rooms["beta"]))[0] == 404, row
        assert _texts(_archive(root, room=str(rooms["alpha"]))[1]) == ["alpha owner"], "the neighbour still reads"

    write_registry(root, [{"id": "Alpha", "name": "Unreadable", "chat_id": rooms["alpha"]}])
    assert _offered(root) == {"beta"}, "an unreadable identity's reservation contests that chat alone"
    (root / "state" / "projects.json").unlink()
    write_bindings(root, [binding("beta-task", "beta", origin=ORIGIN, origin_id="cm-beta"),
                          {**binding("odd-task", "beta"), "project_chat_id": "not a chat"}])
    assert _offered(root) == {"alpha"}, "a binding naming beta at an unreadable address withholds beta, not alpha"


def test_a_store_published_before_its_import_plan_answers_migrating_and_stays_unplanned(root, monkeypatch):
    """A cutover that stopped after ``create`` leaves a store whose coordinates the import has not planned: the
    archive answers the retryable 503 ``migrating`` and plans, imports and writes nothing."""
    lost_registry(root)
    ds.create(root)
    assert ds.present(root) and ds.meta(root, "cutover") is None
    monkeypatch.setattr(ds, "_PLANNED", {str(ds.path(root))})  # the process saw the previous file here before Reset
    for room in ("main", "12345"):
        assert _archive(root, room=room) == (503, {"error": "migrating", "reason_code": "migrating"}), room
    assert ds.meta(root, "cutover") is None and ds.counts(root) == {}, "the read planned and stored nothing"


def test_an_older_binding_names_its_origin_by_chat_time_and_digest_never_by_coincident_words(root, monkeypatch):
    """A binding older than client ids names the owner's line by its chat, time and words' digest: one indexed
    chat-and-time lookup a place finds the one physical row they identify. The same words at another time are not
    it; two rows that identity fits are no origin; a binding naming only a digest, or an identity field of another
    type than the host stores, names nothing."""
    rooms = _world(root)
    earlier, lantern_ts = "2026-11-17T09:00:00Z", "2026-11-17T08:00:00Z"
    for row in ({"direction": "in", "chat_id": 1, "ts": earlier, "text": "Start the fictional kiln log"},
                {"direction": "in", "chat_id": 1, "ts": TS, "text": "Start the fictional kiln log"},
                {"direction": "in", "chat_id": 1, "ts": lantern_ts, "text": "Light the fictional lantern"},
                {"direction": "in", "chat_id": 1, "ts": lantern_ts, "text": "Light the fictional lantern"},
                {"direction": "in", "chat_id": 1, "ts": "2026-11-17T09:30:00Z", "text": "Only a digest names me"},
                {"direction": "in", "chat_id": 1, "ts": TS, "client_message_id": "7", "text": "A mistyped origin"}):
        ds.append(root, "chat", row)

    def ref_row(task, text, **ref):
        return {"task_id": task, "project_id": "beta", "project_chat_id": rooms["beta"], "bound_at": TS,
                "source_ref": {"chat_id": 1, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), **ref}}

    write_bindings(root, [binding("beta-task", "beta", origin=ORIGIN, origin_id="cm-beta"),
                          binding("kiln-task", "beta", origin="Start the fictional kiln log", origin_ts=earlier),
                          binding("kiln-twin", "beta", origin="Start the fictional kiln log", origin_ts=earlier),
                          binding("lantern-task", "beta", origin="Light the fictional lantern", origin_ts=lantern_ts),
                          ref_row("bare-task", "Only a digest names me"),
                          ref_row("odd-task", "A mistyped origin", client_message_id=7, ts=TS)])
    opened, statements = [], []
    original = ds.transaction

    @contextmanager
    def observe(path, **kwargs):
        opened.append(kwargs)
        with original(path, **kwargs) as conn:
            conn.set_trace_callback(statements.append)
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)

    with monkeypatch.context() as change:
        change.setattr(ds, "transaction", observe)
        status, beta = _archive(root, room=str(rooms["beta"]))
    assert status == 200 and _texts(beta) == [*BETA_ROWS, "Start the fictional kiln log"], "the one line, shown once"
    assert opened == [{"write": False, "existing_only": True}], "one existing source serves the whole page"
    lookups = [s for s in statements if "INDEXED BY dialogue_room_ts" in s]
    assert len(lookups) == 2 and all("LIMIT " in s for s in lookups), "one bounded lookup per chat and time"
    assert all(s.count("INDEXED BY dialogue_room_ts") == 2 and "chat_id IS NULL" in s for s in lookups), (
        "Main's old source references use both of the ordinary identity reader's indexed chat keys")
    texts, _last = _all_pages(root, "main")
    assert texts.count("Start the fictional kiln log") == 2 and texts.count("Light the fictional lantern") == 2
    assert "Only a digest names me" in texts and "A mistyped origin" in texts, "Main keeps the owner's words"


def test_bound_plain_work_follows_its_project_from_the_owner_transport(root):
    rooms = _world(root)
    transport = {"chat_id": 47, "task_id": "beta-task", "ts": TS, "source": "telegram"}
    for row in ({"direction": "in", "text": "owner adjustment"},
                {"direction": "out", "text": "work response"},
                {"direction": "out", "text": "unbound response", "task_id": "other-task"},
                {"direction": "out", "text": "private note", "presence_contact": {"place": "fixture"}},
                {"direction": "out", "text": "place note", "presence_provenance": {"place_id": "fixture"}},
                {"direction": "out", "text": "place summary", "place_origin": {"place_id": "fixture"}},
                {"direction": "out", "text": "internal note", "chat_id": -47},
                {"direction": "out", "text": "hidden note", "chat_id": 0},
                {"direction": "out", "text": "fixed receipt", "type": "task_not_started"}):
        ds.append(root, "chat", {**transport, **row})
    assert _all_pages(root, str(rooms["beta"]))[0] == [*BETA_ROWS, "owner adjustment", "work response"]
    assert _all_pages(root, "main")[0] == MAIN_ROWS
    assert _all_pages(root, str(rooms["alpha"]))[0] == ["alpha owner"]


@pytest.mark.parametrize("include_legacy_bucket", [False, True], ids=["written-chat", "whole-main-origin-scope"])
def test_partially_observed_origin_bucket_never_proves_a_unique_message(root, monkeypatch, include_legacy_bucket):
    rooms = _world(root)
    ts = "2026-11-17T07:00:00Z"
    origin = "Open the fixture notebook"
    row = {"direction": "in", "chat_id": 1, "ts": ts}
    ds.append(root, "chat", {**row, "text": origin})
    note = {"direction": "in", "ts": ts} if include_legacy_bucket else row
    for text in ("another note", "third note"):
        ds.append(root, "chat", {**note, "text": text})
    write_bindings(root, [binding("beta-task", "beta", origin=origin, origin_ts=ts)])
    room = str(rooms["beta"])
    monkeypatch.setattr(room_archive, "PAGE_ROWS", 1)
    status, page = _archive(root, room=room)
    assert status == 200 and page["next_cursor"]
    assert _all_pages(root, room)[0] == [*BETA_ROWS[1:], origin]
    # The whole indexed scope must fit, even though the other saved notes do not have these words.
    monkeypatch.setattr(room_archive, "_ORIGIN_BUDGET", 5)
    assert _archive(root, room=room, before=page["next_cursor"]) == (409, CHANGED)
    limited_texts, limited = _all_pages(root, room)
    assert limited_texts == BETA_ROWS[1:]
    assert "legacy_origins_limited" in limited["limits"]
    assert origin in _all_pages(root, "main")[0]


def test_legacy_origins_skip_only_unaffordable_buckets_and_do_not_consume_the_page(root, monkeypatch):
    rooms = _world(root)
    small, large = "Start the small desk notebook", "A long description of a paper house. " * 80
    a, b = "2026-11-17T05:00:00Z", "2026-11-17T06:00:00Z"
    for ts, text in ((a, "Start the paper crafts notebook"), (a, large), (b, small)):
        ds.append(root, "chat", {"chat_id": 1, "ts": ts, "direction": "in", "text": text})
    write_bindings(root, [binding("beta-task", "beta", origin=ORIGIN, origin_id="cm-beta"),
                          binding("paper-task", "beta", origin="Start the paper crafts notebook", origin_ts=a),
                          binding("desk-task", "beta", origin=small, origin_ts=b)])
    monkeypatch.setattr(room_archive, "_READ_BYTES", 2048)
    beta, page = _all_pages(root, str(rooms["beta"]))
    assert beta == [*BETA_ROWS, small]
    assert "legacy_origins_limited" in page["limits"]
    assert _all_pages(root, "main")[0] == [*MAIN_ROWS, "Start the paper crafts notebook", large, small]


def test_a_relay_answer_shows_in_its_source_room_and_its_bound_work_room_only(root):
    """A relay's answer is written to its source room and is its work's speech: the ordinary membership rule shows
    it there and in the Project its own binding names, never in a room it merely mentions, and never another
    audience's or hidden one."""
    rooms = _world(root)
    relay = {"direction": "out", "ts": TS, "type": "relay_reply", "system_type": "relay_reply", "task_id": "beta-task"}
    for row in ({**relay, "chat_id": 1, "text": "relay answer in main"},
                {**relay, "chat_id": rooms["alpha"], "text": "relay answer in alpha"},
                {**relay, "chat_id": rooms["gamma"], "text": "relay answer in gamma"},
                {**relay, "chat_id": 1, "task_id": "unbound-task", "text": "relay answer of unbound work"},
                {**relay, "chat_id": 0, "text": "relay answer hidden"},
                {**relay, "chat_id": 1, "presence_contact": {"place": "x"}, "text": "relay answer private"},
                *({**relay, "chat_id": 1, "type": kind, "text": "ill-typed kind"}
                  for kind in (["relay_reply"], {"kind": "relay_reply"}, 0, False))):
        ds.append(root, "chat", row)
    main = _all_pages(root, "main")[0]
    beta = _all_pages(root, str(rooms["beta"]))[0]
    alpha = _all_pages(root, str(rooms["alpha"]))[0]
    assert main[len(MAIN_ROWS):] == ["relay answer in main", "relay answer of unbound work"]
    assert beta[len(BETA_ROWS):] == ["relay answer in main", "relay answer in alpha", "relay answer in gamma"]
    assert alpha == ["alpha owner", "relay answer in alpha"]
    assert _archive(root, room=str(rooms["gamma"]))[0] == 404, "its source room is still no recognized room"


def test_a_continuation_whose_rooms_selecting_evidence_changed_restarts_with_409(root, monkeypatch):
    """A cursor holds its room's view: the room's project, the tasks bound to it and their origins. A change of
    those answers 409 ``archive_cursor_changed`` (so is a room no longer recognized), and the newest page reads
    again; a change elsewhere, a new row or another room's binding never stops a continuation."""
    rooms = _world(root)
    monkeypatch.setattr(room_archive, "PAGE_ROWS", 1)
    beta, alpha = str(rooms["beta"]), str(rooms["alpha"])
    status, first = _archive(root, room=beta)
    status_main, main = _archive(root)
    assert status == status_main == 200 and first["next_cursor"] and main["next_cursor"]

    ds.append(root, "chat", {"direction": "in", "chat_id": rooms["alpha"], "ts": TS, "text": "alpha later owner"})
    ds.append(root, "chat", {"direction": "in", "chat_id": rooms["beta"], "ts": TS, "text": "beta later owner"})
    write_bindings(root, [binding("beta-task", "beta", origin=ORIGIN, origin_id="cm-beta"), binding("alpha-task", "alpha")])
    assert _archive(root, room=beta, before=first["next_cursor"])[0] == 200, "nothing beta's rows depend on changed"
    assert _archive(root, before=main["next_cursor"])[0] == 200

    write_bindings(root, [binding("beta-task", "beta", origin=ORIGIN, origin_id="cm-beta"), binding("alpha-task", "alpha")])
    assert _archive(root, room=beta, before=first["next_cursor"])[0] == 200, "the same tasks and origins, rewritten"
    write_bindings(root, [binding("beta-task", "beta", origin="main owner", origin_id="cm-main"),
                          binding("alpha-task", "alpha")])
    assert _archive(root, room=beta, before=first["next_cursor"]) == (409, CHANGED), "beta's origin changed"
    write_bindings(root, [binding("beta-task", "beta", origin=ORIGIN, origin_id="cm-beta"), binding("alpha-task", "alpha"),
                          binding("late-task", "beta")])
    assert _archive(root, room=beta, before=first["next_cursor"]) == (409, CHANGED), "beta's bound work changed"
    assert _archive(root, before=main["next_cursor"])[0] == 200, "Main's rows never depend on a binding"
    status, again = _archive(root, room=beta)
    assert status == 200 and _archive(root, room=beta, before=again["next_cursor"])[0] == 200, "a restart continues"

    write_registry(root, [{**REGISTRY[1], "lifecycle": "tombstoned"}, {"name": "no identity"}])
    assert _archive(root, room=beta, before=again["next_cursor"]) == (409, CHANGED), "its room is no longer recognized"
    assert _archive(root, room=beta)[0] == 404, "a first read of that room is the ordinary unknown room"
    assert _archive(root, room=alpha)[0] == 200


def test_archive_cursor_belongs_to_one_dialogue_store(root, monkeypatch):
    _world(root)
    monkeypatch.setattr(room_archive, "PAGE_ROWS", 1)
    status, first = _archive(root)
    assert status == 200 and first["next_cursor"]
    replacement = root / "replacement"
    _world(replacement)
    with ds.transaction(replacement, write=True) as conn:
        ds._set_meta(conn, "created_at", "2026-11-18T10:00:00Z")
    assert _archive(replacement, before=first["next_cursor"]) == (409, CHANGED)
    assert _all_pages(replacement, "main")[0] == MAIN_ROWS


def test_a_pinned_main_row_proves_its_exact_address_and_audience_before_naming_a_room(root, monkeypatch):
    """Main's pinned host rows recognize a room only as rows the archive could itself read: one carrying private
    contact contributes neither an address nor a label, and an ordinary one beside it still does."""
    _world(root)
    host = {"direction": "out", "role": "system", "ts": TS, "chat_id": 1}
    for pid in ("zeta", "eta", "theta", "iota"):
        ds.append(root, "chat", {"direction": "in", "ts": TS, "chat_id": project_chat_id(pid), "text": f"{pid} owner"})
    ds.append(root, "chat", {**host, "type": "project_started", "system_type": "project_started", "project_id": "zeta",
                             "project_name": "Zeta room", "text": "Zeta room", "presence_contact": {"place": "x"}})
    ds.append(root, "chat", {**host, "type": "project_started", "system_type": "project_started", "project_id": "eta",
                             "project_name": "Eta room", "text": "Eta room"})
    ds.append(root, "chat", {**host, "type": "project_started", "project_id": "theta", "project_name": "Theta room",
                             "text": "Theta room", "task_id": "fixture", "quiz": {
                                 "question": "Another set of fictional choices. " * 150,
                                 "presence_contact": {
                                     "task_id": "fixture", "binding_id": "fictional", "source_event_id": "notice"}}})
    ds.append(root, "chat", {**host, "type": "project_started", "project_id": "iota", "project_name": "Iota room",
                             "text": "Iota room", "chat_id": True})
    monkeypatch.setattr(room_archive, "_READ_BYTES", 4096)  # the quiz's words alone could not fit discovery's quarter
    status, main = _archive(root)
    listed = {room["project_id"]: room for room in main["rooms"] if room["room"] != "main"}
    assert status == 200 and not {"zeta", "theta", "iota"} & listed.keys()
    assert not {"Zeta room", "Theta room", "Iota room"} & set(_all_pages(root, "main")[0])
    assert listed["eta"]["name"] == "Eta room" and listed["eta"]["evidence"] == ["main_record"]
