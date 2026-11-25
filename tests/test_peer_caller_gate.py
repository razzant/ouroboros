"""``NetworkAuthGate.classify_caller``: a linked Ouroboros is told apart by its link key, reaches its letters and is
refused everywhere else; a wrong, revoked or relayed key is refused where it is presented; its place's two facts
change only at my human's door (a domain projection: no request). The roles are projected on fictional routes and
read as who is refused: these tests say nothing of what a caller with no link header may reach."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.responses import JSONResponse
from starlette.routing import Route

from tests._peer_letters_shared import events, introduce, receiver_client

NEUTRAL = ("/neutral/one", "/neutral/two")  # fictional routes: the classifier reads the header and the path alone


def _spy_routes(seen):
    async def spy(request):
        seen.append((request.url.path, request.scope.get("ouroboros_caller")))
        return JSONResponse({"ok": True})

    return [Route("/api/letters", spy, methods=["POST"]), *(Route(path, spy, methods=["GET", "POST"]) for path in NEUTRAL)]


def test_a_link_key_from_this_computer_is_the_peer_on_its_letters_only(tmp_path, monkeypatch):
    pair = introduce(tmp_path, monkeypatch)
    seen = []
    with receiver_client(pair, monkeypatch, routes=_spy_routes(seen)) as client:
        assert client.post("/api/letters", json={}, headers=pair.headers()).status_code == 200
        assert seen == [("/api/letters", {"kind": "peer", "link_id": pair.link_id, "place_id": pair.place_b,
                                          "peer_label": "Laptop", "peer_install_id": seen[0][1]["peer_install_id"],
                                          "own_install_id": seen[0][1]["own_install_id"], "code_confirmed": True})]
        for method, path in (("GET", NEUTRAL[0]), ("POST", NEUTRAL[0]), ("POST", NEUTRAL[1])):
            refused = client.request(method, path, headers=pair.headers())
            assert refused.status_code == 403 and refused.json()["refusal_code"] == "peer_route_forbidden", path
        assert len(seen) == 1, "no other route ran for the peer"
        assert [(row["refusal_code"], row["path"]) for row in events(pair.root_b, "peer_letter_refused")] == [
            ("peer_route_forbidden", NEUTRAL[0]), ("peer_route_forbidden", NEUTRAL[1])], (
            "each refused write is audited; the refused read adds no row")


def test_a_wrong_or_revoked_key_is_refused_where_it_is_presented(tmp_path, monkeypatch):
    from ouroboros.install_identity import acting_as
    from ouroboros.peer_links import revoke

    pair = introduce(tmp_path, monkeypatch)
    seen = []
    good = pair.headers()
    wrong = {**good, "X-Ouroboros-Link": f"{pair.link_id}.not-it"}
    with receiver_client(pair, monkeypatch, routes=_spy_routes(seen)) as client:
        for path in (*NEUTRAL, "/api/letters"):
            refused = client.post(path, json={}, headers=wrong)
            assert refused.status_code == 401 and refused.json()["refusal_code"] == "link_unknown", path
        with acting_as(pair.root_b):
            revoke(pair.root_b, pair.link_id)
        refused = client.post("/api/letters", json={}, headers=good)
        assert refused.status_code == 401 and refused.json()["refusal_code"] == "link_unknown"
    assert seen == [], "no handler saw a refused key"


def test_a_peer_socket_is_closed(tmp_path, monkeypatch):
    from starlette.routing import WebSocketRoute
    from starlette.websockets import WebSocketDisconnect

    async def socket(websocket):
        await websocket.accept()
        await websocket.close()

    pair = introduce(tmp_path, monkeypatch)
    with receiver_client(pair, monkeypatch, routes=[WebSocketRoute("/ws", socket)]) as client:
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect("/ws", headers=pair.headers()):
                pass
    assert closed.value.code == 4403


def test_an_invitation_header_opens_only_the_introduction(tmp_path, monkeypatch):
    pair = introduce(tmp_path, monkeypatch)
    seen = []
    with receiver_client(pair, monkeypatch, routes=_spy_routes(seen)) as client:
        refused = client.get(NEUTRAL[0], headers={"X-Ouroboros-Invite": "anything"})
        assert refused.status_code == 403 and refused.json()["refusal_code"] == "peer_route_forbidden"
    assert seen == []


def test_the_peer_facts_change_only_at_my_humans_door(tmp_path, monkeypatch):
    from ouroboros import places
    from ouroboros.install_identity import acting_as
    from ouroboros.tools.presence import _configure_presence

    pair = introduce(tmp_path, monkeypatch)
    with acting_as(pair.root_b):
        with pytest.raises(places.PlaceError) as refused:
            places.edit_place(pair.root_b, pair.place_b, "set_peer", author=places.AUTHOR_OUROBOROS,
                              mine=True, may_start=False)
        assert refused.value.code == "place_peer_owner_only"
        ctx = SimpleNamespace(task_id="focus", task_metadata={"work_kind": "owner"}, is_direct_chat=False,
                              drive_root=pair.root_b, budget_drive_root=pair.root_b)
        tool = _configure_presence(ctx, "set_peer", place_id=pair.place_b, mine=True, may_start=False)
        assert "place_peer_owner_only" in tool
        assert places.place_records(pair.root_b).peer(pair.place_b) == {"mine": True, "may_start": True}
    from ouroboros.gateway.places_api import update_place
    from ouroboros.place_projection import places_view

    with acting_as(pair.root_b):  # my human's door, as its handler writes and lists it; no request is made
        saved = update_place(pair.root_b, pair.place_b, {"peer": {"mine": True, "may_start": False}})
        assert saved["place"]["peer"] == {"mine": True, "may_start": False}
        listed = next(row for row in places_view(pair.root_b)["places"] if row["place_id"] == pair.place_b)
        assert listed["peer"] == {"mine": True, "may_start": False} and listed["link"]["link_id"] == pair.link_id
        assert places.place_records(pair.root_b).choices[pair.place_b]["peer_by"] == places.AUTHOR_OWNER


def test_a_request_carrying_neither_header_is_not_the_classifiers(tmp_path, monkeypatch):
    """The classifier is one branch of the gate. Projected on it alone, with a fictional route and nothing behind
    it: a request with no link or invitation header is left unanswered and unstamped and reads no link registry,
    while a link key on the same route is answered (refused) from that registry."""
    import asyncio

    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as
    from ouroboros.server_auth import NetworkAuthGate

    pair = introduce(tmp_path, monkeypatch)
    reads, sent, reached = [], [], []
    real = peer_links.read_registry
    monkeypatch.setattr(peer_links, "read_registry", lambda root: reads.append(root) or real(root))

    async def behind(scope, receive, send):
        reached.append(scope.get("ouroboros_caller"))

    behind.state = SimpleNamespace(drive_root=pair.root_b)
    gate = NetworkAuthGate(behind)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    def classified(headers):
        scope = {"type": "http", "method": "GET", "path": NEUTRAL[0], "client": ("127.0.0.1", 50000),
                 "headers": [(name.lower().encode(), value.encode()) for name, value in headers.items()]}
        with acting_as(pair.root_b):
            return asyncio.run(gate.classify_caller(scope, receive, send)), scope.get("ouroboros_caller")

    assert classified({}) == (False, None)
    assert (reads, sent, reached) == ([], [], []), "no header: no answer, no caller, no registry read"
    assert classified(pair.headers()) == (True, None)
    assert reads and sent[0]["status"] == 403 and reached == [], "a link key is judged from the registry"


def test_without_any_link_the_letter_door_is_closed_and_no_link_state_appears(tmp_path, monkeypatch):
    from ouroboros.install_identity import acting_as
    from ouroboros.place_projection import places_view
    from tests._peer_letters_shared import Pair, make_roots

    root_a, root_b, repo = make_roots(tmp_path)
    pair = Pair(root_a, root_b, repo, "", "", "", "")
    with receiver_client(pair, monkeypatch) as client:
        closed = client.post("/api/letters", json={"letter_id": "x", "text": "y", "protocol": 1})
        assert closed.status_code == 401 and closed.json()["refusal_code"] == "link_unknown"
    with acting_as(root_b):
        assert places_view(root_b)["places"] == [], "the places listing of an install with no link"
    assert not (root_b / "state" / "peer_links.json").exists() and not (root_b / "state" / "peer_links").exists()


def test_an_unreadable_registry_is_not_a_revoked_link(tmp_path, monkeypatch):
    """A registry that cannot be read now answers 503 ``peer_links_unreadable``: the sender asks again instead of
    taking it for a revocation; once it reads again, the same key is the peer again."""
    from tests._peer_letters_shared import events, letter

    pair = introduce(tmp_path, monkeypatch)
    registry = pair.root_b / "state" / "peer_links.json"
    kept = registry.read_bytes()
    headers = pair.headers()
    registry.write_text("{not json", encoding="utf-8")
    with receiver_client(pair, monkeypatch) as client:
        posted = client.post("/api/letters", json=letter(), headers=headers)
        read = client.get("/api/letters/x", headers=headers)
        for answer in (posted, read):
            assert answer.status_code == 503 and answer.json()["refusal_code"] == "peer_links_unreadable", answer.text
        refused = [row["refusal_code"] for row in events(pair.root_b, "peer_letter_refused")]
        assert refused == ["peer_links_unreadable"], "the refused write is audited, the refused read is not"
        registry.write_bytes(kept)
        assert client.post("/api/letters", json=letter(), headers=headers).status_code == 200


def test_a_relayed_request_is_not_a_linked_ouroboros(tmp_path, monkeypatch):
    """This step's links join installations of one computer: a key or an invitation counts only on a direct
    connection from it (the neighbour door's raw-peer rule); a relayed one is refused where it is presented."""
    pair = introduce(tmp_path, monkeypatch)
    seen = []
    with receiver_client(pair, monkeypatch, routes=_spy_routes(seen)) as client:
        for extra in ({"X-Forwarded-For": "127.0.0.1"}, {"Forwarded": "for=127.0.0.1"}):
            for headers in ({**pair.headers(), **extra}, {"X-Ouroboros-Invite": "fictional", **extra}):
                path = "/api/letters" if "X-Ouroboros-Link" in headers else "/api/peer-links/introduce"
                refused = client.post(path, json={}, headers=headers)
                assert refused.status_code == 403 and refused.json()["refusal_code"] == "peer_route_forbidden"
    assert seen == []
