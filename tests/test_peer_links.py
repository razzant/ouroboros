"""The introduction of two Ouroboros installations and the keys it leaves: only hashes cross, both sides show one
check code, an invitation works once, revocation and pause stop letters, the secret stays owner-only and unseen,
and a moved data directory is refused."""
from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from tests._peer_letters_shared import events, introduce, letter, receiver_client


def test_only_hashes_cross_and_both_sides_show_one_code(tmp_path, monkeypatch):
    from ouroboros.install_identity import acting_as
    from ouroboros.peer_links import link_code, link_view, read_registry

    pair = introduce(tmp_path, monkeypatch)
    secrets_on_disk = (pair.secret_a(), pair.secret_b())
    crossing = json.dumps({"invite": pair.invite, "wire": pair.wire})
    assert all(secret not in crossing for secret in secrets_on_disk), "no raw secret leaves either side"
    import hashlib
    hashes = [hashlib.sha256(secret.encode()).hexdigest() for secret in secrets_on_disk]
    assert hashes[1] in crossing and all(len(value) == 64 for value in hashes)
    with acting_as(pair.root_a):
        code_a = link_view(pair.root_a, pair.link_id)["code"]
    with acting_as(pair.root_b):
        code_b = link_view(pair.root_b, pair.link_id)["code"]
        b_row = read_registry(pair.root_b)["links"][pair.link_id]
    with acting_as(pair.root_a):
        a_row = read_registry(pair.root_a)["links"][pair.link_id]
    assert code_a == code_b and len(code_a) == 6 and code_a.isdigit()
    # A (the inviter here is B) computes from the same four facts: link, both installs, both key hashes.
    assert code_b == link_code(pair.link_id, b_row["own_install_id"], a_row["own_install_id"],
                               a_row["inbound_key_sha256"], b_row["inbound_key_sha256"])
    assert "token" not in json.dumps(read_registry(pair.root_a)), "the registry holds no secret"


def test_a_stranger_with_a_valid_invitation_starts_no_work_until_my_human_confirms_the_code(tmp_path, monkeypatch):
    """An invitation seen in the channel that carried it: whoever introduces itself first gets a link that waits,
    Paused, until my human confirms the six digits on this side; refusing to confirm is an ordinary revoke."""
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as
    from ouroboros.places import place_records

    pair = introduce(tmp_path, monkeypatch, confirm=False)  # A here is the stranger who used B's invitation
    with acting_as(pair.root_b):
        assert place_records(pair.root_b).places[pair.place_b].state == "paused"
        assert peer_links.link_view(pair.root_b, pair.link_id)["code_check"] == "pending"
    assert peer_links.outgoing_link(pair.place_a, drive_root=pair.root_a) is None, "unconfirmed: nothing is sent"
    with acting_as(pair.root_a):
        peer_links.confirm(pair.root_a, pair.link_id)  # the stranger controls its own side, never B's
    body = letter()
    with receiver_client(pair, monkeypatch) as client:
        refused = client.post("/api/letters", json=body, headers=pair.headers())
        assert refused.status_code == 403, refused.text
        assert (refused.json()["refusal_code"], refused.json()["detail"]) == ("link_paused", "code_check_pending")
        assert not pair.row(body["letter_id"]), "no work was created"
        # The surviving path: my human compares the codes on B and confirms; the same letter is then accepted.
        assert client.post(f"/api/peer-links/{pair.link_id}/confirm", json={}).json()["code_check"] == "confirmed"
        accepted = client.post("/api/letters", json=body, headers=pair.headers())
        assert accepted.status_code == 200 and accepted.json()["state"] == "accepted_running", accepted.text
        assert client.post(f"/api/peer-links/{pair.link_id}/confirm", json={"code": "1"}).status_code == 400
    with acting_as(pair.root_b):
        assert any(row.get("refusal_code") == "link_paused" for row in events(pair.root_b, "peer_letter_refused"))
        assert place_records(pair.root_b).places[pair.place_b].state == "active"
        peer_links.revoke(pair.root_b, pair.link_id)
        with pytest.raises(peer_links.PeerLinkError) as gone:
            peer_links.confirm(pair.root_b, pair.link_id)
    assert gone.value.code == "link_unknown", "a revoked link is never confirmed back"


def test_configure_presence_lists_and_inspects_my_humans_facts_of_a_link(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ouroboros.install_identity import acting_as
    from ouroboros.tools.presence import _configure_presence

    pair = introduce(tmp_path, monkeypatch)  # B lets A start work; A's place for B only answers
    for root, place_id, start in ((pair.root_b, pair.place_b, "yes"), (pair.root_a, pair.place_a, "no")):
        ctx = SimpleNamespace(task_id="t", task_metadata={}, drive_root=root)
        with acting_as(root):
            listed = json.loads(_configure_presence(ctx, "list"))
            inspected = json.loads(_configure_presence(ctx, "inspect", place_id=place_id))
        row = next(view for view in listed["places"] if view["place_id"] == place_id)
        for view in (row, inspected["place"]):
            assert view["peer"]["mine"] is True and view["link"]["code_check"] == "confirmed"
            assert "mine: yes" in view["peer_facts"] and f"may_start: {start}" in view["peer_facts"], view["peer_facts"]


def test_an_invitation_works_once(tmp_path, monkeypatch):
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as

    pair = introduce(tmp_path, monkeypatch)
    used = next(row for row in pair.wire if "token" in row)
    with acting_as(pair.root_b), pytest.raises(peer_links.PeerLinkError) as again:
        peer_links.introduce(pair.root_b, used["token"], {**used["body"], "install_id": "someone-else"})
    assert again.value.code == "invite_unknown" and again.value.status == 401


def test_revocation_stops_letters_and_outgoing_and_pause_stops_reception(tmp_path, monkeypatch):
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, edit_place

    pair = introduce(tmp_path, monkeypatch)
    headers = pair.headers()
    with receiver_client(pair, monkeypatch) as client:
        with acting_as(pair.root_b):
            edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state="paused")
        paused = client.post("/api/letters", json=letter(), headers=headers)
        assert paused.json()["refusal_code"] == "link_paused"
        with acting_as(pair.root_b):
            edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state="active")
        assert client.post("/api/letters", json=letter(), headers=headers).status_code == 200
        with acting_as(pair.root_b):
            view = peer_links.revoke(pair.root_b, pair.link_id)
        assert view["state"] == "removed" and not peer_links.token_path(pair.root_b, pair.link_id).exists()
        refused = client.post("/api/letters", json=letter(), headers=headers)
        assert refused.status_code == 401 and refused.json()["refusal_code"] == "link_unknown"
    assert peer_links.outgoing_link(pair.place_b, drive_root=pair.root_b) is None, "B no longer writes to A"
    with acting_as(pair.root_a):
        edit_place(pair.root_a, pair.place_a, "set_state", author=AUTHOR_OWNER, state="paused")
    assert peer_links.outgoing_link(pair.place_a, drive_root=pair.root_a) is None, "a paused place sends nothing"


def test_outgoing_link_needs_my_mark(tmp_path, monkeypatch):
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, edit_place

    pair = introduce(tmp_path, monkeypatch)
    link = peer_links.outgoing_link(pair.place_a, drive_root=pair.root_a)
    assert (link.link_id, link.peer_label, link.peer_url) == (pair.link_id, "Velm", "http://127.0.0.1:9")
    assert pair.secret_a() not in repr(link)
    with acting_as(pair.root_a):
        edit_place(pair.root_a, pair.place_a, "set_peer", author=AUTHOR_OWNER, mine=False, may_start=False)
    assert peer_links.outgoing_link(pair.place_a, drive_root=pair.root_a) is None
    assert peer_links.outgoing_link("no-such-place", drive_root=pair.root_a) is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits; Windows protects the DACL instead")
def test_the_secret_file_is_owner_only(tmp_path, monkeypatch):
    from ouroboros.peer_links import token_path

    pair = introduce(tmp_path, monkeypatch)
    for root in (pair.root_a, pair.root_b):
        mode = stat.S_IMODE(os.stat(token_path(root, pair.link_id)).st_mode)
        assert mode == 0o600, oct(mode)


def test_the_key_is_compared_in_constant_time(tmp_path, monkeypatch):
    import hmac

    from ouroboros import peer_links

    pair = introduce(tmp_path, monkeypatch)
    calls = []
    real = hmac.compare_digest
    monkeypatch.setattr(peer_links.hmac, "compare_digest", lambda a, b: calls.append((a, b)) or real(a, b))
    caller = peer_links.authenticate(pair.root_b, pair.headers()["X-Ouroboros-Link"])
    assert caller["link_id"] == pair.link_id and len(calls) == 1
    with pytest.raises(peer_links.PeerLinkError):
        peer_links.authenticate(pair.root_b, f"{pair.link_id}.wrong")
    assert len(calls) == 2, "a wrong key is compared the same way, never short-circuited"


def test_the_secret_never_reaches_responses_places_tasks_or_logs(tmp_path, monkeypatch):
    from ouroboros.install_identity import acting_as
    from ouroboros.task_results import load_task_result

    pair = introduce(tmp_path, monkeypatch)
    secrets_on_disk = (pair.secret_a(), pair.secret_b())
    body = letter()
    with receiver_client(pair, monkeypatch) as client:
        seen = [client.post("/api/letters", json=body, headers=pair.headers()).text,
                client.get(f"/api/letters/{body['letter_id']}", headers=pair.headers()).text,
                client.get("/api/places").text,
                client.post("/api/peer-links/invite", json={"may_start": False}).text,
                client.post(f"/api/peer-links/{pair.link_id}/revoke").text]
        files = client.get("/api/files/read", params={"path": f"state/peer_links/{pair.link_id}/token.json"})
    with acting_as(pair.root_b):
        seen.append(json.dumps(load_task_result(pair.root_b, pair.task_id(body["letter_id"]), strict=True)))
    seen.append(json.dumps(events(pair.root_b)))
    seen.extend(path.read_text(encoding="utf-8", errors="replace") for path in (pair.root_b / "logs").glob("*"))
    seen.append((pair.root_b / "state" / "places.json").read_text(encoding="utf-8"))
    assert all(secret not in text for secret in secrets_on_disk for text in seen)
    assert files.status_code != 200 or secrets_on_disk[1] not in files.text


def test_the_secret_file_is_under_the_runtime_secret_policy(tmp_path):
    from ouroboros.peer_links import token_path
    from ouroboros.tools.core_secret_paths import _is_runtime_secret_data_path

    relative = token_path(tmp_path, "0123456789abcdef").relative_to(tmp_path).as_posix()
    assert relative == "state/peer_links/0123456789abcdef/token.json"
    assert _is_runtime_secret_data_path(relative)
    assert _is_runtime_secret_data_path("state/peer_links/0123456789abcdef/.token.json.tmp.1.2.abcd1234")


def test_a_moved_data_directory_is_refused_until_a_new_introduction(tmp_path, monkeypatch):
    from ouroboros.install_identity import IDENTITY_PATH, acting_as

    pair = introduce(tmp_path, monkeypatch)
    identity = pair.root_b / IDENTITY_PATH
    record = json.loads(identity.read_text(encoding="utf-8"))
    identity.write_text(json.dumps({**record, "id": "a-new-install-after-a-move"}), encoding="utf-8")
    with receiver_client(pair, monkeypatch) as client:
        refused = client.post("/api/letters", json=letter(), headers=pair.headers())
    assert refused.json()["refusal_code"] == "install_mismatch"
    with acting_as(pair.root_b):
        assert any(row["refusal_code"] == "install_mismatch" for row in events(pair.root_b, "peer_letter_refused"))


def test_an_unknown_or_revoked_link_at_my_humans_doors_is_link_unknown_401(tmp_path, monkeypatch):
    pair = introduce(tmp_path, monkeypatch)
    with receiver_client(pair, monkeypatch) as client:
        revoked = client.post(f"/api/peer-links/{pair.link_id}/revoke", json={})
        assert revoked.status_code == 200 and revoked.json()["state"] == "removed"
        again = client.post(f"/api/peer-links/{pair.link_id}/revoke", json={})
        assert again.status_code == 200 and again.json() == revoked.json(), "my own revoked link, unchanged"
        for path in (f"/api/peer-links/{pair.link_id}/confirm", "/api/peer-links/0123456789abcdef/revoke",
                     "/api/peer-links/0123456789abcdef/confirm"):
            answer = client.post(path, json={})
            assert answer.status_code == 401 and answer.json()["refusal_code"] == "link_unknown", (path, answer.text)
        for path in (f"/api/peer-links/{pair.link_id}/revoke", f"/api/peer-links/{pair.link_id}/confirm"):
            answer = client.post(path, json=[])
            assert answer.status_code == 400 and answer.json()["refusal_code"] == "invalid_request", (path, answer.text)


def test_an_address_beyond_this_computer_is_refused_before_anything_is_written_or_claimed(tmp_path, monkeypatch):
    """This step joins two installations of one computer: an invitation, an acceptance or an introduction naming
    another address is ``peer_url_refused``; nothing is posted, no secret or place is left, no token is claimed."""
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as
    from tests._peer_letters_shared import make_roots

    root_a, root_b, _repo = make_roots(tmp_path)
    monkeypatch.setattr(peer_links, "_post_introduce", lambda *a: pytest.fail("an introduction was sent"))
    with acting_as(root_b), pytest.raises(peer_links.PeerLinkError) as invited:
        peer_links.create_invite(root_b, own_url="http://192.168.1.20:8765", may_start=True)
    assert (invited.value.code, invited.value.status) == ("peer_url_refused", 400)
    with acting_as(root_b):
        invite = peer_links.create_invite(root_b, own_url="http://127.0.0.1:9", may_start=True)
    payload = peer_links.parse_invite(invite["invite"])
    remote = peer_links._encode_invite({**payload, "url": "https://ouroboros.example.org"})
    with acting_as(root_a), pytest.raises(peer_links.PeerLinkError) as accepted:
        peer_links.accept_invite(root_a, remote, own_url="http://127.0.0.1:8", may_start=False)
    assert accepted.value.code == "peer_url_refused"
    assert not (root_a / "state" / "peer_links").exists() and not peer_links.read_registry(root_a)["links"]
    body = {"link_id": payload["link_id"], "install_id": "install-a", "url": "http://10.0.0.5:8765",
            "key_sha256": "0" * 64, "may_start": False, "protocol": 1}
    with acting_as(root_b), pytest.raises(peer_links.PeerLinkError) as introduced:
        peer_links.introduce(root_b, payload["token"], body)
    assert introduced.value.code == "peer_url_refused"
    with acting_as(root_b):
        assert not any(row.get("used_at") for row in peer_links.read_registry(root_b)["invites"].values())
        assert peer_links.introduce(root_b, payload["token"], {**body, "url": "http://127.0.0.1:8"})["link_id"]


def test_an_expired_invitation_is_refused_and_leaves_the_registry(tmp_path, monkeypatch):
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as
    from tests._peer_letters_shared import make_roots

    root_a, root_b, _repo = make_roots(tmp_path)
    monkeypatch.setattr(peer_links, "INVITE_TTL_SEC", -1)
    with acting_as(root_b):
        invite = peer_links.create_invite(root_b, own_url="http://127.0.0.1:9", may_start=True)
    with acting_as(root_a), pytest.raises(peer_links.PeerLinkError) as late:
        peer_links.accept_invite(root_a, invite["invite"], own_url="http://127.0.0.1:8", may_start=False)
    assert (late.value.code, late.value.status) == ("invite_expired", 410)
    import base64

    raw = invite["invite"][len(peer_links.INVITE_PREFIX):]
    payload = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
    with acting_as(root_b), pytest.raises(peer_links.PeerLinkError) as gone:
        peer_links.introduce(root_b, payload["token"], {"link_id": payload["link_id"], "install_id": "install-a",
                                                        "url": "http://127.0.0.1:8", "key_sha256": "0" * 64})
    assert gone.value.code == "invite_unknown", "an expired token authorises nothing, even at the door"
    with acting_as(root_b):
        assert peer_links.read_registry(root_b)["invites"] == {}, "the write that checked it dropped it"


def test_one_acceptance_holds_an_invitation_and_a_failed_second_takes_nothing_of_the_first(tmp_path, monkeypatch):
    """Two pastes of one invitation on one side: the second is ``invite_used`` before it writes a key or a place, so
    the first one's key, place and link stay whole."""
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as
    from ouroboros.places import place_records
    from tests._peer_letters_shared import make_roots

    root_a, root_b, _repo = make_roots(tmp_path)
    with acting_as(root_b):
        invite = peer_links.create_invite(root_b, own_url="http://127.0.0.1:9", may_start=True)
    second = {}

    def post(url, token, body):  # the first acceptance is under way when the same invitation is pasted again
        with acting_as(root_a), pytest.raises(peer_links.PeerLinkError) as refused:
            peer_links.accept_invite(root_a, invite["invite"], own_url="http://127.0.0.1:8", may_start=False)
        second["code"], second["status"] = refused.value.code, refused.value.status
        with acting_as(root_b):
            return peer_links.introduce(root_b, token, body)

    monkeypatch.setattr(peer_links, "_post_introduce", post)
    with acting_as(root_a):
        view = peer_links.accept_invite(root_a, invite["invite"], own_url="http://127.0.0.1:8", may_start=False)
        registry = peer_links.read_registry(root_a)
        places = [place for place in place_records(root_a).places.values() if place.connector == "ouroboros"]
    assert second == {"code": "invite_used", "status": 409}
    assert [place.place_id for place in places] == [view["place_id"]] and registry["accepting"] == {}
    secret = json.loads(peer_links.token_path(root_a, view["link_id"]).read_text(encoding="utf-8"))["token"]
    with acting_as(root_b):
        assert peer_links.read_registry(root_b)["links"][view["link_id"]]["inbound_key_sha256"] == \
            peer_links._sha256(secret), "B knows the key A kept"


def test_an_introduction_naming_this_installation_leaves_the_invitation_for_the_right_one(tmp_path, monkeypatch):
    """A self-paste (the acceptor names the inviting installation) is refused before the invitation is touched: it
    stays unused, its key stays, and the right installation's introduction then succeeds."""
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as, recorded_id
    from tests._peer_letters_shared import make_roots

    _root_a, root_b, _repo = make_roots(tmp_path)
    with acting_as(root_b):
        invite = peer_links.create_invite(root_b, own_url="http://127.0.0.1:9", may_start=True)
        own = recorded_id(root_b)
    payload = peer_links.parse_invite(invite["invite"])
    body = {"link_id": payload["link_id"], "install_id": own, "url": "http://127.0.0.1:8", "key_sha256": "0" * 64}
    with acting_as(root_b):
        with pytest.raises(peer_links.PeerLinkError) as refused:
            peer_links.introduce(root_b, payload["token"], body)
        assert (refused.value.code, refused.value.status) == ("install_mismatch", 403)
        assert not any(row.get("used_at") or row.get("claimed_by")
                       for row in peer_links.read_registry(root_b)["invites"].values())
        assert peer_links.token_path(root_b, payload["link_id"]).exists()
        answer = peer_links.introduce(root_b, payload["token"], {**body, "install_id": "install-a"})
    assert answer["link_id"] == payload["link_id"]


def test_a_token_no_invitation_names_writes_nothing(tmp_path, monkeypatch):
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as

    pair = introduce(tmp_path, monkeypatch)
    with acting_as(pair.root_b):
        before = peer_links.registry_path(pair.root_b).read_bytes()
        with pytest.raises(peer_links.PeerLinkError) as unknown:
            peer_links.introduce(pair.root_b, "no-such-token", {"link_id": "0123456789abcdef", "install_id": "x",
                                                                "url": "http://127.0.0.1:8", "key_sha256": "0" * 64})
    assert unknown.value.code == "invite_unknown" and peer_links.registry_path(pair.root_b).read_bytes() == before


def test_an_invitations_key_leaves_with_it_unless_a_link_took_it(tmp_path, monkeypatch):
    """An unused invitation's key leaves at the first write after its deadline; a used one's key stays its link's."""
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as

    pair = introduce(tmp_path, monkeypatch)  # B's invitation was used: its key is the link's
    with acting_as(pair.root_b):
        unused = peer_links.create_invite(pair.root_b, own_url="http://127.0.0.1:9", may_start=False)
        assert peer_links.token_path(pair.root_b, unused["link_id"]).exists()
        monkeypatch.setattr(peer_links, "_now", lambda: "9999-01-01T00:00:00+00:00")
        peer_links.update_registry(pair.root_b, lambda document: None)
        assert peer_links.read_registry(pair.root_b)["invites"] == {}
        assert not peer_links.token_path(pair.root_b, unused["link_id"]).parent.exists(), "no key without its row"
        assert peer_links.token_path(pair.root_b, pair.link_id).exists(), "a link's key is never an invitation's"


def test_the_introduction_goes_to_the_invitations_address_and_nowhere_else(monkeypatch):
    """The one request carrying an invitation token uses a direct client: no environment proxy, no redirect
    followed; a redirect is the other side's answer, a refusal."""
    import httpx

    from ouroboros import peer_links

    real, made, seen = httpx.Client, [], []

    def handler(request):
        seen.append((str(request.url), request.headers.get(peer_links.INVITE_HEADER)))
        return httpx.Response(307, headers={"Location": "http://127.0.0.1:7/api/peer-links/introduce"})

    def client(**kwargs):
        made.append(kwargs)
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    with pytest.raises(peer_links.PeerLinkError) as refused:
        peer_links._post_introduce("http://127.0.0.1:9", "fictional-token", {"link_id": "0123456789abcdef"})
    assert refused.value.code == "introduce_refused"
    assert made[0]["trust_env"] is False and made[0]["follow_redirects"] is False
    assert seen == [("http://127.0.0.1:9/api/peer-links/introduce", "fictional-token")]


def test_files_keeps_a_link_key_and_its_landing_temporary_to_my_human_alone(tmp_path, monkeypatch):
    """Files never lists, reads or serves a link's key nor the temporary its owner-only write leaves while landing;
    the folder's other files stay as visible as ever."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    import ouroboros.gateway.files as files
    from ouroboros import config
    from ouroboros.peer_links import is_secret_file, token_path

    data = tmp_path / "data"
    key = token_path(data, "0123456789abcdef")
    key.parent.mkdir(parents=True)
    landing = key.with_name(".token.json.tmp.1.2.abcd1234")
    for path in (key, landing):
        path.write_text('{"token": "fictional-key"}', encoding="utf-8")
    (key.parent / "note.txt").write_text("visible", encoding="utf-8")
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(tmp_path))
    with TestClient(Starlette(routes=files.file_browser_routes())) as client:
        listed = client.get("/api/files/list", params={"path": "data/state/peer_links/0123456789abcdef"}).json()
        assert [entry["name"] for entry in listed["entries"]] == ["note.txt"] and listed["truncated"] is False
        for path in (key, landing):
            for route in ("/api/files/read", "/api/files/download", "/api/files/content"):
                answer = client.get(route, params={"path": path.relative_to(tmp_path).as_posix()})
                assert answer.status_code == 403 and "fictional-key" not in answer.text, (route, path.name)
    spelled = data / "STATE" / "PEER_LINKS" / "0123456789abcdef" / "TOKEN.JSON"  # one file on a case-blind disk
    assert is_secret_file(spelled, data) and is_secret_file(landing, data)
    assert not is_secret_file(key.parent / "replies" / ("0" * 64 + ".json"), data)


def test_a_listing_resolves_only_a_row_that_can_be_a_key_and_leaves_an_unclassified_one_out(tmp_path, monkeypatch):
    """A listing resolves only a row that can be a link's key: one named like the key or its landing temporary, or
    a link, which may point at one. Its ordinary rows are listed from their own names, unresolved, however many
    they are. The resolver fails on one candidate (a link that loops raises ``RuntimeError`` before Python 3.13;
    the error is mocked here on a key-named row every platform can create): Files still lists the folder, a link's
    key and a link to it unlisted, and the one row whose protective classification could not finish left out
    whole, with no name, type or size of its own. The predicate every read asks still resolves, and still raises
    for it: an unresolved target is never taken for an ordinary file."""
    import os
    import pathlib

    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    import ouroboros.gateway.files as files
    from ouroboros import config
    from ouroboros.peer_links import is_secret_file, may_be_secret_file, token_path

    data = tmp_path / "data"
    key = token_path(data, "0123456789abcdef")
    key.parent.mkdir(parents=True)
    key.write_text('{"token": "fictional-key"}', encoding="utf-8")
    (key.parent / "folder").mkdir()
    ordinary = ["a.txt", "z.txt", *(f"row-{index:03}.txt" for index in range(40))]
    odd = ".token.json.tmp.odd"
    for name in (*ordinary, odd):
        (key.parent / name).write_text("neutral", encoding="utf-8")
    try:  # a link to the key under an ordinary name, where this platform and account may create one
        os.symlink(key, key.parent / "alias.txt")
        linked = True
    except (OSError, NotImplementedError):
        linked = False
    real, asked = pathlib.Path.resolve, []

    def resolve(self, strict=False):
        asked.append(self.name)
        if self.name == odd:
            raise RuntimeError(f"Symlink loop from {str(self)!r}")
        return real(self, strict=strict)

    monkeypatch.setattr(pathlib.Path, "resolve", resolve)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(tmp_path))
    with TestClient(Starlette(routes=files.file_browser_routes())) as client:
        answer = client.get("/api/files/list", params={"path": "data/state/peer_links/0123456789abcdef"})
    assert answer.status_code == 200, answer.text
    listed = answer.json()
    assert [(row["name"], row["type"]) for row in listed["entries"]] == [
        ("folder", "dir"), *((name, "file") for name in sorted(ordinary))] and listed["truncated"] is False
    assert not {"folder", *ordinary} & set(asked), "an ordinary row never reaches the resolver"
    assert {"token.json", odd} <= set(asked) and (not linked or "alias.txt" in asked)
    assert "odd" not in answer.text and "token.json" not in answer.text and "alias.txt" not in answer.text
    assert may_be_secret_file(key) and may_be_secret_file(key.parent / odd)
    assert not may_be_secret_file(key.parent / "a.txt") and not may_be_secret_file(key.parent / "folder")
    assert may_be_secret_file(key.parent / "alias.txt") is linked
    with pytest.raises(RuntimeError):
        is_secret_file(key.parent / odd, data)
    assert not is_secret_file(key.parent / "a.txt", data) and "a.txt" in asked, "a read's own check still resolves"


def test_link_keys_and_invitations_are_masked_where_text_is_kept_and_whole_at_my_humans_door(tmp_path, monkeypatch):
    from ouroboros import peer_links
    from ouroboros.redaction import redact_projection
    from ouroboros.utils import is_credential_header_name, sanitize_tool_result_for_log

    pair = introduce(tmp_path, monkeypatch)
    with receiver_client(pair, monkeypatch) as client:
        invite = client.post("/api/peer-links/invite", json={"may_start": False}).json()["invite"]
    assert peer_links.preview_invite(invite)["link_id"], "my human's door gives the invitation whole"
    token = peer_links.parse_invite(invite)["token"]
    for kept in (sanitize_tool_result_for_log(f"paste this at the other one: {invite}"),
                 str(redact_projection({"note": f"paste this at the other one: {invite}"}).value)):
        assert invite not in kept and token not in kept, kept
    assert all(is_credential_header_name(name) for name in ("X-Ouroboros-Link", "x-ouroboros-invite"))


def test_an_expired_row_keeps_custody_until_its_attempts_exact_files_are_gone(tmp_path, monkeypatch):
    """An expired invitation whose key the storage would not remove stays, authorising nothing, and the next
    write that removes the key ends it; the landing temporary a stop left goes with it, another attempt's stays."""
    import pathlib

    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as
    from tests._peer_letters_shared import make_roots

    _root_a, root_b, _repo = make_roots(tmp_path)
    with acting_as(root_b):
        invite = peer_links.create_invite(root_b, own_url="http://127.0.0.1:9", may_start=True)
    payload = peer_links.parse_invite(invite["invite"])
    key = peer_links.token_path(root_b, payload["link_id"])
    attempt = next(iter(peer_links.read_registry(root_b)["invites"].values()))["attempt"]
    landing = key.with_name(f".token.json.tmp.{attempt}")  # what a stop between landing and replace leaves
    landing.write_bytes(b'{"token": "fictional')
    other = key.with_name(".token.json.tmp.another-attempt")
    other.write_bytes(b"{}")
    real_unlink = pathlib.Path.unlink

    def refusing(self, *args, **kwargs):  # the storage refuses to remove this key, for now
        if self == key:
            raise PermissionError("fictional refusal")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "unlink", refusing)
    monkeypatch.setattr(peer_links, "_now", lambda: "9999-01-01T00:00:00+00:00")
    body = {"link_id": payload["link_id"], "install_id": "install-a", "url": "http://127.0.0.1:8",
            "key_sha256": "0" * 64}
    with acting_as(root_b):
        peer_links.update_registry(root_b, lambda document: None)
        assert [row["attempt"] for row in peer_links.read_registry(root_b)["invites"].values()] == [attempt]
        with pytest.raises(peer_links.PeerLinkError) as refused:
            peer_links.introduce(root_b, payload["token"], body)
        assert refused.value.code == "invite_unknown", "an expired row authorises nothing"
        assert key.exists() and not landing.exists() and other.exists()
        monkeypatch.setattr(pathlib.Path, "unlink", real_unlink)
        peer_links.update_registry(root_b, lambda document: None)
        assert peer_links.read_registry(root_b)["invites"] == {}
    assert not key.exists() and other.exists(), "only this attempt's exact files left"


def test_a_late_rollback_takes_back_nothing_of_the_acceptance_that_took_over(tmp_path, monkeypatch):
    """An acceptance stalled past its deadline: its row and its exact files retire, a second acceptance of the same
    link takes over and succeeds; the stalled one's failure then leaves that link, its key and its place whole."""
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as
    from ouroboros.places import place_records
    from tests._peer_letters_shared import make_roots

    root_a, root_b, _repo = make_roots(tmp_path)
    with acting_as(root_b):
        invite = peer_links.create_invite(root_b, own_url="http://127.0.0.1:9", may_start=True)
    sent = []
    real_now = peer_links._now

    def post(url, token, body):
        sent.append(body["key_sha256"])
        if len(sent) == 1:  # the first acceptance stalls past its deadline while a second one runs
            monkeypatch.setattr(peer_links, "_now", lambda: "9999-01-01T00:00:00+00:00")
            peer_links.update_registry(root_a, lambda document: None)
            monkeypatch.setattr(peer_links, "_now", real_now)
            with acting_as(root_a):
                peer_links.accept_invite(root_a, invite["invite"], own_url="http://127.0.0.1:8", may_start=False)
            raise peer_links.PeerLinkError("introduce_unreachable", "fictional timeout", 502)
        with acting_as(root_b):
            return peer_links.introduce(root_b, token, body)

    monkeypatch.setattr(peer_links, "_post_introduce", post)
    with acting_as(root_a), pytest.raises(peer_links.PeerLinkError):
        peer_links.accept_invite(root_a, invite["invite"], own_url="http://127.0.0.1:8", may_start=False)
    link_id = peer_links.parse_invite(invite["invite"])["link_id"]
    with acting_as(root_a):
        registry = peer_links.read_registry(root_a)
        place = place_records(root_a).places.get(registry["links"][link_id]["place_id"])
    assert registry["accepting"] == {} and place is not None, "the second acceptance's link and place stay"
    secret = json.loads(peer_links.token_path(root_a, link_id).read_text(encoding="utf-8"))["token"]
    assert len(sent) == 2 and peer_links._sha256(secret) == sent[1], "its key stays, the one B recorded"


def test_provisional_introduction_never_overwrites_or_disconnects_places_and_cleans_only_attempt_files(
    tmp_path, monkeypatch,
):
    """Without an atomic cross-store creator proof in ``peer_links``, failed or expired introductions never disconnect
    or overwrite places:
    1. A failed/expired acceptance drops only this attempt's key/landing files and reservation, keeps room history, and
       leaves an unlinked ``Paused`` provisional place that authorizes nothing and is cleanly reused on a matching retry
       of the same invitation (including with ``note``).
    2. A pre-existing or owner-modified place is refused with ``place_exists`` (409) without overwriting ``peer``
       choices or notes and without disconnecting it."""
    from ouroboros import dialogue_store, peer_links
    from ouroboros.install_identity import acting_as
    from ouroboros.places import (
        AUTHOR_OWNER, STATE_PAUSED, connect_place, conversation_chat_id, edit_place, place_records,
    )
    from ouroboros.presence_bindings import conversation_key
    from tests._peer_letters_shared import make_roots

    root_a, root_b, _repo = make_roots(tmp_path)
    with acting_as(root_b):
        invite = peer_links.create_invite(root_b, own_url="http://127.0.0.1:9", may_start=True)
    payload = peer_links.parse_invite(invite["invite"])
    link_id = payload["link_id"]

    # Write a room history row first to prove provisional failure never deletes dialogue history.
    conv_key = conversation_key(peer_links.CONNECTOR, payload["install_id"], link_id, "")
    chat_id = conversation_chat_id(conv_key)
    with acting_as(root_a):
        dialogue_store.append(root_a, "chat", {
            "ts": "2026-11-25T07:00:00+00:00", "direction": "system", "chat_id": chat_id,
            "user_id": 0, "text": "Prior room note.", "client_message_id": "prior-room-1",
        })

    # 1. Transient unreachable introduction with a note: drops attempt key & reservation, leaves safe Paused/unlinked
    # residual, and a matching retry of the same invitation reuses it and completes cleanly.
    monkeypatch.setattr(peer_links, "_post_introduce",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            peer_links.PeerLinkError("introduce_unreachable", "down", 502)))
    with acting_as(root_a), pytest.raises(peer_links.PeerLinkError) as down:
        peer_links.accept_invite(root_a, invite["invite"], own_url="http://127.0.0.1:8",
                                 note="Pinned link note.", may_start=False)
    assert down.value.code == "introduce_unreachable"
    assert not peer_links.token_path(root_a, link_id).exists()
    with acting_as(root_a):
        assert peer_links.read_registry(root_a)["accepting"] == {}
        records = place_records(root_a)
        assert len(records.places) == 1
        residual = next(iter(records.places.values()))
        assert residual.state == STATE_PAUSED and peer_links.link_of_place(root_a, residual.place_id) is None
        perms = peer_links.place_permissions(root_a, residual, records.peer(residual.place_id))
        assert perms == {"messages": False, "assignments": False, "reason": "unlinked"}
        assert peer_links.outgoing_link(residual.place_id, drive_root=root_a) is None
        assert len(dialogue_store.rows_by_client_id(root_a, "prior-room-1", chat_id=chat_id)) == 1

    # Conflicting retry (different may_start or note) refuses with place_exists rather than overwriting residual.
    with acting_as(root_a), pytest.raises(peer_links.PeerLinkError) as mismatch:
        peer_links.accept_invite(root_a, invite["invite"], own_url="http://127.0.0.1:8",
                                 note="Different note.", may_start=False)
    assert mismatch.value.code == "place_exists"

    # Matching retry of the same invitation reuses the provisional place and settles the link.
    monkeypatch.setattr(peer_links, "_post_introduce",
                        lambda url, tok, body: peer_links.introduce(root_b, tok, body))
    with acting_as(root_b):
        pass
    def intro_on_b(url, tok, body):
        with acting_as(root_b):
            return peer_links.introduce(root_b, tok, body)

    monkeypatch.setattr(peer_links, "_post_introduce", intro_on_b)
    with acting_as(root_a):
        settled = peer_links.accept_invite(root_a, invite["invite"], own_url="http://127.0.0.1:8",
                                           note="Pinned link note.", may_start=False)
    assert settled["place_id"] == residual.place_id

    # 2. A pre-existing place is never overwritten (peer stays None, label stays "Pre") and never disconnected.
    with acting_as(root_b):
        invite2 = peer_links.create_invite(root_b, own_url="http://127.0.0.1:9", may_start=False)
    p2 = peer_links.parse_invite(invite2["invite"])
    with acting_as(root_a):
        pre = connect_place(root_a, connector=peer_links.CONNECTOR, provider=peer_links.CONNECTOR, label="Pre",
                            account_id=p2["install_id"], conversation_id=p2["link_id"], scope="conversation",
                            state=STATE_PAUSED, author=AUTHOR_OWNER)
    with acting_as(root_a), pytest.raises(peer_links.PeerLinkError) as exists_err:
        peer_links.accept_invite(root_a, invite2["invite"], own_url="http://127.0.0.1:8", may_start=False)
    assert exists_err.value.code == "place_exists" and exists_err.value.status == 409
    with acting_as(root_a):
        recs = place_records(root_a)
        assert recs.places[pre.place_id].label == "Pre"
        assert recs.peer(pre.place_id) is None, "pre-existing place choices must never be overwritten"

    # 3. An owner-edited provisional place keeps the owner's note when introduction fails.
    with acting_as(root_b):
        invite3 = peer_links.create_invite(root_b, own_url="http://127.0.0.1:9", may_start=False)

    def owner_edits_then_fails(url, token, body):
        with acting_as(root_a):
            only = next(p for p in place_records(root_a).places.values()
                        if p.place_id not in {residual.place_id, pre.place_id})
            edit_place(root_a, only.place_id, "add_note", author=AUTHOR_OWNER, text="Owner note while connecting.")
        raise peer_links.PeerLinkError("introduce_unreachable", "down", 502)

    monkeypatch.setattr(peer_links, "_post_introduce", owner_edits_then_fails)
    with acting_as(root_a), pytest.raises(peer_links.PeerLinkError):
        peer_links.accept_invite(root_a, invite3["invite"], own_url="http://127.0.0.1:8", may_start=False)
    with acting_as(root_a):
        kept = [p for p in place_records(root_a).places.values()
                if p.place_id not in {residual.place_id, pre.place_id}]
        assert len(kept) == 1 and kept[0].notes[0].text == "Owner note while connecting."


def test_an_owed_record_leaves_only_in_the_exact_state_its_remover_read(tmp_path):
    """``settle_owed(seen=..., unowed=...)``: under the registry's writer lock that ``record_owed`` writes under, the
    record must still be exactly as its remover read it (``owed_version``) and ``unowed()`` must still read true. A
    rewrite since (a newer attempt's, even an identical one that already ended) keeps it."""
    from ouroboros import peer_links
    from ouroboros.utils import json_update_lock

    link = "link-0001-test"
    peer_links.record_owed(tmp_path, link, "turn-1", place_id="place-1")
    seen = peer_links.owed_version(tmp_path, link, "turn-1")

    def held_while_read():
        with pytest.raises(TimeoutError), json_update_lock(peer_links.registry_path(tmp_path), timeout_sec=0.05):
            pass
        return False  # an attempt of it runs now

    assert peer_links.settle_owed(tmp_path, link, "turn-1", seen=seen, unowed=held_while_read) is False
    peer_links.record_owed(tmp_path, link, "turn-1", place_id="place-1")  # another attempt's write, same words
    assert peer_links.settle_owed(tmp_path, link, "turn-1", seen=seen, unowed=lambda: True) is False
    assert peer_links.owed_record(tmp_path, link, "turn-1") is not None
    now = peer_links.owed_version(tmp_path, link, "turn-1")
    assert now != seen and peer_links.settle_owed(tmp_path, link, "turn-1", seen=now, unowed=lambda: True) is True
    assert peer_links.owed_record(tmp_path, link, "turn-1") is None and peer_links.owed_version(tmp_path, link, "turn-1") is None


def test_a_named_links_owed_records_are_read_from_its_own_folder_alone(tmp_path, monkeypatch):
    """``owed_refs(root, link_id)`` opens only that link's folder; without a link it reads every link, as at a start."""
    from ouroboros import peer_links

    links = ("link-0001-test", "link-0002-test")
    peer_links.update_registry(tmp_path, lambda doc: doc["links"].update({link: {
        "place_id": f"place-{n}", "peer_label": "Velm", "peer_url": "http://127.0.0.1:9", "peer_install_id": "install-b",
        "own_install_id": "install-a", "code_confirmed_at": "2026-11-24T00:00:00+00:00"} for n, link in enumerate(links)}))
    for n, link in enumerate(links):
        peer_links.record_owed(tmp_path, link, f"turn-{n}", place_id=f"place-{n}")
    opened: list = []
    real = peer_links.token_path
    monkeypatch.setattr(peer_links, "token_path", lambda root, link_id: (opened.append(link_id), real(root, link_id))[1])
    assert peer_links.owed_refs(tmp_path, links[1]) == [(links[1], "turn-1")] and opened == [links[1]]
    opened.clear()
    assert sorted(peer_links.owed_refs(tmp_path)) == [(links[0], "turn-0"), (links[1], "turn-1")]
    assert sorted(opened) == sorted(links) and peer_links.owed_refs(tmp_path, "link-9999-test") == []


def test_my_humans_reactivation_of_a_links_place_returns_only_what_that_link_is_owed(tmp_path, monkeypatch):
    """Only my human's confirmed move of a link's place to Active (from Paused or Silent) resumes the replies that
    link is still owed (``peer_replies.resume_returned``); another edit, an already Active place or a Pause starts
    nothing. A return reads only that link's owed records, never a sending record, and nothing while Panic holds."""
    import threading

    from ouroboros import peer_links
    from ouroboros import peer_replies
    from ouroboros.places import AUTHOR_OWNER, STATE_ACTIVE, STATE_PAUSED, STATE_SILENT, connect_place, edit_place

    returned: list = []
    real_returned = peer_replies.resume_returned
    monkeypatch.setattr(peer_replies, "resume_returned", lambda root, **kw: returned.append(kw))
    place = connect_place(tmp_path, connector="ouroboros", provider="ouroboros", label="Velm", account_id="install-b",
                          conversation_id="link-0001-test", scope="conversation", state=STATE_PAUSED, author=AUTHOR_OWNER)
    edit_place(tmp_path, place.place_id, "add_note", author=AUTHOR_OWNER, text="Velm may count files.")
    assert returned == []
    for state in (STATE_ACTIVE, STATE_ACTIVE, STATE_SILENT, STATE_ACTIVE, STATE_PAUSED):
        edit_place(tmp_path, place.place_id, "set_state", author=AUTHOR_OWNER, state=state)
    assert returned == [{"link_id": "link-0001-test"}, {"link_id": "link-0001-test"}]

    started: list = []
    done = threading.Event()
    monkeypatch.setattr(peer_replies, "resume_returned", real_returned)
    asked: list = []
    every = [("link-0001-test", "turn-1"), ("link-0002-test", "turn-2")]
    monkeypatch.setattr(peer_links, "owed_refs",
                        lambda root, link_id="": asked.append(link_id) or [r for r in every if not link_id or r[0] == link_id])
    monkeypatch.setattr(peer_links, "sending_records", lambda root: pytest.fail("a return reads no sending record"))
    monkeypatch.setattr(peer_replies, "_resume_quietly",
                        lambda root, owed, sending: (started.append((owed, list(sending))), done.set()))
    switch = {"now": "off"}
    monkeypatch.setattr("ouroboros.place_projection.switch_state", lambda root: switch["now"])
    peer_replies.resume_returned(tmp_path, link_id="link-0001-test")
    assert done.wait(5) and started == [([("link-0001-test", "turn-1")], [])] and asked == ["link-0001-test"]
    done.clear()
    peer_replies.resume_returned(tmp_path)  # a Panic's end: every link
    assert done.wait(5) and started[-1] == ([("link-0001-test", "turn-1"), ("link-0002-test", "turn-2")], [])
    switch["now"] = "panic"
    peer_replies.resume_returned(tmp_path)
    assert len(started) == 2, "nothing goes while a Panic awaits my human's return"


def test_work_its_turn_did_not_name_as_its_exit_is_promised_no_reply(tmp_path, monkeypatch):
    """A turn's handed-off work answers the link only when the turn's own end names it (``presence_work_ref``): other
    work it started sends nothing automatically and leaves no record owed. Pure stand-ins: no network."""
    from types import SimpleNamespace

    from ouroboros import peer_links
    from ouroboros import peer_replies

    rows = {"child-1": {"task_id": "child-1", "status": "completed", "result": "Counted.",
                        "metadata": {"objective_author": {"kind": "task", "task_id": "turn-1"},
                                     "presence": {"place_id": "place-1"}}},
            "turn-1": {"task_id": "turn-1", "status": "completed",
                       "metadata": {"presence": {"place_id": "place-1"}, "presence_work_ref": "other-1"}}}
    place = SimpleNamespace(place_id="place-1", conversation_id="link-0001-test", provider="ouroboros", account_id="install-b",
                            connector="ouroboros")
    monkeypatch.setattr("ouroboros.task_results.load_task_result", lambda root, task_id, **kw: rows.get(task_id))
    monkeypatch.setattr(peer_replies, "_peer_speaker", lambda root, stored: place)
    monkeypatch.setattr(peer_replies, "_no_live_turn", lambda root, ref: True)
    monkeypatch.setattr(peer_replies, "_deliver_output", lambda *a, **k: pytest.fail("an unnamed handoff sends nothing"))
    peer_links.record_owed(tmp_path, "link-0001-test", "child-1", place_id="place-1")
    seen = peer_links.owed_version(tmp_path, "link-0001-test", "child-1")
    assert peer_replies._deliver_producer(tmp_path, "child-1", link_id="link-0001-test", seen=seen) == (True, None)
    assert peer_links.owed_record(tmp_path, "link-0001-test", "child-1") is None
    rows.pop("child-1")  # no row under a reference (none yet, or none readable): unknown, its record stays
    peer_links.record_owed(tmp_path, "link-0001-test", "child-1", place_id="place-1")
    seen = peer_links.owed_version(tmp_path, "link-0001-test", "child-1")
    assert peer_replies._deliver_producer(tmp_path, "child-1", link_id="link-0001-test", seen=seen) == (False, None)
    assert peer_links.owed_record(tmp_path, "link-0001-test", "child-1") is not None


def test_owed_record_distinguishes_absent_from_unreadable_and_sending_confirm_rollback(tmp_path):
    """``owed_record`` returns None only when absent and raises ``peer_links_unreadable`` on corrupt/unreadable
    files; ``confirm_sending`` preserves delivered over failed and ``rollback_sending`` undoes unstarted passes
    without touching confirmed records."""
    from ouroboros import peer_links

    link = "link-0001-test"
    assert peer_links.owed_record(tmp_path, link, "missing-1") is None
    corrupt = peer_links._owed_path(tmp_path, link, "corrupt-1")
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_text("{not-json", encoding="utf-8")
    with pytest.raises(peer_links.PeerLinkError) as exc_info:
        peer_links.owed_record(tmp_path, link, "corrupt-1")
    assert exc_info.value.code == "peer_links_unreadable" and exc_info.value.status == 503

    bad_shape = peer_links._owed_path(tmp_path, link, "wrong-ref")
    bad_shape.write_text('{"work_ref": "other"}', encoding="utf-8")
    with pytest.raises(peer_links.PeerLinkError) as exc_info2:
        peer_links.owed_record(tmp_path, link, "wrong-ref")
    assert exc_info2.value.code == "peer_links_unreadable"

    p1 = peer_links.record_sending(
        tmp_path, link, "msg-1", account_id="inst-a", place_id="place-1",
        text="Hello", reply_to="m-1", origin={"kind": "tool", "task_id": "t-1"},
    )
    assert p1 == 1
    assert peer_links.rollback_sending(tmp_path, link, "msg-1", expected_passes=2) is False
    assert peer_links.rollback_sending(tmp_path, link, "msg-1", expected_passes=1) is True
    assert peer_links.sending_record(tmp_path, link, "msg-1") is None

    assert peer_links.record_sending(
        tmp_path, link, "msg-2", account_id="inst-a", place_id="place-1",
        text="Two", reply_to="m-1", origin={"kind": "tool", "task_id": "t-1"},
    ) == 1
    assert peer_links.record_sending(
        tmp_path, link, "msg-2", account_id="inst-a",
        text="Two", reply_to="m-1", origin={"kind": "tool", "task_id": "t-1"},
    ) == 2
    assert peer_links.rollback_sending(tmp_path, link, "msg-2", expected_passes=2) is True
    rec = peer_links.sending_record(tmp_path, link, "msg-2")
    assert rec["passes"] == 1 and rec["place_id"] == "place-1"

    confirmed = peer_links.confirm_sending(
        tmp_path, link, "msg-2", state="delivered", account_id="inst-a", place_id="place-1",
        text="Two", reply_to="m-1", origin={"kind": "tool", "task_id": "t-1"},
    )
    assert confirmed["confirmed_state"] == "delivered" and confirmed["place_id"] == "place-1"
    never_downgrade = peer_links.confirm_sending(
        tmp_path, link, "msg-2", state="failed", code="link_paused", message="refused later",
    )
    assert never_downgrade["confirmed_state"] == "delivered"
    assert peer_links.rollback_sending(tmp_path, link, "msg-2", expected_passes=1) is False
    assert peer_links.sending_record(tmp_path, link, "msg-2")["confirmed_state"] == "delivered"

    # record_sending preserves original empty reply_to="" and rejects mismatched text/recipient.
    assert peer_links.record_sending(
        tmp_path, link, "msg-empty-addr", account_id="inst-a", place_id="place-1",
        text="Same", reply_to="", origin={"kind": "automatic", "task_id": "t-1"},
    ) == 1
    assert peer_links.record_sending(
        tmp_path, link, "msg-empty-addr", account_id="inst-a", place_id="place-1",
        text="Same", reply_to="m-newer", origin={"kind": "automatic", "task_id": "t-1"},
    ) == 2
    assert peer_links.sending_record(tmp_path, link, "msg-empty-addr")["reply_to"] == ""
    with pytest.raises(peer_links.PeerLinkError) as mismatch_txt:
        peer_links.record_sending(
            tmp_path, link, "msg-empty-addr", account_id="inst-a", place_id="place-1",
            text="Other", reply_to="", origin={"kind": "automatic", "task_id": "t-1"},
        )
    assert mismatch_txt.value.code == "sending_identity_mismatch"
    with pytest.raises(peer_links.PeerLinkError) as mismatch_acc:
        peer_links.record_sending(
            tmp_path, link, "msg-empty-addr", account_id="inst-other", place_id="place-1",
            text="Same", reply_to="", origin={"kind": "automatic", "task_id": "t-1"},
        )
    assert mismatch_acc.value.code == "sending_identity_mismatch"


def test_p5_sanitize_tool_result_for_log_masks_v2_invite_when_projection_fails(monkeypatch):
    """P5: first pass of sanitize_tool_result_for_log masks both v1 and v2 invitations even if
    observability.redact_projection raises."""
    from ouroboros import observability, utils

    monkeypatch.setattr(observability, "redact_projection", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    v2 = "ouroboros-invite:2:ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    masked = utils.sanitize_tool_result_for_log(f"created invitation {v2} for peer")
    assert v2 not in masked and "ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in masked, masked
    assert "***REDACTED***" in masked, masked
