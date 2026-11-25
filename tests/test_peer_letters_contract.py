"""Acceptance of the letter wire contract between two linked Ouroboros installations, over loopback HTTP.

In-process API, gate and edge coverage live in ``test_peer_letters_api.py``, ``test_peer_letters_edges.py``,
``test_peer_caller_gate.py`` and ``test_peer_links.py``. This file keeps the single real-socket loopback witness:
two live Uvicorn servers complete introduction over HTTP, enforce socket-bound route/header/body gates, run a
work letter through question, answer and unmocked ``/result``, and honour sender ``/cancel``.
"""
from __future__ import annotations

import socket
import threading
import time
import uuid

import httpx
import pytest

from tests._peer_letters_shared import init_control_state, published

# Generous: a host's first file rename can be slow (an endpoint scanner), and these are real loopback requests.
TIMEOUT = 120


class _Installation:
    def __init__(self, root, repo):
        import uvicorn
        from starlette.applications import Starlette

        from ouroboros.gateway.router import collect_routes
        from ouroboros.install_identity import acting_as
        from ouroboros.server_auth import NetworkAuthGate

        app = Starlette(routes=collect_routes(data_dir=root))
        app.state.drive_root, app.state.repo_dir = root, repo
        gate = NetworkAuthGate(app)

        async def as_this_install(scope, receive, send):  # each installation is its own data root and identity
            with acting_as(root):
                await gate(scope, receive, send)

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.root, self.url = root, f"http://127.0.0.1:{port}"
        self.server = uvicorn.Server(uvicorn.Config(as_this_install, host="127.0.0.1", port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, name=f"peer-{port}", daemon=True)
        self.thread.start()
        deadline = time.monotonic() + TIMEOUT
        while not self.server.started:
            assert self.thread.is_alive() and time.monotonic() < deadline, "installation did not start"
            time.sleep(0.02)

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=15)


class _Pair:
    """The receiver B, the sender A's headers for B, and B's receiver-side acts."""

    def __init__(self, sender, receiver, link_id, place_id):
        self.sender, self.receiver = sender, receiver
        self.url, self.link_id, self.place_id = receiver.url, link_id, place_id

    def headers(self, **override):
        from ouroboros.peer_links import outgoing_link

        headers = outgoing_link(self.place_id_at_sender, drive_root=self.sender.root).auth_headers()
        headers.update(override)
        return {key: value for key, value in headers.items() if value is not None}

    def _task_id(self, letter_id):
        from ouroboros.peer_letters import letter_task_id

        return letter_task_id(self.place_id, letter_id)

    def ask(self, letter_id, text):
        from ouroboros.install_identity import acting_as
        from ouroboros.peer_letters import ask

        with acting_as(self.receiver.root):
            return ask(self.receiver.root, self._task_id(letter_id), text,
                       published=published(self.place_id, text, root=self.receiver.root,
                                           link_id=self.link_id))["question_id"]

    def finish(self, letter_id, result):
        import supervisor.queue as queue
        from ouroboros.install_identity import acting_as
        from ouroboros.task_results import write_task_result

        task_id = self._task_id(letter_id)
        queue.PENDING[:] = [row for row in queue.PENDING if row.get("id") != task_id]
        with acting_as(self.receiver.root):
            write_task_result(self.receiver.root, task_id, "completed", result=result,  # admitted for A, as stamped
                              peer_reply=published(self.place_id, result, root=self.receiver.root,
                                                   link_id=self.link_id),
                              finished_at="2026-11-23T01:00:00+00:00")


@pytest.fixture
def peer(tmp_path, monkeypatch):
    """Two installations on one computer, introduced: A may write to B, B only answers A."""
    import supervisor.queue as queue
    import supervisor.workers as workers
    from types import SimpleNamespace

    roots = {}
    for name in ("a", "b"):
        roots[name] = (tmp_path / name / "data", tmp_path / name / "repo")
        for path in roots[name]:
            (path / "memory" if path.name == "data" else path).mkdir(parents=True)
        init_control_state(roots[name][0], enabled=False)
    monkeypatch.setattr(workers, "WORKERS", {0: SimpleNamespace(busy_task_id=None, reaping=False)})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(queue, "DRIVE_ROOT", roots["b"][0])
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda *args, **kwargs: True)
    sender = _Installation(*roots["a"])
    receiver = _Installation(*roots["b"])
    try:
        invite = httpx.post(receiver.url + "/api/peer-links/invite",
                            json={"may_start": True, "label": "Laptop", "own_label": "Velm"}, timeout=TIMEOUT)
        assert invite.status_code == 200, invite.text
        accepted = httpx.post(sender.url + "/api/peer-links/accept",
                              json={"invite": invite.json()["invite"], "may_start": False, "label": "Velm",
                                    "own_label": "Laptop"}, timeout=TIMEOUT)
        assert accepted.status_code == 200, accepted.text
        link_id = accepted.json()["link_id"]
        places = httpx.get(receiver.url + "/api/places", timeout=TIMEOUT).json()["places"]
        place_id = next(row["place_id"] for row in places if row["connector"] == "ouroboros")
        pair = _Pair(sender, receiver, link_id, place_id)
        pair.place_id_at_sender = accepted.json()["place_id"]
        pair.sender_code, pair.receiver_code = accepted.json()["code"], next(
            row["link"]["code"] for row in places if row["connector"] == "ouroboros")
        for side in (receiver, sender):  # my human compared the six digits on both screens
            confirmed = httpx.post(side.url + f"/api/peer-links/{link_id}/confirm", json={}, timeout=TIMEOUT)
            assert confirmed.status_code == 200 and confirmed.json()["code_check"] == "confirmed", confirmed.text
        yield pair
    finally:
        receiver.stop()
        sender.stop()


def _letter(**fields):
    return {"letter_id": fields.pop("letter_id", uuid.uuid4().hex[:12]), "text": "Count the files in /tmp.",
            "protocol": 1, **fields}


def _post(peer, body, **headers):
    return httpx.post(peer.url + "/api/letters", json=body, headers=peer.headers(**headers), timeout=TIMEOUT)


def _code(response):
    body = response.json()
    return body.get("refusal_code") or body.get("error")


def test_letter_wire_contract_over_loopback_http(peer):
    # 1. Real loopback introduction produced matching 6-digit codes on both installations.
    assert peer.sender_code == peer.receiver_code
    assert len(peer.sender_code) == 6 and peer.sender_code.isdigit()

    # 2. Socket-bound caller gate, sub-path boundary and wire schema/header refusals.
    good = peer.headers()
    bad_key = {**good, "X-Ouroboros-Link": good["X-Ouroboros-Link"].split(".")[0] + ".not-the-secret"}
    for method, path in (("POST", "/api/letters"), ("GET", "/api/settings")):
        res = httpx.request(method, peer.url + path, headers=bad_key, json=_letter(), timeout=TIMEOUT)
        assert res.status_code == 401 and _code(res) == "link_unknown", (method, path, res.text)
    for method, path in (("GET", "/api/settings"), ("POST", "/api/tasks"),
                         ("GET", "/api/letters/x/events"), ("GET", "/api/letters/x/result/more")):
        res = httpx.request(method, peer.url + path, headers=good, json={}, timeout=TIMEOUT)
        assert res.status_code == 403 and _code(res) == "peer_route_forbidden", (method, path, res.text)
    for extra, code in (({"workspace_root": "/"}, "unexpected_fields"),
                        ({"project_id": "p"}, "unexpected_fields"),
                        ({"attachments": [{"name": "a.txt"}]}, "attachments_unsupported")):
        refused = _post(peer, _letter(**extra))
        assert refused.status_code == 400 and _code(refused) == code
    bad_proto = _post(peer, _letter(), **{"X-Ouroboros-Protocol": "2"})
    assert bad_proto.status_code == 400 and _code(bad_proto) == "protocol_unsupported"

    # 3. Full work-letter wire round-trip: admit -> premature /result 409 -> question -> answer -> unmocked /result 200.
    body = _letter(title="Files", cost_ceiling_usd=2.5,
                   chain=[{"install_id": "x", "label": "Laptop", "task_id": "t1"}])
    admitted = _post(peer, body)
    assert admitted.status_code == 200, admitted.text
    assert admitted.json()["letter_id"] == body["letter_id"]
    assert admitted.json()["state"] == "accepted_running" and admitted.json()["phase"] in {"queued", "running"}

    early = httpx.get(peer.url + f"/api/letters/{body['letter_id']}/result", headers=good, timeout=TIMEOUT)
    assert early.status_code == 409 and early.json()["error"] == "not_done"
    assert early.json()["state"] == "accepted_running"

    question_id = peer.ask(body["letter_id"], "Which folder do you mean?")
    waiting = httpx.get(peer.url + f"/api/letters/{body['letter_id']}", headers=good, timeout=TIMEOUT).json()
    assert waiting["state"] == "needs_input"
    assert waiting["question"] == {"question_id": question_id, "text": "Which folder do you mean?"}

    reply = {"letter_id": "r-" + body["letter_id"], "text": "/tmp itself.", "protocol": 1,
             "in_reply_to": {"letter_id": body["letter_id"], "question_id": question_id}}
    answered = _post(peer, reply)
    assert answered.status_code == 200, answered.text
    receipt = answered.json()  # POST's own form for an answer (LetterAnswerReceipt), never the letter's GET state
    assert set(receipt) == {"letter_id", "in_reply_to", "delivered", "msg_id"} and receipt["delivered"] is True
    assert (receipt["letter_id"], receipt["in_reply_to"]) == (reply["letter_id"], reply["in_reply_to"])
    again = _post(peer, reply)
    assert again.status_code == 200 and again.json() == receipt, "the same bytes get the same receipt"
    resumed = httpx.get(peer.url + f"/api/letters/{body['letter_id']}", headers=good, timeout=TIMEOUT).json()
    assert resumed["state"] == "accepted_running" and "question" not in resumed

    peer.finish(body["letter_id"], "There are 3 files.")
    done = httpx.get(peer.url + f"/api/letters/{body['letter_id']}/result", headers=good, timeout=TIMEOUT)
    assert done.status_code == 200, done.text
    done_body = done.json()
    assert done_body["letter_id"] == body["letter_id"] and done_body["state"] == "done"
    assert done_body["result"] == "There are 3 files." and done_body["result_available"] is True
    cost = done_body.get("cost")
    assert cost is None or (cost["known_usd"] is None or isinstance(cost["known_usd"], (int, float)))

    # 4. Sender cancellation over HTTP and 404 on unknown letter id.
    to_cancel = _letter()
    assert _post(peer, to_cancel).status_code == 200
    cancelled = httpx.post(peer.url + f"/api/letters/{to_cancel['letter_id']}/cancel", headers=good, timeout=TIMEOUT)
    assert cancelled.status_code == 200 and cancelled.json()["state"] == "cancelled", cancelled.text
    status = httpx.get(peer.url + f"/api/letters/{to_cancel['letter_id']}", headers=good, timeout=TIMEOUT)
    assert status.json()["state"] == "cancelled"
    unknown = httpx.post(peer.url + "/api/letters/never-sent/cancel", headers=good, timeout=TIMEOUT)
    assert unknown.status_code == 404 and _code(unknown) == "letter_not_found"
