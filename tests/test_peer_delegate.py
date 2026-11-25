"""The sending side of a letter to my human's linked Ouroboros: ``delegate_to_ouroboros``, ``gateways/peer.py`` and
the ``peer`` kind of ``execution_domain.bound_gateway``.

Most tests answer from an in-process stand-in shaped by the wire contract (headers, the four routes, refusal
codes, 503 ``unconfirmed``, the result's optional ``finished_at`` and ``cost``), so outcomes the real receiver
rarely gives (a dead transport, a 503) are reachable; the last test sends through the real receiver's gateway of a
second installation. No network, and building the owned Claudexor gateway fails the test.
"""
from __future__ import annotations

import inspect
import json

import httpx
import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import execution_domain as domain
from ouroboros import peer_links
from ouroboros.gateways import claudexor as wire
from ouroboros.gateways import peer as peerwire
from ouroboros.tools import peer_delegate

from tests._safety_fixtures import supervisor as supervisor_fixture

supervisor = supervisor_fixture  # the hermetic Supervisor: the real check path over a fake physical model
LINK_ID, PLACE_ID, SECRET = "link-0001", "place-1", "sekrit-value"
PEER_REF = {"kind": "peer", "place_id": PLACE_ID, "link_id": LINK_ID}


class FakeReceiver:
    """A linked server in-process: letters by letter_id, answers shaped by the contract."""

    def __init__(self):
        self.letters: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []
        self.posts: list[bytes] = []
        self.hooks: list = []  # callables(request) -> Optional[httpx.Response], asked first

    def answer(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        if request.url.path == "/api/letters" and request.method == "POST":
            self.posts.append(bytes(request.content))
        for hook in self.hooks:
            response = hook(request)
            if response is not None:
                return response
        if request.headers.get("X-Ouroboros-Link") != f"{LINK_ID}.{SECRET}":
            return httpx.Response(401, json={"error": "link_unknown", "refusal_code": "link_unknown"})
        if request.headers.get("X-Ouroboros-Install") != "install-a" or request.headers.get("X-Ouroboros-Protocol") != "1":
            return httpx.Response(403, json={"error": "install_mismatch", "refusal_code": "install_mismatch"})
        path = request.url.path
        if path == "/api/letters" and request.method == "POST":
            body = json.loads(request.content)
            letter_id = body["letter_id"]
            if body.get("in_reply_to"):
                if {"cost_ceiling_usd", "title", "chain"} & set(body):
                    return httpx.Response(400, json={"error": "unexpected_fields", "refusal_code": "unexpected_fields"})
                self.letters.setdefault(body["in_reply_to"]["letter_id"], {}).setdefault("answers", []).append(body)
                return httpx.Response(200, json={  # LetterAnswerReceipt: POST's own form for an answer
                    "letter_id": letter_id, "in_reply_to": dict(body["in_reply_to"]), "delivered": True,
                    "msg_id": f"fake-{letter_id}"})
            else:
                self.letters.setdefault(letter_id, {"body": body, "state": "accepted_running", "phase": "queued"})
            return httpx.Response(200, json={"letter_id": letter_id, "state": "accepted_running", "phase": "queued"})
        letter_id = path.split("/")[3]
        letter = self.letters.get(letter_id)
        if letter is None:
            return httpx.Response(404, json={"error": "letter_not_found", "letter_id": letter_id})
        if path.endswith("/result"):
            if letter["state"] != "done":
                return httpx.Response(409, json={"error": "not_done", "state": letter["state"]})
            body: dict = {"letter_id": letter_id, "state": "done"}
            omit_avail = bool(letter.get("omit_result_available"))
            if "result" in letter:
                body["result"] = letter["result"]
                if "result_available" not in letter and not omit_avail:
                    body["result_available"] = True
            elif "result_available" not in letter and not omit_avail:
                body["result"] = ""
                body["result_available"] = True
            for key in ("result_available", "result_held", "finished_at", "cost"):
                if key in letter:
                    body[key] = letter[key]
            return httpx.Response(200, json=body)
        if path.endswith("/cancel"):
            letter["state"] = "cancelled"
            return httpx.Response(200, json={"letter_id": letter_id, "state": "cancelled"})
        if path.endswith("/state"):  # a Stop's read-back: the state alone, answered so
            return httpx.Response(200, json={"letter_id": letter_id, "view": "state", **{key: letter[key] for key in (
                "state", "phase") if key in letter}})
        return httpx.Response(200, json={"letter_id": letter_id, **{key: letter[key] for key in (
            "state", "phase", "question", "error") if key in letter}})


@pytest.fixture
def link(tmp_path):
    """A real outgoing link of this installation: its secret in its owner-only file under the data root."""
    peer_links._write_secret(tmp_path, LINK_ID, SECRET)
    return peer_links.OutgoingLink(link_id=LINK_ID, place_id=PLACE_ID, peer_url="http://127.0.0.1:47231",
                                   peer_label="Velm", peer_install_id="install-b", own_install_id="install-a",
                                   drive_root=str(tmp_path))


@pytest.fixture
def receiver(monkeypatch, link):
    fake = FakeReceiver()
    monkeypatch.setattr(peer_links, "outgoing_link",
                        lambda place_id, drive_root=None: link if place_id == PLACE_ID else None)
    real_client = httpx.Client
    monkeypatch.setattr(peerwire.httpx, "Client",
                        lambda **kw: real_client(**kw, transport=httpx.MockTransport(fake.answer)))
    monkeypatch.setattr(wire, "ClaudexorGateway", lambda *a, **k: pytest.fail("the owned engine was built for a letter"))
    # This stand-in has no place store: the words boundary of an answer is read in tests/test_peer_words_boundary.py,
    # and a letter's one Safety reading (``safety.check_peer_assignment``) in tests/test_peer_assignment_safety.py and
    # through the real registry below.
    monkeypatch.setattr("ouroboros.peer_letters.answer_refusal", lambda ctx, entry, interaction_id, answers: "")
    monkeypatch.setattr("ouroboros.safety.check_peer_assignment", lambda ctx, **kw: (True, ""))
    return fake


@pytest.fixture
def ctx(tmp_path):
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolContext

    context = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    context.task_id = "t-a"
    context.task_metadata = {"root_task_id": "t-a", "parent_task_id": "t-a"}
    write_task_result(tmp_path, "t-a", "running", root_task_id="t-a", task_attempt=1)
    return context


def _call(ctx, **kwargs):
    return json.loads(peer_delegate._delegate_to_ouroboros(ctx, **kwargs).text)


def _record(root, invocation_id):
    return custody.invocation_record(root, invocation_id)


def test_the_intention_lands_before_the_post_and_the_run_names_its_link(ctx, tmp_path, receiver):
    seen = {}

    def before_post(request):
        if request.url.path == "/api/letters":
            body = json.loads(request.content)
            rows = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()]
            seen["requested"] = any(row.get("type") == custody.START_REQUESTED and row.get("engine_ref") == PEER_REF
                                    and row.get("invocation_id") == body["letter_id"] for row in rows)
            seen["headers"] = dict(request.headers)
        return None

    receiver.hooks.append(before_post)
    out = _call(ctx, place_id=PLACE_ID, text="check the build", title="CI?")
    assert out["status"] == "started" and seen["requested"] is True, out
    assert current.run(tmp_path, out["run_id"]).engine_ref == PEER_REF
    post = json.loads(receiver.posts[-1])
    assert (post["letter_id"], post["text"], post["title"], post["protocol"]) == (out["run_id"], "check the build",
                                                                                  "CI?", 1)
    assert post["chain"] == [{"install_id": "install-a", "task_id": "t-a"}] and "cost_ceiling_usd" not in post
    assert SECRET not in json.dumps(out) and SECRET not in json.dumps(_record(tmp_path, out["run_id"]))
    assert seen["headers"]["x-ouroboros-link"] == f"{LINK_ID}.{SECRET}", "the key travels in its header only"


def test_a_letter_written_by_a_letter_root_carries_its_chain_on(ctx, receiver):
    carried = [{"install_id": "install-c", "label": "Desk", "task_id": "t-c"}]
    ctx.task_metadata = {**ctx.task_metadata, "peer_letter": {"place_id": "elsewhere", "chain": carried}}
    assert _call(ctx, place_id=PLACE_ID, text="pass it on")["status"] == "started"
    assert json.loads(receiver.posts[-1])["chain"] == [*carried, {"install_id": "install-a", "task_id": "t-a"}]


@pytest.mark.parametrize("failure", ["503", "transport", "foreign_answer"])
def test_unknown_acceptance_is_never_resent_alone_and_retry_of_sends_the_same_bytes(ctx, tmp_path, receiver, failure):
    """A 503, a dead transport or an answer naming another letter: acceptance unknown, nothing recorded as refused,
    nothing resent; ``retry_of`` sends exactly the recorded bytes and binds the run."""
    armed = {"left": 1}

    def once(request):
        if request.url.path == "/api/letters" and armed["left"]:
            armed["left"] = 0
            if failure == "transport":
                raise httpx.ReadTimeout("the answer never came", request=request)
            if failure == "503":
                return httpx.Response(503, json={"error": "unconfirmed"})
            return httpx.Response(200, json={"letter_id": "another", "state": "accepted_running"})
        return None

    receiver.hooks.append(once)
    out = _call(ctx, place_id=PLACE_ID, text="wire flaky", cost_ceiling_usd=2)
    assert out["status"] == "unconfirmed", out
    token = out["pending_invocation_id"]
    assert _record(tmp_path, token)["state"] == "pending" and current.run(tmp_path, token) is None
    assert len(receiver.posts) == 1, "an unknown outcome is not resent on its own"
    retry = _call(ctx, place_id=PLACE_ID, retry_of=token)
    assert retry["status"] == "started" and retry["run_id"] == token, retry
    assert receiver.posts[1] == receiver.posts[0], "the same letter id and the same bytes"
    again = _call(ctx, place_id=PLACE_ID, retry_of=token)
    assert again["reason"] == "invocation_already_started" and len(receiver.posts) == 2
    assert _call(ctx, place_id=PLACE_ID, retry_of="no-such-id")["reason"] == "unknown_invocation"


def test_a_retry_the_link_cannot_carry_now_leaves_the_letter_unknown(ctx, tmp_path, receiver, monkeypatch, link):
    """A link paused (or revoked) after an unconfirmed send: the retry sends nothing and the recorded letter stays
    pending, never definitely refused, so the same bytes go once the link carries letters again."""
    receiver.hooks.append(lambda request: httpx.Response(503, json={"error": "unconfirmed"})
                          if request.url.path == "/api/letters" and not receiver.posts[1:] else None)
    token = _call(ctx, place_id=PLACE_ID, text="flaky then paused")["pending_invocation_id"]
    monkeypatch.setattr(peer_links, "outgoing_link", lambda place_id, drive_root=None: None)
    paused = _call(ctx, place_id=PLACE_ID, retry_of=token)
    assert (paused["reason"], paused["pending_invocation_id"]) == ("peer_link_unavailable", token), paused
    assert "definitely_unrun" not in paused and _record(tmp_path, token)["state"] == "pending"
    monkeypatch.setattr(peer_links, "outgoing_link",
                        lambda place_id, drive_root=None: link if place_id == PLACE_ID else None)
    assert _call(ctx, place_id=PLACE_ID, retry_of=token)["status"] == "started"
    assert len(receiver.posts) == 2 and receiver.posts[1] == receiver.posts[0]


def test_a_receiver_refusal_is_definite_typed_and_never_retried(ctx, tmp_path, receiver):
    receiver.hooks.append(lambda request: httpx.Response(403, json={
        "error": "peer_may_not_start", "refusal_code": "peer_may_not_start", "detail": "it only answers here"})
        if request.url.path == "/api/letters" else None)
    out = _call(ctx, place_id=PLACE_ID, text="refuse me")
    assert (out["status"], out["reason"], out["refusal_code"]) == ("refused", "peer_may_not_start",
                                                                    "peer_may_not_start"), out
    assert out["definitely_unrun"] is True and _record(tmp_path, out["pending_invocation_id"])["state"] == "failed_definite"
    assert _call(ctx, place_id=PLACE_ID, retry_of=out["pending_invocation_id"])["reason"] == "invocation_definitely_refused"
    assert len(receiver.posts) == 1


def test_refusals_before_any_post(ctx, receiver, monkeypatch):
    handler = peer_delegate.get_tools()[0].handler
    with pytest.raises(TypeError):  # the registry's binding: a delegate_start field is a typed TOOL_ARG_ERROR here
        inspect.signature(handler).bind(ctx, place_id=PLACE_ID, text="hi", model="gpt-x")
    assert {"model", "effort", "access", "workspace"}.isdisjoint(
        peer_delegate.get_tools()[0].schema["parameters"]["properties"])
    assert _call(ctx, place_id=PLACE_ID, text="   ")["reason"] == "empty_prompt"
    assert _call(ctx, place_id="", text="hi")["reason"] == "missing_place_id"
    assert _call(ctx, place_id=PLACE_ID, text="hi", cost_ceiling_usd=-1)["reason"] == "cost_ceiling_invalid"
    monkeypatch.setattr(peer_links, "outgoing_link", lambda place_id, drive_root=None: None)
    assert _call(ctx, place_id=PLACE_ID, text="hi")["reason"] == "peer_link_unavailable"
    assert receiver.posts == []


def test_the_ceiling_never_exceeds_the_trees_remaining_known_budget(ctx, receiver, monkeypatch):
    monkeypatch.setattr(peer_delegate, "_remaining_known_usd", lambda c, drive: 3.0)
    assert _call(ctx, place_id=PLACE_ID, text="capped", cost_ceiling_usd=10.0)["cost_ceiling_usd"] == 3.0
    assert json.loads(receiver.posts[-1])["cost_ceiling_usd"] == 3.0
    _call(ctx, place_id=PLACE_ID, text="capped when omitted")
    assert json.loads(receiver.posts[-1])["cost_ceiling_usd"] == 3.0
    monkeypatch.setattr(peer_delegate, "_remaining_known_usd", lambda c, drive: 0.0)
    assert _call(ctx, place_id=PLACE_ID, text="nothing left")["reason"] == "sender_budget_exhausted"
    assert len(receiver.posts) == 2


def test_the_trees_remaining_counts_its_own_in_task_stop(ctx, tmp_path, monkeypatch):
    """A sender that is itself a letter's root (or a place's work) stops at its producer ceiling: what it may pass on
    is that ceiling less the tree's known spend, when it is less than the ledger caps' room."""
    from ouroboros import usage_admission
    from ouroboros.task_pacing import CostCeiling

    snapshot = {"remaining_known_usd": 18.0, "settled_usd": 2.0}
    monkeypatch.setattr(usage_admission, "task_money_snapshot", lambda root, task, root_id: snapshot)
    assert peer_delegate._remaining_known_usd(ctx, tmp_path) == 18.0
    ctx._cost_ceiling = CostCeiling(state="active", ceiling_usd=5.0)
    assert peer_delegate._remaining_known_usd(ctx, tmp_path) == 3.0
    snapshot["remaining_known_usd"] = None  # no ledger cap binds: the stop alone
    assert peer_delegate._remaining_known_usd(ctx, tmp_path) == 3.0
    ctx._cost_ceiling = CostCeiling(state="disabled")
    assert peer_delegate._remaining_known_usd(ctx, tmp_path) is None
    monkeypatch.setattr(usage_admission, "task_money_snapshot", lambda root, task, root_id: None)
    ctx._cost_ceiling = CostCeiling(state="active", ceiling_usd=5.0)
    assert peer_delegate._remaining_known_usd(ctx, tmp_path) is None, "an unreadable ledger stays unknown"


def test_a_question_waits_on_user_and_the_same_answer_is_the_same_letter(ctx, receiver):
    from ouroboros.delegate_interactions import _delegate_answer
    from ouroboros.tools import delegate as delegate_tools

    run_id = _call(ctx, place_id=PLACE_ID, text="ask me something")["run_id"]
    receiver.letters[run_id].update(state="needs_input", question={"question_id": "q1", "text": "which branch?"})
    payload = json.loads(delegate_tools._read_run(ctx, run_id, 0))
    assert payload["status"] == "waiting_on_user", payload
    interaction = payload["pending_interactions"][0]
    assert interaction["interaction_id"] == "q1" and interaction["questions"][0]["question"] == "which branch?"
    lost: list = []
    receiver.hooks.append(lambda request: (lost.append(1) or httpx.Response(503, json={"error": "unconfirmed"}))
                          if request.method == "POST" and b"in_reply_to" in request.content and not lost else None)
    rows = [{"question_id": "q1", "free_text": "main"}]
    unknown = json.loads(_delegate_answer(ctx, run_id, "q1", rows).text)
    assert unknown["status"] == "delivery_unknown", unknown
    answered = json.loads(_delegate_answer(ctx, run_id, "q1", rows).text)  # its delivery unknown: sent again
    assert answered["status"] == "delivered" and answered["receipt"]["msg_id"], answered
    assert receiver.posts[-1] == receiver.posts[-2], "the same answer is the same letter, byte for byte"
    (sent,) = receiver.letters[run_id]["answers"]
    assert sent["in_reply_to"] == {"letter_id": run_id, "question_id": "q1"}
    assert sent["text"] == "main" and not {"cost_ceiling_usd", "title", "chain"} & set(sent)
    posts = len(receiver.posts)
    again = json.loads(_delegate_answer(ctx, run_id, "q1", rows).text)  # confirmed: its receipt, nothing sent
    assert again["status"] == "delivered" and again.get("replayed") is True, again
    assert again["receipt"] == answered["receipt"] and len(receiver.posts) == posts


def test_an_answer_whose_words_are_refused_sends_nothing(ctx, receiver, monkeypatch):
    """The answer's exact words meet the link's words boundary before any wire call: refused, no answer letter
    leaves, its bytes are not pinned and the question stays open for another answer."""
    from ouroboros.delegate_interactions import _delegate_answer

    seen: list = []

    def refuse(_ctx, entry, interaction_id, answers):
        seen.append((entry.engine_ref, interaction_id, answers))
        return "⚠️ SAFETY_RETURNED: not run. (a fictional reading)"
    monkeypatch.setattr("ouroboros.peer_letters.answer_refusal", refuse)
    run_id = _call(ctx, place_id=PLACE_ID, text="ask me")["run_id"]
    receiver.letters[run_id].update(state="needs_input", question={"question_id": "q1", "text": "which?"})
    posts = len(receiver.posts)
    out = json.loads(_delegate_answer(ctx, run_id, "q1", [{"question_id": "q1", "free_text": "the secret path"}]).text)
    assert (out["status"], out["reason"]) == ("refused", "answer_words_refused"), out
    assert out["detail"] == "⚠️ SAFETY_RETURNED: not run. (a fictional reading)", "this side's refusal alone"
    assert len(receiver.posts) == posts and "answers" not in receiver.letters[run_id], "nothing left"
    assert seen and seen[0][:2] == (PEER_REF, "q1"), seen
    monkeypatch.setattr("ouroboros.peer_letters.answer_refusal", lambda *_args: "")
    rewritten = json.loads(_delegate_answer(ctx, run_id, "q1", [{"question_id": "q1", "free_text": "main"}]).text)
    assert rewritten["status"] == "delivered", "a refused answer pinned nothing: other words may still answer"
    assert [row["text"] for row in receiver.letters[run_id]["answers"]] == ["main"]


def test_waiting_owner_holds_the_wait_without_a_question(ctx, receiver):
    from ouroboros.tools import delegate as delegate_tools

    run_id = _call(ctx, place_id=PLACE_ID, text="owner question pending")["run_id"]
    receiver.letters[run_id]["state"] = "waiting_owner"
    payload = json.loads(delegate_tools._read_run(ctx, run_id, 0))
    assert payload["status"] in ("progress", "no_progress") and payload["waiting_on_user"] is True, payload
    assert not payload.get("pending_interactions")


def test_cancel_reaches_the_linked_server(ctx, receiver):
    from ouroboros.tools import delegate as delegate_tools

    run_id = _call(ctx, place_id=PLACE_ID, text="stop this")["run_id"]
    result = json.loads(delegate_tools._delegate_cancel(ctx, run_id, reason="no longer needed").text)
    assert result["status"] == "confirmed" and result["run_may_still_be_live"] is False, result
    assert ("POST", f"/api/letters/{run_id}/cancel") in receiver.calls and receiver.letters[run_id]["state"] == "cancelled"


def test_a_pause_of_my_place_lets_only_my_stop_of_a_letter_i_started_out(ctx, receiver, monkeypatch, link):
    """My human pauses this side's place of the link while letters I sent run there: no new letter, read or answer
    leaves (each is refused before any request), yet my own Stop of one started letter reaches the other side and is
    read back, so a Pause never strands work I started (``peer_links.stopping_link``); the other letter runs on."""
    from ouroboros.delegate_interactions import _delegate_answer
    from ouroboros.tools import delegate as delegate_tools

    run_id = _call(ctx, place_id=PLACE_ID, text="stop me later")["run_id"]
    other = _call(ctx, place_id=PLACE_ID, text="keep running")["run_id"]
    receiver.letters[run_id].update(state="needs_input", question={"question_id": "q-1", "text": "Which?"})
    monkeypatch.setattr(peer_links, "outgoing_link", lambda place_id, drive_root=None: None)  # my place is Paused
    monkeypatch.setattr(peer_links, "stopping_link",
                        lambda place_id, drive_root=None: link if place_id == PLACE_ID else None)
    before = len(receiver.calls)
    assert _call(ctx, place_id=PLACE_ID, text="one more")["reason"] == "peer_link_unavailable"
    read = delegate_tools._read_run(ctx, other, 0)
    answered = _delegate_answer(ctx, run_id, "q-1", [{"question_id": "q-1", "free_text": "this one"}]).text
    assert receiver.calls[before:] == [], (receiver.calls[before:], read, answered)
    stopped = json.loads(delegate_tools._delegate_cancel(ctx, run_id, reason="no longer needed").text)
    assert stopped["status"] == "confirmed" and receiver.letters[run_id]["state"] == "cancelled", stopped
    assert receiver.calls[before:] == [("POST", f"/api/letters/{run_id}/cancel"),
                                       ("GET", f"/api/letters/{run_id}/state")]
    assert receiver.letters[other]["state"] == "accepted_running", "a Stop of one letter stops no other"


def test_a_stop_read_back_through_a_pause_carries_the_letters_state_alone(link, monkeypatch):
    """Mock double: through a Pause of my place an ordinary get_run sends nothing — even AFTER cancel_run was called
    on the same PeerGateway; only explicit for_stop=True crosses Pause, reading the receiver's state-only view
    (``/state``, no question text, failure text or /result crosses). Even on an Active link, for_stop=True on a done
    letter never hits /result; an earlier receiver that refuses ``/state`` is read nothing else in its place."""
    seen: list[tuple[str, str]] = []
    answers = {"GET": {"letter_id": "L1", "state": "needs_input",
                       "question": {"question_id": "q1", "text": "WORDS-OF-THE-QUESTION"}},
               "POST": {"letter_id": "L1", "state": "needs_input"}}
    earlier = {"version": False}

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.url.path.endswith("/state"):
            if earlier["version"]:  # a receiver without the route: its gate refuses it
                return httpx.Response(403, json={"error": "peer_route_forbidden",
                                                 "refusal_code": "peer_route_forbidden"})
            return httpx.Response(200, json={"letter_id": "L1", "view": "state", "state": answers["GET"]["state"]})
        return httpx.Response(200, json=answers[request.method])

    real_client = httpx.Client
    monkeypatch.setattr(peerwire.httpx, "Client", lambda **kw: real_client(**kw, transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(peer_links, "outgoing_link", lambda place_id, drive_root=None: None)  # my place is Paused
    monkeypatch.setattr(peer_links, "stopping_link", lambda place_id, drive_root=None: link)
    gw = peerwire.PeerGateway(link)
    try:
        with pytest.raises(wire.ClaudexorUnavailable) as observed:
            gw.get_run("L1")
        assert observed.value.code == "peer_link_unavailable" and seen == [], "an observer's read sends nothing"
        gw.cancel_run("L1")
        with pytest.raises(wire.ClaudexorUnavailable) as after_cancel:
            gw.get_run("L1")
        assert after_cancel.value.code == "peer_link_unavailable", "ordinary get_run gains no Pause rights from Stop"
        assert seen == [("POST", "/api/letters/L1/cancel")]
        detail = gw.get_run("L1", for_stop=True)
        assert seen == [("POST", "/api/letters/L1/cancel"), ("GET", "/api/letters/L1/state")]
        assert detail["summary"]["state"] == "running" and detail["summary"]["contentWithheld"] == "paused"
        assert detail["pendingInteractions"] == [] and "WORDS-OF-THE-QUESTION" not in json.dumps(detail)
        answers["GET"] = {"letter_id": "L1", "state": "done"}
        done = gw.get_run("L1", for_stop=True)
        assert done["summary"]["state"] == "succeeded" and done["summary"]["contentWithheld"] == "paused"
        assert "primaryOutput" not in done and "finalSummary" not in done
        with pytest.raises(wire.ClaudexorUnavailable):
            gw.get_run_result("L1")
        assert not any(path.endswith("/result") for _method, path in seen), seen

        # Even on an Active link, explicit for_stop=True on a done letter observes state alone without calling /result.
        monkeypatch.setattr(peer_links, "outgoing_link", lambda place_id, drive_root=None: link)
        seen.clear()
        done_active = gw.get_run("L1", for_stop=True)
        assert seen == [("GET", "/api/letters/L1/state")]
        assert done_active["summary"]["state"] == "succeeded" and done_active["summary"]["contentWithheld"] == "stop"
        assert "primaryOutput" not in done_active and "finalSummary" not in done_active

        # An earlier receiver refuses /state: the Stop's read is unsupported, never its full status in its place.
        earlier["version"] = True
        answers["GET"] = {"letter_id": "L1", "state": "needs_input",
                          "question": {"question_id": "q1", "text": "WORDS-OF-THE-QUESTION"}}
        seen.clear()
        with pytest.raises(wire.ClaudexorUnavailable) as unsupported:
            gw.get_run("L1", for_stop=True)
        assert unsupported.value.code == "peer_state_read_unsupported", unsupported.value
        assert seen == [("GET", "/api/letters/L1/state")], "no fallback to the full status"
    finally:
        gw.close()


def test_a_done_letter_returns_its_result_and_unknown_cost_stays_unknown(ctx, receiver):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.tools import delegate as delegate_tools

    run_id = _call(ctx, place_id=PLACE_ID, text="do it")["run_id"]
    with domain.bound_gateway(PEER_REF) as gateway, pytest.raises(ClaudexorUnavailable) as early:
        gateway.get_run_result(run_id)
    assert (early.value.code, early.value.status_code) == ("not_done", 409)
    receiver.letters[run_id].update(state="done", result="all green", finished_at="2026-11-23T00:30:00Z",
                                    cost={"known_usd": None, "final": False})
    payload = json.loads(delegate_tools._read_run(ctx, run_id, 0))
    assert (payload["status"], payload["state"], payload["primary_output"]["text"]) == ("terminal", "succeeded",
                                                                                         "all green"), payload
    assert payload["final_summary"] == {
        "finished_at": "2026-11-23T00:30:00Z", "result_available": True, "cost": {"known_usd": None, "final": False},
    }
    assert payload["settlement"]["settled"] is True and payload["settlement"]["ledger_recorded"] is False, payload
    assert payload["settlement"]["peer_billed"] is True, payload


def test_bound_gateway_resolves_each_engine_kind(monkeypatch):
    sentinel_work, sentinel_peer = object(), object()
    monkeypatch.setattr(domain, "work_gateway", lambda ref, **kw: sentinel_work)
    monkeypatch.setattr(peerwire, "peer_gateway", lambda ref, **kw: sentinel_peer)
    assert domain.bound_gateway({}) is None
    assert domain.bound_gateway({"kind": "work", "descriptor": "d"}) is sentinel_work
    assert domain.run_gateway(PEER_REF, lambda: pytest.fail("the owned default for a letter")) is sentinel_peer
    assert domain.engine_factory(PEER_REF, lambda: "owned")() is sentinel_peer
    assert domain.engine_factory({}, lambda: "owned")() == "owned"
    assert domain.engine_key({"engine_ref": PEER_REF}) == ("peer", f"{PLACE_ID}\0{LINK_ID}")
    groups = domain.engine_groups([{"engine_ref": PEER_REF}, {}], [])
    assert groups[0][0] == {} and groups[1][0] == PEER_REF


def test_after_a_restart_wait_cancel_supervision_and_collection_reach_the_linked_server(ctx, tmp_path, receiver):
    from ouroboros import delegate_supervision
    from ouroboros.tools import delegate as delegate_tools

    entry = custody.RunCustody(run_id="letter-restart", task_id="t-a", invocation_id="letter-restart",
                               engine_ref=dict(PEER_REF), route_id="peer", category="peer", source="peer_letter",
                               root_task_id="t-a", parent_task_id="t-a", ledger_root=str(tmp_path), access="readonly")
    assert custody.record_started(tmp_path, entry, {"delegated": True})
    custody._CUSTODY.clear()  # the restart: only the durable store speaks now
    reloaded = current.run(tmp_path, "letter-restart")
    assert reloaded is not None and reloaded.engine_ref == PEER_REF
    receiver.letters["letter-restart"] = {"state": "accepted_running", "phase": "running"}
    payload = json.loads(delegate_tools._read_run(ctx, "letter-restart", 0))
    assert payload["state"] == "running" and ("GET", "/api/letters/letter-restart") in receiver.calls, payload
    gateway = delegate_supervision._loop_gateway(reloaded)
    assert isinstance(gateway, peerwire.PeerGateway)
    gateway.close()
    clients = domain.EngineClients()
    client = clients.get(reloaded)
    assert isinstance(client, peerwire.PeerGateway) and clients.reach(reloaded) is client
    client.close()
    result = json.loads(delegate_tools._delegate_cancel(ctx, "letter-restart", reason="restart test").text)
    assert result["status"] == "confirmed" and ("POST", "/api/letters/letter-restart/cancel") in receiver.calls


def test_a_letter_through_the_real_receiver_of_a_second_installation(tmp_path, monkeypatch):
    """A writes through the real gate and letter routes of B (the in-process twin of S45 in
    tests/system_e2e/test_system_scenarios_w11.py): B's work starts with the letter's ceiling, its status and
    question reach A, A's answer lands in the same root, the result returns to A; a pause on B's side then refuses
    the next letter, and B's revocation makes A's key unknown there."""
    from starlette.testclient import TestClient

    from ouroboros.install_identity import acting_as
    from ouroboros.peer_letters import ask
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools import delegate as delegate_tools
    from ouroboros.tools.registry import ToolContext
    from tests._peer_letters_shared import admitted, introduce, published, receiver_client, safety_reads

    from ouroboros.tools.registry_guards import _builtin_tool_availability

    lone = tmp_path / "lone"
    (lone / "state").mkdir(parents=True)
    unlinked = ToolContext(repo_dir=tmp_path, drive_root=lone)
    unlinked.task_id = "t-lone"
    assert _builtin_tool_availability(peer_delegate.TOOL_NAME, unlinked)[:2] == (False, "no_linked_ouroboros")
    pair = introduce(tmp_path, monkeypatch)  # B lets A start work; A only lets B answer
    reads = safety_reads(monkeypatch)  # A's letter is read once as sent, its answer once as words to B
    linked = ToolContext(repo_dir=pair.repo, drive_root=pair.root_a)
    linked.task_id = "t-a"
    assert _builtin_tool_availability(peer_delegate.TOOL_NAME, linked)[0] is True, "shown once a link is confirmed"
    monkeypatch.setattr(wire, "ClaudexorGateway", lambda *a, **k: pytest.fail("the owned engine was built"))
    real_outgoing = peer_links.outgoing_link
    monkeypatch.setattr(peer_links, "outgoing_link",
                        lambda place_id, drive_root=None: real_outgoing(place_id, drive_root=pair.root_a))
    with receiver_client(pair, monkeypatch) as client:
        monkeypatch.setattr(peerwire.httpx, "Client", lambda **kw: client if isinstance(client, TestClient) else None)
        monkeypatch.setattr(client, "close", lambda: None)  # one client serves every gateway of this test
        sender = ToolContext(repo_dir=pair.repo, drive_root=pair.root_a)
        sender.task_id, sender.task_metadata = "t-a", {"root_task_id": "t-a"}
        with acting_as(pair.root_a):
            write_task_result(pair.root_a, "t-a", "running", root_task_id="t-a", task_attempt=1)
            out = _call(sender, place_id=pair.place_a, text="Count the files in /tmp.", cost_ceiling_usd=0.5)
        assert out["status"] == "started", out
        row = pair.row(out["run_id"])
        assert row["metadata"]["peer_letter"]["link_id"] == pair.link_id
        assert row["metadata"]["root_cost_ceiling_usd"] == 0.5
        assert client.get(f"/api/letters/{out['run_id']}", headers=pair.headers()).json()["state"] == "accepted_running"
        with acting_as(pair.root_b):
            question = ask(pair.root_b, pair.task_id(out["run_id"]), "Which folder exactly?",
                           published=published(pair.place_b, "Which folder exactly?",
                                               root=pair.root_b, link_id=pair.link_id))
        status = client.get(f"/api/letters/{out['run_id']}", headers=pair.headers()).json()
        assert (status["state"], status["question"]["question_id"]) == ("needs_input", question["question_id"]), status
        with acting_as(pair.root_a):
            waiting = json.loads(delegate_tools._read_run(sender, out["run_id"], 0))
            assert waiting["status"] == "waiting_on_user", waiting
            from ouroboros.delegate_interactions import _delegate_answer

            answered = json.loads(_delegate_answer(sender, out["run_id"], question["question_id"],
                                                   [{"question_id": question["question_id"], "free_text": "/tmp"}]).text)
        assert answered["status"] == "delivered", answered
        (letter_tool, letter), answer = reads  # the letter read once as sent, the answer once as words there
        assert answer == ("place_reply", {"text": "/tmp", "place_id": pair.place_a}), reads
        assert letter_tool == "delegate_to_ouroboros" and "letter_id" not in letter, letter
        assert (letter["text"], letter["cost_ceiling_usd"], letter["place_id"], letter["link_id"]) == (
            "Count the files in /tmp.", 0.5, pair.place_a, pair.link_id), letter
        with acting_as(pair.root_b):
            import supervisor.queue as queue

            task_id = pair.task_id(out["run_id"])
            stored = load_task_result(pair.root_b, task_id, strict=True)
            assert stored["peer_questions"][question["question_id"]]["state"] == "answered"
            queue.PENDING[:] = [entry for entry in queue.PENDING if entry.get("id") != task_id]  # B's worker ran it
            write_task_result(pair.root_b, task_id, "completed", result="There are 3 files.",
                              peer_reply=admitted(pair, "There are 3 files."),  # admitted for A at its finish
                              finished_at="2026-11-23T01:00:00+00:00")
        with acting_as(pair.root_a):
            done = json.loads(delegate_tools._read_run(sender, out["run_id"], 0))
        # B stamped the final admitted for A's link: A reads exactly that text, marked available.
        assert (done["status"], done["state"]) == ("terminal", "succeeded"), done
        assert done["primary_output"] == {"text": "There are 3 files.", "bytes": 18, "truncated": False}, done
        assert done["final_summary"]["result_available"] is True, done
        assert done["settlement"]["settled"] is True and done["settlement"]["ledger_recorded"] is False, done
        assert done["settlement"]["peer_billed"] is True, done
        assert client.post(f"/api/places/{pair.place_b}", json={"state": "paused"}).status_code == 200
        with acting_as(pair.root_a):
            refused = _call(sender, place_id=pair.place_a, text="One more?")
        assert client.post(f"/api/peer-links/{pair.link_id}/revoke", json={}).json()["state"] == "removed"
        with acting_as(pair.root_a):
            unknown = _call(sender, place_id=pair.place_a, text="And now?")
    assert (refused["reason"], refused["refusal_code"]) == ("link_paused", "link_paused"), refused
    assert (unknown["reason"], unknown["refusal_code"], unknown["definitely_unrun"]) == (
        "link_unknown", "link_unknown", True), unknown
    with acting_as(pair.root_a):  # A's card shows it; with B's server gone, a refusal before any send proves none
        assert peer_links.link_card(pair.root_a, pair.place_a)["remote_unknown_at"]
        again = _call(sender, place_id=pair.place_a, text="Still there?")
    assert (again["reason"], again["refusal_code"]) == ("link_unknown", "link_unknown") and "since" in again["detail"]


def test_an_answer_refused_for_a_spent_ceiling_is_its_own_typed_outcome(ctx, receiver):
    """409 no_budget on an answer letter: not the rows' fault, so not the advice to fix and resend them."""
    from ouroboros.delegate_interactions import _delegate_answer

    run_id = _call(ctx, place_id=PLACE_ID, text="ask me")["run_id"]
    receiver.letters[run_id].update(state="needs_input", question={"question_id": "q1", "text": "which?"})
    receiver.hooks.append(lambda request: httpx.Response(409, json={
        "error": "no_budget", "refusal_code": "no_budget", "detail": "that letter's cost ceiling is spent"})
        if request.url.path == "/api/letters" and b"in_reply_to" in request.content else None)
    out = json.loads(_delegate_answer(ctx, run_id, "q1", [{"question_id": "q1", "free_text": "main"}]).text)
    assert (out["status"], out["accepted"]) == ("no_budget", False), out
    assert "fix the rows" not in json.dumps(out)


def test_a_chain_that_would_pass_sixteen_is_refused_before_any_post(ctx, tmp_path, receiver):
    ctx.task_metadata = {**ctx.task_metadata, "peer_letter": {"place_id": "elsewhere", "chain": [
        {"install_id": f"i{index}", "task_id": f"t{index}"} for index in range(16)]}}
    out = _call(ctx, place_id=PLACE_ID, text="one hop too many")
    assert (out["reason"], out["definitely_unrun"]) == ("chain_too_long", True), out
    assert receiver.posts == []


def test_sender_ledger_and_caps_never_book_receiver_spend_or_unknown_on_any_terminal(ctx, tmp_path, receiver):
    """Sender's usage.sqlite, caps and cost_final stay untouched across every peer terminal shape: positive final,
    non-final subtotal, non-final zero, unknown, failed and cancelled. Receiver cost rides only as attributed info."""
    from ouroboros import delegate_collection as collection
    from ouroboros import delegate_custody_current as custody_current
    from ouroboros import usage_accounting as ua
    from ouroboros import usage_store
    from ouroboros.delegate_evidence import task_execution_evidence
    from ouroboros.tools import delegate as delegate_tools
    from ouroboros.tools.delegate_terminal_evidence import _settlement_projection, retained_terminal_result

    cases = [
        ("done", {"result": "r1", "cost": {"known_usd": 0.42, "final": True}}, 0.42, True, False),
        ("done", {"result": "r2", "cost": {"known_usd": 0.15, "final": False}}, 0.15, False, True),
        ("done", {"result": "r3", "cost": {"known_usd": 0.0, "final": False}}, 0.0, False, True),
        ("done", {"result": "r4", "cost": {"known_usd": None, "final": False}}, None, False, False),
        ("failed", {"error": "boom"}, None, False, False),
        ("cancelled", {}, None, False, False),
    ]
    for state, extra, expected_usd, expected_final, expected_partial in cases:
        run_id = _call(ctx, place_id=PLACE_ID, text=f"case {state} {expected_usd}")["run_id"]
        receiver.letters[run_id].update(state=state, **extra)
        payload = json.loads(delegate_tools._read_run(ctx, run_id, 0))
        assert payload["status"] == "terminal", payload
        cost = payload["cost"]
        assert cost["billed_by"] == "peer" and cost["sender_cost_usd"] == 0.0, cost
        assert cost["cost_usd"] == expected_usd and cost["cost_final"] is expected_final, cost
        assert bool(cost.get("cost_partial")) is expected_partial, cost
        assert cost["receiver_cost"] == {"known_usd": expected_usd, "final": expected_final}, cost
        assert "BILLED" not in cost["note"], cost
        assert payload["settlement"]["settled"] is True and payload["settlement"]["ledger_recorded"] is False, payload
        assert payload["settlement"]["peer_billed"] is True, payload
        assert payload["collection"]["settlement"]["ledger_recorded"] is False, payload["collection"]
        assert payload["collection"]["settlement"]["peer_billed"] is True, payload["collection"]
        entry = custody_current.run(tmp_path, run_id)
        assert entry is not None and entry.settled is True
        assert custody_current.collection_pending(entry) is False
        reread = retained_terminal_result(ctx, run_id)
        assert reread is not None and reread["settlement"]["settled"] is True, reread
        assert reread["settlement"]["ledger_recorded"] is False and reread["settlement"]["peer_billed"] is True, reread
        repeat_collect = collection.collect_terminal(tmp_path, run_id, recapture=True)
        assert repeat_collect["settlement"]["settled"] is True, repeat_collect
        assert repeat_collect["settlement"]["ledger_recorded"] is False, repeat_collect
        assert repeat_collect["settlement"]["peer_billed"] is True, repeat_collect

    ordinary = custody.RunCustody(run_id="cx-ordinary", task_id="t-a", settled=True, ledger_recorded=True)
    ordinary_proj = _settlement_projection(ordinary)
    assert ordinary_proj["settled"] is True and ordinary_proj["ledger_recorded"] is True, ordinary_proj
    assert "peer_billed" not in ordinary_proj, ordinary_proj

    breakdown = ua.usage_breakdown(tmp_path, root_task_id="t-a", include_custody=True)
    assert breakdown["settled_usd"] == 0.0 and breakdown["accounted_usd"] == 0.0, breakdown
    assert breakdown["unknown_unmetered"] == 0 and breakdown["cost_final"] is True, breakdown
    assert breakdown["cost_custody"]["unrecorded_run_ids"] == [], breakdown
    assert breakdown["cost_custody"]["closed_unrecorded_run_ids"] == [], breakdown
    assert breakdown["cost_custody"]["open_run_ids"] == [], breakdown
    with usage_store.read(tmp_path) as txn:
        rows = txn.conn.execute("SELECT COUNT(*) FROM attempts WHERE kind='subscription_session'").fetchone()[0]
    assert rows == 0, "no peer letter may ever write a subscription_session row into the sender's usage.sqlite"
    ev = task_execution_evidence(tmp_path, "t-a")
    assert ev["delegated_runs_settled"] == len(cases)
    assert ev["subscription_cost_usd"] is None and ev["subscription_cost_partial"] is False, ev


def test_two_installations_keep_separate_ledgers_and_caps(tmp_path, monkeypatch):
    """Real A -> B letter: sender's own spend stays on A's ledger ($0.10), receiver's spend stays on B's ledger
    ($0.42), and A's terminal payload attributes B's $0.42 to B without adding it to A's ledger or caps."""
    from starlette.testclient import TestClient

    from ouroboros import usage_accounting as ua
    from ouroboros.install_identity import acting_as
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import delegate as delegate_tools
    from ouroboros.tools.registry import ToolContext
    from tests._peer_letters_shared import introduce, receiver_client, safety_reads

    pair = introduce(tmp_path, monkeypatch)
    safety_reads(monkeypatch)  # the letter's one reading (check_peer_assignment) answered: not this test's subject
    monkeypatch.setattr(wire, "ClaudexorGateway", lambda *a, **k: pytest.fail("owned engine built"))
    real_outgoing = peer_links.outgoing_link
    monkeypatch.setattr(peer_links, "outgoing_link",
                        lambda place_id, drive_root=None: real_outgoing(place_id, drive_root=pair.root_a))

    def _book_api_spend(root, task_id, amount):
        with acting_as(root):
            ua.record_subscription_session(
                f"local-session-{task_id}", drive_root=root, route="codex", model="gpt-5",
                task_id=task_id, root_task_id=task_id, spend_usd=amount,
            )

    with receiver_client(pair, monkeypatch) as client:
        monkeypatch.setattr(peerwire.httpx, "Client", lambda **kw: client if isinstance(client, TestClient) else None)
        monkeypatch.setattr(client, "close", lambda: None)
        sender = ToolContext(repo_dir=pair.repo, drive_root=pair.root_a)
        sender.task_id, sender.task_metadata = "t-a", {"root_task_id": "t-a"}
        with acting_as(pair.root_a):
            write_task_result(pair.root_a, "t-a", "running", root_task_id="t-a", task_attempt=1)
            _book_api_spend(pair.root_a, "t-a", 0.10)
            out = _call(sender, place_id=pair.place_a, text="Compute result.", cost_ceiling_usd=1.0)
        assert out["status"] == "started", out
        b_task_id = pair.task_id(out["run_id"])
        with acting_as(pair.root_b):
            import supervisor.queue as queue

            _book_api_spend(pair.root_b, b_task_id, 0.42)
            queue.PENDING[:] = [entry for entry in queue.PENDING if entry.get("id") != b_task_id]
            write_task_result(pair.root_b, b_task_id, "completed", result="done",
                              finished_at="2026-11-23T01:00:00+00:00", cost_usd=0.42, cost_final=True)
        with acting_as(pair.root_a):
            done = json.loads(delegate_tools._read_run(sender, out["run_id"], 0))
            a_money = ua.usage_breakdown(pair.root_a, root_task_id="t-a")
        with acting_as(pair.root_b):
            b_money = ua.usage_breakdown(pair.root_b, root_task_id=b_task_id)

    assert done["status"] == "terminal" and done["cost"]["billed_by"] == "peer", done
    assert done["cost"]["sender_cost_usd"] == 0.0 and done["cost"]["cost_usd"] == pytest.approx(0.42), done
    assert done["cost"]["cost_final"] is True, done
    assert a_money["settled_usd"] == pytest.approx(0.10) and a_money["cost_final"] is True, a_money
    assert b_money["settled_usd"] == pytest.approx(0.42) and b_money["cost_final"] is True, b_money


def test_lost_receipt_retry_recovers_on_paused_link_and_refusal_never_marks_definitely_unrun(ctx, tmp_path, receiver):
    """Branch A: first POST accepted on receiver, reply lost, link paused -> retry_of recovers via GET without
    re-POSTing. Branch B: first POST unconfirmed (not landed), link paused -> retry refusal keeps pending."""
    # Branch A: accepted on receiver, reply dropped, then link paused
    drop_first = {"active": True}

    def accept_then_drop(request):
        if request.url.path == "/api/letters" and request.method == "POST" and drop_first["active"]:
            drop_first["active"] = False
            body = json.loads(request.content)
            receiver.letters[body["letter_id"]] = {"body": body, "state": "accepted_running", "phase": "running"}
            raise httpx.ReadTimeout("reply lost after receiver accepted", request=request)
        return None

    receiver.hooks.append(accept_then_drop)
    first = _call(ctx, place_id=PLACE_ID, text="accepted but reply lost")
    assert first["status"] == "unconfirmed", first
    token_a = first["pending_invocation_id"]
    assert _record(tmp_path, token_a)["state"] == "pending"
    # Receiver pauses the link: POST /api/letters returns 403 link_paused, GET /api/letters/{id} still answers 200
    paused = {"on": True}
    receiver.hooks.append(lambda req: httpx.Response(403, json={"error": "link_paused", "refusal_code": "link_paused"})
                          if paused["on"] and req.url.path == "/api/letters" and req.method == "POST" else None)
    recovered = _call(ctx, place_id=PLACE_ID, retry_of=token_a)
    assert recovered["status"] == "started" and recovered["run_id"] == token_a, recovered
    assert recovered.get("recovered_via_lookup") is True and recovered.get("custody_durable") is True
    assert len(receiver.posts) == 1, "recovered via non-creating GET without re-POSTing"

    # Branch B: first POST unconfirmed (never created on receiver), then link paused -> retry keeps pending
    paused["on"] = False
    fail_once = {"on": True}
    receiver.hooks.append(lambda req: (fail_once.update(on=False) or httpx.Response(503, json={"error": "unconfirmed"}))
                          if fail_once["on"] and req.url.path == "/api/letters" and req.method == "POST" else None)
    second = _call(ctx, place_id=PLACE_ID, text="unconfirmed then paused")
    token_b = second["pending_invocation_id"]
    assert second["status"] == "unconfirmed" and _record(tmp_path, token_b)["state"] == "pending"
    paused["on"] = True
    refused_retry = _call(ctx, place_id=PLACE_ID, retry_of=token_b)
    assert refused_retry["reason"] == "link_paused" and refused_retry["pending_invocation_id"] == token_b
    assert "definitely_unrun" not in refused_retry, refused_retry
    assert _record(tmp_path, token_b)["state"] == "pending", "retry refusal must NEVER mark failed_definite"
    paused["on"] = False
    resumed = _call(ctx, place_id=PLACE_ID, retry_of=token_b)
    assert resumed["status"] == "started" and resumed["run_id"] == token_b, resumed


def test_owner_pause_blocks_handoff_before_network_and_launch_lock_releases_before_io(ctx, tmp_path, receiver, monkeypatch):
    """Pause landing between START_REQUESTED and POST blocks before network (fresh closes unsent; retry stays pending);
    on an admitted send, the launch lock is released BEFORE HTTP I/O."""
    from ouroboros import owner_pause

    # 1. Verify launch_lock is NOT held during HTTP I/O
    lock_free_during_post = {"ok": False}

    def check_lock(request):
        if request.url.path == "/api/letters" and request.method == "POST":
            with owner_pause.launch_lock(tmp_path, "t-a", wait=False):
                lock_free_during_post["ok"] = True
        return None

    receiver.hooks.append(check_lock)
    started = _call(ctx, place_id=PLACE_ID, text="lock release check")
    assert started["status"] == "started" and lock_free_during_post["ok"] is True
    receiver.hooks.clear()

    # 2. Pause landing AFTER START_REQUESTED and BEFORE POST on a fresh start
    real_run_op = owner_pause.run_operation

    def pause_at_handoff(source, fn, /, *args, **kwargs):
        raise owner_pause.OwnerPauseRefused("owner_pause")

    monkeypatch.setattr(owner_pause, "run_operation", pause_at_handoff)
    posts_before = len(receiver.posts)
    blocked = _call(ctx, place_id=PLACE_ID, text="paused before post")
    assert blocked["status"] == "refused" and blocked["definitely_unrun"] is True, blocked
    assert len(receiver.posts) == posts_before, "zero HTTP POST when Pause landed before handoff"
    assert _record(tmp_path, blocked["pending_invocation_id"])["state"] == "failed_definite"

    # 3. Pause landing before a retry POST leaves the invocation pending (never definitely_unrun)
    monkeypatch.setattr(owner_pause, "run_operation", real_run_op)
    fail_once = {"on": True}
    receiver.hooks.append(lambda req: (fail_once.update(on=False) or httpx.Response(503, json={"error": "unconfirmed"}))
                          if fail_once["on"] and req.url.path == "/api/letters" and req.method == "POST" else None)
    token = _call(ctx, place_id=PLACE_ID, text="retry paused")["pending_invocation_id"]
    monkeypatch.setattr(owner_pause, "run_operation", pause_at_handoff)
    posts_before = len(receiver.posts)
    retry_paused = _call(ctx, place_id=PLACE_ID, retry_of=token)
    assert retry_paused["pending_invocation_id"] == token and "definitely_unrun" not in retry_paused
    assert len(receiver.posts) == posts_before
    assert _record(tmp_path, token)["state"] == "pending"


def test_uncustodied_start_reports_started_uncustodied_when_bind_started_not_durable(ctx, tmp_path, receiver, monkeypatch):
    from ouroboros.delegate_accepted import Bound

    monkeypatch.setattr(custody, "bind_started", lambda drive, entry, **kw: Bound(entry, False, False))
    out = _call(ctx, place_id=PLACE_ID, text="durability loss")
    assert out["status"] == "started_uncustodied" and out["custody_durable"] is False, out
    assert out["pending_invocation_id"] == out["run_id"]


def test_retry_of_enforces_task_ownership_and_uses_recorded_lineage(ctx, tmp_path, receiver):
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolContext

    write_task_result(tmp_path, "root-orig", "running", root_task_id="root-orig", task_attempt=1)
    write_task_result(tmp_path, "t-owner", "running", root_task_id="root-orig", parent_task_id="parent-orig",
                      task_attempt=1)
    ctx.task_id = "t-owner"
    ctx.task_metadata = {"root_task_id": "root-orig", "parent_task_id": "parent-orig"}
    fail_once = {"on": True}
    receiver.hooks.append(lambda req: (fail_once.update(on=False) or httpx.Response(503, json={"error": "unconfirmed"}))
                          if fail_once["on"] and req.url.path == "/api/letters" and req.method == "POST" else None)
    first = _call(ctx, place_id=PLACE_ID, text="owned letter")
    assert first["status"] == "unconfirmed", first
    token = first["pending_invocation_id"]

    foreign = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    foreign.task_id = "t-foreign"
    foreign.task_metadata = {"root_task_id": "root-foreign", "parent_task_id": "parent-foreign"}
    stolen = _call(foreign, place_id=PLACE_ID, retry_of=token)
    assert stolen["reason"] == "invocation_not_owned", stolen
    assert len(receiver.posts) == 1 and _record(tmp_path, token)["state"] == "pending"

    mismatch = _call(ctx, place_id=PLACE_ID, retry_of=token, text="different text")
    assert mismatch["reason"] == "retry_text_mismatch" and len(receiver.posts) == 1

    # Even if caller's metadata mutated, recovery uses recorded task_id/root_task_id/parent_task_id
    ctx.task_metadata = {"root_task_id": "mutated-root", "parent_task_id": "mutated-parent"}
    retried = _call(ctx, place_id=PLACE_ID, retry_of=token)
    assert retried["status"] == "started", retried
    bound = current.run(tmp_path, token)
    assert (bound.task_id, bound.root_task_id, bound.parent_task_id) == ("t-owner", "root-orig", "parent-orig")


def test_answer_interaction_validates_single_matching_question_row_and_unknown_delivery(ctx, receiver):
    from ouroboros.delegate_interactions import _delegate_answer
    from ouroboros.delegate_shared import AGENT_FAULT_CODE

    run_id = _call(ctx, place_id=PLACE_ID, text="ask carefully")["run_id"]
    receiver.letters[run_id].update(state="needs_input", question={"question_id": "q1", "text": "pick one?"})
    posts_before = len(receiver.posts)

    wrong_qid = json.loads(_delegate_answer(ctx, run_id, "q1", [{"question_id": "q-wrong", "free_text": "nope"}]).text)
    assert (wrong_qid["status"], wrong_qid["ok"], wrong_qid["host_code"]) == ("rejected", False, AGENT_FAULT_CODE)
    assert len(receiver.posts) == posts_before, "mismatched question_id must never POST"

    multi_row = json.loads(_delegate_answer(ctx, run_id, "q1", [
        {"question_id": "q1", "free_text": "first"}, {"question_id": "q2", "free_text": "second"},
    ]).text)
    assert (multi_row["status"], multi_row["ok"], multi_row["host_code"]) == ("rejected", False, AGENT_FAULT_CODE)
    assert len(receiver.posts) == posts_before, "multiple answer rows must never POST"

    # Unknown delivery (503) returns delivery_unknown without replaying different text
    fail_once = {"on": True}
    receiver.hooks.append(lambda req: (fail_once.update(on=False) or httpx.Response(503, json={"error": "unconfirmed"}))
                          if fail_once["on"] and req.url.path == "/api/letters" and b"in_reply_to" in req.content else None)
    unknown = json.loads(_delegate_answer(ctx, run_id, "q1", [{"question_id": "q1", "free_text": "yes"}]).text)
    assert unknown["status"] == "delivery_unknown", unknown
    # Same answer retried sends the exact same deterministic reply-* letter id
    ok = json.loads(_delegate_answer(ctx, run_id, "q1", [{"question_id": "q1", "free_text": "yes"}]).text)
    assert ok["status"] == "delivered", ok
    first_post = json.loads(receiver.posts[-2])
    second_post = json.loads(receiver.posts[-1])
    assert first_post == second_post and first_post["letter_id"].startswith("reply-")


def test_live_messages_and_owner_placement_are_unsupported_without_retry_loop_or_leak(ctx, tmp_path, receiver):
    """Direct delegate_message, owner place_at_receipt / place_rows_into_runs, and Feed.place replay all terminate
    immediately as unsupported with zero network calls to the peer and no infinite retry loop."""
    from ouroboros import delegate_message_store as msg_store
    from ouroboros.delegate_interactions import _delegate_message, place_at_receipt

    run_id = _call(ctx, place_id=PLACE_ID, text="long task")["run_id"]
    calls_before = len(receiver.calls)

    direct = json.loads(_delegate_message(ctx, run_id, "secret owner steer").text)
    assert direct["status"] == "unsupported" and direct["reason"] == "engine_lacks_operation", direct
    assert len(receiver.calls) == calls_before, "delegate_message must not touch the peer server"

    entry = current.run(tmp_path, run_id)
    quote = {
        "owner_row": {"text": "private owner follow-up", "ts": "2026-11-24T00:00:00Z"},
        "source": {"chat_id": "c-1", "ts": "2026-11-24T00:00:00Z"},
        "ts": "2026-11-24T00:00:00Z",
        "sender": "owner",
    }
    thread = place_at_receipt(tmp_path, "t-a", [
        ({"msg:m-owner-1"}, "msg:m-owner-1", quote, "t-a", "", False),
    ])
    assert thread is not None
    thread.join(timeout=5)
    placed = msg_store.placed_words(ctx, run_id)
    assert len(placed) == 1 and placed[0]["outcome"] == "unsupported", placed
    assert placed[0]["reason"] == "linked_ouroboros_no_live_input", "no engine is reached for my human's row"
    assert msg_store.pending_frames(ctx, run_id) == [], "unsupported placement must not stay queued for replay"
    assert len(receiver.calls) == calls_before, "owner message must never leak to the linked Ouroboros"

    # Even if a legacy prepared frame existed on disk, replay_frames terminates it on the first pass via PeerGateway
    row, _prior = msg_store.place_frame(ctx, entry, "legacy row", {"address": "msg:m-legacy-1"})
    assert row is not None and len(msg_store.pending_frames(ctx, run_id)) == 1
    assert msg_store.replay_frames(tmp_path, entry) is True
    assert msg_store.pending_frames(ctx, run_id) == [], "legacy frame settles into PLACEMENT_ENDED on first pass"
    legacy_rows = [r for r in msg_store.placed_words(ctx, run_id)
                   if r["frame"]["owner_words"]["address"] == "msg:m-legacy-1"]
    assert len(legacy_rows) == 1 and legacy_rows[0]["outcome"] == "unsupported", legacy_rows
    assert len(receiver.calls) == calls_before


def test_revoked_link_stops_body_polling_without_equating_observation_loss_to_remote_death(ctx, tmp_path, receiver, monkeypatch):
    """When the receiver revokes the link (401 link_unknown), BodyRuntime stops polling, records peer_observation_lost
    (leaving custody OPEN, never CLOSED_ABSENT/failed/cancelled), wakes delegate_wait, and reports unknown remote state."""
    import time

    from ouroboros import delegate_collection as collection
    from ouroboros import delegate_supervision
    from ouroboros.tools import delegate as delegate_tools

    marked = []
    monkeypatch.setattr(peer_links, "mark_remote_unknown", lambda drive, lid: marked.append((str(drive), lid)))
    run_id = _call(ctx, place_id=PLACE_ID, text="watch then revoke")["run_id"]

    # Receiver revokes the link
    receiver.hooks.append(lambda req: httpx.Response(401, json={"error": "link_unknown", "refusal_code": "link_unknown"})
                          if req.url.path == f"/api/letters/{run_id}" else None)
    body = collection.BodyRuntime(tmp_path, gateway_factory=lambda: pytest.fail("owned engine built"))
    try:
        deadline = time.monotonic() + 5.0
        while run_id not in body.observation_gaps and time.monotonic() < deadline:
            time.sleep(0.05)
        assert run_id in body.observation_gaps, "body must record observation gap on 401 link_unknown"
        calls_after_first = len(receiver.calls)
        assert (str(tmp_path), LINK_ID) in marked, "401 link_unknown marks remote_unknown_at"
        reloaded = current.run(tmp_path, run_id)
        assert reloaded is not None and reloaded.settled is False and reloaded.terminal_state == ""
        assert reloaded.containment_fault.get("reason") == "peer_observation_lost"

        # Subsequent body ticks must NOT poll the revoked link again
        body.tick()
        body.tick()
        assert len(receiver.calls) == calls_after_first, "body must stop polling once link_unknown is observed"
    finally:
        body.close(wait=True)

    # A sleeping delegate_wait wakes on observation loss and reports unknown remote state (not death)
    assert delegate_supervision.run_wait_reason(ctx, [run_id]) == f"run_observation_lost:{run_id}"
    read = json.loads(delegate_tools._read_run(ctx, run_id, 0))
    assert read["reason"] == "peer_observation_lost" and read["remote_work_state"] == "unknown_may_still_be_running", read

    # Local revocation on delegate_cancel also refuses honestly with run_may_still_be_live=True
    monkeypatch.setattr(peer_links, "outgoing_link", lambda place_id, drive_root=None: None)
    cancelled = json.loads(delegate_tools._delegate_cancel(ctx, run_id, reason="try stop").text)
    assert cancelled["ok"] is False and cancelled["run_may_still_be_live"] is True, cancelled
    assert cancelled["remote_work_state"] == "unknown_may_still_be_running"
    assert current.run(tmp_path, run_id).settled is False


def test_done_result_available_wire_and_retained_payload(ctx, tmp_path, receiver):
    """Wire ``/result.result_available: bool`` rides unchanged in ``final_summary.result_available`` across both the
    live terminal payload and the full retained record; ``true`` accepts empty text, ``false`` leaves
    ``primary_output`` None with no placeholder, and legacy/incomplete wire without the flag keeps it absent."""
    from ouroboros import delegate_collection as collection
    from ouroboros.tools import delegate as delegate_tools
    from ouroboros.tools.delegate_terminal_evidence import retained_terminal_result

    # 1. result_available=True with non-empty text
    rid_ok = _call(ctx, place_id=PLACE_ID, text="addressed non-empty")["run_id"]
    receiver.letters[rid_ok].update(
        state="done", result_available=True, result="verified answer",
        finished_at="2026-11-24T10:00:00Z", cost={"known_usd": 0.42, "final": True},
    )
    out_ok = json.loads(delegate_tools._read_run(ctx, rid_ok, 0))
    assert (out_ok["status"], out_ok["state"]) == ("terminal", "succeeded"), out_ok
    assert out_ok["primary_output"] == {"text": "verified answer", "bytes": 15, "truncated": False}
    assert out_ok["final_summary"]["result_available"] is True
    assert out_ok["cost"]["cost_usd"] == pytest.approx(0.42) and out_ok["cost"]["cost_final"] is True
    retained_ok = collection.retained_payload(tmp_path, current.run(tmp_path, rid_ok))
    assert retained_ok["final_summary"]["result_available"] is True
    assert retained_ok["primary_output"]["text"] == "verified answer"
    reread_ok = retained_terminal_result(ctx, rid_ok)
    assert reread_ok["final_summary"]["result_available"] is True
    assert reread_ok["primary_output"]["text"] == "verified answer"

    # 2. result_available=True with empty string "" (real addressed empty text)
    rid_empty = _call(ctx, place_id=PLACE_ID, text="addressed empty")["run_id"]
    receiver.letters[rid_empty].update(
        state="done", result_available=True, result="",
        finished_at="2026-11-24T10:01:00Z", cost={"known_usd": 0.0, "final": False},
    )
    out_empty = json.loads(delegate_tools._read_run(ctx, rid_empty, 0))
    assert out_empty["primary_output"] == {"text": "", "bytes": 0, "truncated": False}
    assert out_empty["final_summary"]["result_available"] is True
    assert out_empty["cost"]["cost_usd"] == 0.0 and out_empty["cost"]["cost_final"] is False
    retained_empty = collection.retained_payload(tmp_path, current.run(tmp_path, rid_empty))
    assert retained_empty["final_summary"]["result_available"] is True
    assert retained_empty["primary_output"] == {"text": "", "bytes": 0, "truncated": False}

    # 3. result_available=False (no addressed answer — even if stray text rides on wire, primary_output is None)
    rid_none = _call(ctx, place_id=PLACE_ID, text="no addressed answer")["run_id"]
    receiver.letters[rid_none].update(
        state="done", result_available=False, result="internal non-addressed text",
        finished_at="2026-11-24T10:02:00Z", cost={"known_usd": 0.12, "final": True},
    )
    out_none = json.loads(delegate_tools._read_run(ctx, rid_none, 0))
    assert (out_none["status"], out_none["state"]) == ("terminal", "succeeded"), out_none
    assert out_none["primary_output"] is None, "result_available=False must leave primary_output None"
    assert out_none["final_summary"]["result_available"] is False
    assert out_none["cost"]["cost_usd"] == pytest.approx(0.12) and out_none["cost"]["cost_final"] is True
    retained_none = collection.retained_payload(tmp_path, current.run(tmp_path, rid_none))
    assert retained_none["final_summary"]["result_available"] is False
    assert retained_none["primary_output"] is None
    reread_none = retained_terminal_result(ctx, rid_none)
    assert reread_none["final_summary"]["result_available"] is False and reread_none["primary_output"] is None

    # 4. Legacy wire (no result_available) with synthetic non-empty text: primary_output stays None, flag absent,
    #    while succeeded state and receiver cost remain intact across live, retained and re-read payloads.
    rid_legacy = _call(ctx, place_id=PLACE_ID, text="legacy non-empty")["run_id"]
    receiver.letters[rid_legacy].update(
        state="done", omit_result_available=True, result="legacy answer", finished_at="2026-11-24T10:03:00Z",
        cost={"known_usd": 0.25, "final": True},
    )
    out_legacy = json.loads(delegate_tools._read_run(ctx, rid_legacy, 0))
    assert (out_legacy["status"], out_legacy["state"]) == ("terminal", "succeeded"), out_legacy
    assert out_legacy["primary_output"] is None, "legacy wire without result_available must leave primary_output None"
    assert "result_available" not in (out_legacy["final_summary"] or {}), out_legacy
    assert out_legacy["cost"]["cost_usd"] == pytest.approx(0.25) and out_legacy["cost"]["cost_final"] is True
    retained_legacy = collection.retained_payload(tmp_path, current.run(tmp_path, rid_legacy))
    assert retained_legacy["primary_output"] is None and "result_available" not in (retained_legacy["final_summary"] or {})
    reread_legacy = retained_terminal_result(ctx, rid_legacy)
    assert reread_legacy["primary_output"] is None and "result_available" not in (reread_legacy["final_summary"] or {})

    # 5. Legacy wire without result_available and empty result: availability unknown AND primary_output is None
    rid_legacy_empty = _call(ctx, place_id=PLACE_ID, text="legacy empty")["run_id"]
    receiver.letters[rid_legacy_empty].update(
        state="done", omit_result_available=True, result="", finished_at="2026-11-24T10:04:00Z",
    )
    out_legacy_empty = json.loads(delegate_tools._read_run(ctx, rid_legacy_empty, 0))
    assert (out_legacy_empty["status"], out_legacy_empty["state"]) == ("terminal", "succeeded"), out_legacy_empty
    assert "result_available" not in (out_legacy_empty["final_summary"] or {})
    assert out_legacy_empty["primary_output"] is None

    # 6. Malformed/invalid wire: result_available=True without string result, or non-bool result_available with text ->
    #    succeeded state and cost remain, result_available key stays absent, primary_output is None.
    for idx, extra in enumerate([
        {"result_available": True, "result": None, "cost": {"known_usd": 0.10, "final": False}},
        {"result_available": "true", "result": "non-bool text", "cost": {"known_usd": 0.30, "final": True}},
        {"result_available": 1, "result": "int text", "cost": {"known_usd": 0.05, "final": True}},
    ]):
        rid_bad = _call(ctx, place_id=PLACE_ID, text=f"malformed {idx}")["run_id"]
        receiver.letters[rid_bad].update(state="done", finished_at="2026-11-24T10:05:00Z", **extra)
        out_bad = json.loads(delegate_tools._read_run(ctx, rid_bad, 0))
        assert (out_bad["status"], out_bad["state"]) == ("terminal", "succeeded"), out_bad
        assert "result_available" not in (out_bad["final_summary"] or {}), out_bad
        assert out_bad["primary_output"] is None, out_bad
        assert out_bad["cost"]["cost_usd"] == pytest.approx(extra["cost"]["known_usd"]), out_bad
        assert out_bad["cost"]["cost_final"] is extra["cost"]["final"], out_bad


def test_local_silent_blocks_new_post_and_retry_post_while_preserving_reads_and_lookup(ctx, tmp_path, receiver, monkeypatch):
    """Local Silent/Paused blocks new POST and retry_of POST (and answer POST) before any network call, while
    keeping safe GET reads and non-creating lookup recovery working; a refused retry keeps its prior start unknown."""
    from ouroboros import places
    from ouroboros.delegate_interactions import _delegate_answer
    from ouroboros.tools import delegate as delegate_tools

    place = places.connect_place(
        tmp_path, connector="ouroboros", provider="ouroboros", label="Velm",
        account_id="install-b", conversation_id=LINK_ID, scope="conversation",
        state=places.STATE_ACTIVE, author=places.AUTHOR_OWNER,
    )
    places.edit_place(tmp_path, place.place_id, "set_peer", author=places.AUTHOR_OWNER, mine=True, may_start=True)
    live_link = peer_links.OutgoingLink(
        link_id=LINK_ID, place_id=place.place_id, peer_url="http://127.0.0.1:47231",
        peer_label="Velm", peer_install_id="install-b", own_install_id="install-a",
        drive_root=str(tmp_path),
    )
    monkeypatch.setattr(peer_links, "outgoing_link",
                        lambda pid, drive_root=None: live_link if pid == place.place_id else None)

    # Start one live letter, one unconfirmed-but-accepted letter, and one unconfirmed-not-landed letter while Active
    live_id = _call(ctx, place_id=place.place_id, text="already running")["run_id"]
    receiver.letters[live_id].update(state="needs_input", question={"question_id": "q-s1", "text": "proceed?"})

    drop_accepted = {"on": True}

    def accept_then_drop(req):
        if req.url.path == "/api/letters" and req.method == "POST" and drop_accepted["on"]:
            drop_accepted["on"] = False
            body = json.loads(req.content)
            receiver.letters[body["letter_id"]] = {"body": body, "state": "accepted_running", "phase": "running"}
            raise httpx.ReadTimeout("lost reply after accept", request=req)
        return None

    receiver.hooks.append(accept_then_drop)
    accepted_token = _call(ctx, place_id=place.place_id, text="accepted before silent")["pending_invocation_id"]
    assert _record(tmp_path, accepted_token)["state"] == "pending"

    drop_unseen = {"on": True}
    receiver.hooks.append(lambda req: (drop_unseen.update(on=False) or httpx.Response(503, json={"error": "unconfirmed"}))
                          if drop_unseen["on"] and req.url.path == "/api/letters" and req.method == "POST" else None)
    unseen_token = _call(ctx, place_id=place.place_id, text="unseen before silent")["pending_invocation_id"]
    assert _record(tmp_path, unseen_token)["state"] == "pending"

    # Flip local place to Silent
    places.edit_place(tmp_path, place.place_id, "set_state", author=places.AUTHOR_OWNER, state=places.STATE_SILENT)
    posts_before = len(receiver.posts)

    # 1. Fresh POST is refused before network (definitely_unrun=True)
    blocked_new = _call(ctx, place_id=place.place_id, text="blocked while silent")
    assert (blocked_new["status"], blocked_new["reason"], blocked_new["definitely_unrun"]) == (
        "refused", "place_silent", True), blocked_new
    assert len(receiver.posts) == posts_before

    # 2. Safe read (delegate_wait) still works while Silent
    waited = json.loads(delegate_tools._read_run(ctx, live_id, 0))
    assert waited["status"] == "waiting_on_user", waited

    # 3. Answer POST is blocked while Silent before network (not delivery_unknown)
    blocked_ans = json.loads(_delegate_answer(ctx, live_id, "q-s1", [{"question_id": "q-s1", "free_text": "go"}]).text)
    assert (blocked_ans["status"], blocked_ans["reason"]) == ("refused", "place_silent"), blocked_ans
    assert len(receiver.posts) == posts_before

    # 4. retry_of whose letter WAS accepted recovers via non-creating GET lookup even while Silent
    recovered = _call(ctx, place_id=place.place_id, retry_of=accepted_token)
    assert recovered["status"] == "started" and recovered.get("recovered_via_lookup") is True, recovered
    assert len(receiver.posts) == posts_before

    # 5. retry_of whose letter was NOT found on receiver is blocked before POST and stays pending (never definitely_unrun)
    blocked_retry = _call(ctx, place_id=place.place_id, retry_of=unseen_token)
    assert blocked_retry["reason"] == "place_silent" and blocked_retry["pending_invocation_id"] == unseen_token
    assert "definitely_unrun" not in blocked_retry, blocked_retry
    assert _record(tmp_path, unseen_token)["state"] == "pending"
    assert len(receiver.posts) == posts_before

    # 6. Once local place returns to Active, retry_of sends the exact same bytes and starts
    places.edit_place(tmp_path, place.place_id, "set_state", author=places.AUTHOR_OWNER, state=places.STATE_ACTIVE)
    resumed = _call(ctx, place_id=place.place_id, retry_of=unseen_token)
    assert resumed["status"] == "started" and resumed["run_id"] == unseen_token, resumed
    assert len(receiver.posts) == posts_before + 1

    # 7. Paused local place also blocks before POST
    places.edit_place(tmp_path, place.place_id, "set_state", author=places.AUTHOR_OWNER, state=places.STATE_PAUSED)
    blocked_paused = _call(ctx, place_id=place.place_id, text="blocked while paused")
    assert (blocked_paused["status"], blocked_paused["reason"], blocked_paused["definitely_unrun"]) == (
        "refused", "place_paused", True), blocked_paused
    assert len(receiver.posts) == posts_before + 1


def test_retry_of_checks_title_and_unnamed_4xx_is_never_definite_refusal(ctx, tmp_path, receiver):
    """retry_of enforces exact recorded title (and valid ceiling), tool description names lookup-first/exact send,
    and an unnamed 4xx (408/429/plain 400 without refusal_code) leaves acceptance unknown rather than definite refusal."""
    schema = peer_delegate.get_tools()[0].schema
    assert "lookup-first" in schema["description"] and "exact recorded letter" in schema["description"]
    assert "lookup-first" in schema["parameters"]["properties"]["retry_of"]["description"]

    # Unnamed 4xx responses (408 timeout, 429 rate limit, 400 without refusal_code) must NOT be definite refusals
    for status_code, body in [(408, None), (429, {"error": "rate_limited"}), (400, {"error": "proxy_bad_request"})]:
        armed = {"on": True}

        def unnamed_once(req, _sc=status_code, _body=body, _armed=armed):
            if _armed["on"] and req.url.path == "/api/letters" and req.method == "POST":
                _armed["on"] = False
                return httpx.Response(_sc, json=_body) if _body is not None else httpx.Response(_sc, text="timeout")
            return None

        receiver.hooks.append(unnamed_once)
        out = _call(ctx, place_id=PLACE_ID, text=f"unnamed {status_code}", title="Orig Title")
        assert out["status"] == "unconfirmed", (status_code, out)
        assert "definitely_unrun" not in out
        token = out["pending_invocation_id"]
        assert _record(tmp_path, token)["state"] == "pending"

        # Mismatched title on retry_of is refused before network
        posts_before = len(receiver.posts)
        bad_title = _call(ctx, place_id=PLACE_ID, retry_of=token, title="Different Title")
        assert bad_title["reason"] == "retry_title_mismatch" and len(receiver.posts) == posts_before
        bad_ceil = _call(ctx, place_id=PLACE_ID, retry_of=token, cost_ceiling_usd=True)
        assert bad_ceil["reason"] == "retry_ceiling_mismatch" and len(receiver.posts) == posts_before

        # Exact recorded title (or omitted) succeeds with identical bytes
        ok = _call(ctx, place_id=PLACE_ID, retry_of=token, title="Orig Title")
        assert ok["status"] == "started" and ok["run_id"] == token, ok
        assert receiver.posts[-1] == receiver.posts[-2]

    # Untitled recorded letter also rejects a newly supplied title on retry_of
    armed_u = {"on": True}
    receiver.hooks.append(lambda req: (armed_u.update(on=False) or httpx.Response(503, json={"error": "unconfirmed"}))
                          if armed_u["on"] and req.url.path == "/api/letters" and req.method == "POST" else None)
    untitled_token = _call(ctx, place_id=PLACE_ID, text="no title originally")["pending_invocation_id"]
    added_title = _call(ctx, place_id=PLACE_ID, retry_of=untitled_token, title="Added Title")
    assert added_title["reason"] == "retry_title_mismatch"


def test_answer_interaction_requires_exact_receipt_and_unnamed_4xx_stays_unknown(ctx, receiver):
    """answer_interaction never declares delivered on a missing/foreign/stateless 200 body, and treats an unnamed
    400/409 (no refusal_code) as delivery_unknown rather than rejected."""
    from ouroboros.delegate_interactions import _delegate_answer

    run_id = _call(ctx, place_id=PLACE_ID, text="ask for exact receipt")["run_id"]
    receiver.letters[run_id].update(state="needs_input", question={"question_id": "q-rcpt", "text": "confirm?"})

    bad_200s = [
        lambda body: httpx.Response(200, json={}),
        lambda body: httpx.Response(200, json={"letter_id": "reply-someone-else", "state": "accepted_running"}),
        lambda body: httpx.Response(200, json={"letter_id": body["letter_id"]}),
        lambda body: httpx.Response(200, json={"letter_id": body["letter_id"], "state": "unknown_state"}),
        # GET's form (a LetterStatus) is no answer receipt, nor is one without the mailbox row it landed as.
        lambda body: httpx.Response(200, json={"letter_id": body["letter_id"], "state": "accepted_running",
                                               "delivered": True, "in_reply_to": body["in_reply_to"]}),
        lambda body: httpx.Response(200, json={"letter_id": body["letter_id"], "in_reply_to": body["in_reply_to"],
                                               "delivered": False, "msg_id": "m-1"}),
    ]
    for make_resp in bad_200s:
        armed = {"on": True}
        receiver.hooks.append(
            lambda req, _mk=make_resp, _a=armed: (_a.update(on=False) or _mk(json.loads(req.content)))
            if _a["on"] and req.url.path == "/api/letters" and b"in_reply_to" in req.content else None
        )
        out = json.loads(_delegate_answer(ctx, run_id, "q-rcpt", [{"question_id": "q-rcpt", "free_text": "yes"}]).text)
        assert out["status"] == "delivery_unknown", out

    # Unnamed 400 and 409 without refusal_code -> delivery_unknown (not rejected, not delivered)
    for sc in (400, 409):
        armed = {"on": True}
        receiver.hooks.append(lambda req, _sc=sc, _a=armed: (_a.update(on=False) or httpx.Response(_sc, json={"error": "opaque"}))
                              if _a["on"] and req.url.path == "/api/letters" and b"in_reply_to" in req.content else None)
        out = json.loads(_delegate_answer(ctx, run_id, "q-rcpt", [{"question_id": "q-rcpt", "free_text": "yes"}]).text)
        assert out["status"] == "delivery_unknown", (sc, out)

    # LetterAnswerReceipt (this reply's id, its exact in_reply_to, delivered, the mailbox row's msg_id) -> delivered
    ok = json.loads(_delegate_answer(ctx, run_id, "q-rcpt", [{"question_id": "q-rcpt", "free_text": "yes"}]).text)
    assert ok["status"] == "delivered" and ok["accepted"] is True, ok


def test_peer_answer_durable_sha256_pin_before_post_and_recovery(ctx, tmp_path, receiver, monkeypatch):
    """Peer delegate_answer pins SHA-256 of serialized letter_bytes in a dedicated obligations set before POST:
    locally refused attempts leave the slot unpinned; store unavailability sends nothing; after a lost receipt and
    a fresh gateway/reloaded custody, changed text is refused before POST while an exact repeat sends identical bytes
    without leaking text or rows into live-message journals; ordinary Claudexor sessions stay untouched."""
    import hashlib

    from ouroboros import delegate_message_store as msg_store
    from ouroboros import places, store
    from ouroboros.delegate_interactions import _delegate_answer
    from ouroboros.delegate_shared import AGENT_FAULT_CODE, SUBSTRATE_REFUSAL_CODE

    place = places.connect_place(
        tmp_path, connector="ouroboros", provider="ouroboros", label="Velm",
        account_id="install-b", conversation_id=LINK_ID, scope="conversation",
        state=places.STATE_ACTIVE, author=places.AUTHOR_OWNER,
    )
    places.edit_place(tmp_path, place.place_id, "set_peer", author=places.AUTHOR_OWNER, mine=True, may_start=True)
    live_link = peer_links.OutgoingLink(
        link_id=LINK_ID, place_id=place.place_id, peer_url="http://127.0.0.1:47231",
        peer_label="Velm", peer_install_id="install-b", own_install_id="install-a",
        drive_root=str(tmp_path),
    )
    monkeypatch.setattr(peer_links, "outgoing_link",
                        lambda pid, drive_root=None: live_link if pid == place.place_id else None)

    run_id = _call(ctx, place_id=place.place_id, text="ask pinned question")["run_id"]
    receiver.letters[run_id].update(state="needs_input", question={"question_id": "q-pin", "text": "which branch?"})
    posts_before = len(receiver.posts)

    # 1. Local pre-send refusals (Silent, invalid row, missing link secret) must NOT pin the slot
    places.edit_place(tmp_path, place.place_id, "set_state", author=places.AUTHOR_OWNER, state=places.STATE_SILENT)
    sil = json.loads(_delegate_answer(ctx, run_id, "q-pin", [{"question_id": "q-pin", "free_text": "draft-1"}]).text)
    assert (sil["status"], sil["reason"]) == ("refused", "place_silent"), sil
    places.edit_place(tmp_path, place.place_id, "set_state", author=places.AUTHOR_OWNER, state=places.STATE_ACTIVE)

    bad_row = json.loads(_delegate_answer(ctx, run_id, "q-pin", [{"question_id": "q-other", "free_text": "wrong"}]).text)
    assert (bad_row["status"], bad_row["host_code"]) == ("rejected", AGENT_FAULT_CODE), bad_row

    tok_file = peer_links.token_path(tmp_path, LINK_ID)
    saved_tok = tok_file.read_bytes()
    tok_file.unlink()
    no_key = json.loads(_delegate_answer(ctx, run_id, "q-pin", [{"question_id": "q-pin", "free_text": "draft-2"}]).text)
    assert (no_key["status"], no_key["reason"]) == ("refused", "peer_link_unavailable"), no_key
    tok_file.write_bytes(saved_tok)
    assert len(receiver.posts) == posts_before
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.set_rows(msg_store._peer_answer_pin_set(run_id)) == {}

    # 2. Store unavailable -> typed refusal before POST, zero network calls
    real_open = store.open_store
    monkeypatch.setattr(
        store, "open_store",
        lambda root, *, write=False, **k: (_ for _ in ()).throw(store.StoreBusy("locked"))
        if write else real_open(root, write=False, **k),
    )
    unavail = json.loads(_delegate_answer(ctx, run_id, "q-pin", [{"question_id": "q-pin", "free_text": "branch-alpha"}]).text)
    assert (unavail["status"], unavail["reason"], unavail["ok"], unavail["host_code"]) == (
        "refused", "peer_answer_pin_unavailable", False, SUBSTRATE_REFUSAL_CODE), unavail
    assert len(receiver.posts) == posts_before
    monkeypatch.setattr(store, "open_store", real_open)

    # 3. First real send loses its receipt (ReadTimeout) -> delivery_unknown, no auto-resend, digest pinned durably
    drop_once = {"on": True}

    def drop_reply(req):
        if drop_once["on"] and req.url.path == "/api/letters" and b"in_reply_to" in req.content:
            drop_once["on"] = False
            raise httpx.ReadTimeout("lost reply receipt", request=req)
        return None

    receiver.hooks.append(drop_reply)
    unknown = json.loads(_delegate_answer(ctx, run_id, "q-pin", [{"question_id": "q-pin", "free_text": "branch-alpha"}]).text)
    assert unknown and unknown["status"] == "delivery_unknown", unknown
    assert len(receiver.posts) == posts_before + 1, "no automatic resend after unknown"
    first_post_bytes = receiver.posts[-1]

    with store.open_store(tmp_path, write=False) as txn:
        pins = txn.set_rows(msg_store._peer_answer_pin_set(run_id))
    assert len(pins) == 1
    pin_row = next(iter(pins.values()))
    assert pin_row["kind"] == "peer_answer_pin"
    assert pin_row["letter_bytes_sha256"] == hashlib.sha256(first_post_bytes).hexdigest()
    assert "branch-alpha" not in json.dumps(pin_row), "pin must store only the digest, never a copy of the text"
    assert msg_store.placed_words(ctx, run_id) == []
    assert msg_store.pending_frames(ctx, run_id) == []
    assert msg_store.journal(ctx, run_id) == []

    # 4. Simulate fresh process/gateway reading the same durable store: different text refused before POST
    custody._CUSTODY.clear()
    conflict = json.loads(_delegate_answer(ctx, run_id, "q-pin", [{"question_id": "q-pin", "free_text": "branch-beta"}]).text)
    assert (conflict["status"], conflict["reason"], conflict["ok"], conflict["host_code"]) == (
        "refused", "peer_answer_conflict", False, AGENT_FAULT_CODE), conflict
    assert "forbidden" in conflict["detail"]
    assert len(receiver.posts) == posts_before + 1, "changed text must never be POSTed"

    # 5. Exact same answer retries with identical serialized bytes and succeeds
    ok = json.loads(_delegate_answer(ctx, run_id, "q-pin", [{"question_id": "q-pin", "free_text": "branch-alpha"}]).text)
    assert ok["status"] == "delivered" and ok["accepted"] is True, ok
    assert len(receiver.posts) == posts_before + 2 and receiver.posts[-1] == first_post_bytes

    # 6. Ordinary Claudexor session keeps the exact legacy answer_interaction call without before_send or pin rows
    cx_entry = custody.RunCustody(
        run_id="cx-ord-1", task_id="t-a", invocation_id="cx-ord-1",
        route_id="codex", category="subagent", source="delegated_subagent",
        root_task_id="t-a", parent_task_id="t-a", ledger_root=str(tmp_path), access="readonly",
    )
    assert custody.record_started(tmp_path, cx_entry, {"delegated": True})
    seen_kw = {}

    class _FakeCxGateway:
        def handshake(self, **_kw):
            return {}

        def answer_interaction(self, rid, iid, rows, *, timeout_sec=None):
            seen_kw["args"] = (rid, iid, rows)
            return {"status": "delivered", "accepted": True, "message": ""}

        def close(self):
            pass

    monkeypatch.setattr(wire, "ClaudexorGateway", lambda *a, **k: _FakeCxGateway())
    cx_out = json.loads(_delegate_answer(ctx, "cx-ord-1", "q-cx", [{"question_id": "q-cx", "free_text": "ok"}]).text)
    assert cx_out["status"] == "delivered" and seen_kw["args"][:2] == ("cx-ord-1", "q-cx"), cx_out
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.set_rows(msg_store._peer_answer_pin_set("cx-ord-1")) == {}


def test_body_runtime_observes_real_receiver_through_real_containment_without_cancel(tmp_path, monkeypatch):
    """Through real BodyRuntime._observe/tick and unmocked delegate_containment._reads_its_own_letter,
    a real peer letter served by the in-process receiver (TestClient, synthetic data) is observed in
    running and needs_input states and reaches terminal without a single cancel or fabricated access."""
    import threading
    import time
    from starlette.testclient import TestClient

    from ouroboros import delegate_collection as collection
    from ouroboros import install_identity
    from ouroboros.install_identity import acting_as
    from ouroboros.peer_letters import ask
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.delegate_terminal_evidence import retained_terminal_result
    from ouroboros.tools.registry import ToolContext
    from tests._peer_letters_shared import introduce, published, receiver_client, safety_reads

    pair = introduce(tmp_path, monkeypatch)
    safety_reads(monkeypatch)  # the letter's one reading (check_peer_assignment) answered: not this test's subject
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(pair.root_a))
    monkeypatch.setattr(install_identity, "_CANONICAL_ROOTS", {})
    monkeypatch.setattr(wire, "ClaudexorGateway", lambda *a, **k: pytest.fail("owned engine built"))
    real_outgoing = peer_links.outgoing_link
    monkeypatch.setattr(peer_links, "outgoing_link",
                        lambda place_id, drive_root=None: real_outgoing(place_id, drive_root=pair.root_a))

    requests_seen: list[tuple[str, str]] = []
    req_lock = threading.Lock()
    with receiver_client(pair, monkeypatch) as client:
        real_request = client.request

        def recording_request(method, url, **kwargs):
            with req_lock:
                requests_seen.append((str(method), str(url)))
                return real_request(method, url, **kwargs)

        monkeypatch.setattr(client, "request", recording_request)
        monkeypatch.setattr(peerwire.httpx, "Client", lambda **kw: client if isinstance(client, TestClient) else None)
        monkeypatch.setattr(client, "close", lambda: None)

        sender = ToolContext(repo_dir=pair.repo, drive_root=pair.root_a)
        sender.task_id, sender.task_metadata = "t-a", {"root_task_id": "t-a"}
        with acting_as(pair.root_a):
            write_task_result(pair.root_a, "t-a", "running", root_task_id="t-a", task_attempt=1)
            started = _call(sender, place_id=pair.place_a, text="Count synthetic files.", cost_ceiling_usd=0.5)
        assert started["status"] == "started", started
        run_id = started["run_id"]
        b_task_id = pair.task_id(run_id)
        entry = current.run(pair.root_a, run_id)
        assert entry is not None and entry.access == "readonly"

        # Transition B into active running state (lastSeq=1, no effectiveAccess on peer wire)
        with acting_as(pair.root_b):
            write_task_result(pair.root_b, b_task_id, "running")

        body = collection.BodyRuntime(pair.root_a, gateway_factory=lambda: pytest.fail("owned engine built"))
        try:
            # 1. Running state observed via real BodyRuntime._observe + unmocked hold_to_grant
            body._observe(current.run(pair.root_a, run_id))
            obs_run = body.observation(run_id)
            assert obs_run["answered"] is True and obs_run["run_state"] == "running", obs_run
            assert obs_run["waiting_on_user"] is False, obs_run
            mid_run = current.run(pair.root_a, run_id)
            assert mid_run.terminal_state == "" and mid_run.settled is False and mid_run.containment_fault == {}
            assert mid_run.access_profile == ""

            # 2. needs_input observed via BodyRuntime._observe + tick
            with acting_as(pair.root_b):
                question = ask(pair.root_b, b_task_id, "Which synthetic folder?",
                               published=published(pair.place_b, "Which synthetic folder?",
                                                   root=pair.root_b, link_id=pair.link_id))
            body._observe(current.run(pair.root_a, run_id))
            obs_wait = body.observation(run_id)
            assert obs_wait["answered"] is True and obs_wait["run_state"] == "running", obs_wait
            assert obs_wait["waiting_on_user"] is True, obs_wait
            mid_wait = current.run(pair.root_a, run_id)
            assert mid_wait.pending_interaction_ids == [question["question_id"]]
            assert mid_wait.terminal_state == "" and mid_wait.settled is False and mid_wait.containment_fault == {}

            # 3. Terminal observed and collected via BodyRuntime.tick()
            with acting_as(pair.root_b):
                import supervisor.queue as queue

                queue.PENDING[:] = [row for row in queue.PENDING if row.get("id") != b_task_id]
                write_task_result(pair.root_b, b_task_id, "completed", result="Synthetic count: 3.",
                                  finished_at="2026-11-24T12:00:00+00:00")
            body._observe(current.run(pair.root_a, run_id))
            body.tick()
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                done_entry = current.run(pair.root_a, run_id)
                if done_entry is not None and done_entry.settled and not current.collection_pending(done_entry):
                    break
                body.tick()
                time.sleep(0.05)
        finally:
            body.close(wait=True)

        final_entry = current.run(pair.root_a, run_id)
        assert final_entry is not None and final_entry.terminal_state == "succeeded", final_entry
        assert final_entry.settled is True and current.collection_pending(final_entry) is False
        assert final_entry.containment_fault == {} and final_entry.access_profile == ""
        retained = collection.retained_payload(pair.root_a, final_entry)
        assert retained is not None and retained["status"] == "terminal" and retained["state"] == "succeeded", retained
        assert retained["access_evidence"]["effective"] == "" and retained["access_evidence"]["verified"] is False
        assert "containment_breach" not in retained
        reread = retained_terminal_result(sender, run_id)
        assert reread is not None and reread["settlement"]["settled"] is True
        assert reread["settlement"]["ledger_recorded"] is False and reread["settlement"]["peer_billed"] is True
        assert not any(path.endswith("/cancel") for _m, path in requests_seen), requests_seen
        assert pair.row(run_id)["status"] == "completed"


def _reading_arguments(call):
    """The arguments one Safety reading was asked about, as the Supervisor read them."""
    prompt = call["user_prompt"]
    return json.loads(prompt[prompt.index("\nArguments:\n") + len("\nArguments:\n"):prompt.index("\nIs this safe?")])


def test_the_letter_is_read_once_as_sent_before_each_creating_post_and_a_lookup_reads_nothing(
        ctx, tmp_path, receiver, supervisor, monkeypatch):
    """Through the real ToolRegistry and the real Safety path (hermetic Supervisor, a fake physical model): the
    registry leaves the raw call unread; the letter is read once as it is sent (its place, link, words, title, ceiling,
    chain and protocol, never its id) right before its POST, a first send and a creating retry alike; a retry whose
    lookup finds the letter there reads and sends nothing; one the recorded bytes may no longer carry is refused
    before any reading; a reading that does not let it through sends nothing, a later identical call goes."""
    from ouroboros import peer_letters, safety
    from ouroboros.tools.registry import ToolRegistry

    h = supervisor
    monkeypatch.setattr(safety, "check_peer_assignment", peer_letters.check_peer_assignment)  # the real seam
    monkeypatch.setattr(peer_links, "has_outgoing_link", lambda root: True)
    monkeypatch.setattr(peer_delegate, "_remaining_known_usd", lambda c, drive: 5.0)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx = ctx
    read_at_post: list[int] = []
    receiver.hooks.append(lambda request: read_at_post.append(len(h.calls)) or None
                          if request.method == "POST" and request.url.path == "/api/letters" else None)

    def send(**arguments):
        return json.loads(registry.execute_result("delegate_to_ouroboros", {"place_id": PLACE_ID, **arguments}).text)

    started = send(text="Count the rows.", title="Count", cost_ceiling_usd=2.0)
    assert started["status"] == "started" and len(h.calls) == 1 and read_at_post == [1], started
    posted = json.loads(receiver.posts[-1])
    assert _reading_arguments(h.calls[-1]) == {**{key: value for key, value in posted.items() if key != "letter_id"},
                                               "place_id": PLACE_ID, "link_id": LINK_ID}, "read exactly as sent"

    # Accepted there, its answer lost: the retry's lookup binds it with no reading and no POST.
    receiver.hooks.insert(0, lambda request: (receiver.letters.setdefault(json.loads(request.content)["letter_id"], {
        "body": json.loads(request.content), "state": "accepted_running", "phase": "queued"}) and None)
        or httpx.Response(503, json={"error": "unconfirmed"})
        if request.method == "POST" and request.url.path == "/api/letters" and len(receiver.posts) == 2 else None)
    lost = send(text="Lint the tree.", cost_ceiling_usd=2.0)
    assert lost["status"] == "unconfirmed" and len(h.calls) == 2, lost
    posts = len(receiver.posts)
    bound = send(retry_of=lost["pending_invocation_id"])
    assert bound["status"] == "started" and bound["recovered_via_lookup"] is True, bound
    assert len(h.calls) == 2 and len(receiver.posts) == posts, "a lookup reads nothing and sends nothing"

    # Never arrived: a creating retry meets what a first send meets, with its recorded bytes.
    receiver.hooks.insert(0, lambda request, lost_at=posts + 1: httpx.Response(503, json={"error": "unconfirmed"})
                          if request.method == "POST" and request.url.path == "/api/letters"
                          and len(receiver.posts) == lost_at else None)
    missing = send(text="Run the unit suite.", title="CI", cost_ceiling_usd=2.0)
    token, posts = missing["pending_invocation_id"], len(receiver.posts)
    assert missing["status"] == "unconfirmed" and len(h.calls) == 3, missing
    monkeypatch.setattr(peer_delegate, "_remaining_known_usd", lambda c, drive: 0.75)
    over = send(retry_of=token)
    assert over["reason"] == "sender_budget_exceeded" and "definitely_unrun" not in over, over
    assert len(h.calls) == 3 and len(receiver.posts) == posts and _record(tmp_path, token)["state"] == "pending"
    monkeypatch.setattr(peer_delegate, "_remaining_known_usd", lambda c, drive: 5.0)
    h.replies.append({"status": "SUSPICIOUS", "reason": "look again"})
    returned = registry.execute_result("delegate_to_ouroboros", {"place_id": PLACE_ID, "retry_of": token})
    assert returned.code == "SAFETY_RETURNED" and len(h.calls) == 4 and len(receiver.posts) == posts, returned.text
    assert _reading_arguments(h.calls[-1])["text"] == "Run the unit suite." and _record(tmp_path, token)["state"] == "pending"
    h.later()
    sent = json.loads(registry.execute_result("delegate_to_ouroboros", {"place_id": PLACE_ID, "retry_of": token}).text
                      .split("\n\n⚠️ SAFETY_NOTE")[0])
    assert sent["status"] == "started" and sent["run_id"] == token and len(h.calls) == 4, sent
    assert receiver.posts[-1] == receiver.posts[-2], "the recorded bytes, once the later identical call let them go"

    # A first send its reading holds sends nothing: its intention row closes, definitely unrun.
    monkeypatch.setattr(custody, "new_invocation_id", lambda: "held-letter-0001")
    h.replies.append(RuntimeError("the checker is down"))
    held = registry.execute_result("delegate_to_ouroboros", {"place_id": PLACE_ID, "text": "Delete the cache."})
    assert held.code == "SAFETY_UNAVAILABLE" and len(receiver.posts) == posts + 1, held.text
    assert _record(tmp_path, "held-letter-0001")["state"] == "failed_definite"


def test_a_returned_letter_keeps_the_exact_ceiling_it_was_read_with(ctx, tmp_path, receiver, supervisor, monkeypatch):
    """A reading that does not let a letter go keeps the exact ceiling it read for the identical request. An explicit
    ceiling that still fits is that same operation again: the mind's later identical call goes unread. A ceiling this
    tree's remaining budget chose is said to be so, and once the budget is lower the identical call is refused with
    what remains, reading nothing (bounded, never a loop of paid readings); an explicit smaller ceiling is another
    operation, read once. Synthetic numbers only."""
    from ouroboros import peer_letters, safety
    from ouroboros.tools.registry import ToolRegistry

    h = supervisor
    monkeypatch.setattr(safety, "check_peer_assignment", peer_letters.check_peer_assignment)  # the real seam
    monkeypatch.setattr(peer_links, "has_outgoing_link", lambda root: True)
    budget = {"remaining": 5.0}
    monkeypatch.setattr(peer_delegate, "_remaining_known_usd", lambda c, drive: budget["remaining"])
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx = ctx

    def send(**arguments):
        return registry.execute_result("delegate_to_ouroboros", {"place_id": PLACE_ID, **arguments})

    def started(result):
        return json.loads(result.text.split("\n\n⚠️ SAFETY_NOTE")[0])

    h.replies.append({"status": "SUSPICIOUS", "reason": "look again"})
    returned = send(text="Count the rows.", cost_ceiling_usd=2.0)
    assert returned.code == "SAFETY_RETURNED" and len(h.calls) == 1 and receiver.posts == [], returned.text
    assert "was not your explicit choice" not in returned.text
    budget["remaining"] = 4.25  # the mind's own step spent from it: its explicit ceiling still fits
    h.later()
    sent = started(send(text="Count the rows.", cost_ceiling_usd=2.0))
    assert sent["status"] == "started" and sent["cost_ceiling_usd"] == 2.0 and len(h.calls) == 1, sent

    budget["remaining"] = 3.5
    h.replies.append({"status": "SUSPICIOUS", "reason": "look again"})
    derived = send(text="Lint the tree.")
    assert derived.code == "SAFETY_RETURNED" and len(h.calls) == 2, derived.text
    assert _reading_arguments(h.calls[-1])["cost_ceiling_usd"] == 3.5 and "was not your explicit choice" in derived.text
    budget["remaining"] = 3.25
    h.later()
    posts = len(receiver.posts)
    for _ in range(2):  # the identical call, again and again: one bounded answer, no reading, nothing sent
        lowered = json.loads(send(text="Lint the tree.").text)
        assert lowered["reason"] == "sender_budget_lowered" and lowered["definitely_unrun"] is True, lowered
        assert (lowered["read_ceiling_usd"], lowered["remaining_known_usd"]) == (3.5, 3.25), lowered
        assert "explicit cost_ceiling_usd below what remains" in lowered["detail"], lowered
    assert len(h.calls) == 2 and len(receiver.posts) == posts
    smaller = started(send(text="Lint the tree.", cost_ceiling_usd=1.0))
    assert smaller["status"] == "started" and smaller["cost_ceiling_usd"] == 1.0 and len(h.calls) == 3, smaller
    assert json.loads(receiver.posts[-1])["cost_ceiling_usd"] == 1.0

    # A reading that lets a letter go may itself spend: the boundary is read once more before the handoff, and an
    # exact ceiling that no longer fits is held, typed, with nothing sent and no second reading.
    read_so_far = len(h.calls)
    monkeypatch.setattr(peer_delegate, "_remaining_known_usd",
                        lambda c, drive: 5.0 if len(h.calls) == read_so_far else 1.5)
    posts = len(receiver.posts)
    spent = json.loads(send(text="Check the cache.", cost_ceiling_usd=2.0).text)
    assert spent["reason"] == "sender_budget_lowered" and spent["definitely_unrun"] is True, spent
    assert (spent["read_ceiling_usd"], spent["remaining_known_usd"]) == (2.0, 1.5), spent
    assert len(h.calls) == read_so_far + 1 and len(receiver.posts) == posts, "read once, nothing sent"
    again = json.loads(send(text="Check the cache.", cost_ceiling_usd=2.0).text)
    assert again["reason"] == "sender_budget_lowered" and len(h.calls) == read_so_far + 1, again


def test_a_confirmed_answer_reads_as_delivered_before_any_new_decision(ctx, receiver, monkeypatch, link):
    """An answer already delivered (its receipt's msg_id pinned here for this starter, run, question and bytes) reads
    as delivered before any new decision: after my place is Paused, with no words reading, launch or request. Other
    words for that question are refused before their reading."""
    from ouroboros.delegate_interactions import _delegate_answer

    run_id = _call(ctx, place_id=PLACE_ID, text="ask me")["run_id"]
    receiver.letters[run_id].update(state="needs_input", question={"question_id": "q1", "text": "which?"})
    rows = [{"question_id": "q1", "free_text": "main"}]
    first = json.loads(_delegate_answer(ctx, run_id, "q1", rows).text)
    assert first["status"] == "delivered" and first["receipt"]["msg_id"], first
    monkeypatch.setattr(peer_links, "outgoing_link", lambda place_id, drive_root=None: None)  # my place is Paused
    monkeypatch.setattr(peer_links, "stopping_link",
                        lambda place_id, drive_root=None: link if place_id == PLACE_ID else None)
    monkeypatch.setattr("ouroboros.peer_letters.answer_refusal",
                        lambda *_args: pytest.fail("a settled answer met a new reading"))
    calls = len(receiver.calls)
    again = json.loads(_delegate_answer(ctx, run_id, "q1", rows).text)
    assert again["status"] == "delivered" and again["replayed"] is True and again["receipt"] == first["receipt"], again
    other = json.loads(_delegate_answer(ctx, run_id, "q1", [{"question_id": "q1", "free_text": "another"}]).text)
    assert other["reason"] == "peer_answer_conflict", other
    assert len(receiver.calls) == calls, "nothing read or sent for either"


def test_an_answers_post_passes_this_tasks_launch_gate(ctx, receiver, monkeypatch):
    """A Pause or Stop of the sending task that lands before the answer's POST keeps it from leaving: typed
    ``answer_held``, nothing sent or pinned; once the task runs again the same answer goes."""
    from ouroboros import owner_pause
    from ouroboros.delegate_interactions import _delegate_answer
    from ouroboros.delegate_message_store import read_peer_answer_pin

    run_id = _call(ctx, place_id=PLACE_ID, text="ask me")["run_id"]
    receiver.letters[run_id].update(state="needs_input", question={"question_id": "q1", "text": "which?"})
    rows = [{"question_id": "q1", "free_text": "main"}]
    real_gate = owner_pause.run_operation

    def paused(source, function, *args, **kwargs):
        raise owner_pause.OwnerPauseRefused("owner_paused")
    monkeypatch.setattr(owner_pause, "run_operation", paused)
    posts = len(receiver.posts)
    held = json.loads(_delegate_answer(ctx, run_id, "q1", rows).text)
    assert held["reason"] == "answer_held" and "nothing was sent" in held["detail"], held
    entry = current.run(ctx.drive_root, run_id)
    assert len(receiver.posts) == posts and read_peer_answer_pin(ctx, entry, "q1") is None
    monkeypatch.setattr(owner_pause, "run_operation", real_gate)
    answered = json.loads(_delegate_answer(ctx, run_id, "q1", rows).text)
    assert answered["status"] == "delivered" and len(receiver.posts) == posts + 1, answered


def test_a_question_or_answer_held_there_reads_as_held_with_its_next_step(ctx, receiver):
    """What the receiver holds now is neither running work nor an empty answer: a question held there is
    ``held_there`` with no words to answer; a done letter's answer held by a passing hold keeps the run open and is
    read once given, with no new work there; ``rules_changed`` ends it with that fact."""
    from ouroboros.tools import delegate as delegate_tools

    run_id = _call(ctx, place_id=PLACE_ID, text="ask me")["run_id"]
    receiver.letters[run_id].update(state="needs_input", question={"question_id": "q1", "held": "rules_changed"})
    payload = json.loads(delegate_tools._read_run(ctx, run_id, 0))
    assert payload["status"] != "waiting_on_user" and not payload.get("pending_interactions"), payload
    assert payload["held_there"]["what"] == "question" and payload["held_there"]["reason"] == "rules_changed"
    assert "interaction_id=q1" in payload["held_there"]["note"], payload
    receiver.letters[run_id]["question"] = {"question_id": "q1", "held": "rules_changed", "delivery_state": "reserved"}
    reserved = json.loads(delegate_tools._read_run(ctx, run_id, 0))["held_there"]
    assert reserved["delivery_state"] == "reserved" and "Only that same answer may go" in reserved["note"], reserved
    assert "cannot see" not in reserved["note"], "a reserved question takes no new words"

    receiver.letters[run_id].update(state="done", result="", result_available=False, result_held="link_paused")
    receiver.letters[run_id].pop("question")
    held = json.loads(delegate_tools._read_run(ctx, run_id, 0))
    assert held["state"] == "running" and held["held_there"]["what"] == "result", held
    assert current.run(ctx.drive_root, run_id).terminal_state == "", "a held answer settles nothing"
    receiver.letters[run_id].update(result="all green", result_available=True)
    receiver.letters[run_id].pop("result_held")
    done = json.loads(delegate_tools._read_run(ctx, run_id, 0))
    assert done["status"] == "terminal" and done["primary_output"]["text"] == "all green", done

    other = _call(ctx, place_id=PLACE_ID, text="another")["run_id"]
    receiver.letters[other].update(state="done", result="", result_available=False, result_held="rules_changed")
    ended = json.loads(delegate_tools._read_run(ctx, other, 0))
    assert ended["status"] == "terminal" and ended["held_there"]["reason"] == "rules_changed", ended
    final = ended["final_summary"]
    assert final["result_available"] is False and final["result_held"] == "rules_changed", final


def test_a_sweep_does_not_cancel_a_lookup_only_start_for_a_pause_alone(tmp_path, monkeypatch):
    """A restored lookup-only start whose task ended without a cancellation is kept through a Pause of my place,
    as an Active link keeps it; a letter that was not restored keeps its Stop through the Pause."""
    from ouroboros import delegate_custody_reconcile as reconcile

    cancels, faults = [], []
    monkeypatch.setattr(custody, "cancel_and_verify",
                        lambda *a, **kw: cancels.append(a[2].run_id) or {"outcome": custody.CANCEL_CONFIRMED})
    monkeypatch.setattr(custody, "record_containment_fault",
                        lambda root, run, reason, detail="", **kw: faults.append(reason))
    monkeypatch.setattr(custody, "emit", lambda *a, **kw: True)
    monkeypatch.setattr(custody, "output_disposition", lambda run: {})
    monkeypatch.setattr(reconcile, "_owner_terminal_is_deliberate",
                        lambda root, task_id, cancellation_only=False: not cancellation_only)  # finished, not stopped

    class PausedHere:
        def get_run(self, run_id, **_kw):
            raise wire.ClaudexorUnavailable("peer_link_unavailable", "my place is Paused; nothing was sent")

    def letter(run_id, resume):
        return custody.RunCustody(run_id=run_id, task_id="t-a", invocation_id=run_id, engine_ref=dict(PEER_REF),
                                  route_id="peer", category="peer", source="peer_letter", resume_start_state=resume)

    kept = reconcile._reconcile_one(tmp_path, PausedHere(), letter("letter-restored", "running"))
    assert kept["action"] == "unreadable" and cancels == [], kept
    stopped = reconcile._reconcile_one(tmp_path, PausedHere(), letter("letter-plain", None))
    assert stopped["action"] == "cancelled" and cancels == ["letter-plain"], stopped


def test_observation_loss_wakes_once_per_episode_and_never_over_another_fault(tmp_path, monkeypatch):
    """A lost observation wakes a wait on that run once per episode (a later wait sleeps until something happens),
    and a new episode wakes again; it is never recorded over another fault of the run."""
    import threading
    from types import SimpleNamespace
    from ouroboros import delegate_collection, delegate_supervision, model_sleep

    fault = {"reason": "peer_observation_lost", "ts": "2026-11-25T10:00:00+00:00", "event_id": "e1"}
    run = SimpleNamespace(settled=False, containment_fault=dict(fault))
    states: dict = {"r1": {}}
    monkeypatch.setattr(model_sleep, "_owned_runs", lambda ctx, held_only=False: {"r1": run})
    monkeypatch.setattr(delegate_supervision, "_load_state", lambda ctx, run_id: states[run_id])
    waiter = SimpleNamespace(task_id="t-a")
    assert delegate_collection.peer_observation_lost_wake(waiter, ["r1"]) == "run_observation_lost:r1"
    states["r1"] = {"last_acknowledged_wake": {"acknowledged_at": "2026-11-25T10:00:05+00:00"}}
    assert delegate_collection.peer_observation_lost_wake(waiter, ["r1"]) == "", "shown once: the next wait sleeps"
    run.containment_fault = {**fault, "ts": "2026-11-25T10:10:00+00:00", "event_id": "e2"}
    assert delegate_collection.peer_observation_lost_wake(waiter, ["r1"]) == "run_observation_lost:r1"

    recorded: list = []
    stronger = SimpleNamespace(run_id="r1", containment_fault={"reason": "cancel_unreachable"})
    monkeypatch.setattr(delegate_collection.current, "run", lambda root, run_id: stronger)
    monkeypatch.setattr(delegate_collection.c, "record_containment_fault", lambda *a, **kw: recorded.append(a[2]))
    body = SimpleNamespace(root=tmp_path, lock=threading.Lock(), observation_gaps={},
                           _observation_key=lambda entry: ("peer", "r1", "unavailable"))
    lost = wire.ClaudexorUnavailable("peer_link_unavailable", "Paused")
    delegate_collection.BodyRuntime._record_peer_observation_lost(body, stronger, lost)
    assert recorded == [] and body.observation_gaps == {"r1": ("peer", "r1", "unavailable")}
    stronger.containment_fault = {}
    delegate_collection.BodyRuntime._record_peer_observation_lost(body, stronger, lost)
    assert recorded == ["peer_observation_lost"]


def test_oversized_text_refused_before_safety_and_before_answer_pinning(ctx, tmp_path, receiver, monkeypatch):
    """Envelope limits (TEXT_MAX_CHARS / TITLE_MAX_CHARS) are validated before Safety and before pinning
    a peer answer, so an oversized answer never locks out a corrected retry."""
    from ouroboros import delegate_message_store as msg_store, peer_letters, store
    from ouroboros.gateway.peer_contracts import TEXT_MAX_CHARS, TITLE_MAX_CHARS
    from ouroboros.tools.delegate import _delegate_answer

    # Fresh assignment rejects oversized text and title before claiming a start row.
    too_long = _call(ctx, place_id=PLACE_ID, text="a" * (TEXT_MAX_CHARS + 1))
    assert too_long["reason"] == "invalid_letter" and too_long["definitely_unrun"] is True
    too_titled = _call(ctx, place_id=PLACE_ID, text="ok", title="t" * (TITLE_MAX_CHARS + 1))
    assert too_titled["reason"] == "invalid_letter" and too_titled["definitely_unrun"] is True
    assert receiver.posts == []

    # Start a valid letter and open a question on FakeReceiver.
    started = _call(ctx, place_id=PLACE_ID, text="need answer")
    run_id = started["run_id"]
    receiver.letters[run_id]["state"] = "needs_input"
    receiver.letters[run_id]["question"] = {"question_id": "q-limit", "text": "Which path?"}

    # Spy on words_refusal (inside real answer_refusal) to ensure oversized text fails BEFORE Safety.
    words_checked = []
    monkeypatch.setattr(peer_letters, "answer_refusal",
                        lambda c, entry, iid, rows: words_checked.append(iid) or ""
                        if not _check_answer_valid(iid, rows) else "")

    def _check_answer_valid(iid, rows):
        try:
            peerwire._validated_answer_text(iid, rows)
            return False
        except wire.ClaudexorUnavailable:
            return True

    oversized = json.loads(_delegate_answer(
        ctx, run_id, "q-limit", [{"question_id": "q-limit", "free_text": "z" * (TEXT_MAX_CHARS + 1)}]
    ).text)
    assert oversized["status"] == "rejected" and f"at most {TEXT_MAX_CHARS}" in oversized["detail"], oversized
    assert words_checked == [], "oversized answer must not trigger Safety"
    assert len(receiver.posts) == 1, "oversized answer must not POST"
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.set_rows(msg_store._peer_answer_pin_set(run_id)) == {}, "oversized answer must not pin slot"

    # A valid shortened answer now succeeds cleanly on the unpinned slot.
    ok = json.loads(_delegate_answer(
        ctx, run_id, "q-limit", [{"question_id": "q-limit", "free_text": "/srv/data"}]
    ).text)
    assert ok["status"] == "delivered", ok
    assert len(receiver.posts) == 2


def test_get_run_strict_states_and_answer_binding_receipt_on_mock_double(link, monkeypatch):
    """Mock-double checks (httpx.MockTransport, NOT real receiver):
    - Unknown letter state and malformed needs_input raise malformed_response.
    - answer_interaction takes only ``LetterAnswerReceipt`` for its own reply (delivered, the exact in_reply_to and
      the mailbox row's msg_id); GET's form, a reservation or another question's receipt stays unconfirmed."""
    canned: dict = {"status": 200, "json": {}}
    monkeypatch.setattr(peer_links, "outgoing_link", lambda place_id, drive_root=None: link)

    def handler(request):
        return httpx.Response(canned["status"], json=canned["json"])

    real_client = httpx.Client
    monkeypatch.setattr(peerwire.httpx, "Client",
                        lambda **kw: real_client(**kw, transport=httpx.MockTransport(handler)))
    gw = peerwire.PeerGateway(link)
    try:
        canned["json"] = {"letter_id": "L1", "state": "invented_state"}
        with pytest.raises(wire.ClaudexorUnavailable) as exc:
            gw.get_run("L1")
        assert exc.value.code == "malformed_response"

        canned["json"] = {"letter_id": "L1", "state": "needs_input", "question": {"question_id": "q1", "text": "  "}}
        with pytest.raises(wire.ClaudexorUnavailable) as exc:
            gw.get_run("L1")
        assert exc.value.code == "malformed_response"

        canned["json"] = {"letter_id": "L1", "state": "needs_input"}
        with pytest.raises(wire.ClaudexorUnavailable) as exc:
            gw.get_run("L1")
        assert exc.value.code == "malformed_response"

        # Build the exact reply letter_id for ("L1", "q1", "yes").
        import hashlib
        rid = "reply-" + hashlib.sha256("L1\0q1\0yes".encode("utf-8")).hexdigest()[:32]
        rows = [{"questionId": "q1", "freeText": "yes"}]

        asked = {"letter_id": "L1", "question_id": "q1"}
        for bad_body in (
            {"letter_id": rid, "state": "accepted_running"},  # GET's form: no receipt at all
            {"letter_id": rid, "state": "accepted_running", "delivered": True, "in_reply_to": asked},  # no mailbox row
            {"letter_id": rid, "in_reply_to": asked, "delivered": False, "msg_id": "m-1"},  # a reservation
            {"letter_id": rid, "in_reply_to": {**asked, "question_id": "other-q"}, "delivered": True, "msg_id": "m-1"},
            {"letter_id": "reply-other", "in_reply_to": asked, "delivered": True, "msg_id": "m-1"},
            {"letter_id": rid, "in_reply_to": asked, "delivered": True, "msg_id": " "},
        ):
            canned["json"] = bad_body
            with pytest.raises(wire.ClaudexorUnavailable) as exc:
                gw.answer_interaction("L1", "q1", rows)
            assert exc.value.code == peerwire.UNCONFIRMED, bad_body
        canned["json"] = {"letter_id": rid, "in_reply_to": asked, "delivered": True, "msg_id": "m-1"}
        assert gw.answer_interaction("L1", "q1", rows)["status"] == "delivered"
    finally:
        gw.close()


def test_real_receiver_confirmed_answer_and_cached_observer_link_state_changes(tmp_path, monkeypatch):
    """Real in-process receiver (TestClient -> real peer_letters_api.py, owner_mailbox, peer_links):
    1. Clarification delivery succeeds only with a binding receipt and verified mailbox entry.
    2. Changing link eligibility (Pause, then unmarking mine=False) makes the cached PeerGateway send nothing at
       its next request (it reads the link as it stands, ``PeerGateway._request``); the body drops it, records
       peer_observation_lost with unknown_may_still_be_running (never claiming remote ended or restarting), and
       resumes cleanly when restored."""
    import threading
    from starlette.testclient import TestClient

    from ouroboros import delegate_collection as collection, install_identity, owner_mailbox
    from ouroboros.install_identity import acting_as
    from ouroboros.peer_letters import ask
    from ouroboros.places import AUTHOR_OWNER, STATE_ACTIVE, STATE_PAUSED, edit_place
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.delegate import _delegate_answer
    from ouroboros.tools.registry import ToolContext
    from tests._peer_letters_shared import introduce, published, receiver_client, safety_reads

    pair = introduce(tmp_path, monkeypatch)
    safety_reads(monkeypatch)  # the letter's one reading (check_peer_assignment) answered: not this test's subject
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(pair.root_a))
    monkeypatch.setattr(install_identity, "_CANONICAL_ROOTS", {})
    monkeypatch.setattr(wire, "ClaudexorGateway", lambda *a, **k: pytest.fail("owned engine built"))
    monkeypatch.setattr("ouroboros.peer_letters.answer_refusal", lambda ctx, entry, iid, rows: "")

    requests_seen: list[tuple[str, str]] = []
    req_lock = threading.Lock()
    with receiver_client(pair, monkeypatch) as client:
        real_request = client.request

        def recording_request(method, url, **kwargs):
            with req_lock:
                requests_seen.append((str(method), str(url)))
                return real_request(method, url, **kwargs)

        monkeypatch.setattr(client, "request", recording_request)
        monkeypatch.setattr(peerwire.httpx, "Client", lambda **kw: client if isinstance(client, TestClient) else None)
        monkeypatch.setattr(client, "close", lambda: None)

        sender = ToolContext(repo_dir=pair.repo, drive_root=pair.root_a)
        sender.task_id, sender.task_metadata = "t-a", {"root_task_id": "t-a"}
        with acting_as(pair.root_a):
            write_task_result(pair.root_a, "t-a", "running", root_task_id="t-a", task_attempt=1)
            started = _call(sender, place_id=pair.place_a, text="Check build state.", cost_ceiling_usd=0.5)
        assert started["status"] == "started", started
        run_id = started["run_id"]
        b_task_id = pair.task_id(run_id)

        with acting_as(pair.root_b):
            write_task_result(pair.root_b, b_task_id, "running")
            question = ask(pair.root_b, b_task_id, "Which target?",
                           published=published(pair.place_b, "Which target?",
                                               root=pair.root_b, link_id=pair.link_id))
        qid = question["question_id"]

        body = collection.BodyRuntime(pair.root_a, gateway_factory=lambda: pytest.fail("owned engine built"))
        try:
            # 1. Initial observation caches PeerGateway in EngineClients and sees needs_input.
            with acting_as(pair.root_a):
                body._observe(current.run(pair.root_a, run_id))
            obs = body.observation(run_id)
            assert obs["answered"] is True and obs["waiting_on_user"] is True, obs
            cached_before = body._work.held(current.run(pair.root_a, run_id))
            assert cached_before is not None

            # 2. Confirmed clarification delivery against real peer_letters_api + mailbox + settled receipt.
            with acting_as(pair.root_a):
                ans = json.loads(_delegate_answer(
                    sender, run_id, qid, [{"question_id": qid, "free_text": "linux-arm64"}]
                ).text)
            assert ans["status"] == "delivered", ans
            import hashlib
            reply_id = "reply-" + hashlib.sha256(f"{run_id}\0{qid}\0linux-arm64".encode("utf-8")).hexdigest()[:32]
            with acting_as(pair.root_b):
                receipt = peer_links.reply_receipt(pair.root_b, pair.link_id, reply_id)
                assert receipt is not None and receipt.get("delivered") is True and receipt.get("msg_id"), receipt
                assert owner_mailbox.place_event_retained(pair.root_b, b_task_id, receipt["msg_id"]) is True

            # 3. Pause A's place: cached client MUST be dropped immediately without any HTTP call or invented outcome.
            with acting_as(pair.root_a):
                edit_place(pair.root_a, pair.place_a, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
                count_before_pause = len(requests_seen)
                body._observe(current.run(pair.root_a, run_id))
            assert len(requests_seen) == count_before_pause, "paused link must not be polled via cached client"
            assert body._work.held(current.run(pair.root_a, run_id)) is None
            # (The body's own driver may finish a read it began before the Pause and keep that fact in ``observed``:
            # the durable record below is what this Pause leaves.)
            still_open = current.run(pair.root_a, run_id)
            assert still_open is not None and still_open.terminal_state == "" and still_open.settled is False
            assert still_open.containment_fault.get("reason") == "peer_observation_lost", still_open.containment_fault
            assert still_open.containment_fault.get("remote_work_state") == "unknown_may_still_be_running"
            assert still_open.containment_fault.get("link_error") == "peer_link_unavailable", still_open.containment_fault
            assert run_id in body.observation_gaps

            # 4. Restore Active -> observation key changes and observation resumes cleanly without restarting remote work.
            with acting_as(pair.root_a):
                edit_place(pair.root_a, pair.place_a, "set_state", author=AUTHOR_OWNER, state=STATE_ACTIVE)
                assert body._observation_key(still_open) != body.observation_gaps[run_id]
                body._observe(current.run(pair.root_a, run_id))
            obs_resumed = body.observation(run_id)
            assert obs_resumed["answered"] is True and obs_resumed["run_state"] == "running", obs_resumed
            assert body._work.held(current.run(pair.root_a, run_id)) is not None

            # 5. Unmark mine=False -> cached client dropped immediately, zero HTTP calls, custody stays open.
            with acting_as(pair.root_a):
                edit_place(pair.root_a, pair.place_a, "set_peer", author=AUTHOR_OWNER, mine=False, may_start=True)
                count_before_unmark = len(requests_seen)
                body._observe(current.run(pair.root_a, run_id))
            assert len(requests_seen) == count_before_unmark, "unmarked link must not be polled via cached client"
            assert body._work.held(current.run(pair.root_a, run_id)) is None
            assert run_id in body.observation_gaps
            unmarked_row = current.run(pair.root_a, run_id)
            assert unmarked_row is not None and unmarked_row.terminal_state == "" and unmarked_row.settled is False
            assert unmarked_row.containment_fault.get("remote_work_state") == "unknown_may_still_be_running"
        finally:
            body.close(wait=True)


def _pair_sender_ctx(pair, task_id="t-sender"):
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=pair.repo, drive_root=pair.root_a)
    ctx.task_id = task_id
    ctx.task_metadata = {"root_task_id": task_id}
    return ctx


def _bind_pair_sender_to_receiver(pair, client, monkeypatch):
    from starlette.testclient import TestClient
    from ouroboros import install_identity

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(pair.root_a))
    monkeypatch.setattr(install_identity, "_CANONICAL_ROOTS", {})
    monkeypatch.setattr(wire, "ClaudexorGateway", lambda *a, **k: pytest.fail("owned engine must not be built"))
    monkeypatch.setattr(peerwire.httpx, "Client", lambda **kw: client if isinstance(client, TestClient) else None)
    monkeypatch.setattr(client, "close", lambda: None)


def _seed_recovered_peer_run(pair, letter_id: str, *, resume_start_state: str = "unknown", task_id: str = "t-sender"):
    from ouroboros.install_identity import acting_as
    from ouroboros.task_results import write_task_result

    entry = custody.RunCustody(
        run_id=letter_id, task_id=task_id, invocation_id=letter_id,
        engine_ref={"kind": "peer", "place_id": pair.place_a, "link_id": pair.link_id},
        route_id="peer", category="peer", source="peer_letter",
        root_task_id=task_id, parent_task_id="", ledger_root=str(pair.root_a),
        access="readonly", delegated=True,
    )
    with acting_as(pair.root_a):
        write_task_result(pair.root_a, task_id, "running", root_task_id=task_id, task_attempt=1)
        assert custody.record_started(
            pair.root_a, entry,
            {"access": "readonly", "delegated": True, "resume_start_state": resume_start_state,
             "root_task_id": task_id},
        )
        return current.run(pair.root_a, letter_id)


def test_recovered_peer_binding_rejects_spoofed_mismatched_and_paused_observation(tmp_path, monkeypatch):
    """Confirmed peer binding requires genuine PeerGateway, exact link/place/run_id, confirmed link and no spoofed
    storeId; Paused permits Stop (for_stop=True) while refusing ordinary observation (for_stop=False)."""
    from dataclasses import replace
    from types import SimpleNamespace
    from ouroboros import delegate_resume
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, STATE_ACTIVE, STATE_PAUSED, edit_place
    from tests._peer_letters_shared import introduce

    pair = introduce(tmp_path, monkeypatch)
    entry = _seed_recovered_peer_run(pair, "ltr-peer-01")
    with acting_as(pair.root_a):
        active_link = peer_links.outgoing_link(pair.place_a, drive_root=pair.root_a)
    gw = peerwire.PeerGateway(active_link)
    try:
        good_detail = {"runId": "ltr-peer-01", "lastSeq": 1,
                       "summary": {"runId": "ltr-peer-01", "state": "running", "peerState": "accepted_running"},
                       "pendingInteractions": []}
        paused_detail = {"runId": "ltr-peer-01", "lastSeq": 1,
                         "summary": {"runId": "ltr-peer-01", "state": "running", "peerState": "accepted_running",
                                     "contentWithheld": "paused"},
                         "pendingInteractions": []}

        assert delegate_resume.recovered_cancel_store(gw, entry) is True
        assert delegate_resume.recovered_cancel_detail(gw, entry, good_detail) is True
        assert delegate_resume.recovered_cancel_detail(gw, entry, paused_detail, for_stop=False) is False
        assert delegate_resume.recovered_cancel_detail(gw, entry, paused_detail, for_stop=True) is True

        fake_gw = SimpleNamespace(engine_ref=dict(gw.engine_ref), store_identity="", _link=active_link)
        assert delegate_resume.recovered_cancel_store(fake_gw, entry) is False
        assert delegate_resume.recovered_cancel_detail(fake_gw, entry, good_detail) is False
        assert delegate_resume.recovered_cancel_detail(gw, entry, {**good_detail, "storeId": "daemon-1"}) is False
        assert delegate_resume.recovered_cancel_detail(gw, entry, {
            **good_detail, "runId": "ltr-other", "summary": {"runId": "ltr-other", "state": "running"},
        }) is False
        for bad in (replace(entry, invocation_id="ltr-different"), replace(entry, daemon_store_id="store-1"),
                    replace(entry, engine_ref={"kind": "peer", "place_id": pair.place_a, "link_id": "other"})):
            assert delegate_resume.recovered_cancel_store(gw, bad) is False

        with acting_as(pair.root_a):
            edit_place(pair.root_a, pair.place_a, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
        assert delegate_resume.recovered_cancel_store(gw, entry, for_stop=False) is False
        assert delegate_resume.recovered_cancel_detail(gw, entry, good_detail, for_stop=False) is False
        assert delegate_resume.recovered_cancel_store(gw, entry, for_stop=True) is True
        assert delegate_resume.recovered_cancel_detail(gw, entry, paused_detail, for_stop=True) is True

        with acting_as(pair.root_a):
            edit_place(pair.root_a, pair.place_a, "set_state", author=AUTHOR_OWNER, state=STATE_ACTIVE)
            edit_place(pair.root_a, pair.place_a, "set_peer", author=AUTHOR_OWNER, mine=False, may_start=False)
        assert delegate_resume.recovered_cancel_store(gw, entry, for_stop=False) is False
        assert delegate_resume.recovered_cancel_store(gw, entry, for_stop=True) is False
        with acting_as(pair.root_a):
            edit_place(pair.root_a, pair.place_a, "set_peer", author=AUTHOR_OWNER, mine=True, may_start=False)
            peer_links.revoke(pair.root_a, pair.link_id)
        assert delegate_resume.recovered_cancel_store(gw, entry, for_stop=False) is False
        assert delegate_resume.recovered_cancel_store(gw, entry, for_stop=True) is False
    finally:
        gw.close()


def test_recovered_peer_run_observed_through_body_wait_hold_and_recovery_then_settles(tmp_path, monkeypatch):
    """Real in-process consumer test: recovered peer run stays observable across BodyRuntime, delegate_wait,
    delegate_hold and delegate_recovery, refuses withheld stubs as ordinary observation, and settles cleanly."""
    import time
    import supervisor.queue as queue
    from ouroboros import delegate_collection as collection, delegate_hold, delegate_recovery
    from ouroboros.install_identity import acting_as
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import delegate as delegate_tools
    from tests._peer_letters_shared import admitted, introduce, letter, receiver_client

    pair = introduce(tmp_path, monkeypatch)
    body_req = letter(letter_id="ltr-rec-obs", text="Count synthetic widgets.")
    with receiver_client(pair, monkeypatch) as client:
        assert client.post("/api/letters", json=body_req, headers=pair.headers()).status_code == 200
        b_task_id = pair.task_id(body_req["letter_id"])
        with acting_as(pair.root_b):
            write_task_result(pair.root_b, b_task_id, "running")

        _bind_pair_sender_to_receiver(pair, client, monkeypatch)
        entry = _seed_recovered_peer_run(pair, body_req["letter_id"], resume_start_state="unknown")
        assert entry.resume_start_state == "unknown" and entry.daemon_store_id == ""
        sender = _pair_sender_ctx(pair)

        body = collection.BodyRuntime(pair.root_a, gateway_factory=lambda: pytest.fail("owned engine built"))
        try:
            with acting_as(pair.root_a):
                body._observe(current.run(pair.root_a, body_req["letter_id"]))
            obs = body.observation(body_req["letter_id"])
            assert obs["answered"] is True and obs["run_state"] == "running", obs
            updated = current.run(pair.root_a, body_req["letter_id"])
            assert updated.resume_start_state == "running" and updated.containment_fault == {}

            with acting_as(pair.root_a):
                active_link = peer_links.outgoing_link(pair.place_a, drive_root=pair.root_a)
            gw = peerwire.PeerGateway(active_link)
            try:
                with pytest.raises(ValueError, match="withheld"):
                    body._absorb_detail(updated, gw, {
                        "runId": body_req["letter_id"], "lastSeq": 1,
                        "summary": {"runId": body_req["letter_id"], "state": "running", "contentWithheld": "paused"},
                        "pendingInteractions": [],
                    })
            finally:
                gw.close()

            with acting_as(pair.root_a):
                assert json.loads(delegate_tools._read_run(sender, body_req["letter_id"], 0))["state"] == "running"
                assert delegate_hold._leaf_probe_live(sender, body_req["letter_id"]) is True
                assert delegate_recovery._run_probe_error(pair.root_a, updated) == ""

            with acting_as(pair.root_b):
                queue.PENDING[:] = [row for row in queue.PENDING if row.get("id") != b_task_id]
                write_task_result(pair.root_b, b_task_id, "completed", result="4 widgets.",
                                  peer_reply=admitted(pair, "4 widgets."),
                                  finished_at="2026-11-24T12:00:00+00:00")
            with acting_as(pair.root_a):
                body._observe(current.run(pair.root_a, body_req["letter_id"]))
                body.tick()
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    done = current.run(pair.root_a, body_req["letter_id"])
                    if done is not None and done.settled and not current.collection_pending(done):
                        break
                    body.tick()
                    time.sleep(0.05)
        finally:
            body.close(wait=True)

        final_entry = current.run(pair.root_a, body_req["letter_id"])
        assert final_entry.terminal_state == "succeeded" and final_entry.settled is True
        retained = collection.retained_payload(pair.root_a, final_entry)
        assert retained["primary_output"]["text"] == "4 widgets." and retained["final_summary"]["result_available"] is True


def test_recovered_peer_run_stop_active_paused_reread_and_malformed_cancel(tmp_path, monkeypatch):
    """Recovered peer runs support Stop on Active and Paused links while ordinary waiting stays blocked on Pause,
    _reread_requested_stop confirms termination through Pause, and malformed cancel answers never claim accepted."""
    from ouroboros import external_runs, peer_letters
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, STATE_PAUSED, edit_place
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import delegate as delegate_tools
    from tests._peer_letters_shared import introduce, letter, published, receiver_client

    pair = introduce(tmp_path, monkeypatch)
    active_letter = letter(letter_id="ltr-stop-act", text="Work to cancel while active.")
    paused_letter = letter(letter_id="ltr-stop-pau", text="Work to cancel while paused.")
    reread_letter = letter(letter_id="ltr-stop-rer", text="Work to reread after cancel requested.")
    bad_cancel_letter = letter(letter_id="ltr-stop-bad", text="Work with malformed cancel reply.")

    with receiver_client(pair, monkeypatch) as client:
        for req in (active_letter, paused_letter, reread_letter, bad_cancel_letter):
            assert client.post("/api/letters", json=req, headers=pair.headers()).status_code == 200
            with acting_as(pair.root_b):
                write_task_result(pair.root_b, pair.task_id(req["letter_id"]), "running")
        with acting_as(pair.root_b):
            assert peer_letters.ask(
                pair.root_b, pair.task_id(paused_letter["letter_id"]), "SECRET-PAUSED-QUESTION?",
                published=published(pair.place_b, "SECRET-PAUSED-QUESTION?", root=pair.root_b, link_id=pair.link_id),
            )["question_id"]

        seen: list[tuple[str, str]] = []
        real_req, override_cancel = client.request, {"active": False}

        def spy_req(method, url, **kw):
            seen.append((str(method), str(url)))
            if override_cancel["active"] and method == "POST" and str(url).endswith("/cancel"):
                return httpx.Response(200, json={"letter_id": "foreign-letter", "state": "cancelled"})
            return real_req(method, url, **kw)

        monkeypatch.setattr(client, "request", spy_req)
        _bind_pair_sender_to_receiver(pair, client, monkeypatch)
        sender = _pair_sender_ctx(pair)

        _seed_recovered_peer_run(pair, active_letter["letter_id"], resume_start_state="unknown")
        with acting_as(pair.root_a):
            stopped_act = json.loads(delegate_tools._delegate_cancel(sender, active_letter["letter_id"], "stop now").text)
        assert stopped_act["status"] == custody.CANCEL_CONFIRMED and stopped_act["state"] == "cancelled", stopped_act
        assert current.run(pair.root_a, active_letter["letter_id"]).settled is True
        assert pair.row(active_letter["letter_id"])["status"] == "cancelled"

        bad_entry = _seed_recovered_peer_run(pair, bad_cancel_letter["letter_id"], resume_start_state="running")
        override_cancel["active"] = True
        with acting_as(pair.root_a):
            bad_stop = json.loads(delegate_tools._delegate_cancel(sender, bad_cancel_letter["letter_id"], "stop").text)
        override_cancel["active"] = False
        assert bad_stop["status"] == custody.CANCEL_CONTAINMENT_FAULT and bad_stop["accepted"] is False
        assert bad_stop["run_may_still_be_live"] is True and current.run(pair.root_a, bad_entry.run_id).settled is False

        _seed_recovered_peer_run(pair, paused_letter["letter_id"], resume_start_state="unknown")
        with acting_as(pair.root_a):
            edit_place(pair.root_a, pair.place_a, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
            before_wait = len(seen)
            blocked_wait = json.loads(delegate_tools._read_run(sender, paused_letter["letter_id"], 0))
            assert blocked_wait["reason"] == "peer_observation_lost"
            assert blocked_wait["remote_work_state"] == "unknown_may_still_be_running"
            assert seen[before_wait:] == [], "ordinary waiting must never send through Pause"
            before_stop = len(seen)
            stopped_pau = json.loads(delegate_tools._delegate_cancel(sender, paused_letter["letter_id"], "stop paused").text)
        assert stopped_pau["status"] == custody.CANCEL_CONFIRMED and stopped_pau["state"] == "cancelled"
        assert not any(path.endswith("/result") for _m, path in seen[before_stop:])
        assert "SECRET-PAUSED-QUESTION?" not in json.dumps(stopped_pau)
        assert pair.row(paused_letter["letter_id"])["status"] == "cancelled"

        reread_entry = _seed_recovered_peer_run(pair, reread_letter["letter_id"], resume_start_state="running")
        with acting_as(pair.root_a):
            custody.emit(pair.root_a, custody.CANCEL_OUTCOME, {
                "run_id": reread_entry.run_id, "task_id": reread_entry.task_id,
                "outcome": custody.CANCEL_REQUESTED, "accepted": True, "control_status": "accepted",
                "state": "running", "control_owed": False,
            })
            fresh_reread = current.run(pair.root_a, reread_entry.run_id)
            fresh_gw = peerwire.peer_gateway(fresh_reread.engine_ref)
            try:
                assert external_runs._reread_requested_stop(pair.root_a, fresh_gw, fresh_reread)["outcome"] == custody.CANCEL_REQUESTED
                with acting_as(pair.root_b):
                    write_task_result(pair.root_b, pair.task_id(reread_letter["letter_id"]), "cancelled")
                assert external_runs._reread_requested_stop(pair.root_a, fresh_gw, fresh_reread)["outcome"] == custody.CANCEL_CONFIRMED
            finally:
                fresh_gw.close()


def test_stop_read_on_done_letter_skips_result_and_withheld_terminal_never_fakes_empty_result(tmp_path, monkeypatch):
    """Stop-read of a done letter skips /result even when Active; the withheld terminal never turns into a fake
    empty ordinary result: while Paused it honestly discloses content_withheld (output_delivery.complete=False),
    and once Active a subsequent ordinary read/collection upgrades to the real /result and cost."""
    import supervisor.queue as queue
    from ouroboros import delegate_collection as collection
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, STATE_ACTIVE, STATE_PAUSED, edit_place
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import delegate as delegate_tools
    from tests._peer_letters_shared import admitted, introduce, letter, receiver_client

    pair = introduce(tmp_path, monkeypatch)
    act_req = letter(letter_id="ltr-done-act", text="Done before active stop.")
    pau_req = letter(letter_id="ltr-done-pau", text="Done before paused stop.")

    with receiver_client(pair, monkeypatch) as client:
        for req in (act_req, pau_req):
            assert client.post("/api/letters", json=req, headers=pair.headers()).status_code == 200
            tid = pair.task_id(req["letter_id"])
            with acting_as(pair.root_b):
                queue.PENDING[:] = [row for row in queue.PENDING if row.get("id") != tid]
                write_task_result(
                    pair.root_b, tid, "completed", result="42 widgets.",
                    peer_reply=admitted(pair, "42 widgets."),
                    accounted_upper_bound_usd_with_children=0.12, cost_final=True,
                    finished_at="2026-11-24T15:00:00+00:00",
                )
        seen: list[tuple[str, str]] = []
        real_req = client.request

        def spy_req(method, url, **kw):
            seen.append((str(method), str(url)))
            return real_req(method, url, **kw)

        monkeypatch.setattr(client, "request", spy_req)
        _bind_pair_sender_to_receiver(pair, client, monkeypatch)
        sender = _pair_sender_ctx(pair)

        # 1. Active Stop on already-done letter: ZERO /result calls; subsequent collect_terminal & delegate_wait upgrade!
        _seed_recovered_peer_run(pair, act_req["letter_id"], resume_start_state="running")
        before = len(seen)
        with acting_as(pair.root_a):
            stopped = json.loads(delegate_tools._delegate_cancel(sender, act_req["letter_id"], "stop active").text)
        assert stopped["status"] == custody.CANCEL_CONFIRMED and stopped["state"] == "succeeded", stopped
        assert not any(path.endswith("/result") for _m, path in seen[before:]), seen[before:]
        with acting_as(pair.root_a):
            # Local collection WITHOUT gateway stages honest withheld disclosure (NOT complete empty ordinary result!)
            collection.collect_terminal(pair.root_a, act_req["letter_id"], staging_task_id="t-sender", output_retry=True)
            mid = current.run(pair.root_a, act_req["letter_id"])
            assert mid.output_complete is False
            mid_payload = collection.retained_payload(pair.root_a, mid, require_complete=False)
            assert mid_payload["content_withheld"] == "stop" and mid_payload["primary_output"] is None
            # Ordinary delegate_wait while Active upgrades terminal_source and fetches real /result + cost!
            waited = json.loads(delegate_tools._read_run(sender, act_req["letter_id"], 0))
        assert waited["status"] == "terminal" and waited["state"] == "succeeded", waited
        assert waited["primary_output"]["text"] == "42 widgets."
        assert waited["output_delivery"]["complete"] is True and "content_withheld" not in waited
        assert waited["cost"]["cost_usd"] == 0.12 and waited["cost"]["cost_final"] is True

        # 2. Paused Stop on already-done letter: ZERO /result calls; delegate_wait while Paused discloses withholding;
        #    unpausing then upgrades cleanly on delegate_wait!
        _seed_recovered_peer_run(pair, pau_req["letter_id"], resume_start_state="running")
        with acting_as(pair.root_a):
            edit_place(pair.root_a, pair.place_a, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
            before_pau = len(seen)
            stopped_pau = json.loads(delegate_tools._delegate_cancel(sender, pau_req["letter_id"], "stop paused").text)
            assert stopped_pau["status"] == custody.CANCEL_CONFIRMED and stopped_pau["state"] == "succeeded"
            assert not any(path.endswith("/result") for _m, path in seen[before_pau:]), seen[before_pau:]

            before_wait_pau = len(seen)
            paused_wait = json.loads(delegate_tools._read_run(sender, pau_req["letter_id"], 0))
            assert seen[before_wait_pau:] == [], "delegate_wait while Paused must send nothing"
            assert paused_wait["status"] == "terminal" and paused_wait["state"] == "succeeded", paused_wait
            assert paused_wait["content_withheld"] == "paused" and paused_wait["content_withheld_note"]
            assert paused_wait["output_delivery"]["complete"] is False
            assert paused_wait["output_delivery"]["primary_output_full"]["status"] == "content_withheld"
            assert current.run(pair.root_a, pau_req["letter_id"]).output_complete is False

            # Restore Active -> subsequent delegate_wait legitimately collects /result!
            edit_place(pair.root_a, pair.place_a, "set_state", author=AUTHOR_OWNER, state=STATE_ACTIVE)
            collected = json.loads(delegate_tools._read_run(sender, pau_req["letter_id"], 0))
        assert collected["status"] == "terminal" and collected["primary_output"]["text"] == "42 widgets."
        assert collected["output_delivery"]["complete"] is True and "content_withheld" not in collected


def test_mailbox_failure_distinguishes_reserved_from_confirmed_and_delegate_answer_completes_only_original(
    tmp_path, monkeypatch
):
    """Mailbox write failure keeps question 'reserved' (NOT 'answered'), receipt delivered=False, GET target with
    delivery_state='reserved' and GET reply 404; model-facing delegate_answer with reply_to reports unconfirmed,
    refuses different bytes before POST, enforces Pause/budget, and completes ONLY the original pinned answer."""
    from types import SimpleNamespace
    from ouroboros import delegate_message_store as msg_store, owner_mailbox, peer_letters
    from ouroboros.delegate_interactions import _delegate_answer
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, STATE_ACTIVE, STATE_PAUSED, edit_place
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import delegate as delegate_tools
    from ouroboros.tools.peer_letter import _ask_sender
    from tests._peer_letters_shared import introduce, receiver_client, safety_reads

    pair = introduce(tmp_path, monkeypatch)
    safety_reads(monkeypatch)
    real_write, fail_mailbox = owner_mailbox.write_task_message, {"on": False}
    monkeypatch.setattr(owner_mailbox, "write_task_message",
                        lambda *a, **kw: False if fail_mailbox["on"] else real_write(*a, **kw))

    with receiver_client(pair, monkeypatch) as client:
        posts: list[bytes] = []
        real_req = client.request

        def spy_req(method, url, **kw):
            if method == "POST" and str(url) == "/api/letters":
                posts.append(bytes(kw.get("content") or b""))
            return real_req(method, url, **kw)

        monkeypatch.setattr(client, "request", spy_req)
        _bind_pair_sender_to_receiver(pair, client, monkeypatch)
        sender = _pair_sender_ctx(pair)

        with acting_as(pair.root_a):
            write_task_result(pair.root_a, "t-sender", "running", root_task_id="t-sender", task_attempt=1)
            started = json.loads(peer_delegate._delegate_to_ouroboros(
                sender, place_id=pair.place_a, text="Count files in synthetic dir.", cost_ceiling_usd=1.0,
            ).text)
        run_id, b_task_id = started["run_id"], pair.task_id(started["run_id"])
        with acting_as(pair.root_b):
            write_task_result(pair.root_b, b_task_id, "running")
            b_ctx = SimpleNamespace(task_id=b_task_id, task_attempt=1, task_metadata=pair.row(run_id)["metadata"],
                                    drive_root=pair.root_b, budget_drive_root=pair.root_b)
            assert _ask_sender(b_ctx, question="Which synthetic subfolder?").startswith("OK: question ")

        with acting_as(pair.root_a):
            waited = json.loads(delegate_tools._read_run(sender, run_id, 0))
        q_row = waited["pending_interactions"][0]
        qid, reply_to = q_row["interaction_id"], q_row["reply_to"]
        orig_answers = [{"question_id": qid, "free_text": "/synthetic/alpha"}]
        other_answers = [{"question_id": qid, "free_text": "/synthetic/beta"}]

        fail_mailbox["on"] = True
        with acting_as(pair.root_a):
            first = json.loads(_delegate_answer(sender, run_id, qid, orig_answers, reply_to=reply_to).text)
        assert (first["status"], first["confirmed"], first["delivery_state"], first["still_pending"]) == (
            "delivery_unknown", False, "reserved_unconfirmed", True), first
        reply_id = first["pinned_letter_id"]
        assert pair.row(run_id)["peer_questions"][qid]["state"] == peer_letters.QUESTION_RESERVED
        with acting_as(pair.root_b):
            assert peer_links.reply_receipt(pair.root_b, pair.link_id, reply_id)["delivered"] is False
        assert client.get(f"/api/letters/{reply_id}", headers=pair.headers()).status_code == 404
        target_status = client.get(f"/api/letters/{run_id}", headers=pair.headers()).json()
        assert target_status["state"] == "needs_input" and target_status["question"]["delivery_state"] == "reserved"

        posts_before = len(posts)
        with acting_as(pair.root_a):
            conflict = json.loads(_delegate_answer(sender, run_id, qid, other_answers, reply_to=reply_to).text)
        assert conflict["status"] == "refused" and conflict["reason"] == "peer_answer_conflict"
        assert len(posts) == posts_before

        fail_mailbox["on"] = False
        with acting_as(pair.root_b):
            edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
        with acting_as(pair.root_a):
            paused_try = json.loads(_delegate_answer(sender, run_id, qid, orig_answers, reply_to=reply_to).text)
        assert paused_try["status"] == "refused" and paused_try["reason"] == "link_paused"
        with acting_as(pair.root_b):
            assert [e for e in owner_mailbox.drain_owner_entries(pair.root_b, b_task_id, set(), 1)
                    if (e.get("place_arrival") or {}).get("peer_reply")] == []
            edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_ACTIVE)
            write_task_result(pair.root_b, b_task_id, "running", accounted_upper_bound_usd_with_children=1.0)
        with acting_as(pair.root_a):
            capped = json.loads(_delegate_answer(sender, run_id, qid, orig_answers, reply_to=reply_to).text)
        assert capped["status"] == "no_budget" and capped["accepted"] is False
        with acting_as(pair.root_b):
            write_task_result(pair.root_b, b_task_id, "running", accounted_upper_bound_usd_with_children=0.25)

        with acting_as(pair.root_a):
            done = json.loads(_delegate_answer(sender, run_id, qid, orig_answers, reply_to=reply_to).text)
        assert done["status"] == "delivered" and done["confirmed"] is True and done["delivery_state"] == "confirmed"
        assert done["receipt"]["letter_id"] == reply_id and done["receipt"]["msg_id"]
        assert pair.row(run_id)["peer_questions"][qid]["state"] == peer_letters.QUESTION_ANSWERED
        with acting_as(pair.root_b):
            replies = [e for e in owner_mailbox.drain_owner_entries(pair.root_b, b_task_id, set(), 1)
                       if (e.get("place_arrival") or {}).get("peer_reply")]
            assert len(replies) == 1 and "> /synthetic/alpha" in replies[0]["text"]
        with acting_as(pair.root_a):
            pin = msg_store.read_peer_answer_pin(sender, current.run(pair.root_a, run_id), qid)
        assert pin["delivery_state"] == "confirmed" and pin["msg_id"] == done["receipt"]["msg_id"]
        assert "/synthetic/alpha" not in json.dumps(pin)


def test_lost_http_receipt_verifies_only_pinned_answer_with_strict_single_question_address(tmp_path, monkeypatch):
    """Lost HTTP 200 recovery verifies ONLY exact single-question reply_to and exact pinned bytes; extra
    question_ids or any live pending row address mismatch is NEVER bypassed by a pin."""
    from ouroboros import delegate_interactions, owner_mailbox, peer_letters
    from ouroboros.delegate_interactions import _delegate_answer
    from ouroboros.install_identity import acting_as
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import delegate as delegate_tools
    from tests._peer_letters_shared import introduce, published, receiver_client, safety_reads

    pair = introduce(tmp_path, monkeypatch)
    safety_reads(monkeypatch)
    drop_answer_200 = {"on": False}

    with receiver_client(pair, monkeypatch) as client:
        real_req = client.request

        def drop_once(method, url, **kw):
            resp = real_req(method, url, **kw)
            if (drop_answer_200["on"] and method == "POST" and str(url) == "/api/letters"
                    and b"in_reply_to" in bytes(kw.get("content") or b"")):
                drop_answer_200["on"] = False
                raise httpx.ReadTimeout("lost 200 receipt after receiver delivered")
            return resp

        monkeypatch.setattr(client, "request", drop_once)
        _bind_pair_sender_to_receiver(pair, client, monkeypatch)
        sender = _pair_sender_ctx(pair)

        with acting_as(pair.root_a):
            write_task_result(pair.root_a, "t-sender", "running", root_task_id="t-sender", task_attempt=1)
            run_id = json.loads(peer_delegate._delegate_to_ouroboros(
                sender, place_id=pair.place_a, text="Verify lost receipt.", cost_ceiling_usd=1.0,
            ).text)["run_id"]
        b_task_id = pair.task_id(run_id)
        with acting_as(pair.root_b):
            write_task_result(pair.root_b, b_task_id, "running")
            qid = peer_letters.ask(
                pair.root_b, b_task_id, "Which branch?",
                published=published(pair.place_b, "Which branch?", root=pair.root_b, link_id=pair.link_id),
            )["question_id"]
        with acting_as(pair.root_a):
            reply_to = json.loads(delegate_tools._read_run(sender, run_id, 0))["pending_interactions"][0]["reply_to"]

        orig = [{"question_id": qid, "free_text": "release-v1"}]
        other = [{"question_id": qid, "free_text": "release-v2"}]

        # Extra question_id is rejected BEFORE POST even while pending
        bad_multi = {**reply_to, "question_ids": [qid, "q-extra"]}
        with acting_as(pair.root_a):
            multi_pending = json.loads(_delegate_answer(sender, run_id, qid, orig, reply_to=bad_multi).text)
        assert multi_pending["status"] == "refused" and multi_pending["reason"] == "reply_to_mismatch", multi_pending

        drop_answer_200["on"] = True
        with acting_as(pair.root_a):
            lost = json.loads(_delegate_answer(sender, run_id, qid, orig, reply_to=reply_to).text)
        assert (lost["status"], lost["confirmed"], lost["delivery_state"], lost["still_pending"]) == (
            "delivery_unknown", False, "unconfirmed", False), lost

        # Even WITH a matching pin on disk, extra question_ids is NEVER allowed!
        with acting_as(pair.root_a):
            multi_pinned = json.loads(_delegate_answer(sender, run_id, qid, orig, reply_to=bad_multi).text)
        assert multi_pinned["status"] == "refused" and multi_pinned["reason"] == "reply_to_mismatch", multi_pinned

        # Even WITH a matching pin on disk, a LIVE row for qid whose address disagrees is NEVER bypassed!
        with acting_as(pair.root_a):
            entry = current.run(pair.root_a, run_id)
            fake_live_mismatch = type("GW", (), {"get_run": lambda self, rid, **kw: {
                "runId": rid, "lastSeq": 1, "summary": {"runId": rid, "state": "running"},
                "pendingInteractions": [{"interactionId": qid, "runId": "other-run", "delivery": "blocking",
                                         "questions": [{"id": qid, "question": "Mismatch?"}]}],
            }})()
            wire_orig = [{"questionId": qid, "selectedLabels": [], "freeText": "release-v1"}]
            _deliv, err = delegate_interactions._validate_answer_address_live(
                sender, entry, fake_live_mismatch, reply_to, qid, wire_orig, lambda: 30.0,
            )
            assert err is not None and json.loads(err.text)["reason"] == "reply_to_unresolved"

        # Different bytes with exact reply_to -> peer_answer_conflict
        with acting_as(pair.root_a):
            conflict = json.loads(_delegate_answer(sender, run_id, qid, other, reply_to=reply_to).text)
        assert conflict["status"] == "refused" and conflict["reason"] == "peer_answer_conflict"

        # Exact original pinned bytes with exact single-question reply_to -> confirmed with zero duplicate writes
        with acting_as(pair.root_a):
            verified = json.loads(_delegate_answer(sender, run_id, qid, orig, reply_to=reply_to).text)
        assert verified["status"] == "delivered" and verified["confirmed"] is True and verified["receipt"]["msg_id"]
        with acting_as(pair.root_b):
            assert len([e for e in owner_mailbox.drain_owner_entries(pair.root_b, b_task_id, set(), 1)
                        if (e.get("place_arrival") or {}).get("peer_reply")]) == 1

        # Unpinned absent question -> reply_to_unresolved
        stale_reply_to = {**reply_to, "interaction_id": "q-unpinned", "question_ids": ["q-unpinned"]}
        with acting_as(pair.root_a):
            stale = json.loads(_delegate_answer(
                sender, run_id, "q-unpinned", [{"question_id": "q-unpinned", "free_text": "x"}],
                reply_to=stale_reply_to,
            ).text)
        assert stale["status"] == "refused" and stale["reason"] == "reply_to_unresolved"


def test_delegate_answer_on_hold_does_not_claim_question_is_still_open(tmp_path, monkeypatch):
    """When delegate_answer is held before reading the receiver (e.g. Paused/Silent/unreadable Panic control),
    nothing is sent and the refusal never falsely claims that the remote question is still open."""
    from ouroboros import peer_letters, place_projection
    from ouroboros.delegate_interactions import _delegate_answer
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, STATE_ACTIVE, STATE_PAUSED, edit_place
    from ouroboros.task_results import write_task_result
    from tests._peer_letters_shared import introduce, receiver_client, safety_reads

    pair = introduce(tmp_path, monkeypatch)
    safety_reads(monkeypatch)
    with receiver_client(pair, monkeypatch) as client:
        calls: list[tuple[str, str]] = []
        real_req = client.request

        def spy_req(method, url, **kw):
            calls.append((str(method), str(url)))
            return real_req(method, url, **kw)

        monkeypatch.setattr(client, "request", spy_req)
        _bind_pair_sender_to_receiver(pair, client, monkeypatch)
        sender = _pair_sender_ctx(pair)
        with acting_as(pair.root_a):
            write_task_result(pair.root_a, "t-sender", "running", root_task_id="t-sender", task_attempt=1)
            run_id = json.loads(peer_delegate._delegate_to_ouroboros(
                sender, place_id=pair.place_a, text="Hold answer test.", cost_ceiling_usd=1.0,
            ).text)["run_id"]

        # 1. Local Pause hold -> zero wire calls, PEER_WORDS_HELD, never promises "The question is still open."
        with acting_as(pair.root_a):
            edit_place(pair.root_a, pair.place_a, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
            before = len(calls)
            held_pause = json.loads(_delegate_answer(
                sender, run_id, "q-1", [{"question_id": "q-1", "free_text": "answer"}],
            ).text)
            edit_place(pair.root_a, pair.place_a, "set_state", author=AUTHOR_OWNER, state=STATE_ACTIVE)
        assert (held_pause["status"], held_pause["reason"]) == ("refused", "answer_words_refused"), held_pause
        assert peer_letters.HELD in held_pause["detail"]
        assert "The question is still open." not in held_pause["detail"], held_pause
        assert calls[before:] == []

        # 2. Unreadable Panic control hold -> same: zero wire calls, never promises question is still open
        real_switch = place_projection.switch_state
        monkeypatch.setattr(place_projection, "switch_state", lambda _r: "unknown")
        with acting_as(pair.root_a):
            before = len(calls)
            held_unknown = json.loads(_delegate_answer(
                sender, run_id, "q-1", [{"question_id": "q-1", "free_text": "answer"}],
            ).text)
        monkeypatch.setattr(place_projection, "switch_state", real_switch)
        assert (held_unknown["status"], held_unknown["reason"]) == ("refused", "answer_words_refused"), held_unknown
        assert peer_letters.HELD in held_unknown["detail"]
        assert "The question is still open." not in held_unknown["detail"], held_unknown
        assert calls[before:] == []
