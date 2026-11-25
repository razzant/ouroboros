"""A linked Ouroboros's letter as ordinary work on the receiver: the root it becomes, its ceiling, its
refusals before any work, its identity, what its sender sees, and the question it may ask back."""
from __future__ import annotations

import pytest

from tests._peer_letters_shared import events, introduce, letter, published, receiver_client, safety_reads, task_count


def _post(client, pair, body, **headers):
    return client.post("/api/letters", json=body, headers=pair.headers(**headers))


def test_a_letter_becomes_an_agent_authored_root_in_the_link_room(tmp_path, monkeypatch):
    from ouroboros.dialogue_provenance import LINK_KEY, declared_author, recorded_author, run_origin
    from ouroboros.install_identity import acting_as
    from ouroboros.memory_view_conversations import _conversation_chat
    from ouroboros.places import place_records

    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "3")
    pair = introduce(tmp_path, monkeypatch)
    body = letter(cost_ceiling_usd=7.5, title="Files", chain=[{"install_id": "i1", "label": "Phone", "task_id": "t0"}])
    with receiver_client(pair, monkeypatch) as client:
        assert _post(client, pair, body).status_code == 200
    row = pair.row(body["letter_id"])
    metadata = row["metadata"]
    with acting_as(pair.root_b):
        place = place_records(pair.root_b).places[pair.place_b]
    room = _conversation_chat({"provider": place.provider, "account_id": place.account_id,
                               "conversation_id": place.conversation_id})
    assert row["chat_id"] == room, "the room every reader of the link's place derives from it"
    assert metadata["work_kind"] == "owner" and run_origin(row)["owner_ingress"] is False
    assert declared_author(metadata, key="letter_author") == recorded_author("Laptop", basis=LINK_KEY,
                                                                              key_id=pair.link_id)
    assert metadata["peer_letter"] == {
        "place_id": pair.place_b, "link_id": pair.link_id, "letter_id": body["letter_id"],
        "content_sha256": metadata["peer_letter"]["content_sha256"], "sender_label": "Laptop",
        "sender_install_id": metadata["peer_letter"]["sender_install_id"],
        "chain": body["chain"], "protocol": 1}
    assert metadata["root_cost_ceiling_usd"] == 3.0, "the lesser of the letter's 7.5 and the receiver's 3"
    from ouroboros.context import build_user_content
    from ouroboros.install_identity import acting_as

    with acting_as(pair.root_b):
        framed = str(build_user_content({"id": pair.task_id(body["letter_id"]), "text": body["text"],
                                         "metadata": metadata, "drive_root": pair.root_b}))
    assert framed.startswith("[OBJECTIVE_AUTHOR] The objective below is a letter from Laptop")
    assert "mine: yes" in framed and "may_start: yes" in framed, "the letter's root reads my human's two facts"


def test_a_letter_runs_while_consciousness_is_off_and_outside_its_allowance(tmp_path, monkeypatch):
    from supervisor import consciousness_carriers, queue

    pair = introduce(tmp_path, monkeypatch)
    body = letter()
    with receiver_client(pair, monkeypatch) as client:
        assert _post(client, pair, body).status_code == 200
        task = next(row for row in queue.PENDING if row["id"] == pair.task_id(body["letter_id"]))
    monkeypatch.setattr("supervisor.state.control_is", lambda _state, key, default=None: False
                        if key == "bg_consciousness_enabled" else default)
    monkeypatch.setattr("supervisor.state.load_state", lambda *args, **kwargs: {})
    monkeypatch.setattr("ouroboros.consciousness.panic_blocks_wake", lambda _root: False)
    own = {"metadata": {"work_kind": "consciousness", "initiator": "consciousness"}}
    assert consciousness_carriers.autonomous_work_enabled(own, pair.root_b) is False, "the switch is off"
    assert consciousness_carriers.autonomous_work_enabled(task, pair.root_b) is True, "the letter still starts"
    assert queue.counts_as_consciousness_root(task) is False, "no consciousness slot or daily allowance"


def test_the_letter_ceiling_binds_its_root_and_its_helpers(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ouroboros import task_pacing
    from ouroboros.install_identity import acting_as
    from ouroboros.loop_budget import _ceiling_reached
    from ouroboros.usage_accounting import usage_scope
    from ouroboros.usage_origin import task_usage_scope

    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "50")
    pair = introduce(tmp_path, monkeypatch)
    body = letter(cost_ceiling_usd=1.25)
    with receiver_client(pair, monkeypatch) as client:
        assert _post(client, pair, body).status_code == 200
    root_id, row = pair.task_id(body["letter_id"]), pair.row(body["letter_id"])
    scope = task_usage_scope({**row, "id": root_id}, drive_root=pair.root_b, root_task_id=root_id,
                             parent_task_id="", billing={})
    assert scope.root_cost_ceiling_usd == 1.25, "the queue's row carries the ceiling into accounting"
    ctx = SimpleNamespace(task_metadata=row["metadata"], task_contract={})
    with acting_as(pair.root_b), usage_scope(scope):
        ceiling = task_pacing.resolve_task_cost_ceiling(ctx, None)
    assert (ceiling.state, ceiling.ceiling_usd, ceiling.basis) == ("active", 1.25, "producer_allowance")
    authority = task_pacing.cost_stop_authority(ctx)
    assert _ceiling_reached(1.25, ceiling.ceiling_usd, authority)
    assert not _ceiling_reached(1.24, ceiling.ceiling_usd, authority)
    # A helper carries the root's number (control_scheduling) and its stop reads the root's recorded authority.
    helper = task_usage_scope({"id": "helper-1", "metadata": {"root_cost_ceiling_usd": 1.25}}, drive_root=pair.root_b,
                              root_task_id=root_id, parent_task_id=root_id, billing={})
    helper_ctx = SimpleNamespace(task_metadata={"root_cost_ceiling_usd": 1.25}, task_contract={})
    with acting_as(pair.root_b), usage_scope(helper):
        inherited = task_pacing.resolve_task_cost_ceiling(helper_ctx, None)
    assert (inherited.state, inherited.ceiling_usd, inherited.basis) == ("active", 1.25, "root_resolved_ceiling")


def test_a_letter_without_a_ceiling_gets_the_receivers_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "4")
    pair = introduce(tmp_path, monkeypatch)
    body = letter()
    with receiver_client(pair, monkeypatch) as client:
        assert _post(client, pair, body).status_code == 200
    assert pair.row(body["letter_id"])["metadata"]["root_cost_ceiling_usd"] == 4.0


def _refusal_cases(pair, client, monkeypatch):
    """Each contract refusal, produced through the real door; yields (code, response)."""
    from ouroboros.places import AUTHOR_OWNER, edit_place
    from ouroboros.install_identity import acting_as

    good = pair.headers()
    yield "link_unknown", client.post("/api/letters", json=letter(), headers={
        **good, "X-Ouroboros-Link": good["X-Ouroboros-Link"].split(".")[0] + ".wrong"})
    yield "peer_route_forbidden", client.post("/api/tasks", json={"description": "x"}, headers=good)
    yield "install_mismatch", _post(client, pair, letter(), **{"X-Ouroboros-Install": "other"})
    yield "protocol_unsupported", _post(client, pair, {**letter(), "protocol": 2})
    yield "unexpected_fields", _post(client, pair, letter(executor_ref={}))
    yield "attachments_unsupported", _post(client, pair, letter(attachments=[{"x": 1}]))
    accepted = letter()
    assert _post(client, pair, accepted).status_code == 200
    yield "letter_id_reused", _post(client, pair, {**accepted, "text": "Other words."})
    yield "question_not_found", _post(client, pair, letter(in_reply_to={"letter_id": "nobody", "question_id": "q"}))
    from ouroboros.peer_letters import ask
    with acting_as(pair.root_b):
        question = ask(pair.root_b, pair.task_id(accepted["letter_id"]), "Which folder?",
                       published=published(pair.place_b, "Which folder?"))
    reply = letter(in_reply_to={"letter_id": accepted["letter_id"], "question_id": question["question_id"]})
    assert _post(client, pair, reply).status_code == 200
    yield "question_closed", _post(client, pair, {**reply, "letter_id": "again-" + reply["letter_id"]})
    monkeypatch.setattr("supervisor.state.budget_remaining", lambda *args, **kwargs: 0.0)
    yield "no_budget", _post(client, pair, letter())
    monkeypatch.setattr("supervisor.state.budget_remaining", lambda *args, **kwargs: float("inf"))
    with acting_as(pair.root_b):
        edit_place(pair.root_b, pair.place_b, "set_peer", author=AUTHOR_OWNER, mine=True, may_start=False)
    yield "peer_may_not_start", _post(client, pair, letter())
    with acting_as(pair.root_b):
        edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state="paused")
    yield "link_paused", _post(client, pair, letter())


def test_every_refusal_comes_before_any_work_and_leaves_an_audit_row(tmp_path, monkeypatch):
    import supervisor.queue as queue

    pair = introduce(tmp_path, monkeypatch)
    with receiver_client(pair, monkeypatch) as client:
        seen = set()
        for code, response in _refusal_cases(pair, client, monkeypatch):
            before_tasks, before_pending = task_count(pair.root_b), len(queue.PENDING)
            body = response.json()
            assert 400 <= response.status_code < 500 and body["refusal_code"] == code, (code, response.text)
            assert task_count(pair.root_b) == before_tasks and len(queue.PENDING) == before_pending, code
            assert any(row["refusal_code"] == code for row in events(pair.root_b, "peer_letter_refused")), code
            seen.add(code)
    assert seen == {"link_unknown", "peer_route_forbidden", "install_mismatch", "protocol_unsupported",
                    "unexpected_fields", "attachments_unsupported", "letter_id_reused", "question_not_found",
                    "question_closed", "no_budget", "peer_may_not_start", "link_paused"}
    for row in events(pair.root_b):
        assert pair.secret_a() not in str(row), "an audit row never carries a key"


def test_a_repeated_letter_is_one_record(tmp_path, monkeypatch):
    import supervisor.queue as queue

    pair = introduce(tmp_path, monkeypatch)
    body = letter()
    with receiver_client(pair, monkeypatch) as client:
        first, second = _post(client, pair, body), _post(client, pair, body)
        assert first.status_code == second.status_code == 200
        assert [row["id"] for row in queue.PENDING].count(pair.task_id(body["letter_id"])) == 1
    assert task_count(pair.root_b) == 1


def test_the_sender_sees_state_question_and_result_only(tmp_path, monkeypatch):
    from ouroboros.install_identity import acting_as
    from ouroboros.peer_letters import ask
    from ouroboros.task_store import patch_document

    pair = introduce(tmp_path, monkeypatch)
    body = letter()
    with receiver_client(pair, monkeypatch) as client:
        assert _post(client, pair, body).status_code == 200
        url = f"/api/letters/{body['letter_id']}"
        assert set(client.get(url, headers=pair.headers()).json()) == {"letter_id", "state", "phase"}
        with acting_as(pair.root_b):
            patch_document(pair.root_b, pair.task_id(body["letter_id"]), lambda current: {
                **current, "owner_quiz": {"q": {"quiz_id": "q", "question": "Spend $40 more?", "state": "open"}}})
        waiting = client.get(url, headers=pair.headers()).json()
        assert waiting == {"letter_id": body["letter_id"], "state": "waiting_owner"}
        with acting_as(pair.root_b):
            asked = ask(pair.root_b, pair.task_id(body["letter_id"]), "Which folder?",
                        published=published(pair.place_b, "Which folder?", root=pair.root_b, link_id=pair.link_id))
        needs = client.get(url, headers=pair.headers()).json()
        assert needs == {"letter_id": body["letter_id"], "state": "needs_input",
                         "question": {"question_id": asked["question_id"], "text": "Which folder?"}}
        task_path = f"/api/tasks/{pair.task_id(body['letter_id'])}"
        for path in (task_path, f"{task_path}/events", "/api/chat/history"):
            assert client.get(path, headers=pair.headers()).status_code == 403


def test_a_stops_state_read_carries_the_state_alone_and_a_refused_read_leaves_no_row(tmp_path, monkeypatch):
    """``GET /api/letters/{id}/state`` answers a sender's Stop with the state alone, a question's words never in it,
    also while the link is paused here; a read refused at a letter route (no key, a route the link may not reach)
    leaves no audit row, while a refused write still does."""
    from ouroboros.install_identity import acting_as
    from ouroboros.peer_letters import ask
    from ouroboros.places import AUTHOR_OWNER, STATE_PAUSED, edit_place

    pair = introduce(tmp_path, monkeypatch)
    body = letter()
    with receiver_client(pair, monkeypatch) as client:
        assert _post(client, pair, body).status_code == 200
        state_url = f"/api/letters/{body['letter_id']}/state"
        with acting_as(pair.root_b):
            ask(pair.root_b, pair.task_id(body["letter_id"]), "WORDS OF THE QUESTION?",
                published=published(pair.place_b, "WORDS OF THE QUESTION?", root=pair.root_b, link_id=pair.link_id))
        read = client.get(state_url, headers=pair.headers())
        assert read.status_code == 200 and read.json() == {"letter_id": body["letter_id"], "view": "state",
                                                           "state": "needs_input"}, read.text
        with acting_as(pair.root_b):
            edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state=STATE_PAUSED)
        paused = client.get(state_url, headers=pair.headers())
        assert paused.status_code == 200 and paused.json()["state"] == "needs_input", paused.text
        assert "WORDS OF THE QUESTION" not in read.text + paused.text
        assert client.get("/api/letters/nope/state", headers=pair.headers()).status_code == 404

        rows = len(events(pair.root_b, "peer_letter_refused"))
        unkeyed = client.get(f"/api/letters/{body['letter_id']}")
        forbidden = client.get(f"/api/letters/{body['letter_id']}/events", headers=pair.headers())
        assert (unkeyed.status_code, forbidden.status_code) == (401, 403), (unkeyed.text, forbidden.text)
        assert len(events(pair.root_b, "peer_letter_refused")) == rows, "a refused read leaves no row"
        assert client.post(f"/api/letters/{body['letter_id']}/events", headers=pair.headers()).status_code == 403
        assert len(events(pair.root_b, "peer_letter_refused")) == rows + 1, "a refused write still does"


def test_ask_sender_waits_and_the_answer_wakes_the_same_root(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import supervisor.queue as queue
    from ouroboros.install_identity import acting_as
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.owner_wait import append_wake_notice, checkpoint_owner_wait, classify_wake
    from ouroboros.tools.peer_letter import _ask_sender

    pair = introduce(tmp_path, monkeypatch)
    reads = safety_reads(monkeypatch)  # the Supervisor reads the question once, as words to the link
    body = letter(cost_ceiling_usd=2.0)
    with receiver_client(pair, monkeypatch) as client:
        assert _post(client, pair, body).status_code == 200
        root_id = pair.task_id(body["letter_id"])
        row = pair.row(body["letter_id"])
        ctx = SimpleNamespace(task_id=root_id, task_attempt=1, task_metadata=row["metadata"], drive_root=pair.root_b,
                              budget_drive_root=pair.root_b)
        with acting_as(pair.root_b):
            answer = _ask_sender(ctx, question="Which folder do you mean?")
            checkpoint = checkpoint_owner_wait(ctx, [], {}, {}, 1, [], set())
        assert answer.startswith("OK: question ") and ctx._owner_wait_requested.startswith("peer-question:")
        assert reads == [("place_reply", {"text": "Which folder do you mean?", "place_id": pair.place_b})], reads
        assert (checkpoint["reason"], checkpoint["quiz_id"]) == ("owner", "")
        status = client.get(f"/api/letters/{body['letter_id']}", headers=pair.headers()).json()
        assert status["state"] == "needs_input"
        question_id = status["question"]["question_id"]
        assert checkpoint["peer_question"] == {"question_id": question_id, "sender": "Laptop"}
        tasks_before = task_count(pair.root_b)
        reply = letter(text="/tmp itself.", in_reply_to={"letter_id": body["letter_id"], "question_id": question_id})
        assert _post(client, pair, reply).status_code == 200
        assert client.get(f"/api/letters/{body['letter_id']}", headers=pair.headers()).json()["state"] == "accepted_running"
        assert task_count(pair.root_b) == tasks_before and [r["id"] for r in queue.PENDING] == [root_id], "no new root"
    with acting_as(pair.root_b):
        entries = drain_owner_entries(pair.root_b, root_id, set(), 1)
        messages: list = []
        append_wake_notice(ctx, checkpoint, "answer", messages)
    assert classify_wake(entries, "", peer_question_id=question_id) == "answer", "the answer ends the root's wait"
    assert classify_wake(entries, ctx._owner_wait_requested) == "answer"
    assert messages == [], "an answered peer question appends no false non-answer notice"
    assert "/tmp itself." in entries[0]["text"] and entries[0]["provenance"] == "place_event"
    assert entries[0]["place_arrival"]["author"]["basis"] == "link_key"
    assert pair.row(body["letter_id"])["metadata"]["root_cost_ceiling_usd"] == 2.0, "the same ceiling, not a new one"


def test_ask_sender_belongs_to_a_letter_root_only(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ouroboros.tools.peer_letter import _ask_sender
    from ouroboros.tools.registry_guards import _builtin_tool_availability

    plain = SimpleNamespace(task_id="plain-task", task_metadata={"work_kind": "owner"}, drive_root=tmp_path)
    assert _ask_sender(plain, question="?").startswith("⚠️ ASK_SENDER_UNAVAILABLE")
    assert _builtin_tool_availability("ask_sender", plain)[0] is False
    helper = SimpleNamespace(task_id="child", task_metadata={"peer_letter": {"place_id": "p", "letter_id": "l"},
                                                              "parent_task_id": "root", "delegation_role": "subagent"})
    assert _builtin_tool_availability("ask_sender", helper)[0] is False
    # Its class is what it does (words that leave for another installation), whoever is shown it: no child
    # profile and no core or read set names it, and a wake that only observes is not given it.
    from ouroboros import consciousness_authority as ca
    from ouroboros import tool_capabilities as cap

    assert "ask_sender" in cap.OBSERVE_WORLD_MUTATION_TOOLS and "ask_sender" in ca.disabled_tools_for("observe")
    assert "ask_sender" not in ca.disabled_tools_for("act")
    assert "ask_sender" not in (cap.CORE_TOOL_NAMES | cap.ACTING_SUBAGENT_TOOL_NAMES | cap.PURE_READ_TOOLS
                                | cap.LOCAL_READONLY_SUBAGENT_TOOL_NAMES | cap.INERT_READ_TOOLS)


def test_an_answer_from_another_link_finds_no_question(tmp_path, monkeypatch):
    from ouroboros.install_identity import acting_as
    from ouroboros.peer_letters import ask

    pair = introduce(tmp_path, monkeypatch)
    body = letter()
    with receiver_client(pair, monkeypatch) as client:
        assert _post(client, pair, body).status_code == 200
        with acting_as(pair.root_b):
            question = ask(pair.root_b, pair.task_id(body["letter_id"]), "Which folder?",
                           published=published(pair.place_b, "Which folder?"))
        # A second link from a third installation to the same receiver.
        from ouroboros import peer_links
        third = tmp_path / "c" / "data"
        (third / "memory").mkdir(parents=True)
        with acting_as(pair.root_b):
            invite = peer_links.create_invite(pair.root_b, own_url="http://127.0.0.1:9", may_start=True)

        def post(url, token, request_body):
            with acting_as(pair.root_b):
                return peer_links.introduce(pair.root_b, token, request_body)
        monkeypatch.setattr(peer_links, "_post_introduce", post)
        with acting_as(third):
            other = peer_links.accept_invite(third, invite["invite"], own_url="http://127.0.0.1:7", may_start=False)
            peer_links.confirm(third, other["link_id"])
        with acting_as(pair.root_b):
            peer_links.confirm(pair.root_b, other["link_id"])
        with acting_as(third):
            headers = peer_links.outgoing_link(other["place_id"], drive_root=third).auth_headers()
        reply = letter(in_reply_to={"letter_id": body["letter_id"], "question_id": question["question_id"]})
        refused = client.post("/api/letters", json=reply, headers=headers)
    assert refused.status_code == 404 and refused.json()["refusal_code"] == "question_not_found"


@pytest.mark.parametrize("field", ["title", "chain", "cost_ceiling_usd"])
def test_an_answer_carries_no_new_title_chain_or_ceiling(tmp_path, monkeypatch, field):
    pair = introduce(tmp_path, monkeypatch)
    value = {"title": "x", "chain": [], "cost_ceiling_usd": 1.0}[field]
    with receiver_client(pair, monkeypatch) as client:
        response = _post(client, pair, letter(in_reply_to={"letter_id": "a", "question_id": "q"}, **{field: value}))
    assert response.status_code == 400 and response.json()["refusal_code"] == "unexpected_fields"


def test_status_and_result_are_bound_by_their_routes_whatever_the_id_spells(tmp_path, monkeypatch):
    pair = introduce(tmp_path, monkeypatch)
    body = letter(letter_id="result")
    with receiver_client(pair, monkeypatch) as client:
        assert _post(client, pair, body).status_code == 200
        status = client.get("/api/letters/result", headers=pair.headers())
        early = client.get("/api/letters/result/result", headers=pair.headers())
        dots = _post(client, pair, letter(letter_id=".."))
    assert status.status_code == 200 and status.json() == {"letter_id": "result", "state": "accepted_running",
                                                           "phase": "queued"}
    assert early.status_code == 409 and early.json() == {"error": "not_done", "state": "accepted_running"}
    assert dots.status_code == 400 and dots.json()["refusal_code"] == "invalid_letter", "no id a client drops"


def test_every_route_checks_the_link_as_it_stands_and_a_paused_link_still_reads_its_work(tmp_path, monkeypatch):
    """The protocol, both installs and the place are checked on the four routes alike; a pause stops new letters
    only, so the sender still reads its letter and its own repeat returns the same record."""
    from ouroboros.install_identity import acting_as
    from ouroboros.places import AUTHOR_OWNER, edit_place

    pair = introduce(tmp_path, monkeypatch)
    body = letter()
    url = f"/api/letters/{body['letter_id']}"
    with receiver_client(pair, monkeypatch) as client:
        assert _post(client, pair, body).status_code == 200
        audited = len(events(pair.root_b, "peer_letter_refused"))
        for path in (url, f"{url}/result"):
            other = client.get(path, headers=pair.headers(**{"X-Ouroboros-Install": "other"}))
            old = client.get(path, headers=pair.headers(**{"X-Ouroboros-Protocol": "2"}))
            assert (other.status_code, other.json()["refusal_code"]) == (403, "install_mismatch"), path
            assert (old.status_code, old.json()["refusal_code"]) == (400, "protocol_unsupported"), path
            bare = client.get(path)  # no link key at all
            assert (bare.status_code, bare.json()["refusal_code"]) == (401, "link_unknown"), path
        assert len(events(pair.root_b, "peer_letter_refused")) == audited, "a refused read leaves no row"
        wrong = client.post(f"{url}/cancel", headers=pair.headers(**{"X-Ouroboros-Install": "other"}))
        assert wrong.json()["refusal_code"] == "install_mismatch"
        assert pair.row(body["letter_id"])["status"] != "cancelled"
        with acting_as(pair.root_b):
            edit_place(pair.root_b, pair.place_b, "set_state", author=AUTHOR_OWNER, state="paused")
        assert client.get(url, headers=pair.headers()).json()["state"] == "accepted_running"
        assert _post(client, pair, body).json() == {"letter_id": body["letter_id"], "state": "accepted_running",
                                                    "phase": "queued"}, "its own repeat reads, as GET does"
        assert _post(client, pair, letter()).json()["refusal_code"] == "link_paused"
    assert task_count(pair.root_b) == 1


def test_one_letter_id_is_one_outcome_on_its_link(tmp_path, monkeypatch):
    """An id that carried an answer never becomes a new letter or another answer, and an id that started work never
    becomes an answer; the answer's receipt is read by address, outside the registry, and leaves with the link."""
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as
    from ouroboros.peer_letters import ask

    pair = introduce(tmp_path, monkeypatch)
    first, second = letter(text="first"), letter(text="second")
    with receiver_client(pair, monkeypatch) as client:
        for body in (first, second):
            assert _post(client, pair, body).status_code == 200
        with acting_as(pair.root_b):
            asked = {body["letter_id"]: ask(pair.root_b, pair.task_id(body["letter_id"]), "Which?",
                                            published=published(pair.place_b, "Which?"))["question_id"]
                     for body in (first, second)}
        reply = letter(text="/tmp", in_reply_to={"letter_id": first["letter_id"],
                                                 "question_id": asked[first["letter_id"]]})
        assert _post(client, pair, reply).status_code == 200
        reused = [_post(client, pair, letter(letter_id=reply["letter_id"], text="new work")),
                  _post(client, pair, {**reply, "in_reply_to": {"letter_id": second["letter_id"],
                                                                "question_id": asked[second["letter_id"]]}}),
                  _post(client, pair, {**reply, "text": "/var"}),
                  _post(client, pair, {**reply, "letter_id": second["letter_id"]})]
        assert [(r.status_code, r.json()["refusal_code"]) for r in reused] == [(409, "letter_id_reused")] * 4
        assert client.get(f"/api/letters/{reply['letter_id']}", headers=pair.headers()).json()["letter_id"] == \
            reply["letter_id"], "an answer's id reads the letter it answered"
    assert pair.row(second["letter_id"])["peer_questions"][asked[second["letter_id"]]]["state"] == "open"
    with acting_as(pair.root_b):
        assert "replies" not in peer_links.read_registry(pair.root_b)
        receipt = peer_links.reply_receipt(pair.root_b, pair.link_id, reply["letter_id"])
        assert receipt["target_task_id"] == pair.task_id(first["letter_id"])
        peer_links.revoke(pair.root_b, pair.link_id)
        assert not peer_links._receipt_path(pair.root_b, pair.link_id, reply["letter_id"]).exists()


def test_the_links_cancel_is_its_own_stop_and_the_stops_owner_judges_a_settled_root(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks
    from ouroboros.install_identity import acting_as
    from ouroboros.owner_continue import OWNER_STOP_SOURCES
    from ouroboros.task_results import write_task_result
    from supervisor.cancel_publication import CANCEL_SOURCE_PHRASES

    pair = introduce(tmp_path, monkeypatch)
    running, settled = letter(), letter()
    asked = []
    real_record = tasks._record_stop
    monkeypatch.setattr(tasks, "_record_stop", lambda task_id, **kw: asked.append((task_id, kw)) or real_record(
        task_id, **kw))
    with receiver_client(pair, monkeypatch) as client:
        for body in (running, settled):
            assert _post(client, pair, body).status_code == 200
        cancelled = client.post(f"/api/letters/{running['letter_id']}/cancel", headers=pair.headers())
        assert cancelled.status_code == 200 and cancelled.json()["state"] == "cancelled", cancelled.text
        import supervisor.queue as queue
        queue.PENDING[:] = [row for row in queue.PENDING if row.get("id") != pair.task_id(settled["letter_id"])]
        with acting_as(pair.root_b):
            write_task_result(pair.root_b, pair.task_id(settled["letter_id"]), "completed", result="3 files",
                              finished_at="2026-11-24T00:00:00+00:00")
        done = client.post(f"/api/letters/{settled['letter_id']}/cancel", headers=pair.headers())
    assert done.status_code == 200 and done.json()["state"] == "done"
    assert [task_id for task_id, _kw in asked] == [pair.task_id(running["letter_id"]),
                                                   pair.task_id(settled["letter_id"])], "a settled root is asked too"
    origin = pair.row(running["letter_id"])["cancel_origin"]
    assert origin["source"] == "peer_letter_cancel" and origin["source"] not in OWNER_STOP_SOURCES
    assert origin["request_origin"] == {"kind": "linked_ouroboros", "link_id": pair.link_id, "place_id": pair.place_b}
    assert CANCEL_SOURCE_PHRASES[origin["source"]] == "Stopped by the Ouroboros that sent this letter"


def test_the_result_route_gives_the_addressed_projection_and_nothing_of_the_raw_row(tmp_path, monkeypatch):
    """``peer_letters.disclosed_letter_result`` is the one external-serving projection of a done letter: the route
    passes explicit ``root`` and ``link_id``, adds its id and state, never reads ``row["result"]`` itself, and never
    infers ``result_available`` from the text (an empty kept answer)."""
    import pathlib
    import supervisor.queue as queue
    from ouroboros import peer_letters
    from ouroboros.install_identity import acting_as
    from ouroboros.task_results import write_task_result

    pair = introduce(tmp_path, monkeypatch)
    body = letter()
    projected = {"result": "", "result_available": True, "finished_at": "2026-11-24T00:00:00+00:00",
                 "cost": {"known_usd": None, "final": False}}
    seen: list[tuple[pathlib.Path, str]] = []

    def fake_disclosed(root, row, *, link_id):
        seen.append((pathlib.Path(root), link_id))
        return dict(projected)

    monkeypatch.setattr(peer_letters, "disclosed_letter_result", fake_disclosed)
    with receiver_client(pair, monkeypatch) as client:
        assert _post(client, pair, body).status_code == 200
        queue.PENDING[:] = [row for row in queue.PENDING if row.get("id") != pair.task_id(body["letter_id"])]
        with acting_as(pair.root_b):
            write_task_result(pair.root_b, pair.task_id(body["letter_id"]), "completed", result="local only words",
                              finished_at="2026-11-24T00:00:00+00:00")
        done = client.get(f"/api/letters/{body['letter_id']}/result", headers=pair.headers())
    assert done.status_code == 200 and done.json() == {"letter_id": body["letter_id"], "state": "done", **projected}
    assert seen == [(pair.root_b, pair.link_id)]


def test_an_answer_whose_receipt_cannot_be_written_is_a_typed_503_and_moves_nothing(tmp_path, monkeypatch):
    """The receipt is the first effect of an answer: a disk that refuses it answers 503 (ask again), the question
    stays open and nothing reaches the asking root; the same answer later lands once."""
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.peer_letters import ask

    def replies():
        with acting_as(pair.root_b):
            entries = drain_owner_entries(pair.root_b, pair.task_id(body["letter_id"]), set(), 1)
        return [entry for entry in entries if (entry.get("place_arrival") or {}).get("peer_reply")]

    pair = introduce(tmp_path, monkeypatch)
    body = letter()
    real_write = peer_links.atomic_write_json
    with receiver_client(pair, monkeypatch) as client:
        assert _post(client, pair, body).status_code == 200
        with acting_as(pair.root_b):
            question_id = ask(pair.root_b, pair.task_id(body["letter_id"]), "Which folder?",
                              published=published(pair.place_b, "Which folder?"))["question_id"]
        reply = letter(text="/tmp", in_reply_to={"letter_id": body["letter_id"], "question_id": question_id})

        def refusing(*args, **kwargs):  # the disk refuses the receipt, for now
            raise OSError(28, "No space left on device (fictional)")

        monkeypatch.setattr(peer_links, "atomic_write_json", refusing)
        refused = _post(client, pair, reply)
        assert refused.status_code == 503 and refused.json()["refusal_code"] == "peer_links_unreadable", refused.text
        assert pair.row(body["letter_id"])["peer_questions"][question_id]["state"] == "open"
        assert replies() == [] and peer_links.reply_receipt(pair.root_b, pair.link_id, reply["letter_id"]) is None
        monkeypatch.setattr(peer_links, "atomic_write_json", real_write)
        assert _post(client, pair, reply).status_code == 200
    assert len(replies()) == 1 and pair.row(body["letter_id"])["peer_questions"][question_id]["state"] == "answered"
