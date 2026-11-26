"""Defensive properties of the checked, conversation-addressed cost-cap tool.

One isolated ledger, one exact work and one host-selected decision. The binding step resolves them before Safety
reads the call; the checker sees that subject as the call's target and the cited owner row in its ordinary corpus,
marked once with the provenance and fields every other check shows; the writer records the same basis. The fixture
replaces the physical model call; no provider or live installation participates.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import usage_store
from ouroboros.safety_evidence import CARD_FORM_MAIL, OPENING_CARD_FORM, SOURCE_AUTHORS
from ouroboros.task_results import STATUS_INTERRUPTED, STATUS_REQUESTED, write_task_result
from ouroboros.tool_call_log import invocation_scope
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.usage_admission import work_cap
from tests._safety_fixtures import supervisor as supervisor_fixture
from tests._shared import write_queue_document
from tests.test_living_cards_api import decide as decide_fixture
from tests.test_work_cost_cap import WORDS, _binding, _decision_context, _scope, _spend, _work
from tests.test_work_cost_cap import data_root as data_root_fixture, listed_rooms as listed_rooms_fixture

# The fixtures these tests take by name.
supervisor, decide, data_root, listed_rooms = supervisor_fixture, decide_fixture, data_root_fixture, listed_rooms_fixture


def _tool(ctx, target="work-a", limit=15, **extra):
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    with invocation_scope({"tool_call_id": "choose-cap"}):
        return registry.execute_result("set_task_cost_cap", dict(task_id=target, limit_usd=limit,
                                                                 reason="The owner chose this total", **extra))


def _stored(root):
    with usage_store.read(root) as view:
        return work_cap(view, "work-a")


def _instruction_read(result, error="cost_cap_target_instructions_unseen"):
    body = json.loads(result.text)
    assert result.status == "error" and body["error"] == error, result.text
    assert "If delivery omitted ranges, read them at the exact source named in the result view" in body["note"]
    assert "grants no permission" in body["note"]
    return body["instructions_sha256"], body["target_work_owner_instructions"]


def _bundle(prompt):
    """The host evidence of one Safety reading, as the checker reads it."""
    body = prompt.partition("Host evidence (provenance is not consent):\n")[2].partition("\nProposed tool call:")[0]
    return json.loads(body)


def _cited(prompt):
    return [row for row in _bundle(prompt)["occasion"]["sources"] if row.get("cited_by_this_call")]


def _owner_message(root, monkeypatch, words, client):
    """One more accepted owner message in Main, beside ``_decision_context``'s."""
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    bridge = message_bus.LocalChatBridge({})
    bridge.ui_send(words, client_message_id=client, chat_id=1)
    [update] = bridge.get_updates(offset=0, timeout=0)
    return update["message"]["accepted_source_ref"]


@pytest.mark.parametrize("status", [STATUS_REQUESTED, STATUS_INTERRUPTED])
def test_bound_main_addresses_headless_work(data_root, supervisor, monkeypatch, status):
    from tests.test_launch_roots import _write_witness

    root = data_root
    write_task_result(root, "work-a", status, root_task_id="work-a", chat_id=0, billing_group=_binding("work-a"))
    _spend(_scope(root, "work-a", "work-a", "work-a"), 2)
    write_queue_document(root / "state/queue_snapshot.json", {
        "pending": [{"id": "work-a", "task": {"id": "work-a", "chat_id": 0}}], "running": [],
    })
    write_queue_document(root / "state/queue_dispatch.json", {"budget_root_fences": [], "members": {}})
    _write_witness(root, initialization_id="test-init")
    (root / "state/state.json").write_text(
        json.dumps({"initialization_id": "test-init", "owner_external_chat_id": 43}), encoding="utf-8",
    )
    ctx = _decision_context(root, root, monkeypatch, chat_id=43)

    result = _tool(ctx)
    assert json.loads(result.text)["ok"], result.text
    prompt = supervisor.calls[0]["user_prompt"]
    assert _bundle(prompt)["subject"]["physical_target"]["chat_id"] == 0
    # No retained directives here: the cited row still reaches the checker beside the quoted input, readable, once.
    [cited] = _cited(prompt)
    assert cited["content"] == WORDS and "carried" not in cited and prompt.count(WORDS) == 1
    assert _stored(root)["basis"]["source"] == ctx.task_metadata["origin_message_ref"]


def test_moved_work_keeps_its_exact_opening_decision(data_root, listed_rooms, supervisor, monkeypatch):
    from ouroboros.projects_registry import bind_task_to_project

    root = data_root
    room = listed_rooms["work-a"]
    ctx = _decision_context(root, root, monkeypatch, task_id="work-a")
    ref = ctx.task_metadata["origin_message_ref"]
    write_task_result(root, "work-a", "running", root_task_id="work-a", chat_id=1, billing_group=_binding("work-a"))
    _spend(_scope(root, "work-a", "work-a", "work-a"), 2)
    bind_task_to_project(root, "work-a", room["id"], room["chat_id"], origin={"absent": "system"})

    result = _tool(ctx)
    assert json.loads(result.text)["ok"], result.text
    assert _bundle(supervisor.calls[0]["user_prompt"])["subject"]["physical_target"]["chat_id"] == room["chat_id"]
    assert _stored(root)["basis"]["source"] == ref


@pytest.mark.parametrize("mode", ["low", "max", "nano"])
def test_cost_cap_is_discoverable_and_callable(data_root, listed_rooms, supervisor, monkeypatch, mode):
    from ouroboros.loop import _setup_dynamic_tools
    from ouroboros.tool_policy import initial_tool_schemas

    _work(data_root, "work-a", chat_id=listed_rooms["work-a"]["chat_id"])
    ctx = _decision_context(data_root, data_root, monkeypatch)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    name = "set_task_cost_cap"
    schemas = initial_tool_schemas(registry, context_mode=mode)
    _setup_dynamic_tools(registry, schemas, [])
    catalog = registry.execute("list_available_tools", {"namespace": "builtin"})
    if mode == "nano":
        assert f"- {name} [not loaded]" in catalog
        before = list(schemas)
        assert "Call schemas loaded" in registry.execute("enable_tools", {"tools": name})
        assert schemas == [*before, registry.get_schema_by_name(name)]
    else:
        assert f"- {name} [loaded]" in catalog
        assert schemas == registry.schemas()
    parameters = registry.get_schema_by_name(name)["function"]["parameters"]
    assert parameters["properties"]["instructions_sha256"]["type"] == "string"
    assert "instructions_sha256" not in parameters["required"]
    with invocation_scope({"tool_call_id": "choose-cap"}):
        result = registry.execute_result(name, {"task_id": "work-a", "limit_usd": 15, "reason": "Owner's choice"})
    assert json.loads(result.text)["ok"], result.text
    assert _stored(data_root)["limit_usd"] == 15


def test_old_observe_metadata_keeps_cap_read_only(data_root, listed_rooms, supervisor, monkeypatch):
    from ouroboros.consciousness_authority import CONSCIOUSNESS_INITIATOR, LEVEL_OBSERVE

    root = data_root
    _work(root, "work-a", chat_id=listed_rooms["work-a"]["chat_id"])
    _spend(_scope(root, "work-a", "work-a", "work-a"), 2)
    ctx = _decision_context(root, root, monkeypatch)
    ctx.task_metadata.update(initiator=CONSCIOUSNESS_INITIATOR, consciousness_autonomy=LEVEL_OBSERVE,
                             disabled_tools=["start_task"])

    result = _tool(ctx)
    assert json.loads(result.text)["error"] == "sender_tool_withheld"
    assert _stored(root) is None and not supervisor.calls


@pytest.mark.parametrize("addressed", [True, False], ids=["host-addressed-reply", "context-only-relay"])
def test_only_addressed_relay_has_authority_in_the_recipient(
        data_root, listed_rooms, supervisor, monkeypatch, addressed):
    """The relay's row is cited where it already stands in the checker's corpus, once, and every row there keeps the
    carried mark the ordinary corpus gives it: the words the work started with and a context-only relay stay carried."""
    from ouroboros.owner_relay import inject_relay, owner_quote, owner_row_address, write_relay
    from ouroboros.project_dialogue import resolve_owner_message_source

    root = data_root
    room = listed_rooms["work-a"]
    _work(root, "work-a", chat_id=listed_rooms["work-a"]["chat_id"])
    _spend(_scope(root, "work-a", "work-a", "work-a"), 2)
    origin = _decision_context(root, root, monkeypatch).task_metadata["origin_message_ref"]
    row = resolve_owner_message_source(root, origin)
    aside = resolve_owner_message_source(root, _owner_message(root, monkeypatch, "Other plans for later", "owner-aside"))
    started = {"text": "Finish the migration this week", "source": "origin_message", "carrier": "started from Main",
               "ts": "2026-11-20T10:00:00+00:00", "ref": "chat 1 / earlier-start"}
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id="work-a", current_chat_id=room["chat_id"],
                      task_metadata={"governing_owner_words": [started]})
    for relay_id, quoted, reply in (("owner-aside-to-work", aside, False), ("owner-input-to-work", row, addressed)):
        entry, fresh = write_relay(root, "work-a", relay_id=relay_id, note="", quotes=[owner_quote(root, quoted)],
                                   chat_id=1, teller_task_id="", work_reply=reply, canonical_root=root)
        assert fresh
        inject_relay(ctx, entry, [])

    result = _tool(ctx, owner_source=owner_row_address(row))
    body = json.loads(result.text)
    if addressed:
        from ouroboros.owner_words import safety_sources

        assert body["ok"], result.text
        assert _stored(root)["basis"]["source"] == origin
        [prompt] = [call["user_prompt"] for call in supervisor.calls]
        shown, normal = _bundle(prompt)["occasion"]["sources"], safety_sources(ctx)  # what every other check shows
        assert [(row["content"], row.get("carried")) for row in shown] == [
            (row["content"], row.get("carried")) for row in normal]
        assert {("Other plans for later", True), (started["text"], True)} <= {
            (row["content"], row.get("carried")) for row in shown}
        [cited] = _cited(prompt)
        assert cited["content"] == WORDS and prompt.count(WORDS) == 1
        assert cited["author"] == SOURCE_AUTHORS["owner_mailbox"]
    else:
        assert body["error"] == "cost_cap_basis_unverified"
        assert _stored(root) is None and not supervisor.calls


def test_press_has_asking_task_and_one_option_label(data_root, listed_rooms, supervisor, monkeypatch):
    from ouroboros.owner_quiz import reconcile_terminal, record_asked
    from tests.test_quiz_late_answer import _decision_app, _inbox, _late_bridge, _post

    root = data_root
    label, asker = "Set the work's total to fifteen dollars", "main-asked"
    record_asked(root, asker, quiz_id="choice", question="What cap should work-a have?", options=[label],
                 assumption="Leave it unchanged unless chosen", chat_id=1)
    reconcile_terminal(root, asker)
    bridge, _ = _late_bridge(root, monkeypatch)
    response = _post(_decision_app(root, monkeypatch, live_task=None),
                     {"request_id": "choose", "decision_id": f"quiz:{asker}:choice", "option_index": 0})
    assert response.status_code == 200 and response.json()["forwarded"], response.text
    [arrival] = _inbox(bridge)
    ref = arrival["accepted_source_ref"]
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id="main-a", current_chat_id=1, is_direct_chat=True,
                      task_metadata={"work_kind": "owner", "origin_message_ref": ref})
    ctx._owner_directives = [{"source": "owner_press", "row_sha256": ref["row_sha256"], "content": arrival["text"]}]
    _work(root, "work-a", chat_id=listed_rooms["work-a"]["chat_id"])
    _spend(_scope(root, "work-a", "work-a", "work-a"), 2)

    result = _tool(ctx)
    assert json.loads(result.text)["ok"], result.text
    prompt = supervisor.calls[0]["user_prompt"]
    assert prompt.count(label) == 1
    [cited] = _cited(prompt)
    assert cited["source"] == "owner_press" and "carried" not in cited
    assert cited["author"] == SOURCE_AUTHORS["owner_press"]
    basis = _stored(root)["basis"]
    assert basis["decision_kind"] == "owner_press" and basis["source"] == ref
    assert basis["card"] == {"address": f"quiz choice of task {asker}", "author": f"Ouroboros, task {asker}",
                             "option_index": 0}


@pytest.mark.parametrize("addressed", [False, True], ids=["opening", "addressed-reply"])
def test_a_form_answer_keeps_its_labeled_fields_for_the_checker(
        data_root, listed_rooms, supervisor, decide, monkeypatch, addressed):
    """An ordinary living form keeps the field values distinct from the card author's text."""
    from ouroboros.card_store import write_card
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.owner_relay import inject_relay, owner_quote, owner_row_address, write_relay
    from ouroboros.project_dialogue import resolve_owner_message_source

    import supervisor.message_bus as mb
    import supervisor.queue as q
    from ouroboros.gateway import task_decision as td

    root = data_root
    monkeypatch.setattr(mb, "DATA_DIR", root)  # the decision app answers in this test's data root
    monkeypatch.setattr(td, "request_drive_root", lambda request: root)
    monkeypatch.setattr(q, "_task_drive_for_task", lambda task, task_id: root, raising=False)
    form = {"type": "form", "title": "Raise a cap", "fields": [{"name": "work"}, {"name": "total"}],
            "submit": {"id": "raise", "once": True}}
    card = write_card(root, chat_id=1, task_id="asker", title="Budget", fallback_text="Budget",
                      kind="components", body={"components": [form], "data": {"rows": []}})
    decide.state["live"] = False
    values = {"work": "work-a", "total": "15"}
    response = decide({"request_id": "r1", "revision": 1, "decision_id": f"card:{card['card_id']}:raise",
                       "values": values})
    assert response.json()["delivery"] == "forwarded", response.text
    queued = decide.bridge._inbox.get_nowait()
    metadata = {**queued["task_metadata"], "work_kind": "owner", "origin_message_ref": queued["accepted_source_ref"],
                "origin_message_text": queued["text"]}
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id="main-a", current_chat_id=1, is_direct_chat=True,
                      task_metadata=metadata)
    _work(root, "work-a", chat_id=listed_rooms["work-a"]["chat_id"])
    _spend(_scope(root, "work-a", "work-a", "work-a"), 2)
    if addressed:
        ctx = ToolContext(repo_dir=root, drive_root=root, task_id="work-a",
                          current_chat_id=listed_rooms["work-a"]["chat_id"])
        row = resolve_owner_message_source(root, queued["accepted_source_ref"])
        entry, fresh = write_relay(root, "work-a", relay_id="form-to-work", note="", quotes=[owner_quote(root, row)],
                                   chat_id=1, teller_task_id="", work_reply=True, canonical_root=root)
        assert fresh
        inject_relay(ctx, entry, [])
        result = _tool(ctx, owner_source=owner_row_address(row))
    else:
        _initialize_owner_directives(ctx, [{"role": "user", "content": queued["text"]}])
        result = _tool(ctx)
    assert json.loads(result.text)["ok"], result.text
    [shown] = [row for row in _bundle(supervisor.calls[0]["user_prompt"])["occasion"]["sources"]
               if row["content"] == "work-a\n15"]
    assert shown["source"] == "owner_mailbox" and shown["reply_to"]["submitted_values"] == values
    assert shown["cited_by_this_call"] is True
    assert shown["author"] == (CARD_FORM_MAIL if addressed else OPENING_CARD_FORM)
    basis = _stored(root)["basis"]
    assert basis["decision_kind"] == "owner_words" and basis["card"]["address"] == f"card {card['card_id']} of task asker"


def test_an_owner_maximum_rides_only_on_their_own_cited_words(data_root, listed_rooms, supervisor, monkeypatch):
    """``delegated_max_usd``: a larger finite total from the same decision, carried in its basis; never Ouroboros's own."""
    root = data_root
    _work(root, "work-a", chat_id=listed_rooms["work-a"]["chat_id"])
    _spend(_scope(root, "work-a", "work-a", "work-a"), 2)
    words = WORDS + ", and up to $40 if it still needs more"
    ctx = _decision_context(root, root, monkeypatch, words=words)
    monkeypatch.setattr("ouroboros.tools.control_models.cyber_pro", lambda _ctx: True)

    assert json.loads(_tool(ctx, delegated_max_usd=12).text)["error"] == "invalid_delegated_max"
    assert json.loads(_tool(ctx, delegated_max_usd=15).text)["error"] == "invalid_delegated_max"
    assert json.loads(_tool(ctx, limit=0).text)["error"] == "invalid_limit"
    assert json.loads(_tool(ctx, delegated_max_usd=float("inf")).text)["error"] == "invalid_delegated_max"
    refused = _tool(ctx, owner_source="own_judgement", delegated_max_usd=40)
    assert json.loads(refused.text)["error"] == "delegated_max_needs_owner_words"
    assert _stored(root) is None and not supervisor.calls

    result = _tool(ctx, delegated_max_usd=40)
    assert json.loads(result.text)["ok"], result.text
    target = _bundle(supervisor.calls[0]["user_prompt"])["subject"]["physical_target"]
    assert target["basis"]["delegated_max_usd"] == 40.0 and target["task_id"] == "work-a"
    assert _stored(root)["basis"] == target["basis"]


def test_cyber_pro_raises_on_its_own_judgement_through_the_same_check(data_root, listed_rooms, supervisor, monkeypatch):
    root = data_root
    _work(root, "work-a", chat_id=listed_rooms["work-a"]["chat_id"])
    _spend(_scope(root, "work-a", "work-a", "work-a"), 2)
    ctx = _decision_context(root, root, monkeypatch)
    ctx._owner_directives = []
    monkeypatch.setattr("ouroboros.tools.control_models.cyber_pro", lambda _ctx: True)

    result = _tool(ctx, owner_source="own_judgement")
    assert json.loads(result.text)["ok"], result.text
    prompt = supervisor.calls[0]["user_prompt"]
    assert _bundle(prompt)["subject"]["physical_target"]["basis"] == {"kind": "own_judgement"} and not _cited(prompt)
    assert _stored(root)["basis"] == {"kind": "own_judgement"}


def _recipient(root, room):
    drive = root / "test-recipient"
    drive.mkdir()
    task = {"id": "work-a", "root_task_id": "work-a", "chat_id": room, "child_drive_root": str(drive)}
    write_task_result(root, "work-a", "running", **{k: v for k, v in task.items() if k != "id"},
                      billing_group=_binding("work-a"))
    _spend(_scope(root, "work-a", "work-a", "work-a"), 2)
    return drive, task


def _owner_work_reply(root, task, words, client):
    from ouroboros.owner_relay import owner_row_source
    from ouroboros.work_reply import route_work_reply
    from supervisor.message_ingress import accept_chat

    reply = {"address": "task:work-a", "work_task_id": "work-a"}
    row = accept_chat(1, 1, words, source="web", client_message_id=client, drive_root=root,
                      message_meta={"reply_to": reply}).row
    ref, _ = owner_row_source(row)
    host = SimpleNamespace(DRIVE_ROOT=root, RUNNING={"work-a": {"task": task, "attempt": 1}}, PENDING=[])
    delivered = route_work_reply(host, reply, ref, row)
    assert (delivered.outcome, delivered.target) == ("delivered", "work-a")
    return row


@pytest.mark.parametrize("basis", ["owner_words", "own_judgement"])
@pytest.mark.parametrize("arrives_during", ["direct", "reply"])
def test_checked_binding_observes_real_recipient_mail_and_rechecks_it(
        data_root, listed_rooms, supervisor, monkeypatch, basis, arrives_during):
    from ouroboros.owner_mailbox import write_owner_message

    root = data_root
    drive, task = _recipient(root, listed_rooms["work-a"]["chat_id"])
    ctx = _decision_context(root, root, monkeypatch)
    monkeypatch.setattr("ouroboros.tools.control_models.cyber_pro", lambda _ctx: True)
    first, second = "Keep the final comparison in one folder.", "Keep the optional chart separate."
    assert write_owner_message(drive, first, "work-a", msg_id="owner-direct")
    _owner_work_reply(root, task, second, "owner-reply")
    owner_source = "own_judgement" if basis == "own_judgement" else ""
    seen, _ = _instruction_read(_tool(ctx, owner_source=owner_source))
    assert _stored(root) is None and not supervisor.calls

    check = supervisor.module._safety_model_call

    def new_owner_instruction(**kwargs):
        if arrives_during == "direct":
            assert write_owner_message(drive, "Use the attached plan instead.", "work-a", msg_id="late-owner")
        else:
            _owner_work_reply(root, task, "Start the final comparison tomorrow.", "late-reply")
        return check(**kwargs)

    monkeypatch.setattr(supervisor.module, "_safety_model_call", new_owner_instruction)
    retry, [updated] = _instruction_read(_tool(ctx, owner_source=owner_source, instructions_sha256=seen),
                                         "cost_cap_target_instructions_changed")
    assert retry != seen and len(updated["rows"]) == 3
    [call] = supervisor.calls
    [work] = _bundle(call["user_prompt"])["subject"]["physical_target"]["target_work_owner_instructions"]
    assert work["task_id"] == "work-a" and {row["text"] for row in work["rows"]} == {first, second}
    assert _stored(root) is None


@pytest.mark.parametrize("already_delivered", [True, False], ids=["one-call", "read-before-writing"])
def test_owner_directions_are_seen_before_deciding_with_safety_off(
        data_root, listed_rooms, supervisor, monkeypatch, already_delivered):
    from queue import Queue

    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import write_owner_message

    root = data_root
    room = listed_rooms["work-a"]
    drive, _ = _recipient(root, room["chat_id"])
    ctx = _decision_context(root, root, monkeypatch, task_id="work-a", chat_id=room["chat_id"], project_id=room["id"])
    supervisor.modes["owner"] = "off"
    first, newer = "Use the print edition.", "Use the accessible edition instead."
    assert write_owner_message(drive, first, "work-a", msg_id="owner-format")
    if already_delivered:
        messages = []
        _drain_incoming_messages(messages, Queue(), drive, "work-a", None, set(), owner_ctx=ctx)
        assert any(first in str(row["content"]) for row in messages)
        result = _tool(ctx)
    else:
        read, [work] = _instruction_read(_tool(ctx))
        assert [row["text"] for row in work["rows"]] == [first]
        assert _stored(root) is None
        assert write_owner_message(drive, newer, "work-a", msg_id="owner-format-correction")
        current, [work] = _instruction_read(_tool(ctx, instructions_sha256=read))
        assert current != read and {row["text"] for row in work["rows"]} == {first, newer}
        assert _stored(root) is None
        result = _tool(ctx, instructions_sha256=current)
    assert json.loads(result.text)["ok"], result.text
    assert _stored(root)["basis"]["source"] == ctx.task_metadata["origin_message_ref"]
    assert not supervisor.calls


def test_recipient_can_cite_an_ordinary_addressed_reply(data_root, listed_rooms, supervisor, monkeypatch):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.owner_relay import inject_relay, owner_row_address, owner_row_source

    root = data_root
    room = listed_rooms["work-a"]["chat_id"]
    drive, task = _recipient(root, room)
    row = _owner_work_reply(root, task, WORDS, "chosen-reply")
    ctx = ToolContext(repo_dir=root, drive_root=drive, budget_drive_root=str(root), task_id="work-a",
                      current_chat_id=room)
    [arrival] = drain_owner_entries(drive, "work-a")
    inject_relay(ctx, arrival, [])

    result = _tool(ctx, owner_source=owner_row_address(row))
    assert json.loads(result.text)["ok"], result.text
    assert _stored(root)["basis"]["source"] == owner_row_source(row)[0]
    [cited] = _cited(supervisor.calls[0]["user_prompt"])
    assert cited["content"] == WORDS


@pytest.mark.parametrize("moved", ["caller", "target"])
def test_checked_binding_rechecks_each_room(data_root, listed_rooms, supervisor, monkeypatch, moved):
    from ouroboros.projects_registry import bind_task_to_project

    root = data_root
    room = listed_rooms["work-a"]["chat_id"]
    neighbor = listed_rooms["other-a"]
    _work(root, "work-a", chat_id=room)
    _spend(_scope(root, "work-a", "work-a", "work-a"), 2)
    ctx = _decision_context(root, root, monkeypatch)
    write_task_result(root, "main-a", "running", root_task_id="main-a", chat_id=1)
    check = supervisor.module._safety_model_call

    def move_during_check(**kwargs):
        task = {"caller": "main-a", "target": "work-a"}[moved]
        bind_task_to_project(root, task, neighbor["id"], neighbor["chat_id"], origin={"absent": "system"})
        return check(**kwargs)

    monkeypatch.setattr(supervisor.module, "_safety_model_call", move_during_check)
    result = _tool(ctx)
    expected = "work_out_of_scope" if moved == "caller" else "cost_cap_target_changed"
    assert json.loads(result.text)["error"] == expected, result.text
    assert len(supervisor.calls) == 1 and _stored(root) is None


def test_same_work_owner_ingress_does_not_become_main(data_root, listed_rooms, supervisor, monkeypatch):
    root = data_root
    external_room = 43  # no project or explicit Main alias is attached to this owner ingress
    ctx = _decision_context(root, root, monkeypatch, task_id="work-a", chat_id=external_room)
    ref = ctx.task_metadata["origin_message_ref"]
    write_task_result(root, "work-a", "running", root_task_id="work-a", chat_id=external_room,
                      origin_message_ref=ref, billing_group=_binding("work-a"))
    _spend(_scope(root, "work-a", "work-a", "work-a"), 2)
    _work(root, "other-a", chat_id=listed_rooms["other-a"]["chat_id"])
    _spend(_scope(root, "other-a", "other-a", "other-a"), 2)

    result = _tool(ctx)
    assert json.loads(result.text)["ok"], result.text
    [call] = supervisor.calls
    checked = _bundle(call["user_prompt"])["subject"]["physical_target"]
    assert checked["chat_id"] == checked["calling_chat_id"] == external_room
    assert json.loads(_tool(ctx, target="other-a").text)["ok"] is False
    assert len(supervisor.calls) == 1
    with usage_store.read(root) as view:
        assert work_cap(view, "other-a") is None


def test_existing_external_work_can_use_its_older_ingress_and_mail_sources(
        data_root, listed_rooms, supervisor, monkeypatch):
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.project_dialogue import build_owner_message_ref
    from supervisor.message_ingress import accept_chat

    root, room = data_root, 43  # An ordinary owner door, not a Project binding or a Main alias.

    def saved_ref(text, client, ts):
        row = accept_chat(room, 1, text, ts=ts, source="web", client_message_id=client, drive_root=root).row
        return build_owner_message_ref(chat_id=room, client_message_id=row["client_message_id"], ts=row["ts"],
                                       text=row["text"])

    opening, followup = "Check the report.", "Include the sources."
    origin = saved_ref(opening, "older-opening", "2026-11-19T19:00:00+00:00")
    mail_ref = saved_ref(followup, "older-follow-up", "2026-11-19T19:01:00+00:00")
    write_task_result(root, "work-a", "running", root_task_id="work-a", chat_id=room,
                      origin_message_ref=origin, origin_message_text=opening, billing_group=_binding("work-a"))
    assert write_owner_message(root, followup, "work-a", msg_id="older-mail", origin_message_ref=mail_ref)
    _spend(_scope(root, "work-a", "work-a", "work-a"), 2)
    ctx = _decision_context(root, root, monkeypatch, task_id="work-a", chat_id=room)

    read, _ = _instruction_read(_tool(ctx))
    assert not supervisor.calls and _stored(root) is None
    result = _tool(ctx, instructions_sha256=read)
    assert json.loads(result.text)["ok"], result.text
    [call] = supervisor.calls
    checked = _bundle(call["user_prompt"])["subject"]["physical_target"]
    assert checked["chat_id"] == checked["calling_chat_id"] == room
    [work] = checked["target_work_owner_instructions"]
    assert work["task_id"] == "work-a" and {row["text"] for row in work["rows"]} == {opening, followup}
    cap = _stored(root)
    assert cap["limit_usd"] == 15 and cap["basis"]["source"] == ctx.task_metadata["origin_message_ref"]


def test_an_owner_maximum_and_the_integrators_amount_are_two_decisions(
        data_root, listed_rooms, supervisor, monkeypatch):
    root = data_root
    _work(root, "work-a", chat_id=listed_rooms["work-a"]["chat_id"])
    _spend(_scope(root, "work-a", "work-a", "work-a"), 2)
    ctx = _decision_context(root, root, monkeypatch, words="Choose a total for work-a, up to $40.")

    monkeypatch.setattr("ouroboros.tools.control_models.cyber_pro", lambda _ctx: False)
    assert json.loads(_tool(ctx, delegated_max_usd=40, choose_limit=True).text)["error"] == (
        "integrator_choice_requires_cyber_pro")
    monkeypatch.setattr("ouroboros.tools.control_models.cyber_pro", lambda _ctx: True)
    assert json.loads(_tool(ctx, delegated_max_usd=40, choose_limit=True).text)["ok"]
    first = _stored(root)
    [call] = supervisor.calls
    authorization = _bundle(call["user_prompt"])["subject"]["physical_target"]["basis"]
    assert first["basis"] == {"kind": "own_judgement"} and first["limit_usd"] == 15
    assert first["by"]["actor"] == "ouroboros" and first["by"]["task_id"] == "main-a"
    assert first["owner_decision"] == {"basis": authorization}
    assert authorization == {"kind": "owner_words", "source": ctx.task_metadata["origin_message_ref"],
                             "decision_kind": "owner_words", "choose_limit": True, "delegated_max_usd": 40}

    assert json.loads(_tool(ctx, limit=40, owner_source="own_judgement").text)["ok"]
    assert json.loads(_tool(ctx, limit=41, owner_source="own_judgement").text)["error"] == "owner_cap_binds"
    current = _stored(root)
    assert (current["limit_usd"], current["owner_decision"]) == (40, first["owner_decision"])


def test_discretionary_check_sees_the_work_and_its_continued_billing_group(
        data_root, listed_rooms, supervisor, monkeypatch):
    """One ordinary Main call targets a new episode but checks the inherited group's instructions too."""
    from ouroboros.usage_admission import task_money_snapshot

    root = data_root
    room = listed_rooms["work-a"]["chat_id"]
    opening = _decision_context(root, root, monkeypatch, words="Use the team's weekday review lane only.")
    original = opening.task_metadata["origin_message_ref"]
    write_task_result(root, "earlier-a", STATUS_INTERRUPTED, root_task_id="earlier-a", chat_id=room,
                      billing_group=_binding("earlier-a"), origin_message_ref=original,
                      origin_message_text="Use the team's weekday review lane only.")
    write_task_result(root, "work-a", "running", root_task_id="work-a", chat_id=room,
                      billing_group=_binding("earlier-a"), continuation_of="earlier-a")
    _spend(_scope(root, "earlier-a", "earlier-a", "earlier-a"), 8)
    _spend(_scope(root, "work-a", "work-a", "earlier-a"), 1)
    monkeypatch.setattr("ouroboros.tools.control_models.cyber_pro", lambda _ctx: True)
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id="main-a", current_chat_id=1, is_direct_chat=True)

    read, _ = _instruction_read(_tool(ctx, owner_source="own_judgement"))
    result = _tool(ctx, owner_source="own_judgement", instructions_sha256=read)
    outcome = json.loads(result.text)
    assert outcome["ok"] and outcome["billing_group_id"] == "earlier-a" and outcome["known_usd"] == 9
    [call] = supervisor.calls
    subject = _bundle(call["user_prompt"])["subject"]["physical_target"]
    assert (subject["task_id"], subject["root_task_id"], subject["chat_id"], subject["billing_group_id"]) == (
        "work-a", "work-a", room, "earlier-a")
    assert subject["billing_group_chat_id"] == room
    instructions = {item["task_id"]: item for item in subject["target_work_owner_instructions"]}
    assert instructions["earlier-a"]["rows"][0]["text"] == "Use the team's weekday review lane only."
    with usage_store.read(root) as view:
        assert work_cap(view, "earlier-a")["limit_usd"] == 15 and work_cap(view, "work-a") is None
    assert task_money_snapshot(root, {"id": "work-a"}, "work-a", root_limit=10.0)["remaining_known_usd"] == 6


@pytest.mark.parametrize("move", ["before", "during", "none"])
def test_project_scope_follows_the_payer_of_a_related_occurrence(
        data_root, listed_rooms, supervisor, monkeypatch, move):
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.projects_registry import bind_task_to_project

    root = data_root
    room, other = listed_rooms["work-a"], listed_rooms["other-a"]
    _work(root, "payer-a", chat_id=room["chat_id"])
    write_task_result(root, "work-a", "running", root_task_id="work-a", chat_id=room["chat_id"],
                      billing_group=_binding("payer-a"))
    _spend(_scope(root, "payer-a", "payer-a", "payer-a"), 2)
    _spend(_scope(root, "work-a", "work-a", "payer-a"), 1)
    # A related occurrence uses a captured payer even after Project placement changes.
    ctx = _decision_context(root, root, monkeypatch, chat_id=room["chat_id"], project_id=room["id"])
    text = "Keep the draft in its paying room."
    assert write_owner_message(root, text, "payer-a", msg_id="payer-directions")

    def move_payer():
        bind_task_to_project(root, "payer-a", other["id"], other["chat_id"], origin={"absent": "system"})

    args = {}
    if move == "before":
        move_payer()
    else:
        read, bundles = _instruction_read(_tool(ctx))
        assert any(row["text"] == text for bundle in bundles for row in bundle["rows"])
        args["instructions_sha256"] = read
        if move == "during":
            check = supervisor.module._safety_model_call

            def after_binding(**kwargs):
                move_payer()
                return check(**kwargs)

            monkeypatch.setattr(supervisor.module, "_safety_model_call", after_binding)
    result = _tool(ctx, **args)
    with usage_store.read(root) as view:
        observed = work_cap(view, "payer-a")
    if move == "none":
        assert json.loads(result.text)["ok"] and observed["limit_usd"] == 15
    else:
        assert json.loads(result.text) == {"ok": False, "error": "work_out_of_scope"}
        assert observed is None
    assert len(supervisor.calls) == (0 if move == "before" else 1)


@pytest.mark.parametrize("scope", ["main", "same_room", "other_room"])
def test_pending_ledger_diagnostic_follows_the_payer_room(data_root, listed_rooms, supervisor, monkeypatch, scope):
    root = data_root
    room, other = listed_rooms["work-a"], listed_rooms["other-a"]
    paying_room = room if scope == "same_room" else other
    _work(root, "payer-a", chat_id=paying_room["chat_id"])
    write_task_result(root, "work-a", "running", root_task_id="work-a", chat_id=room["chat_id"],
                      billing_group=_binding("payer-a"))
    ctx = (_decision_context(root, root, monkeypatch) if scope == "main" else
           _decision_context(root, root, monkeypatch, chat_id=room["chat_id"], project_id=room["id"]))
    diagnostic = {"error": "work_cap_reconcile_pending", "billing_group_id": "payer-a", "reason": "pending test import"}
    monkeypatch.setattr("ouroboros.usage_admission.work_cap_target", lambda *_: diagnostic)

    result = _tool(ctx)
    expected = {"error": "work_out_of_scope"} if scope == "other_room" else diagnostic
    assert result.status == "error" and json.loads(result.text) == {"ok": False, **expected}
    assert not supervisor.calls


@pytest.mark.parametrize("payer", ["work-a", "other-a"])
def test_write_diagnostic_follows_the_checked_payer(data_root, listed_rooms, supervisor, monkeypatch, payer):
    root = data_root
    _work(root, "work-a", chat_id=listed_rooms["work-a"]["chat_id"])
    ctx = _decision_context(root, root, monkeypatch)
    diagnostic = {"error": "work_cap_reconcile_pending", "billing_group_id": payer, "reason": "pending test import"}

    def refused_write(*_args, expected_group, **_kwargs):
        assert expected_group == "work-a"
        return {"ok": False, **diagnostic}

    monkeypatch.setattr("ouroboros.usage_admission.set_work_cap", refused_write)
    result = _tool(ctx)
    expected = diagnostic if payer == "work-a" else {"error": "cost_cap_target_changed"}
    assert json.loads(result.text) == {"ok": False, **expected}
    assert len(supervisor.calls) == 1 and _stored(root) is None


def test_retained_legacy_payer_mail_preserves_words_without_an_author(data_root, listed_rooms, supervisor, monkeypatch):
    from ouroboros.owner_mailbox import _mailbox_path
    from ouroboros.safety_evidence import UNKINDED_STAMP
    from ouroboros.task_custody import capture_owner_mail

    root = data_root
    room = listed_rooms["work-a"]["chat_id"]

    def payer(**retained):
        write_task_result(root, "payer-a", STATUS_INTERRUPTED, root_task_id="payer-a", chat_id=room,
                          billing_group=_binding("payer-a"), **retained)

    payer()
    write_task_result(root, "work-a", "running", root_task_id="work-a", chat_id=room, billing_group=_binding("payer-a"))
    _spend(_scope(root, "payer-a", "payer-a", "payer-a"), 2)
    _spend(_scope(root, "work-a", "work-a", "payer-a"), 1)
    text = "Keep the original mockup."
    path = _mailbox_path(root, "payer-a")
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy = json.dumps({"msg_id": "legacy-one", "ts": "2026-11-19T22:00:00+00:00", "text": text})
    path.write_text(legacy + "\n", encoding="utf-8")
    ctx = _decision_context(root, root, monkeypatch)
    read, active = _instruction_read(_tool(ctx))
    retained = capture_owner_mail(root, "payer-a")
    assert retained["read_complete"] and retained["rows"] == [legacy]
    payer(owner_mailbox=retained)
    path.rename(path.with_suffix(".kept"))  # use retained bytes without removing the synthetic source
    again, after = _instruction_read(_tool(ctx))
    assert active == after and read == again
    [direction] = next(bundle["rows"] for bundle in after if bundle["task_id"] == "payer-a")
    assert direction["text"] == text and direction["author"] == UNKINDED_STAMP and "mail_kind" not in direction
    assert not supervisor.calls and _stored(root) is None


def test_unavailable_durable_room_refuses_before_safety(data_root, listed_rooms, supervisor, monkeypatch):
    # The roster's room display is not a replacement for an admitted address.
    _work(data_root, "work-a")
    ctx = _decision_context(data_root, data_root, monkeypatch)
    assert json.loads(_tool(ctx).text)["error"] == "cost_cap_authority_unreadable"
    assert not supervisor.calls and _stored(data_root) is None


@pytest.mark.parametrize("legacy", ["render_address", "client_reference"])
def test_an_older_opening_directive_without_a_digest_still_shows_the_cited_row(
        data_root, listed_rooms, supervisor, monkeypatch, legacy):
    """An older carrier recorded the opening by an address alone: the checker still reads the cited row, its words and
    mark; the render address takes the row's digest (one row), an older client reference stays a row of its own."""
    from ouroboros.chat_chain import format_address

    root = data_root
    _work(root, "work-a", chat_id=listed_rooms["work-a"]["chat_id"])
    _spend(_scope(root, "work-a", "work-a", "work-a"), 2)
    ctx = _decision_context(root, root, monkeypatch)
    ref = ctx.task_metadata["origin_message_ref"]
    address = format_address(ref) if legacy == "render_address" else f"chat 1 / {ref['client_message_id']}"
    ctx._owner_directives = [{"source": "initial_user", "content": WORDS, "msg_id": address}]

    result = _tool(ctx)
    assert json.loads(result.text)["ok"], result.text
    prompt = supervisor.calls[0]["user_prompt"]
    [cited] = _cited(prompt)
    assert cited["content"] == WORDS and "carried" not in cited
    assert cited["author"] == SOURCE_AUTHORS["initial_user" if legacy == "render_address" else "owner_mailbox"]
    assert prompt.count(WORDS) == (1 if legacy == "render_address" else 2)


def test_a_cap_reading_is_kept_by_its_bound_decision_and_other_calls_by_their_arguments():
    """The prior reading a returned call is matched to: a cost cap's includes the work and decision bound for it; every
    other call keeps its arguments alone, a peer letter's prepared operation included."""
    from ouroboros.safety import _subject_digest
    from ouroboros.safety_evidence import call_subject

    args = {"task_id": "work-a", "limit_usd": 15, "reason": "The owner chose this total"}
    one = {"task_id": "work-a", "root_task_id": "work-a", "chat_id": 1,
           "basis": {"kind": "owner_words", "source": {"ts": "1"}}}
    other = {**one, "basis": {"kind": "owner_words", "source": {"ts": "2"}}}
    assert _subject_digest("set_task_cost_cap", call_subject("set_task_cost_cap", args, one)) != _subject_digest(
        "set_task_cost_cap", call_subject("set_task_cost_cap", args, other))
    letter = {"peer_letter": {"letter_id": "letter-1", "operation": "assigns work"}}
    assert call_subject("delegate_to_ouroboros", args, letter) is args
    assert call_subject("write_file", args, None) is args
