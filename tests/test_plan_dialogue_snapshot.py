"""The own-room dialogue a plan ask carries (``plan_dialogue``): the numbered inline
conversation, its per-seat fit with the exact omitted range named, the room-read fact on the
plan row, related rooms as pointers only, the budget priced on the inputs actually fitted, the
plan record's sources surviving a child's promotion, and the recorded snapshot a
``chat:<id>@<sha256>`` locator reads."""
from __future__ import annotations

import json
import pathlib
from pathlib import Path

import pytest

from ouroboros import dialogue_store
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.tools.plan_dialogue import plan_chat_reader
from ouroboros.tools.plan_packet import PLAN_REPLY_CONTRACT
from tests import _dialogue as dialogue_test
from tests._review_substrate_shared import user_text
from tests.test_plan_task_tool import CLEAN, _write
from tests.test_plan_task_tool import harness as _plan_task_harness

harness = _plan_task_harness

_ASK = "plan-ask-task-1-a1"  # the request id plan_ask.prepare_ask gives a task's first ask


def test_inline_view_is_the_numbered_conversation(harness):
    """D1: the inline OWN ROOM DIALOGUE is the conversation as numbered readable lines
    (owner rows with attachment names, Ouroboros rows, quiz cards with the chosen answer),
    each line's number being the physical snapshot line a `::lines=` locator addresses;
    progress rows, host system rows and the archive-file list stay behind the pointer, which the
    OWN ROOM SNAPSHOT section names with every per-capture value (it follows the plan).
    Reverted, the packet is the JSONL snapshot with its metadata dump."""
    from ouroboros.tools.plan_dialogue import attach_own_dialogue, dialogue_view, render_dialogue

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    dialogue_store.append(harness.drive, "chat", {"direction": "in", "chat_id": 1, "ts": "2026-09-01T10:00:00Z", "text": "Use the agreed outline",
                        "attachment_manifest": [{"label": "outline.pdf", "name": "outline.pdf"}]})
    dialogue_store.append(harness.drive, "chat", {"direction": "out", "chat_id": 1, "ts": "2026-09-01T10:01:00Z", "text": "Option A is faster\nOption B keeps choices"})
    dialogue_store.append(harness.drive, "chat", {"direction": "out", "chat_id": 1, "ts": "2026-09-01T10:02:00Z", "type": "quiz", "text": "Five or six slides?",
                        "quiz": {"quiz_id": "q1", "question": "Five or six slides?", "options": ["five", "six"], "recommended_index": 0}})
    dialogue_store.append(harness.drive, "chat", {"direction": "in", "chat_id": 1, "ts": "2026-09-01T10:03:00Z", "type": "quiz_answer", "text": "five",
                        "quiz": {"quiz_id": "q1", "options": ["five", "six"], "answered_index": 0, "comment": "The board asked for five."}})
    dialogue_store.append(harness.drive, "chat", {"direction": "system", "chat_id": 1, "ts": "2026-09-01T10:04:00Z", "type": "task_summary", "text": "HOST SUMMARY ROW"})
    for step in ("Reading notes", "Drafting slide 1", "Drafting slide 2"):
        dialogue_store.append(harness.drive, "progress", {"chat_id": 1, "ts": "2026-09-01T10:05:00Z", "content": f"PROGRESS {step}"})
    manifest = attach_own_dialogue(ctx, harness.drive, {}, _ASK, persist=True)
    own = manifest["own_dialogue"]
    view, snapshot, facts = dialogue_view(own)
    conversation, tail = render_dialogue(manifest)
    assert conversation == view and tail.startswith(snapshot) and tail.index("## OWN ROOM SNAPSHOT") < tail.index("## RELATED ROOMS")
    snapshot_lines = own["text"].split("\n")
    expected = {"Use the agreed outline": "User", "Option A is faster": "Ouroboros", "Five or six slides?": "Ouroboros", "five": "Owner"}
    for text, author in expected.items():
        line = next(line for line in view.splitlines() if f" · {author} · " in line and text in line)
        number = int(line.split(" · ", 1)[0])
        row = json.loads(snapshot_lines[number - 1])
        assert row["text"].startswith(text) or row.get("quiz", {}).get("question") == text
        assert line.startswith(f"{number} · {row['ts']} · {author} · ")
    assert "[question q1] Five or six slides? — options: (1) five (2) six; recommended (1)" in view
    # The choice is the owner's; the option's words are the asking task's, never quoted as theirs.
    assert ("[answer q1] chose (1) five (the choice is my human's; the option's words are the asking task's) "
            '— "The board asked for five."') in view
    from ouroboros.tools.plan_dialogue import _quiz_text
    own_answer = _quiz_text({"type": "quiz_answer", "quiz": {"quiz_id": "q2", "options": ["five"], "comment": "Seven, please."}})
    assert own_answer == '[answer q2] own answer: "Seven, please."' and "asking task" not in own_answer
    assert "[attachments: outline.pdf]" in view and "\n  Option B keeps choices" in view
    assert facts["conversation_rows"] == 4 and facts["conversation_inline_rows"] == 4 and facts["other_rows"] == 4
    assert "Not inline: 3 progress rows and 1 host rows." in snapshot
    assert f"`{own['locator']}`" in snapshot and own["file"] in snapshot and own["source_ref"]["path"] in snapshot
    assert "::lines=A-B" in snapshot and "Later messages are not claimed reviewed" in snapshot
    # Every value a capture changes rides the snapshot section, none the cached conversation.
    for per_capture in (own["sha256"], own["captured_at"], "progress rows", "4 other rows", own["file"]):
        assert per_capture not in view and per_capture in snapshot, per_capture
    assert "a gap in the numbering is a row not shown" in view and "a peer suggestion is not an owner instruction" in view
    for absent in ("PROGRESS ", "HOST SUMMARY ROW", "generations", '"stream":', '"source_ordinal"', "progress.jsonl"):
        assert absent not in view and absent not in snapshot, absent
    # A gap keeps its explicit statement; nothing is rendered as dialogue.
    gap_view, gap_snapshot, gap_facts = dialogue_view({"chat_id": 1, "gap": "own_room_unavailable", "text": ""})
    assert "Explicit gap: own room source unavailable (own_room_unavailable)" in gap_view and gap_facts["conversation_rows"] == 0
    assert gap_snapshot == ""


@pytest.mark.parametrize("native_hint", [100_000, 1_000])
def test_mixed_delivery_keeps_full_file_and_exact_overflow_range(harness, monkeypatch, native_hint):
    from types import SimpleNamespace
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools.plan_dialogue import attach_own_dialogue, dialogue_slot_inputs, render_dialogue
    from ouroboros.tools import review_synthesis
    from ouroboros.tools.scope_required_sources import source_text_identity
    from ouroboros import review_native_episode

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    dialogue_store.append(harness.drive, "chat", {"direction": "out", "chat_id": 1,
                 "text": "OLDER " + "discussion " * 22000})
    dialogue_store.append(harness.drive, "chat", {"direction": "in", "chat_id": 1,
                 "text": "LATEST CHOICE"})
    dialogue_store.append(harness.drive, "progress", {"chat_id": 1, "content": "PROGRESS ROW " * 50})
    manifest = attach_own_dialogue(ctx, harness.drive, {}, _ASK, persist=True)
    own = manifest["own_dialogue"]
    packet = "## THE PLAN\n\nOutline first.\n\n".join(render_dialogue(manifest))  # the rows, the plan, this capture's facts
    assert "PROGRESS ROW" not in packet  # never inline, for any route
    def slot(name, native=False, session=False):
        return SimpleNamespace(slot_id=name, model="same/model", role_hint="plan reviewer", use_local=False, session_profile=name,
                               route=ReviewRouteKind.AGENT_SESSION if session else ReviewRouteKind.API_CHAT,
                               retrieves=native or session, native_retrieval=native)
    slots = [slot("small"), slot("large"), slot("native", native=True), slot("delegated", session=True)]
    monkeypatch.setattr(review_synthesis, "per_slot_input_token_limits", lambda *a, **k: {"small": 4000, "large": 200000})
    declarations = []
    def bound(*a, **kwargs):
        declarations.append(kwargs)
        return native_hint
    monkeypatch.setattr(review_native_episode, "review_native_transcript_bound", bound)
    delivery = dialogue_slot_inputs(slots, system_prompt="governance", user_content=packet,
                                   session_task=packet, manifest=manifest, slot_messages={}, native_mandatory_chars=len(packet), session_root=str(harness.workspace), task_id=ctx.task_id,
                                   output_contract=PLAN_REPLY_CONTRACT)
    small = json.dumps(delivery["slot_messages"]["small"], ensure_ascii=False)
    large = json.dumps(delivery["slot_messages"]["large"], ensure_ascii=False)
    assert "LATEST CHOICE" in small and "exact omitted prefix" in small
    assert "1 earlier conversation rows before line 3" in small and "OLDER " not in small and "OLDER " in large
    coverage = delivery["dialogue_delivery"]
    assert coverage["small"]["conversation_inline_rows"] == 1 and coverage["small"]["conversation_first_inline_line"] == 3
    assert coverage["large"]["conversation_inline_rows"] == 2 and coverage["large"]["delivery"] == "packet"
    native = delivery["slot_session_tasks"]["native"]
    assert "exact omitted prefix" in native and coverage["native"]["delivery"] == "native_retrieving"
    if native_hint == 100_000:
        assert "LATEST CHOICE" in native
    else:
        # An irreducible first send is retained above the hint, with its source.
        assert delivery["slot_prompt_chars"]["native"] > native_hint
        assert own["source_ref"]["path"] in native
        assert Path(own["file"]).read_text(encoding="utf-8") == own["text"]
    assert declarations[0]["mandatory_read_chars"] == len(packet)
    delegated = delivery["slot_session_tasks"]["delegated"]
    # The delegated session gets the same inline conversation (no invented window) plus the pointer.
    assert "OLDER " in delegated and "LATEST CHOICE" in delegated and "MANDATORY FULL READ" not in delegated
    assert coverage["delegated"] == {**coverage["large"], "delivery": "delegated_session", "window": "unknown", "file": own["file"]}
    assert delivery["request_policy"]["observed_sources"] == [{
        "root": "artifact_store", "path": own["source_ref"]["path"], "file": own["file"],
        **source_text_identity(own["text"].encode("utf-8"))}]
    # Every slot's send ends with ITS OWN panel seat: a shared packet cannot say "your".
    last_user = {sid: user_text(delivery["slot_messages"][sid][-1]["content"]) for sid in ("small", "large")}
    seat = {sid: f"\n## YOUR PANEL SEAT\n\n`{sid}`\n" for sid in last_user}
    assert all(last_user[sid].endswith(seat[sid]) for sid in last_user)
    assert last_user["large"].removesuffix(seat["large"]) == packet  # the seat is the only addition
    assert native.endswith("\n## YOUR PANEL SEAT\n\n`native`\n") and delegated.endswith("\n## YOUR PANEL SEAT\n\n`delegated`\n")
    assert own["file"] in delegated and "1M" not in delegated
    # Every seat kind keeps every capture fact, after the plan; a packet seat's marked block holds none.
    for text in (last_user["small"], last_user["large"], native, delegated):
        after = text[text.index("## THE PLAN"):]
        assert f"Captured {own['captured_at']}." in after and f"`{own['locator']}`" in after and "## RELATED ROOMS" in after
    for sid in ("small", "large"):
        marked = delivery["slot_messages"][sid][-1]["content"][0]
        assert "cache_control" in marked and own["sha256"] not in marked["text"] and own["captured_at"] not in marked["text"]
    source = read_actor_source_bytes(harness.drive, ctx.task_id, own["source_ref"])
    assert len(source) > 120000 and source.decode() == own["text"]
    import re
    match = re.search(r'::lines=2-(\d+)', native)
    # The ordinary hint omits only OLDER; the tiny one addresses the whole snapshot.
    assert match and int(match[1]) == (2 if native_hint == 100_000 else own["lines"])
    assert not (harness.workspace / ".ouroboros-review").exists()


def test_a_session_with_a_resolved_window_is_fitted_and_without_one_is_whole(harness, monkeypatch):
    """A session uses the shared route acknowledgement; unknown capacity keeps the whole dialogue."""
    from types import SimpleNamespace
    from ouroboros import capability_evidence as ce, config
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools.plan_dialogue import attach_own_dialogue, dialogue_slot_inputs, render_dialogue

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    dialogue_store.append(harness.drive, "chat", {"direction": "out", "chat_id": 1, "text": "OLDER " + "discussion " * 22000})
    dialogue_store.append(harness.drive, "chat", {"direction": "in", "chat_id": 1, "text": "LATEST CHOICE"})
    manifest = attach_own_dialogue(ctx, harness.drive, {}, _ASK, persist=True)
    packet = "## THE PLAN\n\nOutline first.\n\n".join(render_dialogue(manifest))
    session = SimpleNamespace(slot_id="delegated", model="cursor=grok", role_hint="plan reviewer", use_local=False,
                              session_profile="", route=ReviewRouteKind.AGENT_SESSION, retrieves=True, native_retrieval=False)
    monkeypatch.setattr(config, "DATA_DIR", harness.drive)
    ack = ce.record_owner_ack(harness.drive, provider="agent_session", model=session.model, window_tokens=40_000)
    fitted = dialogue_slot_inputs([session], system_prompt="g", user_content=packet, session_task=packet, manifest=manifest,
                                  slot_messages={}, native_mandatory_chars=len(packet), session_root=str(harness.workspace), task_id=ctx.task_id,
                                  output_contract=PLAN_REPLY_CONTRACT)
    task = fitted["slot_session_tasks"]["delegated"]
    assert "LATEST CHOICE" in task and "OLDER " not in task and "1 earlier conversation rows before line 3" in task
    assert fitted["dialogue_delivery"]["delegated"]["window"] == "owner_ack"
    ce.revoke_owner_ack(harness.drive, ack["route_fp"])
    whole = dialogue_slot_inputs([session], system_prompt="g", user_content=packet, session_task=packet, manifest=manifest,
                                 slot_messages={}, native_mandatory_chars=len(packet), session_root=str(harness.workspace), task_id=ctx.task_id,
                                 output_contract=PLAN_REPLY_CONTRACT)
    assert "OLDER " in whole["slot_session_tasks"]["delegated"] and whole["dialogue_delivery"]["delegated"]["window"] == "unknown"


def test_a_session_reviewers_room_read_rides_its_plan_row():
    """The observed-source fold reaches the plan row a reply settles from as a fact: attestation
    `harness_observed` and `room_read_coverage`; a session with no fold stays `unobserved`
    with no coverage key."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.tools.plan_review_runtime import _plan_row_from_actor

    slot = ReviewSlot(slot_id="s1", model="cursor=grok", route=ReviewRouteKind.AGENT_SESSION)
    usage = {"read_provenance": "harness_observed", "native_read_coverage": {"status": "complete", "sources": [
        {"root": "artifact_store", "path": "plan-dialogue-1.jsonl", "status": "complete", "covered_chars": 120, "complete_chars": 120}]}}
    actor = {"slot_id": "s1", "model": "cursor=grok", "status": "ok", "raw_text": CLEAN, "usage": usage,
             "operation_id": "op-s1", "operation_state": "settled"}
    row = _plan_row_from_actor(actor, slot)
    coverage = {"status": "complete", "covered_chars": 120, "complete_chars": 120, "provenance": "harness_observed"}
    assert row["host_file_read_attestation"] == "harness_observed" and row["room_read_coverage"] == coverage
    plain = _plan_row_from_actor({**actor, "usage": {}}, slot)
    assert plain["host_file_read_attestation"] == "unobserved" and "room_read_coverage" not in plain


def test_plan_record_sources_survive_real_child_promotion_and_cleanup(harness):
    """A plan written and asked in a child's execution drive: once the child's result is
    promoted and its drive removed, the parent reads every revision's exact text and every
    participant's exact delivery (with the room as it was shown) and reply."""
    from ouroboros import plan_record
    from ouroboros.headless import copy_child_task_result, prepare_task_drive, remove_subagent_task_drive, retry_child_task_refs
    from ouroboros.review_ledger import read_source
    from ouroboros.task_results import write_task_result

    ctx = harness.make_ctx(task_id="source")
    parent = harness.drive
    child = prepare_task_drive(parent, "source", "empty")
    ctx.drive_root = child
    ctx.current_chat_id = 1
    harness.install({})
    dialogue_store.append(child, "chat", {"direction": "in", "chat_id": 1, "text": "Must survive cleanup"})
    _write(ctx, ask="council")
    record = plan_record.load_plan_record(child, "source")
    seats = record["asks"][0]["participants"]
    exact = {"revision": read_actor_source_bytes(child, "source", record["revisions"][0]["text_ref"]),
             **{(p["seat_id"], role): read_source(child, "source", p[role]) for p in seats for role in ("delivery_ref", "reply_ref")}}
    assert all(b"Must survive cleanup" in exact[(p["seat_id"], "delivery_ref")] for p in seats)
    write_task_result(child, "source", "completed")
    copied = copy_child_task_result(parent, {"id": "source", "drive_root": str(child)})
    assert copied["child_ref_promotion"]["status"] == "incomplete"
    copied = retry_child_task_refs(parent, child, "source")
    assert copied["child_ref_promotion"]["status"] == "complete"
    assert remove_subagent_task_drive(parent, "source", live=lambda _task: False) is True
    assert not child.exists()
    promoted = plan_record.load_plan_record(parent, "source")
    assert [r["content_sha256"] for r in promoted["revisions"]] == [r["content_sha256"] for r in record["revisions"]]
    assert read_actor_source_bytes(parent, "source", promoted["revisions"][0]["text_ref"]) == exact["revision"]
    for p in promoted["asks"][0]["participants"]:
        for role in ("delivery_ref", "reply_ref"):
            assert read_source(parent, "source", p[role]) == exact[(p["seat_id"], role)], (p["seat_id"], role)


def test_same_root_sibling_rooms_are_pointers_and_unrelated_rooms_stay_private(harness):
    from ouroboros.projects_registry import create_project, bind_task_to_project
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.plan_dialogue import related_rooms

    projects = {name: create_project(harness.drive, name, name=name) for name in ('own', 'sibling', 'unrelated')}
    for tid, name, root in [('child-a', 'own', 'parent'), ('child-b', 'sibling', 'parent'), ('private', 'unrelated', 'different-root')]:
        bind_task_to_project(harness.drive, tid, projects[name]['id'], origin={'absent': 'system'})
        write_task_result(harness.drive, tid, 'running', parent_task_id=root, root_task_id=root)
    ctx = harness.make_ctx(task_id='child-a')
    ctx.task_metadata = {'parent_task_id': 'parent', 'root_task_id': 'parent'}
    pointers = related_rooms(ctx, harness.drive, projects['own']['chat_id'])
    assert {p['locator'] for p in pointers} == {'chat:1', f"chat:{projects['sibling']['chat_id']}"}
    assert all(p['delivery'] == 'pointer_only' and 'text' not in p for p in pointers)


def test_budget_prices_the_actual_window_fitted_inputs(harness, monkeypatch):
    from ouroboros import usage_accounting as ua
    from ouroboros.tools import review_synthesis
    from ouroboros.tools.plan_review_runtime import PLAN_REVIEW_MAX_TOKENS

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    dialogue_store.append(harness.drive, "chat", {'ts': '2026-09-01T00:00:00Z',
        'direction': 'out', 'chat_id': 1, 'text': 'Prior substantive discussion. ' * 20000})
    monkeypatch.setattr(review_synthesis, 'per_slot_input_token_limits',
                        lambda models, **kw: {slot.slot_id: 10000 for slot in kw['slots']})
    substrate = harness.install({})
    captured = []
    monkeypatch.setattr('ouroboros.tools.review_helpers.review_wave_budget_gate', lambda *a, **kw: captured.append(kw))
    _write(ctx, ask='council')
    request = substrate.calls[0]['request']
    slots = substrate.calls[0]['slots']
    actual = [sum(len(user_text(message['content'])) for message in request.slot_messages[slot.slot_id]) for slot in slots]
    assert captured[0]['prompt_chars'] == actual
    assert captured[0]['max_completion_tokens'] == PLAN_REVIEW_MAX_TOKENS
    assert captured[0]['models'] == [slot.model for slot in slots]
    # Illustrative price replaces only vendor lookup, not the admission math.
    monkeypatch.setattr(ua, 'estimate_cost_optional', lambda model, prompt, completion, **kw: prompt / 100000)
    admission = ua.review_wave_admission(root_task_id='budget-probe', models=[slot.model for slot in slots],
        prompt_chars=captured[0]['prompt_chars'], max_completion_tokens=PLAN_REVIEW_MAX_TOKENS, remaining_usd_override=1.0)
    # The conversation-only view keeps whole rows: this room's one oversize row fits no
    # 10k-token slot, so every packet is priced on the small pointer view, never on a
    # byte tail that filled the window.
    assert admission['fits'] is True and 0 < admission['estimated_wave_usd'] < 0.3
    assert len(substrate.calls) == 1


def test_snapshot_qualified_room_keeps_original_gap_disclosure(harness):
    """`chat:<id>@<sha256>` names an exact recorded room snapshot (an earlier-schema wave
    records one): it reads the recorded bytes with their original gap disclosure, whatever
    the room says later, and a line range selects from those bytes. The unqualified
    `chat:<id>` and a new ask read the room as it is now."""
    from ouroboros.task_results import _compact_plan_review_wave
    from ouroboros.tools.plan_dialogue import attach_own_dialogue
    from ouroboros.tools.plan_evidence import resolve_evidence
    from tests._legacy_plan_shared import seed_plan_review_state, v2_state, wave

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    # The unreadable row is a torn line of the older install's chain, imported as the upgrade does.
    dialogue_test.seed(harness.drive, live={"chat": [{"chat_id": 1, "text": "Retained explanation"},
                                                     "{unreadable history row\n"]})
    dialogue_test.import_all(harness.drive)
    ref = attach_own_dialogue(ctx, harness.drive, {}, _ASK, persist=True)["own_dialogue"]["source_ref"]
    recorded = wave(dialogue_source_ref=ref, dialogue_chat_id=1)
    seed_plan_review_state(harness.drive, ctx.task_id, v2_state(recorded))
    assert _compact_plan_review_wave(recorded)["dialogue_source_ref"] == ref  # an old compacted wave still names it
    original = read_actor_source_bytes(harness.drive, ctx.task_id, ref)
    dialogue_store.append(harness.drive, "chat", {"chat_id": 1, "text": "Later room growth"})
    reader = plan_chat_reader(harness.drive, ctx.task_id)
    exact = reader(f"1@{ref['sha256']}")
    assert exact["text"].encode() == original and "Later room growth" not in exact["text"]
    assert exact["coverage"]["chat"]["gaps"] and exact["coverage"]["chat"]["snapshot_stable"]
    manifest = resolve_evidence([f"chat:1@{ref['sha256']}::lines=2-2"],
        active_root=harness.workspace, allowed_roots=[], resolve_chat=reader)
    assert json.loads(manifest["attached"][0]["text"])["text"] == "Retained explanation"
    assert any(row["reason"].startswith("chat_history_gap:") for row in manifest["omissions"])
    assert "Later room growth" in reader("1")["text"]
    later = harness.make_ctx(task_id="task-2")
    later.current_chat_id = 1
    substrate = harness.install({})
    _write(later, ask="council")
    assert "Later room growth" in user_text(substrate.calls[0]["request"].slot_messages["s1"][-1]["content"])


def test_two_asks_over_an_unchanged_conversation_share_the_cached_prefix_and_each_names_its_capture(harness, monkeypatch):
    """The prompt cache (owner, 28.10): host progress rows land and the clock moves between two asks of
    one task, yet the room's conversation is unchanged, so everything before ``## THE PLAN`` (the
    block a breakpoint-honouring route caches) is byte-identical, including a fresh seat's marked
    block; each ask's OWN ROOM SNAPSHOT after the plan names its own locator, counts and capture time.
    A new conversation row changes the prefix. Reverted (the snapshot facts before the plan), the
    two prefixes part at the room's locator."""
    from ouroboros import dialogue_evidence, plan_record
    from ouroboros.tools.plan_packet import plan_user_stable_len
    from tests.test_plan_task_tool import _call

    clock = {"now": "2026-10-28T10:00:00+00:00"}
    monkeypatch.setattr(dialogue_evidence, "utc_now_iso", lambda: clock["now"])
    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    dialogue_store.append(harness.drive, "chat", {"direction": "in", "chat_id": 1, "ts": "2026-10-28T09:00:00Z", "text": "Use the agreed outline"})
    dialogue_store.append(harness.drive, "chat", {"direction": "out", "chat_id": 1, "ts": "2026-10-28T09:01:00Z", "text": "Five slides it is"})
    dialogue_store.append(harness.drive, "progress", {"chat_id": 1, "ts": "2026-10-28T09:02:00Z", "content": "PROGRESS reading"})
    sub = harness.install({})
    _write(ctx, ask=["s1"])
    for step in ("drafting", "checking"):
        dialogue_store.append(harness.drive, "progress", {"chat_id": 1, "ts": "2026-10-28T10:30:00Z", "content": f"PROGRESS {step}"})
    # A host note through the task's mailbox (a settled ask, a review pause) is a host row, not conversation.
    from ouroboros.owner_mailbox import write_task_message
    assert write_task_message(pathlib.Path(str(harness.drive)), "HOST NOTE plan ask settled", ctx.task_id,
                              source_task_id=ctx.task_id, provenance="system", msg_id=f"plan-complete:{ctx.task_id}:a1")
    clock["now"] = "2026-10-28T11:00:00+00:00"
    _call(ctx, ask=["s2"], message="Second look.")  # a fresh seat: its first user block carries the cache marker
    sent = [sub.calls[0]["request"].slot_messages["s1"][-1]["content"], sub.calls[1]["request"].slot_messages["s2"][-1]["content"]]
    texts = [user_text(content) for content in sent]
    bounds = [plan_user_stable_len(text) for text in texts]
    prefixes = [text[:bound] for text, bound in zip(texts, bounds)]
    assert prefixes[0] == prefixes[1] and "Use the agreed outline" in prefixes[0] and "Five slides it is" in prefixes[0]
    assert all(isinstance(c, list) and "cache_control" in c[0] for c in sent) and sent[0][0]["text"] == sent[1][0]["text"]
    for absent in ("chat:1@", "Captured", "progress rows", "## OWN ROOM SNAPSHOT", "## RELATED ROOMS", "PROGRESS ", "HOST NOTE"):
        assert absent not in prefixes[0], absent
    asks = plan_record.load_plan_record(harness.drive, ctx.task_id)["asks"]
    shas = [ask["dialogue_ref"]["sha256"] for ask in asks]
    counts = []
    for text, bound, ask, other, time in zip(texts, bounds, asks, reversed(shas), ("10:00", "11:00")):
        source = read_actor_source_bytes(harness.drive, ctx.task_id, ask["dialogue_ref"]["ref"]).decode("utf-8")
        counts.append(sum(1 for line in source.split("\n")[1:] if line and json.loads(line).get("stream") == "progress"))
        tail = text[bound:]
        assert f"`chat:1@{ask['dialogue_ref']['sha256']}`" in tail and f"chat:1@{other}" not in text
        assert f"Captured 2026-10-28T{time}:00+00:00." in tail and f"Not inline: {counts[-1]} progress rows" in tail
        order = ["## THE PLAN", "## OWN ROOM SNAPSHOT (this ask)", "## RELATED ROOMS", "## THE COUNCIL SO FAR"]
        assert [text.index(part) for part in order] == sorted(text.index(part) for part in order) and text.index(order[0]) == bound
    assert shas[0] != shas[1] and counts[1] > counts[0]  # host rows changed the snapshot, not its conversation
    # The other direction: a new conversation row is the cached block's own change.
    dialogue_store.append(harness.drive, "chat", {"direction": "in", "chat_id": 1, "ts": "2026-10-28T11:30:00Z", "text": "Make it six"})
    _call(ctx, ask=["s3"], message="Third look.")
    third = user_text(sub.calls[2]["request"].slot_messages["s3"][-1]["content"])
    assert third[:plan_user_stable_len(third)] != prefixes[0] and "Make it six" in third[:plan_user_stable_len(third)]
