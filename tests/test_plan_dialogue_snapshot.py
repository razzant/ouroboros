"""A growing room cannot mint a paid cycle during exact replay/collection."""
from __future__ import annotations

import json

from tests.test_plan_review_engine import harness as _harness, _call, _state, CLEAN

harness = _harness
from tests.test_plan_review_reconciliation import _collect, _install_barrier_substrate
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.tools.plan_dialogue import plan_chat_reader
from ouroboros.utils import append_jsonl


def test_room_growth_reuses_snapshot_until_author_changes_plan(harness):
    substrate = harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    chat = harness.drive / "logs" / "chat.jsonl"
    append_jsonl(chat, {"direction": "in", "chat_id": 1, "text": "Owner chooses A"})
    append_jsonl(chat, {"direction": "out", "chat_id": 1, "text": "A costs less; B keeps options"})
    _call(ctx)
    first = _state(harness)["waves"][-1]
    source = read_actor_source_bytes(harness.drive, ctx.task_id, first["dialogue_source_ref"])
    append_jsonl(chat, {"direction": "out", "chat_id": 1, "text": "Panel complete, collecting"})
    append_jsonl(chat, {"direction": "in", "chat_id": 1, "text": "Actually keep B as well"})
    replay_text = _call(ctx)
    replay = _state(harness)["waves"][-1]
    from ouroboros.tools.plan_review_artifacts import read_wave
    own = read_wave(harness.drive, ctx.task_id, first["wave_artifact"])["evidence_manifest_full"]["own_dialogue"]
    assert first["dialogue_source_ref"]["sha256"] in replay_text
    assert first["dialogue_source_ref"]["path"] in replay_text and own["captured_at"] in replay_text
    assert "Later messages are not claimed reviewed" in replay_text
    assert "Snapshot coverage" in replay_text
    assert replay["request_fingerprint"] == first["request_fingerprint"]
    assert _state(harness)["cycles_paid"] == 1 and len(substrate.calls) == 1
    exact = plan_chat_reader(harness.drive, ctx.task_id)(f"1@{first['dialogue_source_ref']['sha256']}")
    assert exact["text"].encode() == source and "Actually keep B" not in exact["text"]
    _call(ctx, plan="Revise the outline to keep both A and B.")
    revised = _state(harness)["waves"][-1]
    assert revised["dialogue_source_ref"] != first["dialogue_source_ref"]
    assert b"Actually keep B as well" in read_actor_source_bytes(harness.drive, ctx.task_id, revised["dialogue_source_ref"])
    assert _state(harness)["cycles_paid"] == 2 and len(substrate.calls) == 2


def test_free_collection_keeps_exact_range_after_progress_and_mailbox_growth(harness, monkeypatch):
    calls = []
    _install_barrier_substrate(monkeypatch, calls)
    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    append_jsonl(harness.drive / "logs" / "chat.jsonl", {"direction": "out", "chat_id": 1, "text": "Before dispatch"})
    _call(ctx)
    first = _state(harness)["waves"][-1]
    append_jsonl(harness.drive / "logs" / "progress.jsonl", {"chat_id": 1, "content": "Panel done"})
    _collect(ctx, first["request_fingerprint"])
    collected = _state(harness)["waves"][-1]
    assert collected["dialogue_source_ref"] == first["dialogue_source_ref"]
    assert [row["reconcile_only"] for row in calls] == [False, True]
    assert _state(harness)["cycles_paid"] == 1
    from ouroboros.tools.plan_evidence import resolve_evidence
    ref = first["dialogue_source_ref"]
    locator = f"chat:1@{ref['sha256']}::lines=2-2"
    selected = resolve_evidence([locator], active_root=harness.workspace, allowed_roots=[],
                               resolve_chat=plan_chat_reader(harness.drive, ctx.task_id))["attached"][0]
    assert json.loads(selected["text"])["text"] == "Before dispatch"
    from ouroboros.task_results import _compact_plan_review_wave
    compact = _compact_plan_review_wave(collected)
    assert compact["dialogue_source_ref"] == ref
    assert compact["author_request_fingerprint"] == collected["author_request_fingerprint"]


def test_inline_view_is_the_numbered_conversation(harness):
    """D1: the inline OWN ROOM DIALOGUE is the conversation as numbered readable lines
    (owner rows with attachment names, Ouroboros rows, quiz cards with the chosen answer),
    each line's number being the physical snapshot line a `::lines=` locator addresses;
    progress rows, host system rows and the archive-file list stay behind the pointer.
    Reverted, the packet is the JSONL snapshot with its metadata dump."""
    from ouroboros.tools.plan_dialogue import attach_own_dialogue, dialogue_view, render_dialogue

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    chat = harness.drive / "logs" / "chat.jsonl"
    append_jsonl(chat, {"direction": "in", "chat_id": 1, "ts": "2026-09-01T10:00:00Z", "text": "Use the agreed outline",
                        "attachment_manifest": [{"label": "outline.pdf", "name": "outline.pdf"}]})
    append_jsonl(chat, {"direction": "out", "chat_id": 1, "ts": "2026-09-01T10:01:00Z", "text": "Option A is faster\nOption B keeps choices"})
    append_jsonl(chat, {"direction": "out", "chat_id": 1, "ts": "2026-09-01T10:02:00Z", "type": "quiz", "text": "Five or six slides?",
                        "quiz": {"quiz_id": "q1", "question": "Five or six slides?", "options": ["five", "six"], "recommended_index": 0}})
    append_jsonl(chat, {"direction": "in", "chat_id": 1, "ts": "2026-09-01T10:03:00Z", "type": "quiz_answer", "text": "five",
                        "quiz": {"quiz_id": "q1", "options": ["five", "six"], "answered_index": 0, "comment": "The board asked for five."}})
    append_jsonl(chat, {"direction": "system", "chat_id": 1, "ts": "2026-09-01T10:04:00Z", "type": "task_summary", "text": "HOST SUMMARY ROW"})
    for step in ("Reading notes", "Drafting slide 1", "Drafting slide 2"):
        append_jsonl(harness.drive / "logs" / "progress.jsonl", {"chat_id": 1, "ts": "2026-09-01T10:05:00Z", "content": f"PROGRESS {step}"})
    manifest = attach_own_dialogue(ctx, harness.drive, {}, "a" * 64, persist=True)
    own = manifest["own_dialogue"]
    view, facts = dialogue_view(own)
    packet = render_dialogue(manifest)
    assert view in packet and packet.index("## OWN ROOM DIALOGUE") < packet.index("## RELATED ROOMS")
    snapshot_lines = own["text"].split("\n")
    expected = {"Use the agreed outline": "User", "Option A is faster": "Ouroboros", "Five or six slides?": "Ouroboros", "five": "Owner"}
    for text, author in expected.items():
        line = next(line for line in view.splitlines() if f" · {author} · " in line and text in line)
        number = int(line.split(" · ", 1)[0])
        row = json.loads(snapshot_lines[number - 1])
        assert row["text"].startswith(text) or row.get("quiz", {}).get("question") == text
        assert line.startswith(f"{number} · {row['ts']} · {author} · ")
    assert "[question q1] Five or six slides? — options: (1) five (2) six; recommended (1)" in view
    assert '[answer q1] chose (1) five — "The board asked for five."' in view
    assert "[attachments: outline.pdf]" in view and "\n  Option B keeps choices" in view
    assert facts["conversation_rows"] == 4 and facts["conversation_inline_rows"] == 4 and facts["other_rows"] == 4
    assert "Not inline: 3 progress rows and 1 host rows." in view
    assert f"`{own['locator']}`" in view and own["file"] in view and own["source_ref"]["path"] in view
    assert "::lines=A-B" in view and "Later messages are not claimed reviewed" in view
    for absent in ("PROGRESS ", "HOST SUMMARY ROW", "generations", '"stream":', '"source_ordinal"', "progress.jsonl"):
        assert absent not in view, absent
    # A gap keeps its explicit statement; nothing is rendered as dialogue.
    gap_view, gap_facts = dialogue_view({"chat_id": 1, "gap": "own_room_unavailable", "text": ""})
    assert "Explicit gap: own room source unavailable (own_room_unavailable)" in gap_view and gap_facts["conversation_rows"] == 0


def test_mixed_delivery_keeps_full_file_and_exact_overflow_range(harness, monkeypatch):
    from types import SimpleNamespace
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools.plan_dialogue import attach_own_dialogue, dialogue_slot_inputs, render_dialogue
    from ouroboros.tools import review_synthesis
    from ouroboros.tools.scope_required_sources import source_text_identity
    from ouroboros import review_native_episode

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    append_jsonl(harness.drive / "logs" / "chat.jsonl", {"direction": "out", "chat_id": 1,
                 "text": "OLDER " + "discussion " * 22000})
    append_jsonl(harness.drive / "logs" / "chat.jsonl", {"direction": "in", "chat_id": 1,
                 "text": "LATEST CHOICE"})
    append_jsonl(harness.drive / "logs" / "progress.jsonl", {"chat_id": 1, "content": "PROGRESS ROW " * 50})
    manifest = attach_own_dialogue(ctx, harness.drive, {}, "a" * 64, persist=True)
    own = manifest["own_dialogue"]
    packet = render_dialogue(manifest)
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
        return 100000
    monkeypatch.setattr(review_native_episode, "review_native_transcript_bound", bound)
    delivery = dialogue_slot_inputs(slots, system_prompt="governance", user_content=packet,
                                   session_task=packet, manifest=manifest, slot_messages={}, native_mandatory_chars=len(packet), session_root=str(harness.workspace), task_id=ctx.task_id)
    small = json.dumps(delivery["slot_messages"]["small"], ensure_ascii=False)
    large = json.dumps(delivery["slot_messages"]["large"], ensure_ascii=False)
    assert "LATEST CHOICE" in small and "exact omitted prefix" in small
    assert "1 earlier conversation rows before line 3" in small and "OLDER " not in small and "OLDER " in large
    coverage = delivery["dialogue_delivery"]
    assert coverage["small"]["conversation_inline_rows"] == 1 and coverage["small"]["conversation_first_inline_line"] == 3
    assert coverage["large"]["conversation_inline_rows"] == 2 and coverage["large"]["delivery"] == "packet"
    native = delivery["slot_session_tasks"]["native"]
    assert "LATEST CHOICE" in native and "exact omitted prefix" in native and coverage["native"]["delivery"] == "native_retrieving"
    assert declarations[0]["mandatory_read_chars"] == len(packet)
    delegated = delivery["slot_session_tasks"]["delegated"]
    # The delegated session gets the same inline conversation (no invented window) plus the pointer.
    assert "OLDER " in delegated and "LATEST CHOICE" in delegated and "MANDATORY FULL READ" not in delegated
    assert coverage["delegated"] == {**coverage["large"], "delivery": "delegated_session", "window": "unasserted", "file": own["file"]}
    assert delivery["request_policy"]["observed_sources"] == [{
        "root": "artifact_store", "path": own["source_ref"]["path"], "file": own["file"],
        **source_text_identity(own["text"].encode("utf-8"))}]
    # Every slot's send ends with ITS OWN panel seat: a shared packet cannot say "your".
    from tests.test_plan_review_engine import _user_text
    last_user = {sid: _user_text(delivery["slot_messages"][sid][-1]["content"]) for sid in ("small", "large")}
    seat = {sid: f"\n## YOUR PANEL SEAT\n\n`{sid}`\n" for sid in last_user}
    assert all(last_user[sid].endswith(seat[sid]) for sid in last_user)
    assert last_user["large"].removesuffix(seat["large"]) == packet  # the seat is the only addition
    assert native.endswith("\n## YOUR PANEL SEAT\n\n`native`\n") and delegated.endswith("\n## YOUR PANEL SEAT\n\n`delegated`\n")
    assert own["file"] in delegated and "1M" not in delegated
    source = read_actor_source_bytes(harness.drive, ctx.task_id, own["source_ref"])
    assert len(source) > 120000 and source.decode() == own["text"]
    import re
    match = re.search(r'::lines=2-(\d+)', native)
    assert match and int(match[1]) == 2  # the OLDER row (line 2) is the exact omitted prefix
    assert not (harness.workspace / ".ouroboros-review").exists()


def test_a_session_with_an_asserted_window_is_fitted_and_without_one_is_whole(harness, monkeypatch):
    """A delegated session's inline conversation is fitted only to an owner-asserted
    `reviewer:<slot>` context window; with none asserted the host invents no window and
    sends the whole conversation with the pointer."""
    from types import SimpleNamespace
    from ouroboros import model_slots
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools.plan_dialogue import attach_own_dialogue, dialogue_slot_inputs, render_dialogue

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    append_jsonl(harness.drive / "logs" / "chat.jsonl", {"direction": "out", "chat_id": 1, "text": "OLDER " + "discussion " * 22000})
    append_jsonl(harness.drive / "logs" / "chat.jsonl", {"direction": "in", "chat_id": 1, "text": "LATEST CHOICE"})
    manifest = attach_own_dialogue(ctx, harness.drive, {}, "b" * 64, persist=True)
    packet = render_dialogue(manifest)
    session = SimpleNamespace(slot_id="delegated", model="cursor=grok", role_hint="plan reviewer", use_local=False,
                              session_profile="", route=ReviewRouteKind.AGENT_SESSION, retrieves=True, native_retrieval=False)
    monkeypatch.setattr(model_slots, "model_role_option", lambda key, role: 40_000 if role == "reviewer:delegated" else 0)
    fitted = dialogue_slot_inputs([session], system_prompt="g", user_content=packet, session_task=packet, manifest=manifest,
                                  slot_messages={}, native_mandatory_chars=len(packet), session_root=str(harness.workspace), task_id=ctx.task_id)
    task = fitted["slot_session_tasks"]["delegated"]
    assert "LATEST CHOICE" in task and "OLDER " not in task and "1 earlier conversation rows before line 3" in task
    assert fitted["dialogue_delivery"]["delegated"]["window"] == "asserted"
    monkeypatch.setattr(model_slots, "model_role_option", lambda key, role: 0)
    whole = dialogue_slot_inputs([session], system_prompt="g", user_content=packet, session_task=packet, manifest=manifest,
                                 slot_messages={}, native_mandatory_chars=len(packet), session_root=str(harness.workspace), task_id=ctx.task_id)
    assert "OLDER " in whole["slot_session_tasks"]["delegated"] and whole["dialogue_delivery"]["delegated"]["window"] == "unasserted"


def test_a_session_reviewers_room_read_rides_its_actor_row_and_the_verdict_text():
    """The observed-source fold reaches the wave as a fact: attestation `harness_observed`,
    `room_read_coverage` on the actor row and record, and one clause in the rendered verdict;
    a session with no fold stays `unobserved` with no coverage key."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.tools.plan_render import _render_wave
    from ouroboros.tools.plan_review_runtime import _plan_row_from_actor, plan_wave_actor_record

    slot = ReviewSlot(slot_id="s1", model="cursor=grok", route=ReviewRouteKind.AGENT_SESSION)
    usage = {"read_provenance": "harness_observed", "native_read_coverage": {"status": "complete", "sources": [
        {"root": "artifact_store", "path": "plan-dialogue-1.jsonl", "status": "complete", "covered_chars": 120, "complete_chars": 120}]}}
    actor = {"slot_id": "s1", "model": "cursor=grok", "status": "ok", "raw_text": "[]\nNO_FINDINGS", "usage": usage,
             "operation_id": "op-s1", "operation_state": "settled"}
    row = _plan_row_from_actor(actor, slot)
    coverage = {"status": "complete", "covered_chars": 120, "complete_chars": 120, "provenance": "harness_observed"}
    assert row["host_file_read_attestation"] == "harness_observed" and row["room_read_coverage"] == coverage
    record = plan_wave_actor_record(row, ok=True, error="", disclosures=[], raw_text_preview_chars=10)
    assert record["room_read_coverage"] == coverage
    text = _render_wave({"aggregate": "GREEN", "closed": True, "request_fingerprint": "f" * 64, "actors": [record],
                         "findings": [], "counts": {}, "reasons": []}, cap=2, cycles_paid=1, enforcement="advisory")
    assert "host_file_read: harness_observed · room snapshot read 120/120 chars (harness_observed) · ok" in text
    plain = _plan_row_from_actor({**actor, "usage": {}}, slot)
    assert plain["host_file_read_attestation"] == "unobserved" and "room_read_coverage" not in plain


def test_dialogue_source_survives_real_child_promotion_and_cleanup(harness):
    from ouroboros.headless import prepare_task_drive, copy_child_task_result, remove_subagent_task_drive
    from ouroboros.task_results import write_task_result, load_plan_review_state
    from ouroboros.tools.plan_review_artifacts import authority_wave

    ctx = harness.make_ctx(task_id="source")
    parent = harness.drive
    child = prepare_task_drive(parent, "source", "empty")
    ctx.drive_root = child
    ctx.current_chat_id = 1
    from tests.test_plan_review_engine import _finding
    from ouroboros.tools import plan_review as pr
    harness.install({"s1": json.dumps([_finding("note", "note")]), "s2": CLEAN, "s3": CLEAN})
    append_jsonl(child / "logs" / "chat.jsonl", {"direction": "in", "chat_id": 1, "text": "Must survive cleanup"})
    _call(ctx)
    first = load_plan_review_state(child, "source")["waves"][-1]
    raw = read_actor_source_bytes(child, "source", first["dialogue_source_ref"])
    predecessor = first["wave_artifact"]
    pr._handle_plan_task(ctx, review_disposition={"review_fingerprint": first["request_fingerprint"],
        "items": [{"finding_id": "s1:note", "decision": "defer", "rationale": "Later cosmetic work"}]})
    write_task_result(child, "source", "completed")
    copied = copy_child_task_result(parent, {"id": "source", "drive_root": str(child)})
    assert copied["child_ref_promotion"]["status"] == "complete"
    assert remove_subagent_task_drive(parent, "source", live=lambda _task: False) is True
    assert not child.exists()
    wave = load_plan_review_state(parent, "source")["waves"][-1]
    assert read_actor_source_bytes(parent, "source", wave["dialogue_source_ref"]) == raw
    restored = authority_wave(parent, "source", wave)
    assert restored["evidence_manifest_full"]["own_dialogue"]["file"].startswith(str(parent))
    from ouroboros.tools.plan_review_artifacts import read_wave
    exact = read_wave(parent, "source", wave["wave_artifact"])
    assert exact["supersedes_wave_artifact"]["sha256"] == predecessor["sha256"]
    assert read_wave(parent, "source", exact["supersedes_wave_artifact"])["plan_prose"]


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
    from ouroboros.tools import plan_review as pr, review_synthesis
    from ouroboros import usage_accounting as ua
    from tests.test_plan_review_engine import _user_text
    from ouroboros.tools.plan_review_runtime import PLAN_REVIEW_MAX_TOKENS

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    append_jsonl(harness.drive / 'logs/chat.jsonl', {'ts': '2026-09-01T00:00:00Z',
        'direction': 'out', 'chat_id': 1, 'text': 'Prior substantive discussion. ' * 20000})
    monkeypatch.setattr(review_synthesis, 'per_slot_input_token_limits',
                        lambda models, **kw: {slot.slot_id: 10000 for slot in kw['slots']})
    substrate = harness.install({'s1': CLEAN, 's2': CLEAN, 's3': CLEAN})
    captured = []
    monkeypatch.setattr(pr, 'review_wave_budget_gate', lambda *a, **kw: captured.append(kw))
    _call(ctx)
    request = substrate.calls[0]['request']
    slots = substrate.calls[0]['slots']
    actual = [sum(len(_user_text(message['content'])) for message in request.slot_messages[slot.slot_id]) for slot in slots]
    assert captured[0]['prompt_chars'] == actual
    assert captured[0]['max_completion_tokens'] == PLAN_REVIEW_MAX_TOKENS
    # Illustrative price replaces only vendor lookup, not the admission math.
    monkeypatch.setattr(ua, 'estimate_cost_optional', lambda model, prompt, completion, **kw: prompt / 100000)
    admission = ua.review_wave_admission(root_task_id='budget-probe', models=[slot.model for slot in slots],
        prompt_chars=captured[0]['prompt_chars'], max_completion_tokens=PLAN_REVIEW_MAX_TOKENS, remaining_usd_override=1.0)
    # The conversation-only view keeps whole rows: this room's one oversize row fits no
    # 10k-token slot, so every packet is priced on the small pointer view, never on a
    # byte tail that filled the window.
    assert admission['fits'] is True and 0 < admission['estimated_wave_usd'] < 0.3
    assert len(substrate.calls) == 1


def test_free_collection_reuses_policy_after_live_exploration_changes(harness, monkeypatch):
    from tests.test_plan_review_event_route import _install_real_substrate, _wait_until, _mailbox_entries
    from ouroboros.tools import plan_review_runtime
    from tests.test_plan_review_engine import _control
    live = ['Read the initial discussion.']
    monkeypatch.setattr(plan_review_runtime, 'root_exploration_log', lambda _ctx: live[0])
    executor = _install_real_substrate(monkeypatch)  # Only the model executor is fake; custody is real.
    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    try:
        _call(ctx)
        first = _state(harness)['waves'][-1]
        from ouroboros.tools.plan_review_artifacts import read_wave
        sent = read_wave(harness.drive, ctx.task_id, first['wave_artifact'])
        assert sent['request_policy']['native_mandatory_read_chars'] > 0
        assert _wait_until(lambda: executor.execute_calls == 3)
        live[0] += ' Processed an owner clarification and waited for the panel.'
        executor.release.set()
        assert _wait_until(lambda: len(_mailbox_entries(harness.drive, ctx.task_id)) == 1)
        collected = _collect(ctx, first['request_fingerprint'])
        assert _control(collected) == {'outcome': 'GREEN', 'closed': True}
        assert executor.execute_calls == 3 and _state(harness)['cycles_paid'] == 1
        assert 'custody is unavailable' not in collected
        settled = read_wave(harness.drive, ctx.task_id, _state(harness)['waves'][-1]['wave_artifact'])
        assert settled['request_policy'] == sent['request_policy']
        assert settled['slot_prompt_chars'] == sent['slot_prompt_chars']
        assert [row['request_messages'] for row in settled['reviewer_outputs']] == [row['request_messages'] for row in sent['reviewer_outputs']]
    finally:
        executor.release.set()


def test_missing_recorded_policy_does_not_infer_current_paid_contract():
    import pytest
    from ouroboros.tools.plan_review_artifacts import frozen_delivery_inputs, PlanReviewSourceUnavailable
    with pytest.raises(PlanReviewSourceUnavailable, match='original request policy/fit was not recorded'):
        frozen_delivery_inputs({'reviewer_outputs': []}, [])


def test_requested_related_room_replay_keeps_one_physical_panel_per_cycle(harness, monkeypatch):
    from ouroboros.review_execution import ReviewAttemptResult
    from ouroboros.projects_registry import create_project, bind_task_to_project
    from tests.test_plan_review_engine import _finding, _slots
    from tests.test_plan_review_event_route import _HeldExecutor, _wait_until, _mailbox_entries
    from ouroboros.tools import review_synthesis

    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "3")
    project = create_project(harness.drive, "own", name="Own room")
    ctx = harness.make_ctx()
    bind_task_to_project(harness.drive, ctx.task_id, project["id"], origin={"absent": "system"})
    append_jsonl(harness.drive / "logs/chat.jsonl", {"chat_id": 1, "text": "Main premise"})
    harness.state["slots"] = _slots(("s1", "m/a"))
    monkeypatch.setattr(review_synthesis, "per_slot_input_token_limits",
                        lambda models, *, slots, **kw: {slot.slot_id: 800000 for slot in slots})

    class Executor(_HeldExecutor):
        def execute(self):
            self.execute_calls += 1
            answer = (json.dumps([_finding("main", "need_evidence", locator="chat:1")])
                      if self.execute_calls == 1 else CLEAN)
            return ReviewAttemptResult(message={"content": answer}, raw_text=answer,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "physical_attempt_state": "settled"})

    executor = Executor()
    monkeypatch.setattr("ouroboros.review_substrate._review_route_executor", lambda *a, **k: executor)
    for cycle in (1, 2):
        _call(ctx)
        wave = _state(harness)["waves"][-1]
        assert _wait_until(lambda: len(_mailbox_entries(harness.drive, ctx.task_id)) >= cycle)
        _collect(ctx, wave["request_fingerprint"])
    second = _state(harness)["waves"][-1]
    replay = _call(ctx)
    current = _state(harness)["waves"][-1]
    if current["request_fingerprint"] != second["request_fingerprint"]:
        assert _wait_until(lambda: len(_mailbox_entries(harness.drive, ctx.task_id)) >= 3)
        _collect(ctx, current["request_fingerprint"])
    assert "cached exact review" in replay
    assert current["request_fingerprint"] == second["request_fingerprint"]
    assert executor.execute_calls == 2 and _state(harness)["cycles_paid"] == 2
    # A real update to requested evidence still earns the existing W3 refresh.
    append_jsonl(harness.drive / "logs/chat.jsonl", {"chat_id": 1, "text": "Main premise changed"})
    _call(ctx)
    changed = _state(harness)["waves"][-1]
    assert _wait_until(lambda: len(_mailbox_entries(harness.drive, ctx.task_id)) >= 3)
    _collect(ctx, changed["request_fingerprint"])
    assert changed["request_fingerprint"] != second["request_fingerprint"]
    assert executor.execute_calls == 3 and _state(harness)["cycles_paid"] == 3


def test_snapshot_qualified_room_keeps_original_gap_disclosure(harness):
    from ouroboros.tools.plan_evidence import resolve_evidence

    ctx = harness.make_ctx()
    ctx.current_chat_id = 1
    harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    chat = harness.drive / "logs/chat.jsonl"
    append_jsonl(chat, {"chat_id": 1, "text": "Retained explanation"})
    with chat.open("a") as stream:
        stream.write("{unreadable history row\n")
    _call(ctx)
    wave = _state(harness)["waves"][-1]
    ref = wave["dialogue_source_ref"]
    original = read_actor_source_bytes(harness.drive, ctx.task_id, ref)
    append_jsonl(chat, {"chat_id": 1, "text": "Later room growth"})
    reader = plan_chat_reader(harness.drive, ctx.task_id)
    exact = reader(f"1@{ref['sha256']}")
    assert exact["text"].encode() == original
    assert exact["coverage"]["chat"]["gaps"] and exact["coverage"]["chat"]["snapshot_stable"]
    manifest = resolve_evidence([f"chat:1@{ref['sha256']}::lines=2-2"],
        active_root=harness.workspace, allowed_roots=[], resolve_chat=reader)
    assert json.loads(manifest["attached"][0]["text"])["text"] == "Retained explanation"
    assert any(row["reason"].startswith("chat_history_gap:") for row in manifest["omissions"])


def test_the_seat_line_follows_the_cache_stable_prefix_on_every_api_slot(harness, monkeypatch):
    """The per-slot seat is appended AFTER `## ROOT EXPLORATION LOG` (the cache-stable prefix
    stays byte-identical across api slots) and each slot gets its own id; reverted, no seat."""
    from types import SimpleNamespace
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools import plan_spec, review_synthesis
    from ouroboros.tools.plan_dialogue import dialogue_slot_inputs
    from ouroboros.tools.plan_packet import build_plan_review_user_content, plan_user_stable_len
    from tests.test_plan_review_engine import DECK_SPEC, _user_text

    spec, _ = plan_spec.normalize_spec({**DECK_SPEC, "goal": "Ship the deck"})
    packet = build_plan_review_user_content(
        objective="o", goal=spec["goal"], plan_prose="p", spec=spec,
        manifest={"declared": [], "attached": [], "omissions": []},
        prior_cycles=[], dispositions=[], spec_delta=None, root_exploration_log="ran: ls")
    def slot(name):
        return SimpleNamespace(slot_id=name, model="same/model", role_hint="plan reviewer", use_local=False,
                               session_profile=name, route=ReviewRouteKind.API_CHAT, retrieves=False, native_retrieval=False)
    monkeypatch.setattr(review_synthesis, "per_slot_input_token_limits", lambda *a, **k: {"a": 200000, "b": 200000})
    delivery = dialogue_slot_inputs([slot("a"), slot("b")], system_prompt="governance", user_content=packet,
                                    session_task=packet, manifest={}, slot_messages={}, native_mandatory_chars=len(packet),
                                    session_root=str(harness.workspace), task_id="task-1")
    sent = {sid: _user_text(delivery["slot_messages"][sid][-1]["content"]) for sid in ("a", "b")}
    boundary = plan_user_stable_len(packet)
    assert boundary > 0 and sent["a"][:boundary] == sent["b"][:boundary] == packet[:boundary]
    assert sent["a"].endswith("\n## YOUR PANEL SEAT\n\n`a`\n") and sent["b"].endswith("\n## YOUR PANEL SEAT\n\n`b`\n")
    assert sent["a"].index("## YOUR PANEL SEAT") > sent["a"].index("## ROOT EXPLORATION LOG")
    # The recorded cache split is the same boundary: stable block, then the dynamic tail with the seat.
    blocks = delivery["slot_messages"]["a"][-1]["content"]
    assert isinstance(blocks, list) and blocks[0]["text"] == packet[:boundary] and blocks[-1]["text"].endswith("`a`\n")
