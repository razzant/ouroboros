"""Ordinary root render and local physical dispatch, with only the server replaced."""
import hashlib
import json
import socket
from types import SimpleNamespace

import pytest

from ouroboros import llm_local, local_model, nano_minimum, usage_accounting as usage
from ouroboros.context_budget import LocalContextTooLargeError
from ouroboros.deployment_policy import MAX_RULES_BYTES, parse_policy
from ouroboros.llm import LLMClient
from ouroboros.local_model_server import input_fingerprint
from ouroboros.send_clock import SendClockPolicy, main_send_scope
from tests.test_nano_minimum import COUNTER, OWNER_INPUT, PORT, TOOLS, WINDOW


@pytest.fixture
def root_capture(tmp_path, monkeypatch):
    from ouroboros import context, deployment_policy
    from ouroboros.chronicle_store import ChronicleStore
    from tests._memory_view_context import world

    env, memory, rooms = world(tmp_path)
    ChronicleStore(memory.drive_root).ensure_activated()
    task = {"type": "task", "id": "synthetic-root", "chat_id": rooms["alpha"], "text": OWNER_INPUT}

    def capture(rules="Stay within the assignment.", *, tools=TOOLS):
        policy = parse_policy(json.dumps({"version": 1, "rules": rules}).encode())
        monkeypatch.setattr(deployment_policy, "organization_rules", lambda: policy.rules)
        core = context._capture_context_core(env, memory, task, None, None)
        return context._build_context_fit_plan(env, core, task, preferred_mode="nano", tool_schemas=tools,
            route_resolver=lambda *_a, **_kw: ({"model": "local-model", "provider": "local"}, SimpleNamespace(
                route_fp="fixture", status="asserted", stale=False, window_tokens=WINDOW)))

    return capture, memory.drive_root


@pytest.fixture
def synthetic_sender(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Synthetic root test tried to open a real connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    counted, sent, records = [], [], []
    # Every input byte contributes. The pad is made of repeated '!4z?' words;
    # for this synthetic tokenizer that particular spelling counts one token
    # per character, so the example does not depend on chars/4 for that input.
    def counter(payload, *, expected_port):
        assert expected_port == PORT
        blob = json.dumps({key: payload.get(key) for key in ("messages", "tools", "tool_choice")},
                          ensure_ascii=False, sort_keys=True).encode()
        hits = blob.count(b"!4z?") * 4
        tokens = 29_000 + (len(blob) - hits) // 4 + hits
        counted.append((payload, tokens))
        return {"supported": True, "input_is_exact": True, "input_tokens": tokens, "context_window": WINDOW,
                "raw_context_window": WINDOW, "process_id": 19, "native_input_sha256": input_fingerprint(payload),
                "output_limit_enforced": True, "reasoning_included_in_limit": True,
                "tokenizer_template_provenance": {**COUNTER, "rendered_prompt_sha256": hashlib.sha256(blob).hexdigest()}}

    def receiver(**body):
        sent.append(body)
        return SimpleNamespace(model_dump=lambda: {"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    client = SimpleNamespace(base_url=f"http://127.0.0.1:{PORT}/v1",
                             chat=SimpleNamespace(completions=SimpleNamespace(create=receiver)))
    monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(
        measure_prepared_input=counter,
        serving_context_evidence=lambda **_: {"process_id": 19, "context_window": WINDOW, "confirmed": True, "port": PORT}))
    monkeypatch.setattr(LLMClient, "_get_local_client", lambda _self: client)
    monkeypatch.setattr(llm_local, "limit_physical_send", lambda _target, send: send)

    def execute(request, send, before):
        records.append(request.physical_context.nano_floor_evidence)
        return send()

    monkeypatch.setattr(llm_local, "_execute_candidate", execute)
    return counted, sent, records


def _send(plan, rows, episode, tools=TOOLS):
    frame = usage.PhysicalAttemptContext("owner_nano", "nano", "cold_estimate", "fixture", episode, WINDOW,
                                         WINDOW, False, False, preferred_mode="nano", nano_floor_plan=(plan, episode))
    with usage.bind_physical_attempt_context(frame), main_send_scope(SendClockPolicy()):
        return LLMClient(api_key="unused")._chat_local(rows, tools, 60_000, "auto")


def test_work_can_use_existing_room_without_a_second_reserve(root_capture, synthetic_sender):
    episode = "cold_start"  # a synthetic entry label, not a lifecycle replay
    capture, root = root_capture
    identity = "!4z?" * 4_850  # fill the budgeted memory near 20k, rather than testing an empty profile
    (root / "memory" / "identity.md").write_text(identity, encoding="utf-8")
    root_plan = capture()
    counted, sent, records = synthetic_sender
    rows = root_plan.messages_for("nano")
    first, _ = _send(root_plan, rows, episode)
    before = records[-1]
    assert before["status"] == "exact"
    assert 19_000 <= before["memory_tokens"] <= 20_000
    assert identity in sent[-1]["messages"][0]["content"], "the positive case must send the whole identity"
    counted.clear()
    work = "!4z?" * 3_750 + " preserved work marker"
    # Keep ordinary assignment and add one distinguishable work row, rather than replacing its source.
    full = [*rows, {"role": "user", "content": work}]
    second, _ = _send(root_plan, full, episode)
    after = records[-1]
    assert first["content"] == second["content"] == "ok"
    assert len(sent) == 2 and len(counted) == 3
    assert [row["content"] for row in sent[-1]["messages"]].count(work) == 1
    assert any(OWNER_INPUT in row["content"] for row in sent[-1]["messages"])
    assert after["status"] == "exact" and after["scope"] == episode
    assert after["actual"]["native_input_sha256"] == input_fingerprint(sent[-1])
    assert after["actual"]["input_tokens"] - before["actual"]["input_tokens"] >= 15_000
    assert after["actual"]["sent_max_tokens"] < before["actual"]["sent_max_tokens"]
    assert after["actual"]["input_tokens"] + sent[-1]["max_tokens"] == WINDOW
    assert sent[-1]["max_tokens"] >= 8_192
    assert after["minimum_work_headroom_at_reply_floor_tokens"] < 15_000 <= before["minimum_work_headroom_at_reply_floor_tokens"]
    assert after["without_memory"]["input_tokens"] > before["without_memory"]["input_tokens"] + 14_990
    print(episode, before["actual"]["input_tokens"], after["actual"]["input_tokens"],
          before["actual"]["sent_max_tokens"], after["actual"]["sent_max_tokens"],
          before["without_memory"]["input_tokens"], before["memory_tokens"])


def test_non_memory_rules_can_break_the_envelope_without_a_memory_write(root_capture, synthetic_sender):
    capture, _root = root_capture
    counted, sent, records = synthetic_sender
    previous = capture()
    _send(previous, previous.messages_for("nano"), "cold_start")
    baseline = records[-1]
    rules = "!4z?" * (MAX_RULES_BYTES // 4)
    changed = capture(rules)
    section = next(part for part in changed.core.semi_stable_sections if part.section_id == "organization_rules")
    assert section.operative_view is None
    rows = changed.messages_for("nano")
    pair, gap = nano_minimum.render_minimum(changed, rows, window=WINDOW, tool_names=("memory_read",))
    assert not gap
    assert rules in "\n".join(block["text"] for block in pair[1][0]["content"])
    with pytest.raises(LocalContextTooLargeError, match="below the floor of 8192") as refusal:
        _send(changed, rows, "cold_start")
    print("non_memory_rules", len(rules.encode()), counted[-1][1], refusal.value)
    assert len(sent) == 1 and len(records) == 1
    assert counted[-1][1] > WINDOW - baseline["reply_floor_tokens"]
    assert any(rules in row["content"] for row in counted[-1][0]["messages"])
    assert any(OWNER_INPUT in row["content"] for row in counted[-1][0]["messages"])


@pytest.mark.parametrize("path", ["prompts/SYSTEM.md", "BIBLE.md", "docs/ARCHITECTURE.md"])
def test_governance_growth_uses_room_without_changing_memory(root_capture, synthetic_sender, path):
    """A synthetic formatter must charge fixed governance, including the shown book form."""
    capture, root = root_capture
    _, sent, records = synthetic_sender
    source = root.parent / "repo" / path
    original = source.read_text(encoding="utf-8")
    # Keep the same number of headings and the same book form before and after.
    stem = original + "\n\n## Example\n"
    source.write_text(stem, encoding="utf-8")
    before = capture()
    _send(before, before.messages_for("nano"), "cold_start")
    # Keep the additional length in a heading for the book: Nano reads its map, not its body.
    added = "!4z?" * 1_000
    source.write_text(stem.rstrip("\n") + " " + added + "\n", encoding="utf-8")
    after = capture()
    _send(after, after.messages_for("nano"), "cold_start")
    old, new = records
    assert all(row["status"] == "exact" for row in (old, new))
    assert new["minimum"]["input_tokens"] - old["minimum"]["input_tokens"] >= 4_000
    assert new["minimum_work_headroom_at_reply_floor_tokens"] <= old["minimum_work_headroom_at_reply_floor_tokens"] - 4_000
    assert abs(new["memory_tokens"] - old["memory_tokens"]) <= 1  # mock formatter rounds the two counts separately
    assert any(added in row["content"] for row in sent[-1]["messages"])


def test_existing_tree_coordination_consumes_work_room_without_spending_memory_budget(root_capture, synthetic_sender):
    from ouroboros.task_tree_ledger import tree_ledger_append, tree_ledger_rows

    capture, root = root_capture
    counted, sent, records = synthetic_sender
    first = capture()
    _send(first, first.messages_for("nano"), "cold_start")
    starting = records[-1]
    for idx in range(11):
        result = tree_ledger_append("synthetic-root", "note", f"Observation {idx}: " + "!4z?" * 980,
                                    task_id="synthetic-root", data_root=root)
        assert not result.startswith("⚠️"), result
    originals = [row["text"] for row in tree_ledger_rows("synthetic-root", data_root=root)]
    assert len(originals) == 11 and all(len(text) < 4_000 for text in originals)
    grown = capture()
    rows = grown.messages_for("nano")
    pair, gap = nano_minimum.render_minimum(grown, rows, window=WINDOW, tool_names=("memory_read",))
    assert not gap
    for n in pair:
        system = "\n".join(block["text"] for block in n[0]["content"])
        assert all(system.count(text) == 1 for text in originals)
    with pytest.raises(LocalContextTooLargeError, match="below the floor of 8192") as refusal:
        _send(grown, rows, "cold_resume")
    print("coordination_growth", starting["without_memory"]["input_tokens"], counted[-1][1], refusal.value)
    assert len(sent) == 1 and len(records) == 1
    assert all(text in counted[-1][0]["messages"][0]["content"] for text in originals)
    assert [row["text"] for row in tree_ledger_rows("synthetic-root", data_root=root)] == originals


@pytest.mark.parametrize(("rules_chars", "read_file"), [
    (0, True), (2_000, True), (2_000, False), (MAX_RULES_BYTES, True),
], ids=["ordinary", "world-reader-required", "no-world-reader", "organization-does-not-fit"])
def test_ordinary_round_refits_or_refuses_without_losing_sources(
        root_capture, synthetic_sender, monkeypatch, tmp_path, rules_chars, read_file):
    from ouroboros import loop, peer_roster
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.focus import normalize_focus
    from ouroboros.live_works import WORKS_LINES_HEADING, WORKS_LINES_LEGEND, live_works_present
    from tests.test_live_works_line_form import _rebuild, _without_handle_path
    from tests.test_live_works_present import _runtime
    from ouroboros.loop_llm_call import _record_llm_call_error, _LlmErrorContext
    from ouroboros.usage_accounting import current_physical_attempt_predicate, PhysicalAttemptPreconditionFailed
    from tests.test_context_refused_density import _round

    capture, root = root_capture
    identity = "!4z?" * 4_850
    (root / "memory" / "identity.md").write_text(identity, encoding="utf-8")
    ref = {"kind": "task_source", "root": "artifact_store", "path": "source_handles/context_checkpoints/fact.md",
           "size": 1, "sha256": "a" * 64}
    focus = normalize_focus("Keep the live work's own decision", {"reader": "recent_tasks"}, task_id="other",
                            authored_at="2026-01-01T00:00:00Z", source_handle=ref)
    roster = {"roots": [{"task_id": "other", "status": "running", "title": "An independent work", "focus": focus}],
              "incomplete": False, "queue_snapshot": {}}
    foreign = [{"work_id": f"done-{i:03}", "title": f"!4z?{'!4z?' * 200} {i}", "room": {},
                "root_status": "completed", "helpers_running": 1, "helpers_waiting": 0,
                "tails": [{"kind": "helper", "state": "live", "code": "helpers_running", "count": 1,
                           "text": "one live helper", "ids": [f"child-{i:03}"]}]} for i in range(30)]
    monkeypatch.setattr(peer_roster, "independent_roots", lambda _root: roster)
    monkeypatch.setattr(peer_roster, "unlisted_live_work", lambda *_args: foreign)
    tools = [*TOOLS]
    if read_file:
        tools.append({"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}})
    # Use a synthetic optional WORLD body; the gap is larger than host-path/clock formatting differences.
    world_detail = "Plain world detail stays visible. " + ("!4z?" * 620 if rules_chars == 2_000 else "")
    (root / "memory" / "WORLD.md").write_text("# World\n\n## Inventory\n" + world_detail, encoding="utf-8")
    rules = "!4z?" * (rules_chars // 4) if rules_chars else "Stay within the assignment."
    plan = capture(rules, tools=tools)
    original = next(part for part in plan.core.dynamic_sections if part.section_id == "runtime")
    world = next(part for part in plan.core.semi_stable_sections if part.section_id == "world")
    [work_fact] = live_works_present(root)["works"]
    assert work_fact["focus"] == focus
    assert len(_runtime(plan.messages_for("nano"))["live_works_present"]["other_live_work"]) == len(foreign)
    counted, sent, manifests = synthetic_sender
    ctx = _round(tmp_path, plan, mode="nano")
    ctx.task_id = "synthetic-root"
    ctx.tool_schemas = tools
    ctx.active_use_local = True
    ctx.llm = LLMClient(api_key="unused")
    work = "!4z?" * 3_750 + " work remains whole"
    ctx.messages.append({"role": "user", "content": work})
    refused = []
    # Leave the round controller and its strict-shrink predicate live; substitute the loop's
    # transport wrapper only so this isolated fixture owns no billing or process slots.
    def transport(llm, rows, _model, schemas, *_args, **kw):
        physical = kw["physical_context"]
        checks = kw.get("candidate_predicate"), kw["physical_fit_predicate"]
        def permit(request):
            return all(check is None or check(request) for check in checks)
        try:
            with usage.bind_physical_attempt_context(physical, candidate_predicate=permit), main_send_scope(SendClockPolicy()):
                return llm._chat_local(rows, schemas, kw.get("max_tokens", 60_000), "auto")[0], 0.0
        except LocalContextTooLargeError as exc:
            refused.append(exc.refused_candidate)
            _record_llm_call_error(exc, _LlmErrorContext(task_id=ctx.task_id, task_type="task",
                execution_id="e", round_id=physical.round_id, llm_call_id="c1", round_idx=ctx.round_idx,
                attempt=0, model=ctx.active_model, request_ref=None, drive_logs=ctx.drive_logs,
                event_queue=None, accumulated_usage=ctx.accumulated_usage))
            return None, 0.0

    def execute(request, send, before):
        permit = current_physical_attempt_predicate()
        if permit is not None and not permit(request):
            raise PhysicalAttemptPreconditionFailed("synthetic dispatch did not pass normal round predicates")
        manifests.append(request.physical_context.nano_floor_evidence)
        return send()

    monkeypatch.setattr(loop, "call_llm_with_retry", transport)
    monkeypatch.setattr(llm_local, "_execute_candidate", execute)
    answer, _, mode = loop._call_round_model(ctx)
    print("round_org_bytes", rules_chars, "read_file", read_file, "counts", [tokens for _payload, tokens in counted],
          "refusals", [row["input_tokens"] for row in refused], "generations", len(sent))
    assert mode == "nano" and refused and all(row["input_tokens"] > WINDOW - 8_192 for row in refused)
    if rules_chars == MAX_RULES_BYTES or not read_file:
        assert answer is None and not sent and not manifests
        assert all(rules in payload["messages"][0]["content"] for payload, _ in counted)
        assert all(any(OWNER_INPUT in row["content"] for row in payload["messages"]) for payload, _ in counted)
        return
    world_needed = rules_chars == 2_000
    assert answer["content"] == "ok" and len(sent) == 1
    assert len(refused) == (2 if world_needed else 1)
    # Failed actual candidates, one accepted actual, then the two count-only copies.
    assert len(counted) == len(refused) + 3
    [wire] = sent
    assert len(manifests) == 1 and manifests[0]["status"] == "exact"
    assert manifests[0]["actual"]["native_input_sha256"] == input_fingerprint(wire)
    assert input_fingerprint(wire) == input_fingerprint(counted[-3][0])
    assert counted[-3][1] + wire["max_tokens"] <= WINDOW and wire["max_tokens"] >= 8_192
    assert (world_detail in wire["messages"][0]["content"]) == (not world_needed)
    if world_needed:
        assert manifests[0]["minimum"]["native_input_sha256"] == input_fingerprint(wire)
        assert manifests[0]["minimum"]["input_tokens"] == counted[-3][1]
    # Decode the work rows in the sent Runtime section; its retained full text is read back below.
    [carrier] = [row["content"] for row in wire["messages"] if WORKS_LINES_HEADING in row["content"]]
    assert carrier.count(WORKS_LINES_HEADING) == 1
    body = carrier.split("## Runtime context\n\n", 1)[1]
    runtime, end = json.JSONDecoder().raw_decode(body)
    assert runtime["live_works_present"]["works"] == "one line per work, after this object"
    opening = f"\n\n{WORKS_LINES_HEADING}{WORKS_LINES_LEGEND}\n"
    assert body[end:].startswith(opening)
    assert ("get_task_result(task_id=<task_id>, include_focus_source=True, "
            "focus_source_sha256=<focus_sha256>)") in opening
    [line] = body[end + len(opening):].split("\n\n", 1)[0].split("\n")
    assert _rebuild(line) == _without_handle_path(work_fact)
    assert _runtime([{"content": original.full_text}])["live_works_present"]["works"] == [work_fact]
    assert any("alpha again" in row["content"] for row in wire["messages"])
    saved = {row["section_id"]: row for row in ctx.context_fit_plan.projection("nano").memory_facts["host_sections_addressed"]}
    for section in (original, world) if world_needed else (original,):
        receipt = saved[section.section_id]
        assert receipt["reader"] in wire["messages"][0]["content"]
        assert read_actor_source_bytes(root, ctx.task_id, receipt["source_ref"]).decode() == section.full_text
    assert ("world" in saved) is world_needed
    assert wire["messages"][0]["content"].count(identity) == 1
    assert sum(row["content"].count(OWNER_INPUT) for row in wire["messages"]) == 1
    assert [row["content"] for row in wire["messages"]].count(work) == 1
