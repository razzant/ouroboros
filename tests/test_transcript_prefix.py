"""The transcript is append-only between the sends of one loop execution.

OpenAI-family caches (Codex backend, OpenAI API, OpenRouter->OpenAI) reuse a
previous request only when it is a byte-prefix of the next, so a transient
trailing message or an in-place rewrite of an already-sent message discards
the whole conversation cache (issue #906, measured 2026-09-14).  The unit
cases pin the recorder; the loop case pins the invariant on the real
``run_llm_loop`` with the host's completion control and owner-source
observation actually injected.
"""
from __future__ import annotations

import copy
import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import extension_loader, loop, skill_context_snapshot, transcript_prefix
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, bound_host_row
from ouroboros.skill_loader import (
    SkillReviewState, load_skill, save_enabled, save_review_state,
    save_skill_grants, skill_state_dir_path,
)
from ouroboros.transcript_prefix import CHECKPOINT_KIND, message_digest, observe_send, sanction_rewrite

ANSWER = "The complete report includes the requested budget."


def call(name, arguments, identifier):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


@pytest.fixture
def full_loop(tmp_path, monkeypatch):
    """The real ``run_llm_loop`` with real tools and a scripted Main; ``model_inputs`` keeps
    every send. A completion selected beside an effect Main has not read is held: the host
    retains the answer, appends its completion control and the owner-source observation."""
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "12")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_args: False)
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry._ctx.repo_dir.mkdir()
    ctx = registry._ctx
    task_id = "prefix-loop-root"
    # The production pre-loop canonical RUNNING admission of a root.
    contract = build_task_contract({"id": task_id, "root_task_id": task_id, "delegation_role": "root"})
    write_task_result(ctx.drive_root, task_id, STATUS_RUNNING, root_task_id=task_id, delegation_role="root",
                      task_contract=contract, result="Task is running.")
    ctx.task_id = ctx.root_task_id = task_id
    ctx.delegation_role = "root"
    ctx.task_metadata = {"root_task_id": task_id, "budget_drive_root": str(ctx.drive_root)}
    ctx.task_contract = contract
    ctx.task_attempt, ctx.current_chat_id, ctx.is_direct_chat = 1, 1, False
    fixture = SimpleNamespace(tools=registry, ctx=ctx, events=queue.Queue(), model_inputs=[], model_step=0,
                              progress=[])
    fixture.run_args = dict(
        messages=[{"role": "system", "content": "Complete the owner task."},
                  {"role": "user", "content": "Prepare the complete report including its budget."}],
        tools=registry, llm=SimpleNamespace(default_model=lambda: "fixture/main"),
        drive_logs=ctx.drive_root / "logs", emit_progress=lambda text, **_kw: fixture.progress.append(text),
        incoming_messages=queue.Queue(), task_id=task_id, drive_root=ctx.drive_root, event_queue=fixture.events,
    )

    def run():
        # The scripted Main replaces the transport, including its actual-context
        # observer. Observe only the request whose response is being returned.
        model = loop.call_llm_with_retry

        def observed(*args, **kwargs):
            result = model(*args, **kwargs)
            observer = kwargs.get("model_context_observer")
            if result[0] is not None and callable(observer):
                observer(args[1])
            return result
        with monkeypatch.context() as patcher:
            patcher.setattr(loop, "call_llm_with_retry", observed)
            return loop.run_llm_loop(**fixture.run_args)
    fixture.run = run
    return fixture


def select_completion(f, **selected):
    """Main selects a completion, acknowledging the owner-source facts the host observed."""
    subject = {"owner_source_sha256": f.ctx._acceptance_observation["owner_source_sha256"]}
    arguments = {"action": "finish", **selected, "acceptance_subject": subject}
    return {"content": None, "tool_calls": [call("finish_task", arguments, f"finish-{f.model_step}")]}


# ---------------------------------------------------------------------------
# message_digest: content identity, not envelope shape
# ---------------------------------------------------------------------------

def test_cache_control_marker_does_not_change_the_digest():
    plain = {"role": "user", "content": "do the work"}
    marked = {"role": "user", "content": [
        {"type": "text", "text": "do the work", "cache_control": {"type": "ephemeral"}}]}
    assert message_digest(plain) == message_digest(marked)


def test_one_block_and_string_tool_results_share_one_digest_but_splits_do_not():
    as_string = {"role": "tool", "tool_call_id": "t1", "content": "result body"}
    as_blocks = {"role": "tool", "tool_call_id": "t1",
                 "content": [{"type": "text", "text": "result "}, {"type": "text", "text": "body"}]}
    assert message_digest(as_string) != message_digest(as_blocks)
    as_blocks["content"] = [{"type": "text", "text": " result body\n"}]
    assert message_digest(as_string) == message_digest(as_blocks)


def test_private_custody_keys_do_not_change_the_digest():
    bare = {"role": "user", "content": "observation"}
    carried = {"role": "user", "content": "observation", "acceptance_observation": True,
               "nativeContinuation": {"id": "x"}, "_context_capsule": {"a": 1},
               "_caption": "cap", "_source_path": "/tmp/x.png"}
    assert message_digest(bare) == message_digest(carried)


def test_role_text_and_tool_calls_do_change_the_digest():
    base = {"role": "user", "content": "a"}
    assert message_digest(base) != message_digest({"role": "assistant", "content": "a"})
    assert message_digest(base) != message_digest({"role": "user", "content": "b"})
    calling = {"role": "assistant", "content": "", "tool_calls": [call("read_file", {"path": "x"}, "c1")]}
    other = {"role": "assistant", "content": "", "tool_calls": [call("read_file", {"path": "y"}, "c1")]}
    assert message_digest(calling) != message_digest(other)
    assert message_digest(calling) != message_digest({"role": "assistant", "content": ""})
    assert message_digest({"role": "tool", "tool_call_id": "a", "content": "r"}) != message_digest(
        {"role": "tool", "tool_call_id": "b", "content": "r"})


# ---------------------------------------------------------------------------
# observe_send: which break, where, and whether it was sanctioned
# ---------------------------------------------------------------------------

SYSTEM = {"role": "system", "content": "Complete the owner task."}
TASK = {"role": "user", "content": "Prepare the report."}
ASSISTANT = {"role": "assistant", "content": "working"}


def test_first_send_and_pure_append_record_nothing():
    slot = SimpleNamespace()
    assert observe_send(slot, [SYSTEM, TASK], round_idx=1) is None
    assert observe_send(slot, [SYSTEM, TASK, ASSISTANT], round_idx=2) is None
    assert observe_send(slot, [SYSTEM, TASK, ASSISTANT], round_idx=3) is None
    assert getattr(slot, transcript_prefix.DIGEST_ATTR) == [
        message_digest(m) for m in (SYSTEM, TASK, ASSISTANT)]


def test_replaced_last_message_is_a_tail_replacement():
    slot = SimpleNamespace()
    observe_send(slot, [SYSTEM, TASK, ASSISTANT], round_idx=1)
    fact = observe_send(slot, [SYSTEM, TASK, {"role": "assistant", "content": "other"}], round_idx=2)
    assert fact == {"checkpoint_kind": CHECKPOINT_KIND, "round": 2, "index": 2, "kind": "tail_replaced",
                    "previous_messages": 3, "current_messages": 3, "sanctioned_by": None}


def test_changed_middle_message_is_a_rewrite():
    slot = SimpleNamespace()
    observe_send(slot, [SYSTEM, TASK, ASSISTANT], round_idx=1)
    fact = observe_send(
        slot, [SYSTEM, {"role": "user", "content": "other task"}, ASSISTANT, TASK], round_idx=2)
    assert fact["kind"] == "rewritten" and fact["index"] == 1
    assert (fact["previous_messages"], fact["current_messages"]) == (3, 4)


def test_changed_system_message_is_a_system_rewrite():
    slot = SimpleNamespace()
    observe_send(slot, [SYSTEM, TASK, ASSISTANT], round_idx=1)
    fact = observe_send(slot, [{"role": "system", "content": "reprojected"}, TASK, ASSISTANT], round_idx=4)
    assert fact["kind"] == "system_rewritten" and fact["index"] == 0 and fact["round"] == 4


def test_shorter_transcript_with_an_equal_prefix_is_a_shrink():
    slot = SimpleNamespace()
    observe_send(slot, [SYSTEM, TASK, ASSISTANT], round_idx=1)
    fact = observe_send(slot, [SYSTEM, TASK], round_idx=2)
    assert fact["kind"] == "shrunk" and fact["index"] == 3
    assert (fact["previous_messages"], fact["current_messages"]) == (3, 2)


def test_sanction_stamp_is_consumed_by_the_next_observation():
    slot = SimpleNamespace()
    observe_send(slot, [SYSTEM, TASK, ASSISTANT], round_idx=1)
    sanction_rewrite(slot, "compaction")
    stamped = observe_send(slot, [SYSTEM, TASK, {"role": "assistant", "content": "other"}], round_idx=2)
    assert stamped["sanctioned_by"] == "compaction"
    later = observe_send(slot, [SYSTEM, TASK, {"role": "assistant", "content": "third"}], round_idx=3)
    assert later["sanctioned_by"] is None, "the stamp is one-shot"


def test_a_compaction_stamp_never_explains_a_system_rewrite():
    slot = SimpleNamespace()
    observe_send(slot, [SYSTEM, TASK, ASSISTANT], round_idx=1)
    sanction_rewrite(slot, "compaction")
    fact = observe_send(slot, [{"role": "system", "content": "reprojected"}, TASK, ASSISTANT], round_idx=2)
    assert fact["kind"] == "system_rewritten" and fact["sanctioned_by"] is None
    assert getattr(slot, transcript_prefix.SANCTION_ATTR) is None, "the stamp is still consumed"


def test_sanctioned_by_rides_through_unchanged():
    slot = SimpleNamespace()
    observe_send(slot, [SYSTEM, TASK, ASSISTANT], round_idx=1)
    fact = observe_send(slot, [SYSTEM, TASK], round_idx=2, sanctioned_by="compaction")
    assert fact["sanctioned_by"] == "compaction"


# ---------------------------------------------------------------------------
# The loop-level invariant
# ---------------------------------------------------------------------------

def _describe(message):
    calls = transcript_prefix._tool_call_identity(message.get("tool_calls"))
    return (f"role={message.get('role')!r} tool_calls={[c[1] for c in calls]!r} "
            f"text={loop._extract_plain_text_from_content(message.get('content'))[:120]!r}")


def _assert_prefix_chain(sends):
    for step in range(1, len(sends)):
        previous, current = sends[step - 1], sends[step]
        for index, (before, after) in enumerate(zip(previous, current)):
            assert message_digest(before) == message_digest(after), (
                f"send {step} is not a prefix extension of send {step - 1}: message {index} of "
                f"{len(previous)} was rewritten\n  before: {_describe(before)}\n  after:  {_describe(after)}")
        assert len(current) >= len(previous), (
            f"send {step} dropped messages: {len(previous)} -> {len(current)}; "
            f"last surviving message {_describe(current[-1])}")


def _prefix_breaks(events):
    rows = []
    for item in list(events.queue):
        data = item.get("data") if isinstance(item, dict) else None
        if isinstance(data, dict) and data.get("type") == "task_checkpoint" \
                and data.get("checkpoint_kind") == CHECKPOINT_KIND:
            rows.append(data)
    return rows


def _four_tool_rounds(f):
    """Script: a completion beside one unread effect (held), then a read per round, then
    selecting the retained answer."""
    def main(_llm, messages, *_args, **_kwargs):
        f.model_inputs.append(copy.deepcopy(messages))
        f.model_step += 1
        if f.model_step == 1:
            completion = select_completion(f, answer=ANSWER)
            return {"content": "The report is ready.", "tool_calls": [
                *completion["tool_calls"],
                call("write_file", {"root": "task_drive", "path": "proof.txt",
                                    "content": "effect"}, "seed"),
            ]}, 0.0
        if f.model_step in (2, 3, 4):
            return {"content": "", "tool_calls": [
                call("read_file", {"root": "task_drive", "path": "proof.txt"},
                     f"read-{f.model_step}")]}, 0.0
        assert f.model_step < 9, f.progress
        return select_completion(f, answer_sha256=f.ctx._delivery_candidate.content_sha256), 0.0
    return main


def test_every_send_of_one_execution_extends_the_previous_send(full_loop, monkeypatch):
    f = full_loop
    monkeypatch.setattr(loop, "call_llm_with_retry", _four_tool_rounds(f))

    result, usage, _trace = f.run()

    assert result == ANSWER, (result, f.progress)
    assert len(f.model_inputs) >= 5, f.model_inputs
    # The held completion left its control and the owner-source observation in the next send.
    held = json.dumps(f.model_inputs[1], ensure_ascii=False)
    assert "[DELIVERY_FINALIZATION_CONTROL]" in held and "[ACCEPTANCE_SUBJECT_OBSERVATION]" in held
    _assert_prefix_chain(f.model_inputs)
    assert _prefix_breaks(f.events) == []
    assert "prompt_prefix_breaks" not in usage


# ---------------------------------------------------------------------------
# One predicate for every producer that appends behind a sent row
# ---------------------------------------------------------------------------

def test_a_sent_tail_row_is_never_merged_into_whatever_produced_it():
    """Only absence from a recorded send permits merging; unknown means append."""
    from ouroboros.loop_messages import _append_or_merge_user_content
    from ouroboros.transcript_prefix import unsent_in_previous_send

    slot = SimpleNamespace()
    messages = [SYSTEM, TASK, {"role": "user", "content": "[Message from my human]: first"}]
    frozen = copy.deepcopy(messages)
    assert unsent_in_previous_send(slot, messages[-1]) is False
    _append_or_merge_user_content(messages, "before any observed send", slot=slot)
    assert len(messages) == 4 and messages[:3] == frozen
    observe_send(slot, messages, round_idx=1)
    frozen = copy.deepcopy(messages)
    assert unsent_in_previous_send(slot, messages[-1]) is False
    _append_or_merge_user_content(messages, "[Message from independent task x]\nhello", slot=slot)
    assert len(messages) == 5 and messages[:4] == frozen
    assert unsent_in_previous_send(slot, messages[-1]) is True
    _append_or_merge_user_content(messages, "same round follow-up", slot=slot)
    assert len(messages) == 5 and messages[-1]["content"].endswith("same round follow-up")
    assert messages[:4] == frozen
    for tail in ({"role": "user", "content": "ordinary user row"},
                 {"role": "user", "content": "[ACCEPTANCE_SUBJECT_OBSERVATION]", "acceptance_observation": True}):
        plain = [SYSTEM, TASK, tail]
        _append_or_merge_user_content(plain, "owner words")
        assert len(plain) == 4 and plain[2] is tail


def test_the_roster_note_rides_the_real_loop_as_an_append(full_loop, monkeypatch):
    """A host-listed independent root appears once, as a fresh tail row, and the
    transcript stays a prefix chain across every send (no tail_replaced)."""
    from ouroboros.utils import atomic_write_json, utc_now_iso

    f = full_loop
    (f.ctx.drive_root / "state").mkdir(parents=True, exist_ok=True)
    atomic_write_json(f.ctx.drive_root / "state" / "queue_snapshot.json", {
        "ts": utc_now_iso(), "running": [
            {"id": "peer-root", "task": {"id": "peer-root", "title": "Peer work", "chat_id": 0}},
        ], "pending": [], "worker_total": 1,
    })
    monkeypatch.setattr(loop, "call_llm_with_retry", _four_tool_rounds(f))

    result, usage, _trace = f.run()

    assert result == ANSWER, (result, f.progress)
    _assert_prefix_chain(f.model_inputs)
    assert _prefix_breaks(f.events) == []
    notes = [m for m in f.model_inputs[-1]
             if m.get("role") == "user" and "[INDEPENDENT_ROOTS]" in loop._extract_plain_text_from_content(m.get("content"))]
    assert len(notes) == 1, "the roster changed once, so exactly one note was appended"
    assert "- peer-root · Peer work · chat=0 · running" in loop._extract_plain_text_from_content(notes[0]["content"])
    assert "prompt_prefix_breaks" not in usage


def _after_step(f, step, hook):
    """Run ``hook`` once the scripted model has answered round ``step``."""
    inner = _four_tool_rounds(f)

    def main(*args, **kwargs):
        out = inner(*args, **kwargs)
        if f.model_step == step:
            hook()
        return out
    return main


def _assert_one_sanctioned_rewrite_at_round_three(f, usage):
    breaks = _prefix_breaks(f.events)
    assert [b["kind"] for b in breaks] == ["rewritten"], breaks
    assert breaks[0]["sanctioned_by"] == "compaction" and breaks[0]["round"] == 3, breaks
    # A sanction names the rewrite's origin; it is counted like every change of sent history.
    assert usage["prompt_prefix_breaks"] == 1
    # The rounds after the compaction extend the compacted transcript again.
    _assert_prefix_chain(f.model_inputs[2:])


def test_authored_compaction_is_recorded_as_a_sanctioned_break_on_its_round(full_loop, monkeypatch):
    from ouroboros.tools.compact_context import _compact_context

    f = full_loop
    # The mind's own view, bound to its last observed send, is applied by _run_round_compaction at the top of round 3.
    monkeypatch.setattr(loop, "call_llm_with_retry", _after_step(
        f, 2, lambda: _compact_context(f.ctx, working_note="The first reads are folded.", keep_unit_ids=[])))

    result, usage, _trace = f.run()

    assert result == ANSWER, (result, f.progress)
    _assert_one_sanctioned_rewrite_at_round_three(f, usage)


def test_predicted_pressure_keeps_the_append_only_prefix_without_automatic_reclaim(full_loop, monkeypatch):
    f = full_loop
    monkeypatch.setattr(loop, "call_llm_with_retry", _four_tool_rounds(f))
    def no_automatic_summary(*args, **kwargs):
        raise AssertionError("Prediction must not buy helper compaction")
    monkeypatch.setattr(loop, "compact_tool_history_llm", no_automatic_summary)
    fired = []

    def measure(ctx, *, automatic_pass_used):
        # The measurement reports pressure at round 3; it is a fact rather
        # than authorization to rewrite the already-sent transcript.
        if ctx.round_idx != 3 or fired or automatic_pass_used:
            return None
        fired.append(True)
        from ouroboros.context_fit import MainFitMeasurement

        # A real measurement: only a positive deficit decides "reclaim_once", and the
        # reclaim's low-water telemetry reads the deficit and boundary fields.
        measurement = MainFitMeasurement(
            route_fp="fp", round_id="r3", profile="owner_low", rendered_mode="low",
            estimated_input_tokens=200_000 - 65_536 + 100, response_reserve_tokens=65_536,
            target_total_tokens=200_000, capacity_total_tokens=None,
            measurement_basis="cold_estimate", measurement_density=1.0,
            target_deficit_tokens=100, capacity_deficit_tokens=None, reclaim_goal_tokens=100,
        )
        return SimpleNamespace(action="reclaim_once", measurement=measurement, automatic_pass_used=False)

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)

    result, usage, _trace = f.run()

    assert result == ANSWER, (result, f.progress)
    assert fired, "the pressure measurement did not run"
    assert not _prefix_breaks(f.events)
    assert "prompt_prefix_breaks" not in usage
    _assert_prefix_chain(f.model_inputs)

# ---------------------------------------------------------------------------
# DEVELOPMENT 06 "Cache-friendliness invariant": an ordinary round appends.
# An installed extension publishes a changing ``context_snapshot`` every round while
# roster, owner words, review facts and completion controls also arrive. The scripted
# model is the only transport replacement; ``run_llm_loop``, tools and extension
# admission are real. The oracle compares the sent requests themselves, so no
# ``sanction_rewrite`` label can make an ordinary rewrite pass: #2076 relabelled the
# older seam test and the rule, and every changed snapshot then made OpenAI-family
# providers re-read the whole conversation (measured 2026-10-21: 99.5 % -> 62.7 % of
# input read from cache on 95 comparable sends).
# ---------------------------------------------------------------------------


@pytest.fixture
def installed_publisher(full_loop):
    f = full_loop
    payload = f.ctx.repo_dir / "cycle-feed"
    payload.mkdir()
    manifest = {"name": "cycle-feed", "version": "1", "description": "Cycle prefix fixture",
                "type": "extension", "entry": "plugin.py", "plugin_api": "2.0",
                "permissions": ["context_snapshot"], "context_snapshot": {"path": "snapshot.json"}}
    (payload / "SKILL.md").write_text(json.dumps(manifest))
    (payload / "plugin.py").write_text("def register(api):\n    pass\n")
    root = f.ctx.drive_root
    skill = load_skill(payload, root)
    save_enabled(root, skill.name, True)
    save_review_state(root, skill.name, SkillReviewState(status="clean", content_hash=skill.content_hash))
    save_skill_grants(root, skill.name, [], content_hash=skill.content_hash, requested_keys=[],
                      granted_permissions=["context_snapshot"], requested_permissions=["context_snapshot"])
    skill = load_skill(payload, root)
    assert extension_loader.load_extension(skill, lambda: {}, drive_root=root,
                                           skills=[skill], _force_in_process=True) is None
    path = skill_state_dir_path(root, skill.name) / "snapshot.json"

    def publish(version):
        # Fixed dates remove sleeps, expiry and wall-clock renewal from the test.
        path.write_text(json.dumps({"schema_version": 1, "observed_at": "2000-01-01T00:00:00Z",
                                   "valid_until": "2999-01-01T00:00:00Z", "text": f"Quota sample {version}"}))
        (root / "state" / "queue_snapshot.json").write_text(json.dumps({
            "ts": "2026-10-21T10:00:00Z", "running": [{"id": "peer-root", "task": {
                "id": "peer-root", "title": f"Peer work {version}", "chat_id": 0}}],
            "pending": [], "worker_total": 1,
        }))
    publish(0)
    try:
        yield f, publish
    finally:
        extension_loader.unload_extension(skill.name)


def _block_view(block):
    return {key: item for key, item in block.items() if key != "cache_control"} if isinstance(block, dict) else block


def message_view(message):
    """Drop only host custody keys and cache placement; keep public bytes and block order."""
    clean = {key: item for key, item in message.items() if key != "cache_control" and not str(key).startswith("_")}
    content = clean.get("content")
    if isinstance(content, list):
        content = [_block_view(block) for block in content]
        # Cache placement wraps a plain string in one text block on some routes.
        if len(content) == 1 and isinstance(content[0], dict) and set(content[0]) == {"type", "text"}:
            content = content[0]["text"]
        clean["content"] = content
    return clean


def tools_view(schemas):
    return [_block_view(schema) for schema in schemas or []]


def _ordinary_rewrites(sends):
    failures = []
    for step, (previous, current) in enumerate(zip(sends, sends[1:]), start=2):
        previous, current = [message_view(m) for m in previous], [message_view(m) for m in current]
        if len(current) < len(previous):
            failures.append((step, "shrunk", len(previous), len(current)))
        for index, (before, after) in enumerate(zip(previous, current)):
            if before != after:
                failures.append((step, index, before.get("content"), after.get("content")))
                break
    return failures


def _address_previous_snapshot(original):
    """The #2076 behaviour: replace the previous full snapshot with a pointer and sanction it."""
    def append(ctx, messages, drive_root):
        changed = original(ctx, messages, drive_root)
        rows = [i for i, m in enumerate(messages)
                if m.get(HOST_CONTEXT_KIND_KEY) == skill_context_snapshot.SNAPSHOT_KIND]
        if changed and len(rows) > 1 and "Superseded" not in str(messages[rows[-2]].get("content")):
            messages[rows[-2]] = bound_host_row(skill_context_snapshot.SNAPSHOT_KIND, "[Superseded skill snapshot]")
            transcript_prefix.sanction_rewrite(ctx, "superseded_skill_snapshot")
        return changed
    return append


@pytest.mark.parametrize("variant", ["unchanged_snapshot", "changing_snapshot", "sanctioned_rewrite"])
def test_an_ordinary_cycle_appends_whatever_a_seam_calls_its_rewrite(installed_publisher, monkeypatch, variant):
    f, publish = installed_publisher
    model = _four_tool_rounds(f)
    schemas, wires = [], []
    if variant == "sanctioned_rewrite":
        monkeypatch.setattr(skill_context_snapshot, "maybe_append_skill_snapshots",
                            _address_previous_snapshot(skill_context_snapshot.maybe_append_skill_snapshots))

    def scripted(*args, **kwargs):
        from ouroboros.llm_attempt import _physical_candidate
        from ouroboros.llm_claudexor import _request

        schemas.append(copy.deepcopy(args[3]))
        wires.append((
            _physical_candidate({"messages": args[1], "tools": args[3]}),
            _request({"source": "codex", "resolved_model": "fixture"}, args[1], args[3],
                     {"model_account_override": "", "_no_account_preference": True}),
        ))
        if f.model_step >= 4:
            f.model_inputs.append(copy.deepcopy(args[1]))
            f.model_step += 1
            assert f.model_step < 9, f.progress
            # New owner words can invalidate the retained selection: select afresh.
            result = select_completion(f, answer=ANSWER), 0.0
        else:
            result = model(*args, **kwargs)
        publish(0 if variant == "unchanged_snapshot" else f.model_step)
        if 2 <= f.model_step < 4:
            f.run_args["incoming_messages"].put(f"Owner follow-up {f.model_step}: keep the complete budget.")
        return result

    monkeypatch.setattr(loop, "call_llm_with_retry", scripted)
    result, usage, _trace = f.run()
    assert result == ANSWER, (result, f.progress)
    assert len(f.model_inputs) >= 5
    failures = _ordinary_rewrites(f.model_inputs)
    if variant == "sanctioned_rewrite":
        # The label explains the rewrite; it neither hides it from the oracle nor from the counts.
        assert failures, "a sanctioned ordinary rewrite must still fail the append-only oracle"
        assert usage["prompt_prefix_breaks"] >= 1
        assert {b["sanctioned_by"] for b in _prefix_breaks(f.events)} == {"superseded_skill_snapshot"}
        record = json.loads(transcript_prefix.history_changes_path(f.ctx.drive_root).read_text())
        assert record["origins"]["superseded_skill_snapshot"]["count"] == usage["prompt_prefix_breaks"]
        return
    assert not failures, {"rewrites": failures, "breaks": _prefix_breaks(f.events)}
    for step in range(1, len(schemas)):
        assert tools_view(schemas[step]) == tools_view(schemas[step - 1]), "tool schemas changed"
    for route in range(2):
        requests = [pair[route] for pair in wires]
        for previous, current in zip(requests, requests[1:]):
            size = len(previous["messages"])
            assert [message_view(m) for m in current["messages"][:size]] == [message_view(m) for m in previous["messages"]], (
                route, "wire prefix changed")
            assert tools_view(current.get("tools")) == tools_view(previous.get("tools")), (route, "wire tools changed")
    assert _prefix_breaks(f.events) == []
    assert "prompt_prefix_breaks" not in usage
    final = json.dumps(f.model_inputs[-1])
    for marker in ("[CONTEXT_FACTS]", "[ACCEPTANCE_SUBJECT_OBSERVATION]",
                   "[DELIVERY_FINALIZATION_CONTROL]", "[INDEPENDENT_ROOTS]", "Owner follow-up"):
        assert marker in final, marker
    snapshots = [m for m in f.model_inputs[-1] if m.get(HOST_CONTEXT_KIND_KEY) == skill_context_snapshot.SNAPSHOT_KIND]
    # A text that changes every round cannot multiply in the conversation: its full text once, then one
    # outdated note naming where the current text is read, until the conversation is next rebuilt.
    assert len(snapshots) == (2 if variant == "changing_snapshot" else 1)
    if variant == "changing_snapshot":
        assert '"state": "outdated"' in snapshots[-1]["content"] and '"text":' not in snapshots[-1]["content"]
