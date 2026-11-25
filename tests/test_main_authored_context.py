"""Authored views use the real Main loop, tool boundary and source materializer."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import queue
from types import SimpleNamespace

import pytest

from ouroboros import context_compaction, loop, usage_accounting as ua
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.tools.registry import ToolRegistry
from tests.test_context_fit_integration import _plan


def call(name, arguments, identifier):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": identifier, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)}}]}


@pytest.fixture
def main_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "12")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_args: False)
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry._ctx.repo_dir.mkdir()
    effort = "medium"  # The synthetic fit and first send price the same request shape.
    registry._ctx.context_fit_plan = replace(_plan(preferred="max", window=900_000), start_reasoning_effort=effort)
    source = "Exact original evidence with its trailing conclusion.\n" * 160
    (registry._ctx.repo_dir / "evidence.txt").write_text(source, encoding="utf-8")
    f = SimpleNamespace(registry=registry, ctx=registry._ctx, source=source,
                        inputs=[], progress=[], incoming=queue.Queue(), events=queue.Queue())
    f.messages = f.ctx.context_fit_plan.messages_for("max")

    def run(responses, *, initial_effort=effort, first_projection=None, first_candidate=None):
        replies = iter(responses)

        class Provider:
            def default_model(self):
                return f.ctx.context_fit_plan.model

            def chat(self, **kwargs):
                from ouroboros.llm_attempt import _physical_candidate, _attempt_request
                from ouroboros.model_send_seal import persist_physical_candidate

                ua.adopt_physical_attempt_capture(None)
                if first_projection is not None and not f.inputs:
                    kwargs["messages"] = first_projection(kwargs["messages"])
                f.inputs.append(deepcopy(kwargs))
                reply = next(replies)
                if isinstance(reply, Exception):
                    raise reply
                message = reply(kwargs) if callable(reply) else deepcopy(reply)
                candidate = _physical_candidate({key: kwargs[key] for key in ("messages", "tools", "model")})
                if first_candidate is not None and len(f.inputs) == 1:
                    candidate = first_candidate(candidate)  # the lane's own sealed shape of the accepted send
                request = _attempt_request({"provider": "openai", "usage_model": kwargs["model"]}, candidate)
                attempt_id = f"authored-send-{len(f.inputs)}"
                persisted = persist_physical_candidate(f.ctx.drive_root, task_id="authored-main",
                    attempt_id=attempt_id, candidate=candidate, candidate_facts={})
                ua.adopt_physical_attempt_capture(ua.PhysicalAttemptCapture(
                    attempt_id, kwargs["model"], "openai", "settled", "canonical_json_v1",
                    candidate_manifest_ref=persisted["manifest_ref"], physical_context=request.physical_context,
                    candidate_raw_sha256=request.candidate_raw_sha256, candidate_raw_size_bytes=request.candidate_raw_size_bytes,
                    candidate_context_sha256=request.candidate_context_sha256,
                    candidate_context_size_bytes=request.candidate_context_size_bytes))
                return message, {"prompt_tokens": 100, "completion_tokens": 10,
                                 "cost": 0.0, "provider": "openai"}

        return loop.run_llm_loop(
            messages=f.messages, tools=f.registry, llm=Provider(),
            initial_effort=initial_effort,
            drive_logs=f.ctx.drive_root / "logs", drive_root=f.ctx.drive_root,
            emit_progress=lambda text, **kw: f.progress.append(text), incoming_messages=f.incoming,
            task_id="authored-main", event_queue=f.events)

    f.run = run
    return f


@pytest.mark.parametrize("carrier", ["messages", "anthropic_system", "local_string", "split_before_task"])
@pytest.mark.parametrize("carried_in_accepted_input", [True, False])
def test_late_helper_delivery_uses_the_first_accepted_owner_words(
        main_loop, tmp_path, carried_in_accepted_input, carrier):
    from ouroboros import context, dialogue_store, owner_relay
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.llm import LLMClient
    from ouroboros.llm_messages import split_leading_system_prefix
    from ouroboros.chat_uploads import store_upload
    from ouroboros.task_results import write_task_result
    from tests.test_doc_context import _make_env_and_memory

    f = main_loop
    env, memory = _make_env_and_memory(tmp_path / "source")
    f.ctx.task_metadata = {"delegation_role": "subagent", "root_task_id": "quarterly-report"}
    task = {"id": "authored-main", "type": "task", "text": "Draft a summary of the quarter.",
            "metadata": f.ctx.task_metadata, "drive_root": str(f.ctx.drive_root), "budget_drive_root": str(f.ctx.drive_root)}
    evidence = SimpleNamespace(route_fp="route-a", status="confirmed", stale=False, window_tokens=900_000)

    def build(subject):
        rows, _ = context.build_llm_messages(env, memory, subject, ctx=f.ctx, reasoning_effort="medium",
            route_resolver=lambda *_a, **_kw: ({"provider": "openai", "model": "openai/test-model"}, evidence))
        return rows

    fallback = build(task)
    data, words = b"region,sales\neast,4\n", "Use the attached final figures."
    _path, upload = store_upload(data, "figures.csv", data_dir=f.ctx.drive_root)
    row = {"direction": "in", "chat_id": 1, "ts": "2026-10-04T06:00:00+00:00", "text": words,
           "client_message_id": "quarterly-figures", "attachments": [upload]}
    dialogue_store.append(f.ctx.drive_root, "chat", row)
    address = owner_relay.owner_row_address(row)
    held = {address: {**row, "carried": {"went_to": "sibling"}}}
    write_task_result(f.ctx.drive_root, task["id"], "running", delegation_role="subagent",
                      parent_task_id="quarterly-report", root_task_id="quarterly-report", held_owner_messages=held)
    f.messages = build({**task, "held_owner_messages": held})

    def attach(_kwargs):
        ref, _ = owner_relay.owner_row_source(row)
        owner_relay.carry_row_at_receipt(f.ctx.drive_root, "quarterly-report", 1, "quarterly-figures", source_ref=ref)
        return call("read_file", {"path": "evidence.txt"}, "draft")

    def physical_projection(messages):
        # Stub the ordinary transport after host selection. Both cases retain
        # the same upcoming host view; this fallback had no such room input.
        return messages if carried_in_accepted_input else [deepcopy(fallback[0]), *messages[1:]]

    def sealed_shape(candidate):  # the layout the lane itself seals for the accepted first send
        lane, rows = LLMClient.__new__(LLMClient), candidate["messages"]
        source = rows if carried_in_accepted_input else [deepcopy(fallback[0]), *rows[1:]]
        if carrier == "anthropic_system":  # the native lane moves the system row's blocks to the request slot
            system, messages = lane._build_anthropic_messages(source)
            assert all(str(row.get("role") or "") != "system" for row in messages)
            return {**candidate, "messages": messages, "system": system}
        if carrier == "local_string":  # a string-system lane folds the declared blocks into one string
            return {**candidate, "messages": lane._build_local_candidate(source, None, 512, "auto")[1]["messages"]}
        if carrier == "split_before_task":  # the Codex/OpenAI projection: leading items, then one notice row
            sent, moved = split_leading_system_prefix(source, placement="before_task")
            assert moved, "the fixture's composed system message declares no splittable stable prefix"
            return {**candidate, "messages": sent}
        return candidate

    f.run([attach, {"content": "The quarterly summary is ready."}],
          first_projection=physical_projection, first_candidate=sealed_shape)
    accepted = str(f.inputs[0]["messages"][0]["content"])
    second = f.inputs[1]["messages"]
    assert (words in accepted) is carried_in_accepted_input
    assert words in str(second[0]["content"])  # deliberately identical current system view
    anchor = f.ctx._historical_author_inputs["anchors"][0]
    assert anchor["physical_source_status"] == "observed_projection"
    digest = owner_relay.owner_row_identity(row)["row_sha256"]
    assert (digest in anchor.get("helper_owner_rows", {})) is carried_in_accepted_input
    tail = str(second[1:])
    if carried_in_accepted_input:
        assert f"the row {address} in your starting words was addressed to this work" in tail
        assert words not in tail
    else:
        assert words in tail and "[Host attachment delivery]" in tail and "figures.csv" in tail and "status=staged" in tail
        assert "in your starting words was addressed" not in tail
    staged = task_artifact_dir_path(f.ctx.drive_root, "authored-main") / "attachments"
    assert any(file.is_file() and file.read_bytes() == data for file in staged.iterdir())


@pytest.mark.parametrize("standing", ["context", "tree"])
@pytest.mark.parametrize("carrier", ["origin", "marked"])
def test_a_background_row_in_the_starting_words_is_answered_as_addressed(main_loop, carrier, standing):
    """The rows a handoff reads as background — the Project's retained origin rows and the rows marks keep in
    view — are not words addressed to this work: the typed builder's own carriers say so, and the birth pin's
    standing follows. When such a row arrives addressed, the helper is told it was already in its starting
    words and is not shown them again; a row that stood as addressed to this work gets nothing more."""
    from ouroboros import dialogue_store, owner_relay
    from ouroboros.context_input_selection import owner_row_standing
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_words import _row

    f, root, task_id = main_loop, main_loop.ctx.drive_root, "authored-main"
    words = "The project opened with these words."
    built = _row(words, source="origin_message" if carrier == "origin" else "marked_message", carrier=carrier,
                 ts="2026-10-04T06:00:00+00:00")
    assert owner_row_standing(built) == "context"                       # background, whatever carrier held it
    assert owner_row_standing({**built, "carrier": "went to sibling"}) == "context"
    assert owner_row_standing({**built, "carrier": "held"}) == "tree"   # the same words addressed to this work
    row = {"direction": "in", "chat_id": 1, "ts": built["ts"], "text": words,
           "client_message_id": f"quarterly-{carrier}"}
    dialogue_store.append(root, "chat", row)
    address, digest = owner_relay.owner_row_address(row), owner_relay.owner_row_identity(row)["row_sha256"]
    f.ctx.task_metadata = {"delegation_role": "subagent", "root_task_id": "quarterly-report"}
    f.ctx.task_id = task_id
    f.ctx._historical_author_inputs = {"version": 1, "task_id": task_id, "status": "captured", "anchors": [
        {"position": "first", "status": "captured", "physical_source_status": "observed_projection",
         "source_ref": "context_checkpoints/historical-author-input", "helper_owner_rows": {digest: standing}}]}
    owner_relay.write_relay(root, task_id, relay_id=f"carry:{address}", note="", chat_id=1,
                            quotes=[owner_relay.owner_quote(root, row, address=address)],
                            teller_task_id="quarterly-report", canonical_root=root, host_carried={"went_to": ""})
    messages = []
    controls = _drain_incoming_messages(messages, queue.Queue(), root, task_id, None, set(), owner_ctx=f.ctx)
    if standing == "context":
        assert [m["content"] for m in messages] == [
            f"[Host fact: the row {address} in your starting words was addressed to this work]"]
        assert controls["owner_carried_addressed"] == [address] and words not in str(messages)
    else:
        assert messages == [] and "owner_carried_addressed" not in controls


@pytest.mark.parametrize("view_rebuilt_before_dispatch", [False, True])
def test_a_row_waiting_at_a_helpers_first_send_is_delivered_whatever_view_is_dispatched(
        main_loop, tmp_path, view_rebuilt_before_dispatch):
    """A carried row already in the helper's mail at its first drain meets a view no input has pinned yet. Its words
    and file status go out as a row of the transcript, so they reach the accepted input whether the drafted view is
    sent as drafted or rebuilt without the row, once, and stay in the next request; no host line claims the starting
    words held it."""
    from ouroboros import context, dialogue_store, owner_relay
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.chat_uploads import store_upload
    from ouroboros.task_results import write_task_result
    from tests.test_doc_context import _make_env_and_memory

    f = main_loop
    env, memory = _make_env_and_memory(tmp_path / "source")
    f.ctx.task_metadata = {"delegation_role": "subagent", "root_task_id": "quarterly-report"}
    task = {"id": "authored-main", "type": "task", "text": "Draft a summary of the quarter.",
            "metadata": f.ctx.task_metadata, "drive_root": str(f.ctx.drive_root), "budget_drive_root": str(f.ctx.drive_root)}
    evidence = SimpleNamespace(route_fp="route-a", status="confirmed", stale=False, window_tokens=900_000)

    def build(subject):
        rows, _ = context.build_llm_messages(env, memory, subject, ctx=f.ctx, reasoning_effort="medium",
            route_resolver=lambda *_a, **_kw: ({"provider": "openai", "model": "openai/test-model"}, evidence))
        return rows

    without_row = build(task)
    data, words = b"region,sales\neast,4\n", "Use the attached final figures."
    _path, upload = store_upload(data, "figures.csv", data_dir=f.ctx.drive_root)
    row = {"direction": "in", "chat_id": 1, "ts": "2026-10-04T06:00:00+00:00", "text": words,
           "client_message_id": "quarterly-figures", "attachments": [upload]}
    dialogue_store.append(f.ctx.drive_root, "chat", row)
    address = owner_relay.owner_row_address(row)
    held = {address: {**row, "carried": {"went_to": "sibling"}}}
    write_task_result(f.ctx.drive_root, task["id"], "running", delegation_role="subagent",
                      parent_task_id="quarterly-report", root_task_id="quarterly-report", held_owner_messages=held)
    f.messages = build({**task, "held_owner_messages": held})
    assert words in str(f.messages[0]["content"])  # the drafted view holds the row as context
    ref, _ = owner_relay.owner_row_source(row)
    owner_relay.carry_row_at_receipt(f.ctx.drive_root, "quarterly-report", 1, "quarterly-figures", source_ref=ref)

    def dispatched(messages):  # the ordinary transport after host selection: the draft, or a view rebuilt without the row
        return [deepcopy(without_row[0]), *messages[1:]] if view_rebuilt_before_dispatch else messages

    f.run([call("read_file", {"path": "evidence.txt"}, "draft"), {"content": "The quarterly summary is ready."}],
          first_projection=dispatched)
    first, second = (request["messages"] for request in f.inputs[:2])
    assert (words in str(first[0]["content"])) is not view_rebuilt_before_dispatch
    for tail in (str(first[1:]), str(second[1:])):  # delivered with the first input and kept in the next
        assert tail.count(f"carried to this work by the host (reached this work); handle {address}]") == 1
        assert words in tail and "[Host attachment delivery]" in tail and "figures.csv" in tail and "status=staged" in tail
        assert "in your starting words was addressed" not in tail
    assert owner_relay.held_owner_messages(f.ctx)[address]["carried"]["went_to"] == ""
    staged = task_artifact_dir_path(f.ctx.drive_root, "authored-main") / "attachments"
    assert any(file.is_file() and file.read_bytes() == data for file in staged.iterdir())


def test_main_records_and_applies_authored_view_without_manual_observation(main_loop):
    f = main_loop
    answer, _, _ = f.run([
        call("read_file", {"path": "evidence.txt"}, "read"),
        call("compact_context", {"working_note": "I retained the trailing conclusion.", "keep_unit_ids": []}, "compact"),
        call("read_file", {"path": "evidence.txt"}, "reread"),
        {"content": "done"},
    ])
    assert answer == "done"
    receipt = f.ctx._context_view_receipt
    assert receipt["status"] == "applied" and receipt["reclaimed_tokens"] > 0
    checkpoint = json.loads(read_actor_source_bytes(f.ctx.drive_root, "authored-main", receipt["checkpoint_ref"]))
    before, after = f.inputs[1]["messages"], f.inputs[2]["messages"]
    assert checkpoint["messages"] == before
    assert f.source.splitlines()[-1] in str(checkpoint["messages"])
    unmarked = lambda rows: [{k: v for k, v in m.items() if k != "cache_control"} for m in rows]  # noqa: E731
    assert unmarked(after[:2]) == unmarked(before[:2])  # only the seal's host-only mark moved on
    assert not any(m.get("tool_call_id") == "read" for m in after)
    assert any(m.get("tool_call_id") == "compact" for m in after)
    assert "I retained the trailing conclusion." in str(after)
    assert "[Context view receipt]" in str(after)
    # The same read after the fold, on the same context, returns the view the first
    # call returned; nothing claims the folded text is still in view.
    first = next(m["content"] for m in before if m.get("tool_call_id") == "read")
    latest = f.inputs[-1]["messages"]
    assert not any(m.get("tool_call_id") == "read" for m in latest)
    assert next(m["content"] for m in latest if m.get("tool_call_id") == "reread") == first


@pytest.mark.parametrize(("mode", "role"), [("nano", "root"), ("low", "subagent"), ("max", "root")])
def test_retelling_instructions_and_authored_account_reach_next_send(main_loop, tmp_path, monkeypatch, role, mode):
    """Mode-specific context assembly, then shared-loop delivery; not model adherence."""
    from ouroboros import context
    from ouroboros.tools.compact_context import get_tools
    from tests.test_doc_context import _make_env_and_memory

    f = main_loop
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", mode)
    env, memory = _make_env_and_memory(tmp_path / "context")
    system = (Path(__file__).resolve().parents[1] / "prompts/SYSTEM.md").read_text(encoding="utf-8")
    env.repo_path("prompts/SYSTEM.md").write_text(system, encoding="utf-8")
    task = {"id": "authored-main", "type": "task", "text": "Investigate the proposed label change.",
            "delegation_role": role}
    if role == "subagent":
        task["configured_subagent"] = {"route": {"kind": "api_model"}}
    f.messages, _ = context.build_llm_messages(env, memory, task, ctx=f.ctx)
    schema = get_tools()[0].schema
    note = ("The reader loses the work's name among repeated labels; the proposal helps identify it. "
            "The owner approved label research, then cancelled only that branch; storage research remains approved. "
            "Implementation is still a proposal, not permission. The layout consequence is inferred; native behavior is unknown.")
    f.source = note + "\n" + f.source
    (f.ctx.repo_dir / "evidence.txt").write_text(f.source, encoding="utf-8")
    restored = {}

    def compact(kwargs):
        inspected = json.loads(next(m["content"] for m in kwargs["messages"] if m.get("tool_call_id") == "inspect"))
        unit = next(unit for unit in inspected["units"] if unit["kind"] == "tool")
        restored.update({"checkpoint_ref": inspected["restore_checkpoint"], "unit_id": unit["unit_id"],
                         "raw_sha256": unit["raw_sha256"]})
        return call("compact_context", {"working_note": note, "keep_unit_ids": []}, "compact")

    def restore(kwargs):
        from ouroboros.context_compaction import note_written

        accounts = [(block["text"], note_written(block["_context_capsule"])) for message in kwargs["messages"]
                    if message.get("role") == "assistant" and not message.get("tool_calls")
                    and isinstance(message.get("content"), list)
                    for block in message["content"]
                    if block.get("type") == "text" and block.get("_context_capsule")]
        assert any(text == f"[Actor-authored working view, written {written}; exact source retained by checkpoint]\n"
                   + note for text, written in accounts)
        assert not any(m.get("tool_call_id") == "read" for m in kwargs["messages"])
        return call("compact_context", {"restore_unit_refs": [restored]}, "restore")

    f.run([call("read_file", {"path": "evidence.txt"}, "read"),
           call("compact_context", {"inspect": True}, "inspect"), compact, restore, {"content": "done"}])
    first, after = f.inputs[0], f.inputs[-1]
    assert system in "\n".join(block["text"] for block in first["messages"][0]["content"])
    assert "later reader, not only my present understanding" in system
    assert "who decided what within which scope" in system
    assert "supersede only what they actually change" in system
    delivered = next(tool["function"] for tool in first["tools"] if tool["function"]["name"] == "compact_context")
    assert delivered == schema
    guidance = delivered["parameters"]["properties"]["working_note"]["description"]
    assert "future reader" in guidance and "within those sources" in guidance
    assert "unaffected branches" in guidance and "invent no detail" in guidance
    assert "raw evidence remains retrievable" in delivered["description"].lower()
    assert f.ctx._context_view_receipt["status"] == "applied"
    visible = "\n".join(m["content"] if isinstance(m.get("content"), str) else
                        "\n".join(b.get("text", "") for b in (m.get("content") or []) if isinstance(b, dict))
                        for m in after["messages"])
    assert note in visible and f.source.splitlines()[-1] in visible
    assert first["messages"][:2] == after["messages"][:2]


def test_explicit_recent_count_compacts_older_units_with_authored_note(main_loop):
    f = main_loop
    f.run([
        *(call("read_file", {"path": "evidence.txt"}, f"read-{i}") for i in range(4)),
        call("compact_context", {"working_note": "All four reads agree.", "keep_last_n": 2}, "compact"),
        {"content": "done"},
    ])
    after = f.inputs[-1]["messages"]
    assert f.ctx._context_view_receipt["status"] == "applied"
    assert [m["tool_call_id"] for m in after if m.get("role") == "tool"] == ["read-2", "read-3", "compact"]
    checkpoint = json.loads(read_actor_source_bytes(f.ctx.drive_root, "authored-main",
                                                  f.ctx._context_view_receipt["checkpoint_ref"]))
    assert [m["tool_call_id"] for m in checkpoint["messages"] if m.get("role") == "tool"] == [f"read-{i}" for i in range(4)]


def test_inspected_ids_override_count_and_new_owner_tail_survives(main_loop):
    f = main_loop
    pinned = {}

    def choose(kwargs):
        inspected = next(m for m in kwargs["messages"] if m.get("tool_call_id") == "inspect")
        pinned.update(json.loads(inspected["content"]))
        f.incoming.put({"text": "Keep the new exact owner correction.", "msg_id": "correction"})
        return call("compact_context", {"working_note": "Retained finding.", "keep_unit_ids": [],
                    "keep_last_n": 20, "expected_view_revision": pinned["view_revision"]}, "compact")

    f.run([call("read_file", {"path": "evidence.txt"}, "read"),
           call("compact_context", {"inspect": True}, "inspect"), choose, {"content": "done"}])
    receipt, after = f.ctx._context_view_receipt, f.inputs[-1]["messages"]
    assert receipt["status"] == "applied" and receipt["observed_view_revision"] == pinned["view_revision"]
    assert not any(m.get("tool_call_id") == "read" for m in after)
    assert all(any(m.get("tool_call_id") == name for m in after) for name in ("inspect", "compact"))
    assert any(m.get("role") == "user" and "Keep the new exact owner correction." in str(m.get("content")) for m in after)


def test_note_only_keeps_raw_and_repeated_note_is_cache_noop(main_loop, monkeypatch):
    from ouroboros import loop_round_limits

    f, invalidations, receipts = main_loop, [], []
    monkeypatch.setattr(loop_round_limits, "invalidate_task_cache_splits", lambda task: invalidations.append(task))
    note = {"working_note": "Retain these raw sources and this conclusion."}

    def repeat(kwargs):
        receipts.append(deepcopy(f.ctx._context_view_receipt))
        return call("compact_context", note, "again")

    f.run([call("read_file", {"path": "evidence.txt"}, "read"),
           call("compact_context", note, "first"), repeat, {"content": "done"}])
    assert receipts[0]["status"] == "applied" and f.ctx._context_view_receipt["status"] == "no_op"
    assert invalidations == ["authored-main"]
    assert f.ctx._context_view_receipt["checkpoint_ref"] is None
    assert any(m.get("tool_call_id") == "read" for m in f.inputs[-1]["messages"])
    assert f.inputs[-1]["tools"] == f.inputs[0]["tools"]


def test_projected_image_stays_in_canonical_authored_checkpoint(main_loop, monkeypatch):
    from ouroboros import vision_routing

    f = main_loop
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}
    f.messages.append({"role": "user", "content": [deepcopy(image)]})
    monkeypatch.setattr(vision_routing, "get_image_input_mode", lambda: "off")
    f.run([call("read_file", {"path": "evidence.txt"}, "read"),
           call("compact_context", {"working_note": "Read the evidence.", "keep_unit_ids": []}, "compact"),
           {"content": "done"}])
    receipt = f.ctx._context_view_receipt
    assert receipt["status"] == "applied"
    checkpoint = json.loads(read_actor_source_bytes(f.ctx.drive_root, "authored-main", receipt["checkpoint_ref"]))
    assert {"role": "user", "content": [image]} in checkpoint["messages"]
    assert {"role": "user", "content": [image]} in f.ctx.messages
    assert "image omitted" in str(f.inputs[-1]["messages"])
    assert "image_url" not in str(f.inputs[-1]["messages"])


@pytest.mark.parametrize("first", ["empty", "malformed", "rejected"])
def test_failed_or_empty_response_cannot_supply_a_false_authored_receipt(main_loop, monkeypatch, first):
    from ouroboros import loop_llm_call
    from ouroboros.tools import compact_context

    f, observed, prior = main_loop, [], {}
    record = compact_context.record_context_view

    def observe(ctx, messages, schemas, **kwargs):
        observed.append(deepcopy(messages))
        record(ctx, messages, schemas, **kwargs)

    monkeypatch.setattr(compact_context, "record_context_view", observe)
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *a, **kw: True)
    bad = {"content": ""} if first == "empty" else call("compact_context", {
        "working_note": "bad", "expected_view_revision": "unrelated"}, "bad")
    if first == "malformed":
        bad["tool_calls"][0]["function"]["arguments"] = "{"

    def resume(kwargs):
        prior["receipt"] = getattr(f.ctx, "_context_view_receipt", None)
        prior["pending"] = getattr(f.ctx, "_pending_compaction", None)
        if first != "empty":
            prior["result"] = next((m["content"] for m in kwargs["messages"] if m.get("tool_call_id") == "bad"), None)
        return call("compact_context", {"working_note": "Verified source retained.", "keep_unit_ids": []}, "good")

    f.run([call("read_file", {"path": "evidence.txt"}, "read"), bad, resume, {"content": "done"}])
    assert len(observed) == (3 if first == "empty" else 4)
    assert f.ctx._context_view_receipt["status"] == "applied"
    assert prior["receipt"] is None and prior["pending"] is None
    if first != "empty":
        expected = "mismatch" if first == "rejected" else "TOOL_ARG_ERROR"
        assert expected in str(prior["result"])


def test_real_source_drift_is_a_visible_materializer_refusal(main_loop):
    f = main_loop

    def stale(kwargs):
        seen = json.loads(next(m["content"] for m in kwargs["messages"] if m.get("tool_call_id") == "inspect"))
        next(m for m in f.ctx.messages if m.get("tool_call_id") == "read")["content"] = "Changed source"
        return call("compact_context", {"expected_view_revision": seen["view_revision"],
                    "working_note": "Old conclusion", "keep_unit_ids": []}, "stale")

    f.run([call("read_file", {"path": "evidence.txt"}, "read"),
           call("compact_context", {"inspect": True}, "inspect"), stale, {"content": "done"}])
    assert f.ctx._context_view_receipt["status"] == "binding_mismatch"
    assert f.ctx._context_view_receipt["checkpoint_ref"] is None
    after = f.inputs[-1]["messages"]
    assert any(m.get("tool_call_id") == "read" and m["content"] == "Changed source" for m in after)
    assert "binding_mismatch" in str(after) and "[Context view receipt]" in str(after)


def test_main_fit_reports_known_pressure_without_rejecting_useful_shrink(tmp_path, monkeypatch):
    from ouroboros import capability_evidence
    from ouroboros.context_budget import ContextReclaimRequest
    from ouroboros.loop_model_call import _measure_main_context_view

    monkeypatch.setattr(capability_evidence, "resolve_main_token_density", lambda *a, **kw: (1.0, "cold_estimate"))
    plan = _plan(preferred="max", window=100)
    observed = [*plan.messages_for("max"), {"role": "user", HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND,
                                            "content": "Full owner requirement. " * 200},
                call("read_file", {}, "read"), {"role": "tool", "tool_call_id": "read", "content": "raw " * 2000}]
    request = ContextReclaimRequest("route", "round", context_compaction.context_reclaim_transcript_sha256(observed),
        "cold_estimate", 1.0, 0, working_note="I understood the source.",
        expected_view_revision=context_compaction.context_reclaim_transcript_sha256(observed), keep_unit_ids=())
    fit = lambda messages, tools: _measure_main_context_view(plan, messages, tools, "max", "high", "1")
    candidate, receipt = context_compaction.compact_tool_history_llm(  # the owner's row is typed at delivery
        observed, request=request, observed_messages=observed, tool_schemas=[], fit_candidate=fit,
        drive_root=tmp_path, task_id="large-main")
    assert receipt.status == "applied" and receipt.reclaimed_tokens > 0
    assert receipt.fit["accepted"] and receipt.fit["predicted_capacity_miss"]
    assert receipt.fit["strict_bound_proven"] is False
    assert candidate[:3] == observed[:3]  # the owner's words stay whole under keep_unit_ids=()
    legacy = [*observed[:2], {"role": "user", "content": observed[2]["content"]}, *observed[3:]]  # delivered untyped
    legacy_request = replace(request, transcript_sha256=context_compaction.context_reclaim_transcript_sha256(legacy),
                             expected_view_revision=context_compaction.context_reclaim_transcript_sha256(legacy))
    untyped, untyped_receipt = context_compaction.compact_tool_history_llm(
        legacy, request=legacy_request, observed_messages=legacy, tool_schemas=[], fit_candidate=fit,
        drive_root=tmp_path, task_id="large-main")
    assert untyped_receipt.status == "applied" and untyped[:2] == legacy[:2]
    assert untyped[2] != legacy[2]  # an untyped row is an addressable dialogue unit the mind may fold (3A)
    unknown = _measure_main_context_view(replace(plan, status="unknown"), candidate, [], "max", "high", "1")
    assert unknown["accepted"] and unknown["capacity_total_tokens"] is None
    unbound = _measure_main_context_view(None, candidate, [], "max", "high", "1")
    assert unbound["accepted"] and unbound["capacity_total_tokens"] is None
    assert unbound["estimated_input_tokens"] > 0 and unbound["strict_bound_proven"] is False


def test_standalone_main_without_route_plan_can_author_a_view(main_loop):
    f = main_loop
    f.ctx.context_fit_plan = None
    f.ctx.task_model_override = "openai/test-model"
    f.run([call("read_file", {"path": "evidence.txt"}, "read"),
           call("compact_context", {"working_note": "The source is retained.", "keep_unit_ids": []}, "compact"),
           {"content": "done"}])
    assert f.ctx._context_view_receipt["status"] == "applied"
    assert f.ctx._context_view_receipt["fit"]["capacity_total_tokens"] is None


@pytest.mark.parametrize("mode", ["nano", "low", "max"])
def test_main_schema_only_view_uses_the_active_envelope(main_loop, monkeypatch, mode):
    f = main_loop
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", mode)
    f.ctx.context_fit_plan = replace(f.ctx.context_fit_plan, preferred_mode=mode, initial_mode=mode,
        nano_projection=replace(f.ctx.context_fit_plan.max_projection, mode="nano"))
    f.messages = f.ctx.context_fit_plan.messages_for(mode)
    # A tool outside Nano's standing work set (``tool_policy.NANO_RESIDENT_SCHEMA_NAMES``) arrives only when named.
    f.run([call("compact_context", {"working_note": "", "schema_names": ["query_code"]}, "schemas"),
           {"content": "done"}])
    before = {s["function"]["name"] for s in f.inputs[0]["tools"]}
    after = {s["function"]["name"] for s in f.inputs[1]["tools"]}
    if mode == "nano":
        assert "query_code" not in before and "query_code" in after
        assert f.ctx._context_view_receipt["status"] == "applied"
    else:
        assert before == after and f.ctx._context_view_receipt["status"] == "no_op"
    assert f.ctx._context_view_receipt["checkpoint_ref"] is None


def test_successful_fallback_owns_the_next_authored_view(main_loop, monkeypatch):
    import httpx
    from ouroboros import context

    f = main_loop
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai/alternate")
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **kw: (
        {"model": task["model"], "provider": "openai"},
        SimpleNamespace(route_fp="fallback-route", status="confirmed", stale=False, window_tokens=900_000)))
    refused = httpx.HTTPStatusError("fixture refusal", request=httpx.Request("POST", "https://fixture.invalid"),
                                    response=httpx.Response(400, json={"error": {"message": "fixture refusal"}}))
    f.run([call("read_file", {"path": "evidence.txt"}, "read"), refused,
           call("compact_context", {"working_note": "The fallback retained the source.", "keep_unit_ids": []}, "compact"),
           {"content": "done"}])
    assert [row["model"] for row in f.inputs] == ["openai/test-model", "openai/test-model", "openai/alternate", "openai/alternate"]
    assert f.ctx._context_view_receipt["status"] == "applied"
    assert f.ctx._context_view_receipt["fit"]["route_fp"] == "fallback-route"
    assert "The fallback retained the source." in str(f.inputs[-1]["messages"])
    checkpoint = json.loads(read_actor_source_bytes(f.ctx.drive_root, "authored-main", f.ctx._context_view_receipt["checkpoint_ref"]))
    assert checkpoint["messages"] == f.inputs[2]["messages"]


def test_main_keeps_earlier_notes_under_empty_keep_and_merges_only_named_ones(main_loop):
    """The owner-visible case (d9b66195): a second fold with keep_unit_ids=[] keeps the first note, the work's
    goal with it; a later merge names both notes, the merged note and the receipt name them, and the
    address the merged note carries restores the goal's exact text."""
    f, seen = main_loop, {}
    goal = "Goal: a pair of elegant generalizable prompt fixes; no new framework."

    def accounts(messages):
        return [block["text"] for m in messages if isinstance(m.get("content"), list)
                for block in m["content"] if isinstance(block, dict) and block.get("_context_capsule")]

    def receipt(messages):
        row = [m["content"] for m in messages if str(m.get("content")).startswith("[Context view receipt]")][-1]
        return json.loads(row.split("\n", 1)[1])

    def merge(kwargs):
        inspected = json.loads(next(m["content"] for m in kwargs["messages"] if m.get("tool_call_id") == "inspect"))
        seen["notes"] = [row["unit_id"] for row in inspected["units"] if row["kind"] == "capsule"]
        return call("compact_context", {"working_note": "Merged: the goal stands; both reads agree.",
                                        "keep_unit_ids": [], "replace_note_ids": seen["notes"]}, "merge")

    def restore(kwargs):
        merged = next(text for text in accounts(kwargs["messages"]) if "Merged: the goal stands" in text)
        seen["address"] = json.loads(merged.partition("\n")[0].split("; replaced notes: ", 1)[1][:-1])
        first = seen["address"]["notes"][0]
        return call("compact_context", {"restore_unit_refs": [
            {"checkpoint_ref": seen["address"]["checkpoint_ref"], **first}]}, "restore")

    f.run([call("read_file", {"path": "evidence.txt"}, "read-1"),
           call("compact_context", {"working_note": goal, "keep_unit_ids": []}, "fold-1"),
           call("read_file", {"path": "evidence.txt"}, "read-2"),
           call("compact_context", {"working_note": "The second read agrees.", "keep_unit_ids": []}, "fold-2"),
           call("compact_context", {"inspect": True}, "inspect"), merge, restore, {"content": "done"}])
    kept = accounts(f.inputs[4]["messages"])
    assert [text.partition("\n")[2] for text in kept] == [goal, "The second read agrees."]
    assert kept[0] == accounts(f.inputs[2]["messages"])[0]  # the goal note, byte for byte
    assert "replaced_notes" not in receipt(f.inputs[4]["messages"])
    assert not any(m.get("tool_call_id") in ("read-1", "read-2") for m in f.inputs[4]["messages"])
    merged = accounts(f.inputs[6]["messages"])
    assert [text.partition("\n")[2] for text in merged] == ["Merged: the goal stands; both reads agree."]
    assert len(seen["notes"]) == 2 and [note["unit_id"] for note in seen["address"]["notes"]] == seen["notes"]
    replaced = receipt(f.inputs[6]["messages"])["replaced_notes"]
    assert [(note["unit_id"], note["first_line"]) for note in replaced["notes"]] == list(
        zip(seen["notes"], [goal, "The second read agrees."]))
    assert replaced["checkpoint_ref"] == seen["address"]["checkpoint_ref"]
    restored = [text for text in accounts(f.inputs[7]["messages"]) if text.startswith("[Historical source")]
    assert len(restored) == 1 and goal in restored[0]


def _plain(rows):
    """A request's rows as a provider's cache compares them: the host's cache marks and text-block wrapping aside."""
    def row(message):
        out = {key: value for key, value in message.items() if key != "cache_control"}
        content = out.get("content")
        if isinstance(content, list) and content and all(isinstance(b, dict) and set(b) - {"cache_control"} == {"type", "text"}
                                                         for b in content):
            out["content"] = "".join(block["text"] for block in content)
        return out
    return [row(message) for message in rows]


def test_two_full_folds_leave_the_start_and_the_first_folds_request_in_place(main_loop):
    """Test 1 of the cheap-folds order: a full fold leaves the request's start rows byte-identical and writes its note
    after them; its own call keeps the arguments the model sent, the note's exact text included; the next full fold
    leaves every row of the request sent after the first fold, up to its last row before the model's answer, as
    that request carried it, and writes its note after them."""
    from ouroboros.context_compaction import _capsule_metadata

    f = main_loop
    first_note, second_note = "The first read holds the conclusion.", "The second read agrees with the first."
    f.run([call("read_file", {"path": "evidence.txt"}, "read-1"),
           call("compact_context", {"working_note": first_note, "keep_unit_ids": []}, "fold-1"),
           call("read_file", {"path": "evidence.txt"}, "read-2"),
           call("compact_context", {"working_note": second_note, "keep_unit_ids": []}, "fold-2"),
           {"content": "done"}])
    start, after_first, after_second = (_plain(f.inputs[i]["messages"]) for i in (0, 2, 4))
    assert [row.get(HOST_CONTEXT_KIND_KEY) for row in start[2:]] == ["ouroboros_roster_snapshot", "ouroboros_context_facts"]
    notes = [(i, meta) for i, row in enumerate(after_first) if (meta := _capsule_metadata(row)[1])]
    (at, meta), = notes
    assert at == len(start) and after_first[:at] == start  # the start's host rows stand, the note follows them
    own = after_first[at + 1]
    assert own["tool_calls"][0]["id"] == "fold-1"
    assert json.loads(own["tool_calls"][0]["function"]["arguments"]) == {"working_note": first_note, "keep_unit_ids": []}
    assert after_first[at]["content"][0]["text"].endswith("\n" + first_note)
    assert after_second[:len(after_first)] == after_first  # that whole request stands, superseded facts included
    second = after_second[len(after_first)]
    assert _capsule_metadata(second)[1] and second["content"][0]["text"].endswith("\n" + second_note)


def _roster_after_a_change(monkeypatch):
    """A roster of twelve independent roots whose titles carry a version; ``bump`` changes it as a reply is produced."""
    from ouroboros import peer_roster

    state = {"v": 0}
    monkeypatch.setattr(peer_roster, "independent_roots", lambda _root: {"roots": [
        {"task_id": f"peer-{n}", "title": f"Peer work {n} at version {state['v']} " + "x" * 80, "project_id": "",
         "chat_id": 1, "status": "running"} for n in range(12)], "incomplete": False})

    def bump(reply):
        def produce(_kwargs):
            state["v"] += 1
            return reply
        return produce
    return bump


def _rosters(rows):
    return [i for i, row in enumerate(rows) if str(row.get(HOST_CONTEXT_KIND_KEY) or "").startswith("ouroboros_roster")]


def test_a_round_that_applies_a_fold_holds_a_changed_roster_back_one_round(main_loop, monkeypatch):
    """Three directions at the loop's seam: with the roster changed, the round that applies a fold appends no roster
    row and the next round appends it; a round without a fold appends it at once; when a fold is applied in two
    consecutive rounds, the first round appends no roster row and the second does."""
    from ouroboros.context_compaction import _capsule_metadata

    f, bump = main_loop, _roster_after_a_change(monkeypatch)
    read = lambda n: call("read_file", {"path": "evidence.txt"}, f"read-{n}")  # noqa: E731
    f.run([read(1), bump(call("compact_context", {"working_note": "N1 holds read one.", "keep_unit_ids": []}, "fold-1")),
           read(2), bump(read(3)),
           bump(call("compact_context", {"working_note": "N2 holds read three.", "fold_last_n": 1}, "fold-2")),
           bump(call("compact_context", {"working_note": "N3 holds the rest.", "keep_unit_ids": []}, "fold-3")),
           {"content": "done"}])
    requests = [_plain(f.inputs[i]["messages"]) for i in range(len(f.inputs))]

    def after_last_note(rows):
        return [i for i in _rosters(rows) if i > max(i for i, row in enumerate(rows) if _capsule_metadata(row)[1])]

    def calls_before(rows, ident):
        return next(i for i, row in enumerate(rows) if ident in {c["id"] for c in row.get("tool_calls") or ()})

    assert not after_last_note(requests[2]) and after_last_note(requests[3])  # held by fold-1, then appended
    assert max(_rosters(requests[4])) > calls_before(requests[4], "read-3")  # no fold: at once
    assert not after_last_note(requests[5])  # held by fold-2
    assert after_last_note(requests[6])  # fold-3 the next round: appended anyway (that fold then stands nothing)


def test_four_full_folds_with_a_changing_roster_keep_at_most_two_snapshots_and_each_request(main_loop, monkeypatch):
    """Over four full folds with a roster that changes before each, the view never holds more than two roster rows,
    and the request sent after each fold is a prefix of the request sent after the next."""
    f, bump = main_loop, _roster_after_a_change(monkeypatch)
    script = []
    for n in range(1, 5):
        script += [bump(call("read_file", {"path": "evidence.txt"}, f"read-{n}")),
                   call("compact_context", {"working_note": f"N{n} holds read {n}.", "keep_unit_ids": []}, f"fold-{n}")]
    f.run([*script, {"content": "done"}])
    requests = [_plain(f.inputs[i]["messages"]) for i in range(len(f.inputs))]
    assert max(len(_rosters(rows)) for rows in requests) <= 2, [len(_rosters(rows)) for rows in requests]
    folded = requests[2::2]
    assert len(folded) == 4 and all(later[:len(earlier)] == earlier for earlier, later in zip(folded, folded[1:]))


def test_six_consecutive_full_folds_with_a_changing_roster_deliver_it_and_never_pin_it(main_loop, monkeypatch):
    """Six full folds in six consecutive rounds, the roster changing before each: the changed roster reaches the view
    within one round every time, no roster row ever stands as a fold's own row (a fold whose round holds one leaves
    nothing standing), and after each fold the view holds at most two roster rows."""
    from ouroboros.tools.compact_context import standing_rows

    f, bump = main_loop, _roster_after_a_change(monkeypatch)
    f.run([call("read_file", {"path": "evidence.txt"}, "read-1"), *(bump(call(
        "compact_context", {"working_note": f"N{n} holds what came before.", "keep_unit_ids": []}, f"fold-{n}"))
        for n in range(1, 7)), {"content": "done"}])
    requests = [f.inputs[i]["messages"] for i in range(len(f.inputs))]
    def versions(rows):
        return {v for i in _rosters(rows) for v in range(8) if f"at version {v} " in str(rows[i].get("content"))}

    for r in range(2, len(requests) - 1):  # request r is built after reply r - 1 set version r - 1: it or the next
        assert max(versions(requests[r]) | versions(requests[r + 1])) >= r - 1, (r, versions(requests[r]))
    for rows in requests:
        standing = standing_rows(rows)
        assert not [i for i in _rosters(rows) if standing.get(i) is not None]
        assert len(_rosters(rows)) <= 2, [len(_rosters(r)) for r in requests]
