"""Native context facts describe observations without granting size-veto authority."""

import copy
import dataclasses
import hashlib
import json

import pytest

from ouroboros import review_native_episode as native
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.context_compaction import context_reclaim_transcript_sha256
from ouroboros.reviewer_window import ReviewerWindow
from tests.test_native_multiwindow_review import _source
from tests.test_native_tool_round_executor import _assignment, _ScriptedLLM, _tool_call, _VERDICT


def _text(content):
    return content if isinstance(content, str) else "".join(
        block.get("text", "") for block in content if isinstance(block, dict))


def _tool_result(call, call_id):
    return next(message["content"] for message in call["messages"]
                if message.get("tool_call_id") == call_id)


def _executor(tmp_path, llm, *, task="Review the exact source.", sources=True):
    repo = tmp_path / "subject"
    repo.mkdir(exist_ok=True)
    source = repo / "source.txt"
    if not source.exists():
        source.write_text("first source fact\nlast source fact\n", encoding="utf-8")
    assignment = dataclasses.replace(_assignment(repo, llm, session_task=task), custody_root=tmp_path / "custody")
    assignment.request.policy["native_required_sources"] = [_source(source)] if sources else []
    return native.NativeToolRoundReviewExecutor(assignment, llm=llm)


class UsageReviewer(_ScriptedLLM):
    def __init__(self, script, usage):
        super().__init__(script)
        self.usage = iter(usage)

    def chat(self, **kwargs):
        self.calls.append({**kwargs, "messages": copy.deepcopy(kwargs["messages"])})
        entry = self.script.pop(0)
        return (entry(kwargs) if callable(entry) else entry), next(self.usage)


@pytest.mark.parametrize("surface", ["multi_model_review", "deep_self_review"])
def test_large_initial_assignment_and_continuation_reach_transport(tmp_path, monkeypatch, surface):
    """Initial layout hints and every historical cap value leave dispatch intact."""
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    task = 'Review all of these exact premises: "quoted" \\ line\n' * 20_000
    assert len(task) > 900_000
    answer = _VERDICT
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "source.txt"}, "read")]},
        {"content": answer},
    ])
    executor = _executor(tmp_path, llm, task=task)
    executor.assignment.request.surface = surface
    result = executor.execute()
    assert result.raw_text == answer and len(llm.calls) == 2
    assert all(native._wire_size(call["messages"], call["tools"]) > 900_000 for call in llm.calls)
    initial = llm.calls[0]["messages"][:2]
    assert task in _text(initial[1]["content"])
    assert all(call["messages"][:2] == initial for call in llm.calls)
    assert result.usage["native_read_coverage"]["status"] == "complete"
    assert "last source fact" in _tool_result(llm.calls[1], "read")
    assert [m["role"] for m in llm.calls[1]["messages"]].count("user") == 1
    assert "[EPISODE_BUDGET]" not in str(llm.calls)
    assert "native_transcript_bound" not in result.usage
    assert "native_transcript_refused_chars" not in result.usage
    assert not any(key.startswith("native_landing_") for key in result.usage)


def test_large_batch_delivers_every_read_past_the_old_transcript_limit(tmp_path, monkeypatch):
    """The shared transcript has no result-room quota; each bounded page is delivered."""
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "source.txt"}, f"read-{i}") for i in range(12)]},
        {"content": _VERDICT},
    ])
    executor = _executor(tmp_path, llm)
    source = tmp_path / "subject" / "source.txt"
    body = 'source "quote" \\ payload ' * 3_000 + "EXACT_TAIL\n"
    source.write_text(body, encoding="utf-8")
    executor.assignment.request.policy["native_required_sources"] = [_source(source)]
    result = executor.execute()
    assert result.raw_text == _VERDICT and len(llm.calls) == 2
    assert native._wire_size(llm.calls[1]["messages"], llm.calls[1]["tools"]) > 900_000
    results = [m for m in llm.calls[1]["messages"] if m.get("role") == "tool"]
    assert len(results) == 12
    assert all(body in result["content"] for result in results)
    assert all("RESULT WITHHELD" not in result["content"] for result in results)
    receipts = result.usage["native_tool_receipts"]
    assert len(receipts) == 12 and all(r["outcome"] == "executed" and r["delivered"] for r in receipts)
    assert result.usage["native_read_coverage"]["status"] == "complete"


def test_large_read_stays_a_source_preserving_page_with_explicit_continuation(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "source.txt"}, "read")]},
        {"content": _VERDICT},
    ])
    executor = _executor(tmp_path, llm)
    source = tmp_path / "subject" / "source.txt"
    body = "prefix " * 40_000 + "EXACT_UNREAD_TAIL\n"
    source.write_text(body, encoding="utf-8")
    executor.assignment.request.policy["native_required_sources"] = [_source(source)]
    result = executor.execute()
    visible = _tool_result(llm.calls[1], "read")
    assert 50_000 < len(visible) <= native._EPISODE_TOOL_RESULT_CHAR_CAP
    assert "RESULT TRUNCATED" in visible and "start_char" in visible
    assert "EXACT_UNREAD_TAIL" not in visible
    receipt = result.usage["native_tool_receipts"][0]
    assert receipt["outcome"] == "executed" and receipt["delivered"]
    assert result.usage["native_read_coverage"]["status"] == "incomplete"
    assert source.read_text(encoding="utf-8") == body


def test_initial_cache_marker_covers_complete_assignment_and_stays_fixed(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "5m")
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "source.txt"}, "read")]},
        {"tool_calls": [_tool_call("compact_context", {"inspect": True}, "inspect")]},
        {"content": _VERDICT},
    ])
    executor = _executor(tmp_path, llm)
    executor.execute()
    prefix = llm.calls[0]["messages"][:2]
    assignment = prefix[1]["content"]
    assert isinstance(assignment, list)
    marked = [block for block in assignment if block.get("cache_control")]
    assert len(marked) == 1 and marked[0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    manifest, _ = json.JSONDecoder().raw_decode(_text(assignment).split("Required source manifest: ", 1)[1])
    raw = read_actor_source_bytes(executor.assignment.custody_root, "t-native", manifest)
    assert hashlib.sha256(raw).hexdigest() == manifest["sha256"]
    assert json.loads(raw) == executor.assignment.request.policy["native_required_sources"]
    assert executor.episode_prompt in _text(assignment)
    for call in llm.calls:
        assert call["messages"][:2] == prefix
        markers = [block for message in call["messages"] if isinstance(message.get("content"), list)
                   for block in message["content"] if isinstance(block, dict) and block.get("cache_control")]
        assert markers == marked
        assert all(not any(key.startswith("_") for key in block) for block in assignment)


def test_inspect_reports_previous_request_not_accumulated_or_cache_added_input(tmp_path, monkeypatch):
    from ouroboros import reviewer_window

    window = ReviewerWindow(200_000, "asserted", True, "2026-10-01T00:00:00Z",
                            "openai/fake-reviewer", asserted_window_tokens=210_000)
    monkeypatch.setattr(reviewer_window, "resolve_reviewer_window", lambda *a, **k: window)
    usage = [
        {"prompt_tokens": 300, "cached_tokens": 250, "resolved_model": "observed-first", "provider": "openrouter", "ledger_attempt_ids": ["first"]},
        {"prompt_tokens": 800, "cached_tokens": 700, "resolved_model": "observed-second", "provider": "openrouter", "ledger_attempt_ids": ["second"]},
        {"cost": 0.0},
    ]
    llm = UsageReviewer([
        {"tool_calls": [_tool_call("read_file", {"path": "source.txt"}, "read")]},
        {"tool_calls": [_tool_call("compact_context", {"inspect": True}, "inspect")]},
        {"content": _VERDICT},
    ], usage)
    executor = _executor(tmp_path, llm)
    result = executor.execute()
    inspected = json.loads(_tool_result(llm.calls[-1], "inspect"))
    facts = inspected["context"]
    assert facts["window"] == dataclasses.asdict(window)
    assert facts["measurement_basis"] == "canonical_serialized_characters"
    assert facts["serialized_chars"] == native._wire_size(llm.calls[1]["messages"], llm.calls[1]["tools"])
    previous = facts["previous_request"]
    assert previous["prompt_tokens"] == 800 and previous["round"] == 2
    assert previous["ledger_attempt_ids"] == ["second"]
    assert previous["model"] == "observed-second" and previous["provider"] == "openrouter"
    assert previous["view_revision"] == inspected["view_revision"] == facts["view_revision"]
    assert previous["requested_model"] == executor.assignment.slot.model
    assert facts["output_allowance_tokens"] == llm.calls[1]["max_tokens"]
    assert result.usage["prompt_tokens"] == 1_100
    assert result.usage["native_context"]["previous_request"]["prompt_tokens"] is None
    assert "free_tokens" not in facts and "current_input_tokens" not in facts


@pytest.mark.parametrize("usage", [{}, {"prompt_tokens": 12, "reported_tokens": {"prompt_tokens": None}}, {"prompt_tokens": True}])
def test_missing_usage_and_unknown_window_are_unknown_not_zero(tmp_path, monkeypatch, usage):
    from ouroboros import reviewer_window

    monkeypatch.setattr(reviewer_window, "resolve_reviewer_window", lambda *a, **k: ReviewerWindow(model="openai/fake-reviewer"))
    llm = UsageReviewer([
        {"tool_calls": [_tool_call("compact_context", {"inspect": True}, "inspect")]},
        {"content": _VERDICT},
    ], [usage, {}])
    executor = _executor(tmp_path, llm, sources=False)
    assert executor._context_facts()["previous_request"] is None
    result = executor.execute()
    facts = json.loads(_tool_result(llm.calls[1], "inspect"))["context"]
    assert facts["window"]["status"] == "unknown"
    assert facts["window"]["window_tokens"] is None and facts["window"]["asserted_window_tokens"] is None
    assert facts["previous_request"]["prompt_tokens"] is None
    assert result.usage["native_context"]["previous_request"]["prompt_tokens"] is None


@pytest.mark.parametrize("raw_usage, expected", [
    (None, None),
    ({"input_tokens": 77, "output_tokens": 1}, None),
    ({"input_tokens": 0, "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0, "output_tokens": 1}, 0),
    ({"input_tokens": 7, "cache_read_input_tokens": 100,
      "cache_creation_input_tokens": 20, "output_tokens": 1}, 127),
])
def test_native_inspection_respects_reported_anthropic_input(tmp_path, monkeypatch, raw_usage, expected):
    """Billing zero/partial totals cannot replace missing diagnostic evidence."""
    from ouroboros import pricing, reviewer_window
    from ouroboros.llm import LLMClient

    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **k: None)
    monkeypatch.setattr(reviewer_window, "resolve_reviewer_window", lambda *a, **k: ReviewerWindow())
    response = {"content": [{"type": "tool_use", "id": "inspect", "name": "compact_context",
                             "input": {"inspect": True}}]}
    if raw_usage is not None:
        response["usage"] = raw_usage
    model = "anthropic::claude-fixture"
    message, usage = LLMClient(api_key="fixture-not-live")._normalize_anthropic_response(
        response, {"provider": "anthropic", "resolved_model": "claude-fixture", "usage_model": model})
    before = copy.deepcopy(usage)
    llm = UsageReviewer([message, {"content": _VERDICT}], [usage, {}])
    executor = _executor(tmp_path, llm, sources=False)
    executor.assignment = dataclasses.replace(executor.assignment,
        slot=dataclasses.replace(executor.assignment.slot, model=model))
    executor.execute()
    fact = json.loads(_tool_result(llm.calls[1], "inspect"))["context"]["previous_request"]
    assert fact["prompt_tokens"] == expected
    assert fact["provider"] == "anthropic" and fact["model"] == model
    assert usage == before  # diagnostics never rewrite the billing counters
    if expected is None:
        assert usage["prompt_tokens"] in (0, 77)


def test_authored_note_without_inspect_applies_even_when_view_still_exceeds_old_limit(tmp_path, monkeypatch):
    from ouroboros.tools.compact_context import get_tools

    note = "I read the exact source and retain its first and last facts."
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "source.txt"}, "read")]},
        {"tool_calls": [_tool_call("compact_context", {"working_note": note, "keep_unit_ids": []}, "compact")]},
        {"content": _VERDICT},
    ])
    executor = _executor(tmp_path, llm, task="immutable assignment " * 48_000)
    result = executor.execute()
    delivered = next(tool["function"] for tool in llm.calls[0]["tools"]
                     if tool["function"]["name"] == "compact_context")
    assert delivered == get_tools()[0].schema
    guidance = delivered["parameters"]["properties"]["working_note"]["description"]
    assert "future reader" in guidance and "within those sources" in guidance
    assert "unaffected branches" in guidance and "invent no detail" in guidance
    receipt = executor._view_receipt
    assert receipt["status"] == "applied" and result.usage["native_view_changes"] == 1
    after = llm.calls[-1]
    assert native._wire_size(after["messages"], after["tools"]) > 900_000
    assert note in str(after["messages"])
    assert not any(m.get("tool_call_id") == "read" for m in after["messages"])
    assert any(m.get("tool_call_id") == "compact" for m in after["messages"])
    assert after["messages"][:2] == llm.calls[0]["messages"][:2]
    assert result.usage["native_read_coverage"]["status"] == "complete"
    checkpoint_bytes = read_actor_source_bytes(executor.assignment.custody_root, "t-native", receipt["checkpoint_ref"])
    assert hashlib.sha256(checkpoint_bytes).hexdigest() == receipt["checkpoint_ref"]["sha256"]
    checkpoint = json.loads(checkpoint_bytes)
    assert "last source fact" in str(checkpoint["messages"])
    assert len(llm.calls) == 3 and not llm.script


def test_authored_keep_last_n_keeps_the_exact_requested_tail_without_helper(tmp_path, monkeypatch):
    llm = _ScriptedLLM([
        *({"tool_calls": [_tool_call("read_file", {"path": "source.txt"}, f"read-{i}")]} for i in range(4)),
        {"tool_calls": [_tool_call("compact_context", {"working_note": "All four reads agree.", "keep_last_n": 2}, "compact")]},
        {"content": _VERDICT},
    ])
    executor = _executor(tmp_path, llm)
    executor.execute()
    assert executor._view_receipt["status"] == "applied"
    assert [m["tool_call_id"] for m in llm.calls[-1]["messages"] if m.get("role") == "tool"] == ["read-2", "read-3", "compact"]


def test_restore_with_authored_note_retains_original_source_and_checkpoint_identity(tmp_path):
    restored = {}

    def compact(kwargs):
        inspected = json.loads(_tool_result(kwargs, "inspect"))
        unit = next(unit for unit in inspected["units"] if unit["kind"] == "tool")
        restored.update({"checkpoint_ref": inspected["restore_checkpoint"], "unit_id": unit["unit_id"],
                         "raw_sha256": unit["raw_sha256"]})
        return {"tool_calls": [_tool_call("compact_context", {
            "working_note": "Retain the verified source in its checkpoint.", "keep_unit_ids": [],
            "expected_view_revision": inspected["view_revision"],
        }, "compact")]}

    def restore(kwargs):
        assert not any(m.get("tool_call_id") == "read" for m in kwargs["messages"])
        return {"tool_calls": [_tool_call("compact_context", {"working_note": "", "restore_unit_refs": [restored]}, "restore")]}

    llm = UsageReviewer([
        {"tool_calls": [_tool_call("read_file", {"path": "source.txt"}, "read")]},
        {"tool_calls": [_tool_call("compact_context", {"inspect": True}, "inspect")]},
        compact, restore, {"content": _VERDICT},
    ], [{"cost": 0.0}] * 5)
    executor = _executor(tmp_path, llm)
    result = executor.execute()
    assert result.raw_text == _VERDICT and result.usage["native_view_changes"] == 2
    assert "last source fact" in str(llm.calls[-1]["messages"])
    raw = read_actor_source_bytes(executor.assignment.custody_root, "t-native", restored["checkpoint_ref"])
    assert hashlib.sha256(raw).hexdigest() == restored["checkpoint_ref"]["sha256"]
    assert "last source fact" in str(json.loads(raw)["messages"])


def test_native_discloses_unsupported_options_and_applies_the_valid_note(tmp_path):
    args = {"working_note": "The exact source remains recoverable.", "keep_unit_ids": [],
            "schema_names": ["read_file"], "review_notes": [{"note": "unsupported"}],
            "review_transfers": [{"source": "unsupported"}]}
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "source.txt"}, "read")]},
        {"tool_calls": [_tool_call("compact_context", args, "compact")]},
        {"content": _VERDICT},
    ])
    executor = _executor(tmp_path, llm)
    result = executor.execute()
    visible = _tool_result(llm.calls[-1], "compact")
    assert "Not applied by native review" in visible
    assert all(key in visible for key in ("schema_names", "review_notes", "review_transfers"))
    assert executor._view_receipt["status"] == "applied" and result.usage["native_view_changes"] == 1
    assert args["working_note"] in str(llm.calls[-1]["messages"])
    assert llm.calls[-1]["tools"] == llm.calls[0]["tools"]


def test_previous_input_observation_keeps_its_old_view_after_authored_compaction(tmp_path):
    observed = {}
    executor = None

    def next_send(kwargs):
        observed.update(executor._context_facts(kwargs["messages"], kwargs["tools"]))
        return {"content": _VERDICT}

    llm = UsageReviewer([
        {"tool_calls": [_tool_call("read_file", {"path": "source.txt"}, "read")]},
        {"tool_calls": [_tool_call("compact_context", {"working_note": "The source is checked.", "keep_unit_ids": []}, "compact")]},
        next_send,
    ], [{"prompt_tokens": 100}, {"prompt_tokens": 400, "resolved_model": "old-model", "provider": "old-provider"}, {}])
    executor = _executor(tmp_path, llm)
    executor.execute()
    previous = observed["previous_request"]
    assert previous["prompt_tokens"] == 400 and previous["round"] == 2
    assert previous["model"] == "old-model" and previous["provider"] == "old-provider"
    assert previous["view_revision"] == context_reclaim_transcript_sha256(llm.calls[1]["messages"])
    assert observed["view_revision"] == context_reclaim_transcript_sha256(llm.calls[2]["messages"])
    assert previous["view_revision"] != observed["view_revision"]
    assert "free_tokens" not in observed


def test_provider_context_refusal_keeps_its_original_failure_without_new_recovery(tmp_path, monkeypatch):
    import httpx
    import openai

    error = openai.BadRequestError(
        "maximum context length exceeded",
        response=httpx.Response(400, request=httpx.Request("POST", "https://provider.invalid/chat/completions")),
        body={"error": {"code": "context_length_exceeded"}},
    )

    class RefusingProvider(_ScriptedLLM):
        def chat(self, **kwargs):
            self.calls.append(kwargs)
            raise error

    llm = RefusingProvider([])
    executor = _executor(tmp_path, llm, sources=False)
    with pytest.raises(openai.BadRequestError) as raised:
        executor.execute()
    assert raised.value is error and len(llm.calls) == 1
    assert executor.failure_custody()["native_end_reason"] == "transport_error"
    with pytest.raises(openai.BadRequestError):
        executor.execute()
    assert len(llm.calls) == 1


@pytest.mark.parametrize("request_form", ["naming_a_note", "saved_without_the_field"])
def test_main_and_native_review_apply_one_request_to_the_same_notes(tmp_path, monkeypatch, request_form):
    """Both executors of a queued compact_context request keep and replace the same notes and name the
    same replaced notes in their receipts; a request saved before replace_note_ids names none in either."""
    from types import SimpleNamespace

    from ouroboros.context_compaction import _capsule_metadata, context_units
    from ouroboros.loop_round_limits import _run_authored_context_view
    from ouroboros.tools.compact_context import _compact_context, record_context_view
    from tests.test_compaction import _large_unit, _note
    from tests.test_context_view_tool import _context

    goal, second = _note(tmp_path, "Goal: a few generalizable fixes.", "goal"), _note(tmp_path, "Second note.", "second")
    view = [goal, *_large_unit("read", "r"), second]
    first = context_units(view, scope="dialogue")[0]
    ctx = SimpleNamespace(active_context_mode="low")
    record_context_view(ctx, view, [])
    named = {"replace_note_ids": [first.unit_id]} if request_form == "naming_a_note" else {}
    assert "requested" in _compact_context(ctx, working_note="Merged account.", keep_unit_ids=[], **named)
    pending = copy.deepcopy(ctx._pending_compaction)
    if request_form == "saved_without_the_field":
        del pending["replace_note_ids"]

    main = _run_authored_context_view(copy.deepcopy(view), _context(tmp_path / "main", ctx, [], monkeypatch),
                                      copy.deepcopy(pending), None)
    (tmp_path / "native").mkdir()
    executor = _executor(tmp_path / "native", _ScriptedLLM([]))
    executor._inspection_ctx = SimpleNamespace(_pending_compaction=copy.deepcopy(pending))
    reviewed = copy.deepcopy(view)
    assert executor._apply_working_view(reviewed, [], 1)

    main_receipt, native_receipt = ctx._context_view_receipt, executor._view_receipt
    assert main_receipt["status"] == native_receipt["status"] == "applied"
    for key in ("selected_unit_ids", "retained_unit_ids"):
        assert list(main_receipt[key]) == list(native_receipt[key])
    main_notice = json.loads(main[-1]["content"].split("\n", 1)[1])
    native_notice = json.loads(reviewed[-1]["content"])
    from ouroboros.tools.compact_context import residue_by_kind
    for rows, notice in ((main, main_notice), (reviewed, native_notice)):
        assert notice.get("residue") == residue_by_kind(rows[:-1])
        assert notice["residue"]["capsule"]["units"] == (2 if named else 3)
        assert "tool" not in notice["residue"]
        assert notice["basis"] == "host estimates"
    named_notes = [(source.get("replaced_notes") or {}).get("notes")
                   for source in (main_receipt, native_receipt, main_notice, native_notice)]
    expected = ([{"unit_id": first.unit_id, "raw_sha256": first.raw_sha256,
                  "first_line": "Goal: a few generalizable fixes."}] if named else None)
    assert named_notes == [expected] * 4

    def accounts(rows):
        return [row["content"][0]["text"].partition("\n")[2] for row in rows if _capsule_metadata(row)[1]]

    assert accounts(main) == accounts(reviewed) == (["Merged account.", "Second note."] if named
                                                    else ["Goal: a few generalizable fixes.", "Merged account.", "Second note."])
    assert all(second in rows and (goal in rows) is not bool(named) for rows in (main, reviewed))
