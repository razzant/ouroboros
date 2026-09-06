"""The bounded native tool-round review executor (configured-subagent api rows).

One episode = ONE logical review attempt of ``chat(tools=…)`` calls against a
fresh instance-local inspection registry until the reviewer answers: no round
cap; the bounds are the window-derived transcript bound (with a once-only
landing notice), the owner deadline and the ledger. Exhaustion is a typed
refusal for verdict shapes and a disclosed incomplete product for the report
shape; the second actor attempt repairs format locally over the collected
answer; every read is host-observed disclosure, never a full-coverage claim.
"""

import copy
import dataclasses
import json

import pytest

from ouroboros.review_execution import (
    ReviewAssignment,
    ReviewRouteKind,
    ReviewRouteUnavailable,
    _review_route_executor,
)
from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
from ouroboros.review_substrate import ReviewRequest, ReviewSlot

_VERDICT = '[{"severity": "advisory", "item": "x", "evidence": "e", "recommendation": "r"}]'


class _ScriptedLLM:
    """chat() replays a script; captures every messages payload it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, **kwargs):
        # Snapshot: the executor mutates ONE messages list across rounds, so a
        # live reference would show every later message on every earlier call.
        self.calls.append({**kwargs, "messages": copy.deepcopy(kwargs.get("messages"))})
        if not self.script:
            raise AssertionError("script exhausted — executor made an extra call")
        entry = self.script.pop(0)
        return entry, {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0}


def _tool_call(name, args, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


@pytest.fixture()
def subject_repo(tmp_path):
    repo = tmp_path / "subject"
    repo.mkdir()
    (repo / "greeting.txt").write_text("hello native reviewer\n", encoding="utf-8")
    return repo


def _assignment(repo, llm, session_task="Review the staged change; cite files."):
    request = ReviewRequest(
        surface="multi_model_review",
        goal="review",
        task_id="t-native",
        session_root=str(repo),
        session_task=session_task,
        policy={"output_contract": "JSON array of findings"},
        no_proxy=True,
    )
    slot = ReviewSlot(
        slot_id="t1",
        model="openai/fake-reviewer",
        effort="low",
        route=ReviewRouteKind.API_CHAT,
        subagent_id="api-critic",
    )
    return ReviewAssignment(request=request, slot=slot, call_id="op-1")


def test_route_seam_selects_native_executor(subject_repo):
    executor = _review_route_executor(_assignment(subject_repo, None))
    assert isinstance(executor, NativeToolRoundReviewExecutor)


def test_episode_reads_then_answers(subject_repo):
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"})]},
        {"content": _VERDICT},
    ])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    observed_usage = []
    executor.usage_observer = observed_usage.append
    result = executor.execute()
    assert result.raw_text == _VERDICT
    assert result.message["native_transcript"] == _VERDICT
    usage = result.usage
    assert usage["native_rounds"] == 2
    assert usage["host_file_read_attestation"] == "host_observed"
    assert usage["native_tool_receipts"][0]["tool"] == "read_file"
    assert usage["native_tool_receipts"][0]["path"] == "greeting.txt"
    assert len(observed_usage) == 2
    # The REAL inspection tool ran against the pinned root: its output (with
    # the file body) went back to the model as a role=tool message.
    round2_messages = llm.calls[1]["messages"]
    tool_msgs = [m for m in round2_messages if m.get("role") == "tool"]
    assert tool_msgs and "hello native reviewer" in tool_msgs[0]["content"]
    assert tool_msgs[0]["tool_call_id"] == "call_1"
    # tools were offered on every round, from the curated inspection set only
    offered = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert "read_file" in offered and "search_code" in offered
    assert "schedule_subagent" not in offered and "write_file" not in offered


def test_second_execute_repairs_locally_without_new_episode(subject_repo):
    llm = _ScriptedLLM([
        {"content": _VERDICT},
    ])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    first = executor.execute()
    calls_after_first = len(llm.calls)
    second = executor.execute()
    # No new provider round: format repair reuses the collected answer.
    assert len(llm.calls) == calls_after_first
    assert second.raw_text == first.raw_text


def test_no_round_cap_many_rounds_then_answer(subject_repo, monkeypatch):
    """P13: the floor is hardcoded, never the ceiling — 40 inspection rounds
    (well past the retired 16-round cap) end in the reviewer's own answer."""
    monkeypatch.delenv("OUROBOROS_REVIEW_NATIVE_MAX_ROUNDS", raising=False)
    script = [
        {"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, f"c{i}")]}
        for i in range(40)
    ] + [{"content": _VERDICT}]
    llm = _ScriptedLLM(script)
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    result = executor.execute()
    assert result.raw_text == _VERDICT
    assert result.usage["native_rounds"] == 41
    assert result.usage["native_end_reason"] == "final_answer"
    assert result.usage["native_landing_notified"] is False
    assert not llm.script


def _first_send_chars(subject_repo):
    """What the first send carries (prompt + instructions + schemas), measured
    from a one-shot episode's own counter rather than re-deriving it here."""
    llm = _ScriptedLLM([{"content": _VERDICT}])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    usage = executor.execute().usage
    return usage["native_transcript_chars"] - len(_VERDICT)


def test_landing_notice_is_posted_once_at_the_landing_fraction(subject_repo, monkeypatch):
    """The host's budget fact: when the transcript crosses 80% of its bound,
    ONE typed user message tells the reviewer to answer on the next send —
    never a silent cut, never repeated on later rounds."""
    import ouroboros.review_native_episode as native_episode

    (subject_repo / "chunk.txt").write_text("y" * 4_000, encoding="utf-8")
    first_send = _first_send_chars(subject_repo)
    assert first_send >= 6_000
    # Below 80% after the first send alone, above it once one 4K read lands,
    # and with room left (below the bound minus the landing reserve) for one
    # more small tool round after the notice, so the notice's carry-over on a
    # later send is observable.
    bound = int(first_send * 1.25) + 4_000
    monkeypatch.setattr(native_episode, "review_native_transcript_bound",
                        lambda *a, **k: bound)
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "chunk.txt"}, "c1")]},
        {"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c2")]},
        {"content": _VERDICT},
    ])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    result = executor.execute()
    assert result.raw_text == _VERDICT
    notices = [
        [m for m in call["messages"] if m.get("role") == "user" and "[EPISODE_BUDGET]" in str(m.get("content"))]
        for call in llm.calls
    ]
    assert [len(n) for n in notices] == [0, 1, 1]  # posted before send 2, carried (not repeated) on send 3
    notice = notices[1][0]["content"]
    assert f"of {bound} chars" in notice and "no tool calls" in notice
    # The notice follows the tool results of the round that crossed the line.
    round2 = llm.calls[1]["messages"]
    assert round2[-1] == notices[1][0] and round2[-2]["role"] == "tool"
    assert result.usage["native_landing_notified"] is True
    assert result.usage["native_transcript_bound"] == bound


def test_transcript_bound_is_the_window_capacity_never_above_the_ceiling(monkeypatch):
    """A 1M reviewer lands on the owner ceiling; a 200K route gets the bound
    its own window can carry (density-calibrated, in chars)."""
    import ouroboros.review_native_episode as native_episode
    from ouroboros.reviewer_window import REVIEWER_FULL_WINDOW

    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "900000")
    monkeypatch.setattr(native_episode, "_CHARS_PER_ESTIMATED_TOKEN", 4)
    import ouroboros.reviewer_window as reviewer_window
    windows = {"openai/big": REVIEWER_FULL_WINDOW, "openai/small": 200_000}
    monkeypatch.setattr(reviewer_window, "reviewer_context_window",
                        lambda model_id, **_: windows[model_id])
    big = native_episode.review_native_transcript_bound("openai/big", output_reserve=16_000)
    small = native_episode.review_native_transcript_bound("openai/small", output_reserve=16_000)
    assert big == 900_000
    assert 0 < small < big
    # (200K − 16K) / 1.65 cold density ≈ 111K tokens ≈ 446K chars, below the
    # absolute-margin form and far below the ceiling.
    assert 400_000 <= small <= 460_000


def _ignores_landing(surface="multi_model_review", draft=""):
    """A reviewer that reads a 60K file under a 50K ceiling and then keeps
    calling tools after the landing notice: the first read is clamped to the
    room below the bound, the notice is posted, later results are withheld
    (empty) and only the reviewer's own envelopes grow the transcript until
    it crosses the bound — the only way a transcript can now exceed it."""
    return [
        {**({"content": draft} if draft else {}),
         "tool_calls": [_tool_call("read_file", {"path": "big.txt"}, "c1")]},
    ] + [
        {"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, f"c{i}")]}
        for i in range(2, 40)
    ]


@pytest.mark.parametrize("surface", ["multi_model_review", "task_acceptance"])
def test_transcript_bound_fails_closed_for_verdict_shapes(subject_repo, monkeypatch, surface):
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    (subject_repo / "big.txt").write_text("x" * 60_000, encoding="utf-8")
    llm = _ScriptedLLM(_ignores_landing(surface))
    assignment = _assignment(subject_repo, llm)
    assignment.request.surface = surface
    executor = NativeToolRoundReviewExecutor(assignment, llm=llm)
    with pytest.raises(ReviewRouteUnavailable) as exc:
        executor.execute()
    assert exc.value.code == "native_transcript_cap_exceeded"
    # The first read was CLAMPED to the room below the bound and disclosed as
    # nearly spent; the landing notice followed it before the next send.
    tool_msgs = [m for m in llm.calls[1]["messages"] if m.get("role") == "tool"]
    assert "RESULT TRUNCATED" in tool_msgs[0]["content"] and "answer now" in tool_msgs[0]["content"]
    assert any("[EPISODE_BUDGET]" in str(m.get("content")) for m in llm.calls[1]["messages"])
    # The settled failure replays; no second paid episode.
    with pytest.raises(ReviewRouteUnavailable):
        executor.execute()
    assert 2 <= len(llm.calls) < 39 and llm.script  # the bound landed before the script ran out


def test_one_read_can_never_jump_over_the_landing_notice_and_the_bound(subject_repo, monkeypatch):
    """The 120K per-result cap is clamped to the room below the bound: a
    single huge read lands the reviewer INSIDE the landing window with the
    notice posted, never past the bound with no notice at all."""
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    (subject_repo / "big.txt").write_text("x" * 200_000, encoding="utf-8")
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "big.txt"}, "c1")]},
        {"content": _VERDICT},
    ])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    result = executor.execute()
    assert result.raw_text == _VERDICT
    usage = result.usage
    assert usage["native_landing_notified"] is True
    assert usage["native_transcript_chars"] <= usage["native_transcript_bound"] == 50_000
    round2 = llm.calls[1]["messages"]
    assert round2[-1]["role"] == "user" and "[EPISODE_BUDGET]" in round2[-1]["content"]


def test_escaping_inflation_cannot_jump_the_landing_notice(subject_repo, monkeypatch):
    """The clamp bounds raw chars but the send carries the JSON-serialized
    message: real text (newlines, quotes, backslashes) inflates 2–5 % under
    escaping. A truncated read that filled its room is re-cut by the
    serialized overshoot, so the charged message fits the room and the
    landing notice is still posted before the bound — never a refusal the
    reviewer was not warned of."""
    import ouroboros.review_native_episode as native_episode

    line = 'def f(x):\n    return "quoted \\ value"\n'
    (subject_repo / "source.py").write_text(line * 8_000, encoding="utf-8")  # ~330K chars, escape-heavy
    first_send = _first_send_chars(subject_repo)
    bound = 50_000  # one read_file result (tool-capped near 38K) lands the transcript past 80 %
    monkeypatch.setattr(native_episode, "review_native_transcript_bound", lambda *a, **k: bound)
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "source.py"}, "c1")]},
        {"content": _VERDICT},
    ])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    result = executor.execute()
    assert result.raw_text == _VERDICT
    usage = result.usage
    assert usage["native_landing_notified"] is True and usage["native_landing_sent"] is True
    assert usage["native_transcript_chars"] <= usage["native_transcript_bound"] == bound
    # The SERIALIZED tool message fits the room it was given (bound − reserve −
    # what the first send carried), escaping inflation included.
    tool_msg = [m for m in llm.calls[1]["messages"] if m.get("role") == "tool"][0]
    assert len(json.dumps(tool_msg, ensure_ascii=False)) <= bound - 2_048 - first_send
    assert "RESULT TRUNCATED" in tool_msg["content"] and "\\n" not in tool_msg["content"]  # real text, escaped only on the wire


@pytest.mark.parametrize("wire_inflated", [False, True], ids=["room-below-landing", "escape-heavy-first-send"])
def test_bound_below_the_first_send_is_a_typed_refusal_before_any_send(subject_repo, monkeypatch, wire_inflated):
    """A bound that leaves no room to read anything must not make the landing
    notice the first thing the reviewer hears (an obedient `[]` would then be a
    strict clean verdict with zero reads): the refusal is typed and precedes
    any send. The first send is charged as the serialized objects the wire
    carries, so escape-heavy task text inflates it: a bound the raw text would
    pass but the wire would not is refused the same way."""
    import ouroboros.review_native_episode as native_episode

    llm = _ScriptedLLM([{"content": _VERDICT}])
    if wire_inflated:
        task = 'line "quoted" \\ back\n' * 2_000  # ~44K raw, far more on the wire
        executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm, session_task=task), llm=llm)
        raw = len(executor.episode_prompt) + len(native_episode._NATIVE_REVIEW_INSTRUCTIONS)
        wire = len(json.dumps([{"role": "system", "content": native_episode._NATIVE_REVIEW_INSTRUCTIONS},
                               {"role": "user", "content": executor.episode_prompt}], ensure_ascii=False))
        assert wire > raw + 2_000
        bound = raw + 12_000
    else:
        executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
        bound = _first_send_chars(subject_repo) + 200  # landing_at <= first send
    monkeypatch.setattr(native_episode, "review_native_transcript_bound", lambda *a, **k: bound)
    with pytest.raises(ReviewRouteUnavailable) as exc:
        executor.execute()
    assert exc.value.code == "native_bound_below_first_send" and not llm.calls


def test_pre_send_refusal_never_projects_a_native_execution(subject_repo, tmp_path, monkeypatch):
    """The receipt keys (resolved model/provider) are filled only from a real
    send: a refusal before the first send hands the error actor its facts via
    failure_custody, but the public execution wire must not mint a native run."""
    import ouroboros.review_native_episode as native_episode
    from ouroboros.review_execution_projection import review_executions_from_actor_usage

    first_send = _first_send_chars(subject_repo)
    monkeypatch.setattr(native_episode, "review_native_transcript_bound",
                        lambda *a, **k: first_send + 200)
    llm = _ScriptedLLM([{"content": _VERDICT}])
    assignment = dataclasses.replace(_assignment(subject_repo, llm), custody_root=tmp_path / "custody")
    executor = NativeToolRoundReviewExecutor(assignment, llm=llm)
    with pytest.raises(ReviewRouteUnavailable) as exc:
        executor.execute()
    assert exc.value.code == "native_bound_below_first_send"
    custody = executor.failure_custody()
    assert custody["delivery"] == "native_tool_rounds"
    assert custody["native_end_reason"] == "bound_below_first_send" and custody["native_rounds"] == 0
    assert custody["resolved_model"] == "" and custody["provider"] == ""
    assert custody["native_custody_row"] == "written"
    assert review_executions_from_actor_usage([{"usage": custody}]) == []
    # A refusal AFTER a paid round keeps the slot model as the resolved stand-in
    # (the historical success semantics) and therefore projects the run.
    monkeypatch.setattr(native_episode, "review_native_transcript_bound", lambda *a, **k: 900_000)
    llm = _ScriptedLLM([{"tool_calls": ["junk"]}])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    with pytest.raises(ReviewRouteUnavailable):
        executor.execute()
    custody = executor.failure_custody()
    assert custody["native_rounds"] == 1 and custody["resolved_model"] == "openai/fake-reviewer"
    assert review_executions_from_actor_usage([{"usage": custody}]) == [
        {"kind": "native", "model": "openai/fake-reviewer"},
    ]


def test_registry_failure_records_its_own_end_reason(subject_repo, tmp_path, monkeypatch):
    """A pre-loop end (no inspection registry) must not read as a transcript-bound landing."""
    def broken_registry(self, root, drive_root):
        raise ReviewRouteUnavailable("no schemas", code="native_inspection_unavailable")

    monkeypatch.setattr(NativeToolRoundReviewExecutor, "_inspection_registry", broken_registry)
    llm = _ScriptedLLM([])
    assignment = dataclasses.replace(_assignment(subject_repo, llm), custody_root=tmp_path / "custody")
    with pytest.raises(ReviewRouteUnavailable) as exc:
        NativeToolRoundReviewExecutor(assignment, llm=llm).execute()
    assert exc.value.code == "native_inspection_unavailable"
    fact = _episode_rows(tmp_path / "custody")
    assert len(fact) == 1 and fact[0]["native_end_reason"] == "registry_unavailable"
    assert fact[0]["native_rounds"] == 0 and not llm.calls


def test_multi_call_round_withholds_reads_once_the_room_is_spent(subject_repo, monkeypatch):
    """The room below the bound is enforced across the calls of ONE round: the
    whole returned message (marker included) fits the room, and a call with no
    room left is WITHHELD — not executed — with a typed marker."""
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    for name in ("a", "b", "c"):
        (subject_repo / f"{name}.txt").write_text(name * 30_000, encoding="utf-8")
    llm = _ScriptedLLM([
        {"tool_calls": [
            _tool_call("read_file", {"path": "a.txt"}, "c1"),
            _tool_call("read_file", {"path": "b.txt"}, "c2"),
            _tool_call("read_file", {"path": "c.txt"}, "c3"),
        ]},
        {"content": _VERDICT},
    ])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    result = executor.execute()
    assert result.raw_text == _VERDICT
    usage = result.usage
    assert usage["native_transcript_chars"] <= usage["native_transcript_bound"] == 50_000
    outcomes = [r["outcome"] for r in usage["native_tool_receipts"]]
    assert outcomes[0] == "executed" and "withheld" in outcomes
    tool_msgs = [m for m in llm.calls[1]["messages"] if m.get("role") == "tool"]
    # First read fits whole; the second is clamped to the room left; the third
    # finds no useful room and is withheld without being executed.
    assert "RESULT TRUNCATED" not in tool_msgs[0]["content"]
    assert "RESULT TRUNCATED" in tool_msgs[1]["content"]
    assert tool_msgs[2]["content"] == "" or "RESULT WITHHELD" in tool_msgs[2]["content"]
    assert outcomes == ["executed", "executed", "withheld"]


def test_many_withheld_calls_never_spend_the_landing_reserve(subject_repo, monkeypatch):
    """A round of many calls: withheld stubs are charged against the room and
    become empty once even a stub would not fit, so the transcript stays
    under the bound, the landing notice is still posted and the reviewer's
    next send delivers — never a cap refusal the reviewer was not warned of."""
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    (subject_repo / "chunk.txt").write_text("q" * 8_000, encoding="utf-8")
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "chunk.txt"}, f"c{i}") for i in range(12)]},
        {"content": _VERDICT},
    ])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    result = executor.execute()
    assert result.raw_text == _VERDICT
    usage = result.usage
    assert usage["native_transcript_chars"] <= usage["native_transcript_bound"] == 50_000
    assert usage["native_landing_notified"] is True and usage["native_end_reason"] == "final_answer"
    outcomes = [r["outcome"] for r in usage["native_tool_receipts"]]
    assert outcomes.count("withheld") >= 5 and outcomes[0] == "executed"
    tool_msgs = [m for m in llm.calls[1]["messages"] if m.get("role") == "tool"]
    assert any(m["content"] == "" for m in tool_msgs)  # stubs beyond the room are empty
    assert llm.calls[1]["messages"][-1]["role"] == "user"  # the landing notice rode the final send
    assert usage["native_landing_notified"] is True


def test_terminal_round_is_kept_on_a_bound_end(subject_repo, tmp_path, monkeypatch):
    """The exact assistant envelope + tool results that led to a bound end are
    not reconstructible from the receipts: a bounded redacted copy rides the
    episode facts and the custody row (P1)."""
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    (subject_repo / "big.txt").write_text("x" * 60_000, encoding="utf-8")
    llm = _ScriptedLLM(_ignores_landing())
    assignment = dataclasses.replace(_assignment(subject_repo, llm), custody_root=tmp_path / "custody")
    executor = NativeToolRoundReviewExecutor(assignment, llm=llm)
    with pytest.raises(ReviewRouteUnavailable):
        executor.execute()
    fact = _episode_rows(tmp_path / "custody")[0]
    terminal = json.loads(fact["native_terminal_round"])  # structurally valid JSON, always
    messages = terminal["messages"]
    # The decision-ending envelope leads; its tool results follow only when
    # they fitted under the bound (an unfittable envelope ends the episode
    # before it is appended).
    assert messages[0]["role"] == "assistant" and all(m["role"] in {"assistant", "tool"} for m in messages)
    assert "greeting.txt" in json.dumps(terminal)
    assert len(fact["native_terminal_round"]) <= 8_000 and terminal["omitted_tool_results"] == 0
    # A delivered episode keeps no terminal-round copy: the answer IS the record.
    llm = _ScriptedLLM([{"content": _VERDICT}])
    usage = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm).execute().usage
    assert "native_terminal_round" not in usage
    # A huge terminal round (many big results) stays valid JSON under the cap:
    # fields are bounded BEFORE serialization and old results are dropped with
    # their count disclosed.
    big = [{"role": "assistant", "content": "x" * 50_000, "tool_calls": [_tool_call("read_file", {"path": "big.txt"}, f"c{i}") for i in range(9)]}]
    big += [{"role": "tool", "tool_call_id": f"c{i}", "content": "y" * 60_000} for i in range(9)]
    doc = NativeToolRoundReviewExecutor._terminal_round_fact(big)
    parsed = json.loads(doc)
    assert len(doc) <= 8_000 and parsed["omitted_tool_results"] == 5 and len(parsed["messages"]) == 5
    assert "OMISSION NOTE" in parsed["messages"][0]["content"]
    # A hostile provider id is bounded like every other field.
    hostile = [{"role": "assistant", "content": "", "tool_calls": []},
               {"role": "tool", "tool_call_id": "z" * 5_000, "content": "ok"}]
    assert len(json.loads(NativeToolRoundReviewExecutor._terminal_round_fact(hostile))["messages"][1]["tool_call_id"]) <= 200


def test_terminal_round_is_kept_when_the_landing_notice_is_the_last_message(subject_repo, tmp_path, monkeypatch):
    """A transport failure on the very send that carries the landing notice
    (the notice is the last message) still leaves the terminal-round record:
    the guard is "an assistant round exists", not "the last message is not
    the notice"."""
    import ouroboros.review_native_episode as native_episode

    (subject_repo / "chunk.txt").write_text("y" * 4_000, encoding="utf-8")
    first_send = _first_send_chars(subject_repo)
    bound = int(first_send * 1.25) + 4_000
    monkeypatch.setattr(native_episode, "review_native_transcript_bound", lambda *a, **k: bound)

    class _FailsAfterNotice(_ScriptedLLM):
        def chat(self, **kwargs):
            if any("[EPISODE_BUDGET]" in str(m.get("content")) for m in kwargs["messages"]):
                self.calls.append(kwargs)
                raise RuntimeError("socket reset on the post-notice send")
            return super().chat(**kwargs)

    llm = _FailsAfterNotice([{"tool_calls": [_tool_call("read_file", {"path": "chunk.txt"}, "c1")]}])
    assignment = dataclasses.replace(_assignment(subject_repo, llm), custody_root=tmp_path / "custody")
    with pytest.raises(RuntimeError):
        NativeToolRoundReviewExecutor(assignment, llm=llm).execute()
    fact = _episode_rows(tmp_path / "custody")[0]
    assert fact["native_end_reason"] == "transport_error" and fact["native_landing_notified"] is True
    assert fact["native_landing_sent"] is False  # the notice was posted, but no send physically carried it
    terminal = json.loads(fact["native_terminal_round"])
    assert "chunk.txt" in fact["native_terminal_round"] and terminal["trailing_host_notice"] is True
    assert all(m["role"] in {"assistant", "tool"} for m in terminal["messages"])  # the notice is never relabelled

    # The notice's OWN charge crossing the bound (notice last, no send after it)
    # is a bound end that still records the terminal round. Tool results are
    # clamped below the reserve, so only the reviewer's own (uncapped) prose
    # can land the transcript within a notice of the bound: a round of 2.5K
    # prose plus a tiny read. Measure that transcript from the executor's own
    # counter, then set the bound 200 chars above it — the landing line
    # (bound − reserve) is crossed and the ~300-char notice pushes it over.
    prose = "p" * 2_500
    monkeypatch.setattr(native_episode, "review_native_transcript_bound", lambda *a, **k: 900_000)
    probe = _ScriptedLLM([{"content": prose, "tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c1")]}, {"content": _VERDICT}])
    after_one_round = NativeToolRoundReviewExecutor(_assignment(subject_repo, probe), llm=probe).execute().usage["native_transcript_chars"]
    monkeypatch.setattr(native_episode, "review_native_transcript_bound", lambda *a, **k: after_one_round + 200)
    llm = _ScriptedLLM([{"content": prose, "tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c1")]}])
    assignment = dataclasses.replace(_assignment(subject_repo, llm), custody_root=tmp_path / "notice-bound")
    with pytest.raises(ReviewRouteUnavailable) as exc:
        NativeToolRoundReviewExecutor(assignment, llm=llm).execute()
    assert exc.value.code == "native_transcript_cap_exceeded" and not llm.script
    fact = _episode_rows(tmp_path / "notice-bound")[0]
    assert fact["native_landing_notified"] is True
    assert "greeting.txt" in fact.get("native_terminal_round", "")
    assert json.loads(fact["native_terminal_round"])["trailing_host_notice"] is True


def test_unfittable_envelope_ends_the_episode_without_another_send(subject_repo, monkeypatch):
    """The serialized-size invariant holds on EVERY outcome: when even the
    mandatory envelope of a withheld call (the provider's exact call id) no
    longer fits under the bound, the episode ends typed — no over-bound send
    is ever made — and a report keeps its draft."""
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    (subject_repo / "big.txt").write_text("x" * 60_000, encoding="utf-8")
    huge_id = "call-" + "i" * 3_000
    calls = [_tool_call("read_file", {"path": "big.txt"}, "c0")] + [
        _tool_call("read_file", {"path": "greeting.txt"}, huge_id + str(i)) for i in range(3)
    ]
    llm = _ScriptedLLM([{"content": "# Draft\n", "tool_calls": calls}, {"content": _VERDICT}])
    assignment = _assignment(subject_repo, llm)
    assignment.request.surface = "deep_self_review"
    executor = NativeToolRoundReviewExecutor(assignment, llm=llm)
    result = executor.execute()
    assert result.raw_text == "# Draft\n" and result.usage["native_incomplete"] == "transcript_bound"
    assert len(llm.calls) == 1 and llm.script  # the over-bound send was never made
    outcomes = [r["outcome"] for r in result.usage["native_tool_receipts"]]
    assert outcomes[0] == "executed" and "withheld" in outcomes
    # Verdict shape: the same end is a typed refusal.
    llm = _ScriptedLLM([{"tool_calls": calls}, {"content": _VERDICT}])
    with pytest.raises(ReviewRouteUnavailable) as exc:
        NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm).execute()
    assert exc.value.code == "native_transcript_cap_exceeded" and len(llm.calls) == 1


def test_transcript_counter_covers_withheld_tool_message_envelopes(subject_repo, monkeypatch):
    """The counter measures what the next send carries: a batch of withheld
    (empty) tool results still costs its message envelopes (role + provider
    call id), so a giant batch is seen by the bound and refused typed before
    the over-bound send is paid for."""
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    (subject_repo / "big.txt").write_text("x" * 60_000, encoding="utf-8")
    calls = [_tool_call("read_file", {"path": "big.txt"}, "c0")] + [
        _tool_call("read_file", {"path": "greeting.txt"}, "call-" + "i" * 40 + f"-{i}") for i in range(1, 300)
    ]
    llm = _ScriptedLLM([{"tool_calls": calls}, {"content": _VERDICT}])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    with pytest.raises(ReviewRouteUnavailable) as exc:
        executor.execute()
    assert exc.value.code == "native_transcript_cap_exceeded"
    custody = executor.failure_custody()
    assert custody["native_transcript_chars"] <= custody["native_transcript_bound"] == 50_000 < custody["native_transcript_refused_chars"]
    assert len(llm.calls) == 1 and llm.script  # refused before the over-bound send was paid for


def test_report_keeps_its_draft_on_a_deadline_or_ledger_end(subject_repo, monkeypatch):
    """A report is a product: an owner deadline or the paid ledger landing
    after a paid round delivers the collected draft marked incomplete instead
    of discarding it; verdict shapes still refuse."""
    import ouroboros.review_native_episode as native_episode
    from ouroboros.usage_accounting import BudgetExceeded

    draft = "# Draft\n\n- read greeting\n"
    ticks = iter([False, True])
    monkeypatch.setattr(native_episode, "owner_deadline_exhausted", lambda **_: next(ticks))
    llm = _ScriptedLLM([{"content": draft, "tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c1")]}])
    assignment = _assignment(subject_repo, llm)
    assignment.request.surface = "deep_self_review"
    result = NativeToolRoundReviewExecutor(assignment, llm=llm).execute()
    assert result.raw_text == draft and result.usage["native_incomplete"] == "deadline_exhausted"
    assert result.usage["native_terminal_round"]  # the evidence of the interrupted round is kept
    # Verdict shape: the same deadline is a typed refusal.
    ticks = iter([False, True])
    monkeypatch.setattr(native_episode, "owner_deadline_exhausted", lambda **_: next(ticks))
    llm = _ScriptedLLM([{"content": "reading", "tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c1")]}])
    with pytest.raises(ReviewRouteUnavailable) as exc:
        NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm).execute()
    assert exc.value.code == "deadline_exhausted"
    # The paid ledger refusing the next send: the report keeps its draft too.
    monkeypatch.setattr(native_episode, "owner_deadline_exhausted", lambda **_: False)

    class _BrokeLLM(_ScriptedLLM):
        def chat(self, **kwargs):
            if len(self.calls) == 1:
                self.calls.append(kwargs)
                raise BudgetExceeded("root budget exhausted")
            return super().chat(**kwargs)

    llm = _BrokeLLM([{"content": draft, "tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c1")]}])
    assignment = _assignment(subject_repo, llm)
    assignment.request.surface = "deep_self_review"
    result = NativeToolRoundReviewExecutor(assignment, llm=llm).execute()
    assert result.raw_text == draft and result.usage["native_incomplete"] == "budget_exhausted"


def test_empty_and_malformed_terminal_rounds_are_recorded(subject_repo, tmp_path):
    """The terminal-round fact describes the decision-ending envelope itself:
    a first-round malformed answer, and a report draft followed by an EMPTY
    terminal round, both leave the actual last envelope on the record."""
    llm = _ScriptedLLM([{"tool_calls": ["junk"]}])
    assignment = dataclasses.replace(_assignment(subject_repo, llm), custody_root=tmp_path / "malformed")
    with pytest.raises(ReviewRouteUnavailable):
        NativeToolRoundReviewExecutor(assignment, llm=llm).execute()
    fact = _episode_rows(tmp_path / "malformed")[0]
    terminal = json.loads(fact["native_terminal_round"])
    assert terminal["messages"][0]["role"] == "assistant" and "junk" in terminal["messages"][0]["tool_calls"]

    draft = "# Draft\n\n- item\n"
    llm = _ScriptedLLM([
        {"content": draft, "tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c1")]},
        {"content": ""},
    ])
    assignment = dataclasses.replace(_assignment(subject_repo, llm), custody_root=tmp_path / "empty")
    assignment.request.surface = "deep_self_review"
    result = NativeToolRoundReviewExecutor(assignment, llm=llm).execute()
    assert result.usage["native_incomplete"] == "empty_answer"
    terminal = json.loads(_episode_rows(tmp_path / "empty")[0]["native_terminal_round"])
    assert terminal["messages"][0] == {"role": "assistant", "content": "", "tool_calls": "[]"}


def test_terminal_round_masks_secrets_structurally(subject_repo):
    """Secrets are masked BEFORE the structured values are flattened: a
    tool-call argument, JSON tool-call arguments and a JSON tool result that
    carry a secret-named field never reach the custody row in clear."""
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search_code",
             "arguments": json.dumps({"query": "x", "password": "hunter2"})}},
            {"id": "c2", "password": "hunter2", "function": {"name": "read_file", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"api_key": "hunter2", "hits": 1})},
        {"role": "tool", "tool_call_id": "c2", "content": "plain text, no secret"},
    ]
    doc = NativeToolRoundReviewExecutor._terminal_round_fact(messages)
    assert "hunter2" not in doc
    parsed = json.loads(doc)
    assert parsed["messages"][2]["content"] == "plain text, no secret"
    # A JSON SCALAR string (a quoted secret) and an assistant prose secret are
    # redacted as text — every shape the text redactor masks.
    from ouroboros.observability import redact_projection

    # Secrets nested INSIDE JSON strings inside JSON (any depth) are masked too.
    nested = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "search_code",
         "arguments": json.dumps({"payload": json.dumps({"password": "hunter2"})})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"wrapped": json.dumps({"api_key": "hunter2"})})}]
    assert "hunter2" not in NativeToolRoundReviewExecutor._terminal_round_fact(nested)
    # A non-list tool_calls container (malformed provider output) never breaks the record.
    for container in (7, {"id": "x", "password": "hunter2"}, "junk", 0, ""):
        doc = json.loads(NativeToolRoundReviewExecutor._terminal_round_fact(
            [{"role": "assistant", "content": "", "tool_calls": container}]))
        assert "hunter2" not in json.dumps(doc) and doc["messages"][0]["role"] == "assistant"
    token = "sk-ant-api03-" + "A" * 40
    if token not in str(redact_projection(token).value):  # the text redactor knows this shape
        scalar = [{"role": "assistant", "content": f"found {token} in config", "tool_calls": []},
                  {"role": "tool", "tool_call_id": "c1", "content": json.dumps(token)}]
        assert token not in NativeToolRoundReviewExecutor._terminal_round_fact(scalar)


@pytest.mark.parametrize("wrappers", [0, 1, 2, 3, 5, 8, 12])
def test_nested_json_secrets_fail_closed_at_any_nesting(wrappers):
    """FAIL CLOSED (P1): a low-entropy secret under a secret-named key wrapped
    in N JSON strings is masked structurally while the expansion budget lasts
    and masked WHOLESALE beyond it — never kept in clear for the text pass,
    which cannot see structure it never parsed. Pinned on both custody paths
    (a tool-call argument sits two container levels deeper than a result), so
    a shared depth counter spending the budget on traversal cannot return."""
    value = json.dumps({"password": "hunter2"})
    for _ in range(wrappers):
        value = json.dumps({"payload": value})
    via_args = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "read_file", "arguments": json.dumps({"path": "x", "note": value})}}]}]
    via_result = [{"role": "assistant", "content": "", "tool_calls": []},
                  {"role": "tool", "tool_call_id": "c1", "content": value}]
    for messages in (via_args, via_result):
        doc = NativeToolRoundReviewExecutor._terminal_round_fact(messages)
        assert "hunter2" not in doc and json.loads(doc)["messages"]
    if wrappers >= 8:  # beyond the expansion budget: masked as a whole, disclosed as such
        assert "unexpanded JSON masked" in NativeToolRoundReviewExecutor._terminal_round_fact(via_result)


def test_terminal_round_guards_never_raise_and_never_keep_structure_in_clear():
    """The container-depth guard and the parser's own recursion limit are
    fail-closed too: a hostile 80-level container and 200K unclosed brackets
    yield a valid bounded record with the deep part masked, not a crash."""
    deep = {"k": None}
    node = deep
    for _ in range(80):
        node["k"] = {"k": None}
        node = node["k"]
    node["k"] = {"password": "hunter2"}
    doc = NativeToolRoundReviewExecutor._terminal_round_fact([{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "x", "arguments": "{}"}, "extra": deep}]}])
    assert "hunter2" not in doc and "container too deep" in doc and json.loads(doc)
    brackets = NativeToolRoundReviewExecutor._terminal_round_fact([{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "x", "arguments": "[" * 200_000}}]}])
    assert json.loads(brackets)["messages"] and "unexpanded JSON masked" in brackets


def test_every_send_is_measured_exactly_on_the_wire(subject_repo, monkeypatch):
    """ONE wire measure: the counter EQUALS the serialized messages list plus
    the schemas — not a lagging sum of envelopes (two chars per message adds
    up over a long episode) — and no captured send ever exceeds the bound."""
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "400000")
    script = [{"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, f"c{i}")]} for i in range(120)]
    script.append({"content": _VERDICT})
    llm = _ScriptedLLM(script)
    usage = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm).execute().usage
    last = llm.calls[-1]  # the counter is the wire size of the LAST send (the answer is output, not a send)
    exact = (len(json.dumps(last["messages"], ensure_ascii=False, default=str))
             + len(json.dumps(last["tools"], ensure_ascii=False, default=str)))
    assert usage["native_rounds"] == 121 and usage["native_transcript_chars"] == exact
    assert "native_transcript_refused_chars" not in usage  # the refused fact belongs to bound ends only
    # An episode that ignores the landing notice and ends at the bound never
    # made a send above it — measured on every captured send, not the counter.
    bound = 50_000
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", str(bound))
    (subject_repo / "big.txt").write_text("x" * 60_000, encoding="utf-8")
    llm = _ScriptedLLM(_ignores_landing())
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    with pytest.raises(ReviewRouteUnavailable) as exc:
        executor.execute()
    assert exc.value.code == "native_transcript_cap_exceeded"
    sends = [len(json.dumps(c["messages"], ensure_ascii=False, default=str))
             + len(json.dumps(c["tools"], ensure_ascii=False, default=str)) for c in llm.calls]
    assert len(sends) > 5 and max(sends) <= bound
    # The bound-end facts are truthful: the counter IS the last send's wire
    # size (never a sentinel) and the refused next send is its own number.
    custody = executor.failure_custody()
    assert custody["native_transcript_chars"] == sends[-1] <= bound
    assert custody["native_transcript_refused_chars"] > bound and custody["native_end_reason"] == "transcript_bound"


def test_materialized_overflow_is_resolved_before_the_clock(subject_repo, monkeypatch):
    """A transcript overflow that has materialized ends the episode
    `transcript_bound` with its refused-size fact even when the owner deadline
    expires in the same instant: the clock is consulted only for an episode
    that could still send, and the refused fact is present EXACTLY when the
    end reason is transcript_bound."""
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    (subject_repo / "big.txt").write_text("x" * 60_000, encoding="utf-8")
    huge_id = "call-" + "i" * 3_000
    calls = [_tool_call("read_file", {"path": "big.txt"}, "c0")] + [
        _tool_call("read_file", {"path": "greeting.txt"}, huge_id + str(i)) for i in range(3)
    ]
    from ouroboros import review_native_episode as native_episode

    llm = _ScriptedLLM([{"tool_calls": calls}, {"content": _VERDICT}])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    # The owner deadline expires the moment the first round has returned —
    # the same instant the unfittable envelope materializes the overflow.
    monkeypatch.setattr(native_episode, "owner_deadline_exhausted", lambda **_: executor._rounds_used >= 1)
    with pytest.raises(ReviewRouteUnavailable) as exc:
        executor.execute()
    assert exc.value.code == "native_transcript_cap_exceeded"
    custody = executor.failure_custody()
    assert custody["native_end_reason"] == "transcript_bound"
    assert custody["native_transcript_refused_chars"] > 50_000 >= custody["native_transcript_chars"]
    assert len(llm.calls) == 1 and llm.script  # no send past the bound, none past the clock
    # The clock still ends an episode that could send: same fake, no overflow.
    llm = _ScriptedLLM([{"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"})]}, {"content": _VERDICT}])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    with pytest.raises(ReviewRouteUnavailable) as exc:
        executor.execute()
    assert exc.value.code == "deadline_exhausted"
    assert "native_transcript_refused_chars" not in executor.failure_custody()


def test_last_send_fact_commits_only_on_a_physical_send(subject_repo):
    """`native_transcript_chars` is the wire size of the LAST PHYSICAL send: a
    ledger refusal of the next send (nothing sent) or a transport failure with
    no positive physical capture never commits the un-dispatched candidate —
    the fact keeps the previous send, or 0 before any."""
    from ouroboros.usage_accounting import BudgetExceeded

    def _wire(call):
        return (len(json.dumps(call["messages"], ensure_ascii=False, default=str))
                + len(json.dumps(call["tools"], ensure_ascii=False, default=str)))

    class _RefusesOn(_ScriptedLLM):
        def __init__(self, script, *, failing_call, error):
            super().__init__(script)
            self.failing_call, self.error = failing_call, error

        def chat(self, **kwargs):
            if len(self.calls) + 1 == self.failing_call:
                raise self.error  # refused BEFORE dispatch: no physical attempt, no capture
            return super().chat(**kwargs)

    read = {"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"})]}
    # The first send refused by the ledger: zero rounds, fact 0.
    llm = _RefusesOn([read, {"content": _VERDICT}], failing_call=1, error=BudgetExceeded("root budget exhausted"))
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    with pytest.raises(BudgetExceeded):
        executor.execute()
    custody = executor.failure_custody()
    assert custody["native_end_reason"] == "budget_exhausted" and custody["native_rounds"] == 0
    assert custody["native_transcript_chars"] == 0 and not llm.calls
    # A later ledger refusal and a no-capture transport failure after ONE
    # successful send: the fact is that send, never the refused candidate.
    for error in (BudgetExceeded("root budget exhausted"), RuntimeError("socket reset before dispatch")):
        llm = _RefusesOn([read, {"content": _VERDICT}], failing_call=2, error=error)
        executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
        with pytest.raises(type(error)):
            executor.execute()
        custody = executor.failure_custody()
        assert custody["native_rounds"] == 1 and len(llm.calls) == 1
        assert custody["native_transcript_chars"] == _wire(llm.calls[0])


@pytest.mark.parametrize("surface", ["multi_model_review", "deep_self_review"])
def test_notice_overflow_is_resolved_before_the_clock(subject_repo, monkeypatch, surface):
    """The landing notice is materialized BEFORE the clocks: a transcript that
    lands within a notice of the bound ends `transcript_bound` with the refused
    fact even when the owner deadline expires in the same instant — for the
    verdict shape (typed refusal) and the report shape (draft kept, incomplete)."""
    import ouroboros.review_native_episode as native_episode

    prose = "p" * 2_500
    round1 = {"content": prose, "tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c1")]}
    monkeypatch.setattr(native_episode, "review_native_transcript_bound", lambda *a, **k: 900_000)
    probe = _ScriptedLLM([round1, {"content": _VERDICT}])
    after_one_round = NativeToolRoundReviewExecutor(_assignment(subject_repo, probe), llm=probe).execute().usage["native_transcript_chars"]
    monkeypatch.setattr(native_episode, "review_native_transcript_bound", lambda *a, **k: after_one_round + 200)
    llm = _ScriptedLLM([round1, {"content": _VERDICT}])
    assignment = _assignment(subject_repo, llm)
    assignment.request.surface = surface
    executor = NativeToolRoundReviewExecutor(assignment, llm=llm)
    monkeypatch.setattr(native_episode, "owner_deadline_exhausted", lambda **_: executor._rounds_used >= 1)
    if surface == "deep_self_review":
        result = executor.execute()
        usage = result.usage
        assert result.raw_text == prose and usage["native_incomplete"] == "transcript_bound"
    else:
        with pytest.raises(ReviewRouteUnavailable) as exc:
            executor.execute()
        assert exc.value.code == "native_transcript_cap_exceeded"
        usage = executor.failure_custody()
    assert usage["native_end_reason"] == "transcript_bound" and usage["native_landing_notified"] is True
    assert usage["native_transcript_refused_chars"] > after_one_round + 200 >= usage["native_transcript_chars"]
    assert len(llm.calls) == 1 and llm.script


def test_slot_logical_window_bounds_the_episode(subject_repo, tmp_path, monkeypatch):
    """The coordinator's logical window for the slot is a bound like the owner
    deadline: past it, a verdict episode refuses typed and a report keeps its
    draft; one send's transport timeout never outlives the window."""
    from types import SimpleNamespace
    import ouroboros.review_native_episode as native_episode

    # Advance only the episode clock at the transitions this test exercises.
    # Real registry preparation may outlive a sub-second window on a busy host;
    # that is a valid pre-dispatch refusal, not the retained-draft case below.
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(native_episode, "time", SimpleNamespace(monotonic=lambda: clock.now))

    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c1")]},
        {"content": _VERDICT},
    ])
    assignment = dataclasses.replace(_assignment(subject_repo, llm), custody_root=tmp_path / "window")
    executor = NativeToolRoundReviewExecutor(assignment, llm=llm)
    executor._logical_deadline_monotonic = clock.now + 0.05
    clock.now += 0.08
    with pytest.raises(ReviewRouteUnavailable) as exc:
        executor.execute()
    assert exc.value.code == "deadline_exhausted" and not llm.calls
    assert _episode_rows(tmp_path / "window")[0]["native_end_reason"] == "deadline_exhausted"

    llm = _ScriptedLLM([{"content": "# Draft\n", "tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c1")]}])
    assignment = _assignment(subject_repo, llm)
    assignment.request.surface = "deep_self_review"
    executor = NativeToolRoundReviewExecutor(assignment, llm=llm)
    executor._logical_deadline_monotonic = clock.now + 0.2  # the window is read once, at the start
    original_chat = llm.chat

    def slow_chat(**kwargs):
        clock.now += 0.3  # the round outlives the window
        return original_chat(**kwargs)

    llm.chat = slow_chat
    result = executor.execute()
    assert result.raw_text == "# Draft\n" and result.usage["native_incomplete"] == "deadline_exhausted"
    # one send never outlives the window: no floor above it (1e-6: the window is deadline - now and
    # two monotonic() reads on Windows can coincide, leaving a float tail above 0.2)
    assert llm.calls[0]["timeout"] <= 0.2 + 1e-6

    # The window expiring between the round's admission check and dispatch
    # takes the deadline path — no send with a floored timeout.
    ticks = iter([1000.0])  # the admission check sees 1000; every later reading (the clamp) sees 1002
    monkeypatch.setattr(native_episode, "time", SimpleNamespace(monotonic=lambda: next(ticks, 1002.0)))
    llm = _ScriptedLLM([{"content": _VERDICT}])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    executor._logical_deadline_monotonic = 1001.0
    with pytest.raises(ReviewRouteUnavailable) as exc:
        executor.execute()
    assert exc.value.code == "deadline_exhausted" and not llm.calls


@pytest.mark.parametrize("container", [["junk", {"id": "x"}], 7, True, "junk", {"id": "x"}, 0, False, "", {}])
def test_round_without_progress_is_a_typed_malformed_end(subject_repo, container):
    """PROGRESS FLOOR: a round with neither prose nor one well-formed tool call
    adds nothing and would re-enter the paid send forever — whatever container
    the provider returned: a non-list `tool_calls` is the same malformed round
    (typed end, truthful reason), never an untyped TypeError."""
    llm = _ScriptedLLM([
        {"tool_calls": container},  # no dict with a function name
        {"content": _VERDICT},  # never reached
    ])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    with pytest.raises(ReviewRouteUnavailable) as exc:
        executor.execute()
    assert exc.value.code == "native_round_without_progress"
    assert len(llm.calls) == 1 and llm.script


def test_report_shape_delivers_the_collected_draft_marked_incomplete(subject_repo, monkeypatch, tmp_path):
    """A report is a product, not a verdict: when the bound lands before the
    final answer, the reviewer's last draft is delivered with a typed
    `native_incomplete` fact on the usage, the MESSAGE and the custody row,
    plus a capability delta — never discarded, never compacted."""
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    (subject_repo / "big.txt").write_text("x" * 60_000, encoding="utf-8")
    draft = "# Deep self-review (draft)\n\nCRITICAL: loop.py finalization race.\n"
    llm = _ScriptedLLM(_ignores_landing("deep_self_review", draft))
    assignment = dataclasses.replace(_assignment(subject_repo, llm), custody_root=tmp_path / "custody")
    assignment.request.surface = "deep_self_review"
    executor = NativeToolRoundReviewExecutor(assignment, llm=llm)
    result = executor.execute()
    assert result.raw_text == draft and result.message["content"] == draft
    assert result.message["native_incomplete"] == "transcript_bound"
    assert result.usage["verdict_method"] == "report"
    assert result.usage["native_incomplete"] == "transcript_bound"
    assert result.usage["native_end_reason"] == "transcript_bound"
    assert result.usage["capability_delta"][0]["reason"] == "native_transcript_bound_before_final_answer"
    row = _episode_rows(tmp_path / "custody")[0]
    assert row["native_incomplete"] == "transcript_bound"
    # An exhausted report episode with NOTHING collected still fails closed.
    llm2 = _ScriptedLLM(_ignores_landing("deep_self_review"))
    assignment2 = _assignment(subject_repo, llm2)
    assignment2.request.surface = "deep_self_review"
    with pytest.raises(ReviewRouteUnavailable) as exc:
        NativeToolRoundReviewExecutor(assignment2, llm=llm2).execute()
    assert exc.value.code == "native_transcript_cap_exceeded"


def test_report_draft_survives_an_empty_final_round(subject_repo):
    """`never discarded`: an empty final round on the report shape delivers the
    draft marked incomplete instead of an empty product."""
    draft = "# Draft\n\n- item\n"
    llm = _ScriptedLLM([
        {"content": draft, "tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c1")]},
        {"content": ""},
    ])
    assignment = _assignment(subject_repo, llm)
    assignment.request.surface = "deep_self_review"
    result = NativeToolRoundReviewExecutor(assignment, llm=llm).execute()
    assert result.raw_text == draft
    assert result.usage["native_incomplete"] == "empty_answer"
    assert result.usage["capability_delta"][0]["reason"] == "native_empty_answer_before_final_answer"


def _episode_rows(custody_root):
    from ouroboros.delegate_custody import event_log_path

    rows = [json.loads(line) for line in event_log_path(custody_root).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("type") == "review_native_episode"]


def test_data_root_is_opt_in_and_never_removed(subject_repo, tmp_path, monkeypatch):
    """Default: the inspection tools see an EMPTY scratch as their data plane
    (removed after the episode). Opt-in: `policy["native_data_root"]` names
    the caller's real root, readable through the same read-only tools and
    left untouched even when the episode fails."""
    import shutil

    import ouroboros.review_native_episode as native_episode

    removed = []
    real_rmtree = shutil.rmtree
    monkeypatch.setattr(shutil, "rmtree", lambda path, **kw: (removed.append(str(path)), real_rmtree(path, **kw)))
    seen_roots = []
    original = native_episode.NativeToolRoundReviewExecutor._inspection_registry

    def spy(self, root, drive_root):
        seen_roots.append(str(drive_root))
        return original(self, root, drive_root)

    monkeypatch.setattr(native_episode.NativeToolRoundReviewExecutor, "_inspection_registry", spy)

    # Default: scratch.
    llm = _ScriptedLLM([{"content": _VERDICT}])
    NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm).execute()
    assert seen_roots[-1] in removed and "ouro-native-review-" in seen_roots[-1]

    # Opt-in: the real root is READABLE through the same tools and survives a
    # FAILED episode.
    data_root = tmp_path / "data"
    (data_root / "task_results").mkdir(parents=True)
    (data_root / "task_results" / "t.json").write_text('{"ok": true}', encoding="utf-8")
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("read_file", {"path": "task_results/t.json", "root": "runtime_data"}, "c1")]},
        {"content": _VERDICT},
    ])
    assignment = _assignment(subject_repo, llm)
    assignment.request.policy["native_data_root"] = str(data_root)
    NativeToolRoundReviewExecutor(assignment, llm=llm).execute()
    tool_msgs = [m for m in llm.calls[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs and '"ok": true' in tool_msgs[0]["content"]
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    (subject_repo / "big.txt").write_text("x" * 60_000, encoding="utf-8")
    llm = _ScriptedLLM(_ignores_landing())
    assignment = _assignment(subject_repo, llm)
    assignment.request.policy["native_data_root"] = str(data_root)
    with pytest.raises(ReviewRouteUnavailable):
        NativeToolRoundReviewExecutor(assignment, llm=llm).execute()
    assert seen_roots[-1] == str(data_root)
    assert str(data_root) not in removed
    assert (data_root / "task_results" / "t.json").read_text(encoding="utf-8") == '{"ok": true}'
    assert any("ouro-native-review-" in path for path in removed[1:])  # the scratch still went


def test_episode_fact_is_custodied_on_every_end_including_exceptions(subject_repo, tmp_path, monkeypatch):
    """One typed custody row per episode END: the transcript bound, the owner
    deadline and a transport failure all leave the row with the true reason."""
    monkeypatch.setenv("OUROBOROS_REVIEW_NATIVE_MAX_TRANSCRIPT_CHARS", "50000")
    (subject_repo / "big.txt").write_text("x" * 60_000, encoding="utf-8")
    llm = _ScriptedLLM(_ignores_landing())
    assignment = dataclasses.replace(_assignment(subject_repo, llm), custody_root=tmp_path / "bound")
    with pytest.raises(ReviewRouteUnavailable):
        NativeToolRoundReviewExecutor(assignment, llm=llm).execute()
    fact = _episode_rows(tmp_path / "bound")
    assert len(fact) == 1
    assert fact[0]["native_end_reason"] == "transcript_bound"
    assert fact[0]["native_rounds"] >= 2 and fact[0]["slot_id"] == "t1"
    assert fact[0]["native_transcript_chars"] <= fact[0]["native_transcript_bound"] == 50_000 < fact[0]["native_transcript_refused_chars"]

    # Owner deadline exhausted mid-episode (after one paid round).
    import ouroboros.review_native_episode as native_episode

    ticks = iter([False, True])
    monkeypatch.setattr(native_episode, "owner_deadline_exhausted", lambda **_: next(ticks))
    llm = _ScriptedLLM([{"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c1")]}])
    assignment = dataclasses.replace(_assignment(subject_repo, llm), custody_root=tmp_path / "deadline")
    with pytest.raises(ReviewRouteUnavailable) as exc:
        NativeToolRoundReviewExecutor(assignment, llm=llm).execute()
    assert exc.value.code == "deadline_exhausted"
    fact = _episode_rows(tmp_path / "deadline")
    assert len(fact) == 1 and fact[0]["native_end_reason"] == "deadline_exhausted"
    assert fact[0]["native_rounds"] == 1 and fact[0]["native_tool_calls"] == 1
    # The terminal round is kept on an exception end too (P1).
    assert "greeting.txt" in fact[0]["native_terminal_round"]

    # The paid ledger refusing the next send is the MONEY floor, not a transport fault.
    from ouroboros.usage_accounting import BudgetExceeded

    monkeypatch.setattr(native_episode, "owner_deadline_exhausted", lambda **_: False)

    class _BrokeLLM(_ScriptedLLM):
        def chat(self, **kwargs):
            if len(self.calls) == 1:
                self.calls.append(kwargs)
                raise BudgetExceeded("global budget exhausted")
            return super().chat(**kwargs)

    llm = _BrokeLLM([{"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c1")]}])
    assignment = dataclasses.replace(_assignment(subject_repo, llm), custody_root=tmp_path / "budget")
    executor = NativeToolRoundReviewExecutor(assignment, llm=llm)
    with pytest.raises(BudgetExceeded):
        executor.execute()
    fact = _episode_rows(tmp_path / "budget")
    assert len(fact) == 1 and fact[0]["native_end_reason"] == "budget_exhausted"
    assert executor.failure_custody()["native_end_reason"] == "budget_exhausted"

    # Transport failure on the second send.
    monkeypatch.setattr(native_episode, "owner_deadline_exhausted", lambda **_: False)

    class _FailingLLM(_ScriptedLLM):
        def chat(self, **kwargs):
            if len(self.calls) == 1:
                self.calls.append(kwargs)
                raise RuntimeError("socket reset")
            return super().chat(**kwargs)

    llm = _FailingLLM([{"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"}, "c1")]}])
    assignment = dataclasses.replace(_assignment(subject_repo, llm), custody_root=tmp_path / "transport")
    with pytest.raises(RuntimeError):
        NativeToolRoundReviewExecutor(assignment, llm=llm).execute()
    fact = _episode_rows(tmp_path / "transport")
    assert len(fact) == 1 and fact[0]["native_end_reason"] == "transport_error"
    assert fact[0]["native_rounds"] == 1


def test_a_dispatched_first_send_that_fails_is_still_a_round(subject_repo):
    """A send the ledger saw dispatched (its capture is positive) is a round of
    this episode even when the response never came back: the receipt keys
    stay filled so the public wire keeps the execution, unlike a proven
    zero-send refusal."""
    from ouroboros.review_execution_projection import review_executions_from_actor_usage

    class _Capture:
        state = "dispatched"
        attempt_id = "attempt-1"
        provider_status_code = None

    class _DispatchedFailureLLM(_ScriptedLLM):
        def chat(self, **kwargs):
            self.calls.append(kwargs)
            exc = RuntimeError("socket reset after dispatch")
            exc.physical_attempt_capture = _Capture()
            raise exc

    llm = _DispatchedFailureLLM([])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    with pytest.raises(RuntimeError):
        executor.execute()
    custody = executor.failure_custody()
    assert custody["native_rounds"] == 1 and custody["native_end_reason"] == "transport_error"
    assert custody["resolved_model"] == "openai/fake-reviewer"
    assert review_executions_from_actor_usage([{"usage": custody}]) == [
        {"kind": "native", "model": "openai/fake-reviewer"},
    ]
    # A LATER dispatched failure commits the captured attempt's own wire size
    # — it differs from the send before it — as the last physical send.

    class _SecondDispatchFails(_ScriptedLLM):
        def chat(self, **kwargs):
            if self.calls:
                self.calls.append({**kwargs, "messages": copy.deepcopy(kwargs.get("messages"))})
                exc = RuntimeError("socket reset after dispatch")
                exc.physical_attempt_capture = _Capture()
                raise exc
            return super().chat(**kwargs)

    llm = _SecondDispatchFails([{"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"})]}])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    with pytest.raises(RuntimeError):
        executor.execute()
    custody = executor.failure_custody()
    wire = [len(json.dumps(c["messages"], ensure_ascii=False, default=str))
            + len(json.dumps(c["tools"], ensure_ascii=False, default=str)) for c in llm.calls]
    assert custody["native_rounds"] == 2 and custody["native_transcript_chars"] == wire[1] != wire[0]


def test_transcript_counter_includes_system_schemas_and_args(subject_repo, monkeypatch):
    """The bound is a SEND bound, so it must measure what every send carries:
    the system instructions and tool schemas ride each provider call, and
    tool-call argument objects accumulate in the message list like results.
    A cap sized to admit the bare prompt but not prompt+system+schemas must
    therefore refuse BEFORE the first send (previously it passed — each send
    was understated by the fixed ~9K system/schema cost plus the argument
    tail). Units are chars on both sides.
    """
    llm = _ScriptedLLM([
        {"content": _VERDICT},
    ])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    # Comfortably above the episode prompt alone, strictly below what the
    # first send actually carries (prompt + instructions + tool schemas).
    # The env knob clamps at a 50K floor, so the getter is patched directly —
    # the subject is the COUNTER's coverage, not the knob's clamp.
    import ouroboros.review_native_episode as native_episode

    cap = len(executor.episode_prompt) + 100
    monkeypatch.setattr(
        native_episode, "review_native_max_transcript_chars", lambda: cap
    )
    with pytest.raises(ReviewRouteUnavailable) as exc:
        executor.execute()
    # A bound the first send already exhausts is the first-send FLOOR's typed
    # refusal (there is no room to read anything) — still before any send.
    assert exc.value.code == "native_bound_below_first_send"
    assert not llm.calls, "the send bound must refuse before paying for a send"


def test_transcript_counter_includes_replayed_reasoning(subject_repo, monkeypatch):
    """The reasoning-echo lane (DeepSeek) keeps ``reasoning_content`` on the
    canonical assistant message the loop appends, and replays it verbatim on
    every later send. The SEND bound must count it like content — previously
    the whole dict joined ``messages`` while the counter saw only content and
    tool-call JSON, so a large thinking tail drifted past the promised bound
    unmeasured.
    """
    import ouroboros.review_native_episode as native_episode

    llm = _ScriptedLLM([
        {
            "tool_calls": [_tool_call("read_file", {"path": "greeting.txt"})],
            "reasoning_content": "r" * 1_000_000,
        },
        # No second entry ON PURPOSE: the counter must refuse before send 2.
    ])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    # Generously admits the first send (prompt + ~9K system/schema cost) and
    # anything the round adds EXCEPT the megachar reasoning tail.
    cap = len(executor.episode_prompt) + 200_000
    monkeypatch.setattr(
        native_episode, "review_native_max_transcript_chars", lambda: cap
    )
    with pytest.raises(ReviewRouteUnavailable) as exc:
        executor.execute()
    assert exc.value.code == "native_transcript_cap_exceeded"
    assert len(llm.calls) == 1, "the bound must refuse before paying for send 2"


def test_uninspectable_tool_is_refused_in_episode(subject_repo):
    llm = _ScriptedLLM([
        {"tool_calls": [_tool_call("write_file", {"path": "greeting.txt", "content": "hacked"})]},
        {"content": _VERDICT},
    ])
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm)
    executor.execute()
    round2_messages = llm.calls[1]["messages"]
    tool_msgs = [m for m in round2_messages if m.get("role") == "tool"]
    assert tool_msgs and "not available" in tool_msgs[0]["content"]
    # The subject was NOT mutated.
    assert (subject_repo / "greeting.txt").read_text(encoding="utf-8") == "hello native reviewer\n"


def test_missing_session_task_refuses_typed(subject_repo):
    llm = _ScriptedLLM([])
    with pytest.raises(ReviewRouteUnavailable) as exc:
        NativeToolRoundReviewExecutor(
            _assignment(subject_repo, llm, session_task=""), llm=llm,
        ).execute()
    assert exc.value.code == "session_task_missing"


def test_read_file_receipts_carry_the_delivered_extent(subject_repo):
    """Receipt contract EXTENSION (Ф3/R8): an executed `read_file` receipt also
    carries the extent the reviewer actually received — start_line/end_line/
    total_lines/eof from the reader's own `ctx.last_read_view` stamp, never
    parsed back from the header — while every existing field and the outcome
    vocabulary (executed | refused | error | withheld) stay as they were. A
    sub-line cursor makes the first line partial, so it is not counted; a
    non-read tool and a refused read carry no extent; a read the registry
    refuses BEFORE dispatch inherits nothing from the read before it."""
    (subject_repo / "many.txt").write_text("".join(f"row {i}\n" for i in range(1, 31)), encoding="utf-8")
    llm = _ScriptedLLM([
        {"tool_calls": [
            _tool_call("read_file", {"path": "many.txt", "start_line": 5, "max_lines": 10}, "c1"),
            _tool_call("read_file", {"path": "many.txt", "start_line": 28, "max_lines": 10}, "c2"),
            _tool_call("read_file", {"path": "many.txt", "start_line": 5, "max_lines": 10, "start_char": 3}, "c3"),
            _tool_call("read_file", {"path": "nope.txt"}, "c4"),
            _tool_call("search_code", {"query": "row"}, "c5"),
            _tool_call("write_file", {"path": "many.txt", "content": "x"}, "c6"),
            # The cursor cases the extent must get right: past the first line
            # (mid-line landing), exactly at a line start, at/past the window's
            # end (empty delivery), and a start past EOF (empty delivery).
            _tool_call("read_file", {"path": "many.txt", "start_line": 5, "max_lines": 10, "start_char": 8}, "c7"),
            _tool_call("read_file", {"path": "many.txt", "start_line": 5, "max_lines": 10, "start_char": 6}, "c8"),
            _tool_call("read_file", {"path": "many.txt", "start_line": 5, "max_lines": 2, "start_char": 12}, "c9"),
            _tool_call("read_file", {"path": "many.txt", "start_line": 100}, "c10"),
            # After a SUCCESSFUL full read, three reads the registry refuses BEFORE
            # dispatch (its binding layer: path traversal) — they never reach the
            # reader, so they must inherit nothing from the read before them.
            _tool_call("read_file", {"path": "many.txt"}, "c11"),
            _tool_call("read_file", {"path": "a/../many.txt"}, "c12"),
            _tool_call("read_file", {"path": "many.txt/../many.txt"}, "c13"),
            _tool_call("read_file", {"path": "../../../etc/passwd"}, "c14"),
        ]},
        {"content": _VERDICT},
    ])
    usage = NativeToolRoundReviewExecutor(_assignment(subject_repo, llm), llm=llm).execute().usage
    r = usage["native_tool_receipts"]
    assert [x["outcome"] for x in r] == ["executed"] * 5 + ["refused"] + ["executed"] * 8
    assert {k: r[0][k] for k in ("start_line", "end_line", "total_lines", "eof")} == {"start_line": 5, "end_line": 14, "total_lines": 30, "eof": False}
    assert {k: r[1][k] for k in ("start_line", "end_line", "eof")} == {"start_line": 28, "end_line": 30, "eof": True}
    assert r[2]["start_line"] == 6 and r[2]["end_line"] == 14  # sub-line cursor lands mid "row 5": that line is partial
    for i in (3, 4, 5):  # a NOT_FOUND read renders nothing; other tools and refusals carry no extent
        assert not any(k in r[i] for k in ("start_line", "end_line", "total_lines", "eof", "opened_path", "opened_root")), r[i]
    assert r[0]["tool"] == "read_file" and r[0]["path"] == "many.txt" and r[0]["result_chars"] > 0
    assert r[0]["opened_path"] == "many.txt"  # the path the reader actually opened rides beside the model's spelling
    assert r[0]["opened_root"] == "active_workspace" and "root" not in r[0]  # the root it used, even when the model named none
    # start_char=8 skips "row 5\n" whole and lands mid "row 6": first complete line is 7.
    assert (r[6]["start_line"], r[6]["end_line"], r[6]["eof"]) == (7, 14, False)
    # start_char=6 lands exactly at the start of "row 6": it counts.
    assert (r[7]["start_line"], r[7]["end_line"]) == (6, 14)
    # A cursor at the window's end delivers nothing: an EMPTY range, never an inverted eof=True.
    assert r[8]["end_line"] < r[8]["start_line"] and r[8]["eof"] is False and r[8]["total_lines"] == 30
    # A start past EOF: empty range, eof False, total still truthful.
    assert r[9]["end_line"] < r[9]["start_line"] and r[9]["eof"] is False and r[9]["total_lines"] == 30
    # c11 delivered the whole file; c12–c14 were refused by the registry's
    # pre-dispatch binding (the tool never ran): the refusal text goes back,
    # the outcome follows control flow, and there is NO extent — not the
    # full-file extent c11 left behind.
    assert (r[10]["start_line"], r[10]["end_line"], r[10]["eof"]) == (1, 30, True)
    tool_msgs = {m["tool_call_id"]: m["content"] for m in llm.calls[1]["messages"] if m.get("role") == "tool"}
    for i, cid in ((11, "c12"), (12, "c13"), (13, "c14")):
        assert tool_msgs[cid].startswith("⚠️ READ_FILE_ERROR") and "traversal" in tool_msgs[cid]
        assert r[i]["result_chars"] == len(tool_msgs[cid]) and r[i]["outcome"] == "executed"
        assert not any(k in r[i] for k in ("start_line", "end_line", "total_lines", "eof", "opened_path", "opened_root")), r[i]


def test_read_extent_counts_only_complete_delivered_lines_from_the_stamp(subject_repo):
    """When this episode's result bound cuts the body, the extent credits ONLY
    the complete lines whose end lies inside the delivered prefix — from the
    renderer's stamped `line_ends` (ONE line definition, `str.splitlines`,
    shared by the stamp and this cut; never by parsing the header back or
    recounting newlines); the stamp carries `opened_path` (what the reader
    opened) onto the receipt; a stamp missing any fact records NO extent
    (coverage then reads `unobserved`). The binding to one call is structural
    (reset on entry, cleared before every dispatch) — no spelling comparison."""
    from types import SimpleNamespace

    from ouroboros.tools.core import _render_line_slice

    def stamped(extent, opened="f", root="active_workspace"):
        return SimpleNamespace(last_read_view={"target": "/x/" + opened, "opened_path": opened, "opened_root": root, **extent})

    content = "".join(f"line {i}\n" for i in range(1, 21))
    executor = NativeToolRoundReviewExecutor(_assignment(subject_repo, None), llm=None)
    # Cursor at 3 lands mid "line 3" (window 3..12): first complete line is 4.
    extent: dict = {}
    full = _render_line_slice("f", content, max_lines=10, start_line=3, start_char=3, extent=extent)
    assert full.count("\n", 0, extent["body_start"]) == 1 and full[extent["body_start"] - 1] == "\n"  # one header line
    assert extent["first_line"] == 4 and extent["partial_head"] is True and extent["end_line"] == 12
    assert extent["line_ends"][:3] == (11, 18, 25) and len(extent["line_ends"]) == 9  # complete lines 4..12, body-relative
    executor._inspection_ctx = stamped(extent)
    whole = {"start_line": 4, "end_line": 12, "total_lines": 20, "eof": False, "opened_path": "f", "opened_root": "active_workspace"}
    assert executor._read_extent(full, len(full)) == whole
    # The opened path and root are the reader's, whatever the model spelled — they ride the extent.
    executor._inspection_ctx = stamped(extent, opened="dir/f", root="system_repo")
    assert {k: executor._read_extent(full, len(full))[k] for k in ("opened_path", "opened_root")} == {"opened_path": "dir/f", "opened_root": "system_repo"}
    executor._inspection_ctx = stamped(extent)
    # Cut after the partial head + two complete lines + half of the next: exactly 2 lines credited.
    body = full[extent["body_start"]:]
    cut = extent["body_start"] + len("e 3\n") + len("line 4\n") + len("line 5\n") + 3
    assert body.startswith("e 3\n")
    assert executor._read_extent(full, cut) == {**whole, "end_line": 5}
    # A cut exactly at a line end credits that line; a cut inside the partial head credits nothing.
    assert executor._read_extent(full, cut - 3)["end_line"] == 5
    assert executor._read_extent(full, extent["body_start"] + 2)["end_line"] < 4
    # ONE line definition: `splitlines` separators end lines too — U+2028 (which
    # read_file does deliver; CR/CRLF never reach the renderer from read_file,
    # whose universal-newline read turns them into LF — the bare CR below is a
    # renderer-level fact only). A cursor right after "a\u2028" is a line START
    # (no partial head), and a cut after "c\r" credits two complete lines where
    # a newline recount would credit none.
    mixed = "a\u2028b\u2028c\rd\r\ne\n"
    extent = {}
    full = _render_line_slice("f", mixed, max_lines=10, start_line=1, start_char=2, extent=extent)
    assert (extent["first_line"], extent["partial_head"], extent["total_lines"], extent["line_ends"]) == (2, False, 5, (2, 4, 7, 9))
    executor._inspection_ctx = stamped(extent)
    assert executor._read_extent(full, extent["body_start"] + len("b\u2028c\r")) == {
        "start_line": 2, "end_line": 3, "total_lines": 5, "eof": False, "opened_path": "f", "opened_root": "active_workspace"}
    extent = {}
    _render_line_slice("f", mixed, max_lines=10, start_line=1, start_char=3, extent=extent)  # inside "b\u2028"
    assert (extent["first_line"], extent["partial_head"], extent["line_ends"]) == (3, True, (3, 6, 8))
    # Fail-safe: a stamp without the delivery facts, without `line_ends`, or
    # without the opened path or root records no extent.
    executor._inspection_ctx = stamped({"start_line": 3, "end_line": 12, "total_lines": 20})
    assert executor._read_extent(full, len(full)) == {}
    executor._inspection_ctx = stamped({k: v for k, v in extent.items() if k != "line_ends"})
    assert executor._read_extent(full, len(full)) == {}
    executor._inspection_ctx = SimpleNamespace(last_read_view={"target": "/x/f", "opened_root": "active_workspace", **extent})
    assert executor._read_extent(full, len(full)) == {}
    executor._inspection_ctx = SimpleNamespace(last_read_view={"target": "/x/f", "opened_path": "f", **extent})
    assert executor._read_extent(full, len(full)) == {}
    executor._inspection_ctx = SimpleNamespace(last_read_view=None)
    assert executor._read_extent(full, len(full)) == {}


def _last_read_view_sites(sources):
    """Every WRITE site of `last_read_view` in `{relative file: source}` as
    `(file, enclosing def, lineno, kind)`: `assign` for an Attribute target of
    an Assign/AnnAssign/AugAssign statement (every target of a chained
    assignment), `setattr` for a `setattr(<expr>, "last_read_view", …)` /
    `__setattr__` call, `subscript` for a `<expr>["last_read_view"] = …` store
    (`__dict__` / `vars()`). Residual porosity, disclosed (owner R45) — writes
    this scanner does NOT see: an unpacking target (`ctx.last_read_view, x =
    …`), `for` / `with` / comprehension targets, a plain-Name target (a
    class-body default `last_read_view = None` would be a declaration, not a
    write, and is skipped by design), an attribute name assembled at run time
    (`setattr(ctx, "last_" + …)`, an aliased `__setattr__`) and dict mutators
    (`ctx.__dict__.update(…)`, a `vars(ctx)` store under a non-constant key)."""
    import ast

    def enclosing(tree, lineno):
        defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.lineno <= lineno <= n.end_lineno]
        return min(defs, key=lambda n: n.end_lineno - n.lineno).name if defs else "<module>"

    sites = set()
    for rel, text in sources.items():
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr == "last_read_view":
                    sites.add((rel, enclosing(tree, node.lineno), node.lineno, "assign"))
                elif (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                      and target.slice.value == "last_read_view"):
                    sites.add((rel, enclosing(tree, node.lineno), node.lineno, "subscript"))
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                if name in ("setattr", "__setattr__") and any(
                        isinstance(a, ast.Constant) and a.value == "last_read_view" for a in node.args[:2]):
                    sites.add((rel, enclosing(tree, node.lineno), node.lineno, "setattr"))
    return sites


_CORE, _EPISODE = "ouroboros/tools/core_file_tools.py", "ouroboros/review_native_episode.py"
_LAST_READ_VIEW_WRITERS = {  # exact (file, enclosing def, kind); the site COUNT is pinned at three
    (_CORE, "_stamp_read_view", "assign"),
    (_CORE, "_read_file", "assign"),
    (_EPISODE, "_execute_inspection_call", "assign"),
}
_RESET = "    ctx.last_read_view = None\n"  # the reader's entry reset inside _read_file


def _assert_three_writers(sites):
    assert {(f, d, k) for f, d, _l, k in sites} == _LAST_READ_VIEW_WRITERS and len(sites) == 3, sorted(sites)


def test_last_read_view_has_exactly_three_writers():
    """Writer-set invariant behind the structural stamp binding: `last_read_view`
    is written at exactly three SITES — the reader's entry reset and its stamp
    (`tools/core_file_tools.py::_read_file` / `_stamp_read_view`) and the episode's
    clear-before-dispatch (`review_native_episode.py::_execute_inspection_call`)
    — and no `setattr` / `__setattr__` call or `__dict__` store names it
    anywhere in the runtime. Sites, not (file, def) members: a fourth
    assignment inside one of the three functions is a distinct site. The one
    negative case runs the SAME scanner on a mutated in-memory copy of the live
    source with one more reset injected into `_read_file` and proves the pin
    fails on it — the pin's own claim is tested, not assumed. What the scanner
    cannot see is listed on `_last_read_view_sites`."""
    import pathlib

    import ouroboros

    repo_root = pathlib.Path(ouroboros.__file__).resolve().parent.parent
    sources = {}
    for py in repo_root.rglob("*.py"):
        rel = py.relative_to(repo_root)
        if rel.parts[0] in ("tests", "web") or any(p.startswith((".", "venv")) or p == "node_modules" for p in rel.parts):
            continue
        try:
            sources[rel.as_posix()] = py.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    _assert_three_writers(_last_read_view_sites(sources))
    live = {rel: sources[rel] for rel in (_CORE, _EPISODE)}  # the three sites all live in these two files
    base = _last_read_view_sites(live)
    _assert_three_writers(base)
    core = live[_CORE]
    assert core.count(_RESET) == 1
    injected_at = core[:core.index(_RESET)].count("\n") + 2  # 1-based line of the injected reset
    live[_CORE] = core.replace(_RESET, _RESET + _RESET.rstrip("\n") + "  # injected\n")
    fourth = _last_read_view_sites(live)
    assert fourth == base | {(_CORE, "_read_file", injected_at, "assign")}
    with pytest.raises(AssertionError):
        _assert_three_writers(fourth)
