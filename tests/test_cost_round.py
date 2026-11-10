"""The cost contract of a round: the host scans exactly the request's new bytes for secrets, at every start size.

A round's request is recorded three times before it leaves (the logical request, the physical candidate and the
Claudexor request: docs/MODEL_SEND_OBSERVABILITY.md), each a request view whose parts are the elements of its
messages and tools (``shared_values``). The loop opens one receipt window per round, from a batch's end to the
round's ``llm_round`` (``llm_observability.round_window``), and that event carries what the host counted in it
(``work``): the bytes of parts scanned for secrets (``secret_scan``, repeated keys again as ``secret_scan_again``)
and of parts the CAS stored for the first time (``new_parts``, ``observability.write_part``). R1: on round 2 the two are equal and nothing is
scanned twice in the round, whether the request started with a few messages or with many (BIBLE P5: dialogue does
not wait for bookkeeping over unrelated history). A breach in today's code is a ``TURN_GAPS`` row
(tests/_turn_law.py), its numbers measured on this fixture.
"""

from __future__ import annotations

import json
import queue
from copy import deepcopy

import pytest

from ouroboros import llm_claudexor as transport
from ouroboros import usage_accounting as ua
from ouroboros.context_timing import counted, section_spans
from ouroboros.llm import LLMClient
from ouroboros.llm_observability import round_window
from ouroboros.loop_llm_call import call_llm_with_retry
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
from tests._turn_law import hold_turn_law, turn_gap
from tests.test_llm_claudexor import MODEL, Gateway, result

TOOLS = [{"type": "function", "function": {"name": f"tool_{i}", "description": "Reads one part of the project. " * 8,
                                             "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
         for i in range(20)]


def _history(turns: int) -> list:
    rows = [{"role": "system", "content": "SYSTEM governance text. " * 400}]
    for index in range(turns):
        rows += [{"role": "user", "content": f"Owner request {index}: inspect module {index}. " * 20},
                 {"role": "assistant", "content": f"Inspected module {index}; it holds one owner. " * 20}]
    return rows


def _answer(text: str) -> dict:
    return {**result(), "message": {"role": "assistant", "content": text}}


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """The real Claudexor transport over a recorded engine (tests/test_llm_claudexor.py): every round writes its
    logical request, its physical candidate and its engine request through production code."""
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.delenv(MODEL_ACCOUNTS_KEY, raising=False)
    monkeypatch.delenv("OUROBOROS_OBSERVABILITY_KEEP_RAW", raising=False)
    monkeypatch.setattr(transport.config, "CLAUDEXOR_MODEL_POLL_INTERVAL_SEC", 0.001)
    from ouroboros import observability, pricing

    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda *_: pytest.fail("Claudexor must not fetch API pricing"))
    observability._LISTED_PARTS.clear()
    gateway = Gateway(results=[_answer("Round one answer."), _answer("Round two answer.")])
    monkeypatch.setattr(transport, "ensure_owned_gateway", lambda **_kw: gateway)
    yield root
    observability._LISTED_PARTS.clear()


def _rounds(root, task_id: str, start: list) -> list:
    """Round 1 on ``start``; round 2 after its answer and one new owner message. Each round's ``llm_round``."""
    usage: dict = {}
    messages = deepcopy(start)
    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id=task_id, root_task_id=task_id)):
        for round_idx in (1, 2):
            with round_window():
                answer, _cost = call_llm_with_retry(LLMClient(), messages, MODEL, TOOLS, "high", 1, root / "logs",
                                                    task_id, round_idx, None, usage, "task")
            assert answer is not None
            messages = messages + [dict(answer, role="assistant"), {"role": "user", "content": "Continue."}]
    rows = [json.loads(line) for line in (root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("type") == "llm_round" and row.get("task_id") == task_id]


@pytest.mark.parametrize("turns", [pytest.param(turns, marks=turn_gap("R1", "round:2"), id=f"{turns}_turns")
                                   for turns in (2, 60)])
def test_r1_round_two_scans_only_the_bytes_it_adds(engine, turns):
    """R1. Round 2 scans for secrets exactly the bytes its records store for the first time, and no part twice; the
    count is the same after a short start and after a long one. The counters themselves count: round 2 always
    stores something new (its answer and the owner's next message)."""
    first, second = _rounds(engine, f"task-{turns}", _history(turns))
    assert first["work"]["secret_scan"] > 0 and second["work"]["new_parts"] > 0, (first["work"], second["work"])
    work = second["work"]
    excess = {"secret_scan": work.get("secret_scan", 0) - work["new_parts"],
              "secret_scan_again": work.get("secret_scan_again", 0)}
    hold_turn_law("R1", "round:2", excess, dict.fromkeys(excess, 0), f"round 2 of {turns} turns: {work}")


def test_the_loop_opens_one_receipt_window_per_round(tmp_path, monkeypatch):
    """The window opens at the loop's start and at each batch's end and closes once its round's model call
    returned: a call sees only the work counted since its window opened (round 2's includes the checkpoint of the
    batch it follows, which stores the batch's result), and nothing stays open after the loop."""
    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry

    seen = []
    replies = iter([{"content": None, "tool_calls": [{"id": "write-1", "type": "function", "function": {
        "name": "write_file", "arguments": json.dumps({"path": "effect.txt", "content": "one effect"})}}]},
        {"content": "Done."}])

    def fake_call(_llm, _messages, *_args, **_kwargs):
        counted("probe", key="same")  # a window carried over from the previous round would show it again
        seen.append(section_spans().get("work"))
        return {"role": "assistant", **next(replies)}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    loop.run_llm_loop(messages=[{"role": "user", "content": "write the effect"}], tools=registry,
                      llm=type("FakeLLM", (), {"default_model": lambda self: "test-model"})(), drive_logs=tmp_path,
                      emit_progress=lambda *_a, **_k: None, incoming_messages=queue.Queue(), task_id="window-task",
                      drive_root=tmp_path)
    assert (tmp_path / "effect.txt").read_text(encoding="utf-8") == "one effect"
    assert [{verb: n for verb, n in work.items() if verb.startswith("probe")} for work in seen] == [{"probe": 1}] * 2, seen
    assert seen[1].get("new_parts", 0) > 0, seen
    assert section_spans() == {}
