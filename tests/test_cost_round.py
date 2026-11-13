"""The cost contract of a round: the host scans exactly the request's new bytes for secrets, at every start size.

A round's request is recorded three times before it leaves (the logical request, the physical candidate and the
Claudexor request: docs/MODEL_SEND_OBSERVABILITY.md), each a request view whose parts are the elements of its
messages and tools (``shared_values``). The loop runs each round's model call in one receipt window
(``llm_observability.round_window``, its first step timed from the batch's end; a failed call's recovery in a window
of its own), and the round's ``llm_round``
carries what the host counted in it (``work``): the bytes of parts scanned for secrets (``secret_scan``, repeated
keys again as ``secret_scan_again``) and of parts the CAS stored for the first time (``new_parts``,
``observability.write_parts``). R1: on round 2 the two are equal and nothing is scanned twice in the round, whether
the request started with a few messages or with many (BIBLE P5: dialogue does not wait for bookkeeping over
unrelated history). A breach is a ``TURN_GAPS`` row (tests/_turn_law.py); the one this law opened with recorded
450,834 bytes scanned for 93 new after a 60-turn start, until each part's redaction became a fact kept per task.
"""

from __future__ import annotations

import json
import queue
import sys
from copy import deepcopy

import pytest

from ouroboros import llm_claudexor as transport
from ouroboros import usage_accounting as ua
from ouroboros.context_timing import counted, section_spans
from ouroboros.llm import LLMClient
from ouroboros.llm_observability import round_window
from ouroboros.loop_llm_call import call_llm_with_retry
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
from tests._turn_law import hold_work_equal, turn_gap
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
    observability._TASK_PARTS.clear()
    gateway = Gateway(results=[_answer(f"Answer {index}.") for index in (1, 2, 3)])
    monkeypatch.setattr(transport, "ensure_owned_gateway", lambda **_kw: gateway)
    yield root
    observability._TASK_PARTS.clear()


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
    """R1. Round 2 scans for secrets exactly the bytes its records store for the first time, and no part twice, after
    a short start and after a long one alike. The counters themselves count: round 2 always stores something new
    (its answer and the owner's next message). The equality is per part as the scan keys it, in its own key order: a
    part met in two orders is scanned once per order and stored once, so live ``work`` may then show more scanned."""
    first, second = _rounds(engine, f"task-{turns}", _history(turns))
    assert first["work"]["secret_scan"] > 0 and second["work"]["new_parts"] > 0, (first["work"], second["work"])
    work = second["work"]
    hold_work_equal("R1", "round:2", {"secret_scan": (work.get("secret_scan", 0), work["new_parts"]),
                                      "secret_scan_again": (work.get("secret_scan_again", 0), 0)})


def test_the_loop_opens_one_receipt_window_per_round(tmp_path, monkeypatch):
    """Each round's model call runs in its own window: a call sees only the work counted in it, round 2's receipt
    times its ``round_top`` step from the batch's end (round 1 follows no batch), and nothing stays open after the
    loop."""
    import ouroboros.loop as loop
    from ouroboros.llm_observability import round_receipt
    from ouroboros.tools.registry import ToolRegistry

    seen = []
    replies = iter([{"content": None, "tool_calls": [{"id": "write-1", "type": "function", "function": {
        "name": "write_file", "arguments": json.dumps({"path": "effect.txt", "content": "one effect"})}}]},
        {"content": "Done."}])

    def fake_call(_llm, _messages, *_args, **_kwargs):
        counted("probe", key="same")  # a window carried over from the previous round would show it again
        seen.append((section_spans()["work"]["probe"], set(round_receipt()["host_steps"]["steps"])))
        return {"role": "assistant", **next(replies)}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    monkeypatch.setattr("ouroboros.loop_nudges._claim_pre_delivery_notice", lambda *_a, **_k: "")  # the door's notice on the unexamined write: no round here
    loop.run_llm_loop(messages=[{"role": "user", "content": "write the effect"}], tools=registry,
                      llm=type("FakeLLM", (), {"default_model": lambda self: "test-model"})(), drive_logs=tmp_path,
                      emit_progress=lambda *_a, **_k: None, incoming_messages=queue.Queue(), task_id="window-task",
                      drive_root=tmp_path)
    assert (tmp_path / "effect.txt").read_text(encoding="utf-8") == "one effect"
    assert seen == [(1, set()), (1, {"round_top"})], seen
    assert section_spans() == {}


def test_a_round_s_recovery_calls_run_in_their_own_window(tmp_path, monkeypatch):
    """When the round's own call fails, its recovery (a configured fallback here, an owner-wait resend alike) runs in
    a window of its own: the answering call's ``llm_round`` carries what the recovery counted, not the failed call's
    (``probe`` 1), with no ``round_top``, and the batch's end still times the next round's own call."""
    import ouroboros.loop as loop
    from ouroboros import fallback_cooldown
    from ouroboros.llm_observability import round_receipt
    from ouroboros.tools.registry import ToolRegistry

    seen = []
    replies = iter([{"content": None, "tool_calls": [{"id": "write-1", "type": "function", "function": {
        "name": "write_file", "arguments": json.dumps({"path": "effect.txt", "content": "one effect"})}}]},
        {"content": "Done."}])

    def fake_call(_llm, _messages, model, *args, **_kwargs):
        counted("probe", key="same")
        seen.append((model, section_spans().get("work", {}).get("probe"),
                     set(round_receipt().get("host_steps", {}).get("steps", ()))))
        if model == "test-model":  # the round's own call fails before any answer
            args[7]["_last_llm_error_kind"] = "provider_unavailable"
            return None, 0.0
        return {"role": "assistant", **next(replies)}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda plan, *_a, **_kw: (plan, "max"))
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: False)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "fb/one")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    monkeypatch.setattr("ouroboros.loop_nudges._claim_pre_delivery_notice", lambda *_a, **_k: "")  # the door's notice on the unexamined write: no round here
    loop.run_llm_loop(messages=[{"role": "user", "content": "write the effect"}], tools=registry,
                      llm=type("FakeLLM", (), {"default_model": lambda self: "test-model"})(), drive_logs=tmp_path,
                      emit_progress=lambda *_a, **_k: None, incoming_messages=queue.Queue(), task_id="recovery-task",
                      drive_root=tmp_path)
    assert (tmp_path / "effect.txt").read_text(encoding="utf-8") == "one effect"
    assert seen == [("test-model", 1, set()), ("fb/one", 1, set()), ("fb/one", 1, {"round_top"})], seen
    assert section_spans() == {}


STEPS = ("model_call", "request_record", "physical_candidate", "daemon_upload", "create_operation")
RECORDS = ("llm_request", "physical_llm_candidate", "llm_claudexor_request")


def test_the_round_receipt_names_each_preparation_step_and_what_the_request_records_spent(engine):
    """``host_steps`` (S11): wall and thread CPU of each step from the call's start to the engine's operation, the
    request records' own sections (secret scanning, blobs, the publication lock's wait, the manifest's write), the
    engine calls and event-bus puts, and the process's context switches, faults and block writes over the window.
    Only facts: a call outside any window carries none of it."""
    _first, second = _rounds(engine, "task-steps", _history(2))
    steps = second["host_steps"]
    assert tuple(steps["steps"]) == STEPS
    assert all(wall >= 0 and (cpu is None or cpu >= 0) for wall, cpu in steps["steps"].values()), steps["steps"]
    assert steps["steps"]["create_operation"][1] is None, "the engine's create runs on the sender's thread"
    assert all(cpu is not None for name, (_wall, cpu) in steps["steps"].items() if name != "create_operation")
    sections = set(steps["sections_ms"])
    assert {f"{record}.{part}" for record in RECORDS for part in ("redact", "blobs", "lock_wait", "manifest_write")} <= sections
    assert {"daemon.gateway", "daemon.operations", "daemon.upload", "daemon.create", "event_put", "total"} <= sections
    assert all(isinstance(value, int) and value >= 0 for value in steps["sections_ms"].values())
    assert len(json.dumps(steps)) < 2048, "a round's line lands in an archive kept for good"
    if sys.platform != "win32":
        assert set(steps["rusage"]) == {"nvcsw", "nivcsw", "majflt", "oublock"} and min(steps["rusage"].values()) >= 0
    usage: dict = {}
    with ua.usage_scope(ua.UsageScope(drive_root=engine, task_id="task-steps", root_task_id="task-steps")):
        call_llm_with_retry(LLMClient(), _history(2), MODEL, TOOLS, "high", 1, engine / "logs", "task-steps", 3, None,
                            usage, "task")
    unwatched = [json.loads(line) for line in (engine / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    unwatched = [row for row in unwatched if row.get("type") == "llm_round" and row.get("round") == 3]
    assert unwatched and "host_steps" not in unwatched[0] and "work" not in unwatched[0]


def test_a_platform_without_rusage_counters_leaves_them_out(monkeypatch):
    """Windows has no ``resource``: the receipt keeps its steps and sections and only omits ``rusage``."""
    from ouroboros import llm_observability

    monkeypatch.setitem(sys.modules, "resource", None)  # the import fails as it does on Windows
    with round_window():
        llm_observability.host_step("model_call")
        receipt = llm_observability.round_receipt()
    assert "rusage" not in receipt["host_steps"] and set(receipt["host_steps"]["steps"]) == {"model_call"}
