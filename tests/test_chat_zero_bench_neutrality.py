"""Closing the chat-id truthiness class must not move a benchmark's answer.

Terminal-Bench reads a run's final answer straight out of ``chat.jsonl``: the
LAST untyped row with ``direction == "out"`` (``atif._final_answer``). It ALSO
reads ``progress.jsonl`` and publishes those rows as the agent's own narration
(``atif.build_trajectory``). Headless benchmark roots run in the hidden partition
(chat 0), which is exactly the partition the class fix stops dropping — so a
notice that now reaches chat 0 must be a typed or system chat row (invisible to
the answer reader), and a live host TOAST must not be written there at all, or a
supervisor line would be published as something the model said.
"""

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from devtools.benchmarks.terminal_bench.atif import _final_answer  # noqa: E402
from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID  # noqa: E402


def _agent_dir(tmp_path, chat_rows):
    data = tmp_path / "ouroboros-data"
    (data / "logs").mkdir(parents=True)
    (data / "logs" / "chat.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chat_rows),
        encoding="utf-8",
    )
    return tmp_path


def test_progress_notices_never_enter_the_answer_stream(tmp_path):
    """The reaper/scheduler toasts the class fix un-drops are progress rows.

    ``send_with_budget(is_progress=True)`` appends to progress.jsonl, so a
    grace toast for a hidden-partition root cannot be mistaken for its answer.
    """
    from types import SimpleNamespace

    from supervisor import message_bus

    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    sent = []
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(message_bus, "DATA_DIR", data)
        monkey.setattr(
            message_bus, "_BRIDGE",
            SimpleNamespace(send_message=lambda *a, **k: sent.append((a, k))),
        )
        message_bus.send_with_budget(
            HIDDEN_CHAT_ID, "⏱️ Task t1 has been running 600s", is_progress=True, task_id="t1",
        )
    finally:
        monkey.undo()
    assert sent, "the notice still reaches the live bridge"
    assert (data / "logs" / "progress.jsonl").exists()
    assert not (data / "logs" / "chat.jsonl").exists()


def test_a_system_incident_row_is_not_read_as_the_answer(tmp_path):
    """A provider-outage notice now reaches chat 0 instead of vanishing (P1).

    It is persisted as a SYSTEM row, so the trajectory still reports the task's
    own last answer rather than the incident sentence.
    """
    agent_dir = _agent_dir(tmp_path, [
        {"direction": "in", "chat_id": HIDDEN_CHAT_ID, "text": "solve it"},
        {"direction": "out", "chat_id": HIDDEN_CHAT_ID, "text": "the real answer"},
        {"direction": "system", "chat_id": HIDDEN_CHAT_ID, "type": "terminal_incident",
         "text": "🔌 Task t1 was stopped by a model-provider outage"},
    ])
    assert _final_answer(agent_dir) == "the real answer"


def test_typed_delivery_rows_after_the_answer_are_still_skipped(tmp_path):
    agent_dir = _agent_dir(tmp_path, [
        {"direction": "out", "chat_id": HIDDEN_CHAT_ID, "text": "the real answer"},
        {"direction": "out", "chat_id": HIDDEN_CHAT_ID, "type": "task_summary",
         "text": "Done with warnings"},
    ])
    assert _final_answer(agent_dir) == "the real answer"


def test_the_extraction_rule_every_producer_must_respect(tmp_path):
    """States the rule the rest of this suite enforces, so it cannot be misread.

    The reader takes the LAST untyped outbound row, whatever it says. That is
    not a contract anyone should satisfy — it is the constraint every notice
    producer has to route around, by being a progress row, a typed row or a
    system row. The assertion below is what happens when a producer does NOT,
    which is why the deep-review acknowledgement was typed rather than left
    plain once the class fix let it reach this partition.
    """
    agent_dir = _agent_dir(tmp_path, [
        {"direction": "out", "chat_id": HIDDEN_CHAT_ID, "text": "the real answer"},
        {"direction": "out", "chat_id": HIDDEN_CHAT_ID, "text": "⚠️ an untyped later notice"},
    ])
    assert _final_answer(agent_dir) == "⚠️ an untyped later notice"


def test_the_deep_review_acknowledgement_is_typed_and_cannot_be_read_as_an_answer():
    """The one plain outbound producer the class fix newly pointed at chat 0.

    `/review` sent with an explicit chat 0 is now answered in the hidden
    partition instead of being re-routed to the owner. Its acknowledgement is a
    SYSTEM row, so a run's recorded answer cannot be replaced by it.
    """
    source = (REPO / "supervisor/queue.py").read_text(encoding="utf-8")
    ack = next(line for line in source.splitlines() if "Deep self-review queued" in line)
    assert 'role="system"' in ack and "system_type=" in ack, ack.strip()


def test_the_degraded_reason_change_moves_no_benchmark_classification():
    """C3 changes the VALUE of an existing key, never a bucket.

    Two benchmark readers consume ``degraded_reason``: the shared result index,
    and the Terminal-Bench installed agent, which writes it into
    ``ouroboros-run-summary.json`` AND prints that object to stdout on every
    trial. Both must keep deciding ``infra_failed``/``truncated`` from their own
    inputs, so a newly specific reason cannot silently re-bucket a run.
    """
    index = (REPO / "devtools/benchmarks/common/result_index.py").read_text(encoding="utf-8")
    harbor = (REPO / "devtools/benchmarks/terminal_bench/harbor_installed_agent.py").read_text(encoding="utf-8")

    assert '"degraded_reason"' in index and '"degraded_reason"' in harbor
    # The classification inputs are independent of the reason string.
    assert 'infra_failed' in harbor and 'truncated' in harbor
    for line in harbor.splitlines():
        if ("infra_failed" in line or "truncated" in line) and "=" in line:
            assert "degraded_reason" not in line, (
                "a benchmark bucket must not be derived from the degraded reason: " + line.strip()
            )


def test_the_two_toasts_this_sprint_touched_stay_out_of_a_headless_progress_log():
    """ATIF republishes progress rows as the agent's own narration.

    The scheduled-subagent and subagent-rejection notices are host lines, not
    the model's. They were dropped for chat 0 before this sprint and stay
    dropped, now for a stated reason rather than by accident: the durable record
    keeps the real address, the live toast needs a reader.

    Scope, stated so this is not read as a general guarantee: OTHER host toasts
    (the finalization-grace warning, for one) already reached the progress log
    before this sprint and still do. They carry a HOST_NARRATION marker the
    trajectory builder does not yet consult, which is a pre-existing gap in the
    harness rather than something this change introduced — filed separately.
    """
    # v7 split supervisor/events.py: _handle_schedule_task moved to
    # events_schedule_task.py and the subagent-admission notice to
    # events_subagent_admission.py. Same two lines, new owning leaves.
    schedule = (REPO / "supervisor/events_schedule_task.py").read_text(encoding="utf-8")
    assert "if _notice_chat is not None and _notice_chat != HIDDEN_CHAT_ID:" in schedule
    admission = (REPO / "supervisor/events_subagent_admission.py").read_text(encoding="utf-8")
    assert "if chat_id is None or chat_id == HIDDEN_CHAT_ID:" in admission

    atif = (REPO / "devtools/benchmarks/terminal_bench/atif.py").read_text(encoding="utf-8")
    assert 'progress.jsonl' in atif and 'narration_rows' in atif, (
        "if ATIF stops reading progress.jsonl this guard can be revisited"
    )
