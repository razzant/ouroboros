"""S6 R1 — upstream's disclosed residual, pinned so v7 cannot change it silently.

`docs/ARCHITECTURE.md` lists four phase-A residuals upstream deliberately did
not fix. This module pins the first one, because it is the only place where a
cancellation SETTLES without having proved the owner was told anything:

A cascade whose root has no lineage chat has nothing to send. The postcondition
still calls the delivery seam unconditionally, and the seam records a typed
`terminal_delivery_handoff` row instead — "consciously not owed", not silently
dropped. But `deliver_cascade_summary` initialises its answer to
``owed = True`` and only overwrites it when an EVENT exists, so when the
handoff row is the outcome the function reports "owed" no matter whether that
row landed. If the append also fails, the summary is neither sent nor recorded
and the cascade intent settles anyway.

Two rare failures at once, and upstream weighed it and left it. v7's job here is
to make sure a refactor cannot flip it by accident — in either direction. The
owner's decision (batch 6, answer 5=A) is: do not fix in v7; pin it and raise it
upstream.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from supervisor import terminal_delivery as td


def _rows(drive_root: pathlib.Path, name: str):
    path = drive_root / "logs" / name
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture
def chatless_root(tmp_path):
    """A settled cascade root with NO lineage chat to route a summary to."""
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result

    write_task_result(tmp_path, "root-nochat", STATUS_CANCELLED, result="stopped")
    write_task_result(tmp_path, "child-nochat", STATUS_CANCELLED, result="stopped")
    return {
        "drive": tmp_path,
        "row": {"id": "root-nochat", "chat_id": 0, "root_task_id": "root-nochat"},
        "outcomes": {"root-nochat": "cancelled", "child-nochat": "cancelled"},
    }


def test_r1_a_chatless_cascade_records_a_handoff_row_and_reports_owed(chatless_root):
    """The healthy half of the residual: no chat means no message, and the typed
    handoff row is the durable evidence that the summary was consciously not
    owed. The function reports True, which the postcondition reads as owed."""
    owed = td.deliver_cascade_summary(
        chatless_root["drive"], "root-nochat", chatless_root["row"],
        chatless_root["outcomes"],
    )

    assert owed is True
    handoffs = [
        row for row in _rows(chatless_root["drive"], "supervisor.jsonl")
        if row.get("type") == "terminal_delivery_handoff"
    ]
    assert [row.get("reason") for row in handoffs] == ["no_lineage_chat"]
    assert td.pending_deliveries(chatless_root["drive"]) == [], "nothing to send"


def test_r1_the_summary_reports_owed_even_when_the_handoff_row_is_lost(
    chatless_root, monkeypatch,
):
    """R1 as it stands: with the handoff append ALSO failing, nothing at all
    records the tree's outcome and the function still answers "owed" — so the
    cascade postcondition settles the root intent. Pinned, NOT fixed (owner
    decision batch 6, answer 5=A); raised upstream separately."""
    import ouroboros.utils as utils

    real_append = utils.append_jsonl

    def _append(path, obj):
        if pathlib.Path(path).name == "supervisor.jsonl":
            raise OSError("supervisor ledger unwritable")
        return real_append(path, obj)

    monkeypatch.setattr(utils, "append_jsonl", _append)

    owed = td.deliver_cascade_summary(
        chatless_root["drive"], "root-nochat", chatless_root["row"],
        chatless_root["outcomes"],
    )

    assert owed is True, (
        "the residual: 'owed' is the default answer, not a proven registration"
    )
    assert [
        row for row in _rows(chatless_root["drive"], "supervisor.jsonl")
        if row.get("type") == "terminal_delivery_handoff"
    ] == [], "the evidence row was lost"
    assert td.pending_deliveries(chatless_root["drive"]) == []


def test_r1_the_postcondition_settles_on_exactly_this_answer(chatless_root):
    """Why the return value matters: the cascade postcondition reads
    `deliver_cascade_summary(...) is not False` as "owed", and settles the
    root's cascade intent — the tree's watchdog replay trigger — on it. This is
    the line a v7 refactor must not quietly re-shape."""
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "supervisor" / "task_lifecycle.py"
    ).read_text(encoding="utf-8")
    assert "summary_owed = deliver_cascade_summary(" in source
    postcondition = source[source.index("summary_owed = deliver_cascade_summary("):]
    assert ") is not False" in postcondition[:400]
    settle_at = postcondition.index("settle_intent(")
    guard_at = postcondition.index("if summary_owed:")
    assert guard_at < settle_at, "the settle is gated on the summary being owed"
    # And the OTHER branch is the honest one: an unowed summary leaves the intent
    # open for the watchdog instead of settling.
    assert "could not be durably owed" in postcondition[:settle_at + 1600]


def test_r1_a_root_with_a_chat_still_owes_a_real_row(tmp_path):
    """The control: when a chat exists the answer is a REGISTERED row, not a
    default. That is what makes the chat-less lane the exception it is."""
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result

    write_task_result(tmp_path, "root-chat", STATUS_CANCELLED, result="stopped")
    row = {"id": "root-chat", "chat_id": 77, "root_task_id": "root-chat"}

    assert td.deliver_cascade_summary(
        tmp_path, "root-chat", row, {"root-chat": "cancelled"},
    ) is True
    owed = td.pending_deliveries(tmp_path)
    assert [entry["task_id"] for entry in owed] == ["root-chat"]
    assert owed[0]["chat_id"] == 77
