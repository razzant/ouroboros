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
