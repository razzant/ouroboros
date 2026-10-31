"""v6.64 acceptance planning: what outlives task acceptance.

The canonical event stream of a split task (where the root admission seal and the
owner-source acknowledgement write their durable rows), budget-profile resolution, the
queue-owned seal's optional context hooks and the owner corpus. The acceptance wallet,
improvement passes, packet and panel verdict rules are retired with task acceptance.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from ouroboros import task_pacing
from ouroboros.utils import append_jsonl


def test_seal_rows_go_to_the_canonical_split_drive_stream(tmp_path):
    """The durable rows the root admission seal writes belong to the CANONICAL drive
    of a split task, not the child's own logs."""
    canonical = tmp_path / "canonical"
    child = tmp_path / "child"
    events = canonical / "logs" / "events.jsonl"
    append_jsonl(events, {"type": "supervisor_ack_unavailable", "op": "begin", "outcome": "unknown"})
    ctx = SimpleNamespace(drive_root=str(child), budget_drive_root=str(canonical))
    assert task_pacing.acceptance_timing_events_path(ctx) == events
    assert not (child / "logs" / "events.jsonl").exists()


def test_resolve_budget_profile_emits_no_deprecation_events(tmp_path):
    """ABI 7.0 (Q10=A): the alias deprecation machinery is gone - a profile
    carrying retired spellings resolves quietly, without the retired keys."""
    legacy = SimpleNamespace(
        drive_root=tmp_path,
        task_id="legacy",
        task_contract={"budget_profile": {
            "improvement_policy": "until_deadline", "stall_rounds_threshold": 2,
        }},
    )
    resolved = task_pacing.resolve_budget_profile(legacy)
    assert resolved["improvement_policy"] == "fixed"
    assert "stall_rounds_threshold" not in resolved
    assert not (tmp_path / "logs" / "events.jsonl").exists()


def test_queue_owned_acceptance_fence_uses_only_optional_ctx_hooks():
    from ouroboros.loop import _begin_task_acceptance_fence, _end_task_acceptance_fence

    calls = []

    def begin(**kwargs):
        calls.append(("begin", kwargs))
        return "fence-1"

    def end(**kwargs):
        calls.append(("end", kwargs))

    ctx = SimpleNamespace(
        task_metadata={"root_task_id": "root"},
        begin_acceptance_fence=begin,
        end_acceptance_fence=end,
    )
    opened, token = _begin_task_acceptance_fence(ctx, "root")
    assert opened and opened.status == "ok" and token == "fence-1"
    released = _end_task_acceptance_fence(ctx, outcome="revision")
    assert released and released.status == "ok"
    assert calls == [
        ("begin", {"root_task_id": "root", "task_id": "root"}),
        # A bare-token begin still yields a number: ``end`` always carries the generation the queue compares.
        ("end", {"token": "fence-1", "outcome": "revision", "expected_generation": 0}),
    ]


def test_owner_corpus_preserves_followups_without_system_messages(tmp_path):
    """The owner corpus reviewers and the post-task synthesis read: initial words and
    follow-ups, never host reminders."""
    from ouroboros.review_evidence_sections import _accept_owner_directives

    ctx = SimpleNamespace(
        task_id="root",
        root_task_id="root",
        task_metadata={"root_task_id": "root"},
        task_contract={"objective": "Implement exactly the approved plan"},
        repo_dir=tmp_path,
        messages=[
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "Initial owner requirement"},
            {"role": "user", "content": "[SYSTEM REMINDER] internal"},
            {"role": "user", "content": "[Message from my human]: choose A"},
        ],
        _owner_directives=[
            {"source": "initial_user", "content": "Initial owner requirement"},
            {"source": "direct_incoming", "content": "choose A", "msg_id": "m1"},
        ],
    )
    corpus = _accept_owner_directives(ctx, tmp_path, "root")
    assert [row["content"] for row in corpus] == ["Initial owner requirement", "choose A"]
    assert corpus[1]["msg_id"] == "m1"
    assert "SYSTEM REMINDER" not in json.dumps(corpus)
