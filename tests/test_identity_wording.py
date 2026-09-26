from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_prompts_do_not_infer_current_human_from_authors():
    system = (REPO_ROOT / "prompts" / "SYSTEM.md").read_text(encoding="utf-8")
    consciousness = (REPO_ROOT / "prompts" / "CONSCIOUSNESS.md").read_text(encoding="utf-8")
    memory = (REPO_ROOT / "ouroboros" / "memory.py").read_text(encoding="utf-8")

    assert "my human" in system
    assert "I do not know their name" in system
    assert "README, BIBLE, git history, or author" in system
    assert "your human" in consciousness and "the user" not in consciousness
    assert "I do not yet know my human's name or profile" in memory
    assert "Anton" not in system
    assert "Razzhigaev" not in system


def test_live_task_message_marker_uses_my_human_wording():
    system = (REPO_ROOT / "prompts" / "SYSTEM.md").read_text(encoding="utf-8")
    # v7 L-B split: sweep the whole loop family (facade + leaves).
    loop_dir = REPO_ROOT / "ouroboros"
    loop = "".join(
        path.read_text(encoding="utf-8")
        for path in [loop_dir / "loop.py", *sorted(loop_dir.glob("loop_*.py"))]
    )
    tools = (REPO_ROOT / "ouroboros" / "tools" / "core.py").read_text(encoding="utf-8")

    assert "[Message from my human]" in system
    # The drained mailbox text (plus its optional surface note) must still go
    # through the owner-marking wrapper before injection.
    assert "_owner_marked_content(noted_owner_text(owner_ctx, entry, " in loop
    # Addressed task-tree messages are peer/ancestor/peer-root communication,
    # not owner dialogue, and must never borrow the owner's priority marker.
    # Ask the render ladder itself: every provenance it can frame — including
    # the independent-root prefix a task may now be addressed by — names a
    # TASK, so none of them can be mistaken for my human.
    from ouroboros.owner_mailbox import PROVENANCE_INDEPENDENT_TASK, deliver_task_message

    rendered = {}
    for provenance in (
        "ancestor_task", "descendant_task", "peer_via_ancestor", "system",
        PROVENANCE_INDEPENDENT_TASK, "",
    ):
        lines: list[str] = []
        deliver_task_message(
            {
                "provenance": provenance, "source_task_id": "t-source",
                "relayed_from_task_id": "t-relayed", "text": "body",
            },
            "t-recipient", None, lines.append,
        )
        rendered[provenance] = lines[0]
    assert all("human" not in text for text in rendered.values()), rendered
    assert rendered[PROVENANCE_INDEPENDENT_TASK].startswith(
        "[Message from independent task t-source]")
    assert rendered["ancestor_task"].startswith("[Message from ancestor task t-source]")
    assert rendered[""] == rendered["ancestor_task"], "unknown provenance keeps the tree fallback"
    assert "[Message from my human]" not in tools
    assert "[Owner message during task]" not in system
    assert "[Owner message during task]" not in loop


def test_system_prompt_carries_outcome_honesty_and_capability_acquisition():
    """v6.29.0 doctrine pins: the outcome-honesty doctrine and the capability-
    acquisition boldness clause must stay in SYSTEM.md. The doctrine now says
    the three endings in WORDS: the ledger identifiers belong to the reviewers'
    JSON contract, and a prompt that spells them is a prompt the model parrots
    back to its human. v6.60.0: the FINAL ANSWER marker doctrine deliberately
    MOVED to the per-task contract (answer_protocol) — the default prompt must
    NOT carry it (ordinary tasks never see the marker)."""
    import pathlib

    text = (pathlib.Path(__file__).parent.parent / "prompts" / "SYSTEM.md").read_text(encoding="utf-8")
    assert "### Outcome honesty" in text
    # Whitespace-normalized: the doctrine sentence is line-wrapped in the file.
    normalized = " ".join(text.split())
    assert "I do not abandon an owed answer" in normalized
    assert "Presence observation may deliberately end silently without leaving accepted work unfinished" in normalized
    assert "blocked_with_evidence" not in text
    assert "best_effort" not in text
    # Whitespace-normalized: a line-wrapped "FINAL\nANSWER" must not slip past.
    assert "FINAL ANSWER" not in " ".join(text.split())
    assert "## Capability Acquisition" in text
    assert "NOT a \"broad fallback or shim\"" in text


def test_public_publishing_still_requires_creator_permission():
    system = (REPO_ROOT / "prompts" / "SYSTEM.md").read_text(encoding="utf-8")
    assert "Do not publish or make content publicly accessible" in system
    assert "permission from the creator" in system
    assert "My human may grant that permission only if they" in system
