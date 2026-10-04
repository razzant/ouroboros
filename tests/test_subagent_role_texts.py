"""What a delegated child and its parent are told about the child's memory rights.

Both delegated-child sets carry knowledge_write and memory_mark (the host signs
both with the child's focus) and chronicle_write for the child's own drafts of
pages and parts, which the integrating mind accepts or rejects, while identity and scratchpad stay with the
parent. The child's and the nanny's ``## Working sources`` line is the child
role text plus one sentence about those drafts; the assignment phrases of both
branches and the schedule_subagent description say the same and no longer claim
the child "cannot write cognitive memory" or that chronicle pages stay with the
parent. Each check has both sides: the new words are present, the false ones are
gone, and a root never gets the child's line.
"""
from __future__ import annotations

import pathlib

from ouroboros.memory import Memory

ROLE_TEXT = (
    "Work from this assignment first — it is written to be enough; read memory or sources only to fill a "
    "gap it leaves, and name what you read in your report."
)
DRAFT_RIGHT = ("You may also publish chronicle pages and parts as drafts in your own name; the integrating mind "
               "accepts or rejects them.")
MEMORY_SENTENCE = "Knowledge notes and memory marks may be written in your own name"
DRAFT_SENTENCE = ("Chronicle pages and parts you may publish only as drafts in your own name; the integrating mind "
                  "accepts or rejects them.")
FALSE_CLAIMS = ("write cognitive memory", "data/memory state", "chronicle pages stay with the parent",
                "scratchpad or chronicle pages", "(it cannot write local state")


def _env(tmp_path: pathlib.Path):
    repo_dir, drive_root = tmp_path / "repo", tmp_path / "drive"
    for path in (repo_dir / "prompts", repo_dir / "docs", drive_root / "memory" / "knowledge",
                 drive_root / "logs", drive_root / "state"):
        path.mkdir(parents=True, exist_ok=True)
    files = {
        repo_dir / "prompts" / "SYSTEM.md": "System prompt",
        repo_dir / "BIBLE.md": "Bible",
        repo_dir / "VERSION": "1.2.3",
        repo_dir / "pyproject.toml": 'version = "1.2.3"',
        repo_dir / "README.md": "README",
        repo_dir / "docs" / "ARCHITECTURE.md": "# Ouroboros v1.2.3",
        repo_dir / "docs" / "DEVELOPMENT.md": "### File Size Budgets\n| Path | Budget chars |\n|------|--------------|\n",
        drive_root / "state" / "state.json": '{"spent_usd": 0, "budget_drift_alert": false}',
        drive_root / "memory" / "identity.md": "identity",
        drive_root / "memory" / "scratchpad.md": "scratchpad",
    }
    for name in ("chat", "supervisor", "task_reflections", "tools", "events", "progress"):
        files[drive_root / "logs" / f"{name}.jsonl"] = ""
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")

    class FakeEnv:
        def drive_path(self, rel):
            return drive_root / rel

        def repo_path(self, rel):
            return repo_dir / rel

        @property
        def repo_dir(self):
            return repo_dir

        @property
        def drive_root(self):
            return drive_root

    return repo_dir, drive_root, FakeEnv()


def _system_text(tmp_path: pathlib.Path, task: dict) -> str:
    from ouroboros.context import build_llm_messages

    repo_dir, drive_root, env = _env(tmp_path)
    messages, _cap = build_llm_messages(env=env, memory=Memory(drive_root=drive_root, repo_dir=repo_dir), task=task)
    return "\n".join(block.get("text", "") for block in messages[0]["content"] if isinstance(block, dict))


def test_child_working_sources_line_is_the_role_text_and_a_root_has_none(tmp_path):
    child_task = {"id": "kid00001", "type": "task", "text": "work", "delegation_role": "subagent",
                  "parent_task_id": "root0001", "root_task_id": "root0001"}
    nanny_task = {**child_task, "id": "nan00001", "configured_subagent": {"route": {"kind": "agent_session"}}}
    for name, task in (("child", child_task), ("nanny", nanny_task)):
        text = _system_text(tmp_path / name, task)
        section = text[text.index("## Working sources"):].split("\n## ", 1)[0]
        # The pinned lowercase phrase (tests/test_recent_sections_per_task.py) stays inside the new line,
        # and the draft right is its one added sentence, at the end.
        assert ROLE_TEXT in section and "your own recent process" in section
        assert section.rstrip().endswith(ROLE_TEXT + " " + DRAFT_RIGHT)
        for stale in ("not preloaded", "Your parent's selected discussion", "shared biography"):
            assert stale not in section
    root = _system_text(tmp_path / "root", {"id": "root0001", "type": "task", "text": "work"})
    assert "## Working sources" not in root and ROLE_TEXT not in root and DRAFT_RIGHT not in root


def test_both_assignment_branches_grant_signed_memory_and_drop_the_old_bans():
    from supervisor.events_subagent_admission import _compose_subagent_text

    acting = _compose_subagent_text("obj", role="builder", expected_output="out", constraints="", context="",
                                    task_constraint={"mode": "acting_subagent", "surface": "self_worktree"})
    readonly = _compose_subagent_text("obj", role="researcher", expected_output="out", constraints="", context="")
    for text in (acting, readonly):
        assert MEMORY_SENTENCE in text and "the host signs them with your focus" in text
        assert DRAFT_SENTENCE in text
        assert "Your result goes to your parent as a report." in text
        for claim in FALSE_CLAIMS:
            assert claim not in text
    # What stays forbidden is still said in each branch.
    assert "Do NOT commit" in acting and "or write identity or scratchpad." in acting
    assert "Do not write local repo/data state" in readonly and "tree_note" in readonly


def test_schedule_subagent_tells_the_parent_children_write_signed_memory_and_drafts_but_not_identity():
    from ouroboros.tools.control import get_tools

    description = next(entry.schema for entry in get_tools() if entry.name == "schedule_subagent")["description"]
    # Addressed to the parent, so "their own name" where the child's assignment says "your own name".
    assert "children may write knowledge notes and memory marks in their own name" in description
    assert ("publish chronicle pages and parts only as drafts, which the integrating mind accepts or rejects "
            "(chronicle_write kind=decision)") in description
    assert "identity and scratchpad stay with the parent" in description
    assert "apart from knowledge notes, memory marks and chronicle drafts in its own name" in description
    assert "Mutative children cannot commit or enable tools" in description
    for claim in FALSE_CLAIMS:
        assert claim not in description
