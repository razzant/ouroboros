"""The checklist layer is the ONE external switch of governance delivery.

For the `core` layer (the subject is not the Ouroboros body) the final
reviewer prompt of every triad delivery — api packet, agent session and
native retrieving episode — carries no BIBLE, no standing-disclosure archive,
no handbook/architecture book material and no body-only rules, and it carries
the subject's own document navigation plus its required-source manifest. For
the `body` layer every delivery is byte-identical to the default (today's)
behaviour. The governance root is always the installed system repository;
the subject root is the reviewed repository.
"""

from __future__ import annotations

import pathlib

import pytest

from ouroboros.review_records import ReviewSlot
from ouroboros.tools import review
from ouroboros.tools.governance_context import (
    NOT_APPLICABLE_DISPOSITION,
    SHARED_CHECKLIST_SECTION,
    governance_context,
)
from ouroboros.tools.review_helpers import (
    CRITICAL_FINDING_CALIBRATION,
    REVIEW_PREAMBLE,
    REVIEW_PREAMBLE_CORE,
    anti_pattern_lock_guard,
    load_checklist_layers,
)
from ouroboros.triad_review import REVIEW_JSON_ARRAY_CONTRACT
from ouroboros.tools.review_multi_model import TRIAD_USER_TURN, triad_api_messages
from ouroboros.tools.review_subject import build_triad_session_task

# Body-only material a core-layer prompt must never carry. Each marker is a
# sentence written ONLY into the fake governance root below, or the name of a
# document that belongs to the body.
BIBLE_TEXT = "P1 Continuity governs the reviewer."
ARCHIVE_TEXT = "A standing disclosure of the body."
RELEASE_SYNC_TEXT = "Keep the release-sync carriers byte-identical."
HERMETIC_TEXT = "Run the hermetic pytest lane before tagging."
ARCH_TEXT = "The loop lives in ouroboros/loop.py."
BODY_ITEM = "bible_compliance"
SUBJECT_README = "This project ships a CLI; see docs/GUIDE.md for its promises."
SUBJECT_GUIDE = "The subject's own guide: exit codes are documented here."
TOUCHED = ["src/cli.py"]


@pytest.fixture()
def governance_root(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "system_repo"
    (root / "docs" / "development").mkdir(parents=True)
    (root / "docs" / "architecture").mkdir(parents=True)
    (root / "BIBLE.md").write_text(f"# Constitution\n\n{BIBLE_TEXT}\n", encoding="utf-8", newline="\n")
    (root / "docs" / "CHECKLISTS_ARCHIVE.md").write_text(
        f"# Archive\n\n{ARCHIVE_TEXT}\n", encoding="utf-8", newline="\n")
    (root / "docs" / "CHECKLISTS.md").write_text(
        "# Checklists\n\n## Change Review Checklist\n\n| 1 | secrets_check | x | critical |\n\n---\n\n"
        f"## Ouroboros Body Layer\n\n| 10 | {BODY_ITEM} | x | critical |\n\n"
        f"## {SHARED_CHECKLIST_SECTION}\n\nThe reviewed tree's copy of the shared rule.\n",
        encoding="utf-8", newline="\n")
    (root / "docs" / "DESIGN.md").write_text("# Design\n\nThe design system.\n", encoding="utf-8", newline="\n")
    (root / "docs" / "development" / "05-review-and-commit-protocol.md").write_text(
        f"# Protocol\n\nIntro.\n\n## Release\n\n{RELEASE_SYNC_TEXT} {HERMETIC_TEXT} Touches src/cli.py.\n",
        encoding="utf-8", newline="\n")
    (root / "docs" / "DEVELOPMENT.md").write_text(
        "# Development\n\nThe handbook.\n\n## Chapters\n\n"
        "- [05-review-and-commit-protocol.md](development/05-review-and-commit-protocol.md)\n",
        encoding="utf-8", newline="\n")
    (root / "docs" / "architecture" / "01-core.md").write_text(
        f"# 01-core.md\n\nIntro.\n\n## Core loop\n\n{ARCH_TEXT} Also src/cli.py.\n",
        encoding="utf-8", newline="\n")
    (root / "docs" / "ARCHITECTURE.md").write_text(
        "# Architecture\n\nThe map.\n\n## Chapters\n\n- [01-core.md](architecture/01-core.md)\n",
        encoding="utf-8", newline="\n")
    return root


@pytest.fixture()
def subject_root(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "subject_repo"
    (root / "docs").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "README.md").write_text(f"# Subject\n\n{SUBJECT_README}\n", encoding="utf-8", newline="\n")
    (root / "docs" / "GUIDE.md").write_text(f"# Guide\n\n## Exit codes\n\n{SUBJECT_GUIDE}\n",
                                           encoding="utf-8", newline="\n")
    (root / "src" / "cli.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    return root


def _ctx(governance_root: pathlib.Path):
    return type("_Ctx", (), {"repo_dir": str(governance_root)})()


def _system_text(messages: list) -> str:
    content = messages[0]["content"]
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content)


def _packet_prompt(governance_root, *, layer: str, subject_root=None, explicit: bool = True) -> str:
    """The api packet exactly as `_prepare_unified_review` assembles it: the
    stable template over the layered checklist and the shared governance
    builder, the governance tail opening the dynamic half, then the
    constitutional head `triad_api_messages` prepends for a body row."""
    layer_kwargs = {"layer": layer, "subject_root": subject_root} if explicit else {}
    checklist = load_checklist_layers(layer, governance_root / "docs" / "CHECKLISTS.md")
    governance = review._triad_governance_context(
        _ctx(governance_root), list(TOUCHED), checklist, ["openai/packet"],
        [ReviewSlot(slot_id="triad_slot_1", model="openai/packet")], **layer_kwargs)
    stable = review._REVIEW_PROMPT_TEMPLATE_STABLE.format(
        preamble=REVIEW_PREAMBLE_CORE if layer == "core" else REVIEW_PREAMBLE,
        critical_calibration=CRITICAL_FINDING_CALIBRATION,
        json_contract=REVIEW_JSON_ARRAY_CONTRACT,
        anti_pattern_lock_guard=anti_pattern_lock_guard(layer),
        checklist_section=checklist,
    ) + (f"\n{governance.stable_inline}\n" if governance.stable_inline.strip() else "")
    tail = "\n\n".join(part for part in (governance.selected_inline, governance.navigation) if part.strip())
    dynamic = (f"{tail}\n\n" if tail else "") + review._REVIEW_PROMPT_TEMPLATE_DYNAMIC.format(
        goal_section="## Goal\n\ngoal", scope_section="## Scope\n\nscope",
        current_files_section="(files)", rebuttal_section="", review_history_section="",
        diff_text="(diff)", changed_files="src/cli.py", task_evidence_section="")
    prompt = stable + "\n" + dynamic
    messages, _bible = triad_api_messages(prompt, len(stable) + 1, TRIAD_USER_TURN,
                                          **({"layer": layer} if explicit else {}))
    return _system_text(messages)


def _session_prompt(governance_root, *, layer: str, subject_root=None, explicit: bool = True) -> str:
    layer_kwargs = {"layer": layer, "subject_root": subject_root} if explicit else {}
    return build_triad_session_task(
        goal_section="## Goal\n\ngoal", scope_section="## Scope\n\nscope",
        checklist_section=load_checklist_layers(layer, governance_root / "docs" / "CHECKLISTS.md"),
        rebuttal_section="", review_history_section="",
        governance_repo_dir=governance_root, **layer_kwargs)


def _native_prompt(governance_root, *, layer: str, subject_root=None, explicit: bool = True) -> str:
    """A native retrieving episode: the shared governance builder in retrieving
    delivery, handed to the session-task builder as `build_retrieving_brief` does."""
    layer_kwargs = {"layer": layer, "subject_root": subject_root} if explicit else {}
    checklist = load_checklist_layers(layer, governance_root / "docs" / "CHECKLISTS.md")
    governance = review._triad_governance_context(
        _ctx(governance_root), list(TOUCHED), checklist, ["openai/native"],
        [ReviewSlot(slot_id="triad_1", model="openai/native")], delivery="retrieving", **layer_kwargs)
    return build_triad_session_task(
        goal_section="## Goal\n\ngoal", scope_section="## Scope\n\nscope",
        checklist_section=checklist, rebuttal_section="", review_history_section="",
        governance_repo_dir=governance_root, governance=governance,
        **({"layer": layer} if explicit else {}))


DELIVERIES = {"packet": _packet_prompt, "session": _session_prompt, "native": _native_prompt}


@pytest.fixture(autouse=True)
def _wide_window(monkeypatch):
    monkeypatch.setattr(review, "reviewer_context_window", lambda *_a, **_k: 200_000)


@pytest.mark.parametrize("delivery", sorted(DELIVERIES))
def test_core_layer_prompt_carries_no_body_governance_in_any_delivery(delivery, governance_root, subject_root):
    prompt = DELIVERIES[delivery](governance_root, layer="core", subject_root=subject_root)

    for marker in (BIBLE_TEXT, ARCHIVE_TEXT, RELEASE_SYNC_TEXT, HERMETIC_TEXT, ARCH_TEXT, BODY_ITEM,
                   "BIBLE", "CHECKLISTS_ARCHIVE", "DEVELOPMENT.md", "ARCHITECTURE.md", "DESIGN.md",
                   "Constitution", SHARED_CHECKLIST_SECTION):
        assert marker not in prompt, (delivery, marker)
    assert REVIEW_PREAMBLE_CORE.strip() in prompt
    assert REVIEW_PREAMBLE.strip() not in prompt
    assert "## Change Review Checklist" in prompt
    assert "## Ouroboros Body Layer" not in prompt


@pytest.mark.parametrize("delivery", sorted(DELIVERIES))
def test_core_layer_prompt_carries_the_subject_documents_and_source_manifest(delivery, governance_root, subject_root):
    prompt = DELIVERIES[delivery](governance_root, layer="core", subject_root=subject_root)

    assert "## Governance navigation (core layer)" in prompt
    assert "### Subject documents" in prompt
    assert "README.md (navigation map)" in prompt and "docs/GUIDE.md (navigation map)" in prompt
    assert "Exit codes — lines 3-5" in prompt, "the subject's own heading map is the navigation"
    assert SUBJECT_GUIDE not in prompt, "a map, never the subject's text"
    assert "REQUIRED SOURCES" in prompt
    assert "not the Ouroboros body" in prompt
    if delivery == "packet":
        assert "were NOT delivered" in prompt
    else:
        assert "read any range you need with your own tools" in prompt


@pytest.mark.parametrize("delivery", sorted(DELIVERIES))
def test_body_layer_is_byte_identical_to_the_default_delivery(delivery, governance_root):
    explicit = DELIVERIES[delivery](governance_root, layer="body")
    default = DELIVERIES[delivery](governance_root, layer="body", explicit=False)

    assert explicit == default
    for marker in (BODY_ITEM, "## Ouroboros Body Layer", "05-review-and-commit-protocol.md",
                   SHARED_CHECKLIST_SECTION):
        assert marker in explicit, (delivery, marker)
    if delivery != "session":
        # Sized deliveries inline the tier-2 chapter naming the touched file;
        # the bare session fallback (no usable window) names it in navigation.
        assert RELEASE_SYNC_TEXT in explicit and HERMETIC_TEXT in explicit
    if delivery == "packet":
        # BIBLE rides the constitutional head (the install's own BIBLE.md, or
        # the loud "could not be loaded" line), never the governance root's copy.
        assert "BIBLE.md" in explicit and "## REVIEW INSTRUCTIONS" in explicit
    else:
        assert BIBLE_TEXT in explicit
    assert REVIEW_PREAMBLE.strip() in explicit
    assert REVIEW_PREAMBLE_CORE.strip() not in explicit


def test_core_layer_manifest_names_what_was_withheld(governance_root, subject_root):
    context = governance_context(
        governance_root, surface="triad", touched_paths=TOUCHED, usable_window_tokens=200_000,
        delivery="retrieving", checklist_section_text="## Change Review Checklist\n\n| 1 | x |",
        layer="core", subject_root=subject_root)

    assert context.layer == "core"
    by_path = {row["path"]: row for row in context.manifest}
    for path in ("BIBLE.md", "docs/CHECKLISTS_ARCHIVE.md", "docs/DEVELOPMENT.md", "docs/DESIGN.md",
                 "docs/ARCHITECTURE.md", f"docs/CHECKLISTS.md#{SHARED_CHECKLIST_SECTION}"):
        assert by_path[path]["disposition"] == NOT_APPLICABLE_DISPOSITION, path
    assert by_path["docs/CHECKLISTS.md"]["disposition"] == "inline"
    assert by_path["subject:README.md"]["disposition"] == "navigation"
    assert by_path["subject:docs/GUIDE.md"]["tier"] == 3
    assert context.stable_inline == "" and context.selected_inline == ""
    assert len({row["path"] for row in context.manifest}) == len(context.manifest)


def test_core_layer_without_a_subject_root_discloses_the_gap(governance_root):
    context = governance_context(
        governance_root, surface="triad", touched_paths=TOUCHED, delivery="packet",
        checklist_section_text="## Change Review Checklist\n\n| 1 | x |", layer="core")

    assert "No subject root was supplied" in context.navigation
    assert "REQUIRED SOURCES" in context.navigation
    assert not any(row["path"].startswith("subject:") for row in context.manifest)


def test_an_unknown_layer_is_refused(governance_root):
    with pytest.raises(ValueError, match="layer"):
        governance_context(governance_root, surface="triad", layer="skill")


def test_triad_api_messages_core_head_carries_no_constitution():
    core_messages, core_bible = triad_api_messages("STABLE\nDYNAMIC", 7, TRIAD_USER_TURN, layer="core")
    body_messages, body_bible = triad_api_messages("STABLE\nDYNAMIC", 7, TRIAD_USER_TURN)

    core_text = _system_text(core_messages)
    assert core_bible == ""
    assert core_text.startswith("## REVIEW INSTRUCTIONS\n\nSTABLE\n")
    assert "BIBLE" not in core_text and "Constitution" not in core_text
    assert core_messages[1] == {"role": "user", "content": TRIAD_USER_TURN}
    # The body head is today's: the constitutional preamble, then the prompt.
    body_text = _system_text(body_messages)
    assert body_text.endswith("## REVIEW INSTRUCTIONS\n\nSTABLE\nDYNAMIC")
    assert (body_bible == "") == ("(BIBLE.md could not be loaded)" in body_text)
    assert body_messages == triad_api_messages("STABLE\nDYNAMIC", 7, TRIAD_USER_TURN, layer="body")[0]
