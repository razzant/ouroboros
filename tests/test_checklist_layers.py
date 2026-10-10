"""The layered change-review checklist (docs/CHECKLISTS.md): the universal
`Change Review Checklist` core every reviewed change carries, and the
`Ouroboros Body Layer` appended only when the subject IS the body
(`review_body_fact.layer_for`). Pins the loader, the numbering contract, the
fingerprint and the retirement of the former single-section name."""

from __future__ import annotations

import hashlib
import pathlib
import re
import subprocess

import pytest

from ouroboros.tools.review_helpers import (
    BODY_CHECKLIST_SECTION,
    CHECKLIST_LAYERS,
    CORE_CHECKLIST_SECTION,
    checklist_fingerprint,
    load_checklist_layers,
    load_checklist_section,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
CHECKLIST = REPO / "docs" / "CHECKLISTS.md"
RETIRED_SECTION_NAME = "Repo Commit" + " Checklist"  # spelled apart so this file never carries the name
ITEM_ROW = re.compile(r"^\| (\d+) \| ([a-z_]+) \|", re.MULTILINE)


def _items(text: str) -> list[tuple[int, str]]:
    return [(int(number), key) for number, key in ITEM_ROW.findall(text)]


def test_both_layers_load_and_an_unknown_layer_is_refused():
    core = load_checklist_layers("core")
    body = load_checklist_layers("body")
    assert CHECKLIST_LAYERS == ("core", "body")
    assert core.startswith(f"## {CORE_CHECKLIST_SECTION}")
    assert f"## {BODY_CHECKLIST_SECTION}" not in core
    assert body.startswith(core)
    assert f"## {BODY_CHECKLIST_SECTION}" in body
    with pytest.raises(ValueError, match="unknown checklist layer"):
        load_checklist_layers("skill")


def test_body_layer_is_the_contiguous_file_slice_core_then_body():
    """The body text is exactly what a reader of the file sees: the core section
    immediately followed by the body section, nothing between or after."""
    text = CHECKLIST.read_text(encoding="utf-8")
    start = text.index(f"## {CORE_CHECKLIST_SECTION}")
    body_start = text.index(f"## {BODY_CHECKLIST_SECTION}")
    end = text.index("\n## ", body_start + 1)
    file_slice = text[start:end]
    assert load_checklist_layers("body") == file_slice
    # The core section ends where the body section starts, separated by the
    # one horizontal rule the layout uses between top-level sections.
    core = load_checklist_section(CORE_CHECKLIST_SECTION)
    assert text[start:body_start] == core + "\n"
    assert core.rstrip().endswith("---")


def test_item_numbering_is_continuous_across_the_layers():
    core_items = _items(load_checklist_layers("core"))
    body_items = _items(load_checklist_section(BODY_CHECKLIST_SECTION))
    assert [number for number, _ in core_items] == list(range(1, len(core_items) + 1))
    assert body_items, "the body layer carries its own numbered items"
    assert body_items[0][0] == len(core_items) + 1
    assert [number for number, _ in body_items] == list(
        range(len(core_items) + 1, len(core_items) + len(body_items) + 1))
    keys = [key for _, key in core_items + body_items]
    assert len(keys) == len(set(keys)), "every item key is unique across the layers"
    # The universal core carries no Ouroboros-only duty; the body does.
    core_keys = {key for _, key in core_items}
    body_keys = {key for _, key in body_items}
    assert "bible_compliance" not in core_keys
    assert {"bible_compliance", "development_compliance", "version_bump", "self_consistency",
            "size_cap_paydown"} <= body_keys
    assert {"secrets_check", "code_quality", "security_issues", "tests_affected"} <= core_keys


def test_the_core_layer_names_no_body_document():
    core = load_checklist_layers("core")
    for marker in ("BIBLE", "CHECKLISTS_ARCHIVE", "DEVELOPMENT.md", "ARCHITECTURE.md", "DESIGN.md"):
        assert marker not in core, marker


def test_the_retired_section_name_is_gone_from_the_executing_code_and_docs():
    hits: list[str] = []
    for tree in ("ouroboros", "prompts", "docs"):
        for path in sorted((REPO / tree).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".md", ".txt", ".toml", ".json"}:
                if RETIRED_SECTION_NAME in path.read_text(encoding="utf-8", errors="replace"):
                    hits.append(path.relative_to(REPO).as_posix())
    assert hits == [], hits


def test_fingerprint_names_the_rules_that_ran():
    core = checklist_fingerprint("core")
    body = checklist_fingerprint("body")
    assert set(core) == {"checklist_hash", "rules_source"}
    assert core["rules_source"]["path"] == "docs/CHECKLISTS.md"
    assert core["checklist_hash"] == hashlib.sha256(
        load_checklist_layers("core").encode("utf-8")).hexdigest()
    assert body["checklist_hash"] == hashlib.sha256(
        load_checklist_layers("body").encode("utf-8")).hexdigest()
    assert core["checklist_hash"] != body["checklist_hash"]
    # Both layers come from the same file, so they name the same source blob.
    assert core["rules_source"]["sha"] == body["rules_source"]["sha"]
    hashed = subprocess.run(["git", "hash-object", str(CHECKLIST)], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout.strip()
    assert core["rules_source"]["sha"] == hashed


def test_fingerprint_changes_when_the_rules_change(tmp_path: pathlib.Path):
    copy = tmp_path / "CHECKLISTS.md"
    copy.write_bytes(CHECKLIST.read_bytes())
    before = checklist_fingerprint("body", copy)
    assert before == checklist_fingerprint("body", CHECKLIST)
    text = copy.read_text(encoding="utf-8")
    assert "| 1 | secrets_check |" in text
    copy.write_text(text.replace("| 1 | secrets_check |", "| 1 | secrets_check | (edited)", 1),
                    encoding="utf-8")
    after = checklist_fingerprint("body", copy)
    assert after["checklist_hash"] != before["checklist_hash"]
    assert after["rules_source"]["sha"] != before["rules_source"]["sha"]
    assert after["rules_source"]["path"] == "docs/CHECKLISTS.md"
