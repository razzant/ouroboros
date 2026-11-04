"""Regenerate the small source fixture of #2766 from its original Git objects.

Only the named word producers are retained; the test needs no historical Git
objects (including in a shallow CI checkout). Run from the repository root.
"""

import ast
import json
import pathlib
import subprocess

from tree_sitter_language_pack import get_parser

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS = {
    "ouroboros/gateway/owner_settings.py": {"_ROW_PHRASES", "settings_change_text"},
    "ouroboros/merge_receipts.py": {"_GAP_WORDS", "coverage_words", "coverage_codes", "card_row_text"},
    "web/modules/model_roles.js": {"MAIN_FOLLOW_NOTE"},
    "web/modules/question_presentation.js": {"STATUS", "PAUSED_STATUS", "questionPresentation"},
}
COMMITS = ("e385fdf40", "ba091a394", "7f76503bb")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")


def excerpt(path, source):
    wanted = TARGETS[path]
    if path.endswith(".py"):
        nodes = ast.parse(source).body
        selected = [node for node in nodes if getattr(node, "name", "") in wanted
                    or isinstance(node, ast.Assign) and any(ast.unparse(t) in wanted for t in node.targets)]
        lines = source.splitlines(keepends=True)
        return "\n".join("".join(lines[node.lineno - 1:node.end_lineno]) for node in selected)
    raw = source.encode("utf-8")
    tree = get_parser("javascript").parse(raw)
    selected = []
    for top in tree.root_node.named_children:
        node = top.named_children[0] if top.type == "export_statement" else top
        declarations = node.named_children if node.type == "lexical_declaration" else [node]
        if any(raw[name.start_byte:name.end_byte].decode() in wanted for child in declarations
               if (name := child.child_by_field_name("name"))):
            selected.append(raw[top.start_byte:top.end_byte].decode())
    return "\n\n".join(selected) + "\n"


def main():
    rows = []
    for commit in COMMITS:
        touched = set(git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines())
        rows.append({"commit": commit, "files": [
            {"path": path, "before": excerpt(path, git("show", f"{commit}^:{path}")),
             "after": excerpt(path, git("show", f"{commit}:{path}"))}
            for path in TARGETS if path in touched]})
    (ROOT / "tests/fixtures/review_words_2766.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
