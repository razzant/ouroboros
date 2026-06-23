"""Tests for the managed-update per-path merge policy (P2)."""

from supervisor.update_merge_policy import (
    classify_conflicts,
    is_auto_reconcile_doc,
    is_hot_code,
    is_protected_doc,
)


def test_clean_when_no_conflicts():
    res = classify_conflicts([])
    assert res["kind"] == "clean"
    assert res["doc_conflict_paths"] == []
    assert res["code_conflict_paths"] == []


def test_doc_only_conflict_is_reconcile():
    res = classify_conflicts(["README.md", "docs/ARCHITECTURE.md", "CHANGELOG.md"])
    assert res["kind"] == "doc_reconcile"
    assert set(res["doc_conflict_paths"]) == {"README.md", "docs/ARCHITECTURE.md", "CHANGELOG.md"}
    assert res["code_conflict_paths"] == []
    assert res["protected_conflict_paths"] == []


def test_protected_docs_never_reconcile():
    for doc in ("BIBLE.md", "docs/CHECKLISTS.md", "prompts/SAFETY.md"):
        assert is_protected_doc(doc), doc
        assert not is_auto_reconcile_doc(doc), doc
        res = classify_conflicts([doc])
        assert res["kind"] == "conflicting", doc
        assert res["protected_conflict_paths"] == [doc], doc


def test_code_conflict_is_conflicting_even_with_docs():
    res = classify_conflicts(["README.md", "ouroboros/loop.py"])
    assert res["kind"] == "conflicting"
    assert res["code_conflict_paths"] == ["ouroboros/loop.py"]
    assert res["doc_conflict_paths"] == ["README.md"]
    assert "ouroboros/loop.py" in res["hot_code_paths"]


def test_hot_code_is_label_only():
    # hot code is a sharper label inside code conflicts, not its own kind.
    assert is_hot_code("supervisor/queue.py")
    assert is_hot_code("ouroboros/config.py")
    assert not is_hot_code("ouroboros/some_new_module.py")
    res = classify_conflicts(["ouroboros/some_new_module.py"])
    assert res["kind"] == "conflicting"
    assert res["hot_code_paths"] == []


def test_path_normalization():
    # leading ./ and backslashes normalize to repo-relative POSIX.
    assert is_protected_doc("./BIBLE.md")
    assert is_auto_reconcile_doc("docs\\guide.md")
    assert not is_auto_reconcile_doc("docs/notes.txt")  # non-markdown under docs/ is code-ish
