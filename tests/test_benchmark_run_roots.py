"""Run-root containment across Windows extended path namespaces."""

import pathlib

import pytest

from devtools.benchmarks.common.run_roots import _comparison_path, safe_join_under


@pytest.mark.parametrize(("base", "child", "inside"), [
    (r"C:\runs\root", r"\\?\C:\runs\root\task", True),
    (r"C:\runs\root", r"\\?\C:\runs\root-other\task", False),
    (r"C:\runs\root", r"\\?\D:\runs\root\task", False),
    (r"\\server\share\root", r"\\?\UNC\server\other\root\task", False),
])
def test_windows_resolved_namespace_containment(base, child, inside):
    assert _comparison_path(pathlib.PureWindowsPath(child)).is_relative_to(
        _comparison_path(pathlib.PureWindowsPath(base))
    ) is inside


def test_safe_join_under_keeps_long_path_usable(tmp_path):
    root = tmp_path / "run"
    parts = ["nested-" + "x" * 70] * 4 + ["result.json"]

    resolved = safe_join_under(root, *parts)
    resolved.parent.mkdir(parents=True)
    resolved.write_text("{}", encoding="utf-8")

    assert resolved.parts[-len(parts):] == tuple(parts)
    assert resolved.read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize("parts", [("..", "escape"), ("..", "run-other", "task")])
def test_safe_join_under_still_rejects_escape(tmp_path, parts):
    with pytest.raises(ValueError, match="escapes run root"):
        safe_join_under(tmp_path / "run", *parts)
