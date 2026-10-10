"""The packaged app registers tools from an explicit module list, not package discovery."""
from __future__ import annotations

import pathlib
import sys

import pytest


def _registry(tmp_path: pathlib.Path):
    from ouroboros.tools.registry import ToolRegistry

    return ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)


def test_a_frozen_build_registers_review_change(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ouroboros.tools.registry import ToolRegistry

    assert "review_change" in ToolRegistry._FROZEN_TOOL_MODULES
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    names = {schema["function"]["name"] for schema in _registry(tmp_path).schemas()}
    assert {"review_change", "commit_reviewed", "preflight_review"} <= names


def test_the_frozen_list_is_what_registers_it(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(ToolRegistry, "_FROZEN_TOOL_MODULES",
                        [name for name in ToolRegistry._FROZEN_TOOL_MODULES if name != "review_change"])
    names = {schema["function"]["name"] for schema in _registry(tmp_path).schemas()}
    assert "review_change" not in names and "commit_reviewed" in names


def test_package_discovery_and_the_frozen_list_agree_on_review_change(tmp_path: pathlib.Path) -> None:
    names = {schema["function"]["name"] for schema in _registry(tmp_path).schemas()}
    assert "review_change" in names
