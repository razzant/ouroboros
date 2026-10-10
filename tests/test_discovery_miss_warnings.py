"""#1074: an admitted, absent search path or local image is a typed discovery miss.

``search_code`` and the shared local-image loader (``view_image``, ``vlm_query`` and
the host's auto-attach) publish ``ok/LEGACY_WARNING`` for a well-formed request to an
absent path only after every admission guard. A denial, an escape, an invalid query
and a bad image keep their error meaning, and a miss creates no image, copy or VLM call.
Relative image paths keep their process-working-directory meaning.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_vision import _real_png_bytes, _vision_registry


def _protected(paths, deny):
    from ouroboros.contracts.task_contract import build_task_contract

    return build_task_contract({"resource_policy": {"protected_artifacts": [
        {"id": "reference", "role": "black_box_reference", "paths": [str(p) for p in paths],
         "allow": ["execute"], "deny": list(deny)}]}})


def _symlink_or_skip(link, target, *, directory=False):
    """Symlink escapes are optional evidence: a host that cannot create links (Windows
    without the privilege) skips only them, never the ordinary refusals beside them."""
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def _assert_refused(result, marker, label, miss):
    assert result.status != "ok" and miss not in result.text, (label, result)
    assert marker in (result.code + result.text), (label, result)


@pytest.fixture
def search(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    repo, data, outside = tmp_path / "repo", tmp_path / "data", tmp_path / "outside"
    for folder in (repo / "src", repo / "vault", data, outside):
        folder.mkdir(parents=True)
    (repo / "src" / "app.py").write_text("needle = 1\n", encoding="utf-8")
    (outside / "leak.txt").write_text("needle outside\n", encoding="utf-8")
    contract = _protected([repo / "sealed", repo / "vault", repo / "src" / "secret.py"],
                          ["read_bytes", "static_introspection"])
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(repo_dir=repo, drive_root=data, task_contract=contract,
                                     task_metadata={"task_contract": contract}))
    return registry, repo, data, outside


@pytest.mark.parametrize("path", ["nope", "src/missing.py", "deeper/still/absent"])
def test_an_admitted_absent_search_path_is_a_typed_warning(search, path):
    registry, *_ = search

    result = registry.execute_result("search_code", {"query": "needle", "path": path})

    assert (result.status, result.code) == ("ok", "LEGACY_WARNING"), result.text
    assert result.text == f"⚠️ SEARCH_NOT_FOUND: path not found: active_workspace:{path}"
    assert result.meta["operation_outcome"] == "completed_no_effect"


def test_every_search_refusal_precedes_the_miss(search):
    registry, repo, data, outside = search
    cases = {
        "empty query": ({"query": "", "path": "nope"}, "SEARCH_ERROR: query is required"),
        "non-string query": ({"query": 5, "path": "nope"}, "SEARCH_ERROR: query must be a string"),
        "invalid regex": ({"query": "[", "regex": True, "path": "nope"}, "SEARCH_ERROR: invalid regex"),
        "outside, existing": ({"query": "needle", "path": str(outside)}, "SEARCH_ERROR"),
        "outside, missing": ({"query": "needle", "path": str(outside / "missing")}, "SEARCH_ERROR"),
        "traversal, missing": ({"query": "needle", "path": "../outside/missing"}, "SEARCH_ERROR"),
        "protected, missing": ({"query": "needle", "path": "sealed"}, "RESOURCE_POLICY_BLOCKED"),
        "inside protected, missing": ({"query": "needle", "path": "vault/absent"}, "RESOURCE_POLICY_BLOCKED"),
        "project store, missing": ({"query": "needle", "path": "projects/demo/absent", "root": "runtime_data"},
                                   "BLOCKED"),
        "unknown root": ({"query": "needle", "path": "nope", "root": "nowhere"}, ""),
    }
    for label, (args, marker) in cases.items():
        result = registry.execute_result("search_code", args)
        _assert_refused(result, marker, label, "SEARCH_NOT_FOUND")
        assert "needle outside" not in result.text, label


@pytest.mark.parametrize("target", ["", "missing-dir"], ids=["symlink escape", "dangling escape"])
def test_a_search_symlink_escape_precedes_the_miss(search, target):
    registry, repo, _data, outside = search
    _symlink_or_skip(repo / "link", outside / target, directory=True)

    result = registry.execute_result("search_code", {"query": "needle", "path": "link"})

    _assert_refused(result, "SEARCH_ERROR", target, "SEARCH_NOT_FOUND")
    assert "needle outside" not in result.text


def test_directory_search_keeps_its_per_file_rules(search):
    registry, repo, *_ = search
    (repo / "src" / "secret.py").write_text("needle = 'sealed'\n", encoding="utf-8")

    listing = registry.execute_result("search_code", {"query": "needle", "path": "src"})
    assert listing.status == "ok" and "app.py:1" in listing.text and "sealed" not in listing.text, listing.text
    single = registry.execute_result("search_code", {"query": "needle", "path": "src/secret.py"})
    assert single.status == "blocked" and "RESOURCE_POLICY_BLOCKED" in single.text, single


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs POSIX permissions enforced for this user")
def test_an_unreadable_candidate_stays_a_disclosed_partial_search(search):
    registry, repo, *_ = search
    locked = repo / "src" / "locked.py"
    locked.write_text("needle = 2\n", encoding="utf-8")
    locked.chmod(0)
    try:
        result = registry.execute_result("search_code", {"query": "needle", "path": "src"})
    finally:
        locked.chmod(0o600)
    assert result.status == "ok" and "app.py:1" in result.text and "SEARCH_NOT_FOUND" not in result.text, result


# --- the shared local-image loader ------------------------------------------------------


def _blocks(registry):
    return [block for message in registry._ctx.messages for block in (message.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "image_url"]


def test_an_admitted_absent_image_is_a_warning_with_no_image_effect(tmp_path, monkeypatch):
    import ouroboros.tools.vision as vision

    registry, uploads = _vision_registry(tmp_path, monkeypatch)
    monkeypatch.setattr(vision, "_get_llm_client", lambda: pytest.fail("a miss must not reach a VLM"))
    missing = uploads / "missing.png"

    viewed = registry.execute_result("view_image", {"path": str(missing)})
    queried = registry.execute_result("vlm_query", {"prompt": "what?", "file_path": str(missing)})

    for result in (viewed, queried):
        assert (result.status, result.code) == ("ok", "LEGACY_WARNING"), result.text
        assert result.text == f"⚠️ FILE_NOT_FOUND: image file not found: {missing} (resolved: {missing.resolve()})."
    assert viewed.meta["operation_outcome"] == "completed_no_effect"
    assert registry._ctx.messages == [] and not (tmp_path / "uploads" / "views").exists()


def test_every_image_refusal_precedes_the_miss(tmp_path, monkeypatch):
    import ouroboros.tools.vision as vision

    registry, uploads = _vision_registry(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "real.png").write_bytes(_real_png_bytes())
    (uploads / "notes.png").write_bytes(b"plain text, not an image")
    (uploads / "folder.png").mkdir()
    (uploads / "big.png").write_bytes(_real_png_bytes())
    contract = _protected([uploads / "sealed.png"], ["read_bytes"])
    registry._ctx.task_contract, registry._ctx.task_metadata = contract, {"task_contract": contract}
    # Runtime-data parity: an admitted root inside the drive's per-project store.
    project_store = tmp_path / "projects" / "demo" / "absent.png"
    monkeypatch.setattr(vision, "_allowed_file_roots", lambda *_a, **_k: [uploads, tmp_path / "projects"])
    monkeypatch.setattr(vision, "_VLM_MAX_FILE_BYTES", 32)
    cases = {
        "outside, missing": (outside / "gone.png", "TOOL_ARG_ERROR"),
        "outside, existing": (outside / "real.png", "TOOL_ARG_ERROR"),
        "protected, missing": (uploads / "sealed.png", "RESOURCE_POLICY_BLOCKED"),
        "project store, missing": (project_store, "BLOCKED"),
        "not an image": (uploads / "notes.png", "TOOL_ARG_ERROR"),
        "unreadable": (uploads / "folder.png", "TOOL_ARG_ERROR"),
        "too large": (uploads / "big.png", "TOOL_ARG_ERROR"),
    }
    for label, (path, marker) in cases.items():
        _assert_refused(registry.execute_result("view_image", {"path": str(path)}), marker, label, "FILE_NOT_FOUND")
    assert registry._ctx.messages == []


@pytest.mark.parametrize("target", ["real.png", "gone.png"], ids=["symlink escape", "dangling escape"])
def test_an_image_symlink_escape_precedes_the_miss(tmp_path, monkeypatch, target):
    registry, uploads = _vision_registry(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "real.png").write_bytes(_real_png_bytes())
    _symlink_or_skip(uploads / "link.png", outside / target)

    result = registry.execute_result("view_image", {"path": str(uploads / "link.png")})

    _assert_refused(result, "TOOL_ARG_ERROR", target, "FILE_NOT_FOUND")
    assert registry._ctx.messages == []


def test_relative_image_paths_keep_the_process_working_directory(tmp_path, monkeypatch):
    registry, uploads = _vision_registry(tmp_path, monkeypatch)
    (uploads / "chart.png").write_bytes(_real_png_bytes())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.chdir(uploads)
    assert registry.execute_result("view_image", {"path": "chart.png"}).status == "ok"
    assert len(_blocks(registry)) == 1
    miss = registry.execute_result("view_image", {"path": "missing.png"})
    assert miss.code == "LEGACY_WARNING" and f"(resolved: {(uploads / 'missing.png').resolve()})" in miss.text
    monkeypatch.chdir(elsewhere)  # outside every admitted root: a refusal, never a quiet miss
    refused = registry.execute_result("view_image", {"path": "missing.png"})
    assert (refused.status, refused.code) == ("error", "TOOL_ARG_ERROR"), refused
    assert len(_blocks(registry)) == 1


def test_host_auto_attach_of_an_absent_image_attaches_nothing(tmp_path, monkeypatch):
    from ouroboros.loop_tool_execution import _maybe_auto_attach_image

    registry, uploads = _vision_registry(tmp_path, monkeypatch)
    exec_result = {"fn_name": "ext_demo_screenshot", "is_error": False,
                   "result": json.dumps({"ok": True, "auto_attach_image": str(uploads / "missing.png")})}

    assert _maybe_auto_attach_image(exec_result, registry) == {"status": "unavailable"}
    assert registry._ctx.messages == [] and not (tmp_path / "uploads" / "views").exists()
    (uploads / "shot.png").write_bytes(_real_png_bytes())
    exec_result["result"] = json.dumps({"ok": True, "auto_attach_image": str(pathlib.Path(uploads / "shot.png"))})
    assert _maybe_auto_attach_image(exec_result, registry) == {"status": "attached"}
    assert len(_blocks(registry)) == 1
