"""Structural contracts for the semantic-no-op core tool extraction.

Carried from the v7 reference (ouroboros_v7_wip @ 9f691656) with the following
identity continuations to THIS tree's bytes:

1. The catalog schema hash is re-pinned to tip bytes (upstream drifted the
   read/list/write schemas after the reference cutoff).
2. This tree keeps a re-export facade on ``tools/core.py`` (the §5.3 partial-
   split idiom, matching the shell facade) instead of the reference's
   no-facade cutover, so the reference's ``isdisjoint(vars(core))`` clause is
   replaced by the facade identity clause; the consumer-rebinding rows of the
   reference (browser/vision/query_code/edit_ops/test bindings) stay pending
   and ride with their own hygiene wave.
3. The frozen-tool-inventory clauses are dropped: ``ouroboros.tool_module_
   inventory`` is a D04-family v7 leaf absent from this tree; the
   non-catalog-owner and no-backedge clauses keep the structural half of that
   contract. The inventory clause returns with its leaf.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib

from ouroboros.tools import core, core_artifacts, core_file_tools
from ouroboros.tools.registry import ToolContext


REPO = pathlib.Path(__file__).parents[1]
TOOLS = REPO / "ouroboros" / "tools"

_MOVED_NAMES = frozenset({
    "_ListingFailure",
    "_MAX_DOCUMENT_FILE_BYTES",
    "_MAX_PHOTO_FILE_BYTES",
    "_MAX_VIDEO_FILE_BYTES",
    "_MEMORY_AT_DRIVE_MEMORY",
    "_SKILL_OWNER_STATE_FILENAMES",
    "_SUBAGENT_SECRET_FILE_NAMES",
    "_access_or_block",
    "_annotate_reread",
    "_coerce_line_window",
    "_coerce_start_char",
    "_data_list",
    "_data_read",
    "_detect_document_mime",
    "_detect_image_mime",
    "_detect_video_mime",
    "_direct_resource_binding",
    "_filter_subagent_secret_listing",
    "_filter_subagent_secret_repo_listing",
    "_is_cognitive_data_path",
    "_is_skill_owner_state_target",
    "_is_subagent_secret_data_path",
    "_is_subagent_secret_repo_path",
    "_is_subagent_secret_repo_target",
    "_list_dir",
    "_list_files",
    "_list_user_files_dir",
    "_local_readonly_resource_block",
    "_normalize_data_read_path",
    "_profile_roots_hint",
    "_read_file",
    "_render_line_slice",
    "_repo_list",
    "_repo_read",
    "_root_display_path",
    "_send_file",
    "_send_photo",
    "_send_video",
    "is_restricted_subagent_profile",
})


def test_core_leaves_are_non_catalog_owners_without_core_backedges():
    for module in (core_file_tools, core_artifacts):
        source_path = pathlib.Path(module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "get_tools"
            for node in tree.body
        )
        assert not any(
            isinstance(node, ast.ImportFrom)
            and node.module == "ouroboros.tools.core"
            for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.Import)
            and any(alias.name == "ouroboros.tools.core" for alias in node.names)
            for node in ast.walk(tree)
        )


def test_core_catalog_schema_bytes_and_handler_owners_are_stable():
    entries = core.get_tools()
    assert tuple(entry.name for entry in entries) == (
        "read_file",
        "list_files",
        "write_file",
        "edit_text",
        "send_photo",
        "send_video",
        "send_file",
        "send_links",
        "search_code",
        "escalate",
        "forward_to_worker",
    )
    schema_bytes = json.dumps(
        [entry.schema for entry in entries],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(schema_bytes).hexdigest() == (
        "f330a93f741c666aab47f06a76151dd7e056d489a08a153a225c18183cd5435c"
    )
    assert {
        entry.name: (entry.handler.__module__, entry.handler.__name__)
        for entry in entries
    } == {
        "read_file": ("ouroboros.tools.core_file_tools", "_read_file"),
        "list_files": ("ouroboros.tools.core_file_tools", "_list_files"),
        "write_file": ("ouroboros.tools.core", "_write_file"),
        "edit_text": ("ouroboros.tools.core", "_edit_text"),
        "send_photo": ("ouroboros.tools.core_artifacts", "_send_photo"),
        "send_video": ("ouroboros.tools.core_artifacts", "_send_video"),
        "send_file": ("ouroboros.tools.core_artifacts", "_send_file"),
        "send_links": ("ouroboros.tools.core_artifacts", "_send_links"),
        "search_code": ("ouroboros.tools.core", "_code_search"),
        "escalate": ("ouroboros.tools.core_artifacts", "_escalate"),
        "forward_to_worker": ("ouroboros.tools.core", "_forward_to_worker"),
    }
    artifact_names = {
        name for name in _MOVED_NAMES
        if name.startswith(("_MAX_", "_detect_", "_send_"))
    }
    assert artifact_names <= vars(core_artifacts).keys()
    assert (_MOVED_NAMES - artifact_names) <= vars(core_file_tools).keys()


def test_core_facade_reexports_every_moved_identity():
    """``tools/core.py`` keeps the exact leaf objects importable, so every
    existing importer (browser, vision, query_code, edit_ops, delegate_output,
    shell_guards, the test suites) sees no identity change."""
    for name in sorted(_MOVED_NAMES):
        owner = core_artifacts if name.startswith(("_MAX_", "_detect_", "_send_")) else core_file_tools
        assert hasattr(core, name), name
        assert getattr(core, name) is getattr(owner, name), name


def test_extracted_read_and_list_result_bytes_are_stable(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    (repo / "nested").mkdir(parents=True)
    data.mkdir()
    (repo / "sample.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    (repo / "nested" / "child.txt").write_text("inside\n", encoding="utf-8")
    ctx = ToolContext(repo_dir=repo, drive_root=data)

    assert core_file_tools._repo_read(ctx, "sample.txt").encode() == (
        b"# sample.txt \xe2\x80\x94 lines 1\xe2\x80\x932 of 2\nalpha\nbeta\n"
    )
    assert core_file_tools._repo_list(ctx, ".").encode() == (
        b'[\n  "nested/",\n  "sample.txt"\n]'
    )


def test_core_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines())
        for module in (core, core_file_tools, core_artifacts)
    }
    assert 1200 <= counts["ouroboros.tools.core"] <= 1499
    assert 750 <= counts["ouroboros.tools.core_file_tools"] <= 1000
    assert 150 <= counts["ouroboros.tools.core_artifacts"] <= 1000
