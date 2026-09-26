"""Structural contracts for the semantic-no-op core tool extraction.

``tools/core.py`` keeps a re-export facade, matching the shell facade, so
existing importers retain the exact leaf objects. The catalog schema hash
pins the current tool schemas and the handler map pins their owners.
The leaves are non-catalog owners without backedges into the facade.
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
    # Rolled for P1-9 (owner Q7=B): the escalate description states the ABCD rule and
    # QuizOption gained the optional `recommended` boolean. Rolled again for owner
    # decision 6C (structural-health wave 2): the forward_to_worker description now
    # names its second addressee, any host-listed independent root, delivered as a
    # message from an independent task. Rolled again for owner decision В17a=A: the
    # escalate description stops promising expiry at task end (a late answer reaches
    # the chat as an ordinary owner message) and the optional `max_wait_minutes`
    # bound joins its parameters. Rolled again for the owner's "autonomy first" decision:
    # the escalate description states when waiting is worth it (an irreversible or costly
    # next step, or a choice that is the owner's to make) and that waiting questions of one
    # batch share one wait, ended by the first incoming message; 935 -> 932 bytes. Rolled
    # again for owner decision 7A: the read_file description states that an absolute path
    # with no root selects the permitted root holding it (one sentence replaced, 103 -> 100
    # bytes). Rolled again for serial addressed turns: the forward_to_worker description
    # names its peer addressees (your own parent or a sibling, delivered as a message from a
    # peer task naming the relation; relay refused there), the 8000-char body bound and the
    # await_messages companion (395 -> 698 bytes); the `message` parameter description states
    # the bound. Rolled again for the truthful owner-question work (PR1, owner 1D/2A): the
    # escalate description and its nine field descriptions were replaced (the card is written
    # for a reader outside the room, names the source of the fork, and the question has no
    # quiz-specific length cap); schema shape, types, defaults and required keys are unchanged,
    # and the entry's literal moved beside its validator in core_artifacts (byte-identical
    # serialization). Rolled again for TZ-2 B1 (zero-option questions) and owner V13: the
    # escalate description offers 0-6 alternatives (none for an open question answered in the
    # human's own words), states that a shared wait ends on any incoming message and that a
    # plain-text clarification ends the turn while a waited question keeps it alive; the
    # `options` description says optional 0-6 and `options` leaves the required keys.
    # Diffing the whole catalog base to head shows exactly those edits and nothing else.
    assert hashlib.sha256(schema_bytes).hexdigest() == (
        "8dbf49802f42ef87f0279c102103d151774db22fdfaf32668907d971599cd7cd"
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
    assert counts["ouroboros.tools.core"] <= 1499
    assert counts["ouroboros.tools.core_file_tools"] <= 1000
    assert counts["ouroboros.tools.core_artifacts"] <= 1000
