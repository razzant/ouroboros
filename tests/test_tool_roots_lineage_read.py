"""A read-only child reads what its parent points it to (owner T4=A + 7A, #1105).

Measured on a live install: a root task put files into its own ``task_drive``
and sent four read-only children to check them; all four called ``read_file``
with the ABSOLUTE path and NO root, and the default ``active_workspace`` refused
them with ``outside selected root``. The recorded reason for hiding the
orchestrator roots from children (``tool_access.py``: "a child must not read
sibling projects") covers ``subagent_projects`` only. Now a child reads the
owner-visible Deliverables root and the ``task_drive``/``artifact_store`` of its
OWN lineage (parent and root ids from its own lineage fields), anchored on the
canonical data root while the child itself runs on a headless drive; and an
absolute path given without a root runs under the permitted root that
physically contains it, the path itself never rewritten. A sibling's or a
stranger's task files stay refused, and the refusal names the roots this
profile can actually use; a NAMED wrong root still refuses; secret-named files
in a parent's drive stay denied by name; ``subagent_projects`` stays top-level
only.
"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tool_access import (
    _POLICY,
    _TOP_LEVEL_PRINCIPAL_POLICY,
    decide_tool_access,
    summarize_subagent_profile,
)
from ouroboros.tools.registry import ToolContext, ToolRegistry

PARENT = "p07499dc017c01f83"
ROOT = "r5173b7c3c15d4c0b"
CHILD = "c11ae4fd0aa111111"
SIBLING = "s339e97de0b222222"
STRANGER = "x8b8af5e9cc333333"


@pytest.fixture
def geometry(tmp_path, monkeypatch):
    """The parent's task files live on the CANONICAL data root; the child runs
    on its own headless drive; the owner home is a fake tmp home."""
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    canonical = tmp_path / "data"
    headless = tmp_path / "headless"
    for path in (home, repo, canonical, headless):
        path.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    (repo / "README.md").write_text("repo readme\n", encoding="utf-8")
    parent_drive = canonical / "task_drives" / PARENT
    (parent_drive / "source" / "ouroboros").mkdir(parents=True)
    (parent_drive / "source" / "ouroboros" / "update_letter.py").write_text(
        "PARENT_DRIVE_BYTES = 1\n", encoding="utf-8")
    (parent_drive / "triage-draft.json").write_text('{"triage": "draft"}\n', encoding="utf-8")
    (parent_drive / ".env").write_text("SECRET_TOKEN=1\n", encoding="utf-8")
    (parent_drive / "settings.json").write_text('{"OPENAI_API_KEY": "sk-secret"}\n', encoding="utf-8")
    root_artifacts = canonical / "task_results" / "artifacts" / ROOT
    root_artifacts.mkdir(parents=True)
    (root_artifacts / "report.txt").write_text("ROOT_ARTIFACT_BYTES\n", encoding="utf-8")
    sibling_drive = canonical / "task_drives" / SIBLING
    sibling_drive.mkdir(parents=True)
    (sibling_drive / "notes.txt").write_text("SIBLING_BYTES\n", encoding="utf-8")
    stranger_artifacts = canonical / "task_results" / "artifacts" / STRANGER
    stranger_artifacts.mkdir(parents=True)
    (stranger_artifacts / "out.txt").write_text("STRANGER_BYTES\n", encoding="utf-8")
    deliverables = home / "Deliverables"
    deliverables.mkdir()
    (deliverables / "answer.txt").write_text("DELIVERABLE_BYTES needle\n", encoding="utf-8")
    return SimpleNamespace(
        home=home, repo=repo, canonical=canonical, headless=headless,
        parent_drive=parent_drive, root_artifacts=root_artifacts,
        sibling_drive=sibling_drive, stranger_artifacts=stranger_artifacts,
        deliverables=deliverables,
    )


def child_registry(geo, *, drive=None, acting=False):
    """A delegated child of PARENT under ROOT, through the real registry."""
    ctx = ToolContext(repo_dir=geo.repo, drive_root=drive or geo.headless, task_id=CHILD)
    ctx.budget_drive_root = str(geo.canonical)
    ctx.task_metadata = {
        "delegation_role": "subagent",
        "parent_task_id": PARENT,
        "root_task_id": ROOT,
        "budget_drive_root": str(geo.canonical),
    }
    if acting:
        work = geo.home / "work"
        work.mkdir(exist_ok=True)
        ctx.workspace_root = work
        ctx.workspace_mode = "external"
        ctx.task_constraint = TaskConstraint(
            mode="acting_subagent", allow_enable=False, surface="external_workspace")
    else:
        ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent", allow_enable=False)
    registry = ToolRegistry(repo_dir=geo.repo, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry, ctx


def top_level_registry(geo, *, external=False):
    """A top-level task (the ROOT itself) on the canonical drive; ``external``
    gives it a workspace outside the system repo so the two roots differ."""
    ctx = ToolContext(repo_dir=geo.repo, drive_root=geo.canonical, task_id=ROOT)
    if external:
        work = geo.home / "work"
        work.mkdir(exist_ok=True)
        ctx.workspace_root = work
        ctx.workspace_mode = "external"
    registry = ToolRegistry(repo_dir=geo.repo, drive_root=geo.canonical)
    registry.set_context(ctx)
    return registry, ctx


# --- the lineage read: parent's and root's task files, never a sibling's ------

def test_child_reads_its_parents_task_drive_from_a_headless_drive(geometry):
    registry, ctx = child_registry(geometry)
    target = geometry.parent_drive / "source" / "ouroboros" / "update_letter.py"

    out = registry.execute("read_file", {"root": "task_drive", "path": str(target)})

    assert "PARENT_DRIVE_BYTES" in out, out
    assert out.startswith("# task_drive:"), out
    assert ctx.last_read_view["opened_root"] == "task_drive"
    assert ctx.last_read_view["target"] == str(target.resolve())


def test_child_reads_the_root_tasks_artifact(geometry):
    registry, _ctx = child_registry(geometry)
    target = geometry.root_artifacts / "report.txt"

    out = registry.execute("read_file", {"root": "artifact_store", "path": str(target)})

    assert "ROOT_ARTIFACT_BYTES" in out, out
    assert out.startswith("# artifact_store:"), out


def test_single_drive_child_reads_the_parent_drive_too(geometry):
    registry, _ctx = child_registry(geometry, drive=geometry.canonical)
    out = registry.execute(
        "read_file", {"root": "task_drive", "path": str(geometry.parent_drive / "triage-draft.json")})
    assert '"triage": "draft"' in out, out


def test_an_acting_child_shares_the_lineage_read(geometry):
    registry, _ctx = child_registry(geometry, acting=True)
    out = registry.execute(
        "read_file", {"root": "task_drive", "path": str(geometry.parent_drive / "triage-draft.json")})
    assert '"triage": "draft"' in out, out


def test_a_siblings_drive_and_a_strangers_artifacts_stay_refused(geometry):
    registry, _ctx = child_registry(geometry)

    sibling = registry.execute(
        "read_file", {"root": "task_drive", "path": str(geometry.sibling_drive / "notes.txt")})
    stranger = registry.execute(
        "read_file", {"root": "artifact_store", "path": str(geometry.stranger_artifacts / "out.txt")})

    assert "SIBLING_BYTES" not in sibling and "outside selected root=task_drive" in sibling, sibling
    assert "STRANGER_BYTES" not in stranger and "outside selected root=artifact_store" in stranger, stranger


def test_lineage_is_read_only_even_for_a_top_level_parent_drive(geometry):
    """The rule is a READ rule: an acting child never writes into its parent's
    drive through the same path, and its own task_drive stays the write target
    the matrix says (none for an acting child)."""
    registry, _ctx = child_registry(geometry, acting=True)
    target = geometry.parent_drive / "triage-draft.json"
    before = target.read_text(encoding="utf-8")

    out = registry.execute("write_file", {"root": "task_drive", "path": str(target), "content": "x"})

    assert out.startswith("⚠️"), out
    assert target.read_text(encoding="utf-8") == before


def test_the_lineage_redirect_is_a_read_redirect_whatever_the_matrix_says(geometry):
    """The write above is refused by the matrix first, so it never reaches the
    resolver. This one does: the redirect onto a lineage root exists for READ
    operations alone, so a profile whose matrix allows the write (an acting child
    in cyber_pro) still cannot be resolved into its parent's drive."""
    from ouroboros.tool_access import _resolve_target_in_selected_base

    _registry, ctx = child_registry(geometry, acting=True)
    target = geometry.parent_drive / "triage-draft.json"
    own_base = geometry.headless / "task_drives" / CHILD

    assert _resolve_target_in_selected_base(
        ctx, root="task_drive", base_path=own_base, path=str(target), operation="read",
    ) == target.resolve()
    for operation in ("write", "edit"):
        with pytest.raises(ValueError):
            _resolve_target_in_selected_base(
                ctx, root="task_drive", base_path=own_base, path=str(target), operation=operation)


# --- owner 7A: an absolute path without a root runs under the root holding it --

def test_child_reads_its_parents_task_drive_by_absolute_path_and_no_root(geometry):
    """The 2026-09-19 shape: the child names the absolute path of a file in
    the PARENT's task_drive and no root; the host selects task_drive."""
    registry, ctx = child_registry(geometry)
    target = geometry.parent_drive / "source" / "ouroboros" / "update_letter.py"

    out = registry.execute("read_file", {"path": str(target)})

    assert "PARENT_DRIVE_BYTES" in out, out
    assert ctx.last_read_view["opened_root"] == "task_drive"
    assert ctx.last_read_view["target"] == str(target.resolve())


def test_child_reads_the_root_tasks_artifact_by_absolute_path_and_no_root(geometry):
    registry, ctx = child_registry(geometry)

    out = registry.execute("read_file", {"path": str(geometry.root_artifacts / "report.txt")})

    assert "ROOT_ARTIFACT_BYTES" in out, out
    assert ctx.last_read_view["opened_root"] == "artifact_store"


def test_child_reads_and_lists_deliverables_by_absolute_path_and_no_root(geometry):
    registry, ctx = child_registry(geometry)

    read = registry.execute("read_file", {"path": str(geometry.deliverables / "answer.txt")})
    listing = registry.execute("list_files", {"path": str(geometry.deliverables)})

    assert "DELIVERABLE_BYTES" in read, read
    assert ctx.last_read_view["opened_root"] == "deliverables"
    assert "answer.txt" in json.loads(listing), listing


def test_search_selects_the_root_for_its_own_operation(geometry):
    """Selection follows the tool's operation, and the matrix is closed under
    read⇒search (TZ-1 E): a child searches Deliverables and its lineage's
    task_drive exactly where it may read them — the per-file secret guard still
    hides the parent's .env and settings.json — while a sibling's drive stays
    refused and the refusal names the roots this profile can search."""
    registry, _ctx = child_registry(geometry)

    found = registry.execute("search_code", {"query": "needle", "path": str(geometry.deliverables)})
    parent = registry.execute("search_code", {"query": "PARENT|SECRET|sk-", "regex": True,
                                              "path": str(geometry.parent_drive)})
    refused = registry.execute("search_code", {"query": "SIBLING", "path": str(geometry.sibling_drive)})

    assert "answer.txt:1:" in found and "needle" in found, found
    assert "PARENT_DRIVE_BYTES" in parent, parent
    assert "SECRET_TOKEN" not in parent and "sk-secret" not in parent, parent
    assert "SIBLING_BYTES" not in refused, refused
    assert "outside selected root=active_workspace" in refused, refused
    named = refused.split("Roots your profile can search:")[1]
    assert "deliverables" in named and "task_drive" in named and "user_files" not in named, refused


def test_a_siblings_or_strangers_file_without_root_is_refused_naming_real_roots(geometry):
    registry, _ctx = child_registry(geometry)

    sibling = registry.execute("read_file", {"path": str(geometry.sibling_drive / "notes.txt")})
    stranger = registry.execute("read_file", {"path": str(geometry.stranger_artifacts / "out.txt")})

    for out in (sibling, stranger):
        assert "SIBLING_BYTES" not in out and "STRANGER_BYTES" not in out, out
        assert "outside selected root=active_workspace" in out, out
        named = out.split("Roots your profile can read:")[1]
        assert "task_drive" in named and "deliverables" in named, out
        assert "user_files" not in named and "subagent_projects" not in named, out


def test_a_named_wrong_root_still_refuses_a_reachable_file(geometry):
    """No silent re-rooting: the containing root is chosen only when none was named."""
    registry, _ctx = child_registry(geometry)
    target = geometry.parent_drive / "triage-draft.json"

    out = registry.execute("read_file", {"root": "artifact_store", "path": str(target)})

    assert '"triage"' not in out and "outside selected root=artifact_store" in out, out


def test_an_unreachable_path_without_root_never_reads_a_same_named_workspace_mirror(geometry):
    """The `01aea0663` pin on the no-root path: an absolute path no permitted
    root holds is refused, never sliced by safe_relpath into a same-named file
    inside the workspace."""
    registry, _ctx = child_registry(geometry)
    outside = geometry.home / "elsewhere" / "target.txt"  # under the owner home: no child root
    outside.parent.mkdir()
    outside.write_text("correct", encoding="utf-8")
    mirror = geometry.repo.joinpath(*outside.parts[1:])
    mirror.parent.mkdir(parents=True)
    mirror.write_text("wrong-file", encoding="utf-8")

    out = registry.execute("read_file", {"path": str(outside)})

    assert "outside selected root" in out, out
    assert "wrong-file" not in out and "correct" not in out, out


def test_top_level_task_reads_the_system_repo_by_absolute_path_without_root(geometry):
    """An external-workspace task names no root for an absolute repo path: the
    host selects system_repo; the same path under a NAMED active_workspace stays refused."""
    registry, _ctx = top_level_registry(geometry, external=True)
    target = geometry.repo / "README.md"

    out = registry.execute("read_file", {"path": str(target)})
    named = registry.execute("read_file", {"root": "active_workspace", "path": str(target)})

    assert "repo readme" in out and out.startswith("# system_repo:README.md"), out
    assert "repo readme" not in named and "outside selected root=active_workspace" in named, named


def test_top_level_task_reads_its_own_artifact_by_absolute_path_without_root(geometry):
    """The deepest containing root wins: the artifact lies under runtime_data
    too, and the binding names artifact_store."""
    registry, ctx = top_level_registry(geometry)

    out = registry.execute("read_file", {"path": str(geometry.root_artifacts / "report.txt")})

    assert "ROOT_ARTIFACT_BYTES" in out, out
    assert ctx.last_read_view["opened_root"] == "artifact_store"


def test_dispatch_selects_a_root_only_for_an_absolute_path_without_one(geometry):
    from ouroboros.tools.tool_resolution import _normalize_dispatch_path_args_result

    _registry, ctx = child_registry(geometry)
    parent_file = str(geometry.parent_drive / "triage-draft.json")

    selected = {"path": parent_file}
    assert _normalize_dispatch_path_args_result(ctx, "read_file", selected).text == ""
    assert selected == {"path": parent_file, "root": "task_drive"}  # the path is never rewritten
    named = {"root": "artifact_store", "path": parent_file}
    _normalize_dispatch_path_args_result(ctx, "read_file", named)
    assert named["root"] == "artifact_store"
    for untouched in ({"path": "README.md"}, {"path": str(geometry.sibling_drive / "notes.txt")}):
        _normalize_dispatch_path_args_result(ctx, "read_file", untouched)
        assert "root" not in untouched, untouched
    in_workspace = {"path": str(geometry.repo / "README.md")}
    _normalize_dispatch_path_args_result(ctx, "read_file", in_workspace)
    assert in_workspace == {"path": "README.md"}  # the in-workspace normalization is unchanged
    query = {"path": parent_file, "op": "digest"}
    _normalize_dispatch_path_args_result(ctx, "query_code", query)
    assert "root" not in query  # query_code keeps its own external-target contract


# --- secrets in a parent's drive stay denied by NAME -------------------------

@pytest.mark.parametrize("name", [".env", "settings.json"])
def test_secret_named_files_in_the_parents_drive_stay_denied(geometry, name):
    registry, _ctx = child_registry(geometry)
    out = registry.execute("read_file", {"root": "task_drive", "path": str(geometry.parent_drive / name)})
    assert "READ_FILE_BLOCKED" in out and "secret" in out, out
    assert "SECRET_TOKEN" not in out and "sk-secret" not in out


def test_child_lists_the_parents_drive_with_secret_names_hidden(geometry):
    registry, _ctx = child_registry(geometry)

    out = registry.execute("list_files", {"root": "task_drive", "path": str(geometry.parent_drive)})

    items = json.loads(out)
    assert "triage-draft.json" in items and "source/" in items, items
    assert ".env" not in items and "settings.json" not in items, items
    assert any("hidden from this subagent" in item for item in items), items


def test_child_lists_deliverables_with_secret_names_hidden(geometry):
    """Deliverables is a new listing root for the child, so it gets the same
    secret-name filter as every other root it lists; an ordinary file stays."""
    (geometry.deliverables / ".env").write_text("SECRET_TOKEN=sk-secret\n", encoding="utf-8")
    registry, _ctx = child_registry(geometry)

    items = json.loads(registry.execute("list_files", {"root": "deliverables", "path": str(geometry.deliverables)}))

    assert "answer.txt" in items and ".env" not in items, items
    assert any("hidden from this subagent" in item for item in items), items


# --- the pure lineage function ------------------------------------------------

def test_lineage_task_ids_are_own_parent_and_root_and_nothing_else(geometry):
    from ouroboros.tool_access import lineage_task_ids

    _registry, ctx = child_registry(geometry)
    assert lineage_task_ids(ctx) == (CHILD, PARENT, ROOT)

    ctx.task_metadata["root_task_id"] = PARENT  # parent IS the root: no duplicate
    assert lineage_task_ids(ctx) == (CHILD, PARENT)

    ctx.task_metadata["parent_task_id"] = "../escape"  # malformed ids are dropped, not guessed
    ctx.task_metadata["root_task_id"] = ""
    assert lineage_task_ids(ctx) == (CHILD,)

    top = ToolContext(repo_dir=geometry.repo, drive_root=geometry.canonical, task_id=ROOT)
    assert lineage_task_ids(top) == (ROOT,)


def test_lineage_read_base_names_the_containing_lineage_root(geometry):
    from ouroboros.tool_access import lineage_read_base

    _registry, ctx = child_registry(geometry)
    parent_file = geometry.parent_drive / "triage-draft.json"
    root_file = geometry.root_artifacts / "report.txt"

    assert lineage_read_base(ctx, "task_drive", parent_file) == geometry.parent_drive.resolve()
    assert lineage_read_base(ctx, "artifact_store", root_file) == geometry.root_artifacts.resolve()
    # The label must match the physical kind: a task_drive path is not an artifact base.
    assert lineage_read_base(ctx, "artifact_store", parent_file) is None
    assert lineage_read_base(ctx, "task_drive", geometry.sibling_drive / "notes.txt") is None
    assert lineage_read_base(ctx, "runtime_data", parent_file) is None
    # The child's OWN task root on its headless drive is a lineage base as well.
    own = geometry.headless / "task_drives" / CHILD / "scratch.txt"
    assert lineage_read_base(ctx, "task_drive", own) == (geometry.headless / "task_drives" / CHILD).resolve()


# --- Deliverables: a read-only child reads, never writes ---------------------

def test_deliverables_row_reads_only_and_only_for_the_readonly_child():
    for op in ("read", "list", "search"):
        assert decide_tool_access(profile="local_readonly_subagent", root="deliverables", operation=op).allow, op
    for op in ("write", "edit", "shell", "vcs", "service", "review", "delegate"):
        assert not decide_tool_access(profile="local_readonly_subagent", root="deliverables", operation=op).allow, op
    for profile in ("acting_subagent", "local_readonly_subagent"):
        assert not decide_tool_access(profile=profile, root="subagent_projects", operation="read").allow, profile
    assert not decide_tool_access(profile="acting_subagent", root="deliverables", operation="read").allow
    # Top-level principals are untouched: one shared matrix object, unchanged rows.
    for profile in ("workspace_task", "external_workspace_task", "self_modification"):
        assert _POLICY[profile] is _TOP_LEVEL_PRINCIPAL_POLICY
    assert _TOP_LEVEL_PRINCIPAL_POLICY["deliverables"] == {"read", "list", "search"}
    assert _TOP_LEVEL_PRINCIPAL_POLICY["subagent_projects"] == {"read", "list", "search"}


def test_child_reads_lists_and_searches_deliverables_but_cannot_touch_them(geometry):
    registry, _ctx = child_registry(geometry)
    answer = geometry.deliverables / "answer.txt"

    read = registry.execute("read_file", {"root": "deliverables", "path": "answer.txt"})
    listing = registry.execute("list_files", {"root": "deliverables", "path": "."})
    search = registry.execute("search_code", {"root": "deliverables", "query": "needle"})

    assert "DELIVERABLE_BYTES" in read and read.startswith("# deliverables:answer.txt"), read
    assert "answer.txt" in json.loads(listing), listing
    assert "deliverables:answer.txt:1:" in search, search

    write = registry.execute("write_file", {"root": "deliverables", "path": "answer.txt", "content": "x"})
    edit = registry.execute("edit_text", {"root": "deliverables", "path": "answer.txt",
                                          "old_str": "needle", "new_str": "x"})
    shell = registry.execute("run_command", {"command": "ls", "cwd": "deliverables"})
    for out in (write, edit, shell):
        assert out.startswith("⚠️"), out
    assert answer.read_text(encoding="utf-8") == "DELIVERABLE_BYTES needle\n"
    assert "⚠️" in registry.execute("list_files", {"root": "subagent_projects", "path": "."})


def test_readonly_child_schema_enums_follow_the_matrix(geometry):
    registry, _ctx = child_registry(geometry)

    def enum(name):
        return registry.get_schema_by_name(name)["function"]["parameters"]["properties"]["root"]["enum"]

    for name in ("read_file", "list_files"):
        assert "deliverables" in enum(name), name
        assert "subagent_projects" not in enum(name) and "user_files" not in enum(name), name
    # Read⇒search closure (TZ-1 E): the child searches every root it may read.
    assert set(enum("search_code")) == {
        "active_workspace", "system_repo", "skill_payload", "deliverables",
        "runtime_data", "task_drive", "artifact_store",
    }
    assert enum("query_code") == ["active_workspace", "system_repo"]


# --- both sides see what the child can read -----------------------------------

def test_profile_summary_names_readable_and_unreadable_roots(monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    readonly = summarize_subagent_profile("local_readonly_subagent", effective_lane="light").splitlines()
    assert len(readonly) == 2, readonly
    assert readonly[0].startswith("child capabilities — ") and "model_lane=light" in readonly[0]
    readable, unreadable = readonly[1].split(" · unreadable=")
    assert readable.startswith("readable=") and "deliverables" in readable and "task_drive" in readable
    assert "parent" in readable and "sibling" in readable, readable
    assert unreadable == "subagent_projects, user_files", unreadable

    acting = summarize_subagent_profile("acting_subagent").splitlines()
    assert len(acting) == 2, acting
    acting_readable, acting_unreadable = acting[1].split(" · unreadable=")
    assert "deliverables" not in acting_readable and "deliverables" in acting_unreadable
