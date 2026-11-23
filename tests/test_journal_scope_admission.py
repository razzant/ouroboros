"""One journal read policy, whichever interface reads: address tests on fictional identifiers only.

1. ``_project_scope(write=False)``, the decision ``journal_read`` applies: a root, a parent-independent root
   acting for a place binding and a real place turn read any project; a child (a place-bound parent's child
   included) and an undetermined actor read only their own.
2. ``journal_resource_read_refusal`` applies that same decision to a project's journal or reflections file
   named by path before content I/O; other paths keep their rules.
3. Chronicle's one source byte path (a read, a quote check) applies the cross-focus catalogue's role rule
   to a retained focus source before its bytes are loaded.
4. A workpad's version read selects the page's own scope (``journal=False``, as ``workpad_read`` does), never
   the journal default; the journal default keeps its own, stricter decision.
5. The store comparison itself uses one filesystem-identity rule (``confined_files.same_store_object``):
   a differently spelled but physically identical store folder still locates the same object; a genuinely
   separate copy stays separate; an identity that cannot be observed is neither a match nor a confirmed
   difference, so the guard keeps comparing the remaining candidates and, with none observable, refuses.

Negative cases decide over fictional ids with no stored data, and the source loader fails if called.
Positive controls read bytes only of objects the actor may read.
"""
from __future__ import annotations

import types

import pytest

_BINDING = {"binding_id": "place-fictional"}


def test_focus_artifact_homes_follow_producer_layout(tmp_path):
    from pathlib import Path

    from ouroboros.headless import TASK_DRIVES_DIR, task_artifacts_dir, task_state_dir
    from ouroboros.task_finalization import FOCUS_SOURCE_CATEGORY_DIR
    from ouroboros.tools.recent_tasks import _focus_artifact_homes

    root = tmp_path / "fictional-data"
    category = Path(FOCUS_SOURCE_CATEGORY_DIR)
    for drive in (root, root / TASK_DRIVES_DIR / "producer", task_state_dir(root, "producer") / "data"):
        home = task_artifacts_dir(drive, "fictional-task", create=False) / category
        assert home in _focus_artifact_homes(home / "fixture.json", (root,), category)


def _actor(tmp_path, kind, *, project_id="mine", task_id="t1"):
    metadata = {"budget_drive_root": str(tmp_path)}
    if kind in ("root", "bound_root"):
        metadata["root_task_id"] = task_id
    if kind == "bound_root":
        metadata["presence_binding_authority"] = dict(_BINDING)
    if kind == "place":
        metadata["presence"] = dict(_BINDING)
    if kind in ("child", "place_child"):
        metadata.update(parent_task_id="p1", delegation_role="subagent")
    if kind == "place_child":
        metadata["presence_binding_authority"] = dict(_BINDING)
    if kind == "unverified_place":
        metadata.update(presence={}, root_task_id=task_id)
    contract = {"lineage": {"parent_task_id": "p1", "delegation_role": "subagent"}} if kind == "contract_child" else {}
    return types.SimpleNamespace(task_id=task_id, drive_root=str(tmp_path), project_id=project_id,
                                 is_direct_chat=False, task_metadata=metadata, task_contract=contract)


READ_ANY = ("root", "bound_root", "place")
READ_OWN = ("child", "place_child", "contract_child", "unverified_place", "undetermined")


def test_project_read_scope_by_actor(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    from ouroboros.tools.project_journal import _project_scope

    for kind in READ_ANY:
        assert _project_scope(_actor(tmp_path, kind), "beta", write=False) == ("beta", "", ""), kind
    for kind in READ_OWN:
        actor = _actor(tmp_path, kind)
        assert _project_scope(actor, "mine", write=False)[0] == "mine", kind
        pid, refusal, code = _project_scope(actor, "beta", write=False)
        assert pid == "" and code == "ACCESS_BLOCKED" and "only its current project" in refusal, kind
    # An undetermined actor without a project of its own is no root either.
    pid, _refusal, code = _project_scope(_actor(tmp_path, "undetermined", project_id=""), "beta", write=False)
    assert pid == "" and code == "ACCESS_BLOCKED"
    # Writes keep their rule: only the current project.
    for kind in READ_ANY + READ_OWN:
        assert _project_scope(_actor(tmp_path, kind), "beta", write=True)[2] == "ACCESS_BLOCKED", kind


def test_documented_readers_page_a_foreign_journal(tmp_path, monkeypatch):
    """Positive control through the real reader on synthetic rows of an allowed object."""
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    from ouroboros.project_facts import project_journal_path
    from ouroboros.tools.project_journal import _journal_read

    journal = project_journal_path("alpha")
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text('{"ts": "2026-01-01T00:00:00Z", "kind": "note", "text": "SYNTHETIC-ALPHA-ROW"}\n',
                       encoding="utf-8")
    for kind in READ_ANY:
        assert "SYNTHETIC-ALPHA-ROW" in _journal_read(_actor(tmp_path, kind), project_id="alpha"), kind
    # A child's refusal precedes the store: "beta" holds no data at all.
    for kind in READ_OWN:
        for source in ("journal", "reflections"):
            assert "TOOL_FORBIDDEN" in _journal_read(_actor(tmp_path, kind), project_id="beta", source=source)


def test_journal_resource_by_path_follows_the_same_scope(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    from ouroboros.project_facts import project_journal_path, project_reflections_path, project_workpad_path
    from ouroboros.tools.core_file_tools import _runtime_data_read_block
    from ouroboros.tools.project_journal import journal_resource_read_refusal

    foreign = (project_journal_path("beta"), project_reflections_path("beta"))  # never created
    for kind in READ_OWN:
        actor = _actor(tmp_path, kind)
        for path in foreign:
            assert "TOOL_FORBIDDEN" in journal_resource_read_refusal(actor, path), (kind, path)
            assert "TOOL_FORBIDDEN" in _runtime_data_read_block(actor, path, root="user_files"), (kind, path)
    for kind in READ_ANY:
        for path in foreign:
            assert journal_resource_read_refusal(_actor(tmp_path, kind), path) == "", (kind, path)
    child = _actor(tmp_path, "child")
    assert journal_resource_read_refusal(child, project_journal_path("mine")) == ""
    assert journal_resource_read_refusal(child, project_reflections_path("mine")) == ""
    # Not a journal resource: no new rule here, the interface's own rules decide.
    assert journal_resource_read_refusal(child, project_workpad_path("beta")) == ""
    assert journal_resource_read_refusal(child, tmp_path / "state" / "journal.jsonl") == ""
    assert _runtime_data_read_block(child, project_journal_path("mine"), root="runtime_data")  # stricter interface


def test_journal_guard_distinguishes_unobserved_identity_from_difference(tmp_path, monkeypatch):
    """A confirmed mismatch leaves the object to the interface's own rules, an identity no candidate could
    observe is refused rather than decided, and an unobservable candidate never hides a later admissible
    one: the scope decision still answers. Fictional ids and mocked comparison outcomes; no file is read
    or written."""
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    from ouroboros import confined_files
    from ouroboros.tools.project_journal import journal_resource_read_refusal

    child, root = _actor(tmp_path, "child"), _actor(tmp_path, "root")
    elsewhere = tmp_path / "elsewhere"
    target = elsewhere / "projects" / "beta" / "journal.jsonl"

    monkeypatch.setattr(confined_files, "same_store_object", lambda *_args, **_kwargs: False)
    assert journal_resource_read_refusal(child, target, data_roots=[tmp_path]) == ""

    state = {"unobserved": True}

    def unobservable_once_then_compare(candidate, source, **_kwargs):
        if state["unobserved"]:
            state["unobserved"] = False
            raise PermissionError("synthetic identity observation failed")
        return candidate == source

    monkeypatch.setattr(confined_files, "same_store_object", unobservable_once_then_compare)
    assert "only its current project" in journal_resource_read_refusal(child, target, data_roots=[tmp_path, elsewhere])
    assert journal_resource_read_refusal(root, target, data_roots=[tmp_path, elsewhere]) == ""

    def always_unobservable(_candidate, _source, **_kwargs):
        raise PermissionError("synthetic identity observation failed")

    monkeypatch.setattr(confined_files, "same_store_object", always_unobservable)
    for actor in (child, root):
        assert "could not be observed" in journal_resource_read_refusal(actor, target, data_roots=[tmp_path]), actor


def _focus_ref():
    """A fictional focus-source reference; nothing is stored under it."""
    return {"kind": "task_source", "root": "artifact_store", "task_id": "author-fictional",
            "path": "source_handles/context_checkpoints/focus_source_journal_read-" + "0" * 64 + ".md",
            "size": 1, "sha256": "0" * 64}


def test_retained_focus_source_is_refused_before_the_loader(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    import ouroboros.artifacts as artifacts
    from ouroboros.tools import chronicle
    from ouroboros.tools.recent_tasks import RetainedFocusForbidden, retained_focus_source_refusal

    def loader_must_not_run(*_args, **_kwargs):
        raise AssertionError("the source loader ran for a refused actor")

    monkeypatch.setattr(artifacts, "read_actor_source_bytes", loader_must_not_run)
    for kind in ("child", "place_child", "place"):
        actor = _actor(tmp_path, kind)
        assert "TOOL_FORBIDDEN" in retained_focus_source_refusal(actor, tmp_path, _focus_ref()), kind
        assert "TOOL_FORBIDDEN" in chronicle._memory_read(actor, source_ref=_focus_ref()), kind
        with pytest.raises(RetainedFocusForbidden):
            chronicle._mark_target(actor, tmp_path, None, {"node_id": "", "address": "", "task_id": "",
                                                           "source_ref": _focus_ref(), "quote": "row"})


def test_allowed_retained_sources_read_and_quote_real_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.tools import chronicle

    focus = {**store_actor_source_bytes(tmp_path, "author", category="context_checkpoints",
                                        source_id="focus_source_journal_read",
                                        data=b"SYNTHETIC-FOCUS-SNAPSHOT\nrow one\n", extension="md"),
             "task_id": "author"}
    root = _actor(tmp_path, "root", task_id="r1")
    assert "SYNTHETIC-FOCUS-SNAPSHOT" in chronicle._memory_read(root, source_ref=focus)
    target, _room, refusal = chronicle._mark_target(root, tmp_path, None, {
        "node_id": "", "address": "", "task_id": "", "source_ref": focus, "quote": "row one"})
    assert refusal == "" and target == focus
    # An ordinary retained source stays readable to a child.
    ordinary = {**store_actor_source_bytes(tmp_path, "author", category="tool_results", source_id="call1",
                                           data=b"SYNTHETIC-TOOL-OUTPUT\n", extension="txt"),
                "task_id": "author"}
    assert "SYNTHETIC-TOOL-OUTPUT" in chronicle._memory_read(_actor(tmp_path, "child"), source_ref=ordinary)


def test_retained_focus_admission_uses_the_artifact_object(tmp_path, monkeypatch):
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.tools.core_file_tools import _runtime_data_read_block

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    locator = _focus_ref()
    origin = task_artifact_dir_path(tmp_path, locator["task_id"]) / locator["path"]
    unrelated = tmp_path / "workspace" / locator["path"]
    assert not origin.exists() and not unrelated.exists()
    recipient = _actor(tmp_path, "contract_child")
    assert "retained focus source" in _runtime_data_read_block(recipient, origin, root="artifact_store")
    assert _runtime_data_read_block(recipient, unrelated, root="active_workspace") == ""
    assert _runtime_data_read_block(_actor(tmp_path, "root"), origin, root="artifact_store") == ""


def test_packed_source_is_checked_before_materialization(tmp_path, monkeypatch):
    from pathlib import Path

    from ouroboros import evidence_packs
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.tools.core_file_tools import _runtime_data_read_check

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    locator = _focus_ref()
    base = task_artifact_dir_path(tmp_path, locator["task_id"])
    assert not (base / locator["path"]).exists()
    monkeypatch.setattr(evidence_packs, "packed_handle_names", lambda *_: [Path(locator["path"]).name])
    monkeypatch.setattr(evidence_packs, "materialize_handle", lambda *_: pytest.fail("must decide before restoring"))
    check = _runtime_data_read_check(_actor(tmp_path, "child"), root="artifact_store")
    evidence_packs.materialize_tree(base, path_allowed=lambda path: not check(path))


def test_selection_reader_uses_receiver_scope_and_narrowed_profile(tmp_path, monkeypatch):
    from ouroboros.delegate_readonly_inputs import receiving_reader
    from ouroboros.tool_access import active_tool_profile
    from ouroboros.tools.project_journal import _project_scope
    from ouroboros.tools.tool_context import ToolContext

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    root = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="starter", project_id="own")
    root.task_contract = {"lineage": {"delegation_role": "root", "root_task_id": "starter"}}
    assert _project_scope(root, "beta") == ("beta", "", "")
    reader = receiving_reader(root)
    assert active_tool_profile(reader) == "local_readonly_subagent"
    assert _project_scope(reader, "beta")[2] == "ACCESS_BLOCKED"
    assert _project_scope(reader, "own") == ("own", "", "")
    child = receiving_reader(root, child_task_id="receiver", project_id="selected")
    assert child.task_id == "receiver" and child.task_contract["lineage"]["parent_task_id"] == "starter"
    assert _project_scope(child, "selected") == ("selected", "", "")
    assert _project_scope(child, "own")[2] == "ACCESS_BLOCKED"


def test_same_receiver_admits_the_source_and_copy_of_an_export(tmp_path, monkeypatch):
    from ouroboros import delegate_readonly_inputs as inputs
    from ouroboros.tool_access import resource_root_path
    from ouroboros.tools.tool_context import ToolContext

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    root = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="starter", project_id="my-project")
    source = resource_root_path(root, "task_drive") / "export.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"one synthetic export\n")
    source_readers, copy_readers = [], []
    admit_source, admit_copy = inputs.admit_selected_source, inputs._admit_copy_address

    def record_source(ctx, binding, **kwargs):
        source_readers.append(ctx)
        return admit_source(ctx, binding, **kwargs)

    def record_copy(ctx, path):
        copy_readers.append(ctx)
        return admit_copy(ctx, path)

    monkeypatch.setattr(inputs, "admit_selected_source", record_source)
    monkeypatch.setattr(inputs, "_admit_copy_address", record_copy)
    [receipt] = inputs.schedule_selected_inputs(root, [{"root": "task_drive", "path": "export.txt"}],
                                                "receiver", project_id="my-project")
    assert receipt["status"] == "prepared"
    assert source_readers[0] is root and len(source_readers) == 2 and len(copy_readers) == 1
    assert copy_readers[0] is source_readers[1]
    assert (copy_readers[0].task_id, copy_readers[0].project_id) == ("receiver", "my-project")


def test_workpad_versions_select_the_page_policy_not_the_journal_default(tmp_path, monkeypatch):
    """A workpad's version read applies the policy ``workpad_read`` already chose, never the journal default:
    both doors hand the resolver ``journal=False``. Substituted resolvers record the selection only; no page
    is opened for either caller."""
    from ouroboros import memory_texts
    from ouroboros.tools import project_journal

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    actor = _actor(tmp_path, "child")
    selected = []

    def scope_recording(ctx, explicit, *, write=False, journal=True):
        selected.append(("memory_read", explicit, write, journal))
        return "mine", "", ""

    monkeypatch.setattr(project_journal, "_project_scope", scope_recording)
    assert memory_texts.resolve_source(actor, "workpad:mine").name == "workpad:mine"
    assert memory_texts.resolve_source(actor, "workpad").name == "workpad:mine"
    assert selected == [("memory_read", "mine", False, False), ("memory_read", "", False, False)]

    def resolve_recording(ctx, explicit, *, write=False, journal=True):
        selected.append(("workpad_read", explicit, write, journal))
        return "", "recorded refusal"

    monkeypatch.setattr(project_journal, "_resolve_project_id", resolve_recording)
    assert project_journal._workpad_page(actor, "mine") == ("", None, "recorded refusal (workpad_read)")
    assert selected[-1][1:] == selected[0][1:], "both doors select the page's policy, not the journal default"


def test_store_identity_distinguishes_difference_from_failed_comparison(tmp_path, monkeypatch):
    from pathlib import Path

    from ouroboros.confined_files import same_store_object

    left, right, separate = (tmp_path / name for name in ("one", "two", "three"))

    def compare(self, other):
        if self == separate or other == separate:
            return False
        return {self, other} == {left, right}

    monkeypatch.setattr(Path, "samefile", compare)
    assert same_store_object(left, left, strict=True)
    assert same_store_object(left, right, strict=True)
    assert not same_store_object(left, separate, strict=True)

    def unreadable(self, other):
        raise PermissionError("synthetic identity observation failed")

    monkeypatch.setattr(Path, "samefile", unreadable)
    with pytest.raises(PermissionError):
        same_store_object(left, right, strict=True)


def test_own_journal_keeps_generic_read_profile_rules(tmp_path, monkeypatch):
    from ouroboros import config
    from ouroboros.project_facts import project_journal_path
    from ouroboros.tools.core_file_tools import read_admission
    from ouroboros.tools.project_journal import _journal_read
    from ouroboros.tools.tool_context import ToolContext

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path))
    journal = project_journal_path("mine")
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text('{"ts": "2026-01-01T00:00:00Z", "kind": "note", "text": "SYNTHETIC-OWN-ROW"}\n',
                       encoding="utf-8")
    child = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path, task_id="reader", project_id="mine")
    child.task_metadata = {"parent_task_id": "parent", "delegation_role": "subagent"}
    child.task_constraint = {"mode": "acting_subagent"}
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "cyber_pro")
    _, binding, refusal = read_admission(child, "user_files", str(journal))
    assert refusal is None and binding.target_path == journal.resolve()
    assert "SYNTHETIC-OWN-ROW" in _journal_read(child)

    monkeypatch.setattr(config, "get_runtime_mode", lambda: "advanced")
    _, _, refusal = read_admission(child, "user_files", str(journal))
    assert refusal is not None and refusal.status == "blocked"
