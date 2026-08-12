"""Multi-project lease + registry + chat-id policy (v6.32.0)."""

from __future__ import annotations

import os

from ouroboros.contracts.chat_id_policy import (
    PROJECT_CHAT_ID_MIN,
    WEB_UI_CHAT_ID,
    is_a2a_chat_id,
    is_project_chat_id,
    project_chat_id,
)
from ouroboros.project_lease import (
    candidate_is_leasable,
    mark_task_project,
    running_project_ids,
    running_project_lanes,
)


def _task(project_id="", role="", tid="t1", workspace_root=""):
    task = {"id": tid, "type": "task"}
    if project_id:
        task["project_id"] = project_id
    if role:
        task["delegation_role"] = role
    if workspace_root:
        task["workspace_root"] = workspace_root
    return task


def _meta(task):
    """Production RUNNING value shape: meta dict wrapping the task."""
    return {"task": task, "worker_id": 0, "last_heartbeat_at": 1.0}


def test_running_project_lanes_counts_top_level_scoped_tasks_only():
    # Mix the PRODUCTION meta shape (workers.py RUNNING values) with bare task
    # dicts — the lane query must unwrap meta and still count both.
    running = [
        _meta(_task("alpha")),               # production shape
        _task("beta"),                       # bare task dict
        _meta(_task("", tid="plain")),       # unscoped: no lane
        _meta(_task("gamma", role="subagent")),  # swarm member: no lease of its own
        "garbage",
        None,
    ]
    # No task carries a workspace_root and no project folder map was supplied,
    # so no folder can be determined and each lane is its project's narrow key.
    assert running_project_lanes(running) == {("alpha", ""), ("beta", "")}
    # The project-WIDE ACTIVITY query (merge/remove preconditions) is a SEPARATE
    # answer, deliberately not the lease key — and it counts the SUBAGENT the
    # lane exempts (T3R-14). That exemption is a SCHEDULING rule, so a swarm
    # cannot deadlock against its own parent; a subagent still runs commands and
    # writes files, so "is anything happening in this project" must see it.
    # Exempting it here let a merge rewrite the folder mid-swarm-write.
    assert running_project_ids(running) == {"alpha", "beta", "gamma"}


def test_the_activity_query_counts_QUEUED_work_too():
    """T3R-14, stated plainly rather than left to be inferred: a PENDING task for
    this project can be assigned at ANY instant, including the one right after
    this answer is read, and a merge holds no lock against the scheduler.

    Counting it costs the owner a wait. Not counting it costs a folder rewritten
    under a task that has just started.
    """
    running = [_meta(_task("alpha"))]
    pending = [_task("beta", tid="queued"), _task("", tid="unscoped")]

    assert running_project_ids(running, pending) == {"alpha", "beta"}
    # Omitted, the answer is about running work alone — a caller that can only
    # see one collection still gets a true answer about that one.
    assert running_project_ids(running) == {"alpha"}


def test_running_project_lanes_unwraps_production_meta_shape():
    """Regression for the inert-lease bug: RUNNING.values() are meta dicts."""
    running = {"t1": _meta(_task("racer"))}.values()
    lanes = running_project_lanes(running)
    assert lanes == {("racer", "")}
    assert candidate_is_leasable(_task("racer", tid="t2"), lanes) is False


def test_candidate_is_leasable_matrix():
    leased = {("alpha", "")}
    # `{}` is the EMPTY map — "no project has a registered folder" — and is a
    # different argument from omitting it, which means "the folders are unknown"
    # and queues (I3, exercised in test_an_unreadable_folder_map_never_admits_a_
    # second_writer).
    folders: dict = {}
    # Unscoped tasks never serialize.
    assert candidate_is_leasable(_task(""), leased, folders) is True
    # A second writer for a leased project's own folder waits.
    assert candidate_is_leasable(_task("alpha"), leased, folders) is False
    # A different project proceeds in parallel.
    assert candidate_is_leasable(_task("beta"), leased, folders) is True
    # The leased project's OWN subagents must not deadlock the swarm.
    assert candidate_is_leasable(_task("alpha", role="subagent"), leased, folders) is True


def test_lane_is_keyed_on_the_folder_whenever_a_task_names_one():
    """The precondition that makes "branch off for parallel work" real.

    Two threads of ONE project in the SAME folder must still serialize; a
    thread branched off into its own git worktree gets its own lane and runs
    concurrently. Keying the lane on project_id alone made branching a promise
    the queue could not keep.
    """
    main_folder = _task("alpha", tid="t1", workspace_root="/w/alpha")
    same_folder = _task("alpha", tid="t2", workspace_root="/w/alpha")
    branched = _task("alpha", tid="t3", workspace_root="/w/alpha-thread-2")

    lanes = running_project_lanes([_meta(main_folder)])

    assert candidate_is_leasable(same_folder, lanes) is False
    assert candidate_is_leasable(branched, lanes) is True
    # ...and the branched worktree then holds a lane of its own.
    both = running_project_lanes([_meta(main_folder), _meta(branched)])
    assert len(both) == 2
    assert candidate_is_leasable(_task("alpha", tid="t4", workspace_root="/w/alpha-thread-2"), both) is False
    # The project-wide activity query still sees ONE busy project.
    assert running_project_ids([_meta(main_folder), _meta(branched)]) == {"alpha"}


def test_two_projects_on_one_folder_share_the_lane():
    """T0R2-5, a DELIBERATE reversal of T0's `(project_id, workspace_root)` key.

    Folder exclusivity used to hold only WITHIN one project: two projects
    attached to the same folder each got their own lane and ran at the same
    time, while the lane docstring and ARCHITECTURE both stated the invariant as
    "one folder is one writer lane". A named folder is now the lane key on its
    own, so the claim and the behaviour finally agree.
    """
    alpha = _task("alpha", tid="t1", workspace_root="/w/shared")
    beta = _task("beta", tid="t2", workspace_root="/w/shared")

    lanes = running_project_lanes([_meta(alpha)])

    assert lanes == {("", os.path.normcase("/w/shared"))}
    assert candidate_is_leasable(beta, lanes) is False
    # A task that names NO folder is narrower on purpose: nothing at this layer
    # may read the registry, so it can only serialize within its own project.
    # `{}` says the registry HAS no folder for it; omitting the map would say the
    # registry could not be read, which queues instead (I3).
    assert candidate_is_leasable(_task("beta", tid="t3"), lanes, {}) is True
    assert running_project_lanes([_meta(_task("beta", tid="t3"))]) == {("beta", "")}


def test_lane_reads_the_metadata_mirror_and_normalizes_the_path():
    """workspace_root rides both the task record and its metadata mirror; the
    comparison is pure normpath/normcase (no filesystem access under the queue
    lock), so a trailing slash or a redundant segment is the SAME lane."""
    mirrored = {"id": "t1", "project_id": "alpha", "metadata": {"workspace_root": "/w/alpha"}}
    noisy = _task("alpha", tid="t2", workspace_root="/w/alpha/./")

    lanes = running_project_lanes([_meta(mirrored)])
    assert lanes == {("", os.path.normcase("/w/alpha"))}
    assert candidate_is_leasable(noisy, lanes) is False


def test_a_post_hoc_scoped_task_shares_the_project_folder_lane():
    """The regression this decision closes: only the promote/room path stamps
    workspace_root. A task scoped POST-HOC through mark_task_project carries the
    project id alone, so reading the raw field split ONE folder into two lanes —
    ("", "/w/alpha") and ("alpha", "") — and let TWO top-level writers into it.
    The caller-supplied project->folder map is what closes that."""
    room_task = _task("alpha", tid="t1", workspace_root="/w/alpha")
    converted = {"id": "t2", "type": "task"}
    assert mark_task_project({}, [converted], "t2", "alpha") is True
    assert "workspace_root" not in converted        # the SSOT stamps only the id

    folders = {"alpha": "/w/alpha"}
    lanes = running_project_lanes([_meta(room_task)], folders)
    assert lanes == {("", os.path.normcase("/w/alpha"))}
    # The absent workspace resolves to the project's registered working_dir.
    assert candidate_is_leasable(converted, lanes, folders) is False
    # ...and it would NOT have, read against an EMPTY map: this is the whole
    # point. `{}` means "the registry says no folder", which is an answer the
    # narrow key may act on; a MISSING map means "the registry could not be read"
    # and is the separate, conservative case in
    # test_an_unreadable_folder_map_never_admits_a_second_writer (I3).
    assert candidate_is_leasable(converted, lanes, {}) is True
    # A thread branched into its OWN worktree still runs concurrently.
    branched = _task("alpha", tid="t3", workspace_root="/w/alpha-thread-2")
    assert candidate_is_leasable(branched, lanes, folders) is True


def test_an_unknown_project_folder_keys_on_the_project_alone():
    """When neither the task record nor the registry can name the folder, the
    lane is (project_id, "") — honestly narrower than a folder lane rather than
    a wildcard pretending to be one.

    T0 keyed this case on a WILDCARD conflicting with every lane of the same
    project. Under folder-keyed lanes (T0R2-5) that had nothing left to match: a
    folder lane does not carry the project it belongs to, so "every lane of this
    project" is not a set this module can name. The map above is the mechanism
    that actually closes the hole; the wildcard was deleted rather than kept as a
    name promising a behaviour it no longer had.
    """
    room_task = _task("alpha", tid="t1", workspace_root="/w/alpha")
    unknown = {"id": "t2", "type": "task", "project_id": "alpha"}

    # No folder map at all (unreadable registry / file-less project).
    lanes = running_project_lanes([_meta(room_task)], {})
    assert candidate_is_leasable(unknown, lanes, {}) is True
    held_narrow = running_project_lanes([_meta(unknown)], {})
    assert held_narrow == {("alpha", "")}
    # Two placeless tasks of ONE project still serialize against each other.
    assert candidate_is_leasable(
        {"id": "t3", "type": "task", "project_id": "alpha"}, held_narrow, {}
    ) is False
    # A different project is untouched.
    assert candidate_is_leasable(_task("beta", tid="t9"), held_narrow, {}) is True


def test_the_pin_uses_the_same_folder_map_the_admission_check_used():
    """CROSS-STREAM: pinning without the map admits the second writer itself.

    T0 gave the lane a caller-supplied project->folder map; T3 pinned the lane at
    the RUNNING transition. Composed naively, a task that names no folder is
    ADMITTED as ("", registered_folder) and then PINNED as (project_id, "") — so
    the folder it is writing in reads as unheld to the very next candidate, and
    the pin, whose entire purpose is to stop a live writer drifting out of its
    lane, becomes the thing that drifts it.
    """
    from ouroboros.project_lease import LANE_PIN_FIELD, pin_task_lane

    folders = {"alpha": "/w/alpha"}
    placeless = {"id": "t1", "type": "task", "project_id": "alpha"}

    assert pin_task_lane(placeless, folders) == ("", os.path.normcase("/w/alpha"))
    assert placeless[LANE_PIN_FIELD] == ["", os.path.normcase("/w/alpha")]

    lanes = running_project_lanes([_meta(placeless)], folders)
    assert lanes == {("", os.path.normcase("/w/alpha"))}
    assert candidate_is_leasable(
        _task("alpha", tid="t2", workspace_root="/w/alpha"), lanes, folders
    ) is False


def test_workspace_none_task_still_serializes_against_its_project_folder():
    """`workspace="none"` is an explicit opt-out that yields NO workspace_root
    (workspace_admission.resolve_room_workspace). "I write nowhere" is not a
    claim the lease can verify, so it queues behind the folder's writer."""
    room_task = _task("alpha", tid="t1", workspace_root="/w/alpha")
    opted_out = {"id": "t2", "type": "task", "project_id": "alpha", "workspace": "none"}
    folders = {"alpha": "/w/alpha"}

    lanes = running_project_lanes([_meta(room_task)], folders)
    assert candidate_is_leasable(opted_out, lanes, folders) is False


def test_case_and_symlink_normalization_boundaries():
    """This module never touches the filesystem, so SYMLINK resolution is a
    RECORD-WRITE-time job: workspace_admission resolves a task's workspace_root
    and projects_registry resolves a project's working_dir before either is
    stored. Both therefore arrive here already realpath'd."""
    room = _task("alpha", tid="t1", workspace_root="/w/Alpha")
    lanes = running_project_lanes([_meta(room)])
    # Spelling equality follows the PLATFORM's filesystem, not `normcase` alone:
    # `normcase` lowercases on win32 only, and darwin's default filesystem is
    # case-insensitive too, so the lane casefolds there as well (T0R2-4).
    same_case = _task("alpha", tid="t2", workspace_root="/w/Alpha/")
    assert candidate_is_leasable(same_case, lanes) is False
    other_case = _task("alpha", tid="t3", workspace_root="/w/alpha")
    # IMPORTED from the PLATFORM LAYER, which owns the fact, not re-listed here:
    # a second copy means a platform added there without the test's makes this
    # assertion pass vacuously (I18).
    from ouroboros.platform_layer import PATH_CASE_INSENSITIVE

    case_insensitive = PATH_CASE_INSENSITIVE
    assert candidate_is_leasable(other_case, lanes) is not case_insensitive
    # An UNRESOLVED symlink spelling is a different string: the lease cannot
    # resolve it (no FS access under the queue lock), which is exactly why the
    # writers canonicalize before storing.
    via_link = _task("alpha", tid="t4", workspace_root="/link/to/alpha")
    assert candidate_is_leasable(via_link, lanes) is True


def test_lane_key_shape_is_validated_before_the_unscoped_short_circuit():
    """A caller passing bare project ids must be told IMMEDIATELY. Checking the
    shape after the unscoped short-circuit meant the misuse stayed silent for
    every unscoped candidate and surfaced only once a project task happened to
    be considered — by which time two writers could already be running."""
    import pytest

    with pytest.raises(TypeError, match="running_project_lanes"):
        candidate_is_leasable(_task(""), {"alpha"})
    with pytest.raises(TypeError):
        candidate_is_leasable(_task("alpha"), {("alpha", "", "extra")})


def test_project_working_dirs_feeds_the_lane_resolver(tmp_path):
    """The map the supervisor hands the lease comes from the registry, and the
    registry canonicalizes working_dir at WRITE time so the lease's pure
    comparison meets an already-resolved path."""
    from ouroboros.projects_registry import create_project, project_working_dirs

    folder = tmp_path / "alpha"
    folder.mkdir()
    create_project(tmp_path, "alpha", working_dir=str(folder))
    fileless = create_project(tmp_path, "notes")

    folders = project_working_dirs(tmp_path)
    assert folders["alpha"] == str(folder.resolve())
    assert "notes" not in folders            # file-less -> wildcard, not a lane
    assert fileless["working_dir"] == ""

    # A LEGACY row stored an unresolved spelling. Against a task's already
    # realpath'd workspace_root that is a different string — a second concrete
    # lane, exactly the split this map exists to close — so the read
    # canonicalizes too, not only the write.
    from ouroboros.projects_registry import _registry_path
    from ouroboros.utils import atomic_write_json, read_json_dict

    link = tmp_path / "alpha-link"
    link.symlink_to(folder, target_is_directory=True)
    data = read_json_dict(_registry_path(tmp_path))
    for entry in data["projects"]:
        if entry.get("id") == "alpha":
            entry["working_dir"] = str(link)
    atomic_write_json(_registry_path(tmp_path), data)
    assert project_working_dirs(tmp_path)["alpha"] == str(folder.resolve())

    room = _task("alpha", tid="t1", workspace_root=str(folder))
    converted = {"id": "t2", "type": "task", "project_id": "alpha"}
    lanes = running_project_lanes([_meta(room)], folders)
    assert candidate_is_leasable(converted, lanes, folders) is False


def test_lane_folds_case_on_a_case_insensitive_filesystem():
    """T0R2-4: `normcase` lowercases on win32 ONLY, so on macOS `/Users/x/Repo`
    and `/Users/x/repo` — the SAME folder — produced two lanes and admitted two
    writers onto it. The docstring claimed a case-insensitive lane it did not
    deliver; now it delivers one wherever the platform's filesystem is."""
    upper = _task("alpha", tid="t1", workspace_root="/W/Alpha")
    lower = _task("beta", tid="t2", workspace_root="/w/alpha")

    lanes = running_project_lanes([_meta(upper)])

    from ouroboros.platform_layer import PATH_CASE_INSENSITIVE

    if PATH_CASE_INSENSITIVE:  # I18: one copy of the platform fact, in the layer
        assert candidate_is_leasable(lower, lanes) is False
    else:
        # A case-SENSITIVE filesystem genuinely has two folders here, and
        # folding them together would serialize two unrelated writers.
        assert candidate_is_leasable(lower, lanes) is True


def test_the_case_fold_fact_lives_in_the_platform_layer():
    """P3: "which platforms fold path case" is a PLATFORM fact.

    It was answered inside `project_lease.normalize_workspace_root` by reading
    `sys.platform` against a module-local tuple, i.e. platform-specific behaviour
    outside the one module that owns platform facts — while `platform_layer`
    already exported `IS_WINDOWS`/`IS_MACOS` next door. Behaviour is unchanged;
    the SEAM moves, so a platform added to the fact is added once.
    """
    import inspect

    from ouroboros import project_lease
    from ouroboros.platform_layer import (
        IS_MACOS,
        IS_WINDOWS,
        PATH_CASE_INSENSITIVE,
        casefold_path,
    )

    assert PATH_CASE_INSENSITIVE is (IS_MACOS or IS_WINDOWS)
    # The lane consumes the seam and no longer decides the platform question.
    # Read from the AST, not from the text: prose EXPLAINING why `sys.platform`
    # is not consulted here would otherwise fail this assertion.
    import ast

    tree = ast.parse(inspect.getsource(project_lease))
    reads_sys_platform = any(
        isinstance(node, ast.Attribute)
        and node.attr == "platform"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
        for node in ast.walk(tree)
    )
    assert not reads_sys_platform
    assert not hasattr(project_lease, "_CASE_INSENSITIVE_PLATFORMS")
    assert "casefold_path" in inspect.getsource(project_lease.normalize_workspace_root)
    # ...and the answer it produces is byte-identical to the old expression.
    for raw in ("/w/Alpha", "/W/ALPHA/", "/w/alpha", "relative/Path"):
        expected = casefold_path(os.path.normpath(raw))
        assert project_lease.normalize_workspace_root(raw) == expected


def test_a_running_lane_is_pinned_and_cannot_drift():
    """T0R2-7: a live writer must not be moved off the folder it is writing in.

    The lane used to be recomputed from the task record on every assignment
    pass, so a mid-run mutation (the post-hoc project conversion is the live
    one) released the lane the task actually held and let a second writer into
    the same folder.
    """
    from ouroboros.project_lease import LANE_PIN_FIELD, pin_task_lane

    running = _task("alpha", tid="t1", workspace_root="/w/alpha")
    assert pin_task_lane(running) == ("", os.path.normcase("/w/alpha"))
    assert running[LANE_PIN_FIELD] == ["", os.path.normcase("/w/alpha")]

    # Something edits the live record underneath the writer.
    running["workspace_root"] = "/w/somewhere-else"
    running["project_id"] = "beta"

    lanes = running_project_lanes([_meta(running)])
    assert lanes == {("", os.path.normcase("/w/alpha"))}
    assert candidate_is_leasable(
        _task("alpha", tid="t2", workspace_root="/w/alpha"), lanes
    ) is False
    # The pin is write-once: re-pinning never overwrites a held lane.
    assert pin_task_lane(running) == ("", os.path.normcase("/w/alpha"))


def test_post_hoc_conversion_pins_the_lane_it_acquires():
    """A RUNNING task that was never a lane occupant ACQUIRES one when the owner
    converts it into a project. That is a deliberate take, not drift, so it pins
    on the spot — otherwise a later registry edit could move a live writer.

    Since P5(ii) the task that acquires a lane here is the PLACELESS one. A task
    that already NAMES a folder is a lane occupant before any conversion, because
    naming a folder is what writing in it means.
    """
    from ouroboros.project_lease import LANE_PIN_FIELD, mark_task_project

    placeless = {"id": "t1"}
    running = {"t1": {"task": placeless}}
    folders = {"alpha": "/w/alpha"}
    assert running_project_lanes(running.values(), folders) == set()

    assert mark_task_project(running, [], "t1", "alpha", folders) is True

    assert placeless[LANE_PIN_FIELD] == ["", os.path.normcase("/w/alpha")]
    assert running_project_lanes(running.values(), folders) == {
        ("", os.path.normcase("/w/alpha"))
    }

    # The folder-naming task holds that folder's lane with no project at all, so
    # the conversion widens what serializes WITH it rather than granting the lane.
    named = {"id": "t2", "workspace_root": "/w/beta"}
    assert running_project_lanes([{"task": named}]) == {("", os.path.normcase("/w/beta"))}


def test_project_chat_id_policy():
    assert is_project_chat_id(WEB_UI_CHAT_ID) is False
    assert is_project_chat_id(-5) is False
    cid = project_chat_id("my-game")
    assert cid >= PROJECT_CHAT_ID_MIN
    assert is_project_chat_id(cid) is True
    assert is_a2a_chat_id(cid) is False
    # Deterministic and id-sensitive.
    assert project_chat_id("my-game") == cid
    assert project_chat_id("other") != cid
    # Empty scope falls back to the main chat.
    assert project_chat_id("") == WEB_UI_CHAT_ID


def test_registry_create_idempotent_and_summary(tmp_path):
    from ouroboros.projects_registry import (
        create_project,
        get_project,
        list_projects,
        projects_summary,
    )

    entry = create_project(tmp_path, "racer", name="Cyber Racer")
    assert entry["id"] == "racer"
    assert "status" not in entry  # statuses removed (v6.33.0)
    assert entry["chat_id"] == project_chat_id("racer")

    again = create_project(tmp_path, "racer", name="ignored on existing")
    assert again["name"] == "Cyber Racer"
    assert len(list_projects(tmp_path)) == 1

    rows = projects_summary(tmp_path)
    assert rows and rows[0]["id"] == "racer" and rows[0]["chat_id"] == entry["chat_id"]
    assert "status" not in rows[0]
    assert get_project(tmp_path, "missing") is None


def test_registry_reconcile_registers_existing_stores_never_prunes(tmp_path):
    from ouroboros.projects_registry import create_project, list_projects, reconcile_projects

    create_project(tmp_path, "kept")
    (tmp_path / "projects" / "legacy-store" / "knowledge").mkdir(parents=True)

    added = reconcile_projects(tmp_path)

    assert added == 1
    ids = {p["id"] for p in list_projects(tmp_path)}
    assert ids == {"kept", "legacy-store"}
    # Second run is a no-op (idempotent) and nothing is pruned.
    assert reconcile_projects(tmp_path) == 0
    assert {p["id"] for p in list_projects(tmp_path)} == ids


def test_journal_and_workpad_roundtrip(tmp_path, monkeypatch):
    import types

    # Scope the project store to tmp_path WITHOUT importlib.reload(config): a
    # reload permanently rebinds ouroboros.config.DATA_DIR for the rest of the
    # pytest process (monkeypatch restores only the env var, not the reloaded
    # module), polluting later tests. project_facts reads config.DATA_DIR at call
    # time, so monkeypatch.setattr (auto-restored) is sufficient and isolated.
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    from ouroboros.tools import project_journal as pj

    ctx = types.SimpleNamespace(project_id="racer", task_id="t-9", drive_root=tmp_path)
    tools = {t.name: t for t in pj.get_tools()}

    out = tools["journal_write"].handler(ctx, kind="start", text="Bootstrapping the racer")
    assert out.startswith("OK:")
    out = tools["journal_write"].handler(ctx, kind="bogus", text="x")
    assert "TOOL_ARG_ERROR" in out
    listing = tools["journal_read"].handler(ctx)
    assert "Bootstrapping the racer" in listing and "START" in listing

    assert tools["workpad_write"].handler(ctx, content="## plan\n- wheels").startswith("OK:")
    assert "wheels" in tools["workpad_read"].handler(ctx)

    digest = pj.journal_tail_digest("racer")
    assert "Bootstrapping the racer" in digest

    # Unscoped ctx without explicit id refuses honestly.
    bare = types.SimpleNamespace(project_id="", task_id="t", drive_root=tmp_path)
    assert "no project scope" in tools["journal_write"].handler(bare, kind="note", text="x")


def test_a_candidate_is_compared_by_what_it_DESCRIBES_never_by_a_stale_pin():
    """Only OCCUPANCY reads the pin. A pending candidate holds nothing.

    A crash retry re-enqueues the very dict the pin was stamped on, so if a
    candidate were compared by its pin it would be matched against a lane it no
    longer holds — over-serializing at best, and under-serializing the moment
    the record legitimately named a different folder.
    """
    from ouroboros.project_lease import pin_task_lane

    retried = _task("alpha", tid="t1", workspace_root="/w/alpha")
    pin_task_lane(retried)
    # The retry will write in a DIFFERENT folder than the run that was pinned.
    retried["workspace_root"] = "/w/beta"

    busy_beta = running_project_lanes([_meta(_task("alpha", tid="t9", workspace_root="/w/beta"))])

    assert candidate_is_leasable(retried, busy_beta) is False
    stale_alpha = running_project_lanes([_meta(_task("alpha", tid="t9", workspace_root="/w/alpha"))])
    assert candidate_is_leasable(retried, stale_alpha) is True


def test_a_projectless_task_naming_a_reserved_folder_is_not_leasable():
    """P5(ii): the reservation was only half a reservation.

    With `reserved_folder_lane(folder)` HELD — a merge-back rewriting it, a
    checkout removal deleting it — a project-scoped candidate for that folder was
    correctly refused while a PROJECTLESS candidate naming the SAME folder was
    admitted straight into it, because `_is_lane_occupant` short-circuited on
    `bool(project_id)`. Reproduced 3/3 against the merged tree.
    """
    from ouroboros.project_lease import normalize_workspace_root, reserved_folder_lane

    folder = "/tmp/owner_folder"
    with reserved_folder_lane(folder) as key:
        lanes = running_project_lanes([], {})
        assert key == ("", normalize_workspace_root(folder))
        assert key in lanes

        scoped = {"id": "t1", "project_id": "alpha", "workspace_root": folder}
        projectless = {"id": "t2", "workspace_root": folder}
        subagent = {
            "id": "t3", "project_id": "alpha", "workspace_root": folder,
            "delegation_role": "subagent",
        }
        elsewhere = {"id": "t4", "workspace_root": "/tmp/other_folder"}
        placeless = {"id": "t5"}

        assert candidate_is_leasable(scoped, lanes, {}) is False
        assert candidate_is_leasable(projectless, lanes, {}) is False
        # A swarm member must still be admitted: the parent task IS the writer and
        # a task cannot be made to wait for its own children.
        assert candidate_is_leasable(subagent, lanes, {}) is True
        # Nothing that is not in THAT folder becomes slower.
        assert candidate_is_leasable(elsewhere, lanes, {}) is True
        assert candidate_is_leasable(placeless, lanes, {}) is True
    # The reservation is released, so nothing is refused for good.
    assert running_project_lanes([], {}) == set()
    assert candidate_is_leasable({"id": "t2", "workspace_root": "/tmp/owner_folder"}, set(), {}) is True


def test_two_projectless_writers_in_one_folder_serialize():
    """The same fact from the RUNNING side: a projectless task carrying a
    workspace_root holds that folder's lane, so a second writer for the folder
    queues instead of entering it. Two folders stay parallel."""
    first = {"id": "t1", "workspace_root": "/w/shared"}
    lanes = running_project_lanes([{"task": first}], {})
    assert lanes == {("", os.path.normcase("/w/shared"))}
    assert candidate_is_leasable({"id": "t2", "workspace_root": "/w/shared"}, lanes, {}) is False
    assert candidate_is_leasable({"id": "t3", "workspace_root": "/w/other"}, lanes, {}) is True
    # An unscoped, placeless task is untouched: it names nothing, so it has no lane.
    assert candidate_is_leasable({"id": "t4"}, lanes, {}) is True
