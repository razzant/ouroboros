"""A scheduled ``input_files`` selection reaches what each child actually receives (#2593).

The parent's real ``schedule_subagent`` event goes through supervisor admission, the durable
queue snapshot and its restore; the worker's own copy of that payload then composes a native
child's real first input and context and reads through the real registry, or drives a session
child's real bootstrap into the engine double. Only the engine and the worker process boundary
(a JSON copy) are synthetic: the saved row alone proved nothing in the original gap.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys

import pytest

from ouroboros import delegate_custody as custody
from tests.test_delegate_documents_admission import chain  # noqa: F401
from tests.test_doc_context import _make_env_and_memory
from tests.test_scheduled_effort_intent import _admit, _restore, runtime  # noqa: F401
from tests._delegated_transport_shared import _owned_gateway_uses_each_test_transport  # noqa: F401

BRIEF, BINARY = b"parent brief\n", b"\xff\x00binary\n"
SELECTION = [{"root": "task_drive", "path": "brief.md"}, {"root": "task_drive", "path": "fixture.bin"}]


def _parent_files(rt):
    drive = rt.drive / "task_drives" / "parent"
    drive.mkdir(parents=True, exist_ok=True)
    (drive / "brief.md").write_bytes(BRIEF)
    (drive / "fixture.bin").write_bytes(BINARY)
    return drive


def _api_row(monkeypatch):
    from tests._shared import configure_test_subagent

    return configure_test_subagent(monkeypatch, kind="api_model", target="openai/effort-fixture-model")


def _schedule(chain, sid, **params):  # noqa: F811
    from ouroboros.tools.control_scheduling import _schedule_task

    _rt, parent, _engine, _sid = chain
    parent.pending_events.clear() if hasattr(parent.pending_events, "clear") else None
    reply = _schedule_task(parent, subagent_id=sid, objective="Compare the inputs", expected_output="A comparison",
                           **{"memory_mode": "empty", **params})
    assert len(parent.pending_events) == 1, reply
    return reply, parent.pending_events[0]


def _child_context(rt, payload):
    """The tool context the worker gives this child, from its own copy of the payload (``agent.py``)."""
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.project_facts import resolve_project_id
    from ouroboros.tools.registry import ToolContext

    child = ToolContext(repo_dir=rt.repo, drive_root=pathlib.Path(payload.get("child_drive_root") or payload["drive_root"]),
                        task_constraint=TaskConstraint(mode="local_readonly_subagent"),
                        task_contract=payload.get("task_contract") or {})  # as agent.py builds it
    child.task_id, child.project_id = payload["id"], resolve_project_id(payload)
    child.workspace_root = (pathlib.Path(payload["workspace_root"]).resolve(strict=False)
                            if payload.get("workspace_root") else None)
    child.workspace_mode = str(payload.get("workspace_mode") or "")
    child.task_metadata = {**payload["metadata"], **{key: payload[key] for key in (
        "root_task_id", "parent_task_id", "drive_root", "child_drive_root", "budget_drive_root") if payload.get(key)}}
    return child


def _session_first_start(chain, monkeypatch, payload):  # noqa: F811
    """The worker's first start of the admitted child, from its own copy of the payload."""
    from ouroboros import claudexor_daemon
    from ouroboros.subagent_bootstrap import bootstrap_before_context
    from tests.test_delegate_documents import _HARNESS

    rt, _parent, engine, _sid = chain
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: engine)  # admission never touched it
    custody._CUSTODY.clear()  # a fresh worker process
    child = _child_context(rt, payload)
    return child, bootstrap_before_context(child, payload, _HARNESS)


def _native_read(rt, payload, monkeypatch, **arguments):
    from ouroboros import safety
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    child = _child_context(rt, payload)
    registry = ToolRegistry(repo_dir=rt.repo, drive_root=child.drive_root)
    registry.set_context(child)
    return registry.execute("read_file", arguments)


def _first_input(tmp_path, payload):
    """A native child's composed first input and dynamic context, as its worker composes them."""
    from ouroboros import context

    env, memory = _make_env_and_memory(tmp_path / "first-input")
    core = context._capture_context_core(env, memory, payload, None, None)
    return json.loads(core.user_content_json), core.dynamic_text


def _text_of(content):
    return content if isinstance(content, str) else "\n".join(str(block.get("text") or "") for block in content)


@pytest.mark.parametrize(("sources", "memory_mode"), [("shared", "forked"), ("declared", "empty")])
def test_a_native_child_receives_and_reads_the_retained_copies_through_the_real_chain(
        chain, monkeypatch, tmp_path, memory_mode, sources):  # noqa: F811
    from ouroboros.artifacts import collect_task_artifact_records
    from ouroboros.task_results import load_task_result

    rt, _parent, _engine, _sid = chain
    drive = _parent_files(rt)
    reply, event = _schedule(chain, _api_row(monkeypatch), memory_mode=memory_mode, input_sources=sources,
                             input_files=SELECTION)
    assert event["configured_subagent"]["route"]["kind"] == "api_model"
    rows = event["prepared_input_files"]
    assert [row["status"] for row in rows] == ["prepared", "prepared"] and '"status": "prepared"' in reply
    home = rt.drive / "task_results" / "artifacts" / "parent" / "source_handles" / "delegated_readonly_inputs"
    assert all(pathlib.Path(row["local"]).parents[3] == home / f"scheduled-{event['task_id']}" for row in rows)

    task = _admit(rt, event)
    assert task["prepared_input_files"] == task["metadata"]["prepared_input_files"] == rows
    assert load_task_result(rt.drive, task["id"])["prepared_input_files"] == rows
    payload = _restore(rt)  # the durable snapshot, restored, as the worker's own copy
    assert payload["prepared_input_files"] == payload["metadata"]["prepared_input_files"] == rows

    # The original changes and its drive retires: the child still names and reads what was selected.
    (drive / "brief.md").write_bytes(b"edited after queueing\n")
    shutil.rmtree(drive)
    content, dynamic = _first_input(tmp_path, payload)
    first = _text_of(content)
    assert "[SELECTED INPUT FILES]" in first and "read_file(path=<its absolute path below>)" in first
    assert first.index("[EXPECTED_OUTPUT]") < first.index("[SELECTED INPUT FILES]") < first.index("[HANDOFF CONTRACT]")
    for row in rows:
        assert f"- task_drive:{row['path']} -> {row['local']} (sha256 {row['sha256']}, {row['size']} bytes)" in first
    assert ("explicitly selected input files (input_files)" in dynamic) is (sources == "declared")
    text = _native_read(rt, payload, monkeypatch, path=rows[0]["local"])
    assert "parent brief" in text and "edited" not in text, text
    # The block's statement about binary bytes is the reader's actual behavior.
    assert "READ_FILE_ERROR" in _native_read(rt, payload, monkeypatch, path=rows[1]["local"])
    assert pathlib.Path(rows[1]["local"]).read_bytes() == BINARY
    # Retained inputs are never offered as anyone's deliverable.
    for task_id in ("parent", task["id"]):
        assert not [row for row in collect_task_artifact_records(rt.drive, task_id)
                    if "delegated_readonly_inputs" in row["path"]]


@pytest.mark.parametrize("folderless,project_id", [(False, ""), (True, ""), (True, "prj-inputs")],
                         ids=["repository", "folderless", "folderless-project"])
def test_a_session_childs_selection_crosses_admission_snapshot_and_bootstrap_to_the_sender(
        chain, monkeypatch, folderless, project_id):  # noqa: F811
    rt, parent, engine, sid = chain
    drive = _parent_files(rt)
    if folderless:
        parent.project_id = project_id
        parent.task_metadata["resource_intent"] = {"kind": "explicit_none"}
    reply, event = _schedule(chain, sid, input_sources="declared", input_files=SELECTION[:1])
    if folderless:
        assert "agent_session selected source: folderless selected files" in reply
    else:
        assert "agent_session selected source: inherited repository/folder" in reply
    assert "supervisor: child capabilities — profile=" in reply
    assert "Prompt-only paths are not prepared" in reply
    assert "preparation is not a read" in reply
    [row] = event["prepared_input_files"]
    digest = hashlib.sha256(BRIEF).hexdigest()
    assert row["sha256"] == digest
    task = _admit(rt, event)
    assert "[SELECTED INPUT FILES]" in task["text"] and row["local"] in task["text"], "the nanny can read them too"
    if folderless:
        assert event.get("resource_intent") == {"kind": "explicit_none"}
        assert task["metadata"].get("resource_intent") == {"kind": "explicit_none"}
        from ouroboros.task_results import load_task_result
        assert load_task_result(rt.drive, task["id"])["resource_intent"] == {"kind": "explicit_none"}
    payload = _restore(rt)
    assert payload["prepared_input_files"] == [row]
    (drive / "brief.md").write_bytes(b"edited after queueing\n")

    if folderless:
        from ouroboros import delegate_readonly_inputs as inputs

        def unexpected_lineage(_base):
            raise AssertionError("prepared selection must skip automatic lineage enumeration")
            yield

        monkeypatch.setattr(inputs, "_input_files", unexpected_lineage)

    child, wake = _session_first_start(chain, monkeypatch, payload)
    receipt = json.loads(wake)
    assert receipt["status"] == "configured_session_started", receipt
    (_, _, body), = engine.of("run")
    assert body["prompt"] == child._configured_actor_bootstrap["canonical_work_order"]
    assert "SELECTED INPUT FILES" in body["prompt"] and f"- task_drive:brief.md (sha256 {digest}" in body["prompt"]
    assert "edited" not in body["prompt"]
    [started] = receipt["startup"]["input_files"]
    assert started["retained"] == row["local"] and started["source"] == row["source"] and started["sha256"] == digest
    assert started["local"] in body["instructions"] and "Preparation is not a read" in body["instructions"]
    assert pathlib.Path(started["local"]).read_bytes() == BRIEF
    if folderless:
        scope = pathlib.Path(body["scope"]["root"])
        assert started["carrier"] == "scope" and scope in pathlib.Path(started["local"]).parents
        manifest = json.loads((scope / "inputs.json").read_text(encoding="utf-8"))
        assert sum(item["sha256"] == digest for item in manifest) == 1, "one copy of the bytes, no lineage recopy"
        assert not [item for item in manifest if "delegated_readonly_inputs" in item["source"]]
        assert [item for item in manifest if "selected_path" not in item] == [], "the selection replaced lineage"
        if sys.platform == "darwin":  # #2945: the first start clones the retained copy into its scope, not rewritten
            from ouroboros.artifact_storage import _may_share

            assert _may_share(pathlib.Path(started["retained"])) is True and _may_share(pathlib.Path(started["local"])) is True
    else:
        assert body["scope"]["root"] == str(rt.repo) and "execution" not in body, "the repository stays scope and cwd"
        assert started["carrier"] == "outside_scope_path" and started["local"] == row["local"], "no second copy"
    request, = [item for item in custody._iter_rows(custody.event_log_path(rt.drive))
                if item.get("type") == custody.START_REQUESTED and item.get("task_id") == payload["id"]]
    assert request["input_files"] == receipt["startup"]["input_files"]


def test_a_session_workspace_root_preserves_explicit_resource_and_leaf_scope(chain, monkeypatch):  # noqa: F811
    rt, parent, engine, sid = chain
    parent.task_metadata["resource_intent"] = {"kind": "explicit_none"}
    _parent_files(rt)
    reply, event = _schedule(chain, sid, input_sources="declared", workspace_root=str(rt.repo),
                             input_files=SELECTION[:1])
    from ouroboros.task_results import load_task_result
    requested = load_task_result(rt.drive, event["task_id"])
    assert requested["status"] == "requested"
    assert requested["resource_intent"] == event["resource_intent"]
    assert "agent_session selected source: workspace_root=" in reply
    assert "Prompt-only paths are not prepared" in reply
    assert "supervisor read roots are not a leaf-access guarantee" in reply
    assert event["resource_intent"] == {"kind": "explicit_resource", "root": str(rt.repo)}
    [row] = event["prepared_input_files"]
    task = _admit(rt, event)
    assert task["metadata"]["resource_intent"] == event["resource_intent"]
    payload = _restore(rt)
    child, wake = _session_first_start(chain, monkeypatch, payload)
    receipt = json.loads(wake)
    assert receipt["status"] == "configured_session_started", receipt
    (_, _, body), = engine.of("run")
    assert body["scope"]["root"] == str(rt.repo)
    [started] = receipt["startup"]["input_files"]
    assert started["carrier"] == "outside_scope_path" and started["local"] == row["local"]
    assert child.workspace_root == pathlib.Path(rt.repo).resolve()


def test_a_tampered_retained_copy_refuses_the_first_start(chain, monkeypatch):  # noqa: F811
    rt, _parent, engine, sid = chain
    _parent_files(rt)
    _reply, event = _schedule(chain, sid, input_files=SELECTION[:1])
    _admit(rt, event)
    payload = _restore(rt)
    pathlib.Path(event["prepared_input_files"][0]["local"]).write_bytes(b"tampered\n")
    child, wake = _session_first_start(chain, monkeypatch, payload)
    assert engine.of("run") == [] and not wake, wake  # a zero-spend terminal, its typed reason kept for the parent
    refusal = child._configured_startup_refusal
    assert refusal["reason"] == "input_files_unavailable", refusal
    assert [row["status"] for row in refusal["input_files"]] == ["changed"]


def test_losing_resource_intent_reproduces_a_folderless_leaf_scope_mismatch(chain, monkeypatch):  # noqa: F811
    """Negative control at the real bootstrap: saved input bytes alone did not prove the scope."""
    from ouroboros import subagents

    rt, parent, engine, sid = chain
    _parent_files(rt)
    parent.task_metadata["resource_intent"] = {"kind": "explicit_none"}
    parent.project_id = ""
    monkeypatch.setattr(subagents, "SUBAGENT_INTENT_FIELDS",
                        tuple(key for key in subagents.SUBAGENT_INTENT_FIELDS if key != "resource_intent"))
    reply, event = _schedule(chain, sid, input_sources="declared", input_files=SELECTION[:1])
    assert "selected source: folderless" in reply
    task = _admit(rt, event)
    assert not task["metadata"].get("resource_intent")
    _, wake = _session_first_start(chain, monkeypatch, _restore(rt))
    (_, _, body), = engine.of("run")
    assert body["scope"]["root"] == str(rt.repo), "the missing intent silently changes the scope"
    assert json.loads(wake)["startup"]["input_files"][0]["carrier"] == "outside_scope_path"


def test_prose_only_worktrees_are_not_selected_or_claimed_prepared(chain, monkeypatch):  # noqa: F811
    rt, parent, engine, sid = chain
    parent.project_id = ""
    parent.task_metadata["resource_intent"] = {"kind": "explicit_none"}
    drive = _parent_files(rt)
    worktrees = [rt.repo.parent / f"external-{n}" for n in range(3)]
    for root in worktrees:
        root.mkdir()
        (root / "source.md").write_text("not a selected source", encoding="utf-8")
    reply, event = _schedule(chain, sid, context="Inspect these worktrees: " + ", ".join(map(str, worktrees)))
    assert "Prompt-only paths are not prepared" in reply
    assert "prepared_input_files" not in event
    _admit(rt, event)
    _, wake = _session_first_start(chain, monkeypatch, _restore(rt))
    assert json.loads(wake)["status"] == "configured_session_started"
    (_, _, body), = engine.of("run")
    manifest = json.loads((pathlib.Path(body["scope"]["root"]) / "inputs.json").read_text())
    assert any(item["source"] == str(drive / "brief.md") for item in manifest)
    assert not any(str(root) in item["source"] for item in manifest for root in worktrees)


def test_without_the_intent_registry_entry_the_selection_never_reaches_a_child(chain, monkeypatch, tmp_path):  # noqa: F811
    """Negative control: the registry is the one carrier. The producer's own row still holds the
    selection, which is exactly what let a saved-row test pass over the lost payload."""
    from ouroboros import subagents
    from ouroboros.task_results import load_task_result

    rt, _parent, _engine, _sid = chain
    _parent_files(rt)
    monkeypatch.setattr(subagents, "SUBAGENT_INTENT_FIELDS",
                        tuple(key for key in subagents.SUBAGENT_INTENT_FIELDS if key != "prepared_input_files"))
    _reply, event = _schedule(chain, _api_row(monkeypatch), input_files=SELECTION[:1])
    requested = load_task_result(rt.drive, event["task_id"])
    task = _admit(rt, event)
    assert "prepared_input_files" not in task and "prepared_input_files" not in task["metadata"]
    assert "SELECTED INPUT FILES" not in task["text"]
    payload = _restore(rt)
    assert "prepared_input_files" not in payload
    assert "[SELECTED INPUT FILES]" not in _text_of(_first_input(tmp_path, payload)[0])
    assert requested["prepared_input_files"] == event["prepared_input_files"]


@pytest.mark.parametrize("consumer", ["api", "session"])
def test_an_explicit_empty_selection_reports_an_empty_receipt_and_changes_nothing_else(
        chain, monkeypatch, tmp_path, consumer):  # noqa: F811
    from ouroboros.subagent_work_order import compile_external_work_order

    rt, _parent, engine, sid = chain
    sid = _api_row(monkeypatch) if consumer == "api" else sid
    omitted_reply, omitted = _schedule(chain, sid)
    assert "input_files" not in omitted_reply and "prepared_input_files" not in omitted
    empty_reply, empty = _schedule(chain, sid, input_files=[])
    assert "input_files (retained before queueing; prepared is not read): []" in empty_reply
    assert empty["prepared_input_files"] == []
    null_reply, null = _schedule(chain, sid, input_files=None)  # a harness's spelling of omission
    assert "input_files" not in null_reply and "prepared_input_files" not in null
    assert not (rt.drive / "task_results" / "artifacts" / "parent" / "source_handles" / "delegated_readonly_inputs").exists()
    task = _admit(rt, empty)
    assert task["prepared_input_files"] == [] and "SELECTED INPUT FILES" not in task["text"]
    assert "SELECTED INPUT FILES" not in compile_external_work_order(task)
    payload = _restore(rt)
    assert payload["prepared_input_files"] == []
    assert "SELECTED INPUT FILES" not in _text_of(_first_input(tmp_path, payload)[0])
    if consumer == "session":
        receipt = json.loads(_session_first_start(chain, monkeypatch, payload)[1])
        assert receipt["status"] == "configured_session_started", receipt
        (_, _, body), = engine.of("run")
        assert "SELECTED INPUT FILES" not in body["prompt"] + body["instructions"]
        assert "input_files" not in receipt["startup"], "nothing was selected for this start to prepare"


def test_a_parent_on_its_own_drive_retains_canonically_and_its_child_still_reads(chain, monkeypatch):  # noqa: F811
    """A forked or empty parent's execution drive settles and goes; the retained bytes are canonical."""
    from ouroboros.delegate_readonly_inputs import schedule_selected_inputs
    from ouroboros.headless import task_state_dir
    from ouroboros.tools.registry import ToolContext

    rt, _parent, _engine, _sid = chain
    isolated = task_state_dir(rt.drive, "middle") / "data"
    source = isolated / "task_drives" / "middle" / "brief.md"
    source.parent.mkdir(parents=True)
    source.write_bytes(BRIEF)
    middle = ToolContext(repo_dir=rt.repo, drive_root=isolated, task_id="middle", task_metadata={
        "root_task_id": "parent", "parent_task_id": "parent", "budget_drive_root": str(rt.drive)})
    [row] = schedule_selected_inputs(middle, SELECTION[:1], "leaf")
    assert pathlib.Path(row["local"]).is_relative_to(rt.drive / "task_results" / "artifacts" / "middle")
    shutil.rmtree(isolated)
    payload = {"id": "leaf", "metadata": {}, "root_task_id": "parent", "parent_task_id": "middle",
               "drive_root": str(rt.drive), "budget_drive_root": str(rt.drive)}
    assert "parent brief" in _native_read(rt, payload, monkeypatch, path=row["local"])


def test_under_an_inherited_presence_ceiling_a_native_child_reads_exactly_its_copies(chain, monkeypatch):  # noqa: F811
    """The source passes the parent's Presence ceiling; the child, under that same inherited ceiling, reads its
    retained copies and nothing else in that store, with no artifact or runtime grant."""
    from tests.test_delegate_input_files import _presence

    rt, parent, _engine, _sid = chain
    drive = _parent_files(rt)
    (rt.drive / "elsewhere.md").write_bytes(b"ungranted runtime data\n")
    _presence(parent, ("task_drive", "."))  # the task drive only: no artifact_store or runtime_data grant
    parent.pending_events.clear()
    from ouroboros.tools.control_scheduling import _schedule_task

    refused = _schedule_task(parent, subagent_id=_api_row(monkeypatch), objective="Read", expected_output="Summary",
                             memory_mode="empty", input_files=[{"root": "runtime_data", "path": "elsewhere.md"}])
    assert "PRESENCE_RESOURCE_BLOCKED" in refused and not parent.pending_events, refused
    _reply, event = _schedule(chain, _api_row(monkeypatch), input_files=SELECTION)
    rows = event["prepared_input_files"]
    _admit(rt, event)
    payload = _restore(rt)
    assert payload["task_contract"]["capability_ceiling"] == parent.task_contract["capability_ceiling"], "inherited"
    (drive / "brief.md").write_bytes(b"edited after queueing\n")
    text = _native_read(rt, payload, monkeypatch, path=rows[0]["local"])
    assert "parent brief" in text and "PRESENCE" not in text, text
    manifest = pathlib.Path(rows[0]["local"]).parents[3] / "inputs.json"  # same store, not a handed copy
    assert "PRESENCE_RESOURCE_BLOCKED" in _native_read(rt, payload, monkeypatch, path=str(manifest))
    assert "PRESENCE_RESOURCE_BLOCKED" in _native_read(rt, payload, monkeypatch, root="runtime_data", path="elsewhere.md")
    pathlib.Path(rows[0]["local"]).write_bytes(b"tampered\n")  # a copy whose bytes no longer match is not admitted
    assert "PRESENCE_RESOURCE_BLOCKED" in _native_read(rt, payload, monkeypatch, path=rows[0]["local"])
