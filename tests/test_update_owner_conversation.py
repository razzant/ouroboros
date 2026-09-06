"""Main control during a self-update (#283).

Conversation admission is separate from repo-writing permission: while the ONE
authorized assisted resolver holds the repository the owner keeps talking to
Main, steering reaches the resolver through the ordinary mailbox, the registry
guard still refuses repo tools to that conversation, and the server's own
owner-control path is resident BEFORE conflict markers land in the live tree.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

import supervisor.git_ops as git_ops
import supervisor.update_merge as update_merge
import supervisor.worker_chat_lane as lane
import supervisor.workers as workers
from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESOLVER_ID = "update_assisted_merge_abc12345"
_TX = {
    "task_id": RESOLVER_ID,
    "pre_update_sha": "a" * 40,
    "target_sha": "b" * 40,
    "local_snapshot": "a" * 40,
    "pre_update_branch": "ouroboros",
    "owner_chat_id": 1,
}
LOCK_NOTICE = "An update is using the repository"


@pytest.fixture(autouse=True)
def _reset_writer_admission():
    workers.open_repo_writer_admission()
    yield
    workers.open_repo_writer_admission()


@pytest.fixture
def tx_repo(tmp_path, monkeypatch):
    """A checkout whose ``.git`` holds the durable update marker."""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    return repo


def _write_tx(phase: str) -> dict:
    tx = dict(_TX, phase=phase)
    update_merge.write_update_tx(tx)
    return tx


# --- admission predicate -----------------------------------------------------


@pytest.mark.parametrize("phase", ["assisted_resolution", "committing_assisted"])
def test_resolver_held_update_admits_conversation(tx_repo, phase):
    tx = _write_tx(phase)
    # A post-restart resume: only the durable marker closes the gate.
    assert lane.conversation_admitted_during_update(f"managed_update_tx:{phase}")
    # The same process: the assisted latch of this very transaction.
    assert lane.conversation_admitted_during_update(update_merge.assisted_writer_gate_reason(tx))
    # A destructive window latched over the same marker still refuses.
    for destructive in ("managed_update:rollback", "managed_update:smart",
                        "managed_update:replace_recovery", "managed_update:manual_rollback"):
        assert not lane.conversation_admitted_during_update(destructive), destructive
    # A latch of a DIFFERENT assisted transaction is not this one.
    assert not lane.conversation_admitted_during_update("managed_update:assisted:other")


@pytest.mark.parametrize("phase", [
    "materializing_assisted", "rolling_back", "stashing_local_work",
    "committing", "pending_boot_smoke", "gate_blocked",
])
def test_destructive_and_unproven_phases_keep_refusing(tx_repo, phase):
    _write_tx(phase)
    assert not lane.conversation_admitted_during_update(f"managed_update_tx:{phase}")


def test_open_gate_admits_and_unreadable_markers_fail_closed(tx_repo):
    assert lane.conversation_admitted_during_update("")
    # Destructive prologue before any marker exists.
    assert not lane.conversation_admitted_during_update("managed_update:smart")
    marker = tx_repo / ".git" / update_merge.UPDATE_TX_MARKER_NAME
    marker.write_text("{not json", encoding="utf-8")
    assert not lane.conversation_admitted_during_update("managed_update_tx:corrupt")
    marker.write_text(json.dumps({**_TX, "phase": "assisted_resolution",
                                  SCHEMA_VERSION_KEY: update_merge.UPDATE_TX_SCHEMA_VERSION + 1}),
                      encoding="utf-8")
    assert not lane.conversation_admitted_during_update("managed_update_tx:assisted_resolution")


# --- chat lanes ----------------------------------------------------------------


def _wire_gate(monkeypatch, reason: str, notices: list):
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: reason)
    monkeypatch.setattr(workers, "send_with_budget", lambda _chat_id, text, **_k: notices.append(text))


def test_direct_lane_runs_the_turn_while_the_resolver_holds_the_repo(tx_repo, monkeypatch):
    tx = _write_tx("assisted_resolution")
    notices, turns = [], []
    _wire_gate(monkeypatch, update_merge.assisted_writer_gate_reason(tx), notices)
    monkeypatch.setattr(lane, "_handle_chat_direct_locked", lambda *a, **k: turns.append(a[1]))

    lane.handle_chat_direct(1, "how is the update going?")

    assert turns == ["how is the update going?"]
    assert notices == []


def test_direct_lane_refuses_while_the_tree_is_being_materialized(tx_repo, monkeypatch):
    _write_tx("materializing_assisted")
    notices, turns = [], []
    _wire_gate(monkeypatch, "managed_update_tx:materializing_assisted", notices)
    monkeypatch.setattr(lane, "_handle_chat_direct_locked", lambda *a, **k: turns.append(a[1]))

    lane.handle_chat_direct(1, "hello?")

    assert turns == []
    assert len(notices) == 1 and LOCK_NOTICE in notices[0]


def test_ephemeral_lane_runs_the_turn_while_the_resolver_holds_the_repo(tx_repo, monkeypatch):
    import ouroboros.agent as agent_mod
    import supervisor.state as state

    tx = _write_tx("assisted_resolution")
    notices, turns = [], []
    _wire_gate(monkeypatch, update_merge.assisted_writer_gate_reason(tx), notices)
    monkeypatch.setattr(state, "load_state", lambda: {})
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(workers, "get_event_q", lambda: None)
    monkeypatch.setattr(agent_mod, "make_agent", lambda **_k: "ephemeral-agent")
    monkeypatch.setattr(
        lane, "_run_chat_task",
        lambda agent, chat_id, text, image_data, **kw: turns.append((agent, text, kw.get("ephemeral"))),
    )

    lane.handle_chat_ephemeral(1, "why did the merge conflict?")

    assert turns == [("ephemeral-agent", "why did the merge conflict?", True)]
    assert notices == []


def test_ephemeral_lane_refuses_during_a_destructive_window(tx_repo, monkeypatch):
    import supervisor.state as state

    _write_tx("assisted_resolution")
    notices, turns = [], []
    _wire_gate(monkeypatch, "managed_update:rollback", notices)
    monkeypatch.setattr(state, "load_state", lambda: {})
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(lane, "_run_chat_task", lambda *a, **k: turns.append(a))

    lane.handle_chat_ephemeral(1, "hello?")

    assert turns == []
    assert len(notices) == 1 and LOCK_NOTICE in notices[0]


# --- conversation admission is not repo-writing permission ----------------------


def test_admitted_conversation_turn_still_cannot_write_the_repo(tx_repo):
    from ouroboros.tools.registry_guards import _managed_update_code_tool_block_result

    tx = _write_tx("assisted_resolution")
    assert lane.conversation_admitted_during_update("managed_update_tx:assisted_resolution")

    direct_turn = SimpleNamespace(task_id="chat_direct_1", task_metadata={"source": "owner_chat"})
    blocked = _managed_update_code_tool_block_result(direct_turn, "write_file")
    assert blocked is not None and blocked.status == "blocked"
    assert "MANAGED_UPDATE_IN_PROGRESS" in blocked.text

    resolver = SimpleNamespace(
        task_id=RESOLVER_ID,
        task_metadata={"managed_update": {
            "authority_fingerprint": update_merge.assisted_authority_fingerprint(tx),
        }},
    )
    assert _managed_update_code_tool_block_result(resolver, "write_file") is None


def test_main_steers_the_resolver_through_the_mailbox_while_the_gate_is_closed(
    tx_repo, tmp_path, monkeypatch,
):
    import supervisor.queue as supervisor_queue
    from ouroboros.owner_mailbox import drain_owner_messages
    from supervisor.steering import _handle_steer_task

    tx = _write_tx("assisted_resolution")
    monkeypatch.setattr(supervisor_queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(
        workers, "repo_writer_admission_closed",
        lambda: update_merge.assisted_writer_gate_reason(tx),
    )
    assert lane.conversation_admitted_during_update(workers.repo_writer_admission_closed())
    running = {RESOLVER_ID: {"task": {
        "id": RESOLVER_ID, "chat_id": 1, "type": "update_assisted_merge",
        "title": "Resolve the update merge",
    }}}
    handler_ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING=running, PENDING=[], bridge=None,
        get_chat_agent=lambda: None, persist_queue_snapshot=lambda **_k: True,
    )

    _handle_steer_task({
        "type": "steer_task", "target_task_id": RESOLVER_ID, "chat_id": 1,
        "message": "Keep our local config.py changes; take upstream for loop.py.",
        "client_message_id": "steer-owner-1",
    }, handler_ctx)

    assert drain_owner_messages(tmp_path, RESOLVER_ID) == [
        "Keep our local config.py changes; take upstream for loop.py.",
    ]


# --- the control path is resident before conflict markers land -----------------

_OWNER_CONTROL_CHAIN = (
    "ouroboros/server_owner_routing.py",
    "ouroboros/server_routing_context.py",
    "supervisor/worker_chat_lane.py",
    "supervisor/steering.py",
    "supervisor/events_project_routing.py",
    "ouroboros/tools/control_routing.py",
    "ouroboros/routing_wait.py",
    "ouroboros/owner_mailbox.py",
    "ouroboros/loop_round_limits.py",
)


def _is_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _first_party_imports(path: pathlib.Path) -> set[str]:
    """Every ``ouroboros.*``/``supervisor.*`` module the file imports at ANY depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {n for n in names if n.split(".")[0] in {"ouroboros", "supervisor"} and _is_module(n)}


@pytest.mark.serial
def test_preload_covers_every_import_of_the_owner_control_chain():
    expected: set[str] = set()
    for rel in _OWNER_CONTROL_CHAIN:
        expected |= _first_party_imports(REPO_ROOT / rel)
    assert expected, "the chain scan found nothing — the file list is stale"
    code = textwrap.dedent("""
        import json, sys
        import supervisor.worker_chat_lane as lane
        failed = lane.preload_owner_control_path()
        resident = sorted(m for m in sys.modules if m.split(".")[0] in ("ouroboros", "supervisor"))
        print(json.dumps({"failed": failed, "resident": resident}))
    """)
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
        timeout=240, env=dict(os.environ),
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["failed"] == [], report["failed"]
    missing = expected - set(report["resident"])
    assert not missing, f"first imported only AFTER conflict markers could land: {sorted(missing)}"
    assert "ouroboros.tools.core" in report["resident"], "the tool catalog was not loaded"


def _git(repo: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)


@pytest.mark.serial
def test_preloaded_module_keeps_working_once_real_conflict_markers_land(tmp_path):
    """An isolated checkout with a REAL merge conflict: the module imported before
    the markers keeps answering, the module first imported after them cannot."""
    repo = tmp_path / "repo"
    pkg = repo / "ctl"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "resident.py").write_text("def answer():\n    return 'resident ok'\n", encoding="utf-8")
    (pkg / "late.py").write_text("def steer():\n    return 'late ok'\n", encoding="utf-8")
    assert _git(repo, "init", "-q", "-b", "main").returncode == 0
    for key, value in (("user.email", "t@example.com"), ("user.name", "t"), ("commit.gpgsign", "false")):
        _git(repo, "config", key, value)
    _git(repo, "add", "-A")
    assert _git(repo, "commit", "-q", "-m", "base").returncode == 0
    assert _git(repo, "checkout", "-q", "-b", "release").returncode == 0
    (pkg / "resident.py").write_text("def answer():\n    return 'resident upstream'\n", encoding="utf-8")
    (pkg / "late.py").write_text("def steer():\n    return 'late upstream'\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "upstream")
    assert _git(repo, "checkout", "-q", "main").returncode == 0
    (pkg / "resident.py").write_text("def answer():\n    return 'resident local'\n", encoding="utf-8")
    (pkg / "late.py").write_text("def steer():\n    return 'late local'\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "local")

    ready, go = tmp_path / "ready", tmp_path / "go"
    driver = tmp_path / "driver.py"
    driver.write_text(textwrap.dedent(f"""
        import json, pathlib, sys, time
        sys.path.insert(0, {str(repo)!r})
        import ctl.resident  # the preload: imported while the tree is clean
        pathlib.Path({str(ready)!r}).write_text("1")
        deadline = time.time() + 30
        while not pathlib.Path({str(go)!r}).exists() and time.time() < deadline:
            time.sleep(0.02)
        out = {{"resident": ctl.resident.answer()}}
        try:
            import ctl.late
            out["late"] = ctl.late.steer()
        except SyntaxError as exc:
            out["late"] = "SyntaxError:" + pathlib.Path(exc.filename).name
        print(json.dumps(out))
    """), encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(driver)], cwd=tmp_path, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.time() + 30
        while not ready.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "the driver never finished its clean-tree import"

        merge = _git(repo, "merge", "--no-commit", "--no-ff", "release")
        assert merge.returncode != 0, "the fixture merge did not conflict"
        for name in ("resident.py", "late.py"):
            text = (pkg / name).read_text(encoding="utf-8")
            assert "<<<<<<<" in text and ">>>>>>>" in text, f"{name} carries no conflict markers"
        go.write_text("1", encoding="utf-8")
        stdout, stderr = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    assert proc.returncode == 0, stderr[-4000:]
    out = json.loads(stdout.strip().splitlines()[-1])
    # The preloaded module still answers with the code imported BEFORE the merge;
    # the module first imported after the markers landed cannot be imported at all.
    assert out == {"resident": "resident local", "late": "SyntaxError:late.py"}
