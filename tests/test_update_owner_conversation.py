"""Main control during a self-update (#283).

Conversation admission is separate from repo-writing permission: while the ONE
authorized assisted resolver holds the repository the owner keeps talking to
Main, steering reaches the resolver through the ordinary mailbox, the registry
guard still refuses repo tools to that conversation, and the server's own
owner-control path is resident BEFORE conflict markers land in the live tree.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
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


@pytest.mark.parametrize('live', [False, True])
def test_native_turn_reaches_execution_while_the_resolver_holds_the_repo(tx_repo, monkeypatch, live):
    import supervisor.state as state

    tx = _write_tx("assisted_resolution")
    notices, turns = [], []
    _wire_gate(monkeypatch, update_merge.assisted_writer_gate_reason(tx), notices)
    monkeypatch.setattr(state, "load_state", lambda: {})
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(
        lane, "_run_chat_task",
        lambda agent, chat_id, text, image_data, **kw: turns.append((agent, text, kw)),
    )

    turn = object() if live else None
    lane.handle_chat_direct(1, "why did the merge conflict?", realtime_turn=turn)

    expected = {"task_constraint": None, "task_metadata": None}
    if live:
        expected['realtime_turn'] = turn
    assert turns == [(None, "why did the merge conflict?", expected)]  # construction belongs to registered execution
    assert notices == []


def test_native_execution_refuses_during_a_destructive_window(tx_repo, monkeypatch):
    import supervisor.state as state

    _write_tx("assisted_resolution")
    notices, turns = [], []
    _wire_gate(monkeypatch, "managed_update:rollback", notices)
    monkeypatch.setattr(state, "load_state", lambda: {})
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(lane, "_run_chat_task", lambda *a, **k: turns.append(a))

    lane.handle_chat_direct(1, "hello?")

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
    from ouroboros.owner_mailbox import drain_owner_entries
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

    assert [e["text"] for e in drain_owner_entries(tmp_path, RESOLVER_ID)] == [
        "Keep our local config.py changes; take upstream for loop.py.",
    ]


# --- the control path is resident before conflict markers land -----------------

@pytest.fixture
def package_repo(tmp_path):
    """The materialized source layout used by the packaged server as well as source installs."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for package in ("ouroboros", "supervisor"):
        shutil.copytree(
            REPO_ROOT / package, repo / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copyfile(REPO_ROOT / "VERSION", repo / "VERSION")
    return repo


def _run_package_code(repo: pathlib.Path, code: str) -> dict:
    env = dict(os.environ, OUROBOROS_REPO_DIR=str(repo))
    # The copied package wins over this test process and any editable install;
    # imports never depend on cwd being the repository.
    script = "import sys; sys.path.insert(0, " + repr(str(repo)) + ")\n" + textwrap.dedent(code)
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=repo.parent, capture_output=True,
        text=True, timeout=240, env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.serial
@pytest.mark.parametrize("frozen", [False, True])
def test_preload_discovers_nested_packages_and_preserves_tool_catalog(package_repo, frozen):
    # The shipped server is a standalone-Python subprocess of the frozen
    # launcher, with both first-party packages on disk. A frozen caller with
    # those same bundled paths also retains the registry's frozen inventory.
    nested = package_repo / "ouroboros" / "_discovery_fixture"
    nested.mkdir()
    (nested / "__init__.py").write_text("", encoding="utf-8")
    (nested / "late.py").write_text("value = 42\n", encoding="utf-8")
    expected = {
        ".".join(path.relative_to(package_repo).with_suffix("").parts).removesuffix(".__init__")
        for package in ("ouroboros", "supervisor")
        for path in (package_repo / package).rglob("*.py")
    }
    out = _run_package_code(package_repo, f"""
        import json, pathlib, sys
        import supervisor.worker_chat_lane as lane
        from ouroboros.tools.registry import ToolRegistry
        from ouroboros.config import DATA_DIR, REPO_DIR
        sys.frozen = {frozen!r}
        before = ToolRegistry(REPO_DIR, DATA_DIR).available_tools()
        failed = lane.preload_owner_control_path()
        after = ToolRegistry(REPO_DIR, DATA_DIR).available_tools()
        print(json.dumps({{"failed": failed, "resident": sorted(sys.modules),
                          "same_catalog": before == after}}))
    """)
    assert out["failed"] == []
    assert expected <= set(out["resident"]), sorted(expected - set(out["resident"]))
    assert out["same_catalog"], "preloading code must not change tool admission"


@pytest.mark.serial
def test_preload_reports_failed_optional_modules_and_continues(package_repo):
    nested = package_repo / "ouroboros" / "_discovery_fixture"
    nested.mkdir()
    (nested / "__init__.py").write_text("", encoding="utf-8")
    (nested / "broken.py").write_text("raise ImportError('optional dependency unavailable')\n", encoding="utf-8")
    (nested / "valid.py").write_text("value = 42\n", encoding="utf-8")
    out = _run_package_code(package_repo, """
        import io, json, logging, sys
        from supervisor.worker_chat_lane import preload_owner_control_path
        log = io.StringIO()
        logging.getLogger('supervisor.worker_chat_lane').addHandler(logging.StreamHandler(log))
        failed = preload_owner_control_path()
        print(json.dumps({"failed": failed, "valid": 'ouroboros._discovery_fixture.valid' in sys.modules,
                          "diagnostic": log.getvalue()}))
    """)
    assert out["failed"] == ["ouroboros._discovery_fixture.broken"]
    assert out["valid"]
    assert "optional dependency unavailable" in out["diagnostic"]


def _git(repo: pathlib.Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if check:
        assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc


@pytest.mark.serial
@pytest.mark.parametrize("preload", [False, True])
def test_update_recovery_uses_preloaded_policy_after_real_git_conflict(package_repo, preload):
    """Exercise the real late import that the old manual closure missed."""
    policy = package_repo / "supervisor" / "update_merge_policy.py"
    source = policy.read_text(encoding="utf-8")
    policy.write_text(source + "\nCONFLICT_FIXTURE = 0\n", encoding="utf-8")
    _git(package_repo, "init", "-q", "-b", "main")
    for key, value in (("user.email", "test@example.invalid"), ("user.name", "Fixture"),
                       ("commit.gpgsign", "false")):
        _git(package_repo, "config", key, value)
    _git(package_repo, "add", "supervisor/update_merge_policy.py")
    _git(package_repo, "commit", "-qm", "base")
    _git(package_repo, "checkout", "-qb", "release")
    policy.write_text(source + "\nCONFLICT_FIXTURE = 1\n", encoding="utf-8")
    _git(package_repo, "commit", "-qam", "upstream")
    _git(package_repo, "checkout", "-q", "main")
    policy.write_text(source + "\nCONFLICT_FIXTURE = 2\n", encoding="utf-8")
    _git(package_repo, "commit", "-qam", "local")

    out = _run_package_code(package_repo, f"""
        import json, pathlib, subprocess, sys
        import supervisor.worker_chat_lane as lane
        if {preload!r}:
            assert lane.preload_owner_control_path() == []
        resident = 'supervisor.update_merge_policy' in sys.modules
        repo = pathlib.Path({str(package_repo)!r})
        merged = subprocess.run(['git', 'merge', '--no-commit', '--no-ff', 'release'],
                                cwd=repo, capture_output=True, text=True)
        assert merged.returncode == 1, merged.stdout + merged.stderr
        text = (repo / 'supervisor/update_merge_policy.py').read_text()
        assert '<<<<<<<' in text and '>>>>>>>' in text
        from supervisor import workers, update_merge
        # Reach the real objective/metadata path without starting any work.
        workers.RUNNING = {{'probe': {{}}}}
        workers.PENDING = []
        workers.ensure_worker_pool_started = lambda **_: True
        result = {{'resident': resident}}
        try:
            result['task'] = update_merge.enqueue_assisted_resolution_task({{
                'task_id': 'probe', 'owner_chat_id': 1, 'target_sha': '1' * 40,
                'conflict_paths': ['supervisor/update_merge_policy.py'],
            }})
        except SyntaxError as exc:
            result['syntax_error'] = pathlib.Path(exc.filename).name
        print(json.dumps(result))
    """)
    if preload:
        assert out == {"resident": True, "task": "probe"}
    else:
        assert out == {"resident": False, "syntax_error": "update_merge_policy.py"}
