"""Structural contracts for the semantic-no-op worker child-process extraction."""

from __future__ import annotations

import ast
import pathlib
import pickle

from supervisor import worker_process, workers

_MOVED = (
    "WORKER_LOG_SINK_SUPPRESSED_TYPES",
    "_bind_worker_repo_root",
    "_current_custody_session_id",
    "_log_worker_crash",
    "_prepare_worker_task_runtime",
    "worker_main",
)

# What could NOT move: the pool's own state, and everything that reads it. The
# child process has none of it, which is exactly why the seam sits here.
_POOL_STATE = (
    "REPO_DIR", "DRIVE_ROOT", "WORKERS", "PENDING", "RUNNING",
    "QUEUE_SEQ_COUNTER_REF", "_CTX", "_LAST_SPAWN_TIME",
)


def test_the_child_process_module_never_imports_the_pool():
    tree = ast.parse(pathlib.Path(worker_process.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "supervisor.workers"
        if isinstance(node, ast.Import):
            assert all(alias.name != "supervisor.workers" for alias in node.names)


def test_workers_facade_reexports_every_moved_identity():
    for name in _MOVED:
        assert getattr(workers, name) is getattr(worker_process, name), name


def test_worker_main_is_still_module_level_and_picklable_by_name():
    """Spawn platforms re-import the target by qualified name; a nested or
    wrapped function would break every non-fork host."""
    assert worker_process.worker_main.__module__ == "supervisor.worker_process"
    assert worker_process.worker_main.__qualname__ == "worker_main"
    assert pickle.loads(pickle.dumps(worker_process.worker_main)) is worker_process.worker_main
    tree = ast.parse(pathlib.Path(worker_process.__file__).read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "worker_main"
        for node in tree.body
    )


def test_the_pool_kept_its_state_and_the_child_module_declares_none_of_it():
    child = vars(worker_process)
    for name in _POOL_STATE:
        assert hasattr(workers, name), name
        assert name not in child, name
