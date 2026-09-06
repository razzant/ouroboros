"""Structural contracts for the semantic-no-op server composition split."""

from __future__ import annotations

import ast
import pathlib

import server
from ouroboros import (
    server_liveness,
    server_maintenance,
    server_owner_routing,
    server_process,
    server_restart,
    server_routing_context,
)


REPO = pathlib.Path(__file__).parents[1]

_LEAVES = (
    server_process,
    server_routing_context,
    server_owner_routing,
    server_liveness,
    server_maintenance,
    server_restart,
)

_MOVED_OWNERS = {
    "DATA_DIR": server_process,
    "log": server_process,
    "_owner_restart_requested": server_process,
    "_request_restart_exit": server_process,
    "_restart_requested": server_process,
    "_active_direct_root": server_routing_context,
    "_addressable_root_tasks": server_routing_context,
    "_chat_running_tasks": server_routing_context,
    "_clip_marked": server_routing_context,
    "_decision_turn_metadata": server_routing_context,
    "_latest_project_task_result": server_routing_context,
    "_main_routing_manifest": server_routing_context,
    "_owner_binding_chat_id": server_routing_context,
    "_project_id_for_registered_chat": server_routing_context,
    "_reserved_project_for_chat": server_routing_context,
    "_scoped_task_metadata": server_routing_context,
    "_task_belongs_to_chat": server_routing_context,
    "_task_result_ground_truth": server_routing_context,
    "_owner_evolution_stop": server_owner_routing,
    "_record_routing_receipt": server_owner_routing,
    "_route_owner_message": server_owner_routing,
    "_route_project_chat_to_running_task": server_owner_routing,
    "_stage_mailbox_attachments": server_owner_routing,
    "_alert_chat_turn_wedge": server_liveness,
    "_chat_turn_wedged": server_liveness,
    "_start_supervisor_liveness_watchdog": server_liveness,
    "_supervisor_loop_stalled": server_liveness,
    "_LAST_CANCEL_INTENT_SWEEP": server_maintenance,
    "_installed_skill_names": server_maintenance,
    "_periodic_supervisor_maintenance": server_maintenance,
    "_periodic_zombie_reconcile": server_maintenance,
    "_prune_delegated_snapshots": server_maintenance,
    "_reconcile_delegated_runs": server_maintenance,
    "_resume_interrupted_project_deletions": server_maintenance,
    "_run_startup_task_recovery": server_maintenance,
    "_startup_custody_sweep": server_maintenance,
    "_startup_prune_sweeps": server_maintenance,
    "_startup_worktree_prune": server_maintenance,
    "_live_running_task_ids": server_restart,
    "_managed_update_pending_kwargs": server_restart,
    "_safe_restart_serialized": server_restart,
    "_shutdown_supervisor_event_bus": server_restart,
    "_shutdown_task_cleanup_args": server_restart,
}

# Process-scoped state and the composition itself: a leaf that needed one of
# these would have to import the parent back, so they must stay defined in
# server.py rather than arriving through an import. The restart transaction —
# the deferred drain record and the three functions around it — stays here too
# (HOT-DEFERRED): the upstream delegation train coupled the performer to
# ``main()`` through the written module global
# ``_planned_delegate_restart_transaction_id``, so a byte-preserving relocation
# would fork that state (docs/v7next/LEDGER_CORRECTIONS.md, D11 lane).
_SERVER_OWNED = (
    "_planned_delegate_restart_transaction_id",
    "_pending_restart",
    "_handle_restart_in_supervisor",
    "_check_pending_restart_drain",
    "_perform_supervisor_restart",
    "REPO_DIR",
    "PORT_FILE",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "RESTART_EXIT_CODE",
    "PANIC_EXIT_CODE",
    "_LAUNCHER_MANAGED",
    "_BIND_HOST",
    "_ACTUAL_BOUND_PORT",
    "_actual_bound_port",
    "_event_loop",
    "_supervisor_ready",
    "_supervisor_error",
    "_supervisor_thread",
    "_consciousness",
    "_execute_panic_stop",
    "_emergency_process_cleanup",
    "_process_bridge_updates",
    "_run_supervisor",
    "lifespan",
    "routes",
    "app",
    "main",
)


def _module_tree(module) -> ast.Module:
    return ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))


def test_server_leaves_never_import_the_composition_root():
    """A leaf that imports ``server`` back would reintroduce the cycle the split
    removed, at any depth — module level or inside a lazy function-local import."""
    for module in _LEAVES:
        for node in ast.walk(_module_tree(module)):
            if isinstance(node, ast.Import):
                assert not any(
                    alias.name == "server" or alias.name.startswith("server.")
                    for alias in node.names
                ), module.__name__
            if isinstance(node, ast.ImportFrom):
                assert node.module != "server", module.__name__


def test_server_facade_reexports_every_moved_identity():
    """``server`` keeps the exact objects, so importers and the tests that reach
    for ``server.<name>`` see no identity change."""
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(server, name), name
        assert getattr(server, name) is getattr(owner, name), name
    owned = {name for module in _LEAVES for name in vars(module)}
    assert set(_MOVED_OWNERS) <= owned


def test_shared_server_state_has_exactly_one_home():
    """The signals, the drive root, the logger, and the drain record are single
    objects shared by reference — not per-module copies that could drift."""
    assert server._restart_requested is server_process._restart_requested
    assert server._owner_restart_requested is server_process._owner_restart_requested
    assert server._restart_requested is server_liveness._restart_requested
    assert server.DATA_DIR is server_process.DATA_DIR
    assert server_maintenance.DATA_DIR is server_process.DATA_DIR
    assert server_liveness.DATA_DIR is server_process.DATA_DIR
    assert server.log is server_process.log
    assert server.log.name == "server"


def test_composition_root_still_defines_its_own_process_state():
    tree = _module_tree(server)
    defined: set[str] = set()
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imported.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
    for name in _SERVER_OWNED:
        assert name in defined, name
        assert name not in imported, name


def test_server_route_composition_is_owned_by_the_composition_root():
    paths = [getattr(route, "path", "") for route in server.routes]
    assert paths[0] == "/"
    assert paths[-1] == "/static"
    assert server.routes[0].endpoint is server.index_page
    keyed = [
        (path, tuple(sorted(getattr(route, "methods", None) or ())))
        for path, route in zip(paths, server.routes)
    ]
    assert len(keyed) == len(set(keyed))
    settings_route = next(
        route for route in server.routes if getattr(route, "path", "") == "/api/settings"
    )
    assert settings_route.endpoint.__module__ == "server"


def test_server_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in _LEAVES
    }
    counts["server"] = len((REPO / "server.py").read_text(encoding="utf-8").splitlines())
    assert all(count <= 1000 for name, count in counts.items() if name != "server")
    # server.py keeps the lifespan, the supervisor loop, the owner-command
    # dispatch, the process state those three need, AND (on this tree) the
    # deferred restart transaction plus post-cutoff upstream drift, so the
    # bound is looser than the reference's 1500 until the delegation organ
    # (F2) frees the restart rows.
    assert counts["server"] <= 1700
    assert 400 <= counts["ouroboros.server_routing_context"] <= 1000
    assert 400 <= counts["ouroboros.server_owner_routing"] <= 1000
