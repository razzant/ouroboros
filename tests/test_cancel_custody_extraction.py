"""Structural contracts for the semantic-no-op cancellation-custody extraction."""

from __future__ import annotations

import ast
import pathlib

from supervisor import cancel_custody, queue, task_lifecycle

REPO = pathlib.Path(__file__).parents[1]

_MOVED = (
    "SETTLED_ALREADY",
    "_active_intent",
    "_claim_intent",
    "_durable_settled_status",
    "_finalize_cancel_intent_on_miss",
    "_finish_captured_pending",
    "_finish_captured_running",
    "_intent_outcome_fields",
    "_queue_module",
    "_reaping_owner_abandoned",
    "_recover_stranded_reaping_slot",
    "_release_intent_claim",
    "_restore_custody",
    "_settle_intent",
    "_worker_possibly_alive",
    "cancel_task_custody",
)

# The cascade protocol is ONE protocol over module-local state: the token
# sequence, the protected-fence sets and the sweep that reads them stay together,
# or a cross-module mutable global replaces a local invariant.
_CASCADE_PROTOCOL = (
    "CANCELLED_ROOT_FENCES",
    "_ACTIVE_CASCADE_FENCES",
    "_CASCADE_TOKEN_SEQ",
    "_cancel_subtree_sweep",
    "_next_cascade_token",
    "_prune_cancellation_fences",
    "_record_cascade_scope",
    "cancel_task_by_id",
)


def _top_level_names(module) -> set[str]:
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_custody_never_imports_the_lifecycle_module_it_was_split_from():
    tree = ast.parse(pathlib.Path(cancel_custody.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "supervisor.task_lifecycle"
        if isinstance(node, ast.Import):
            assert all(a.name != "supervisor.task_lifecycle" for a in node.names)


def test_task_lifecycle_facade_reexports_every_moved_identity():
    """``supervisor.task_lifecycle`` keeps the exact objects, and through it
    ``supervisor.queue`` stays the single public import surface."""
    owned = _top_level_names(cancel_custody)
    for name in _MOVED:
        assert name in owned, name
        assert getattr(task_lifecycle, name) is getattr(cancel_custody, name), name
    assert queue.cancel_task_by_id is task_lifecycle.cancel_task_by_id
    assert queue.record_scheduled_admission is task_lifecycle.record_scheduled_admission


def test_the_cascade_protocol_stayed_whole_with_its_module_local_state():
    lifecycle_names = _top_level_names(task_lifecycle)
    custody_names = _top_level_names(cancel_custody)
    for name in _CASCADE_PROTOCOL:
        assert name in lifecycle_names, name
        assert name not in custody_names, name


def test_custody_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (task_lifecycle, cancel_custody)
    }
    assert all(count <= 1000 for count in counts.values())
    assert 600 <= counts["supervisor.cancel_custody"] <= 1000
