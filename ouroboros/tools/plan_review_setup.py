"""Plan-review SETUP: open the subject tree, settle the review's shape.

The two steps that must happen before a plan review reads anything, in this order:
resolve WHERE the subject is (on a remote placement that is a mirror of the target's
files, opened into the review's own exit stack so nothing else can close it early),
then WHAT KIND of review this is (class, then the context level derived from it).

Split from the review body because they are the only parts that can fail with a typed
refusal before a single model call is made — keeping them together keeps the two
failure prefixes (`PLAN_REMOTE_SUBJECT_UNAVAILABLE`, `PLAN_SUBJECT_ROOT_INVALID`)
next to the code that can raise them.
"""

from __future__ import annotations

from dataclasses import replace

from ouroboros.tools.plan_review_runtime import (
    classify_reviewer_error as _classify_reviewer_error,  # noqa: F401 — test-compat re-export
    resolve_plan_class as _resolve_plan_class,
)
from ouroboros.tools.review_synthesis import (
    emit_plan_review_usage as _emit_plan_review_usage,  # noqa: F401 — test-compat re-export
    minted_plan_slot_ids as _minted_plan_slot_ids,  # noqa: F401 — test-compat re-export
    resolve_plan_context_level as _resolve_plan_context_level,
)
import pathlib
from typing import Any
from ouroboros.review_substrate import review_repo_dirs_for
from ouroboros.tools.registry import ToolContext


def _resolve_plan_roots(
    ctx: ToolContext, files_to_touch: list, *, snapshot: Any = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Resolve governance and subject roots without silently mixing them.

    ``snapshot`` is the verified remote mirror when the task's workspace lives on
    another host: governance stays Home's own repo (it is Home BY DEFINITION), and the
    SUBJECT becomes the mirror, because that is the tree the reviewer will read. It is
    an explicit parameter rather than something read off ctx so the root the paths are
    validated against is provably the root that will be read."""
    from ouroboros.remote_plan_review import plan_subject_root

    if snapshot is not None:
        governance = pathlib.Path(
            getattr(ctx, "system_repo_dir", None) or ctx.repo_dir
        ).resolve(strict=False)
        subject = plan_subject_root(snapshot, governance)
    else:
        governance, subject = review_repo_dirs_for(ctx)
    for raw in files_to_touch or []:
        candidate = pathlib.Path(str(raw or ""))
        resolved = (candidate if candidate.is_absolute() else subject / candidate).resolve(strict=False)
        try:
            resolved.relative_to(subject)
        except ValueError as exc:
            raise ValueError(
                f"planned path {raw!r} escapes active subject root {subject}"
            ) from exc
    return governance, subject


def _open_plan_subject_roots(ctx, files_to_touch, exit_stack):
    """Open the review's subject tree and resolve its two roots.

    On a remote placement the subject is a MIRROR of the target's files, and it is
    opened here so the exit stack that owns the rest of the review owns its lifetime
    too — nothing else can close it early, and nothing leaks if the review returns
    from any of the branches below.

    Returns ``(snapshot, governance_repo, subject_repo, error)``; a non-empty error
    is the caller's return value verbatim, so the two typed failures keep their
    exact prefixes.
    """
    from ouroboros.remote_patch_bridge import RemotePatchError
    from ouroboros.remote_plan_review import open_plan_subject
    from ouroboros.remote_transfer import RemoteSnapshotError

    try:
        subject = open_plan_subject(ctx, files_to_touch)
        snapshot = exit_stack.enter_context(subject) if subject is not None else None
    except (RemotePatchError, RemoteSnapshotError) as exc:
        return None, None, None, f"ERROR: PLAN_REMOTE_SUBJECT_UNAVAILABLE: {exc}"
    try:
        governance_repo, subject_repo = _resolve_plan_roots(
            ctx, files_to_touch, snapshot=snapshot,
        )
    except ValueError as exc:
        return None, None, None, f"ERROR: PLAN_SUBJECT_ROOT_INVALID: {exc}"
    return snapshot, governance_repo, subject_repo, ""


def _resolve_plan_shape(ctx, request, *, plan_class, context_level, files_to_touch, scope):
    """Settle the review's CLASS and CONTEXT LEVEL, then rebuild the request on them.

    The two travel together because the second is derived from the first, and an
    escalation the class resolver decides has to be announced before anything reads
    the level it produced. Returns ``(request, error)``.
    """
    resolved_class, escalation_note = _resolve_plan_class(ctx, plan_class, files_to_touch)
    if escalation_note:
        ctx.emit_progress_fn(f"📐 plan_task: {escalation_note}")
    try:
        resolved_context_level = _resolve_plan_context_level(context_level, plan_class=resolved_class)
    except ValueError as exc:
        return None, f"ERROR: {exc}"
    return replace(
        request, context_level=resolved_context_level, plan_class=resolved_class, scope=scope,
    ), ""


__all__ = ['_open_plan_subject_roots', '_resolve_plan_shape']
