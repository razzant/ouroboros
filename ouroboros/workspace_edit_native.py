"""Target-native multi-file editors: `apply_patch` and `edit_batch`.

Their own module rather than more of `workspace_native`, and the seam is a real one:
both are pure applications of `patch_core` through the target's own doors, with no
process, no git and no transport. Everything policy-bearing is either in `patch_core`
(shared with Home, so the two routes cannot drift) or in the two path doors below.

TWO doors, not one. `native_mutation_target` answers "may this be written", asked at
each mutation; `native_target(question=QUESTION_NAMED_SOURCE)` answers "may these
bytes be read", asked before each `read_text`. Both editors located hunks and counted
occurrences through the WRITE door alone, which by design narrows to protected
artifacts — so they read files whose `read_file` the same task was refused, and
answered about their contents. The residual, deliberate cost is the one `_edit_text`
already states: an excluded file may still be WRITTEN, it may no longer be READ.
"""

from __future__ import annotations

import pathlib
from typing import Any, Mapping

from ouroboros.export_policy_contract import QUESTION_NAMED_SOURCE
from ouroboros.patch_core import _parse_patch, plan_patch
from ouroboros.workspace_native_contract import ToolExecutionEnvelope
from ouroboros.workspace_native_paths import (
    atomic_write as _atomic_write,
    native_mutation_target as _mutation_target,
    native_relative_spelling as _relative_text,
    native_target as _read_target,
)

def _apply_patch(
    root: pathlib.Path,
    args: Mapping[str, Any],
    *,
    native_facts: Mapping[str, Any] | None = None,
) -> ToolExecutionEnvelope:
    """Target-native `apply_patch`.

    The patch is PARSED, PLANNED and APPLIED by `patch_core` — the same functions
    the Home handler runs — so a patch that applies on Home applies here, byte for
    byte, and an unmatched hunk aborts both the same way. What this route supplies
    is only its own doors: `_mutation_target` is the target's write policy, and its
    filesystem answers `exists`/`read`. Nothing about atomicity, hunk location or
    chained updates is decided twice.
    """
    patch = str(args.get("patch") or "")
    if not patch.strip():
        return ToolExecutionEnvelope(
            text="⚠️ APPLY_PATCH_ERROR: patch is required.",
            trace={"completion": "complete", "paths": []},
        )
    ops, err = _parse_patch(patch)
    if err:
        return ToolExecutionEnvelope(text=err, trace={"completion": "complete", "paths": []})

    targets: dict[str, pathlib.Path] = {}
    # The ops that will READ the file to locate their hunks. `add` composes its content
    # and `delete` needs none, so only `update` asks the read question — which is what
    # keeps a write to an excluded path permitted, as `_edit_text` already documents.
    reading = {_relative_text(op.path) for op in ops if op.kind == "update"}

    def _resolve(raw_path: Any) -> tuple[str, str]:
        rel = _relative_text(raw_path)
        # Both doors are asked for EVERY operation during planning, before a single
        # byte moves: a policy refusal must not land after earlier files of the same
        # patch are already on disk.
        #
        # MUTATION first, so the typed write refusal outranks the read one — the same
        # precedence `_edit_text` sets. The READ door then answers the question these
        # editors never asked: locating a hunk MATCHES the file's bytes, and every
        # diagnostic it can return is a fact about them ("anchor not found", "context
        # is ambiguous — matches at line 3, line 7"), so a patch against `.env` or
        # `id_rsa` was a content oracle over a file whose `read_file` the same task had
        # just been refused. It is asked HERE and not at the read itself because the
        # planner checks existence before reading: judged later, an absent excluded
        # path would answer "file not found" while a present one answered with the
        # policy — the existence oracle `native_target` refuses to be.
        targets[rel] = _mutation_target(root, rel, facts=native_facts)
        if rel in reading:
            _read_target(root, rel, question=QUESTION_NAMED_SOURCE, facts=native_facts)
        return rel, ""

    def _read(rel: str) -> tuple[str, str]:
        try:
            return targets[rel].read_text(encoding="utf-8"), ""
        except Exception as exc:  # noqa: BLE001 - report unreadable target
            return "", f"⚠️ APPLY_PATCH_ERROR: cannot read {rel}: {exc}"

    plan = plan_patch(
        ops,
        resolve=_resolve,
        exists=lambda rel: targets[rel].exists(),
        read_text=_read,
    )
    if plan.error:
        return ToolExecutionEnvelope(
            text=plan.error, trace={"completion": "complete", "paths": []}
        )

    # The door is asked AGAIN at the moment of each mutation rather than trusting the
    # map the planning pass filled: the write and the delete are the acts that need
    # confinement, and a path that reaches them through a dict lookup is a path no
    # reader (and no gate) can trace back to the policy that admitted it.
    changed: list[str] = []
    for rel, content in plan.writes:
        _atomic_write(_mutation_target(root, rel, facts=native_facts), content.encode("utf-8"))
        changed.append(rel)
    for rel in plan.deletes:
        _mutation_target(root, rel, facts=native_facts).unlink()
        changed.append(rel)

    body = "\n".join(plan.summaries)
    if plan.notes:
        body += "\nNotes:\n" + "\n".join("  " + note for note in plan.notes)
    return ToolExecutionEnvelope(
        text=body,
        trace={"completion": "complete", "paths": changed},
    )

def _edit_batch(
    root: pathlib.Path,
    args: Mapping[str, Any],
    *,
    native_facts: Mapping[str, Any] | None = None,
) -> ToolExecutionEnvelope:
    """Target-native `edit_batch`: counted exact replacements, whole batch atomic.

    Same contract as Home's: every edit declares how many occurrences it expects,
    ANY mismatch aborts the whole batch before anything is written, and the
    diagnostics name each failing edit. The count check is the batch's entire
    reason to exist, so it is stated once here in the same shape Home states it.
    """
    rows = args.get("edits")
    edits = [dict(row) for row in rows] if isinstance(rows, list) else []
    if not edits:
        return ToolExecutionEnvelope(
            text="⚠️ EDIT_BATCH_ERROR: edits is required (a non-empty list).",
            trace={"completion": "complete", "paths": []},
        )

    pending: dict[str, str] = {}
    order: list[str] = []
    problems: list[str] = []
    summaries: list[str] = []

    for index, edit in enumerate(edits, start=1):
        rel = _relative_text(edit.get("path"))
        old = str(edit.get("old_str") or "")
        new = str(edit.get("new_str") or "")
        expected = int(edit.get("count", 1) or 1)
        if not old:
            problems.append(f"edit {index} ({rel}): old_str is required (cannot be empty)")
            continue
        # Both doors, in the same order and at the same moment as `_apply_patch`.
        # Every row of this batch reads its file — the occurrence COUNT the batch is
        # built around IS the oracle: `expected 1, found 0` against `.env` answers
        # whether a guessed substring is in it, one guess at a time. Asked BEFORE the
        # existence check below so an absent excluded path and a present one get the
        # same answer.
        target = _mutation_target(root, rel, facts=native_facts)
        _read_target(root, rel, question=QUESTION_NAMED_SOURCE, facts=native_facts)
        if rel in pending:
            content = pending[rel]
        elif not target.exists():
            problems.append(f"edit {index} ({rel}): file not found")
            continue
        else:
            try:
                content = target.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001 - report unreadable target
                problems.append(f"edit {index} ({rel}): cannot read: {exc}")
                continue
        found = content.count(old)
        if found != expected:
            problems.append(
                f"edit {index} ({rel}): expected {expected} occurrence(s), found {found}"
            )
            continue
        pending[rel] = content.replace(old, new)
        if rel not in order:
            order.append(rel)
        summaries.append(f"✅ {rel}: {found} replacement(s)")

    if problems:
        return ToolExecutionEnvelope(
            text=(
                "⚠️ EDIT_BATCH_ERROR: nothing was written (the batch is atomic).\n"
                + "\n".join("  " + p for p in problems)
            ),
            trace={"completion": "complete", "paths": []},
        )

    changed: list[str] = []
    for rel in order:
        _atomic_write(_mutation_target(root, rel, facts=native_facts), pending[rel].encode("utf-8"))
        changed.append(rel)
    return ToolExecutionEnvelope(
        text="\n".join(summaries),
        trace={"completion": "complete", "paths": changed},
    )


__all__ = ["_apply_patch", "_edit_batch"]
