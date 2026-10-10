"""The skill reviewer's prompt: its contract, its governance context, its waves.

Owns what the tri-model skill reviewer is asked: the closed list of Skill
Review Checklist items every actor must answer, the checklist section name and
the governance artifacts loaded beside it with an explicit omission marker,
the assembled prompt with its stable cacheable prefix, and the per-round
assembly that binds history and accepted rebuttals to the current review round
of the group. The attempt's third element (advisory evidence, persisted as
``advisory_result``) is always empty: no advisory critic feeds the skill reviewer.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

from ouroboros.reference_books import BOOK_ENTRYPOINTS, compose_book, load_reference_book
from ouroboros.skill_review_status import CRITICAL_ITEMS
from ouroboros.tools.review_helpers import (
    build_rebuttal_section,
    build_skill_host_context,
    load_checklist_section,
)
from ouroboros.skill_review_cycles import load_accepted_rebuttals as _load_accepted_rebuttals
from ouroboros.skill_review_rebuttals import (
    _build_skill_review_history_section,
    _render_accepted_rebuttals_section,
)

_SKILL_CHECKLIST_SECTION = "Skill Review Checklist"


_SKILL_REVIEW_ITEMS = (
    "manifest_schema",
    "permissions_honesty",
    "no_repo_mutation",
    "path_confinement",
    "env_allowlist",
    "timeout_and_output_discipline",
    "extension_namespace_discipline",
    # Module widgets are arbitrary JS in an opaque-origin sandbox (storage throws there); review
    # checks cross-prefix fetch, bespoke parent messaging, launch-policy fit, dispose-state handling.
    "widget_module_safety",
    "inject_chat_minimization",
    "event_subscription_minimization",
    "companion_process_safety",
    "host_token_handling",
    "error_handling",
    "integration_preflight",
    "bug_hunting",
    "completion_notification",
)


_CRITICAL_ITEMS = CRITICAL_ITEMS


def _load_governance_artifact(
    repo_root: pathlib.Path,
    relpath: str,
) -> str:
    """Load governance context with an explicit omission marker on failure."""
    from ouroboros.tools.review_helpers import load_governance_doc

    for book_id, entrypoint in BOOK_ENTRYPOINTS.items():
        if relpath == entrypoint:
            try:
                return compose_book(load_reference_book(repo_root, book_id))
            except (OSError, ValueError) as exc:
                return f"[⚠️ OMISSION: {relpath} book could not be loaded: {exc}]"
    return load_governance_doc(repo_root, relpath, on_missing="explicit")


# Resolve repo root from this file for source and packaged builds.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _build_review_prompt(
    skill_name: str,
    skill_dir: pathlib.Path,
    manifest_dump: str,
    content_hash: str,
    file_pack: str,
    review_rebuttal: str = "",
    review_history_section: str = "",
) -> tuple[str, int]:
    try:
        checklist_section = load_checklist_section(_SKILL_CHECKLIST_SECTION)
    except ValueError as exc:
        checklist_section = (
            f"(⚠️ SKILL_REVIEW_ERROR: checklist section missing: {exc})"
        )
    architecture_text = _load_governance_artifact(_REPO_ROOT, "docs/ARCHITECTURE.md")
    development_text = _load_governance_artifact(_REPO_ROOT, "docs/DEVELOPMENT.md")
    bible_text = _load_governance_artifact(_REPO_ROOT, "BIBLE.md")
    skill_host_context = build_skill_host_context(_REPO_ROOT)
    items_json = json.dumps(list(_SKILL_REVIEW_ITEMS))
    # STABLE-FIRST assembly for provider prompt caching: the checklist,
    # governance docs, and host contracts are byte-identical across review
    # rounds and form the cache-marked prefix; the per-skill identity,
    # manifest, payload, rebuttal, and history are the dynamic tail.
    # The output contract stays LAST — after the untrusted payload — which is
    # the prompt-injection boundary this review relies on (never move it).
    stable = f"""\
You are performing a SKILL review, not a repo-commit review.

This review vets a single external skill package that lives OUTSIDE the
self-modifying Ouroboros repository (its identity, manifest, and payload
appear AFTER the governance context below). The skill cannot execute until it
produces a fresh review verdict (`clean`, `warnings`, or `blockers`) from
this review. Execution then depends on `skill_review_gate` and the current
review enforcement mode.

## Checklist (source of truth — follow it literally)

{checklist_section}

## Governance context — docs/ARCHITECTURE.md

Use the named sections "Key Invariants", "Host Service, Companion Processes,
and Chat IDs", and "External Skills Layer" as the binding description of what
the skill is allowed to touch. The "Skill gates do not collapse" criterion
keeps executable review, owner grants, dependencies, enablement, and execution
distinct; apply the Skill Review Checklist's `no_repo_mutation` item.

{architecture_text}

## Governance context — docs/DEVELOPMENT.md

Use this as the engineering-standards baseline when judging
``timeout_and_output_discipline`` and when checking whether the skill's
code conforms to the module/function size expectations and the
no-silent-truncation rule for cognitive artifacts.

{development_text}

## Governance context — BIBLE.md

BIBLE.md is Ouroboros' constitutional core. Skills execute inside the
Ouroboros runtime, so a skill that violates a constitutional principle
(for example P0 bounded agency, or P9 version-history limits if the
skill manipulates release metadata) is grounds for FAIL even when the
Skill Review Checklist items permit the behaviour in isolation. Treat
BIBLE.md as the tie-breaker when a skill looks checklist-compliant but
contradicts the runtime's constitutional commitments.

After the first actual review, the author may finish the advisory dialogue for
the exact current content hash.
That author disposition is a separate durable stance beside these raw findings;
it is never a reviewer PASS, never valid for stale content, and never bypasses
deterministic preflight or a blocking enforcement gate.

{bible_text}

{skill_host_context}
"""
    dynamic = f"""\
## Skill identity
- name: {skill_name}
- skill_dir: {skill_dir}
- content_hash: {content_hash}

## Manifest (parsed)
```json
{manifest_dump}
```

## Skill files (every runtime-reachable file in skill_dir, text-only)

{file_pack}

{build_rebuttal_section(review_rebuttal)}
{review_history_section}

## Output contract

Return ONLY a JSON array that covers every checklist item at least once.
Expected items (in order): {items_json}

Each entry MUST have this shape:

{{"item": "<one of the items above>",
  "verdict": "PASS" | "FAIL",
  "severity": "critical" | "advisory",
  "reason": "<why, citing concrete files/lines inside the skill pack>"}}

Rules:

- Every expected item must appear at least once.
- If an item has no problems, return one PASS entry for that item.
- If an item has multiple distinct problems, return one FAIL entry per distinct
  root cause; do not hide additional bugs behind a single summary.
- Do not return a PASS for an item that also has a FAIL. A concrete FAIL wins.
- Do not repeat PASS entries for the same item.
- No prose before or after the JSON array.
- If the skill's ``type`` is not ``extension``, mark
  ``extension_namespace_discipline`` as PASS with reason
  "Not applicable — type != extension".
- Base every critical FAIL on a concrete file/line you can quote from
  the skill pack. Do not invent violations.
- For every FAIL, include a concrete proposed fix (file/symbol/change)
  so the skill author knows how to correct it.
"""
    return stable + "\n" + dynamic, len(stable) + 1


def _build_review_prompt_for_attempt(
    ctx: Any,
    drive_root: pathlib.Path,
    skill: Any,
    *,
    manifest_dump: str,
    content_hash: str,
    file_pack: str,
    history: List[Dict[str, Any]],
    review_rebuttal: str,
) -> tuple[str, int, Dict[str, Any]]:
    accepted_rebuttals = _load_accepted_rebuttals(drive_root, skill.name)
    # Coaching follows the series across payload edits, not identical-byte attempts.
    # Keep _build_review_prompt unchanged: its source binds the free-replay contract.
    attempt_idx = int(
        getattr(ctx, "_skill_review_round", 0)
        or (int(history[-1].get("review_round") or 0) + 1 if history else 1)
    )
    review_history_section = (
        _render_accepted_rebuttals_section(accepted_rebuttals)
        + _build_skill_review_history_section(history, attempt_idx=attempt_idx)
    )
    prompt, stable_prefix_len = _build_review_prompt(
        skill_name=skill.name,
        skill_dir=skill.skill_dir,
        manifest_dump=manifest_dump,
        content_hash=content_hash,
        file_pack=file_pack,
        review_rebuttal=review_rebuttal,
        review_history_section=review_history_section,
    )
    return prompt, stable_prefix_len, {}
