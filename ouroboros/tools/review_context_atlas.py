"""Deterministic bounded repository context for broad review flows."""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from ouroboros.runtime_mode_policy import (
    PROTECTED_RUNTIME_PATH_PREFIXES,
    PROTECTED_RUNTIME_PATHS,
)
from ouroboros.tools.review_helpers import (
    _FULL_REPO_BINARY_EXTENSIONS,
    _FULL_REPO_SKIP_DIR_PREFIXES,
    _MAX_FULL_REPO_FILE_BYTES,
    _SENSITIVE_EXTENSIONS,
    _SENSITIVE_NAMES,
    _VENDORED_NAMES,
    _VENDORED_SUFFIXES,
    _is_probably_binary,
    format_prompt_code_block,
    list_git_tracked_paths,
    redact_prompt_secrets,
)
from ouroboros.utils import estimate_tokens

ATLAS_SCHEMA_VERSION = 1
DEFAULT_ATLAS_TARGET_TOTAL_TOKENS = 850_000
DEFAULT_ATLAS_HARD_TOTAL_TOKENS = 920_000
_ATLAS_HARD_HEADROOM_TOKENS = 5_000

# Dispositions whose per-path rows carry no review-relevant detail: the compact
# coverage index collapses them to one `disposition<TAB>directory/ (N files)`
# row per directory (the durable manifest keeps every per-path row regardless).
_COLLAPSED_INDEX_DISPOSITIONS = frozenset({
    "excluded_test",
    "excluded_dir",
    "binary_media",
    "vendored_minified",
})

_CANONICAL_CONTEXT_DOCS = frozenset({
    "BIBLE.md",
    "docs/DEVELOPMENT.md",
    "docs/DESIGN.md",
    "docs/ARCHITECTURE.md",
    "docs/CHECKLISTS.md",
})

_REVIEW_STACK_PATHS = frozenset({
    "ouroboros/size_ratchet_manifest.py",
    "ouroboros/tools/review.py",
    "ouroboros/tools/review_context_atlas.py",
    "ouroboros/tools/scope_review.py", "ouroboros/tools/scope_review_pack.py", "ouroboros/tools/scope_review_budget.py",
    "ouroboros/tools/preflight_review_prompt.py", "ouroboros/tools/preflight_review_run.py",  # the advisory preflight's own prompt assembly and native-episode run decide what the advisory gate sees  # the scope reviewer's own pack assembly and fit arithmetic decide what the blocking scope gate sees
    "ouroboros/tools/parallel_review.py",
    "ouroboros/tools/review_helpers.py", "ouroboros/tools/review_prompt_text.py", "ouroboros/tools/review_file_pack.py",  # the shared reviewer vocabulary and the packs read from the working tree: what every reviewer is told, and what it is shown
    "ouroboros/tools/review_revalidation.py",
    "ouroboros/tools/claude_advisory_review.py",
    "ouroboros/tools/plan_review.py",
    # v6.87.21 extracted review execution (route vocabulary, transport dispatch,
    # api_chat prompt rendering) below the substrate's seam; the seam module is
    # part of the immune system's review surface and must never be reduced to an
    # ordinary budget-selected dependency in a broad review pack.
    "ouroboros/review_execution.py",
    "ouroboros/tools/plan_review_runtime.py",
    "ouroboros/triad_review.py",
    "ouroboros/review_state.py", "ouroboros/review_state_records.py", "ouroboros/review_state_model.py", "ouroboros/review_state_custody.py",  # the review ledger's record rules, its in-memory transitions and the pending-invocation custody: obligation and attempt lifecycle decide what the gates see
    "ouroboros/review_evidence.py", "ouroboros/review_evidence_sections.py",
    "ouroboros/deep_self_review.py",
})

_FORCE_INCLUDE_PREFIXES = (
    "prompts/",
    "ouroboros/contracts/",
)

_ROUTE_RE = re.compile(r"""['"](/(?:api|ws|owner|static|assets)/[^'"\s{}]+)['"]""")

# BIBLE P3 scope floor: a pack in one of these statuses did NOT assemble. Review
# does not proceed on the remainder — disclosure accompanies the refusal, it does
# not replace it. Every atlas consumer asks `atlas_assembly_failed`; nobody
# re-derives the condition from a status string of its own.
ATLAS_ASSEMBLY_FAILURE_STATUSES = frozenset({"budget_exceeded", "required_artifact_omitted"})

AtlasStatus = Literal[
    "ok", "under_target", "budget_constrained", "budget_exceeded", "required_artifact_omitted"
]


def atlas_assembly_failed(pack: Any) -> bool:
    """True when this pack must NOT be reviewed (see ATLAS_ASSEMBLY_FAILURE_STATUSES)."""
    return str(getattr(pack, "status", "") or "") in ATLAS_ASSEMBLY_FAILURE_STATUSES


# The remedy that matches a missing-REQUIRED-artifact refusal, shared by every
# consumer's refusal wording. Budget knobs do NOT move it: narrowing the reviewed
# change (a smaller diff, a smaller plan, a lower context level) leaves the
# artifact just as required, so either it shrinks or the reviewer window grows.
ATLAS_MISSING_ARTIFACT_REMEDY = (
    "Shrink or split the named artifact(s), or configure a larger-window reviewer — "
    "narrowing the reviewed change cannot help, the artifact is required regardless "
    "of what this change touches."
)

# The remedy for the MIXED assembly failure — required artifact(s) missing AND the
# pack itself (even content-free) overflowing the hard budget. Neither single-cause
# remedy is honest alone: the missing-artifact remedy says narrowing cannot help
# (true for the artifact, false for the overflow), the overflow remedy says split
# the change (inert for the artifact). Shared by every consumer that renders both.
ATLAS_MIXED_ASSEMBLY_REMEDY = (
    "Two causes must BOTH clear: shrink or split the named required artifact(s), AND "
    "free hard-budget room for the pack itself (a smaller change or plan shrinks the "
    "fixed prompt; the atlas manifest scales with the repository, not with this "
    "change). Configuring a larger-window reviewer is the one remedy that resolves both."
)


def atlas_unassembled_required(manifest: Mapping[str, Any] | None) -> tuple[dict, ...]:
    """The required artifacts an assembly failure could not fit, from the manifest SSOT.

    A non-empty result means a REQUIRED artifact is missing. This is NOT an
    exclusive discriminator against the hard-budget overflow: required candidates
    are marked ``budget_omitted`` before the rendered pack is tested against the
    hard budget, so the same pack can ALSO carry ``status="budget_exceeded"`` —
    read that second cause with ``atlas_hard_budget_overflowed``. Consumers word
    their refusal from BOTH facts and render every cause that applies, instead of
    re-deriving the distinction from a status string or asserting the wrong cause.
    """
    rows = dict(manifest or {}).get("unassembled_required") or ()
    return tuple(row for row in rows if isinstance(row, dict))


def atlas_hard_budget_overflowed(manifest: Mapping[str, Any] | None) -> bool:
    """True when the pack itself overflowed the hard budget (``budget_exceeded``).

    Manifest-based so contexts holding only the durable manifest (the scope
    ladder's refusal, plan review's remedy pick) read the same fact as pack
    holders. Can be True TOGETHER with non-empty ``atlas_unassembled_required``:
    that mixed state is two simultaneous assembly causes, not either/or.
    """
    return str(dict(manifest or {}).get("status") or "") == "budget_exceeded"


def atlas_assembly_failure_reason(pack: Any) -> str:
    """Typed sentence(s) naming why the pack did not assemble, for every caller's refusal.

    Renders EVERY cause that applies: a missing required artifact and a hard-budget
    overflow are not mutually exclusive, and suppressing the overflow behind the
    artifact rows would prescribe a remedy that cannot resolve it (P1).
    """
    manifest = dict(getattr(pack, "manifest", None) or {})
    rows = atlas_unassembled_required(manifest)
    tokens = int(manifest.get("estimated_total_tokens") or 0)
    overflow_cause = "generated repository atlas exceeded hard budget" + (
        f" (~{tokens:,} estimated tokens)" if tokens else ""
    )
    if not rows:
        return overflow_cause
    named = "; ".join(
        f"{row.get('path') or '?'} ({row.get('reason') or 'did not fit'})" for row in rows[:5]
    )
    more = f" (+{len(rows) - 5} more)" if len(rows) > 5 else ""
    required_cause = (
        "required artifact could not be assembled into the review pack: "
        f"{named}{more}"
    )
    if atlas_hard_budget_overflowed(manifest):
        # Mixed failure: both causes are real; neither may shadow the other.
        return required_cause + "; AND even the content-free " + overflow_cause
    return required_cause


def atlas_assembly_remedy(manifest: Mapping[str, Any] | None, overflow_remedy: str) -> str:
    """The remedy matching the manifest's cause set — ONE pick for every consumer.

    ``overflow_remedy`` is the caller's surface-specific overflow wording; the
    missing-artifact and mixed remedies are shared, because each single-cause
    remedy states something false for the other cause of the mixed failure.
    """
    missing = bool(atlas_unassembled_required(manifest))
    if missing and atlas_hard_budget_overflowed(manifest):
        return ATLAS_MIXED_ASSEMBLY_REMEDY
    return ATLAS_MISSING_ARTIFACT_REMEDY if missing else overflow_remedy


@dataclass(frozen=True)
class ReviewContextAtlasRequest:
    repo_dir: pathlib.Path
    anchors: tuple[str, ...] = ()
    already_included: frozenset[str] = field(default_factory=frozenset)
    # The subset of ``already_included`` whose CHANGES are in the fixed prompt but
    # whose full snapshot was dropped to fit the budget. Excluded from selection
    # exactly the same way — only the coverage row differs, because a durable
    # manifest that calls a dropped snapshot "included in fixed prompt context" is
    # a false disclosure (P1), not a smaller one.
    diff_only_included: frozenset[str] = field(default_factory=frozenset)
    # Per-path reason for a ``diff_only_included`` row omitted BY DESIGN rather
    # than by budget (the scope pack's span-only release carriers); a path
    # without an entry keeps the budget reason. Same disposition, same
    # selection, truthful row (P1).
    diff_only_reasons: Mapping[str, str] = field(default_factory=dict)
    tracked_paths: tuple[str, ...] = ()
    fixed_prompt_tokens: int = 0
    target_total_tokens: int = DEFAULT_ATLAS_TARGET_TOTAL_TOKENS
    hard_total_tokens: int = DEFAULT_ATLAS_HARD_TOTAL_TOKENS
    include_tests: bool = False
    title: str = "Generated Scope Atlas"
    drive_root: pathlib.Path | None = None
    # Compact coverage is the DEFAULT prompt form (approved with issue #284's
    # pack-arithmetic fixes): the durable manifest keeps full per-file coverage
    # either way, and the full JSON coverage array in the prompt was pure
    # render overhead (~121K chars observed) charged against review content.
    compact_manifest: bool = True
    # Optional additive per-path score bonus (rel_path -> bonus), e.g. import-graph
    # centrality. Default empty = selection identical to the heuristic baseline;
    # scope/plan review never pass it (deep self-review is the only producer).
    # Additive on top of — never replacing — the anchor-relative scoring.
    centrality_scores: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AtlasFileRecord:
    rel_path: str
    disposition: str
    reason: str
    token_count: int = 0
    score: float = 0.0
    sha256: str = ""
    size_bytes: int = 0
    language: str = ""
    symbols: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    routes: tuple[str, ...] = ()
    symbol_count: int = 0
    import_count: int = 0
    js_import_count: int = 0
    route_count: int = 0


@dataclass(frozen=True)
class ReviewContextAtlasPack:
    text: str
    manifest: dict
    selected: tuple[AtlasFileRecord, ...]
    omitted: tuple[AtlasFileRecord, ...]
    token_count: int
    status: AtlasStatus
    # NOTE: the required artifacts the assembler could not fit live in
    # ``manifest["unassembled_required"]`` — read them with
    # ``atlas_unassembled_required``. That dict is the durable, persisted record
    # every consumer already reads; a parallel typed tuple here was a second
    # carrier for one fact, and only a test ever read it.


@dataclass
class _FileFacts:
    rel_path: str
    size_bytes: int = 0
    sha256: str = ""
    language: str = ""
    content: str = ""
    note: str = ""
    token_count: int = 0
    disposition: str = "manifest_only"
    reason: str = ""
    score: float = 0.0
    required: bool = False
    symbols: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    js_imports: tuple[str, ...] = ()
    routes: tuple[str, ...] = ()
    symbol_count: int = 0
    import_count: int = 0
    js_import_count: int = 0
    route_count: int = 0
    reasons: list[str] = field(default_factory=list)


def compile_review_context_atlas(req: ReviewContextAtlasRequest) -> ReviewContextAtlasPack:
    """Build a bounded deterministic repo context pack plus coverage manifest."""
    repo_dir = pathlib.Path(req.repo_dir)
    anchors = frozenset(_normalize_path(path) for path in req.anchors if _normalize_path(path))
    already_included = frozenset(
        _normalize_path(path) for path in req.already_included if _normalize_path(path)
    )
    diff_only_included = frozenset(
        _normalize_path(path) for path in req.diff_only_included if _normalize_path(path)
    )
    # Budget for the RENDERED atlas text (manifest render included); the
    # content-only selection budgets are derived below, after the measured
    # render overhead is known.
    hard_text_tokens = max(
        0,
        int(req.hard_total_tokens)
        - max(0, int(req.fixed_prompt_tokens))
        - _ATLAS_HARD_HEADROOM_TOKENS,
    )

    if req.tracked_paths:
        tracked_paths = [_normalize_path(path) for path in req.tracked_paths]
        inventory_summary = {}
        inventory_by_path: dict[str, Any] = {}
    else:
        try:
            from ouroboros.code_intelligence import build_code_inventory

            inventory = build_code_inventory(repo_dir, drive_root=req.drive_root, persist=True)
            tracked_paths = [_normalize_path(file.path) for file in inventory.files]
            inventory_by_path = {file.path: file for file in inventory.files}
            inventory_summary = {
                "schema_version": inventory.schema_version,
                "git_head": inventory.git_head,
                "file_count": len(inventory.files),
                "coverage": inventory.coverage,
            }
        except Exception as exc:
            try:
                tracked_paths = [_normalize_path(path) for path in list_git_tracked_paths(repo_dir)]
                inventory_summary = {"fallback": "git_ls_files"}
                inventory_by_path = {}
            except Exception as git_exc:
                raise RuntimeError(f"git tracked path inventory unavailable: {exc}; fallback failed: {git_exc}") from git_exc

    facts_by_path = {
        rel: _build_file_facts(
            repo_dir,
            rel,
            anchors,
            already_included,
            req.include_tests,
            diff_only_included=diff_only_included,
            diff_only_reasons=req.diff_only_reasons,
            inventory_fact=inventory_by_path.get(rel),
        )
        for rel in tracked_paths
        if rel
    }
    _score_relationships(facts_by_path, anchors)
    # Optional graph-centrality bonus (deep self-review only; empty for
    # scope/plan). Strictly additive so anchor-relative scoring is untouched.
    if req.centrality_scores:
        for rel, facts in facts_by_path.items():
            bonus = float(req.centrality_scores.get(rel) or 0.0)
            if bonus > 0.0:
                facts.score += bonus
                facts.reasons.append("graph_centrality")

    selected_paths: list[str] = []
    used_tokens = 0

    # The rendered manifest/index skeleton is part of the pack, so it is
    # MEASURED and charged against the selection budgets up front. The old
    # blind 30K reserve let a real render (observed ~121K chars on this
    # repository) dwarf it, so the greedy loop admitted content it could not
    # keep and the shrink waves then evicted already-charged (even required)
    # content — an admit-then-evict failure the arithmetic itself caused.
    base_render_tokens = estimate_tokens(
        _render_atlas_text(req, facts_by_path, [], status_hint="")
    )
    hard_context_tokens = max(0, hard_text_tokens - base_render_tokens)
    # The target admission budget can never exceed what the hard rail will
    # keep: when target-vs-hard gap is smaller than the hard headroom (small
    # configured budgets), an uncapped target admits content the shrink waves
    # must then evict — the same admit-then-evict failure by another route.
    target_context_tokens = min(
        hard_context_tokens,
        max(
            0,
            int(req.target_total_tokens)
            - max(0, int(req.fixed_prompt_tokens))
            - base_render_tokens,
        ),
    )

    candidates = [
        facts
        for facts in facts_by_path.values()
        if facts.disposition == "manifest_only" and facts.content
    ]
    candidates.sort(
        key=lambda item: (
            not item.required,
            -item.score,
            item.token_count,
            item.rel_path,
        )
    )

    for facts in candidates:
        limit = hard_context_tokens if facts.required else target_context_tokens
        # The admission cost is the EXACT bytes selection adds to the render:
        # the file section (header + metadata + fenced content) PLUS this
        # file's JSON row in the manifest preview's `selected` array — the
        # bare content estimate undercounted both, which was the second half
        # of the admit-then-evict arithmetic failure.
        # The preview pretty-prints the row (indent=2 inside the `selected`
        # array); charging the same form (plus a small list-punctuation pad)
        # keeps admission a slight OVERcharge, never an undercharge.
        row_cost = estimate_tokens(_render_file_content(facts)) + estimate_tokens(
            json.dumps(_manifest_row(facts), ensure_ascii=False, indent=2)
        ) + 4
        if used_tokens + row_cost <= limit:
            facts.disposition = "full"
            facts.reason = ", ".join(facts.reasons) or "selected"
            selected_paths.append(facts.rel_path)
            used_tokens += row_cost
            continue
        if facts.required:
            # BIBLE P3 scope floor: a REQUIRED artifact that does not fit is a
            # failure to ASSEMBLE, not a smaller pack. The row records artifact +
            # reason (disclosure, P1); the pack status refuses the review. The
            # reason states what actually happened: the REMAINDER was too small,
            # not necessarily the file alone against the whole budget.
            facts.disposition = "budget_omitted"
            facts.reason = (
                "required file does not fit the atlas hard budget: needs "
                f"{row_cost:,} tokens rendered, {max(0, limit - used_tokens):,} remain "
                "after higher-priority content and the rendered manifest"
            )
        else:
            facts.disposition = "manifest_only"
            facts.reason = "not selected within atlas target budget"

    text = _render_atlas_text(req, facts_by_path, selected_paths, status_hint="")
    token_count = estimate_tokens(text)

    if token_count > hard_text_tokens:
        # Backstop shrink waves for residual estimator drift (per-file estimates
        # vs the joined render; the selected-rows part of the manifest preview):
        # non-required content first, then required content (largest first) — the
        # atlas always converges to at worst a manifest-only pack instead of
        # giving up with budget_exceeded. Dropping a REQUIRED artifact here is
        # the same assembly failure as above, not a legal degradation: identical
        # disposition, identical refusal. The reasons say what actually
        # happened — admitted, then evicted because the RENDER overflowed — so
        # the row cannot be read as "this file alone exceeded the budget".
        removable = [path for path in reversed(selected_paths) if not facts_by_path[path].required]
        removable += sorted(
            (path for path in selected_paths if facts_by_path[path].required),
            key=lambda path: -facts_by_path[path].token_count,
        )
        for path in removable:
            facts = facts_by_path[path]
            if facts.required:
                facts.disposition = "budget_omitted"
                facts.reason = (
                    "required file admitted, then removed because the rendered "
                    "atlas exceeded the hard budget"
                )
            else:
                facts.disposition = "manifest_only"
                facts.reason = (
                    "admitted, then removed because the rendered atlas exceeded "
                    "the hard budget"
                )
            selected_paths.remove(path)
            text = _render_atlas_text(req, facts_by_path, selected_paths, status_hint="")
            token_count = estimate_tokens(text)
            if token_count <= hard_text_tokens:
                break

    target_text_tokens = max(0, int(req.target_total_tokens) - max(0, int(req.fixed_prompt_tokens)))
    if token_count > target_text_tokens:
        removable = [path for path in reversed(selected_paths) if not facts_by_path[path].required]
        for path in removable:
            facts = facts_by_path[path]
            facts.disposition = "manifest_only"
            facts.reason = (
                "admitted, then removed to keep the assembled prompt within "
                "the atlas target"
            )
            selected_paths.remove(path)
            text = _render_atlas_text(req, facts_by_path, selected_paths, status_hint="")
            token_count = estimate_tokens(text)
            if token_count <= target_text_tokens:
                break

    # budget_exceeded survives ONLY when even the content-free atlas (manifest
    # alone) cannot fit the hard budget. An omitted REQUIRED artifact is the
    # other assembly failure: the pack is disclosed but never reviewable. The two
    # causes are NOT exclusive — required candidates were marked budget_omitted
    # above, before this overflow test — so a budget_exceeded pack keeps its
    # `unassembled_required` rows (the disclosure survives) and consumers read
    # BOTH facts: rows via atlas_unassembled_required, overflow via
    # atlas_hard_budget_overflowed. The status ranks the overflow first only
    # because a pack that cannot even render is the harder failure.
    unassembled = _unassembled_required(facts_by_path)
    if token_count > hard_text_tokens:
        status: AtlasStatus = "budget_exceeded"
    elif unassembled:
        status = "required_artifact_omitted"
    elif any(
        facts.disposition == "manifest_only" and facts.content
        for facts in facts_by_path.values()
    ):
        status = "budget_constrained"
    elif int(req.fixed_prompt_tokens) + token_count < int(req.target_total_tokens):
        status = "under_target"
    else:
        status = "ok"

    text = _render_atlas_text(req, facts_by_path, selected_paths, status_hint=status)
    token_count = estimate_tokens(text)
    manifest = _build_manifest(req, facts_by_path, selected_paths, token_count, status, unassembled)
    manifest["code_inventory"] = inventory_summary
    selected = tuple(_record_for(facts_by_path[path]) for path in selected_paths)
    omitted = tuple(
        _record_for(facts)
        for facts in facts_by_path.values()
        if facts.rel_path not in selected_paths
    )
    return ReviewContextAtlasPack(
        text=text,
        manifest=manifest,
        selected=selected,
        omitted=omitted,
        token_count=token_count,
        status=status,
    )


def _unassembled_required(facts_by_path: dict[str, _FileFacts]) -> list[_FileFacts]:
    """Required artifacts the assembler could not fit — the assembly-failure SSOT."""
    return [
        facts
        for facts in facts_by_path.values()
        if facts.required and facts.disposition == "budget_omitted"
    ]


def _build_file_facts(
    repo_dir: pathlib.Path,
    rel: str,
    anchors: frozenset[str],
    already_included: frozenset[str],
    include_tests: bool,
    diff_only_included: frozenset[str] = frozenset(),
    diff_only_reasons: Mapping[str, str] | None = None,
    inventory_fact: Any = None,
) -> _FileFacts:
    facts = _FileFacts(rel_path=rel, language=pathlib.PurePosixPath(rel).suffix.lstrip("."))
    if inventory_fact is not None:
        facts.size_bytes = int(getattr(inventory_fact, "size", 0) or 0)
        digest = str(getattr(inventory_fact, "sha256", "") or "")
        facts.sha256 = digest[:16]
        facts.language = str(getattr(inventory_fact, "language", "") or facts.language)
    path = repo_dir / rel
    # BIBLE P3 / D17: requiredness is a property of the path and the request,
    # decided HERE — before any classification below can drop the artifact.
    # Every early return sees the same answer; no branch re-derives it.
    force_include = _is_force_include(rel)
    is_anchor = rel in anchors
    is_canonical = rel in _CANONICAL_CONTEXT_DOCS
    facts.required = force_include or is_anchor or is_canonical
    # The classes owed IN FULL regardless of the change: for these the staged
    # diff is never a substitute for the artifact. An anchor is required
    # BECAUSE it is touched, and its complete change-evidence is the staged
    # diff itself — so merely-touched files may legally degrade to diff-only
    # (the ladder's disclosed step), while these classes may not.
    required_beyond_diff = atlas_required_beyond_diff(rel)
    if rel in already_included or rel in diff_only_included:
        facts.disposition = "already_included"
        # The caller owns which snapshots actually survived in the fixed prompt;
        # the row must not claim more than it was told (P1).
        facts.reason = (
            (diff_only_reasons or {}).get(rel)
            or "changes included in the fixed staged diff; full snapshot omitted "
            "to fit the reviewer input budget"
            if rel in diff_only_included
            else "included in fixed prompt context"
        )

    try:
        resolved = path.resolve()
        resolved.relative_to(repo_dir.resolve())
    except (OSError, ValueError):
        facts.disposition = "path_escape"
        facts.reason = "path escapes repository root"
        return facts

    if not path.is_file():
        if facts.disposition != "already_included":
            facts.disposition = "missing"
            facts.reason = "tracked path is not a regular file"
        return facts

    try:
        raw = path.read_bytes()
        facts.size_bytes = len(raw)
        facts.sha256 = hashlib.sha256(raw).hexdigest()[:16]
    except OSError as exc:
        facts.disposition = "read_error"
        facts.reason = f"read error: {exc}"
        return facts

    if facts.disposition == "already_included":
        if rel in diff_only_included and required_beyond_diff:
            # BIBLE P3 scope floor: dropping the full snapshot of an artifact
            # the reviewer is owed in full is a failure to ASSEMBLE, not a
            # disclosed degradation — same typed disposition, same single
            # predicate (`_unassembled_required`) as every other required drop.
            facts.disposition = "budget_omitted"
            facts.reason = (
                "required artifact degraded to diff-only: full snapshot "
                "dropped to fit the reviewer input budget"
            )
        return facts

    suffix = pathlib.PurePosixPath(rel).suffix.lower()
    fname = pathlib.PurePosixPath(rel).name.lower()
    if rel.startswith("tests/") and not include_tests and not force_include and not is_anchor:
        # Anchors are exempt exactly like in the directory exclusion below:
        # tests are excludable only when UNRELATED to the change (BIBLE P3),
        # and an anchor is related by definition.
        facts.disposition = "excluded_test"
        facts.reason = "wider tests excluded by atlas policy"
        return facts
    if _skip_by_dir(rel) and not is_anchor and not force_include and not (include_tests and rel.startswith("tests/")):
        facts.disposition = "excluded_dir"
        facts.reason = "excluded non-agent-logic directory"
        return facts
    if fname in _SENSITIVE_NAMES or suffix in _SENSITIVE_EXTENSIONS:
        facts.disposition = "sensitive"
        facts.reason = "sensitive filename or extension"
        facts.size_bytes = 0
        facts.sha256 = ""
        return facts
    if suffix in _FULL_REPO_BINARY_EXTENSIONS:
        facts.disposition = "binary_media"
        facts.reason = "binary/media extension"
        return facts
    if fname in _VENDORED_NAMES or any(fname.endswith(suffix_) for suffix_ in _VENDORED_SUFFIXES):
        facts.disposition = "vendored_minified"
        facts.reason = "vendored or minified file"
        return facts
    if facts.size_bytes > _MAX_FULL_REPO_FILE_BYTES:
        if facts.required:
            # BIBLE P3: a required artifact over the per-file cap cannot be
            # assembled — a typed failure, never a silent coverage row.
            facts.disposition = "budget_omitted"
            facts.reason = (
                "required file exceeds the per-file atlas cap "
                f"(>{_MAX_FULL_REPO_FILE_BYTES // 1024}KB)"
            )
        else:
            facts.disposition = "oversized"
            facts.reason = f">{_MAX_FULL_REPO_FILE_BYTES // 1024}KB"
        return facts
    if _is_probably_binary(path):
        facts.disposition = "binary_media"
        facts.reason = "binary content"
        return facts

    try:
        content = raw.decode("utf-8", errors="replace")
    except Exception as exc:
        facts.disposition = "read_error"
        facts.reason = f"text decode error: {exc}"
        return facts

    content, redacted = redact_prompt_secrets(content)
    facts.content = content
    facts.note = "*(secret-like content redacted)*\n" if redacted else ""
    if inventory_fact is not None and str(getattr(inventory_fact, "language", "") or "") == "python":
        inventory_symbols = list(getattr(inventory_fact, "symbols", []) or [])
        inventory_imports = list(getattr(inventory_fact, "imports", []) or [])
        facts.symbols = tuple(str(getattr(symbol, "name", "") or "") for symbol in inventory_symbols[:16] if str(getattr(symbol, "name", "") or ""))
        facts.imports = tuple(str(item) for item in inventory_imports[:24] if str(item))
        facts.symbol_count = len({str(getattr(symbol, "name", "") or "") for symbol in inventory_symbols if str(getattr(symbol, "name", "") or "")})
        facts.import_count = len(set(str(item) for item in inventory_imports if str(item)))
    else:
        facts.symbols, facts.imports, facts.symbol_count, facts.import_count = _extract_python_facts(rel, content)
    facts.js_imports, facts.js_import_count = _extract_js_imports(rel, content)
    inventory_routes = list(getattr(inventory_fact, "routes", []) or []) if inventory_fact is not None else []
    routes = sorted(set(str(route) for route in inventory_routes if str(route))) or sorted(set(_ROUTE_RE.findall(content)))
    facts.routes = tuple(routes[:12])
    facts.route_count = len(routes)
    facts.token_count = estimate_tokens(_render_file_content(facts))

    # Scoring reads the SAME hoisted flags that own `facts.required` above —
    # requiredness has exactly one computation in this function.
    if force_include:
        facts.score += 10_000
        facts.reasons.append("protected_or_review_surface")
    if is_anchor:
        facts.score += 9_000
        facts.reasons.append("anchor")
    if is_canonical:
        facts.score += 8_000
        facts.reasons.append("canonical_context_doc")
    if rel.startswith("ouroboros/"):
        facts.score += 200
    if rel.startswith("web/"):
        facts.score += 120
    if rel.startswith("docs/"):
        facts.score += 80
    facts.score += min(len(facts.imports) + len(facts.js_imports), 30) * 10
    facts.reason = "eligible text file"
    return facts


def _score_relationships(facts_by_path: dict[str, _FileFacts], anchors: frozenset[str]) -> None:
    if not anchors:
        return
    module_to_path = {}
    for path, facts in facts_by_path.items():
        module = _python_module_name(path)
        if module:
            module_to_path[module] = path

    anchor_modules = {
        module
        for path in anchors
        for module in (_python_module_name(path),)
        if module
    }
    anchor_dirs = {str(pathlib.PurePosixPath(path).parent) for path in anchors}
    anchor_names = {pathlib.PurePosixPath(path).name for path in anchors}

    for facts in facts_by_path.values():
        if facts.disposition != "manifest_only" or not facts.content:
            continue
        parent = str(pathlib.PurePosixPath(facts.rel_path).parent)
        if parent in anchor_dirs:
            facts.score += 600
            facts.reasons.append("same_directory_as_anchor")
        imported_paths = {
            module_to_path.get(imported)
            for imported in facts.imports
            if module_to_path.get(imported)
        }
        if imported_paths & anchors:
            facts.score += 1_200
            facts.reasons.append("imports_anchor")
        if set(facts.imports) & anchor_modules:
            facts.score += 1_000
            facts.reasons.append("imports_anchor_module")
        if any(name in facts.content for name in anchor_names):
            facts.score += 300
            facts.reasons.append("mentions_anchor_path")

    for anchor in anchors:
        anchor_facts = facts_by_path.get(anchor)
        if not anchor_facts:
            continue
        for imported in anchor_facts.imports:
            imported_path = module_to_path.get(imported)
            if imported_path and imported_path in facts_by_path:
                facts = facts_by_path[imported_path]
                if facts.disposition == "manifest_only" and facts.content:
                    facts.score += 1_200
                    facts.reasons.append("imported_by_anchor")
        for js_path in anchor_facts.js_imports:
            if js_path in facts_by_path:
                facts = facts_by_path[js_path]
                if facts.disposition == "manifest_only" and facts.content:
                    facts.score += 1_200
                    facts.reasons.append("js_imported_by_anchor")


def _extract_python_facts(
    rel: str,
    content: str,
) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
    if not rel.endswith(".py"):
        return (), (), 0, 0
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return (), (), 0, 0
    symbols: list[str] = []
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    symbols.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id.isupper():
            symbols.append(node.target.id)
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import_from_module(rel, node)
            if module:
                imports.append(module)
    unique_symbols = sorted(set(symbols))
    unique_imports = sorted(set(imports))
    return tuple(unique_symbols[:16]), tuple(unique_imports[:24]), len(unique_symbols), len(unique_imports)


def _extract_js_imports(rel: str, content: str) -> tuple[tuple[str, ...], int]:
    if not rel.endswith((".js", ".mjs", ".ts", ".tsx", ".jsx")):
        return (), 0
    parent = pathlib.PurePosixPath(rel).parent
    found: set[str] = set()
    from ouroboros.code_intelligence import extract_js_imports

    for spec in extract_js_imports(content):
        if not spec.startswith("."):
            continue
        base = (parent / spec).as_posix()
        candidates = [base]
        if not pathlib.PurePosixPath(base).suffix:
            candidates.extend([base + ".js", base + ".mjs", base + "/index.js"])
        found.update(candidates)
    sorted_found = sorted(found)
    return tuple(sorted_found[:24]), len(sorted_found)


def _python_module_name(rel: str) -> str:
    if not rel.endswith(".py"):
        return ""
    path = rel[:-3]
    if path.endswith("/__init__"):
        path = path[: -len("/__init__")]
    return path.replace("/", ".")


def _resolve_import_from_module(rel: str, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    current = _python_module_name(rel)
    package_parts = current.split(".")[:-1] if current else []
    base = package_parts[: max(0, len(package_parts) - (node.level - 1))]
    if node.module:
        return ".".join([*base, *node.module.split(".")])
    if node.names:
        return ".".join([*base, node.names[0].name])
    return ".".join(base)


def _is_force_include(rel: str) -> bool:
    return (
        rel in PROTECTED_RUNTIME_PATHS
        or rel in _REVIEW_STACK_PATHS
        or any(rel.startswith(prefix) for prefix in PROTECTED_RUNTIME_PATH_PREFIXES)
        or any(rel.startswith(prefix) for prefix in _FORCE_INCLUDE_PREFIXES)
    )


def atlas_required_beyond_diff(rel: str) -> bool:
    """True when the staged diff is never a substitute for this artifact.

    The ONE definition of the class owed in full (BIBLE P3 scope floor), so the
    ladder that chooses what to degrade and the assembler that refuses a
    degraded required artifact cannot drift apart.
    """
    return _is_force_include(rel) or rel in _CANONICAL_CONTEXT_DOCS


def _skip_by_dir(rel: str) -> bool:
    return rel.startswith(_FULL_REPO_SKIP_DIR_PREFIXES)


def _render_file_content(facts: _FileFacts) -> str:
    return (
        f"### {facts.rel_path}\n"
        f"sha256={facts.sha256} size={facts.size_bytes}B"
        + (
            f" symbols={', '.join(facts.symbols[:8])}"
            + (f" (+{facts.symbol_count - len(facts.symbols)} more)" if facts.symbol_count > len(facts.symbols) else "")
            if facts.symbols else ""
        )
        + (
            f" imports={', '.join(facts.imports[:8])}"
            + (f" (+{facts.import_count - len(facts.imports)} more)" if facts.import_count > len(facts.imports) else "")
            if facts.imports else ""
        )
        + "\n"
        f"{facts.note}{format_prompt_code_block(facts.content, facts.language)}\n\n"
    )


def _render_atlas_text(
    req: ReviewContextAtlasRequest,
    facts_by_path: dict[str, _FileFacts],
    selected_paths: list[str],
    *,
    status_hint: str,
) -> str:
    counts = Counter(facts.disposition for facts in facts_by_path.values())
    selected = [facts_by_path[path] for path in selected_paths]
    coverage_rows = [_manifest_row(facts) for facts in facts_by_path.values()]
    manifest_preview = {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "strategy": "generated_scope_atlas",
        "status": status_hint or "building",
        "fixed_prompt_tokens": int(req.fixed_prompt_tokens),
        "target_total_tokens": int(req.target_total_tokens),
        "hard_total_tokens": int(req.hard_total_tokens),
        "selected_count": len(selected),
        "tracked_count": len(facts_by_path),
        "dispositions": dict(sorted(counts.items())),
        "selected": [_manifest_row(facts) for facts in selected],
    }
    if req.compact_manifest:
        manifest_preview["compact_manifest"] = True
        manifest_preview["coverage_in_prompt"] = "compact_full_index_plus_bounded_samples"
        manifest_preview["coverage_note"] = (
            "Full per-file coverage is preserved in the scope review context_manifest; "
            "the prompt includes a compact full path/disposition index plus bounded "
            "per-disposition samples instead of the full JSON coverage array."
        )
        sample_buckets: dict[str, list[dict]] = {}
        for row in coverage_rows:
            disposition = str(row.get("disposition") or "unknown")
            bucket = sample_buckets.setdefault(disposition, [])
            if len(bucket) < 8:
                bucket.append(row)
        samples = dict(sorted(sample_buckets.items()))
        manifest_preview["coverage_samples"] = samples
        manifest_preview["coverage_sample_counts"] = {
            key: len(value) for key, value in samples.items()
        }
        manifest_preview["coverage_index_count"] = len(coverage_rows)
    else:
        manifest_preview["coverage"] = coverage_rows
    parts = [
        f"## {req.title}",
        "",
        "This is a generated, deterministic, bounded repository atlas. It replaces the raw full-repo pack for this review flow. Summaries here are structural facts only; they are not LLM-generated claims.",
        (
            "NOTE: The atlas prompt uses compact coverage mode (the default): a "
            "full path/disposition index with collapsed excluded groups plus "
            "bounded per-disposition samples. Full per-file coverage remains "
            "available in the durable review context_manifest."
            if req.compact_manifest else ""
        ),
        "",
        "### Atlas token accounting",
        "",
        f"- fixed_prompt_tokens: {int(req.fixed_prompt_tokens):,}",
        f"- target_total_tokens: {int(req.target_total_tokens):,}",
        f"- hard_total_tokens: {int(req.hard_total_tokens):,}",
        f"- selected_full_files: {len(selected):,}",
        f"- tracked_files_accounted: {len(facts_by_path):,}",
        "",
        "### Coverage manifest",
        "",
        format_prompt_code_block(json.dumps(manifest_preview, ensure_ascii=False, indent=2), "json"),
        "",
    ]
    if req.compact_manifest:
        detailed_rows: list[tuple[str, str]] = []
        collapsed_groups: Counter[tuple[str, str]] = Counter()
        for row in coverage_rows:
            disposition = str(row.get("disposition") or "unknown")
            path = str(row.get("path") or "")
            if disposition in _COLLAPSED_INDEX_DISPOSITIONS:
                parent = str(pathlib.PurePosixPath(path).parent)
                collapsed_groups[(disposition, "" if parent == "." else parent)] += 1
            else:
                detailed_rows.append((disposition, path))
        index_lines = [f"{disposition}\t{path}" for disposition, path in sorted(detailed_rows)]
        index_lines += [
            f"{disposition}\t{directory}/ ({count} files)"
            for (disposition, directory), count in sorted(collapsed_groups.items())
        ]
        parts.extend([
            "### Compact full coverage index",
            "",
            (
                "Every tracked path appears below as `disposition<TAB>path`; "
                "policy-excluded classes (excluded_test, excluded_dir, "
                "binary_media, vendored_minified) are collapsed to one "
                "`disposition<TAB>directory/ (N files)` row per directory. "
                "Detailed hashes, sizes, symbols, and imports remain in the "
                "durable context_manifest."
            ),
            "",
            format_prompt_code_block("\n".join(index_lines), "text"),
            "",
        ])
    parts.extend([
        "### Atlas full file contents",
        "",
    ])
    if selected:
        parts.extend(_render_file_content(facts) for facts in selected)
    else:
        parts.append("(no additional files selected for full atlas context)\n")
    return "\n".join(parts)


def _build_manifest(
    req: ReviewContextAtlasRequest,
    facts_by_path: dict[str, _FileFacts],
    selected_paths: list[str],
    token_count: int,
    status: str,
    unassembled: list[_FileFacts],
) -> dict:
    counts = Counter(facts.disposition for facts in facts_by_path.values())
    return {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "strategy": "generated_scope_atlas",
        "status": status,
        "fixed_prompt_tokens": int(req.fixed_prompt_tokens),
        "atlas_tokens": int(token_count),
        "estimated_total_tokens": int(req.fixed_prompt_tokens) + int(token_count),
        "target_total_tokens": int(req.target_total_tokens),
        "hard_total_tokens": int(req.hard_total_tokens),
        "selected_count": len(selected_paths),
        "tracked_count": len(facts_by_path),
        "dispositions": dict(sorted(counts.items())),
        "selected": [_manifest_row(facts_by_path[path]) for path in selected_paths],
        "coverage": [_manifest_row(facts) for facts in facts_by_path.values()],
        "compact_manifest_in_prompt": bool(req.compact_manifest),
        # Typed assembly-failure record (BIBLE P3): named here rather than left
        # for a reader to rediscover among ~900 coverage rows. This dict is the
        # ONE carrier — read it with ``atlas_unassembled_required``.
        "unassembled_required": [_manifest_row(facts) for facts in unassembled],
    }


def _manifest_row(facts: _FileFacts) -> dict:
    row = {
        "path": facts.rel_path,
        "disposition": facts.disposition,
        "reason": facts.reason or ", ".join(facts.reasons),
        "sha256": facts.sha256,
        "size": facts.size_bytes,
        "tokens": facts.token_count,
    }
    if facts.symbols:
        row["symbols"] = list(facts.symbols[:12])
        row["symbols_total"] = facts.symbol_count
    if facts.imports:
        row["imports"] = list(facts.imports[:12])
        row["imports_total"] = facts.import_count
    if facts.js_imports:
        row["js_imports"] = list(facts.js_imports[:12])
        row["js_imports_total"] = facts.js_import_count
    if facts.routes:
        row["routes"] = list(facts.routes[:12])
        row["routes_total"] = facts.route_count
    return row


def _record_for(facts: _FileFacts) -> AtlasFileRecord:
    return AtlasFileRecord(
        rel_path=facts.rel_path,
        disposition=facts.disposition,
        reason=facts.reason or ", ".join(facts.reasons),
        token_count=facts.token_count,
        score=facts.score,
        sha256=facts.sha256,
        size_bytes=facts.size_bytes,
        language=facts.language,
        symbols=facts.symbols,
        imports=facts.imports,
        routes=facts.routes,
        symbol_count=facts.symbol_count,
        import_count=facts.import_count,
        js_import_count=facts.js_import_count,
        route_count=facts.route_count,
    )


def _normalize_path(path: str) -> str:
    cleaned = str(path or "").strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return pathlib.PurePosixPath(cleaned).as_posix() if cleaned else ""
