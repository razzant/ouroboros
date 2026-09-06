"""Deep self-review of the whole Ouroboros system against BIBLE.md.

The review runs on the configured ``deep_review`` reviewer row
(``reviewer_slot_config.deep_review_slot``) and, like every other review
surface, has THREE deliveries chosen by the row's ``retrieves`` predicate:

* a direct ``api_chat`` row is the historical PACKED review — one 1M-context
  call carrying the Generated Deep Self-Review Atlas plus the full memory
  whitelist, assembled fail-closed (a required artifact that does not fit is
  a refusal, never a smaller pack);
* a configured-subagent api row is a NATIVE inspection episode — the reviewer
  reads the repository through the host's read-only tools (the runtime root
  is its readable data plane) while the memory whitelist reaches it inline
  byte-exact; every repository read is host-observed and the mandatory
  BIBLE.md read is checked against the receipts afterwards;
* an ``agent_session`` row is a delegated read-only session — the same task,
  reads not host-observed (disclosed as ``unobserved``).

The retrieving deliveries ride the shared executor seam
(``review_execution._review_route_executor``) exactly like the advisory: the
product is free markdown (``triad_review`` shape ``report``), a bound landing
before the final answer delivers the collected draft marked INCOMPLETE, and
the host prepends a provenance header naming the delivery, model, rounds,
receipts, coverage and completeness so consecutive reports stay comparable.
"""

from __future__ import annotations

import logging
import os
import pathlib
import posixpath
import time
from typing import Any, Callable, Dict, Optional, Tuple

log = logging.getLogger(__name__)

# Pack filtering is shared with scope review.
from ouroboros.tools.review_context_atlas import (  # noqa: E402
    ReviewContextAtlasRequest,
    atlas_assembly_failed,
    atlas_assembly_failure_reason,
    compile_review_context_atlas,
)
from ouroboros.tools.review_helpers import (  # noqa: E402
    _MAX_FULL_REPO_FILE_BYTES,
    REVIEW_PROMPT_TOKEN_BUDGET,
    calibrated_input_token_limit,
    density_probe_sample,
    load_governance_doc,
)
from ouroboros.shell_parse import is_absolute_path_text  # noqa: E402
from ouroboros.utils import atomic_write_json, estimate_tokens, utc_now_iso  # noqa: E402
from ouroboros.config import get_context_mode  # noqa: E402
from ouroboros.provider_models import provider_for_model, provider_has_credentials  # noqa: E402
from ouroboros.context_layout import generate_doc_nav_map  # noqa: E402
from ouroboros.reviewer_slot_config import (  # noqa: E402
    ROUTE_KIND_API,
    ROUTE_KIND_SESSION,
    ConfiguredReviewerSlot,
    deep_review_slot,
    row_effort,
)
from ouroboros.usage_accounting import BudgetExceeded  # noqa: E402
from ouroboros.outcomes import REASON_DEEP_SELF_REVIEW_PACK_UNFIT  # noqa: E402
from ouroboros.triad_review import REVIEW_REPORT_CONTRACT  # noqa: E402

# Output reservation inside the reviewer's 1M window (same class of fix as
# scope_review._SCOPE_INPUT_TOKEN_LIMIT): 920K input + 100K output exceeds 1M
# and yields a deterministic provider 400, so the assembled INPUT prompt is
# gated on min(SSOT budget, window − output − tokenizer margin).
_DEEP_MAX_OUTPUT_TOKENS = 100_000
_DEEP_MODEL_CONTEXT_WINDOW = 1_000_000
_DEEP_OUTPUT_MARGIN_TOKENS = 155_000
# The cold-start density probe (R60/R61) is the shared rung
# ``capability_evidence.cold_start_density_probe``; the commit gate runs the same one.
_DEEP_INPUT_TOKEN_LIMIT = min(
    REVIEW_PROMPT_TOKEN_BUDGET,
    _DEEP_MODEL_CONTEXT_WINDOW - _DEEP_MAX_OUTPUT_TOKENS - _DEEP_OUTPUT_MARGIN_TOKENS,
)

_MEMORY_WHITELIST = [
    "memory/identity.md",
    "memory/scratchpad.md",
    "memory/registry.md",
    "memory/WORLD.md",
    "memory/knowledge/index-full.md",
    "memory/knowledge/patterns.md",
    "memory/knowledge/improvement-backlog.md",
]

# The omission section is appended to the pack AFTER the atlas has filled its
# budget, so it must be (a) bounded and (b) reserved inside atlas_fixed_tokens.
# An unbounded per-file listing here is exactly what historically overflowed the
# assembled prompt past the final gate by a few hundred tokens (the atlas filled
# to its ceiling, then the uncounted omission listing was appended on top).
_OMISSION_SECTION_RESERVE_TOKENS = 2_000
_OMISSION_SAMPLE_MAX_ENTRIES = 40

# Bonus scale for graph-centrality ranking (D2). Bounded well below the atlas's
# force/anchor/canonical tiers (10000/9000/8000) so protected and governance
# surfaces always outrank a merely well-connected module; meaningfully above the
# generic path-prefix bonuses (~200) so hub modules win among peers.
_CENTRALITY_MAX_BONUS = 600.0
_CENTRALITY_PER_IMPORTER = 30.0

# The role half of the reviewer prompt is shared by every delivery; only the
# "how to work" half differs (a pack to read vs tools to read with).
_ROLE_PROMPT = """\
You are conducting a deep self-review of the Ouroboros project — a self-creating AI agent.

Primary directive: The Constitution (BIBLE.md) is your absolute reference.
Every finding must be checked against it.

What to look for: bugs, crashes, race conditions,
BIBLE.md violations (P0–P12), contradictions between code and docs,
security gaps, dead code, missing error handling, architectural issues,
known error patterns from patterns.md that remain unfixed, and ideas how to improve Ouroboros to work better and better comply with the Bible."""

# The PACKED system prompt — byte-identical to the pre-row deep review (pinned
# by a golden digest test): the packed delivery's wire payload is unchanged.
_SYSTEM_PROMPT = _ROLE_PROMPT + """

How to work: Use the generated atlas coverage manifest systematically. Raw code is
included for selected functional/protected surfaces; every tracked file is still
accounted for by hash, size, classification, and omission/manifest disposition.
Cross-reference interactions between modules. Prioritize: CRITICAL > IMPORTANT > ADVISORY.

Output: Structured report with prioritized findings, each citing the
specific file, line/section, the problem, and the proposed fix."""

# The RETRIEVING task: the reviewer reads the repository itself. BIBLE.md is a
# mandatory full read (host-checked on the native delivery), the memory files
# ride inline byte-exact, the three big docs are navigation maps read on demand.
_RETRIEVING_METHOD = """

How to work: you are reading the repository yourself with read-only tools. Read
`BIBLE.md` IN FULL first (about {bible_chars:,} chars — in bounded chunks): every
finding is checked against it, and a report that did not read it is not a deep
self-review. The memory files below are inlined byte-exact; `docs/ARCHITECTURE.md`,
`docs/DEVELOPMENT.md` and `docs/CHECKLISTS.md` are given as navigation maps — read the
sections you need on demand. Then inspect the code (search_code, query_code,
read_file), cross-reference interactions between modules and follow call chains out
of the files you open. Prioritize: CRITICAL > IMPORTANT > ADVISORY.

Output: Structured markdown report, MOST CRITICAL findings first, each citing the
specific file, line/section, the problem, and the proposed fix. Begin with a one-line
coverage header naming what you actually read (documents and files, in full or by
section) and what you did not — your host records only the reads it observed."""

# The report contract for a retrieving row: the shared report shape plus the
# deep review's own coverage-header sentence. The SHAPE is already `report`
# through REVIEW_OUTPUT_SHAPES (the executors' default contract for it is the
# report contract, never an array); the policy hands over the same contract
# WITH the deep-review sentence, so the ask and the parse cannot disagree.
_REPORT_CONTRACT = REVIEW_REPORT_CONTRACT + (
    "Begin with one line naming what you read (in full or by section) and what you did "
    "not; the rest is the prioritized report."
)

_MANDATORY_READS = ("BIBLE.md",)
# The inspection roots that resolve to the REPOSITORY (both name the review's
# session root); a read under the data plane never satisfies a repository read.
_REPO_ROOTS = frozenset({"", "active_workspace", "system_repo"})
_NAV_MAP_DOCS = ("docs/ARCHITECTURE.md", "docs/DEVELOPMENT.md", "docs/CHECKLISTS.md")


def _dulwich_tracked_paths(repo_dir: pathlib.Path) -> tuple[list[str], list[str]]:
    """Return git-tracked paths through dulwich for macOS fork safety."""
    try:
        import dulwich.repo as _dulwich_repo  # local import — avoid top-level cost if unused
        _repo = _dulwich_repo.Repo(str(repo_dir))
        tracked = sorted(p.decode("utf-8", errors="replace") for p in _repo.open_index())
        if not tracked:
            raise RuntimeError("dulwich index is empty — cannot build review pack")
        return tracked, []
    except ImportError:
        return [], ["FATAL: dulwich not installed. Run: pip install dulwich"]
    except Exception as exc:
        return [], [f"FATAL: {exc}"]


def _append_memory_whitelist(
    parts: list[str],
    skipped: list[str],
    *,
    drive_root: pathlib.Path,
) -> Dict[str, Any]:
    """Inline the memory whitelist byte-exact and return the typed memory fact
    every delivery carries: ``{"inlined": n, "total": 7, "dispositions": {rel:
    inlined | missing | empty | oversized | read_error}}`` — one disposition per
    whitelisted path (task text, usage fact, provenance header), never a silent
    skip; every non-inlined entry also joins ``skipped`` for the omission section."""
    dispositions: Dict[str, str] = {}
    for rel_mem in _MEMORY_WHITELIST:
        full_path = drive_root / rel_mem
        try:
            if not full_path.is_file():
                dispositions[rel_mem] = "missing"
                skipped.append(f"drive/{rel_mem} (missing: not present under the data root)")
                continue
            size = full_path.stat().st_size
            if size > _MAX_FULL_REPO_FILE_BYTES:
                dispositions[rel_mem] = "oversized"
                skipped.append(f"drive/{rel_mem} (oversized: >{_MAX_FULL_REPO_FILE_BYTES // 1024}KB)")
                continue
            content = full_path.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                dispositions[rel_mem] = "empty"
                skipped.append(f"drive/{rel_mem} (empty: no content)")
                continue
            parts.append(f"## FILE: drive/{rel_mem}\n{content}\n")
            dispositions[rel_mem] = "inlined"
        except Exception as exc:
            dispositions[rel_mem] = "read_error"
            skipped.append(f"drive/{rel_mem} (read error: {exc})")
    return {"inlined": sum(1 for d in dispositions.values() if d == "inlined"),
            "total": len(_MEMORY_WHITELIST), "dispositions": dispositions}


def _append_omission_section(parts: list[str], skipped: list[str]) -> None:
    """Append a BOUNDED omission summary: counts per reason + a capped sample.

    Full per-file coverage (hash, size, disposition, reason for every tracked
    path) already lives in the atlas coverage manifest persisted to
    ``state/deep_self_review_context.json`` — this in-prompt section is a
    summary with an explicit pointer, not the coverage SSOT. Its size is
    reserved via ``_OMISSION_SECTION_RESERVE_TOKENS`` in ``atlas_fixed_tokens``
    and enforced here, so the assembled prompt provably fits the gate the
    atlas budgeted for. The cap is an explicit, visible summarization with an
    omission note — not silent truncation.
    """
    if not skipped:
        return
    counts: dict[str, int] = {}
    for entry in skipped:
        tag = entry.split("(", 1)[1].split(":", 1)[0].strip() if "(" in entry else "other"
        counts[tag] = counts.get(tag, 0) + 1
    header = [
        "## OMITTED FILES (not included in review pack)",
        "Reasons: sensitive=secrets/keys, vendored/minified=third-party bundled asset, "
        "binary/media=images/fonts/compiled blobs, excluded_dir=non-agent-logic directory, "
        "excluded_test=wider tests excluded, oversized=>1MB, read_error=unreadable, "
        "missing=whitelisted memory file absent, empty=whitelisted memory file blank. "
        "(A required atlas file that does not fit never reaches this list: it fails "
        "the pack instead of shrinking it.)",
        "Full per-file coverage for every tracked path is in the atlas coverage "
        "manifest (persisted to state/deep_self_review_context.json).",
        "",
        "Omitted counts by reason: "
        + ", ".join(f"{tag}={count}" for tag, count in sorted(counts.items())),
        "",
    ]
    sample = skipped[:_OMISSION_SAMPLE_MAX_ENTRIES]
    lines = header + [f"Sample ({len(sample)} of {len(skipped)} entries):"]
    lines.extend(f"  - {entry}" for entry in sample)
    if len(skipped) > len(sample):
        lines.append(
            f"  - … {len(skipped) - len(sample)} more entries omitted here "
            "(complete list in the coverage manifest)"
        )
    section = "\n".join(lines) + "\n"
    # Defensive hard bound: pathological entry lengths must never exceed the
    # reserve the atlas budgeted for. Trim sample rows (never the header) with a
    # visible note until the section fits.
    while estimate_tokens(section) > _OMISSION_SECTION_RESERVE_TOKENS and sample:
        sample = sample[: max(0, len(sample) - 5)]
        lines = header + [f"Sample ({len(sample)} of {len(skipped)} entries):"]
        lines.extend(f"  - {entry}" for entry in sample)
        lines.append(
            f"  - … {len(skipped) - len(sample)} more entries omitted here to fit "
            "the reserved omission budget (complete list in the coverage manifest)"
        )
        section = "\n".join(lines) + "\n"
    parts.append(section)


def _compute_graph_centrality(
    repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
) -> Dict[str, float]:
    """Per-path centrality bonus from the code-intelligence import graph.

    Reverse-import in-degree over ``resolved_import_paths``: a module imported
    by many others is structurally load-bearing and the most useful raw code to
    inline in a bounded full-repo pack. Returns a bounded score bonus per
    rel_path; empty dict on any failure (ranking then falls back to the atlas's
    existing path/size heuristics — selection still works, just less informed).
    Deep-review-only: scope/plan review never pass centrality to the atlas.
    """
    try:
        from ouroboros.code_intelligence import build_code_inventory

        inventory = build_code_inventory(repo_dir, drive_root=drive_root, persist=True)
        in_degree: Dict[str, int] = {}
        for file in inventory.files:
            for target in file.resolved_import_paths or ():
                if target and target != file.path:
                    in_degree[target] = in_degree.get(target, 0) + 1
        return {
            path: min(_CENTRALITY_MAX_BONUS, count * _CENTRALITY_PER_IMPORTER)
            for path, count in in_degree.items()
            if count > 0
        }
    except Exception:
        # Keep the documented "empty dict on ANY failure" contract: inventory
        # shape drift must degrade to heuristic ranking, not kill the review.
        log.debug("Graph centrality unavailable; using heuristic ranking", exc_info=True)
        return {}


def build_review_pack(
    repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
    fixed_prompt_tokens: int = 0,
    hard_budget_reduction: int = 0,
    input_token_limit: int = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Build bounded repo atlas + full memory whitelist pack.

    ``hard_budget_reduction`` lowers the budgets handed to the atlas — used by
    the final-shrink retry in ``run_deep_self_review`` when estimator drift
    between the atlas's per-section accounting and the final concatenation
    pushes the assembled prompt over the input gate. ``input_token_limit``
    overrides the default GPT-family cap with the model-family-calibrated cap
    resolved by the caller (Claude-family reviewers need a smaller estimated
    budget for the same 1M window — see review_helpers).
    """
    tracked, fatal = _dulwich_tracked_paths(repo_dir)
    if fatal:
        return "", {"file_count": 0, "total_chars": 0, "skipped": fatal}

    skipped: list[str] = []
    memory_parts: list[str] = []
    memory = _append_memory_whitelist(memory_parts, skipped, drive_root=drive_root)
    memory_text = "\n".join(memory_parts)

    # Low context mode: render ARCHITECTURE.md as a navigation map (full sections
    # read on demand) and exclude it from the atlas full-file selection instead of
    # inlining ~32K tokens. Reuses the atlas ``already_included`` mechanism so the
    # shared commit-gate atlas (scope / plan review) is unaffected.
    nav_parts: list[str] = []
    already_included: frozenset[str] = frozenset()
    if get_context_mode() == "low":
        try:
            arch_text = (repo_dir / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        except Exception:
            arch_text = ""
        if arch_text.strip():
            nav_parts.append(
                generate_doc_nav_map(
                    arch_text, title="ARCHITECTURE.md", rel_path="docs/ARCHITECTURE.md"
                )
                + "\n\nNote for this deep self-review call: this surface has no tool loop, "
                "so the navigation map is an index of omitted sections, not an actionable "
                "read_file instruction. Flag any needed full ARCHITECTURE.md section explicitly."
            )
            already_included = frozenset({"docs/ARCHITECTURE.md"})

    # Reserve the (bounded) omission section inside the atlas's fixed budget —
    # it is appended to the pack after the atlas fills, so an unreserved section
    # arithmetically guarantees overflow whenever the atlas reaches its ceiling.
    atlas_fixed_tokens = (
        int(fixed_prompt_tokens)
        + estimate_tokens(memory_text)
        + estimate_tokens("\n".join(nav_parts))
        + _OMISSION_SECTION_RESERVE_TOKENS
    )
    effective_limit = int(input_token_limit) or _DEEP_INPUT_TOKEN_LIMIT
    hard_budget = max(10_000, effective_limit - max(0, int(hard_budget_reduction)))
    centrality = _compute_graph_centrality(repo_dir, drive_root)

    def _compile(compact: bool):
        return compile_review_context_atlas(
            ReviewContextAtlasRequest(
                repo_dir=repo_dir,
                tracked_paths=tuple(tracked),
                already_included=already_included,
                fixed_prompt_tokens=atlas_fixed_tokens,
                target_total_tokens=min(850_000, hard_budget),
                hard_total_tokens=hard_budget,
                include_tests=False,
                title="Generated Deep Self-Review Atlas",
                compact_manifest=compact,
                centrality_scores=centrality,
            )
        )

    # Compact coverage is the atlas default (the durable manifest keeps full
    # per-file coverage either way), so there is no fuller form to fall back
    # from and no compact retry rung anymore.
    atlas = _compile(True)
    if atlas_assembly_failed(atlas):
        # No pack: a review that could not assemble a required artifact does not
        # run on the remainder (BIBLE P3). The manifest carries the disclosure.
        return "", {
            "file_count": 0,
            "total_chars": 0,
            "skipped": [
                "FATAL: "
                + atlas_assembly_failure_reason(atlas)
                + " (even with the compact manifest)"
            ],
            "context_manifest": atlas.manifest,
        }
    skipped.extend(
        f"{record.rel_path} ({record.disposition}: {record.reason})"
        for record in atlas.omitted
        if record.disposition not in {"already_included", "manifest_only"}
    )
    parts = [atlas.text]
    parts.extend(nav_parts)
    parts.extend(memory_parts)
    file_count = len(atlas.selected) + memory["inlined"]
    _append_omission_section(parts, skipped)

    pack_text = "\n".join(parts)
    stats = {
        "file_count": file_count,
        "total_chars": len(pack_text),
        "skipped": skipped,
        "context_manifest": atlas.manifest,
        "memory": memory,
    }
    return pack_text, stats


# ---------------------------------------------------------------------------
# Availability — route-aware on the configured row.
# ---------------------------------------------------------------------------


def _resolve_packed_window(model: str) -> Any:
    """ONE `ReviewerWindow` for the packed model from the shared resolver — the
    caller validates the floor on THIS object and sizes with THIS object."""
    from ouroboros import reviewer_window as _rw
    from ouroboros.config import review_model_uses_local

    return _rw.resolve_reviewer_window(model, use_local=review_model_uses_local(model))


def _packed_window_reason(window: Any, model: str) -> str:
    """The ≥1M floor of the PACKED delivery on an already-resolved window: a
    route whose Capability Evidence puts its window BELOW the full window
    cannot hold the pack that IS this delivery's guarantee, so it is refused
    typed (a native or session deep_review row serves a sub-1M route). An
    UNKNOWN window (no evidence) keeps the documented full-window assumption
    every packet surface shares (`reviewer_window`: guessing small is a
    certain loss of review on every cold-evidence install) and is disclosed in
    the report header; '' = the floor holds or is assumed."""
    from ouroboros import reviewer_window as _rw

    if 0 < int(window.window_tokens) < _rw.REVIEWER_FULL_WINDOW:
        return (
            f"the packed deep review needs a ≥{_rw.REVIEWER_FULL_WINDOW:,}-token window and {model} is "
            f"{window.status or 'evidenced'} at {int(window.window_tokens):,} tokens — pick a native or "
            "session deep_review row for a sub-1M route"
        )
    return ""


def _packed_route(configured: str) -> Tuple[str, Optional[str]]:
    """The packed delivery's ``(unavailable_reason, sendable_model)``.

    Provider/credential knowledge comes from the provider registry SSOT; two
    deliberate deep-review-specific rules live here: ``openai::`` is only
    trusted when ``OPENAI_BASE_URL`` is unset (a redirected endpoint cannot be
    assumed to honor the 1M-context contract the packed review depends on),
    and the payable model must not be EVIDENCED below the ≥1M floor
    (`_packed_window_reason`).
    """
    reason, model = _packed_credentials(configured)
    if reason:
        return reason, None
    reason = _packed_window_reason(_resolve_packed_window(str(model)), str(model))
    return (reason, None) if reason else ("", model)


def _packed_credentials(configured: str) -> Tuple[str, Optional[str]]:
    """The packed row's payable spelling, or the typed credentials reason."""
    provider = provider_for_model(configured)
    if provider == "openai":
        if provider_has_credentials("openai") and not os.environ.get("OPENAI_BASE_URL"):
            return "", configured
        return f"no direct OpenAI credentials for {configured} (or OPENAI_BASE_URL redirects the route)", None
    if configured.startswith("openai/"):
        # OpenRouter route with a direct-OpenAI rewrite fallback.
        if provider_has_credentials("openrouter"):
            return "", configured
        if provider_has_credentials("openai") and not os.environ.get("OPENAI_BASE_URL"):
            slug = configured.split("/", 1)[1]
            if slug.endswith("-pro"):
                # A `-pro` suffix is an OpenRouter ROUTING slug (reasoning
                # mode), not an OpenAI model id — `gpt-5.6-sol-pro` 404s on
                # api.openai.com (live-probed 2026-07-29). Only for these does
                # the direct route's own default take over; an owner's explicit
                # pin of a REAL model keeps the mechanical rewrite below, so a
                # pinned openai/gpt-5.5 still runs deep review on gpt-5.5.
                from ouroboros.provider_models import OPENAI_DIRECT_DEFAULTS

                return "", OPENAI_DIRECT_DEFAULTS["deep_self_review"]
            return "", "openai::" + slug
        return f"no OpenRouter or direct OpenAI credentials for {configured}", None
    if provider_has_credentials(provider):
        return "", configured
    return f"no {provider} credentials for {configured}", None


def _session_route_reason(row: ConfiguredReviewerSlot) -> str:
    """Why a delegated session row cannot run now, or '' — the substrate's own
    route health (the executor refuses on the same reader before it starts)."""
    from ouroboros.subagents import delegated_run_shape, parse_subagent_harness, route_health

    route = parse_subagent_harness(row.session_target or row.target_id)
    if route is None:
        return "session_target_unparsable"
    try:
        from ouroboros.claudexor_daemon import ensure_owned_gateway

        gateway = ensure_owned_gateway(admission_wait_sec=0)
    except Exception as exc:
        return f"agent_service_unavailable: {type(exc).__name__}: {exc}"
    try:
        unavailable, _reset_at = route_health(
            gateway, route.route_id, delegated_run_shape(False), route_model=route.model,
            pinned_profile=str(row.profile_id or getattr(route, "profile_id", "") or ""),
        )
    finally:
        gateway.close()
    return str(unavailable or "")


def deep_review_route(row: Optional[ConfiguredReviewerSlot] = None) -> Tuple[str, Optional[str]]:
    """``(unavailable_reason, identity)`` for the deep-review row.

    '' means available; ``identity`` is then the model the review runs on (the
    packed row's payable spelling, a native row's routed model, a session row's
    ``harness[=model]`` target). Availability is ROUTE-AWARE: the ≥1M /
    ``OPENAI_BASE_URL`` rule binds only the packed row; a native row needs the
    routed model's credentials; a session row needs a healthy delegated route.
    A malformed reviewer-slot setting is the typed reason, never a fallback.
    """
    try:
        row = row or deep_review_slot()
    except ValueError as exc:
        return str(exc), None
    if row.kind not in (ROUTE_KIND_API, ROUTE_KIND_SESSION):
        return f"deep_review row has an unknown route kind {row.kind!r}", None
    if not str(row.target_id or "").strip():
        return "deep_review row has no target (empty model id / session target)", None
    if not row.retrieves:
        return _packed_route(row.target_id)
    if row.is_session:
        reason = _session_route_reason(row)
        return reason, (None if reason else (row.session_target or row.target_id))
    from ouroboros.provider_models import model_has_credentials

    if model_has_credentials(row.target_id):
        return "", row.target_id
    return f"no provider credentials for {row.target_id}", None


def deep_review_unavailable_text(reason: str) -> str:
    """The ONE unavailable message (prefix classified by ``outcomes``)."""
    return (
        f"❌ Deep self-review unavailable: {reason}. Configure the deep-review row in "
        "Settings → Agents → Review lanes (or OUROBOROS_MODEL_DEEP_SELF_REVIEW) with a "
        "route this install can pay."
    )


# ---------------------------------------------------------------------------
# The three deliveries.
# ---------------------------------------------------------------------------


def _review_slot(row: ConfiguredReviewerSlot, model: str, timeout_sec: Optional[float]) -> Any:
    from ouroboros.config import review_model_uses_local
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import ReviewSlot

    return ReviewSlot(
        slot_id=row.slot_id, model=model, effort=row_effort(row, "deep_self_review"),
        timeout_sec=timeout_sec, max_tokens=_DEEP_MAX_OUTPUT_TOKENS,
        role_hint="deep self-reviewer", use_local=review_model_uses_local(model),
        route=ReviewRouteKind.AGENT_SESSION if row.is_session else ReviewRouteKind.API_CHAT,
        session_target=row.session_target, session_profile=row.profile_id,
        subagent_id=row.subagent_id,
    )


def _record_execution(slot: Any, usage: Dict[str, Any], *, status: str, error: str = "") -> None:
    """«Выполняется как» (D22) for the deep-review row — disclosure, best-effort."""
    try:
        from ouroboros.review_substrate import ReviewActorRecord
        from ouroboros.reviewer_slot_config import record_reviewer_slot_executions

        actor = ReviewActorRecord(slot_id=slot.slot_id, model=slot.model, status=status,
                                  usage=dict(usage or {}), error=error)
        record_reviewer_slot_executions("deep_self_review", [actor], {slot.slot_id: slot})
    except Exception:
        log.debug("deep self-review last-execution write failed", exc_info=True)


def _repo_relative(path: Any, repo_dir: pathlib.Path) -> str:
    """A receipt path as a repo-relative POSIX path — on EVERY host OS.
    Coverage hands it the receipt's ``opened_path`` (the root-relative path
    the reader actually opened — already free of the model's spelling:
    absolute, whitespace-padded, ``repo/``-prefixed or ``/``-qualified forms
    all arrive as ``BIBLE.md``) and, for a receipt WITHOUT one (nothing
    rendered), the raw spelling. Absolute paths under the repository are
    relativized, relative ones normalized — but a ``..`` component is kept AS
    SPELLED and so names no mandatory read: the registry refuses traversal
    shapes before dispatch, so ``a/../BIBLE.md`` delivered nothing and is
    never folded onto ``BIBLE.md``. The POSIX contract is by construction:
    separators are folded to ``/`` first, absolute spellings are recognized
    for every OS (``/``, drive-letter and UNC forms — ``PurePosixPath`` alone
    is blind to ``C:/``), and normalization is ``posixpath``'s, never
    ``os.path``'s, whose Windows form renders ``docs\\ARCHITECTURE.md`` and
    would never match a mandatory read."""
    text = str(path or "").replace("\\", "/")
    pure = pathlib.PurePosixPath(text)
    if ".." in pure.parts:
        return text
    if is_absolute_path_text(text):
        try:
            return pathlib.Path(text).resolve().relative_to(pathlib.Path(repo_dir).resolve()).as_posix()
        except (ValueError, OSError):
            return pure.as_posix()
    return posixpath.normpath(text).removeprefix("./")


def _native_read_coverage(usage: Dict[str, Any], repo_dir: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    """R8: how much of each mandatory read the host OBSERVED, from the episode's
    receipts — the merged line intervals of every executed repository-root
    ``read_file`` receipt for the path (a single result is capped, so a full
    read of BIBLE.md is multi-chunk by construction).

    ``read`` only when the union of extent-bearing receipts covers the whole
    file; otherwise ``unobserved`` when the receipt list was capped below the
    call count OR any matching executed receipt carries no extent (absence
    proves nothing there — full coverage must be proven by measured receipts
    alone); ``partial`` with the covered fraction; ``missing`` when nothing of
    the file was delivered — no receipt names it at a repository root (a
    data-plane read never counts), or every measured receipt delivered zero
    lines (a cursor past the window, a start past EOF). Receipts are matched
    on the path AND root the reader actually OPENED (``opened_path`` /
    ``opened_root``, stamped by the reader; the model's spellings are only
    disclosure — a padded ``" system_repo "`` counts, a ``runtime_data`` read
    never does), falling back to the raw spellings for a receipt that rendered
    nothing — where a ``..`` component names nothing (the registry refuses
    traversal shapes before dispatch — see ``_repo_relative``). Disclosure,
    never a refusal: the report is delivered with the flag in its header.
    """
    receipts = [r for r in (usage.get("native_tool_receipts") or []) if isinstance(r, dict)]
    capped = int(usage.get("native_tool_calls") or 0) > len(receipts)
    out: Dict[str, Dict[str, Any]] = {}
    for rel in _MANDATORY_READS:
        spans: list[tuple[int, int]] = []
        total, unmeasured = 0, False
        for r in receipts:
            named = r.get("opened_path") if isinstance(r.get("opened_path"), str) and r.get("opened_path") else r.get("path")
            root = r.get("opened_root") if isinstance(r.get("opened_root"), str) and r.get("opened_root") else str(r.get("root") or "")
            if (r.get("tool") != "read_file" or r.get("outcome") != "executed"
                    or root not in _REPO_ROOTS or _repo_relative(named, repo_dir) != rel):
                continue
            if not all(isinstance(r.get(k), int) for k in ("start_line", "end_line", "total_lines")):
                # Names the file but carries no extent: it may never have opened
                # it (an argument error, a registry refusal answered with text)
                # or opened it without a recorded extent — either way it proves
                # nothing, and keeps `read`/`missing` unproven (`unobserved`).
                unmeasured = True
                continue
            total = max(total, int(r["total_lines"]))
            if r["end_line"] >= r["start_line"]:
                spans.append((int(r["start_line"]), int(r["end_line"])))
        covered, cursor = 0, 0
        for start, end in sorted(spans):  # merge overlapping / re-read chunks; clip to the file
            lo, hi = max(start, cursor + 1, 1), min(end, total)
            if hi >= lo:
                covered += hi - lo + 1
                cursor = hi
        if total and covered >= total:
            state = "read"
        elif capped or unmeasured:
            state = "unobserved"
        else:
            state = "partial" if covered else "missing"
        out[rel] = {"state": state, "covered_lines": covered, "total_lines": total,
                    "fraction": round(covered / total, 3) if total else 0.0}
    return out


_HEADER_VALUE_MAX_CHARS = 120


def _header_value(value: Any) -> str:
    """One header value, bounded and unable to break the comment or the line:
    newlines become spaces, `--` (the comment terminator's body) collapses to
    `-`, and the text is cut disclosed at a fixed bound."""
    from ouroboros.utils import truncate_within_limit

    text = truncate_within_limit(str(value), _HEADER_VALUE_MAX_CHARS)
    # Sanitize AFTER the bound: the disclosed omission marker itself carries a
    # newline, and nothing may leave this function able to break the comment.
    text = text.replace("\r", " ").replace("\n", " ")
    while "--" in text:
        text = text.replace("--", "-")
    return text


def _delivery_incomplete(delivery: str, usage: Dict[str, Any], message: Optional[Dict[str, Any]] = None) -> str:
    """Completeness from the facts each delivery carries: the native episode's
    typed `native_incomplete`; the packed call's provider stop marker — the
    OpenAI-compatible normalizer's `usage["response_finish_reason"] == "length"`
    or the message's own `finish_reason` (fail-safe for a non-normalized
    message shape — no shipped normalizer authors it), and the direct-Anthropic lane's
    `message["stop_reason"] == "max_tokens"` (the shipped direct default route
    sets no usage finish reason at all) — meaning the report hit the output
    reserve; a session's completeness is not host-observable."""
    if delivery == "native_tool_rounds":
        return str(usage.get("native_incomplete") or "") or "none"
    if delivery == "api_packet":
        msg = message if isinstance(message, dict) else {}
        cut = (str(usage.get("response_finish_reason") or "") == "length"
               or str(msg.get("finish_reason") or "") == "length"
               or str(msg.get("stop_reason") or "") == "max_tokens")
        return "output_reserve" if cut else "none"
    return "unobserved"


def _provenance_header(delivery: str, model: str, usage: Dict[str, Any], memory: Dict[str, Any],
                       coverage: Dict[str, str], human: str, *, incomplete: str,
                       extra: Optional[Dict[str, Any]] = None) -> str:
    """R9: the host's provenance header (machine-readable comment + one human
    line) prepended to every delivered report. The fact set is built PER
    DELIVERY (a session never carries rounds/receipts; its attestation is
    `unobserved` by construction); every comment value goes through
    `_header_value` (bounded, sanitized), and the human line — whose external
    values the callers pass through `_header_value` too — is kept to one line
    with no comment terminator in it."""
    facts: Dict[str, Any] = {"delivery": delivery, "model": model, "memory": f"{memory['inlined']}/{memory['total']}"}
    # One value PER disposition (`memory_missing=…`, `memory_empty=…`, …): each
    # lists at most the seven whitelisted basenames (≈91 chars worst case), so
    # it fits the value bound; `_header_value` bounds it regardless.
    for d in ("missing", "empty", "oversized", "read_error"):
        names = [rel.rsplit("/", 1)[-1] for rel, got in memory["dispositions"].items() if got == d]
        if names:
            facts[f"memory_{d}"] = ",".join(names)
    facts["coverage"] = ",".join(f"{rel}:{state}" for rel, state in coverage.items())
    facts["incomplete"] = incomplete  # computed ONCE by the caller from the facts its delivery holds
    if delivery == "native_tool_rounds":
        facts.update({
            "attestation": usage.get("host_file_read_attestation") or "unobserved",
            "rounds": usage.get("native_rounds", 0), "tool_calls": usage.get("native_tool_calls", 0),
            "receipts": len(usage.get("native_tool_receipts") or []),
            "end_reason": usage.get("native_end_reason", ""),
            "transcript": f"{usage.get('native_transcript_chars', 0)}/{usage.get('native_transcript_bound', 0)}",
            "landing": f"{usage.get('native_landing_notified', False)}/{usage.get('native_landing_sent', False)}",
        })
    elif delivery == "api_packet":
        facts["attestation"] = "packed"
    else:
        facts["attestation"] = "unobserved"
    facts.update(extra or {})
    comment = ", ".join(f"{key}={_header_value(value)}" for key, value in facts.items())
    line = str(human).replace("\r", " ").replace("\n", " ")
    while "--" in line:  # the callers bound each external value; the line itself never carries a terminator
        line = line.replace("--", "-")
    return f"<!-- deep-review provenance: {comment} -->\n_{line}_\n\n"


def _memory_line(memory: Dict[str, Any]) -> str:
    """The human half of the memory fact: `memory 3/7 inlined (omitted: …)`."""
    omitted = [f"{rel.rsplit('/', 1)[-1]} {d}" for rel, d in memory["dispositions"].items() if d != "inlined"]
    return f"memory {memory['inlined']}/{memory['total']} inlined" + (f" (omitted: {', '.join(omitted)})" if omitted else "")


def _failed(text: str, *, reason_code: str, usage: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    """A failure result: the text plus TYPED usage, so the caller keeps the
    previous report instead of overwriting durable memory with an error.
    Callers spell ``reason_code`` as a literal — the runtime's reason-code
    drift guard (outcomes vocabulary) reads emit sites, not constants."""
    out = dict(usage or {})
    out.update({"execution_status": "infra_failed", "reason_code": reason_code})
    return text, out


def _retrieving_task(repo_dir: pathlib.Path, drive_root: pathlib.Path) -> Tuple[str, Dict[str, Any]]:
    """The route-owned task text for a retrieving row: role + method, the
    memory whitelist inline (byte-exact, as the packed pack carries it), and
    the governance navigation maps. BIBLE.md is a mandatory READ, never
    inlined — on the native delivery the host checks the receipts for it."""
    from ouroboros.tools.scope_review_session import governance_nav_maps

    bible = load_governance_doc(repo_dir, "BIBLE.md", on_missing="silent")
    if not bible.strip():
        raise RuntimeError("BIBLE.md is missing at the repository root — a deep self-review has no constitution to check against")
    memory_parts: list[str] = []
    skipped: list[str] = []
    memory = _append_memory_whitelist(memory_parts, skipped, drive_root=drive_root)
    parts = [
        _ROLE_PROMPT + _RETRIEVING_METHOD.format(bible_chars=len(bible)),
        "## Memory (runtime data root, inlined byte-exact)",
        *memory_parts,
        # EVERY whitelisted entry gets its disposition here — the omission
        # section of this delivery — so an absent or blank memory file is a
        # stated fact the reviewer (and the header) can rely on, never a gap.
        f"Memory dispositions ({memory['total']} whitelisted): "
        + "; ".join(f"{rel} {d}" for rel, d in memory["dispositions"].items()),
    ]
    parts.append(governance_nav_maps(repo_dir, _NAV_MAP_DOCS))
    return "\n\n".join(parts), {"memory": memory, "bible_chars": len(bible)}


def _run_retrieving_review(
    repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
    llm: Any,
    emit_progress: Callable[[str], None],
    row: ConfiguredReviewerSlot,
    *,
    task_id: str,
    deadline_at: str,
) -> Tuple[str, Dict[str, Any]]:
    """A retrieving row (native episode or delegated session) through the
    shared executor seam, exactly like the advisory: hand-built request, slot
    and assignment; the product is the report text."""
    from dataclasses import asdict, replace as _dc_replace

    from ouroboros.config import get_finalization_grace_sec, get_task_abs_ceiling_sec
    from ouroboros.deadline_utils import review_operation_timeout_sec
    from ouroboros.observability import persist_call
    from ouroboros.review_execution import ReviewAssignment, _review_route_executor
    from ouroboros.review_substrate import ReviewRequest
    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

    task_text, task_facts = _retrieving_task(repo_dir, drive_root)
    request = ReviewRequest(
        surface="deep_self_review",
        goal="Deep self-review of the whole Ouroboros system against BIBLE.md.",
        task_id=task_id, call_type="deep_self_review",
        max_tokens=_DEEP_MAX_OUTPUT_TOKENS, no_proxy=True,
        session_root=str(repo_dir), session_task=task_text,
        # The report contract rides the policy with the deep review's header
        # sentence (the shape is `report` either way). The data plane is the
        # REAL runtime root (R5), readable by the reviewer's own tools; memory
        # coverage itself is the inline whitelist in the task (byte-exact,
        # disposition-disclosed) — never receipts.
        policy={"output_contract": _REPORT_CONTRACT, "native_data_root": str(drive_root)},
        deadline_at=deadline_at,
    )
    # The logical window: the task's absolute ceiling narrowed by the owner
    # deadline — the same clock the coordinator gives a slot; without it the
    # native episode would run with no window at all and a session would fall
    # to the transport's own defaults.
    window = review_operation_timeout_sec(
        float(get_task_abs_ceiling_sec()),
        route="agent_session" if row.is_session else "api_chat",
        deadline_at=deadline_at, reserve_sec=get_finalization_grace_sec(),
    )
    slot = _review_slot(row, row.target_id, window)
    assignment = ReviewAssignment(
        request=request, slot=slot, call_id=f"deep_self_review:{task_id or 'manual'}",
        call_type="deep_self_review", custody_root=pathlib.Path(drive_root),
    )
    executor = _review_route_executor(assignment, llm=llm)
    executor._logical_deadline_monotonic = time.monotonic() + window
    delivery = "agent_session" if row.is_session else "native_tool_rounds"
    emit_progress(
        f"Deep self-review via {delivery} on {row.target_id}: {_memory_line(task_facts['memory'])}, "
        f"BIBLE.md ({task_facts['bible_chars']:,} chars) as a mandatory read; window {window:.0f}s..."
    )
    try:
        persist_call(
            pathlib.Path(drive_root), task_id=task_id or "deep_self_review",
            call_id=f"{assignment.call_id}_prompt", call_type="deep_self_review_prompt",
            payload={"request": asdict(request), "slot": asdict(slot), **executor.prompt_payload()},
            manifest={"surface": "deep_self_review", "slot_id": slot.slot_id, "model": slot.model},
        )
    except Exception:
        log.debug("deep self-review prompt custody write failed", exc_info=True)
    scope = _dc_replace(current_usage_scope() or UsageScope(), category="deep_self_review", source="deep_self_review")
    memory = task_facts["memory"]
    try:
        with usage_scope(scope):
            attempt = executor.execute()
    except Exception as exc:
        # The memory fact precedes EVERY «Выполняется как» record — this
        # failure-custody row included — and rides the typed failure the caller
        # receives (with the executor's proven custody facts, so a failed
        # execution stays visible). BudgetExceeded is recorded, then propagates
        # to the agent's budget rail like every other budget refusal.
        custody = {**executor.failure_custody(), "deep_review_memory": memory}
        _record_execution(slot, custody, status="error", error=f"{type(exc).__name__}: {exc}")
        if isinstance(exc, BudgetExceeded):
            raise
        log.error("Deep self-review failed: %s", exc, exc_info=True)
        return _failed(f"❌ Deep self-review failed: {type(exc).__name__}: {exc}",
                       reason_code="deep_self_review_error", usage=custody)
    usage = dict(attempt.usage or {})
    # Attached FIRST: the usage handed to every «Выполняется как» record below
    # and the returned usage carry the memory fact. The durable D22 projection
    # itself persists route/model/status/capability_delta and the typed failure
    # facts only — memory is disclosed durably by the header and this usage.
    usage["deep_review_memory"] = memory
    # The executor's own list is never mutated (shallow copy): the coverage
    # deltas below are appended to THIS record's copy.
    usage["capability_delta"] = list(usage.get("capability_delta") or [])
    text = str(attempt.raw_text or "")
    if not text.strip():
        # An empty product is an ERROR row in «Выполняется как», exactly like
        # the packed path — never recorded as a responded review.
        _record_execution(slot, usage, status="error", error="empty response")
        return _failed("⚠️ Model returned an empty response for the deep self-review.",
                       reason_code="deep_self_review_error", usage=usage)
    if delivery == "native_tool_rounds":
        detail = _native_read_coverage(usage, repo_dir)
        coverage = {rel: (f"partial({c['fraction']:.2f})" if c["state"] == "partial" else c["state"])
                    for rel, c in detail.items()}
        for rel, c in detail.items():
            if c["state"] != "read":
                usage["capability_delta"].append({
                    "kind": "capability_delta",
                    "requested": f"mandatory full read of {rel}",
                    "effective": {
                        "partial": f"{c['covered_lines']} of {c['total_lines']} lines of {rel} delivered (merged receipts)",
                        "missing": f"no executed repository-root read_file receipt for {rel}",
                    }.get(c["state"], f"the {rel} read extent is unobserved (receipts capped or extent not recorded)"),
                    "reason": f"deep_review_mandatory_read_{c['state']}",
                })
    else:
        coverage = {rel: "unobserved" for rel in _MANDATORY_READS}
    _record_execution(slot, usage, status="responded")
    try:
        from ouroboros.anthropic_native_custody import public_custody_projection

        persist_call(
            pathlib.Path(drive_root), task_id=task_id or "deep_self_review",
            call_id=f"{assignment.call_id}_response", call_type="deep_self_review_response",
            payload={"message": public_custody_projection(attempt.message), "usage": usage},
            manifest={"surface": "deep_self_review", "slot_id": slot.slot_id, "model": slot.model},
        )
    except Exception:
        log.debug("deep self-review response custody write failed", exc_info=True)
    if not usage.get("capability_delta"):
        usage.pop("capability_delta", None)  # an empty list is not a disclosure
    if not usage.get("resolved_model"):
        usage["resolved_model"] = row.target_id
    incomplete = _delivery_incomplete(delivery, usage)
    model = str(usage.get("resolved_model") or row.target_id)
    # Every external value on the HUMAN line is bounded and sanitized too.
    shown_model, shown_reason = _header_value(model), _header_value(incomplete)
    shown_target = _header_value(row.session_target or row.target_id)
    completeness = "complete" if incomplete == "none" else (
        "completeness not host-observed" if incomplete == "unobserved" else f"INCOMPLETE ({shown_reason})")
    if delivery == "native_tool_rounds":
        reads = "; ".join(
            f"{rel} " + (f"{c['fraction']:.0%} read ({c['covered_lines']}/{c['total_lines']} lines)" if c["state"] == "partial"
                         else {"read": "read in full", "missing": "NOT read"}.get(c["state"], "read extent unobserved"))
            for rel, c in detail.items())
        human = (
            f"Deep self-review: native inspection episode on {shown_model} — {int(usage.get('native_rounds') or 0)} rounds, "
            f"{int(usage.get('native_tool_calls') or 0)} tool calls ({len(usage.get('native_tool_receipts') or [])} host-observed receipts); "
            f"{reads}; {_memory_line(task_facts['memory'])}; {completeness}"
        )
    else:
        human = (
            f"Deep self-review: agent session {shown_target}"
            + (f" (model {shown_model})" if model and model != (row.session_target or row.target_id) else "")
            + f" — reads not host-observed (coverage unobserved); {_memory_line(task_facts['memory'])}; {completeness}"
        )
    emit_progress(f"Deep self-review complete ({len(text):,} chars; {delivery}, incomplete={incomplete}).")
    return _provenance_header(delivery, model, usage, task_facts["memory"], coverage, human, incomplete=incomplete) + text, usage


def _pack_unfit_failure(
    drive_root: pathlib.Path, model: str, stats: Dict[str, Any], input_limit: int, deep_window: int,
) -> Tuple[str, Dict[str, Any]]:
    """The typed refusal for a required set that does not fit this model's
    calibrated cap even warm. Its text is the agent's cue to ask the owner to
    switch the ``deep_review`` row — the host never switches deliveries itself."""
    from ouroboros.capability_evidence import resolve_review_token_density

    density, source = resolve_review_token_density(drive_root, model)
    return _failed(
        "❌ Deep self-review pack unfit: the one-packet review does not fit this repository on "
        f"{model} — {stats['skipped'][0]}. Calibrated input cap ~{input_limit:,} tokens "
        f"({deep_window:,}-token window, token density {density:.2f} {source}). "
        "No automatic fallback runs: ask the owner to switch the `deep_review` reviewer row "
        "(Settings → Review lanes) to a retrieving delivery — a configured subagent (native tool "
        "rounds) or an agent session — or to a model with a larger context window.",
        reason_code=REASON_DEEP_SELF_REVIEW_PACK_UNFIT,
    )


def _run_packed_review(
    repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
    llm: Any,
    emit_progress: Callable[[str], None],
    row: ConfiguredReviewerSlot,
    model: str,
) -> Tuple[str, Dict[str, Any]]:
    """The packed delivery: one 1M-context call carrying the Atlas + memory,
    byte-identical to the pre-row deep review. ``model`` is the payable
    spelling ``deep_review_route`` resolved for the row.

    no_proxy=True avoids macOS fork-safety SIGSEGV by using a one-shot httpx
    client with trust_env=False in llm.py; regular task calls are unaffected.
    """
    # ONE window fact for this run (Capability Evidence): the floor is judged
    # on THIS object and the pack is sized with THIS object — a second
    # resolution could disagree with the one that was validated. A confirmed
    # sub-1M route is refused typed here too (the availability check ran
    # earlier, but evidence can land between the reads); the pack is never
    # silently shrunk; an unknown window keeps the documented full-window
    # assumption and is disclosed in the header.
    from ouroboros import reviewer_window as _rw

    window_fact = _resolve_packed_window(model)
    sub_floor = _packed_window_reason(window_fact, model)
    if sub_floor:
        return _failed(deep_review_unavailable_text(sub_floor), reason_code="deep_self_review_unavailable")
    deep_window = window_fact.sizing_window()
    deep_output_reserve, deep_margin = _rw.window_scaled_reserves(
        deep_window,
        output_reserve=_DEEP_MAX_OUTPUT_TOKENS,
        tokenizer_margin=_DEEP_OUTPUT_MARGIN_TOKENS,
    )
    input_limit = max(0, calibrated_input_token_limit(
        model,
        context_window=deep_window,
        output_reserve=deep_output_reserve,
        tokenizer_margin=deep_margin,
        drive_root=drive_root,
    ))

    emit_progress("Building generated review atlas and memory pack...")
    pack_text, stats = build_review_pack(
        repo_dir,
        drive_root,
        fixed_prompt_tokens=estimate_tokens(_SYSTEM_PROMPT),
        input_token_limit=input_limit,
    )
    if not pack_text and stats.get("skipped") and stats.get("context_manifest"):
        # The required set did not fit under the calibrated cap. Cold store →
        # one probe (the shared rung; a slice of the real pack, a few output
        # tokens, through chat_observed), one rebuild under the recalibrated
        # cap; still unfit → the typed refusal that asks the owner to switch
        # the row (never a fallback). BudgetExceeded propagates to the agent's
        # budget rail exactly like the review send itself.
        from ouroboros.capability_evidence import cold_start_density_probe

        if cold_start_density_probe(
            drive_root, llm, emit_progress, model,
            density_probe_sample(repo_dir, stats["context_manifest"]),
            task_id="deep_self_review", call_type="deep_self_review_density_probe",
            source="deep_review_cold_start_probe",
        ) == "measured":
            input_limit = max(0, calibrated_input_token_limit(
                model,
                context_window=deep_window,
                output_reserve=deep_output_reserve,
                tokenizer_margin=deep_margin,
                drive_root=drive_root,
            ))
            emit_progress(f"Rebuilding the pack under the recalibrated input cap (~{input_limit:,} tokens)...")
            pack_text, stats = build_review_pack(
                repo_dir,
                drive_root,
                fixed_prompt_tokens=estimate_tokens(_SYSTEM_PROMPT),
                input_token_limit=input_limit,
            )
        if not pack_text and stats.get("skipped") and stats.get("context_manifest"):
            return _pack_unfit_failure(drive_root, model, stats, input_limit, deep_window)
    if not pack_text and stats.get("skipped"):
        return _failed(f"❌ Failed to build review pack: {stats['skipped'][0]}", reason_code="deep_self_review_error")

    emit_progress(
        f"Review pack built: {stats['file_count']} files, "
        f"{stats['total_chars']:,} chars"
        + (f", {len(stats['skipped'])} skipped" if stats["skipped"] else "")
    )

    # Gate full system+pack like scope review: reserve output headroom
    # inside the 1M window (min(SSOT, window − output − margin)) so a large
    # pack cannot trigger the deterministic input+output>window provider 400.
    estimated_tokens = estimate_tokens(_SYSTEM_PROMPT + pack_text)
    if estimated_tokens > input_limit:
        # Deterministic final shrink (instead of the historical fatal error):
        # rebuild once with the atlas budget reduced by the measured overage
        # plus margin, so residual estimator drift between per-section
        # accounting and this final concatenation cannot kill the review.
        overage = estimated_tokens - input_limit
        emit_progress(
            f"Pack overshot the input limit by ~{overage:,} tokens; "
            "rebuilding with a tighter atlas budget..."
        )
        pack_text, stats = build_review_pack(
            repo_dir,
            drive_root,
            fixed_prompt_tokens=estimate_tokens(_SYSTEM_PROMPT),
            hard_budget_reduction=overage + 8_000,
            input_token_limit=input_limit,
        )
        if not pack_text and stats.get("skipped"):
            return _failed(f"❌ Failed to build review pack: {stats['skipped'][0]}", reason_code="deep_self_review_error")
        estimated_tokens = estimate_tokens(_SYSTEM_PROMPT + pack_text)
    full_prompt_chars = len(_SYSTEM_PROMPT) + len(pack_text)
    if estimated_tokens > input_limit:
        return _failed(
            f"❌ Review pack too large: ~{estimated_tokens:,} tokens "
            f"({full_prompt_chars:,} chars of system+pack, {stats['file_count']} files). "
            f"Maximum is ~{input_limit:,} tokens "
            f"({deep_window:,}-token window minus {deep_output_reserve:,} output reserve, "
            f"calibrated for {model}). "
            "Reduce codebase size or split review.",
            reason_code="deep_self_review_error",
        )

    if stats.get("context_manifest"):
        try:
            atomic_write_json(
                drive_root / "state" / "deep_self_review_context.json",
                {
                    "ts": utc_now_iso(),
                    "model": model,
                    "context_manifest": stats["context_manifest"],
                },
                trailing_newline=True,
            )
        except Exception:
            log.warning("Failed to persist deep self-review context manifest", exc_info=True)

    emit_progress(f"Sending to {model} (~{estimated_tokens:,} tokens). This may take several minutes...")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": pack_text},
    ]

    # no_proxy prevents macOS fork-safety SIGSEGV in bundled child process.
    from ouroboros.llm_observability import chat_observed

    response, usage = chat_observed(
        llm,
        drive_root=drive_root,
        task_id="deep_self_review",
        call_type="deep_self_review",
        messages=messages,
        model=model,
        tools=None,
        reasoning_effort=row_effort(row, "deep_self_review"),
        max_tokens=_DEEP_MAX_OUTPUT_TOKENS,
        temperature=None,
        no_proxy=True,
    )
    usage = dict(usage or {})
    memory = stats.get("memory") or {"inlined": 0, "total": len(_MEMORY_WHITELIST), "dispositions": {}}
    # FIRST: the usage of both «Выполняется как» records below (error or
    # responded) and the returned usage carry the memory fact; the durable D22
    # projection itself does not (route/model/status/capability_delta and typed
    # failure facts only).
    usage["deep_review_memory"] = memory
    slot = _review_slot(row, model, None)
    text = response.get("content") or "" if isinstance(response, dict) else ""
    if not text:
        _record_execution(slot, usage, status="error", error="empty response")
        return _failed("⚠️ Model returned an empty response for the deep self-review.",
                       reason_code="deep_self_review_error", usage=usage)
    usage.setdefault("resolved_model", model)
    _record_execution(slot, usage, status="responded")
    # Completeness from the response the packed path holds: the provider's
    # stop marker (OpenAI-compatible finish_reason OR direct-Anthropic
    # stop_reason), not only the normalizer's usage projection.
    incomplete = _delivery_incomplete("api_packet", usage, response if isinstance(response, dict) else None)
    emit_progress(f"Deep self-review complete ({len(text):,} chars; incomplete={incomplete}).")
    window_label = f"{deep_window}" if int(window_fact.window_tokens) > 0 else f"assumed_{deep_window}"
    header = _provenance_header(
        "api_packet", model, usage, memory, {"pack": f"{stats['file_count']}_files"},
        f"Deep self-review: one packed API review on {_header_value(model)} — {stats['file_count']} files; "
        f"{_memory_line(memory)}; window {deep_window:,}" + (" (unknown, full window assumed)" if int(window_fact.window_tokens) <= 0 else "")
        + "; " + ("complete" if incomplete == "none" else f"INCOMPLETE ({incomplete}: the report hit the output reserve)"),
        incomplete=incomplete, extra={"window": window_label},
    )
    return header + text, usage


def run_deep_self_review(
    repo_dir: pathlib.Path,
    drive_root: pathlib.Path,
    llm: Any,
    emit_progress: Callable[[str], None],
    *,
    task_id: str = "",
    deadline_at: str = "",
    slot: Optional[ConfiguredReviewerSlot] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Execute the deep self-review on the configured row.

    Returns ``(text, usage)``. A delivered report carries the host provenance
    header; every ordinary review failure returns its text with typed usage
    (``execution_status="infra_failed"`` + ``reason_code``) so the caller can
    keep the previous report instead of overwriting it with an error; on a
    retrieving row a failure after the task was assembled also carries the
    memory fact (``deep_review_memory``) and the executor's failure custody —
    the same usage its «Выполняется как» error row was recorded from. The ONE
    exception that propagates is ``BudgetExceeded`` — the paid ledger's
    refusal is budget vocabulary for the agent's budget-pause rail, not a
    review error.
    ``slot`` overrides the configured row (tests, callers that already resolved it).
    """
    try:
        try:
            row = slot or deep_review_slot()
        except ValueError as exc:
            return _failed(deep_review_unavailable_text(str(exc)), reason_code="deep_self_review_unavailable")
        reason, model = deep_review_route(row)
        if reason:
            return _failed(deep_review_unavailable_text(reason), reason_code="deep_self_review_unavailable")
        if row.retrieves:
            return _run_retrieving_review(
                repo_dir, drive_root, llm, emit_progress, row, task_id=task_id, deadline_at=deadline_at,
            )
        return _run_packed_review(repo_dir, drive_root, llm, emit_progress, row, str(model or ""))
    except BudgetExceeded:
        # The paid ledger's refusal is BUDGET vocabulary, not a review error:
        # it propagates so the agent's own `except BudgetExceeded` rail (the
        # budget-pause checkpoint) stays live for the deep review too.
        raise
    except Exception as e:
        log.error("Deep self-review failed: %s", e, exc_info=True)
        return _failed(f"❌ Deep self-review failed: {type(e).__name__}: {e}", reason_code="deep_self_review_error")
