"""Enforcement-aware Atlas-backed scope reviewer for the commit pipeline.

Runs beside triad review and sees touched context plus a generated repo atlas. Critical findings follow
``OUROBOROS_REVIEW_ENFORCEMENT``: blocking enforcement blocks, advisory
enforcement reports them without blocking. Infrastructure failures such as
model errors, empty output, parse failures, and touched-context errors still
fail closed; oversized prompts are the explicit non-blocking skip path.
"""

from __future__ import annotations

import contextvars
import inspect
import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

from ouroboros.llm import LLMClient
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.review_context_atlas import (
    ReviewContextAtlasRequest,
    compile_review_context_atlas,
)
from ouroboros.tools.review_helpers import (
    build_goal_section,
    build_rebuttal_section as _shared_build_rebuttal_section,
    build_scope_section,
    build_touched_file_pack,
    load_checklist_section,
    review_drive_root,
    CRITICAL_FINDING_CALIBRATION,
    REPO_ANTI_PATTERN_LOCK_GUARD,
    REVIEW_JSON_ARRAY_CONTRACT,
    REVIEW_PREAMBLE,
    BINARY_EXTENSIONS,
    _SENSITIVE_EXTENSIONS,
    _SENSITIVE_NAMES,
    load_governance_doc,
    _ANTI_THRASHING_RULE_VERDICT,
    _CONVERGENCE_RULE_TEXT,
    _HISTORY_VERIFICATION_ONLY_RULE,
    build_review_history_section as _shared_review_history_section,
    emit_review_usage,
    format_review_history_entry,
    parse_git_name_status,
)
from ouroboros.triad_review import extract_json_array
from ouroboros.utils import (
    run_cmd,
    utc_now_iso,
    append_jsonl,
    estimate_tokens,
    truncate_review_artifact as _truncate_review_artifact,
)

log = logging.getLogger(__name__)

_SCOPE_MODEL_DEFAULT = "openai/gpt-5.5"
_SCOPE_MAX_TOKENS = 100_000  # 100K output tokens
_SCOPE_REVIEW_SLOT_TIMEOUT_SEC = 900

# Budget gate: estimate_tokens under-counts real tokens, so this non-blocking
# skip limit leaves headroom for 1M-context reviewer models.
from ouroboros.tools.review_helpers import REVIEW_PROMPT_TOKEN_BUDGET as _REVIEW_BUDGET

_SCOPE_BUDGET_TOKEN_LIMIT = _REVIEW_BUDGET

# Scope reviewers run on a >=1M-context model (BIBLE P3 context floor). The
# shared prompt-size SSOT (920K) governs INPUT only, but the reviewer also
# reserves _SCOPE_MAX_TOKENS for OUTPUT inside that same 1M window. 920K input +
# 100K output exceeds 1M, and provider tokenizers can exceed estimate_tokens by
# tens of thousands of tokens on atlas-heavy prompts. Gate assembled INPUT on a
# conservative effective cap and retry once with a compact atlas prompt before
# routing to the existing NON-blocking budget_exceeded skip.
_SCOPE_MODEL_CONTEXT_WINDOW = 1_000_000
# Conservative sub-floor window assumed for an UNKNOWN reviewer (no Capability
# Evidence) when the owner has NOT declared blocking_1m. Below the 1M floor, so the
# P3 authority check downgrades to advisory + the token budget routes to the
# visible budget_exceeded skip instead of pretending the route is 1M-capable.
_SCOPE_FAILCLOSED_WINDOW = 200_000
_SCOPE_OUTPUT_MARGIN_TOKENS = 155_000
_SCOPE_INPUT_TOKEN_LIMIT = min(
    _SCOPE_BUDGET_TOKEN_LIMIT,
    _SCOPE_MODEL_CONTEXT_WINDOW - _SCOPE_MAX_TOKENS - _SCOPE_OUTPUT_MARGIN_TOKENS,
)

# Tokenizer-density calibration per reviewer family (rationale + ratio SSOT in
# review_helpers.calibrated_input_token_limit): Claude-family tokenizers cut
# code-heavy packs at ~2.5 chars/token, so the chars/4 estimate undercounts by
# ~1.58x and a 739K-estimated pack drew a deterministic provider 400. The
# calibration shrinks the PROMPT for the same pinned reviewer — never the
# reviewer model or the >=1M window floor (BIBLE P3).
from ouroboros.tools.review_helpers import (
    calibrated_input_token_limit as _calibrated_input_token_limit,
    is_claude_family_model as _is_anthropic_family_model,
)

_ANTHROPIC_SCOPE_INPUT_TOKEN_LIMIT = _calibrated_input_token_limit(
    "anthropic/claude",
    context_window=_SCOPE_MODEL_CONTEXT_WINDOW,
    output_reserve=_SCOPE_MAX_TOKENS,
    tokenizer_margin=_SCOPE_OUTPUT_MARGIN_TOKENS,
    budget_cap=_SCOPE_BUDGET_TOKEN_LIMIT,
)

# Opt-in degraded low-context scope review (OUROBOROS_SCOPE_REVIEW_DEGRADED):
# when the owner selects low context mode for a local/no-1M setup, this may run a
# window-fitting ADVISORY scope review instead of the non-blocking skip. The atlas
# selects the highest-scored touched + import-seam + contract files in full and
# lists the rest as manifest_only (named uncovered files). 90K input + the 100K
# scope output reserve = 190K, fitting a ~200K reviewer window; truly tiny local
# reviewers still fail-soft to the skip. It never lowers the blocking scope floor:
# degraded findings are advisory-only and active only when BOTH low mode and the
# opt-in are set.
_LOW_SCOPE_INPUT_TOKEN_LIMIT = 90_000


def _degraded_scope_requested() -> bool:
    """Whether supplemental degraded (advisory) low-context scope feedback is on.

    True when the P3 floor config is 'advisory', OR (legacy) when low context mode
    is set and OUROBOROS_SCOPE_REVIEW_DEGRADED is enabled. In either case the
    degraded findings are advisory-only and never satisfy the blocking 1M floor."""
    try:
        from ouroboros.config import get_scope_review_floor
        if get_scope_review_floor() == "advisory":
            return True
    except Exception:
        pass
    try:
        from ouroboros.config import get_context_mode
        low = get_context_mode() == "low"
    except Exception:
        low = False
    return low and os.environ.get("OUROBOROS_SCOPE_REVIEW_DEGRADED", "").strip().lower() in ("1", "true", "yes", "on")


def _is_designated_default_reviewer(model: str) -> bool:
    """True iff ``model`` is the shipped default reviewer, across provider spellings."""
    def _normalized(m: str) -> str:
        text = str(m or "").strip()
        if text.startswith("openrouter::"):
            text = text[len("openrouter::"):]
        try:
            from ouroboros.provider_models import normalize_model_identity
            return normalize_model_identity(text)
        except Exception:
            return text
    return bool(model) and _normalized(model) == _normalized(_SCOPE_MODEL_DEFAULT)


def _scope_reviewer_window(model: str) -> int:
    """Reviewer context window (tokens) from Capability Evidence, FAIL-CLOSED on
    absent evidence. Replaces the deleted static per-model window table: a
    confirmed/asserted probe (provider metadata or owner-ack) for the reviewer's REAL
    active route gives the real window. With NO evidence, the 1M blocking-floor
    sentinel is granted ONLY to the SHIPPED designated reviewer under ``blocking_1m``
    (the default for gpt-5.5); any other no-evidence reviewer — including an operator's
    off-default ``OUROBOROS_SCOPE_REVIEW_MODEL`` pin — returns a conservative sub-floor
    window so the P3 authority check downgrades it (visibly) instead of silently
    treating a 200K model as 1M and overflowing its real window into a provider 400
    (the v6.46.0 scope-discard bug). A non-default >=1M reviewer must be owner-acked to
    regain 1M. Hot-path safe (allow_fetch=False): never blocks on the network."""
    model = str(model or "")
    try:
        from ouroboros.capability_evidence import probe
        from ouroboros.config import DATA_DIR, load_settings
        from ouroboros.provider_models import provider_for_model
        settings = load_settings()
        provider = provider_for_model(model)
        base_url = ""
        if provider == "openai":
            base_url = str(settings.get("OPENAI_BASE_URL") or "")
        elif provider == "openai-compatible":
            base_url = str(settings.get("OPENAI_COMPATIBLE_BASE_URL") or "")
        elif provider == "cloudru":
            base_url = str(settings.get("CLOUDRU_FOUNDATION_MODELS_BASE_URL") or "")
        elif provider == "gigachat":
            base_url = str(settings.get("GIGACHAT_BASE_URL") or "")
        # Probe the scope slot, not the active main route (which honors USE_LOCAL_MAIN).
        use_local = provider == "local" or model.endswith(" (local)")
        ev = probe(
            DATA_DIR,
            provider="local" if use_local else provider,
            model=model,
            base_url=base_url,
            use_local=use_local,
            allow_fetch=False,
        )
        if int(ev.window_tokens or 0) > 0:
            return int(ev.window_tokens)
    except Exception:
        pass
    try:
        from ouroboros.config import get_scope_review_floor
        floor = get_scope_review_floor()
    except Exception:
        floor = "blocking_1m"
    # blocking_1m declares the SHIPPED reviewer is the >=1M blocking gate; it does not
    # extend that 1M trust to an arbitrary off-default pin with no evidence.
    if floor == "blocking_1m" and _is_designated_default_reviewer(model):
        return _SCOPE_MODEL_CONTEXT_WINDOW
    return _SCOPE_FAILCLOSED_WINDOW


def _window_scaled_reserves(window: int) -> tuple:
    """(output_reserve, tokenizer_margin) scaled to the reviewer window.

    The absolute 1M-calibrated reserves (100K output + 155K margin) would
    swallow a small window whole (gigachat 131K => input limit 0, bricking the
    slot — Provider Independence). Sub-floor windows scale the reserves to the
    window instead: a quarter for output (floored at 8K so the reviewer can
    still produce the full checklist JSON) and an eighth for tokenizer margin.
    >=1M windows keep the absolute reserves unchanged.
    """
    if window >= _SCOPE_MODEL_CONTEXT_WINDOW:
        return _SCOPE_MAX_TOKENS, _SCOPE_OUTPUT_MARGIN_TOKENS
    output_reserve = min(_SCOPE_MAX_TOKENS, max(8_192, window // 4))
    tokenizer_margin = min(_SCOPE_OUTPUT_MARGIN_TOKENS, window // 8)
    return output_reserve, tokenizer_margin


def _effective_scope_input_limit(*, degraded: bool = False, scope_model: str = "") -> int:
    """Scope input token cap for normal vs supplemental degraded review.

    The commit gate calls the normal full-cap path. Degraded is explicit so the
    low/no-1M advisory path cannot silently replace the blocking 1M floor.
    The cap is model-aware on two axes: Claude-family reviewers get the
    code-density-calibrated cap so the assembled prompt fits their REAL
    tokenizer (rationale above), and a KNOWN reviewer window (Capability Evidence,
    not a static table) replaces the assumed 1M so a small-window reviewer
    overflows into the visible non-blocking budget_exceeded skip instead of a
    deterministic provider 400.
    """
    if degraded and _degraded_scope_requested():
        return _LOW_SCOPE_INPUT_TOKEN_LIMIT
    model = scope_model or _get_scope_model()
    window = _scope_reviewer_window(model)
    output_reserve, tokenizer_margin = _window_scaled_reserves(window)
    if _is_anthropic_family_model(model):
        if window == _SCOPE_MODEL_CONTEXT_WINDOW:
            return _ANTHROPIC_SCOPE_INPUT_TOKEN_LIMIT
        return max(0, _calibrated_input_token_limit(
            model,
            context_window=window,
            output_reserve=output_reserve,
            tokenizer_margin=tokenizer_margin,
            budget_cap=_SCOPE_BUDGET_TOKEN_LIMIT,
        ))
    return max(0, min(_SCOPE_BUDGET_TOKEN_LIMIT, window - output_reserve - tokenizer_margin))

# Defense-in-depth cap for deleted-file HEAD content inlined into the prompt.
_DELETED_INLINE_MAX_BYTES = 1_048_576  # 1 MB

_SCOPE_CONTEXT_MANIFEST = contextvars.ContextVar("scope_context_manifest", default={})


class _ScopeAtlasBudgetExceeded(RuntimeError):
    def __init__(self, manifest: dict):
        self.manifest = dict(manifest or {})
        token_count = int(self.manifest.get("estimated_total_tokens") or 0)
        super().__init__(
            "Generated Scope Atlas exceeded hard budget"
            + (f" (~{token_count:,} estimated tokens)" if token_count else "")
        )


def _current_scope_context_manifest() -> dict:
    return dict(_SCOPE_CONTEXT_MANIFEST.get({}) or {})


@dataclass
class ScopeReviewResult:
    """Structured outcome from ``run_scope_review``."""
    blocked: bool = False
    block_message: str = ""
    parsed_items: List[dict] = field(default_factory=list)
    critical_findings: List[dict] = field(default_factory=list)
    advisory_findings: List[dict] = field(default_factory=list)
    # Canonical per-actor evidence.
    raw_text: str = ""
    model_id: str = ""
    status: str = "responded"  # "responded"|"error"|"parse_failure"|"empty_response"|"budget_exceeded"|"omitted"|"empty"
    prompt_chars: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    context_manifest: dict = field(default_factory=dict)
    prompt_ref: dict = field(default_factory=dict)
    response_ref: dict = field(default_factory=dict)


@dataclass
class _TouchedContextStatus:
    """Touched-context sentinel; ``None`` means context OK."""
    status: str  # "empty" | "omitted" | "budget_exceeded" | "fixed_overflow"
    omitted_paths: List[str] = field(default_factory=list)
    token_count: int = 0  # estimated full prompt tokens when budget is exceeded


def _get_scope_model() -> str:
    """Return the configured scope review model (env → settings default)."""
    try:
        from ouroboros.config import get_scope_review_models

        models = get_scope_review_models()
        if models:
            return models[0]
    except Exception:
        pass
    return os.environ.get("OUROBOROS_SCOPE_REVIEW_MODEL", "").strip() or _SCOPE_MODEL_DEFAULT

_CANONICAL_CONTEXT_DOCS = (
    "BIBLE.md",
    "docs/DEVELOPMENT.md",
    "docs/ARCHITECTURE.md",
    "docs/CHECKLISTS.md",
)
_SCOPE_REQUIRED_ITEMS = frozenset({
    "intent_alignment",
    "forgotten_touchpoints",
    "cross_surface_consistency",
    "regression_surface",
    "prompt_doc_sync",
    "architecture_fit",
    "cross_module_bugs",
    "implicit_contracts",
})
_SCOPE_VALID_SEVERITIES = frozenset({"critical", "advisory"})
_CURRENT_TOUCHED_CONTEXT_SKIP_PREFIXES = (
    "tests/",
)


def _load_canonical_context_docs(repo_dir: pathlib.Path) -> str:
    parts: list[str] = []
    for rel_path in _CANONICAL_CONTEXT_DOCS:
        parts.append(f"## {rel_path}\n\n{load_governance_doc(repo_dir, rel_path, on_missing='placeholder')}")
    return "\n\n---\n\n".join(parts)


def _should_skip_current_touched_context(path: str) -> bool:
    norm = str(path or "").replace("\\", "/").lstrip("./")
    return (
        norm in _CANONICAL_CONTEXT_DOCS
        or any(norm.startswith(prefix) for prefix in _CURRENT_TOUCHED_CONTEXT_SKIP_PREFIXES)
    )


def _build_review_history_section(history: list, open_obligations: list = None) -> str:
    """Format previous triad rounds for scope-review context."""
    return _shared_review_history_section(
        history,
        open_obligations,
        title="## Previous triad review rounds",
        include_commit_message=False,
        compact_labels=True,
    )


def _parse_staged_name_status(repo_dir: pathlib.Path) -> list:
    """Parse staged changes with rename/delete/copy awareness."""
    try:
        name_status_raw = run_cmd(
            ["git", "diff", "--cached", "--name-status"], cwd=repo_dir
        )
    except Exception:
        name_status_raw = ""

    entries = parse_git_name_status(name_status_raw)

    # Fallback to --name-only if --name-status produced nothing.
    if not entries:
        try:
            changed = run_cmd(["git", "diff", "--cached", "--name-only"], cwd=repo_dir)
            for p in changed.strip().splitlines():
                p = p.strip()
                if p:
                    entries.append(("M", p, p))
        except Exception:
            pass

    return entries


def _classify_deleted_for_inline(path: str) -> Optional[str]:
    """Return a suppression reason for deleted HEAD content, or None to inline."""
    fp = pathlib.Path(path)
    fname_lower = fp.name.lower()
    suffix_lower = fp.suffix.lower()
    if suffix_lower in _SENSITIVE_EXTENSIONS or fname_lower in _SENSITIVE_NAMES:
        return "sensitive (env/credential/key)"
    if suffix_lower in BINARY_EXTENSIONS:
        return "binary extension"
    return None


def _inline_deleted_file_pack(
    current_files_section: str,
    deleted_paths: list,
    repo_dir: pathlib.Path,
) -> str:
    """Append deleted-file HEAD content or explicit suppression markers."""
    if not deleted_paths:
        return current_files_section

    notes: list[str] = []
    for dp in deleted_paths:
        suffix = pathlib.Path(dp).suffix.lstrip(".") or "text"
        suppress_reason = _classify_deleted_for_inline(dp)
        if suppress_reason is not None:
            notes.append(
                f"### {dp}\n\n*(DELETED — {suppress_reason}; content suppressed)*\n"
            )
            continue

        try:
            head_content = run_cmd(
                ["git", "show", f"HEAD:{dp}"], cwd=repo_dir
            )
        except Exception:
            head_content = ""

        if head_content and len(
            head_content.encode("utf-8", errors="replace")
        ) > _DELETED_INLINE_MAX_BYTES:
            notes.append(
                f"### {dp}\n\n*(DELETED — content > "
                f"{_DELETED_INLINE_MAX_BYTES // 1024} KB; suppressed)*\n"
            )
            continue

        if head_content:
            notes.append(
                f"### {dp}\n\n*(DELETED — content from HEAD)*\n\n"
                f"```{suffix}\n{head_content}\n```\n"
            )
        else:
            notes.append(
                f"### {dp}\n\n*(DELETED — HEAD content unavailable; "
                "see staged diff for removed lines)*\n"
            )

    joint = "\n".join(notes)
    if current_files_section.strip():
        return current_files_section + "\n\n" + joint
    return joint


def _compute_touched_status(
    current_files_section: str,
    deleted_paths: list,
    omitted: list,
    current_paths: list,
) -> Optional["_TouchedContextStatus"]:
    """Return touched-context failure status, or None when context is complete."""
    if not current_files_section.strip() and not deleted_paths:
        return _TouchedContextStatus(status="empty")
    if omitted and current_paths:
        return _TouchedContextStatus(status="omitted", omitted_paths=list(omitted))
    return None


def _gather_scope_packs(
    repo_dir: pathlib.Path,
    all_touched_paths: list,
    fixed_prompt_tokens: int = 0,
    drive_root: Optional[pathlib.Path] = None,
    degraded: bool = False,
    compact: bool = False,
    scope_model: str = "",
) -> str:
    """Collect the bounded wider repository atlas, failing closed on git errors."""
    # Canonical docs and touched files are injected explicitly; avoid duplicating them.
    already_included = frozenset(set(all_touched_paths) | set(_CANONICAL_CONTEXT_DOCS))
    _input_limit = _effective_scope_input_limit(degraded=degraded, scope_model=scope_model)
    try:
        atlas = compile_review_context_atlas(
            ReviewContextAtlasRequest(
                repo_dir=repo_dir,
                anchors=tuple(all_touched_paths),
                already_included=already_included,
                fixed_prompt_tokens=fixed_prompt_tokens,
                target_total_tokens=min(850_000, _input_limit),
                hard_total_tokens=_input_limit,
                include_tests=False,
                title="Generated Scope Atlas",
                drive_root=drive_root,
                compact_manifest=compact,
            )
        )
        _SCOPE_CONTEXT_MANIFEST.set(atlas.manifest)
        if atlas.status == "budget_exceeded":
            raise _ScopeAtlasBudgetExceeded(atlas.manifest)
        repo_pack_section = atlas.text or "(no additional repo files)"
    except _ScopeAtlasBudgetExceeded:
        raise
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"review_context_atlas error: {exc}") from exc

    return repo_pack_section


def _ladder_terminal_status(scope_model: str, token_count: int) -> "_TouchedContextStatus":
    """Terminal status when the guaranteed-fit ladder exhausts every step."""
    known_window = _scope_reviewer_window(scope_model)
    if known_window and known_window < _SCOPE_MODEL_CONTEXT_WINDOW:
        return _TouchedContextStatus(status="budget_exceeded", token_count=token_count)
    return _TouchedContextStatus(status="fixed_overflow", token_count=token_count)


def _render_touched_section(
    repo_dir: pathlib.Path,
    current_context_paths: list,
    deleted_paths: list,
    skipped_by_design: list,
    diff_only_paths: list,
) -> tuple:
    """Build the touched-files prompt section.

    ``diff_only_paths`` are degraded to an explicit disclosed note (their
    changes stay fully visible in the staged diff) — the guaranteed-fit
    ladder's step for oversized fixed parts.
    """
    kept = [path for path in current_context_paths if path not in diff_only_paths]
    section, pack_omitted = build_touched_file_pack(repo_dir, kept)
    section = _inline_deleted_file_pack(section, deleted_paths, repo_dir)
    if skipped_by_design:
        skip_note = (
            "## CURRENT FILE CONTEXT DEDUPLICATION NOTE\n"
            "The following touched files are not duplicated as full current-file "
            "snapshots because they are either canonical docs injected above or "
            "tests whose exact changes are visible in the staged diff below:\n"
            + "\n".join(f"- {path}" for path in skipped_by_design)
            + "\n"
        )
        section = section + "\n\n" + skip_note if section.strip() else skip_note
    if diff_only_paths:
        degrade_note = (
            "## TOUCHED FILE BUDGET DEGRADATION NOTE\n"
            "The full post-change snapshots of the following touched files were "
            "OMITTED to fit the reviewer input budget (largest files first). "
            "Their complete changes are still visible in the staged diff below; "
            "treat this as an explicit, disclosed omission of unchanged "
            "surrounding context, not a hidden gap:\n"
            + "\n".join(f"- {path}" for path in diff_only_paths)
            + "\n"
        )
        section = section + "\n\n" + degrade_note if section.strip() else degrade_note
    return section, pack_omitted


def _build_scope_history_section(scope_review_history: Optional[list]) -> str:
    """Format prior scope review rounds into a prompt section."""
    if not scope_review_history:
        return ""
    rounds = []
    for i, entry in enumerate(scope_review_history, 1):
        status = str(entry.get("status") or "responded").strip()
        label = (
            "BLOCKED" if entry.get("blocked")
            else status.upper() if status and status != "responded"
            else "PASSED"
        )
        parts = [f"Round {i}: {label}"]
        critical_findings = list(entry.get("critical_findings") or [])
        advisory_findings = list(entry.get("advisory_findings") or [])
        if critical_findings:
            parts.append("Critical findings:")
            for finding in critical_findings:
                parts.append(f"- {format_review_history_entry(finding, default_severity='critical')}")
        if advisory_findings:
            parts.append("Advisory findings:")
            for finding in advisory_findings:
                parts.append(f"- {format_review_history_entry(finding)}")
        if not critical_findings and not advisory_findings:
            parts.append(str(entry.get("summary") or "(no summary)"))
        rounds.append("\n".join(parts))
    return (
        "\n## Prior scope review rounds (your previous findings for this commit)\n\n"
        + "\n\n---\n".join(rounds)
        + "\n\nAddress any previously raised issues. If the same issue persists, "
        "mark it FAIL again with a reference to the prior round.\n"
        f"\nIMPORTANT: {_HISTORY_VERIFICATION_ONLY_RULE}\n"
        f"\nIMPORTANT: {_ANTI_THRASHING_RULE_VERDICT}\n"
    )


def _build_scope_prompt(
    repo_dir: pathlib.Path,
    commit_message: str,
    goal: str = "",
    scope: str = "",
    review_rebuttal: str = "",
    review_history: Optional[list] = None,
    scope_review_history: Optional[list] = None,
    drive_root: Optional[pathlib.Path] = None,
    degraded: bool = False,
    scope_model: str = "",
) -> tuple:
    """Build the scope prompt or a touched-context/budget status sentinel."""
    _SCOPE_CONTEXT_MANIFEST.set({})
    # Fail-closed (immune-system parity with the triad): a scope review running
    # WITHOUT its checklist silently reviews against nothing. Raising turns it
    # into an explicit SCOPE_REVIEW_BLOCKED error instead of a placeholder pass.
    scope_checklist = load_checklist_section("Intent / Scope Review Checklist")
    if not str(scope_checklist or "").strip():
        raise RuntimeError(
            "Intent / Scope Review Checklist could not be loaded from docs/CHECKLISTS.md — "
            "scope review cannot run without its checklist (fail-closed)."
        )

    goal_section = build_goal_section(goal, scope, commit_message)
    scope_section = build_scope_section(scope)
    canonical_docs = _load_canonical_context_docs(repo_dir)
    critical_calibration = CRITICAL_FINDING_CALIBRATION  # noqa: F841 — used in f-string below
    rebuttal_section = _shared_build_rebuttal_section(review_rebuttal)
    # Scope can block independently of triad, so load obligations even without triad history.
    _open_obs_for_scope = []
    _drive_root = pathlib.Path(drive_root) if drive_root else None
    if _drive_root is not None:
        try:
            from ouroboros.review_state import load_state, make_repo_key
            _rs = load_state(_drive_root)
            _repo_key = make_repo_key(repo_dir)
            _open_obs_for_scope = _rs.get_open_obligations(repo_key=_repo_key)
        except Exception:
            pass  # Non-fatal: best-effort hint
    history_section = _build_review_history_section(
        review_history or [], open_obligations=_open_obs_for_scope,
    )
    scope_history_section = _build_scope_history_section(scope_review_history)

    # Scope-only retry chains need the convergence rule even without triad history.
    if (
        scope_review_history
        and len(scope_review_history) >= 2
        and _CONVERGENCE_RULE_TEXT not in history_section
    ):
        scope_history_section = (
            (scope_history_section.rstrip() + "\n\n")
            if scope_history_section
            else ""
        ) + f"**IMPORTANT: {_CONVERGENCE_RULE_TEXT}**\n"

    try:
        diff_text = run_cmd(["git", "diff", "--cached"], cwd=repo_dir)
    except Exception:
        diff_text = "(failed to get staged diff)"

    touched_entries = _parse_staged_name_status(repo_dir)
    current_paths = [ep[1] for ep in touched_entries if ep[0] != "D"]
    deleted_paths = [ep[1] for ep in touched_entries if ep[0] == "D"]
    all_touched_paths = [ep[1] for ep in touched_entries]

    current_context_paths = [
        path for path in current_paths
        if not _should_skip_current_touched_context(path)
    ]
    current_skipped_by_design = [
        path for path in current_paths
        if _should_skip_current_touched_context(path)
    ]

    def _render_current_section(diff_only_paths: list) -> tuple:
        return _render_touched_section(
            repo_dir,
            current_context_paths,
            deleted_paths,
            current_skipped_by_design,
            diff_only_paths,
        )

    current_files_section, omitted = _render_current_section([])
    touched_status = _compute_touched_status(
        current_files_section, deleted_paths, omitted, current_context_paths
    )

    # Touched-file omissions fail closed before the budget skip can apply.
    if touched_status is not None:
        return None, touched_status

    repo_pack_placeholder = "__GENERATED_SCOPE_ATLAS_PENDING__"

    def _assemble_prompt(current_files_section: str) -> str:
        return f"""\
{REVIEW_PREAMBLE}

## Your role

You are the Atlas-backed whole-repository reviewer. Diff reviewers cover line-level mistakes;
you cover cross-module contracts, forgotten touchpoints, hidden regressions,
prompt/doc sync, architecture fit, and end-to-end intent completeness.

## Your task

For each finding, you MUST name the exact file, symbol, test, prompt, doc,
config, or sibling flow that proves the issue. Vague concerns without a
concrete artifact reference must be marked advisory, not critical.

## Output format

Output ONLY a valid JSON array.

You MUST cover every checklist item from the Intent / Scope Review
Checklist below. Skipping an item is not allowed — a missing entry
indicates the item was not actually reviewed.

The eight checklist item identifiers you MUST return (exactly these strings
in the "item" field; no substitutions):

    1. intent_alignment
    2. forgotten_touchpoints
    3. cross_surface_consistency
    4. regression_surface
    5. prompt_doc_sync
    6. architecture_fit
    7. cross_module_bugs
    8. implicit_contracts

Each element must follow the shared review JSON contract:
{REVIEW_JSON_ARRAY_CONTRACT}

Additional scope-review requirements:
- "item" must be one of the eight identifiers above — verbatim, case-sensitive.
- optional "obligation_id" when resolving or re-checking a previously surfaced obligation.
- "reason":
  - For FAIL: concrete artifact (file/symbol/line/contract) + what is wrong + how to fix.
  - For PASS: 1–2 sentences stating WHY this item passes, naming a concrete
    artifact or code path that you checked. A bare "PASS" or single-word
    reason without justification indicates the item was not actually
    reviewed and will be treated as a reviewer failure.

If one checklist item has multiple distinct concrete problems, return one
FAIL entry per distinct root cause. Do not compress unrelated bugs into a
single summary. If an item has no problems, return one PASS entry. Do not
return duplicate PASS entries, and do not return PASS for an item that also
has a FAIL — the concrete FAIL is authoritative.

Severity rules: critical requires a concrete current artifact and a required
change to this diff; otherwise use advisory. Scope affects only unchanged
legacy code outside the diff. Apply the `Critical surface whitelist` in
`docs/CHECKLISTS.md` for prose-vs-code mismatches.

If an open obligation record above already names an `obligation_id` for this root cause,
reuse that exact `obligation_id`. Do NOT invent a new id for the same root cause.

## Anti pattern-lock guard

{REPO_ANTI_PATTERN_LOCK_GUARD}

{critical_calibration}

{scope_checklist}
{scope_section}

{goal_section}

## Canonical Documentation Context

These files are always included explicitly. Do not treat their absence from the
wider repository pack as omission.

{canonical_docs}

{rebuttal_section}{history_section}{scope_history_section}

## Current touched files (post-change — what the file looks like NOW)

Files deleted by this diff appear here with an explicit `DELETED` marker and
their HEAD content inlined; other removed lines are visible via the staged
diff below. HEAD versions of modified files are not sent as a separate
section — the staged diff below already shows every `-` line.

{current_files_section}

## Staged diff

{diff_text}

## Wider repository context

{repo_pack_placeholder}
"""

    gather_signature = inspect.signature(_gather_scope_packs)
    gather_accepts_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in gather_signature.parameters.values()
    )
    gather_accepted = set(gather_signature.parameters)

    def _atlas_section(fixed_tokens: int, compact: bool) -> str:
        gather_kwargs = {
            "fixed_prompt_tokens": fixed_tokens,
            "drive_root": drive_root,
            "degraded": degraded,
            "scope_model": scope_model,
            "compact": compact,
        }
        return _gather_scope_packs(
            repo_dir,
            all_touched_paths,
            **(
                gather_kwargs
                if gather_accepts_kwargs
                else {key: value for key, value in gather_kwargs.items() if key in gather_accepted}
            ),
        )

    def _touched_token_estimate(path: str) -> int:
        try:
            return int((repo_dir / path).stat().st_size) // 4 + 64
        except OSError:
            return 0

    # Guaranteed-fit ladder: 1) full atlas; 2) compact atlas; 3) degrade the
    # largest touched files to diff-only (explicit disclosed note — their
    # changes stay fully visible in the staged diff); 4) only the irreducible
    # prompt (checklist + canonical docs + staged diff) not fitting remains,
    # which fails CLOSED (fixed_overflow), never a silent skip.
    input_limit = _effective_scope_input_limit(degraded=degraded, scope_model=scope_model)
    _atlas_min_allowance = 35_000  # manifest reserve + hard headroom, see review_context_atlas
    diff_only_paths: list = []
    degradable = sorted(
        current_context_paths,
        key=lambda path: -_touched_token_estimate(path),
    )
    compact = False
    last_known_tokens = 0
    while True:
        prompt = _assemble_prompt(current_files_section)
        fixed_prompt_tokens = estimate_tokens(prompt)
        atlas_text = None
        try:
            atlas_text = _atlas_section(fixed_prompt_tokens, compact)
        except _ScopeAtlasBudgetExceeded as exc:
            if not compact:
                compact = True
                try:
                    atlas_text = _atlas_section(fixed_prompt_tokens, True)
                except _ScopeAtlasBudgetExceeded as compact_exc:
                    last_known_tokens = int(compact_exc.manifest.get("estimated_total_tokens") or 0)
            else:
                last_known_tokens = int(exc.manifest.get("estimated_total_tokens") or 0)

        deficit = 0
        if atlas_text is not None:
            head, sep, tail = prompt.rpartition(repo_pack_placeholder)
            if not sep:
                raise RuntimeError("scope review atlas placeholder missing")
            prompt = head + atlas_text + tail
            prompt_tokens = estimate_tokens(prompt)
            last_known_tokens = prompt_tokens
            if prompt_tokens <= input_limit:
                return prompt, None
            if not compact:
                # Retry the same touched set with the compact atlas first.
                compact = True
                continue
            deficit = prompt_tokens - input_limit
        else:
            # Even the atlas manifest cannot fit beside the fixed part: shrink
            # the fixed part enough to give the manifest its minimum room.
            deficit = max(50_000, fixed_prompt_tokens + _atlas_min_allowance - input_limit)

        if not degradable:
            # Terminal split by verdict authority: the >=1M blocking reviewer
            # fails CLOSED (fixed_overflow — split the diff); a KNOWN sub-floor
            # reviewer is advisory-only and routes to the disclosed
            # non-blocking skip (Provider Independence for small-window slots).
            return None, _ladder_terminal_status(
                scope_model or _get_scope_model(),
                last_known_tokens or fixed_prompt_tokens,
            )
        freed = 0
        while degradable and freed < deficit + 2_000:
            path = degradable.pop(0)
            diff_only_paths.append(path)
            freed += _touched_token_estimate(path)
        current_files_section, _ = _render_current_section(diff_only_paths)


def _normalize_scope_items(items: list) -> tuple[list[dict], str]:
    """Validate and normalize the scope-review checklist coverage contract."""
    if not isinstance(items, list):
        return [], "reviewer output is not a JSON array"

    normalized: list[dict] = []
    seen_pass: set[str] = set()
    seen_fail: set[str] = set()
    seen_items: set[str] = set()
    unexpected: list[str] = []
    invalid: list[str] = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            invalid.append(f"entry {index} is not an object")
            continue
        item_id = str(item.get("item", "") or "").strip()
        verdict = str(item.get("verdict", "") or "").strip().upper()
        if item_id not in _SCOPE_REQUIRED_ITEMS:
            unexpected.append(item_id or f"<missing item at {index}>")
            continue
        if verdict not in {"PASS", "FAIL"}:
            invalid.append(f"{item_id}: invalid verdict {verdict!r}")
            continue
        severity = str(item.get("severity", "") or "").strip().lower()
        if verdict == "PASS" and not severity:
            # Severity is semantically void on PASS rows (it only classifies
            # FAIL blocking-ness), and reviewer models legitimately omit it
            # there — fable-5 emits severity on every FAIL but not on PASS.
            # Defaulting keeps the coverage contract about what matters; the
            # triad parser applies the same "advisory" default convention.
            severity = "advisory"
        if severity not in _SCOPE_VALID_SEVERITIES:
            # FAIL rows must carry an explicit valid severity — it decides
            # blocking, so a missing/garbled value stays fail-closed.
            invalid.append(f"{item_id}: missing or invalid severity {severity!r}")
            continue
        reason = str(item.get("reason", "") or "").strip()
        if not reason:
            invalid.append(f"{item_id}: missing reason")
            continue
        if verdict == "PASS":
            reason_words = [
                word.strip(".,;:!?()[]{}\"'")
                for word in reason.split()
                if word.strip(".,;:!?()[]{}\"'")
            ]
            if (
                reason.lower().strip(".!?:;") in {"pass", "ok", "okay", "yes", "n/a", "na", "none"}
                or len(reason_words) < 4
            ):
                invalid.append(f"{item_id}: PASS reason is too terse")
                continue
        if verdict == "PASS":
            if item_id in seen_pass:
                invalid.append(f"{item_id}: duplicate PASS")
            seen_pass.add(item_id)
        else:
            seen_fail.add(item_id)
        seen_items.add(item_id)

        normalized_item = dict(item)
        normalized_item["item"] = item_id
        normalized_item["verdict"] = verdict
        normalized_item["severity"] = severity
        normalized_item["reason"] = reason
        normalized.append(normalized_item)

    pass_and_fail = sorted(seen_pass & seen_fail)
    if pass_and_fail:
        invalid.append("items with both PASS and FAIL: " + ", ".join(pass_and_fail))
    missing = sorted(_SCOPE_REQUIRED_ITEMS - seen_items)
    errors: list[str] = []
    if missing:
        errors.append("missing required items: " + ", ".join(missing))
    if unexpected:
        errors.append("unexpected items: " + ", ".join(unexpected))
    if invalid:
        errors.append("invalid entries: " + "; ".join(invalid))
    return normalized, "; ".join(errors)


def _classify_scope_findings(items: list) -> tuple:
    """Classify raw JSON items into (critical_findings, advisory_findings) lists."""
    critical_findings: List[dict] = []
    advisory_findings: List[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict", "")).upper()
        severity = str(item.get("severity", "advisory")).lower()
        if verdict != "FAIL":
            continue
        finding = {
            "verdict": "FAIL",
            "severity": severity,
            "item": str(item.get("item", "scope_review")),
            "reason": str(item.get("reason", "")),
            "model": "scope_reviewer",
        }
        obligation_id = str(item.get("obligation_id", "") or "")
        if obligation_id:
            finding["obligation_id"] = obligation_id
        if severity == "critical":
            critical_findings.append(finding)
        else:
            advisory_findings.append(finding)
    return critical_findings, advisory_findings


def _log_scope_result(
    ctx: ToolContext,
    critical_count: int,
    advisory_count: int,
    prompt_chars: int = 0,
    prompt_tokens: int = 0,
    model_id: str = "",
    degraded: bool = False,
) -> None:
    """Append a scope_review_complete event to events.jsonl.

    Also emits budget headroom metrics so operators can see when the scope
    pack is approaching the gate. ``headroom_tokens`` is a signed delta
    (negative when the prompt exceeds the gate — would have been skipped).
    """
    prompt_tokens = int(prompt_tokens or 0)
    if prompt_tokens <= 0 and prompt_chars:
        prompt_tokens = max(0, int(prompt_chars) // 4)
    input_limit = _effective_scope_input_limit(degraded=degraded, scope_model=model_id)
    try:
        append_jsonl(ctx.drive_logs() / "events.jsonl", {
            "ts": utc_now_iso(), "type": "scope_review_complete",
            "task_id": getattr(ctx, "task_id", "") or "",
            "model": model_id or _get_scope_model(),
            "critical_count": critical_count,
            "advisory_count": advisory_count,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_budget": input_limit,
            "headroom_tokens": input_limit - prompt_tokens,
        })
    except Exception:
        pass


def _call_scope_llm(prompt: str, scope_model: str | None = None, ctx: ToolContext | None = None) -> tuple:
    """Execute the scope review LLM call synchronously.

    Returns (raw_text, usage, error_msg) — error_msg is non-empty on failure.
    ``usage`` may contain a private ``_review_refs`` entry with durable prompt
    and response refs from the shared review substrate.
    """
    from ouroboros.config import resolve_effort as _resolve_effort
    scope_model = scope_model or _get_scope_model()
    scope_effort = _resolve_effort("scope_review")
    # Output budget scales with the reviewer window: requesting the absolute
    # 100K reserve on a small-window model would 400 on input+max_tokens.
    _scope_output_tokens, _ = _window_scaled_reserves(_scope_reviewer_window(scope_model))
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": "Review the staged change and context above. Output ONLY a JSON array.",
        },
    ]
    try:
        from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

        request = ReviewRequest(
            surface="scope_review",
            goal="Review the staged change and context above. Output ONLY a JSON array.",
            messages=messages,
            task_id=str(getattr(ctx, "task_id", "") or "scope_review") if ctx is not None else "scope_review",
            call_type="scope_review",
            max_tokens=_scope_output_tokens,
            temperature=0.2,
            no_proxy=True,
        )
        slot = ReviewSlot(
            slot_id="scope_slot_1",
            model=scope_model,
            effort=scope_effort,
            timeout_sec=_SCOPE_REVIEW_SLOT_TIMEOUT_SEC,
            max_tokens=_scope_output_tokens,
            temperature=0.2,
            role_hint="scope reviewer",
        )
        result = run_review_request(
            request,
            slots=[slot],
            drive_root=review_drive_root(ctx),
            llm=LLMClient(),
            usage_ctx=None,
        )
        actor = (result.actors or [{}])[0]
        usage = dict(actor.get("usage") or {})
        usage["_review_refs"] = {
            "prompt_ref": actor.get("prompt_ref") or {},
            "response_ref": actor.get("response_ref") or {},
        }
        if actor.get("status") not in {"ok", "empty"}:
            error_msg = (
                f"⚠️ SCOPE_REVIEW_BLOCKED: Scope reviewer ({scope_model}) failed — commit blocked.\n"
                f"Error: {actor.get('error') or actor.get('status') or 'scope reviewer failed'}\n"
                "Retry the commit, or check API key and network connectivity."
            )
            return "", usage, error_msg
        return str(actor.get("raw_text") or ""), usage, ""
    except Exception as e:
        error_msg = (
            f"⚠️ SCOPE_REVIEW_BLOCKED: Scope reviewer ({scope_model}) failed — commit blocked.\n"
            f"Error: {type(e).__name__}: {e}\n"
            "Retry the commit, or check API key and network connectivity."
        )
        return "", None, error_msg


_PROVIDER_OVERSIZE_MARKERS = (
    # Anthropic: "prompt is too long: 1166914 tokens > 1000000 maximum"
    "prompt is too long",
    # Anthropic: "input length and `max_tokens` exceed context limit"
    "exceed context limit",
    # OpenAI error code + message variants
    "context_length_exceeded",
    "maximum context length",
)


def _is_provider_oversize_error(error_text: str) -> bool:
    """Mechanical fault classification: does this provider error mean the prompt
    exceeded the model's REAL context window? Deliberately tight markers — any
    other provider/transport error keeps the fail-closed blocking path."""
    low = str(error_text or "").lower()
    return any(marker in low for marker in _PROVIDER_OVERSIZE_MARKERS)


def _provider_error_is_oversize(usage: dict, prompt_tokens_est: int, scope_model: str) -> bool:
    """Gateway-route oversize detection from ``usage['provider_error']``."""
    pe = usage.get("provider_error") if isinstance(usage, dict) else None
    if not isinstance(pe, dict):
        return False
    try:
        code = int(pe.get("code") or 0)
    except (TypeError, ValueError):
        code = 0
    if code != 400:  # never 429/5xx (already rerouted as transient), never non-400
        return False
    # Non-empty 400 messages must explicitly say oversize; only opaque gateway 400s can
    # use size proximity, so auth/param/policy errors stay fail-closed.
    message = str(pe.get("message") or "").strip()
    if message:
        return _is_provider_oversize_error(message)
    try:
        input_limit = int(_effective_scope_input_limit(scope_model=scope_model) or 0)
    except Exception:
        input_limit = 0
    return input_limit > 0 and int(prompt_tokens_est or 0) >= int(0.8 * input_limit)


def _scope_oversize_advisory_result(
    *,
    scope_model_id: str,
    prompt_chars: int,
    prompt_tokens_est: int,
    prompt_ref: dict,
    response_ref: dict,
    provider_detail: str,
) -> "ScopeReviewResult":
    """The NON-blocking budget_exceeded advisory for a provider oversize rejection —
    shared by the raised-error (anthropic-direct ``prompt is too long``) and the
    empty-body gateway (``provider_error`` code=400) oversize paths so BOTH converge on
    one visible advisory instead of a fail-closed block (the real tokenizer is denser
    than the estimate gate; the pack cannot fit the reviewer window)."""
    return ScopeReviewResult(
        blocked=False,
        block_message="",
        status="budget_exceeded",
        model_id=scope_model_id,
        prompt_chars=prompt_chars,
        context_manifest=_current_scope_context_manifest(),
        prompt_ref=prompt_ref,
        response_ref=response_ref,
        advisory_findings=[{
            "verdict": "FAIL",
            "severity": "advisory",
            "item": "scope_review_skipped",
            "reason": (
                f"⚠️ SCOPE_REVIEW_SKIPPED: the provider rejected the assembled scope prompt "
                f"(~{prompt_tokens_est} estimated tokens) as exceeding the model's real "
                f"context window. Scope review downgraded to non-blocking warning. "
                "Provider error: "
                + _truncate_review_artifact(str(provider_detail), 1000)
            ),
            "model": scope_model_id,
        }],
    )


def _handle_prompt_signals(
    prompt: Optional[str],
    context_status: Optional["_TouchedContextStatus"],
    input_limit: int = _SCOPE_INPUT_TOKEN_LIMIT,
    scope_model: str = "",
) -> Optional[ScopeReviewResult]:
    """Translate touched-context status into an early ScopeReviewResult."""
    if context_status is None:
        return None  # proceed with LLM call

    if context_status.status == "budget_exceeded":
        token_count = context_status.token_count
        # Back-compute prompt chars from the budget-gate token estimate.
        _prompt_chars_est = token_count * 4
        # Report the REAL window-scaled reserves, not the 1M constants.
        _window = _scope_reviewer_window(scope_model) if scope_model else _SCOPE_MODEL_CONTEXT_WINDOW
        _output_reserve, _ = _window_scaled_reserves(_window)
        log.warning(
            "Scope review skipped: full scope-review prompt (~%d tokens) exceeds budget limit (%d). "
            "Scope review downgraded to non-blocking warning.",
            token_count, input_limit,
        )
        return ScopeReviewResult(
            blocked=False,
            block_message="",
            status="budget_exceeded",
            prompt_chars=_prompt_chars_est,
            advisory_findings=[{
                "verdict": "FAIL",
                "severity": "advisory",
                "item": "scope_review_skipped",
                "reason": (
                    f"⚠️ SCOPE_REVIEW_SKIPPED: Full scope-review prompt (~{token_count} tokens) "
                    f"exceeds the scope input budget ({input_limit} tokens, "
                    f"reserving {_output_reserve} for output within a {_window}-token window). "
                    "Scope review downgraded to non-blocking warning. "
                    "Consider reducing codebase size or splitting the review."
                ),
                "model": scope_model or "scope_reviewer",
            }],
        )

    if context_status.status == "fixed_overflow":
        # The guaranteed-fit ladder exhausted every degradation step: even with
        # all touched files reduced to diff-only and the atlas reduced to its
        # manifest, the irreducible prompt (checklist + canonical docs + staged
        # diff) exceeds the reviewer input budget. This is a structural
        # condition the owner must see — fail CLOSED, never a silent skip.
        token_count = context_status.token_count
        return ScopeReviewResult(
            blocked=True,
            status="fixed_overflow",
            prompt_chars=token_count * 4,
            block_message=(
                f"⚠️ SCOPE_REVIEW_BLOCKED: the irreducible scope prompt (checklist + canonical "
                f"docs + staged diff) is ~{token_count} estimated tokens and exceeds the scope "
                f"reviewer input budget ({input_limit}). Every touched file was already degraded "
                "to diff-only and the atlas to its manifest. Split the commit into smaller "
                "staged diffs, or configure a larger-window scope reviewer. "
                "Fail-closed stop — not a skippable budget condition."
            ),
        )

    if context_status.status == "empty":
        return ScopeReviewResult(
            blocked=True,
            status="empty",
            block_message=(
                "⚠️ SCOPE_REVIEW_BLOCKED: Could not read any touched files — "
                "scope review requires direct file context. Commit blocked."
            ),
        )

    if context_status.status == "omitted":
        omitted_names = ", ".join(context_status.omitted_paths) or "(unknown)"
        return ScopeReviewResult(
            blocked=True,
            status="omitted",
            block_message=(
                f"⚠️ SCOPE_REVIEW_BLOCKED: Some touched file(s) could not be included "
                f"in direct context (binary/oversize/unreadable): {omitted_names}.\n"
                "Scope review requires complete touched-file context. Commit blocked.\n"
                "Possible fixes: reduce file size, commit binary files separately, "
                "or ensure all touched files are readable text."
            ),
        )

    # Unknown status is a programming error; fail closed.
    log.error(
        "Scope review: unrecognised _TouchedContextStatus.status=%r — blocking commit (fail-closed).",
        context_status.status,
    )
    return ScopeReviewResult(
        blocked=True,
        status="error",
        block_message=(
            f"⚠️ SCOPE_REVIEW_BLOCKED: Unexpected context status '{context_status.status}' — "
            "commit blocked (fail-closed). This is a programming error; please report it."
        ),
    )


def _build_block_message(
    critical_findings: List[dict], advisory_findings: List[dict]
) -> str:
    """Format critical + advisory findings into a human-readable block message."""
    crit_lines = "\n".join(
        f"  CRITICAL: [scope:{f['item']}] {f['reason']}" for f in critical_findings
    )
    adv_section = ""
    if advisory_findings:
        adv_lines = "\n".join(
            f"  WARN: [scope:{f['item']}] {f['reason']}" for f in advisory_findings
        )
        adv_section = f"\n\nAdvisory warnings:\n{adv_lines}"
    return (
        "⚠️ SCOPE_REVIEW_BLOCKED: Scope reviewer found critical completeness issues.\n"
        "Commit has NOT been created. Fix the issues and try again.\n\n"
        + crit_lines + adv_section
    )


def run_scope_review(
    ctx: ToolContext,
    commit_message: str,
    goal: str = "",
    scope: str = "",
    review_rebuttal: str = "",
    review_history: Optional[list] = None,
    scope_review_history: Optional[list] = None,  # prior scope rounds for this commit
    scope_model: Optional[str] = None,
    degraded: bool = False,
) -> ScopeReviewResult:
    """Run normal blocking scope review or explicit supplemental degraded review."""
    repo_dir = pathlib.Path(ctx.repo_dir)
    scope_model_id = scope_model or _get_scope_model()

    try:
        prompt, context_status = _build_scope_prompt(
            repo_dir, commit_message,
            goal=goal, scope=scope,
            review_rebuttal=review_rebuttal,
            review_history=review_history,
            scope_review_history=scope_review_history,
            drive_root=pathlib.Path(ctx.drive_root) if getattr(ctx, "drive_root", None) else None,
            degraded=degraded,
            scope_model=scope_model_id,
        )
    except RuntimeError as exc:
        return ScopeReviewResult(
            blocked=True,
            block_message=(
                "⚠️ SCOPE_REVIEW_BLOCKED: Failed to build review context — commit blocked.\n"
                f"Error: {exc}\n"
                "Ensure git is available and the repository is in a valid state."
            ),
            model_id=scope_model_id,
            status="error",
            context_manifest=_current_scope_context_manifest(),
        )

    signal_result = _handle_prompt_signals(
        prompt,
        context_status,
        input_limit=_effective_scope_input_limit(degraded=degraded, scope_model=scope_model_id),
        scope_model=scope_model_id,
    )
    if signal_result is not None:
        # Keep _handle_prompt_signals as the status SSOT for early exits.
        signal_result.model_id = scope_model_id
        signal_result.context_manifest = _current_scope_context_manifest()
        return signal_result

    _prompt_chars = len(prompt)  # type: ignore[arg-type]
    _prompt_tokens_est = estimate_tokens(prompt)  # type: ignore[arg-type]
    raw_text, usage, llm_error = _call_scope_llm(prompt, scope_model=scope_model_id, ctx=ctx)  # type: ignore[arg-type]
    _usage = dict(usage or {})
    _review_refs = dict(_usage.pop("_review_refs", {}) or {})
    _prompt_ref = dict(_review_refs.get("prompt_ref") or {})
    _response_ref = dict(_review_refs.get("response_ref") or {})
    _tokens_in = int(_usage.get("prompt_tokens", 0) or 0)
    _tokens_out = int(_usage.get("completion_tokens", 0) or 0)
    _cost_usd = float(_usage.get("cost", 0.0) or 0.0)
    if llm_error:
        if _is_provider_oversize_error(llm_error):
            # The estimate-based budget gate passed but the provider's REAL
            # tokenizer rejected the prompt as oversize. This is the same
            # failure class as the pre-call budget_exceeded skip (pack cannot
            # fit the reviewer window), so it downgrades to the same VISIBLE
            # non-blocking advisory instead of a fail-closed block. Any other
            # provider/transport error stays blocking.
            log.warning(
                "Scope review skipped: provider rejected the prompt as oversize "
                "(estimate-gate passed; real tokenizer denser). Downgrading to "
                "non-blocking budget_exceeded. Error: %s", llm_error,
            )
            return _scope_oversize_advisory_result(
                scope_model_id=scope_model_id,
                prompt_chars=_prompt_chars,
                prompt_tokens_est=_prompt_tokens_est,
                prompt_ref=_prompt_ref,
                response_ref=_response_ref,
                provider_detail=llm_error,
            )
        return ScopeReviewResult(
            blocked=True,
            block_message=llm_error,
            model_id=scope_model_id,
            status="error",
            prompt_chars=_prompt_chars,
            context_manifest=_current_scope_context_manifest(),
            prompt_ref=_prompt_ref,
            response_ref=_response_ref,
        )
    if _usage:
        emit_review_usage(ctx, model=scope_model_id, usage=_usage, source="scope_review")

    if _provider_error_is_oversize(_usage, _prompt_tokens_est, scope_model_id):
        # Gateway route (openai-compatible/OpenRouter): a real oversize 400 arrives as
        # an EMPTY body + usage['provider_error']{code:400}, NOT a raised error carrying
        # the "prompt is too long" text — so the llm_error oversize branch above never
        # fires and the empty body would otherwise hard-block as empty_response. With
        # INDEPENDENT size evidence (see _provider_error_is_oversize), downgrade to the
        # SAME visible non-blocking budget_exceeded advisory the raised-error path uses.
        # A non-size 400 (auth/param/policy) lacks that evidence and stays blocking below.
        _pe_msg = str((_usage.get("provider_error") or {}).get("message") or "")
        log.warning(
            "Scope review skipped: gateway rejected the prompt as oversize via "
            "provider_error code=400 (empty body; estimate-gate passed). Downgrading to "
            "non-blocking budget_exceeded. provider_error: %s", _pe_msg or "(no message)",
        )
        return _scope_oversize_advisory_result(
            scope_model_id=scope_model_id,
            prompt_chars=_prompt_chars,
            prompt_tokens_est=_prompt_tokens_est,
            prompt_ref=_prompt_ref,
            response_ref=_response_ref,
            provider_detail=_pe_msg,
        )

    if not raw_text.strip():
        # Empty model response is distinct from transport/API error.
        return ScopeReviewResult(
            blocked=True,
            block_message=(
                "⚠️ SCOPE_REVIEW_BLOCKED: Scope reviewer returned empty response — commit blocked.\n"
                "Retry the commit."
            ),
            model_id=scope_model_id,
            status="empty_response",
            prompt_chars=_prompt_chars,
            tokens_in=_tokens_in,
            tokens_out=_tokens_out,
            cost_usd=_cost_usd,
            context_manifest=_current_scope_context_manifest(),
            prompt_ref=_prompt_ref,
            response_ref=_response_ref,
        )

    items = extract_json_array(raw_text, normalize=True)
    if items is None:
        return ScopeReviewResult(
            blocked=True,
            block_message=(
                "⚠️ SCOPE_REVIEW_BLOCKED: Could not parse scope reviewer output as JSON — commit blocked.\n"
                "Full raw response preserved in scope_raw_result (status='parse_failure')."
            ),
            model_id=scope_model_id,
            status="parse_failure",
            raw_text=raw_text,
            prompt_chars=_prompt_chars,
            tokens_in=_tokens_in,
            tokens_out=_tokens_out,
            cost_usd=_cost_usd,
            context_manifest=_current_scope_context_manifest(),
            prompt_ref=_prompt_ref,
            response_ref=_response_ref,
        )

    parsed_items, contract_error = _normalize_scope_items(items)
    if contract_error:
        return ScopeReviewResult(
            blocked=True,
            block_message=(
                "⚠️ SCOPE_REVIEW_BLOCKED: Scope reviewer output violated the "
                "Intent / Scope Review Checklist coverage contract — commit blocked.\n"
                f"{contract_error}\n"
                "Retry the commit so scope review covers all required checklist items."
            ),
            model_id=scope_model_id,
            status="parse_failure",
            raw_text=raw_text,
            parsed_items=parsed_items,
            prompt_chars=_prompt_chars,
            tokens_in=_tokens_in,
            tokens_out=_tokens_out,
            cost_usd=_cost_usd,
            context_manifest=_current_scope_context_manifest(),
            prompt_ref=_prompt_ref,
            response_ref=_response_ref,
        )

    critical_findings, advisory_findings = _classify_scope_findings(parsed_items)
    if degraded and _effective_scope_input_limit(degraded=True) == _LOW_SCOPE_INPUT_TOKEN_LIMIT:
        # Degraded low-context scope review is ADVISORY-ONLY (BIBLE P3): it ran on a
        # sub-floor reviewer over a partial surface, so it must NOT block. Its
        # findings are surfaced as advisory (the diff itself is still blocking-
        # reviewed by triad), and the degraded coverage is disclosed. The blocking
        # >=1M scope floor is therefore untouched — degraded never acts as the gate.
        for _f in critical_findings:
            _f["severity"] = "advisory"
            _f["reason"] = "[degraded scope review] " + str(_f.get("reason", ""))
        advisory_findings = list(critical_findings) + list(advisory_findings)
        critical_findings = []
        advisory_findings.append({
            "verdict": "FAIL",
            "severity": "advisory",
            "item": "scope_review_degraded",
            "reason": (
                "⚠️ SCOPE_REVIEW_DEGRADED: ran on a window-fitting repository pack "
                "(owner-selected low context mode + degraded review opt-in) and is "
                "ADVISORY-ONLY. The "
                "coverage manifest lists which files are full vs manifest-only — "
                "findings are real but full-content coverage is partial, so they "
                "do not block; the blocking >=1M scope floor is unchanged."
            ),
            "model": scope_model_id,
        })
    elif critical_findings:
        # BIBLE P3 floor: only a >=1M-window reviewer may act as the BLOCKING
        # scope gate. A scope model with a KNOWN sub-floor window (Capability
        # Evidence) — or an explicit OUROBOROS_SCOPE_REVIEW_FLOOR=advisory config —
        # still gets a right-sized pack and can respond, but its verdict authority
        # is advisory-only (it can NEVER satisfy a required blocking scope gate).
        from ouroboros.config import get_scope_review_floor as _scope_floor

        _known_window = _scope_reviewer_window(scope_model_id)
        _sub_floor = bool(_known_window and _known_window < _SCOPE_MODEL_CONTEXT_WINDOW)
        if _sub_floor or _scope_floor() == "advisory":
            for _f in critical_findings:
                _f["severity"] = "advisory"
                _f["reason"] = "[sub-floor scope reviewer] " + str(_f.get("reason", ""))
            advisory_findings = list(critical_findings) + list(advisory_findings)
            critical_findings = []
            advisory_findings.append({
                "verdict": "FAIL",
                "severity": "advisory",
                "item": "scope_review_sub_floor",
                "reason": (
                    f"⚠️ SCOPE_REVIEW_SUB_FLOOR: scope reviewer {scope_model_id} has a known "
                    f"{_known_window}-token context window, below the >=1M blocking scope floor "
                    "(BIBLE P3). Its findings are delivered ADVISORY-ONLY and cannot block the "
                    "commit; configure a >=1M-window scope model to restore the blocking gate."
                ),
                "model": scope_model_id,
            })
    _log_scope_result(
        ctx,
        len(critical_findings),
        len(advisory_findings),
        prompt_chars=_prompt_chars,
        prompt_tokens=_prompt_tokens_est,
        model_id=scope_model_id,
        degraded=degraded,
    )

    if critical_findings:
        from ouroboros import config as _cfg
        if _cfg.get_review_enforcement() == "blocking":
            return ScopeReviewResult(
                blocked=True,
                block_message=_build_block_message(critical_findings, advisory_findings),
                critical_findings=critical_findings,
                advisory_findings=advisory_findings,
                parsed_items=parsed_items,
                model_id=scope_model_id,
                status="responded",
                raw_text=raw_text,
                prompt_chars=_prompt_chars,
                tokens_in=_tokens_in,
                tokens_out=_tokens_out,
                cost_usd=_cost_usd,
                context_manifest=_current_scope_context_manifest(),
                prompt_ref=_prompt_ref,
                response_ref=_response_ref,
            )
        # Parallel review aggregates advisory findings on the main thread.

    return ScopeReviewResult(
        blocked=False,
        critical_findings=critical_findings,
        advisory_findings=advisory_findings,
        parsed_items=parsed_items,
        model_id=scope_model_id,
        status="responded",
        raw_text=raw_text,
        prompt_chars=_prompt_chars,
        tokens_in=_tokens_in,
        tokens_out=_tokens_out,
        cost_usd=_cost_usd,
        context_manifest=_current_scope_context_manifest(),
        prompt_ref=_prompt_ref,
        response_ref=_response_ref,
    )
