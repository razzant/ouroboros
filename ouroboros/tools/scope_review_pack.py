"""Assembly of the scope-review pack: touched context, atlas, guaranteed-fit ladder.

Owns everything that turns a staged change into the reviewer's prompt — the
canonical governance docs, the touched-file snapshots and deleted-file HEAD
content, the generated repo atlas, the prior-round history sections, and the
ladder that degrades the fixed part until the pack fits the input cap or
refuses. The context manifest and the stable-prefix boundary of the last
assembled prompt are recorded here; the caller reads them to publish evidence.
Extracted from ouroboros/tools/scope_review.py (v7 D06 split, re-cut on the
v7next tip); scope_review.py re-exports every name. Four prompt-vocabulary
names are read inside f-strings, which the call-time handle cannot carry —
they stay import-bound to their prompt-text owner; ``_load_canonical_context_docs``
stayed a facade def for the same reason (its f-string reads
``load_governance_doc``, which tests rebind on the parent).
"""

from __future__ import annotations

import contextvars
import inspect
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Optional

from ouroboros.tools.review_prompt_text import (
    _ANTI_THRASHING_RULE_VERDICT,
    _CONVERGENCE_RULE_TEXT,
    _HISTORY_VERIFICATION_ONLY_RULE,
    format_review_history_entry,
)
from ouroboros.tools.review_file_pack import CARRIER_CUT_REASON, span_only_release_carriers

# The parent logger name is pinned on purpose: records moved with their code
# keep the exact `%(name)s` every handler and reader saw before the split.
log = logging.getLogger("ouroboros.tools.scope_review")


def _sr():
    """The parent scope-review module, read at call time.

    The scope members stay monkeypatch-addressable at their historical
    ``ouroboros.tools.scope_review`` bindings (tests rebind them there), so
    this leaf resolves every such cross-reference through the module at each
    call instead of freezing whatever object a from-import saw at import time.
    """
    from ouroboros.tools import scope_review

    return scope_review


_DELETED_INLINE_MAX_BYTES = 1_048_576  # 1 MB

# The atlas coverage row of a span-only release carrier (owner decision, F3 Q4):
# the same disposition as a ladder-degraded snapshot, a by-design reason.
_CARRIER_DIFF_ONLY_REASON = (
    "changes included in the fixed staged diff; full snapshot omitted by design: "
    + CARRIER_CUT_REASON
)


_SCOPE_CONTEXT_MANIFEST = contextvars.ContextVar("scope_context_manifest", default={})


_SCOPE_STABLE_PREFIX_LEN = contextvars.ContextVar("scope_stable_prefix_len", default=0)


class _ScopeAtlasNotAssembled(RuntimeError):
    """The atlas did not assemble — an oversized pack, or an omitted REQUIRED artifact.

    Both are refusals under the BIBLE P3 scope floor: the ladder degrades the
    fixed part and retries, and scope review never runs on the remainder.
    """

    def __init__(self, manifest: dict, reason: str = ""):
        self.manifest = dict(manifest or {})
        token_count = int(self.manifest.get("estimated_total_tokens") or 0)
        super().__init__(
            "Generated Scope Atlas did not assemble: "
            + (
                reason
                or "exceeded hard budget"
                + (f" (~{token_count:,} estimated tokens)" if token_count else "")
            )
        )


def _current_scope_context_manifest() -> dict:
    return dict(_SCOPE_CONTEXT_MANIFEST.get({}) or {})


_CANONICAL_CONTEXT_DOCS = (
    "BIBLE.md",
    "docs/DEVELOPMENT.md",
    "docs/DESIGN.md",
    "docs/ARCHITECTURE.md",
    "docs/CHECKLISTS.md",
)


_CURRENT_TOUCHED_CONTEXT_SKIP_PREFIXES = (
    "tests/",
)


def _should_skip_current_touched_context(path: str) -> bool:
    """Touched paths whose full snapshots the fixed part omits by design: canonical
    docs (injected whole elsewhere) and tests/ paths (changes ride the staged diff;
    full atlas anchors, ladder-degradable — but never canonical docs)."""
    norm = str(path or "").replace("\\", "/").lstrip("./")
    return (
        norm in _CANONICAL_CONTEXT_DOCS
        or any(norm.startswith(prefix) for prefix in _CURRENT_TOUCHED_CONTEXT_SKIP_PREFIXES)
    )


def _build_review_history_section(history: list, open_obligations: list = None) -> str:
    """Format previous triad rounds for scope-review context."""
    return _sr()._shared_review_history_section(
        history,
        open_obligations,
        title="## Previous triad review rounds",
        include_commit_message=False,
        compact_labels=True,
    )


def _parse_staged_name_status(repo_dir: pathlib.Path) -> list:
    """Parse staged changes with rename/delete/copy awareness."""
    try:
        name_status_raw = _sr().run_cmd(
            ["git", "diff", "--cached", "--name-status"], cwd=repo_dir
        )
    except Exception:
        name_status_raw = ""

    entries = _sr().parse_git_name_status(name_status_raw)

    # Fallback to --name-only if --name-status produced nothing.
    if not entries:
        try:
            changed = _sr().run_cmd(["git", "diff", "--cached", "--name-only"], cwd=repo_dir)
            for p in changed.strip().splitlines():
                p = p.strip()
                if p:
                    entries.append(("M", p, p))
        except Exception:
            pass

    return entries


def _classify_deleted_for_inline(
    path: str, repo_dir: pathlib.Path, *, m0_tree: str = "", staged_tree: str = ""
) -> Optional[str]:
    """Return a suppression reason for deleted HEAD content, or None to inline.

    The trees (managed resolutions only) extend the binary probe to the
    reviewed M0→S delta — a binary the official target added and the resolver
    deleted is invisible to the HEAD-only ``--cached`` numstat but must still
    classify as binary. Non-managed callers pass nothing, byte-identical."""
    fp = pathlib.Path(path)
    fname_lower = fp.name.lower()
    suffix_lower = fp.suffix.lower()
    if suffix_lower in _sr()._SENSITIVE_EXTENSIONS or fname_lower in _sr()._SENSITIVE_NAMES:
        return "sensitive (env/credential/key)"
    if suffix_lower in _sr().BINARY_EXTENSIONS:
        return "binary extension"
    return "binary content" if _sr().staged_path_is_binary(
        repo_dir, path, m0_tree=m0_tree, staged_tree=staged_tree) else None


def _degradable_diff_only_paths(repo_dir: pathlib.Path, current: list,
                                skipped: list, deleted: list,
                                renamed: frozenset = frozenset(), *,
                                m0_tree: str = "", staged_tree: str = "") -> list:
    """Touched paths the ladder may hand to the diff-only tier. Current paths
    join freely EXCEPT atlas-required-beyond-diff ones: the atlas refuses a
    diff-only required artifact by design (typed ``budget_omitted`` assembly
    failure), so degrading one can only convert the pack into a refusal, never
    into a fitting pack — a self-defeating rung, removed. Touched TESTS —
    skipped-by-design current ones and deleted ones — join the free tier too,
    with cheap conservative guards: atlas-required tests never degrade; binary and
    RENAMED paths keep their snapshot/metadata (the staged text diff may not carry
    their change); an oversized/sensitive deletion keeps its suppression marker."""

    def _degradable_test(p: str, is_deleted: bool) -> bool:
        if _sr().atlas_required_beyond_diff(p.replace("\\", "/").lstrip("./")):
            return False
        if p in renamed or _sr().staged_path_is_binary(
                repo_dir, p, m0_tree=m0_tree, staged_tree=staged_tree):
            return False
        if is_deleted:
            try:
                head_bytes = int(_sr().run_cmd(["git", "cat-file", "-s", f"HEAD:{p}"], cwd=repo_dir))
            except Exception:
                return False
            return (
                head_bytes <= _DELETED_INLINE_MAX_BYTES
                and _should_skip_current_touched_context(p)
                and _classify_deleted_for_inline(
                    p, repo_dir, m0_tree=m0_tree, staged_tree=staged_tree) is None
            )
        return True

    return (
        [
            p for p in current
            if not _sr().atlas_required_beyond_diff(p.replace("\\", "/").lstrip("./"))
        ]
        + [p for p in skipped if _degradable_test(p, False)]
        + [p for p in deleted if _degradable_test(p, True)]
    )


def _inline_deleted_file_pack(
    current_files_section: str,
    deleted_paths: list,
    repo_dir: pathlib.Path,
    *,
    represent_binary: bool = False,
    diff_only_paths: Optional[list] = None,
    m0_tree: str = "",
    staged_tree: str = "",
) -> str:
    """Append deleted-file HEAD content or explicit suppression markers;
    ``diff_only_paths`` members skip the HEAD inline (ladder-degraded): a text
    deletion's complete content is the staged diff's own minus-lines."""
    if not deleted_paths:
        return current_files_section

    notes: list[str] = []
    for dp in deleted_paths:
        suffix = pathlib.Path(dp).suffix.lstrip(".") or "text"
        if dp in (diff_only_paths or ()):
            notes.append(
                f"### {dp}\n\n*(DELETED — full HEAD snapshot omitted to fit the "
                "reviewer input budget; the complete removal is visible in the "
                "staged diff below)*\n"
            )
            continue
        suppress_reason = _classify_deleted_for_inline(
            dp, repo_dir, m0_tree=m0_tree, staged_tree=staged_tree)
        if suppress_reason is not None:
            if represent_binary and suppress_reason.startswith("binary"):
                from ouroboros.tools.review_binary_context import render_staged_binary_metadata

                metadata = render_staged_binary_metadata(repo_dir, dp, m0_tree=m0_tree)
                if metadata is None:
                    raise RuntimeError(f"deleted binary {dp} has no exact staged Git metadata")
                notes.append(f"### {dp}\n\n{metadata}\n")
                continue
            notes.append(
                f"### {dp}\n\n*(DELETED — {suppress_reason}; content suppressed)*\n"
            )
            continue

        try:
            head_content = _sr().run_cmd(["git", "show", f"HEAD:{dp}"], cwd=repo_dir)
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


def _gather_scope_packs(
    repo_dir: pathlib.Path,
    all_touched_paths: list,
    fixed_prompt_tokens: int = 0,
    drive_root: Optional[pathlib.Path] = None,
    compact: bool = False,
    scope_model: str = "",
    diff_only_paths: Optional[list] = None,
    snapshot_included_paths: Optional[frozenset] = None,
    diff_only_reasons: Optional[dict] = None,
) -> str:
    """Collect the bounded wider repository atlas, failing closed on git errors."""
    # WHICH snapshots the fixed part holds is the assembler's fact, never re-derived
    # from the touched LIST: `all_touched_paths` also names files the fixed part
    # omits by design (touched tests) or suppresses (sensitive/oversized deletion) —
    # claiming those would be a false coverage claim (BIBLE P1) that also hides them
    # from requiredness classification. A canonical doc is claimed only if it exists.
    already_included = frozenset(
        set(snapshot_included_paths or frozenset())
        | {doc for doc in _CANONICAL_CONTEXT_DOCS if (repo_dir / doc).is_file()}
    )
    _input_limit = _sr()._effective_scope_input_limit(scope_model=scope_model)
    try:
        atlas = _sr().compile_review_context_atlas(
            _sr().ReviewContextAtlasRequest(
                repo_dir=repo_dir,
                anchors=tuple(all_touched_paths),
                already_included=already_included,
                diff_only_included=frozenset(diff_only_paths or ()),
                diff_only_reasons=dict(diff_only_reasons or {}),
                fixed_prompt_tokens=fixed_prompt_tokens,
                target_total_tokens=min(850_000, _input_limit),
                hard_total_tokens=_input_limit,
                include_tests=False,
                title="Generated Scope Atlas",
                drive_root=drive_root,
                compact_manifest=compact,
            )
        )
        # Set the manifest FIRST: disclosure accompanies the refusal, never replaces it (P3).
        _SCOPE_CONTEXT_MANIFEST.set(atlas.manifest)
        if _sr().atlas_assembly_failed(atlas):
            raise _ScopeAtlasNotAssembled(atlas.manifest, _sr().atlas_assembly_failure_reason(atlas))
        repo_pack_section = atlas.text or "(no additional repo files)"
    except RuntimeError:  # includes _ScopeAtlasNotAssembled
        raise
    except Exception as exc:
        raise RuntimeError(f"review_context_atlas error: {exc}") from exc

    return repo_pack_section


def _record_ladder_steps(steps: list) -> None:
    """Attach the aggregated guaranteed-fit ladder trace to the context manifest."""
    if not steps:
        return
    manifest = dict(_SCOPE_CONTEXT_MANIFEST.get({}) or {})
    manifest["ladder_steps"] = list(steps)
    _SCOPE_CONTEXT_MANIFEST.set(manifest)


def _carrier_span_only_paths(
    repo_dir: pathlib.Path, current_context_paths: list, managed_subject: Any,
) -> list:
    """The touched release carriers the scope pack cuts by design (owner
    decision, F3 Q4): the commit triad's span-only cut over the SAME HEAD→index
    pair this pack reviews. Such a carrier keeps no full snapshot — it is named
    in the dedup note, declared diff-only to the atlas with that reason and
    traced as the ladder's first entry. Never for a managed subject (its delta
    is M0→staged, not HEAD→staged) and never for an artifact the atlas owes in
    full (canonical docs never reach this list; the guard is structural, so
    the cut can never turn into a refusal)."""
    if managed_subject is not None:
        return []
    return [
        p for p in span_only_release_carriers(repo_dir, current_context_paths)
        if not _sr().atlas_required_beyond_diff(p)
    ]


def _render_touched_section(
    repo_dir: pathlib.Path,
    current_context_paths: list,
    deleted_paths: list,
    skipped_by_design: list,
    diff_only_paths: list,
    *,
    represent_binary: bool = False,
    m0_tree: str = "",
    staged_tree: str = "",
    carrier_span_only: Optional[list] = None,
) -> tuple:
    """Build the touched-files prompt section.

    ``diff_only_paths`` are degraded to an explicit disclosed note (changes stay
    fully visible in the staged diff) — the guaranteed-fit ladder's step.
    ``carrier_span_only`` are the release carriers cut by design (their
    complete change is the staged diff); they are named in the dedup note.
    Returns ``(section, pack_omitted, snapshot_included)``; the latter is the
    CONSERVATIVE set of paths whose full snapshot this section really carries, so
    no coverage row can claim content the pack does not hold (BIBLE P1)."""
    kept = [path for path in current_context_paths if path not in diff_only_paths]
    section, pack_omitted = _sr().build_touched_file_pack(
        repo_dir, kept, represent_binary=represent_binary,
        m0_tree=m0_tree, staged_tree=staged_tree,
    )
    section = _inline_deleted_file_pack(
        section, deleted_paths, repo_dir,
        represent_binary=represent_binary, diff_only_paths=diff_only_paths,
        m0_tree=m0_tree, staged_tree=staged_tree,
    )
    # A ladder-degraded touched test moves to the degradation note below; listing
    # it HERE too would claim an atlas snapshot the pack no longer holds.
    skip_listed = [p for p in skipped_by_design if p not in diff_only_paths]
    carriers = list(carrier_span_only or ())
    if skip_listed or carriers:
        skip_note = "## CURRENT FILE CONTEXT DEDUPLICATION NOTE\n"
        if skip_listed:
            skip_note += (
                "The following touched files are not duplicated as full current-file "
                "snapshots HERE because they are either canonical docs injected above "
                "or tests whose exact changes are visible in the staged diff below. "
                "A touched test listed here is delegated to the generated atlas (full "
                "snapshot, or a typed binary/oversize row); tests degraded to diff-only "
                "under budget pressure move to the degradation note instead:\n"
                + "\n".join(f"- {path}" for path in skip_listed)
                + "\n"
            )
        if carriers:
            skip_note += (
                f"Release carrier(s) omitted by design — {CARRIER_CUT_REASON}; their "
                "complete change is the staged diff below and the atlas records them "
                "diff-only with that reason:\n"
                + "\n".join(f"- {path}" for path in carriers)
                + "\n"
            )
        section = section + "\n\n" + skip_note if section.strip() else skip_note
    if diff_only_paths:
        degrade_note = (
            "## TOUCHED FILE BUDGET DEGRADATION NOTE\n"
            "The full snapshots (post-change; HEAD content for deletions) of the "
            "following touched files were OMITTED to fit the budget (freely "
            "degradable first, largest per tier). Their complete changes are still "
            "visible in the staged diff below; treat this as an explicit, disclosed "
            "omission of unchanged surrounding context, not a hidden gap:\n"
            + "\n".join(f"- {path}" for path in diff_only_paths)
            + "\n"
        )
        section = section + "\n\n" + degrade_note if section.strip() else degrade_note
    # Only paths that CANNOT be absent: kept, not omitted by the pack builder, and a
    # real file on disk. Deleted paths are never claimed — they leave the index.
    snapshot_included = frozenset(
        path for path in kept
        if path not in set(pack_omitted) and (repo_dir / path).is_file()
    )
    return section, pack_omitted, snapshot_included


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


@dataclass(frozen=True)
class _ScopePromptContext:
    drive_root: Optional[pathlib.Path] = None
    scope_model: str = ""
    governance_repo_dir: Optional[pathlib.Path] = None
    represent_binary: bool = False
    # The managed resolution-delta artifact (review_subject.ManagedReviewSubject);
    # None for every ordinary commit — the pack then reads the staged diff.
    managed_subject: Optional[Any] = None


def _build_scope_prompt(
    repo_dir: pathlib.Path,
    commit_message: str,
    goal: str = "",
    scope: str = "",
    review_rebuttal: str = "",
    review_history: Optional[list] = None,
    scope_review_history: Optional[list] = None,
    context: Optional[_ScopePromptContext] = None,
) -> tuple:
    """Build the scope prompt or a touched-context/budget status sentinel."""
    context = context or _ScopePromptContext()
    drive_root = context.drive_root
    scope_model = context.scope_model
    governance_repo_dir = context.governance_repo_dir
    represent_binary = context.represent_binary
    _SCOPE_CONTEXT_MANIFEST.set({})
    # Missing checklist is fail-closed, matching the triad.
    scope_checklist = _sr().load_checklist_section("Intent / Scope Review Checklist")
    if not str(scope_checklist or "").strip():
        raise RuntimeError(
            "Intent / Scope Review Checklist could not be loaded from docs/CHECKLISTS.md — "
            "scope review cannot run without its checklist (fail-closed)."
        )

    goal_section = _sr().build_goal_section(goal, scope, commit_message)
    scope_section = _sr().build_scope_section(scope)
    canonical_docs = _sr()._load_canonical_context_docs(
        pathlib.Path(governance_repo_dir or repo_dir)
    )
    rebuttal_section = _sr()._shared_build_rebuttal_section(review_rebuttal)
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

    subject = context.managed_subject
    # Hardened, byte-exact, fail-closed: it raises rather than yield a placeholder.
    # A managed resolution reviews its disclosed resolution-delta artifact (Δ4).
    diff_text = (
        subject.render_prompt_diff() if subject is not None
        else _sr().capture_staged_diff(repo_dir)
    )

    if subject is not None:
        # Touched set = resolution delta ∪ conflict anchors: an anchor stays a
        # reviewed path even when the resolver left it byte-identical to M0.
        delta_paths = {p for _s, p in subject.name_status}
        touched_entries = [(s, p, p) for s, p in subject.name_status] + [
            ("M", p, p) for p in sorted(set(subject.conflict_paths) - delta_paths)
        ]
    else:
        touched_entries = _parse_staged_name_status(repo_dir)
    current_paths = [ep[1] for ep in touched_entries if ep[0] != "D"]
    deleted_paths = [ep[1] for ep in touched_entries if ep[0] == "D"]
    all_touched_paths = [ep[1] for ep in touched_entries]
    renamed_paths = frozenset(
        ep[1] for ep in touched_entries if str(ep[0]).upper().startswith("R"))

    current_context_paths = [
        p for p in current_paths if not _should_skip_current_touched_context(p)
    ]
    current_skipped_by_design = [
        p for p in current_paths if _should_skip_current_touched_context(p)
    ]
    carrier_span_only = _carrier_span_only_paths(repo_dir, current_context_paths, subject)
    current_context_paths = [p for p in current_context_paths if p not in carrier_span_only]

    def _render_current_section(diff_only_paths: list) -> tuple:
        return _render_touched_section(
            repo_dir, current_context_paths, deleted_paths,
            current_skipped_by_design, diff_only_paths, represent_binary=represent_binary,
            m0_tree=getattr(subject, "m0_tree", "") or "",
            staged_tree=getattr(subject, "staged_tree", "") or "",
            carrier_span_only=carrier_span_only,
        )

    current_files_section, omitted, snapshot_included = _render_current_section([])
    touched_status = _sr()._compute_touched_status(
        current_files_section, deleted_paths, omitted, current_context_paths
    )

    # Touched-file omissions fail closed before the budget skip can apply.
    if touched_status is not None:
        return None, touched_status

    repo_pack_placeholder = "__GENERATED_SCOPE_ATLAS_PENDING__"

    def _assemble_prompt(current_files_section: str) -> str:
        prompt_text, stable_len = _sr().build_scope_review_prompt(
            current_files_section,
            scope_checklist=scope_checklist,
            canonical_docs=canonical_docs,
            intent_context=f"{scope_section}\n\n{goal_section}",
            history_block=f"{rebuttal_section}{history_section}{scope_history_section}",
            diff_text=diff_text,
            repo_pack_placeholder=repo_pack_placeholder,
            critical_calibration=_sr().CRITICAL_FINDING_CALIBRATION,
        )
        _SCOPE_STABLE_PREFIX_LEN.set(stable_len)
        return prompt_text

    gather_signature = inspect.signature(_gather_scope_packs)
    gather_accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in gather_signature.parameters.values()
    )
    gather_accepted = set(gather_signature.parameters)

    def _atlas_section(fixed_tokens: int, compact: bool) -> str:
        gather_kwargs = {
            "fixed_prompt_tokens": fixed_tokens, "drive_root": drive_root,
            "scope_model": scope_model, "compact": compact,
            # The ladder owns which snapshots survived; the atlas is TOLD —
            # the by-design carriers ride the same diff-only channel, reasoned.
            "diff_only_paths": list(diff_only_paths) + list(carrier_span_only),
            "diff_only_reasons": {p: _CARRIER_DIFF_ONLY_REASON for p in carrier_span_only},
            "snapshot_included_paths": snapshot_included,
        }
        return _gather_scope_packs(
            repo_dir, all_touched_paths,
            **(gather_kwargs if gather_accepts_kwargs
               else {k: v for k, v in gather_kwargs.items() if k in gather_accepted}),
        )

    def _touched_token_estimate(path: str) -> int:
        try:
            return int((repo_dir / path).stat().st_size) // 4 + 64
        except OSError:  # deleted: the fixed part inlines the HEAD blob instead
            try:
                return int(_sr().run_cmd(["git", "cat-file", "-s", f"HEAD:{path}"], cwd=repo_dir)) // 4 + 64
            except Exception:
                return 0

    # Guaranteed-fit ladder: compact atlas (the default form); degrade degradable
    # touched files to diff-only (largest first); drop unchanged diff context.
    # Else CLOSED — atlas-required-beyond-diff artifacts never degrade to
    # diff-only, because the atlas refuses such a pack by design.
    input_limit = _sr()._effective_scope_input_limit(scope_model=scope_model)
    _atlas_min_allowance = 35_000  # rendered-manifest + hard headroom allowance, see review_context_atlas
    diff_only_paths: list = []
    # FREE tier includes touched tests and eligible deletions (guards in the helper).
    degradable = sorted(
        _degradable_diff_only_paths(
            repo_dir, current_context_paths, current_skipped_by_design, deleted_paths,
            renamed_paths,
            m0_tree=getattr(subject, "m0_tree", "") or "",
            staged_tree=getattr(subject, "staged_tree", "") or ""),
        key=lambda path: -_touched_token_estimate(path),
    )
    compact = True
    compact_diff_attempted = False
    last_known_tokens = 0
    unassembled_required: list = []
    atlas_overflowed = False
    # One AGGREGATED ladder record (RS5); a silent ladder is unexplainable (BIBLE P1).
    # A by-design carrier cut is its first, typed entry.
    ladder_steps: list = [{
        "step": "carrier_span_only_omitted", "paths": list(carrier_span_only),
        "reason": _CARRIER_DIFF_ONLY_REASON,
    }] if carrier_span_only else []
    while True:
        prompt = _assemble_prompt(current_files_section)
        fixed_prompt_tokens = _sr().estimate_tokens(prompt)
        atlas_text = None
        try:
            atlas_text = _atlas_section(fixed_prompt_tokens, compact)
        except _ScopeAtlasNotAssembled as refusal:
            last_known_tokens = int(refusal.manifest.get("estimated_total_tokens") or 0)
            # The atlas manifest is the ONE carrier of what did not assemble; a
            # refusal is a ladder STEP (P1) that can carry TWO causes — capture both.
            unassembled_required = [
                str(row.get("path") or "?") for row in _sr().atlas_unassembled_required(refusal.manifest)
            ]
            atlas_overflowed = _sr().atlas_hard_budget_overflowed(refusal.manifest)
            ladder_steps.append({
                "step": "atlas_refused", "compact": compact, "reason": str(refusal),
                "unassembled_required": list(unassembled_required),
                "atlas_overflowed": atlas_overflowed,
                "tokens_after": last_known_tokens,
                "diff_only_files": len(diff_only_paths),
                "diff_only_paths": list(diff_only_paths),
                "zero_context_diff": compact_diff_attempted,
            })

        deficit = 0
        if atlas_text is not None:
            head, sep, tail = prompt.rpartition(repo_pack_placeholder)
            if not sep:
                raise RuntimeError("scope review atlas placeholder missing")
            prompt = head + atlas_text + tail
            prompt_tokens = _sr().estimate_tokens(prompt)
            unassembled_required = []  # assembled: no earlier refusal is the cause now
            atlas_overflowed = False
            ladder_steps.append({
                "step": "compact_atlas",
                "tokens_before": last_known_tokens,
                "tokens_after": prompt_tokens,
                "diff_only_files": len(diff_only_paths),
                "diff_only_paths": list(diff_only_paths),
                "zero_context_diff": compact_diff_attempted,
                "deficit": max(0, prompt_tokens - input_limit),
            })
            last_known_tokens = prompt_tokens
            if prompt_tokens <= input_limit:
                _record_ladder_steps(ladder_steps)
                return prompt, None
            deficit = prompt_tokens - input_limit
        else:
            # Even the manifest cannot fit beside the fixed part: shrink it for room.
            deficit = max(50_000, fixed_prompt_tokens + _atlas_min_allowance - input_limit)

        # Degradable never holds atlas-required-beyond-diff paths: the atlas
        # refuses a diff-only required artifact by design, so that rung could
        # only convert this pack into a typed refusal, never into a fit.
        def can_degrade() -> bool:
            return bool(degradable)

        if not can_degrade():
            if not compact_diff_attempted:  # every +/- line, no unchanged context
                compact_diff_attempted = True
                try:
                    compact_diff = (
                        subject.render_prompt_diff(unified=0) if subject is not None
                        else _sr().capture_staged_diff(repo_dir, unified=0)
                    )
                except _sr().StagedDiffUnavailable:
                    compact_diff = ""  # the full capture above stays the evidence
                if compact_diff.strip() and compact_diff != diff_text:
                    diff_text = compact_diff
                    continue
            # Terminal pack status: >=1M authority is fixed_overflow; a sub-floor pack is
            # budget_exceeded (blocked unless owner advisory). CAUSE travels separately.
            _record_ladder_steps(ladder_steps)
            known = _sr()._scope_window(
                scope_model or _sr()._get_scope_model()
            ).sizing_window(_sr()._SCOPE_FAILCLOSED_WINDOW)
            return None, _sr()._TouchedContextStatus(
                status="budget_exceeded" if known and known < _sr()._SCOPE_MODEL_CONTEXT_WINDOW else "fixed_overflow",
                token_count=last_known_tokens or fixed_prompt_tokens,
                unassembled_required=list(unassembled_required),
                atlas_overflowed=bool(atlas_overflowed),
            )
        freed = 0
        while can_degrade() and freed < deficit + 2_000:
            path = degradable.pop(0)
            diff_only_paths.append(path)
            freed += _touched_token_estimate(path)
        # Re-render AND re-read what the shrunken section now holds: a freshly
        # degraded path is no survivor, and the next atlas build must know that.
        current_files_section, _, snapshot_included = _render_current_section(diff_only_paths)
