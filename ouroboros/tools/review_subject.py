"""The managed-resolution review subject: the resolution delta M0 -> S.

For an ordinary commit the review subject is the staged diff. For a
managed-update resolution commit — the merge that lands an already-released
official target into the local line — the staged diff is the WHOLE two-parent
candidate (the entire official delta plus the resolver's work), which
structurally overflows every packet and asks reviewers to re-review released
code. The declared review subject for that commit is the RESOLUTION DELTA:
``git diff <M0> <S>`` where

- ``M0`` is the mechanical merge baseline pinned ONCE into the durable update
  tx at materialization (phase 0 of the update-flow redesign; conflict markers
  are plain file content), and
- ``S`` is the final candidate tree, and its definition FOLLOWS THE SURFACE:
  the COMMIT GATE (triad, scope, the -U0 fit rung) serializes the REAL index
  (``git write-tree``) — the exact tree the review-binding fingerprint pins and
  the commit will write, so the reviewed subject is bound to the tree that
  commits (and is stable and cheap across every gate consumer within one
  attempt); the ADVISORY pre-review serializes the live worktree through a
  private index (``supervisor.update_candidate.worktree_snapshot_tree``) —
  advisory reviews work-in-progress by contract, exactly like the non-managed
  staged+unstaged advisory capture, and advisory freshness handles staleness.

Everything a reviewer needs to know about that substitution is DISCLOSED in the
artifact header: both tree identities, both real merge parents, the conflict
anchors, and two counters (full candidate paths vs reviewed resolution paths).
Review binding, advisory freshness, preflight staged lists, doc-only
classification and the scope snapshot key stay on the FULL candidate — this
module only changes what the reviewers read, never what the gate fingerprints.

``capture_review_diff`` is the shared capture used by every review consumer:
byte-identical to ``capture_staged_diff`` for non-managed callers, the
resolution-delta artifact for the authorized managed resolver.

This module also renders the triad SESSION task (``build_triad_session_task``):
the session delivery of the same subject, where the managed artifact is inlined
instead of asking the session to retrieve ``git diff --cached`` itself.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pathlib
import subprocess
from typing import Any, List, Optional, Tuple

from ouroboros.tools import review_binary_context as _rbc
from ouroboros.tools.review_binary_context import StagedDiffUnavailable

# The staged capture stays resolvable through its owning module at call time:
# ``review_binary_context.capture_staged_diff`` is a documented patch seam.

log = logging.getLogger(__name__)

# The exact hardened flag tail capture_staged_diff pins (external diff drivers,
# textconv, colour and prefix rewrites are operator config that would make the
# rendered artifact stop describing the reviewed trees).
_HARDENED_DIFF_FLAGS = (
    "--no-ext-diff", "--no-textconv", "--no-color",
    "--src-prefix=a/", "--dst-prefix=b/",
)


def _git_bytes(repo_dir, args: List[str]) -> Tuple[int, bytes, str]:
    """Run git in ``repo_dir`` without GIT_DIFF_OPTS; return (rc, stdout, stderr)."""
    env = {k: v for k, v in os.environ.items() if k != "GIT_DIFF_OPTS"}
    try:
        result = subprocess.run(
            ["git", *args], cwd=str(repo_dir),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, b"", repr(exc)
    return result.returncode, result.stdout or b"", (
        (result.stderr or b"").decode("utf-8", "replace").strip()
    )


def _decode_diff(raw: bytes) -> str:
    """Strict-then-disclosed decoding, exactly like ``capture_staged_diff``."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        rendered = raw.decode("utf-8", "backslashreplace")
        return (
            f"{rendered}\n\n*(diff contained non-UTF-8 bytes; they are "
            "rendered above as backslash escapes)*\n"
        )


def _tree_delta_diff(repo_dir, m0_tree: str, staged_tree: str, unified: int) -> str:
    """The hardened resolution delta between the two pinned trees."""
    rc, raw, err = _git_bytes(repo_dir, [
        "diff", *_HARDENED_DIFF_FLAGS, f"--unified={int(unified)}",
        m0_tree, staged_tree,
    ])
    if rc != 0:
        raise StagedDiffUnavailable(
            f"resolution delta capture failed (rc {rc}): {err or 'no detail'}"
        )
    return _decode_diff(raw)


def _tree_delta_name_status(repo_dir, m0_tree: str, staged_tree: str) -> Tuple[Tuple[str, str], ...]:
    """(status, path) rows of the resolution delta (rename target path for R rows)."""
    rc, raw, err = _git_bytes(
        repo_dir, ["diff", "--name-status", "-z", m0_tree, staged_tree]
    )
    if rc != 0:
        raise StagedDiffUnavailable(
            f"resolution delta name-status failed (rc {rc}): {err or 'no detail'}"
        )
    fields = [f.decode("utf-8", "replace") for f in raw.split(b"\0")]
    if fields and not fields[-1]:
        fields.pop()  # trailing NUL terminator
    rows: List[Tuple[str, str]] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        # rename/copy rows carry TWO paths (src, dst); the reviewed identity is dst
        width = 3 if status[:1].upper() in ("R", "C") else 2
        if index + width - 1 >= len(fields):
            break  # malformed tail: keep the well-formed prefix
        rows.append((status, fields[index + width - 1]))
        index += width
    return tuple(rows)


def _full_candidate_path_count(repo_dir, staged_tree: str) -> Optional[int]:
    """Paths the full candidate changes vs the local pre-update HEAD (the count a
    whole-tree review would have had to cover). ``None`` when git itself failed:
    a failed count must render as "n/a", never masquerade as a real 0."""
    rc, raw, _err = _git_bytes(
        repo_dir, ["diff", "--name-only", "-z", "HEAD", staged_tree]
    )
    if rc != 0:
        return None
    return sum(1 for f in raw.split(b"\0") if f)


def _live_unmerged_paths(repo_dir) -> Optional[List[str]]:
    """cwd-aware twin of ``supervisor.update_candidate.live_unmerged_paths``:
    the live unmerged inventory, ``None`` when git itself failed (an unreadable
    inventory must never masquerade as "no conflicts")."""
    rc, raw, _err = _git_bytes(
        repo_dir, ["diff", "--name-only", "--diff-filter=U", "-z"]
    )
    if rc != 0:
        return None
    return [f.decode("utf-8", "replace") for f in raw.split(b"\0") if f]


def _real_index_tree(repo_dir) -> str:
    """Serialize the REAL index (``git write-tree``) — the exact tree
    ``_fingerprint_staged_diff`` pins and the commit will write. Fails closed
    (an unmerged index, or any git failure): a commit-gate subject must never
    be built from a tree that cannot commit."""
    rc, raw, err = _git_bytes(repo_dir, ["write-tree"])
    tree = raw.decode("ascii", "replace").strip()
    if rc != 0 or not tree:
        raise StagedDiffUnavailable(
            f"staged index tree could not be serialized (git write-tree rc {rc}): "
            f"{err or 'no detail'}"
        )
    return tree


@dataclasses.dataclass
class ManagedReviewSubject:
    """The structured resolution-delta artifact every managed review consumer reads."""

    repo_dir: str
    m0_tree: str                 # "" when the tx recorded m0_missing_reason
    staged_tree: str             # S — final candidate tree
    m0_missing_reason: str
    pre_update_sha: str
    target_sha: str
    conflict_paths: Tuple[str, ...]   # tx.conflict_paths ∪ live unmerged inventory
    diff: str                    # delta body at the requested context width
    name_status: Tuple[Tuple[str, str], ...]
    # Disclosed counter: paths of the full candidate; None = the count itself
    # failed and renders as "n/a" (a fake 0 would claim an empty candidate).
    full_candidate_paths: Optional[int]
    resolution_paths: int        # disclosed counter: paths of the reviewed delta
    fallback_full_diff: bool     # True → M0 missing, diff is the FULL HEAD→S diff
    unified: int = 3
    # Which S this subject serializes — "gate" (the real index: the staged
    # candidate) or "advisory" (the live worktree snapshot). Prose that names
    # the candidate must stay surface-correct: on the advisory surface the
    # full candidate is the WORKTREE, not the staged index.
    surface: str = "gate"
    # Per-subject cache of re-rendered bodies keyed by width (C5 tail): the
    # -U0 fit rung re-renders per consumer, and both trees are pinned on this
    # subject, so a width's body is immutable for its lifetime. The subject
    # itself is ctx-memoized per attempt — the cache inherits that
    # invalidation. Excluded from comparison/repr: it is not identity.
    _render_cache: dict = dataclasses.field(default_factory=dict, repr=False, compare=False)

    def _candidate_noun(self) -> str:
        return "staged candidate" if self.surface != "advisory" else "worktree candidate"

    def counters_line(self) -> str:
        full = (
            "n/a (count unavailable)" if self.full_candidate_paths is None
            else str(self.full_candidate_paths)
        )
        if self.fallback_full_diff:
            # No M0 baseline → no resolution delta exists to count: claiming a
            # number here would present the full candidate as a reviewed delta.
            return (
                f"full candidate paths: {full}; "
                "reviewed resolution paths: n/a (M0 missing — full candidate "
                "under review)"
            )
        return (
            f"full candidate paths: {full}; "
            f"reviewed resolution paths: {self.resolution_paths}"
        )

    def touched_paths(self) -> List[str]:
        """Reviewed path set: resolution-delta paths ∪ conflict anchors."""
        paths = {path for _status, path in self.name_status}
        paths.update(self.conflict_paths)
        return sorted(paths)

    def header(self, body_rendered: bool = True) -> str:
        """The disclosure header. ``body_rendered=False`` is the SESSION variant
        of the M0-missing fallback, where NO diff body follows the header: the
        text must instruct retrieval instead of claiming a rendering below."""
        anchors = ", ".join(self.conflict_paths) or "(none)"
        if not self.fallback_full_diff:
            lead_tail = (
                f"diff = final candidate vs mechanical merge M0 {self.m0_tree[:12]}; "
                "the official base→target delta is already-released code and is "
                "not re-rendered."
            )
        elif body_rendered:
            # M0 missing WITH a body below: the claim must match what follows —
            # the FULL candidate is rendered, official delta included.
            lead_tail = (
                "M0 is unavailable, so the diff rendered below is the FULL "
                f"{self._candidate_noun()} vs the local pre-update HEAD — it "
                "INCLUDES the already-released official base→target delta."
            )
        else:
            lead_tail = (
                "M0 is unavailable, so the review subject is the FULL "
                f"{self._candidate_noun()} — it INCLUDES the already-released "
                "official base→target delta."
            )
        lines = [
            "## Managed-update resolution review subject",
            "",
            "This commit is a managed-update resolution: the merge of an "
            "already-released official target into the local line. "
            + lead_tail,
            "",
            f"- merge parents: local pre-update HEAD `{self.pre_update_sha[:12] or 'unknown'}` "
            f"+ official target `{self.target_sha[:12] or 'unknown'}`",
            f"- mechanical merge baseline M0 (tree): `{self.m0_tree or 'unavailable'}`",
            f"- final candidate tree S: `{self.staged_tree}`",
            f"- {self.counters_line()}",
            f"- conflict anchors ({len(self.conflict_paths)}): {anchors}",
        ]
        if not self.fallback_full_diff:
            # Accepted Δ2 residual, disclosure only: M0 is trusted as pinned.
            lines.append(
                "- M0 provenance: pinned once into the durable update tx at "
                "materialization — a forensic baseline that is NOT re-verified "
                "at review time."
            )
        if self.fallback_full_diff:
            if body_rendered:
                fallback_tail = (
                    f"the FULL {self._candidate_noun()} diff is rendered below "
                    "instead of the resolution delta — it includes the entire "
                    "already-released official delta."
                )
            else:
                fallback_tail = (
                    "no resolution delta exists to inline — retrieve the FULL "
                    "staged candidate diff yourself (`git diff --cached`); it "
                    "includes the entire already-released official delta."
                )
            lines += [
                "",
                "⚠️ M0 BASELINE UNAVAILABLE "
                f"({self.m0_missing_reason or 'no recorded reason'}): {fallback_tail}",
            ]
        if not self.fallback_full_diff and not self.diff.strip():
            lines += [
                "",
                "(the resolution delta is empty: the final candidate is byte-identical "
                "to the mechanical merge M0 — the resolver made no edits beyond it)",
            ]
        return "\n".join(lines)

    def render_prompt_diff(self, unified: Optional[int] = None) -> str:
        """Header + delta body; recomputes the body for a non-default width.

        The M0-missing fallback body (every width, both surfaces) diffs the
        PINNED subject tree S — ``git diff HEAD..S`` — never a fresh ``--cached``
        capture: on the advisory surface (S = worktree snapshot) a ``--cached``
        body would omit the unstaged changes the counters and name-status
        describe, and on the gate a second capture would weaken the binding to
        the pinned S. Header, name-status, counters and body all describe the
        same S."""
        width = self.unified if unified is None else int(unified)
        if width == self.unified:
            body = self.diff
        elif width in self._render_cache:
            body = self._render_cache[width]
        else:
            base = "HEAD" if self.fallback_full_diff else self.m0_tree
            body = _tree_delta_diff(self.repo_dir, base, self.staged_tree, width)
            self._render_cache[width] = body
        return f"{self.header()}\n\n{body}"


def managed_review_subject(
    ctx: Any, repo_dir, surface: str = "gate"
) -> Optional[ManagedReviewSubject]:
    """Build the resolution-delta subject for the AUTHORIZED managed resolver.

    Returns ``None`` for every non-managed caller (including ``ctx=None``), so
    the non-managed review path stays byte-identical to today. The managed
    predicate is the protected registry authority — called, never reimplemented.
    Any git failure while building the managed artifact raises
    ``StagedDiffUnavailable`` (the same fail-closed channel as the staged
    capture): a review must not run authoritatively on a placeholder.

    ``surface`` selects the S definition (module docstring): ``"gate"`` (the
    default — triad, scope, the -U0 fit rung) serializes the REAL index, the
    exact tree the review-binding fingerprint pins and the commit writes, and
    records it on the ctx for the gate's binding assertion; ``"advisory"``
    (pre-review only) serializes the live worktree — advisory reviews
    work-in-progress by contract.
    """
    if ctx is None:
        return None
    try:
        from ouroboros.tools.registry import _authorized_managed_update_resolver

        authorized = _authorized_managed_update_resolver(ctx)
    except Exception:
        # An exception ESCAPING the predicate (programming/import error — its
        # internal evidence handling normally resolves to False + a typed
        # marker) must not silently downgrade a genuinely managed task to an
        # ordinary staged-diff review. Cheap existence probe (no parse): a
        # present managed-update tx marker says this repo APPEARS mid-update,
        # so fail loudly on the staged-capture channel; absent marker — log
        # loudly and stay non-managed (an ordinary commit must never be
        # blocked by a managed-code bug).
        log.warning("managed review subject: authority predicate failed", exc_info=True)
        marker_present = False
        try:
            from supervisor.update_merge import _update_tx_marker_path

            marker_present = _update_tx_marker_path().is_file()
        except Exception:
            marker_present = False
        if marker_present:
            raise StagedDiffUnavailable(
                "managed-update authority predicate crashed while a managed "
                "update tx marker is present — the review subject cannot be "
                "determined (not proven managed, not proven ordinary)"
            )
        return None
    if not authorized:
        # "Not the resolver" is only trustworthy when the authority EVIDENCE was
        # readable. The predicate marks an unreadable read on the ctx (typed
        # marker, cleared on every successful evaluation): reviewing then would
        # either present a managed candidate as an ordinary staged diff or an
        # ordinary diff under a possibly-active managed tx — fail loudly on the
        # same channel as the staged capture instead (mutative tools are blocked
        # closed in that state anyway, see _managed_update_code_tool_block).
        read_error = str(getattr(ctx, "_managed_authority_read_error", "") or "")
        if read_error:
            raise StagedDiffUnavailable(
                "managed-update authority evidence is unreadable — the review "
                "subject cannot be determined (not proven managed, not proven "
                f"ordinary): {read_error}"
            )
        return None

    # From here on the caller IS the authorized managed resolver: a failed or
    # empty tx read must NEVER silently degrade to the non-managed full staged
    # capture (official code would be reviewed as resolver work). It becomes
    # the LOUD M0-missing fallback subject instead.
    tx_error = ""
    try:
        from supervisor.update_merge import authorized_assisted_task

        tx = authorized_assisted_task(
            getattr(ctx, "task_id", ""), getattr(ctx, "task_metadata", None)
        )
    except Exception as exc:
        log.warning(
            "managed review subject: tx read failed for the authorized resolver",
            exc_info=True,
        )
        tx, tx_error = None, f"tx_unreadable: {exc!r}"
    if not tx:
        tx = {}
        tx_error = tx_error or (
            "tx_missing: the authorized resolver's update transaction "
            "could not be read"
        )

    if surface == "gate":
        staged_tree = _real_index_tree(repo_dir)
        # Defense-in-depth: the commit gate asserts (typed failure) that every
        # tree a gate subject carried equals the binding fingerprint's tree_sha
        # — the reviewed subject is provably the tree that commits.
        trees = getattr(ctx, "_last_review_subject_trees", None)
        if not isinstance(trees, set):
            trees = set()
            try:
                setattr(ctx, "_last_review_subject_trees", trees)
            except Exception:
                pass
        trees.add(staged_tree)
    else:
        from supervisor.update_candidate import worktree_snapshot_tree

        staged_tree, tree_error = worktree_snapshot_tree("HEAD", cwd=str(repo_dir))
        if not staged_tree:
            raise StagedDiffUnavailable(
                f"managed candidate tree S could not be serialized: {tree_error}"
            )
    m0_tree = "" if tx_error else str(tx.get("m0_tree") or "")
    m0_missing_reason = tx_error or str(tx.get("m0_missing_reason") or "")
    # Per-attempt memo (C5): the gate S serialization is cheap, but the M0→S
    # diff/name-status/counting below rebuild identically for every consumer
    # of one attempt (triad + each scope row + advisory). Key = the exact
    # subject identity; invalidated wherever the attempt resets
    # ``_last_review_subject_trees`` (and per advisory pre-review). A memo hit
    # returns the SAME built subject — no consumer-visible content changes.
    # The M0-MISSING fallback is deliberately NOT memoized: its identity
    # includes the volatile failure reason (tx_unreadable vs tx_missing vs a
    # recorded m0_missing_reason), which the tree-based key cannot see.
    memo_key = (str(repo_dir), m0_tree, staged_tree, surface, 3)
    memo = getattr(ctx, "_managed_review_subject_memo", None)
    if m0_tree and isinstance(memo, dict):
        cached = memo.get(memo_key)
        if cached is not None:
            return cached
    anchors = {str(p) for p in (tx.get("conflict_paths") or []) if str(p).strip()}
    live = _live_unmerged_paths(repo_dir)
    if live is not None:
        anchors.update(live)
    if m0_tree:
        diff = _tree_delta_diff(repo_dir, m0_tree, staged_tree, 3)
        name_status = _tree_delta_name_status(repo_dir, m0_tree, staged_tree)
        fallback = False
    else:
        # M0 missing: the review subject IS the full candidate. Body, path set
        # and counters are ALL rendered from the pinned subject tree S — never
        # a fresh `--cached` capture, which on the advisory surface (S =
        # worktree snapshot) would omit the unstaged changes the counters
        # describe and on the gate would weaken the binding to the pinned S.
        # The path set covers exactly what the diff and the commit contain —
        # never just the conflict anchors (which would narrow packs, touched
        # context and the syntax preflight while the diff shows everything).
        diff = _tree_delta_diff(repo_dir, "HEAD", staged_tree, 3)
        name_status = _tree_delta_name_status(repo_dir, "HEAD", staged_tree)
        fallback = True
    subject = ManagedReviewSubject(
        repo_dir=str(repo_dir),
        m0_tree=m0_tree,
        staged_tree=staged_tree,
        m0_missing_reason=m0_missing_reason,
        pre_update_sha=str(tx.get("pre_update_sha") or ""),
        target_sha=str(tx.get("target_sha") or ""),
        conflict_paths=tuple(sorted(anchors)),
        diff=diff,
        name_status=name_status,
        full_candidate_paths=_full_candidate_path_count(repo_dir, staged_tree),
        resolution_paths=len(name_status),
        fallback_full_diff=fallback,
        surface=surface,
    )
    if not m0_tree:
        return subject  # fallback subjects are never cached (see above)
    if not isinstance(memo, dict):
        memo = {}
        try:
            ctx._managed_review_subject_memo = memo
        except Exception:
            return subject  # a ctx that cannot carry the memo simply rebuilds
    memo[memo_key] = subject
    return subject


def capture_review_diff(ctx: Any, repo_dir, *, unified: int = 3) -> str:
    """The review diff exactly as the reviewer must see it, for EVERY consumer.

    Non-managed: byte-identical to ``capture_staged_diff``. The authorized
    managed resolver: the disclosed resolution-delta artifact (header + delta;
    the ``unified`` parameter mirrors the staged capture's ladder rungs).
    """
    subject = managed_review_subject(ctx, repo_dir)
    if subject is None:
        return _rbc.capture_staged_diff(pathlib.Path(repo_dir), unified=unified)
    return subject.render_prompt_diff(unified=unified)


_SESSION_SUBJECT_RETRIEVE = (
    "## Subject (session delivery)\n"
    "The review subject is the STAGED diff of the repository you are running "
    "in. Retrieve it yourself with whatever your read-only tools allow: if you "
    "can run commands, `git diff --cached` (and `git diff --cached --name-only` "
    "for the file list); if your read-only mode withholds command execution — it "
    "commonly does — read the touched files directly and compare them against "
    "`.git`. Read the touched files as needed either way."
)


def _session_subject_section(subject: Optional[ManagedReviewSubject]) -> str:
    """The session task's Subject block: retrieval instructions for an ordinary
    commit, the INLINED authoritative artifact for a managed resolution. When
    the M0 baseline is missing the delta does not exist, so the session keeps
    retrieving itself — with the disclosure header prepended."""
    if subject is None:
        return _SESSION_SUBJECT_RETRIEVE
    if subject.fallback_full_diff:
        # No body follows in session delivery: the header must say "retrieve",
        # never "rendered below".
        return f"{subject.header(body_rendered=False)}\n\n{_SESSION_SUBJECT_RETRIEVE}"
    return (
        "## Subject (session delivery — inlined managed resolution delta)\n"
        "The inlined artifact below is the AUTHORITATIVE review subject for this "
        "managed-update resolution commit. Judge it as rendered; do NOT "
        "substitute your own `git diff --cached` — the staged diff is the whole "
        "two-parent merge candidate and re-renders already-released code. Read "
        "the touched files with your own tools as needed.\n\n"
        + subject.render_prompt_diff()
    )


def build_triad_session_task(*, goal_section: str, scope_section: str,
                             checklist_section: str, rebuttal_section: str,
                             review_history_section: str, dev_guide_text: str,
                             architecture_text: str,
                             governance_repo_dir: Optional[Any] = None,
                             subject: Optional[ManagedReviewSubject] = None) -> str:
    """The commit-triad task in SESSION delivery (5.2/5.3): the SAME preamble,
    calibration, checklist and goal/scope/history the api pack carries — but no
    assembled evidence. The subject is a pointer (the session takes the staged
    diff itself) — except for a managed resolution, whose authoritative delta
    artifact is inlined — and the governance docs arrive as navigation maps (5.7)."""
    from ouroboros.context_layout import book_navigation, generate_doc_nav_map
    from ouroboros.reference_books import BOOK_ENTRYPOINTS, load_reference_book
    from ouroboros.tools.review_helpers import (
        CRITICAL_FINDING_CALIBRATION,
        REPO_ANTI_PATTERN_LOCK_GUARD,
        REVIEW_PREAMBLE,
    )

    # The supplied texts are the COMPOSED books, so mapping them against the
    # entrypoint path would hand the session offsets into a file that holds a
    # membership list. With a governance root the map is built from the book and
    # addresses each physical chapter; without one (a synthetic or historical
    # input) the supplied text is mapped as the single source it is.
    nav_maps: list[str] = []
    for book_id, rel, title, text in (
        ("development", BOOK_ENTRYPOINTS["development"], "DEVELOPMENT.md", dev_guide_text),
        ("architecture", BOOK_ENTRYPOINTS["architecture"], "ARCHITECTURE.md", architecture_text),
    ):
        if not str(text or "").strip():
            continue
        if governance_repo_dir is not None:
            try:
                nav_maps.append(book_navigation(load_reference_book(governance_repo_dir, book_id)))
                continue
            except (OSError, ValueError):
                pass  # Fall back to the supplied text rather than drop the doc.
        nav_maps.append(generate_doc_nav_map(text, title=title, rel_path=rel))
    return "\n\n".join(part for part in [
        REVIEW_PREAMBLE,
        CRITICAL_FINDING_CALIBRATION,
        REPO_ANTI_PATTERN_LOCK_GUARD,
        checklist_section,
        goal_section,
        scope_section,
        rebuttal_section,
        review_history_section,
        _session_subject_section(subject),
        "## Governance context (navigation maps)\n"
        "Read BIBLE.md and docs/DESIGN.md in full from the repository root "
        "(DESIGN.md is short). The maps below index "
        "the other governance docs by line range; the paths are relative to the "
        "repository root — read the sections you need with your own tools.",
        *nav_maps,
    ] if str(part or "").strip())
