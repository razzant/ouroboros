"""The change-relative required-source manifest of the scope review.

What a scope reviewer is OWED IN FULL for THIS change: every touched protected
runtime path, frozen contract and prompt, the derived families of such a path,
and the declared cross-language twins. A merely-touched ordinary file is not
owed in full — its complete change evidence is the staged diff, and its body is
one ``read_file`` away.

The manifest is a MINIMUM, never a sufficiency claim: the brief says so, and the
reviewer reads whatever else it judges necessary. Each row carries the exact
identity a ``read_file`` receipt stamps (``source_revision`` over the file bytes,
``complete_sha256``/``complete_chars`` over the universal-newline text), so the
native episode folds its observed reads over the manifest and the host learns
which required sources were covered.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ouroboros.runtime_mode_policy import (
    FROZEN_CONTRACT_PATH_PREFIXES,
    GIT_OPS_FAMILY_PATHS,
    SAFETY_CRITICAL_PATHS,
    normalize_repo_path,
    protected_path_category,
)

# Manifest policy identity. It rides the commit gate's review-contract
# fingerprint, so a change to what a scope reviewer is owed lapses recorded
# free-replay authority instead of surviving it.
SCOPE_REQUIRED_SOURCES_POLICY = "v2"

# The logical root every row addresses — ``read_file``'s own default root, so a
# reviewer that opens a required source without naming a root already matches.
REQUIRED_SOURCE_ROOT = "active_workspace"

# The range basis the reader stamps on every delivered extent.
RANGE_BASIS = "unicode_text_universal_newlines"

# A touched prompt is owed in full exactly like a protected runtime path: the
# runtime prompts are behaviour, and only ``prompts/SAFETY.md`` is named in the
# protected inventory.
PROMPT_PATH_PREFIX = "prompts/"

# Derived families, read from the protection constants so a future leaf joins
# without a second list here: the git_ops family, and the tool-dispatch surface
# (the registry facade, its typed leaves and the extension dispatcher) whose
# safety-critical members all live under ouroboros/tools/. Touching one member
# owes the reviewer the whole family, because the behaviour was split across it.
_DERIVED_FAMILIES: Tuple[frozenset, ...] = (
    GIT_OPS_FAMILY_PATHS,
    frozenset(path for path in SAFETY_CRITICAL_PATHS if path.startswith("ouroboros/tools/")),
)

# Declared cross-language twins: one contract, two languages. Extend the table
# when another pair of files states the same contract on both sides.
_WEB_CONTRACT_TWIN = "web/modules/api_types.js"
_HOST_CONTRACT_TWIN = "ouroboros/gateway/contracts.py"
_DECLARED_TWINS: Dict[str, Tuple[str, ...]] = {
    _HOST_CONTRACT_TWIN: (_WEB_CONTRACT_TWIN,),
    _WEB_CONTRACT_TWIN: (_HOST_CONTRACT_TWIN,),
}

# Git name-status letters as the manifest's own dispositions.
_STATUS_DISPOSITIONS = {"A": "added", "C": "added", "D": "deleted"}


def staged_touched_paths(repo_dir: Any, subject: Any = None) -> List[Tuple[str, str]]:
    """``(status, path)`` pairs of the reviewed change.

    A managed resolution states its own reviewed path set (the resolution delta
    plus its conflict anchors); every other subject reads the staged index.
    """
    from ouroboros.tools.review_file_pack import parse_git_name_status
    from ouroboros.utils import run_cmd

    declared = getattr(subject, "name_status", None) if subject is not None else None
    pairs: List[Tuple[str, str]] = []
    diff_refs = ["--cached"]
    if declared is not None:
        pairs = [(str(status or "M"), normalize_repo_path(path)) for status, path in declared]
        named = {path for _status, path in pairs}
        pairs.extend(("M", normalize_repo_path(path))
                     for path in (getattr(subject, "conflict_paths", None) or ())
                     if normalize_repo_path(path) not in named)
        if not any(status[:1] == "R" for status, _path in pairs) or not getattr(subject, "staged_tree", ""):
            return pairs
        # The managed subject's name-status rows name rename targets only.
        # Recover their old sides from the same pinned trees, never the live index.
        diff_refs = [str(getattr(subject, "m0_tree", "") or "HEAD"), str(subject.staged_tree)]

    try:
        raw = run_cmd(["git", "diff", "--name-status", *diff_refs], cwd=pathlib.Path(repo_dir))
    except Exception:
        return pairs
    for status, current_path, source_path in parse_git_name_status(raw):
        if declared is None:
            pairs.append((status, normalize_repo_path(current_path)))
        if status == "R" and source_path != current_path:
            # A rename removes the preimage path: the reviewer is owed it under
            # the same rule that owes it any other deletion.
            pairs.append(("D", normalize_repo_path(source_path)))
    return pairs


def staged_tree_identity(repo_dir: Any, subject: Any = None) -> str:
    """The candidate tree the manifest is computed against (``""`` when unknown)."""
    if subject is not None and str(getattr(subject, "staged_tree", "") or ""):
        return str(subject.staged_tree)
    from ouroboros.utils import run_cmd

    try:
        return run_cmd(["git", "write-tree"], cwd=pathlib.Path(repo_dir)).strip()
    except Exception:
        return ""


def source_text_identity(raw: bytes) -> Dict[str, Any]:
    """The identity a ``read_file`` receipt stamps for exact source bytes.

    ``source_revision`` names the actual bytes, including CRLF; the character
    ranges address the universal-newline text the reader delivers.
    """
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return {
        "source_revision": hashlib.sha256(raw).hexdigest(),
        "complete_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "complete_chars": len(text),
        "range_basis": RANGE_BASIS,
    }


def with_inline_sources(rows: Sequence[Dict[str, Any]], documents: Dict[str, str]) -> List[Dict[str, Any]]:
    """Credit only a whole document whose delivered text matches the required source."""
    result = []
    for source in rows:
        row = dict(source)
        text = documents.get(str(row.get("path") or ""))
        if isinstance(text, str) and row.get("coverage_basis") in (None, "candidate_blob"):
            identity = source_text_identity(text.encode("utf-8"))
            if (identity["complete_sha256"] == row.get("complete_sha256")
                    and identity["complete_chars"] == row.get("complete_chars")):
                row["coverage_basis"] = "delivered_inline"
        result.append(row)
    return result


def scope_required_sources(
    repo_dir: Any,
    touched_paths: Optional[Sequence[Tuple[str, str]]] = None,
    *,
    staged_tree_sha: str = "",
    subject: Any = None,
    layer: str = "body",
) -> List[Dict[str, Any]]:
    """The required-source rows for one staged change, sorted by path.

    ``touched_paths`` are ``(git status letter, repo-relative path)`` pairs;
    ``None`` reads the reviewed change from ``subject`` or the staged index.
    ``staged_tree_sha`` binds every identity to the candidate tree it was read
    from, so a drifted read is visible in the durable coverage row. A deleted
    required source names its exact baseline preimage; the brief materializes
    that source for its reader. An unavailable source stays in the manifest as
    a diagnostic gap, never disappears into a declared-empty result.

    ``layer`` is the checklist layer (`review_body_fact.layer_for`). The
    protected-path inventory, the ``prompts/`` rule, the derived families and
    the declared twins all name files of the Ouroboros body; for the ``core``
    layer (``repo_dir`` is another repository) none of them applies, so no
    source is owed categorically and the manifest is empty — the reviewer reads
    what its own judgment needs. The policy version is unchanged: the body
    manifest is byte-identical to before.
    """
    if layer != "body":
        return []
    root = pathlib.Path(repo_dir)
    if touched_paths is None:
        touched_paths = staged_touched_paths(repo_dir, subject)
    touched: Dict[str, str] = {}
    for status, raw_path in touched_paths or ():
        path = normalize_repo_path(raw_path)
        if not path or path == ".":
            continue
        disposition = _STATUS_DISPOSITIONS.get(str(status or "M")[:1].upper(), "modified")
        # A path that is both added and deleted in one change (a rename pair on
        # the same path) keeps the deletion: its preimage is the weaker evidence.
        if touched.get(path) != "deleted":
            touched[path] = disposition

    # A touched prompt or protected path is owed to the reviewer in full.
    required: Dict[str, str] = {
        path: disposition for path, disposition in touched.items()
        if protected_path_category(path) or path.startswith(PROMPT_PATH_PREFIX)
    }
    for family in _DERIVED_FAMILIES:
        if not family & set(touched):
            continue
        for member in family:
            required.setdefault(member, "family")
    for path in touched:
        twins = _DECLARED_TWINS.get(path, ())
        if any(path.startswith(prefix) for prefix in FROZEN_CONTRACT_PATH_PREFIXES):
            twins = twins + (_WEB_CONTRACT_TWIN,)
        if not twins:
            continue
        # One contract, two languages: a change on either side owes the reviewer
        # BOTH sides in full, or it could check only the half it can see.
        required.setdefault(path, touched[path])
        for twin in twins:
            required.setdefault(twin, "twin")

    rows: List[Dict[str, Any]] = []
    tree = {"candidate_tree": str(staged_tree_sha)} if staged_tree_sha else {}
    for path in sorted(required):
        disposition = required[path]
        row: Dict[str, Any] = {"root": REQUIRED_SOURCE_ROOT, "path": path,
                               "disposition": disposition, **tree}
        if disposition == "deleted":
            base = str(getattr(subject, "m0_tree", "") or "HEAD")
            rows.append({**row, "coverage_basis": "preimage_unavailable",
                         "preimage": f"{base}:{path}",
                         "reason": "the candidate tree no longer carries this path; "
                                   "its preimage is not delivered as a readable source"})
            continue
        try:
            identity = source_text_identity((root / path).read_bytes())
        except (OSError, UnicodeDecodeError):
            rows.append({**row, "coverage_basis": "source_unavailable",
                         "reason": "the candidate tree does not carry readable text at this path"})
            continue
        rows.append({**row, **identity, "coverage_basis": "candidate_blob"})
    return rows


def required_sources_ref(rows: Sequence[Dict[str, Any]], *, staged_tree_sha: str = "") -> Dict[str, Any]:
    """The manifest's own identity, recorded with the review and its coverage."""
    payload = json.dumps(list(rows or ()), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "policy": SCOPE_REQUIRED_SOURCES_POLICY,
        "staged_tree_sha": str(staged_tree_sha or ""),
        "required_source_count": len(rows or ()),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def render_required_sources(rows: Sequence[Dict[str, Any]], *, layer: str = "body") -> str:
    """The reviewer's own view of the manifest: path, disposition and size.
    The empty manifest is worded by layer: the body owes nothing for THIS
    change; the core layer owes nothing categorically (no body inventory)."""
    if not rows and layer != "body":
        return (
            "REQUIRED SOURCES: none is owed categorically. This subject is not the "
            "Ouroboros body, so no protected-path inventory, frozen-contract twin or "
            "prompt rule applies to it; the staged diff is the complete change "
            "evidence. Read any file of the subject your judgment needs with your "
            "own tools."
        )
    if not rows:
        return (
            "REQUIRED SOURCES: none. This change touches no protected runtime path, "
            "frozen contract or prompt, so no source is owed in full. Read whatever "
            "the checklist needs with your own tools."
        )
    lines = []
    for row in rows:
        size = row.get("complete_chars")
        extent = f"{int(size):,} chars" if isinstance(size, int) else str(row.get("coverage_basis") or "")
        detail = ("; delivered inline in full, no second read needed"
                  if row.get("coverage_basis") == "delivered_inline" else "")
        if row.get("preimage_of"):
            detail += f"; preimage of {row['preimage_of']}"
        if row.get("root") in ("artifact_store", "session_root"):
            detail += f"; root={row['root']}"
        if row.get("reason"):
            detail += f"; {row['reason']}"
        lines.append(f"- {row.get('path')} ({row.get('disposition')}, {extent}){detail}")
    return (
        "REQUIRED SOURCES — inspect each source in full; a source delivered inline "
        "does not need a second tool read. This "
        "list is a MINIMUM, not a sufficiency claim: covering it does not make the "
        "review complete, and you may read anything else in the tree. A required "
        "source you do not read is recorded as a diagnostic coverage gap; coverage "
        "does not change the verdict, quorum or enforcement.\n" + "\n".join(lines)
    )


def render_touched_manifest(rows: Sequence[Dict[str, Any]]) -> str:
    """The touched-path manifest: what changed, how, and how large it is now."""
    if not rows:
        return ""
    lines = [f"- {row.get('path')} ({row.get('disposition')}, {row.get('extent')})" for row in rows]
    return (
        "TOUCHED PATHS (complete change evidence is the staged diff; the sizes are "
        "the candidate's, so you can choose what to open):\n" + "\n".join(lines)
    )


def touched_manifest(repo_dir: Any, touched_paths: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Every touched path with its disposition and candidate size."""
    root = pathlib.Path(repo_dir)
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for status, raw_path in touched_paths or ():
        path = normalize_repo_path(raw_path)
        if not path or path in seen:
            continue
        seen.add(path)
        disposition = _STATUS_DISPOSITIONS.get(str(status or "M")[:1].upper(), "modified")
        try:
            extent = f"{(root / path).stat().st_size:,} bytes"
        except OSError:
            extent = "not in the candidate tree"
        rows.append({"path": path, "disposition": disposition, "extent": extent})
    return sorted(rows, key=lambda row: row["path"])


def coverage_state(fact: Any) -> str:
    """One of ``complete``/``declared_empty``/``unobserved``/``incomplete``."""
    status = str((fact or {}).get("status") or "") if isinstance(fact, dict) else ""
    if status == "complete":
        return "declared_empty" if str(fact.get("reason") or "") == "declared_empty" else "complete"
    return "incomplete" if status == "incomplete" else "unobserved"


def uncovered_sources(fact: Any) -> List[str]:
    """The required sources this delivery did not cover, by path."""
    if not isinstance(fact, dict):
        return []
    return [str(row.get("path") or "")
            for row in (fact.get("sources") or [])
            if isinstance(row, dict) and row.get("status") != "complete"]
