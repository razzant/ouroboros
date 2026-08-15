"""CLASS GATE 4 — no SILENT elision on an owner- or model-visible surface (BIBLE P1).

The recurring defect this closes is not any single truncation: it is that a bound was
applied to a COLLECTION on its way to a reader, and the reader was never told a bound
existed.  Found repeatedly during the remote-workspaces feature (the remote result's
artifact list, `changed_files[:40]` in the remote edit summary, the plan reviewer's own
omission-row list, the verification receipt's artifact rows).  Enumerating those four is
what "fixing the local problem" looks like; this gate instead makes the CLASS unable to
recur silently: every bounded slice in the feature's modules must either disclose its
own remainder inside the function that applies it, or appear in ONE registry below with
a stated reason.

BOUNDARY — design notes, so the limits of this gate are honest (see also
``test_source_gate_boundaries.py``, which asserts these limits are documented):

* The disclosure window is the ENCLOSING FUNCTION, not ±N lines.  A function that
  elides is the function that must disclose; a caller three frames up cannot be trusted
  to know a bound was applied.  Using a small line window produced false positives on
  code that discloses correctly a few lines later (``cli_connections`` prints the
  omitted count, ``gateway/connections`` sets ``truncated``).
* Only slices of COLLECTIONS are policed.  A ``str(x)[:200]`` field clamp, a hash or id
  prefix, and a bounded byte write are not elisions of items from a list a reader
  counts, and treating them as such would produce a registry nobody maintains.  The
  classifier is fail-closed: an expression it cannot prove string-shaped is policed, so
  a new pattern lands in a maintainer's lap rather than slipping through.
* A stale registry entry (the code moved or was fixed) is NOT a failure — the gate must
  fail on regressions, never on cleanups, or it teaches people to delete gates.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

# The remote-workspaces feature's own modules. A slice here is in scope; a slice in the
# rest of the tree is another gate's business (this one stays falsifiable by staying
# narrow — see the boundary test in tests/test_source_gate_boundaries.py).
FEATURE_MODULES: tuple[str, ...] = (
    "ouroboros/cli_connections.py",
    "ouroboros/cli_projects.py",
    "ouroboros/connection_store.py",
    "ouroboros/export_policy_contract.py",
    "ouroboros/gateway/connections.py",
    "ouroboros/gateway/projects.py",
    "ouroboros/gateway/tasks.py",
    "ouroboros/remote_export_policy.py",
    "ouroboros/remote_patch_bridge.py",
    "ouroboros/remote_plan_review.py",
    "ouroboros/remote_reconciliation.py",
    "ouroboros/remote_task_binding.py",
    "ouroboros/remote_task_files.py",
    "ouroboros/remote_transfer.py",
    "ouroboros/remote_worker_proxy.py",
    "ouroboros/remote_workspace.py",
    "ouroboros/tools/verify.py",
    "ouroboros/workspace_media_native.py",
    "ouroboros/workspace_payload_native.py",
    "ouroboros/workspace_query_native.py",
    "ouroboros/workspace_snapshot_native.py",
)

# Words that constitute telling the reader a bound was applied. Deliberately broad: the
# gate's job is to catch SILENCE, and a maintainer who wrote any of these was thinking
# about the reader. `count`/`total`/`len(` cover the "disclose as a sibling number" shape.
DISCLOSURE_TOKENS: tuple[str, ...] = (
    "OMISSION", "omission", "omitted", "truncat", "bounded", "bound at", "…",
    "excluded", "not listed", "undisclosed", "unprobed", "disclos", "remaining",
    "elided", "_count", "total", "len(", "more", "partial", "cap",
)

# Expression shapes that are provably NOT a collection of reader-counted items.
_STRINGISH_CALLS = frozenset({
    "str", "repr", "join", "decode", "encode", "strip", "lstrip", "rstrip", "lower",
    "upper", "replace", "format", "dumps", "hexdigest", "hex", "as_posix", "casefold",
    "expandtabs", "title", "removeprefix", "removesuffix",
})
# Names that hold a scalar identifier/text, not a list of items.
_STRINGISH_NAME_TOKENS = (
    "text", "reason", "name", "id", "label", "mime", "build", "summary", "platform",
    "digest", "hash", "sha", "token", "message", "msg", "line", "path", "root", "key",
    "stderr", "stdout", "out", "err", "detail", "note", "body", "content", "fingerprint",
    "generation", "hex", "version", "prefix", "suffix", "cmd_text", "argv_text",
)


def _stringish(node: ast.AST) -> bool:
    """True when the sliced expression is provably scalar text/bytes, not a collection.

    Fail-closed: anything unrecognised returns False and is therefore policed.
    """

    if isinstance(node, (ast.JoinedStr, ast.FormattedValue)):
        return True
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, bytes))
    if isinstance(node, ast.BinOp):  # "a" + b, b % args
        return _stringish(node.left) or _stringish(node.right)
    if isinstance(node, ast.Call):
        fn = node.func
        label = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if label in _STRINGISH_CALLS:
            return True
        if label == "bytes":
            return True
        return False
    if isinstance(node, ast.Attribute):
        return any(tok in node.attr.lower() for tok in _STRINGISH_NAME_TOKENS)
    if isinstance(node, ast.Name):
        low = node.id.lower()
        # `paths`/`files`/`rows` are plural collections even though "path"/"file" are
        # scalar tokens; the plural check runs first so it wins.
        if low.endswith("s") and not low.endswith("ss"):
            return False
        return any(tok in low for tok in _STRINGISH_NAME_TOKENS)
    if isinstance(node, ast.Subscript):
        return _stringish(node.value)
    return False


# ── The registry: every bounded collection slice that legitimately discloses nothing ──
# Keyed by (module, exact stripped source line). One reason per entry, in words, so the
# next reader can judge it. This is the ONLY sanctioned way to be silent, and adding a
# row is a reviewed act rather than an accident.
SILENT_BOUND_REASONS: dict[tuple[str, str], str] = {
    (
        "ouroboros/remote_browser_forward.py",
        'endpoints = [":".join(endpoints[:2]), ":".join(endpoints[2:])]',
    ): "regroups a fixed 4-part address; both halves are kept, nothing is dropped",
    (
        "ouroboros/gateway/connections.py",
        '"error": _sanitize_live_text(f"{type(exc).__name__}: {exc}")[:2000],',
    ): (
        "clamps one exception's TEXT for an error field; the classifier cannot prove a "
        "helper call returns a string, so it fails closed onto this registry"
    ),
    (
        "ouroboros/remote_patch_bridge.py",
        'f"git {\' \'.join(argv[:2])} failed in the mirror: "',
    ): (
        "names the git subcommand in an error message (\"git apply --check failed\"); the "
        "slice picks a label out of argv, it does not withhold items from a reader"
    ),
    (
        "ouroboros/remote_pending_operations.py",
        "not isinstance(raw.get(key), str) for key in required[:-1]",
    ): "iterates a fixed required-field tuple minus its last element; not reader output",
    (
        "ouroboros/tool_capabilities.py",
        'return tuple(".".join(parts[:index]) for index in range(1, len(parts)))',
    ): "generates every dotted module prefix; the slice is the generator, not a bound",
    (
        "ouroboros/remote_ssh_bootstrap.py",
        "pathlib.PurePosixPath(*parts[:index]).as_posix() for index in range(1, len(parts) + 1)",
    ): "generates every ancestor path prefix; enumeration, not elision",
    (
        "ouroboros/tools/verify.py",
        "for raw in list(artifact_paths or [])[:20]:",
    ): (
        "pre-existing host-side probe bound (predates this feature); the remote path's "
        "shortfall is disclosed as artifact_paths_unprobed_count at the call boundary"
    ),
    (
        "ouroboros/remote_export_policy.py",
        'f"policy excludes, so source-side filtering did not run: {leaked[:10]}"',
    ): (
        "the message states the full count before the sample; the sliced list is a "
        "sample inside a sentence that already names the total"
    ),
    (
        "ouroboros/remote_plan_review.py",
        'f"{blocked[:10]}. They exist on the target; the mirror does not contain them."',
    ): "same shape: len(blocked) is stated in the preceding line of the same message",
}


def _iter_bounded_slices():
    """Yield (module, lineno, source_line, enclosing_function_source) per policed site."""

    for rel in FEATURE_MODULES:
        path = REPO / rel
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        lines = src.splitlines()
        tree = ast.parse(src)
        # Map each node to its nearest enclosing function so disclosure is judged over
        # the function, which is the unit that owes the reader an explanation.
        enclosing: dict[int, ast.AST] = {}
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(func):
                enclosing.setdefault(id(sub), func)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            sl = node.slice
            if not isinstance(sl, ast.Slice):
                continue
            if sl.lower is not None or sl.upper is None or sl.step is not None:
                continue
            if _stringish(node.value):
                continue
            upper = sl.upper
            # A negative bound (`[:-1]`) drops exactly one known element; a bound of 0/1
            # is a "first or nothing" pick, not a list a reader would count.
            if isinstance(upper, ast.UnaryOp) and isinstance(upper.op, ast.USub):
                continue
            if isinstance(upper, ast.Constant) and isinstance(upper.value, int) and upper.value <= 1:
                continue
            func = enclosing.get(id(node))
            if func is None:
                scope = src
            else:
                scope = "\n".join(lines[func.lineno - 1: func.end_lineno])
            yield rel, node.lineno, lines[node.lineno - 1].strip(), scope


def test_no_silent_collection_elision_in_feature_modules():
    """Every bounded collection slice discloses its remainder, or is a registered reason.

    Failure names the module, the line and the code, because the fix is always local:
    say how many items the reader is not seeing.
    """

    violations: list[str] = []
    for rel, lineno, text, scope in _iter_bounded_slices():
        if any(token in scope for token in DISCLOSURE_TOKENS):
            continue
        if (rel, text) in SILENT_BOUND_REASONS:
            continue
        violations.append(
            f"{rel}:{lineno}: bounded slice with no disclosure in its own function\n"
            f"        {text}"
        )
    assert not violations, (
        "Silent elision on an owner/model-visible surface (BIBLE P1).\n"
        "Either disclose the remainder (a count, a marker row, an OMISSION NOTE) in the\n"
        "same function, or add a reasoned entry to SILENT_BOUND_REASONS in this file:\n\n"
        + "\n".join(violations)
    )


def test_registry_entries_are_reasoned_and_not_wholesale_stale():
    """The registry must carry real reasons, and must still describe real code.

    A stale entry alone is not a failure (removing an elision must never break the
    gate), but a registry that has ENTIRELY rotted means the gate is no longer reading
    the code it claims to police, which is the failure mode a green gate hides.
    """

    for (rel, text), reason in SILENT_BOUND_REASONS.items():
        assert len(reason) > 25, f"{rel}: {text!r} needs a real reason, got {reason!r}"
        assert text.strip() == text, f"{rel}: registry key must be the stripped line"

    live = 0
    for (rel, text), _reason in SILENT_BOUND_REASONS.items():
        path = REPO / rel
        if path.exists() and text in path.read_text(encoding="utf-8"):
            live += 1
    assert live >= max(1, len(SILENT_BOUND_REASONS) // 2), (
        f"only {live}/{len(SILENT_BOUND_REASONS)} registry entries still match real code — "
        "this gate is reading a codebase that moved on; re-derive it"
    )


def test_the_three_known_elisions_now_disclose_themselves():
    """The concrete instances that produced this class, pinned so they cannot regress.

    Written as source assertions rather than behaviour because that is what the class is
    about: the DISCLOSURE has to be adjacent to the bound. Behavioural coverage of the
    same four lives in the mutation checks (tests/test_seam_producers.py documents that
    split).
    """

    transfer = (REPO / "ouroboros/remote_transfer.py").read_text(encoding="utf-8")
    assert "undisclosed_artifacts" in transfer and "OMISSION NOTE" in transfer, (
        "remote result import must disclose artifacts dropped by the Home artifact bound"
    )
    assert "eligible[: max(0, _HOME_ARTIFACT_LIMIT" in transfer, (
        "the bound must be applied AFTER the eligibility filter, or the disclosed count "
        "undercounts (filtered rows must not consume the budget)"
    )

    plan = (REPO / "ouroboros/remote_plan_review.py").read_text(encoding="utf-8")
    assert "_OMISSION_ROW_LIMIT" in plan and "further omission(s) not listed" in plan, (
        "the omission list itself must disclose omissions it dropped"
    )

    verify = (REPO / "ouroboros/tools/verify.py").read_text(encoding="utf-8")
    assert "artifact_paths_unprobed_count" in verify, (
        "declared artifact paths beyond the probe bound are never probed; say so"
    )
    assert "artifact_lifecycle_undisclosed_count" in verify, (
        "receipt artifact rows are bounded; the receipt must carry the remainder count"
    )
