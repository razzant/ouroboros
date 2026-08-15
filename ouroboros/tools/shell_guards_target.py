"""The TARGET-SPELLING arms of the shell guard (RWS v2 §3.1 step 3).

A guard that resolves a `pathlib.Path` is silently asserting "the execution target
is this Home filesystem". For an ssh placement that assertion is not just wrong,
it SUCCEEDS: a remote POSIX spelling read as a Home path resolves, stats, and
answers about a completely different tree. So the shell guard's path-resolving
arms cannot be parameterized by placement — they have to have a counterpart that
never touches a filesystem at all, and that counterpart lives here, beside the
Home arms rather than inside them, so neither can quietly become the other.

The reduction this module encodes is deliberate and narrow. Of everything
`run_shell_safety_check` protects, exactly two rules are about the operation and
therefore travel with it to the target: a write stays inside the active workspace,
and a protected artifact stays protected. The rest of the path-resolving arms —
the Ouroboros system repo, every data drive, the owner's credential files, Home's
own executor process ledger — protect HOME state that a remote process cannot
address, so on the target they have nothing to judge. Stating that here, once,
is the alternative to fabricating a Home `Path` for a remote root and hoping the
containment arithmetic means something.

Everything is pure `posixpath` over facts from the operation's ONE prepare.
"""

from __future__ import annotations

import dataclasses
import posixpath
from typing import Any, List, Optional

from ouroboros.shell_parse import shell_argv_with_path_tokens


@dataclasses.dataclass(frozen=True)
class NativeShellTarget:
    """The active workspace of a TARGET-NATIVE shell operation, as facts.

    Not a `pathlib.Path`, and that is the whole point: these are POSIX spellings
    on another host, and a Home `Path` would resolve symlinks and existence
    against the WRONG filesystem — succeeding while answering about nothing. The
    three fields are exactly what the write-containment arm needs and all three
    come from the operation's ONE prepare: the target's canonical workspace root,
    the target's canonical process cwd, and the workspace-relative protected
    spellings out of the hash-bound export policy Home issued for this operation.
    """

    root: str
    cwd: str = ""
    protected: tuple[str, ...] = ()

    def contains(self, target_path: str) -> bool:
        """Whether a target-canonical path lies inside the workspace root."""
        root = self.root.rstrip("/")
        return bool(root) and (target_path == root or target_path.startswith(root + "/"))

    def resolve(self, token: str) -> str:
        """One shell token as a target-canonical path (pure ``posixpath``)."""
        text = str(token or "").replace("\\", "/")
        base = self.cwd or self.root
        return posixpath.normpath(text if text.startswith("/") else posixpath.join(base, text))

    def relative(self, target_path: str) -> str:
        """The workspace-relative spelling of a contained target path."""
        root = self.root.rstrip("/")
        return target_path[len(root) + 1 :] if target_path != root else "."


def native_shell_target(prepared: Any) -> NativeShellTarget | None:
    """Build the guard's target facts from a prepared call, or ``None``.

    Keyed on the PLACEMENT, not on whether the tool is natively routed: under an
    ssh placement there is no Home path to hand the guard for ANY tool, routed or
    not, so a routed-only check would leave the unrouted ones raising
    `RemoteWorkspacePathError` out of the guard — the original defect, just
    narrower. ``None`` for every local/docker dispatch, which is what keeps the
    Home arms byte-identical: target facts exist exactly where a Home path does not.

    The cwd and the protected list come from the operation's prepare when it has
    one; without a bundle the workspace root is the only fact available, and the
    containment rule still holds against it.
    """

    if prepared is None or str(getattr(prepared, "placement", "")) != "ssh":
        return None
    executor = getattr(prepared, "executor", None)
    root = str(getattr(executor, "remote_root", "") or "")
    if not root:
        return None
    native = getattr(prepared, "native", None)
    document = getattr(getattr(prepared, "export_policy", None), "document", {}) or {}
    return NativeShellTarget(
        root=root,
        cwd=str((native.execution_args.get("cwd") if native is not None else "") or ""),
        protected=tuple(
            str(item).strip("/")
            for item in (document.get("protected_paths") or ())
            if str(item or "").strip()
        ),
    )


def native_shell_write_block(
    target: NativeShellTarget,
    raw_cmd: Any,
    *,
    writeish: bool,
    explicit_write_targets: List[str],
    executable_path_tokens: set,
) -> Optional[str]:
    """Write containment for a target-native shell command, in TARGET spellings.

    This is the native counterpart of `_workspace_write_shell_block`, and it is a
    separate arm rather than a parameterization of it because the Home arm's whole
    body is `pathlib` containment against Home roots. On the target those roots do
    not exist: the Ouroboros system repo, the data drives and the owner's
    credential files are Home state that a remote process cannot reach, so the arms
    that protect them are vacuous there, not skipped for convenience. What DOES
    still bind is the pair of rules that are about the operation itself — a write
    stays inside the active workspace, and a protected artifact stays protected —
    and both are decidable from the prepare facts without touching a filesystem.

    The target enforces confinement again on its own side (execd revalidates path
    confinement as execution integrity), so this is the Home POLICY answer, spoken
    before the operation leaves, not the only line of defence.
    """

    if not writeish:
        return None
    tokens = list(shell_argv_with_path_tokens(raw_cmd))
    tokens.extend(item for item in explicit_write_targets if item and item not in tokens)
    for token in tokens:
        text = str(token or "").strip()
        if (
            not text
            or text.startswith("-")
            or text == "/dev/null"
            or text in {"|", "&&", "||", ";", ">", ">>", "<", "<<", "&"}
            or (text in executable_path_tokens and text not in explicit_write_targets)
        ):
            continue
        if not text.startswith("/") and "/" not in text and text not in explicit_write_targets:
            continue
        resolved = target.resolve(text)
        if not target.contains(resolved):
            return (
                "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell commands may not target "
                f"paths outside the active workspace ({target.root})."
            )
        relative = target.relative(resolved)
        if any(relative == item or relative.startswith(item + "/") for item in target.protected):
            return (
                "⚠️ PROTECTED_ARTIFACT_SHELL_BLOCKED: shell command would write a "
                f"protected artifact of this task ({relative}). Protected artifacts are "
                "evidence: change the source and re-verify instead of rewriting them."
            )
    return None


__all__ = [
    "NativeShellTarget",
    "native_shell_target",
    "native_shell_write_block",
]
