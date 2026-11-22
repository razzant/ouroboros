"""Process/shell guard helpers: the pre-execution shell check, self-change tripwires and light-mode repo snapshots.

The facade re-exports these definitions so existing imports and monkeypatch
targets retain the same bindings.
"""

from __future__ import annotations

import hashlib
import pathlib
from ouroboros import work_domain

from ouroboros.runtime_mode_policy import (
    mode_has_unrestricted_agency,
    protected_bible_history_delete_reason,
)

import ouroboros.tools.registry_guards as registry_guards
from ouroboros.tools.tool_result import (
    LegacyTextResultAdapter,
    ToolResult,
    _replace_tool_result,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only imports (inert at runtime)
    from typing import Any
    from typing import Dict
    from typing import Optional


def _registry():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time.
    """
    from ouroboros.tools import registry

    return registry


def _light_repo_snapshot(repo_dir: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Worktree tripwire for light-mode shell writes, not rollback machinery."""
    from ouroboros.git_observation import host_git_env
    try:
        repo = work_domain.settle(repo_dir)
        status = work_domain.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=str(repo), capture_output=True, text=True, timeout=5, env=host_git_env(),
        )
        if status.returncode != 0:
            return None
        unstaged = work_domain.run(
            ["git", "diff", "--binary", "--no-ext-diff"],
            cwd=str(repo), capture_output=True, text=True, timeout=10, env=host_git_env(),
        )
        staged = work_domain.run(
            ["git", "diff", "--cached", "--binary", "--no-ext-diff"],
            cwd=str(repo), capture_output=True, text=True, timeout=10, env=host_git_env(),
        )
        paths = _registry().parse_porcelain_paths(status.stdout)
        digest = hashlib.sha256()
        digest.update((status.stdout or "").encode("utf-8", errors="replace"))
        digest.update((unstaged.stdout if unstaged.returncode == 0 else "").encode("utf-8", errors="replace"))
        digest.update((staged.stdout if staged.returncode == 0 else "").encode("utf-8", errors="replace"))
        for rel in paths:
            try:
                target = work_domain.settle(repo / _registry().safe_relpath(rel))
                target.relative_to(repo)
                if work_domain.is_file(target) and rel in (status.stdout or ""):
                    stat = work_domain.lstat(target)
                    digest.update(f"{rel}\0{stat.st_size}\0{stat.st_mtime_ns}".encode("utf-8"))
            except Exception:
                continue
        return {"digest": digest.hexdigest(), "paths": paths}
    except Exception:
        return None


def _format_light_repo_write_note(before: Dict[str, Any], after: Dict[str, Any], tool_name: str = "run_command") -> str:
    """The light-lane tripwire NOTE, appended after the command payload (#447 В12)."""
    before_paths = set(before.get("paths") or [])
    after_paths = set(after.get("paths") or [])
    touched = sorted(after_paths | before_paths)
    listed = ", ".join(touched[:30]) if touched else "(status changed; no paths parsed)"
    if len(touched) > 30:
        listed += f", ... (+{len(touched) - 30} more)"
    return (
        "⚠️ LIGHT_MODE_REPO_CHANGED: runtime_mode=light observed "
        f"a mutation of the Ouroboros repository after {tool_name}. "
        "The execution result is preserved and no automatic rollback was attempted "
        "to avoid overwriting concurrent human edits. "
        f"Affected/dirty paths: {listed}. Inspect these changes against the task contract."
    )


def _git_ref_snapshot(repo_dir: pathlib.Path) -> Optional[Dict[str, str]]:
    try:
        repo = work_domain.settle(repo_dir)
        head = work_domain.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo), capture_output=True, text=True, timeout=5,
        )
        refs = work_domain.run(
            ["git", "show-ref", "--head", "--dereference"],
            cwd=str(repo), capture_output=True, text=True, timeout=5,
        )
        if head.returncode != 0 or refs.returncode not in (0, 1):
            return None
        digest = hashlib.sha256()
        digest.update((head.stdout or "").encode("utf-8", errors="replace"))
        digest.update((refs.stdout or "").encode("utf-8", errors="replace"))
        return {"head": (head.stdout or "").strip(), "digest": digest.hexdigest()}
    except Exception:
        return None




def _run_shell_safety_check(
    self, args: Dict[str, Any], runtime_mode: str, binding: Any = None,
) -> ToolResult | None:
    """Prepare one process target; semantic judgment belongs to the Supervisor.

    Script text, unknown interpreter effects and incidental path mentions do not
    prove a forbidden operation. Structured task/resource admission precedes this
    seam; Git keeps its actual repository target, and execution keeps its original
    arguments. No denied result is replayed as a second process attempt.
    """
    raw_cmd = args.get("cmd", args.get("command", ""))
    work_dir = registry_guards._resolved_shell_cwd(self, args, binding)
    if isinstance(work_dir, ToolResult):
        return work_dir
    self._ctx._protected_shell_notice_paths = []
    if mode_has_unrestricted_agency(runtime_mode):
        return None
    if direct_block := registry_guards._direct_shell_write_block(self, raw_cmd, work_dir, runtime_mode, binding):
        return direct_block
    if _registry().sudo_noninteractive_violation(raw_cmd):
        return ToolResult(
            status="blocked", code="SUDO_INTERACTIVE_BLOCKED",
            text="⚠️ SUDO_INTERACTIVE_BLOCKED: this process interface cannot answer a password prompt. Use sudo -n or another available privileged execution path.",
        )
    if reason := protected_bible_history_delete_reason(
        raw_cmd, cwd=work_dir, bible_path=_registry().system_repo_dir_for(self._ctx) / "BIBLE.md",
        identity_path=pathlib.Path(self._ctx.drive_root) / "memory" / "identity.md",
    ):
        return ToolResult(status="blocked", code="CORE_PROTECTION_BLOCKED", text=reason)
    from ouroboros.protected_artifacts import shell_block_reason

    items = _registry()._binding_items(binding)
    if reason := shell_block_reason(self._ctx, raw_cmd, cwd=str(work_dir),
                                    default_cwd=work_dir, binding=items[0] if items else None):
        return ToolResult(status="blocked", code="RESOURCE_POLICY_BLOCKED", text=reason)
    return registry_guards._shell_git_and_runtime_block(
        self, raw_cmd, args, "", bool(getattr(self._ctx, "is_workspace_mode", lambda: False)()),
        self._acting_self_worktree(), binding,
    )


def _owner_settings_snapshot() -> Optional[str]:
    """Text of the live settings.json, "" if absent, None if UNREADABLE.

    None disarms the tripwire below: the deleted restore recorded an OSError as
    "file absent" and could unlink the live settings.json — a baseline is either
    read successfully or not used at all."""
    from ouroboros import config as _cfg

    path = pathlib.Path(_cfg.SETTINGS_PATH)
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError:
        return None


def _run_shell_post_checks(
    self,
    result: str | ToolResult,
    *,
    light_repo_before: Optional[Dict[str, Any]],
    workspace_refs_before: Optional[Dict[str, str]],
    settings_before: Optional[str] = None,
    tool_name: str = "run_command",
) -> str | ToolResult:
    """Post-execution tripwires. They ANNOTATE, they never roll back.

    The owner-state snapshot/restore (OWNER_STATE_RESTORED) that used to run here
    was DELETED (issue #447, owner decision): it reverted ANY post-command
    difference without proving the command caused it, so a concurrent owner edit
    (Settings UI, grant click) was silently rolled back and blamed on the command;
    and its snapshot recorded an OSError while reading as "file absent", so the
    restore could UNLINK the live settings.json after a transient read error. The
    light-lane guard below refuses auto-rollback for exactly this reason. Pre-exec
    checks see only explicit write targets, so an obfuscated argument-level
    writer (``S=settings; cat > data/$S.json``) can pass them — the tripwire below
    makes that LOUD (a typed ``tripwire`` fact plus an appended note) without
    re-introducing the unsound rollback. Disclosed residual: the write itself is
    not reverted; the owner surface is the remedy. Every note TRAILS the payload
    (#447 В12/H1) so line 1 stays with the command's own outcome."""
    text = result.text if isinstance(result, ToolResult) else result
    typed = result if isinstance(result, ToolResult) else None

    def _typed_base() -> ToolResult:
        # A tripwire fact must survive even when the producer returned plain
        # text: since the notes TRAIL the payload (#447 H1), the marker no
        # longer owns line 1, so a text-only reader could not re-derive the
        # classification. Adapt once, through the ONE legacy adapter, so the
        # fact is carried typed instead of being lost in prose.
        nonlocal typed
        if typed is None:
            typed = LegacyTextResultAdapter.from_text(tool_name, text)
        return typed

    if settings_before is not None:
        settings_after = _owner_settings_snapshot()
        if settings_after is not None and settings_after != settings_before:
            text = (
                f"{text}\n\n⚠️ OWNER_SETTINGS_CHANGED: data/settings.json changed while "
                "this command ran. Owner settings change only through save_settings / "
                "the Settings UI; this write was NOT auto-reverted (a post-hoc rollback "
                "can clobber a concurrent legitimate owner edit) — the owner surface is "
                "the place to verify and restore."
            )
            typed = _replace_tool_result(
                _typed_base(),
                text=text,
                meta_updates={"tripwire": "owner_settings_changed"},
            )
    if light_repo_before is not None:
        light_repo_after = _light_repo_snapshot(_registry().system_repo_dir_for(self._ctx))
        if (
            light_repo_after is not None
            and light_repo_after.get("digest") != light_repo_before.get("digest")
        ):
            text = (
                f"{text}\n\n"
                + _format_light_repo_write_note(
                    light_repo_before, light_repo_after, tool_name=tool_name
                )
            )
            typed = _replace_tool_result(
                _typed_base(),
                text=text,
                meta_updates={"light_repo_changed": True},
            )
    if workspace_refs_before is not None:
        workspace_refs_after = _git_ref_snapshot(_registry().active_repo_dir_for(self._ctx))
        if (
            workspace_refs_after is not None
            and workspace_refs_after.get("digest") != workspace_refs_before.get("digest")
        ):
            text = (
                f"{text}\n\n"
                "⚠️ WORKSPACE_GIT_REF_CHANGED: run_command changed git HEAD or refs "
                "inside the external workspace. External workspace runs must leave "
                "changes as files/patch artifacts, not commits/tags/resets."
            )
            typed = _replace_tool_result(
                _typed_base(),
                text=text,
                code="WORKSPACE_GIT_REF_CHANGED",
                meta_updates={"workspace_git_refs_changed": True},
            )
    return typed if typed is not None else text
