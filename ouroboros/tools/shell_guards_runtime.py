"""The external-workspace guard over HOME's own runtime and owner credentials.

The counterpart of `shell_guards_target`: that module judges paths on the
execution target, this one judges whether a shell command reaches back into the
machine running Ouroboros. It exists because `read_file` and the `user_files` path
guard already deny these locations, and raw shell (`cat`, `python -c open(...)`)
would otherwise be the unenforced channel next door — the "one policy × N doors"
shape, exactly.

Two layers, because string matching alone is bypassable by relative paths and
symlinks, and both are DEFENSE-IN-DEPTH rather than the primary control: the
primary control is the gated `read_file` path plus the LLM safety supervisor
judging intent. The residual — a relative path hidden INSIDE an interpreter
one-liner string — is deliberately NOT chased with a regex over code strings
(an unwinnable arms race; BIBLE P5 / no-string-gate doctrine).

It takes a `ctx` rather than living on the registry so the dispatcher keeps only
the injection, and it stays Home-only by construction: every root it protects is a
Home path, so a target-native operation routes past it (see `shell_guards_target`
for what replaces it there).
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional, Sequence

from ouroboros.artifacts import task_artifact_dir_path, task_id_for_artifacts
from ouroboros.shell_parse import shell_argv_with_path_tokens
from ouroboros.tools.shell_guards import _command_mentions_protected_root


def external_runtime_protected_paths(
    ctx: Any, *, authorized_roots: Sequence[Any] = (),
) -> tuple[list, list, list, list]:
    """Ouroboros runtime roots that an EXTERNAL-workspace task must not touch via
    shell (system repo + EVERY data drive incl child/budget + owner credential
    locations) plus the task's own exempt task_drive/artifact_store roots. Returns
    (protected_texts, allowed_texts, protected_paths, allowed_paths): the *_texts
    feed the embedded-string boundary check; the *_paths feed token resolution
    (relative->cwd, ~->home, symlink canonicalization) so relative/symlink bypasses
    are closed. SSOT for the read + write guards."""
    meta = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    protected_values = [getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir", None),
                        getattr(ctx, "drive_root", None)]
    try:
        from ouroboros.config import DATA_DIR as _PARENT_DATA_DIR
        protected_values.append(_PARENT_DATA_DIR)
    except Exception:
        pass
    for _dk in ("drive_root", "child_drive_root", "headless_child_drive_root", "budget_drive_root"):
        if meta.get(_dk):
            protected_values.append(meta.get(_dk))
    # Owner/runtime credential locations, as ABSOLUTE paths. Blocking by
    # absolute containment (not a substring marker) means the OWNER's personal
    # secrets (~/.ssh/id_rsa, ~/.aws, ~/file1.txt) are off-limits while a
    # project-relative file merely NAMED like a credential (site/.ssh/config, a
    # project .env) stays the task's own — and a non-path token like
    # "os.environ" can never spuriously match.
    try:
        _home = pathlib.Path.home()
        for _rel in (".ssh", ".aws", ".gnupg", ".netrc", ".pgpass", ".config/gcloud",
                     ".docker/config.json", ".kube/config", ".npmrc", "file1.txt"):
            protected_values.append(_home / _rel)
    except Exception:
        pass
    def _text_forms(value: Any) -> list:
        # Both the as-given and the symlink-resolved form, so a command using
        # /var/... matches a root resolved to /private/var/... (macOS) and vice
        # versa. In production ($HOME paths) the two coincide.
        out = []
        for variant in (value, None):
            try:
                p = pathlib.Path(value)
                if variant is None:
                    p = p.resolve(strict=False)
                t = str(p).replace("\\", "/").lower().rstrip("/")
                if t and t not in out:
                    out.append(t)
            except Exception:
                continue
        return out

    def _resolved(value: Any):
        try:
            return pathlib.Path(value).resolve(strict=False)
        except Exception:
            return None

    protected_texts: list = []
    protected_paths: list = []
    for v in protected_values:
        if not v:
            continue
        for t in _text_forms(v):
            if t not in protected_texts:
                protected_texts.append(t)
        rp = _resolved(v)
        if rp is not None and rp not in protected_paths:
            protected_paths.append(rp)
    allowed_texts: list = []
    allowed_paths: list = []
    task_id = task_id_for_artifacts(ctx)
    for data_root in (getattr(ctx, "drive_root", None), meta.get("drive_root"), meta.get("budget_drive_root")):
        if not data_root:
            continue
        for rp_src in (pathlib.Path(data_root) / "task_drives" / task_id, task_artifact_dir_path(pathlib.Path(data_root), task_id, create=False)):
            for t in _text_forms(rp_src):
                if t not in allowed_texts:
                    allowed_texts.append(t)
            rp = _resolved(rp_src)
            if rp is not None and rp not in allowed_paths:
                allowed_paths.append(rp)
    # An explicitly selected system repo or exact skill payload is an authorized
    # process target. Keep every other runtime/credential root protected, but do not
    # re-block that exact selection merely because the task also has an external
    # workspace focus. The caller passes the roots ALREADY resolved from the binding
    # — this module never resolves a cwd itself, because the operation's cwd has one
    # resolution site and duplicating it here is the D1 regression class.
    for selected in authorized_roots or ():
        selected = pathlib.Path(selected)
        for t in _text_forms(selected):
            if t not in allowed_texts:
                allowed_texts.append(t)
        rp = _resolved(selected)
        if rp is not None and rp not in allowed_paths:
            allowed_paths.append(rp)
    return protected_texts, allowed_texts, protected_paths, allowed_paths

def external_shell_runtime_or_secret_block(
    ctx: Any, raw_cmd: Any, cmd_path_lower: str, args: Dict[str, Any],
    *, work_dir: pathlib.Path, authorized_roots: Sequence[Any] = (),
) -> Optional[str]:
    """External-workspace shell guard for READ and write commands alike: block any
    command that targets the Ouroboros runtime (system repo / any data drive) or an
    owner credential path. read_file/user_files already enforce this; raw shell
    (cat, python -c open(...), etc.) would otherwise bypass it. Two layers, because
    string matching alone is bypassable by relative paths and symlinks:
      (1) embedded-string boundary match of ABSOLUTE protected roots (catches a path
          literal inside e.g. python -c "open('/abs/data/settings.json')");
      (2) path-token RESOLUTION — every path-like arg is expanduser'd, joined to the
          command cwd when relative, and resolve()'d (canonicalizing symlinks + ..),
          then containment-checked. This closes a relative path passed as its own
          argv token (`cat ../../data/settings.json`) and a workspace-internal symlink
          to the data drive (round-2 review).
    Both layers are best-effort DEFENSE-IN-DEPTH, not the primary control: a relative
    path hidden INSIDE an interpreter one-liner string (e.g. node -e
    "readFileSync('../../data/settings.json')") is not a standalone token, so it is
    not extracted here — and that residual is deliberately NOT chased with a regex
    over code strings (an unwinnable arms race; BIBLE P5 / no-string-gate doctrine).
    The PRIMARY control is the gated read_file/user_files path, which fully resolves
    and containment-checks every read against the protected drives, plus the LLM
    safety supervisor judging intent on each shell call."""
    _BLOCK = (
        "⚠️ WORKSPACE_SHELL_BLOCKED: shell command targets the Ouroboros runtime "
        "(system repo / data drive) or an owner credential path. External-workspace "
        "tasks may not read or write those; use the gated read_file tool for any "
        "inspection you need. Run your command against the task's own surfaces "
        "instead: the active workspace root (e.g. /app) or scratch such as /tmp."
    )
    protected_texts, allowed_texts, protected_paths, allowed_paths = (
        external_runtime_protected_paths(ctx, authorized_roots=authorized_roots)
    )
    # (1) embedded-string boundary match (absolute roots only — no substring secret
    # markers, which would false-block the task's own project files / "os.environ").
    for pt in protected_texts:
        if _command_mentions_protected_root(cmd_path_lower, pt) and not any(
            _command_mentions_protected_root(cmd_path_lower, t) for t in allowed_texts
        ):
            return _BLOCK
    # (2) path-token resolution (relative -> cwd, ~ -> home, symlinks canonicalized).
    # The cwd arrives ALREADY resolved: it has exactly one resolution per operation
    # (D1), and re-deriving it here — or worse, joining the raw cwd label onto a root
    # — is the regression that resolution site exists to prevent.
    work_dir = pathlib.Path(work_dir)

    def _within(child: pathlib.Path, parent: pathlib.Path) -> bool:
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    for tok in shell_argv_with_path_tokens(raw_cmd):
        tok_text = str(tok or "").strip()
        if not tok_text or tok_text.startswith("-") or tok_text in {"|", "&&", "||", ";", ">", ">>", "<", "<<", "&"}:
            continue
        try:
            p = pathlib.Path(tok_text).expanduser()
            resolved = p.resolve(strict=False) if p.is_absolute() else (work_dir / p).resolve(strict=False)
        except Exception:
            continue
        if any(_within(resolved, ap) for ap in allowed_paths):
            continue
        if any(_within(resolved, pp) for pp in protected_paths):
            return _BLOCK
    return None


__all__ = [
    "external_runtime_protected_paths",
    "external_shell_runtime_or_secret_block",
]
