"""The user_files confinement: secret-name policy and path resolution.

Every span is extracted VERBATIM from the parent's tip bytes by
scripts/v7next_transplant.py (D18/D33 module-handle split, proof-checked);
the parent re-exports every moved name, so historical imports and
monkeypatch targets keep working unchanged.
"""

from __future__ import annotations

import os
import pathlib

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only imports (inert at runtime)
    from typing import Any


def _tool_access():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from ouroboros import tool_access

    return tool_access


# The credential-shape dictionaries/regex live in ouroboros.credential_shapes
# (capinv-447): user_files READ authorization for the root principal is
# location-only, so this module must not import the name shapes at module
# level — only the MUTATION branch of user_files_path_block_reason reaches
# them, via a function-local import (pinned by the import-boundary test).


def _subagent_projects_read_hint(
    ctx: Any,
    resolved: pathlib.Path,
    hard_protected_roots: list[pathlib.Path],
) -> str:
    """A targeted refusal for a user_files path that actually lives inside the
    subagent-projects area: name root=subagent_projects with the exact relative
    path instead of steering the model at roots that cannot reach the target.
    Empty when the target is not there, the active profile cannot read that root,
    or the projects root is misconfigured to overlap a HARD drive (never steer a
    read at the control plane)."""
    try:
        profile_policy = _tool_access()._POLICY.get(_tool_access().active_tool_profile(ctx), {})
        if "read" not in profile_policy.get("subagent_projects", set()):
            return ""
        projects_root = _tool_access().resource_root_path(ctx, "subagent_projects")
        if any(
            _tool_access().path_is_relative_to(projects_root, hard) or _tool_access()._path_is_relative_to_casefold(projects_root, hard)
            for hard in hard_protected_roots
        ):
            return ""
        if not (
            _tool_access().path_is_relative_to(resolved, projects_root)
            or _tool_access()._path_is_relative_to_casefold(resolved, projects_root)
        ):
            return ""
        try:
            rel = str(resolved.relative_to(projects_root))
        except ValueError:
            rel = os.path.relpath(str(resolved), str(projects_root))
        rel = rel if rel not in ("", ".") else "."
        return (
            "this path is inside root=subagent_projects (the durable child-project "
            f"area); read it via root=subagent_projects, path={rel!r} "
            "(read/list/search only — no write/shell there by design: children "
            "write via write_surface=external_workspace, and the host "
            "checkpoint-commits dirty coop trees at root finalization)"
        )
    except Exception:
        return ""


def user_files_path_block_reason(
    ctx: Any,
    candidate: pathlib.Path,
    *,
    allow_protected_descendants: bool = False,
    operation: str = "",
) -> str:
    """Return a block reason when candidate is not an external user file.

    Location checks (outside-home, control-plane overlap) apply to every
    operation. The credential-shape name gate applies ONLY to non-read
    operations (capinv-447 / В23=A): the ROOT principal reads the owner's home
    in full — secret bytes are masked at egress, not refused at read time —
    while writes/edits/shell targets keep the shape deny (overwriting
    ~/.bashrc or ~/.ssh material is a persistence hazard, not a read).
    ``operation=""`` (unknown caller) keeps the shape deny, fail-closed for
    mutation surfaces. Children never hold a user_files grant (profile
    policy), so the read-side lift is root-only by construction.
    """

    resolved = pathlib.Path(candidate).expanduser().resolve(strict=False)
    home = _tool_access()._user_files_root()
    outside_home = not _tool_access().path_is_relative_to(resolved, home) and not _tool_access()._path_is_relative_to_casefold(resolved, home)
    # External-workspace tasks may reach host scratch outside home (/tmp, /build,
    # sibling checkouts). The runtime-overlap guard BELOW still runs on the full
    # path, so the Ouroboros repo/data drive stays protected even when home
    # confinement is lifted.
    if outside_home and not _tool_access().is_external_workspace(ctx):
        return f"path is outside user home {home}"

    # The Ouroboros runtime/control surface is the system repo PLUS every data
    # drive the task touches: the parent drive (ctx.drive_root) and any child /
    # budget drive carried in task_metadata. External-workspace mode lifts home
    # confinement, so these must be enumerated explicitly here — otherwise a
    # child-drive control path (e.g. <child_drive>/memory) would slip through.
    protected_values: list[Any] = [
        getattr(ctx, "drive_root", None),
        getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir", None),
    ]
    meta = getattr(ctx, "task_metadata", {})
    if isinstance(meta, dict):
        for key in ("drive_root", "child_drive_root", "headless_child_drive_root", "budget_drive_root"):
            if meta.get(key):
                protected_values.append(meta.get(key))
    protected_roots: list[pathlib.Path] = []
    hard_protected_roots: list[pathlib.Path] = []  # the data/repo/budget drives THEMSELVES
    for value in protected_values:
        try:
            root = pathlib.Path(value).resolve(strict=False)
        except (OSError, TypeError, ValueError):
            continue
        protected_roots.append(root)
        hard_protected_roots.append(root)
        parent = root.parent.resolve(strict=False)
        if root.name in {"repo", "data"} and _tool_access().path_is_relative_to(parent, home):
            # The workspace PARENT is a SOFT boundary (keeps user_files out of ~/Ouroboros at large);
            # it is deliberately NOT a hard root, so the Deliverables sibling under it stays allowed.
            protected_roots.append(parent)
    # The configured Deliverables container is an INTENDED user-output root, allowed past the
    # workspace-overlap guard — but ONLY when it is a genuine sibling: a misconfigured
    # OUROBOROS_DELIVERABLES_ROOT that overlaps or contains a HARD data/repo/budget drive must NOT
    # open a bypass. The outside-home, credential, and hidden-name checks still apply regardless.
    in_deliverables = False
    try:
        _deliverables = _tool_access()._deliverables_root()
        _deliverables_safe = not any(
            _tool_access().path_is_relative_to(_deliverables, pr) or _tool_access()._path_is_relative_to_casefold(_deliverables, pr)
            or _tool_access().path_is_relative_to(pr, _deliverables) or _tool_access()._path_is_relative_to_casefold(pr, _deliverables)
            for pr in hard_protected_roots
        )
        if _deliverables_safe and (
            _tool_access().path_is_relative_to(resolved, _deliverables) or _tool_access()._path_is_relative_to_casefold(resolved, _deliverables)
        ):
            in_deliverables = True
    except Exception:
        in_deliverables = False
    if not in_deliverables:
        for protected in protected_roots:
            overlaps_protected = _tool_access().path_is_relative_to(resolved, protected) or _tool_access()._path_is_relative_to_casefold(resolved, protected)
            contains_protected = _tool_access().path_is_relative_to(protected, resolved) or _tool_access()._path_is_relative_to_casefold(protected, resolved)
            if overlaps_protected or (
                not allow_protected_descendants and contains_protected
            ):
                # Name the root that ACTUALLY contains the target (the v6.54.3
                # shell_cwd_block_message lesson applied to this surface): the
                # subagent-projects area lives under the SOFT ~/Ouroboros parent,
                # so every coop-tree read used to get a message naming four roots
                # that cannot reach it while omitting the one that can. MESSAGE
                # ONLY — subagent_projects stays a read-only root (no user_files
                # write carve-out), and a target inside a HARD drive never takes
                # this branch.
                projects_hint = _subagent_projects_read_hint(ctx, resolved, hard_protected_roots)
                if projects_hint:
                    return projects_hint
                return (
                    "path overlaps the Ouroboros repo/runtime workspace; use "
                    "root=active_workspace, root=task_drive, root=artifact_store, "
                    "or root=skill_payload instead"
                )

    if operation in _tool_access()._READ_OPS:
        # Root READ authorization is location-only (capinv-447 / В23=A) — the
        # name shapes are never consulted here, so this branch must stay free
        # of any credential_shapes import (import-boundary test).
        return ""
    from ouroboros.credential_shapes import user_files_mutation_shape_reason

    return user_files_mutation_shape_reason(resolved, home)


class UserFilesPathBlockedError(ValueError):
    """Typed user_files confinement refusal (a POLICY denial, not an I/O failure).

    Subclasses ``ValueError`` so every existing generic handler keeps working;
    the read-surface wrappers (read_file/list_files/search_code) render it with
    the typed ``⚠️ USER_FILES_PATH_BLOCKED`` prefix so the outcome axis can
    partition it into ``execution.policy_denials`` (v6.57.0) instead of the
    generic ``error`` status that falsely degraded a shipped task to
    ``tool_failure`` (the submarine wave-3 incident)."""


def resolve_user_file_path(
    ctx: Any,
    path: str,
    *,
    allow_protected_descendants: bool = False,
    allow_outside_home: bool = False,
    operation: str = "",
) -> pathlib.Path:
    """Resolve a user_files path under the user's home and outside Ouroboros control-plane roots.

    Absolute paths OUTSIDE the user_files home (and the Deliverables container) are
    rejected EARLY with an actionable error instead of resolving to a foreign root
    and failing later with an opaque ``relative_to`` crash (v6.54.3 — the TB2.1
    ``'/app' is not in the subpath of '/root'`` class). ``allow_outside_home=True``
    (the ``query_code`` external-target caller) skips only this EARLY actionable
    check; ``user_files_path_block_reason`` below remains the outside-home
    AUTHORITY, and it permits outside-home only for external-workspace contexts —
    the mode the documented query_code contract (benchmark ``/app``) runs in.
    Neither flag expands authority: a non-external context could not reach
    outside-home before this check existed either."""

    raw_text = str(path or ".").strip() or "."
    try:
        raw = pathlib.Path(raw_text).expanduser()
    except Exception:
        # expanduser() raises RuntimeError for an unknown '~user'; leave it unexpanded —
        # the '~' branch below maps it into the jail home (raw is only used elsewhere for
        # absolute paths, where expanduser is a no-op anyway).
        raw = pathlib.Path(raw_text)
    home = _tool_access()._user_files_root()
    # is_absolute_path_text gives consistent cross-platform absolute detection
    # (drive-less "/x" roots and "C:\\x"/"\\\\unc" are all absolute) so Windows
    # does not silently treat a rooted path as home-relative.
    if _tool_access().is_absolute_path_text(raw_text):
        candidate = raw.resolve(strict=False)
        # External-workspace tasks legitimately reach host scratch outside home
        # (/tmp, /build, sibling checkouts) — for them the generic
        # user_files_path_block_reason below stays the authority, mirroring its
        # own is_external_workspace carve-out.
        if not allow_outside_home and not _tool_access().is_external_workspace(ctx):
            home_resolved = home.resolve(strict=False)
            # Case-insensitive-platform parity with the user_files_path_block_reason
            # authority: a differently-cased safe home path must not be rejected
            # early where the casefold-aware guard would accept it (review round 7).
            inside_home = _tool_access().path_is_relative_to(candidate, home_resolved) or _tool_access()._path_is_relative_to_casefold(
                candidate, home_resolved
            )
            inside_deliverables = False
            if not inside_home:
                try:
                    deliverables_resolved = _tool_access()._deliverables_root().resolve(strict=False)
                    inside_deliverables = _tool_access().path_is_relative_to(
                        candidate, deliverables_resolved
                    ) or _tool_access()._path_is_relative_to_casefold(candidate, deliverables_resolved)
                except (OSError, ValueError):
                    inside_deliverables = False
            if not inside_home and not inside_deliverables:
                raise UserFilesPathBlockedError(
                    "user_files path blocked: absolute path "
                    f"{raw_text!r} is outside the user_files home ({home_resolved}). "
                    "Use root='active_workspace' for workspace paths, or a "
                    "home-relative path (e.g. 'Desktop/file.txt') for user files."
                )
    elif raw_text.startswith("~"):
        # '~' / '~user' must expand to the CONFIGURED user_files home (the jail), NOT the
        # real OS home — otherwise OUROBOROS_USER_FILES_ROOT isolation is bypassed by a
        # '~/...' path. The jail has a single home, so '~user/sub' maps to '<home>/sub'.
        _after = raw_text[1:]
        if _after[:1] in ("/", "\\"):
            _rel = _after[1:]
        elif "/" in _after or "\\" in _after:
            _rel = _after.replace("\\", "/").split("/", 1)[1]
        else:
            _rel = ""  # bare '~' or '~user' -> the home directory itself
        candidate = (home / _tool_access().safe_relpath(_rel)).resolve(strict=False) if _rel else home.resolve(strict=False)
    else:
        # safe_relpath has already normalized any Windows backslash to a POSIX '/', so the
        # directory test below is separator-correct on every platform.
        rel = _tool_access().safe_relpath(raw_text)
        home_candidate = home / rel
        if "/" in rel.strip("/") or home_candidate.exists():
            # An explicit placement (a path WITH a directory — Desktop/..., Downloads/..., a subdir)
            # OR a bare name that ALREADY EXISTS under home (an existing file or directory such as
            # `Desktop`) is honored under the owner home exactly as given. This keeps read/list/search
            # of existing user files and directory names home-relative — only a genuinely NEW unnamed
            # output is containerized.
            candidate = home_candidate.resolve(strict=False)
        else:
            # A bare name with no directory that does NOT already exist under home is an unnamed NEW
            # deliverable: route it into the visible Deliverables container instead of cluttering the
            # home root (a later read of the same bare name resolves there too, staying consistent).
            candidate = (_tool_access()._deliverables_root() / rel).resolve(strict=False)
    reason = user_files_path_block_reason(
        ctx,
        candidate,
        allow_protected_descendants=allow_protected_descendants,
        operation=operation,
    )
    if reason:
        raise UserFilesPathBlockedError(f"user_files path blocked: {reason}")
    return candidate
