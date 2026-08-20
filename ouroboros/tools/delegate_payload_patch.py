"""The skill-payload patch pipeline: capture artifacts, guards, and the live apply.

The payload-specific terminal capture over the skill-loader inventory, the
reserved-path and symlink-escape guards it shares with the apply, the one
post-apply finalizer, and ``integrate_payload_patch`` — the R1 item 3 seam that
applies or rejects ONE payload run's captured patch into the LIVE non-Git
payload. Extracted from ``ouroboros/tools/delegate_integration.py`` at its size
gate (v7 DEL1 split); ``tools.delegate_integration`` re-exports every name
(same objects), so sibling code, the tests and monkeypatch targets keep
addressing them on THAT surface.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, Tuple

from ouroboros import delegate_custody as custody
from ouroboros.delegate_custody import RunCustody as _RunCustody
from ouroboros.tools.registry import ToolContext

# The parent logger name is pinned on purpose: records moved with their code
# keep the exact `%(name)s` every handler and reader saw before the split.
log = logging.getLogger("ouroboros.tools.delegate_integration")


def _di():
    """The parent integration-seam module, read at call time.

    The integration members stay monkeypatch-addressable at their historical
    ``ouroboros.tools.delegate_integration`` bindings (tests rebind them
    there), so this leaf resolves every cross-reference through the module at
    each call instead of freezing whatever object a from-import saw at import
    time.
    """
    from ouroboros.tools import delegate_integration

    return delegate_integration


def _reserved_payload_rel_path(rel: str) -> bool:
    """Name-rule half of reserved-path detection (R1 item 3): lifecycle/control
    filenames and directories from the frozen skill-payload policy, plus an
    explicit ``.git`` rule (a live payload must never receive VCS internals).
    The live-target half (`is_skill_control_plane_path` / owner-state aliases)
    runs at apply time against the real destination paths."""
    from ouroboros.contracts.skill_payload_policy import (
        SKILL_PAYLOAD_CONTROL_DIRNAMES,
        SKILL_PAYLOAD_CONTROL_FILENAMES,
    )

    parts = [part.lower() for part in pathlib.PurePosixPath(str(rel or "")).parts]
    if not parts:
        return False
    if ".git" in parts:
        return True
    if any(part in SKILL_PAYLOAD_CONTROL_DIRNAMES for part in parts):
        return True
    return parts[-1] in SKILL_PAYLOAD_CONTROL_FILENAMES


def _snapshot_head_textual(exec_root: pathlib.Path) -> str:
    """The snapshot's HEAD commit read TEXTUALLY (no git, no child config).

    Informational input to the head_moved disclosure only. Fails soft to ""
    on anything unusual (symlinked HEAD/ref, packed refs, unreadable files) —
    the capture itself never depends on the child-writable HEAD.
    """
    git_dir = exec_root / ".git"
    head = git_dir / "HEAD"
    try:
        if head.is_symlink():
            return ""
        text = head.read_text(encoding="utf-8", errors="replace").strip()
        if not text.startswith("ref: "):
            return text
        ref = _di()._resolved(git_dir / text[5:].strip())
        if ref is None or git_dir.resolve() not in ref.parents or ref.is_symlink():
            return ""
        return ref.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _write_payload_patch_artifacts(
    exec_root: pathlib.Path, cap_dir: pathlib.Path, entry: _RunCustody,
) -> Dict[str, Any]:
    """The payload-specific terminal capture (R1 items 5/6), same artifact contract.

    Trusts NOTHING under the child-writable snapshot's ``.git`` (Sol P1):
    baseline identity comes from the host-owned snapshot registry, git runs
    against a parent-owned control GIT_DIR/temp index seeded from that commit
    (``payload_capture_git_env``), and exactly the final loader inventory plus
    explicit baseline deletions is staged there from RAW bytes
    (``stage_raw_payload_inventory``: no .gitattributes filters; regular modes
    pinned to baseline/100644; symlinks as 120000 raw targets). A non-empty
    patch whose result loader hash equals the baseline is a typed
    ``unreviewable_metadata_change`` refusal. ``workspace.patch`` is
    ``git diff --binary`` against the recorded baseline. A candidate
    add/modify of genuinely non-UTF-8 content is a typed FAILURE (permanent
    text-only contract); reserved lifecycle/control paths never block capture —
    reported as ``blocked_reserved_paths`` and refused whole at apply, with the
    candidate always preserved for the parent's decision.
    """
    import hashlib
    import subprocess

    from ouroboros.headless import (
        ARTIFACT_STATUS_FAILED,
        ARTIFACT_STATUS_READY_NO_CHANGES,
        ARTIFACT_STATUS_READY_WITH_CHANGES,
    )
    from ouroboros.skill_loader import _iter_payload_files
    from ouroboros.subagent_worktrees import (
        find_execution_snapshot,
        payload_capture_git_env,
        payload_git_metadata_refusal,
        stage_raw_payload_inventory,
    )
    from ouroboros.utils import atomic_write_json, utc_now_iso

    diff_isolation = ("--no-ext-diff", "--no-textconv", "--no-renames")

    def _manifest(status: str, **extra: Any) -> Dict[str, Any]:
        payload = {
            "schema_version": 1, "created_at": utc_now_iso(), "status": status,
            "capture_kind": "skill_payload",
            "baseline_payload_hash": str((entry.resource_ref or {}).get("payload_hash") or ""),
            **extra,
        }
        atomic_write_json(cap_dir / "workspace_patch.json", payload, trailing_newline=True)
        return payload

    # NOTHING under the child-writable snapshot's .git is trusted (Sol P1):
    # metadata replaced by symlinks is refused before any git operation, the
    # baseline identity comes from the HOST-owned snapshot registry, and every
    # parent git command runs against a parent-owned control GIT_DIR + fresh
    # temp index (child .git/index and .git/config are never read or written —
    # a child-forged index-only blob does not exist for this environment).
    # Diff commands additionally pin --no-ext-diff/--no-textconv/--no-renames.
    untrusted = payload_git_metadata_refusal(exec_root)
    if untrusted:
        return _manifest(ARTIFACT_STATUS_FAILED,
                         note=f"snapshot git metadata untrusted: {untrusted}")
    registered = find_execution_snapshot(entry.snapshot_id or "")
    if not registered or not registered.get("standalone"):
        return _manifest(ARTIFACT_STATUS_FAILED,
                         note="host snapshot registry carries no standalone record for "
                              "this snapshot; the baseline identity cannot be trusted")
    baseline = str(registered.get("baseline_sha") or "")
    recorded_tree = str(registered.get("baseline_tree") or "")
    if not baseline or not recorded_tree:
        return _manifest(ARTIFACT_STATUS_FAILED,
                         note="host snapshot registry record carries no baseline "
                              "commit/tree identity")
    if entry.baseline_sha and entry.baseline_sha != baseline:
        return _manifest(ARTIFACT_STATUS_FAILED,
                         note="custody row and host snapshot registry disagree on the "
                              "baseline commit; refusing to capture over an ambiguous "
                              "baseline")
    resolved_root = exec_root.resolve()
    # Final loader-visible inventory; raises SkillPayloadUnreadable on
    # credential-shaped files (the existing refusal — caller discloses it typed).
    final_rel = sorted(
        path.relative_to(resolved_root).as_posix()
        for path in _iter_payload_files(resolved_root)
    )
    with payload_capture_git_env(exec_root) as git_env:

        def _git(*args: str, input_bytes: bytes = b"") -> subprocess.CompletedProcess:
            return subprocess.run(["git", *args], cwd=str(exec_root), env=git_env,
                                  capture_output=True, input=input_bytes or None)

        # The recorded commit must be PRESENT and carry the host-recorded tree
        # identity — a child-substituted object cannot keep the content address.
        shown = _git("rev-parse", f"{baseline}^{{tree}}")
        seen_tree = (shown.stdout or b"").decode("utf-8", errors="replace").strip()
        if shown.returncode != 0 or seen_tree != recorded_tree:
            detail = (shown.stderr or shown.stdout or b"").decode("utf-8", errors="replace")
            return _manifest(ARTIFACT_STATUS_FAILED,
                             note="recorded baseline commit is absent or does not match "
                                  f"the host-recorded tree identity ({detail.strip()[:200]})")
        # Seed the FRESH parent-owned index from the immutable recorded baseline.
        seeded = _git("read-tree", baseline)
        if seeded.returncode != 0:
            detail = (seeded.stderr or seeded.stdout or b"").decode("utf-8", errors="replace")
            return _manifest(ARTIFACT_STATUS_FAILED,
                             note=f"baseline unreadable: {detail.strip()[:300]}")
        listed = _git("ls-tree", "-r", "-z", baseline)
        if listed.returncode != 0:
            detail = (listed.stderr or listed.stdout or b"").decode("utf-8", errors="replace")
            return _manifest(ARTIFACT_STATUS_FAILED,
                             note=f"baseline unreadable: {detail.strip()[:300]}")
        baseline_modes: Dict[str, str] = {}
        for chunk in (listed.stdout or b"").split(b"\0"):
            if not chunk:
                continue
            meta, _sep, name = chunk.partition(b"\t")
            baseline_modes[name.decode("utf-8", errors="surrogateescape")] = (
                meta.split()[0].decode("ascii", errors="replace"))

        # Only the FINAL loader-visible inventory rides as content, staged from
        # RAW bytes (Sol P1 modes/filters: a .gitattributes eol/clean filter must
        # not forge staged content, and regular-file modes are pinned to
        # baseline/100644 so an executable-bit flip cannot ride). A baseline path
        # absent from the final inventory is staged as a DELETION even when
        # something still sits on disk there (e.g. a file replaced by an escaping
        # symlink, which the inventory drops).
        dropped = sorted(set(baseline_modes) - set(final_rel))
        try:
            normalized_modes = stage_raw_payload_inventory(
                exec_root, final_rel, git_env, baseline_modes=baseline_modes)
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "stderr", b"") or b""
            detail = detail.decode("utf-8", errors="replace") if isinstance(detail, bytes) else str(detail)
            return _manifest(ARTIFACT_STATUS_FAILED,
                             note=f"staging failed: {type(exc).__name__}: "
                                  f"{(detail or str(exc)).strip()[:300]}")
        if dropped:
            removed = _git("update-index", "-z", "--force-remove", "--stdin",
                           input_bytes=b"\0".join(
                               p.encode("utf-8", errors="surrogateescape")
                               for p in dropped) + b"\0")
            if removed.returncode != 0:
                detail = (removed.stderr or removed.stdout or b"").decode("utf-8", errors="replace")
                return _manifest(ARTIFACT_STATUS_FAILED,
                                 note=f"staging failed: {detail.strip()[:300]}")
        named = _git("diff", *diff_isolation, "--cached", "--name-only", "-z", baseline)
        if named.returncode != 0:
            detail = (named.stderr or named.stdout or b"").decode("utf-8", errors="replace")
            return _manifest(ARTIFACT_STATUS_FAILED, note=f"diff failed: {detail.strip()[:300]}")
        changed = sorted(
            chunk.decode("utf-8", errors="surrogateescape")
            for chunk in (named.stdout or b"").split(b"\0") if chunk
        )
        result_hash = _di().payload_content_hash(resolved_root)
        # Informational only (the head_moved disclosure): read TEXTUALLY — a git
        # invocation against the child GIT_DIR would consult child config again.
        current_head = _snapshot_head_textual(resolved_root)
        if not changed:
            if normalized_modes:
                # The run's ONLY change is an executable-bit flip the review
                # content hash cannot see: refused typed, nothing rides.
                return _manifest(
                    ARTIFACT_STATUS_FAILED,
                    refusal_kind="unreviewable_metadata_change",
                    normalized_mode_paths=normalized_modes,
                    note="unreviewable_metadata_change: the run's only change is "
                         "an executable-bit flip "
                         f"({', '.join(normalized_modes[:5])}), invisible to the "
                         "payload review content hash; nothing rides. The "
                         "snapshot is preserved for inspection.")
            return _manifest(ARTIFACT_STATUS_READY_NO_CHANGES, sha256="", diffstat="",
                             tracked_changed=[], untracked_included=[],
                             blocked_reserved_paths=[], result_content_hash=result_hash,
                             current_head=current_head)
        non_utf8 = []
        for rel in changed:
            if rel in dropped:
                continue  # rides as a deletion; on-disk leftovers are not content
            candidate = resolved_root / rel
            if candidate.is_symlink() or not candidate.is_file():
                continue
            try:
                candidate.read_bytes().decode("utf-8", "strict")
            except (OSError, UnicodeDecodeError):
                non_utf8.append(rel)
        if non_utf8:
            return _manifest(
                ARTIFACT_STATUS_FAILED, non_utf8_paths=non_utf8,
                note="the candidate adds/modifies non-UTF-8 payload content, which the "
                     "permanent text-only skill contract refuses "
                     f"({', '.join(non_utf8[:5])}); the snapshot is preserved")
        diff = _git("diff", *diff_isolation, "--cached", "--binary", baseline)
        if diff.returncode != 0:
            detail = (diff.stderr or diff.stdout or b"").decode("utf-8", errors="replace")
            return _manifest(ARTIFACT_STATUS_FAILED, note=f"patch emit failed: {detail.strip()[:300]}")
        patch_bytes = diff.stdout or b""
        (cap_dir / "workspace.patch").write_bytes(patch_bytes)
        baseline_hash = str((entry.resource_ref or {}).get("payload_hash") or "")
        if baseline_hash and result_hash == baseline_hash:
            # Non-empty patch, result hash EQUAL to baseline: the change is
            # invisible to the review hash (symlink topology / metadata) — a
            # fresh verdict could not distinguish result from reviewed baseline.
            return _manifest(
                ARTIFACT_STATUS_FAILED,
                refusal_kind="unreviewable_metadata_change",
                tracked_changed=changed,
                note="unreviewable_metadata_change: the candidate patch is "
                     "non-empty but the result payload content hash equals the "
                     "baseline (the change is invisible to the review hash — "
                     "e.g. symlink topology or file metadata). The snapshot and "
                     "the candidate patch are preserved for inspection.")
        stat = _git("diff", *diff_isolation, "--cached", "--shortstat", baseline)
        return _manifest(
            ARTIFACT_STATUS_READY_WITH_CHANGES,
            sha256=hashlib.sha256(patch_bytes).hexdigest(),
            patch_size=len(patch_bytes),
            diffstat=(stat.stdout or b"").decode("utf-8", errors="replace").strip(),
            tracked_changed=changed,
            untracked_included=[],
            blocked_reserved_paths=[p for p in changed if _reserved_payload_rel_path(p)],
            result_content_hash=result_hash,
            current_head=current_head,
            normalized_mode_paths=normalized_modes,
        )


def _payload_reserved_paths(
    ordered: list, target: pathlib.Path, state_root: pathlib.Path,
) -> Tuple[list, str]:
    """Reserved/escaping destinations among the patch's touched paths (R1 item 3).

    Name rules plus the LIVE-target predicates of the frozen skill-payload
    policy (control-plane paths and owner-state hardlink aliases), judged
    against the real destination each path lands on. Returns
    ``(reserved_paths, escape_refusal)``.
    """
    from ouroboros.contracts.skill_payload_policy import (
        is_skill_control_plane_path,
        is_skill_owner_state_alias,
    )

    resolved_target = target.resolve()
    reserved = []
    for rel in ordered:
        if pathlib.PurePosixPath(rel).is_absolute() or _reserved_payload_rel_path(rel):
            reserved.append(rel)
            continue
        live = _di()._resolved(target / rel)
        if live is None:
            return [], f"touched path {rel!r} cannot be resolved"
        try:
            live.relative_to(resolved_target)
        except ValueError:
            return [], f"touched path {rel!r} escapes the payload root"
        if is_skill_control_plane_path(live, state_root) or is_skill_owner_state_alias(live, state_root):
            reserved.append(rel)
    return sorted(set(reserved)), ""


def _candidate_symlink_escapes(
    patch_path: pathlib.Path, target: pathlib.Path,
) -> Tuple[list, str]:
    """Symlink-introducing patch entries whose target would escape the LIVE payload.

    Containment is judged on the CANDIDATE, not the live preimage (gate fix 1):
    a mode-120000 hunk's link target is resolved as it would land under the
    live payload root; an escape is refused like a ``../`` path escape.
    ``--no-renames`` keeps the parse total; an unparseable symlink entry fails
    CLOSED. Returns ``(escaping_rel_paths, parse_refusal)``.
    """
    import os

    try:
        raw = patch_path.read_bytes()
    except OSError as exc:
        return [], f"candidate patch unreadable: {exc}"
    resolved_target = target.resolve()
    escapes: list = []
    path, is_link, link_target, in_hunk = "", False, None, False

    def _flush() -> str:
        nonlocal path, is_link, link_target, in_hunk
        if is_link:
            if not path or link_target is None:
                return "a symlink-introducing patch entry could not be parsed"
            dest = resolved_target / pathlib.PurePosixPath(path)
            cand = (pathlib.Path(link_target) if os.path.isabs(link_target)
                    else dest.parent / link_target)
            landed = _di()._resolved(cand)
            if landed is None or not (
                    landed == resolved_target or resolved_target in landed.parents):
                escapes.append(path)
        path, is_link, link_target, in_hunk = "", False, None, False
        return ""

    for line in raw.split(b"\n"):
        if line.startswith(b"diff --git "):
            err = _flush()
            if err:
                return [], err
        elif line in (b"new file mode 120000", b"new mode 120000"):
            is_link = True
        elif line.startswith(b"+++ "):
            name = line[4:].split(b"\t")[0].decode("utf-8", errors="surrogateescape")
            if name.startswith("b/"):
                path = name[2:]
            elif name.startswith('"'):
                # git-quoted (control/non-ASCII bytes in the name): fail closed
                # rather than guess the octal unescaping for a symlink entry.
                path = ""
        elif line.startswith(b"@@"):
            in_hunk = True
        elif is_link and in_hunk and line.startswith(b"+") and link_target is None:
            link_target = line[1:].decode("utf-8", errors="surrogateescape")
    err = _flush()
    if err:
        return [], err
    return sorted(set(escapes)), ""


def _finalize_payload_apply(
    ctx: ToolContext, *, rid: str, reason: str, target: pathlib.Path,
    touched: list, ordered: list, manifest: Dict[str, Any],
    state_root: pathlib.Path, skill_name: str, dispose: Any, already: bool,
) -> str:
    """The ONE post-apply finalizer (gate fix 4): advisory invalidation →
    extension reconcile → verdict artifact → disposal, in this order, for BOTH
    the fresh-apply and the already-applied/idempotent outcomes — an earlier
    attempt may have died after mutating but before invalidation/reconcile.
    A reconcile queue-write failure degrades the receipt honestly instead of
    claiming the extension was reconciled off.
    """
    from ouroboros.tools.subagent_integration import (
        _unwritten_disposition_text,
        _write_verdict,
    )

    try:
        from ouroboros.review_state import invalidate_advisory_after_mutation

        invalidate_advisory_after_mutation(
            pathlib.Path(getattr(ctx, "drive_root", ".")), mutation_root=target,
            changed_paths=ordered, source_tool="integrate_delegated_patch")
    except Exception:
        pass
    reconcile_err = ""
    try:
        # A stale ENABLED extension must stop being live until re-review (R1
        # item 10); enablement/grants state itself is untouched by delegation.
        from ouroboros.extension_reconcile_queue import request_extension_reconcile

        request_extension_reconcile(state_root, skill_name,
                                    reason="delegated_payload_apply", source="worker")
    except Exception as exc:
        log.warning("extension reconcile request failed after payload apply %s",
                    rid, exc_info=True)
        reconcile_err = f"{type(exc).__name__}: {exc}"
    verdict_path = _write_verdict(
        ctx, f"run_{rid}", outcome="applied",
        reason=reason or ("already applied" if already else ""),
        files=touched, manifest=manifest, applied=True, conflicts=[], protected=[],
        target=str(target))
    recorded, note = dispose("applied", True)
    if not recorded:
        return _unwritten_disposition_text(rid, str(target), "applied", True,
                                           payload=True)
    staleness = (
        "The payload CONTENT CHANGED, so any prior skill review is now STALE for "
        "the new content hash: run skill_preflight and skill_review before "
        "relying on this skill. Enablement and grants were not changed by this "
        "apply; "
        + ("an extension reconcile was QUEUED (a marker the server processes "
           "asynchronously — a stale enabled extension stops being live when "
           "that reconcile runs, not at this receipt)."
           if not reconcile_err else
           f"WARNING: the extension reconcile could NOT be queued ({reconcile_err})"
           " — the review staleness above still holds, but a stale enabled "
           "extension may remain live until the next restart or a manual "
           "reconcile."))
    if already:
        return (f"OK: the live payload ALREADY carries run {rid}'s captured result "
                f"(content hash match) — recorded as applied, nothing re-applied. "
                f"Verdict: {verdict_path or '(unwritten)'}.\n{staleness}{note}")
    return (
        f"✅ Integrated delegated run {rid}'s patch into the live skill payload "
        f"{target} ({len(ordered)} file(s)). No .git, index, or staging was created "
        f"there.\n{str(manifest.get('diffstat') or '').strip()}\n"
        f"Verdict: {verdict_path or '(unwritten)'}. The standalone snapshot is "
        f"released.\n{staleness}{note}")


def integrate_payload_patch(
    ctx: ToolContext, *, drive: pathlib.Path, entry: _RunCustody, rid: str,
    decision: str, reason: str, cap_dir: pathlib.Path,
    manifest: Dict[str, Any], patch_path: pathlib.Path,
) -> str:
    """Apply or reject ONE payload run's captured patch into the LIVE payload (R1 item 3).

    The payload counterpart of the Git apply branch. Deliberate differences:
    the target is the live NON-Git payload (no active-root comparison, no
    staging); target authority is a FRESH exact binding equal to the recorded
    target; drift is the whole-payload loader content-hash CAS (already-applied
    disposes idempotently); reserved destinations refuse the WHOLE apply with
    the candidate preserved; ``git apply`` runs with the live payload as cwd
    (index-free, probed); after a real apply the LIVE loader hash must equal
    the recorded result hash or the run fails typed with its apply intent left
    PENDING (ambiguous machinery); ANY mutating apply outcome — success or
    post-apply hash mismatch — QUEUES the existing extension reconcile request
    (receipt says queued, never reconciled).
    """
    import subprocess

    from ouroboros.headless import (
        ARTIFACT_STATUS_READY_NO_CHANGES,
        ARTIFACT_STATUS_READY_WITH_CHANGES,
    )
    from ouroboros.tools.subagent_integration import (
        _READY_CAPTURE_STATUSES,
        _capture_failed_refusal,
        _dispose_delegated,
        _patch_touched_paths,
        _sha256_file,
        _unwritten_disposition_text,
        _write_verdict,
    )

    status = str(manifest.get("status") or "")
    touched = [str(p) for p in (manifest.get("tracked_changed") or [])]
    snapshot_key = entry.snapshot_id or entry.run_id

    def _dispose(disposition: str, cleanup: bool) -> Tuple[bool, str]:
        return _dispose_delegated(drive, entry, snapshot_key, reason, disposition, cleanup)

    if decision == "reject":
        # A reject RELEASES the snapshot (the child's only copy): ready-only. It
        # deliberately needs NO fresh target authority — the owner can release
        # retained material even after the skill was deleted or revoked.
        if status not in _READY_CAPTURE_STATUSES:
            return _capture_failed_refusal(
                rid, status, "a reject would release the snapshot over it")
        verdict_path = _write_verdict(
            ctx, f"run_{rid}", outcome="rejected", reason=reason, files=touched,
            manifest=manifest, applied=False, conflicts=[], protected=[],
            target=str(entry.target_root))
        recorded, note = _dispose("rejected", cleanup=True)
        if not recorded:
            return _unwritten_disposition_text(rid, str(entry.target_root), "rejected", False)
        return (
            f"🚫 Rejected delegated run {rid}'s captured payload patch ({len(touched)} "
            f"file(s) not applied); the live skill payload is unchanged and the "
            f"standalone snapshot is released. Verdict: {verdict_path or '(unwritten)'}. "
            f"Reason: {reason or '(none)'}.{note}")

    if status == ARTIFACT_STATUS_READY_NO_CHANGES:
        recorded, note = _dispose("applied", cleanup=True)
        if not recorded:
            return _unwritten_disposition_text(rid, str(entry.target_root), "applied", False)
        return (f"OK: delegated run {rid} changed NOTHING in its payload snapshot; "
                f"there is no patch to apply and the snapshot is released.{note}")
    if status != ARTIFACT_STATUS_READY_WITH_CHANGES:
        return (
            f"⚠️ INTEGRATE_DELEGATED_NO_CAPTURE: run {rid}'s payload capture status is "
            f"{status or 'missing'!r} — no applicable patch "
            f"({str(manifest.get('note') or '')[:300]}). A failed capture keeps the "
            "snapshot for direct inspection; fix the cause, then retry.")
    if not patch_path.exists():
        return f"⚠️ INTEGRATE_PATCH_MISSING: captured patch not found at {patch_path}."
    expected_digest = str(manifest.get("sha256") or "")
    if expected_digest and _sha256_file(patch_path) != expected_digest:
        return (f"⚠️ INTEGRATE_PATCH_CORRUPT: sha256 mismatch for run {rid}; "
                "refusing to apply.")

    target, binding, rebind_refusal = _di()._rebind_payload_reference(
        ctx, entry.resource_ref, entry.target_root,
        tool="integrate_delegated_patch", context=f"run_id={rid}")
    if rebind_refusal:
        return rebind_refusal
    patch_touched, parse_error = _patch_touched_paths(patch_path, target)
    if parse_error:
        return (f"⚠️ INTEGRATE_PATCH_UNREADABLE: cannot parse run {rid}'s captured "
                f"patch (git apply --numstat failed): {parse_error[:300]}")
    ordered = sorted(patch_touched)
    state_root = pathlib.Path(binding.state_drive_root)
    reserved, escape = _payload_reserved_paths(ordered, target, state_root)
    if not escape:
        # The CANDIDATE is judged too: a patch that lands an escaping symlink is
        # refused whole, exactly like a ../ path escape (gate fix 1).
        link_escapes, escape = _candidate_symlink_escapes(patch_path, target)
        reserved = sorted(set(reserved) | set(link_escapes))
    if escape:
        return (f"⚠️ INTEGRATE_DELEGATED_PATH_ESCAPE: run {rid}'s patch was NOT "
                f"applied — {escape}. The snapshot and the patch are preserved.")
    if reserved:
        verdict_path = _write_verdict(
            ctx, f"run_{rid}", outcome="blocked_reserved_paths", reason=reason,
            files=touched, manifest=manifest, applied=False, conflicts=reserved,
            protected=reserved, target=str(target))
        return (
            f"⚠️ INTEGRATE_DELEGATED_RESERVED_PATHS: run {rid}'s patch touches "
            f"{len(reserved)} reserved lifecycle/control or escaping-symlink "
            f"path(s) ({', '.join(reserved[:5])}{' …' if len(reserved) > 5 else ''}), "
            "so the WHOLE apply is refused — nothing was partially filtered or "
            "applied. The exact patch and the snapshot are preserved: read the "
            "patch, have the change redone without those paths, or "
            "integrate_delegated_patch(decision='reject') to discard. "
            f"Verdict: {verdict_path or '(unwritten)'}.")

    baseline_hash = str((entry.resource_ref or {}).get("payload_hash")
                        or manifest.get("baseline_payload_hash") or "")
    result_hash = str(manifest.get("result_content_hash") or "")
    skill_name = str((entry.resource_ref or {}).get("skill_name") or "")
    if not baseline_hash:
        return (f"⚠️ INTEGRATE_DELEGATED_BASELINE_UNVERIFIABLE: run {rid} carries no "
                "recorded baseline payload hash, so drift cannot be judged. Nothing "
                "was changed; the snapshot and the patch are preserved.")
    try:
        live_hash = _di().payload_content_hash(target)
    except Exception as exc:
        return (f"⚠️ INTEGRATE_DELEGATED_BASELINE_UNVERIFIABLE: the live payload "
                f"could not be hashed ({type(exc).__name__}: {exc}). Nothing was "
                "changed; the snapshot and the patch are preserved.")
    if live_hash != baseline_hash:
        if result_hash and live_hash == result_hash:
            # Already applied (a crashed prior attempt landed the patch before its
            # disposition row): dispose as applied instead of a false CAS conflict,
            # through the SAME finalizer — the prior attempt may have died before
            # its advisory invalidation and extension reconcile (gate fix 4).
            return _finalize_payload_apply(
                ctx, rid=rid, reason=reason, target=target, touched=touched,
                ordered=ordered, manifest=manifest, state_root=state_root,
                skill_name=skill_name, dispose=_dispose, already=True)
        verdict_path = _write_verdict(
            ctx, f"run_{rid}", outcome="baseline_drift", reason=reason, files=touched,
            manifest=manifest, applied=False, conflicts=[f"live={live_hash[:12]}",
            f"baseline={baseline_hash[:12]}"], protected=[], target=str(target))
        return (
            f"⚠️ INTEGRATE_CONFLICT: the live payload {target} CHANGED since run "
            f"{rid}'s snapshot was taken (whole-payload content hash differs), so its "
            "patch was NOT applied. YOU own this conflict: the snapshot and the patch "
            "are preserved — reconcile the payload with the captured diff, then retry, "
            "or integrate_delegated_patch(decision='reject') to discard. "
            f"Verdict: {verdict_path or '(unwritten)'}.")

    if not custody.record_patch_apply_started(drive, entry, target_root=str(target)):
        return (f"⚠️ INTEGRATE_INTENT_UNWRITTEN: the durable apply-intent row for run "
                f"{rid} could not be written. Refusing to mutate; fix the drive/event "
                "log and retry. Nothing was changed.")
    # Index-free apply with cwd = the LIVE payload (R1 item 3, probed): no .git,
    # no index, no staging is created in the live payload. Atomic on failure.
    # Config-isolated like every parent-side git invocation of this surface.
    from ouroboros.subagent_worktrees import isolated_git_env

    proc = subprocess.run(["git", "apply", str(patch_path)], cwd=str(target),
                          capture_output=True, text=True, env=isolated_git_env())
    if proc.returncode != 0:
        custody.record_patch_apply_resolved(drive, entry, reason="apply_failed")
        stderr = (proc.stderr or proc.stdout or "").strip()
        verdict_path = _write_verdict(
            ctx, f"run_{rid}", outcome="conflict", reason=reason, files=touched,
            manifest=manifest, applied=False, conflicts=[stderr[:500]], protected=[],
            target=str(target))
        return (
            f"⚠️ INTEGRATE_CONFLICT: applying run {rid}'s patch into {target} did not "
            f"apply cleanly (git apply is atomic — the payload is unchanged). git "
            f"said: {stderr[:600]}\nThe snapshot and the patch are preserved; "
            "reconcile and retry, or integrate_delegated_patch(decision='reject'). "
            f"Verdict: {verdict_path or '(unwritten)'}.")
    # Post-apply representation assert (Sol P1): the LIVE loader hash must equal
    # the recorded result hash before ANY success is claimed. On mismatch no
    # rollback is pretended: the apply intent stays PENDING (next integrate
    # answers APPLY_AMBIGUOUS) and all forensic material is preserved.
    try:
        live_after = _di().payload_content_hash(target)
        hash_error = ""
    except Exception as exc:
        live_after, hash_error = "", f"{type(exc).__name__}: {exc}"
    if result_hash and live_after != result_hash:
        try:
            from ouroboros.review_state import invalidate_advisory_after_mutation

            invalidate_advisory_after_mutation(
                pathlib.Path(str(getattr(ctx, "drive_root", "") or ".")),
                mutation_root=target, changed_paths=ordered,
                source_tool="integrate_delegated_patch")
        except Exception:
            pass
        reconcile_err = ""
        try:
            # Final Sol scope P1: the payload DID mutate, so the stale-extension
            # rule (R1 item 10) holds despite the mismatch — queue the reconcile
            # marker while recording NO success and NO disposition.
            from ouroboros.extension_reconcile_queue import request_extension_reconcile

            request_extension_reconcile(state_root, skill_name,
                                        reason="delegated_payload_apply_hash_mismatch",
                                        source="worker")
        except Exception as exc:
            log.warning("extension reconcile request failed after apply-hash "
                        "mismatch %s", rid, exc_info=True)
            reconcile_err = f"{type(exc).__name__}: {exc}"
        verdict_path = _write_verdict(
            ctx, f"run_{rid}", outcome="apply_hash_mismatch", reason=reason,
            files=touched, manifest=manifest, applied=True,
            conflicts=[f"live={live_after[:12] or hash_error[:80]}",
                       f"recorded={result_hash[:12]}"],
            protected=[], target=str(target))
        reconciled = (
            "a stale-extension reconcile marker was still QUEUED (the payload DID "
            "mutate; the server processes that marker asynchronously)"
            if not reconcile_err else
            "WARNING: the stale-extension reconcile marker could NOT be queued "
            f"({reconcile_err}) — a stale enabled extension may remain live until "
            "restart or a manual reconcile")
        return (
            f"⚠️ INTEGRATE_APPLY_HASH_MISMATCH: run {rid}'s patch WAS applied into "
            f"{target}, but the live payload loader hash does not equal the recorded "
            "result content hash — the applied bytes are NOT the reviewed candidate "
            f"representation ({hash_error or 'hash divergence'}). No success is "
            f"claimed: nothing was disposed; {reconciled}; the durable "
            "apply intent stays PENDING, so the next integrate_delegated_patch "
            "answers APPLY_AMBIGUOUS for explicit owner recovery "
            "(decision='acknowledge_ambiguous' after inspection). The snapshot and "
            f"the patch are preserved as forensic material. Verdict: "
            f"{verdict_path or '(unwritten)'}.")
    return _finalize_payload_apply(
        ctx, rid=rid, reason=reason, target=target, touched=touched,
        ordered=ordered, manifest=manifest, state_root=state_root,
        skill_name=skill_name, dispose=_dispose, already=False)
