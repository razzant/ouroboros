"""Skill discovery plus durable enabled/review/grant state.

Skills are directories with ``SKILL.md`` or ``skill.json`` manifests. State
lives under ``data/state/skills/<name>/``; missing files mean disabled and
pending review. Scripts execute via ``skill_exec``; extensions load through
``extension_loader`` after the same fresh review and grant gates.
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from ouroboros.contracts.skill_manifest import SkillManifest, SkillManifestError, canonical_skill_name, parse_skill_manifest_text
from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS
from ouroboros.contracts.schema_versions import with_schema_version
from ouroboros.skill_review_status import STATUS_BLOCKERS, STATUS_CLEAN, STATUS_PENDING, STATUS_WARNINGS, VALID_SKILL_REVIEW_STATUSES, aggregate_skill_review_status, normalize_skill_review_status, skill_review_gate
from ouroboros.utils import append_jsonl, atomic_write_json, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)


# Constants

_MANIFEST_NAMES = ("SKILL.md", "skill.json")
# Only metadata/cache names are skipped. Non-metadata dotfiles remain hashed
# and reviewed because a skill subprocess can import/source/read them.
_SKILL_DIR_CACHE_NAMES = frozenset({"__pycache__", "node_modules", ".git", ".hg", ".svn", ".idea", ".vscode", ".tox", ".pytest_cache", ".ouroboros_env", ".DS_Store"})
# Launcher/lifecycle control files are NOT runtime payload: they mark seed
# provenance and must not invalidate (or be covered by) the review hash —
# otherwise writing .seed-origin after hashing flips the verdict stale.
HASH_EXEMPT_CONTROL_FILENAMES = frozenset({".seed-origin"})

# Sensitive files are excluded from review prompts and hashes; their presence
# in a runtime-reachable skill tree is handled as a hard block below.

_REVIEW_STATUS_PASS = STATUS_CLEAN
_REVIEW_STATUS_FAIL = STATUS_BLOCKERS
_REVIEW_STATUS_ADVISORY = STATUS_WARNINGS
_REVIEW_STATUS_ADVISORY_PASS = STATUS_WARNINGS
_REVIEW_STATUS_PENDING = STATUS_PENDING
_REVIEW_STATUS_DEFERRED_PHASE4 = "pending_phase4"

VALID_REVIEW_STATUSES = VALID_SKILL_REVIEW_STATUSES


def review_status_allows_execution(status: str) -> bool:
    return bool(skill_review_gate(status)["executable_review"])


GRANTS_FILENAME = "grants.json"
SELF_AUTHORED_MARKER_FILENAME = ".self_authored.json"
# CPL4-C10: every per-skill owner-state document the runtime authors carries
# the shared ABI-2 stamp on write (review.json, enabled.json, grants.json,
# review_job.json, owner_attestation.json, accepted_rebuttals.json). Readers
# keep legacy-0 tolerance: unstamped files are never retrofitted on read.
SKILL_OWNER_STATE_SCHEMA_VERSION = 1


# Dataclasses


@dataclass
class SkillReviewState:
    """Persisted skill review verdict tied to a content hash."""

    status: str = _REVIEW_STATUS_PENDING
    content_hash: str = ""
    findings: List[Dict[str, Any]] = field(default_factory=list)
    reviewer_models: List[str] = field(default_factory=list)
    timestamp: str = ""
    prompt_chars: int = 0
    cost_usd: float = 0.0
    raw_result: str = ""
    raw_actor_records: List[Dict[str, Any]] = field(default_factory=list)
    advisory_result: Dict[str, Any] = field(default_factory=dict)
    review_profile: str = ""

    def is_stale_for(self, current_hash: str) -> bool:
        if not current_hash:
            return True
        if not self.content_hash:
            return True
        return self.content_hash != current_hash

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "content_hash": self.content_hash,
            "findings": list(self.findings),
            "reviewer_models": list(self.reviewer_models),
            "timestamp": self.timestamp,
            "prompt_chars": int(self.prompt_chars or 0),
            "cost_usd": float(self.cost_usd or 0.0),
            "raw_result": self.raw_result,
            "raw_actor_records": list(self.raw_actor_records),
        }
        if self.review_profile:
            data["review_profile"] = str(self.review_profile)
        if self.advisory_result:
            data["advisory_result"] = dict(self.advisory_result)
        has_review_verdicts = any(
            str(f.get("verdict") or "").upper() in {"PASS", "FAIL"}
            for f in self.findings
            if isinstance(f, dict)
        )
        if self.status == _REVIEW_STATUS_PENDING or not has_review_verdicts:
            data["status"] = normalize_skill_review_status(self.status)
        return data


@dataclass
class LoadedSkill:
    """Discovered skill plus durable state and source tag for UI lifecycle actions."""

    name: str
    skill_dir: pathlib.Path
    manifest: SkillManifest
    content_hash: str
    enabled: bool = False
    review: SkillReviewState = field(default_factory=SkillReviewState)
    load_error: str = ""
    source: str = "native"
    is_self_authored: bool = False
    identity_collision: bool = False

    @property
    def available_for_execution(self) -> bool:
        """True when an enabled script skill has a fresh executable review."""
        if self.load_error:
            return False
        if not self.enabled:
            return False
        if not self.manifest.is_script():
            # instruction has no payload; extension runs through PluginAPI.
            return False
        if not review_status_allows_execution(self.review.status):
            return False
        if self.review.is_stale_for(self.content_hash):
            return False
        from ouroboros.tools.skill_exec import _resolve_runtime_binary, _resolve_script_path

        runtime = (self.manifest.runtime or "").strip().lower()
        if _resolve_runtime_binary(runtime)[0] is None:
            return False
        for entry in self.manifest.scripts or []:
            if not isinstance(entry, dict):
                continue
            declared_name = str(entry.get("name") or "").strip()
            if not declared_name:
                continue
            relpath = (
                declared_name
                if "/" in declared_name or declared_name.startswith(".")
                else f"scripts/{declared_name}"
            )
            if _resolve_script_path(self.skill_dir, relpath) is not None:
                return True
        return False


@dataclass(frozen=True)
class _SkillLocationCandidate:
    """Package location without payload or state reads."""

    name: str
    location: str
    skill_dir: pathlib.Path


# Disk paths


def _skills_state_root(drive_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / "skills"


def skill_state_dir_path(drive_root: pathlib.Path, name: str) -> pathlib.Path:
    """The skill's state dir path WITHOUT creating it — for read-only
    consumers (``load_review_state``, the RC auditor's admission reuse):
    reading state must never mutate the root it reads from.

    The generation-fenced companion recovery resolves the state dir through
    here BEFORE its fence (fix-round-6); creation belongs to the post-fence
    attach in ``_publish_registrations``, so a stale refusal creates no
    directories. ``skill_state_dir`` is the creating variant for load paths.

    The name is normalized to its alnum-dashes shape before joining so a
    malicious manifest ``name: ../foo`` cannot escape the state root.
    """
    return _skills_state_root(drive_root) / _sanitize_skill_name(name)


def skill_state_dir(drive_root: pathlib.Path, name: str) -> pathlib.Path:
    """Return ``~/Ouroboros/data/state/skills/<name>/`` (created on demand)."""
    path = skill_state_dir_path(drive_root, name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_skill_name(name: str) -> str:
    """Clamp a skill name to a safe on-disk identifier.

    Keep alphanumerics, dashes, underscores, and dots; replace everything
    else with ``_``. Empty / pathological inputs become ``"_unnamed"``.
    """
    return canonical_skill_name(name)


def is_self_authored_skill_dir(
    skill_dir: pathlib.Path,
    *,
    drive_root: pathlib.Path | None = None,
) -> bool:
    """Return True when a skill carries the agent-authored provenance marker."""
    skill_dir = pathlib.Path(skill_dir)
    marker = skill_dir / SELF_AUTHORED_MARKER_FILENAME
    data = read_json_dict(marker)
    if data is None:
        return False
    try:
        payload_schema = int(data.get("schema_version") or 0)
    except (TypeError, ValueError):
        return False
    if not (
        isinstance(data, dict)
        and payload_schema == 1
        and str(data.get("origin") or "") == "self_authored"
    ):
        return False
    try:
        if drive_root is None:
            from ouroboros.config import DATA_DIR
            drive_root = pathlib.Path(DATA_DIR)
        state_marker = pathlib.Path(drive_root) / "state" / "skills" / _sanitize_skill_name(skill_dir.name) / "self_authored.json"
        state_data = read_json_dict(state_marker)
    except OSError:
        return False
    if state_data is None:
        return False
    try:
        state_schema = int(state_data.get("schema_version") or 0)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(state_data, dict)
        and state_schema == 1
        and str(state_data.get("origin") or "") == "self_authored"
        and str(state_data.get("task_id") or "") == str(data.get("task_id") or "")
        and str(state_data.get("created_at") or "") == str(data.get("created_at") or "")
    )


# Manifest discovery


class _ManifestUnreadable(RuntimeError):
    """A manifest file exists but could not be read (permissions,
    truncation, IO error, etc.). Callers translate this into a
    ``LoadedSkill`` with ``load_error`` set so the broken skill is
    still visible in ``list_skills`` instead of silently disappearing
    from discovery."""

    def __init__(self, path: pathlib.Path, err: BaseException) -> None:
        super().__init__(f"manifest {path}: {type(err).__name__}: {err}")
        self.path = path
        self.err = err


def _manifest_text_for_dir(skill_dir: pathlib.Path) -> Optional[tuple[str, pathlib.Path]]:
    """Return (manifest_text, manifest_path) for a skill dir.

    Returns ``None`` ONLY when the directory has no manifest at all
    (i.e. "this is not a skill dir"). A manifest that exists but can't
    be read raises ``_ManifestUnreadable`` so the caller can surface
    the broken skill with a ``load_error`` instead of pretending the
    dir was not a skill dir in the first place.
    """
    for candidate in _MANIFEST_NAMES:
        mf = skill_dir / candidate
        if mf.is_file():
            try:
                return mf.read_text(encoding="utf-8"), mf
            except (OSError, UnicodeDecodeError) as exc:
                # Catch BOTH IO failures and decode failures: a manifest
                # with invalid UTF-8 would otherwise crash discovery for
                # the whole skills checkout instead of degrading to a
                # single broken-skill entry.
                log.warning("Failed to read skill manifest %s", mf, exc_info=True)
                raise _ManifestUnreadable(mf, exc) from exc
    return None


def _broken_skill(
    skill_dir: pathlib.Path,
    load_error: str,
    *,
    identity_collision: bool = False,
) -> LoadedSkill:
    name = _sanitize_skill_name(skill_dir.name)
    return LoadedSkill(
        name=name,
        skill_dir=skill_dir,
        manifest=SkillManifest(
            name=name,
            description="",
            version="",
            type="instruction",
        ),
        content_hash="",
        load_error=load_error,
        identity_collision=identity_collision,
    )


def _iter_payload_files(
    skill_dir: pathlib.Path,
    *,
    manifest_entry: str = "",
    manifest_scripts: Optional[List[Dict[str, Any]]] = None,
    include_control_files: bool = False,
) -> List[pathlib.Path]:
    """Return files hashed for review freshness.

    The hash covers every regular runtime-reachable file under ``skill_dir``
    except metadata/cache/sensitive paths, lifecycle control files
    (``HASH_EXEMPT_CONTROL_FILENAMES``), and symlink escapes. Manifest entry
    points are re-added only when confined, keeping executable and reviewed
    surfaces aligned. ``include_control_files=True`` reproduces the legacy
    pre-v6.31 hash (control files included) for one-shot state migration.
    """
    out: List[pathlib.Path] = []
    resolved_root = skill_dir.resolve()

    def _add(path: pathlib.Path) -> None:
        if path not in out:
            out.append(path)

    def _add_if_confined(relpath: str) -> None:
        rel = str(relpath or "").strip()
        if not rel or rel.startswith("/") or rel.startswith("~"):
            return
        if ".." in pathlib.PurePosixPath(rel).parts:
            return
        resolved = (skill_dir / rel).resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError:
            return
        if resolved.is_file():
            _add(resolved)

    # Broad walk: everything runtime-reachable, minus metadata/cache names.
    # Every candidate is resolved back under skill_dir so symlinks cannot leak
    # outside files into reviewer prompts. Sensitive-path policy is shared with
    # repo review.
    from ouroboros.tools.review_helpers import (
        _SENSITIVE_EXTENSIONS,
        _SENSITIVE_NAMES,
    )

    def _is_sensitive(path: pathlib.Path) -> bool:
        lowered = path.name.lower()
        if lowered in _SENSITIVE_NAMES:
            return True
        for ext in _SENSITIVE_EXTENSIONS:
            if lowered.endswith(ext):
                return True
        return False

    if resolved_root.is_dir():
        for path in sorted(resolved_root.rglob("*")):
            if not path.is_file():
                continue
            try:
                rel_parts = path.relative_to(resolved_root).parts
            except ValueError:
                continue
            if any(part in _SKILL_DIR_CACHE_NAMES for part in rel_parts):
                continue
            # Only the TOP-LEVEL lifecycle marker of a NATIVE-bucket payload is
            # hash-exempt (the launcher writes it there; P3: everywhere else a
            # file by that name is ordinary runtime-reachable payload and stays
            # in the reviewed surface).
            if (
                not include_control_files
                and len(rel_parts) == 1
                and path.name in HASH_EXEMPT_CONTROL_FILENAMES
                and resolved_root.parent.name == "native"
            ):
                continue
            if _is_sensitive(path):
                # Fail closed: a reviewed skill could still read a skipped
                # credential-shaped file at runtime.
                raise SkillPayloadUnreadable(
                    str(path.relative_to(resolved_root)),
                    RuntimeError(
                        "sensitive-shape filename present in skill tree "
                        "(e.g. .env / credentials.json / .pem). Rename "
                        "or relocate the file outside the skill checkout."
                    ),
                )
            # Symlink escape guard: resolve the final path and re-check
            # confinement under skill_dir.
            try:
                real = path.resolve()
            except (OSError, RuntimeError):
                log.warning("Could not resolve skill file %s", path, exc_info=True)
                continue
            try:
                real.relative_to(resolved_root)
            except ValueError:
                log.warning(
                    "Skill file %s resolves outside skill_dir (%s) — excluded from review pack.",
                    path, resolved_root,
                )
                continue
            _add(path)

    # Add manifest-declared entry/scripts explicitly after confinement checks.
    _add_if_confined(manifest_entry)
    for script_entry in manifest_scripts or []:
        if not isinstance(script_entry, dict):
            continue
        declared_name = str(script_entry.get("name") or "").strip()
        if not declared_name:
            continue
        _add_if_confined(declared_name)
        if "/" not in declared_name:
            _add_if_confined(f"scripts/{declared_name}")

    out.sort()
    return out


class SkillPayloadUnreadable(RuntimeError):
    """Raised when the skill hash cannot cover the full runtime surface."""

    def __init__(self, relpath: str, err: BaseException) -> None:
        super().__init__(
            f"Skill payload {relpath!r} unreadable: {type(err).__name__}: {err}"
        )
        self.relpath = relpath
        self.err = err


def reduce_skill_content_hash(file_digests: Iterable[tuple[str, bytes]]) -> str:
    """Reduce canonically ordered relative paths and file digests."""
    digest = hashlib.sha256()
    for rel, file_digest in file_digests:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest)
    return digest.hexdigest()


def compute_content_hash(
    skill_dir: pathlib.Path,
    *,
    manifest_entry: str = "",
    manifest_scripts: Optional[List[Dict[str, Any]]] = None,
    include_control_files: bool = False,
) -> str:
    """Compute the review-staleness hash for manifest plus payload files.

    Unreadable payload files fail closed via :class:`SkillPayloadUnreadable`
    so callers never emit a PASS over a partial runtime surface.
    ``include_control_files=True`` reproduces the legacy pre-v6.31 hash for
    one-shot state migration only.
    """
    skill_dir = skill_dir.resolve()
    file_digests: List[tuple[str, bytes]] = []
    for file_path in _iter_payload_files(
        skill_dir,
        manifest_entry=manifest_entry,
        manifest_scripts=manifest_scripts,
        include_control_files=include_control_files,
    ):
        rel = file_path.relative_to(skill_dir).as_posix()
        # Stream hashing so large assets cannot force whole-file allocation.
        file_digest = hashlib.sha256()
        try:
            with file_path.open("rb") as fh:
                while True:
                    chunk = fh.read(64 * 1024)
                    if not chunk:
                        break
                    file_digest.update(chunk)
        except OSError as exc:
            log.warning("Failed to read skill payload file %s", file_path, exc_info=True)
            raise SkillPayloadUnreadable(rel, exc) from exc
        file_digests.append((rel, file_digest.digest()))
    return reduce_skill_content_hash(file_digests)


# State persistence


def load_enabled(drive_root: pathlib.Path, name: str) -> bool:
    state = read_json_dict(skill_state_dir(drive_root, name) / "enabled.json")
    if not isinstance(state, dict):
        return False
    enabled = state.get("enabled")
    return enabled if isinstance(enabled, bool) else False


def save_enabled(
    drive_root: pathlib.Path,
    name: str,
    enabled: bool,
    *,
    actor: str = "",
    reason: str = "",
) -> None:
    """Persist enablement and best-effort append one typed disclosure row.

    ``actor`` is a data label, never a gate: no branch reads it, and an
    unlabelled writer records an empty actor rather than nothing at all. An
    append failure is logged and never blocks the enablement change.
    """
    previous = load_enabled(drive_root, name)
    atomic_write_json(
        skill_state_dir(drive_root, name) / "enabled.json",
        with_schema_version(
            {
                "enabled": bool(enabled),
                "updated_at": utc_now_iso(),
            },
            SKILL_OWNER_STATE_SCHEMA_VERSION,
        ),
    )
    try:
        append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "skill_enabled_changed",
            "skill": name,
            "enabled": bool(enabled),
            "previous": bool(previous),
            "actor": str(actor or ""),
            "reason": str(reason or ""),
        })
    except Exception:
        log.debug("Failed to append skill_enabled_changed for %s", name, exc_info=True)


def load_review_state(
    drive_root: pathlib.Path,
    name: str,
    *,
    skill_type: str = "",
    is_module_widget: bool = False,
    skill_dir: Optional[pathlib.Path] = None,
) -> SkillReviewState:
    # Read-only over the state root (skill_state_dir_path, never mkdir): the
    # RC auditor reuses this admission read over a root it must not mutate.
    data = read_json_dict(skill_state_dir_path(drive_root, name) / "review.json")
    if not isinstance(data, dict):
        return SkillReviewState()
    raw_status = str(data.get("status") or "").lower()
    # Legacy ``pending_phase4`` persisted states now normalize to pending.
    if raw_status == _REVIEW_STATUS_DEFERRED_PHASE4:
        raw_status = _REVIEW_STATUS_PENDING
    raw_status = normalize_skill_review_status(raw_status)
    findings = data.get("findings") if isinstance(data.get("findings"), list) else []
    clean_findings = [f for f in findings if isinstance(f, dict)]
    review_profile = str(data.get("review_profile") or "").strip()
    # The official_hub downgrade only applies to a payload whose Hub provenance is
    # still locally intact. If the sidecar was removed or the payload moved out of
    # the OuroborosHub bucket since review, drop the persisted profile so the
    # severity-driven downgrade is NOT trusted on a no-longer-official payload.
    if review_profile == "official_hub" and skill_dir is not None:
        if not (pathlib.Path(skill_dir) / ".ouroboroshub.json").is_file():
            review_profile = ""
    # A native_seed trust verdict is valid ONLY while launcher provenance holds
    # (.seed-origin present). The marker is hash-exempt, so its removal does not
    # stale the hash — invalidate the verdict explicitly: removing the marker
    # reclassifies the skill as user-managed (CHECKLISTS §Skills) and the
    # launcher-trust verdict must not survive into that life. Provenance that
    # CANNOT be verified (caller did not pass skill_dir) fails safe the same
    # way: native_seed trust is meaningless without the marker check.
    if review_profile == "native_seed":
        if skill_dir is None or not (pathlib.Path(skill_dir) / ".seed-origin").is_file():
            return SkillReviewState()
    # An owner-attested verdict (C1, v6.39: the owner explicitly skipped the EXPENSIVE LLM
    # review for their OWN skill) is valid ONLY while the owner-issued marker is present in
    # the protected owner-state dir. Removing the marker invalidates the verdict (fail-safe:
    # the skill drops back to pending), exactly like native_seed provenance. Content edits
    # still stale the verdict through the normal content_hash check.
    if review_profile == "owner_attested":
        if not (skill_state_dir_path(drive_root, name) / "owner_attestation.json").is_file():
            return SkillReviewState()
    has_review_verdicts = any(
        str(f.get("verdict") or "").upper() in {"PASS", "FAIL"}
        for f in clean_findings
    )
    if clean_findings and has_review_verdicts:
        status = aggregate_skill_review_status(
            clean_findings,
            skill_type or "script",
            is_module_widget=is_module_widget,
            review_profile=review_profile,
        )
    elif raw_status == _REVIEW_STATUS_PENDING:
        status = _REVIEW_STATUS_PENDING
    else:
        status = raw_status if raw_status in VALID_REVIEW_STATUSES else _REVIEW_STATUS_PENDING
    reviewers = (
        data.get("reviewer_models")
        if isinstance(data.get("reviewer_models"), list)
        else []
    )
    raw_actor_records = (
        data.get("raw_actor_records")
        if isinstance(data.get("raw_actor_records"), list)
        else []
    )
    advisory_result = (
        data.get("advisory_result")
        if isinstance(data.get("advisory_result"), dict)
        else {}
    )
    try:
        prompt_chars = int(data.get("prompt_chars") or 0)
    except (TypeError, ValueError):
        prompt_chars = 0
    try:
        cost_usd = float(data.get("cost_usd") or 0.0)
    except (TypeError, ValueError):
        cost_usd = 0.0
    return SkillReviewState(
        status=status,
        content_hash=str(data.get("content_hash") or ""),
        findings=clean_findings,
        reviewer_models=[str(m) for m in reviewers if m],
        timestamp=str(data.get("timestamp") or ""),
        prompt_chars=prompt_chars,
        cost_usd=cost_usd,
        raw_result=str(data.get("raw_result") or ""),
        raw_actor_records=[r for r in raw_actor_records if isinstance(r, dict)],
        advisory_result=dict(advisory_result),
        review_profile=review_profile,
    )


def save_review_state(
    drive_root: pathlib.Path,
    name: str,
    review: SkillReviewState,
) -> None:
    atomic_write_json(
        skill_state_dir(drive_root, name) / "review.json",
        with_schema_version(review.to_dict(), SKILL_OWNER_STATE_SCHEMA_VERSION),
    )


def requested_core_setting_keys(env_keys: List[str]) -> List[str]:
    """Return manifest-requested setting keys that require explicit owner grants."""
    forbidden_upper = {key.upper() for key in FORBIDDEN_SKILL_SETTINGS}
    requestable_when_absent = {"TELEGRAM_BOT_TOKEN"}
    try:
        from ouroboros.config import SETTINGS_DEFAULTS, load_settings
        settings = load_settings()
        custom_secret_upper = {
            str(key).upper()
            for key in settings
            if str(key).upper() not in SETTINGS_DEFAULTS
            and str(key).upper().replace("_", "").isalnum()
        }
    except Exception:
        custom_secret_upper = set()
    out: List[str] = []
    for raw_key in env_keys or []:
        key = str(raw_key or "").strip().upper()
        if key and (key in forbidden_upper or key in custom_secret_upper or key in requestable_when_absent) and key not in out:
            out.append(key)
    return out



_GRANTABLE_SKILL_PERMISSIONS = frozenset({
    "inject_chat",
    "presence",
    "subscribe_event:chat.outbound",
    "subscribe_event:chat.typing",
    "subscribe_event:chat.photo",
    "subscribe_event:chat.video",
})


def requested_skill_permissions(
    permissions: List[str],
    subscribe_events: Optional[List[str]] = None,
) -> List[str]:
    """Return manifest-requested privileged permissions needing owner grants."""
    requested: List[str] = []
    permission_set = {str(item or "").strip() for item in (permissions or [])}
    if "inject_chat" in permission_set:
        requested.append("inject_chat")
    if "presence" in permission_set:
        requested.append("presence")
    if "subscribe_event" in permission_set:
        for raw_topic in subscribe_events or []:
            topic = str(raw_topic or "").strip()
            grant = f"subscribe_event:{topic}"
            if grant in _GRANTABLE_SKILL_PERMISSIONS and grant not in requested:
                requested.append(grant)
    return requested


def _unique_text(values: Any, *, upper: bool = False) -> List[str]:
    out: List[str] = []
    for raw in values or []:
        item = str(raw or "").strip()
        item = item.upper() if upper else item
        if item and item not in out:
            out.append(item)
    return out


def _merge_allowed(*value_groups: Any, allowed: set[str], upper: bool = False) -> List[str]:
    return [
        item
        for item in _unique_text(
            (raw for group in value_groups for raw in (group or [])),
            upper=upper,
        )
        if item in allowed
    ]


def load_skill_grants(drive_root: pathlib.Path, name: str) -> Dict[str, Any]:
    data = read_json_dict(skill_state_dir(drive_root, name) / GRANTS_FILENAME)
    if not isinstance(data, dict):
        return {"granted_keys": [], "granted_permissions": [], "updated_at": ""}
    keys = _unique_text(data.get("granted_keys"), upper=True)
    requested = _unique_text(data.get("requested_keys"), upper=True)
    permissions = _unique_text(data.get("granted_permissions"))
    requested_permissions = _unique_text(data.get("requested_permissions"))
    return {
        "granted_keys": keys,
        "requested_keys": requested,
        "granted_permissions": permissions,
        "requested_permissions": requested_permissions,
        "content_hash": str(data.get("content_hash") or ""),
        "updated_at": str(data.get("updated_at") or ""),
    }


def save_skill_grants(
    drive_root: pathlib.Path,
    name: str,
    granted_keys: List[str],
    *,
    content_hash: str,
    requested_keys: List[str],
    granted_permissions: Optional[List[str]] = None,
    requested_permissions: Optional[List[str]] = None,
) -> None:
    """Persist grants, merging partial approvals only for the same request hash."""
    allowed = set(requested_core_setting_keys(requested_keys))
    existing = load_skill_grants(drive_root, name)
    persisted_match = (
        str(existing.get("content_hash") or "") == str(content_hash or "")
        and sorted(existing.get("requested_keys") or []) == sorted(allowed)
    )
    merged = _merge_allowed(
        existing.get("granted_keys") if persisted_match else [],
        granted_keys,
        allowed=allowed,
        upper=True,
    )
    allowed_permissions = set(requested_permissions or [])
    existing_permissions = _merge_allowed(
        existing.get("granted_permissions") if persisted_match else [],
        granted_permissions,
        allowed=allowed_permissions,
    )
    # Stamp at write (CPL4-C10): load_skill_grants normalizes to a fixed key
    # set, so the stamp must be re-authored here rather than carried through
    # the read-merge round trip.
    atomic_write_json(
        skill_state_dir(drive_root, name) / GRANTS_FILENAME,
        with_schema_version(
            {
                "granted_keys": merged,
                "requested_keys": sorted(allowed),
                "granted_permissions": sorted(existing_permissions),
                "requested_permissions": sorted(allowed_permissions),
                "content_hash": str(content_hash or ""),
                "updated_at": utc_now_iso(),
            },
            SKILL_OWNER_STATE_SCHEMA_VERSION,
        ),
    )


def grant_status_for_skill(drive_root: pathlib.Path, skill: LoadedSkill) -> Dict[str, Any]:
    if bool(getattr(skill, "identity_collision", False)):
        # Collision placeholders deliberately carry no manifest/state payload.
        # Do not create ``state/skills/<name>`` merely to summarize a target
        # whose physical identity has not been selected.
        return {
            "requested_keys": [],
            "granted_keys": [],
            "missing_keys": [],
            "requested_permissions": [],
            "granted_permissions": [],
            "missing_permissions": [],
            "all_granted": True,
            "usable": False,
            "unsupported_for_skill_type": False,
            "content_hash": "",
            "updated_at": "",
        }
    requested = requested_core_setting_keys(list(skill.manifest.env_from_settings or []))
    requested_permissions = requested_skill_permissions(
        list(skill.manifest.permissions or []),
        list(getattr(skill.manifest, "subscribe_events", []) or []),
    )
    grants = load_skill_grants(drive_root, skill.name)
    grant_hash_ok = str(grants.get("content_hash") or "") == str(skill.content_hash or "")
    grant_request_ok = sorted(grants.get("requested_keys") or []) == sorted(requested)
    permission_request_ok = sorted(grants.get("requested_permissions") or []) == sorted(requested_permissions)
    persisted_grants = set(grants.get("granted_keys") or []) if grant_hash_ok and grant_request_ok else set()
    persisted_permissions = (
        set(grants.get("granted_permissions") or [])
        if grant_hash_ok and permission_request_ok
        else set()
    )
    granted = [key for key in requested if key in persisted_grants]
    granted_permissions = [perm for perm in requested_permissions if perm in persisted_permissions]
    missing = [key for key in requested if key not in set(granted)]
    missing_permissions = [perm for perm in requested_permissions if perm not in set(granted_permissions)]
    review_ready = review_status_allows_execution(skill.review.status) and not skill.review.is_stale_for(skill.content_hash)
    # Scripts receive core keys via _scrub_env; extensions via PluginAPI.
    # Instruction skills cannot receive core keys.
    eligible_type = skill.manifest.is_script() or skill.manifest.is_extension()
    unsupported = bool((requested or requested_permissions) and not eligible_type)
    return {
        "requested_keys": requested,
        "granted_keys": granted,
        "missing_keys": missing,
        "requested_permissions": requested_permissions,
        "granted_permissions": granted_permissions,
        "missing_permissions": missing_permissions,
        "all_granted": not missing and not missing_permissions and not unsupported,
        "usable": review_ready and not missing and not missing_permissions and not unsupported,
        "unsupported_for_skill_type": unsupported,
        "content_hash": grants.get("content_hash", ""),
        "updated_at": grants.get("updated_at", ""),
    }


@dataclass
class AutoGrantOutcome:
    granted: bool = False
    requested_keys: List[str] = field(default_factory=list)
    granted_keys: List[str] = field(default_factory=list)
    requested_permissions: List[str] = field(default_factory=list)
    granted_permissions: List[str] = field(default_factory=list)


def auto_grant_if_enabled(drive_root: pathlib.Path, skill: LoadedSkill) -> AutoGrantOutcome:
    """Grant requested keys/permissions after review when the owner toggle is on."""
    requested_keys = requested_core_setting_keys(list(skill.manifest.env_from_settings or []))
    requested_permissions = requested_skill_permissions(
        list(skill.manifest.permissions or []),
        list(getattr(skill.manifest, "subscribe_events", []) or []),
    )
    outcome = AutoGrantOutcome(
        requested_keys=requested_keys,
        requested_permissions=requested_permissions,
    )
    try:
        from ouroboros.config import get_auto_grant_enabled
    except Exception:
        return outcome
    if not get_auto_grant_enabled():
        return outcome
    if skill.load_error:
        return outcome
    if skill.review.is_stale_for(skill.content_hash):
        return outcome
    if not review_status_allows_execution(skill.review.status):
        return outcome
    if normalize_skill_review_status(skill.review.status) == _REVIEW_STATUS_PENDING:
        return outcome
    if not requested_keys and not requested_permissions:
        return outcome
    save_skill_grants(
        drive_root,
        skill.name,
        requested_keys,
        content_hash=skill.content_hash,
        requested_keys=requested_keys,
        granted_permissions=requested_permissions,
        requested_permissions=requested_permissions,
    )
    return AutoGrantOutcome(
        granted=True,
        requested_keys=requested_keys,
        granted_keys=list(requested_keys),
        requested_permissions=requested_permissions,
        granted_permissions=list(requested_permissions),
    )


# Discovery / loading


def _safe_listdir(root: pathlib.Path) -> List[pathlib.Path]:
    try:
        return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        log.warning("Failed to list skills repo %s", root, exc_info=True)
        return []


def _looks_like_skill_dir(path: pathlib.Path) -> bool:
    """Return True when ``path`` directly contains a skill manifest."""
    if not path.is_dir():
        return False
    for candidate in _MANIFEST_NAMES:
        if (path / candidate).is_file():
            return True
    return False


def load_skill(
    skill_dir: pathlib.Path,
    drive_root: pathlib.Path,
) -> Optional[LoadedSkill]:
    """Load one skill, returning ``None`` only when no manifest exists."""
    skill_dir = skill_dir.resolve()
    try:
        manifest_read = _manifest_text_for_dir(skill_dir)
    except _ManifestUnreadable as exc:
        return _broken_skill(skill_dir, f"manifest unreadable: {exc}")
    if manifest_read is None:
        return None
    manifest_text, manifest_path = manifest_read

    try:
        manifest = parse_skill_manifest_text(manifest_text)
    except SkillManifestError as exc:
        return _broken_skill(skill_dir, f"manifest parse error: {exc}")

    # Runtime/state/tool identity is the directory basename. manifest.name is
    # display metadata and may be localized or renamed.
    if not manifest.name:
        manifest.name = skill_dir.name

    name = _sanitize_skill_name(skill_dir.name)
    load_error = ""
    try:
        content_hash = compute_content_hash(
            skill_dir,
            manifest_entry=manifest.entry,
            manifest_scripts=manifest.scripts,
        )
    except SkillPayloadUnreadable as exc:
        content_hash = ""
        load_error = f"payload unreadable: {exc}"
    enabled = load_enabled(drive_root, name)
    is_module_widget = (
        manifest.is_extension()
        and isinstance(manifest.ui_tab, dict)
        and str(((manifest.ui_tab or {}).get("render") or {}).get("kind") or "") == "module"
    )
    review = load_review_state(
        drive_root,
        name,
        skill_type=manifest.type,
        is_module_widget=is_module_widget,
        skill_dir=skill_dir,
    )

    # Extensions share review/enable/hash gates with scripts, but register
    # through PluginAPI instead of skill_exec.

    return LoadedSkill(
        name=name,
        skill_dir=skill_dir,
        manifest=manifest,
        content_hash=content_hash,
        enabled=enabled,
        review=review,
        load_error=load_error,
        is_self_authored=is_self_authored_skill_dir(skill_dir, drive_root=drive_root),
    )


def _resolve_data_skills_dir(
    drive_root: Optional[pathlib.Path] = None,
) -> Optional[pathlib.Path]:
    """Return the data-plane skills root if it exists; never create it."""
    if drive_root is not None:
        candidate = pathlib.Path(drive_root) / "skills"
        return candidate if candidate.is_dir() else None
    try:
        from ouroboros.config import resolve_data_skills_dir, DATA_DIR
        return resolve_data_skills_dir(DATA_DIR)
    except Exception:
        return None


_ORPHAN_NAME_FRAGMENTS = (".replaced-", ".staging-", ".tmp-")


def _is_orphan_marker_name(name: str) -> bool:
    """Return True for install backup/staging names that are not live skills."""
    cleaned = (name or "").strip()
    if not cleaned:
        return False
    return any(token in cleaned for token in _ORPHAN_NAME_FRAGMENTS)


def _walk_skill_packages(
    root: pathlib.Path,
    *,
    selected_manifestless_name: str = "",
    include_manifestless_root: bool = False,
    excluded_manifestless_children: frozenset[str] = frozenset(),
    listdir: Optional[Callable[[pathlib.Path], List[pathlib.Path]]] = None,
) -> List[pathlib.Path]:
    """Yield ordinary packages plus one exact selected manifestless target.

    ``listdir`` overrides the traversal reader: the runtime keeps the
    fail-soft ``_safe_listdir`` default (an unreadable repo loads as empty),
    while a strict consumer (the RC auditor) passes a raising lister so a
    traversal ``OSError`` surfaces instead of silently emptying the walk."""
    if listdir is None:
        listdir = _safe_listdir
    out: List[pathlib.Path] = []
    if not root.is_dir():
        return out
    if _looks_like_skill_dir(root):
        # Back-compat: OUROBOROS_SKILLS_REPO_PATH may point at one skill.
        out.append(root)
        return out
    root_is_selected_manifestless = (
        include_manifestless_root
        and selected_manifestless_name
        and not _is_orphan_marker_name(root.name)
        and _sanitize_skill_name(root.name) == selected_manifestless_name
    )
    found_manifest_package = False
    for child in listdir(root):
        if _is_orphan_marker_name(child.name):
            continue
        if _looks_like_skill_dir(child):
            out.append(child)
            found_manifest_package = True
            continue
        child_has_manifest_package = False
        # One level deeper for grouping containers such as native/clawhub.
        for grandchild in listdir(child):
            if _is_orphan_marker_name(grandchild.name):
                continue
            if _looks_like_skill_dir(grandchild):
                out.append(grandchild)
                child_has_manifest_package = True
                found_manifest_package = True
            elif (
                selected_manifestless_name
                and _sanitize_skill_name(grandchild.name)
                == selected_manifestless_name
            ):
                out.append(grandchild)
        if (
            selected_manifestless_name
            and not child_has_manifest_package
            and child.name not in excluded_manifestless_children
            and _sanitize_skill_name(child.name) == selected_manifestless_name
        ):
            out.append(child)
    if root_is_selected_manifestless and not found_manifest_package:
        # Preserve the direct-checkout interpretation only for a leaf package;
        # a checkout root containing real child packages remains a container.
        return [root]
    return out


def _classify_skill_location(
    skill_dir: pathlib.Path,
    *,
    data_skills_root: Optional[pathlib.Path],
    user_repo_root: Optional[pathlib.Path],
) -> str:
    """Classify physical location without consulting provenance or state."""
    from ouroboros.config import (
        SKILL_SOURCE_EXTERNAL,
        SKILL_SOURCE_SUBDIRS,
        SKILL_SOURCE_USER_REPO,
    )

    resolved = skill_dir.resolve()
    if data_skills_root is not None:
        try:
            rel = resolved.relative_to(data_skills_root.resolve())
            if rel.parts and rel.parts[0] in SKILL_SOURCE_SUBDIRS:
                return rel.parts[0]
            return SKILL_SOURCE_EXTERNAL
        except ValueError:
            pass
    if user_repo_root is not None:
        try:
            resolved.relative_to(user_repo_root.resolve())
            return SKILL_SOURCE_USER_REPO
        except ValueError:
            pass
    return SKILL_SOURCE_EXTERNAL


def _skill_location_inventory(
    drive_root: pathlib.Path,
    repo_path: str | None = None,
    *,
    selected_manifestless_name: str = "",
) -> tuple[_SkillLocationCandidate, ...]:
    """Return deduplicated manifest locations without payload/state reads."""
    from ouroboros.config import SKILL_SOURCE_SUBDIRS

    if repo_path is None:
        from ouroboros.config import get_skills_repo_path

        repo_path = get_skills_repo_path()
    repo_path = str(repo_path or "").strip()

    data_skills_root = _resolve_data_skills_dir(drive_root)
    user_repo_root: Optional[pathlib.Path] = None
    if repo_path:
        try:
            candidate = pathlib.Path(repo_path).expanduser().resolve()
        except OSError:
            candidate = None
        if candidate is not None and candidate.is_dir():
            user_repo_root = candidate

    roots: List[tuple[pathlib.Path, bool]] = []
    if data_skills_root is not None:
        roots.append((data_skills_root, False))
    if user_repo_root is not None:
        roots.append((user_repo_root, True))

    inventory: List[_SkillLocationCandidate] = []
    seen_dirs: set[pathlib.Path] = set()
    for root, include_manifestless_root in roots:
        excluded_children = (
            frozenset(SKILL_SOURCE_SUBDIRS)
            if not include_manifestless_root
            else frozenset()
        )
        for entry in _walk_skill_packages(
            root,
            selected_manifestless_name=selected_manifestless_name,
            include_manifestless_root=include_manifestless_root,
            excluded_manifestless_children=excluded_children,
        ):
            try:
                resolved = entry.resolve()
            except OSError:
                continue
            if resolved in seen_dirs:
                continue
            seen_dirs.add(resolved)
            try:
                location = _classify_skill_location(
                    resolved,
                    data_skills_root=data_skills_root,
                    user_repo_root=user_repo_root,
                )
            except OSError:
                continue
            inventory.append(
                _SkillLocationCandidate(
                    name=_sanitize_skill_name(resolved.name),
                    location=location,
                    skill_dir=resolved,
                )
            )

    return tuple(
        sorted(
            inventory,
            key=lambda item: (item.name, item.location, str(item.skill_dir)),
        )
    )


def skill_identity_collision_names(
    drive_root: pathlib.Path,
    repo_path: str | None = None,
) -> frozenset[str]:
    """Return ambiguous canonical identities without loading payload state."""
    counts: Dict[str, int] = {}
    for candidate in _skill_location_inventory(drive_root, repo_path=repo_path):
        counts[candidate.name] = counts.get(candidate.name, 0) + 1
    return frozenset(name for name, count in counts.items() if count > 1)


def _select_skill_location(
    candidates: tuple[_SkillLocationCandidate, ...],
    *,
    name: str,
    location: str,
    require_unique_identity: bool = True,
) -> Optional[_SkillLocationCandidate]:
    """Select one physical location, preserving global collision evidence."""
    canonical_name = _sanitize_skill_name(name)
    identity = tuple(item for item in candidates if item.name == canonical_name)
    if not identity:
        # The caller alone decides whether a missing explicit ``external``
        # location starts the existing manifest-first creation flow.
        return None

    evidence = ", ".join(
        f"{item.location}:{item.skill_dir}" for item in identity
    )
    if require_unique_identity and len(identity) > 1:
        raise ValueError(
            f"Skill name collision for {canonical_name!r}: {evidence}"
        )

    selected = tuple(item for item in identity if item.location == location)
    if len(selected) > 1:
        raise ValueError(
            f"Skill location collision for {canonical_name!r} in "
            f"{location!r}: {evidence}"
        )
    if not selected:
        raise ValueError(
            f"Skill {canonical_name!r} exists, but not in requested "
            f"location {location!r}: {evidence}"
        )
    return selected[0]


def _skill_location_conflict_error(
    drive_root: pathlib.Path,
    *,
    name: str,
    location: str,
    target_dir: pathlib.Path,
    repo_path: str | None = None,
) -> str:
    """Purely reject an identity that cannot map to one exact target."""

    try:
        selected = _select_skill_location(
            _skill_location_inventory(drive_root, repo_path=repo_path),
            name=name,
            location=location,
        )
    except ValueError as exc:
        return f"Skill identity collision: {exc}"
    if selected is not None and selected.skill_dir.resolve() != pathlib.Path(target_dir).resolve():
        return (
            f"Skill identity collision: {name!r} resolves to "
            f"{selected.skill_dir}, not {target_dir}"
        )
    return ""


def _classify_skill_source(
    skill_dir: pathlib.Path,
    *,
    location: str,
    drive_root: pathlib.Path,
) -> str:
    """Return the source tag, using provenance sidecars for trusted buckets."""
    from ouroboros.config import (
        SKILL_SOURCE_CLAWHUB,
        SKILL_SOURCE_EXTERNAL,
        SKILL_SOURCE_NATIVE,
        SKILL_SOURCE_OUROBOROSHUB,
        SKILL_SOURCE_SELF_AUTHORED,
        SKILL_SOURCE_USER_REPO,
    )
    if location == SKILL_SOURCE_USER_REPO:
        return SKILL_SOURCE_USER_REPO
    if is_self_authored_skill_dir(skill_dir, drive_root=drive_root):
        return SKILL_SOURCE_SELF_AUTHORED
    marker_by_source = {
        SKILL_SOURCE_NATIVE: ".seed-origin",
        SKILL_SOURCE_CLAWHUB: ".clawhub.json",
        SKILL_SOURCE_OUROBOROSHUB: ".ouroboroshub.json",
    }
    marker = marker_by_source.get(location)
    if marker:
        return location if (skill_dir / marker).is_file() else SKILL_SOURCE_EXTERNAL
    return SKILL_SOURCE_EXTERNAL


def _load_skill_location_candidates(
    candidates: tuple[_SkillLocationCandidate, ...],
    *,
    drive_root: pathlib.Path,
    include_manifestless: bool = False,
) -> List[LoadedSkill]:
    """Load one inventory with the ordinary collision/source semantics."""
    skills: List[LoadedSkill] = []
    by_name: Dict[str, List[_SkillLocationCandidate]] = {}
    for candidate in candidates:
        by_name.setdefault(candidate.name, []).append(candidate)
    for name, group in by_name.items():
        if len(group) > 1:
            dirs = ", ".join(
                f"{candidate.location}:{candidate.skill_dir}"
                for candidate in group
            )
            error = (
                f"Skill name collision: multiple checkout directories "
                f"({dirs}) sanitise to {name!r}. Rename the directories "
                "so their basenames yield distinct identifiers before "
                "enabling / reviewing / executing."
            )
            for candidate in group:
                broken = _broken_skill(
                    candidate.skill_dir,
                    error,
                    identity_collision=True,
                )
                broken.source = candidate.location
                skills.append(broken)
            continue

        candidate = group[0]
        loaded = load_skill(candidate.skill_dir, drive_root)
        if loaded is None and include_manifestless:
            loaded = _broken_skill(candidate.skill_dir, "manifest missing")
        if loaded is None:
            continue
        loaded.source = _classify_skill_source(
            candidate.skill_dir,
            location=candidate.location,
            drive_root=drive_root,
        )
        skills.append(loaded)

    skills.sort(key=lambda skill: (skill.name, str(skill.skill_dir)))
    return skills


def discover_skills(
    drive_root: pathlib.Path,
    repo_path: str | None = None,
) -> List[LoadedSkill]:
    """Scan data-plane skills plus the optional user checkout."""
    if repo_path is None:
        from ouroboros.config import get_skills_repo_path
        repo_path = get_skills_repo_path()
    repo_path = str(repo_path or "").strip()

    candidates = _skill_location_inventory(drive_root, repo_path=repo_path)

    return _load_skill_location_candidates(candidates, drive_root=drive_root)


def discover_selected_skill_candidates(
    drive_root: pathlib.Path,
    name: str,
    *,
    repo_path: str | None = None,
) -> List[LoadedSkill]:
    """Resolve one selected identity, including a just-deleted manifest.

    The manifestless opt-in is intentionally unavailable to ordinary passive
    discovery. It exists only so a card opened before deletion can still start
    an agent repair task and so hidden physical collisions remain ambiguous.
    """
    if repo_path is None:
        from ouroboros.config import get_skills_repo_path

        repo_path = get_skills_repo_path()
    configured_repo = str(repo_path or "").strip()
    canonical_name = _sanitize_skill_name(name)
    candidates = tuple(
        candidate
        for candidate in _skill_location_inventory(
            drive_root,
            repo_path=configured_repo,
            selected_manifestless_name=canonical_name,
        )
        if candidate.name == canonical_name
    )
    return _load_skill_location_candidates(
        candidates,
        drive_root=drive_root,
        include_manifestless=True,
    )


def find_skill(
    drive_root: pathlib.Path,
    name: str,
    *,
    repo_path: str | None = None,
) -> Optional[LoadedSkill]:
    """Return one skill by name, including broken manifests with ``load_error``."""
    safe = _sanitize_skill_name(name)
    for skill in discover_skills(drive_root, repo_path=repo_path):
        if skill.name == safe:
            return skill
    return None


_MAX_CONFLICT_PROJECTION = 8


def enabled_skill_conflicts(
    skill: LoadedSkill,
    skills: List[LoadedSkill],
) -> List[str]:
    """Return enabled installed peers conflicting with ``skill``.

    A declaration on either side is authoritative, so one-sided manifests are
    enforced symmetrically. Missing and disabled peers are deliberately inert.
    """
    declared = set(skill.manifest.conflicts or [])
    conflicts = {
        peer.name
        for peer in skills
        if peer.name != skill.name
        and peer.enabled
        and (
            peer.name in declared
            or skill.name in set(peer.manifest.conflicts or [])
        )
    }
    return sorted(conflicts)


def skill_conflict_status(
    skill: LoadedSkill,
    skills: List[LoadedSkill],
) -> Optional[Dict[str, Any]]:
    """Return a bounded API-safe projection of enabled peer conflicts."""
    names = enabled_skill_conflicts(skill, skills)
    if not names:
        return None
    visible = names[:_MAX_CONFLICT_PROJECTION]
    return {
        "code": "skill_conflict",
        "skills": visible,
        "omitted": len(names) - len(visible),
    }


def list_available_for_execution(
    drive_root: pathlib.Path,
    *,
    repo_path: str | None = None,
) -> List[LoadedSkill]:
    """Return only skills that are enabled + have a fresh executable review."""
    from ouroboros.skill_readiness import skill_readiness_for_execution

    out: List[LoadedSkill] = []
    skills = discover_skills(drive_root, repo_path=repo_path)
    for skill in skills:
        if skill.available_for_execution and skill_readiness_for_execution(drive_root, skill, skills=skills).ready:
            out.append(skill)
    return out


# Status helpers consumed by the model-facing catalogue: the per-turn Installed
# Skills section (context.py) and the list_skills tool.


def summarize_skills(drive_root: pathlib.Path) -> Dict[str, Any]:
    """Return a compact catalogue summary for the model-facing skill catalogue."""
    skills = discover_skills(drive_root)
    tool_surfaces_by_skill: Dict[str, List[Dict[str, str]]] = {}
    try:
        from ouroboros.extension_loader import _lock as _ext_lock, _tools as _ext_tools
        with _ext_lock:
            for tool in _ext_tools.values():
                skill_name = str(tool.get("skill") or "")
                if not skill_name:
                    continue
                tool_surfaces_by_skill.setdefault(skill_name, []).append({
                    "name": str(tool.get("name") or ""),
                    "description": str(tool.get("description") or ""),
                })
    except Exception:
        tool_surfaces_by_skill = {}
    from ouroboros.config import get_runtime_mode
    from ouroboros.skill_readiness import skill_readiness_for_execution

    rows: List[Dict[str, Any]] = []
    available = blocked_by_grants = pending_review = blocker_review = warning_review = broken = 0
    for s in skills:
        stale = s.review.is_stale_for(s.content_hash)
        gate = skill_review_gate(s.review.status, stale=stale, findings=s.review.findings)
        if s.identity_collision:
            # Readiness probes include lifecycle/dependency state. A collision
            # has no unique lifecycle identity, so its UI projection must stay
            # topology-only just like discovery.
            grants_usable = True
            runnable = False
            conflict = None
            readiness = None
        else:
            readiness = skill_readiness_for_execution(drive_root, s, skills=skills)
            grant_status = readiness.grant_status or grant_status_for_skill(drive_root, s)
            grants_usable = grant_status.get("usable", True)
            runnable = s.available_for_execution and readiness.ready
            conflict = readiness.conflict or None
        available += int(runnable)
        blocked_by_grants += int(s.available_for_execution and not grants_usable)
        pending_review += int(
            s.review.status in (_REVIEW_STATUS_PENDING, "")
            or (review_status_allows_execution(s.review.status) and stale)
        )
        blocker_review += int(s.review.status == _REVIEW_STATUS_FAIL)
        warning_review += int(s.review.status == _REVIEW_STATUS_ADVISORY)
        broken += int(bool(s.load_error))
        rows.append({
            "name": s.name,
            "description": s.manifest.description,
            "when_to_use": s.manifest.when_to_use,
            # CPL-7: optional Model Experience prose travels to every
            # model-visible skill surface (list_skills JSON, context section).
            "model_experience": s.manifest.model_experience,
            "type": s.manifest.type,
            "version": s.manifest.version,
            "enabled": s.enabled,
            "review_status": s.review.status,
            "review_stale": stale,
            "review_gate": gate,
            "executable_review": gate["executable_review"],
            "available_for_execution": runnable,
            "runnable_via_skill_exec": s.available_for_execution,
            "tool_surfaces": tool_surfaces_by_skill.get(s.name, []),
            "static_ready": runnable,
            "blocked_by_grants": not grants_usable,
            "load_error": s.load_error,
            "source": s.source,
            "conflicts": list(s.manifest.conflicts or []),
            "conflict": conflict,
            # #447 G3: an unresolvable manual dependency must reach the CALLER,
            # not just the internal readiness object — a skill that needs
            # `brew:ffmpeg` installed by a human is otherwise reported ready
            # and fails only at execution time.
            "manual_dependencies": list(getattr(readiness, "manual_dependencies", []) or []),
        })
        # Extension liveness is a different fact from script executability: the
        # same producer /api/extensions reads, so the catalogue and the HTTP
        # surface cannot disagree. `skills=` is mandatory — without it the state
        # helper re-walks the whole skills tree once per row.
        live: Dict[str, Any] = {}
        if s.manifest.is_extension() and not s.identity_collision:
            try:
                from ouroboros.extension_loader import runtime_state_for_loaded_skill

                live = runtime_state_for_loaded_skill(s, drive_root, skills=skills)
            except Exception:
                log.debug("extension liveness projection failed for %s", s.name, exc_info=True)
                live = {}
        rows[-1].update({
            "desired_live": bool(live.get("desired_live")),
            "live_loaded": bool(live.get("live_loaded")),
            "live_reason": str(live.get("reason") or "not_extension"),
            "process": str(live.get("process") or ""),
        })
        if live.get("load_error"):
            rows[-1]["load_error"] = str(live.get("load_error"))
    return {
        "count": len(skills),
        "runtime_mode": get_runtime_mode(),
        "available": available,
        "blocked_by_grants": blocked_by_grants,
        "pending_review": pending_review,
        "blocker_review": blocker_review,
        "warning_review": warning_review,
        "broken": broken,
        "skills": rows,
    }


__all__ = [
    "AutoGrantOutcome", "LoadedSkill", "HASH_EXEMPT_CONTROL_FILENAMES",
    "SkillReviewState", "auto_grant_if_enabled",
    "VALID_REVIEW_STATUSES", "compute_content_hash", "reduce_skill_content_hash", "discover_skills",
    "discover_selected_skill_candidates", "find_skill",
    "enabled_skill_conflicts", "skill_conflict_status",
    "grant_status_for_skill", "is_self_authored_skill_dir", "list_available_for_execution",
    "load_enabled", "load_review_state", "load_skill_grants", "load_skill",
    "requested_core_setting_keys", "review_status_allows_execution", "skill_review_gate",
    "save_enabled", "save_review_state", "save_skill_grants", "skill_state_dir",
    "summarize_skills",
]
