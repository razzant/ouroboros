"""Shared skill-payload path resolution policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional
from contextlib import suppress

from ouroboros.contracts.task_constraint import TaskConstraint, normalize_task_constraint
from ouroboros.utils import safe_relpath

SKILL_PAYLOAD_BUCKETS = frozenset({
    "external",
    "clawhub",
    "ouroboroshub",
})

SKILL_PAYLOAD_ALL_BUCKETS = frozenset({
    "native",
    *SKILL_PAYLOAD_BUCKETS,
})

SKILL_PAYLOAD_CONTROL_FILENAMES = frozenset({
    ".clawhub.json",
    ".ouroboroshub.json",
    ".self_authored.json",
    ".seed-origin",
    "skill.openclaw.md",
})

SKILL_PAYLOAD_CONTROL_DIRNAMES = frozenset({
    ".ouroboros_env",
    "node_modules",
    "__pycache__",
})

PRESENCE_PROFILE_STATE_FILENAME = "presence_profile_state.json"
PRESENCE_PROFILE_STATE_STEM = "presence_profile_state"

SKILL_OWNER_STATE_FILENAMES = frozenset({
    "enabled.json",
    "grants.json",
    "review.json",
    "review_job.json",
    "review_history.jsonl",
    "accepted_rebuttals.json",
    "clawhub.json",
    # OuroborosHub publication receipt: forging it would fake a "published by
    # this owner" fact that hub sync verdicts and adopt confirmations trust.
    "ouroboroshub.json",
    "self_authored.json",
    "auth_token.json",
    # Owner/lifecycle state: forged deps.json would bypass dependency gates.
    "deps.json",
    # Durable health vector (live->broken regression memory); forging it would
    # mask a regression or fake recovery.
    "health.json",
    # Owner attestation (C1, v6.39): an owner-only "skip the LLM review for my own
    # skill" marker. The agent must NEVER forge it — that would self-bypass the immune
    # system's expensive review. Owner-issued via the owner-only endpoint only.
    "owner_attestation.json",
    # Host-owned presence selections and local runtime overrides. A behavior
    # skill may read its resolved projection, but payload/runtime writes must
    # not be able to forge the owner's confirmed capability mapping.
    PRESENCE_PROFILE_STATE_FILENAME,
    # Uninstall tombstone (CPL4-C11): forging it would let the agent mark a
    # live skill payload-gone and have the startup sweep clear its owner state.
    "uninstalled.json",
})

SKILL_OWNER_STATE_STEMS = (
    "grants",
    "review",
    "review_job",
    "review_history",
    "accepted_rebuttals",
    "enabled",
    "clawhub",
    "ouroboroshub",
    "deps",
    "self_authored",
    "auth_token",
    "health",
    "owner_attestation",
    PRESENCE_PROFILE_STATE_STEM,
    "uninstalled",
)


class SkillPayloadPathError(ValueError):
    """Raised when a path cannot be confined to a skill payload."""


@dataclass(frozen=True)
class SkillPayloadTarget:
    bucket: str
    skill: str
    payload_root: Path
    target_path: Path
    rel_path: str
    control_plane: bool = False


@dataclass(frozen=True)
class PayloadShortFormDecision:
    """Resolution decision for optional ``bucket`` + ``skill_name`` edit args."""

    constraint: Optional[TaskConstraint] = None
    error: str = ""
    ignored_reason: str = ""


_OPTIONAL_ARG_SENTINELS = frozenset({
    "__omit__",
    "<omit>",
    "__none__",
    "<none>",
    "null",
    "none",
    "undefined",
})

_DATA_ROOT_PREFIXES = frozenset({
    "archive",
    "logs",
    "memory",
    "skills",
    "state",
    "task_results",
    "uploads",
})

_DATA_ROOT_FILENAMES = frozenset({
    "settings.json",
})


def _clean_optional_short_form_arg(value: str) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in _OPTIONAL_ARG_SENTINELS else text


def _clean_data_rel(raw: str) -> str:
    norm = str(raw or "").replace("\\", "/").strip().lstrip("/")
    if norm.startswith("data/"):
        norm = norm[len("data/"):]
    return norm


def _constraint_payload_root(constraint: Optional[TaskConstraint]) -> str:
    tc = normalize_task_constraint(constraint)
    if not tc or tc.mode != "skill_repair" or not tc.payload_root:
        return ""
    return _clean_data_rel(tc.payload_root)


def _sanitize_skill_name(name: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name or "").strip()
    )
    cleaned = cleaned.strip("._")
    return (cleaned or "_unnamed")[:64]


def _rel_from_raw(drive: Path, raw_path: str) -> tuple[str, bool]:
    raw = str(raw_path or "").strip()
    if not raw:
        return "", False
    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            rel = candidate.resolve(strict=False).relative_to(drive)
        except ValueError as exc:
            raise SkillPayloadPathError("absolute path is outside data root") from exc
        return rel.as_posix(), True
    return _clean_data_rel(raw), False


def resolve_constrained_payload_path(
    drive_root: Path,
    constraint: TaskConstraint,
    path_text: str,
) -> Path:
    """Resolve a skill_repair path under its constrained payload root.

    This preserves the legacy ``task_constraint.resolve_payload_path``
    behavior and messages while making the payload-root bucket policy shared.
    """
    drive = Path(drive_root).resolve(strict=False)
    payload_root = safe_relpath(constraint.payload_root)
    payload_parts = PurePosixPath(payload_root).parts
    if (
        len(payload_parts) < 3
        or payload_parts[0] != "skills"
        or payload_parts[1] not in SKILL_PAYLOAD_BUCKETS
    ):
        raise ValueError(
            "Repair payload root must be data/skills/{external,clawhub,ouroboroshub}/<skill>"
        )
    if constraint.skill_name and payload_parts[2] != constraint.skill_name:
        raise ValueError("Repair payload root does not match constrained skill name")
    base = (drive / payload_root).resolve(strict=False)
    raw = str(path_text or "").replace("\\", "/").strip().lstrip("/")
    if raw.startswith("data/"):
        raw = raw[len("data/"):]
    if raw.startswith("skills/") and raw != payload_root and not raw.startswith(payload_root + "/"):
        raise ValueError("Path points at a different skill payload")
    if raw == payload_root:
        raw = ""
    elif raw.startswith(payload_root + "/"):
        raw = raw[len(payload_root) + 1:]
    rel = safe_relpath(raw or ".")
    target = (base / rel).resolve(strict=False)
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError("Path escapes constrained skill payload") from exc
    return target


def resolve_skill_payload_target(
    drive_root: Path,
    path_text: str,
    *,
    constraint: Optional[TaskConstraint] = None,
    allow_short_relative: bool = False,
) -> SkillPayloadTarget:
    """Resolve *path_text* to a path confined inside one skill payload.

    Without a constraint, callers must pass an explicit data-skill path
    (``skills/<bucket>/<skill>/...`` or ``data/skills/...`` or an absolute
    path under ``drive_root``).  With a repair constraint and
    ``allow_short_relative=True``, short paths such as ``plugin.py`` resolve
    under the selected payload root.
    """

    drive = Path(drive_root).resolve(strict=False)
    rel, was_absolute = _rel_from_raw(drive, path_text)
    payload_root = _constraint_payload_root(constraint)
    if payload_root:
        root_parts = PurePosixPath(payload_root).parts
        if len(root_parts) < 3 or root_parts[0] != "skills" or root_parts[1] not in SKILL_PAYLOAD_BUCKETS:
            raise SkillPayloadPathError("repair payload root must be data/skills/<bucket>/<skill>")
        tc = normalize_task_constraint(constraint)
        if tc and tc.skill_name and root_parts[2] != _sanitize_skill_name(tc.skill_name):
            raise SkillPayloadPathError("repair payload root does not match constrained skill name")
        if rel in ("", ".", "./"):
            rel = payload_root
        elif rel.startswith("skills/"):
            if rel != payload_root and not rel.startswith(payload_root + "/"):
                raise SkillPayloadPathError("path points at a different skill payload")
        elif allow_short_relative and not was_absolute:
            rel = f"{payload_root}/{safe_relpath(rel or '.')}"
        else:
            raise SkillPayloadPathError("path must be explicit or payload-relative under the repair constraint")

    parts = PurePosixPath(rel).parts
    if len(parts) < 3 or parts[0] != "skills" or parts[1] not in SKILL_PAYLOAD_BUCKETS:
        raise SkillPayloadPathError("path must point inside data/skills/<bucket>/<skill>")
    if any(part in {"", ".", ".."} for part in parts):
        raise SkillPayloadPathError("path contains unsafe path segment")

    bucket, skill = parts[1], parts[2]
    payload = (drive / "skills" / bucket / skill).resolve(strict=False)
    suffix = PurePosixPath(*parts[3:]).as_posix() if len(parts) > 3 else "."
    target = (payload / safe_relpath(suffix)).resolve(strict=False)
    try:
        target.relative_to(payload)
    except ValueError as exc:
        raise SkillPayloadPathError("path escapes skill payload") from exc

    rel_inside = "." if suffix in ("", ".") else suffix
    rel_parts = [part.lower() for part in PurePosixPath(rel_inside).parts]
    control = any(
        part in SKILL_PAYLOAD_CONTROL_FILENAMES or part in SKILL_PAYLOAD_CONTROL_DIRNAMES
        for part in rel_parts
    )
    return SkillPayloadTarget(
        bucket=bucket,
        skill=skill,
        payload_root=payload,
        target_path=target,
        rel_path=rel_inside,
        control_plane=control,
    )


def is_skill_payload_path(
    drive_root: Path,
    path_text: str,
    *,
    constraint: Optional[TaskConstraint] = None,
    allow_short_relative: bool = False,
    allow_control_plane: bool = False,
) -> bool:
    try:
        target = resolve_skill_payload_target(
            drive_root,
            path_text,
            constraint=constraint,
            allow_short_relative=allow_short_relative,
        )
    except SkillPayloadPathError:
        return False
    return allow_control_plane or not target.control_plane


def is_skill_payload_control_filename(path_text: str) -> bool:
    name = PurePosixPath(str(path_text or "").replace("\\", "/")).name
    return name.lower() in SKILL_PAYLOAD_CONTROL_FILENAMES


def is_skill_owner_state_target(target: Path, data_root: Path) -> bool:
    """Return True for skill owner/review state files under data/state/skills."""
    target_path = Path(target)
    if target_path.name.lower() not in SKILL_OWNER_STATE_FILENAMES:
        return False
    root = Path(data_root).resolve(strict=False)
    for candidate in (target_path, target_path.resolve(strict=False)):
        with suppress(OSError, ValueError):
            parts = candidate.relative_to(root).parts
            if len(parts) == 4 and parts[0].lower() == "state" and parts[1].lower() == "skills":
                return True
    skills_state_root = root / "state" / "skills"
    if not skills_state_root.is_dir():
        return False
    with suppress(OSError):
        target_parent = target_path.parent.resolve(strict=False)
        for skill_state_dir in skills_state_root.iterdir():
            with suppress(OSError):
                if skill_state_dir.resolve(strict=False) == target_parent:
                    return True
    return False


def is_skill_owner_state_alias(target: Path, data_root: Path) -> bool:
    """Return True for hardlinks/aliases to owner state files."""
    target_path = Path(target)
    root = Path(data_root).resolve(strict=False)
    skills_state_root = root / "state" / "skills"
    if not target_path.exists() or not skills_state_root.is_dir():
        return False
    for owner_state_file in skills_state_root.glob("*/*"):
        try:
            if owner_state_file.name.lower() in SKILL_OWNER_STATE_FILENAMES and owner_state_file.exists() and target_path.samefile(owner_state_file):
                return True
        except OSError:
            continue
    return False


def is_skill_control_plane_path(target: Path, data_root: Path) -> bool:
    """Return True for skill owner/provenance files blocked from generic writes."""
    target_path = Path(target)
    root = Path(data_root).resolve(strict=False)
    if is_skill_owner_state_target(target_path, root):
        return True

    payload_candidates = [target_path]
    with suppress(OSError):
        payload_candidates.append(target_path.resolve(strict=False))
    for candidate in payload_candidates:
        with suppress(OSError, ValueError):
            parts = candidate.relative_to(root).parts
            if (
                len(parts) >= 4
                and parts[0].lower() == "skills"
                and parts[1].lower() in SKILL_PAYLOAD_ALL_BUCKETS
            ):
                rel_tail = [part.lower() for part in parts[3:]]
                if (
                    any(part in SKILL_PAYLOAD_CONTROL_DIRNAMES for part in rel_tail)
                    or candidate.name.lower() in SKILL_PAYLOAD_CONTROL_FILENAMES
                ):
                    return True

    # Hardlink/inode defense for benign basenames pointing at protected sidecars.
    if not target_path.exists():
        return False
    with suppress(OSError, ValueError):
        parts = target_path.resolve(strict=False).relative_to(root).parts
        if len(parts) >= 4 and parts[0].lower() == "skills" and parts[1].lower() in SKILL_PAYLOAD_ALL_BUCKETS:
            payload_root = root / parts[0] / parts[1] / parts[2]
            for protected in payload_root.iterdir():
                try:
                    if protected.name.lower() in SKILL_PAYLOAD_CONTROL_FILENAMES and protected.exists() and protected.samefile(target_path):
                        return True
                except OSError:
                    continue
    skills_root = root / "skills"
    if not skills_root.is_dir():
        return False
    for bucket in SKILL_PAYLOAD_ALL_BUCKETS:
        bucket_root = skills_root / bucket
        if not bucket_root.is_dir():
            continue
        for protected in bucket_root.glob("*/*"):
            try:
                if (
                    protected.name.lower() in SKILL_PAYLOAD_CONTROL_FILENAMES
                    and protected.exists()
                    and protected.samefile(target_path)
                ):
                    return True
            except OSError:
                continue
    return False


def synthesize_payload_constraint(
    bucket: str,
    skill_name: str,
) -> Optional[TaskConstraint]:
    """Build a skill_repair constraint for valid bucket+skill_name short form."""
    b = _clean_optional_short_form_arg(bucket)
    raw_skill_name = _clean_optional_short_form_arg(skill_name)
    s = _sanitize_skill_name(raw_skill_name)
    if not b or not s or s == "_unnamed":
        return None
    if b not in SKILL_PAYLOAD_BUCKETS:
        return None
    return TaskConstraint(
        mode="skill_repair",
        skill_name=s,
        payload_root=f"skills/{b}/{s}",
    )


def _explicit_path_kind(path_text: str, *, repo_dir: Path, drive_root: Path) -> str:
    raw = str(path_text or "").replace("\\", "/").strip()
    if raw in ("", ".", "./"):
        return ""
    drive = Path(drive_root).resolve(strict=False)
    repo = Path(repo_dir).resolve(strict=False)
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(repo)
            return "repo"
        except ValueError:
            pass
        try:
            rel = resolved.relative_to(drive).as_posix()
        except ValueError:
            return ""
        return "skill" if rel.startswith("skills/") else "data"
    raw_lstripped = raw.lstrip("/")
    raw_lstripped_lower = raw_lstripped.lower()
    if raw_lstripped_lower.startswith("data/"):
        data_rel = raw_lstripped[len("data/"):]
        return "skill" if data_rel.lower().startswith("skills/") else "data"
    rel = _clean_data_rel(raw)
    rel_lower = rel.lower()
    if rel_lower.startswith("skills/"):
        return "skill"
    parts = PurePosixPath(rel).parts
    if not parts:
        return ""
    first_lower = parts[0].lower()
    if first_lower in _DATA_ROOT_PREFIXES or first_lower in _DATA_ROOT_FILENAMES:
        return "data"
    if (repo / parts[0]).exists():
        return "repo"
    return ""


_SKILL_MANIFEST_BASENAMES = frozenset({"skill.md", "skill.json"})


def _is_skill_create_signal(path_text: str) -> bool:
    """A missing payload is a typo for an arbitrary file but a legitimate NEW skill when the write IS
    the skill manifest at the payload ROOT (``SKILL.md``/``skill.json`` — NOT ``nested/SKILL.md`` and
    NOT an absolute path) — the explicit authoring signal. Keying CREATE on the root manifest (not on
    mere directory existence or a bare basename anywhere) restores light skill creation that the
    f705b37 blanket is_dir gate regressed, while a misspelled or nested path still errors (typo guard)."""
    raw = str(path_text or "").replace("\\", "/").strip()
    if raw.startswith("./"):
        raw = raw[2:]
    return "/" not in raw and raw.lower() in _SKILL_MANIFEST_BASENAMES


def is_skill_create_typo(*, payload_root: Path, bucket: str, rel_within_payload: str) -> bool:
    """Manifest-first typo guard SSOT, shared by the bucket/skill_name short-form and the explicit
    ``runtime_data`` ``skills/<bucket>/<skill>/...`` write path. A write into a NON-existent payload
    is a typo for an arbitrary file, but a legitimate NEW skill when it IS the root manifest
    (SKILL.md/skill.json) under bucket=external. Returns True when the write must be BLOCKED (missing
    payload, not a new-external-skill manifest) so neither entry point can silently mkdir a bogus
    payload from a misspelled name; writing into an EXISTING payload is always allowed."""
    if payload_root.is_dir():
        return False
    return not (bucket == "external" and _is_skill_create_signal(rel_within_payload))


def decide_payload_short_form(
    *,
    bucket: str,
    skill_name: str,
    path_text: str,
    repo_dir: Path,
    drive_root: Path,
) -> PayloadShortFormDecision:
    """Resolve optional skill short-form args without overriding explicit paths."""
    clean_bucket = _clean_optional_short_form_arg(bucket)
    clean_skill_name = _clean_optional_short_form_arg(skill_name)
    if not clean_bucket and not clean_skill_name:
        return PayloadShortFormDecision()
    kind = _explicit_path_kind(path_text, repo_dir=repo_dir, drive_root=drive_root)
    if kind:
        return PayloadShortFormDecision(
            ignored_reason=(
                f"ignored bucket/skill_name because {path_text!r} is an explicit "
                f"{kind} path"
            )
        )
    synth = synthesize_payload_constraint(clean_bucket, clean_skill_name)
    if synth is None:
        return PayloadShortFormDecision(
            error=(
                "bucket and skill_name must be supplied together; bucket must be "
                "one of external/clawhub/ouroboroshub (native excluded); "
                "skill_name must sanitize to a non-empty slug."
            )
        )
    payload_root = (Path(drive_root) / synth.payload_root).resolve(strict=False)
    # CREATE-from-scratch is for AGENT-authored skills only: the `external` bucket. The marketplace
    # buckets (clawhub/ouroboroshub) are installed FROM the marketplace, never authored into a
    # missing payload, so a missing marketplace payload stays an error (install it, don't create).
    # SSOT with the explicit runtime_data path via is_skill_create_typo (path_text is payload-relative).
    if is_skill_create_typo(payload_root=payload_root, bucket=clean_bucket, rel_within_payload=path_text):
        return PayloadShortFormDecision(
            error=(
                f"skill payload not found: {synth.payload_root}. "
                "Use an existing skill_name; for a NEW skill write its manifest (SKILL.md/skill.json) "
                "under bucket=external; or omit bucket/skill_name for a repo/data edit."
            )
        )
    return PayloadShortFormDecision(constraint=synth)


def cross_skill_redirect_error(
    existing_tc: Optional[TaskConstraint],
    synth_tc: Optional[TaskConstraint],
) -> str:
    """Reject bucket+skill_name when it would escape an active skill_repair task."""
    if not (existing_tc and synth_tc):
        return ""
    if existing_tc.mode != "skill_repair":
        return ""
    if existing_tc.skill_name == synth_tc.skill_name:
        return ""
    return (
        f"a skill_repair task is active for {existing_tc.skill_name!r}; "
        f"cannot use bucket+skill_name args to redirect this call to "
        f"{synth_tc.skill_name!r}. Drop the bucket/skill_name args, or "
        f"finish/cancel the active repair task first."
    )


def constraint_bucket_skill(constraint: Optional[TaskConstraint]) -> tuple[str, str]:
    """Return the skill payload bucket/name implied by a repair constraint."""

    tc = normalize_task_constraint(constraint)
    if not tc or tc.mode != "skill_repair" or not tc.payload_root:
        return "", ""
    parts = PurePosixPath(_clean_data_rel(tc.payload_root)).parts
    if len(parts) >= 3 and parts[0] == "skills":
        return parts[1], parts[2]
    return "", tc.skill_name or ""


__all__ = [
    "SKILL_PAYLOAD_BUCKETS",
    "SKILL_PAYLOAD_ALL_BUCKETS",
    "SKILL_PAYLOAD_CONTROL_FILENAMES",
    "SKILL_PAYLOAD_CONTROL_DIRNAMES",
    "SKILL_OWNER_STATE_FILENAMES",
    "SKILL_OWNER_STATE_STEMS",
    "PRESENCE_PROFILE_STATE_FILENAME",
    "PRESENCE_PROFILE_STATE_STEM",
    "SkillPayloadPathError",
    "SkillPayloadTarget",
    "PayloadShortFormDecision",
    "constraint_bucket_skill",
    "decide_payload_short_form",
    "is_skill_control_plane_path",
    "is_skill_owner_state_alias",
    "is_skill_owner_state_target",
    "is_skill_payload_control_filename",
    "is_skill_payload_path",
    "resolve_constrained_payload_path",
    "resolve_skill_payload_target",
    "synthesize_payload_constraint",
    "cross_skill_redirect_error",
]
