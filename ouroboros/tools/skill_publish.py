"""Publish one immutable reviewed skill snapshot through a GitHub pull request."""

from __future__ import annotations

import base64
import json
import os
import pathlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ouroboros.betterleaks_runtime import resolve_betterleaks
from ouroboros.config import (
    SKILL_SOURCE_CLAWHUB,
    SKILL_SOURCE_EXTERNAL,
    SKILL_SOURCE_NATIVE,
    SKILL_SOURCE_OUROBOROSHUB,
    SKILL_SOURCE_SELF_AUTHORED,
    SKILL_SOURCE_USER_REPO,
    get_light_model,
    get_ouroboroshub_catalog_url,
)
from ouroboros.llm import LLMClient
from ouroboros.marketplace.provenance import write_publication_record
from ouroboros.skill_loader import SkillPayloadUnreadable, _sanitize_skill_name
from ouroboros.skill_publish_eligibility import PUBLISHABLE_STATUSES
from ouroboros.skill_publish_github import (
    SkillPublishGitHubError,
    commit_payload,
    create_pr_receipt,
    ensure_branch,
    fetch_upstream_catalog,
    github_login,
    prepare_publish_repository,
)
from ouroboros.skill_publish_result import (
    SKILL_PUBLISH_STAGES,
    SkillPublishDestinationError,
    parse_skill_publish_destination,
    serialize_skill_publish_result,
)
from ouroboros.skill_publish_scanner import (
    ScannerExecutable,
    SecretScanResult,
    scan_named_bytes,
)
from ouroboros.skill_publish_snapshot import (
    SkillPublishSnapshot,
    SkillPublishSnapshotError,
    capture_skill_publish_snapshot,
)
from ouroboros.skill_review_status import normalize_skill_review_status
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    build_resolved_resource_binding,
    canonical_data_root,
    load_bound_skill,
)
from ouroboros.tools.github import github_token_from_env_or_settings
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import utc_now_iso

_BRANCH_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
_PROVENANCE_SLUG_MAX = 128
_PR_BODY_MODEL_TIMEOUT_SEC = 45.0
_GENERATED_H2_HEADINGS = (
    "Author Checklist",
    "Known advisory findings",
    "Secret scan attestation",
)


class _PublishFailure(RuntimeError):
    """Closed, candidate-free publication failure."""

    def __init__(self, reason_code: str, repair_hint: str, *, status: str = "blocked") -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.repair_hint = repair_hint
        self.status = status


@dataclass
class _PublishAttempt:
    """In-memory facts for one call, never a durable recovery workflow."""

    skill: str
    snapshot_hash: str = ""
    scanner: Dict[str, Any] = field(default_factory=dict)
    completed_stage: str = ""
    completed_effects: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    blocker_count: int = 0
    warning_count: int = 0
    audited_false_positive_count: int = 0

    def mark(self, stage: str, **facts: str) -> None:
        if stage not in SKILL_PUBLISH_STAGES:
            raise ValueError("unknown skill publish stage")
        self.completed_stage = stage
        effect: Dict[str, Any] = {"stage": stage}
        effect.update({key: value for key, value in facts.items() if value})
        self.completed_effects.append(effect)

    def observe_scan(self, result: SecretScanResult, *, include_findings: bool = True) -> None:
        facts = {
            "engine": result.engine,
            "version": result.version,
            "ruleset_sha256": result.ruleset_sha256,
        }
        for key, value in facts.items():
            if value:
                self.scanner[key] = value
        if not include_findings:
            return
        self.findings.extend(asdict(item) for item in result.findings)
        self.blocker_count += result.blocker_count
        self.warning_count += result.warning_count
        self.audited_false_positive_count += result.audited_false_positive_count

    def result(
        self,
        *,
        ok: bool,
        status: str,
        reason_code: str,
        repair_hint: str = "",
        receipt: Mapping[str, Any] | None = None,
        expected_repository: str = "",
        extra_fields: Mapping[str, Any] | None = None,
    ) -> str:
        return serialize_skill_publish_result(
            ok=ok,
            status=status,
            reason_code=reason_code,
            skill=self.skill,
            snapshot_hash=self.snapshot_hash,
            scanner=self.scanner,
            completed_stage=self.completed_stage,
            completed_effects=self.completed_effects,
            findings=self.findings,
            blocker_count=self.blocker_count,
            warning_count=self.warning_count,
            audited_false_positive_count=self.audited_false_positive_count,
            repair_hint=repair_hint,
            receipt=receipt,
            expected_repository=expected_repository,
            extra_fields=extra_fields,
        )


def _safe_result_skill(raw: object) -> str:
    safe = _sanitize_skill_name(str(raw or ""))
    return safe if safe and safe != "_unnamed" else "unknown"


def _validate_local_skill(
    ctx: ToolContext,
    skill: str,
    binding: ResolvedResourceBinding | None = None,
):
    safe = _sanitize_skill_name(skill)
    if not safe or safe == "_unnamed":
        raise _PublishFailure("skill_invalid", "Choose one installed skill, then retry.")
    if not github_token_from_env_or_settings():
        raise _PublishFailure(
            "github_token_missing",
            "Add GitHub authority in Settings -> Secrets, then retry.",
        )
    try:
        binding = binding or build_resolved_resource_binding(
            ctx,
            root="skill_payload",
            operation="review",
            path=".",
            skill_name=safe,
        )
        loaded = load_bound_skill(binding)
    except (SkillPayloadUnreadable, OSError, RuntimeError, ValueError) as exc:
        raise _PublishFailure(
            "skill_load_failed",
            "Repair the installed skill payload, then retry.",
        ) from exc
    if loaded is None:
        raise _PublishFailure("skill_not_found", "Choose one installed skill, then retry.")
    allowed_sources = {
        SKILL_SOURCE_EXTERNAL,
        SKILL_SOURCE_SELF_AUTHORED,
        SKILL_SOURCE_USER_REPO,
        SKILL_SOURCE_OUROBOROSHUB,
        SKILL_SOURCE_CLAWHUB,
    }
    if loaded.source == SKILL_SOURCE_NATIVE and not (loaded.skill_dir / ".seed-origin").is_file():
        allowed_sources.add(SKILL_SOURCE_NATIVE)
    if loaded.source not in allowed_sources:
        raise _PublishFailure(
            "skill_source_unsupported",
            "Copy or adapt the skill into a publishable user-managed source, then retry.",
        )
    if loaded.load_error:
        raise _PublishFailure(
            "skill_load_failed",
            "Repair the installed skill payload, then retry.",
        )
    if normalize_skill_review_status(loaded.review.status) not in PUBLISHABLE_STATUSES:
        raise _PublishFailure(
            "review_not_publishable",
            "Resolve review blockers or pending review work, then retry.",
        )
    if str(getattr(loaded.review, "review_profile", "") or "") == "owner_attested":
        raise _PublishFailure(
            "review_owner_attested",
            "Run the full skill review for public publication, then retry.",
        )
    return safe, loaded


def _payload_files(snapshot: SkillPublishSnapshot) -> List[Dict[str, Any]]:
    return [
        {
            "path": item.path,
            "sha256": item.sha256,
            "size": item.byte_count,
            "content_b64": base64.b64encode(item.content).decode("ascii"),
        }
        for item in snapshot.public_files
    ]


def _catalog_entry(
    skill: str,
    snapshot: SkillPublishSnapshot,
    payload_files: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    manifest = snapshot.manifest
    entry: Dict[str, Any] = {
        "slug": skill,
        "name": manifest.name or skill,
        "description": manifest.description,
        "version": manifest.version,
        "type": manifest.skill_type,
        "files": [
            {
                "path": str(item["path"]),
                "sha256": str(item["sha256"]),
                "size": int(item["size"]),
            }
            for item in payload_files
        ],
    }
    install_specs = manifest.install_specs()
    if install_specs:
        entry["install_specs"] = install_specs
    if manifest.when_to_use:
        entry["when_to_use"] = manifest.when_to_use
    return entry


def _catalog_file_paths(entry: Mapping[str, Any]) -> set[str]:
    files = entry.get("files")
    if files is None:
        return set()
    if not isinstance(files, list):
        raise _PublishFailure(
            "upstream_catalog_invalid",
            "Repair the upstream Hub catalog, then retry.",
        )
    paths: set[str] = set()
    for item in files:
        raw = item.get("path") if isinstance(item, Mapping) else None
        path = str(raw or "")
        pure = pathlib.PurePosixPath(path)
        if (
            not path
            or pure.is_absolute()
            or path != pure.as_posix()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or path in paths
        ):
            raise _PublishFailure(
                "upstream_catalog_invalid",
                "Repair the upstream Hub catalog, then retry.",
            )
        paths.add(path)
    return paths


def _update_catalog(
    catalog: Dict[str, Any],
    entry: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Tuple[str, ...]]:
    skills = catalog.get("skills")
    if not isinstance(skills, list):
        raise _PublishFailure(
            "upstream_catalog_invalid",
            "Repair the upstream Hub catalog, then retry.",
        )
    slug = str(entry.get("slug") or "")
    matching_indexes = [
        index for index, item in enumerate(skills) if isinstance(item, dict) and item.get("slug") == slug
    ]
    if len(matching_indexes) > 1:
        raise _PublishFailure(
            "upstream_catalog_invalid",
            "Repair the upstream Hub catalog, then retry.",
        )
    existing_index = matching_indexes[0] if matching_indexes else None
    if existing_index is None:
        mode = "add"
        obsolete_paths: Tuple[str, ...] = ()
        skills.append(entry)
    else:
        existing = skills[existing_index]
        if str(existing.get("version") or "") == str(entry.get("version") or ""):
            raise _PublishFailure(
                "catalog_version_exists",
                "Bump the skill version, run a fresh review, then retry.",
            )
        mode = "update"
        obsolete_paths = tuple(sorted(_catalog_file_paths(existing) - _catalog_file_paths(entry)))
        skills[existing_index] = entry
    skills.sort(key=lambda item: str(item.get("slug") or "") if isinstance(item, dict) else "")
    catalog["skills"] = skills
    return mode, catalog, obsolete_paths


def _captured_control_json(snapshot: SkillPublishSnapshot, filename: str) -> Dict[str, Any] | None:
    item = snapshot.file(filename)
    if item is None:
        return None
    try:
        value = json.loads(item.content.decode("utf-8"))
    except (UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _provenance_hint(snapshot: SkillPublishSnapshot) -> str:
    if snapshot.source == SKILL_SOURCE_OUROBOROSHUB:
        data = _captured_control_json(snapshot, ".ouroboroshub.json")
        label = "OuroborosHub"
        slug_key = "slug"
    elif snapshot.source == SKILL_SOURCE_CLAWHUB:
        data = _captured_control_json(snapshot, ".clawhub.json")
        label = "ClawHub"
        slug_key = "clawhub_slug"
    else:
        return ""
    if data is None:
        return ""
    original = str(data.get(slug_key) or data.get("slug") or "").strip()
    safe = "".join(ch for ch in original if 0x20 <= ord(ch) != 0x7F)
    safe = safe.replace("`", "").strip()
    if not safe:
        return ""
    if len(safe) > _PROVENANCE_SLUG_MAX:
        safe = safe[:_PROVENANCE_SLUG_MAX] + "…"
    return f"## Provenance\nLocally installed from {label} as `{safe}`. This PR submits a locally adapted version.\n"


def _advisory_findings_section(review: Any) -> str:
    """Render every bounded non-blocking FAIL row from the local review."""
    rows: List[str] = []
    for finding in getattr(review, "findings", None) or []:
        if not isinstance(finding, dict):
            continue
        if str(finding.get("verdict") or "").upper() != "FAIL":
            continue
        severity = " ".join(str(finding.get("severity") or "").lower().split()) or "advisory"
        item = " ".join(str(finding.get("item") or "?").replace("`", "").split()) or "?"
        reason = " ".join(str(finding.get("reason") or "").split())
        if len(reason) > 500:
            reason = reason[:500] + "…"
        rows.append(f"- `{item}` ({severity}): {reason}" if reason else f"- `{item}` ({severity})")
    unique = list(dict.fromkeys(rows))
    if not unique:
        return ""
    return (
        "## Known advisory findings\n"
        "Non-blocking FAIL findings from the local skill review:\n" + "\n".join(unique) + "\n"
    )


def _fence_marker(line: str) -> Tuple[str, int, str] | None:
    indent = len(line) - len(line.lstrip(" "))
    if indent > 3:
        return None
    stripped = line[indent:]
    if not stripped or stripped[0] not in {"`", "~"}:
        return None
    char = stripped[0]
    run = len(stripped) - len(stripped.lstrip(char))
    if run < 3:
        return None
    return char, run, stripped[run:]


def _h2_heading(line: str) -> str | None:
    indent = len(line) - len(line.lstrip(" "))
    if indent > 3:
        return None
    stripped = line[indent:].rstrip()
    if not stripped.startswith("## "):
        return None
    heading = stripped[3:]
    return heading if heading and not heading.startswith("#") else None


def _strip_generated_h2_sections(body: str, headings: Sequence[str]) -> str:
    """Remove real target H2 sections while preserving fenced/code examples."""
    targets = frozenset(str(heading) for heading in headings)
    output: List[str] = []
    skipping = False
    fence: Tuple[str, int] | None = None
    for raw_line in body.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        marker = _fence_marker(line)
        if fence is not None:
            if not skipping:
                output.append(raw_line)
            if marker is not None and marker[0] == fence[0] and marker[1] >= fence[1] and not marker[2].strip():
                fence = None
            continue
        if marker is not None:
            fence = (marker[0], marker[1])
            if not skipping:
                output.append(raw_line)
            continue
        heading = _h2_heading(line)
        if heading is not None:
            if heading in targets:
                skipping = True
                continue
            if skipping:
                skipping = False
        if not skipping:
            output.append(raw_line)
    return "".join(output)


def _close_unterminated_fence(body: str) -> str:
    """Keep the next independently rendered Markdown component outside a fence."""
    fence: Tuple[str, int] | None = None
    for raw_line in body.splitlines():
        marker = _fence_marker(raw_line)
        if marker is None:
            continue
        if fence is None:
            fence = (marker[0], marker[1])
        elif marker[0] == fence[0] and marker[1] >= fence[1] and not marker[2].strip():
            fence = None
    if fence is None:
        return body
    separator = "" if body.endswith(("\n", "\r")) else "\n"
    return f"{body}{separator}{fence[0] * fence[1]}\n"


def _author_checklist(review: Any) -> str:
    advisory = bool(_advisory_findings_section(review))
    review_line = (
        "- Fresh review with no blockers verified locally; advisory findings are disclosed below."
        if advisory
        else "- Fresh clean review verified locally."
    )
    return (
        "## Author Checklist\n"
        f"{review_line}\n"
        "- Published bytes match the immutable reviewed snapshot.\n"
        "- The pull request is the only requested public effect.\n"
    )


def _fallback_pr_core(mode: str, skill: str, snapshot: SkillPublishSnapshot) -> str:
    manifest = snapshot.manifest
    description = manifest.description or "See the published skill files."
    return (
        "## Summary\n"
        f"- {mode.title()} `{skill}` v{manifest.version} to OuroborosHub.\n"
        f"- Type: `{manifest.skill_type}`; files: {len(snapshot.public_files)}.\n\n"
        "## What This Skill Does\n"
        f"{description}\n"
    )


def _pr_body_prompt(
    mode: str,
    skill: str,
    snapshot: SkillPublishSnapshot,
    note: str,
    provenance: str,
    review: Any,
) -> str:
    facts = {
        "mode": mode,
        "skill": skill,
        "version": snapshot.manifest.version,
        "type": snapshot.manifest.skill_type,
        "description": snapshot.manifest.description[:2000],
        "when_to_use": snapshot.manifest.when_to_use[:2000],
        "file_count": len(snapshot.public_files),
        "files": [item.path[:240] for item in snapshot.public_files[:50]],
        "author_note": note[:2000],
        "provenance": provenance[:1000],
        "review_status": normalize_skill_review_status(str(getattr(review, "status", "") or "")),
    }
    return (
        "Write concise Markdown for only these sections: Summary and What This Skill Does. "
        "Use only the structured facts below. Do not invent claims or reproduce Author "
        "Checklist, Known advisory findings, Secret scan attestation, Note, or Provenance "
        "sections; the host renders those.\n\n"
        + json.dumps(facts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _record_llm_usage(ctx: ToolContext, model: str, usage: Mapping[str, Any]) -> None:
    if not usage:
        return
    raw_cost = usage.get("cost")
    ctx.pending_events.append(
        {
            "type": "llm_usage",
            "task_id": str(getattr(ctx, "task_id", "") or ""),
            "provider": "skill_publish",
            "model": model,
            "usage": dict(usage),
            "cost": float(raw_cost) if raw_cost is not None else None,
            "source": "submit_skill_to_hub",
            "ts": utc_now_iso(),
            "category": "task",
        }
    )


def _scanner_executable(ctx: ToolContext) -> ScannerExecutable:
    runtime = resolve_betterleaks(data_root=canonical_data_root(ctx))
    path = pathlib.Path(runtime.binary_path) if runtime.binary_path else None
    status = runtime.status if runtime.status in {"ready", "missing", "corrupt"} else "corrupt"
    return ScannerExecutable(path=path, identity=runtime.binary_sha256, status=status)


def _scan(
    ctx: ToolContext,
    attempt: _PublishAttempt,
    executable: ScannerExecutable,
    named_bytes: Mapping[str, bytes],
    *,
    honor_inline_allowances: bool,
) -> SecretScanResult:
    result = scan_named_bytes(
        named_bytes,
        executable=executable,
        drive_root=pathlib.Path(ctx.drive_root),
        scope="task",
        owner_task_id=str(ctx.task_id or ""),
        honor_inline_allowances=honor_inline_allowances,
    )
    attempt.observe_scan(result, include_findings=False)
    if result.status == "scanner_error":
        raise _PublishFailure(
            result.reason_code or "scanner_report_invalid",
            result.repair_hint or "Repair the Betterleaks runtime, then retry.",
            status="repair_needed",
        )
    return result


def _select_optional_pr_core(
    ctx: ToolContext,
    attempt: _PublishAttempt,
    executable: ScannerExecutable,
    *,
    mode: str,
    skill: str,
    snapshot: SkillPublishSnapshot,
    note: str,
    provenance: str,
    review: Any,
) -> str:
    fallback = _fallback_pr_core(mode, skill, snapshot)
    prompt = _pr_body_prompt(mode, skill, snapshot, note, provenance, review)
    prompt_scan = _scan(
        ctx,
        attempt,
        executable,
        {"pr-body-model-prompt.txt": prompt.encode("utf-8")},
        honor_inline_allowances=False,
    )
    if prompt_scan.blocker_count:
        return fallback
    try:
        model = get_light_model()
        response, usage = LLMClient().chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            reasoning_effort="low",
            max_tokens=8192,
            use_local=os.environ.get("USE_LOCAL_LIGHT", "").lower() in {"true", "1"},
            timeout=_PR_BODY_MODEL_TIMEOUT_SEC,
        )
        _record_llm_usage(ctx, model, usage)
        body = str(response.get("content") or "").strip()
    except Exception:
        ctx.emit_progress_fn("PR body formatter unavailable; using deterministic fallback.")
        return fallback
    if not body:
        return fallback
    body_scan = _scan(
        ctx,
        attempt,
        executable,
        {"optional-pr-body.md": body.encode("utf-8")},
        honor_inline_allowances=False,
    )
    return fallback if body_scan.blocker_count else body


def _scan_public_derived(
    ctx: ToolContext,
    attempt: _PublishAttempt,
    executable: ScannerExecutable,
    named_bytes: Mapping[str, bytes],
) -> SecretScanResult:
    result = _scan(
        ctx,
        attempt,
        executable,
        named_bytes,
        honor_inline_allowances=False,
    )
    attempt.observe_scan(result)
    if result.blocker_count:
        raise _PublishFailure(
            "secret_blocked",
            "Remove or rotate the high-confidence candidate, then retry.",
        )
    return result


def _render_pr_body(
    ctx: ToolContext,
    attempt: _PublishAttempt,
    executable: ScannerExecutable,
    *,
    mode: str,
    skill: str,
    snapshot: SkillPublishSnapshot,
    note: str,
    provenance: str,
    review: Any,
) -> str:
    core = _select_optional_pr_core(
        ctx,
        attempt,
        executable,
        mode=mode,
        skill=skill,
        snapshot=snapshot,
        note=note,
        provenance=provenance,
        review=review,
    )
    prefix_parts = []
    if provenance:
        prefix_parts.append(_close_unterminated_fence(provenance.strip()).rstrip())
    if note.strip():
        prefix_parts.append(_close_unterminated_fence(f"## Note\n{note.strip()}").rstrip())
    prefix_parts.append(_close_unterminated_fence(core.strip()).rstrip())
    selected = "\n\n".join(prefix_parts) + "\n"
    selected = _strip_generated_h2_sections(selected, _GENERATED_H2_HEADINGS).rstrip()
    host_sections = [_author_checklist(review).strip()]
    advisory = _advisory_findings_section(review)
    if advisory:
        host_sections.append(advisory.strip())
    body_without_attestation = selected + "\n\n" + "\n\n".join(host_sections) + "\n"
    _scan_public_derived(
        ctx,
        attempt,
        executable,
        {"pull-request-body.md": body_without_attestation.encode("utf-8")},
    )
    attestation = (
        "## Secret scan attestation\n"
        f"- Engine: {attempt.scanner.get('engine', '')} "
        f"{attempt.scanner.get('version', '')}\n"
        f"- Ruleset SHA-256: {attempt.scanner.get('ruleset_sha256', '')}\n"
        f"- blockers=0; warnings={attempt.warning_count}; "
        f"audited={attempt.audited_false_positive_count}.\n"
    )
    return body_without_attestation.rstrip() + "\n\n" + attestation


def _record_publication_receipt(
    ctx: ToolContext, skill: str, version: str, receipt: Mapping[str, Any],
) -> Tuple[bool, str]:
    """Best-effort durable publication receipt beside the skill's review state.

    Never raises; a write problem becomes ``(False, diagnostic)`` in the result
    JSON instead of converting a real PR success into a failure. The mapping
    mirrors the validated PR receipt (frozen contract).
    """
    try:
        write_publication_record(canonical_data_root(ctx), skill, {
            "slug": skill,
            "version": str(version),
            "content_hash": str(receipt.get("snapshot_hash") or "").lower(),
            "repository": str(receipt.get("repository") or ""),
            "pr_number": receipt.get("number"),
            "pr_url": str(receipt.get("url") or ""),
            "published_at": utc_now_iso(),
        })
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:200]


def _annotate_publication_recorded(serialized: str, recorded: bool, record_error: str) -> str:
    """Add the receipt-write outcome beside the PR receipt; never raises."""
    try:
        envelope = json.loads(serialized)
        envelope["publication_recorded"] = recorded
        if not recorded:
            envelope["publication_record_error"] = record_error
        return json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return serialized


def _submit_skill_to_hub(
    ctx: ToolContext,
    skill: str,
    note: str = "",
    confirm_public_submission: bool = False,
    _resolved_binding: ResolvedResourceBinding | None = None,
) -> str:
    attempt = _PublishAttempt(skill=_safe_result_skill(skill))
    expected_repository = ""
    try:
        if not confirm_public_submission:
            raise _PublishFailure(
                "confirmation_required",
                "Confirm this public submission, then retry.",
            )
        safe_skill, loaded = _validate_local_skill(ctx, skill, _resolved_binding)
        attempt.skill = safe_skill
        attempt.mark("local_validation")
        try:
            owner, repo, base_branch = parse_skill_publish_destination(get_ouroboroshub_catalog_url())
        except SkillPublishDestinationError as exc:
            raise _PublishFailure(
                exc.reason_code,
                "Repair the configured OuroborosHub catalog URL, then retry.",
            ) from exc
        expected_repository = f"{owner}/{repo}"
        snapshot = capture_skill_publish_snapshot(loaded)
        attempt.snapshot_hash = snapshot.content_hash
        attempt.mark("snapshot_captured")
        if not snapshot.manifest.version.strip():
            raise _PublishFailure(
                "manifest_version_missing",
                "Add a skill manifest version, run a fresh review, then retry.",
            )
        executable = _scanner_executable(ctx)
        payload_scan = _scan(
            ctx,
            attempt,
            executable,
            {item.path: item.content for item in snapshot.public_files},
            honor_inline_allowances=True,
        )
        attempt.observe_scan(payload_scan)
        if payload_scan.blocker_count:
            raise _PublishFailure(
                "secret_blocked",
                "Remove or rotate the high-confidence candidate, then retry.",
            )
        provenance = _provenance_hint(snapshot)
        preliminary: Dict[str, bytes] = {}
        if note.strip():
            preliminary["author-note.md"] = note.strip().encode("utf-8")
        if provenance:
            preliminary["provenance.md"] = provenance.encode("utf-8")
        if preliminary:
            preliminary_scan = _scan(
                ctx,
                attempt,
                executable,
                preliminary,
                honor_inline_allowances=False,
            )
            if preliminary_scan.blocker_count:
                attempt.observe_scan(preliminary_scan)
                raise _PublishFailure(
                    "secret_blocked",
                    "Remove or rotate the high-confidence candidate, then retry.",
                )
        attempt.mark("local_preflight")
        login = github_login(ctx)
        catalog, base_sha = fetch_upstream_catalog(ctx, owner, repo, base_branch)
        payload_files = _payload_files(snapshot)
        entry = _catalog_entry(safe_skill, snapshot, payload_files)
        mode, updated_catalog, obsolete_paths = _update_catalog(catalog, entry)
        try:
            catalog_bytes = json.dumps(
                updated_catalog,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise _PublishFailure(
                "upstream_catalog_invalid",
                "Repair the upstream Hub catalog, then retry.",
            ) from exc
        title = f"{mode.title()} skill: {safe_skill} v{snapshot.manifest.version}"
        branch_version = _BRANCH_SEGMENT_RE.sub("-", snapshot.manifest.version).strip("-") or "unknown"
        branch = f"submit/{safe_skill}-v{branch_version}"
        attempt.mark(
            "upstream_read",
            repository=expected_repository,
            actor=login,
            base_sha=base_sha,
            mode=mode,
        )
        _scan_public_derived(
            ctx,
            attempt,
            executable,
            {
                "catalog.json": catalog_bytes,
                "pull-request-title.txt": title.encode("utf-8"),
            },
        )
        body = _render_pr_body(
            ctx,
            attempt,
            executable,
            mode=mode,
            skill=safe_skill,
            snapshot=snapshot,
            note=note,
            provenance=provenance,
            review=loaded.review,
        )
        prepare_publish_repository(
            ctx,
            attempt,
            owner=owner,
            repo=repo,
            base_branch=base_branch,
            login=login,
        )
        branch_sha = ensure_branch(ctx, login, repo, branch, base_sha)
        attempt.mark(
            "branch_created",
            repository=f"{login}/{repo}",
            actor=login,
            branch=branch,
            base_sha=branch_sha,
        )
        additions = [
            {
                "path": f"skills/{safe_skill}/{item['path']}",
                "contents": str(item["content_b64"]),
            }
            for item in payload_files
        ]
        additions.append(
            {
                "path": "catalog.json",
                "contents": base64.b64encode(catalog_bytes).decode("ascii"),
            }
        )
        deletions = [{"path": f"skills/{safe_skill}/{path}"} for path in obsolete_paths]
        commit_sha, commit_url = commit_payload(
            ctx,
            login,
            repo,
            branch,
            branch_sha,
            title,
            additions,
            deletions,
        )
        attempt.mark(
            "commit_created",
            repository=f"{login}/{repo}",
            actor=login,
            branch=branch,
            commit_sha=commit_sha,
            commit_url=commit_url,
        )
        receipt = create_pr_receipt(
            ctx,
            attempt,
            owner=owner,
            repo=repo,
            base_branch=base_branch,
            login=login,
            branch=branch,
            title=title,
            body=body,
            commit_sha=commit_sha,
        )
        if receipt is None:
            raise _PublishFailure(
                "pr_open_indeterminate",
                "Inspect the recorded branch and commit before deciding whether to retry.",
                status="partial",
            )
        attempt.mark(
            "pr_opened",
            repository=expected_repository,
            actor=login,
            branch=branch,
            commit_sha=commit_sha,
        )
        serialized = attempt.result(
            ok=True,
            status="pr_opened",
            reason_code="",
            receipt=receipt,
            expected_repository=expected_repository,
        )
        # State-plane receipt write happens only AFTER serializer receipt
        # validation — and maps from the VALIDATED envelope receipt (canonical
        # repository spelling, normalized fields), never the raw input.
        try:
            validated_receipt = json.loads(serialized).get("receipt") or receipt
        except Exception:
            validated_receipt = receipt
        recorded, record_error = _record_publication_receipt(
            ctx, safe_skill, snapshot.manifest.version, validated_receipt,
        )
        try:
            # Re-serialize with the annotation INSIDE the cap loop so the added
            # fields participate in findings trimming (a post-hoc append could
            # push a just-under-cap envelope past the transport limit and get
            # head-truncated into invalid JSON).
            extra: Dict[str, Any] = {"publication_recorded": recorded}
            if not recorded:
                extra["publication_record_error"] = record_error
            return attempt.result(
                ok=True,
                status="pr_opened",
                reason_code="",
                receipt=receipt,
                expected_repository=expected_repository,
                extra_fields=extra,
            )
        except Exception:
            # Never lose a real PR success: fall back to the bounded post-hoc
            # annotation (which itself never raises and never exceeds the cap
            # by more than the annotation bytes).
            return _annotate_publication_recorded(serialized, recorded, record_error)
    except SkillPublishSnapshotError as exc:
        return attempt.result(
            ok=False,
            status="blocked",
            reason_code=exc.reason_code,
            repair_hint="Repair or re-review the captured skill payload, then retry.",
            expected_repository=expected_repository,
        )
    except (_PublishFailure, SkillPublishGitHubError) as exc:
        return attempt.result(
            ok=False,
            status=exc.status,
            reason_code=exc.reason_code,
            repair_hint=exc.repair_hint,
            expected_repository=expected_repository,
        )
    except Exception:
        return attempt.result(
            ok=False,
            status="error",
            reason_code="unexpected_publish_error",
            repair_hint="Inspect the safe publication facts, repair the cause, then retry.",
            expected_repository=expected_repository,
        )


_PUBLISH_SCHEMA = {
    "name": "submit_skill_to_hub",
    "description": (
        "Publish one immutable reviewed skill snapshot to OuroborosHub by "
        "opening a GitHub pull request. A failed result is repair evidence "
        "for the next agent turn; success contains a validated PR receipt."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "Installed skill name (slug).",
            },
            "note": {
                "type": "string",
                "default": "",
                "description": "Optional public author note for the pull request.",
            },
            "confirm_public_submission": {
                "type": "boolean",
                "description": (
                    "Must be true: confirms the human approved this public "
                    "OuroborosHub submission."
                ),
            },
        },
        "required": ["skill", "confirm_public_submission"],
    },
}


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="submit_skill_to_hub",  # literal: the tool-policy AST scan keys on it
            schema=_PUBLISH_SCHEMA,
            handler=_submit_skill_to_hub,
            is_code_tool=False,
            timeout_sec=180,
        )
    ]


__all__ = ["get_tools"]
