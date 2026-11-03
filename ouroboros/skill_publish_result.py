"""Bounded skill-publication results and the narrow terminal receipt veto.

This module is deliberately pure: it owns the JSON transport shape, validates
the one authoritative GitHub pull-request receipt, and applies the
``skill_publish`` objective veto over pre-truncation trace metadata.  It does
not perform I/O, GitHub calls, scanning, retries, or task-contract mutation.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import Any, Dict, Mapping, Sequence

from ouroboros.tool_capabilities import tool_result_limit

SKILL_PUBLISH_OPERATION = "skill_publish"
SKILL_PUBLISH_TARGET_METADATA_KEY = "skill_publish_target"
SKILL_PUBLISH_STAGES = (
    "local_validation",
    "snapshot_captured",
    "local_preflight",
    "upstream_read",
    "fork_ready",
    "fork_synced",
    "branch_created",
    "commit_created",
    "pr_create_attempted",
    "pr_opened",
)

_STAGE_INDEX = {stage: index for index, stage in enumerate(SKILL_PUBLISH_STAGES)}
_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_REPOSITORY_PART_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SUCCESS_OBJECTIVE_STATUSES = frozenset({"pass", "best_effort"})
SKILL_PUBLISH_PR_NOT_CREATED = "skill_publish_pr_not_created"
SKILL_PUBLISH_PR_NOT_CREATED_DETAIL = "PR not created; publication stopped before creating a submission branch"

_FINDING_TEXT_LIMITS = {
    "path": 320,
    "detector": 128,
    "reason": 320,
}
_FINDING_CONFIDENCES = frozenset({"low", "medium", "high", "unknown"})
_FINDING_DISPOSITIONS = frozenset({"blocker", "warning", "audited_false_positive"})
_FINDING_DISPLAY_PRIORITY = {
    "blocker": 0,
    "warning": 1,
    "audited_false_positive": 2,
}
_EFFECT_TEXT_LIMITS = {
    "kind": 64,
    "repository": 160,
    "actor": 80,
    "branch": 240,
    "base_sha": 64,
    "commit_sha": 64,
    "commit_url": 360,
    "mode": 32,
}


class SkillPublishDestinationError(ValueError):
    """Closed failure for a destination this install cannot publish to (the reason is the message)."""

    def __init__(self, reason_code: str = "hub_destination_invalid") -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def parse_skill_publish_destination(catalog_url: str) -> tuple[str, str, str]:
    """Return the canonical ``(owner, repository, base_branch)`` Hub target.

    The public preflight response and the authoritative publication tool must
    bind the same configured destination. Keep that URL interpretation beside
    the existing repository/receipt normalization rather than duplicating it at
    either caller.
    """

    try:
        parsed = urllib.parse.urlsplit(str(catalog_url or "").strip())
        port = parsed.port
    except ValueError as exc:
        raise SkillPublishDestinationError("hub_destination_invalid") from exc
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").casefold() != "raw.githubusercontent.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or len(parts) < 4
        or parts[-1] != "catalog.json"
        or not _REPOSITORY_PART_RE.fullmatch(parts[0])
        or not _REPOSITORY_PART_RE.fullmatch(parts[1])
    ):
        raise SkillPublishDestinationError("hub_destination_invalid")
    return parts[0], parts[1], "/".join(parts[2:-1])


def _loads_unique(text: str) -> Any:
    """Load JSON while rejecting duplicate object keys."""

    def _object(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=_object)


def _bounded_text(value: Any, limit: int, *, required: bool = False) -> str:
    text = "".join(" " if ord(ch) < 0x20 or ord(ch) == 0x7F else ch for ch in str(value or "").strip())
    if required and not text:
        raise ValueError("required text field is empty")
    return text[:limit]


def _bounded_identifier(value: Any, *, field: str) -> str:
    text = _bounded_text(value, 65, required=True)
    if (
        len(text) > 64
        or text in {".", ".."}
        or text != text.strip("._")
        or any(not (ch.isalnum() or ch in "-_.") for ch in text)
    ):
        raise ValueError(f"invalid {field}")
    return text


def _normalized_hash(value: Any, *, required: bool = False) -> str:
    text = _bounded_text(value, 65)
    if not text and not required:
        return ""
    if not _HASH_RE.fullmatch(text):
        raise ValueError("invalid sha256 value")
    return text.lower()


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return parsed


def _normalize_repository(value: Any) -> str:
    text = _bounded_text(value, 161, required=True)
    if len(text) > 160:
        raise ValueError("repository must be owner/name")
    parts = text.split("/")
    if (
        len(parts) != 2
        or any(not part or part in {".", ".."} for part in parts)
        or any(not _REPOSITORY_PART_RE.fullmatch(part) for part in parts)
    ):
        raise ValueError("repository must be owner/name")
    return f"{parts[0]}/{parts[1]}"


def _normalize_scanner(scanner: Mapping[str, Any] | None) -> Dict[str, Any]:
    source = scanner if isinstance(scanner, Mapping) else {}
    result = {
        "engine": _bounded_text(source.get("engine"), 80),
        "version": _bounded_text(source.get("version"), 80),
        "ruleset_sha256": _normalized_hash(source.get("ruleset_sha256")),
    }
    return result


def _normalize_finding(finding: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(finding, Mapping):
        raise ValueError("findings must contain JSON objects")
    line = _nonnegative_int(finding.get("line"), field="line")
    if line <= 0:
        raise ValueError("line must be positive")
    confidence = _bounded_text(finding.get("confidence"), 32, required=True)
    if confidence not in _FINDING_CONFIDENCES:
        raise ValueError("invalid finding confidence")
    verification = _bounded_text(finding.get("verification"), 32, required=True)
    if verification != "not_attempted":
        raise ValueError("invalid finding verification")
    disposition = _bounded_text(finding.get("disposition"), 64, required=True)
    if disposition not in _FINDING_DISPOSITIONS:
        raise ValueError("invalid finding disposition")
    return {
        "path": _bounded_text(finding.get("path"), _FINDING_TEXT_LIMITS["path"], required=True),
        "line": line,
        "detector": _bounded_text(
            finding.get("detector"),
            _FINDING_TEXT_LIMITS["detector"],
            required=True,
        ),
        "confidence": confidence,
        "reason": _bounded_text(
            finding.get("reason"),
            _FINDING_TEXT_LIMITS["reason"],
            required=True,
        ),
        "verification": verification,
        "disposition": disposition,
    }


def normalize_skill_publish_findings(
    findings: Sequence[Mapping[str, Any]] | None,
) -> list[Dict[str, Any]]:
    """Return bounded safe findings in blocker-first display order."""
    if findings is None:
        return []
    if isinstance(findings, (str, bytes)) or not isinstance(findings, Sequence):
        raise ValueError("findings must be a sequence")
    normalized = [_normalize_finding(finding) for finding in findings]
    normalized.sort(
        key=lambda row: (
            _FINDING_DISPLAY_PRIORITY[row["disposition"]],
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    )
    return normalized


def _normalize_effects(
    effects: Sequence[Mapping[str, Any]] | None,
    *,
    completed_stage: str,
) -> list[Dict[str, Any]]:
    if effects is None:
        return []
    if isinstance(effects, (str, bytes)) or not isinstance(effects, Sequence):
        raise ValueError("completed_effects must be a sequence")
    max_stage = _STAGE_INDEX.get(completed_stage, -1)
    normalized: list[Dict[str, Any]] = []
    seen_stages: set[str] = set()
    for effect in effects:
        if not isinstance(effect, Mapping):
            raise ValueError("completed_effects must contain JSON objects")
        stage = _bounded_text(effect.get("stage"), 64, required=True)
        if stage not in _STAGE_INDEX or _STAGE_INDEX[stage] > max_stage:
            raise ValueError("completed effect is outside the completed stage")
        if stage in seen_stages:
            raise ValueError("completed_effects may contain at most one row per stage")
        seen_stages.add(stage)
        row: Dict[str, Any] = {"stage": stage}
        for key, limit in _EFFECT_TEXT_LIMITS.items():
            if key in effect and str(effect.get(key) or "").strip():
                row[key] = _bounded_text(effect.get(key), limit)
        normalized.append(row)
    normalized.sort(
        key=lambda row: (
            _STAGE_INDEX[row["stage"]],
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    )
    return normalized


def validate_skill_publish_receipt(
    receipt: Mapping[str, Any] | None,
    *,
    expected_repository: str,
    expected_skill: str = "",
    expected_snapshot_hash: str = "",
    expected_ruleset_sha256: str = "",
) -> Dict[str, Any] | None:
    """Validate the exact GitHub effect independently of scanner availability."""

    if not isinstance(receipt, Mapping):
        return None
    expected_keys = {
        "kind",
        "repository",
        "url",
        "number",
        "skill",
        "snapshot_hash",
        "ruleset_sha256",
    }
    if set(receipt) != expected_keys:
        return None
    try:
        repository = _normalize_repository(receipt.get("repository"))
        wanted_repository = _normalize_repository(expected_repository)
        skill = _bounded_identifier(receipt.get("skill"), field="skill")
        snapshot_hash = _normalized_hash(receipt.get("snapshot_hash"), required=True)
        ruleset_sha256 = _normalized_hash(receipt.get("ruleset_sha256"))
        number = receipt.get("number")
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            return None
        url = _bounded_text(receipt.get("url"), 501, required=True)
        if len(url) > 500:
            return None
        parsed = urllib.parse.urlsplit(url)
        try:
            port = parsed.port
        except ValueError:
            return None
        if (
            parsed.scheme != "https"
            or (parsed.hostname or "").casefold() != "github.com"
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
            or parsed.query
            or parsed.fragment
        ):
            return None
        segments = parsed.path.split("/")
        if len(segments) != 5 or segments[0] != "" or segments[3] != "pull" or not segments[4].isdigit():
            return None
        url_repository = f"{segments[1]}/{segments[2]}"
        if (
            repository.casefold() != wanted_repository.casefold()
            or url_repository.casefold() != wanted_repository.casefold()
            or int(segments[4]) != number
            or receipt.get("kind") != "github_pull_request"
        ):
            return None
        if expected_skill and skill != _bounded_identifier(expected_skill, field="skill"):
            return None
        if expected_snapshot_hash and snapshot_hash != _normalized_hash(expected_snapshot_hash, required=True):
            return None
        if expected_ruleset_sha256 and ruleset_sha256 != _normalized_hash(expected_ruleset_sha256, required=True):
            return None
    except (TypeError, ValueError):
        return None
    return {
        "kind": "github_pull_request",
        "repository": wanted_repository,
        "url": url,
        "number": number,
        "skill": skill,
        "snapshot_hash": snapshot_hash,
        "ruleset_sha256": ruleset_sha256,
    }


def serialize_skill_publish_result(
    *,
    ok: bool,
    status: str,
    reason_code: str,
    skill: str,
    snapshot_hash: str = "",
    scanner: Mapping[str, Any] | None = None,
    completed_stage: str = "",
    completed_effects: Sequence[Mapping[str, Any]] | None = None,
    findings: Sequence[Mapping[str, Any]] | None = None,
    blocker_count: int = 0,
    warning_count: int = 0,
    audited_false_positive_count: int = 0,
    repair_hint: str = "",
    receipt: Mapping[str, Any] | None = None,
    expected_repository: str = "",
    extra_fields: Mapping[str, Any] | None = None,
) -> str:
    """Serialize one parseable envelope below the real tool-result cap.

    ``extra_fields`` adds caller-owned top-level keys BEFORE the transport
    cap loop, so late annotations (e.g. the publication-receipt write
    outcome) participate in the findings-trimming discipline instead of
    growing an already-fitted envelope past the cap. Keys must be new,
    string-named, and JSON-primitive-valued.
    """

    if type(ok) is not bool:
        raise ValueError("ok must be a boolean")
    safe_status = _bounded_text(status, 80, required=True)
    safe_reason = _bounded_text(reason_code, 120)
    safe_skill = _bounded_identifier(skill, field="skill")
    safe_snapshot_hash = _normalized_hash(snapshot_hash)
    safe_scanner = _normalize_scanner(scanner)
    safe_stage = _bounded_text(completed_stage, 64)
    if safe_stage and safe_stage not in _STAGE_INDEX:
        raise ValueError("unknown completed stage")
    safe_effects = _normalize_effects(completed_effects, completed_stage=safe_stage)
    safe_findings = normalize_skill_publish_findings(findings)
    total_findings = len(safe_findings)

    if ok:
        if safe_status != "pr_opened" or safe_stage != "pr_opened":
            raise ValueError("successful publication must finish at pr_opened")
        if not safe_snapshot_hash:
            raise ValueError("successful publication requires a captured snapshot")
        safe_receipt = validate_skill_publish_receipt(
            receipt,
            expected_repository=expected_repository,
            expected_skill=safe_skill,
            expected_snapshot_hash=safe_snapshot_hash,
            expected_ruleset_sha256=safe_scanner.get("ruleset_sha256") or "",
        )
        if safe_receipt is None:
            raise ValueError("successful publication requires a valid expected-repository receipt")
    else:
        if not safe_reason:
            raise ValueError("failed or partial publication requires a reason_code")
        if safe_status == "pr_opened" or receipt is not None:
            raise ValueError("failed or partial publication cannot carry a PR receipt")
        safe_receipt = None

    envelope: Dict[str, Any] = {
        "ok": ok,
        "operation": SKILL_PUBLISH_OPERATION,
        "status": safe_status,
        "reason_code": safe_reason,
        "skill": safe_skill,
        "snapshot_hash": safe_snapshot_hash,
        "scanner": safe_scanner,
        "completed_stage": safe_stage,
        "completed_effects": safe_effects,
        "findings": safe_findings,
        "omitted_count": 0,
        "blocker_count": _nonnegative_int(blocker_count, field="blocker_count"),
        "warning_count": _nonnegative_int(warning_count, field="warning_count"),
        "audited_false_positive_count": _nonnegative_int(
            audited_false_positive_count, field="audited_false_positive_count"
        ),
        "repair_hint": _bounded_text(repair_hint, 600),
    }
    if safe_receipt is not None:
        envelope["receipt"] = safe_receipt

    if extra_fields:
        for raw_key, raw_value in extra_fields.items():
            key = str(raw_key)
            if key in envelope:
                raise ValueError(f"extra field collides with envelope key: {key}")
            if raw_value is not None and not isinstance(raw_value, (bool, int, float, str)):
                raise ValueError(f"extra field must be a JSON primitive: {key}")
            envelope[key] = raw_value

    limit = tool_result_limit("submit_skill_to_hub")
    while True:
        envelope["omitted_count"] = total_findings - len(envelope["findings"])
        encoded = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(encoded) < limit:
            parsed = _loads_unique(encoded)
            if not isinstance(parsed, dict):
                raise AssertionError("skill publish result stopped being a JSON object")
            return encoded
        if not envelope["findings"]:
            raise ValueError("skill publish result fixed fields exceed the transport limit")
        envelope["findings"].pop()


def extract_skill_publish_result_metadata(result: Any) -> Dict[str, Any]:
    """Project safe attempt/receipt facts from the full, untruncated result."""

    if not isinstance(result, str) or not result.lstrip().startswith("{"):
        return {}
    from ouroboros.tools.tool_result import _HOST_NOTE_SEPARATOR

    try:
        payload = _loads_unique(result.partition(_HOST_NOTE_SEPARATOR)[0])
        if not isinstance(payload, dict) or payload.get("operation") != SKILL_PUBLISH_OPERATION:
            return {}
        if type(payload.get("ok")) is not bool:
            return {}
        status = _bounded_text(payload.get("status"), 80, required=True)
        skill = _bounded_identifier(payload.get("skill"), field="skill")
        snapshot_hash = _normalized_hash(payload.get("snapshot_hash"))
        scanner = _normalize_scanner(payload.get("scanner"))
        completed_stage = _bounded_text(payload.get("completed_stage"), 64)
        if completed_stage and completed_stage not in _STAGE_INDEX:
            return {}
        effects = _normalize_effects(payload.get("completed_effects"), completed_stage=completed_stage)
        attempt = {
            "ok": payload["ok"],
            "status": status,
            "reason_code": _bounded_text(payload.get("reason_code"), 120),
            "skill": skill,
            "snapshot_hash": snapshot_hash,
            "ruleset_sha256": str(scanner.get("ruleset_sha256") or ""),
            "completed_stage": completed_stage,
            "completed_effects": effects,
            "omitted_count": _nonnegative_int(payload.get("omitted_count", 0), field="omitted_count"),
            "blocker_count": _nonnegative_int(payload.get("blocker_count", 0), field="blocker_count"),
            "warning_count": _nonnegative_int(payload.get("warning_count", 0), field="warning_count"),
            "audited_false_positive_count": _nonnegative_int(
                payload.get("audited_false_positive_count", 0),
                field="audited_false_positive_count",
            ),
        }
        # The GitHub cause the transport observed rides beside the stage, only when present.
        for key, limit in (("error_detail", 640), ("github_operation", 64)):
            if payload.get(key):
                attempt[key] = _bounded_text(payload.get(key), limit)
        github_status = payload.get("github_status")
        if isinstance(github_status, int) and not isinstance(github_status, bool):
            attempt["github_status"] = github_status
        # Advice remains original evidence even when the requested PR was opened.
        for key in ("safety_advisory", "review_stale"):
            if type(payload.get(key)) is bool:
                attempt[key] = payload[key]
        for key, limit in (
            ("scanner_status", 32), ("scanner_errors", 2400), ("review_status", 80),
            ("review_profile", 80), ("reviewed_content_hash", 64), ("review_record", 1024),
            # Where the files went and what lists them are separate facts of one publication.
            ("destination_kind", 16), ("destination_source", 80), ("destination_folder", 240),
            ("files_published", 32), ("listed", 32), ("merged_commit", 64), ("release_digest", 64),
        ):
            if key in payload:
                attempt[key] = _bounded_text(payload[key], limit)
        metadata: Dict[str, Any] = {"skill_publish_attempt": attempt}
        receipt = payload.get("receipt")
        valid_receipt = validate_skill_publish_receipt(
            receipt,
            expected_repository=(receipt or {}).get("repository", "") if isinstance(receipt, Mapping) else "",
            expected_skill=skill,
            expected_snapshot_hash=snapshot_hash,
            expected_ruleset_sha256=str(scanner.get("ruleset_sha256") or ""),
        )
        if (
            payload["ok"] is True
            and status == "pr_opened"
            and completed_stage == "pr_opened"
            and valid_receipt is not None
        ):
            metadata["skill_publish_receipt"] = valid_receipt
        return metadata
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def skill_publish_target_from_task(task: Mapping[str, Any]) -> Dict[str, str] | None:
    """Read the one canonical structured target from a task."""

    if not isinstance(task, Mapping):
        return None
    metadata = task.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    target = metadata.get(SKILL_PUBLISH_TARGET_METADATA_KEY)
    if not isinstance(target, Mapping):
        return None
    try:
        return {
            "skill": _bounded_identifier(target.get("skill"), field="skill"),
            "repository": _normalize_repository(target.get("repository")),
        }
    except (TypeError, ValueError):
        return None


def apply_skill_publish_receipt_veto(
    loop_outcome: Dict[str, Any],
    task: Mapping[str, Any],
    llm_trace: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project explicit Publish failure without manufacturing success or certainty."""

    if not isinstance(task, Mapping) or task.get("type") != SKILL_PUBLISH_OPERATION:
        return loop_outcome
    axes = loop_outcome.get("outcome_axes") if isinstance(loop_outcome, dict) else None
    objective = axes.get("objective") if isinstance(axes, dict) else None
    if not isinstance(objective, dict):
        return loop_outcome
    objective_status = str(objective.get("status") or "")
    if objective_status not in _SUCCESS_OBJECTIVE_STATUSES | {"fail", "degraded", "not_evaluated"}:
        return loop_outcome

    target = skill_publish_target_from_task(task)
    saw_attempt = False
    saw_receipt = False
    pre_branch_failures = []
    if target is not None and isinstance(llm_trace, Mapping):
        calls = llm_trace.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, Mapping) or call.get("tool") != "submit_skill_to_hub":
                    continue
                attempt = call.get("skill_publish_attempt")
                if isinstance(attempt, Mapping):
                    saw_attempt = True
                if not isinstance(attempt, Mapping) or attempt.get("skill") == target["skill"]:
                    # completed_stage is LAST CONFIRMED, never proof that the next
                    # request was not sent. Branch creation follows fork_ready; a fork is
                    # synchronized only after GitHub refused that branch (422), so a typed
                    # fork-sync failure still leaves no branch on either path.
                    stage = attempt.get("completed_stage", "") if isinstance(attempt, Mapping) else None
                    pre_branch_failures.append(
                        isinstance(attempt, Mapping)
                        and attempt.get("ok") is False
                        and (
                            stage in ("", *SKILL_PUBLISH_STAGES[:_STAGE_INDEX["fork_ready"]])
                            or stage == "fork_ready" and attempt.get("reason_code") == "fork_sync_failed"
                        )
                    )
                raw_receipt = call.get("skill_publish_receipt")
                if isinstance(raw_receipt, Mapping):
                    saw_receipt = True
                receipt = validate_skill_publish_receipt(
                    raw_receipt if isinstance(raw_receipt, Mapping) else None,
                    expected_repository=target["repository"],
                    expected_skill=target["skill"],
                    expected_snapshot_hash=str((attempt or {}).get("snapshot_hash") or "")
                    if isinstance(attempt, Mapping)
                    else "",
                    expected_ruleset_sha256=str((attempt or {}).get("ruleset_sha256") or "")
                    if isinstance(attempt, Mapping)
                    else "",
                )
                if (
                    receipt is not None
                    and isinstance(attempt, Mapping)
                    and attempt.get("ok") is True
                    and attempt.get("status") == "pr_opened"
                    and attempt.get("completed_stage") == "pr_opened"
                    and attempt.get("skill") == target["skill"]
                    and bool(attempt.get("snapshot_hash"))
                ):
                    # Any valid same-target receipt is sufficient.  Earlier failed
                    # attempts stay in the trace and retain their execution semantics.
                    return loop_outcome

    definite_pre_branch_failure = bool(pre_branch_failures) and all(pre_branch_failures) and not saw_receipt
    if definite_pre_branch_failure:
        reason = SKILL_PUBLISH_PR_NOT_CREATED
        loop_outcome["reason_code"] = reason
    elif objective_status not in _SUCCESS_OBJECTIVE_STATUSES:
        # A degraded review says nothing about publication. Preserve it unless
        # every relevant attempt proves that no branch/PR request could start.
        return loop_outcome
    elif target is None:
        reason = "skill_publish_target_missing"
    elif saw_receipt:
        reason = "skill_publish_receipt_mismatch"
    elif saw_attempt:
        reason = "skill_publish_receipt_absent"
    else:
        reason = "skill_publish_not_attempted"
    objective.update(
        {
            "status": "fail",
            # Keep the objective-authority source vocabulary: the receipt is a veto over
            # the reviewed result, not a second positive oracle. The normalizer rejects
            # every objective whose source is not a review's (``result_review``, or a
            # recorded task acceptance on older records), so the host veto is carried as
            # a typed negative fact under that source rather than minting authority.
            "source": "result_review",
            "outcome_tier": "blocked_with_evidence",
            "reason": reason,
            "receipt_veto": {
                "status": "failed", "reason": reason,
                **({"detail": SKILL_PUBLISH_PR_NOT_CREATED_DETAIL} if definite_pre_branch_failure else {}),
            },
        }
    )
    return loop_outcome


__all__ = [
    "SKILL_PUBLISH_OPERATION",
    "SKILL_PUBLISH_STAGES",
    "SKILL_PUBLISH_TARGET_METADATA_KEY",
    "SkillPublishDestinationError",
    "apply_skill_publish_receipt_veto",
    "extract_skill_publish_result_metadata",
    "normalize_skill_publish_findings",
    "parse_skill_publish_destination",
    "serialize_skill_publish_result",
    "skill_publish_target_from_task",
    "validate_skill_publish_receipt",
]
