"""Selected-skill publication preflight and safe in-process scan cache.

Preflight returns exactly one backend-authored state (``ready``, ``warnings``,
``needs_attention``, ``repairable``, ``hard_block``); the browser renders it, and
only ``hard_block`` sets ``task_start_allowed`` false.
"""

from __future__ import annotations

import asyncio
import collections
import json
import pathlib
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ouroboros.betterleaks_runtime import resolve_betterleaks
from ouroboros.config import get_ouroboroshub_catalog_url, get_runtime_mode, get_skills_repo_path
from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency
from ouroboros.gateway._helpers import request_drive_root
from ouroboros.gateway.contracts import SkillPublishPreflightResponse
from ouroboros.skill_loader import (
    _sanitize_skill_name,
    discover_selected_skill_candidates,
)
from ouroboros.skill_publish_eligibility import (
    PUBLISHABLE_SOURCES,
    PUBLISHABLE_STATUSES,
    publication_author_acceptance,
)
from ouroboros.skill_publish_result import (
    SkillPublishDestinationError,
    normalize_skill_publish_findings,
    parse_skill_publish_destination,
)
from ouroboros.skill_publish_scanner import (
    BETTERLEAKS_ENGINE,
    ScannerContractResult,
    ScannerExecutable,
    SecretFinding,
    SecretScanResult,
    probe_scanner_contract,
    scan_named_bytes,
)
from ouroboros.skill_publish_snapshot import (
    SkillPublishSnapshotError,
    capture_skill_publish_candidate,
)
from ouroboros.skill_review_status import (
    STATUS_BLOCKERS,
    STATUS_PENDING,
    STATUS_WARNINGS,
    normalize_skill_review_status,
)
from ouroboros.tools.github import github_token_from_env_or_settings

PREFLIGHT_STATES = frozenset({"ready", "warnings", "needs_attention", "repairable", "hard_block"})
_MAX_PREFLIGHT_FINDINGS = 100
_SCAN_CACHE_CAPACITY = 32


@dataclass(frozen=True)
class _SafeScanProjection:
    """Bounded candidate-free scan facts safe to retain in memory."""

    status: str
    engine: str
    version: str
    ruleset_sha256: str
    scan_contract_sha256: str
    findings: tuple[SecretFinding, ...]
    omitted_count: int
    blocker_count: int
    warning_count: int
    audited_false_positive_count: int
    reason_code: str
    repair_hint: str


@dataclass(frozen=True)
class _CachedScan:
    executable_identity: str
    projection: _SafeScanProjection


def _safe_scan_projection(result: SecretScanResult) -> _SafeScanProjection:
    ordered = normalize_skill_publish_findings([asdict(finding) for finding in result.findings])
    visible = tuple(SecretFinding(**finding) for finding in ordered[:_MAX_PREFLIGHT_FINDINGS])
    return _SafeScanProjection(
        status=result.status,
        engine=result.engine,
        version=result.version,
        ruleset_sha256=result.ruleset_sha256,
        scan_contract_sha256=result.scan_contract_sha256,
        findings=visible,
        omitted_count=max(0, len(result.findings) - len(visible)),
        blocker_count=int(result.blocker_count),
        warning_count=int(result.warning_count),
        audited_false_positive_count=int(result.audited_false_positive_count),
        reason_code=result.reason_code,
        repair_hint=result.repair_hint,
    )


def _contract_failure_projection(
    result: ScannerContractResult,
) -> _SafeScanProjection:
    return _SafeScanProjection(
        status="scanner_error",
        engine=result.engine,
        version=result.version,
        ruleset_sha256=result.ruleset_sha256,
        scan_contract_sha256=result.scan_contract_sha256,
        findings=(),
        omitted_count=0,
        blocker_count=0,
        warning_count=0,
        audited_false_positive_count=0,
        reason_code=result.reason_code,
        repair_hint=result.repair_hint,
    )


class _SafeScanCache:
    """Bounded LRU containing only bounded candidate-free projections."""

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, int(capacity))
        self._entries: collections.OrderedDict[tuple[str, str, str], _CachedScan] = collections.OrderedDict()
        self._lock = threading.Lock()

    def get(
        self,
        *,
        skill: str,
        snapshot_hash: str,
        executable_identity: str,
        scan_contract_sha256: str,
    ) -> _SafeScanProjection | None:
        """Return one exact snapshot/current-contract projection."""
        key = (skill, snapshot_hash, scan_contract_sha256)
        with self._lock:
            cached = self._entries.get(key)
            if cached is None or cached.executable_identity != executable_identity:
                return None
            self._entries.move_to_end(key)
            return cached.projection

    def put(
        self,
        *,
        skill: str,
        snapshot_hash: str,
        executable_identity: str,
        projection: _SafeScanProjection,
    ) -> None:
        contract = str(projection.scan_contract_sha256 or "")
        if not contract or projection.status == "scanner_error":
            return
        key = (skill, snapshot_hash, contract)
        with self._lock:
            self._entries[key] = _CachedScan(executable_identity, projection)
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


_PREFLIGHT_SCAN_CACHE = _SafeScanCache(_SCAN_CACHE_CAPACITY)


@dataclass(frozen=True)
class SkillPublishPreflightOutcome:
    payload: SkillPublishPreflightResponse
    status_code: int = 200


def _review_projection(loaded: Any, *, stale: bool | None = None, current_hash: str = "") -> Dict[str, Any]:
    return {
        "status": normalize_skill_review_status(loaded.review.status),
        "stale": (loaded.review.is_stale_for(loaded.content_hash) if stale is None else bool(stale)),
        "profile": str(getattr(loaded.review, "review_profile", "") or ""),
        "reviewed_content_hash": loaded.review.reviewed_content_hash or loaded.review.content_hash,
        "author_disposition": dict(loaded.review.author_disposition),
        "author_accepted": bool(publication_author_acceptance(loaded.review, current_hash or loaded.content_hash)),
    }


def _scanner_projection(result: _SafeScanProjection | None = None) -> Dict[str, Any]:
    return {
        "status": result.status if result is not None else "not_run",
        "engine": result.engine if result is not None else BETTERLEAKS_ENGINE,
        "version": result.version if result is not None else "",
        "ruleset_sha256": result.ruleset_sha256 if result is not None else "",
        **({"reason_code": result.reason_code, "repair_hint": result.repair_hint}
           if result is not None and result.status == "scanner_error" else {}),
    }


def _response(
    *,
    ok: bool,
    skill: str,
    repository: str,
    state: str,
    publication_ready: bool,
    task_start_allowed: bool,
    snapshot_hash: str = "",
    review: Dict[str, Any] | None = None,
    scan: _SafeScanProjection | None = None,
    reason_code: str = "",
    summary: str = "",
    repair_hint: str = "",
) -> SkillPublishPreflightResponse:
    if state not in PREFLIGHT_STATES:
        raise ValueError("invalid skill publish preflight state")
    return {
        "ok": bool(ok),
        "skill": str(skill or ""),
        "repository": str(repository or ""),
        "state": state,
        "publication_ready": bool(publication_ready),
        "task_start_allowed": bool(task_start_allowed),
        "snapshot_hash": str(snapshot_hash or ""),
        "review": dict(review or {"status": "", "stale": True, "profile": ""}),
        "scanner": _scanner_projection(scan),
        "findings": [asdict(item) for item in (scan.findings if scan is not None else ())],
        "omitted_count": int(scan.omitted_count if scan is not None else 0),
        "blocker_count": int(scan.blocker_count if scan is not None else 0),
        "warning_count": int(scan.warning_count if scan is not None else 0),
        "audited_false_positive_count": int(scan.audited_false_positive_count if scan is not None else 0),
        "reason_code": str(reason_code or ""),
        "summary": str(summary or ""),
        "repair_hint": str(repair_hint or ""),
    }


def _hard_block(
    *,
    skill: str,
    repository: str,
    reason_code: str,
    summary: str,
    repair_hint: str,
    status_code: int,
    review: Dict[str, Any] | None = None,
) -> SkillPublishPreflightOutcome:
    return SkillPublishPreflightOutcome(
        _response(
            ok=False,
            skill=skill,
            repository=repository,
            state="hard_block",
            publication_ready=False,
            task_start_allowed=False,
            review=review,
            reason_code=reason_code,
            summary=summary,
            repair_hint=repair_hint,
        ),
        status_code=status_code,
    )


def _needs_attention(
    *,
    skill: str,
    repository: str,
    snapshot_hash: str,
    review: Dict[str, Any],
    reason_code: str,
    summary: str,
    repair_hint: str,
    scan: _SafeScanProjection | None = None,
) -> SkillPublishPreflightOutcome:
    return SkillPublishPreflightOutcome(
        _response(
            ok=True,
            skill=skill,
            repository=repository,
            state="needs_attention",
            publication_ready=False,
            task_start_allowed=True,
            snapshot_hash=snapshot_hash,
            review=review,
            scan=scan,
            reason_code=reason_code,
            summary=summary,
            repair_hint=repair_hint,
        )
    )


def build_skill_publish_preflight(
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    repo_path: str | None = None,
    catalog_url: str | None = None,
    github_token_configured: bool | None = None,
    admission_problem: tuple[int, str, str] | None = None,
) -> SkillPublishPreflightOutcome:
    """Resolve, capture, scan, cache, and classify one selected skill."""

    cyber = mode_has_unrestricted_agency(get_runtime_mode())
    raw_skill = str(skill_name or "").strip()
    safe_skill = _sanitize_skill_name(raw_skill)
    try:
        owner, repo, _base_branch = parse_skill_publish_destination(
            catalog_url if catalog_url is not None else get_ouroboroshub_catalog_url()
        )
        repository = f"{owner}/{repo}"
    except SkillPublishDestinationError as exc:
        return _hard_block(
            skill=safe_skill if safe_skill != "_unnamed" else "",
            repository="",
            reason_code=exc.reason_code,
            summary="The configured Hub destination is invalid.",
            repair_hint="Repair the configured OuroborosHub catalog URL, then retry.",
            status_code=409,
        )
    if not raw_skill or safe_skill == "_unnamed" or safe_skill != raw_skill:
        return _hard_block(
            skill="",
            repository=repository,
            reason_code="skill_invalid",
            summary="The selected skill name is invalid.",
            repair_hint="Choose one installed skill, then retry.",
            status_code=400,
        )
    if admission_problem is not None:
        status_code, reason_code, summary = admission_problem
        return _hard_block(
            skill=safe_skill,
            repository=repository,
            reason_code=reason_code or "task_admission_unavailable",
            summary=summary or "Managed task admission is unavailable.",
            repair_hint="Wait for managed-task admission to recover, then retry.",
            status_code=status_code,
        )
    if github_token_configured is None:
        github_token_configured = bool(github_token_from_env_or_settings())
    if not github_token_configured:
        return _hard_block(
            skill=safe_skill,
            repository=repository,
            reason_code="github_token_missing",
            summary="GitHub authority is not configured.",
            repair_hint="Add GitHub authority in Settings -> Secrets, then retry.",
            status_code=409,
        )

    skills = discover_selected_skill_candidates(
        pathlib.Path(drive_root),
        safe_skill,
        repo_path=get_skills_repo_path() if repo_path is None else repo_path,
    )
    if not skills:
        return _hard_block(
            skill=safe_skill,
            repository=repository,
            reason_code="skill_not_found",
            summary="The selected skill is not installed.",
            repair_hint="Choose one installed skill, then retry.",
            status_code=404,
        )
    if len(skills) != 1 or bool(getattr(skills[0], "identity_collision", False)):
        return _hard_block(
            skill=safe_skill,
            repository=repository,
            reason_code="skill_identity_ambiguous",
            summary="The selected skill name resolves to more than one payload.",
            repair_hint="Rename the colliding skill directories, then retry.",
            status_code=409,
        )
    loaded = skills[0]
    review = _review_projection(loaded)
    if str(loaded.source or "").lower() not in PUBLISHABLE_SOURCES and not cyber:
        return _hard_block(
            skill=safe_skill,
            repository=repository,
            reason_code="skill_source_unsupported",
            summary="This installed skill source cannot be published.",
            repair_hint="Copy or adapt the skill into a publishable user-managed source.",
            status_code=409,
            review=review,
        )
    try:
        snapshot = capture_skill_publish_candidate(loaded)
    except SkillPublishSnapshotError as exc:
        return _needs_attention(
            skill=safe_skill,
            repository=repository,
            snapshot_hash="",
            review=_review_projection(loaded, stale=True),
            reason_code=exc.reason_code,
            summary="The current skill bytes could not complete publication capture.",
            repair_hint="Repair or re-review the skill payload, then retry.",
        )

    review = _review_projection(
        loaded,
        stale=loaded.review.is_stale_for(snapshot.content_hash),
        current_hash=snapshot.content_hash,
    )
    review_status = str(review["status"] or STATUS_PENDING)
    review_profile = str(review["profile"] or "")
    version_missing = not str(snapshot.manifest.version or "").strip()

    runtime = resolve_betterleaks(data_root=pathlib.Path(drive_root))
    executable = ScannerExecutable(
        path=pathlib.Path(runtime.binary_path) if runtime.binary_path else None,
        identity=str(runtime.binary_sha256 or ""),
        status=(runtime.status if runtime.status in {"ready", "missing", "corrupt"} else "corrupt"),
    )
    current_contract = probe_scanner_contract(
        executable=executable,
        drive_root=pathlib.Path(drive_root),
        scope="session",
        owner_task_id="",
        honor_inline_allowances=True,
    )
    if current_contract.status != "ready":
        scan = _contract_failure_projection(current_contract)
    else:
        scan = _PREFLIGHT_SCAN_CACHE.get(
            skill=safe_skill,
            snapshot_hash=snapshot.content_hash,
            executable_identity=executable.identity,
            scan_contract_sha256=current_contract.scan_contract_sha256,
        )
    if scan is None:
        full_scan = scan_named_bytes(
            {item.path: item.content for item in snapshot.public_files},
            executable=executable,
            drive_root=pathlib.Path(drive_root),
            scope="session",
            owner_task_id="",
            honor_inline_allowances=True,
        )
        scan = _safe_scan_projection(full_scan)
        _PREFLIGHT_SCAN_CACHE.put(
            skill=safe_skill,
            snapshot_hash=snapshot.content_hash,
            executable_identity=executable.identity,
            projection=scan,
        )

    if scan.status == "scanner_error" and not cyber:
        return SkillPublishPreflightOutcome(
            _response(
                ok=True,
                skill=safe_skill,
                repository=repository,
                state="repairable",
                publication_ready=False,
                task_start_allowed=True,
                snapshot_hash=snapshot.content_hash,
                review=review,
                scan=scan,
                reason_code=scan.reason_code or "scanner_report_invalid",
                summary="The Betterleaks publication scanner needs repair.",
                repair_hint=scan.repair_hint,
            )
        )
    if scan.blocker_count and not cyber:
        attention = (
            "secret_blocked",
            "High-confidence secret candidates need attention.",
            "Inspect the redacted locations, repair or audit them, then retry.",
        )
    elif review_profile == "owner_attested" and not cyber and not review["author_accepted"]:
        attention = (
            "review_owner_attested",
            "A full skill review is required before public publication.",
            "Run the full skill review, then retry publication.",
        )
    elif review["stale"] and not cyber and not review["author_accepted"]:
        attention = (
            "review_stale",
            "The skill review is stale for the captured bytes.",
            "Run a fresh skill review, then retry publication.",
        )
    elif review_status not in PUBLISHABLE_STATUSES and not cyber and not review["author_accepted"]:
        reason_code = "review_blockers" if review_status == STATUS_BLOCKERS else "review_pending"
        attention = (
            reason_code,
            (
                "The skill review has publication blockers."
                if reason_code == "review_blockers"
                else "The skill review is pending."
            ),
            "Resolve the review work, run a fresh review, then retry.",
        )
    elif version_missing:
        attention = (
            "manifest_version_missing",
            "The skill manifest has no publication version.",
            "Add a manifest version, run a fresh review, then retry.",
        )
    else:
        attention = None
    if attention is not None:
        return _needs_attention(
            skill=safe_skill,
            repository=repository,
            snapshot_hash=snapshot.content_hash,
            review=review,
            reason_code=attention[0],
            summary=attention[1],
            repair_hint=attention[2],
            scan=scan,
        )

    has_warnings = bool(
        review["author_accepted"] or review_status == STATUS_WARNINGS or scan.warning_count or scan.audited_false_positive_count
        or cyber and (
            scan.status == "scanner_error" or scan.blocker_count or review["stale"]
            or review_status not in PUBLISHABLE_STATUSES or review_profile == "owner_attested"
        )
    )
    return SkillPublishPreflightOutcome(
        _response(
            ok=True,
            skill=safe_skill,
            repository=repository,
            state="warnings" if has_warnings else "ready",
            publication_ready=True,
            task_start_allowed=True,
            snapshot_hash=snapshot.content_hash,
            review=review,
            scan=scan,
            reason_code=scan.reason_code or ("warnings_present" if has_warnings else ""),
            summary=(
                "Publication can proceed with author-accepted bytes; original reviewer evidence remains disclosed."
                if review["author_accepted"] else "Cyber Pro publication can proceed; review and scanner findings remain advisory."
                if cyber and has_warnings else "Publication preflight completed with redacted warnings."
                if has_warnings
                else "Publication preflight is ready."
            ),
            repair_hint="",
        )
    )


def _task_admission_problem(request: Request) -> tuple[int, str, str] | None:
    """Project the existing task-admission readiness check into safe facts."""

    from ouroboros.gateway.tasks import _supervisor_ready_error

    response = _supervisor_ready_error(request)
    if response is None:
        return None
    try:
        payload = json.loads(bytes(response.body).decode("utf-8"))
    except (UnicodeError, ValueError):
        payload = {}
    return (
        int(response.status_code),
        str(payload.get("reason_code") or "task_admission_unavailable"),
        str(payload.get("error") or "Managed task admission is unavailable."),
    )


async def api_skill_publish_preflight(request: Request) -> JSONResponse:
    """POST selected-skill preflight without creating a task or remote effect."""

    skill_name = str(request.path_params.get("skill") or "")
    outcome = await asyncio.to_thread(
        build_skill_publish_preflight,
        request_drive_root(request),
        skill_name,
        admission_problem=_task_admission_problem(request),
    )
    return JSONResponse(outcome.payload, status_code=outcome.status_code)


def skill_publish_routes() -> list[Route]:
    return [
        Route(
            "/api/skills/{skill}/publish-preflight",
            endpoint=api_skill_publish_preflight,
            methods=["POST"],
        ),
    ]


__all__ = [
    "PREFLIGHT_STATES",
    "SkillPublishPreflightOutcome",
    "api_skill_publish_preflight",
    "build_skill_publish_preflight",
    "skill_publish_routes",
]
