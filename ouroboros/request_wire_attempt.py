"""Physical-attempt validation and user-visible request-wire disclosure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping, Tuple

from ouroboros.request_wire_contract import (
    TOOL_DIALECTS,
    WIRE_REASON_CODES,
    _valid_sha256,
)

if TYPE_CHECKING:
    from ouroboros.request_wire_receipts import WireCandidateManifest
    from ouroboros.usage_accounting import PhysicalAttemptCapture


def validate_normalized_wire_success(
    normalized_response: Mapping[str, Any],
    normalized_usage: Mapping[str, Any],
) -> None:
    """Reject transport/body-error results before semantic classification."""
    if not isinstance(normalized_response, Mapping):
        raise ValueError("normalized wire response must be an object")
    if not isinstance(normalized_usage, Mapping):
        raise ValueError("normalized wire usage must be an object")
    if "provider_error" in normalized_usage:
        raise ValueError("provider-error response cannot prove wire semantic success")


def validate_physical_wire_attempt(
    candidate: "WireCandidateManifest",
    capture: "PhysicalAttemptCapture",
    *,
    allow_unresolved_accounting: bool = False,
) -> None:
    """Require the settled accounting attempt for this exact physical candidate.

    ``allow_unresolved_accounting`` also accepts a capture whose response was
    received but whose post-response settle failed (state ``unresolved``, money
    bound at the reservation): the candidate WAS sent and answered, so what was
    applied on the wire is a fact regardless of the ledger's availability.
    Disclosure uses this; compatibility-profile learning keeps requiring settled.
    """
    from ouroboros.usage_accounting import PhysicalAttemptCapture

    if not isinstance(capture, PhysicalAttemptCapture):
        raise ValueError("wire evidence requires a physical-attempt capture")
    accepted_states = {"settled", "unresolved"} if allow_unresolved_accounting else {"settled"}
    if capture.state not in accepted_states or not capture.attempt_id:
        raise ValueError("wire evidence requires a settled physical attempt")
    if capture.candidate_measurement_kind != "canonical_json_v1":
        raise ValueError("wire evidence requires an inspectable canonical candidate")
    if capture.candidate_raw_sha256 != candidate.candidate_sha256:
        raise ValueError("wire evidence candidate digest differs from the physical attempt")
    if (
        capture.provider != candidate.accepted_profile.provider
        or capture.model != candidate.physical_model
    ):
        raise ValueError("wire evidence capture belongs to another route")
    manifest_ref = capture.candidate_manifest_ref
    if (
        not isinstance(manifest_ref, Mapping)
        or str(manifest_ref.get("call_id") or "") != capture.attempt_id
        or not _valid_sha256(manifest_ref.get("sha256"))
    ):
        raise ValueError("wire evidence lacks the physical-candidate manifest receipt")


@dataclass(frozen=True)
class WireUsageDisclosure:
    requested_effort: str
    applied_effort: str
    requested_tool_dialect: str
    applied_tool_dialect: str
    reason_code: str
    source_profile_fingerprint: str
    accepted_profile_fingerprint: str
    attempt_id: str
    candidate_sha256: str
    ladder_ordinal: int
    applied_actions: Tuple[Mapping[str, Any], ...]
    task_local: bool = False

    @classmethod
    def from_candidate(
        cls,
        candidate: "WireCandidateManifest",
        physical_attempt: "PhysicalAttemptCapture",
    ) -> "WireUsageDisclosure":
        # A response whose settle failed (UsageLockUnavailable under a 64-lane
        # convoy, CyberGym r8 2026-09-04) is still a served call: without the
        # disclosure it carried no applied_effort and the benchmark executor
        # refused the WHOLE task ("gateway telemetry omitted effort for a served
        # call") — 6 tasks, 40-100 rounds of finished work each.
        validate_physical_wire_attempt(candidate, physical_attempt, allow_unresolved_accounting=True)
        return cls(
            requested_effort=candidate.requested_effort,
            applied_effort=candidate.candidate_spec.effort,
            requested_tool_dialect=candidate.source_profile.tool_dialect,
            applied_tool_dialect=candidate.accepted_profile.tool_dialect,
            reason_code=candidate.candidate_spec.reason_code,
            source_profile_fingerprint=candidate.source_profile.fingerprint,
            accepted_profile_fingerprint=candidate.accepted_profile.fingerprint,
            attempt_id=physical_attempt.attempt_id,
            candidate_sha256=candidate.candidate_sha256,
            ladder_ordinal=candidate.ladder_ordinal,
            applied_actions=candidate.disclosed_actions(),
            task_local=candidate.task_local,
        )

    def __post_init__(self) -> None:
        from ouroboros.config import EFFORT_SCALE

        if self.reason_code not in WIRE_REASON_CODES:
            raise ValueError("unsupported wire disclosure reason")
        if self.requested_effort not in EFFORT_SCALE or self.applied_effort not in (
            *EFFORT_SCALE,
            "provider_default",
        ):
            raise ValueError("unsupported wire disclosure effort")
        if (
            self.requested_tool_dialect not in TOOL_DIALECTS
            or self.applied_tool_dialect not in TOOL_DIALECTS
        ):
            raise ValueError("unsupported wire disclosure dialect")
        if not self.attempt_id or not _valid_sha256(self.candidate_sha256):
            raise ValueError("wire disclosure requires physical candidate identity")
        if (
            not _valid_sha256(self.source_profile_fingerprint)
            or not _valid_sha256(self.accepted_profile_fingerprint)
        ):
            raise ValueError("wire disclosure requires exact profile fingerprints")
        if type(self.ladder_ordinal) is not int or self.ladder_ordinal < 1:
            raise ValueError("wire disclosure requires a positive ladder ordinal")
        if not isinstance(self.applied_actions, tuple):
            raise ValueError("wire disclosure actions must be a tuple")
        if self.task_local != (self.reason_code == "task_local_availability_fallback"):
            raise ValueError("wire disclosure task-local reason mismatch")
        if (
            self.applied_effort == "none"
            and self.requested_effort != "none"
            and not self.task_local
        ):
            raise ValueError("non-requested explicit none must remain task-local")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested_effort": self.requested_effort,
            "applied_effort": self.applied_effort,
            "requested_tool_dialect": self.requested_tool_dialect,
            "applied_tool_dialect": self.applied_tool_dialect,
            "reason_code": self.reason_code,
            "source_profile_fingerprint": self.source_profile_fingerprint,
            "accepted_profile_fingerprint": self.accepted_profile_fingerprint,
            "attempt_id": self.attempt_id,
            "candidate_sha256": self.candidate_sha256,
            "ladder_ordinal": self.ladder_ordinal,
            "applied_actions": [dict(item) for item in self.applied_actions],
            "task_local": self.task_local,
        }
