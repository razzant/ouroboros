"""Route-neutral execution evidence for ``run_external_review --contributor``."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess


# Capability deltas are disclosures, not a second verdict vocabulary. Most
# describe a supported fallback (strict parsing, transcript extraction, or a
# route-resolved model). Only this delta carries an otherwise-unprojected fact
# that contradicts the configured execution route; model identity, access,
# profile, custody, and settlement are checked from their typed receipt fields.
_EXECUTION_CONTRADICTION_DELTAS = frozenset({"session_ran_off_pinned_route"})


def _git_file_at_ref(repo: pathlib.Path, ref: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout if result.returncode == 0 else None


def _release_carrier_projection(repo: pathlib.Path, ref: str) -> dict[str, str]:
    """Extract release-only values without executing code from either revision."""
    projection: dict[str, str] = {}

    version = _git_file_at_ref(repo, ref, "VERSION")
    if version is not None:
        projection["VERSION"] = version.strip()

    pyproject = _git_file_at_ref(repo, ref, "pyproject.toml")
    if pyproject is not None:
        project_match = re.search(
            r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)",
            pyproject,
        )
        version_match = re.search(
            r'(?m)^version\s*=\s*"([^"]+)"',
            project_match.group(1) if project_match else "",
        )
        if version_match:
            projection["pyproject.project.version"] = version_match.group(1)

    package = _git_file_at_ref(repo, ref, "web/package.json")
    if package is not None:
        try:
            package_version = str((json.loads(package) or {}).get("version") or "")
        except Exception:
            package_version = "<invalid-json>"
        if package_version:
            projection["web.package.version"] = package_version

    api_types = _git_file_at_ref(repo, ref, "web/modules/api_types.js")
    if api_types is not None:
        match = re.search(
            r"GATEWAY_CONTRACT_VERSION\s*=\s*['\"]([^'\"]+)['\"]",
            api_types,
        )
        if match:
            projection["gateway.contract.version"] = match.group(1)

    readme = _git_file_at_ref(repo, ref, "README.md")
    if readme is not None:
        badge = re.search(r"\[!\[Version\s+([^\]]+)\]", readme)
        if badge:
            projection["readme.badge.version"] = badge.group(1)
        history = readme.split("## Version History", 1)
        if len(history) == 2:
            row = re.search(r"(?m)^\|\s*\d+\.\d+\.\d+[^\n]*$", history[1])
            if row:
                projection["readme.latest_history_row"] = row.group(0).strip()
        download_occurrences: dict[str, int] = {}
        for proof_id, url in re.findall(
            r"(?m)^\[download-([^\]]+)\]:\s*(\S+)\s*$", readme
        ):
            occurrence = download_occurrences.get(proof_id, 0)
            download_occurrences[proof_id] = occurrence + 1
            projection[f"readme.download.{proof_id}.{occurrence}"] = url

    for rel_path, prefix in (
        ("site/install/index.html", "site.install.download"),
        ("docs/install/index.html", "docs.install.download"),
    ):
        html = _git_file_at_ref(repo, ref, rel_path) or ""
        download_occurrences = {}
        for anchor in re.findall(r"(?is)<a\b[^>]*>", html):
            proof = re.search(
                r'data-release-download="([^"]+)"', anchor, re.IGNORECASE
            )
            href = re.search(r'href="([^"]+)"', anchor, re.IGNORECASE)
            if proof and href:
                proof_id = proof.group(1)
                occurrence = download_occurrences.get(proof_id, 0)
                download_occurrences[proof_id] = occurrence + 1
                projection[f"{prefix}.{proof_id}.{occurrence}"] = href.group(1)

    architecture = _git_file_at_ref(repo, ref, "docs/ARCHITECTURE.md")
    if architecture is not None:
        header = re.search(r"(?m)^# Ouroboros v([^\s]+)", architecture)
        if header:
            projection["architecture.header.version"] = header.group(1)

    uv_lock = _git_file_at_ref(repo, ref, "uv.lock")
    if uv_lock is not None:
        for block in re.findall(
            r"(?ms)^\[\[package\]\]\s*(.*?)(?=^\[\[package\]\]|\Z)", uv_lock
        ):
            if not re.search(r'(?m)^name\s*=\s*"ouroboros"\s*$', block):
                continue
            if not re.search(
                r'(?m)^source\s*=\s*\{\s*editable\s*=\s*"\."\s*\}\s*$', block
            ):
                continue
            match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', block)
            if match:
                projection["uv.editable_root.version"] = match.group(1)
            break

    return projection


def release_sensitive_changes(
    repo: pathlib.Path,
    base_sha: str,
    head_sha: str,
    changed_paths: list[str],
    release_machinery_paths: frozenset[str],
) -> dict:
    """Compare release carriers and name touched release machinery."""
    base_projection = _release_carrier_projection(repo, base_sha)
    head_projection = _release_carrier_projection(repo, head_sha)
    fields = sorted(
        key
        for key in set(base_projection) | set(head_projection)
        if base_projection.get(key) != head_projection.get(key)
    )
    machinery = sorted(set(changed_paths) & release_machinery_paths)
    return {
        "changed": bool(fields or machinery),
        "carrier_fields": fields,
        "machinery_paths": machinery,
    }


def _call_payload(call_ref: dict, drive_root: pathlib.Path) -> dict:
    from ouroboros.observability import read_blob_ref

    projection = (call_ref or {}).get("redacted_projection_ref") or {}
    if not projection:
        return {}
    payload = read_blob_ref(drive_root, projection)
    if not isinstance(payload, dict):
        raise ValueError("review call payload is not an object")
    return payload


def _api_model_identity(model: str) -> str:
    from ouroboros.provider_models import normalize_model_identity

    return normalize_model_identity(str(model or "").removeprefix("openrouter::"))


def _receipt_payloads(
    actor: dict,
    *,
    drive_root: pathlib.Path,
    surface: str,
    slot_id: str,
    mismatches: list[str],
) -> tuple[dict, dict]:
    payloads: list[dict] = []
    for label in ("prompt", "response"):
        try:
            payloads.append(_call_payload(dict(actor.get(f"{label}_ref") or {}), drive_root))
        except Exception as exc:
            payloads.append({})
            mismatches.append(
                f"unreadable_{label}_receipt:{surface}:{slot_id}:{type(exc).__name__}"
            )
    return payloads[0], payloads[1]


def _compare_dispatch(
    *,
    surface: str,
    slot_id: str,
    row: dict,
    dispatched_slot: dict,
    mismatches: list[str],
) -> dict:
    route = dict(row.get("route") or {})
    kind = str(route.get("kind") or "")
    target = str(route.get("target_id") or "")
    dispatched = {
        "route_kind": str(dispatched_slot.get("route") or "") or None,
        "model": str(dispatched_slot.get("model") or "") or None,
        "session_target": str(dispatched_slot.get("session_target") or "") or None,
        "profile_id": str(dispatched_slot.get("session_profile") or "") or None,
        "effort": str(dispatched_slot.get("effort") or "") or None,
    }
    if not dispatched_slot:
        mismatches.append(f"prompt_receipt_absent:{surface}:{slot_id}")
        return dispatched
    expected = {
        "route": kind,
        "model": target,
        "effort": str(row.get("effort") or ""),
        "session_target": target if kind == "agent_session" else "",
        "session_profile": str(route.get("profile_id") or ""),
    }
    for key, value in expected.items():
        actual = str(dispatched_slot.get(key) or "")
        if actual != value:
            mismatches.append(
                f"dispatch_{key}_mismatch:{surface}:{slot_id}:"
                f"{value or 'absent'}->{actual or 'absent'}"
            )
    return dispatched


def _session_evidence(
    *,
    surface: str,
    slot_id: str,
    route: dict,
    status: str,
    observed_model: str,
    usage: dict,
    transcript: str,
    deltas: list[dict],
    receipt: dict,
    mismatches: list[str],
) -> None:
    from ouroboros.subagents import parse_subagent_harness

    expected = parse_subagent_harness(str(route.get("target_id") or ""))
    expected_harness = str(getattr(expected, "route_id", "") or "")
    delegated_route = str(usage.get("delegated_route") or "")
    if not expected_harness or delegated_route != expected_harness:
        mismatches.append(
            f"harness_mismatch:{surface}:{slot_id}:"
            f"{expected_harness or 'invalid'}->{delegated_route or 'absent'}"
        )
    expected_model = str(getattr(expected, "model", "") or "")
    if expected_model and observed_model:
        if _api_model_identity(expected_model) == _api_model_identity(observed_model):
            receipt["model_verification"] = "exact"
        elif any(char.isspace() for char in observed_model):
            receipt["model_verification"] = "observed_display_label"
            mismatches.append(
                f"model_identity_unverified:{surface}:{slot_id}:"
                f"{expected_model}->{observed_model}"
            )
        else:
            receipt["model_verification"] = "mismatch"
            mismatches.append(
                f"model_mismatch:{surface}:{slot_id}:{expected_model}->{observed_model}"
            )
    elif expected_model:
        receipt["model_verification"] = "absent"
    else:
        receipt["model_verification"] = "route_resolved"

    requested_profile = str(route.get("profile_id") or "")
    applied_profile = str(usage.get("applied_profile") or "")
    if not applied_profile:
        mismatches.append(f"profile_absent:{surface}:{slot_id}")
    elif requested_profile and applied_profile != requested_profile:
        mismatches.append(
            f"profile_mismatch:{surface}:{slot_id}:{requested_profile}->{applied_profile}"
        )
    if str(usage.get("applied_access") or "") != "readonly":
        mismatches.append(f"readonly_access_unproven:{surface}:{slot_id}")
    if usage.get("custody_durable") is not True:
        mismatches.append(f"custody_unproven:{surface}:{slot_id}")
    if not str(usage.get("delegated_run_id") or ""):
        mismatches.append(f"delegated_run_id_absent:{surface}:{slot_id}")
    settlement = usage.get("settlement")
    settlement = settlement if isinstance(settlement, dict) else {}
    unsettled = [
        key for key in ("settled", "ledger_recorded", "project_retired")
        if settlement.get(key) is not True
    ]
    if unsettled:
        mismatches.append(
            f"session_settlement_unproven:{surface}:{slot_id}:"
            + ",".join(unsettled)
        )
    if status == "responded" and not transcript:
        mismatches.append(f"session_transcript_absent:{surface}:{slot_id}")
    for item in deltas:
        reason = str(item.get("reason") or "unclassified")
        if reason in _EXECUTION_CONTRADICTION_DELTAS:
            mismatches.append(f"capability_delta:{surface}:{slot_id}:{reason}")


def bind_execution_receipts(
    *,
    actors: list[tuple[str, dict]],
    resolved_config: dict,
    drive_root: pathlib.Path,
    live_plan_sha256: str = "",
) -> tuple[list[dict], list[str], list[dict]]:
    """Bind configured, dispatched and observed facts for every reviewer slot."""
    requested: dict[tuple[str, str], dict] = {}
    for surface, key in (("triad", "triad_slots"), ("scope", "scope_slots")):
        for row in resolved_config.get(key) or []:
            requested[(surface, str(row.get("slot_id") or ""))] = dict(row)

    keys = [(surface, str(actor.get("slot_id") or "")) for surface, actor in actors]
    key_set = set(keys)
    mismatches = [f"missing_actor:{s}:{i}" for s, i in sorted(set(requested) - key_set)]
    mismatches += [f"unexpected_actor:{s}:{i}" for s, i in sorted(key_set - set(requested))]
    mismatches += [
        f"duplicate_actor:{surface}:{slot_id}"
        for surface, slot_id in sorted(key_set)
        if keys.count((surface, slot_id)) > 1
    ]
    expected_plan_sha = str(resolved_config.get("slot_plan_sha256") or "")
    if expected_plan_sha and live_plan_sha256 != expected_plan_sha:
        mismatches.append(
            f"slot_plan_drift:{expected_plan_sha}->{live_plan_sha256 or 'unreadable'}"
        )

    receipts: list[dict] = []
    transcripts: list[dict] = []
    for surface, actor in actors:
        slot_id = str(actor.get("slot_id") or "")
        row = requested.get((surface, slot_id), {})
        route = dict(row.get("route") or {})
        status = str(actor.get("status") or "")
        prompt, response = _receipt_payloads(
            actor, drive_root=drive_root, surface=surface, slot_id=slot_id,
            mismatches=mismatches,
        )
        dispatched_slot = (
            dict(prompt.get("slot") or {}) if isinstance(prompt.get("slot"), dict) else {}
        )
        dispatched = _compare_dispatch(
            surface=surface, slot_id=slot_id, row=row,
            dispatched_slot=dispatched_slot, mismatches=mismatches,
        )
        usage = dict(response.get("usage") or {}) if isinstance(response.get("usage"), dict) else {}
        delegated_route = str(usage.get("delegated_route") or "")
        provider = str(usage.get("provider") or "")
        observed_kind = (
            "agent_session" if delegated_route or provider == "claudexor"
            else "api_chat" if provider else ""
        )
        observed_model = str(usage.get("resolved_model") or "")
        settlement = usage.get("settlement")
        settlement = dict(settlement) if isinstance(settlement, dict) else None
        observed = {
            "route_kind": observed_kind or None,
            "provider": provider or None,
            "harness": delegated_route or None,
            "model": observed_model or None,
            "profile_id": str(usage.get("applied_profile") or "") or None,
            "access": str(usage.get("applied_access") or "") or None,
            "effort": usage.get("applied_effort"),
            "delegated_run_id": str(usage.get("delegated_run_id") or "") or None,
            "custody_durable": usage.get("custody_durable"),
            "settlement": settlement,
            "output_conformance": str(usage.get("output_conformance") or "") or None,
            "verdict_method": str(usage.get("verdict_method") or "") or None,
        }
        deltas = [
            item for item in (usage.get("capability_delta") or []) if isinstance(item, dict)
        ]
        receipt = {
            "surface": surface, "slot_id": slot_id, "actor_status": status,
            "configured": row, "dispatched": dispatched, "observed": observed,
            "model_verification": "not_requested", "capability_delta": deltas,
        }
        receipts.append(receipt)

        message = dict(response.get("message") or {}) if isinstance(response.get("message"), dict) else {}
        transcript = str(message.get("session_transcript") or "")
        if transcript:
            provenance = dict(usage.get("verdict_provenance") or {})
            digest = hashlib.sha256(transcript.encode("utf-8", "replace")).hexdigest()
            redaction_rules = (
                ((actor.get("response_ref") or {}).get("redaction") or {}).get("rules")
                or []
            )
            transcript_redacted = any(
                str(item.get("path") or "").startswith("$.message.session_transcript")
                for item in redaction_rules if isinstance(item, dict)
            )
            try:
                declared_chars = int(provenance.get("raw_transcript_chars"))
            except (TypeError, ValueError):
                declared_chars = -1
            provenance_matches = declared_chars == len(transcript) and str(
                provenance.get("raw_transcript_sha256") or ""
            ) == digest
            if not provenance_matches and not transcript_redacted:
                mismatches.append(f"session_transcript_mismatch:{surface}:{slot_id}")
            transcripts.append({
                "surface": surface, "slot_id": slot_id,
                "sha256": digest, "chars": len(transcript),
                "source_redacted": transcript_redacted,
                "source_provenance_verified": provenance_matches,
                "transcript": transcript,
            })

        if not response or (status == "responded" and not usage):
            mismatches.append(f"response_receipt_absent:{surface}:{slot_id}")
        if not usage:
            continue
        expected_kind = str(route.get("kind") or "")
        expected_target = str(route.get("target_id") or "")
        if observed_kind != expected_kind:
            mismatches.append(
                f"route_kind_mismatch:{surface}:{slot_id}:"
                f"{expected_kind}->{observed_kind or 'absent'}"
            )
        if not observed_model:
            mismatches.append(f"model_absent:{surface}:{slot_id}")
        if expected_kind == "agent_session":
            _session_evidence(
                surface=surface, slot_id=slot_id, route=route, status=status,
                observed_model=observed_model, usage=usage, transcript=transcript,
                deltas=deltas, receipt=receipt, mismatches=mismatches,
            )
        elif expected_kind == "api_chat":
            from ouroboros.provider_models import provider_for_model

            expected_provider = provider_for_model(expected_target)
            if provider != expected_provider:
                mismatches.append(
                    f"provider_mismatch:{surface}:{slot_id}:"
                    f"{expected_provider}->{provider or 'absent'}"
                )
            if observed_model and _api_model_identity(expected_target) != _api_model_identity(observed_model):
                mismatches.append(
                    f"model_mismatch:{surface}:{slot_id}:{expected_target}->{observed_model}"
                )
                receipt["model_verification"] = "mismatch"
            elif observed_model:
                receipt["model_verification"] = "exact"
        applied_effort = str(usage.get("applied_effort") or "")
        if applied_effort and applied_effort != str(row.get("effort") or ""):
            mismatches.append(
                f"effort_mismatch:{surface}:{slot_id}:"
                f"{row.get('effort') or 'absent'}->{applied_effort}"
            )
    return receipts, sorted(set(mismatches)), transcripts


def finalize_contributor_outcome(
    *, outcome: dict, exit_code: int, mismatches: list[str],
) -> tuple[int, dict]:
    """Turn execution-receipt drift into the contributor lane's typed outcome.

    Nothing about WHICH files the proposal touches is consulted: the lane always
    executes the target base's review machinery (owner decision 2026-08-19), so
    there is no per-proposal trust downgrade left to apply.
    """
    if mismatches:
        exit_code = 3
        outcome = {
            **outcome,
            "status": "blocked",
            "block_reason": "execution_receipt_mismatch",
            "message": (
                "Configured reviewer slots did not match their observed execution "
                "receipts; the run is preserved as incomplete evidence."
            ),
            "execution_receipt_mismatches": mismatches,
        }
    return exit_code, outcome
