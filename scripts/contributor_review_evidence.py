"""Route-neutral execution evidence for ``run_external_review --contributor``."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import time
import zipfile


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
        "subagent_id": str(dispatched_slot.get("subagent_id") or "") or None,
    }
    if not dispatched_slot:
        mismatches.append(f"prompt_receipt_absent:{surface}:{slot_id}")
        return dispatched
    expected = {
        "route": kind,
        "subagent_id": str(row.get("subagent_id") or ""),
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


def _final_session_settlements(drive_root: pathlib.Path) -> dict[str, dict] | None:
    """Project session settlement from custody after every panel slot finished."""
    try:
        from ouroboros.delegate_custody import custody_log_unreadable, replay

        if custody_log_unreadable(drive_root):
            return None
        rows = replay(drive_root)
    except Exception:
        return None
    return {
        str(run_id): {
            "settled": row.settled,
            "ledger_recorded": row.ledger_recorded,
            "project_retired": not row.project_owned and not row.project_persistent,
            "project_persistent": row.project_persistent,
            "bound_at": "panel_complete_custody_replay",
        }
        for run_id, row in rows.items()
    }


def bind_execution_receipts(
    *,
    actors: list[tuple[str, dict]],
    resolved_config: dict,
    drive_root: pathlib.Path,
    live_plan_sha256: str = "",
) -> tuple[list[dict], list[str], list[dict]]:
    """Bind configured, dispatched and observed facts for every reviewer slot."""
    requested: dict[tuple[str, str], dict] = {}
    for row in resolved_config.get("pool_slots") or []:
        requested[("pool", str(row.get("slot_id") or ""))] = dict(row)

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
    final_settlements = _final_session_settlements(drive_root)
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
        expected_kind = str(route.get("kind") or "")
        delegated_run_id = str(usage.get("delegated_run_id") or "")
        if expected_kind == "agent_session":
            final_settlement = (
                final_settlements.get(delegated_run_id)
                if final_settlements is not None else None
            )
            usage["settlement"] = final_settlement
            if final_settlements is None:
                mismatches.append(f"session_custody_replay_unreadable:{surface}:{slot_id}")
            elif not final_settlement:
                mismatches.append(
                    f"session_custody_settlement_absent:{surface}:{slot_id}:"
                    f"{delegated_run_id or 'absent'}"
                )
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
    executes the installed body's review flow and rules, never the proposal's
    copy (D31), so there is no per-proposal trust downgrade left to apply.
    """
    if mismatches:
        original_block_reason = str(outcome.get("block_reason") or "")
        original_message = str(outcome.get("message") or "")
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
            **({"original_block_reason": original_block_reason}
               if original_block_reason else {}),
            **({"original_message": original_message} if original_message else {}),
        }
    return exit_code, outcome


# The wrapper's exit vocabulary and the contributor profile the packet declares
# (``run_external_review`` pins the same literals).
CONTRIBUTOR_PROFILE = "external_pr_readiness"
EXIT_CLASS = {0: "passed", 1: "genuine_review_block", 3: "infrastructure"}


def _json_text(value) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _write_json(path: pathlib.Path, value) -> None:
    path.write_text(_json_text(value) + "\n", encoding="utf-8")


_NATIVE_SEPARATORS = tuple(sep for sep in (os.sep, os.altsep) if sep)


def _public_text(text: str, replacements: list[tuple[str, str]]) -> str:
    """Machine-local roots become their placeholders. A value that IS a path under
    one of them (one line, the root then a separator) also gets posix separators,
    so the packet reads the same on every OS; other text keeps its characters."""
    is_path = "\n" not in text and any(
        raw and text.startswith(raw) and text[len(raw):len(raw) + 1] in ("", *_NATIVE_SEPARATORS)
        for raw, _replacement in replacements
    )
    for raw, replacement in replacements:
        if raw:
            text = text.replace(raw, replacement)
    if is_path:
        for sep in _NATIVE_SEPARATORS:
            text = text.replace(sep, "/")
    return text


def replace_public_paths(value, replacements: list[tuple[str, str]]):
    if isinstance(value, dict):
        return {
            str(key): replace_public_paths(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [replace_public_paths(item, replacements) for item in value]
    if isinstance(value, str):
        return _public_text(value, replacements)
    return value


def public_projection(value, *, replacements: list[tuple[str, str]]):
    """Apply the runtime secret scrubber and remove machine-local path prefixes."""
    from ouroboros.observability import redact_projection

    redacted = redact_projection(value).value
    return replace_public_paths(redacted, replacements)


def contributor_result(exit_code: int) -> str:
    """The exit code is the whole input: no proposal fact downgrades a result."""
    if exit_code != 0:
        return "BLOCKED" if exit_code == 1 else "INCOMPLETE"
    return "READY_FOR_INTEGRATION"


def write_contributor_packet(
    *,
    output_dir: pathlib.Path,
    snapshot: dict,
    resolved_config: dict,
    outcome: dict,
    exit_code: int,
    evidence_refs: list[dict],
    cost_report: dict,
    elapsed_sec: float,
    seats: list[dict],
    review_record: dict,
    execution_receipts: list[dict],
    execution_mismatches: list[str],
    session_transcripts: list[dict],
    degraded_reasons: list[str],
    replacements: list[tuple[str, str]],
) -> pathlib.Path:
    result = contributor_result(exit_code)
    telemetry_limitations = [
        f"{item.get('surface')}:{item.get('slot_id')}:observed_model_is_display_label"
        for item in execution_receipts
        if item.get("model_verification") == "observed_display_label"
    ]
    public_transcripts = public_projection(session_transcripts, replacements=replacements)
    for item in public_transcripts:
        transcript = str(item.get("transcript") or "")
        item["chars"] = len(transcript)
        item["sha256"] = hashlib.sha256(
            transcript.encode("utf-8", "replace")
        ).hexdigest()
    public_snapshot = {
        key: value
        for key, value in snapshot.items()
        if key not in ("patch", "installed_head_sha")
    }
    evidence = {
        "schema_version": 3,
        "review_profile": CONTRIBUTOR_PROFILE,
        "result": result,
        "complete": exit_code == 0,
        "exit_code": exit_code,
        "exit_class": EXIT_CLASS.get(exit_code, "unknown"),
        "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "snapshot": public_snapshot,
        "review_config": resolved_config,
        "review_execution": {
            "receipts": execution_receipts,
            "mismatches": execution_mismatches,
            "consistent": not execution_mismatches,
            "telemetry_limitations": telemetry_limitations,
            "session_transcript_artifacts": [
                {key: value for key, value in item.items() if key != "transcript"}
                for item in public_transcripts
            ],
            "effort_note": (
                "Configured effort is recorded under configured slots. Applied effort "
                "is null unless the execution route exposes it."
            ),
        },
        "review_completeness": {
            "contract": "production_pool_quorum_plus_coupling",
            "degraded_reasons": list(degraded_reasons),
        },
        "advisory": {
            "included": False,
            "reason": "excluded_by_external_pr_readiness_profile",
        },
        "release_metadata": {
            "contributor_version_bump_required": False,
            "owner": "maintainer_final_landing",
            "final_production_review_required": True,
        },
        "trust": {
            "execution_receipts_consistent": not execution_mismatches,
            # Diagnostic evidence only, never a gate (D31).
            "review_substrate_changed": snapshot.get("review_substrate_changed", []),
            "installed_body_execution": {
                "statement": (
                    "The installed body's review flow and rules ran this review: the "
                    "wrapper ran from a clean checkout that does not contain the "
                    "proposal, and the review operation froze base..head as a subject "
                    "it only reads (D31)."
                ),
                "executing_checkout_head": snapshot.get("installed_head_sha"),
                "rules_source": (review_record.get("checklist") or {}).get("rules_source"),
            },
            "note": (
                "Contributor evidence is not merge authorization or cryptographic "
                "proof of execution."
            ),
        },
        "production_outcome": outcome,
        "review_record": review_record,
        "raw_evidence_refs": evidence_refs,
        "cost_report": cost_report,
        "budget": {
            "run_cap_usd": (resolved_config.get("data_isolation") or {}).get("run_cap_usd"),
            "authority": "isolated_review_ledger",
            "note": ("The run cap is the whole global limit of a ledger that starts empty and sees no "
                     "host spend or concurrent host work; agent-session seats are recorded at settlement."),
        },
        "elapsed_sec": round(elapsed_sec, 1),
    }
    public_evidence = public_projection(evidence, replacements=replacements)
    public_seats = public_projection(list(seats), replacements=replacements)

    evidence_path = output_dir / "review-evidence.json"
    outcome_path = output_dir / "outcome.json"
    full_output_path = output_dir / "full-output.txt"
    _write_json(evidence_path, public_evidence)
    _write_json(outcome_path, public_projection({"exit_code": exit_code, "outcome": outcome},
                                                 replacements=replacements))
    sep = "=" * 80
    full_output = "\n".join([
        sep, "CONTRIBUTOR REVIEW EVIDENCE", sep,
        _json_text(public_evidence),
        sep, "REVIEW POOL SEAT RECORDS (ledger rows with retained answers, full, redacted)", sep,
        _json_text(public_seats),
        sep, "AGENT SESSION TRANSCRIPTS (full, redacted)", sep,
        _json_text(public_transcripts),
    ])
    full_output_path.write_text(full_output + "\n", encoding="utf-8")
    packet_path = output_dir / "review-packet.zip"
    with zipfile.ZipFile(packet_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (evidence_path, outcome_path, full_output_path):
            archive.write(path, arcname=path.name)
    return packet_path
