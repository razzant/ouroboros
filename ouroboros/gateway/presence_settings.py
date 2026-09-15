"""Owner-facing runtime overrides for reviewed Presence behavior skills."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import json_error, request_drive_root
from ouroboros.presence_capabilities import (
    PresenceState,
    PresenceStateError,
    load_presence_state,
    presence_state_fingerprint,
    save_presence_state,
)
from ouroboros.presence_profile import PresenceProfileError, parse_presence_profile
from ouroboros.presence_runtime import PresenceRuntimeError, PresenceRuntimeOverrides
from ouroboros.skill_loader import find_skill

log = logging.getLogger(__name__)
_REQUEST_FIELDS = frozenset({"expected_state_fingerprint", "runtime_overrides"})
_OVERRIDE_FIELDS = frozenset({"model_slot", "inline_max_rounds"})


def _runtime_projection(profile: Any, state: PresenceState) -> dict[str, Any]:
    defaults = profile.runtime_defaults
    overrides = state.runtime_overrides
    return {
        "defaults": {
            "model_slot": defaults.model_slot,
            "inline_max_rounds": defaults.inline_max_rounds,
        },
        "overrides": {
            "model_slot": overrides.model_slot,
            "inline_max_rounds": overrides.inline_max_rounds,
        },
        "state_fingerprint": presence_state_fingerprint(state),
        "workspace_root": state.workspace_root,
    }


def presence_runtime_card_projection(drive_root: Path, loaded: Any) -> dict[str, Any] | None:
    """Project reviewed defaults and owner-local overrides into one Skills card."""
    extras = getattr(loaded.manifest, "raw_extra", {})
    if not isinstance(extras, Mapping) or "presence" not in extras:
        return None
    if not loaded.review.gate_for(loaded.content_hash)["executable_review"]:
        return None
    try:
        profile = parse_presence_profile(loaded.manifest, loaded.skill_dir)
        if profile is None:
            return None
        state = load_presence_state(drive_root, loaded.name)
        return _runtime_projection(profile, state)
    except (PresenceProfileError, PresenceStateError, PresenceRuntimeError, ValueError) as exc:
        return {
            "error": str(getattr(exc, "code", "presence_runtime_unavailable")),
        }


def _parse_overrides(body: Any) -> tuple[str, Mapping[str, Any]]:
    if (not isinstance(body, Mapping) or not _REQUEST_FIELDS <= set(body)
            or set(body) - _REQUEST_FIELDS - {"workspace_root"}):
        raise ValueError("body must contain expected_state_fingerprint and runtime_overrides, with optional workspace_root")
    expected = body.get("expected_state_fingerprint")
    raw = body.get("runtime_overrides")
    if not isinstance(expected, str) or not isinstance(raw, Mapping):
        raise ValueError("expected_state_fingerprint must be a string and runtime_overrides an object")
    if set(raw) != _OVERRIDE_FIELDS:
        raise ValueError("runtime_overrides must contain exactly model_slot and inline_max_rounds")
    if "workspace_root" in body and not isinstance(body["workspace_root"], str):
        raise ValueError("workspace_root must be a string; empty clears the selection")
    return expected, raw


def _update_runtime_overrides(
    drive_root: Path,
    skill_name: str,
    body: Any,
    *,
    repo_path: str | None,
) -> dict[str, Any]:
    expected, raw = _parse_overrides(body)
    loaded = find_skill(drive_root, skill_name, repo_path=repo_path)
    if loaded is None:
        return {"error": "skill not found", "status_code": 404}
    if not loaded.review.gate_for(loaded.content_hash)["executable_review"]:
        return {"error": "presence runtime overrides require a fresh executable review", "status_code": 409}
    profile = parse_presence_profile(loaded.manifest, loaded.skill_dir)
    if profile is None:
        return {"error": "skill has no presence profile", "status_code": 404}
    state = load_presence_state(drive_root, loaded.name)
    overrides = PresenceRuntimeOverrides(
        model_slot=raw.get("model_slot"),
        inline_max_rounds=raw.get("inline_max_rounds"),
    )
    updated = replace(state, runtime_overrides=overrides)
    if "workspace_root" in body:
        from ouroboros.workspace_admission import validate_workspace_root

        workspace = validate_workspace_root(
            body["workspace_root"], system_repo_dir=Path(__file__).resolve().parents[2],
            drive_root=drive_root,
        )
        updated = replace(updated, workspace_root=str(workspace) if workspace is not None else "")
    saved = save_presence_state(
        drive_root,
        loaded.name,
        updated,
        expected_state_fingerprint=expected,
    )
    return {
        "ok": True,
        "skill": loaded.name,
        "presence_runtime": _runtime_projection(profile, saved),
    }


async def api_owner_skill_presence_runtime(request: Request) -> JSONResponse:
    """CAS-update the local runtime and workspace of one reviewed profile."""
    from ouroboros.config import get_skills_repo_path

    skill_name = str(request.path_params.get("skill") or "").strip()
    if not skill_name:
        return json_error("missing skill name", 400)
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return json_error("invalid json", 400)
    try:
        result = await asyncio.to_thread(
            _update_runtime_overrides,
            request_drive_root(request),
            skill_name,
            body,
            repo_path=get_skills_repo_path(),
        )
    except PresenceStateError as exc:
        status = 409 if getattr(exc, "code", "") == "presence_state_conflict" else 400
        return json_error(str(exc), status, code=getattr(exc, "code", "presence_state_invalid"))
    except (PresenceProfileError, PresenceRuntimeError, ValueError) as exc:
        return json_error(str(exc), 400, code=getattr(exc, "code", "presence_runtime_invalid"))
    except Exception as exc:
        log.exception("presence runtime override update failed")
        return json_error(f"{type(exc).__name__}: {exc}", 500)
    if result.get("error"):
        return JSONResponse(result, status_code=int(result.get("status_code") or 400))
    return JSONResponse(result)


__all__ = ["api_owner_skill_presence_runtime", "presence_runtime_card_projection"]
