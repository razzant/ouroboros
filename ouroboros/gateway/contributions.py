"""Live contribution discovery and revision-bound host assets; passive HTTP reads.

The transport-free TypedDicts are re-exported by gateway.contracts. Extension
rows come from one locked registry snapshot; instruction rows use the installed
catalog's enabled/review gate. No GET loads code, reconciles, writes or fetches.
"""
from __future__ import annotations

import asyncio
import copy
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TypedDict

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

try:
    from typing import NotRequired, Required
except ImportError:  # pragma: no cover
    from typing_extensions import NotRequired, Required


class UiContribution(TypedDict, total=False):
    key: Required[str]
    skill: Required[str]
    skill_label: Required[str]
    id: Required[str]
    placement: str
    provides: str
    label: Required[str]
    title: Required[str]
    icon: Required[str]
    icon_spec: Dict[str, str]
    order: Required[int]
    action: Dict[str, Any]
    source: Dict[str, Any]
    runtime: str
    route_url: str
    destination: str
    scope: str
    wakes: List[str]
    settings: str
    read_only: str
    tools: List[Dict[str, Any]]
    languages: List[str]
    formats: List[str]
    install_size: str
    max_bytes: int
    max_seconds: int
    render: Required[Dict[str, str]]
    revision: Required[str]


class UiContributionsResponse(TypedDict):
    contributions: List[UiContribution]
    offers: Dict[str, List[Dict[str, Any]]]
    preferences: Dict[str, Any]
    voice: Dict[str, Dict[str, Any]]  # kind -> {key, available, reason}; never falls back
    icons: List[str]
    problems: Dict[str, List[str]]


def _uninstalled_builtin_offers(installed: set) -> List[Dict[str, Any]]:
    """Dictation engines of built-ins this install has not installed (the bundled ``skills/`` folder, this host only):
    the chooser sends the owner to Browse, where the built-in source installs them; nothing here installs."""
    from ouroboros.extension_ui_validation import validate_contributions
    from ouroboros.skill_builtin import applies_here, bundled_skills_dir
    from ouroboros.skill_identity import _parse_manifest

    root, rows = bundled_skills_dir(), []
    for folder in sorted(root.iterdir()) if root.is_dir() else []:
        if not folder.is_dir() or folder.name.startswith(".") or folder.name in installed or not applies_here(folder):
            continue
        manifest = _parse_manifest(folder)
        if manifest is None:
            continue
        clean, _warnings = validate_contributions(manifest)
        rows.extend({"skill": folder.name, "skill_label": manifest.name, "id": item["id"], "label": item["label"],
                     "title": item["title"], "destination": item["destination"],
                     "install_size": item.get("install_size", ""), "enabled": False, "installed": False}
                    for item in clean if item.get("provides") == "stt")
    return rows


def contribution_catalog(drive_root, repo_path=None, *, skills=None, live=None) -> dict:
    from ouroboros.extension_contributions import project_contributions
    from ouroboros.extension_registry_state import live_contribution_problems, live_contribution_projection
    from ouroboros.extension_ui_validation import CONTRIBUTION_ICONS, validate_contributions
    from ouroboros.gateway.ui_preferences import DEFAULT_UI_PREFERENCES, _stored_preferences
    from ouroboros.skill_loader import discover_skills, skill_conflict_status

    if skills is None:
        skills = discover_skills(drive_root, repo_path=repo_path)
    rows = live_contribution_projection() if live is None else list(live.get("contributions", []))
    live_names = {row["skill"] for row in rows}
    problems, offers = live_contribution_problems(), []
    for skill in skills:
        if skill.load_error or skill.identity_collision:
            continue
        clean, warnings = validate_contributions(skill.manifest)
        if warnings:
            problems[skill.name] = list(dict.fromkeys(problems.get(skill.name, []) + warnings))
        if (skill.manifest.is_instruction() and skill.enabled
                and skill.review.gate_for(skill.content_hash)["executable_review"]
                and not skill_conflict_status(skill, skills)):
            rows.extend(project_contributions(skill.name, clean, skill.content_hash, skill.manifest.name))
            live_names.add(skill.name)
        if skill.source == "native" and skill.name not in live_names:
            offers.extend({"skill": skill.name, "skill_label": skill.manifest.name, "id": item["id"],
                           "label": item["label"], "title": item["title"], "destination": item["destination"],
                           "install_size": item.get("install_size", ""), "enabled": False, "installed": True}
                          for item in clean if item.get("provides") == "stt")
    offers.extend(_uninstalled_builtin_offers({skill.name for skill in skills}))
    try:
        prefs = _stored_preferences(drive_root / "state" / "ui_preferences.json")
    except ValueError:  # a stored value this release refuses (a hand edit, a newer release's) never fails the Skills page
        prefs = copy.deepcopy(DEFAULT_UI_PREFERENCES)
    voice = {}
    for kind in ("stt", "tts"):
        key = prefs["voice"][kind]
        available = any(row["key"] == key and row.get("provides") == kind for row in rows)
        voice[kind] = {"key": key, "available": available,
                       "reason": "" if available else ("provider_unavailable" if key else "not_selected")}
    return {"contributions": sorted(rows, key=lambda row: (row["order"], row["key"])),
            "offers": {"stt": offers}, "preferences": {key: prefs[key] for key in ("voice", "contributions")},
            "voice": voice, "icons": list(CONTRIBUTION_ICONS), "problems": problems}


async def api_ui_contributions(request: Request) -> Response:
    from starlette.responses import JSONResponse
    from ouroboros.config import get_skills_repo_path
    from ouroboros.gateway._helpers import request_drive_root

    payload = await asyncio.to_thread(contribution_catalog, request_drive_root(request), get_skills_repo_path())
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


async def api_extension_host_asset(request: Request) -> Response:
    from pathlib import PurePosixPath
    from starlette.responses import Response
    from ouroboros.extension_contributions import HOST_ASSET_TYPES
    from ouroboros.extension_registry_state import live_host_asset

    params = request.path_params
    path = str(params.get("path") or "")
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
    media_type = HOST_ASSET_TYPES.get(PurePosixPath(path).suffix)
    if PurePosixPath(path).suffix.lower() == ".svg":
        headers["Content-Security-Policy"] = "default-src 'none'; style-src 'unsafe-inline'; sandbox"
    if (not path.startswith("ui/") or not media_type or "\\" in path or "\0" in path
            or any(part in {"", ".", ".."} or part.startswith(".") for part in path.split("/"))):
        return Response(status_code=404, headers=headers)
    content = live_host_asset(str(params["skill"]), str(params["revision"]), path)
    if content is None:
        return Response(status_code=404, headers=headers)
    return Response(content, media_type=media_type, headers=headers)


class UiPreferencesResponse(TypedDict):
    ok: NotRequired[bool]
    voice: dict[str, str]
    contributions: dict[str, list[str]]
    widget_order: list[str]
    widget_start_mode: dict[str, Literal["auto", "manual", "retain"]]  # owner per-card launch-policy override
    widget_size: dict[str, dict[str, int]]  # owner Widgets card width: {w: masonry columns the card spans, 12 = full width; h: 0}
    nested_subagents_expanded: bool
    sidebar_width: int  # px; 0 = CSS default (resizable side sections, v6.33.0)
    project_panel_width: int  # px; 0 = CSS default
    project_seen_revision: dict[str, int]  # monotonic paint ACK per active Project
    main_seen_revision: Optional[int]  # Main's monotonic paint ACK; null until a client first sets it
    read_ahead_revisions: dict[str, list[list[int]]]  # per chat id: revisions read in full in another room (host-owned)
    welcome: dict[str, str]  # install-wide empty-Main UI copy: mode default|hidden|custom and plain text
    model_favorites: list[str]  # the chat model picker's starred routes, source included (never an allowlist)

