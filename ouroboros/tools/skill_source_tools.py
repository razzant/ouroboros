"""Ouroboros's own hands on the skill sources: the same operations the Skills page uses.

``skill_sources`` reads the source list (and, on the owner's instruction or in Cyber Pro, adds,
re-trusts or removes a source). ``skill_search`` finds catalog ids in the connected sources (ClawHub
joins a text search). ``skill_install`` installs one by catalog id, never by URL: from a trusted source
the skill gets the keys it asks for (under the owner's auto-grant setting) and turns on when ready; from
a review source the configured review checks it (in Cyber Pro Ouroboros may record its own check instead).
``skill_update`` applies one update choice. Skipping without a check stays the owner's act.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

from ouroboros.reviewer_slot_config import PanelCompositionError
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

_OWNER_SOURCE = {
    "type": "object",
    "description": (
        "The owner's instruction this change carries out: {kind:'chat',ref:{chat_id,client_message_id,ts,"
        "text_sha256}}; {kind:'quiz',task_id,quiz_id}; or {kind:'mailbox',task_id,msg_id}. Not needed in Cyber Pro."
    ),
}
_MAX_RESULTS = 40


def _dump(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=1)


def _drive(ctx: ToolContext) -> pathlib.Path:
    from ouroboros.tool_access import canonical_data_root

    return canonical_data_root(ctx)


def _cyber() -> bool:
    from ouroboros.config import get_runtime_mode
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency

    return mode_has_unrestricted_agency(get_runtime_mode())


def _owner_instruction_error(ctx: ToolContext, owner_source: Any) -> str:
    if _cyber():
        return ""
    from ouroboros.owner_source import resolve_owner_source

    if resolve_owner_source(ctx, owner_source):
        return ""
    return ("this change needs the owner's instruction: pass owner_source pointing at the owner's message, "
            "answered quiz or mailbox entry in this task")


def _handle_skill_sources(ctx: ToolContext, action: str = "list", url: str = "", path: str = "", name: str = "",
                          trust: str = "", source_id: str = "", owner_source: Any = None, **_kw: Any) -> str:
    from ouroboros.skill_source_catalog import probe_source, snapshot_for
    from ouroboros.skill_sources import (
        SkillSourceError,
        load_sources,
        save_sources,
        with_added_source,
        with_changed_source,
        without_source,
    )

    drive = _drive(ctx)
    if action not in {"list", "add", "trust", "remove"}:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
            text="⚠️ SKILL_SOURCES_ERROR: action is list, add, trust or remove."))
    if action != "list":
        problem = _owner_instruction_error(ctx, owner_source)
        if problem:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ERROR",
                text=f"⚠️ SKILL_SOURCES_REFUSED: {problem}."))
        try:
            rows = load_sources()
            if action == "add":
                probe = probe_source(drive, url=url, path=path)
                if probe.get("error"):
                    return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ERROR",
                        text=f"⚠️ SKILL_SOURCES_ERROR: {probe['error']}"))
                rows, _row = with_added_source(rows, kind=str(probe["kind"]), url=url, path=path,
                                               name=name or str(probe.get("suggested_name") or "source"),
                                               trust=trust or "review")
            elif action == "trust":
                rows = with_changed_source(rows, source_id, trust=trust, confirmed=True)
            else:
                rows = without_source(rows, source_id)
            save_sources(rows)
        except SkillSourceError as exc:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ERROR",
                text=f"⚠️ SKILL_SOURCES_ERROR: {exc}"))
    out: List[Dict[str, Any]] = []
    for row in load_sources():
        _snapshot, status = snapshot_for(drive, row)
        out.append({**row, **status})
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=_dump({
        "sources": out, "note": "trusted installs without review; review reviews each skill. "
                                "A trust change never rewrites an existing install."})))


def _compact(item: Dict[str, Any], drive: pathlib.Path) -> Dict[str, Any]:
    keep = ("id", "name", "source_name", "trust", "official", "recommended", "category", "version", "withdrawn",
            "installed", "needs_keys", "type")
    out = {key: item.get(key) for key in keep if item.get(key) not in (None, "", [], False)}
    out["description"] = str(item.get("description") or "")[:240]
    out.update(digest=item.get("digest", ""), needs_keys=list(item.get("needs_keys") or []),
               permissions=list(item.get("permissions") or []), type=item.get("type", ""))
    if (item.get("installed") or {}).get("state") == "installed":
        from ouroboros.config import get_skills_repo_path
        from ouroboros.skill_loader import find_skill

        loaded = find_skill(drive, item["name"], repo_path=get_skills_repo_path())
        if loaded is not None:
            out["local_digest"] = loaded.content_hash
    return out


def _handle_skill_search(ctx: ToolContext, query: str = "", source: str = "", category: str = "", **_kw: Any) -> str:
    from ouroboros.skill_source_catalog import catalog_view, read_snapshot, refresh_source, registry_items
    from ouroboros.skill_sources import load_sources

    drive = _drive(ctx)
    rows = load_sources()
    for row in rows:  # a source never read on this computer is read once now; others refresh in the background
        if row["kind"] in {"hub", "git"} and (not source or row["id"] == source) and read_snapshot(drive, row) is None:
            refresh_source(drive, row, wait_sec=120)
    view = catalog_view(drive, rows=rows, refresh=True)
    words = [word for word in str(query or "").lower().split() if word]
    items = [item for item in view["items"]
             if (not source or item["source_id"] == source) and (not category or category in item.get("category", []))
             and all(word in f"{item['name']} {item.get('description', '')}".lower() for word in words)]
    notes = [f"{s['name']}: {s.get('error') or s.get('state')}" for s in view["sources"]
             if s.get("state") not in {"ready", "search"}]
    if words:
        for row in rows:
            if row["kind"] == "registry" and (not source or row["id"] == source):
                try:
                    items.extend(registry_items(row, " ".join(words)))
                except Exception as exc:
                    notes.append(f"{row['name']}: {exc}")
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=_dump({
        "count": len(items), "items": [_compact(item, drive) for item in items[:_MAX_RESULTS]],
        "omitted": max(0, len(items) - _MAX_RESULTS), "notes": notes, "categories": view["categories"],
        "hint": "Install with skill_install(id, reason); the id is <source>/<name>."})))


def _handle_skill_install(ctx: ToolContext, id: str = "", reason: str = "", replace: bool = False,
                          owner_source: Any = None, check: str = "review", check_notes: str = "", reviewers: List[str] | None = None,
                          review_reason: str = "", **_kw: Any) -> str:
    from ouroboros.skill_source_install import SkillInstallError, install_item, run_in_lane, start_review

    item_id = str(id or "").strip()
    if "/" not in item_id or "://" in item_id or item_id.startswith(("git@", "/", "~")):
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
            text="⚠️ SKILL_INSTALL_ERROR: id is a catalog id <source>/<name> from skill_search, never a URL."))
    if not str(reason or "").strip():
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
            text="⚠️ SKILL_INSTALL_ERROR: reason is required: say why this work needs the skill."))
    if replace:
        problem = _owner_instruction_error(ctx, owner_source)
        if problem:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ERROR",
                text=f"⚠️ SKILL_INSTALL_REFUSED: replacing another skill of the same name: {problem}."))
    drive = _drive(ctx)
    task_id = str(getattr(ctx, "task_id", "") or "")
    try:
        panel = _skill_panel(reviewers, review_reason, reason_field="review_reason") if check == "review" else None
    except PanelCompositionError as exc:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=str(exc)))
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ERROR",
            text=f"invalid review pool configuration blocks skill review: {exc}"))
    try:
        result = run_in_lane(drive, "install", item_id, lambda: install_item(
            drive, item_id, replace=bool(replace), actor="ouroboros", reason=str(reason), task_id=task_id,
            turn_on_when_ready=True, check=check, check_notes=check_notes),
            message=f"Ouroboros installs {item_id}: {str(reason)[:160]}",
            chat_id=int(getattr(ctx, "current_chat_id", 0) or 0))
    except SkillInstallError as exc:
        if exc.code in {"bad_mode", "bad_check"}:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                text="⚠️ SKILL_INSTALL_REFUSED: " + _dump(exc.payload())))
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ERROR",
            text="⚠️ SKILL_INSTALL_REFUSED: " + _dump(exc.payload())))
    if result.get("needs_review"):
        start_review(drive, result["name"], panel=panel, chosen_by="ouroboros", ctx=ctx)
        result["next"] = ("The review runs now in the skill lifecycle lane; the skill turns on when it leaves no "
                          "blockers. list_skills(name=...) shows its state.")
    acceptance = result.get("acceptance") or {}
    basis_text = {"trusted": "installs without review", "self_check": "Ouroboros self-check recorded"}
    result["message"] = (f"Installed {result.get('name')} from {result['source'].get('name')} "
                         f"({basis_text.get(result.get('basis'), 'reviewed before it runs')})"
                         f"; reason: {reason}. Keys granted: {', '.join(acceptance.get('granted_keys') or []) or 'none'}.")
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=_dump(result)))


def _handle_skill_update(ctx: ToolContext, name: str = "", mode: str = "", digest: str = "", local_digest: str = "",
                         check: str = "review", check_notes: str = "", reviewers: List[str] | None = None, reason: str = "", **_kw: Any) -> str:
    from ouroboros.skill_source_install import SkillInstallError, run_in_lane, start_review, update_installed

    skill = str(name or "").strip()
    if not skill or mode not in {"replace", "keep", "merge"}:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
            text="⚠️ SKILL_UPDATE_ERROR: name and mode (replace, keep or merge) are required."))
    drive = _drive(ctx)
    try:
        panel = _skill_panel(reviewers, reason) if check == "review" and mode != "keep" else None
    except PanelCompositionError as exc:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=str(exc)))
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ERROR",
            text=f"invalid review pool configuration blocks skill review: {exc}"))
    try:
        result = run_in_lane(drive, "update", skill, lambda: update_installed(
            drive, skill, mode=mode, digest=str(digest or ""), local_digest=str(local_digest or ""), actor="ouroboros",
            check=check, check_notes=check_notes,
            task_id=str(getattr(ctx, "task_id", "") or "")), message=f"Ouroboros updates {skill} ({mode})",
            chat_id=int(getattr(ctx, "current_chat_id", 0) or 0))
    except SkillInstallError as exc:
        if exc.code in {"bad_mode", "bad_check"}:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                text="⚠️ SKILL_UPDATE_REFUSED: " + _dump(exc.payload())))
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ERROR",
            text="⚠️ SKILL_UPDATE_REFUSED: " + _dump(exc.payload())))
    if result.get("needs_review"):
        start_review(drive, skill, panel=panel, chosen_by="ouroboros", ctx=ctx)
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=_dump(result)))


def _skill_panel(reviewers: Any, reason: str, *, reason_field: str = "reason") -> Any:
    from ouroboros.reviewer_slot_config import compose_review_panel
    if reviewers is not None and (not isinstance(reviewers, list) or any(not isinstance(row, str) for row in reviewers)):
        raise PanelCompositionError("REVIEW_PANEL_ARGUMENT: reviewers must be a list of enabled row ids or handles.")
    return compose_review_panel(reviewers or (), reason, surface="skill", reason_field=reason_field)


_CHECK_PROPERTIES = {
    "reviewers": {"type": "array", "items": {"type": "string"}, "description": "Enabled row ids/handles; omitted means the whole pool. Blocking widens to all ticked reviewers; Advisory requires at least one ticked reviewer and a reason when narrowed. Unticked advisers are heard, not counted. Cyber Pro composes freely."},
    "check": {"type": "string", "enum": ["review", "self"], "description":
              "review (default): the configured reviewers check the exact release, and it runs when they leave no "
              "blockers; outside Cyber Pro this is the only check, and only the owner skips a review. self: Cyber Pro "
              "only, after you read the exact release and record why it may run; review findings stay evidence. "
              "A trusted source ignores this choice."},
    "check_notes": {"type": "string", "description": "self requires 1..2000 characters recording what you read and why."},
}

_SCHEMAS = {
    "skill_sources": {
        "description": (
            "List the skill sources this install reads, with each one's trust (trusted: installs without review; "
            "review: each skill is reviewed) and read state. action add|trust|remove changes the list: only on the "
            "owner's instruction (owner_source), except in Cyber Pro."
        ),
        "properties": {
            "action": {"type": "string", "enum": ["list", "add", "trust", "remove"], "description": "Default list."},
            "url": {"type": "string", "description": "add: a git repository address (https or ssh)."},
            "path": {"type": "string", "description": "add: a folder on this computer."},
            "name": {"type": "string", "description": "add: the source's name."},
            "trust": {"type": "string", "enum": ["trusted", "review"]},
            "source_id": {"type": "string", "description": "trust/remove: the source id."},
            "owner_source": _OWNER_SOURCE,
        },
    },
    "skill_search": {
        "description": (
            "Search the skills offered by the connected sources (ClawHub joins a text search). Returns catalog ids "
            "for skill_install with source, trust, digest, keys, permissions, type and installed state; "
            "local_digest names an installed copy's current bytes. skill_install, skill_update and skill_sources "
            "load with enable_tools."
        ),
        "properties": {
            "query": {"type": "string", "description": "Words to match in names and descriptions; empty lists all."},
            "source": {"type": "string", "description": "Optional source id."},
            "category": {"type": "string", "description": "Optional category id."},
        },
    },
    "skill_install": {
        "description": (
            "Install one skill by catalog id from skill_search (never a URL). From a trusted source it gets the keys "
            "it asks for under the owner's auto-grant setting and turns on when ready; from a review source it "
            "installs now and the configured review decides (in Cyber Pro you may record your own check instead). "
            "After installing, tell the owner what you installed, why, and which keys it received."
        ),
        "properties": {
            "id": {"type": "string", "description": "Catalog id <source>/<name>."},
            "reason": {"type": "string", "description": "Why this work needs the skill."},
            "review_reason": {"type": "string", "description": "Why these reviewers when you name fewer than the whole pool; Advisory requires it then."},
            "replace": {"type": "boolean", "description": "Replace another skill of the same name (owner's instruction)."},
            "owner_source": _OWNER_SOURCE,
            **_CHECK_PROPERTIES,
        },
        "required": ["id", "reason"],
    },
    "skill_update": {
        "description": (
            "Update one installed skill to a release found by skill_search. Modes: replace replaces local edits "
            "(the previous copy is kept as .previous), keep skips "
            "this release, merge merges it into local changes. digest names the release you saw; local_digest "
            "detects edits since you looked. From a review source the configured review decides (Cyber Pro: your own check)."
        ),
        "properties": {
            "name": {"type": "string"},
            "mode": {"type": "string", "enum": ["replace", "keep", "merge"]},
            "reason": {"type": "string", "description": "Why these reviewers; required for a narrowed Advisory panel."},
            "digest": {"type": "string", "description": "The candidate release digest from skill_search."},
            "local_digest": {"type": "string", "description": "The installed copy's local_digest from skill_search."},
            **_CHECK_PROPERTIES,
        },
        "required": ["name", "mode"],
    },
}
def _schema(name: str) -> Dict[str, Any]:
    spec = _SCHEMAS[name]
    return {"name": name, "description": spec["description"],
            "parameters": {"type": "object", "properties": spec["properties"],
                           "required": spec.get("required", [])}}


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("skill_sources", _schema("skill_sources"), _handle_skill_sources,
                  is_code_tool=False, timeout_sec=300, safety_policy="check"),
        ToolEntry("skill_search", _schema("skill_search"), _handle_skill_search,
                  is_code_tool=False, timeout_sec=300, safety_policy="skip"),
        ToolEntry("skill_install", _schema("skill_install"), _handle_skill_install,
                  is_code_tool=False, timeout_sec=1800, safety_policy="check"),
        ToolEntry("skill_update", _schema("skill_update"), _handle_skill_update,
                  is_code_tool=False, timeout_sec=1800, safety_policy="check"),
    ]


__all__ = ["get_tools"]
