"""Model-facing skill selection over the shared lifecycle summary.

Without a name, ``list_skills`` pages compact whole records under its ordinary
result cap. A name returns that skill's complete summary row plus the
``read_file`` arguments of the manifest the loader parsed: instruction text is
read there, never copied here. Every call is a fresh read; ``snapshot`` only
detects skills added or removed between pages. A named row longer than the cap
rides the loop's generic result source like any other tool result.
"""
from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from ouroboros.tool_capabilities import tool_result_limit

LIST_SKILLS_SCHEMA = {
    "name": "list_skills",
    "description": (
        "Find installed skills, including disabled, unreviewed and broken ones. Without name: "
        "compact pages of whole records with state and purpose, trigger, model-experience and "
        "tool-name previews; omitted counts what a preview left out. available_for_execution is "
        "SCRIPT-only; extension liveness is desired_live/live_loaded. Call again with the "
        "returned next; SKILLS_SNAPSHOT_CHANGED means restart at offset 0. With name: that "
        "skill's full diagnostics and manifest.read, the read_file call for its SKILL.md or "
        "skill.json instructions. Read-only; each call reads current state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Exact skill name from the index; returns its full details."},
            "offset": {"type": "integer", "minimum": 0, "description": "Index page offset; default 0."},
            "snapshot": {"type": "string", "description": "snapshot from the previous index page."},
        },
        "required": [],
    },
}

# Preview bounds; list_skills(name=...) returns every full value.
_PREVIEW_CHARS = {"type": 80, "version": 80, "description": 200, "when_to_use": 160,
                  "what_model_sees": 160, "token_effect": 160, "load_error": 200,
                  "live_reason": 120}
_PREVIEW_TOOLS = 6
_PREVIEW_TOOL_CHARS = 80
_STATE = ("name", "source", "enabled", "review_status", "review_stale")
_COUNTS = ("count", "available", "blocked_by_grants", "pending_review", "blocker_review",
           "warning_review", "broken")


def _encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _compact(row: dict) -> dict:
    """State facts plus bounded previews, with what each preview omitted."""
    out = {key: row.get(key) for key in _STATE}
    out["ready"] = bool((row.get("readiness") or {}).get("ready"))
    experience = row.get("model_experience")
    experience = experience if isinstance(experience, dict) else {}
    texts = {"type": row.get("type"), "version": row.get("version"),
             "description": row.get("description"), "when_to_use": row.get("when_to_use"),
             "what_model_sees": experience.get("what_model_sees"),
             "token_effect": experience.get("token_effect"),
             "load_error": row.get("load_error")}
    # Script executability and extension liveness are different facts.
    if row.get("type") == "script":
        out["available_for_execution"] = row.get("available_for_execution")
    elif row.get("type") == "extension":
        out.update(desired_live=row.get("desired_live"), live_loaded=row.get("live_loaded"),
                   process=row.get("process"))
        texts["live_reason"] = row.get("live_reason")
    omitted = {}
    for key, value in texts.items():
        text = " ".join(str(value or "").split())
        out[key] = text[:_PREVIEW_CHARS[key]]
        if len(text) > _PREVIEW_CHARS[key]:
            omitted[key] = len(text) - _PREVIEW_CHARS[key]
    tools = [str(item.get("name") or "") for item in row.get("tool_surfaces") or []
             if isinstance(item, dict)]
    out["tools"] = []
    for index, name in enumerate(tools[:_PREVIEW_TOOLS]):
        out["tools"].append(name[:_PREVIEW_TOOL_CHARS])
        if len(name) > _PREVIEW_TOOL_CHARS:
            omitted[f"tools[{index}]"] = len(name) - _PREVIEW_TOOL_CHARS
    if len(tools) > _PREVIEW_TOOLS:
        omitted["tools"] = len(tools) - _PREVIEW_TOOLS
    if omitted:
        out["omitted"] = omitted
    return out


def _named(rows: list, name: str) -> dict:
    matches = [row for row in rows if row.get("name") == name]
    if not matches:
        return {"found": False, "name": name,
                "message": "No installed skill has this exact name; list_skills() pages every name."}
    out = {"found": True, "name": name, "skills": matches}
    if len(matches) > 1:
        out.update(ok=False, error={"code": "SKILL_IDENTITY_COLLISION", "message": (
            "Several directories claim this name, so none is selected as its manifest; "
            "each load_error names them and the repair.")})
    elif not matches[0].get("manifest_file"):
        out.update(ok=False, error={"code": "SKILL_MANIFEST_UNREADABLE", "message": (
            "The manifest could not be read or parsed; load_error has the cause.")})
    else:
        out["manifest"] = {"read": {"tool": "read_file", "arguments": {
            "root": "skill_payload", "bucket": matches[0].get("location"),
            "skill_name": name, "path": matches[0]["manifest_file"]}}}
    return out


def list_skills_payload(summary: dict, *, name: str = "", offset: int = 0,
                        snapshot: str = "") -> dict:
    """Project one fresh ``summarize_skills`` result; other consumers keep it whole."""
    if not isinstance(name, str) or not isinstance(snapshot, str):
        raise ValueError("name and snapshot must be strings")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    name = name.strip()
    if name and (offset or snapshot):
        raise ValueError("name selects one skill; offset and snapshot page the index without name")
    rows = sorted(summary.get("skills") or [],
                  key=lambda row: (str(row.get("name") or ""), str(row.get("location") or "")))
    if name:
        return _named(rows, name)
    # Membership only: readiness, review and payload edits never refuse a page.
    token = sha256(_encode([[row.get("name"), row.get("location")] for row in rows])
                   .encode("utf-8")).hexdigest()[:16]
    start = min(offset, len(rows))
    out = {key: summary.get(key) for key in _COUNTS}
    out.update(offset=start, returned=0, next=None, snapshot=token, skills=[])
    if snapshot and snapshot != token:
        out.update(ok=False, error={"code": "SKILLS_SNAPSHOT_CHANGED", "message": (
            "Skills were added or removed; no mixed page was returned; restart at offset=0.")})
        return out
    if not rows:
        out["hint"] = (
            "No skills are installed. Install skills into the data plane, or point "
            "OUROBOROS_SKILLS_REPO_PATH at a local checkout in Settings → "
            "Behavior → External Skills Repo."
        )
    # Whole records only; the reserve leaves room for host notes on the result.
    budget = tool_result_limit("list_skills") - 1000 - len(
        _encode({**out, "next": {"offset": len(rows), "snapshot": token}}))
    page = []
    for row in rows[start:]:
        record = _compact(row)
        cost = len(_encode(record)) + 1
        if page and cost > budget:
            break
        page.append(record)
        budget -= cost
    end = start + len(page)
    out.update(returned=len(page), skills=page,
               next={"offset": end, "snapshot": token} if end < len(rows) else None)
    return out
