"""Manifest contributions staged with an extension and projected from its live bundle.

No PluginAPI surface: contributions are host-owned manifest data. UI assets are
captured while hashing the exact bytes, then served only from that publication.
"""
from __future__ import annotations

import copy
import hashlib
import logging
from urllib.parse import quote

from ouroboros.extension_ui_validation import validate_contributions
from ouroboros.icon_geometry import icon_spec, sanitize_svg, validate_icon
from ouroboros.skill_identity import payload_bucket, release_declaration, release_digest_from_file_digests
from ouroboros.skill_loader import _iter_payload_files

log = logging.getLogger(__name__)
HOST_ASSET_TYPES = {
    ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml", ".ico": "image/x-icon",
}


def capture_browser_sources(skill, revision: str, *, paths: set[str] | None = None,
                            max_bytes: int | None = None) -> dict[str, bytes]:
    """The page assets of the reviewed release, read while hashing it: the bytes captured are the bytes whose
    release digest (skill_identity, the skill's ``content_hash``) equals ``revision``."""
    root = skill.skill_dir.resolve()
    bucket = payload_bucket(root)
    sources, digests = {}, []
    captured_bytes = 0
    for path in _iter_payload_files(root, manifest_entry=skill.manifest.entry, manifest_scripts=skill.manifest.scripts):
        rel = path.relative_to(root).as_posix()
        capture = (rel in paths if paths is not None else
                   rel.startswith("ui/") and path.suffix in HOST_ASSET_TYPES
                   and not any(part.startswith(".") for part in rel.split("/")))
        digest, content = hashlib.sha256(), bytearray()
        with path.open("rb") as stream:
            while chunk := stream.read(64 * 1024):
                digest.update(chunk)
                if capture:
                    captured_bytes += len(chunk)
                    if max_bytes is not None and captured_bytes > max_bytes:
                        raise ValueError(f"browser files exceed {max_bytes} bytes")
                    content.extend(chunk)
        digests.append((rel, digest.digest()))
        if capture:
            sources[rel] = bytes(content)
    declaration = release_declaration(root, skill.manifest, bucket=bucket)
    if release_digest_from_file_digests(digests, declaration, bucket=bucket) != revision:
        raise ValueError("payload changed while capturing browser files; reload the reviewed revision")
    return sources


def stage_contributions(api, skill, revision: str) -> None:
    """Stage before the loader's atomic publish; invalid entries never abort register()."""
    entries, problems = validate_contributions(skill.manifest, skill_dir=skill.skill_dir)
    try:
        icon = validate_icon(skill.manifest.raw_extra.get("icon", ""))
    except ValueError as exc:
        icon = ""
        problems.append(f"icon: {exc}")
    sources = {}
    if (icon == "ui/icon.svg" or any(entry["icon"] == "ui/icon.svg" or entry["render"]["kind"] == "host" for entry in entries)):
        try:
            sources = capture_browser_sources(skill, revision)
        except (OSError, ValueError, RuntimeError) as exc:
            problems.append(f"contributes: {exc}")
        available = []
        for entry in entries:
            if entry["render"]["kind"] == "host" and entry["render"]["entry"] not in sources:
                problems.append(f"contributes.{entry['id']}: host entry is not in the captured reviewed payload")
            else:
                available.append(entry)
        entries = available
    if icon == "ui/icon.svg" or any(entry["icon"] == "ui/icon.svg" for entry in entries):
        try:
            sources["ui/icon.svg"] = sanitize_svg(sources.get("ui/icon.svg", b"")).encode("utf-8")
        except ValueError as exc:
            sources.pop("ui/icon.svg", None)
            problems.append(f"icon: {exc}; fix ui/icon.svg or use a host icon name")
    api._staged.contributions = project_contributions(
        skill.name, entries, revision, skill.manifest.name, sources=sources,
        tools=skill.manifest.raw_extra.get("tools", []))
    api._staged.contribution_problems = problems
    api._staged.host_sources = sources
    for problem in problems:
        log.warning("skill %s: %s", skill.name, problem)


def project_contributions(skill: str, entries: list[dict], revision: str, label: str = "", *,
                          sources: dict[str, bytes] | None = None, tools: list | None = None) -> list[dict]:
    """Host-owned absolute paths; authors never choose a different skill's namespace."""
    prefix = f"/api/extensions/{quote(skill, safe='')}/"
    rows = []
    for entry in entries:
        row = copy.deepcopy(entry)
        row.update(key=f"{skill}:{entry['id']}", skill=skill, skill_label=label or skill, revision=revision)
        row["icon_spec"] = icon_spec(row.get("icon", ""), (sources or {}).get("ui/icon.svg"))
        if row.get("provides") == "place":
            from ouroboros.extension_surface_names import extension_surface_name
            from ouroboros.contracts.plugin_api import ExtensionRegistrationError

            row["tools"] = []
            for tool in tools if isinstance(tools, list) else []:
                if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                    continue
                try:
                    full_name = extension_surface_name(skill, tool["name"])
                except ExtensionRegistrationError:
                    continue
                row["tools"].append({**{k: copy.deepcopy(tool[k]) for k in ("effect", "target", "conversation", "thread") if k in tool},
                                     "name": full_name})
        route = row.pop("route", "")
        if row.get("action", {}).get("verb") == "skill.request":
            route = row["action"].pop("route")
            row["action"]["url"] = prefix + quote(route, safe="/")
        if route:
            row["route_url"] = prefix + quote(route, safe="/")
        if "source" in row:
            row["source"]["url"] = prefix + quote(row["source"].pop("route"), safe="/")
        render = row["render"]
        render["url"] = (prefix + f"ui/{quote(revision, safe='')}/" + quote(render.pop("entry"), safe="/")
                         if render["kind"] == "host" else "")
        rows.append(project_place_status(row) if row.get("provides") == "place" else row)
    return rows


def project_place_status(row: dict) -> dict:
    """Expose the manifest's labels without executing its status source."""
    return {**row, "labels": {fact["id"]: fact["label"] for fact in row.get("facts", []) if isinstance(fact, dict)}}


def live_place_declarations() -> list[dict]:
    """Read published manifest facts without dispatching any connector route."""
    from ouroboros.extension_registry_state import live_contribution_projection

    return [project_place_status(row) for row in live_contribution_projection() if row.get("provides") == "place"]
