"""Translate staged OpenClaw skill metadata into Ouroboros ``SKILL.md`` with preserved provenance."""

from __future__ import annotations

import json
import logging
import pathlib
import shutil
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS
from ouroboros.contracts.skill_manifest import (
    SKILL_MANIFEST_SCHEMA_VERSION,
    SkillManifest,
    SkillManifestError,
    parse_skill_manifest_text,
)
from ouroboros.marketplace.clawhub import (
    _coerce_str_list,
    _extract_metadata_openclaw as _extract_metadata_block,
)
from ouroboros.marketplace.install_specs import install_specs_hash, normalize_install_specs
from ouroboros.utils import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)


_ALLOWED_RUNTIME_BINS = frozenset({"python", "python3", "bash", "node"})
_NAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_MAX_SLUG_LEN = 64
ADAPTER_VERSION = "5.5.0"


@dataclass
class AdapterResult:
    """Outcome of ``adapt_openclaw_skill``."""

    ok: bool
    sanitized_name: str
    target_dirname: str
    manifest: Optional[SkillManifest] = None
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    translated_frontmatter: Dict[str, Any] = field(default_factory=dict)
    original_frontmatter: Dict[str, Any] = field(default_factory=dict)
    original_body: str = ""
    is_plugin: bool = False


def sanitize_clawhub_slug(slug: str) -> str:
    cleaned = (slug or "").strip()
    if not cleaned:
        return "_clawhub_skill"
    cleaned = cleaned.replace("/", "__").replace("\\", "__")
    cleaned = _NAME_SAFE_RE.sub("_", cleaned)
    cleaned = cleaned.strip("._")
    if not cleaned:
        return "_clawhub_skill"
    return cleaned[:_MAX_SLUG_LEN]


def _read_skill_md(staging_dir: pathlib.Path) -> tuple[str, str, Dict[str, Any]]:
    skill_md = staging_dir / "SKILL.md"
    skill_json = staging_dir / "skill.json"
    if skill_md.is_file():
        text = skill_md.read_text(encoding="utf-8")
        manifest = parse_skill_manifest_text(text)
        return text, manifest.body, _manifest_frontmatter_dict(manifest)
    if skill_json.is_file():
        text = skill_json.read_text(encoding="utf-8")
        manifest = parse_skill_manifest_text(text)
        return text, "", _manifest_frontmatter_dict(manifest)
    raise SkillManifestError(
        "staged skill has neither SKILL.md nor skill.json after fetcher validation"
    )


def _manifest_frontmatter_dict(manifest: SkillManifest) -> Dict[str, Any]:
    front: Dict[str, Any] = {
        "name": manifest.name,
        "description": manifest.description,
        "version": manifest.version,
        "type": manifest.type,
        "when_to_use": manifest.when_to_use,
        "requires": list(manifest.requires),
        "os": manifest.os,
        "runtime": manifest.runtime,
        "timeout_sec": manifest.timeout_sec,
        "env_from_settings": list(manifest.env_from_settings),
        "scripts": [dict(s) for s in manifest.scripts],
        "entry": manifest.entry,
        "permissions": list(manifest.permissions),
        "conflicts": list(manifest.conflicts),
        "ui_tab": dict(manifest.ui_tab) if manifest.ui_tab else None,
        # CPL-7: preserve the author's Model Experience prose across adaptation.
        "model_experience": dict(manifest.model_experience) if manifest.model_experience else None,
    }
    front.update(manifest.raw_extra or {})
    return front

def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return str(value)


def _openclaw_compat_snapshot(
    front: Dict[str, Any],
    metadata_block: Dict[str, Any],
    warnings: List[str],
) -> Dict[str, Any]:
    requires = metadata_block.get("requires") if isinstance(metadata_block, dict) else {}
    if not isinstance(requires, dict):
        requires = {}
    command_fields = {
        key: front.get(key)
        for key in (
            "user-invocable",
            "disable-model-invocation",
            "command-dispatch",
            "command-tool",
            "command-arg-mode",
            "argument-hint",
            "arguments",
        )
        if key in front
    }
    if requires.get("config"):
        warnings.append(
            "OpenClaw metadata declares requires.config gates. Ouroboros preserves them in provenance but does not treat them as runtime permissions or auto-enable conditions."
        )
    if metadata_block.get("always") is True:
        warnings.append(
            "OpenClaw metadata declares always=true. Ouroboros ignores it: marketplace installs still require review and explicit enablement."
        )
    return {
        "adapter_version": ADAPTER_VERSION,
        "metadata_openclaw": _json_safe(metadata_block),
        "requires": {
            "bins": _coerce_str_list(requires.get("bins")),
            "anyBins": _coerce_str_list(requires.get("anyBins")),
            "env": _coerce_str_list(requires.get("env")),
            "config": _coerce_str_list(requires.get("config")),
        },
        "skill_key": str(metadata_block.get("skillKey") or "").strip(),
        "emoji": str(metadata_block.get("emoji") or "").strip(),
        "homepage": str(metadata_block.get("homepage") or front.get("homepage") or front.get("website") or "").strip(),
        "primary_env": str(metadata_block.get("primaryEnv") or "").strip(),
        "always": metadata_block.get("always") is True,
        "command_fields": _json_safe(command_fields),
    }


def _normalise_os(value: Any) -> str:
    items = _coerce_str_list(value)
    if not items:
        return "any"
    lowered = {x.lower() for x in items}
    aliases = {"macos": "darwin", "win32": "windows", "win": "windows"}
    normalised = {aliases.get(x, x) for x in lowered}
    if normalised >= {"darwin", "linux", "windows"}:
        return "any"
    if len(normalised) == 1:
        only = next(iter(normalised))
        return only if only in {"darwin", "linux", "windows"} else "any"
    # Mixed OS values stay visible; later validation may fall back to "any".
    return ",".join(sorted(normalised))


def _detect_runtime(
    metadata_block: Dict[str, Any],
    staging_dir: pathlib.Path,
    warnings: List[str],
) -> str:
    requires = metadata_block.get("requires") or {}
    if not isinstance(requires, dict):
        requires = {}
    bins = _coerce_str_list(requires.get("bins") or requires.get("anyBins"))
    declared = [b.lower() for b in bins if b]
    declared_in_allowlist = [b for b in declared if b in _ALLOWED_RUNTIME_BINS]
    declared_outside = [b for b in declared if b not in _ALLOWED_RUNTIME_BINS]
    if declared_outside:
        warnings.append(
            f"Skill declares host tools outside the launchable-interpreter set "
            f"({sorted(_ALLOWED_RUNTIME_BINS)}): {declared_outside}. These are runtime "
            "tools a script calls via PATH, not script interpreters Ouroboros launches, "
            "so with no runnable script the skill lands as 'type: instruction'. Use the "
            "Skills 'Make runnable' repair to author a bash/python/node script that "
            "invokes them, then review and enable."
        )
    if declared_in_allowlist:
        # Prefer the common case when only one runtime can be emitted.
        for preferred in ("python3", "python", "node", "bash"):
            if preferred in declared_in_allowlist:
                return preferred
    scripts_dir = staging_dir / "scripts"
    if scripts_dir.is_dir():
        for path in scripts_dir.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if name.endswith(".py"):
                return "python3"
            if name.endswith(".js") or name.endswith(".mjs"):
                return "node"
            if name.endswith(".sh") or name.endswith(".bash"):
                return "bash"
    return ""


def _list_scripts_dir(staging_dir: pathlib.Path) -> List[Dict[str, str]]:
    scripts_dir = staging_dir / "scripts"
    out: List[Dict[str, str]] = []
    if not scripts_dir.is_dir():
        return out
    for path in sorted(scripts_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        out.append({"name": path.name, "description": ""})
    return out


def _translate_permissions(
    metadata_block: Dict[str, Any],
    front: Dict[str, Any],
    warnings: List[str],
) -> List[str]:
    perms: set[str] = set()
    unrecognised: List[str] = []
    allowed_tools_raw = front.get("allowed-tools") or front.get("allowed_tools") or ""
    if isinstance(allowed_tools_raw, str):
        tokens = [t.strip() for t in allowed_tools_raw.split() if t.strip()]
    elif isinstance(allowed_tools_raw, list):
        tokens = [str(t).strip() for t in allowed_tools_raw if str(t).strip()]
    else:
        tokens = []
    for token in tokens:
        upper = token.upper()
        matched = False
        if upper.startswith("BASH") or upper.startswith("SHELL"):
            perms.add("subprocess")
            matched = True
        if upper in ("READ", "WRITE", "READFILE", "WRITEFILE", "FS"):
            perms.add("fs")
            matched = True
        if upper.startswith("FETCH") or upper.startswith("HTTP") or upper.startswith("WEB"):
            perms.add("net")
            matched = True
        if not matched:
            unrecognised.append(token)
    if unrecognised:
        warnings.append(
            f"OpenClaw 'allowed-tools' tokens not mapped to Ouroboros permissions: {sorted(set(unrecognised))}. Reviewer must cross-check SKILL.openclaw.md to confirm the publisher's declared capabilities are still honoured by the translated manifest."
        )
    requires = metadata_block.get("requires") or {}
    if isinstance(requires, dict):
        if _coerce_str_list(requires.get("bins")) or _coerce_str_list(requires.get("anyBins")):
            perms.add("subprocess")
        if _coerce_str_list(requires.get("env")):
            # Env vars often imply API tokens and outbound HTTP.
            perms.add("net")
    if not perms:
        warnings.append(
            "Could not derive any specific permissions from the OpenClaw manifest; skill will be installed with an empty permissions list. Reviewer must verify scripts make no privileged calls."
        )
    return sorted(perms)


def _translate_env_from_settings(
    metadata_block: Dict[str, Any],
    blockers: List[str],
    warnings: Optional[List[str]] = None,
) -> List[str]:
    requires = metadata_block.get("requires") or {}
    keys = _coerce_str_list(requires.get("env") if isinstance(requires, dict) else None)
    forbidden_upper = {k.upper() for k in FORBIDDEN_SKILL_SETTINGS}
    blocked: List[str] = []
    out: List[str] = []
    for key in keys:
        canonical = key.strip().upper()
        if not canonical:
            continue
        if canonical in forbidden_upper:
            blocked.append(canonical)
            continue
        out.append(canonical)
    if blocked:
        message = f"OpenClaw manifest requests core settings keys that require explicit per-skill grants before execution: {sorted(set(blocked))}."
        if warnings is not None:
            warnings.append(message)
        else:
            blockers.append(message)
    return out + [key for key in sorted(set(blocked)) if key not in out]

def _render_frontmatter(front: Dict[str, Any]) -> str:
    import yaml  # type: ignore

    order = ("name", "description", "version", "type", "runtime", "timeout_sec", "when_to_use", "model_experience", "permissions", "env_from_settings", "os", "requires", "entry", "scripts")
    ordered = {
        key: front[key]
        for key in order
        if key in front and front[key] not in ("", [], None)
    }
    dumped = yaml.safe_dump(
        ordered,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()
    return "---\n" + dumped + "\n---"


def _render_skill_md(translated_front: Dict[str, Any], body: str) -> str:
    body_clean = (body or "").strip()
    if body_clean:
        return f"{_render_frontmatter(translated_front)}\n\n{body_clean}\n"
    return f"{_render_frontmatter(translated_front)}\n"


def _append_manual_install_guidance(body: str, manual_specs: List[Dict[str, Any]]) -> str:
    if not manual_specs:
        return body
    lines = [
        "",
        "## Manual setup required",
        "",
        "Ouroboros refused to run these publisher-declared installers automatically",
        "because they cannot be confined to this skill's isolated dependency directory.",
        "Install the required tools manually only if you trust the upstream project:",
        "",
    ]
    for spec in manual_specs:
        label = spec.get("package") or spec.get("kind") or "dependency"
        reason = spec.get("reason") or "manual setup required"
        lines.append(f"- `{label}`: {reason}")
    return (body or "").rstrip() + "\n" + "\n".join(lines) + "\n"

def adapt_openclaw_skill(
    staging_dir: pathlib.Path,
    *,
    slug: str,
    version: str = "",
    sha256: str = "",
    is_plugin: bool = False,
) -> AdapterResult:
    """Translate staged package metadata; blockers prevent install/review."""
    sanitized = sanitize_clawhub_slug(slug)
    target_dirname = sanitized
    warnings: List[str] = []
    blockers: List[str] = []
    provenance: Dict[str, Any] = {
        "schema_version": 1,
        "source": "clawhub",
        "slug": slug,
        "sanitized_name": sanitized,
        "version": (version or "").strip(),
        "sha256": (sha256 or "").strip(),
        "is_plugin": bool(is_plugin),
        "adapter_version": ADAPTER_VERSION,
        "installed_at": utc_now_iso(),
    }

    def _result(
        ok: bool,
        *,
        manifest: Optional[SkillManifest] = None,
        translated_front: Optional[Dict[str, Any]] = None,
        original_front: Optional[Dict[str, Any]] = None,
        original_body: str = "",
        plugin: bool = False,
    ) -> AdapterResult:
        return AdapterResult(
            ok=ok,
            sanitized_name=sanitized,
            target_dirname=target_dirname,
            manifest=manifest,
            warnings=warnings,
            blockers=blockers,
            provenance=provenance,
            translated_frontmatter=translated_front or {},
            original_frontmatter=original_front or {},
            original_body=original_body,
            is_plugin=plugin,
        )

    if is_plugin:
        blockers.append(
            "Package is an OpenClaw Node/TypeScript plugin (openclaw.plugin.json present). Ouroboros does not run Node-host plugins; refusing to install. Ask the author for a Python port or expose via MCP."
        )
        return _result(False, plugin=True)

    try:
        original_text, body, original_front = _read_skill_md(staging_dir)
    except SkillManifestError as exc:
        blockers.append(f"Manifest unreadable: {exc}")
        return _result(False)

    metadata_block = _extract_metadata_block(original_front)
    openclaw_compat = _openclaw_compat_snapshot(original_front, metadata_block, warnings)

    raw_install_specs = metadata_block.get("install")
    auto_install_specs, manual_install_specs, install_warnings = normalize_install_specs(raw_install_specs)
    warnings.extend(install_warnings)
    if auto_install_specs:
        warnings.append(
            "Skill declares dependency install specs. Ouroboros will land the payload disabled, require a fresh executable review, then install these dependencies only inside the skill's .ouroboros_env directory."
        )
    if manual_install_specs:
        warnings.append(
            "Some install specs were converted to manual setup guidance because they cannot be isolated without mutating global host state."
        )

    env_keys = _translate_env_from_settings(metadata_block, blockers, warnings)

    runtime = _detect_runtime(metadata_block, staging_dir, warnings)
    scripts_entries = _list_scripts_dir(staging_dir)
    if runtime and not scripts_entries:
        warnings.append(
            f"Runtime '{runtime}' detected but no executable files found under scripts/; skill becomes 'type: instruction'."
        )
    skill_type = "script" if runtime and scripts_entries else "instruction"

    permissions = _translate_permissions(metadata_block, original_front, warnings)
    if skill_type != "script":
        # Instruction-only skills have no executable subprocess surface.
        permissions = [p for p in permissions if p != "subprocess"]

    name = sanitized
    description = str(original_front.get("description") or "").strip()
    when_to_use = str(original_front.get("when_to_use") or "").strip()
    if not when_to_use and description:
        # Preserve OpenClaw's description-as-trigger behavior.
        when_to_use = description

    os_field = _normalise_os(metadata_block.get("os"))
    if os_field not in {"any", "darwin", "linux", "windows"}:
        warnings.append(
            f"OS restriction '{os_field}' could not be normalised to a single OS literal; falling back to 'any' so the skill is discoverable."
        )
        os_field = "any"

    homepage = str(original_front.get("homepage") or original_front.get("website") or "").strip()
    license_field = str(original_front.get("license") or "").strip()
    primary_env = str(metadata_block.get("primaryEnv") or "").strip()

    # Keep untrusted publisher URLs in provenance, not rendered SKILL.md.
    provenance_extras: Dict[str, Any] = {"clawhub_slug": slug}
    if homepage:
        provenance_extras["homepage"] = homepage
    if license_field:
        provenance_extras["license"] = license_field
    if primary_env:
        provenance_extras["primary_env"] = primary_env
    if raw_install_specs not in (None, "", [], {}):
        provenance_extras["install_specs"] = {
            "schema_version": 1,
            "auto": auto_install_specs,
            "manual": manual_install_specs,
            "raw": _json_safe(raw_install_specs),
            "specs_hash": install_specs_hash(auto_install_specs),
        }
    requested_grants = [
        key for key in env_keys
        if key.upper() in {item.upper() for item in FORBIDDEN_SKILL_SETTINGS}
    ]
    if requested_grants:
        provenance_extras["requested_key_grants"] = requested_grants

    raw_timeout = original_front.get("timeout_sec")
    if raw_timeout in (None, ""):
        timeout_sec = 60
    else:
        try:
            timeout_sec = int(raw_timeout)
        except (TypeError, ValueError):
            warnings.append(
                f"Manifest timeout_sec={raw_timeout!r} is not integer-valued; defaulting to 60s. Reviewer should confirm the publisher's intent (some OpenClaw skills ship suffixes like '60s'."
            )
            timeout_sec = 60
        if timeout_sec <= 0:
            warnings.append(
                f"Manifest timeout_sec={raw_timeout!r} must be positive; defaulting to 60s."
            )
            timeout_sec = 60

    translated_front: Dict[str, Any] = {
        "name": name,
        "description": description or f"ClawHub skill {slug}",
        "version": str(original_front.get("version") or version or "").strip(),
        "type": skill_type,
        "runtime": runtime if skill_type == "script" else "",
        "timeout_sec": timeout_sec,
        "when_to_use": when_to_use,
        "permissions": permissions,
        "env_from_settings": env_keys,
        "os": os_field,
        "scripts": scripts_entries if skill_type == "script" else [],
        "schema_version": SKILL_MANIFEST_SCHEMA_VERSION,
    }
    # CPL-7: a source manifest that already carries Ouroboros Model Experience
    # prose keeps it; the re-parse below re-validates the section fail-closed.
    source_model_experience = original_front.get("model_experience")
    if isinstance(source_model_experience, dict) and source_model_experience:
        translated_front["model_experience"] = source_model_experience

    rendered_body = _append_manual_install_guidance(body, manual_install_specs)
    rendered_skill_md = _render_skill_md(translated_front, rendered_body)
    try:
        new_manifest = parse_skill_manifest_text(rendered_skill_md)
    except SkillManifestError as exc:
        blockers.append(
            f"Adapter produced an unparseable Ouroboros manifest: {exc}. This is an internal bug; please report."
        )
        return _result(False, translated_front=translated_front, original_front=original_front, original_body=body)

    manifest_warnings = new_manifest.validate()
    for w in manifest_warnings:
        warnings.append(f"manifest validate: {w}")

    provenance["original_manifest_sha256"] = _sha256_of_text(original_text)
    provenance["translated_manifest_sha256"] = _sha256_of_text(rendered_skill_md)
    provenance["adapter_warnings"] = list(warnings)
    provenance["openclaw_compat"] = openclaw_compat
    provenance["original_frontmatter"] = _json_safe(original_front)
    provenance.update(provenance_extras)

    if blockers:
        return _result(False, manifest=new_manifest, translated_front=translated_front, original_front=original_front, original_body=body)

    # Persist both manifests inside the already-validated staging directory.
    skill_md_path = staging_dir / "SKILL.md"
    openclaw_path = staging_dir / "SKILL.openclaw.md"
    try:
        if openclaw_path.exists():
            openclaw_path.unlink()
        if skill_md_path.exists():
            shutil.copy2(str(skill_md_path), str(openclaw_path))
        else:
            # Preserve skill.json inputs for audit.
            (staging_dir / "skill.openclaw.json").write_text(original_text, encoding="utf-8")
        skill_md_path.write_text(rendered_skill_md, encoding="utf-8")
        # Sidecar makes source obvious without reading durable state.
        atomic_write_json(staging_dir / ".clawhub.json", provenance, trailing_newline=True)
    except OSError as exc:
        blockers.append(f"Failed to persist translated manifest: {exc}")
        return _result(False, manifest=new_manifest, translated_front=translated_front, original_front=original_front, original_body=body)

    return _result(True, manifest=new_manifest, translated_front=translated_front, original_front=original_front, original_body=body)


def _sha256_of_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


__all__ = [
    "ADAPTER_VERSION",
    "AdapterResult",
    "adapt_openclaw_skill",
    "sanitize_clawhub_slug",
]
