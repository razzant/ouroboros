"""Unified SKILL.md/skill.json parser; tolerant extras, fail-closed structure."""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SKILL_MANIFEST_SCHEMA_VERSION = 1
MAX_SKILL_CONFLICTS = 32
MAX_SKILL_NAME_LENGTH = 64

VALID_SKILL_TYPES = frozenset({"instruction", "script", "extension"})
VALID_SKILL_RUNTIMES = frozenset({
    "",
    "python",
    "python3",
    "node",
    "bash",
    # Binaries resolve at exec time; missing runtimes still fail closed there.
    "deno",
    "ruby",
    "go",
})
VALID_SKILL_PERMISSIONS = frozenset(
    {
        "net",
        "fs",
        "subprocess",
        "widget",
        "ws_handler",
        # Keep extension permissions aligned with plugin_api's frozen contract.
        "route",
        "tool",
        "read_settings",
        "iframe_raw",
        "companion_process",
        "supervised_task",
        "subscribe_event",
        "inject_chat",
        "presence",
    }
)
_EVENT_TOPIC_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")

# CPL-7 Model Experience: optional prose section describing what the skill adds
# to the model's context and what that costs in tokens. Prose by design — the
# section is rendered verbatim (bounded) on model-visible surfaces, not parsed.
MODEL_EXPERIENCE_KEYS = ("what_model_sees", "token_effect")


class SkillManifestError(ValueError):
    """Manifest has structural contract damage.

    A refusal teaches (CPL-7): ``problem`` names what is wrong and
    ``fix_hint`` — when the refusal knows the repair — says how to fix the
    manifest. The hint is rendered into ``str(exc)`` so every surface that
    relays the refusal (loader state, skill preflight, marketplace install,
    review output) teaches the author without any consumer change; typed
    consumers can read ``.problem`` / ``.fix_hint`` separately.
    """

    def __init__(self, problem: str, *, fix_hint: str = "") -> None:
        self.problem = str(problem)
        self.fix_hint = str(fix_hint or "").strip()
        message = self.problem
        if self.fix_hint:
            message = f"{self.problem} — fix: {self.fix_hint}"
        super().__init__(message)


def canonical_skill_name(name: str) -> str:
    """Return the canonical bounded identifier used for skill state and routing."""
    cleaned = "".join(
        ch if ch.isalnum() or ch in "-_." else "_"
        for ch in str(name or "").strip()
    )
    cleaned = cleaned.strip("._")
    if not cleaned:
        return "_unnamed"
    return cleaned[:MAX_SKILL_NAME_LENGTH]


@dataclass
class SkillManifest:
    """Structural description of one skill package."""

    name: str
    description: str
    version: str
    type: str  # instruction | script | extension
    when_to_use: str = ""
    requires: List[str] = field(default_factory=list)
    os: str = "any"
    runtime: str = ""
    timeout_sec: int = 60
    env_from_settings: List[str] = field(default_factory=list)
    # Script manifests list script mappings.
    scripts: List[Dict[str, str]] = field(default_factory=list)
    # Extension manifests point at a Python entry module.
    entry: str = ""
    permissions: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    subscribe_events: List[str] = field(default_factory=list)
    companion_processes: List[Dict[str, Any]] = field(default_factory=list)
    scheduled_tasks: List[Dict[str, Any]] = field(default_factory=list)
    ui_tab: Optional[Dict[str, Any]] = None
    # ABI-1 (PluginAPI 2.0): optional declared generation, normalized to
    # {"version": "M.m", "capabilities": [...]}. Absent (None) means the
    # payload binds against the LEGACY generation by construction.
    plugin_api: Optional[Dict[str, Any]] = None
    # CPL-7: optional Model Experience prose, normalized to a mapping with
    # keys from MODEL_EXPERIENCE_KEYS (at least one non-empty). Absent (None)
    # keeps the pre-section behavior everywhere.
    model_experience: Optional[Dict[str, str]] = None
    # Human-readable body after SKILL.md frontmatter.
    body: str = ""
    # Unknown fields preserved for forward compatibility.
    raw_extra: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SKILL_MANIFEST_SCHEMA_VERSION

    def is_instruction(self) -> bool:
        return self.type == "instruction"

    def is_script(self) -> bool:
        return self.type == "script"

    def is_extension(self) -> bool:
        return self.type == "extension"

    def validate(self) -> List[str]:
        """Return soft warnings; parse errors already raised on structural damage."""
        warnings: List[str] = []
        if self.type not in VALID_SKILL_TYPES:
            warnings.append(
                f"unknown type '{self.type}' (expected one of "
                f"{sorted(VALID_SKILL_TYPES)})"
            )
        if self.runtime not in VALID_SKILL_RUNTIMES:
            warnings.append(
                f"unknown runtime '{self.runtime}' (expected empty or one of "
                f"{sorted(r for r in VALID_SKILL_RUNTIMES if r)})"
            )
        for perm in self.permissions:
            if perm not in VALID_SKILL_PERMISSIONS:
                warnings.append(
                    f"unknown permission '{perm}' (expected one of "
                    f"{sorted(VALID_SKILL_PERMISSIONS)})"
                )
        for topic in self.subscribe_events:
            if not _EVENT_TOPIC_RE.match(topic):
                warnings.append(
                    f"invalid subscribe_events topic '{topic}' "
                    "(expected lower.dotted format)"
                )
        if self.is_extension() and not self.entry:
            warnings.append("type=extension requires non-empty 'entry'")
        if self.is_script() and not self.scripts:
            warnings.append("type=script requires at least one entry in 'scripts'")
        # An instruction skill is pure guidance (SKILL.md) with no executable surface; a
        # declared entry/scripts is a structural type mismatch the manifest reviewer flags.
        if self.type in VALID_SKILL_TYPES and not self.is_extension() and not self.is_script() and (self.entry or self.scripts):
            warnings.append(
                f"type='{self.type}' must not declare executable 'entry'/'scripts' "
                "(only extension/script skills run code)"
            )
        if self.timeout_sec <= 0:
            warnings.append("timeout_sec must be positive")
        if self.scheduled_tasks and "supervised_task" not in self.permissions:
            warnings.append("scheduled_tasks require the supervised_task permission")
        if self.plugin_api is not None:
            # Semantic validation is owned by the negotiation contract; a
            # function-level import keeps the contracts package acyclic.
            from ouroboros.contracts.plugin_api import negotiate_plugin_api

            negotiation = negotiate_plugin_api(self)
            if not negotiation.ok:
                warnings.append(f"invalid plugin_api declaration: {negotiation.error}")
        return warnings


_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z",
    re.DOTALL,
)


def parse_skill_manifest_text(text: str) -> SkillManifest:
    """Parse JSON, YAML frontmatter, or body-only instruction markdown."""
    src = text.lstrip("\ufeff")
    stripped = src.lstrip()

    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SkillManifestError(
                f"invalid skill.json: {exc}",
                fix_hint="skill.json must be one valid JSON document; repair the syntax at the reported line/column",
            ) from exc
        if not isinstance(data, dict):
            raise SkillManifestError(
                "skill.json root must be a mapping",
                fix_hint="wrap the manifest fields in a single top-level JSON object {...}",
            )
        return _manifest_from_mapping(data, body="")

    match = _FRONTMATTER_RE.match(src)
    if match is not None:
        front, body = match.group(1), match.group(2) or ""
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SkillManifestError(
                "PyYAML is required to parse SKILL.md frontmatter",
                fix_hint="install the pyyaml dependency, or ship the manifest as skill.json instead",
            ) from exc
        try:
            data: Any = yaml.safe_load(front) or {}
        except yaml.YAMLError as exc:  # type: ignore[name-defined]
            raise SkillManifestError(
                f"invalid SKILL.md frontmatter: {exc}",
                fix_hint="the block between the two '---' fences must be valid YAML; repair the reported line",
            ) from exc
        if not isinstance(data, dict):
            raise SkillManifestError(
                "SKILL.md frontmatter must be a mapping",
                fix_hint="write 'key: value' lines between the '---' fences, not a list or bare scalar",
            )
        return _manifest_from_mapping(data, body=body.strip())
    # A leading thematic break is valid body markdown, not broken frontmatter.
    name = _derive_name_from_body(src)
    return SkillManifest(
        name=name,
        description="",
        version="",
        type="instruction",
        body=src.strip(),
        schema_version=SKILL_MANIFEST_SCHEMA_VERSION,
    )


def _parse_model_experience(raw: Any) -> Optional[Dict[str, str]]:
    """CPL-7: Model Experience is prose with a fixed key set; refusals teach."""
    if raw in (None, ""):
        return None
    if isinstance(raw, str):
        # The string form IS the one-key mapping form; routing it through the
        # mapping branch is what makes the two shapes refuse identically — a
        # whitespace-only section used to be stored as empty prose, so every
        # model-visible surface rendered the label with no sentence after it.
        raw = {"what_model_sees": raw}
    if isinstance(raw, dict):
        unknown = sorted(set(raw) - set(MODEL_EXPERIENCE_KEYS))
        if unknown:
            raise SkillManifestError(
                f"'model_experience' mapping has unknown keys {unknown}",
                fix_hint=(
                    "use only 'what_model_sees' (prose: what the skill adds to the "
                    "model's context) and 'token_effect' (prose: roughly what that "
                    "costs in tokens and when it is loaded)"
                ),
            )
        cleaned: Dict[str, str] = {}
        for key in MODEL_EXPERIENCE_KEYS:
            value = raw.get(key)
            if value in (None, ""):
                continue
            if not isinstance(value, str):
                raise SkillManifestError(
                    f"'model_experience.{key}' must be a prose string",
                    fix_hint=f"write {key} as free-form prose in quotes, not a list or mapping",
                )
            if value.strip():
                cleaned[key] = value.strip()
        if not cleaned:
            raise SkillManifestError(
                "'model_experience' must carry at least one non-empty prose field",
                fix_hint=(
                    "fill 'what_model_sees' and/or 'token_effect' with prose, "
                    "or drop the section (it is optional)"
                ),
            )
        return cleaned
    raise SkillManifestError(
        "'model_experience' must be a prose string or a mapping when provided",
        fix_hint=(
            "either write model_experience: \"<prose>\" or a mapping with "
            "'what_model_sees' / 'token_effect' string fields"
        ),
    )


def _parse_companion_processes(companion_raw: Any) -> List[Dict[str, Any]]:
    """Companion descriptors: reviewed argv shapes only; every refusal teaches."""
    if companion_raw in (None, ""):
        companion_raw = []
    if not isinstance(companion_raw, list):
        raise SkillManifestError(
            "'companion_processes' must be a list when provided",
            fix_hint="declare companion_processes as a list of {name, command, runtime} mappings",
        )
    companion_processes: List[Dict[str, Any]] = []
    for item in companion_raw:
        if not isinstance(item, dict):
            raise SkillManifestError(
                "each 'companion_processes' item must be a mapping",
                fix_hint="each companion needs at least {name: ..., command: [...], runtime: ...}",
            )
        if not str(item.get("name") or "").strip():
            raise SkillManifestError(
                "each 'companion_processes' item must include name",
                fix_hint="add a unique 'name' so the companion can be supervised and reported",
            )
        if not isinstance(item.get("command"), list) or not item.get("command"):
            raise SkillManifestError(
                "each 'companion_processes' item must include a non-empty command list",
                fix_hint="write command as an argv list, e.g. command: [python3, scripts/companion.py]",
            )
        runtime = str(item.get("runtime") or "").strip().lower()
        if not runtime:
            raise SkillManifestError(
                "each 'companion_processes' item must include runtime",
                fix_hint=f"add runtime: one of {sorted(r for r in VALID_SKILL_RUNTIMES if r)}",
            )
        if runtime and runtime not in VALID_SKILL_RUNTIMES:
            raise SkillManifestError(
                f"companion_processes runtime '{runtime}' is not supported",
                fix_hint=f"use one of {sorted(r for r in VALID_SKILL_RUNTIMES if r)}",
            )
        command0 = str((item.get("command") or [""])[0] or "").strip().lower()
        command = [str(part or "").strip() for part in (item.get("command") or [])]
        inline_flags = {"-c", "-m", "-e", "--eval", "eval"}
        if any(arg in inline_flags for arg in command[1:]):
            raise SkillManifestError(
                "companion inline/eval commands are not allowed",
                fix_hint="put the code in a reviewed script file inside the skill and name it in command",
            )
        for arg in command[1:]:
            arg_path = pathlib.PurePosixPath(arg)
            if arg_path.is_absolute() or ".." in arg_path.parts:
                raise SkillManifestError(
                    "companion command arguments must stay inside the reviewed skill tree",
                    fix_hint="use paths relative to the skill directory, without '..' or absolute prefixes",
                )
        if runtime in {"python", "python3"} and command0 not in {"python", "python3"}:
            raise SkillManifestError(
                "python companion runtime must use python/python3 command",
                fix_hint="start command with 'python3' (or 'python') followed by the reviewed script path",
            )
        if runtime in {"python", "python3"}:
            if len(command) < 2:
                raise SkillManifestError(
                    "python companion command must name a reviewed script",
                    fix_hint="add the script path after the interpreter, e.g. [python3, scripts/companion.py]",
                )
            if pathlib.PurePosixPath(command[1]).is_absolute() or ".." in pathlib.PurePosixPath(command[1]).parts:
                raise SkillManifestError(
                    "python companion script must be a relative reviewed path",
                    fix_hint="reference the script relative to the skill directory, without '..'",
                )
        if runtime in {"node", "npm"} and command0 not in {"node", "npm"}:
            raise SkillManifestError(
                "node companion runtime must use node/npm command",
                fix_hint="start command with 'node' or 'npm' when runtime is node/npm",
            )
        if runtime in {"bash", "deno", "ruby", "go"} and command0 != runtime:
            raise SkillManifestError(
                f"{runtime} companion runtime must use {runtime} command",
                fix_hint=f"start command with '{runtime}' when runtime is {runtime}",
            )
        if runtime in {"bash", "deno", "ruby", "go"} and len(command) > 1:
            script_path = pathlib.PurePosixPath(command[1])
            if script_path.is_absolute() or ".." in script_path.parts:
                raise SkillManifestError(
                    f"{runtime} companion script must be a relative reviewed path",
                    fix_hint="reference the script relative to the skill directory, without '..'",
                )
        companion_processes.append(dict(item))
    return companion_processes


def _parse_scheduled_tasks(scheduled_raw: Any) -> List[Dict[str, Any]]:
    """Reviewed cron descriptors; id/cron/timezone validated by the shared SSOT."""
    if scheduled_raw in (None, ""):
        scheduled_raw = []
    if not isinstance(scheduled_raw, list):
        raise SkillManifestError(
            "'scheduled_tasks' must be a list when provided",
            fix_hint="declare scheduled_tasks as a list of {name, cron, ...} mappings",
        )
    scheduled_tasks: List[Dict[str, Any]] = []
    for item in scheduled_raw:
        if not isinstance(item, dict):
            raise SkillManifestError(
                "each 'scheduled_tasks' item must be a mapping",
                fix_hint="each scheduled task needs at least {name: ..., cron: ...}",
            )
        name = str(item.get("name") or "").strip()
        if not name:
            raise SkillManifestError(
                "each 'scheduled_tasks' item must include name",
                fix_hint="add a unique schedule 'name' (it becomes the schedule id)",
            )
        cron = str(item.get("cron") or "").strip()
        from ouroboros.schedule_contract import cron_error, schedule_id_error, timezone_error

        if err := schedule_id_error(name):
            raise SkillManifestError(
                f"scheduled_tasks name is invalid: {err}",
                fix_hint="rename the schedule to satisfy the reported constraint",
            )
        if err := cron_error(cron):
            raise SkillManifestError(
                f"scheduled_tasks cron expression is invalid: {err}",
                fix_hint="use a standard 5-field cron expression, e.g. cron: \"0 8 * * *\"",
            )
        timezone = str(item.get("timezone") or "").strip()
        if err := timezone_error(timezone):
            raise SkillManifestError(
                f"scheduled_tasks timezone is invalid: {err}",
                fix_hint="use an IANA timezone name like Europe/Amsterdam, or omit the field",
            )
        scheduled_tasks.append(dict(item))
    return scheduled_tasks


def _manifest_from_mapping(data: Dict[str, Any], *, body: str) -> SkillManifest:
    known = {
        "name",
        "description",
        "version",
        "type",
        "when_to_use",
        "requires",
        "os",
        "runtime",
        "timeout_sec",
        "env_from_settings",
        "scripts",
        "entry",
        "permissions",
        "conflicts",
        "subscribe_events",
        "companion_processes",
        "scheduled_tasks",
        "ui_tab",
        "plugin_api",
        "model_experience",
        "schema_version",
    }
    extras: Dict[str, Any] = {
        key: value for key, value in data.items() if key not in known
    }

    timeout_raw = data.get("timeout_sec", 60)
    try:
        timeout_sec = int(timeout_raw) if timeout_raw not in (None, "") else 60
    except (TypeError, ValueError):
        timeout_sec = 60

    scripts_raw = data.get("scripts", [])
    scripts: List[Dict[str, str]] = []
    if scripts_raw in (None, ""):
        scripts_raw = []
    if not isinstance(scripts_raw, list):
        raise SkillManifestError(
            "'scripts' must be a list when provided",
            fix_hint="declare scripts as a YAML/JSON list of {name, ...} mappings or plain script names",
        )
    for item in scripts_raw:
        if isinstance(item, dict):
            scripts.append({str(k): str(v) for k, v in item.items()})
        elif isinstance(item, str):
            scripts.append({"name": item})
        else:
            raise SkillManifestError(
                "each 'scripts' item must be a mapping or string",
                fix_hint="use a mapping like {name: run.py} or the bare script name as a string",
            )

    ui_tab = data.get("ui_tab")
    if ui_tab is not None and not isinstance(ui_tab, dict):
        raise SkillManifestError(
            "'ui_tab' must be a mapping when provided",
            fix_hint="declare ui_tab as a mapping (e.g. {title: ..., path: ...}) or drop the field",
        )

    # ABI-1: structural shape is fail-closed here; the semantic version /
    # capability contract is enforced by plugin_api negotiation.
    plugin_api_raw = data.get("plugin_api")
    plugin_api: Optional[Dict[str, Any]] = None
    if plugin_api_raw not in (None, ""):
        if isinstance(plugin_api_raw, str):
            plugin_api = {"version": plugin_api_raw.strip(), "capabilities": []}
        elif isinstance(plugin_api_raw, dict):
            unknown_keys = sorted(set(plugin_api_raw) - {"version", "capabilities"})
            if unknown_keys:
                raise SkillManifestError(
                    f"'plugin_api' mapping has unknown keys {unknown_keys} "
                    "(expected 'version' and optional 'capabilities')",
                    fix_hint="keep only plugin_api.version (\"major.minor\") and optional plugin_api.capabilities",
                )
            version_value = plugin_api_raw.get("version")
            if not isinstance(version_value, str) or not version_value.strip():
                raise SkillManifestError(
                    "'plugin_api.version' must be a non-empty string",
                    fix_hint="declare the generation as a quoted \"major.minor\" string, e.g. version: \"2.0\"",
                )
            plugin_api = {
                "version": version_value.strip(),
                "capabilities": _string_list(plugin_api_raw.get("capabilities")),
            }
        else:
            raise SkillManifestError(
                "'plugin_api' must be a version string or a mapping when provided",
                fix_hint="write plugin_api: \"2.0\" or a mapping with 'version' and optional 'capabilities'",
            )

    model_experience = _parse_model_experience(data.get("model_experience"))

    companion_processes = _parse_companion_processes(data.get("companion_processes", []))

    scheduled_tasks = _parse_scheduled_tasks(data.get("scheduled_tasks", []))
    schema_version = data.get("schema_version", SKILL_MANIFEST_SCHEMA_VERSION)
    try:
        schema_version_int = int(schema_version)
    except (TypeError, ValueError):
        raise SkillManifestError(
            "'schema_version' must be an integer",
            fix_hint=f"set schema_version: {SKILL_MANIFEST_SCHEMA_VERSION}, or omit the field",
        ) from None
    if schema_version_int != SKILL_MANIFEST_SCHEMA_VERSION:
        raise SkillManifestError(
            f"unsupported schema_version {schema_version_int}; "
            f"expected {SKILL_MANIFEST_SCHEMA_VERSION}",
            fix_hint=(
                f"this runtime speaks manifest schema {SKILL_MANIFEST_SCHEMA_VERSION}; "
                "set that value or drop the field"
            ),
        )

    conflicts = _string_list(data.get("conflicts"))
    if len(conflicts) > MAX_SKILL_CONFLICTS:
        raise SkillManifestError(
            f"'conflicts' may contain at most {MAX_SKILL_CONFLICTS} skill names",
            fix_hint="trim the conflicts list to the skills that actually clash",
        )
    canonical_conflicts: List[str] = []
    for conflict in conflicts:
        canonical = canonical_skill_name(conflict)
        if canonical == "_unnamed" or conflict != canonical:
            raise SkillManifestError(
                f"conflicts entry {conflict!r} must be a canonical skill name "
                f"(letters/numbers plus '-', '_', or '.', max {MAX_SKILL_NAME_LENGTH} characters)",
                fix_hint=(f"rename the entry to its canonical form {canonical!r}"
                          if canonical != "_unnamed" else
                          "name an installable skill; the entry normalizes to nothing"),
            )
        if canonical not in canonical_conflicts:
            canonical_conflicts.append(canonical)

    return SkillManifest(
        name=str(data.get("name") or "").strip(),
        description=str(data.get("description") or "").strip(),
        version=str(data.get("version") or "").strip(),
        type=str(data.get("type") or "instruction").strip().lower(),
        when_to_use=str(data.get("when_to_use") or "").strip(),
        requires=_string_list(data.get("requires")),
        os=str(data.get("os") or "any").strip().lower() or "any",
        runtime=str(data.get("runtime") or "").strip().lower(),
        timeout_sec=timeout_sec,
        env_from_settings=_string_list(data.get("env_from_settings")),
        scripts=scripts,
        entry=str(data.get("entry") or "").strip(),
        permissions=_string_list(data.get("permissions")),
        conflicts=canonical_conflicts,
        subscribe_events=_string_list(data.get("subscribe_events")),
        companion_processes=companion_processes,
        scheduled_tasks=scheduled_tasks,
        ui_tab=ui_tab,
        plugin_api=plugin_api,
        model_experience=model_experience,
        body=body,
        raw_extra=extras,
        schema_version=schema_version_int,
    )


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _derive_name_from_body(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip().lower().replace(" ", "_") or "unnamed"
    return "unnamed"


__all__ = [
    "SKILL_MANIFEST_SCHEMA_VERSION",
    "MAX_SKILL_CONFLICTS",
    "MAX_SKILL_NAME_LENGTH",
    "MODEL_EXPERIENCE_KEYS",
    "VALID_SKILL_TYPES",
    "VALID_SKILL_RUNTIMES",
    "VALID_SKILL_PERMISSIONS",
    "SkillManifest",
    "SkillManifestError",
    "canonical_skill_name",
    "parse_skill_manifest_text",
]
