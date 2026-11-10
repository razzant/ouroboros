"""The deployment policy: an administrator's constraints over the employee's settings.

An installation is managed when its trusted launch (compose, systemd, the container runtime) names one strict
JSON file in ``OUROBOROS_DEPLOYMENT_POLICY``: an absolute path outside the data root and the checkout, which no
Settings surface writes. Without the variable nothing here changes behaviour. A named file that is missing,
unreadable, malformed or of another version never reads as unmanaged: ``current()`` raises
``DeploymentPolicyError``, a server refuses to start (``config.verify_settings_integrity``) and every other
process's settings reads fail closed.

Generation. A server reads the file once at its start (``load_for_generation``) and exports the digest of those
bytes in ``OUROBOROS_DEPLOYMENT_POLICY_SHA256``; each process it starts inherits the pin, reads the file once and
refuses different bytes. A changed file applies at the next start: nothing polls or reloads it, no copy is kept.

Version 1 (``version`` required, every other field optional, unknown fields refused)::

    {"version": 1,
     "access": {"max": "light" | "advanced" | "pro"},
     "safety": {"min_coverage": "light" | "full",
                "checker": {"model": "<route>", "account": "<profile, claudexor:: routes only>",
                            "fallbacks": ["<route>", ...]},
                "allowed_exits": ["repeat_after_dangerous", "release_after_dangerous", "release_unjudged"]},
     "rules": "<the organization's rules>",
     "extensions": {"when_unmet": "hold" | "warn",
                    "required": [{"name": "<skill>", "path": "<absolute package directory>",
                                  "content_hash": "<64 hex>", "settings": {"<KEY>": "<value>"}}],
                    "trusted": [{"name": "<skill>", "path": "<absolute package directory>",
                                 "content_hash": "<64 hex>"}]},
     "execution": {...}}  # where employees' work runs: ``execution_domain``

``access.max`` caps effective Access (Cyber Pro is out of reach below it); ``min_coverage`` floors both Safety
coverage levels; ``checker`` is the Safety Supervisor route: its model, account and the only fallbacks its walk
may use, never a Settings slot, role option, wait-card switch or second-check row; ``allowed_exits`` names
which actor/outcome-specific Safety exits stay open (``SAFETY_EXITS``, ``safety_exit_open``); ``rules`` reach every
acting mind, delegated session and Safety check as the organization's, beside the employee's own rules;
``extensions`` names installed extensions that must run, each by its canonical name, package directory and the
digest the loader computes for its bytes, with the custom settings it fixes, and whether ordinary work holds
or continues while one is not running (``deployment_extensions`` reads it; no default is assumed). ``trusted``
exists only with ``execution``: the exact packages, named the same way, whose code the service imports, catalogs
or launches, each where the work domain cannot write; every required package must be one of them.
Each field constrains only what it names, except that a named checker also closes ``release_unjudged`` unless
``allowed_exits`` lists it: a checker without the Access cap still leaves Cyber Pro, where Safety findings are
advice, and without the coverage floor coverage may be lowered.

Constraints are views, never intent. Readers apply them last (``effective_value``, ``effective_settings``,
``effective_environ``), after task snapshots, room cognition, live route views and the environment, so none of
those undoes them; an extension's fixed settings reach only the settings view, which a skill reads through its
grants (``effective_environ`` never exports them). The settings document keeps the employee's raw choices, shadowed
ones included:
``prepare_write``, the first step of ``config.prepare_settings_for_persist`` (before any receipt or byte), keeps
the document's raw value where a write carries a managed value back unchanged or as the effective view showed
it, and refuses any other change to a managed choice; it also refuses a newly marked second-check row and a
write after which a checker route would reach another target, judged as dispatch reads the document
(``_checker_targets``). Single-decision writers judge their request with ``choice_refusal``; a local checker's
server starts only as its saved settings name it (``local_launch_refusal``). The serving program enforcing all
this is the administrator's: every in-band door that would replace it refuses (``body_replacement_refusal``).
The policy constrains only processes launched with it: it is no protection of the files, code or credentials
that enforce it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ouroboros.model_slots import (
    MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY, MODEL_PROCESSING_PREFERENCES_KEY, normalize_model_role_options,
)
from ouroboros.settings_defaults import SETTINGS_DEFAULTS, _DISK_AUTHORED_SETTINGS
from ouroboros.settings_integrity import SettingsIntegrityError
from ouroboros.settings_scales import (
    _RUNTIME_MODE_RANK, _SAFETY_MODE_RANK, normalize_runtime_mode, normalize_safety_mode,
)

POLICY_ENV = "OUROBOROS_DEPLOYMENT_POLICY"
GENERATION_ENV = "OUROBOROS_DEPLOYMENT_POLICY_SHA256"
SOURCE = "deployment_policy"
SCHEMA_VERSION = 1
MAX_POLICY_BYTES = 256 * 1024
MAX_RULES_BYTES = 64_000  # the owner rules' own bound
MAX_FALLBACKS = 8
ACCESS_KEY = "OUROBOROS_RUNTIME_MODE"
COVERAGE_KEYS = ("OUROBOROS_SAFETY_MODE", "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE")
CHECKER_KEY = "OUROBOROS_MODEL_SAFETY"
CHECKER_ROLE = "safety"
UPDATE_CHANNEL_KEY = "OUROBOROS_UPDATE_CHANNEL"  # which official feed an in-app Update applies: a body door
ROLE_OPTION_KEYS = (MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY, MODEL_PROCESSING_PREFERENCES_KEY)
_ACCESS_CEILINGS = ("light", "advanced", "pro")
_COVERAGE_FLOORS = ("light", "full")
# Keep agency and outcome distinct: an agent's repeat, a human's card after DANGEROUS, and an unchecked release.
REPEAT_AFTER_DANGEROUS, RELEASE_AFTER_DANGEROUS, RELEASE_UNJUDGED = SAFETY_EXITS = (
    "repeat_after_dangerous", "release_after_dangerous", "release_unjudged")
# A local checker answers from this model server: which weights, template and port.
_LOCAL_IDENTITY_KEYS = ("LOCAL_MODEL_SOURCE", "LOCAL_MODEL_FILENAME", "LOCAL_MODEL_CHAT_FORMAT", "LOCAL_MODEL_PORT")
# What ordinary work does while a required extension is not running: wait for it, or go on with the gap shown.
WHEN_UNMET = ("hold", "warn")
MAX_REQUIRED_EXTENSIONS = 16
MAX_TRUSTED_EXTENSIONS = 64  # the bundled built-ins and an organization's own, with room
MAX_FIXED_VALUE_BYTES = 8192
_DIGEST = re.compile(r"[0-9a-f]{64}")
_ABSENT = object()


class DeploymentPolicyError(SettingsIntegrityError):
    """The configured deployment policy cannot be used; nothing may read as unmanaged. A settings trust root
    that failed, like a changed integrity pin: every reader that re-raises one re-raises the other."""


class DeploymentPolicyRefusal(PermissionError):
    """A write would change a choice the deployment policy manages; nothing was written."""


@dataclass(frozen=True)
class Checker:
    """The organization's Safety Supervisor: route, account pin and approved fallbacks, in walk order."""

    model: str
    account: str = ""
    fallbacks: tuple[str, ...] = ()

    def routes(self) -> tuple[tuple[str, bool, str], ...]:
        """``(model, use_local, account)`` the Safety walk sends to, primary first, each once."""
        found: list[tuple[str, bool, str]] = []
        for text, account in ((self.model, self.account), *((item, "") for item in self.fallbacks)):
            route = (text.removesuffix(" (local)"), text.endswith(" (local)"), account)
            if route not in found:
                found.append(route)
        return tuple(found)


@dataclass(frozen=True)
class RequiredExtension:
    """An installed extension the deployment requires: its canonical name, the package directory the loader must find
    it in and the digest the loader computes for the bytes it runs (``LoadedSkill.content_hash``), with the custom
    settings the organization fixes for it as ``(key, value)`` pairs, which are never printed."""

    name: str
    path: str
    content_hash: str
    settings: tuple = field(default=(), repr=False)


@dataclass(frozen=True)
class TrustedExtension:
    """A package whose code the service of a deployment with an execution domain may run: its canonical name, its
    directory and the digest the loader computes for the bytes it runs. Trust admits the code only; whether it runs
    stays with its switch, review, grants, dependencies and conflicts, as for any extension."""

    name: str
    path: str
    content_hash: str


@dataclass(frozen=True)
class DeploymentPolicy:
    path: str
    digest: str
    access_max: str = ""
    min_coverage: str = ""
    checker: Optional[Checker] = None
    rules: str = ""
    managed: frozenset = field(default=frozenset())
    allowed_exits: frozenset = frozenset(SAFETY_EXITS)  # as the file names them, else their default (``parse_policy``)
    extensions: tuple = ()  # ``RequiredExtension`` rows, in the file's order
    when_unmet: str = ""  # one of ``WHEN_UNMET`` whenever ``extensions`` names one
    execution: Any = None  # ``execution_domain.ExecutionDomain`` when the file names one
    trusted: tuple = ()  # ``TrustedExtension`` rows, in the file's order (only with ``execution``)


# --- the document --------------------------------------------------------------------------------------

def _unique_keys(pairs: list) -> dict:
    keys = [key for key, _value in pairs]
    duplicate = next((key for key in keys if keys.count(key) > 1), None)
    if duplicate is not None:
        raise DeploymentPolicyError(f"the field {duplicate!r} appears twice")
    return dict(pairs)


def _fields(value: Any, where: str, allowed: tuple[str, ...]) -> dict:
    if not isinstance(value, dict):
        raise DeploymentPolicyError(f"{where} must be a JSON object")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise DeploymentPolicyError(f"{where} has the unsupported field {unknown[0]!r}")
    return value


def _choice(section: dict, name: str, where: str, allowed: tuple[str, ...]) -> str:
    if name not in section:
        return ""
    if section[name] not in allowed:
        raise DeploymentPolicyError(f"{where} must be one of: {', '.join(allowed)}")
    return section[name]


def _route(value: Any, where: str) -> str:
    from ouroboros.provider_models import parse_model_route

    if not isinstance(value, str) or not value.strip() or value != value.strip() or not parse_model_route(value).valid:
        raise DeploymentPolicyError(f"{where} must be a model route such as 'anthropic::<model>'")
    return value


def _fixed_settings(value: Any, where: str, taken: set) -> tuple:
    """A required extension's fixed settings: custom keys only (the domain a skill's ``env_from_settings`` may name
    and an owner grants), each once in the file, never a built-in, ``OUROBOROS_`` or protected core key."""
    from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS
    from ouroboros.secret_masking import is_custom_secret_setting_key

    if not isinstance(value, dict):
        raise DeploymentPolicyError(f"{where} must be a JSON object of setting names and text values")
    pairs = []
    for key, text in value.items():
        if not is_custom_secret_setting_key(key, known_setting_keys=SETTINGS_DEFAULTS) or key in FORBIDDEN_SKILL_SETTINGS:
            raise DeploymentPolicyError(f"{where} may name only custom settings (A_KEY, never built-in or OUROBOROS_), "
                                        f"not {key!r}")
        if key in taken:
            raise DeploymentPolicyError(f"{where}.{key} is fixed twice in the file")
        if not isinstance(text, str) or not text or "\0" in text or len(text.encode("utf-8")) > MAX_FIXED_VALUE_BYTES:
            raise DeploymentPolicyError(f"{where}.{key} must be text of 1 to {MAX_FIXED_VALUE_BYTES} UTF-8 bytes")
        taken.add(key)
        pairs.append((key, text))
    return tuple(pairs)


def _package(item: Any, where: str, allowed: tuple, names: set, paths: set) -> dict:
    """One package row of its list: a distinct canonical name (its folder name), a distinct absolute directory and the
    digest the loader computes for its bytes (``skill_identity``'s release digest shape)."""
    from ouroboros.contracts.skill_manifest import canonical_skill_name

    row = _fields(item, where, allowed)
    name, path, digest = row.get("name"), row.get("path"), row.get("content_hash")
    if not isinstance(name, str) or not name or canonical_skill_name(name) != name or name in names:
        raise DeploymentPolicyError(f"{where}.name must be a distinct canonical skill name (its folder name)")
    if not isinstance(path, str) or path != path.strip() or not os.path.isabs(path) or path in paths:
        raise DeploymentPolicyError(f"{where}.path must be a distinct absolute package directory")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise DeploymentPolicyError(f"{where}.content_hash must be the 64-hex digest the loader computes")
    names.add(name)
    paths.add(path)
    return row


def _real(path: Any) -> str:
    return os.path.normcase(os.path.realpath(str(path)))


def _work_folders(execution: Any) -> list:
    """``(label, real path)`` of every folder the work domain writes: its mappings and the skills' state root."""
    return [*((f"path mapping {mapping.host_path}", _real(mapping.host_path)) for mapping in execution.executor.mappings),
            *((("skill_state_root", _real(execution.skill_state_root)),) if execution.skill_state_root else ())]


def _overlap(path: str, folder: str) -> bool:
    try:
        return os.path.commonpath([path, folder]) in (path, folder)
    except ValueError:  # another drive (Windows) is never inside
        return False


def _extensions_section(value: Any, execution: Any) -> tuple[str, tuple, tuple]:
    """``(when_unmet, required, trusted)`` of the ``extensions`` section; a mode must be named whenever a required
    row is. ``trusted`` exists only with ``execution``, each package outside every folder the work domain writes,
    and there every required package is one of them, name, directory and digest alike."""
    section = _fields(value, "extensions", ("when_unmet", "required", "trusted"))
    rows, listed = section.get("required", []), section.get("trusted", [])
    if not isinstance(rows, list) or len(rows) > MAX_REQUIRED_EXTENSIONS:
        raise DeploymentPolicyError(f"extensions.required must be a list of at most {MAX_REQUIRED_EXTENSIONS} extensions")
    if not isinstance(listed, list) or len(listed) > MAX_TRUSTED_EXTENSIONS:
        raise DeploymentPolicyError(f"extensions.trusted must be a list of at most {MAX_TRUSTED_EXTENSIONS} extensions")
    when_unmet = _choice(section, "when_unmet", "extensions.when_unmet", WHEN_UNMET)
    if rows and not when_unmet:
        raise DeploymentPolicyError("extensions.when_unmet must say whether ordinary work holds or warns while a "
                                    "required extension is not running: one of hold, warn")
    required, names, paths, keys = [], set(), set(), set()
    for index, item in enumerate(rows):
        where = f"extensions.required[{index}]"
        row = _package(item, where, ("name", "path", "content_hash", "settings"), names, paths)
        required.append(RequiredExtension(row["name"], row["path"], row["content_hash"], _fixed_settings(
            row.get("settings", {}), f"{where}.settings", keys)))
    if "trusted" in section and execution is None:
        raise DeploymentPolicyError("extensions.trusted applies only with execution: without a separate work domain, "
                                    "employees' processes share the service's machine and no list protects its code")
    trusted, names, paths = [], set(), set()
    for index, item in enumerate(listed):
        where = f"extensions.trusted[{index}]"
        row = _package(item, where, ("name", "path", "content_hash"), names, paths)
        real = _real(row["path"])
        for label, folder in _work_folders(execution):
            if _overlap(real, folder):
                raise DeploymentPolicyError(f"{where}.path overlaps the work domain's {label}, which employees' "
                                            "processes write; keep a trusted package where only the service writes")
        trusted.append(TrustedExtension(row["name"], row["path"], row["content_hash"]))
    pins = {row.name: (_real(row.path), row.content_hash) for row in trusted}
    for index, row in enumerate(required if execution is not None else ()):
        if pins.get(row.name) != (_real(row.path), row.content_hash):
            raise DeploymentPolicyError(f"extensions.required[{index}] must also be in extensions.trusted with the same "
                                        "path and content_hash: with execution, only trusted packages run on the service")
    return when_unmet, tuple(required), tuple(trusted)


def parse_policy(data: bytes, *, path: str = "") -> DeploymentPolicy:
    """Version 1, strictly: a JSON object with ``version`` and any of the documented fields; anything else raises
    ``DeploymentPolicyError``."""
    from ouroboros.provider_models import provider_for_model

    if len(data) > MAX_POLICY_BYTES:
        raise DeploymentPolicyError(f"the file exceeds {MAX_POLICY_BYTES} bytes")
    try:  # a repeated field raises its own error from the hook
        document = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_keys)
    except (UnicodeDecodeError, ValueError) as exc:
        raise DeploymentPolicyError(f"the file is not a UTF-8 JSON document ({type(exc).__name__})") from None
    document = _fields(document, "the policy", ("version", "access", "safety", "rules", "extensions", "execution"))
    version = document.get("version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise DeploymentPolicyError(f"version {version!r} is unsupported; this release reads version {SCHEMA_VERSION}")
    access_max = _choice(_fields(document.get("access", {}), "access", ("max",)), "max", "access.max", _ACCESS_CEILINGS)
    safety = _fields(document.get("safety", {}), "safety", ("min_coverage", "checker", "allowed_exits"))
    min_coverage = _choice(safety, "min_coverage", "safety.min_coverage", _COVERAGE_FLOORS)
    checker = None
    if "checker" in safety:
        raw = _fields(safety["checker"], "safety.checker", ("model", "account", "fallbacks"))
        model = _route(raw.get("model"), "safety.checker.model")
        account = raw.get("account", "")
        if not isinstance(account, str) or account != account.strip():
            raise DeploymentPolicyError("safety.checker.account must be a profile name, or empty for rotation")
        if account and provider_for_model(model) != "claudexor":
            raise DeploymentPolicyError("safety.checker.account pins a subscription account: it needs a claudexor:: model")
        fallbacks = raw.get("fallbacks", [])
        if not isinstance(fallbacks, list) or len(fallbacks) > MAX_FALLBACKS:
            raise DeploymentPolicyError(f"safety.checker.fallbacks must be a list of at most {MAX_FALLBACKS} routes")
        checker = Checker(model, account, tuple(_route(item, f"safety.checker.fallbacks[{index}]")
                                                for index, item in enumerate(fallbacks)))
    # Unnamed: both stay open, except that the organization's own checker releases nothing it did not judge.
    exits = safety.get("allowed_exits", [name for name in SAFETY_EXITS if checker is None or name != RELEASE_UNJUDGED])
    if not isinstance(exits, list) or any(name not in SAFETY_EXITS for name in exits) or len(set(exits)) < len(exits):
        raise DeploymentPolicyError(f"safety.allowed_exits must list distinct exits, each one of: {', '.join(SAFETY_EXITS)}")
    rules = document.get("rules", "")
    if not isinstance(rules, str) or len(rules.encode("utf-8")) > MAX_RULES_BYTES:
        raise DeploymentPolicyError(f"rules must be text of at most {MAX_RULES_BYTES} UTF-8 bytes")
    execution = None
    if "execution" in document:
        from ouroboros.execution_domain import parse_execution

        try:
            execution = parse_execution(document["execution"])
        except ValueError as exc:
            raise DeploymentPolicyError(str(exc)) from None
    when_unmet, extensions, trusted = _extensions_section(document.get("extensions", {}), execution)
    managed = ({ACCESS_KEY} if access_max else set()) | (set(COVERAGE_KEYS) if min_coverage else set())
    if checker is not None:
        managed |= {CHECKER_KEY, *ROLE_OPTION_KEYS}
    return DeploymentPolicy(path, hashlib.sha256(data).hexdigest(), access_max, min_coverage, checker, rules,
                            managed=frozenset(managed), allowed_exits=frozenset(exits), extensions=extensions,
                            when_unmet=when_unmet, execution=execution, trusted=trusted)


# --- this process's generation ------------------------------------------------------------------------

_LOCK = threading.Lock()
_STATE: Optional[tuple] = None  # ("ok", policy or None) | ("error", reason), once per process


def _policy_path(raw: str) -> Path:
    from ouroboros.launch_roots import configured_roots

    path = Path(raw)
    if not path.is_absolute():
        raise DeploymentPolicyError(f"{POLICY_ENV} must be an absolute path, not {raw!r}")
    resolved, roots = os.path.realpath(path), configured_roots()
    for name, root in (("data root", roots.data_dir), ("checkout", roots.repo_dir)):
        base = os.path.realpath(root)
        try:
            inside = os.path.commonpath([resolved, base]) == base
        except ValueError:  # absolute Windows paths on different drives are disjoint
            inside = False
        if inside:
            raise DeploymentPolicyError(f"{path} lies inside the {name}, which the workspace writes; place it outside")
    return path


def _resolve(*, fresh: bool) -> Optional[DeploymentPolicy]:
    raw = os.environ.get(POLICY_ENV, "").strip()
    pin = "" if fresh else os.environ.get(GENERATION_ENV, "").strip().lower()
    if not raw:
        if pin:
            raise DeploymentPolicyError(f"{GENERATION_ENV} is set without {POLICY_ENV}: this process belongs to a "
                                        "managed server generation whose policy it can no longer name")
        return None
    path = _policy_path(raw)
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_POLICY_BYTES + 1)
    except FileNotFoundError:
        raise DeploymentPolicyError(f"{path} does not exist") from None
    except OSError as exc:
        raise DeploymentPolicyError(f"{path} cannot be read ({type(exc).__name__})") from None
    try:
        policy = parse_policy(data, path=str(path))
    except DeploymentPolicyError as exc:
        raise DeploymentPolicyError(f"{path}: {exc}") from None
    if pin and policy.digest != pin:
        raise DeploymentPolicyError(f"{path} changed after this server generation started; restart to apply it")
    return policy


def current() -> Optional[DeploymentPolicy]:
    """This process's policy (``None``: unmanaged), read once; a configured but unusable one raises every time."""
    global _STATE
    state = _STATE
    if state is None:
        with _LOCK:
            if _STATE is None:
                try:
                    _STATE = ("ok", _resolve(fresh=False))
                except DeploymentPolicyError as exc:
                    _STATE = ("error", str(exc))
            state = _STATE
    if state[0] == "error":
        raise DeploymentPolicyError(f"The deployment policy cannot be used: {state[1]}")
    return state[1]


def load_for_generation() -> Optional[DeploymentPolicy]:
    """A server's start: read the file fresh (an inherited pin belongs to the previous generation) and pin it."""
    global _STATE
    with _LOCK:
        try:
            policy = _resolve(fresh=True)
        except DeploymentPolicyError as exc:
            _STATE = ("error", str(exc))
            raise DeploymentPolicyError(f"The deployment policy cannot be used: {exc}") from None
        _STATE = ("ok", policy)
        if policy is None:
            os.environ.pop(GENERATION_ENV, None)
        else:
            os.environ[GENERATION_ENV] = policy.digest
    return policy


def reset_for_tests() -> None:
    """Forget this process's read; the next ``current()`` reads the environment again."""
    global _STATE
    with _LOCK:
        _STATE = None


def checker() -> Optional[Checker]:
    policy = current()
    return policy.checker if policy is not None else None


def managed_model_roles() -> frozenset:
    """Model roles whose route the policy sets: a wait-card switch or a restored override never moves them."""
    return frozenset({CHECKER_ROLE}) if checker() is not None else frozenset()


def served_wait(row: Any, roles: Optional[frozenset] = None) -> Any:
    """A model-wait row as served now: ``managed`` follows this process's policy whenever the row was written, so
    one from before it (history, a carried or restored wait) never offers the switch the decision door refuses."""
    if not isinstance(row, dict):
        return row
    served = {key: value for key, value in row.items() if key != "managed"}
    return {**served, "managed": True} if row.get("role") in (managed_model_roles() if roles is None else roles) else served


def served_waits(rows: Any) -> Any:
    """``served_wait`` over a ``model_waits`` map."""
    if not isinstance(rows, dict):
        return rows
    roles = managed_model_roles()
    return {key: served_wait(row, roles) for key, row in rows.items()}


def organization_rules() -> str:
    policy = current()
    return policy.rules if policy is not None else ""


def required_extensions() -> tuple:
    """``RequiredExtension`` rows of this process's policy; ``()`` unmanaged or when it names none."""
    policy = current()
    return policy.extensions if policy is not None else ()


def trusted_extensions() -> Optional[tuple]:
    """The ``TrustedExtension`` rows, the only packages whose code the service runs, when the policy names an
    execution domain (``()``: none); ``None`` unmanaged or without ``execution``, where extensions load as before."""
    policy = current()
    return policy.trusted if policy is not None and policy.execution is not None else None


def when_unmet() -> str:
    """``hold`` or ``warn`` (``WHEN_UNMET``) while the policy requires an extension, else ""."""
    policy = current()
    return policy.when_unmet if policy is not None and policy.extensions else ""


def fixed_settings() -> dict:
    """``{key: the required extension it is fixed for}``: names only; the values reach the settings view alone."""
    return {key: row.name for row in required_extensions() for key, _value in row.settings}


def child_policy_env() -> dict:
    """What a child of this generation needs to read settings as this process does, added after any environment
    scrub: the file's path and the digest pinned at the start ({} unmanaged). It grants no secret: the child's own
    settings reads still pass its grants, and a changed file still refuses there."""
    policy = current()
    return {} if policy is None else {POLICY_ENV: policy.path, GENERATION_ENV: policy.digest}


def safety_exit_open(name: str) -> bool:
    """Whether Safety's exit ``name`` (``SAFETY_EXITS``) may act where ``safety.check_safety`` offers it; always so
    unmanaged. The file only closes an exit: an open one still needs every fact that exit reads."""
    policy = current()
    return policy is None or name in policy.allowed_exits


BODY_STATUS = ("This installation runs the program its organization deployed, which enforces its deployment policy: "
               "only its administrator replaces it, outside Ouroboros.")
BODY_REFUSAL = f"{BODY_STATUS} Nothing was changed."


def body_replacement_refusal() -> str:
    """Why a managed deployment refuses an in-band request to replace the program it serves; "" unmanaged.

    The serving source and dependencies enforce this policy, so they change only by the administrator's own
    deployment (an image, package or checkout outside Ouroboros), never at an employee's or agent's request:
    every supported door that would update, switch, restore, roll back, promote, recover or adopt them refuses
    before any write, pause or restart, whoever asks. A restart of the same program, Stop and Panic stay."""
    return BODY_REFUSAL if current() is not None else ""


# --- readers -------------------------------------------------------------------------------------------

def _entry(policy: DeploymentPolicy, key: str) -> Any:
    """The checker role's own entry of a role-option map: the pinned account, else Auto (absent)."""
    return policy.checker.account if key == MODEL_ACCOUNTS_KEY and policy.checker.account else _ABSENT


def _managed_value(policy: DeploymentPolicy, key: str, value: Any) -> Any:
    if key == ACCESS_KEY:
        mode = normalize_runtime_mode(value)
        return policy.access_max if _RUNTIME_MODE_RANK[mode] > _RUNTIME_MODE_RANK[policy.access_max] else mode
    if key in COVERAGE_KEYS:
        mode = normalize_safety_mode(value)
        return policy.min_coverage if _SAFETY_MODE_RANK[mode] < _SAFETY_MODE_RANK[policy.min_coverage] else mode
    if key == CHECKER_KEY:
        return policy.checker.model
    options = normalize_model_role_options(key, value)[0]
    entry = _entry(policy, key)
    if entry is _ABSENT:
        options.pop(CHECKER_ROLE, None)
    else:
        options[CHECKER_ROLE] = entry
    if isinstance(value, dict):
        return options
    return value if not options and value in (None, "") else normalize_model_role_options(key, options)[1]


def effective_value(key: str, value: Any) -> Any:
    """``value`` (env- or document-shaped) as this deployment lets it read; an unmanaged key passes unchanged."""
    policy = current()
    if policy is None or key not in policy.managed:
        return value
    return _managed_value(policy, key, value)


def effective_settings(settings: dict) -> dict:
    """A settings mapping with every managed key as this deployment reads it, in place; returned for chaining. An
    extension's fixed settings are read here too, so a skill sees them through its grants like any custom setting."""
    policy = current()
    for key in (policy.managed if policy is not None else ()):
        value = _managed_value(policy, key, settings.get(key))
        if key in settings or value not in (None, ""):
            settings[key] = value
    for row in (policy.extensions if policy is not None else ()):
        settings.update(row.settings)
    return settings


def effective_environ(env: dict) -> None:
    """A child environment (``runtime_environ``) carrying every managed key as this deployment reads it."""
    policy = current()
    for key in (policy.managed if policy is not None else ()):
        value = _managed_value(policy, key, env.get(key))
        if value not in (None, ""):
            env[key] = str(value)


# --- writers -------------------------------------------------------------------------------------------

def _normalized(key: str, value: Any) -> Any:
    if key == ACCESS_KEY:
        return normalize_runtime_mode(value)
    if key in COVERAGE_KEYS:
        return normalize_safety_mode(value)
    return str(value or "").strip()


def _refusal(policy: DeploymentPolicy, what: str) -> str:
    detail = {ACCESS_KEY: f"at most {policy.access_max}", CHECKER_KEY: "the organization's Safety Supervisor"}.get(
        what, f"at least {policy.min_coverage}" if what in COVERAGE_KEYS else "the organization's Safety Supervisor")
    return (f"{what} is managed by this deployment's policy file ({detail}); nothing was saved. "
            "Its administrator changes it there.")


def choice_refusal(key: str, value: Any) -> str:
    """Why this deployment refuses ``value`` as an explicit choice of ``key``; "" when it allows it."""
    policy = current()
    if policy is None or key not in policy.managed or key in ROLE_OPTION_KEYS:
        return ""
    allowed = _normalized(key, _managed_value(policy, key, value)) == _normalized(key, value)
    return "" if allowed else _refusal(policy, key)


def _restore(written: dict, document: Mapping, key: str) -> None:
    if key in document:
        written[key] = document[key]
    else:
        written.pop(key, None)


def _keep_raw_scalar(policy: DeploymentPolicy, written: dict, document: Mapping, environ: Mapping, key: str) -> None:
    proposed, raw = written.get(key, _ABSENT), document.get(key, _ABSENT)
    unmanaged = raw if raw is not _ABSENT else (
        SETTINGS_DEFAULTS.get(key) if key in _DISK_AUTHORED_SETTINGS else environ.get(key, SETTINGS_DEFAULTS.get(key)))
    shown = _managed_value(policy, key, unmanaged)
    blank = raw is _ABSENT and proposed in (None, "")  # a form's empty field over no saved value: no choice at all
    if proposed is _ABSENT or blank or _normalized(key, proposed) == _normalized(key, shown) or (
            raw is not _ABSENT and _normalized(key, proposed) == _normalized(key, raw)):
        _restore(written, document, key)
    elif reason := choice_refusal(key, proposed):
        raise DeploymentPolicyRefusal(reason)


def _keep_raw_entry(policy: DeploymentPolicy, written: dict, document: Mapping, key: str) -> None:
    if key not in written:
        _restore(written, document, key)
        return
    proposed = written[key]
    options, raw_options = normalize_model_role_options(key, proposed)[0], normalize_model_role_options(key, document.get(key))[0]
    empty = 0 if key == MODEL_CONTEXT_WINDOWS_KEY else ""

    def value(item: Any) -> Any:
        return empty if item is _ABSENT else item

    entry, raw_entry = options.get(CHECKER_ROLE, _ABSENT), raw_options.get(CHECKER_ROLE, _ABSENT)
    if value(entry) not in (value(raw_entry), value(_entry(policy, key))):
        raise DeploymentPolicyRefusal(_refusal(policy, f"{key}[{CHECKER_ROLE}]"))
    if raw_entry is _ABSENT:
        options.pop(CHECKER_ROLE, None)
    else:
        options[CHECKER_ROLE] = raw_entry
    written[key] = options if isinstance(proposed, dict) else normalize_model_role_options(key, options)[1]


def _keep_raw_fixed(written: dict, document: Mapping, key: str, value: str, owner: str) -> None:
    """A fixed extension setting carried back unchanged, as Settings masks it, blank over nothing saved or not at all
    keeps the document's raw value; any other value is refused, naming the key and never a value."""
    from ouroboros.secret_masking import looks_masked_settings_secret

    proposed, raw = written.get(key, _ABSENT), document.get(key, _ABSENT)
    if (proposed is _ABSENT or (raw is _ABSENT and proposed in (None, "")) or _same(proposed, value)
            or looks_masked_settings_secret(key, proposed) or (raw is not _ABSENT and _same(proposed, raw))):
        _restore(written, document, key)
        return
    raise DeploymentPolicyRefusal(f"{key} is managed by this deployment's policy file (the organization sets it for "
                                  f"{owner}); nothing was saved. Its administrator changes it there.")


def _local_binding(values: Mapping) -> dict:
    """The local model server a local route reaches: artifact, template and port, as a launch reads them."""
    from ouroboros.local_model import local_model_settings

    try:
        launch = local_model_settings(values)
    except (TypeError, ValueError):  # a malformed value no launch could use is still a value
        launch = {key: str(values.get(key, "")) for key in _LOCAL_IDENTITY_KEYS}
    return {key: launch[key] for key in _LOCAL_IDENTITY_KEYS}


def _served(document: Mapping) -> tuple:
    """``(effective settings, env-shaped route projection)`` once ``document`` is the saved one, read as dispatch reads
    it (``route_choice.route_view``: defaults, the legacy compatible pair, an operator's variable, never this process's
    echo of an earlier save)."""
    from ouroboros import config
    from ouroboros.route_choice import route_view

    return route_view(lambda: config.settings_of_document(dict(document)))


class _Reads(dict):
    """A route projection that names every key a resolver reads from it."""

    def __init__(self, values: Mapping):
        super().__init__(values)
        self.names: set = set()

    def get(self, key: Any, default: Any = None) -> Any:
        self.names.add(key)
        return super().get(key, default)


def _checker_reads(policy: DeploymentPolicy, served: tuple) -> list:
    """The flat settings the checker's remote routes read as dispatch reads them under ``served`` (a built-in
    connection's fields, the compatible lane's dedicated or legacy pair); a named connection's row and a local
    server are named by ``connections`` and ``local``."""
    from ouroboros.llm_attempt import ConnectionUnavailable
    from ouroboros.llm_routing import resolve_target
    from ouroboros.provider_models import CONNECTIONS_KEY

    reads = _Reads(served[1])
    for model, local, _account in policy.checker.routes():
        if not local:
            try:
                resolve_target(model, reads)
            except (ConnectionUnavailable, ValueError):
                pass  # what it read before refusing still decides its target
    return sorted(reads.names - {CONNECTIONS_KEY})


def _keep_raw_field(written: dict, document: Mapping, shown: Mapping, key: str) -> None:
    """A field a checker route reads, carried back as the effective view showed it, blank over no saved value or
    not at all, keeps the document's own value: absent stays absent, so a form never writes what the launch or a
    default provides. Anything else is judged by its target (``_checker_targets``)."""
    proposed, raw = written.get(key, _ABSENT), document.get(key, _ABSENT)
    if proposed is _ABSENT or (raw is _ABSENT and proposed in (None, "")) or _same(proposed, shown.get(key, "")) or (
            raw is not _ABSENT and _same(proposed, raw)):
        _restore(written, document, key)


def _same(left: Any, right: Any) -> bool:
    return str("" if left is None else left).strip() == str("" if right is None else right).strip()


def _checker_targets(policy: DeploymentPolicy, served: tuple) -> str:
    """Where every checker route's call goes under ``served`` (``_served``): the transport's own target, or why the
    connection cannot serve; a local route, its server. One digest, so no credential leaves."""
    from ouroboros.llm_attempt import ConnectionUnavailable
    from ouroboros.llm_routing import resolve_target

    effective, projection = served
    targets: list = []
    for model, local, _account in policy.checker.routes():
        try:
            targets.append(_local_binding(effective) if local else resolve_target(model, projection))
        except (ConnectionUnavailable, ValueError) as exc:
            targets.append(["unavailable", type(exc).__name__, str(exc)])
    return hashlib.sha256(json.dumps(targets, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def local_launch_refusal(*, source: str, filename: str, port: Any, chat_format: str) -> str:
    """Why this deployment refuses to start this local model server; "" when it allows it. A checker route that
    answers locally binds the server its saved settings name; any other artifact, template or port is refused
    before a download or start (Stop stays free: the check then holds its action)."""
    policy = current()
    if policy is None or policy.checker is None or not any(local for _m, local, _a in policy.checker.routes()):
        return ""
    from ouroboros import config
    from ouroboros.route_choice import route_view

    saved = _local_binding(route_view(lambda: config.load_settings_lock_held(_settings_lock_held=False))[0])
    requested = _local_binding({"LOCAL_MODEL_SOURCE": source, "LOCAL_MODEL_FILENAME": filename,
                                "LOCAL_MODEL_PORT": port, "LOCAL_MODEL_CHAT_FORMAT": chat_format})
    if requested == saved:
        return ""
    return ("The local model server answers the organization's Safety Supervisor and is managed by this "
            "deployment's policy file: only the model, template and port its settings name may start; nothing "
            "was started.")


def _second_checkers(settings: Mapping) -> frozenset:
    from ouroboros.configured_subagents import SUBAGENTS_SETTING, parse_configured_subagents

    try:
        catalog = parse_configured_subagents(settings.get(SUBAGENTS_SETTING)) if settings.get(SUBAGENTS_SETTING) else None
    except ValueError:
        return frozenset()
    return frozenset(row.subagent_id for row in catalog.items if row.rechecks_flags) if catalog else frozenset()


def prepare_write(proposed: dict, read_document: Callable[[], Mapping], environ: Optional[Mapping] = None) -> dict:
    """The deployment's step of every persisting write, before any receipt or byte; returns what may be written.

    ``read_document`` answers the normalized settings document on disk ({} when absent); an unmanaged process
    never calls it. A managed value the write carries back unchanged, or as the effective view showed it, keeps
    the document's own raw choice (absent where the document had none). Any other value of a managed choice must
    be one the policy allows, the update channel and every checker route's target (``_checker_targets``) must read
    as before and no new row may take the second check: otherwise the whole write is refused with
    ``DeploymentPolicyRefusal``. A flat field a checker route reads, carried back as shown, blank over nothing
    saved or not at all, keeps the document's raw value too (``_keep_raw_field``), as does an extension's fixed
    setting carried back unchanged or masked (``_keep_raw_fixed``)."""
    policy = current()
    if policy is None:
        return proposed
    document, environ = read_document(), os.environ if environ is None else environ
    written = dict(proposed)
    for key in sorted(policy.managed - set(ROLE_OPTION_KEYS)):
        _keep_raw_scalar(policy, written, document, environ, key)
    for row in policy.extensions:
        for key, value in row.settings:
            _keep_raw_fixed(written, document, key, value, row.name)
    if policy.checker is not None:
        for key in ROLE_OPTION_KEYS:
            _keep_raw_entry(policy, written, document, key)
    before = _served(document)
    if policy.checker is not None:
        for key in _checker_reads(policy, before):
            _keep_raw_field(written, document, before[0], key)
    after = _served(written)
    if before[0].get(UPDATE_CHANNEL_KEY) != after[0].get(UPDATE_CHANNEL_KEY):  # the feed only its administrator updates from
        raise DeploymentPolicyRefusal(f"{UPDATE_CHANNEL_KEY} is managed by this deployment's policy file. {BODY_REFUSAL}")
    if policy.checker is not None:
        if _checker_targets(policy, before) != _checker_targets(policy, after):
            raise DeploymentPolicyRefusal(
                "A connection the organization's Safety Supervisor uses is managed by this deployment's policy "
                "file; nothing was saved. Use another connection for your own models.")
        if _second_checkers(written) - _second_checkers(document):
            raise DeploymentPolicyRefusal(
                "The second check of flagged Safety verdicts is managed by this deployment's policy file: the "
                "organization's Supervisor decides alone, so no Settings row can take it; nothing was saved.")
    return written


# --- the Settings projection ---------------------------------------------------------------------------

def _checker_state(policy: DeploymentPolicy) -> dict:
    """The checker as Settings shows it: its routes, and what they reach (the connections it uses, by the picker's
    ids; the flat fields its built-in routes read now, ``_checker_reads``; whether the local model server answers),
    so exactly those controls are shown as fixed."""
    from ouroboros import config
    from ouroboros.provider_models import PROVIDER_KINDS, connection_id, parse_model_route
    from ouroboros.settings_integrity import read_settings_json_verified

    checker_route = policy.checker
    routes = [parse_model_route(text) for text in (checker_route.model, *checker_route.fallbacks)]
    try:
        stored = read_settings_json_verified(config.SETTINGS_PATH)
        fields = _checker_reads(policy, _served(config.normalize_settings_raw(stored) if isinstance(stored, dict) else {}))
    except Exception:  # keeps the rest of the projection (without it Settings reads unmanaged) but names no field
        fields = []  # here; a save is still judged by its target, which reads the same way unguarded
    return {"model": checker_route.model, "account": checker_route.account, "fallbacks": list(checker_route.fallbacks),
            "connections": sorted({connection_id(route.kind, route.connection) for route in routes
                                   if route.kind in PROVIDER_KINDS}),
            "fields": fields, "local": any(route.local for route in routes)}


def policy_state() -> Optional[dict]:
    """``_meta.policy_state.deployment``: what this deployment's policy fixes and where it comes from, no secret.
    A managed installation's serving program is always its administrator's (``body_replacement_refusal``). Required
    extensions name their fixed settings by key only; their running state is the reader's (``deployment_extensions``).
    With an execution domain ``extensions.trusted`` lists the packages whose code the service runs."""
    policy = current()
    if policy is None:
        return None
    extensions = None if not policy.extensions and policy.execution is None else {
        "when_unmet": policy.when_unmet, "required": [
            {"name": row.name, "path": row.path, "content_hash": row.content_hash,
             "settings": [key for key, _v in row.settings]} for row in policy.extensions],
        **({"trusted": [{"name": row.name, "path": row.path, "content_hash": row.content_hash}
                        for row in policy.trusted]} if policy.execution is not None else {})}
    return {"source": SOURCE, "version": SCHEMA_VERSION, "digest": policy.digest[:16],
            "access_max": policy.access_max or None, "min_coverage": policy.min_coverage or None,
            "checker": None if policy.checker is None else _checker_state(policy),
            "allowed_exits": [name for name in SAFETY_EXITS if name in policy.allowed_exits],
            "organization_rules": bool(policy.rules.strip()), "extensions": extensions}
