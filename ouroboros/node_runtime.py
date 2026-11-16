"""Node runtime health and skill-family runtime policy.

Split out of ``platform_layer`` (which keeps the cross-platform primitives:
``_hidden_run``, ``bootstrap_process_path``, ``resolve_bundled_node``): this
module owns the EXECUTION-probed health verdicts and the skill-family
precedence policy built on them. ``platform_layer`` re-exports the public
names so existing importers keep working.

Platform access is by module attribute (``_platform.<name>``) on call, not by
from-import, so the ``platform_layer -> node_runtime`` re-export cycle stays
inert at import time.
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import subprocess
from typing import Any, Dict, List, NamedTuple, Tuple

from ouroboros import platform_layer as _platform

log = logging.getLogger(__name__)


def _probe_node_version_outcome(node_path: str, timeout_sec: float = 10) -> Tuple[str, str]:
    """One ``node --version`` execution probe: ``(version, failure_reason)``.

    Exactly one of the pair is non-empty. The reason vocabulary is small and
    typed-ish (``exec_failed:...``, ``timeout``, ``exit:N``, ``signal:NAME``)
    because the health consumers disclose it verbatim in traces and receipts.
    """
    # A metadata probe must not inherit runtime/test hooks. In particular,
    # NODE_OPTIONS can contain test filters or preload modules that either make
    # `node --version` fail before the hermetic lane gets a chance to scrub the
    # variable or execute arbitrary operator code during a supposedly inert
    # version check.
    probe_env = dict(os.environ)
    probe_env.pop("NODE_OPTIONS", None)
    try:
        result = _platform._hidden_run(
            [str(node_path), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_sec,
            check=False,
            env=probe_env,
        )
    except subprocess.TimeoutExpired:
        return "", "timeout"
    except (OSError, subprocess.SubprocessError) as exc:
        return "", f"exec_failed:{type(exc).__name__}"
    rc = int(result.returncode or 0)
    if rc != 0:
        if rc < 0:
            name = _platform.posix_signal_name(abs(rc))
            return "", f"signal:{name}"
        return "", f"exit:{rc}"
    version = str(result.stdout or "").strip().removeprefix("v")
    if not version:
        # rc=0 with empty stdout: not a healthy node and not a named failure —
        # give the consumers a real reason instead of an empty pair.
        return "", "empty_version_output"
    return version, ""


def probe_node_version(node_path: str) -> str:
    """Return a normalized Node version, or ``""`` on probe failure."""
    version, _reason = _probe_node_version_outcome(node_path)
    return version


class NodeRuntimeHealth(NamedTuple):
    """Execution-probed usability of one Node executable path.

    ``status`` is ``healthy`` (probe returned a version), ``broken`` (the file
    exists but the probe failed — the incident class is a kernel
    SIGKILL/CODESIGNING death of a corrupted Homebrew node), or ``missing``.
    ``reason`` carries the probe failure verbatim for traces/receipts.
    """

    status: str
    version: str = ""
    reason: str = ""
    path: str = ""
    # For a memoized ``timeout`` verdict: the probe budget it was observed
    # under. A caller with a LARGER budget re-probes instead of trusting it.
    probed_timeout: float = 0.0

    @property
    def healthy(self) -> bool:
        return self.status == "healthy"


# Process-local probe memo keyed by (lexical path, mtime_ns, size). A changed
# binary re-probes; ``missing`` is deliberately never memoized so a runtime
# installed mid-session (e.g. `brew reinstall node`, D9) is noticed without a
# restart. A ``timeout`` verdict memoizes WITH the budget it was observed
# under (``probed_timeout``): callers probe with DIFFERENT budgets (workspace
# preflight uses a short 3s cap on the bounded admission path, skill surfaces
# allow 10s), so a caller whose budget exceeds the cached one re-probes — a
# short-budget timeout can therefore never poison a longer-budget consumer —
# while a same-or-smaller budget reuses the verdict instead of stalling for
# the full timeout again on every call (T15). The incident class (kernel
# SIGKILL on launch) dies in milliseconds and memoizes normally. Known residual: a fix that does not
# touch the file bytes (xattr / Gatekeeper requalification) keeps a stale
# ``broken`` verdict for this process's lifetime — the trace disclosing the
# memoized reason makes that visible rather than silent.
_NODE_HEALTH_MEMO: Dict[Tuple[str, int, int], NodeRuntimeHealth] = {}


def node_runtime_health(node_path: str, timeout_sec: float = 10) -> NodeRuntimeHealth:
    """Probe (memoized) whether ``node_path`` is an actually-runnable Node.

    ``shutil.which`` proves only that a file exists and is executable; the
    incident class this exists for is a PATH node the kernel kills on launch,
    which only a real execution probe can see.
    """
    text = str(node_path or "").strip()
    if not text:
        return NodeRuntimeHealth(status="missing", reason="empty_path")
    candidate = pathlib.Path(text)
    try:
        stat = candidate.stat()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            return NodeRuntimeHealth(status="missing", reason="not_executable", path=text)
    except OSError as exc:
        return NodeRuntimeHealth(status="missing", reason=f"stat_failed:{type(exc).__name__}", path=text)
    key = (text, int(stat.st_mtime_ns), int(stat.st_size))
    cached = _NODE_HEALTH_MEMO.get(key)
    if cached is not None and (
        cached.reason != "timeout" or float(timeout_sec) <= cached.probed_timeout
    ):
        return cached
    version, reason = _probe_node_version_outcome(text, timeout_sec=timeout_sec)
    if version:
        health = NodeRuntimeHealth(status="healthy", version=version, path=text)
    elif reason == "timeout":
        health = NodeRuntimeHealth(
            status="broken", reason=reason, path=text,
            probed_timeout=float(timeout_sec),
        )
    else:
        health = NodeRuntimeHealth(status="broken", reason=reason, path=text)
    _NODE_HEALTH_MEMO[key] = health
    return health


def _path_node_runtime_health(timeout_sec: float = 10) -> NodeRuntimeHealth:
    """Execution-probed health of the PATH-resolved ``node`` (post-bootstrap)."""
    _platform.bootstrap_process_path()
    located = shutil.which("node")
    if not located:
        return NodeRuntimeHealth(status="missing", reason="not_on_path")
    if not os.path.isabs(located):
        # A relative PATH entry resolves against THIS process's cwd here but
        # against the skill/companion cwd at exec time: health is unprovable,
        # so the rollback never trusts it (full-scope finding F-2, the same
        # contract as the generic resolver's T10 branch).
        return NodeRuntimeHealth(
            status="missing", reason="relative_path_entry_unprovable", path=located,
        )
    return node_runtime_health(located, timeout_sec=timeout_sec)


def select_skill_node_runtime(timeout_sec: float = 10) -> Tuple[str, str]:
    """The ONE owner of the skill-family Node runtime precedence.

    Bundled-first: a skill's own runtime is the reviewed Node (``resolve_bundled_node``: the pin's, prepared or
    shipped) when it is execution-probed healthy, the python symmetry (python skills/companions already run on the
    embedded interpreter). Where the pin governs this host (``pinned_node_wait``) nothing else stands in, in a source
    checkout too (owner's choice, 14 Nov 2026): while it is unprepared the skill waits, and the reason names the pin,
    the path its preparation publishes and the older shipped copy it refused. Without a pin, or on a host the pin has
    no archive for, a health ROLLBACK to a healthy PATH node when the bundled runtime is absent or provably broken.

    Returns ``(path, provenance)`` with provenance ``"bundled"`` or ``"path"``.
    When no runtime is usable, returns ``("", reason)`` where the reason
    strings the probe verdicts verbatim for honest surface errors.
    """
    facts: List[str] = []
    bundled = _platform.resolve_bundled_node()
    if bundled:
        bundled_health = node_runtime_health(bundled, timeout_sec=timeout_sec)
        if bundled_health.healthy:
            return bundled, "bundled"
        facts.append(f"bundled:{bundled_health.status}:{bundled_health.reason}")
    pinned, planned = pinned_node_wait()
    if pinned:
        shipped = _platform._resolve_bundled_payload(_platform.embedded_node_candidates)
        older = node_runtime_health(shipped, timeout_sec=timeout_sec).version if shipped and not bundled else ""
        _manager, pin, pin_error = _pin_manager()
        facts.append(f"pin:unreadable:{pin_error}" if pin_error else
                     f"pinned:{pin.node_version}:not_prepared" + (f" (planned {planned})" if planned else "")
                     + (f"; refused shipped {shipped} v{older}" if older else ""))
        return "", "; ".join(facts) + ("; a manifest PATH chooses the runtime explicitly" if not pin_error else "")
    if not bundled:
        facts.append("bundled:absent")
    path_health = _path_node_runtime_health(timeout_sec=timeout_sec)
    if path_health.healthy:
        return path_health.path, "path"
    facts.append(
        f"path:{path_health.status}"
        + (f":{path_health.reason}" if path_health.reason else "")
    )
    return "", "; ".join(facts)


def skill_node_emergency_path_dir(timeout_sec: float = 10) -> str:
    """The selected reviewed Node's directory to PREPEND to a skill-family child's PATH, or ``""``.

    For a ``node`` that a child or a ``#!/usr/bin/env node`` launcher (npx, pnpm, yarn, an npm script) finds on PATH;
    an ``npm`` child of the skill family runs the reviewed Node itself (``reviewed_npm_command``). Where the pin governs
    this host (``pinned_node_wait``) the child PATH always starts with the pin's Node directory (owner's choice, 14 Nov
    2026: an ``npm`` companion runs the reviewed Node too, never a healthy PATH one). Without a pin it is an emergency
    only: the PATH node missing or execution-probed broken AND the healthy bundled node selected (on POSIX that
    directory holds just the ``node`` executable; on Windows the node-standalone root with ``node.exe``); on a healthy
    PATH ``""``, and the child env stays byte-identical. Unprepared under the pin: ``""`` here, and ``skill_node_argv``
    makes the child wait.

    Known residuals (disclosed): a launcher with an absolute node shebang ignores PATH (and fails or runs that Node
    honestly); a custom ``OUROBOROS_BUNDLE_DIR`` pointing at a FULL Node install also shadows npx with its copies while
    the prepend is active.
    """
    if not _platform.resolve_bundled_node():
        # No reviewed or bundled runtime here (unprepared pin, source checkouts, bench clones): nothing to prepend
        # and no probe is spent deciding that.
        return ""
    selected, provenance = select_skill_node_runtime(timeout_sec=timeout_sec)
    if not selected or provenance != "bundled":
        return ""
    if not pinned_node_wait()[0] and _path_node_runtime_health(timeout_sec=timeout_sec).healthy:
        return ""
    return str(pathlib.Path(selected).parent)


def prepend_skill_node_emergency_path(env: Dict[str, str], *, fallback_path: str = "") -> None:
    """Front-load a skill-family child's PATH with the reviewed node dir.

    The APPLYING half of ``skill_node_emergency_path_dir``: without a pin, on a
    healthy PATH, ``env`` is left untouched and the child environment stays
    byte-identical. Shared by the isolated-dep installer env and the
    extension companion spawn env so the two cannot drift on which node an
    `#!/usr/bin/env node` shebang resolves. ``fallback_path`` is the PATH the
    child would otherwise inherit, for an ``env`` carrying none of its own.
    """
    prepend_dir = skill_node_emergency_path_dir()
    if not prepend_dir:
        return
    existing = env.get("PATH") or fallback_path
    env["PATH"] = os.pathsep.join([prepend_dir, existing]) if existing else prepend_dir


def skill_manifest_owns_path(spec: Dict[str, Any]) -> bool:
    """Whether a skill's manifest declares its own PATH for a child process.

    An explicit manifest PATH means the author owns the runtime lookup, so
    neither the bundled argv rewrite below nor the emergency prepend above may
    shadow it (T14).
    """
    return any(str(key).upper() == "PATH" for key in (spec.get("env") or {}))


def skill_node_argv(spec: Dict[str, Any], declared_runtime: str, argv: List[str]) -> List[str]:
    """A node-family child's argv, rewritten onto the selected node runtime.

    T14 symmetry with the python -> ``sys.executable`` rewrite: python skills
    and companions already run on the embedded interpreter, so a node one runs
    on the runtime ``select_skill_node_runtime`` picked (bundled-first, health
    rollback included). Where the pin governs, an ``npm`` child runs on that Node
    too (``reviewed_npm_command``), and a ``node`` or ``npm`` child waits while the
    pin's Node is unprepared (``NodeRuntimeUnavailable``). A manifest PATH
    (``skill_manifest_owns_path``) chooses the runtime itself and is never
    rewritten or held. Without a pin, on a healthy PATH argv stays byte-identical.
    """
    if declared_runtime not in {"node", "npm"} or skill_manifest_owns_path(spec):
        return argv
    selected, reason = select_skill_node_runtime()
    if selected and argv and argv[0] == "node":
        return [selected, *argv[1:]]
    if not selected and argv and argv[0] in {"node", "npm"} and pinned_node_wait()[0]:
        raise NodeRuntimeUnavailable(f"no reviewed Node for this {argv[0]} child: {reason}")
    if selected and argv and argv[0] == "npm" and pinned_node_wait()[0]:
        _platform.bootstrap_process_path()
        return [*reviewed_npm_command(shutil.which("npm") or "npm", selected=selected), *argv[1:]]
    return argv


def reviewed_npm_command(launcher: str, *, selected: str = "") -> List[str]:
    """The command an ``npm`` child of the skill family starts with, in place of the resolved ``launcher``.

    Where the pin governs this host, npm runs on the reviewed Node as ``[node, npm-cli.js]``: an npm launcher can pick
    its own Node, not PATH's (the official Windows ``npm.cmd`` runs the ``node.exe`` beside it, PATH's only when there
    is none; a shim or a script may name any Node). The npm CLI is the one the launcher would load where its layout
    shows it: the launcher itself resolved to ``npm-cli.js`` (POSIX installs link it), ``npm/bin/npm-cli.js`` beside the
    ``node_modules/.bin`` holding it (an npm package installed locally, the version a skill chose), or
    ``node_modules/npm/bin/npm-cli.js`` beside it (Windows, as ``npm.cmd`` does; a global ``npm`` that ``npm.cmd`` would
    prefer through its prefix is not read). Only a locally installed npm comes before the pin's own npm, beside its Node
    in a complete generation (``ClaudexorRuntimeManager._managed_npm_cli``). With none, the launcher is refused on any
    OS (``NpmLauncherRefused``): it is never run as if it were the reviewed Node. A manifest PATH or an explicit command
    path chooses npm deliberately and never comes here. Without a pin, or on a host the pin has no archive for,
    ``[launcher]`` unchanged. ``selected``: the reviewed Node already chosen; while it is unprepared this waits."""
    if not pinned_node_wait()[0]:
        return [launcher]
    if not selected:
        selected, reason = select_skill_node_runtime()
        if not selected:
            raise NodeRuntimeUnavailable(f"no reviewed Node for this npm child: {reason}")
    from ouroboros.claudexor_runtime import ClaudexorRuntimeManager

    candidates = [ClaudexorRuntimeManager._managed_npm_cli(pathlib.Path(selected))]
    if os.path.isabs(launcher):
        found = pathlib.Path(launcher)
        local = [found.parent.name.lower(), found.parent.parent.name.lower()] == [".bin", "node_modules"]
        beside = found.parent.parent / "npm" if local else found.parent / "node_modules" / "npm"
        own = [found.resolve(), beside / "bin" / "npm-cli.js"]
        candidates = [*own, *candidates] if local else [*candidates, *own]
    cli = next((path for path in candidates if path.name == "npm-cli.js" and path.is_file()), None)
    if cli is not None:
        return [selected, str(cli)]
    raise NpmLauncherRefused(f"npm launcher {launcher!r} would choose its own Node: no npm CLI of it or of the pin is "
                             f"found to run on the reviewed Node {selected}; a manifest PATH or an explicit command path "
                             "chooses npm deliberately")


PLAYWRIGHT_NODE_ENV = "PLAYWRIGHT_NODEJS_PATH"
# Automatic driver selections carry their value beside the SDK variable through re-exec, child spawn and env reuse:
# it says only where the driver value came from.
SELECTED_DRIVER_NODE_ENV = "OUROBOROS_PLAYWRIGHT_NODE_SELECTED"
# The pinned Node a test or candidate boundary carried in from the installation it isolates, read only by the
# generation lookup (``claudexor_node_seed.generation_roots``): a fact of its own, beside any driver value.
CARRIED_NODE_ENV = "OUROBOROS_PINNED_NODE_CARRIED"
_FORWARD_REASONS_LOGGED: set = set()  # ``forward_playwright_node``: one warning per reason in this process


def _driver_env_value(environment, name: str) -> str:
    return next((str(value or "") for key, value in environment.items()
                 if (key.upper() if _platform.IS_WINDOWS else key) == name), "")


def _write_driver_node(environment, node: str, *, selected: bool) -> None:
    """Both driver variables of ``environment``, each assigned in place: another thread copying a shared one
    (``os.environ``, to spawn a child) never meets a key that is only changing as a missing one, and Playwright never
    starts between a removal and its return. Removed: a variable that has no value now, and on Windows a differently
    cased spelling, so the SDK reads the canonical one. The driver's value is written before its marker."""
    values = {PLAYWRIGHT_NODE_ENV: node, SELECTED_DRIVER_NODE_ENV: node if selected else ""}
    for key in list(environment):
        name = key.upper() if _platform.IS_WINDOWS else key
        if name in values and (key != name or not values[name]):
            environment.pop(key, None)
    environment.update({name: value for name, value in values.items() if value})


def _explicit_driver_node(*environments) -> str:
    """The first ``PLAYWRIGHT_NODEJS_PATH`` someone chose (a child's, a manifest's, the operator's), never one an
    Ouroboros process selected for itself, which an inherited or copied environment also carries."""
    for environment in environments:
        value = _driver_env_value(environment, PLAYWRIGHT_NODE_ENV)
        if value and value != _driver_env_value(environment, SELECTED_DRIVER_NODE_ENV):
            return value
    return ""


class NodeRuntimeUnavailable(RuntimeError):
    """A node child of Ouroboros's own (a skill's companion) has no reviewed Node; the message says why."""


class NpmLauncherRefused(NodeRuntimeUnavailable):
    """The reviewed Node is selected, but the npm launcher would run another (``reviewed_npm_command``): preparing the
    pin's Node again changes nothing, so nothing asks for it."""


def _pin_manager():
    """``(manager, pin, pin_error)`` of the reviewed runtime pin; a failing read is an unreadable pin."""
    try:
        from ouroboros.claudexor_runtime import get_runtime_manager

        manager = get_runtime_manager()
        return manager, manager.pin, manager.pin_error
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def pinned_node_wait() -> Tuple[bool, str]:
    """Whether the pin governs this host's own Node consumers (skills, companions, the attach role), and the path its
    preparation publishes: ``(True, planned)`` when it has an archive for this host, ``(True, "")`` when it is
    unreadable, ``(False, "")`` without a pin or without this host's archive (Windows on Arm, 32-bit Windows), where
    they keep a shipped copy or PATH. Read from the pin alone: nothing runs or is prepared."""
    manager, pin, pin_error = _pin_manager()
    if pin_error:
        return True, ""
    planned = manager.planned_node(pin) if pin is not None else ""
    return bool(planned), planned


def reviewed_host_node(bundled: "str | None") -> "str | None":
    """``resolve_bundled_node`` policy, the pin's Node or None: the shipped copy ``bundled`` when it runs as the pin's
    version, else the prepared managed generation (a boundary's carried one included) when it runs as the pin; when
    that recorded generation does not, the owner's probed lookup (``reviewed_node()``: the other generations, then an
    exact shipped copy); with no generation, an exact shipped copy at any base (``claudexor_node_seed.bundled_node``),
    with or without a first copy. Never an older shipped Node; None with an unreadable pin too (logged). Callers then
    decide: skills wait (``select_skill_node_runtime``), the web test lane may name a PATH Node
    (``preflight_node.resolve_node``). A broken first copy is returned when nothing qualifies, so callers keep
    reporting it; no pin at all (``release: null``) leaves the shipped copy as found. The hot path probes each file
    once per process (the health memo); only a recorded generation that does not run as the pin costs the owner's
    probed lookup on each call, as before; an unprepared pin runs nothing beyond the shipped copies found."""
    manager, pin, pin_error = _pin_manager()
    if pin_error:
        if bundled:
            log.warning("Shipped Node %s not used: the reviewed Node pin is unreadable (%s)", bundled, pin_error)
        return None
    if pin is None:
        return bundled
    health = node_runtime_health(bundled, timeout_sec=5) if bundled else None
    if health is not None and health.version == pin.node_version:
        return bundled
    try:
        node = manager.reviewed_node(prepared=True)  # by its record and file
    except Exception:
        node = ""
    if node and node_runtime_health(node, timeout_sec=5).version == pin.node_version:
        return node
    if node:  # a recorded generation that does not run as the pin: the owner's probed lookup tries the others first
        try:
            exact = manager.reviewed_node()
        except Exception:
            exact = ""
    else:
        from ouroboros.claudexor_node_seed import bundled_node

        exact = bundled_node(pin.node_version, probe=lambda path: node_runtime_health(path, timeout_sec=5).version)
    return exact or (bundled if health is not None and not health.healthy else None)


def discovered_host_node() -> "str | None":
    """``reviewed_host_node`` for an environment that only names a Node, running nothing: the pin's prepared managed
    Node by its record (a boundary's carried one included), with or without a shipped copy, else None, never a shipped
    copy, whose version is unknown without running it and may be older. None with an unreadable pin, as there; without
    a pin the first shipped copy as found. Whoever executes still probes what it runs."""
    manager, pin, pin_error = _pin_manager()
    if pin_error:
        return None
    if pin is None:
        return _platform._resolve_bundled_payload(_platform.embedded_node_candidates)
    try:
        return manager.reviewed_node(prepared=True)
    except Exception:
        return None


def select_playwright_node(env=None, *, allow_install: bool = False) -> str:
    """Point Playwright's driver (``PLAYWRIGHT_NODEJS_PATH``) at the pinned Node in ``env`` (this process).

    An explicit value keeps its authority. Otherwise the installation's exact Node (bundled or managed,
    ``allow_install`` permitting a Node-only install) is set; when none exists this raises
    ``ClaudexorRuntimeError`` rather than leave the SDK's own, possibly older Node in use. A host the pin has no Node
    archive for (Windows on Arm, 32-bit Windows) keeps that SDK Node, as children do, with a warning: it is not the
    security-qualified pin and nothing claims it is. The SDK reads only this
    process's environment, so the value is written there and marked as selected (``SELECTED_DRIVER_NODE_ENV``): never
    an explicit choice for a later call, a re-exec or a descendant (which select anew) or for
    ``forward_playwright_node``."""
    target = os.environ if env is None else env
    explicit = _explicit_driver_node(target)
    if explicit:
        if env is not None:  # this process's own is what the SDK reads already; a carried executable stays beside it
            _write_driver_node(target, explicit, selected=False)
        return explicit
    from ouroboros.claudexor_runtime import ClaudexorRuntimeError, get_runtime_manager

    manager = get_runtime_manager()
    try:
        node = manager.reviewed_node(allow_install=allow_install)
    except ClaudexorRuntimeError as exc:
        if manager.pin_error or manager.pin is not None and manager.planned_node(manager.pin):
            raise
        log.warning("Playwright keeps its SDK's own Node: %s (no reviewed Node for this host)", exc)
        node = ""
    _write_driver_node(target, node, selected=True)
    return node


def select_local_playwright_node() -> bool:
    """``forward_playwright_node`` for this process itself, before anything loads Playwright in it (the server's boot
    step), with one more local choice: an exact bundled Node, probed, while the pinned one is not prepared yet. True
    when the pin's Node still has to be prepared (copied from that bundle or downloaded); the caller prepares it off
    its serial path, and children get the planned path meanwhile. An operator's explicit driver stays as given and
    decides nothing else: the other host consumers of the pin's Node (skills, ``node --check``, the commit gate's
    Node lane) still get it prepared. False when it is prepared, or this host has no reviewed Node (the SDK's own)."""
    from ouroboros.claudexor_runtime import get_runtime_manager

    manager = get_runtime_manager()
    if _explicit_driver_node(os.environ):
        try:  # by record and file, as a child's lookup; os.environ is left as found, a carried executable included
            manager.reviewed_node(prepared=True)
        except Exception as exc:
            return bool(getattr(exc, "planned_node", ""))
        return False
    forward_playwright_node(os.environ)
    selected = _driver_env_value(os.environ, SELECTED_DRIVER_NODE_ENV)
    if not selected or os.path.isfile(selected):
        return False
    from ouroboros.claudexor_node_seed import bundled_node

    exact = bundled_node(manager.pin.node_version)
    if exact:
        _write_driver_node(os.environ, exact, selected=True)
    return True


def carry_pinned_node(env: Dict[str, str], data_root: "str | None") -> str:
    """Before a test boundary runs its command with a disposable data root: give ``env`` the pinned Node prepared in
    ``data_root``, the installation it isolates, found by record and file, nothing run. It is carried as its own fact
    (``CARRIED_NODE_ENV``, which the lookups inside read) and, unless ``env`` names the driver explicitly, as the
    automatic driver selection (both driver variables). An explicit value stays the driver's, byte-exact and unmarked,
    even when it equals the carried path. Nothing when ``env`` already carries one or that installation prepared none.
    Returns the carried executable or ``""``."""
    if not data_root or env.get(CARRIED_NODE_ENV):
        return ""
    from ouroboros.claudexor_runtime import get_runtime_manager

    try:
        node = get_runtime_manager().prepared_node(data_root)
    except Exception as exc:  # describing the caller's Node never stops the command
        log.warning("The caller's pinned Node was not looked up: %s", exc)
        return ""
    if node:
        env[CARRIED_NODE_ENV] = node
        if not _explicit_driver_node(env):
            _write_driver_node(env, node, selected=True)
    return node


def keep_explicit_driver_node(env: Dict[str, str], chosen) -> None:
    """``chosen`` (a manifest's env, already overlaid onto ``env``) names the driver Node itself: it stays as given and
    is never again an automatic selection, even when its value equals the one selected before."""
    value = _driver_env_value(chosen, PLAYWRIGHT_NODE_ENV)
    if value:
        _write_driver_node(env, value, selected=False)


def forward_playwright_node(env: Dict[str, str]) -> str:
    """Give one child environment the driver Node before any browser call: looked up by record and file, nothing run,
    written or installed, so building an environment never starts a process.

    An explicit value (the child's own, a manifest's or the operator's) is kept as given; a selected one is not.
    Otherwise the prepared pinned managed Node (by the boot step ``server_boot.prepare_pinned_node``, the browser tool or Android provisioning).
    Not prepared: the child gets ``planned_node``, the real generation a later preparation publishes, so only a
    Playwright launch fails, naming it, never the SDK's own Node; recovery makes that same path valid. Nothing is
    written to this process's environment unless it is ``env`` (the boot step), and then only after the lookup, which
    reads the executable a test or candidate boundary carried in. Every automatic path carries its selection marker
    through child spawn and reuse (a re-exec drops both, ``without_selected_driver_node``); an explicit override
    clears it. Returns ``""`` or the human-readable reason, logged as a
    warning once per reason and process, then at debug: every child environment built while unprepared has it."""
    explicit = _explicit_driver_node(env, os.environ)
    if explicit:
        _write_driver_node(env, explicit, selected=False)
        return ""
    try:
        from ouroboros.claudexor_runtime import get_runtime_manager

        node, reason = get_runtime_manager().reviewed_node(prepared=True), ""
    except Exception as exc:
        node = getattr(exc, "planned_node", "")
        reason = (f"Playwright in this child has no reviewed Node ({exc}); "
                  + (f"its driver path {node} becomes valid once the boot step, an acting consumer or a browser tool "
                     "call outside read-only work prepares it." if node else "it would run its SDK's own Node.")
                  + " Set PLAYWRIGHT_NODEJS_PATH to choose one.")
        log.log(logging.DEBUG if reason in _FORWARD_REASONS_LOGGED else logging.WARNING, reason)
        _FORWARD_REASONS_LOGGED.add(reason)
    _write_driver_node(env, node, selected=True)
    return reason


def without_selected_driver_node(env: Dict[str, str]) -> Dict[str, str]:
    """``env`` for a successor this process re-execs into (restart, update, rollback) or a server a benchmark starts:
    an automatic selection is not handed on, neither its value (a body from before the marker would take it for the
    operator's choice and start its driver from a path that may never have been prepared) nor its marker; every body
    selects anew at its boot. The executable a boundary carried in (``CARRIED_NODE_ENV``) stays for the successor's
    lookups. An explicit value is the operator's and stays byte-exact."""
    if not _explicit_driver_node(env):
        for key in [key for key in env if (key.upper() if _platform.IS_WINDOWS else key) in
                    (PLAYWRIGHT_NODE_ENV, SELECTED_DRIVER_NODE_ENV)]:
            del env[key]
    return env
