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

import os
import pathlib
import shutil
import subprocess
from typing import Any, Dict, List, NamedTuple, Tuple

from ouroboros import platform_layer as _platform


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

    Bundled-first: a skill's own runtime is the packaged, signed node when it
    is present AND execution-probed healthy (the python symmetry — python
    skills/companions already run on the embedded interpreter) — with a health
    ROLLBACK to a healthy PATH node when the bundled runtime is absent or
    provably broken. A candidate that fails its execution probe is never
    selected while a usable neighbour exists.

    Returns ``(path, provenance)`` with provenance ``"bundled"`` or ``"path"``.
    When neither runtime is usable, returns ``("", reason)`` where the reason
    strings both probe verdicts verbatim for honest surface errors.
    """
    facts: List[str] = []
    bundled = _platform.resolve_bundled_node()
    if bundled:
        bundled_health = node_runtime_health(bundled, timeout_sec=timeout_sec)
        if bundled_health.healthy:
            return bundled, "bundled"
        facts.append(f"bundled:{bundled_health.status}:{bundled_health.reason}")
    else:
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
    """Bundled-node dir to PREPEND to a skill-family child PATH — emergency only.

    npm/npx/pnpm/yarn are NOT bundled and are never rewritten; their launchers
    resolve node via a ``#!/usr/bin/env node`` shebang. Only when the PATH node
    is missing or execution-probed broken AND the healthy bundled node was
    selected does the child PATH gain the bundled-node directory (on POSIX it
    contains exactly the ``node`` executable, so nothing else is shadowed; on
    Windows it is the node-standalone root holding ``node.exe``). On a healthy
    PATH this returns ``""`` and the child env stays byte-identical.

    Known residuals (disclosed): an npm launcher rewritten to an absolute node
    shebang ignores PATH and keeps failing honestly; and the "nothing else is
    shadowed" claim holds for the official download scripts (which prune the
    archive to the bare node binary) — a custom ``OUROBOROS_BUNDLE_DIR``
    pointing at a FULL Node install would also shadow npm/npx with the bundled
    copies while the emergency is active.
    """
    if not _platform.resolve_bundled_node():
        # No bundled runtime installed (source checkouts, bench clones): no
        # emergency lane exists and no probe is spent deciding that.
        return ""
    selected, provenance = select_skill_node_runtime(timeout_sec=timeout_sec)
    if not selected or provenance != "bundled":
        return ""
    if _path_node_runtime_health(timeout_sec=timeout_sec).healthy:
        return ""
    return str(pathlib.Path(selected).parent)


def prepend_skill_node_emergency_path(env: Dict[str, str], *, fallback_path: str = "") -> None:
    """Front-load a skill-family child's PATH with the emergency node dir.

    The APPLYING half of ``skill_node_emergency_path_dir``: on a healthy PATH
    there is no emergency, ``env`` is left untouched and the child environment
    stays byte-identical. Shared by the isolated-dep installer env and the
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
    rollback included). ``npm`` is never rewritten — its launcher resolves node
    through a ``#!/usr/bin/env node`` shebang, which the emergency PATH prepend
    covers instead. On a healthy PATH argv stays byte-identical.
    """
    if declared_runtime not in {"node", "npm"} or skill_manifest_owns_path(spec):
        return argv
    selected, _provenance = select_skill_node_runtime()
    if selected and argv and argv[0] == "node":
        return [selected, *argv[1:]]
    return argv
