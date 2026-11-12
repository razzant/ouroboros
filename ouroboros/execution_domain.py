"""A managed deployment's execution domain: where its employees' work runs, apart from the trusted service.

A deployment policy (``deployment_policy``) may name ``execution``; without it, or unmanaged, nothing here changes
behaviour: commands keep their local or task-scoped executor and every Claudexor operation the owned engine::

    "execution": {"executor": {"type": "docker_exec", "container_name": "<work container>", "network": "host",
                               "path_mappings": [{"host_path": "/srv/work", "backend_path": "/srv/work"}]},
                  "engine": {"descriptor": "<absolute path of the work engine's control descriptor>"},
                  "dedicated_containers": ["<work container>", "<work engine container>"],
                  "skill_state_root": "<persistent work folder for skills' own runtime state>",
                  "staging_root": "<work folder for private copies, staged inputs, scratch, skill copies>"}

``executor`` is the installation-pinned ``workspace_executor`` backend (``managed_executor_ref``): foreground
commands, scripts, checks and services run there whatever a task or its environment names, and so do the processes
the service attaches to its own streams, stdio MCP servers, skill scripts and dependency builds
(``attached_launch``). A path outside its mappings refuses before anything starts
(``workspace_executor_paths.ManagedPathRefused``), never on the service, and ``network: none`` is checked before each
start as the executor checks its own. Its mappings may not reach what enforces the policy (the data root, the
checkout, the policy file, the engine descriptor) or the service's import path. ``skill_state_root`` is where each
skill's own processes keep their runtime state (``skill_state_dir``), made in the domain as its user; the service's
records about the skill stay its own.

``engine`` names the Claudexor engine that owns native runs, through a descriptor the administrator writes on the
service (``{"host", "port", "tokenPath"}``, loopback only, outside the data root and the checkout); the token stays
in the gateway. Raw model operations keep the service's own engine. A new native run records ``engine_ref`` on its
pending invocation before the POST and on its STARTED row (with the store the handshake proved); every later
operation on it opens that recorded engine (``run_gateway``), a retry or continuation included, and a generation
whose policy names another engine refuses it (``execution_domain_changed``) instead of migrating it. A run
recorded without one belongs to the owned engine. A store answers only for its own runs (``same_engine``). Without
``engine`` a domain starts no native run. The work engine's ``runDir`` names a path inside the domain: the service
never reads it (``WorkEngineGateway``). Every path a native run's request names must be one the domain sees at the
same address (``request_paths_refusal``). Every effect on a work path — a file written or removed, Git over a work
repository, a private copy, its capture and apply, a native run's staged inputs — runs as the work domain through
this executor's custody (``work_domain``); ``staging_root`` is the administrator's work folder that holds what the
service prepares for the domain (``<staging_root>/copies``, ``scratch``, ``tmp``, and ``skills`` for an
installed script skill's verified payload copy, ``work_skill_payloads``).

``dedicated_containers`` belong to this domain alone. Panic stops them by the identity this server generation
measured (``panic_request``), never by name, and never waits; its settlement reads whether each stopped
(``panic_settle``); a container recreated after the measurement stays unconfirmed. Nothing restarts them.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import pathlib
import posixpath
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from ouroboros.workspace_executor_paths import ExecutorRef, map_host_path, normalize_executor_ref

log = logging.getLogger(__name__)

POLICY_FIELD = "execution"
WORK_ENGINE = "work"
_DOCKER_TIMEOUT_SEC = 10.0  # one Docker CLI call: the transport's bound, never a judgment
_LOCK = threading.Lock()
_MEASURED: Dict[str, str] = {}  # container name -> id this server generation measured (Panic runs in this process)


@dataclass(frozen=True)
class ExecutionDomain:
    executor: ExecutorRef
    engine_descriptor: str = ""
    dedicated_containers: Tuple[str, ...] = ()
    skill_state_root: str = ""
    staging_root: str = ""
    # Each mapping as the administrator spelled it and as it resolved, with its container spelling: which domain a
    # path belongs to is read from its spelling against both (``work_domain.work_spelling``), never a later resolve.
    work_roots: Tuple[Tuple[str, str], ...] = ()


# --- the policy section --------------------------------------------------------------------------------

def _fields(value: Any, where: str, allowed: Tuple[str, ...]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"{where} has the unsupported field {unknown[0]!r}")
    return value


def _service_roots() -> List[Tuple[str, str]]:
    from ouroboros.launch_roots import configured_roots

    roots = configured_roots()
    return [("data root", os.path.realpath(roots.data_dir)), ("checkout", os.path.realpath(roots.repo_dir))]


def _protected_roots(descriptor: str) -> List[Tuple[str, str]]:
    """What the work domain may never reach: what enforces the policy (the data root, the checkout, the policy file,
    the work engine's descriptor) and every folder the service's interpreter imports code from (never ``/``)."""
    import sys

    from ouroboros import deployment_policy

    policy = os.environ.get(deployment_policy.POLICY_ENV, "").strip()
    named = [("deployment policy", policy), ("work engine descriptor", descriptor)]
    imports = {os.path.realpath(path) for path in (sys.prefix, sys.base_prefix, sys.exec_prefix, *sys.path)
               if isinstance(path, str) and os.path.isabs(path)}
    return [*_service_roots(), *((name, os.path.realpath(path)) for name, path in named if path),
            *(("service's import path", root) for root in sorted(imports) if root != os.path.sep)]


def _overlaps(path: str, base: str) -> bool:
    common = os.path.commonpath([path, base])
    return common in (path, base)


def _outside_service(text: Any, where: str, roots: Optional[List[Tuple[str, str]]] = None) -> str:
    """An absolute path that neither lies in nor contains the data root or the checkout (or any of ``roots``)."""
    if not isinstance(text, str) or not text or not os.path.isabs(text) or text != text.strip():
        raise ValueError(f"{where} must be an absolute path")
    resolved = os.path.realpath(text)
    for name, base in _service_roots() if roots is None else roots:
        if _overlaps(resolved, base):
            raise ValueError(f"{where} overlaps the {name}, which the work domain must never reach; keep them apart")
    return text


def parse_execution(value: Any) -> ExecutionDomain:
    """The ``execution`` section, strictly; ``ValueError`` names the first problem (the policy wraps it)."""
    section = _fields(value, "execution", ("executor", "engine", "dedicated_containers", "skill_state_root",
                                           "staging_root"))
    raw = _fields(section.get("executor"), "execution.executor", ("type", "container_name", "network", "path_mappings"))
    if raw.get("type") != "docker_exec":
        raise ValueError("execution.executor.type must be 'docker_exec': a local executor runs on the service")
    name = raw.get("container_name")
    if not isinstance(name, str) or not name.strip() or name != name.strip() or any(ch.isspace() for ch in name):
        raise ValueError("execution.executor.container_name must name one container")
    descriptor = ""
    if "engine" in section:
        engine = _fields(section["engine"], "execution.engine", ("descriptor",))
        descriptor = _outside_service(engine.get("descriptor"), "execution.engine.descriptor")
    protected = _protected_roots(descriptor)
    mappings = raw.get("path_mappings")
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("execution.executor.path_mappings must list at least one mapping")
    for index, item in enumerate(mappings):
        item = _fields(item, f"execution.executor.path_mappings[{index}]", ("host_path", "backend_path"))
        _outside_service(item.get("host_path"), f"execution.executor.path_mappings[{index}].host_path", protected)
    try:
        executor = replace(normalize_executor_ref({**raw, "id": "deployment"}), managed=True)
    except ValueError as exc:
        raise ValueError(f"execution.executor: {exc}") from None
    state_root, staging_root = (_work_folder(section, field, executor, protected)
                                for field in ("skill_state_root", "staging_root"))
    containers = section.get("dedicated_containers", [])
    if (not isinstance(containers, list) or len(set(containers)) < len(containers)
            or any(not isinstance(item, str) or not item.strip() or item != item.strip() for item in containers)):
        raise ValueError("execution.dedicated_containers must list distinct container names")
    if descriptor and not containers:
        raise ValueError("execution.dedicated_containers must name the work engine's container: Panic stops "
                         "native sessions only through it")
    from ouroboros.workspace_executor_paths import _normalize_backend_path

    work_roots = tuple(sorted({(posixpath.normpath(str(item["host_path"])), _normalize_backend_path(str(item["backend_path"])))
                               for item in mappings}
                              | {(str(mapping.host_path), mapping.backend_path) for mapping in executor.mappings}))
    return ExecutionDomain(executor, descriptor, tuple(containers), state_root, staging_root, work_roots)


def _work_folder(section: dict, field: str, executor: ExecutorRef, protected: List[Tuple[str, str]]) -> str:
    """An administrator's work folder (``skill_state_root``, ``staging_root``): normalized, apart from what enforces
    the policy, and seen by the work at the same address (inside an identical path mapping); "" when unnamed."""
    if field not in section:
        return ""
    folder = _outside_service(section[field], f"execution.{field}", protected)
    try:
        seen = map_host_path(executor, pathlib.Path(folder))
    except ValueError:
        seen = ""
    if posixpath.normpath(folder) != folder or seen != folder:
        raise ValueError(f"execution.{field} must be a normalized path the work sees at the same address "
                         "(inside an identical path mapping)")
    return folder


def current() -> Optional[ExecutionDomain]:
    """This process's execution domain (``None``: unmanaged, or a policy without ``execution``); an unusable
    policy raises, as every policy reader does."""
    from ouroboros import deployment_policy

    policy = deployment_policy.current()
    return policy.execution if policy is not None else None


def managed_executor_ref() -> Optional[ExecutorRef]:
    """The installation-pinned executor, which no task or environment selector replaces."""
    domain = current()
    return domain.executor if domain is not None else None


# --- engine identity -----------------------------------------------------------------------------------

def is_work_ref(ref: Any) -> bool:
    return isinstance(ref, dict) and ref.get("kind") == WORK_ENGINE


def recorded_ref(record: Any) -> dict:
    """The engine a custody row, invocation record or reference names; ``{}`` is the owned engine."""
    if isinstance(record, dict):
        ref = record if "kind" in record else record.get("engine_ref")
    else:
        ref = getattr(record, "engine_ref", None)
    return ref if isinstance(ref, dict) else {}


def same_engine(gateway: Any, record: Any) -> bool:
    """Whether ``gateway`` reaches the engine ``record`` is bound to, so its answer (404 included) speaks for it."""
    ref, held = recorded_ref(record), getattr(gateway, "engine_ref", None)
    if not is_work_ref(ref):
        return not is_work_ref(held)
    return (is_work_ref(held) and held.get("descriptor") == ref.get("descriptor")
            and (not ref.get("store") or held.get("store") == ref.get("store")))


def engine_key(record: Any) -> Optional[Tuple[str, str]]:
    """A work engine's identity as ``record`` names it (descriptor, store); ``None`` for the owned engine."""
    ref = recorded_ref(record)
    return (str(ref.get("descriptor") or ""), str(ref.get("store") or "")) if is_work_ref(ref) else None


def engine_groups(*collections: Iterable[Any]) -> List[Tuple[dict, List[List[Any]]]]:
    """Split parallel collections (runs, pending invocations) by bound engine, the owned engine's group first;
    each group keeps one list per collection, in order. One group with an empty reference: nothing is bound."""
    groups: Dict[Any, Tuple[dict, List[List[Any]]]] = {None: ({}, [[] for _ in collections])}
    for index, items in enumerate(collections):
        for item in items or ():
            key = engine_key(item)
            if key not in groups:
                groups[key] = (dict(recorded_ref(item)), [[] for _ in collections])
            groups[key][1][index].append(item)
    owned = groups.pop(None)
    return [owned, *groups.values()]


# --- gateways ------------------------------------------------------------------------------------------

def _unavailable(code: str, message: str) -> Exception:
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    return ClaudexorUnavailable(code, message)


@functools.lru_cache(maxsize=1)
def _work_gateway_class():
    from ouroboros.gateways.claudexor import ClaudexorGateway

    class WorkEngineGateway(ClaudexorGateway):
        """The work engine's transport: its run detail's ``runDir`` is a path inside the domain, so it is moved to
        ``workRunDir`` and no reader on the service dereferences it; a start names only paths the domain sees."""

        engine_ref: Optional[dict] = None

        def get_run(self, run_id: str, *, timeout_sec: Optional[float] = None) -> Dict[str, Any]:
            detail = super().get_run(run_id, timeout_sec=timeout_sec)
            summary = detail.get("summary") if isinstance(detail, dict) else None
            if isinstance(summary, dict) and "runDir" in summary:
                detail["summary"] = {**{k: v for k, v in summary.items() if k != "runDir"},
                                     "workRunDir": summary["runDir"]}
            return detail

        def start_run(self, request: Dict[str, Any], *, idempotency_key: str = "") -> Dict[str, Any]:
            refusal = request_paths_refusal(request)
            if refusal:
                raise _unavailable("work_path_unmapped", refusal)
            return super().start_run(request, idempotency_key=idempotency_key)

    return WorkEngineGateway


def work_gateway(ref: dict, *, timeout_sec: Optional[float] = None) -> Any:
    """An authenticated gateway to the work engine ``ref`` names, as this generation's policy reaches it; a policy
    that names another engine or none refuses, and a different store refuses a reference that names its own."""
    from ouroboros.gateways.claudexor import _endpoint_from_descriptor

    domain = current()
    descriptor = str(ref.get("descriptor") or "")
    if domain is None or not domain.engine_descriptor:
        raise _unavailable("execution_domain_unavailable", "This work belongs to the deployment's work engine, which "
                           "this server generation's policy no longer names; it is kept, not moved to another engine.")
    if descriptor and descriptor != domain.engine_descriptor:
        raise _unavailable("execution_domain_changed", "This work belongs to a work engine this generation's policy "
                           "no longer names; it is kept, not moved to the engine the policy names now.")
    gateway = _work_gateway_class()(_endpoint_from_descriptor(pathlib.Path(domain.engine_descriptor)))
    try:
        gateway.handshake(**({"timeout_sec": timeout_sec} if timeout_sec is not None else {}))
        store = str(gateway.store_identity or "")
        if not store:
            raise _unavailable("work_engine_identity_unproven", "The work engine did not prove its store identity, so "
                               "its answers could not be told apart from another engine's; nothing was done.")
        if ref.get("store") and ref["store"] != store:
            raise _unavailable("work_engine_store_changed", "The work engine now serves another store than the one "
                               "this work was recorded in; its answers speak only for that store, so nothing was done.")
        gateway.engine_ref = {"kind": WORK_ENGINE, "descriptor": domain.engine_descriptor, "store": store}
    except BaseException:
        gateway.close()
        raise
    return gateway


def native_gateway(**owned_kwargs: Any) -> Any:
    """The engine a NEW native run and its route checks use: the domain's work engine, else the owned engine."""
    domain = current()
    if domain is None:
        from ouroboros import claudexor_daemon

        return claudexor_daemon.ensure_owned_gateway(**owned_kwargs)
    if not domain.engine_descriptor:
        raise _unavailable("work_engine_unconfigured", "This deployment's execution domain names no work engine, "
                           "and native sessions never run on the service; nothing was started.")
    return work_gateway({"kind": WORK_ENGINE, "descriptor": domain.engine_descriptor})


def run_gateway(record: Any, default: Callable[[], Any], *, timeout_sec: Optional[float] = None) -> Any:
    """The gateway of the engine ``record`` is bound to: ``default()`` for the owned engine (the caller's own
    discovery, unchanged), else the recorded work engine, its handshake bounded by ``timeout_sec``."""
    ref = recorded_ref(record)
    return work_gateway(ref, timeout_sec=timeout_sec) if is_work_ref(ref) else default()


def engine_factory(ref: dict, default: Callable[[], Any]) -> Callable[[], Any]:
    return (lambda: work_gateway(ref)) if is_work_ref(ref) else default


class _Unreachable:
    """A work engine this pass cannot reach now: each operation raises its typed cause, so every caller keeps its
    own transport-failure handling (wait for a later pass), and no store identity answers for anything."""

    store_identity = ""
    engine_version = ""
    engine_ref = None  # bound to nothing: never the engine any record names

    def __init__(self, cause: BaseException) -> None:
        self._cause = cause

    def close(self) -> None:
        """Nothing was opened."""

    def __getattr__(self, _name: str) -> Callable[..., Any]:
        def unreachable(*_args: Any, **_kwargs: Any) -> Any:
            raise _unavailable(str(getattr(self._cause, "code", "") or "work_engine_unreachable"), str(self._cause))
        return unreachable


class EngineClients:
    """The engines one pass reaches for many runs and engine-held objects (``delegate_collection``,
    ``review_thread_retention``): ``owned()`` for the owned engine, and one reused client per bound work engine,
    opened outside the caller's lock on first need, dropped after a transport fault, closed with its owner."""

    def __init__(self, owned: Optional[Callable[[], Any]] = None) -> None:
        self._owned = owned
        self._lock = threading.Lock()
        self._clients: Dict[Tuple[str, str], Any] = {}

    def held(self, record: Any) -> Any:
        with self._lock:
            return self._clients.get(engine_key(record))

    def get(self, record: Any) -> Any:
        """The client of the work engine ``record`` is bound to (``work_gateway``'s refusals raise)."""
        client = self.held(record)
        if client is not None:
            return client
        client = work_gateway(recorded_ref(record))
        with self._lock:
            kept = self._clients.setdefault(engine_key(record), client)
        if kept is not client:
            client.close()
        return kept

    def reach(self, record: Any) -> Any:
        """``record``'s own engine: ``owned()`` when unbound, else its work engine's client, or a stand-in raising the
        typed cause while that engine cannot be reached (what it holds waits for a later pass)."""
        if not is_work_ref(recorded_ref(record)):
            return self._owned() if self._owned is not None else None
        try:
            return self.get(record)
        except Exception as exc:
            return _Unreachable(exc)

    def for_store(self, store_id: Any = "") -> Any:
        """The engine whose store answers for ``store_id``: the owned engine unless the domain's work engine serves
        that store (a mismatch then keeps the caller waiting, as the store check of each owner decides)."""
        owned = self._owned() if self._owned is not None else None
        if not store_id or str(getattr(owned, "store_identity", "") or "") == store_id or current() is None:
            return owned
        work = self.reach({"kind": WORK_ENGINE})
        return work if str(getattr(work, "store_identity", "") or "") == store_id else owned

    def drop(self, client: Any = None) -> bool:
        """Close ``client`` (every work client when ``None``); whether this owner held what it closed."""
        with self._lock:
            gone = [held for held in self._clients.values() if client is None or held is client]
            self._clients = {key: held for key, held in self._clients.items() if held not in gone}
        for held in gone:
            held.close()
        return bool(gone)


def start_engine(drive: Any, retry_token: str = "", continuation_token: str = "") -> Tuple[dict, Any]:
    """``(custody facts, gateway)`` of one delegate start: a retry replays the engine its pending invocation recorded,
    a continuation its predecessor's, a fresh start ``native_gateway``. The facts ride the pending invocation before
    the POST and the STARTED row (``{}`` for the owned engine, so those rows stay as they were)."""
    if retry_token or continuation_token:
        try:
            if retry_token:
                from ouroboros.delegate_invocation import invocation_record

                ref = recorded_ref(invocation_record(drive, retry_token) or {})
            else:
                from ouroboros.delegate_custody_current import runs as current_runs

                ref = recorded_ref(current_runs(drive, "run_id=?", (continuation_token,)).get(continuation_token))
        except Exception as exc:  # the binding decides the engine: unread, nothing is sent anywhere
            raise _unavailable("engine_binding_unreadable", "The engine this start must reuse could not be read "
                               f"({type(exc).__name__}); nothing was started.") from exc
        from ouroboros import claudexor_daemon

        gateway = run_gateway(ref, claudexor_daemon.ensure_owned_gateway)
    else:
        gateway = native_gateway()
    ref = getattr(gateway, "engine_ref", None)
    return ({"engine_ref": dict(ref)} if is_work_ref(ref) else {}), gateway


def seat_gateway(record: Any, root: str) -> Tuple[dict, Any]:
    """``(custody facts, gateway)`` of one review seat: a recovered invocation's recorded engine, else the native
    engine, whose domain must see the seat's root at the same address."""
    if record:
        from ouroboros import claudexor_daemon

        gateway = run_gateway(record, claudexor_daemon.ensure_owned_gateway)
    else:
        if refusal := request_paths_refusal({"scope": {"root": root}}):
            raise _unavailable("work_path_unmapped", refusal)
        gateway = native_gateway()
    ref = getattr(gateway, "engine_ref", None)
    return ({"engine_ref": dict(ref)} if is_work_ref(ref) else {}), gateway


# --- attached launches: stdio MCP servers, skill scripts, dependency builds -----------------------------

# The container keeps its own shell state, and every home, temp folder, login or bytecode cache the caller merely
# inherited from the service (the same value as the service's own); a value the caller chose itself rides along.
_CONTAINER_ENV = frozenset({"PWD", "OLDPWD", "SHELL"})
_INHERITED_ENV = frozenset({"HOME", "TMPDIR", "TMP", "TEMP", "USER", "LOGNAME", "USERNAME", "PYTHONPYCACHEPREFIX",
                            "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SYSTEMROOT"})
# The service-side CLI's own lookup and configuration: never forwarded into the container.
_CLI_ENV = ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "DOCKER_CERT_PATH", "DOCKER_TLS_VERIFY")


@dataclass
class AttachedLaunch:
    """One process the pinned executor runs with its standard streams attached to the service's Docker CLI.

    ``argv``/``env`` are what the host starts (the CLI); ``pidfile`` is the launched group's leader in the
    container, then ``completed``. ``adopt`` writes the executor's foreground record before the start, so Stop and
    Panic settle the container side through the executor's own records (``workspace_executor``); ``settle``
    confirms the group ended (stopping it when it did not) and retires the record; ``abandon`` retires the record of
    a start that never happened. Ending the CLI alone never ends the process in the container."""

    argv: List[str]
    env: Dict[str, str]
    container: str
    pidfile: str
    backend_cwd: str = ""
    record: Optional[pathlib.Path] = None

    def require_cli(self) -> None:
        """The one positive pre-spawn proof a caller may abandon on: the service finds no Docker CLI on the launch's
        own search path, so nothing can start (``process_not_started``). Any failure after this point may follow a
        spawn, and is settled (``settle``), never abandoned."""
        import errno
        import shutil

        if shutil.which(self.argv[0], path=self.env.get("PATH", "")) is None:
            missing = FileNotFoundError(errno.ENOENT, "the service has no Docker CLI to start this work-domain "
                                        "process; nothing was started", self.argv[0])
            missing.process_not_started = True
            raise missing

    def adopt(self, drive_root: Any = None) -> None:
        from ouroboros import workspace_executor as executor

        self.record = executor._register_process(executor._custody_drive_root(drive_root or _data_root()), {
            "record_type": "foreground", "executor_type": "docker_exec", "executor_id": "deployment",
            "host_pid": 0, "container_name": self.container, "backend_pidfile": self.pidfile,
            "backend_cwd": self.backend_cwd, "cmd": executor._redacted_cmd(self.argv[-1:])})

    def settle(self) -> bool:
        from ouroboros import workspace_executor as executor

        done = executor._cleanup_docker_exec_timeout(self.container, self.pidfile)
        return bool(done and (self.record is None or executor._retire_docker_completion(self.record)))

    def abandon(self) -> None:
        from ouroboros import workspace_executor as executor

        executor._forget_process(self.record)


def _data_root() -> pathlib.Path:
    from ouroboros import config

    return pathlib.Path(config.DATA_DIR)


def attached_launch(cmd: List[str], cwd: Any = None, env: Optional[Dict[str, str]] = None) -> Optional[AttachedLaunch]:
    """``None`` unmanaged. Under a managed domain, ``cmd`` as its pinned executor runs it with the host's stdin,
    stdout and stderr; a working directory or a path-valued variable the domain does not see at the same address
    refuses before anything starts (``ManagedPathRefused``), and nothing runs on the service instead, as does a
    ``network: none`` container that has a network (``_network_none``). What the caller only inherited from the
    service and its scope markers stay on the service (``_work_env``); its own PATH entries precede the container's."""
    import shlex

    from ouroboros.workspace_executor import _docker_env_command, _docker_exec_wrapper
    from ouroboros.workspace_executor_paths import ManagedPathRefused

    domain = current()
    if domain is None:
        return None
    executor = domain.executor
    backend_cwd = map_host_path(executor, pathlib.Path(cwd)) if cwd else ""
    target = _work_env(env)
    unseen = sorted(key for key, value in target.items() if _unseen_paths(executor, value))
    if unseen:
        raise ManagedPathRefused(f"{', '.join(unseen)} name paths outside this deployment's execution domain; nothing "
                                 "was started, and nothing runs on the service instead")
    if executor.network == "none":  # the selected contract, checked before each start as the executor's own starts are
        _network_none(executor)
    prefix = f"OUROBOROS_PROCESS_ENV_{uuid.uuid4().hex}_"
    aliases = {key: f"{prefix}{index}" for index, key in enumerate(target)}
    pidfile = f"/tmp/ouroboros-exec-{uuid.uuid4().hex}.pid"  # the executor's foreground pidfile form
    command = _docker_env_command(shlex.join(str(part) for part in cmd), aliases)
    if "PATH" in aliases:  # the caller's own entries first, then the container's own search
        command = command.replace(f'PATH="${{{aliases["PATH"]}}}"', f'PATH="${{{aliases["PATH"]}}}:$PATH"', 1)
    wrapper = _docker_exec_wrapper(command, pidfile)
    argv = ["docker", "exec", "-i", *[part for alias in aliases.values() for part in ("--env", alias)],
            *(["--workdir", backend_cwd] if backend_cwd else []), executor.container_name, "sh", "-c", wrapper]
    cli_env = {key: os.environ[key] for key in _CLI_ENV if key in os.environ}
    cli_env.update({key: value for key, value in (env or {}).items() if key.startswith("OURO_")})  # host census
    cli_env.update({aliases[key]: value for key, value in target.items()})
    return AttachedLaunch(argv, cli_env, executor.container_name, pidfile, backend_cwd)


def _network_none(executor: ExecutorRef) -> None:
    """``network: none`` holds only while its container has no network: the executor's own check
    (``workspace_executor._assert_docker_network_none``); whatever it raises, nothing was started."""
    from ouroboros import workspace_executor

    try:
        workspace_executor._assert_docker_network_none(executor.container_name)
    except RuntimeError as refused:
        refused.process_not_started = True
        raise
    except Exception as exc:  # an unanswered check proves nothing: refused, as a different network would be
        refused = RuntimeError(f"the network of executor container {executor.container_name} could not be checked "
                               f"({type(exc).__name__}); nothing was started")
        refused.process_not_started = True
        raise refused from exc


def _work_env(env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """The variables the launched process receives in the container (``attached_launch``)."""
    service_path = set(os.environ.get("PATH", "").split(os.pathsep))
    target = {}
    for key, value in (env or {}).items():
        value = str(value)
        if key.startswith("OURO_") or key in _CONTAINER_ENV or (key in _INHERITED_ENV and value == os.environ.get(key)):
            continue
        if key == "PATH":  # the service's own search is the service's: only the caller's additions cross
            value = os.pathsep.join(part for part in value.split(os.pathsep) if part and part not in service_path)
            if not value:
                continue
        target[key] = value
    return target


def _unseen_paths(executor: ExecutorRef, value: str) -> bool:
    """Whether a variable's value names an absolute path (or a path list) the domain does not map identically."""
    for part in value.split(os.pathsep):
        if part.startswith("/") and not part.startswith("//") and part != "/dev/null":  # no URL ``//host``, no device
            try:
                if map_host_path(executor, pathlib.Path(part)) != posixpath.normpath(part):
                    return True
            except ValueError:
                return True
    return False


# --- what a skill's own processes keep -----------------------------------------------------------------

def skill_state_dir(drive_root: Any, name: str) -> pathlib.Path:
    """The folder a skill's own processes keep their runtime state in: its ``OUROBOROS_SKILL_STATE_DIR``, Deno's write
    permission and Go's cache, packages and build output (``tools/skill_exec``), one path for all of them.

    Unmanaged, the skill's state folder under the data root, as before (``skill_loader.skill_state_dir``). Managed,
    ``execution.skill_state_root``/<canonical skill name>: a folder of the work domain's own, made there as its user.
    The service's records about the skill (review, grants, enablement) stay in the service's state folder; none of
    them is mounted or copied, and runtime bytes an earlier run kept there are not migrated. A domain naming no such
    root refuses before anything starts (``ManagedPathRefused``)."""
    from ouroboros.contracts.skill_manifest import canonical_skill_name
    from ouroboros.workspace_executor_paths import ManagedPathRefused

    domain = current()
    if domain is None:
        from ouroboros.skill_loader import skill_state_dir as service_state_dir

        return service_state_dir(drive_root, name)
    if not domain.skill_state_root:
        raise ManagedPathRefused("This deployment's execution domain names no skill_state_root, where a skill's own "
                                 "processes keep their state; nothing was started, and nothing runs on the service "
                                 "instead")
    path = pathlib.Path(domain.skill_state_root) / canonical_skill_name(name)
    _work_command(domain, "mkdir", "-p", "--", str(path))
    return path


@contextlib.contextmanager
def build_dir(parent: Any, prefix: str) -> Iterator[str]:
    """A private folder for one build's output inside ``parent`` (a ``skill_state_dir``), removed after: a temporary
    folder unmanaged; managed, made and removed in the work domain as its user, never on the service."""
    domain = current()
    if domain is None:
        with tempfile.TemporaryDirectory(prefix=prefix, dir=parent) as directory:
            yield directory
        return
    path = f"{parent}/{prefix}{uuid.uuid4().hex}"
    _work_command(domain, "mkdir", "--", path)
    try:
        yield path
    finally:
        try:
            _work_command(domain, "rm", "-rf", "--", path)
        except Exception:  # the build's own answer stands; the folder stays in the work domain's state
            log.warning("Build folder %s was not removed from the work domain", path, exc_info=True)


def _work_command(domain: ExecutionDomain, *argv: str) -> None:
    """One short command in the work container as its own user (a folder made or removed there). Its name resolves
    inside the work domain, so it is the work's code: ``network: none`` is checked first (``_network_none``), whose
    refusal alone proves nothing started. A command the CLI did not confirm may have run there: it raises saying so,
    never marked as not started."""
    if domain.executor.network == "none":
        _network_none(domain.executor)
    try:
        result = _docker("exec", domain.executor.container_name, *argv)
    except Exception as exc:
        result = subprocess.CompletedProcess(argv, -1, "", f"{type(exc).__name__}: {exc}")
    if result.returncode != 0:
        raise RuntimeError(f"The work domain did not confirm {argv[0]} for {argv[-1]} "
                           f"({(result.stderr or '').strip()[:300] or result.returncode}); it may have run there, "
                           "and nothing after it was started")


# --- what crosses into the domain ----------------------------------------------------------------------

def _request_paths(request: Dict[str, Any]) -> List[str]:
    scope = request.get("scope") if isinstance(request.get("scope"), dict) else {}
    execution = request.get("execution") if isinstance(request.get("execution"), dict) else {}
    roots = execution.get("auxiliaryWritableRoots") if isinstance(execution.get("auxiliaryWritableRoots"), list) else []
    return [str(path) for path in (scope.get("root"), execution.get("workspaceRoot"), *roots) if path]


def request_paths_refusal(request: Dict[str, Any]) -> str:
    """Why a native run's request names a path the domain does not see at the same address; "" when none.

    The work engine and its sessions read the request, the instructions and every host reference in the domain:
    a path is usable there only when the domain's mapping is identity for it."""
    domain = current()
    if domain is None:
        return ""
    unseen = []
    for path in _request_paths(request):
        try:  # the literal path is what the engine reads: its address there must be that same spelling
            if map_host_path(domain.executor, pathlib.Path(path)) != posixpath.normpath(path):
                unseen.append(path)
        except ValueError:
            unseen.append(path)
    if not unseen:
        return ""
    return (f"This deployment's work engine would not see {', '.join(sorted(set(unseen)))} at the same address: its "
            "execution domain maps only its work paths, so the run was not started. Choose a folder inside them.")


# --- Panic ---------------------------------------------------------------------------------------------

def _docker(*args: str) -> subprocess.CompletedProcess:
    from ouroboros.platform_layer import bootstrap_process_path

    bootstrap_process_path()
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=_DOCKER_TIMEOUT_SEC)


def measure_containers() -> Dict[str, str]:
    """Measure each dedicated container's identity now and keep it for this server generation's Panic. Kept in
    memory only: an identity another generation measured could stand for a container since recreated."""
    domain = current()
    measured: Dict[str, str] = {}
    for name in domain.dedicated_containers if domain is not None else ():
        try:
            answer = _docker("inspect", "-f", "{{.Id}}", name)
        except (OSError, subprocess.SubprocessError):
            continue
        identity = answer.stdout.strip()
        if answer.returncode == 0 and identity and not any(ch.isspace() for ch in identity):
            measured[name] = identity
    with _LOCK:
        _MEASURED.update(measured)
    return measured


def measure_generation_identity() -> bool:
    """A server start measures its dedicated containers off the startup path; ``False``: none to measure."""
    if not panic_scope():
        return False
    threading.Thread(target=measure_containers, name="execution-domain-identity", daemon=True).start()
    return True


def panic_scope() -> bool:
    """Whether Panic has a domain of its own to stop: a usable policy naming dedicated containers."""
    try:
        domain = current()
    except Exception:
        return False
    return bool(domain is not None and domain.dedicated_containers)


def _recorded() -> Dict[str, str]:
    domain = current()
    names = set(domain.dedicated_containers) if domain is not None else set()
    with _LOCK:
        return {name: identity for name, identity in _MEASURED.items() if name in names}


def panic_request() -> Dict[str, Any]:
    """Panic's request for this domain: one ``docker kill`` of the containers measured as its own (measured now
    when nothing was). The caller never waits for it; what each container settled into is ``panic_settle``'s."""
    targets = _recorded() or measure_containers()
    if not targets:
        return {"requested": False, "error": "no measured identity of this deployment's dedicated containers"}
    answer = _docker("kill", *targets.values())
    return {"requested": True, "containers": {name: identity[:12] for name, identity in targets.items()},
            "returncode": answer.returncode}


def panic_settle(data_dir: Any) -> Dict[str, Any]:
    """Whether each measured container stopped: ``stopped`` only on Docker's answer that it is not running or no
    longer exists; anything else, a name now naming another running container included, stays unconfirmed and is
    disclosed for the next start."""
    targets = _recorded()
    stopped, unconfirmed = [], {}
    for name, identity in targets.items():
        try:
            answer = _docker("inspect", "-f", "{{.State.Running}}", identity)
            gone = answer.returncode != 0 and "no such" in answer.stderr.lower()
            now = _docker("inspect", "-f", "{{.Id}} {{.State.Running}}", name)
            parts = now.stdout.split()
            if now.returncode == 0 and len(parts) == 2 and parts[0] != identity and parts[1] == "true":
                unconfirmed[name] = f"recreated as {parts[0][:12]} after measurement and still running"
            elif gone or (answer.returncode == 0 and answer.stdout.strip() == "false"):
                stopped.append(name)
            else:
                unconfirmed[name] = answer.stdout.strip() or answer.stderr.strip()[:200] or "unknown"
        except (OSError, subprocess.SubprocessError) as exc:
            unconfirmed[name] = type(exc).__name__
    domain = current()
    unconfirmed.update({name: "never measured" for name in (domain.dedicated_containers if domain else ())
                        if name not in targets})
    if unconfirmed:
        from ouroboros.utils import append_jsonl, utc_now_iso

        append_jsonl(pathlib.Path(data_dir) / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "process_stop_unconfirmed", "purpose": "execution_domain",
            "unconfirmed": unconfirmed})
    return {"stopped": stopped, "unconfirmed": unconfirmed}
