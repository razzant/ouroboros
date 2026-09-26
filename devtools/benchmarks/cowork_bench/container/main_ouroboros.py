"""Cowork Bench engine entrypoint that runs full Ouroboros inside the task container.

The benchmark's own runner (``run_parallel.sh`` / ``scripts/run_containerized.sh``) executes
this file with ITS interpreter (``/opt/venv``), so nothing here imports ``ouroboros``: the
file only orchestrates. It prepares the workspace exactly like the reference engines,
starts a persistent MCP gateway plus an ordinary Ouroboros server, submits the task as an
external-workspace task and translates the terminal result into the benchmark's contract
(``traj_log.json`` + ``workspace/`` + a ``Status:`` line the runner greps).

Engine contract (README "Подключение своего раннера"): ``--task_dir``, ``--max_steps``,
``--phase agent|eval|all``. ``--phase eval`` runs only the benchmark's evaluator, at most once
per task dump (``eval_attempt.py``, copied beside this file), plus the run's opt-in
audit-only residual-state diagnostic.
``--phase inventory`` checks MCP startup and tool discovery without model calls.
``--phase stateful`` proves PPTX and browser state survives separate HTTP client sessions.
Both preserve complete probe logs under ``--probe-output`` before any paid run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Mapping

ENGINE = "ouroboros"
CONFIG_ENV = "OUROBOROS_COWORK_CONFIG"
DEFAULT_CONFIG_PATH = "/workspace/configs/ouroboros_bench.json"
DEFAULT_SECRET_PATH = "/workspace/configs/ouroboros_bench.secret.json"
OUROBOROS_SRC = "/opt/ouroboros-src"
OUROBOROS_PYTHON = "/opt/ouroboros-venv/bin/python"
OUROBOROS_DATA = "/opt/ouroboros-data"
OUROBOROS_RUNTIME = "/opt/ouroboros-runtime"
MCP_PROXY_BIN = "/opt/mcp-proxy-venv/bin/mcp-proxy"
MCP_PROXY_PYTHON = "/opt/mcp-proxy-venv/bin/python"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_MAX_TURNS = "max_turns_reached"
# Seconds kept between the Ouroboros deadline and the runner's hard `timeout` kill, so the
# agent is asked to wrap up while this process can still persist its artefacts.
DEADLINE_SAFETY_SEC = 105
_LIBPQ_BRIDGE = {
    "PG_HOST": "PGHOST",
    "PG_PORT": "PGPORT",
    "PG_DATABASE": "PGDATABASE",
    "PG_USER": "PGUSER",
    "PG_PASSWORD": "PGPASSWORD",
}
_AUDIT_LOGS = ("tools.jsonl", "events.jsonl", "progress.jsonl", "chat.jsonl", "supervisor.jsonl")
# Public API fields copied without renaming or inventing amounts/finality. This
# entrypoint cannot import the runtime from the benchmark interpreter; a test
# pins this consumer mirror to cost_projection's canonical names/openness set.
COST_RESULT_FIELDS = (
    "accounted_upper_bound_usd", "accounted_upper_bound_usd_with_children", "cost_known",
    "cost_accounting_status", "cost_accounting_error", "cost_final", "cost_with_children_partial",
    "unknown_unmetered", "non_final_rows", "reserved_usd", "unresolved_upper_bound_usd",
    "ledger_integrity_degraded", "cost_presentation",
)


class EngineFailure(RuntimeError):
    """An adapter-stage failure with a typed infrastructure reason."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class WallClockInterrupt(BaseException):
    """The runner's `timeout` sent SIGTERM; artefacts must still be persisted."""


# --------------------------------------------------------------------------- pure helpers


def resolve_placeholders(value: Any, *, local_servers: str, workspace: str, task_dir: str) -> Any:
    if not isinstance(value, str):
        return value
    return (
        value.replace("${local_servers_paths}", local_servers)
        .replace("${agent_workspace}", workspace)
        .replace("${task_dir}", task_dir)
    )


def render_mcp_servers(
    needed: list[str],
    yaml_configs: list[Mapping[str, Any]],
    *,
    local_servers: str,
    workspace: str,
    task_dir: str,
    environ: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Project the benchmark's MCP yaml entries onto stdio launch specs.

    Mirrors ``utils/mcp/tool_servers.py`` of the benchmark: servers are matched by the yaml
    ``name`` field, placeholders are substituted, and the libpq variables the runner injects
    (``PGHOST``/``PGPORT``/...) override the yaml's ``PG_*`` defaults so every server reaches
    the per-task Postgres. A needed server without a yaml entry is a refusal, not a warning:
    the reference engines would silently run the task with a missing tool family.
    """
    by_name: dict[str, Mapping[str, Any]] = {}
    for cfg in yaml_configs:
        if isinstance(cfg, Mapping) and cfg.get("name"):
            by_name.setdefault(str(cfg["name"]), cfg)
    missing = [name for name in needed if name not in by_name]
    if missing:
        raise EngineFailure("mcp_config_missing", f"no MCP yaml config for: {', '.join(missing)}")

    def resolve(item: Any) -> Any:
        return resolve_placeholders(item, local_servers=local_servers, workspace=workspace, task_dir=task_dir)

    rendered: dict[str, dict[str, Any]] = {}
    for name in needed:
        params = by_name[name].get("params") or {}
        command = str(resolve(params.get("command") or ""))
        if not command:
            raise EngineFailure("mcp_config_invalid", f"MCP yaml config for {name!r} has no command")
        env = {str(k): str(resolve(v)) for k, v in (params.get("env") or {}).items()}
        for target, source in _LIBPQ_BRIDGE.items():
            if environ.get(source):
                env[target] = str(environ[source])
        args = [str(resolve(arg)) for arg in (params.get("args") or [])]
        # This entrypoint runs in disposable benchmark containers. Their default root
        # user cannot launch Chromium's inner sandbox; container resource/isolation
        # bounds remain in force. Leave upstream YAML and non-browser servers intact.
        if name == "playwright_with_chunk" and getattr(os, "geteuid", lambda: -1)() == 0:
            if "--no-sandbox" not in args:
                args.append("--no-sandbox")
        rendered[name] = {"command": command, "args": args, "env": env}
    return rendered


def proxy_config(servers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {"mcpServers": {name: dict(spec) for name, spec in servers.items()}}


def ouroboros_mcp_servers(names: list[str], *, proxy_port: int) -> list[dict[str, Any]]:
    """One streamable-HTTP row per benchmark server, so tool names keep their server prefix."""
    return [
        {
            "id": name,
            "name": name,
            "enabled": True,
            "transport": "streamable_http",
            "url": f"http://127.0.0.1:{int(proxy_port)}/servers/{urllib.parse.quote(name)}/mcp",
        }
        for name in names
    ]


def build_settings(
    bench_config: Mapping[str, Any], secret: Mapping[str, Any], mcp_names: list[str]
) -> dict[str, Any]:
    settings = dict(bench_config.get("settings") or {})
    settings.update(
        {
            "MCP_ENABLED": True,
            "MCP_SERVERS": ouroboros_mcp_servers(mcp_names, proxy_port=int(bench_config["proxy_port"])),
            "MCP_TOOL_TIMEOUT_SEC": int(bench_config.get("mcp_tool_timeout_sec") or 300),
        }
    )
    for key, value in (secret.get("settings") or {}).items():
        settings[str(key)] = str(value)
    return settings


def build_task_body(
    bench_config: Mapping[str, Any], *, description: str, workspace: str, timeout_sec: int
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "description": description,
        "workspace_root": workspace,
        "workspace_mode": "external",
        "memory_mode": "empty",
        "actor_id": "cowork-bench",
        "source": "cowork-bench",
        "metadata": {"source": "cowork-bench", "delegation_role": "root"},
        "disabled_tools": [str(name) for name in (bench_config.get("disabled_tools") or [])],
    }
    if timeout_sec > 0:
        body["timeout_sec"] = int(timeout_sec)
    return body


def task_description(system_prompt: str | None, task_text: str) -> str:
    """The benchmark's own per-task system prompt followed by its task statement, nothing else."""
    parts = [part.strip() for part in (system_prompt or "", task_text or "") if part and part.strip()]
    return "\n\n".join(parts)


def classify_outcome(result: Mapping[str, Any], truncation_codes: list[str]) -> dict[str, Any]:
    """Translate a terminal Ouroboros task result into the benchmark's status vocabulary.

    ``completed`` means the agent ended its work voluntarily, which is exactly the reference
    engines' ``success`` ("the model ended its turn"); the benchmark's evaluator, not the
    acceptance panel, then judges the side effects. A runtime rail (round cap, budget, local
    deadline) is the benchmark's ``max_turns_reached``. Provider/transport death is ``failed``
    AND flagged ``infra_failed`` so the launcher may re-run it without touching genuine misses.
    """
    status = str(result.get("status") or "")
    reason_code = str(result.get("reason_code") or "")
    axes = result.get("outcome_axes") if isinstance(result.get("outcome_axes"), dict) else {}
    execution = axes.get("execution") if isinstance(axes.get("execution"), dict) else {}
    infra_failed = (
        reason_code in {"llm_api_error", "provider_unavailable"}
        or str(execution.get("status") or "") == "infra_failed"
        or str(execution.get("reason_code") or "") in {"llm_api_error", "provider_unavailable"}
    )
    truncated = reason_code in set(truncation_codes)
    if infra_failed:
        bench_status = STATUS_FAILED
    elif truncated:
        bench_status = STATUS_MAX_TURNS
    elif status == "completed":
        bench_status = STATUS_SUCCESS
    else:
        bench_status = STATUS_FAILED
    return {
        "bench_status": bench_status,
        "ouroboros_status": status,
        "reason_code": reason_code,
        "infra_failed": infra_failed,
        "truncated": truncated,
    }


def scrub(text: str, secrets: list[str]) -> str:
    for value in secrets:
        if value and len(value) >= 8:
            text = text.replace(value, "***REDACTED***")
    return text


# --------------------------------------------------------------------------- process helpers


def _http_json(method: str, url: str, body: Any = None, timeout: int = 30) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def _spawn(argv: list[str], *, cwd: str, env: Mapping[str, str], log_path: pathlib.Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    try:
        return subprocess.Popen(
            argv, cwd=cwd, env=dict(env), stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
        )
    finally:
        handle.close()


def _terminate(proc: subprocess.Popen | None, grace_sec: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=grace_sec)
            return
        except subprocess.TimeoutExpired:
            continue


def _wait_http(url: str, proc: subprocess.Popen, *, timeout_sec: float, accept) -> None:
    deadline = time.monotonic() + timeout_sec
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise EngineFailure("process_exited", f"process exited with {proc.returncode} before {url} answered")
        try:
            if accept(_http_json("GET", url, timeout=5)):
                return
            last = "endpoint answered but was not ready"
        except Exception as exc:  # noqa: BLE001 - readiness probe, every failure is "not yet"
            last = repr(exc)
        time.sleep(1.0)
    raise EngineFailure("readiness_timeout", f"{url} not ready within {timeout_sec:.0f}s: {last}")


# --------------------------------------------------------------------------- benchmark glue


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_bench_config() -> dict[str, Any]:
    path = os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH
    config = load_json(path)
    for key in ("model", "settings", "proxy_port", "server_port", "task_timeout_sec", "truncation_reason_codes"):
        if key not in config:
            raise ValueError(f"{path} is missing required key {key!r}")
    return config


def build_task_config(task_dir: str, model: str, max_steps: int):
    from utils.data_structures.task_config import TaskConfig
    from utils.general.helper import read_json

    eval_config = read_json("scripts/eval_config_strands.json")
    global_task_config = {
        "dump_path": eval_config.get("dump_path", "./dumps/"),
        "max_steps_under_single_turn_mode": max_steps,
    }
    return TaskConfig.build(
        task_dir,
        agent_short_name=f"{ENGINE}/{model}",
        global_task_config=global_task_config,
        single_turn_mode=True,
        cn_mode=False,
    )


def setup_workspace(task_config) -> str:
    from utils.general.helper import copy_folder_contents

    workspace = os.path.abspath(task_config.agent_workspace)
    if os.path.isdir(workspace):
        shutil.rmtree(workspace, ignore_errors=True)
    os.makedirs(workspace, exist_ok=True)
    init = task_config.initialization
    if init and init.workspace and os.path.exists(str(init.workspace)):
        asyncio.run(copy_folder_contents(str(init.workspace), workspace))
    for server, folder in (
        ("arxiv_local", "arxiv_local_storage"),
        ("memory", "memory"),
        ("playwright_with_chunk", ".playwright_output"),
    ):
        if server in (task_config.needed_mcp_servers or []):
            os.makedirs(os.path.join(workspace, folder), exist_ok=True)
    return workspace


def run_preprocess(task_config, log_path: pathlib.Path) -> None:
    init = task_config.initialization
    if not (init and init.process_command):
        return
    launch = " ".join((task_config.launch_time or "").split()[:2])
    command = f'{init.process_command} --agent_workspace {task_config.agent_workspace} --launch_time "{launch}"'
    with log_path.open("ab") as handle:
        completed = subprocess.run(command, shell=True, stdout=handle, stderr=subprocess.STDOUT)  # noqa: S602
    if completed.returncode != 0:
        raise EngineFailure("preprocess_failed", f"preprocess exited with {completed.returncode}")


def load_yaml_configs(config_dir: str = "configs/mcp_servers") -> list[Mapping[str, Any]]:
    import yaml

    configs: list[Mapping[str, Any]] = []
    for path in sorted(pathlib.Path(config_dir).glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if isinstance(loaded, dict):
            loaded.setdefault("name", path.stem)
            configs.append(loaded)
    return configs


def collect_audit_artifacts(task_root: pathlib.Path, task_id: str, secrets: list[str]) -> None:
    """Copy the scrubbed Ouroboros logs next to the benchmark artefacts.

    The data root itself stays outside ``dumps/`` because it holds ``settings.json`` with the
    provider credential; only value-scrubbed text leaves the container. Polling
    checkpoints publish each file atomically, so an abrupt container removal
    leaves the previous snapshot intact even if final export never runs.
    """
    target = task_root / "ouroboros"
    target.mkdir(parents=True, exist_ok=True)
    sources = [pathlib.Path(OUROBOROS_DATA) / "logs" / name for name in _AUDIT_LOGS]
    if task_id:
        sources.append(pathlib.Path(OUROBOROS_DATA) / "task_results" / f"{task_id}.json")
    for source in sources:
        try:
            text = source.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        staged = target / f".{source.name}.tmp"
        staged.write_text(scrub(text, secrets), encoding="utf-8")
        staged.replace(target / source.name)


def write_bench_logs(task_config, status: str, started: datetime, summary: Mapping[str, Any], final_answer: str) -> None:
    log_path = pathlib.Path(task_config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ended = datetime.now().isoformat()
    record = {
        "config": task_config.to_dict(),
        "status": status,
        "start_time": started.isoformat(),
        "end_time": ended,
    }
    log_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    traj = {
        "status": status,
        "start_time": started.isoformat(),
        "end_time": ended,
        "engine": ENGINE,
        "final_answer": final_answer,
        "summary": dict(summary),
        "trajectory": "ouroboros/tools.jsonl + ouroboros/events.jsonl (scrubbed runtime logs)",
    }
    (log_path.parent / "traj.json").write_text(json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
    (log_path.parent / "ouroboros_summary.json").write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --------------------------------------------------------------------------- agent phase


def run_agent_phase(task_dir: str, max_steps: int) -> str:
    entry_started = time.monotonic()
    started = datetime.now()
    bench_config = load_bench_config()
    secret = load_json(os.environ.get(CONFIG_ENV + "_SECRET") or DEFAULT_SECRET_PATH)
    secrets = [str(value) for value in (secret.get("settings") or {}).values()]
    model = str(bench_config["model"])
    env_model = os.environ.get("LLM_MODEL") or os.environ.get("MODEL_NAME") or ""
    if env_model and env_model != model:
        raise ValueError(f"runner model {env_model!r} differs from the run config model {model!r}")

    task_config = build_task_config(task_dir, model, max_steps)
    task_root = pathlib.Path(task_config.log_file).parent
    task_root.mkdir(parents=True, exist_ok=True)
    print(f"====== {task_dir} | {ENGINE}/{model} | steps={max_steps} ======", flush=True)
    print(f"workspace : {task_config.agent_workspace}", flush=True)
    print(f"log       : {task_config.log_file}", flush=True)

    summary: dict[str, Any] = {
        "engine": ENGINE,
        "model": model,
        "task": task_dir,
        "max_steps": max_steps,
        "bench_status": STATUS_FAILED,
        "adapter_stage": "init",
        "infra_failed": False,
    }
    status = STATUS_FAILED
    final_answer = ""
    task_id = ""
    task_submission_started = False
    result: dict[str, Any] = {}
    proxy: subprocess.Popen | None = None
    server: subprocess.Popen | None = None

    def _on_sigterm(_signum, _frame):
        raise WallClockInterrupt()

    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        summary["adapter_stage"] = "workspace"
        workspace = setup_workspace(task_config)
        summary["adapter_stage"] = "preprocess"
        run_preprocess(task_config, task_root / "preprocess.log")

        summary["adapter_stage"] = "mcp"
        needed = list(task_config.needed_mcp_servers or [])
        servers = render_mcp_servers(
            needed,
            load_yaml_configs(),
            local_servers=os.environ.get("LOCAL_SERVERS_PATH", os.path.abspath("./local_servers")),
            workspace=workspace,
            task_dir=os.path.abspath(os.path.join("tasks/finalpool", task_config.task_dir)),
            environ=os.environ,
        )
        runtime_dir = pathlib.Path(OUROBOROS_RUNTIME)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        proxy_config_path = runtime_dir / "mcp_proxy.json"
        proxy_config_path.write_text(json.dumps(proxy_config(servers)), encoding="utf-8")
        proxy_port = int(bench_config["proxy_port"])
        proxy = _spawn(
            [MCP_PROXY_BIN, "--host", "127.0.0.1", "--port", str(proxy_port), "--stateless",
             "--pass-environment", "--named-server-config", str(proxy_config_path)],
            cwd=workspace, env=os.environ, log_path=task_root / "mcp_proxy.log",
        )
        try:
            _wait_http(f"http://127.0.0.1:{proxy_port}/status", proxy, timeout_sec=300, accept=lambda _data: True)
        except EngineFailure as exc:
            raise EngineFailure("mcp_start_failed", str(exc)) from exc
        summary["mcp_servers"] = needed

        summary["adapter_stage"] = "server"
        data_dir = pathlib.Path(OUROBOROS_DATA)
        for sub in ("logs", "state"):
            (data_dir / sub).mkdir(parents=True, exist_ok=True)
        # The existing runtime marker keeps the full task history in its active
        # logs, so rotation cannot discard evidence when this container is removed.
        (data_dir / ".ouroboros_isolated_benchmark").touch()
        settings_path = data_dir / "settings.json"
        settings = build_settings(bench_config, secret, needed)
        settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
        os.chmod(settings_path, 0o600)
        applied = {key: value for key, value in settings.items() if key not in (secret.get("settings") or {})}
        (task_root / "applied_settings.json").write_text(
            scrub(json.dumps(applied, ensure_ascii=False, indent=2), secrets), encoding="utf-8"
        )
        server_port = int(bench_config["server_port"])
        server_env = {
            key: value for key, value in os.environ.items() if not key.startswith(("OUROBOROS_", "LLM_", "MODEL_"))
        }
        server_env.update(
            {
                "OUROBOROS_APP_ROOT": "/opt/ouroboros-app",
                "OUROBOROS_REPO_DIR": OUROBOROS_SRC,
                "OUROBOROS_DATA_DIR": OUROBOROS_DATA,
                "OUROBOROS_SETTINGS_PATH": str(settings_path),
                "OUROBOROS_PID_FILE": str(runtime_dir / "ouroboros.pid"),
                "OUROBOROS_PORT_FILE": str(data_dir / "state" / "server_port"),
                "OUROBOROS_SERVER_HOST": "127.0.0.1",
                "OUROBOROS_SERVER_PORT": str(server_port),
                "OUROBOROS_WORKER_START_METHOD": "spawn",
                "PYTHONUNBUFFERED": "1",
            }
        )
        server = _spawn(
            [OUROBOROS_PYTHON, "server.py", "--host", "127.0.0.1", "--port", str(server_port)],
            cwd=OUROBOROS_SRC, env=server_env, log_path=task_root / "ouroboros_server.log",
        )
        base_url = f"http://127.0.0.1:{server_port}"
        try:
            _wait_http(
                f"{base_url}/api/state", server, timeout_sec=300,
                accept=lambda data: bool(data.get("supervisor_ready")),
            )
        except EngineFailure as exc:
            raise EngineFailure("server_start_failed", str(exc)) from exc

        summary["adapter_stage"] = "task"
        overhead = time.monotonic() - entry_started
        timeout_sec = int(float(bench_config["task_timeout_sec"]) - overhead - DEADLINE_SAFETY_SEC)
        sp = task_config.system_prompts
        body = build_task_body(
            bench_config,
            description=task_description(getattr(sp, "agent", None), task_config.task_str),
            workspace=workspace,
            timeout_sec=max(timeout_sec, 60),
        )
        summary["adapter_overhead_sec"] = round(overhead, 1)
        summary["ouroboros_timeout_sec"] = body.get("timeout_sec")
        task_submission_started = True
        created = _http_json("POST", f"{base_url}/api/tasks", body)
        task_id = str(created.get("task_id") or "")
        if not task_id:
            raise EngineFailure("task_create_failed", f"task creation returned no task_id: {created!r}")
        summary["ouroboros_task_id"] = task_id

        outer_deadline = entry_started + float(bench_config["task_timeout_sec"])
        cost_deadline: float | None = None
        terminal = False
        while True:
            now = time.monotonic()
            if terminal and cost_deadline is not None and now >= cost_deadline:
                summary["cost_finality_wait_exhausted"] = True
                break
            if now >= outer_deadline:
                raise WallClockInterrupt()
            # dumps/ is host-bound; preserve available usage/tool evidence before
            # a host stop can remove this container without running our finally.
            try:
                collect_audit_artifacts(task_root, task_id, secrets)
            except OSError as exc:
                detail = scrub(f"{type(exc).__name__}: {exc}", secrets)
                print(f"[{ENGINE}] audit checkpoint unavailable; continuing task: {detail}",
                      file=sys.stderr, flush=True)
            try:
                result = _http_json("GET", f"{base_url}/api/tasks/{urllib.parse.quote(task_id)}",
                                    timeout=min(30, outer_deadline - now))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if server.poll() is not None:
                    raise EngineFailure("server_died", f"Ouroboros server exited with {server.returncode}") from exc
                time.sleep(min(2, max(0, outer_deadline - time.monotonic())))
                continue
            result_status = str(result.get("status") or "").lower()
            bundle = result.get("artifact_bundle")
            bundle = bundle if isinstance(bundle, dict) else {}
            artifact_status = str(bundle.get("status") or result.get("artifact_status") or "").lower()
            terminal = (result_status in {"completed", "failed", "cancelled", "rejected_duplicate"}
                        and artifact_status not in {"pending", "finalizing"})
            if terminal:
                if cost_deadline is not None and (
                    result.get("cost_final") or result.get("cost_with_children_partial") is False
                ):
                    break
                # Match cli._await_cost_finality: only explicit partial accounting
                # on completed/degraded outcomes waits, at most 60s and within
                # this invocation's existing outer deadline. Unknown stays unknown.
                cost_pending = result_status in {"completed", "degraded"} and (
                    result.get("cost_final") is False or result.get("cost_with_children_partial") is True
                )
                if not cost_pending:
                    break
                if cost_deadline is None:
                    cost_deadline = min(time.monotonic() + 60, outer_deadline)
            sleep_until = min(outer_deadline, cost_deadline) if terminal and cost_deadline is not None else outer_deadline
            remaining = max(0, sleep_until - time.monotonic())
            time.sleep(min(2, remaining))

        outcome = classify_outcome(result, list(bench_config["truncation_reason_codes"]))
        status = outcome["bench_status"]
        summary.update(outcome)
        summary.update(
            {
                "adapter_stage": "finished",
                "prompt_tokens": result.get("prompt_tokens"),
                "completion_tokens": result.get("completion_tokens"),
                "total_rounds": result.get("total_rounds"),
                "degraded": bool(result.get("degraded")),
            }
        )
    except WallClockInterrupt:
        # Before POST admission there is definitely no solve attempt. A missing
        # task id/usage after POST began is uncertainty, not proof of zero work.
        summary.update({"reason_code": "wall_clock_timeout", "truncated": True,
                        "infra_failed": not task_submission_started})
        status = STATUS_FAILED
    except EngineFailure as exc:
        summary.update({"reason_code": exc.reason, "infra_failed": True, "error": scrub(str(exc), secrets)[:2000]})
        status = STATUS_FAILED
    except Exception as exc:  # noqa: BLE001 - every adapter crash must still leave artefacts
        summary.update(
            {"reason_code": "adapter_crashed", "infra_failed": True,
             "error": scrub(f"{type(exc).__name__}: {exc}", secrets)[:2000]}
        )
        status = STATUS_FAILED
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        summary["bench_status"] = status
        summary["elapsed_sec"] = round(time.monotonic() - entry_started, 1)
        summary["task_submission_started"] = task_submission_started
        final_answer = str(result.get("final_answer") or result.get("result") or "")
        if result.get("status"):
            summary["ouroboros_status"] = result["status"]
        summary.update({key: result[key] for key in COST_RESULT_FIELDS if key in result})
        summary.update({key: result[key] for key in (
            "artifact_status", "artifact_bundle", "prompt_tokens", "completion_tokens", "total_rounds"
        ) if key in result})
        observed_tokens = [result.get(key) for key in ("prompt_tokens", "completion_tokens")]
        summary["model_activity_observed"] = (
            False if not task_submission_started else
            True if any(isinstance(n, (int, float)) and n > 0 for n in observed_tokens) else None
        )
        try:
            collect_audit_artifacts(task_root, task_id, secrets)
        finally:
            write_bench_logs(task_config, status, started, summary, scrub(final_answer, secrets))
            _terminate(server)
            _terminate(proxy)
    print(f"[{ENGINE}] summary: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    print(f"\n====== Status: {status} ======", flush=True)
    return status


# --------------------------------------------------------------------------- inventory phase

# Executed by the MCP gateway's interpreter (it carries a current MCP client SDK); the
# benchmark interpreter running this file makes no promise about its own `mcp` version.
_INVENTORY_HELPER = """
import asyncio, json, os, sys
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

spec = json.loads(sys.argv[1])

async def main():
    params = StdioServerParameters(
        command=spec["command"], args=spec["args"], env={**os.environ, **spec["env"]}, cwd=spec["cwd"]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            print(json.dumps({"tools": [tool.name for tool in listed.tools]}))

asyncio.run(asyncio.wait_for(main(), timeout=float(spec["timeout"])))
"""


def capture_probe(argv: list[str], prefix: pathlib.Path, *, timeout: int) -> dict[str, Any]:
    """Keep complete diagnostic streams, including startup errors preceding an MCP traceback."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = prefix.with_suffix(".stdout.log")
    stderr_path = prefix.with_suffix(".stderr.log")
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            proc = subprocess.run(argv, stdout=stdout, stderr=stderr, text=True, timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            stderr.write(f"\nProbe exceeded {timeout}s.\n")
            code = 124
    return {"returncode": code, "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}


def run_inventory_phase(task_dir: str, output_dir: pathlib.Path) -> int:
    """Start every benchmark MCP server once and list its tools; no model call is made.

    A vendored server that no longer starts (dependency drift in a rebuilt image) turns every
    task needing it into a structural zero that looks like a capability miss, so an image is
    checked here before any paid run. ``--task_dir`` narrows the check to one task's servers.
    """
    configs = load_yaml_configs()
    pool = pathlib.Path("tasks/finalpool")
    if task_dir:
        needed = list(load_json(str(pool / task_dir / "task_config.json")).get("needed_mcp_servers") or [])
        sample_task = pool / task_dir
    else:
        needed = [str(cfg["name"]) for cfg in configs]
        sample_task = next((p.parent for p in sorted(pool.glob("*/email_config.json"))), pool)
    workspace = output_dir / "workspace"
    for folder in ("", "arxiv_local_storage", "memory", ".playwright_output"):
        (workspace / folder).mkdir(parents=True, exist_ok=True)
    servers = render_mcp_servers(
        needed, configs,
        local_servers=os.environ.get("LOCAL_SERVERS_PATH", os.path.abspath("./local_servers")),
        workspace=str(workspace), task_dir=str(sample_task.resolve()), environ=os.environ,
    )
    failures = 0
    for name, spec in servers.items():
        payload = json.dumps({**spec, "cwd": str(workspace), "timeout": 180})
        row: dict[str, Any] = {"server": name, "ok": False, **capture_probe(
            [MCP_PROXY_PYTHON, "-c", _INVENTORY_HELPER, payload], output_dir / name, timeout=240,
        )}
        try:
            stdout = pathlib.Path(row["stdout_path"]).read_text(encoding="utf-8")
            row["tool_count"] = len(json.loads(stdout.strip().splitlines()[-1])["tools"])
            row["ok"] = row["returncode"] == 0 and row["tool_count"] > 0
        except (IndexError, KeyError, ValueError):
            row["error"] = "Invalid inventory output; see the complete stdout/stderr files."
        failures += 0 if row["ok"] else 1
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f"====== Inventory: {len(servers) - failures}/{len(servers)} servers ok ======", flush=True)
    return 1 if failures else 0


# Each call opens and closes its own HTTP transport and ClientSession, exactly the
# lifecycle that previously lost a stdio server's in-memory presentations/browser tabs.
_STATEFUL_HELPER = """
import asyncio, json, sys
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

base = sys.argv[1]
marker = "cowork-state-survives-http-session"

async def call(server, name, arguments):
    async with streamablehttp_client(f"{base}/servers/{server}/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            if result.isError:
                raise RuntimeError(str(result))
            text = "\\n".join(block.text for block in result.content if block.type == "text")
            print(json.dumps({"server": server, "tool": name, "result": text}), flush=True)
            return text

async def main():
    created = json.loads(await call("pptx", "create_presentation", {"id": marker}))
    assert created["presentation_id"] == marker and created["slide_count"] == 0, created
    added = json.loads(await call("pptx", "add_slide", {"presentation_id": marker, "title": marker}))
    assert added["slide_index"] == 0, added
    info = json.loads(await call("pptx", "get_presentation_info", {"presentation_id": marker}))
    assert info["slide_count"] == 1, info
    await call("playwright_with_chunk", "browser_navigate", {
        "url": "data:text/html,<title>Cowork probe</title><h1>" + marker + "</h1>"
    })
    snapshot = await call("playwright_with_chunk", "browser_snapshot", {})
    assert 'heading "' + marker + '"' in snapshot, snapshot
    print(json.dumps({"ok": True, "pptx_slide_count": 1, "browser_heading": marker,
                      "fresh_http_sessions": 5}), flush=True)

asyncio.run(asyncio.wait_for(main(), timeout=180))
"""


def run_stateful_phase(output_dir: pathlib.Path) -> int:
    """Exercise the actual persistent proxy with PPTX and browser state, without an LLM."""
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = output_dir / "workspace"
    (workspace / ".playwright_output").mkdir(parents=True, exist_ok=True)
    servers = render_mcp_servers(
        ["pptx", "playwright_with_chunk"], load_yaml_configs(),
        local_servers=os.environ.get("LOCAL_SERVERS_PATH", os.path.abspath("./local_servers")),
        workspace=str(workspace.resolve()), task_dir=str(pathlib.Path("tasks/finalpool").resolve()), environ=os.environ,
    )
    config_path = output_dir / "mcp_proxy.json"
    config_path.write_text(json.dumps(proxy_config(servers)), encoding="utf-8")
    proxy = _spawn(
        [MCP_PROXY_BIN, "--host", "127.0.0.1", "--port", "8096", "--stateless", "--pass-environment",
         "--named-server-config", str(config_path.resolve())],
        cwd=str(workspace.resolve()), env=os.environ, log_path=output_dir / "mcp_proxy.log",
    )
    try:
        _wait_http("http://127.0.0.1:8096/status", proxy, timeout_sec=300, accept=lambda _data: True)
        row = capture_probe(
            [MCP_PROXY_PYTHON, "-c", _STATEFUL_HELPER, "http://127.0.0.1:8096"],
            output_dir / "stateful", timeout=210,
        )
        row["ok"] = row["returncode"] == 0
        print(json.dumps(row), flush=True)
        return 0 if row["ok"] else 1
    finally:
        _terminate(proxy)


# --------------------------------------------------------------------------- eval phase


def _eval_attempt():
    """The stdlib helper the image copies beside this entrypoint; the repository package otherwise."""
    try:
        import eval_attempt
    except ImportError:
        from devtools.benchmarks.cowork_bench import eval_attempt
    return eval_attempt


def run_eval_phase(task_dir: str, max_steps: int) -> int:
    """One claimed official evaluation; the opt-in residual diagnostic follows its final lines."""
    attempts = _eval_attempt()
    # Host evidence for the diagnostic only; the official evaluator keeps its original environment.
    agent_state = os.environ.pop(attempts.AGENT_STATE_ENV, "") or "unknown"
    from utils.evaluation.evaluator import TaskEvaluator

    bench_config = load_bench_config()
    model = str(bench_config["model"])
    task_root = attempts.task_dump_root(f"{ENGINE}/{model}".replace("/", "_"), task_dir)
    print("\n====== Evaluating ======", flush=True)

    def prepare() -> str:
        # Built only inside a fresh claim: `TaskConfig.build` deletes an existing eval_res.json.
        log_file = build_task_config(task_dir, model, max_steps).log_file
        if os.path.abspath(log_file) != str(task_root / "traj_log.json"):
            raise attempts.NotRun("log_path_mismatch")
        return log_file

    bindings = attempts.official_bindings(task_root, task_dir, entry_path=__file__,
                                          config_path=os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH)
    record, ran_here = attempts.official_attempt(
        task_root, bindings, {"agent_container_state": agent_state}, prepare,
        lambda log_file: asyncio.run(TaskEvaluator.evaluate_from_log_file(log_file)),
    )
    eval_res = attempts.official_verdict(record)
    if eval_res is None:
        print(attempts.refusal_line(record), flush=True)
        return 1
    print(f"Pass:    {eval_res.get('pass', False)}", flush=True)
    print(f"Details: {eval_res.get('details', eval_res.get('failure', 'N/A'))}", flush=True)
    code = 0 if eval_res.get("pass", False) else 1
    if ran_here and bench_config.get(attempts.FLAG) is True and eval_res.get("pass", False) is None:
        attempts.run_residual_diagnostic(task_root, record["attempt_id"])
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", default="")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--phase", choices=["all", "agent", "eval", "inventory", "stateful"], default="all")
    parser.add_argument("--eval_config", default="")
    parser.add_argument("--model_name", default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--probe-output", type=pathlib.Path, default=pathlib.Path("/opt/ouroboros-runtime/probes"))
    args = parser.parse_args(argv)
    if args.phase == "inventory":
        return run_inventory_phase(args.task_dir, args.probe_output)
    if args.phase == "stateful":
        return run_stateful_phase(args.probe_output)
    if not args.task_dir:
        parser.error("--task_dir is required for the agent and eval phases")
    if args.phase in ("all", "agent"):
        run_agent_phase(args.task_dir, args.max_steps)
        if args.phase == "agent":
            return 0
    return run_eval_phase(args.task_dir, args.max_steps)


if __name__ == "__main__":
    raise SystemExit(main())
