"""CyberGym reconcile arm: adopt an interrupted run and deliver its results.

When the launcher dies after the gateway accepted tasks but before their rows
were delivered, ``--reconcile <run root>`` re-reads the manifest, attaches to
the still-running isolated server and workspace containers, and runs the
shared delivery path for every checkpointed attempt that has no
``result_index.jsonl`` row.  It never re-runs an agent, never starts new
infrastructure, and never rewrites an existing row; attempts whose gateway
task is still alive are reported ``left_running`` for a later pass, and the
report lands in ``extra.reconcile`` of the finalized manifest.

``_ReconcileMixin`` carries the executor-side adoption/delivery methods and is
assembled into ``CyberGymExecutor``; ``reconcile_main`` is the launcher entry.
This module is the launcher's second arm, split out of ``run_cybergym.py`` so
the submit-shaped entry point stays inside its size band.

Durability contract: a result row is appended (under the shared
``.result_index.lock``, after re-reading the recorded set) BEFORE the claim is
settled, and the adopted workspace container is released only after both are
durable — a crash anywhere earlier keeps the container so a later pass can
redeliver from the checkpoint.  Every pass appends its report to
``extra.reconcile_passes``; earlier passes are never overwritten.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import sys
import time
import urllib.parse
from collections.abc import Iterator, Mapping
from typing import Any

from devtools.benchmarks.cybergym.cybergym_adapter import (
    DEFAULT_LEVEL,
    BudgetLedger,
    CyberGymError,
    CyberGymIntegrationUnavailable,
    LedgerError,
    TaskSpec,
    _TERMINAL_GATEWAY_STATUSES,
    finalize_outcome_row,
    load_task_catalog,
    safe_task_id,
    safe_task_path,
    settle_finished_attempt,
    task_slug,
)
from devtools.benchmarks.cybergym.cybergym_result_index import (
    _append_result_pair,
    acquire_campaign_execution_lock,
)
from devtools.benchmarks.cybergym.cybergym_docker import (
    _GATEWAY_TASK_ID,
    _inside,
    _safe_abs,
    _write_json,
)
from devtools.benchmarks.cybergym.cybergym_sidecar import make_opaque_agent_id
from devtools.benchmarks.cybergym.cybergym_wire import (
    ExecutorFailure,
    _gateway_path,
    _redeliverable_terminal_frame,
    _response_status,
    _unwrap_http_json,
)


class _ReconcileMixin:
    """Adoption and redelivery methods mixed into the CyberGym executor."""

    def adopt_campaign(self) -> Mapping[str, Any]:
        """Attach to the still-running campaign resources of an interrupted run.

        Reconcile mode never starts a new server, network, or workspace: it
        re-derives the deterministic identities from ``sidecar_state.json``
        and Docker inspection, verifies ownership labels, and registers the
        exact immutable ids so the shared delivery path can run.  Any mismatch
        is a typed refusal, never a name-based guess.
        """
        if self.started:
            raise ExecutorFailure("adopt_campaign requires a fresh executor")
        self._verify_settings_snapshot()
        self._ensure_key()
        state_path = self.config.run_root / "sidecar_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExecutorFailure("campaign sidecar state is unavailable for adoption") from exc
        if not isinstance(state, Mapping):
            raise ExecutorFailure("campaign sidecar state is malformed")
        server_id = str(state.get("server_id") or "").strip()
        network_id = str(state.get("network_id") or "").strip()
        if not server_id or not _GATEWAY_TASK_ID.fullmatch(server_id):
            raise ExecutorFailure("campaign sidecar state has no usable server id")
        if not network_id:
            raise ExecutorFailure("campaign sidecar state has no usable network id")
        observed = self._inspect("container", self.server_name)
        actual_id = str(observed.get("Id") or "").strip()
        if actual_id != server_id:
            raise ExecutorFailure("adopted server container id mismatch")
        config = observed.get("Config")
        labels = config.get("Labels", {}) if isinstance(config, Mapping) else {}
        if not isinstance(labels, Mapping) or labels.get("com.ouroboros.campaign") != self.config.campaign_id:
            raise ExecutorFailure("adopted server ownership attestation failed")
        state_docker_host = str(state.get("docker_host") or "").strip()
        if state_docker_host and state_docker_host != self.host.value:
            raise ExecutorFailure("campaign sidecar state belongs to a different Docker host")
        status = str((observed.get("State") or {}).get("Status") or "").strip().lower() if isinstance(observed.get("State"), Mapping) else ""
        if status != "running":
            raise ExecutorFailure("adopted server container is not running")
        network = self._inspect_optional("network", network_id)
        if network is None:
            raise ExecutorFailure("campaign network is absent; cannot adopt")
        network_labels = network.get("Labels") if isinstance(network.get("Labels"), Mapping) else {}
        if str(network.get("Id") or "") != network_id or network_labels.get("com.ouroboros.campaign") != self.config.campaign_id:
            raise ExecutorFailure("adopted network ownership attestation failed")
        self.server_id = server_id
        self.network_id = network_id
        self._network_created = False
        self._server_observation = observed
        plan = self._network_plan("campaign")
        self.server_url = plan.server_url
        self.started = True
        self._adopted = True
        self._wait_server(plan)
        return {
            "status": "adopted",
            "ok": True,
            "server_id": server_id,
            "network_id": network_id,
            "server_url": self.server_url,
        }

    def _adopt_workspace_container(self, container_name: str) -> str:
        """Register a still-running workspace container by deterministic name.

        Returns the immutable container id.  A missing, stopped, replaced, or
        foreign-labeled container is a typed refusal: reconcile never adopts
        by name alone.
        """
        observed = self._inspect_optional("container", container_name)
        if observed is None:
            raise ExecutorFailure("workspace container is absent")
        actual_id = str(observed.get("Id") or "").strip()
        if not actual_id or not _GATEWAY_TASK_ID.fullmatch(actual_id):
            raise ExecutorFailure("workspace container id is unsafe")
        config = observed.get("Config")
        labels = config.get("Labels", {}) if isinstance(config, Mapping) else {}
        if (
            not isinstance(labels, Mapping)
            or labels.get("com.ouroboros.campaign") != self.config.campaign_id
            or labels.get("com.ouroboros.role") != "workspace"
        ):
            raise ExecutorFailure("workspace adoption ownership attestation failed")
        status = ""
        state = observed.get("State")
        if isinstance(state, Mapping):
            status = str(state.get("Status") or "").strip().lower()
        if status != "running":
            raise ExecutorFailure("workspace container is not running")
        networks = ((observed.get("NetworkSettings") or {}).get("Networks") or {})
        network = networks.get("cybergym-internal") if isinstance(networks, Mapping) else None
        if not isinstance(network, Mapping) or str(network.get("NetworkID") or "") != self.network_id:
            raise ExecutorFailure("workspace adoption network identity attestation failed")
        with self._registry_lock:
            self._task_containers[container_name] = actual_id
        return actual_id

    def adopt_reconciled_workspace(self, task: TaskSpec, attempt_id: str) -> str:
        """Adopt the deterministic exact-ID workspace for cleanup-only recovery."""
        plan = self._plans.get(str(attempt_id))
        if plan is None:
            agent_id = make_opaque_agent_id(
                self.config.campaign_id,
                task.task_id,
                str(attempt_id),
            )
            plan = self._task_network_plan(task.task_id, agent_id)
        return self._adopt_workspace_container(
            f"cybergym-workspace-{plan.opaque_agent_id}"
        )

    def _terminal_result_from_isolate_disk(
        self, gateway_task_id: str
    ) -> Mapping[str, Any] | None:
        """Read a terminal task record persisted on the isolate's data root.

        The gateway process may be dead while its ``task_results/`` tree
        persists on disk.  A record is accepted only when it names the exact
        gateway task and is deliverable — settled with final cost accounting,
        or released by the abandoned-residue grace with its disclosure;
        anything else is treated as absent so the caller keeps its typed
        refusal path.
        """
        root = self.config.isolate_data_root
        if root is None:
            return None
        path = root / "task_results" / f"{gateway_task_id}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(value, Mapping):
            return None
        if str(value.get("task_id") or "").strip() != gateway_task_id:
            return None
        return _redeliverable_terminal_frame(value)

    def reconcile_task(
        self,
        task: TaskSpec,
        task_dir: pathlib.Path,
        attempt_id: str,
        checkpoint: pathlib.Path,
    ) -> Mapping[str, Any]:
        """Deliver one already-terminal gateway result without re-running the agent.

        The checkpoint's last observed frame is authoritative when it is
        settled and cost-final; otherwise the live gateway is polled once.
        A non-terminal or unreachable task is reported with
        ``reconcile_disposition`` and left untouched for a later pass.  This
        method never raises for task-level conditions; unexpected internal
        errors become an ``infra_failed`` outcome with the ``reconcile``
        lifecycle so the outer ledger can settle the claim truthfully.
        """
        attempt_id = str(attempt_id or "").strip()
        if not attempt_id:
            raise ExecutorFailure("reconcile requires the checkpoint attempt id")
        agent_id = make_opaque_agent_id(self.config.campaign_id, task.task_id, attempt_id)
        plan = self._task_network_plan(task.task_id, agent_id)
        self._plans[attempt_id] = plan
        task_dir = _safe_abs(task_dir, "task_dir")
        _inside(task_dir, _safe_abs(self.config.run_root, "run_root"), "task_dir")
        workspace_dir = self._opaque_workspace_path(agent_id)
        container_name = f"cybergym-workspace-{plan.opaque_agent_id}"
        cleanup_ref = safe_task_path(
            self.config.run_root / "attestations", task.task_id, attempt_id
        ) / "workspace_cleanup.json"
        alias_ref = safe_task_path(
            self.config.run_root / "attestations", task.task_id, attempt_id
        ) / "workspace_backend_alias.json"
        terminal_evidence: dict[str, Any] = {}
        gateway_result: Mapping[str, Any] | None = None
        try:
            try:
                raw_checkpoint = json.loads(checkpoint.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ExecutorFailure("gateway checkpoint is unreadable") from exc
            if not isinstance(raw_checkpoint, Mapping):
                raise ExecutorFailure("gateway checkpoint is malformed")
            gateway_task_id = str(raw_checkpoint.get("gateway_task_id") or "").strip()
            if not gateway_task_id or not _GATEWAY_TASK_ID.fullmatch(gateway_task_id):
                raise ExecutorFailure("gateway checkpoint has no usable task id")
            cached = _redeliverable_terminal_frame(raw_checkpoint.get("result"))
            if cached is not None:
                # The cached frame is delivered without any gateway poll, so it
                # must be bound to this checkpoint's task exactly like the
                # isolate-disk fallback is; a foreign or id-less cached result
                # is an infra error, never deliverable.
                cached_task_id = str(cached.get("task_id") or "").strip()
                if cached_task_id != gateway_task_id:
                    raise ExecutorFailure("cached checkpoint result belongs to a different task")
                gateway_result = cached
            else:
                latest: Mapping[str, Any] | None = None
                poll_error: BaseException | None = None
                try:
                    latest = _unwrap_http_json(
                        self.config.http_runner(
                            "GET",
                            _gateway_path(
                                self.config.ouroboros_url,
                                "/api/tasks/" + urllib.parse.quote(gateway_task_id, safe=""),
                            ),
                            timeout=60,
                        ),
                        operation="Ouroboros task status",
                    )
                except Exception as exc:
                    # A dead isolate makes the live poll fail; its data root
                    # still persists terminal task records on disk.
                    poll_error = exc
                source = "gateway_poll"
                if latest is None:
                    latest = self._terminal_result_from_isolate_disk(gateway_task_id)
                    source = "isolate_task_results"
                    if latest is None:
                        raise ExecutorFailure(
                            "gateway task is unreachable and has no terminal "
                            "record on the isolate data root"
                        ) from poll_error
                returned_id = str(latest.get("task_id") or "").strip()
                if returned_id and returned_id != gateway_task_id:
                    raise ExecutorFailure("Ouroboros status response belongs to a different task")
                status = _response_status(latest)
                deliverable = _redeliverable_terminal_frame(latest)
                terminal_cost_unverifiable = (
                    status == "completed"
                    and status in _TERMINAL_GATEWAY_STATUSES
                    and deliverable is None
                )
                terminal = deliverable is not None or terminal_cost_unverifiable
                if deliverable is not None:
                    latest = deliverable
                if terminal and returned_id != gateway_task_id:
                    # A terminal frame we may deliver must be bound to this
                    # checkpoint's task exactly, like the isolate-disk
                    # fallback; an empty id is an infra error, not a delivery,
                    # and the foreign frame is never cached into the checkpoint.
                    raise ExecutorFailure("Ouroboros status response has no usable task id")
                _write_json(
                    checkpoint,
                    {
                        "gateway_task_id": gateway_task_id,
                        "status": status,
                        "result": dict(latest),
                        "reconciled": True,
                        "reconcile_source": source,
                    },
                )
                if not terminal:
                    return {
                        "status": "infra_failed",
                        "lifecycle": "reconcile_pending",
                        "infra_reason": "gateway_not_terminal",
                        "reconcile_disposition": "left_running",
                        "gateway_task_id": gateway_task_id,
                        "cost_usd": 0.0,
                        "cost_estimated": False,
                        "cost_final": True,
                        "cost_status": "known_no_dispatch",
                        "artifact_refs": {
                            "task_dir": str(task_dir),
                            "checkpoint": str(checkpoint),
                            "workspace_cleanup": str(cleanup_ref),
                        },
                        "error": "gateway task is not terminal; left for a later reconcile pass",
                    }
                if terminal_cost_unverifiable:
                    return {
                        "runtime_result": dict(latest),
                        "status": "infra_failed",
                        "lifecycle": "terminal_cost_unverifiable",
                        "infra_reason": "terminal_cost_unverifiable",
                        "reconcile_disposition": "delivery_failed",
                        "artifact_refs": {
                            "task_dir": str(task_dir),
                            "checkpoint": str(checkpoint),
                            "workspace_cleanup": str(cleanup_ref),
                        },
                        "error": (
                            "gateway task is terminal but its sparse accounting "
                            "cannot prove final or grace-eligible cost"
                        ),
                    }
                gateway_result = latest
            # Delivery re-runs the official submit inside the workspace
            # container, so a completed result needs the live container to
            # adopt.  A terminal non-completed result delivers its typed infra
            # row without touching Docker.
            if _response_status(gateway_result) == "completed":
                self._adopt_workspace_container(container_name)
            return self._deliver_gateway_result(
                task,
                task_dir,
                workspace_dir,
                container_name,
                agent_id,
                gateway_result,
                checkpoint=checkpoint,
                cleanup_ref=cleanup_ref,
                alias_ref=alias_ref,
                attestation_ref="",
                sidecar_attestation={"status": "adopted", "reason": "reconcile"},
                terminal_evidence=terminal_evidence,
            )
        except Exception as exc:
            if gateway_result is None:
                return {
                    "status": "infra_failed",
                    "lifecycle": "reconcile_blocked",
                    "infra_reason": type(exc).__name__,
                    "reconcile_disposition": "undeliverable",
                    "cost_usd": 0.0,
                    "cost_estimated": False,
                    "cost_final": True,
                    "cost_status": "known_no_dispatch",
                    "artifact_refs": {
                        "task_dir": str(task_dir),
                        "checkpoint": str(checkpoint),
                        "workspace_cleanup": str(cleanup_ref),
                    },
                    "error": str(exc),
                }
            return {
                "runtime_result": dict(gateway_result),
                **terminal_evidence,
                "status": "infra_failed",
                "lifecycle": "post_gateway_evaluation_failed",
                "infra_reason": type(exc).__name__,
                "reconcile_disposition": "delivery_failed",
                "artifact_refs": {
                    "task_dir": str(task_dir),
                    "workspace_dir": str(workspace_dir),
                    "checkpoint": str(checkpoint),
                    "workspace_backend_alias": str(alias_ref),
                    "workspace_cleanup": str(cleanup_ref),
                },
                "error": str(exc),
            }

    def release_reconciled_workspace(
        self, task: TaskSpec, attempt_id: str
    ) -> Mapping[str, Any] | None:
        """Release the workspace container adopted for a reconciled attempt.

        Reconcile defers this release until the result row AND the claim
        settle are both durable: cleaning the container inside
        ``reconcile_task`` would let a crash between cleanup and the row
        append strand a completed result undeliverable (container gone, row
        absent), and a crash between append and cleanup would leave the claim
        active while the next pass skips the task as recorded.  Returns
        ``None`` when the attempt never adopted a container (a terminal
        non-completed result delivers without touching Docker); a
        left-running attempt keeps its container for a later pass.
        """
        attempt_id = str(attempt_id or "").strip()
        plan = self._plans.get(attempt_id)
        if plan is None:
            agent_id = make_opaque_agent_id(self.config.campaign_id, task.task_id, attempt_id)
            plan = self._task_network_plan(task.task_id, agent_id)
        container_name = f"cybergym-workspace-{plan.opaque_agent_id}"
        cleanup_ref = safe_task_path(
            self.config.run_root / "attestations", task.task_id, attempt_id
        ) / "workspace_cleanup.json"
        with self._registry_lock:
            adopted = bool(self._task_containers.get(container_name))
        if not adopted:
            return None
        try:
            return self._cleanup_workspace_container(
                container_name, task.task_id, attempt_id, cleanup_ref
            )
        except Exception as cleanup_exc:
            try:
                _write_json(
                    cleanup_ref,
                    {
                        "schema": "ouroboros.benchmark.cybergym.workspace_cleanup.v1",
                        "status": "failed",
                        "ok": False,
                        "error_type": type(cleanup_exc).__name__,
                        "container_name": container_name,
                    },
                )
            except Exception:
                pass
            return {
                "status": "failed",
                "ok": False,
                "error_type": type(cleanup_exc).__name__,
                "container_name": container_name,
            }

    def finalize_adopted_campaign(self) -> Mapping[str, Any]:
        """Transfer verified adoption into cleanup ownership after full recovery."""
        if not self._adopted:
            raise ExecutorFailure("campaign was not adopted")
        with self._registry_lock:
            if self._task_containers:
                raise ExecutorFailure("adopted workspaces remain in custody")
        self._adopted = False
        self._network_created = True
        return self.close() or {"status": "not_needed", "ok": True}


def _recorded_attempt_rows(
    run_dir: pathlib.Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Read the latest row per task attempt from the run's result index.

    Lenient like the reconcile preamble has always been: blank or non-object
    lines are skipped, but an unreadable index is a typed refusal.
    """
    recorded: dict[tuple[str, str], dict[str, Any]] = {}
    index_path = run_dir / "result_index.jsonl"
    if not index_path.exists():
        return recorded
    try:
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                continue
            raw_task = value.get("task_id", value.get("instance_id", ""))
            if raw_task:
                key = (safe_task_id(str(raw_task)), str(value.get("attempt_id") or ""))
                recorded[key] = dict(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CyberGymError(f"result index is unreadable: {exc}") from exc
    return recorded


def _recorded_task_ids(run_dir: pathlib.Path) -> set[str]:
    return {task_id for task_id, _attempt_id in _recorded_attempt_rows(run_dir)}


def _workspace_cleanup_complete(
    run_dir: pathlib.Path,
    task_id: str,
    attempt_id: str,
) -> bool:
    path = (
        safe_task_path(run_dir / "attestations", task_id, attempt_id)
        / "workspace_cleanup.json"
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(value, Mapping)
        and value.get("ok") is True
        and value.get("status") == "verified"
    )


def _reconcile_task_dir(
    run_dir: pathlib.Path,
    task_id: str,
    attempt_id: str,
) -> pathlib.Path:
    """Select the isolated retry directory, falling back to the legacy layout."""
    retry_dir = safe_task_path(run_dir, task_id, attempt_id)
    if retry_dir.is_dir():
        return retry_dir
    return safe_task_path(run_dir, task_id)


@contextlib.contextmanager
def _result_index_lock(run_dir: pathlib.Path) -> Iterator[None]:
    """Hold the run's ``.result_index.lock`` across a check+append sequence.

    This is the same lock file ``append_cybergym_result`` serializes its
    paired writes through, so a reconcile pass that re-reads the recorded set
    and appends under one hold is atomic against every other writer.
    """
    lock_path = run_dir / ".result_index.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        locked = False
        try:
            try:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                locked = True
            except ImportError:
                pass
            yield
            lock.flush()
            os.fsync(lock.fileno())
        finally:
            if locked:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _append_row_if_unrecorded(run_dir: pathlib.Path, row: Mapping[str, Any]) -> bool:
    """Append ``row`` unless its task gained a row while delivery was running.

    Returns True when this call appended.  A False result means another
    writer — the original launcher finishing late, or an earlier attempt of
    the same task in this pass — already recorded the task officially, and
    the duplicate row is dropped instead of double-submitting it.
    """
    task = safe_task_id(str(row.get("task_id", row.get("instance_id", ""))))
    attempt = str(row.get("attempt_id") or "")
    with _result_index_lock(run_dir):
        recorded = _recorded_attempt_rows(run_dir)
        if (task, attempt) in recorded:
            _append_result_pair(run_dir, recorded[(task, attempt)])
            return False
        _append_result_pair(run_dir, row)
        return True


def _record_reconcile_pass(manifest: dict[str, Any], report: Mapping[str, Any]) -> None:
    """Append one reconcile pass to the manifest; passes are never overwritten.

    Each pass is a self-describing entry in ``extra.reconcile_passes`` so a
    later pass can never erase the record of an earlier reconcile attempt.
    """
    extra = manifest.setdefault("extra", {})
    passes = extra.get("reconcile_passes")
    if not isinstance(passes, list):
        passes = []
        extra["reconcile_passes"] = passes
    entry = dict(report)
    entry["pass_unix_ts"] = time.time()
    passes.append(entry)


def reconcile_main(args: argparse.Namespace) -> int:
    """Adopt an interrupted run root and deliver its terminal gateway results.

    The original launcher died after the gateway accepted tasks but before the
    delivery path wrote their rows.  This mode re-derives the deterministic
    campaign identities, attaches to the still-running isolate server and
    workspace containers, and runs the shared delivery path for every
    checkpointed attempt that has no row in ``result_index.jsonl``.  It never
    re-runs an agent, never starts new infrastructure, and never rewrites an
    existing row; tasks whose gateway task is still alive are reported and
    left for a later pass.
    """
    from devtools.benchmarks.common.manifests import finalize_run_manifest
    from devtools.benchmarks.cybergym.run_cybergym import (
        _build_default_executor,
        _task_specs,
        _validate_launcher_values,
    )

    try:
        _validate_launcher_values(args)
    except ValueError as exc:
        print(f"[cybergym] pre-admission refusal: {exc}", file=sys.stderr)
        return 2
    run_dir = pathlib.Path(args.reconcile).expanduser().resolve(strict=False)
    campaign_lock_handle = acquire_campaign_execution_lock(run_dir, blocking=False)
    if campaign_lock_handle is None:
        print(
            "[cybergym] reconcile refusal: another launcher or --reconcile process "
            "already holds this run root",
            file=sys.stderr,
        )
        return 2
    manifest_path = run_dir / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        campaign_lock_handle.close()
        print("[cybergym] reconcile refusal: run manifest is unavailable", file=sys.stderr)
        return 2
    if not isinstance(manifest, dict):
        campaign_lock_handle.close()
        print("[cybergym] reconcile refusal: run manifest is malformed", file=sys.stderr)
        return 2
    harness = manifest.get("harness")
    harness = harness if isinstance(harness, Mapping) else {}
    manifest_model = str(harness.get("model") or "").strip()
    if manifest_model and manifest_model != str(args.model):
        campaign_lock_handle.close()
        print(
            "[cybergym] reconcile refusal: manifest model does not match --model",
            file=sys.stderr,
        )
        return 2
    extra = manifest.get("extra")
    extra = extra if isinstance(extra, Mapping) else {}
    contract = extra.get("task_contract")
    contract = dict(contract) if isinstance(contract, Mapping) else None
    requested_ids: list[str] = []
    for raw in manifest.get("requested_task_ids") or ():
        try:
            requested_ids.append(safe_task_id(str(raw)))
        except ValueError:
            continue
    if not requested_ids and args.tasks_file:
        try:
            catalog = load_task_catalog(
                args.tasks_file,
                expected_sha256=str(args.expected_tasks_sha256 or ""),
                level=DEFAULT_LEVEL,
            )
            requested_ids = list(catalog["task_ids"])
        except CyberGymError as exc:
            campaign_lock_handle.close()
            print(f"[cybergym] reconcile refusal: {exc}", file=sys.stderr)
            return 2
    if not requested_ids:
        campaign_lock_handle.close()
        print(
            "[cybergym] reconcile refusal: the manifest names no tasks and no "
            "tasks file was supplied",
            file=sys.stderr,
        )
        return 2
    # The isolate's data root carries the durable terminal records used when
    # the gateway itself is dead.  Resolution order: an explicit --state-dir
    # override, the manifest's state layout (runs launched with --state-dir),
    # then the historical run-root layout of runs launched before the flag.
    isolate_data_root: pathlib.Path | None = None
    state_dir_override = str(getattr(args, "state_dir", "") or "").strip()
    if state_dir_override:
        override_root = (
            pathlib.Path(state_dir_override).expanduser().resolve(strict=False)
            / "ouroboros-data"
        )
        # The manifest is the provenance: when it records a state layout, an
        # override pointing anywhere else would read terminal records that are
        # not this run's, so the mismatch is refused rather than warned away.
        state_layout = extra.get("state_layout")
        recorded_root = ""
        if isinstance(state_layout, Mapping):
            recorded_root = str(state_layout.get("data_root") or "").strip()
        if recorded_root:
            manifest_root = pathlib.Path(recorded_root).expanduser().resolve(strict=False)
            if manifest_root != override_root:
                campaign_lock_handle.close()
                print(
                    "[cybergym] reconcile refusal: --state-dir disagrees with the "
                    "manifest's recorded state root "
                    f"(manifest: {manifest_root}, override: {override_root})",
                    file=sys.stderr,
                )
                return 2
        isolate_data_root = override_root
    if isolate_data_root is None:
        state_layout = extra.get("state_layout")
        if isinstance(state_layout, Mapping):
            recorded_root = str(state_layout.get("data_root") or "").strip()
            if recorded_root:
                isolate_data_root = pathlib.Path(recorded_root).expanduser().resolve(
                    strict=False
                )
    if isolate_data_root is None:
        # Runs launched before ``--state-dir`` kept the isolate data root at
        # ``run_root/ouroboros-data``; when that tree exists, adopt it so the
        # on-disk terminal-record fallback works for a dead gateway too.
        historical = run_dir / "ouroboros-data"
        if (historical / "task_results").is_dir():
            isolate_data_root = historical
    ouroboros_url = str(getattr(args, "ouroboros_url", "") or "").strip()
    if not ouroboros_url:
        ouroboros_url = str(harness.get("ouroboros_url") or "").strip()
    if not ouroboros_url and isolate_data_root is not None:
        # An interrupted run's manifest predates the server attestation; the
        # isolate persists its port under its data root.
        try:
            port = int(
                (isolate_data_root / "state" / "server_port")
                .read_text(encoding="utf-8")
                .strip()
            )
        except (OSError, ValueError):
            port = 0
        if 1 <= port <= 65535:
            ouroboros_url = f"http://127.0.0.1:{port}"
    if not ouroboros_url:
        campaign_lock_handle.close()
        print(
            "[cybergym] reconcile refusal: no gateway URL in --ouroboros-url, "
            "the manifest, or the isolate's persisted server_port",
            file=sys.stderr,
        )
        return 2

    # The lock was acquired before the manifest snapshot and remains held
    # through every recovery decision and final manifest publication.
    with contextlib.closing(campaign_lock_handle):
        ledger = BudgetLedger(
            run_dir / "claims.jsonl",
            cap_usd=float(args.budget_usd),
        )
        try:
            recorded_rows = _recorded_attempt_rows(run_dir)
        except CyberGymError as exc:
            print(f"[cybergym] reconcile refusal: {exc}", file=sys.stderr)
            return 2
        recorded_attempts = set(recorded_rows)
        recorded_tasks = {task_id for task_id, _attempt_id in recorded_attempts}

        slug_to_id = {task_slug(task_id): task_id for task_id in requested_ids}
        pending: list[tuple[str, str, pathlib.Path]] = []
        checkpointed: set[str] = set()
        checkpoints_root = run_dir / "checkpoints"
        if checkpoints_root.is_dir():
            for slug_dir in sorted(checkpoints_root.iterdir()):
                if not slug_dir.is_dir():
                    continue
                task_id = slug_to_id.get(slug_dir.name)
                if task_id is None:
                    continue
                for attempt_dir in sorted(slug_dir.iterdir()):
                    checkpoint = attempt_dir / "gateway_checkpoint.json"
                    if not (attempt_dir.is_dir() and checkpoint.is_file()):
                        continue
                    checkpointed.add(task_id)
                    try:
                        claim_state = ledger.attempt_state(attempt_dir.name)
                    except LedgerError as exc:
                        print(
                            f"[cybergym] reconcile refusal: {exc}",
                            file=sys.stderr,
                        )
                        return 2
                    if (
                        (task_id, attempt_dir.name) not in recorded_attempts
                        or claim_state == "reserved"
                        or not _workspace_cleanup_complete(
                            run_dir, task_id, attempt_dir.name,
                        )
                    ):
                        pending.append((task_id, attempt_dir.name, checkpoint))

        # A requested task with neither a result row nor a checkpoint was never
        # admitted by the gateway; reconcile cannot account for it, and calling
        # that "reconciled" would bless an incomplete run as recovered.
        unaccounted = sorted(
            task_id
            for task_id in requested_ids
            if task_id not in recorded_tasks and task_id not in checkpointed
        )

        reconcile_report: dict[str, Any] = {
            "schema": "ouroboros.benchmark.cybergym.reconcile.v1",
            "run_root": str(run_dir),
            "requested_count": len(requested_ids),
            "already_recorded": sorted(recorded_tasks),
            "pending_attempts": len(pending),
            "delivered": [],
            "left_running": [],
            "undeliverable": [],
            "skipped_recorded": [],
        }
        if unaccounted:
            reconcile_report["unaccounted"] = unaccounted
        if not pending:
            if unaccounted:
                reconcile_report["status"] = "incomplete"
                with finalize_run_manifest(
                    manifest_path, manifest, outcome="reconcile_incomplete", exit_code=2
                ) as final:
                    _record_reconcile_pass(manifest, reconcile_report)
                    final.update({"outcome": "reconcile_incomplete", "exit_code": 2})
                print(json.dumps(reconcile_report, indent=2, sort_keys=True))
                print(
                    "[cybergym] reconcile incomplete: requested tasks have neither "
                    "a result row nor a checkpoint: " + ", ".join(unaccounted),
                    file=sys.stderr,
                )
                return 2
            reconcile_report["status"] = "nothing_pending"
            with finalize_run_manifest(manifest_path, manifest, outcome="reconciled") as final:
                _record_reconcile_pass(manifest, reconcile_report)
                final.update({"outcome": "reconciled", "exit_code": 0})
            print(json.dumps(reconcile_report, indent=2, sort_keys=True))
            return 0

        try:
            callback = _build_default_executor(
                args,
                run_dir,
                ouroboros_url=ouroboros_url,
                isolate_data_root=isolate_data_root,
            )
        except CyberGymIntegrationUnavailable as exc:
            print(f"[cybergym] reconcile refusal: {exc}", file=sys.stderr)
            return 2
        executor_obj = getattr(callback, "executor", None)
        if executor_obj is None or not callable(getattr(executor_obj, "adopt_campaign", None)):
            print(
                "[cybergym] reconcile refusal: the concrete executor does not expose adoption",
                file=sys.stderr,
            )
            return 2
        try:
            adopt_report = executor_obj.adopt_campaign()
        except Exception as exc:
            print(
                "[cybergym] reconcile refusal: campaign adoption failed: "
                + type(exc).__name__,
                file=sys.stderr,
            )
            return 2
        reconcile_report["adoption"] = dict(adopt_report)

        specs = {spec.task_id: spec for spec in _task_specs(requested_ids, contract=contract)}
        exit_code = 0
        delivery_loop_completed = False
        try:
            for task_id, attempt_id, checkpoint in pending:
                spec = specs[task_id]
                attempt_key = (task_id, attempt_id)
                if attempt_key in recorded_attempts:
                    entry = {
                        "task_id": task_id,
                        "attempt_id": attempt_id,
                        "disposition": "recorded_recovery",
                    }
                    row = recorded_rows[attempt_key]
                    try:
                        with _result_index_lock(run_dir):
                            _append_result_pair(run_dir, row)
                        row_attempt = str(row.get("attempt_id") or "")
                        claim_state = ledger.attempt_state(attempt_id)
                        if claim_state in {"reserved", "unresolved"}:
                            if row_attempt != attempt_id:
                                raise LedgerError(
                                    "recorded row belongs to a different active attempt"
                                )
                            settle_finished_attempt(ledger, attempt_id, row)
                            claim_state = ledger.attempt_state(attempt_id)
                    except CyberGymError as exc:
                        entry.update(
                            disposition="settlement_pending",
                            error_type=type(exc).__name__,
                        )
                        reconcile_report["undeliverable"].append(entry)
                        exit_code = 2
                        continue
                    if _workspace_cleanup_complete(run_dir, task_id, attempt_id):
                        entry["disposition"] = "already_complete"
                        reconcile_report["skipped_recorded"].append(entry)
                        continue
                    try:
                        executor_obj.adopt_reconciled_workspace(spec, attempt_id)
                        release = executor_obj.release_reconciled_workspace(
                            spec, attempt_id,
                        )
                    except CyberGymError as exc:
                        entry.update(
                            disposition="cleanup_pending",
                            error_type=type(exc).__name__,
                        )
                        reconcile_report["undeliverable"].append(entry)
                        exit_code = 2
                        continue
                    if isinstance(release, Mapping) and release.get("ok") is False:
                        entry["disposition"] = "cleanup_pending"
                        reconcile_report["undeliverable"].append(entry)
                        exit_code = 2
                        continue
                    entry["claim_state"] = claim_state
                    entry["disposition"] = "recorded_recovered"
                    reconcile_report["skipped_recorded"].append(entry)
                    continue
                task_contract = spec.metadata.get("task_contract")
                task_dir = _reconcile_task_dir(run_dir, task_id, attempt_id)
                task_dir.mkdir(parents=True, exist_ok=True)
                outcome = dict(executor_obj.reconcile_task(spec, task_dir, attempt_id, checkpoint))
                disposition = str(outcome.get("reconcile_disposition") or "")
                entry = {
                    "task_id": task_id,
                    "attempt_id": attempt_id,
                    "status": str(outcome.get("status") or ""),
                    "lifecycle": str(outcome.get("lifecycle") or ""),
                    "disposition": disposition or "delivered",
                }
                if disposition == "left_running":
                    reconcile_report["left_running"].append(entry)
                    continue
                if disposition == "undeliverable":
                    # No terminal evidence could be obtained: the attempt may
                    # still be spending.  Keep its claim active and its
                    # denominator slot open instead of writing a zero-cost row.
                    reconcile_report["undeliverable"].append(entry)
                    exit_code = 2
                    continue
                try:
                    row = finalize_outcome_row(
                        run_dir,
                        spec,
                        task_dir,
                        outcome,
                        attempt_id=attempt_id,
                        contract=task_contract if isinstance(task_contract, Mapping) else contract,
                    )
                except Exception as exc:
                    entry["disposition"] = "row_refused"
                    entry["error_type"] = type(exc).__name__
                    reconcile_report["undeliverable"].append(entry)
                    exit_code = 2
                    continue
                try:
                    appended = _append_row_if_unrecorded(run_dir, row)
                except CyberGymError as exc:
                    # The index became unreadable mid-pass: fail closed and
                    # keep the adopted container for a later pass.
                    entry["disposition"] = "row_refused"
                    entry["error_type"] = type(exc).__name__
                    reconcile_report["undeliverable"].append(entry)
                    exit_code = 2
                    continue
                if not appended:
                    entry["disposition"] = "recorded_elsewhere"
                    try:
                        claim_state = ledger.attempt_state(attempt_id)
                    except LedgerError as exc:
                        entry["error_type"] = type(exc).__name__
                        reconcile_report["undeliverable"].append(entry)
                        exit_code = 2
                        continue
                    if claim_state in {"reserved", "unresolved"}:
                        entry["claim_state"] = claim_state
                        reconcile_report["undeliverable"].append(entry)
                        exit_code = 2
                        continue
                    reconcile_report["skipped_recorded"].append(entry)
                    executor_obj.release_reconciled_workspace(spec, attempt_id)
                    continue
                recorded_attempts.add(attempt_key)
                recorded_tasks.add(task_id)
                recorded_rows[attempt_key] = dict(row)
                entry["row_status"] = str(row.get("status") or "")
                try:
                    settle_finished_attempt(ledger, attempt_id, outcome)
                except CyberGymError as exc:
                    # The row is durable, but cleanup must wait until a later
                    # pass proves the claim terminal.
                    entry["settle"] = "pending"
                    entry["settle_type"] = type(exc).__name__
                    reconcile_report["undeliverable"].append(entry)
                    exit_code = 2
                    continue
                # The row and the claim settle are both durable; only now may
                # the adopted workspace container be released.  Every earlier
                # exit from this iteration keeps the container so a later pass
                # can redeliver from the checkpoint.
                release = executor_obj.release_reconciled_workspace(spec, attempt_id)
                if isinstance(release, Mapping) and release.get("ok") is False:
                    entry["workspace_cleanup"] = "failed"
                if disposition == "delivery_failed":
                    reconcile_report["undeliverable"].append(entry)
                    exit_code = 2
                else:
                    reconcile_report["delivered"].append(entry)
            delivery_loop_completed = True
        finally:
            try:
                fully_recovered = (
                    delivery_loop_completed
                    and exit_code == 0
                    and not unaccounted
                    and not reconcile_report["left_running"]
                    and not reconcile_report["undeliverable"]
                )
                cleanup = (
                    executor_obj.finalize_adopted_campaign()
                    if fully_recovered
                    else executor_obj.close()
                )
                reconcile_report[
                    "cleanup" if fully_recovered else "detach"
                ] = dict(cleanup or {})
            except Exception as exc:
                reconcile_report["detach"] = {"status": "error", "error_type": type(exc).__name__}
                exit_code = 2

        if unaccounted:
            exit_code = 2
        reconcile_report["status"] = "completed" if exit_code == 0 else "partial"
        manifest_outcome = "reconciled" if exit_code == 0 else "reconcile_partial"
        with finalize_run_manifest(
            manifest_path,
            manifest,
            outcome=manifest_outcome,
            exit_code=exit_code,
        ) as final:
            _record_reconcile_pass(manifest, reconcile_report)
            final.update({"outcome": manifest_outcome, "exit_code": exit_code})
        print(json.dumps(reconcile_report, indent=2, sort_keys=True))
        return exit_code
