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
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.parse
from collections.abc import Mapping
from typing import Any

from devtools.benchmarks.cybergym.cybergym_adapter import (
    DEFAULT_LEVEL,
    BudgetLedger,
    CyberGymError,
    CyberGymIntegrationUnavailable,
    TaskSpec,
    append_cybergym_result,
    finalize_outcome_row,
    load_task_catalog,
    safe_task_id,
    safe_task_path,
    settle_finished_attempt,
    task_slug,
)
from devtools.benchmarks.cybergym.cybergym_docker import (
    _GATEWAY_TASK_ID,
    _inside,
    _safe_abs,
    _write_json,
)
from devtools.benchmarks.cybergym.cybergym_lifecycle import _SETTLED
from devtools.benchmarks.cybergym.cybergym_sidecar import make_opaque_agent_id
from devtools.benchmarks.cybergym.cybergym_wire import (
    ExecutorFailure,
    _cost_is_pending,
    _gateway_path,
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

    def _terminal_result_from_isolate_disk(
        self, gateway_task_id: str
    ) -> Mapping[str, Any] | None:
        """Read a terminal task record persisted on the isolate's data root.

        The gateway process may be dead while its ``task_results/`` tree
        persists on disk.  A record is accepted only when it names the exact
        gateway task and is settled with final cost accounting; anything else
        is treated as absent so the caller keeps its typed refusal path.
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
        status = _response_status(value)
        if status not in _SETTLED or (status == "completed" and _cost_is_pending(value)):
            return None
        return value

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
            cached = raw_checkpoint.get("result")
            if (
                isinstance(cached, Mapping)
                and _response_status(cached) in _SETTLED
                and not (_response_status(cached) == "completed" and _cost_is_pending(cached))
            ):
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
                _write_json(
                    checkpoint,
                    {
                        "gateway_task_id": gateway_task_id,
                        "status": _response_status(latest),
                        "result": dict(latest),
                        "reconciled": True,
                        "reconcile_source": source,
                    },
                )
                status = _response_status(latest)
                if status not in _SETTLED or (
                    status == "completed" and _cost_is_pending(latest)
                ):
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
        finally:
            # A reconciled attempt releases its adopted workspace slot exactly
            # like a live one; a task left running keeps its container.
            with self._registry_lock:
                has_exact_id = bool(container_name and self._task_containers.get(container_name))
            if has_exact_id and gateway_result is not None:
                try:
                    self._cleanup_workspace_container(
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
    manifest_path = run_dir / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        print("[cybergym] reconcile refusal: run manifest is unavailable", file=sys.stderr)
        return 2
    if not isinstance(manifest, dict):
        print("[cybergym] reconcile refusal: run manifest is malformed", file=sys.stderr)
        return 2
    harness = manifest.get("harness")
    harness = harness if isinstance(harness, Mapping) else {}
    manifest_model = str(harness.get("model") or "").strip()
    if manifest_model and manifest_model != str(args.model):
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
            print(f"[cybergym] reconcile refusal: {exc}", file=sys.stderr)
            return 2
    if not requested_ids:
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
        isolate_data_root = (
            pathlib.Path(state_dir_override).expanduser().resolve(strict=False)
            / "ouroboros-data"
        )
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
        print(
            "[cybergym] reconcile refusal: no gateway URL in --ouroboros-url, "
            "the manifest, or the isolate's persisted server_port",
            file=sys.stderr,
        )
        return 2

    recorded: set[str] = set()
    index_path = run_dir / "result_index.jsonl"
    if index_path.exists():
        try:
            for line in index_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    continue
                raw_task = value.get("task_id", value.get("instance_id", ""))
                if raw_task:
                    recorded.add(safe_task_id(str(raw_task)))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            print(
                f"[cybergym] reconcile refusal: result index is unreadable: {exc}",
                file=sys.stderr,
            )
            return 2

    slug_to_id = {task_slug(task_id): task_id for task_id in requested_ids}
    pending: list[tuple[str, str, pathlib.Path]] = []
    checkpoints_root = run_dir / "checkpoints"
    if checkpoints_root.is_dir():
        for slug_dir in sorted(checkpoints_root.iterdir()):
            if not slug_dir.is_dir():
                continue
            task_id = slug_to_id.get(slug_dir.name)
            if task_id is None or task_id in recorded:
                continue
            for attempt_dir in sorted(slug_dir.iterdir()):
                checkpoint = attempt_dir / "gateway_checkpoint.json"
                if attempt_dir.is_dir() and checkpoint.is_file():
                    pending.append((task_id, attempt_dir.name, checkpoint))

    reconcile_report: dict[str, Any] = {
        "schema": "ouroboros.benchmark.cybergym.reconcile.v1",
        "run_root": str(run_dir),
        "requested_count": len(requested_ids),
        "already_recorded": sorted(recorded),
        "pending_attempts": len(pending),
        "delivered": [],
        "left_running": [],
        "undeliverable": [],
    }
    if not pending:
        reconcile_report["status"] = "nothing_pending"
        with finalize_run_manifest(manifest_path, manifest, outcome="reconciled") as final:
            manifest.setdefault("extra", {})["reconcile"] = reconcile_report
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

    ledger = BudgetLedger(run_dir / "claims.jsonl", cap_usd=float(args.budget_usd))
    specs = {spec.task_id: spec for spec in _task_specs(requested_ids, contract=contract)}
    exit_code = 0
    try:
        for task_id, attempt_id, checkpoint in pending:
            spec = specs[task_id]
            task_contract = spec.metadata.get("task_contract")
            task_dir = safe_task_path(run_dir, task_id)
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
            append_cybergym_result(run_dir, row)
            entry["row_status"] = str(row.get("status") or "")
            try:
                settle_finished_attempt(ledger, attempt_id, outcome)
            except CyberGymError as exc:
                # The row is the denominator surface and is already written; a
                # settle refusal means the claim was no longer active (e.g. the
                # original launcher settled it before dying).  Record the fact
                # rather than losing the row.
                entry["settle"] = "not_active"
                entry["settle_type"] = type(exc).__name__
            if disposition == "delivery_failed":
                reconcile_report["undeliverable"].append(entry)
                exit_code = 2
            else:
                reconcile_report["delivered"].append(entry)
    finally:
        try:
            reconcile_report["detach"] = dict(executor_obj.close() or {})
        except Exception as exc:
            reconcile_report["detach"] = {"status": "error", "error_type": type(exc).__name__}
            exit_code = 2

    reconcile_report["status"] = "completed" if exit_code == 0 else "partial"
    with finalize_run_manifest(manifest_path, manifest, outcome="reconciled") as final:
        manifest.setdefault("extra", {})["reconcile"] = reconcile_report
        final.update({"outcome": "reconciled", "exit_code": exit_code})
    print(json.dumps(reconcile_report, indent=2, sort_keys=True))
    return exit_code
