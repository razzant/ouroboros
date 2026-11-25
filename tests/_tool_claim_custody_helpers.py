"""Durable local tool-claim fixtures and their process identity observations."""

# Bind these modules' platform names BEFORE any test patches ``platform_layer``: a module first
# imported under a patch would keep the fake for every later test in this process.
from ouroboros import platform_layer, process_containment, process_custody, process_identity  # noqa: F401
from ouroboros.task_results import load_task_result, write_task_result


INCIDENT_BIRTH = "Tue Oct 27 14:57:17 2026"


def _claim(root, holder, op, *, pid, birth=INCIDENT_BIRTH, attempt=1, kernel="", state="claimed", extra=None,
           tree="root"):
    row = load_task_result(root, holder) or {}
    claims = dict(row.get("launch_handoffs") or {})
    claim = {"tool": "run_command", "task_id": holder, "root_task_id": tree, "state": state,
             "claimed_at": "2026-10-27T17:23:28+00:00",
             "local_owner": {"pid": pid, "process_birth": birth, "task_attempt": attempt}}
    if kernel:
        claim["owner_proof"] = {"pid": pid, "kernel_birth": kernel, "session": "an-earlier-session"}
    claim.update(extra or {})
    claims[op] = claim
    write_task_result(root, holder, row.get("status") or "completed", root_task_id=tree, launch_handoffs=claims)


def _retired(root, holder):
    return dict((load_task_result(root, holder) or {}).get("retired_tool_invocations") or {})


def _platform(monkeypatch, *, gone=(), kernel=None, lstart="Tue Oct 27 23:28:21 2026"):
    monkeypatch.setattr(platform_layer, "pid_provably_gone", lambda pid: pid in set(gone))
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda _pid: False)
    monkeypatch.setattr(process_identity, "process_kernel_birth", lambda pid: (kernel or {}).get(pid))
    monkeypatch.setattr(platform_layer, "process_start_time", lambda _pid: lstart)
