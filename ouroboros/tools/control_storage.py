"""The mind's hands on its own disk: Storage -> Free, asked from a task (owner Q3 = A, 05.11.2026).

The Runtime section's ``storage`` fact gives the mind free space, what every retirer holds and why, and what
the last passes released (``context_runtime_facts.storage_fact``); ``query_data(kind="storage")`` reads the
count and the removers. This tool is the one way to ACT through the same proofs the owner's Dashboard ->
Storage -> Free uses: the request goes to the server as a control event, the server's one Free owner
(``storage_free.FreeSpace.request``) previews with the supervisor's liveness probe and interlock, runs the
retirers with the retention wait at zero over exactly the previewed objects, respects the owner's Keep and
every custody proof (what a retirer refuses stays, with its reason), records the receipt as the
``storage_free`` job row and tells the owner in Main. The tool waits a bounded time for that receipt and
returns it, its ``status`` and ``ok`` following the receipt (``done``; ``failed``, ``stopped`` or ``refused``
are never ok), or says the request is still under way (``requested``, ``queued``: no receipt is written before
its end). No threshold decides here: when and whether to free is the mind's judgment over the facts it already reads.
"""
from __future__ import annotations

import json
import time
import uuid

from ouroboros.tools.registry import ToolContext, ToolEntry

RECEIPT_POLL_SEC = 0.5  # how often the receipt row is read while the tool waits (transport pacing, not judgment)
WAIT_SEC_MAX = 300  # the longest one tool call waits for the receipt; a longer run reports as still running


def _free_storage(ctx: ToolContext, dry_run: bool = False, wait_sec: int = 90) -> str:
    from ouroboros import maintenance_jobs
    from ouroboros.storage_free import REQUEST_JOB
    from ouroboros.tool_access_paths import canonical_data_root
    from ouroboros.tools.control_events import _emit_control_event
    from ouroboros.utils import utc_now_iso

    request_id = uuid.uuid4().hex[:16]
    task_id = str(getattr(ctx, "task_id", "") or "")
    transport = _emit_control_event(ctx, {"type": "free_storage", "task_id": task_id, "request_id": request_id,
                                          "dry_run": bool(dry_run), "ts": utc_now_iso()})
    root = canonical_data_root(ctx)
    deadline = time.monotonic() + max(0, min(int(wait_sec or 0), WAIT_SEC_MAX))
    receipt: dict = {}
    while True:
        try:
            row = maintenance_jobs.job(root, REQUEST_JOB) or {}
        except Exception:
            row = {}
        if row.get("request_id") == request_id:
            receipt = row
            if row.get("state") in ("done", "failed"):
                break
        if time.monotonic() >= deadline:
            break
        time.sleep(RECEIPT_POLL_SEC)
    if receipt.get("state") in ("done", "failed"):
        outcome = receipt.get("outcome") if isinstance(receipt.get("outcome"), dict) else {}
        status = ("refused" if receipt.get("refusal") else "failed" if receipt["state"] == "failed" or outcome.get("error")
                  else "stopped" if outcome.get("stopped") else "done")
    else:
        status = "requested" if transport == "live" else "queued"
    return json.dumps({
        "ok": status in ("done", "requested", "queued"), "status": status, "request_id": request_id, "transport": transport,
        "receipt": {key: value for key, value in receipt.items() if key not in ("updated_at",)},
        "note": ("The run continues on the server; its receipt lands on the storage_free job row "
                 "(query_data kind=storage reads the removers; the owner sees the outcome in Main)."
                 if status in ("requested", "queued") else
                 "Not run: Free was busy or stopping; only this refusal receipt was written, and Main was not told."
                 if receipt.get("refusal") == "busy" and "finished_at" not in receipt else  # refused at request time
                 "Recorded as a storage_free event with actor=this task and reported to the owner in Main."
                 if receipt.get("outcome") else "No run, so no storage_free event; reported to the owner in Main."),
    }, ensure_ascii=False, default=str)


def storage_tool_entries() -> list[ToolEntry]:
    return [
        ToolEntry("free_storage", {
            "name": "free_storage",
            "description": (
                "Run the same Storage -> Free the owner has on the Dashboard: the app's own retirers release the work "
                "folders of finished tasks, finished working copies and declared temporary environments whose proofs "
                "hold (results saved into the task record, task over and not live, not kept by the owner), with the "
                "retention wait at zero. Nothing a proof refuses goes; every kept item carries its typed reason, and the "
                "owner's Keep is always respected. Read the Runtime storage fact first (free space, what each retirer "
                "holds and why, what the last passes released): use this when, in your judgment, the disk needs room now "
                "and the finished folders are worth less than their space; the owner's retention days are their "
                "inspection window, so say in Main why you freed early. dry_run previews the ordinary pool and the "
                "saved-work groups the owner can review and free from the Dashboard, without removing anything. Every run "
                "is recorded with you as the actor and reported to the owner in Main; the tool waits wait_sec for the "
                "receipt, else the run goes on and the receipt is on the storage_free job row."
            ),
            "parameters": {"type": "object", "properties": {
                "dry_run": {"type": "boolean", "default": False, "description": "Preview only: what would go and what stays, with reasons."},
                "wait_sec": {"type": "integer", "default": 90, "minimum": 0, "maximum": WAIT_SEC_MAX,
                             "description": "How long to wait for the receipt before returning 'requested' or 'queued' (the run goes on)."},
            }, "additionalProperties": False},
        }, _free_storage, safety_policy="skip"),  # a trusted built-in: it acts only through the retirers' own proofs
    ]
