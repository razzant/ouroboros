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

``propose_payload_release`` only PROPOSES: exact heavy files a finished task keeps, with their meaning and loss
and the report, method and result that stay (``payload_release.propose``). The owner alone releases them, in
Storage -> Free's by-hand form; no tool of the mind confirms a loss.
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


def _read_refusal(ctx: ToolContext, root, task_id: str) -> str:
    """Why this caller may not read TASK_ID's store, or "" (the scope ``get_task_result`` reads by): a Presence turn
    only the work its binding addresses (``dialogue_provenance.presence_related_work``) or its own tree, a delegated
    child only its own tree, Main and a root task every task."""
    from ouroboros.dialogue_provenance import presence_caller_binding, presence_related_work, presence_target_record
    from ouroboros.task_store import read_document
    from ouroboros.tools.recent_tasks import _restricted_actor

    binding = presence_caller_binding(ctx)
    if binding is None and not _restricted_actor(ctx):
        return ""
    metadata = getattr(ctx, "task_metadata", None)
    own = str(getattr(ctx, "task_id", "") or "")
    own_root = str((metadata or {}).get("root_task_id") or own) if isinstance(metadata, dict) else own
    document = read_document(root, task_id) or {}
    if own and (task_id in (own, own_root) or str(document.get("root_task_id") or "") == own_root):
        return ""
    if binding is not None and presence_related_work(binding, presence_target_record(root, task_id) or {}):
        return ""
    return f"⚠️ ACCESS_BLOCKED: task {task_id or '?'} is outside what this task may read."


def _propose_payload_release(ctx: ToolContext, task_id: str = "", files: list | None = None,
                             retained: dict | None = None, consequence: str = "",
                             replace_offer: str = "", withdraw_offer: str = "") -> str:
    from ouroboros.payload_release import propose
    from ouroboros.tool_access_paths import canonical_data_root

    root, task_id = canonical_data_root(ctx), str(task_id or "")
    refusal = _read_refusal(ctx, root, task_id)
    if refusal:
        return json.dumps({"ok": False, "error": "ACCESS_BLOCKED", "detail": refusal}, ensure_ascii=False)
    answer = propose(root, task_id, files, retained, consequence, proposer=str(getattr(ctx, "task_id", "") or ""),
                     replace_offer=replace_offer, withdraw_offer=withdraw_offer)
    if answer.get("ok") and not answer.get("withdrawn"):
        answer["note"] = ("Proposed only. The owner sees it unchecked in Storage -> Free with every file, its role and "
                          "loss, and the refs that stay; nothing goes unless they choose it. Tell them why in Main.")
    return json.dumps(answer, ensure_ascii=False, default=str)


_FILE = {"type": "object", "properties": {
    "path": {"type": "string", "description": "Exact path relative to the task's artifact store (task_results/artifacts/<task>/)."},
    "role": {"type": "string", "description": "What this file is, in plain words (e.g. 'frozen copy of memory.sqlite taken for the measurement')."},
    "loss": {"type": "string", "description": "What can no longer be checked again once it is gone."},
}, "required": ["path", "role", "loss"], "additionalProperties": False}
_REFS = {"type": "array", "items": {"type": "string"}, "minItems": 1}


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
        ToolEntry("propose_payload_release", {
            "name": "propose_payload_release",
            "description": (
                "Propose that the owner release exact heavy files ONE finished task you may read keeps (database snapshots, "
                "VM disks, run data), only after you read what they are. You never release anything: the owner sees the proposal "
                "unchecked in Storage -> Free and alone decides. Name each file by its exact path in that task's artifact "
                "store with its role and loss, and the retained refs that keep what matters: the report, the method "
                "(scripts, recipe) and the result ('task_result' names the task's own answer). The host refuses links, "
                "files with another name, missing retained refs, overlap, and the always-protected classes: history, "
                "model requests and answers, reviews, memory, manifests, summaries, patches, chat media, attachments "
                "and source handles but raw test streams and a result-material copy its digest proves. An evidence "
                "folder may hold copies of model sessions or journals (a nanny's copies of its executor's transcripts): "
                "those are model inputs and outputs, never propose them; the host cannot tell them by name. A snapshot "
                "still needed by unfinished work is not proposed. replace_offer replaces one open proposal with a fresh "
                "owner choice; withdraw_offer closes one without releasing files. Otherwise files, retained and consequence are required. Old records stay. Proposal hashing "
                "reads every selected file once outside admission locks; large files can take time."
            ),
            "parameters": {"type": "object", "properties": {
                "task_id": {"type": "string", "description": "The finished task whose store holds the files."},
                "files": {"type": "array", "items": _FILE, "minItems": 1, "maxItems": 200},
                "retained": {"type": "object", "properties": {"report": _REFS, "method": _REFS, "result": _REFS},
                             "required": ["report", "method", "result"], "additionalProperties": False,
                             "description": "Store-relative paths that stay; each must exist and is checked again before the release."},
                "consequence": {"type": "string", "description": "What goes and what can no longer be redone; preserved whole."},
                "replace_offer": {"type": "string", "description": "Open offer id to supersede, with new files, retained refs and consequence."},
                "withdraw_offer": {"type": "string", "description": "Open offer id to withdraw; omit proposal fields and replace_offer."},
            }, "required": ["task_id"], "additionalProperties": False},
        }, _propose_payload_release, safety_policy="skip"),  # a proposal: the owner's choice in Storage is the effect
    ]
