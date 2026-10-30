"""Runtime self-control: restart, promotion, evolution, memory and model.

The verbs by which the agent changes its own running state or its durable
self — request a restart against an exact reviewed commit receipt, promote the
stable branch, ask for a deep self-review, read and write chat history,
scratchpad and identity, toggle evolution and background consciousness, and
switch the model or reasoning effort for the next round.
"""

from __future__ import annotations

from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, completed_local_read, publish_no_effect

import logging
import os

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import run_cmd, utc_now_iso

log = logging.getLogger(__name__)


from pathlib import Path
from ouroboros.config import runtime_setting


def _evolution_restart_block_reason(ctx: ToolContext) -> str:
    if str(ctx.current_task_type or "") != "evolution":
        return ""
    try:
        status = run_cmd(["git", "status", "--porcelain"], cwd=ctx.repo_dir).strip()
        head = run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir).strip()
    except Exception as exc:
        return f"could not verify local git durability: {exc}"
    reviewed_sha = str(getattr(ctx, "last_reviewed_commit_sha", "") or "").strip()
    if reviewed_sha and reviewed_sha == head and not status:
        metadata = getattr(ctx, "task_metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        tx = metadata.get("evolution_transaction")
        tx = tx if isinstance(tx, dict) else {}
        from supervisor.evolution_lifecycle import check_evolution_authority

        authority = check_evolution_authority(
            str(tx.get("campaign_id") or ""),
            str(tx.get("transaction_id") or ""),
            str(getattr(ctx, "task_id", "") or tx.get("task_id") or ""),
            commit_sha=head,
        )
        return "" if authority.get("ok") else (
            "the exact evolution commit receipt is no longer active "
            f"({authority.get('reason') or 'unknown'})"
        )
    if not reviewed_sha:
        return "commit_reviewed has not recorded an exact local commit receipt"
    if reviewed_sha and reviewed_sha != head:
        return "HEAD changed after the last reviewed local commit"
    return "commit_reviewed must create a local reviewed commit before evolution restart"


def _prepare_self_change(ctx: ToolContext, resume: str = "", in_place: bool = False) -> str:
    """Prepare (or deliberately resume) this task's body candidate before authoring or testing."""
    from ouroboros import body_candidate

    if in_place:
        from ouroboros.config import get_runtime_mode
        from ouroboros.consciousness_authority import effective_runtime_mode

        if effective_runtime_mode(get_runtime_mode(), getattr(ctx, "task_metadata", None)) != "cyber_pro":
            return _publish_tool_result(ctx, ToolResult(
                status="blocked", code="ACCESS_BLOCKED",
                text="⚠️ IN_PLACE_REQUIRES_CYBER_PRO: outside Cyber Pro, self-authoring uses a candidate."))
        if body_candidate.is_bound(ctx):
            return _publish_tool_result(ctx, ToolResult(
                status="blocked", code="ACCESS_BLOCKED",
                text="⚠️ CANDIDATE_ALREADY_BOUND: this task already authors a candidate; in_place applies before it."))
        body_candidate.choose_in_place(ctx)
        return ("OK: by your explicit Cyber Pro decision this task writes the serving checkout directly. "
                "Unfinished edits are live for every reader of that tree.")
    try:
        bound = body_candidate.prepare(ctx, resume=str(resume or "").strip())
    except body_candidate.CandidateRefused as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="blocked", code=exc.code, text=f"⚠️ {exc.code}: {exc.text}"))
    from ouroboros.tools.tool_resolution import active_repo_dir_for
    from ouroboros.workspace_copies import same_directory

    active = active_repo_dir_for(ctx)  # a room or external workspace stays this task's own folder
    users = ("Body writes, default process cwd, child copies, review and commit_reviewed now use it"
             if same_directory(active, bound["path"]) else
             f"Writes and review_change with root='system_repo', commit_reviewed and self_worktree children given "
             f"workspace_root={bound['path']} now use it, while default writes, reviews, commands without cwd and "
             f"delegate_start runs still use {active} (pass cwd={bound['path']} for commands in the candidate)")
    return (
        f"OK: body candidate {bound['candidate_id']} is {bound['state']} — path {bound['path']}, branch "
        f"{bound['branch']}, base {str(bound['base_sha'])[:12]}. {users}; processes started inside it get an "
        f"isolated HOME/data/settings environment. The running body at {bound['repo_dir']} is unchanged until "
        "an exact reviewed commit is adopted with request_restart(adopt_commit=...), or published through the Git/PR tools."
        + (f" The earlier checkout of this task was gone; its commits remain on branch {bound['previous_branch']}."
           if bound.get("previous_branch") else "")
    )


def _restart_receipt_failure(ctx: ToolContext, receipt: dict, text: str) -> str:
    """Keep an unknown ACK distinct from a known refusal, with the same request identity."""
    refused = receipt.get("status") in {"refused", "released"}
    return _publish_tool_result(ctx, ToolResult(
        status="blocked" if refused else "unavailable",
        code="LEGACY_BLOCKED" if refused else "LEGACY_UNAVAILABLE", text=text,
        meta={"request_id": receipt["request_id"], "restart_status": receipt["status"],
              **({"refusal": receipt["refusal"]} if receipt.get("refusal") else {})},
    ))


def _request_restart(ctx: ToolContext, reason: str, adopt_commit: str = "") -> str:
    from ouroboros.tools.control_events import request_restart_intent

    prior = getattr(ctx, "pending_restart_request", None)
    if prior and (prior["task_id"], prior["task_attempt"]) == (str(ctx.task_id or ""), int(getattr(ctx, "task_attempt", None) or 1)):
        receipt = request_restart_intent(ctx, prior["reason"])
        return ("Restart requested. The previously acknowledged request is unchanged." if receipt.get("status") in {"accepted", "ready"}
                else _restart_receipt_failure(ctx, receipt,
                    f"RESTART_{receipt['status'].upper()}: request {prior['request_id']}; no new request sent."))
    block_reason = _evolution_restart_block_reason(ctx)
    if block_reason:
        return f"⚠️ RESTART_BLOCKED: in evolution mode, {block_reason}."
    is_evolution = str(ctx.current_task_type or "") == "evolution"
    stated = str(reason or "").strip()
    restart_reason = stated or "agent_requested_restart"
    import json

    from ouroboros import body_adoption, body_candidate

    serving_dir = body_candidate.serving_repo_dir_for(ctx)
    adoption_note, handoff = "", None
    try:  # the restart marker names the SERVING checkout's expected state, never the candidate's
        if is_evolution and body_candidate.is_bound(ctx) and not str(adopt_commit or "").strip():
            adopt_commit = run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir).strip()
        if str(adopt_commit or "").strip():
            handoff = body_adoption.authorize(ctx, adopt_commit, reason=restart_reason)
            adoption_note = (f" It adopts candidate commit {handoff['cand'][:12]} onto {handoff['branch']} "
                             "once this generation's readers have stopped.")
    except body_adoption.AdoptionRefused as exc:
        return f"⚠️ RESTART_BLOCKED: {exc.code}: {exc.text}"
    if not adoption_note and body_candidate.is_bound(ctx):
        adoption_note = " The candidate is NOT adopted by this restart; it stays retained."
    # Persist expected ref for post-restart verification.
    try:
        sha = handoff["cand"] if handoff else run_cmd(["git", "rev-parse", "HEAD"], cwd=serving_dir)
        branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=serving_dir)
        evolution_claim = {}
        if is_evolution:
            metadata = getattr(ctx, "task_metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            tx = metadata.get("evolution_transaction")
            tx = tx if isinstance(tx, dict) else {}
            evolution_claim = {
                "campaign_id": str(tx.get("campaign_id") or ""),
                "transaction_id": str(tx.get("transaction_id") or ""),
                "task_id": str(ctx.task_id or tx.get("task_id") or ""),
                "commit_sha": str(sha or "").strip(),
            }
        # One marker schema with the supervisor's evolution restart (W4-F3).
        from supervisor.evolution_lifecycle import write_pending_restart_marker

        from ouroboros.tool_access_paths import canonical_data_root

        write_pending_restart_marker(
            canonical_data_root(ctx), expected_sha=sha, expected_branch=branch,
            reason=restart_reason, evolution_claim=evolution_claim,
        )
        if evolution_claim:
            ctx.pending_restart_is_evolution = True
            try:
                from supervisor.evolution_lifecycle import update_evolution_transaction

                update_evolution_transaction(
                    str(ctx.task_id or ""),
                    restart_decision="requested",
                    restart_required=True,
                    restart_requested_at=utc_now_iso(),
                    restart_expected_sha=str(sha or "").strip(),
                )
            except Exception:
                log.debug("Failed to record evolution restart request", exc_info=True)
    except Exception as exc:
        log.debug("Failed to read VERSION file or git ref for restart verification", exc_info=True)
        if is_evolution:
            return (
                "⚠️ RESTART_BLOCKED: the exact evolution restart receipt could not "
                f"be persisted ({exc})."
            )
    ctx.pending_restart_reason = restart_reason
    receipt = request_restart_intent(ctx, restart_reason, evolution=is_evolution)
    if receipt.get("status") not in {"accepted", "ready"}:
        return _restart_receipt_failure(ctx, receipt,
                f"RESTART_{receipt['status'].upper()}: request {receipt['request_id']}. "
                "Host acceptance is not confirmed; do not claim that new work is held. "
                "Calling again checks this same request.")
    ctx.last_push_succeeded = False
    ctx.last_reviewed_commit_sha = ""
    # The reason is the caller's own text, kept exact for the restart binding: quoted after the host's facts,
    # never stated as one of them.
    said = (f" Reason (the caller's own words): {json.dumps(stated, ensure_ascii=False)}." if stated
            else " No reason was given; the host default agent_requested_restart is recorded.")
    return f"Restart requested.{adoption_note}{said}"


def self_change_tool_entries() -> list:
    """The two verbs of own-body self-change, in catalog order: prepare the candidate, restart (and adopt)."""
    return [
        ToolEntry("prepare_self_change", {
            "name": "prepare_self_change",
            "description": (
                "Prepare this task's own-body candidate BEFORE running tests, scripts or commands that author or "
                "exercise Ouroboros's code: a separate checkout at the running body's commit, so unfinished work "
                "never reaches the files the live server imports. Body file writes and acting self_worktree "
                "children prepare it automatically; processes do not, so call this first for process-first work. "
                "Idempotent. resume deliberately continues ONE exact retained candidate (its id, owner task or "
                "branch) whose previous owner has ended. in_place is Cyber Pro's explicit decision to edit the "
                "serving checkout directly instead."),
            "parameters": {"type": "object", "properties": {
                "resume": {"type": "string", "description": "Exact retained candidate to continue; omit for this task's own."},
                "in_place": {"type": "boolean", "description": "Cyber Pro only: author the serving checkout directly."},
            }, "required": []},
        }, _prepare_self_change),
        ToolEntry("request_restart", {
            "name": "request_restart",
            "description": (
                "Ask supervisor to restart runtime after this task finalizes; confirmed acceptance holds new "
                "independent work, including Main, for Restart. An unconfirmed reply proves no acceptance. "
                "evolution requires its exact active commit receipt. A restart alone adopts nothing: pass "
                "adopt_commit (the exact reviewed commit of your body candidate) to adopt it locally at this "
                "restart, after the running generation's readers have stopped."),
            "parameters": {"type": "object", "properties": {
                "reason": {"type": "string"},
                "adopt_commit": {"type": "string", "description": "Exact reviewed candidate commit SHA to adopt at this restart; it needs that commit's passing post-commit run on record: selected over the whole candidate delta, or full (verifier, cutover or broad changes, Cyber Pro's unreviewed commits). Only full evidence permits this adoption's automatic branch/tag push."},
            }, "required": ["reason"]},
        }, _request_restart),
    ]


def _set_tool_timeout(ctx: ToolContext, seconds: int) -> str:
    """Persist the tool timeout, one field under the settings lock, and apply it at once."""
    try:
        timeout_sec = int(seconds)
    except (TypeError, ValueError):
        return f"⚠️ TOOL_ARG_ERROR (set_tool_timeout): invalid seconds={seconds!r}"
    if timeout_sec < 1:
        return "⚠️ TOOL_ARG_ERROR (set_tool_timeout): seconds must be >= 1"

    # Only this field is written: the runtime mode and every other saved value stay as they are on
    # disk (an owner's pending next-boot mode included), so the agent cannot change its Access or
    # lower a ratcheted mode here, and the process environment takes the field alone.
    from ouroboros.gateway.owner_settings import (
        _owner_update_settings,
        fields_on_saved_document,
        settings_document_mutation,
    )

    key = "OUROBOROS_TOOL_TIMEOUT_SEC"
    with settings_document_mutation():
        _owner_update_settings(fields_on_saved_document({key: timeout_sec}))
        os.environ[key] = str(timeout_sec)  # under the same lock as the commit
    return f"OK: OUROBOROS_TOOL_TIMEOUT_SEC set to {timeout_sec}s and applied immediately."


def _promote_to_stable(ctx: ToolContext, reason: str) -> str:
    event = {"type": "promote_to_stable", "reason": reason, "ts": utc_now_iso()}
    if str(ctx.current_task_type or "") == "evolution":
        metadata = getattr(ctx, "task_metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        tx = metadata.get("evolution_transaction")
        tx = tx if isinstance(tx, dict) else {}
        event["evolution_claim"] = {
            "campaign_id": str(tx.get("campaign_id") or ""),
            "transaction_id": str(tx.get("transaction_id") or ""),
            "task_id": str(getattr(ctx, "task_id", "") or tx.get("task_id") or ""),
            "commit_sha": str(
                getattr(ctx, "last_reviewed_commit_sha", "") or tx.get("commit_sha") or ""
            ),
        }
    ctx.pending_events.append(event)
    return f"Promote to stable requested: {reason}"


def _request_deep_self_review(ctx: ToolContext, reason: str, reviewer: str = "") -> str:
    # The executor is the one enabled catalog row the caller names (decision 3A),
    # else the Main model; availability follows that row (a native inspection
    # episode or a delegated session), not a model key alone.
    from ouroboros.deep_self_review import deep_review_route, deep_review_unavailable_text
    from ouroboros.consciousness_authority import consciousness_origin_metadata
    reviewer = str(reviewer or "").strip()
    if reviewer:
        from ouroboros.tools.arg_feedback import argument_refusal
        from ouroboros.tools.review_change import ReviewChangeArgumentError, system_review_row

        try:
            unavailable, identity = deep_review_route(system_review_row(reviewer))
        except ReviewChangeArgumentError as exc:
            return argument_refusal(ctx, "TOOL_ARG_ERROR (request_deep_self_review)", [str(exc)],
                                    effect="No review was queued.")
    else:
        unavailable, identity = deep_review_route()
    if unavailable:
        return deep_review_unavailable_text(unavailable)
    # A consciousness turn names itself: the review root then goes through the ONE
    # admission door and its spend stays inside the consciousness allowance.
    ctx.pending_events.append({"type": "deep_self_review_request", "reason": reason, "reviewer": reviewer,
                               "model": identity, "ts": utc_now_iso(),
                               **consciousness_origin_metadata({**(getattr(ctx, "task_metadata", None) or {}), "task_id": str(getattr(ctx, "task_id", "") or "")}, started_root=True)})
    return (f"Deep self-review requested (reviewer: {reviewer or 'Main'}, runs on {identity}). "
            "It will be queued and executed asynchronously.")


@completed_local_read
def _chat_history(
    ctx: ToolContext, count: int = 100, offset: int = 0, search: str = "",
    snapshot: str = "", **filters: str,
) -> str:
    from ouroboros.memory import Memory
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(
        getattr(ctx, "task_metadata", {}), dict
    ) else {}
    canonical_root = Path(str(
        metadata.get("budget_drive_root")
        or getattr(ctx, "budget_drive_root", "")
        or ctx.drive_root
    ))
    mem = Memory(drive_root=canonical_root)
    # Full project awareness (v6.32.0): the one mind's active recall spans every
    # thread (main + projects). The project-task working FOCUS is applied to the
    # passive default context only, never to this deliberate recall tool.
    return mem.chat_history(
        count=count, offset=offset, search=search, snapshot=snapshot, **filters,
    )


def _update_scratchpad(ctx: ToolContext, content: str, replaces: list | None = None, draft_source: str = "") -> str:
    """LLM-driven scratchpad write (Constitution P5: LLM-first): append a timestamped block, or
    fold the named blocks into one text the integrating mind accepts (#918; ``Memory.fold_scratchpad_blocks``)."""
    if not content or not isinstance(content, str) or len(content.strip()) < 10:
        return (
            _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("⚠️ REJECTED: content is empty or too short "
            f"(got {type(content).__name__}, len={len(content) if isinstance(content, str) else 'N/A'}). "
            "Scratchpad must have meaningful content (10+ chars). "
            "This likely means the tool call was malformed — check your arguments.")))
        )
    if replaces is not None and not (isinstance(replaces, list) and all(isinstance(item, str) for item in replaces)):
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
            "⚠️ REJECTED: replaces is a list of block sha256 strings (shown after each block's heading).")))
    from ouroboros.memory import Memory, ScratchpadFoldFailed, ScratchpadProjectionStale, scratchpad_block_sha256
    from ouroboros.knowledge import focus_signature
    from ouroboros.tool_access import canonical_data_root

    # One working memory, every room (P1): the scratchpad is the same file in
    # the main chat, in a project room, and in an external conversation, so a
    # project-scoped turn writes it like any other turn. The root follows the
    # same precedence as _chat_history, so a forked execution drive still
    # remembers into the canonical root the next context reads.
    mem = Memory(drive_root=canonical_data_root(ctx))
    mem.ensure_files()
    # The accepting focus, signed by the host; never a model name standing for authorship.
    metadata = {
        "focus": focus_signature(ctx)["focus"],
        "task_id": str(getattr(ctx, "task_id", "") or ""),
        "task_type": str(getattr(ctx, "current_task_type", "") or ""),
        "delegation_role": str((getattr(ctx, "task_metadata", {}) or {}).get("delegation_role", "")) if isinstance(getattr(ctx, "task_metadata", {}), dict) else "",
    }
    draft = str(draft_source or "").strip()
    stale = ""
    try:
        if replaces:
            written = mem.fold_scratchpad_blocks(content, replaces, metadata=metadata, draft_source=draft)
        else:
            written = mem.append_scratchpad_block(content, source="task", metadata={
                **metadata, **({"draft_source": draft} if draft else {})})
    except ScratchpadProjectionStale as error:
        written, stale = error.result, str(error)
    except ScratchpadFoldFailed as error:
        return _fold_refusal(ctx, error)
    except RuntimeError as exc:
        if "LEGACY_SCRATCHPAD_REQUIRES_MANUAL_UPGRADE" in str(exc):
            return _publish_tool_result(ctx, ToolResult(status="unavailable", code="LEGACY_UNAVAILABLE", text=(f"⚠️ {exc}")))
        raise
    if replaces:
        fold = written["block"]
        text = (f"OK: scratchpad fold applied: {len(written['sources'])} blocks "
                f"({sum(len(str(block.get('content', ''))) for block in written['sources'])} chars) replaced by one "
                f"fold block ({len(content)} chars, sha256 {scratchpad_block_sha256(fold)}); exact originals journaled "
                f"at entry_id={written['entry_id']}: {mem.scratchpad_journal_reader(ctx)}. "
                f"Active now: {written['blocks_after']} blocks, {written['content_len']} chars.")
    else:
        text = (f"OK: scratchpad block appended ({len(content)} chars, ts={written.get('ts', '?')[:16]}, "
                f"sha256 {scratchpad_block_sha256(written)})")
    if not stale:
        return text
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=text + (
        f"\n⚠️ SCRATCHPAD_PROJECTION_STALE: the canonical blocks hold this change, but the memory/scratchpad.md "
        f"projection was not regenerated ({stale}); the next context renders the canonical blocks and the next "
        "scratchpad write regenerates the file.")))


def _fold_refusal(ctx: ToolContext, error: RuntimeError) -> str:
    """No block changed (``ScratchpadFoldFailed``): a conflict names what is active now, a failure its cause."""
    from ouroboros.memory import scratchpad_block_sha256 as sha256
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.tool_access_reads import canonical_runtime_reader

    if error.kind not in ("not_active", "ambiguous"):
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_REPORTED_FAILURE", text=(
            f"⚠️ SCRATCHPAD_FOLD_FAILED ({error.kind}): {error}; no block changed — every active block stays "
            "(a preserved source row without blocks_folded in memory/scratchpad_journal.jsonl is preparation only).")))
    active = "\n".join(f"- [{str(block.get('ts', ''))[:16]} — {block.get('source', '?')}] sha256 {sha256(block)}: "
                       f"{' '.join(str(block.get('content', '')).split())[:80]}" for block in reversed(error.active))
    reader = canonical_runtime_reader(ctx, canonical_data_root(ctx), "memory/scratchpad_blocks.json")
    return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_REPORTED_FAILURE", text=(
        f"⚠️ SCRATCHPAD_FOLD_CONFLICT ({error.kind}): {error}; no block changed. Another focus may have folded "
        "them, or the view is older. Your earlier Scratchpad section may be stale. "
        f"Read the complete current blocks with {reader}. "
        f"Active blocks observed at this refusal, newest first (previews):\n{active}")))


def owner_contact_refusal(ctx: ToolContext, chat_id: object) -> str:
    """The owner-note gate shared by service-style notices and scheduled notes.

    A delegated child answers its parent, and a Presence or agent-to-agent turn
    speaks for an external conversation. Presence roots have a separate,
    source-bound private Main contact through ``_main_notice_refusal``; that
    exception does not grant scheduled notes or a generic service-notice voice.
    A note started by ordinary consciousness is legitimate contact (BIBLE P0).
    """
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
    from ouroboros.dialogue_provenance import presence_caller_binding

    for attr in ("task_metadata", "task_contract"):
        data = getattr(ctx, attr, None)
        if not isinstance(data, dict):
            continue
        lineage = data.get("lineage") if isinstance(data.get("lineage"), dict) else {}
        if (str(data.get("delegation_role") or lineage.get("delegation_role") or "").strip() == "subagent"
                or str(data.get("parent_task_id") or lineage.get("parent_task_id") or "").strip()):
            return "a delegated task reports to its parent (final result, tree_note or escalate), which decides what reaches the owner"
    if presence_caller_binding(ctx) is not None:
        return "a Presence turn speaks for its external conversation, not in the owner's main chat"
    if is_a2a_chat_id(chat_id):
        return "an agent-to-agent conversation has no main-chat voice"
    return ""


def _main_notice_refusal(ctx: ToolContext, chat_id: object) -> str:
    """Why this caller may not address Main, or "" when it may.

    A retained Presence root uses source-bound private contact. Other callers
    need an owner-visible root room (a Project or an owner-started root).
    """
    from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID
    from ouroboros.dialogue_provenance import run_origin

    from ouroboros.presence_owner_contact import contact_origin
    try:
        if contact_origin(ctx) is not None:
            return ""
    except (OSError, ValueError) as exc:
        return str(exc)
    if refusal := owner_contact_refusal(ctx, chat_id):
        return refusal
    if str(chat_id) == str(HIDDEN_CHAT_ID):
        return "a hidden/headless conversation is not an owner-visible root room"
    # Positive external transport ids can share the Project range; neither a
    # number (Main's own id included: a wake or scheduled root runs there too)
    # nor an agent-supplied destination proves an owner-visible room.
    from ouroboros.projects_registry import project_chat_for_task_tree, reserved_project_chat_ids
    from ouroboros.tool_access import canonical_data_root

    try:
        visible_chat = int(chat_id)
    except (TypeError, ValueError):
        return "the current chat has no proven owner-visible destination"
    data_root = canonical_data_root(ctx)
    projects = reserved_project_chat_ids(data_root)
    # The durable binding is the one truth about a root's Project: a mid-run
    # conversion or self-scope binds it without ever reaching current_chat_id.
    bound_chat = project_chat_for_task_tree(data_root, str(getattr(ctx, "task_id", "") or ""))
    if (visible_chat not in projects and bound_chat not in projects
            and not run_origin({"metadata": getattr(ctx, "task_metadata", None)})["owner_ingress"]):
        return "this root is neither bound to a registered Project nor started by the owner"
    return ""


def _send_user_message(ctx: ToolContext, text: str, reason: str = "", destination: str = "current",
                       reply_to_relay: str | None = None, request_id: str = "") -> str:
    """Send a separate owner reply without completing the ongoing task.

    ``destination="main"`` addresses the owner's main chat from an owner-visible
    root room: the row is typed ``main_notice``, written to Main whatever the
    sender's Project binding and shown by that Project too (one row,
    ``room_membership.row_rooms``); it never counts as the task's final answer.
    When to send one is the model's judgment (BIBLE P5); no host counter or
    timer triggers it.
    """
    chat_id = getattr(ctx, "current_chat_id", None)
    if chat_id is None or chat_id == "":  # 0 is a real hidden session, not absence
        return publish_no_effect(ctx, ToolResult(status="unavailable", code="CAPABILITY_UNAVAILABLE", text=("⚠️ No active chat — cannot send proactive message.")))
    if not text or not text.strip():
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("⚠️ Empty message.")))
    if reply_to_relay == "" and not getattr(ctx, "_open_relays", None):
        # Engines that fill every optional field send "" for a return they do not mean
        # (Grok, 27.10 stand call); with nothing open to return to, it is an ordinary send.
        reply_to_relay = None
    # Models may fill optional keys: an empty value is the omitted default.
    target = str(destination or "").strip().lower() or "current"
    if target not in ("current", "main"):
        from ouroboros.tools.arg_feedback import argument_refusal

        return publish_no_effect(ctx, argument_refusal(ctx, "SEND_USER_MESSAGE_DESTINATION", [
            f"destination={destination!r} is not a destination; use 'current' (this room) or 'main' (the owner's main chat)",
        ], effect="Nothing was sent."), tool_name="send_user_message")
    if target == "main":
        refusal = _main_notice_refusal(ctx, chat_id)
        if refusal:
            return publish_no_effect(ctx, ToolResult(status="blocked", code="ACCESS_BLOCKED", text=(
                f"⚠️ MAIN_NOTICE_BLOCKED: destination='main' refused: {refusal}. Nothing was sent; "
                "destination='current' still reaches this conversation.")))
        from ouroboros.presence_owner_contact import ContactRequestConflict, ContactRequestInvalid, contact_origin, send_main_message
        try:
            origin = contact_origin(ctx)
            if origin is not None:
                if reply_to_relay is not None:
                    raise ContactRequestInvalid("omit reply_to_relay; Presence Main contact uses its host-bound source")
                receipt = send_main_message(ctx, text, origin, request_id)
                ctx._owner_message_sent = True
                return receipt
        except ContactRequestConflict as exc:
            return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
                f"OWNER_CONTACT_REQUEST_CONFLICT: {exc}. The original message is unchanged. "
                "Use a new request_id only for an intentional new message.")))
        except ContactRequestInvalid as exc:
            return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
                text=f"TOOL_ARG_ERROR: {exc}. Nothing was sent."))
        except (OSError, ValueError) as exc:
            return _publish_tool_result(ctx, ToolResult(status="unavailable", code="CAPABILITY_UNAVAILABLE",
                text=f"OWNER_CONTACT_UNAVAILABLE: {exc}. Retry the same request_id and words."))
        from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
        from ouroboros.project_dialogue import MAIN_NOTICE_TYPE

        chat_id, system_type = WEB_UI_CHAT_ID, MAIN_NOTICE_TYPE
    else:
        # Discriminates the row from a bare final on history replay: the
        # client treats an UNtyped assistant row with a task_id as the task's
        # last word and would finalize a still-running live card. Persisted
        # via log_chat(record_type=...) exactly like media rows.
        system_type = "proactive_message"

    relay = {}
    reply_meta = {}
    if reply_to_relay is not None:
        from ouroboros.owner_relay import select_return
        from ouroboros.chat_references import resolve_reply
        from ouroboros.tool_access import canonical_data_root
        relay, error = select_return(ctx, reply_to_relay)
        error = error or owner_contact_refusal(ctx, chat_id)
        if relay and relay["source"]["chat_id"] != chat_id:
            error = error or _main_notice_refusal(ctx, chat_id)
        if error:
            return publish_no_effect(ctx, ToolResult(status="blocked", code="ACCESS_BLOCKED", text=error, meta={"reason_code": "RELAY_RETURN_UNRESOLVED"}))
        chat_id, system_type = relay["source"]["chat_id"], "relay_reply"
        reply_meta = {"reply_to": resolve_reply(canonical_data_root(ctx), relay["source"]["address"]),
                      "reply_to_relay": relay["relay_id"]}
    from ouroboros.tools.owner_delivery import deliver_owner_event, stamp_lineage
    from ouroboros.utils import append_jsonl
    event = stamp_lineage(ctx, {
        "type": "send_message",
        "chat_id": chat_id,
        "text": text,
        "format": "markdown",
        "is_progress": False,
        "system_type": system_type,
        "progress_meta": reply_meta,
        "ts": utc_now_iso(),
    })  # the producer before the debt and the dispatch: a replayed answer is the same work's
    if relay:
        # A relay's answer is owed before the queue sees it (the terminal outbox, as Presence Main contact is):
        # a send that fails after enqueue is retried under this id, and the open return retires only once owed.
        from supervisor.terminal_delivery import register_pending_delivery
        from ouroboros.tool_access import canonical_data_root

        event["delivery_id"] = f"relay-return:{relay['relay_id']}:{getattr(ctx, 'task_id', '') or ''}"
        if not register_pending_delivery(canonical_data_root(ctx), event):
            return publish_no_effect(ctx, ToolResult(status="unavailable", code="CAPABILITY_UNAVAILABLE", text=(
                f"⚠️ RELAY_RETURN_NOT_OWED: the answer to relay {relay['relay_id']} could not be recorded for delivery. "
                "Nothing was sent and the relay stays open; retry the same words."), meta={"reason_code": "RELAY_RETURN_NOT_OWED"}))
    mode = deliver_owner_event(ctx, event)
    append_jsonl(ctx.drive_logs() / "events.jsonl", {
        "ts": utc_now_iso(),
        "type": "proactive_message",
        "task_id": str(getattr(ctx, "task_id", "") or ""),
        "reason": reason,
        "destination": target,
        "transport_mode": mode,
        "text_preview": text[:200],
    })
    if relay:
        ctx._open_relays.pop(relay["relay_id"], None)
        from ouroboros.owner_relay import persist_held_state
        persist_held_state(ctx)
        return (f"OK: answer to relay {relay['relay_id']} owed to {relay['source'].get('label') or 'its source room'} "
                f"(delivery {event['delivery_id']}); the outbox retries it until sent.")
    if target == "main":
        return "OK: notice sent to the main chat." if mode == "live" else "OK: notice queued for delivery to the main chat."
    if mode == "live":
        return "OK: message sent to owner chat."
    return "OK: message queued for delivery."


def _update_identity(ctx: ToolContext, content: str, expected_revision: str | None = None) -> str:
    """Update identity manifest (who you are, who you want to become)."""
    if not content or not isinstance(content, str) or len(content.strip()) < 50:
        return (
            _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("⚠️ REJECTED: content is empty or too short "
            f"(got {type(content).__name__}, len={len(content) if isinstance(content, str) else 'N/A'}). "
            "Identity must be a substantial text (50+ chars). "
            "This likely means the tool call was malformed — check your arguments.")))
        )
    from ouroboros.memory import Memory
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.tool_access_reads import canonical_runtime_reader

    # One identity, every room (P1): who I am does not change with the room I
    # am speaking in, so a project room or an external conversation revises the
    # same continuous file. The root follows the same precedence as
    # _chat_history, so a forked execution drive still writes the identity the
    # canonical root reads back.
    mem = Memory(drive_root=canonical_data_root(ctx))

    from ouroboros.knowledge import focus_signature
    from ouroboros.memory_publication import publish_memory_file, publication_refusal

    publication = publish_memory_file(
        mem.identity_path(), expected_revision, lambda _old: content,
        journal=mem.identity_journal_path(), author=focus_signature(ctx),
        source_type=str((getattr(ctx, "task_metadata", {}) or {}).get("delegation_role") or "task"))
    reader = "\nCanonical identity reader:\n" + canonical_runtime_reader(ctx, canonical_data_root(ctx), "memory/identity.md")
    if not publication.ok:
        return _publish_tool_result(ctx, ToolResult(status="blocked", code="RESOURCE_CONSTRAINT_BLOCKED",
            text=publication_refusal(publication, "canonical memory/identity.md") + reader))
    old_content = publication.old_content
    result = f"OK: identity updated ({len(content)} chars); revision={publication.revision}"
    old_len = len(old_content)
    if old_len >= 400 and len(content) < old_len * 0.5:
        result += (
            f"\n⚠️ SELF_OVERWRITE_NOTICE: this replaced a {old_len}-char identity with "
            f"{len(content)} chars (>50% shrink). Identity is intentionally mutable (Bible P4), "
            "but full rewrites should be rare and reflect genuine self-creation — not a trivial turn. "
            "Read before writing (P12) and prefer evolving over replacing wholesale."
        )
    return result + reader


def _update_world(ctx: ToolContext, content: str, expected_revision: str | None = None) -> str:
    """Publish my map of my world; the host's measured block is kept or restored."""
    if not isinstance(content, str) or not content.strip():
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
            "⚠️ REJECTED: content is empty. Read memory/WORLD.md, revise the map and send the whole file.")))
    from ouroboros.context_budget import WORLD_SECTION_BUDGET_CHARS
    from ouroboros.memory import Memory
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.world_profiler import measured_block_was_edited, with_measured_block

    # One map, every room and every drive (P1): like identity, the canonical root
    # is the one the context reads back, so a forked child drive or Light mode
    # keeps what it learned instead of writing a copy that dies with the task.
    canonical = canonical_data_root(ctx)
    mem = Memory(drive_root=canonical)

    from ouroboros.knowledge import focus_signature
    from ouroboros.memory_publication import publish_memory_file, publication_refusal

    publication = publish_memory_file(
        mem.world_path(), expected_revision,
        lambda old: with_measured_block(content, old, str(getattr(ctx, "repo_dir", "") or ""), str(canonical)),
        journal=mem.world_journal_path(), author=focus_signature(ctx),
        source_type=str((getattr(ctx, "task_metadata", {}) or {}).get("delegation_role") or "task"))
    if not publication.ok:
        return _publish_tool_result(ctx, ToolResult(status="blocked", code="RESOURCE_CONSTRAINT_BLOCKED",
            text=publication_refusal(publication, "memory/WORLD.md")
            + f" The map lives at {mem.world_path()}; a forked drive holds only a copy of it."))
    # A forked drive's copy (what a child's read_file opens) mirrors what it just published,
    # so its next read carries the revision the canonical file now has.
    fork_copy = Memory(drive_root=Path(getattr(ctx, "drive_root"))).world_path()
    if fork_copy.resolve(strict=False) != mem.world_path().resolve(strict=False) and fork_copy.exists():
        try:
            from ouroboros.utils import write_bytes_atomic

            write_bytes_atomic(fork_copy, publication.content.encode("utf-8"))
        except OSError:
            # The map already landed canonically; a stale copy only costs the next read a refusal.
            log.debug("world map fork copy not mirrored", exc_info=True)
    size = len(publication.content)
    result = f"OK: world map updated ({size} chars); revision={publication.revision}"
    if measured_block_was_edited(content, publication.old_content):
        result += "\nThe measured block is the host's: your edits inside it were not applied; it refreshes at startup."
    if size > WORLD_SECTION_BUDGET_CHARS:
        result += (
            f"\n⚠️ WORLD_SIZE_NOTICE: the map is {size} chars, above the {WORLD_SECTION_BUDGET_CHARS}-char "
            "context budget. It still loads whole; move detail to knowledge_write and keep the map compact."
        )
    return result


def _toggle_evolution(ctx: ToolContext, enabled: bool, objective: str = "") -> str:
    """Toggle evolution mode on/off via supervisor event."""
    if bool(enabled):
        # Reflect the light-mode hard block in the tool's own result so the agent
        # is not told "ON" while the supervisor silently refuses it.
        try:
            from supervisor.evolution_lifecycle import evolution_block_reason

            block = evolution_block_reason()
        except Exception:
            block = ""
        if block:
            return block
    from ouroboros.consciousness_authority import consciousness_origin_metadata

    ctx.pending_events.append({
        "type": "toggle_evolution",
        "enabled": bool(enabled),
        "objective": str(objective or "").strip(),
        "ts": utc_now_iso(),
        # A Full-level consciousness turn/tree names itself: the campaign and its
        # cycle tasks then stay inside the consciousness allowance (PLAN 5.14 п.7).
        **consciousness_origin_metadata({**(getattr(ctx, "task_metadata", None) or {}), "task_id": str(getattr(ctx, "task_id", "") or "")}, started_root=True),
    })
    state_str = "ON" if enabled else "OFF"
    return f"OK: evolution mode toggled {state_str}."


def _toggle_consciousness(ctx: ToolContext, action: str = "status") -> str:
    """Control background consciousness: start, stop, or status.

    Start and stop are supervisor acts (queued events, unchanged). Status is a
    READ answered to the caller alone -- never a line in the owner's chat: the
    facts the runtime state persists, named with their source, and the clock's
    in-memory facts listed as not read rather than guessed.
    """
    if action == "status":
        return _consciousness_status_facts(ctx)
    ctx.pending_events.append({
        "type": "toggle_consciousness",
        "action": action,
        "ts": utc_now_iso(),
    })
    return f"OK: consciousness '{action}' requested."


def _consciousness_status_facts(ctx: ToolContext) -> str:
    """The persisted consciousness fields of the CALLER's canonical data root.

    One strict read of ``state/state.json`` under the root the caller's other
    canonical reads use (``budget_drive_root``, else ``drive_root``), not the
    process-global ``supervisor.state`` path and not its loader, whose display
    projection may substitute the backup's values. The state file is
    replaced atomically, so one lock-free read sees one whole version and
    writes nothing. A missing, unreadable or corrupt file is that named gap
    with no field guessed; a field the file lacks is listed, never defaulted.
    The toggle is a #1307 control: a value this copy cannot prove (no completed
    initialization witness, or unconfirmed after a recovery) is unknown, and a
    kept Panic flag, which bars every wake, is named.
    ``observed_at`` is when this read happened, not when the file was written.
    """
    import json
    import math

    from ouroboros.config import get_bg_wakeup_max_sec, get_bg_wakeup_min_sec
    from ouroboros.consciousness import (
        INTERVAL_STATE_KEY,
        LAST_WAKE_STATE_KEY,
        NEXT_WAKE_STATE_KEY,
        _iso,
        panic_blocks_wake,
    )
    from supervisor.state import control_value
    from supervisor.state_initialization import authority_reason

    metadata = ctx.task_metadata if isinstance(getattr(ctx, "task_metadata", None), dict) else {}
    path = Path(str(metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "")
                    or ctx.drive_root)) / "state" / "state.json"
    stored, gap = None, ""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        gap = "missing: no runtime state file at this path"
    except OSError as exc:
        gap = f"unreadable: {type(exc).__name__}"
    else:
        try:
            loaded = json.loads(raw.decode("utf-8"))
        except ValueError as exc:  # UnicodeDecodeError and JSONDecodeError alike
            gap = f"corrupt: {type(exc).__name__}"
        else:
            stored = loaded if isinstance(loaded, dict) else None
            gap = "" if stored is not None else "corrupt: the file is not a JSON object"
    facts = {"source": str(path), "observed_at": utc_now_iso()}
    fields = (("enabled", "bg_consciousness_enabled"), ("stored_next_wake_at", NEXT_WAKE_STATE_KEY),
              ("last_wake_ended_at", LAST_WAKE_STATE_KEY), ("chosen_interval_sec", INTERVAL_STATE_KEY))
    if stored is None:
        facts.update(read_gap=gap, not_read=[name for name, _key in fields])
    else:
        for name, key in fields:
            if key not in stored:
                facts.setdefault("not_recorded", []).append(name)
                continue
            value = facts[name] = stored[key]  # exactly as stored unless it is a readable time
            if key in (NEXT_WAKE_STATE_KEY, LAST_WAKE_STATE_KEY) and type(value) in (int, float) \
                    and math.isfinite(value) and value > 0:
                try:
                    facts[name] = _iso(value)
                except (OverflowError, OSError, ValueError):
                    pass
        unproven = authority_reason(path.parent.parent, str(stored.get("initialization_id") or "")) or (
            "" if control_value(stored, "bg_consciousness_enabled")[0] else "unconfirmed after a state recovery")
        if "enabled" in facts and unproven:
            facts["enabled"] = {"status": "unknown", "reason": unproven}
    if panic_blocks_wake(path.parent.parent):
        facts["panic_flag_kept"] = "state/panic_stop.flag is unreceived or unreadable: no wake starts until it is received"
    facts["configured_bounds_sec"] = {"min": get_bg_wakeup_min_sec(), "max": get_bg_wakeup_max_sec(),
                                      "source": "owner settings, not the state file"}
    facts["notes"] = [
        "stored_next_wake_at is the last time the clock persisted and fires only while enabled; the running "
        "clock keeps it no sooner than MIN after boot, and no event moves it.",
        "Not in this read (held in the supervisor's memory): the last wake outcome and error, failure "
        "backoff, the allowance window and a live wake task."]
    return json.dumps(facts, ensure_ascii=False, indent=2)


def _set_next_wakeup(ctx: ToolContext, seconds: int) -> str:
    """Choose the interval before the next consciousness wake-up.

    The requested seconds are clamped into the owner's configured bounds
    (``OUROBOROS_BG_WAKEUP_MIN``/``MAX``) and persisted on the runtime state as
    ``consciousness_next_interval_sec``, where the alarm clock reads the choice
    when it schedules the next wake. Any turn may call it (a wake-up picks its
    own rhythm; a Main turn may adjust it); with consciousness off the choice is
    stored, not refused, and applies once it is enabled. The alarm clock
    (``consciousness.py``) reads the value when the wake-up ends.
    """
    from ouroboros.config import get_bg_wakeup_max_sec, get_bg_wakeup_min_sec
    from supervisor.state import StateUnavailable, update_state

    try:
        requested = int(seconds)
    except (TypeError, ValueError):
        return f"⚠️ TOOL_ARG_ERROR (set_next_wakeup): invalid seconds={seconds!r}"
    low, high = get_bg_wakeup_min_sec(), get_bg_wakeup_max_sec()
    interval = max(low, min(high, requested))
    try:
        state = update_state(lambda st: st.__setitem__("consciousness_next_interval_sec", interval))
    except StateUnavailable as exc:
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="CAPABILITY_UNAVAILABLE", text=(
            f"⚠️ CAPABILITY_UNAVAILABLE: the interval was not stored: runtime state is unavailable ({exc.reason}).")))
    clamp_note = f" (requested {requested} s, clamped into {low}-{high} s)" if interval != requested else ""
    from supervisor.state import control_value

    known, enabled = control_value(state, "bg_consciousness_enabled")
    if not (known and enabled):
        return (f"OK: consciousness is {'off' if known else 'unknown (runtime state is recovering)'}; the next "
                f"wake-up interval of {interval} s{clamp_note} is stored for when it is enabled.")
    # The interval is finish-relative: the alarm reads it when a wake-up ends. Said plainly,
    # so a Main turn is not promised a wake it did not move (astra scope, round 7).
    return (f"OK: the wake-up interval is now {interval} s{clamp_note}; it applies from the end of the "
            "next wake-up (a wake-up already pending keeps its time; a wake-up calling this sets its own next one).")


def _switch_model(ctx: ToolContext, model: str = "", effort: str = "", primary: str = "") -> str:
    """LLM-driven model/effort switch (Constitution P5: LLM-first).

    Stored in ToolContext, applied on the next LLM call in the loop. ``primary``
    returns to this turn's primary binding (the acting model's choice): its model, role,
    locality and account policy with the owner's live wait-card choice for that
    role; "wait" also keeps a refusal there on the primary's own wait instead of
    paid alternatives. Effort intent is untouched.

    Effort follows the owner's range (``_effort_switch``): a Main focus — a chat turn,
    an ordinary root — keeps the level it works at outside Cyber Pro and the reply says
    where deeper thinking goes; a child, an evolution task or a consciousness wake moves
    inside the range; a pinned or model-named focus keeps its level.
    """
    from ouroboros.config import EFFORT_SCALE
    from ouroboros.llm import LLMClient
    available = LLMClient().available_models()
    changes = []

    # Validated before anything is applied: an unknown effort refuses the WHOLE call,
    # so a same-call model switch is not half-applied behind a rejected tier.
    requested_effort = str(effort or "").strip().lower()
    if requested_effort and requested_effort not in EFFORT_SCALE:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(f"⚠️ Unknown effort: {effort}. Valid: {', '.join(EFFORT_SCALE)}")))
    requested_primary = str(primary or "").strip().lower()
    route = getattr(ctx, "primary_route", None)
    if requested_primary and (requested_primary not in ("return", "wait") or model):
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
            "⚠️ primary must be 'return' or 'wait', without model.")))
    if requested_primary and not (isinstance(route, dict) and route.get("model")):
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="CAPABILITY_UNAVAILABLE", text=(
            "⚠️ This turn has no recorded primary route to return to.")))

    from ouroboros import route_choice
    if requested_primary:
        waiter = getattr(ctx, "model_wait_context", None)
        chosen = ((getattr(waiter, "overrides", None) or {}).get(route["role"]) or {})
        ctx.active_model_override = str(chosen.get("model") or route["model"])
        ctx.active_use_local_override = bool(chosen.get("use_local", route["use_local"]))
        ctx.active_role_override = route["role"]
        ctx.route_wait_on_primary = requested_primary == "wait"
        # The primary's own choice again: a following primary is re-read at the round boundary (the CURRENT one).
        ctx.route_choice = route_choice.explicit("wait_card") if chosen else route_choice.opening_choice(ctx)
        changes.append(f"primary route {ctx.active_model_override}{' (local)' if ctx.active_use_local_override else ''}"
                       f" (role {route['role']}; if it refuses: "
                       f"{'wait for it' if requested_primary == 'wait' else 'configured routes again'})")
    elif model:
        if model not in available:
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(f"⚠️ Unknown model: {model}. Available: {', '.join(available)}")))

        use_local = False
        if model == runtime_setting("OUROBOROS_MODEL") and runtime_setting("USE_LOCAL_MAIN", "").lower() in ("true", "1"):
            use_local = True
        elif model == runtime_setting("OUROBOROS_MODEL_LIGHT") and runtime_setting("USE_LOCAL_LIGHT", "").lower() in ("true", "1"):
            use_local = True
        else:
            from ouroboros.config import get_fallback_models
            if model in get_fallback_models() and runtime_setting("USE_LOCAL_FALLBACK", "").lower() in ("true", "1"):
                use_local = True

        ctx.active_model_override = model
        ctx.active_use_local_override = use_local
        ctx.route_wait_on_primary, ctx.active_role_override = False, None  # an explicit route ends a declared wait
        ctx.route_choice = route_choice.explicit("switch_model")  # the mind's own route: Settings no longer move it
        changes.append(f"model={model}{' (local)' if use_local else ''}")

    effort_note = ""
    if requested_effort:
        applied, effort_note = _effort_switch(ctx, requested_effort)
        if applied:
            ctx.active_effort_override = applied
            changes.append(f"effort={applied}")

    if not changes:
        if effort_note:
            return effort_note
        return (f"Current available models: {', '.join(available)}. Pass model and/or effort to switch, "
                "or primary='return'/'wait' to go back to this turn's primary route.")

    return f"OK: switching to {', '.join(changes)} on next round." + (f" {effort_note}" if effort_note else "")


def _effort_switch(ctx: ToolContext, requested: str) -> tuple[str, str]:
    """``(level to apply or "", one sentence)`` for ``switch_model(effort=…)`` under the
    owner's effort range (``settings_scales.choose_effort``). The decision never rewrites
    the dispatch record or a session leaf's effort fact: it is this focus's next round."""
    from ouroboros.route_spec import RouteSpec, model_named_effort
    from ouroboros.runtime_mode_policy import effort_range_binds
    from ouroboros.settings_scales import choose_effort, effort_range, resolve_effort

    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    child = str(metadata.get("delegation_role") or "").lower() == "subagent"
    task_type = str(getattr(ctx, "current_task_type", "") or metadata.get("type") or "").lower()
    strong_role = task_type == "evolution" or str(metadata.get("model_role") or "").lower() == "consciousness"
    binds = effort_range_binds(metadata)
    rng = effort_range()
    holding = str(getattr(ctx, "active_effort", "") or "")
    if child:
        snapshot = metadata.get("configured_subagent") if isinstance(metadata.get("configured_subagent"), dict) else {}
        route = snapshot.get("route") if isinstance(snapshot.get("route"), dict) else {}
        named = model_named_effort(RouteSpec(str(route.get("kind") or "api_model"), str(route.get("target_id") or "")))
        pin = str(snapshot.get("effort") or "")
        leaf = (" Your delegated run keeps the level it started at; this moves your own rounds."
                if str(metadata.get("effective_executor") or metadata.get("requested_executor") or "") == "harness" else "")
        level, source = choose_effort(requested, pin=pin, model_named=named, binds=binds, rng=rng)
        if source == "model_name":
            return "", f"effort={requested} not applied: your row's model name carries {level}, which holds in every mode.{leaf}"
        if source == "pin":
            return "", f"effort={requested} not applied: my human pinned your row at {level}; it holds outside Cyber Pro.{leaf}"
        moved = f" (requested {requested}, moved into my human's range {rng['min']}..{rng['max']})" if level != requested else ""
        return level, (moved + leaf).strip() if (moved or leaf) else ""
    from ouroboros.agent_dispatch import main_model_named_effort

    named = main_model_named_effort(metadata)
    if named:
        return "", f"effort={requested} not applied: the level in the model name ({named}) holds in every mode."
    if strong_role:
        level, source = choose_effort(requested, pin=str(metadata.get("reasoning_effort") or ""), binds=binds, rng=rng)
        if source == "pin":
            return "", f"effort={requested} not applied: my human pinned this task at {level}; it holds outside Cyber Pro."
        moved = f" (requested {requested}, moved into my human's range {rng['min']}..{rng['max']})" if level != requested else ""
        return level, moved.strip()
    if not binds:
        return requested, ""
    pinned = str(metadata.get("reasoning_effort") or "")
    holding = holding or pinned or resolve_effort("task")
    where = (f"the level my human pinned for this task ({holding})" if pinned
             else f"my human's recommended level ({holding})")
    return "", (f"effort={requested} not applied: Main works at {where}; deeper thinking is "
                "delegated — schedule_subagent(effort=…).")


def _finish_task(ctx: ToolContext, action: str, answer: str | None = None,
                 answer_sha256: str | None = None, rationale: str = "",
                 acceptance_subject: dict | None = None, pending_review: str | None = None,
                 silent: bool = False, swarm_comparison: str | None = None, summary: str = "") -> str:
    """Stage an author act; a self-started run (an own-clock or focused wake, a schedule-born root
    without a Presence binding) may choose silence and may leave one sentence for its human."""
    from ouroboros.consciousness_authority import is_focus_wake, is_own_clock_wake
    from ouroboros.dialogue_provenance import presence_caller_binding
    from ouroboros.task_results import resolve_ctx_lineage

    metadata = getattr(ctx, "task_metadata", None) or {}
    occurrence = metadata.get("schedule_occurrence")
    self_started = (bool(getattr(ctx, "is_direct_chat", False)) and (is_own_clock_wake(metadata) or is_focus_wake(metadata))
                    or (isinstance(occurrence, dict)
                        and str(occurrence.get("schedule_id") or "").strip()
                        and str(occurrence.get("token") or "").strip()
                        and resolve_ctx_lineage(ctx)["is_root_task"]
                        and presence_caller_binding(ctx) is None))
    summary = " ".join(str(summary or "").split())
    if silent and action != "finish":
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
            text="ERROR: COMPLETION_ARGUMENT: silent requires action=finish"))
    if (summary or silent) and not self_started:
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
            text="ERROR: COMPLETION_ARGUMENT: silent and summary are available only on a self-started run: "
                 "a consciousness wake's own turn, or a schedule-born root without a Presence binding"))
    allow_empty = False
    if silent:
        allow_empty = True
        if answer is None and not answer_sha256:
            answer = ""  # silence needs no words; complete bytes given anyway stay an internal result
    return stage_completion_request(ctx, {
        "action": action, "answer": answer, "answer_sha256": answer_sha256,
        "rationale": rationale, "acceptance_subject": acceptance_subject,
        "pending_review": pending_review, "wake_silent": bool(silent) or None, "wake_summary": summary or None,
        "swarm_comparison": swarm_comparison,
    }, allow_empty=allow_empty)


def stage_completion_request(ctx: ToolContext, request: dict, *, source: str = "finish_task",
                             allow_empty: bool = False, reply_later: bool = False) -> str:
    import copy
    import json
    from ouroboros.task_results import resolve_task_lineage

    action, answer, selector = request.get("action"), request.get("answer"), request.get("answer_sha256")
    # Strict tool schemas make a model fill the field it did not choose with "": an empty
    # selector beside an answer (or an empty answer beside a selector) names nothing.
    if isinstance(selector, str) and not selector.strip() and answer is not None:
        selector = None
        request = {**request, "answer_sha256": None}
    elif isinstance(answer, str) and not answer.strip() and isinstance(selector, str) and selector.strip():
        answer = None
        request = {**request, "answer": None}
    error = ""
    if action not in {"finish", "stop"}:
        error = "action must be finish or stop"
    elif action == "stop" and not str(request.get("rationale") or "").strip():
        error = "stop requires a rationale naming unfinished work"
    elif not reply_later and ((answer is None) == (selector is None)):
        error = "select exactly one of answer and answer_sha256"
    elif not reply_later and answer is not None and (not isinstance(answer, str) or (not allow_empty and not answer.strip())):
        error = "answer must be complete nonempty text"
    elif selector is not None and (not isinstance(selector, str) or not selector.strip()):
        error = "answer_sha256 must name an offered answer"
    elif request.get("pending_review") not in {None, "wait", "finish"}:
        error = "pending_review must be wait or finish"
    elif request.get("swarm_comparison") is not None and not isinstance(request["swarm_comparison"], str):
        error = "swarm_comparison must be text"
    elif request.get("pending_review") is not None and not resolve_task_lineage(
        getattr(ctx, "task_id", ""), metadata=getattr(ctx, "task_metadata", {}),
        parent_task_id=getattr(ctx, "parent_task_id", None),
    )["is_root_task"]:
        error = "pending_review is available only on root tasks"
    if error:
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text="ERROR: COMPLETION_ARGUMENT: " + error))
    staged = {key: copy.deepcopy(value) for key, value in request.items() if value is not None}
    staged.update(source=source, reply_later=reply_later, allow_empty=allow_empty,
                  observation=copy.deepcopy(getattr(ctx, "_completion_observation", {})))
    previous = getattr(ctx, "_completion_request", None)
    if previous is not None and previous.get("observation") == staged["observation"] and previous != staged:
        ctx._completion_conflict = True
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR",
            text="ERROR: COMPLETION_CONFLICT: contradictory completion requests in one response; select again after seeing all results."))
    ctx._completion_request = staged
    return json.dumps({"status": "completion_requested", "completion_control": True, "action": action}, ensure_ascii=False)
