"""Control tools: restart, timeout settings, scheduling, review, chat history, model switching."""

from __future__ import annotations

import json  # noqa: F401
import logging
import os  # noqa: F401
import queue  # noqa: F401
import shutil  # noqa: F401
import threading  # noqa: F401
import time  # noqa: F401
import uuid  # noqa: F401
from hashlib import sha256  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Callable, Dict, List, Optional  # noqa: F401

from ouroboros.config import (
    EFFORT_SCALE,
    apply_settings_to_env,  # noqa: F401
    get_max_subagent_depth,  # noqa: F401
    load_settings,  # noqa: F401
    save_settings,  # noqa: F401
)
from ouroboros.contracts.task_contract import (
    build_task_contract,  # noqa: F401
    effective_acceptance_claims,  # noqa: F401
    normalize_allowed_resources,  # noqa: F401
)
from ouroboros.depth_evidence import parse_task_depth  # noqa: F401
from ouroboros.headless import prepare_task_drive, task_state_dir  # noqa: F401
from ouroboros.outcomes import normalize_outcome_axes  # noqa: F401
from ouroboros.subagent_runtime import (
    SubagentSelectionError,  # noqa: F401
    effective_runtime_subagent_settings,  # noqa: F401
    select_subagent_snapshot,  # noqa: F401
)
from ouroboros.subagents import (
    LEGACY_SUBAGENT_FIELDS,  # noqa: F401
    build_subagent_envelope,  # noqa: F401
)
from ouroboros.task_results import (
    STATUS_COMPLETED,  # noqa: F401
    STATUS_REJECTED_DUPLICATE,  # noqa: F401
    STATUS_REQUESTED,  # noqa: F401
    validate_task_id,  # noqa: F401
    write_task_result,  # noqa: F401
)
from ouroboros.task_status import load_effective_task_result, wait_for_effective_tasks  # noqa: F401
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE, LOCAL_READONLY_SUBAGENT_MODE  # noqa: F401
from ouroboros.tools.control_delegation import (
    _ensure_project_scope,
    admitted_depth_cap,  # noqa: F401
    child_budget_for_schedule,  # noqa: F401
    normalize_required_capabilities,  # noqa: F401
    profile_from_task_constraint,  # noqa: F401
    record_depth_limit_refusal,  # noqa: F401
    resolve_cooperative_write_root,  # noqa: F401
    schedule_delegation_refusal,  # noqa: F401
)
from ouroboros.tools.registry import (  # noqa: F401  # noqa: F401
    ToolContext,
    ToolEntry,
    active_repo_dir_for,
    system_repo_dir_for,
)
from ouroboros.utils import (  # noqa: F401
    append_jsonl,
    atomic_write_json,
    run_cmd,
    truncate_review_artifact,
    utc_now_iso,
)

log = logging.getLogger(__name__)


# Guards parent-side shared ctx state mutated during (possibly parallel)
# schedule_subagent emission within one tool-call round. Process-local: a parent
# ctx is never shared across processes, so a threading.Lock is sufficient.


# Runtime-INTERNAL scheduling options, deliberately absent from the public schema and
# structurally unreachable from a model tool call: they ride in the POSITIONAL-ONLY `internal`
# mapping, which no keyword argument produced from tool-call JSON can ever bind to. Keeping
# them out of the signature is also what holds the handler inside the <8-parameter contract.
#
# The set is EMPTY as of v6.87.7: its only member, `deadline_at`, became a public parameter
# once the caller judged to be the right one turned out to be the parent LLM itself — it is
# the parent that knows when a child's handoff stops being useful. The seam stays because it
# is closed and cheap, and an unknown key here still fails loudly rather than being ignored.


# A parameter this tool used to publish, mapped to the durable field it wrote.
# Separate from "unsupported" because a caller passing one is not guessing: it read
# a schema that was real, and "unsupported argument" hides that the capability still
# exists and is now derived. The REASON is not restated here — it is
# `LEGACY_SUBAGENT_FIELDS`, the same sentence the dispatch resolution puts on the
# record when it ignores a stored value, so the live refusal and the durable
# disclosure cannot come to disagree about why the field went away.

# D23: accepted only by the real registry invocation path and intentionally absent
# from ``schedule_subagent_properties``.  The handler attribute is consumed by the
# generic registry seam; no tool-name special case or public schema alias exists.


# Registration-race grace for a wait set in which NOTHING was minted (v6.91):
# "not YET registered" is a real state for a child scheduled moments ago, so a
# phantom-only wait still polls — but only for this long, instead of blocking
# the parent for the whole requested window on ids that exist nowhere.


# promote_chat_to_task tool description (hoisted from get_tools for the
# 300-line function gate; v6.70.0 added the ground-truth-probe contract).
_PROMOTE_CHAT_DESCRIPTION = (
    "Start a supervised pooled task while this conversation remains available. Use for "
    "independent work needing its own queue slot, admission and reviews, or an owner's "
    "explicit request for a separate task; tools/files/multiple steps alone can stay here. "
    "For an EXISTING artifact, ground-truth its existence with one cheap probe "
    "(list_skills/list_files): memory of past work is not evidence it still exists. "
    "Give a short human-readable title. project_name creates a named Project and a NEW "
    "task there; project_id uses an existing Project. Interpret the owner's intended name, "
    "not keywords. To move THIS task instead, use ensure_project_scope. An unmet Swarm "
    "force_plan obligation moves to the new task; further work here is unplanned. "
    "Use predecessor_task_id for one settled result (see argument); steer_task for a live root. "
    "workspace_root selects the working folder; by default a Project task's file/shell/git "
    "tools use its registered folder, not the Ouroboros repo. workspace='none' opts out. "
    "Owner follow-ups can steer the task. Claim creation only on OK, never PROMOTE_REJECTED "
    "or PROMOTE_UNCONFIRMED; do not retry UNCONFIRMED automatically."
)

# route_to_project tool description, hoisted from get_tools for the same function gate.
_ROUTE_TO_PROJECT_DESCRIPTION = (
    "Route a main-chat message to an EXISTING project so the work continues in that "
    "project's own context (memory/journal/thread), keeping the main chat free. Use "
    "when a message clearly belongs to a known project (call list_projects first if "
    "unsure of the id). If confidence is low or several projects/tasks could match, "
    "CALL THIS TOOL with project_id='' and the owner's message: it emits the typed "
    "needs_manual_target acknowledgement with host-validated task options and New task "
    "in Project; prose alone cannot emit that typed choice. For brand-new work that is not yet a project, "
    "use promote_chat_to_task instead. When continuing one settled result (any project; "
    "the host list is a hint), pass its internal `predecessor_task_id`; pass an empty "
    "string for fresh work. Returns a visible routing receipt."
)

# The new root's starting effort: one optional choice shared by both verbs that mint a
# root from a conversation. Child actors keep their configured profiles (schedule_subagent).
_ROOT_EFFORT_PARAM = {"type": "string", "enum": list(EFFORT_SCALE), "description": (
    "Optional: the reasoning effort the NEW task starts on, chosen for this work (it can still "
    "switch_model later). Omit for the configured Task default. A request: the route may adapt it.")}


_SCHEDULE_SUBAGENT_DESCRIPTION = (
    "Schedule a live child of Ouroboros; returns task_id. Default READ-ONLY children inspect "
    "local repo/data/history and web/browser. Each returns findings; apart from knowledge notes, memory marks "
    "and chronicle drafts in its own name, it cannot write local state, commit, enable tools, or run "
    "shell/review/runtime/skills. write_surface selects a MUTATIVE (acting) child; you alone commit "
    "the live Ouroboros body. workspace_root selects the starting folder or inherits yours. "
    "self_worktree copies that Git source's current eligible files, including uncommitted work, into an isolated tree; "
    "use integrate_subagent_patch for its parallel/best-of-N patch. Native children on "
    "external_workspace write directly to the SHARED external directory (write_root or parent workspace); "
    "integrate_subagent_patch verifies those files without reapplying. For several builders of ONE "
    "new deliverable, use external_workspace and OMIT write_root: the host creates one shared Git "
    "tree, inherited by descendants; integrate_subagent_patch verifies their combined files. "
    "genesis gives EACH child a separate empty Git repo under the durable projects root, for a new "
    "game/site/app/Ouroboros or independent best-of-N builds; the project directory IS the deliverable, "
    "never integrated into this repo. Harness-delegated work uses a private snapshot and integrate_delegated_patch. "
    "Runtime data, including installed skill payloads, is never a write_surface. For skill mutation, "
    "call delegate_start(subagent_id=..., prompt=..., root='skill_payload', bucket=..., skill_name=...) "
    "yourself; children cannot open payload delegation and may only design/review it read-only. "
    "Mutative children cannot commit or enable tools; children may write knowledge notes and memory marks "
    "in their own name and publish chronicle pages and parts only as drafts, which the integrating mind "
    "accepts or rejects (chronicle_write kind=decision); identity and scratchpad stay with the parent. Cyber-effective "
    "children inherit selected review/skill/runtime tools, subject to explicit task restrictions. "
    "Nested delegation obeys configured depth/cap limits; delegation_intent, may_mutate and may_fan_out "
    "propagate requests for further children/grandchildren structurally. Independent children scheduled "
    "in one round run concurrently; wait_tasks(any_terminal) absorbs whichever finishes first. On "
    "cache-write-priced routes, each sibling launched before the first sibling's first response pays "
    "a full prefix write: choose burst latency or spaced cost savings. For ongoing addressed turns, "
    "state in objective/constraints what is interim, whom to address and what ends participation. "
    "Native children use forward_to_worker for you, siblings or any task in their tree, await_messages "
    "to wait, and a final answer to end participation. For sessions, answer mid-run questions with "
    "delegate_answer; after settlement, delegate_start(subagent_id=..., continue_from=run_id, prompt=...) starts another "
    "run, reusing the session where possible or retained evidence in a new session; repetition may be "
    "needed. Retrieve the handoff with get_task_result, wait_task or wait_tasks before relying on it."
)


def get_tools() -> List[ToolEntry]:
    from ouroboros.tools.control_maintenance import maintenance_tool_entries

    return [
        *maintenance_tool_entries(),
        ToolEntry("finish_task", {"name": "finish_task",
            "description": "Select the complete answer and request completion of your current task. "
                "Use finish after considering the observed work, or stop with a rationale naming unfinished work. "
                "Select exactly one of answer or a host-offered answer_sha256. This grants no success, cancels no children, and retains all configured review and owner controls.",
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["finish", "stop"]}, "answer": {"type": "string", "description": "The complete answer, including a short correction."},
                "answer_sha256": {"type": "string", "description": "Exact offered retained or whole held response hash."},
                "rationale": {"type": "string", "description": "For stop, what remains unfinished."}, "acceptance_subject": {"type": "object", "properties": {"owner_source_sha256": {"type": "string"},
                    "effective_criteria": {"type": "string"}, "material_tool_indices": {"type": "array", "items": {"type": "integer"}},
                }, "required": ["owner_source_sha256"]},
                "pending_review": {"type": "string", "enum": ["wait", "finish"], "default": "wait"},
            }, "required": ["action"]}}, _finish_task),
        ToolEntry("set_tool_timeout", {
            "name": "set_tool_timeout",
            "description": "Update the global tool timeout in settings.json and apply it immediately without restart.",
            "parameters": {"type": "object", "properties": {
                "seconds": {"type": "integer", "description": "New timeout in seconds (>= 1)"},
            }, "required": ["seconds"]},
        }, _set_tool_timeout),
        *self_change_tool_entries(),
        ToolEntry("promote_to_stable", {
            "name": "promote_to_stable",
            "description": "Promote ouroboros -> ouroboros-stable. Call when you consider the code stable.",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
        }, _promote_to_stable),
        ToolEntry("promote_chat_to_task", {
            "name": "promote_chat_to_task",
            "description": _PROMOTE_CHAT_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {"type": "string", "description": "What the task must accomplish."},
                    "title": {"type": "string", "description": "Human-readable task name, <=80 chars; also used as the Project name if the owner later converts the task.", "default": ""},
                    "project_name": {"type": "string", "description": "Display name of a NEW Project for the NEW task; its filesystem id is derived. Use ensure_project_scope to move THIS task instead.", "default": ""},
                    "expected_output": {"type": "string", "description": "What done looks like.", "default": ""},
                    "project_id": {"type": "string", "description": "Optional EXISTING project scope (filesystem-clean id).", "default": ""},
                    "workspace_root": {"type": "string", "description": "Absolute ordinary folder or Git worktree root outside Ouroboros repo/data, validated at admission. File/process work supports ordinary folders; Git operations need a worktree. Omit for the Project's registered working_dir or, in Main, Ouroboros's own repo.", "default": ""},
                    "workspace": {"type": "string", "description": "'none' opts out of the Project's default folder; empty otherwise.", "default": ""},
                    "context_requires_self_body_docs": {"type": "boolean", "description": "True for work on Ouroboros code, including copies elsewhere: this task gets the full development handbook in Max. Low/Nano and helpers keep their usual book maps.", "default": False},
                    "source": {"type": "string", "description": "Attach an existing folder or clone a git URL (https://... or git@host:path) server-side into the projects root; private auth failures are typed auth_required. Registers the folder with provenance + trusted_at as the Project/task workspace. Use for work on a supplied repo/folder.", "default": ""},
                    "predecessor_task_id": {"type": "string", "description": "Required: empty for fresh work, or one settled result id from the host manifest/recent_tasks/get_task_result. Any settled status/project; lists are hints. Name the root for a helper's result. Live roots and pending promotes refuse."},
                    "reasoning_effort": _ROOT_EFFORT_PARAM,
                },
                "required": ["objective", "predecessor_task_id"],
            },
        }, _promote_chat_to_task),
        ToolEntry("ensure_project_scope", {
            "name": "ensure_project_scope",
            "description": (
                "Create (or attach to) a named Ouroboros PROJECT and bind THE CURRENT running "
                "task to it DURABLY. Use this when you are ALREADY working a task and realize it "
                "should be a named project (the owner asked to 'create a project called X', or the "
                "work has grown into a real deliverable) — instead of a bare filesystem mkdir. "
                "Unlike promote_chat_to_task (which starts a NEW independent task in a project), "
                "this binds the task you are in: its journal_write and per-project knowledge start "
                "working, and its live progress routes to the project thread. The result states "
                "the REAL outcome the host recorded — the durable binding, a typed refusal, or "
                "unconfirmed — never a promise. Idempotent for the same project; a task already "
                "bound to a different project stays there (a requested name is carried to that "
                "project as a rename). A planning obligation stays with this task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "Display name for a NEW project (a filesystem id is derived from it). Honor the owner's stated name.", "default": ""},
                    "project_id": {"type": "string", "description": "Optional EXISTING project id (filesystem-clean) to attach to instead of creating one.", "default": ""},
                },
                "required": [],
            },
        }, _ensure_project_scope),
        ToolEntry("list_projects", {
            "name": "list_projects",
            "description": (
                "List the owner's projects (id, name, recency, running flag) — read-only. "
                "Use it in a main-chat turn to decide whether a message belongs to an existing "
                "project, then route it there with route_to_project."
            ),
            "parameters": {"type": "object", "properties": {
                "limit": {"type": "integer", "default": 50, "description": "Max projects to list."},
            }},
        }, _list_projects),
        ToolEntry("route_to_project", {
            "name": "route_to_project",
            "description": _ROUTE_TO_PROJECT_DESCRIPTION,
            "parameters": {"type": "object", "properties": {
                "project_id": {"type": "string", "default": "", "description": "Target project id (filesystem-clean; see list_projects), or empty to emit typed needs_manual_target."},
                "message": {"type": "string", "description": "The owner message / work to route into the project."},
                "reason": {"type": "string", "default": "", "description": "Optional short why-this-project note (provenance)."},
                "predecessor_task_id": {"type": "string", "description": "Required explicit selector: pass an empty string for fresh work, or the id of a settled result (any settled status; any project, the host list is a hint; a helper's result is continued with its root named) to continue it. A live root or a pending promote is refused."},
                "candidates": {"type": "array", "items": {"type": "string"}, "description": "Optional, ONLY with project_id='': the task/project ids you consider plausible, in preference order. The typed picker shows them first; ids not in the host-built option list are ignored."},
                "reasoning_effort": _ROOT_EFFORT_PARAM,
            }, "required": ["message", "predecessor_task_id"]},
        }, _route_to_project),
        ToolEntry("steer_task", {
            "name": "steer_task",
            "description": (
                "Deliver a message to any host-listed active independent root (a running or pending "
                "root task; hidden/headless roots included) — YOU pick from current_chat.addressable_root_tasks, "
                "main_routing_manifest.root_tasks, or the [INDEPENDENT_ROOTS] note. Use it when a message "
                "continues or redirects a task already in flight, instead of spawning a duplicate. Who "
                "you are decides how it lands: in an owner conversation turn it is delivered as the "
                "owner's steering text; from a task it is written as a message from THIS task (never "
                "owner text, no file attachments), and the result says written, not read. The task picks "
                "it up at its next step. If no running task clearly fits, use promote_chat_to_task "
                "(new work) or answer inline — never steer a task you are unsure about. "
                "A Presence-bound turn can steer only work in its own binding; the host checks it."
            ),
            "parameters": {"type": "object", "properties": {
                "task_id": {"type": "string", "description": "Id of the running task to steer (from current_chat.running_tasks)."},
                "message": {"type": "string", "description": "The follow-up / steering message to deliver to that task."},
            }, "required": ["task_id", "message"]},
        }, _steer_task),
        ToolEntry("schedule_subagent", {
            "name": "schedule_subagent",
            "description": _SCHEDULE_SUBAGENT_DESCRIPTION,
            "parameters": {
                "type": "object",
                # DERIVED, not restated: schedule_subagent_properties() is the single source
                # this schema and the handler's allowed-key set both read from.
                "properties": schedule_subagent_properties(),
                "required": ["subagent_id", "objective", "expected_output"],
                "additionalProperties": False,
            },
        }, _schedule_task),
        # cancel_task + peek_task + discard_child_result are registered by ouroboros/tools/join_ledger.py.
        ToolEntry("request_deep_self_review", {
            "name": "request_deep_self_review",
            "description": "Request a deep self-review of the entire Ouroboros project against the Constitution: review_change(subject=system, surface=system) with one seat — the enabled catalog row you name in `reviewer` (review pool member or not), else the Main model. An API row runs a bounded read-only inspection episode, an agent-session row reads the repository itself; both receive the core memory whitelist inline byte-exact (memory is never receipt-checked). The report goes to chat and memory/deep_review.md, with the surface=system review record linked.",
            "parameters": {"type": "object", "properties": {
                "reason": {"type": "string", "description": "Why you want a review (context for the reviewer)"},
                "reviewer": {"type": "string", "description": "One enabled catalog row by id or handle; empty = the Main model"},
            }, "required": ["reason"]},
        }, _request_deep_self_review),
        ToolEntry("chat_history", {
            "name": "chat_history",
            "description": "Retrieve live and recent archived chat messages. Supports exact provenance/date filters, substring search, and pagination.",
            "parameters": {"type": "object", "properties": {
                "count": {"type": "integer", "default": 100, "description": "Number of messages (from latest)"},
                "offset": {"type": "integer", "default": 0, "description": "Skip N from end (pagination)"},
                "search": {"type": "string", "default": "", "description": "Text filter"},
                "provider": {"type": "string", "default": "", "description": "Exact transport provider"},
                "account_id": {"type": "string", "default": "", "description": "Exact transport account ID"},
                "conversation_id": {"type": "string", "default": "", "description": "Exact transport conversation ID"},
                "thread_id": {"type": "string", "default": "", "description": "Exact transport thread ID"},
                "actor_id": {"type": "string", "default": "", "description": "Exact platform actor ID"},
                "date_from": {"type": "string", "default": "", "description": "Inclusive ISO-8601 lower timestamp bound"},
                "date_to": {"type": "string", "default": "", "description": "Inclusive ISO-8601 upper timestamp bound"},
                "snapshot": {"type": "string", "default": "", "description": "Opaque snapshot returned by the first page; reuse it with offset to refuse shifted/mixed pages"},
            }, "required": [], "additionalProperties": False},
        }, _chat_history),
        ToolEntry("update_scratchpad", {
            "name": "update_scratchpad",
            "description": "Append a block to your working memory (scratchpad). Each call adds a "
                           "timestamped block; oldest blocks are auto-evicted when either cap is reached "
                           "(10 blocks, 60000 characters of content). "
                           "Write what matters NOW — active tasks, decisions, observations. "
                           "Persists across sessions, read at every task start. "
                           "Project rooms included — the scratchpad is the same working memory in every room.",
            "parameters": {"type": "object", "properties": {
                "content": {"type": "string", "description": "Content for this scratchpad block"},
            }, "required": ["content"]},
        }, _update_scratchpad),
        ToolEntry("send_user_message", {
            "name": "send_user_message",
            "description": "Send a separate reply to the owner while work continues: the first "
                           "line of longer work (what I am about to do and why), or a mid-work "
                           "insight, a question, or an invitation to collaborate. It appears in "
                           "the conversation as a normal reply and leaves the work running; later "
                           "progress stays in the card and the final answer is delivered "
                           "automatically.",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string", "description": "Message text"},
                "reason": {"type": "string", "description": "Why you're reaching out (logged, not sent)"},
                "destination": {"type": "string", "enum": ["current", "main"], "default": "current",
                                "description": "'current' (default): this conversation's room. 'main': the "
                                               "owner's main chat, for a brief plain-text notice that belongs "
                                               "there while this work lives in an owner-visible Project room. "
                                               "It never appears in the Project thread and is not this task's "
                                               "answer. Delegated, Presence and agent-to-agent work cannot use it."},
            }, "required": ["text"]},
        }, _send_user_message),
        ToolEntry("update_identity", {
            "name": "update_identity",
            "description": "Update your identity manifest (who you are, who you want to become). "
                           "Persists across sessions. Obligation to yourself (Principle 1: Continuity). "
                           "Read your current identity first, then evolve it — add, refine, deepen. "
                           "Full rewrites are allowed but should be rare; continuity of self matters. "
                           "Use this only after substantive reflection or real experience — not on a "
                           "greeting or trivial turn. This is the only correct way to write identity; "
                           "never write memory/identity.md through write_file/edit_text. "
                           "Project rooms included — identity is the same continuous file in every room.",
            "parameters": {"type": "object", "properties": {
                "content": {"type": "string", "description": "Full identity content (prefer evolving over rewriting from scratch)"},
            }, "required": ["content"]},
        }, _update_identity),
        ToolEntry("toggle_evolution", {
            "name": "toggle_evolution",
            "description": "Enable or disable evolution mode. When enabled, Ouroboros runs continuous self-improvement cycles. Enabling requires runtime_mode 'advanced', 'pro', or 'cyber_pro'; it is refused in 'light' mode.",
            "parameters": {"type": "object", "properties": {
                "enabled": {"type": "boolean", "description": "true to enable, false to disable"},
                "objective": {"type": "string", "default": "", "description": "Optional Evolution Campaign objective when enabling."},
            }, "required": ["enabled"]},
        }, _toggle_evolution),
        ToolEntry("toggle_consciousness", {
            "name": "toggle_consciousness",
            "description": ("Control background consciousness: 'start' or 'stop' (the owner is told), or "
                            "'status' (answered to you only: the persisted state with its source, never posted to "
                            "the owner's chat)."),
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["start", "stop", "status"], "description": "Action to perform"},
            }, "required": ["action"]},
        }, _toggle_consciousness),
        ToolEntry("set_next_wakeup", {
            "name": "set_next_wakeup", "description": (
                "Choose the consciousness wake-up interval in seconds: how long after a wake-up ends the next one "
                "starts, clamped into the owner's OUROBOROS_BG_WAKEUP_MIN/MAX. A wake-up calling this sets its own "
                "next one; a wake-up already pending keeps its time; with consciousness off it is stored for later. "
                "The alarm still adjusts it: a failed wake-up doubles the interval (up to MAX), a pending event "
                "brings the next wake-up forward, a skipped one retries after MIN (an exhausted allowance waits for "
                "its reset), and no wake-up starts sooner than MIN after the last wake-up, boot or skip."),
            "parameters": {"type": "object", "properties": {"seconds": {"type": "integer", "description": "Seconds from the end of a wake-up to the next one"}}, "required": ["seconds"]},
        }, _set_next_wakeup),
        ToolEntry("switch_model", {
            "name": "switch_model",
            "description": "Switch to a different LLM model or reasoning effort level. "
                           "Use when you need more power (complex code, deep reasoning) "
                           "or want to save budget (simple tasks). Takes effect on next round. "
                           "After the host moved this turn to a configured fallback, primary='return' "
                           "or 'wait' goes back to the turn's primary route.",
            "parameters": {"type": "object", "properties": {
                "model": {"type": "string", "description": "Model name (e.g. anthropic/claude-sonnet-4). Leave empty to keep current."},
                "effort": {"type": "string", "enum": list(EFFORT_SCALE),
                           "description": "Reasoning effort level (adapted down per route when a model tops out lower). Leave empty to keep current."},
                "primary": {"type": "string", "enum": ["return", "wait"],
                            "description": ("Omit to keep the current route. Return to this turn's primary route: its model, role and account policy "
                                            "(Auto stays Auto) plus the owner's wait-card choice; effort stays. 'return': "
                                            "if the primary refuses, configured routes are tried again. 'wait': if it "
                                            "refuses or is unreachable, wait for it where this turn may wait instead of "
                                            "paid alternatives. The next real request tests it; no timer or probe does. "
                                            "Not with model.")},
            }, "required": []},
        }, _switch_model),
        get_task_result_entry(),
        ToolEntry("wait_task", {
            "name": "wait_task",
            "description": "Wait once for the named child plus an entry snapshot of your live direct children, returning on the first terminal or actionable input; all snapshot results are compact and source-linked. Use wait_tasks([id]) for an exact dependency. The result is a compact JSON tasks envelope, not the former full single-child text. Informational mail stays queued for full delivery in the same resumed request; owner/control and typed escalations always wake. Omitted timeout parks warm when supported, otherwise holds one bounded operation; explicit 0 is a snapshot. A terminal is not success.",
            "parameters": {"type": "object", "required": ["task_id"], "properties": {
                "task_id": {"type": "string", "description": "Task ID to check"},
                "known_result_sha256": {"type": "string", "description": "Optional child_result_sha256 already obtained for this task. An exact match returns unchanged without repeating result/trace; current facts remain. Omit to return full text. This does not change when the wait ends."},
                "timeout_sec": {"type": "integer", "description": f"Optional explicit wait clamped to {_WAIT_TASK_CLAMP_SEC}s; 0 is a snapshot. Omission is event-owned."},
            }},
        }, _wait_for_task, timeout_sec=_event_wait_window(None) + NESTED_SETTLEMENT_MARGIN_SEC),
        ToolEntry("wait_tasks", {
            "name": "wait_tasks",
            "description": "Wait once for an exact batch (all_terminal, or any_terminal for first completion), returning compact outcomes, hashes, accounted_upper_bound_usd/cost_final and retained full sources for every selected child. Owner/control and typed escalations wake; routine mail/progress do not. Omitted timeout parks warm when supported; explicit 0 is a snapshot. Unknown IDs are disclosed, not terminal. Read omitted detail explicitly before relying on it.",
            "parameters": {"type": "object", "required": ["task_ids"], "properties": {
                "task_ids": {"type": "array", "items": {"type": "string"}, "description": "Task IDs returned by schedule_subagent."},
                "known_result_sha256_by_task": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Optional task_id to previously obtained child_result_sha256 map. Matching children omit only result/trace and return result_unchanged plus a full-read reference. Missing or different hashes return the usual complete body/trace. Current status/cost/outcome/capability facts remain; wait timing is unchanged."},
                "timeout_sec": {"type": "integer", "description": f"Optional explicit wait clamped to {_WAIT_TASKS_CLAMP_SEC}s; 0 is a snapshot. Omission is event-owned."},
                "mode": {"type": "string", "enum": ["all_terminal", "any_terminal"], "default": "all_terminal"},
            }},
        }, _wait_for_tasks, timeout_sec=_event_wait_window(None) + NESTED_SETTLEMENT_MARGIN_SEC),
        await_messages_entry(),
    ]


# v7next F1 (D08): moved spans live in their owner leaves; re-exported here
# so this facade stays the single import surface for callers and tests.
from ouroboros.tools.control_events import (  # noqa: E402, F401 -- intentional public re-exports
    _PROMOTE_CONFIRM_POLL_SEC,
    _PROMOTE_CONFIRM_TIMEOUT_SEC,
    _SCHEDULE_EMIT_LOCK,
    _emit_and_wait_for_routing,
    _emit_control_event,
    _promotion_pool_disabled_from_snapshot,
    _routing_status_root,
    _wait_for_promotion_admission,
    _wait_for_routing_annotation,
)
from ouroboros.tools.control_routing import (  # noqa: E402, F401 -- intentional public re-exports
    _MISSING_PREDECESSOR_SELECTOR,
    _attach_client_surface,
    _attach_origin_from_metadata,
    _attach_predecessor_authority_from_metadata,
    _finish_swarm_handoff,
    _list_projects,
    _predecessor_selector_error,
    _promote_chat_to_task,
    _route_to_project,
    _steer_task,
)
from ouroboros.tools.control_runtime import (  # noqa: E402, F401 -- intentional public re-exports
    _chat_history,
    _evolution_restart_block_reason,
    _finish_task,
    _prepare_self_change,
    _promote_to_stable,
    _request_deep_self_review,
    _request_restart,
    _send_user_message,
    _set_next_wakeup,
    _set_tool_timeout,
    _switch_model,
    _toggle_consciousness,
    _toggle_evolution,
    _update_identity,
    _update_scratchpad,
    self_change_tool_entries,
)
from ouroboros.tools.control_scheduling import (  # noqa: E402, F401 -- intentional public re-exports
    HIDDEN_LEGACY_SCHEDULE_PARAMS,
    _build_acting_constraint,
    _build_child_subagent_contract,
    _capability_mismatch_message,
    _context_task_depth,
    _earliest_deadline_at,
    _emit_swarm_fanout,
    _finalize_schedule_emission,
    _inherited_workspace_from_active_repo,
    _materialize_child_attachment_manifest,
    _populate_subagent_event_extras,
    _prepare_child_drive,
    _record_scheduled_subagent,
    _resolve_executor_ref,
    _schedule_task,
    _select_subagent_constraint,
    _subagent_slot_note,
    maybe_emit_delegated_run_fanout,
)

# v7next F2 (D07): moved spans live in their owner leaves; re-exported here
# so this facade stays the single import surface for callers and tests.
from ouroboros.tools.control_subagent_spec import (  # noqa: E402, F401 -- intentional public re-exports
    _INTERNAL_SCHEDULE_OPTIONS,
    RETIRED_SCHEDULE_PARAMS,
    VALID_SUBTASK_MEMORY_MODES,
    _validated_schedule_fields,
    schedule_subagent_param_names,
    schedule_subagent_properties,
)
from ouroboros.tools.control_task_results import (  # noqa: E402, F401 -- intentional public re-exports
    _UNMINTED_WAIT_GRACE_SEC,
    _WAIT_TASK_CLAMP_SEC,
    _WAIT_TASKS_CLAMP_SEC,
    NESTED_SETTLEMENT_MARGIN_SEC,
    _await_messages,
    _children_roster_projection,
    _count_live_sibling_children,
    _event_wait_window,
    _get_task_result,
    _subtask_outcome_summary,
    _unminted_wait_ids,
    _wait_attention_poll,
    _wait_for_task,
    _wait_for_tasks,
    await_messages_entry,
    cache_horizon_note,
    disclosable_capability_delta,
    get_task_result_entry,
)

# The D23 hidden-params handler attribute is stamped AFTER the re-exports bind
# _schedule_task and HIDDEN_LEGACY_SCHEDULE_PARAMS above (the moved spans own
# the definitions; the attribute consumer reads it off the registered handler).
setattr(_schedule_task, "_hidden_legacy_params", HIDDEN_LEGACY_SCHEDULE_PARAMS)
