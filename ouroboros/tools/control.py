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
    apply_settings_to_env,  # noqa: F401
    get_max_subagent_depth,  # noqa: F401
    load_settings,  # noqa: F401
    save_settings,  # noqa: F401
)
from ouroboros.depth_evidence import parse_task_depth  # noqa: F401
from ouroboros.headless import prepare_task_drive, task_state_dir  # noqa: F401
from ouroboros.contracts.task_contract import (
    build_task_contract,  # noqa: F401
    effective_acceptance_claims,  # noqa: F401
    normalize_allowed_resources,  # noqa: F401
)
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
from ouroboros.tools.registry import active_repo_dir_for, system_repo_dir_for  # noqa: F401
from ouroboros.outcomes import normalize_outcome_axes  # noqa: F401
from ouroboros.task_results import (
    STATUS_COMPLETED,  # noqa: F401
    STATUS_REJECTED_DUPLICATE,  # noqa: F401
    STATUS_REQUESTED,  # noqa: F401
    validate_task_id,  # noqa: F401
    write_task_result,  # noqa: F401
)
from ouroboros.task_status import load_effective_task_result, wait_for_effective_tasks  # noqa: F401
from ouroboros.subagents import (
    LEGACY_SUBAGENT_FIELDS,  # noqa: F401
    build_subagent_envelope,  # noqa: F401
)
from ouroboros.subagent_runtime import (
    SubagentSelectionError,  # noqa: F401
    effective_runtime_subagent_settings,  # noqa: F401
    select_subagent_snapshot,  # noqa: F401
)
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE, LOCAL_READONLY_SUBAGENT_MODE  # noqa: F401
from ouroboros.tool_policy import swarm_router_turn  # noqa: F401
from ouroboros.tools.registry import ToolContext, ToolEntry  # noqa: F401
from ouroboros.utils import append_jsonl, atomic_write_json, truncate_review_artifact, utc_now_iso, run_cmd  # noqa: F401

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
    "Promote real work out of this conversation into a supervised pooled task "
    "while the conversation remains available. Use it "
    "whenever a chat request needs tools/files/multi-step work rather than a "
    "conversational answer. Before framing the objective around an EXISTING artifact "
    "('check/fix/extend the X skill/file'), ground-truth its existence with one cheap probe "
    "first (skills: list_skills; files: list_files) — memory of past work is not evidence "
    "the referent still exists. Always give a short, human-readable task `title`. To "
    "CREATE A NEW NAMED PROJECT and do the work there (owner asked to 'create a "
    "project called X and …'), set `project_name` — the project is created now "
    "and this task runs inside it (my own judgment: the owner's phrasing is intent, "
    "not a keyword trigger — I name the project from what they actually want it "
    "called, and do not just answer or spawn a project-less task). `project_id` "
    "scopes to an existing project. When this new task continues one specific "
    "completed result shown by the host (the Main manifest or Project last-result "
    "preview), pass its internal id as `predecessor_task_id`; pass an empty string for fresh work. "
    "`workspace_root` points at a working folder. A project-scoped task inherits "
    "the project's working folder as its ACTIVE WORKSPACE by default (its file/"
    "shell/git tools operate there, not on the Ouroboros repo); pass "
    "workspace='none' for a folder-less task. Owner follow-ups can steer the "
    "running task. Report creation only when this tool returns "
    "OK; PROMOTE_REJECTED or PROMOTE_UNCONFIRMED means the task must not be "
    "claimed as created, and UNCONFIRMED must not be retried automatically."
)


def get_tools() -> List[ToolEntry]:
    from ouroboros.config import EFFORT_SCALE

    return [
        ToolEntry("set_tool_timeout", {
            "name": "set_tool_timeout",
            "description": "Update the global tool timeout in settings.json and apply it immediately without restart.",
            "parameters": {"type": "object", "properties": {
                "seconds": {"type": "integer", "description": "New timeout in seconds (>= 1)"},
            }, "required": ["seconds"]},
        }, _set_tool_timeout),
        ToolEntry("request_restart", {
            "name": "request_restart",
            "description": "Ask supervisor to restart runtime after a reviewed local commit or a non-evolution clean no-op; evolution requires its exact active commit receipt.",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
        }, _request_restart),
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
                    "title": {"type": "string", "description": "A short human-readable task name (<=80 chars, e.g. 'Tic-tac-toe game'). Reused as the project name if the owner later turns the task into a project — so coin a clean, concise one.", "default": ""},
                    "project_name": {"type": "string", "description": "Set ONLY to create a brand-new NAMED project now and run this task inside it (e.g. 'airi research'). The display name; a filesystem id is derived from it.", "default": ""},
                    "expected_output": {"type": "string", "description": "What done looks like.", "default": ""},
                    "project_id": {"type": "string", "description": "Optional EXISTING project scope (filesystem-clean id).", "default": ""},
                    "workspace_root": {"type": "string", "description": "Optional absolute working-folder path (validated at admission: must be a git worktree root outside the Ouroboros repo/data). When omitted for a project-scoped task, the project's registered working_dir is used by default.", "default": ""},
                    "workspace": {"type": "string", "description": "Pass 'none' to opt OUT of the project room's default working folder (a folder-less task in a folder-ful project). Leave empty otherwise.", "default": ""},
                    "source": {"type": "string", "description": "Attach or clone the project's working folder in ONE move: a git URL (https://... or git@host:path — cloned server-side into the projects root; private repos fail typed auth_required) or an existing folder path (validated attach). The folder is registered on the project (provenance + trusted_at) and becomes this task's active workspace. Use for 'help me debug this GitHub repo / this folder' asks.", "default": ""},
                    "predecessor_task_id": {"type": "string", "description": "Required explicit selector: pass an empty string for fresh work, or the completed result id shown by the host routing manifest to continue it."},
                },
                "required": ["objective", "predecessor_task_id"],
            },
        }, _promote_chat_to_task),
        ToolEntry("ensure_project_scope", {
            "name": "ensure_project_scope",
            "description": (
                "Create (or attach to) a named Ouroboros PROJECT and scope THE CURRENT running "
                "task into it. Use this when you are ALREADY working a task and realize it should "
                "be a named project (the owner asked to 'create a project called X', or the work "
                "has grown into a real deliverable) — instead of a bare filesystem mkdir. Unlike "
                "promote_chat_to_task (which creates a NEW task in a project), this binds the task "
                "you are in: its journal_write and per-project knowledge start working, and its "
                "live progress routes to the project thread. Idempotent for the same project; it "
                "will NOT re-scope a task that already belongs to a different project."
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
            "description": (
                "Route a main-chat message to an EXISTING project so the work continues in that "
                "project's own context (memory/journal/thread), keeping the main chat free. Use "
                "when a message clearly belongs to a known project (call list_projects first if "
                "unsure of the id). If confidence is low or several projects/tasks could match, "
                "CALL THIS TOOL with project_id='' and the owner's message: it emits the typed "
                "needs_manual_target acknowledgement with host-validated task options and New task "
                "in Project; prose alone cannot emit that typed choice. For brand-new work that is not yet a project, "
                "use promote_chat_to_task instead. When continuing one completed result from the "
                "Main host manifest, pass its internal `predecessor_task_id`; pass an empty string for fresh work. "
                "Returns a visible routing receipt."
            ),
            "parameters": {"type": "object", "properties": {
                "project_id": {"type": "string", "default": "", "description": "Target project id (filesystem-clean; see list_projects), or empty to emit typed needs_manual_target."},
                "message": {"type": "string", "description": "The owner message / work to route into the project."},
                "reason": {"type": "string", "default": "", "description": "Optional short why-this-project note (provenance)."},
                "predecessor_task_id": {"type": "string", "description": "Required explicit selector: pass an empty string for fresh work, or the completed result id listed by the Main host manifest to continue it."},
                "candidates": {"type": "array", "items": {"type": "string"}, "description": "Optional, ONLY with project_id='': the task/project ids you consider plausible, in preference order. The typed picker shows them first; ids not in the host-built option list are ignored."},
            }, "required": ["message", "predecessor_task_id"]},
        }, _route_to_project),
        ToolEntry("steer_task", {
            "name": "steer_task",
            "description": (
                "Deliver a follow-up/steering message to a host-listed RUNNING/PENDING owner root — YOU "
                "pick from current_chat.addressable_root_tasks in a Project room, or from "
                "main_routing_manifest.root_tasks in Main (including Project-bound roots). Use it when a message continues or redirects a task already "
                "in flight, instead of spawning a duplicate. The message reaches that task's mailbox and "
                "it picks it up at its next step. If no running task clearly fits, use promote_chat_to_task "
                "(new work) or answer inline — never steer a task you are unsure about."
            ),
            "parameters": {"type": "object", "properties": {
                "task_id": {"type": "string", "description": "Id of the running task to steer (from current_chat.running_tasks)."},
                "message": {"type": "string", "description": "The follow-up / steering message to deliver to that task."},
            }, "required": ["task_id", "message"]},
        }, _steer_task),
        ToolEntry("schedule_subagent", {
            "name": "schedule_subagent",
            "description": (
                "Schedule a live subagent (a child of Ouroboros). Returns task_id for later retrieval. "
                "DEFAULT is READ-ONLY: the child inspects local repo/data/history plus web/browser and "
                "returns findings (it cannot write local state, commit, enable tools, or run "
                "shell/review/runtime/skills). Set write_surface to spawn a MUTATIVE (acting) child that "
                "writes on the selected surface. You remain the sole committer of the live Ouroboros body. "
                "self_worktree is an isolated git worktree of THIS repo: apply its workspace.patch with "
                "integrate_subagent_patch for parallel self-modification / best-of-N. Native children on "
                "external_workspace write directly to the SHARED external project directory (write_root or "
                "the parent workspace); integrate_subagent_patch verifies the files already there without reapplying. "
                "genesis (a from-scratch new project — game/site/app/new Ouroboros — auto-provisioned as a fresh "
                "empty git repo under the durable projects root; the project directory IS the deliverable, not "
                "integrated into this repo). "
                "An installed skill payload under data/ is NOT a write_surface (runtime data is never one, by "
                "design): mutate it YOURSELF via delegate_start(subagent_id=..., prompt=..., root='skill_payload', bucket=..., skill_name=...) "
                "— a child cannot open a payload delegation — and schedule children only as read-only "
                "designers/reviewers for that work. "
                "COOPERATIVE MULTI-BUILDER vs GENESIS: when SEVERAL builder children must contribute to ONE new "
                "deliverable together, give each write_surface=external_workspace and OMIT write_root — the host "
                "mints ONE shared git tree the whole subagent tree writes into cooperatively (deeper descendants "
                "inherit it), and you verify their combined files with integrate_subagent_patch. Use genesis only when EACH child "
                "should own its OWN standalone durable repo (e.g. best-of-N separate builds). "
                "Harness-delegated work uses a private snapshot; integrate_delegated_patch handles that separate patch. "
                "Mutative children still cannot commit, run "
                "review/runtime/skills lifecycle, enable tools, or write cognitive memory. Nested delegation "
                "is allowed within configured depth/cap limits — use delegation_intent / may_mutate / "
                "may_fan_out to tell a child to recurse further, so a 'maximum subagents / grandchildren' "
                "request propagates structurally instead of collapsing into one flat layer. "
                "BURST + ABSORB: when several children are INDEPENDENT, emit them in ONE batch (parallel "
                "schedule_subagent calls in the same round) so they run concurrently, then absorb with "
                "wait_tasks(any_terminal) — handling whichever finishes first — instead of scheduling and "
                "blocking on them one at a time with serial wait_task calls — on cache-write-priced "
                "routes each sibling launched before the first sibling's first response pays its own full "
                "prefix write, so burst buys latency and spacing buys cash; your call. "
                "INDEPENDENT VERIFIER: to check a finished deliverable without builder bias, spawn a "
                "read-only child with memory_mode=empty whose objective carries ONLY the deliverable "
                "location + the task's acceptance criteria (NOT your own probes/assumptions) and have it "
                "verify through the task's own interface. Always retrieve "
                "the handoff with get_task_result, wait_task, or wait_tasks before relying on its results."
            ),
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
            "description": "Request a deep self-review of the entire Ouroboros project against the Constitution, on the configured deep-review reviewer row (Settings → Agents → Review lanes; absent, the OUROBOROS_MODEL_DEEP_SELF_REVIEW model runs one packed Atlas review): a packed API model reads the Atlas plus the full core memory whitelist; a configured subagent or agent session reads the repository itself with read-only tools and receives the same memory whitelist inline byte-exact (memory is never receipt-checked). Results go to chat and memory.",
            "parameters": {"type": "object", "properties": {
                "reason": {"type": "string", "description": "Why you want a review (context for the reviewer)"},
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
                           "timestamped block; oldest blocks are auto-evicted when the cap (10) is reached. "
                           "Write what matters NOW — active tasks, decisions, observations. "
                           "Persists across sessions, read at every task start. "
                           "No-op on a project-scoped task (no per-project scratchpad); use knowledge_write for project facts.",
            "parameters": {"type": "object", "properties": {
                "content": {"type": "string", "description": "Content for this scratchpad block"},
            }, "required": ["content"]},
        }, _update_scratchpad),
        ToolEntry("send_user_message", {
            "name": "send_user_message",
            "description": "Send a proactive message to the user. Use when you have something "
                           "genuinely worth saying — an insight, a question, or an invitation to collaborate. "
                           "This is NOT for task responses (those go automatically).",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string", "description": "Message text"},
                "reason": {"type": "string", "description": "Why you're reaching out (logged, not sent)"},
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
                           "No-op on a project-scoped task (identity is global and continuous, never per-project).",
            "parameters": {"type": "object", "properties": {
                "content": {"type": "string", "description": "Full identity content (prefer evolving over rewriting from scratch)"},
            }, "required": ["content"]},
        }, _update_identity),
        ToolEntry("toggle_evolution", {
            "name": "toggle_evolution",
            "description": "Enable or disable evolution mode. When enabled, Ouroboros runs continuous self-improvement cycles. Enabling requires runtime_mode 'advanced' or 'pro'; it is refused in 'light' mode.",
            "parameters": {"type": "object", "properties": {
                "enabled": {"type": "boolean", "description": "true to enable, false to disable"},
                "objective": {"type": "string", "default": "", "description": "Optional Evolution Campaign objective when enabling."},
            }, "required": ["enabled"]},
        }, _toggle_evolution),
        ToolEntry("toggle_consciousness", {
            "name": "toggle_consciousness",
            "description": "Control background consciousness: 'start', 'stop', or 'status'.",
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["start", "stop", "status"], "description": "Action to perform"},
            }, "required": ["action"]},
        }, _toggle_consciousness),
        ToolEntry("switch_model", {
            "name": "switch_model",
            "description": "Switch to a different LLM model or reasoning effort level. "
                           "Use when you need more power (complex code, deep reasoning) "
                           "or want to save budget (simple tasks). Takes effect on next round.",
            "parameters": {"type": "object", "properties": {
                "model": {"type": "string", "description": "Model name (e.g. anthropic/claude-sonnet-4). Leave empty to keep current."},
                "effort": {"type": "string", "enum": list(EFFORT_SCALE),
                           "description": "Reasoning effort level (adapted down per route when a model tops out lower). Leave empty to keep current."},
            }, "required": []},
        }, _switch_model),
        ToolEntry("get_task_result", {
            "name": "get_task_result",
            "description": "Read the effective result or exact authority of a task, including one bounded canonical work-order source range when requested.",
            "parameters": {"type": "object", "required": ["task_id"], "properties": {
                "task_id": {"type": "string", "description": "Task ID returned by scheduling or exposed by the host routing manifest."},
                "include_authority": {"type": "boolean", "default": False,
                                      "description": "Return the exact selected result, task contract, origin, artifact references, and current plan-review authority."},
                "include_work_order_source": {"type": "boolean", "default": False,
                                               "description": "Return the canonical work-order source projection; provide both source_start_char and source_end_char for the exact bounded range."},
                "include_completion_source": {"type": "boolean", "default": False,
                                              "description": "Read the full stored completion observations for this task, including returns omitted from the summary. Omit bounds for source length/hash, then request explicit character ranges."},
                "source_start_char": {"type": "integer", "description": "Inclusive character offset for the requested canonical source range."},
                "source_end_char": {"type": "integer", "description": "Exclusive character offset for the requested canonical source range."},
            }},
        }, _get_task_result),
        ToolEntry("wait_task", {
            "name": "wait_task",
            "description": "Wait for ONE subtask to reach a terminal status and return its effective result. May return EARLY (before terminal) if the child raises a tree_note blocker/question/interface_contract/review_requested/delegation_constraint beacon — the result then carries a [CHILD_BEACONS] block so you can steer, review, or override it. An unread message in your own mailbox also returns early so the ordinary loop can deliver and acknowledge it; the child keeps running. With SEVERAL children in flight, prefer wait_tasks(any_terminal) to absorb whichever finishes first rather than blocking serially on one id at a time.",
            "parameters": {"type": "object", "required": ["task_id"], "properties": {
                "task_id": {"type": "string", "description": "Task ID to check"},
                "timeout_sec": {"type": "integer", "default": 180, "description": "Maximum seconds to wait (default 180)."},
            }},
        }, _wait_for_task, timeout_sec=7200),
        ToolEntry("wait_tasks", {
            "name": "wait_tasks",
            "description": "Wait for MULTIPLE subtasks at once and return a compact structural projection per child (task_id, status, accounted_upper_bound_usd, cost_final, child_result_sha256, outcome_axes, result, trace_summary, capability_delta when the child has something to disclose, duplicate_of) — the right tool to ABSORB a batch of independent children you scheduled in one burst. The full per-child envelope stays on disk in task_results/<task_id>.json (child_result_sha256 pins the exact result you saw; get_task_result returns the full result text plus trace/outcome summaries). With mode=any_terminal it returns as soon as the FIRST child finishes (handle it, then call again for the rest) instead of blocking serially. The JSON also includes live_child_status (running/scheduled/terminal per child) and may early_return (before all terminal) on a child tree_note blocker/question/interface_contract/review_requested/delegation_constraint beacon so you can steer, review, or override mid-flight, or on an unread message in your own mailbox (reason=owner_mailbox_pending); the ordinary loop then handles delivery and acknowledgement. An id no surface of this tree ever minted (no task result, no queue row, no tree-ledger row) is flagged unknown_task_id — 'not yet registered or never scheduled' — and unknown_task_ids + a compact children_roster of your ACTUAL direct children are attached so you can repair the wait set instead of re-polling phantoms.",
            "parameters": {"type": "object", "required": ["task_ids"], "properties": {
                "task_ids": {"type": "array", "items": {"type": "string"}, "description": "Task IDs returned by schedule_subagent."},
                "timeout_sec": {"type": "integer", "default": 600, "description": "Maximum seconds to wait (default 600)."},
                "mode": {"type": "string", "enum": ["all_terminal", "any_terminal"], "default": "all_terminal"},
            }},
        }, _wait_for_tasks, timeout_sec=7200),
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
    _attach_swarm_intent,
    _cached_swarm_handoff,
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
    _promote_to_stable,
    _request_deep_self_review,
    _request_restart,
    _send_user_message,
    _set_tool_timeout,
    _switch_model,
    _toggle_consciousness,
    _toggle_evolution,
    _update_identity,
    _update_scratchpad,
)


# v7next F2 (D07): moved spans live in their owner leaves; re-exported here
# so this facade stays the single import surface for callers and tests.
from ouroboros.tools.control_subagent_spec import (  # noqa: E402, F401 -- intentional public re-exports
    RETIRED_SCHEDULE_PARAMS,
    VALID_SUBTASK_MEMORY_MODES,
    _INTERNAL_SCHEDULE_OPTIONS,
    _validated_schedule_fields,
    schedule_subagent_param_names,
    schedule_subagent_properties,
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
from ouroboros.tools.control_task_results import (  # noqa: E402, F401 -- intentional public re-exports
    _UNMINTED_WAIT_GRACE_SEC,
    _children_roster_projection,
    _count_live_sibling_children,
    _get_task_result,
    _subtask_outcome_summary,
    _unminted_wait_ids,
    _wait_attention_poll,
    _wait_for_task,
    _wait_for_tasks,
    cache_horizon_note,
    disclosable_capability_delta,
)

# The D23 hidden-params handler attribute is stamped AFTER the re-exports bind
# _schedule_task and HIDDEN_LEGACY_SCHEDULE_PARAMS above (the moved spans own
# the definitions; the attribute consumer reads it off the registered handler).
setattr(_schedule_task, "_hidden_legacy_params", HIDDEN_LEGACY_SCHEDULE_PARAMS)
