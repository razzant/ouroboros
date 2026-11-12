"""Control tools: restart, timeout settings, scheduling, review, chat history, model switching, interface wording."""

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
from ouroboros.depth_evidence import parse_task_depth  # noqa: F401
from ouroboros.headless import prepare_task_drive, task_state_dir  # noqa: F401
from ouroboros.contracts.task_contract import (
    build_task_contract,  # noqa: F401
    effective_acceptance_claims,  # noqa: F401
    normalize_allowed_resources,  # noqa: F401
)
from ouroboros.tools.control_delegation import (
    _move_to_project,
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
from ouroboros.tools.registry import ToolContext, ToolEntry  # noqa: F401
from ouroboros.utils import append_jsonl, atomic_write_json, truncate_review_artifact, utc_now_iso, run_cmd  # noqa: F401

log = logging.getLogger(__name__)

GIST_RULE = (
    "On a root task or self-started run, `summary` is my gist of the run for my human and my later self, "
    "in my human's language; in a tight window it stands for this work's replies. "
    "Its first sentence says what the run did, found or decided: a clean finish shows it on the card, "
    "room row and portal, and that sentence represents this work in my other conversations. "
    "Further sentences, when the work needs them, keep what a later reader needs without the rows: "
    "numbers, paths, decisions, promises, what remains and whose step it is."
)


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


# A cite: my human's exact words, copied from a message of theirs that reached this run. The host verifies only
# provenance (their confirmed account, not forwarded, the bytes inside that message); whether the words ask for
# this work is the citing mind's reading, shown on the card. Cited work is my human's (their allowance, not stopped
# by the consciousness switch); uncited work from a wake is my own.
_OWNER_WORDS_PARAM = (
    "Optional: the exact words of your human asking for this work, copied verbatim from one message of theirs that "
    "reached this run (their confirmed account; a forward or a colleague's line is not theirs). Verified by bytes: the "
    "work then runs as their order; a quote not found is refused and nothing starts. Omitted in a place turn or a wake, "
    "the work is your own; in Main the work is already your human's.")
_START_TASK_DESCRIPTION = (
    "Start a NEW, separate independent work with its own task id and card. "
    "The work reading the current request already exists and can do that request itself. "
    "move_to_project places THIS work in a room with the same id. "
    "For an EXISTING artifact, ground-truth its existence with one cheap probe "
    "(list_skills/list_files): memory of past work is not evidence it still exists. "
    "Give the separate work a short title and objective. project_name creates a named room; "
    "project_id selects an existing room. workspace_root selects its folder; "
    "workspace='none' opts out of the room's default folder. "
    "From a Presence conversation or work acting for a place, name where each new work belongs with destination; "
    "without one nothing starts. Elsewhere omit it. "
    "Set context_requires_development=true only when the task itself needs the whole development handbook in every "
    "mode, never for its subject; a route too small for it refuses the task. "
    "A durable pending intent is recorded but not yet confirmed; inspect its recorded "
    "outcome before explicitly retrying."
)


_MESSAGE_DESTINATION_DESCRIPTION = (
    "'current' (default): this work's room. 'main': also the "
    "owner's main chat while this work lives in an owner-visible Project "
    "room, for an expected result, a decision or a substantial change to "
    "what the owner relies on (such as lost promised verification or "
    "unique data, risk to shared work, a correction of a reported "
    "result), even when a workaround means they need not act. One "
    "message: the Project shows it too, and it is read once; it is not "
    "this task's answer. A Presence root reports privately, with host-bound "
    "source attribution, including from Silent places: Main keeps its "
    "address, the Project its work lives in (if any) shows the same row, "
    "and its place never sees it. "
    "Delegated and agent-to-agent work cannot use it."
)


_ROOT_EFFORT_PARAM = {"type": "string", "enum": list(EFFORT_SCALE), "description": (
    "Optional: the reasoning effort the NEW task starts on; applied in Cyber Pro only — otherwise the "
    "task starts at my human's recommended level and the result says so. Omit for that level.")}
_START_CHOICE_PARAMS = {
    "reasoning_effort": _ROOT_EFFORT_PARAM,
    "destination": {"type": "string", "enum": ["place", "owner"], "description": (
        "Presence callers only. 'place': this place's work, which may answer its conversation. 'owner': the owner's own "
        "private work, asked for by their answer to a private question; it runs as ordinary owner work in Main or a "
        "Project chosen here without an automatic place reply; the place can still read and stop it. That root schedules "
        "its own follow-ups.")},
    "owner_quote": {"type": "string", "default": "", "description": (
        "With destination='owner': the held owner_quote_handle of the answer that asks for this work.")},
}


# The direct-OpenAI custom projection drops parameter prose and keeps this lead only as far as it fits
# beside the compact schema; tests/test_openai_chat_custom_contract.py holds every actual profile whole.
_SCHEDULE_SUBAGENT_DESCRIPTION = (
    # SYSTEM also says start_task is separate work, an evidence-only copy owes no apply, and shared-tree work is
    # verified, not reapplied (integrate_subagent_patch's lead too): this lead pays the typed input_files schema.
    "Start a child on one objective; it reports back and returns task_id. Children may write knowledge "
    "notes, memory marks and chronicle drafts in their own name (the integrating mind accepts or rejects drafts: "
    "chronicle_write kind=decision); no child commits, grants itself tools or writes identity/scratchpad. Default "
    "native READ-ONLY children inspect local repo/data/history and web/browser and return findings; no other local "
    "writes or shell/review/runtime/skills. workspace_root: start folder, else yours. write_surface "
    "picks a MUTATIVE (acting) child: self_worktree isolates its Git source's current eligible files (uncommitted "
    "too) for integrate_subagent_patch; native external_workspace writes in the SHARED directory (write_root, else "
    "the selected/inherited workspace_root, else one new Git tree for such descendants); genesis gives EACH an empty "
    "Git repo in the durable projects root that IS the deliverable; acting harness work uses a private snapshot and "
    "integrate_delegated_patch. "
    # Custom-tool projection strips parameter prose; this choice must survive in the tool lead.
    "A non-acting Agent session keeps configured access in an evidence-only private checking copy; "
    "access='readonly' selects read-only session execution. Runtime data (installed skill payloads too) is "
    "never a write_surface. Cyber-effective children inherit "
    "selected review/skill/runtime tools within explicit task restrictions. Nesting obeys configured "
    "depth/cap limits. Children scheduled in one round run concurrently; on cache-write-priced routes each "
    "sibling started before the first responds pays a full prefix write. Native children can address you, "
    "siblings or any task in their tree; ongoing addressed turns need "
    "objective/constraints stating what is interim, whom to address and what ends participation. Continue "
    "settled sessions via delegate_start(subagent_id=id, continue_from=run_id, prompt=''): new run, "
    "same session if possible, else a new one with retained evidence; work may repeat. "
    "Retrieve the handoff with get_task_result, "
    "wait_task or wait_tasks before relying on it."
)


def _project_tool_entries() -> List[ToolEntry]:
    """Reversible project decisions and their paginated reader, beside work placement."""
    return [
        ToolEntry("close_project", {
            "name": "close_project",
            "description": "Close a finished project with a note, or reopen it. Closing makes the note the room's last page over its open rows, from what I hold (name what I did not re-read); its room and history stay; a new owner message or task reopens it.",
            "parameters": {"type": "object", "properties": {
                "project_id": {"type": "string"},
                "note": {"type": "string", "description": "Why I close or reopen this project; closing: also the room's page, so what it did, found and left open."},
                "reopen": {"type": "boolean", "default": False},
            }, "required": ["project_id", "note"], "additionalProperties": False},
        }, _close_project, safety_policy="skip"),
        ToolEntry("list_projects", {
            "name": "list_projects",
            "description": (
                "List projects with id, name, last_active_at, closed record and activity — read-only. "
                "Follow the next-page call to enumerate every match. "
                "Use it in a main-chat turn to decide whether a message belongs to an existing "
                "project, then place this work there with move_to_project."
            ),
            "parameters": {"type": "object", "properties": {
                "limit": {"type": "integer", "default": 50, "description": "Max projects to list."},
                "start": {"type": "integer", "minimum": 0, "default": 0, "description": "Offset in the filtered, sorted list; use the returned next-page call."},
                "closed": {"type": "boolean", "description": "False: open projects; true: closed projects; omit for both."},
                "oldest_first": {"type": "boolean", "default": False, "description": "List by last_active_at from oldest to newest."},
            }},
        }, _list_projects),
    ]


def _sibling_tool_entries() -> List[ToolEntry]:
    """The control tools homed in sibling modules: harness maintenance, the mind's Storage -> Free, the
    mind's lifecycle choices, UI translation and update descriptions."""
    from ouroboros.tools.control_maintenance import maintenance_tool_entries
    from ouroboros.tools.control_mind import mind_tool_entries
    from ouroboros.tools.control_storage import storage_tool_entries
    from ouroboros.tools.control_ui_translation import ui_translation_tool_entries
    from ouroboros.tools.control_update_description import update_description_tool_entries

    return [*maintenance_tool_entries(), *storage_tool_entries(), *mind_tool_entries(), *ui_translation_tool_entries(),
            *update_description_tool_entries()]


def get_tools() -> List[ToolEntry]:
    from ouroboros.native_task_wait import SLEEP_PARAMS
    from ouroboros.subagent_inspection import inspect_subagent_tool

    return [
        *_sibling_tool_entries(),
        ToolEntry("inspect_subagent", {
            "name": "inspect_subagent",
            "description": "Inspect one configured actor from this task's captured catalog: exact route, model, account pin and access, with dated passive engine quota/resource facts. Works without Quotas. Never starts or refreshes an engine/provider, changes accounts, or chooses an executor. Missing/stale facts and unhonored model/request selectors remain unknown; availability is not an execution guarantee.",
            "parameters": {"type": "object", "properties": {
                "subagent_id": {"type": "string", "description": "Exact configured actor id or handle."},
                "request_kind": {"type": "string", "enum": ["text", "image"], "default": "text"},
            }, "required": ["subagent_id"], "additionalProperties": False},
        }, inspect_subagent_tool),
        ToolEntry("finish_task", {"name": "finish_task",
            "description": GIST_RULE + "\n\n" + "Select the complete answer after considering the work. Use finish when nothing of this work is left for you. If you end while a step of it is still yours (a deadline, a budget or a round limit can end a turn first), use stop, with a rationale naming what remains and why you end. "
                "A full window is not such a limit: after a real size refusal the host can renew this same task's view from its exact record, keeping its custody and budget; an irreducible or non-shrinking view, or an excess a directly required handbook alone causes, can still refuse. "
                "A root's stop records unfinished work for explicit Continue under the same limits (a reached deadline or money limit refuses it): your human's Continue or a later focus's continue_task starts a new root that reads this one, with a root's work room, also when this is a chat turn; continue_in_background keeps this same turn and its work room. Continue does not reopen an ordinary finish or a final reply without this tool, even when its text calls it partial. "
                "Ending closes this task's browser, task-session connections and services not kept alive, and any link they serve; when your work can go on only after your human's act or choice, a live root asks with escalate (wait_for_answer=true) before ending. "
                "Select exactly one of answer or a host-offered answer_sha256. Successful page, note, mark and addressed-message writes may ride in the same response; any other result is read before the finish. On a self-started run (a consciousness wake's own turn, or a schedule-born root without a Presence binding) or a root answering its place, silent=true sends no message (any answer stays your internal note), refused while your human awaits a reply. No answer is needed. The run records no message unless a host notice is owed. This grants no success, cancels no children, and retains all configured review and owner controls.",
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["finish", "stop"]}, "answer": {"type": "string", "description": "The complete answer, including a short correction; a child's parent reads it as your report, a root's goes where that root answers (its owner's chat, or the conversation a Presence root answers, as your context names)."},
                "answer_sha256": {"type": "string", "description": "Exact offered retained or whole held response hash."},
                "rationale": {"type": "string", "description": "For stop, what remains unfinished, whose step it is and why you end now; for a finish with pending_review=finish (required there) or over reviewers' findings, why you accept it now. The owner reads it on the task row."}, "acceptance_subject": {"type": "object", "properties": {"owner_source_sha256": {"type": "string"},
                    "effective_criteria": {"type": "string"}, "material_tool_indices": {"type": "array", "items": {"type": "integer"}},
                }, "required": ["owner_source_sha256"]},
                "pending_review": {"type": "string", "enum": ["wait", "finish"], "description": "For a finish while your result's review still runs: wait (the default) for its verdict, or, where offered, finish now with a rationale saying why; the result then shows Done with warnings and the reviewers' late verdict is kept for you. A stop waits for nothing and this has no effect on it: to wait for a review of yours that still runs, do not end; collect it with its review_reference, a $0 call that waits for seats still running in this process; after a restart it returns what is recorded, possibly still pending."},
                "swarm_comparison": {"type": "string", "description": "Optional solo-versus-swarm comparison in your own words, attributed to you separately from measured tree facts."},
                "silent": {"type": "boolean", "default": False, "description": "Finish without a message where allowed above and no human reply is owed; any answer stays internal (wake_silent)."},
                "summary": {"type": "string", "default": "", "description": "The gist, as this tool's description says; a held cause takes precedence. An unnamed self-started run, including a schedule-born root without a Presence binding, may take its first sentence as its title; an authored title and an ordinary task's name stay. It also appears in the schedule's last-run line or Consciousness row; the next scheduled run or wake reads it, and my other foci read it beside this room until a page seals its rows. Your full answer stays as written. A child's sentence is dropped and named in its result (ignored: summary)."},
                "fold_read": {"type": "boolean", "default": False, "description": "With summary, when this run read words here and owes nothing: my other conversations read this room's words up to this finish as this gist with their address, until a page seals them. My human's words stay whole; nothing is sealed."},
            }, "required": ["action"]}}, _finish_task),
        ToolEntry("set_tool_timeout", {
            "name": "set_tool_timeout",
            "description": (
                "Set the saved global tool-call timeout immediately. Tool minimums still apply; use a call's "
                'timeout_sec for a single long command.'
            ),
            "parameters": {"type": "object", "properties": {
                "seconds": {"type": "integer", "description": "New timeout in seconds (>= 1)"},
            }, "required": ["seconds"]},
        }, _set_tool_timeout),
        *self_change_tool_entries(),
        ToolEntry("promote_to_stable", {
            "name": "promote_to_stable",
            "description": (
                'Ask the supervisor to advance ouroboros-stable to ouroboros and push it. Evolution promotes its exact '
                'reviewed commit. Returns before completion; the owner receives the outcome.'
            ),
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
        }, _promote_to_stable),
        ToolEntry("start_task", {
            "name": "start_task",
            "description": _START_TASK_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {"type": "string", "description": "What the task must accomplish."},
                    "title": {"type": "string", "description": "Short task name the owner sees on its card and on the Main receipt, <=80 chars; also the Project name if the owner converts the task. The objective is not shown in its place."},
                    "project_name": {"type": "string", "description": "Display name of a NEW Project for the NEW task; its filesystem id is derived. Use move_to_project to move THIS task instead.", "default": ""},
                    "expected_output": {"type": "string", "description": "What done looks like.", "default": ""},
                    "project_id": {"type": "string", "description": "Optional EXISTING project scope (filesystem-clean id).", "default": ""},
                    "workspace_root": {"type": "string", "description": "Absolute ordinary folder or Git worktree root outside Ouroboros repo/data, validated at admission. File/process work supports ordinary folders; Git operations need a worktree. Omit for the Project's registered working_dir or, in Main, Ouroboros's own repo.", "default": ""},
                    "workspace": {"type": "string", "description": "'none' opts out of the Project's default folder; empty otherwise.", "default": ""},
                    "context_requires_self_body_docs": {"type": "boolean", "description": "True for work on Ouroboros code, including copies elsewhere: this task gets the full development handbook in Max. Low/Nano and helpers keep their usual book maps.", "default": False},
                    "context_requires_development": {"type": "boolean", "description": "true: all of docs/DEVELOPMENT.md in every mode; false: its chapter map."},
                    "review_answer": {"type": "boolean", "default": False, "description": "true when your human asks that this work's answer itself be checked: under task review Required the host reviews its whole final answer before delivery even when no file was made."},
                    "source": {"type": "string", "description": "Attach an existing folder or clone a git URL (https://... or git@host:path) server-side into the projects root; private auth failures are typed auth_required. Registers the folder with provenance + trusted_at as the Project/task workspace. Use for work on a supplied repo/folder.", "default": ""},
                    **_START_CHOICE_PARAMS,
                    "model": {"type": "string", "description": "Model for this new work, with your human's verbatim owner_words. An uncited choice is your own only in Cyber Pro; an owner choice still binds.", "default": ""},
                    "model_as_typed": {"type": "boolean", "default": False, "description": "Use the exact typed id despite similar catalog suggestions. The route is unverified; this confirms your choice, not availability."},
                    "owner_words": {"type": "string", "description": _OWNER_WORDS_PARAM, "default": ""},
                },
                "required": ["objective", "title"],
            },
        }, _start_task),
        ToolEntry("move_to_project", {
            "name": "move_to_project",
            "description": (
                "Place this work in a named Project room, keeping its task id and card. "
                "Use it when the current request belongs in a project; only this work moves. "
                "Its journal, knowledge tools and progress routing use the room immediately. "
                "The current execution keeps its folder and writer lease until readmission "
                "at the next execution boundary; knowledge is not automatically reloaded. "
                "The result reports the recorded durable binding, refusal or unconfirmed outcome. "
                "A task already bound to a room stays there; a requested name renames that room. "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "Display name for a NEW project (a filesystem id is derived from it). Honor the owner's stated name.", "default": ""},
                    "project_id": {"type": "string", "description": "Optional EXISTING project id (filesystem-clean) to attach to instead of creating one.", "default": ""},
                },
                "required": [],
            },
        }, _move_to_project),
        *_project_tool_entries(),
        ToolEntry("tell_task", {
            "name": "tell_task",
            "description": "Send a note to another work, with optional exact owner quotes selected from "
                           "your held owner_quote_handles. The note is yours; quoted words remain the owner's. "
                           "The work reads it at its next checkpoint. Paused work keeps the mail without resuming. "
                           "The owner sees it in that work's room as a note card: your summary first, the note on request. "
                           "Use forward_to_worker for messages inside your task tree. "
                           "Presence can address work of its own binding only.",
            "parameters": {"type": "object", "properties": {
                "task_id": {"type": "string", "description": "Work to receive the message."},
                "note": {"type": "string", "default": "", "description": "Your own words and context."},
                "summary": {"type": "string", "default": "",
                            "description": "One plain sentence for the owner, in the owner's language: what this note tells that work."},
                "quote": {"type": "array", "items": {"type": "string"}, "default": [],
                          "description": "Owner row addresses held by this turn; exact words and attachments travel together."},
                "arrival": {"type": ["boolean", "string"], "default": False,
                            "description": "Place focus only: forward the exact message as it arrived (your human's own words stay theirs, any other is context). true selects the opening "
                                           "message; a string, a later source_event_id delivered here from this conversation. One receiving work. "
                                           "Not with quote; note stays your own, separately."},
            }, "required": ["task_id"]},
        }, _tell_task),
        ToolEntry("schedule_subagent", {
            "name": "schedule_subagent",
            "description": _SCHEDULE_SUBAGENT_DESCRIPTION,
            "parameters": {
                "type": "object",
                # DERIVED: schedule_subagent_properties() is the one source this schema and the handler's keys read.
                "properties": schedule_subagent_properties(),
                "required": ["subagent_id", "objective", "expected_output"], "additionalProperties": False,
            },
        }, _schedule_task),
        # cancel_task + peek_task + discard_child_result are registered by ouroboros/tools/join_ledger.py.
        ToolEntry("request_deep_self_review", {
            "name": "request_deep_self_review",
            "description": "Request a deep self-review of the entire Ouroboros project against the Constitution: review_change(subject=system, surface=system) with one seat — the enabled catalog row you name in `reviewer` (review pool member or not), else the Main model. An API row runs a bounded read-only inspection episode, an agent-session row uses its configured access (writable sessions run in a private checking copy and retain evidence without applying changes); both receive the core memory whitelist inline byte-exact (memory is never receipt-checked). The report goes to chat and memory/deep_review.md, with the surface=system review record linked.",
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
            "description": "Write working memory (scratchpad): append a timestamped block, or with replaces fold the named blocks (the sha256 after each heading; an "
                           "earlier fold too) into one text you accept, placed where the oldest was. Originals are journaled first, unnamed blocks stay, a stale or "
                           "ambiguous selection changes nothing; nothing leaves by count, size or age. Write what you carry now, in any form: intentions, concerns, "
                           "questions, debts, waits, decisions that coordinate work, and why you would return; several can stand side by side. Another work's status "
                           "stands in its own facts: repeat it only where it changes what you intend. Persists across sessions, read whole at every task start. Project "
                           "rooms included: one working memory in every room.",
            "parameters": {"type": "object", "properties": {
                "content": {"type": "string", "description": "The new block, or the accepted text of a fold"},
                "replaces": {"type": "array", "items": {"type": "string"}, "description": "Fold: sha256 of each "
                             "active block this text replaces (a byte-identical block once per copy); omit to append"},
                "draft_source": {"type": "string", "description": "Source of a helper draft you accept (its task id)"},
            }, "required": ["content"]},
        }, _update_scratchpad),
        ToolEntry("send_user_message", {
            "name": "send_user_message",
            "description": "Reach the owner while work continues: an opening that says what I will check and why, "
                           "a substantial interim result they need now, a non-blocking question or an invitation to "
                           "collaborate. Do not pre-send a duplicate of the final answer: it already reaches this work's "
                           "answer room. A needed owner decision uses escalate: its card shows options, stays listed "
                           "as waiting and brings the answer here. This tool sends a normal reply in this work's room, "
                           "calls the owner back and leaves work running; routine news goes on update_focus, later "
                           "progress stays in the card. For Presence work that room is local, not the place conversation: "
                           "a connector reaches that, and the final answer only where context names that route.",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string", "description": "Message text"},
                "reason": {"type": "string", "description": "Why you're reaching out (logged, not sent)"},
                "request_id": {"type": "string", "description": "Presence Main contact retry key (at most 128 characters). Reuse the same key and words after uncertain delivery; a new key permits a distinct identical report. Omitted: identical words from this work rejoin."},
                "reply_to_relay": {"type": "string", "description": "Reply to a drained relay in its source room; this work's Project shows the same answer. Empty selects the single open relay; omit for an ordinary room message."},
                "destination": {"type": "string", "enum": ["current", "main"], "default": "current",
                                "description": _MESSAGE_DESTINATION_DESCRIPTION},
            }, "required": ["text"]},
        }, _send_user_message),
        ToolEntry("update_identity", {
            "name": "update_identity",
            "description": (
                'Replace your persistent identity manifest with the full supplied text after substantive reflection. '
                'Read it first and preserve continuity. This is the identity-writing tool in every project room; do not '
                'edit memory/identity.md through file tools.'
            ),
            "parameters": {"type": "object", "properties": {
                "content": {"type": "string", "description": "Full identity content (prefer evolving over rewriting from scratch)"},
                "expected_revision": {"type": "string", "description": "Required for an existing identity: source_revision (SHA-256 of bytes) from read_file. A stale revision refuses with the current revision."},
            }, "required": ["content"]},
        }, _update_identity),
        ToolEntry("update_world", {
            "name": "update_world",
            "description": "Update my map of my world (memory/WORLD.md): what lives in my environment — "
                           "agents, services, hosts, devices — how I reach each, what worked, what is held "
                           "or refused. The host owns the measured block (system, roots, tools found at "
                           "startup) and keeps it; I send the whole file with the rest revised. Read it "
                           "first and keep it compact (detail belongs in knowledge). The same continuous "
                           "file in every room and drive; this is the only correct way to write it — never "
                           "write memory/WORLD.md through write_file/edit_text.",
            "parameters": {"type": "object", "properties": {
                "content": {"type": "string", "description": "Full file content; the host keeps or restores its measured block"},
                "expected_revision": {"type": "string", "description": "Required for an existing file: source_revision (SHA-256 of bytes) from read_file. A stale revision refuses with the current revision."},
            }, "required": ["content"]},
        }, _update_world),
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
            "description": ("Control background consciousness: 'start' or 'stop' (the owner is told; a start does "
                            "not lift a stop the owner or Panic placed), or 'status' (answered to you only: the "
                            "persisted state with its source, never posted to the owner's chat)."),
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["start", "stop", "status"], "description": "Action to perform"},
            }, "required": ["action"]},
        }, _toggle_consciousness),
        ToolEntry("set_next_wakeup", {
            "name": "set_next_wakeup", "description": (
                "Choose the consciousness wake-up interval: how long after a wake-up ends the next one starts, "
                "clamped into the owner's OUROBOROS_BG_WAKEUP_MIN/MAX. A wake-up calling this sets its own next "
                "one; one already pending keeps its time; with consciousness off it is stored for later. The alarm "
                "adjusts it: a failed wake-up doubles the interval (up to MAX), a skipped one retries after MIN (an "
                "exhausted allowance waits for its reset or a changed daily limit), and no wake-up starts sooner "
                "than MIN after the last wake-up, boot or skip. Events never bring a wake-up forward; the next one "
                "observes them."),
            "parameters": {"type": "object", "properties": {"seconds": {"type": "integer", "description": "Seconds from the end of a wake-up to the next one"}}, "required": ["seconds"]},
        }, _set_next_wakeup),
        ToolEntry("switch_model", {
            "name": "switch_model",
            "description": (
                'Switch model or reasoning effort on the next round. No arguments lists available models. After a host '
                "fallback, primary='return' or 'wait' returns to this turn's primary route. "
                "A model beyond Main and Light needs your human's cited words (owner_words, scope)."
            ),
            "parameters": {"type": "object", "properties": {
                "model": {"type": "string", "description": "Exact model route. Cited work/Main choices require it; cited scope room accepts empty or main to follow Main. An uncited empty work switch keeps its acting route."},
                "model_as_typed": {"type": "boolean", "default": False, "description": "With owner_words, use the exact typed id despite similar catalog suggestions. It remains unverified; this grants no additional model authority."},
                "owner_words": {"type": "string", "description": "Your human's verbatim words choosing the model; the host checks provenance by bytes, you read their meaning."},
                "scope": {"type": "string", "enum": ["work", "room", "main"], "default": "work",
                          "description": "With owner_words: work sets this work's primary; room saves its room's model; main changes Main itself. Without a cite, work keeps the existing Main/Light acting switch; in Cyber Pro use the settings writer for your own room or Main changes."},
                "project_id": {"type": "string", "description": "For scope room: existing room id; omitted uses this work's room."},
                "effort": {"type": "string", "enum": list(EFFORT_SCALE),
                           "description": "Reasoning effort for the next round, inside my human's effort range. Main keeps its level outside Cyber Pro (delegate deeper thinking: schedule_subagent(effort=…)); a child, an evolution task or a wake moves inside the range; a pinned or model-named focus keeps its level. Leave empty to keep current."},
                "primary": {"type": "string", "enum": ["return", "wait"],
                            "description": ("Omit to keep the current route. Return to this turn's primary route: its model, role and account policy "
                                            "(Auto stays Auto) plus the owner's wait-card choice; effort stays. 'return': "
                                            "if the primary refuses, configured routes are tried again. 'wait': if it "
                                            "refuses or is unreachable, wait for it where this turn may wait instead of "
                                            "paid alternatives; after its content filter's stop nothing is there to wait for, "
                                            "so this run ends as interrupted, without them. The next real request tests it; no timer or probe does. "
                                            "Not with model.")},
            }, "required": []},
        }, _switch_model),
        get_task_result_entry(),
        ToolEntry("wait_task", {
            "name": "wait_task",
            "description": "Wait for ONE subtask to reach a terminal status and return its effective result: the full single-child handoff once it settled (or when your known_result_sha256 no longer matches); a return BEFORE it settled carries the compact wait_tasks projection plus delegated_runs (its open delegated runs: the observation the host records when a run's state or the engine's answer changes, dated by when that began, never per read, so no date proves a recent read; no liveness verdict). May return EARLY (before terminal) if the child raises a tree_note blocker/question/interface_contract/review_requested/delegation_constraint beacon — the result then carries a [CHILD_BEACONS] block so you can steer, review, or override it. Unread mailbox messages return early for the ordinary loop to deliver and acknowledge; host-marked context-only place arrivals stay unread until a genuine wake. The child keeps running. A child held by a current stop you have not yet been shown returns once with [HELD_DEPENDENCY] (its current_stop: cause, who releases it, the request door if yours); waiting again keeps waiting. Omitting sleep_mode sleeps even with timeout_sec (the wake time); saved state/obligations release your process unless live custody keeps it warm, named in the receipt. The result reaches your first request after wake. Explicit in_slot keeps the process and same-call result while yielding acting capacity; warm/cold choose the sleep mode. With SEVERAL children in flight, prefer wait_tasks(any_terminal) to absorb whichever finishes first rather than blocking serially on one id at a time.",
            "parameters": {"type": "object", "required": ["task_id"], "properties": {
                "task_id": {"type": "string", "description": "Task ID to check"},
                "view": {"type": "string", "enum": ["answer", "summary", "evidence", "full"], "description": "Settled result view; omitted/full preserves the existing handoff. Early returns remain compact wait observations, not a result or full read."},
                "known_result_sha256": {"type": "string", "description": "Prior child_result_sha256 in every view; current facts still refresh. This does not change when the wait ends."},
                "timeout_sec": {"type": "integer", "description":
                                "With no sleep_mode: the sleep's wake time, in seconds. With sleep_mode=in_slot: maximum seconds to wait (default 180); a larger value is clamped to "
                                f"{_WAIT_TASK_CLAMP_SEC}, and a deadline narrows it (named in the result). Size the window to the child's expected life."},
                **SLEEP_PARAMS,
            }},
        }, _wait_for_task, timeout_sec=7200),
        ToolEntry("wait_tasks", {
            "name": "wait_tasks",
            "description": "Wait for MULTIPLE subtasks at once: the right tool to ABSORB a batch of independent children you scheduled in one burst. With mode=any_terminal it returns as soon as the FIRST child finishes (handle it, then call again for the rest) instead of blocking serially. Each child gets a compact structural projection, its cost as accounted_upper_bound_usd with cost_final; the full per-child envelope stays in the task store (child_result_sha256 pins the exact result you saw; get_task_result returns the full result text plus trace/outcome summaries). The wait may early_return (before all terminal) on a child tree_note blocker/question/interface_contract/review_requested/delegation_constraint beacon so you can steer, review, or override mid-flight, or on unread mailbox messages (reason=owner_mailbox_pending); host-marked context-only place arrivals stay unread until a genuine wake. The ordinary loop handles delivery and acknowledgement. A child held by a current stop you have not yet been shown returns once (reason=dependency_held); waiting again keeps waiting. An id no surface of this tree ever minted (no task result, queue row or tree-ledger row) is flagged unknown_task_id ('not yet registered or never scheduled'), with unknown_task_ids and a compact children_roster of your ACTUAL direct children so you can repair the wait set instead of re-polling phantoms. Omitting sleep_mode sleeps even with timeout_sec (the wake time); unminted ids are then refused. Saved state/obligations release your process unless named live custody keeps it warm; the result reaches your first request after wake. Explicit in_slot keeps the process and same-call result while yielding acting capacity; warm/cold choose the sleep mode.",
            "parameters": {"type": "object", "required": ["task_ids"], "properties": {
                "task_ids": {"type": "array", "items": {"type": "string"}, "description": "Task IDs returned by schedule_subagent."},
                "known_result_sha256_by_task": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Optional task_id to previously obtained child_result_sha256 map. Matching children omit only result/trace and return result_unchanged plus a full-read reference. Missing or different hashes return the usual complete body/trace. Current status/cost/outcome/capability facts remain; wait timing is unchanged."},
                "timeout_sec": {"type": "integer", "description":
                                "With no sleep_mode: the sleep's wake time, in seconds. With sleep_mode=in_slot: maximum seconds to wait (default 600); a larger value is clamped to "
                                f"{_WAIT_TASKS_CLAMP_SEC}. Size the window to the children's expected life; "
                                "an expired wait returns the still-live ids and this ceiling; a deadline narrows it (window_bound)."},
                "mode": {"type": "string", "enum": ["all_terminal", "any_terminal"], "default": "all_terminal"},
                "view": {"type": "string", "enum": ["answer", "summary", "evidence", "full"], "description": "Optional: each settled child in this get_task_result view instead of the compact projection; unsettled children stay compact. Omitted keeps the compact batch."},
                **SLEEP_PARAMS,
            }},
        }, _wait_for_tasks, timeout_sec=_WAIT_TASKS_CLAMP_SEC + NESTED_SETTLEMENT_MARGIN_SEC),
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
    _attach_client_surface,
    _attach_origin_from_metadata,
    _finish_swarm_handoff,
    _close_project,
    _list_projects,
    _start_task,
)
from ouroboros.tools.control_runtime import (  # noqa: E402, F401 -- intentional public re-exports
    _chat_history,
    _evolution_restart_block_reason,
    _prepare_self_change,
    _promote_to_stable,
    self_change_tool_entries,
    _request_deep_self_review,
    _request_restart,
    _finish_task,
    _send_user_message,
    _set_next_wakeup,
    _set_tool_timeout,
    _switch_model,
    _toggle_consciousness,
    _toggle_evolution,
    _update_identity,
    _update_scratchpad,
    _update_world,
)


# v7next F2 (D07): moved spans live in their owner leaves; re-exported here
# so this facade stays the single import surface for callers and tests.
from ouroboros.tools.control_subagent_spec import (  # noqa: E402, F401 -- intentional public re-exports
    VALID_SUBTASK_MEMORY_MODES,
    _INTERNAL_SCHEDULE_OPTIONS,
    _validated_schedule_fields,
    requested_child_effort,
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
    maybe_emit_delegated_run_fanout,
)
from ouroboros.tools.control_task_results import (  # noqa: E402, F401 -- intentional public re-exports
    NESTED_SETTLEMENT_MARGIN_SEC,
    _UNMINTED_WAIT_GRACE_SEC,
    _WAIT_TASK_CLAMP_SEC,
    _WAIT_TASKS_CLAMP_SEC,
    _await_messages,
    _children_roster_projection,
    await_messages_entry,
    get_task_result_entry,
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

from ouroboros.tools.control_relay import _tell_task
