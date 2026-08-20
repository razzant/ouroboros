"""The single classifier answers exactly what the retired loop pair answered,
except on a table of deltas the owner approved.

The oracle is a golden snapshot of the OLD pair's answers, captured from the tree
named in ``GOLDEN_SOURCE_SHA``, not a copy of the old parser: a copy is dead code
that invites cleanup, a data file cannot drift silently.

Both directions fail. An unapproved divergence fails because the cutover would then
be changing behaviour nobody signed off. An APPROVED delta that no longer fires ALSO
fails, so the table cannot rot into a permanent excuse list — once a delta is gone,
its row goes with it.
"""

from __future__ import annotations

import json
import pathlib
from types import MappingProxyType
from typing import Mapping, NamedTuple

import pytest

from ouroboros._outcome_tool_errors import (
    _BLOCKING_TOOL_STATUSES,
    _NON_BLOCKING_READONLY_BLOCK_STATUSES,
    _NON_BLOCKING_RECOVERABLE_STATUSES,
    _OK_TOOL_STATUSES,
    _POLICY_DENIAL_STATUSES,
    _UNPARTITIONED_BUCKETS,
)
from ouroboros.loop_tool_execution import _typed_execution_failure, _typed_result_metadata
from ouroboros.tools.tool_result import TOOL_CODE_SPECS, LegacyTextResultAdapter
from tests.tool_classification_corpus import (
    GOLDEN_SOURCE_SHA,
    build_corpus,
    harvested_identifiers,
    harvested_native_codes,
    typed_result,
)

GOLDEN_PATH = pathlib.Path(__file__).resolve().parent / "fixtures" / "legacy_tool_classification_306f8827.json"


class Delta(NamedTuple):
    old_is_error: bool
    old_status: str
    new_is_error: bool
    new_status: str
    owner_item: str
    reason: str


# Every entry traces to a numbered item of the delta list the owner approved
# (batch #3 answer 1=A, batch #4 answers 1-3). Nothing else may differ.
APPROVED_DELTAS: Mapping[str, Delta] = MappingProxyType({
    "ACCESS_DENIED": Delta(False, "ok", True, "blocked", "A.4", "an access denial recorded as success is the worst under-reporting"),
    "ACTING_SUBAGENT_TOOL_NOT_GRANTED": Delta(False, "ok", True, "blocked", "A.4", "an ungranted tool for the acting subagent is a denial"),
    "CANCEL_INTENT_PROJECTION_CORRUPT": Delta(False, "ok", True, "error", "A.5", "a corrupted cancel projection is an error, not a success"),
    "CAPABILITY_UNAVAILABLE": Delta(True, "error", True, "unavailable", "A.18", "unavailability gets its own status name; the report bucket is unchanged"),
    "CHILD_RESULT_LINEAGE_FORBIDDEN": Delta(False, "ok", True, "blocked", "A.4", "a refused child-result lineage is a denial"),
    "CI_UNAVAILABLE": Delta(True, "error", True, "unavailable", "A.18", "unavailability gets its own status name; the report bucket is unchanged"),
    "COGNITIVE_TOOL_REQUIRED": Delta(True, "cognitive_tool_required", False, "ok", "A.11", "owner batch #4: the cognitive redirect is a hint, the error flag is removed"),
    "EXECUTOR_UNAVAILABLE": Delta(True, "error", True, "unavailable", "A.18", "unavailability gets its own status name; the report bucket is unchanged"),
    "EXTRACT_VIDEO_FRAMES_UNAVAILABLE": Delta(True, "error", True, "unavailable", "A.18", "unavailability gets its own status name; the report bucket is unchanged"),
    "GH_TIMEOUT": Delta(False, "ok", True, "timeout", "A.2", "an expired GitHub operation is a timeout, not a success"),
    "GIT_ERROR": Delta(False, "error", False, "git_error", "A.17", "the version-control refusal gets its own bucket; is_error is unchanged"),
    "INVALID_ARG": Delta(False, "ok", True, "argument_error", "A.6", "a bad pull-request argument is an error, not a success"),
    "MANAGED_UPDATE_STATE_UNAVAILABLE": Delta(True, "error", True, "unavailable", "A.18", "unavailability gets its own status name; the report bucket is unchanged"),
    "MCP_DISABLED": Delta(False, "ok", True, "unavailable", "A.3", "an MCP provider that is off is unavailable, not a success"),
    "MCP_TOOL_DISALLOWED": Delta(False, "ok", True, "blocked", "A.3", "an MCP tool refused by policy is a denial, not a success"),
    "MCP_TOOL_ERROR": Delta(True, "error", True, "mcp_error", "A.17", "the MCP error gets its own bucket, homed to the blocking partition"),
    "MCP_TOOL_NOT_FOUND": Delta(False, "ok", True, "unavailable", "A.3", "a missing MCP tool is unavailable, not a success"),
    "MCP_TOOL_TIMEOUT": Delta(False, "ok", True, "timeout", "A.3", "an expired MCP call is a timeout, not a success"),
    "MUTATIVE_SUBAGENTS_DISABLED": Delta(False, "ok", True, "blocked", "A.4", "a disabled mutative subagent is a denial"),
    "OCR_PDF_SCANNED_UNAVAILABLE": Delta(True, "error", True, "unavailable", "A.18", "unavailability gets its own status name; the report bucket is unchanged"),
    "OCR_PDF_UNAVAILABLE": Delta(True, "error", True, "unavailable", "A.18", "unavailability gets its own status name; the report bucket is unchanged"),
    "PYTHON_INTERPRETER_UNAVAILABLE": Delta(True, "error", True, "unavailable", "A.18", "unavailability gets its own status name; the report bucket is unchanged"),
    "REVIEW_BLOCKED": Delta(False, "blocked", False, "review_blocked", "A.17", "the review refusal gets its own bucket; is_error is unchanged"),
    "SKILLS_UNAVAILABLE": Delta(True, "error", True, "unavailable", "A.18", "unavailability gets its own status name; the report bucket is unchanged"),
    "SKILL_EXEC_TIMEOUT": Delta(False, "ok", True, "timeout", "A.2", "an expired skill run is a timeout, not a success"),
    "TASK_FORBIDDEN": Delta(False, "ok", True, "blocked", "A.4", "a forbidden task surface is a denial"),
    "TOOL_ARG_ERROR": Delta(True, "error", True, "argument_error", "A.17", "the argument error gets its own bucket, homed to the blocking partition"),
    "VIEW_IMAGE_UNAVAILABLE": Delta(True, "error", True, "unavailable", "A.18", "unavailability gets its own status name; the report bucket is unchanged"),
    "YOUTUBE_TRANSCRIPT_UNAVAILABLE": Delta(True, "error", True, "unavailable", "A.18", "unavailability gets its own status name; the report bucket is unchanged"),
    "body:empty": Delta(False, "ok", False, "untyped", "A.17", "a dynamic provider body is untyped rather than assumed ok; is_error is unchanged"),
    "body:list": Delta(False, "ok", False, "untyped", "A.17", "a dynamic provider body is untyped rather than assumed ok; is_error is unchanged"),
    "body:nested_only": Delta(False, "ok", False, "untyped", "A.17", "a dynamic provider body is untyped rather than assumed ok; is_error is unchanged"),
    "body:prose": Delta(False, "ok", False, "untyped", "A.17", "a dynamic provider body is untyped rather than assumed ok; is_error is unchanged"),
    "body:string_false": Delta(False, "ok", False, "untyped", "A.17", "a dynamic provider body is untyped rather than assumed ok; is_error is unchanged"),
    "body:true": Delta(False, "ok", False, "untyped", "A.17", "a dynamic provider body is untyped rather than assumed ok; is_error is unchanged"),
    "compose:exit:route+safety": Delta(False, "ok", True, "non_zero_exit", "A.7", "the safety wrapper no longer masks what it wraps"),
    "compose:exit:safety": Delta(False, "ok", True, "non_zero_exit", "A.7", "the safety wrapper no longer masks what it wraps"),
    "compose:integrate:route+safety": Delta(False, "ok", True, "integration_blocked", "A.7", "the safety wrapper no longer masks what it wraps"),
    "compose:integrate:safety": Delta(False, "ok", True, "integration_blocked", "A.7", "the safety wrapper no longer masks what it wraps"),
    "compose:protected:route+safety": Delta(False, "ok", True, "protected_blocked", "A.7", "the safety wrapper no longer masks what it wraps"),
    "compose:protected:safety": Delta(False, "ok", True, "protected_blocked", "A.7", "the safety wrapper no longer masks what it wraps"),
    "compose:reported:safety": Delta(False, "ok", True, "tool_reported_failure", "A.7", "the safety wrapper no longer masks what it wraps"),
    "compose:timeout:route+safety": Delta(False, "ok", True, "timeout", "A.7", "the safety wrapper no longer masks what it wraps"),
    "compose:timeout:safety": Delta(False, "ok", True, "timeout", "A.7", "the safety wrapper no longer masks what it wraps"),
    "compose:violation:route+safety": Delta(False, "ok", True, "safety_violation", "A.7", "the safety wrapper no longer masks what it wraps"),
    "compose:violation:safety": Delta(False, "ok", True, "safety_violation", "A.7", "the safety wrapper no longer masks what it wraps"),
    "edge:autocorrect_line2": Delta(True, "shell_error", True, "non_zero_exit", "A.13", "the wrapper body is classified by its own first line, so the exit error is named precisely"),
    "edge:autocorrect_line3": Delta(True, "shell_error", False, "ok_autocorrected", "A.13", "the loop's whole-remainder scan matched a marker three lines down; the body's first line governs"),
    "edge:safety_inner_block": Delta(False, "ok", True, "resource_policy_blocked", "A.7", "the safety wrapper no longer masks what it wraps"),
    "edge:unknown_tool": Delta(False, "ok", True, "unknown_tool", "A.1", "a call to a tool that does not exist was never a success"),
    "envelope:empty": Delta(False, "ok", False, "untyped", "A.17", "a dynamic provider body is untyped rather than assumed ok; is_error is unchanged"),
    "envelope:false": Delta(False, "ok", False, "untyped", "A.17", "a dynamic provider body is untyped rather than assumed ok; is_error is unchanged"),
    "envelope:false_indented": Delta(False, "ok", False, "untyped", "A.17", "a dynamic provider body is untyped rather than assumed ok; is_error is unchanged"),
    "envelope:list": Delta(False, "ok", False, "untyped", "A.17", "a dynamic provider body is untyped rather than assumed ok; is_error is unchanged"),
    "envelope:nested_only": Delta(False, "ok", False, "untyped", "A.17", "a dynamic provider body is untyped rather than assumed ok; is_error is unchanged"),
    "envelope:prose": Delta(False, "ok", False, "untyped", "A.17", "a dynamic provider body is untyped rather than assumed ok; is_error is unchanged"),
    "envelope:string_false": Delta(False, "ok", False, "untyped", "A.17", "a dynamic provider body is untyped rather than assumed ok; is_error is unchanged"),
    "envelope:true": Delta(False, "ok", False, "untyped", "A.17", "a dynamic provider body is untyped rather than assumed ok; is_error is unchanged"),
    "native:ACCESS_BLOCKED:ACTING_SUBAGENT_TOOL_NOT_GRANTED": Delta(False, "ok", True, "blocked", "A.4", "same denial through its native code"),
    # Same shape again: the `MUTATIVE_SUBAGENTS_DISABLED` identifier is already
    # approved above under A.4, and the two subagent-constraint guards in
    # `control_scheduling` now publish the code the adapter already assigned to
    # their text. The golden is the RETIRED LOOP, so the same answer reached
    # through the producer's own code shows up as a second row.
    "native:ACCESS_BLOCKED:MUTATIVE_SUBAGENTS_DISABLED": Delta(False, "ok", True, "blocked", "A.4", "same denial as the MUTATIVE_SUBAGENTS_DISABLED identifier row, through the native code the subagent-constraint guards publish"),
    "native:ACCESS_BLOCKED:MANAGED_UPDATE_IN_PROGRESS": Delta(False, "ok", True, "blocked", "A.4", "the managed-update denial is an access block its text never marked"),
    "native:CAPABILITY_UNAVAILABLE:CAPABILITY_UNAVAILABLE": Delta(True, "error", True, "unavailable", "A.18", "unavailability gets its own status name; the report bucket is unchanged"),
    "native:CAPABILITY_UNAVAILABLE:MANAGED_UPDATE_STATE_UNAVAILABLE": Delta(True, "error", True, "unavailable", "A.18", "unavailability gets its own status name; the report bucket is unchanged"),
    "native:CAPABILITY_UNAVAILABLE:PYTHON_INTERPRETER_UNAVAILABLE": Delta(True, "error", True, "unavailable", "A.18", "unavailability gets its own status name; the report bucket is unchanged"),
    "native:HEAL_MODE_BLOCKED:SKILL_REDIRECT_BLOCKED": Delta(True, "skill_payload_blocked", True, "heal_mode_blocked", "A.18", "the publisher's code wins over its text; both statuses are policy denials"),
    # Not a new owner decision: the identical move is already approved above for the
    # `TOOL_ARG_ERROR` identifier, and the root-argument refusal in
    # `core_file_tools._access_or_block` now publishes the code the adapter already
    # assigned to its text. The golden is the RETIRED LOOP, so reaching the same
    # answer through the producer's own code shows up as a second row.
    "native:TOOL_ARG_ERROR:TOOL_ARG_ERROR": Delta(True, "error", True, "argument_error", "A.17", "same bucket as the TOOL_ARG_ERROR identifier row, through the native code the read/list/write/edit/search root guard publishes"),
    # Same shape, same reason: `TASK_FORBIDDEN` is already approved above under A.4,
    # and `forward_to_worker` now publishes the code the adapter gave that text.
    "native:LEGACY_BLOCKED:TASK_FORBIDDEN": Delta(False, "ok", True, "blocked", "A.4", "same denial as the TASK_FORBIDDEN identifier row, through the native code the worker-forwarding guard publishes"),
    # Owner batch #10 item 2 (A.20): refusing a write because the room's files belong
    # to a promoted task is a policy denial, and it is answered by the tool whose
    # family the caller asked for, so each surface keeps its own bucket.
    "native:WRITE_FILE_BLOCKED:ROOM_WRITE_VIA_TASK": Delta(False, "ok", True, "write_file_blocked", "A.20", "a refused room write is a denial, not a written file"),
    "native:EDIT_TEXT_BLOCKED:ROOM_WRITE_VIA_TASK": Delta(False, "ok", True, "edit_text_blocked", "A.20", "a refused room edit is a denial, not an edited file"),
    # Owner batch #10 item 3 (A.20): forward_to_worker reported `ok` for every message
    # it did NOT deliver. A worker that is gone or finished is an unavailable target;
    # a worker being torn down refuses by policy. TASK_FORBIDDEN was already blocked.
    "native:LEGACY_UNAVAILABLE:TASK_NOT_FOUND": Delta(False, "ok", True, "unavailable", "A.20", "an unregistered task cannot receive a message"),
    "native:LEGACY_UNAVAILABLE:TASK_NOT_ACTIVE": Delta(False, "ok", True, "unavailable", "A.20", "a settled or not-yet-running task cannot receive a message"),
    "native:LEGACY_BLOCKED:TASK_CANCEL_PENDING": Delta(False, "ok", True, "blocked", "A.20", "steering a task under teardown is refused; the message was explicitly not delivered"),
    # Owner item A.21 (batch #13): the control tools reported `ok` for routing they
    # REFUSED or could not confirm. A rejected admission, a steer the supervisor
    # declined and a Swarm scope denial are policy denials; an unconfirmed receipt
    # is exactly the `unavailable` it describes, because the work may or may not
    # exist and the caller must not report it as done. Every sentence is unchanged.
    "native:ACCESS_BLOCKED:SWARM_PROJECT_SCOPE_OWNED": Delta(False, "ok", True, "blocked", "A.21", "a Project-room Swarm refused a project route; the denial was reported as a route"),
    "native:ACCESS_BLOCKED:SWARM_NEW_ROOT_REQUIRED": Delta(False, "ok", True, "blocked", "A.21", "an explicit Swarm refused to steer an existing task; the denial was reported as a steer"),
    "native:LEGACY_BLOCKED:NEEDS_MANUAL_TARGET": Delta(False, "ok", True, "blocked", "A.21", "no route was dispatched and the owner must choose the target"),
    "native:LEGACY_UNAVAILABLE:ROUTING_UNCONFIRMED": Delta(False, "ok", True, "unavailable", "A.21", "no route was dispatched and the manual-target delivery was never confirmed"),
    "native:LEGACY_BLOCKED:STEER_REJECTED": Delta(False, "ok", True, "blocked", "A.21", "a steer the supervisor declined delivered nothing to the target's mailbox"),
    "native:LEGACY_UNAVAILABLE:STEER_UNCONFIRMED": Delta(False, "ok", True, "unavailable", "A.21", "an unconfirmed mailbox delivery must not be reported as delivered"),
    # A.21, the memory writers: `REJECTED` ends in none of the suffixes the family
    # chain reads, so a scratchpad or identity write that was refused for a
    # malformed argument reported ok — the one answer that tells the caller its
    # arguments were fine.
    "native:TOOL_ARG_ERROR:REJECTED": Delta(False, "ok", True, "argument_error", "A.21", "a memory write refused for empty or too-short content is an argument error, not a write"),
    "shape:cognitive_redirect": Delta(True, "cognitive_tool_required", False, "ok", "A.11", "owner batch #4, through the native producer"),
    "shape:ephemeral_turn_denial": Delta(False, "ok", True, "blocked", "A.4",
        "the decision-turn denial is an access block its own first line never marked"),
    "shape:executor_crash": Delta(True, "error", True, "executor_error", "A.17", "the executor crash gets its own bucket, homed to the blocking partition"),
    "shape:git_error_untyped_text": Delta(False, "error", False, "git_error", "A.17", "same bucket rename through the native code, with no marker in the text"),
    "shape:mcp_provider_error": Delta(False, "ok", True, "mcp_error", "A.3",
        "the provider error was already typed; only the status beside it said success"),
    "shape:review_blocked_untyped_text": Delta(False, "blocked", False, "review_blocked", "A.17", "same bucket rename through the native code, with no marker in the text"),
    # Producers whose TEXT IS ASSEMBLED AT RUNTIME. The static harvest pairs a code
    # with a first line only when that line is a literal, so until the shapes below
    # entered the corpus these eight status changes were real and invisible: the
    # differential could not have failed on them, and neither could a mutation that
    # moved one of these codes to another status.
    "shape:extension_handler_error": Delta(True, "error", True, "extension_error", "A.17", "the extension error gets its own bucket, homed to the blocking partition"),
    "shape:extension_async_timeout": Delta(True, "error", True, "timeout", "A.18", "an expired extension handler is named a timeout; the report bucket is unchanged"),
    "shape:extension_not_live": Delta(True, "error", True, "unavailable", "A.18", "an extension that may not dispatch is unavailable; the report bucket is unchanged"),
    "shape:mcp_disabled": Delta(False, "ok", True, "unavailable", "A.3", "same fix as MCP_DISABLED, through the native code the provider publishes"),
    "shape:mcp_tool_not_found": Delta(False, "ok", True, "unavailable", "A.3", "same fix as MCP_TOOL_NOT_FOUND, through the native code the provider publishes"),
    "shape:mcp_transport_timeout": Delta(False, "ok", True, "timeout", "A.3", "same fix as MCP_TOOL_TIMEOUT, through the native code the provider publishes"),
    "shape:unknown_tool_extension_down": Delta(False, "ok", True, "unavailable", "A.1",
        "a call to a tool whose extension is not live was never a success; the registry publishes the more precise `unavailable` rather than `unknown_tool`"),
    "shape:binding_arg_error": Delta(True, "error", True, "argument_error", "A.17", "same bucket as TOOL_ARG_ERROR, through the interpolated binding-error text"),
    # Owner batch #10 item 1 (A.20): a media delivery that queued NOTHING reported
    # `ok` on both axes, because its refusal sentence has no identifier for the
    # adapter to key on. The three surfaces now name the failure themselves.
    "shape:send_photo_no_chat": Delta(False, "ok", True, "unavailable", "A.20", "no owner chat to deliver into is an unavailable surface, not a sent photo"),
    "shape:send_video_no_chat": Delta(False, "ok", True, "unavailable", "A.20", "no owner chat to deliver into is an unavailable surface, not a sent video"),
    "shape:send_file_no_chat": Delta(False, "ok", True, "unavailable", "A.20", "no owner chat to deliver into is an unavailable surface, not a sent file"),
    "shape:send_photo_read_failure": Delta(False, "ok", True, "error", "A.20", "an image the tool could not read was never delivered"),
    "shape:send_photo_empty_payload": Delta(False, "ok", True, "error", "A.20", "an empty payload was never delivered"),
    "shape:send_video_missing_file": Delta(False, "ok", True, "error", "A.20", "a missing video file was never delivered"),
    "shape:send_file_missing_argument": Delta(False, "ok", True, "error", "A.20", "a call with no file_path delivered nothing"),
    # A.21 control refusals stop reporting ok. These four sentences are the ones no
    # harvest can reach: the two promotion receipts carry no warning marker for the
    # identifier scan, and the two project-routing receipts reach their result
    # through the swarm-handoff latch, so no (code, first line) pair exists either.
    "shape:promote_rejected": Delta(False, "ok", True, "blocked", "A.21", "a promotion the supervisor refused created no task"),
    "shape:promote_unconfirmed": Delta(False, "ok", True, "unavailable", "A.21", "an unconfirmed admission must not be reported as a created task"),
    "shape:route_rejected": Delta(False, "ok", True, "blocked", "A.21", "a project route the supervisor refused scheduled nothing"),
    "shape:route_unconfirmed": Delta(False, "ok", True, "unavailable", "A.21", "an unconfirmed project route must not be reported as routed"),
    # A.21 across the remaining control leaves. Same blindness as above: a
    # markerless sentence, or one that reaches its result through a helper the
    # publication wraps, has no (code, first line) pair to harvest.
    "shape:deep_self_review_unavailable": Delta(False, "ok", True, "unavailable", "A.21", "a deep self-review nobody can run is an unavailable capability, not a queued review"),
    "shape:scratchpad_legacy_upgrade": Delta(False, "ok", True, "blocked", "A.21", "a scratchpad that needs a manual upgrade refused the append"),
    "shape:proactive_message_no_chat": Delta(False, "ok", True, "argument_error", "A.21", "no chat to deliver into means nothing was queued"),
    "shape:proactive_message_empty": Delta(False, "ok", True, "argument_error", "A.21", "an empty message was never queued"),
    "shape:switch_model_unknown": Delta(False, "ok", True, "argument_error", "A.21", "an unknown model name switched nothing"),
    "shape:subtask_depth_limit": Delta(False, "ok", True, "resource_constraint_blocked", "A.21", "a child beyond the depth limit was never scheduled; the limit is the constraint"),
    "shape:subagent_capability_mismatch": Delta(False, "ok", True, "argument_error", "A.21", "a profile that cannot satisfy the declared capabilities scheduled no child"),
    "shape:task_result_unknown_id": Delta(False, "ok", True, "unavailable", "A.21", "an id this tree never registered has no result to read"),
    # Owner item A.22 — owner decision 2026-08-19 ("B": schedule_followup publishes
    # native typed ToolResult; golden/corpus regeneration sanctioned). The adopted
    # one-shot follow-up tool reported `ok` for every registration it REFUSED,
    # because its sentences carry no identifier the adapter can key on: the same
    # defect as A.21, on the tool that mints FUTURE ROOT TASKS, where "registered"
    # and "refused" reading alike is the most expensive confusion in the family —
    # an agent told to wait for an instant that will never come. The producer names
    # each failure itself; every sentence is byte-identical.
    "shape:followup_subagent_refused": Delta(False, "ok", True, "blocked", "A.22", "a delegated subagent may not mint future root tasks; the denial was reported as a registration"),
    "shape:followup_task_id_required": Delta(False, "ok", True, "unavailable", "A.22", "no real task to own the durable record is an unavailable surface, not a scheduled follow-up"),
    "shape:followup_run_at_invalid": Delta(False, "ok", True, "argument_error", "A.22", "an unparseable run_at scheduled nothing"),
    "shape:followup_objective_required": Delta(False, "ok", True, "argument_error", "A.22", "an empty objective scheduled nothing"),
    "shape:followup_text_too_long": Delta(False, "ok", True, "argument_error", "A.22", "an over-limit objective or context is refused whole — never truncated, never scheduled"),
    "shape:followup_data_root_unresolved": Delta(False, "ok", True, "error", "A.22", "a drive root that could not be resolved wrote no record"),
    "shape:followup_cap_reached": Delta(False, "ok", True, "resource_constraint_blocked", "A.22", "a follow-up beyond the per-task cap was never registered; the cap is the constraint"),
    "shape:followup_persist_failed": Delta(False, "ok", True, "error", "A.22", "a follow-up the table refused to store does not exist"),
})

# Deltas the classifier WOULD produce for which no producer exists, recorded so a
# later reader can tell "checked, unreachable" from "missed". Neither can appear in
# APPROVED_DELTAS: that table fails on rows that do not fire, and these cannot fire.
_DELTAS_WITHOUT_A_PRODUCER: Mapping[str, str] = MappingProxyType({
    "SKILL_PAYLOAD_CONTROL_BLOCKED": (
        "would move (is_error=True, skill_payload_control_blocked) -> (is_error=True, "
        "blocked). The retired loop branch was its only mention; no producer emits "
        "the identifier, so the corpus harvest never sees it and the generic "
        "`_BLOCKED` marker would answer if one ever did."
    ),
    "four nested SAFETY_WARNING wrappers": (
        "would move ok -> error (LEGACY_TOOL_ERROR, wrapper_depth_exceeded). The "
        "composer wraps at most once, so a body reaching depth four is a producer "
        "quoting the wrapper's own text at itself, not a runtime shape."
    ),
})


def _golden() -> dict[str, dict]:
    payload = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert payload["source_sha"] == GOLDEN_SOURCE_SHA, "golden was captured from another tree"
    return payload["entries"]


def _live_answer(case) -> tuple[bool, str]:
    typed = typed_result(case)
    is_error = _typed_execution_failure(True, typed)
    return is_error, _typed_result_metadata(case.tool, case.text, is_error, typed)["status"]


def test_single_classifier_matches_the_retired_pair_except_approved_deltas() -> None:
    golden = _golden()
    corpus = build_corpus()
    assert len(corpus) >= 600, "the corpus collapsed; a harvest regression would hide every delta"

    unexpected: list[tuple[str, dict, tuple[bool, str]]] = []
    unfired = set(APPROVED_DELTAS)
    for case in corpus:
        assert case.key in golden, f"no golden answer for {case.key}: regenerate before trusting this run"
        old = golden[case.key]
        live = _live_answer(case)
        if live == (old["is_error"], old["status"]):
            continue
        delta = APPROVED_DELTAS.get(case.subject)
        expected = None if delta is None else (
            (delta.old_is_error, delta.old_status), (delta.new_is_error, delta.new_status)
        )
        if expected != ((old["is_error"], old["status"]), live):
            unexpected.append((case.key, old, live))
        else:
            unfired.discard(case.subject)

    assert not unexpected, f"unapproved classification changes: {unexpected[:12]}"
    assert not unfired, f"approved deltas that no longer fire (delete the rows): {sorted(unfired)}"


def test_every_approved_delta_names_an_owner_item() -> None:
    for subject, delta in APPROVED_DELTAS.items():
        assert delta.owner_item.startswith("A."), subject
        assert delta.reason.strip(), subject
        assert (delta.old_is_error, delta.old_status) != (delta.new_is_error, delta.new_status), subject


def test_every_delta_without_a_producer_is_named_with_its_reason() -> None:
    """The two unreachable deltas stay documented, never approved: an approved row
    that cannot fire would fail the table's own staleness direction."""
    assert set(_DELTAS_WITHOUT_A_PRODUCER).isdisjoint(APPROVED_DELTAS)
    for subject, reason in _DELTAS_WITHOUT_A_PRODUCER.items():
        assert reason.strip(), subject
    assert "SKILL_PAYLOAD_CONTROL_BLOCKED" not in harvested_identifiers()


def test_every_native_code_is_covered_by_the_corpus() -> None:
    """A producer cannot publish a code the differential has never classified.

    The text corpus is blind to a producer that assembles its text at runtime, and
    the (code, first line) harvest is blind for the same reason: it needs a literal.
    That blind spot let four extension terminals, both MCP unavailable terminals and
    the registry's unknown-tool publish change status with no test able to notice.
    Any new native code now fails here until a corpus case exercises it."""
    covered = {case.code for case in build_corpus() if case.code}
    uncovered = sorted(set(harvested_native_codes()) - covered)
    assert not uncovered, (
        "native producer codes no corpus case classifies (add a _PRODUCER_SHAPES "
        f"entry transcribed from the producer): {uncovered}"
    )


def test_a_self_reported_failure_is_telemetry_on_the_execution_axis() -> None:
    """Owner homing of `tool_reported_failure`, asserted where it is consumed.

    A provider that RAN and answered `{"ok": false}` is is_error=True — the counters
    and the anti-loop scan need that — but it must not degrade execution health,
    because `outcomes._LEDGER_NON_FAILURE_STATUSES` has declared the SAME status a
    non-failure since v6.83.0. Homing it as blocking-only made every unrecovered
    ext_/mcp_/read `{"ok": false}` land in `unresolved` on one axis while the ledger
    called it fine on the other. Asserted through the classifier, not by frozenset
    membership, so a future re-homing has to face the contradiction again."""
    from ouroboros._outcome_tool_errors import _classify_tool_errors

    buckets = _classify_tool_errors({"tool_calls": [{
        "tool": "ext_1_demo_screenshot",
        "status": "tool_reported_failure",
        "is_error": True,
        "result": '{"ok": false, "error": "HTTP 500"}',
    }]})
    assert [row["tool"] for row in buckets["policy_denials"]] == ["ext_1_demo_screenshot"]
    assert buckets["unresolved"] == []
    # And the two consumers agree: the ledger says the same about the same status.
    from ouroboros.outcomes import _LEDGER_NON_FAILURE_STATUSES

    assert "tool_reported_failure" in _LEDGER_NON_FAILURE_STATUSES
    # A recovered one is still credited: it is walked, not skipped.
    recovered = _classify_tool_errors({"tool_calls": [
        {"tool": "ext_1_demo_screenshot", "status": "tool_reported_failure",
         "is_error": True, "args": {"path": "/x/shot.png"}},
        {"tool": "ext_1_demo_screenshot", "status": "ok",
         "is_error": False, "args": {"path": "/x/shot.png"}},
    ]})
    assert len(recovered["recovered"]) == 1
    assert recovered["policy_denials"] == []


def test_a_control_refusal_typed_unavailable_is_the_substrates_answer() -> None:
    """Owner homing of `unavailable` (spec §1.15), asserted where it is consumed.

    A target the runtime cannot serve — a legacy control surface that is off, a
    task id this tree never registered — answers with the typed `unavailable`
    the A.21 producers ship (e.g. control_task_results' LEGACY_UNAVAILABLE). That
    is the SUBSTRATE saying no, not the agent failing: it stays is_error=True and
    blocking, but it must not land in `unresolved` and degrade execution health.
    `argument_error` deliberately stays degrading — a malformed call is the
    agent's own defect and feeds reflection. Asserted through the classifier, not
    by frozenset membership, so a re-homing has to face the split again."""
    from ouroboros._outcome_tool_errors import _classify_tool_errors

    buckets = _classify_tool_errors({"tool_calls": [{
        "tool": "get_task_result",
        "status": "unavailable",
        "is_error": True,
        "result": "⚠️ LEGACY_UNAVAILABLE: no result recorded for task_00000000",
    }]})
    assert [row["tool"] for row in buckets["policy_denials"]] == ["get_task_result"]
    assert buckets["unresolved"] == []
    # The other half of the §1.15 split: the agent's own malformed call degrades.
    mistake = _classify_tool_errors({"tool_calls": [{
        "tool": "get_task_result",
        "status": "argument_error",
        "is_error": True,
        "result": "⚠️ ROUTING_ARGUMENT: task_id is required",
    }]})
    assert mistake["policy_denials"] == []
    assert [row["tool"] for row in mistake["unresolved"]] == ["get_task_result"]


def test_golden_covers_every_harvested_producer() -> None:
    """A producer added after the cutover has no golden answer, so it fails here
    instead of silently entering the tree with an unverified classification."""
    golden = _golden()
    missing = [
        identifier for identifier in harvested_identifiers()
        if f"ident:{identifier}:plain" not in golden
    ]
    assert not missing, f"new warning identifiers without a golden answer: {missing}"


def test_specific_identifiers_beat_their_family_and_families_beat_generic_markers() -> None:
    """Order, asserted as behaviour rather than as a position in a table."""
    def bucket(text: str) -> str:
        return TOOL_CODE_SPECS[LegacyTextResultAdapter.from_text("fixture_tool", text).code].outcome_bucket

    assert bucket("⚠️ SHELL_CWD_BLOCKED: escapes roots") == "cwd_blocked"
    assert bucket("⚠️ SHELL_EXIT_ERROR: exit_code=1") == "non_zero_exit"
    assert bucket("⚠️ SHELL_ENV_ERROR: bad env") == "shell_error"
    assert bucket("⚠️ RUN_SCRIPT_BLOCKED: interpreter") == "run_script_blocked"
    assert bucket("⚠️ RUN_SCRIPT_LAUNCH_ERROR: boom") == "run_script_error"
    assert bucket("⚠️ LIGHT_MODE_REPO_WRITE_BLOCKED: repo") == "light_mode_blocked"
    assert bucket("⚠️ INTEGRATE_TARGET_ERROR: not git") == "integration_blocked"
    assert bucket("⚠️ INTEGRATE_LOCK_TIMEOUT: busy") == "integration_blocked"
    assert bucket("⚠️ WRITE_FILE_ERROR: boom") == "write_file_blocked"
    assert bucket("⚠️ EDIT_TEXT_ERROR: old_str not found") == "edit_text_blocked"
    assert bucket("⚠️ APPLY_PATCH_ERROR: occurrence miscount") == "edit_ops_blocked"
    assert bucket("⚠️ EDIT_BATCH_ERROR: occurrence miscount") == "edit_ops_blocked"
    assert bucket("⚠️ DATA_WRITE_ERROR: refused") == "data_blocked"
    assert bucket("⚠️ SKILL_PAYLOAD_ARG_ERROR: bad selector") == "skill_payload_blocked"
    assert bucket("⚠️ ROOT_REQUIRED_USER_FILES: retry") == "root_required_user_files"
    assert bucket("⚠️ ROOT_REQUIRED_ACTIVE_WORKSPACE: retry") == "root_required_active_workspace"
    assert bucket("⚠️ RESOURCE_CONSTRAINT_BLOCKED: no network") == "resource_constraint_blocked"
    assert bucket("⚠️ RESOURCE_POLICY_BLOCKED: protected") == "resource_policy_blocked"
    assert bucket("⚠️ UNKNOWN_COARSE_BLOCKED: generic") == "blocked"
    assert bucket("⚠️ UNKNOWN_COARSE_ERROR: generic") == "error"
    # The one negation in the retired chain: an autocorrected command that also
    # exited non-zero must not read as a plain autocorrected success.
    assert bucket("⚠️ SHELL_REGEX_AUTO_CORRECTED: fixed\n⚠️ SHELL_EXIT_ERROR: exit_code=1") == "non_zero_exit"
    assert bucket("⚠️ SHELL_REGEX_AUTO_CORRECTED: fixed\nexit_code=0") == "ok_autocorrected"


def test_every_outcome_bucket_is_partitioned() -> None:
    """A new code cannot acquire a bucket the outcome classifier does not know.

    Without this, a call can be an honest error while `unresolved`, `policy_denials`,
    `recovered`, `cosmetic` and `ignored` are all empty — two numbers in one artifact
    contradicting each other, neither of them wrong."""
    known = (
        set(_BLOCKING_TOOL_STATUSES)
        | set(_POLICY_DENIAL_STATUSES)
        | set(_NON_BLOCKING_RECOVERABLE_STATUSES)
        | set(_NON_BLOCKING_READONLY_BLOCK_STATUSES)
        | set(_OK_TOOL_STATUSES)
        | set(_UNPARTITIONED_BUCKETS)
    )
    unhomed = sorted({spec.outcome_bucket for spec in TOOL_CODE_SPECS.values()} - known)
    assert not unhomed, f"outcome buckets with no partition: {unhomed}"
    # Everything deliberately left out is named, and nothing else is.
    assert set(_UNPARTITIONED_BUCKETS) == {"vlm_error"}


# Text inspections that survive OUTSIDE the one classifier, with the reason each
# cannot be expressed as a tool-result code. The cap may shrink, never grow, and a
# module absent from this inventory may hold none at all: that is the executable
# form of "one adapter plus an inventory of residual string producers".
_RESIDUAL_TEXT_INSPECTIONS: Mapping[str, tuple[int, str]] = MappingProxyType({
    "ouroboros/outcomes.py": (5, "the FINAL ANSWER and service-teardown text, for which no ToolResult exists"),
    "ouroboros/reflection.py": (6, "markers emitted INSIDE a result body, which a first-line parser cannot see"),
    "ouroboros/memory.py": (1, "tools.jsonl rows appended by consciousness carry neither status nor code"),
    "ouroboros/skill_review_prompt.py": (2, "skill review verdict text, not a tool result"),
    "ouroboros/tools/github.py": (12, "private helper-failure checks between two functions of one tool"),
    "ouroboros/tools/skill_publish.py": (9, "private helper-failure checks between two functions of one tool"),
    # The v7 L-C advisory split moved these checks verbatim with their code;
    # the three caps sum to the parent's former 8 — redistribution, not growth.
    "ouroboros/tools/claude_advisory_review.py": (4, "private helper-failure checks between two functions of one tool"),
    "ouroboros/tools/review_advisory_prompt.py": (1, "private helper-failure checks between two functions of one tool"),
    "ouroboros/tools/review_advisory_run.py": (3, "private helper-failure checks between two functions of one tool"),
    "ouroboros/tools/core_file_tools.py": (3, "private helper-failure checks between two functions of one tool"),
    "ouroboros/tools/services.py": (1, "private helper-failure check between two functions of one tool"),
    "ouroboros/tools/core.py": (1, "private helper-failure check between two functions of one tool"),
    "ouroboros/tools/control_delegation.py": (1, "private helper-failure check between two functions of one tool"),
})
_RESIDUAL_PATTERNS = ('startswith("⚠️', 'startswith(("⚠️', "_ERROR_MARKERS", "_INFRA_TEXT_PREFIXES")


def test_residual_text_inspection_inventory_does_not_grow() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    counted: dict[str, int] = {}
    for path in sorted((root / "ouroboros").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel == "ouroboros/tools/tool_result.py":
            continue  # the one classifier IS the inspection
        text = path.read_text(encoding="utf-8")
        hits = sum(text.count(pattern) for pattern in _RESIDUAL_PATTERNS)
        if hits:
            counted[rel] = hits

    new_modules = sorted(set(counted) - set(_RESIDUAL_TEXT_INSPECTIONS))
    assert not new_modules, f"a new module started classifying result text: {new_modules}"
    grew = {
        rel: (hits, _RESIDUAL_TEXT_INSPECTIONS[rel][0])
        for rel, hits in counted.items()
        if hits > _RESIDUAL_TEXT_INSPECTIONS[rel][0]
    }
    assert not grew, f"residual text inspections grew: {grew}"
    # The loop is the one that mattered: it holds none.
    assert "ouroboros/loop_tool_execution.py" not in counted


@pytest.mark.parametrize("case_key", ["shape:shell_no_match_autocorrected", "shape:shell_ok"])
def test_process_facts_stay_typed_not_parsed(case_key: str) -> None:
    """The three post-rules that are NOT text classification keep working off meta."""
    case = next(item for item in build_corpus() if item.key == case_key)
    typed = typed_result(case)
    meta = _typed_result_metadata(case.tool, case.text, False, typed)

    assert meta["exit_code"] == dict(case.meta)["exit_code"]
    expected = "ok_autocorrected" if dict(case.meta).get("shell_regex_auto_corrected") else "ok"
    assert meta["status"] == expected
