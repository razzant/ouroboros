"""The control tool producers publish their own result code, with unchanged text.

Same two things are pinned per site as in ``tests/test_core_native_results.py``,
because either one alone would let the cutover change what the loop records:

* the EXACT text the producer returned before it published anything — the string
  ABI the model sees is unchanged;
* what the published code says about the call, computed rather than restated. For
  the argument and access refusals that is equality with the single adapter's
  answer for the same bytes, so nativisation carries no owner semantics; for the
  owner-approved A.21 rows it is the OPPOSITE — the divergence has to be real, so
  an approved exception cannot rot into a silent one.
"""

from __future__ import annotations

import pathlib

import pytest

from ouroboros.tools import control_routing, control_runtime, control_scheduling, control_task_results
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import (
    LegacyTextResultAdapter,
    ToolResult,
    _install_tool_result_sidecar,
    _published_tool_result,
    _restore_tool_result_sidecar,
)


def _published(ctx, tool: str, call, *, owner_delta: str = "") -> ToolResult:
    """Run one producer under the registry's own result-consumption rule.

    ``registry_core`` installs a per-invocation sentinel and accepts the published
    result only when its text is exactly the string the handler returned; a helper
    called outside a dispatch must therefore still return that same text.
    """
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        text = call()
        published = _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)
    assert isinstance(published, ToolResult), f"{tool}: producer published no typed result"
    assert published.text == text, f"{tool}: published text is not the returned text"
    adapter_code = LegacyTextResultAdapter.from_text(tool, text).code
    if owner_delta:
        assert published.code != adapter_code, (
            f"{tool}: {owner_delta} claims a divergence from the adapter that is not there"
        )
    else:
        assert published.code == adapter_code, (
            f"{tool}: published code diverges from the adapter answer for the same text"
        )
    return published


def _ctx(tmp_path: pathlib.Path) -> ToolContext:
    repo = tmp_path / "repo"
    drive = tmp_path / "drive"
    repo.mkdir(exist_ok=True)
    (drive / "logs").mkdir(parents=True, exist_ok=True)
    return ToolContext(repo_dir=repo, drive_root=drive, task_metadata={})


# --- Table 1: the adapter's own answer, published by the branch that made it ---


@pytest.mark.parametrize(
    ("label", "tool", "text"),
    [
        (
            "promote_no_objective",
            "promote_chat_to_task",
            "⚠️ TOOL_ARG_ERROR (promote_chat_to_task): objective is required",
        ),
        (
            "promote_bad_project_id",
            "promote_chat_to_task",
            "⚠️ TOOL_ARG_ERROR (promote_chat_to_task): project_id 'Not/Clean!' is not "
            "filesystem-clean; use lowercase alphanumeric/_/-/. (<=64 chars)",
        ),
        (
            "route_no_message",
            "route_to_project",
            "⚠️ TOOL_ARG_ERROR (route_to_project): message is required",
        ),
        (
            "steer_no_task_id",
            "steer_task",
            "⚠️ TOOL_ARG_ERROR (steer_task): task_id is required — pick one from "
            "current_chat.running_tasks (or promote_chat_to_task to start new work).",
        ),
        (
            "steer_no_message",
            "steer_task",
            "⚠️ TOOL_ARG_ERROR (steer_task): message is required.",
        ),
    ],
)
def test_routing_argument_refusals_publish_their_adapter_code(tmp_path, label, tool, text):
    ctx = _ctx(tmp_path)
    calls = {
        "promote_no_objective": lambda: control_routing._promote_chat_to_task(ctx, ""),
        "promote_bad_project_id": lambda: control_routing._promote_chat_to_task(
            ctx, "do the work", project_id="Not/Clean!"),
        "route_no_message": lambda: control_routing._route_to_project(ctx, project_id="p"),
        "steer_no_task_id": lambda: control_routing._steer_task(ctx, "", "hello"),
        "steer_no_message": lambda: control_routing._steer_task(ctx, "abc123", ""),
    }

    published = _published(ctx, tool, calls[label])

    assert published.code == "TOOL_ARG_ERROR"
    assert published.status == "error"
    assert published.text == text


@pytest.mark.parametrize(
    ("label", "text"),
    [
        (
            "evolution_block",
            "⚠️ RESTART_BLOCKED: in evolution mode, HEAD changed after the last reviewed local commit.",
        ),
        (
            "receipt_not_persisted",
            "⚠️ RESTART_BLOCKED: the exact evolution restart receipt could not be persisted (boom).",
        ),
    ],
)
def test_restart_denials_publish_their_adapter_code(tmp_path, monkeypatch, label, text):
    ctx = _ctx(tmp_path)
    ctx.current_task_type = "evolution"
    if label == "evolution_block":
        monkeypatch.setattr(
            control_runtime, "_evolution_restart_block_reason",
            lambda _ctx: "HEAD changed after the last reviewed local commit",
        )
    else:
        monkeypatch.setattr(control_runtime, "_evolution_restart_block_reason", lambda _ctx: "")

        def _boom(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(control_runtime, "run_cmd", _boom)

    published = _published(
        ctx, "request_restart", lambda: control_runtime._request_restart(ctx, "why"))

    assert published.code == "LEGACY_BLOCKED"
    assert published.status == "blocked"
    assert published.text == text


def test_subagent_constraint_denials_publish_their_adapter_code(tmp_path, monkeypatch):
    """Both guards that refuse an acting child name the denial themselves.

    The selector's own refusal and the one it delegates to the acting-constraint
    builder are the same policy answer, and the builder receives the invocation so
    the branch that made the decision is the branch that reports it.
    """
    import ouroboros.config as config

    monkeypatch.setattr(config, "get_allow_mutative_subagents", lambda _surface: False)
    ctx = _ctx(tmp_path)

    toggled_off = _published(
        ctx, "schedule_subagent",
        lambda: control_scheduling._build_acting_constraint(
            write_surface="self_worktree", write_root="", protected_paths_grant=False,
            external_tool_grants=None, parent_workspace_root="", ctx=ctx),
    )
    assert toggled_off.code == "ACCESS_BLOCKED"
    assert toggled_off.status == "blocked"
    assert toggled_off.text.startswith(
        "⚠️ MUTATIVE_SUBAGENTS_DISABLED: acting children with "
        "write_surface='self_worktree' are disabled here. "
    )

    readonly_parent = _published(
        ctx, "schedule_subagent",
        lambda: control_scheduling._select_subagent_constraint(
            "self_worktree", "", False, [], "", caller_readonly=True, ctx=ctx),
    )
    assert readonly_parent.code == "ACCESS_BLOCKED"
    assert readonly_parent.status == "blocked"
    assert readonly_parent.text == (
        "⚠️ MUTATIVE_SUBAGENTS_DISABLED: a read-only subagent cannot spawn a mutative (acting) "
        "child. Only the root agent, workspace tasks, or acting subagents may pass write_surface; "
        "schedule a read-only child instead."
    )


def test_a_direct_selector_call_without_an_invocation_still_returns_its_text(tmp_path, monkeypatch):
    """``ctx`` is optional, so a caller outside a dispatch keeps the exact string.

    The publication seam must not become a reason for the selector to require a
    context it does not otherwise need; without an invocation there is simply
    nothing to publish into.
    """
    import ouroboros.config as config

    monkeypatch.setattr(config, "get_allow_mutative_subagents", lambda _surface: False)

    refusal = control_scheduling._select_subagent_constraint("self_worktree", "", False, [], "")

    assert isinstance(refusal, str)
    assert refusal.startswith("⚠️ MUTATIVE_SUBAGENTS_DISABLED: acting children with ")


@pytest.mark.parametrize(
    ("label", "prefix"),
    [
        ("retired_param", "⚠️ TOOL_ARG_ERROR (schedule_subagent): effort was withdrawn: "),
        ("unsupported_param", "⚠️ TOOL_ARG_ERROR (schedule_subagent): unsupported argument(s): bogus."),
        ("validator_refusal", "⚠️ TOOL_ARG_ERROR (schedule_subagent): objective is required."),
        (
            "capability_arg_error",
            "⚠️ TOOL_ARG_ERROR (schedule_subagent): required_capabilities must be a list of strings.",
        ),
    ],
)
def test_schedule_argument_refusals_publish_their_adapter_code(tmp_path, label, prefix):
    ctx = _ctx(tmp_path)
    calls = {
        "retired_param": lambda: control_scheduling._schedule_task(ctx, effort="high"),
        "unsupported_param": lambda: control_scheduling._schedule_task(ctx, bogus=1),
        "validator_refusal": lambda: control_scheduling._schedule_task(ctx, objective=""),
        "capability_arg_error": lambda: control_scheduling._schedule_task(
            ctx, objective="o", expected_output="e", required_capabilities="shell"),
    }

    published = _published(ctx, "schedule_subagent", calls[label])

    assert published.code == "TOOL_ARG_ERROR"
    assert published.status == "error"
    assert published.text.startswith(prefix)


@pytest.mark.parametrize(
    ("label", "tool", "text"),
    [
        (
            "wait_task_bad_id",
            "wait_task",
            "⚠️ TOOL_ARG_ERROR (wait_task): task_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
        ),
        (
            "wait_tasks_empty",
            "wait_tasks",
            "⚠️ TOOL_ARG_ERROR (wait_tasks): task_ids must be a non-empty list.",
        ),
        (
            "wait_tasks_bad_id",
            "wait_tasks",
            "⚠️ TOOL_ARG_ERROR (wait_tasks): task_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
        ),
        (
            "wait_tasks_bad_mode",
            "wait_tasks",
            "⚠️ TOOL_ARG_ERROR (wait_tasks): mode must be all_terminal or any_terminal.",
        ),
    ],
)
def test_wait_argument_refusals_publish_their_adapter_code(tmp_path, label, tool, text):
    ctx = _ctx(tmp_path)
    calls = {
        "wait_task_bad_id": lambda: control_task_results._wait_for_task(ctx, "not a task id!"),
        "wait_tasks_empty": lambda: control_task_results._wait_for_tasks(ctx, []),
        "wait_tasks_bad_id": lambda: control_task_results._wait_for_tasks(ctx, ["not a task id!"]),
        "wait_tasks_bad_mode": lambda: control_task_results._wait_for_tasks(
            ctx, ["abc123"], mode="whenever"),
    }

    published = _published(ctx, tool, calls[label])

    assert published.code == "TOOL_ARG_ERROR"
    assert published.status == "error"
    assert published.text == text


# --- Table 2 / owner item A.21: routing refusals stop reporting ok ---


def _routing_ctx(tmp_path: pathlib.Path, monkeypatch, receipt: dict, *, mode: str = "live"):
    """One promote/route/steer invocation with the supervisor receipt it gets back."""
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(control_routing, "_promotion_pool_disabled_from_snapshot", lambda _ctx: "")
    monkeypatch.setattr(
        control_routing, "_emit_and_wait_for_routing",
        lambda _ctx, _evt: (mode, dict(receipt)),
    )
    return ctx


def test_a_promotion_that_scheduled_nothing_is_not_a_created_task(tmp_path, monkeypatch):
    """Owner item A.21: PROMOTE_REJECTED/PROMOTE_UNCONFIRMED reported `ok`.

    Their sentences carry no warning marker at all, so the adapter had nothing to
    key on and a task that was refused — or whose admission was never confirmed —
    looked to the caller exactly like a task that had been created. The refusal is
    a policy denial now and the unconfirmed receipt is the `unavailable` it
    describes; both sentences are byte-identical.
    """
    from ouroboros.tools.control_events import _PROMOTE_CONFIRM_TIMEOUT_SEC

    pool_off = _ctx(tmp_path)
    monkeypatch.setattr(
        control_routing, "_promotion_pool_disabled_from_snapshot", lambda _ctx: "no workers",
    )
    disabled = _published(
        pool_off, "promote_chat_to_task",
        lambda: control_routing._promote_chat_to_task(pool_off, "build it"),
        owner_delta="A.21",
    )
    assert (disabled.code, disabled.status) == ("LEGACY_BLOCKED", "blocked")
    assert disabled.text.startswith("PROMOTE_REJECTED: task ")
    assert disabled.text.endswith(
        " was not scheduled (worker_pool_unavailable: no workers). "
        "No project/workspace admission side effects were started."
    )

    ctx = _routing_ctx(tmp_path, monkeypatch, {"status": "rejected", "reason": "admission_rejected"})
    rejected = _published(
        ctx, "promote_chat_to_task",
        lambda: control_routing._promote_chat_to_task(ctx, "build it"),
        owner_delta="A.21",
    )
    assert (rejected.code, rejected.status) == ("LEGACY_BLOCKED", "blocked")
    assert rejected.text.startswith("PROMOTE_REJECTED: task ")
    assert rejected.text.endswith(
        " was not scheduled (admission_rejected). Do not report this task as created."
    )

    unconfirmed_ctx = _routing_ctx(tmp_path, monkeypatch, {})
    unconfirmed = _published(
        unconfirmed_ctx, "promote_chat_to_task",
        lambda: control_routing._promote_chat_to_task(unconfirmed_ctx, "build it"),
        owner_delta="A.21",
    )
    assert (unconfirmed.code, unconfirmed.status) == ("LEGACY_UNAVAILABLE", "unavailable")
    assert unconfirmed.text.startswith("PROMOTE_UNCONFIRMED: task ")
    assert unconfirmed.text.endswith(
        f" admission was not confirmed within {int(_PROMOTE_CONFIRM_TIMEOUT_SEC)} seconds. "
        "Do not report this task as created and do not retry automatically; keep this "
        "task id for reconciliation."
    )


def test_a_project_route_that_dispatched_nothing_is_not_a_route(tmp_path, monkeypatch):
    """Owner item A.21, the same fix on the project-routing receipts."""
    import ouroboros.projects_registry as projects_registry

    manual_ctx = _routing_ctx(tmp_path, monkeypatch, {"status": "needs_manual_target"})
    manual = _published(
        manual_ctx, "route_to_project",
        lambda: control_routing._route_to_project(manual_ctx, message="continue there"),
        owner_delta="A.21",
    )
    assert (manual.code, manual.status) == ("LEGACY_BLOCKED", "blocked")
    assert manual.text == (
        "⚠️ NEEDS_MANUAL_TARGET (target_unspecified, live): no route was dispatched. "
        "Host-validated options: []"
    )

    silent_ctx = _routing_ctx(tmp_path, monkeypatch, {}, mode="deferred")
    silent = _published(
        silent_ctx, "route_to_project",
        lambda: control_routing._route_to_project(silent_ctx, message="continue there"),
        owner_delta="A.21",
    )
    assert (silent.code, silent.status) == ("LEGACY_UNAVAILABLE", "unavailable")
    assert silent.text == (
        "⚠️ ROUTING_UNCONFIRMED (target_unspecified, deferred): no route was dispatched and "
        "delivery of the manual target options was not confirmed."
    )

    monkeypatch.setattr(projects_registry, "get_project", lambda _root, _pid: {"name": "Dinos"})
    rejected_ctx = _routing_ctx(tmp_path, monkeypatch, {"status": "rejected", "reason": "target_not_found"})
    rejected = _published(
        rejected_ctx, "route_to_project",
        lambda: control_routing._route_to_project(rejected_ctx, project_id="dinos", message="go on"),
        owner_delta="A.21",
    )
    assert (rejected.code, rejected.status) == ("LEGACY_BLOCKED", "blocked")
    assert rejected.text.startswith("⚠️ ROUTE_REJECTED: task ")
    assert rejected.text.endswith(" was not routed to project 'Dinos' (target_not_found).")

    unconfirmed_ctx = _routing_ctx(tmp_path, monkeypatch, {})
    unconfirmed = _published(
        unconfirmed_ctx, "route_to_project",
        lambda: control_routing._route_to_project(unconfirmed_ctx, project_id="dinos", message="go on"),
        owner_delta="A.21",
    )
    assert (unconfirmed.code, unconfirmed.status) == ("LEGACY_UNAVAILABLE", "unavailable")
    assert unconfirmed.text.startswith("⚠️ ROUTE_UNCONFIRMED: task ")
    assert unconfirmed.text.endswith(
        " routing to project 'Dinos' was not durably confirmed. Do not report it as "
        "routed and do not retry automatically."
    )


def test_a_steer_that_delivered_nothing_is_not_a_delivery(tmp_path, monkeypatch):
    """Owner item A.21: a declined steer and an unconfirmed one both said `ok`."""
    rejected_ctx = _routing_ctx(tmp_path, monkeypatch, {"status": "rejected"})
    rejected = _published(
        rejected_ctx, "steer_task",
        lambda: control_routing._steer_task(rejected_ctx, "abc123", "hurry"),
        owner_delta="A.21",
    )
    assert (rejected.code, rejected.status) == ("LEGACY_BLOCKED", "blocked")
    assert rejected.text == "⚠️ STEER_REJECTED: task abc123 was not steered (target_not_steerable)."

    unconfirmed_ctx = _routing_ctx(tmp_path, monkeypatch, {}, mode="deferred")
    unconfirmed = _published(
        unconfirmed_ctx, "steer_task",
        lambda: control_routing._steer_task(unconfirmed_ctx, "abc123", "hurry"),
        owner_delta="A.21",
    )
    assert (unconfirmed.code, unconfirmed.status) == ("LEGACY_UNAVAILABLE", "unavailable")
    assert unconfirmed.text == (
        "⚠️ STEER_UNCONFIRMED: mailbox delivery to task abc123 was not durably confirmed "
        "(deferred). Do not report the message as delivered."
    )


@pytest.mark.parametrize("verb", ["route_to_project", "steer_task"])
def test_a_swarm_scope_denial_is_a_denial(tmp_path, monkeypatch, verb):
    """Owner item A.21: both Swarm scope refusals reported `ok`.

    Their identifiers end in neither `_BLOCKED` nor `_ERROR`, so the adapter read
    them as ordinary warnings and a turn that was refused a route or a steer looked
    like a turn that had taken one.
    """
    ctx = _ctx(tmp_path)
    ctx.project_id = "dinos"
    monkeypatch.setattr(control_routing, "swarm_router_turn", lambda _ctx: True)
    calls = {
        "route_to_project": lambda: control_routing._route_to_project(ctx, message="continue"),
        "steer_task": lambda: control_routing._steer_task(ctx, "abc123", "hurry"),
    }
    expected = {
        "route_to_project": (
            "⚠️ SWARM_PROJECT_SCOPE_OWNED: this Project-room Swarm must create its new "
            "root with promote_chat_to_task in the current Project."
        ),
        "steer_task": (
            "⚠️ SWARM_NEW_ROOT_REQUIRED: explicit Swarm cannot steer an existing task; "
            "use promote_chat_to_task or, from Main, route_to_project."
        ),
    }

    published = _published(ctx, verb, calls[verb], owner_delta="A.21")

    assert (published.code, published.status) == ("ACCESS_BLOCKED", "blocked")
    assert published.text == expected[verb]


def test_the_project_listing_failure_names_the_tool_error_it_is(tmp_path, monkeypatch):
    """The registry vocabulary's own `TOOL_ERROR`, not the legacy text fallback.

    This one carries no differential row on purpose: `LEGACY_TOOL_ERROR` and
    `TOOL_ERROR` share the `error` bucket, so the observable classification does not
    move and an APPROVED_DELTAS row for it would fail the table's own staleness
    direction. What changes is which code the trace records.
    """
    import ouroboros.projects_registry as projects_registry

    def _boom(*_args, **_kwargs):
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(projects_registry, "projects_summary", _boom)
    ctx = _ctx(tmp_path)

    published = _published(
        ctx, "list_projects", lambda: control_routing._list_projects(ctx), owner_delta="A.21")

    assert (published.code, published.status) == ("TOOL_ERROR", "error")
    assert published.text == "⚠️ PROJECTS_ERROR: RuntimeError: registry unreadable"
    # Same bucket on both sides of the change: the differential cannot see it.
    from ouroboros.tools.tool_result import TOOL_CODE_SPECS

    assert TOOL_CODE_SPECS["TOOL_ERROR"].outcome_bucket == (
        TOOL_CODE_SPECS["LEGACY_TOOL_ERROR"].outcome_bucket
    )


# --- Table 2 / owner item A.21: the remaining control refusals ---


def test_a_memory_write_refused_for_its_argument_is_an_argument_error(tmp_path):
    """Owner item A.21: the `REJECTED` identifier ends in none of the suffixes the
    family chain reads, so a scratchpad or identity write refused for a malformed
    argument answered `ok` — the one answer that says the arguments were fine."""
    ctx = _ctx(tmp_path)

    for tool, call, tail in (
        ("update_scratchpad", lambda: control_runtime._update_scratchpad(ctx, "short"),
         "Scratchpad must have meaningful content (10+ chars). "
         "This likely means the tool call was malformed — check your arguments."),
        ("update_identity", lambda: control_runtime._update_identity(ctx, "too short to be identity"),
         "Identity must be a substantial text (50+ chars). "
         "This likely means the tool call was malformed — check your arguments."),
    ):
        published = _published(ctx, tool, call, owner_delta="A.21")
        assert (published.code, published.status) == ("TOOL_ARG_ERROR", "error")
        assert published.text.startswith("⚠️ REJECTED: content is empty or too short (got str, len=")
        assert published.text.endswith(tail)


def test_a_scratchpad_that_needs_a_manual_upgrade_refuses_the_append(tmp_path, monkeypatch):
    """Owner item A.21: the refusal reported `ok` while appending nothing."""
    import ouroboros.memory as memory

    message = (
        "LEGACY_SCRATCHPAD_REQUIRES_MANUAL_UPGRADE: "
        "memory/scratchpad.md exists without scratchpad_blocks.json. "
        "Move preserved notes manually before appending new scratchpad blocks."
    )

    def _refuse(self, *_args, **_kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr(memory.Memory, "append_scratchpad_block", _refuse)
    ctx = _ctx(tmp_path)

    published = _published(
        ctx, "update_scratchpad",
        lambda: control_runtime._update_scratchpad(ctx, "a genuinely long enough note"),
        owner_delta="A.21",
    )

    assert (published.code, published.status) == ("LEGACY_BLOCKED", "blocked")
    assert published.text == f"⚠️ {message}"


@pytest.mark.parametrize(
    ("label", "tool", "text"),
    [
        ("no_chat", "send_user_message", "⚠️ No active chat — cannot send proactive message."),
        ("empty", "send_user_message", "⚠️ Empty message."),
    ],
)
def test_a_proactive_message_that_queued_nothing_is_not_a_message(tmp_path, label, tool, text):
    """Owner item A.21: neither sentence carries an identifier, so both said `ok`."""
    ctx = _ctx(tmp_path)
    if label == "empty":
        ctx.current_chat_id = 7
    calls = {
        "no_chat": lambda: control_runtime._send_user_message(ctx, "hello"),
        "empty": lambda: control_runtime._send_user_message(ctx, "   "),
    }

    published = _published(ctx, tool, calls[label], owner_delta="A.21")

    assert (published.code, published.status) == ("TOOL_ARG_ERROR", "error")
    assert published.text == text
    assert ctx.pending_events == []


def test_an_unknown_model_switches_nothing(tmp_path, monkeypatch):
    """Owner item A.21: the refusal named no identifier and reported `ok`."""
    import ouroboros.llm as llm

    monkeypatch.setattr(
        llm.LLMClient, "available_models", lambda _self: ["gpt-5.6-luna", "sonnet-4.6"])
    ctx = _ctx(tmp_path)

    published = _published(
        ctx, "switch_model", lambda: control_runtime._switch_model(ctx, model="gpt-9"),
        owner_delta="A.21",
    )

    assert (published.code, published.status) == ("TOOL_ARG_ERROR", "error")
    assert published.text == "⚠️ Unknown model: gpt-9. Available: gpt-5.6-luna, sonnet-4.6"
    assert getattr(ctx, "active_model_override", "") in ("", None)


def test_a_deep_self_review_nobody_can_run_is_unavailable(tmp_path, monkeypatch):
    """Owner item A.21: the notice reported `ok` and queued no review."""
    import ouroboros.deep_self_review as deep_self_review

    monkeypatch.setattr(deep_self_review, "is_review_available", lambda: (False, ""))
    ctx = _ctx(tmp_path)

    published = _published(
        ctx, "request_deep_self_review",
        lambda: control_runtime._request_deep_self_review(ctx, "audit myself"),
        owner_delta="A.21",
    )

    assert (published.code, published.status) == ("CAPABILITY_UNAVAILABLE", "unavailable")
    assert published.text == (
        "❌ Deep self-review unavailable: configure OUROBOROS_MODEL_DEEP_SELF_REVIEW "
        "and the matching provider API key."
    )
    assert ctx.pending_events == []


def test_a_child_beyond_the_depth_limit_is_a_resource_refusal(tmp_path, monkeypatch):
    """Owner item A.21: the depth refusal reported `ok` and scheduled nothing.

    The limit is a configured budget on the tree, not a malformed argument, so it is
    the constraint code rather than the argument one the sibling refusals publish.
    """
    monkeypatch.setattr(control_scheduling, "get_max_subagent_depth", lambda: 3)
    ctx = _ctx(tmp_path)
    ctx.task_depth = 3

    published = _published(
        ctx, "schedule_subagent",
        lambda: control_scheduling._schedule_task(ctx, objective="o", expected_output="e"),
        owner_delta="A.21",
    )

    assert (published.code, published.status) == ("RESOURCE_CONSTRAINT_BLOCKED", "blocked")
    assert published.text == "ERROR: Subtask depth limit (3) exceeded. Simplify your approach."


def test_a_capability_mismatch_is_the_argument_error_its_remedy_describes(tmp_path):
    """Owner item A.21, and the choice the owner table left to the adapter's evidence.

    Both inputs are arguments of THIS call — `required_capabilities` and the surface
    implied by `write_surface` — and the message's own remedy is to change one of
    them, exactly like the malformed-`required_capabilities` refusal a few lines
    above it, which already publishes `TOOL_ARG_ERROR`. Nothing in the environment
    constrains the spawn, so `RESOURCE_CONSTRAINT_BLOCKED` ("use a resource the task
    contract allows") would name a constraint that does not exist here.
    """
    ctx = _ctx(tmp_path)

    published = _published(
        ctx, "schedule_subagent",
        lambda: control_scheduling._schedule_task(
            ctx, objective="o", expected_output="e", required_capabilities=["shell"]),
        owner_delta="A.21",
    )

    assert (published.code, published.status) == ("TOOL_ARG_ERROR", "error")
    assert published.text.startswith(
        "⚠️ SUBAGENT_CAPABILITY_MISMATCH: selected child profile 'local_readonly_subagent' "
        "cannot satisfy required_capabilities=['shell']. These need an ACTING child: "
    )


def test_an_id_this_tree_never_registered_has_no_result_to_read(tmp_path):
    """Owner item A.21: the read reported `ok` for a task it could not find."""
    ctx = _ctx(tmp_path)

    published = _published(
        ctx, "get_task_result",
        lambda: control_task_results._get_task_result(ctx, "4f2a1c"),
        owner_delta="A.21",
    )

    assert (published.code, published.status) == ("LEGACY_UNAVAILABLE", "unavailable")
    assert published.text == "Task 4f2a1c: unknown or not yet registered"


def test_a_wait_that_embeds_the_unknown_read_keeps_the_wait_result(tmp_path):
    """The embedded read publishes, but the wait returns a LONGER string.

    The registry accepts a published result only when its text is exactly what the
    handler returned, so the wait's own answer is never replaced by the read's
    `unavailable` — the guard that keeps a helper's publication from escaping its
    caller, asserted rather than assumed.
    """
    ctx = _ctx(tmp_path)
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        text = control_task_results._wait_for_task(ctx, "4f2a1c", timeout_sec=0)
        published = _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)

    assert text.startswith("Task wait timed out after ")
    assert text.endswith("Task 4f2a1c: unknown or not yet registered")
    assert isinstance(published, ToolResult) and published.text != text


def test_the_wait_set_cap_refusal_names_the_configured_cap(tmp_path):
    from ouroboros.config import MAX_ACTIVE_SUBAGENTS_HARD_CAP

    ctx = _ctx(tmp_path)
    oversized = [f"t{index}" for index in range(MAX_ACTIVE_SUBAGENTS_HARD_CAP + 1)]

    published = _published(
        ctx, "wait_tasks", lambda: control_task_results._wait_for_tasks(ctx, oversized))

    assert published.code == "TOOL_ARG_ERROR"
    assert published.text == (
        "⚠️ TOOL_ARG_ERROR (wait_tasks): task_ids is capped at "
        f"{MAX_ACTIVE_SUBAGENTS_HARD_CAP}."
    )
