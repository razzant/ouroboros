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

v7next F3.1 adaptation, disclosed: this lane's sanctioned control rows are
2548/2549/2556/2571/2574/2579 (the six HOT-DEFERRED D02 rows the F2.1 lane cut
in tip form). The reference also typed the OTHER control leaves
(control_routing/control_runtime and the memory/scratchpad/proactive/model
producers); those clauses are NOT carried here and return with their rows.
Tip drift honoured: _schedule_task now runs the configured-subagent roster
gate before the capability checks, so the capability clauses seed a test
roster via tests._shared.configure_test_subagent.
"""

from __future__ import annotations

import pathlib

import pytest

from ouroboros.tools import control_routing, control_scheduling, control_task_results
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
def test_schedule_argument_refusals_publish_their_adapter_code(tmp_path, monkeypatch, label, prefix):
    from tests._shared import configure_test_subagent

    configure_test_subagent(monkeypatch)
    ctx = _ctx(tmp_path)
    calls = {
        "retired_param": lambda: control_scheduling._schedule_task(ctx, effort="high"),
        "unsupported_param": lambda: control_scheduling._schedule_task(ctx, bogus=1),
        "validator_refusal": lambda: control_scheduling._schedule_task(ctx, objective=""),
        "capability_arg_error": lambda: control_scheduling._schedule_task(
            ctx, subagent_id="api-scout", objective="o", expected_output="e",
            required_capabilities="shell"),
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


def test_a_capability_mismatch_is_the_argument_error_its_remedy_describes(tmp_path, monkeypatch):
    """Owner item A.21, and the choice the owner table left to the adapter's evidence.

    Both inputs are arguments of THIS call — `required_capabilities` and the surface
    implied by `write_surface` — and the message's own remedy is to change one of
    them, exactly like the malformed-`required_capabilities` refusal a few lines
    above it, which already publishes `TOOL_ARG_ERROR`. Nothing in the environment
    constrains the spawn, so `RESOURCE_CONSTRAINT_BLOCKED` ("use a resource the task
    contract allows") would name a constraint that does not exist here.
    """
    from tests._shared import configure_test_subagent

    configure_test_subagent(monkeypatch)
    ctx = _ctx(tmp_path)

    published = _published(
        ctx, "schedule_subagent",
        lambda: control_scheduling._schedule_task(
            ctx, subagent_id="api-scout", objective="o", expected_output="e",
            required_capabilities=["shell"]),
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
