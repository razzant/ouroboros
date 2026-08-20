"""S6 C7-C10 — structural inventories of the cancellation protocol's owners.

Four properties the protocol relies on that no test asserted structurally, so a
relocation could grow or move one of them silently:

- C7 the set of call sites that write a TERMINAL task status;
- C8 the set of ``settle_intent`` call sites, and the single one allowed to
  settle a cascade scope;
- C9 owed-before-settle as the TWO contracts actually implemented, plus the
  enumerated lanes that deliberately owe nothing;
- C10 the split-drive root: every cancel ingress writes its intent to the root
  the watchdog reads.

The enumeration walks ``ouroboros/`` and ``supervisor/`` by SYMBOL, never by a
hard-coded module list, so a module split moves rows around inside the manifests
below instead of hiding a call site from them. The manifests are data at the top
of the module: an extraction updates the path in a row, and a NEW writer has to
add a row, which is the point.
"""

from __future__ import annotations

import ast
import pathlib
import types

import pytest


REPO = pathlib.Path(__file__).resolve().parents[1]

# Statuses that are NOT terminal: the reducer ranks them below the sticky set,
# so a writer that only ever passes one of these cannot end a task.
_NON_TERMINAL_STATUS_EXPRESSIONS = frozenset({
    "STATUS_RUNNING", "STATUS_SCHEDULED", "STATUS_REQUESTED", "STATUS_INTERRUPTED",
    "STATUS_CANCEL_REQUESTED",
})
_TERMINAL_TOKENS = (
    "STATUS_CANCELLED", "STATUS_FAILED", "STATUS_COMPLETED",
    "STATUS_REJECTED_DUPLICATE", '"failed"', '"cancelled"', '"completed"',
)

# C7 — every call site that can write a terminal status, keyed by
# (path::qualname, the status EXPRESSION as written). "terminal" names a
# constant from the sticky set; "dynamic" is a variable or expression that can
# carry one, which counts because the reducer, not the caller, decides.
TERMINAL_WRITERS = {
    ('ouroboros/agent.py::OuroborosAgent._handle_task_scoped', 'STATUS_FAILED'): 'terminal',
    ('ouroboros/agent_task_pipeline.py::_store_task_result', 'status'): 'dynamic',
    ('ouroboros/agent_task_pipeline.py::recover_pending_root_post_task_synthesis', 'str(task.get("status") or STATUS_COMPLETED)'): 'terminal',
    ('ouroboros/gateway/tasks.py::_admission_rejection_response', 'STATUS_FAILED'): 'terminal',
    ('ouroboros/gateway/tasks.py::_complete_api_task_admission', '"failed"'): 'terminal',
    ('ouroboros/headless.py::copy_child_task_result', 'child_status'): 'dynamic',
    ('ouroboros/headless.py::finalize_task_artifacts', 'status'): 'dynamic',
    ('ouroboros/headless.py::finalize_task_artifacts', 'str(existing.get("status") or status or "completed")'): 'terminal',
    ('ouroboros/mutation_attribution.py::advance_mutation_baseline', 'status'): 'dynamic',
    ('ouroboros/mutation_attribution.py::capture_mutation_baseline', 'status'): 'dynamic',
    ('ouroboros/mutation_attribution.py::record_terminal_mutation_candidates', 'status'): 'dynamic',
    ('ouroboros/post_task_checkpoint.py::set_root_post_task_checkpoint', 'str(existing.get("status") or task.get("status") or STATUS_COMPLETED)'): 'terminal',
    ('ouroboros/project_naming.py::spawn_proactive_namer._work', 'status'): 'dynamic',
    ('ouroboros/task_results.py::fail_tasks', 'STATUS_CANCELLED'): 'terminal',
    ('ouroboros/task_results.py::fail_tasks', 'STATUS_FAILED'): 'terminal',
    ('ouroboros/task_status.py::reconcile_orphaned_running_tasks', 'eff_status'): 'dynamic',
    ('supervisor/events_task_done.py::_finish_task_done_dispatch', 'STATUS_FAILED'): 'terminal',
    ('supervisor/events_task_done.py::_handle_task_done', 'str(existing.get("status") or "")'): 'dynamic',
    ('supervisor/events_project_routing.py::_persist_promote_rejection', 'STATUS_FAILED'): 'terminal',
    ('supervisor/events_schedule_task.py::_reject_schedule_task', 'status'): 'dynamic',
    ('supervisor/events_task_done.py::_resolve_lifecycle_fault', 'STATUS_FAILED'): 'terminal',
    ('supervisor/queue_snapshot.py::restore_pending_from_snapshot', 'STATUS_CANCELLED'): 'terminal',
    ('supervisor/cancel_custody.py::_finalize_cancel_intent_on_miss', 'STATUS_CANCELLED'): 'terminal',
    ('supervisor/cancel_custody.py::_finish_captured_pending', 'STATUS_CANCELLED'): 'terminal',
    ('supervisor/cancel_custody.py::_finish_captured_running', 'STATUS_CANCELLED'): 'terminal',
    ('supervisor/task_lifecycle.py::record_scheduled_admission', 'STATUS_FAILED'): 'terminal',
    ('supervisor/task_reaper.py::_enqueue_retry', 'STATUS_FAILED'): 'terminal',
    ('supervisor/task_reaper.py::reap_timed_out_task', 'STATUS_INTERRUPTED if will_retry else STATUS_FAILED'): 'terminal',
    ('supervisor/worker_assignment.py::_cancel_unauthorized_evolution', 'STATUS_CANCELLED'): 'terminal',
    ('supervisor/workers.py::_drop_cancelled_pending', 'STATUS_CANCELLED'): 'terminal',
    ('supervisor/worker_health.py::_ensure_workers_healthy_locked', 'STATUS_CANCELLED'): 'terminal',
    ('supervisor/worker_health.py::_ensure_workers_healthy_locked', 'STATUS_FAILED'): 'terminal',
    ('supervisor/worker_promotion.py::_fail_promoted_task_loudly', 'STATUS_FAILED'): 'terminal',
    ('supervisor/worker_pool_lifecycle.py::_write_failure_result', 'final_status'): 'dynamic',
    ('supervisor/worker_assignment.py::assign_tasks', 'STATUS_CANCELLED'): 'terminal',
    ('supervisor/worker_assignment.py::assign_tasks', 'STATUS_FAILED'): 'terminal',
}

# C8 — the settle owners. Exactly one may pass allow_cascade_scope=True: the
# cascade postcondition, which owes the tree's one summary before it settles.
SETTLE_INTENT_CALLERS = {
    'ouroboros/task_results.py::fail_tasks': False,
    'supervisor/cancel_custody.py::_settle_intent': False,
    'supervisor/task_lifecycle.py::cancel_task_by_id': True,
    'supervisor/workers.py::_drop_cancelled_pending': False,
}

# C9 — lanes that terminalize a task and deliberately register NOTHING as owed,
# each with the reason there is nothing to deliver.
NO_DELIVERABLE_LANES = {
    'supervisor/cancel_custody.py::_finish_captured_pending':
        'cancelled before it ever started: no answer exists',
    'supervisor/task_lifecycle.py::record_scheduled_admission':
        'a cron dispatch refused at admission never had an owner answer',
    'supervisor/workers.py::_drop_cancelled_pending':
        'dropped before assignment; the salvage receipt belongs to custody',
    'ouroboros/task_results.py::fail_tasks':
        'budget drain before start',
    'supervisor/queue_snapshot.py::restore_pending_from_snapshot':
        'restore-time reconciliation of a task cancelled while the server was down',
    'supervisor/events_task_done.py::_finish_task_done_dispatch':
        'lifecycle fault: the durable row, not a message, is the disclosure',
    'supervisor/events_task_done.py::_resolve_lifecycle_fault':
        'same fault class, resolved into a terminal',
    'supervisor/events_schedule_task.py::_reject_schedule_task':
        'a refused schedule never became a task with an answer',
    'supervisor/events_project_routing.py::_persist_promote_rejection':
        'a refused promotion never became a task with an answer',
}

_OWE_CALLS = ("register_final_answer_owed", "register_pending_delivery",
              "_register_owed_terminal_delivery", "enqueue_terminal_delivery")


# ---------------------------------------------------------------------------
# Enumeration by symbol, never by module list
# ---------------------------------------------------------------------------


def _sources():
    paths = sorted(REPO.glob("*.py"))
    for directory in ("ouroboros", "supervisor"):
        paths.extend(sorted((REPO / directory).rglob("*.py")))
    return paths


def _calls(source: str, target: str):
    """Every call to ``target`` with the enclosing lexical qualname."""
    stack: list[str] = []
    found: list[tuple[str, ast.Call]] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_Call(self, node):
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else ""
            )
            if name == target:
                found.append((".".join(stack) or "<module>", node))
            self.generic_visit(node)

    Visitor().visit(ast.parse(source))
    return found


def _call_sites(target: str):
    for path in _sources():
        source = path.read_text(encoding="utf-8")
        if target not in source:
            continue
        for qualname, node in _calls(source, target):
            yield path, source, f"{path.relative_to(REPO).as_posix()}::{qualname}", node


def _expression(source: str, node: ast.Call, index: int, keyword: str) -> str:
    if len(node.args) > index:
        return " ".join((ast.get_source_segment(source, node.args[index]) or "").split())
    for kw in node.keywords:
        if kw.arg == keyword:
            return " ".join((ast.get_source_segment(source, kw.value) or "").split())
    return ""


def _function_source(path: pathlib.Path, qualname: str) -> str:
    """The source of one function, addressed by its lexical qualname."""
    source = path.read_text(encoding="utf-8")
    wanted = qualname.split(".")
    node: ast.AST = ast.parse(source)
    for name in wanted:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ) and child.name == name:
                node = child
                break
        else:  # pragma: no cover - a missing qualname fails the caller's assert
            return ""
    return ast.get_source_segment(source, node) or ""


# ---------------------------------------------------------------------------
# C7 — terminal-writer inventory
# ---------------------------------------------------------------------------


def test_c7_the_set_of_terminal_status_writers_equals_the_manifest():
    """C7: ~24 writers reach the ONE terminal primitive, and nine of them
    discard its return value, so none of them can know it lost a race. The
    monotonic reducer under the per-file lock is what makes that safe. Nothing
    asserted the writer SET, so an extraction could add writer #37 unnoticed —
    this is that assertion.
    """
    live: dict[tuple[str, str], str] = {}
    for _path, source, identity, node in _call_sites("write_task_result"):
        expression = _expression(source, node, 2, "status")
        if expression in _NON_TERMINAL_STATUS_EXPRESSIONS:
            continue
        live[(identity, expression)] = (
            "terminal" if any(token in expression for token in _TERMINAL_TOKENS)
            else "dynamic"
        )

    added = sorted(set(live) - set(TERMINAL_WRITERS))
    removed = sorted(set(TERMINAL_WRITERS) - set(live))
    assert not added, (
        "a NEW terminal-status writer appeared; add it to TERMINAL_WRITERS with "
        f"its reason, or route it through an existing owner: {added}"
    )
    assert not removed, (
        "a terminal-status writer named in TERMINAL_WRITERS is gone (moved or "
        f"retired) — update the manifest row in the same commit: {removed}"
    )
    assert live == TERMINAL_WRITERS


def test_c7_the_one_terminal_write_primitive_is_still_one():
    """The property the manifest is guarding: there is a single durable writer
    with the monotonic reducer inside it, not a family of them."""
    from ouroboros import task_results

    source = (REPO / "ouroboros" / "task_results.py").read_text(encoding="utf-8")
    assert source.count("def write_task_result(") == 1
    assert callable(task_results.write_task_result)
    body = _function_source(REPO / "ouroboros" / "task_results.py", "write_task_result")
    assert "update_json_locked" in body, "the write holds the per-file lock"
    assert "_is_status_regression" in body, "the reducer is inside the lock, not at callers"


# ---------------------------------------------------------------------------
# C8 — settle-owner inventory
# ---------------------------------------------------------------------------


def test_c8_settle_intent_has_exactly_the_documented_callers():
    """C8: the intent settle is claimed to have four owners, exactly one of
    which may settle a `scope=cascade` row. Both halves are asserted, because
    a naive relocation that adds a fifth caller — or copies the cascade flag —
    breaks the exclusivity the cascade summary depends on."""
    live: dict[str, bool] = {}
    for _path, source, identity, node in _call_sites("settle_intent"):
        if identity.startswith("ouroboros/cancel_intents.py::"):
            continue  # the definition module: its own name, not a call site
        live[identity] = (
            _expression(source, node, 99, "allow_cascade_scope") == "True"
        )

    assert live == SETTLE_INTENT_CALLERS, live
    assert sum(1 for flag in live.values() if flag) == 1, (
        "exactly one caller may settle a cascade scope: the postcondition"
    )
    assert [identity for identity, flag in live.items() if flag] == [
        "supervisor/task_lifecycle.py::cancel_task_by_id",
    ]


def test_c8_the_cascade_scope_guard_is_enforced_inside_the_locked_mutate():
    """Why the count above is sufficient: the refusal is atomic against the
    CURRENT durable row, so a stale claim snapshot cannot settle a scope that
    was widened mid-flight — the caller list is a guard, not the mechanism."""
    body = _function_source(REPO / "ouroboros" / "cancel_intents.py", "settle_intent")
    mutate = body[body.index("def _mutate("):]
    assert "allow_cascade_scope" in mutate and "SCOPE_CASCADE" in mutate


# ---------------------------------------------------------------------------
# C9 — owed-before-settle, restated as the two contracts implemented
# ---------------------------------------------------------------------------


def test_c9_the_natural_path_owes_before_the_durable_result_write():
    """Contract 1 (already pinned at tests/test_gate_round3_fixes.py:451, kept
    here so the pair reads as one rule): on the ordinary completion path the
    owed row is registered BEFORE the result is persisted, so a crash in the
    window leaves a row the boot replay delivers."""
    body = _function_source(
        REPO / "ouroboros" / "agent_task_pipeline.py", "emit_task_results")
    assert body
    assert body.index("register_final_answer_owed(") < body.index("_store_task_result("), (
        "the owed registration must precede the durable result write"
    )


@pytest.mark.parametrize("module, qualname", [
    ("supervisor/cancel_custody.py", "_finish_captured_running"),
    ("supervisor/cancel_custody.py", "_finalize_cancel_intent_on_miss"),
    ("supervisor/task_lifecycle.py", "cancel_task_by_id"),
])
def test_c9_every_cancelled_settle_is_preceded_by_an_owed_registration(module, qualname):
    """Contract 2: on the cancel lanes the order is write -> owe -> SETTLE.

    The invariant is owed-before-SETTLE, not owed-before-write, and on these
    lanes it is the stronger one: an owed registration that could not be made
    durable leaves the intent OPEN for the watchdog instead of closing the
    cancellation. Checked per SETTLE CALL, and only for the calls that publish a
    real cancellation: a `not_found` settle (the task never existed) and the
    `already_settled` branches have nothing of their own to deliver — the
    already-settled lanes still deliver first, which the ordering below shows.
    """
    source = (REPO / module).read_text(encoding="utf-8")
    owe_lines: list[int] = []
    for call in _OWE_CALLS + ("_deliver_on_miss", "deliver_cascade_summary"):
        owe_lines.extend(node.lineno for qual, node in _calls(source, call)
                         if qual == qualname or qual.endswith(f".{qualname}"))
    settles = [
        (node.lineno, _expression(source, node, 99, "outcome"))
        for target in ("_settle_intent", "_settle_or_reopen_intent", "settle_intent")
        for qual, node in _calls(source, target)
        if qual == qualname or qual.endswith(f".{qualname}")
    ]
    assert settles, f"{module}::{qualname} settles no intent — retarget this row"
    assert owe_lines, f"{module}::{qualname} registers nothing as owed"
    for lineno, outcome in settles:
        if outcome not in ('"cancelled"', "'cancelled'"):
            continue  # not_found / already_settled: nothing of its own to deliver
        assert any(owe < lineno for owe in owe_lines), (
            f"{module}::{qualname}:{lineno} settles a CANCELLED outcome with no "
            "owed registration before it"
        )


def test_c9_the_registration_failure_rule_is_the_one_shared_helper():
    """The uniform GR4-1 rule behind contract 2: an owed row that could not be
    written releases the claim and leaves the intent OPEN, in ONE helper."""
    body = _function_source(
        REPO / "supervisor" / "cancel_publication.py", "_settle_or_reopen_intent")
    assert body
    assert "_release_intent_claim" in body and "_settle_intent" in body, (
        "the helper owns both outcomes: settle when owed, reopen when not"
    )
    callers = {
        identity for _p, _s, identity, _n in _call_sites("_settle_or_reopen_intent")
        if not identity.startswith("supervisor/cancel_publication.py::_settle_or_reopen")
    }
    assert callers, "the helper is actually used"
    assert all(
        identity.startswith("supervisor/cancel_custody.py::") for identity in callers
    ), callers


def test_c9_the_lanes_that_owe_nothing_are_an_enumerated_list():
    """C9's third part: "owed before settle" does not mean "every terminal owes
    something". These lanes terminalize a task with nothing to deliver, by
    design — the list is checked in so a lane that STARTS owing something, or a
    new silent lane, shows up as a diff."""
    for identity, reason in NO_DELIVERABLE_LANES.items():
        module, qualname = identity.split("::")
        body = _function_source(REPO / module, qualname)
        assert body, f"{identity} not found — retarget this row"
        assert reason
        present = [call for call in _OWE_CALLS if f"{call}(" in body]
        assert present == [], (
            f"{identity} now registers an owed delivery ({present}); either it "
            "gained a deliverable answer (move it out of this list and pin the "
            "ordering) or the call is misplaced"
        )


# ---------------------------------------------------------------------------
# C10 — split-drive root inventory
# ---------------------------------------------------------------------------


def _sweep_sees(tmp_path, task_id: str, monkeypatch) -> bool:
    """Whether the watchdog, reading its own canonical root, sees the intent."""
    from ouroboros.cancel_intents import active_intents
    from supervisor import queue as q

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path, raising=False)
    return task_id in active_intents(q.DRIVE_ROOT)


def _agent_ctx(canonical, child_drive, *, carry_budget_root: bool):
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=canonical, drive_root=child_drive or canonical)
    ctx.task_id = "parent-1"
    ctx.task_metadata = {"root_task_id": "parent-1", "parent_task_id": "parent-1"}
    if carry_budget_root:
        ctx.task_metadata["budget_drive_root"] = str(canonical)
    return ctx


def test_c10_the_agent_tool_writes_a_split_drive_intent_at_the_canonical_root(
    tmp_path, monkeypatch,
):
    """C10: the agent `cancel_task` tool resolves
    `metadata["budget_drive_root"] or ctx.budget_drive_root or ctx.drive_root`.
    For a split-drive child the FIRST term is the canonical supervisor root, so
    the intent lands where `sweep_cancel_intents` reads — the divergence a
    reviewer suspected in the third fallback does not exist here."""
    from ouroboros.tools.join_ledger import _cancel_task

    canonical = tmp_path / "data"
    child = tmp_path / "task_drives" / "child"
    child.mkdir(parents=True)
    ctx = _agent_ctx(canonical, child, carry_budget_root=True)

    out = _cancel_task(ctx, "victim-1", reason="stop")

    assert "Cancel requested" in out, out
    assert (canonical / "state" / "cancel_intents.json").is_file()
    assert not (child / "state" / "cancel_intents.json").exists()
    assert _sweep_sees(canonical, "victim-1", monkeypatch)


def test_c10_a_task_without_a_child_drive_resolves_to_the_same_root(
    tmp_path, monkeypatch,
):
    """C10: the third fallback (`ctx.drive_root`) is reached only when NO child
    drive exists — and then `drive_root` IS the canonical root, so the two
    shapes agree. The pairing is what holds the invariant, asserted below."""
    from ouroboros.tools.join_ledger import _cancel_task

    canonical = tmp_path / "data"
    canonical.mkdir()
    ctx = _agent_ctx(canonical, None, carry_budget_root=False)

    assert "Cancel requested" in _cancel_task(ctx, "victim-2", reason="stop")
    assert _sweep_sees(canonical, "victim-2", monkeypatch)


def test_c10_a_child_drive_is_never_set_without_its_budget_drive_root():
    """C10, the invariant the two tests above ride on: every production site
    that points a task's `drive_root` at a child drive sets `budget_drive_root`
    to the canonical root in the same block. A site that set one without the
    other WOULD write intents onto the child drive, where no supervisor reader
    looks."""
    for module in ("ouroboros/gateway/tasks.py", "supervisor/workers.py",
                   "supervisor/events.py"):
        source = (REPO / module).read_text(encoding="utf-8")
        for index, line in enumerate(source.splitlines()):
            if 'task["drive_root"] = str(child_drive' not in line:
                continue
            window = "\n".join(source.splitlines()[index:index + 4])
            assert "budget_drive_root" in window, f"{module}:{index + 1}\n{window}"


def test_c10_the_settled_probe_reads_the_intent_root_not_the_child_drive(tmp_path):
    """C10, the one real divergence — and it is noise, not a wedge. The mint's
    already-settled probe reads the task result at the INTENT root, so a
    split-drive child whose result lives on its own drive reads as unsettled
    and an intent IS minted. Custody then settles it `already_settled`, so the
    cost is one watchdog round trip, not a lost cancellation."""
    from ouroboros.cancel_intents import request_cancel, settled_status
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    canonical = tmp_path / "data"
    child = tmp_path / "task_drives" / "child"
    canonical.mkdir()
    child.mkdir(parents=True)
    write_task_result(child, "done-1", STATUS_COMPLETED, result="finished")

    assert settled_status(child, "done-1") == STATUS_COMPLETED
    assert settled_status(canonical, "done-1") == "", "the probe cannot see it"
    intent = request_cancel(canonical, "done-1", reason="late cancel")
    assert intent.get("already_settled") is False
    assert intent.get("request_id"), "an intent is minted over a settled child"


def test_c10_the_http_and_supervisor_roots_come_from_one_configured_value():
    """C10 for the HTTP and project-deletion ingresses: they mint at the app's
    `state.drive_root`, the watchdog reads `supervisor.queue.DRIVE_ROOT`, and
    both are bound from the ONE configured `DATA_DIR` at startup. Two globals
    that merely happen to agree would be the gap; one source is the answer."""
    server_source = (REPO / "server.py").read_text(encoding="utf-8")
    assert "app.app.state.drive_root = pathlib.Path(DATA_DIR)" in server_source
    assert "DRIVE_ROOT=DATA_DIR" in server_source
    from ouroboros.gateway._helpers import request_drive_root

    app = types.SimpleNamespace(state=types.SimpleNamespace(drive_root="/pinned/root"))
    request = types.SimpleNamespace(app=app)
    assert str(request_drive_root(request)) == "/pinned/root", (
        "the ingress root is whatever the app was bound to, never a re-derivation"
    )
