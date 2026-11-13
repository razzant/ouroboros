"""A task's own explicit requirement of the whole handbook, as its real consumers read it (owner 1.11.2026).

``start_task(context_requires_development=true)`` is the one producer: the whole composed book joins any mode as its
own stable block, beside the view the same capture would get without it. A subject marker, an older top-level answer,
a parent's answer and absence never ask for it. The book is charged to the window and to nothing else: no form, mode,
schema, operative input or unasked tool output pays for it through a target or a rescue, and a window that physically
cannot hold it is a named task outcome. Unknown capacity proves nothing. Paired runs hold density fixed.
"""
from __future__ import annotations

import json
import math
import queue
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import context, loop, memory_view
from ouroboros import context_fit as fit
from ouroboros.context_input_selection import development_required, development_resident
from ouroboros.context_measurement import measure_main_fit, required_book_addition
from ouroboros.contracts.task_contract import attach_task_contract, build_task_contract
from ouroboros.loop_model_call import _measure_main_context_view
from ouroboros.tool_result_delivery import project_tool_result_batch
from tests import _memory_view_synthetic as syn
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_book_context_capture import _capture, _text
from tests.test_book_context_capture import _plan as _capture_plan
from tests.test_doc_context import foreign_tree
from tests.test_llm_claudexor import setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_owner_continue import NONCE, _interrupted
from tests.test_root_effort_ingress import _promoted
from tests.test_subscription_main_wait import main_call as main_fixture

setup, live_wait, main_call = gateway_fixture, wait_fixture, main_fixture
KEY = "context_requires_development"
RESERVE = 65_536
RATIO = 1.25
SCHEMAS = [{"type": "function", "function": {"name": name, "description": "d", "parameters": {"type": "object"}}}
           for name in ("read_file", "enable_tools", "compact_context", "list_available_tools", *(f"x{i}" for i in range(40)))]


@pytest.fixture(autouse=True)
def _isolated_projects_root(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path_factory.mktemp("projects_root")))


@pytest.fixture
def fixed_density(monkeypatch):
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: RESERVE)
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *_a, **_kw: RATIO)
    monkeypatch.setattr("ouroboros.capability_evidence.resolve_main_token_density",
                        lambda *_a, **_kw: (RATIO, "fresh_route_usage"))


def _route(window, known=True):
    evidence = SimpleNamespace(route_fp="route", status="confirmed" if known else "unprobeable",
                               stale=False, window_tokens=window)
    return lambda *_a, **_kw: ({"provider": "fixture", "model": "fixture"}, evidence)


def _core(root, *, required=False, subject=False, book_tokens=40_000, assignment="Do the work.", memory=True,
          arch_words=40_000, page_chars=16_000):
    """A real capture shape over synthetic memory: pages, rows and task lines the floor can walk."""
    lane = [*(syn.spoken(i, "ouroboros", 8_000) for i in range(2, 8)),
            {**syn.spoken(1, "human", 200), "owner_side": True, "immune": True}]
    snapshot = syn.snapshot(room_facts=syn.room("1", lane1=lane, lane2=6, legacy=4, legacy_chars=8_000),
                            story=syn.pages(12, page_chars), conversations=[syn.conversation(i) for i in range(4)], mark_count=2)
    return fit.ContextCore(
        base_prompt="SYSTEM", bible_md="BIBLE", architecture_md="# Architecture\n\n" + "body " * arch_words,
        development_md="# Development\n\n## Review\n\n" + "handbook " * (book_tokens * 4 // 9),
        semi_stable_text="IDENTITY", dynamic_text="", user_content_json=json.dumps(assignment),
        docs_need_development=subject, development_required=required, section_root=str(root), section_task_id="t",
        memory_view_json=memory_view.snapshot_json(snapshot) if memory else "")


def _plan(root, core, preferred, window, *, known=True, task=None):
    return fit.build_context_fit_plan(SimpleNamespace(drive_root=root), core, task or {"id": "t", "type": "task"},
                                      preferred_mode=preferred, tool_schemas=SCHEMAS, route_resolver=_route(window, known))


def _blocks(plan, mode):
    return plan.projection(mode).system_message()["content"]


def _book_blocks(plan, mode):
    return [block["text"] for block in _blocks(plan, mode) if block["text"].startswith("## DEVELOPMENT.md\n")]


def _choices(plan):
    """Everything automatic this start decided: mode, memory forms, host forms and the schemas it sends."""
    from ouroboros.tool_policy import select_tool_schemas

    facts = plan.projection(plan.initial_mode).memory_facts
    floor = facts.get("floor") or {}
    return (plan.initial_mode, floor.get("steps"), floor.get("runs"), floor.get("mode_switch"),
            [row["section_id"] for row in facts.get("host_sections_addressed", ())],
            select_tool_schemas(SCHEMAS, context_mode=plan.initial_mode).chosen,
            json.loads(plan.projection(plan.initial_mode).user_content_json or plan.user_content_json))


# --- Who asks ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("flag", [True, False, None])
def test_start_task_authors_its_answer_on_its_own_contract_and_absence_stays_absent(tmp_path, monkeypatch, flag):
    task, row, _workers = _promoted(tmp_path, monkeypatch, **({} if flag is None else {KEY: flag}))
    for contract in (task["task_contract"], row["task_contract"]):
        assert contract.get(KEY, "absent") == ("absent" if flag is None else flag)
    assert task.get(KEY, "absent") == ("absent" if flag is None else flag)
    assert development_required(task) is (flag is True)


def test_an_older_top_level_answer_keeps_its_max_meaning_and_gains_no_requirement(tmp_path):
    workspace = str(foreign_tree(tmp_path / "foreign"))
    legacy = attach_task_contract({"id": "old", "type": "task", KEY: True, "workspace_root": workspace,
                                   "workspace_mode": "external", "metadata": {}})
    assert KEY not in legacy["task_contract"] and development_required(legacy) is False
    ctx = SimpleNamespace(repo_dir=tmp_path, system_repo_dir=tmp_path, drive_root=tmp_path)
    assert development_resident(legacy, ctx) is True  # Max, as it was described when it was written
    env, core, task = _capture(tmp_path / "capture", legacy)
    assert core.development_required is False
    for mode in ("low", "nano"):
        assert not _book_blocks(_capture_plan(env, core, task, mode), mode)
    assert _book_blocks(_capture_plan(env, core, task, "max"), "max")
    # Its restart loses the row's copy as before; it never promotes itself through the contract.
    from supervisor.queue_snapshot import _PENDING_FIELDS

    assert KEY not in _PENDING_FIELDS
    restored = {key: legacy.get(key) for key in _PENDING_FIELDS}
    assert KEY not in restored["task_contract"] and development_required(restored) is False


@pytest.mark.parametrize(("mode", "answer"), [
    pytest.param("low", {"context_requires_self_body_docs": True}, id='low-answer0'),
    pytest.param("low", {"task_contract": {KEY: False}}, id='low-answer2'),
    pytest.param("low", {}, id='low-answer4'),
    pytest.param("nano", {"task_contract": {"context_requires_self_body_docs": True}}, id='nano-answer1'),
    pytest.param("nano", {KEY: False}, id='nano-answer3'),
])
def test_a_subject_marker_an_explicit_no_or_absence_keeps_the_compact_map(tmp_path, answer, mode):
    env, core, task = _capture(tmp_path, answer)
    plan = _capture_plan(env, core, task, mode)
    assert not core.development_required and not _book_blocks(plan, mode)
    assert "Full engineering discipline." not in _text(plan, mode)
    assert "DEVELOPMENT.md (navigation map)" in _text(plan, mode) or "docs/development/change.md" in _text(plan, mode)


def test_an_explicit_no_survives_restore_on_its_contract_in_max(tmp_path):
    env, core, task = _capture(tmp_path, {"task_contract": {KEY: False}})  # the top-level copy is gone
    assert core.docs_need_development is False and not _book_blocks(_capture_plan(env, core, task, "max"), "max")


def _parent_contract(answer):
    return build_task_contract({"id": "parent", "type": "task", "objective": "Parent work",
                                "task_contract": {KEY: answer}})


@pytest.mark.parametrize("answer", [True, False])
def test_no_child_inherits_either_answer(answer):
    from ouroboros.tools.control import _build_child_subagent_contract

    parent = _parent_contract(answer)
    assert parent[KEY] is answer
    child = _build_child_subagent_contract({"tid": "child", "objective": "Help", "parent_contract": parent,
                                            "parent_task_id": "parent", "root_task_id": "parent",
                                            "workspace_root": "", "workspace_mode": "", "parent_project_id": ""})
    assert KEY not in child and development_required({"task_contract": child}) is False


def _presence_promote(tmp_path, monkeypatch, inherited, **event):
    import supervisor.workers as workers
    from ouroboros.presence_authority import presence_ceiling_payload
    from supervisor.events import _handle_promote_chat_to_task
    from tests.test_host_service_api import _seed_presence_behavior
    from tests.test_presence_own_work import THREAD, _ceiling, _presence

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    binding = _seed_presence_behavior(tmp_path)
    pending = []
    pool = SimpleNamespace(
        DRIVE_ROOT=tmp_path, WORKERS={0: SimpleNamespace()}, PENDING=pending, bridge=None,
        append_jsonl=lambda *_a, **_k: None, persist_queue_snapshot=lambda **_k: True,
        enqueue_task=lambda task: pending.append(dict(task)) or pending[-1], load_state=lambda: {"owner_chat_id": 1})
    ceiling = presence_ceiling_payload(_ceiling())
    outcome = _handle_promote_chat_to_task({
        "type": "promote_chat_to_task", "task_id": "promoted-1", "routing_token": "tok-1", "objective": "Compile",
        "chat_id": 4242, "client_message_id": "evt-1", "project_id": "", "workspace_root": "", "source": "",
        "presence": _presence(binding, THREAD), "task_contract": {"capability_ceiling": ceiling, KEY: inherited}, **event},
        pool)
    assert outcome["status"] == "scheduled", outcome
    return pending[0], ceiling


@pytest.mark.parametrize(("own", "inherited"), [("absent", True), (True, False), (False, True)])
def test_a_presence_started_root_keeps_the_authority_and_only_its_own_answer(tmp_path, monkeypatch, own, inherited):
    """The starting work's contract replaces the new root's wholesale: neither of its answers rides along, and the
    root's own true or false survives that replacement, on the queued task and on its durable admission row."""
    from ouroboros.task_results import load_task_result

    task, ceiling = _presence_promote(tmp_path, monkeypatch, inherited, **({} if own == "absent" else {KEY: own}))
    stored = load_task_result(tmp_path, task["id"])
    for contract in (task["task_contract"], task["metadata"]["task_contract"], stored["task_contract"]):
        assert contract["capability_ceiling"] == ceiling  # the rest of that authority rides on
        assert contract.get(KEY, "absent") == own
    assert development_required(task) is (own is True)


def test_a_follow_up_is_separate_work_and_inherits_no_answer(tmp_path, monkeypatch):
    import ouroboros.tools.followup as followup
    from ouroboros.tools.followup import _handle_schedule_followup
    from ouroboros.tools.tool_context import ToolContext
    from supervisor import queue as task_queue

    _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(followup, "presence_root_carrier", lambda *_a, **_kw: {"presence": {"binding_id": "b"}})
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="origin", current_chat_id=1,
                      task_metadata={"root_task_id": "origin"}, task_contract={KEY: True, "constraints": "keep"})
    assert _handle_schedule_followup(ctx, objective="Later work", run_at="2030-01-01T00:00:00Z", relation="independent",
                                     destination="place").startswith("FOLLOWUP_SCHEDULED")
    row = next(item for item in task_queue.list_scheduled_tasks(tmp_path)["tasks"] if item.get("kind") != "notify")
    assert KEY not in row["task"]["task_contract"] and row["task"]["task_contract"]["constraints"] == "keep"
    # No schedule writer holds this door, whatever a template carries: the public API takes a raw task body.
    raw = {**row, "task": {**row["task"], "metadata": {"task_contract": {KEY: True}}, "task_contract": {KEY: True}}}
    occurrence = task_queue._task_from_schedule(raw)
    assert KEY not in occurrence["task_contract"] and KEY not in occurrence["metadata"]["task_contract"]


# --- Same meaning across its own lifecycle ---------------------------------------------------------------------

def _queue_restart(queue_module):
    assert queue_module.persist_queue_snapshot(reason="test") is not False
    queue_module.PENDING.clear()
    assert queue_module.restore_pending_from_snapshot() >= 1


@pytest.mark.parametrize("flag", [True, False])
def test_a_restart_restores_the_answer_from_the_contract_alone(tmp_path, monkeypatch, flag):
    task, _row, workers = _promoted(tmp_path, monkeypatch, **{KEY: flag})
    from supervisor import queue as task_queue

    monkeypatch.setattr(task_queue, "INITIALIZED", True, raising=False)
    monkeypatch.setattr(task_queue, "_keeps_saved_queue", lambda *_a, **_kw: False, raising=False)
    _queue_restart(task_queue)
    restored = next(row for row in workers.PENDING if row["id"] == task["id"])
    assert KEY not in restored  # the row's copy never survived a restart; the carrier does
    assert restored["task_contract"][KEY] is flag and development_required(restored) is flag
    ctx = SimpleNamespace(repo_dir=tmp_path, system_repo_dir=tmp_path, drive_root=tmp_path)
    assert development_resident(restored, ctx) is flag


@pytest.mark.parametrize("stored,carried", [(True, True), (False, False), ("absent", "absent")])
def test_continue_keeps_the_same_works_own_answer(tmp_path, monkeypatch, stored, carried):
    from supervisor.continuation_admission import admit_continuation

    _queue_module, _state, workers = _install_queue(tmp_path, monkeypatch)
    contract = {"context": "predecessor", **({} if stored == "absent" else {KEY: stored})}
    _interrupted(tmp_path, task_contract=contract)
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"] is True, ack
    row = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    assert row["task_contract"].get(KEY, "absent") == carried
    assert KEY not in row  # the carrier is the contract; a second Continue reads it the same way
    assert development_required(row) is (carried is True)
    # The successor's own capture reads it as any start does.
    env, core, task = _capture(tmp_path / "capture", {"task_contract": row["task_contract"]})
    assert core.development_required is (carried is True)
    assert bool(_book_blocks(_capture_plan(env, core, task, "nano"), "nano")) is (carried is True)


def test_continue_leaves_an_older_top_level_answer_where_it_was(tmp_path, monkeypatch):
    from supervisor.continuation_admission import admit_continuation

    _queue_module, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, task_contract={"context": "predecessor"}, **{KEY: True})
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    row = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    assert KEY not in row["task_contract"] and development_required(row) is False


# --- The view -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["low", "nano"])
def test_compact_modes_add_the_whole_composed_book_as_a_block_beside_the_shared_prefix(tmp_path, mode):
    env, core, task = _capture(tmp_path / "with", {"task_contract": {KEY: True}})
    plan = _capture_plan(env, core, task, mode)
    # The block holds its sources' exact bytes (``compose_book``), whatever line ending the fixture's writer produced.
    entry = (env.repo_dir / "docs/DEVELOPMENT.md").read_bytes().decode("utf-8")
    chapter = (env.repo_dir / "docs/development/change.md").read_bytes().decode("utf-8")
    assert _book_blocks(plan, mode) == ["## DEVELOPMENT.md\n\n" + entry + "\n\n" + chapter]
    env2, ordinary, task2 = _capture(tmp_path / "without")
    without = _blocks(_capture_plan(env2, ordinary, task2, mode), mode)
    with_book = _blocks(plan, mode)
    # One shared cache prefix: the first block is byte-identical and keeps the map; the book is the next stable one.
    assert with_book[0] == without[0] and "docs/development/change.md" in with_book[0]["text"]
    assert with_book[1]["text"].startswith("## DEVELOPMENT.md\n") and "cache_control" in with_book[1]
    assert _text(plan, mode).count("Full engineering discipline.") == 1
    assert plan.core_sha256 != _capture_plan(env2, ordinary, task2, mode).core_sha256
    assert replace(core, development_required=False).required_book_tokens(mode) == 0


def test_a_book_the_ordinary_max_view_holds_is_neither_repeated_nor_charged(tmp_path):
    env, core, task = _capture(tmp_path / "with", {"task_contract": {KEY: True}})
    plan = _capture_plan(env, core, task, "max")
    assert core.docs_need_development and core.required_book_tokens("max") == 0
    assert len(_book_blocks(plan, "max")) == 1 and core.required_book_tokens("low") > 0
    env2, ordinary, task2 = _capture(tmp_path / "without")
    assert _blocks(plan, "max")[:2] == _blocks(_capture_plan(env2, ordinary, task2, "max"), "max")[:2]
    assert measure_main_fit(plan, plan.messages_for("max"), [], profile="owner_max", rendered_mode="max",
                            round_id="r").measurement.required_book_tokens == 0


def test_a_foreign_subjects_max_view_gets_the_book_without_paying_for_it(tmp_path, fixed_density):
    ordinary = _plan(tmp_path, _core(tmp_path), "max", 1_000_000)
    required = _plan(tmp_path, _core(tmp_path, required=True), "max", 1_000_000)
    assert _choices(required) == _choices(ordinary) and required.initial_mode == "max"
    assert not _book_blocks(ordinary, "max") and len(_book_blocks(required, "max")) == 1
    stable = "\n".join(block["text"] for block in _blocks(required, "max"))
    assert "DEVELOPMENT.md (navigation map)" not in stable  # its own block replaces the map it stood in for
    held, plain = (plan.projection("max").memory_facts["max_room"] for plan in (required, ordinary))
    book = math.ceil(required.core.required_book_tokens("max") * RATIO)
    # The settled facts state the real room: choices were made without the book, the numbers carry it.
    assert held["need"] == plain["need"] + book and held["headroom"] == plain["headroom"] - book
    assert held["cause"] == plain["cause"] == "held" and ["directly required handbook", book] in held["parts"]


def test_a_missing_chapter_is_a_disclosed_gap_never_a_partial_book(tmp_path):
    env, _core_first, task = _capture(tmp_path, {"task_contract": {KEY: True}})
    from ouroboros.memory import Memory

    (env.repo_dir / "docs/development/change.md").unlink()
    core = context._capture_context_core(env, Memory(drive_root=env.drive_root, repo_dir=env.repo_dir), task, None, None)
    plan = _capture_plan(env, core, task, "low")
    assert core.development_required and core.required_book_tokens("low") == 0 and not _book_blocks(plan, "low")
    assert "Reference book source unavailable: docs/DEVELOPMENT.md" in _text(plan, "low")


def test_a_declared_capture_reads_the_same_signal(tmp_path, monkeypatch):
    """The capture seam only: no public caller can author this for a child in this slice."""
    from ouroboros.memory import Memory
    from ouroboros.reference_books import compose_book

    env, _first, _task = _capture(tmp_path)
    monkeypatch.setattr("ouroboros.subagent_runtime.current_model_visible_subagent_catalog", lambda: None)
    task = {"id": "declared", "type": "task", "text": "Case", "delegation_role": "subagent", "chat_id": 1,
            "configured_subagent": {"id": "helper", "route": {"kind": "api_model"}},
            "task_contract": {"input_sources": "declared", KEY: True}}
    core = context._capture_context_core(env, Memory(drive_root=env.drive_root, repo_dir=env.repo_dir), task, None, None)
    assert core.development_required and core.compact_reference_docs and core.docs_need_development is False
    plan = _capture_plan(env, core, task, "max")
    book = next(b for b in core.reference_books if b.book_id == "development")
    assert _book_blocks(plan, "max") == ["## DEVELOPMENT.md\n\n" + compose_book(book)]


# --- It costs the window, and nothing automatic ---------------------------------------------------------------------

@pytest.mark.parametrize("preferred,window", [("low", 1_000_000), ("low", 330_000), ("max", 330_000)])
def test_every_automatic_choice_matches_the_same_capture_without_the_book(tmp_path, fixed_density, preferred, window):
    ordinary = _plan(tmp_path, _core(tmp_path), preferred, window)
    required = _plan(tmp_path, _core(tmp_path, required=True), preferred, window)
    assert _choices(required) == _choices(ordinary)
    mode = required.initial_mode
    assert len(_book_blocks(required, mode)) == 1 and not _book_blocks(ordinary, mode)
    assert required.projection(mode).estimated_tokens > ordinary.projection(mode).estimated_tokens
    if window == 1_000_000:  # nothing else differs: the measured addition is the helper's exact additive cost
        added = required.projection(mode).estimated_tokens - ordinary.projection(mode).estimated_tokens
        assert abs(added - required.core.required_book_tokens(mode)) <= 1, (mode, added)  # chars/4 rounding
    usable = window - RESERVE
    assert math.ceil(required.projection(mode).estimated_tokens * RATIO) <= usable


def test_the_owner_nano_target_is_missed_without_a_refusal_or_a_narrower_view(tmp_path, fixed_density):
    plan = _plan(tmp_path, _core(tmp_path, required=True, book_tokens=120_000), "nano", 1_000_000)
    assert plan.initial_mode == "nano" and len(_book_blocks(plan, "nano")) == 1
    assert math.ceil(plan.projection("nano").estimated_tokens * RATIO) > 85_000
    ordinary = _plan(tmp_path, _core(tmp_path), "nano", 1_000_000)
    assert _choices(plan) == _choices(ordinary)


def test_unknown_capacity_proves_no_incompatibility(tmp_path, fixed_density):
    plan = _plan(tmp_path, _core(tmp_path, required=True, book_tokens=500_000), "low", 1, known=False)
    assert plan.initial_mode == "low" and len(_book_blocks(plan, "low")) == 1


def _tight_window(tmp_path, preferred, *, headroom=0):
    """The smallest window the same capture without the book holds, plus ``headroom``."""
    ordinary = _plan(tmp_path, _core(tmp_path), preferred, 1_000_000)
    reserve = fit.context_budget.context_mode_limits(preferred, preferred, RESERVE)[1]
    return math.ceil(ordinary.projection(preferred).estimated_tokens * RATIO) + reserve + 40_000 + headroom


@pytest.mark.parametrize("preferred", ["low", "nano"])
def test_a_physical_nonfit_is_named_and_enters_no_rescue(tmp_path, fixed_density, monkeypatch, preferred):
    window = _tight_window(tmp_path, preferred)
    ordinary = _plan(tmp_path, _core(tmp_path), preferred, window)
    assert ordinary.initial_mode == preferred
    from ouroboros import context_input_selection

    monkeypatch.setattr(context_input_selection, "task_input_pointer",
                        lambda *_a, **_kw: pytest.fail("the book's own excess never replaces the assignment"))
    with pytest.raises(fit.MinimumViewUnfit, match="directly required handbook") as raised:
        _plan(tmp_path, _core(tmp_path, required=True, book_tokens=400_000), preferred, window)
    unfit = raised.value
    assert unfit.required_book_tokens > 0 and unfit.window_tokens == window and unfit.deficit_tokens > 0
    assert unfit.deficit_tokens < unfit.required_book_tokens  # without it, this request fits


def test_the_assignment_rescue_stays_for_what_a_view_without_the_book_needs(tmp_path, fixed_density):
    huge = "The exact whole task. " * 40_000
    ordinary = _plan(tmp_path, _core(tmp_path, assignment=huge), "nano", 200_000, task={"id": "t", "type": "task"})
    assert ordinary.nano_projection.user_content_json  # the same rescue, for the same reason
    required = _plan(tmp_path, _core(tmp_path, assignment=huge, required=True, book_tokens=20_000), "nano", 200_000,
                     task={"id": "t", "type": "task"})
    assert required.nano_projection.user_content_json and len(_book_blocks(required, "nano")) == 1


def test_a_soft_work_room_shortfall_starts_with_the_real_room_in_its_notice(tmp_path, fixed_density):
    book = 40_000
    window = _tight_window(tmp_path, "low", headroom=math.ceil(book * RATIO) - 40_000 + 5_000)
    plan = _plan(tmp_path, _core(tmp_path, required=True, book_tokens=book), "low", window)
    assert plan.initial_mode == "low" and len(_book_blocks(plan, "low")) == 1
    room = _blocks(plan, "low")[-1]["text"]
    assert "Start room is smaller than requested: this task's directly required handbook takes" in room
    assert "Start room is smaller than requested" not in _blocks(_plan(tmp_path, _core(tmp_path), "low", window), "low")[-1]["text"]


def test_a_low_start_line_does_not_call_the_required_book_a_map(tmp_path, fixed_density):
    from ouroboros.context_view_facts import low_start_line

    shape = dict(arch_words=80_000, page_chars=40_000)
    plan = _plan(tmp_path, _core(tmp_path, required=True, **shape), "max", 330_000)
    assert plan.initial_mode == "low" and len(_book_blocks(plan, "low")) == 1
    assert plan.low_projection.memory_facts["max_room"]["cause"] == "crowded"
    assert "This task's directly required handbook stays whole beside these chapter maps." in _blocks(plan, "low")[-1]["text"]
    assert "(this task's directly required handbook stays whole)" in low_start_line(plan, "low", "t")
    plain = _plan(tmp_path, _core(tmp_path, **shape), "max", 330_000)
    assert _choices(plan) == _choices(plain)  # its words follow the choices, never enter them
    assert "stays whole" not in low_start_line(plain, "low", "t") and "started in Low with chapter maps:" in low_start_line(plain, "low", "t")


def test_reprojection_keeps_the_book_and_asks_nothing_of_memory_for_it(tmp_path, fixed_density):
    ordinary = _plan(tmp_path, _core(tmp_path), "low", 1_000_000)
    required = _plan(tmp_path, _core(tmp_path, required=True, book_tokens=30_000), "low", 1_000_000)

    def on_route(plan, window):
        return plan.reproject_for_route(window_tokens=window, known_window=True, ratio=RATIO, output_reserve=RESERVE,
                                        tool_schemas=SCHEMAS, start_mode="low")

    # The same capture without the book still fits here; with it the request does not, and nothing narrows.
    enough = (math.ceil((ordinary.low_projection.estimated_tokens + fit.tool_schema_tokens(SCHEMAS)) * RATIO)
              + RESERVE + 1_000)
    same = on_route(required, enough)
    assert same.low_projection.system_content_json == required.low_projection.system_content_json
    # A window the ordinary view itself cannot hold narrows both alike; the book stays resident.
    tight = enough - 20_000
    plain, kept = on_route(ordinary, tight), on_route(required, tight)
    assert kept.low_projection.memory_facts["floor"]["runs"] == plain.low_projection.memory_facts["floor"]["runs"]
    assert plain.low_projection.memory_facts["floor"]["runs"]
    assert len(_book_blocks(kept, "low")) == 1
    assert kept.initial_mode == plain.initial_mode == "low"


def test_the_target_facts_stay_while_the_goal_excludes_the_book(tmp_path, fixed_density):
    ordinary = _plan(tmp_path, _core(tmp_path), "low", 1_000_000)
    required = _plan(tmp_path, _core(tmp_path, required=True, book_tokens=250_000), "low", 1_000_000)

    def measured(plan):
        return measure_main_fit(plan, plan.messages_for("low"), SCHEMAS, profile="owner_low", rendered_mode="low",
                                round_id="r").measurement

    with_book, without = measured(required), measured(ordinary)
    assert with_book.target_total_tokens == without.target_total_tokens == 250_000
    assert with_book.target_deficit_tokens > 0  # the recorded fact compares the whole request
    assert with_book.required_book_tokens > 0 and without.required_book_tokens == 0
    assert abs(with_book.reclaim_goal_tokens - without.reclaim_goal_tokens) <= 2  # the goal asks nothing for the book


def _batch(plan, text, *, exempt=True):
    rows = [{"tool_call_id": "c1", "fn_name": "run_command", "result": text, "tool_args": {}, "is_error": False}]

    def measure(messages, schemas):
        fact = _measure_main_context_view(plan, messages, schemas, "low", "medium", "r")
        return fact if exempt else {**fact, "required_book_tokens": 0}

    delivered, receipt = project_tool_result_batch(rows, plan.messages_for("low"), SCHEMAS, drive_root=plan.core.section_root,
                                                   task_id="t", fit_candidate=measure, policy="measured_frame")
    return delivered[0], receipt


@pytest.mark.parametrize("chars", [40_000, 400_000], ids=["whole", "partial"])
def test_ordinary_unasked_outputs_are_the_same_beside_the_book_on_a_roomy_route(tmp_path, fixed_density, chars):
    ordinary = _plan(tmp_path, _core(tmp_path, memory=False), "low", 1_000_000)
    required = _plan(tmp_path, _core(tmp_path, memory=False, required=True, book_tokens=200_000), "low", 1_000_000)
    text = "An ordinary command output line.\n" * (chars // 33)
    plain, plain_receipt = _batch(ordinary, text)
    held, held_receipt = _batch(required, text)
    assert held_receipt["unsolicited_allowance_tokens"] == plain_receipt["unsolicited_allowance_tokens"] > 0
    assert bool(held.get("result_partial")) is bool(plain.get("result_partial")) is (chars == 400_000)
    # The same form: a partial view may differ only by the estimator's rounding on the larger request.
    assert abs(len(held["result"].split("RESULT_VIEW_JSON=")[0]) - len(plain["result"].split("RESULT_VIEW_JSON=")[0])) <= 64
    # The distinguishing control: the same batch with the book inside the target loses its body to an address.
    charged, charged_receipt = _batch(required, text, exempt=False)
    assert charged_receipt["unsolicited_allowance_tokens"] == 0 and charged.get("result_partial")


def test_a_tight_route_delivers_the_ordinary_output_and_the_admission_charges_the_book(tmp_path, fixed_density):
    """Astra's pair: a window with 2,000 tokens beside the book's start. The output is the one the same view without
    the book gets; the next request is then over by what the book adds alone, which the admission refuses by name."""
    shape = dict(memory=False, book_tokens=100_000)
    roomy = _plan(tmp_path, _core(tmp_path, required=True, **shape), "low", 1_000_000)
    window = math.ceil((roomy.low_projection.estimated_tokens + fit.tool_schema_tokens(SCHEMAS)) * RATIO) + RESERVE + 2_000
    text = "An ordinary command output line.\n" * 1_000
    (plain, plain_receipt), (held, receipt) = (_batch(_plan(tmp_path, _core(tmp_path, required=required, **shape), "low",
                                                            window), text) for required in (False, True))
    assert held["result"] == plain["result"] == text and not held.get("result_partial")
    assert receipt["unsolicited_allowance_tokens"] == plain_receipt["unsolicited_allowance_tokens"] > 2_000
    assert not plain_receipt["fit"]["capacity_deficit_tokens"]
    assert 0 < receipt["fit"]["capacity_deficit_tokens"] < receipt["fit"]["required_book_tokens"]


# --- Growth, refusal and the end of that path ---------------------------------------------------------------------

def _pass_context(plan, book, *, known=True):
    measurement = SimpleNamespace(required_book_tokens=book, capacity_total_tokens=1_000 if known else None,
                                  capacity_deficit_tokens=None, estimated_input_tokens=0)
    return SimpleNamespace(context_fit_plan=SimpleNamespace(window_record=None), active_context_mode="low",
                           accumulated_usage={}), SimpleNamespace(measurement=measurement)


@pytest.mark.parametrize(("book", "target", "slack", "runs", "passes"), [
    pytest.param(50_000, 30_000, 10_000, False, 0, id='book_only-first_pass'),  # the book alone explains the excess
    pytest.param(50_000, None, 10_000, False, 1, id='unknown_beside_book-second_pass'),  # nothing measures the excess: it is not guessed onto anything
    pytest.param(50_000, 90_000, 10_000, True, 0, id='ordinary_too-first_pass'),  # what a view without the book would also exceed: ordinary recovery, sized without it
    pytest.param(0, None, 10_000, True, 1, id='no_book-second_pass'),  # no book: as ever, each step takes its whole category
])
def test_neither_refusal_pass_pays_for_the_book(monkeypatch, passes, book, target, slack, runs):
    from ouroboros import loop_model_call as lmc

    calls = []

    def step(_ctx, name, _fit, remaining, _slack):
        calls.append((name, remaining))
        return None

    monkeypatch.setattr(lmc, "_apply_refusal_step", step)
    ctx, measured = _pass_context(None, book)
    steps = lmc.REFUSAL_PASSES[passes]
    assert lmc._apply_refusal_pass(ctx, steps, measured, target, slack) is False
    rest = ["memory_rest"] if "unseen_bodies" in steps else []  # memory's rest before a just-chosen body (2A)
    assert [name for name, _ in calls] == (list(steps) + rest if runs else [])
    if runs and target:
        assert calls[0][1] == target - book


def _renewal_context(tmp_path, messages, **usage):
    return SimpleNamespace(messages=messages, drive_root=tmp_path, drive_logs=tmp_path / "logs", task_id="renewal",
                           round_idx=12, event_queue=None, tool_schemas=[], accumulated_usage={
                               "_context_refusal_round": 12, "_last_llm_error_kind": "context_overflow", **usage})


def test_renewal_and_the_physical_cut_never_run_for_the_books_own_excess(tmp_path, monkeypatch):
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from ouroboros.context_renewal import renew_after_refusal
    from tests.test_context_renewal import _noted

    messages = _noted(tmp_path)
    before = json.loads(json.dumps(messages))
    size = estimate_context_prompt_tokens(messages, [])
    tools = SimpleNamespace(_ctx=SimpleNamespace(messages=messages))
    book = 2_000
    # A view without the book fits this window, or nothing measures it: the transcript stays as it was.
    for usage in ({"_context_capacity_total_tokens": size - book + 100}, {"_context_capacity_total_tokens": None}):
        ctx = _renewal_context(tmp_path, messages, _context_required_book_tokens=book, **usage)
        assert renew_after_refusal(ctx, tools) is False
        assert messages == before and "_context_refusal_frame" in ctx.accumulated_usage
    # What a view without the book cannot hold renews as it would for any task, and the record stays exact.
    import ouroboros.loop as loop_module

    monkeypatch.setattr(loop_module, "_emit_checkpoint_event", lambda *_a, **_kw: None)
    ctx = _renewal_context(tmp_path, messages, _context_required_book_tokens=book,
                           _context_capacity_total_tokens=size - book - 100)
    assert renew_after_refusal(ctx, tools) is True and len(messages) < len(before)


def test_the_physical_cut_is_sized_without_the_book(tmp_path, monkeypatch):
    from ouroboros import context_renewal
    from ouroboros.context_fit import estimate_context_prompt_tokens
    from tests.test_context_renewal import _noted

    messages = _noted(tmp_path)
    size = estimate_context_prompt_tokens(messages, [])
    seen = {}

    def cut(seed, *, drive_root, task_id, measure, usable):
        seen.update(measured=measure(seed), raw=estimate_context_prompt_tokens(seed, []), usable=usable)
        return seed, None

    monkeypatch.setattr(context_renewal, "physical_window_cut", cut)
    import ouroboros.loop as loop_module
    monkeypatch.setattr(loop_module, "_emit_checkpoint_event", lambda *_a, **_kw: None)
    tools = SimpleNamespace(_ctx=SimpleNamespace(messages=messages))
    book = 10
    ctx = _renewal_context(tmp_path, messages, _context_required_book_tokens=book, _context_capacity_total_tokens=book + 20)
    context_renewal.renew_after_refusal(ctx, tools)
    assert seen and seen["measured"] == seen["raw"] - book and seen["usable"] == book + 20 and size > seen["raw"]


@pytest.mark.parametrize("required", [False, True], ids=["ordinary", "required"])
def test_a_real_round_refuses_only_the_books_own_excess_and_keeps_the_view(main_call, tmp_path, fixed_density, required):
    """The real measure, pre-send check and refusal passes on a grown transcript; only the provider is a stand-in."""
    from ouroboros.context_renewal import renew_after_refusal
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.owner_continue import continuation_offer
    from tests.test_provider_unknown_terminal_text import _ctx as _limit_ctx
    from tests.test_subscription_main_wait import MODEL, ROUTE

    ctx, gateway, *_ = main_call
    book = 60_000
    plain = _plan(tmp_path, _core(tmp_path), "low", 1_000_000)
    room = 20_000
    window = (math.ceil((plain.low_projection.estimated_tokens + fit.tool_schema_tokens(SCHEMAS) + book) * RATIO)
              + RESERVE + room)
    plan = _plan(tmp_path, _core(tmp_path, required=required, book_tokens=book), "low", window)
    plan = replace(plan, model=MODEL, provider="claudexor", model_role="main", model_route=ROUTE)
    grown = [*plan.messages_for("low"),
             {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                 "name": "read_file", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "c1", "content": "A chosen read. " * 12_000}]  # more than the room left
    ctx.messages[:] = grown
    ctx.tool_schemas = SCHEMAS
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    ctx.active_context_mode = ctx.tools._ctx.active_context_mode = "low"
    shown = json.loads(json.dumps(ctx.messages[0]))
    answer, _cost, mode = loop._call_round_model(ctx)
    assert mode == "low"
    if not required:
        # The same transcript without the book fits this window: the ordinary send, the control that this is the book's.
        assert answer and len(gateway.creates) == 1
        return
    assert answer is None and not gateway.creates, "a request the window cannot hold is never sent"
    assert ctx.messages[0] == shown, "memory, books and the book itself stay as they were"
    assert any(row.get("tool_call_id") == "c1" and "A chosen read." in str(row.get("content")) for row in ctx.messages)
    assert ctx.context_fit_plan.low_projection.memory_facts == plan.low_projection.memory_facts
    rows = [json.loads(line) for line in (ctx.drive_logs / "events.jsonl").read_text().splitlines()]
    assert any(row.get("checkpoint_kind") == "context_fit_refused" for row in rows)
    assert not any(row.get("checkpoint_kind") in {"context_memory_coarsened", "context_fit_low_retry"} for row in rows)
    usage = ctx.accumulated_usage
    assert required_book_addition(plan, "low") > plan.core.required_book_tokens("low")  # its start note rides with it
    assert usage["_context_required_book_tokens"] == math.ceil(required_book_addition(plan, "low") * RATIO)
    # The loop's own end of this path (Opus's counterexample: context_overflow, a promised Continue, none offered).
    limit = SimpleNamespace(messages=ctx.messages, drive_root=tmp_path, drive_logs=ctx.drive_logs, task_id="grown",
                            round_idx=ctx.round_idx, event_queue=None, tool_schemas=SCHEMAS, accumulated_usage=usage)
    assert renew_after_refusal(limit, ctx.tools) is False
    text, usage, trace = loop._handle_provider_unavailable(_limit_ctx(tmp_path, usage), error_kind="context_overflow")
    assert usage["reason_code"] == "context_capacity" and "Context mode" in text and "a Continue keeps it" in text
    outcome = derive_loop_outcome(text, usage, trace)  # an infra-failed execution settles its row as failed
    assert outcome["outcome_axes"]["execution"]["status"] == "infra_failed" and outcome["reason_code"] == "context_capacity"
    row = {"status": "failed", "reason_code": outcome["reason_code"], "outcome_axes": outcome["outcome_axes"]}
    assert continuation_offer({**row, "root_task_id": "grown", "task_contract": {KEY: True}}, "grown")["eligible"]


EXPOSED = "Exposed command output.\n" * 8_000


@pytest.mark.parametrize("excess", ["book_only", "ordinary_too"])
def test_an_exposed_body_never_pays_for_the_book_at_the_real_round(main_call, tmp_path, fixed_density, excess):
    """A shown tool body the first refusal pass may address stays when the request is over only by the required book;
    what the same request without the book would also exceed is still recovered as before, from that body."""
    from tests.test_context_source_view import _observation, _tool
    from tests.test_subscription_main_wait import MODEL, ROUTE

    ctx, gateway, *_ = main_call
    core = _core(tmp_path, required=True, book_tokens=30_000, memory=False)
    exposed, latest = _tool("exposed", invocation="inv-exposed", text=EXPOSED), _tool("latest", invocation="inv-latest",
                                                                                     text="Latest output.\n" * 200)

    def build(window):
        return replace(_plan(tmp_path, core, "low", window), model=MODEL, provider="claudexor", model_role="main",
                       model_route=ROUTE)

    roomy = build(1_000_000)
    head = roomy.messages_for("low")
    fit_before = measure_main_fit(roomy, [*head, *exposed, *latest], SCHEMAS, profile="owner_low", rendered_mode="low",
                                  round_id="r").measurement
    book = fit_before.required_book_tokens
    shortfall = book // 2 if excess == "book_only" else book + 8_000
    plan = build(fit_before.estimated_input_tokens + fit_before.response_reserve_tokens - shortfall)
    assert plan.low_projection.system_content_json == roomy.low_projection.system_content_json
    ctx.messages[:] = [*head, *exposed, *latest]
    ctx.tool_schemas = SCHEMAS
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    ctx.active_context_mode = ctx.tools._ctx.active_context_mode = "low"
    ctx.tools._ctx._last_context_observation = _observation([*head, *exposed])  # a usable send showed this body
    answer, _cost, mode = loop._call_round_model(ctx)
    rows = [json.loads(line) for line in (ctx.drive_logs / "events.jsonl").read_text().splitlines()]
    passes = [(row["rung"], row["status"]) for row in rows if row.get("checkpoint_kind") == "context_reclaim_emergency"]
    shown = [row for row in ctx.messages if row.get("tool_call_id") == "exposed"]
    assert any(row.get("checkpoint_kind") == "context_fit_refused" for row in rows) and mode == "low"
    if excess == "book_only":
        assert answer is None and not gateway.creates and passes == []
        assert shown and shown[0]["content"] == EXPOSED
    else:  # ordinary pressure recovery, as without the requirement
        assert answer and len(gateway.creates) == 1 and ("bodies", "applied") in passes
        assert not shown and len(_book_blocks(ctx.context_fit_plan, "low")) == 1


def test_the_overflow_terminal_names_the_book_and_what_fits_without_it():
    from ouroboros.context_budget import REQUIRED_BOOK_UNFIT_KEY
    from ouroboros.loop_transport import provider_terminal_fallback_text

    flags = dict(is_context_overflow=True, is_transport_wait=False, waited_sec=0, interactive=False, is_deadline_exhausted=False)
    usage = {"_context_required_book_tokens": 51_000, "_context_prompt_estimate": 160_000,
             "_context_capacity_total_tokens": 240_000, "_context_response_reserve_tokens": 65_536}
    text = provider_terminal_fallback_text(usage, is_context_overflow=True, is_transport_wait=False, waited_sec=0,
                                            interactive=False, is_deadline_exhausted=False)
    assert "directly requires the whole development handbook (~51,000 tokens)" in text
    assert "without it the input is ~109,000 against 174,464 usable" in text
    assert "larger window" in text and "Context mode" in text and "without this requirement" in text
    # The Continue it names is the terminal cause's: a host forecast gets none, the book's known unfit gets it.
    forecast = {**usage, "reason_code": "context_overflow"}
    assert "a Continue keeps it" in text and "a Continue keeps it" not in provider_terminal_fallback_text(forecast, **flags)
    assert "a Continue keeps it" in provider_terminal_fallback_text({**forecast, REQUIRED_BOOK_UNFIT_KEY: True}, **flags)
    assert "handbook" not in provider_terminal_fallback_text(
        {**usage, "_context_required_book_tokens": 0}, is_context_overflow=True, is_transport_wait=False, waited_sec=0,
        interactive=False, is_deadline_exhausted=False)


def test_the_facts_row_sizes_the_ordinary_target_without_repeating_the_book_note():
    from ouroboros.loop_messages import context_facts_line

    usage = {"_context_prompt_estimate": 280_000, "_context_reply_allowance_tokens": 65_536,
             "_context_response_reserve_tokens": 65_536, "_context_capacity_total_tokens": 1_000_000,
             "_context_target_total_tokens": 250_000, "_context_required_book_tokens": 100_000}
    ctx = SimpleNamespace(accumulated_usage=usage, active_context_mode="low", active_model="m", messages=[],
                          context_fit_plan=None, round_idx=1, tools=None, tool_schemas=[])
    line = context_facts_line(ctx, measured=True)
    assert "target 250,000" in line and "target room 4,464" in line
    assert "required handbook" not in line  # another explanation must not itself shrink later results


def test_a_fallback_too_small_for_the_book_changes_no_memory(tmp_path, fixed_density, monkeypatch):
    """A primary outage walks to a configured route the book does not fit: its view stays, the request is refused."""
    from ouroboros import config, fallback_cooldown
    from tests.test_primary_view_return import _registry

    primary, reserve = "openai/gpt-primary", "openai/gpt-reserve"
    plan = _plan(tmp_path, _core(tmp_path, required=True, book_tokens=60_000), "low", 1_000_000)
    ordinary = _plan(tmp_path, _core(tmp_path), "low", 1_000_000)
    small = math.ceil(ordinary.low_projection.estimated_tokens * RATIO) + RESERVE + 2_000  # holds it without the book
    plan = replace(plan, model=primary, model_role="main")
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_kw: (
        {"model": task["model"], "provider": "openrouter", "base_url": "", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, route_fp=task["model"],
                        window_tokens={primary: 1_000_000, reserve: small}[task["model"]])))
    monkeypatch.setattr(config, "fallback_candidate_targets", lambda **_kw: [SimpleNamespace(model_id=reserve)])
    fallback_cooldown.reset_for_tests()
    registry = _registry(tmp_path)
    registry._ctx.primary_route["model"] = registry._ctx.active_model = primary
    messages = plan.messages_for("low")
    shown, sent = messages[0], {}

    def refused_round(call):
        sent.update(view=call.messages[0], plan=call.context_fit_plan, mode=call.active_context_mode)
        call.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
        return None, 0.0, call.active_context_mode

    monkeypatch.setattr(loop, "_call_round_model", refused_round)
    msg, model, _local, after, mode = loop._run_cross_model_fallback_chain(
        llm=SimpleNamespace(), ctx=registry._ctx, tools=registry, messages=messages, active_model=primary,
        active_use_local=False, tool_schemas=SCHEMAS, active_effort="medium", max_retries=3, drive_logs=tmp_path,
        task_id="outage", round_idx=2, event_queue=None, accumulated_usage={"_last_llm_error_kind": "transport_unavailable"},
        task_type="", emit_progress=lambda *_a, **_kw: None, context_fit_plan=plan, active_context_mode="low")
    assert msg is None and model == primary and mode == "low"
    assert sent["view"] == shown  # the candidate carried the same memory and the book; nothing paid for it
    assert sent["mode"] == "low" and messages[0] == shown


# --- A known incompatibility at a pooled start is the task's outcome ------------------------------------------------

def _agent_world(tmp_path, monkeypatch, window):
    """One real agent over a real capture: only the route's window and the model round are stand-ins."""
    from ouroboros import agent as agent_module

    repo, drive = tmp_path / "repo", tmp_path
    files = {"prompts/SYSTEM.md": "I am Ouroboros.", "BIBLE.md": "The constitution.",
             "docs/ARCHITECTURE.md": "# Body\n\nThe map.\n\n## Chapters\n\n- [Runtime](architecture/runtime.md)\n",
             "docs/architecture/runtime.md": "# Runtime\n\nHow.\n\n## Execution\n\nWhy.\n",
             "docs/DEVELOPMENT.md": "# Engineering\n\nHow I change.\n\n## Chapters\n\n- [Change](development/change.md)\n",
             "docs/development/change.md": "# Change\n\nCoherent.\n\n## Practice\n\n" + "Discipline word. " * 40_000}
    for path, text in files.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")
    monkeypatch.setattr(context, "_context_fit_route", _route(window))
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 8_192)
    monkeypatch.setattr(agent_module.OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    ran = []

    def model_round(**kwargs):
        ran.append(kwargs["task_id"])
        return "done", {}, {"reasoning_notes": [], "tool_calls": []}

    monkeypatch.setattr(agent_module, "run_llm_loop", model_round)
    return repo, drive, ran


def _queued(task_id, **fields):
    return {"id": task_id, "root_task_id": task_id, "type": "task", "chat_id": 7, "text": "Work", "_attempt": 1,
            "workspace": "none", **fields}


def _worker_run(tmp_path, monkeypatch, repo, tasks):
    """The real worker loop in this process: a crash would return before its next task."""
    from supervisor import worker_process

    for name in ("_bind_worker_repo_root", "_prepare_worker_task_runtime", "_adopt_published_extensions",
                 "_configure_worker_logging"):
        monkeypatch.setattr(worker_process, name, lambda *args: None)
    for path in ("ouroboros.platform_layer.create_new_session", "ouroboros.process_custody.start_parent_lifeline",
                 "ouroboros.extension_loader.reload_all", "ouroboros.config.initialize_runtime_mode_baseline",
                 "ouroboros.utils.set_log_sink"):
        monkeypatch.setattr(path, lambda *args, **kwargs: None)
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: "")
    monkeypatch.setattr("ouroboros.utils.get_git_info", lambda *args: ("fixture", "fixture"))
    monkeypatch.setattr("ouroboros.agent_task_pipeline._run_post_task_processing_async", lambda *a, **kw: None)
    crashes = []
    real_crash = worker_process._log_worker_crash
    monkeypatch.setattr(worker_process, "_log_worker_crash",
                        lambda wid, drive, phase, exc, tb: (crashes.append((phase, repr(exc))), real_crash(wid, drive, phase, exc, tb)))
    incoming, outgoing = queue.Queue(), queue.Queue()
    for task in (*tasks, None):
        incoming.put(task)
    worker_process.worker_main(1, incoming, outgoing, str(repo), str(tmp_path))
    return list(outgoing.queue), crashes, incoming


def test_a_nonfit_start_ends_its_task_named_and_the_same_worker_runs_the_next(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result

    repo, drive, ran = _agent_world(tmp_path, monkeypatch, 150_000)
    required = _queued("needs-book", task_contract={KEY: True}, context_requires_self_body_docs=False)
    events, crashes, incoming = _worker_run(tmp_path, monkeypatch, repo, [required, _queued("fits")])
    assert crashes == [] and incoming.empty(), "the worker survived to take the next task"
    done = {row["task_id"]: row for row in events if row.get("type") == "task_done"}
    assert set(done) == {"needs-book", "fits"} and ran == ["fits"]
    assert done["needs-book"]["status"] == "failed" and done["needs-book"]["reason_code"] == "context_capacity"
    stored = load_task_result(drive, "needs-book")
    assert stored["status"] == "failed" and stored["reason_code"] == "context_capacity"
    assert stored["outcome_axes"]["execution"]["reason_code"] == "context_capacity"
    text = next(row["text"] for row in events if row.get("type") == "send_message" and row.get("task_id") == "needs-book")
    assert "directly requires the whole development handbook" in text and "without it the start fits" in text
    assert "the model's limit here is 150,000" in text
    assert "window=" not in text, "a person reads the limit in words, never the refusal's key=value record"
    # The measured parts name the handbook on its own and add up to the input: the core part excludes it.
    measured = text.split("Measured, in tokens: ", 1)[1]
    parts = dict(re.findall(r"(core, request and tools|handbook|further memory) ([\d,]+)", measured))
    total = int(re.match(r"input ([\d,]+)", measured).group(1).replace(",", ""))
    assert "handbook" in parts and sum(int(value.replace(",", "")) for value in parts.values()) == total, measured
    assert "larger window" in text and "Context mode" in text and "without this requirement" in text
    assert "Error during processing" not in text and "task_exception" not in json.dumps(stored.get("outcome_axes"))
    # Nothing is invented for this attempt: its own loop evidence is absent, never a zero or a no-effect claim.
    loop_usage = stored["loop_outcome"]["usage"]
    assert loop_usage["prompt_tokens"] is None and loop_usage["total_rounds"] is None
    assert load_task_result(drive, "fits")["status"] == "completed"


def _real_agent(repo, drive):
    from ouroboros import agent as agent_module

    return agent_module.OuroborosAgent(agent_module.Env(repo_dir=repo, drive_root=drive), event_queue=queue.Queue())


def test_a_required_book_its_capture_cannot_read_ends_with_the_existing_source_outcome(tmp_path, monkeypatch):
    """No map or partial book stands in for the explicitly required whole handbook, and no fit is pretended: the
    existing source-unavailable outcome, under its own cause, ends that task before any model round. Without the
    requirement the same unreadable chapter stays a disclosed gap and the next task runs in that worker, as before."""
    from ouroboros import agent as agent_module
    from ouroboros.task_results import load_task_result

    repo, drive, ran = _agent_world(tmp_path, monkeypatch, 1_000_000)
    (repo / "docs/development/change.md").unlink()
    seen = {}

    def model_round(**kwargs):
        ran.append(kwargs["task_id"])
        seen[kwargs["task_id"]] = json.dumps(kwargs["messages"][0], ensure_ascii=False)
        return "done", {}, {"reasoning_notes": [], "tool_calls": []}

    monkeypatch.setattr(agent_module, "run_llm_loop", model_round)
    events, crashes, incoming = _worker_run(tmp_path, monkeypatch, repo,
                                            [_queued("needs-book", task_contract={KEY: True}), _queued("ordinary")])
    assert crashes == [] and incoming.empty()
    done = {row["task_id"]: row for row in events if row.get("type") == "task_done"}
    assert set(done) == {"needs-book", "ordinary"} and ran == ["ordinary"]
    assert done["needs-book"]["status"] == "failed"
    assert done["needs-book"]["reason_code"] == "required_reference_unavailable"
    stored = load_task_result(drive, "needs-book")
    assert stored["reason_code"] == "required_reference_unavailable"  # the producer's own cause
    text = next(row["text"] for row in events if row.get("type") == "send_message" and row.get("task_id") == "needs-book")
    assert "AUTHORITY_SOURCE_UNAVAILABLE" in text and "docs/DEVELOPMENT.md" in text
    assert "context_capacity" not in text and "Error during processing" not in text
    assert done["ordinary"]["status"] == "completed"
    assert "Reference book source unavailable: docs/DEVELOPMENT.md" in seen["ordinary"]


def test_the_terminal_preserves_saved_work_started_custody_and_unknown_money(tmp_path, monkeypatch):
    """A later attempt can meet a smaller window: what it saved or started stays with its owners."""
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros import delegate_custody as custody
    from ouroboros import working_checkpoint
    from ouroboros.task_results import load_task_result, write_task_result

    repo, drive, ran = _agent_world(tmp_path, monkeypatch, 150_000)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **kw: pytest.fail("no paid synthesis"))
    task_id = "resumed"
    write_task_result(drive, task_id, "running", delegated_runs_unreconciled=["run-a1"], root_task_id=task_id, task_attempt=2)
    assert custody.emit(drive, custody.STARTED, {"task_id": task_id, "run_id": "run-a1"})
    saved = working_checkpoint.checkpoint_path(drive, task_id, 2, create=True)
    saved.write_text('{"task_id": "resumed", "task_attempt": 2, "working": {}}', encoding="utf-8")
    unknown = {"cost_accounting_status": "ok", "cost_final": False, "accounted_upper_bound_usd": None, "total_rounds": None,
               "prompt_tokens": None, "completion_tokens": None, "reserved_usd": 1.5, "unresolved_upper_bound_usd": 1.5,
               "unknown_unmetered": True, "cost_presentation": None}
    monkeypatch.setattr("supervisor.state.reconstruct_task_cost", lambda *_a, **_kw: dict(unknown))
    agent = _real_agent(repo, drive)
    events = agent._handle_task_scoped(_queued(task_id, _attempt=2, task_contract={KEY: True}))
    done = next(row for row in events if row["type"] == "task_done")
    assert ran == [] and done["reason_code"] == "context_capacity"
    assert done["delegated_runs_unreconciled"] == ["run-a1"]  # the started run stays open, disclosed beside the cause
    assert done["unknown_unmetered"] is True and done["unresolved_upper_bound_usd"] == 1.5  # the ledger's answer, unchanged
    assert saved.exists(), "the only saved cognition is no result's to retire"
    stored = load_task_result(drive, task_id)
    assert stored["reason_code"] == "context_capacity" and stored["delegated_runs_unreconciled"] == ["run-a1"]


@pytest.mark.parametrize("fields", [{"_is_direct_chat": True}, {"_presence_turn": True, "_is_direct_chat": True}])
def test_a_direct_turn_keeps_its_own_typed_owner(tmp_path, monkeypatch, fields):
    from ouroboros import agent as agent_module

    repo, drive, _ran = _agent_world(tmp_path, monkeypatch, 150_000)
    monkeypatch.setattr(agent_module, "emit_task_results", lambda *_a, **_kw: pytest.fail("no pooled terminal here"))
    monkeypatch.setattr(agent_module.OuroborosAgent, "_prepare_task_context", lambda *_a, **_kw: (_ for _ in ()).throw(
        fit.MinimumViewUnfit("cannot", fixed_tokens=1, memory_tokens=1, reserve_tokens=1, window_tokens=1, ratio=1.0)))
    with pytest.raises(fit.MinimumViewUnfit):
        _real_agent(repo, drive)._handle_task_scoped(_queued("direct", **fields))


def test_only_a_known_incompatibility_takes_this_terminal(tmp_path, monkeypatch):
    from ouroboros import agent as agent_module

    repo, drive, _ran = _agent_world(tmp_path, monkeypatch, 150_000)
    monkeypatch.setattr(agent_module, "emit_task_results", lambda *_a, **_kw: pytest.fail("a fault is not an outcome"))
    monkeypatch.setattr(agent_module.OuroborosAgent, "_prepare_task_context",
                        lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("a genuine fault")))
    with pytest.raises(ValueError, match="a genuine fault"):
        _real_agent(repo, drive)._handle_task_scoped(_queued("faulted"))


# --- The reply beside an owner Nano target ----------------------------------------------------------------------------

def _reply(plan, ceiling):
    """The planned reply and the max_tokens the real send finalizer writes for this candidate."""
    from ouroboros.context_fit import _physical_context_for_fit
    from ouroboros.llm_attempt import bound_reply_allowance
    from ouroboros.usage_accounting import bind_physical_attempt_context

    messages = plan.messages_for("nano")
    disposition = measure_main_fit(plan, messages, SCHEMAS, profile="owner_nano", rendered_mode="nano", round_id="r")
    payload = {"model": "fixture", "messages": messages, "tools": SCHEMAS, "max_tokens": ceiling}
    with bind_physical_attempt_context(_physical_context_for_fit(disposition)):
        sent = bound_reply_allowance({"provider": "openai", "model": "fixture"}, payload)["max_tokens"]
    return disposition.measurement, sent


@pytest.mark.parametrize("assignment", ["Do the work.", "The exact whole task. " * 24_000], ids=["roomy", "at_floor"])
def test_the_book_beside_the_nano_target_never_shrinks_the_reply(tmp_path, fixed_density, assignment):
    def pair(window, ceiling, known=True):
        return [_reply(_plan(tmp_path, _core(tmp_path, memory=False, required=required, book_tokens=30_000,
                                             assignment=assignment), "nano", window, known=known), ceiling)
                for required in (False, True)]

    # Unknown window: the owner's Nano target stands in for it, and the book stands beside that target.
    (plain, plain_sent), (held, held_sent) = pair(1, 100_000, known=False)
    assert held.required_book_tokens > 0 and held.estimated_input_tokens > plain.estimated_input_tokens
    assert plain.capacity_total_tokens is None and held.capacity_total_tokens is None
    # Equal up to the chars/4 estimate's integer rounding of two separately counted requests.
    assert abs(held.reply_allowance_tokens - plain.reply_allowance_tokens) <= 2 and abs(held_sent - plain_sent) <= 2
    if assignment == "Do the work.":
        assert 8_192 < plain_sent < 85_000  # Nano's ordinary policy: room under the target, nothing larger
    else:
        assert plain_sent == held_sent == 8_192  # the ordinary floor stays the floor
    # A known caller ceiling still binds. A known window gives the reply the same request without the book gets, and one
    # that cannot hold the book beside that reply refuses the start by name (Astra's 58,868 -> 21,345 Nano pair).
    assert [sent for _measurement, sent in pair(1, 4_000, known=False)] == [4_000, 4_000]
    with pytest.raises(fit.MinimumViewUnfit, match="directly required handbook") as raised:
        pair(held.estimated_input_tokens + 30_000, 100_000)
    assert 0 < raised.value.deficit_tokens < raised.value.required_book_tokens  # without it, this request fits
    (plain_window, plain_window_sent), (held_window, held_window_sent) = pair(1_000_000, 100_000)
    assert held_window.reply_allowance_tokens == plain_window.reply_allowance_tokens and held_window_sent == plain_window_sent


def test_a_starts_answer_review_requirement_rides_its_contract_through_promotion_and_continue(tmp_path, monkeypatch):
    """``start_task(review_answer=true)`` (owner, 8 Nov 2026) is the mind's recorded "this answer itself must be
    checked": the new root's own contract carries it, and its Continue keeps it; absent stays absent."""
    from supervisor.continuation_admission import admit_continuation

    task, row, _workers = _promoted(tmp_path / "start", monkeypatch, review_answer=True)
    assert task["task_contract"]["review_answer"] is True and row["task_contract"]["review_answer"] is True
    plain, _row, _w = _promoted(tmp_path / "plain", monkeypatch)
    assert "review_answer" not in plain["task_contract"]

    _queue_module, _state, workers = _install_queue(tmp_path / "continue", monkeypatch)
    _interrupted(tmp_path / "continue", task_contract={"context": "predecessor", "review_answer": True})
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"] is True, ack
    successor = next(row for row in workers.PENDING if row["id"] == ack["successor_task_id"])
    assert successor["task_contract"]["review_answer"] is True
