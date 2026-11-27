"""A task's own explicit requirement of the whole handbook, as its real consumers read it (owner 1.11.2026).

``start_task(context_requires_development=true)`` is the one producer: the whole composed book joins any mode as its
own stable block, beside the view the same capture would get without it, whose one resident Body Map lists the
handbook's chapters (owner 25.11.2026: no mode, subject or binding preloads it any more). A subject marker, an older
top-level answer, a parent's answer and absence never ask for it. The book is charged to the window and to nothing else: no form, mode,
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
from ouroboros.context_input_selection import development_required
from ouroboros.context_measurement import measure_main_fit, required_book_addition
from ouroboros.contracts.task_contract import attach_task_contract, build_task_contract
from ouroboros.loop_model_call import _measure_main_context_view
from ouroboros.tool_result_delivery import project_tool_result_batch
from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed
from tests import _memory_view_synthetic as syn
from tests._budget_pause_exact_helpers import _install_queue
from tests._reference_book_fixtures import booked, map_saving
from tests.test_book_context_capture import _capture, _text
from tests.test_book_context_capture import _plan as _capture_plan
from tests.test_doc_context import foreign_tree
from tests.test_llm_claudexor import result, setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_owner_continue import NONCE, _interrupted
from tests.test_root_effort_ingress import _promoted
from tests.test_subscription_main_wait import ROUTE_B, _failed, main_call as main_fixture

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


def _route(window, known=True, source="metadata", provider="fixture"):
    from ouroboros.capability_evidence import utc_now_iso  # a serving statement is dated, as its read is
    evidence = SimpleNamespace(route_fp="route", status="confirmed" if known else "unprobeable",
                               stale=False, window_tokens=window, source=source, ts=utc_now_iso())
    return lambda *_a, **_kw: ({"provider": provider, "model": "fixture"}, evidence)


MAP_PARTS = ("## Body map (parts and addresses)\n\n- Core: `docs/architecture/core.md`\n\n"
             "Each reads with `read_file(root=\"system_repo\", path=...)`.")


def _core(root, *, required=False, book_tokens=40_000, assignment="Do the work.", memory=True,
          arch_words=4_000, page_chars=16_000):
    """A real capture shape over synthetic memory: pages, rows and task lines the floor can walk. ``arch_words`` sizes
    the resident Body Map; its parts and addresses are short, and the capture keeps a reader for its full text
    (``section_root``), so a frame the window lowered from Max may stand it at its parts (``memory_floor.lowered_maps``)."""
    lane = [*(syn.spoken(i, "ouroboros", 8_000) for i in range(2, 8)),
            {**syn.spoken(1, "human", 200), "owner_side": True, "immune": True}]
    snapshot = syn.snapshot(room_facts=syn.room("1", lane1=lane, lane2=6, legacy=4, legacy_chars=8_000),
                            story=syn.pages(12, page_chars), conversations=[syn.conversation(i) for i in range(4)], mark_count=2)
    return fit.ContextCore(
        base_prompt="SYSTEM", bible_md="BIBLE", body_map="## Body map\n\n" + "body " * arch_words,
        body_map_parts=MAP_PARTS if arch_words else "",
        development_md="# Development\n\n## Review\n\n" + "handbook " * (book_tokens * 4 // 9),
        semi_stable_text="IDENTITY", dynamic_text="", user_content_json=json.dumps(assignment),
        development_required=required, section_root=str(root), section_task_id="t",
        memory_view_json=memory_view.snapshot_json(snapshot) if memory else "")


def _plan(root, core, preferred, window, *, known=True, task=None, source="metadata", provider="fixture"):
    return fit.build_context_fit_plan(SimpleNamespace(drive_root=root), core, task or {"id": "t", "type": "task"},
                                      preferred_mode=preferred, tool_schemas=SCHEMAS,
                                      route_resolver=_route(window, known, source, provider))


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


def test_an_older_top_level_answer_gains_no_requirement_and_no_handbook_in_any_mode(tmp_path):
    """An older row's top-level true once kept the handbook in Max (its subject answer); since no mode preloads the
    handbook (owner 25.11.2026) it means nothing at all: only the contract's own true sends it."""
    workspace = str(foreign_tree(tmp_path / "foreign"))
    legacy = attach_task_contract({"id": "old", "type": "task", KEY: True, "workspace_root": workspace,
                                   "workspace_mode": "external", "metadata": {}})
    assert KEY not in legacy["task_contract"] and development_required(legacy) is False
    env, core, task = _capture(tmp_path / "capture", legacy)
    assert core.development_required is False and core.development_md == ""
    for mode in ("low", "nano", "max"):
        assert not _book_blocks(_capture_plan(env, core, task, mode), mode)
    # Its restart loses the row's copy as before; it never promotes itself through the contract.
    from supervisor.queue_snapshot import _PENDING_FIELDS

    assert KEY not in _PENDING_FIELDS
    restored = {key: legacy.get(key) for key in _PENDING_FIELDS}
    assert KEY not in restored["task_contract"] and development_required(restored) is False


@pytest.mark.parametrize(("mode", "answer"), [
    pytest.param("max", {"context_requires_self_body_docs": True}, id='max-answer0'),
    pytest.param("max", {"task_contract": {"context_requires_self_body_docs": True}}, id='max-answer1'),
    pytest.param("low", {"context_requires_self_body_docs": True}, id='low-answer0'),
    pytest.param("low", {"task_contract": {KEY: False}}, id='low-answer2'),
    pytest.param("low", {}, id='low-answer4'),
    pytest.param("nano", {"task_contract": {"context_requires_self_body_docs": True}}, id='nano-answer1'),
    pytest.param("nano", {KEY: False}, id='nano-answer3'),
])
def test_a_subject_marker_an_explicit_no_or_absence_keeps_the_body_map_alone(tmp_path, answer, mode):
    env, core, task = _capture(tmp_path, answer)
    plan = _capture_plan(env, core, task, mode)
    assert not core.development_required and core.development_md == "" and not _book_blocks(plan, mode)
    assert "Full engineering discipline." not in _text(plan, mode)
    assert core.body_map in _text(plan, mode) and "- Change: `docs/development/change.md`" in _text(plan, mode)


def test_an_explicit_no_survives_restore_on_its_contract_in_max(tmp_path):
    env, core, task = _capture(tmp_path, {"task_contract": {KEY: False}})  # the top-level copy is gone
    assert core.development_required is False and not _book_blocks(_capture_plan(env, core, task, "max"), "max")


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
    # The restored row's own capture reads it as any start does, in Max too.
    env, core, task = _capture(tmp_path / "capture", {"task_contract": restored["task_contract"]})
    assert core.development_required is flag and bool(_book_blocks(_capture_plan(env, core, task, "max"), "max")) is flag


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

@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_every_mode_adds_the_whole_composed_book_as_a_block_beside_the_shared_prefix(tmp_path, mode):
    env, core, task = _capture(tmp_path / "with", {"task_contract": {KEY: True}})
    plan = _capture_plan(env, core, task, mode)
    # The block holds its sources' exact bytes (``compose_book``), whatever line ending the fixture's writer produced.
    entry = (env.repo_dir / "docs/DEVELOPMENT.md").read_bytes().decode("utf-8")
    chapter = (env.repo_dir / "docs/development/change.md").read_bytes().decode("utf-8")
    assert _book_blocks(plan, mode) == ["## DEVELOPMENT.md\n\n" + entry + "\n\n" + chapter]
    env2, ordinary, task2 = _capture(tmp_path / "without")
    without = _blocks(_capture_plan(env2, ordinary, task2, mode), mode)
    with_book = _blocks(plan, mode)
    # One shared cache prefix: the first block is byte-identical and keeps the Map; the book is the next stable one.
    assert with_book[0] == without[0] and "- Change: `docs/development/change.md`" in with_book[0]["text"]
    assert with_book[1]["text"].startswith("## DEVELOPMENT.md\n") and "cache_control" in with_book[1]
    assert _text(plan, mode).count("Full engineering discipline.") == 1
    assert plan.core_sha256 != _capture_plan(env2, ordinary, task2, mode).core_sha256
    assert replace(core, development_required=False).required_book_tokens() == 0
    # No mode holds the handbook by itself any more (Max did for body work): every mode charges it, the same amount.
    assert core.required_book_tokens() > 0
    profile = {"max": "owner_max", "low": "owner_low", "nano": "owner_nano"}[mode]
    assert measure_main_fit(plan, plan.messages_for(mode), [], profile=profile, rendered_mode=mode,
                            round_id="r").measurement.required_book_tokens > 0


def test_a_max_view_gets_the_book_without_paying_for_it(tmp_path, fixed_density):
    ordinary = _plan(tmp_path, _core(tmp_path), "max", 1_000_000)
    required = _plan(tmp_path, _core(tmp_path, required=True), "max", 1_000_000)
    assert _choices(required) == _choices(ordinary) and required.initial_mode == "max"
    assert not _book_blocks(ordinary, "max") and len(_book_blocks(required, "max")) == 1
    assert _blocks(required, "max")[0] == _blocks(ordinary, "max")[0]  # one governance block, the Map beside the book
    held, plain = (plan.projection("max").memory_facts["max_room"] for plan in (required, ordinary))
    book = math.ceil(required.core.required_book_tokens() * RATIO)
    # The settled facts state the real room: choices were made without the book, the numbers carry it.
    assert held["need"] == plain["need"] + book and held["headroom"] == plain["headroom"] - book
    assert held["cause"] == plain["cause"] == "held" and ["directly required handbook", book] in held["parts"]


def test_a_missing_chapter_is_a_disclosed_gap_never_a_partial_book(tmp_path):
    env, _core_first, task = _capture(tmp_path, {"task_contract": {KEY: True}})
    from ouroboros.memory import Memory

    (env.repo_dir / "docs/development/change.md").unlink()
    core = context._capture_context_core(env, Memory(drive_root=env.drive_root, repo_dir=env.repo_dir), task, None, None)
    plan = _capture_plan(env, core, task, "low")
    assert core.development_required and core.required_book_tokens() == 0 and not _book_blocks(plan, "low")
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
    assert core.development_required and core.delegated_focus
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
        assert abs(added - required.core.required_book_tokens()) <= 1, (mode, added)  # chars/4 rounding
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
    assert "Start room at planning was smaller than the" in room and "this task's directly required handbook takes" in room
    assert "requested remain." not in room and "Start room is smaller than requested" not in room
    plain = _blocks(_plan(tmp_path, _core(tmp_path), "low", window), "low")[-1]["text"]
    assert "Start room at planning was smaller than" not in plain and "Start room is smaller than requested" not in plain


def test_a_low_start_line_does_not_call_the_required_book_a_map(tmp_path, fixed_density):
    from ouroboros.context_view_facts import low_start_line

    shape = dict(arch_words=80_000, page_chars=40_000)
    plan = _plan(tmp_path, _core(tmp_path, required=True, **shape), "max", 330_000)
    assert plan.initial_mode == "low" and len(_book_blocks(plan, "low")) == 1
    assert plan.low_projection.memory_facts["max_room"]["cause"] == "crowded"
    assert "This task's directly required handbook stays whole beside the body map." in _blocks(plan, "low")[-1]["text"]
    assert "(this task's directly required handbook stays whole)" in low_start_line(plan, "low", "t")
    plain = _plan(tmp_path, _core(tmp_path, **shape), "max", 330_000)
    assert _choices(plan) == _choices(plain)  # its words follow the choices, never enter them
    assert "stays whole" not in low_start_line(plain, "low", "t") and "runs in Low for now, not in your Max:" in low_start_line(plain, "low", "t")


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


def test_running_max_rebind_credits_only_what_can_yield(tmp_path, fixed_density, monkeypatch):
    from ouroboros.context_budget import RECLAIM_LOW_WATER_DIVISOR
    from tests._context_fit_fixtures import _rebind

    # Max carries the required handbook whole beside the Body Map. Lowering may credit the Map's step to its parts,
    # never the handbook, which Low keeps whole. (The base's body-subject case, where Max held the handbook by itself,
    # merged into this one: no subject preloads it any more.)
    core = _core(tmp_path, required=True, book_tokens=65_000, arch_words=96_000, page_chars=30_000)
    plan = _plan(tmp_path, core, "max", 1_000_000)
    messages = plan.messages_for("max") + [{"role": "user", "content": "Exact owner follow-up."}]
    tail = messages[1:]
    start = math.ceil(fit.estimate_context_prompt_tokens(messages, []) * RATIO)
    window = start + 8_192 - 174_000
    kept, mode, _ = _rebind(monkeypatch, tmp_path / 'route', plan, messages, window=window, ratio=RATIO, start_mode='max')
    assert mode == 'low' and messages[1:] == tail
    assert _book_blocks(kept, mode) == _book_blocks(plan, 'max')
    final = measure_main_fit(kept, messages, [], profile='task_local_low', rendered_mode=mode, round_id='r').measurement
    physical_free = final.capacity_total_tokens - final.response_reserve_tokens - final.estimated_input_tokens
    print(f'measured Max-to-Low required handbook: {window=} {physical_free=} book={final.required_book_tokens}')
    assert final.required_book_tokens > 0
    assert physical_free < 0 and physical_free + final.required_book_tokens >= math.ceil(window / RECLAIM_LOW_WATER_DIVISOR)
    assert kept.projection(mode).memory_facts['floor']['steps'] == plan.max_projection.memory_facts['floor']['steps']


@pytest.mark.parametrize('ordinary_fits', [True, False], ids=['book-only-wall', 'ordinary-wall-too'])
def test_max_to_low_can_spend_the_body_map_but_not_the_handbook_addition(tmp_path, fixed_density, monkeypatch,
                                                                        ordinary_fits):
    from ouroboros.llm_attempt import main_context_fit_refusal
    from ouroboros.usage_accounting import AttemptRequest
    from tests._context_fit_fixtures import _rebind

    # The Map is large enough that its step to its parts pays more than the 14,000 tokens of headroom below.
    core = _core(tmp_path, required=True, book_tokens=55_000, arch_words=20_000)
    held = _plan(tmp_path, core, 'max', 1_000_000)
    plain = _plan(tmp_path, replace(core, development_required=False), 'max', 1_000_000)
    tail = {'role': 'user', 'content': 'Keep this precise owner follow-up.'}
    # The lowered frame's size: the plain Max request with the Map at its parts (``memory_floor.lowered_maps``).
    max_size = math.ceil(fit.estimate_context_prompt_tokens([*plain.messages_for('max'), tail], []) * RATIO)
    low_size = max_size - math.ceil(map_saving(core) * RATIO)
    window = low_size + 8_192 + (14_000 if ordinary_fits else -4_000)
    outputs = []
    for plan in (plain, held):
        messages = [*plan.messages_for('max'), tail]
        assert math.ceil(fit.estimate_context_prompt_tokens(messages, []) * RATIO) + 8_192 > window
        kept, mode, _ = _rebind(monkeypatch, tmp_path / ('held' if plan.core.development_required else 'plain'),
                                plan, messages, window=window, ratio=RATIO, start_mode='max')
        assert mode == 'low' and messages[-1] == tail
        disposition = measure_main_fit(kept, messages, [], profile='task_local_low', rendered_mode=mode, round_id='r')
        measured = disposition.measurement
        refusal = main_context_fit_refusal(AttemptRequest(model='fixture', provider='fixture',
            prompt_tokens_bounded_estimate=measured.raw_input_tokens, physical_context=fit._physical_context_for_fit(disposition)))
        assert bool(_book_blocks(kept, mode)) is plan.core.development_required
        assert bool(refusal) is plan.core.development_required
        if refusal:
            assert 0 < -refusal['free_tokens'] <= measured.required_book_tokens
        outputs.append(kept.low_projection.memory_facts['floor']['runs'])
    assert outputs[0] == outputs[1]
    if not ordinary_fits:
        assert outputs[0]


@pytest.mark.parametrize("map_steps", [False, True], ids=["map-has-no-parts", "map-steps-to-its-parts"])
def test_the_books_rung_selects_low_only_where_the_body_map_saves(tmp_path, fixed_density, map_steps):
    """The measured books rung (``books_must_save``) has one thing left to give: the Body Map at its parts and
    addresses. Where that saves nothing (here a Map with no shorter form) it changes nothing and the required handbook
    never pays for a switch; where the Map steps, the same held view lowers to Low beside the whole handbook. (Replaces
    the base's "only the required Development map could save": there is no handbook map any more.)"""
    core = _core(tmp_path, required=True, memory=False, book_tokens=60_000, arch_words=20_000)
    if not map_steps:
        core = replace(core, body_map_parts="")
    held = _plan(tmp_path, core, 'max', 1_000_000)
    assert _book_blocks(held, 'max') == _book_blocks(held, 'low') and _book_blocks(held, 'max')

    def count(plan, mode):
        return fit.estimate_context_prompt_tokens(plan.messages_for(mode), [])

    chosen = held.reproject_for_route(window_tokens=80_000, known_window=True, ratio=RATIO, output_reserve=RESERVE,
        start_mode='max', select_mode='low', tool_schemas=[], current_messages=held.messages_for('max'),
        books_must_save=True, start_work=False)
    if not map_steps:
        assert count(held, 'low') >= count(held, 'max')  # nothing but the frame would change, and it saves nothing
        assert chosen.initial_mode == 'max' and chosen.max_projection.system_content_json == held.max_projection.system_content_json
        return
    assert chosen.initial_mode == 'low' and _rows(chosen) == ['body_map'] and len(_book_blocks(chosen, 'low')) == 1
    assert count(chosen, 'low') < count(held, 'max')
    text = "\n".join(block["text"] for block in _blocks(chosen, 'low'))
    assert MAP_PARTS in text and "body body body" not in text


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
    """Astra's pair: a window with 2,000 tokens beside the book's start, on a route whose one buffer holds the reply
    too (a local server states it), so a request under way has the room its start had. The output is the one the same
    view without the book gets; the next request is then over by what the book adds alone, which the admission
    refuses by name."""
    shape = dict(memory=False, book_tokens=100_000)
    roomy = _plan(tmp_path, _core(tmp_path, required=True, **shape), "low", 1_000_000)
    window = math.ceil((roomy.low_projection.estimated_tokens + fit.tool_schema_tokens(SCHEMAS)) * RATIO) + RESERVE + 2_000
    text = "An ordinary command output line.\n" * 1_000
    (plain, plain_receipt), (held, receipt) = (_batch(_plan(tmp_path, _core(tmp_path, required=required, **shape), "low",
                                                            window, source="local_health"), text)
                                               for required in (False, True))
    assert held["result"] == plain["result"] == text and not held.get("result_partial")
    assert receipt["unsolicited_allowance_tokens"] == plain_receipt["unsolicited_allowance_tokens"] > 2_000
    assert not plain_receipt["fit"]["capacity_deficit_tokens"]
    assert 0 < receipt["fit"]["capacity_deficit_tokens"] < receipt["fit"]["required_book_tokens"]


def test_the_required_nano_book_and_full_reply_fit_an_input_only_limit(tmp_path, fixed_density):
    from ouroboros.capability_evidence import CapabilityEvidence
    from ouroboros.context_window import WindowLimit, WindowRecord
    from ouroboros.llm_attempt import main_context_fit_refusal
    from ouroboros.llm_helper_window import window_reply_candidate
    from ouroboros.usage_accounting import AttemptRequest

    width = 100_000
    core = _core(tmp_path, required=True, book_tokens=60_000, memory=False, arch_words=0, assignment='work ' * 12_000)
    route = {'provider': 'fixture', 'model': 'fixture'}
    evidence = CapabilityEvidence(width, 'confirmed', 'metadata', 'input-only', window_kind='input')
    plan = fit.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), core, {'id': 't', 'type': 'task'},
        preferred_mode='nano', tool_schemas=[], route_resolver=lambda *_a, **_kw: (route, evidence))
    rows = plan.messages_for('nano')
    disposition = measure_main_fit(plan, rows, [], profile='owner_nano', rendered_mode='nano', round_id='r')
    measured = disposition.measurement
    physical = fit._physical_context_for_fit(disposition)
    native, shortfall, _count, _window = window_reply_candidate({'max_tokens': RESERVE, 'messages': rows},
        'max_tokens', physical, physical.capacity_total_tokens, {}, helper=False)
    request = AttemptRequest(model='fixture', provider='fixture', candidate_measurement_kind='canonical_json_v1',
        prompt_tokens_bounded_estimate=measured.raw_input_tokens, max_completion_tokens=native['max_tokens'],
        physical_context=physical)
    assert 90_000 < measured.estimated_input_tokens < width < measured.estimated_input_tokens + RESERVE
    assert _book_blocks(plan, 'nano') and measured.required_book_tokens > 74_000
    assert measured.reply_allowance_tokens == native['max_tokens'] == RESERVE and not shortfall
    assert main_context_fit_refusal(request) is None and not disposition.predicted_capacity_miss
    total = WindowRecord('total', total=WindowLimit(width, 'total', 'refusal'), reserve=8192)
    refused = main_context_fit_refusal(replace(request, physical_context=replace(physical, window_record=total)))
    assert refused['free_tokens'] == width - RESERVE - measured.estimated_input_tokens < 0
    print(f'input={measured.estimated_input_tokens} book={measured.required_book_tokens} reply={RESERVE}; '
          f'input-only accepted, equally-sized proved total {refused["free_tokens"]} tokens short')


BOOK_BODY = "Its practice and why. " * 70  # sizes both the handbook's rules and the Map's mechanism lines
PARTS = "## Body map (parts and addresses)"  # the Map's parts and addresses, the form a lowered or refused frame shows
WHOLE = "Mechanism 149: its facts."  # a line only the whole Map holds (``booked``'s default 150 headings)


def _booked(core, root, *, headings=150, book_body="", development_title="Change"):
    """``booked`` in the base's geometry: a Body Map of short mechanism lines (the base's detailed chapter maps were
    about that size) beside a handbook that ``book_body`` sizes rule by rule, so the Map's step is smaller than the
    book. A test whose stimulus is the Map's own size (a Max frame the window lowers) uses ``booked`` directly."""
    core = booked(core, root, headings=headings, development_title=development_title)
    return replace(core, development_md=core.development_md + "".join(
        f"\n\n## Practice {i}\n\n{book_body}" for i in range(headings)))


def _saving(core):
    """What the Map saves at its parts beside its retained reader, calibrated."""
    return math.ceil(map_saving(core) * RATIO)


def _rows(plan):
    return [row["section_id"] for row in plan.low_projection.memory_facts.get("host_sections_addressed", ())]


def test_a_required_book_that_fits_only_beside_the_maps_parts_steps_the_map(tmp_path, fixed_density):
    """Owner 23.11.2026 (1), with one resident Map (25.11.2026). The twin without the requirement decides every form;
    where the start's first request with the whole book would not go out, the Body Map steps to its parts and addresses
    beside its retained full text. Memory, mode, schemas, the reply and the book stay; past the Map, the named refusal.
    A route that holds the Map whole changes nothing but the book (the equality test above). The base's second step
    (ARCHITECTURE's map after the handbook's own) merged into this one: there is one Map."""
    from ouroboros.tool_policy import select_tool_schemas

    ordinary, required = (_booked(_core(tmp_path, required=flag, page_chars=2_000), tmp_path, book_body=BOOK_BODY)
                          for flag in (False, True))
    roomy = _plan(tmp_path, required, "low", 1_000_000)
    tools = fit.tool_schema_tokens(list(select_tool_schemas(SCHEMAS, context_mode="low").schemas))
    whole = math.ceil((roomy.low_projection.estimated_tokens + tools) * RATIO) + RESERVE  # the start beside the Map
    saving = _saving(required)
    one, twin = (_plan(tmp_path, core, "low", whole - saving // 2) for core in (required, ordinary))
    assert not _rows(roomy) and _rows(one) == [*_rows(twin), "body_map"]
    assert [part for i, part in enumerate(_choices(one)) if i != 4] == [part for i, part in enumerate(_choices(twin)) if i != 4]
    text = "\n".join(block["text"] for block in _blocks(one, "low"))
    assert len(_book_blocks(one, "low")) == 1 and PARTS in text and WHOLE not in text
    assert "omitted for the physical window; retained full text: read_file(root='artifact_store'" in text
    assert one.low_projection.memory_facts["host_sections_addressed"][-1].get("start_room") is None
    with pytest.raises(fit.MinimumViewUnfit, match="directly required handbook"):
        _plan(tmp_path, required, "low", whole - saving - 2_000)


def test_a_local_route_leaves_the_required_book_to_the_senders_exact_count(tmp_path, fixed_density):
    """Where the sender counts the sealed request exactly (a local route), the planner's estimate neither shortens a
    map nor refuses for the required handbook: the detailed request goes to that count, which alone can refuse it
    (``llm_attempt.main_context_fit_refusal``; then one yield per refusal). An estimate-only route at the same window
    shortens the book's own map (its terminal refusal is covered in the map-order test above)."""
    from ouroboros.tool_policy import select_tool_schemas

    core = _booked(_core(tmp_path, required=True, page_chars=2_000), tmp_path, book_body=BOOK_BODY)
    roomy = _plan(tmp_path, core, "low", 1_000_000)
    tools = fit.tool_schema_tokens(list(select_tool_schemas(SCHEMAS, context_mode="low").schemas))
    whole = math.ceil((roomy.low_projection.estimated_tokens + tools) * RATIO) + RESERVE
    saving = _saving(core)

    for window in (whole - saving // 2, whole - saving - 2_000):
        deferred = _plan(tmp_path, core, "low", window, provider="local")
        text = "\n".join(block["text"] for block in _blocks(deferred, "low"))
        assert deferred.initial_mode == "low" and not _rows(deferred) and len(_book_blocks(deferred, "low")) == 1
        assert WHOLE in text and PARTS not in text  # the Map whole beside the book
    assert _rows(_plan(tmp_path, core, "low", whole - saving // 2)) == ["body_map"]


def test_counterfactual_start_marks_deferred_local_required_book_without_claiming_a_definite_start(tmp_path, fixed_density):
    """A local route defers required-handbook admission and map shortening to its exact count: the pairwise start
    forecast shows the projected mode and states that actual admission and final navigation are unknown until a
    matching exact local count (Astra's 165,496 window / 65,536 reply reserve frame), while definite cases keep
    ``starts ...``."""
    from ouroboros import memory_floor

    core = _booked(_core(tmp_path, required=True, page_chars=2_000), tmp_path, book_body=BOOK_BODY)
    snapshot = memory_view.snapshot_from_json(core.memory_view_json)
    window = 165_496  # Astra's frame: 65,536 reply reserve leaves 99,960, below even the Map's parts' input
    with pytest.raises(fit.MinimumViewUnfit, match="directly required handbook"):
        _plan(tmp_path, core, "low", window)  # non-local proves this frame cannot admit the required book

    # The owner's Max: a small Map beside my tellings and the work's room holds Max (only whole books once lowered it);
    # a Map the Max frame cannot hold lowers it, and that start is projected in Low, temporarily, for the window.
    big_map = booked(core, tmp_path / "big", body=BOOK_BODY)
    big_map = replace(big_map, development_md=core.development_md)
    for preferred, captured, mode, mode_label in (("low", core, "low", "Low"), ("max", core, "max", "Max"),
                                                  ("max", big_map, "low", memory_floor.TEMPORARY_LOW)):
        deferred = memory_floor.counterfactual_starts(_plan(tmp_path, captured, preferred, window, provider="local"),
                                                      {"same": snapshot})
        assert deferred["measured"] and deferred["arms"]["same"]["mode"] == mode
        assert deferred["arms"]["same"]["local_admission_deferred"] is True
        lines = "\n".join(memory_floor.counterfactual_lines(deferred))
        assert f"- same: projected {mode_label} (local required-handbook admission deferred: actual admission and final navigation" in lines
        assert "- same: starts " not in lines

    # Definite cases keep ``starts ...``: a non-local route and a local route without a required handbook. (The base's
    # local self-body Max arm, whose handbook Max held by itself, is gone: no Max view holds it, so that arm defers too.)
    for plan, expected_mode in (
        (_plan(tmp_path, core, "low", 1_000_000), "Low"),
        (_plan(tmp_path, core, "max", 1_000_000), "Max"),
        (_plan(tmp_path, _booked(_core(tmp_path, required=False, page_chars=2_000), tmp_path, book_body=BOOK_BODY),
               "low", window, provider="local"), "Low"),
    ):
        definite = memory_floor.counterfactual_starts(plan, {"same": snapshot})
        assert "local_admission_deferred" not in definite["arms"]["same"]
        assert f"- same: starts {expected_mode};" in "\n".join(memory_floor.counterfactual_lines(definite))


def test_step_book_map_keeps_the_at_planning_book_notice_without_a_stale_remaining_count(main_call, tmp_path, fixed_density):
    """After ``step_book_map`` shortens a map on an unread Low start, the outgoing request keeps the handbook's
    at-planning notice and cost without a stale remaining-room count or present-tense shortfall."""
    from ouroboros.context_budget import bound_host_row
    from ouroboros.context_measurement import BOOK_NOTE_KEY

    def rows(saving, book, size):
        return [bound_host_row("ouroboros_context_facts", "[CONTEXT_FACTS] " + _filler(saving // 4 + 2_000))]

    ctx, gateway, *_ = _map_start(main_call, tmp_path / "round", rows, lambda saving: saving // 4, memory=True)
    before_note = ctx.context_fit_plan.low_projection.memory_facts[BOOK_NOTE_KEY]
    answer, _cost, mode = loop._call_round_model(ctx)
    assert answer and mode == "low" and len(gateway.creates) == 1 and _rows(ctx.context_fit_plan) == ["body_map"]
    assert ctx.context_fit_plan.low_projection.memory_facts[BOOK_NOTE_KEY] == before_note > 0
    outgoing = "\n".join(fit.extract_plain_text_from_content(row.get("content", ""))
                         for row in gateway.uploads[0][0]["messages"])
    assert PARTS in outgoing
    assert "Start room at planning was smaller than the" in outgoing and "this task's directly required handbook takes" in outgoing
    assert "requested remain." not in outgoing and "Start room is smaller than requested" not in outgoing


# The sealed first request's excess beyond what the planner admitted, in the calibrated saving of the Map at its parts,
# the maps it shortens and whether it goes out. A new start's own rows may hold an applied actor capsule
# (``review_history_view``), a role the step never reads; a prefix an answered send carried
# (``loop._record_transcript_prefix``) keeps its view. The base's "own_then_architecture" case merged into "own_map":
# there is one Map.
FIRST_REQUESTS = {
    "own_map": (lambda saving: saving // 2, ["body_map"], 1),
    "past_the_map": (lambda saving: saving + 2_000, ["body_map"], 0),
    "new_start_with_a_capsule": (lambda saving: saving // 2, ["body_map"], 1),
    "answered_prefix": (lambda saving: saving // 2, [], 0),
}


@pytest.mark.parametrize("case", list(FIRST_REQUESTS))
def test_a_first_request_the_book_cannot_send_steps_the_map_by_the_final_senders_measure(main_call, tmp_path,
                                                                                         fixed_density, case):
    """The planner admitted the start; its sealed first request carries more than the planner priced (here a facts row
    a round writes) and the final sender's own check refuses it by the book's excess alone. The refusal passes step the
    Body Map to its parts while the request still cannot go out; memory, mode, schemas, the reply and the whole book
    stay. Past the Map, nothing is sent and the refusal is named. A prefix an answered send already carried is shown
    as it was."""
    from ouroboros.context_budget import bound_host_row
    from ouroboros.context_view_facts import SIZE_REFUSAL_KEY
    from ouroboros.transcript_prefix import observe_send

    ctx, gateway, *_ = main_call  # a Map of 600 mechanisms: half its saving is above what a round adds itself
    fixture_plan = ctx.context_fit_plan
    core = _booked(_core(tmp_path, required=True, memory=False), tmp_path, headings=600, book_body="Its practice. " * 10)
    excess, expected, sends = FIRST_REQUESTS[case]
    over = excess(_saving(core))

    def build(window, **route):
        return replace(_plan(tmp_path, core, "low", window, **route), model=fixture_plan.model,
                       provider="claudexor", model_role="main", model_route=fixture_plan.model_route)

    roomy = build(1_000_000)
    facts = bound_host_row("ouroboros_context_facts", "[CONTEXT_FACTS] " + "a measured fact of this round "
                           * math.ceil((over / RATIO + 1_500) * 4 / 30))  # larger than the excess: the planner admits
    capsule = ([{"role": "assistant", "content": [{"type": "text", "text": "An applied capsule of my earlier review."}]}]
               if case == "new_start_with_a_capsule" else [])
    first = measure_main_fit(roomy, [*roomy.messages_for("low"), *capsule, facts], SCHEMAS, profile="owner_low",
                             rendered_mode="low", round_id="r").measurement
    plan = build(first.estimated_input_tokens + RESERVE - over, source="local_health")  # one buffer holds the reply
    assert plan.low_projection.system_content_json == roomy.low_projection.system_content_json  # admitted whole
    ctx.messages[:] = [*plan.messages_for("low"), *capsule, facts]
    ctx.tool_schemas = SCHEMAS
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    ctx.active_context_mode = ctx.tools._ctx.active_context_mode = "low"
    if case == "answered_prefix":  # what Main records of a usable send
        observe_send(ctx.tools._ctx, plan.messages_for("low"), round_idx=0, tools=SCHEMAS)
    shown = json.loads(json.dumps(ctx.messages[0]))
    answer, _cost, mode = loop._call_round_model(ctx)
    assert _rows(ctx.context_fit_plan) == expected and mode == "low"
    assert len(gateway.creates) == sends and bool(answer) is bool(sends)
    view = [block["text"] for block in ctx.messages[0]["content"]]
    assert view[1:] == [block["text"] for block in shown["content"][1:]]  # the book and everything after it, as admitted
    assert (PARTS in view[0]) is bool(expected) and ("Mechanism 599: its facts." in view[0]) is not bool(expected)
    if not sends:  # the final sender's own prediction refused it by name; no request went out
        rows = [json.loads(line) for line in (ctx.drive_logs / "events.jsonl").read_text().splitlines()]
        assert any(row.get("checkpoint_kind") == "context_fit_refused" for row in rows)
        slot = ctx.accumulated_usage[SIZE_REFUSAL_KEY]
        assert slot["basis"] == "known_window_prediction" and slot["book"] == ctx.accumulated_usage["_context_required_book_tokens"]


def _map_start(main_call, tmp_path, rows, over, *, memory=False):
    """The test above's start with the round's own rows the planner never priced, ``rows(saving, book, size)``
    (``size`` prices one row beside the start); ``over(saving)`` is the sealed first request's excess beyond what
    the planner admitted, in calibrated tokens. Returns the round, the gateway, the Map's saving and the builder of a
    plan on the same route."""
    ctx, gateway, *_ = main_call
    fixture_plan = ctx.context_fit_plan
    core = _booked(_core(tmp_path, required=True, memory=memory), tmp_path, headings=600, book_body="Its practice. " * 10)
    saving = _saving(core)

    def build(window, *, held_note_resume=None, **route):
        captured = replace(core, held_note_resume=held_note_resume)
        return replace(_plan(tmp_path, captured, "low", window, **route), model=fixture_plan.model,
                       provider="claudexor", model_role="main", model_route=fixture_plan.model_route)

    roomy = build(1_000_000)

    def measured(extra):
        return measure_main_fit(roomy, [*roomy.messages_for("low"), *extra], SCHEMAS, profile="owner_low",
                                rendered_mode="low", round_id="r").measurement

    alone = measured([])
    rows = rows(saving, alone.required_book_tokens,
                lambda row: measured([row]).estimated_input_tokens - alone.estimated_input_tokens)
    plan = build(measured(rows).estimated_input_tokens + RESERVE - over(saving), source="local_health")
    assert not _rows(plan)  # admitted with the Map whole; memory's start note names its own room
    assert memory or plan.low_projection.system_content_json == roomy.low_projection.system_content_json
    ctx.messages[:] = [*plan.messages_for("low"), *rows]
    ctx.tool_schemas = SCHEMAS
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    ctx.active_context_mode = ctx.tools._ctx.active_context_mode = "low"
    return ctx, gateway, saving, build


def _filler(tokens, phrase="a measured fact of this round "):
    return phrase * math.ceil(tokens / RATIO * 4 / len(phrase))


def _events(ctx):
    return [json.loads(line) for line in (ctx.drive_logs / "events.jsonl").read_text().splitlines()]


def test_a_facts_row_no_send_carried_is_rewritten_in_place_so_the_map_pays(main_call, tmp_path, fixed_density,
                                                                           monkeypatch):
    """The first request's facts row went out with no candidate (the final sender refused it before dispatch): the
    retry rewrites it instead of keeping it as a second, superseded row the Map's step would pay for. Here the Map at
    its parts pays the excess with one current facts row and could not beside two."""
    from ouroboros import loop_messages
    from ouroboros.context_budget import bound_host_row

    real = loop_messages.context_facts_line
    monkeypatch.setattr(loop_messages, "context_facts_line", lambda *a, **kw: real(*a, **kw).ljust(3_000, "~"))
    width = {}

    def rows(saving, book, size):
        width["facts"] = size(bound_host_row("ouroboros_context_facts", "[CONTEXT_FACTS] ".ljust(3_000, "~")))
        excess = saving - width["facts"] * 3 // 2  # one facts row of this width beside the Map's parts fits, two do not
        return [bound_host_row("ouroboros_context_facts", "[CONTEXT_FACTS] " + _filler(excess + 2_000))]

    ctx, gateway, saving, _build = _map_start(main_call, tmp_path, rows, lambda saving: saving - width["facts"] * 3 // 2)
    assert width["facts"] * 2 < saving  # the case is the facts row's, not the Map's
    answer, _cost, mode = loop._call_round_model(ctx)
    assert answer and len(gateway.creates) == 1 and mode == "low"
    assert _rows(ctx.context_fit_plan) == ["body_map"]
    padded = [row for row in ctx.messages if str(row.get("content")).endswith("~~~")]
    assert len(padded) == 1 and padded[0]["content"] == ctx.accumulated_usage["_context_facts_line"]


def test_an_ordinary_pass_before_the_book_leaves_the_map_its_turn(main_call, tmp_path, fixed_density):
    """A late host row the view without the book could not hold either takes the round's first pass (superseded state
    at its address); what is left is the book's own excess, which the Map at its parts pays. The second pass still
    steps the Map, priced on the prepared send, and the one retry goes out. (The base's "both maps" pass merged into
    the one Map's step.)"""
    from ouroboros.context_budget import bound_host_row

    sizes = {}

    def rows(saving, book, size):
        stale = bound_host_row("ouroboros_context_facts", "[CONTEXT_FACTS] " + _filler(book + 3_000, "an earlier fact "))
        sizes["stale"] = size(stale)
        return [stale, {"role": "user", "content": _filler(saving + 2_000, "A late word of my human. ")}]

    # Past the book with the stale row; past the Map whole, not past its parts, once it leaves.
    ctx, gateway, saving, _build = _map_start(main_call, tmp_path, rows, lambda saving: sizes["stale"] + saving // 2)
    answer, _cost, mode = loop._call_round_model(ctx)
    passes = [(row["rung"], row["status"]) for row in _events(ctx) if row.get("checkpoint_kind") == "context_reclaim_emergency"]
    assert ("host_copies", "applied") in passes
    assert _rows(ctx.context_fit_plan) == ["body_map"]
    assert answer and len(gateway.creates) == 1 and mode == "low"
    assert [row.get("book_map") for row in _events(ctx) if row.get("checkpoint_kind") == "context_fit_low_retry"] == [
        "body_map"]  # one owner-visible fact naming the Map's step
    assert any("A late word of my human." in str(row.get("content")) for row in ctx.messages)


def test_an_answer_that_read_the_start_keeps_its_whole_map(main_call, tmp_path, fixed_density):
    """A metered reply cut at its output limit read the start though no usable prefix was recorded (``departure`` has
    no baseline): its view keeps the Map whole, and the book's own excess is the named refusal."""
    from ouroboros.context_budget import bound_host_row
    from ouroboros.transcript_prefix import note_prefix_answer

    def rows(saving, book, size):
        return [bound_host_row("ouroboros_context_facts", "[CONTEXT_FACTS] " + _filler(saving // 2 + 2_000))]

    ctx, gateway, saving, _build = _map_start(main_call, tmp_path, rows, lambda saving: saving // 2)
    note_prefix_answer(ctx.accumulated_usage, {"prompt_tokens": 90_000, "completion_tokens": 4_096}, ctx.active_model, False)
    shown = json.loads(json.dumps(ctx.messages[0]))
    answer, _cost, _mode = loop._call_round_model(ctx)
    assert answer is None and not gateway.creates and _rows(ctx.context_fit_plan) == []
    assert ctx.messages[0] == shown
    assert any(row.get("checkpoint_kind") == "context_fit_refused" for row in _events(ctx))


def _facts_rows(ctx):
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY

    return [row for row in ctx.messages if row.get(HOST_CONTEXT_KIND_KEY) == "ouroboros_context_facts"]


def test_a_facts_row_a_candidate_carried_stays_through_an_unsent_preparation(main_call, monkeypatch):
    """A facts row whose candidate was admitted may have left the process: an unchanged re-preparation reuses it, its
    own candidate is refused before dispatch, and the next fresh facts row joins it; neither that row nor my human's
    words of the same text leave the view."""
    from ouroboros import llm_attempt
    from ouroboros.loop_messages import append_context_facts
    from ouroboros.loop_model_call import _presend_context_fit

    ctx, *_ = main_call
    verdicts = iter([None, {"cause": "known_window_prediction", "free_tokens": -1}])  # admitted, then refused
    monkeypatch.setattr(llm_attempt, "main_context_fit_refusal", lambda _request: next(verdicts))
    candidate = SimpleNamespace(candidate_raw_sha256="a" * 64, prompt_tokens_bounded_estimate=10)
    assert append_context_facts(ctx)
    carried = ctx.messages[-1]
    with _presend_context_fit(ctx, {}) as admit:
        assert admit(candidate)  # a candidate carrying it may go out
    assert append_context_facts(ctx) is False  # an unchanged re-preparation reuses the row
    with _presend_context_fit(ctx, {}) as admit:
        assert admit(candidate) is False  # its candidate refused before dispatch, nothing admitted
        raise PhysicalAttemptPreconditionFailed("refused before dispatch", attempt_id="re-prepared")
    words = {"role": "user", "content": carried["content"]}  # my human's words, the same text
    ctx.messages.append(words)
    assert append_context_facts(ctx)
    assert carried in ctx.messages and words in ctx.messages and len(_facts_rows(ctx)) == 2


def test_a_reprepare_inside_the_call_keeps_the_row_its_admitted_attempt_carried(main_call):
    """One call: its first attempt is admitted (it may leave) and its account refuses it; the wait's reprepare for the
    repaired route (``_reprepare_waiting_main``) writes fresh facts inside the same call, before the call ends. The
    row the admitted attempt carried stays history in the view and in the repaired send beside the fresh one."""
    from ouroboros.loop_messages import CONTEXT_FACTS_HEADER

    ctx, gateway, *_ = main_call
    gateway.results = [_failed("invalid_continuation", ROUTE_B), result(route=ROUTE_B)]
    gateway.dispatch = ["not_started", "response_received"]
    answer, _cost, _mode = loop._call_round_model(ctx)
    assert answer and len(gateway.accepted_operations) == 2

    def facts(rows):
        return [row["content"] for row in rows if str(row.get("content")).startswith(CONTEXT_FACTS_HEADER + " ")]

    [carried] = facts(gateway.uploads[0][0]["messages"])  # the row the admitted first attempt carried
    assert carried in facts(gateway.uploads[1][0]["messages"]) and carried in facts(ctx.messages)
    assert len(facts(gateway.uploads[1][0]["messages"])) == 2  # and the repaired route's own fresh facts


@pytest.mark.parametrize("rejection", ["shrink_check", "rebound_hint"])
def test_a_facts_row_no_admitted_candidate_carried_is_rewritten_in_place(main_call, monkeypatch, rejection):
    """The two other candidates that never pass admission: the strict-shrink check rejects a retry (a real dispatch,
    ``candidate_predicate``), or a start hint read for another scope is rebound before its send
    (``first_contact_guard``). Their facts row never left: the next facts row takes its place."""
    from ouroboros import context_window
    from ouroboros.loop_messages import append_context_facts
    from ouroboros.loop_model_call import _presend_context_fit
    from ouroboros.loop_routes import first_contact_guard

    ctx, gateway, *_ = main_call
    if rejection == "shrink_check":
        disposition = loop._measure_round_main_fit(ctx, automatic_pass_used=False)
        with pytest.raises(PhysicalAttemptPreconditionFailed):
            loop._dispatch_round_model(ctx, disposition, attempt_cap=1, candidate_predicate=lambda _request: False)
    else:
        assert append_context_facts(ctx)
        monkeypatch.setattr(context_window, "first_contact_rebound", lambda _physical: True)
        with first_contact_guard(ctx) as hint, _presend_context_fit(ctx, {}) as admit:
            assert hint(admit)(SimpleNamespace(physical_context=None)) is False
            raise PhysicalAttemptPreconditionFailed("rebound before its send", attempt_id="hinted")
        assert hint.rebound
    assert not gateway.creates and len(_facts_rows(ctx)) == 1
    unsent = _facts_rows(ctx)[0]
    ctx.messages.append({"role": "user", "content": "A later word of my human."})
    assert append_context_facts(ctx)
    assert unsent not in ctx.messages and len(_facts_rows(ctx)) == 1


def _numberless(main_call, tmp_path, results):
    """A start on a route whose window is unknown, the provider's own size refusals stating no number."""
    ctx, gateway, *_ = main_call
    fixture_plan = ctx.context_fit_plan
    core = _booked(_core(tmp_path, required=True, memory=False), tmp_path, headings=600, book_body="Its practice. " * 10)
    plan = replace(_plan(tmp_path, core, "low", 1_000_000, known=False), model=fixture_plan.model,
                   provider="claudexor", model_role="main", model_route=fixture_plan.model_route)
    refused = result(outcome="failed", problem={"code": "provider_failed", "message": "Controlled model refusal",
                                                "retryable": True, "context": {"httpStatus": 400,
                                                "vendorCode": "context_length_exceeded", "parameter": "input"}})
    gateway.results = [refused if verdict == "too_large" else result() for verdict in results]
    gateway.dispatch = ["response_received"] * len(results)
    ctx.messages[:], ctx.tool_schemas = plan.messages_for("low"), SCHEMAS
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    ctx.active_context_mode = ctx.tools._ctx.active_context_mode = "low"
    return ctx, gateway


@pytest.mark.parametrize(("results", "maps", "answered"), [
    pytest.param(["sent"], [], True, id="unknown_window_no_refusal"),
    pytest.param(["too_large", "sent"], ["body_map"], True, id="the_maps_parts_suffice"),
    pytest.param(["too_large", "too_large"], ["body_map"], False, id="past_the_map_nothing_is_left"),
])
def test_a_size_refusal_no_number_measures_steps_the_map_once(main_call, tmp_path, fixed_density, results, maps, answered):
    """Through the fake transport: a route of unknown window sends the Map whole while nothing refuses. A provider's
    size refusal that states no number prices nothing, so its retry carries the Body Map at its parts alone. When that
    shorter request is refused for size too, nothing is left to yield for the book: no further call. One provider call
    per verdict. (The base's "then_architecture" step is gone with the second map.)"""
    ctx, gateway = _numberless(main_call, tmp_path, results)
    answer, _cost, mode = loop._call_round_model(ctx)
    assert bool(answer) is answered and mode == "low" and len(gateway.creates) == len(results)
    assert _rows(ctx.context_fit_plan) == maps
    assert [row.get("book_map") for row in _events(ctx) if row.get("checkpoint_kind") == "context_fit_low_retry"] == maps


@pytest.mark.parametrize("turn", ["first", "later"])
def test_max_with_required_book_lowers_to_low_with_the_map_at_its_parts_on_first_and_later_turns(
        main_call, tmp_path, fixed_density, turn):
    """Max with a directly required handbook has required_book_tokens() > 0: its books rung must still lower Max to
    Low (the Body Map at its parts, all that rung has to give, while the handbook stays whole) on both the first turn
    and a later turn, instead of refusing before books runs."""
    from ouroboros.transcript_prefix import note_prefix_answer, observe_send

    ctx, gateway, *_ = main_call
    fixture_plan = ctx.context_fit_plan
    core = booked(_core(tmp_path, required=True, memory=False), tmp_path, headings=600,
                  body="Its practice and why. " * 20)
    assert core.required_book_tokens() > 0

    def build(window, **route):
        return replace(_plan(tmp_path, core, "max", window, **route), model=fixture_plan.model,
                       provider="claudexor", model_role="main", model_route=fixture_plan.model_route)

    roomy = build(1_000_000)
    assert roomy.initial_mode == "max"
    base_max = measure_main_fit(roomy, roomy.messages_for("max"), SCHEMAS, profile="owner_max",
                                rendered_mode="max", round_id="r").measurement
    arch_save = _saving(core)  # what the Low frame gives: the Map at its parts
    over = min(base_max.required_book_tokens, arch_save) // 3
    assert over > 2_000
    ordinary_max = base_max.estimated_input_tokens - base_max.required_book_tokens
    extra_tokens = (ordinary_max + RESERVE) // 3 + over + 4_000
    extra = {"role": "user", "content": _filler(extra_tokens, "An owner follow-up note. ")}
    grown_max = measure_main_fit(roomy, [*roomy.messages_for("max"), extra], SCHEMAS, profile="owner_max",
                                 rendered_mode="max", round_id="r").measurement
    window = grown_max.estimated_input_tokens + RESERVE - over
    plan = build(window, source="local_health")
    assert plan.initial_mode == "max" and len(_book_blocks(plan, "max")) == 1
    ctx.messages[:] = [*plan.messages_for("max"), extra]
    ctx.tool_schemas = SCHEMAS
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    ctx.active_context_mode = ctx.tools._ctx.active_context_mode = "max"
    ctx.tools._ctx._accumulated_usage = ctx.accumulated_usage
    if turn == "later":
        observe_send(ctx.tools._ctx, plan.messages_for("max"), round_idx=0, tools=SCHEMAS)
        note_prefix_answer(ctx.accumulated_usage, {"prompt_tokens": 90_000, "completion_tokens": 2_000},
                           ctx.active_model, False)
    answer, _cost, mode = loop._call_round_model(ctx)
    assert answer and len(gateway.creates) == 1 and mode == "low"
    assert _rows(ctx.context_fit_plan) == ["body_map"] and len(_book_blocks(ctx.context_fit_plan, "low")) == 1
    common = ctx.messages[0]["content"][0]["text"]
    assert PARTS in common and "Mechanism 599: its facts." not in common
    assert "## Rule 599" in "\n".join(_book_blocks(ctx.context_fit_plan, "low"))  # the handbook whole
    assert [row.get("book_map") for row in _events(ctx) if row.get("checkpoint_kind") == "context_fit_low_retry"] == [None]
    assert not any(row.get("checkpoint_kind") == "context_reclaim_emergency" for row in _events(ctx))


@pytest.mark.parametrize("low_fits", [True, False], ids=["low-accepted", "low-refused-nothing-left"])
def test_max_numberless_refusal_with_known_window_lowers_to_low_with_the_map_at_its_parts(
        main_call, tmp_path, fixed_density, low_fits):
    """Max with a required handbook on a known window with a numberless provider refusal: Pass 2 lowers Max to Low,
    whose frame is the Body Map at its parts beside the whole handbook (the base tried a "full Low" with detailed maps
    first; with one resident Map the Low frame IS its parts). A further refusal finds nothing left to yield."""
    ctx, gateway, *_ = main_call
    fixture_plan = ctx.context_fit_plan
    core = booked(_core(tmp_path, required=True, memory=False), tmp_path, headings=600,
                  body="Its practice and why. " * 20)
    assert core.required_book_tokens() > 0
    plan = replace(_plan(tmp_path, core, "max", 1_000_000), model=fixture_plan.model,
                   provider="claudexor", model_role="main", model_route=fixture_plan.model_route)
    assert plan.initial_mode == "max" and plan.window_record.usable is not None
    refused = result(outcome="failed", problem={"code": "provider_failed", "message": "Controlled model refusal",
                                                "retryable": True, "context": {"httpStatus": 400,
                                                "vendorCode": "context_length_exceeded", "parameter": "input"}})
    gateway.results = [refused, result()] if low_fits else [refused, refused]
    gateway.dispatch = ["response_received"] * len(gateway.results)
    ctx.messages[:], ctx.tool_schemas = plan.messages_for("max"), SCHEMAS
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    ctx.active_context_mode = ctx.tools._ctx.active_context_mode = "max"
    ctx.tools._ctx._accumulated_usage = ctx.accumulated_usage
    answer, _cost, mode = loop._call_round_model(ctx)
    assert bool(answer) is low_fits and mode == "low" and len(gateway.creates) == len(gateway.results)
    max_attempt = json.dumps(gateway.uploads[0][0]["messages"], ensure_ascii=False)
    low_attempt = json.dumps(gateway.uploads[1][0]["messages"], ensure_ascii=False)
    assert "Mechanism 599: its facts." in max_attempt and PARTS not in max_attempt  # Max held the Map whole
    assert PARTS in low_attempt and "Mechanism 599: its facts." not in low_attempt and "## Rule 599" in low_attempt
    assert _rows(ctx.context_fit_plan) == ["body_map"]
    assert [row.get("book_map") for row in _events(ctx) if row.get("checkpoint_kind") == "context_fit_low_retry"] == [None]


@pytest.mark.parametrize("map_tokens", [84_302, 85_000], ids=["design-map-fits", "still-unfit-no-arch-dev-concession"])
def test_local_nano_required_book_refusal_yields_optional_design_map_without_arch_dev_concessions_and_keeps_one_facts_row(
        main_call, tmp_path, monkeypatch, map_tokens):
    """Exact local Nano refusal from bound_reply_allowance before admission (window 92,494, reserve 8,192, full 88,090,
    DESIGN map 84,302): takes only the one permissible optional DESIGN->map yield with no new ARCH/DEV concessions,
    and re-preparation replaces the unsent [CONTEXT_FACTS] row in place instead of appending a second block."""
    from ouroboros import local_model, response_limits
    from ouroboros.local_model_server import input_fingerprint
    from ouroboros.section_floor import HostSection
    from ouroboros.tool_policy import select_tool_schemas

    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 8_192)
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr("ouroboros.capability_evidence.resolve_main_token_density",
                        lambda *_a, **_kw: (1.0, "fresh_route_usage"))
    monkeypatch.setattr(response_limits, "response_limit_for_target",
                        lambda _target: response_limits.ResponseLimit(max_output_tokens=8_192))

    design = HostSection(
        "design_bible",
        "## Design bible\n\nDESIGN_BIBLE_FULL_EXCERPT_SENTINEL " + "design rule " * 400,
        "## Design bible\n\nDESIGN_BIBLE_MAP_SENTINEL",
    )
    core = replace(
        _booked(_core(tmp_path, required=True, memory=False), tmp_path, headings=150, book_body="Its practice. " * 20),
        dynamic_head_sections=(design,),
        dynamic_head_text=design.full_text,
    )
    evidence = _route(92_494, source="local_health")()[1]
    plan = replace(
        fit.build_context_fit_plan(
            SimpleNamespace(drive_root=tmp_path), core, {"id": "t", "type": "task"},
            preferred_mode="nano", tool_schemas=SCHEMAS,
            route_resolver=lambda *_a, **_kw: ({"provider": "local", "model": "local-model"}, evidence),
        ),
        model_role="main",
    )
    assert plan.initial_mode == "nano"
    assert not plan.nano_projection.memory_facts.get("host_sections_addressed")

    ctx, _gateway, *_ = main_call
    ctx.messages[:] = plan.messages_for("nano")
    ctx.tool_schemas = list(select_tool_schemas(SCHEMAS, context_mode="nano").schemas)
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    ctx.active_context_mode = ctx.tools._ctx.active_context_mode = "nano"
    ctx.active_model = ctx.tools._ctx.active_model = "local-model"
    ctx.active_use_local = ctx.tools._ctx.active_use_local = True
    ctx.tools._ctx._accumulated_usage = ctx.accumulated_usage
    before_common = ctx.messages[0]["content"][0]["text"]

    sent_payloads, measurements = [], []

    class FakeLocalManager:
        def serving_context_evidence(self, *, bind_endpoint=False):
            return {"confirmed": True, "context_window": 92_494, "port": 8765, "process_id": "local-proc-1"}

        def measure_prepared_input(self, candidate, *, expected_port=None):
            rendered = json.dumps(candidate.get("messages") or [], ensure_ascii=False)
            tokens = 88_090 if "DESIGN_BIBLE_FULL_EXCERPT_SENTINEL" in rendered else map_tokens
            measurements.append((tokens, rendered.count("[CONTEXT_FACTS] ")))
            return {
                "supported": True,
                "input_is_exact": True,
                "input_tokens": tokens,
                "context_window": 92_494,
                "process_id": "local-proc-1",
                "native_input_sha256": input_fingerprint(candidate),
            }

    def fake_create(**candidate):
        sent_payloads.append(json.loads(json.dumps(candidate)))
        return SimpleNamespace(model_dump=lambda: {
            "choices": [{"message": {"role": "assistant", "content": "Local Nano succeeded."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": map_tokens, "completion_tokens": 12},
        })

    monkeypatch.setattr(local_model, "get_manager", lambda: FakeLocalManager())
    monkeypatch.setattr(ctx.llm, "_get_local_client", lambda: SimpleNamespace(
        base_url="http://127.0.0.1:8765/v1", timeout=30.0,
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)),
    ))

    answer, _cost, mode = loop._call_round_model(ctx)
    nano_rows = [row["section_id"] for row in ctx.context_fit_plan.nano_projection.memory_facts.get("host_sections_addressed", ())]
    assert mode == "nano" and nano_rows == ["design_bible"]
    assert ctx.messages[0]["content"][0]["text"] == before_common  # zero new ARCH/DEV concessions
    assert len(_book_blocks(ctx.context_fit_plan, "nano")) == 1
    assert len(_facts_rows(ctx)) == 1
    assert measurements == [(88_090, 1), (map_tokens, 1)]
    if map_tokens == 84_302:
        assert answer and answer["content"] == "Local Nano succeeded." and len(sent_payloads) == 1
        sent_text = json.dumps(sent_payloads[0]["messages"], ensure_ascii=False)
        assert "DESIGN_BIBLE_MAP_SENTINEL" in sent_text and "DESIGN_BIBLE_FULL_EXCERPT_SENTINEL" not in sent_text
        assert sent_text.count("[CONTEXT_FACTS] ") == 1
    else:
        assert answer is None and len(sent_payloads) == 0


def test_fresh_facts_kept_on_pre_admission_local_refusal_but_cleared_after_admission_or_unknown_dispatch(
        main_call, monkeypatch):
    """Pre-admission local refusal with zero sends keeps FRESH_FACTS_KEY so retry replaces the unsent facts row;
    after genuine admission or unknown dispatch, FRESH_FACTS_KEY is cleared and prior messages are never mutated."""
    from ouroboros import llm_attempt, usage_accounting as ua
    from ouroboros.context_view_facts import FRESH_FACTS_KEY
    from ouroboros.loop_llm_call import REFUSED_CANDIDATE_KEY, TRANSPORT_DEATHS_KEY
    from ouroboros.loop_messages import append_context_facts
    from ouroboros.loop_model_call import _presend_context_fit

    ctx, *_ = main_call
    monkeypatch.setattr(llm_attempt, "main_context_fit_refusal", lambda _request: None)
    candidate = SimpleNamespace(candidate_raw_sha256="b" * 64, prompt_tokens_bounded_estimate=100)
    try:
        ua.adopt_physical_attempt_capture(None)
        # 1) Pre-admission local bound_reply_allowance refusal (zero admissions, zero captures): keeps FRESH_FACTS_KEY.
        assert append_context_facts(ctx)
        unsent_row = _facts_rows(ctx)[-1]
        with _presend_context_fit(ctx, {}) as _admit:
            ctx.accumulated_usage[REFUSED_CANDIDATE_KEY] = {
                "provider": "local", "input_tokens": 88_090, "context_window_tokens": 92_494,
            }
        assert FRESH_FACTS_KEY in ctx.accumulated_usage
        ctx.accumulated_usage.pop(REFUSED_CANDIDATE_KEY, None)
        ctx.messages.append({"role": "user", "content": "Note before retry."})
        assert append_context_facts(ctx)
        assert unsent_row not in ctx.messages and len(_facts_rows(ctx)) == 1

        # 2) Genuine admission: pops FRESH_FACTS_KEY immediately; prior row stays history even if refusal follows.
        admitted_row = _facts_rows(ctx)[-1]
        with _presend_context_fit(ctx, {}) as admit:
            assert admit(candidate) is True
            ctx.accumulated_usage[REFUSED_CANDIDATE_KEY] = {"provider": "local", "input_tokens": 88_090}
        assert FRESH_FACTS_KEY not in ctx.accumulated_usage
        ctx.accumulated_usage.pop(REFUSED_CANDIDATE_KEY, None)
        ctx.messages.append({"role": "user", "content": "Note after admission."})
        assert append_context_facts(ctx)
        assert admitted_row in ctx.messages and len(_facts_rows(ctx)) == 2

        # 3) Unknown dispatch (unresolved capture / TRANSPORT_DEATHS_KEY): clears FRESH_FACTS_KEY; prior row stays.
        unknown_row = _facts_rows(ctx)[-1]
        with _presend_context_fit(ctx, {}) as _admit:
            ua.adopt_physical_attempt_capture(ua.PhysicalAttemptCapture(
                "unknown-send-1", ctx.active_model, "openai", "unresolved", "canonical_json_v1"))
            ctx.accumulated_usage[TRANSPORT_DEATHS_KEY] = {"attempt": "unknown-send-1"}
            ctx.accumulated_usage[REFUSED_CANDIDATE_KEY] = {"provider": "local", "input_tokens": 88_090}
        assert FRESH_FACTS_KEY not in ctx.accumulated_usage
        ctx.accumulated_usage.pop(REFUSED_CANDIDATE_KEY, None)
        ctx.accumulated_usage.pop(TRANSPORT_DEATHS_KEY, None)
        ctx.messages.append({"role": "user", "content": "Note after unknown dispatch."})
        assert append_context_facts(ctx)
        assert unknown_row in ctx.messages and len(_facts_rows(ctx)) == 3
    finally:
        ua.adopt_physical_attempt_capture(None)


@pytest.mark.parametrize("read", [None, "recorded_send", "metered_answer"])
def test_every_text_refit_says_whether_its_start_is_unread(main_call, tmp_path, fixed_density, monkeypatch, read):
    """The seam with the planner (``ContextFitPlan.reproject_for_route(unsent_start=...)``): a Max-to-Low retry, a
    memory pass and a route rebind each say whether no request of this work has been read yet (``start_unread``). A
    recorded send has been, and so has a metered reply cut at its output limit that left no usable prefix."""
    from ouroboros import loop_model_call, loop_round_limits
    from ouroboros.transcript_prefix import note_prefix_answer, observe_send

    ctx, *_ = main_call
    seen, inner = [], fit.ContextFitPlan.reproject_for_route

    def recorded(self, **kwargs):
        seen.append(kwargs["unsent_start"])
        return inner(self, **kwargs)

    monkeypatch.setattr(fit.ContextFitPlan, "reproject_for_route", recorded)
    plan = replace(_plan(tmp_path, _core(tmp_path, required=True), "max", 1_000_000), model=ctx.context_fit_plan.model,
                   provider="claudexor", model_role="main", model_route=ctx.context_fit_plan.model_route)
    ctx.messages[:], ctx.tool_schemas = plan.messages_for("max"), SCHEMAS
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    ctx.active_context_mode = ctx.tools._ctx.active_context_mode = "max"
    ctx.tools._ctx._accumulated_usage = ctx.accumulated_usage  # as the loop binds it
    if read == "recorded_send":
        observe_send(ctx.tools._ctx, ctx.messages, round_idx=0, tools=SCHEMAS)
    elif read == "metered_answer":
        note_prefix_answer(ctx.accumulated_usage, {"prompt_tokens": 9_000, "completion_tokens": 4_096}, ctx.active_model, False)
    loop_model_call._reproject_actual_overflow_low(ctx)
    loop_round_limits._coarsen_memory_view(ctx, loop._measure_round_main_fit(ctx, automatic_pass_used=True), saving=1)
    loop._rebind_context_fit_plan(ctx.context_fit_plan, ctx.tools, ctx.messages, model=ctx.active_model,
                                  use_local=False, tool_schemas=SCHEMAS, start_mode=ctx.active_context_mode)
    assert len(seen) >= 3 and set(seen) == {read is None}


@pytest.mark.parametrize("case", [
    "accepted_view_observed", "prepared_without_observation", "legacy_observation_only",
    "transcript_grown", "unrecorded_detailed",
])
def test_a_cold_return_shows_again_the_map_parts_its_accepted_view_held(main_call, tmp_path, fixed_density,
                                                                       monkeypatch, case):
    """The accepted or prepared request held the Body Map at its parts. A cold return's own fit of the same book and
    route replays that parts list from ``held_note_capture.book_maps`` (or a legacy observation), and the restore
    (``owner_wait.rebind_restored_route``) keeps it before its rebind measures anything, so memory keeps the forms the
    answer read and the next request goes out. Where the transcript grew since, so that even the view without the book
    fits only beside the parts, the rebind still takes nothing from memory for the Map. A cold return without a
    proven-unsent pause is never an unread start; one whose saved state held only the whole Map has no compact
    receipt, so nothing steps and the book's own excess is named."""
    observed = case in {"accepted_view_observed", "legacy_observation_only", "transcript_grown"}
    has_saved_list = case != "unrecorded_detailed"
    from ouroboros import context, continuation_tools, held_notes, owner_wait
    from ouroboros.capability_evidence import utc_now_iso
    from ouroboros.context_budget import bound_host_row
    from ouroboros.loop_round_limits import start_unread
    from ouroboros.tools.compact_context import record_context_view
    from ouroboros.transcript_prefix import observe_send

    def rows(saving, book, size):
        return [bound_host_row("ouroboros_context_facts", "[CONTEXT_FACTS] " + _filler(saving // 4 + 2_000))]

    ctx, gateway, saving, build = _map_start(main_call, tmp_path, rows, lambda saving: saving // 4, memory=True)
    detailed_plan = ctx.context_fit_plan
    gateway.results, gateway.dispatch = [result(), result()], ["response_received"] * 2
    window = ctx.context_fit_plan.window_tokens
    answer, _cost, _mode = loop._call_round_model(ctx)
    accepted = ctx.context_fit_plan
    assert answer and _rows(accepted) == ["body_map"]
    if observed:
        observe_send(ctx.tools._ctx, ctx.messages, round_idx=ctx.round_idx, tools=SCHEMAS)  # what Main records of it
        record_context_view(ctx.tools._ctx, ctx.messages, SCHEMAS)
    if case == "unrecorded_detailed":
        ctx.tools._ctx.context_fit_plan = detailed_plan
    ctx.messages.append({"role": "assistant", "content": "Read the handbook; continuing."})
    if case == "transcript_grown":  # beside the parts the view without the book still fits, beside the whole Map not
        shown = measure_main_fit(accepted, ctx.messages, SCHEMAS, profile="owner_low", rendered_mode="low",
                                 round_id="r").measurement
        spare = window - RESERVE - (shown.estimated_input_tokens - shown.required_book_tokens)
        ctx.messages += [{"role": "assistant", "content": "", "tool_calls": [{"id": "grown", "type": "function", "function": {
                              "name": "read_file", "arguments": "{}"}}]},
                         {"role": "tool", "tool_call_id": "grown", "content": _filler(spare - saving // 2, "A long read. ")}]
    state = owner_wait.continuation_state(ctx.tools._ctx, ctx.messages, {}, {}, ctx.round_idx, SCHEMAS, set())
    if case == "legacy_observation_only":
        state.pop(held_notes.CAPTURE_KEY, None)
    fresh = build(window, source="local_health", held_note_resume=held_notes.restored_capture(state))
    assert _rows(fresh) == (["body_map"] if has_saved_list else []) and fresh.core.required_book_tokens()
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = fresh
    served = fresh.model_route  # the same account, window and route identity the work ran on
    evidence = SimpleNamespace(route_fp="route", status="confirmed", stale=False, window_tokens=window,
                               source="local_health", ts=utc_now_iso(), source_id=served["source"],
                               credential_profile_id=served["credentialProfileId"],
                               account_fingerprint=served["accountFingerprint"])
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_kw: ({"model": task["model"], "provider": "claudexor"}, evidence))
    monkeypatch.setattr(owner_wait, "_continuation_fit_route", lambda *_a, **_kw: ({
        "model": ctx.active_model, "use_local_model": False, "model_role": "main", "model_route": fresh.model_route,
        "credential_profile_id": None}, "medium"))
    from ouroboros.window_learning import RouteSchemas

    monkeypatch.setattr(continuation_tools, "reconcile_restored_tools",  # the saved list priced as one catalog, no notice
                        lambda _tools, state, *_a, **_kw: continuation_tools.RestoredTools(RouteSchemas(state["tool_schemas"], [])))
    plan, mode = owner_wait.rebind_restored_route(ctx.tools, {"tool_schemas": SCHEMAS, "context_start": list(fresh.start)},
                                                  ctx.messages)
    assert mode == "low" and ctx.tools._ctx.context_fit_plan is plan and not start_unread(ctx.tools._ctx, ctx.accumulated_usage)
    ctx.context_fit_plan, ctx.active_context_mode, ctx.round_idx = plan, mode, ctx.round_idx + 1
    view = ctx.messages[0]["content"][0]["text"]
    answer, _cost, mode = loop._call_round_model(ctx)
    if not has_saved_list:  # nothing to show again, and no unread start to step the Map on a guess
        assert _rows(plan) == [] and "Mechanism 599: its facts." in view and answer is None and len(gateway.creates) == 1
        assert _rows(ctx.context_fit_plan) == []
        return
    assert _rows(plan) == ["body_map"] and PARTS in view and "Mechanism 599: its facts." not in view
    floor = lambda shown: (shown.low_projection.memory_facts.get("floor") or {}).get("steps")  # noqa: E731
    assert floor(plan) == floor(accepted)  # no early refit paid for the whole Map with memory
    assert _rows(ctx.context_fit_plan) == ["body_map"] and mode == "low"
    if case == "transcript_grown":  # with the book it is over by the growth: the Map never steps for a read view
        assert answer is None and len(gateway.creates) == 1
        return
    assert answer and len(gateway.creates) == 2


def test_cold_return_without_held_heads_never_newly_steps_a_formerly_whole_map(main_call, tmp_path,
                                                                               fixed_density, monkeypatch):
    """A task without held heads whose Map was shown whole returns cold onto a tighter known window. Neither the
    cold planner (``build_context_fit_plan``), ``rebind_restored_route`` nor the round sender may newly step the Map
    to pay for the mandatory book, nor prematurely raise ``MinimumViewUnfit`` before rebind."""
    from ouroboros import context, continuation_tools, held_notes, owner_wait
    from ouroboros.capability_evidence import utc_now_iso
    from ouroboros.loop_round_limits import start_unread
    from ouroboros.tool_policy import select_tool_schemas
    from ouroboros.tools.compact_context import record_context_view
    from ouroboros.transcript_prefix import observe_send
    from ouroboros.window_learning import RouteSchemas

    ctx, gateway, *_ = main_call
    fixture_plan = ctx.context_fit_plan
    core = _booked(_core(tmp_path, required=True, page_chars=2_000), tmp_path, book_body=BOOK_BODY)
    assert not any(isinstance(part, held_notes.HeldNoteSection) for part in core.dynamic_sections)

    def build(window, *, held_note_resume=None, source="local_health"):
        captured = replace(core, held_note_resume=held_note_resume)
        return replace(_plan(tmp_path, captured, "low", window, source=source), model=fixture_plan.model,
                       provider="claudexor", model_role="main", model_route=fixture_plan.model_route)

    roomy = build(1_000_000)
    tools = fit.tool_schema_tokens(list(select_tool_schemas(SCHEMAS, context_mode="low").schemas))
    whole = math.ceil((roomy.low_projection.estimated_tokens + tools) * RATIO) + RESERVE
    saving = _saving(core)
    # Sanity: a fresh start at these tighter windows DOES step the Map or raise MinimumViewUnfit.
    assert _rows(build(whole - saving // 2)) == ["body_map"]
    with pytest.raises(fit.MinimumViewUnfit, match="directly required handbook"):
        build(whole - saving - 2_000)

    ctx.messages[:], ctx.tool_schemas = roomy.messages_for("low"), SCHEMAS
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = roomy
    ctx.active_context_mode = ctx.tools._ctx.active_context_mode = "low"
    observe_send(ctx.tools._ctx, ctx.messages, round_idx=0, tools=SCHEMAS)
    record_context_view(ctx.tools._ctx, ctx.messages, SCHEMAS)
    ctx.messages.append({"role": "assistant", "content": "Read the whole Map; continuing."})
    state = owner_wait.continuation_state(ctx.tools._ctx, ctx.messages, {}, {}, 1, SCHEMAS, set())
    capture = held_notes.restored_capture(state)
    assert capture["entries"] == [] and capture.get("book_maps") == []

    served = roomy.model_route
    monkeypatch.setattr(continuation_tools, "reconcile_restored_tools",
                        lambda _tools, saved_state, *_a, **_kw: continuation_tools.RestoredTools(
                            RouteSchemas(saved_state["tool_schemas"], [])))
    monkeypatch.setattr(owner_wait, "_continuation_fit_route", lambda *_a, **_kw: ({
        "model": ctx.active_model, "use_local_model": False, "model_role": "main", "model_route": served,
        "credential_profile_id": None}, "medium"))

    for tight in (whole - saving // 2, whole - saving - 2_000):
        cold = build(tight, held_note_resume=capture)
        assert _rows(cold) == []  # neither newly shortened nor prematurely refused before rebind
        evidence = SimpleNamespace(route_fp="route", status="confirmed", stale=False, window_tokens=tight,
                                   source="local_health", ts=utc_now_iso(), source_id=served["source"],
                                   credential_profile_id=served["credentialProfileId"],
                                   account_fingerprint=served["accountFingerprint"])
        monkeypatch.setattr(context, "_context_fit_route",
                            lambda task, _ev=evidence, **_kw: ({"model": task["model"], "provider": "claudexor"}, _ev))
        restored_messages = [json.loads(json.dumps(row)) for row in ctx.messages]
        ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = cold
        ctx.active_context_mode = ctx.tools._ctx.active_context_mode = "low"
        plan, mode = owner_wait.rebind_restored_route(
            ctx.tools, {**state, "tool_schemas": SCHEMAS, "context_start": list(roomy.start)}, restored_messages)
        assert mode == "low" and _rows(plan) == [] and not start_unread(ctx.tools._ctx, ctx.accumulated_usage)
        view = restored_messages[0]["content"][0]["text"]
        assert WHOLE in view and PARTS not in view
    ctx.messages[:] = restored_messages
    ctx.context_fit_plan, ctx.active_context_mode = plan, mode
    answer, _cost, _mode = loop._call_round_model(ctx)
    assert answer is None and not gateway.creates and _rows(ctx.context_fit_plan) == []
    assert any(row.get("checkpoint_kind") == "context_fit_refused" for row in _events(ctx))


def test_cold_return_required_book_fits_unknown_reply_total_between_start_and_under_way_limits(tmp_path, fixed_density):
    """A cold-returned task with a directly required handbook on a non-local TOTAL window with unknown reply semantics
    (``source="metadata"``, not in ``_REPLY_INSIDE``) must succeed in the gap between the fresh-start limit
    (``window - RESERVE``) and the under-way limit (``window - NANO_MIN_HEADROOM_TOKENS``) without newly shortening
    maps. A fresh start at the same window, a reply-inside TOTAL (``source="refusal"``), and a TOTAL smaller than the
    book plus reply floor must still raise ``MinimumViewUnfit``."""
    from ouroboros import held_notes, owner_wait
    from ouroboros.context_budget import NANO_MIN_HEADROOM_TOKENS
    from ouroboros.llm_attempt import main_context_fit_refusal
    from ouroboros.tools.compact_context import record_context_view
    from ouroboros.usage_accounting import AttemptRequest

    snapshot = syn.snapshot(
        room_facts=syn.room("1", lane1=[{**syn.spoken(1, "human", 200), "owner_side": True, "immune": True}]),
        story=syn.pages(2, 400),
    )
    core = replace(_booked(_core(tmp_path, required=True, memory=False), tmp_path, book_body=BOOK_BODY),
                   memory_view_json=memory_view.snapshot_json(snapshot))
    roomy = _plan(tmp_path, core, "low", 1_000_000, source="metadata")
    book = math.ceil(core.required_book_tokens() * RATIO)
    gap_window = book + RESERVE - 10_000
    assert gap_window - RESERVE < book < gap_window - NANO_MIN_HEADROOM_TOKENS

    with pytest.raises(fit.MinimumViewUnfit, match="directly required handbook"):
        _plan(tmp_path, core, "low", gap_window, source="metadata")

    messages = [*roomy.messages_for("low"), {"role": "assistant", "content": "Read the whole Map; continuing."}]
    ctx = SimpleNamespace(
        task_id="t", task_attempt=1, drive_root=tmp_path, budget_drive_root=tmp_path, repo_dir=tmp_path,
        context_fit_plan=roomy, active_context_mode="low", active_model="fixture", active_use_local=False,
        active_effort="medium", model_wait_context=None, _actor_loaded_tool_names=set(), task_metadata={},
    )
    record_context_view(ctx, messages[:-1], SCHEMAS)
    state = json.loads(json.dumps(owner_wait.continuation_state(ctx, messages, {}, {}, 1, SCHEMAS, set())))
    capture = held_notes.restored_capture(state)
    assert capture["entries"] == [] and "messages" not in capture and capture.get("book_maps") == []

    resumed_core = replace(core, held_note_resume=capture)
    cold = _plan(tmp_path, resumed_core, "low", gap_window, source="metadata")
    assert cold.initial_mode == "low" and _rows(cold) == [] and len(_book_blocks(cold, "low")) == 1
    assert cold.window_record.input_room(RESERVE, under_way=False) == gap_window - RESERVE < book
    assert cold.window_record.admitted == gap_window - NANO_MIN_HEADROOM_TOKENS > book
    disposition = measure_main_fit(cold, messages, SCHEMAS, profile="owner_low", rendered_mode="low", round_id="r")
    assert disposition.measurement.capacity_deficit_tokens == 0
    request = AttemptRequest(model="fixture", provider="fixture", candidate_measurement_kind="canonical_json_v1",
                             prompt_tokens_bounded_estimate=disposition.measurement.raw_input_tokens,
                             max_completion_tokens=RESERVE,
                             physical_context=fit._physical_context_for_fit(disposition))
    assert main_context_fit_refusal(request) is None

    with pytest.raises(fit.MinimumViewUnfit, match="directly required handbook"):
        _plan(tmp_path, resumed_core, "low", gap_window, source="refusal")
    with pytest.raises(fit.MinimumViewUnfit, match="directly required handbook"):
        _plan(tmp_path, resumed_core, "low", book + NANO_MIN_HEADROOM_TOKENS - 2_000, source="metadata")


def test_owner_max_temporary_low_keeps_the_maps_parts_on_a_wider_cold_route(tmp_path, fixed_density, monkeypatch):
    """An owner-Max task that ran in Low, temporarily, for the window, with the Body Map at its parts returns cold on a
    wider route where Max fits. Both ``cold.low_projection`` (the alternate Low frame when ``cold`` starts in Max),
    ``rebind_restored_route`` into that Low, and a Max reprojection keep the saved parts. (Stimulus moved: a Map the
    Max frame cannot hold beside my tellings and the work's room lowers the start; whole books used to.)"""
    from ouroboros import context, continuation_tools, held_notes, owner_wait
    from ouroboros.capability_evidence import utc_now_iso
    from ouroboros.window_learning import RouteSchemas

    core = booked(_core(tmp_path, required=True, page_chars=2_000), tmp_path, body=BOOK_BODY)  # a Map of ~58,000
    roomy = _plan(tmp_path, core, "max", 1_000_000)
    assert roomy.initial_mode == "max"  # at a roomy window the whole Map fits Max's start room
    from ouroboros.tool_policy import select_tool_schemas

    tools = fit.tool_schema_tokens(list(select_tool_schemas(SCHEMAS, context_mode="low").schemas))
    # The lowered frame: the same start with the Map at its parts. Just above it Max, which holds the Map whole, cannot.
    low_whole = math.ceil((roomy.max_projection.estimated_tokens - map_saving(core) + tools) * RATIO) + RESERVE
    narrow = low_whole + 2_000
    accepted = _plan(tmp_path, core, "max", narrow)
    assert accepted.preferred_mode == "max" and accepted.initial_mode == "low" and _rows(accepted) == ["body_map"]
    assert accepted.low_projection.memory_facts["floor"]["mode_switch"] == {"from": "max", "to": "low"}

    ctx = SimpleNamespace(
        task_id="t", task_attempt=1, drive_root=tmp_path, budget_drive_root=tmp_path, repo_dir=tmp_path,
        context_fit_plan=accepted, active_context_mode="low", active_model="fixture", active_use_local=False,
        active_effort="medium", model_wait_context=None, _actor_loaded_tool_names=set(), task_metadata={},
        # Even if _last_context_observation later recorded Max (or was absent), low_projection's map survives:
        _last_context_observation={"memory_mode": "max", "memory_facts": {"host_sections_addressed": []}},
    )
    messages = accepted.messages_for("low")
    state = json.loads(json.dumps(owner_wait.continuation_state(ctx, messages, {}, {}, 1, SCHEMAS, set())))
    capture = held_notes.restored_capture(state)
    assert [row["section_id"] for row in capture["book_maps"]] == ["body_map"]

    wide = 1_000_000
    cold = _plan(tmp_path, replace(core, held_note_resume=capture), "max", wide)
    assert cold.preferred_mode == "max" and cold.initial_mode == "max"
    assert not cold.max_projection.memory_facts.get("host_sections_addressed")
    assert _rows(cold) == ["body_map"]  # the alternate Low frame retained its saved parts
    low_text = "\n".join(block["text"] for block in _blocks(cold, "low"))
    assert PARTS in low_text and WHOLE not in low_text

    # Reprojecting in Max (where low_projection stays the alternate frame) and then back to Low keeps it too.
    in_max = cold.reproject_for_route(window_tokens=wide, known_window=True, ratio=RATIO, output_reserve=RESERVE,
                                      tool_schemas=SCHEMAS, start_mode="max", select_mode="max",
                                      current_messages=cold.messages_for("max"), admitted=True)
    assert in_max.initial_mode == "max" and _rows(in_max) == ["body_map"]
    # Back on the narrow route the task ran on: reprojection from the mode the task itself ran in keeps effective
    # Low and the saved parts (the Max reprojection is the step above).
    back_to_low = in_max.reproject_for_route(window_tokens=narrow, known_window=True, ratio=RATIO,
                                             output_reserve=RESERVE, tool_schemas=SCHEMAS, start_mode="low",
                                             current_messages=in_max.messages_for("max"), admitted=True)
    assert back_to_low.initial_mode == "low" and _rows(back_to_low) == ["body_map"]

    # And rebind_restored_route with owner="max" and active_context_mode="low" on the wider route keeps the parts.
    monkeypatch.setattr("ouroboros.loop.get_context_mode", lambda: "max")
    evidence = SimpleNamespace(route_fp="route", status="confirmed", stale=False, window_tokens=wide,
                               source="metadata", ts=utc_now_iso())
    monkeypatch.setattr(context, "_context_fit_route", lambda *_a, **_kw: ({"provider": "fixture", "model": "fixture"}, evidence))
    monkeypatch.setattr(owner_wait, "_continuation_fit_route", lambda *_a, **_kw: ({
        "model": "fixture", "use_local_model": False, "model_role": "main", "model_route": {},
        "credential_profile_id": None}, "medium"))
    monkeypatch.setattr(continuation_tools, "reconcile_restored_tools",
                        lambda _tools, saved_state, *_a, **_kw: continuation_tools.RestoredTools(
                            RouteSchemas(saved_state["tool_schemas"], [])))
    ctx.context_fit_plan = cold
    ctx.active_context_mode = "low"
    tools = SimpleNamespace(_ctx=ctx, schemas=lambda: SCHEMAS, get_schema_by_name=lambda _n: SCHEMAS[0])
    rebound_messages = [json.loads(json.dumps(row)) for row in messages]
    plan, mode = owner_wait.rebind_restored_route(tools, {**state, "context_start": list(accepted.start)}, rebound_messages)
    assert mode == "low" and _rows(plan) == ["body_map"]
    assert PARTS in rebound_messages[0]["content"][0]["text"]
    assert WHOLE not in rebound_messages[0]["content"][0]["text"]


def test_cold_return_without_held_heads_retains_the_updated_map_behind_a_fresh_reader(tmp_path, fixed_density,
                                                                                       monkeypatch):
    """After the books change across a cold return without held heads, the Body Map shown at its parts stays at its
    parts and points via a fresh ``artifact_store`` reader to the CURRENT whole Map rather than silently expanding to
    the whole Map or serving stale bytes."""
    from ouroboros import context, continuation_tools, held_notes, memory_floor, owner_wait
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.capability_evidence import utc_now_iso
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.window_learning import RouteSchemas

    core = _booked(_core(tmp_path, required=True, page_chars=2_000), tmp_path, headings=150, book_body=BOOK_BODY)
    accepted = memory_floor.step_book_map(_plan(tmp_path, core, "low", 1_000_000), "low")
    assert accepted is not None and _rows(accepted) == ["body_map"]
    original = next(row for row in accepted.low_projection.memory_facts["host_sections_addressed"]
                    if row["section_id"] == "body_map")
    old_bytes = read_actor_source_bytes(core.section_root, core.section_task_id, original["source_ref"]).decode()
    assert old_bytes == core.body_map and WHOLE in old_bytes and "- Change: `docs/development/change.md`" in old_bytes

    ctx = SimpleNamespace(
        task_id="t", task_attempt=1, drive_root=tmp_path, budget_drive_root=tmp_path, repo_dir=tmp_path,
        context_fit_plan=accepted, active_context_mode="low", active_model="fixture", active_use_local=False,
        active_effort="medium", model_wait_context=None, _actor_loaded_tool_names=set(), task_metadata={},
    )
    messages = accepted.messages_for("low")
    state = json.loads(json.dumps(owner_wait.continuation_state(ctx, messages, {}, {}, 1, SCHEMAS, set())))
    updated_core = _booked(replace(core, held_note_resume=held_notes.restored_capture(state)), tmp_path,
                           headings=165, book_body="Revised chapter guidance. " * 75, development_title="Rework")
    cold = _plan(tmp_path, updated_core, "low", 1_000_000)
    assert _rows(cold) == ["body_map"]

    monkeypatch.setattr("ouroboros.loop.get_context_mode", lambda: "low")
    evidence = SimpleNamespace(route_fp="route", status="confirmed", stale=False, window_tokens=1_000_000,
                               source="metadata", ts=utc_now_iso())
    monkeypatch.setattr(context, "_context_fit_route", lambda *_a, **_kw: ({"provider": "fixture", "model": "fixture"}, evidence))
    monkeypatch.setattr(owner_wait, "_continuation_fit_route", lambda *_a, **_kw: ({
        "model": "fixture", "use_local_model": False, "model_role": "main", "model_route": {},
        "credential_profile_id": None}, "medium"))
    monkeypatch.setattr(continuation_tools, "reconcile_restored_tools",
                        lambda _tools, saved_state, *_a, **_kw: continuation_tools.RestoredTools(
                            RouteSchemas(saved_state["tool_schemas"], [])))
    ctx.context_fit_plan = cold
    tools = SimpleNamespace(_ctx=ctx, schemas=lambda: SCHEMAS, get_schema_by_name=lambda _n: SCHEMAS[0])
    rebound = [json.loads(json.dumps(row)) for row in messages]
    plan, mode = owner_wait.rebind_restored_route(tools, {**state, "context_start": list(accepted.start)}, rebound)
    assert mode == "low" and _rows(plan) == ["body_map"]
    current = next(row for row in plan.low_projection.memory_facts["host_sections_addressed"]
                   if row["section_id"] == "body_map")
    assert current["reader"] != original["reader"] and current["source_ref"] != original["source_ref"]
    view = rebound[0]["content"][0]["text"]
    assert PARTS in view and current["reader"] in view
    # The parts of the CURRENT Map (its handbook chapter now titled Rework), never its whole text or the old reader.
    assert "`docs/development/<name>.md`:\nchange: Rework" in view and "\nchange: Change" not in view
    assert original["reader"] not in view and "Mechanism 164: its facts." not in view
    current_bytes = read_actor_source_bytes(core.section_root, core.section_task_id, current["source_ref"]).decode()
    assert current_bytes == updated_core.body_map
    assert "- Rework: `docs/development/change.md`" in current_bytes and "Mechanism 164: its facts." in current_bytes
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "t"
    read_out = registry.execute("read_file", {"root": "artifact_store", "path": current["source_ref"]["path"]})
    assert "- Rework: `docs/development/change.md`" in read_out and "Mechanism 164: its facts." in read_out
    assert read_actor_source_bytes(core.section_root, core.section_task_id, original["source_ref"]).decode() == old_bytes


def test_pre_first_reply_saves_preserve_prepared_maps_and_distinguish_never_sent_from_unknown_and_preview(
        main_call, tmp_path, fixed_density, monkeypatch):
    """Cold restores of saves and pauses made before the first reply:
    1) ``working_checkpoint.save_round(..., 'ready')`` / ``save_model_call`` before the first reply preserves a
       reducer-selected compact map without ``_last_context_observation``.
    2) Missing answer is never proof of no dispatch: ``working_checkpoint``, ``recovered_from_working`` pauses,
       ``physical_calls=None`` pauses, abandoned-attempt pauses and ``physical_calls>=1`` pauses restore as NOT unread
       (preserving unknown-attempt fences and never newly shortening detailed maps); ONLY an exact live pause proven
       ``never sent`` (``physical_calls == 0``) keeps ``start_unread`` True so its first real send may still yield a map.
    3) ``memory_floor.counterfactual_starts`` clears ``held_note_resume`` even without held heads so fresh selection
       preview stays distinct from real cold restore."""
    from ouroboros import context, continuation_tools, held_notes, memory_floor, owner_wait, working_checkpoint
    from ouroboros.capability_evidence import utc_now_iso
    from ouroboros.context_budget import bound_host_row
    from ouroboros.loop_round_limits import start_unread
    from ouroboros.window_learning import RouteSchemas

    def rows(saving, book, size):
        return [bound_host_row("ouroboros_context_facts", "[CONTEXT_FACTS] " + _filler(saving // 4 + 2_000))]

    ctx, gateway, saving, build = _map_start(main_call, tmp_path, rows, lambda saving: saving // 4, memory=True)
    detailed = ctx.context_fit_plan
    window = detailed.window_tokens
    served = detailed.model_route
    evidence = SimpleNamespace(route_fp="route", status="confirmed", stale=False, window_tokens=window,
                               source="local_health", ts=utc_now_iso(), source_id=served["source"],
                               credential_profile_id=served["credentialProfileId"],
                               account_fingerprint=served["accountFingerprint"])
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_kw: ({"model": task["model"], "provider": "claudexor"}, evidence))
    monkeypatch.setattr(owner_wait, "_continuation_fit_route", lambda *_a, **_kw: ({
        "model": ctx.active_model, "use_local_model": False, "model_role": "main", "model_route": served,
        "credential_profile_id": None}, "medium"))
    monkeypatch.setattr(continuation_tools, "reconcile_restored_tools",
                        lambda _tools, saved_state, *_a, **_kw: continuation_tools.RestoredTools(
                            RouteSchemas(saved_state["tool_schemas"], [])))

    # 1) Reducer selected compact map BEFORE first reply; save_round("ready") writes working checkpoint with no observation.
    prepared = memory_floor.step_book_map(detailed, "low")
    assert prepared is not None and _rows(prepared) == ["body_map"]
    ctx.tools._ctx.task_attempt = 1
    ctx.tools._ctx.budget_drive_root = ctx.drive_root
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = prepared
    prepared_messages = prepared.reproject_transcript(ctx.messages, "low")
    limit_ctx = SimpleNamespace(tools=ctx.tools, messages=prepared_messages, llm_trace={}, accumulated_usage={},
                                owner_msg_seen=set(), round_idx=0, tool_schemas=SCHEMAS, budget_tail="tool")
    assert working_checkpoint.save_round(limit_ctx, "ready") is True
    _raw, saved_ready = working_checkpoint.read_working(ctx.drive_root, ctx.task_id, 1)
    assert not (saved_ready.get("context_observations") or {}).get("_last_context_observation")
    cold_prepared = build(window, source="local_health", held_note_resume=held_notes.restored_capture(saved_ready))
    assert _rows(cold_prepared) == ["body_map"]
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = cold_prepared
    restored_msgs, restored_trace, restored_usage, restored_seen = [], {}, {}, set()
    owner_wait.restore_continuation_state(ctx.tools, saved_ready, restored_msgs, restored_trace, restored_usage, restored_seen)
    rebound_prepared, mode = owner_wait.rebind_restored_route(ctx.tools, saved_ready, restored_msgs)
    assert mode == "low" and _rows(rebound_prepared) == ["body_map"]
    assert not start_unread(ctx.tools._ctx, restored_usage)  # working checkpoint is unknown, never proven unsent

    # 3) Fresh selection preview (counterfactual_starts) on a cold-restored plan without held heads stays fresh.
    from ouroboros.tool_policy import select_tool_schemas

    roomy_cold = build(1_000_000, source="local_health", held_note_resume=held_notes.restored_capture(saved_ready))
    assert _rows(roomy_cold) == ["body_map"]
    snapshot = memory_view.snapshot_from_json(roomy_cold.core.memory_view_json)
    before_resume = dict(roomy_cold.core.held_note_resume)
    preview = memory_floor.counterfactual_starts(roomy_cold, {"now": snapshot})
    assert roomy_cold.core.held_note_resume == before_resume
    assert "body_map" not in [name for name, _saved in preview["arms"]["now"]["host"]]
    roomy_fresh = build(1_000_000, source="local_health")
    low_tools = fit.tool_schema_tokens(list(select_tool_schemas(SCHEMAS, context_mode="low").schemas))
    tight_start = math.ceil((roomy_fresh.low_projection.estimated_tokens + low_tools) * RATIO) + RESERVE - saving // 2
    assert _rows(build(tight_start, source="local_health")) == ["body_map"]
    for empty_resume in ({"entries": []}, {}):
        tight_cold = replace(build(tight_start, source="local_health", held_note_resume=empty_resume),
                             core=replace(roomy_fresh.core, held_note_resume=empty_resume))
        assert _rows(tight_cold) == []
        tight_preview = memory_floor.counterfactual_starts(tight_cold, {"now": snapshot})
        assert tight_cold.core.held_note_resume == empty_resume
        assert "body_map" in [name for name, _saved in tight_preview["arms"]["now"]["host"]]

    # 2) Detailed maps saved before first reply: unknown vs already-shown vs proven never-sent pause.
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = detailed
    limit_detailed = SimpleNamespace(
        tools=ctx.tools, messages=[json.loads(json.dumps(r)) for r in ctx.messages], llm_trace={},
        accumulated_usage={"_pending_transport_outcome": {"attempt_id": "unknown-1"}}, owner_msg_seen=set(),
        round_idx=0, tool_schemas=SCHEMAS, budget_tail="tool")
    with working_checkpoint.model_call_scope(limit_detailed):
        working_checkpoint.save_model_call()
    _raw, saved_unknown = working_checkpoint.read_working(ctx.drive_root, ctx.task_id, 1)

    base_state = owner_wait.continuation_state(ctx.tools._ctx, ctx.messages, {}, {}, 0, SCHEMAS, set())
    unknown_carriers = [
        saved_unknown,  # working checkpoint with pending transport outcome and no _pause_row
        {**base_state, "_pause_row": {"physical_calls": None, "resume_point": {"recovered_from_working": "ready"}}},
        {**base_state, "_pause_row": {"physical_calls": None, "resume_point": {}}},  # degraded ledger
        {**base_state, "_pause_row": {"physical_calls": 0, "resume_point": {"abandoned_model_attempts": ["a1"]}}},
        {**base_state, "_pause_row": {"physical_calls": 1, "resume_point": {}}},  # already dispatched
    ]
    for carrier in unknown_carriers:
        cold_det = build(window, source="local_health", held_note_resume=held_notes.restored_capture(carrier))
        assert _rows(cold_det) == []
        ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = cold_det
        msgs, tr, usg, sn = [], {}, {}, set()
        owner_wait.restore_continuation_state(ctx.tools, carrier, msgs, tr, usg, sn)
        plan_det, _ = owner_wait.rebind_restored_route(ctx.tools, carrier, msgs)
        assert _rows(plan_det) == [] and not start_unread(ctx.tools._ctx, usg)
        if carrier is saved_unknown:
            assert usg.get("_pending_transport_outcome") == {"attempt_id": "unknown-1"}

    # Unknown working recovery with detailed maps refuses rather than newly shortening a map on a guess.
    ctx.messages[:], ctx.accumulated_usage = msgs, {}
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan_det
    ctx.active_context_mode = ctx.tools._ctx.active_context_mode = "low"
    answer_unknown, _cost, _ = loop._call_round_model(ctx)
    assert answer_unknown is None and not gateway.creates and _rows(ctx.context_fit_plan) == []

    # Exact live pause positively proven never sent (physical_calls == 0): start_unread stays True, first send yields map.
    proven_unsent = {**base_state, "_pause_row": {"physical_calls": 0, "resume_point": {}}}
    cold_unsent = build(window, source="local_health", held_note_resume=held_notes.restored_capture(proven_unsent))
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = cold_unsent
    msgs_u, tr_u, usg_u, sn_u = [], {}, {}, set()
    owner_wait.restore_continuation_state(ctx.tools, proven_unsent, msgs_u, tr_u, usg_u, sn_u)
    plan_unsent, mode_u = owner_wait.rebind_restored_route(ctx.tools, proven_unsent, msgs_u)
    assert mode_u == "low" and start_unread(ctx.tools._ctx, usg_u)
    gateway.results, gateway.dispatch = [result()], ["response_received"]
    ctx.messages[:], ctx.accumulated_usage = msgs_u, usg_u
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan_unsent
    ctx.active_context_mode = ctx.tools._ctx.active_context_mode = "low"
    answer_unsent, _cost, _ = loop._call_round_model(ctx)
    assert answer_unsent and len(gateway.creates) == 1 and _rows(ctx.context_fit_plan) == ["body_map"]


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
    from ouroboros.context_view_facts import SIZE_REFUSAL_KEY
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
             # With the book the request passes its send ceiling (the stated window less the reply floor); the same
             # transcript without the book stays under it.
             {"role": "tool", "tool_call_id": "c1", "content": "A chosen read. " * 26_000}]
    ctx.messages[:] = grown
    ctx.tool_schemas = SCHEMAS
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    ctx.active_context_mode = ctx.tools._ctx.active_context_mode = "low"
    # A grown transcript is a later round: its start was read (no unread start's Map yield applies here).
    from ouroboros.transcript_prefix import observe_send

    observe_send(ctx.tools._ctx, plan.messages_for("low"), round_idx=0, tools=SCHEMAS)
    shown = json.loads(json.dumps(ctx.messages[0]))
    answer, _cost, mode = loop._call_round_model(ctx)
    assert mode == "low"
    if not required:
        # The same transcript without the book is admitted on this window: the ordinary send, the control that the
        # refusal below is the book's own.
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
    assert required_book_addition(plan, "low") > plan.core.required_book_tokens()  # its start note rides with it
    assert usage["_context_required_book_tokens"] == math.ceil(required_book_addition(plan, "low") * RATIO)
    # The loop's own end of this path (Opus's counterexample: context_overflow, a promised Continue, none offered).
    limit = SimpleNamespace(messages=ctx.messages, drive_root=tmp_path, drive_logs=ctx.drive_logs, task_id="grown",
                            round_idx=ctx.round_idx, event_queue=None, tool_schemas=SCHEMAS, accumulated_usage=usage)
    assert renew_after_refusal(limit, ctx.tools) is False
    text, usage, trace = loop._handle_provider_unavailable(_limit_ctx(tmp_path, usage), error_kind="context_overflow")
    assert usage["reason_code"] == "context_capacity" and "Context mode" in text and "a Continue keeps it" in text
    # What the input is without the book: the admission's own refusal of this round's unchanged measured request.
    slot = usage[SIZE_REFUSAL_KEY]
    usable = slot["capacity"] - slot["reserve"]
    assert slot["basis"] == "known_window_prediction" and slot["book"] == usage["_context_required_book_tokens"]
    assert slot["capacity"] == usage["_context_capacity_total_tokens"] and slot["tokens"] - slot["book"] <= usable < slot["tokens"]
    assert (f"handbook (~{slot['book']:,} tokens), which stayed; without it the input is "
            f"~{slot['tokens'] - slot['book']:,} against {usable:,} usable.") in text
    outcome = derive_loop_outcome(text, usage, trace)  # an infra-failed execution settles its row as failed
    assert outcome["outcome_axes"]["execution"]["status"] == "infra_failed" and outcome["reason_code"] == "context_capacity"
    row = {"status": "failed", "reason_code": outcome["reason_code"], "outcome_axes": outcome["outcome_axes"]}
    assert continuation_offer({**row, "root_task_id": "grown", "task_contract": {KEY: True}}, "grown")["eligible"]


EXPOSED = "Exposed command output.\n" * 24_000  # larger than the reply reserve: a start without it fits the window below


@pytest.mark.parametrize("excess", ["book_only", "ordinary_too"])
def test_an_exposed_body_never_pays_for_the_book_at_the_real_round(main_call, tmp_path, fixed_density, excess):
    """A shown tool body the first refusal pass may address stays when the request is over only by the required book
    (without it the send would go out); what the same request without the book would also exceed is still recovered
    as before, from that body."""
    from ouroboros.context_budget import NANO_MIN_HEADROOM_TOKENS
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
    # Over its send ceiling (the stated window less the reply floor) by ``shortfall``: by less than the book, or by
    # the book and 8,000 more.
    plan = build(fit_before.estimated_input_tokens + NANO_MIN_HEADROOM_TOKENS - shortfall)
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
    """The figure is the round's own size refusal's: the host's admission refused the candidate before sending, with
    the book, size, window and reserve it judged (``remember_size_refusal``). A request the provider refused keeps
    its own size of the book and no arithmetic; with no refusal kept for this round the book is named and no figure is
    made up. The refusal's own book names it also where the current view, restored after a failed walk, adds none."""
    from ouroboros.context_budget import REQUIRED_BOOK_UNFIT_KEY
    from ouroboros.context_view_facts import SIZE_REFUSAL_KEY, remember_size_refusal
    from ouroboros.loop_transport import provider_terminal_fallback_text
    from ouroboros.physical_attempt_context import PhysicalAttemptContext

    flags = dict(round_idx=1, is_context_overflow=True, is_transport_wait=False, waited_sec=0, interactive=False,
                 is_deadline_exhausted=False)
    physical = PhysicalAttemptContext(profile="owner_low", rendered_mode="low", measurement_basis="fresh_route_usage",
                                      route_fp="r", round_id="e:round:1", target_total_tokens=None,
                                      capacity_total_tokens=240_000, context_target_miss=False, automatic_pass_used=False,
                                      measurement_density=1.0, response_reserve_tokens=65_536, required_book_tokens=51_000)
    refused = SimpleNamespace(physical_context=physical, estimated_input_tokens=190_000, predicted_free_tokens=-15_536,
                              capacity_total_tokens=240_000, response_reserve_tokens=65_536)
    usage = {"execution_id": "e", "_context_required_book_tokens": 51_000}
    remember_size_refusal(SimpleNamespace(accumulated_usage=usage, context_fit_plan=None, round_idx=1, active_model="m"),
                          refused, measured=False)
    text = provider_terminal_fallback_text(usage, **flags)
    assert "directly requires the whole development handbook (~51,000 tokens)" in text
    assert "without it the input is ~139,000 against 174,464 usable" in text
    assert "larger window" in text and "Context mode" in text and "without this requirement" in text
    # Both provider refusals and host context_overflow forecasts now offer Continue; it keeps the required book.
    forecast = {**usage, "reason_code": "context_overflow"}
    assert "a Continue keeps it" in text and "a Continue keeps it" in provider_terminal_fallback_text(forecast, **flags)
    assert "a Continue keeps it" in provider_terminal_fallback_text({**forecast, REQUIRED_BOOK_UNFIT_KEY: True}, **flags)
    # A restored Max primary holds the book with no addition (0): the refusing candidate's own facts still stand.
    assert provider_terminal_fallback_text({**usage, "_context_required_book_tokens": 0}, **flags) == text
    sent = {key: value for key, value in usage[SIZE_REFUSAL_KEY].items() if key not in {"book", "capacity"}}
    stated = {**sent, "basis": "provider_stated"}
    sized = provider_terminal_fallback_text({**usage, SIZE_REFUSAL_KEY: {**stated, "book": 51_000}}, **flags)
    assert "handbook (~51,000 tokens), which stayed. The choices" in sized and "without it" not in sized
    for slot in (stated, {}, {**usage[SIZE_REFUSAL_KEY], "round": 0}):  # no book of this round's own refusal
        named = provider_terminal_fallback_text({**usage, SIZE_REFUSAL_KEY: slot}, **flags)
        assert "handbook, which stayed. The choices" in named and "without it" not in named and "~51,000" not in named
        assert "handbook" not in provider_terminal_fallback_text({"execution_id": "e", SIZE_REFUSAL_KEY: slot}, **flags)


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
    """A primary outage walks to a configured route the book does not fit: the request is refused there and the
    primary's view stays. The book costs the candidate no form: its memory is exactly what the same move shows
    without the book."""
    from ouroboros import config, fallback_cooldown
    from tests.test_primary_view_return import _registry

    primary, reserve = "openai/gpt-primary", "openai/gpt-reserve"
    ordinary = _plan(tmp_path, _core(tmp_path), "low", 1_000_000)
    small = math.ceil(ordinary.low_projection.estimated_tokens * RATIO) + RESERVE + 2_000  # holds it without the book
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_kw: (
        {"model": task["model"], "provider": "openrouter", "base_url": "", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, route_fp=task["model"],
                        window_tokens={primary: 1_000_000, reserve: small}[task["model"]])))
    monkeypatch.setattr(config, "fallback_candidate_targets", lambda **_kw: [SimpleNamespace(model_id=reserve)])

    def walk(root, plan):
        fallback_cooldown.reset_for_tests()
        registry = _registry(root)
        registry._ctx.primary_route["model"] = registry._ctx.active_model = primary
        plan = replace(plan, model=primary, model_role="main")
        messages = plan.messages_for("low")
        shown, sent = messages[0], {}

        def refused_round(call):
            sent.update(view=call.messages[0], plan=call.context_fit_plan, mode=call.active_context_mode)
            call.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0, call.active_context_mode

        monkeypatch.setattr(loop, "_call_round_model", refused_round)
        msg, model, _local, _after, mode = loop._run_cross_model_fallback_chain(
            llm=SimpleNamespace(), ctx=registry._ctx, tools=registry, messages=messages, active_model=primary,
            active_use_local=False, tool_schemas=SCHEMAS, active_effort="medium", max_retries=3, drive_logs=root,
            task_id="outage", round_idx=2, event_queue=None, accumulated_usage={"_last_llm_error_kind": "transport_unavailable"},
            task_type="", emit_progress=lambda *_a, **_kw: None, context_fit_plan=plan, active_context_mode="low")
        assert msg is None and model == primary and mode == "low" and sent["mode"] == "low"
        assert sent["view"] == shown  # the candidate carried the same memory (and the book, if any): nothing paid
        assert messages[0] == shown  # the primary's own view is untouched by the refused candidate
        return sent

    (tmp_path / "with").mkdir()
    (tmp_path / "without").mkdir()
    book = walk(tmp_path / "with", _plan(tmp_path, _core(tmp_path, required=True, book_tokens=60_000), "low", 1_000_000))
    plain = walk(tmp_path / "without", ordinary)
    assert len(_book_blocks(book["plan"], "low")) == 1 and not _book_blocks(plain["plan"], "low")  # the book rode along
    floors = [walked["plan"].low_projection.memory_facts["floor"] for walked in (book, plain)]
    assert floors[0]["steps"] == floors[1]["steps"] and floors[0]["runs"] == floors[1]["runs"]  # nothing paid for it


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
    [start] = [frame["data"] for frame in agent._event_queue.queue if frame.get("data", {}).get("type") == "task_started"]
    assert (start["task_id"], start["task_attempt"]) == (task_id, 2)
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


def test_first_conversation_turn_names_the_same_attempt_at_start_and_finish(tmp_path, monkeypatch):
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.task_status import execution_owner_record
    from supervisor.worker_chat_lane import _seed_task

    repo, drive, ran = _agent_world(tmp_path, monkeypatch, 150_000)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **kw: None)
    task = _seed_task(7, "Check the inventory.", None, None, None)
    task["_execution_owner"] = execution_owner_record(drive, task, "direct")
    agent = _real_agent(repo, drive)
    events = agent._handle_task_scoped(task)
    start, = [frame["data"] for frame in agent._event_queue.queue if frame.get("data", {}).get("type") == "task_started"]
    metrics, = [row for row in events if row["type"] == "task_metrics"]
    done, = [row for row in events if row["type"] == "task_done"]
    assert ran == [task["id"]] and done["status"] == "completed"
    assert start["task_attempt"] == metrics["task_attempt"] == done["execution_owner"]["task_attempt"] == 0


def test_a_crash_retried_conversation_turn_names_its_own_attempt_at_start_and_finish(tmp_path, monkeypatch):
    """The lane's crash readmission (``conversation_lane.readmit_after_crash``) is the retry's assignment: its first
    ``task_started`` names the retry's attempt, as its metrics and terminal do, never the first admission's 0."""
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.task_status import execution_owner_record
    from supervisor import conversation_lane, workers
    from supervisor.worker_chat_lane import _seed_task

    repo, drive, ran = _agent_world(tmp_path, monkeypatch, 150_000)
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **kw: None)
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": None)
    seed = _seed_task(7, "Check the inventory.", None, None, None)
    seed["_execution_owner"] = execution_owner_record(drive, seed, "direct")  # its first admission: physical 0
    with conversation_lane._queue_lock:
        meta = conversation_lane.readmit_after_crash({**seed, "_attempt": 2}, {})
    workers.RUNNING.clear()  # the carrier is another process: the server's row is not the agent's input
    agent = _real_agent(repo, drive)
    events = agent._handle_task_scoped(dict(meta["task"]))
    start, = [frame["data"] for frame in agent._event_queue.queue if frame.get("data", {}).get("type") == "task_started"]
    metrics, = [row for row in events if row["type"] == "task_metrics"]
    done, = [row for row in events if row["type"] == "task_done"]
    assert meta["attempt"] == 2 and ran == [seed["id"]] and done["status"] == "completed"
    assert start["task_attempt"] == metrics["task_attempt"] == done["execution_owner"]["task_attempt"] == 2


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
