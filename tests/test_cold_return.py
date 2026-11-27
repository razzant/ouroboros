"""A returned work's window park: its proof, its offer of the current overview, and the owner's two choices.

The whole path runs in the scenarios (``tests/system_e2e/test_cold_return_window_park.py``). These read what a
scenario cannot see byte for byte or reach without a race: that the request a park counts with the current overview
is rendered by the overview's own sections (an outlined index beside the address of its own full text), that a round
after a rebuilt, unsent view keeps the park's proof without another floor count, that a choice is honoured only for
the hold it names and outlives the grant that carried it, and that the loop's ordinary controls decide before the
owner's chosen end of an attempt.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import cold_return, held_notes, knowledge, loop, usage_accounting
from ouroboros.budget_pause import (
    RAIL_OWNER_PAUSE,
    STATE_PAUSED,
    STATE_RESUME_GRANTED,
    budget_pause_row,
    set_budget_pause,
)
from ouroboros.context_view_facts import SIZE_REFUSAL_KEY
from ouroboros.llm import LLMClient
from ouroboros.owner_wait import continuation_state
from ouroboros.task_results import STATUS_RUNNING, write_task_result
from ouroboros.tools.registry import ToolRegistry
from ouroboros.tools.tool_context import ToolContext
from tests.test_held_notes import TASK, _plan, _write, world  # noqa: F401 - ``world`` is a pytest fixture

FACTS = {"input_tokens": 73_308, "window_tokens": 80_000, "reply_floor_tokens": 8_192, "over_tokens": 1_500,
         "model": "local-model", "basis": "exact_local_count"}
OFFER = {"held_revision": "rev-held", "current_revision": "rev-current", "input_tokens": 70_808}
_ARM = cold_return.arm  # the shipped arming, for a run that replaces the loop's call of it


def _park(**row) -> dict:
    """A window park's durable row as ``budget_pause.request_pause`` writes it."""
    facts = {**FACTS, **({cold_return.OVERVIEW_KEY: dict(row.pop("offer"))} if "offer" in row else {})}
    return {"pause_id": "park-1", "pause_generation": 3, "state": STATE_PAUSED, "reason": "authority",
            "rail": cold_return.RAIL_CONTEXT_WINDOW, "cause": cold_return.CAUSE, cold_return.FACTS_KEY: facts,
            "exact_continuation": True, "auto_resume": False, **row}


def _named(row: dict, **action) -> dict:
    return {"pause_id": row["pause_id"], "pause_generation": row["pause_generation"], **action}


def _flat(rows: list) -> list:
    """The rows as the built-in lane sends and counts them: every text block of a row joined into one string."""
    return LLMClient()._local_text_messages(rows)


# ---------------------------------------------------------------------------------------------------------------
# The current overview in the held one's place: counted on the request the next restore renders.
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_the_request_counted_with_the_current_overview_is_the_one_the_next_restore_renders(world, mode):  # noqa: F811
    """A restored work carries the overview it captured. ``held_notes.current_overview`` names what a start now
    carries in its place, each section in the form this request shows it, and ``cold_return._exchanged`` puts
    exactly those texts into the request the park counts.
    Saved under ``held_notes.holding`` (a ``Taking``), the same work restores to that very request: nothing else moved, and the
    earlier overview stays readable at its revision."""
    kept = _write(world, "moving", "A held head that stays as this work captured it. " * 40, summary="Its summary.")
    earlier = _write(world, "overview", "The earlier overview. " * 300 + "\n\n## Below the head\nStill whole.",
                     title="Earlier title", summary="Earlier summary.")
    env = world[0]

    def context():
        ctx = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id=TASK["id"])
        ctx.task_attempt = 1
        return ctx

    def restore(saved):
        cold = context()
        cold._held_note_capture = held_notes.restored_capture(saved)
        plan = cold.context_fit_plan = _plan(world, 400_000, mode=mode, ctx=cold)
        cold.active_context_mode = plan.initial_mode
        return cold, plan.messages_for(plan.initial_mode)

    ctx = context()
    write_task_result(ctx.drive_root, ctx.task_id, STATUS_RUNNING, chat_id=TASK["chat_id"])
    started = ctx.context_fit_plan = _plan(world, 400_000, mode=mode, ctx=ctx)
    ctx.active_context_mode = started.initial_mode
    rows = started.messages_for(started.initial_mode)
    saved = json.loads(json.dumps(continuation_state(ctx, rows, {}, {}, 1, [], set())))
    assert held_notes.current_overview(ctx) is None, "a work that restored nothing holds no earlier overview"

    current = _write(world, "overview", "The current overview, shortened.", revision=earlier.revision,
                     title="Current title", summary="Current summary.")
    cold, restored = restore(saved)
    before = _flat(restored)
    assert before[0]["content"].count("The earlier overview. ") == 300 and "The current overview" not in before[0]["content"]
    assert earlier.revision in before[0]["content"] and current.revision not in before[0]["content"]

    forms = held_notes.current_overview(cold)
    assert (forms["held"], forms["current"]) == (earlier.revision, current.revision)
    exchanged = cold_return._exchanged(before, forms["swaps"])
    assert exchanged is not None and exchanged[1:] == before[1:], "only the system row's overview texts are exchanged"
    assert all(new is old for new, old in zip(exchanged[1:], before[1:])), "every other row is the very same row"
    assert "The earlier overview. " not in exchanged[0]["content"] and earlier.revision not in exchanged[0]["content"]
    assert exchanged[0]["content"].count("The current overview, shortened.") == 1

    notice = cold_return._overview_notice(forms)
    measured = []

    def price(candidate):
        measured.append(candidate)
        return FACTS["input_tokens"] if candidate == before else 61_000

    facts = cold_return._overview_forms(_limit(cold, _Counter(price), restored), FACTS)
    assert facts is not None and facts["input_tokens"] == 61_000
    assert len(measured) == 2 and measured[0] == before
    counted = measured[1]
    # This projection has no wire clock. The notice is the one extra tail row, not part of _exchanged.
    assert counted == [*exchanged, *_flat([notice])]
    assert counted.count(_flat([notice])[0]) == 1
    taking = held_notes.Taking(cold, forms, restored, notice)
    with held_notes.holding(taking):
        taken = json.loads(json.dumps(continuation_state(cold, restored, {}, {}, 1, [], set())))
    assert taking.seen == "accepted" and held_notes.current_taking() is None, "decided at the build, bound to it alone"
    assert taken["messages"] == [*restored, notice], "the saved transcript is a copy that ends with the host's row"
    # Below, ``_flat(restore(x)[1])`` compares the restored plan's INITIAL projection of the capture ``x`` holds (the
    # rendered request), never the saved transcript, which is compared above on its own.
    untaken = json.loads(json.dumps(continuation_state(cold, restored, {}, {}, 1, [], set())))
    assert _flat(restore(untaken)[1]) == before, "without the choice the same work keeps the overview it held"
    again, rendered = restore(taken)
    assert _flat(rendered) == counted, "the restore projection matches the complete candidate priced by the park"
    assert kept.revision in counted[0]["content"], "another held head keeps the revision this work captured"
    assert held_notes.current_overview(again) is None, "it now holds the current overview: nothing more to offer"
    # A later pause of that work captures the overview it now holds, and the earlier one stays readable.
    later = json.loads(json.dumps(continuation_state(again, rendered, {}, {}, 2, [], set())))
    assert _flat(restore(later)[1]) == counted
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
    read = registry.execute("knowledge_read", {"topic": "overview", "scope": "global", "version": earlier.revision})
    assert read.endswith(earlier.text), "the replaced revision is still read exactly"


def _returned_work(world, mode="max"):  # noqa: F811
    """A work saved on an earlier overview and restored after the shelf moved to a shorter one:
    ``(restore, cold, rows, forms, earlier, current)``."""
    earlier = _write(world, "overview", "The earlier overview. " * 300, title="Earlier title", summary="Earlier summary.")
    env = world[0]

    def context():
        ctx = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id=TASK["id"])
        ctx.task_attempt = 1
        return ctx

    ctx = context()
    write_task_result(ctx.drive_root, ctx.task_id, STATUS_RUNNING, chat_id=TASK["chat_id"])
    started = ctx.context_fit_plan = _plan(world, 400_000, mode=mode, ctx=ctx)
    ctx.active_context_mode = started.initial_mode
    saved = json.loads(json.dumps(continuation_state(ctx, started.messages_for(started.initial_mode), {}, {}, 1, [], set())))
    current = _write(world, "overview", "The current overview, shortened.", revision=earlier.revision,
                     title="Current title", summary="Current summary.")

    def restore(state):
        cold = context()
        cold._held_note_capture = held_notes.restored_capture(state)
        plan = cold.context_fit_plan = _plan(world, 400_000, mode=mode, ctx=cold)
        cold.active_context_mode = plan.initial_mode
        return cold, plan.messages_for(plan.initial_mode)

    cold, rows = restore(saved)
    return restore, cold, rows, held_notes.current_overview(cold), earlier, current


@pytest.mark.parametrize("seen", ["accepted", "moved", "lock_unavailable", "note_unreadable", "foreign_error"])
def test_a_chosen_overview_is_decided_by_one_note_read_under_the_writers_lock_and_never_fails_the_save(  # noqa: PLR0915
        world, monkeypatch, seen):  # noqa: F811
    """The decision reads the one overview note under the shelf lock an overview write takes (no scan of the shelf,
    its claims or its history), compares its revision with the counted one and fixes the save: accepted, the saved
    state holds the chosen forms and a transcript ending with the host's row; the shelf moved, the lock unavailable
    or the note unreadable, it keeps the pins, rows and transcript the work held and nothing raises."""
    restore, cold, rows, forms, earlier, current = _returned_work(world)
    before, untaken = _flat(rows), json.loads(json.dumps(continuation_state(cold, rows, {}, {}, 1, [], set())))
    assert forms["current"] == current.revision
    if seen == "moved":
        _write(world, "overview", "A third overview, written after the count.", revision=current.revision,
               title="Newer title", summary="Newer summary.")
    events, real_lock, real_read = [], knowledge.knowledge_write_lock, knowledge.read_knowledge_note

    @contextlib.contextmanager
    def lock(shelf):
        events.append("lock")
        with real_lock(shelf):
            yield
        events.append("unlock")

    def read(address):
        events.append("read")
        return real_read(address)

    def refuse(error):
        def raised(*_a, **_k):
            raise error
        return raised

    notice = cold_return._overview_notice(forms)
    taking = held_notes.Taking(cold, forms, rows, notice)
    with monkeypatch.context() as patch:
        patch.setattr(knowledge, "knowledge_write_lock", lock)
        patch.setattr(knowledge, "read_knowledge_note", read)
        if seen == "lock_unavailable":
            patch.setattr(knowledge, "knowledge_write_lock", refuse(OSError("no lock")))
        elif seen == "note_unreadable":
            patch.setattr(knowledge, "read_knowledge_note", refuse(FileNotFoundError("the note is gone")))
        elif seen == "foreign_error":
            patch.setattr(knowledge, "read_knowledge_note", refuse(RuntimeError("not this check's error")))
        walk = refuse(AssertionError("the decision is not a walk of the shelf or its claims"))
        patch.setattr(knowledge, "inventory_knowledge", walk)
        patch.setattr(knowledge, "active_claims", walk)
        if seen == "foreign_error":  # an error that is not an unavailable path, read or lock is not swallowed
            with held_notes.holding(taking), pytest.raises(RuntimeError, match="not this check"):
                continuation_state(cold, rows, {}, {}, 1, [], set())
            return
        with held_notes.holding(taking):
            built = json.loads(json.dumps(continuation_state(cold, rows, {}, {}, 1, [], set())))
    assert taking.seen == {"accepted": "accepted", "moved": "moved"}.get(seen, "unavailable")
    # ``restore(...)[1]`` is the restored plan's INITIAL projection of the capture the state holds, not the saved
    # transcript: the transcript is compared on its own, as ``built["messages"]``.
    projected = _flat(restore(built)[1])[0]["content"]
    if seen == "accepted":
        assert events == ["lock", "read", "unlock"], "one read of the one note, inside the writers' lock"
        assert taking.decided["messages"] == [*rows, notice] and built["messages"] == [*rows, notice]
        assert built[held_notes.CAPTURE_KEY]["companions"]["shared_understanding"] == json.loads(
            json.dumps(forms["hold"]["companion"]))
        assert current.revision in projected and earlier.revision not in projected
    else:
        assert taking.decided is None
        assert built["messages"] == rows and built[held_notes.CAPTURE_KEY] == untaken[held_notes.CAPTURE_KEY], \
            "every pin, row and the transcript are exactly the held ones"
        assert earlier.revision in projected and current.revision not in projected, "the held overview, as it was"


@pytest.mark.parametrize("wait", ["the_overview_is_written", "the_lock_is_lost", "nothing_changes"])
def test_a_real_window_park_decides_its_pins_where_its_state_is_built_whatever_the_wait_did(  # noqa: F811
        world, monkeypatch, wait):
    """The owner chose the counted revision. While the real ``request_pause`` waits for a producer that is not yet
    settled (after the count, before the save), the overview is written or the shelf lock is lost. The save, stored
    before its row, then holds the exact pins and transcript the work held, its row carries no ``taken`` (nor the
    stale offer when the check saw another revision) and the work stays an ordinary window park that Resume returns.
    Where nothing changed, the same park takes the chosen overview."""
    from ouroboros import budget_pause
    from ouroboros.artifacts import read_actor_source_bytes
    from tests._budget_pause_exact_helpers import _fast_hold, _loop_ctx

    restore, cold, rows, forms, earlier, current = _returned_work(world)
    root, task_id = cold.drive_root, cold.task_id
    untaken = json.loads(json.dumps(continuation_state(cold, rows, {}, {}, 1, [], set())))
    shell, limit = _loop_ctx(root, task_id)
    for key in ("owner_wait_callback", "budget_drive_root", "task_started_at", "root_task_id", "_cost_ceiling",
                "model_wait_context", "_owner_directives", "_delivery_candidate"):
        setattr(cold, key, getattr(shell, key))
    limit.tools = SimpleNamespace(_ctx=cold)
    transcript = [{"role": "user", "content": "Continue the saved work."}]
    limit.messages = list(transcript)
    setattr(cold, cold_return._ARMED, (None, None, {"overview_revision": forms["current"]}))
    counted = {**forms, "input_tokens": OFFER["input_tokens"]}
    monkeypatch.setattr(cold_return, "refusal_facts", lambda *_a: dict(FACTS))
    monkeypatch.setattr(cold_return, "_overview_forms", lambda *_a: counted)
    _fast_hold(monkeypatch, budget_pause)
    observed = []

    def producers(_ctx, **_kw):
        observed.append(1)
        if len(observed) > 1:
            return {"quiescent": True}
        if wait == "the_overview_is_written":  # between the count and the save, during the producers' wait
            _write(world, "overview", "A third overview, written while the park waited.", revision=current.revision,
                   title="Newer title", summary="Newer summary.")
        elif wait == "the_lock_is_lost":
            def lost(_shelf):
                raise OSError("the shelf lock is unavailable")
            monkeypatch.setattr(knowledge, "knowledge_write_lock", lost)
        return {"quiescent": False}

    monkeypatch.setattr(budget_pause, "local_producer_observation", producers)
    try:
        with pytest.raises(budget_pause.BudgetPauseRequested) as parked:
            cold_return.park_refused(limit)
    finally:
        budget_pause.end_dispatch_fence(task_id)
    row = parked.value.pause
    state = json.loads(read_actor_source_bytes(root, task_id, row["source_ref"]))
    facts = row[cold_return.FACTS_KEY]
    assert len(observed) == 2, "the park waited once and saved after producers settled"
    assert row["rail"] == cold_return.RAIL_CONTEXT_WINDOW and row["cause"] == cold_return.CAUSE
    assert row["exact_continuation"] is True and row["auto_resume"] is False
    assert state["pause_id"] == row["pause_id"], "the stored source is the very one the row names"
    assert held_notes.current_taking() is None and limit.messages == transcript
    assert {key: facts[key] for key in FACTS} == FACTS
    offer = facts.get(cold_return.OVERVIEW_KEY)
    if wait == "nothing_changes":
        assert offer == {**OFFER, "held_revision": earlier.revision, "current_revision": current.revision, "taken": True}
        assert state["messages"] == [*transcript, cold_return._overview_notice(forms)]
        assert state[held_notes.CAPTURE_KEY]["companions"]["shared_understanding"] == json.loads(
            json.dumps(forms["hold"]["companion"]))
        assert cold_return.park_wait_text(facts) == cold_return.READY_TEXT
    else:
        assert state["messages"] == transcript, "no host row of a choice nobody took"
        assert state[held_notes.CAPTURE_KEY] == untaken[held_notes.CAPTURE_KEY], "the held pins, exactly"
        if wait == "the_overview_is_written":
            assert offer is None and cold_return.park_wait_text(facts) == cold_return.WAIT_TEXT, "no stale offer"
        else:
            assert offer == {**OFFER, "held_revision": earlier.revision, "current_revision": current.revision}
            assert cold_return.park_wait_text(facts) == cold_return.OFFER_TEXT, "the plain offer, as of the last check"
    shown = cold_return.window_facts({**row, "state": STATE_PAUSED})
    assert shown["pause_id"] == row["pause_id"] and shown["over_tokens"] == FACTS["over_tokens"], \
        "a window park that an owner's Resume returns"


def test_no_current_overview_is_offered_when_the_shelf_has_none_a_start_would_hold(world):  # noqa: F811
    """Unchanged, archived away from the claim, or unauthored: a start now would not hold a newer overview here."""
    earlier = _write(world, "overview", "The earlier overview.")
    env = world[0]
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id=TASK["id"])
    ctx.task_attempt = 1
    write_task_result(ctx.drive_root, ctx.task_id, STATUS_RUNNING, chat_id=TASK["chat_id"])
    plan = ctx.context_fit_plan = _plan(world, 400_000, ctx=ctx)
    ctx.active_context_mode = plan.initial_mode
    saved = json.loads(json.dumps(continuation_state(ctx, plan.messages_for(plan.initial_mode), {}, {}, 1, [], set())))

    def restored():
        cold = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id=TASK["id"])
        cold.task_attempt = 1
        cold._held_note_capture = held_notes.restored_capture(saved)
        cold.context_fit_plan = _plan(world, 400_000, ctx=cold)
        return cold

    assert held_notes.current_overview(restored()) is None, "the shelf still holds the revision this work holds"
    unclaimed = _write(world, "overview", "The overview no longer claims this room.", revision=earlier.revision,
                       claims=["room:another"])
    assert held_notes.current_overview(restored()) is None, "a start here would not hold it"
    claimed = _write(world, "overview", "The overview claims this room again.", revision=unclaimed.revision,
                     claims=["room:1"])
    assert held_notes.current_overview(restored())["current"] == claimed.revision


def test_an_exchange_needs_every_held_text_exactly_once():
    rows = [{"role": "system", "content": "A\nheld section\nB\n- overview — Old\nC"}, {"role": "user", "content": "held section"}]
    swapped = cold_return._exchanged(rows, [("held section", "current section"), ("- overview — Old", "- overview — New")])
    assert swapped == [{"role": "system", "content": "A\ncurrent section\nB\n- overview — New\nC"}, rows[1]]
    assert swapped[1] is rows[1] and rows[0]["content"].startswith("A\nheld section"), "the rows given are not changed"
    assert cold_return._exchanged(rows, [("absent", "x")]) is None, "a held text the request does not carry"
    twice = [{"role": "system", "content": "held section and held section"}]
    assert cold_return._exchanged(twice, [("held section", "x")]) is None, "nor one it carries twice"
    assert cold_return._exchanged(rows, []) == rows, "nothing to exchange leaves the request as it is"
    assert cold_return._exchanged([{"role": "user", "content": "no system row"}], []) is None
    assert cold_return._exchanged([{"role": "system", "content": [{"type": "text", "text": "blocks"}]}], []) is None


def _index_world(*, outlined, held_row, current_row, predicted=False):
    """A captured knowledge index holding ``held_row`` as the overview's entry, in the form a request shows it."""
    from ouroboros import knowledge
    from ouroboros.section_floor import HostSection, predicted_section

    other = {"topic": "people/synthetic", "title": "A synthetic person", "summary": "Kept beside the overview.", "type": None}
    full = "## Knowledge base\n\n" + "\n".join(knowledge.index_entry(row) for row in (held_row, other))
    outline = "## Knowledge base\n\n" + "\n".join(knowledge.outline_entry(row) for row in (held_row, other))
    section = HostSection("knowledge_index", full, outline, reader="knowledge_list(scope='global')")
    receipt = {key: value for key, value in predicted_section(section).items() if predicted or key != "predicted"}
    facts = {"host_sections_addressed": [receipt]} if outlined else {}
    plan = SimpleNamespace(core=SimpleNamespace(dynamic_head_sections=(HostSection("shared_understanding", "pinned"), section)),
                           initial_mode="nano", projection=lambda _mode: SimpleNamespace(memory_facts=facts))
    ctx = SimpleNamespace(active_context_mode="nano")
    held, entry = SimpleNamespace(index_row=held_row), SimpleNamespace(index_row=current_row)
    return ctx, plan, held, entry, section, receipt


def test_an_outlined_index_is_exchanged_beside_the_address_of_its_own_full_text():
    """The outline shows one line per note and a reader for the retained whole index. A new summary changes no
    outline line of an ordinary note, yet the whole text is another text at another address: the exchanged request
    carries the address its own capture would publish, never the held one's."""
    from ouroboros.section_floor import predicted_section

    held_row = {"topic": "overview", "title": "Orientation", "summary": "The earlier summary.", "type": None}
    summary = {**held_row, "summary": "The current summary, shorter."}
    ctx, plan, held, entry, section, receipt = _index_world(outlined=True, held_row=held_row, current_row=summary)
    [(shown, exchanged)] = held_notes._index_exchange(ctx, plan, held, entry)
    assert shown == section.view_for_receipt(receipt, {}) and receipt["reader"] in shown
    assert shown.partition("\n\n[Host section details")[0] == exchanged.partition("\n\n[Host section details")[0], \
        "the outline itself did not change"
    whole = section.full_text.replace("  The earlier summary.", "  The current summary, shorter.")
    assert whole != section.full_text
    current_reader = predicted_section(replace_section(section, whole))["reader"]
    assert current_reader != receipt["reader"] and current_reader in exchanged and receipt["reader"] not in exchanged

    # A new title moves the outline's own line too, beside the address of that other whole text.
    titled = {**held_row, "title": "Orientation, second edition"}
    ctx, plan, held, entry, section, receipt = _index_world(outlined=True, held_row=held_row, current_row=titled)
    [(shown, exchanged)] = held_notes._index_exchange(ctx, plan, held, entry)
    assert "- overview — Orientation\n" in shown and "- overview — Orientation, second edition\n" in exchanged
    assert receipt["reader"] not in exchanged and "A synthetic person" in exchanged

    # Shown whole, the pair is the two whole texts; alike rows exchange nothing.
    ctx, plan, held, entry, section, _receipt = _index_world(outlined=False, held_row=held_row, current_row=summary)
    [(shown, exchanged)] = held_notes._index_exchange(ctx, plan, held, entry)
    assert shown == section.full_text and "The current summary, shorter." in exchanged and "earlier summary" not in exchanged
    assert held_notes._index_exchange(ctx, plan, held, SimpleNamespace(index_row=dict(held_row))) == []
    # A receipt that was only predicted reached no sent view: the request shows the index whole.
    ctx, plan, held, entry, section, _receipt = _index_world(outlined=True, held_row=held_row, current_row=summary, predicted=True)
    assert held_notes._index_exchange(ctx, plan, held, entry)[0][0] == section.full_text
    # An entry the captured index does not carry exactly once cannot be placed: nothing comparable.
    stranger = SimpleNamespace(index_row={**held_row, "title": "Never rendered here"})
    assert held_notes._index_exchange(ctx, plan, stranger, entry) is None


def replace_section(section, full_text):
    from dataclasses import replace

    return replace(section, full_text=full_text)


class _Counter:
    """``llm_local.recount_with`` as the park calls it: the count of the rows given, and of the rewritten ones."""

    def __init__(self, price, *, first=None):
        self.price, self.first, self.asked = price, first, []

    def recount_with(self, messages, tools, rewrite, *, refusal):
        self.asked.append((messages, tools, refusal))
        other = rewrite(_flat(messages))
        mine = self.price(_flat(messages)) if self.first is None else self.first
        return (mine, None if other is None else self.price(other)) if mine is not None else None


def _limit(ctx, llm, messages):
    refused = {"round": 1, "tokens": 73_308, "basis": "exact_local_count", "local_window": 80_000,
               "nano_reply_floor": {"floor_sufficient": True, "reply_floor_tokens": 8_192}}
    return SimpleNamespace(tools=SimpleNamespace(_ctx=ctx), llm=llm, messages=messages, tool_schemas=[], task_type="task",
                           accumulated_usage={SIZE_REFUSAL_KEY: refused}, round_idx=1)


def test_a_park_offers_the_current_overview_only_on_one_exact_form_where_it_fits(world):  # noqa: F811
    """The offer needs the same count again for the request as it stands (the refusal's own), an exact count of the
    exchanged one, and room for it with the reply floor. Anything less leaves the park without an offer."""
    earlier = _write(world, "overview", "EARLIER " * 50)
    env = world[0]
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id=TASK["id"])
    ctx.task_attempt = 1
    write_task_result(ctx.drive_root, ctx.task_id, STATUS_RUNNING, chat_id=TASK["chat_id"])
    plan = ctx.context_fit_plan = _plan(world, 400_000, ctx=ctx)
    ctx.active_context_mode = plan.initial_mode
    saved = json.loads(json.dumps(continuation_state(ctx, plan.messages_for(plan.initial_mode), {}, {}, 1, [], set())))
    current = _write(world, "overview", "CURRENT", revision=earlier.revision)
    cold = ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id=TASK["id"])
    cold.task_attempt = 1
    cold._held_note_capture = held_notes.restored_capture(saved)
    restored = cold.context_fit_plan = _plan(world, 400_000, ctx=cold)
    rows = restored.messages_for(restored.initial_mode)

    def price(flat):  # 1,000 per word of either overview, on top of a fixed background
        text = flat[0]["content"]
        return 60_000 + 1_000 * (text.count("EARLIER ") + text.count("CURRENT"))

    held_count, facts = price(_flat(rows)), dict(FACTS)
    assert held_count > 60_000 + 50_000 - 1 and price(cold_return._exchanged(
        _flat(rows), held_notes.current_overview(cold)["swaps"])) == 61_000
    facts.update(input_tokens=held_count, over_tokens=held_count + 8_192 - 80_000)
    counter = _Counter(price)
    forms = cold_return._overview_forms(_limit(cold, counter, rows), facts)
    assert (forms["held"], forms["current"], forms["input_tokens"]) == (earlier.revision, current.revision, 61_000)
    assert counter.asked[0][0] is rows, "the rows the park is about to save are the rows counted"
    assert counter.asked[0][2]["reply_floor_tokens"] == 8_192, "with the refused request's own reply floor"
    # The request as it stands must count as it was refused: another number is another request.
    assert cold_return._overview_forms(_limit(cold, _Counter(price, first=held_count + 1), rows), facts) is None
    def with_notice(flat):
        return price(flat) + 73 * sum("The owner chose the current overview" in row.get("content", "") for row in flat)

    priced = cold_return._overview_forms(_limit(cold, _Counter(with_notice), rows), facts)
    assert priced["input_tokens"] == 61_073
    assert cold_return._overview_forms(_limit(cold, _Counter(with_notice), rows),
                                       {**facts, "window_tokens": 61_073 + 8_192 - 1}) is None
    # The exchanged request must itself fit with the reply floor.
    tight = {**facts, "window_tokens": 61_000 + 8_192 - 1}
    assert cold_return._overview_forms(_limit(cold, _Counter(price), rows), tight) is None
    assert cold_return._overview_forms(_limit(cold, _Counter(price), rows), {**tight, "window_tokens": 61_000 + 8_192})
    # No exact count of either (a busy counter), a lane with no counter, or nothing held: no offer.
    assert cold_return._overview_forms(_limit(cold, _Counter(lambda _flat: None), rows), facts) is None
    assert cold_return._overview_forms(_limit(cold, SimpleNamespace(), rows), facts) is None
    assert cold_return._overview_forms(_limit(ctx, _Counter(price), rows), facts) is None
    # The exchanged system row must state the current revision where the overview's own sections render it.
    real = held_notes.current_overview
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(held_notes, "current_overview",
                        lambda tool_ctx: {**real(tool_ctx), "current": "a revision the exchanged row does not state"})
        assert cold_return._overview_forms(_limit(cold, _Counter(price), rows), facts) is None
    # An independent reference to the held revision elsewhere in the system row is not the overview's own place:
    # it survives the exchange and does not withdraw the offer (only the overview's sections are swapped).
    forms_now = real(cold)
    wire = _flat(rows)
    quoted = [{**wire[0], "content": f"Superseded revision {forms_now['held']} is kept for reference.\n"
                                   + wire[0]["content"]}, *wire[1:]]
    assert cold_return._overview_forms(_limit(cold, _Counter(price), quoted), facts) is not None


class _Shelf:
    """The shelf as a window park's save finds it where its state is built: ``held_notes.Taking.accepts`` with the
    outcome a test names (``accepted``, ``moved`` or ``unavailable``). It never touches a real data root."""

    def __init__(self, monkeypatch, outcome="accepted"):
        self.outcome = outcome
        shelf = self

        def accepts(taking):  # a plain function on the class, so ``taking.accepts()`` binds the taking
            taking.seen = shelf.outcome
            return shelf.outcome == "accepted"

        monkeypatch.setattr(held_notes.Taking, "accepts", accepts)


def _saving(saved, forms):
    """``budget_pause.request_pause`` as far as a window park uses it: the seed's facts and what the facts are once
    the save has been built (``settled_facts``), with the ``Taking`` bound while it is built and its decision."""
    def request_pause(limit_ctx, **kwargs):
        live, taking = list(limit_ctx.messages), held_notes.current_taking()
        if taking is not None:
            assert taking.forms is forms and taking.rows is limit_ctx.messages, "the park binds what it counted"
            assert taking.notice == cold_return._overview_notice(forms)
            taking.accepts()
        seed = kwargs["facts"][cold_return.FACTS_KEY]
        settled = cold_return.settled_facts({cold_return.FACTS_KEY: seed})[cold_return.FACTS_KEY]
        saved.append((kwargs, taking, seed, settled))
        assert limit_ctx.messages == live, "the live transcript is never changed"
    return request_pause


def test_a_park_takes_the_current_overview_only_for_the_very_revision_its_resume_chose(monkeypatch):
    """What a park records, and whether its save holds the current overview: only when this start's grant chose the
    revision the park counted and the shelf still holds it where the save is built. A choice the shelf moved past, a
    form that no longer fits and a count that could not be taken change nothing: the work stays on the overview it
    holds. The seed and the live transcript never say the choice was taken. A save that saw the shelf hold another
    revision offers the stale one no longer; one whose check could not be made keeps the plain offer."""
    saved, forms = [], {"held": "rev-held", "current": "rev-current", "input_tokens": 70_808,
                        "hold": {"entry": {"topic": "overview"}, "companion": {"section_id": "shared_understanding"}}}
    shelf = _Shelf(monkeypatch)
    monkeypatch.setattr("ouroboros.budget_pause.request_pause", _saving(saved, forms))
    monkeypatch.setattr(cold_return, "refusal_facts", lambda *_a: dict(FACTS))

    prior = {"role": "user", "content": "The earlier conversation remains."}

    def park(chosen, offered, outcome="accepted"):
        saved.clear()
        shelf.outcome = outcome
        monkeypatch.setattr(cold_return, "_overview_forms", offered if callable(offered) else lambda *_a: offered)
        ctx = SimpleNamespace()
        setattr(ctx, cold_return._ARMED, (None, None, dict(chosen)))
        rows = [prior]
        cold_return.park_refused(SimpleNamespace(tools=SimpleNamespace(_ctx=ctx), accumulated_usage={}, round_idx=1, messages=rows))
        [(kwargs, taking, seed, settled)] = saved
        assert rows == [prior]
        assert (kwargs["rail"], kwargs["cause"], kwargs["scope"]) == (cold_return.RAIL_CONTEXT_WINDOW, cold_return.CAUSE, "task")
        assert {key: seed[key] for key in FACTS} == FACTS, "the refusal's own numbers ride every park"
        assert "taken" not in (seed.get(cold_return.OVERVIEW_KEY) or {}), "the seed never claims the choice"
        assert {key: settled[key] for key in FACTS} == FACTS
        assert held_notes.current_taking() is None, "the taking is bound for that one park only"
        return settled.get(cold_return.OVERVIEW_KEY), (taking.forms["hold"] if taking and taking.seen == "accepted" else None)

    assert park({}, forms) == (OFFER, None), "offered; nothing is taken without the owner's choice"
    assert saved[0][1] is None, "and no save is bound to the shelf then"
    assert park({"overview_revision": "rev-current"}, forms) == ({**OFFER, "taken": True}, forms["hold"])
    assert park({"overview_revision": "rev-current"}, forms, "moved") == (None, None), \
        "the shelf holds another revision now: the stale one is neither taken nor offered again"
    assert park({"overview_revision": "rev-current"}, forms, "unavailable") == (OFFER, None), \
        "a check that could not be made takes nothing and leaves the plain offer"
    assert park({"overview_revision": "rev-offered-before"}, forms) == (OFFER, None), "the shelf moved since that choice"
    assert park({"end_attempt": True}, forms) == (OFFER, None)
    assert park({"overview_revision": "rev-current"}, None) == (None, None), "no form fits now: nothing to take"

    def unavailable(*_a):
        raise RuntimeError("the counter could not be asked")

    assert park({"overview_revision": "rev-current"}, unavailable) == (None, None), "the park itself needs no count"
    monkeypatch.setattr(cold_return, "refusal_facts", lambda *_a: {})
    saved.clear()
    cold_return.park_refused(SimpleNamespace(tools=SimpleNamespace(_ctx=SimpleNamespace()), accumulated_usage={}, round_idx=1))
    assert saved == [], "without the proof nothing parks: the ordinary terminal follows"


# ---------------------------------------------------------------------------------------------------------------
# The proof a park needs, and the hold a choice names.
# ---------------------------------------------------------------------------------------------------------------

def _slot(**changed) -> dict:
    return {"round": 4, "model": "local-model", "kind": "input", "tokens": 73_308, "basis": "exact_local_count",
            "local_window": 80_000, "reply_limit": 8_192,
            "nano_reply_floor": {"reply_floor_tokens": 8_192}, **changed}


def test_only_an_exact_refusal_with_nothing_sent_since_the_return_is_a_parks_proof():
    def facts(*, arm=True, sent=False, usage_extra=None, **slot):
        ctx, usage = SimpleNamespace(), {"_last_llm_error_kind": "context_overflow"}
        if arm:
            cold_return.arm(ctx, usage, {"messages": []})
        if sent:
            usage_accounting._LAST_PHYSICAL_ATTEMPT.set(object())
        usage[SIZE_REFUSAL_KEY] = _slot(**slot)
        usage.update(usage_extra or {})
        return cold_return.refusal_facts(ctx, usage, 4)

    run = lambda **kw: contextvars.copy_context().run(lambda: facts(**kw))  # noqa: E731 - each case in its own context
    assert run() == {"input_tokens": 73_308, "window_tokens": 80_000, "reply_floor_tokens": 8_192,
                     "over_tokens": 1_500, "model": "local-model", "basis": "exact_local_count"}
    assert run(arm=False) == {}, "a start that restored nothing keeps the ordinary terminal"
    assert run(sent=True) == {}, "something was sent since the return"
    assert run(basis="known_window_prediction") == {} and run(basis="provider_stated") == {}
    assert run(nano_reply_floor=None) == {}, "an exact count without the scoped floor is not a park proof"
    assert run(tokens=71_808) == {} and run(tokens=70_000) == {}
    assert run(round=3) == {}, "an earlier round's refusal is not this round's"
    assert run(usage_extra={"_last_llm_error_kind": "deadline_exhausted"}) == {}
    assert run(usage_extra={"_pending_transport_outcome": {"attempt": "a"}}) == {}
    assert run(usage_extra={"_transport_deaths": {"count": 1}}) == {}


def _exact_measurement(payload, count, window=80_000):
    from ouroboros.local_model_server import input_fingerprint

    digest = input_fingerprint(payload)
    return {"supported": True, "input_is_exact": True, "input_tokens": count, "process_id": 123,
            "context_window": window, "raw_context_window": window + 128, "native_input_sha256": digest,
            "output_limit_enforced": True, "reasoning_included_in_limit": True,
            "tokenizer_template_provenance": {"source": "serving_llama_formatter", "formatter": "fixture",
                "library_version": "fixture", "chat_format": "fixture", "launched_n_ctx": window,
                "rendered_prompt_sha256": digest}}


@pytest.mark.parametrize("window, count, book, preferred, entry, parks", [
    (80_000, 71_809, 0, "nano", True, True),
    (80_000, 70_000, 2_000, "nano", True, False),
    (80_000, 71_808, 2_000, "nano", True, False),
    (80_000, 71_809, 0, "max", False, False),
    (80_000, 71_809, 0, "nano", False, False),
    (120_000, 111_809, 0, "nano", True, False),
])
def test_only_the_scoped_reply_limiter_supplies_the_park_proof(monkeypatch, window, count, book, preferred, entry, parks):
    from ouroboros.context_budget import LocalContextTooLargeError
    from ouroboros.context_view_facts import remember_size_refusal
    from ouroboros.llm_attempt import _attempt_request, bound_reply_allowance, main_context_fit_refusal
    from ouroboros.loop_model_call import _refused_candidate

    monkeypatch.setattr("ouroboros.response_limits.response_limit_for_target",
                        lambda target: SimpleNamespace(ceiling=lambda value: value))
    physical = usage_accounting.PhysicalAttemptContext(
        "owner_nano" if preferred == "nano" else "task_local_nano", "nano", "cold_estimate", "local", "x:round:4",
        None, window, False, False, view_kind="owner_turn", response_reserve_tokens=8_192,
        required_book_tokens=book, preferred_mode=preferred,
        nano_floor_plan=(object(), "cold_resume") if entry else None)
    payload = {"model": "local-model", "messages": [{"role": "user", "content": "ordinary input"}], "max_tokens": 20_000}
    target = {"provider": "local", "context_window_tokens": window, "context_window_confirmed": True,
              "window_instance_id": "123", "local_dispatch_base_url": "http://127.0.0.1:1/v1",
              "local_input_measurement": _exact_measurement(payload, count, window)}
    with usage_accounting.bind_physical_attempt_context(physical):
        if book:
            candidate = bound_reply_allowance(target, payload)
            facts = main_context_fit_refusal(_attempt_request(target, candidate))
            assert facts["cause"] == "exact_local_count" and facts["free_tokens"] == -2_000
        else:
            with pytest.raises(LocalContextTooLargeError) as error:
                bound_reply_allowance(target, payload)
            facts = error.value.refused_candidate
    usage = {"execution_id": "x", "_context_route_fp": "local", "_last_llm_error_kind": "context_overflow"}
    tool_ctx = SimpleNamespace()
    cold_return.arm(tool_ctx, usage, {"messages": []})
    remember_size_refusal(SimpleNamespace(round_idx=4, accumulated_usage=usage, active_model="another view"),
                          _refused_candidate(facts), measured=True)
    parked = cold_return.refusal_facts(tool_ctx, usage, 4)
    if parks:
        assert parked["over_tokens"] == 1 and parked["model"] == "local-model"
    else:
        assert parked == {}


@pytest.mark.parametrize("surface", ["main", "project", "direct_chat", "child"])
def test_reprojected_root_frame_gives_floor_proof_to_queued_roots_only(world, monkeypatch, surface):  # noqa: F811
    from ouroboros.context_budget import LocalContextTooLargeError
    from ouroboros.context_measurement import measure_main_fit
    from ouroboros.context_view_facts import remember_size_refusal
    from ouroboros.llm_attempt import bound_reply_allowance
    from ouroboros.loop_model_call import _physical_context_for_round, _refused_candidate
    from ouroboros.loop_round_limits import COLD_RETURN_ATTR
    from ouroboros.projects_registry import create_project

    task = dict(TASK)
    if surface == "project":
        project = create_project(world[0].drive_root, "window-project", name="Window Project")
        task.update(project_id=project["id"], chat_id=project["chat_id"])
    elif surface == "direct_chat":
        task["_is_direct_chat"] = True
    elif surface == "child":
        task.update(delegation_role="subagent", parent_task_id="parent")

    plan = _plan(world, mode="nano", task=task, resolver=lambda *_a, **_kw: (
        {"provider": "local", "model": "local-model"},
        SimpleNamespace(route_fp="local", status="asserted", stale=False, window_tokens=80_000)))
    queued = surface in {"main", "project"}
    slot = SimpleNamespace(task_metadata={"_is_direct_chat": surface == "direct_chat"})
    setattr(slot, COLD_RETURN_ATTR, True)
    usage = {"execution_id": "e", "_context_route_fp": "local", "_last_llm_error_kind": "context_overflow"}
    call = SimpleNamespace(tools=SimpleNamespace(_ctx=slot), active_use_local=True, task_type="task",
                           round_idx=4, accumulated_usage=usage, active_model="local-model")
    cold_return.arm(slot, usage, {"messages": []})

    def frame(candidate, rows):
        fitted = measure_main_fit(candidate, rows, [], profile="owner_nano", rendered_mode="nano",
                                  round_id=f"e:round:{call.round_idx}")
        return _physical_context_for_round(call, fitted, candidate)

    rows = plan.messages_for("nano")
    assert bool(frame(plan, rows).nano_floor_plan) == queued
    rebound = plan.reproject_for_route(window_tokens=80_000, known_window=True, ratio=1,
        output_reserve=plan.output_reserve_tokens, tool_schemas=[], start_mode="nano", current_messages=rows,
        select_mode="nano", refused_tokens=1, admitted=True, restoring_held=True)
    rows = rebound.reproject_transcript(rows, "nano")
    physical = frame(rebound, rows)
    if queued:
        current_plan, episode = physical.nano_floor_plan
        assert current_plan is rebound and episode == "cold_resume"
    else:
        assert physical.nano_floor_plan is None

    payload = {"model": "local-model", "messages": _flat(rows), "max_tokens": 20_000}
    target = {"provider": "local", "context_window_tokens": 80_000, "context_window_confirmed": True,
              "window_instance_id": "123", "local_dispatch_base_url": "http://127.0.0.1:1/v1",
              "local_input_measurement": _exact_measurement(payload, 71_809)}
    monkeypatch.setattr("ouroboros.response_limits.response_limit_for_target",
                        lambda _target: SimpleNamespace(ceiling=lambda value: value))
    with usage_accounting.bind_physical_attempt_context(physical), pytest.raises(LocalContextTooLargeError) as error:
        bound_reply_allowance(target, payload)
    remember_size_refusal(call, _refused_candidate(error.value.refused_candidate), measured=True)
    parked = cold_return.refusal_facts(slot, usage, 4)
    assert parked == ({**FACTS, "input_tokens": 71_809, "over_tokens": 1} if queued else {})

    # The size recovery rebuilt the view and sent nothing: the next round is no entry round, so it observes no
    # Nano floor again, and it still attests the same refusal for a queued root.
    call.round_idx = 5
    later = frame(rebound, rows)
    assert later.nano_floor_plan is None, "the floor is observed on the entry round alone"
    assert later.nano_floor_return is queued
    with usage_accounting.bind_physical_attempt_context(later), pytest.raises(LocalContextTooLargeError) as error:
        bound_reply_allowance(target, payload)
    remember_size_refusal(call, _refused_candidate(error.value.refused_candidate), measured=True)
    assert cold_return.refusal_facts(slot, usage, 4) == {}, "the earlier round's refusal is not this round's"
    assert cold_return.refusal_facts(slot, usage, 5) == ({**FACTS, "input_tokens": 71_809, "over_tokens": 1} if queued else {})

    def after_a_send():
        usage_accounting._LAST_PHYSICAL_ATTEMPT.set(object())
        call.round_idx = 6
        return frame(rebound, rows)

    sent = contextvars.copy_context().run(after_a_send)
    assert sent.nano_floor_plan is None and sent.nano_floor_return is False, "once something was sent it is no return"
    setattr(slot, cold_return._ARMED, None)
    call.round_idx = 7
    assert frame(rebound, rows).nano_floor_return is False, "a start that restored nothing has no such witness"


@pytest.mark.parametrize("drift", [None, "rows", "first_process", "second_formatter", "second_window"])
def test_recount_compares_the_input_and_both_serving_bindings(monkeypatch, drift):
    from ouroboros.context_budget import LocalContextTooLargeError
    from ouroboros.send_clock import SendClockPolicy, main_send_scope

    client, called = LLMClient(api_key="unused"), []
    rows = [{"role": "system", "content": "held overview"}]
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
    original = []

    def first_count(candidate, *, expected_port):
        assert expected_port == 1
        original.append(candidate)
        return _exact_measurement(candidate, 73_308)

    def count(candidate, *, expected_port):
        assert expected_port == 1 and candidate["messages"][-1] == original[0]["messages"][-1]
        assert candidate["tools"] == original[0]["tools"]
        called.append(candidate)
        measured = _exact_measurement(candidate, 73_308 if len(called) == 1 else 70_808,
                                      81_000 if len(called) == 2 and drift == "second_window" else 80_000)
        if len(called) == 1 and drift == "first_process":
            measured["process_id"] = 124
        if len(called) == 2 and drift == "second_formatter":
            measured["tokenizer_template_provenance"]["formatter"] = "another"
        return measured

    manager = SimpleNamespace(measure_prepared_input=first_count,
        serving_context_evidence=lambda **_kw: {"port": 1, "confirmed": True, "context_window": 80_000, "process_id": 123})
    monkeypatch.setattr("ouroboros.llm_local.measurement_slot", lambda *_a, **_kw: contextlib.nullcontext(True))
    monkeypatch.setattr("ouroboros.local_model.local_dispatch_port", lambda: 1)
    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda: manager)
    physical = usage_accounting.PhysicalAttemptContext(
        "owner_nano", "nano", "cold_estimate", "local", "e:round:4", None, 80_000, False, False,
        view_kind="owner_turn", response_reserve_tokens=8_192, preferred_mode="nano", nano_floor_plan=(object(), "cold_resume"))
    with usage_accounting.bind_physical_attempt_context(physical), main_send_scope(SendClockPolicy()):
        with pytest.raises(LocalContextTooLargeError) as error:
            client._prepared_local_candidate(rows, tools, 20_000, "auto", None, None, queued=True,
                                             base_url="http://127.0.0.1:1/v1")
    proof = error.value.refused_candidate["nano_reply_floor"]
    assert proof["clock_note"] and proof["sample"]["input_tokens"] == 73_308
    manager.measure_prepared_input = count
    given = [{"role": "system", "content": "another input"}] if drift == "rows" else rows
    result = client.recount_with(given, tools, lambda flat: [
        {"role": "system", "content": "current overview"}, *flat[1:]], refusal=proof)
    if drift is None:
        assert result == (73_308, 70_808) and len(called) == 2
    elif drift in {"rows", "first_process"}:
        assert result is None and len(called) == (0 if drift == "rows" else 1)
    else:
        assert result == (73_308, None) and len(called) == 2


def test_arming_is_per_start_and_reads_a_choice_only_from_the_grant_of_the_hold_it_names():
    ctx, row = SimpleNamespace(), _park(offer=OFFER)
    granted = {**row, "state": STATE_RESUME_GRANTED, "grant": {"grant_id": "g", cold_return.CHOICE_KEY: _named(row, end_attempt=True)}}
    cold_return.arm(ctx, {}, {"_pause_row": granted})
    assert getattr(ctx, cold_return._ARMED)[2] == {**_named(row, end_attempt=True), "facts": row[cold_return.FACTS_KEY]}
    stale = {**granted, "grant": {"grant_id": "g", cold_return.CHOICE_KEY: {**_named(row, end_attempt=True), "pause_generation": 2}}}
    cold_return.arm(ctx, {}, {"_pause_row": stale})
    assert getattr(ctx, cold_return._ARMED)[2] == {}, "a choice about another generation of the pause is not read"
    owner = {**granted, "rail": RAIL_OWNER_PAUSE, "reason": "owner", "cause": ""}
    cold_return.arm(ctx, {}, {"_pause_row": owner})
    assert getattr(ctx, cold_return._ARMED)[2] == {}, "nor one riding the grant of an ordinary Pause"
    cold_return.arm(ctx, {}, {})
    assert getattr(ctx, cold_return._ARMED) is None, "a fresh start on the same context is not a return"


def _granted_row(row: dict, **action) -> dict:
    return {**row, "state": STATE_RESUME_GRANTED, "grant": {"grant_id": "g", cold_return.CHOICE_KEY: _named(row, **action)}}


def _restored(usage: dict) -> dict:
    """The usage a later start restores from a saved checkpoint: what was saved, through JSON."""
    return json.loads(json.dumps(usage))


@pytest.mark.parametrize("action", [{"end_attempt": True}, {"overview_revision": OFFER["current_revision"]}])
def test_a_choice_leaves_its_grant_for_the_works_own_state_and_an_ordinary_restart_restores_it(action):
    """The grant is spent when the worker takes the saved work over. From then on the pause no longer restores:
    a process lost after that restores the work's checkpoint, and the choice is read from the usage it saved."""
    from ouroboros.transcript_prefix import ANSWER_KEY

    row = _park(offer=OFFER)
    choice = {**_named(row, **action), "facts": row[cold_return.FACTS_KEY]}
    usage = {ANSWER_KEY: {"n": 3, "route": "local-model (local)", "at": 1.0}}  # replies this work read before its park
    assert cold_return.carry_choice(usage, _granted_row(row, **action)) == choice
    assert usage[cold_return.CARRIED_KEY] == {**choice, "answers_read": 3}

    def armed(saved, continuation):
        ctx = SimpleNamespace()
        cold_return.arm(ctx, saved, continuation)
        return getattr(ctx, cold_return._ARMED)[2]

    # The pause restores (the grant is still live), or its checkpoint does (the grant was spent): the same choice.
    assert armed(_restored(usage), {"_pause_row": _granted_row(row, **action)}) == choice
    saved = _restored(usage)
    assert armed(saved, {"messages": []}) == choice and saved[cold_return.CARRIED_KEY]["answers_read"] == 3
    # The owner's next decision about this work ends the choice: the plain Resume of a later window park, or of
    # the owner's own Pause, applies nothing an earlier hold's grant carried, even where the saved usage still
    # names it. A grant with its own choice carries that one.
    plain = {**_park(pause_id="park-2", pause_generation=4), "state": STATE_RESUME_GRANTED, "grant": {"grant_id": "h"}}
    ordinary = {**plain, "rail": RAIL_OWNER_PAUSE, "reason": "owner", "cause": ""}
    for later in (plain, ordinary):
        kept = _restored(usage)
        assert armed(kept, {"_pause_row": later}) == {} and cold_return.CARRIED_KEY not in kept, later["rail"]
    # A wait that decides nothing about the window returns the work with what it carried: a technical pause that
    # comes back by itself, or one over money that its own Resume returns.
    for rail in ("control_authority_unreadable", "dispatch_refused"):
        waited = {**plain, "rail": rail, "cause": "dispatch_fence_unreadable"}
        kept = _restored(usage)
        assert armed(kept, {"_pause_row": waited}) == choice and kept[cold_return.CARRIED_KEY]["answers_read"] == 3, rail
    # The owner's later Pause over such a saved stop keeps that row's rail; the Resume that releases it names the
    # closed fence on its grant. That Resume is the owner's next decision too: the old chosen end is not applied,
    # whichever stop the Pause lay over, and a grant's own choice still carries.
    for rail, reason, cause in (("control_authority_unreadable", "authority", "dispatch_fence_unreadable"),
                                ("global_exhausted", "budget", ""), ("dispatch_refused", "budget", "")):
        saved_stop = {**plain, "rail": rail, "reason": reason, "cause": cause}
        over = {**saved_stop, "grant": {"grant_id": "h", "owner_pause_fence_id": "fence-1"}}
        kept = _restored(usage)
        assert armed(kept, {"_pause_row": over}) == {} and cold_return.CARRIED_KEY not in kept, rail
        kept = _restored(usage)  # no owner Pause over it: that grant names no fence, and the choice stays
        assert armed(kept, {"_pause_row": saved_stop}) == choice and cold_return.CARRIED_KEY in kept, rail
    again = _park(pause_id="park-2", pause_generation=4)
    over_window = {**_granted_row(again, end_attempt=True), "grant": {
        **_granted_row(again, end_attempt=True)["grant"], "owner_pause_fence_id": "fence-1"}}
    assert armed(_restored(usage), {"_pause_row": over_window}) == {
        **_named(again, end_attempt=True), "facts": again[cold_return.FACTS_KEY]}
    replaced = armed(_restored(usage), {"_pause_row": _granted_row(again, end_attempt=True)})
    assert replaced == {**_named(again, end_attempt=True), "facts": again[cold_return.FACTS_KEY]}
    # Once the work read a reply, the attempt the choice was made about is over: nothing is carried, nothing kept.
    answered = {**_restored(usage), ANSWER_KEY: {"n": 4, "route": "local-model (local)", "at": 2.0}}
    assert armed(answered, {"messages": []}) == {} and cold_return.CARRIED_KEY not in answered
    # A start that restored nothing reads no choice, whatever its usage holds.
    fresh = SimpleNamespace()
    cold_return.arm(fresh, _restored(usage), {})
    assert getattr(fresh, cold_return._ARMED) is None


def test_the_next_window_park_ends_the_choice_whether_or_not_it_could_take_it(monkeypatch):
    """The choice is made once. The park that follows its Resume reads it: it takes the chosen revision, or finds
    nothing to take (no form fits now, or the shelf moved past that revision). Either way that park is the owner's
    next decision: its plain Resume ("Try again as it is") is only the attempt as it is, and a still later park
    never takes the earlier revision by itself."""
    saved, forms = [], {"held": "rev-held", "current": "rev-current", "input_tokens": 70_808,
                        "hold": {"entry": {"topic": "overview"}, "companion": {"section_id": "shared_understanding"}}}

    _Shelf(monkeypatch)
    saving = _saving(spent := [], forms)

    def request_pause(limit_ctx, **kwargs):
        saving(limit_ctx, **kwargs)
        [(_kwargs, taking, _seed, facts)] = spent
        spent.clear()
        saved.append((_restored(limit_ctx.accumulated_usage), facts, taking.forms["hold"] if taking else None))

    monkeypatch.setattr("ouroboros.budget_pause.request_pause", request_pause)
    monkeypatch.setattr(cold_return, "refusal_facts", lambda *_a: dict(FACTS))
    first = _park(offer=OFFER)

    def returned(usage, continuation, offered):
        """One return: arm as the loop does, then the park its refusal takes; what that park saved."""
        saved.clear()
        monkeypatch.setattr(cold_return, "_overview_forms", lambda *_a: offered)
        ctx = SimpleNamespace()
        cold_return.arm(ctx, usage, continuation)
        cold_return.park_refused(SimpleNamespace(tools=SimpleNamespace(_ctx=ctx), accumulated_usage=usage,
                                                 round_idx=1, messages=[]))
        [(kept, facts, hold)] = saved
        return kept, facts.get(cold_return.OVERVIEW_KEY), hold

    chosen = _granted_row(first, overview_revision=OFFER["current_revision"])
    second = {**_park(pause_id="park-2", pause_generation=4), "state": STATE_RESUME_GRANTED, "grant": {"grant_id": "h"}}
    for offered, shown in ((None, None), ({**forms, "current": "rev-newer"}, {**OFFER, "current_revision": "rev-newer"})):
        # The chosen revision could not be taken: the park shows what it measured and holds the earlier overview.
        kept, offer, hold = returned({}, {"_pause_row": chosen}, offered)
        assert (offer, hold) == (shown, None)
        # The new hold's plain Resume restores what that park saved. The chosen revision is current again and
        # fits, and nothing takes it: the owner did not choose it for this hold.
        after, offer, hold = returned(kept, {"_pause_row": second}, forms)
        assert (offer, hold) == (OFFER, None), "offered again, never taken late"
        assert cold_return.CARRIED_KEY not in after, "and nothing of it is saved on"
    # Taken: the park saves the work holding it; its own plain Resume carries no choice either.
    kept, offer, hold = returned({}, {"_pause_row": chosen}, forms)
    assert (offer, hold) == ({**OFFER, "taken": True}, forms["hold"])
    after, offer, hold = returned(kept, {"_pause_row": second}, {**forms, "current": "rev-newer"})
    assert hold is None and "taken" not in offer and cold_return.CARRIED_KEY not in after
    # Before that park, an ordinary restart keeps the choice: the checkpoint's usage restores it and its park takes it.
    carried = {}
    cold_return.carry_choice(carried, chosen)
    _kept, offer, hold = returned(_restored(carried), {"messages": []}, forms)
    assert (offer, hold) == ({**OFFER, "taken": True}, forms["hold"])


def _chosen_end():
    """A return whose grant carried the owner's chosen end, armed as the loop arms it."""
    row = _park()
    ctx, usage = SimpleNamespace(owner_wait_callback=lambda *_a: None, task_id=""), {}
    cold_return.arm(ctx, usage, {"_pause_row": _granted_row(row, end_attempt=True)})
    limit = SimpleNamespace(tools=SimpleNamespace(_ctx=ctx), accumulated_usage=usage, round_idx=1)
    return row, ctx, usage, limit


def _still_chosen(ctx, usage, limit):
    return cold_return._choice(ctx, usage).get("end_attempt") is True and cold_return.CARRIED_KEY in usage


def test_the_owners_own_pause_ends_a_chosen_end_and_its_resume_is_a_plain_return(monkeypatch):
    """The owner chose to end the attempt; before the loop reached that end the owner paused the tree (the ordered
    controls decide first, and the Pause is never answered with the end). That Pause is the owner's next decision:
    its Resume returns the work as any Pause's does."""
    from ouroboros import budget_pause

    row, ctx, usage, limit = _chosen_end()
    assert _still_chosen(ctx, usage, limit)
    # Warm: the Pause kept this very stack. ``enter_owner_pause`` names the closed fence it waits under.
    ctx._owner_pause_fence_id = "fence-of-this-pause"
    assert cold_return.end_chosen(limit) is None, "the loop asks as a plain return after the owner's Resume"
    assert cold_return.CARRIED_KEY not in usage and getattr(ctx, cold_return._ARMED)[2] == {}, "nor is it saved on"
    # Cold: the Pause parked the work; what restores is its own row, whose plain grant carries no choice.
    _row, _ctx, usage, _limit = _chosen_end()
    paused = {"pause_id": "owner-1", "pause_generation": 4, "state": STATE_RESUME_GRANTED, "rail": RAIL_OWNER_PAUSE,
              "reason": "owner", "grant": {"grant_id": "h"}}
    cold = SimpleNamespace()
    cold_return.arm(cold, _restored(usage), {"_pause_row": paused})
    assert getattr(cold, cold_return._ARMED)[2] == {}
    # A stack that had waited under an earlier Pause before it armed is not paused again by that old fence.
    earlier = SimpleNamespace(_owner_pause_fence_id="an-earlier-fence")
    kept = {}
    cold_return.arm(earlier, kept, {"_pause_row": _granted_row(row, end_attempt=True)})
    assert cold_return._choice(earlier, kept).get("end_attempt") is True
    # The real entry with an open fence changes nothing.
    row, ctx, usage, limit = _chosen_end()
    monkeypatch.setattr("ouroboros.owner_pause.wait_member_fence", lambda _ctx: {})
    assert budget_pause.enter_owner_pause(limit) is False and _still_chosen(ctx, usage, limit)


def test_a_hold_that_did_not_take_place_or_a_technical_wait_keeps_the_chosen_end(monkeypatch):
    """Only the owner's next decision ends the choice. A pause that was requested and could not be saved, a fence
    that could not be read and a technical wait that returns by itself decide nothing about the window: the work
    still carries the chosen end, so none of them lets it send an ordinary request instead."""
    from ouroboros import budget_pause
    from ouroboros.model_wait import ModelWaitInterrupted

    # A pause request that cannot be saved (no task id): nothing was held.
    row, ctx, usage, limit = _chosen_end()
    assert budget_pause.request_pause(limit, rail=RAIL_OWNER_PAUSE, scope="root", reason_text="This task tree was paused.") is None
    assert usage["exact_pause_unavailable"] == "no_task_id" and _still_chosen(ctx, usage, limit)
    # A fence whose state could not be read: the wait is interrupted, no Pause is known to have held this work.
    monkeypatch.setattr("ouroboros.owner_pause.wait_member_fence", lambda _ctx: {"state": "unknown"})
    with pytest.raises(ModelWaitInterrupted):
        budget_pause.enter_owner_pause(limit)
    assert _still_chosen(ctx, usage, limit)
    # The technical pause that followed returns by itself: what it saved is restored with the choice in it.
    technical = {"pause_id": "tech-1", "pause_generation": 4, "state": STATE_RESUME_GRANTED, "reason": "authority",
                 "rail": "control_authority_unreadable", "cause": "owner_pause_authority_unreadable",
                 "auto_resume": True, "grant": {"grant_id": "auto"}}
    back = SimpleNamespace()
    restored = _restored(usage)
    cold_return.arm(back, restored, {"_pause_row": technical})
    assert getattr(back, cold_return._ARMED)[2] == {**_named(row, end_attempt=True), "facts": row[cold_return.FACTS_KEY]}
    assert cold_return._choice(back, restored).get("end_attempt") is True
    # A pause row that could not be read at all restores from the checkpoint: the same.
    unread = SimpleNamespace()
    cold_return.arm(unread, _restored(usage), {"messages": [], "_pause_row": None})
    assert getattr(unread, cold_return._ARMED)[2]["end_attempt"] is True


@pytest.mark.parametrize("held", [
    True, "end", {"end_attempt": True},
    {"pause_id": "park-1", "pause_generation": 3, "end_attempt": True, "overview_revision": "rev", "facts": {}, "answers_read": 0},
    {"pause_id": "park-1", "pause_generation": 3, "end_attempt": False, "facts": {}, "answers_read": 0},
    {"pause_id": "park-1", "pause_generation": 3, "end_attempt": True, "facts": {}},
    {"pause_id": "park-1", "pause_generation": 3, "end_attempt": True, "facts": {}, "answers_read": "0"},
])
def test_a_carried_choice_is_read_whole_or_not_at_all(held):
    usage = {cold_return.CARRIED_KEY: held}
    assert cold_return._carried(usage) == {} and usage == {}, "what is not one whole choice is dropped, never guessed"


def test_the_choice_is_in_the_handover_checkpoint_before_the_grant_is_spent(tmp_path, monkeypatch):
    """The loop takes a restoring pause over (``loop._resume_continuation``), and ``resume_paused_loop`` saves the
    restored work as this attempt's checkpoint before it consumes the grant. The choice must already be in that
    checkpoint's usage: a process lost between the two, or after the consumption, restores from it."""
    from ouroboros import budget_pause

    class ProcessLost(BaseException):
        """The process ends at this exact boundary; no handler runs."""

    row = _park(offer=OFFER)
    granted = _granted_row(row, overview_revision=OFFER["current_revision"])
    handed = {}

    def restore(_tools, state, _messages, _trace, usage, _seen):
        usage.update(state["usage"])

    def save_resumed(_tools, _state, _messages, _trace, usage, _seen):
        handed.update(_restored(usage))
        raise ProcessLost

    monkeypatch.setattr("ouroboros.presence_carrier.reenter", lambda _ctx: None)
    monkeypatch.setattr("ouroboros.owner_wait.restore_continuation_state", restore)
    monkeypatch.setattr("ouroboros.task_pacing.restore_cost_ceiling", lambda *_a, **_kw: SimpleNamespace(ceiling_usd=None))
    monkeypatch.setattr(budget_pause, "_resume_disclosure", lambda *_a, **_kw: None)
    monkeypatch.setattr("ouroboros.working_checkpoint.save_resumed", save_resumed)
    monkeypatch.setattr(budget_pause, "set_budget_pause", lambda *_a, **_kw: pytest.fail("the grant was spent first"))
    ctx = SimpleNamespace(budget_drive_root=str(tmp_path), drive_root=tmp_path, task_id="returned", task_attempt=1,
                          model_wait_context=None, budget_pause_resume=None)
    state = {"usage": {"earlier": "kept"}, "round_idx": 4, "tool_schemas": [], "_pause_row": granted}
    with pytest.raises(ProcessLost):
        loop._resume_continuation(SimpleNamespace(_ctx=ctx), ({}, state, {}), [], {}, {}, set(),
                                  ("model", "medium", False, "nano", 0, None), None, None)
    assert handed["earlier"] == "kept"
    assert handed[cold_return.CARRIED_KEY] == {**_named(row, overview_revision=OFFER["current_revision"]),
                                               "facts": row[cold_return.FACTS_KEY], "answers_read": 0}
    # The next start restores that checkpoint, with no pause row: the park that follows takes the chosen revision.
    restored = SimpleNamespace()
    cold_return.arm(restored, handed, {"messages": []})
    assert getattr(restored, cold_return._ARMED)[2]["overview_revision"] == OFFER["current_revision"]


@pytest.mark.parametrize("body, read", [
    ({}, (None, "")),
    ({"accept_unknown": []}, (None, "")),
    ({"context_choice": {"pause_id": "p", "pause_generation": 3, "end_attempt": True}}, "choice"),
    ({"context_choice": {"pause_id": "p", "pause_generation": 3, "overview_revision": "rev"}}, "choice"),
    ({"context_choice": {"pause_id": "p", "pause_generation": 3, "end_attempt": False}}, "refused"),
    ({"context_choice": {"pause_id": "p", "pause_generation": 3}}, "refused"),
    ({"context_choice": {"pause_id": "p", "pause_generation": 3, "end_attempt": True, "overview_revision": "rev"}}, "refused"),
    ({"context_choice": {"pause_id": "p", "pause_generation": "3", "end_attempt": True}}, "refused"),
    ({"context_choice": {"pause_id": "p", "pause_generation": True, "end_attempt": True}}, "refused"),
    ({"context_choice": {"pause_id": "", "pause_generation": 3, "end_attempt": True}}, "refused"),
    ({"context_choice": {"pause_generation": 3, "end_attempt": True}}, "refused"),
    ({"context_choice": {"end_attempt": True}}, "refused"),
    ({"context_choice": {"pause_id": "p", "pause_generation": 3, "overview_revision": ""}}, "refused"),
    ({"context_choice": {"pause_id": "p", "pause_generation": 3, "end_attempt": True, "extra": 1}}, "refused"),
    ({"context_choice": True}, "refused"),
    ({"context_choice": None}, "refused"),
])
def test_the_door_reads_a_choice_whole_or_refuses_it(body, read):
    choice, error = cold_return.door_choice(body)
    if read == "choice":
        assert choice == body["context_choice"] and choice is not body["context_choice"] and error == ""
    elif read == "refused":
        assert choice is None and "exactly one of" in error
    else:
        assert (choice, error) == read


def test_a_choice_is_offered_only_by_the_waiting_hold_it_names(tmp_path):
    """The grant checks the live stop identity, then the pause and the offered action."""
    def at_grant(row, choice, identity="shown"):
        with cold_return.choosing(choice):
            return cold_return.granted_choice(row, identity if row.get("state") == STATE_PAUSED else "")

    park = _park(offer=OFFER)
    def named(**action):
        return {**_named(park, **action), "stop_identity": "shown"}
    end, take = named(end_attempt=True), named(overview_revision=OFFER["current_revision"])
    for choice in (end, take, named(try_again=True)):
        assert at_grant(park, choice) == ({cold_return.CHOICE_KEY: choice}, None)
        assert at_grant(park, choice, "new") == ({}, cold_return._NOT_OFFERED)
        assert at_grant(park, choice, "") == ({}, cold_return._NOT_OFFERED)
    refused = dict(cold_return._NOT_OFFERED)
    assert refused["error"] == "context_choice_not_offered"
    cases = {
        "another pause id": (park, {**end, "pause_id": "park-0"}),
        "an earlier generation of the same id": (park, {**end, "pause_generation": 2}),
        "an ordinary owner Pause": ({**park, "rail": RAIL_OWNER_PAUSE, "reason": "owner", "cause": ""}, end),
        "another technical cause": ({**park, "cause": "dispatch_fence_unreadable"}, end),
        "a park that recorded no numbers": ({key: value for key, value in park.items() if key != cold_return.FACTS_KEY}, end),
        "a revision this park did not count": (park, {**take, "overview_revision": "rev-other"}),
        "the held revision itself": (park, {**take, "overview_revision": OFFER["held_revision"]}),
        "a park that offers no overview": (_park(), take),
        "a park that already took it": (_park(offer={**OFFER, "taken": True}), take),
    }
    for why, (row, choice) in cases.items():
        assert at_grant(row, choice) == ({}, refused), why
    assert at_grant({**park, "state": STATE_RESUME_GRANTED}, end) == ({}, refused)
    assert at_grant(park, None) == ({}, None), "plain Resume carries no choice"


def test_a_clients_view_of_the_hold_and_the_owners_notice_carry_its_numbers():
    plain, offered = _park(), _park(offer=OFFER)
    taken = _park(offer={**OFFER, "taken": True})
    assert cold_return.window_facts(plain) == {
        "pause_id": "park-1", "pause_generation": 3, "input_tokens": 73_308, "window_tokens": 80_000,
        "reply_floor_tokens": 8_192, "over_tokens": 1_500}
    assert cold_return.window_facts(offered)[cold_return.OVERVIEW_KEY] == OFFER
    assert cold_return.window_facts({**plain, "state": STATE_RESUME_GRANTED}) == {}, "a hold being resumed offers nothing"
    assert cold_return.window_facts({**plain, "rail": RAIL_OWNER_PAUSE}) == {} and cold_return.window_facts(None) == {}
    said = ("At the last check, the request needed 81,500 tokens with room for a reply, "
            "1,500 more than the model's 80,000-token window.")
    step = "Its work is saved: make room in always-loaded memory or raise the window, then press Resume."
    notice = cold_return.park_notice
    assert notice(plain[cold_return.FACTS_KEY]) == f"{said} {step}"
    assert notice(offered[cold_return.FACTS_KEY]) == (
        f"{said} With the current overview, it fit at 79,000 tokens with room for a reply. "
        "The earlier overview is still in use. Press Resume to see your options.")
    assert notice(taken[cold_return.FACTS_KEY]) == (
        "As you chose, the current overview is in place. At the last check, the request with the earlier overview "
        "needed 81,500 tokens with room for a reply, 1,500 more than the model's 80,000-token window. With the chosen "
        "overview, it fit at 79,000 tokens with room for a reply. The earlier overview stays readable. Press Resume.")
    assert notice(None) == notice({"input_tokens": "73308"}) == f"The task's saved work did not fit the model's window. {step}"


def test_the_card_uses_the_parks_measured_overview_even_with_a_queue_marker():
    from ouroboros.budget_pause import exact_pause_marker
    from ouroboros.work_stop import activity_stop_view, stop_presentation

    cases = (
        (_park(), cold_return.WAIT_TEXT),
        (_park(offer=OFFER), cold_return.OFFER_TEXT),
        (_park(offer={**OFFER, "taken": True}), cold_return.READY_TEXT),
    )
    for row, words in cases:
        task = {"id": "held", "_budget_pause": exact_pause_marker(row), "budget_pause": row}
        view = stop_presentation(task)
        assert view["reason"]["text"] == words and view["resumable"]
        activity = {"activity_id": "held", "phase": "budget_paused", "stop": view["stop"], "pause_reason": words}
        assert activity_stop_view(activity, {"_budget_pause": task["_budget_pause"]})["reason"]["text"] == words


def test_each_parks_notice_keeps_the_numbers_of_its_own_episode(tmp_path):
    """One notice per park, each with what that park recorded: a later park of the same task never rewords it."""
    from ouroboros import pause_notices
    from ouroboros.task_results import load_task_result

    task_id = "noticed-park"
    write_task_result(tmp_path, task_id, STATUS_RUNNING)
    first, second = _park(offer=OFFER), _park(pause_id="park-2", pause_generation=4, offer={**OFFER, "taken": True})
    set_budget_pause(tmp_path, task_id, first)
    set_budget_pause(tmp_path, task_id, second)
    owed = load_task_result(tmp_path, task_id)["pause_notices"]
    assert sorted(owed) == [f"pause:{task_id}:park-1", f"pause:{task_id}:park-2"]
    said = [pause_notices.notice_text(owed[identity]) for identity in sorted(owed)]
    assert said == [cold_return.park_notice(row[cold_return.FACTS_KEY]) for row in (first, second)]
    assert "As you chose" in said[1] and "As you chose" not in said[0] and "79,000" in said[0]
    assert pause_notices.authority_wait_text(cold_return.CAUSE) == cold_return.WAIT_TEXT
    assert "automatically" in pause_notices.authority_wait_text("dispatch_fence_unreadable"), "other technical pauses return by themselves"


def test_main_offers_resume_for_a_window_park_and_stays_quiet_for_a_pause_that_returns_by_itself():
    """Main's attention rule (``main_decisions.owner_actions``) over the census row of each technical pause: only
    the window park waits for the owner's Resume; an unreadable authority returns by itself and offers none."""
    from ouroboros.budget_pause import exact_pause_marker
    from ouroboros.gateway.main_decisions import owner_actions
    from ouroboros.work_stop import current_stop

    def census(row):
        stop = current_stop({"id": "held", "_budget_pause": exact_pause_marker(row)}, {})
        return {"activity_id": "held", "phase": "budget_paused", "live": False, "can_resume": True, "task_attempt": 1,
                "pause_cause": "authority", "stop": stop, "refusal": {}}

    parked = census(_park())
    assert parked["stop"]["rail"] == cold_return.RAIL_CONTEXT_WINDOW and parked["stop"]["release"] == "resume"
    assert owner_actions(parked) == ({}, True)
    unreadable = census({**_park(), "rail": "control_authority_unreadable", "cause": "dispatch_fence_unreadable",
                         "auto_resume": True, "resume_policy": "authority_readable"})
    assert owner_actions(unreadable) == ({}, False)
    assert owner_actions({**parked, "can_resume": False}) == ({}, False), "the census's own release rule still decides"
    assert owner_actions({**parked, "live": True}) == ({}, False)


# ---------------------------------------------------------------------------------------------------------------
# The owner's chosen end of an attempt, in the loop.
# ---------------------------------------------------------------------------------------------------------------

def _run(tmp_path, monkeypatch, *, armed, early=None):
    """The real loop over one scripted reply, started as a return whose grant carried ``armed``."""
    asked = []

    def respond(_llm, messages, *_a, **_kw):
        asked.append(json.dumps(messages))
        return {"role": "assistant", "content": "The model's own answer."}, 0.0

    def arm(tool_ctx, usage, _continuation):
        setattr(tool_ctx, cold_return._ARMED, (usage_accounting.last_physical_attempt_capture(), None, dict(armed)))

    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    monkeypatch.setattr(cold_return, "arm", arm)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    if early is not None:
        monkeypatch.setattr(loop, "_maybe_early_finalize", early)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_metadata = {"budget_drive_root": str(tmp_path), "root_task_id": "returned"}
    text, usage, trace = loop.run_llm_loop(
        [{"role": "user", "content": "Continue the saved work."}], registry,
        SimpleNamespace(default_model=lambda: "test-model"), tmp_path / "logs", lambda *_a, **_kw: None,
        queue.Queue(), task_id="returned", drive_root=tmp_path)
    return text, usage, trace, asked


def test_the_chosen_end_of_an_attempt_is_the_size_terminal_with_nothing_sent(tmp_path, monkeypatch):
    chosen = {"pause_id": "park-1", "pause_generation": 3, "end_attempt": True, "facts": dict(FACTS)}
    text, usage, _trace, asked = _run(tmp_path, monkeypatch, armed=chosen)
    assert asked == [], "nothing is sent"
    assert (usage["execution_status"], usage["reason_code"]) == ("infra_failed", "context_overflow")
    assert usage["_last_llm_error_kind"] == "context_overflow" and "The model's own answer." not in text
    refused = usage[SIZE_REFUSAL_KEY]
    assert (refused["tokens"], refused["local_window"], refused["basis"]) == (73_308, 80_000, "exact_local_count")
    from ouroboros.owner_continue import TECHNICAL_REASON_CODES

    assert usage["reason_code"] in TECHNICAL_REASON_CODES, "the ordinary Continue reads the cause it always read"


@pytest.mark.parametrize("armed", [{}, {"pause_id": "park-1", "pause_generation": 3, "overview_revision": "rev-current",
                                         "facts": dict(FACTS)}])
def test_a_return_without_that_choice_asks_the_model_as_before(tmp_path, monkeypatch, armed):
    text, usage, _trace, asked = _run(tmp_path, monkeypatch, armed=armed)
    assert len(asked) == 1 and text == "The model's own answer." and usage.get("reason_code") != "context_overflow"


def _run_restored(tmp_path, monkeypatch, carried):
    """The real loop started as a checkpoint restore whose saved usage carries ``carried``: the real ``arm`` reads
    it, with no pause row and no grant left to read it from."""
    asked = []

    def respond(_llm, messages, *_a, **_kw):
        asked.append(json.dumps(messages))
        return {"role": "assistant", "content": "The model's own answer."}, 0.0

    def arm(tool_ctx, usage, _continuation):
        usage.update(_restored(carried))
        _ARM(tool_ctx, usage, {"messages": []})

    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    monkeypatch.setattr(cold_return, "arm", arm)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_metadata = {"budget_drive_root": str(tmp_path), "root_task_id": "returned"}
    text, usage, _trace = loop.run_llm_loop(
        [{"role": "user", "content": "Continue the saved work."}], registry,
        SimpleNamespace(default_model=lambda: "test-model"), tmp_path / "logs", lambda *_a, **_kw: None,
        queue.Queue(), task_id="returned", drive_root=tmp_path)
    return text, usage, asked


def test_a_restart_after_the_grant_was_spent_still_ends_the_attempt_the_owner_ended(tmp_path, monkeypatch):
    """The ordinary restart: the choice's grant is spent, the process is lost, and the next start restores the
    work's checkpoint. It ends on the same size terminal with nothing sent, never as a plain return that asks."""
    row = _park()
    usage = {}
    cold_return.carry_choice(usage, _granted_row(row, end_attempt=True))
    text, ended, asked = _run_restored(tmp_path, monkeypatch, usage)
    assert asked == [], "nothing is sent"
    assert (ended["execution_status"], ended["reason_code"]) == ("infra_failed", "context_overflow")
    refused = ended[SIZE_REFUSAL_KEY]
    assert (refused["tokens"], refused["local_window"]) == (FACTS["input_tokens"], FACTS["window_tokens"])
    assert "The model's own answer." not in text


def test_a_restored_overview_choice_or_a_lapsed_one_asks_the_model_as_a_plain_return(tmp_path, monkeypatch):
    row = _park(offer=OFFER)
    usage = {}
    cold_return.carry_choice(usage, _granted_row(row, overview_revision=OFFER["current_revision"]))
    text, _usage, asked = _run_restored(tmp_path, monkeypatch, usage)
    assert len(asked) == 1 and text == "The model's own answer.", "the held overview is tried first, as it is"
    ended = {}
    cold_return.carry_choice(ended, _granted_row(row, end_attempt=True))
    # Carried at another count of replies than this work has read: the attempt it was made about is over.
    ended[cold_return.CARRIED_KEY]["answers_read"] = 7
    text, after, asked = _run_restored(tmp_path / "lapsed", monkeypatch, ended)
    assert len(asked) == 1 and text == "The model's own answer." and after.get("reason_code") != "context_overflow"
    assert cold_return.CARRIED_KEY not in after


def test_stop_pause_time_and_money_decide_before_the_chosen_end(tmp_path, monkeypatch):
    """The ordered pre-round exit (``loop_round_limits._maybe_early_finalize``: a Stop, the owner's Pause, the deadline,
    the cost rail) is read first, as for any round: what it decides is the result, never the size terminal."""
    reads = []

    def early(limit_ctx, _tools, _controls, **_kwargs):
        reads.append(limit_ctx.round_idx)
        return "Ended by its rail.", dict(limit_ctx.accumulated_usage, reason_code="finalization_grace"), limit_ctx.llm_trace

    chosen = {"pause_id": "park-1", "pause_generation": 3, "end_attempt": True, "facts": dict(FACTS)}
    text, usage, _trace, asked = _run(tmp_path, monkeypatch, armed=chosen, early=early)
    assert (text, reads, asked) == ("Ended by its rail.", [1], [])
    assert usage["reason_code"] == "finalization_grace" and SIZE_REFUSAL_KEY not in usage


def test_a_returned_round_excludes_a_direct_chat_in_either_service_representation(monkeypatch):
    """``unsent_return`` witnesses an ordinary queued root, not a direct chat: the slot flag and the task metadata
    each mark a direct chat, and either one alone excludes the round, as the entry witness reads both."""
    sentinel = object()
    monkeypatch.setattr(usage_accounting, "last_physical_attempt_capture", lambda: sentinel)
    plan = SimpleNamespace(preferred_mode="nano", model_role="main", view_kind="owner_turn", core={"x": 1})

    def call(slot):
        return SimpleNamespace(tools=SimpleNamespace(_ctx=slot), active_use_local=True, task_type="task")

    ordinary = SimpleNamespace(**{cold_return._ARMED: (sentinel, None, None)}, task_metadata={})
    assert cold_return.unsent_return(call(ordinary), plan) is True
    flagged = SimpleNamespace(**{cold_return._ARMED: (sentinel, None, None)}, is_direct_chat=True, task_metadata={})
    assert cold_return.unsent_return(call(flagged), plan) is False
    marked = SimpleNamespace(**{cold_return._ARMED: (sentinel, None, None)}, task_metadata={"_is_direct_chat": True})
    assert cold_return.unsent_return(call(marked), plan) is False


def test_current_window_offer_uses_the_painted_stop_identity_and_needs_readable_tree(monkeypatch):
    from ouroboros import work_stop

    task = {"id": "window-park-root", "_attempt": 1}
    park = _park(offer=OFFER)
    basis = {"owner": {"generation": 2}}
    stop = {"rail": cold_return.RAIL_CONTEXT_WINDOW, "pause_id": park["pause_id"]}
    control = {"phase": "budget_paused", "view": {"stop": stop}}
    monkeypatch.setattr(work_stop, "queue_control", lambda *_a, **_kw: control)
    monkeypatch.setattr(work_stop, "_tree_basis", lambda *_a: basis)
    monkeypatch.setattr(work_stop, "_pause_row", lambda *_a: park)
    identity = work_stop.current_window_offer_id(None, task, {})
    assert identity == work_stop._stop_identity(task, stop, basis)
    basis["owner"]["generation"] += 1
    assert work_stop.current_window_offer_id(None, task, {}) != identity
    control["view"]["stop"] = {"cause": "pause", "member_stop": stop}
    assert work_stop.current_window_offer_id(None, task, {}) == ""
    control["view"]["stop"] = stop
    monkeypatch.setattr(work_stop, "_tree_basis", lambda *_a: None)
    assert work_stop.current_window_offer_id(None, task, {}) == ""


def test_clean_window_retries_do_not_grow_their_transcript_or_edit_prior_words():
    from ouroboros.budget_pause import _resume_disclosure

    prior = {"role": "user", "content": "[SYSTEM NOTICE]\nKeep these earlier words exactly."}
    messages = [prior]
    ctx = SimpleNamespace(_budget_paused_sec=10)
    row = {**_park(), "reason": "authority", "external_runs": {"runs": []}}
    for seconds in (10, 20, 30):
        ctx._budget_paused_sec = seconds
        _resume_disclosure(ctx, row, {}, {"consumed_at": 100}, {}, messages)
    assert len(messages) == 1 and messages[0] is prior
    assert prior["content"] == "[SYSTEM NOTICE]\nKeep these earlier words exactly."

    # A genuinely interrupted model request is still disclosed; earlier words stand.
    row["resume_point"] = {"abandoned_model_attempts": ["attempt-1"]}
    _resume_disclosure(ctx, row, {}, {"consumed_at": 100}, {"reason": "not_a_money_rail"}, messages)
    assert messages[0] is prior and "attempt-1" in messages[1]["content"]
