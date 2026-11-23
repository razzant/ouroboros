"""The seed note for my own time: owed by a fresh install's first accepted Start, written once and held in every own-clock
wake, never written on an install that predates it, which hears its text once in a direct Main turn of my human.

One isolated data root carries the real state store (mint, adoption, locked writes), the real alarm's Start and boot,
the real note writer, the real request builder (``test_held_notes``'s world) and the real loop over a stub transport
that records each physical candidate as the send path does; no model, server or pool runs."""
from __future__ import annotations

import itertools
import json
import os
import queue
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store as ds
from ouroboros import knowledge, knowledge_versions
from ouroboros import own_time_seed as seed
from ouroboros.consciousness import AGENT_STOP, OWNER_STOP, BackgroundConsciousness, _persist_start
from tests.test_consciousness import T0
from tests.test_consciousness_wake import _owner
from tests.test_held_notes import RESERVE, TASK, WAKE_TURN, _held, _plan
from tests.test_held_notes import world as world  # noqa: F401 -- the shared world
from tests.test_state_authority import _prior

MAIN = {**TASK, "id": "main-turn", "_is_direct_chat": True, "origin_message_ref": {"chat_id": 1, "client_message_id": "w1"}}
PHONE_ROOM = 9_000_001  # an owner's bound transport chat: its direct turns are Main's


@pytest.fixture
def install(world, monkeypatch):  # noqa: F811 -- the imported fixture is this one's input
    from supervisor import evolution_lifecycle as lifecycle
    from supervisor import queue as task_queue
    from supervisor import state, workers
    from supervisor.active_activity import get_direct_activity_registry

    root = world[0].drive_root
    prior_state, prior_queue = state.DRIVE_ROOT, task_queue.DRIVE_ROOT
    state.init(root)
    task_queue.init(root)
    monkeypatch.setattr(state, "check_openrouter_ground_truth", lambda: None)
    monkeypatch.setitem(lifecycle._STOP_LATCH, "stopped", False)
    for module in (workers, task_queue):
        monkeypatch.setattr(module, "RUNNING", {})
        monkeypatch.setattr(module, "PENDING", [])
    get_direct_activity_registry().clear()
    address = knowledge.resolve_knowledge_address(root, seed.TOPIC, "global")
    yield SimpleNamespace(root=root, state=state, address=address, world=world,
                          clock=lambda: BackgroundConsciousness(root, root, lambda: 7, now=T0),
                          stored=lambda: state.read_state(root).values.get(seed.STATE_KEY, "absent"),
                          on=lambda: state.read_state(root).values.get("bg_consciousness_enabled"))
    get_direct_activity_registry().clear()
    state.init(prior_state)
    task_queue.init(prior_queue)


def _legacy(install):
    """A state an earlier release wrote, adopted by the real boot: an install that predates the seed."""
    path = install.root / "state" / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_prior(bg_consciousness_enabled=False)), encoding="utf-8")
    assert install.state.init_state().quality == "current"


def _boot(install):
    """A supervisor start in the server's order (``server._run_supervisor``): the state's mint or adoption, then the
    alarm's construction, which settles a debt whatever the switch says and starts nothing."""
    install.state.init_state()
    return install.clock()


def _system_text(plan) -> str:
    return "".join(block["text"] for block in plan.messages_for(plan.initial_mode)[0]["content"])


def _rows(install, *, commits=False):
    return [row for _offset, row in knowledge_versions.NoteHistory(install.address, commits=commits)]


def _held_by_wake(install):
    return knowledge.active_claims(install.root).get("wake", [])


def _settled(install):
    """The seed's event rows, oldest first: where its outcome and a failure's reason are kept."""
    path = install.root / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [(row["outcome"], row.get("error", "")) for row in rows if row.get("type") == "own_time_seed"]


def _plain(reply):
    """The Start's reply reaches my human's chat: no path, code or seed jargon, whatever the seed's outcome."""
    assert not any(word in reply for word in ("seed", "my-own-time", "memory/", "Error", "publication")), reply
    return reply


def test_a_fresh_installs_first_start_writes_one_note_every_wake_holds_and_an_older_install_writes_none(install):
    """The mix-up this guards: the same Start on a minted root and on an adopted one. Only the minted one owes and
    writes, and only its wake request (built by the real context builder) holds the note."""
    minted = install.state.init_state().values  # the real mint, then the boot's settle: nothing is owed yet
    assert minted[seed.STATE_KEY] == seed.ELIGIBLE and not install.address.path.exists()
    reply = install.clock().start()
    assert reply.startswith("Consciousness on:") and "seed" not in reply
    assert install.address.path.read_bytes() == seed.SEED_NOTE.encode("utf-8")
    assert _held_by_wake(install) == [install.address] and install.stored() == seed.WRITTEN
    (row,) = _rows(install)  # the honest source: the host's factory text, stamped with this initialization
    assert (row["writer"], row["focus"], row["writer_input_ref"]["kind"]) == ("host_seed", {"role": "host"}, "factory_text")
    assert row["writer_input_ref"]["initialization_id"] == minted["initialization_id"]
    wake = _held(_plan(install.world, task=WAKE_TURN))
    assert "### My own time" in wake and "Matched claims: 'wake'" in wake and "choosing nothing is a choice." in wake
    main = _plan(install.world)
    assert "choosing nothing is a choice." not in json.dumps(main.messages_for(main.initial_mode))
    ds.append(install.root, "chat", _owner("Hello.", T0, cmid="w1"))
    assert seed.OFFER_LINE not in _system_text(_plan(install.world, task=MAIN)), "a fresh install hears no offer"

    # The other direction: an adopted state of an earlier release. Start owes nothing and writes nothing.
    install.address.path.unlink()
    for name in ("state.json", "state.last_good.json", "state.initialized.json"):
        (install.root / "state" / name).unlink(missing_ok=True)
    _legacy(install)
    assert install.clock().start().startswith("Consciousness on:")
    assert not install.address.path.exists() and install.stored() == "absent"
    assert "Matched claims: 'wake'" not in _held(_plan(install.world, task=WAKE_TURN))


def test_a_repeated_start_a_boot_an_edit_a_removed_claim_and_an_archive_are_never_undone(install):
    install.state.init_state()
    install.clock().start()
    rows = _rows(install, commits=True)
    assert install.clock().start().startswith("Consciousness is already on.")
    _boot(install)
    assert _rows(install, commits=True) == rows
    live = knowledge.read_knowledge_note(install.address)
    unbound = knowledge.write_knowledge_note(install.address, "---\nclaims: []\n---\n", "edit",
                                             expected_revision=live.revision, writer="turn")
    assert unbound.ok and not _held_by_wake(install)
    install.clock().start()
    _boot(install)
    assert not _held_by_wake(install), "the removed claim stays removed"
    assert [row["writer"] for row in _rows(install)] == ["turn", "host_seed"]  # the history keeps both, newest first
    archived = knowledge.write_knowledge_note(install.address, "", "archive", expected_revision=unbound.current.revision,
                                              writer="turn", reason="mine to retire")
    assert archived.ok
    install.address.path.unlink()  # even a deletion stands
    install.clock().start()
    _boot(install)
    assert not install.address.path.exists() and install.stored() == seed.WRITTEN


def test_a_setup_that_left_the_switch_on_owes_the_seed_and_the_first_boot_writes_it(install):
    from ouroboros.settings_setup_contract import ONBOARDING_CONSCIOUSNESS_KEY
    from supervisor.state_initialization import first_state

    (install.root / "settings.json").write_text(json.dumps({ONBOARDING_CONSCIOUSNESS_KEY: "on"}), encoding="utf-8")
    assert first_state(install.root, "probe")[seed.STATE_KEY] == seed.OWED  # born on: its setup accepted the Start
    clock = _boot(install)  # the mint, then the alarm's construction settles the debt before any wake
    assert install.on() is True and install.stored() == seed.WRITTEN
    assert install.address.path.read_bytes() == seed.SEED_NOTE.encode("utf-8") and _held_by_wake(install) == [install.address]
    assert clock.start() == "Consciousness is already on."  # the server's restore of the saved On adds nothing


def test_an_accepted_start_cut_before_its_write_leaves_a_debt_a_later_boot_settles_while_off(install, monkeypatch):
    """A refused Start, a pending Panic and a switch left off owe nothing, and a boot writes nothing for them. The
    locked write that accepts a Start owes the seed; cut before the write, or failing in it, the debt survives an Off,
    and the next boot writes the one note and its claim without turning the alarm on. A Panic waiting at boot defers
    the write and keeps the debt."""
    clock = _boot(install)
    clock.stop(source=OWNER_STOP)
    assert clock.start(source=AGENT_STOP).startswith("Background consciousness stayed off")  # the mind's refused Start
    panic = install.root / "state" / "panic_stop.flag"
    panic.write_text("panic", encoding="utf-8")
    assert "Panic" in clock.start()
    panic.unlink()
    _boot(install)
    assert install.stored() == seed.ELIGIBLE and not install.address.path.exists() and install.on() is False

    with monkeypatch.context() as patch:  # the accepted Start's write fails; the reply says so; then my human's Off
        patch.setattr(seed, "_write", lambda *_a: (_ for _ in ()).throw(OSError("disk full")))
        _plain(clock.start())
        assert _settled(install)[-1] == ("retrying", "OSError: disk full")
        clock.stop(source=OWNER_STOP)
    assert install.stored() == seed.OWED and not install.address.path.exists() and install.on() is False
    panic.write_text("panic", encoding="utf-8")
    _boot(install)
    assert install.stored() == seed.OWED and not install.address.path.exists(), "Emergency Stop first"
    panic.unlink()
    off = _boot(install)
    assert install.stored() == seed.WRITTEN and _held_by_wake(install) == [install.address]
    assert install.on() is False and off.enabled is False, "the boot wrote the debt and turned nothing on"
    assert [row["writer"] for row in _rows(install)] == ["host_seed"]

    # A cut right after the accepting write: the debt alone, then the boot.
    for name in ("state.json", "state.last_good.json", "state.initialized.json"):
        (install.root / "state" / name).unlink(missing_ok=True)
    install.address.path.unlink()
    install.state.init_state()
    assert _persist_start(OWNER_STOP) == ("", "") and install.stored() == seed.OWED
    _boot(install)
    assert install.stored() == seed.WRITTEN and install.address.path.exists()


def test_a_note_that_stands_at_the_address_is_left_as_it_is_and_said_once(install):
    install.state.init_state()
    authored = knowledge.write_knowledge_note(install.address, "My human's own words about evenings.", writer="turn")
    _plain(install.clock().start())
    assert _settled(install) == [("collision", "")] and install.stored() == seed.COLLISION
    assert install.address.path.read_bytes() == authored.current.raw and not _held_by_wake(install)
    _plain(install.clock().start())
    assert _settled(install) == [("collision", "")] and [row["writer"] for row in _rows(install)] == ["turn"]


@pytest.mark.parametrize("after", ["deleted", "moved", "unbound", "edited", "rewritten_outside"])
def test_a_published_seed_whose_outcome_record_was_lost_is_done_whatever_came_after(install, after):
    install.state.init_state()
    install.clock().start()
    live = knowledge.read_knowledge_note(install.address)
    if after == "deleted":
        install.address.path.unlink()
    elif after == "moved":
        os.replace(install.address.path, install.address.shelf / "evenings.md")
    elif after == "rewritten_outside":  # bytes replaced past the writer: no row, and still not the seed's to undo
        install.address.path.write_text("Rewritten by hand.", encoding="utf-8")
    else:
        change = "---\nclaims: []\n---\n" if after == "unbound" else "Mine now."
        assert knowledge.write_knowledge_note(install.address, change, "edit" if after == "unbound" else "append",
                                              expected_revision=live.revision, writer="turn").ok
    raw = install.address.path.read_bytes() if install.address.path.exists() else None
    install.state.update_state(lambda st: st.update({seed.STATE_KEY: seed.OWED}))  # the outcome's record lost
    _boot(install)
    assert install.stored() == seed.WRITTEN
    assert (install.address.path.read_bytes() if install.address.path.exists() else None) == raw, "nothing recreated"
    assert _held_by_wake(install) == ([install.address] if after == "edited" else [])  # no claim restored
    assert [row["writer"] for row in _rows(install)].count("host_seed") == 1


@pytest.mark.parametrize("then", ["untouched", "gone", "edited", "edited_then_gone", "rewritten_outside"])
def test_an_unfinished_write_is_finished_only_while_untouched_and_a_later_choice_stands(install, monkeypatch, then):
    """A capture whose claims never landed is no seed. Untouched, the next boot publishes that same write, and one that
    left nothing to finish is written again; a later write by another, kept or deleted since, and bytes changed outside
    the writer stand as they are, unclaimed."""
    install.state.init_state()
    with monkeypatch.context() as patch:
        patch.setattr(knowledge_versions.ClaimWrite, "publish", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk")))
        _plain(install.clock().start())
    assert _settled(install) == [("retrying", "publication_incomplete")]
    assert install.address.path.exists() and not _held_by_wake(install) and install.stored() == seed.OWED
    if then.startswith("edited"):
        live = knowledge.read_knowledge_note(install.address)
        knowledge.write_knowledge_note(install.address, "Mine now.", "append", expected_revision=live.revision,
                                       writer="turn")
    if then.endswith("gone"):
        install.address.path.unlink()
    if then == "rewritten_outside":
        install.address.path.write_text("Rewritten by hand.", encoding="utf-8")
    finished = then in ("untouched", "gone")
    raw = seed.SEED_NOTE.encode("utf-8") if finished else (
        install.address.path.read_bytes() if install.address.path.exists() else None)
    _boot(install)
    assert (install.address.path.read_bytes() if install.address.path.exists() else None) == raw
    assert install.stored() == (seed.WRITTEN if finished else seed.COLLISION)
    assert _held_by_wake(install) == ([install.address] if finished else [])


def test_a_claim_committed_but_never_projected_is_held_after_the_next_boot(install, monkeypatch):
    install.state.init_state()
    with monkeypatch.context() as patch:  # the claims commit lands; the record a wake reads does not
        patch.setattr(knowledge_versions, "_project_claim_record", lambda *_a: (_ for _ in ()).throw(OSError("disk")))
        _plain(install.clock().start())
    assert _settled(install) == [("retrying", "publication_incomplete")]
    rows = _rows(install, commits=True)
    assert knowledge_versions.is_claim_commit(rows[0]) and not _held_by_wake(install) and install.stored() == seed.OWED
    _boot(install)
    assert install.stored() == seed.WRITTEN and _held_by_wake(install) == [install.address]
    assert _rows(install, commits=True) == rows, "that same publication finished: no row added"


@pytest.mark.parametrize("old_note", ["kept", "deleted"])
def test_an_owner_reset_that_left_the_old_history_inherits_nothing_from_the_earlier_initialization(install, old_note):
    from supervisor.state_initialization import mark_pending

    first = install.state.init_state().values["initialization_id"]
    install.clock().start()
    if old_note == "deleted":
        install.address.path.unlink()
    raw = install.address.path.read_bytes() if old_note == "kept" else None
    mark_pending(install.root, origin="owner_reset")  # the Reset's witness; its memory wipe did not happen here
    second = install.state.init_state().values
    assert second["initialization_id"] != first and second[seed.STATE_KEY] == seed.ELIGIBLE
    _plain(install.clock().start())
    if old_note == "kept":  # the earlier initialization's note is a note that stood here: left as it is
        assert _settled(install)[-1] == ("collision", "") and install.stored() == seed.COLLISION
        assert install.address.path.read_bytes() == raw
    else:  # its history alone proves nothing for this initialization: a seed of its own
        assert install.stored() == seed.WRITTEN and install.address.path.exists()
        newest = _rows(install)[0]
        assert newest["writer_input_ref"]["initialization_id"] == second["initialization_id"]


def test_an_older_install_hears_the_offer_in_direct_main_only_and_building_requests_spends_nothing(install):
    """Main, my human's word, the owner's bound phone chat: offered. A Project room, a wake, a helper, a turn no word
    of my human starts: never. Building requests (a preview) spends nothing and writes no note."""
    from ouroboros.projects_registry import create_project

    _legacy(install)
    project = create_project(install.root, "garden", name="Garden")["chat_id"]
    ds.append(install.root, "chat", _owner("Hello.", T0, cmid="w1"))
    ds.append(install.root, "chat", _owner("From the phone.", T0 + 1, chat_id=PHONE_ROOM, cmid="t1"))
    ds.append(install.root, "chat", _owner("In the room.", T0 + 2, chat_id=project, cmid="p1"))
    phone = {**MAIN, "chat_id": PHONE_ROOM, "origin_message_ref": {"chat_id": PHONE_ROOM, "client_message_id": "t1"}}
    room = {**MAIN, "chat_id": project, "origin_message_ref": {"chat_id": project, "client_message_id": "p1"}}
    child = {**MAIN, "delegation_role": "subagent"}
    quiet = (room, WAKE_TURN, child, {**MAIN, "origin_message_ref": {}}, {**MAIN, "_is_direct_chat": False})
    for _preview in range(2):
        assert all(seed.OFFER_TEXT in _system_text(_plan(install.world, task=task)) for task in (MAIN, phone))
        assert not any(seed.OFFER_LINE in _system_text(_plan(install.world, task=task)) for task in quiet)
    assert install.stored() == "absent" and not install.address.path.exists()


def _main_loop(install, monkeypatch, *, sent=None, answer=None, window=900_000):
    """The real loop over the real plan of a direct Main turn; the stub transport records each physical candidate as
    the send path does (``persist_physical_candidate`` and the attempt's capture) from the request it is given."""
    from ouroboros import loop
    from ouroboros import usage_accounting as ua
    from ouroboros.tools.registry import ToolRegistry

    for key, value in (("OUROBOROS_CONTEXT_MODE", "max"), ("OUROBOROS_TASK_REVIEW_MODE", "off"),
                       ("OUROBOROS_SAFETY_MODE", "off"), ("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off"),
                       ("OUROBOROS_MAX_ROUNDS", "4"), ("MCP_ENABLED", "false")):
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_args: False)
    registry = ToolRegistry(repo_dir=install.world[0].repo_dir, drive_root=install.root)
    registry._ctx.task_id = MAIN["id"]
    registry._ctx.context_fit_plan = plan = replace(_plan(install.world, window, task=MAIN), start_reasoning_effort="medium")
    assert seed.SECTION_ID in {part.section_id for part in plan.core.dynamic_sections}  # the plan holds the offer
    sends = itertools.count(1)

    class Provider:
        def default_model(self):
            return plan.model

        def chat(self, **kwargs):
            from ouroboros.llm_attempt import _attempt_request, _physical_candidate
            from ouroboros.model_send_seal import persist_physical_candidate

            ua.adopt_physical_attempt_capture(None)
            if sent is not None:
                kwargs["messages"] = sent(deepcopy(kwargs["messages"]))
            candidate = _physical_candidate({key: kwargs[key] for key in ("messages", "tools", "model")})
            request = _attempt_request({"provider": "openai", "usage_model": kwargs["model"]}, candidate)
            attempt = f"seed-send-{next(sends)}"
            persisted = persist_physical_candidate(install.root, task_id=MAIN["id"], attempt_id=attempt,
                                                   candidate=candidate, candidate_facts={})
            ua.adopt_physical_attempt_capture(ua.PhysicalAttemptCapture(
                attempt, kwargs["model"], "openai", "settled", "canonical_json_v1",
                candidate_manifest_ref=persisted["manifest_ref"], physical_context=request.physical_context,
                candidate_raw_sha256=request.candidate_raw_sha256, candidate_raw_size_bytes=request.candidate_raw_size_bytes,
                candidate_context_sha256=request.candidate_context_sha256,
                candidate_context_size_bytes=request.candidate_context_size_bytes))
            reply = answer(int(attempt.rpartition("-")[2])) if callable(answer) else answer
            return deepcopy(reply or {"role": "assistant", "content": "Noted."}), {
                "prompt_tokens": 100, "completion_tokens": 10, "cost": 0.0, "provider": "openai"}

    return loop.run_llm_loop(messages=plan.messages_for(plan.initial_mode), tools=registry, llm=Provider(),
                             initial_effort="medium", drive_logs=install.root / "logs", drive_root=install.root,
                             emit_progress=lambda *_a, **_kw: None, incoming_messages=queue.Queue(),
                             task_id=MAIN["id"], event_queue=queue.Queue())


def _without_offer(messages):
    for row in messages:
        if row.get("role") == "system":
            for block in row["content"] if isinstance(row["content"], list) else ():
                block["text"] = block["text"].replace(seed.OFFER_TEXT, "")
    return messages


@pytest.mark.parametrize("frame", ["carries_the_block", "lacks_the_block", "only_the_words"])
def test_only_an_answered_physical_request_that_carried_the_offer_spends_it(install, monkeypatch, frame):
    _legacy(install)
    ds.append(install.root, "chat", _owner("Hello.", T0, cmid="w1"))
    sent = {"carries_the_block": None, "lacks_the_block": _without_offer,
            "only_the_words": lambda _messages: [{"role": "user", "content": "Hello."}]}[frame]
    answer, _usage, _trace = _main_loop(install, monkeypatch, sent=sent)
    assert answer == "Noted."  # every frame was answered
    assert install.stored() == (seed.OFFERED if frame == "carries_the_block" else "absent")
    assert not install.address.path.exists()


def test_a_send_answered_by_nothing_spends_nothing(install, monkeypatch):
    import ouroboros.loop as loop_mod

    def unanswered(ctx):
        ctx.accumulated_usage.update(_last_llm_error_kind="context_overflow", execution_status="infra_failed",
                                     reason_code="llm_api_error")
        return None, 0.0, ctx.active_context_mode

    _legacy(install)
    ds.append(install.root, "chat", _owner("Hello.", T0, cmid="w1"))
    monkeypatch.setattr(loop_mod, "_call_round_model", unanswered)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", lambda **_kw: None)
    _main_loop(install, monkeypatch)
    assert install.stored() == "absent"


# Recovery edges of the seed's own lifecycle: a Panic between accept and publish, half-written captures,
# journal failures, later writes and deletions under a lost outcome record.

def test_a_panic_pending_after_an_accepted_start_writes_nothing_and_the_debt_waits(install, monkeypatch):
    """A Panic that arrives between the Start's accepting write and the seed's publication holds it, whichever door
    accepted the Start; the next boot after the Panic clears writes it (an ordinary accepted Start still publishes)."""
    import ouroboros.consciousness as clock_module

    install.state.init_state()
    panic = install.root / "state" / "panic_stop.flag"
    accepting = clock_module._persist_start

    def then_panic(source):
        answer = accepting(source)
        panic.write_text("panic", encoding="utf-8")
        return answer

    monkeypatch.setattr(clock_module, "_persist_start", then_panic)
    install.clock().start()
    assert install.stored() == seed.OWED and not install.address.path.exists() and not _settled(install)
    panic.unlink()
    _boot(install)
    assert install.stored() == seed.WRITTEN and _held_by_wake(install) == [install.address]


def _half_written(install, monkeypatch):
    """The seed's bytes landed, its claims commit did not: the debt stays, the note holds the seed unclaimed."""
    install.state.init_state()
    with monkeypatch.context() as patch:
        patch.setattr(knowledge_versions.ClaimWrite, "publish", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk")))
        install.clock().start()
    assert install.stored() == seed.OWED and install.address.path.exists() and not _held_by_wake(install)


def test_a_journal_read_that_fails_concludes_nothing_and_the_debt_retries(install, monkeypatch):
    _half_written(install, monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(knowledge_versions, "_journal_lines", lambda *_a, **_kw: (_ for _ in ()).throw(PermissionError()))
        _boot(install)
    assert install.stored() == seed.OWED, "an unread journal is no collision"
    assert _settled(install)[-1] == ("retrying", "history_unreadable")
    _boot(install)
    assert install.stored() == seed.WRITTEN and _held_by_wake(install) == [install.address]


def test_a_later_write_that_replaced_no_bytes_is_no_edit_and_the_seed_is_finished(install, monkeypatch):
    _half_written(install, monkeypatch)
    live = knowledge.read_knowledge_note(install.address)
    with monkeypatch.context() as patch:  # the mind's append records its capture, then fails before the bytes land
        patch.setattr(knowledge, "write_bytes_atomic", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk")))
        failed = knowledge.write_knowledge_note(install.address, "Mine now.", "append",
                                                expected_revision=live.revision, writer="turn")
    assert failed.reason == "publication_incomplete" and [row["writer"] for row in _rows(install)] == ["turn", "host_seed"]
    _boot(install)
    assert install.stored() == seed.WRITTEN and _held_by_wake(install) == [install.address]
    assert install.address.path.read_bytes() == seed.SEED_NOTE.encode("utf-8")


def test_a_claim_removed_and_the_text_restored_while_the_seed_retries_stays_removed(install, monkeypatch):
    """The mind's two writes land between the retry's start and its write: the decision under the shelf lock sees them,
    so the restored bytes do not bring back the removed claim."""
    install.state.init_state()
    install.clock().start()
    install.state.update_state(lambda st: st.update({seed.STATE_KEY: seed.OWED}))  # the outcome's record lost
    lock, raced = knowledge.knowledge_write_lock, []

    def mind_first(shelf):
        if not raced:
            raced.append(True)
            live = knowledge.read_knowledge_note(install.address)
            unbound = knowledge.write_knowledge_note(install.address, "---\nclaims: []\n---\n", "edit",
                                                     expected_revision=live.revision, writer="turn")
            assert unbound.ok and not _held_by_wake(install)
            restored = knowledge.write_knowledge_note(install.address, "", "revert", writer="turn",
                                                      expected_revision=unbound.current.revision, version=live.revision)
            assert restored.ok and install.address.path.read_bytes() == seed.SEED_NOTE.encode("utf-8")
        return lock(shelf)

    monkeypatch.setattr(knowledge, "knowledge_write_lock", mind_first)
    _boot(install)
    assert raced and install.stored() == seed.WRITTEN
    assert not _held_by_wake(install), "the removed claim stays removed"


@pytest.mark.parametrize("after", ["moved", "edited"])
def test_a_second_initialization_publishes_its_own_seed_beside_an_identical_earlier_one(install, after):
    """A Reset whose memory wipe left the journal: the first initialization published the same bytes and claims. The
    second's seed still records its own publication, so with its outcome lost a move is not undone and an edit is no
    collision."""
    from supervisor.state_initialization import mark_pending

    install.state.init_state()
    install.clock().start()
    install.address.path.unlink()
    mark_pending(install.root, origin="owner_reset")
    install.state.init_state()
    install.clock().start()
    assert install.stored() == seed.WRITTEN and install.address.path.exists()
    install.state.update_state(lambda st: st.update({seed.STATE_KEY: seed.OWED}))  # the outcome's record lost
    if after == "moved":
        os.replace(install.address.path, install.address.shelf / "evenings.md")
    else:
        live = knowledge.read_knowledge_note(install.address)
        assert knowledge.write_knowledge_note(install.address, "Mine now.", "append", expected_revision=live.revision,
                                              writer="turn").ok
    raw = install.address.path.read_bytes() if install.address.path.exists() else None
    _boot(install)
    assert install.stored() == seed.WRITTEN, "its own seed was published: no collision"
    assert (install.address.path.read_bytes() if install.address.path.exists() else None) == raw, "nothing recreated"


def test_a_publication_restored_onto_this_initializations_source_honors_a_later_deletion(install, monkeypatch):
    """A Reset that kept the first initialization's publication: the second's half-written capture retries with
    equal bytes and claims, which must not dedup onto the earlier initialization's commit — the restored names are
    committed for this source, so a later lost outcome honors the deletion instead of resurrecting the seed."""
    from supervisor.state_initialization import mark_pending

    install.state.init_state()
    install.clock().start()  # the first initialization: its capture and its claims commit
    mark_pending(install.root, origin="owner_reset")
    install.state.init_state()
    install.address.path.unlink()  # the mind deleted the note before this initialization's first settle
    with monkeypatch.context() as patch:  # its own write lands, its claims commit does not
        patch.setattr(knowledge_versions.ClaimWrite, "publish", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk")))
        install.clock().start()
    assert install.stored() == seed.OWED and install.address.path.exists()  # the claim record keeps the earlier
    # initialization's published names until this source's own commit replaces them.
    _boot(install)  # the retry: equal bytes and claims restore the names
    assert install.stored() == seed.WRITTEN and _held_by_wake(install) == [install.address]
    newest = _rows(install, commits=True)[0]
    assert knowledge_versions.is_claim_commit(newest) and newest["writer"] == knowledge.SEED_WRITER, \
        "the restored publication is this source's own commit, not the earlier initialization's"
    rows = _rows(install, commits=True)
    install.state.update_state(lambda st: st.update({seed.STATE_KEY: seed.OWED}))
    _boot(install)
    assert _rows(install, commits=True) == rows and install.stored() == seed.WRITTEN, \
        "an already published record adds no row"
    install.address.path.unlink()
    install.state.update_state(lambda st: st.update({seed.STATE_KEY: seed.OWED}))  # the outcome's record lost
    _boot(install)  # the boot settles the debt itself: once published, the deletion stands
    assert not install.address.path.exists() and install.stored() == seed.WRITTEN


def test_a_start_reads_the_journal_once(install, monkeypatch):
    install.state.init_state()
    scans, lines = [], knowledge_versions._journal_lines
    monkeypatch.setattr(knowledge_versions, "_journal_lines",
                        lambda journal, needle, **kw: scans.append(needle) or lines(journal, needle, **kw))
    install.clock().start()
    assert install.stored() == seed.WRITTEN and scans.count(json.dumps(seed.TOPIC).encode()) == 1


def test_a_seed_a_failure_left_owed_is_published_before_the_next_wake_and_a_settled_clock_does_no_work(install,
                                                                                                    monkeypatch):
    import ouroboros.consciousness as clock_module
    from supervisor import workers
    from tests.test_consciousness import AVAILABLE

    install.state.init_state()
    clock = install.clock()
    with monkeypatch.context() as patch:
        patch.setattr(seed, "_write", lambda *_a: (_ for _ in ()).throw(OSError("disk full")))
        clock.start()
    assert install.stored() == seed.OWED and install.on() is True
    launched = []
    monkeypatch.setattr(workers, "handle_wake_direct", lambda *_a, **_kw: launched.append(_held_by_wake(install)) or {
        "admitted": False, "task_id": "", "reason": "held by the test"})
    monkeypatch.setattr(clock_module, "allowance_window", lambda *_a, **_kw: dict(AVAILABLE))
    monkeypatch.setattr("supervisor.queue.live_consciousness_roots", lambda: [])
    clock.tick(clock.next_wake_at)
    assert launched == [[install.address]] and install.stored() == seed.WRITTEN
    asked = []
    monkeypatch.setattr(seed, "settle", lambda *_a, **_kw: asked.append(True))
    clock.tick(clock.next_wake_at)
    assert len(launched) == 2 and not asked, "a settled clock asks nothing before a wake"


def _measured_window(install):
    """This rig's own two boundaries, measured through the real builder: one refusal of a window too small for the
    turn's minimum view names its exact size, and the first window that stands the whole offer block is found by
    asking the builder downward from _main_loop's known-good window, verified before the search starts. Between
    them the offer rides as one line with its reader, whatever a neighbouring run shifts the estimate by."""
    from ouroboros import context_fit

    try:
        _plan(install.world, RESERVE, task=MAIN)
    except context_fit.MinimumViewUnfit as unfit:
        smallest = unfit.estimated_input_tokens + unfit.response_reserve_tokens
    else:
        raise AssertionError("a window of the output reserve alone held this rig's view; the rig changed")
    whole = 900_000  # the large known-good _plan setup (_main_loop's window): a finite, verified upper bound
    assert seed.OFFER_TEXT in _system_text(_plan(install.world, whole, task=MAIN)), \
        "the bound itself stands the whole offer block"
    below = smallest
    while whole - below > 1:
        middle = (below + whole) // 2
        if seed.OFFER_TEXT in _system_text(_plan(install.world, middle, task=MAIN)):
            whole = middle
        else:
            below = middle
    return smallest, whole


def test_a_tight_window_admits_the_offer_in_one_line_and_its_answered_send_spends_it(install, monkeypatch):
    """The window is this rig's own measurement, not a number: below the first window that stands the whole offer
    block the builder itself reduces it to one line with a working reader, an answered send carrying that form
    spends it, and the next direct Main turn carries none."""
    _legacy(install)
    ds.append(install.root, "chat", _owner("Hello.", T0, cmid="w1"))
    smallest, whole = _measured_window(install)
    window = (smallest + whole) // 2
    assert smallest < window < whole, "the whole offer block needs more room than the window has"
    plan = _plan(install.world, window, task=MAIN)
    text = _system_text(plan)
    assert seed.OFFER_VIEW in text and seed.OFFER_TEXT not in text and "retained full text: read_file(" in text
    from ouroboros.artifacts import read_actor_source_bytes

    (offer,) = [row for row in plan.projection(plan.initial_mode).memory_facts.get("host_sections_addressed", ())
                if row.get("section_id") == seed.SECTION_ID]
    assert read_actor_source_bytes(install.root, MAIN["id"], offer["source_ref"]) == seed.OFFER_TEXT.encode("utf-8"), \
        "the one-line form's named reader holds the whole offer block"
    answer, _usage, _trace = _main_loop(install, monkeypatch, window=window)
    assert answer == "Noted." and install.stored() == seed.OFFERED
    ds.append(install.root, "chat", _owner("Again.", T0 + 5, cmid="w2"))
    later = _system_text(_plan(install.world, window, task={**MAIN, "origin_message_ref": {"chat_id": 1,
                                                                                          "client_message_id": "w2"}}))
    assert seed.OFFER_VIEW not in later and seed.OFFER_LINE not in later


def test_one_turn_reads_my_humans_word_once(install, monkeypatch):
    _legacy(install)
    ds.append(install.root, "chat", _owner("Hello.", T0, cmid="w1"))
    reads, accepted = [], ds.accepted_row
    monkeypatch.setattr(ds, "accepted_row", lambda *a, **kw: reads.append(a[1:]) or accepted(*a, **kw))
    assert seed.OFFER_TEXT in _system_text(_plan(install.world, task=MAIN))
    assert reads.count((1, "w1")) == 1


def test_a_forced_final_answer_that_carried_the_offer_spends_it(install, monkeypatch):
    """Every round's request went without the offer and asked for a tool; the forced final, the one request that
    carried it, was answered: it is spent there as a round's would be."""
    _legacy(install)
    ds.append(install.root, "chat", _owner("Hello.", T0, cmid="w1"))
    rounds = itertools.count(1)

    def answer_for(n):
        if n <= 4:
            return {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call-{n}", "type": "function", "function": {"name": "no_such_tool", "arguments": "{}"}}]}
        return {"role": "assistant", "content": "Final."}

    final, _usage, _trace = _main_loop(install, monkeypatch, sent=lambda m: _without_offer(m) if next(rounds) <= 4 else m,
                                       answer=answer_for)
    assert final.strip().endswith("Final.") and install.stored() == seed.OFFERED
