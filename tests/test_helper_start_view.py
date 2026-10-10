"""What a delegated child, its sibling and a forked child start with.

Children are scheduled by the real ``schedule_subagent`` and their first request is built
by ``context.build_llm_messages``, as their worker builds it. Two children of different
trees read one story: block B is the same bytes for both, and its story is the mind's.
Two siblings of one parent read one room page and one block of the owner's words; a
child of another tree reads its own. Before the chronicle can be read a child still
starts, with a visible gap and no model call. A forked child's own drive holds only its
process: its story and its room page come from the canonical data root. Every rule is
checked in both directions.
"""
from __future__ import annotations

import hashlib
import pathlib
import queue

import pytest

from ouroboros import memory_view as mv
from ouroboros.agent import Env
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.loop_messages import _initialize_owner_directives
from ouroboros.memory import Memory
from ouroboros.tools import control
from ouroboros.tools.control_scheduling import _schedule_task
from ouroboros.tools.registry import ToolContext
from supervisor.task_dispatch import build_scheduled_task_payload
from tests._memory_view_context import blocks, section, world
from tests.test_available_subagents_runtime import _api_row, _settings

ASKED = "Count the alpha inventory\nand name what is missing"
ASKED_BETA = "Look at the beta numbers"
WORDS = "## Words of my human that caused this work (verbatim)"
MAIN = {"id": "tmain", "chat_id": 1}


@pytest.fixture(autouse=True)
def _one_api_actor(monkeypatch):
    monkeypatch.setattr(control, "load_settings", lambda: _settings(_api_row()))
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "4")


def _root(env, memory, task_id, chat_id, asked):
    """A root that holds the owner's words in its corpus, as its worker records them."""
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=memory.drive_root, current_task_type="task",
                      current_chat_id=chat_id)
    ctx.task_id, ctx.event_queue = task_id, queue.Queue()
    ctx.task_metadata = {"root_task_id": task_id, "origin_message_ref": {
        "chat_id": chat_id, "ts": "2026-10-04T05:00:00+00:00", "client_message_id": f"msg-{task_id}"}}
    ctx.task_contract = build_task_contract({"id": task_id, "type": "task", "text": asked})
    _initialize_owner_directives(ctx, [{"role": "user", "content": asked}])
    return ctx


def _schedule(ctx, objective, memory_mode="forked", **extra):
    """One real schedule: the supervisor's payload of the child, as its worker receives it."""
    result = _schedule_task(ctx, subagent_id="api-builder", objective=objective, expected_output="a list",
                            memory_mode=memory_mode, **extra)
    assert not result.startswith("⚠️"), result
    event = ctx.event_queue.get_nowait()
    return build_scheduled_task_payload({**event, "tid": event["task_id"], "parent_id": ctx.task_id,
                                         "text": event["objective"], "desc": event["objective"]})


def _start(env, payload):
    """``(A, B, C, cap)`` of the child's first request, built on its own drive like its worker."""
    own = pathlib.Path(payload["drive_root"] or payload["budget_drive_root"])
    child_env = Env(repo_dir=env.repo_dir, drive_root=own, budget_drive_root=pathlib.Path(payload["budget_drive_root"]))
    return blocks(child_env, Memory(drive_root=own, repo_dir=env.repo_dir), payload)


def _files(root: pathlib.Path) -> dict:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


# --- one story for every child; one room page and one block of words for siblings ----------------

def test_children_of_two_trees_share_block_b_and_siblings_share_the_room_page_and_the_words(tmp_path):
    env, memory, rooms = world(tmp_path)
    alpha_root = _root(env, memory, "bound", 1, ASKED)  # bound to Project alpha by an owner message in Main
    beta_root = _root(env, memory, "tb", rooms["beta"], ASKED_BETA)
    first, second = _schedule(alpha_root, "Count shelf one"), _schedule(alpha_root, "Count shelf two")
    cousin = _schedule(beta_root, "Check the beta totals")
    assert first["metadata"]["governing_owner_words"] == second["metadata"]["governing_owner_words"]
    views = {name: _start(env, payload) for name, payload in (("first", first), ("second", second), ("cousin", cousin))}
    _a_main, b_main, _c_main, _cap = blocks(env, memory, MAIN)

    # B: one story whatever the tree, the mind's own, and no fact of the child in it; it names by pointer the
    # first block of the old retelling that Main reads whole.
    assert views["first"][1] == views["second"][1] == views["cousin"][1]
    story, main_story = section(views["first"][1], "## My story"), section(b_main, "## My story")
    assert story.startswith("## My story\n") and "memory_read(node_id='legacy-b00-r1')" in story
    assert "  Main talk." in main_story and "Main talk." not in story
    assert views["first"][0] == views["second"][0] == views["cousin"][0]  # A: governance and the books' maps
    for payload in (first, second, cousin):
        assert payload["id"] not in views["first"][1] and payload["id"] in _start(env, payload)[2]

    # C: siblings read one room page and one block of the owner's words, verbatim.
    alpha = f"## This room (Project Alpha [chat_id={rooms['alpha']}])"
    beta = f"## This room (Project Beta [chat_id={rooms['beta']}])"
    page = {name: section(views[name][2], alpha if name != "cousin" else beta) for name in views}
    words = {name: section(views[name][2], WORDS) for name in views}
    assert page["first"] == page["second"] and "### Retold before the update" in page["first"]
    assert words["first"] == words["second"]
    assert "\n  Count the alpha inventory\n  and name what is missing" in words["first"]
    assert "Count shelf" not in words["first"]  # the assignment is not the owner's words
    # A child of another tree reads its own room and its own root's words.
    assert alpha not in views["cousin"][2] and page["cousin"] != page["first"]
    assert ASKED_BETA in words["cousin"] and "alpha inventory" not in words["cousin"]
    # Each child keeps its own process: the whole changing block is not shared.
    assert views["first"][2] != views["second"][2]


# --- a chronicle that cannot be read leaves a visible gap and no model call --------------------------

@pytest.fixture
def no_model_client(monkeypatch):
    """A model client fails when it is made: a start that needs one is a red test."""
    from ouroboros.llm import LLMClient

    def forbidden(*_a, **_kw):
        raise AssertionError("a helper's start makes no model call")

    monkeypatch.setattr(LLMClient, "__init__", forbidden)
    monkeypatch.setattr(LLMClient, "chat", forbidden)
    with pytest.raises(AssertionError, match="makes no model call"):  # the stub has teeth
        LLMClient()


@pytest.mark.parametrize("failure,reason", [
    ({"kind": "import_pending", "reason": "legacy_memory_lock_busy"}, "legacy_memory_lock_busy"),
    ({"kind": "import_refused", "reason": "invalid"}, "invalid"),
    (OSError("journal io"), "OSError: journal io"),
    (ValueError("chronicle authority shortened"), "ValueError: chronicle authority shortened"),
])
def test_a_child_starts_with_a_visible_gap_when_its_chronicle_cannot_be_read(tmp_path, monkeypatch, no_model_client,
                                                                            failure, reason):
    env, memory, rooms = world(tmp_path)
    child = _schedule(_root(env, memory, "bound", 1, ASKED), "Count shelf one")

    def unavailable(self, **_kw):
        if isinstance(failure, Exception):
            raise failure
        return dict(failure)

    readable = ChronicleStore.ensure_activated
    monkeypatch.setattr(ChronicleStore, "ensure_activated", unavailable)
    _a, b, c, cap = _start(env, child)
    assert f"## My story — unavailable now ({reason})" in b
    assert "read_file(root='runtime_data', path='memory/dialogue_blocks.json')" in b
    assert (f"## This room (Project Alpha [chat_id={rooms['alpha']}])\n\nOpen conversation unavailable until my "
            f"memory is activated ({reason}); read it: chat_history(count=100)") in c
    assert "### Retold before the update" not in c and "## Marks I keep in view" not in c
    # What does not come from the chronicle is whole: the owner's words and the role line.
    assert "\n  Count the alpha inventory\n  and name what is missing" in section(c, WORDS)
    loaded, missing = section(c, "## Working sources").split("Loaded above: ", 1)[1].split(" Not loaded: ", 1)
    assert f"(my memory is not activated yet: {reason})" in missing
    assert "the words of my human that caused this work" in loaded
    assert cap["memory_view"]["store_status"]["state"] != "active"
    # The same child on a readable chronicle sees its story and its room page, and no gap (still no model).
    monkeypatch.setattr(ChronicleStore, "ensure_activated", readable)
    _a, healthy_b, healthy_c, _cap = _start(env, child)
    assert section(healthy_b, "## My story").startswith("## My story\n") and "unavailable now" not in healthy_b
    assert "### Retold before the update" in healthy_c and "not activated yet" not in healthy_c


# --- a forked child reads its story and its room page from the canonical root -------------------------

def test_a_forked_child_reads_its_story_and_room_page_canonically_not_from_its_own_drive(tmp_path):
    import json

    env, memory, rooms = world(tmp_path)
    canonical = memory.drive_root
    child = _schedule(_root(env, memory, "bound", 1, ASKED), "Count shelf one", memory_mode="forked")
    fork = pathlib.Path(child["drive_root"])
    assert child["memory_mode"] == "forked" and fork != canonical
    assert pathlib.Path(child["budget_drive_root"]) == canonical
    assert not (fork / "memory" / "chronicle").exists()  # the fork copies stable memory, not the chronicle
    _a, _b, _c, _cap = _start(env, child)
    assert not (fork / "memory" / "chronicle").exists()  # its start activates the canonical chronicle only

    # The fork's drive gets a memory of its own: an old retelling the canonical root never had.
    (fork / "memory" / "dialogue_blocks.json").write_text(json.dumps([{
        "ts": "2026-09-02T01:00:00+00:00", "type": "summary", "range": "2026-09-01 00:00 - 00:05", "message_count": 1,
        "content": "FORK DRIVE RETELLING", "rooms": [{"room_id": str(rooms["alpha"]), "label": "Alpha",
                                                      "message_count": 1, "content": "FORK DRIVE ALPHA PAGE"}]}]),
        encoding="utf-8")
    assert ChronicleStore(fork).ensure_activated()["kind"] == "activation"
    spec = mv.view_spec_for_task(child, canonical)
    assert spec.room_id == str(rooms["alpha"])
    own = mv.capture_memory_view(fork, child, spec)  # read from the fork, the same room would show this memory
    assert "FORK DRIVE ALPHA PAGE" in mv.render_room(own)
    before = _files(fork / "memory" / "chronicle")

    a, b, c, _cap = _start(env, child)
    assert "FORK DRIVE" not in a + b + c
    _a_main, b_main, _c_main, _cap = blocks(env, memory, MAIN)
    alpha = f"## This room (Project Alpha [chat_id={rooms['alpha']}])"
    canonical_view = mv.capture_memory_view(canonical, child, spec)
    assert section(b, "## My story").rstrip("\n") == mv.render_story(canonical_view)  # the mind's story, canonically
    assert section(b, "## My story") != section(b_main, "## My story")  # a child's: the first block by pointer
    assert section(c, alpha).rstrip("\n") == section(mv.render_room(canonical_view), alpha)
    assert "Alpha began." in section(c, alpha)  # the canonical retelling of the room
    assert _files(fork / "memory" / "chronicle") == before  # the fork's own memory is untouched


# --- a declared child starts from its parent's selection; the words change nothing there -------------

def test_a_declared_child_starts_from_its_parents_selection_and_the_carried_words_change_nothing(tmp_path, monkeypatch):
    import copy
    import json

    from ouroboros.context import build_llm_messages

    env, memory, _rooms = world(tmp_path)
    monkeypatch.setattr("ouroboros.context.utc_now_iso", lambda: "2026-10-04T06:00:00+00:00")
    root = _root(env, memory, "bound", 1, ASKED)
    declared = _schedule(root, "Count shelf one", input_sources="declared", context="Shelf one holds forty boxes")
    assert declared["metadata"]["governing_owner_words"]  # the words ride with every child
    own = pathlib.Path(declared["drive_root"])

    def start(task):
        child_env = Env(repo_dir=env.repo_dir, drive_root=own, budget_drive_root=memory.drive_root)
        return build_llm_messages(env=child_env, memory=Memory(drive_root=own, repo_dir=env.repo_dir),
                                  task={"type": "task", "text": "hi", **task})

    messages, cap = start(declared)
    text = json.dumps(messages, ensure_ascii=False)
    assert "Shelf one holds forty boxes" in text and "Input source selection" in text
    assert "memory_view" not in cap and "## My story" not in text and "Words of my human" not in text
    assert "alpha inventory" not in text  # the parent did not select the owner's words
    receipt = json.loads(section(messages[0]["content"][-1]["text"], "## Input source selection").split("\n\n", 1)[1])
    assert any(item.startswith("the owner's words that caused this work") for item in receipt["omitted_automatic"])
    assert not (memory.drive_root / "memory" / "chronicle").exists()  # no capture, no activation
    bare = copy.deepcopy(declared)
    bare["metadata"].pop("governing_owner_words")
    assert start(bare)[0] == messages  # the carried field is not an input of a declared start
    # A shared sibling of the same parent starts with the words and its view.
    shared_child = _schedule(root, "Count shelf two")
    _a, _b, c, shared_cap = _start(env, shared_child)
    assert "alpha inventory" in section(c, WORDS) and shared_cap["memory_view"]["role"] == "child"
