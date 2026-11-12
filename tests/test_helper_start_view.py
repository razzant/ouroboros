"""What a delegated child, its sibling, a forked child and a declared child start with.

Children are scheduled by the real ``schedule_subagent`` and their first request is built
by ``context.build_llm_messages``, as their worker builds it. Two children of different
trees read one story: block B is the same bytes for both, and for Main: its story is the mind's.
Two siblings of one parent read one room page and one block of the owner's words; a
child of another tree reads its own. Before the chronicle can be read a child still
starts, with a visible gap and no model call. A forked child's own drive holds only its
process: its story and its room page come from the canonical data root. A declared child
reads the same block of words, the same marks and the same outline, and nothing else of
memory; one scheduled before its schedule recorded the words keeps its parent's selection.
Every rule is checked in both directions.
"""
from __future__ import annotations

import hashlib
import pathlib
import queue
from types import SimpleNamespace

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
WORDS = "## My human's words for this work (verbatim, dated, host-attested)"
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

    # B: one story whatever the tree, the mind's own, and no fact of the child in it: the same bytes Main reads,
    # the old retelling whole (no block is privileged, for a child either).
    assert views["first"][1] == views["second"][1] == views["cousin"][1]
    story, main_story = section(views["first"][1], "## My story"), section(b_main, "## My story")
    assert story.startswith("## My story\n") and story == main_story
    assert "  Main talk." in story and "  Alpha began." in story and "memory_read(node_id='legacy-" not in story
    assert views["first"][0] == views["second"][0] == views["cousin"][0]  # A: governance and the books' maps
    for payload in (first, second, cousin):
        assert payload["id"] not in views["first"][1] and payload["id"] in _start(env, payload)[2]

    # C: siblings read one room page and one block of the owner's words, verbatim.
    alpha = f"## This room (Project Alpha [chat_id={rooms['alpha']}])"
    beta = f"## This room (Project Beta [chat_id={rooms['beta']}])"
    page = {name: section(views[name][2], alpha if name != "cousin" else beta) for name in views}
    words = {name: section(views[name][2], WORDS) for name in views}
    assert page["first"] == page["second"] and page["first"].startswith(alpha + " — head ")
    assert "### Retold before the update" not in page["first"]  # its room's retold records are whole in the story
    assert words["first"] == words["second"]
    assert "\n  Count the alpha inventory\n  and name what is missing" in words["first"]
    assert "Count shelf" not in words["first"]  # the assignment is not the owner's words
    # A child of another tree reads its own room and its own root's words.
    assert alpha not in views["cousin"][2] and page["cousin"] != page["first"]
    assert ASKED_BETA in words["cousin"] and "alpha inventory" not in words["cousin"]
    # Each child keeps its own process: the whole changing block is not shared.
    assert views["first"][2] != views["second"][2]


# --- a helper is born with the knowledge outline: every note by address and title, both audiences --------

def test_a_child_and_a_nanny_start_with_the_knowledge_outline_not_the_whole_index(tmp_path):
    """A helper's first request holds every active note of the global shelf and its room's by address and title
    (a person's summary kept, a plain note's one read away), under its own heading; the whole index and the
    overview are not in it (they ride its first accepted delegation), and the working-sources line says so."""
    from ouroboros.knowledge import resolve_knowledge_address, write_knowledge_note

    env, memory, _rooms = world(tmp_path)
    root = memory.drive_root
    for topic, scope, note in (
            ("overview", "global", "---\ntitle: Overview\nsummary: WHOLE_OVERVIEW\n---\n# Overview\nOVERVIEW_BODY\n"),
            ("people/owner", "global", "---\ntitle: My human\ntype: person\nsummary: PERSON_SUMMARY\n---\n# Owner\n"),
            ("shelf-plan", "global", "---\ntitle: Shelf plan\nsummary: PLAIN_SUMMARY\n---\n# Plan\nPLAN_BODY\n"),
            ("room-plan", "project:alpha", "---\ntitle: Alpha plan\nsummary: ROOM_SUMMARY\n---\n# Alpha\nROOM_BODY\n")):
        assert write_knowledge_note(resolve_knowledge_address(root, topic, scope), note).ok
    alpha_root = _root(env, memory, "bound", 1, ASKED)
    alpha_root.project_id = "alpha"  # the parent's Project: its child is scheduled into the same shelf
    child = _schedule(alpha_root, "Count shelf one")
    assert child["project_id"] == "alpha"
    nanny = {"id": "nanny1", "chat_id": 1, "delegation_role": "subagent", "parent_task_id": "bound", "root_task_id": "bound",
             "project_id": "alpha", "configured_subagent": {"id": "h", "route": {"kind": "agent_session"}}}
    for who, (_a, _b, c, cap) in (("child", _start(env, child)), ("nanny", blocks(env, memory, nanny))):
        assert cap["memory_view"]["role"] == who and not cap["memory_view"]["spec"]["knowledge"], who
        outline, room = section(c, "## Knowledge outline"), section(c, "## Project knowledge outline (alpha)")
        for address, title in (("people/owner", "My human"), ("shelf-plan", "Shelf plan"), ("overview", "Overview")):
            assert address in outline and title in outline, who  # each note by address and title; the form is the outline's
        assert "PERSON_SUMMARY" in outline and "room-plan" in room and "Alpha plan" in room, who  # a person's summary stays
        assert "knowledge_list(scope='project:alpha')" in room, who
        assert "PLAIN_SUMMARY" not in c and "WHOLE_OVERVIEW" not in c and "ROOM_SUMMARY" not in c, who  # one read away
        assert "PLAN_BODY" not in c and "ROOM_BODY" not in c and "OVERVIEW_BODY" not in c, who
        assert "## Knowledge base" not in c and "## Shared understanding" not in c and "## Project workpad" not in c, who
        assert c.index("## Knowledge outline") < c.index("## Project knowledge outline") < c.index("## Runtime context"), who
        loaded, missing = section(c, "## Working sources").split("Loaded above: ", 1)[1].split(" Not loaded: ", 1)
        assert "the knowledge outline (every note by address and title)" in loaded, who
        assert "the whole knowledge index (it arrives with your first delegation)" in missing, who
        assert "knowledge_list, knowledge_read, journal_read, workpad_read" in missing, who


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
    assert "Alpha began." not in b + c and "## Marks I keep in view" not in c
    # What does not come from the chronicle is whole: the owner's words and the role line.
    assert "\n  Count the alpha inventory\n  and name what is missing" in section(c, WORDS)
    loaded, missing = section(c, "## Working sources").split("Loaded above: ", 1)[1].split(" Not loaded: ", 1)
    assert f"(my memory is not activated yet: {reason})" in missing
    assert "my human's words for this work" in loaded
    assert cap["memory_view"]["store_status"]["state"] != "active"
    # The same child on a readable chronicle sees its story and its room page, and no gap (still no model).
    monkeypatch.setattr(ChronicleStore, "ensure_activated", readable)
    _a, healthy_b, healthy_c, _cap = _start(env, child)
    assert section(healthy_b, "## My story").startswith("## My story\n") and "unavailable now" not in healthy_b
    assert "  Alpha began." in section(healthy_b, "## My story") and "not activated yet" not in healthy_c


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
    assert "FORK DRIVE ALPHA PAGE" in mv.render_story(own)
    before = _files(fork / "memory" / "chronicle")

    a, b, c, _cap = _start(env, child)
    assert "FORK DRIVE" not in a + b + c
    _a_main, b_main, _c_main, _cap = blocks(env, memory, MAIN)
    alpha = f"## This room (Project Alpha [chat_id={rooms['alpha']}])"
    canonical_view = mv.capture_memory_view(canonical, child, spec)
    assert section(b, "## My story").rstrip("\n") == mv.render_story(canonical_view)  # the mind's story, canonically
    assert section(b, "## My story") == section(b_main, "## My story")  # a child's story is Main's, byte for byte
    assert section(c, alpha).rstrip("\n") == section(mv.render_room(canonical_view), alpha)
    assert "Alpha began." in section(b, "## My story")  # the canonical retelling of the room, whole in the story
    assert _files(fork / "memory" / "chronicle") == before  # the fork's own memory is untouched


# --- a declared child starts with the owner's words, the marks and the outline, nothing else of memory --------

def test_a_declared_child_starts_with_the_owner_words_and_nothing_of_shared_memory(tmp_path, monkeypatch):
    import copy
    import json

    from ouroboros.context import build_llm_messages
    from ouroboros.subagent_work_order import DECLARED_WORDS_FACT
    from tests import _memory_inventory_shared as shared

    env, memory, rooms = world(tmp_path)
    monkeypatch.setattr("ouroboros.context.utc_now_iso", lambda: "2026-10-04T06:00:00+00:00")
    monkeypatch.setattr("shutil.disk_usage", lambda _path: SimpleNamespace(free=1, total=2))  # a dated fact as well
    store = ChronicleStore(memory.drive_root)
    assert store.ensure_activated()["kind"] == "activation"
    store.mark({"kind": "task", "task_id": "x"}, "For everyone", shared.MIND, room_id="1", scope="global")
    store.mark({"kind": "task", "task_id": "ta"}, "Watch alpha", shared.MIND, room_id=str(rooms["alpha"]))
    store.mark({"kind": "task", "task_id": "tb"}, "Watch beta", shared.MIND, room_id=str(rooms["beta"]))
    root = _root(env, memory, "bound", 1, ASKED)
    declared = _schedule(root, "Count shelf one", input_sources="declared", context="Shelf one holds forty boxes")
    assert declared["metadata"]["governing_owner_words"] and declared["metadata"][DECLARED_WORDS_FACT] is True
    own = pathlib.Path(declared["drive_root"])
    words_item = "my human's words for this work"

    def start(task):
        child_env = Env(repo_dir=env.repo_dir, drive_root=own, budget_drive_root=memory.drive_root)
        messages, cap = build_llm_messages(env=child_env, memory=Memory(drive_root=own, repo_dir=env.repo_dir),
                                           task={"type": "task", "text": "hi", **task})
        changing = messages[0]["content"][-1]["text"]  # block C: the outline, the words, the marks, the receipt, the facts
        receipt = json.loads(section(changing, "## Input source selection").split("\n\n", 1)[1])
        return messages, cap, changing, receipt

    messages, cap, changing, receipt = start(declared)
    text = json.dumps(messages, ensure_ascii=False)
    assert "Shelf one holds forty boxes" in text and "Count shelf one" in text
    assert "\n  Count the alpha inventory\n  and name what is missing" in section(changing, WORDS)
    marks, outline = section(changing, "## Marks I keep in view"), section(changing, "## Knowledge outline")
    assert "For everyone" in marks and "Watch alpha" in marks and "Watch beta" not in marks  # a shared child's marks
    assert (changing.index("## Knowledge outline") < changing.index(WORDS) < changing.index("## Marks I keep in view")
            < changing.index("## Input source selection") < changing.index("## Runtime context"))
    assert any(item.startswith(words_item) for item in receipt["included"])
    assert not any(item.startswith(words_item) for item in receipt["omitted_automatic"])
    assert "the memory marks in force (this room's and the global ones) and the knowledge outline" in str(receipt["included"])
    # Nothing else of memory: the view draws no story, room page or conversation; no identity or working-sources line.
    spec = cap["memory_view"]["spec"]
    assert cap["memory_view"]["role"] == "child" and not spec["story"] and not spec["room_page"] and not spec["room_lanes"]
    absent = ("## My story", "## This room", "## Identity", "## Working sources", "Main talk.", "Alpha began.")
    assert not [marker for marker in absent if marker in text]
    # The other side: scheduled before the fact, the child keeps its parent's selection; the words ride unread.
    older = copy.deepcopy(declared)
    older["metadata"].pop(DECLARED_WORDS_FACT)
    older_messages, _cap, older_changing, older_receipt = start(older)
    assert WORDS not in older_changing and "alpha inventory" not in json.dumps(older_messages)
    assert any(item.startswith(words_item) for item in older_receipt["omitted_automatic"])
    assert section(older_changing, "## Marks I keep in view") == marks  # the marks and the outline ride either way
    bare = copy.deepcopy(older)
    bare["metadata"].pop("governing_owner_words")
    assert start(bare)[0] == older_messages  # the carried field is not an input of an older declared start
    # A shared sibling of the same parent reads the same words, marks and outline, byte for byte, beside its memory.
    shared_child = _schedule(root, "Count shelf two")
    _a, b, c, shared_cap = _start(env, shared_child)
    assert section(c, WORDS) == section(changing, WORDS) and shared_cap["memory_view"]["role"] == "child"
    assert section(c, "## Marks I keep in view") == marks and section(c, "## Knowledge outline") == outline
    assert "## My story" in b and "  Alpha began." in b and "## Working sources" in c and "## This room" in c


# --- a file my human sent is staged at a birth: a declared helper and a grandchild open it ---------------

def test_a_declared_helper_and_a_grandchild_born_after_a_carried_screenshot_open_it_at_their_first_round(tmp_path):
    """A row with a file, relayed to the root and drained as its worker drains it (the chat row held, its words
    recorded in the root's corpus), reaches a declared helper's and a grandchild's birth block with the file
    staged on each one's own drive and its line naming the path it opens there: the corpus row wins the identity
    and takes the held twin's upload references, so the scheduled field names the file. A helper born before the
    row has neither the row nor a copy."""
    from ouroboros import dialogue_store
    from ouroboros.chat_uploads import claim_refs, store_upload
    from ouroboros.context import build_llm_messages
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_relay import owner_quote, owner_row_address, write_relay

    env, memory, _rooms = world(tmp_path)
    canonical = memory.drive_root
    root = _root(env, memory, "bound", 1, ASKED)
    earlier = _schedule(root, "Count shelf one")
    _path, ref = store_upload(b"PNG bytes of the shelf", "shelf.png", data_dir=canonical)
    claim_refs([ref], canonical)
    row = {"chat_id": 1, "direction": "in", "ts": "2026-10-04T05:30:00+00:00", "text": "Here is the shelf",
           "client_message_id": "msg-shot", "attachments": [ref]}
    dialogue_store.append(canonical, "chat", row)
    address = owner_row_address(row)
    write_relay(canonical, "bound", relay_id="shot", note="the shelf, for the count", chat_id=1, teller_task_id="tmain",
                quotes=[owner_quote(canonical, row, address=address)], canonical_root=canonical)
    _drain_incoming_messages([{"role": "user", "content": ASKED}], queue.Queue(), canonical, "bound", None, set(),
                             owner_ctx=root)
    assert root._held_owner_messages[address]["attachments"] == [ref]  # the chat row is held with its file
    corpus = [item for item in root._owner_directives if item.get("content") == "Here is the shelf"]
    assert len(corpus) == 1 and "attachments" not in corpus[0]  # its corpus row records the words alone
    declared = _schedule(root, "Count shelf two", input_sources="declared", context="Shelf two holds ten boxes")
    carried = [item for item in declared["metadata"]["governing_owner_words"] if item["text"] == "Here is the shelf"]
    assert len(carried) == 1 and carried[0]["carrier"] == "ctx" and carried[0]["ref"] == "chat 1 / msg-shot"
    assert [item["upload"] for item in carried[0]["attachments"]] == [ref["upload"]]  # one row, with its file
    child = _schedule(root, "Count shelf three")
    grandparent = ToolContext(repo_dir=env.repo_dir, drive_root=pathlib.Path(child["drive_root"]),
                              budget_drive_root=str(canonical), current_task_type="task")
    grandparent.task_id, grandparent.task_depth, grandparent.event_queue = child["id"], child["depth"], queue.Queue()
    grandparent.task_metadata, grandparent.task_contract = dict(child["metadata"]), child["task_contract"]
    _initialize_owner_directives(grandparent, [{"role": "user", "content": child["text"]}])
    grandchild = _schedule(grandparent, "Count the top row of shelf three")

    def words_of(payload):
        own = pathlib.Path(payload["drive_root"])
        child_env = Env(repo_dir=env.repo_dir, drive_root=own, budget_drive_root=canonical)
        messages, _cap = build_llm_messages(env=child_env, memory=Memory(drive_root=own, repo_dir=env.repo_dir),
                                            task={"type": "task", "text": "hi", **payload})
        return section(messages[0]["content"][-1]["text"], WORDS), own / "task_results/artifacts" / payload["id"]

    for payload in (declared, grandchild):
        words, store = words_of(payload)
        assert "\n  Here is the shelf\n" in words
        assert f"read_file(root='artifact_store', path='attachments/{ref['upload']}')" in words
        assert (store / "attachments" / ref["upload"]).read_bytes() == b"PNG bytes of the shelf"
    words, store = words_of(earlier)
    assert "Here is the shelf" not in words and not store.exists()
