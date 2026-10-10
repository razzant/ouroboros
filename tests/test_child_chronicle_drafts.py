"""A delegated child or nanny publishes chronicle pages and parts only as its own drafts: it is not the integrating mind.

Through the real ToolRegistry: both child sets see ``chronicle_write``; a child's page or part
is a helper's draft signed with the child's focus (role, task, route) that acts at once until
the integrating mind's ``decision``; a child's part folds only legacy sections and its own
drafts; a child's note, correction or decision, or its part over the mind's or another
writer's records, is refused ``not_integrator`` and nothing lands in the journal; a child
whose contract withholds ``chronicle_write`` drafts nothing; the root still writes as the mind. The
view and ``memory_read`` name a child's draft by the child and a Light draft by Light. Each
rule is pinned in both directions; no test calls a model or the network.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from ouroboros import chat_chain
from ouroboros import memory_view as mv
from ouroboros.chronicle_store import ChronicleStore, draft_signer
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools.chronicle import _memory_read
from ouroboros.tools.registry import ToolContext, ToolRegistry

CHILD_META = {"delegation_role": "subagent", "parent_task_id": "root0001", "root_task_id": "root0001"}
NANNY_META = {**CHILD_META, "configured_subagent": {"route": {"kind": "agent_session"}}}
LIGHT = {"kind": "helper", "writer": "fallback_page", "route": {"model": "configured-light"},
         "attribution": "helper draft, not lived"}
LEGACY = {"kind": "legacy_helper", "writer": "old_consolidator", "attribution": "retelling by Light, not lived"}
READONLY = "local_readonly_subagent"
ACTING = "acting_subagent"


def ts(n: int) -> str:
    return f"2026-10-01T00:00:{n:02d}+00:00"


def chat(root: pathlib.Path, count: int = 6) -> list:
    rows = [{"chat_id": 1, "direction": "in" if n % 2 else "out", "ts": ts(n), "text": f"words {n}",
             **({} if n % 2 else {"task_id": "root0001"})} for n in range(1, count + 1)]
    path = root / "logs" / "chat.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def addr(row) -> str:
    return chat_chain.format_address(chat_chain.row_address(row))


def journal(root: pathlib.Path) -> str:
    path = root / "memory" / "chronicle" / "records.jsonl"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


@pytest.fixture
def data(tmp_path) -> pathlib.Path:
    """The data root; the repo and an acting child's worktree sit beside it, never inside."""
    for name in ("data", "repo"):
        (tmp_path / name).mkdir()
    return tmp_path / "data"


def registry_for(root: pathlib.Path, task_id: str, meta=None, mode: str = "", monkeypatch=None) -> ToolRegistry:
    fields = {"task_metadata": dict(meta)} if meta is not None else {}
    if mode == READONLY:
        fields["task_constraint"] = TaskConstraint(mode=READONLY, allow_enable=False)
    elif mode == ACTING:
        worktree = root.parent / f"wt-{task_id}"
        worktree.mkdir(exist_ok=True)
        monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
        fields.update(workspace_root=str(worktree), workspace_mode="self_worktree",
                      task_constraint=TaskConstraint(mode=ACTING, surface="self_worktree", write_root=str(worktree)))
    repo = root.parent / "repo"
    registry = ToolRegistry(repo_dir=repo, drive_root=root)
    registry.set_context(ToolContext(repo_dir=repo, drive_root=root, task_id=task_id, current_chat_id=1, **fields))
    return registry


def call(registry: ToolRegistry, **args) -> dict:
    return json.loads(registry.execute("chronicle_write", args))


def status_of(root: pathlib.Path, record_id: str) -> str:
    return next(record.get("status") for record in ChronicleStore(root).room_records("1") if record["id"] == record_id)


@pytest.mark.parametrize("mode", [READONLY, ACTING])
@pytest.mark.parametrize("meta, role", [(CHILD_META, "child"), (NANNY_META, "nanny")])
def test_a_child_or_nanny_page_is_a_draft_signed_with_its_own_focus(data, monkeypatch, mode, meta, role):
    rows = chat(data)
    kid = registry_for(data, "kid00001", meta, mode, monkeypatch)
    assert kid.get_schema_by_name("chronicle_write") is not None
    page = call(kid, kind="page", text="What the child closed.", covers={"from": addr(rows[0]), "to": addr(rows[1])})
    assert page["ok"] and page["kind"] == "page"
    author = ChronicleStore(data).get(page["node_id"])["author"]
    # The host signs the draft: helper kind, the child's own focus, task and route; the model chose none of it.
    assert author["kind"] == "helper" and author["task_id"] == "kid00001" and "route" in author
    assert author["focus"]["role"] == role and author["focus"]["parent_task_id"] == "root0001"
    assert status_of(data, page["node_id"]) == "draft"
    # The other side: the root writes as the mind, and its page is final, never a draft.
    root = registry_for(data, "root0001")
    mine = call(root, kind="page", text="My own page.", covers={"from": addr(rows[2]), "to": addr(rows[3])})
    assert mine["ok"] and ChronicleStore(data).get(mine["node_id"])["author"]["kind"] == "mind"
    assert status_of(data, mine["node_id"]) == "final"


@pytest.mark.parametrize("mode", [READONLY, ACTING])
def test_a_childs_note_correction_and_decision_are_refused_and_nothing_lands(data, monkeypatch, mode):
    rows = chat(data)
    root = registry_for(data, "root0001")
    mine = call(root, kind="page", text="My page.", covers={"from": addr(rows[0]), "to": addr(rows[1])})
    kid = registry_for(data, "kid00001", CHILD_META, mode, monkeypatch)
    draft = call(kid, kind="page", text="The child's draft.", covers={"from": addr(rows[2]), "to": addr(rows[3])})
    assert mine["ok"] and draft["ok"]
    before = journal(data)
    for args in ({"kind": "note", "text": "for later"},
                 {"kind": "correction", "target_id": mine["node_id"], "text": "corrected"},
                 {"kind": "decision", "target_id": draft["node_id"], "accepted": True, "reason": "mine"}):
        refused = call(kid, **args)
        assert refused["ok"] is False and refused["reason"] == "not_integrator"
        assert args["kind"] in refused["detail"] and "Nothing was written" in refused["detail"]
    assert journal(data) == before
    # The other side: the integrating mind writes each of them.
    assert call(root, kind="note", text="for later")["ok"]
    assert call(root, kind="correction", target_id=mine["node_id"], text="corrected")["ok"]
    assert call(root, kind="decision", target_id=draft["node_id"], accepted=True, reason="right")["ok"]
    assert status_of(data, draft["node_id"]) == "accepted"


@pytest.mark.parametrize("mode", [READONLY, ACTING])
def test_a_child_whose_contract_withholds_chronicle_write_drafts_nothing(data, monkeypatch, mode):
    """Without the tool a child cannot draft; the host refuses the call and nothing lands."""
    rows = chat(data)
    page = {"kind": "page", "text": "What the child closed.", "covers": {"from": addr(rows[0]), "to": addr(rows[1])}}
    withheld = {**CHILD_META, "task_contract": {"disabled_tools": ["chronicle_write"]}}
    kid = registry_for(data, "kid00001", withheld, mode, monkeypatch)
    assert kid.get_schema_by_name("chronicle_write") is None
    refused = kid.execute("chronicle_write", page)
    assert '"ok": true' not in refused and "disabled_tools" in refused
    assert journal(data) == ""  # no record, no activation
    # The other side: the same child holding the tool drafts the same page.
    drafted = call(registry_for(data, "kid00001", CHILD_META, mode, monkeypatch), **page)
    assert drafted["ok"] and status_of(data, drafted["node_id"]) == "draft"


@pytest.mark.parametrize("mode", [READONLY, ACTING])
def test_a_childs_account_and_selection_are_refused_and_the_mind_writes_them(data, monkeypatch, mode):
    """An account across rooms and its selection are the integrating mind's: a child's or nanny's is refused
    ``not_integrator`` through the real registry and nothing lands; the root publishes both and the child reads
    the selected account in its story."""
    rows = chat(data)
    root = registry_for(data, "root0001")
    first = call(root, kind="page", text="The first arc.", covers={"from": addr(rows[0]), "to": addr(rows[1])})
    second = call(root, kind="page", text="The second arc.", covers={"from": addr(rows[2]), "to": addr(rows[3])})
    assert first["ok"] and second["ok"]
    before = journal(data)
    for meta in (CHILD_META, NANNY_META):
        kid = registry_for(data, "kid00001", meta, mode, monkeypatch)
        for args in ({"kind": "account", "text": "Both arcs, as I see them.", "sources": [first["node_id"], second["node_id"]]},
                     {"kind": "selection", "target_id": first["node_id"], "replaces": [], "reason": "mine"}):
            refused = call(kid, **args)
            assert refused["ok"] is False and refused["reason"] == "not_integrator", (meta, args)
            assert args["kind"] in refused["detail"] and "Nothing was written" in refused["detail"]
    assert journal(data) == before
    account = call(root, kind="account", text="Both arcs, as I see them.", sources=[first["node_id"], second["node_id"]])
    assert account["ok"] and account["kind"] == "account" and account["sources"] == 2
    chosen = call(root, kind="selection", target_id=account["node_id"], replaces=[first["node_id"]], reason="through it")
    assert chosen["ok"] and chosen["replaces"] == 1
    kid = {"id": "kid00001", "chat_id": 1, "delegation_role": "subagent", "root_task_id": "root0001"}
    story = mv.render_story(mv.capture_memory_view(data, kid, mv.view_spec_for_task(kid, data)))
    assert "Both arcs, as I see them." in story and "The first arc." not in story and "The second arc." in story
    assert "1 story records told through this account (including nested selections); 2026-10-01 00:00 → 2026-10-01 00:00" in story
    assert f"memory_read(node_id='{account['node_id']}')" in story
    reader = registry_for(data, "kid00001", CHILD_META, mode, monkeypatch)
    exact = reader.execute("memory_read", {"node_id": account["node_id"]})
    assert f"replaces 1: {first['node_id']};" in exact and f"revision {first['node_id']} used" in exact


def test_a_childs_refused_note_does_not_activate_the_chronicle(data, monkeypatch):
    kid = registry_for(data, "kid00001", CHILD_META, READONLY, monkeypatch)
    assert call(kid, kind="note", text="nothing")["reason"] == "not_integrator"
    assert not (data / "memory" / "chronicle").exists()
    assert call(registry_for(data, "root0001"), kind="note", text="mine")["ok"]
    assert (data / "memory" / "chronicle" / "records.jsonl").exists()


def folded_into(root: pathlib.Path) -> dict:
    return {record["id"]: record.get("folded_into") for record in ChronicleStore(root).room_records("1")}


def legacy_sections(root: pathlib.Path) -> list:
    """Two adjacent legacy sections of room 1, as the one-time import lays them down."""
    ids = [f"legacy-b{block:02d}-r1" for block in (0, 1)]
    assert ChronicleStore(root).publish([{"id": rid, "kind": "legacy", "room_id": "1", "text": f"old era {n}",
                                          "author": LEGACY, "metadata": {"legacy_type": "era", "legacy_block": n}}
                                         for n, rid in enumerate(ids)]).ok
    return ids


def test_a_childs_part_folds_legacy_and_its_own_drafts_and_the_minds_rejection_unfolds_them(data, monkeypatch):
    rows = chat(data)
    root = registry_for(data, "root0001")
    assert call(root, kind="page", text="the mind's page", covers={"from": addr(rows[0]), "to": addr(rows[1])})["ok"]
    old = legacy_sections(data)
    kid = registry_for(data, "kid00001", CHILD_META, READONLY, monkeypatch)
    part = call(kid, kind="part", text="The old era, folded.", member_ids=old,
                expected_sequence=ChronicleStore(data).room_head("1"))
    assert part["ok"] and part["kind"] == "part"
    assert ChronicleStore(data).get(part["node_id"])["author"]["kind"] == "helper"
    assert status_of(data, part["node_id"]) == "draft"
    assert folded_into(data)[old[0]] == folded_into(data)[old[1]] == part["node_id"]
    # Its own drafts fold the same way.
    mine = [call(kid, kind="page", text=f"my page {n}", covers={"from": addr(rows[n]), "to": addr(rows[n + 1])})
            for n in (2, 4)]
    own = call(kid, kind="part", text="My two pages.", member_ids=[page["node_id"] for page in mine],
               expected_sequence=mine[1]["room_head"])
    assert own["ok"] and status_of(data, own["node_id"]) == "draft"
    assert call(root, kind="decision", target_id=part["node_id"], accepted=False, reason="not my reading")["ok"]
    folded = folded_into(data)
    assert part["node_id"] not in folded and not folded[old[0]] and not folded[old[1]]


def test_a_childs_part_over_the_minds_or_anothers_records_is_refused_and_nothing_lands(data, monkeypatch):
    rows = chat(data, count=8)
    root = registry_for(data, "root0001")
    first = call(root, kind="page", text="first", covers={"from": addr(rows[0]), "to": addr(rows[1])})
    second = call(root, kind="page", text="second", covers={"from": addr(rows[2]), "to": addr(rows[3])})
    other = call(registry_for(data, "kid00002", CHILD_META, READONLY, monkeypatch), kind="page", text="a sibling's",
                 covers={"from": addr(rows[4]), "to": addr(rows[5])})
    kid = registry_for(data, "kid00001", CHILD_META, READONLY, monkeypatch)
    mine = call(kid, kind="page", text="mine", covers={"from": addr(rows[6]), "to": addr(rows[7])})
    assert first["ok"] and second["ok"] and other["ok"] and mine["ok"]
    head, before = ChronicleStore(data).room_head("1"), journal(data)
    for members, foreign in (([first["node_id"], second["node_id"]], [first["node_id"], second["node_id"]]),
                             ([other["node_id"], mine["node_id"]], [other["node_id"]])):
        refused = call(kid, kind="part", text="The child's fold.", member_ids=members, expected_sequence=head)
        assert refused["ok"] is False and refused["reason"] == "not_integrator"
        assert refused["conflict_ids"] == foreign and "Nothing was written" in refused["detail"]
    assert journal(data) == before and not any(folded_into(data).values())
    # The other side: the integrating mind folds its own pages.
    part = call(root, kind="part", text="The mind's fold.", member_ids=[first["node_id"], second["node_id"]],
                expected_sequence=head)
    assert part["ok"] and status_of(data, part["node_id"]) == "final"
    assert folded_into(data)[first["node_id"]] == part["node_id"]


def test_the_view_and_memory_read_name_a_childs_draft_by_the_child_and_lights_by_light(data, monkeypatch):
    rows = chat(data)
    kid = registry_for(data, "kid00001", CHILD_META, READONLY, monkeypatch)
    child_draft = call(kid, kind="page", text="The child's page.", covers={"from": addr(rows[0]), "to": addr(rows[1])})
    assert child_draft["ok"]
    covers = {"room_id": "1", "mode": "range", "rows": [chat_chain.source_row_id(rows[3])], "stream_span": [3, 3]}
    light_draft = ChronicleStore(data).publish_page(room_id="1", text="Light's page.", covers=covers, author=LIGHT)
    assert light_draft.ok
    task = {"id": "turn0001", "chat_id": 1}
    story = mv.render_story(mv.capture_memory_view(data, task, mv.view_spec_for_task(task, data)))
    child_at, light_at = story.index("The child's page."), story.index("Light's page.")
    assert story.count("(draft by a helper (child, task kid00001), not yet accepted or rejected by me)") == 1
    assert story.count("(draft by a helper (Light), not yet accepted or rejected by me)") == 1
    assert story.index("(child, task kid00001)") > child_at and story.index("(Light)") > light_at
    assert call(registry_for(data, "root0001"), kind="decision", target_id=child_draft["node_id"],
                accepted=True, reason="right")["ok"]
    story = mv.render_story(mv.capture_memory_view(data, task, mv.view_spec_for_task(task, data)))
    assert "(drafted by a helper (child, task kid00001), accepted by me)" in story
    listing = _memory_read(ToolContext(repo_dir=data, drive_root=data, task_id="root0001", current_chat_id=1))
    child_line = next(line for line in listing.split("\n") if line.startswith(f"[page {child_draft['node_id']}"))
    light_line = next(line for line in listing.split("\n") if line.startswith(f"[page {light_draft.record['id']}"))
    assert "; helper (child, task kid00001); accepted;" in child_line
    assert "; helper (helper draft, not lived); draft;" in light_line and "child" not in light_line


def test_draft_signer_names_a_delegated_focus_and_otherwise_light():
    child = {"kind": "helper", "task_id": "kid00001", "focus": {"role": "child", "task_id": "kid00001"}}
    nanny = {"kind": "helper", "task_id": "nan00001", "focus": {"role": "nanny", "task_id": "nan00001"}}
    assert draft_signer(child) == "child, task kid00001" and draft_signer(nanny) == "nanny, task nan00001"
    assert draft_signer(LIGHT) == draft_signer({"kind": "helper"}) == draft_signer(None) == "Light"
