"""The memory view's capture and the one activation of the chronicle (``ouroboros.memory_view``).

A capture activates the chronicle exactly once (the one legacy import, no model call);
until the import has completed the view still renders, names the reason and reads no
chat chain. The first capture on an empty install creates the chronicle, the second
writes nothing. The module imports no model client and computes no "folded" verdict of
its own. Each rule is pinned in both directions.
"""
from __future__ import annotations

import ast
import hashlib
import pathlib

import pytest

from ouroboros import chat_chain
from ouroboros import memory_view as mv
from ouroboros.chronicle_store import ChronicleStore
from tests import _memory_inventory_shared as shared

REPO = pathlib.Path(__file__).resolve().parents[1]
MAIN_TASK = {"id": "turn0001", "chat_id": 1}


def _files(root: pathlib.Path) -> dict:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def _count_activations(monkeypatch) -> list:
    calls = []
    real = ChronicleStore.ensure_activated

    def counting(self, **kwargs):
        calls.append(kwargs)
        return real(self, **kwargs)

    monkeypatch.setattr(ChronicleStore, "ensure_activated", counting)
    return calls


def test_a_capture_activates_the_chronicle_exactly_once_and_imports_the_legacy_memory(tmp_path, monkeypatch):
    rooms = shared.world(tmp_path, activate=False)
    calls = _count_activations(monkeypatch)
    spec = mv.view_spec_for_task(MAIN_TASK, tmp_path)
    assert calls == []  # choosing the spec activates nothing
    snapshot = mv.capture_memory_view(tmp_path, MAIN_TASK, spec)
    assert len(calls) == 1 and calls[0] == {}  # one non-waiting activation per capture
    assert snapshot.active and snapshot.store_status == {"state": "active"}
    assert snapshot.frontier == {"status": "exact", "pos": 10}
    assert ChronicleStore(tmp_path).activation()["kind"] == "activation"
    # The next capture finds the receipt (the fast path) and still activates once.
    mv.capture_memory_view(tmp_path, {"id": "proj0001", "chat_id": rooms["alpha"]}, spec)
    assert len(calls) == 2


@pytest.mark.parametrize("receipt", [
    {"kind": "import_pending", "reason": "legacy_memory_lock_busy"},
    {"kind": "import_refused", "reason": "invalid", "detail": "x", "conflict_ids": ["a"]},
])
def test_an_import_not_completed_leaves_a_view_with_its_reason_and_reads_no_chain(tmp_path, monkeypatch, receipt):
    shared.world(tmp_path, activate=False)
    spec = mv.view_spec_for_task(MAIN_TASK, tmp_path)
    monkeypatch.setattr(ChronicleStore, "ensure_activated", lambda self, **kw: dict(receipt))

    def no_chain(*_a, **_k):
        raise AssertionError("the chain is not read before the import has completed")

    monkeypatch.setattr(chat_chain, "iter_rows", no_chain)
    monkeypatch.setattr(chat_chain, "generation_signatures", no_chain)
    snapshot = mv.capture_memory_view(tmp_path, MAIN_TASK, spec)
    assert not snapshot.active
    assert snapshot.store_status == {"state": receipt["kind"], "reason": receipt["reason"]}
    assert snapshot.story == () and snapshot.live_rooms == () and snapshot.marks == ()
    assert snapshot.room == {"room_id": "1", "label": "Main"} and snapshot.frontier == {}


@pytest.mark.parametrize("error", [ValueError("chronicle authority shortened"), TimeoutError("lock"), OSError("io")])
def test_a_journal_that_cannot_be_opened_still_leaves_a_view(tmp_path, monkeypatch, error):
    shared.world(tmp_path)

    def broken(self, **_kw):
        raise error

    monkeypatch.setattr(ChronicleStore, "ensure_activated", broken)
    snapshot = mv.capture_memory_view(tmp_path, MAIN_TASK, mv.ROLE_DEFAULTS["integrator"])
    assert snapshot.store_status["state"] == "journal_unreadable"
    assert type(error).__name__ in snapshot.store_status["reason"]
    assert not snapshot.active and snapshot.story == ()


def test_the_first_capture_on_an_empty_install_creates_the_chronicle_and_the_second_writes_nothing(tmp_path):
    spec = mv.view_spec_for_task(MAIN_TASK, tmp_path)
    assert not (tmp_path / "memory" / "chronicle").exists()
    first = mv.capture_memory_view(tmp_path, MAIN_TASK, spec)
    assert first.active and (tmp_path / "memory" / "chronicle" / "records.jsonl").is_file()
    before = _files(tmp_path)
    second = mv.capture_memory_view(tmp_path, MAIN_TASK, spec)
    assert second == first and _files(tmp_path) == before


def test_the_snapshot_round_trips_through_its_canonical_json(tmp_path):
    shared.world(tmp_path)
    task = {"id": "kid00001", "chat_id": 1, "delegation_role": "subagent",
            "metadata": {"governing_owner_words": [{"text": "Do the inventory", "source": "initial_user"}]}}
    spec = mv.view_spec_for_task(task, tmp_path)
    snapshot = mv.capture_memory_view(tmp_path, task, spec)
    text = mv.snapshot_json(snapshot)
    assert mv.snapshot_from_json(text) == snapshot and mv.snapshot_json(mv.snapshot_from_json(text)) == text
    assert "Do the inventory" in snapshot.owner_words  # the helper's owner words are captured
    no_words = mv.capture_memory_view(tmp_path, task, mv.ROLE_DEFAULTS["integrator"])
    assert no_words.owner_words == ""


def _imports(name: str):
    tree = ast.parse((REPO / "ouroboros" / name).read_text(encoding="utf-8"))
    top, nested = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            (top if node in tree.body else nested).update(names)
    return tree, top, nested


def test_the_view_imports_no_model_client_or_retired_memory_machinery_and_crosses_domains_lazily():
    tree, top, nested = _imports("memory_view.py")
    imported = top | nested
    forbidden = ("ouroboros.llm", "ouroboros.consolidator", "ouroboros.room_consolidation", "ouroboros.chronicle_view")
    assert not [name for name in imported if name.startswith(forbidden)], imported
    # D17 (owner words, projects), D06 (own room) and D07 (the nanny fact) are reached only inside functions.
    lazy = {"ouroboros.owner_words", "ouroboros.dialogue_evidence", "ouroboros.subagent_dispatch_notes"}
    assert lazy <= nested and not lazy & top
    assert not {"ouroboros.project_dialogue", "ouroboros.projects_registry"} & top
    # The guard has teeth: the same check flags a module that imports the model client.
    probe = ast.parse("from ouroboros.llm import LLMClient\n")
    assert any(isinstance(node, ast.ImportFrom) and node.module.startswith(forbidden) for node in ast.walk(probe))


def test_the_floor_imports_no_model_client_and_reads_nothing_of_its_own():
    tree, top, nested = _imports("memory_floor.py")
    imported = top | nested
    assert not [name for name in imported if name.startswith(("ouroboros.llm", "ouroboros.consolidator"))], imported
    assert {"ouroboros.chronicle_store", "ouroboros.memory_inventory"}.isdisjoint(imported)
    chain = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "ouroboros.chat_chain"]
    assert [alias.name for node in chain for alias in node.names] == ["parse_address"]  # a parser, not a reader
    assert "ouroboros" in top  # memory_view and context_budget, the view's renderer and the one budget frame


def test_the_view_takes_open_and_folded_from_the_inventory_and_reads_no_row_stream_of_its_own():
    tree, _top, _nested = _imports("memory_view.py")
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert {"legacy_units", "legacy_progress"} <= names  # the one folded rule, consumed
    own = {"iter_rows", "iter_room_rows", "_stream_rows", "_exact_range", "_legacy_row_sets", "_legacy_stream"}
    assert not own & names, own & names
    # The guard has teeth: a module that walks the chain itself is flagged.
    probe = ast.parse("from ouroboros import chat_chain\nrows = chat_chain.iter_rows(root)\n")
    assert own & {node.attr for node in ast.walk(probe) if isinstance(node, ast.Attribute)}


def test_a_journal_that_fails_while_the_story_is_read_still_leaves_a_view(tmp_path, monkeypatch):
    from ouroboros import memory_inventory

    shared.world(tmp_path)

    def broken(*_a, **_k):
        raise ValueError("chronicle authority shortened")

    monkeypatch.setattr(memory_inventory, "legacy_units", broken)
    snapshot = mv.capture_memory_view(tmp_path, MAIN_TASK, mv.ROLE_DEFAULTS["integrator"])
    assert snapshot.store_status == {"state": "journal_unreadable",
                                     "reason": "ValueError: chronicle authority shortened"}
    assert mv.render_story(snapshot).startswith("## My story — unavailable now (ValueError: chronicle authority")
    monkeypatch.undo()
    healthy = mv.capture_memory_view(tmp_path, MAIN_TASK, mv.ROLE_DEFAULTS["integrator"])
    assert healthy.active and mv.render_story(healthy).startswith("## My story\n")


def test_the_answer_path_takes_no_model_client_and_runs_no_paid_memory_upkeep():
    """Rendering my memory is free, never a paid pass: the builder accepts no model client
    or fit callback, and the assembler imports no model client or memory writer."""
    import inspect

    from ouroboros.context import build_llm_messages

    params = inspect.signature(build_llm_messages).parameters
    assert {"llm", "fit_candidate"}.isdisjoint(params) and "tool_schemas" in params
    _tree, top, nested = _imports("context.py")
    forbidden = ("ouroboros.llm", "ouroboros.consolidator", "ouroboros.room_consolidation", "ouroboros.chronicle_view")
    assert not [name for name in top | nested if name.startswith(forbidden)], top | nested
    # The guard has teeth: the retired upkeep's own import is flagged.
    probe = ast.parse("def f():\n    from ouroboros.consolidator import maintain_memory_pressure\n")
    assert [node.module for node in ast.walk(probe) if isinstance(node, ast.ImportFrom)][0].startswith(forbidden)
