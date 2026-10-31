"""#918: scratchpad blocks stay until the integrating mind folds them (owner, 29 Oct 2026: «А — блоки, свёртку принимаю я»).

Real consumers end to end: the ``update_scratchpad`` tool through ``ToolRegistry`` dispatch,
the canonical ``memory/scratchpad_blocks.json`` and ``memory/scratchpad_journal.jsonl`` owned by
``Memory``, and the next task's actual first request (``context.build_llm_messages``). Nothing
leaves the active list by count, size, age or a post-task Light pass; a fold replaces exactly the
blocks it names (their sha256 as the next context shows them), after the exact originals are
journaled, and a failed, empty, stale or ambiguous fold leaves every active block in place.
Every reader a fold emits opens those originals from the canonical drive and from a forked
execution drive alike, within the follower's own restrictions.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

import pytest

from ouroboros.memory import Memory
from ouroboros.tools.registry import ToolContext, ToolRegistry

_DIGEST = re.compile(r"^### \[[^\]]*\] sha256 ([0-9a-f]{64})$", re.M)


@pytest.fixture
def install(tmp_path):
    from tests.test_cache_optimization import _make_env_and_memory

    env, memory = _make_env_and_memory(tmp_path)
    # The shared fixture seeds a flat pre-block scratchpad; this installation keeps blocks.
    memory.scratchpad_path().unlink()
    memory.ensure_files()
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
    registry.set_context(ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id="root-918"))
    return env, memory, registry


def _tool(registry, **args):
    return registry.execute_result("update_scratchpad", args)


def _blocks(memory):
    return json.loads(memory.scratchpad_blocks_path().read_text(encoding="utf-8"))


def _journal(memory, kind=None):
    path = memory.journal_path()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    return [row for row in rows if kind is None or row.get("type") == kind]


def _next_request(env, memory) -> str:
    """The changing tail of the next task's real first request, where the scratchpad lives."""
    from ouroboros.context import build_llm_messages

    messages, _cap = build_llm_messages(env=env, memory=memory, task={"id": "next-918", "type": "task", "text": "hi"})
    return messages[0]["content"][-1]["text"]


def _scratchpad_section(text: str) -> str:
    start = text.index("## Scratchpad (from")
    following = [match.start() for match in re.finditer(r"\n## (?!Scratchpad \(working memory)", text[start + 1:])]
    return text[start:start + 1 + following[0]] if following else text[start:]


def _digests_in_view(env, memory):
    return _DIGEST.findall(_scratchpad_section(_next_request(env, memory)))


def _canonical_sha(block) -> str:
    return hashlib.sha256(json.dumps(block, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def scratchpad_block_sha256(block) -> str:
    from ouroboros.memory import scratchpad_block_sha256 as owner

    return owner(block)


def _note(index: int, size: int = 0) -> str:
    return f"NOTE-{index:02d} " + ("n" * size if size else f"working thought number {index}")


# --- nothing rotates out ------------------------------------------------------------------


def test_twelve_small_appends_all_stay_in_storage_and_in_the_next_request(install):
    env, memory, registry = install
    for index in range(12):
        result = _tool(registry, content=_note(index))
        assert result.status == "ok", result.text
    blocks = _blocks(memory)
    assert [block["content"] for block in blocks] == [_note(index) for index in range(12)]
    assert _journal(memory, "block_evicted") == []
    section = _scratchpad_section(_next_request(env, memory))
    assert all(_note(index) in section for index in range(12))
    assert "12 blocks" in section and "/10" not in section
    assert "PARTIAL" not in section and "budget gap" not in section


def test_content_above_sixty_thousand_characters_stays_active(install):
    env, memory, registry = install
    for index in range(3):
        assert _tool(registry, content=_note(index, 25_000)).status == "ok"
    assert [len(block["content"]) for block in _blocks(memory)] == [len(_note(index, 25_000)) for index in range(3)]
    assert _journal(memory, "block_evicted") == []


def test_a_section_above_ninety_thousand_characters_reaches_the_next_request_whole(install):
    env, memory, registry = install
    for index in range(4):
        assert _tool(registry, content=_note(index, 25_000)).status == "ok"
    assert len(memory.load_scratchpad()) > 90_000
    section = _scratchpad_section(_next_request(env, memory))
    assert all(_note(index, 25_000) in section for index in range(4))
    assert "budget gap" not in section and "PARTIAL" not in section and "omitted" not in section
    assert "already loaded" in section.split("\n", 1)[0]


def test_each_block_shows_the_sha256_of_its_exact_stored_json(install):
    env, memory, registry = install
    for index in range(3):
        _tool(registry, content=_note(index))
    stored = _blocks(memory)
    assert [scratchpad_block_sha256(block) for block in stored] == [_canonical_sha(block) for block in stored]
    # Newest first, as the model has always read its working memory.
    assert _digests_in_view(env, memory) == [_canonical_sha(block) for block in reversed(stored)]


def test_the_tool_schema_offers_the_fold_and_promises_no_eviction(install):
    _env, _memory, registry = install
    schema = registry.get_schema_by_name("update_scratchpad")
    function = schema.get("function", schema)
    properties = function["parameters"]["properties"]
    assert properties["replaces"]["type"] == "array" and properties["replaces"]["items"]["type"] == "string"
    assert properties["draft_source"]["type"] == "string"
    assert function["parameters"]["required"] == ["content"]
    description = function["description"].lower()
    assert "evict" not in description and "10 blocks" not in description and "60000" not in description
    assert "replaces" in description


# --- the fold ----------------------------------------------------------------------------


def test_a_fold_replaces_exactly_the_named_blocks_after_journaling_their_exact_originals(install):
    env, memory, registry = install
    for index in range(4):
        _tool(registry, content=_note(index))
    before = _blocks(memory)
    newest_first = _digests_in_view(env, memory)
    chosen = [newest_first[-1], newest_first[-2]]  # the two oldest, as the mind read them

    result = _tool(registry, content="FOLD-A: notes 0 and 1 mean one decision still open.", replaces=chosen)

    assert result.status == "ok", result.text
    after = _blocks(memory)
    assert after[1:] == before[2:]  # unnamed blocks: same bytes, same order
    fold = after[0]
    assert fold["source"] == "fold" and fold["content"].startswith("FOLD-A")
    meta = fold["metadata"]
    assert meta["fold"]["replaces"] == chosen
    assert meta["fold"]["covers"] == {"from": before[0]["ts"], "to": before[1]["ts"], "blocks": 2}
    assert meta["focus"]["task_id"] == "root-918" and "model" not in json.dumps(meta)
    entry_id = meta["source_ref"]["entry_id"]
    [preserved] = _journal(memory, "fold_sources_preserved")
    [completed] = _journal(memory, "blocks_folded")
    assert preserved["entry_id"] == completed["entry_id"] == entry_id
    assert preserved["source_blocks"] == before[:2]  # exact originals, metadata included
    assert preserved["source_sha256"] == chosen
    kinds = [row["type"] for row in _journal(memory)]
    assert kinds.index("fold_sources_preserved") < kinds.index("blocks_folded")
    assert completed["content_len"] == sum(len(block["content"]) for block in after)
    assert entry_id in result.text and "2 blocks" in result.text

    section = _scratchpad_section(_next_request(env, memory))
    assert "FOLD-A" in section and _note(0) not in section and _note(1) not in section
    assert _note(2) in section and _note(3) in section
    assert f"entry_id={entry_id}" in section


def test_the_preservation_entry_id_names_one_event_not_only_its_sources(install, monkeypatch):
    env, memory, registry = install
    for index in range(2):
        _tool(registry, content=_note(index))
    chosen = _digests_in_view(env, memory)
    from ouroboros import memory as memory_module

    real_write, attempts = memory_module.atomic_write_json, []

    def fail_first_canonical_write(*args, **kwargs):
        attempts.append(args[0])
        if len(attempts) == 1:
            raise OSError("disk full")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(memory_module, "atomic_write_json", fail_first_canonical_write)
    failed = _tool(registry, content="FOLD attempt one, different words.", replaces=chosen)
    assert failed.status != "ok" and "disk full" in failed.text
    retried = _tool(registry, content="FOLD attempt two, the accepted words.", replaces=chosen)
    assert retried.status == "ok", retried.text
    preserved = _journal(memory, "fold_sources_preserved")
    assert len(preserved) == 2 and preserved[0]["source_blocks"] == preserved[1]["source_blocks"]
    assert preserved[0]["entry_id"] != preserved[1]["entry_id"]
    assert _blocks(memory)[0]["metadata"]["source_ref"]["entry_id"] == preserved[1]["entry_id"]
    [failure] = _journal(memory, "block_fold_failed")
    assert failure["entry_id"] == preserved[0]["entry_id"] and "disk full" in failure["error"]


def test_a_fold_over_an_earlier_fold_keeps_every_original_resolvable(install):
    env, memory, registry = install
    for index in range(4):
        _tool(registry, content=_note(index))
    originals = _blocks(memory)
    newest_first = _digests_in_view(env, memory)
    assert _tool(registry, content="FOLD-1 carries notes 0 and 1.", replaces=newest_first[-2:][::-1]).status == "ok"
    _tool(registry, content=_note(4))
    first_fold = _blocks(memory)[0]
    view = _digests_in_view(env, memory)
    fold_sha = scratchpad_block_sha256(first_fold)
    assert fold_sha in view
    note2_sha = scratchpad_block_sha256(originals[2])

    result = _tool(registry, content="FOLD-2 carries FOLD-1 and note 2 forward.", replaces=[fold_sha, note2_sha])

    assert result.status == "ok", result.text
    after = _blocks(memory)
    assert [block["content"] for block in after] == [
        "FOLD-2 carries FOLD-1 and note 2 forward.", _note(3), _note(4)]
    # Resolve the chain through the journal alone: FOLD-2 -> FOLD-1 (+note 2) -> notes 0 and 1.
    rows = {row["entry_id"]: row for row in _journal(memory, "fold_sources_preserved")}
    second = rows[after[0]["metadata"]["source_ref"]["entry_id"]]
    assert second["source_blocks"] == [first_fold, originals[2]]
    first = rows[second["source_blocks"][0]["metadata"]["source_ref"]["entry_id"]]
    assert first["source_blocks"] == originals[:2]


def test_a_block_appended_after_the_mind_read_its_context_survives_the_fold(install):
    env, memory, registry = install
    for index in range(3):
        _tool(registry, content=_note(index))
    chosen = _digests_in_view(env, memory)[-2:]  # read by the folding focus
    concurrent = Memory(memory.drive_root).append_scratchpad_block("CONCURRENT from another room", source="task")

    assert _tool(registry, content="FOLD of notes 0 and 1.", replaces=chosen).status == "ok"

    after = _blocks(memory)
    assert [block["content"] for block in after] == ["FOLD of notes 0 and 1.", _note(2), "CONCURRENT from another room"]
    assert after[-1] == concurrent


@pytest.mark.parametrize("change_same_source", [False, True])
def test_a_concurrent_change_or_append_is_never_lost(install, change_same_source):
    """The retired Light pass's prefix check, now at the fold: a block another writer changed after
    the mind read it is a conflict that keeps the newer bytes; an appended block survives the fold."""
    env, memory, registry = install
    original = {"ts": "2026-10-01T10:00:00+00:00", "source": "task", "content": "old " * 2000}
    memory.mutate_scratchpad_blocks(lambda _rows: [original])
    [read] = _digests_in_view(env, memory)
    newer = {**original, "content": "new concurrent content"}
    appended = {"ts": "2026-10-01T10:05:00+00:00", "source": "task", "content": "A new episode"}
    memory.mutate_scratchpad_blocks(lambda rows: [newer] if change_same_source else rows + [appended])

    result = _tool(registry, content="FOLD of what the mind read earlier.", replaces=[read])

    if change_same_source:
        assert result.status != "ok" and "SCRATCHPAD_FOLD_CONFLICT" in result.text
        assert _blocks(memory) == [newer]
    else:
        assert result.status == "ok", result.text
        assert [block["content"] for block in _blocks(memory)] == ["FOLD of what the mind read earlier.", "A new episode"]
        assert _blocks(memory)[1] == appended


def test_a_stale_or_unknown_sha256_is_a_conflict_and_changes_nothing(install):
    env, memory, registry = install
    for index in range(3):
        _tool(registry, content=_note(index))
    oldest = _digests_in_view(env, memory)[-1]
    assert _tool(registry, content="FOLD by the first focus.", replaces=[oldest]).status == "ok"
    before_bytes = memory.scratchpad_blocks_path().read_bytes()
    preserved = len(_journal(memory, "fold_sources_preserved"))

    for stale in ([oldest], ["0" * 64], [oldest[:12]]):
        refused = _tool(registry, content="FOLD by a second focus that read an older view.", replaces=stale)
        assert refused.status != "ok" and "SCRATCHPAD_FOLD_CONFLICT" in refused.text
        assert memory.scratchpad_blocks_path().read_bytes() == before_bytes
    assert len(_journal(memory, "fold_sources_preserved")) == preserved
    refusals = _journal(memory, "block_fold_refused")
    assert len(refusals) == 3 and refusals[0]["fold_block"]["content"].startswith("FOLD by a second focus")
    # The refusal names what is active now, so the mind can read and choose again.
    assert all(digest in refused.text for digest in _digests_in_view(env, memory))


def test_identical_blocks_fold_only_when_every_copy_is_selected(install):
    env, memory, registry = install
    twin = {"ts": "2026-10-01T10:00:00+00:00", "source": "task", "content": "TWIN legacy note"}
    memory.mutate_scratchpad_blocks(lambda _rows: [dict(twin), dict(twin), {**twin, "content": "OTHER"}])
    digest = scratchpad_block_sha256(twin)
    before_bytes = memory.scratchpad_blocks_path().read_bytes()

    ambiguous = _tool(registry, content="FOLD of one twin is ambiguous.", replaces=[digest])
    assert ambiguous.status != "ok" and "2 active blocks" in ambiguous.text
    assert memory.scratchpad_blocks_path().read_bytes() == before_bytes

    both = _tool(registry, content="FOLD of both twins, named twice.", replaces=[digest, digest])
    assert both.status == "ok", both.text
    assert [block["content"] for block in _blocks(memory)] == ["FOLD of both twins, named twice.", "OTHER"]
    [preserved] = _journal(memory, "fold_sources_preserved")
    assert preserved["source_blocks"] == [twin, twin]


def test_an_empty_fold_changes_nothing(install):
    env, memory, registry = install
    for index in range(2):
        _tool(registry, content=_note(index))
    before_bytes = memory.scratchpad_blocks_path().read_bytes()
    chosen = _digests_in_view(env, memory)
    for content in ("", "   ", "short"):
        refused = _tool(registry, content=content, replaces=chosen)
        assert refused.status != "ok" and refused.code == "TOOL_ARG_ERROR"
    assert memory.scratchpad_blocks_path().read_bytes() == before_bytes
    assert _journal(memory, "fold_sources_preserved") == []


def test_a_helper_draft_is_accepted_through_the_same_fold_with_its_source(install):
    env, memory, registry = install
    for index in range(2):
        _tool(registry, content=_note(index))
    chosen = _digests_in_view(env, memory)
    result = _tool(registry, content="FOLD drafted by a helper, accepted as written.", replaces=chosen,
                   draft_source="helper task c0ffee: Light draft of notes 0 and 1")
    assert result.status == "ok", result.text
    fold = _blocks(memory)[0]
    assert fold["metadata"]["fold"]["draft_source"] == "helper task c0ffee: Light draft of notes 0 and 1"
    assert fold["metadata"]["focus"]["task_id"] == "root-918"  # the accepting focus, not the helper
    section = _scratchpad_section(_next_request(env, memory))
    assert "helper draft: helper task c0ffee" in section


# --- the exact originals open from every root --------------------------------------------

_CALL = re.compile(r"read_file\([^)]*\)")


@pytest.fixture(params=["light", "advanced", "pro", "cyber_pro"])
def roots(tmp_path, monkeypatch, request, provider_catalog_offline):
    """The canonical drive and a supported forked execution drive (a headless task's) below it;
    the context builds stay local (no live provider catalog fetch)."""
    from tests.test_cache_optimization import _make_env_and_memory

    env, memory = _make_env_and_memory(tmp_path)
    memory.scratchpad_path().unlink()
    memory.ensure_files()
    fork = env.drive_root / "state" / "headless_tasks" / "fold-reader" / "data"
    fork.mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", request.param)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path / "elsewhere"))
    monkeypatch.setattr("ouroboros.config.DATA_DIR", env.drive_root)
    return env, memory, fork


def _rooted(env, drive_root, profile="root", **fields):
    """A focus executing on ``drive_root`` whose canonical root is ``env.drive_root``."""
    meta = {"budget_drive_root": str(env.drive_root), "delegation_role": "root", "root_task_id": "fold-reader"}
    if profile != "root":
        from ouroboros.contracts.task_constraint import TaskConstraint

        work = env.drive_root.parent / "work"
        work.mkdir(exist_ok=True)
        meta.update(delegation_role="subagent", parent_task_id="parent", root_task_id="parent")
        fields.update(task_constraint=TaskConstraint(mode=profile, surface="external_workspace", write_root=str(work)),
                      workspace_root=work, workspace_mode="external", project_id="p1")
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=drive_root, task_id="fold-reader",
                      budget_drive_root=str(env.drive_root), task_metadata=meta, **fields)
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=drive_root)
    registry.set_context(ctx)
    return ctx, registry


def _fold_from(registry, memory, label):
    for index in range(3):
        assert registry.execute_result("update_scratchpad", {"content": _note(index)}).status == "ok"
    originals, accepted = _blocks(memory)[-3:-1], f"FOLD of notes 0 and 1 accepted on the {label} drive."
    result = registry.execute_result("update_scratchpad", {
        "content": accepted, "replaces": [scratchpad_block_sha256(block) for block in originals]})
    assert result.status == "ok", result.text
    [fold] = [block for block in _blocks(memory) if block["content"] == accepted]
    return originals, fold, result.text


def _reader_beside(text, entry_id):
    """The one reader on the line that names ``entry_id``: its call, or the plain read of the
    address a text composed without a focus names (``canonical_runtime_reader`` with no context)."""
    [line] = [line for line in text.splitlines() if entry_id in line]
    calls = _CALL.findall(line)
    if calls:
        [call] = calls
        return call
    [address] = re.findall(r"(\S+) \(no tool context here", line)
    return f"read_file(path={address!r})"


def _opened(text, entry_id):
    """The source blocks the opened journal preserved for ``entry_id``; None when no journal row was read."""
    rows = []
    for line in text.splitlines():
        body = line.split("\t", 1)[-1]  # read_file numbers each line
        if body.startswith("{"):
            try:
                rows.append(json.loads(body))
            except ValueError:
                continue
    return next((row["source_blocks"] for row in rows
                 if row.get("type") == "fold_sources_preserved" and row.get("entry_id") == entry_id), None)


def _follow(registry, read):
    return registry.execute(read["tool"], dict(read["arguments"]))


def _context_section(env, ctx):
    """The scratchpad section of this focus's own next first request."""
    from ouroboros.agent import Env
    from ouroboros.context import build_llm_messages

    drive = pathlib.Path(ctx.drive_root)
    messages, _cap = build_llm_messages(
        env=Env(repo_dir=env.repo_dir, drive_root=drive, budget_drive_root=env.drive_root),
        memory=Memory(drive_root=drive, repo_dir=env.repo_dir),
        task={"id": "fold-reader", "type": "task", "text": "hi", "budget_drive_root": str(env.drive_root)}, ctx=ctx)
    return _scratchpad_section(messages[0]["content"][-1]["text"])


@pytest.mark.parametrize("folding", ["canonical", "forked"])
def test_every_emitted_reader_opens_the_exact_originals_from_the_canonical_and_a_forked_root(roots, folding):
    """Astra's #918 reproduction: a fold accepted on a forked execution drive emitted a relative
    ``runtime_data`` reader, and following it there returned DATA_NOT_YET_CREATED. The stored
    ``source_ref``, the tool's success text, the next context's line and the persisted projection
    each open the canonical journal's exact originals through real dispatch from either root."""
    from tests.test_authored_memory import _run

    env, memory, fork = roots
    actors = {"canonical": _rooted(env, env.drive_root), "forked": _rooted(env, fork)}
    originals, fold, said = _fold_from(actors[folding][1], memory, folding)
    entry_id = fold["metadata"]["source_ref"]["entry_id"]
    read = fold["metadata"]["source_ref"]["read"]
    projection = memory.scratchpad_path().read_text(encoding="utf-8")
    [preserved] = _journal(memory, "fold_sources_preserved")
    assert preserved["source_blocks"] == originals and preserved["fold_block"] == fold
    assert preserved["source_ref"] == fold["metadata"]["source_ref"]  # one recipe, journaled and stored
    assert not (fork / "memory" / "scratchpad_journal.jsonl").exists()  # the fork holds no copy to fall back on

    own_ctx, _own = actors[folding]
    assert _opened(_run(own_ctx, _reader_beside(said, entry_id)), entry_id) == originals
    sections = {}
    for name, (ctx, registry) in actors.items():
        assert _opened(_follow(registry, read), entry_id) == originals, name
        assert _opened(_run(ctx, _reader_beside(projection, entry_id)), entry_id) == originals, name
        sections[name] = _context_section(env, ctx)
        assert _opened(_run(ctx, _reader_beside(sections[name], entry_id)), entry_id) == originals, name
    # One canonical journal: the stored recipe names its absolute path; a focus is shown the existing
    # canonical reader, relative on the canonical drive and absolute from a fork.
    assert read == {"tool": "read_file", "arguments": {
        "root": "runtime_data", "path": str(memory.journal_path()), "start_line": 1}}
    shown = {"canonical": "read_file(root='runtime_data', path='memory/scratchpad_journal.jsonl')",
             "forked": f"read_file(root='runtime_data', path={str(memory.journal_path())!r})"}
    assert _reader_beside(said, entry_id) == shown[folding]
    assert {name: _reader_beside(section, entry_id) for name, section in sections.items()} == shown
    # The boundary the qualified readers cross: a relative runtime path names the reader's own drive.
    _ctx, forked = actors["forked"]
    assert _opened(forked.execute("read_file", {"root": "runtime_data", "path": "memory/scratchpad_journal.jsonl"}),
                   entry_id) is None


def test_a_qualified_reader_is_an_address_never_a_grant(roots):
    """Dispatch keeps every follower's own restrictions: read-only children read and never write, a
    disabled ``read_file`` and a protected journal refuse, and a Presence ceiling stays bound to the
    physical runtime root it selected."""
    from tests.frozen_presence_capabilities import PresenceResourceTarget, PresenceToolTarget
    from tests.test_authored_memory import _run, _speaker

    env, memory, fork = roots
    originals, fold, _said = _fold_from(_rooted(env, fork)[1], memory, "forked")
    entry_id = fold["metadata"]["source_ref"]["entry_id"]
    read = fold["metadata"]["source_ref"]["read"]
    journal = memory.journal_path()
    before = journal.read_bytes()
    target = {key: value for key, value in read["arguments"].items() if key in ("root", "path")}

    for profile in ("local_readonly_subagent", "acting_subagent"):
        _ctx, child = _rooted(env, fork, profile)
        assert _opened(_follow(child, read), entry_id) == originals, profile
        for name, payload in (("write_file", {"content": "changed"}),
                              ("edit_text", {"old_str": entry_id, "new_str": "changed"})):
            assert child.execute_result(name, {**target, **payload}).status != "ok", (profile, name)
    _ctx, disabled = _rooted(env, fork, task_contract={"disabled_tools": ["read_file"]})
    assert "RESOURCE_CONSTRAINT_BLOCKED" in _follow(disabled, read)
    _ctx, protected = _rooted(env, fork, task_contract={"resource_policy": {"protected_artifacts": [
        {"id": "scratchpad-journal", "paths": [str(journal)], "allow": ["execute"]}]}})
    assert "RESOURCE_POLICY_BLOCKED" in _follow(protected, read)

    call = "read_file(" + ", ".join(f"{key}={value!r}" for key, value in read["arguments"].items()) + ")"
    assert _run(_speaker(env, "speaker"), call).startswith("⚠️ PRESENCE_CAPABILITY_BLOCKED")
    for drive, opens in ((env.drive_root, True), (fork, False)):
        speaker = _speaker(env, "speaker", PresenceToolTarget("builtin", "read_file"),
                           PresenceResourceTarget("runtime_data", ("read",), "memory"))
        speaker.drive_root, speaker.budget_drive_root = drive, str(env.drive_root)
        opened = _run(speaker, call)
        assert (_opened(opened, entry_id) == originals) is opens, opened
        assert opens or opened.startswith("⚠️")
    assert journal.read_bytes() == before


def test_a_historical_relative_reference_keeps_its_bytes_and_is_shown_a_qualified_reader(roots):
    """Old sources are never rewritten: a fold stored before this repair keeps its relative
    ``source_ref`` (so its sha256) and the journal keeps its rows through later folds, while the
    context shows a forked focus the reader that opens that fold's canonical originals."""
    from ouroboros.utils import append_jsonl
    from tests.test_authored_memory import _run

    env, memory, fork = roots
    forked_ctx, forked = _rooted(env, fork)
    source = {"ts": "2026-10-29T10:00:00+00:00", "source": "task", "content": "LEGACY original note"}
    entry_id = "scratchpad-fold:" + "a" * 64
    relative = {"read": {"tool": "read_file", "arguments": {
        "root": "runtime_data", "path": "memory/scratchpad_journal.jsonl", "start_line": 1}}, "entry_id": entry_id}
    legacy = {"ts": "2026-10-29T10:05:00+00:00", "source": "fold", "content": "LEGACY fold accepted before the repair",
              "metadata": {"fold": {"replaces": [scratchpad_block_sha256(source)],
                                    "covers": {"from": source["ts"], "to": source["ts"], "blocks": 1}},
                           "source_ref": relative}}
    memory.mutate_scratchpad_blocks(lambda _rows: [legacy])
    assert append_jsonl(memory.journal_path(), {
        "ts": legacy["ts"], "type": "fold_sources_preserved", "entry_id": entry_id,
        "source_blocks": [source], "fold_block": legacy, "source_ref": relative})
    stored = memory.scratchpad_blocks_path().read_text(encoding="utf-8")
    journaled = memory.journal_path().read_bytes()

    _fold_from(forked, memory, "forked")

    assert _blocks(memory)[0] == json.loads(stored)[0] == legacy
    assert memory.journal_path().read_bytes().startswith(journaled)
    assert _opened(_follow(forked, relative["read"]), entry_id) is None  # the historical address, unchanged
    section = _context_section(env, forked_ctx)
    assert _opened(_run(forked_ctx, _reader_beside(section, entry_id)), entry_id) == [source]


@pytest.mark.parametrize("reading", ["canonical", "forked"])
def test_a_conflict_points_the_stale_focus_to_current_complete_blocks(roots, reading):
    """A second focus's accepted fold is newer than this caller's already-built context."""
    from tests.test_authored_memory import _run

    env, memory, fork = roots
    ctx, registry = _rooted(env, env.drive_root if reading == "canonical" else fork)
    assert _tool(registry, content=_note(0)).status == "ok"
    old_context = _context_section(env, ctx)
    old_hashes = _DIGEST.findall(old_context)
    newer = "NEW accepted understanding " + "detail " * 40 + "COMPLETE-TAIL"
    other = _rooted(env, fork if reading == "canonical" else env.drive_root)[1]
    assert _tool(other, content=newer, replaces=old_hashes).status == "ok"
    before = memory.scratchpad_blocks_path().read_bytes()

    refused = _tool(registry, content="Fold from the now outdated first context.", replaces=old_hashes)

    assert refused.status != "ok" and "SCRATCHPAD_FOLD_CONFLICT" in refused.text
    assert newer not in old_context and newer not in refused.text  # previews are not source bodies
    assert memory.scratchpad_blocks_path().read_bytes() == before
    [reader] = _CALL.findall(refused.text)
    opened = _run(ctx, reader)
    body = "\n".join(line.split("\t", 1)[1] for line in opened.splitlines() if "\t" in line)
    assert json.loads(body) == json.loads(before)
    assert "COMPLETE-TAIL" in opened


# --- failure and crash points ------------------------------------------------------------


def test_a_failed_preservation_write_leaves_every_active_block(install, monkeypatch):
    env, memory, registry = install
    for index in range(3):
        _tool(registry, content=_note(index))
    chosen = _digests_in_view(env, memory)[-2:]
    before_bytes = memory.scratchpad_blocks_path().read_bytes()
    before_md = memory.scratchpad_path().read_bytes()
    from ouroboros import memory as memory_module

    real = memory_module.append_jsonl
    monkeypatch.setattr(memory_module, "append_jsonl", lambda path, row, *a, **k: (
        False if row.get("type") == "fold_sources_preserved" else real(path, row, *a, **k)))

    refused = _tool(registry, content="FOLD whose sources could not be preserved.", replaces=chosen)

    assert refused.status != "ok" and "journal" in refused.text.lower()
    assert memory.scratchpad_blocks_path().read_bytes() == before_bytes
    assert memory.scratchpad_path().read_bytes() == before_md
    section = _scratchpad_section(_next_request(env, memory))
    assert all(_note(index) in section for index in range(3))


def test_a_retry_after_a_torn_preservation_record_keeps_parseable_originals(install, monkeypatch):
    """A partial write stays visible, but must not swallow the next preservation record."""
    from ouroboros import memory as memory_module
    from ouroboros.memory_reader import read_memory

    _env, memory, registry = install
    assert _tool(registry, content=_note(0)).status == "ok"
    originals = _blocks(memory)
    chosen = [scratchpad_block_sha256(block) for block in originals]
    before = memory.scratchpad_blocks_path().read_bytes()
    real, failed = memory_module.append_jsonl, False
    torn = b'{"type":"fold_sources_preserved","source_blocks":['

    def partial_once(path, row, *args, **kwargs):
        nonlocal failed
        if row.get("type") == "fold_sources_preserved" and not failed:
            failed = True
            with path.open("ab") as stream:
                stream.write(torn)
            raise OSError("disk full after partial preservation")
        return real(path, row, *args, **kwargs)

    monkeypatch.setattr(memory_module, "append_jsonl", partial_once)
    refused = _tool(registry, content="Fold interrupted before preserving its sources.", replaces=chosen)
    assert refused.status != "ok" and memory.scratchpad_blocks_path().read_bytes() == before
    prefix = memory.journal_path().read_bytes()
    assert prefix.endswith(torn)
    retried = _tool(registry, content="Accepted fold after storage recovered.", replaces=chosen)
    assert retried.status == "ok", retried.text
    entry = _blocks(memory)[0]["metadata"]["source_ref"]
    assert memory.journal_path().read_bytes().startswith(prefix)
    assert _opened(_follow(registry, entry["read"]), entry["entry_id"]) == originals
    history = read_memory(memory.drive_root, "catalog", {
        "area": "history", "address": {"kind": "journal", "journal": "scratchpad"}})
    assert history["complete"] and history["unterminated_tail_bytes"] == 0
    assert sum(bool(row.get("unreadable")) for row in history["entries"]) == 1
    assert len(history["entries"]) == 4  # append, torn row, preserved originals, completed fold


def test_a_failed_canonical_write_after_preservation_keeps_the_active_blocks_and_says_so(install, monkeypatch):
    env, memory, registry = install
    for index in range(3):
        _tool(registry, content=_note(index))
    chosen = _digests_in_view(env, memory)[-2:]
    before_bytes = memory.scratchpad_blocks_path().read_bytes()
    from ouroboros import memory as memory_module

    def read_only_volume(*_args, **_kwargs):
        raise OSError("read-only volume")

    monkeypatch.setattr(memory_module, "atomic_write_json", read_only_volume)

    failed = _tool(registry, content="FOLD that could not be published.", replaces=chosen)

    assert failed.status != "ok" and "read-only volume" in failed.text
    monkeypatch.undo()
    assert memory.scratchpad_blocks_path().read_bytes() == before_bytes
    [preserved] = _journal(memory, "fold_sources_preserved")
    [failure] = _journal(memory, "block_fold_failed")
    assert failure["entry_id"] == preserved["entry_id"] and _journal(memory, "blocks_folded") == []
    section = _scratchpad_section(_next_request(env, memory))
    assert all(_note(index) in section for index in range(3)) and "FOLD that could not" not in section


def test_a_projection_failure_after_the_canonical_fold_reports_the_applied_fold(install, monkeypatch):
    env, memory, registry = install
    for index in range(3):
        _tool(registry, content=_note(index))
    chosen = _digests_in_view(env, memory)[-2:]
    stale_md = memory.scratchpad_path().read_bytes()

    def broken_projection(self, blocks):
        raise OSError("projection disk error")

    monkeypatch.setattr(Memory, "_write_scratchpad_markdown", broken_projection)
    result = _tool(registry, content="FOLD applied while the projection failed.", replaces=chosen)

    assert result.status == "ok", result.text  # the canonical change happened: no false failure
    assert "projection" in result.text and "projection disk error" in result.text
    assert _blocks(memory)[0]["content"] == "FOLD applied while the projection failed."
    assert memory.scratchpad_path().read_bytes() == stale_md  # the projection really is stale
    [completed] = _journal(memory, "blocks_folded")
    assert "projection disk error" in completed["projection_error"]
    section = _scratchpad_section(_next_request(env, memory))
    # The next request reads the canonical blocks, never the stale projection.
    assert "FOLD applied while the projection failed." in section
    assert _note(0) not in section and _note(2) in section


def test_a_projection_failure_on_append_does_not_hide_the_new_block(install, monkeypatch):
    env, memory, registry = install
    _tool(registry, content=_note(0))

    def broken_projection(self, blocks):
        raise OSError("projection disk error")

    monkeypatch.setattr(Memory, "_write_scratchpad_markdown", broken_projection)
    result = _tool(registry, content=_note(1))

    assert result.status == "ok" and "projection disk error" in result.text
    assert [block["content"] for block in _blocks(memory)] == [_note(0), _note(1)]
    assert _journal(memory, "block_append_failed") == []
    assert _note(1) in _scratchpad_section(_next_request(env, memory))


# --- no automatic writer remains ----------------------------------------------------------


def test_no_count_or_size_cap_remains():
    """The former 10-block and 60 000-character caps (the automatic writer's own removal is pinned
    in ``test_scratchpad_consolidation``)."""
    from ouroboros import context_budget
    from ouroboros import memory as memory_module

    assert not hasattr(memory_module, "_SCRATCHPAD_MAX_BLOCKS")
    assert not hasattr(context_budget, "SCRATCHPAD_MAX_CONTENT_CHARS")
