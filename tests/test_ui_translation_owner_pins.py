"""The owner's interface wording from conversation (ouroboros/tools/control_ui_translation.py).

``search_ui_translations`` reads the memory, its miss queue and the host catalog;
``pin_ui_translation`` writes the ``owner`` layer through ``i18n_memory.update_memory``
and hands ``ui_i18n_updated`` on for open windows. Both directions for each rule: what a
call writes AND what it leaves alone, which principal may call it AND which one the
existing gates refuse, when a refresh signal is handed on AND when the receipt says not.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from ouroboros import i18n_memory as memory
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools.registry import ToolContext, ToolRegistry

RU_CATEGORIES = ["one", "few", "many", "other"]
# Intl.PluralRules("ru") for 0..100 as the browser records it at the language choice.
RU_SELECT = {"map": {str(n): ("one" if n % 10 == 1 and n % 100 != 11
                              else "few" if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14
                              else "many") for n in range(0, 101)}, "period": 100}
THEME_LIGHT = "Light" + memory.SCOPE_SEPARATOR + "[data-theme-control]"


@pytest.fixture
def frames(monkeypatch):
    """What reaches the WebSocket broadcast seam; the seam reports a live server loop."""
    from ouroboros.gateway import ws

    sent = []
    monkeypatch.setattr(ws, "broadcast_ws_sync", lambda frame: sent.append(frame) or True)
    return sent


def _drive(tmp_path):
    drive = tmp_path / "data"
    drive.mkdir(exist_ok=True)
    return drive


def _registry(tmp_path, **fields):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    drive = _drive(tmp_path)
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)
    reg.set_context(ToolContext(repo_dir=repo, drive_root=drive, task_id="turn-1", **fields))
    return reg


def _seed(tmp_path, monkeypatch, *, language="ru"):
    """A Russian memory as the generator, an import and the browser left it."""
    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", language)
    drive = _drive(tmp_path)

    def _fill(doc):
        memory.apply_generated(doc, {
            "Sections": {"text": "Разделы"},
            "{n} notes": {"forms": {"one": "{n} заметка", "few": "{n} заметки", "many": "{n} заметок", "other": "{n} заметки"}},
            "code:task.headline.done": {"text": "Выполнено"},
        }, model="fake/light", sources={"code:task.headline.done": "Done"},
            source_hashes={"code:task.headline.done": memory.source_hash("Done")})
        memory.apply_import(doc, {"entries": {"Archive": {"text": "Архив"}, THEME_LIGHT: {"text": "Светлая"}}})
        memory.refuse_keys(doc, ["Wizard step"], reason="no valid answer after the attempt bound")
        return doc

    memory.update_memory(drive, "ru", _fill, create=True, plural_categories=RU_CATEGORIES, plural_select=RU_SELECT)
    memory.record_missing(drive, "ru", [{"key": "Regenerate", "context": {"role": "button", "page": "settings"}}])
    return drive


def _json(result):
    assert result.status == "ok", result.text
    return json.loads(result.text)


def _doc(drive):
    return memory.load_memory(drive, "ru")


# --- search ------------------------------------------------------------------------


def test_search_finds_what_the_owner_sees_and_what_is_still_english(tmp_path, monkeypatch):
    drive = _seed(tmp_path, monkeypatch)
    reg = _registry(tmp_path)

    body = _json(reg.execute_result("search_ui_translations", {"query": "РАЗДЕЛЫ"}))
    assert body["language"] == "ru" and body["revision"] == _doc(drive)["revision"]
    assert body["plural_categories"] == RU_CATEGORIES and body["total"] == 1
    assert body["matches"] == [{"key": "Sections", "provenance": "generated", "wording": "Разделы"}]

    # A host sentence is found by its English from the catalog and named by where it shows.
    done = _json(reg.execute_result("search_ui_translations", {"query": "Done"}))["matches"]
    code = next(row for row in done if row["key"] == "code:task.headline.done")
    assert code["english"] == "Done" and code["wording"] == "Выполнено" and "task card" in code["place"]

    # A scoped word comes back with its scope as a plain field, never the separator.
    light = _json(reg.execute_result("search_ui_translations", {"query": "светлая"}))["matches"]
    assert light == [{"key": "Light", "scope": "[data-theme-control]", "provenance": "imported", "wording": "Светлая"}]

    # Untranslated strings the install knows: queued by a window, refused by the generator.
    queued = _json(reg.execute_result("search_ui_translations", {"query": "regenerate"}))["matches"]
    assert queued[0] == {"key": "Regenerate", "provenance": "untranslated", "note": "queued for the generator",
                         "place": {"role": "button", "page": "settings"}}
    refused = _json(reg.execute_result("search_ui_translations", {"query": "wizard"}))["matches"]
    assert refused[0]["provenance"] == "untranslated" and refused[0]["note"] == "the generator gave up on it"

    # Plural forms are matched form by form; whole-text matches sort first; the limit is disclosed.
    notes = _json(reg.execute_result("search_ui_translations", {"query": "заметок", "limit": 500}))
    assert notes["limit"] == 50 and notes["matches"][0]["key"] == "{n} notes"
    assert _json(reg.execute_result("search_ui_translations", {"query": "архив"}))["matches"][0]["key"] == "Archive"


def test_search_refuses_typed_and_reads_nothing_it_cannot(tmp_path, monkeypatch):
    drive = _seed(tmp_path, monkeypatch)
    reg = _registry(tmp_path)
    empty = reg.execute_result("search_ui_translations", {"query": "  "})
    assert (empty.status, empty.code, empty.meta["reason_code"]) == ("error", "TOOL_ARG_ERROR", "query_empty")
    bad_tag = reg.execute_result("search_ui_translations", {"query": "x", "language": "Russian"})
    assert bad_tag.code == "TOOL_ARG_ERROR" and bad_tag.meta["reason_code"] == "language_not_a_tag"
    absent = reg.execute_result("search_ui_translations", {"query": "x", "language": "de"})
    assert (absent.status, absent.meta["reason_code"]) == ("unavailable", "no_memory")

    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "en")
    english = reg.execute_result("search_ui_translations", {"query": "Sections"})
    assert (english.status, english.code, english.meta["reason_code"]) == ("unavailable", "CAPABILITY_UNAVAILABLE", "english_interface")
    assert english.meta["operation_outcome"] == "completed_no_effect"

    monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
    path = memory.memory_path(drive, "ru")
    path.write_text('{"schema": 2}', encoding="utf-8")
    broken = reg.execute_result("search_ui_translations", {"query": "Sections"})
    assert broken.meta["reason_code"] == "memory_invalid" and path.read_text(encoding="utf-8") == '{"schema": 2}'


# --- pin / unpin through the memory's writer ------------------------------------------


def test_pin_replaces_a_generated_wording_and_hands_on_a_refresh(tmp_path, monkeypatch, frames):
    drive = _seed(tmp_path, monkeypatch)
    before = _doc(drive)["revision"]
    reg = _registry(tmp_path)

    receipt = _json(reg.execute_result("pin_ui_translation", {"action": "pin", "key": "Sections", "text": "Проекты"}))
    doc = _doc(drive)
    assert doc["revision"] == before + 1 and receipt["revision"] == before + 1
    assert doc["entries"]["Sections"]["provenance"] == "owner" and doc["entries"]["Sections"]["text"] == "Проекты"
    assert receipt["changed"] is True and receipt["window_refresh"] == "handed_to_server_loop"
    assert "window_refresh_note" not in receipt
    assert receipt["previous"] == {"provenance": "generated", "wording": "Разделы"}
    assert receipt["now"] == {"provenance": "owner", "wording": "Проекты"}
    assert frames == [{"type": "ui_i18n_updated", "language": "ru", "revision": before + 1}]
    assert memory.tr("Sections", "ru", "Sections", drive_root=drive) == "Проекты"  # Telegram and host seams read it too

    # The same wording again writes nothing and tells no window.
    again = reg.execute_result("pin_ui_translation", {"action": "pin", "key": "Sections", "text": "Проекты"})
    assert _json(again)["reason"] == "already_pinned" and again.meta["operation_outcome"] == "completed_no_effect"
    assert _doc(drive)["revision"] == before + 1 and len(frames) == 1


def test_owner_pins_keep_the_memory_precedence(tmp_path, monkeypatch, frames):
    drive = _seed(tmp_path, monkeypatch)
    reg = _registry(tmp_path)
    _json(reg.execute_result("pin_ui_translation", {"action": "pin", "key": "Archive", "text": "Хранилище"}))
    _json(reg.execute_result("pin_ui_translation", {"action": "pin", "key": "Sections", "text": "Проекты"}))
    doc = _doc(drive)
    assert doc["shadow"]["Archive"]["text"] == "Архив"  # the import waits under the pin

    # Generation never overwrites a pin, Regenerate keeps it, a new import shadows itself under it.
    def _pressure(current):
        assert memory.apply_generated(current, {"Sections": {"text": "Секции"}}, model="fake/light") == 0
        memory.regenerate_reset(current)
        memory.apply_import(current, {"entries": {"Sections": {"text": "Секции (пакет)"}}})
        return current

    memory.update_memory(drive, "ru", _pressure)
    doc = _doc(drive)
    assert (doc["entries"]["Sections"]["provenance"], doc["entries"]["Sections"]["text"]) == ("owner", "Проекты")
    assert doc["shadow"]["Sections"]["text"] == "Секции (пакет)"
    # The search tells the mind what unpinning would bring back.
    row = _json(reg.execute_result("search_ui_translations", {"query": "Проекты"}))["matches"][0]
    assert row["under_pin"] == {"provenance": "imported", "wording": "Секции (пакет)"}


def test_unpin_restores_an_import_or_leaves_english_without_promising_the_old_wording(tmp_path, monkeypatch, frames):
    drive = _seed(tmp_path, monkeypatch)
    reg = _registry(tmp_path)
    _json(reg.execute_result("pin_ui_translation", {"action": "pin", "key": "Archive", "text": "Хранилище"}))
    _json(reg.execute_result("pin_ui_translation", {"action": "pin", "key": "Sections", "text": "Проекты"}))

    back = _json(reg.execute_result("pin_ui_translation", {"action": "unpin", "key": "Archive"}))
    assert back["now"] == {"provenance": "imported", "wording": "Архив"} and "note" not in back
    assert "Archive" not in _doc(drive)["shadow"]

    gone = _json(reg.execute_result("pin_ui_translation", {"action": "unpin", "key": "Sections"}))
    assert gone["now"] == {"provenance": "untranslated"} and "need not match" in gone["note"]
    assert "Sections" not in _doc(drive)["entries"]
    assert memory.tr("Sections", "ru", "Sections", drive_root=drive) == "Sections"  # English until translated again
    revisions = [frame["revision"] for frame in frames]  # two pins, two unpins: each its own revision
    assert revisions == list(range(revisions[0], revisions[0] + 4))

    # Nothing pinned: no write, typed as no effect, the current layer reported.
    revision = _doc(drive)["revision"]
    idle = reg.execute_result("pin_ui_translation", {"action": "unpin", "key": "{n} notes"})
    body = _json(idle)
    assert body["reason"] == "not_pinned" and body["previous"]["provenance"] == "generated"
    assert idle.meta["operation_outcome"] == "completed_no_effect" and _doc(drive)["revision"] == revision


def test_unpin_of_a_refused_key_explains_the_required_regenerate(tmp_path, monkeypatch, frames):
    drive = _seed(tmp_path, monkeypatch)
    reg = _registry(tmp_path)
    _json(reg.execute_result("pin_ui_translation", {"action": "pin", "key": "Wizard step", "text": "Шаг мастера"}))
    receipt = _json(reg.execute_result("pin_ui_translation", {"action": "unpin", "key": "Wizard step"}))
    assert receipt["changed"] and receipt["now"]["provenance"] == "untranslated"
    assert "Regenerate" in receipt["note"] and "previously refused" in receipt["note"]
    assert "Wizard step" in _doc(drive)["refused"]
    memory.record_missing(drive, "ru", [{"key": "Wizard step"}])
    assert "Wizard step" not in json.loads(memory.pending_path(drive, "ru").read_text())["items"]


def test_pin_reaches_untranslated_strings_codes_and_plural_forms(tmp_path, monkeypatch, frames):
    drive = _seed(tmp_path, monkeypatch)
    reg = _registry(tmp_path)
    queued = _json(reg.execute_result("pin_ui_translation", {"action": "pin", "key": "  Regenerate ", "text": "Перевести заново"}))
    assert queued["previous"] == {"provenance": "untranslated"} and _doc(drive)["entries"]["Regenerate"]["provenance"] == "owner"

    code = _json(reg.execute_result("pin_ui_translation", {"action": "pin", "key": "code:task.headline.done", "text": "Готово"}))
    entry = _doc(drive)["entries"]["code:task.headline.done"]
    assert code["changed"] and entry["source"] == "Done" and entry["source_hash"] == memory.source_hash("Done")

    forms = {"one": "{n} запись", "few": "{n} записи", "many": "{n} записей", "other": "{n} записи"}
    _json(reg.execute_result("pin_ui_translation", {"action": "pin", "key": "{n} notes", "forms": forms}))
    assert memory.fmt("{n} notes", {"n": 5}, "ru", drive_root=drive) == "5 записей"

    single = _json(reg.execute_result("pin_ui_translation", {"action": "pin", "key": "{n} notes", "text": "записей: {n}"}))
    assert "every count" in single["note"]

    scoped = _json(reg.execute_result("pin_ui_translation", {
        "action": "pin", "key": "Light", "scope": "[data-theme-control]", "text": "Светлая тема"}))
    assert scoped["scope"] == "[data-theme-control]" and scoped["previous"]["provenance"] == "imported"
    doc = _doc(drive)
    assert (doc["entries"][THEME_LIGHT]["provenance"], doc["entries"][THEME_LIGHT]["text"]) == ("owner", "Светлая тема")
    assert doc["shadow"][THEME_LIGHT]["text"] == "Светлая"


@pytest.mark.parametrize(("args", "reason"), [
    ({"action": "pin", "key": "Section", "text": "Проекты"}, "key_unknown"),          # a near key is not a key
    ({"action": "pin", "key": "Light", "scope": "[data-review]", "text": "x"}, "key_unknown"),  # a scope nothing has
    ({"action": "pin", "key": "{n} notes", "text": "заметки"}, "wording_invalid"),     # dropped placeholder
    ({"action": "pin", "key": "Sections", "text": "<b>Проекты</b>"}, "wording_invalid"),  # markup the English lacks
    ({"action": "pin", "key": "Sections", "text": "  "}, "wording_invalid"),
    ({"action": "pin", "key": "Sections"}, "wording_invalid"),
    ({"action": "pin", "key": "Sections", "text": "a", "forms": {"other": "b"}}, "wording_invalid"),
    ({"action": "pin", "key": "{n} notes", "forms": {"plural": "{n} x"}}, "plural_category_unknown"),
    ({"action": "pin", "key": "code:task.headline.done", "scope": "x", "text": "Готово"}, "key_invalid"),
    ({"action": "pin", "key": "", "text": "x"}, "key_invalid"),
    ({"action": "unpin", "key": "Sections", "text": "Проекты"}, "wording_with_unpin"),
    ({"action": "rename", "key": "Sections"}, "action_invalid"),
])
def test_invalid_requests_are_refused_typed_and_write_nothing(tmp_path, monkeypatch, frames, args, reason):
    drive = _seed(tmp_path, monkeypatch)
    before = memory.memory_path(drive, "ru").read_bytes()
    result = _registry(tmp_path).execute_result("pin_ui_translation", args)
    assert (result.status, result.code, result.meta["reason_code"]) == ("error", "TOOL_ARG_ERROR", reason), result.text
    assert result.text.startswith("⚠️ TOOL_ARG_ERROR: ") and result.meta["operation_outcome"] == "completed_no_effect"
    assert memory.memory_path(drive, "ru").read_bytes() == before and frames == []


def test_a_busy_or_broken_memory_is_never_written_and_a_concurrent_batch_survives(tmp_path, monkeypatch, frames):
    drive = _seed(tmp_path, monkeypatch)
    reg = _registry(tmp_path)
    path = memory.memory_path(drive, "ru")
    monkeypatch.setattr(memory, "_MEMORY_LOCK_TIMEOUT_SEC", 0.2)
    with memory._memory_lock(path):  # another writer holds the memory
        busy = reg.execute_result("pin_ui_translation", {"action": "pin", "key": "Sections", "text": "Проекты"})
    assert (busy.status, busy.meta["reason_code"]) == ("unavailable", "memory_busy") and "nothing was written" in busy.text
    assert _doc(drive)["entries"]["Sections"]["provenance"] == "generated"

    # The generator lands a batch between the owner's search and the pin: both writes stand.
    seen = _json(reg.execute_result("search_ui_translations", {"query": "Разделы"}))["revision"]
    memory.update_memory(drive, "ru", lambda doc: doc if memory.apply_generated(doc, {"Open": {"text": "Открыть"}}, model="fake") else None)
    pinned = _json(reg.execute_result("pin_ui_translation", {"action": "pin", "key": "Sections", "text": "Проекты"}))
    assert pinned["revision"] == seen + 2
    assert _doc(drive)["entries"]["Open"]["text"] == "Открыть" and _doc(drive)["entries"]["Sections"]["text"] == "Проекты"

    original = path.read_bytes()
    path.write_text("{not json", encoding="utf-8")
    broken = reg.execute_result("pin_ui_translation", {"action": "unpin", "key": "Sections"})
    assert (broken.status, broken.meta["reason_code"]) == ("unavailable", "memory_invalid")
    assert path.read_text(encoding="utf-8") == "{not json"
    path.write_bytes(original)
    absent = reg.execute_result("pin_ui_translation", {"action": "pin", "key": "Sections", "text": "x", "language": "de"})
    assert absent.meta["reason_code"] == "no_memory" and not memory.memory_path(drive, "de").exists()


def test_the_receipt_says_when_no_refresh_was_handed_on(tmp_path, monkeypatch):
    from ouroboros.gateway import ws

    class _RefusingQueue:
        def put_nowait(self, _event):
            raise OSError("the host's queue is gone")

    drive = _seed(tmp_path, monkeypatch)
    monkeypatch.setattr(ws, "_event_loop", None)  # no host event queue and no server loop in this process
    receipt = _json(_registry(tmp_path).execute_result(
        "pin_ui_translation", {"action": "pin", "key": "Sections", "text": "Проекты"}))
    assert receipt["changed"] is True and receipt["window_refresh"] == "not_sent"
    assert "reached neither" in receipt["window_refresh_note"]
    assert _doc(drive)["entries"]["Sections"]["text"] == "Проекты"  # saved all the same
    lost = _json(_registry(tmp_path, event_queue=_RefusingQueue()).execute_result(
        "pin_ui_translation", {"action": "pin", "key": "Sections", "text": "Мои проекты"}))
    assert lost["changed"] is True and lost["window_refresh"] == "not_sent"
    assert _doc(drive)["entries"]["Sections"]["text"] == "Мои проекты"

    # A language the interface does not show is saved and signals no window.
    memory.update_memory(drive, "de", lambda doc: doc if memory.apply_generated(doc, {"Sections": {"text": "Bereiche"}}, model="x") else None,
                         create=True)
    other = _json(_registry(tmp_path).execute_result(
        "pin_ui_translation", {"action": "pin", "key": "Sections", "text": "Projekte", "language": "de"}))
    assert other["window_refresh"] == "not_sent" and "not the interface language" in other["window_refresh_note"]


@pytest.mark.parametrize("binding", ["context", "metadata"])
def test_a_forked_drive_reads_and_writes_the_canonical_memory(tmp_path, monkeypatch, frames, binding):
    """``agent.py`` runs a root whose memory mode forks its drive on ``state/headless_tasks/<id>/data``
    with the install's data root as ``budget_drive_root``, which ``canonical_data_root`` reads from the
    task metadata or the context. Search and pin both use that one memory; a dictionary file on the
    forked drive is neither read nor written, and the binding widens no one: a read-only child on the
    same drives is still refused."""
    canonical = _seed(tmp_path, monkeypatch)
    fork = canonical / "state" / "headless_tasks" / "fork-1" / "data"
    memory.update_memory(fork, "ru", lambda doc: doc if memory.apply_generated(
        doc, {"Sections": {"text": "Секции"}}, model="fake/light") else None, create=True)
    fork_file = memory.memory_path(fork, "ru")
    fork_before = fork_file.read_bytes()
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    bound = {"budget_drive_root": str(canonical)} if binding == "context" else {
        "task_metadata": {"budget_drive_root": str(canonical)}}

    def _forked(**extra):
        reg = ToolRegistry(repo_dir=repo, drive_root=fork)
        reg.set_context(ToolContext(repo_dir=repo, drive_root=fork, memory_mode="forked", task_id="fork-1",
                                    **bound, **extra))
        return reg

    root = _forked()
    found = _json(root.execute_result("search_ui_translations", {"query": "Sections"}))
    assert found["revision"] == _doc(canonical)["revision"] and found["matches"][0]["wording"] == "Разделы"
    assert _json(root.execute_result("search_ui_translations", {"query": "Секции"}))["total"] == 0
    receipt = _json(root.execute_result("pin_ui_translation", {"action": "pin", "key": "Sections", "text": "Проекты"}))
    assert receipt["revision"] == _doc(canonical)["revision"] and _doc(canonical)["entries"]["Sections"]["text"] == "Проекты"
    assert fork_file.read_bytes() == fork_before

    canonical_now = memory.memory_path(canonical, "ru").read_bytes()
    child = _forked(task_constraint=TaskConstraint(mode="local_readonly_subagent"))
    assert child.execute_result("pin_ui_translation", {"action": "unpin", "key": "Sections"}).status != "ok"
    assert memory.memory_path(canonical, "ru").read_bytes() == canonical_now and fork_file.read_bytes() == fork_before


# --- who may call it: the existing gates ------------------------------------------------


def test_children_presence_and_observe_are_refused_by_the_existing_gates(tmp_path, monkeypatch, frames):
    from ouroboros import consciousness_authority as ca
    from ouroboros.presence_authority import PresenceCapabilityCeiling, PresenceToolGrant, presence_ceiling_payload

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    drive = _seed(tmp_path, monkeypatch)
    before = memory.memory_path(drive, "ru").read_bytes()
    pin = {"action": "pin", "key": "Sections", "text": "Проекты"}

    readonly = _registry(tmp_path, task_constraint=TaskConstraint(mode="local_readonly_subagent"))
    worktree = tmp_path / "wt"
    worktree.mkdir()
    acting = _registry(tmp_path, workspace_root=str(worktree), workspace_mode="self_worktree", task_constraint=TaskConstraint(
        mode="acting_subagent", surface="self_worktree", write_root=str(worktree)))
    for reg, why in ((readonly, "read-only subagent"), (acting, "acting subagent")):
        for name, args in (("pin_ui_translation", pin), ("search_ui_translations", {"query": "Разделы"})):
            assert why in reg.policy_hidden_reason(name) and name not in reg.initial_tool_names()
            assert reg.execute_result(name, args).status != "ok"

    presence = _registry(tmp_path)
    presence._ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(PresenceCapabilityCeiling(
        skill_name="guest-room", skill_content_hash="a" * 64, profile_fingerprint="b" * 64,
        state_fingerprint="c" * 64, selection_fingerprint="d" * 64, model_slot="main", inline_max_rounds=3,
        tool_grants=(PresenceToolGrant("knowledge_read"),), resource_grants=(), digest="0" * 64))}
    assert presence.policy_hidden_reason("pin_ui_translation") == "outside this presence task's positive capability ceiling"
    assert presence.execute_result("pin_ui_translation", pin).status == "blocked"

    wake = {"id": "wake-1", "type": "task", "text": "wake", "_is_direct_chat": True, "chat_id": 1, "metadata": {
        "initiator": "consciousness", "usage_category": "consciousness", "wake_reason": "heartbeat",
        "consciousness_autonomy": "observe", "model_role": "consciousness"}}
    observe = _registry(tmp_path, is_direct_chat=True, task_metadata=ca.apply_consciousness_authority(wake)["metadata"])
    assert "RESOURCE_CONSTRAINT_BLOCKED" in observe.execute("pin_ui_translation", pin)
    assert _json(observe.execute_result("search_ui_translations", {"query": "Разделы"}))["total"] == 1  # reading stays

    assert memory.memory_path(drive, "ru").read_bytes() == before and frames == []


def test_both_tools_are_registered_with_one_policy_and_their_capability_class():
    from ouroboros import consciousness_authority as ca
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        CORE_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
        OBSERVE_WORLD_MUTATION_TOOLS,
        READ_ONLY_PARALLEL_TOOLS,
    )
    from ouroboros.tools.control import get_tools

    entries = {entry.name: entry for entry in get_tools()}
    assert entries["search_ui_translations"].safety_policy == entries["pin_ui_translation"].safety_policy == "skip"
    assert "search_ui_translations" in READ_ONLY_PARALLEL_TOOLS and "pin_ui_translation" in OBSERVE_WORLD_MUTATION_TOOLS
    assert "pin_ui_translation" in ca.disabled_tools_for("observe") and "pin_ui_translation" not in ca.disabled_tools_for("act")
    for name in ("search_ui_translations", "pin_ui_translation"):
        assert name not in CORE_TOOL_NAMES | LOCAL_READONLY_SUBAGENT_TOOL_NAMES | ACTING_SUBAGENT_TOOL_NAMES


# --- the consumer: an open window hears the frame and re-reads the memory ---------------


def test_a_pin_from_a_server_turn_reaches_an_open_window(tmp_path, monkeypatch):
    """The SPA's `ws.on('ui_i18n_updated', refreshDictionary)` re-reads GET /api/ui/i18n: here a
    socket on the real gateway routes receives the frame from a turn thread, and the GET it
    triggers answers the pinned wording."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway import ui_i18n, ws
    from ouroboros.gateway.router import collect_routes

    drive = _seed(tmp_path, monkeypatch)
    monkeypatch.setattr(ui_i18n, "_LANGUAGE_HOOKS", [])
    monkeypatch.setattr(ws, "_ws_clients", [])
    monkeypatch.setattr(ws, "_event_loop", None)

    @asynccontextmanager
    async def lifespan(_app):
        ws.set_event_loop(asyncio.get_running_loop())  # what server.lifespan does
        yield

    app = Starlette(routes=collect_routes(data_dir=drive), lifespan=lifespan)
    app.state.drive_root = drive
    with TestClient(app) as client, client.websocket_connect("/ws") as socket:
        assert client.get("/api/ui/i18n").json()["entries"]["Sections"]["text"] == "Разделы"
        receipt = _json(_registry(tmp_path).execute_result(
            "pin_ui_translation", {"action": "pin", "key": "Sections", "text": "Проекты"}))
        assert receipt["window_refresh"] == "handed_to_server_loop"
        assert socket.receive_json() == {"type": "ui_i18n_updated", "language": "ru", "revision": receipt["revision"]}
        body = client.get("/api/ui/i18n").json()
    assert body["revision"] == receipt["revision"] and body["stats"]["owner"] == 1
    assert body["entries"]["Sections"]["text"] == "Проекты" and body["entries"]["Sections"]["provenance"] == "owner"



def test_the_host_relays_only_a_well_formed_update(tmp_path):
    from types import SimpleNamespace

    from supervisor.events import dispatch_event

    sent = []
    host = SimpleNamespace(bridge=SimpleNamespace(broadcast=sent.append), DRIVE_ROOT=tmp_path,
                           append_jsonl=lambda *_args: None)
    for bad in ({"language": "", "revision": 1}, {"language": 5, "revision": 1}, {"language": "ru", "revision": "1"},
                {"language": "ru", "revision": True}, {"language": "ru", "revision": -1}, {"language": "ru"}):
        dispatch_event({"type": "ui_i18n_updated", **bad}, host)
    assert sent == []
    dispatch_event({"type": "ui_i18n_updated", "language": " ru ", "revision": 7}, host)
    assert sent == [{"type": "ui_i18n_updated", "language": "ru", "revision": 7}]


def test_a_pooled_worker_pin_reaches_an_open_window_through_its_host(tmp_path, monkeypatch):
    """A pooled worker owns no window: its pin goes on the worker's event queue, never to a
    loop in its own process; the supervisor's real dispatch hands it to the bridge
    (``_broadcast_fn = broadcast_ws_sync``, as ``server.py`` binds it), the open socket receives
    the frame, and the read the SPA makes on it answers the pinned wording."""
    import queue
    from types import SimpleNamespace

    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway import ui_i18n, ws
    from ouroboros.gateway.router import collect_routes
    from supervisor.events import dispatch_event
    from supervisor.message_bus import LocalChatBridge

    drive = _seed(tmp_path, monkeypatch)
    monkeypatch.setattr(ui_i18n, "_LANGUAGE_HOOKS", [])
    monkeypatch.setattr(ws, "_ws_clients", [])
    monkeypatch.setattr(ws, "_event_loop", None)
    real_broadcast, calls = ws.broadcast_ws_sync, []
    monkeypatch.setattr(ws, "broadcast_ws_sync", lambda frame: calls.append("direct") or real_broadcast(frame))
    bridge = LocalChatBridge()
    bridge._broadcast_fn = lambda frame: calls.append("bridge") or real_broadcast(frame)
    host = SimpleNamespace(bridge=bridge, DRIVE_ROOT=drive, append_jsonl=lambda *_args: None)

    @asynccontextmanager
    async def lifespan(_app):
        ws.set_event_loop(asyncio.get_running_loop())  # what server.lifespan does
        yield

    app = Starlette(routes=collect_routes(data_dir=drive), lifespan=lifespan)
    app.state.drive_root = drive
    out_q = queue.Queue()  # the worker's outbound queue to the supervisor
    with TestClient(app) as client, client.websocket_connect("/ws") as socket:
        assert client.get("/api/ui/i18n").json()["entries"]["Sections"]["text"] == "Разделы"
        receipt = _json(_registry(tmp_path, event_queue=out_q).execute_result(
            "pin_ui_translation", {"action": "pin", "key": "Sections", "text": "Проекты"}))
        frame = {"type": "ui_i18n_updated", "language": "ru", "revision": receipt["revision"]}
        assert receipt["window_refresh"] == "queued_for_host" and calls == []
        event = out_q.get_nowait()
        assert event == frame and out_q.empty()
        dispatch_event(event, host)  # the supervisor's drain of the worker's queue
        assert socket.receive_json() == frame and calls == ["bridge"]
        body = client.get("/api/ui/i18n").json()
    assert body["revision"] == receipt["revision"] and body["entries"]["Sections"]["text"] == "Проекты"
