"""The Nano publication gate on synthetic counts and fixtures: deficits, the shared fence, the checked profile.

No model, endpoint, installation state or hosted task is used; counts, launches and files are fixture values.
"""
import hashlib
import json
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

from ouroboros import config, nano_host_publication, nano_publication
from ouroboros.nano_publication import (MemoryChange, NanoPublication, NanoPublicationRefused, PUBLICATION_REVISION,
                                        _invalidate_publication_reads, publication_fence)

REPLY = 8_192
RULES = "OUROBOROS_SAFETY_OWNER_RULES"
FORMATTER = {"source": "serving_llama_formatter", "formatter": "Jinja2ChatFormatter", "library_version": "fixture",
             "chat_format": "chatml", "launched_n_ctx": 80_000}


def _sample(background, memory=5_000, **extra):
    return {"status": "exact", "context_window": 80_000, "process_id": 7, "raw_context_window": 80_128,
            "formatter": "fixture", "caller_max_tokens": 65_536, "reply_floor_tokens": REPLY,
            "memory_tokens": memory, "without_memory": {"input_tokens": background},
            "minimum_work_headroom_at_reply_floor_tokens": 80_000 - background - memory - REPLY, **extra}


def _floor(sample):
    return SimpleNamespace(count=lambda: sample, variant="active", instruction_digest=lambda: "same",
                           catalogue_digest=lambda: "same")


def _publish(root, before, after, *, memory=(5_000, 5_000)):
    pair = (_floor(_sample(before, memory[0])), _floor(_sample(after, memory[1])))
    gate = NanoPublication(root, lambda: "fixed", lambda changes: pair, None)
    with gate.publication(MemoryChange(None, None, b"synthetic", subject="synthetic")):
        pass


# --- three independent deficits -------------------------------------------------------------------

@pytest.mark.parametrize("after", [37_000, 36_500])
def test_equal_or_smaller_background_passes_even_when_it_already_crowds_the_reserve(tmp_path, after):
    # 37 000 + reply + 20 000 + 15 000 exceeds 80 000 before the write, and no proof was ever recorded.
    _publish(tmp_path, 37_000, after)


def test_growing_an_over_reserve_background_is_refused_without_an_earlier_proof(tmp_path):
    with pytest.raises(NanoPublicationRefused, match="over_capacity") as refusal:
        _publish(tmp_path, 37_000, 37_001)
    assert "193 tokens" in refusal.value.detail  # 37 001 + 8 192 + 35 000 − 80 000


@pytest.mark.parametrize(("before", "after"), [(37_000, 37_010), (36_808, 36_818)], ids=["grows", "appears"])
def test_a_smaller_whole_request_does_not_license_background_into_the_reserve(tmp_path, before, after):
    # A fold of 1 000 memory tokens shrinks N; a receipt or a returning WORLD adds 10 to B.
    with pytest.raises(NanoPublicationRefused, match="over_capacity"):
        _publish(tmp_path, before, after, memory=(6_000, 5_000))


@pytest.mark.parametrize(("background", "memory"), [((37_000, 36_990), (6_000, 5_000)), ((37_000, 37_000), (6_000, 5_000)),
                                                    ((37_000, 37_000), (5_000, 6_000))],
                         ids=["smaller-background", "fold", "memory-within-its-ceiling"])
def test_a_write_that_worsens_no_deficit_passes_on_a_crowded_profile(tmp_path, background, memory):
    _publish(tmp_path, *background, memory=memory)


def test_background_may_grow_up_to_the_reserve_and_not_past_it(tmp_path):
    _publish(tmp_path, 36_000, 80_000 - REPLY - 35_000)
    with pytest.raises(NanoPublicationRefused, match="over_capacity"):
        _publish(tmp_path, 36_000, 80_000 - REPLY - 35_000 + 1)


def test_a_count_without_its_background_is_unavailable_never_free(tmp_path):
    bare = {key: value for key, value in _sample(36_000).items() if key != "without_memory"}
    gate = NanoPublication(tmp_path, lambda: "fixed", lambda changes: (_floor(_sample(36_000)), _floor(bare)), None)
    with pytest.raises(NanoPublicationRefused, match="unavailable"):
        gate.prepare(MemoryChange(None, None, b"synthetic", subject="synthetic"))


# --- an incomplete source read ------------------------------------------------------------------

def _incomplete():
    sample = _sample(0)
    sample.pop("without_memory")
    return {**sample, "status": "unknown", "counter_confirmed": True, "formatter": FORMATTER,
            "reason_code": nano_publication.SOURCE_READ_INCOMPLETE, "reason": "source_read_incomplete:knowledge_index"}


@pytest.mark.parametrize("counted", [False, True], ids=["never-confirmed", "confirmed"])
def test_an_incomplete_source_read_is_its_own_refusal_and_voids_the_old_proof(tmp_path, counted):
    marker = tmp_path / PUBLICATION_REVISION
    marker.parent.mkdir(parents=True)
    if counted:
        _invalidate_publication_reads(marker, checked_profile="launch:x")
        marker.write_text(json.dumps({**json.loads(marker.read_bytes()), "proof": {"background_tokens": 30_000}}))
    pair = (_floor(_incomplete()), _floor(_incomplete()))
    gate = NanoPublication(tmp_path, lambda: "fixed", lambda changes: pair, None, profile_key=lambda: "launch:x")
    with pytest.raises(NanoPublicationRefused) as refusal:
        gate.prepare(MemoryChange(None, None, b"synthetic", subject="synthetic"))
    # Neither the never-confirmed fallback (an ordinary write) nor a recorded counter outage.
    assert refusal.value.reason == "incomplete" and "unreadable source" in refusal.value.tool_detail()
    row = json.loads(marker.read_bytes())
    assert "proof" not in row and row.get("reason") != "confirmed_counter_unavailable"
    assert row["checked_profile"] == "launch:x"


@pytest.mark.parametrize("background,memory,reason", [(36_808, 20_000, ""), (36_809, 10_000, "over_capacity"),
                                                     (30_000, 20_001, "over_capacity")])
def test_exact_repair_of_incomplete_baseline_has_to_fit_every_reserve(tmp_path, background, memory, reason):
    proposed = _sample(background, memory, formatter=FORMATTER)
    gate = NanoPublication(tmp_path, lambda: "fixed", lambda _: (_floor(_incomplete()), _floor(proposed)), None,
                           profile_key=lambda: "launch:x")
    edit = MemoryChange(None, None, b"synthetic", subject="synthetic")
    if reason:
        with pytest.raises(NanoPublicationRefused, match=reason):
            gate.prepare(edit)
    else:
        with gate.publication(edit):
            pass


# --- the shared fence -----------------------------------------------------------------------------

def test_unqualified_writers_take_the_same_fence_and_exclude_a_snapshot(tmp_path, monkeypatch):
    gate = NanoPublication(tmp_path, lambda: "fixed", lambda changes: None, None, profile_key=lambda: None)
    monkeypatch.setattr(nano_publication, '_NAME_LOCK_TIMEOUT_SEC', 0.05)
    with gate.ordinary() as ordinary:
        assert ordinary
        with pytest.raises(TimeoutError), publication_fence(tmp_path):
            pytest.fail('A snapshot entered while an ordinary writer held the fence')
    with publication_fence(tmp_path):
        with pytest.raises(TimeoutError), gate.ordinary():
            pytest.fail('An ordinary writer entered while a snapshot held the fence')
    with gate.ordinary() as ordinary:
        assert ordinary


def test_overlapping_max_scratchpad_writes_both_land_without_a_count(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from ouroboros import platform_layer
    from ouroboros.memory import Memory

    memory = Memory(tmp_path)
    gate = NanoPublication(tmp_path, lambda: "fixed", lambda changes: pytest.fail("no count"), None,
                           profile_key=lambda: None)
    first_in_write, second_contended, release = Event(), Event(), Event()
    lock_name = tmp_path / "memory" / ".nano-publication.name.lock"
    real_open = platform_layer.os.open

    def observe_open(path, *args, **kwargs):
        try:
            return real_open(path, *args, **kwargs)
        except FileExistsError:
            if path == str(lock_name):
                second_contended.set()
            raise

    def before_write():
        first_in_write.set()
        assert release.wait(5)

    def first_append(blocks):
        return [*blocks, {"ts": "2026-11-27T00:00:00Z", "source": "task", "content": "First note"}]

    monkeypatch.setattr(platform_layer.os, "open", observe_open)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(memory.mutate_scratchpad_blocks, first_append,
                            before_write=before_write, nano_admission=gate)
        try:
            assert first_in_write.wait(5)
            second = pool.submit(memory.append_scratchpad_block, "Second note", nano_admission=gate)
            assert second_contended.wait(5)
            assert not memory.scratchpad_blocks_path().exists()
        finally:
            release.set()
        assert first.result(timeout=5)[0]["content"] == "First note"
        assert second.result(timeout=5)["content"] == "Second note"

    assert [block["content"] for block in memory.load_scratchpad_blocks()] == ["First note", "Second note"]
    assert not (tmp_path / PUBLICATION_REVISION).exists()


def test_an_unqualified_ticket_commits_after_an_ordinary_writer(tmp_path, monkeypatch):
    gate = NanoPublication(tmp_path, lambda: "fixed", lambda changes: pytest.fail("no count"), None,
                           profile_key=lambda: None)
    rules = MemoryChange(None, b"old", b"new", subject="owner_rules")
    ticket = gate.prepare(rules)
    monkeypatch.setattr(nano_publication, '_NAME_LOCK_TIMEOUT_SEC', 0.05)
    with publication_fence(tmp_path):
        with pytest.raises(NanoPublicationRefused, match='TimeoutError'), ticket.commit():
            pytest.fail('A Max Settings writer entered during an ordinary memory write')
    with gate.publication(rules):
        assert not (tmp_path / PUBLICATION_REVISION).exists()


def test_an_unqualified_ticket_is_stale_once_the_profile_qualifies(tmp_path):
    profile = {"key": None}
    gate = NanoPublication(tmp_path, lambda: "fixed", lambda changes: None, None, profile_key=lambda: profile["key"])
    ticket = gate.prepare(MemoryChange(None, b"old", b"new", subject="owner_rules"))
    profile["key"] = "launch:x"
    with pytest.raises(NanoPublicationRefused, match="stale"), ticket.commit():
        pytest.fail("an ordinary commit cannot publish into a qualified profile")


# --- the checked capability -----------------------------------------------------------------------

def test_invalidation_carries_the_checked_capability_and_a_new_profile_starts_clean(tmp_path):
    marker = tmp_path / PUBLICATION_REVISION
    marker.parent.mkdir(parents=True)
    _invalidate_publication_reads(marker, checked_profile="launch:one", checked_saved="saved-one", checked_formatter="f1")
    _invalidate_publication_reads(marker)
    row = json.loads(marker.read_bytes())
    assert (row["checked_profile"], row["checked_saved"], row["checked_formatter"]) == ("launch:one", "saved-one", "f1")
    _invalidate_publication_reads(marker, checked_profile="launch:two", checked_formatter="f2")
    row = json.loads(marker.read_bytes())
    assert (row["checked_profile"], row["checked_formatter"]) == ("launch:two", "f2") and "checked_saved" not in row


def test_a_recorded_observation_keeps_the_whole_checked_capability(tmp_path):
    pair = (_floor(_sample(30_000, formatter=FORMATTER)), _floor(_sample(30_000, 5_100, formatter=FORMATTER)))
    gate = NanoPublication(tmp_path, lambda: "fixed", lambda changes: pair, None, profile_key=lambda: "launch:x",
                           profile_stamp=lambda: {"checked_saved": "saved-x"})
    with gate.publication(MemoryChange(None, None, b"synthetic", subject="synthetic")):
        pass
    row = json.loads((tmp_path / PUBLICATION_REVISION).read_bytes())
    assert row["proof"]["background_tokens"] == 30_000
    assert (row["checked_profile"], row["checked_saved"]) == ("launch:x", "saved-x")
    assert row["checked_formatter"] == nano_publication._formatter_digest(pair[0].count())
    assert row["proof"]["catalogue_digest"] == "same"  # the catalogue as counted, compared when it is shown


@pytest.mark.parametrize("confirmed", [True, False], ids=["confirmed-counter", "not-yet-confirmed"])
def test_an_edit_before_the_ordinary_start_waits_for_it_and_the_same_retry_then_counts(tmp_path, confirmed):
    marker = tmp_path / PUBLICATION_REVISION
    marker.parent.mkdir(parents=True)
    if confirmed:
        _invalidate_publication_reads(marker, checked_profile="launch:x")
    started, kept, landed = {"value": False}, [], []
    pair = (_floor(_sample(30_000)), _floor(_sample(30_000, 5_100)))

    def capture(changes):  # WORLD and the scratchpad exist only once the ordinary start wrote them
        if not started["value"]:
            raise NanoPublicationRefused("bootstrap_pending", "WORLD.md is written by the ordinary start")
        return pair

    gate = NanoPublication(tmp_path, lambda: "fixed", capture, lambda changes: kept.append(changes) or "kept",
                           profile_key=lambda: "launch:x")
    edit = MemoryChange(None, None, b"A large synthetic proposal.", subject="synthetic")
    with pytest.raises(NanoPublicationRefused) as refusal, gate.publication(edit):
        landed.append(True)
    assert refusal.value.reason == "bootstrap_pending" and "ordinary start" in refusal.value.tool_detail()
    assert not landed and kept == [(edit,)]  # nothing published unmeasured; the proposal is retained
    if confirmed:
        row = json.loads(marker.read_bytes())
        assert row["checked_profile"] == "launch:x" and row.get("reason") != "confirmed_counter_unavailable"
    started["value"] = True
    with gate.publication(edit):
        landed.append(True)
    assert landed == [True]


def test_both_shown_forms_of_the_catalogue_bind_the_observation():
    from ouroboros.section_floor import HostSection

    def floor(short):
        core = SimpleNamespace(semi_stable_sections=[HostSection("always_on_skills", "whole block")],
                               dynamic_head_sections=[], dynamic_sections=[HostSection("installed_skills", "full", short)])
        return nano_publication.RootFloor(SimpleNamespace(core=core), [], {}, 0, None, None)

    counted, now = floor("short A"), floor("short B")
    assert counted.catalogue_digest() != now.catalogue_digest()  # only the short form changed
    assert counted.instruction_digest() == now.instruction_digest()
    assert counted.catalogue_digest() == nano_publication.shown_digest(HostSection("installed_skills", "full", "short A"))
    assert counted.instruction_digest() == nano_publication.shown_digest("whole block")  # as the read-only check renders it


# --- the serving profile -------------------------------------------------------------------------

POSIX = "/opt/py/bin/python3 -m ouroboros.local_model_server --model /models/m.gguf --port 8767 --n_gpu_layers 0 " \
        "--chat_format chatml --n_ctx 80000"
HF = ("/opt/py/bin/python3 -m ouroboros.local_model_server --model "
      "/home/u/.cache/huggingface/hub/models--org--repo/snapshots/abc/sub/m.gguf --port 8767 --n_gpu_layers 0 --n_ctx 80000")
WINDOWS = ('"C:\\Program Files\\Ouroboros\\python.exe" -m ouroboros.local_model_server --model "C:\\Users\\a b\\m.gguf" '
           '--port 8767 --n_gpu_layers 0 --n_ctx 80000')


def test_the_serving_launch_is_read_from_its_own_command_line_on_every_platform():
    launch = nano_host_publication._launch_of
    assert launch(POSIX) == {"model": "/models/m.gguf", "port": 8767, "chat_format": "chatml", "context": 80_000}
    first = launch(HF)
    second = launch(HF.replace("/snapshots/abc/", "/snapshots/def/"))
    assert first.pop("snapshot_path") != second.pop("snapshot_path")
    assert first == second == {"model": "hf:org/repo:sub/m.gguf", "port": 8767, "chat_format": "", "context": 80_000}
    assert launch(WINDOWS) == {"model": "C:\\Users\\a b\\m.gguf", "port": 8767, "chat_format": "", "context": 80_000}
    assert launch("/usr/bin/vim notes.txt") is None
    assert launch(POSIX.replace(" --n_ctx 80000", "")) is None


def _saved(source="org/repo", filename="m.gguf", port=8767, chat_format="", context=80_000):
    return {"LOCAL_MODEL_SOURCE": source, "LOCAL_MODEL_FILENAME": filename, "LOCAL_MODEL_PORT": port,
            "LOCAL_MODEL_CHAT_FORMAT": chat_format, "LOCAL_MODEL_CONTEXT_LENGTH": context}


def test_saved_values_name_the_serving_launch_only_when_they_start_it_again(tmp_path):
    names, served = nano_host_publication._names, nano_host_publication._launch_of(HF)
    assert names(_saved(), served)  # a bare file name resolves to its one match in the repository
    assert names(_saved(filename="sub/m.gguf"), served)
    for pending in (_saved(source="org/other"), _saved(filename="other.gguf"), _saved(port=8768),
                    _saved(chat_format="chatml"), _saved(context=100_000)):
        assert not names(pending, served)
    model = tmp_path / "local.gguf"
    model.write_bytes(b"synthetic")
    local = {"model": str(model), "port": 8767, "chat_format": "", "context": 80_000}
    assert names(_saved(source=str(model), filename=""), local)
    assert not names(_saved(source=str(tmp_path / "other.gguf"), filename=""), local)


@pytest.fixture
def profile(tmp_path, monkeypatch):
    from ouroboros import local_model

    state = {"settings": {**_saved(), "OUROBOROS_CONTEXT_MODE": "nano"}, "serving": {"confirmed": True},
             "served": nano_host_publication._launch_of(HF), "report": None}
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(nano_host_publication, "_qualified", lambda: (state["settings"], state["serving"]))
    monkeypatch.setattr(nano_host_publication, "_served_launch", lambda serving: state["served"])
    monkeypatch.setattr(local_model, "get_manager", lambda **_: SimpleNamespace(
        **({"settings_application": lambda settings: state["report"]} if state["report"] is not None else {})))
    (tmp_path / "memory").mkdir()
    return tmp_path, state


def _digest(value):
    return nano_host_publication._digest(value)


def test_profile_key_names_the_serving_launch_not_settings_saved_but_not_applied(profile):
    _root, state = profile
    key = nano_host_publication._profile_key()
    state["settings"] = {**state["settings"], "LOCAL_MODEL_SOURCE": "org/saved-but-not-applied"}
    assert key == nano_host_publication._profile_key() == "launch:" + _digest(state["served"])


def test_a_stopped_server_keeps_its_checked_capability_until_its_launch_values_change(profile):
    root, state = profile
    checked = nano_host_publication._profile_key()
    assert nano_host_publication.profile_stamp() == {"checked_saved": _digest(nano_host_publication._saved_launch(state["settings"]))}
    _invalidate_publication_reads(root / PUBLICATION_REVISION, checked_profile=checked, **nano_host_publication.profile_stamp())
    state["served"], state["serving"] = None, {"confirmed": False}
    assert nano_host_publication._profile_key() == checked  # an outage of a known formatter, not a new one
    state["settings"] = {**state["settings"], "LOCAL_MODEL_FILENAME": "other.gguf"}
    assert nano_host_publication._profile_key() == "settings:" + _digest(nano_host_publication._saved_launch(state["settings"]))


def test_pending_saved_values_are_never_recorded_as_the_ones_that_serve(profile):
    _root, state = profile
    state["settings"] = {**state["settings"], "LOCAL_MODEL_SOURCE": "org/saved-but-not-applied"}
    assert nano_host_publication.profile_stamp() == {}


@pytest.mark.parametrize(("pending", "spelling"), [(["LOCAL_MODEL_SOURCE"], "pending:"), (["LOCAL_MODEL_N_GPU_LAYERS"], "settings:")])
def test_without_a_binding_the_owner_report_keeps_unapplied_values_apart(profile, pending, spelling):
    _root, state = profile
    state["served"], state["report"] = None, {"pending_keys": pending, "unknown_keys": []}
    assert nano_host_publication._profile_key() == spelling + _digest(nano_host_publication._saved_launch(state["settings"]))


def test_the_serving_launch_needs_the_binding_of_the_live_process(monkeypatch, tmp_path):
    from ouroboros import platform_layer, server_process

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    binding = {"pid": 7, "port": 8767}
    monkeypatch.setattr(server_process, "read_service_bindings", lambda root: {"local_model": binding})
    monkeypatch.setattr(platform_layer, "process_command", lambda pid: POSIX if pid == 7 else "")
    serving = {"confirmed": True, "process_id": 7, "port": 8767}
    assert nano_host_publication._served_launch(serving)["model"] == "/models/m.gguf"
    assert nano_host_publication._served_launch({**serving, "confirmed": False}) is None
    binding["pid"] = 8  # a binding of another process names nothing about this one
    assert nano_host_publication._served_launch(serving) is None
    binding["pid"] = 7
    monkeypatch.setattr(platform_layer, "process_command", lambda pid: "")
    assert nano_host_publication._served_launch(serving) is None  # an unreadable argv names no launch

    def unreadable(root):
        raise ValueError("runtime service bindings are unreadable")

    monkeypatch.setattr(server_process, "read_service_bindings", unreadable)
    with pytest.raises(NanoPublicationRefused, match="unavailable"):
        nano_host_publication._served_launch(serving)


# --- the first start, read without writing ------------------------------------------------------

def test_the_first_start_view_prices_only_bytes_every_start_writes_alike(tmp_path, monkeypatch):
    from ouroboros import world_profiler
    from ouroboros.nano_publication import _TEXT_PREVIEW

    monkeypatch.setattr(world_profiler, "measured_block", lambda *a: pytest.fail("no host probe for a count"))
    view = nano_host_publication._first_start_view(tmp_path, tmp_path / "repo")
    view.ensure_files()
    assert view.load_identity() == view._default_identity()  # a constant default
    assert view.load_scratchpad_blocks() == []
    for load in (view.load_scratchpad, view.load_world_profile):  # time, cwd, PATH and date of another start
        with pytest.raises(NanoPublicationRefused) as refusal:
            load()
        assert refusal.value.reason == "bootstrap_pending"
    token = _TEXT_PREVIEW.set({view.world_path(): b"# My World\nA proposed map."})
    try:
        assert view.load_world_profile() == "# My World\nA proposed map."  # the writer's own canonical bytes
    finally:
        _TEXT_PREVIEW.reset(token)
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]  # neither memory nor WORLD was created


def test_the_first_start_view_never_reads_an_unreadable_file_as_absent(tmp_path):
    view = nano_host_publication._first_start_view(tmp_path, tmp_path / "repo")
    view.identity_path().mkdir(parents=True)  # present, and not readable as a file
    with pytest.raises(NanoPublicationRefused) as refusal:
        view.load_identity()
    assert refusal.value.reason == "incomplete"


@pytest.mark.parametrize("stored", [b"[{\"content\": ", b"\xff\xfe", b"{\"content\": \"not a list\"}"],
                         ids=["torn", "undecodable", "not-a-list"])
def test_blocks_that_exist_but_do_not_parse_are_an_incomplete_read(tmp_path, stored):
    from ouroboros.context import _scratchpad_for_context

    view = nano_host_publication._first_start_view(tmp_path, tmp_path / "repo")
    view.scratchpad_blocks_path().parent.mkdir(parents=True)
    view.scratchpad_path().write_text("Legacy Markdown", encoding="utf-8")
    failures = []
    assert _scratchpad_for_context(view, read_failures=failures) == ("memory/scratchpad.md", "Legacy Markdown")
    assert not failures  # legacy without a sidecar is known input, not a lost block list
    view.scratchpad_blocks_path().write_bytes(stored)
    assert _scratchpad_for_context(view, read_failures=failures) == ("memory/scratchpad.md", "Legacy Markdown")
    assert failures == ["scratchpad"]
    assert not Path(str(view.scratchpad_blocks_path()) + '.lock').exists()


def test_blocks_whose_journal_the_next_start_recreates_are_priced_with_its_reader(tmp_path):
    from ouroboros.context import _scratchpad_for_context
    from ouroboros.memory import Memory

    view = nano_host_publication._first_start_view(tmp_path, tmp_path / "repo")
    view.scratchpad_blocks_path().parent.mkdir(parents=True)
    view.scratchpad_blocks_path().write_text(json.dumps([{"content": "A synthetic block."}]), encoding="utf-8")
    failures = []
    fresh = _scratchpad_for_context(view, read_failures=failures)
    assert "A synthetic block." in fresh[1] and "scratchpad_journal.jsonl" in fresh[1]
    assert not failures and not view.journal_path().exists()
    assert not Path(str(view.scratchpad_blocks_path()) + '.lock').exists()
    # The real root renders only the reader, never this separate journal's payload.
    view.journal_path().write_bytes(b'not a journal record')
    assert fresh == _scratchpad_for_context(view)
    assert not Path(str(view.scratchpad_blocks_path()) + '.lock').exists()
    view.journal_path().write_text("", encoding="utf-8")
    assert fresh == _scratchpad_for_context(Memory(drive_root=tmp_path, repo_dir=tmp_path / "repo"))


def test_block_reader_io_failure_is_not_a_repairable_parse_problem(tmp_path):
    from ouroboros.context import _scratchpad_for_context

    view = nano_host_publication._first_start_view(tmp_path, tmp_path / "repo")
    view.scratchpad_blocks_path().mkdir(parents=True)  # exists but cannot be read as a regular file
    view.scratchpad_path().write_text("Known flat projection", encoding="utf-8")
    with pytest.raises(NanoPublicationRefused) as refusal:
        _scratchpad_for_context(view)
    assert refusal.value.reason == "incomplete" and "IsADirectoryError" in refusal.value.detail
    assert not Path(str(view.scratchpad_blocks_path()) + '.lock').exists()


def test_publication_does_not_activate_a_missing_chronicle(tmp_path):
    with pytest.raises(NanoPublicationRefused, match="bootstrap_pending"):
        nano_host_publication._root_counts(tmp_path, {}, ())
    assert not list(tmp_path.iterdir())


def test_a_refused_map_edit_keeps_the_map_its_measured_block_and_its_history(tmp_path):
    from contextlib import contextmanager

    from ouroboros.memory_publication import publish_text
    from ouroboros.memory_texts import world_source

    source = world_source(tmp_path)
    source.path.parent.mkdir(parents=True)
    original = b"# My World\n<!-- measured -->\nFixture block.\n<!-- /measured -->\nThe old map.\n"
    source.path.write_bytes(original)

    class Refusing:
        @contextmanager
        def ordinary(self):
            yield False

        @contextmanager
        def publication(self, change):
            assert change.path == source.path and change.before == original and b"A larger map." in change.after
            raise NanoPublicationRefused("over_capacity", "synthetic background growth")
            yield

    result = publish_text(source, hashlib.sha256(original).hexdigest(), lambda old: old + "A larger map.\n",
                          author={"task_id": "t", "focus": {}, "route": {}}, source_type="task", mode="overwrite",
                          nano_admission=Refusing())
    assert (result.ok, result.reason) == (False, "nano_over_capacity")
    assert source.path.read_bytes() == original and not Path(source.journal).exists()


def test_a_registry_edit_is_shown_from_its_own_bytes_and_a_refusal_keeps_the_registry(tmp_path):
    from contextlib import contextmanager

    from ouroboros.context_sections import _build_registry_section
    from ouroboros.memory_publication import publish_memory_file
    from ouroboros.nano_publication import _TEXT_PREVIEW

    env = SimpleNamespace(drive_path=lambda rel: (tmp_path / rel).resolve())
    path = (tmp_path / "memory" / "registry.md").resolve()
    assert _build_registry_section(env) == ""
    token = _TEXT_PREVIEW.set({path: b"### fixture-source\nA proposed entry.\n"})
    try:  # the count shows the proposal even before the file exists
        assert "A proposed entry." in str(_build_registry_section(env))
    finally:
        _TEXT_PREVIEW.reset(token)

    class Refusing:
        @contextmanager
        def ordinary(self):
            yield False

        @contextmanager
        def publication(self, change):
            assert change.path == path and change.before is None and b"A larger entry." in change.after
            raise NanoPublicationRefused("over_capacity", "synthetic background growth")
            yield

    journal = path.with_name("registry_history.jsonl")
    result = publish_memory_file(path, "", lambda old: old + "### fixture-source\nA larger entry.\n", journal=journal,
                                 author={"task_id": "t", "focus": {}, "route": {}}, source_type="memory_update_registry",
                                 nano_admission=Refusing())
    assert (result.ok, result.reason) == (False, "nano_over_capacity")
    assert not path.exists() and not journal.exists()


# --- the witness ----------------------------------------------------------------------------------

def test_a_reset_or_a_lost_source_withdraws_the_witness(tmp_path, monkeypatch):
    from ouroboros import deployment_policy, launch_roots, local_model, server_process

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(local_model, "local_dispatch_port", lambda: 8767)
    monkeypatch.setattr(deployment_policy, "current", lambda: None)
    monkeypatch.setattr(server_process, "read_service_bindings", lambda root: {})
    serving = {"confirmed": True, "context_window": 80_000, "process_id": 7, "port": 8767}
    monkeypatch.setattr(nano_host_publication, "_qualified", lambda: ({}, serving))
    status = {"value": "missing"}
    monkeypatch.setattr(launch_roots, "read_witness", lambda root: (status["value"], {}))
    identity = tmp_path / "memory" / "identity.md"
    identity.parent.mkdir(parents=True)
    identity.write_text("A synthetic identity.", encoding="utf-8")
    witness = nano_host_publication._profile_witness(tmp_path)
    assert witness is not None and witness == nano_host_publication._profile_witness(tmp_path)
    identity.unlink()
    assert nano_host_publication._profile_witness(tmp_path) != witness
    status["value"] = launch_roots.RESETTING
    with pytest.raises(NanoPublicationRefused, match="unavailable"):
        nano_host_publication._profile_witness(tmp_path)


# --- one check for every editor of the rules ------------------------------------------------------

def test_the_settings_framing_leaves_out_the_rules_every_secret_value_and_explicit_defaults():
    framing = nano_host_publication._framing_of
    base = {RULES: "old", "OUROBOROS_CONTEXT_MODE": "nano", "OPENROUTER_API_KEY": "sk-fixture-one"}
    assert framing(base) == framing({**base, RULES: "new"})  # the rules are counted memory
    assert framing(base) != framing({**base, "OUROBOROS_CONTEXT_MODE": "max"})
    assert framing(base) == framing({**base, "OPENROUTER_API_KEY": "sk-fixture-two"})  # a secret is its presence
    assert framing(base) != framing({**base, "OPENROUTER_API_KEY": ""})
    # A rules save that writes the whole document puts defaults on disk; that is no other change.
    assert framing(base) == framing({**base, "LOCAL_MODEL_PORT": config.SETTINGS_DEFAULTS["LOCAL_MODEL_PORT"]})


def test_a_rules_save_that_also_changes_other_settings_records_no_observation(tmp_path):
    state = {"identity": 1, "framing": "counted"}
    pair = (_floor(_sample(30_000)), _floor(_sample(30_000)))
    gate = NanoPublication(tmp_path, lambda: (state["identity"], state["framing"], "serving"), lambda changes: pair,
                           None, profile_key=lambda: "launch:x")
    rules = MemoryChange(None, b"old", b"new", subject="owner_rules")
    marker = tmp_path / PUBLICATION_REVISION
    with gate.publication(rules):
        state["identity"] = 2  # only the rules landed: the document's identity moved
    assert json.loads(marker.read_bytes())["proof"]["background_tokens"] == 30_000
    with gate.publication(rules):
        state["identity"], state["framing"] = 3, "another setting changed too"
    row = json.loads(marker.read_bytes())
    assert "proof" not in row and row["checked_profile"] == "launch:x"  # the old measurement is withdrawn

def test_a_rules_commit_carries_only_the_count_taken_for_exactly_these_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    qualified = {"value": {}}
    monkeypatch.setattr(nano_host_publication, "_current_profile", lambda: qualified["value"])
    ticket = SimpleNamespace(changes=(MemoryChange(None, b"old", b"new", subject="owner_rules"),))
    commit = nano_host_publication.rules_commit_ticket
    assert commit(tmp_path, {RULES: "old"}, {RULES: "new"}, ticket) is ticket
    assert commit(tmp_path, {RULES: "old"}, {RULES: "old"}, ticket) is None  # a count of other text never lands
    with pytest.raises(NanoPublicationRefused, match="stale"):
        commit(tmp_path, {RULES: "old"}, {RULES: "other"}, ticket)
    with pytest.raises(NanoPublicationRefused, match="stale"):
        commit(tmp_path, {RULES: "old"}, {RULES: "new"}, None)  # a qualified profile's rules are never uncounted
    assert commit(tmp_path, {}, {RULES: ""}, None) is None
    qualified["value"] = None
    assert commit(tmp_path, {RULES: "old"}, {RULES: "new"}, None) is None  # Max: an ordinary write, fenced


def test_only_a_cyber_save_that_authors_the_rules_is_counted_before_the_document_lock(monkeypatch, tmp_path):
    counted = []
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(nano_host_publication, "rules_save_ticket", lambda root, rules: counted.append(rules) or "ticket")
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "advanced")
    assert nano_host_publication.settings_body_rules_ticket({RULES: "new"}) is None  # the merge ignores the key here
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "cyber_pro")
    assert nano_host_publication.settings_body_rules_ticket({"OUROBOROS_CONTEXT_MODE": "nano"}) is None
    assert nano_host_publication.settings_body_rules_ticket({RULES: "new"}) == "ticket" and counted == ["new"]


def test_a_nano_refusal_is_a_typed_unsaved_settings_answer():
    from ouroboros.gateway.owner_settings import settings_write_refusal

    response = settings_write_refusal(NanoPublicationRefused("stale", "these rules were not checked"))
    body = json.loads(response.body)
    assert (response.status_code, body["code"], body["saved"]) == (409, "nano_publication_stale", False)
    assert body["nano_reason"] == "these rules were not checked"


# --- a mandatory receipt changes the visible background ------------------------------------------

def test_a_fallback_refusal_receipt_withdraws_the_observation_without_a_count(tmp_path):
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.memory_inventory import FALLBACK_REFUSAL_PREFIX, fallback_refusals

    store = ChronicleStore(tmp_path)
    assert store.ensure_activated()["kind"] == "activation"
    marker = tmp_path / PUBLICATION_REVISION
    _invalidate_publication_reads(marker, checked_profile="launch:x", checked_saved="saved-x")
    marker.write_text(json.dumps({**json.loads(marker.read_bytes()), "proof": {"background_tokens": 30_000}}))
    result = store.publish([], scan_state={FALLBACK_REFUSAL_PREFIX + "unit": {"kind": "invalid", "at": "fixture"}})
    assert result.ok and fallback_refusals(store.scan_state())["unit"]["kind"] == "invalid"
    row = json.loads(marker.read_bytes())
    assert "proof" not in row and (row["checked_profile"], row["checked_saved"]) == ("launch:x", "saved-x")


# --- filesystems without kernel file locks --------------------------------------------------------

@pytest.mark.parametrize("lock_errno", [
    pytest.param("ENOLCK", id="enolck"),
    pytest.param("EOPNOTSUPP", id="eopnotsupp"),
])
def test_name_tier_storage_preserves_max_settings_save_and_nano_serialization(tmp_path, monkeypatch, lock_errno):
    import errno
    from threading import Event
    from ouroboros import platform_layer

    code = getattr(errno, lock_errno)
    # A leftover kernel-lock placeholder from an earlier mount must not block the name-tier lock.
    (tmp_path / "memory").mkdir(parents=True)
    (tmp_path / "memory" / ".nano-publication.lock").write_bytes(b"")

    def unsupported(_fd):
        raise OSError(code, "kernel file locks unsupported on this mount")

    monkeypatch.setattr(platform_layer, "_KERNEL_LOCK_TIER", {})
    monkeypatch.setattr(platform_layer, "file_lock_exclusive_nb", unsupported)

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "advanced")
    initial = {"OUROBOROS_RUNTIME_MODE": "advanced", "OUROBOROS_CONTEXT_MODE": "max"}
    (tmp_path / "settings.json").write_text(json.dumps(initial), encoding="utf-8")

    marker = tmp_path / PUBLICATION_REVISION
    _invalidate_publication_reads(marker, checked_profile="launch:x")
    marker.write_text(json.dumps({**json.loads(marker.read_bytes()), "proof": {"background_tokens": 30_000}}))

    # Max Settings save updates an ordinary setting across settings_publication_boundary -> publication_fence.
    config.save_settings({**initial, "OUROBOROS_IMAGE_INPUT_MODE": "native"})
    assert json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))["OUROBOROS_IMAGE_INPUT_MODE"] == "native"
    assert not platform_layer.kernel_file_locks_enforced(tmp_path / "memory" / ".nano-publication.lock")
    assert "proof" not in json.loads(marker.read_bytes())

    # Writers that selected the name tier serialize and reject stale witnesses on .nano-publication.name.lock.
    pair = (_floor(_sample(30_000, formatter=FORMATTER)), _floor(_sample(30_000, 5_100, formatter=FORMATTER)))
    gate = NanoPublication(tmp_path, lambda: "fixed", lambda _changes: pair, None, profile_key=lambda: "launch:x")
    ticket = gate.prepare(MemoryChange(None, None, b"proposal", subject="synthetic"))

    ready, release, outcomes = Event(), Event(), []

    def hold_fence():
        with publication_fence(tmp_path):
            ready.set()
            assert release.wait(5)

    def contender():
        assert ready.wait(5)
        try:
            with ticket.commit():
                outcomes.append("committed")
        except NanoPublicationRefused as refusal:
            outcomes.append(refusal.reason)

    holder = Thread(target=hold_fence, daemon=True)
    waiter = Thread(target=contender, daemon=True)
    holder.start()
    waiter.start()
    assert ready.wait(5)
    waiter.join(0.1)
    assert waiter.is_alive() and not outcomes  # blocked behind the live name-tier holder, never lockless
    release.set()
    holder.join(5)
    waiter.join(5)
    assert not holder.is_alive() and not waiter.is_alive()
    assert outcomes == ["stale"]  # the fence invalidated the witness before releasing

    with gate.publication(MemoryChange(None, None, b"fresh", subject="synthetic")):
        pass
    assert json.loads(marker.read_bytes())["proof"]["background_tokens"] == 30_000


def test_unsupported_lock_errors_and_unreadable_identities_fail_closed(tmp_path, monkeypatch):
    import errno
    import os
    from ouroboros import platform_layer

    kept = []
    pair = (_floor(_sample(30_000, formatter=FORMATTER)), _floor(_sample(30_000, 5_100, formatter=FORMATTER)))
    edit = MemoryChange(None, None, b"synthetic", subject="synthetic")

    def fail_lock(code):
        def _raise(_fd):
            raise OSError(code, "synthetic lock error")
        return _raise

    def make_gate(root):
        return NanoPublication(root, lambda: "fixed", lambda _changes: pair,
                               lambda changes: kept.append(changes) or "kept", profile_key=lambda: "launch:x")

    # 1. Non-name-tier kernel lock errors (e.g. EIO) stay on the enforced tier and fail closed.
    with monkeypatch.context() as mp:
        eio = fail_lock(errno.EIO)
        gate = make_gate(tmp_path / "eio")
        mp.setattr(platform_layer, "_KERNEL_LOCK_TIER", {})
        mp.setattr(platform_layer, "file_lock_exclusive_nb", eio)
        with pytest.raises(OSError):
            with publication_fence(tmp_path / "eio"):
                pytest.fail("EIO must not bypass the publication fence")
        with pytest.raises(NanoPublicationRefused) as refusal:
            gate.prepare(edit)
        assert refusal.value.reason == "unavailable" and kept == [(edit,)]

    # 2. A live lock error after the directory probed as enforced never downgrades mid-flight.
    with monkeypatch.context() as mp:
        enforced_root = tmp_path / "enforced"
        mp.setattr(platform_layer, "_KERNEL_LOCK_TIER", {})
        (enforced_root / "memory").mkdir(parents=True)
        assert platform_layer.kernel_file_locks_enforced(enforced_root / "memory" / ".nano-publication.lock")
        mp.setattr(platform_layer, "file_lock_exclusive_nb", fail_lock(errno.ENOLCK))
        with pytest.raises(OSError):
            with publication_fence(enforced_root):
                pytest.fail("live lock failure on an enforced directory must fail closed")

    # 3. Name-tier storage whose _lock_identity returns () refuses cleanly via identity_unreadable.
    with monkeypatch.context() as mp:
        kept.clear()
        name_root = tmp_path / "name-tier"
        mp.setattr(platform_layer, "_KERNEL_LOCK_TIER", {})
        mp.setattr(platform_layer, "file_lock_exclusive_nb", fail_lock(errno.EOPNOTSUPP))
        ticket = make_gate(name_root).prepare(edit)
        mp.setattr(platform_layer, "_lock_identity", lambda _target: ())
        with pytest.raises(OSError, match="identity_unreadable"):
            with publication_fence(name_root):
                pytest.fail("unproven inode identity must not enter the fence")
        assert not (name_root / "memory" / ".nano-publication.name.lock").exists()
        with pytest.raises(NanoPublicationRefused) as id_refusal, ticket.commit():
            pytest.fail("unproven inode identity must not commit")
        assert id_refusal.value.reason == "unavailable" and kept == [(edit,)]

    # 4. Live contention past timeout refuses and retains the proposal instead of committing unlocked.
    with monkeypatch.context() as mp:
        kept.clear()
        contended_root = tmp_path / "contended"
        mp.setattr(platform_layer, "_KERNEL_LOCK_TIER", {})
        mp.setattr(platform_layer, "file_lock_exclusive_nb", fail_lock(errno.EOPNOTSUPP))
        contended_ticket = make_gate(contended_root).prepare(edit)
        mp.setattr(nano_publication, "_NAME_LOCK_TIMEOUT_SEC", 0.05)
        lock_file = contended_root / "memory" / ".nano-publication.name.lock"
        lock_file.write_text(f"pid={os.getpid()} ts=9999999999\n", encoding="utf-8")
        with pytest.raises(NanoPublicationRefused) as busy_refusal, contended_ticket.commit():
            pytest.fail("contended name lock must not commit unlocked")
        assert busy_refusal.value.reason == "unavailable" and "TimeoutError" in busy_refusal.value.detail
        assert kept == [(edit,)]


@pytest.mark.parametrize("enforced_holder", [False, True])
def test_nano_publication_tiers_exclude_on_one_name(tmp_path, monkeypatch, enforced_holder):
    from threading import Event, Thread, current_thread
    from ouroboros import platform_layer

    ready, release = Event(), Event()
    outcomes = []

    def policy(_path):
        return enforced_holder if current_thread() is holder else not enforced_holder

    def hold():
        with publication_fence(tmp_path):
            ready.set()
            assert release.wait(5)

    holder = Thread(target=hold, daemon=True)
    monkeypatch.setattr(platform_layer, 'kernel_file_locks_enforced', policy)
    monkeypatch.setattr(nano_publication, 'kernel_file_locks_enforced', policy, raising=False)
    monkeypatch.setattr(nano_publication, '_NAME_LOCK_TIMEOUT_SEC', 0.05)
    holder.start()
    try:
        assert ready.wait(5)
        with pytest.raises(TimeoutError), publication_fence(tmp_path):
            pytest.fail('A differing probe entered another writer’s critical section')
    finally:
        release.set()
        holder.join(5)
    assert not holder.is_alive()
    with publication_fence(tmp_path):
        outcomes.append('entered after release')
    assert outcomes == ['entered after release']


def test_world_growth_cannot_escape_into_its_reader_or_grow_behind_it_while_repairs_pass(tmp_path):
    def world_floor(background, *, variant="active", world_form="whole", world_digest="w1"):
        sample = _sample(background, 15_000, formatter=FORMATTER, world_form=world_form)
        return SimpleNamespace(count=lambda: sample, variant=variant, instruction_digest=lambda: "same",
                               catalogue_digest=lambda: "same", world_digest=lambda: world_digest)

    def run(root, floors):
        gate = NanoPublication(root, lambda: "fixed", lambda _changes: floors, None, profile_key=lambda: "launch:x")
        with gate.publication(MemoryChange(None, None, b"synthetic", subject="synthetic")):
            pass

    root = tmp_path / "repair"
    run(root, (world_floor(60_000), world_floor(52_000, world_digest="w2"),
               world_floor(61_000, variant="alternate"), world_floor(53_000, variant="alternate", world_digest="w2")))
    assert "proof" not in json.loads((root / PUBLICATION_REVISION).read_bytes())

    with pytest.raises(NanoPublicationRefused, match="over_capacity"):
        run(root, (world_floor(52_000), world_floor(52_000),
                   world_floor(53_000, variant="alternate"), world_floor(54_000, variant="alternate")))
    with pytest.raises(NanoPublicationRefused, match="publication count must use whole WORLD"):
        run(root, (world_floor(52_000), world_floor(52_000),
                   world_floor(31_600, variant="alternate", world_form="reader"),
                   world_floor(31_600, variant="alternate", world_form="reader")))

    run(root, (world_floor(52_000), world_floor(36_500, world_digest="w2")))
    assert "proof" not in json.loads((root / PUBLICATION_REVISION).read_bytes())
    run(root, (world_floor(36_500, world_digest="w2"), world_floor(36_500, world_digest="w2")))
    proof = json.loads((root / PUBLICATION_REVISION).read_bytes())["proof"]
    assert (proof["world_form"], proof["world_digest"], proof["background_tokens"]) == ("whole", "w2", 36_500)


def test_fresh_exact_count_clears_stale_counter_unavailable_without_reviving_protection(tmp_path, monkeypatch):
    from ouroboros import local_model
    from ouroboros.nano_publication import read_root_sample, record_root_sample

    serving = {"confirmed": True, "context_window": 80_000, "process_id": 7, "port": 8767}
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(local_model, "local_dispatch_port", lambda: 8767)
    monkeypatch.setattr(local_model, "get_manager", lambda **_: SimpleNamespace(
        serving_context_evidence=lambda **_: serving))
    monkeypatch.setattr(nano_host_publication, "_qualified", lambda: ({**_saved(), "OUROBOROS_CONTEXT_MODE": "nano"}, serving))
    monkeypatch.setattr(nano_host_publication, "_served_launch", lambda _serving: nano_host_publication._launch_of(HF))
    monkeypatch.setattr(nano_host_publication, "_profile_witness", lambda _root: ("s", "f", 7, 8767))
    monkeypatch.setattr(nano_host_publication, "current_rules_digest", lambda: "rules-fixture")

    key = nano_host_publication._profile_key()
    marker = tmp_path / PUBLICATION_REVISION
    marker.parent.mkdir(parents=True)

    def mark_outage():
        _invalidate_publication_reads(marker, checked_profile=key, counter_unavailable=True)
        status = read_root_sample(tmp_path, provider="local")
        assert (status["status"], status["reason"]) == ("not_claimed", "confirmed_counter_unavailable")

    # 1. NanoPublication.prepare clears stale outage on an exact baseline even when the candidate exceeds capacity.
    mark_outage()
    over_pair = (_floor(_sample(36_000, 20_000, formatter=FORMATTER)),
                 _floor(_sample(36_000, 20_500, formatter=FORMATTER)))
    gate = NanoPublication(tmp_path, lambda: ("s", "f", 7, 8767), lambda _c: over_pair, None,
                           profile_key=lambda: key, profile_stamp=nano_host_publication.profile_stamp)
    with pytest.raises(NanoPublicationRefused, match="over_capacity"):
        gate.prepare(MemoryChange(None, None, b"over", subject="synthetic"))
    row = json.loads(marker.read_bytes())
    assert "reason" not in row and "proof" not in row
    assert read_root_sample(tmp_path, provider="local")["reason"] != "confirmed_counter_unavailable"
    assert read_root_sample(tmp_path, provider="local")["status"] == "not_claimed"

    # 2. record_root_sample clears stale outage without fabricating a write-protection proof.
    mark_outage()
    record_root_sample(tmp_path, {
        **_sample(36_000, 19_000, formatter=FORMATTER),
        "scope": "cold_start", "room": "main", "counter_confirmed": True, "port": 8767,
        "rules_sha256": "rules-fixture", "minimum": {"input_tokens": 55_000},
        "minimum_reply_allowance_tokens": REPLY,
        "minimum_work_headroom_at_wire_allowance_tokens": 80_000 - 55_000 - REPLY,
        "actual": {"sent_max_tokens": REPLY, "input_tokens": 56_000},
    })
    row = json.loads(marker.read_bytes())
    assert "reason" not in row and "proof" not in row
    sampled = read_root_sample(tmp_path, provider="local")
    assert sampled["status"] == "not_claimed" and sampled["reason"] == "publication_scope_unqualified"

    # 3. measure_publication_floor clears stale outage without fabricating a write-protection proof.
    mark_outage()
    exact_floor = _floor({**_sample(36_000, 19_000, formatter=FORMATTER), "counter_confirmed": True})
    monkeypatch.setattr(nano_host_publication, "host_admission", lambda _root, **_: SimpleNamespace(
        revision=lambda: ("s", "f", 7, 8767), capture=lambda _changes: (exact_floor,)))
    measured = nano_host_publication.measure_publication_floor(tmp_path)
    assert measured["status"] == "exact"
    row = json.loads(marker.read_bytes())
    assert "reason" not in row and "proof" not in row
    assert nano_host_publication.current_protection(tmp_path) == {}
