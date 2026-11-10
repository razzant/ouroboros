"""A secret save's recorded outcome stays truthful at its edges.

Through the real ``POST /api/decisions/secret`` over a real task store and Settings file (the
``tests.test_secret_hitl`` fixtures): a Settings document that exists but cannot be read is unknown to
the form and is never written as if it were new; an attempt whose outcome stayed unknown is never
repeated under its own id, however many newer attempts followed it; readers that could not be
discovered are recorded as unknown, never as "no skill reads it", and the save stays a save; the
asking task's next selected process receives what is saved, never its admission snapshot.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import threading
import time

import pytest

from ouroboros import secret_requests as sr
from tests.test_secret_hitl import (
    CANARY,
    CANARY_B,
    TASK,
    _ask,
    _assert_absent,
    _block,
    _cancel,
    _mail,
    _observed,
    _put_settings,
    _save,
    _settings,
    _status,
)
from tests.test_secret_hitl import env as env  # the shared real task store, Settings file and ingress

CANARY_C = "Third·canary·entry-17"


def _answers(env):
    return [entry for entry in _mail(env) if entry.get("kind") == "quiz_answer"]


# ------------------------------------------------------- an unreadable Settings document

@pytest.mark.parametrize("damage", ["undecodable", "unreadable"])
def test_an_unreadable_settings_document_is_unknown_to_the_form_and_never_overwritten(env, monkeypatch, damage):
    """The form read ``configured: false`` from a document it could not read, and Save then wrote the
    value over it, losing the old key and every unrelated setting."""
    from ouroboros import config as cfg

    _put_settings(env, FAKEPLUG_TOKEN="old-synthetic", UNRELATED_SETTING="keep")
    readable = cfg.SETTINGS_PATH.read_bytes()
    quiz_id = _ask(env)
    real_open = pathlib.Path.open
    if damage == "undecodable":
        cfg.SETTINGS_PATH.write_bytes(readable[:-1])  # a hand edit that lost its closing brace
    damaged = cfg.SETTINGS_PATH.read_bytes()
    if damage == "unreadable":  # injected: chmod is not a refusal on every platform

        def refused(path, *args, **kwargs):
            if path == cfg.SETTINGS_PATH:
                raise PermissionError(13, "injected: settings unreadable")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "open", refused)
    status = _status(env, quiz_id).json()
    response = _save(env, quiz_id, {"botToken": CANARY}, observed={"botToken": False})  # what an empty read shows
    monkeypatch.setattr(pathlib.Path, "open", real_open)
    assert cfg.SETTINGS_PATH.read_bytes() == damaged, "the unreadable document was written over"
    assert status["keys_unknown"] is True and "keys" not in status, status
    body = response.json()
    assert response.status_code == 409 and body["reason_code"] == "settings_unreadable", body
    assert body["saved"] is False and body["keys_unknown"] is True
    block = _block(env, quiz_id)
    assert block["state"] == "open" and "secret_save" not in block and not _mail(env)
    # Repaired, the form reads the key as set and only an explicit Replace overwrites it.
    cfg.SETTINGS_PATH.write_bytes(readable)
    assert _observed(env, quiz_id) == {"botToken": True}
    replaced = _save(env, quiz_id, {"botToken": CANARY}, request_id="after-repair")
    assert replaced.status_code == 200 and replaced.json()["secret_outcome"]["event"] == "secret_replaced"
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY and _settings(env)["UNRELATED_SETTING"] == "keep"
    _assert_absent(env, CANARY, texts=[response.text, replaced.text])


# ------------------------------------------------------- readers that could not be read

def test_readers_that_could_not_be_discovered_are_recorded_unknown_and_the_save_stays_saved(env, monkeypatch):
    from tests import _dialogue

    quiz_id = _ask(env)
    observed = _observed(env, quiz_id)

    def undiscoverable(*_args, **_kwargs):
        raise OSError("injected: the skills tree could not be read")

    monkeypatch.setattr(sr, "key_consumers", undiscoverable)
    response = _save(env, quiz_id, {"botToken": CANARY}, observed=observed)
    body = response.json()
    assert response.status_code == 200 and body["saved"] is True and body["state"] == "answered", body
    outcome = _block(env, quiz_id)["secret_outcome"]
    assert body["secret_outcome"] == outcome
    assert outcome.get("consumers_unknown") is True and "consumers" not in outcome, outcome
    [mail] = _answers(env)
    [row] = [row for row in _dialogue.chat_rows(env.root) if row.get("type") == "quiz_answer"]
    for text in (mail["text"], row["text"]):  # the model's frame and the history row
        assert "No installed skill reads" not in text
        assert "Which installed skills read these keys could not be read" in text
    _assert_absent(env, CANARY, texts=[response.text])


@pytest.fixture
def unlistable_skills(env):
    """A skills bucket the process can search but not list (a real permission fault, mode -wx): it
    still resolves a skill by path, so only a directory LISTING fails."""
    import os
    import stat

    bucket = env.root / "skills" / "external"
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        pytest.skip("directory permissions do not bind this process")
    mode = stat.S_IMODE(bucket.stat().st_mode)

    class Fault:
        def apply(self):
            bucket.chmod(0o300)
            try:
                list(bucket.iterdir())
            except OSError:
                return self
            bucket.chmod(mode)
            pytest.skip("this filesystem does not enforce directory permissions")

        def __enter__(self):
            return self.apply()

        def __exit__(self, *_exc):
            bucket.chmod(mode)

    fault = Fault()
    yield fault
    bucket.chmod(mode)


def test_an_unlistable_skills_directory_after_the_commit_is_unknown_readers_and_the_save_stays_saved(
        env, monkeypatch, unlistable_skills):
    from ouroboros.gateway import owner_settings
    from tests import _dialogue

    quiz_id = _ask(env)
    observed = _observed(env, quiz_id)
    real, reads = owner_settings._owner_settings_document, []

    def read_back_then_fault(*args, **kwargs):
        document = real(*args, **kwargs)
        if kwargs.get("strict") and not reads:  # the save's own read-back: Settings are committed
            reads.append(document)
            unlistable_skills.apply()
        return document

    monkeypatch.setattr(owner_settings, "_owner_settings_document", read_back_then_fault)
    response = _save(env, quiz_id, {"botToken": CANARY}, observed=observed)
    body = response.json()
    assert reads and reads[0].get("FAKEPLUG_TOKEN") == CANARY, "the fault came after the committed write"
    assert response.status_code == 200 and body["saved"] is True and body["state"] == "answered", body
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY, "the committed Settings write stands"
    outcome = _block(env, quiz_id)["secret_outcome"]
    assert outcome.get("consumers_unknown") is True and "consumers" not in outcome, outcome
    [mail] = _answers(env)
    [row] = [row for row in _dialogue.chat_rows(env.root) if row.get("type") == "quiz_answer"]
    for text in (mail["text"], row["text"]):
        assert "No installed skill reads" not in text and "could not be read" in text
    _assert_absent(env, CANARY, texts=[response.text])


def test_an_unlistable_skills_directory_is_unknown_readers_in_the_form_status_too(env, unlistable_skills):
    quiz_id = _ask(env)
    with unlistable_skills:
        facts = _status(env, quiz_id).json()
    assert facts["consumers_unknown"] is True and "consumers" not in facts, facts
    assert facts["keys"][0]["configured"] is False, "the key facts are independent of the readers' read"
    assert [row["name"] for row in _status(env, quiz_id).json()["consumers"]] == ["fakeplug"], "readable again: known rows"


def test_discovery_keeps_its_fail_soft_default_under_the_same_fault(env, unlistable_skills):
    from ouroboros import skill_loader

    with unlistable_skills:
        assert skill_loader.discover_skills(env.root) == []
        with pytest.raises(OSError):
            skill_loader.discover_skills(env.root, strict=True)
    assert [skill.name for skill in skill_loader.discover_skills(env.root, strict=True)] == ["fakeplug"]


@pytest.mark.parametrize("readers", ["none", "grant_unknown"])
def test_no_readers_and_an_unreadable_grant_stay_distinct_known_facts(env, monkeypatch, readers):
    from ouroboros import skill_loader

    quiz_id = _ask(env)
    observed = _observed(env, quiz_id)
    if readers == "none":
        monkeypatch.setattr(sr, "key_consumers", lambda *_args: [])
    else:

        def grant_unreadable(*_args, **_kwargs):
            raise OSError("injected: grants unreadable")

        monkeypatch.setattr(skill_loader, "grant_status_for_skill", grant_unreadable)
    outcome = _save(env, quiz_id, {"botToken": CANARY}, observed=observed).json()["secret_outcome"]
    [mail] = _answers(env)
    assert "consumers_unknown" not in outcome
    if readers == "none":
        assert outcome["consumers"] == [] and "No installed skill reads these keys." in mail["text"]
    else:
        assert outcome["consumers"] == [{"name": "fakeplug", "enabled": False, "reads": ["FAKEPLUG_TOKEN"],
                                         "grant_unknown": True}]
        assert "fakeplug (disabled; its grant could not be read)" in mail["text"]


# ------------------------------------------------------- unconfirmed attempts never replay

def _unconfirmed(env, monkeypatch, quiz_id, request_id, value):
    """A save whose Settings write landed but whose completion was never recorded: its outcome is unknown."""
    real = sr.finalize_save

    def store_unavailable(*_args, **_kwargs):
        raise RuntimeError("store unavailable after the settings commit")

    monkeypatch.setattr(sr, "finalize_save", store_unavailable)
    try:
        response = _save(env, quiz_id, {"botToken": value}, request_id=request_id)
    finally:
        monkeypatch.setattr(sr, "finalize_save", real)
    assert response.status_code == 500 and response.json()["reason_code"] == "secret_completion_unrecorded"


def test_an_older_unconfirmed_request_never_replays_after_newer_ones(env, monkeypatch):
    """A, then B, both of unknown outcome: an old A arriving last passed as a new request and wrote its
    value over B's, although A was never to be repeated."""
    quiz_id = _ask(env)
    _unconfirmed(env, monkeypatch, quiz_id, "A", CANARY)
    _unconfirmed(env, monkeypatch, quiz_id, "B", CANARY_B)
    replay = _save(env, quiz_id, {"botToken": CANARY}, request_id="A")
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY_B, "an older unknown attempt wrote over a newer one"
    assert replay.status_code == 409 and replay.json()["reason_code"] == "secret_save_unconfirmed", replay.text
    assert replay.json()["saved"] is None
    # Not only the adjacent one: after a third unknown attempt no earlier id passes; a new id does.
    _unconfirmed(env, monkeypatch, quiz_id, "C", CANARY_C)
    for old, value in (("A", CANARY), ("B", CANARY_B), ("C", CANARY_C)):
        refused = _save(env, quiz_id, {"botToken": value}, request_id=old)
        assert refused.status_code == 409 and refused.json()["reason_code"] == "secret_save_unconfirmed", old
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY_C and not _answers(env)
    assert _block(env, quiz_id)["secret_save"]["unconfirmed_request_ids"] == ["A", "B"]
    saved = _save(env, quiz_id, {"botToken": CANARY}, request_id="D")
    assert saved.status_code == 200 and saved.json()["secret_outcome"]["event"] == "secret_replaced"
    assert _block(env, quiz_id)["secret_save"]["unconfirmed_request_ids"] == ["A", "B", "C"]
    # A confirmed save replays its receipt; an old unknown id is now simply settled.
    duplicate = _save(env, quiz_id, {"botToken": CANARY}, request_id="D", observed={"botToken": True})
    assert duplicate.status_code == 200 and duplicate.json()["duplicate"] is True
    late = _save(env, quiz_id, {"botToken": CANARY_B}, request_id="B", observed={"botToken": True})
    assert late.status_code == 409 and late.json()["reason_code"] == "secret_request_settled"
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY and len(_answers(env)) == 1
    _assert_absent(env, CANARY, CANARY_B, CANARY_C, texts=[replay.text, saved.text, duplicate.text, late.text])


def test_a_refusal_before_the_write_stays_retryable_over_an_unknown_attempt(env, monkeypatch):
    quiz_id = _ask(env)
    _unconfirmed(env, monkeypatch, quiz_id, "A", CANARY)
    stale = _save(env, quiz_id, {"botToken": CANARY_B}, request_id="B", observed={"botToken": False})
    assert stale.status_code == 409 and stale.json()["reason_code"] == "secret_key_state_changed"
    assert _block(env, quiz_id)["secret_save"]["request_id"] == "A", "B saved nothing; A's unknown claim stays"
    retry = _save(env, quiz_id, {"botToken": CANARY_B}, request_id="B")
    assert retry.status_code == 200 and _settings(env)["FAKEPLUG_TOKEN"] == CANARY_B
    assert _block(env, quiz_id)["secret_save"]["unconfirmed_request_ids"] == ["A"]
    _assert_absent(env, CANARY, CANARY_B, texts=[stale.text, retry.text])


def test_a_cancel_after_unknown_attempts_settles_every_earlier_id(env, monkeypatch):
    quiz_id = _ask(env)
    _unconfirmed(env, monkeypatch, quiz_id, "A", CANARY)
    _unconfirmed(env, monkeypatch, quiz_id, "B", CANARY_B)
    body = _cancel(env, quiz_id).json()
    assert body["state"] == "closed" and body["saved"] is None and body["secret_outcome"]["interrupted_save"] is True
    for old, value in (("A", CANARY), ("B", CANARY_B)):
        late = _save(env, quiz_id, {"botToken": value}, request_id=old, observed={"botToken": True})
        assert late.status_code == 409 and late.json()["reason_code"] == "secret_request_settled", old
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY_B


def test_a_concurrent_replay_of_an_unknown_id_never_races_a_new_save(env, monkeypatch):
    from ouroboros.gateway.secret_decision import save_secret

    quiz_id = _ask(env)
    _unconfirmed(env, monkeypatch, quiz_id, "A", CANARY)
    _unconfirmed(env, monkeypatch, quiz_id, "B", CANARY_B)
    decision, values = f"quiz:{TASK}:{quiz_id}", {"A": CANARY, "C": CANARY_C}
    results, gate = {}, threading.Barrier(len(values))

    def run(request_id):
        gate.wait()
        results[request_id] = save_secret(env.root, {"request_id": request_id, "decision_id": decision,
                                                     "values": {"botToken": values[request_id]},
                                                     "observed": {"botToken": True}})

    threads = [threading.Thread(target=run, args=(request_id,)) for request_id in values]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert results["C"][0] == 200, results
    assert results["A"][0] == 409 and results["A"][1]["reason_code"] in ("secret_save_unconfirmed",
                                                                         "secret_request_settled")
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY_C and len(_answers(env)) == 1


# ------------------------------------------------------- later readers

def test_later_memory_attributes_secret_outcomes_to_the_host_and_names_them(env):
    """Later work's memory view read the answer row without its card, so a host form outcome printed as
    the owner's words: ``[…; Owner; row:…] [secret request …] secret_filled: …``."""
    from ouroboros import chat_chain, memory_view
    from ouroboros.dialogue_provenance import row_class
    from ouroboros.memory_inventory import row_meta

    filled, cancelled = _ask(env), _ask(env)
    assert _save(env, filled, {"botToken": CANARY}).status_code == 200
    assert _cancel(env, cancelled).status_code == 200
    entries = [(address, row_meta(row), pos) for address, row, pos in chat_chain.iter_room_rows(env.root, 1)]
    outcomes = [row_class(meta, pos=pos) for _address, meta, pos in entries
                if meta.get("type") in ("quiz_answer", "quiz_closed")]
    assert [cls["author"]["kind"] for cls in outcomes] == ["host", "host"], outcomes
    lane1, lane2 = memory_view._lanes(env.root, entries, {})
    spoken = "\n".join(item["line"] for item in lane1)
    assert "secret request" not in spoken and "Owner" not in spoken
    [line] = [item["line"] for item in lane2 if item["task"] == TASK]
    assert f"; host; task {TASK}]" in line, line
    assert "secret request secret_filled: row:1@" in line and "secret request secret_cancelled: row:1@" in line
    assert CANARY not in line


def test_the_wake_reads_a_saved_secret_as_its_host_event_without_a_value(env):
    from ouroboros import consciousness_wake as wake

    quiz_id = _ask(env)
    assert _save(env, quiz_id, {"botToken": CANARY}).status_code == 200
    line = wake._answered_card_line(TASK, quiz_id, _block(env, quiz_id), now=time.time())
    assert "host outcome secret_filled: fakeplug, botToken" in line, line
    assert "answer text unavailable" not in line and CANARY not in line


# One row per selection reader: commands, scripts and run-kind verification share process_facts' (their
# per-tool wiring has its witness in test_workflow_process_environment.py); services resolve their own.
@pytest.mark.parametrize("tool", ["run_script", "start_service"])
def test_a_running_task_selects_what_the_form_saved_into_its_next_process(env, monkeypatch, tool):
    """The asking task need not restart to use the answer: what the form fills or replaces, or the owner clears or
    deletes, while it runs is what its next ``env_from_settings`` process receives at exec; the task's own
    next-task reads keep its admission snapshot, and egress masks the value the child actually received."""
    from ouroboros import config, subagent_runtime, workspace_executor
    from ouroboros.gateway.owner_settings import _owner_update_settings, fields_on_saved_document
    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.tools import services
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *args, **kwargs: (True, ""))  # an unrelated gate
    system, workspace, data = (env.root / name for name in ("system", "workspace", "tools"))
    for path in (system, workspace, data / "logs"):
        path.mkdir(parents=True)
    tools = ToolRegistry(system, data)
    tools.set_context(ctx := ToolContext(repo_dir=system, drive_root=data, workspace_root=workspace,
                                         workspace_mode="external", task_id=TASK))
    witness, rows = workspace / "witness", []
    program = ("import hashlib,os,pathlib; value=os.environ['TOKEN']; print('child received '+value, flush=True); "
               f"pathlib.Path({str(witness)!r}).write_text(hashlib.sha256(value.encode()).hexdigest())")

    def received():
        """The child's own digest of its TOKEN at exec; None when nothing ran."""
        witness.unlink(missing_ok=True)
        args = {"cwd": str(workspace), "env_from_settings": {"TOKEN": "FAKEPLUG_TOKEN"}}
        if tool == "run_script":
            args.update(script=program, interpreter=sys.executable)
        else:
            args.update(cmd=[sys.executable, "-c", program])
        rows.append(_execute_single_tool(tools, {"id": f"call-{len(rows)}", "function": {
            "name": tool, "arguments": json.dumps(args)}}, data / "logs", task_id=TASK))
        if tool == "start_service" and not rows[-1]["is_error"]:
            assert services._SERVICES[workspace_executor.service_key(ctx, "service")].proc.wait(timeout=10) == 0
        return witness.read_text(encoding="utf-8") if witness.exists() else None

    def digest(value):
        return hashlib.sha256(value.encode()).hexdigest()

    _put_settings(env, FAKEPLUG_TOKEN="synthetic-old-token", OUROBOROS_EFFORT_TASK="high")
    try:
        with config.task_settings_scope(subagent_runtime.apply_task_start_settings()):
            assert received() == digest("synthetic-old-token")
            replaced = _save(env, _ask(env), {"botToken": "synthetic-new-token"})
            assert replaced.status_code == 200 and replaced.json()["saved"] is True
            assert replaced.json()["secret_outcome"]["event"] == "secret_replaced"
            assert replaced.json()["secret_replaced"]["fields"] == ["botToken"]
            assert replaced.json()["secret_outcome"]["consumers"][0]["granted"] == []
            assert received() == digest("synthetic-new-token")
            # The Settings page's Clear (and Remove) saves "" through the owner writer.
            _owner_update_settings(fields_on_saved_document({"FAKEPLUG_TOKEN": "", "OUROBOROS_EFFORT_TASK": "low"}))
            assert received() == digest("")
            assert config.load_settings()["OUROBOROS_EFFORT_TASK"] == "low"
            assert config.runtime_setting("OUROBOROS_EFFORT_TASK") == "high", "the task's own reads keep its snapshot"
            config.save_settings({key: value for key, value in _settings(env).items() if key != "FAKEPLUG_TOKEN"})
            assert received() is None and rows[-1]["is_error"] and "missing" in rows[-1]["result"]
        with config.task_settings_scope(subagent_runtime.apply_task_start_settings()):  # admitted without the key
            filled = _save(env, _ask(env), {"botToken": "synthetic-filled-token"})
            assert filled.json()["secret_outcome"]["event"] == "secret_filled"
            assert received() == digest("synthetic-filled-token")
    finally:
        services.kill_all_services(data)
    records = (data / "logs" / "tools.jsonl").read_text(encoding="utf-8")
    assert tool == "start_service" or "child received ***" in records
    assert not any(token in records for token in ("synthetic-old-token", "synthetic-new-token", "synthetic-filled-token"))
