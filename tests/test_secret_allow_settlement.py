"""The secret form's "Save and allow" after its press: how a result is settled, kept and read back.

A neighbour of ``test_secret_hitl.py`` (its fixture and helpers, the same real task store, Settings file, decision
ingress and lifecycle lane): an early reply names a job that is already bound; keys only the form granted hold for
that installation alone while an ordinary grant stays; a rolled-back update ends the consent that waited for its files
and keeps the form's key of the copy it restores; a claim a restart cut is recorded unknown at startup and never
granted, while a press whose consent was already recorded reads waiting; a lane that never started its job reads not
started; the author's ordinary acceptance admits files in Cyber Pro too; an owner-verified edit ends a consent for
other files; a late result is one host row that updates its card and is no one's words."""

from __future__ import annotations

import json

import pytest

from tests.test_secret_hitl import (  # noqa: F401  (the env fixture)
    _MANIFEST,
    CANARY,
    CANARY_B,
    TASK,
    _admit,
    _allow,
    _allow_body,
    _answers,
    _ask,
    _block,
    _cancel,
    _grants,
    _put_settings,
    _reinstall,
    _review_clean,
    _save,
    _save_allow,
    _secret_form_grants,
    _settings,
    _status,
    _switch,
    _write_skill,
    env,
)


def _deferred(env, quiz_id):  # noqa: F811
    response = _save_allow(env, quiz_id, {"botToken": CANARY})
    assert response.status_code == 200, response.text
    assert response.json()["secret_outcome"]["allow"]["state"] == "deferred"
    return response


def test_early_allow_reply_names_a_bound_job_after_the_save(env, monkeypatch):  # noqa: F811
    import asyncio

    import httpx

    from ouroboros import skill_lifecycle_queue as queue

    _admit(env)
    quiz_id = _ask(env)
    selection = _allow(env, quiz_id)
    decision = {"decision_id": f"quiz:{TASK}:{quiz_id}"}
    monkeypatch.setattr(queue, "JOB_ANSWER_WAIT_SEC", 0)

    async def journey():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=env.client.app),
                                     base_url="http://testserver") as client:
            lane = queue._get_lock()
            assert lane.acquire(blocking=False)  # the lane is busy: the grant job is bound, then waits
            try:
                response = await asyncio.wait_for(client.post("/api/decisions/secret", json={
                    **decision, "request_id": "early-allow", "values": {"botToken": CANARY},
                    "observed": {"botToken": False}, "allow": selection,
                }), 5)
                assert response.status_code == 200
                early = response.json()
                assert early["saved"] is True and early["state"] == "answered" and early["allow_job"]
                assert early["secret_outcome"]["allow"]["state"] == "requested"
                assert any(item["id"] == early["allow_job"] and item["status"] == "queued"
                           for item in queue.queue_snapshot()["events"])
                assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY and _grants(env)["granted_keys"] == []
                assert CANARY not in response.text
            finally:
                work = [task for task in queue._BACKGROUND if task.get_loop() is asyncio.get_running_loop()]
                lane.release()
                await asyncio.wait_for(asyncio.gather(*work), 5)
            after = (await client.post("/api/decisions/secret", json={**decision, "action": "status"})).json()
            assert after["secret_outcome"]["allow"]["granted_keys"] == ["FAKEPLUG_TOKEN"]
            assert len(_secret_form_grants(env)) == 1

    asyncio.run(journey())


def test_keys_only_the_form_granted_hold_for_that_installation_and_ordinary_grants_stay(env):  # noqa: F811
    from ouroboros.skill_lifecycle_actions import _grant
    from ouroboros.skill_loader import find_skill, skill_state_dir_path

    _write_skill(env.root, fields="  botToken: FAKEPLUG_TOKEN\n  region: FAKEPLUG_REGION")
    _put_settings(env, FAKEPLUG_REGION="synthetic-region")
    skill = _admit(env)
    assert _grant(skill, env.root, "", ["FAKEPLUG_REGION"], explicit_via="owner_ui")["ok"]  # the Skills button
    quiz_id = _ask(env)
    assert _save_allow(env, quiz_id, {"botToken": CANARY}).json()["secret_outcome"]["allow"]["state"] == "granted"
    assert _grants(env)["granted_keys"] == ["FAKEPLUG_TOKEN", "FAKEPLUG_REGION"]
    stored = json.loads((skill_state_dir_path(env.root, "fakeplug") / "grants.json").read_text())
    assert stored["granted_keys"] == ["FAKEPLUG_REGION"] and stored["form"]["keys"] == ["FAKEPLUG_TOKEN"]
    assert CANARY not in json.dumps(stored)

    _reinstall(env)  # the same release lands again: another installation
    assert find_skill(env.root, "fakeplug").content_hash == skill.content_hash
    assert _grants(env)["granted_keys"] == ["FAKEPLUG_REGION"]  # the ordinary decision stands, the form's key not
    assert _status(env, _ask(env)).json()["skill"]["granted"] == []  # the form may offer it to this installation
    assert _status(env, quiz_id).json()["secret_outcome"]["allow"]["state"] == "granted"  # its result stays history


def _hub_skill(env, name="hubplug"):  # noqa: F811
    """A Hub install of a skill with one secret field, recorded as the review-basis source it came from."""
    from ouroboros.skill_install_record import write_install_record
    from ouroboros.skill_loader import find_skill

    folder = env.root / "skills" / "ouroboroshub" / name
    (folder / "scripts").mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(_MANIFEST.format(name=name, env="HUBPLUG_TOKEN",
                                                      fields="  botToken: HUBPLUG_TOKEN", version="0.1.0"),
                                     encoding="utf-8")
    (folder / "scripts" / "run.py").write_text("print('one')\n", encoding="utf-8")
    write_install_record(env.root, name, {"basis": "review", "source": {"kind": "hub", "id": "hub-synthetic"},
                                          "release": {"digest": find_skill(env.root, name).content_hash}})
    return folder


@pytest.mark.parametrize("fail_receipt", [False, True], ids=["normal", "retry"])
def test_a_rolled_back_update_ends_the_consent_that_waited_for_its_files(env, monkeypatch, fail_receipt):  # noqa: F811
    from ouroboros import skill_grant_follow
    from ouroboros.marketplace import install as storage
    from ouroboros.marketplace.install import snapshot_payload_state
    from ouroboros.skill_grant_follow import consent_of
    from ouroboros.skill_install_record import update_backup_path
    from ouroboros.skill_install_record import (
        read_install_record,
        restore_update_snapshot,
        retain_update_snapshot,
        write_install_record,
    )
    from ouroboros.skill_loader import find_skill

    folder = _hub_skill(env)
    previous = find_skill(env.root, "hubplug").content_hash
    snapshot = snapshot_payload_state(env.root, "hubplug", folder)
    (folder / "scripts" / "run.py").write_text("print('two')\n", encoding="utf-8")  # the candidate files landed
    record = {**read_install_record(env.root, "hubplug"),
              "release": {"digest": find_skill(env.root, "hubplug").content_hash}}
    retain_update_snapshot(snapshot, record)
    write_install_record(env.root, "hubplug", record)
    _admit(env, "hubplug")  # even its review passing does not grant before the update settles
    quiz_id = _ask(env, plugin="hubplug")
    _deferred(env, quiz_id)
    assert _grants(env, "hubplug")["granted_keys"] == [] and consent_of(env.root, "hubplug")

    backup = update_backup_path(env.root, "hubplug")
    if fail_receipt:
        def write_failed(*args, **kwargs):
            raise OSError("synthetic receipt store unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(skill_grant_follow, "update_json_locked", write_failed)
            assert restore_update_snapshot(env.root, "hubplug", "review_blockers") is False
        rejected = read_install_record(env.root, "hubplug")["update_rejected"]
        assert rejected["reason"] == "review_blockers" and backup.is_file()
        assert find_skill(env.root, "hubplug").content_hash == previous and consent_of(env.root, "hubplug")
        monkeypatch.setattr(storage, "restore_payload_state", lambda *a, **kw: pytest.fail("restored files replayed"))

    assert restore_update_snapshot(env.root, "hubplug", "review_blockers")
    assert not backup.exists()
    if fail_receipt:
        assert read_install_record(env.root, "hubplug")["update_rejected"] == rejected
    assert find_skill(env.root, "hubplug").content_hash == previous
    ended = _block(env, quiz_id)["secret_outcome"]["allow"]
    assert ended["state"] == "refused" and ended["reason"] == "update_rolled_back" and ended["resolved_at"]
    assert ended["enabled"] is False and ended["enable_reason"] == "not_allowed"
    assert consent_of(env.root, "hubplug") == {} and _grants(env, "hubplug")["granted_keys"] == []


def test_a_claim_a_restart_cut_is_unknown_at_startup_and_never_granted(env):  # noqa: F811
    from ouroboros.skill_grant_follow import allow_installation, claim_consent, consent_of
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_source_install import settle_pending_updates_at_startup

    quiz_id = _ask(env)
    _deferred(env, quiz_id)
    skill = find_skill(env.root, "fakeplug")
    assert claim_consent(env.root, skill, allow_installation(env.root, skill))[0] == "claimed"  # then it ended
    settle_pending_updates_at_startup(env.root)  # the next start, before any admission
    cut = _block(env, quiz_id)["secret_outcome"]["allow"]
    assert cut["state"] == "unknown" and cut["reason"] == "grant_unconfirmed" and cut["enabled"] is None
    assert consent_of(env.root, "fakeplug") == {} and _grants(env)["granted_keys"] == []
    assert not [row for row in _secret_form_grants(env) if row.get("fulfilled")]


def test_admitted_other_files_end_a_consent_for_the_earlier_ones(env):  # noqa: F811
    from ouroboros.skill_grant_follow import consent_of, end_superseded_consent
    from ouroboros.skill_loader import find_skill

    quiz_id = _ask(env)
    _deferred(env, quiz_id)
    (env.root / "skills" / "external" / "fakeplug" / "scripts" / "run.py").write_text("print('edited')\n")
    end_superseded_consent(env.root, "fakeplug", find_skill(env.root, "fakeplug").content_hash)  # an owner edit's
    ended = _block(env, quiz_id)["secret_outcome"]["allow"]
    assert ended["state"] == "refused" and ended["reason"] == "skill_changed"
    assert consent_of(env.root, "fakeplug") == {} and _grants(env)["granted_keys"] == []


def test_a_late_result_is_one_host_row_that_updates_its_card_and_says_nothing(env):  # noqa: F811
    from ouroboros.dialogue_provenance import row_class
    from ouroboros.dialogue_store import rows_by_client_id
    from ouroboros.gateway.history import _collect_chat_rows
    from ouroboros.room_messages import counts

    quiz_id = _ask(env)
    _deferred(env, quiz_id)
    _review_clean(env)
    [row] = [item.row for item in rows_by_client_id(env.root, f"secret_allow_result:{TASK}:{quiz_id}", chat_id=1)]
    assert row["direction"] == "system" and row["ts"] == row["result"]["at"]  # dated by the result, not the answer
    assert row["ts"] != _block(env, quiz_id)["answered_at"]
    assert row_class(row)["lane"] == 2 and row_class(row)["author"]["kind"] == "host"
    assert counts(row) is False  # no message: no unread, no room ordering
    assert row["result"]["consumers"] == _block(env, quiz_id)["secret_outcome"]["consumers"]  # one snapshot
    evidence = []
    rows, _quota = _collect_chat_rows(env.root / "logs" / "chat.jsonl", env.root / "archive", 30,
                                       lambda *_: True, {}, selected_entries=[row], replay_evidence=evidence)
    assert rows == []  # never a bubble of its own
    [frame] = evidence
    assert frame["system_type"] == "quiz_answer" and frame["ts"] == row["ts"]
    assert frame["quiz"]["quiz_id"] == quiz_id and frame["quiz"]["state"] == "answered"
    assert frame["quiz"]["secret_outcome"]["allow"]["state"] == "granted"


def test_an_evicted_cancelled_secret_card_reads_back_closed_never_open(env):  # noqa: F811
    from ouroboros.gateway.history import _collect_chat_rows
    from ouroboros.owner_quiz import _KEEP, _mutate_projection
    from supervisor.message_bus import log_chat

    quiz_id = _ask(env)
    asked = _block(env, quiz_id)
    ask_row = log_chat("out", 1, 0, asked["question"], ts=asked["asked_at"], task_id=TASK, record_type="quiz",
                       require_write=True, client_message_id=f"quiz_publication:{TASK}:{quiz_id}",
                       quiz={"quiz_id": quiz_id, "question": asked["question"], "options": [], "state": "open",
                             "secret_request": dict(asked["secret_request"])})
    assert _cancel(env, quiz_id).status_code == 200
    closed = _block(env, quiz_id)
    assert closed["state"] == "closed" and closed["secret_outcome"]["event"] == "secret_cancelled"
    log_chat("system", 1, 0, closed["closed_reason"], task_id=TASK, record_type="quiz_closed", quiz=closed,
             ts=closed["closed_at"], source="owner_question_close", require_write=True,
             client_message_id=f"quiz_closed:{TASK}:{quiz_id}")  # the closure's history row, as its writer stores it
    _mutate_projection(env.root, TASK, lambda quizzes: True if quizzes.pop(quiz_id, None) else _KEEP)
    status = _status(env, quiz_id).json()
    assert status["state"] == "closed" and status["secret_outcome"]["event"] == "secret_cancelled"
    rows, _quota = _collect_chat_rows(env.root / "logs" / "chat.jsonl", env.root / "archive", 30,
                                       lambda *_: True, {}, selected_entries=[ask_row])
    [card] = [row["quiz"] for row in rows if row.get("msg_type") == "quiz"]
    assert card["state"] == "closed" and card["closed_reason"] == closed["closed_reason"]
    assert card["secret_outcome"]["event"] == "secret_cancelled" and "question_delivery" not in card


@pytest.mark.parametrize("bucket,kind,bound", [
    ("native", "builtin", True), ("native", "registry", False), ("native", None, False),
    ("clawhub", "registry", True), ("clawhub", "hub", False), ("external", None, True),
])
def test_a_native_or_clawhub_folder_binds_the_form_only_with_the_record_that_lands_there(env, bucket, kind, bound):  # noqa: F811
    from ouroboros.skill_grant_follow import installation_token
    from ouroboros.skill_install_record import write_install_record

    folder = env.root / "skills" / bucket / "boundplug"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(_MANIFEST.format(name="boundplug", env="BOUNDPLUG_TOKEN",
                                                      fields="  botToken: BOUNDPLUG_TOKEN", version="0.1.0"),
                                     encoding="utf-8")
    if kind:  # a name-indexed record: only its source's own bucket links it to this folder
        write_install_record(env.root, "boundplug", {"basis": "review", "source": {"kind": kind, "id": "synthetic"},
                                                     "release": {"digest": "sha256:" + "0" * 64}})
    token = installation_token(env.root, "boundplug", folder)
    assert bool(token) is bound
    if bound:  # the concrete folder is part of it: another folder of the same name and record is another installation
        moved = folder.with_name("boundplug-previous")
        folder.rename(moved)
        moved.with_name("boundplug").mkdir()
        assert installation_token(env.root, "boundplug", folder) not in ("", token)


def test_successive_immediate_allows_accumulate_and_preserve_ordinary_decisions(env):  # noqa: F811
    from ouroboros.skill_loader import save_skill_grants, skill_state_dir_path

    _write_skill(
        env.root,
        env="FAKEPLUG_TOKEN, FAKEPLUG_REGION, FAKEPLUG_BASE, FAKEPLUG_EXTRA",
        fields="  botToken: FAKEPLUG_TOKEN\n  region: FAKEPLUG_REGION",
    )
    _put_settings(env, FAKEPLUG_BASE="synthetic-base", FAKEPLUG_EXTRA="synthetic-extra")
    old = _admit(env)
    # Independent ordinary decision on older bytes: FAKEPLUG_BASE granted, FAKEPLUG_EXTRA covered and ungranted (blocking).
    save_skill_grants(
        env.root,
        "fakeplug",
        ["FAKEPLUG_BASE"],
        content_hash=old.content_hash,
        requested_keys=["FAKEPLUG_BASE", "FAKEPLUG_EXTRA"],
    )
    script = env.root / "skills" / "external" / "fakeplug" / "scripts" / "run.py"
    script.write_text("print('v2')\n", encoding="utf-8")
    current = _admit(env)
    assert current.content_hash != old.content_hash

    first_quiz = _ask(env, fields=("botToken",))
    first_reply = _save_allow(
        env, first_quiz, {"botToken": CANARY}, request_id="r-tok",
        allow=_allow(env, first_quiz, fields=("botToken",), enable=False),
    )
    assert first_reply.status_code == 200 and first_reply.json()["secret_outcome"]["allow"]["state"] == "granted"
    assert _grants(env)["granted_keys"] == ["FAKEPLUG_TOKEN"]

    second_quiz = _ask(env, fields=("region",))
    second_reply = _save_allow(
        env, second_quiz, {"region": CANARY_B}, request_id="r-reg",
        allow=_allow(env, second_quiz, fields=("region",), enable=False),
    )
    assert second_reply.status_code == 200 and second_reply.json()["secret_outcome"]["allow"]["state"] == "granted"

    status = _grants(env)
    assert status["granted_keys"] == ["FAKEPLUG_TOKEN", "FAKEPLUG_REGION"]
    assert "FAKEPLUG_EXTRA" in status["missing_keys"] and "FAKEPLUG_EXTRA" not in status["pending_keys"]
    assert status["blocked"] is True
    assert status["earlier_grant"] == {"keys": ["FAKEPLUG_BASE"], "permissions": []}
    stored = json.loads((skill_state_dir_path(env.root, "fakeplug") / "grants.json").read_text(encoding="utf-8"))
    assert stored["content_hash"] == old.content_hash
    assert stored["granted_keys"] == ["FAKEPLUG_BASE"]
    assert stored["requested_keys"] == ["FAKEPLUG_BASE", "FAKEPLUG_EXTRA"]
    assert stored["form"]["digest"] == current.content_hash
    assert stored["form"]["keys"] == ["FAKEPLUG_REGION", "FAKEPLUG_TOKEN"]

    # Older bytes never gain any key granted by the form for the newer digest.
    script.write_text("print('ok')\n", encoding="utf-8")
    restored = _admit(env)
    assert restored.content_hash == old.content_hash
    old_status = _grants(env)
    assert old_status["granted_keys"] == ["FAKEPLUG_BASE"]
    assert "FAKEPLUG_EXTRA" in old_status["missing_keys"] and "FAKEPLUG_EXTRA" not in old_status["pending_keys"]


def test_startup_recovery_finishes_unclaimed_admitted_consent_once(env):  # noqa: F811
    from ouroboros.skill_grant_follow import consent_of
    from ouroboros.skill_loader import save_enabled
    from ouroboros.skill_source_install import settle_pending_updates_at_startup

    quiz_id = _ask(env)
    _deferred(env, quiz_id)
    _admit(env)  # files are admitted before restart without the admission callback finishing the consent
    assert _grants(env)["granted_keys"] == [] and consent_of(env.root, "fakeplug")
    assert "claimed_at" not in consent_of(env.root, "fakeplug")

    settle_pending_updates_at_startup(env.root)
    assert _grants(env)["granted_keys"] == ["FAKEPLUG_TOKEN"]
    assert _switch(env)["enabled"] is True and _switch(env)["actor"] == "owner_ui"
    assert consent_of(env.root, "fakeplug") == {}
    settled = _block(env, quiz_id)["secret_outcome"]["allow"]
    assert settled["state"] == "granted" and settled["enabled"] is True and settled["resolved_at"]
    assert [bool(row.get("fulfilled")) for row in _secret_form_grants(env)] == [False, True]

    settle_pending_updates_at_startup(env.root)  # already settled: never runs a second time
    assert len(_secret_form_grants(env)) == 2

    # Owner Off stands across startup recovery as well.
    _write_skill(env.root, "offplug", env="OFFPLUG_TOKEN", fields="  botToken: OFFPLUG_TOKEN")
    off_quiz = _ask(env, plugin="offplug")
    _deferred(env, off_quiz)
    save_enabled(env.root, "offplug", False, actor="owner_ui")
    _admit(env, "offplug")
    settle_pending_updates_at_startup(env.root)
    assert _grants(env, "offplug")["granted_keys"] == ["OFFPLUG_TOKEN"]
    assert _switch(env, "offplug")["enabled"] is False
    off_allow = _block(env, off_quiz)["secret_outcome"]["allow"]
    assert off_allow["state"] == "granted" and off_allow["enabled"] is False and off_allow["enable_reason"] == "owner_off"


def test_settle_pending_update_refreshes_already_enabled_extension_once(env, monkeypatch):  # noqa: F811
    from ouroboros import extension_loader
    from ouroboros.config import load_settings
    from ouroboros.marketplace.install import snapshot_payload_state
    from ouroboros.skill_install_record import (
        read_install_record,
        retain_update_snapshot,
        settle_pending_update,
        write_install_record,
    )
    from ouroboros.skill_loader import find_skill, save_enabled
    from tests._extension_loader_shared import _write_ext_skill, extension_tool_answer
    from tests._shared import clean_extension_runtime_state

    def plugin_source(tag: str) -> str:
        return (
            "def register(api):\n"
            f"    api.register_tool('status', lambda ctx, **_kw: '{tag}:' + "
            "('yes' if 'HUBEXT_TOKEN' in api.get_settings(['HUBEXT_TOKEN']) else 'no'), "
            "description='status', schema={})\n"
        )

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    try:
        hub_root = env.root / "skills" / "ouroboroshub"
        folder = _write_ext_skill(
            hub_root,
            "hubext",
            plugin_body=plugin_source("v1"),
            permissions=["tool", "read_settings"],
            env_from_settings=["HUBEXT_TOKEN"],
            extra_frontmatter="secret_fields:\n  botToken: HUBEXT_TOKEN\n",
        )
        write_install_record(env.root, "hubext", {
            "basis": "review",
            "source": {"kind": "hub", "id": "hub-synthetic"},
            "release": {"digest": find_skill(env.root, "hubext").content_hash},
        })
        _admit(env, "hubext")
        save_enabled(env.root, "hubext", True, actor="owner_ui")
        extension_loader.reconcile_extension("hubext", env.root, load_settings)
        assert extension_tool_answer(env.root, "hubext", "status") == "v1:no"

        snapshot = snapshot_payload_state(env.root, "hubext", folder)
        (folder / "plugin.py").write_text(plugin_source("v2"), encoding="utf-8")
        record = {**read_install_record(env.root, "hubext"),
                  "release": {"digest": find_skill(env.root, "hubext").content_hash}}
        retain_update_snapshot(snapshot, record)
        write_install_record(env.root, "hubext", record)
        _admit(env, "hubext")
        extension_loader.reconcile_extension("hubext", env.root, load_settings)
        assert extension_tool_answer(env.root, "hubext", "status") == "v2:no"

        quiz_id = _ask(env, plugin="hubext")
        reply = _save_allow(env, quiz_id, {"botToken": CANARY}, allow=_allow(env, quiz_id, enable=False))
        assert reply.status_code == 200 and reply.json()["secret_outcome"]["allow"]["state"] == "deferred"
        assert extension_tool_answer(env.root, "hubext", "status") == "v2:no"

        calls = []
        real_reconcile = extension_loader.reconcile_extension

        def counted(skill_name, *args, **kwargs):
            calls.append(skill_name)
            return real_reconcile(skill_name, *args, **kwargs)

        monkeypatch.setattr(extension_loader, "reconcile_extension", counted)
        result = settle_pending_update(env.root, "hubext")
        assert result["status"] == "accepted"
        assert calls == ["hubext"]
        assert _grants(env, "hubext")["granted_keys"] == ["HUBEXT_TOKEN"]
        assert _block(env, quiz_id)["secret_outcome"]["allow"]["state"] == "granted"
        assert extension_tool_answer(env.root, "hubext", "status") == "v2:yes"
    finally:
        clean_extension_runtime_state()


def test_unreadable_switch_stays_unknown_beside_grant_with_historical_receipt(env):  # noqa: F811
    from ouroboros.skill_loader import skill_state_dir_path

    _admit(env)
    quiz_id = _ask(env)
    shown = _allow(env, quiz_id, enable=True)
    enabled_path = skill_state_dir_path(env.root, "fakeplug") / "enabled.json"
    enabled_path.parent.mkdir(parents=True, exist_ok=True)
    enabled_path.write_text("{not-json", encoding="utf-8")
    assert _status(env, quiz_id).json()["skill"]["switch"] == "unknown"

    reply = _save_allow(env, quiz_id, {"botToken": CANARY}, allow=shown)
    assert reply.status_code == 200, reply.text
    allow = reply.json()["secret_outcome"]["allow"]
    assert allow["state"] == "granted" and allow["granted_keys"] == ["FAKEPLUG_TOKEN"]
    assert allow["enabled"] is None and allow["enable_reason"] == "switch_unreadable"
    assert _grants(env)["granted_keys"] == ["FAKEPLUG_TOKEN"]
    assert enabled_path.read_text(encoding="utf-8") == "{not-json"
    [mail] = _answers(env)
    assert "Whether fakeplug was on at the time is unknown: its saved switch could not be read." in mail


def test_orphan_recovery_serializes_behind_active_allow_reservation(env, monkeypatch):  # noqa: F811
    from ouroboros import secret_requests as sr
    from ouroboros.gateway import secret_decision as sd

    skill = _admit(env)
    real_run_allow = sr.run_allow

    def pause_before_claim(drive_root, task_id, quiz_id, **kwargs):
        if kwargs.get("reserved"):
            sr.release_allow(task_id, quiz_id)
        return _block(env, quiz_id)

    monkeypatch.setattr(sd.sr, "run_allow", pause_before_claim)
    quiz_id = _ask(env)
    reply = _save_allow(env, quiz_id, {"botToken": CANARY}, allow=_allow_body(env, skill))
    assert reply.status_code == 200 and reply.json()["secret_outcome"]["allow"]["state"] == "requested"
    monkeypatch.setattr(sd.sr, "run_allow", real_run_allow)

    assert sr.reserve_allow(TASK, quiz_id)
    try:
        sd.deliver_secret_outcome(env.root, TASK, quiz_id, recovery=True)
        assert _block(env, quiz_id)["secret_outcome"]["allow"]["state"] == "requested"
        assert not _answers(env)
    finally:
        sr.release_allow(TASK, quiz_id)

    sd.deliver_secret_outcome(env.root, TASK, quiz_id, recovery=True)
    settled = _block(env, quiz_id)["secret_outcome"]["allow"]
    assert settled["state"] == "refused" and settled["reason"] == "not_started"
    assert len(_answers(env)) == 1


def test_form_only_repair_keeps_held_notify_owner_ungranted_and_reading_unblocked(env, monkeypatch):  # noqa: F811
    from ouroboros import extension_loader
    from ouroboros.config import load_settings
    from ouroboros.skill_grant_follow import (
        extend_verified_edit,
        finish_local_edit_grants,
        installation_of,
        load_receipt,
    )
    from ouroboros.skill_loader import find_skill, save_skill_grants, skill_state_dir_path
    from tests._extension_loader_shared import _write_ext_skill, extension_tool_answer
    from tests._shared import clean_extension_runtime_state

    def plugin_source(tag: str) -> str:
        return (
            "def register(api):\n"
            f"    api.register_tool('read', lambda ctx, **_kw: '{tag}:' + "
            "('yes' if 'CALEXT_TOKEN' in api.get_settings(['CALEXT_TOKEN']) else 'no'), "
            "description='read', schema={})\n"
        )

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    try:
        ext_root = env.root / "skills" / "external"
        folder = _write_ext_skill(
            ext_root,
            "calext",
            plugin_body=plugin_source("v1"),
            permissions=["tool", "read_settings", "notify_owner"],
            env_from_settings=["CALEXT_TOKEN"],
            extra_frontmatter="secret_fields:\n  botToken: CALEXT_TOKEN\n",
        )
        v1 = _admit(env, "calext")
        # Ordinary decision on v1 approved notify_owner before the secret form granted CALEXT_TOKEN.
        save_skill_grants(
            env.root,
            "calext",
            [],
            content_hash=v1.content_hash,
            requested_keys=[],
            granted_permissions=["notify_owner"],
            requested_permissions=["notify_owner"],
        )
        quiz_id = _ask(env, plugin="calext")
        reply = _save_allow(env, quiz_id, {"botToken": CANARY}, allow=_allow(env, quiz_id, enable=True))
        assert reply.status_code == 200 and reply.json()["secret_outcome"]["allow"]["state"] == "granted"
        assert _grants(env, "calext")["granted_keys"] == ["CALEXT_TOKEN"]
        assert _grants(env, "calext")["granted_permissions"] == ["notify_owner"]
        assert extension_tool_answer(env.root, "calext", "read") == "v1:yes"

        # Verified Repair edits v1 -> v2: form-only rebind keeps v1's ordinary decision and holds notify_owner.
        installation = installation_of(env.root, "calext", folder)
        (folder / "plugin.py").write_text(plugin_source("v2"), encoding="utf-8")
        v2 = find_skill(env.root, "calext")
        assert v2.content_hash != v1.content_hash
        assert extend_verified_edit(env.root, "calext", installation, v1.content_hash, v2.content_hash)
        _admit(env, "calext")
        finish_local_edit_grants(env.root, find_skill(env.root, "calext"))

        stored = json.loads((skill_state_dir_path(env.root, "calext") / "grants.json").read_text(encoding="utf-8"))
        assert stored["content_hash"] == v1.content_hash and stored["granted_permissions"] == ["notify_owner"]
        assert stored["form"]["digest"] == v2.content_hash and stored["form"]["keys"] == ["CALEXT_TOKEN"]
        receipt = load_receipt(env.root, "calext")
        assert receipt["digest"] == v2.content_hash and receipt.get("held_permissions") == ["notify_owner"]

        status = _grants(env, "calext")
        assert status["granted_keys"] == ["CALEXT_TOKEN"]
        assert status["granted_permissions"] == []
        assert status["missing_permissions"] == ["notify_owner"]
        assert status["pending_permissions"] == ["notify_owner"]
        assert status["blocked"] is False and status["usable"] is True

        extension_loader.reconcile_extension("calext", env.root, load_settings)
        assert extension_tool_answer(env.root, "calext", "read") == "v2:yes"
    finally:
        clean_extension_runtime_state()


def test_orphan_recovery_does_not_fall_back_to_stale_block_when_absent(env, monkeypatch):  # noqa: F811
    from ouroboros import owner_quiz
    from ouroboros import secret_requests as sr
    from ouroboros.gateway import secret_decision as sd

    skill = _admit(env)

    def pause_before_claim(drive_root, task_id, quiz_id, **kwargs):
        if kwargs.get("reserved"):
            sr.release_allow(task_id, quiz_id)
        return _block(env, quiz_id)

    monkeypatch.setattr(sd.sr, "run_allow", pause_before_claim)
    quiz_id = _ask(env)
    reply = _save_allow(env, quiz_id, {"botToken": CANARY}, allow=_allow_body(env, skill))
    assert reply.status_code == 200 and reply.json()["secret_outcome"]["allow"]["state"] == "requested"

    recorded = []
    real_reserve = sd.sr.reserve_allow
    real_record = sd.sr.record_allow

    def reserve_after_eviction(task_id, qid):
        if not real_reserve(task_id, qid):
            return False
        owner_quiz._mutate_projection(
            env.root, task_id, lambda quizzes: True if quizzes.pop(qid, None) else owner_quiz._KEEP
        )
        return True

    def spy_record(*args, **kwargs):
        recorded.append(args)
        return real_record(*args, **kwargs)

    monkeypatch.setattr(sd.sr, "reserve_allow", reserve_after_eviction)
    monkeypatch.setattr(sd.sr, "record_allow", spy_record)

    outcome = sd.deliver_secret_outcome(env.root, TASK, quiz_id, recovery=True)
    assert outcome == {} and recorded == [] and not _answers(env)
    assert real_reserve(TASK, quiz_id)
    sr.release_allow(TASK, quiz_id)


def test_a_press_whose_consent_was_recorded_before_a_restart_reads_waiting_not_unknown(env, monkeypatch):  # noqa: F811
    """The step claimed the press and recorded the owner's consent for files not admitted yet; the process ended before
    the card said so. The next start reads the records under the question's reservation: the consent still waits for
    exactly those files and that installation, so the card and the asker say it waits, nothing is granted on that read,
    and the admission later grants it once."""
    from ouroboros import secret_requests as sr
    from ouroboros.gateway import secret_decision as sd
    from ouroboros.skill_grant_follow import consent_of, record_consent
    from ouroboros.skill_loader import find_skill

    def recorded_then_cut(drive_root, task_id, quiz_id, **kwargs):
        block = _block(env, quiz_id)
        allow, decision = block["secret_outcome"]["allow"], {"decision_id": f"quiz:{task_id}:{quiz_id}",
                                                             "request_id": block["request_id"]}
        assert sr.claim_allow(drive_root, decision)
        assert record_consent(drive_root, find_skill(drive_root, "fakeplug"), allow["keys"],
                              installation=allow["installation"], enable=allow["enable"], decision=decision) == "recorded"
        if kwargs.get("reserved"):
            sr.release_allow(task_id, quiz_id)
        return _block(env, quiz_id)

    monkeypatch.setattr(sd.sr, "run_allow", recorded_then_cut)
    quiz_id = _ask(env)
    reply = _save_allow(env, quiz_id, {"botToken": CANARY})
    assert reply.json()["secret_outcome"]["allow"]["state"] == "running" and not _answers(env)
    sd.deliver_secret_outcome(env.root, TASK, quiz_id, recovery=True)  # the next start: no step holds the press
    allow = _block(env, quiz_id)["secret_outcome"]["allow"]
    assert allow["state"] == "deferred" and allow["reason"] == "not_admitted"
    assert allow["enabled"] is False and allow["enable_reason"] == "deferred"
    waiting = consent_of(env.root, "fakeplug")
    assert waiting["decisions"] == [{"decision_id": f"quiz:{TASK}:{quiz_id}", "request_id": "r1"}]
    assert "claimed_at" not in waiting and _grants(env)["granted_keys"] == [] and _switch(env) is None
    [mail] = _answers(env)
    assert "consent was recorded to grant it once exactly those files passed a review" in mail
    assert "whether it was granted is unknown" not in mail
    _review_clean(env)
    granted = _block(env, quiz_id)["secret_outcome"]["allow"]
    assert granted["state"] == "granted" and granted["resolved_at"] and _grants(env)["granted_keys"] == ["FAKEPLUG_TOKEN"]
    assert [bool(row.get("fulfilled")) for row in _secret_form_grants(env)] == [True]


def test_a_grant_step_a_busy_lane_never_started_reads_not_started_and_never_runs(env, monkeypatch):  # noqa: F811
    """The lifecycle lane stayed busy past its deadline, so the queue never reached the job's start (``on_started``,
    which its runner always follows): nothing was granted, both readers say the step did not start (no restart is
    claimed), and the job never runs after it."""
    from ouroboros import skill_lifecycle_queue as queue

    skill = _admit(env)
    quiz_id = _ask(env)
    monkeypatch.setattr(queue, "_lifecycle_deadline_sec", lambda: 0.05)
    lane = queue._get_lock()
    assert lane.acquire(blocking=False)  # another job holds the lane
    try:
        response = _save_allow(env, quiz_id, {"botToken": CANARY}, allow=_allow_body(env, skill))
    finally:
        lane.release()
    assert response.status_code == 200 and response.json()["saved"] is True
    allow = response.json()["secret_outcome"]["allow"]
    assert allow["state"] == "refused" and allow["reason"] == "not_started" and "started_at" not in allow
    assert allow["enabled"] is False and allow["enable_reason"] == "not_allowed"
    assert _grants(env)["granted_keys"] == [] and _switch(env) is None and not _secret_form_grants(env)
    [mail] = _answers(env)
    assert "nothing was granted: the grant step did not start." in mail and "restarted" not in mail
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY  # the save itself stands


def _author_accepts(env, name="fakeplug"):  # noqa: F811
    """The author's ordinary acceptance of exactly the current files: reviewer feedback exists and its preflight passed."""
    from ouroboros.review_records import build_author_disposition
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_review_state

    digest = find_skill(env.root, name).content_hash
    author = build_author_disposition(disposition="accepted", rationale="The finding is handled in these files.",
                                      subject_hash=digest, enforcement="advisory", action="finish")
    author["review_reference"] = {"surface": "skill", "basis": "reviewed", "content_hash": digest,
                                  "preflight": {"content_hash": digest, "status": "pass"}}
    save_review_state(env.root, name, SkillReviewState(
        status="blockers", content_hash=digest, author_disposition=author,
        findings=[{"item": "bug_hunting", "verdict": "FAIL", "severity": "critical", "reason": "Controlled finding."}]))
    return find_skill(env.root, name)


@pytest.mark.parametrize("source,admitted", [("ordinary", True), ("strict", False)])
def test_cyber_pro_admits_files_by_the_authors_ordinary_acceptance_never_by_the_mode(monkeypatch, source, admitted):
    """``files_accepted`` reads the gate Cyber Pro would name only by its own authority as it reads outside Cyber Pro:
    the author's ordinary acceptance of exactly these files counts; the mode alone, and an author acceptance of a source
    that reviews each skill (which needs its review or the owner's Skip), never do."""
    from ouroboros import skill_review_status
    from ouroboros.review_records import build_author_disposition
    from ouroboros.skill_grant_follow import files_accepted
    from ouroboros.skill_loader import SkillReviewState

    monkeypatch.setattr(skill_review_status, "cyber_authority", lambda: True)
    digest = "sha256:" + "a" * 64
    author = build_author_disposition(disposition="accepted", rationale="The finding is handled in these files.",
                                      subject_hash=digest, enforcement="advisory", action="finish")
    author["review_reference"] = {"surface": "skill", "basis": "reviewed", "content_hash": digest,
                                  "preflight": {"content_hash": digest, "status": "pass"}}
    review = SkillReviewState(status="blockers", content_hash=digest, author_disposition=author,
                              findings=[{"item": "bug_hunting", "verdict": "FAIL", "severity": "critical",
                                         "reason": "Controlled finding."}],
                              acceptance={"strict_review": True} if source == "strict" else {})
    assert review.gate_for(digest)["blocking_reason"] == "cyber_authority"  # the mode would run them either way
    assert files_accepted(review, digest) is admitted
    assert files_accepted(SkillReviewState(status="pending", content_hash=digest), digest) is False


def test_an_ordinary_author_acceptance_in_cyber_pro_completes_the_waiting_allow_with_one_preparation(env, monkeypatch):  # noqa: F811
    """Cyber Pro would run these files on its own authority, which never admits them for the key; the author's ordinary
    acceptance of exactly these files does, as outside Cyber Pro. The allow that waited is granted and turns the skill
    on once, on the admission's own dependency preparation."""
    from ouroboros import skill_dependencies, skill_review_status
    from ouroboros.skill_grant_follow import consent_of
    from ouroboros.skill_lifecycle_actions import grant_consent_on_admission

    monkeypatch.setattr(skill_review_status, "cyber_authority", lambda: True)
    quiz_id = _ask(env)
    _deferred(env, quiz_id)  # the mode alone admits nothing: the consent waits
    accepted = _author_accepts(env)
    assert accepted.review.gate_for(accepted.content_hash)["blocking_reason"] == "cyber_authority"
    prepared = []

    def prepare(_drive_root, skill, **_kwargs):  # the admission's one dependency preparation
        prepared.append(skill.name)
        return "not_required", ""

    monkeypatch.setattr(skill_dependencies, "ensure_skill_dependencies", prepare)
    deps = skill_dependencies.ensure_skill_dependencies(env.root, accepted)  # as the author's finish prepares them
    grant_consent_on_admission(env.root, "fakeplug", expected_skill=accepted, deps_result=deps)
    assert prepared == ["fakeplug"]  # that one preparation also served the turn-on
    allow = _block(env, quiz_id)["secret_outcome"]["allow"]
    assert allow["state"] == "granted" and allow["granted_keys"] == ["FAKEPLUG_TOKEN"] and allow["resolved_at"]
    assert allow["enabled"] is True and _switch(env)["enabled"] is True and _switch(env)["actor"] == "owner_ui"
    assert _grants(env)["granted_keys"] == ["FAKEPLUG_TOKEN"] and consent_of(env.root, "fakeplug") == {}
    assert [bool(row.get("fulfilled")) for row in _secret_form_grants(env)] == [False, True]


def test_a_form_grant_then_an_ordinary_grant_of_another_field_keeps_both_and_names_each_readers_own_keys(env):  # noqa: F811
    """The form grants one field's key; the Skills button then grants another field's key as an ordinary decision. Both
    stand, and the result the task reads names, for every reader, exactly the asked keys it requests and those it was
    allowed to read: names only, never a value."""
    from ouroboros.skill_lifecycle_actions import _grant
    from ouroboros.skill_loader import find_skill

    _write_skill(env.root, fields="  botToken: FAKEPLUG_TOKEN\n  region: FAKEPLUG_REGION")
    _write_skill(env.root, "mailer", env="FAKEPLUG_TOKEN", fields="  token: FAKEPLUG_TOKEN")
    _admit(env)
    _admit(env, "mailer")
    first = _ask(env)
    allow = _save_allow(env, first, {"botToken": CANARY}, allow=_allow(env, first, enable=False)).json()[
        "secret_outcome"]["allow"]
    assert allow["state"] == "granted" and allow["granted_keys"] == ["FAKEPLUG_TOKEN"]
    _put_settings(env, FAKEPLUG_REGION="synthetic-region")
    assert _grant(find_skill(env.root, "fakeplug"), env.root, "", ["FAKEPLUG_REGION"], explicit_via="owner_ui")["ok"]
    assert _grants(env)["granted_keys"] == ["FAKEPLUG_TOKEN", "FAKEPLUG_REGION"]
    assert _grants(env, "mailer")["granted_keys"] == []

    both = _ask(env, fields=("botToken", "region"))
    response = _save(env, both, {"botToken": CANARY_B, "region": "second-synthetic-region"}, request_id="r2")
    assert response.status_code == 200 and response.json()["secret_outcome"]["event"] == "secret_replaced"
    assert _grants(env)["granted_keys"] == ["FAKEPLUG_TOKEN", "FAKEPLUG_REGION"]  # a plain Replace changes no access
    mail = _answers(env)[-1]
    assert ("fakeplug (disabled; requests FAKEPLUG_TOKEN, FAKEPLUG_REGION; is allowed to read FAKEPLUG_TOKEN, "
            "FAKEPLUG_REGION)") in mail
    assert ("mailer (disabled; requests FAKEPLUG_TOKEN; is not allowed to read FAKEPLUG_TOKEN; "
            "use Grant access in Skills)") in mail
    assert CANARY_B not in mail and "second-synthetic-region" not in mail


def test_a_rolled_back_update_restores_the_forms_key_and_its_retry_only_finishes_that(env, monkeypatch):  # noqa: F811
    """The form granted a key to the installed files; an update landed, then rolled back to them as a restored copy.
    While writing the key's binding to that copy fails, the rollback keeps its snapshot with the earlier copy's identity
    and grants nothing; its ordinary retry only finishes the restoration, and the previously granted key reads granted
    again without any new grant."""
    from ouroboros import skill_grant_follow
    from ouroboros.marketplace.install import snapshot_payload_state
    from ouroboros.skill_install_record import (
        read_install_record,
        restore_update_snapshot,
        retain_update_snapshot,
        update_backup_path,
        write_install_record,
    )
    from ouroboros.skill_loader import find_skill

    folder = _hub_skill(env)
    _admit(env, "hubplug")
    quiz_id = _ask(env, plugin="hubplug")
    allow = _save_allow(env, quiz_id, {"botToken": CANARY}, allow=_allow(env, quiz_id, enable=False)).json()[
        "secret_outcome"]["allow"]
    assert allow["state"] == "granted" and _grants(env, "hubplug")["granted_keys"] == ["HUBPLUG_TOKEN"]
    previous = find_skill(env.root, "hubplug").content_hash
    snapshot = snapshot_payload_state(env.root, "hubplug", folder)
    assert snapshot.prior_form_installation == allow["installation"]  # the identity the key was granted for
    (folder / "scripts" / "run.py").write_text("print('two')\n", encoding="utf-8")  # the update's files land
    record = {**read_install_record(env.root, "hubplug"),
              "release": {"digest": find_skill(env.root, "hubplug").content_hash}}
    retain_update_snapshot(snapshot, record)
    write_install_record(env.root, "hubplug", record)
    assert _grants(env, "hubplug")["granted_keys"] == []  # the update's files read none of the form's keys

    real, failed = skill_grant_follow.update_json_locked, []

    def first_binding_write_fails(path, mutate, **kwargs):
        if path.name == "grants.json" and not failed:
            failed.append(path.name)
            raise OSError("synthetic: the grant file could not be written")
        return real(path, mutate, **kwargs)

    monkeypatch.setattr(skill_grant_follow, "update_json_locked", first_binding_write_fails)
    assert restore_update_snapshot(env.root, "hubplug", "review_blockers") is False
    assert failed == ["grants.json"] and find_skill(env.root, "hubplug").content_hash == previous
    kept = json.loads(update_backup_path(env.root, "hubplug").read_text(encoding="utf-8"))
    assert kept["payload_restored"] is True and kept["prior_form_installation"] == allow["installation"]
    assert _grants(env, "hubplug")["granted_keys"] == []  # held until its binding is written, never granted anew
    assert restore_update_snapshot(env.root, "hubplug", "review_blockers") is True  # the ordinary retry
    assert not update_backup_path(env.root, "hubplug").exists()
    assert find_skill(env.root, "hubplug").content_hash == previous
    assert _grants(env, "hubplug")["granted_keys"] == ["HUBPLUG_TOKEN"]
    assert [row["state"] for row in _secret_form_grants(env)] == ["granted"]  # restored, never granted again
    assert read_install_record(env.root, "hubplug")["update_rejected"]["reason"] == "review_blockers"
