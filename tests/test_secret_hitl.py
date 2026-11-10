"""Integration-secret HITL: showSecretHitl asks, the protected form's saveSecretToken saves.

Proved against a real task store, a real Settings file and the real decision ingress: the
value reaches ``settings.json`` under the skill's declared key and NOWHERE else (response,
store, mailbox, history, audit, logs, transport events); the model receives metadata-only
``secret_filled``/``secret_replaced``/``secret_cancelled``; ordinary answer paths refuse the
request; a save writes only while each key is still set or unset as the form showed it, so
Replace is explicit; the form stays usable after its task ended and a late save resumes
nothing; save/cancel/failure/duplicate/interruption keep ``saved`` truthful. The canaries have
no token shape, so no redaction pattern could be what keeps them out of an output.
"""

from __future__ import annotations

import json
import logging
import pathlib
import uuid
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import owner_quiz
from ouroboros import secret_requests as sr
from ouroboros.contracts.skill_manifest import SkillManifestError, parse_skill_manifest_text
from ouroboros.task_results import write_task_result

CANARY = "zq7-CANARY-Ωlj3~p0+kx/2 mid space"
CANARY_B = "Second·canary·entry-9"
TASK = "task-1"

_MANIFEST = """---
name: {name}
description: Test integration.
version: {version}
type: script
runtime: python3
timeout_sec: 30
env_from_settings: [{env}]
secret_fields:
{fields}
scripts:
  - name: run.py
---
# {name}
"""


def _write_skill(root: pathlib.Path, name: str = "fakeplug", *, env: str = "FAKEPLUG_TOKEN, FAKEPLUG_REGION",
                 fields: str = "  botToken: FAKEPLUG_TOKEN", version: str = "0.1.0") -> pathlib.Path:
    skill = root / "skills" / "external" / name
    (skill / "scripts").mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(_MANIFEST.format(name=name, env=env, fields=fields, version=version), encoding="utf-8")
    (skill / "scripts" / "run.py").write_text("print('ok')\n", encoding="utf-8")
    return skill


@pytest.fixture
def env(tmp_path, monkeypatch, caplog):
    import supervisor.message_bus as mb
    import supervisor.queue as q
    from ouroboros import config as cfg
    from ouroboros.gateway import secret_decision as sd
    from ouroboros.gateway import task_decision as td
    from ouroboros.tools import skill_exec

    root = tmp_path
    monkeypatch.setattr(cfg, "DATA_DIR", root)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", root / "settings.json")
    # Import the consumer before patching config: its from-import must not retain this fixture's lambda.
    monkeypatch.setattr(cfg, "get_skills_repo_path", lambda: "")
    monkeypatch.setattr(skill_exec, "get_skills_repo_path", lambda: "")
    cfg.reset_runtime_mode_baseline_for_tests()
    _write_skill(root)
    write_task_result(root, TASK, "running", chat_id=1)
    frames, events = [], []

    class Bridge:
        def send_quiz_state(self, quiz_id, task_id, state, **fields):
            frames.append({"quiz_id": quiz_id, "task_id": task_id, "state": state, **fields})

    monkeypatch.setattr(mb, "get_bridge", lambda: Bridge())
    monkeypatch.setattr(mb, "DATA_DIR", root)
    monkeypatch.setattr(mb, "publish_event", lambda *args: events.append(args))
    # The asker as the queue sees it: reading (""), paused/held ("task_not_reading", a mail holder)
    # or gone ("task_not_live"), with upstream's holds_mail semantics.
    live = {"task": {"id": TASK}, "reason": ""}

    def live_root_task(task_id, *, holds_mail=False):
        if live["task"] is None:
            return None, "task_not_live"
        if live["reason"] == "task_not_reading":
            return (live["task"] if holds_mail else None), "task_not_reading"
        return live["task"], ""

    monkeypatch.setattr(td, "_live_root_task", live_root_task)
    monkeypatch.setattr(q, "_task_drive_for_task", lambda task, task_id: root, raising=False)
    app = Starlette(routes=[Route("/api/decisions/secret", endpoint=sd.api_secret_decision, methods=["POST"]),
                            Route("/api/decisions", endpoint=td.api_decision_answer, methods=["POST"])])
    app.state.drive_root = root
    caplog.set_level(logging.DEBUG)
    yield SimpleNamespace(root=root, client=TestClient(app), frames=frames, events=events, live=live, caplog=caplog)
    cfg.reset_runtime_mode_baseline_for_tests()


def _ask(env, fields=("botToken",), plugin="fakeplug") -> str:
    request = sr.resolve_secret_request(env.root, plugin, list(fields))
    quiz_id = uuid.uuid4().hex
    owner_quiz.record_asked(env.root, TASK, quiz_id=quiz_id, question=sr.question_text(request), options=[],
                            wait_for_answer=True, chat_id=1, secret_request=request)
    owner_quiz.record_publication(env.root, TASK, quiz_id, 1)
    return quiz_id


def _status(env, quiz_id):
    return env.client.post("/api/decisions/secret", json={"decision_id": f"quiz:{TASK}:{quiz_id}", "action": "status"})


def _observed(env, quiz_id):
    """What a form that just read status shows: per field, whether its key is set."""
    return {row["id"]: row["configured"] for row in _status(env, quiz_id).json()["keys"]}


def _save(env, quiz_id, values, request_id="r1", observed=None):
    return env.client.post("/api/decisions/secret", json={
        "request_id": request_id, "decision_id": f"quiz:{TASK}:{quiz_id}", "values": values,
        "observed": _observed(env, quiz_id) if observed is None else observed})


def _cancel(env, quiz_id, request_id="c1"):
    return env.client.post("/api/decisions/secret", json={
        "request_id": request_id, "decision_id": f"quiz:{TASK}:{quiz_id}", "action": "cancel"})


def _settings(env) -> dict:
    path = env.root / "settings.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _put_settings(env, **values):
    (env.root / "settings.json").write_text(json.dumps({**_settings(env), **values}), encoding="utf-8")


def _mail(env):
    from ouroboros.owner_mailbox import drain_owner_entries

    return drain_owner_entries(env.root, TASK, set())


def _block(env, quiz_id):
    return owner_quiz.quiz_states(env.root, TASK)[quiz_id]


def _assert_absent(env, *values, texts=()):
    """No file of the data root but Settings, no log record and no given text holds a value."""
    needles = {form for value in values for form in (
        value.encode(), json.dumps(value)[1:-1].encode(), json.dumps(value, ensure_ascii=False)[1:-1].encode())}
    for path in env.root.rglob("*"):
        if path.is_file() and not path.name.startswith("settings.json"):
            data = path.read_bytes()
            for needle in needles:
                assert needle not in data, f"a submitted value leaked into {path}"
    logged = env.caplog.text.encode()
    for text in [*texts, logged.decode(), json.dumps(env.events, ensure_ascii=False, default=str)]:
        for value in values:
            assert value not in text and json.dumps(value)[1:-1] not in text


# ------------------------------------------------------------------ manifest

def test_canonical_manifest_field_maps_ids_to_declared_env_keys():
    manifest = parse_skill_manifest_text(_MANIFEST.format(
        name="x", version="1", env="X_TOKEN, X_REGION", fields="  botToken: X_TOKEN"))
    assert manifest.secret_fields == {"botToken": "X_TOKEN"}
    assert "secret_fields" not in manifest.raw_extra  # a canonical field, not a forward-compat extra
    for fields, problem in (("  botToken: X_OTHER", "env_from_settings"), ("  9bad: X_TOKEN", "letter"),
                            ("  a: X_TOKEN\n  b: X_TOKEN", "two fields")):
        with pytest.raises(SkillManifestError, match=problem):
            parse_skill_manifest_text(_MANIFEST.format(name="x", version="1", env="X_TOKEN", fields=fields))


@pytest.mark.parametrize("skill,mapping", [("telegram", {"botToken": "TELEGRAM_BOT_TOKEN"}),
                                           ("gemini_live", {"apiKey": "GEMINI_API_KEY"}),
                                           ("grok_realtime", {"apiKey": "XAI_API_KEY"})])
def test_bundled_skills_declare_their_owner_entered_keys(skill, mapping):
    repo = pathlib.Path(__file__).resolve().parents[1]
    manifest = parse_skill_manifest_text((repo / "skills" / skill / "SKILL.md").read_text(encoding="utf-8"))
    assert manifest.secret_fields == mapping and sr.declared_secret_fields(manifest) == mapping


def test_openai_realtime_keeps_its_provider_key_in_settings():
    """OPENAI_API_KEY is a provider credential: no secret form may front it."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    manifest = parse_skill_manifest_text((repo / "skills" / "openai_realtime" / "SKILL.md").read_text(encoding="utf-8"))
    assert manifest.secret_fields == {}


@pytest.mark.parametrize("key", ["OPENROUTER_API_KEY", "OUROBOROS_NETWORK_PASSWORD", "GITHUB_TOKEN", "TOTAL_BUDGET"])
def test_a_plugin_name_cannot_front_a_core_or_provider_setting(env, key):
    _write_skill(env.root, "fronted", env=key, fields=f"  apiKey: {key}")
    with pytest.raises(sr.SecretRequestError, match="custom Settings secret") as caught:
        sr.resolve_secret_request(env.root, "fronted", ["apiKey"])
    assert caught.value.code == "SECRET_FIELDS_INVALID"


def test_requests_name_only_declared_fields_of_an_installed_skill(env):
    for plugin, fields, code in (("missing", ["botToken"], "SECRET_PLUGIN_UNKNOWN"),
                                 ("fakeplug", ["region"], "SECRET_FIELD_UNDECLARED"),
                                 ("fakeplug", ["botToken", "botToken"], "SECRET_FIELDS_INVALID"),
                                 ("fakeplug", [], "SECRET_FIELDS_INVALID")):
        with pytest.raises(sr.SecretRequestError) as caught:
            sr.resolve_secret_request(env.root, plugin, fields)
        assert caught.value.code == code
    request = sr.resolve_secret_request(env.root, "fakeplug", ["botToken"])
    assert request == {"plugin_name": "fakeplug", "fields": [{"id": "botToken", "setting_key": "FAKEPLUG_TOKEN"}]}


def test_secret_fields_is_appended_so_positional_manifest_construction_keeps_every_slot():
    import dataclasses

    from ouroboros.contracts.skill_manifest import SkillManifest

    assert [item.name for item in dataclasses.fields(SkillManifest)][-1] == "secret_fields"
    positional = SkillManifest("n", "d", "1", "script", "", [], "any", "python3", 30, ["K"], [{"name": "run.py"}])
    assert positional.env_from_settings == ["K"] and positional.scripts == [{"name": "run.py"}]
    assert positional.secret_fields == {}


# ------------------------------------------------------------------ the tool

def test_show_secret_hitl_records_a_waiting_metadata_only_question(env, monkeypatch):
    from ouroboros.tools import core_artifacts, owner_delivery
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.secret_hitl import _show_secret_hitl, get_tools

    sent = []
    monkeypatch.setattr(owner_delivery, "deliver_owner_event", lambda ctx, evt: sent.append(evt) or "live")
    monkeypatch.setattr(core_artifacts, "_quiz_host_facts", lambda *args: "Asked by task task-1.")
    ctx = ToolContext(repo_dir=env.root, drive_root=env.root)
    ctx.task_id, ctx.current_chat_id, ctx.owner_wait_callback = TASK, 1, lambda *args: "answer"
    [entry] = get_tools()
    assert entry.name == "showSecretHitl" and entry.safety_policy == "skip"
    assert entry.schema["parameters"]["required"] == ["pluginName", "fields"]

    text = _show_secret_hitl(ctx, pluginName="fakeplug", fields=["botToken"])
    assert text.startswith("OK: secret request ") and "secret_filled or secret_replaced" in text
    assert "If this task ends first, the form stays usable" in text and "stops saving" not in text
    [event] = sent
    assert event["type"] == "send_quiz" and event["wait_for_answer"] is True and event["options"] == []
    block = _block(env, event["quiz_id"])
    assert block["secret_request"] == {"plugin_name": "fakeplug",
                                       "fields": [{"id": "botToken", "setting_key": "FAKEPLUG_TOKEN"}]}
    assert block["wait_for_answer"] is True and ctx._owner_wait_requested == event["quiz_id"]

    child = ToolContext(repo_dir=env.root, drive_root=env.root, task_metadata={"parent_task_id": "root"})
    child.task_id, child.owner_wait_callback = "child", lambda *args: ""
    assert "CAPABILITY_UNAVAILABLE" in _show_secret_hitl(child, pluginName="fakeplug", fields=["botToken"])
    refused = _show_secret_hitl(ctx, pluginName="fakeplug", fields=["FAKEPLUG_TOKEN"])
    assert "SECRET_FIELD_UNDECLARED" in refused and len(sent) == 1


# ------------------------------------------------------------- status / save / cancel

def test_status_names_key_states_and_readers_without_any_value(env):
    _write_skill(env.root, "reader", env="FAKEPLUG_TOKEN", fields="  token: FAKEPLUG_TOKEN")
    quiz_id = _ask(env)
    body = _status(env, quiz_id).json()
    assert body["state"] == "open" and body["keys"] == [{"id": "botToken", "setting_key": "FAKEPLUG_TOKEN",
                                                        "configured": False}]
    assert [row["name"] for row in body["consumers"]] == ["fakeplug", "reader"]
    assert all(row["granted"] == [] and row["reads"] == ["FAKEPLUG_TOKEN"] for row in body["consumers"])
    _put_settings(env, FAKEPLUG_TOKEN=CANARY)
    response = _status(env, quiz_id)
    assert response.json()["keys"][0]["configured"] is True
    assert response.headers["cache-control"] == "no-store"
    _assert_absent(env, CANARY, texts=[response.text])
    assert env.client.post("/api/decisions/secret", json={"decision_id": f"quiz:{TASK}:{quiz_id}", "action": "status",
                                                          "values": {"botToken": CANARY}}).status_code == 400


def test_save_writes_settings_only_and_delivers_metadata_only_secret_filled(env):
    quiz_id = _ask(env)
    response = _save(env, quiz_id, {"botToken": CANARY})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["saved"] is True and body["state"] == "answered" and body["duplicate"] is False
    assert body["secret_filled"] == {"pluginName": "fakeplug", "fields": ["botToken"]}
    [reader] = body["secret_outcome"]["consumers"]
    assert reader["name"] == "fakeplug" and reader["granted"] == []  # saving granted nothing
    assert "replaced_fields" not in body["secret_outcome"]
    assert response.headers["cache-control"] == "no-store"
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY

    [mail] = [entry for entry in _mail(env) if entry.get("kind") == "quiz_answer"]
    assert 'secret_filled: {"pluginName": "fakeplug", "fields": ["botToken"]}' in mail["text"]
    assert "saved it to Settings (FAKEPLUG_TOKEN)" in mail["text"] and "do not ask for it here" in mail["text"]
    assert "Saving is not a connection" in mail["text"] and "needs the owner's grant for FAKEPLUG_TOKEN" in mail["text"]
    assert "Verbatim" not in mail["text"]
    block = _block(env, quiz_id)
    assert block["secret_outcome"]["event"] == "secret_filled" and "comment" not in block
    assert block["secret_save"]["state"] == "saved"
    assert owner_quiz.late_answer_owner_text(block) == ""  # never speakable as the owner's words
    from tests import _dialogue

    history = [row for row in _dialogue.chat_rows(env.root) if row.get("type") == "quiz_answer"]
    assert history and "saved it to Settings (FAKEPLUG_TOKEN)" in history[-1]["text"]
    assert any(frame.get("secret_outcome", {}).get("event") == "secret_filled" for frame in env.frames)
    audit = [json.loads(line) for line in (env.root / "logs" / "events.jsonl").read_text().splitlines()]
    assert any(row.get("action") == "secret_request_save" and row.get("fields") == ["botToken"] for row in audit)
    _assert_absent(env, CANARY, texts=[response.text, json.dumps(env.frames, ensure_ascii=False)])

    replay = _save(env, quiz_id, {"botToken": CANARY}, observed={"botToken": False})  # the same tab's retry
    assert replay.status_code == 200 and replay.json()["duplicate"] is True
    assert len([entry for entry in _mail(env) if entry.get("kind") == "quiz_answer"]) == 1
    loser = _save(env, quiz_id, {"botToken": CANARY_B}, request_id="other-tab", observed={"botToken": True})
    assert loser.status_code == 409 and loser.json()["saved"] is False
    assert loser.json()["reason_code"] == "secret_request_settled" and loser.json()["secret_filled"]["fields"] == ["botToken"]
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY
    _assert_absent(env, CANARY, CANARY_B, texts=[replay.text, loser.text])


def test_a_stale_empty_form_never_replaces_a_key_saved_since_and_replace_is_explicit(env):
    """Save -> Replace: the form showed no key; another writer saved one; the empty form's save is
    refused with the fresh key state, nothing written, and only an explicit Replace overwrites."""
    _put_settings(env, UNRELATED_SETTING="keep")
    first, second = _ask(env), _ask(env)
    shown = _observed(env, second)
    assert shown == {"botToken": False}
    assert _save(env, first, {"botToken": CANARY}).status_code == 200  # saved elsewhere meanwhile
    stale = _save(env, second, {"botToken": CANARY_B}, request_id="stale", observed=shown)
    body = stale.json()
    assert stale.status_code == 409 and body["reason_code"] == "secret_key_state_changed" and body["saved"] is False
    assert body["keys"] == [{"id": "botToken", "setting_key": "FAKEPLUG_TOKEN", "configured": True}]
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY and "secret_save" not in _block(env, second)
    # An unrelated Settings change is not this request's observation.
    _put_settings(env, UNRELATED_SETTING="changed elsewhere")
    replaced = _save(env, second, {"botToken": CANARY_B}, request_id="replace", observed={"botToken": True})
    assert replaced.status_code == 200 and replaced.json()["secret_replaced"]["fields"] == ["botToken"]
    assert replaced.json()["secret_outcome"]["replaced_fields"] == ["botToken"]
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY_B and _settings(env)["UNRELATED_SETTING"] == "changed elsewhere"
    mail = [entry for entry in _mail(env) if entry.get("kind") == "quiz_answer"][-1]
    assert "secret_replaced" in mail["text"] and "replaced the existing value" in mail["text"]
    assert "some keep the old one until the owner disables and re-enables them" in mail["text"]
    # A form that showed a set key while it was cleared is refused the other way round.
    third = _ask(env)
    _put_settings(env, FAKEPLUG_TOKEN="")
    cleared = _save(env, third, {"botToken": CANARY}, request_id="c", observed={"botToken": True})
    assert cleared.json()["reason_code"] == "secret_key_state_changed"
    assert cleared.json()["keys"][0]["configured"] is False and _settings(env)["FAKEPLUG_TOKEN"] == ""
    _assert_absent(env, CANARY, CANARY_B, texts=[stale.text, replaced.text, cleared.text])


def test_replace_while_readers_are_enabled_reports_them_and_grants_nothing(env):
    from ouroboros.skill_loader import find_skill, grant_status_for_skill, save_enabled, save_skill_grants

    _put_settings(env, FAKEPLUG_TOKEN="old-synthetic", FAKEPLUG_REGION="eu")
    skill = find_skill(env.root, "fakeplug")
    save_enabled(env.root, "fakeplug", True)
    save_skill_grants(env.root, "fakeplug", ["FAKEPLUG_TOKEN", "FAKEPLUG_REGION"], content_hash=skill.content_hash,
                      requested_keys=["FAKEPLUG_TOKEN", "FAKEPLUG_REGION"])
    before = grant_status_for_skill(env.root, find_skill(env.root, "fakeplug"))
    quiz_id = _ask(env)
    status = _status(env, quiz_id).json()
    assert status["keys"][0]["configured"] is True
    assert status["consumers"] == [{"name": "fakeplug", "enabled": True, "reads": ["FAKEPLUG_TOKEN"],
                                    "granted": ["FAKEPLUG_TOKEN"]}]
    response = _save(env, quiz_id, {"botToken": CANARY})
    assert response.status_code == 200 and response.json()["secret_outcome"]["event"] == "secret_replaced"
    assert response.json()["secret_outcome"]["consumers"][0]["enabled"] is True
    after = grant_status_for_skill(env.root, find_skill(env.root, "fakeplug"))
    assert after["granted_keys"] == before["granted_keys"] and after["missing_keys"] == before["missing_keys"]
    _assert_absent(env, CANARY, texts=[response.text])


def test_a_first_fill_of_a_shared_absent_key_keeps_other_approvals(env):
    """A general first use (a key like GEMINI_API_KEY absent from Settings): the save makes every
    declaring skill request it, missing; the other keys those skills were approved for stay."""
    from ouroboros.skill_loader import find_skill, grant_status_for_skill, save_skill_grants

    _write_skill(env.root, "mailer", env="MAILER_HOST_KEY, FAKEPLUG_TOKEN", fields="  token: FAKEPLUG_TOKEN")
    _put_settings(env, MAILER_HOST_KEY="host-synthetic")
    mailer = find_skill(env.root, "mailer")
    save_skill_grants(env.root, "mailer", ["MAILER_HOST_KEY"], content_hash=mailer.content_hash,
                      requested_keys=["MAILER_HOST_KEY"])
    quiz_id = _ask(env)
    response = _save(env, quiz_id, {"botToken": CANARY})
    assert response.status_code == 200
    readers = {row["name"]: row for row in response.json()["secret_outcome"]["consumers"]}
    assert readers["mailer"]["granted"] == [] and readers["fakeplug"]["granted"] == []
    status = grant_status_for_skill(env.root, find_skill(env.root, "mailer"))
    assert status["granted_keys"] == ["MAILER_HOST_KEY"] and status["missing_keys"] == ["FAKEPLUG_TOKEN"]
    _assert_absent(env, CANARY, texts=[response.text])


def test_cancel_saves_nothing_and_wakes_the_task_with_a_distinct_cancellation(env):
    quiz_id = _ask(env)
    response = _cancel(env, quiz_id)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "closed" and body["saved"] is False
    assert body["secret_cancelled"] == {"pluginName": "fakeplug", "fields": ["botToken"]}
    [mail] = [entry for entry in _mail(env) if entry.get("kind") == "quiz_closed"]
    assert mail["text"].startswith("[Owner secret request cancelled]")
    assert 'secret_cancelled: {"pluginName": "fakeplug", "fields": ["botToken"]}' in mail["text"]
    assert "[Host note] The task can tell the owner what is still missing" in mail["text"]
    assert "unknown" not in mail["text"]
    late = _save(env, quiz_id, {"botToken": CANARY}, observed={"botToken": False})
    assert late.status_code == 409 and late.json()["reason_code"] == "secret_request_settled"
    assert "FAKEPLUG_TOKEN" not in _settings(env)
    assert _cancel(env, quiz_id).json()["duplicate"] is True
    _assert_absent(env, CANARY, texts=[late.text])


def test_a_save_first_wins_over_a_later_cancel(env):
    quiz_id = _ask(env)
    assert _save(env, quiz_id, {"botToken": CANARY}).status_code == 200
    cancel = _cancel(env, quiz_id)
    assert cancel.status_code == 409 and cancel.json()["state"] == "answered"
    assert cancel.json()["secret_filled"]["pluginName"] == "fakeplug"
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY


def test_ordinary_answer_paths_refuse_options_and_comments_without_echo(env):
    import asyncio

    from ouroboros.gateway.task_decision import answer_decision

    quiz_id = _ask(env)
    decision = f"quiz:{TASK}:{quiz_id}"
    for body in ({"request_id": "a1", "decision_id": decision, "comment": CANARY_B},
                 {"request_id": "a2", "decision_id": decision, "option_index": 0, "comment": CANARY_B}):
        response = env.client.post("/api/decisions", json=body)
        assert response.status_code == 409 and response.json()["reason_code"] == "secret_request_form_only"
        assert CANARY_B not in response.text
    # The host-service relay (Telegram's /chat/decision) reaches the same ingress.
    status, payload = asyncio.run(answer_decision(env.root, {"request_id": "a3", "decision_id": decision,
                                                             "comment": CANARY_B}, source="telegram"))
    assert status == 409 and payload["reason_code"] == "secret_request_form_only"
    direct = owner_quiz.record_answered(env.root, TASK, quiz_id=quiz_id, option_index=None,
                                        request_id="a4", comment=CANARY_B)
    assert direct["ok"] is False and direct["error"] == "secret_request_form_only"
    block = _block(env, quiz_id)
    assert block["state"] == "open" and "comment" not in block and "secret_outcome" not in block
    assert not _mail(env)
    # The owner's generic close of the card is its cancel.
    close = env.client.post("/api/decisions", json={"request_id": "x1", "decision_id": decision, "action": "close"})
    assert close.status_code == 200 and close.json()["secret_cancelled"]["fields"] == ["botToken"]
    _assert_absent(env, CANARY_B, texts=[json.dumps(payload, ensure_ascii=False)])


def test_wrong_request_and_cross_task_identities_write_nothing(env):
    quiz_id = _ask(env)
    write_task_result(env.root, "task-2", "running", chat_id=1)
    other = env.client.post("/api/decisions/secret", json={"request_id": "x", "decision_id": f"quiz:task-2:{quiz_id}",
                                                           "values": {"botToken": CANARY}, "observed": {"botToken": False}})
    assert other.status_code == 404 and other.json()["saved"] is False
    plain = uuid.uuid4().hex
    owner_quiz.record_asked(env.root, TASK, quiz_id=plain, question="Which?", options=["a", "b"], chat_id=1)
    ordinary = env.client.post("/api/decisions/secret", json={"request_id": "y", "decision_id": f"quiz:{TASK}:{plain}",
                                                              "values": {"botToken": CANARY}, "observed": {"botToken": False}})
    assert ordinary.status_code == 409 and ordinary.json()["reason_code"] == "not_a_secret_request"
    for broken in ({"decision_id": f"quiz:{TASK}:{quiz_id}", "values": {"botToken": CANARY}, "observed": {"botToken": False}},
                   {"request_id": "z", "decision_id": "quiz:../etc:x", "values": {"botToken": CANARY}},
                   {"request_id": "z", "decision_id": f"quiz:{TASK}:{quiz_id}", "values": {"botToken": CANARY}},
                   {"request_id": "z", "decision_id": f"quiz:{TASK}:{quiz_id}", "values": {"botToken": CANARY},
                    "observed": {"botToken": False}, "extra": CANARY}):
        refused = env.client.post("/api/decisions/secret", json=broken)
        assert refused.status_code == 400 and refused.json()["saved"] is False, broken
        assert CANARY not in refused.text
    assert not _settings(env) and _block(env, quiz_id)["state"] == "open"
    _assert_absent(env, CANARY, texts=[other.text, ordinary.text])


def test_values_must_match_the_request_and_be_keepable(env):
    quiz_id = _ask(env)
    cases = ({"botToken": "abcdefgh..."}, {"botToken": "***set***"}, {"botToken": "  "},
             {"botToken": "line\nbreak"}, {"botToken": CANARY, CANARY_B: "x"}, {})
    for values in cases:
        response = _save(env, quiz_id, values)
        assert response.status_code == 400 and response.json()["saved"] is False
        assert CANARY_B not in response.text and CANARY not in response.text
    assert [_save(env, quiz_id, values).json()["reason_code"] for values in cases[:2]] == [
        "secret_value_placeholder", "secret_value_placeholder"]
    assert not _settings(env) and _block(env, quiz_id)["state"] == "open"
    assert "secret_save" not in _block(env, quiz_id)
    _assert_absent(env, CANARY, CANARY_B)


def test_an_old_request_never_gains_a_changed_mapping_but_survives_an_unrelated_update(env):
    kept = _ask(env)
    _write_skill(env.root, version="0.2.0")  # new bytes, same mapping
    assert _save(env, kept, {"botToken": CANARY}).status_code == 200
    remapped = _ask(env)
    _write_skill(env.root, env="FAKEPLUG_TOKEN, FAKEPLUG_OTHER_TOKEN", fields="  botToken: FAKEPLUG_OTHER_TOKEN",
                 version="0.3.0")
    response = _save(env, remapped, {"botToken": CANARY_B}, observed={"botToken": True})
    assert response.status_code == 409 and response.json()["reason_code"] == "secret_request_stale"
    assert "secret_save" not in _block(env, remapped) and "FAKEPLUG_OTHER_TOKEN" not in _settings(env)
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY
    _assert_absent(env, CANARY, CANARY_B)


def test_a_finished_task_keeps_the_form_and_a_late_save_resumes_nothing(env, monkeypatch):
    """Owner choice: the form stays usable after the asking task ends. The save is recorded on
    the card and in history as a host fact; no owner words, no fresh head, no mailbox."""
    import supervisor.message_bus as mb

    spoken = []
    monkeypatch.setattr(mb, "accept_local_message", lambda *args, **kwargs: spoken.append(args) or ({}, False))
    finished = _ask(env)
    owner_quiz.reconcile_terminal(env.root, TASK)
    env.live["task"] = None
    assert _status(env, finished).json()["state"] == "expired_terminal"
    response = _save(env, finished, {"botToken": CANARY})
    assert response.status_code == 200 and response.json()["saved"] is True
    assert response.json()["answered_after_terminal"] is True
    block = _block(env, finished)
    assert block["state"] == "answered" and block["answered_after_terminal"] is True
    assert block["delivery"]["state"] == "delivered" and block["delivery"]["route"] == "history"
    assert not spoken and not _mail(env)
    from ouroboros.dialogue_provenance import row_author
    from tests import _dialogue

    [row] = [row for row in _dialogue.chat_rows(env.root) if row.get("type") == "quiz_answer"]
    assert row_author(row)["kind"] == "host"
    assert "this save resumed nothing" in row["text"]
    _assert_absent(env, CANARY, texts=[response.text])


def test_a_finished_asker_before_its_terminal_reconcile_is_settled_then_saved_late(env, monkeypatch):
    """The task-done seam has not expired the block yet (crash or post-work window): the save reads
    the live queue, settles the question as an ordinary answer does, and records a late save."""
    import supervisor.message_bus as mb

    monkeypatch.setattr(mb, "accept_local_message", lambda *args, **kwargs: pytest.fail("no owner turn"))
    quiz_id = _ask(env)
    env.live["task"] = None
    assert _block(env, quiz_id)["state"] == "open"
    response = _save(env, quiz_id, {"botToken": CANARY})
    assert response.status_code == 200 and response.json()["saved"] is True
    block = _block(env, quiz_id)
    assert block["answered_after_terminal"] is True and not _mail(env)
    assert any(frame["state"] == "expired_terminal" for frame in env.frames)
    _assert_absent(env, CANARY)


def test_a_refused_settings_write_saves_nothing_and_stays_retryable(env, monkeypatch):
    from ouroboros.gateway import owner_settings

    quiz_id = _ask(env)
    real_update, calls = owner_settings._owner_update_settings, []

    def locked_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise owner_settings.SettingsLockUnavailable("another process holds the settings lock")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(owner_settings, "_owner_update_settings", locked_once)
    refused = _save(env, quiz_id, {"botToken": CANARY})
    assert refused.status_code == 503 and refused.json() == {
        "ok": False, "saved": False, "reason_code": "settings_locked", "decision_id": f"quiz:{TASK}:{quiz_id}",
        "error": "Settings could not be written (SettingsLockUnavailable); nothing was saved. Try again."}
    block = _block(env, quiz_id)
    assert block["state"] == "open" and "secret_save" not in block and "secret_outcome" not in block
    assert not _mail(env) and not any(frame.get("secret_outcome") for frame in env.frames)
    retry = _save(env, quiz_id, {"botToken": CANARY})
    assert retry.status_code == 200 and retry.json()["saved"] is True
    _assert_absent(env, CANARY, texts=[refused.text])


def test_interruption_after_the_settings_commit_stays_saved_and_unknown_never_reposted(env, monkeypatch):
    quiz_id = _ask(env)
    real_finalize = sr.finalize_save

    def crash(*_args, **_kwargs):
        raise RuntimeError("store unavailable after the settings commit")

    monkeypatch.setattr(sr, "finalize_save", crash)
    response = _save(env, quiz_id, {"botToken": CANARY})
    assert response.status_code == 500
    body = response.json()
    assert body["saved"] is True and body["reason_code"] == "secret_completion_unrecorded"
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY
    block = _block(env, quiz_id)
    # The durable pre-write marker survives the process: unknown, never reported saved.
    assert block["state"] == "open" and block["secret_save"]["state"] == "writing"
    assert block["secret_save"]["request_id"] == "r1" and "secret_outcome" not in block
    assert not _mail(env)
    monkeypatch.setattr(sr, "finalize_save", real_finalize)
    # Another path may not close it underneath a claimed write.
    model = owner_quiz.record_closed(env.root, TASK, quiz_id=quiz_id, reason="done", actor_task_id=TASK)
    assert model["ok"] is False and model["error"] == "secret_save_in_progress"
    # Status first: it says unknown, and a set key does not prove this attempt saved it.
    status = _status(env, quiz_id).json()
    assert status["save_unconfirmed"] is True and status["keys"][0]["configured"] is True
    repeat = _save(env, quiz_id, {"botToken": CANARY})
    assert repeat.status_code == 409 and repeat.json()["saved"] is None
    assert repeat.json()["reason_code"] == "secret_save_unconfirmed" and repeat.json()["save_unconfirmed"] is True
    assert not _mail(env)
    # The owner's explicit save-again is a new request over whatever it stored: a Replace now.
    again = _save(env, quiz_id, {"botToken": CANARY}, request_id="r2")
    assert again.status_code == 200 and again.json()["saved"] is True
    assert again.json()["secret_outcome"]["event"] == "secret_replaced"
    assert _block(env, quiz_id)["secret_save"]["unconfirmed_request_ids"] == ["r1"]
    assert len([entry for entry in _mail(env) if entry.get("kind") == "quiz_answer"]) == 1
    _assert_absent(env, CANARY, texts=[response.text, repeat.text, again.text])


@pytest.mark.parametrize("failure", ["unreadable", "mismatch"])
def test_a_read_back_that_does_not_confirm_is_written_but_never_reported_saved(env, monkeypatch, failure):
    from ouroboros.gateway import owner_settings

    quiz_id = _ask(env)
    real_document = owner_settings._owner_settings_document

    def read_back(**kwargs):
        document = real_document(**kwargs)
        if document.get("FAKEPLUG_TOKEN") != CANARY:  # the status read, before the write
            return document
        if failure == "unreadable":
            raise OSError("settings read failed")
        return {**document, "FAKEPLUG_TOKEN": "something-else"}

    monkeypatch.setattr(owner_settings, "_owner_settings_document", read_back)
    response = _save(env, quiz_id, {"botToken": CANARY})
    assert response.status_code == 500 and response.json()["saved"] is True
    assert response.json()["reason_code"] == {"unreadable": "secret_save_unverified",
                                              "mismatch": "secret_not_retained"}[failure]
    block = _block(env, quiz_id)
    assert block["state"] == "open" and block["secret_save"]["state"] == "writing"
    assert "secret_outcome" not in block and not _mail(env)
    _assert_absent(env, CANARY, texts=[response.text])


def test_an_interrupted_claim_is_taken_over_or_cancelled_with_its_unknown_disclosed(env):
    taken = _ask(env)
    assert sr.claim_save(env.root, TASK, taken, "dead-process")["ok"] is True  # then the process died
    response = _save(env, taken, {"botToken": CANARY}, request_id="new")
    assert response.status_code == 200
    assert _block(env, taken)["secret_save"]["unconfirmed_request_ids"] == ["dead-process"]
    cancelled = _ask(env)
    sr.claim_save(env.root, TASK, cancelled, "dead-process")
    body = _cancel(env, cancelled).json()
    # The cancel saved nothing itself, but cannot say nothing is stored: saved is unknown.
    assert body["state"] == "closed" and body["secret_outcome"]["interrupted_save"] is True and body["saved"] is None
    assert "unknown" in body["closed_reason"] and "nothing" not in body["closed_reason"]
    [mail] = [entry for entry in _mail(env) if entry.get("kind") == "quiz_closed"]
    assert "whether its value reached Settings (FAKEPLUG_TOKEN) is unknown" in mail["text"]
    assert "saved nothing for" not in mail["text"] and "unchanged" not in mail["text"]


def test_recovery_redelivers_a_saved_secret_through_its_own_path(env, monkeypatch):
    import asyncio

    from ouroboros import owner_mailbox
    from ouroboros.gateway.task_decision import recover_question_deliveries

    quiz_id = _ask(env)
    real_write = owner_mailbox.write_owner_message
    monkeypatch.setattr(owner_mailbox, "write_owner_message", lambda *args, **kwargs: False)
    assert _save(env, quiz_id, {"botToken": CANARY}).json()["saved"] is True
    assert _block(env, quiz_id)["delivery"]["reason_code"] == "mailbox_write_failed"
    monkeypatch.setattr(owner_mailbox, "write_owner_message", real_write)
    row = owner_quiz.quiz_states(env.root, TASK)
    row[quiz_id]["delivery"]["next_retry_at"] = "2000-01-01T00:00:00+00:00"
    write_task_result(env.root, TASK, "running", owner_quiz=row)
    asyncio.run(recover_question_deliveries(env.root))
    [mail] = [entry for entry in _mail(env) if entry.get("kind") == "quiz_answer"]
    assert "saved it to Settings (FAKEPLUG_TOKEN)" in mail["text"]
    from ouroboros.owner_quiz import SECRET_OUTCOME_ROW_KIND

    ctx, messages = _drain(env)  # the recovered delivery is the same typed host row
    assert SECRET_OUTCOME_ROW_KIND in _kinds(messages) and ctx._owner_directives == []
    _assert_absent(env, CANARY)


def test_transports_get_a_hint_never_an_answerable_secret_card(env):
    import supervisor.message_bus as mb

    quiz_id = uuid.uuid4().hex
    request = sr.resolve_secret_request(env.root, "fakeplug", ["botToken"])
    owner_quiz.record_asked(env.root, TASK, quiz_id=quiz_id, question=sr.question_text(request), options=[],
                            wait_for_answer=True, chat_id=1, secret_request=request)
    bridge = mb.LocalChatBridge({})
    shown = []
    bridge._broadcast_fn = shown.append
    ok, _ = bridge.send_quiz(1, quiz_id=quiz_id, question=sr.question_text(request), options=[], task_id=TASK,
                             wait_for_answer=True)
    assert ok
    [frame] = [frame for frame in shown if frame.get("type") == "quiz"]
    assert frame["secret_request"]["plugin_name"] == "fakeplug" and frame["question"] == sr.question_text(request)
    [(_topic, event)] = [event for event in env.events if event[0] == "chat.quiz"]
    assert event["options"] == [] and event["secret_request"] == request
    assert event["question"] == sr.transport_hint(request)
    assert "protected form in the Ouroboros app" in event["question"] and "http" not in event["question"]
    bridge.send_quiz_state(quiz_id, TASK, "answered", secret_outcome={"event": "secret_filled"})
    [(_topic, state)] = [event for event in env.events if event[0] == "chat.quiz_state"]
    assert state["state"] == "answered" and state["secret_outcome"] == {"event": "secret_filled"}
    from ouroboros.gateway.history import _assemble_history_response

    replay = json.loads(_assemble_history_response(env.root, 1, 10, 0))["messages"]
    assert any(row.get("quiz", {}).get("secret_request", {}).get("plugin_name") == "fakeplug" for row in replay)


def test_the_leak_scan_is_not_vacuous(env):
    """The absence assertions above prove something only if the scan finds an escaped echo."""
    probe = env.root / "logs" / "probe.jsonl"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(json.dumps({"echo": CANARY}) + "\n", encoding="utf-8")  # ASCII-escaped form
    with pytest.raises(AssertionError, match="leaked into"):
        _assert_absent(env, CANARY)
    probe.unlink()
    with pytest.raises(AssertionError):
        _assert_absent(env, CANARY, texts=[f"error: {CANARY}"])


def test_simultaneous_saves_and_cancel_have_exactly_one_winner(env):
    import threading

    from ouroboros.gateway.secret_decision import save_secret

    quiz_id = _ask(env)
    decision = f"quiz:{TASK}:{quiz_id}"
    shown = {"botToken": False}
    bodies = [{"request_id": "tab-a", "decision_id": decision, "values": {"botToken": CANARY}, "observed": shown},
              {"request_id": "tab-b", "decision_id": decision, "values": {"botToken": CANARY_B}, "observed": shown},
              {"request_id": "tab-c", "decision_id": decision, "action": "cancel"}]
    results, gate = {}, threading.Barrier(len(bodies))

    def run(body):
        gate.wait()
        results[body["request_id"]] = save_secret(env.root, body)

    threads = [threading.Thread(target=run, args=(body,)) for body in bodies]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    winners = [rid for rid, (status, _payload) in results.items() if status == 200]
    assert len(winners) == 1, results
    block = _block(env, quiz_id)
    if winners == ["tab-c"]:
        assert block["state"] == "closed" and "FAKEPLUG_TOKEN" not in _settings(env)
    else:
        assert block["state"] == "answered" and block["request_id"] == winners[0]
        assert _settings(env)["FAKEPLUG_TOKEN"] == {"tab-a": CANARY, "tab-b": CANARY_B}[winners[0]]
    assert all(status == 409 for rid, (status, _payload) in results.items() if rid not in winners)
    _assert_absent(env, CANARY, CANARY_B, texts=[json.dumps(results, ensure_ascii=False)])


def test_a_task_ending_mid_save_keeps_the_outcome_without_inventing_an_owner_turn(env, monkeypatch):
    import supervisor.message_bus as mb
    from ouroboros.gateway import owner_settings

    quiz_id = _ask(env)
    shown = _observed(env, quiz_id)
    real_update = owner_settings._owner_update_settings

    def ends_then_writes(*args, **kwargs):
        owner_quiz.reconcile_terminal(env.root, TASK)  # the asking task finishes during the write
        env.live["task"] = None
        return real_update(*args, **kwargs)

    monkeypatch.setattr(owner_settings, "_owner_update_settings", ends_then_writes)
    spoken = []
    monkeypatch.setattr(mb, "accept_local_message", lambda *args, **kwargs: spoken.append(args) or ({}, False))
    response = _save(env, quiz_id, {"botToken": CANARY}, observed=shown)
    assert response.status_code == 200 and response.json()["saved"] is True
    block = _block(env, quiz_id)
    assert block["state"] == "answered" and block["answered_after_terminal"] is True
    assert block["delivery"]["state"] == "delivered" and block["delivery"]["history_at"]
    assert not spoken and not _mail(env)  # the history row is the record; no owner words were made up
    from ouroboros.dialogue_provenance import row_author
    from tests import _dialogue

    [row] = [row for row in _dialogue.chat_rows(env.root) if row.get("type") == "quiz_answer"]
    assert row_author(row)["kind"] == "host"  # the late record is the host's, never the owner's
    _assert_absent(env, CANARY)


def test_the_waiting_task_wakes_with_metadata_only_secret_filled_in_its_next_model_input(env, monkeypatch):
    """Public tool -> real publication -> the loop's own wait -> the owner's direct POST during it
    -> the same task's next model input carries typed ``secret_filled`` and never the value."""
    import queue

    import supervisor.message_bus as mb
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_wait import wait_after_tools
    from ouroboros.task_results import load_task_result
    from ouroboros.tools import core_artifacts
    from ouroboros.tools.secret_hitl import _show_secret_hitl
    from supervisor.events_chat_delivery import _handle_send_quiz
    from tests.test_native_owner_wait import native_context

    ctx = native_context(env.root)
    ctx.current_chat_id, ctx.task_metadata, ctx._owner_wait_requested = 1, {}, ""
    monkeypatch.setattr(core_artifacts, "_quiz_host_facts", lambda *args: "Asked by task root-1.")
    result = _show_secret_hitl(ctx, pluginName="fakeplug", fields=["botToken"])
    assert result.startswith("OK: secret request "), result
    event = ctx.event_queue.get_nowait()
    shown = []
    bridge = mb.LocalChatBridge({})
    bridge._broadcast_fn = shown.append
    _handle_send_quiz(event, SimpleNamespace(DRIVE_ROOT=env.root, bridge=bridge))
    assert any(frame.get("type") == "quiz" and frame.get("secret_request") for frame in shown)
    quiz_id = event["quiz_id"]
    env.live["task"] = {"id": ctx.task_id}
    posted = []

    def owner_fills_the_form(_seconds):
        if not posted:
            decision = f"quiz:{ctx.task_id}:{quiz_id}"
            status = env.client.post("/api/decisions/secret", json={"decision_id": decision, "action": "status"})
            observed = {row["id"]: row["configured"] for row in status.json()["keys"]}
            posted.append(env.client.post("/api/decisions/secret", json={
                "request_id": "form-1", "decision_id": decision, "values": {"botToken": CANARY},
                "observed": observed}))

    monkeypatch.setattr("ouroboros.owner_wait.time.sleep", owner_fills_the_form)
    messages = [{"role": "tool", "tool_call_id": "c1", "content": result}]
    wait_after_tools(ctx, messages, {}, {}, 1, [], set())
    assert posted and posted[0].status_code == 200 and posted[0].json()["saved"] is True
    wait = load_task_result(env.root, ctx.task_id)["owner_wait"]
    assert wait["state"] == "resumed" and wait["resume_reason"] == "answer" and wait["quiz_id"] == quiz_id
    assert ctx._owner_wait_requested == ""

    directives, before = list(ctx._owner_directives), len(messages)
    _drain_incoming_messages(messages, queue.Queue(), env.root, ctx.task_id, None, ctx._loop_mailbox_seen_ids,
                             owner_ctx=ctx)
    model_input = "\n".join(str(row.get("content") or "") for row in messages[before:])
    assert 'secret_filled: {"pluginName": "fakeplug", "fields": ["botToken"]}' in model_input
    assert "saved it to Settings (FAKEPLUG_TOKEN)" in model_input and "do not ask for it here" in model_input
    assert ctx._owner_directives == directives  # no owner words were made up for the host's outcome
    from ouroboros.owner_quiz import SECRET_OUTCOME_ROW_KIND

    assert SECRET_OUTCOME_ROW_KIND in _kinds(messages[before:]) and "owner_words" not in _kinds(messages[before:])
    assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY
    _assert_absent(env, CANARY, texts=[json.dumps(messages, ensure_ascii=False), posted[0].text,
                                       json.dumps(shown, ensure_ascii=False)])


# ------------------------------------------------------- typed host provenance

def _drain(env):
    import queue

    from ouroboros.loop_round_limits import _drain_incoming_messages

    ctx = SimpleNamespace(task_id=TASK, task_metadata={}, drive_root=env.root, budget_drive_root=env.root,
                          _owner_directives=[])
    messages = []
    _drain_incoming_messages(messages, queue.Queue(), env.root, TASK, None, set(), owner_ctx=ctx)
    return ctx, messages


def _kinds(messages):
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY

    return [row.get(HOST_CONTEXT_KIND_KEY) for row in messages]


def test_a_saved_secret_is_a_typed_host_row_and_adds_nothing_to_the_owner_corpus(env):
    from ouroboros.context_budget import OWNER_WORDS_KIND
    from ouroboros.owner_quiz import SECRET_OUTCOME_ROW_KIND

    quiz_id = _ask(env)
    assert _save(env, quiz_id, {"botToken": CANARY}).status_code == 200
    ctx, messages = _drain(env)
    secret_rows = [row for row in messages if "secret_filled:" in str(row.get("content"))]
    assert len(secret_rows) == 1 and _kinds(secret_rows) == [SECRET_OUTCOME_ROW_KIND]
    assert OWNER_WORDS_KIND not in _kinds(messages)
    assert ctx._owner_directives == []  # the owner typed only the value; no owner words exist
    _assert_absent(env, CANARY, texts=[json.dumps(messages, ensure_ascii=False)])


@pytest.mark.parametrize("interrupted", [False, True])
def test_a_cancelled_secret_frame_is_host_typed_and_only_the_owner_act_is_retained(env, interrupted):
    from ouroboros.context_budget import OWNER_WORDS_KIND
    from ouroboros.owner_quiz import SECRET_OUTCOME_ROW_KIND

    quiz_id = _ask(env)
    if interrupted:
        sr.claim_save(env.root, TASK, quiz_id, "dead-process")
    assert _cancel(env, quiz_id).status_code == 200
    [mail] = [entry for entry in _mail(env) if entry.get("kind") == "quiz_closed"]
    assert mail["quiz_delivery"]["answer_snapshot"]["secret_outcome"]["event"] == "secret_cancelled"
    ctx, messages = _drain(env)
    frames = [row for row in messages if "[Owner secret request cancelled]" in str(row.get("content"))]
    assert len(frames) == 1 and _kinds(frames) == [SECRET_OUTCOME_ROW_KIND]
    assert OWNER_WORDS_KIND not in _kinds(messages)
    assert ctx._owner_directives == []  # a click is retained as an action, not invented owner wording
    assert _block(env, quiz_id)["closed_actor"] == "owner"
    assert _block(env, quiz_id)["secret_outcome"]["event"] == "secret_cancelled"


def test_owner_corpus_readers_keep_secret_frames_host_authored(env):
    from ouroboros.dialogue_evidence import _row_projection
    from ouroboros.dialogue_provenance import row_author
    from ouroboros.owner_continue import owner_corpus_rows, owner_sources
    from ouroboros.owner_relay import quotable_owner_row
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.plan_dialogue import _quiz_text
    from tests import _dialogue

    filled, cancelled = _ask(env), _ask(env)
    assert _save(env, filled, {"botToken": CANARY}).status_code == 200
    assert _cancel(env, cancelled).status_code == 200
    rows = {(row.get("type"), (row.get("quiz") or {}).get("quiz_id")): row for row in _dialogue.chat_rows(env.root)}
    fill_row, cancel_row = rows[("quiz_answer", filled)], rows[("quiz_closed", cancelled)]
    assert row_author(fill_row)["kind"] == "host" and row_author(fill_row)["via"] == "secret_request"
    assert row_author(cancel_row)["kind"] == "host" and row_author(cancel_row)["via"] == "secret_request"
    assert cancel_row["quiz"]["closed_actor"] == "owner"
    assert not quotable_owner_row(fill_row)  # never a quotable owner row
    assert quotable_owner_row({**fill_row, "quiz": {k: v for k, v in fill_row["quiz"].items() if not k.startswith("secret_")}})
    assert _quiz_text(fill_row) == f"[secret request {filled}] secret_filled: fakeplug, botToken"
    for i, entry in enumerate(_mail(env), 1):
        assert _row_projection({**entry, "task_id": TASK}, "mailbox", i)["author"] == "host"
    later = owner_sources(env.root, load_task_result(env.root, TASK), TASK)["later"]
    assert later == [] and owner_corpus_rows({"later": later}) == []
    _assert_absent(env, CANARY, texts=[json.dumps(later, ensure_ascii=False)])


@pytest.mark.parametrize("change", ["mapping", "content", "unchanged", "key_set"])
def test_mapping_and_key_state_are_checked_after_waiting_for_the_real_settings_lock(env, monkeypatch, change):
    """A blocked POST must not use the mapping or key state it saw before another writer's lock."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from ouroboros import config as cfg

    cfg.SETTINGS_PATH.write_text('{"UNRELATED": "keep"}\n', encoding="utf-8")
    quiz_id = _ask(env)
    shown = _observed(env, quiz_id)
    acquire = cfg._acquire_settings_lock
    held = acquire()
    assert held is not None
    entered = threading.Event()

    def observed_acquire(*args, **kwargs):
        entered.set()
        return acquire(*args, **kwargs)

    monkeypatch.setattr(cfg, "_acquire_settings_lock", observed_acquire)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(_save, env, quiz_id, {"botToken": CANARY}, "waited-save", shown)
        try:
            assert entered.wait(2) and not pending.done()
            if change == "mapping":
                _write_skill(env.root, env="FAKEPLUG_TOKEN, FAKEPLUG_OTHER_TOKEN",
                             fields="  botToken: FAKEPLUG_OTHER_TOKEN", version="0.2.0")
            elif change == "content":
                _write_skill(env.root, version="0.2.0")
            elif change == "key_set":  # another process writing under its own lock hold
                cfg.SETTINGS_PATH.write_text(json.dumps({"UNRELATED": "keep", "FAKEPLUG_TOKEN": CANARY_B}), encoding="utf-8")
            before = cfg.SETTINGS_PATH.read_bytes()
        finally:
            cfg._release_settings_lock(held)
        response = pending.result(timeout=5)
    body = response.json()
    if change in ("unchanged", "content"):
        assert response.status_code == 200 and body["saved"] is True
        assert _settings(env)["FAKEPLUG_TOKEN"] == CANARY
    else:
        reason = "secret_request_stale" if change == "mapping" else "secret_key_state_changed"
        assert response.status_code == 409 and body["reason_code"] == reason, body
        assert body["saved"] is False and "secret_filled" not in body
        assert cfg.SETTINGS_PATH.read_bytes() == before
        block = _block(env, quiz_id)
        assert block["state"] == "open" and not block.get("secret_save")
        assert not _mail(env) and not env.frames
    _assert_absent(env, CANARY, texts=[response.text])


def test_a_paused_asker_keeps_the_saved_outcome_owed_until_it_reads_again(env):
    """History records the save, but Pause must neither discharge nor wake: the outcome waits in
    the paused asker's mailbox, owed until it reads."""
    quiz_id = _ask(env)
    env.live["reason"] = "task_not_reading"
    response = _save(env, quiz_id, {"botToken": CANARY})
    assert response.status_code == 200 and response.json()["saved"] is True
    block = _block(env, quiz_id)
    assert block["delivery"]["history_at"] and block["delivery"]["state"] == "owed"
    assert block["delivery"]["reason_code"] == "awaiting_recipient_read" and not block.get("discharged_at")
    assert "answered_after_terminal" not in block and "delivery_after_terminal" not in block
    [mail] = [row for row in _mail(env) if row.get("kind") == "quiz_answer"]
    assert "saved it to Settings (FAKEPLUG_TOKEN)" in mail["text"]
    assert _block(env, quiz_id)["delivery"]["recipient_task_id"] == TASK
    _assert_absent(env, CANARY, texts=[response.text, mail["text"]])
