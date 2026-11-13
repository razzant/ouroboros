"""A review pool none of whose rows reads the repository (every row Packet).

No seat is asked the coupling part, so every commit-gate review of the body ends NOT_PERFORMED
(``coupling_not_performed``), in every Access level. Save warns — never refuses — in the Settings save
``warnings`` and in the wizard's summary, and ``## Review`` carries ``coupling_unanswerable: true``, a block
fact the shrink above four seats keeps. Each rule is pinned in both directions.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ouroboros.configured_subagents import SUBAGENTS_SETTING
from ouroboros.subagent_runtime import review_facts_block
from tests._cases import chosen_rows

REPO = Path(__file__).resolve().parents[1]
# Spelled out so a missing warning fails on behaviour, not on a missing name.
WARNING = (
    "Every reviewer is a Packet row, so none reads the repository and the coupling question (how the "
    "change fits the rest of the code) goes unanswered: every review of a commit to Ouroboros itself "
    "ends without a verdict. Mark a reviewer that reads the work itself.")
JUDGE = """
import fs from 'node:fs';
import * as editor from './web/modules/subagents_settings.js';
const catalogs = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify({
    warning: editor.PACKET_ONLY_POOL_WARNING ?? null,
    packetOnly: catalogs.map((catalog) => (editor.packetOnlyReviewPool ? editor.packetOnlyReviewPool(catalog) : null)),
}));
"""


def _api(row_id: str, target: str, *, packet: bool = False, marked: bool = True, **extra) -> dict:
    row = {"subagent_id": row_id, "recommended_use": f"use {row_id}", "effort": "high",
           "route": {"kind": "api_model", "target_id": target}, **extra}
    if packet:
        row["delivery"] = "packet"
    if marked:
        row["review_eligible"] = True
    return row


def _session(row_id: str, target: str = "codex=gpt-6-astra") -> dict:
    return {"subagent_id": row_id, "recommended_use": f"use {row_id}", "effort": "high",
            "route": {"kind": "agent_session", "target_id": target}, "review_eligible": True}


def _catalog(*rows: dict) -> dict:
    return {"enabled": True, "items": list(rows)}


PACKET_A = _api("packet-a", "openai/gpt-5.6-sol", packet=True)
PACKET_B = _api("packet-b", "anthropic/claude-fable-5", packet=True)
PACKET_ONLY = {
    "two Packet reviewers": _catalog(PACKET_A, PACKET_B),
    "an unmarked reading helper beside them": _catalog(PACKET_A, _api("helper", "x-ai/grok-4.6", marked=False)),
    "a reading reviewer switched off": _catalog(PACKET_A, _api("off", "x-ai/grok-4.6", enabled=False)),
}
READING = {
    "a reading API reviewer": _catalog(PACKET_A, _api("reader", "x-ai/grok-4.6")),
    "a session reviewer": _catalog(PACKET_A, _session("session")),
}


@pytest.fixture(autouse=True)
def _clean_review_plane(monkeypatch):
    for key in ("OUROBOROS_REVIEWER_SLOTS", SUBAGENTS_SETTING):
        monkeypatch.delenv(key, raising=False)


def _post(monkeypatch, body: dict, stored: dict | None = None, mode: str = "advanced"):
    import asyncio

    from starlette.requests import Request

    import ouroboros.gateway.settings as gws

    saved = dict(stored or {})

    def _load():
        from ouroboros.config import SETTINGS_DEFAULTS
        return {**SETTINGS_DEFAULTS, **saved}

    def _write(payload, *, allow_elevation=False, allow_context_lowering=False, authored_keys=(), boundary=None,
               expected_digest="", if_unchanged=None, unset=()):
        saved.clear()
        saved.update(payload)
        if boundary is not None:
            boundary.commit()
        return payload

    monkeypatch.setattr(gws, "load_settings", _load)
    monkeypatch.setattr(gws, "_owner_write_settings", _write)
    monkeypatch.setattr(gws, "_owner_settings_document", lambda: dict(saved))  # the document the fake write keeps
    monkeypatch.setattr(gws, "_unrecognised_review_models", lambda models: [])
    monkeypatch.setattr(gws, "_apply_settings_to_env", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: mode)

    async def _receive():
        return {"type": "http.request", "body": json.dumps(body).encode()}

    request = Request({"type": "http", "method": "POST", "path": "/api/settings",
                       "headers": [("content-type", "application/json")], "query_string": b"", "app": None},
                      receive=_receive)
    response = asyncio.run(gws.api_settings_post(request))
    return response.status_code, json.loads(response.body), saved


@pytest.mark.parametrize("saved", [{}, {"OUROBOROS_REVIEW_ENFORCEMENT": "advisory"}],
                         ids=["nothing saved", "a leftover saved Advisory"])
def test_the_save_warning_judges_the_pool_the_real_loader_makes_effective(tmp_path, monkeypatch, saved):
    """A save that never names the pool warns from the pool the real loader makes effective; a leftover value of the
    retired enforcement setting, saved or in the environment, changes nothing. Real document, real ``load_settings``
    and real write."""
    import asyncio

    from starlette.requests import Request

    import ouroboros.config as cfg
    import ouroboros.gateway.settings as gws

    path = tmp_path / "settings.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    for attribute, value in (("SETTINGS_PATH", path), ("DATA_DIR", tmp_path), ("_BOOT_RUNTIME_MODE", None)):
        monkeypatch.setattr(cfg, attribute, value)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")  # retired: read by nothing
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(gws, "_unrecognised_review_models", lambda models: [])
    monkeypatch.setattr(gws, "_apply_settings_to_env", lambda *a, **k: None)
    body = {SUBAGENTS_SETTING: PACKET_ONLY["two Packet reviewers"]}

    async def _receive():
        return {"type": "http.request", "body": json.dumps(body).encode()}

    request = Request({"type": "http", "method": "POST", "path": "/api/settings", "headers": [("content-type", "application/json")],
                       "query_string": b"", "app": None}, receive=_receive)
    response = asyncio.run(gws.api_settings_post(request))
    answer = json.loads(response.body)
    assert response.status_code == 200 and answer["status"] == "saved", answer
    assert WARNING in answer.get("warnings", [])


def test_a_save_of_a_pool_with_no_reading_row_warns_and_still_saves(subtests, monkeypatch):
    for catalog in list(PACKET_ONLY.values()):
        with subtests.test(catalog=catalog), monkeypatch.context() as monkeypatch:
            status, body, saved = _post(monkeypatch, {SUBAGENTS_SETTING: catalog})

            assert status == 200 and body["status"] == "saved", body
            assert json.loads(saved[SUBAGENTS_SETTING])["items"] == catalog["items"], "a warning never refuses the save"
            assert WARNING in body.get("warnings", [])


@pytest.mark.parametrize("catalog, mode, warned", chosen_rows([
    *[(catalog, "advanced", False) for catalog in READING.values()],
    (PACKET_ONLY["two Packet reviewers"], "cyber_pro", True),
], 0, 2), ids=chosen_rows([*READING, "Cyber Pro"], 0, 2))
def test_the_warning_follows_the_pool_alone_in_every_access_level(monkeypatch, catalog, mode, warned):
    stored = {"OUROBOROS_RUNTIME_MODE": mode}
    status, body, _saved = _post(monkeypatch, {SUBAGENTS_SETTING: catalog}, stored, mode)

    assert status == 200, body
    assert (WARNING in body.get("warnings", [])) is warned


def test_every_save_over_a_stored_packet_pool_says_so_until_a_row_reads(monkeypatch):
    stored = {SUBAGENTS_SETTING: json.dumps(PACKET_ONLY["two Packet reviewers"])}
    # While the pool still cannot answer, an unrelated save repeats the fact rather than reading as fine.
    status, body, saved = _post(monkeypatch, {"TOTAL_BUDGET": 12.5}, stored)
    assert status == 200 and WARNING in body.get("warnings", [])

    status, body, _saved = _post(monkeypatch, {SUBAGENTS_SETTING: READING["a reading API reviewer"]}, dict(saved))
    assert status == 200 and WARNING not in body.get("warnings", [])


def _review_block(monkeypatch, catalog: dict) -> dict:
    import ouroboros.subagent_runtime as runtime

    monkeypatch.setattr(runtime, "_api_review_cost_hint", lambda slot, *_price: "cost unknown")
    monkeypatch.setenv(SUBAGENTS_SETTING, json.dumps(catalog))
    text = review_facts_block()
    return json.loads(text.split("\n\n", 1)[1])


def test_review_block_states_an_unanswerable_coupling_part_and_keeps_it_above_four_seats(monkeypatch):
    block = _review_block(monkeypatch, PACKET_ONLY["two Packet reviewers"])
    assert block["coupling_unanswerable"] is True and block["omitted"] == {"rows": 0}

    five = _catalog(*[_api(f"packet-{index}", f"openai/model-{index}", packet=True) for index in range(5)])
    block = _review_block(monkeypatch, five)
    assert block["omitted"] == {"rows": 0}
    assert [row["seat_id"] for row in block["pool"]] == [f"packet-{index}" for index in range(5)]
    assert all({"seat_id", "model", "effort", "delivery"} <= row.keys() and row["delivery"] == "packet"
               for row in block["pool"])
    activity: list = []
    review_facts_block(activity_out=activity)
    seats = json.loads(activity[0].split("\n\n", 1)[1])["seats"]
    assert [seat["seat_id"] for seat in seats] == [row["seat_id"] for row in block["pool"]]
    assert all({"cost_hint", "cost_usd", "cost_kind", "last_review"} <= seat.keys() for seat in seats)
    assert block["coupling_unanswerable"] is True, "larger pools must keep the coupling fact and full rows"


@pytest.mark.parametrize("catalog", [*READING.values(), _catalog(_api("helper", "x-ai/grok-4.6", marked=False))],
                         ids=[*READING, "an empty pool"])
def test_review_block_carries_no_coupling_fact_when_a_seat_reads_or_the_pool_is_empty(monkeypatch, catalog):
    block = _review_block(monkeypatch, catalog)
    assert "coupling_unanswerable" not in block
    assert block["pool_empty"] is (not any(row.get("review_eligible") for row in catalog["items"]))


def test_the_wizard_judges_the_pool_and_words_the_warning_as_the_server_does(monkeypatch):
    catalogs = [*PACKET_ONLY.values(), *READING.values(), _catalog()]
    result = subprocess.run(["node", "--input-type=module", "-e", JUDGE], cwd=REPO, capture_output=True,
                            input=json.dumps(catalogs), text=True, encoding="utf-8", check=True)
    shown = json.loads(result.stdout)

    assert shown["warning"] == WARNING
    server = [_review_block(monkeypatch, catalog).get("coupling_unanswerable", False) for catalog in catalogs]
    assert shown["packetOnly"] == server == [True] * len(PACKET_ONLY) + [False] * (len(READING) + 1)
