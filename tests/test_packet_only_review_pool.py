"""A review pool none of whose rows reads the repository (every row Packet) under Blocking.

No seat is asked the coupling part, so every commit-gate review of the body ends NOT_PERFORMED
(``coupling_not_performed``). Save warns — never refuses — in the Settings save ``warnings`` and in the
wizard's summary, and ``## Review`` carries ``coupling_unanswerable: true``, a block fact the shrink
above four seats keeps. Each rule is pinned in both directions.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ouroboros.configured_subagents import SUBAGENTS_SETTING
from ouroboros.subagent_runtime import review_facts_block

REPO = Path(__file__).resolve().parents[1]
# Spelled out so a missing warning fails on behaviour, not on a missing name.
WARNING = (
    "Every reviewer is a Packet row, so none reads the repository and the coupling question (how the "
    "change fits the rest of the code) goes unanswered: with Blocking review, every commit to Ouroboros "
    "itself stops as “not performed”. Mark a reviewer that reads the work itself, or choose Advisory.")
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
    for key in ("OUROBOROS_REVIEW_ENFORCEMENT", "OUROBOROS_REVIEWER_SLOTS", SUBAGENTS_SETTING):
        monkeypatch.delenv(key, raising=False)


def _post(monkeypatch, body: dict, stored: dict | None = None, mode: str = "advanced"):
    import asyncio

    from starlette.requests import Request

    import ouroboros.gateway.settings as gws

    saved = dict(stored or {})

    def _load():
        from ouroboros.config import SETTINGS_DEFAULTS
        return {**SETTINGS_DEFAULTS, **saved}

    def _write(payload, *, allow_elevation=False, allow_context_lowering=False, authored_keys=(), boundary=None):
        saved.clear()
        saved.update(payload)
        if boundary is not None:
            boundary.commit()
        return payload

    monkeypatch.setattr(gws, "load_settings", _load)
    monkeypatch.setattr(gws, "_owner_write_settings", _write)
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


@pytest.mark.parametrize("catalog", list(PACKET_ONLY.values()), ids=list(PACKET_ONLY))
def test_a_blocking_save_of_a_pool_with_no_reading_row_warns_and_still_saves(monkeypatch, catalog):
    status, body, saved = _post(monkeypatch, {SUBAGENTS_SETTING: catalog, "OUROBOROS_REVIEW_ENFORCEMENT": "blocking"})

    assert status == 200 and body["status"] == "saved", body
    assert json.loads(saved[SUBAGENTS_SETTING])["items"] == catalog["items"], "a warning never refuses the save"
    assert saved["OUROBOROS_REVIEW_ENFORCEMENT"] == "blocking"
    assert WARNING in body.get("warnings", [])


@pytest.mark.parametrize("catalog, enforcement, mode", [
    *[(catalog, "blocking", "advanced") for catalog in READING.values()],
    (PACKET_ONLY["two Packet reviewers"], "advisory", "advanced"),
    (PACKET_ONLY["two Packet reviewers"], "blocking", "cyber_pro"),
], ids=[*READING, "advisory review", "Cyber Pro"])
def test_no_warning_with_a_reading_row_under_advisory_or_in_cyber_pro(monkeypatch, catalog, enforcement, mode):
    stored = {"OUROBOROS_RUNTIME_MODE": mode}
    status, body, _saved = _post(
        monkeypatch, {SUBAGENTS_SETTING: catalog, "OUROBOROS_REVIEW_ENFORCEMENT": enforcement}, stored, mode)

    assert status == 200, body
    assert WARNING not in body.get("warnings", [])


def test_switching_to_blocking_over_a_stored_packet_pool_warns_and_every_later_save_says_so(monkeypatch):
    stored = {SUBAGENTS_SETTING: json.dumps(PACKET_ONLY["two Packet reviewers"]),
              "OUROBOROS_REVIEW_ENFORCEMENT": "advisory"}
    status, body, saved = _post(monkeypatch, {"OUROBOROS_REVIEW_ENFORCEMENT": "blocking"}, stored)
    assert status == 200 and WARNING in body.get("warnings", [])

    # While the pool still cannot answer, an unrelated save repeats the fact rather than reading as fine.
    status, body, _saved = _post(monkeypatch, {"TOTAL_BUDGET": 12.5}, dict(saved))
    assert status == 200 and WARNING in body.get("warnings", [])


def _review_block(monkeypatch, catalog: dict) -> dict:
    import ouroboros.subagent_runtime as runtime

    monkeypatch.setattr(runtime, "_api_review_cost_hint", lambda slot: "cost unknown")
    monkeypatch.setenv(SUBAGENTS_SETTING, json.dumps(catalog))
    text = review_facts_block()
    return json.loads(text.split("\n\n", 1)[1])


def test_review_block_states_an_unanswerable_coupling_part_and_keeps_it_above_four_seats(monkeypatch):
    block = _review_block(monkeypatch, PACKET_ONLY["two Packet reviewers"])
    assert block["coupling_unanswerable"] is True and block["omitted"] == {"rows": 0}

    five = _catalog(*[_api(f"packet-{index}", f"openai/model-{index}", packet=True) for index in range(5)])
    block = _review_block(monkeypatch, five)
    assert block["omitted"] == {"rows": 5}
    assert all(set(row) == {"seat_id", "model"} for row in block["pool"]), "the rows shrank"
    assert block["coupling_unanswerable"] is True, "the shrink must not lose the fact"


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
