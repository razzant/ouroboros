"""The isolated review's paid OpenRouter key probe is an attempt of the run's own ledger.

``_prepare_review_configuration`` reaches the wrapper's one-token completion probe
before any review gate. Each probe runs here in a real child process through the
real isolation, pinned settings, panel resolution and key pool, against a
synthetic key endpoint, pricing catalog and provider transport (no provider is
contacted): every send is admitted against the review drive's cumulative run cap,
settled or held there before any later preflight step, keeps its exact key and
payload, and is never retried; an exhausted drive sends nothing.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from ouroboros.openrouter_attribution import OPENROUTER_APP_HEADERS
from ouroboros.review_run_isolation import ATTACH_HOME_ENV, REVIEW_RUN_CAP_ENV
from ouroboros.settings_integrity import SETTINGS_INTEGRITY_ENV

pytestmark = pytest.mark.serial  # real child processes

REPO = pathlib.Path(__file__).resolve().parents[1]
_ONE, _TWO = "openai/probe-one", "openai/probe-two"
_URL = "https://openrouter.ai/api/v1/chat/completions"
_DRIVER = '''import json, pathlib, sys
from types import SimpleNamespace
import httpx, requests
import scripts.run_external_review as wrapper

drive, outcomes, attach = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3] == "attach"
sends = []

def post(url, *, headers, json, timeout):
    sends.append({"url": url, "headers": headers, "payload": json, "timeout": timeout})
    outcome, request = outcomes[len(sends) - 1], httpx.Request("POST", url)
    if "raise" in outcome:
        raise getattr(httpx, outcome["raise"])("synthetic", request=request)
    return httpx.Response(outcome["status"], request=request, **{k: v for k, v in outcome.items() if k != "status"})

class Catalog:  # the free pricing catalog read: priced reservations, never a paid send
    def raise_for_status(self):
        return None
    def json(self):
        price = {"prompt": "0.000001", "completion": "0.000002"}
        return {"data": [{"id": model, "pricing": price} for model in ("openai/probe-one", "openai/probe-two")]}

httpx.get = lambda url, **_kw: httpx.Response(200, json={"data": {"limit": None}}, request=httpx.Request("GET", url))
httpx.post, requests.get = post, lambda *_a, **_kw: Catalog()
wrapper._contributor_proposal = lambda *_args: {"base_sha": "b" * 40}
args = SimpleNamespace(contributor=True, base_ref="base", head_ref="head", drive_root=drive,
                       run_cap_usd="1", attach_host_engine=attach)
try:
    wrapper._prepare_review_configuration(args)
    result = "prepared"
except Exception as exc:
    result = f"{type(exc).__name__}: {exc}"
from ouroboros import usage_store
rows = [{key: row.get(key) for key in ("model", "state", "cost_usd", "cost_final", "task_id", "category")}
        for row in usage_store.read_usage_records(pathlib.Path(drive)) if row.get("kind") == "attempt"]
print(json.dumps({"result": result, "sends": sends, "rows": rows}))
'''


def _host(root: pathlib.Path, document: dict, *, pool: tuple[str, ...] = ()):
    """A host whose settings are ``document`` and whose keys file names ``pool``; returns its runner."""
    from ouroboros.settings_defaults import RETIRED_COMMA_LIST_SETTING_KEYS, settings_env_keys

    host = root / "host-data"
    host.mkdir(parents=True)
    (host / "settings.json").write_text(json.dumps(document), encoding="utf-8")
    keys = root / "keys.txt"
    keys.write_text("".join(f"openrouter_{n}: {value}\n" for n, value in enumerate(pool, 2)), encoding="utf-8")
    dropped = {*settings_env_keys(), *RETIRED_COMMA_LIST_SETTING_KEYS, SETTINGS_INTEGRITY_ENV,
               REVIEW_RUN_CAP_ENV, ATTACH_HOME_ENV}
    env = {key: value for key, value in os.environ.items() if key not in dropped}
    env.update(OUROBOROS_DATA_DIR=str(host), OUROBOROS_SETTINGS_PATH=str(host / "settings.json"),
               OUROBOROS_KEYS_FILE=str(keys))

    def run(drive: pathlib.Path, outcomes: list[dict], *, attach: bool = False):
        child = subprocess.run([sys.executable, "-c", _DRIVER, str(drive), json.dumps(outcomes),
                                "attach" if attach else "-"],
                               cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300)
        assert child.returncode == 0, child.stderr[-4000:]
        return json.loads(child.stdout.strip().splitlines()[-1]), child.stderr

    return run


def _ok(cost: float) -> dict:
    return {"status": 200, "json": {"choices": [{"message": {"role": "assistant", "content": "p"}}],
                                    "usage": {"prompt_tokens": 8, "completion_tokens": 1, "cost": cost}}}


def _pool(*routes: dict) -> str:
    """A review pool (``OUROBOROS_SUBAGENTS``) of these routes, one row each."""
    return json.dumps({"enabled": True, "items": [
        {"subagent_id": f"r{index}", "name": f"r{index}", "recommended_use": "Panel row.",
         "review_eligible": True, "route": route} for index, route in enumerate(routes, 1)]})


_OPENROUTER_PANEL = {
    "OPENROUTER_API_KEY": "probe-key-1",
    "OUROBOROS_SUBAGENTS": _pool({"kind": "api_model", "target_id": f"openrouter::{_ONE}"},
                                 {"kind": "api_model", "target_id": f"openrouter::{_TWO}"}),
}


def test_isolated_key_probes_are_attempts_of_the_runs_own_ledger(tmp_path):
    run = _host(tmp_path, _OPENROUTER_PANEL, pool=("probe-key-2", "probe-key-3", "probe-key-4", "probe-key-5"))
    drive = tmp_path / "drive"

    # Five keys, each probed once (no retry), the healthy one on every reviewer model.
    first, stderr = run(drive, [
        {"status": 403, "json": {"error": {"code": 403, "message": "synthetic refusal"}}},
        {"status": 200, "text": "<html>gateway</html>"},
        {"raise": "ReadTimeout"},
        {"raise": "ConnectError"},
        _ok(0.25),
        {"status": 200, "json": {"error": {"code": 502, "message": "upstream"}}},
    ])
    assert first["result"].startswith("RuntimeError: no healthy OpenRouter key")
    sent = [(send["headers"]["Authorization"], send["payload"]["model"]) for send in first["sends"]]
    assert sent == [*((f"Bearer probe-key-{n}", _ONE) for n in range(1, 6)), ("Bearer probe-key-5", _TWO)]
    for send in first["sends"]:  # the exact one-token payload on the exact route
        model = send["payload"]["model"]
        assert send["payload"] == {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]}
        assert (send["url"], send["timeout"]) == (_URL, 60)
        assert {key: send["headers"][key] for key in OPENROUTER_APP_HEADERS} == OPENROUTER_APP_HEADERS
    for detail in ("model_probe_http_403", "model_probe_unreadable", "model_probe_error:ReadTimeout",
                   "model_probe_error:ConnectError", f"model_ok({_ONE});model_probe_body_502"):
        assert detail in stderr
    # Each outcome is honest in the drive's ledger, and the paid answer stays recorded
    # although the preflight failed after it: refusal and timeout keep their money
    # unknown, the unreadable answer is settled without a price, the unsent connect is
    # released, the provider's 200-body error is its documented free settlement.
    rows = first["rows"]
    assert [(row["model"], row["state"]) for row in rows] == [
        (_ONE, "unresolved"), (_ONE, "settled"), (_ONE, "unresolved"), (_ONE, "released"),
        (_ONE, "settled"), (_TWO, "settled")]
    assert [(row["cost_usd"], row["cost_final"]) for row in rows if row["state"] == "settled"] == [
        (None, False), (0.25, True), (0.0, True)]
    assert {(row["task_id"], row["category"]) for row in rows} == {("system:review_key_probe", "provider_test")}
    # Candidates and ledger rows are recorded on the drive; no key under test is.
    assert not [path for path in drive.rglob("*") if path.is_file() and b"probe-key-" in path.read_bytes()]

    # A continuation on the same drive counts that spend: the next paid answer fills
    # the cap, so the following model is refused before any send, as the run's fact.
    second, _ = run(drive, [_ok(0.8)])
    assert second["result"].startswith("BudgetExceeded: global model budget exhausted")
    assert [send["payload"]["model"] for send in second["sends"]] == [_ONE]
    assert len(second["rows"]) == len(rows) + 1
    # An exhausted drive sends nothing at all.
    third, _ = run(drive, [])
    assert third["result"].startswith("BudgetExceeded: global model budget exhausted")
    assert third["sends"] == [] and third["rows"] == second["rows"]


@pytest.mark.parametrize("panel", ["local", "session"])
def test_panels_without_openrouter_rows_send_no_probe(tmp_path, panel):
    # A local-only install (an OpenRouter credential would give it the remote default
    # panel), and a session-only panel beside a saved OpenRouter key.
    from ouroboros.subscription_install_presets import factory_review_rows

    session = {"kind": "agent_session", "target_id": "codex=gpt-host"}
    local = {"USE_LOCAL_MAIN": True, "LOCAL_MODEL_SOURCE": "owner/local-model.gguf", "OUROBOROS_MODEL": "owner-local"}
    documents = {
        # The pool the one-time migration mints for a local-only install: Main on the local lane.
        "local": {**local, "OUROBOROS_SUBAGENTS": json.dumps({"enabled": False, "items": factory_review_rows(local)})},
        "session": {"OPENROUTER_API_KEY": "probe-key-1", "OUROBOROS_SUBAGENTS": _pool(session, session)},
    }
    run = _host(tmp_path, documents[panel])

    report, _ = run(tmp_path / "drive", [], attach=panel == "session")

    assert (report["result"], report["sends"], report["rows"]) == ("prepared", [], [])


def test_operator_lane_probe_keeps_the_live_ledger_untouched(tmp_path, monkeypatch):
    """No run cap: no review drive exists yet, and the live data root is never written."""
    import httpx

    import scripts.run_external_review as wrapper

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv(REVIEW_RUN_CAP_ENV, raising=False)
    sends = []
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: sends.append(kwargs["json"]) or httpx.Response(
        200, json=_ok(0.25)["json"], request=httpx.Request("POST", url)))

    assert wrapper._probe_model_for_key("probe-key-1", _ONE) == (True, f"model_ok({_ONE})")
    assert len(sends) == 1 and list(tmp_path.iterdir()) == []
