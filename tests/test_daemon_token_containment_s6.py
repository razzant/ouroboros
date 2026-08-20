"""S6 C6/O5 — the daemon-token containment claim, made falsifiable.

``ouroboros/tools/delegate.py`` states the boundary in prose: the operator's
Claudexor daemon control token must never leave ``ouroboros/gateways/
claudexor.py``, where it is read from the descriptor's ``tokenPath`` and
injected into exactly one place — the loopback client's ``Authorization``
header. Everything else about it was already pinned (loopback-only, never
returned to a caller, absent from the refusal text), but nothing asserted the
headline itself: that the token is not in the run REQUEST, not in the durable
custody rows, and not in the artifacts staged on the task drive.

This module drives ONE real delegated run through the REAL gateway — a live
token in the header, a mock transport underneath — and then greps every surface
the run produced. The first assertion is that the token is genuinely in play:
a fixture that quietly carries no token would make every "absent" assertion
below pass for the wrong reason.

Test-only. No production change (S5 O5).
"""

from __future__ import annotations

import json
import pathlib

import httpx
import pytest

from ouroboros.gateways import claudexor as cx


TOKEN = "s6-daemon-control-token-do-not-leak"
RUN_ID = "run-s6-token"
# Long enough that the terminal payload cannot ride inline: the whole detail is
# then STAGED under `delegated_runs/` on the task drive, which is the surface
# this module has to be able to grep.
BIG_ANSWER = "the child's answer. " * 20_000


def _handler(seen: list):
    """A daemon that starts one run and reports it succeeded."""

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append({
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": request.content.decode("utf-8", "replace"),
        })
        path = request.url.path
        if path == "/v2/handshake":
            return httpx.Response(200, json={
                "protocolMajor": cx.CLAUDEXOR_PROTOCOL_MAJOR, "compatible": True,
                "engine": {"version": "99.0.0"},
            })
        if path == "/v2/agent-capabilities":
            return httpx.Response(200, json={"harnesses": [{
                "id": "some-route", "enabled": True, "status": "ok",
                "accessProfilesSupported": ["readonly"],
            }]})
        if path == "/v2/quota":
            return httpx.Response(200, json={"quotas": []})
        if path == "/v2/projects":
            # GET lists (an empty registry -> the caller registers), POST registers.
            if request.method == "POST":
                return httpx.Response(200, json={"id": "prj-s6"})
            return httpx.Response(200, json={"projects": []})
        if path == "/v2/runs":
            return httpx.Response(200, json={"runId": RUN_ID})
        if path.startswith(f"/v2/runs/{RUN_ID}"):
            return httpx.Response(200, json={
                "lastSeq": 2,
                "summary": {"state": "succeeded", "effectiveAccess": "readonly"},
                "primaryOutput": BIG_ANSWER,
                "finalSummary": "the child's answer",
            })
        return httpx.Response(404, json={"error": {"code": "not_found"}})

    return handle


_REAL_GATEWAY_CLS = cx.ClaudexorGateway


def _gateway_with_token(seen: list) -> cx.ClaudexorGateway:
    """The REAL gateway: real header injection, mock transport underneath.

    Built from the class captured at import time, because the fixture below
    replaces the module attribute with this very factory.
    """
    gateway = _REAL_GATEWAY_CLS(cx.DaemonEndpoint("127.0.0.1", 1, TOKEN))
    gateway._client = httpx.Client(
        base_url="http://127.0.0.1:1",
        transport=httpx.MockTransport(_handler(seen)),
        headers=dict(gateway._client.headers),
    )
    return gateway


def _ctx(tmp_path):
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "t-token"
    ctx.task_metadata = {"root_task_id": "t-token", "parent_task_id": "t-token"}
    return ctx


def _event_rows(tmp_path):
    path = pathlib.Path(tmp_path) / "logs" / "events.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _drive_one_run(tmp_path, monkeypatch, *, leak: bool = False) -> dict:
    """Start and wait one delegated run against the real gateway.

    ``leak=True`` makes the run body carry the token, which is how this module
    proves its own assertions can fail (a containment test that cannot go red is
    decoration).
    """
    import ouroboros.tools.delegate as delegate
    from ouroboros import claudexor_daemon
    from ouroboros.gateways import claudexor as gateway_module

    seen: list = []
    # A FRESH gateway per acquisition (the verbs close the one they were handed),
    # each carrying the same token and appending to the same capture list.
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(
        claudexor_daemon, "ensure_owned_gateway", lambda: _gateway_with_token(seen))
    monkeypatch.setattr(
        gateway_module, "ClaudexorGateway", lambda *a, **k: _gateway_with_token(seen))
    if leak:
        real_request = delegate._start_request

        def _leaky(*args, **kwargs):
            request = real_request(*args, **kwargs)
            request["instructions"] = f"{request.get('instructions') or ''}\n{TOKEN}"
            return request

        monkeypatch.setattr(delegate, "_start_request", _leaky)
    delegate._CUSTODY.clear()
    ctx = _ctx(tmp_path)
    started = json.loads(delegate._delegate_start(ctx, "review the diff"))
    assert started["status"] == "started", started
    waited = json.loads(delegate._delegate_wait(ctx, RUN_ID, wait_sec=1))
    return {"seen": seen, "started": started, "waited": waited,
            "drive": pathlib.Path(tmp_path), "ctx": ctx}


@pytest.fixture
def delegated_run(tmp_path, monkeypatch):
    """One delegated run start + wait against the real gateway."""
    import ouroboros.tools.delegate as delegate

    try:
        yield _drive_one_run(tmp_path, monkeypatch)
    finally:
        delegate._CUSTODY.clear()


def test_c6_the_fixture_really_carries_the_token_on_the_wire(delegated_run):
    """The guard that keeps every assertion below honest: the token IS used, in
    the one place it belongs — the loopback client's Authorization header."""
    seen = delegated_run["seen"]
    assert seen, "the run never reached the transport"
    assert {row["headers"].get("authorization") for row in seen} == {f"Bearer {TOKEN}"}
    assert {row["url"].split("/v2")[0] for row in seen} == {"http://127.0.0.1:1"}


def test_c6_the_token_is_absent_from_the_start_request_body(delegated_run):
    """O5: the `delegate_start` POST body — including `instructions` and
    `prompt`, the two fields the child itself reads — carries no token."""
    runs = [row for row in delegated_run["seen"]
            if row["method"] == "POST" and row["url"].endswith("/v2/runs")]
    assert len(runs) == 1, [row["url"] for row in delegated_run["seen"]]
    body = json.loads(runs[0]["body"])
    assert TOKEN not in runs[0]["body"]
    assert TOKEN not in str(body.get("instructions") or "")
    assert TOKEN not in str(body.get("prompt") or "")
    # And no request of the whole run smuggles it into a body.
    assert [row["url"] for row in delegated_run["seen"] if TOKEN in row["body"]] == []


def test_c6_the_token_is_absent_from_the_durable_custody_rows(delegated_run):
    """O5: `delegate_run_start_requested` stores the canonical request body for
    replay, so a token in the body would be durable in `logs/events.jsonl`."""
    rows = _event_rows(delegated_run["drive"])
    types = [row.get("type") for row in rows]
    assert "delegate_run_start_requested" in types, types
    for row in rows:
        assert TOKEN not in json.dumps(row), row.get("type")


def test_c6_the_token_is_absent_from_every_staged_delegated_artifact(delegated_run):
    """O5: the terminal detail is staged under `delegated_runs/` on the task
    drive and read back with the ordinary read_file contract — a child, or a
    later reviewer, reads those bytes."""
    # `delegated_runs/` lives under the TASK drive
    # (`<drive_root>/task_drives/<task_id>/delegated_runs`), found here by walking
    # so the assertion cannot go vacuous if that layout moves.
    staged = [
        path for directory in delegated_run["drive"].rglob("delegated_runs")
        if directory.is_dir() for path in directory.rglob("*")
    ]
    files = sorted(path for path in staged if path.is_file())
    assert files, "the wait staged no artifact, so this assertion would be vacuous"
    for path in files:
        assert TOKEN not in path.read_bytes().decode("utf-8", "replace"), path


def test_c6_the_token_is_absent_from_the_verb_payloads_and_the_whole_drive(delegated_run):
    """O5, the backstop: nothing the tools RETURN to the agent carries it, and
    no file the run wrote anywhere under the task drive does either."""
    assert TOKEN not in json.dumps(delegated_run["started"])
    assert TOKEN not in json.dumps(delegated_run["waited"])
    offenders = [
        str(path) for path in delegated_run["drive"].rglob("*")
        if path.is_file() and TOKEN in path.read_bytes().decode("utf-8", "replace")
    ]
    assert offenders == [], offenders


def test_c6_the_assertions_above_do_catch_a_leak(tmp_path, monkeypatch):
    """The negative control: with the token deliberately appended to the run's
    `instructions`, the same three greps find it — on the wire, in the durable
    replay row, and in the staged artifact. Without this, "absent everywhere"
    could just mean "looked in the wrong places"."""
    import ouroboros.tools.delegate as delegate

    try:
        run = _drive_one_run(tmp_path, monkeypatch, leak=True)
    finally:
        delegate._CUSTODY.clear()

    posted = [row for row in run["seen"] if row["url"].endswith("/v2/runs")]
    assert posted and TOKEN in posted[0]["body"], "the injection did not reach the wire"
    rows = _event_rows(run["drive"])
    assert [row.get("type") for row in rows if TOKEN in json.dumps(row)] == [
        "delegate_run_start_requested",
    ], "the durable replay row is where a leaked body becomes permanent"
    staged = [
        path for directory in run["drive"].rglob("delegated_runs")
        if directory.is_dir() for path in directory.rglob("*") if path.is_file()
    ]
    assert staged, "the artifact grep needs a staged file to be meaningful"
