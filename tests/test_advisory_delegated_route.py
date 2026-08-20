"""Phase 5.8: the advisory review's delegated route and the four key sites.

The ANTHROPIC_API_KEY checks are ROUTE-DEPENDENT: an api route requires the key
byte-for-byte as before, and the delegated route runs without it — most
importantly, the constitutional pre-commit gate RUNS on the delegated route
instead of recording a routine-looking "auto-bypassed".

Offline fixtures throughout (owner test rule): the FakeGateway from the
agent-session route tests stands in for the Claudexor control plane.
"""

import json
import subprocess

import pytest

import ouroboros.tools.claude_advisory_review as advisory
from tests._review_session_route_shared import FakeGateway, _terminal_detail


@pytest.fixture(autouse=True)
def _owned_gateway_uses_each_test_transport(monkeypatch):
    from ouroboros import claudexor_daemon
    from ouroboros.gateways import claudexor as gateway_module

    monkeypatch.setattr(
        claudexor_daemon,
        "ensure_owned_gateway",
        lambda: gateway_module.ClaudexorGateway(),
    )


@pytest.fixture()
def fake_route(monkeypatch):
    from ouroboros import delegate_custody as custody

    FakeGateway.reset()
    monkeypatch.setattr("ouroboros.gateways.claudexor.ClaudexorGateway", FakeGateway)
    monkeypatch.setenv("OUROBOROS_REVIEW_SESSION_ROUTE", "fake-review=fake-small:low")
    custody._CUSTODY.clear()
    return FakeGateway


def _ctx(tmp_path):
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir(exist_ok=True)
    drive.mkdir(exist_ok=True)
    return ToolContext(repo_dir=repo, drive_root=drive)


_ADVISORY_ITEMS = json.dumps([
    {"item": "correctness", "verdict": "PASS", "severity": "advisory",
     "reason": "checked the change end to end"},
])


# ---------------------------------------------------------------------------
# Site 1 — the run_readonly entry check
# ---------------------------------------------------------------------------


def test_api_route_without_key_errors_exactly_as_before(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv(advisory.ADVISORY_REVIEW_ROUTE_ENV, raising=False)
    ctx = _ctx(tmp_path)
    items, raw, model, chars = advisory._run_claude_advisory(
        ctx.repo_dir, "msg", ctx, options={"include_repo_diff": False},
    )
    assert items == [] and model == "" and chars == 0
    assert raw.startswith("⚠️ ADVISORY_ERROR: ANTHROPIC_API_KEY not set")


def test_delegated_route_runs_without_the_key(tmp_path, monkeypatch, fake_route):
    """The whole free-route walk: no key anywhere, route=agent_session, and the
    advisory runs as a delegated session whose checklist comes back parsed."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv(advisory.ADVISORY_REVIEW_ROUTE_ENV, "agent_session")
    # No catalog `json_schema_output` is set here: `CatalogHarness` carries NO
    # transport flags (agent-capabilities.ts), the reader takes the flag off the
    # /v2/harnesses manifest row, and the invented catalog key was a dead no-op that
    # modelled a response Claudexor cannot emit.
    fake_route.detail = _terminal_detail(_ADVISORY_ITEMS)
    ctx = _ctx(tmp_path)
    items, raw, model, chars = advisory._run_claude_advisory(
        ctx.repo_dir, "msg", ctx, options={"include_repo_diff": False},
    )
    assert not raw.startswith("⚠️ ADVISORY_ERROR")
    assert [i["item"] for i in items] == ["correctness"]
    assert model  # the effective session model/route is reported
    start = fake_route.instances[0].start_requests[0]
    assert start["authPreference"] == "subscription"
    assert start["access"] == "readonly"


def test_unknown_route_token_is_a_loud_error_not_a_transport_pick(tmp_path, monkeypatch):
    monkeypatch.setenv(advisory.ADVISORY_REVIEW_ROUTE_ENV, "codex")
    ctx = _ctx(tmp_path)
    items, raw, _model, _chars = advisory._run_claude_advisory(
        ctx.repo_dir, "msg", ctx, options={"include_repo_diff": False},
    )
    assert items == []
    assert raw.startswith("⚠️ ADVISORY_ERROR") and "codex" in raw


# ---------------------------------------------------------------------------
# Site 3 — the constitutional pre-commit gate's auto-bypass
# ---------------------------------------------------------------------------


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    for cmd in (["git", "init", "-q"],
                ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"]):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True, capture_output=True)
    return repo


def test_missing_key_auto_bypasses_only_on_the_api_route(tmp_path, monkeypatch):
    """THE dangerous site: on the api route a missing key still auto-bypasses
    with the audited record, byte-compatible with today; on the delegated route
    the gate RUNS — the advisory is actually invoked and no bypass is recorded."""
    from ouroboros.tools.registry import ToolContext

    repo = _git_repo(tmp_path)
    (repo / "README.md").write_text("hello\nchanged\n", encoding="utf-8")  # a real diff to review
    drive = tmp_path / "data"
    drive.mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # API route: auto-bypass, exactly as today (with the route named).
    monkeypatch.delenv(advisory.ADVISORY_REVIEW_ROUTE_ENV, raising=False)
    payload = json.loads(advisory._handle_advisory_pre_review(
        ctx, commit_message="m", skip_tests=True,
    ))
    assert payload["status"] == "bypassed"
    assert "auto-bypassed" in payload["bypass_reason"]
    assert "agent_session" in payload["message"]  # the keyless path is named

    # Delegated route: the gate RUNS instead of bypassing. The downstream
    # deterministic pre-SDK gate (P9 metadata preflight, test preflight) and
    # the transport are not under test here and are stubbed; the site under
    # test — the auto-bypass — sits UPSTREAM of both.
    called = {}

    def _capture(repo_dir, commit_message, ctx_, goal="", scope="", paths=None, options=None):
        called["ran"] = True
        return [], "⚠️ ADVISORY_ERROR: sentinel — transport not under test here", "", 0

    monkeypatch.setattr(advisory, "_run_claude_advisory", _capture)
    monkeypatch.setattr(advisory, "_advisory_pre_sdk_gate",
                        lambda **_kwargs: ([], "README.md", None))
    monkeypatch.setenv(advisory.ADVISORY_REVIEW_ROUTE_ENV, "agent_session")
    payload = json.loads(advisory._handle_advisory_pre_review(
        ctx, commit_message="m", skip_tests=True,
    ))
    assert called.get("ran") is True
    assert payload["status"] != "bypassed"
    # No auto-bypass row was recorded for the delegated attempt.
    events = (drive / "logs" / "events.jsonl")
    if events.exists():
        rows = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
        assert not any(
            r.get("type") == "advisory_review_bypassed"
            and "auto-bypassed" in str(r.get("bypass_reason"))
            and "route=api" not in str(r.get("bypass_reason"))
            for r in rows
        )
        api_bypasses = [r for r in rows if r.get("type") == "advisory_review_bypassed"]
        assert len(api_bypasses) == 1  # only the api-route attempt above


def test_explicit_skip_still_bypasses_on_the_delegated_route(tmp_path, monkeypatch):
    """The owner's explicit audited bypass is route-independent and untouched."""
    from ouroboros.tools.registry import ToolContext

    repo = _git_repo(tmp_path)
    drive = tmp_path / "data"
    drive.mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    monkeypatch.setenv(advisory.ADVISORY_REVIEW_ROUTE_ENV, "agent_session")
    payload = json.loads(advisory._handle_advisory_pre_review(
        ctx, commit_message="m", skip_advisory_review=True, skip_tests=True,
    ))
    assert payload["status"] == "bypassed"
    assert payload["bypass_reason"] == "explicit skip_advisory_review=True"


def test_commit_gate_bypass_detection_rides_the_route_slot_aware_predicate():
    """The commit gate's twin of the four key sites (#123): the stage cycle's
    bypass decision must go through the named route/slot-aware predicate
    (``advisory_gate_unavailable``), never a bare ANTHROPIC_API_KEY env probe
    that a new route or the advisory enable switch would silently defeat."""
    import inspect

    from ouroboros.tools import git as git_mod

    source = inspect.getsource(git_mod._run_reviewed_stage_cycle)
    assert "advisory_gate_unavailable" in source
    assert 'os.environ.get("ANTHROPIC_API_KEY"' not in source


def _clear_session_route_envs(monkeypatch):
    monkeypatch.delenv("OUROBOROS_REVIEW_SESSION_ROUTE", raising=False)
    monkeypatch.delenv("OUROBOROS_SUBAGENT_HARNESS", raising=False)


def test_disabled_slot_has_a_stable_reason_and_boolean_projection(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-sentinel")
    monkeypatch.delenv(advisory.ADVISORY_REVIEW_ROUTE_ENV, raising=False)
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/x"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/y"}}],
        "advisory": {"enabled": False, "route": {"kind": "api", "target_id": "sonnet"}},
    }))
    assert advisory.advisory_gate_unavailability_reason() == "advisory_slot_disabled"
    assert advisory.advisory_gate_unavailable() is True


def test_keyless_api_slot_has_a_stable_reason_and_boolean_projection(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    monkeypatch.delenv(advisory.ADVISORY_REVIEW_ROUTE_ENV, raising=False)
    assert advisory.advisory_gate_unavailability_reason() == "anthropic_api_key_missing"
    assert advisory.advisory_gate_unavailable() is True


def test_unroutable_session_slot_reports_the_gate_unavailable(monkeypatch):
    """Triad a4 follow-up to #123: kind=agent_session with NO resolvable route
    anywhere (no row target, no shared review/subagent route) structurally
    cannot run — ``run_delegated_review_session`` refuses that exact state with
    ``ReviewRouteUnavailable`` — so the gate must report UNAVAILABLE. The key
    is present to prove the decision is route-driven, not key-driven."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-sentinel")
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    monkeypatch.setenv(advisory.ADVISORY_REVIEW_ROUTE_ENV, "agent_session")
    _clear_session_route_envs(monkeypatch)
    assert advisory.advisory_gate_unavailability_reason() == "agent_session_route_unavailable"
    assert advisory.advisory_gate_unavailable() is True


def test_session_slot_with_shared_route_reports_the_gate_available(monkeypatch):
    """The shared subagent route is the delegated advisory's documented
    fallback target — with it set, the keyless session slot is AVAILABLE."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    monkeypatch.setenv(advisory.ADVISORY_REVIEW_ROUTE_ENV, "agent_session")
    _clear_session_route_envs(monkeypatch)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "claude")
    assert advisory.advisory_gate_unavailability_reason() is None
    assert advisory.advisory_gate_unavailable() is False


def test_session_slot_with_its_own_target_reports_the_gate_available(monkeypatch):
    """A structured advisory row carrying its own parseable session target
    needs no shared route at all."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv(advisory.ADVISORY_REVIEW_ROUTE_ENV, raising=False)
    _clear_session_route_envs(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/x"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/y"}}],
        "advisory": {"enabled": True,
                     "route": {"kind": "agent_session", "target_id": "codex"}},
    }))
    assert advisory.advisory_gate_unavailability_reason() is None
    assert advisory.advisory_gate_unavailable() is False


def test_malformed_advisory_configuration_keeps_value_error_authority(monkeypatch):
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    monkeypatch.setenv(advisory.ADVISORY_REVIEW_ROUTE_ENV, "cursor")
    with pytest.raises(ValueError):
        advisory.advisory_gate_unavailability_reason()
    with pytest.raises(ValueError):
        advisory.advisory_gate_unavailable()


# ---------------------------------------------------------------------------
# The advisory row's model/effort on the api route (6.1 + the D-5b fix)
# ---------------------------------------------------------------------------


_SLOTS_API_ADVISORY = json.dumps({
    "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/x"}}],
    "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/y"}}],
    "advisory": {"enabled": True, "route": {"kind": "api", "target_id": "sonnet"},
                 "effort": "high"},
})


def _stub_run_readonly(captured):
    from types import SimpleNamespace

    def _fake(prompt, cwd, model, max_turns=None, effort="", max_budget_usd=None, **kwargs):
        captured.update({"model": model, "effort": effort})
        return SimpleNamespace(success=True, result_text=_ADVISORY_ITEMS, session_id="sess-1",
                               cost_usd=0.0, usage={}, error="", stderr_tail="")
    return _fake


def test_api_route_applies_the_advisory_rows_model_and_effort(tmp_path, monkeypatch):
    """On kind=api the row's target_id IS the Claude-SDK model and the row's
    effort rides the call — not resolve_effort('scope_review') (D-5b): the
    owner's advisory effort used to be silently ignored on this route."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", _SLOTS_API_ADVISORY)
    captured = {}
    monkeypatch.setattr("ouroboros.gateways.claude_code.run_readonly",
                        _stub_run_readonly(captured))
    ctx = _ctx(tmp_path)
    items, raw, model, _chars = advisory._run_claude_advisory(
        ctx.repo_dir, "msg", ctx, options={"include_repo_diff": False},
    )
    assert not raw.startswith("⚠️ ADVISORY_ERROR"), raw
    assert [i["item"] for i in items] == ["correctness"]
    assert captured["model"] == "sonnet" and model == "sonnet"
    assert captured["effort"] == "high"


def test_api_route_empty_target_falls_back_to_the_environment_default(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "opus-env-default")
    slots = json.loads(_SLOTS_API_ADVISORY)
    slots["advisory"] = {"enabled": True, "route": {"kind": "api", "target_id": ""}}
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps(slots))
    captured = {}
    monkeypatch.setattr("ouroboros.gateways.claude_code.run_readonly",
                        _stub_run_readonly(captured))
    ctx = _ctx(tmp_path)
    _items, raw, model, _chars = advisory._run_claude_advisory(
        ctx.repo_dir, "msg", ctx, options={"include_repo_diff": False},
    )
    assert not raw.startswith("⚠️ ADVISORY_ERROR"), raw
    assert captured["model"] == "opus-env-default" and model == "opus-env-default"
    # The parser's non-empty default ("low") is still the ROW's field — the
    # scope reviewer's effort never leaks in.
    assert captured["effort"] == "low"


def test_advisory_session_target_still_rejects_double_colon():
    from ouroboros.reviewer_slot_config import parse_reviewer_slots

    slots = json.loads(_SLOTS_API_ADVISORY)
    slots["advisory"] = {"enabled": True,
                         "route": {"kind": "agent_session", "target_id": "codex::gpt"}}
    with pytest.raises(ValueError, match="'::'"):
        parse_reviewer_slots(json.dumps(slots))


# ---------------------------------------------------------------------------
# The route reader itself
# ---------------------------------------------------------------------------


def test_advisory_route_reader_vocabulary(monkeypatch):
    monkeypatch.delenv(advisory.ADVISORY_REVIEW_ROUTE_ENV, raising=False)
    assert advisory.advisory_review_route() == "api"
    assert advisory.advisory_route_requires_api_key() is True
    monkeypatch.setenv(advisory.ADVISORY_REVIEW_ROUTE_ENV, "agent_session")
    assert advisory.advisory_review_route() == "agent_session"
    assert advisory.advisory_route_requires_api_key() is False
    monkeypatch.setenv(advisory.ADVISORY_REVIEW_ROUTE_ENV, "cursor")
    with pytest.raises(ValueError):
        advisory.advisory_review_route()
