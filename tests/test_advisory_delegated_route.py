"""The advisory review's two deliveries after the Claude-SDK retirement.

An ``api_chat`` advisory row runs the bounded NATIVE inspection episode on a
routed catalog model — its availability follows the model's provider
credentials (loud typed auto-bypass at the gate, typed error on a direct
call), never a hardcoded ANTHROPIC_API_KEY probe. An ``agent_session`` row is
a delegated Claudexor run, unchanged. The retired legacy ``api`` kind parses
and migrates same-model; an untranslatable target force-disables the row with
a typed reason.

Offline fixtures throughout (owner test rule): the FakeGateway from the
agent-session route tests stands in for the Claudexor control plane.
"""

import json
import pathlib
import subprocess
from types import SimpleNamespace

import pytest

import ouroboros.tools.claude_advisory_review as advisory
from tests.test_review_agent_session_route import FakeGateway, _terminal_detail


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


_PROVIDER_KEY_ENVS = (
    "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    "MINIMAX_API_KEY", "GIGACHAT_AUTH_KEY", "CLOUD_RU_API_KEY",
)


def _clear_provider_keys(monkeypatch):
    for key in _PROVIDER_KEY_ENVS:
        monkeypatch.delenv(key, raising=False)


def _fake_attempt(text):
    from ouroboros.review_execution import ReviewAttemptResult

    return ReviewAttemptResult(
        message={"content": text, "native_transcript": text},
        usage={"cost": 0.01, "resolved_model": "", "native_rounds": 1},
        raw_text=text,
    )

_ADVISORY_ITEMS = json.dumps([
    {"item": "correctness", "verdict": "PASS", "severity": "advisory",
     "reason": "checked the change end to end"},
])


# ---------------------------------------------------------------------------
# Site 1 — the run_readonly entry check
# ---------------------------------------------------------------------------


def test_native_route_without_model_credentials_errors_typed(tmp_path, monkeypatch):
    _clear_provider_keys(monkeypatch)
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    ctx = _ctx(tmp_path)
    items, raw, model, chars = advisory._run_claude_advisory(
        ctx.repo_dir, "msg", ctx, options={"include_repo_diff": False},
    )
    assert items == [] and model == "" and chars == 0
    assert raw.startswith("⚠️ ADVISORY_ERROR: no provider credentials for advisory model")


def test_delegated_route_runs_without_the_key(tmp_path, monkeypatch, fake_route):
    """The whole free-route walk: no key anywhere, route=agent_session, and the
    advisory runs as a delegated session whose checklist comes back parsed."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # ABI-10: the delegated advisory is configured through the structured slots
    # (target = the fixture's fake harness route).
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/x"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/y"}}],
        "advisory": {"enabled": True,
                     "route": {"kind": "agent_session", "target_id": "fake-review=fake-small"}},
    }))
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
    assert model == ""  # No final-attempt telemetry: the requested label is not observation.
    start = fake_route.instances[0].start_requests[0]
    assert start["authPreference"] == "subscription"
    assert start["access"] == "readonly"


def test_delegated_advisory_passes_expired_owner_deadline_before_dispatch(
    tmp_path, monkeypatch, fake_route,
):
    """The advisory consumer must pass its task deadline into the shared runner.

    An expired deadline is a host-side admission refusal, so the paid Claudexor
    start must never be posted.  This exercises the real advisory caller rather
    than only testing ``SessionInvocation`` in isolation.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = _ctx(tmp_path)
    ctx.task_metadata = {"deadline_at": "2000-01-01T00:00:00Z"}

    result, _model = advisory._run_advisory_delegated(
        "review", pathlib.Path(ctx.repo_dir), ctx,
    )

    assert result.success is False
    assert "owner deadline leaves no dispatch window" in result.error
    assert not any(instance.start_requests for instance in fake_route.instances)


def test_delegated_advisory_narrows_poll_window_to_owner_deadline(
    tmp_path, monkeypatch, fake_route,
):
    """A live advisory poll cannot outlive the task's remaining window."""
    from datetime import datetime, timedelta, timezone

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("ouroboros.config.get_finalization_grace_sec", lambda: 0)
    captured = {}

    def _fake_runner(**kwargs):
        captured["invocation"] = kwargs["invocation"]
        return {
            "text": "[]", "run_id": "run-1", "route_id": "fake-review",
            "model": "fake-small", "spend": None, "spend_estimated": False,
            "settlement": {}, "schema_asked": False, "conformance": "failed",
            "effective_route_ids": ["fake-review"],
        }

    import ouroboros.review_execution as review_execution
    monkeypatch.setattr(review_execution, "run_delegated_review_session", _fake_runner)
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=45)).isoformat()
    ctx = _ctx(tmp_path)
    ctx.task_metadata = {"deadline_at": deadline}

    result, _model = advisory._run_advisory_delegated(
        "review", pathlib.Path(ctx.repo_dir), ctx,
    )

    assert result.success is True
    invocation = captured["invocation"]
    assert invocation.owner_deadline_at == deadline
    assert 0 < invocation.timeout_sec <= 45


def test_delegated_advisory_does_not_start_inside_finalization_reserve(
    tmp_path, monkeypatch, fake_route,
):
    from datetime import datetime, timedelta, timezone

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    ctx = _ctx(tmp_path)
    ctx.task_metadata = {
        "deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat(),
    }

    result, _model = advisory._run_advisory_delegated(
        "review", pathlib.Path(ctx.repo_dir), ctx,
    )

    assert result.success is False
    assert "owner deadline leaves no dispatch window" in result.error
    assert not any(instance.start_requests for instance in fake_route.instances)


def test_structured_session_without_effort_preserves_the_route_default(
    tmp_path, monkeypatch, fake_route,
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/x"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/y"}}],
        "advisory": {
            "enabled": True,
            "route": {"kind": "agent_session", "target_id": "fake-review=fake-small"},
        },
    }))
    fake_route.detail = _terminal_detail(_ADVISORY_ITEMS)
    ctx = _ctx(tmp_path)

    items, raw, _model, _chars = advisory._run_claude_advisory(
        ctx.repo_dir, "msg", ctx,
        options={"include_repo_diff": False},
    )
    assert not raw.startswith("⚠️ ADVISORY_ERROR")
    assert [item["item"] for item in items] == ["correctness"]
    assert "effort" not in fake_route.instances[0].start_requests[0]


def test_unknown_route_token_is_a_loud_error_not_a_transport_pick(tmp_path, monkeypatch):
    # ABI-10: an unknown advisory route kind arrives via the structured value.
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/x"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/y"}}],
        "advisory": {"enabled": True, "route": {"kind": "codex", "target_id": "x"}},
    }))
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


def test_missing_credentials_auto_bypass_only_on_the_native_route(tmp_path, monkeypatch):
    """THE dangerous site: a native route whose model has no provider
    credentials auto-bypasses with the audited record; on the delegated route
    the gate RUNS — the advisory is actually invoked and no bypass is recorded."""
    from ouroboros.tools.registry import ToolContext

    repo = _git_repo(tmp_path)
    (repo / "README.md").write_text("hello\nchanged\n", encoding="utf-8")  # a real diff to review
    drive = tmp_path / "data"
    drive.mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    _clear_provider_keys(monkeypatch)

    # Native route: credential-less model auto-bypasses, loudly and audited.
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    payload = json.loads(advisory._handle_advisory_pre_review(
        ctx, commit_message="m", skip_tests=True,
    ))
    assert payload["status"] == "bypassed"
    assert "auto-bypassed" in payload["bypass_reason"]
    assert "no provider credentials" in payload["bypass_reason"]

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
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/x"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/y"}}],
        "advisory": {"enabled": True,
                     "route": {"kind": "agent_session", "target_id": "claude"}},
    }))
    payload = json.loads(advisory._handle_advisory_pre_review(
        ctx, commit_message="m", skip_tests=True,
    ))
    assert called.get("ran") is True
    assert payload["status"] != "bypassed"
    # No auto-bypass row was recorded for the delegated attempt.
    events = (drive / "logs" / "events.jsonl")
    if events.exists():
        rows = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
        api_bypasses = [r for r in rows if r.get("type") == "advisory_review_bypassed"]
        assert len(api_bypasses) == 1  # only the native-route attempt above


def test_explicit_skip_still_bypasses_on_the_delegated_route(tmp_path, monkeypatch):
    """The owner's explicit audited bypass is route-independent and untouched."""
    from ouroboros.tools.registry import ToolContext

    repo = _git_repo(tmp_path)
    drive = tmp_path / "data"
    drive.mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/x"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/y"}}],
        "advisory": {"enabled": True,
                     "route": {"kind": "agent_session", "target_id": "claude"}},
    }))
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

    # The bypass decision lives in the advisory+tests gate helper the stage
    # cycle calls (extracted whole at the function-size gate).
    stage_source = inspect.getsource(git_mod._run_reviewed_stage_cycle)
    assert "_advisory_and_tests_gate" in stage_source
    assert 'os.environ.get("ANTHROPIC_API_KEY"' not in stage_source
    gate_source = inspect.getsource(git_mod._advisory_and_tests_gate)
    assert "advisory_gate_unavailable" in gate_source
    assert 'os.environ.get("ANTHROPIC_API_KEY"' not in gate_source


def _clear_session_route_envs(monkeypatch):
    monkeypatch.delenv("OUROBOROS_REVIEW_SESSION_ROUTE", raising=False)
    monkeypatch.delenv("OUROBOROS_SUBAGENT_HARNESS", raising=False)


def test_disabled_slot_has_a_stable_reason_and_boolean_projection(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-sentinel")
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/x"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/y"}}],
        "advisory": {"enabled": False, "route": {"kind": "api", "target_id": "sonnet"}},
    }))
    assert advisory.advisory_gate_unavailability_reason() == "advisory_slot_disabled"
    assert advisory.advisory_gate_unavailable() is True


def test_credential_less_native_slot_has_a_stable_reason_and_boolean_projection(monkeypatch):
    _clear_provider_keys(monkeypatch)
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    assert advisory.advisory_gate_unavailability_reason() == "advisory_model_credentials_missing"
    assert advisory.advisory_gate_unavailable() is True


def test_unroutable_session_slot_reports_the_gate_unavailable(monkeypatch):
    """Triad a4 follow-up to #123: kind=agent_session with NO resolvable route
    anywhere (no row target, no shared review/subagent route) structurally
    cannot run — ``run_delegated_review_session`` refuses that exact state with
    ``ReviewRouteUnavailable`` — so the gate must report UNAVAILABLE. The key
    is present to prove the decision is route-driven, not key-driven."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-sentinel")
    _clear_session_route_envs(monkeypatch)
    # ABI-10: the parser refuses an enabled empty-target session advisory at
    # save AND load, so the unroutable state is synthesized to keep the
    # defensive gate branch covered.
    from ouroboros import reviewer_slot_config as slot_cfg

    _unroutable = slot_cfg.AdvisorySlotConfig(
        enabled=True, kind="agent_session", target_id="", effort="")
    monkeypatch.setattr(slot_cfg, "advisory_slot_config", lambda: _unroutable)
    assert advisory.advisory_gate_unavailability_reason() == "agent_session_route_unavailable"
    assert advisory.advisory_gate_unavailable() is True


def test_session_slot_with_shared_route_reports_the_gate_available(monkeypatch):
    """The shared subagent route is the delegated advisory's documented
    fallback target — with it set, the keyless session slot is AVAILABLE."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _clear_session_route_envs(monkeypatch)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "claude")
    # ABI-10: a config-reachable session advisory always carries its own
    # target; the shared-route fallback line is kept covered synthetically.
    from ouroboros import reviewer_slot_config as slot_cfg

    _shared = slot_cfg.AdvisorySlotConfig(
        enabled=True, kind="agent_session", target_id="", effort="")
    monkeypatch.setattr(slot_cfg, "advisory_slot_config", lambda: _shared)
    assert advisory.advisory_gate_unavailability_reason() is None
    assert advisory.advisory_gate_unavailable() is False


def test_structured_empty_session_slot_never_uses_the_shared_route(monkeypatch):
    """A saved structured row names its own exact session route or refuses.

    The shared route remains the legacy environment fallback covered above;
    it must not silently replace an incomplete structured owner setting.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _clear_session_route_envs(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_SESSION_ROUTE", "codex=gpt-5.6-sol:high")
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/x"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/y"}}],
        "advisory": {"enabled": True,
                     "route": {"kind": "agent_session", "target_id": ""}},
    }))
    with pytest.raises(ValueError, match="needs a non-empty target_id"):
        advisory.advisory_gate_unavailability_reason()


def test_session_slot_with_its_own_target_reports_the_gate_available(monkeypatch):
    """A structured advisory row carrying its own parseable session target
    needs no shared route at all."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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
    # ABI-10: malformed configuration arrives via the structured value.
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", "{not-valid-json")
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

    def _fake(prompt, cwd, model, max_turns=None, effort="", max_budget_usd=None, **kwargs):
        captured.update({"model": model, "effort": effort})
        return SimpleNamespace(success=True, result_text=_ADVISORY_ITEMS, session_id="sess-1",
                               cost_usd=0.0, usage={}, error="", stderr_tail="")
    return _fake


def test_native_route_applies_the_advisory_rows_model_and_effort(tmp_path, monkeypatch):
    """The row's routed target (legacy 'sonnet' migrates same-model to the
    shipped routed default) and the row's own effort ride the native episode
    (D-5b) — never resolve_effort('scope_review')."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", _SLOTS_API_ADVISORY)
    captured = {}

    def _capture_native(prompt, repo_dir, ctx_, slot, model, **_):

        captured.update({"model": model, "effort": slot.effort})
        return SimpleNamespace(
            success=True, result_text=_ADVISORY_ITEMS, session_id="",
            cost_usd=0.0, usage={}, error="", stderr_tail="",
        ), model

    monkeypatch.setattr(advisory, "_run_advisory_native", _capture_native)
    ctx = _ctx(tmp_path)
    items, raw, model, _chars = advisory._run_claude_advisory(
        ctx.repo_dir, "msg", ctx, options={"include_repo_diff": False},
    )
    assert not raw.startswith("⚠️ ADVISORY_ERROR"), raw
    assert [i["item"] for i in items] == ["correctness"]
    assert captured["model"] == advisory._advisory_default_model() == model
    assert captured["effort"] == "high"


def test_native_advisory_passes_owner_deadline_into_the_episode(tmp_path, monkeypatch):
    """The bounded episode enforces the owner deadline per round; the advisory
    caller must hand it the task's deadline_at verbatim."""
    from datetime import datetime, timedelta, timezone

    from ouroboros import review_native_episode as native

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", _SLOTS_API_ADVISORY)
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "1")
    captured = {}

    def _capture_execute(self):
        captured["deadline_at"] = self.assignment.request.deadline_at
        captured["session_root"] = self.assignment.request.session_root
        return _fake_attempt(_ADVISORY_ITEMS)

    monkeypatch.setattr(
        native.NativeToolRoundReviewExecutor, "execute", _capture_execute,
    )
    ctx = _ctx(tmp_path)
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=600)).isoformat()
    ctx.task_metadata = {"deadline_at": deadline}

    _items, raw, _model, _chars = advisory._run_claude_advisory(
        ctx.repo_dir, "msg", ctx, options={"include_repo_diff": False},
    )

    assert not raw.startswith("⚠️ ADVISORY_ERROR"), raw
    assert captured["deadline_at"] == deadline
    assert captured["session_root"] == str(ctx.repo_dir)


def test_native_advisory_does_not_start_inside_finalization_reserve(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", _SLOTS_API_ADVISORY)
    calls = []
    monkeypatch.setattr(
        advisory, "_run_advisory_native",
        lambda *a, **k: calls.append(1),
    )
    ctx = _ctx(tmp_path)
    ctx.task_metadata = {
        "deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat(),
    }

    _items, raw, _model, _chars = advisory._run_claude_advisory(
        ctx.repo_dir, "msg", ctx, options={"include_repo_diff": False},
    )

    assert calls == []
    assert "owner deadline leaves no dispatch window" in raw


def test_native_route_empty_target_falls_back_to_the_routed_default(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    slots = json.loads(_SLOTS_API_ADVISORY)
    slots["advisory"] = {"enabled": True, "route": {"kind": "api", "target_id": ""}}
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps(slots))
    captured = {}

    def _capture_native(prompt, repo_dir, ctx_, slot, model, **_):

        captured.update({"model": model, "effort": slot.effort or "low"})
        return SimpleNamespace(
            success=True, result_text=_ADVISORY_ITEMS, session_id="",
            cost_usd=0.0, usage={}, error="", stderr_tail="",
        ), model

    monkeypatch.setattr(advisory, "_run_advisory_native", _capture_native)
    ctx = _ctx(tmp_path)
    _items, raw, model, _chars = advisory._run_claude_advisory(
        ctx.repo_dir, "msg", ctx, options={"include_repo_diff": False},
    )
    assert not raw.startswith("⚠️ ADVISORY_ERROR"), raw
    assert captured["model"] == advisory._advisory_default_model() == model
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
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    # Default panel advisory is the routed native episode.
    assert advisory.advisory_review_route() == "api_chat"
    # ABI-10: route selection arrives via the structured slots only — the
    # retired route env is ignored even when set.
    assert advisory.advisory_review_route() == "api_chat"
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", json.dumps({
        "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "openai/x"}}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "openai/y"}}],
        "advisory": {"enabled": True,
                     "route": {"kind": "agent_session", "target_id": "claude"}},
    }))
    assert advisory.advisory_review_route() == "agent_session"
    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", "{not-valid-json")
    with pytest.raises(ValueError):
        advisory.advisory_review_route()


def test_delegated_advisory_rides_the_shared_executor_seam():
    """Phase C unification (owner decision 2=B, 2026-08-30): the delegated
    advisory is one AgentSessionReviewExecutor — retry/invocation custody,
    D19 verdict order, and delta disclosure all come from the substrate; the
    advisory's own transport dialect (_advisory_session_deltas and a direct
    runner call) is gone."""
    import inspect

    from ouroboros.tools import claude_advisory_review as adv

    source = inspect.getsource(adv._run_advisory_delegated)
    assert "AgentSessionReviewExecutor" in source
    assert "run_delegated_review_session" not in source
    assert not hasattr(adv, "_advisory_session_deltas")
