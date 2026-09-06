"""S1-S5 — the deep-integration suite scenarios (v7next plan §8, roast F22).

WHAT THIS FILE IS. The harness skeleton (``tests/system_e2e/harness.py``) landed in
Ф0; Ф4 lane 1 adds the first mandatory surfaces. The remaining scenario matrix
(subagents, delegation, update engine, cancellation E-suite, skills, UI truth, …)
lands WITH its phases and must survive the domain transplants unchanged:

* S1 — boot / identity / WS / port-file / task contract: a real ``server.py`` on an
  isolated clone + data root boots to the frozen readiness contract, attests its
  identity against its own checkout, publishes an honest ``state/server_port``,
  answers a WS chat frame with an assistant reply, runs one scripted stub task to
  completion, and leaves a sane durable ``task_results/<id>.json`` behind.
* S2 — review organ: a scripted task drives ``commit_reviewed`` over a doc-only diff
  with the advisory pre-review explicitly skipped (audited bypass) and BLOCKING
  enforcement, the stub answers the triad packet and the scope-matrix packet with
  all-clean verdicts, and the commit lands in the isolated clone. Landing under
  ``blocking`` makes the git log itself the proof that both review organs ran and
  passed — under advisory a failed review would still commit.
* S3 — egress hardening (plan: "дыра ANTHROPIC_API_KEY — закрыть в Ф4 первой"): a
  poisoned parent env, a live keyless server tree, and a /proc environ probe of every
  process in it — no planted or real credential value is visible to any child.
* S4 — typed tools + safety: a protected-path write is denied with the typed
  ``CORE_PROTECTION_BLOCKED`` refusal and ZERO side effects (tree snapshot equality).
* S5 — cost-truth smoke (ABI-3): a completed task's public projections carry
  honest-only cost names; the retired aliases appear nowhere in the outbound bytes.

LANES. Default (always-on) tests pin the harness's own contracts with no server and no
sockets: the scenario manifest (gen/verify in BOTH directions), the stub's branch
classification (review-organ branch BEFORE the finalization check), the prompt-marker
literals against the tree's source, the ReplayModel binding/consumption contract, and
the keyless/egress hardening (roast F21). The ``mock`` lane spawns real isolated
servers; every scenario test carries BOTH ``integration`` and ``serial`` markers (so
neither the default local run, nor either CI pytest pass, nor the CI-shape battery
picks it up) AND the ``OUROBOROS_E2E_DEEP=mock`` env gate. Run it with::

    OUROBOROS_E2E_DEEP=mock pytest tests/system_e2e/ -o addopts="" -q

No paid lane exists yet.

Every scenario asserts durable artifacts — never an HTTP 200 on its own and never a
harness exit code (AGENTS.md: the exit code is not the run status). Fail-injection /
completion synchronization is durable-event polling (``wait_until`` over oracle
readers), never bare sleeps.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

from ouroboros.settings_defaults import SETTINGS_DEFAULTS, settings_env_keys
from tests.system_e2e.harness import (
    _CREDENTIAL_SHAPE_RE,
    ACCEPTANCE_KEYS_MARKER,
    LANE_MOCK,
    MARKER_SOURCES,
    MOCK_SLUG,
    PROXY_ENV_KEYS,
    REPO_ROOT,
    REVIEWER_SLOT_MARKER,
    SCENARIOS,
    SCOPE_USER_MARKER,
    STRIPPED_PROVIDER_ENV_KEYS,
    TRIAD_USER_MARKER,
    ArtifactOracle,
    KeylessIsolatedServer,
    ReplayModel,
    ScriptedStubModel,
    assert_settings_keyless,
    classify_call,
    keyless_reviewer_slots,
    keyless_settings,
    proc_environ,
    process_tree_pids,
    repo_tree_fingerprint,
    require_lane,
    retired_cost_alias_paths,
    runtime_credential_env_key_reads,
    scripted_completion,
    secret_values_in_parent_env,
    start_server,
    submit_running,
    supervisor_state_is_ready,
    wait_durable_result,
    wait_until,
    write_settings_file,
    ws_url,
)

# ===========================================================================
# Default lane: harness self-contracts. No server, no model, no egress.
# ===========================================================================


_SCENARIO_TEST_RE = re.compile(r"^test_(s\d+)_")


def _scenario_modules() -> list:
    """Every scenario module of the suite package — the manifest pins must see the
    WHOLE package, not this file: a scenario wave landing in its own module (wave 2
    did) would otherwise be invisible to the gen/verify discipline."""
    import importlib

    package_dir = pathlib.Path(__file__).resolve().parent
    return [
        importlib.import_module(f"tests.system_e2e.{path.stem}")
        for path in sorted(package_dir.glob("test_*.py"))
    ]


def _scenario_tests() -> list:
    """``(qualified name, function)`` for every test_s<N>_* test in the package."""
    found = []
    for module in _scenario_modules():
        for name in dir(module):
            if _SCENARIO_TEST_RE.match(name):
                found.append((f"{module.__name__}.{name}", name, getattr(module, name)))
    return found


def test_system_manifest_is_covered():
    """Every S-id in the scenario manifest still has at least one test in the package."""
    names = [name for _qual, name, _fn in _scenario_tests()]
    for scenario_id, (title, _lane) in SCENARIOS.items():
        prefix = f"test_{scenario_id.lower()}_"
        assert any(name.startswith(prefix) for name in names), (
            f"scenario {scenario_id} ({title}) has no {prefix}* test"
        )


def test_every_scenario_test_is_declared_in_the_manifest():
    """The verify direction of the manifest pin: a NEW ``test_s<N>_*`` test without a
    ``SCENARIOS`` row is red — an undeclared scenario would be invisible to the lane
    budget and to the retirement discipline."""
    declared = {scenario_id.lower() for scenario_id in SCENARIOS}
    tests = _scenario_tests()
    assert tests, "the package-wide scenario scan found no scenario tests at all"
    for qualified, name, _fn in tests:
        sid = _SCENARIO_TEST_RE.match(name).group(1)
        assert sid in declared, (
            f"{qualified} has no SCENARIOS[{sid.upper()!r}] manifest row "
            "(tests/system_e2e/harness.py) — declare the scenario first"
        )


def test_every_scenario_test_carries_integration_and_serial_markers():
    """The marker discipline that keeps the suite OUT of the default non-serial run,
    both CI pytest passes, and the CI-shape battery: every scenario test must carry
    ``integration`` AND ``serial``. (The ``OUROBOROS_E2E_DEEP`` env gate is the second
    belt; markers are what the -m expressions see.)"""
    for qualified, _name, fn in _scenario_tests():
        marks = {mark.name for mark in getattr(fn, "pytestmark", [])}
        assert {"integration", "serial"} <= marks, (
            f"{qualified} must be decorated with @pytest.mark.integration and "
            f"@pytest.mark.serial (has: {sorted(marks)})"
        )


def test_prompt_markers_still_exist_in_the_tree():
    """The stub's review-organ classification is prompt-marker based; a marker that
    drifts out of the source it was pinned from would leave the stub silently mute on
    that organ — surface the drift as a NAMED failure instead."""
    for marker, relpath in MARKER_SOURCES.items():
        source = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert marker in source, (
            f"marker {marker!r} no longer appears in {relpath}: upstream prompt drifted, "
            "re-pin the literal in tests/system_e2e/harness.py"
        )


def _agent_body(text: str = "keep going", *, tools: bool = True) -> dict:
    body: dict = {"messages": [{"role": "user", "content": text}]}
    if tools:
        body["tools"] = [{"type": "function", "function": {"name": "list_files"}}]
    return body


def test_stub_classification_review_branch_beats_finalization():
    """Roast F22: the review-organ branch sits BEFORE the finalization-turn check.

    A triad/scope/reviewer-slot packet that happens to QUOTE a finalization marker
    (review of a stopped task's transcript) must still be answered as a review."""
    scope_body = {"messages": [
        {"role": "system", "content": [{"type": "text", "text": "scope pack [OWNER_STOP] quoted"}]},
        {"role": "user", "content": SCOPE_USER_MARKER},
    ], "tools": []}
    triad_body = {"messages": [
        {"role": "system", "content": [{"type": "text", "text": "triad pack [FINALIZE_NOW] quoted"}]},
        {"role": "user", "content": "Review the staged diff and context provided in the instructions above."},
    ]}
    slot_body = {"messages": [
        {"role": "system", "content": REVIEWER_SLOT_MARKER + "\nSurface: plan_review\n [OWNER_STOP]"},
        {"role": "user", "content": "Subject: ..."},
    ]}
    acceptance_body = {"messages": [
        {"role": "system", "content": REVIEWER_SLOT_MARKER + "\n" + ACCEPTANCE_KEYS_MARKER},
        {"role": "user", "content": "Subject: ..."},
    ]}
    assert classify_call(scope_body) == "scope_review"
    assert classify_call(triad_body) == "triad_review"
    assert classify_call(slot_body) == "reviewer_slot"
    assert classify_call(acceptance_body) == "acceptance"
    assert classify_call({"messages": [{"role": "user", "content": "[FINALIZE_NOW] wrap up"}]}) == "finalization"
    assert classify_call({"messages": [{"role": "user", "content": "hi"}],
                          "response_format": {"type": "json_object"}}) == "safety"
    assert classify_call(_agent_body()) == "agent"


def test_stub_verdicts_satisfy_the_trees_own_parsers():
    """The canned all-clean answers must parse under the REAL review contracts of this
    tree — a stub that emits an unparseable verdict turns every review into a
    parse_failure and the S2 smoke into a lie."""
    from ouroboros.tools.scope_review_contract import (
        SCOPE_REQUIRED_ITEMS,
        classify_scope_findings,
        normalize_scope_items,
    )
    from ouroboros.triad_review import empty_array_is_verified_clean

    _kind, scope_message = scripted_completion(
        {"messages": [{"role": "user", "content": SCOPE_USER_MARKER}]}, 1, lambda _b: None, "x")
    items, errors = normalize_scope_items(json.loads(scope_message["content"]))
    assert not errors, f"stub scope verdict rejected by normalize_scope_items: {errors}"
    assert {item["item"] for item in items} == set(SCOPE_REQUIRED_ITEMS)
    critical, advisory = classify_scope_findings(items)
    assert critical == [] and advisory == []

    _kind, triad_message = scripted_completion(
        {"messages": [{"role": "user", "content": TRIAD_USER_MARKER}]}, 1, lambda _b: None, "x")
    assert empty_array_is_verified_clean(triad_message["content"])

    _kind, slot_message = scripted_completion(
        {"messages": [{"role": "system", "content": REVIEWER_SLOT_MARKER}]}, 1, lambda _b: None, "x")
    verdict = json.loads(slot_message["content"])
    assert verdict["verdict"] == "PASS" and verdict["findings"] == []


def test_stub_consumes_the_script_in_order_then_finalizes():
    steps = iter([{"tool": "write_file", "arguments": {"path": "a.md"}},
                  {"tool": "commit_reviewed", "arguments": {"commit_message": "m"}}])

    def _next(_body):
        return next(steps, None)

    kind1, msg1 = scripted_completion(_agent_body(), 1, _next, "done")
    kind2, msg2 = scripted_completion(_agent_body(), 2, _next, "done")
    kind3, msg3 = scripted_completion(_agent_body(), 3, _next, "done")
    assert (kind1, kind2, kind3) == ("agent", "agent", "final")
    assert msg1["tool_calls"][0]["function"]["name"] == "write_file"
    assert msg2["tool_calls"][0]["function"]["name"] == "commit_reviewed"
    assert "tool_calls" not in msg3 and msg3["content"] == "done"
    # A tool-less prompt (final synthesis turn) never consumes a script step.
    kind4, _ = scripted_completion(_agent_body(tools=False), 4, _next, "done")
    assert kind4 == "final"


# ---------------------------------------------------------------------------
# Egress hardening (roast F21): the regression the plan names — a planted
# ANTHROPIC_API_KEY in the CALLER env must never reach the child server env.
# ---------------------------------------------------------------------------

def test_f21_planted_provider_key_never_reaches_child_env(tmp_path, monkeypatch):
    planted = {
        "ANTHROPIC_API_KEY": "sk-ant-planted-must-not-leak",
        "OPENROUTER_API_KEY": "sk-or-planted-must-not-leak",
        "OPENAI_API_KEY": "sk-planted-must-not-leak",
        "OPENAI_COMPATIBLE_API_KEY": "planted-must-not-leak",
        "GIGACHAT_CREDENTIALS": "planted-must-not-leak",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY": "planted-must-not-leak",
        "HTTP_PROXY": "http://proxy.invalid:3128",
        "https_proxy": "http://proxy.invalid:3128",
        "ALL_PROXY": "socks5://proxy.invalid:1080",
        "NO_PROXY": "localhost",
    }
    for key, value in planted.items():
        monkeypatch.setenv(key, value)
    server = KeylessIsolatedServer(
        tmp_path / "clone", tmp_path / "data", tmp_path / "data" / "settings.json")
    child_env = server._env()
    leaked = sorted(set(planted) & set(child_env))
    assert not leaked, f"planted caller-env values leaked into the child env: {leaked}"
    # The whole families, not just the planted samples:
    assert not (STRIPPED_PROVIDER_ENV_KEYS & set(child_env))
    assert not (PROXY_ENV_KEYS & set(child_env))
    # The child still gets its 4-var isolation set, pointing INTO the throwaway root.
    for key in ("OUROBOROS_APP_ROOT", "OUROBOROS_REPO_DIR",
                "OUROBOROS_DATA_DIR", "OUROBOROS_SETTINGS_PATH"):
        assert str(tmp_path) in child_env[key], (key, child_env[key])


def test_f21_base_isolated_server_still_leaks_provider_keys(tmp_path, monkeypatch):
    """The hole the keyless lane closes, pinned so its future upstream fix is VISIBLE:
    the base ``IsolatedServer`` deliberately keeps provider keys in the child. When
    this test starts failing, upstream closed the hole itself — collapse
    ``KeylessIsolatedServer`` accordingly instead of keeping a dead override."""
    from devtools.benchmarks.common.server_runner import IsolatedServer

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-planted")
    server = IsolatedServer(
        tmp_path / "clone", tmp_path / "data", tmp_path / "data" / "settings.json")
    assert server._env().get("ANTHROPIC_API_KEY") == "sk-ant-planted"


def test_f21_keyless_settings_pin_every_slot_and_refuse_credentials():
    from ouroboros.provider_models import (
        ACTIVE_MODEL_SETTING_KEYS,
        LEGACY_MODEL_SETTING_KEYS,
    )

    class _FakeStub:
        base_url = "http://127.0.0.1:1/v1"

    cfg = keyless_settings(_FakeStub())
    for slot in (*ACTIVE_MODEL_SETTING_KEYS, *LEGACY_MODEL_SETTING_KEYS):
        assert slot in cfg, f"model slot {slot} left unpinned"
        assert cfg[slot] in ("", MOCK_SLUG), (slot, cfg[slot])
    assert cfg["OUROBOROS_MODEL"] == MOCK_SLUG
    assert_settings_keyless(cfg)
    with pytest.raises(ValueError, match="provider credentials"):
        keyless_settings(_FakeStub(), ANTHROPIC_API_KEY="sk-ant-nope")
    with pytest.raises(AssertionError):
        assert_settings_keyless({**cfg, "OPENROUTER_API_KEY": "sk-or-nope"})
    with pytest.raises(AssertionError):
        assert_settings_keyless({**cfg, "OPENAI_COMPATIBLE_BASE_URL": "https://api.example.com/v1"})


def test_f21_projected_provider_family_credentials_default_empty():
    """The carve-out S3's name pin relies on can never legitimise a secret: the server
    projects settings ∪ ``SETTINGS_DEFAULTS`` into its children, so a credential-shaped
    member of the stripped provider family that is projected (``settings_env_keys``)
    must ship an EMPTY default — a keyless file then projects only endpoint fields
    (base URLs, scope, TLS flag), never a key. Credential shape comes from the
    harness's own regex over the family it strips, not from a hand list."""
    projected_family = STRIPPED_PROVIDER_ENV_KEYS & set(settings_env_keys())
    credential_shaped = sorted(k for k in projected_family if _CREDENTIAL_SHAPE_RE.search(k))
    # The derivation must see the canonical credentials, or an empty set passes vacuously.
    assert {"ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GIGACHAT_CREDENTIALS",
            "CLOUDRU_FOUNDATION_MODELS_API_KEY"} <= set(credential_shaped), credential_shaped
    non_empty = [k for k in credential_shaped if str(SETTINGS_DEFAULTS.get(k) or "").strip()]
    assert not non_empty, f"credential-shaped provider keys ship a non-empty default: {non_empty}"


def test_keyless_reviewer_slots_parse_under_the_trees_own_parser():
    """ABI 7.0 (ABI-10): the comma-list reviewer keys are RETIRED settings — pinning
    them in the isolated settings.json is a silent no-op and the review organ falls
    back to the shipped OpenRouter default panel (the exact failure observed live:
    S2's triad dispatched gemini/terra/opus keyless and blocked at pack assembly).
    The keyless lane therefore pins the STRUCTURED surface, and this test feeds it to
    the tree's own strict parser: every configured row must be an api_chat route onto
    the stub slug."""
    from ouroboros.reviewer_slot_config import parse_reviewer_slots

    config = parse_reviewer_slots(keyless_reviewer_slots())
    assert config.source == "structured"
    assert len(config.triad) >= 1 and len(config.scope) >= 1
    for row in (*config.triad, *config.scope):
        assert row.kind == "api_chat", row
        assert row.target_id == MOCK_SLUG, row


def test_f21_every_runtime_credential_env_read_is_stripped_from_the_keyless_child():
    """The CLASS pin behind the strip list: scan the runtime tree for every
    credential-shaped env key it actually reads (os.environ[...] / .get / os.getenv)
    and require each to be unreachable in a keyless child — stripped as a provider
    credential, stripped by the base secret-shape sanitizer, or stripped as a stale
    inherited runtime key. A provider credential added upstream tomorrow fails HERE
    by name instead of silently reaching a keyless server."""
    from devtools.benchmarks.common.server_runner import (
        STALE_INHERITED_ENV_KEYS,
        _is_secret_env_key,
    )

    observed = runtime_credential_env_key_reads()
    # The scan must actually see the canonical provider keys — an empty or blind scan
    # would vacuously pass; ANTHROPIC_API_KEY is the exact hole the plan names.
    assert {"ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"} <= observed, observed
    uncovered = sorted(
        key for key in observed
        if key not in STRIPPED_PROVIDER_ENV_KEYS
        and key not in STALE_INHERITED_ENV_KEYS
        and not _is_secret_env_key(key)
    )
    assert not uncovered, (
        "credential-shaped env keys the runtime reads but the keyless lane does not "
        f"strip: {uncovered} — extend STRIPPED_PROVIDER_ENV_KEYS / the sanitizer"
    )


# ---------------------------------------------------------------------------
# ReplayModel contract pins (plan §8): binding by (lineage, slot, attempt); the
# fixture must be consumed WHOLE — a leftover row or a miss is red.
# ---------------------------------------------------------------------------


def _replay(fixture) -> ReplayModel:
    return ReplayModel(fixture)


def test_replay_model_binds_by_lineage_slot_and_ordinal_attempt():
    model = _replay({
        ("root", "mock-model", 1): {"tool": "list_files", "arguments": {"path": "."}},
        ("root", "mock-model", 2): {"final": "root done"},
        ("child-a", "mock-model", 1): {"final": "child done"},
    })
    body_root = {"messages": [{"role": "user", "content": "work [E2E-LINEAGE:root]"}],
                 "tools": [{"type": "function", "function": {"name": "list_files"}}],
                 "model": "mock-model"}
    kind1, msg1 = model._answer(body_root, 1)
    assert kind1 == "replay" and msg1["tool_calls"][0]["function"]["name"] == "list_files"
    kind2, msg2 = model._answer(body_root, 2)
    assert (kind2, msg2["content"]) == ("replay_final", "root done")
    body_child = {"messages": [{"role": "user", "content": "go [E2E-LINEAGE:child-a]"}],
                  "model": "mock-model"}
    kind3, msg3 = model._answer(body_child, 3)
    assert (kind3, msg3["content"]) == ("replay_final", "child done")
    model.assert_consumed()


def test_replay_model_untagged_prompt_binds_to_root_and_last_tag_wins():
    model = _replay({
        ("root", "mock-model", 1): {"final": "untagged is root"},
        ("leaf", "mock-model", 1): {"final": "leaf"},
    })
    _, msg = model._answer({"messages": [{"role": "user", "content": "no tag"}],
                            "model": "mock-model"}, 1)
    assert msg["content"] == "untagged is root"
    # A child prompt quoting the parent's tag: the LAST tag wins.
    _, msg = model._answer({"messages": [
        {"role": "system", "content": "parent said [E2E-LINEAGE:root] earlier"},
        {"role": "user", "content": "you are [E2E-LINEAGE:leaf]"},
    ], "model": "mock-model"}, 2)
    assert msg["content"] == "leaf"
    model.assert_consumed()


def test_replay_model_review_calls_never_consume_the_fixture():
    model = _replay({("root", "mock-model", 1): {"final": "done"}})
    kind, _ = model._answer({"messages": [{"role": "user", "content": SCOPE_USER_MARKER}],
                             "model": "mock-model"}, 1)
    assert kind == "scope_review"
    kind, _ = model._answer({"messages": [{"role": "user", "content": "x"}],
                             "response_format": {"type": "json_object"},
                             "model": "mock-model"}, 2)
    assert kind == "safety"
    assert model.consumed == [] and model.misses == []
    _, msg = model._answer({"messages": [{"role": "user", "content": "go"}],
                            "model": "mock-model"}, 3)
    assert msg["content"] == "done"
    model.assert_consumed()


def test_replay_model_unconsumed_fixture_or_miss_is_red():
    model = _replay({
        ("root", "mock-model", 1): {"final": "served"},
        ("root", "mock-model", 2): {"final": "never asked for"},
    })
    model._answer({"messages": [{"role": "user", "content": "go"}], "model": "mock-model"}, 1)
    with pytest.raises(AssertionError, match="unconsumed fixture rows"):
        model.assert_consumed()

    empty = _replay({})
    kind, msg = empty._answer({"messages": [{"role": "user", "content": "go"}],
                               "model": "mock-model"}, 1)
    assert kind == "replay_miss" and "REPLAY_MISS" in msg["content"]
    with pytest.raises(AssertionError, match="missed keys"):
        empty.assert_consumed()

    with pytest.raises(ValueError, match="lineage, slot, attempt"):
        ReplayModel({"not-a-tuple": {"final": "x"}})


def test_interface_stubs_refuse_instantiation_until_their_lanes_land():
    """FakeClaudexorDaemon LANDED with the wave-3b delegated-transport lane (it
    constructs a bound loopback socket only on ``start()``); the UI client still
    refuses until the gateway/UI-truth wave lands."""
    from tests.system_e2e.interfaces import FakeClaudexorDaemon, PlaywrightUIClient

    daemon = FakeClaudexorDaemon()
    assert daemon.harness_id and daemon.token  # constructible, not yet serving
    daemon._server.server_close()              # release the bound (never-served) socket
    with pytest.raises(NotImplementedError, match="gateway/UI-truth"):
        PlaywrightUIClient()


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits are meaningless on Windows")
def test_settings_file_is_created_secret_safe(tmp_path):
    """0600-before-content (carried over from the v7_wip cancellation harness)."""
    settings_path = tmp_path / "settings.json"
    write_settings_file(settings_path, {"OPENAI_COMPATIBLE_API_KEY": "not-a-credential"})
    assert (settings_path.stat().st_mode & 0o777) == 0o600
    settings_path.chmod(0o664)
    write_settings_file(settings_path, {"OPENAI_COMPATIBLE_API_KEY": "not-a-credential"})
    assert (settings_path.stat().st_mode & 0o777) == 0o600


# ===========================================================================
# Mock lane: real isolated servers. Opt in with OUROBOROS_E2E_DEEP=mock.
# ===========================================================================


# The shared session clone fixture (``e2e_clone``) lives in the package conftest.

S1_SCRIPT = [
    {"tool": "list_files", "arguments": {"path": "."}},
]

S2_COMMIT_MESSAGE = "docs: system_e2e S2 review-organ smoke (doc-only)"
S2_DOC_PATH = "docs/notes/system_e2e_smoke.md"
S2_SCRIPT = [
    {"tool": "write_file", "arguments": {
        "root": "system_repo",
        "path": S2_DOC_PATH,
        "content": ("# system_e2e S2 smoke\n\n"
                    "Doc-only change landed through commit_reviewed by the scripted stub.\n"),
    }},
    {"tool": "commit_reviewed", "arguments": {
        "commit_message": S2_COMMIT_MESSAGE,
        "paths": [S2_DOC_PATH],
        # Audited advisory-only skip (recorded as `bypassed` in the ledger) — the
        # scenario's subject is the triad+scope organ, not the advisory pre-review.
        "skip_advisory_review": True,
        # The post-commit hermetic pytest is out of scope for a smoke that proves the
        # review organ; the skip is recorded in the commit attempt.
        "skip_tests": True,
        "goal": "Land a doc-only smoke note through the full triad+scope review organ.",
        "scope": f"{S2_DOC_PATH} only.",
    }},
]


@pytest.mark.integration
@pytest.mark.serial
def test_s1_boot_identity_ws_chat_and_task_contract(e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s1")
    with ScriptedStubModel(S1_SCRIPT) as stub:
        server = start_server(e2e_clone, root, keyless_settings(stub))
        try:
            # Boot + identity: the frozen readiness contract, and the attestation the
            # readiness path took (runtime identity == the clone it booted from).
            state = server._state()
            assert supervisor_state_is_ready(state), state
            attestation = server.attestation
            assert attestation.get("ok") is True, attestation
            assert re.fullmatch(r"[0-9a-f]{40}", str(attestation.get("repo_head") or "")), attestation
            assert attestation.get("runtime_version") == attestation.get("repo_version")

            # Port-file honesty: the DURABLE port claim (state/server_port) names the
            # port this driver is actually talking to — the CLI attach contract.
            oracle = ArtifactOracle(server.data_root)
            assert wait_until(lambda: oracle.server_port() == server.port, 30), (
                f"state/server_port says {oracle.server_port()}, driver talks to {server.port}"
            )

            # Contract: one scripted task to completion over the same HTTP surface the
            # UI posts to.
            task_id = submit_running(server, "List the repository root and finish.")
            result = server.wait_task(task_id, timeout=300)
            assert result.get("status") == "completed", result

            # Durable truth, not the HTTP answer: task_results/<id>.json.
            stored = wait_durable_result(oracle, task_id)
            assert stored.get("task_id") == task_id, stored
            assert stored.get("status") == "completed", stored
            assert str(stored.get("result") or "").strip(), "durable result text is empty"
            json.loads(oracle.task_result_bytes(task_id))  # bytes on disk are valid JSON

            # The queue drained and the stub actually drove the loop.
            assert wait_until(lambda: task_id not in oracle.running_ids(), 60)
            kinds = stub.kinds()
            assert "agent" in kinds and "final" in kinds, kinds
            assert stub.script_consumed(), "S1 script was not fully consumed"
            assert oracle.events(), "events.jsonl is empty after a completed task"

            # WS chat: the SAME /ws surface the SPA opens answers a chat frame with an
            # assistant reply (script exhausted -> the stub's final answer), and the
            # exchange lands durably in chat.jsonl. Runs AFTER the task so the direct
            # turn cannot race the task for the script's tool step.
            from websockets.sync.client import connect as ws_connect

            probe = "E2E WS chat probe s1"
            with ws_connect(ws_url(server), open_timeout=30) as ws:
                ws.send(json.dumps({"type": "chat", "content": probe, "chat_id": 1}))
                reply = ""
                deadline_frames = 240  # bounded frame reads, each with its own timeout
                for _ in range(deadline_frames):
                    try:
                        frame = json.loads(ws.recv(timeout=1.0))
                    except TimeoutError:
                        continue
                    if (isinstance(frame, dict) and frame.get("type") == "chat"
                            and frame.get("role") == "assistant"
                            and str(frame.get("content") or "").strip()):
                        reply = str(frame["content"])
                        break
                assert reply, "no assistant chat frame arrived over /ws"
            assert wait_until(lambda: probe.encode() in oracle.chat_bytes(), 60), (
                "the WS chat exchange never landed in chat.jsonl"
            )
        finally:
            server.stop()


@pytest.mark.integration
@pytest.mark.serial
def test_s2_commit_reviewed_triad_and_scope_pass_on_doc_only_diff(e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s2")
    with ScriptedStubModel(S2_SCRIPT) as stub:
        settings = keyless_settings(
            stub,
            # The review organ needs the self-modification surface: advanced runtime
            # (light restricts repo writes), BLOCKING enforcement so the landed commit
            # PROVES the organ passed rather than being waved through.
            OUROBOROS_RUNTIME_MODE="advanced",
            OUROBOROS_REVIEW_ENFORCEMENT="blocking",
        )
        server = start_server(e2e_clone, root, settings)
        try:
            task_id = submit_running(
                server,
                "Write the smoke note and land it through commit_reviewed, then finish.",
            )
            result = server.wait_task(task_id, timeout=600)
            assert result.get("status") == "completed", result

            oracle = ArtifactOracle(server.data_root)
            stored = wait_durable_result(oracle, task_id)
            assert stored.get("status") == "completed", stored

            # The review organ ran: the stub answered a triad packet AND a scope packet.
            kinds = stub.kinds()
            assert "triad_review" in kinds, kinds
            assert "scope_review" in kinds, kinds

            # The commit LANDED in the isolated clone — under blocking enforcement this
            # is only reachable through PASS verdicts from both organs.
            log_output = subprocess.run(
                ["git", "log", "-n", "5", "--format=%s"],
                cwd=str(e2e_clone), check=True, capture_output=True, text=True,
            ).stdout
            assert S2_COMMIT_MESSAGE in log_output, log_output
            committed_doc = subprocess.run(
                ["git", "show", f"HEAD:{S2_DOC_PATH}"],
                cwd=str(e2e_clone), check=False, capture_output=True, text=True,
            )
            assert committed_doc.returncode == 0, "smoke doc is not in the committed tree"

            # Durable review evidence lives in the task's FORKED drive root
            # (state/headless_tasks/<id>/data — headless-task isolation on this tree):
            # the audited advisory bypass and the scope round.
            task_oracle = oracle.task_drive(task_id)
            assert task_oracle.data_root != oracle.data_root, (
                "task drive root missing — headless drive layout changed?")
            runs = task_oracle.advisory_review().get("advisory_runs") or []
            bypassed = [r for r in runs if isinstance(r, dict) and r.get("status") == "bypassed"]
            assert bypassed, f"no bypassed advisory run in the task ledger: {runs!r}"
            assert bypassed[0].get("commit_message") == S2_COMMIT_MESSAGE, bypassed[0]
            assert task_oracle.events("advisory_review_bypassed"), "bypass event missing"
            assert task_oracle.events("scope_review_complete"), "scope completion event missing"
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# S3 — egress hardening (plan §8: "дыра ANTHROPIC_API_KEY — закрыть в Ф4 первой").
# The default-lane F21 tests pin the strip LIST; this scenario pins the LIVE FACT:
# a poisoned parent environment, a REAL server tree, and a /proc environ probe of
# every process in it.
# ---------------------------------------------------------------------------

S3_PLANTED = {
    "ANTHROPIC_API_KEY": "sk-ant-e2e-planted-must-not-leak-0001",
    "OPENROUTER_API_KEY": "sk-or-e2e-planted-must-not-leak-0001",
    "OPENAI_API_KEY": "sk-e2e-planted-must-not-leak-0001",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY": "e2e-planted-must-not-leak-0001",
    "GIGACHAT_CREDENTIALS": "e2e-planted-must-not-leak-0001",
    "GITHUB_TOKEN": "ghp-e2e-planted-must-not-leak-0001",
}


@pytest.mark.integration
@pytest.mark.serial
@pytest.mark.skipif(sys.platform != "linux", reason="/proc environ probe is Linux-only")
def test_s3_poisoned_parent_credentials_never_reach_the_server_tree(
        e2e_clone, tmp_path_factory, monkeypatch):
    require_lane(LANE_MOCK)
    for key, value in S3_PLANTED.items():
        monkeypatch.setenv(key, value)
    # Everything that must be invisible to the child tree: the planted fakes PLUS
    # every real credential-shaped value the operator shell happens to carry.
    must_not_appear = set(S3_PLANTED.values()) | set(secret_values_in_parent_env().values())
    from ouroboros.config import normalize_settings_raw
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    root = tmp_path_factory.mktemp("s3")
    with ScriptedStubModel(S1_SCRIPT) as stub:
        settings = keyless_settings(stub)
        # What the server legitimately projects into its children: the boot pipeline
        # (server.py lifespan) is load_settings — normalize_settings_raw over the file,
        # merged over SETTINGS_DEFAULTS — then apply_runtime_provider_defaults, then
        # apply_settings_to_env exporting every non-empty settings_env_keys() value.
        # The same pure seams here, so the name pin below is equality against the
        # projection, not a hand list of what the file happens to name.
        effective, _changed, _keys = apply_runtime_provider_defaults(
            {**SETTINGS_DEFAULTS, **normalize_settings_raw(settings)})
        projected = {
            key: str(effective[key])
            for key in settings_env_keys()
            if key in STRIPPED_PROVIDER_ENV_KEYS and effective.get(key) not in (None, "")
        }
        server = start_server(e2e_clone, root, settings)
        try:
            # The server BOOTED and WORKS with the poisoned parent env: one scripted
            # task runs to durable completion, keyless.
            task_id = submit_running(server, "List the repository root and finish.")
            result = server.wait_task(task_id, timeout=300)
            assert result.get("status") == "completed", result
            wait_durable_result(ArtifactOracle(server.data_root), task_id)

            # /proc probe of the WHOLE live server tree (server + supervisor workers):
            # no planted or real credential VALUE is visible in any child environ —
            # values, not names, so a smuggling rename cannot hide one.
            pids = process_tree_pids(server.proc.pid)
            assert len(pids) >= 2, f"expected server + workers in the tree, saw {pids}"
            for pid in pids:
                try:
                    child_env = proc_environ(pid)
                except OSError:
                    continue  # a worker exited between the walk and the read
                child_values = set(child_env.values())
                leaked = sorted(
                    key for key, value in S3_PLANTED.items() if value in child_values
                ) + sorted(
                    {value for value in must_not_appear if value in child_values}
                    - set(S3_PLANTED.values())
                )
                assert not leaked, f"credential values visible in pid {pid} environ: {leaked}"
                # And by NAME: a stripped-family key may appear ONLY with exactly its
                # projected value — every credential key has an empty effective value
                # and so must be ABSENT; a family name carrying any other value is red.
                # /proc environ is the exec-time block: forkserver/spawn children are
                # exec'd from the server's projected os.environ; fork copied the
                # server's own scrubbed exec-time block and hid that projection.
                foreign = sorted(
                    key for key in STRIPPED_PROVIDER_ENV_KEYS & set(child_env)
                    if child_env[key] != projected.get(key)
                )
                assert not foreign, (
                    f"provider env keys in pid {pid} beyond the settings projection: {foreign}")
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# S4 — typed tools + safety: a protected-path write denial leaves ZERO side effects
# (tree snapshot before/after), while the task itself still completes.
# ---------------------------------------------------------------------------

S4_PROTECTED_PATH = "prompts/SAFETY.md"
S4_SCRIPT = [
    {"tool": "write_file", "arguments": {
        "root": "system_repo",
        "path": S4_PROTECTED_PATH,
        "content": "E2E S4: this write must be DENIED, not landed.\n",
    }},
]
S4_SNAPSHOT_PATHS = (S4_PROTECTED_PATH, "BIBLE.md")


@pytest.mark.integration
@pytest.mark.serial
def test_s4_protected_path_write_denial_has_zero_side_effects(e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s4")
    with ScriptedStubModel(S4_SCRIPT) as stub:
        settings = keyless_settings(
            stub,
            # advanced (not pro): the mode where the protected-path policy DENIES the
            # write; light would refuse repo writes wholesale and prove nothing about
            # the protected-surface guard specifically.
            OUROBOROS_RUNTIME_MODE="advanced",
        )
        server = start_server(e2e_clone, root, settings)
        try:
            before = repo_tree_fingerprint(e2e_clone, S4_SNAPSHOT_PATHS)
            task_id = submit_running(
                server, "Try to update the safety prompt file, then finish.")
            result = server.wait_task(task_id, timeout=300)
            # The DENIAL must not kill the task: the agent gets the typed refusal and
            # finishes normally.
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            wait_durable_result(oracle, task_id)
            assert stub.script_consumed(), "S4 script was not fully consumed"

            # The durable tool log carries the typed denial.
            task_oracle = oracle.task_drive(task_id)
            rows = [row for row in task_oracle.tools_rows()
                    if str(row.get("tool") or row.get("name") or "") == "write_file"]
            assert rows, "write_file call missing from the task tools log"
            denial = json.dumps(rows)
            assert "CORE_PROTECTION_BLOCKED" in denial, rows
            assert S4_PROTECTED_PATH in denial, rows

            # ZERO side effects: HEAD, porcelain status and the protected bytes are
            # IDENTICAL — not "still clean", identical.
            after = repo_tree_fingerprint(e2e_clone, S4_SNAPSHOT_PATHS)
            assert after == before, {"before": before, "after": after}
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# S5 — cost-truth smoke (ABI-3 on a LIVE server): a completed task's PUBLIC
# projections carry honest-only cost names; the retired aliases appear nowhere in
# the outbound bytes.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.serial
def test_s5_public_task_projections_carry_honest_only_cost_names(e2e_clone, tmp_path_factory):
    require_lane(LANE_MOCK)
    root = tmp_path_factory.mktemp("s5")
    with ScriptedStubModel(S1_SCRIPT) as stub:
        server = start_server(e2e_clone, root, keyless_settings(stub))
        try:
            task_id = submit_running(server, "List the repository root and finish.")
            result = server.wait_task(task_id, timeout=300)
            assert result.get("status") == "completed", result
            oracle = ArtifactOracle(server.data_root)
            stored = wait_durable_result(oracle, task_id)

            # The task DETAIL projection (the same GET the UI drives): deep-scanned —
            # no retired alias key anywhere in the outbound bytes, and the honest
            # top-level name is present (None is honest; a fabricated $0 is not).
            from devtools.benchmarks.common.server_runner import _api

            detail = _api(server.base_url, "GET", f"/api/tasks/{task_id}", timeout=30)
            assert retired_cost_alias_paths(detail) == [], detail
            assert "accounted_upper_bound_usd" in detail, sorted(detail)

            # The task LIST projection: every row deep-clean.
            listing = _api(server.base_url, "GET", "/api/tasks", timeout=30)
            assert retired_cost_alias_paths(listing) == [], listing

            # The durable stored row: the write seam normalizes the TOP LEVEL and the
            # known public planes to honest-only (internal evidence planes keep their
            # own schemas by design, so the stored-row pin stays top-level).
            top_level_aliases = [
                path for path in retired_cost_alias_paths(stored)
                if path.count(".") == 1  # "$.<key>" only
            ]
            assert top_level_aliases == [], stored
        finally:
            server.stop()
