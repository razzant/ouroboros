"""WS5 — v6.33.0 review carryover fixes (v6.34.0)."""

from __future__ import annotations



# --- CW1 lineage: owner-only settings stay out of the generic settings write.      ---
# --- ABI 7.0 (owner Q10=A): OUROBOROS_SCOPE_REVIEW_FLOOR and its whole surface are  ---
# --- REMOVED (see tests/test_abi5_q10_removals.py); scope-review applicability      ---
# --- comes solely from the owner context mode.                                      ---

def test_context_mode_is_owner_only_not_generic_settings():
    from ouroboros.gateway.settings import _merge_settings_payload

    current = {"OUROBOROS_CONTEXT_MODE": "max"}
    merged = _merge_settings_payload(current, {"OUROBOROS_CONTEXT_MODE": "low"})
    # The generic /api/settings merge must NOT narrow the horizon — and since v6.80.0
    # the same setting decides whether the blocking scope review applies at all.
    assert merged["OUROBOROS_CONTEXT_MODE"] == "max"


# --- CW1: the owner-control mention family and its shared read-carve ---

def test_read_exemption_is_option_aware_not_head_only():
    """Review round 2: an allowlisted HEAD is not evidence that the command only reads.

    Several allowed heads execute or mutate through their own options (`find -exec`,
    `-delete`, `rg --pre`, `sort -o`, git's external-diff/textconv helpers), and the
    environment prefix decides what runs at all (`PATH=`, `LD_PRELOAD=`,
    `GIT_EXTERNAL_DIFF=`) — dropping it, as the first version did, let a command headed by
    a read token reach the owner-only endpoint. Membership is now necessary, not
    sufficient: options are validated per command, assignments are refused rather than
    stripped, and the executable must resolve to a bare name or a system bin.
    """
    from ouroboros.tools.registry import _detect_safety_mode_self_lowering as det
    from ouroboros.tools.shell_guards import shell_has_write_indicator

    def verdict(cmd: str) -> bool:
        return det(cmd.lower(), writeish=shell_has_write_indicator(cmd))

    # find: execution and deletion under a read head.
    assert verdict(
        "find ouroboros -name '*.py' -exec curl "
        "http://127.0.0.1:8765/api/owner/safety-mode ;"
    ) is True
    assert verdict("find . -name settings.json -delete # ouroboros_safety_mode") is True
    assert verdict(
        "find . -name '*.json' -fprintf /tmp/x '%p' # ouroboros_safety_mode "
        "data/settings.json"
    ) is True
    assert verdict("fd -x sh -c 'curl http://127.0.0.1:8765/api/owner/safety-mode'") is True
    # git: an external diff / textconv helper is an arbitrary configured program.
    assert verdict("git diff --ext-diff data/settings.json # ouroboros_safety_mode") is True
    assert verdict("git show --textconv head:data/settings.json # ouroboros_safety_mode") is True
    assert verdict("git grep -o /api/owner/safety-mode") is True  # -O opens a pager
    # Execution-affecting environment assignments are REFUSED, never discarded.
    assert verdict("git_external_diff=/tmp/x.sh git diff data/settings.json "
                   "# ouroboros_safety_mode") is True
    assert verdict("path=/tmp/evil grep ouroboros_safety_mode data/settings.json") is True
    assert verdict("ld_preload=/tmp/x.so cat data/settings.json | grep "
                   "ouroboros_safety_mode") is True
    assert verdict("env git_config_global=/tmp/g git log -1 -- data/settings.json "
                   "# ouroboros_safety_mode") is True
    assert verdict("env -i grep ouroboros_safety_mode data/settings.json") is True
    # Executable shadowing: an absolute path outside the system bins, or a relative one.
    assert verdict("/tmp/evil/grep ouroboros_safety_mode data/settings.json") is True
    assert verdict("./grep ouroboros_safety_mode data/settings.json") is True
    assert verdict("../bin/rg /api/owner/safety-mode ouroboros/") is True
    # Other allowlisted heads that write or execute through an option.
    assert verdict("sort -o /tmp/out data/settings.json # ouroboros_safety_mode") is True
    assert verdict("rg --pre /tmp/evil.sh /api/owner/safety-mode ouroboros/") is True

    # The exemption itself SURVIVES: legitimate inspection of the same surface is allowed,
    # including a trusted absolute path, a benign option and a read-only pipeline.
    assert verdict("/usr/bin/grep ouroboros_safety_mode data/settings.json") is False
    assert verdict("find ouroboros -name '*.py' -newer /api/owner/safety-mode") is False
    assert verdict("rg -n --no-heading '/api/owner/safety-mode' ouroboros/ | sort") is False
    assert verdict("git diff --stat -- data/settings.json # ouroboros_safety_mode") is False
    assert verdict("git grep -n /api/owner/safety-mode") is False

    # Pin the MECHANISM, not just the verdict: the classifier itself must refuse these,
    # so a future change to the write-shape fact cannot silently mask the exemption hole.
    from ouroboros.tools.registry import _is_pure_read_inspection as pure

    for hostile in (
        "find . -name '*.py' -exec sh -c ':' ;",
        "git diff --ext-diff data/settings.json",
        "path=/tmp/evil grep floor data/settings.json",
        "/tmp/evil/grep floor data/settings.json",
        "sort -o /tmp/out data/settings.json",
    ):
        assert pure(hostile) is False, hostile
    assert pure("/usr/bin/grep floor data/settings.json") is True


def test_stored_singular_scope_pin_is_ghost_purged(monkeypatch, tmp_path):
    """ABI 7.0 (ABI-10): both comma spellings are RETIRED settings keys — a
    stored pin (singular or plural) is ghost-purged on load, never promoted.
    (The pre-7.0 singular→plural promotion left with the migration read.)"""
    import json

    import ouroboros.config as cfg

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"OUROBOROS_SCOPE_REVIEW_MODEL": "anthropic/claude-opus-4.8"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.delenv("OUROBOROS_SCOPE_REVIEW_MODEL", raising=False)
    monkeypatch.delenv("OUROBOROS_SCOPE_REVIEW_MODELS", raising=False)

    loaded = cfg.load_settings()
    assert "OUROBOROS_SCOPE_REVIEW_MODEL" not in loaded
    assert "OUROBOROS_SCOPE_REVIEW_MODELS" not in loaded


# --- CW3: an ephemeral decision turn is barred from durable mutators ---

def test_ephemeral_turn_blocks_durable_mutators(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    reg = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    reg.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, is_ephemeral_turn=True))

    out = reg.execute("update_identity", {"content": "x"})
    assert "EPHEMERAL_TURN_RESTRICTED" in out  # failed closed, not executed

    names = {(s.get("function") or {}).get("name") or s.get("name") for s in reg.schemas()}
    assert "update_identity" not in names and "knowledge_write" not in names
    assert "toggle_evolution" not in names and "set_tool_timeout" not in names
    # The decision/answer/steer tools remain available to the ephemeral turn.
    assert "steer_task" in names and "promote_chat_to_task" in names


def test_non_ephemeral_turn_allows_durable_mutators(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    reg = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    reg.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, is_ephemeral_turn=False))
    names = {(s.get("function") or {}).get("name") or s.get("name") for s in reg.schemas()}
    assert "update_identity" in names  # a normal turn sees the durable mutators
    out = reg.execute("update_identity", {})
    assert "EPHEMERAL_TURN_RESTRICTED" not in out  # the ephemeral gate did not fire


# --- CW4: the external-shell secret guard catches relative interpreter paths ---

def test_secret_guard_catches_relative_interpreter_path(tmp_path):
    from types import SimpleNamespace
    from ouroboros.tools.registry import _subagent_shell_targets_secret

    data = tmp_path / "data"
    ctx = SimpleNamespace(drive_root=data, task_metadata={})
    assert _subagent_shell_targets_secret(["python", "-c", "open('data/settings.json')"], ctx=ctx, cwd=tmp_path)
    assert _subagent_shell_targets_secret(["node", "-e", "readFileSync('../../data/settings.json')"], ctx=ctx, cwd=tmp_path / "a" / "b")
    assert _subagent_shell_targets_secret("cat ~/.ssh/id_rsa")
    assert not _subagent_shell_targets_secret("cat /tmp/notes.txt")


# --- Exact-route context fitting honours USE_LOCAL_MAIN ---

def test_active_main_route_honours_use_local_main():
    from ouroboros.gateway.settings import _active_main_route

    local = _active_main_route({"OUROBOROS_MODEL": "openai/gpt-5.5", "USE_LOCAL_MAIN": True})
    assert local["use_local"] is True and local["provider"] == "local"
    remote = _active_main_route({"OUROBOROS_MODEL": "openai/gpt-5.5", "USE_LOCAL_MAIN": False})
    assert remote["use_local"] is False and remote["provider"] != "local"


# --- CW9: the pacing-interval timeout constant lives in the SETTINGS_DEFAULTS SSOT ---

def test_pacing_interval_in_settings_defaults():
    from ouroboros.config import PACING_INTERVAL_DEFAULT_SEC, SETTINGS_DEFAULTS

    assert SETTINGS_DEFAULTS.get("OUROBOROS_PACING_INTERVAL_SEC") == PACING_INTERVAL_DEFAULT_SEC


# === Triad+scope review-fix regressions (v6.34.0) ===

# Predicted route evidence is measurement input, never a global-mode writer or
# an initial Max-to-Low authority. Functional fit cases are pinned in the Phase
# 2 matrix; this carryover suite guards deletion of the old compatibility seam.
def test_predicted_route_downgrade_seam_stays_deleted():
    from ouroboros import loop as loopmod

    assert not hasattr(loopmod, "_maybe_downgrade_max_unconfirmed")


# --- CW3: the ephemeral deny surface is complete (core envelope + non-core mutators) ---
# --- CW3: the ephemeral deny surface is complete (core envelope + non-core mutators) ---

def test_ephemeral_allowlist_excludes_every_mutator_class():
    from ouroboros.tools.registry import _EPHEMERAL_ALLOWED_TOOLS, _REPO_MUTATION_TOOLS

    # CW3 default-deny: no durable repo/git mutator is in the allowlist...
    assert not (_REPO_MUTATION_TOOLS & _EPHEMERAL_ALLOWED_TOOLS)
    # ...nor any review/skill/publish/control mutator (the whack-a-mole denylist kept
    # missing these), nor run_command (shell is durable-capable).
    for name in ("fetch_pr_ref", "create_integration_branch", "advisory_review", "skill_review",
                 "submit_skill_to_hub", "skill_exec", "toggle_skill", "cancel_task",
                 "task_acceptance_review", "run_command", "switch_model", "update_identity",
                 "commit_reviewed", "toggle_evolution",
                 # subagent-only tools must NOT leak in: spawn / blocking-wait / page-interaction
                 "schedule_subagent", "wait_task", "wait_tasks", "browser_action"):
        assert name not in _EPHEMERAL_ALLOWED_TOOLS
    # ...while the read/inspect + decision tools ARE allowed.
    for name in ("read_file", "query_code", "search_code", "web_search",
                 "route_to_project", "promote_chat_to_task", "steer_task"):
        assert name in _EPHEMERAL_ALLOWED_TOOLS


def test_ephemeral_core_envelope_is_allowlisted_and_mutators_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    from ouroboros.tools.registry import ToolContext, ToolRegistry, _EPHEMERAL_ALLOWED_TOOLS

    reg = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    reg.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, is_ephemeral_turn=True))

    # The CORE/initial envelope is allowlisted too (every visible tool is allowed).
    core_names = {(s.get("function") or {}).get("name") or s.get("name") for s in reg.schemas(core_only=True)}
    assert core_names <= _EPHEMERAL_ALLOWED_TOOLS

    # A non-allowlisted mutator fails closed at execute() up front (so enabling it via
    # enable_tools cannot bypass the gate), and get_schema_by_name won't surface it.
    assert "EPHEMERAL_TURN_RESTRICTED" in reg.execute("fetch_pr_ref", {})
    assert "EPHEMERAL_TURN_RESTRICTED" in reg.execute("advisory_review", {})
    assert reg.get_schema_by_name("skill_review") is None  # enable_tools can't surface it


def test_switch_model_does_not_blanket_gate_on_context_window(monkeypatch, tmp_path):
    """The loop rebinds/fits the exact route after the override; the tool only selects it."""
    from ouroboros.tools import control
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setattr(
        "ouroboros.llm.LLMClient.available_models",
        lambda self: ["small-model"],
    )
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.active_context_mode = "max"

    out = control._switch_model(ctx, model="small-model")

    assert "SWITCH_BLOCKED" not in out
    assert ctx.active_model_override == "small-model"

# --- CW3 (claudexor): the ephemeral turn is barred from extension/MCP tools too ---

def test_ephemeral_blocks_extension_and_mcp_tools(tmp_path):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    reg = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    reg.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, is_ephemeral_turn=True))
    # an extension tool (resolved ext_tool) and an MCP tool both fail closed at execute()
    from ouroboros.tools.registry_guards import _ephemeral_block_result
    assert "EPHEMERAL_TURN_RESTRICTED" in _ephemeral_block_result(
        reg._ctx, "skill__do", ext_tool={"name": "skill__do"}
    ).text
    assert "EPHEMERAL_TURN_RESTRICTED" in _ephemeral_block_result(
        reg._ctx, "mcp__srv__x", is_mcp=True
    ).text
    # a normal turn does not block external tools
    reg.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, is_ephemeral_turn=False))
    assert _ephemeral_block_result(
        reg._ctx, "skill__do", ext_tool={"name": "skill__do"}
    ) is None


def test_ephemeral_schemas_omit_extension_and_mcp_surfaces(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    reg = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    reg.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, is_ephemeral_turn=True))
    reg.schemas()  # populates capability_omissions
    omissions = {(o.get("surface"), o.get("reason")) for o in reg.capability_omissions()}
    assert ("extensions", "ephemeral_turn") in omissions
    assert ("mcp", "ephemeral_turn") in omissions


# --- running_tasks routing context never silently truncates (codex no-[:N] rule) ---

def test_running_tasks_clip_marker_is_explicit():
    import server

    assert server._clip_marked("short objective", 600) == "short objective"
    clipped = server._clip_marked("x" * 1000, 600)
    assert clipped.startswith("x" * 600)
    assert "chars omitted]" in clipped  # explicit omission marker, not a silent cut


def test_settings_save_probes_review_slots_with_the_needs_ack_contract(monkeypatch, tmp_path):
    """RS1: a PINNED scope reviewer must have a REACHABLE path to Capability Evidence.

    The Max gate only ever probed the MAIN route, so a pin could never become "known"
    and silently ran in the conservative sub-floor window. The save-time probe reuses
    the EXISTING needs_ack:{route, route_fp, evidence} contract, which Settings RENDERS
    through the same confirm -> owner-capability-ack flow; it is advisory and never
    rewrites the pin. Only the scope surface is probed (its >=1M evidence is the only
    one that gates anything) and only on a scope-slot change."""
    import pathlib
    from types import SimpleNamespace

    import ouroboros.config as cfg
    from ouroboros.gateway import settings as smod

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "get_scope_review_models", lambda: ["anthropic/claude-opus-4.8"])
    monkeypatch.setattr(cfg, "get_review_models", lambda: ["openai/gpt-5.6-sol"])
    seen = []

    def fake_probe(drive_root, **kwargs):
        seen.append(kwargs["model"])
        return SimpleNamespace(
            window_tokens=200_000, status="confirmed", route_fp="fp",
            to_json=lambda: {"window_tokens": 200_000, "status": "confirmed"},
        )

    monkeypatch.setattr("ouroboros.capability_evidence.probe", fake_probe)

    notices = smod._review_capability_notices({})

    assert "anthropic/claude-opus-4.8" in seen, "the scope slot's own route must be probed"
    assert len(notices) == 1
    notice = notices[0]
    assert notice["surface"] == "scope_review"
    assert set(notice["needs_ack"]) >= {"provider", "model", "base_url", "route_fp", "evidence"}
    assert notice["window_tokens"] == 200_000
    assert notice["verified"] is True
    assert "openai/gpt-5.6-sol" not in seen, (
        "the triad surface never yields a notice, so probing it was network work on "
        "every settings save whose result was discarded"
    )

    # The response key must have a CONSUMER: unrendered, the owner sees no prompt and
    # every commit keeps blocking with SCOPE_REVIEW_SUB_FLOOR telling them to owner-ack
    # a route the UI never offered.
    repo = pathlib.Path(__file__).resolve().parent.parent
    settings_js = (repo / "web" / "modules" / "settings.js").read_text(encoding="utf-8")
    for needle in (
        "data.review_capability_notices",
        "ackReviewCapabilityNotices",
        "apiClient.ownerCapabilityAck(",
    ):
        assert needle in settings_js, needle
    gateway_src = (repo / "ouroboros" / "gateway" / "settings.py").read_text(encoding="utf-8")
    # The gate grew the 6.1 slots SSOT key (a slot change IS a route change) and
    # wrapped; the pinned semantics is unchanged — probe only on ROUTE-affecting
    # keys, never on every save.
    assert (
        'k.startswith("OUROBOROS_SCOPE_REVIEW_MODEL") or k == "OUROBOROS_REVIEWER_SLOTS"\n'
        '            or k in _REVIEW_ROUTE_BASE_URL_KEYS'
        in gateway_src
    ), "the probe must be gated on a ROUTE-affecting change, not run on every save"


def test_capability_evidence_is_route_aware_not_model_aware(monkeypatch, tmp_path):
    """A base-URL change is a NEW ROUTE and must reprobe + renotify.

    Capability is a property of provider+base_url+model, and evidence is stored under
    that route fingerprint. The lazy scope probe memoised by MODEL NAME and the
    save-time notice fired only on `OUROBOROS_SCOPE_REVIEW_MODEL*`, so hot-changing
    `OPENAI_BASE_URL` (or the openai-compatible / cloudru / gigachat equivalents) with an
    unchanged model produced a route with no evidence, no second probe and no notice —
    the next scope review fell silently to the conservative sub-floor and the advertised
    owner-ack path was unreachable."""

    import ouroboros.config as cfg
    from ouroboros import capability_evidence as ce
    from ouroboros.gateway import settings as smod
    from ouroboros.tools import scope_review as sr

    model = "openai::gpt-5.5-pinned"
    base_urls = {"OPENAI_BASE_URL": "https://route-a.example/v1"}
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "load_settings", lambda: dict(base_urls))

    fetched: list = []

    # The REAL probe, so what is counted is real network work: a route whose stored
    # record is current is served from the cache, which is the rate limit — not a
    # process memo that outlives the record and can never re-source it.
    def fake_metadata(_provider, _model, base_url, allow_fetch, **_kw):
        fetched.append(str(base_url or ""))
        return 200_000

    monkeypatch.setattr(ce, "_provider_metadata_window", fake_metadata)

    sr._scope_window(model)
    sr._scope_window(model)
    assert fetched == ["https://route-a.example/v1"], "one probe per route, not per call"

    # Same model, DIFFERENT base URL: a new route, so the lazy probe must run again.
    base_urls["OPENAI_BASE_URL"] = "https://route-b.example/v1"
    sr._scope_window(model)
    assert fetched == [
        "https://route-a.example/v1", "https://route-b.example/v1",
    ], "a base-URL change is a new route fingerprint and must be probed"

    # ...and the save-time owner-facing notice describes the INCOMING candidate route,
    # taken from the submitted settings rather than from process env.
    monkeypatch.setattr(cfg, "get_scope_review_models", lambda: ["anthropic/claude-fable-5"])
    # ABI-10: the incoming candidate route arrives via the structured key.
    import json as _json
    notices = smod._review_capability_notices({
        "OUROBOROS_REVIEWER_SLOTS": _json.dumps({
            "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": model}}],
            "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": model}}],
        }),
        "OPENAI_BASE_URL": "https://route-b.example/v1",
    })
    assert len(notices) == 1
    assert notices[0]["needs_ack"]["model"] == model
    assert notices[0]["needs_ack"]["base_url"] == "https://route-b.example/v1"


def test_scope_capability_notice_is_offered_on_a_route_whose_record_expired(
    monkeypatch, tmp_path,
):
    """The `require_fresh=True` on the save-time notice is REACHABLE, and it is what
    puts the owner-ack in front of the owner.

    Route fingerprints are content-addressed, so re-selecting a slot the install has
    used before finds that route's PRIOR record — which may have expired meanwhile.
    With the provider unreachable `probe` keeps that record (module invariant) and
    marks it stale, and a stale 1M record is exactly the shape that reads as `confirmed
    1M` yet cannot authorize the commit-time gate. Without the freshness argument the
    save reports the slot as fine and the next commit blocks with SCOPE_REVIEW_SUB_FLOOR
    pointing at an ack the UI never offered."""
    import datetime
    import json
    import pathlib

    import ouroboros.config as cfg
    from ouroboros import capability_evidence as ce
    from ouroboros.gateway import settings as smod

    model = "openai/gpt-5.6-terra"
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "get_scope_review_models", lambda: [model])

    route = smod._review_slot_route({}, model)
    fp = ce.route_fingerprint(provider=route["provider"], base_url=route["base_url"], model=model)
    expired = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=5)
    ).isoformat()
    store = tmp_path / "state" / "capability_evidence.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps({"probes": {fp: {
        "window_tokens": 1_000_000, "status": "confirmed", "source": "provider_metadata",
        "route_fp": fp, "model": model, "provider": route["provider"], "ts": expired,
    }}}), encoding="utf-8")

    # The provider cannot re-confirm it right now.
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: 0)
    monkeypatch.setattr(ce, "_metadata_fetch_transport_failed", lambda *a, **k: True)

    notices = smod._review_capability_notices({"OUROBOROS_SCOPE_REVIEW_MODELS": model})
    assert len(notices) == 1, "an expired, unverifiable record must still offer the ack"
    assert notices[0]["needs_ack"]["model"] == model
    assert notices[0]["needs_ack"]["evidence"]["stale"] is True
    assert notices[0]["window_tokens"] == 1_000_000, (
        "the number clears the floor — freshness, not size, is what withholds authority"
    )

    # And the owner-facing prompt says WHY a 1M reading is being questioned, instead of
    # asking them to confirm 1000000 tokens because the route reports 1000000 tokens.
    settings_js = (
        pathlib.Path(__file__).resolve().parent.parent / "web" / "modules" / "settings.js"
    ).read_text(encoding="utf-8")
    assert "evidence?.stale" in settings_js
    assert "EXPIRED reading the provider could not re-confirm" in settings_js


def test_unrecognised_review_model_ids_are_reported_loudly(monkeypatch):
    """RS5: a truncated slot value (the owner's `-5`) used to surface only as three
    waves of `400 ... is not a valid model ID`, destroying the review quorum. It is
    reported at save time — evidence-based (absent from a SUCCESSFULLY fetched
    catalog), never a guess, and never a save rejection."""
    from ouroboros.gateway import settings as smod
    from ouroboros.llm import LLMClient

    monkeypatch.setattr(LLMClient, "openrouter_context_length", classmethod(lambda cls, m, **k: 0))
    monkeypatch.setattr(LLMClient, "_CAPABILITIES_FETCH_OK", True, raising=False)
    monkeypatch.setattr(
        LLMClient, "_CONTEXT_LENGTH_CACHE",
        {"anthropic/claude-fable-5": 1_000_000}, raising=False,
    )

    unknown = smod._unrecognised_review_models(["anthropic/claude-fable-5", "-5"])
    assert unknown == ["-5"]

    # Without an authoritative catalog nothing may be CLAIMED unknown.
    monkeypatch.setattr(LLMClient, "_CAPABILITIES_FETCH_OK", False, raising=False)
    assert smod._unrecognised_review_models(["-5"]) == []


def test_read_exemption_fails_closed_on_nested_execution_constructs():
    """An allowlisted head may not smuggle a writer inside a command substitution.

    `_is_pure_read_inspection` checks the HEAD of each segment, so `echo "$(curl -X POST
    .../api/owner/safety-mode)"` presented `echo` — allowlisted — while the nested
    `curl` was never validated as a head at all, and the write-shape detector does not
    recognise an HTTP POST. The exemption was therefore granted to a command line whose whole
    purpose was to reach the owner-only endpoint. Nested EXECUTION is now refused outright
    (`$()`, backticks, process substitution, subshells) rather than enumerated: the writer
    inside it need not be a shape anybody listed.
    """
    from ouroboros.tools.registry import _detect_safety_mode_self_lowering as det
    from ouroboros.tools.registry import _is_pure_read_inspection as pure
    from ouroboros.tools.shell_guards import shell_has_write_indicator

    ep = "http://127.0.0.1:8765/api/owner/safety-mode"

    def verdict(cmd: str) -> bool:
        return det(cmd.lower(), writeish=shell_has_write_indicator(cmd))

    # Command substitution under three different allowlisted heads, quoted and bare.
    assert verdict(f'echo "$(curl -X POST {ep} -d mode=off)"') is True
    assert verdict(f'echo $(curl -X POST {ep} -d mode=off)') is True
    assert verdict(f'grep "$(curl -X POST {ep} -d mode=off)" data/settings.json') is True
    assert verdict(f'cat "$(curl -X POST {ep} -d mode=off)"') is True
    assert verdict(f'sort "$(curl -X POST {ep})"') is True
    # Backticks and process substitution are the same capability by another spelling.
    assert verdict(f'echo `curl -X POST {ep} -d mode=off`') is True
    assert verdict(f'cat <(curl -X POST {ep} -d mode=off)') is True
    assert verdict(f'grep floor <(curl -X POST {ep})') is True
    # A subshell is nested execution too.
    assert verdict(f'(curl -X POST {ep} -d mode=off)') is True

    # Pin the MECHANISM: the classifier must refuse these on its own, so the verdict cannot
    # come to depend on the write-shape fact noticing an HTTP POST (it does not).
    for hostile in (
        f'echo "$(curl -X POST {ep})"',
        f'grep "$(curl -X POST {ep})" data/settings.json',
        f'cat `curl -X POST {ep}`',
        f'cat <(curl -X POST {ep})',
    ):
        assert pure(hostile.lower()) is False, hostile
        assert shell_has_write_indicator(hostile) is False, (
            "precondition: the write-shape detector does not catch an HTTP POST, which is "
            "exactly why the read exemption has to fail closed by itself"
        )

    # The exemption SURVIVES for genuine inspection, including pipes between reads.
    assert verdict("grep ouroboros_safety_mode data/settings.json") is False
    assert verdict("cat data/settings.json | grep ouroboros_safety_mode") is False
    assert verdict(f"rg -n --no-heading '{ep}' ouroboros/ | sort") is False
    assert verdict("git grep -n /api/owner/safety-mode") is False
    assert verdict("/usr/bin/grep ouroboros_safety_mode data/settings.json") is False
    assert pure("grep ouroboros_safety_mode data/settings.json") is True
    assert pure("cat data/settings.json | grep floor") is True


def test_scope_capability_notice_fires_on_stale_evidence(monkeypatch, tmp_path):
    """The save-time notice and the review-time authority check are twins: both ask
    "can this slot supply a BLOCKING scope verdict". The notice used to accept an
    expired 1M record, so the owner was told the slot was fine and then blocked at
    commit time by the check that reads the same evidence with the freshness applied."""
    from types import SimpleNamespace

    import ouroboros.config as cfg
    from ouroboros.gateway import settings as smod
    from ouroboros.reviewer_window import ReviewerWindow

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "get_scope_review_models", lambda: ["anthropic/claude-fable-5"])

    evidence = {"stale": False}

    def fake_probe(drive_root, **kwargs):
        return SimpleNamespace(
            window_tokens=1_000_000, status="confirmed", route_fp="fp",
            stale=evidence["stale"], ts="",
            to_json=lambda: {"window_tokens": 1_000_000, "status": "confirmed"},
        )

    monkeypatch.setattr("ouroboros.capability_evidence.probe", fake_probe)

    assert smod._review_capability_notices({}) == [], "current 1M evidence needs no ack"

    evidence["stale"] = True
    notices = smod._review_capability_notices({})
    assert len(notices) == 1 and notices[0]["surface"] == "scope_review"
    # ...and the review-time twin agrees, which is the point of the shared predicate.
    assert ReviewerWindow(1_000_000, "confirmed", stale=True).blocking_authority_allowed is False
