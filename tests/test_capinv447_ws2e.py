"""WS2-e (capinv-447) contract tests: one-author projections + honest results.

Covers H2 (write-like typing across the two profile projections), H3 (module
load failures reach the capability ledger and survive schemas() rebuilds), H4
(typed preflight_blocked reason_kind), H5 (repo_commit_ready SSOT is
enforcement-aware), D3 (search filter receipt), D4 (export member policy reuses
the workspace-patch SSOT), G10 (attachment routes unified + rule-named
reasons), G3 (npm version pins + manual-dependency disclosure), and E5 (MCP
pagination, injective tool slugs, resource/structuredContent fidelity).
"""

from __future__ import annotations

import asyncio
import types
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# H2 — the two profile projections agree, and vcs alone is not write-like
# ---------------------------------------------------------------------------

def test_affordance_writable_matches_actual_mutation_authority():
    """Property over every profile+root: writable ⟺ a mutating operation is
    actually granted (decide_tool_access), and readonly/writable are disjoint."""
    from ouroboros.tool_access import (
        _POLICY,
        _WRITE_LIKE_OPS,
        decide_tool_access,
    )

    for profile, matrix in _POLICY.items():
        writable = {root for root, ops in matrix.items() if ops & _WRITE_LIKE_OPS}
        readonly = {root for root, ops in matrix.items() if ops and not (ops & _WRITE_LIKE_OPS)}
        assert not (writable & readonly), profile
        for root in matrix:
            has_mutation = any(
                decide_tool_access(profile=profile, root=root, operation=op).allow
                for op in _WRITE_LIKE_OPS
            )
            assert (root in writable) == has_mutation, (profile, root)


def test_readonly_profile_claims_no_writable_roots(tmp_path):
    """H2 regression: {read,list,search,vcs} must not project as writable."""
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tool_access import filesystem_affordance_map, summarize_subagent_profile

    ctx = SimpleNamespace(
        repo_dir=str(tmp_path / "repo"),
        drive_root=str(tmp_path / "data"),
        workspace_root="",
        workspace_mode="",
        task_constraint=TaskConstraint(mode="local_readonly_subagent"),
        task_metadata={},
    )
    affordance = filesystem_affordance_map(ctx)
    assert affordance["profile"] == "local_readonly_subagent"
    assert affordance["writable_roots"] == []
    assert "active_workspace" in affordance["readonly_roots"]
    assert "system_repo" in affordance["readonly_roots"]
    summary = summarize_subagent_profile("local_readonly_subagent")
    assert "writable=none (read-only)" in summary


def test_summary_write_roots_subset_of_affordance_writable():
    from ouroboros.tool_access import _POLICY, _WRITE_LIKE_OPS

    for profile, matrix in _POLICY.items():
        summary_write = {root for root, ops in matrix.items() if ops & {"write", "edit"}}
        affordance_write = {root for root, ops in matrix.items() if ops & _WRITE_LIKE_OPS}
        assert summary_write <= affordance_write, profile


# ---------------------------------------------------------------------------
# H3 — module load failure lands in the capability ledger and is not erased
# ---------------------------------------------------------------------------

def test_module_load_failure_recorded_and_survives_schema_rebuilds(tmp_path, monkeypatch):
    import pkgutil

    from ouroboros.tools.registry import ToolRegistry

    real_iter_modules = pkgutil.iter_modules

    def _with_broken(path=None, prefix=""):
        found = list(real_iter_modules(path, prefix))
        found.append((None, "zz_capinv447_broken_module", False))
        return found

    monkeypatch.setattr(pkgutil, "iter_modules", _with_broken)
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")

    def _load_failures():
        return [
            item for item in registry.capability_omissions()
            if item.get("reason") == "module_load_failed"
        ]

    failures = _load_failures()
    assert failures and failures[0]["module"] == "zz_capinv447_broken_module"
    assert failures[0]["surface"] == "tools"
    assert failures[0]["error"]
    # schemas() rebuild resets the per-call omissions but must keep load facts.
    registry.schemas()
    registry.schemas()
    assert _load_failures(), "schemas() rebuild erased the module-load omission"


# ---------------------------------------------------------------------------
# H4 — typed preflight_blocked reason_kind
# ---------------------------------------------------------------------------

def _guidance_for(reason_kind: str) -> str:
    from ouroboros.review_state import AdvisoryReviewState, AdvisoryRunRecord
    from ouroboros.tools.claude_advisory_review import _next_step_guidance

    latest = AdvisoryRunRecord(
        snapshot_hash="cafe" * 4,
        commit_message="m",
        status="preflight_blocked",
        ts="2026-09-01T00:00:00Z",
        raw_result="detail text",
        reason_kind=reason_kind,
    )
    return _next_step_guidance(
        latest=latest, state=AdvisoryReviewState(),
        stale_from_edit=False, stale_from_edit_ts=None,
        open_obs=[], open_debts=[], effective_is_fresh=False,
    )


def test_release_metadata_block_never_claims_syntax_error():
    guidance = _guidance_for("release_metadata")
    assert "SyntaxError" not in guidance
    assert "release metadata" in guidance


def test_untyped_preflight_block_stays_generic():
    guidance = _guidance_for("")
    assert "SyntaxError" not in guidance
    assert "raw_result" in guidance


def test_commit_gate_block_message_branches_on_reason_kind(tmp_path, monkeypatch):
    from ouroboros.review_state import AdvisoryRunRecord, compute_snapshot_hash, load_state, make_repo_key, save_state
    from ouroboros.tools.commit_gate import _check_advisory_freshness

    repo = tmp_path / "repo"
    repo.mkdir()
    ctx = SimpleNamespace(
        repo_dir=str(repo), drive_root=tmp_path,
        drive_logs=lambda: tmp_path / "logs", task_id="t1",
    )
    snapshot_hash = compute_snapshot_hash(repo, "msg", paths=None)
    state = load_state(tmp_path)
    state.add_run(AdvisoryRunRecord(
        snapshot_hash=snapshot_hash, commit_message="msg",
        status="preflight_blocked", ts="2026-09-01T00:00:00Z",
        raw_result="⚠️ PREFLIGHT_BLOCKED: VERSION is 1.0 but README says 0.9",
        reason_kind="release_metadata", repo_key=make_repo_key(repo),
    ))
    save_state(tmp_path, state)

    message = _check_advisory_freshness(ctx, "msg")
    assert message is not None
    assert "SyntaxError" not in message
    assert "release metadata preflight failed" in message


# ---------------------------------------------------------------------------
# H5 — repo_commit_ready SSOT mirrors advisory enforcement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("fresh", "debt", "enforcement", "expected"),
    [
        (False, False, "advisory", False),
        (False, False, "blocking", False),
        (True, False, "advisory", True),
        (True, False, "blocking", True),
        (True, True, "advisory", True),   # debt disclosed, not gating
        (True, True, "blocking", False),
    ],
)
def test_advisory_commit_ready_is_enforcement_aware(fresh, debt, enforcement, expected):
    from ouroboros.review_state import advisory_commit_ready

    debts = [object()] if debt else []
    assert advisory_commit_ready(fresh, [], debts, enforcement) is expected


def test_review_context_heading_no_longer_claims_full_gate():
    import inspect

    from ouroboros import agent_task_pipeline

    source = inspect.getsource(agent_task_pipeline)
    assert "Live repo gate" not in source
    assert "Advisory readiness" in source


# ---------------------------------------------------------------------------
# D3 — search filter receipt
# ---------------------------------------------------------------------------

def _search_ctx(tmp_path, repo):
    return types.SimpleNamespace(
        drive_root=str(tmp_path / "data"),
        repo_dir=str(repo),
        workspace_root="",
        workspace_mode="",
        task_metadata={},
    )


def test_search_no_matches_discloses_dropped_files(tmp_path, monkeypatch):
    import ouroboros.code_search_rg as rg_mod
    from ouroboros.tools import core as core_mod

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "big.txt").write_bytes(b"needle" + b"x" * (rg_mod.MAX_FILE_SIZE_BYTES + 1))

    def _no_rg(*a, **k):
        raise RuntimeError("force fallback")

    monkeypatch.setattr(rg_mod, "search_with_rg", _no_rg)
    result = core_mod._code_search(_search_ctx(tmp_path, repo), "needle", root="active_workspace", path=".")
    assert "No matches found" in result
    assert "were present but not searched" in result
    assert "oversized=1" in result


def test_search_clean_no_matches_stays_clean(tmp_path, monkeypatch):
    import ouroboros.code_search_rg as rg_mod
    from ouroboros.tools import core as core_mod

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "plain.txt").write_text("nothing here", encoding="utf-8")

    monkeypatch.setattr(rg_mod, "search_with_rg", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    result = core_mod._code_search(_search_ctx(tmp_path, repo), "needle", root="active_workspace", path=".")
    assert "No matches found" in result
    assert "were present but not searched" not in result


def test_rg_path_receipt_counts_skipped_files(tmp_path):
    import ouroboros.code_search_rg as rg_mod
    from ouroboros.tools import core as core_mod

    if not rg_mod._rg_binary():
        pytest.skip("rg not available")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "match.txt").write_text("needle", encoding="utf-8")
    (repo / "big.txt").write_bytes(b"needle" + b"x" * (rg_mod.MAX_FILE_SIZE_BYTES + 1))
    result = core_mod._code_search(_search_ctx(tmp_path, repo), "needle", root="active_workspace", path=".")
    assert "match.txt" in result
    assert "oversized=1" in result


def test_search_skip_reason_vocabulary(tmp_path):
    from ouroboros.code_search_rg import is_search_skippable, search_skip_reason

    regular = tmp_path / "a.txt"
    regular.write_text("x", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(regular)
    assert search_skip_reason(regular) == ""
    assert search_skip_reason(link) == "symlink"
    assert is_search_skippable(link) is True


# ---------------------------------------------------------------------------
# D4 — export policy reuses the workspace-patch SSOT, per-member semantics
# ---------------------------------------------------------------------------

def test_export_component_policy_matches_patch_policy():
    from ouroboros.tools.output_export_policy import _sensitive_output_component_reason

    assert _sensitive_output_component_reason((".gitignore",)) == ""
    assert _sensitive_output_component_reason((".github", "workflows", "ci.yml")) == ""
    assert _sensitive_output_component_reason((".env.example",)) == ""
    assert "dotenv secret" in _sensitive_output_component_reason((".env",))
    assert "credential filename" in _sensitive_output_component_reason(("keys", "id_rsa"))
    assert "private key" in _sensitive_output_component_reason(("server.pem",))


def test_single_declared_dotfile_output_is_exportable(tmp_path):
    from ouroboros.tools.output_export_policy import _protected_output_source_reason

    ctx = SimpleNamespace(repo_dir=str(tmp_path / "repo"), drive_root=str(tmp_path / "data"))
    project = tmp_path / "project"
    project.mkdir()
    gitignore = project / ".gitignore"
    gitignore.write_text("*.pyc\n", encoding="utf-8")
    assert _protected_output_source_reason(ctx, gitignore, "task_drive", set()) == ""
    dotenv = project / ".env"
    dotenv.write_text("TOKEN=x\n", encoding="utf-8")
    assert "credential-like output .env" in _protected_output_source_reason(ctx, dotenv, "task_drive", set())


def test_directory_scan_skips_members_with_receipts_instead_of_failing(tmp_path):
    from ouroboros.tools.output_export_policy import _scan_directory_output_members

    ctx = SimpleNamespace(repo_dir=str(tmp_path / "repo"), drive_root=str(tmp_path / "data"))
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>ok</h1>", encoding="utf-8")
    (site / ".gitignore").write_text("node_modules\n", encoding="utf-8")
    (site / ".env").write_text("TOKEN=x", encoding="utf-8")

    members, _size, block_reason, skipped = _scan_directory_output_members(
        ctx, site, label="task_drive", changed_paths=set(),
    )
    assert block_reason == ""
    names = {member.name for member in members}
    assert names == {"index.html", ".gitignore"}
    assert len(skipped) == 1 and ".env" in skipped[0]


# ---------------------------------------------------------------------------
# G10 — one attachment policy for both routes, rule-named reasons
# ---------------------------------------------------------------------------

def test_uploaded_secret_named_bytes_rejected_like_path_route(tmp_path, monkeypatch):
    import ouroboros.config as config
    from ouroboros.artifacts import stage_task_attachments

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    uploads = tmp_path / "data" / "uploads"
    uploads.mkdir(parents=True)
    secret = uploads / ("a" * 32 + "_.env")
    secret.write_text("TOKEN=x", encoding="utf-8")
    ordinary = uploads / ("b" * 32 + "_report.txt")
    ordinary.write_text("report", encoding="utf-8")

    manifest = stage_task_attachments(tmp_path / "drive", "task-1", [
        {"path": str(secret), "label": ".env"},
        {"path": str(ordinary), "label": "report.txt"},
    ])
    assert [row["status"] for row in manifest] == ["rejected", "staged"]
    assert manifest[0]["reason"] == "secret_source"
    assert "dotenv secret" in manifest[0]["rule"]


def test_upload_route_permits_env_example_and_ordinary_names(tmp_path, monkeypatch):
    import ouroboros.config as config
    from ouroboros.artifacts import stage_task_attachments

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    uploads = tmp_path / "data" / "uploads"
    uploads.mkdir(parents=True)
    example = uploads / ("c" * 32 + "_.env.example")
    example.write_text("TOKEN=fill-me", encoding="utf-8")

    manifest = stage_task_attachments(tmp_path / "drive", "task-2", [
        {"path": str(example), "label": ".env.example"},
    ])
    assert [row["status"] for row in manifest] == ["staged"]


def test_host_path_secret_rejection_names_the_rule(tmp_path):
    from ouroboros.artifacts import stage_task_attachments

    aws = tmp_path / ".aws"
    aws.mkdir()
    credentials = aws / "credentials"
    credentials.write_text("[default]", encoding="utf-8")

    manifest = stage_task_attachments(tmp_path / "drive", "task-3", [str(credentials)])
    assert manifest[0]["status"] == "rejected"
    assert manifest[0]["reason"] == "secret_source"
    assert ".aws" in manifest[0]["rule"]


# ---------------------------------------------------------------------------
# G3 — npm version pins are automatic; manual deps stay disclosed
# ---------------------------------------------------------------------------

def test_npm_version_pins_are_automatic():
    from ouroboros.marketplace.install_specs import normalize_install_specs

    auto, manual, warnings = normalize_install_specs([
        {"kind": "npm", "package": "axios@1.2.3"},
        {"kind": "npm", "package": "@scope/pkg@2.0.0"},
        {"kind": "npm", "package": "left-pad"},
    ])
    assert [spec["package"] for spec in auto] == ["axios@1.2.3", "@scope/pkg@2.0.0", "left-pad"]
    assert manual == [] and warnings == []


def test_manual_dependency_specs_are_not_dropped():
    from ouroboros.skill_dependencies import manual_install_specs_for_skill

    loaded = SimpleNamespace(manifest=SimpleNamespace(raw_extra={
        "install_specs": [
            {"kind": "pip", "package": "requests"},
            {"kind": "brew", "package": "ffmpeg"},
        ],
    }))
    manual, warnings = manual_install_specs_for_skill(loaded)
    assert len(manual) == 1 and manual[0]["package"] == "ffmpeg"
    assert warnings and "manual" in warnings[0]


def test_skill_readiness_discloses_manual_dependencies_without_blocking(tmp_path, monkeypatch):
    from ouroboros.skill_readiness import skill_readiness_for_execution

    skill = SimpleNamespace(
        name="demo",
        skill_dir=tmp_path / "skill",
        content_hash="h",
        load_error="",
        enabled=True,
        source="",
        review=SimpleNamespace(status="pass", is_stale_for=lambda _h: False),
        manifest=SimpleNamespace(raw_extra={"install_specs": [{"kind": "brew", "package": "ffmpeg"}]}),
    )
    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", lambda _root: [])
    monkeypatch.setattr("ouroboros.skill_loader.skill_conflict_status", lambda *_a, **_k: {})
    monkeypatch.setattr(
        "ouroboros.skill_loader.grant_status_for_skill", lambda *_a, **_k: {"all_granted": True}
    )
    readiness = skill_readiness_for_execution(tmp_path, skill)
    assert readiness.ready is True
    assert readiness.manual_dependencies == ["brew:ffmpeg"]


# ---------------------------------------------------------------------------
# E5 — MCP pagination, injective slugs, resource/structuredContent fidelity
# ---------------------------------------------------------------------------

def test_distinct_legal_mcp_tool_names_no_longer_collide():
    from ouroboros import mcp_client

    names = {mcp_client.make_tool_name("srv", raw) for raw in ("get-user", "get_user", "get.User", "get_User")}
    assert len(names) == 4
    # A clean lowercase name keeps its historical stable slug.
    assert mcp_client.make_tool_name("github", "search_repos") == "mcp_github__search_repos"
    for name in names:
        assert mcp_client.is_mcp_tool_name(name)


def test_list_tools_follows_next_cursor(monkeypatch):
    from ouroboros import mcp_client

    calls = []

    @asynccontextmanager
    async def fake_stdio(params):
        yield "read", "write"

    class Session:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def initialize(self):
            return None

        async def list_tools(self, cursor=None):
            calls.append(cursor)
            if cursor is None:
                return SimpleNamespace(
                    tools=[SimpleNamespace(name="one", description="", inputSchema={})],
                    nextCursor="page2",
                )
            return SimpleNamespace(
                tools=[SimpleNamespace(name="two", description="", inputSchema={})],
                nextCursor=None,
            )

    monkeypatch.setattr(mcp_client, "_MCP_SDK_AVAILABLE", True)
    monkeypatch.setattr(mcp_client, "StdioServerParameters", lambda *, command, args: None)
    monkeypatch.setattr(mcp_client, "stdio_client", fake_stdio)
    monkeypatch.setattr(mcp_client, "ClientSession", Session)

    cfg = mcp_client.normalize_server_config(
        {"id": "local", "transport": "stdio", "command": "python3", "args": []}
    )
    tools = asyncio.run(mcp_client._list_tools_async(cfg, timeout_sec=2))
    assert [tool["name"] for tool in tools] == ["one", "two"]
    assert calls == [None, "page2"]


def test_stringify_keeps_structured_content_alongside_text():
    from ouroboros.mcp_client import _stringify_call_result

    result = SimpleNamespace(
        content=[SimpleNamespace(text="prose answer")],
        structuredContent={"answer": 42},
        isError=False,
    )
    body = _stringify_call_result(result)
    assert "prose answer" in body
    assert '"answer": 42' in body


def test_serialize_content_part_carries_embedded_resource():
    from ouroboros.mcp_client import _serialize_content_part

    part = SimpleNamespace(
        type="resource",
        resource=SimpleNamespace(uri="file:///tmp/a.txt", mimeType="text/plain", text="hello"),
    )
    out = _serialize_content_part(part)
    assert out["resource"] == {"uri": "file:///tmp/a.txt", "mimeType": "text/plain", "text": "hello"}
