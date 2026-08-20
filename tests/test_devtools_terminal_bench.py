"""Terminal-Bench: the installed agent, its preflights and what it may carry into a container.

Split verbatim out of ``tests/test_devtools_benchmarks.py`` by theme. This module owns the
adapter's metadata and acceptance defaults, the provider and credit preflights it runs
before spending, the source copy that must leave secret-shaped files behind, and the
container environment that may never forward a fallback model or an injected secret.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import importlib.util
import json
import shlex
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest


from tests._devtools_benchmarks_shared import REPO_ROOT
from tests._devtools_benchmarks_shared import _isolate_bench_runs_root as __isolate_bench_runs_root

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_isolate_bench_runs_root = __isolate_bench_runs_root


def test_terminal_bench_harbor_adapter_is_optional_import():
    spec = importlib.util.spec_from_file_location(
        "tb_harbor_adapter",
        REPO_ROOT / "devtools" / "benchmarks" / "terminal_bench" / "harbor_installed_agent.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.OuroborosTerminalBenchAgent.name() == "Ouroboros Installed"

def test_terminal_bench_harbor_adapter_reads_canonical_version(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    monkeypatch.setattr(tb_agent, "_repo_root", lambda: tmp_path)
    (tmp_path / "VERSION").write_text("6.64.2\n", encoding="utf-8")
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path / "logs")

    assert agent.version() == "6.64.2"
    (tmp_path / "VERSION").unlink()
    assert agent.version() is None

def test_terminal_bench_harbor_context_uses_physical_metrics(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path, task_timeout_sec=900)
    monkeypatch.setattr(agent, "_container_env", lambda: {})
    monkeypatch.setattr(agent, "_enforce_container_secret_policy", lambda _env: None)
    monkeypatch.setattr(agent, "_openrouter_credit_preflight", lambda _settings: None)
    monkeypatch.setattr(agent, "_host_settings", lambda: {})

    async def _noop(*_args, **_kwargs):
        return None

    async def _run(*_args, **_kwargs):
        return {"cost_usd": 0.2, "prompt_tokens": 10, "completion_tokens": 5}

    async def _physical(*_args, **_kwargs):
        return {
            "cost_usd": 0.6,
            "prompt_tokens": 34,
            "completion_tokens": 14,
            "cached_tokens": 13,
            "cost_final": True,
            "accounting_authority": "physical_attempt_ledger",
        }

    for name in (
        "_network_preflight",
        "_resolve_workspace_dir",
        "_ensure_workspace_git_root",
        "_start_server",
        "_capture_current_task_summary",
        "_stop_server",
    ):
        monkeypatch.setattr(agent, name, _noop)
    monkeypatch.setattr(agent, "_run_ouroboros_task", _run)
    monkeypatch.setattr(agent, "_emit_trajectory", _physical)

    class Environment:
        async def upload_file(self, *_args, **_kwargs):
            return None

    context = SimpleNamespace(metadata={})
    asyncio.run(agent.run("Solve it", Environment(), context))

    assert context.cost_usd == 0.6
    assert context.n_input_tokens == 34
    assert context.n_output_tokens == 14
    assert context.n_cache_tokens == 13
    assert context.metadata["summary"]["cost_final"] is True

def test_terminal_bench_adapter_does_not_commit_target_workspace():
    adapter = (REPO_ROOT / "devtools" / "benchmarks" / "terminal_bench" / "harbor_installed_agent.py").read_text(encoding="utf-8")
    assert "git add -A" not in adapter
    assert "git commit --allow-empty" not in adapter

def test_terminal_bench_metadata_declares_all_assisting_models(monkeypatch):
    """NW-6: with task_review_mode=required the review triad (incl. a frontier
    model) assists the measured run; metadata.yaml must declare every assisting
    model, not only the measured one."""
    import sys as _sys
    spec = importlib.util.spec_from_file_location(
        "tb_run_for_meta", REPO_ROOT / "devtools" / "benchmarks" / "terminal_bench" / "run_tb.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(_sys.modules, spec.name, module)  # dataclass field resolution needs this
    spec.loader.exec_module(module)
    monkeypatch.delenv("OUROBOROS_REVIEW_MODELS", raising=False)
    # Both reviewer keys have to be owned, not just the triad one: the assertions below
    # read the SSOT defaults, while leaderboard_metadata reads the environment, and
    # ouroboros/reviewer_slot_config.py assigns OUROBOROS_SCOPE_REVIEW_MODELS through
    # os.environ directly — a write no fixture undoes, so any earlier test in the same
    # worker that reaches that code leaves this one comparing a leaked slot to the default.
    monkeypatch.delenv("OUROBOROS_SCOPE_REVIEW_MODELS", raising=False)
    monkeypatch.delenv("OUROBOROS_SCOPE_REVIEW_MODEL", raising=False)
    meta = module.leaderboard_metadata(
        agent_name="Ouroboros", org_name="Ouroboros",
        model="openai/gpt-5.5", light_model="google/gemini-3.5-flash")
    from ouroboros.config import SETTINGS_DEFAULTS

    # Every shipped default is read from the config SSOT and must be visible.
    for helper in SETTINGS_DEFAULTS["OUROBOROS_REVIEW_MODELS"].split(","):
        assert helper in meta
    assert SETTINGS_DEFAULTS["OUROBOROS_SCOPE_REVIEW_MODELS"] in meta
    assert "commit_review_triad" in meta
    assert meta.count("model_name:") >= 3

def test_terminal_bench_adapter_defaults_to_required_acceptance_review(tmp_path):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)
    env = agent._container_env()
    assert env["OUROBOROS_TASK_REVIEW_MODE"] == "required"
    assert env["OUROBOROS_MODEL_LIGHT"] == "google/gemini-3.5-flash"

    agent = tb_agent.OuroborosTerminalBenchAgent(
        logs_dir=tmp_path,
        task_review_mode="auto",
        ouroboros_model="openai/gpt-5.5",
        ouroboros_light_model="google/gemini-3.5-flash",
    )
    env = agent._container_env()
    assert env["OUROBOROS_TASK_REVIEW_MODE"] == "auto"
    assert env["OUROBOROS_MODEL"] == "openai/gpt-5.5"
    # v6.39 slot rename: the bulk lane is OUROBOROS_MODEL_HEAVY (legacy _CODE retired);
    # the container HEAVY lane reads os.environ["OUROBOROS_MODEL_HEAVY"], not _CODE.
    assert env["OUROBOROS_MODEL_HEAVY"] == "openai/gpt-5.5"
    assert env["OUROBOROS_MODEL_LIGHT"] == "google/gemini-3.5-flash"

def test_terminal_bench_source_copy_excludes_secret_shaped_files(tmp_path):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "module.py").write_text("print('ok')\n", encoding="utf-8")
    secret_names = (
        ".env",
        ".env.example",
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "aws-credentials.json",
        "credentials.json",
        "gcp-service-account.json",
        "id_rsa",
        "openrouter.token.txt",
        "prod.env",
        "repo.bundle",
        "repo_bundle_manifest.json",
        "secrets.json",
        "service-account.json",
    )
    for name in secret_names:
        (source / name).write_text("secret\n", encoding="utf-8")
    (source / "cert.pem").write_text("secret\n", encoding="utf-8")
    (source / "python-standalone").mkdir()
    (source / "python-standalone" / "python").write_text("binary\n", encoding="utf-8")

    tb_agent._copy_clean_source(source, target)

    assert (target / "module.py").exists()
    for name in (*secret_names, "cert.pem", "python-standalone"):
        assert not (target / name).exists()

def test_terminal_bench_source_provenance_hashes_copied_tree(tmp_path):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    source = tmp_path / "source"
    clean = tmp_path / "clean"
    source.mkdir()
    (source / "module.py").write_text("print('v1')\n", encoding="utf-8")
    (source / "untracked.txt").write_text("copied\n", encoding="utf-8")
    tb_agent._copy_clean_source(source, clean)

    provenance = tb_agent._source_copy_provenance(source, clean)

    assert provenance["copy_policy"]["secret_shaped_file_copy_allowed"] is False
    assert provenance["copied_tree"]["files"] == 2
    assert provenance["copied_tree"]["sha256"]

def test_terminal_bench_network_preflight_uses_configured_provider(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    class Env:
        def __init__(self) -> None:
            self.command = ""

        async def exec(self, *, command, timeout_sec=None, env=None, cwd=None):
            self.command = command
            script = command.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
            stdout = io.StringIO()
            code = 0
            try:
                with contextlib.redirect_stdout(stdout):
                    exec(script, {})
            except SystemExit as exc:
                code = int(exc.code or 0)
            return SimpleNamespace(return_code=code, stdout=stdout.getvalue(), stderr="")

    from types import SimpleNamespace

    env = Env()
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)

    asyncio.run(agent._network_preflight(env, {"OPENAI_API_KEY": "sk-test"}))

    assert "api.openai.com" in env.command
    assert "openrouter.ai" not in env.command
    assert "urllib.error.HTTPError" in env.command
    assert "openai_preflight_status 401" in (tmp_path / "network-preflight.txt").read_text(encoding="utf-8")

def test_terminal_bench_openrouter_credit_preflight_uses_authoritative_limit_remaining(tmp_path, monkeypatch):
    """v6.79.0: the preflight reads `/api/v1/key` `limit_remaining` through the shared helper.

    The old `/api/v1/credits` arithmetic (`total_credits − total_usage`) is the metric documented
    to lie on a nearly exhausted key, so this pins BOTH facts: the endpoint actually called, and
    that the credits-style body no longer decides anything."""
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    calls = []

    class _Response:
        def __init__(self, body):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self._body

    def fake_urlopen(req, timeout=0):
        assert req.headers["Authorization"] == "Bearer or-key"
        calls.append(req.full_url)
        # A body that the DEAD credits arithmetic would have read as $10 of headroom.
        return _Response(b'{"data":{"limit_remaining":0.25,"total_credits":10,"total_usage":0}}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path, openrouter_min_credit_usd=1.0)

    with pytest.raises(RuntimeError, match="remaining \\$0.25 below threshold \\$1.00"):
        agent._openrouter_credit_preflight({})

    assert calls == ["https://openrouter.ai/api/v1/key"]
    payload = json.loads((tmp_path / "openrouter-credit-preflight.json").read_text(encoding="utf-8"))
    assert payload["remaining_usd"] == 0.25
    assert payload["source"] == "openrouter:/api/v1/key:limit_remaining"

def test_terminal_bench_openrouter_preflight_admits_an_uncapped_key(tmp_path, monkeypatch):
    """`limit: null` means NO cap, not "$0 left" — an uncapped key must not be refused."""
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"data":{"limit":null,"usage":123.0}}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0: _Response())
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path, openrouter_min_credit_usd=1.0)

    agent._openrouter_credit_preflight({})

    payload = json.loads((tmp_path / "openrouter-credit-preflight.json").read_text(encoding="utf-8"))
    assert payload["ok"] is True and payload["uncapped"] is True and payload["remaining_usd"] is None

def test_run_ouroboros_task_terminal_nonzero_exit_is_not_interruption(tmp_path):
    """The in-container runner exits 2 to SIGNAL a terminal infra_failed result; that is a real
    terminal task outcome (status completed/failed), NOT a Harbor wall-clock interruption.
    _run_ouroboros_task must RETURN such a summary (so run() sets reached_terminal_result=True and
    the captured summary is not mislabeled captured_after_cancellation). A nonzero exit with NO
    terminal summary (a genuine runner crash) still raises."""
    import asyncio
    from types import SimpleNamespace
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)

    class _Env:
        def __init__(self, return_code, stdout):
            self._rc, self._out = return_code, stdout

        async def exec(self, *, command, timeout_sec=None, env=None, cwd=None):
            return SimpleNamespace(return_code=self._rc, stdout=self._out, stderr="")

    terminal = json.dumps(
        {"status": "failed", "reason_code": "provider_unavailable", "infra_failed": True, "return_code": 2}
    )
    out = asyncio.run(agent._run_ouroboros_task(_Env(2, terminal), {}))
    assert out["status"] == "failed" and out["reason_code"] == "provider_unavailable"

    with pytest.raises(RuntimeError):
        asyncio.run(agent._run_ouroboros_task(_Env(2, "Traceback: boom\nnot-json"), {}))

def test_terminal_bench_openrouter_credit_preflight_skips_when_unconfigured(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)

    agent._openrouter_credit_preflight({})

    assert not (tmp_path / "openrouter-credit-preflight.json").exists()

def test_terminal_bench_network_preflight_supports_openai_compatible(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    class Env:
        def __init__(self) -> None:
            self.command = ""

        async def exec(self, *, command, timeout_sec=None, env=None, cwd=None):
            self.command = command
            script = command.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
            stdout = io.StringIO()
            code = 0
            try:
                with contextlib.redirect_stdout(stdout):
                    exec(script, {})
            except SystemExit as exc:
                code = int(exc.code or 0)
            return SimpleNamespace(return_code=code, stdout=stdout.getvalue(), stderr="")

    env = Env()
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)

    asyncio.run(
        agent._network_preflight(
            env,
            {
                "OPENAI_COMPATIBLE_API_KEY": "sk-compatible",
                "OPENAI_COMPATIBLE_BASE_URL": "https://provider.example.invalid/v1",
            },
        )
    )

    assert "provider.example.invalid/v1/models" in env.command
    assert "openai_compatible_preflight_status 401" in (tmp_path / "network-preflight.txt").read_text(encoding="utf-8")

def test_terminal_bench_adapter_forwards_gigachat_and_preflights_direct_provider(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    monkeypatch.setenv("OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS", "1")
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "gigachat-test-credentials")
    monkeypatch.setenv("GIGACHAT_BASE_URL", "https://gigachat.example.invalid/api/v1")

    class Env:
        def __init__(self) -> None:
            self.command = ""

        async def exec(self, *, command, timeout_sec=None, env=None, cwd=None):
            self.command = command
            script = command.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
            stdout = io.StringIO()
            code = 0
            try:
                with contextlib.redirect_stdout(stdout):
                    exec(script, {})
            except SystemExit as exc:
                code = int(exc.code or 0)
            return SimpleNamespace(return_code=code, stdout=stdout.getvalue(), stderr="")

    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)
    injected = agent._container_env()
    env = Env()

    asyncio.run(agent._network_preflight(env, injected))

    assert injected["GIGACHAT_CREDENTIALS"] == "gigachat-test-credentials"
    assert "gigachat.example.invalid/api/v1/models" in env.command
    assert "gigachat_preflight_status 401" in (tmp_path / "network-preflight.txt").read_text(encoding="utf-8")

def test_terminal_bench_adapter_refuses_container_secret_injection_by_default(tmp_path, monkeypatch):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    monkeypatch.delenv("OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-container-secret")
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)
    injected = agent._container_env()

    assert "OPENROUTER_API_KEY" not in injected
    with pytest.raises(RuntimeError, match="refuses to inject long-lived provider credentials"):
        agent._enforce_container_secret_policy(injected)

def test_terminal_bench_task_body_uses_top_level_actor_id():
    adapter = (REPO_ROOT / "devtools" / "benchmarks" / "terminal_bench" / "harbor_installed_agent.py").read_text(encoding="utf-8")
    assert '"actor_id": "harbor-terminal-bench"' in adapter
    assert '"metadata": {{"source": "terminal-bench", "delegation_role": "root"}}' in adapter
    assert '"metadata": {{"actor_id": "harbor-terminal-bench"' not in adapter

def test_terminal_bench_adapter_quotes_hostile_workspace_dir(tmp_path):
    from devtools.benchmarks.terminal_bench.harbor_installed_agent import OuroborosTerminalBenchAgent

    class FakeResult:
        return_code = 0
        stdout = '{"return_code": 0}\n'
        stderr = ""

    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, **kwargs):
            self.calls.append(kwargs)
            return FakeResult()

    hostile = "/tmp/ws'; touch /tmp/pwn; echo '"
    agent = OuroborosTerminalBenchAgent(logs_dir=tmp_path, workspace_dir=hostile, task_timeout_sec=900)
    environment = FakeEnvironment()

    asyncio.run(agent._resolve_workspace_dir(environment))
    asyncio.run(agent._ensure_workspace_git_root(environment))
    summary = asyncio.run(agent._run_ouroboros_task(environment, {}))

    assert summary["return_code"] == 0
    quoted = shlex.quote(hostile)
    assert environment.calls[0]["command"] == f"test -d {quoted}"
    git_command = environment.calls[1]["command"]
    assert f"workspace_dir={quoted}" in git_command
    assert "cd \"$workspace_dir\"" in git_command
    runner_command = environment.calls[-1]["command"]
    runner = runner_command.split("cat > /tmp/run_ouroboros_task.py <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    assert f'"workspace_root": {json.dumps(hostile)}' in runner
    assert '"service_teardown": "keep"' in runner
    assert 'task_body["timeout_sec"] = task_timeout' in runner
    assert "task_timeout = 795" in runner  # 900 - _DEADLINE_SAFETY_SEC (105)
    compile(runner, "run_ouroboros_task.py", "exec")

def test_container_env_never_forwards_model_fallback(tmp_path, monkeypatch):
    """6b: the benchmark metric is single-model — a host-configured
    OUROBOROS_MODEL_FALLBACK must never leak into the container env."""
    import json as _json

    from devtools.benchmarks.terminal_bench.harbor_installed_agent import (
        OuroborosTerminalBenchAgent,
    )

    settings = tmp_path / "settings.json"
    settings.write_text(_json.dumps({
        "OUROBOROS_MODEL": "openai/gpt-5.5",
        "OUROBOROS_MODEL_FALLBACK": "google/gemini-3.5-flash",
    }), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACK", "google/gemini-3.5-flash")
    monkeypatch.setenv("OUROBOROS_MODEL", "openai/gpt-5.5")

    agent = OuroborosTerminalBenchAgent(
        logs_dir=tmp_path, model_name="test",
        host_settings_path=str(settings),
        ouroboros_model="openai/gpt-5.5",
    )
    env = agent._container_env()
    # The fallback is PINNED to the measured model (not absent: the container
    # has no settings.json, so absence would resurrect the SETTINGS_DEFAULTS
    # fallback — a different model — inside the container).
    assert env.get("OUROBOROS_MODEL_FALLBACK") == "openai/gpt-5.5"
    assert env.get("OUROBOROS_MODEL") == "openai/gpt-5.5"

    # No explicit kwarg: the pin follows the forwarded host main model.
    agent_no_kwarg = OuroborosTerminalBenchAgent(
        logs_dir=tmp_path, model_name="test",
        host_settings_path=str(settings),
    )
    env2 = agent_no_kwarg._container_env()
    assert env2.get("OUROBOROS_MODEL_FALLBACK") == env2.get("OUROBOROS_MODEL") == "openai/gpt-5.5"

    # No model anywhere: the pin falls back to the packaged default main model
    # (fallback == main holds in EVERY reachable configuration).
    monkeypatch.delenv("OUROBOROS_MODEL", raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACK", raising=False)
    empty_settings = tmp_path / "empty_settings.json"
    empty_settings.write_text("{}", encoding="utf-8")
    agent_bare = OuroborosTerminalBenchAgent(
        logs_dir=tmp_path, model_name="test",
        host_settings_path=str(empty_settings),
    )
    env3 = agent_bare._container_env()
    from ouroboros.config import SETTINGS_DEFAULTS
    assert env3.get("OUROBOROS_MODEL_FALLBACK") == SETTINGS_DEFAULTS["OUROBOROS_MODEL"]

def test_harbor_agent_defaults_max_workers_four_and_probes_context_timeout(tmp_path):
    """6c: 4 decomposition slots for the agent's own subagents (root takes one
    lane; container memory caps the pool — plan review needs no pool);
    6d: per-task timeout adopted from the harbor AgentContext when a future
    harbor exposes it (today: metadata probe)."""
    import types as _types

    from devtools.benchmarks.terminal_bench.harbor_installed_agent import (
        OuroborosTerminalBenchAgent,
    )

    agent = OuroborosTerminalBenchAgent(
        logs_dir=tmp_path, model_name="test",
        host_settings_path=str(tmp_path / "settings.json"),
    )
    assert agent.max_workers == 4
    assert agent.task_timeout_sec is None

    ctx = _types.SimpleNamespace(metadata={"task_timeout_sec": 900})
    assert agent._context_task_timeout_sec(ctx) == 900
    ctx_attr = _types.SimpleNamespace(agent_timeout_sec=600, metadata=None)
    assert agent._context_task_timeout_sec(ctx_attr) == 600
    ctx_none = _types.SimpleNamespace(metadata={})
    assert agent._context_task_timeout_sec(ctx_none) is None
    # Explicit kwarg still wins over the probe.
    agent_explicit = OuroborosTerminalBenchAgent(
        logs_dir=tmp_path, model_name="test",
        host_settings_path=str(tmp_path / "settings.json"),
        task_timeout_sec=300,
    )
    assert agent_explicit.task_timeout_sec == 300

def test_bench_template_scaffold_defaults_v655(tmp_path):
    """v6.55.0 shared bench-template decisions: safety light inside the jail,
    claude_code_edit disabled regardless of the web gate, the raised
    finalization margin, and the workers=4 templates across GAIA/SWE-pro."""
    import json as _json
    import pathlib as _pathlib

    from devtools.benchmarks.terminal_bench.harbor_installed_agent import (
        OuroborosTerminalBenchAgent,
    )

    agent = OuroborosTerminalBenchAgent(
        logs_dir=tmp_path, model_name="test",
        host_settings_path=str(tmp_path / "settings.json"),
    )
    env = agent._container_env()
    assert env["OUROBOROS_SAFETY_MODE"] == "light"
    assert env["OUROBOROS_MAX_WORKERS"] == "4"
    # claude_code_edit is withheld in BOTH web modes; the web group must mirror
    # the registry's REAL _WEB_TOOLS set (the adapter list had drifted when
    # youtube_transcript joined _WEB_TOOLS in v6.52.1), and view_image stays
    # available.
    from ouroboros.tools.registry import _WEB_TOOLS

    assert set(OuroborosTerminalBenchAgent._WEB_TOOLS_MIRROR) == set(_WEB_TOOLS)
    web_off = agent._disabled_tools()
    assert web_off[-2:] == ["claude_code_edit", "schedule_subagent"]
    assert set(_WEB_TOOLS) <= set(web_off)
    assert {"analyze_screenshot", "vlm_query"} <= set(web_off)
    assert "view_image" not in web_off
    agent.disable_agent_web = False
    assert agent._disabled_tools() == ["claude_code_edit", "schedule_subagent"]
    assert OuroborosTerminalBenchAgent._DEADLINE_SAFETY_SEC == 105

    bench_root = _pathlib.Path(__file__).resolve().parents[1] / "devtools" / "benchmarks"
    gaia = _json.loads((bench_root / "gaia" / "settings_base.json").read_text(encoding="utf-8"))
    assert gaia["OUROBOROS_MAX_WORKERS"] == 4
    assert gaia["OUROBOROS_SAFETY_MODE"] == "light"
    swepro = _json.loads((bench_root / "swe_bench_pro" / "e1v2" / "settings_base.json").read_text(encoding="utf-8"))
    assert swepro["OUROBOROS_MAX_WORKERS"] == 4
    assert swepro["OUROBOROS_SAFETY_MODE"] == "light"
    assert swepro["OUROBOROS_RUNTIME_MODE"] == "pro"
