"""Benchmark surfaces must recognize every registered direct provider.

Wave-2 scope review (DeepSeek sprint) found three benchmark enumerations that
silently missed DeepSeek (and, in two of them, MiniMax): the CyberGym
settings-authoritative env sweep, ProgramBench's exclusive-direct-provider
mirror, and Terminal-Bench's container network preflight. These regressions pin
the fixes registry-first, so the NEXT provider cannot repeat the class.
"""

import asyncio
import contextlib
import io
import urllib.error
import urllib.request
from types import SimpleNamespace

from ouroboros.provider_models import (
    DEEPSEEK_BASE_URL,
    PROVIDER_CREDENTIAL_GROUPS,
    resolve_minimax_base_url,
)


def test_authoritative_env_prefixes_cover_every_registered_credential():
    """The settings-authoritative sweep promises to remove "provider/SDK
    families" — so every credential key the runtime registry knows must match
    one of its strip prefixes (or be an explicit exact/keep entry). DeepSeek's
    ambient key survived the sweep because DEEPSEEK_ was missing here."""
    from devtools.benchmarks.common.server_runner import (
        _AUTHORITATIVE_ENV_EXACT,
        _AUTHORITATIVE_ENV_KEEP,
        _AUTHORITATIVE_ENV_PREFIXES,
    )

    for provider, group in PROVIDER_CREDENTIAL_GROUPS.items():
        for key in group:
            covered = (
                key in _AUTHORITATIVE_ENV_KEEP
                or key in _AUTHORITATIVE_ENV_EXACT
                or key.startswith(_AUTHORITATIVE_ENV_PREFIXES)
            )
            assert covered, f"{provider}: ambient {key} would survive the authoritative sweep"


def test_programbench_mirror_recognizes_deepseek_and_minimax(monkeypatch):
    from devtools.benchmarks.programbench.run_programbench_e2e import _active_direct_provider

    for group in PROVIDER_CREDENTIAL_GROUPS.values():
        for key in group:
            monkeypatch.delenv(key, raising=False)

    assert _active_direct_provider({"DEEPSEEK_API_KEY": "sk-x"}) == "deepseek"
    assert _active_direct_provider({"MINIMAX_API_KEY": "mm-x"}) == "minimax"
    # Two configured direct providers are ambiguous — the OpenRouter-style route.
    assert _active_direct_provider(
        {"DEEPSEEK_API_KEY": "sk-x", "OPENAI_API_KEY": "sk-o"}
    ) == ""


class _PreflightEnv:
    """Replays the container-side probe script in-process (same harness the
    existing openai preflight test uses)."""

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


def _run_preflight(tmp_path, monkeypatch, env):
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    harness = _PreflightEnv()
    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)
    asyncio.run(agent._network_preflight(harness, env))
    report = (tmp_path / "network-preflight.txt").read_text(encoding="utf-8")
    return harness.command, report


def test_terminal_bench_preflight_probes_deepseek(tmp_path, monkeypatch):
    command, report = _run_preflight(tmp_path, monkeypatch, {"DEEPSEEK_API_KEY": "sk-x"})
    assert DEEPSEEK_BASE_URL.rstrip("/") + "/models" in command
    assert "deepseek_preflight_status 401" in report


def test_terminal_bench_preflight_probes_minimax_by_region(tmp_path, monkeypatch):
    command, _report = _run_preflight(
        tmp_path, monkeypatch, {"MINIMAX_API_KEY": "mm-x", "MINIMAX_REGION": "cn_zh"}
    )
    assert resolve_minimax_base_url("cn_zh").rstrip("/") + "/models" in command

    command, report = _run_preflight(tmp_path, monkeypatch, {"MINIMAX_API_KEY": "mm-x"})
    assert resolve_minimax_base_url("").rstrip("/") + "/models" in command
    assert "minimax_preflight_status 401" in report


def test_terminal_bench_container_env_forwards_minimax_region(tmp_path, monkeypatch):
    """The container resolves the MiniMax endpoint from MINIMAX_REGION; without
    the forward, a cn_zh owner silently probes and routes the global host."""
    import devtools.benchmarks.terminal_bench.harbor_installed_agent as tb_agent

    agent = tb_agent.OuroborosTerminalBenchAgent(logs_dir=tmp_path)
    monkeypatch.setattr(agent, "_host_settings", lambda: {})
    monkeypatch.setattr(agent, "_container_secret_injection_allowed", lambda settings: False)
    monkeypatch.setenv("MINIMAX_REGION", "cn_zh")
    env = agent._container_env()
    assert env.get("MINIMAX_REGION") == "cn_zh"
