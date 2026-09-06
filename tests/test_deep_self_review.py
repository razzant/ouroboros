"""Tests for ouroboros.deep_self_review module."""

from __future__ import annotations

import os
import pathlib
from unittest import mock

import pytest

from ouroboros.provider_models import OPENAI_DIRECT_DEFAULTS
from ouroboros.deep_self_review import (
    build_review_pack,
    deep_review_route,
    run_deep_self_review,
)
from ouroboros.reviewer_slot_config import DEEP_REVIEW_SLOT_ID, ConfiguredReviewerSlot
from ouroboros.tools.review_helpers import _is_probably_binary


def _packed_row(model: str) -> ConfiguredReviewerSlot:
    """The packed deep-review row (a direct api row: the historical delivery)."""
    return ConfiguredReviewerSlot(slot_id=DEEP_REVIEW_SLOT_ID, kind="api_chat", target_id=model)


def _make_dulwich_mock(file_list: list[str]):
    """Return a mock for dulwich.repo.Repo that yields the given file list from open_index()."""
    mock_index = mock.Mock()
    mock_index.__iter__ = mock.Mock(return_value=iter(f.encode() for f in file_list))
    mock_repo = mock.Mock()
    mock_repo.open_index.return_value = mock_index
    mock_repo_cls = mock.Mock(return_value=mock_repo)
    return mock_repo_cls


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a minimal git repo with tracked files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (repo / "lib.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")
    return repo


@pytest.fixture
def tmp_drive(tmp_path):
    """Create a drive root with some memory files."""
    drive = tmp_path / "drive"
    drive.mkdir()
    mem = drive / "memory"
    mem.mkdir()
    (mem / "identity.md").write_text("I am Ouroboros.\n", encoding="utf-8")
    (mem / "scratchpad.md").write_text("Working notes.\n", encoding="utf-8")
    know = mem / "knowledge"
    know.mkdir()
    (know / "patterns.md").write_text("## Patterns\n- Error class A\n", encoding="utf-8")
    return drive


class TestBuildReviewPack:
    def test_reads_tracked_files(self, tmp_repo, tmp_drive):
        """git ls-files output determines which repo files are included."""
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py", "lib.py"])):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)

        assert "### main.py" in pack
        assert "### lib.py" in pack
        assert "print('hello')" in pack
        assert stats["file_count"] >= 2

        atlas = mock.Mock(
            status="budget_exceeded",
            manifest={"estimated_total_tokens": 950_000},
            omitted=(),
            selected=(),
            text="small atlas",
        )
        with (
            mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py"])),
            mock.patch("ouroboros.deep_self_review.compile_review_context_atlas", return_value=atlas),
        ):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)
        assert pack == ""
        assert "exceeded hard budget" in stats["skipped"][0]
        assert stats["context_manifest"]["estimated_total_tokens"] == 950_000

    def test_includes_memory_whitelist(self, tmp_repo, tmp_drive):
        """Memory whitelist files from drive_root are included."""
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py"])):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)

        assert "## FILE: drive/memory/identity.md" in pack
        assert "I am Ouroboros." in pack
        assert "## FILE: drive/memory/scratchpad.md" in pack
        assert "## FILE: drive/memory/knowledge/patterns.md" in pack

    def test_includes_improvement_backlog_when_present(self, tmp_repo, tmp_drive):
        (tmp_drive / "memory" / "knowledge" / "improvement-backlog.md").write_text(
            "# Improvement Backlog\n\n### ibl-1\n- summary: Fix recurring review blocker\n",
            encoding="utf-8",
        )
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py"])):
            pack, _stats = build_review_pack(tmp_repo, tmp_drive)

        assert "## FILE: drive/memory/knowledge/improvement-backlog.md" in pack
        assert "Fix recurring review blocker" in pack

    def test_skips_missing_memory(self, tmp_repo, tmp_drive):
        """Missing memory files are not inlined — and their absence is DISCLOSED
        (omission section + typed disposition), never silently skipped."""
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py"])):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)

        for rel in ("memory/registry.md", "memory/WORLD.md", "memory/knowledge/index-full.md"):
            assert f"## FILE: drive/{rel}" not in pack
            assert f"drive/{rel} (missing: not present under the data root)" in pack[pack.index("## OMITTED FILES"):]
            assert stats["memory"]["dispositions"][rel] == "missing"
        assert stats["memory"]["inlined"] == 3 and stats["memory"]["total"] == 7
        assert stats["memory"]["dispositions"]["memory/identity.md"] == "inlined"


class TestPackedRouteAvailability:
    """Availability (`deep_review_route`) of the SYNTHESIZED packed row: no `deep_review` row saved,
    so the legacy model key is the migration source and the packed rules
    (credentials, the -pro rewrite, the OPENAI_BASE_URL trust rule) apply."""

    def test_openrouter(self):
        with mock.patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "sk-or-test", "OUROBOROS_MODEL_DEEP_SELF_REVIEW": "openai/gpt-5.5-pro"},
            clear=True,
        ):
            reason, model = deep_review_route()
            available = not reason
        assert available is True
        assert model == "openai/gpt-5.5-pro"

    def test_openai(self):
        env = {"OPENAI_API_KEY": "sk-test"}
        with mock.patch.dict(os.environ, env, clear=False):
            # Ensure OPENROUTER_API_KEY and OPENAI_BASE_URL are not set
            os.environ.pop("OPENROUTER_API_KEY", None)
            os.environ.pop("OPENAI_BASE_URL", None)
            reason, model = deep_review_route()
            available = not reason
        assert available is True
        # The direct route lands on the PROVIDER default, not a mechanical
        # `openai::` + router-slug rewrite: `-pro` is an OpenRouter routing slug
        # that 404s on api.openai.com (live-probed 2026-07-29).
        assert model == OPENAI_DIRECT_DEFAULTS["deep_self_review"]
        assert not model.endswith("-pro")

    def test_none(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            reason, model = deep_review_route()
            available = not reason
        assert available is False
        assert model is None

    def test_direct_provider_prefix_requires_matching_key_even_with_openrouter(self):
        with mock.patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "sk-or-test", "OUROBOROS_MODEL_DEEP_SELF_REVIEW": "anthropic::claude-opus-4.8"},
            clear=True,
        ):
            reason, model = deep_review_route()
            available = not reason

        assert available is False
        assert model is None

    def test_direct_provider_prefix_available_with_matching_key(self):
        with mock.patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "sk-ant-test", "OUROBOROS_MODEL_DEEP_SELF_REVIEW": "anthropic::claude-opus-4.8"},
            clear=True,
        ):
            reason, model = deep_review_route()
            available = not reason

        assert available is True
        assert model == "anthropic::claude-opus-4.8"


class TestRequestToolEmitsEvent:
    def test_emits_correct_event(self):
        """_request_deep_self_review emits a deep_self_review_request event."""
        from ouroboros.tools.control import _request_deep_self_review

        class FakeCtx:
            pending_events = []

        ctx = FakeCtx()
        with mock.patch(
            "ouroboros.deep_self_review.deep_review_route",
            return_value=("", "openai/gpt-5.5-pro"),
        ):
            result = _request_deep_self_review(ctx, "test reason")
        assert len(ctx.pending_events) == 1
        evt = ctx.pending_events[0]
        assert evt["type"] == "deep_self_review_request"
        assert evt["reason"] == "test reason"
        assert evt["model"] == "openai/gpt-5.5-pro"
        assert "Deep self-review" in result

    def test_unavailable_returns_error(self):
        """An unavailable ROW returns the typed reason without emitting an event."""
        from ouroboros.tools.control import _request_deep_self_review

        class FakeCtx:
            pending_events = []

        ctx = FakeCtx()
        with mock.patch(
            "ouroboros.deep_self_review.deep_review_route",
            return_value=("no provider credentials for openai/x", None),
        ):
            result = _request_deep_self_review(ctx, "test reason")
        assert len(ctx.pending_events) == 0
        assert result.startswith("❌ Deep self-review unavailable: no provider credentials for openai/x")
        assert "Review lanes" in result


class TestVendoredFilesExcluded:
    def test_minified_js_skipped(self, tmp_repo, tmp_drive):
        """Files with .min.js suffix are excluded from the review pack."""
        (tmp_repo / "lib.min.js").write_text("!function(){var a=1;}()\n")
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py", "lib.min.js"])):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)

        assert "vendored_minified" in str(stats["skipped"])
        assert "## FILE: lib.min.js" not in pack

    def test_chart_umd_skipped(self, tmp_repo, tmp_drive):
        """chart.umd.min.js (vendored Chart.js) is excluded by name and appears in OMITTED section."""
        (tmp_repo / "chart.umd.min.js").write_text("!function(t,e){/* chart.js minified */}()\n")
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py", "chart.umd.min.js"])):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)

        assert "## FILE: chart.umd.min.js" not in pack
        assert any("chart.umd.min.js" in s for s in stats["skipped"])
        # Omission section must be present and mention the file
        assert "## OMITTED FILES" in pack
        assert "chart.umd.min.js" in pack

    def test_min_css_skipped(self, tmp_repo, tmp_drive):
        """Files with .min.css suffix are excluded."""
        (tmp_repo / "style.min.css").write_text("body{margin:0}a{color:red}\n")
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py", "style.min.css"])):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)

        assert "## FILE: style.min.css" not in pack
        assert any("style.min.css" in s for s in stats["skipped"])

    def test_regular_js_included(self, tmp_repo, tmp_drive):
        """Regular (non-minified) JS files are NOT excluded."""
        (tmp_repo / "app.js").write_text("console.log('hello');\n")
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py", "app.js"])):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)

        assert "### app.js" in pack
        assert "console.log('hello');" in pack
        assert not any("app.js" in s for s in stats["skipped"])

    def test_omission_section_after_memory_whitelist(self, tmp_repo, tmp_drive):
        """OMITTED FILES section is appended after both repo and memory passes, capturing all skips.

        Simulates a memory-whitelist read error by patching pathlib.Path.read_text so that
        identity.md raises PermissionError, ensuring it lands in skipped and the OMITTED section.
        """
        (tmp_repo / "lib.min.js").write_text("minified\n")
        (tmp_drive / "memory" / "identity.md").write_text("I am Ouroboros.\n")
        target_path = str(tmp_drive / "memory" / "identity.md")

        original_read_text = pathlib.Path.read_text

        def patched_read_text(self, encoding="utf-8", errors="replace"):
            if str(self) == target_path:
                raise PermissionError("mocked read error")
            return original_read_text(self, encoding=encoding, errors=errors)

        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py", "lib.min.js"])):
            with mock.patch("pathlib.Path.read_text", patched_read_text):
                pack, stats = build_review_pack(tmp_repo, tmp_drive)

        assert "## OMITTED FILES" in pack
        omitted_section_pos = pack.index("## OMITTED FILES")
        # Vendored file listed in omitted section
        assert "lib.min.js" in pack[omitted_section_pos:]
        # Memory read error captured in skipped
        memory_errors = [s for s in stats["skipped"] if "identity.md" in s and "read error" in s]
        assert memory_errors, "identity.md read error should appear in skipped"
        # And it appears in the OMITTED section too
        assert "identity.md" in pack[omitted_section_pos:]


class TestIsProbablyBinary:
    def test_nul_byte_is_binary(self, tmp_path):
        """File containing a NUL byte is detected as binary."""
        f = tmp_path / "blob.bin"
        f.write_bytes(b"some text\x00more text")
        assert _is_probably_binary(f) is True

    def test_plain_text_is_not_binary(self, tmp_path):
        """Plain text file is not detected as binary."""
        f = tmp_path / "script.py"
        f.write_text("def hello():\n    return 'world'\n")
        assert _is_probably_binary(f) is False

    def test_high_non_printable_ratio_is_binary(self, tmp_path):
        """File with >30% non-printable bytes (ASCII control range) is detected as binary."""
        # 40% non-printable (bytes 1–8 range, ASCII control chars)
        payload = bytes(range(1, 9)) * 10 + b"normal text" * 3
        f = tmp_path / "data.unknown"
        f.write_bytes(payload)
        assert _is_probably_binary(f) is True

    def test_high_byte_ratio_is_binary(self, tmp_path):
        """File with invalid UTF-8 high bytes (no NUL) is detected as binary.

        bytes >= 128 alone are safe for valid UTF-8 (Cyrillic, CJK), but
        invalid UTF-8 sequences (e.g. raw Latin-1 bytes 0x80-0xFF) must still
        be caught by the incremental UTF-8 decode check.
        """
        # Raw Latin-1 bytes 0x80-0xFF: invalid UTF-8, no NUL, few control chars
        payload = bytes(range(128, 256)) * 5 + b"ascii text" * 5
        f = tmp_path / "data.blob"
        f.write_bytes(payload)
        assert _is_probably_binary(f) is True

    def test_only_first_sniff_bytes_read(self, tmp_path):
        """_is_probably_binary only reads _BINARY_SNIFF_BYTES bytes, not the whole file."""
        from ouroboros.tools.review_helpers import _BINARY_SNIFF_BYTES
        # File is mostly text but has NUL in the first 8KB window
        payload = b"text data\x00more" + b"a" * (_BINARY_SNIFF_BYTES * 2)
        f = tmp_path / "big.bin"
        f.write_bytes(payload)
        # Should detect NUL in the first chunk and return True
        assert _is_probably_binary(f) is True

    def test_empty_file_is_not_binary(self, tmp_path):
        """Empty file does not crash and returns False."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert _is_probably_binary(f) is False

    def test_missing_file_returns_false(self, tmp_path):
        """Missing file returns False (let caller handle read failure)."""
        f = tmp_path / "does_not_exist.bin"
        assert _is_probably_binary(f) is False

    def test_unlisted_extension_binary_excluded_from_pack(self, tmp_repo, tmp_drive):
        """Binary file with unlisted extension (.bin) is excluded via content sniffer."""
        (tmp_repo / "model.bin").write_bytes(b"GGUF\x00" + b"\x00\xff" * 100)
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py", "model.bin"])):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)

        assert "## FILE: model.bin" not in pack
        assert any("model.bin" in s for s in stats["skipped"])


class TestBinaryFilesExcluded:
    def test_png_skipped(self, tmp_repo, tmp_drive):
        """PNG images are excluded — reading them produces garbage replacement chars."""
        (tmp_repo / "screenshot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py", "screenshot.png"])):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)

        assert "## FILE: screenshot.png" not in pack
        assert any("screenshot.png" in s for s in stats["skipped"])

    def test_jpg_skipped(self, tmp_repo, tmp_drive):
        """JPEG images are excluded."""
        (tmp_repo / "logo.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py", "logo.jpg"])):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)

        assert "## FILE: logo.jpg" not in pack
        assert any("logo.jpg" in s for s in stats["skipped"])

    def test_svg_skipped(self, tmp_repo, tmp_drive):
        """SVG files are excluded (provider icons can be large XML)."""
        (tmp_repo / "icon.svg").write_text("<svg><circle r='10'/></svg>\n")
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py", "icon.svg"])):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)

        assert "## FILE: icon.svg" not in pack
        assert any("icon.svg" in s for s in stats["skipped"])

    def test_ico_skipped(self, tmp_repo, tmp_drive):
        """ICO files are excluded."""
        (tmp_repo / "favicon.ico").write_bytes(b"\x00\x00\x01\x00" + b"\x00" * 50)
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py", "favicon.ico"])):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)

        assert "## FILE: favicon.ico" not in pack
        assert any("favicon.ico" in s for s in stats["skipped"])

    def test_python_source_not_skipped(self, tmp_repo, tmp_drive):
        """Python source files (.py) are NOT excluded by the binary filter."""
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py"])):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)

        assert "### main.py" in pack


class TestSkipDirPrefixes:
    def test_assets_dir_excluded(self, tmp_repo, tmp_drive):
        """Files under assets/ are excluded (README screenshots, app icons)."""
        assets = tmp_repo / "assets"
        assets.mkdir()
        (assets / "chat.png").write_bytes(b"\x89PNG\r\n" + b"\x00" * 100)
        (assets / "logo.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py", "assets/chat.png", "assets/logo.jpg"])):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)

        assert "## FILE: assets/chat.png" not in pack
        assert "## FILE: assets/logo.jpg" not in pack
        assert any("assets/chat.png" in s for s in stats["skipped"])
        assert any("assets/logo.jpg" in s for s in stats["skipped"])
        assert "### main.py" in pack  # non-assets file still present

    def test_web_dir_not_excluded(self, tmp_repo, tmp_drive):
        """Files under web/ (SPA modules) are NOT excluded."""
        web = tmp_repo / "web" / "modules"
        web.mkdir(parents=True)
        (web / "chat.js").write_text("// chat module\n")
        with mock.patch("dulwich.repo.Repo", _make_dulwich_mock(["main.py", "web/modules/chat.js"])):
            pack, stats = build_review_pack(tmp_repo, tmp_drive)

        assert "### web/modules/chat.js" in pack
        assert not any("web/modules/chat.js" in s for s in stats["skipped"])


class TestNoProxyLlmChat:
    """LLMClient.chat(no_proxy=True) — proxy-free httpx transport for macOS fork-safety."""

    def test_chat_no_proxy_uses_trust_env_false(self):
        """chat(no_proxy=True) builds an httpx.Client with trust_env=False and mounts={}."""
        import httpx
        from ouroboros.llm import LLMClient

        captured_clients = []

        real_httpx_client = httpx.Client

        def capturing_httpx_client(*args, **kwargs):
            c = real_httpx_client(*args, **kwargs)
            captured_clients.append(c)
            return c

        llm = LLMClient()
        mock_resp = mock.Mock()
        mock_resp.model_dump.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

        # Resolve the SDK before mocking the HTTP class it inherits on import.
        with mock.patch("openai.OpenAI") as mock_openai_cls:
            with mock.patch("httpx.Client", side_effect=capturing_httpx_client):
                mock_oa = mock.Mock()
                mock_oa.chat.completions.create.return_value = mock_resp
                mock_openai_cls.return_value = mock_oa

                with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=False):
                    llm.chat(
                        messages=[{"role": "user", "content": "hi"}],
                        model="openai/gpt-5.5-pro",
                        max_tokens=8,
                        no_proxy=True,
                    )

        # At least one httpx.Client was created
        assert len(captured_clients) >= 1
        created = captured_clients[0]
        # trust_env=False and mounts={} are the key invariants
        assert created._mounts == {} or not created._mounts

    def test_chat_no_proxy_closes_http_client(self):
        """chat(no_proxy=True) closes the one-shot httpx.Client after the call."""
        import httpx
        from ouroboros.llm import LLMClient

        closed_clients = []
        real_httpx_client = httpx.Client

        class TrackingClient(real_httpx_client):
            def close(self):
                closed_clients.append(self)
                super().close()

        llm = LLMClient()
        mock_resp = mock.Mock()
        mock_resp.model_dump.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

        with mock.patch("openai.OpenAI") as mock_openai_cls:
            with mock.patch("httpx.Client", TrackingClient):
                mock_oa = mock.Mock()
                mock_oa.chat.completions.create.return_value = mock_resp
                mock_openai_cls.return_value = mock_oa

                with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=False):
                    llm.chat(
                        messages=[{"role": "user", "content": "hi"}],
                        model="openai/gpt-5.5-pro",
                        max_tokens=8,
                        no_proxy=True,
                    )

        assert len(closed_clients) >= 1, "httpx.Client must be closed after no_proxy call"

    def test_chat_no_proxy_closes_on_exception(self):
        """chat(no_proxy=True) closes the http client even when the API call raises."""
        import httpx
        from ouroboros.llm import LLMClient

        closed_clients = []
        real_httpx_client = httpx.Client

        class TrackingClient(real_httpx_client):
            def close(self):
                closed_clients.append(self)
                super().close()

        llm = LLMClient()

        with mock.patch("openai.OpenAI") as mock_openai_cls:
            with mock.patch("httpx.Client", TrackingClient):
                mock_oa = mock.Mock()
                mock_oa.chat.completions.create.side_effect = RuntimeError("boom")
                mock_openai_cls.return_value = mock_oa

                with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=False):
                    with pytest.raises(RuntimeError, match="boom"):
                        llm.chat(
                            messages=[{"role": "user", "content": "hi"}],
                            model="openai/gpt-5.5-pro",
                            max_tokens=8,
                            no_proxy=True,
                        )

        assert len(closed_clients) >= 1, "httpx.Client must be closed even after exception"

    def test_chat_no_proxy_skips_generation_cost_fetch(self):
        """chat(no_proxy=True) does not call _fetch_generation_cost (proxy/OS path)."""
        from ouroboros.llm import LLMClient

        llm = LLMClient()
        mock_resp = mock.Mock()
        mock_resp.model_dump.return_value = {
            "id": "gen-abc123",  # has a generation id — would trigger cost fetch normally
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        with mock.patch("openai.OpenAI") as mock_openai_cls:
            with mock.patch("httpx.Client") as mock_httpx_cls:
                mock_http = mock.Mock()
                mock_httpx_cls.return_value = mock_http
                mock_oa = mock.Mock()
                mock_oa.chat.completions.create.return_value = mock_resp
                mock_openai_cls.return_value = mock_oa
                with mock.patch.object(llm, "_fetch_generation_cost") as mock_cost:
                    with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=False):
                        llm.chat(
                            messages=[{"role": "user", "content": "hi"}],
                            model="openai/gpt-5.5-pro",
                            max_tokens=8,
                            no_proxy=True,
                        )
                    mock_cost.assert_not_called()

    def test_chat_no_proxy_false_uses_cached_client(self):
        """chat(no_proxy=False, default) uses the shared cached client, not a new one."""
        import httpx
        from ouroboros.llm import LLMClient

        new_clients = []
        real_httpx_client = httpx.Client

        def counting_httpx_client(*args, **kwargs):
            c = real_httpx_client(*args, **kwargs)
            new_clients.append(c)
            return c

        llm = LLMClient()
        mock_resp = mock.Mock()
        mock_resp.model_dump.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

        with mock.patch("httpx.Client", side_effect=counting_httpx_client):
            with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=False):
                with mock.patch.object(llm, "_get_remote_client") as mock_get:
                    mock_oa = mock.Mock()
                    mock_oa.chat.completions.create.return_value = mock_resp
                    mock_get.return_value = mock_oa
                    llm.chat(
                        messages=[{"role": "user", "content": "hi"}],
                        model="openai/gpt-5.5-pro",
                        max_tokens=8,
                        no_proxy=False,
                    )
                    mock_get.assert_called_once()

        # no_proxy=False must not construct a new httpx.Client
        assert len(new_clients) == 0

    def test_run_deep_self_review_calls_llm_with_no_proxy_and_configured_effort(self, tmp_repo, tmp_drive, monkeypatch):
        """The packed row's wire call: no_proxy=True, the surface effort when
        the row names none, and the report delivered behind the host header."""
        from ouroboros.deep_self_review import run_deep_self_review
        small_pack = "x" * 100
        manifest = {"status": "ok", "selected_count": 1}
        mock_llm = mock.Mock()
        mock_llm.chat.return_value = ({"content": "Review result."}, {"cost": 0.01})
        monkeypatch.setenv("OUROBOROS_EFFORT_DEEP_SELF_REVIEW", "medium")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with mock.patch(
            "ouroboros.deep_self_review.build_review_pack",
            return_value=(
                small_pack,
                {
                    "file_count": 1,
                    "total_chars": len(small_pack),
                    "skipped": [],
                    "context_manifest": manifest,
                },
            ),
        ):
            result, usage = run_deep_self_review(
                repo_dir=tmp_repo,
                drive_root=tmp_drive,
                llm=mock_llm,
                emit_progress=lambda x, *, incident=None: None,
                slot=_packed_row("openai/gpt-5.5-pro"),
            )

        assert result.endswith("\n\nReview result.")
        assert result.startswith("<!-- deep-review provenance: delivery=api_packet, model=openai/gpt-5.5-pro, ")
        assert usage["resolved_model"] == "openai/gpt-5.5-pro" and "execution_status" not in usage
        mock_llm.chat.assert_called_once()
        _, kwargs = mock_llm.chat.call_args
        assert kwargs.get("no_proxy") is True, "llm.chat must be called with no_proxy=True"
        assert kwargs.get("reasoning_effort") == "medium"
        sidecar = tmp_drive / "state" / "deep_self_review_context.json"
        assert sidecar.is_file()
        assert '"context_manifest"' in sidecar.read_text(encoding="utf-8")
        assert '"selected_count": 1' in sidecar.read_text(encoding="utf-8")


class TestReviewPackOverflow:
    def test_overflow_shrinks_and_proceeds(self, tmp_repo, tmp_drive, monkeypatch):
        """An estimator-drift overshoot triggers ONE tighter rebuild, then the
        review proceeds — the historical '+853 tokens' fatal error class."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        huge_pack = "x" * 4_000_000  # > 745K-token gate
        small_pack = "y" * 4_000     # comfortably under
        mock_llm = mock.Mock()
        mock_llm.chat.return_value = ({"content": "Review result."}, {"cost": 0.0})
        build_calls = []

        def fake_build(repo_dir, drive_root, fixed_prompt_tokens=0, hard_budget_reduction=0, input_token_limit=0):
            build_calls.append(hard_budget_reduction)
            if hard_budget_reduction:
                return small_pack, {"file_count": 5, "total_chars": len(small_pack), "skipped": []}
            return huge_pack, {"file_count": 100, "total_chars": len(huge_pack), "skipped": []}

        with (
            mock.patch("ouroboros.deep_self_review.build_review_pack", side_effect=fake_build),
            mock.patch(
                "ouroboros.llm_observability.chat_observed",
                return_value=({"content": "Review result."}, {"cost": 0.0}),
            ),
        ):
            result, _usage = run_deep_self_review(
                repo_dir=tmp_repo,
                drive_root=tmp_drive,
                llm=mock_llm,
                emit_progress=lambda x, *, incident=None: None,
                slot=_packed_row("test-model"),
            )

        assert result.endswith("\n\nReview result.")
        assert len(build_calls) == 2, "must rebuild once with a tighter budget"
        assert build_calls[1] > 0, "retry must reduce the atlas hard budget"

    def test_explicit_error_when_shrink_cannot_fit(self, tmp_repo, tmp_drive, monkeypatch):
        """If even the tighter rebuild stays over the gate, fail closed with the
        explicit error (the pinned last-resort assertion) and TYPED usage, so
        the caller never writes the error over the previous report."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        huge_pack = "x" * 4_000_000
        mock_llm = mock.Mock()

        with mock.patch(
            "ouroboros.deep_self_review.build_review_pack",
            return_value=(huge_pack, {"file_count": 100, "total_chars": 4_000_000, "skipped": []}),
        ):
            result, usage = run_deep_self_review(
                repo_dir=tmp_repo,
                drive_root=tmp_drive,
                llm=mock_llm,
                emit_progress=lambda x, *, incident=None: None,
                slot=_packed_row("test-model"),
            )

        assert "too large" in result
        # v6.80.0: the deep reviewer's cap is DENSITY-calibrated per model at call
        # time (the module constant is only the uncalibrated window arithmetic), so
        # the message must quote the calibrated number actually enforced.
        # v6.87.9 / Ф3: the window is resolved from Capability Evidence per
        # reviewer. An unknown route keeps the full-window assumption and is
        # dispatched-and-disclosed (`window=assumed_1000000`, owner fork A
        # pending ratification); a CONFIRMED sub-1M route is a typed refusal
        # (no shrink). The quoted number follows that same resolution.
        from ouroboros.reviewer_window import (
            reviewer_context_window,
            window_scaled_reserves,
        )
        from ouroboros.tools.review_helpers import calibrated_input_token_limit
        from ouroboros.deep_self_review import (
            _DEEP_MAX_OUTPUT_TOKENS, _DEEP_OUTPUT_MARGIN_TOKENS,
        )
        window = reviewer_context_window("test-model")
        output_reserve, margin = window_scaled_reserves(
            window,
            output_reserve=_DEEP_MAX_OUTPUT_TOKENS,
            tokenizer_margin=_DEEP_OUTPUT_MARGIN_TOKENS,
        )
        enforced = calibrated_input_token_limit(
            "test-model",
            context_window=window,
            output_reserve=output_reserve,
            tokenizer_margin=margin,
        )
        assert f"{enforced:,}" in result
        assert usage == {"execution_status": "infra_failed", "reason_code": "deep_self_review_error"}
        mock_llm.chat.assert_not_called()


class TestOmissionSectionBound:
    def test_omission_section_stays_within_reserved_budget(self):
        """The in-prompt omission summary is bounded + reserved; a huge skipped
        list (the +853 root cause) can no longer push the assembled pack over
        the budget the atlas filled to."""
        from ouroboros.deep_self_review import (
            _OMISSION_SECTION_RESERVE_TOKENS,
            _append_omission_section,
        )
        from ouroboros.utils import estimate_tokens

        skipped = [
            f"some/very/long/path/segment_{i}/deeply/nested/file_{i}.py (excluded_test: wider tests excluded)"
            for i in range(500)
        ]
        parts: list[str] = []
        _append_omission_section(parts, skipped)

        assert len(parts) == 1
        section = parts[0]
        assert estimate_tokens(section) <= _OMISSION_SECTION_RESERVE_TOKENS
        assert "Omitted counts by reason" in section
        assert "excluded_test=500" in section
        assert "coverage manifest" in section  # explicit pointer, not silent truncation

    def test_omission_section_small_list_lists_everything(self):
        from ouroboros.deep_self_review import _append_omission_section

        skipped = ["a.py (oversized: >1MB)", "b.bin (binary/media: binary)"]
        parts: list[str] = []
        _append_omission_section(parts, skipped)
        assert "a.py (oversized: >1MB)" in parts[0]
        assert "b.bin (binary/media: binary)" in parts[0]
        assert "oversized=1" in parts[0]


def test_direct_openai_deep_review_sends_a_real_openai_model_id():
    """PHYSICAL-PAYLOAD proof, not a defaults-table assertion.

    An owner may pin the slug `openai/gpt-5.6-sol-pro`. That `-pro`
    suffix is an OpenRouter routing slug, NOT an OpenAI model id: live-probed
    2026-07-29, `gpt-5.6-sol-pro` on api.openai.com /v1/chat/completions returns
    404, while pro reasoning exists only on /v1/responses as
    `reasoning.mode="pro"` (200) — and /v1/chat/completions rejects a `reasoning`
    parameter outright (400 "Unknown parameter"). Every LLM call in llm.py is a
    chat.completions call, so the direct-OpenAI deep-review slot ships plain Sol
    and this test pins what actually reaches the wire.
    """
    import os
    from unittest import mock

    from ouroboros.llm import LLMClient
    from ouroboros.provider_models import OPENAI_DIRECT_DEFAULTS

    slot = OPENAI_DIRECT_DEFAULTS["deep_self_review"]
    assert slot.startswith("openai::"), slot

    mock_resp = mock.Mock()
    mock_resp.model_dump.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    with mock.patch("openai.OpenAI") as mock_openai_cls:
        mock_oa = mock.Mock()
        mock_oa.chat.completions.create.return_value = mock_resp
        mock_openai_cls.return_value = mock_oa
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-direct-test"}, clear=False):
            LLMClient().chat(
                messages=[{"role": "user", "content": "hi"}],
                model=slot, max_tokens=8, no_proxy=True,
            )
        assert mock_oa.chat.completions.create.called
        payload = mock_oa.chat.completions.create.call_args.kwargs

    # The id on the wire is a REAL OpenAI model, never the OpenRouter slug.
    assert payload["model"] == "gpt-5.6-sol"
    assert not payload["model"].endswith("-pro")
    # ...and no `reasoning` object is smuggled onto a chat.completions call, which
    # the live API rejects with 400 (the only pro carrier is the Responses API).
    assert "reasoning" not in payload
    assert "reasoning" not in (payload.get("extra_body") or {})


def test_direct_fallback_preserves_an_explicit_real_model_pin():
    """Only router-only `-pro` slugs are substituted by the provider default; an
    owner's explicit pin of a REAL OpenAI model keeps the mechanical rewrite."""
    import os
    from unittest import mock

    from ouroboros.provider_models import OPENAI_DIRECT_DEFAULTS

    env = {"OPENAI_API_KEY": "sk-test"}
    with mock.patch.dict(os.environ, env, clear=False):
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ.pop("OUROBOROS_REVIEWER_SLOTS", None)
        with mock.patch.dict(os.environ, {"OUROBOROS_MODEL_DEEP_SELF_REVIEW": "openai/gpt-5.5"}):
            reason, model = deep_review_route()
            available = not reason
        assert available is True
        assert model == "openai::gpt-5.5", "an explicit real-model pin survives"
        with mock.patch.dict(os.environ, {"OUROBOROS_MODEL_DEEP_SELF_REVIEW": "openai/gpt-5.5-pro"}):
            reason, model = deep_review_route()
            available = not reason
        assert available is True
        assert model == OPENAI_DIRECT_DEFAULTS["deep_self_review"], (
            "a router-only -pro slug lands on the provider default"
        )
