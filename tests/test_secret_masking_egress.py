"""Admitted file content stays exact; diagnostic projections still redact."""
from __future__ import annotations

import hashlib
import json
import re

import pytest

from ouroboros.observability import redact_projection
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools.registry import ToolRegistry

pytestmark = pytest.mark.serial

TOKEN = "ghp_" + "syntheticfixture0123456789" * 2
PEM = "-----BEGIN PRIVATE KEY-----\nsynthetic-key-material\n-----END PRIVATE KEY-----\n"
_FINGERPRINT_RE = re.compile(r"^\*\*\*REDACTED\[\w+:len=\d+:sha256_8=[0-9a-f]{8}\]\*\*\*$")


@pytest.fixture()
def reader(tmp_path, monkeypatch):
    repo, data, home = (tmp_path / name for name in ("repo", "data", "home"))
    for path in (repo, data, home):
        path.mkdir()
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    return ToolRegistry(repo, data), repo, home


def _actor(registry, monkeypatch, mode, actor):
    from ouroboros.config import reset_runtime_mode_baseline_for_tests
    reset_runtime_mode_baseline_for_tests()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    if actor != "parent":
        registry._ctx.task_constraint = TaskConstraint(
            mode=actor, surface="external_workspace", write_root=str(registry._ctx.repo_dir))


@pytest.mark.parametrize("mode", ["light", "advanced", "pro", "cyber_pro"])
@pytest.mark.parametrize("actor", ["parent", "local_readonly_subagent", "acting_subagent"])
def test_file_reads_preserve_source_and_exact_receipt(reader, monkeypatch, mode, actor):
    registry, repo, _home = reader
    _actor(registry, monkeypatch, mode, actor)
    source = "prefix\n" + TOKEN + "\n" + PEM + "x" * 4000 + "\n"
    raw = source.replace("\n", "\r\n").encode("utf-8")
    (repo / "source.txt").write_bytes(raw)
    result = registry.execute("read_file", {"path": "source.txt"})
    assert "SECRET_BYTES_MASKED" not in result
    view = registry._ctx.last_read_view
    from ouroboros.tools.core_file_tools import delivered_source_prefix
    assert delivered_source_prefix(view, result, len(result)) == source
    assert view["source_masked"] is False
    assert view["source_revision"] == hashlib.sha256(raw).hexdigest()
    assert view["complete_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert view["source_end_char"] == len(source)
    fragment = registry.execute("read_file", {"path": "source.txt", "start_line": 4, "max_lines": 1, "start_char": 3})
    assert "thetic-key-material\n" in fragment
    assert registry._ctx.last_read_view["source_masked"] is False


@pytest.mark.parametrize("mode", ["light", "advanced", "pro", "cyber_pro"])
@pytest.mark.parametrize("actor", ["parent", "local_readonly_subagent", "acting_subagent"])
@pytest.mark.parametrize("fallback", [False, True])
def test_search_and_query_preserve_source_identifiers(reader, monkeypatch, mode, actor, fallback):
    registry, repo, _home = reader
    _actor(registry, monkeypatch, mode, actor)
    source = f"def {TOKEN}():\n    return 'synthetic-key-material'\n"
    (repo / "source.py").write_text(source, encoding="utf-8")
    if fallback:
        monkeypatch.setattr("ouroboros.code_search_rg._rg_binary", lambda: "")
    else:
        from tests.test_code_search_rg import _install_fake_rg
        _install_fake_rg(repo.parent, monkeypatch)
    result = registry.execute("search_code", {"query": "def "})
    assert TOKEN in result and "SECRET_BYTES_MASKED" not in result
    assert ("files searched" if fallback else "ripgrep") in result
    for query in ({"op": "symbols"}, {"op": "digest"}):
        result = registry.execute("query_code", query)
        assert TOKEN in result and "SECRET_BYTES_MASKED" not in result
    if actor == "local_readonly_subagent" or (actor == "acting_subagent" and mode != "cyber_pro"):
        assert not list((registry._ctx.drive_root / "state" / "code_intel").glob("*/inventory.json"))


@pytest.mark.parametrize("mode", ["light", "advanced", "pro", "cyber_pro"])
def test_owner_home_read_search_query_and_pdf_preserve_content(reader, monkeypatch, mode):
    from ouroboros.tools import media
    from tests.test_media_tools import _patch_pypdf, _FakePage
    registry, _repo, home = reader
    _actor(registry, monkeypatch, mode, "parent")
    source = f"def {TOKEN}():\n    pass\n"
    (home / "source.py").write_text(source, encoding="utf-8")
    for name, args in (
        ("read_file", {"path": "source.py"}),
        ("search_code", {"query": "def "}),
        ("query_code", {"op": "symbols", "path": str(home)}),
        ("query_code", {"op": "digest", "path": str(home)}),
    ):
        result = registry.execute(name, {"root": "user_files", **args})
        assert TOKEN in result and "SECRET_BYTES_MASKED" not in result
    pdf = home / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    _patch_pypdf(monkeypatch, [_FakePage(TOKEN + "\n" + PEM)])
    result = media._ocr_pdf(registry._ctx, str(pdf))
    assert TOKEN in result and PEM.strip() in result
    assert "SECRET_BYTES_MASKED" not in result


def test_redaction_preserves_credential_metadata_keys_without_allowlist():
    # G11: token_budget / token_estimate / credential_profile_id are metadata
    # ABOUT credentials; the old segment test destroyed them irreversibly and
    # was patched per-name via _NON_SECRET_KEY_NAMES (now deleted).
    payload = {
        "token_budget": 40000,
        "token_estimate": 789,
        "prompt_token_details": {"cached_tokens": 6},
        "credential_profile_id": "proton4",
        "api_key_id": "AKIA-style-identifier-name",
    }
    redacted = redact_projection(payload)
    assert redacted.value == payload
    assert redacted.manifest()["redacted"] is False


def test_redaction_still_masks_real_secret_keys_and_id_token():
    payload = {
        "id_token": "eyJhbGciOiJIUzI1NiJ9.payloadpayload.signaturesignature",
        "auth_token": "real-secret-value-123456",
    }
    redacted = redact_projection(payload)
    rendered = json.dumps(redacted.value)
    assert "real-secret-value-123456" not in rendered
    # id_token (OIDC) is a credential: the trailing-qualifier rule is
    # trailing-only and must not exempt it.
    assert "signaturesignature" not in rendered


def test_secret_key_redaction_fingerprints_instead_of_destroying():
    secret = "real-secret-value-123456"
    first = redact_projection({"auth_token": secret}).value["auth_token"]
    second = redact_projection({"auth_token": secret}).value["auth_token"]
    other = redact_projection({"auth_token": secret + "x"}).value["auth_token"]
    assert _FINGERPRINT_RE.fullmatch(first)
    assert secret not in first
    assert f"len={len(secret)}" in first
    # Deterministic: equality/rotation stays auditable without the raw bytes.
    assert first == second
    assert first != other
