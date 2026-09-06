"""Secret-byte egress masking (#447 X1/В23) and observability G11 contracts.

Egress contract: root may read owner-home files in full; the masked form of a
credential (``***``) may enter model context, the raw bytes never. G11
contract: key-name redaction preserves non-secret meta as a fingerprint
(type/len/sha256_8) instead of destroying it, and credential-metadata keys
(counts, budgets, ids) are structurally non-secret without a per-name allowlist.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from ouroboros.observability import redact_projection
from ouroboros.secret_masking import mask_secret_bytes
from ouroboros.tools.core import _read_file
from ouroboros.tools.registry import ToolContext


OPENROUTER_KEY = "sk-or-" + "abcd1234" * 4
GITHUB_TOKEN = "ghp_" + "abcdefghijklmnopqrstuvwxyz123456"
PEM_BLOCK = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gt\n"
    "-----END OPENSSH PRIVATE KEY-----"
)

_FINGERPRINT_RE = re.compile(r"^\*\*\*REDACTED\[\w+:len=\d+:sha256_8=[0-9a-f]{8}\]\*\*\*$")


@pytest.mark.parametrize("mask_opaque", [True, False])
def test_mask_secret_bytes_masks_entropy_formats(mask_opaque):
    text = f"config a\nkey={OPENROUTER_KEY}\nAuthorization: Bearer {GITHUB_TOKEN}\nplain tail"
    masked, count = mask_secret_bytes(text, mask_opaque=mask_opaque)
    assert OPENROUTER_KEY not in masked
    assert GITHUB_TOKEN not in masked
    assert count >= 2
    assert "***" in masked
    # Non-secret content survives byte-for-byte.
    assert "config a" in masked and "plain tail" in masked


@pytest.mark.parametrize("mask_opaque", [True, False])
def test_mask_secret_bytes_masks_pem_block(mask_opaque):
    masked, count = mask_secret_bytes(f"prefix\n{PEM_BLOCK}\nsuffix", mask_opaque=mask_opaque)
    assert "PRIVATE KEY" not in masked
    assert "b3BlbnNzaC1rZXktdjE" not in masked
    assert count == 1
    assert masked.startswith("prefix\n") and masked.endswith("\nsuffix")


@pytest.mark.parametrize("mask_opaque", [True, False])
def test_mask_secret_bytes_masks_unterminated_pem_to_end(mask_opaque):
    # A read slice can cut the file before the END marker; the tail is still
    # key material and must not survive.
    head, _, _ = PEM_BLOCK.partition("-----END")
    masked, count = mask_secret_bytes(f"prefix\n{head}", mask_opaque=mask_opaque)
    assert count == 1
    assert "b3BlbnNzaC1rZXktdjE" not in masked
    assert masked == "prefix\n***"


def test_mask_secret_bytes_leaves_plain_text_untouched():
    text = "ordinary notes\nmodel: anthropic/claude-fable-5\npath: ~/.config/app/settings.toml\n"
    masked, count = mask_secret_bytes(text)
    assert masked == text
    assert count == 0


def test_mask_secret_bytes_masks_long_opaque_runs():
    """s2r2 F1: line-oriented egresses surface key MATERIAL without block
    markers (a PEM body line, an AWS secret key). Any unbroken 40+ char opaque
    run is masked; a long hash is the documented accepted false positive."""
    body_line = "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMw" + "x" * 10
    masked, count = mask_secret_bytes(f"match: {body_line}\n")
    assert body_line not in masked and count == 1
    aws_secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    masked2, count2 = mask_secret_bytes(f"aws_secret_access_key = {aws_secret}\n")
    assert aws_secret not in masked2 and count2 == 1
    # accepted FP, disclosed by design: a bare sha256 is an opaque run too
    masked3, count3 = mask_secret_bytes("sha256: " + "a" * 64 + "\n")
    assert "a" * 64 not in masked3 and count3 == 1


def test_repo_precision_masking_preserves_long_source_and_hashes():
    source = "x" * 4000 + "\nsha256: " + "ab12cd34" * 8 + "\n"
    assert mask_secret_bytes(source, mask_opaque=False) == (source, 0)
    masked, count = mask_secret_bytes(source)
    assert count == 2 and "x" * 4000 not in masked


@pytest.mark.parametrize("profile", ["local_readonly_subagent", "acting_subagent"])
def test_restricted_repo_read_delivers_full_source_and_masks_known_credentials(tmp_path, profile):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolRegistry

    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    body = "x" * 4000 + "\nsha256: " + "ab12cd34" * 8 + "\n"
    (repo / "source.txt").write_text(body + GITHUB_TOKEN + "\n" + PEM_BLOCK, encoding="utf-8")
    registry = ToolRegistry(repo, data)
    registry._ctx.task_constraint = TaskConstraint(mode=profile, write_root=str(repo), surface="external_workspace")
    out = registry.execute("read_file", {"path": "source.txt"})
    assert body in out
    assert GITHUB_TOKEN not in out and "b3BlbnNzaC1rZXktdjE" not in out
    assert "SECRET_BYTES_MASKED" in out
    assert registry._ctx.last_read_view["end_line"] == registry._ctx.last_read_view["total_lines"]
    assert registry._ctx.last_read_view["opened_path"] == "source.txt"
    chunk = registry.execute("read_file", {"path": "source.txt", "start_char": 2000, "max_lines": 1})
    assert "x" * 2000 in chunk and "SECRET_BYTES_MASKED" not in chunk


@pytest.mark.parametrize("fallback", [False, True])
def test_restricted_repo_search_preserves_source_identifiers(tmp_path, monkeypatch, fallback):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolRegistry

    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    identifier = "ordinary_source_identifier_" + "x" * 50
    source = f"def {identifier}(value='{GITHUB_TOKEN}'):\n    return value\ndef {GITHUB_TOKEN}():\n    pass\n"
    (repo / "source.py").write_text(source, encoding="utf-8")
    registry = ToolRegistry(repo, data)
    registry._ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent")
    if fallback:
        monkeypatch.setattr("ouroboros.code_search_rg._rg_binary", lambda: "")
    out = registry.execute("search_code", {"query": "ordinary_source_identifier"})
    assert identifier in out and "SECRET_BYTES_MASKED" in out
    assert GITHUB_TOKEN not in out
    assert ("files searched" if fallback else "ripgrep") in out
    query = registry.execute("query_code", {"op": "definition", "query": identifier})
    assert identifier in query and "source.py:1" in query
    assert GITHUB_TOKEN not in query and "SECRET_BYTES_MASKED" not in query
    credential = registry.execute("query_code", {"op": "symbols", "path": "source.py"})
    assert GITHUB_TOKEN not in credential and "SECRET_BYTES_MASKED" in credential


def test_project_settings_source_is_readable_by_verify_guard(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.shell_guards import process_shell_guard_args
    from tests._typed_guard_shared import _shell_guard_text

    repo, data = tmp_path / "repo", tmp_path / "runtime"
    (repo / "data").mkdir(parents=True)
    data.mkdir()
    (repo / "data" / "settings.json").write_text('{"ordinary": "project fixture"}', encoding="utf-8")
    registry = ToolRegistry(repo, data)
    registry._ctx.task_constraint = TaskConstraint(mode="acting_subagent", surface="external_workspace", write_root=str(repo))
    mapped = process_shell_guard_args("verify_and_record", {"check": "cat data/settings.json", "cwd": str(repo)})
    result = _shell_guard_text(registry, mapped, "advanced")
    assert result is None, result
    assert "project fixture" in registry.execute("read_file", {"path": "data/settings.json"})


@pytest.mark.parametrize("profile", ["local_readonly_subagent", "acting_subagent"])
def test_runtime_data_inside_repo_keeps_its_read_protection(tmp_path, monkeypatch, profile):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    data = repo / "data"
    (data / "auth").mkdir(parents=True)
    (repo / "auth").mkdir()
    (repo / "auth" / "secret.py").write_text("def public_source():\n    pass\n", encoding="utf-8")
    (data / "auth" / "secret.py").write_text("def runtime_private():\n    pass\n", encoding="utf-8")
    (data / "settings.json").write_text('{"fixture": "runtime_private"}', encoding="utf-8")
    registry = ToolRegistry(repo, data)
    registry._ctx.task_constraint = TaskConstraint(mode=profile, write_root=str(repo), surface="external_workspace")
    for path in ("data/auth/secret.py", str(data / "auth" / "secret.py"), "data/settings.json"):
        result = registry.execute("read_file", {"path": path})
        assert "BLOCKED" in result and "runtime_private" not in result
    assert "auth/" not in registry.execute("list_files", {"path": "data"})
    assert "secret.py" in registry.execute("list_files", {"path": "auth"})
    assert "public_source" in registry.execute("read_file", {"path": "auth/secret.py"})
    query = registry.execute("query_code", {"op": "symbols"})
    assert "public_source" in query and "runtime_private" not in query
    monkeypatch.setattr("ouroboros.code_search_rg._rg_binary", lambda: "")
    search = registry.execute("search_code", {"query": "def "})
    assert "public_source" in search and "runtime_private" not in search
    assert "files searched" in search


@pytest.fixture()
def user_files_ctx(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for p in (system, workspace, data):
        p.mkdir()
    ctx = ToolContext(repo_dir=system, drive_root=data, workspace_root=workspace, task_id="t-egress")
    return ctx, home


def test_read_file_user_files_masks_secret_bytes_with_disclosure(user_files_ctx):
    ctx, home = user_files_ctx
    (home / "notes.txt").write_text(
        f"remember: openrouter {OPENROUTER_KEY}\n{PEM_BLOCK}\nplain line\n",
        encoding="utf-8",
    )
    out = _read_file(ctx, "notes.txt", root="user_files")
    assert OPENROUTER_KEY not in out
    assert "b3BlbnNzaC1rZXktdjE" not in out
    assert "plain line" in out
    assert "SECRET_BYTES_MASKED" in out  # disclosure note, not a refusal
    assert not out.startswith("⚠️")  # the read itself succeeds


def test_read_file_user_files_plain_file_has_no_masking_note(user_files_ctx):
    ctx, home = user_files_ctx
    (home / "notes.txt").write_text("just prose, nothing secret\n", encoding="utf-8")
    out = _read_file(ctx, "notes.txt", root="user_files")
    assert "just prose, nothing secret" in out
    assert "SECRET_BYTES_MASKED" not in out


def test_read_file_non_user_files_roots_are_not_masked(user_files_ctx, tmp_path):
    # Scope pin: the egress seam is the user_files read path only. A task's own
    # drive legitimately carries tokens the task itself staged (e.g. for a
    # service it runs); masking there was not ratified (#447 В23 covers X1).
    ctx, _home = user_files_ctx
    drive_file = pathlib.Path(ctx.drive_root) / "task_drives" / ctx.task_id / "staged.txt"
    drive_file.parent.mkdir(parents=True)
    drive_file.write_text(f"token {GITHUB_TOKEN}\n", encoding="utf-8")
    out = _read_file(ctx, "staged.txt", root="task_drive")
    assert GITHUB_TOKEN in out


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


def test_search_user_files_masks_secret_bytes_on_both_egresses(user_files_ctx, monkeypatch):
    """#447 В23 seam a×b: search over the owner's home surfaces file CONTENT in
    match lines — the raw key must be masked on the rg path AND the Python
    fallback, with the same disclosure note as the read seam."""
    from ouroboros.tools.core import _code_search as _search_code

    ctx, home = user_files_ctx
    (home / "creds.txt").write_text(
        f"api entry openrouter {OPENROUTER_KEY} end\n", encoding="utf-8",
    )

    out_rg = _search_code(ctx, "openrouter", root="user_files")
    assert OPENROUTER_KEY not in out_rg, out_rg[:300]
    # The fixture guarantees a match: an empty result here means the rg binary
    # is genuinely unavailable AND the fallback was not reached — fail loudly
    # rather than skip the masking assert silently.
    assert "No matches" not in out_rg, out_rg[:300]
    assert "SECRET_BYTES_MASKED" in out_rg

    # Force the Python fallback by making rg unavailable.
    import ouroboros.code_search_rg as rg_mod

    def _raise(*a, **k):
        raise FileNotFoundError("rg unavailable (forced)")

    monkeypatch.setattr(rg_mod, "search_with_rg", _raise)
    out_fb = _search_code(ctx, "openrouter", root="user_files")
    assert OPENROUTER_KEY not in out_fb, out_fb[:300]
    assert "creds.txt" in out_fb  # the match itself is still reported
    assert "SECRET_BYTES_MASKED" in out_fb


def test_search_non_user_files_root_is_not_masked(user_files_ctx):
    from ouroboros.tools.core import _code_search as _search_code

    ctx, _home = user_files_ctx
    (ctx.repo_dir / "sample.txt").write_text(
        "fixture openrouter sk-or-aaaaaaaabbbbbbbbccccccccdddddddd here\n", encoding="utf-8",
    )
    out = _search_code(ctx, "openrouter", root="system_repo")
    assert "SECRET_BYTES_MASKED" not in out
