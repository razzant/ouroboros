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


def test_mask_secret_bytes_masks_entropy_formats():
    text = f"config a\nkey={OPENROUTER_KEY}\nAuthorization: Bearer {GITHUB_TOKEN}\nplain tail"
    masked, count = mask_secret_bytes(text)
    assert OPENROUTER_KEY not in masked
    assert GITHUB_TOKEN not in masked
    assert count >= 2
    assert "***" in masked
    # Non-secret content survives byte-for-byte.
    assert "config a" in masked and "plain tail" in masked


def test_mask_secret_bytes_masks_pem_block():
    masked, count = mask_secret_bytes(f"prefix\n{PEM_BLOCK}\nsuffix")
    assert "PRIVATE KEY" not in masked
    assert "b3BlbnNzaC1rZXktdjE" not in masked
    assert count == 1
    assert masked.startswith("prefix\n") and masked.endswith("\nsuffix")


def test_mask_secret_bytes_masks_unterminated_pem_to_end():
    # A read slice can cut the file before the END marker; the tail is still
    # key material and must not survive.
    head, _, _ = PEM_BLOCK.partition("-----END")
    masked, count = mask_secret_bytes(f"prefix\n{head}")
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
