"""capinv-447 WS2-a: credential-shape vocabulary is a leaf module; root READ
authorization of user_files is location-only (В23=A), mutation keeps the shape
deny, children stay location-denied, and denied children get typed disclosure
instead of invisibility."""

import os
import pathlib
import subprocess
import sys
import types

from ouroboros.tool_access import user_files_path_block_reason


def _root_ctx(tmp_path):
    return types.SimpleNamespace(
        drive_root=str(tmp_path / "home" / "Ouroboros" / "data"),
        repo_dir=str(tmp_path / "home" / "Ouroboros" / "repo"),
        workspace_root="",
        workspace_mode="",
        task_constraint=None,
        task_metadata={},
    )


def _home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    return home


# ── root READ authorization is location-only ─────────────────────────────────

def test_root_read_list_search_are_location_only(tmp_path, monkeypatch):
    """В23=A: the root principal reads the owner's home in full — credential-
    SHAPED names are not a read-authorization signal (bytes are masked at
    egress). Control-plane and outside-home LOCATION denials stay."""
    home = _home(tmp_path, monkeypatch)
    ctx = _root_ctx(tmp_path)
    for op in ("read", "list", "search"):
        for rel in (".ssh/id_rsa", ".env", "secrets.json", ".aws/credentials",
                    ".bash_history", "proj/api_key.txt", "Desktop/cert.pem"):
            assert user_files_path_block_reason(ctx, home / rel, operation=op) == "", (op, rel)
    # Location denials survive for reads: control-plane drives + outside-home.
    assert user_files_path_block_reason(
        ctx, home / "Ouroboros" / "data" / "settings.json", operation="read"
    ) != ""
    assert "outside user home" in user_files_path_block_reason(
        ctx, tmp_path / "elsewhere" / "x.txt", operation="read"
    )


def test_mutation_operations_keep_credential_shape_deny(tmp_path, monkeypatch):
    """Writes/edits (and unknown-operation callers, fail-closed) keep the
    pre-capinv-447 shape gate: overwriting ~/.bashrc / ~/.ssh material is a
    persistence hazard, not a read."""
    home = _home(tmp_path, monkeypatch)
    ctx = _root_ctx(tmp_path)
    for op in ("", "write", "edit"):
        assert "credential-like" in user_files_path_block_reason(
            ctx, home / "Desktop" / "Credentials.json", operation=op
        )
        assert "hidden or credential-like" in user_files_path_block_reason(
            ctx, home / ".ssh" / "authorized_keys", operation=op
        )
    # Benign dotted project components stay writable under the allowlist.
    assert user_files_path_block_reason(ctx, home / ".github" / "ci.yml", operation="write") == ""


def test_import_boundary_root_read_decision_never_touches_credential_shapes(tmp_path):
    """The root READ decision path in tool_access must not import
    ouroboros.credential_shapes (ledger WS2-a import-boundary contract). The
    hook is proven non-vacuous by the mutation branch tripping it."""
    code = r"""
import pathlib, sys, tempfile, types, os

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "ouroboros.credential_shapes":
            raise ImportError("credential_shapes reached from authorization")
        return None

sys.meta_path.insert(0, _Block())
home = tempfile.mkdtemp()
os.environ["OUROBOROS_USER_FILES_ROOT"] = home
import ouroboros.tool_access as ta
assert "ouroboros.credential_shapes" not in sys.modules, "module-level import leak"
ctx = types.SimpleNamespace(
    drive_root=os.path.join(home, "Ouroboros", "data"),
    repo_dir=os.path.join(home, "Ouroboros", "repo"),
    workspace_root="", workspace_mode="", task_constraint=None, task_metadata={},
)
for op in ("read", "list", "search"):
    assert ta.user_files_path_block_reason(
        ctx, pathlib.Path(home) / ".ssh" / "id_rsa", operation=op
    ) == ""
assert "ouroboros.credential_shapes" not in sys.modules, "read decision imported shapes"
# Non-vacuity: the MUTATION branch does consult the shapes, so the hook fires.
try:
    ta.user_files_path_block_reason(ctx, pathlib.Path(home) / "x.pem", operation="write")
except ImportError:
    print("OK")
else:
    raise SystemExit("import hook never intercepted anything (vacuous test)")
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env={**os.environ},
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


# ── consolidation keeps the child (subagent) contract byte-for-byte ──────────

def test_child_secret_shape_contract_preserved():
    from ouroboros.tools.core import (
        _is_subagent_secret_data_path,
        _is_subagent_secret_repo_path,
    )

    for norm in ("state/settings.json", "foo.pem", "secrets/x.txt",
                 ".env.production", "my_api_key.json", "keys.json"):
        assert _is_subagent_secret_data_path(norm), norm
    for norm in ("memory/identity.md", "logs/progress.jsonl", "notes.txt"):
        assert not _is_subagent_secret_data_path(norm), norm
    for norm in ("settings.json", ".git/config", "deploy.key", "token.yaml"):
        assert _is_subagent_secret_repo_path(norm), norm
    for norm in ("README.md", "src/main.py", "docs/token_economics.md"):
        assert not _is_subagent_secret_repo_path(norm), norm


def test_shape_vocabulary_is_single_sourced():
    from ouroboros import credential_shapes as cs

    assert cs.CREDENTIAL_NAME_RE.search("api_key.json")
    assert cs.CREDENTIAL_NAME_RE.search("my-token")
    assert not cs.CREDENTIAL_NAME_RE.search("README.md")
    assert "settings.json" in cs.SUBAGENT_CREDENTIAL_FILE_NAMES
    assert ".ssh" in cs.CREDENTIAL_COMPONENT_NAMES
    assert cs.user_files_mutation_shape_reason(
        pathlib.Path("/h/.ssh/id_rsa"), pathlib.Path("/h")
    ) != ""
    assert cs.user_files_mutation_shape_reason(
        pathlib.Path("/h/Desktop/report.html"), pathlib.Path("/h")
    ) == ""


# ── root listing no longer hides credential-shaped entries ───────────────────

def test_user_files_listing_shows_credential_shaped_entries_to_root(tmp_path, monkeypatch):
    from ouroboros.tools.core import _list_user_files_dir

    home = _home(tmp_path, monkeypatch)
    (home / ".ssh").mkdir()
    (home / "secrets.json").write_text("{}", encoding="utf-8")
    (home / "notes.txt").write_text("n", encoding="utf-8")
    (home / "Ouroboros" / "data").mkdir(parents=True)  # control-plane drive
    ctx = _root_ctx(tmp_path)

    items = _list_user_files_dir(ctx, home, home)
    rendered = "\n".join(items)
    assert ".ssh/" in rendered
    assert "secrets.json" in rendered
    assert "notes.txt" in rendered
    # The control-plane workspace parent stays omitted, with the typed marker.
    assert "Ouroboros" not in rendered.replace("Ouroboros listing", "")
    assert "hidden/control" in rendered


# ── denied children get typed disclosure, not invisibility ───────────────────

def test_child_search_reports_omitted_secret_files(tmp_path, monkeypatch):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools import core as core_mod
    import ouroboros.code_search_rg as rg_mod

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "notes.txt").write_text("needle here", encoding="utf-8")
    (repo / "secrets.json").write_text('"needle"', encoding="utf-8")

    def _no_rg(*a, **k):
        raise RuntimeError("rg disabled for fallback test")

    monkeypatch.setattr(rg_mod, "search_with_rg", _no_rg)
    ctx = types.SimpleNamespace(
        drive_root=str(tmp_path / "data"),
        repo_dir=str(repo),
        workspace_root="",
        workspace_mode="",
        task_constraint=TaskConstraint(mode="local_readonly_subagent"),
        task_metadata={},
    )

    result = core_mod._code_search(ctx, "needle", root="active_workspace", path=".")
    assert "secrets.json" not in result
    assert "notes.txt" in result
    assert "secret/control file(s) omitted from this subagent's search" in result
