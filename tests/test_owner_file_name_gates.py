"""The owner's own files survive every gate that used to judge their NAME.

Capability-preservation acceptance for the phase that removed the credential
suffix tables, the name regex, the dotted-component default-deny and the
reviewer suffix list (owner answers Q6 of batch 2 and 3=A / 4=A / 5=A of batch
3). Every gate here refused ordinary owner work before: a Keynote deck named
``deck.key``, a report inside ``~/.codex``, a document in ``~/Library``.

The point of the module is the POSITIVE path on product-shaped geometry, with
the surviving refusals asserted beside it: real key material is still caught,
now on content evidence, and a real credential directory is still refused with
the rule named in the row the owner sees.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import zipfile
from io import BytesIO
from types import SimpleNamespace

import pytest


def _keynote_shaped_bytes() -> bytes:
    """A few KB of real zip container, the shape a .key deck actually has."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Index/Document.iwa", b"\x00\x01\x02\x03" * 512)
        archive.writestr("Metadata/BuildVersionHistory.plist", "<plist/>")
        archive.writestr("preview.jpg", b"\xff\xd8\xff\xe0" + b"\x11" * 2048)
    return buffer.getvalue()


PEM_PRIVATE_KEY = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gt\n"
    "-----END OPENSSH PRIVATE KEY-----\n"
)


def _init_repo(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=str(path), check=True)
    (path / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(path), check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@local", "commit", "-qm", "init"],
        cwd=str(path), check=True,
    )


# --- (a) ingress: the owner attaches their own files -------------------------


def test_owner_attaches_a_key_deck_and_files_from_dotted_and_library_folders(tmp_path):
    from ouroboros.artifacts import stage_task_attachments

    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    (home / "Library" / "Mobile Documents").mkdir(parents=True)
    deck = home / "Desktop" / "custom.key"
    deck.parent.mkdir(parents=True)
    deck.write_bytes(_keynote_shaped_bytes())
    report = home / ".codex" / "report.md"
    report.write_text("# findings\n", encoding="utf-8")
    icloud = home / "Library" / "Mobile Documents" / "deck.pdf"
    icloud.write_bytes(b"%PDF-1.4\n" + b"0" * 1024)

    drive = tmp_path / "data"
    drive.mkdir()
    manifest = stage_task_attachments(
        drive, "task-owner-files", [str(deck), str(report), str(icloud)],
    )

    assert [row["status"] for row in manifest] == ["staged", "staged", "staged"], manifest
    for row in manifest:
        assert row["mime"] and row["abs_path"]
        assert pathlib.Path(row["abs_path"]).is_file()
    assert {row["label"] for row in manifest} == {"custom.key", "report.md", "deck.pdf"}


def test_a_real_credential_directory_is_still_refused_and_names_its_rule(tmp_path, monkeypatch):
    from ouroboros.artifacts import stage_task_attachments

    ssh = tmp_path / "home" / ".ssh"
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: ssh.parent))
    ssh.mkdir(parents=True)
    key = ssh / "id_rsa"
    key.write_text(PEM_PRIVATE_KEY, encoding="utf-8")

    drive = tmp_path / "data"
    drive.mkdir()
    manifest = stage_task_attachments(drive, "task-owner-ssh", [str(key)])

    assert manifest[0]["status"] == "rejected"
    assert manifest[0]["reason"] == "secret_source"
    assert manifest[0]["rule"] == "path is hidden or credential-like (owner credential location)"


@pytest.mark.serial
@pytest.mark.parametrize("mode", ["light", "advanced", "pro", "cyber_pro"])
@pytest.mark.parametrize("relative, below_cyber_status", [
    ("x/prod.env", "rejected"),
    ("x/.env.local", "rejected"),
    ("Desktop/deck.key", "staged"),
    (".codex/report.md", "staged"),
    ("x/.env.example", "staged"),
])
def test_owner_file_ingest_routes_preserve_dotenv_policy_and_ordinary_files(
    tmp_path, monkeypatch, mode, relative, below_cyber_status,
):
    """Path selection and the real multipart/WS ingest share the retained rule.

    Cyber Pro keeps the owner's explicit sensitive-input capability. Elsewhere
    dotenv refusals name their rule, while a real ZIP deck, a dotted-directory
    report and a dotenv example remain physically readable staged inputs.
    """
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import config
    from ouroboros.artifacts import stage_task_attachments
    from ouroboros.gateway import files, ws
    from ouroboros.gateway.tasks import _render_attachment_lines

    drive = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(drive))
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.setattr(config, "get_runtime_mode", lambda: mode)
    monkeypatch.setattr(ws, "DATA_DIR", drive)
    source = tmp_path / "home" / relative
    source.parent.mkdir(parents=True)
    payload = _keynote_shaped_bytes() if source.suffix == ".key" else b"VALUE=fixture\n"
    source.write_bytes(payload)
    app = Starlette(routes=[Route("/api/chat/upload", files.api_chat_upload, methods=["POST"])])
    with TestClient(app) as client:
        response = client.post(
            "/api/chat/upload", files={"file": (source.name, payload, "application/octet-stream")},
        )
    assert response.status_code == 200, response.text
    upload = response.json()
    assert upload["ok"] is True and upload["filename"].endswith("_" + source.name)
    assert pathlib.Path(upload["path"]).read_bytes() == payload
    routes = {"path": [str(source)], "upload": ws._chat_attachment_uploads([upload])}
    manifests = {
        route: stage_task_attachments(drive, f"owner-ingest-{route}", attachments)
        for route, attachments in routes.items()
    }
    expected = "staged" if mode == "cyber_pro" else below_cyber_status
    for route, manifest in manifests.items():
        assert len(manifest) == 1, (route, manifest)
        row = manifest[0]
        assert row["status"] == expected, (mode, route, manifests)
        rendered = _render_attachment_lines(manifest)
        if expected == "rejected":
            assert row["reason"] == "secret_source"
            assert "dotenv secret" in row["rule"] and source.name in row["rule"]
            assert f"rule: {row['rule']}" in rendered
        else:
            assert row["reason"] == "" and row["mime"] and row["abs_path"]
            assert pathlib.Path(row["abs_path"]).read_bytes() == payload
            assert "script/process path:" in rendered


# --- (b) export: the same deck declared as a process output ------------------


def test_owner_declares_a_key_deck_as_a_process_output(tmp_path):
    from ouroboros.tools.shell_outputs import _protected_output_source_reason

    ctx = SimpleNamespace(repo_dir=str(tmp_path / "repo"), drive_root=str(tmp_path / "data"))
    project = tmp_path / "project"
    project.mkdir()
    deck = project / "deck.key"
    deck.write_bytes(_keynote_shaped_bytes())

    assert _protected_output_source_reason(ctx, deck, "task_drive", set()) == ""
    dotenv = project / ".env"
    dotenv.write_text("TOKEN=x\n", encoding="utf-8")
    assert "credential-like output .env" in _protected_output_source_reason(
        ctx, dotenv, "task_drive", set(),
    )


@pytest.mark.serial
def test_a_declared_key_deck_output_is_actually_registered(tmp_path):
    """The predicate saying "no reason to refuse" is not the capability.

    This drives the registration path a real `run_command(outputs=[...])` runs,
    so the acceptance covers what the owner sees: a canonical artifact record for
    the deck in the task artifact store, no ``ARTIFACT_OUTPUT_ERROR`` in the
    rendered result, and a published tool result that still classifies as ``OK``,
    which is what keeps the task from being degraded by the export. The dotenv
    negative stays beside it: a declared dotenv tail is still refused, now
    before the command runs (#1797), so nothing is written or exported.
    """
    from ouroboros.artifacts import registered_task_artifact
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.shell import _run_shell
    from ouroboros.tools.tool_result import (
        ToolResult,
        _install_tool_result_sidecar,
        _published_tool_result,
        _restore_tool_result_sidecar,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    drive = tmp_path / "data"
    drive.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="t-owner-deck")
    build_deck = (
        "import zipfile\n"
        "with zipfile.ZipFile('deck.key', 'w') as archive:\n"
        "    archive.writestr('Index/Document.iwa', b'\\x00\\x01\\x02\\x03' * 512)\n"
    )

    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        rendered = _run_shell(
            ctx, [sys.executable, "-c", build_deck], cwd="task_drive", outputs=["deck.key"],
        )
        published = _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)

    assert "ARTIFACT_OUTPUT_ERROR" not in rendered, rendered
    assert "ARTIFACT_OUTPUTS" in rendered, rendered
    assert "registered output" in rendered and "artifact_store:" in rendered
    assert isinstance(published, ToolResult)
    assert (published.code, published.status) == ("OK", "ok")
    assert published.meta.get("artifact_registered") is True
    record = registered_task_artifact(drive, ctx.task_id, "deck.key")
    assert record and record["kind"] == "process_output"
    assert record["name"] == "deck.key" and record["size"] > 0 and record["sha256"]
    assert zipfile.is_zipfile(record["path"])

    # The negative control on the SAME path, so the assertions above are not
    # vacuous: a declared credential leaf still refuses, before the process
    # starts, while the deck above passes the same pre-launch check.
    build_dotenv = "open('prod.env', 'w').write('TOKEN=x\\n')\n"
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        refused = _run_shell(
            ctx, [sys.executable, "-c", build_dotenv], cwd="task_drive", outputs=["prod.env"],
        )
        refused_result = _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)

    assert "RESOURCE_POLICY_BLOCKED" in refused and "nothing ran" in refused
    assert "credential-like output prod.env" in refused and "dotenv secret" in refused
    assert isinstance(refused_result, ToolResult)
    assert refused_result.code == "RESOURCE_POLICY_BLOCKED"
    assert refused_result.meta.get("operation_outcome") == "completed_no_effect"
    assert not (ctx.task_drive_root() / "prod.env").exists()
    assert registered_task_artifact(drive, ctx.task_id, "prod.env") is None


# --- (c) delegated snapshot: transport rules stay, name authority is gone ----


def test_delegated_snapshot_keeps_a_text_deck_and_states_the_binary_boundary(tmp_path):
    from ouroboros.workspace_patch_capture import untracked_capture_veto_reason

    root = tmp_path / "run"
    _init_repo(root)
    (root / "deck.key").write_text("Keynote outline, plain text\n", encoding="utf-8")
    assert untracked_capture_veto_reason(root, "deck.key") == ""

    # The honest boundary: the name stopped mattering, the TRANSPORT rules of the
    # patch did not. A binary deck is still excluded, as "binary file" and never
    # as "private key or certificate".
    binary = tmp_path / "binary-run"
    _init_repo(binary)
    (binary / "deck.key").write_bytes(_keynote_shaped_bytes())
    reason = untracked_capture_veto_reason(binary, "deck.key")
    assert reason == "binary file"


# --- (d) negative control: key material under an innocent name ---------------


def test_key_material_under_an_innocent_name_is_vetoed_in_the_patch_lane(tmp_path):
    from ouroboros.workspace_patch_capture import untracked_capture_veto_reason
    from ouroboros.headless import write_workspace_patch_artifacts

    root = tmp_path / "repo"
    _init_repo(root)
    (root / "notes.txt").write_text(PEM_PRIVATE_KEY, encoding="utf-8")
    (root / "deck.key").write_text("Keynote outline, plain text\n", encoding="utf-8")

    assert untracked_capture_veto_reason(root, "notes.txt") == (
        "private key material (PEM private-key header)"
    )
    _artifacts, manifest = write_workspace_patch_artifacts(root, tmp_path / "artifacts", task={})
    excluded = {item["path"]: item["reason"] for item in manifest["untracked_excluded"]}
    assert excluded == {"notes.txt": "private key material (PEM private-key header)"}


def test_key_material_under_an_innocent_name_is_vetoed_in_the_attach_snapshot_lane(tmp_path):
    from ouroboros.project_sources import attach_snapshot_init

    folder = tmp_path / "attached"
    folder.mkdir()
    (folder / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (folder / "deck.key").write_text("Keynote outline, plain text\n", encoding="utf-8")
    (folder / "notes.txt").write_text(PEM_PRIVATE_KEY, encoding="utf-8")

    error, skipped = attach_snapshot_init(folder)
    assert error == ""
    assert skipped == ["notes.txt"]
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=str(folder), capture_output=True, text=True,
    ).stdout.split()
    assert "deck.key" in tracked and "notes.txt" not in tracked


@pytest.mark.serial
def test_key_material_under_an_innocent_name_is_vetoed_in_the_coop_checkpoint_lane(
    tmp_path, monkeypatch,
):
    from ouroboros.coop_checkpoint import checkpoint_commit_coop_roots
    from ouroboros.task_results import write_task_result

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(projects_root))
    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    tree = projects_root / "coop_root1"
    _init_repo(tree)
    write_task_result(data, "root1", "failed", reason_code="budget_exhausted", title="Deck")
    write_task_result(
        data, "child1", "failed",
        delegation_role="subagent", parent_task_id="root1", root_task_id="root1",
        task_constraint={"mode": "acting_subagent", "surface": "external_workspace",
                         "write_root": str(tree)},
    )
    (tree / "deck.key").write_text("Keynote outline, plain text\n", encoding="utf-8")
    (tree / "notes.txt").write_text(PEM_PRIVATE_KEY, encoding="utf-8")

    receipts = checkpoint_commit_coop_roots(data, "root1", title="Deck")
    assert [receipt["skipped_sensitive"] for receipt in receipts] == [
        [{"path": "notes.txt", "reason": "private key material (PEM private-key header)"}]
    ]
    committed = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"], cwd=str(tree),
        capture_output=True, text=True,
    ).stdout.split()
    assert "deck.key" in committed and "notes.txt" not in committed


# --- (e) what the owner reads, and the inspection carve ----------------------


def test_a_rejected_attachment_line_shows_the_rule_not_only_the_code():
    from ouroboros.gateway.tasks import _render_attachment_lines

    rendered = _render_attachment_lines([
        {"ordinal": 0, "status": "rejected", "reason": "secret_source", "label": "id_rsa",
         "rule": "credential/control directory component '.ssh'"},
        {"ordinal": 1, "status": "rejected", "reason": "source_missing", "label": "missing"},
    ])
    assert "rule: credential/control directory component '.ssh'" in rendered
    assert "- missing: rejected (reason=source_missing, ordinal=1)" in rendered
    assert "rule:" not in rendered.splitlines()[-1]


@pytest.mark.parametrize("argv", [
    ["grep", "-n", "x >= 1", "f"],
    ["rg", "a->b", "."],
    # `>=` as its OWN token: the redirect grammar used to accept it because the
    # token pattern only refused a following `&`, `|` or `-`.
    ["rg", ">=", "."],
])
def test_a_comparison_sign_no_longer_makes_an_inspection_a_write(argv):
    """Disclosed micro-residual of closing the standalone `>=` token: a redirect
    whose TARGET name begins with `=` (`cmd >=out`, writing the file `=out`) is
    no longer reported as write shape. No product path names a file that way,
    and the comparison form it buys back is the one owners actually type."""
    from ouroboros.tools.write_shape import non_interpreter_write_shape

    assert non_interpreter_write_shape(argv, argv, argv[0]) is False


@pytest.mark.parametrize("argv", [
    ["tee", "f"],
    ["grep", "-n", "x", "f", ">", "out.txt"],
    ["cmd", ">", "out"],
    ["cmd", ">>", "out"],
    # `>&` is a stdout+stderr redirect, never a comparison: a standalone token
    # of it stays a write channel in both lanes.
    ["ls", ">&", "out.txt"],
    ["ls", ">&1"],
])
def test_a_real_write_channel_is_still_write_shaped(argv):
    from ouroboros.tools.write_shape import non_interpreter_write_shape

    assert non_interpreter_write_shape(argv, argv, argv[0]) is True


def test_a_standalone_stdout_stderr_redirect_token_is_write_shaped_as_a_string():
    from ouroboros.tools.write_shape import shell_has_write_indicator

    assert shell_has_write_indicator("cmd >& out.txt") is True


def test_a_standalone_comparison_token_is_not_write_shaped_as_a_string():
    from ouroboros.tools.write_shape import shell_has_write_indicator

    assert shell_has_write_indicator("rg '>=' .") is False


@pytest.mark.parametrize("argv", [
    ["grep", "-n", "a > b", "f"],
    ["rg", "actual>0", "tests/"],
])
def test_disclosed_residual_a_quoted_operand_with_a_redirect_shape_stays_write_shaped(argv):
    """shlex strips the quotes before any guard sees the line, so the redirect
    grammar still matches inside the operand. Only the comparison forms (>=,
    -> and =>) were closed. Asserted as EXPECTED behaviour so the remaining
    gap is visible rather than assumed fixed."""
    from ouroboros.tools.write_shape import non_interpreter_write_shape

    assert non_interpreter_write_shape(argv, argv, argv[0]) is True
