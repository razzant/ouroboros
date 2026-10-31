"""A skill-payload builder gets the serving host's skill contract by reference (#818).

The files are the ones skill review judges a payload by (one list), named with their byte
digest and serving identity: a commit only when the bytes are that commit's blob, otherwise
dirty or unknown, an unreadable file a named gap. Nothing enters the payload snapshot, its
inventory, hash, capture or apply; a declared-input run still gets it; an exact retry
replays the recorded bytes.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess

import pytest

from ouroboros.skill_catalogue import skill_contract_references
from ouroboros.tools.governance_context import (
    CREATING_SKILLS_PATH, SKILL_PLUGIN_API_PATH, SKILL_UI_VALIDATION_PATH,
)
from tests.test_delegated_skill_payload import (  # noqa: F401 -- autouse transport fixture
    _exact_payload_start,
    _owned_gateway_uses_each_test_transport,
    _payload_ctx,
    _seed_skill,
    _StartStub,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = [CREATING_SKILLS_PATH, SKILL_PLUGIN_API_PATH, SKILL_UI_VALIDATION_PATH]


def _git(root, *args):
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()


def _serving(root: pathlib.Path) -> str:
    """A serving checkout holding the real contract files at one commit."""
    for rel in CONTRACT:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, root / rel)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "serving")
    return _git(root, "rev-parse", "HEAD")


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_review_and_builders_share_one_host_contract_list():
    from ouroboros.tools.governance_context import governance_context

    assert CONTRACT == ["docs/CREATING_SKILLS.md", "ouroboros/contracts/plugin_api.py",
                        "ouroboros/extension_ui_validation.py"]
    review = governance_context(REPO, surface="skill", layer="skill", delivery="native",
                                manifest_facts={"type": "extension"})
    dispositions = {row["path"]: row["disposition"] for row in review.manifest}
    assert dispositions[CREATING_SKILLS_PATH] == "navigation"
    assert dispositions[SKILL_PLUGIN_API_PATH] == dispositions[SKILL_UI_VALIDATION_PATH] == "required"
    rows = [line for line in skill_contract_references(REPO).splitlines() if line.startswith("- ")]
    assert [row.split(": ", 1)[0][2:] for row in rows] == [str(REPO / path) for path in CONTRACT]


def test_committed_bytes_name_the_serving_commit_and_the_text_grants_reading(tmp_path):
    serving = tmp_path / "serving"
    head = _serving(serving)
    text = skill_contract_references(serving)
    assert text.startswith("\n\nHOST SKILL CONTRACT (read-only references to the serving host")
    assert "You may read them although they are outside your workspace; never copy them into the payload" in text
    assert "say so in your answer" in text
    for rel in CONTRACT:
        assert f"- {serving / rel}: sha256={_digest(serving / rel)}; serving commit {head}" in text
    assert "register_ui_tab" not in text  # referenced, never a second copied contract


def test_dirty_missing_and_unknown_never_claim_a_commit(tmp_path):
    serving = tmp_path / "serving"
    head = _serving(serving)
    guide, api, ui = (serving / rel for rel in CONTRACT)
    guide.write_text(guide.read_text(encoding="utf-8") + "\nlocal edit\n", encoding="utf-8")
    api.unlink()
    text = skill_contract_references(serving)
    assert f"- {guide}: sha256={_digest(guide)}; dirty: these bytes are not the blob of serving commit {head}" in text
    assert f"- {api}: unavailable (FileNotFoundError" in text and "ask your host rather than reconstruct it" in text
    assert f"- {ui}: sha256={_digest(ui)}; serving commit {head}" in text

    for plain in (tmp_path / "plain", serving / "nested"):  # no repository, or another repository's folder
        for rel in CONTRACT:
            (plain / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / rel, plain / rel)
        unknown = skill_contract_references(plain)
        assert unknown.count("serving identity unknown") == len(CONTRACT), unknown
        assert "; serving commit" not in unknown and "; dirty:" not in unknown


def test_identity_describes_the_bytes_read_never_the_file_read_later(tmp_path):
    """Digest and identity come from one read: bytes that are not the blob never borrow the commit,
    and the committed bytes keep it even when the file on disk changed after they were read."""
    from ouroboros.skill_catalogue import _serving_identities

    serving = tmp_path / "serving"
    head = _serving(serving)
    rel = CONTRACT[0]
    committed = (serving / rel).read_bytes()
    assert _serving_identities(serving, {rel: committed + b"\nedited later\n"})[rel].startswith("dirty")
    (serving / rel).write_bytes(committed + b"\nedited after the read\n")
    assert _serving_identities(serving, {rel: committed}) == {rel: f"serving commit {head}"}


def _payload_world(tmp_path, monkeypatch, input_sources):
    ctx = _payload_ctx(tmp_path, monkeypatch)
    ctx.task_contract = {"objective": "Edit the skill", "input_sources": input_sources}
    head = _serving(pathlib.Path(ctx.repo_dir))  # the checkout this context's server imports
    skill = _seed_skill(tmp_path / "data")
    return ctx, head, skill


@pytest.mark.parametrize("input_sources", ["shared", "declared"])
def test_a_payload_builder_gets_the_contract_and_its_copy_stays_pure(tmp_path, monkeypatch, input_sources):
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.delegate_integration import payload_content_hash

    ctx, head, skill = _payload_world(tmp_path, monkeypatch, input_sources)
    before = payload_content_hash(skill)
    seen: dict = {}
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _StartStub(seen))
    delegate._CUSTODY.clear()
    try:
        payload = json.loads(_exact_payload_start(ctx, "edit the skill", root="skill_payload", bucket="external",
                                                  skill_name="alpha"))
        assert payload["status"] == "started", payload
        instructions = seen["request"]["instructions"]
        assert instructions.count("HOST SKILL CONTRACT") == 1  # even when the skill index is suppressed
        for rel in CONTRACT:
            assert f"- {pathlib.Path(ctx.repo_dir) / rel}: sha256=" in instructions
        assert f"serving commit {head}" in instructions
        assert "PROJECT ENVIRONMENT" not in instructions and ".review-evidence" not in instructions
        assert "no evidence folder is retained for this run" in instructions and "safe_test" not in instructions
        copy = pathlib.Path(seen["request"]["scope"]["root"])
        names = {path.name for path in copy.rglob("*") if ".git" not in path.parts}
        assert not names & {"CREATING_SKILLS.md", "plugin_api.py", "extension_ui_validation.py"}
        assert payload_content_hash(copy) == before == payload_content_hash(skill)
    finally:
        delegate._CUSTODY.clear()


def test_an_exact_payload_retry_replays_the_recorded_contract(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate
    from ouroboros import skill_catalogue
    from ouroboros.gateways import claudexor as gw
    from ouroboros.subagent_runtime import exact_start

    ctx, _head, _skill = _payload_world(tmp_path, monkeypatch, "shared")
    seen: dict = {}

    class _UnknownFirst(_StartStub):
        def start_run(self, request, *, idempotency_key=""):
            self._seen.setdefault("requests", []).append(request)
            if len(self._seen["requests"]) == 1:
                raise gw.ClaudexorUnavailable("daemon_unreachable", "connection reset", status_code=503)
            return {"runId": "run-p1", "runDir": "/tmp/run-p1"}

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _UnknownFirst(seen))
    delegate._CUSTODY.clear()
    try:
        unknown = json.loads(_exact_payload_start(ctx, "edit the skill", root="skill_payload", bucket="external",
                                                  skill_name="alpha"))
        token = unknown.get("pending_invocation_id")
        assert token, unknown

        def forbidden(*_args, **_kwargs):
            raise AssertionError("an exact retry replays recorded bytes; it reads no contract again")

        monkeypatch.setattr(skill_catalogue, "skill_contract_references", forbidden)
        delegate._CUSTODY.clear()
        # A retry carries no selector or snapshot: it replays the recorded invocation.
        payload = json.loads(exact_start(ctx, "edit the skill", {"retry_of": token}).text)
        assert payload["status"] == "started", payload
        first, replay = seen["requests"]
        assert replay == first and "HOST SKILL CONTRACT" in replay["instructions"]
    finally:
        delegate._CUSTODY.clear()
