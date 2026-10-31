"""Keyless real-consumer proofs for Tier 6. Scripted choices prove wiring, not LLM quality."""
from __future__ import annotations

import json
import hashlib
import http.server
from pathlib import Path
import sys
import threading
from unittest.mock import patch
import zipfile

import pytest

import tests.test_skill_smoke_official as smoke
from tests.skill_smoke_author import run_author_review
from tests.system_e2e.harness import (
    LANE_MOCK, MOCK_SLUG, ScriptedStubModel, classify_call, require_lane, skill_review_clean_text,
)

pytestmark = [pytest.mark.integration]


@pytest.fixture(autouse=True)
def _mock_lane():
    require_lane(LANE_MOCK)


REPO = Path(__file__).resolve().parents[2]
NAME = "smoke_probe"
ARGUMENT = "plugin.py has no network calls or process execution; the claimed shell call does not exist."


def verdict(red=False, *, real_bug=False):
    rows = json.loads(skill_review_clean_text())
    if red:
        row = next(row for row in rows if row["item"] == "bug_hunting")
        row.update(verdict="FAIL", severity="critical", reason=(
            "plugin.py register always divides by zero." if real_bug else "Claims an unsafe shell call in plugin.py."))
    return json.dumps(rows)


class AuthorModel(ScriptedStubModel):
    def __init__(self, script, verdicts):
        super().__init__(script=script)
        self.verdicts = list(verdicts)
        self.review_calls = []

    def _answer(self, body, seq):
        if classify_call(body) == "skill_review":
            self.review_calls.append(body)
            assert self.verdicts, "unexpected paid reassessment"
            return "skill_review", {"role": "assistant", "content": self.verdicts.pop(0)}
        return super()._answer(body, seq)


def call(**kwargs):
    return {"tool": "skill_review", "arguments": {"skill": NAME, **kwargs}}


@pytest.fixture
def source(tmp_path):
    data = tmp_path / "source"
    payload = data / "skills" / "ouroboroshub" / NAME
    payload.mkdir(parents=True)
    (payload / "SKILL.md").write_text("""---
name: smoke_probe
description: An inert keyless review fixture.
type: extension
entry: plugin.py
plugin_api: "2.0"
---
# Smoke probe
No external effects.
""")
    (payload / "plugin.py").write_text(
        "from pathlib import Path\nPath(__file__).with_name('imported.marker').write_text('IMPORTED')\n"
        "raise RuntimeError('SMOKE_PAYLOAD_IMPORTED_WITH_KEY')\n\ndef register(api):\n    pass\n")
    return data


def assert_not_imported(payload):
    assert not (payload / "imported.marker").exists(), "payload was imported in the key-bearing server"


def consume(tmp_path, source, script, verdicts, *, error="", dependency=False):
    """Invoke the SAME CI test entry; replace only catalog input and model transport."""
    result = {}
    with AuthorModel(script, verdicts) as model:
        def transport(**kwargs):
            kwargs.update(model=MOCK_SLUG, timeout_sec=600,
                          credential=("OPENAI_COMPATIBLE_API_KEY", "stub-key-not-a-credential"),
                          settings_extra={"OPENAI_COMPATIBLE_BASE_URL": model.base_url})
            result.update(run_author_review(**kwargs))
            return result
        with patch.object(smoke, "DATA_DIR", source), patch.object(smoke, "_REVIEW_MODEL", MOCK_SLUG), \
                patch.object(smoke, "_REVIEW_EXPECTED_KEYS", {NAME: set()}), \
                patch.object(smoke, "DEPS_SKILLS", {NAME} if dependency else set()), \
                patch("tests.skill_smoke_author.run_author_review", transport), \
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "not-a-real-key"}):
            if error:
                with pytest.raises(AssertionError, match=error):
                    smoke.test_review_grants_and_enable(NAME, None, lambda _slug: None, tmp_path)
            else:
                smoke.test_review_grants_and_enable(NAME, None, lambda _slug: None, tmp_path)
        assert model.script_consumed()
        assert not model.verdicts
        assert_not_imported(tmp_path / "author/data/skills/ouroboroshub" / NAME)
        return result, model


def test_s41_clean_review_runs_in_ordinary_task_and_stops_before_enable(tmp_path, source):
    result, model = consume(tmp_path, source, [call()], [verdict()])
    assert result["server_stopped"] and result["review"]["status"] == "clean"
    assert len(model.review_calls) == 1


def test_s41_declared_local_wheel_installs_only_after_review(tmp_path, source):
    """Real review/install owners; only resource bytes and LLM transport are fixtures."""
    wheel = tmp_path / "smoke_dep-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, text in {
            "smoke_dep/__init__.py": "VALUE = 41\n",
            "smoke_dep-1.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: smoke-dep\nVersion: 1.0\n",
            "smoke_dep-1.0.dist-info/WHEEL": "Wheel-Version: 1.0\nGenerator: stdlib-fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            "smoke_dep-1.0.dist-info/RECORD": "",
        }.items():
            archive.writestr(name, text)
    raw = wheel.read_bytes()

    class WheelHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/" + wheel.name
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), WheelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # Hash-covered install actions use a private venv and forbid PyPI access.
        python = "installer/Scripts/python.exe" if sys.platform == "win32" else "installer/bin/python"
        spec = {"kind": "download", "url": f"http://127.0.0.1:{server.server_port}/{wheel.name}",
                "target": wheel.name, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw),
                "steps": [{"argv": [sys.executable, "-m", "venv", "installer"]},
                          {"argv": [python, "-m", "pip", "install", "--no-index", "--no-deps",
                                    "--target", "site", wheel.name]}],
                "outputs": ["site/smoke_dep/__init__.py"],
                "check": {"argv": [python, "-c", "import sys; sys.path.insert(0, 'site'); import smoke_dep; assert smoke_dep.VALUE == 41"]}}
        payload = source / "skills/ouroboroshub" / NAME
        (payload / ".ouroboroshub.json").write_text(json.dumps({"install_specs": {"auto": [spec]}}))

        def before_review(_body):
            target = tmp_path / "author/data/skills/ouroboroshub" / NAME
            assert not (target / ".ouroboros_env/site/smoke_dep/__init__.py").exists()
            from ouroboros.skill_loader import load_enabled
            assert not load_enabled(tmp_path / "author/data", NAME)
            return call()

        result, _ = consume(tmp_path, source, [before_review], [verdict()], dependency=True)
        installed = result["data_root"] / "skills/ouroboroshub" / NAME / ".ouroboros_env/site/smoke_dep/__init__.py"
        assert installed.read_text() == "VALUE = 41\n"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive(), "wheel resource server teardown unconfirmed"


def test_s41_false_criticism_gets_one_same_root_independent_reconsideration(tmp_path, source):
    result, model = consume(tmp_path, source, [call(), call(review_rebuttal=ARGUMENT)], [verdict(True), verdict()])
    assert result["review"]["status"] == "clean"
    assert len(model.review_calls) == 2
    # The reviewer actually receives the argument, not a Python auto-PASS rule.
    assert ARGUMENT in json.dumps(model.review_calls[-1])


def test_s41_true_bug_stays_red_and_payload_unchanged(tmp_path, source):
    plugin = source / "skills" / "ouroboroshub" / NAME / "plugin.py"
    plugin.write_text("def register(api):\n    return 1 / 0\n")
    original = plugin.read_bytes()
    consume(tmp_path, source, [call()], [verdict(True, real_bug=True)], error="criticism remains unresolved")
    assert (tmp_path / "author/data/skills/ouroboroshub" / NAME / "plugin.py").read_bytes() == original


def test_s41_repeated_and_changed_arguments_cannot_buy_second_substantive_reroll(tmp_path, source):
    _, model = consume(tmp_path, source, [call(), call(review_rebuttal=ARGUMENT),
        call(review_rebuttal=ARGUMENT), call(review_rebuttal=ARGUMENT + " A different argument.")],
        [verdict(True), verdict(True)], error="criticism remains unresolved")
    assert len(model.review_calls) == 2
    evidence = json.loads((tmp_path / "author/author-evidence.json").read_text())
    assert len([row for row in evidence["review_history"] if row.get("paid")]) == 2
    assert [row["status"] for row in evidence["author_tool_calls"] if row["tool"] == "skill_review"] == [
        "ok", "ok", "ok", "blocked"]


def test_s41_author_override_cannot_replace_independent_blocking_verdict(tmp_path, source):
    _, model = consume(tmp_path, source, [call(), call(author_disposition="accepted", author_rationale="I accept it.")],
                       [verdict(True)], error="criticism remains unresolved")
    assert len(model.review_calls) == 1
    evidence = json.loads((tmp_path / "author/author-evidence.json").read_text())
    assert not evidence["review"].get("author_disposition")


@pytest.mark.parametrize("kind", ["missing", "pending"])
def test_s41_missing_or_pending_independent_review_stays_red(tmp_path, source, kind):
    script = [] if kind == "missing" else [call()]
    verdicts = [] if kind == "missing" else ["not a review verdict"]
    consume(tmp_path, source, script, verdicts, error="missing review|pending/unknown")


@pytest.mark.parametrize("silent", ["", "not a review verdict"])
def test_s41_a_first_review_without_any_verdict_buys_one_retry(tmp_path, source, silent):
    """An empty or unreadable first answer is no opinion: the second paid cycle may ask again, and the retry's
    verdict decides; a second silence stays red."""
    retry = call(review_rebuttal="The previous review of this skill returned no verdict.")
    result, model = consume(tmp_path, source, [call(), retry], [silent, verdict()])
    assert result["review"]["status"] == "clean" and len(model.review_calls) == 2
    evidence = json.loads((tmp_path / "author/author-evidence.json").read_text())
    paid = [row for row in evidence["review_history"] if row.get("paid")]
    assert [row["status"] for row in paid][1] == "clean" and paid[0]["status"] not in {"clean", "warnings", "blockers"}
    consume(tmp_path / "twice", source, [call(), retry], [silent, silent], error="pending/unknown")


@pytest.mark.parametrize("field", ["group_id", "content_hash", "review_contract_fingerprint"])
def test_s41_group_material_or_panel_drift_is_red_at_actual_ci_entry(tmp_path, source, field):
    def drift(_body):
        # An independent race at the durable boundary, NOT an author write capability.
        from ouroboros import review_ledger
        from ouroboros.tools.review_skill import skill_record_rows
        data = tmp_path / "author/data"
        record_id = skill_record_rows(data, NAME, limit=1)[0]["record_id"]
        path = review_ledger.record_path(data, record_id)
        record = json.loads(path.read_text())
        if field == "group_id":
            record["brief"]["skill"]["group_id"] = "drifted"
        elif field == "content_hash":
            record["subject"]["tree_sha"] = "drifted"
        else:
            record["fingerprints"]["review_contract"] = "drifted"
        path.write_text(json.dumps(record))
        return {"final": "The author's prose says PASS; the CI evidence reader must refuse."}
    consume(tmp_path, source, [call(), call(review_rebuttal=ARGUMENT), drift],
            [verdict(True), verdict()], error="review group drift|review material drift|panel/contract drift")


def test_s41_real_author_calls_cannot_write_execute_enable_or_delegate(tmp_path, source):
    attacks = [
        {"tool": "write_file", "arguments": {"root": "skill_payload", "bucket": "ouroboroshub",
            "skill_name": NAME, "path": "plugin.py", "content": "raise RuntimeError('changed')"}},
        {"tool": "run_command", "arguments": {"cmd": ["sh", "-c", "echo should-not-run"]}},
        {"tool": "toggle_skill", "arguments": {"skill": NAME, "enabled": True}},
        {"tool": "skill_exec", "arguments": {"skill": NAME, "script": "plugin.py"}},
        {"tool": "start_task", "arguments": {"title": "reset", "objective": "reset review ceiling"}},
    ]
    _, model = consume(tmp_path, source, attacks + [call()], [verdict()])
    evidence = json.loads((tmp_path / "author/author-evidence.json").read_text())
    denied = [row for row in evidence["author_tool_calls"] if row["tool"] != "skill_review"]
    assert len(denied) == len(attacks)
    assert all(row.get("status") == "resource_constraint_blocked" for row in denied)
    agent_bodies = [body for kind, body in model.calls if kind not in {"skill_review", "safety"}]
    allowed = {"read_file", "list_files", "list_skills", "skill_review", "finish_task"}
    assert agent_bodies and all({tool["function"]["name"] for tool in body.get("tools") or []} <= allowed
                                for body in agent_bodies)
