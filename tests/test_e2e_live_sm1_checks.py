"""SM1's clean-worktree check after the reviewed commit: the runtime's own transient scratch under ``.ouroboros/``
(``run_script`` in the active workspace, tools/shell.py) is recorded but does not fail the check — the post-task
evolution cycle starts seconds after the commit in the same clone (rc.15 run3, SM1_a1, issue #701) — while any other
untracked or modified path still does."""
from __future__ import annotations

import colorsys
import pathlib
import subprocess
import types

import pytest

from devtools.e2e_live import scenarios
from devtools.e2e_live.ui_probe import UIProbe
from tests._cases import case_folders


def _repo(root: pathlib.Path) -> pathlib.Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for key, value in (("user.name", "t"), ("user.email", "t@example.com")):
        subprocess.run(["git", "-C", str(root), "config", key, value], check=True)
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "base"], check=True)
    return root


def test_the_runtimes_transient_scratch_is_recorded_but_does_not_fail_the_clean_check(tmp_path):
    root = _repo(tmp_path)
    assert scenarios.worktree_after_commit(root) == (True, "", [])
    (root / ".ouroboros" / "tmp_scripts").mkdir(parents=True)
    (root / ".ouroboros" / "tmp_scripts" / "script_deadbeef.py").write_text("print(1)\n", encoding="utf-8")
    clean, porcelain, transient = scenarios.worktree_after_commit(root)
    assert clean is True and porcelain == "?? .ouroboros/" and transient == ["?? .ouroboros/"]


def test_any_other_untracked_or_modified_path_still_fails_the_clean_check(tmp_path):
    root = _repo(tmp_path)
    (root / ".ouroboros" / "tmp_scripts").mkdir(parents=True)
    (root / ".ouroboros" / "tmp_scripts" / "script_deadbeef.py").write_text("print(1)\n", encoding="utf-8")
    (root / "stray.txt").write_text("x\n", encoding="utf-8")
    clean, porcelain, transient = scenarios.worktree_after_commit(root)
    assert clean is False and "?? stray.txt" in porcelain and transient == ["?? .ouroboros/"]
    (root / "stray.txt").unlink()
    (root / "a.txt").write_text("changed\n", encoding="utf-8")
    clean, porcelain, transient = scenarios.worktree_after_commit(root)
    assert clean is False and "M a.txt" in porcelain and transient == ["?? .ouroboros/"]   # ``_git`` strips the leading column


def test_the_landing_commit_call_records_its_skip_flags():
    """rc.15 run2/run3: two SM1 lanes landed through review_rebuttal + skip_advisory_review=True; the documented
    path has no skip flags, so the landing call's truthy ``skip_*`` arguments are a recorded fact and a check."""
    rows = [{"tool": "commit_reviewed", "result_preview": "⚠️ REVIEW_BLOCKED (attempt 1)", "args": {"skip_advisory_review": True}},
            {"tool": "commit_reviewed", "result_preview": "OK: committed to ouroboros: x",
             "args": {"skip_advisory_review": True, "skip_tests": False, "paths": ["a"]}},
            {"tool": "read_file", "result_preview": "OK", "args": {"skip_advisory_review": True}}]
    facts = scenarios.commit_refusal_facts({}, rows, {})
    assert facts["landing_skip_flags"] == ["skip_advisory_review"]
    clean = [{"tool": "commit_reviewed", "result_preview": "OK: committed", "args": {"paths": ["a"]}}]
    assert scenarios.commit_refusal_facts({}, clean, {})["landing_skip_flags"] == []
    assert scenarios.commit_refusal_facts({}, [], {})["landing_skip_flags"] == []


def test_t2_the_wave_fact_is_the_commit_gates_record_of_this_task_with_both_questions_answered():
    """FIX3 T2: the oracle reads what the wave really writes — the commit gate's review-ledger record
    (built here by the ledger itself, never a hand-shaped dict) — and ignores a dispatch-less
    refusal record, another task's record and another surface's; the newest answering record wins."""
    from ouroboros import review_ledger as rl
    from tests.test_review_ledger import BOTH, _facts, _raw, _three

    raws = [_raw("s1", "openai/gpt-5"), _raw("s2", "anthropic/claude-x", parts=BOTH), _raw("s3", "google/gemini")]
    landed = rl.build_commit_gate_record(_facts(raws, task_id="sm1-task")).to_dict()
    assert landed["verdict"]["per_question"] == {"change": "PASS", "coupling": "PASS"}, landed["verdict"]
    refused = rl.build_commit_gate_record(_facts([], task_id="sm1-task", dispatch_refusal={
        "kind": "pool_empty", "message": "the review pool is empty"})).to_dict()
    other_task = rl.build_commit_gate_record(_facts(_three(), task_id="other-task")).to_dict()
    other_surface = {**landed, "surface": "plan_review", "ts": "2099-01-01T00:00:00+00:00"}

    fact = scenarios.commit_wave_fact([refused, other_task, other_surface, landed], "sm1-task")
    assert fact == {"record_id": landed["record_id"], "aggregate": "PASS",
                    "per_question": {"change": "PASS", "coupling": "PASS"}, "seats": 3, "dispatched_seats": 3}
    assert scenarios.commit_wave_fact([refused, other_task, other_surface], "sm1-task") == {}, (
        "a record that dispatched nothing, another task's or another surface's is not the wave")
    assert scenarios.commit_wave_fact([], "sm1-task") == {} and scenarios.commit_wave_fact([None, 3], "sm1-task") == {}
    # Several attempts: the NEWEST record that answered both questions is the fact, whatever its verdict
    # (the landing itself is commit_landed's business), never an older PASS over a newer answer.
    newer_fail = {**landed, "record_id": "rv-newer", "ts": "2099-01-01T00:00:00+00:00",
                  "verdict": {**landed["verdict"], "aggregate": "FAIL", "per_question": {"change": "FAIL", "coupling": "PASS"}}}
    assert scenarios.commit_wave_fact([landed, newer_fail], "sm1-task")["record_id"] == "rv-newer"


def _gate_record(raws, **over):
    from ouroboros import review_ledger as rl
    from tests.test_review_ledger import _facts

    return rl.build_commit_gate_record(_facts(raws, task_id="sm1-task", **over)).to_dict()


def test_new_t2_the_wave_fact_needs_a_settled_record_with_dispatched_seats_and_pass_or_fail_on_both_questions():
    """NEW-T2 (Fable): the oracle accepted any non-empty ``per_question`` strings over any rows —
    ``unanswered``, ``not_performed`` and the reserved rows of a NOT_DISPATCHED record all passed
    as a wave. Each record below is built by the ledger itself. Positive: a PASS wave and a FAIL
    wave (the panel ran and found a critical: evidence, not a judgement). Negative: a budget
    refusal that reserved three rows and dispatched none; a wave without quorum (``change``
    ``unanswered``); an all-packet panel nobody asked the coupling (``not_performed``); the
    coupling block missing from the one retrieving seat (``unanswered``); an open-custody wave
    (``pending``) whose answered seats already read PASS."""
    from tests.test_review_ledger import BOTH, CRITICAL, _answer, _panel, _raw, _three

    withheld = _panel(status="not_dispatched")
    for row in withheld:
        row["operation_state"] = "not_dispatched"
    reserved = _gate_record(withheld, blocked=True, block_reason="review_wave_budget_insufficient")
    assert reserved["verdict"]["aggregate"] == "NOT_DISPATCHED" and len(reserved["rows"]) == 3, "rows reserved, none sent"
    short = _panel(status="error")
    short[1] = _raw("s2", "anthropic/claude-x")
    no_quorum = _gate_record(short, blocked=True, block_reason="review_quorum")
    assert no_quorum["verdict"]["per_question"]["change"] == "unanswered"
    packets_only = _gate_record(_three())
    assert packets_only["verdict"]["per_question"] == {"change": "PASS", "coupling": "not_performed"}
    missing_rows = _panel()
    missing_rows[0]["answers"]["coupling"] = _answer("coupling", "", status="unanswered", error="coupling_block_missing")
    coupling_missing = _gate_record(missing_rows)
    assert coupling_missing["verdict"]["per_question"] == {"change": "PASS", "coupling": "unanswered"}
    pending_rows = [_raw("s1", "openai/gpt-5", parts=BOTH, raw_text="two-part answer"), _raw("s2", "anthropic/claude-x"),
                    _raw("s3", "google/gemini", "pending", operation_state="in_flight", late_result_pending=True)]
    pending = _gate_record(pending_rows)
    assert pending["state"] == "pending" and pending["verdict"]["per_question"] == {"change": "PASS", "coupling": "PASS"}
    for name, record in (("reserved", reserved), ("no_quorum", no_quorum), ("packets_only", packets_only),
                         ("coupling_missing", coupling_missing), ("pending", pending)):
        assert scenarios.commit_wave_fact([record], "sm1-task") == {}, name

    passed = _gate_record(_panel())
    failing_rows = _panel()
    failing_rows[1]["answers"] = {"change": _answer("change", "FAIL", [CRITICAL])}
    failed = _gate_record(failing_rows, blocked=True, block_reason="critical_findings", critical_findings=[CRITICAL])
    assert (passed["verdict"]["aggregate"], failed["verdict"]["aggregate"]) == ("PASS", "FAIL")
    for record in (passed, failed):
        fact = scenarios.commit_wave_fact([record], "sm1-task")
        assert fact["record_id"] == record["record_id"] and fact["aggregate"] == record["verdict"]["aggregate"]
        assert set(fact["per_question"].values()) <= {"PASS", "FAIL"} and fact["dispatched_seats"] == fact["seats"] == 3
    # Beside a real wave the non-answers are skipped, not mistaken for a newer wave.
    later = {"ts": "2099-01-01T00:00:00+00:00"}
    assert scenarios.commit_wave_fact([passed, {**reserved, **later}, {**no_quorum, **later}], "sm1-task")["record_id"] == passed["record_id"]


def test_ui_probe_waits_for_the_requested_document():
    calls = []
    probe = UIProbe("http://lane.test")
    probe.page = types.SimpleNamespace(
        goto=lambda url, **kw: calls.append(("goto", url)),
        wait_for_selector=lambda selector, **kw: calls.append(("ready", selector)),
    )
    probe.goto()
    probe.goto("/onboarding", ready_selector=".wizard-shell")
    assert calls == [("goto", "http://lane.test/"), ("ready", "#chat-input"),
                     ("goto", "http://lane.test/onboarding"), ("ready", ".wizard-shell")]


SHEET_SEED = """:root {
    --native-color-scheme: dark;
    --project: #ee90fa;
    --user-sender: #a0c6ff;
    --status-error-fg: #fe9498;
    --assistant-sender: #feaaae;
    --accent: #f45152;
    --accent-light: #fe7d85;
    --accent-dim: rgba(201, 53, 69, 0.15);
    /* Accent alpha tokens; keep future tweaks off hardcoded rgba literals. */
    --accent-04: rgba(201, 53, 69, 0.04);
    --accent-10: rgba(201, 53, 69, 0.10);
    --accent-55: rgba(201, 53, 69, 0.55);
    --status-live-fg: var(--accent-light);
    --focus-accent-border: #fe7d85;
    --focus-accent-ring: rgba(211, 45, 67, 0.28);
    --tone-crimson: #d32d43;
    --tone-crimson-ink: #ffffff;
    --tone-crimson-word: var(--accent-light);
}

:root[data-theme="light"] {
    --native-color-scheme: light;
    --accent-light: #bf2f40;
    --project: #8f2499;
    --focus-accent-border: #bf2f40;
    --tone-crimson: var(--accent-10);
    --tone-crimson-ink: #b52b3b;
    --tone-crimson-word: var(--tone-crimson-ink);
    --accent: #c93545;
    --focus-accent-ring: var(--accent-10);
}

:root:not([data-theme="light"]) .chat-bubble.assistant {
    --accent-light: #ffc2c6;
    --accent-10: var(--surface-capsule);
}
:root:not([data-theme="light"]) .chat-bubble.user { --accent-light: var(--user-sender); }
.x { --accent: red; }
"""


def _sheet_lines_changed(before: str, after: str) -> list[str]:
    return [old.split(":")[0].strip() for old, new in zip(before.splitlines(), after.splitlines(), strict=True) if old != new]


def _lightness(hex_value: str) -> float:
    return colorsys.rgb_to_hls(*(channel / 255 for channel in bytes.fromhex(hex_value[1:])))[1]


def test_the_scripted_authors_edit_reaches_every_theme_and_only_the_accent_roles():
    """The real rehearsal's root cause: the edit changed the first ``--accent`` and light kept ``#c93545``, with its roles
    and ladder red in both themes. Owner, Project, status and voice roles stay, an alias keeps what it means in its own
    theme, each theme keeps its own light or dark word, and a voice's own material is no theme root."""
    changed = scenarios.css_with_accent(SHEET_SEED, scenarios.SM1_NEW_ACCENT)
    tokens = scenarios.css_theme_tokens(changed)
    assert {theme: values["--accent"] for theme, values in tokens.items()} == {"light": "#2f7de1", "dark": "#2f7de1"}
    for theme in scenarios.SM1_THEMES:
        assert tokens[theme]["--accent-10"] == "rgba(47, 125, 225, 0.10)" and tokens[theme]["--accent-dim"] == "rgba(47, 125, 225, 0.15)"
        assert "201, 53, 69" not in " ".join(tokens[theme].values()), theme
    assert tokens["dark"]["--focus-accent-ring"] == "rgba(47, 125, 225, 0.28)"
    assert tokens["light"]["--focus-accent-ring"] == "rgba(47, 125, 225, 0.10)" == tokens["light"]["--tone-crimson"]
    # Each theme's word keeps its lightness on the new hue; the neutral ink has no hue to move.
    assert tokens["dark"]["--accent-light"] == "#7db6fe" and tokens["light"]["--accent-light"] == "#2f6ebf"
    assert _lightness(tokens["dark"]["--accent-light"]) > 0.5 > _lightness(tokens["light"]["--accent-light"])
    assert tokens["dark"]["--tone-crimson"] == "#2d76d3" and tokens["dark"]["--tone-crimson-ink"] == "#ffffff"
    assert tokens["light"]["--tone-crimson-ink"] == "#2b67b5" == tokens["light"]["--tone-crimson-word"]
    assert tokens["dark"]["--tone-crimson-word"] == tokens["dark"]["--accent-light"] == tokens["dark"]["--status-live-fg"]
    for alias in ("--tone-crimson: var(--accent-10);", "--focus-accent-ring: var(--accent-10);", "--tone-crimson-word: var(--accent-light);",
                  "--tone-crimson-word: var(--tone-crimson-ink);", "--status-live-fg: var(--accent-light);"):
        assert alias in changed, alias
    assert set(_sheet_lines_changed(SHEET_SEED, changed)) == {
        "--accent", "--accent-light", "--accent-dim", "--accent-04", "--accent-10", "--accent-55", "--focus-accent-border",
        "--focus-accent-ring", "--tone-crimson", "--tone-crimson-ink"}, "nothing outside the accent's roles moved"
    tail = SHEET_SEED[SHEET_SEED.index(":root:not("):]
    assert changed.endswith(tail), "a voice's own material and any other rule stay as declared"
    for theme, values in scenarios.css_theme_tokens(SHEET_SEED).items():
        for role in ("--project", "--user-sender", "--status-error-fg", "--assistant-sender", "--native-color-scheme"):
            assert tokens[theme][role] == values[role], (theme, role)
    assert scenarios.css_with_accent(changed, scenarios.SM1_NEW_ACCENT) == changed, "already there, it changes nothing"
    assert scenarios.accent_value(SHEET_SEED) == "#f45152" and scenarios.accent_value(changed) == scenarios.SM1_NEW_ACCENT
    one_theme = ":root {\n    --accent: #c93545;\n}\n"
    assert scenarios.css_with_accent(one_theme, "#2f7de1") == ":root {\n    --accent: #2f7de1;\n}\n", "no light block, one theme to change"


def test_the_scripted_authors_edit_of_the_real_sheet_leaves_no_seed_role_in_either_theme():
    """The actual stylesheet, whatever its values are today: every role of the family is off its seed in both themes, and
    the only lines that differ declare an accent role (what a future token would slip through)."""
    original = (pathlib.Path(__file__).resolve().parents[1] / scenarios.SM1_CSS_PATH).read_text(encoding="utf-8")
    seed = original if scenarios.accent_value(original) != scenarios.SM1_NEW_ACCENT else scenarios.css_with_accent(original, "#c93545")
    changed = scenarios.css_with_accent(seed, scenarios.SM1_NEW_ACCENT)
    names = scenarios.sm1_palette_tokens(seed).keys()
    assert scenarios.SM1_REQUIRED_PALETTE <= names
    before, after = scenarios.css_theme_tokens(seed), scenarios.css_theme_tokens(changed)
    for theme in scenarios.SM1_THEMES:
        assert after[theme]["--accent"] == scenarios.SM1_NEW_ACCENT, theme
        assert not [name for name in names if after[theme][name] == before[theme][name]], theme
        assert after[theme]["--native-color-scheme"] == theme == before[theme]["--native-color-scheme"], "the oracle's own marker"
    assert scenarios.SM1_CONTROL_FILLS <= names, "a control's fill is part of what the browser reads"
    assert all(name.startswith(("--accent", "--focus-accent-", "--tone-crimson")) for name in _sheet_lines_changed(seed, changed))


THEME_READ_PROBLEMS = {
    "none": {}, "no-commit": {}, "not-adopted": {},
    "missing-sheet": {("onboarding", "light"): "--accent empty", ("onboarding", "dark"): "--accent empty"},
    "wizard-stale-accent": {("onboarding", "light"): "--accent is #c93545", ("onboarding", "dark"): "--accent is #c93545"},
    "wizard-stale-focus": {},
    # The real rehearsal: the dark seed changed, the light declaration kept the old value on both documents.
    "light-keeps-old-accent": {("app", "light"): "--accent is #c93545", ("onboarding", "light"): "--accent is #c93545"},
    "dark-keeps-old-accent": {("app", "dark"): "--accent is #f45152", ("onboarding", "dark"): "--accent is #f45152"},
    # A control's fill on one theme's old value: the red primary action on a blue palette.
    "dark-control-fill-at-seed": {("app", "dark"): "--tone-crimson keeps its seed value",
                                  ("onboarding", "dark"): "--tone-crimson keeps its seed value"},
    # One theme's role still at its seed value: a mixed palette, invisible to the accent token itself.
    "light-role-at-seed": {("app", "light"): "--focus-accent-border keeps its seed value",
                           ("onboarding", "light"): "--focus-accent-border keeps its seed value"},
    # The page ignored the asked appearance: the seed it does paint must not read as the other theme's proof.
    "light-never-painted": {("app", "light"): "paints dark", ("onboarding", "light"): "paints dark"},
    # What a client opening on a target-accent seed reads: its own bytes are no proof of a change.
    "seed-already-target": {("app", "light"): "--accent keeps its seed value", ("app", "dark"): "--accent keeps its seed value",
                            ("onboarding", "light"): "--accent keeps its seed value", ("onboarding", "dark"): "--accent keeps its seed value"},
}


def test_sm1_oracle_requires_every_theme_of_both_documents_and_the_landed_commit(subtests, tmp_path):
    case_paths = case_folders(tmp_path)
    for problem in sorted(THEME_READ_PROBLEMS):
        tmp_path = next(case_paths)
        with subtests.test(problem=problem):
            calls = []
            landed = scenarios.css_theme_tokens(scenarios.css_with_accent(SHEET_SEED, scenarios.SM1_NEW_ACCENT))

            class PaletteUI:
                path = scheme = ""

                def color_scheme(self, scheme):
                    calls.append(("scheme", scheme))
                    self.scheme = scheme

                def goto(self, path, *, ready_selector):
                    self.path = path
                    calls.append((path, ready_selector))

                def computed_property(self, selector, name):
                    assert selector == ":root"
                    wizard, light = self.path == "/onboarding", self.scheme == "light"
                    if wizard and problem == "missing-sheet":
                        return ""
                    if name == "--native-color-scheme":
                        return "dark" if problem == "light-never-painted" else self.scheme
                    if name == "--accent":
                        if wizard and problem == "wizard-stale-accent":
                            return "#c93545"
                        if (light and problem == "light-keeps-old-accent") or (not light and problem == "dark-keeps-old-accent"):
                            return "#c93545" if light else "#f45152"
                    if name == "--tone-crimson" and not light and problem == "dark-control-fill-at-seed":
                        return "#d32d43"
                    if name == "--focus-accent-border":
                        if wizard and problem == "wizard-stale-focus":
                            return "#00ff00"
                        if light and problem == "light-role-at-seed":
                            return "#bf2f40"
                    return landed[self.scheme].get(name, "")

                screenshot = close = lambda self, *args: None

            server = types.SimpleNamespace(base_url="http://lane.test")
            ctx = scenarios.LaneContext(server=server, clone=tmp_path, data_root=tmp_path, oracle=None,
                                        harness=None, ui_resolver=lambda _: (PaletteUI(), ""), ui_reason="",
                                        shots=tmp_path, log=lambda _: None, task_timeout=1,
                                        restart=lambda: (calls.append("restart") or server))
            ctx.check("commit_landed", problem != "no-commit")
            seed = scenarios.css_with_accent(SHEET_SEED, scenarios.SM1_NEW_ACCENT) if problem == "seed-already-target" else SHEET_SEED
            names = scenarios.SM1_REQUIRED_PALETTE | scenarios.sm1_palette_tokens(SHEET_SEED).keys()
            scenarios.check_sm1_rendered_palette(ctx, names, served=problem != "not-adopted", seed_css=seed)
            assert calls == [step for path, ready in (("/", "#chat-input"), ("/onboarding", ".wizard-shell"))
                             for theme in ("light", "dark") for step in (("scheme", theme), (path, ready))], (
                "every theme is asked for before its document opens; the adoption restarted the server, the stand never does")
            expected = THEME_READ_PROBLEMS[problem]
            for (document, theme), reason in expected.items():
                assert reason in ctx.facts["palette_problems"][document][theme], ctx.facts["palette_problems"]
            for document in ("app", "onboarding"):
                for theme in scenarios.SM1_THEMES:
                    assert bool(ctx.facts["palette_problems"][document][theme]) is ((document, theme) in expected), ctx.facts["palette_problems"]
            documents_ok = {document: not any(key[0] == document for key in expected) and problem not in {"no-commit", "not-adopted"}
                            for document in ("app", "onboarding")}
            assert ctx.checks["ui_app_palette"] is documents_ok["app"] and ctx.checks["ui_onboarding_palette"] is documents_ok["onboarding"]
            assert ctx.checks["ui_computed_style"] is (problem == "none"), "a document or theme apart, or an unserved commit, is no consistency"
            assert all(len(ctx.facts["palette_computed"][document][theme]) == len(names)
                       for document in ("app", "onboarding") for theme in scenarios.SM1_THEMES)
            ctx.close_ui()


def test_a_shot_waits_for_the_page_briefly_and_never_gates_on_it(tmp_path):
    """The real rehearsal's app capture still read `Connecting…`. A page that never goes quiet is shot as it stands; a dead
    target still raises, for ``GuardedUI`` to type, instead of being mistaken for one."""
    sync_api = pytest.importorskip("playwright.sync_api")
    events = []

    def shot(**kwargs):
        events.append("shot")

    def still_polling(state, *, timeout):
        events.append((state, timeout))
        raise sync_api.TimeoutError("still polling")

    probe = UIProbe("http://lane.test")
    probe.page = types.SimpleNamespace(wait_for_load_state=still_polling, screenshot=shot)
    probe.screenshot(tmp_path / "shots" / "a.png")
    assert events == [("networkidle", 10_000), "shot"] and (tmp_path / "shots").is_dir()

    def dead(*_args, **_kwargs):
        raise RuntimeError("Target page, context or browser has been closed")

    probe.page = types.SimpleNamespace(wait_for_load_state=dead, screenshot=shot)
    with pytest.raises(RuntimeError):
        probe.screenshot(tmp_path / "shots" / "b.png")
    assert events.count("shot") == 1
