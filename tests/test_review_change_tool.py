"""``review_change``: one review-only wave over a change in any registered root.

The neighbour seams (the frozen review subject and the rules layer of the body
fact) are substituted on the module under test with fakes of the SAME contract
(``review_subject.FrozenSubject`` / ``review_body_fact.BodyFact``); the identities
(reuse and retry keys) and the ledger are real. The paid wave is a stub that reads
the panel in force, stamps the paid fact at dispatch and answers the way
``parallel_review`` leaves its forensic facts on the context. Every assertion is
about what ``review_change`` decides, dispatches, records and returns; the
end-to-end proof over the real subject operation is
``test_review_change_end_to_end.py``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib
import subprocess
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from ouroboros import review_ledger as rl
from ouroboros import reviewer_slot_config as slots
from ouroboros.tools import commit_gate
from ouroboros.tools import review_change as rc
from ouroboros.tools.review_subject import ReviewSubjectSpec, review_retry_key, review_round_sha
from tests.review_pool_rosters import mixed_pool_rows, pool_roster, pool_seat, set_review_pool


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT).strip()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _repo(path: pathlib.Path, marker: str, *, origin: str = "") -> pathlib.Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / f"{marker}.txt").write_text("base\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "base")
    if origin:
        _git(path, "remote", "add", "origin", origin)
    return path


def _stage(repo: pathlib.Path, name: str, text: str) -> None:
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", name)


def _commit(repo: pathlib.Path, name: str, text: str) -> str:
    _stage(repo, name, text)
    _git(repo, "commit", "-qm", f"add {name}")
    return _git(repo, "rev-parse", "HEAD")


def _pool() -> list:
    """The review pool in force, by seat id (inside a composed wave: that wave's seats)."""
    return [slot.slot_id for slot in slots.review_pool_slots()]


def _seats() -> list:
    """``(slot, parts, additional)`` of the wave in force: the composed seats inside a
    composed wave, else the configured pool with the parts its delivery decides."""
    composed = slots.composed_pool_seats()
    if composed is not None:
        return [(seat.slot, tuple(seat.parts), seat.additional) for seat in composed]
    return [(slot, tuple(rl.seat_parts(slot)), False) for slot in slots.review_pool_slots()]


@dataclass(frozen=True)
class Frozen:
    """The ``FrozenSubject`` contract: the exact bytes one wave reviews."""

    spec: ReviewSubjectSpec
    diff_text: str
    diff_sha: str
    tree_sha: str
    parent_sha: str
    checkout: str = ""
    name_status: tuple = ()

    def record_subject(self) -> Dict[str, Any]:
        return {"root_kind": self.spec.root_kind, "root": self.spec.root, "kind": self.spec.kind,
                "base": self.parent_sha, "head": self.spec.head, "tree_sha": self.tree_sha,
                "diff_sha": self.diff_sha, "checkout": self.checkout}


class Wave:
    """The paid wave stub: ONE wave, every seat asked its ``parts`` — a pool seat
    ``seat_parts(row)`` (both when it retrieves, ``change`` when it reads the packet),
    an added ``coupling_only`` seat ``coupling`` alone — each answering by part
    (contract B's shape as the ledger records it). ``triad`` names the assigned seats,
    ``coupling`` the added coupling-only ones."""

    def __init__(self) -> None:
        self.calls: List[SimpleNamespace] = []
        self.failing: set = set()
        self.status = "responded"
        self.last = SimpleNamespace()

    def _answer(self, row: Any, parts: tuple) -> Dict[str, Any]:
        failing, answered = row.slot_id in self.failing, self.status == "responded"
        answers: Dict[str, Any] = {}
        for part in parts:
            if part == rl.PART_COUPLING:
                findings = [{"item": "forgotten_touchpoints", "verdict": "FAIL", "severity": "critical",
                             "reason": f"{row.slot_id}: a caller breaks"}] if failing else []
            else:
                findings = [{"item": "code_quality", "verdict": "FAIL", "severity": "critical",
                             "reason": f"{row.slot_id}: defect"}] if failing else []
            answers[part] = {"status": "responded", "verdict": "FAIL" if findings else "PASS", "findings": findings,
                             "critical": len(findings), "coverage": "n/a" if part == rl.PART_CHANGE else "complete"}
        answer = {"slot_id": row.slot_id, "model_id": row.model, "status": self.status, "parts": list(parts),
                  "raw_text": f"{row.slot_id} {'+'.join(parts)} answer", "cost_usd": 0.01}
        if answered:
            answer["answers"] = answers
        return answer

    def __call__(self, ctx: Any, commit_message: str, *, goal: str = "", scope: str = "", review_rebuttal: str = "",
                 review_binding_fingerprint: str = "", subject: Any = None):
        from ouroboros.review_dispatch import invoke_review_paid_stamp

        seated = _seats()
        triad = [row for row, _parts, additional in seated if not additional]
        coupling = [row for row, _parts, additional in seated if additional]
        seats = [(row, parts) for row, parts, _additional in seated]
        prompt = f"PACKET\n{commit_message}\n{goal}\n{subject.diff_text}"
        brief = f"TWO-PART BRIEF\n{goal}\n{subject.diff_text}"
        texts = {_sha(prompt): prompt, _sha(brief): brief}
        self.calls.append(SimpleNamespace(
            subject=subject, label=commit_message, goal=goal, scope=scope, rebuttal=review_rebuttal,
            fingerprint=review_binding_fingerprint, tool=ctx._current_review_tool_name,
            retry_key=ctx._current_review_retry_key, record_id=ctx._current_review_record_id,
            triad=[row.slot_id for row in triad], coupling=[row.slot_id for row in coupling],
            parts={row.slot_id: parts for row, parts in seats},
            efforts={row.slot_id: row.effort for row in (*triad, *coupling)}, prompt=prompt, brief=brief))
        invoke_review_paid_stamp(ctx._review_paid_stamp)
        ctx._last_triad_raw_results = [self._answer(row, parts) for row, parts in seats]
        plan = [{"slot_id": row.slot_id, "model": row.model, "route": "api_chat", "effort": row.effort or "high",
                 "parts": list(parts), "retrieves": rl.PART_COUPLING in parts,
                 "brief_sha": _sha(brief) if rl.PART_COUPLING in parts else _sha(prompt)} for row, parts in seats]
        ctx._last_review_structured = {"started_ts": "2026-10-07T00:00:00+00:00", "rows": plan, "brief_texts": texts,
                                       "quorum": rl._quorum_for(len(plan))}
        rows = rl.build_rows({"structured": ctx._last_review_structured, "triad_raw": ctx._last_triad_raw_results})
        verdict = rl.reduce_verdict([seat for seat in rows if not seat["additional"]])
        ctx._last_review_verdict = verdict
        ctx._last_coupling_result = rl.coupling_outcome(verdict, rows)
        critical = verdict["aggregate"] == rl.VERDICT_FAIL
        ctx._last_review_block_reason = "critical_findings" if critical else ""
        self.last = SimpleNamespace(structured=dict(ctx._last_review_structured), triad_raw=list(ctx._last_triad_raw_results))
        return ("⚠️ REVIEW_BLOCKED: critical findings" if critical else None), ctx._last_coupling_result, ctx._last_review_block_reason, []


def gate_record(facts: Dict[str, Any], *, record_id: str = "", drive_root: Any = None) -> Any:
    """The commit gate's record builder over the same wave facts and frozen binding."""
    frozen = facts["subject"]
    binding = {"tree_sha": frozen.tree_sha, "parents": [frozen.parent_sha], "diff_sha256": frozen.diff_sha}
    return rl.build_commit_gate_record({**facts, "binding": binding}, record_id=record_id, drive_root=drive_root)


@dataclass
class Harness:
    ctx: Any
    system: pathlib.Path
    project: pathlib.Path
    drive: pathlib.Path
    wave: Wave
    calls: List[tuple] = field(default_factory=list)
    written: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    built: List[Dict[str, Any]] = field(default_factory=list)

    def run(self, **args: Any) -> Dict[str, Any]:
        return rc.run_review_change(self.ctx, **args)

    def tool(self, **args: Any) -> str:
        return rc._handle_review_change(self.ctx, **args)

    def attempts(self, root: pathlib.Path, tool_name: str = "review_change") -> list:
        from ouroboros.review_state import load_state, make_repo_key

        return load_state(self.drive).filter_attempts(repo_key=make_repo_key(root.resolve()), tool_name=tool_name)


def _install_seams(monkeypatch: pytest.MonkeyPatch, h: Harness) -> None:
    def frozen_of(spec: ReviewSubjectSpec, checkout: str = "") -> Frozen:
        root, base = pathlib.Path(spec.root), spec.base or "HEAD"
        if spec.kind == "base..head":
            diff = _git(root, "diff", "--binary", spec.base, spec.head)
            tree = _git(root, "rev-parse", f"{spec.head}^{{tree}}")
        else:
            diff = _git(root, "diff", "--binary", *(["--cached"] if spec.kind == "index" else []), base)
            tree = _git(root, "write-tree") if spec.kind == "index" else _sha(diff)
        return Frozen(spec, diff, _sha(diff), tree, _git(root, "rev-parse", base), checkout=checkout)

    def freeze(ctx: Any, spec: ReviewSubjectSpec) -> Frozen:
        h.calls.append(("freeze", spec))
        return frozen_of(spec)

    @contextlib.contextmanager
    def checkout(ctx: Any, spec: ReviewSubjectSpec, *, retain=None, token=None):
        """``review_subject.isolated_checkout``'s contract for every subject kind: the
        frozen subject reads in a checkout under the data root, named by ``token`` from
        the frozen identity (one path per round) when the caller gives one."""
        h.calls.append(("checkout", spec))
        identity = frozen_of(spec)
        name = token(identity) if token is not None else spec.kind.replace("..", "-")
        yield frozen_of(spec, checkout=str(h.drive / "checkouts" / name))
        # The runtime's exit question (review_subject.checkout_retention): kept or removed.
        from ouroboros.tools.review_subject import checkout_retention

        h.calls.append(("checkout_retained" if checkout_retention(retain) else "checkout_closed", spec))

    def write(drive_root: Any, record: Any) -> Dict[str, Any]:
        payload = rl.write_record(drive_root, record)
        h.written[payload["record_id"]] = payload
        return payload

    def fact(root: Any, *, system_repo: Any, manifest: Any = None, data_dir: Any = None, treat_as_body: bool = False):
        """``review_body_fact.body_fact``'s contract: the system repository is the body
        (``dir``); a root with a remote that reaches neither the managed remote nor the
        install's origin is a recognized foreign root (``false``); a root git cannot
        place (no remote) is ``unknown``, and ONLY that one is raised by ``treat_as_body``
        — ``how`` keeps the fact's value and ``detail`` records the raise."""
        from ouroboros.review_body_fact import BodyFact

        h.calls.append(("body_fact", pathlib.Path(root), treat_as_body))
        root = pathlib.Path(root).resolve()
        if root == pathlib.Path(system_repo).resolve():
            return BodyFact("true", "dir", f"{root} is the system repository")
        if _git(root, "remote"):
            return BodyFact("false", "remote_chain", "a remote reaches neither the managed remote nor the install's origin")
        if treat_as_body:
            return BodyFact("true", "unknown", f"{root} has no remote; raised to body by treat_as_body")
        return BodyFact("unknown", "unknown", f"{root} has no remote")

    def build(facts: Dict[str, Any], *, surface: str, record_id: str = "", drive_root: Any = None) -> Any:
        h.built.append(facts)
        record = gate_record(facts, record_id=record_id, drive_root=drive_root)
        record.surface = surface
        return record

    monkeypatch.setattr(rc, "freeze_subject", freeze)
    monkeypatch.setattr(rc, "isolated_checkout", checkout)
    monkeypatch.setattr(rc, "write_record", write)
    monkeypatch.setattr(rc, "body_fact", fact)
    monkeypatch.setattr(rc, "layer_for", lambda fact: "body" if fact.body == "true" else "core")
    monkeypatch.setattr(rc, "checklist_fingerprint", lambda layer: {
        "checklist_hash": f"{layer}-checklist", "rules_source": {"path": "docs/CHECKLISTS.md", "sha": f"{layer}-rules"}})
    monkeypatch.setattr(rc, "build_wave_record", build)


@pytest.fixture
def h(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    from ouroboros.tools.registry import ToolContext

    # The production geometry: the body and its data share one Ouroboros home; project
    # repositories live elsewhere under the user's files.
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path))
    system = _repo(tmp_path / "ouroboros" / "repo", "system")
    project = _repo(tmp_path / "work" / "project", "project", origin="https://example.com/third-party/project.git")
    drive = tmp_path / "ouroboros" / "data"
    for sub in ("logs", "locks", "state"):
        (drive / sub).mkdir(parents=True)
    ctx = ToolContext(repo_dir=system, system_repo_dir=system, drive_root=drive, workspace_root=project,
                      workspace_mode="external", task_id="task-rc")
    harness = Harness(ctx=ctx, system=system, project=project, drive=drive, wave=Wave())
    _install_seams(monkeypatch, harness)
    monkeypatch.setattr(rc, "run_parallel_review", harness.wave)
    monkeypatch.setattr(rc, "get_runtime_mode", lambda: "advanced")
    monkeypatch.setattr(rc, "get_review_enforcement", lambda: "advisory")
    # The shipped cycle cap is the ceiling test's subject; every other test pays freely.
    monkeypatch.setattr(commit_gate, "review_max_cycles", lambda: None)
    # The pool: two natively retrieving seats (asked both parts) and one packet seat; an
    # enabled unmarked row (``scout``) is a reviewer only when the author names it.
    set_review_pool(monkeypatch, pool_roster(*mixed_pool_rows(), pool_seat("scout", "openai/scout-model", marked=False)))
    pool = _pool()
    assert len(pool) >= 2 and any(slot.retrieves for slot in slots.review_pool_slots()), "several seats, one retrieving"
    return harness


def _prompt_shas(record: Dict[str, Any]) -> set:
    return {(ref["part"], ref["ref"]["sha256"]) for seat in record["rows"] for ref in seat["source_refs"]
            if ref.get("role") == "prompt"}


# --- Schema -----------------------------------------------------------------------

def test_the_schema_is_the_planned_call() -> None:
    from ouroboros.settings_scales import EFFORT_SCALE

    [entry] = rc.get_tools()
    params = entry.schema["parameters"]
    assert entry.name == "review_change" and params["required"] == ["subject"]
    assert list(params["properties"]) == [
        "root", "workspace_root", "subject", "base", "head", "surface", "goal", "scope", "author_questions",
        "reviewers", "reason", "coupling_only", "reviewer_effort", "review_rebuttal", "treat_as_body"]
    props = params["properties"]
    assert props["root"]["enum"] == ["active_workspace", "system_repo"] and props["root"]["default"] == "active_workspace"
    assert props["subject"]["enum"] == ["index", "worktree", "base..head", "system"]
    assert props["surface"]["enum"] == ["change", "preflight", "system"] and props["surface"]["default"] == "change"
    assert props["reviewer_effort"]["enum"] == list(EFFORT_SCALE)
    assert (props["treat_as_body"]["type"], props["treat_as_body"]["default"]) == ("boolean", False)
    # The schema states the predicate's rule (review_body_fact.body_fact): the flag raises
    # only an UNKNOWN body fact; a recognized body or foreign root is unchanged.
    description = props["treat_as_body"]["description"]
    assert "unknown" in description and "unchanged" in description and "even if it is not the body" not in description
    assert entry.timeout_sec and entry.timeout_sec > 0


# --- Root, subject and rules layer --------------------------------------------------

def test_the_system_index_is_the_gate_path_on_one_frozen_subject(h: Harness, tmp_path: pathlib.Path) -> None:
    _stage(h.system, "body.py", "print('change')\n")
    result = h.run(root="system_repo", subject="index", goal="Fix the body")

    [call] = h.wave.calls
    frozen = call.subject
    assert isinstance(frozen, Frozen) and [kind for kind, *_ in h.calls if kind == "checkout"] == []
    assert (frozen.spec.root_kind, frozen.spec.kind, frozen.spec.layer, frozen.spec.body_fact) == (
        "system_repo", "index", "body", "true")
    assert pathlib.Path(frozen.spec.root) == h.system.resolve()
    assert frozen.diff_text == _git(h.system, "diff", "--binary", "--cached", "HEAD")
    assert call.fingerprint == frozen.diff_sha

    gate = gate_record({"task_id": "task-rc", "repo_dir": str(h.system), "subject": frozen, "goal": "Fix the body",
                        **vars(h.wave.last)}, drive_root=tmp_path / "gate-data").to_dict()
    record = rl.load_record(h.drive, result["record_id"])
    pool = _pool()
    expected = {(",".join(parts), _sha(call.brief if "coupling" in parts else call.prompt)) for parts in call.parts.values()}
    assert _prompt_shas(record) == _prompt_shas(gate) == expected and ("change,coupling", _sha(call.brief)) in expected
    assert [seat["seat_id"] for seat in record["rows"]] == [seat["seat_id"] for seat in gate["rows"]] == pool
    assert (record["verdict"]["aggregate"], record["verdict"]["per_question"]) == (
        gate["verdict"]["aggregate"], gate["verdict"]["per_question"])
    assert result["aggregate"] == rl.VERDICT_PASS and result["record_id"] == call.record_id
    assert result["checklist"] == {"layer": "body", "body_fact": "true", "how": "dir", "treat_as_body": False}
    assert record["surface"] == "change" and record["brief"]["checklist"]["rules_source"]["sha"] == "body-rules"
    assert result["panel"]["composition"] == "full_pool" and result["panel"]["chosen_by"] == "owner"


def test_a_foreign_base_head_is_core_untested_and_locks_nothing(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _pool()
    base = _git(h.project, "rev-parse", "HEAD")
    head = _commit(h.project, "feature.py", "print('feature')\n")
    h.wave.failing = {pool[0]}  # a retrieving seat: both of its answers carry the defect
    monkeypatch.setattr(rc, "get_review_enforcement", lambda: "blocking")
    result = h.run(subject="base..head", base=base, head=head, goal="Add the feature")

    [call] = h.wave.calls
    assert (call.subject.spec.root_kind, call.subject.spec.layer, call.subject.spec.body_fact) == (
        "active_workspace", "core", "false")
    assert call.subject.checkout and [kind for kind, *_ in h.calls if kind.startswith("checkout")] == [
        "checkout", "checkout_closed"]
    assert result["checklist"] == {"layer": "core", "body_fact": "false", "how": "remote_chain", "treat_as_body": False}
    assert result["tests"] == {"policy": "NOT_RUN", "result": "unknown"}
    assert result["aggregate"] == rl.VERDICT_FAIL and {f["seat_id"] for f in result["findings"]["critical_findings"]} == {pool[0]}
    assert sorted(f["part"] for f in result["findings"]["critical_findings"]) == ["change", "coupling"]
    assert (result["enforcement"], result["enforcement_blocks"]) == ("blocking", False)
    assert (result["subject"]["base"], result["subject"]["head"], result["subject"]["root"]) == (
        base, head, str(h.project.resolve()))
    assert _git(h.project, "rev-parse", "HEAD") == head and _git(h.project, "status", "--porcelain") == ""
    assert [attempt.status for attempt in h.attempts(h.project)] == ["reviewed"]

    # The same verdict on the body is enforced by the record.
    _stage(h.system, "body.py", "x = 1\n")
    body = h.run(root="system_repo", subject="index")
    assert (body["aggregate"], body["enforcement"], body["enforcement_blocks"]) == (rl.VERDICT_FAIL, "blocking", True)
    assert _git(h.system, "diff", "--cached", "--name-only") == "body.py"


def test_treat_as_body_raises_only_an_unknown_root_to_the_body_layer(
        h: Harness, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    pool = _pool()
    monkeypatch.setattr(rc, "get_review_enforcement", lambda: "blocking")
    # A recognized foreign root (its remote reaches neither the managed remote nor the
    # install's origin) stays on the core layer whatever the flag says; the flag is recorded.
    _stage(h.project, "a.py", "a = 1\n")
    foreign = h.run(subject="index", treat_as_body=True, reviewers=[pool[0]], reason="one seat")
    assert ("body_fact", h.project.resolve(), True) in h.calls
    [call] = h.wave.calls
    assert call.subject.spec.layer == "core" and (call.triad, call.coupling) == ([pool[0]], [])
    assert foreign["checklist"] == {"layer": "core", "body_fact": "false", "how": "remote_chain", "treat_as_body": True}
    assert foreign["enforcement_blocks"] is False

    # A root git cannot place (no remote, no copy binding) is raised: the body layer, the
    # owner's panel outside Cyber Pro, enforcement binding; ``how`` keeps the fact's value.
    local = _repo(tmp_path / "work" / "local", "local")
    _stage(local, "b.py", "b = 2\n")
    raised = h.run(subject="index", workspace_root=str(local), treat_as_body=True, reviewers=[pool[0]])
    call = h.wave.calls[-1]
    assert call.subject.spec.layer == "body" and (call.triad, call.coupling) == (pool, [])
    assert raised["checklist"] == {"layer": "body", "body_fact": "true", "how": "unknown", "treat_as_body": True}
    assert raised["panel"]["reviewers_subset_ignored"] is True and raised["enforcement_blocks"] is True

    _stage(local, "c.py", "c = 3\n")
    plain = h.run(subject="index", workspace_root=str(local), reviewers=[pool[0]], reason="one seat")
    assert plain["checklist"] == {"layer": "core", "body_fact": "unknown", "how": "unknown", "treat_as_body": False}
    assert h.wave.calls[-1].triad == [pool[0]] and plain["enforcement_blocks"] is False


def test_workspace_root_names_any_registered_repository(h: Harness, tmp_path: pathlib.Path) -> None:
    other = _repo(tmp_path / "work" / "other", "other")
    (other / "other.txt").write_text("live edit\n", encoding="utf-8")
    absolute = h.run(subject="worktree", workspace_root=str(other))
    assert (absolute["subject"]["root"], absolute["subject"]["kind"]) == (str(other.resolve()), "worktree")
    assert "live edit" in h.wave.calls[-1].subject.diff_text

    vendored = _repo(h.project / "vendor", "vendor")
    _stage(vendored, "v.py", "v = 1\n")
    relative = h.run(subject="index", workspace_root="vendor")
    assert relative["subject"]["root"] == str(vendored.resolve())
    assert [call.subject.spec.root_kind for call in h.wave.calls] == ["active_workspace", "active_workspace"]


# --- Panel ----------------------------------------------------------------------------

def test_one_named_seat_is_the_whole_panel_of_a_foreign_root(h: Harness) -> None:
    pool = _pool()
    _stage(h.project, "a.py", "a = 1\n")
    result = h.run(subject="index", reviewers=[pool[1]], reason="one cheap check")

    [call] = h.wave.calls
    assert (call.triad, call.coupling) == ([pool[1]], [])
    assert [row["seat_id"] for row in result["rows"]] == [pool[1]]
    assert (result["panel"]["composition"], result["panel"]["chosen_by"]) == ("composed", "author")
    assert (result["panel"]["reason"], result["panel"]["reason_missing"]) == ("one cheap check", False)
    # A retrieving seat alone is asked both parts: the one seat IS the wave's quorum.
    assert result["rows"][0]["parts"] == ["change", "coupling"] and result["per_question"]["coupling"] == rl.VERDICT_PASS
    assert result["aggregate"] == rl.VERDICT_PASS and result["quorum"]["required"] == 1


def test_a_narrowed_panel_without_a_reason_is_recorded_not_refused(h: Harness) -> None:
    pool = _pool()
    _stage(h.project, "a.py", "a = 1\n")
    narrowed = h.run(subject="index", reviewers=[pool[0]])
    assert narrowed["panel"]["reason_missing"] is True and len(h.wave.calls) == 1

    _stage(h.project, "b.py", "b = 2\n")
    everyone = h.run(subject="index", reviewers=pool)
    assert everyone["panel"]["composition"] == "composed" and everyone["panel"]["reason_missing"] is False


def test_no_list_seats_every_configured_seat(h: Harness) -> None:
    pool = _pool()
    _stage(h.project, "a.py", "a = 1\n")
    result = h.run(subject="index")
    [call] = h.wave.calls
    assert (call.triad, call.coupling) == (pool, [])
    # Each seat is asked the parts its delivery decides: both when it retrieves.
    assert call.parts == {slot.slot_id: tuple(rl.seat_parts(slot)) for slot in slots.review_pool_slots()}
    assert {len(parts) for parts in call.parts.values()} == {1, 2}
    assert result["panel"]["composition"] == "full_pool" and result["panel"]["reason_missing"] is False


def test_a_body_subset_outside_cyber_pro_is_ignored_and_recorded(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _pool()
    _stage(h.system, "body.py", "x = 1\n")
    ignored = h.run(root="system_repo", subject="index", reviewers=[pool[0]])
    assert (h.wave.calls[-1].triad, h.wave.calls[-1].coupling) == (pool, [])
    assert (ignored["panel"]["composition"], ignored["panel"]["reviewers_subset_ignored"]) == ("full_pool", True)

    _stage(h.system, "more.py", "y = 2\n")
    named_all = h.run(root="system_repo", subject="index", reviewers=pool)
    assert named_all["panel"]["reviewers_subset_ignored"] is False

    # An enabled row outside the pool is heard beside the owner's pool, not counted.
    _stage(h.system, "extra.py", "e = 3\n")
    added = h.run(root="system_repo", subject="index", reviewers=["scout"])
    assert (h.wave.calls[-1].triad, h.wave.calls[-1].coupling) == (pool, ["scout"])
    assert added["panel"]["additional"] == ["scout"] and added["panel"]["reviewers_subset_ignored"] is False
    assert {row["seat_id"]: row["additional"] for row in added["rows"]} == {**{seat: False for seat in pool}, "scout": True}

    monkeypatch.setattr(rc, "get_runtime_mode", lambda: "cyber_pro")
    composed = h.run(root="system_repo", subject="index", reviewers=[pool[0]], reason="cyber pro chooses")
    assert h.wave.calls[-1].triad == [pool[0]] and h.wave.calls[-1].coupling == []
    assert (composed["panel"]["composition"], composed["panel"]["reviewers_subset_ignored"]) == ("composed", False)


def test_a_composed_panel_counts_pool_seats_only_and_hears_an_unmarked_row_as_a_critic(
        h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """D1-02 / V02, decision 1A: the marked rows are the menu the author composes from.
    On a foreign root and on the body in Cyber Pro an enabled row the owner did not
    mark (``scout``) is seated as an added critic beside the named pool seats — its
    findings are additional, the quorum is the pool seats' — and a composition that
    names NO pool seat has no quorum and is refused before any wave."""
    pool = _pool()
    h.wave.failing = {"scout"}
    _stage(h.project, "a.py", "a = 1\n")
    foreign = h.run(subject="index", reviewers=[pool[0], "scout"], reason="one pool seat, one scout")
    assert (h.wave.calls[-1].triad, h.wave.calls[-1].coupling) == ([pool[0]], ["scout"])
    assert (foreign["panel"]["composition"], foreign["panel"]["additional"]) == ("composed", ["scout"])
    assert {row["seat_id"]: row["additional"] for row in foreign["rows"]} == {pool[0]: False, "scout": True}
    assert foreign["aggregate"] == rl.VERDICT_PASS and foreign["quorum"]["required"] == 1
    assert [f["seat_id"] for f in foreign["findings"]["additional_findings"]] == ["scout"]

    monkeypatch.setattr(rc, "get_runtime_mode", lambda: "cyber_pro")
    _stage(h.system, "body.py", "x = 1\n")
    body = h.run(root="system_repo", subject="index", reviewers=["scout", pool[1]], reason="cyber pro composes")
    assert (h.wave.calls[-1].triad, h.wave.calls[-1].coupling) == ([pool[1]], ["scout"])
    assert (body["panel"]["composition"], body["panel"]["additional"]) == ("composed", ["scout"])
    assert body["aggregate"] == rl.VERDICT_PASS

    _stage(h.system, "more.py", "y = 2\n")
    with pytest.raises(rc.ReviewChangeArgumentError, match="from the review pool"):
        h.run(root="system_repo", subject="index", reviewers=["scout"], reason="only the scout")
    assert len(h.wave.calls) == 2, "no wave was paid for a composition without a pool seat"


def test_an_added_critic_is_any_enabled_catalog_row_and_a_switched_off_row_is_refused(
        h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """The added critic outside the quorum is ANY enabled catalog row (the owner's words:
    Ouroboros keeps the possibility to take any model; BIBLE «I may add a critic») — an
    unmarked api row or an unmarked agent-session row with its own effort, named in
    ``reviewers`` or in ``coupling_only``, on the body below Cyber Pro (the pool judges, the
    rows only add) and in Cyber Pro (the named pool seat counts). The marked rows stay the
    only counted seats; a switched-off row is refused before any wave, like an unknown name."""
    pool = _pool()
    set_review_pool(monkeypatch, pool_roster(
        *mixed_pool_rows(), pool_seat("scout", "openai/scout-model", marked=False),
        pool_seat("session-critic", "codex=gpt-5.6-sol", kind="agent_session", effort="xhigh", marked=False),
        pool_seat("retired", "openai/retired-model", marked=False, enabled=False)))
    assert _pool() == pool, "unmarked rows do not change the pool"

    _stage(h.system, "body.py", "x = 1\n")
    below = h.run(root="system_repo", subject="index", reviewers=["session-critic"], coupling_only=["scout"])
    call = h.wave.calls[-1]
    assert (call.triad, call.coupling) == (pool, ["session-critic", "scout"])
    assert rl.PART_CHANGE in call.parts["session-critic"] and call.parts["scout"] == (rl.PART_COUPLING,)
    assert call.efforts["session-critic"] == "xhigh"
    assert (below["panel"]["composition"], below["panel"]["additional"]) == ("full_pool", ["session-critic", "scout"])
    assert {row["seat_id"]: row["additional"] for row in below["rows"]} == {
        **{seat: False for seat in pool}, "session-critic": True, "scout": True}

    monkeypatch.setattr(rc, "get_runtime_mode", lambda: "cyber_pro")
    _stage(h.system, "more.py", "y = 2\n")
    composed = h.run(root="system_repo", subject="index", reviewers=[pool[0], "session-critic"],
                     coupling_only=["scout"], reason="one pool seat, two critics")
    call = h.wave.calls[-1]
    assert (call.triad, call.coupling) == ([pool[0]], ["session-critic", "scout"])
    assert (composed["panel"]["composition"], composed["panel"]["additional"]) == ("composed", ["session-critic", "scout"])
    assert composed["quorum"]["required"] == 1

    _stage(h.system, "later.py", "z = 3\n")
    paid = len(h.wave.calls)
    for args in ({"reviewers": [pool[0], "retired"]}, {"reviewers": [pool[0]], "coupling_only": ["retired"]}):
        with pytest.raises(rc.ReviewChangeArgumentError, match="switched off"):
            h.run(root="system_repo", subject="index", reason="a retired row", **args)
    assert len(h.wave.calls) == paid, "no wave was paid for a refused composition"


def test_reviewer_effort_is_this_waves_order(h: Harness) -> None:
    pool = _pool()
    _stage(h.project, "a.py", "a = 1\n")
    ordered = h.run(subject="index", reviewer_effort="low")
    assert set(h.wave.calls[-1].efforts.values()) == {"low"}
    effort = ordered["panel"]["reviewer_effort"]
    assert effort["order"] == "low" and sorted(effort["applied"]) == sorted(pool)

    own = {slot.slot_id: slot.effort for slot in slots.review_pool_slots()}
    _stage(h.project, "b.py", "b = 2\n")
    h.run(subject="index")
    assert h.wave.calls[-1].efforts == own


def test_coupling_only_critics_answer_outside_the_quorum(h: Harness) -> None:
    pool = _pool()
    critic = pool[1]
    h.wave.failing = {critic}
    _stage(h.project, "a.py", "a = 1\n")
    result = h.run(subject="index", reviewers=[pool[0]], coupling_only=[critic], reason="extra eyes")

    [call] = h.wave.calls
    assert (call.triad, call.coupling) == ([pool[0]], [critic]) and call.parts[critic] == (rl.PART_COUPLING,)
    assert result["aggregate"] == rl.VERDICT_PASS and result["per_question"]["coupling"] == rl.VERDICT_PASS
    assert [f["seat_id"] for f in result["findings"]["additional_findings"]] == [critic]
    assert result["findings"]["critical_findings"] == [] and result["panel"]["additional"] == [critic]
    assert {row["seat_id"]: row["additional"] for row in result["rows"]} == {pool[0]: False, critic: True}
    # NEW-W3: the FINAL record's panel block describes everyone who sat — one assigned seat,
    # one added critic, two models — while the verdict and its quorum stay the assigned seat's.
    record = h.written[result["record_id"]]
    models = {row["seat_id"]: row["observed_model"] for row in record["rows"]}
    assert len(set(models.values())) == 2, "the fixture seats two different models"
    panel = record["panel"]
    assert (panel["seats"], panel["additional_seats"], panel["assigned"], panel["additional"]) == (1, 1, [pool[0]], [critic])
    assert (panel["distinct_models"], panel["distinct_engines"], panel["single_model_panel"]) == (2, 2, False)
    assert record["verdict"]["quorum"]["assigned"] == 1 and record["verdict"]["quorum"]["required"] == 1

    _stage(h.project, "b.py", "b = 2\n")
    assigned = h.run(subject="index", reviewers=[pool[0], critic], reason="the critic decides")
    assert assigned["aggregate"] == rl.VERDICT_FAIL


# --- Identities: reuse, retry and the shared ceiling ------------------------------------

def test_the_same_identity_returns_the_settled_record_free(h: Harness) -> None:
    pool = _pool()
    _stage(h.project, "a.py", "a = 1\n")
    first = h.run(subject="index", goal="Bound the cache")
    again = h.run(subject="index", goal="Bound the cache")
    assert len(h.wave.calls) == 1 and again["reused"] is True and first["reused"] is False
    assert again["record_id"] == first["record_id"] and again["aggregate"] == first["aggregate"]
    assert again["cost"] == {"usd": 0.0, "unknown": False} and first["cost"]["usd"] > 0

    h.run(subject="index", reviewers=[pool[0]], reason="another panel")
    assert len(h.wave.calls) == 2
    rebutted = h.run(subject="index", review_rebuttal="The finding is stale: line 3 already bounds it.")
    assert len(h.wave.calls) == 3 and rebutted["reused"] is False
    assert h.wave.calls[2].rebuttal.startswith("The finding is stale")
    # A new round is a new PHYSICAL operation too (identity c carries the round): the
    # custody layer must not hand the first round's answers back to the rebuttal.
    assert h.wave.calls[2].retry_key != h.wave.calls[0].retry_key
    same_round = h.run(subject="index", review_rebuttal="The finding is stale: line 3 already bounds it.")
    assert len(h.wave.calls) == 3 and same_round["reused"] is True and same_round["record_id"] == rebutted["record_id"]
    h.run(subject="index", author_questions=["Is the cache bounded?"])
    assert len(h.wave.calls) == 4 and h.wave.calls[3].retry_key not in {c.retry_key for c in h.wave.calls[:3]}
    # The semantic brief is part of the round: another goal or scope is another wave;
    # the unchanged request (same goal) is the settled record, free.
    h.run(subject="index", goal="Unbound the cache")
    h.run(subject="index", goal="Bound the cache", scope="only the cache module")
    assert len(h.wave.calls) == 6 and len({c.retry_key for c in h.wave.calls}) == 6
    assert h.run(subject="index", goal="Bound the cache")["reused"] is True and len(h.wave.calls) == 6
    assert [attempt.attempt for attempt in h.attempts(h.project)] == [1, 2, 3, 4, 5, 6]


def test_an_undecided_record_is_not_reused(h: Harness) -> None:
    _stage(h.project, "a.py", "a = 1\n")
    h.wave.status = "error"
    undecided = h.run(subject="index")
    assert undecided["aggregate"] not in {rl.VERDICT_PASS, rl.VERDICT_FAIL}
    h.wave.status = "responded"
    decided = h.run(subject="index")
    assert len(h.wave.calls) == 2 and decided["reused"] is False and decided["aggregate"] == rl.VERDICT_PASS


def test_the_wave_runs_under_its_own_identities_and_restores_the_task(h: Harness) -> None:
    _stage(h.project, "a.py", "a = 1\n")
    h.ctx._current_review_tool_name = "commit_reviewed"
    h.ctx._current_review_retry_key = "the-gate-key"
    h.ctx._review_history = ["a gate round"]
    result = h.run(subject="index")

    [call] = h.wave.calls
    frozen = call.subject
    assert call.tool == "review_change" and call.record_id == result["record_id"]
    round_sha = review_round_sha(frozen)  # no rebuttal, no questions, empty brief: the bare round
    assert call.retry_key == review_retry_key(frozen, round_sha=round_sha) != review_retry_key(frozen)
    assert call.retry_key.startswith("review:") and f":index:{frozen.diff_sha}:change:" in call.retry_key
    record = rl.load_record(h.drive, result["record_id"])
    assert record["fingerprints"]["retry_key"] == call.retry_key and record["fingerprints"]["reuse_key"]
    assert (h.ctx._current_review_tool_name, h.ctx._current_review_retry_key, h.ctx._review_history) == (
        "commit_reviewed", "the-gate-key", ["a gate round"])
    assert getattr(h.ctx, "_review_paid_stamp", None) is None


def test_the_settled_record_is_bound_to_the_seats_own_execution_rows(
        h: Harness, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """The seats record their executions during the wave (keyed by seat, stamped with
    the wave's timestamps); settling names the record on exactly those rows — the
    gate's ``bind_reviewer_slot_record_id(executions, record_id)`` contract."""
    monkeypatch.setattr(slots, "_last_execution_path", lambda: tmp_path / "reviewer_last_execution.json")
    wave, pool = h.wave, _pool()

    def recording(ctx: Any, commit_message: str, **kwargs: Any):
        answer = wave(ctx, commit_message, **kwargs)
        seated = {slot.slot_id: slot for slot in slots.review_pool_slots()}
        actors = [SimpleNamespace(slot_id=row["slot_id"], status="responded", usage={}, operation_state="settled")
                  for row in ctx._last_triad_raw_results]
        slots.record_reviewer_slot_executions("change", actors, seated, keep_on=ctx)
        return answer

    monkeypatch.setattr(rc, "run_parallel_review", recording)
    _stage(h.project, "a.py", "a = 1\n")
    result = h.run(subject="index")
    assert result["state"] == "settled" and result["record_id"]
    last = slots.reviewer_slot_last_executions()
    assert {seat: last[seat].get("review_record_id") for seat in pool} == {seat: result["record_id"] for seat in pool}


def test_the_ceiling_is_shared_by_every_subject_of_one_root(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    from ouroboros.review_state import CommitAttemptRecord, make_repo_key, update_state

    monkeypatch.setattr(commit_gate, "review_max_cycles", lambda: 1)
    base = _git(h.project, "rev-parse", "HEAD")
    head = _commit(h.project, "feature.py", "x = 1\n")
    h.run(subject="base..head", base=base, head=head)
    _stage(h.project, "other.py", "y = 2\n")
    refused = h.run(subject="index")

    assert len(h.wave.calls) == 1
    assert refused["dispatch_refusal"]["kind"] == "review_cycles_exhausted" and refused["message"]
    assert refused["aggregate"] == rl.VERDICT_NOT_DISPATCHED and refused["rows"] == []
    assert rl.load_record(h.drive, refused["record_id"])["dispatch_refusal"]["kind"] == "review_cycles_exhausted"
    text = h.tool(subject="index")
    assert text.startswith(refused["message"]) and '"kind": "review_cycles_exhausted"' in text
    assert h.run(subject="base..head", base=base, head=head)["reused"] is True
    assert len(h.wave.calls) == 1

    # Another root keeps its own ceiling, and the gate's spend on that root is not this tool's.
    update_state(h.drive, lambda state: state.record_attempt(CommitAttemptRecord(
        ts="2026-10-07T00:00:00+00:00", commit_message="gate", task_id="task-rc", root_task_id="task-rc",
        repo_key=make_repo_key(h.system.resolve()), tool_name="commit_reviewed", status="reviewed", paid=True, attempt=1)))
    _stage(h.system, "body.py", "z = 3\n")
    assert h.run(root="system_repo", subject="index")["reused"] is False
    assert len(h.wave.calls) == 2


# --- The call surface ---------------------------------------------------------------------

def test_author_questions_reach_the_wave_and_the_record_verbatim(h: Harness) -> None:
    questions = ["Does the retry loop terminate?", "  Is the cache\nbounded?  "]
    _stage(h.project, "a.py", "a = 1\n")
    result = h.run(subject="index", goal="Bound the cache", author_questions=questions)
    [call] = h.wave.calls
    assert call.goal.startswith("Bound the cache") and all(question in call.goal for question in questions)
    record = rl.load_record(h.drive, result["record_id"])
    assert record["brief"]["author_questions"] == questions and record["brief"]["goal"] == "Bound the cache"


@pytest.mark.parametrize("args, needle", [
    ({"subject": "base..head", "base": "HEAD"}, "needs both base and head"),
    ({"subject": "base..head", "head": "HEAD"}, "needs both base and head"),
    ({"subject": "everything"}, "subject must be one of"),
    ({"subject": "index", "head": "HEAD"}, "head belongs to subject=base..head"),
    ({"subject": "index", "base": "--output=/tmp/x"}, "must name a revision"),
    ({"subject": "index", "root": "/somewhere/else"}, "root must be one of"),
    ({"subject": "index", "root": "system_repo", "workspace_root": "project"}, "leave workspace_root empty"),
    ({"subject": "index", "surface": "advisory"}, "surface must be one of change, preflight, system"),
    ({"subject": "index", "surface": "preflight"}, "surface=preflight seats exactly one reviewer"),
    ({"subject": "index", "surface": "preflight", "reviewers": ["a", "b"]}, "surface=preflight seats exactly one reviewer"),
    ({"subject": "index", "surface": "system"}, "subject=system goes with surface=system"),
    ({"subject": "system"}, "subject=system goes with surface=system"),
    ({"subject": "system", "surface": "system", "base": "HEAD"}, "workspace_root, base and head do not apply"),
    ({"subject": "system", "surface": "system", "coupling_only": ["x"]}, "coupling_only does not apply"),
    ({"subject": "index", "reviewer_effort": "turbo"}, "reviewer_effort must be one of"),
    ({"subject": "index", "reviewers": "slot_1"}, "reviewers must be a list of strings"),
    ({"subject": "index", "reviewers": ["nobody-at-all"]}, "is not an enabled catalog row"),
    ({"subject": "base..head", "base": "0" * 40, "head": "HEAD"}, "is not a commit"),
    ({"subject": "index", "treat_as_body": "yes"}, "treat_as_body must be a boolean"),
])
def test_call_errors_are_typed_and_dispatch_nothing(h: Harness, args: Dict[str, Any], needle: str) -> None:
    _stage(h.project, "a.py", "a = 1\n")
    text = h.tool(**args)
    assert text.startswith("⚠️ TOOL_ARG_ERROR (review_change): ") and needle in text
    assert text.endswith("No reviewer was dispatched.")
    assert h.wave.calls == [] and h.written == {}


def test_a_folder_outside_the_registered_roots_or_without_a_change_is_refused(
        h: Harness, tmp_path: pathlib.Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    from ouroboros.tools.registry import ToolContext

    elsewhere = _repo(tmp_path_factory.mktemp("elsewhere") / "repo", "elsewhere")
    _stage(elsewhere, "e.py", "e = 1\n")
    plain = ToolContext(repo_dir=h.system, system_repo_dir=h.system, drive_root=h.drive, task_id="task-rc")
    outside_home = rc._handle_review_change(plain, subject="index", workspace_root=str(elsewhere))
    missing = h.tool(subject="index", workspace_root=str(tmp_path / "work" / "missing"))
    for refused in (outside_home, missing):
        assert refused.startswith("⚠️ TOOL_ARG_ERROR (review_change): ")
        assert "is not a registered folder this task can read" in refused
    assert "is outside the user_files home" in outside_home

    readable = h.tool(subject="index", workspace_root=str(h.drive))
    assert readable.startswith("⚠️ TOOL_ARG_ERROR (review_change): ") and "is not inside a git repository" in readable
    clean = h.tool(subject="index")
    assert clean.startswith("⚠️ TOOL_ARG_ERROR (review_change): ") and "has no change to review" in clean
    assert h.wave.calls == [] and h.written == {}


def test_the_dispatcher_binds_the_root_and_answers_with_the_record(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    registry = ToolRegistry(h.system, h.drive)
    registry.set_context(h.ctx)
    schema = registry.get_schema_by_name("review_change")
    assert schema["function"]["parameters"]["properties"]["root"]["enum"] == ["active_workspace", "system_repo"]

    _stage(h.project, "a.py", "a = 1\n")
    _stage(h.system, "body.py", "x = 1\n")
    project = json.loads(registry.execute("review_change", {"subject": "index"}))
    system = json.loads(registry.execute("review_change", {"subject": "index", "root": "system_repo"}))
    assert project["subject"]["root"] == str(h.project.resolve()) and project["checklist"]["layer"] == "core"
    assert system["subject"]["root"] == str(h.system.resolve()) and system["checklist"]["layer"] == "body"
    assert {project["record_id"], system["record_id"]} == set(h.written)
    refused = registry.execute("review_change", {"subject": "base..head", "base": "HEAD"})
    assert "TOOL_ARG_ERROR" in refused and len(h.wave.calls) == 2
