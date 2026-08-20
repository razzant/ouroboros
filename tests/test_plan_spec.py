"""Contract tests for ``ouroboros.tools.plan_spec`` + ``plan_packet`` (Phase B of the
plan_task redesign, 2026-08-15): schema, ids, hash, delta, the ONE structural
path fact, evidence manifest + hash, findings protocol, host aggregate, the
closure table, packet builders — and domain independence (a spec with zero
file paths is first-class)."""

from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros.config import adaptive_quorum
from ouroboros.tools.plan_render import _parse_plan_review_control
from ouroboros.tools import plan_evidence, plan_packet, plan_spec
from ouroboros.tools.review_synthesis import PLAN_REVIEW_CONTROL_PREFIX


CODE_SPEC = {
    "goal": "Add a retry wrapper around the flaky uploader",
    "in_scope": ["retry wrapper", "unit tests"],
    "non_goals": ["changing the upload protocol"],
    "acceptance_claims": [
        {"claim": "uploads retry 3 times", "surface": "tests", "support": "pytest -k retry"},
        "a plain-string claim",
    ],
    "invariants": ["no new dependencies", "public API unchanged"],
    "decisions": [{"choice": "exponential backoff", "rejected": ["fixed delay"], "why": "bursty upstream"}],
    "deferred": [{"what": "metrics emission", "why_safe_to_defer": "observability only"}, "log wording"],
    "affected_resources": ["ouroboros/uploader.py"],
    "evidence": ["ouroboros/uploader.py", "https://example.com/spec", "task:abc123"],
}

DECK_SPEC = {
    "goal": "Prepare a 5-slide deck on Q3 results for the board",
    "in_scope": ["5 slides", "one chart per slide"],
    "non_goals": ["speaker notes"],
    "acceptance_claims": ["deck has exactly 5 slides", "every number traces to the Q3 report"],
    "invariants": ["deadline Friday 17:00", "no confidential customer names"],
    "decisions": [{"choice": "one message per slide", "rejected": ["dense slides"], "why": "board attention"}],
    "deferred": ["colour palette"],
    "affected_resources": [],
    "evidence": [],
}


# ------------------------------------------------------------------ B1 spec schema


def test_normalize_spec_mints_ids_and_keeps_schema():
    spec, errors = plan_spec.normalize_spec(CODE_SPEC)
    assert errors == []
    assert [c["id"] for c in spec["acceptance_claims"]] == ["claim_1", "claim_2"]
    assert spec["acceptance_claims"][1]["claim"] == "a plain-string claim"
    assert spec["decisions"][0]["id"] == "decision_1"
    assert [d["id"] for d in spec["deferred"]] == ["deferred_1", "deferred_2"]
    assert spec["deferred"][1] == {"id": "deferred_2", "what": "log wording", "why_safe_to_defer": ""}
    assert plan_spec.spec_ids(spec) == frozenset({
        "goal", "claim_1", "claim_2", "invariant_1", "invariant_2", "decision_1", "deferred_1", "deferred_2",
    })
    assert spec["normalization_omissions"] == []
    assert list(spec) == [
        "goal", "in_scope", "non_goals", "acceptance_claims", "invariants", "decisions",
        "deferred", "affected_resources", "evidence", "normalization_omissions",
    ]


def test_normalize_spec_errors_are_typed():
    _spec, errors = plan_spec.normalize_spec({"in_scope": ["x"], "bogus": 1})
    assert any(e.startswith("goal:") for e in errors)
    assert any("unknown fields: bogus" in e for e in errors)
    _spec, errors = plan_spec.normalize_spec({"goal": "g", "decisions": [{"why": "no choice"}]})
    assert errors == ["decisions[0]: choice is required"]
    _spec, errors = plan_spec.normalize_spec({"goal": "g", "in_scope": [{"a": 1}]})
    assert errors == ["in_scope[0]: must be a string"]
    assert plan_spec.normalize_spec("not a mapping") == ({}, ["spec: must be an object"])  # type: ignore[arg-type]


def test_normalize_spec_bounds_lists_with_recorded_omission():
    spec, errors = plan_spec.normalize_spec({
        "goal": "g", "in_scope": [f"item {i}" for i in range(plan_spec.MAX_LIST_ITEMS + 3)],
    })
    assert errors == []
    assert len(spec["in_scope"]) == plan_spec.MAX_LIST_ITEMS
    assert spec["normalization_omissions"] == [
        f"in_scope: {plan_spec.MAX_LIST_ITEMS + 3} items declared, kept the first "
        f"{plan_spec.MAX_LIST_ITEMS} (bound {plan_spec.MAX_LIST_ITEMS})"
    ]
    # B-10: nested cap on decision.rejected, recorded the same way.
    spec, errors = plan_spec.normalize_spec({
        "goal": "g", "decisions": [{"choice": "c", "rejected": [f"r{i}" for i in range(plan_spec.MAX_REJECTED_PER_DECISION + 2)]}],
    })
    assert errors == [] and len(spec["decisions"][0]["rejected"]) == plan_spec.MAX_REJECTED_PER_DECISION
    assert spec["normalization_omissions"] == [
        f"decisions[0].rejected: {plan_spec.MAX_REJECTED_PER_DECISION + 2} items declared, kept the first "
        f"{plan_spec.MAX_REJECTED_PER_DECISION} (bound {plan_spec.MAX_REJECTED_PER_DECISION})"
    ]


def test_normalize_spec_goal_type_and_bool_scalars():
    _spec, errors = plan_spec.normalize_spec({"goal": ["a", "b"]})
    assert errors == ["goal: must be a string"]
    _spec, errors = plan_spec.normalize_spec({"goal": True})
    assert errors == ["goal: must be a string"]
    spec, errors = plan_spec.normalize_spec({"goal": "g", "in_scope": [0, True, "x", 2.5]})
    assert errors == ["in_scope[1]: must be a string"]
    assert spec["in_scope"] == ["0", "x", "2.5"]  # 0 is a value, True is not a spec item


def test_normalize_spec_tolerates_bare_strings_and_mints_every_id():
    spec, errors = plan_spec.normalize_spec({
        "goal": "g", "in_scope": "single item", "decisions": "just a choice",
        "deferred": {"id": "later!", "what": "w"},
        "acceptance_claims": [{"id": "invariant_1", "claim": "collides with a positional invariant id"}],
        "invariants": ["i1"],
    })
    assert errors == []
    assert spec["in_scope"] == ["single item"]
    assert spec["decisions"] == [{"id": "decision_1", "choice": "just a choice", "rejected": [], "why": ""}]
    assert spec["deferred"][0]["id"] == "deferred_1"
    # Ids are HOST-MINTED positionally: a caller-chosen id can never shadow another element's
    # id (it is the only thing a blocking finding may name), and the declared one is kept.
    assert spec["acceptance_claims"][0]["id"] == "claim_1"
    assert spec["acceptance_claims"][0]["declared_id"] == "invariant_1"
    assert len(plan_spec.spec_ids(spec)) == 5  # goal + invariant_1 + claim_1 + decision_1 + deferred_1
    # D32: a supplied id "goal" cannot shadow the reserved goal id either.
    spec, _ = plan_spec.normalize_spec({"goal": "g", "decisions": [{"id": "goal", "choice": "c"}]})
    assert spec["decisions"][0]["id"] == "decision_1"


def test_spec_hash_stable_under_key_order_and_delta_reports_changes():
    spec, _ = plan_spec.normalize_spec(CODE_SPEC)
    reordered = {k: spec[k] for k in reversed(list(spec))}
    assert plan_spec.spec_hash(spec) == plan_spec.spec_hash(reordered)
    assert plan_spec.spec_hash(spec) == plan_spec.spec_hash(json.loads(json.dumps(spec, indent=4)))
    changed_raw = json.loads(json.dumps(CODE_SPEC))
    changed_raw["goal"] = "Add a retry wrapper (v2)"
    changed_raw["invariants"].append("no network in tests")
    changed_raw["decisions"][0]["why"] = "different rationale"
    changed_raw["deferred"] = ["log wording"]
    changed, _ = plan_spec.normalize_spec(changed_raw)
    assert plan_spec.spec_hash(changed) != plan_spec.spec_hash(spec)
    delta = plan_spec.spec_delta(spec, changed)
    assert delta["changed"] and delta["goal_changed"]
    assert delta["ids"]["changed"] == ["decision_1", "deferred_1"]  # deferred_1 now carries the former deferred_2 text
    assert delta["ids"]["removed"] == ["deferred_2"]
    assert delta["lists"]["invariants"] == {"added": ["no network in tests"], "removed": []}
    assert plan_spec.spec_delta(spec, spec)["changed"] is False
    assert plan_spec.spec_delta(None, spec)["ids"]["added"] == ["claim_1", "claim_2", "decision_1", "deferred_1", "deferred_2"]


def test_spec_delta_reports_renumbered_ids_and_convergence_rule_says_retarget():
    prev, _ = plan_spec.normalize_spec({"goal": "g", "invariants": ["budget <= $10", "no data loss", "deadline Friday"],
                                        "deferred": ["colours", "fonts"]})
    cur, _ = plan_spec.normalize_spec({"goal": "g", "invariants": ["no data loss", "deadline Friday"],
                                       "deferred": ["fonts"]})
    delta = plan_spec.spec_delta(prev, cur)
    assert delta["renumbered"] == [
        {"from": "invariant_2", "to": "invariant_1", "text": "no data loss"},
        {"from": "invariant_3", "to": "invariant_2", "text": "deadline Friday"},
        {"from": "deferred_2", "to": "deferred_1", "text": "fonts"},
    ]
    assert delta["lists"]["invariants"] == {"added": [], "removed": ["budget <= $10"]}
    assert plan_spec.spec_delta(cur, cur)["renumbered"] == []
    # A reviewer re-targeting via the delta lands on the right element; the stale id is demoted.
    stale, disclosures, _ = plan_spec.validate_findings(
        [{"id": "f1", "class": "blocking", "breaks": "invariant_3", "summary": "deadline"}],
        spec_ids=plan_spec.spec_ids(cur), seen_locators=(),
    )
    assert stale[0]["class"] == "note" and disclosures == ["blocking_without_valid_breaks:f1"]
    retargeted, disclosures, _ = plan_spec.validate_findings(
        [{"id": "f1", "class": "blocking", "breaks": "invariant_2", "summary": "deadline"}],
        spec_ids=plan_spec.spec_ids(cur), seen_locators=(),
    )
    assert retargeted[0]["class"] == "blocking" and disclosures == []
    from ouroboros.tools.plan_packet import build_plan_review_user_content

    user = build_plan_review_user_content(
        objective="o", goal="g", plan_prose="p", spec=cur,
        manifest={"declared": [], "attached": [], "omissions": []},
        prior_cycles=[{"cycle_index": 1, "aggregate": "REVIEW_REQUIRED", "findings": []}],
        dispositions=[], spec_delta=delta, root_exploration_log=None, cycle_index=2)
    assert "re-target `breaks` against the CURRENT spec ids" in user and "renumbered" in user


# ------------------------------------------------------------ B2 constitutional


def test_constitutional_active_binding_alone_does_not_decide(tmp_path):
    """D29: active == system with no declared system path is NOT constitutional."""
    system = tmp_path / "repo"
    system.mkdir()
    ok, note = plan_spec.resolve_constitutional(
        active_root=system, system_repo_root=system, affected_resources=[], evidence=[],
    )
    assert ok is False and note.startswith("not constitutional")
    ok, _ = plan_spec.resolve_constitutional(
        active_root=system, system_repo_root=system, affected_resources=["ouroboros/loop.py"], evidence=[],
    )
    assert ok is True  # a relative path under the (system) active root still decides


def test_constitutional_declared_paths_relative_absolute_and_file_scheme(tmp_path):
    system = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    system.mkdir()
    workspace.mkdir()
    ok, note = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_resources=["../repo/ouroboros/loop.py"], evidence=[],
    )
    assert ok and "affected_resources" in note
    # An EVIDENCE locator counts only when it actually exists: a plan that merely wants to
    # LOOK at a repo file is constitutional, a typo pointing at nothing is not (E2E finding).
    (system / "BIBLE.md").write_text("# constitution\n", encoding="utf-8")
    ok, note = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_resources=["src/app.py"], evidence=[str(system / "BIBLE.md")],
    )
    assert ok and "evidence" in note
    ok, _ = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_resources=["src/app.py"], evidence=[str(system / "NO_SUCH_FILE.md")],
    )
    assert ok is False
    # ...while a TARGET counts whether or not it exists yet — creating a module is self-modification.
    ok, _ = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_resources=[str(system / "ouroboros" / "brand_new_module.py")], evidence=[],
    )
    assert ok is True
    ok, _ = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_resources=["src/app.py"], evidence=["docs/x.md"],
    )
    assert ok is False
    # B-06: a system-repo path dressed as file:// is still a path.
    ok, note = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_resources=[f"file://{system}/ouroboros/loop.py"], evidence=[],
    )
    assert ok is True and "file://" in note


def test_constitutional_payload_exempt_url_task_never_and_empty_false(tmp_path):
    system = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    payload = tmp_path / "data" / "skills" / "native" / "demo"  # data plane, outside the system repo
    payload.mkdir(parents=True)
    workspace.mkdir()
    (system / "x").mkdir(parents=True)
    ok, _ = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_resources=[str(payload / "SKILL.md")], evidence=[], payload_roots=[payload],
    )
    assert ok is False
    # A payload root that resolves INSIDE the system repo still exempts: the roots come
    # from the frozen skill-payload predicate, and a skill task whose active workspace IS
    # the repo touches only data-plane payload files (task 7881ad77902d4d25). Reviewer
    # proposal S-B05 (drop such roots) was rejected: it re-broke that frozen behaviour.
    nested = system / "data" / "skills" / "native" / "demo"
    ok, _ = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_resources=[str(nested / "SKILL.md")], evidence=[], payload_roots=[nested],
    )
    assert ok is False
    # A system path OUTSIDE every payload root is still constitutional.
    ok, _ = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_resources=[str(system / "ouroboros" / "loop.py")], evidence=[], payload_roots=[nested],
    )
    assert ok is True
    ok, _ = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system, affected_resources=[],
        evidence=["https://github.com/razzant/ouroboros/blob/main/ouroboros/loop.py", "task:xyz"],
    )
    assert ok is False
    ok, note = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system, affected_resources=[], evidence=[],
    )
    assert ok is False and note.startswith("not constitutional")


# ------------------------------------------------------------------ B3 evidence


def _write(path: pathlib.Path, text: str | bytes) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        # bytes, not write_text: Windows newline translation would change the on-disk hash
        path.write_bytes(text.encode("utf-8"))
    return path


def test_resolve_evidence_classifies_every_absence(tmp_path):
    root = tmp_path / "ws"
    _write(root / "notes.md", "hello evidence")
    _write(root / ".env", "SECRET=1")
    _write(root / "creds" / "credentials.json", "{}")
    _write(root / "blob.bin", b"\x00\x01\x02binary")
    (root / "adir").mkdir()
    outside = _write(tmp_path / "elsewhere.txt", "nope")
    manifest = plan_evidence.resolve_evidence(
        ["notes.md", ".env", "creds/credentials.json", "blob.bin", "adir", "missing.txt",
         str(outside), "https://example.com/x", "task:42", "notes.md", "  "],
        active_root=root, allowed_roots=[root],
    )
    assert manifest["declared"] == [
        "notes.md", ".env", "creds/credentials.json", "blob.bin", "adir", "missing.txt",
        str(outside), "https://example.com/x", "task:42",
    ]
    assert [a["locator"] for a in manifest["attached"]] == ["notes.md"]
    attached = manifest["attached"][0]
    assert attached["kind"] == "path" and attached["text"] == "hello evidence"
    assert attached["bytes"] == attached["attached_bytes"] == len("hello evidence")
    assert {(o["locator"], o["reason"]) for o in manifest["omissions"]} == {
        (".env", "sensitive"), ("creds/credentials.json", "sensitive"), ("blob.bin", "binary"),
        ("adir", "directory"), ("missing.txt", "missing"), (str(outside), "outside_allowed_roots"),
        ("https://example.com/x", "url_not_fetched"), ("task:42", "unsupported_locator"),
        ("notes.md", "duplicate_locator"),
    }
    assert manifest["bounds"] == {
        "per_item_bytes": plan_evidence.EVIDENCE_PER_ITEM_BYTES, "total_bytes": plan_evidence.EVIDENCE_TOTAL_BYTES,
    }


def test_resolve_evidence_truncation_budget_and_task_callback(tmp_path):
    root = tmp_path / "ws"
    _write(root / "big.txt", "x" * 100)
    _write(root / "small.txt", "y" * 10)
    _write(root / "later.txt", "z" * 10)
    manifest = plan_evidence.resolve_evidence(
        ["big.txt", "small.txt", "later.txt", "task:t1", "task:t2"],
        active_root=root, allowed_roots=[root], per_item_bytes=40, total_bytes=59,
        resolve_task=lambda tid: "task text" if tid == "t1" else None,
    )
    by_locator = {a["locator"]: a for a in manifest["attached"]}
    assert by_locator["big.txt"]["text"] == "x" * 40
    assert by_locator["big.txt"]["bytes"] == 100 and by_locator["big.txt"]["attached_bytes"] == 40
    assert by_locator["small.txt"]["text"] == "y" * 10
    assert "later.txt" not in by_locator  # 40 + 10 = 50 used; 10 more would exceed 59 (no partial fill)
    assert by_locator["task:t1"]["kind"] == "task"  # 9 bytes still fit after later.txt was refused
    assert by_locator["task:t1"]["text"] == "task text"
    assert [(o["locator"], o["reason"]) for o in manifest["omissions"]] == [
        ("big.txt", "truncated_to_40"), ("later.txt", "budget_exhausted"), ("task:t2", "task_not_found"),
    ]


def test_resolve_evidence_streams_full_hash_symlink_escape_and_multibyte_cut(tmp_path):
    import hashlib

    root = tmp_path / "ws"
    body = ("é" * 50 + "\n") * 4  # 2-byte chars: a byte cut lands mid-character
    _write(root / "utf.txt", body)
    (tmp_path / "secret_outside.txt").write_text("outside", encoding="utf-8")
    (root / "link.txt").symlink_to(tmp_path / "secret_outside.txt")
    (root / "envlink").symlink_to(root / "utf.txt")
    manifest = plan_evidence.resolve_evidence(
        ["utf.txt", "link.txt", "envlink", "\x00bad"], active_root=root, allowed_roots=[root], per_item_bytes=102,
    )
    utf = manifest["attached"][0]
    assert utf["sha256"] == hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert utf["bytes"] == len(body.encode("utf-8")) and utf["attached_bytes"] <= 102
    assert utf["text"] == "é" * 50 + "\n"  # the half character at byte 102 is dropped, never a decode crash
    reasons = {(o["locator"], o["reason"]) for o in manifest["omissions"]}
    assert ("utf.txt", "truncated_to_102") in reasons
    assert ("link.txt", "outside_allowed_roots") in reasons  # symlink target is what is judged
    assert ("\x00bad", "unsupported_locator") in reasons
    assert [a["locator"] for a in manifest["attached"]] == ["utf.txt", "envlink"]


def test_resolve_evidence_and_constitutional_never_raise_on_hostile_locators(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "loop1").symlink_to(root / "loop2")
    (root / "loop2").symlink_to(root / "loop1")
    long = "a" * 300
    manifest = plan_evidence.resolve_evidence(
        [long, long + "/x.txt", "loop1", "\x00bad", "task:s"], active_root=root, allowed_roots=[root],
        resolve_task=lambda _tid: "bad \ud800 surrogate",
    )
    reasons = {o["locator"]: o["reason"] for o in manifest["omissions"]}
    assert reasons[long] in {"unreadable", "missing"} and reasons[long + "/x.txt"] in {"unreadable", "missing"}
    assert reasons["loop1"] == "symlink_loop"
    assert reasons["\x00bad"] == "unsupported_locator"
    assert reasons["task:s"] == "binary"
    assert manifest["attached"] == []
    for locator in (long, "loop1", "\x00bad"):
        ok, _ = plan_spec.resolve_constitutional(
            active_root=root, system_repo_root=tmp_path / "repo", affected_resources=[locator], evidence=[],
        )
        assert ok is False
    # A fifo / device is a non-regular file: refused, never opened (would block forever).
    import os
    if hasattr(os, "mkfifo"):  # no fifos on Windows; the regular-file guard is exercised on POSIX
        os.mkfifo(root / "fifo")
        assert plan_evidence.resolve_evidence(["fifo"], active_root=root, allowed_roots=[root])["omissions"] == [
            {"locator": "fifo", "reason": "unreadable"}
        ]


def test_resolve_evidence_size_ceiling_latin1_and_lexical_sensitive(tmp_path):
    root = tmp_path / "ws"
    _write(root / "big.log", "x" * 5000)
    manifest = plan_evidence.resolve_evidence(["big.log"], active_root=root, allowed_roots=[root], hash_bytes_limit=4096)
    assert manifest["omissions"] == [{"locator": "big.log", "reason": "too_large:5000"}] and manifest["attached"] == []
    # B-11: invalid UTF-8 beyond the sniff window is `binary`, never silently dropped.
    _write(root / "latin.txt", b"a" * 9000 + "caf\xe9 au lait\n".encode("latin-1") * 3)
    manifest = plan_evidence.resolve_evidence(["latin.txt"], active_root=root, allowed_roots=[root], per_item_bytes=9100)
    assert manifest["omissions"] == [{"locator": "latin.txt", "reason": "binary"}]
    # S-B06: a `.env` symlink pointing at an ordinary file is sensitive by its lexical name.
    _write(root / "ordinary.txt", "plain")
    (root / ".env").symlink_to(root / "ordinary.txt")
    (root / "cfg").mkdir()
    (root / "cfg" / "plain.txt").symlink_to(root / "ordinary.txt")
    manifest = plan_evidence.resolve_evidence([".env", "cfg/plain.txt"], active_root=root, allowed_roots=[root])
    assert {(o["locator"], o["reason"]) for o in manifest["omissions"]} == {(".env", "sensitive")}
    assert [a["locator"] for a in manifest["attached"]] == ["cfg/plain.txt"]


def test_resolve_evidence_read_text_override_and_unreadable(tmp_path):
    root = tmp_path / "ws"
    _write(root / "a.txt", "disk text")

    def fake_read(path: pathlib.Path) -> str:
        if path.name == "a.txt":
            return "snapshot text"
        raise OSError("boom")

    _write(root / "b.txt", "x")
    manifest = plan_evidence.resolve_evidence(["a.txt", "b.txt"], active_root=root, allowed_roots=[root], read_text=fake_read)
    assert manifest["attached"][0]["text"] == "snapshot text"
    assert manifest["omissions"] == [{"locator": "b.txt", "reason": "unreadable"}]


def test_evidence_manifest_hash_ignores_text_but_sees_sha_and_omissions(tmp_path):
    root = tmp_path / "ws"
    _write(root / "a.txt", "one")
    base = plan_evidence.resolve_evidence(["a.txt", "https://u"], active_root=root, allowed_roots=[root])
    same_identity = json.loads(json.dumps(base))
    same_identity["attached"][0]["text"] = "REDACTED"
    assert plan_evidence.evidence_manifest_hash(base) == plan_evidence.evidence_manifest_hash(same_identity)
    _write(root / "a.txt", "two")
    changed = plan_evidence.resolve_evidence(["a.txt", "https://u"], active_root=root, allowed_roots=[root])
    assert plan_evidence.evidence_manifest_hash(base) != plan_evidence.evidence_manifest_hash(changed)
    fewer = plan_evidence.resolve_evidence(["a.txt"], active_root=root, allowed_roots=[root])
    assert plan_evidence.evidence_manifest_hash(changed) != plan_evidence.evidence_manifest_hash(fewer)


# ------------------------------------------------------------------ B4 findings

FINDING = {"id": "f1", "class": "blocking", "breaks": "claim_1", "summary": "claim 1 is not checkable",
           "recommendation": "name the checker"}


def test_parse_findings_protocol():
    assert plan_spec.parse_findings("[]\nNO_FINDINGS") == ([], None)
    assert plan_spec.parse_findings("```json\n[]\n```\nNO_FINDINGS") == ([], None)
    findings, err = plan_spec.parse_findings(json.dumps([FINDING]))
    assert err is None and findings == [FINDING]
    findings, err = plan_spec.parse_findings("```json\n" + json.dumps([FINDING]) + "\n```")
    assert err is None and findings[0]["id"] == "f1"
    _f, err = plan_spec.parse_findings("The plan looks fine to me, nothing to add.")
    assert err and "prose-only" in err
    _f, err = plan_spec.parse_findings("Looks fine. []")
    assert err and "NO_FINDINGS" in err
    _f, err = plan_spec.parse_findings("")
    assert err
    _f, err = plan_spec.parse_findings("[1, 2, 3]")
    assert err
    _f, err = plan_spec.parse_findings("I cannot review this.\n[]\nNO_FINDINGS")
    assert err  # sentinel with prose is a non-response
    findings, err = plan_spec.parse_findings("Here are my findings:\n" + json.dumps([FINDING]) + "\nHope this helps.")
    assert err is None and findings == [FINDING]  # inherited best-effort scan (non-empty array in prose)


def test_validate_findings_demotes_and_drops_structurally():
    ids = frozenset({"claim_1", "invariant_1"})
    raw = [
        FINDING,
        {"class": "blocking", "breaks": "claim_9", "summary": "invalid breaks"},
        {"id": "n1", "class": "need_evidence", "summary": "no locator"},
        {"id": "n2", "class": "need_evidence", "locator": "docs/a.md", "summary": "seen before"},
        {"id": "n3", "class": "need_evidence", "locator": "docs/b.md", "summary": "fresh"},
        {"id": "n4", "class": "need_evidence", "locator": "docs/b.md", "summary": "same wave repeat"},
        {"id": "x", "class": "weird", "summary": "unknown class"},
        {"id": "empty", "class": "note"},
        "not an object",
    ]
    normalized, disclosures, seen_after = plan_spec.validate_findings(
        raw, spec_ids=ids, seen_locators={"docs/a.md"}, slot="2",
    )
    by_id = {f["id"]: f for f in normalized}
    assert by_id["f1"]["class"] == "blocking" and by_id["f1"]["breaks"] == "claim_1"
    assert by_id["f2_2"]["class"] == "note" and by_id["f2_2"]["breaks"] == "claim_9"  # minted id, demoted
    assert by_id["n1"]["class"] == "note"
    # I-03: a repeat is DEMOTED, not dropped — a re-asked question must not close the wave.
    assert by_id["n2"]["class"] == "note" and by_id["n4"]["class"] == "note"
    assert by_id["n3"]["class"] == "need_evidence"
    assert by_id["x"]["class"] == "note"
    assert by_id["empty"]["class"] == "note" and by_id["empty"]["summary"] == "(missing summary)"
    assert disclosures == [
        "blocking_without_valid_breaks:f2_2", "need_evidence_without_locator:n1",
        "need_evidence_repeat:docs/a.md", "need_evidence_repeat:docs/b.md",
        "unknown_class:x:weird", "missing_summary:empty", "finding_9_not_object",
    ]
    assert seen_after == frozenset({"docs/a.md", "docs/b.md"})
    incoming = {"docs/a.md"}
    plan_spec.validate_findings(raw, spec_ids=ids, seen_locators=incoming, slot="2")
    assert incoming == {"docs/a.md"}  # input memory is never mutated


def test_claims_less_spec_can_be_blocked_on_goal():
    """D32: `breaks: "goal"` is always valid, so a thin spec cannot dodge REVISE_PLAN."""
    spec, errors = plan_spec.normalize_spec({"goal": "Rewrite the auth system", "in_scope": ["auth"]})
    assert errors == [] and plan_spec.spec_ids(spec) == frozenset({"goal"})
    raw = [{"id": "g1", "class": "blocking", "breaks": "goal", "summary": "no success condition at all"}]
    normalized, disclosures, _ = plan_spec.validate_findings(raw, spec_ids=plan_spec.spec_ids(spec), seen_locators=())
    assert disclosures == [] and normalized[0]["class"] == "blocking"
    verdict = plan_spec.aggregate([_slot("1", normalized), _slot("2", normalized), _slot("3", [])])
    assert verdict["aggregate"] == "REVISE_PLAN"
    assert plan_spec.closure_after_disposition(
        "REVISE_PLAN", verdict["findings"],
        [{"finding_id": f["finding_id"], "decision": "reject", "rationale": "r"} for f in verdict["findings"]], "blocking",
    )["closed"] is False


def test_validate_findings_never_drops_a_blocking_finding_without_summary():
    raw = [{"id": "b1", "class": "blocking", "breaks": "claim_1"}]
    normalized, disclosures, _ = plan_spec.validate_findings(raw, spec_ids={"claim_1"}, seen_locators=(), slot="1")
    assert normalized[0]["class"] == "blocking" and normalized[0]["summary"] == "(missing summary)"
    assert disclosures == ["missing_summary:b1"]
    assert plan_spec.aggregate([_slot("1", normalized)])["aggregate"] == "REVISE_PLAN"


def test_validate_findings_caps_per_slot_with_disclosure():
    raw = [{"id": f"f{i}", "class": "note", "summary": "s"} for i in range(plan_spec.MAX_FINDINGS_PER_SLOT + 5)]
    normalized, disclosures, _ = plan_spec.validate_findings(raw, spec_ids=(), seen_locators=())
    assert len(normalized) == plan_spec.MAX_FINDINGS_PER_SLOT
    assert disclosures[0].startswith(f"findings_capped:{plan_spec.MAX_FINDINGS_PER_SLOT}/{plan_spec.MAX_FINDINGS_PER_SLOT + 5}")


def _slot(slot: str, findings, ok=True, error=None):
    return {"slot": slot, "model": f"model-{slot}", "ok": ok, "findings": findings, "error": error}


BLOCK = {"id": "b", "class": "blocking", "breaks": "claim_1", "summary": "s", "recommendation": ""}
NOTE = {"id": "n", "class": "note", "breaks": "", "locator": "", "summary": "s", "recommendation": ""}
NEED = {"id": "e", "class": "need_evidence", "locator": "x.md", "summary": "s", "recommendation": ""}


def test_aggregate_quorum_math_three_slots():
    assert adaptive_quorum(3) == 2
    one_blocking = plan_spec.aggregate([_slot("1", [BLOCK]), _slot("2", [NOTE]), _slot("3", [])])
    assert one_blocking["aggregate"] == "REVIEW_REQUIRED"
    assert "blocking_below_quorum:1/2" in one_blocking["reasons"]
    two_blocking = plan_spec.aggregate([_slot("1", [BLOCK]), _slot("2", [BLOCK]), _slot("3", [])])
    assert two_blocking["aggregate"] == "REVISE_PLAN"
    assert two_blocking["counts"] == {
        "configured": 3, "parseable": 3, "quorum": 2, "blocking_slots": 2,
        "blocking": 2, "note": 0, "need_evidence": 0,
    }
    assert [f["finding_id"] for f in two_blocking["findings"]] == ["1:b", "2:b"]
    assert two_blocking["findings"][0]["model"] == "model-1"
    green = plan_spec.aggregate([_slot("1", []), _slot("2", []), _slot("3", [])])
    assert green["aggregate"] == "GREEN" and green["reasons"] == []
    # Two blocking findings in ONE slot count as one slot, not quorum.
    assert plan_spec.aggregate([_slot("1", [BLOCK, dict(BLOCK, id="b2")]), _slot("2", []), _slot("3", [])])["aggregate"] == "REVIEW_REQUIRED"


def test_aggregate_single_slot_follows_adaptive_quorum_loudly():
    result = plan_spec.aggregate([_slot("1", [BLOCK])])
    assert result["aggregate"] == "REVISE_PLAN"
    assert "single_reviewer_no_diversity" in result["reasons"]
    assert plan_spec.aggregate([_slot("1", [])])["aggregate"] == "GREEN"


def test_aggregate_need_evidence_only_wave_and_degraded():
    wave = plan_spec.aggregate([_slot("1", [NEED]), _slot("2", [dict(NEED, id="e2")]), _slot("3", [NEED])])
    assert wave["aggregate"] == "REVIEW_REQUIRED" and "need_evidence_only" in wave["reasons"]
    degraded = plan_spec.aggregate([_slot("1", [BLOCK]), _slot("2", [], ok=False, error="prose"), _slot("3", [], ok=False)])
    assert degraded["aggregate"] == "DEGRADED"
    assert "parseable_slots_below_quorum:1/2" in degraded["reasons"]
    assert any(r.startswith("slot_unparseable:2:prose") for r in degraded["reasons"])
    assert plan_spec.aggregate([])["aggregate"] == "DEGRADED"
    # Explicit quorum override is honoured; a non-positive quorum never turns a clean wave red (B-05).
    assert plan_spec.aggregate([_slot("1", [BLOCK]), _slot("2", [])], quorum=1)["aggregate"] == "REVISE_PLAN"
    assert plan_spec.aggregate([_slot("1", [])], quorum=0)["aggregate"] == "GREEN"
    assert plan_spec.aggregate([_slot("1", [], ok=False)], quorum=-3)["aggregate"] == "DEGRADED"
    # Duplicate slot labels never merge finding ids (B-14).
    dup = plan_spec.aggregate([_slot("rev", [BLOCK]), _slot("rev", [BLOCK]), _slot("rev", [])])
    assert [f["finding_id"] for f in dup["findings"]] == ["rev:b", "rev_2:b"]


def _control(outcome: str, closed: bool):
    return _parse_plan_review_control(PLAN_REVIEW_CONTROL_PREFIX + json.dumps({"outcome": outcome, "closed": closed}))


def test_closure_table_and_control_line_invariants():
    findings = [dict(NOTE, finding_id="1:n"), dict(NEED, finding_id="2:e")]
    full = [
        {"finding_id": "1:n", "decision": "accept", "rationale": "ok"},
        {"finding_id": "2:e", "decision": "reject", "rationale": "not needed"},
    ]
    green = plan_spec.closure_after_disposition("GREEN", [], [], "blocking")
    assert green["closed"] is True and green["open_ids"] == []
    partial = plan_spec.closure_after_disposition("REVIEW_REQUIRED", findings, full[:1], "blocking")
    assert partial["closed"] is False and partial["open_ids"] == ["2:e"]
    no_rationale = plan_spec.closure_after_disposition(
        "REVIEW_REQUIRED", findings, full[:1] + [{"finding_id": "2:e", "decision": "reject"}], "blocking",
    )
    assert no_rationale["closed"] is False and "invalid_disposition:2:e" in no_rationale["notes"]
    done = plan_spec.closure_after_disposition("REVIEW_REQUIRED", findings, full, "blocking")
    assert done["closed"] is True and done["open_ids"] == []
    deferred = plan_spec.closure_after_disposition(
        "REVIEW_REQUIRED", findings, [dict(full[0], decision="defer"), full[1]], "blocking",
    )
    assert deferred["closed"] is True
    revise = plan_spec.closure_after_disposition(
        "REVISE_PLAN", [dict(BLOCK, finding_id="1:b")] + findings,
        [{"finding_id": "1:b", "decision": "reject", "rationale": "disagree"}] + full, "blocking",
    )
    assert revise["closed"] is False and revise["open_ids"] == ["1:b"]
    assert any(n.startswith("revise_plan_not_closable_by_disposition") for n in revise["notes"])
    advisory = plan_spec.closure_after_disposition("REVISE_PLAN", [dict(BLOCK, finding_id="1:b")], [], "advisory")
    assert advisory["closed"] is False and any(n.startswith("advisory_enforcement") for n in advisory["notes"])
    degraded = plan_spec.closure_after_disposition("DEGRADED", [], [], "blocking")
    assert degraded["closed"] is False
    unknown = plan_spec.closure_after_disposition("REVIEW_REQUIRED", findings, full + [{"finding_id": "9:z", "decision": "accept", "rationale": "r"}], "blocking")
    assert unknown["closed"] is True and "unknown_finding_id:9:z" in unknown["notes"]
    vacuous = plan_spec.closure_after_disposition("REVIEW_REQUIRED", [], [], "blocking")
    assert vacuous["closed"] is True and any(n.startswith("no_findings_recorded") for n in vacuous["notes"])
    # The host control-line parser accepts exactly what the table produces for GREEN / REVISE_PLAN.
    assert _control("GREEN", green["closed"]) == ("GREEN", True)
    assert _control("REVISE_PLAN", revise["closed"]) == ("REVISE_PLAN", False)
    assert _control("REVIEW_REQUIRED", done["closed"]) == ("REVIEW_REQUIRED", True)
    assert _control("REVIEW_REQUIRED", partial["closed"]) == ("REVIEW_REQUIRED", False)
    assert _control("GREEN", False) is None and _control("REVISE_PLAN", True) is None


# ------------------------------------------------------------------- B5 packet


def test_system_prompt_stance_and_bible_gating():
    checklist = "## Plan Review Checklist\n\n- item one\n- item two"
    plain = plan_packet.build_plan_review_system_prompt(
        checklist_section=checklist, constitutional=False, bible_text="BIBLE BODY",
        cycle_index=1, enforcement="blocking",
    )
    lowered = plain.lower()
    assert "your own approach" not in lowered
    assert "before any code" not in lowered
    assert "before the work starts" in lowered
    assert "NO_FINDINGS" in plain
    # No POSITIVE quota instruction: every mention of "quota" is a negation.
    for line in plain.splitlines():
        if "quota" in line.lower():
            assert "no findings quota" in line.lower() or "do not fill a quota" in line.lower(), line
    assert "BIBLE BODY" not in plain and "on-demand pointer" in plain and "`BIBLE.md`" in plain
    # P1: a navigation map the host could not supply is a NAMED absence in the packet, never silence
    assert "OMISSION NOTE: BIBLE navigation map not supplied" in plain
    assert "OMISSION NOTE: ARCHITECTURE navigation map not supplied" in plain
    assert checklist in plain
    assert "Convergence rule" not in plain
    assert "blocking" in lowered and "`breaks`" in plain
    assert "need_evidence" in plain
    for phrase in ("Success conditions", "Load-bearing decisions", "Constraints and invariants",
                   "Deferrals", "Evidence sufficiency"):
        assert phrase in plain
    constitutional = plan_packet.build_plan_review_system_prompt(
        checklist_section=checklist, constitutional=True, bible_text="BIBLE BODY",
        cycle_index=2, enforcement="advisory", architecture_text="ARCH BODY",
    )
    assert "BIBLE BODY" in constitutional and "Governance" in constitutional
    # W3: ARCHITECTURE.md rides inline, in full, in the self-modification pack
    assert "## ARCHITECTURE.md" in constitutional and "ARCH BODY" in constitutional
    # the convergence rule is cycle-dependent and now lives in the USER prior-cycles section
    # (system prompt stays byte-stable for its cache block)
    assert "Convergence rule" not in constitutional
    assert "advisory" in constitutional
    # S-B09 / D26: a constitutional review without BIBLE.md is an assembly failure, never a prompt.
    with pytest.raises(plan_packet.PlanPacketError):
        plan_packet.build_plan_review_system_prompt(
            checklist_section="c", constitutional=True, bible_text=None, cycle_index=1, enforcement="blocking",
            architecture_text="ARCH BODY",
        )
    # W3: and without ARCHITECTURE.md as well — a typed failure, never a packet that silently lacks it
    with pytest.raises(plan_packet.PlanPacketError, match="ARCHITECTURE"):
        plan_packet.build_plan_review_system_prompt(
            checklist_section="c", constitutional=True, bible_text="BIBLE BODY", cycle_index=1,
            enforcement="blocking", architecture_text=None,
        )
    assert "PLAN_REVIEW_CONTROL_JSON" not in plain + constitutional
    # B-09 / D30: resolvable pointer locator and the BIBLE navigation map only when NOT constitutional.
    pointed = plan_packet.build_plan_review_system_prompt(
        checklist_section="c", constitutional=False, bible_text="BIBLE BODY", cycle_index=1, enforcement="blocking",
        bible_locator="/srv/ouro/repo/BIBLE.md", bible_nav_map="NAV-MAP-ROWS",
        architecture_text="ARCH BODY", architecture_locator="/srv/ouro/repo/docs/ARCHITECTURE.md",
        architecture_nav_map="ARCH-NAV-ROWS",
    )
    assert "`/srv/ouro/repo/BIBLE.md`" in pointed and "NAV-MAP-ROWS" in pointed
    assert "BIBLE navigation map (pointer, not a copy)" in pointed and "BIBLE BODY" not in pointed
    # W3: every other plan carries the ARCHITECTURE navigation map + a resolvable pointer, never the body
    assert "`/srv/ouro/repo/docs/ARCHITECTURE.md`" in pointed and "ARCH-NAV-ROWS" in pointed
    assert "ARCHITECTURE navigation map (pointer, not a copy)" in pointed and "ARCH BODY" not in pointed
    assert "OMISSION NOTE" not in pointed
    assert "the host attaches it on the next cycle" in pointed
    with_bible = plan_packet.build_plan_review_system_prompt(
        checklist_section="c", constitutional=True, bible_text="BIBLE BODY", cycle_index=1, enforcement="blocking",
        bible_nav_map="NAV-MAP-ROWS", architecture_text="ARCH BODY", architecture_nav_map="ARCH-NAV-ROWS",
    )
    assert "NAV-MAP-ROWS" not in with_bible and "BIBLE BODY" in with_bible and "ARCH BODY" in with_bible


def _packet(spec_raw, manifest, prior_cycles=(), dispositions=(), delta=None, log=None):
    spec, errors = plan_spec.normalize_spec(spec_raw)
    assert errors == []
    return plan_packet.build_plan_review_user_content(
        objective="OBJECTIVE-TEXT", goal=spec["goal"], plan_prose="PROSE-TEXT", spec=spec,
        manifest=manifest, prior_cycles=list(prior_cycles), dispositions=list(dispositions),
        spec_delta=delta, root_exploration_log=log,
    )


def test_user_content_sections_in_order_with_omissions_and_prior_cycles(tmp_path):
    root = tmp_path / "ws"
    _write(root / "ouroboros" / "uploader.py", "def upload(): ...")
    manifest = plan_evidence.resolve_evidence(CODE_SPEC["evidence"], active_root=root, allowed_roots=[root])
    first = _packet(CODE_SPEC, manifest, log="ran: ls -la")
    order = ["## TASK OBJECTIVE", "## SPEC", "## PLAN PROSE", "## EVIDENCE", "### OMISSIONS",
             "## ROOT EXPLORATION LOG", "## PRIOR CYCLES"]
    positions = [first.index(h) for h in order]
    assert positions == sorted(positions)
    assert "OBJECTIVE-TEXT" in first and "PROSE-TEXT" in first and "ran: ls -la" in first
    assert "def upload(): ..." in first and manifest["attached"][0]["sha256"] in first
    assert "| https://example.com/spec | url_not_fetched |" in first
    piped = plan_packet.build_plan_review_user_content(
        objective="o", goal="g", plan_prose="", spec={"goal": "g"},
        manifest={"declared": ["a|b"], "attached": [], "omissions": [{"locator": "a|b", "reason": "missing"}]},
        prior_cycles=[], dispositions=[], spec_delta=None, root_exploration_log=None,
    )
    assert "| a\\|b | missing |" in piped
    assert "| task:abc123 | unsupported_locator |" in first
    assert '"id": "invariant_1"' in first and '"claim_1"' in first and '"id": "goal"' in first
    assert "First cycle" in first
    prior = [{"cycle_index": 1, "aggregate": "REVIEW_REQUIRED", "findings": [dict(NOTE, finding_id="1:n", summary="PRIOR-FINDING")]}]
    dispositions = [{"finding_id": "1:n", "decision": "reject", "rationale": "DISPO-RATIONALE"}]
    second = _packet(CODE_SPEC, manifest, prior_cycles=prior, dispositions=dispositions,
                     delta={"changed": True, "goal_changed": False, "ids": {"added": ["deferred_3"]}})
    assert "PRIOR-FINDING" in second and "DISPO-RATIONALE" in second and "deferred_3" in second
    assert "First cycle" not in second
    assert "PLAN_REVIEW_CONTROL_JSON" not in second


def test_prior_blocking_findings_survive_the_section_bound(monkeypatch):
    monkeypatch.setattr(plan_packet, "PACKET_PRIOR_CYCLES_CHARS", 20_000)  # 96 compact rows do not fit
    long_summary, long_rec = "S" * 1500, "R" * 1000
    prior = [{"cycle_index": 1, "aggregate": "REVISE_PLAN", "findings": [
        {"finding_id": f"{slot}:f{i}", "class": "blocking" if i == 31 else "note", "breaks": "claim_1" if i == 31 else "",
         "locator": "", "summary": long_summary, "recommendation": long_rec}
        for slot in ("1", "2", "3") for i in range(32)
    ]}]
    spec, _ = plan_spec.normalize_spec(DECK_SPEC)
    packet = plan_packet.build_plan_review_user_content(
        objective="o", goal=spec["goal"], plan_prose="p", spec=spec,
        manifest=plan_evidence.resolve_evidence([], active_root=".", allowed_roots=["."]),
        prior_cycles=prior, dispositions=[], spec_delta=None, root_exploration_log=None, cycle_index=2,
    )
    prior_section = packet[packet.index("## PRIOR CYCLES"):]
    for fid in ('"1:f31"', '"2:f31"', '"3:f31"'):
        assert fid in prior_section  # blocking first: never behind the bound
    assert "OMISSION NOTE (structural)" in prior_section and "kept " in prior_section and "full-set sha256=" in prior_section
    assert '"finding_id"' in prior_section and long_rec not in prior_section  # compact projection, no recommendation
    assert prior_section.count("```json") == prior_section.count("```\n") or prior_section.count("```") % 2 == 0
    empty_cycle2 = plan_packet.build_plan_review_user_content(
        objective="o", goal="g", plan_prose="p", spec=spec, manifest={"declared": [], "attached": [], "omissions": []},
        prior_cycles=[], dispositions=[], spec_delta=None, root_exploration_log=None, cycle_index=2,
    )
    assert "no prior findings recorded" in empty_cycle2 and "First cycle" not in empty_cycle2


def test_spec_section_is_bounded_structurally_never_clipped():
    worst = {"goal": "g", "decisions": [
        {"choice": "c" * 600, "rejected": ["r" * 600] * plan_spec.MAX_REJECTED_PER_DECISION, "why": "w" * 600}
        for _ in range(plan_spec.MAX_LIST_ITEMS)
    ], "in_scope": ["i" * 600] * plan_spec.MAX_LIST_ITEMS, "invariants": ["v" * 600] * plan_spec.MAX_LIST_ITEMS}
    spec, errors = plan_spec.normalize_spec(worst)
    assert errors == []
    text, notes = plan_spec.bounded_json(plan_spec.spec_with_ids(spec), plan_spec.PACKET_SPEC_CHARS)
    assert len(text) <= plan_spec.PACKET_SPEC_CHARS
    json.loads(text)  # whole items only — always valid JSON
    assert notes and all("kept " in n and "full-set sha256=" in n for n in notes)
    packet = plan_packet.build_plan_review_user_content(
        objective="o", goal="g", plan_prose="p", spec=spec, manifest={"declared": [], "attached": [], "omissions": []},
        prior_cycles=[], dispositions=[], spec_delta=None, root_exploration_log=None,
    )
    spec_section = packet[packet.index("## SPEC"):packet.index("## PLAN PROSE")]
    assert len(spec_section) < plan_spec.PACKET_SPEC_CHARS + 2000 and "OMISSION NOTE (structural)" in spec_section
    # Oversized scalars with no list left → typed omission object, never clipped JSON.
    text, notes = plan_spec.bounded_json({"blob": "x" * 500}, 100)
    assert json.loads(text)["omitted"] is True and "full_payload_sha256" in text and notes


def test_user_content_bounds_are_disclosed_not_silent():
    spec, _ = plan_spec.normalize_spec(DECK_SPEC)
    manifest = plan_evidence.resolve_evidence([], active_root=".", allowed_roots=["."])
    content = plan_packet.build_plan_review_user_content(
        objective="o", goal=spec["goal"], plan_prose="P" * (plan_spec.PACKET_PROSE_CHARS + 500), spec=spec,
        manifest=manifest, prior_cycles=[], dispositions=[], spec_delta=None, root_exploration_log=None,
    )
    assert "OMISSION NOTE" in content and "(no evidence declared)" in content
    assert "(not provided by host)" in content


def test_capped_reviewer_requests_are_rendered_even_when_the_agent_declared_nothing():
    """W3 (delta review round 2, S-3 edge): an empty declared list must not swallow the named
    `reviewer_request_cap` omissions — every absence stays visible to the reviewers (P1)."""
    spec, _ = plan_spec.normalize_spec(DECK_SPEC)
    manifest = plan_evidence.resolve_evidence([], active_root=".", allowed_roots=["."])
    manifest["reviewer_requested"] = ["a.md"]
    manifest["omissions"] = [{"locator": "zzz.md", "reason": "reviewer_request_cap"}]
    content = _packet(DECK_SPEC, manifest)
    assert "(no evidence declared by the agent)" in content
    assert "| zzz.md | reviewer_request_cap |" in content and "OMISSIONS (every absence named)" in content


# --------------------------------------------------------- domain independence


def test_domain_independence_deck_spec_without_paths(tmp_path):
    spec, errors = plan_spec.normalize_spec(DECK_SPEC)
    assert errors == [] and spec["affected_resources"] == [] and spec["evidence"] == []
    assert plan_spec.spec_ids(spec) == frozenset({"goal", "claim_1", "claim_2", "invariant_1", "invariant_2", "decision_1", "deferred_1"})
    assert len(plan_spec.spec_hash(spec)) == 64
    ok, _ = plan_spec.resolve_constitutional(
        active_root=tmp_path / "ws", system_repo_root=tmp_path / "repo", affected_resources=spec["affected_resources"], evidence=spec["evidence"],
    )
    assert ok is False
    manifest = plan_evidence.resolve_evidence(spec["evidence"], active_root=tmp_path, allowed_roots=[tmp_path])
    assert manifest == {"declared": [], "attached": [], "omissions": [], "bounds": manifest["bounds"]}
    assert len(plan_evidence.evidence_manifest_hash(manifest)) == 64
    raw = [{"id": "d1", "class": "blocking", "breaks": "claim_2", "summary": "no source named for the Q3 numbers", "recommendation": "name the report"}]
    normalized, disclosures, _ = plan_spec.validate_findings(raw, spec_ids=plan_spec.spec_ids(spec), seen_locators=(), slot="1")
    assert disclosures == [] and normalized[0]["class"] == "blocking"
    verdict = plan_spec.aggregate([_slot("1", normalized), _slot("2", normalized), _slot("3", [])])
    assert verdict["aggregate"] == "REVISE_PLAN"
    packet = _packet(DECK_SPEC, manifest)
    code_packet = _packet(CODE_SPEC, plan_evidence.resolve_evidence([], active_root=tmp_path, allowed_roots=[tmp_path]))
    # Same section skeleton for a deck and for code — the host never interprets the domain.
    skeleton = lambda text: [line for line in text.splitlines() if line.startswith("## ")]  # noqa: E731
    assert skeleton(packet) == skeleton(code_packet)
    assert "5 slides" in packet and "(no evidence declared)" in packet


def test_module_ratchet_targets():
    root = pathlib.Path(plan_spec.__file__).parent
    for name, limit in (("plan_spec.py", 1000), ("plan_evidence.py", 1000), ("plan_packet.py", 1000)):
        assert len((root / name).read_text(encoding="utf-8").splitlines()) <= limit


@pytest.mark.parametrize("text", ["```\n[]\n```", "[ ]\nNO_FINDINGS\n", "[]"])
def test_parse_findings_clean_variants(text):
    assert plan_spec.parse_findings(text) == ([], None)


def test_blank_spec_items_are_disclosed_not_silently_dropped():
    """I-20: P1 — a blank list item is a cut, and every cut in this module is named."""
    spec, errors = plan_spec.normalize_spec(
        {"goal": "Ship", "in_scope": ["auth", "   ", ""], "acceptance_claims": ["it ships"]}
    )
    assert errors == []
    assert spec["in_scope"] == ["auth"]
    assert any("blank item(s) dropped" in note for note in spec["normalization_omissions"])  # one bounded note per list
