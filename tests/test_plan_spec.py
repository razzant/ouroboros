"""Contract tests for ``ouroboros.tools.plan_spec`` and ``plan_evidence``: the ONE
structural path fact (``resolve_constitutional``, the body predicate's installed-body
half), the bounded evidence manifest every plan ask attaches through, and the spec hash
(``spec_hash``) the read-only views apply to earlier-schema plan records — and domain
independence (a plan with zero file paths is first-class)."""

from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros.tools import plan_evidence, plan_spec


# ------------------------------------------------------- earlier-schema spec reads


# Specs as finished v1/v2 plan records hold them (normalized by the retired spec parser:
# minted ids, positional invariants); the read-only views verify them by spec_hash.
STORED_CODE_SPEC = {
    "goal": "Add a retry wrapper around the flaky uploader", "in_scope": ["retry wrapper", "unit tests"],
    "non_goals": ["changing the upload protocol"],
    "acceptance_claims": [{"id": "claim_1", "claim": "uploads retry 3 times", "surface": "tests", "support": "pytest -k retry", "priority": "must"},
                          {"id": "claim_2", "claim": "a plain-string claim", "surface": "", "support": "", "priority": "must"}],
    "invariants": ["no new dependencies", "public API unchanged"],
    "decisions": [{"id": "decision_1", "choice": "exponential backoff", "rejected": ["fixed delay"], "why": "bursty upstream"}],
    "deferred": [{"id": "deferred_1", "what": "metrics emission", "why_safe_to_defer": "observability only"},
                 {"id": "deferred_2", "what": "log wording", "why_safe_to_defer": ""}],
    "affected_paths": ["ouroboros/uploader.py"], "affected_resources": ["the upload pipeline", "the storage provider"],
    "evidence": ["ouroboros/uploader.py", "https://example.com/spec", "task:abc123"], "normalization_omissions": [],
}


def _stored(goal="g"):
    return {"goal": goal, "in_scope": [], "non_goals": [], "acceptance_claims": [], "invariants": [],
            "decisions": [], "deferred": [], "affected_resources": [], "evidence": [], "normalization_omissions": []}


def test_spec_hash_stable_under_key_order_and_changed_by_content():
    spec = STORED_CODE_SPEC
    reordered = {k: spec[k] for k in reversed(list(spec))}
    assert plan_spec.spec_hash(spec) == plan_spec.spec_hash(reordered)
    assert plan_spec.spec_hash(spec) == plan_spec.spec_hash(json.loads(json.dumps(spec, indent=4)))
    changed = json.loads(json.dumps(spec))
    changed["goal"] = "Add a retry wrapper (v2)"
    changed["invariants"].append("no network in tests")
    changed["decisions"][0]["why"] = "different rationale"
    changed["deferred"] = [{"id": "deferred_1", "what": "log wording", "why_safe_to_defer": ""}]
    assert plan_spec.spec_hash(changed) != plan_spec.spec_hash(spec)


# The three plan-review waves recorded on the live install while `affected_resources` was still
# resolved as a path list (structural-health sprint evidence, plan-resource-live-examples.json:
# tasks 12102247e791ff1e, 0be6fb0690f44bbd, 6596589bb3674aaa), with the spec digests recorded
# BEFORE `affected_paths` existed: the read-only views verify a stored wave's spec by this hash,
# so a stored spec must keep its identity or its record reads as source-unavailable.
STORED_LEGACY_WAVES = (
    ("12102247e791ff1e",
     ["Отдельный проект «TSMC — 10-летний инвестиционный анализ»",
      "Файлы исследования, моделей и PDF внутри его project workspace",
      "Финальные пользовательские deliverables"],
     [],
     "8832d9db81cbd61dea584c193130e25b3c63fea25b007b7978456f07f92e1f82"),
    ("0be6fb0690f44bbd",
     ["/Users/anton/Ouroboros/projects/TSMC___10-____________________________"],
     [],
     "a8aea6bb3adbb7dcb635dbebb7161d238d34168502420b7be4fd4f4c1692f06f"),
    ("6596589bb3674aaa",
     ["/Users/anton/Ouroboros/data/skills/external/context-lens",
      "/Users/anton/Ouroboros/projects/Context_Lens___________________________________"],
     ["data/skills/external/context-lens/SKILL.md"],
     "2e2764e09bc74383c082bae1eabc44c9acb24c5c8cd4048c4e77187375bbc240"),
)


def test_stored_specs_from_before_affected_paths_keep_their_recorded_identity():
    """A spec stored under the old shape carries no `affected_paths` key, and it hashes to
    the digest it was recorded with."""
    for task_id, resources, evidence, spec_digest in STORED_LEGACY_WAVES:
        spec = {**_stored(goal=f"stored plan wave {task_id}"), "affected_resources": list(resources),
                "evidence": list(evidence)}
        assert "affected_paths" not in spec
        assert plan_spec.spec_hash(spec) == spec_digest


# ------------------------------------------------------------ B2 constitutional


def test_constitutional_active_binding_alone_does_not_decide(tmp_path):
    """D29: active == system with no declared system path is NOT constitutional."""
    system = tmp_path / "repo"
    system.mkdir()
    ok, note = plan_spec.resolve_constitutional(
        active_root=system, system_repo_root=system, affected_paths=[], evidence=[],
    )
    assert ok is False and note.startswith("not constitutional")
    ok, _ = plan_spec.resolve_constitutional(
        active_root=system, system_repo_root=system, affected_paths=["ouroboros/loop.py"], evidence=[],
    )
    assert ok is True  # a relative path under the (system) active root still decides


def test_constitutional_declared_paths_relative_absolute_and_file_scheme(tmp_path):
    system = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    system.mkdir()
    workspace.mkdir()
    ok, note = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_paths=["../repo/ouroboros/loop.py"], evidence=[],
    )
    assert ok and "affected_paths" in note
    # Owner 16=A: an EVIDENCE locator is something to LOOK AT, and reading a repo file is not
    # changing it — existing or not, it never buys the constitutional pack on its own. Existence
    # therefore stopped deciding anything, and the check went with it.
    (system / "BIBLE.md").write_text("# constitution\n", encoding="utf-8")
    ok, note = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_paths=["src/app.py"], evidence=[str(system / "BIBLE.md")],
    )
    assert ok is False and "EVIDENCE reads" in note and str(system / "BIBLE.md") in note
    ok, _ = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_paths=["src/app.py"], evidence=[str(system / "NO_SUCH_FILE.md")],
    )
    assert ok is False
    # ...while a TARGET counts whether or not it exists yet — creating a module is self-modification.
    ok, _ = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_paths=[str(system / "ouroboros" / "brand_new_module.py")], evidence=[],
    )
    assert ok is True
    ok, _ = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_paths=["src/app.py"], evidence=["docs/x.md"],
    )
    assert ok is False
    # B-06: a system-repo path dressed as file:// is still a path.
    ok, note = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_paths=[f"file://{system}/ouroboros/loop.py"], evidence=[],
    )
    assert ok is True and "file://" in note


def test_constitutional_selector_classifies_the_source_path(tmp_path):
    """A `::lines=A-B` suffix names a RANGE of one file: the path in front of it decides, on
    either list — as a change target it escalates, as an evidence read it is only named."""
    system = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = _write(system / "docs" / "ARCHITECTURE.md", "# Architecture\n")

    ok, note = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_paths=[f"{source}::lines=1-1"], evidence=[],
    )
    assert ok is True
    assert f"{source}::lines=1-1" in note

    ok, note = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_paths=[], evidence=[f"{source}::lines=1-1"],
    )
    assert ok is False
    assert f"{source}::lines=1-1" in note and "EVIDENCE reads" in note


def test_constitutional_payload_exempt_url_task_never_and_empty_false(tmp_path):
    system = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    payload = tmp_path / "data" / "skills" / "native" / "demo"  # data plane, outside the system repo
    payload.mkdir(parents=True)
    workspace.mkdir()
    (system / "x").mkdir(parents=True)
    ok, _ = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_paths=[str(payload / "SKILL.md")], evidence=[], payload_roots=[payload],
    )
    assert ok is False
    # A payload root that resolves INSIDE the system repo still exempts: the roots come
    # from the frozen skill-payload predicate, and a skill task whose active workspace IS
    # the repo touches only data-plane payload files (task 7881ad77902d4d25). Reviewer
    # proposal S-B05 (drop such roots) was rejected: it re-broke that frozen behaviour.
    nested = system / "data" / "skills" / "native" / "demo"
    ok, _ = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_paths=[str(nested / "SKILL.md")], evidence=[], payload_roots=[nested],
    )
    assert ok is False
    # A system path OUTSIDE every payload root is still constitutional.
    ok, _ = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system,
        affected_paths=[str(system / "ouroboros" / "loop.py")], evidence=[], payload_roots=[nested],
    )
    assert ok is True
    ok, _ = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system, affected_paths=[],
        evidence=["https://github.com/razzant/ouroboros/blob/main/ouroboros/loop.py", "task:xyz"],
    )
    assert ok is False
    ok, note = plan_spec.resolve_constitutional(
        active_root=workspace, system_repo_root=system, affected_paths=[], evidence=[],
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
            active_root=root, system_repo_root=tmp_path / "repo", affected_paths=[locator], evidence=[],
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


# ------------------------------------------------------------------ B4 findings


def test_live_settings_are_never_attachable_as_evidence(tmp_path):
    """C-06: the runtime data plane is denied whatever root the caller declares."""
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    settings = data_root / "settings.json"
    settings.write_text('{"OPENROUTER_API_KEY": "sk-live-secret"}', encoding="utf-8")
    manifest = plan_evidence.resolve_evidence(
        ["data/settings.json"], active_root=tmp_path, allowed_roots=[tmp_path],
        deny_paths=[str(data_root)],
    )
    assert manifest["attached"] == []
    assert manifest["omissions"] == [{"locator": "data/settings.json", "reason": "denied_path"}]
    assert "sk-live-secret" not in json.dumps(manifest)


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_evidence_content_reaches_reviewers_as_the_source_wrote_it(tmp_path, newline):
    """Admission is decided by path, root and deny rules; content is never rewritten by pattern
    (owner 2026-10-29: the legacy secret-shaped masking broke variable-bearing code), and the
    file's own line endings survive on every OS."""
    notes = tmp_path / "notes.md"
    probe = tmp_path / "probe.ts"
    # Bytes, not text mode: text mode would translate newlines on Windows; the fixture fixes them.
    notes.write_bytes(
        f"benchmark notes{newline}OPENROUTER_API_KEY=sk-or-FAKEFIXTURE-not-a-real-key-ABCDEFGHIJKLMNOP{newline}"
        .encode("utf-8"))
    probe.write_bytes(
        (f'const API_KEY = process.env.API_KEY ?? "";{newline}'
         f'export const config = {{ token: "abcd1234", password: pw }};{newline}').encode("utf-8"))
    manifest = plan_evidence.resolve_evidence(
        ["notes.md", "probe.ts", "probe.ts::lines=1-1"], active_root=tmp_path, allowed_roots=[tmp_path])
    import hashlib
    rows = {row["locator"]: row for row in manifest["attached"]}
    for name, path in (("notes.md", notes), ("probe.ts", probe)):
        assert rows[name]["text"] == path.read_bytes().decode("utf-8")
        assert rows[name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashlib.sha256(rows[name]["text"].encode()).hexdigest() == rows[name]["sha256"]
        assert "secrets_redacted" not in rows[name]
    assert rows["probe.ts::lines=1-1"]["text"] == f'const API_KEY = process.env.API_KEY ?? "";{newline}'
    assert "REDACTED" not in json.dumps(manifest)


@pytest.mark.parametrize("notice_prefix", ["", "source detail " * 4000], ids=["complete", "budgeted"])
def test_task_locator_carries_the_host_notice_and_a_changed_notice_changes_the_evidence(tmp_path, notice_prefix):
    """A ``task:`` attachment reads the task's terminal facts with its host notice; a changed
    notice is changed evidence (a new hash), the original answer is never rewritten, and a
    notice beyond the per-item view is a named cut."""
    from ouroboros.task_results import load_task_result, write_task_result

    answer = "The report is complete."
    write_task_result(tmp_path, "previous", "completed", result=answer)
    reader = plan_evidence.task_evidence_reader(tmp_path)
    assert "terminal_host_notice" not in json.loads(reader("previous"))
    first_notice = notice_prefix + "First source limitation."
    write_task_result(tmp_path, "previous", "completed", terminal_host_notice=first_notice)

    def attach():
        return plan_evidence.resolve_evidence(["task:previous"], active_root=tmp_path, allowed_roots=[tmp_path],
                                              resolve_task=reader)

    attached = attach()["attached"][0]
    assert '"terminal_host_notice":' in attached["text"]
    assert json.loads(reader("previous"))["result"] == answer
    write_task_result(tmp_path, "previous", "completed",
                      terminal_host_notice=notice_prefix + "Second source limitation.")
    manifest = attach()
    changed = manifest["attached"][0]
    assert changed["sha256"] != attached["sha256"]
    assert load_task_result(tmp_path, "previous")["result"] == answer
    if notice_prefix:
        assert changed["text"] == attached["text"]  # the changed tail is beyond the view
        assert any(o["reason"].startswith("truncated_to_") for o in manifest["omissions"])
    else:
        assert json.loads(attached["text"])["terminal_host_notice"] == first_notice
        assert json.loads(changed["text"])["terminal_host_notice"] == "Second source limitation."
