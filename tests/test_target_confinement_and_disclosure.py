"""CLASS 5a — the invariants that survived a mutation, now defended behaviourally.

Every gate this audit added was checked by MUTATION: break the invariant in a copy of the
tree, run the tests that claim to protect it, and see whether anything goes red.  Six of
the eight key invariants were defended (the root matrix, the prepared-token binding, the
task→session binding, the returned-manifest export policy, the shell-side subagent secret
guard, and panic reaching every live broker).  Two were not, and this file closes both.

FALSE GUARANTEE 1 — the target's symlink confinement had no test at all.
``workspace_native._target`` is the execd kernel's ONE confinement: it resolves a
workspace-relative path and refuses when the result escapes the workspace root through a
symlink.  Disabling that check entirely (``if False and not path_is_relative_to(...)``)
left the whole suite green — including all 58 tests whose names mention symlink, escape,
containment or confine.  On a remote task this is the only thing standing between a
workspace-relative argument and the rest of the target's filesystem, so "no test" is not
a coverage statistic, it is an unguarded boundary.

FALSE GUARANTEE 2 — the elision gate could not fail for the right reason.
``tests/test_disclosure_elision_gate.py`` pins the disclosure by reading the SOURCE for
its marker tokens, which its own BOUNDARY paragraph admits.  The mutation
``if False and undisclosed_artifacts:`` leaves every token in place, so the gate stayed
green while the disclosure stopped happening.  A source gate cannot be the only defence of
a behaviour; the behavioural half lives here.
"""

from __future__ import annotations

import pathlib


def _HOME_ARTIFACT_LIMIT() -> int:
    from ouroboros.remote_transfer import _HOME_ARTIFACT_LIMIT as limit

    return int(limit)


def _artifacts(count: int) -> list[dict[str, str]]:
    return [{"name": f"artifact-{index}.txt", "kind": "file"} for index in range(count)]


def _imported(artifacts: list[dict[str, str]]) -> dict:
    """Run the real Home import of a remote result carrying `artifacts`.

    The production entry point, with no blob refs to fetch, so the assertion is about
    what a model would actually be handed rather than about a rebuilt dict.
    """

    import tempfile

    from ouroboros.remote_transfer import RemoteTransferService, _import_remote_result

    drive_root = pathlib.Path(tempfile.mkdtemp())
    return _import_remote_result(
        RemoteTransferService(),
        drive_root,
        "task-disclosure",
        operation_id="op-disclosure",
        connection_id="conn-1",
        workspace_id="ws-1",
        channel="workspace_query",
        envelope={
            "text": "done",
            "artifacts": artifacts,
            "diagnostic": None,
            "process": None,
            "trace": {},
        },
        fetched={},
    )


def _workspace(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("print('in')\n", encoding="utf-8")
    return root


def test_a_symlink_out_of_the_workspace_is_refused_by_the_target(tmp_path):
    """The execd kernel refuses a path that resolves outside the workspace root.

    Written against the PUBLIC native entry point, not the private helper, so it fails if
    the confinement is removed anywhere on the path an authorized operation really takes.
    The refusal arrives as a typed diagnostic rather than an exception — the kernel converts
    it — and the assertion covers what a remote caller would actually receive, including
    that the file's bytes are NOT in the answer.
    """

    from ouroboros.workspace_native import execute_native_operation

    root = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "credentials.json"
    secret.write_text('{"token": "PLAINTEXT-SENTINEL"}', encoding="utf-8")
    (root / "escape.json").symlink_to(secret)

    envelope = execute_native_operation(root, "read_file", {"path": "escape.json"}).envelope
    assert "escapes workspace" in envelope.text, envelope.text
    assert "PLAINTEXT-SENTINEL" not in envelope.text, "the file's bytes must not travel"
    assert envelope.diagnostic is not None
    assert envelope.diagnostic.code == "permission_denied"
    assert envelope.diagnostic.completion == "not_started", (
        "a refused read must report that nothing ran, not an ambiguous completion"
    )


def test_a_symlinked_directory_out_of_the_workspace_is_refused_too(tmp_path):
    """The same for a directory hop, which is how a real escape is usually built.

    Two refusals overlap on this path and the test keeps them apart on purpose. The file
    is deliberately named `notes.txt`: a credential-shaped name (`key.pem`) is now
    refused one layer EARLIER by the read-side export policy, which would make this test
    pass without the confinement ever being consulted. The second half then checks the
    credential-named case as its own fact, so both rules are proved rather than one
    masking the other.
    """

    from ouroboros.workspace_native import execute_native_operation

    root = _workspace(tmp_path)
    outside = tmp_path / "outside"
    (outside / "deep").mkdir(parents=True)
    (outside / "deep" / "notes.txt").write_text("OUTSIDE-SENTINEL", encoding="utf-8")
    (outside / "deep" / "key.pem").write_text("PRIVATE-KEY-SENTINEL", encoding="utf-8")
    (root / "linked").symlink_to(outside)

    envelope = execute_native_operation(
        root, "read_file", {"path": "linked/deep/notes.txt"}
    ).envelope
    assert "escapes workspace" in envelope.text, envelope.text
    assert "OUTSIDE-SENTINEL" not in envelope.text

    # The same hop with a credential-shaped name: whichever rule answers first, the bytes
    # must not travel and the read must not have happened.
    guarded = execute_native_operation(
        root, "read_file", {"path": "linked/deep/key.pem"}
    ).envelope
    assert "PRIVATE-KEY-SENTINEL" not in guarded.text
    assert guarded.diagnostic is not None and guarded.diagnostic.completion == "not_started"


def test_a_path_inside_the_workspace_still_reads(tmp_path):
    """The confinement must not be satisfiable by refusing everything."""

    from ouroboros.workspace_native import execute_native_operation

    root = _workspace(tmp_path)
    envelope = execute_native_operation(root, "read_file", {"path": "src/app.py"}).envelope
    assert "print('in')" in envelope.text
    assert envelope.diagnostic is None


def test_an_over_bound_artifact_list_discloses_its_remainder_in_the_text():
    """BEHAVIOUR, not spelling: the bound is exercised and the disclosure is read back.

    This is the half the source gate cannot provide — it is what goes red when the
    disclosure branch is disabled while its tokens stay in the file.
    """

    result = _imported(_artifacts(_HOME_ARTIFACT_LIMIT() + 7))
    assert len(result["artifacts"]) <= _HOME_ARTIFACT_LIMIT()
    assert "OMISSION NOTE" in result["text"], (
        "artifacts were dropped and the model was not told — the silent-elision class"
    )
    assert "7" in result["text"], "the disclosure must name HOW MANY are missing"
    assert result["trace"].get("artifacts_undisclosed_count") == 7


def test_a_short_artifact_list_says_nothing_about_bounds():
    """A disclosure that always fires is noise and teaches readers to ignore it."""

    result = _imported(_artifacts(1))
    assert "OMISSION NOTE" not in result["text"]
    assert "artifacts_undisclosed_count" not in result["trace"]






# ── CLASS 1 (returning): one policy × N doors, on the mutation side ──────────
#
# `_target` had a test and `_mutation_target` did not, which is the whole shape of the
# failure again. It confined the PARENT and left the final component unresolved, and
# that was correct for the ONE caller it was written against — `_atomic_write`, where
# `os.replace` substitutes the link rather than following it. The next write mode
# reasoned differently: `write_file` with `mode="append"` opened the same path with
# `"a"`, which follows the link, so a workspace-relative argument wrote outside the
# workspace on the target. Reproduced live before the fix: a symlink from the workspace
# to a file outside it, appended to, and the outside file grew.
#
# The fix is ONE door (`workspace_native_paths.native_mutation_target`), so the tests
# below are per-MODE behavioural cases plus a mechanical sweep that fails when a NEW
# mutation site is added outside it. The mechanical half is the part that closes the
# class: a per-mode test only ever covers the modes someone thought of.

_ESCAPE_MARKER = "escapes workspace"


def _escape_fixture(tmp_path: pathlib.Path, link_name: str = "escape.txt"):
    """A workspace, an outside victim file, and a symlink from one to the other.

    The link is what a model can build for itself with a single `run_command`
    (`ln -s`), which is why the final component is not a spelling the target may trust.
    """
    root = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)
    victim = outside / "authorized_keys"
    victim.write_text("ORIGINAL-ONLY\n", encoding="utf-8")
    (root / link_name).symlink_to(victim)
    return root, victim


def test_appending_through_a_symlink_out_of_the_workspace_is_refused(tmp_path):
    """THE live defect: `mode="append"` opened the link and wrote outside the root.

    Asserted on the victim file's BYTES, not only on the envelope: a refusal that still
    wrote would pass a text-only assertion, and the whole point of the finding is that
    the operation reported success while the escape happened.
    """

    from ouroboros.workspace_native import execute_native_operation

    root, victim = _escape_fixture(tmp_path)
    envelope = execute_native_operation(
        root,
        "write_file",
        {"path": "escape.txt", "content": "ATTACKER-APPENDED\n", "mode": "append"},
    ).envelope

    assert victim.read_text(encoding="utf-8") == "ORIGINAL-ONLY\n", (
        "the file outside the workspace was modified through a symlink"
    )
    assert _ESCAPE_MARKER in envelope.text, envelope.text
    assert envelope.diagnostic is not None
    assert envelope.diagnostic.code == "permission_denied"
    assert envelope.diagnostic.completion == "not_started"


def test_overwriting_through_a_symlink_out_of_the_workspace_is_refused(tmp_path):
    """The sibling mode, which did not ESCAPE but did not match the local route either.

    `os.replace` never followed the link, so nothing outside was written — but the link
    itself was silently destroyed and replaced by a regular file, where the local route
    (`tool_access.resolve_resource_path`) refuses the same call outright. Same door, so
    now both modes answer the same way, and the link is still a link afterwards.
    """

    from ouroboros.workspace_native import execute_native_operation

    root, victim = _escape_fixture(tmp_path)
    envelope = execute_native_operation(
        root,
        "write_file",
        {"path": "escape.txt", "content": "CLOBBERED\n", "mode": "overwrite"},
    ).envelope

    assert victim.read_text(encoding="utf-8") == "ORIGINAL-ONLY\n"
    assert (root / "escape.txt").is_symlink(), "the link must not be replaced either"
    assert _ESCAPE_MARKER in envelope.text, envelope.text


def test_editing_through_a_symlink_out_of_the_workspace_is_refused(tmp_path):
    """`edit_text` resolves a read path AND a write path; both must refuse."""

    from ouroboros.workspace_native import execute_native_operation

    root, victim = _escape_fixture(tmp_path)
    envelope = execute_native_operation(
        root,
        "edit_text",
        {"path": "escape.txt", "old_str": "ORIGINAL-ONLY", "new_str": "EDITED"},
    ).envelope

    assert victim.read_text(encoding="utf-8") == "ORIGINAL-ONLY\n"
    assert _ESCAPE_MARKER in envelope.text, envelope.text


def test_a_batch_write_cannot_smuggle_an_escape_in_a_later_row(tmp_path):
    """The escape is judged per ROW, and no earlier row is applied first.

    `write_file` takes a `files` list. A refusal that only looked at `args["path"]`
    would miss `files[1]`, and a refusal that judged row by row WHILE writing would
    leave a half-applied batch — the module already argues that for protected paths and
    the confinement has to hold to the same standard.
    """

    from ouroboros.workspace_native import execute_native_operation

    root, victim = _escape_fixture(tmp_path)
    envelope = execute_native_operation(
        root,
        "write_file",
        {
            "files": [
                {"path": "src/legit.py", "content": "ok\n"},
                {"path": "escape.txt", "content": "ATTACKER\n"},
            ],
            "mode": "append",
        },
    ).envelope

    assert victim.read_text(encoding="utf-8") == "ORIGINAL-ONLY\n"
    assert _ESCAPE_MARKER in envelope.text, envelope.text


def test_a_symlinked_parent_directory_cannot_host_a_new_file(tmp_path):
    """The parent half of the door, which was already right, pinned so it stays right."""

    from ouroboros.workspace_native import execute_native_operation

    root = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside)

    envelope = execute_native_operation(
        root, "write_file", {"path": "linked/new.txt", "content": "ATTACKER\n"}
    ).envelope

    assert not (outside / "new.txt").exists(), "a new file was created outside the root"
    assert _ESCAPE_MARKER in envelope.text, envelope.text


def test_an_in_root_symlink_is_written_through_only_when_its_target_is_permitted(tmp_path):
    """WHAT THIS NOW ASSERTS, and what it used to assert instead.

    It used to assert one thing: an in-root link is FOLLOWED. That half is correct and is
    still here — the door must not be satisfiable by refusing every link, because the
    local route follows one (`resolve_resource_path` returns the resolved path) and a
    native route that refused would be a new asymmetry pointing the other way.

    But "followed" was the whole test, and it had no POLICY dimension at all — so it
    pinned as CORRECT the exact behaviour that leaked: the door followed the link and
    nothing judged the file at the far end. Reproduced live, `innocent.bin -> golden.bin`
    turned a protected artifact into `TAMPERED` and the operation reported success.

    So the test now asserts a CONJUNCTION, and both halves have to hold: an in-root link
    onto an ORDINARY file is written through, and the same link onto a PROTECTED file is
    refused with the protected artifact's bytes intact. Following the link is not the
    permission; the target being permitted is.
    """

    from ouroboros.export_policy_contract import build_policy_document
    from ouroboros.workspace_native import execute_native_operation

    root = _workspace(tmp_path)
    (root / "src" / "real.txt").write_text("first\n", encoding="utf-8")
    (root / "alias.txt").symlink_to(root / "src" / "real.txt")
    (root / "src" / "golden.bin").write_text("PROTECTED-BASELINE\n", encoding="utf-8")
    (root / "innocent.bin").symlink_to(root / "src" / "golden.bin")
    facts = {
        "export_policy": build_policy_document(
            channel="workspace_snapshot", protected_paths=["src/golden.bin"]
        )
    }

    allowed = execute_native_operation(
        root,
        "write_file",
        {"path": "alias.txt", "content": "second\n", "mode": "append"},
        native_facts=facts,
    ).envelope

    assert allowed.diagnostic is None, allowed.text
    assert (root / "src" / "real.txt").read_text(encoding="utf-8") == "first\nsecond\n"
    assert (root / "alias.txt").is_symlink(), "an in-root link is followed, not replaced"

    refused = execute_native_operation(
        root,
        "write_file",
        {"path": "innocent.bin", "content": "TAMPERED\n", "mode": "overwrite"},
        native_facts=facts,
    ).envelope

    assert (root / "src" / "golden.bin").read_text(encoding="utf-8") == (
        "PROTECTED-BASELINE\n"
    ), "a protected artifact was overwritten through an in-root symlink alias"
    assert refused.diagnostic is not None, refused.text
    assert "REMOTE_PROTECTED_ARTIFACT_BLOCKED" in refused.text, refused.text
    assert "src/golden.bin" in refused.text, (
        "the refusal must name the file the alias resolves to, not only the alias"
    )


def test_the_service_log_path_is_confined_before_the_directory_is_created(tmp_path):
    """`.ouroboros/services` is INSIDE the workspace, so the model can redirect it.

    `start_service` opened `<root>/.ouroboros/services/<name>.log` in append mode with
    no confinement at all. Every component of that path is model-writable, so a prior
    `run_command` replacing `.ouroboros` with a link made the service log — and every
    later `service_logs` read — land outside the workspace. The refusal has to happen
    BEFORE `mkdir(parents=True)`, or the directory tree is materialized at the far end
    of the link even when the open is refused.
    """

    from ouroboros.workspace_native import execute_native_operation

    root = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / ".ouroboros").symlink_to(outside)

    envelope = execute_native_operation(
        root,
        "start_service",
        {"name": "svc", "cmd": ["true"], "cwd": "."},
    ).envelope

    assert not (outside / "services").exists(), (
        "the log directory was created outside the workspace"
    )
    assert _ESCAPE_MARKER in envelope.text, envelope.text


def test_an_inline_script_cannot_be_staged_outside_the_workspace(tmp_path):
    """Same shape for `run_script`'s `<root>/.ouroboros/tmp_scripts` staging directory."""

    from ouroboros.workspace_native import execute_native_operation

    root = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / ".ouroboros").symlink_to(outside)

    envelope = execute_native_operation(
        root,
        "run_script",
        {"script": "print('x')\n", "interpreter": "python3", "cwd": "."},
    ).envelope

    assert not (outside / "tmp_scripts").exists(), (
        "the script staging directory was created outside the workspace"
    )
    assert _ESCAPE_MARKER in envelope.text, envelope.text


def test_a_service_name_the_local_route_refuses_is_refused_here_too(tmp_path):
    """The native route SANITIZED what the local route REFUSES, and that merged logs.

    `re.sub(r'[^A-Za-z0-9_.-]+', '_', name)` turned `a/b` and `a_b` into one filename,
    so two services shared a log and `service_logs` could return the other one's output.
    Both routes now read the rule from one place (`SERVICE_NAME_PATTERN`), so the test
    imports it rather than restating the regex.
    """

    from ouroboros.workspace_native import execute_native_operation
    from ouroboros.workspace_native_contract import SERVICE_NAME_PATTERN

    root = _workspace(tmp_path)
    illegal = "a/b"
    assert not SERVICE_NAME_PATTERN.fullmatch(illegal)

    envelope = execute_native_operation(
        root, "start_service", {"name": illegal, "cmd": ["true"], "cwd": "."}
    ).envelope

    assert "[A-Za-z0-9_.-]" in envelope.text, envelope.text
    assert not (root / ".ouroboros" / "services" / "a_b.log").exists()


def test_the_snapshot_rollback_writes_only_through_the_door(tmp_path):
    """`_restore_rows` was the last native mutation site outside the confinement kernel.

    It deletes, writes, chmods and symlinks a path built by joining onto the root, with
    only a LEXICAL `..` check in front of it. Lexically clean is not the same as
    confined — that is the entire finding above — and a rollback runs precisely when
    something has already gone wrong. Called directly because it is a private failure
    path: the alternative is a `guarded_patch_apply` whose git apply is made to fail,
    which proves less about this loop and more about git.
    """

    from ouroboros.workspace_snapshot_native import _restore_rows

    root = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "secret.txt"
    victim.write_text("ORIGINAL-ONLY\n", encoding="utf-8")
    (root / "escape.txt").symlink_to(victim)

    errors = _restore_rows(
        root,
        [{"path": "escape.txt", "before": {"kind": "file", "mode": 0o644}, "data": b"ROLLED"}],
    )

    assert victim.read_text(encoding="utf-8") == "ORIGINAL-ONLY\n"
    assert errors and _ESCAPE_MARKER in errors[0], errors


# ── the mechanical half: no NEW mutation site may skip the door ──────────────
#
# Every per-mode test above covers a mode somebody thought of. This one covers the
# next one. It reads the native kernel modules and requires that every filesystem
# MUTATION derives its path from the confinement kernel — or is named below with a
# reason. A registry with reasons is the point: an entry is a decision on the record,
# not an exemption that accumulates silently.

_MUTATION_DOORS = frozenset({
    "_mutation_target", "native_mutation_target",
    "_target", "native_target",
    "_cwd", "native_cwd",
})
# Names that are private-by-construction: a freshly created temporary directory or file
# nobody else has a spelling for. Confining them against the workspace root would be
# meaningless — they are deliberately NOT under it.
_PRIVATE_ORIGINS = frozenset({"mkdtemp", "mkstemp", "NamedTemporaryFile", "TemporaryDirectory"})
# (module, function, receiver spelling) -> why this mutation legitimately skips the door.
_MUTATION_OUTSIDE_THE_DOOR = {
    ("workspace_payload_native.py", "stage_reviewed_payload", "skill_dir"):
        "reviewed-payload staging lives under a fresh mkdtemp root, not the workspace",
    ("workspace_payload_native.py", "stage_reviewed_payload", "destination"):
        "same stage root; the row path is validated by validate_reviewed_payload",
    ("workspace_payload_native.py", "stage_reviewed_payload", "private_home"):
        "the stage's private HOME, created inside the mkdtemp root",
    ("workspace_payload_native.py", "stage_reviewed_payload", "state_dir"):
        "the stage's skill-state dir, created inside the mkdtemp root",
    ("workspace_payload_native.py", "stage_reviewed_payload", "input_path"):
        "the stage's call input file, created inside the mkdtemp root",
    ("workspace_payload_native.py", "stage_reviewed_payload", "stage_root"):
        "the mkdtemp root itself, chmod'ed and finally removed",
    ("workspace_payload_native.py", "_run_extension_call", "state_dir"):
        "host-service side of a reviewed payload: paths come from the stage, not a model",
    ("workspace_payload_native.py", "skill_job_dir", "target"):
        "job dir under a caller-supplied private root; no workspace-relative input",
    ("workspace_payload_native.py", "_write_extension_result", "path"):
        "the stage's result file, opened O_CREAT|O_TRUNC under the mkdtemp root",
}


def _native_kernel_sources():
    import ouroboros

    base = pathlib.Path(ouroboros.__file__).parent
    return [
        base / name
        for name in (
            "workspace_native.py",
            "workspace_payload_native.py",
            "workspace_snapshot_native.py",
            "workspace_query_native.py",
            "workspace_media_native.py",
        )
    ]


def _paths_kernel_source():
    import ouroboros

    return pathlib.Path(ouroboros.__file__).parent / "workspace_native_paths.py"


def _mutation_call_sites():
    """Every filesystem mutation in the native kernel, with the name it mutates.

    Deliberately spelled as a syntactic sweep rather than a list of known sites: a list
    is what the previous guard effectively was, and the append arm was not on it.

    Two blind spots a paid reviewer found in the sweep itself, both of the shape this file
    exists to prevent — a guard that reads one spelling of a thing:

    * it only ENTERED the scan when `node.func` was an `ast.Attribute`, so the whole
      `elif verb == "open"` branch was unreachable for a BUILTIN `open(path, "w")`, whose
      func is an `ast.Name`. A future direct writable `open()` in a kernel would have
      sailed past the completeness sweep;
    * the verb tables omitted `Path.chmod`, `Path.replace` and `os.link` — and
      docs/ARCHITECTURE.md says of `_restore_rows` that it "deletes, writes, chmods and
      symlinks a path", so `chmod` is a verb this kernel really uses.
    """

    import ast

    # `chmod` is here AND in `os_verbs`, so the owner is tested first below: a
    # `path.chmod()` mutates its RECEIVER while `os.chmod(path)` mutates its ARGUMENT, and
    # reading the receiver of `os.chmod` reported the subject as `os`.
    #
    # `replace` is deliberately NOT here, and the reason is stated rather than forgotten:
    # `str.replace` and `Path.replace` are the same attribute name, so sweeping it by name
    # flags every string substitution in the kernel (`content.replace(old, new, 1)` in
    # `_edit_text` is the first one) and the exemptions needed to quiet that would be a
    # bigger hole than the verb. The kernel's atomic rename goes through `os.replace`,
    # which IS swept below and is unambiguous. RESIDUAL: a future `Path.replace()` on a
    # workspace path would not be seen by this sweep.
    path_verbs = {
        "write_text", "write_bytes", "touch", "mkdir", "symlink_to",
        "unlink", "rename", "rmdir", "hardlink_to", "chmod",
    }
    os_verbs = {
        "symlink", "chmod", "remove", "unlink", "replace", "rename", "truncate",
        "link", "mkdir", "makedirs", "rmdir",
    }
    shutil_verbs = {"rmtree", "move", "copy", "copy2", "copyfile", "copytree"}
    # `os.link(existing, new)` and `os.symlink(target, link)` mutate the SECOND argument;
    # `os.replace`/`os.rename` create the second one. Everything else mutates the first.
    second_arg_verbs = {"symlink", "replace", "rename", "link"}
    sites = []
    for source in _native_kernel_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for scope in ast.walk(tree):
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(scope):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    # The BUILTIN `open(path, "w")`. Spelled as its own case rather than
                    # folded into the attribute walk, because its subject is an ARGUMENT
                    # while `handle.open(...)`'s subject is the receiver.
                    if node.func.id != "open" or not node.args:
                        continue
                    mode = next(
                        (
                            kw.value.value
                            for kw in node.keywords
                            if kw.arg == "mode" and isinstance(kw.value, ast.Constant)
                        ),
                        node.args[1].value
                        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                        else "r",
                    )
                    if not isinstance(mode, str) or not any(c in mode for c in "wax+"):
                        continue
                    sites.append(
                        (source.name, scope.name, ast.unparse(node.args[0]), "open",
                         node.lineno)
                    )
                    continue
                if not isinstance(node.func, ast.Attribute):
                    continue
                verb = node.func.attr
                owner = ast.unparse(node.func.value)
                # The MODULE owners come first: `chmod` is both a `Path` method and an
                # `os` function, and testing the receiver first reported the subject of
                # `os.chmod(target, mode)` as `os`.
                if owner == "os" and verb in os_verbs:
                    index = 1 if verb in second_arg_verbs else 0
                    if len(node.args) <= index:
                        continue
                    subject = ast.unparse(node.args[index])
                elif verb in path_verbs and owner not in {"os", "shutil"}:
                    subject = owner
                elif owner == "shutil" and verb in shutil_verbs:
                    index = 1 if verb in {"move", "copy", "copy2", "copyfile", "copytree"} else 0
                    if len(node.args) <= index:
                        continue
                    subject = ast.unparse(node.args[index])
                elif verb == "open":
                    mode = next(
                        (
                            kw.value.value
                            for kw in node.keywords
                            if kw.arg == "mode" and isinstance(kw.value, ast.Constant)
                        ),
                        node.args[0].value
                        if node.args and isinstance(node.args[0], ast.Constant)
                        else "r",
                    )
                    if not isinstance(mode, str) or not any(c in mode for c in "wax+"):
                        continue
                    subject = owner
                else:
                    continue
                sites.append((source.name, scope.name, subject, verb, node.lineno))
    return sites


def _parameter_index(scope_node, name: str):
    """The positional index of `name` in this function's signature, or None."""

    params = [arg.arg for arg in scope_node.args.posonlyargs + scope_node.args.args]
    return params.index(name) if name in params else None


def _delegated_origins(tree, scope_name: str, index: int) -> set:
    """Origins of the argument every in-module caller passes at `index`.

    A helper that mutates one of its PARAMETERS is only as confined as its callers, so
    the sweep follows one level of delegation instead of exempting the helper — an
    exemption would cover the SECOND caller nobody has written yet, which is the shape
    of the defect this whole file is about. Any call site the sweep cannot resolve
    contributes nothing, so the check fails: the same fail-closed choice the production
    door just made about an unanswerable question.
    """

    import ast

    origins = set()
    seen_call = False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != scope_name:
            continue
        seen_call = True
        if len(node.args) <= index or not isinstance(node.args[index], ast.Name):
            return set()
        enclosing = next(
            (
                candidate
                for candidate in ast.walk(tree)
                if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
                and any(inner is node for inner in ast.walk(candidate))
            ),
            None,
        )
        if enclosing is None:
            return set()
        caller_origins = _origin_names(enclosing, node.args[index].id)
        if not caller_origins:
            return set()
        origins |= caller_origins
    return origins if seen_call else set()


def _origin_names(scope_node, name: str) -> set:
    """The call names `name` was ever bound from inside this function."""

    import ast

    origins = set()
    for node in ast.walk(scope_node):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.withitem)):
            continue
        value = node.value if not isinstance(node, ast.withitem) else node.context_expr
        targets = (
            node.targets if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign)
            else ([node.optional_vars] if node.optional_vars is not None else [])
        )
        if value is None or not any(
            isinstance(t, ast.Name) and t.id == name for t in targets if t is not None
        ):
            continue
        for inner in ast.walk(value):
            if isinstance(inner, ast.Call):
                func = inner.func
                origins.add(func.id if isinstance(func, ast.Name) else getattr(func, "attr", ""))
    return origins


def test_every_native_mutation_derives_its_path_from_the_confinement_door():
    """No native filesystem mutation may build its own path.

    THE structural close of the class. The append escape was not a missing check; it was
    a check whose PLACEMENT let one caller out of two decide differently. So the test is
    not "does append refuse" — the case above already asks that — but "can a new
    mutation site exist that never asks". Adding one fails here, and the only ways to
    pass are to route it through `native_mutation_target` or to write down why it does
    not need to.
    """

    import ast

    trees = {
        source.name: ast.parse(source.read_text(encoding="utf-8"))
        for source in _native_kernel_sources()
    }
    scopes = {
        (name, node.name): node
        for name, tree in trees.items()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    sites = _mutation_call_sites()
    assert len(sites) >= 12, (
        f"the sweep found only {len(sites)} mutation sites, which means it stopped "
        "seeing them rather than that they stopped existing"
    )

    undoored = []
    for module, function, subject, verb, lineno in sites:
        # `x.parent`, `x / "y"` and `x.joinpath(...)` inherit x's confinement: the door
        # resolved every component of x, so a child of x is still under the root.
        root_name = subject.split(".")[0].split(" / ")[0].split("[")[0].strip("()")
        if not root_name.isidentifier():
            continue
        key = (module, function, root_name)
        if key in _MUTATION_OUTSIDE_THE_DOOR:
            continue
        scope = scopes[(module, function)]
        origins = _origin_names(scope, root_name)
        index = _parameter_index(scope, root_name)
        if index is not None:
            origins |= _delegated_origins(trees[module], function, index)
        if origins & _MUTATION_DOORS or origins & _PRIVATE_ORIGINS:
            continue
        undoored.append(f"{module}:{lineno} {function}() mutates {subject}.{verb}")

    assert not undoored, (
        "native filesystem mutation outside the confinement kernel:\n  "
        + "\n  ".join(undoored)
        + "\n\nRoute the path through `workspace_native_paths.native_mutation_target`, "
        "or add it to `_MUTATION_OUTSIDE_THE_DOOR` with the reason it is safe."
    )


def test_every_door_call_site_names_a_policy_question():
    """The second half of the class: a door that resolves must also JUDGE.

    The confinement sweep above proves no mutation builds its own path. It says nothing
    about what the door then hands over, and that is where the leak was: the policy was
    applied to the requested SPELLING and the door resolved a different file. So both
    doors now take a REQUIRED `question`/`facts` keyword — a caller that omits one does
    not run at all — and this sweep is the completeness half: every call site in the
    native kernel must pass them EXPLICITLY, so the audit is a grep rather than a memory.

    `QUESTION_NONE` is a legal answer and is deliberately spelled rather than implied,
    because "no question" has to be a decision on the record.
    """

    import ast

    from ouroboros.export_policy_contract import EXPORT_QUESTIONS

    doors = {"_target", "native_target"}
    mutation_doors = {"_mutation_target", "native_mutation_target"}
    sources = [*_native_kernel_sources(), _paths_kernel_source()]
    unjudged = []
    seen = 0
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            name = node.func.id
            if name not in doors and name not in mutation_doors:
                continue
            seen += 1
            keywords = {kw.arg for kw in node.keywords}
            required = {"facts"} | ({"question"} if name in doors else set())
            missing = sorted(required - keywords)
            if missing:
                unjudged.append(f"{source.name}:{node.lineno} {name}() omits {missing}")
    assert seen >= 8, (
        f"the sweep found only {seen} door call sites, which means it stopped seeing "
        "them rather than that they stopped existing"
    )
    assert not unjudged, (
        "a path door was called without naming the policy question its caller asks:\n  "
        + "\n  ".join(unjudged)
        + f"\n\nPass question=<one of {sorted(EXPORT_QUESTIONS)}> and facts=<the "
        "operation's native_facts, or None for an unbound operation>."
    )


def test_the_question_registry_and_the_doors_cannot_drift_apart():
    """A question the doors cannot express, or a door that accepts an unknown one.

    Both halves fail closed and both are asserted, because the closed set is only worth
    having if an unrecognised value is refused rather than treated as "no question".
    """

    import inspect

    from ouroboros.export_policy_contract import (
        EXPORT_QUESTIONS,
        ExportChannelUnknownError,
        judged_exclusion,
        unaliased_exclusion,
    )
    from ouroboros.workspace_native_paths import native_mutation_target, native_target

    for door in (native_target, native_mutation_target):
        signature = inspect.signature(door)
        assert signature.parameters["facts"].default is inspect.Parameter.empty, (
            f"{door.__name__} lets a caller skip `facts`, so a new door can skip the policy"
        )
    assert (
        native_target.__signature__.parameters["question"].default
        if hasattr(native_target, "__signature__")
        else inspect.signature(native_target).parameters["question"].default
    ) is inspect.Parameter.empty, "the read door lets a caller skip `question`"
    # BOTH public doors, because an unknown question accepted by either of them is a
    # door that decided the question did not matter.
    for ask in (
        lambda: unaliased_exclusion("x", {}, question="whatever_i_like"),
        lambda: judged_exclusion("/nope", None, "x", {}, question="whatever_i_like"),
    ):
        try:
            ask()
        except (ExportChannelUnknownError, ValueError):
            continue
        raise AssertionError("an unknown policy question was accepted instead of refused")
    assert "none" in EXPORT_QUESTIONS, (
        "the 'no question' answer must be a NAMED member of the closed set, or a door "
        "that skips the policy is indistinguishable from one that decided to"
    )


def test_the_mutation_sweep_would_notice_a_new_undoored_site():
    """The sweep itself must be shown FAILING, or it is another vacuous guard.

    A syntactic sweep that silently matched nothing would keep this file green forever.
    So the parser is pointed at a constructed module holding one honest violation and
    one honest non-violation, and both verdicts are asserted.
    """

    import ast

    module = ast.parse(
        "import pathlib\n"
        "def offender(root, rel):\n"
        "    target = root / rel\n"
        "    target.write_bytes(b'x')\n"
        "def compliant(root, rel):\n"
        "    target = _mutation_target(root, rel)\n"
        "    target.write_bytes(b'x')\n"
    )
    scopes = {
        node.name: node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef)
    }
    assert not (_origin_names(scopes["offender"], "target") & _MUTATION_DOORS), (
        "a path built by joining onto the root must NOT read as door-derived"
    )
    assert _origin_names(scopes["compliant"], "target") & _MUTATION_DOORS, (
        "a path returned by the door must read as door-derived"
    )


def test_the_sweep_sees_every_mutating_spelling_it_claims_to(tmp_path, monkeypatch):
    """The VERB half of the same proof, and it earned its own test by being wrong.

    A paid reviewer showed the sweep entered its scan only for an `ast.Attribute` callee,
    so a BUILTIN `open(path, "w")` — an `ast.Name` — was invisible and the `elif verb ==
    "open"` branch below it was dead code for that form. The verb tables also omitted
    `Path.chmod` and `os.link`, and ARCHITECTURE.md says of `_restore_rows` that it
    "chmods" a path, so that was a live verb the completeness sweep could not see.

    Every spelling the sweep claims is asserted against a constructed module, because "the
    tables look right" is what was true before.
    """

    source = tmp_path / "kernel_probe.py"
    source.write_text(
        "import os, shutil, pathlib\n"
        "def offender(target, other):\n"
        "    open(target, 'w')\n"
        "    open(target, mode='a')\n"
        "    open(target)\n"
        "    target.chmod(0o600)\n"
        "    os.chmod(target, 0o600)\n"
        "    os.link(other, target)\n"
        "    os.symlink(other, target)\n"
        "    os.replace(other, target)\n"
        "    shutil.move(other, target)\n"
        "    target.write_bytes(b'x')\n",
        encoding="utf-8",
    )
    import sys

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_native_kernel_sources", lambda: [source])
    rows = module._mutation_call_sites()
    seen = {(verb, subject) for _m, _f, subject, verb, _l in rows}
    for verb in ("open", "chmod", "link", "symlink", "replace", "move", "write_bytes"):
        assert any(row[0] == verb for row in seen), f"the sweep cannot see {verb}"
    # `open(target)` with no writable mode must NOT be a site — TWO of the three `open`
    # calls are writes. A sweep that flagged reads would need exemptions broad enough to
    # swallow a real write, which is the failure mode the whole file is about.
    assert sum(1 for _m, _f, _s, verb, _l in rows if verb == "open") == 2, rows
    assert all(subject == "target" for _v, subject in seen), sorted(seen)
