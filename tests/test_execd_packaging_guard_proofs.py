"""Every prohibition the execd packaging line declares is proven by a REFUSAL.

Three guards in this branch turned out to be structurally unable to fire, and all
three were found by hand: the elision gate behind `if False and ...`, the missing
symlink-confinement case, and `build_execd_bundle.FORBIDDEN_HOME_MODULES`, whose
entries were repo-relative spellings (`ouroboros/config.py`) compared against stage
paths (`lib/ouroboros/config.py`) — so it had never refused anything.  Nothing
noticed, because the packaging scripts had ZERO behavioural coverage: 64 `raise`
statements across the three of them, not one of which any test had ever executed.

So the rule is not "add a test for that one guard".  It is:

    a guard that forbids something must be shown REFUSING something.

expressed so it cannot be satisfied by inspection.  `CASES` below is the refusal
evidence: each entry constructs a real violation and names the message fragment the
guard must answer with.  `test_every_declared_prohibition_is_proven_or_reasoned`
then reads the three scripts' `raise` statements straight out of the AST and demands
that every message fragment in the source is either proven by a case or carries a
written reason.  A guard that CANNOT fire cannot produce its fragment, so it fails
here; a guard added later without a case fails here too; and a reason that has
become false (the case now exists, or the message is gone) also fails, so the
exemption list cannot quietly rot.
"""

from __future__ import annotations

import ast
import gzip
import io
import json
import os
import pathlib
import tarfile
import zipfile

import pytest

import scripts.assemble_execd_stage as stage_script
import scripts.build_execd_bundle as bundle_script
import scripts.smoke_execd_stage as smoke_script

REPO = pathlib.Path(__file__).resolve().parents[1]
GUARD_SCRIPTS = (
    "scripts/build_execd_bundle.py",
    "scripts/assemble_execd_stage.py",
    "scripts/smoke_execd_stage.py",
)
# Raises that carry no message of their own: re-raise and process-exit plumbing.
# Shapes are matched, not line numbers, so a real guard cannot hide in here.
PLUMBING_SHAPES = (
    "raise",
    "raise SystemExit(_main())",
    "raise SystemExit(main(arguments.stage))",
)
# A prohibition may go unproven only with a reason, keyed by its own message so a
# changed or deleted message forces the reason to be revisited.
UNPROVEN_WITH_REASON = {
    "assembled execd failed smoke": (
        "`_smoke` runs the assembled launcher, so it needs a real PBS runtime on a "
        "Linux runner of the stage's own architecture; the architecture-native "
        "`execd-stage` CI job is where it is exercised."
    ),
    "assembled execd launcher mutated its immutable stage": (
        "Same as above: requires the real launcher to have run."
    ),
    "assembled execd import smoke failed": (
        "Same as above: requires the staged interpreter to import the kernel."
    ),
    "assembled execd loaded Home imports": (
        "Same as above. The prohibition itself is proven in the packager instead: "
        "`Home-only module leaked into execd stage` judges the same "
        "FORBIDDEN_REMOTE_IMPORT_PREFIXES source of truth, offline."
    ),
    "assembled execd import smoke mutated its immutable stage": (
        "Same as above: requires the staged interpreter to have run."
    ),
    "execd handshake facts are incomplete": (
        "`_service_smoke` drives a live ExecdService inside an assembled stage."
    ),
    "execd operation did not complete": ("Same as above: live service required."),
    "execd write failed": ("Same as above: live service required."),
    "execd read did not observe the write": ("Same as above: live service required."),
    "execd process result is invalid": ("Same as above: live service required."),
    "execd typed error was lost": ("Same as above: live service required."),
    "bundled tree-sitter Go grammar failed": (
        "Needs the wheels staged into a real runtime; the stage job runs it."
    ),
    "bundled ffmpeg digest differs from stage provenance": (
        "Same as above: needs the staged ffmpeg binary."
    ),
    "bundled structural query failed": (
        "Same as above: needs the staged tree-sitter grammars."
    ),
}

LOCK = json.loads(
    (REPO / "scripts" / "execd_dependency_lock.json").read_text(encoding="utf-8")
)
ARCHITECTURES = ("x86_64", "aarch64")


# ── stage fixtures ──────────────────────────────────────────────────────────


def _write_stage(
    root: pathlib.Path,
    architecture: str,
    *,
    extra_files: tuple[tuple[str, str], ...] = (),
    modules: tuple[str, ...] = ("ouroboros", "ouroboros.execd"),
    declared: tuple[str, ...] | None = None,
    provenance_edit=None,
) -> pathlib.Path:
    """A minimal stage in the layout `assemble_execd_stage` really produces."""

    stage = root / f"stage-{architecture}"
    (stage / "bin").mkdir(parents=True)
    (stage / "bin" / "ouroboros-execd").write_text("#!/bin/sh\n", encoding="utf-8")
    (stage / "bin" / "rg").write_text("rg\n", encoding="utf-8")
    for module in modules:
        relative = (
            pathlib.Path("ouroboros", "__init__.py")
            if module == "ouroboros"
            else pathlib.Path(*module.split(".")).with_suffix(".py")
        )
        target = stage / "lib" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    for relative_path, body in extra_files:
        target = stage / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    provenance = {
        "schema_version": 1,
        "architecture": architecture,
        "python_build_standalone": LOCK["python_build_standalone"]["architectures"][
            architecture
        ],
        "ripgrep": LOCK["ripgrep"]["architectures"][architecture],
        "video_helper": LOCK["video_helper"]["architectures"][architecture],
        "python_wheels": LOCK["python_wheels"][architecture],
        "kernel_roots": ["ouroboros.execd"],
        "kernel_modules": list(modules if declared is None else declared),
        "contract_set_version": 1,
    }
    if provenance_edit is not None:
        provenance_edit(provenance)
    (stage / "stage-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stage


def _build(tmp_path: pathlib.Path, *, lock=None, **stage_kwargs):
    stages = {
        architecture: _write_stage(tmp_path, architecture, **stage_kwargs)
        for architecture in ARCHITECTURES
    }
    return bundle_script.build(
        version="0.0.0-guard-proof",
        stages=stages,
        output_dir=tmp_path / "out",
        dependency_lock=LOCK if lock is None else lock,
    )


def _lock_without(*keys: str) -> dict:
    lock = json.loads(json.dumps(LOCK))
    for key in keys:
        lock.pop(key, None)
    return lock


def _tar_gz(target: pathlib.Path, members: tuple[tuple[str, bytes], ...]) -> pathlib.Path:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    target.write_bytes(gzip.compress(raw.getvalue()))
    return target


# ── build_execd_bundle.py ───────────────────────────────────────────────────


def _case_stage_not_a_directory(tmp_path, monkeypatch):  # noqa: ARG001
    bundle_script._stage_files(tmp_path / "absent", frozenset({"ouroboros"}))


def _case_stage_link(tmp_path, monkeypatch):  # noqa: ARG001
    stage = _write_stage(tmp_path, "x86_64")
    (stage / "bin" / "link").symlink_to(stage / "bin" / "rg")
    bundle_script._stage_files(stage, frozenset({"ouroboros", "ouroboros.execd"}))


def _case_stage_special_file(tmp_path, monkeypatch):  # noqa: ARG001
    stage = _write_stage(tmp_path, "x86_64")
    os.mkfifo(stage / "bin" / "pipe")
    bundle_script._stage_files(stage, frozenset({"ouroboros", "ouroboros.execd"}))


def _case_library_non_module(tmp_path, monkeypatch):  # noqa: ARG001
    _build(tmp_path, extra_files=(("lib/ouroboros/data.json", "{}"),))


def _case_home_module_leak(tmp_path, monkeypatch):  # noqa: ARG001
    # The real namespace: `lib/` prefixed, which is where a leak would land, and
    # declared in the provenance too so ONLY the prohibition can refuse it.
    _build(
        tmp_path,
        modules=("ouroboros", "ouroboros.execd", "ouroboros.config"),
    )


def _case_undeclared_module(tmp_path, monkeypatch):  # noqa: ARG001
    _build(
        tmp_path,
        modules=("ouroboros", "ouroboros.execd", "ouroboros.shell_parse"),
        declared=("ouroboros", "ouroboros.execd"),
    )


def _case_stage_over_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle_script, "MAX_FILES", 1)
    _build(tmp_path)


def _case_missing_required_files(tmp_path, monkeypatch):  # noqa: ARG001
    stage = _write_stage(tmp_path, "x86_64")
    (stage / "bin" / "rg").unlink()
    bundle_script._stage_files(stage, frozenset({"ouroboros", "ouroboros.execd"}))


def _case_unsupported_lock(tmp_path, monkeypatch):  # noqa: ARG001
    bundle_script._validate_dependency_lock({"schema_version": 99})


def _case_lock_version_drift(tmp_path, monkeypatch):  # noqa: ARG001
    lock = json.loads(json.dumps(LOCK))
    lock["ripgrep"]["version"] = "0.0.0"
    bundle_script._validate_dependency_lock(lock)


def _case_lock_architecture_drift(tmp_path, monkeypatch):  # noqa: ARG001
    lock = json.loads(json.dumps(LOCK))
    lock["python_build_standalone"]["architectures"]["x86_64"]["sha256"] = "0" * 64
    bundle_script._validate_dependency_lock(lock)


def _case_lock_incomplete(tmp_path, monkeypatch):  # noqa: ARG001
    bundle_script._validate_dependency_lock(_lock_without("ripgrep"))


def _case_provenance_missing(tmp_path, monkeypatch):  # noqa: ARG001
    stage = _write_stage(tmp_path, "x86_64")
    (stage / "stage-provenance.json").unlink()
    bundle_script._validate_stage_provenance(stage, "x86_64", LOCK)


def _case_provenance_differs(tmp_path, monkeypatch):  # noqa: ARG001
    def edit(provenance):
        provenance["ripgrep"] = {"filename": "rg", "url": "https://x", "sha256": "0" * 64}

    stage = _write_stage(tmp_path, "x86_64", provenance_edit=edit)
    bundle_script._validate_stage_provenance(stage, "x86_64", LOCK)


def _case_provenance_without_modules(tmp_path, monkeypatch):  # noqa: ARG001
    def edit(provenance):
        provenance["kernel_modules"] = ["", 7]

    stage = _write_stage(tmp_path, "x86_64", provenance_edit=edit)
    bundle_script._validate_stage_provenance(stage, "x86_64", LOCK)


def _case_single_architecture(tmp_path, monkeypatch):  # noqa: ARG001
    bundle_script.build(
        version="0.0.0-guard-proof",
        stages={"x86_64": _write_stage(tmp_path, "x86_64")},
        output_dir=tmp_path / "out",
        dependency_lock=LOCK,
    )


def _case_contract_set_disagreement(tmp_path, monkeypatch):  # noqa: ARG001
    stages = {}
    for index, architecture in enumerate(ARCHITECTURES):

        def edit(provenance, value=index + 1):
            provenance["contract_set_version"] = value

        stages[architecture] = _write_stage(
            tmp_path, architecture, provenance_edit=edit
        )
    bundle_script.build(
        version="0.0.0-guard-proof",
        stages=stages,
        output_dir=tmp_path / "out",
        dependency_lock=LOCK,
    )


def _case_cli_lock_not_an_object(tmp_path, monkeypatch):
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_execd_bundle.py",
            "--version",
            "0.0.0-guard-proof",
            "--x86-stage",
            str(tmp_path / "x86"),
            "--aarch64-stage",
            str(tmp_path / "aarch64"),
            "--dependency-lock",
            str(lock_path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    bundle_script._main()


# ── assemble_execd_stage.py ─────────────────────────────────────────────────


def _case_artifact_not_an_object(tmp_path, monkeypatch):  # noqa: ARG001
    stage_script._artifact(["not", "a", "mapping"])


def _case_artifact_metadata_invalid(tmp_path, monkeypatch):  # noqa: ARG001
    stage_script._artifact(
        {"filename": "../escape.tar.gz", "url": "https://x/y", "sha256": "a" * 64}
    )


def _lock_file(tmp_path: pathlib.Path, mutate) -> pathlib.Path:
    lock = json.loads(json.dumps(LOCK))
    mutate(lock)
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    return path


def _case_load_lock_unsupported(tmp_path, monkeypatch):  # noqa: ARG001
    def mutate(lock):
        lock["schema_version"] = 2

    stage_script.load_lock(_lock_file(tmp_path, mutate), "x86_64")


def _case_load_lock_architecture_absent(tmp_path, monkeypatch):  # noqa: ARG001
    stage_script.load_lock(_lock_file(tmp_path, lambda lock: None), "riscv64")


def _case_load_lock_wheels_not_an_array(tmp_path, monkeypatch):  # noqa: ARG001
    def mutate(lock):
        lock["python_wheels"]["x86_64"] = {"wheel": "one"}

    stage_script.load_lock(_lock_file(tmp_path, mutate), "x86_64")


def _case_load_lock_video_invalid(tmp_path, monkeypatch):  # noqa: ARG001
    def mutate(lock):
        lock["video_helper"]["architectures"]["x86_64"]["path"] = "../ffmpeg"

    stage_script.load_lock(_lock_file(tmp_path, mutate), "x86_64")


def _case_load_lock_duplicate_wheels(tmp_path, monkeypatch):  # noqa: ARG001
    def mutate(lock):
        wheels = lock["python_wheels"]["x86_64"]
        wheels.append(json.loads(json.dumps(wheels[0])))

    stage_script.load_lock(_lock_file(tmp_path, mutate), "x86_64")


def _case_load_lock_missing_package(tmp_path, monkeypatch):  # noqa: ARG001
    def mutate(lock):
        lock["python_wheels"]["x86_64"] = [
            row
            for row in lock["python_wheels"]["x86_64"]
            if not row["filename"].startswith("tree_sitter-")
        ]

    stage_script.load_lock(_lock_file(tmp_path, mutate), "x86_64")


def _case_unsafe_archive_member(tmp_path, monkeypatch):  # noqa: ARG001
    stage_script._relative("../../etc/passwd")


def _case_download_over_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_script, "MAX_DOWNLOAD_BYTES", 4)
    monkeypatch.setattr(
        stage_script.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _FakeResponse(b"x" * 64),
    )
    stage_script._download(
        {"filename": "f.tar.gz", "url": "https://x/f.tar.gz", "sha256": "a" * 64},
        tmp_path / "cache",
    )


def _case_download_digest_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage_script.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _FakeResponse(b"wrong bytes"),
    )
    stage_script._download(
        {"filename": "f.tar.gz", "url": "https://x/f.tar.gz", "sha256": "a" * 64},
        tmp_path / "cache",
    )


class _FakeResponse:
    """The read-in-chunks contract `_download` uses, with no network."""

    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)

    def read(self, size: int) -> bytes:
        return self._stream.read(size)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info) -> None:
        return None


def _case_python_archive_escapes(tmp_path, monkeypatch):  # noqa: ARG001
    archive = _tar_gz(tmp_path / "py.tar.gz", (("elsewhere/bin/python3.12", b""),))
    stage_script._extract_python(archive, tmp_path / "runtime")


def _case_python_archive_special_file(tmp_path, monkeypatch):  # noqa: ARG001
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as bundle:
        info = tarfile.TarInfo("python/lib/device")
        info.type = tarfile.CHRTYPE
        bundle.addfile(info)
    (tmp_path / "py.tar.gz").write_bytes(gzip.compress(raw.getvalue()))
    stage_script._extract_python(tmp_path / "py.tar.gz", tmp_path / "runtime")


def _case_python_archive_over_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_script, "MAX_STAGE_FILES", 0)
    archive = _tar_gz(tmp_path / "py.tar.gz", (("python/bin/python3.12", b"x"),))
    stage_script._extract_python(archive, tmp_path / "runtime")


def _case_python_member_unreadable(tmp_path, monkeypatch):
    monkeypatch.setattr(tarfile.TarFile, "extractfile", lambda *args, **kwargs: None)
    archive = _tar_gz(tmp_path / "py.tar.gz", (("python/bin/python3.12", b"x"),))
    stage_script._extract_python(archive, tmp_path / "runtime")


def _case_python_archive_without_interpreter(tmp_path, monkeypatch):  # noqa: ARG001
    archive = _tar_gz(tmp_path / "py.tar.gz", (("python/lib/libpython.so", b"x"),))
    stage_script._extract_python(archive, tmp_path / "runtime")


def _case_wheel_link(tmp_path, monkeypatch):  # noqa: ARG001
    wheel = tmp_path / "pkg.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        info = zipfile.ZipInfo("pkg/link.py")
        info.external_attr = 0xA1FF << 16
        archive.writestr(info, "target")
    stage_script._extract_wheel(wheel, tmp_path / "site-packages")


def _case_wheel_member_over_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_script, "MAX_STAGE_BYTES", 1)
    wheel = tmp_path / "pkg.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pkg/__init__.py", "x" * 32)
    stage_script._extract_wheel(wheel, tmp_path / "site-packages")


def _case_ripgrep_not_exactly_one(tmp_path, monkeypatch):  # noqa: ARG001
    archive = _tar_gz(
        tmp_path / "rg.tar.gz", (("a/rg", b"one"), ("b/rg", b"two")),
    )
    stage_script._extract_ripgrep(archive, tmp_path / "bin" / "rg")


def _case_ripgrep_unreadable(tmp_path, monkeypatch):
    archive = _tar_gz(tmp_path / "rg.tar.gz", (("a/rg", b"one"),))
    monkeypatch.setattr(tarfile.TarFile, "extractfile", lambda *args, **kwargs: None)
    stage_script._extract_ripgrep(archive, tmp_path / "bin" / "rg")


def _case_closure_non_package_module(tmp_path, monkeypatch):  # noqa: ARG001
    stage_script._copy_kernel(REPO, tmp_path / "lib", ["server"])


def _case_kernel_source_missing(tmp_path, monkeypatch):  # noqa: ARG001
    stage_script._copy_kernel(REPO, tmp_path / "lib", ["ouroboros.absent_module_xyz"])


def _case_stage_path_not_sha256sum_safe(tmp_path, monkeypatch):  # noqa: ARG001
    (tmp_path / "two words.txt").write_text("x", encoding="utf-8")
    stage_script._write_stage_checksums(tmp_path)


def _case_assemble_unknown_architecture(tmp_path, monkeypatch):  # noqa: ARG001
    stage_script.assemble(
        repo_root=REPO,
        architecture="riscv64",
        output=tmp_path / "out",
        cache=tmp_path / "cache",
        lock_path=REPO / "scripts" / "execd_dependency_lock.json",
    )


def _case_assemble_output_exists(tmp_path, monkeypatch):  # noqa: ARG001
    (tmp_path / "out").mkdir()
    stage_script.assemble(
        repo_root=REPO,
        architecture="x86_64",
        output=tmp_path / "out",
        cache=tmp_path / "cache",
        lock_path=REPO / "scripts" / "execd_dependency_lock.json",
    )


def _case_assemble_authority_tree_differs(tmp_path, monkeypatch):  # noqa: ARG001
    """`--repo-root` naming a tree that is not the one the authorities came from.

    The closure algorithm and `CONTRACT_SET_VERSION` are imported through
    `sys.path`; the modules that get copied are read from `--repo-root`. A
    plausible-looking stage built by one tree's rules and stamped with another
    tree's contract number is exactly what the compatibility gate cannot survive.
    """

    (tmp_path / "elsewhere").mkdir()
    stage_script.assemble(
        repo_root=tmp_path / "elsewhere",
        architecture="x86_64",
        output=tmp_path / "out",
        cache=tmp_path / "cache",
        lock_path=REPO / "scripts" / "execd_dependency_lock.json",
    )


def _case_smoke_under_a_host_interpreter(tmp_path, monkeypatch):  # noqa: ARG001
    """The smoke must refuse to grade an artifact from outside it.

    Run under a host Python, the stage's `lib/` would be imported into the host's
    own runtime and everything would pass — while proving nothing about the
    interpreter that actually reaches the target.
    """

    stage = tmp_path / "stage"
    stage.mkdir()
    smoke_script.main(stage)


def _case_staged_ffmpeg_differs(tmp_path, monkeypatch):
    # The pinned inputs are stubbed out — what is under test is the digest gate on
    # the ffmpeg the wheels were supposed to have staged, which nothing staged.
    monkeypatch.setattr(
        stage_script, "_download", lambda row, cache: tmp_path / "unused"
    )
    for name in ("_extract_python", "_extract_wheel", "_extract_ripgrep"):
        monkeypatch.setattr(stage_script, name, lambda *args, **kwargs: None)
    stage_script.assemble(
        repo_root=REPO,
        architecture="x86_64",
        output=tmp_path / "out",
        cache=tmp_path / "cache",
        lock_path=REPO / "scripts" / "execd_dependency_lock.json",
    )


# ── smoke_execd_stage.py ────────────────────────────────────────────────────


def _hashed_stage(tmp_path: pathlib.Path, rows: tuple[str, ...]) -> pathlib.Path:
    (tmp_path / "stage-files.sha256").write_text(
        "".join(f"{row}\n" for row in rows), encoding="utf-8"
    )
    return tmp_path


def _case_checksum_manifest_malformed(tmp_path, monkeypatch):  # noqa: ARG001
    smoke_script._verify_stage_hashes(_hashed_stage(tmp_path, ("nonsense",)))


def _case_checksum_manifest_unsafe_path(tmp_path, monkeypatch):  # noqa: ARG001
    smoke_script._verify_stage_hashes(
        _hashed_stage(tmp_path, (f"{'a' * 64}  ../outside",))
    )


def _case_stage_contains_a_link(tmp_path, monkeypatch):  # noqa: ARG001
    stage = _hashed_stage(tmp_path, (f"{'a' * 64}  bin/rg",))
    (stage / "bin").mkdir()
    (stage / "bin" / "rg").write_text("rg", encoding="utf-8")
    (stage / "bin" / "link").symlink_to(stage / "bin" / "rg")
    smoke_script._verify_stage_hashes(stage)


def _case_checksum_tree_mismatch(tmp_path, monkeypatch):  # noqa: ARG001
    smoke_script._verify_stage_hashes(_hashed_stage(tmp_path, (f"{'a' * 64}  bin/rg",)))


def _case_checksum_mismatch(tmp_path, monkeypatch):  # noqa: ARG001
    stage = _hashed_stage(tmp_path, (f"{'a' * 64}  bin/rg",))
    (stage / "bin").mkdir()
    (stage / "bin" / "rg").write_text("rg", encoding="utf-8")
    smoke_script._verify_stage_hashes(stage)


# ── the refusal evidence ────────────────────────────────────────────────────

CASES: tuple[tuple[str, str, object], ...] = (
    ("stage is not a directory", "stage is not a directory", _case_stage_not_a_directory),
    ("stage link", "stage links are forbidden", _case_stage_link),
    ("stage special file", "stage special file is forbidden", _case_stage_special_file),
    ("library non-module", "execd stage library holds a non-module", _case_library_non_module),
    ("Home module leak", "Home-only module leaked into execd stage", _case_home_module_leak),
    ("undeclared module", "execd stage carries an undeclared module", _case_undeclared_module),
    ("stage over limits", "execd stage exceeds deterministic bundle limits", _case_stage_over_limits),
    ("required files absent", "execd stage is missing required files", _case_missing_required_files),
    ("lock unsupported", "unsupported execd dependency lock", _case_unsupported_lock),
    ("lock version drift", "execd dependency lock version drift", _case_lock_version_drift),
    ("lock architecture drift", "execd dependency lock architecture drift", _case_lock_architecture_drift),
    ("lock incomplete", "execd dependency lock is incomplete", _case_lock_incomplete),
    ("provenance absent", "execd stage provenance is missing or invalid", _case_provenance_missing),
    ("provenance drift", "execd stage provenance differs from dependency lock", _case_provenance_differs),
    ("provenance without modules", "execd stage provenance declares no kernel module set", _case_provenance_without_modules),
    ("one architecture", "both x86_64 and aarch64 stages are required", _case_single_architecture),
    ("contract set disagreement", "execd stages declare no single Home↔execd contract set", _case_contract_set_disagreement),
    ("cli lock not an object", "dependency lock must be a JSON object", _case_cli_lock_not_an_object),
    ("artifact not an object", "dependency lock artifact must be an object", _case_artifact_not_an_object),
    ("artifact metadata", "dependency lock artifact metadata is invalid", _case_artifact_metadata_invalid),
    ("load_lock unsupported", "unsupported execd dependency lock", _case_load_lock_unsupported),
    ("load_lock architecture", "dependency lock does not support", _case_load_lock_architecture_absent),
    ("load_lock wheel set", "dependency lock wheel set must be an array", _case_load_lock_wheels_not_an_array),
    ("load_lock video helper", "dependency lock video helper is invalid", _case_load_lock_video_invalid),
    ("load_lock duplicate wheels", "dependency lock contains duplicate wheels", _case_load_lock_duplicate_wheels),
    ("load_lock package omitted", "dependency lock omits an approved execd runtime package", _case_load_lock_missing_package),
    ("unsafe archive member", "unsafe archive member", _case_unsafe_archive_member),
    ("download over limit", "dependency download exceeds stage limit", _case_download_over_limit),
    ("download digest", "dependency digest mismatch", _case_download_digest_mismatch),
    ("python archive escape", "python archive escaped its top-level directory", _case_python_archive_escapes),
    ("python special file", "python archive contains a special file", _case_python_archive_special_file),
    ("python over limits", "python runtime exceeds stage limits", _case_python_archive_over_limits),
    ("python member unreadable", "python archive member is unreadable", _case_python_member_unreadable),
    ("python without interpreter", "python archive omitted python3.12", _case_python_archive_without_interpreter),
    ("wheel link", "wheel links are forbidden", _case_wheel_link),
    ("wheel member size", "wheel member exceeds stage limit", _case_wheel_member_over_limit),
    ("ripgrep count", "ripgrep archive must contain exactly one rg binary", _case_ripgrep_not_exactly_one),
    ("ripgrep unreadable", "ripgrep binary is unreadable", _case_ripgrep_unreadable),
    ("closure non-package", "execd closure contains a non-package module", _case_closure_non_package_module),
    ("kernel source absent", "execd kernel source is missing", _case_kernel_source_missing),
    ("stage path unsafe for sha256sum", "execd stage paths must be sha256sum-safe", _case_stage_path_not_sha256sum_safe),
    ("assemble architecture", "architecture must be x86_64 or aarch64", _case_assemble_unknown_architecture),
    ("assemble output exists", "output already exists", _case_assemble_output_exists),
    ("assemble authority tree", "execd stage authorities come from a different tree than --repo-root: importable ouroboros lives under", _case_assemble_authority_tree_differs),
    ("smoke host interpreter", "stage smoke must run under the staged interpreter, not", _case_smoke_under_a_host_interpreter),
    ("staged ffmpeg digest", "staged ffmpeg differs from the approved dependency lock", _case_staged_ffmpeg_differs),
    ("checksum manifest malformed", "stage checksum manifest is malformed", _case_checksum_manifest_malformed),
    ("checksum manifest path", "stage checksum manifest contains an unsafe path", _case_checksum_manifest_unsafe_path),
    ("smoke stage link", "stage contains a link", _case_stage_contains_a_link),
    ("checksum tree mismatch", "stage checksum manifest does not match the artifact tree", _case_checksum_tree_mismatch),
    ("checksum digest mismatch", "stage checksum mismatch", _case_checksum_mismatch),
)


@pytest.mark.parametrize(
    ("fragment", "violate"),
    [pytest.param(fragment, violate, id=name) for name, fragment, violate in CASES],
)
def test_declared_prohibition_refuses_its_violation(fragment, violate, tmp_path, monkeypatch):
    """Feed the guard a violating input and require the refusal, by message."""

    with pytest.raises((ValueError, RuntimeError, SystemExit)) as caught:
        violate(tmp_path, monkeypatch)
    assert fragment in str(caught.value), (
        f"guard answered {str(caught.value)!r}, which does not contain {fragment!r}"
    )


def test_the_packaging_line_still_accepts_a_clean_stage(tmp_path):
    """The refusals above must be refusals of the VIOLATION, not of everything."""

    manifest = _build(tmp_path)
    assert manifest["contract_set_version"] == 1
    assert sorted(manifest["assets"]) == ["linux-aarch64", "linux-x86_64"]


def _message_fragments(relative: str) -> dict[str, int]:
    """Every raise in `relative`, keyed by the literal text it answers with.

    Read from the AST rather than from a hand list, so the demand below covers
    guards nobody remembered to mention.
    """

    source = (REPO / relative).read_text(encoding="utf-8")
    tree = ast.parse(source)
    fragments: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        literals: list[str] = []
        for candidate in ast.walk(node):
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                literals.append(candidate.value)
        longest = max(literals, key=len, default="")
        if len(longest.strip()) < 8:
            shape = " ".join((ast.get_source_segment(source, node) or "").split())
            assert shape in PLUMBING_SHAPES, (
                f"{relative}:{node.lineno} raises without a message it can be "
                f"recognized by: {shape!r}"
            )
            continue
        fragments[longest.strip().rstrip(":").strip()] = node.lineno
    return fragments


def test_every_declared_prohibition_is_proven_or_reasoned():
    """No guard in the packaging line may rest on inspection.

    This is the structural half. `_message_fragments` enumerates the prohibitions
    from the source, so the demand cannot be satisfied by editing a list: a guard
    that cannot fire never produces its message and lands in `unproven`, and a new
    guard lands there too until it has a case or a written reason.
    """

    proven = {fragment for _name, fragment, _violate in CASES}
    declared: dict[str, str] = {}
    for relative in GUARD_SCRIPTS:
        for fragment, lineno in _message_fragments(relative).items():
            declared[fragment] = f"{relative}:{lineno}"

    unproven = {
        fragment: site
        for fragment, site in declared.items()
        if fragment not in proven and fragment not in UNPROVEN_WITH_REASON
    }
    assert not unproven, (
        "these prohibitions are declared but never shown refusing anything — add a "
        f"case to CASES, or a reason to UNPROVEN_WITH_REASON: {unproven}"
    )
    # An exemption that has become false must go, or it turns into the same
    # unexamined claim the whole module exists to prevent.
    stale = sorted(set(UNPROVEN_WITH_REASON) - set(declared))
    assert not stale, f"UNPROVEN_WITH_REASON names messages no guard raises: {stale}"
    settled = sorted(set(UNPROVEN_WITH_REASON) & proven)
    assert not settled, f"these are proven now, so drop their reason: {settled}"
    assert all(len(reason) > 30 for reason in UNPROVEN_WITH_REASON.values())
    # Every case must name a prohibition that really exists in the source.
    invented = sorted(proven - set(declared))
    assert not invented, f"CASES name messages no guard raises: {invented}"
