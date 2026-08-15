# tests/test_subagent_placement_parity.py — the CHECKLIST item `subagent_isolation`,
# asserted across PLACEMENTS.
#
# The repo had thorough tests for the restricted-subagent secret/control denial and
# thorough tests for the ssh dispatch seam, and not one test where a subagent profile
# met a remote placement. That gap was the defect: the denial lived inside
# `tools/core._repo_read`/`_repo_list`, the native route REPLACES the Home handler,
# and so on an ssh placement a restricted subagent read `.env`, `credentials.json`
# and `secrets/db.txt` out of the remote workspace and listed `secrets/` — while the
# byte-identical local call was refused.
#
# So the matrix is the test: profile × placement × path × tool, with the SAME
# assertion applied to both placements. A guard that is only true on one filesystem
# is not a guard, and the only way to keep saying so is to ask both.
#
# Serial: the ssh half runs a live broker with real I/O threads over a fake wire
# (the same fixture the dispatch-seam file uses).
from __future__ import annotations

import pathlib

import pytest

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE, LOCAL_READONLY_SUBAGENT_MODE
from ouroboros.tools.registry import ToolContext, ToolRegistry

from tests.test_registry_remote_dispatch import _target_repo, wire_ssh_registry

pytestmark = pytest.mark.serial

# The two restricted profiles `is_restricted_subagent_profile` names, plus the
# unrestricted control that must keep seeing everything.
_RESTRICTED = [
    pytest.param(LOCAL_READONLY_SUBAGENT_MODE, "", id="local_readonly_subagent"),
    pytest.param(ACTING_SUBAGENT_MODE, "self_worktree", id="acting_subagent"),
]

_SECRET_PATHS = [".env", "secrets/db.txt", "credentials.json"]
_ORDINARY_PATH = "hello.txt"

_READ_BLOCK = "⚠️ REPO_READ_BLOCKED: this subagent cannot read repo secret or control files."
_LIST_BLOCK = "⚠️ REPO_LIST_BLOCKED: this subagent cannot list repo secret or control paths."
_SEARCH_BLOCK = "⚠️ SEARCH_BLOCKED: this subagent cannot access repo secret or control paths."
_QUERY_BLOCK = "⚠️ QUERY_BLOCKED: this subagent cannot access repo secret or control paths."


def seed_secrets(root: pathlib.Path) -> None:
    (root / ".env").write_text("SECRET=remote-env\n", encoding="utf-8")
    (root / "credentials.json").write_text('{"token": "remote-cred"}\n', encoding="utf-8")
    (root / "secrets").mkdir(exist_ok=True)
    (root / "secrets" / "db.txt").write_text("remote-db-password\n", encoding="utf-8")


def _constrain(registry: ToolRegistry, mode: str, surface: str) -> None:
    registry._ctx.task_constraint = (
        TaskConstraint(mode=mode, surface=surface) if mode else None
    )


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    """A registry on an SSH placement, wired exactly as the dispatch-seam file wires
    it — built from the shared harness rather than by importing its fixture, so the
    two modules cannot drift and neither shadows the other's name."""
    yield from wire_ssh_registry(tmp_path, monkeypatch)


@pytest.fixture()
def local(tmp_path):
    """A registry on a LOCAL external workspace holding the same secrets."""
    root = _target_repo(tmp_path)
    seed_secrets(root)
    drive = tmp_path / "data"
    repo = tmp_path / "repo"
    for path in (drive, repo):
        path.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(
        repo_dir=repo,
        drive_root=drive,
        task_id="local-parity-task",
        workspace_root=str(root),
        workspace_mode="external",
    )
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry.set_context(ctx)

    class _Local:
        def __init__(self):
            self.registry = registry
            self.root = root

    return _Local()


def _placements(local, wired):  # noqa: ANN001 — fixtures
    return {"local": local, "ssh": wired}


# ── the refusals: same decision, same bytes, either placement ────────────────


@pytest.mark.parametrize("mode,surface", _RESTRICTED)
@pytest.mark.parametrize("path", _SECRET_PATHS)
def test_a_restricted_subagent_read_of_a_secret_is_refused_on_both_placements(
    local, wired, mode, surface, path,
):
    for name, place in _placements(local, wired).items():
        seed_secrets(place.root)
        _constrain(place.registry, mode, surface)
        result = place.registry.execute(
            "read_file", {"root": "active_workspace", "path": path}
        )
        assert result == _READ_BLOCK, (name, path, result)
        # The refusal is the WHOLE answer: no byte of the file rides along with it.
        assert "remote-db-password" not in result and "remote-cred" not in result


@pytest.mark.parametrize("mode,surface", _RESTRICTED)
def test_a_restricted_subagent_listing_of_a_secret_dir_is_refused_on_both_placements(
    local, wired, mode, surface,
):
    for name, place in _placements(local, wired).items():
        seed_secrets(place.root)
        _constrain(place.registry, mode, surface)
        result = place.registry.execute(
            "list_files", {"root": "active_workspace", "path": "secrets"}
        )
        assert result == _LIST_BLOCK, (name, result)


@pytest.mark.parametrize("mode,surface", _RESTRICTED)
@pytest.mark.parametrize("path", _SECRET_PATHS)
def test_a_restricted_subagent_search_scoped_to_a_secret_is_refused_on_both_placements(
    local, wired, mode, surface, path,
):
    for name, place in _placements(local, wired).items():
        seed_secrets(place.root)
        _constrain(place.registry, mode, surface)
        result = place.registry.execute(
            "search_code", {"root": "active_workspace", "query": "password", "path": path}
        )
        assert result == _SEARCH_BLOCK, (name, path, result)


@pytest.mark.parametrize("mode,surface", _RESTRICTED)
@pytest.mark.parametrize("path", _SECRET_PATHS)
def test_a_restricted_subagent_query_scoped_to_a_secret_is_refused_on_both_placements(
    local, wired, mode, surface, path,
):
    for name, place in _placements(local, wired).items():
        seed_secrets(place.root)
        _constrain(place.registry, mode, surface)
        result = place.registry.execute(
            "query_code", {"op": "symbols", "root": "active_workspace", "path": path}
        )
        assert result == _QUERY_BLOCK, (name, path, result)


# ── the ordinary file: unchanged, so the guard is a guard and not a wall ──────


@pytest.mark.parametrize("mode,surface", [*_RESTRICTED, pytest.param("", "", id="unrestricted")])
def test_an_ordinary_file_is_read_on_both_placements_under_every_profile(
    local, wired, mode, surface,
):
    for name, place in _placements(local, wired).items():
        seed_secrets(place.root)
        _constrain(place.registry, mode, surface)
        result = place.registry.execute(
            "read_file", {"root": "active_workspace", "path": _ORDINARY_PATH}
        )
        assert "hello target" in result, (name, result)
        assert "BLOCKED" not in result


def test_an_unrestricted_task_still_reads_its_own_workspace_secrets_on_both_placements(
    local, wired,
):
    """The denial is about the READER, not about the file: a full task keeps access."""
    for name, place in _placements(local, wired).items():
        seed_secrets(place.root)
        _constrain(place.registry, "", "")
        result = place.registry.execute(
            "read_file", {"root": "active_workspace", "path": "secrets/db.txt"}
        )
        assert "remote-db-password" in result, (name, result)


# ── the listing CONTENTS, not just the listing target ────────────────────────


@pytest.mark.parametrize("mode,surface", _RESTRICTED)
def test_a_listing_of_an_ordinary_dir_hides_the_same_entries_on_both_placements(
    local, wired, mode, surface,
):
    rendered = {}
    for name, place in _placements(local, wired).items():
        seed_secrets(place.root)
        _constrain(place.registry, mode, surface)
        rendered[name] = place.registry.execute(
            "list_files", {"root": "active_workspace", "path": "."}
        )
    for name, text in rendered.items():
        # The LISTING is the JSON array; a disclosure paragraph may follow it and
        # legitimately names what was withheld, so the array is what is compared.
        listing = text.partition("\n\n")[0]
        assert ".env" not in listing, (name, text)
        assert "credentials.json" not in listing, (name, text)
        assert "secrets/" not in listing, (name, text)
        assert ".git/" not in listing, (name, text)
        assert "hello.txt" in listing, (name, text)
        # The omission is DISCLOSED, never a silent shortening. The two placements
        # word it differently on purpose: on ssh the export boundary declined three of
        # these AT THE SOURCE and says so under its own marker, and the subagent
        # filter then declines the rest — two authorities, two exact counts, rather
        # than one filter rewriting the other's disclosure into a single number.
        assert (
            "secret/control entries hidden from this subagent" in text
            or "LIST_POLICY_FILTERED" in text
        ), (name, text)
    assert "secret/control entries hidden from this subagent" in rendered["local"]
    assert "LIST_POLICY_FILTERED" in rendered["ssh"]


# ── the walk-level filter: a subagent's remote query never READS the secret ──


@pytest.mark.parametrize("mode,surface", _RESTRICTED)
def test_a_restricted_subagent_search_of_the_tree_never_returns_secret_content(
    local, wired, mode, surface,
):
    for name, place in _placements(local, wired).items():
        seed_secrets(place.root)
        _constrain(place.registry, mode, surface)
        result = place.registry.execute(
            "search_code",
            {"root": "active_workspace", "query": "remote-db-password", "path": "."},
        )
        # The match LINE is what carries the bytes; its absence is what matters.
        assert "secrets/db.txt:" not in result, (name, result)
        assert "No matches found" in result, (name, result)


def test_an_unrestricted_task_search_still_finds_its_own_secrets_on_both_placements(
    local, wired,
):
    for name, place in _placements(local, wired).items():
        seed_secrets(place.root)
        _constrain(place.registry, "", "")
        result = place.registry.execute(
            "search_code",
            {"root": "active_workspace", "query": "remote-db-password", "path": "."},
        )
        assert "secrets/db.txt" in result, (name, result)


# ── the predicate is the pipeline's, and it is pure ──────────────────────────


def test_the_denial_is_decided_before_the_target_is_ever_asked(wired):
    """A refused call makes no prepare: the decision precedes the placement read."""
    _constrain(wired.registry, LOCAL_READONLY_SUBAGENT_MODE, "")
    seed_secrets(wired.root)
    before = len(wired.transport.calls)
    result = wired.registry.execute("read_file", {"root": "active_workspace", "path": ".env"})
    assert result == _READ_BLOCK
    assert wired.transport.calls[before:] == []


def test_a_home_native_root_keeps_its_own_handler_refusal(wired):
    """The lift covers `active_workspace` only; the Home-native roots are declared
    Home-only and answer with their own (already-correct) guard."""
    _constrain(wired.registry, LOCAL_READONLY_SUBAGENT_MODE, "")
    result = wired.registry.execute(
        "read_file", {"root": "runtime_data", "path": "settings.json"}
    )
    assert "BLOCKED" in result, result


# ── the class's remaining halves, closed in one pass and asserted per placement ──
#
# Everything below is the same shape as the file's original matrix — one assertion,
# both placements — applied to the policies the handler-policy registry had recorded
# as ESCAPING. The registry now says `travels` for each of them; these are the
# behavioural claims behind those words.


_MARKER_NAMES = ["db_password.conf", "api_key.yaml", "x.env"]
_MARKER_SENTINEL = "MARKER-NAMED-SECRET"


def seed_marker_named(root: pathlib.Path) -> None:
    """Credential-MARKER filenames: the pattern half of the same denial.

    Home catches these with a delimited-marker regex scoped to config-ish suffixes.
    They are not enumerable as tokens, so until the document grew a rule field for the
    pattern the target's export policy did not know about them at all.
    """

    for name in _MARKER_NAMES:
        (root / name).write_text(f"value={_MARKER_SENTINEL}\n", encoding="utf-8")


@pytest.mark.parametrize("mode,surface", _RESTRICTED)
@pytest.mark.parametrize("name", _MARKER_NAMES)
def test_a_marker_named_credential_file_is_refused_on_both_placements(
    local, wired, mode, surface, name,
):
    for place_name, place in _placements(local, wired).items():
        seed_marker_named(place.root)
        _constrain(place.registry, mode, surface)
        result = place.registry.execute(
            "read_file", {"root": "active_workspace", "path": name}
        )
        assert result == _READ_BLOCK, (place_name, name, result)
        assert _MARKER_SENTINEL not in result


@pytest.mark.parametrize("mode,surface", _RESTRICTED)
def test_a_search_walk_never_reads_a_marker_named_file_on_either_placement(
    local, wired, mode, surface,
):
    """The WALK is where the pipeline's spelling rule cannot help.

    A walk names no path, so `subagent_secret_path_refusal` has nothing to judge — the
    only thing standing between a restricted subagent and the CONTENTS of
    `db_password.conf` on the target is the export policy, which had no field for a
    marker pattern. It has one now (`marker_scoped_suffixes`), so the source declines
    to open the file rather than shipping the matching line.
    """

    for place_name, place in _placements(local, wired).items():
        seed_marker_named(place.root)
        _constrain(place.registry, mode, surface)
        result = place.registry.execute(
            "search_code",
            {"root": "active_workspace", "query": _MARKER_SENTINEL, "path": "."},
        )
        # The match LINE is what carries the bytes, and its absence is the assertion.
        # Neither the echoed query nor the D7 disclosure list (which NAMES what the
        # policy declined, on purpose) counts as a leak, so the assertion is on the
        # `active_workspace:<file>:<line>:` match prefix the walk emits.
        for name in _MARKER_NAMES:
            assert f"active_workspace:{name}:" not in result, (place_name, name, result)
        assert "No matches found" in result or "Found 0 " in result, (place_name, result)


def test_an_unrestricted_task_still_greps_a_marker_named_file_on_both_placements(
    local, wired,
):
    """The tightening belongs to the READER. A full task's own config stays visible."""

    for place_name, place in _placements(local, wired).items():
        seed_marker_named(place.root)
        _constrain(place.registry, "", "")
        result = place.registry.execute(
            "search_code",
            {"root": "active_workspace", "query": _MARKER_SENTINEL, "path": "."},
        )
        assert "db_password.conf" in result, (place_name, result)


@pytest.mark.parametrize("mode,surface", _RESTRICTED)
def test_a_hardlink_alias_of_a_secret_is_refused_on_both_placements(
    local, wired, mode, surface,
):
    """An ALIAS is not a spelling, and the target holds the filesystem.

    `notes_copy.txt` hardlinked onto `.env` clears every rule the document can state.
    Home refuses it with a `samefile` probe; the pipeline cannot (it holds no
    filesystem), and that limitation was recorded as "Home-only" — which was then read
    as if the REMOTE route were also excused. The target owns the workspace, so the
    probe happens there, against the same document.
    """

    for place_name, place in _placements(local, wired).items():
        seed_secrets(place.root)
        alias = place.root / "notes_copy.txt"
        if alias.exists():
            alias.unlink()
        alias.hardlink_to(place.root / ".env")
        _constrain(place.registry, mode, surface)
        result = place.registry.execute(
            "read_file", {"root": "active_workspace", "path": "notes_copy.txt"}
        )
        assert "remote-env" not in result, (place_name, result)
        assert "BLOCKED" in result or "EXCLUDED" in result, (place_name, result)


def test_an_arbitrary_run_script_interpreter_is_refused_on_both_placements(local, wired):
    """The interpreter allowlist, which the target took verbatim into its argv.

    `execute_inline_script` reads `args["interpreter"]` and puts it at argv[0] with no
    check of its own, so the allowlist living in the Home handler body meant a remote
    task could name any executable on the target. The rule is judged on the RAW
    argument before prepare now, so the two placements return the same sentence and the
    remote one never reaches the wire.
    """

    expected = (
        "⚠️ RUN_SCRIPT_BLOCKED: interpreter must be one of "
        "['bash', 'node', 'python', 'python.exe', 'python3', 'python3.exe', 'ruby', 'sh']."
    )
    for place_name, place in _placements(local, wired).items():
        _constrain(place.registry, "", "")
        before = len(getattr(place, "transport", None).calls) if hasattr(place, "transport") else 0
        result = place.registry.execute(
            "run_script", {"script": "print(1)", "interpreter": "/bin/definitely-not-allowed"}
        )
        assert result == expected, (place_name, result)
        if hasattr(place, "transport"):
            assert place.transport.calls[before:] == [], place.transport.calls


@pytest.mark.parametrize(
    "scratch,why",
    [
        pytest.param(["../top_level.txt"], "escapes the command cwd", id="outside_cwd"),
        pytest.param(["nested"], "is a directory", id="directory"),
    ],
)
def test_an_unsafe_declared_scratch_is_refused_on_both_placements(
    local, wired, scratch, why,
):
    """Two of the four scratch rules ran only on Home.

    The target's prepare has always had the git-worktree probe and the git-tracked
    check; it confined a declared path to the workspace ROOT rather than to the COMMAND
    CWD, and said nothing about a directory. Both matter for the same reason the tracked
    check does — scratch is EXCLUDED from the workspace patch, so each of them is a way
    real work could be kept out of the deliverable.

    The two placements do not share a refusal STRING here (a target-side prepare failure
    is a typed remote diagnostic, by design), so what is asserted is the decision: the
    call is refused and the command does not run.
    """

    for place_name, place in _placements(local, wired).items():
        _constrain(place.registry, "", "")
        (place.root / "sub").mkdir(exist_ok=True)
        (place.root / "sub" / "nested").mkdir(exist_ok=True)
        result = place.registry.execute(
            "run_command",
            {"cmd": ["echo", "ran-anyway"], "cwd": "sub", "scratch": scratch},
        )
        assert "ran-anyway" not in result, (place_name, why, result)
        assert "⚠️" in result, (place_name, why, result)
