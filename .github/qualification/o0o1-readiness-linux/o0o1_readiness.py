"""O0 -> O1 published-range readiness driver (test equipment of task 7c342fa3; not product).

One installation root at the published O0; the official target O1 is what the root's managed remote serves. The
frozen Unified Update implementation's own preparation runs on it, unstubbed: ``update_prepare.admit`` and
``update_prepare.prepare`` with ``update_checks.workspace_checks``, the API's own sequence for
``{"strategy": "prepare"}``. It runs on the interpreter and packages an installation chose, and a create-only receipt
records what it did.

    <chosen interpreter> o0o1_readiness.py run BIND_JSON OUT_DIR

The order is the claim. Each step refuses before the next, and no product module is loaded before step 4:
  1. The bind (schema checked) and OUT_DIR (create-only).
  2. Standard library only:
     - the interpreter: real path, prefix, flags, no venv, and the evidence that measured it serving;
     - its packages against the root's own runtime lock;
     - the implementation checkout: tree, clean including ignored files, the bound file digests, no armed adoption;
     - the root: its kind's geometry, managed meta, no armed adoption, a fresh data root;
     - the official target the root's remote serves;
     - and the root's footprint.
  3. The configuration. The process environment must be exactly the bind's (asserted), the implementation goes first
     on sys.path, and no product module is loaded yet (asserted).
  4. The implementation's own modules, each from the implementation with the bound bytes, and the configuration they
     resolve, read back: launch roots, work root, channel, managed target.
  5. ``git_ops.init``, as the server's boot (``_bootstrap_supervisor_repo``) binds it.
  6. ``admit`` then ``prepare``, synchronously. Nothing of the product is replaced, wrapped or patched.
  7. The receipt: the record, the status projection, every run folder the checks name (copied, sha256), the loaded
     modules, the footprint after, timing, the outcome (READY, or a named NONREADY with its cost lower bound) and the
     verdict on the bound expectation. Every named run is read against its role (``bound_run``). A READY rests on W'
     always and on W when it excused W' failures: one of those missing, incomplete or bound elsewhere is not
     established (UNKNOWN, unless a check measured a FAIL), never PASS. T's attribution run belongs to blocking
     failures only and is never demanded. ``green`` (a PASS with no failure in any run read) stays apart from the
     outcome and the verdict. What a run executed is read from its JUnit reports (unique node ids, so unique (file,
     node) pairs), never from its target lines: a full tier names one suite directory. Each run's deadline is the one
     its run.json records (``deadline_seconds``), and the runs share the checks' one recorded budget
     (``budget_seconds``). A READY on a run the record or run.json calls incomplete is a contradiction (FAIL).

Exit: 0 PASS, 1 FAIL (an expectation measured not to hold), 3 UNKNOWN (not established), 2 refused before any effect.
"""
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ElementTree
from pathlib import Path

SCHEMA = "o0o1-readiness-bind/1"
PRODUCT_TOPS = ("ouroboros", "supervisor")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
FLAGS = ("isolated", "ignore_environment", "no_site", "no_user_site", "dont_write_bytecode")
ROLES = ("new", "baseline", "target_run")
LANES = ["parallel", "serial"]   # update_checks._lanes, in their order
# A lane's keys that run.json and the record carry only when set (update_checks.run_tests, 015ce40b): ``containment``
# (its tree not proven gone) and ``no_failing_case`` (exit 1 though none of its cases failed). Either leaves the run
# incomplete; kept and compared as recorded, never assumed absent.
LANE_OPTIONAL = ("containment", "no_failing_case")
SELECTOR = "ouroboros/test_selection.py"   # where the official target declares its floor (update_checks._SELECTOR)
SYSTEM_ADDED = ("__CF_USER_TEXT_ENCODING",)
# The margins of a deadline judgment; the deadline itself is always the run's recorded ``deadline_seconds``. TOLERANCE
# covers the record's 0.1 s rounding and the moments between two monotonic reads: a run cut at its deadline (exit
# 124) ends no earlier than this before it, and a later run's deadline is no more than this above what the one before
# left of the budget. GRACE is what may follow the deadline before run.json is written: the tracked owner reaping the
# lane's tree, then the reports read (the Git owner's own bound, update_prepare._GIT_TIMEOUT_SEC, is of that order).
TOLERANCE = 5.0
GRACE = 300.0
# The Git owner's own bound (update_prepare._GIT_TIMEOUT_SEC, 015ce40b). run_tests gives the materialization the lesser
# of it and what remains, and records that bound (``materialization.timeout``): out of time under it, the run ends
# unstarted before a longer deadline.
GIT_BOUND = 300.0


class Refused(Exception):
    """Refused before any effect of the product: (stage, text)."""

    def __init__(self, stage, text):
        super().__init__("%s: %s" % (stage, text))
        self.stage, self.text = stage, text


def need(condition, stage, text):
    if not condition:
        raise Refused(stage, text)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def real(path):
    return os.path.realpath(str(path))


def write_new(path, text):
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(text)


# ------------------------------------------------------------------ the bind
def load_bind(path):
    raw = Path(path).read_bytes()
    try:
        bind = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise Refused("bind", "unreadable JSON (%s)" % exc)
    need(isinstance(bind, dict) and bind.get("schema") == SCHEMA, "bind", "not an %s document" % SCHEMA)
    for key in ("label", "kind", "range", "git", "implementation", "interpreter", "root", "env", "expect"):
        need(key in bind, "bind", "no %r" % key)
    need(bind["kind"] in ("clean", "evolved"), "bind", "kind is clean or evolved")
    need(all(HEX40.match(str(bind["range"].get(k, ""))) for k in ("o0", "o1")), "bind", "range o0/o1 are full ids")
    need(HEX40.match(str(bind["implementation"].get("tree", ""))), "bind", "implementation tree is a full id")
    need(isinstance(bind["implementation"].get("sha256"), dict) and bind["implementation"]["sha256"], "bind",
         "implementation sha256 names the bound files")
    need(isinstance(bind["env"], dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in bind["env"].items()),
         "bind", "env maps names to strings")
    for key in ("app_root", "repo", "data", "home", "official_url", "branch", "meta"):
        need(key in bind["root"], "bind", "root has no %r" % key)
    for path in (bind["git"], bind["implementation"]["dir"], bind["interpreter"]["path"], bind["root"]["repo"]):
        need(os.path.isabs(str(path)), "bind", "%s is not absolute" % path)
    expect = bind["expect"]
    need(expect.get("tier") in ("selected", "full"), "bind", "expect.tier is selected or full")
    need(expect.get("target") == bind["range"]["o1"], "bind", "the expected target is the bound O1")
    need(isinstance(expect.get("floor"), dict) and expect["floor"], "bind", "expect.floor is the target's floor")
    if bind["kind"] == "evolved":
        for key in ("local_test", "local_test_sha256", "cutover", "marker", "notes", "untracked"):
            need(key in bind["root"].get("evolved", {}), "bind", "root.evolved has no %r" % key)
    return bind, raw


def git(bind, repo, *args, check=True, timeout=300):
    """The bound Git on REPO with the bound environment (its HOME's insteadOf included), optional locks off. A checked
    read that exits non-zero or writes any warning is refused: Git omits an untracked directory it cannot open and
    still exits 0, so a listing that warned is never complete."""
    env = dict(bind["env"])
    env["GIT_OPTIONAL_LOCKS"] = "0"
    done = subprocess.run([bind["git"], "-C", str(repo), *args], capture_output=True, env=env, timeout=timeout)
    if check and (done.returncode != 0 or done.stderr.strip()):
        raise Refused("git", "git %s in %s exited %d: %s" % (" ".join(args), repo, done.returncode,
                                                           done.stderr.decode("utf-8", "replace").strip()[:300]))
    return done


def out(bind, repo, *args):
    return git(bind, repo, *args).stdout.decode("utf-8").strip()


def nul_list(data):
    return sorted(item.decode("utf-8", "surrogateescape") for item in data.split(b"\0") if item)


# ------------------------------------------------------------------ step 2: standard library checks
def check_interpreter(bind):
    want = bind["interpreter"]
    facts = {"executable": sys.executable, "real": real(sys.executable), "prefix": sys.prefix,
             "base_prefix": sys.base_prefix, "version": sys.version,
             "flags": {name: getattr(sys.flags, name, None) for name in FLAGS}}
    need(facts["real"] == real(want["path"]), "interpreter", "running %s, not the bound %s" % (facts["real"], want["path"]))
    need(real(sys.prefix) == real(want["prefix"]), "interpreter", "prefix %s, not %s" % (sys.prefix, want["prefix"]))
    need(sys.prefix == sys.base_prefix, "interpreter", "a virtual environment (%s over %s)" % (sys.prefix, sys.base_prefix))
    need(not Path(sys.prefix, "pyvenv.cfg").exists(), "interpreter", "a pyvenv.cfg beside the prefix")
    for name, value in (want.get("flags") or {}).items():
        need(facts["flags"].get(name) == value, "interpreter", "sys.flags.%s is %r, not %r" % (
            name, facts["flags"].get(name), value))
    evidence = want.get("evidence")
    if evidence:   # the step that read this interpreter as a serving server's argv[0] from the kernel
        folder = Path(evidence["dir"])
        files = {name: folder / name for name in ("verdict.txt", "interpreter.txt", "localfn.json")}
        need(all(path.is_file() for path in files.values()), "interpreter", "evidence %s is incomplete" % folder)
        need(files["verdict.txt"].read_text(encoding="utf-8").startswith("verdict=PASS"), "interpreter",
             "the measuring step %s did not pass" % folder)
        chosen = files["interpreter.txt"].read_text(encoding="utf-8").strip()
        need(real(chosen) == facts["real"], "interpreter", "the measured serving interpreter is %s" % chosen)
        fact = json.loads(files["localfn.json"].read_text(encoding="utf-8"))
        need(real(fact.get("prefix", "")) == real(sys.prefix), "interpreter",
             "the measured serving prefix is %s" % fact.get("prefix"))
        facts["evidence"] = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in files.items()}
    return facts


def _version(text):
    return tuple(int(part) for part in re.findall(r"\d+", text)[:4])


def marker_true(marker, values):
    """A runtime-lock environment marker: comparisons joined by ``and``/``or``, no parentheses (else ValueError)."""
    if "(" in marker or ")" in marker:
        raise ValueError("a parenthesized marker: %s" % marker)
    def compare(term):
        match = re.fullmatch(r"\s*([a-z_]+)\s*(==|!=|<=|>=|<|>)\s*(['\"])([^'\"]*)\3\s*", term)
        if not match or match.group(1) not in values:
            raise ValueError("an unread marker term: %s" % term)
        have, op, want = values[match.group(1)], match.group(2), match.group(4)
        if match.group(1).endswith("version"):
            have, want = _version(have), _version(want)
        return {"==": have == want, "!=": have != want, "<": have < want, "<=": have <= want,
                ">": have > want, ">=": have >= want}[op]
    return any(all(compare(term) for term in re.split(r"\s+and\s+", part)) for part in re.split(r"\s+or\s+", marker))


def check_packages(bind):
    """Every pin of the root's own runtime lock that applies here is installed at that version, as this interpreter
    finds its distributions (standard library ``importlib.metadata``, before any product import)."""
    import importlib.metadata as metadata

    lock = Path(bind["root"]["repo"]) / "requirements-runtime.lock"
    text = lock.read_text(encoding="utf-8")
    values = {"platform_python_implementation": platform.python_implementation(), "sys_platform": sys.platform,
              "python_full_version": platform.python_version(), "python_version": "%d.%d" % sys.version_info[:2],
              "platform_machine": platform.machine(), "implementation_name": sys.implementation.name,
              "os_name": os.name, "platform_system": platform.system()}
    applicable, problems = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        spec, _, marker = line.partition(";")
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", spec.strip())
        need(match, "packages", "an unread lock line: %r" % line)
        try:
            applies = not marker.strip() or marker_true(marker.strip(), values)
        except ValueError as exc:
            raise Refused("packages", str(exc))
        if not applies:
            continue
        name, version = match.groups()
        applicable.append(name)
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append("%s missing" % name)
            continue
        if have != version:
            problems.append("%s %s, not %s" % (name, have, version))
    if bind["expect"].get("lock_sha256"):
        need(sha256_bytes(text.encode("utf-8")) == bind["expect"]["lock_sha256"], "packages",
             "the root's lock is not the bound one")
    need(not problems, "packages", "this interpreter's packages are not the root's lock: " + "; ".join(problems[:12]))
    return {"lock": str(lock), "lock_sha256": sha256_file(lock), "applicable_pins": len(applicable),
            "markers": values, "installed": "every applicable pin at its version"}


def check_implementation(bind):
    impl = bind["implementation"]
    folder = Path(impl["dir"])
    tree = out(bind, folder, "rev-parse", "--verify", "HEAD^{tree}")
    need(tree == impl["tree"], "implementation", "tree %s, not the bound %s" % (tree, impl["tree"]))
    status = git(bind, folder, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored=matching").stdout
    need(status == b"", "implementation", "the checkout is not clean, ignored files included: %r" % status[:200])
    digests = {}
    for rel, want in sorted(impl["sha256"].items()):
        digests[rel] = sha256_file(folder / rel)
        need(digests[rel] == want, "implementation", "%s is %s, not %s" % (rel, digests[rel], want))
    gitdir = Path(out(bind, folder, "rev-parse", "--absolute-git-dir"))
    need(not (gitdir / "ouroboros-body-adoption").exists(), "implementation", "an armed body adoption")
    return {"dir": str(folder), "real": real(folder), "tree": tree, "sha256": digests,
            "head": out(bind, folder, "rev-parse", "HEAD")}


def check_root(bind):
    root, kind, (o0, o1) = bind["root"], bind["kind"], (bind["range"]["o0"], bind["range"]["o1"])
    repo = Path(root["repo"])
    gitdir = Path(out(bind, repo, "rev-parse", "--absolute-git-dir"))
    need(not (gitdir / "ouroboros-body-adoption").exists(), "root", "an armed body adoption")
    meta_path = gitdir / "ouroboros-managed.json"
    need(meta_path.is_file(), "root", "no managed repo meta (%s)" % meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    need(meta == root["meta"], "root", "the managed meta is not the bound one")
    need(out(bind, repo, "remote").split() == ["managed"], "root", "remotes other than managed")
    # The configured URL itself (``remote get-url`` would expand the home's insteadOf to the mirror).
    need(out(bind, repo, "config", "--get", "remote.managed.url") == root["official_url"], "root",
         "managed is not the official URL")
    branch = git(bind, repo, "symbolic-ref", "-q", "HEAD", check=False).stdout.decode().strip()
    need(branch == "refs/heads/" + root["branch"], "root", "HEAD is %r, not the %s branch" % (branch, root["branch"]))
    data = Path(root["data"])
    need(not data.exists() or not any(data.iterdir()), "root", "the data root %s is not fresh" % data)
    head = out(bind, repo, "rev-parse", "--verify", "HEAD^{commit}")
    served = out(bind, repo, "ls-remote", root["official_url"], "refs/heads/" + root["branch"])
    need(served.split() == [o1, "refs/heads/" + root["branch"]], "root", "the official remote serves %r, not O1" % served)
    facts = {"head": head, "branch": branch, "meta_sha256": sha256_file(meta_path), "served": served}
    if kind == "clean":
        need(head == o0, "root", "HEAD %s is not O0 %s" % (head, o0))
        status = git(bind, repo, "status", "--porcelain=v1", "-z", "--untracked-files=all").stdout
        need(status == b"", "root", "the clean root has local changes: %r" % status[:200])
        return facts
    evolved = root["evolved"]
    parents = out(bind, repo, "rev-list", "--parents", "-n", "1", "HEAD").split()
    need(parents == [head, o0], "root", "H is not one commit on O0 (%s)" % parents)
    changed = nul_list(git(bind, repo, "diff", "--name-only", "-z", o0, "HEAD").stdout)
    want = sorted([evolved["local_test"], evolved["cutover"], evolved["notes"]])
    need(changed == want, "root", "H changes %s, not %s" % (changed, want))
    cutover = git(bind, repo, "show", "HEAD:" + evolved["cutover"]).stdout.decode("utf-8")
    need(evolved["marker"] in cutover, "root", "H's %s lacks the local cutover marker" % evolved["cutover"])
    probe = git(bind, repo, "show", "HEAD:" + evolved["local_test"]).stdout
    need(sha256_bytes(probe) == evolved["local_test_sha256"], "root", "H's local test is not the bound bytes")
    staged = nul_list(git(bind, repo, "diff", "--cached", "--name-only", "-z").stdout)
    unstaged = nul_list(git(bind, repo, "diff", "--name-only", "-z").stdout)
    untracked = nul_list(git(bind, repo, "ls-files", "--others", "--exclude-standard", "-z").stdout)
    need(staged == [evolved["notes"]], "root", "I is %s, not %s" % (staged, evolved["notes"]))
    need(unstaged == [evolved["notes"]], "root", "W is %s, not %s" % (unstaged, evolved["notes"]))
    need(untracked == sorted(evolved["untracked"]), "root", "U is %s" % untracked)
    facts.update(local_commit_parents=parents, h_changes=changed, staged=staged, unstaged=unstaged, untracked=untracked)
    return facts


def footprint(bind):
    """The serving checkout's H/I/W/U witnesses: HEAD, tree, branch, the index listing, and every path that differs from
    HEAD or is untracked, hashed from disk. Any read that fails makes it ``None`` (a preservation claim is then unknown)."""
    repo = Path(bind["root"]["repo"])
    try:
        record = {"head": out(bind, repo, "rev-parse", "--verify", "HEAD^{commit}"),
                  "tree": out(bind, repo, "rev-parse", "--verify", "HEAD^{tree}"),
                  "branch": git(bind, repo, "symbolic-ref", "-q", "HEAD", check=False).stdout.decode().strip(),
                  "index_sha256": sha256_bytes(git(bind, repo, "ls-files", "-s", "-z").stdout), "files": {}}
        paths = set(nul_list(git(bind, repo, "diff", "--name-only", "-z", "HEAD").stdout))
        paths |= set(nul_list(git(bind, repo, "ls-files", "--others", "--exclude-standard", "-z").stdout))
        for rel in sorted(paths):
            path = repo / rel
            record["files"][rel] = ({"sha256": sha256_file(path), "bytes": path.stat().st_size,
                                     "mode": oct(path.stat().st_mode & 0o777)} if path.is_file() else "absent")
        return record
    except (Refused, OSError) as exc:
        return {"unread": str(exc)}


# ------------------------------------------------------------------ steps 3-5: configuration, import, boot binding
def bind_configuration(bind):
    wanted = dict(bind["env"])
    # macOS CoreFoundation adds this one at process start when it is absent: recorded, removed, never configuration.
    system = {name: os.environ.pop(name) for name in SYSTEM_ADDED if name in os.environ and name not in wanted}
    extra = sorted(set(os.environ) - set(wanted))
    differs = sorted(name for name in wanted if os.environ.get(name) != wanted[name])
    need(not extra and not differs, "configuration",
         "the process did not start with exactly the bound environment (extra %s, different %s)" % (extra, differs))
    loaded = sorted(name for name in sys.modules if name.split(".")[0] in PRODUCT_TOPS)
    need(not loaded, "configuration", "product modules already loaded: %s" % loaded)
    sys.path.insert(0, bind["implementation"]["dir"])
    return {"environment": wanted, "removed_system_variables": system, "sys_path": list(sys.path)}


def loaded_modules(impl_dir):
    inside, foreign = {}, []
    base = real(impl_dir) + os.sep
    for name, module in sorted(sys.modules.items()):
        if name.split(".")[0] not in PRODUCT_TOPS:
            continue
        where = getattr(module, "__file__", None)
        path = real(where) if where else ""
        inside[name] = path
        if not path.startswith(base):
            foreign.append("%s (%s)" % (name, path or "no file"))
    return inside, foreign


def import_implementation(bind):
    """The package first, as every entry (server.py included) reaches it: its package-init hooks run on the bound
    environment before any other product module."""
    import importlib

    names = ("ouroboros", "ouroboros.config", "ouroboros.launch_roots", "ouroboros.update_channels",
             "ouroboros.test_selection", "supervisor.git_ops", "supervisor.update_prepare", "supervisor.update_checks")
    return {name: importlib.import_module(name) for name in names}


def verify_loaded(bind, modules):
    impl = bind["implementation"]
    inside, foreign = loaded_modules(impl["dir"])
    need(not foreign, "import", "product modules from outside the implementation: %s" % foreign[:8])
    for rel, want in impl["sha256"].items():
        name = rel[:-3].replace("/", ".") if rel.endswith(".py") else ""
        if name.endswith(".__init__"):
            name = name[:-len(".__init__")]
        if name in sys.modules:
            need(inside.get(name) == real(Path(impl["dir"]) / rel), "import", "%s was loaded from %s" % (name, inside.get(name)))
            need(sha256_file(inside[name]) == want, "import", "%s changed since step 2" % rel)
    env, roots = bind["env"], modules["ouroboros.launch_roots"].configured_roots()
    read = {"app_root": str(roots.app_root), "repo_dir": str(roots.repo_dir), "data_dir": str(roots.data_dir),
            "settings_path": str(roots.settings_path), "pid_file": str(roots.pid_file), "port_file": str(roots.port_file),
            "subagent_worktree_root": str(modules["ouroboros.config"].get_subagent_worktree_root()),
            "update_channel": modules["ouroboros.update_channels"].get_update_channel(),
            "update_branch": modules["ouroboros.update_channels"].get_update_branch(),
            "managed_target": list(modules["supervisor.git_ops"]._managed_update_target()),
            "policy_digest": modules["ouroboros.test_selection"].policy_digest()}
    # The checks' one budget as the implementation resolves it in this environment (its config default, or an explicit
    # OUROBOROS_PREFLIGHT_TIMEOUT_SEC); None where the implementation has no such resolver or it raised (kept as
    # checks_budget_error). Read, never assumed here.
    budget = getattr(modules["supervisor.update_checks"], "_budget", None)
    try:
        read["checks_budget_seconds"] = budget() if callable(budget) else None
    except Exception as exc:   # a resolver that fails decides nothing here; the record's own budget is still read
        read["checks_budget_seconds"], read["checks_budget_error"] = None, "%s: %s" % (type(exc).__name__, exc)
    for key, name in (("app_root", "OUROBOROS_APP_ROOT"), ("repo_dir", "OUROBOROS_REPO_DIR"),
                      ("data_dir", "OUROBOROS_DATA_DIR"), ("settings_path", "OUROBOROS_SETTINGS_PATH"),
                      ("pid_file", "OUROBOROS_PID_FILE"), ("port_file", "OUROBOROS_PORT_FILE"),
                      ("subagent_worktree_root", "OUROBOROS_SUBAGENT_WORKTREE_ROOT")):
        need(read[key] == env[name], "configuration", "%s resolves to %s, not %s" % (key, read[key], env[name]))
    need((read["update_channel"], read["update_branch"]) == ("development", bind["root"]["branch"]), "configuration",
         "the channel resolves to %s/%s" % (read["update_channel"], read["update_branch"]))
    need(read["managed_target"] == ["managed", bind["root"]["branch"], "managed/" + bind["root"]["branch"]],
         "configuration", "the managed target resolves to %s" % read["managed_target"])
    return {"modules": inside, "readback": read}


def init_like_boot(bind, git_ops):
    """``server._bootstrap_supervisor_repo``'s binding only: ``_runtime_branch_defaults`` then ``git_ops.init``."""
    branch_dev, branch_stable = "ouroboros", "ouroboros-stable"
    if os.environ.get("OUROBOROS_MANAGED_BY_LAUNCHER") == "1":
        branch_dev, branch_stable = git_ops.managed_branch_defaults(Path(os.environ["OUROBOROS_REPO_DIR"]))
    git_ops.init(repo_dir=Path(os.environ["OUROBOROS_REPO_DIR"]), drive_root=Path(os.environ["OUROBOROS_DATA_DIR"]),
                 remote_url="", branch_dev=branch_dev, branch_stable=branch_stable)
    need(str(git_ops.REPO_DIR) == os.environ["OUROBOROS_REPO_DIR"] and str(git_ops.DRIVE_ROOT) == os.environ[
        "OUROBOROS_DATA_DIR"], "configuration", "git_ops did not bind the root")
    return {"repo_dir": str(git_ops.REPO_DIR), "drive_root": str(git_ops.DRIVE_ROOT), "branch_dev": branch_dev,
            "branch_stable": branch_stable, "remote_url": ""}


# ------------------------------------------------------------------ step 7: what the checks ran
def case_id(case):
    """A ``<testcase>``'s identity as the producer keys it (``update_checks._nodeid``, e9890caf; the same at 015ce40b): the complete pytest
    node id ``file::Class::Nested::name[params]``, never the name or the file alone."""
    path, classname = str(case.get("file") or ""), str(case.get("classname") or "")
    module = path[:-3].replace("/", ".") if path.endswith(".py") else ""
    if not path:   # no file attribute: the dotted module path is all there is
        path, classes = classname.replace(".", "/") + ".py", []
    elif module and classname.startswith(module + "."):
        classes = classname[len(module) + 1:].split(".")
    else:
        classes = []
    return "::".join([path, *classes, str(case.get("name") or "")])


def copy_run(folder, dest):
    """Copy a run folder the checks name (regular files only) and read it: its record, argument file and reports.

    A lane's cases are counted as run.json counts them (``update_checks._cases``): one per node id. pytest's report
    holds two entries for a node that fails in its call and again in its teardown, so ``cases`` counts nodes and
    ``entries`` the ``<testcase>`` elements. Every entry is read: ``by_file`` keeps each one's outcome.

    What ran is the reports' membership, never the target lines: ``executed`` maps each file to its unique node ids
    across both lanes (a node id names its file, so these are unique (file, node) pairs), and ``reports_read`` says
    whether every lane that collected cases left a report that was read. A full tier's one target line (the suite
    directory) and a W or T run's node ids are therefore read the same way as a selected tier's files."""
    files, problems = {}, []
    source = Path(folder)
    if not source.is_dir():
        return {"source": str(source), "files": files, "problems": ["the run folder is gone"], "complete": False}
    dest.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and not path.is_symlink():
            shutil.copyfile(path, dest / path.name)
            files[path.name] = {"sha256": sha256_file(dest / path.name), "bytes": (dest / path.name).stat().st_size}
    try:
        run = json.loads((dest / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"source": str(source), "files": files, "problems": ["run.json unread (%s)" % exc], "complete": False}
    targets, targets_file = run.get("targets") or [], str(run.get("targets_file") or "")
    if targets_file != str(source / "targets.txt"):
        problems.append("targets_file is %r, not this folder's targets.txt" % targets_file)
    elif "targets.txt" not in files:
        problems.append("no targets.txt in the run folder")
    elif (dest / "targets.txt").read_text(encoding="utf-8").splitlines() != targets:
        problems.append("targets.txt is not run.json targets")
    cases, entries, local, reports_read = set(), 0, {}, True
    for lane in run.get("lanes") or []:
        name = lane.get("lane")
        argv = [str(item) for item in lane.get("argv") or []]
        if [item for item in argv if item.startswith("@")] != ["@" + targets_file]:
            problems.append("the %s lane does not read exactly its argument file" % name)
        for stream in ("stdout", "stderr"):
            if "%s.%s.txt" % (name, stream) not in files:
                problems.append("no %s.%s.txt" % (name, stream))
        if lane.get("returncode") == 5 and not lane.get("cases"):
            continue
        try:
            found = list(ElementTree.parse(dest / ("%s.xml" % name)).getroot().iter("testcase"))
        except (OSError, ElementTree.ParseError) as exc:
            reports_read = False
            if lane.get("output") != "complete" and not lane.get("cases"):   # cut or never started: no report to read
                problems.append("the %s lane ended %s (exit %s) and left no report" % (
                    name, lane.get("output"), lane.get("returncode")))
            else:
                problems.append("the %s report is unread (%s)" % (name, exc))
            continue
        nodes = {case_id(case) for case in found}
        if len(nodes) != lane.get("cases"):
            problems.append("the %s report holds %d cases in %d entries, run.json %s" % (
                name, len(nodes), len(found), lane.get("cases")))
        cases |= nodes
        entries += len(found)
        for case in found:
            outcome = ("failed" if case.find("failure") is not None or case.find("error") is not None else
                       "skipped" if case.find("skipped") is not None else "passed")
            local.setdefault(str(case.get("file") or ""), []).append(outcome)
    if reports_read and run.get("lanes") and len(cases) != run.get("cases"):   # run.json counts the lanes' union
        problems.append("the reports hold %d unique cases, run.json %s" % (len(cases), run.get("cases")))
    executed = {}
    for node in cases:
        executed.setdefault(node.split("::", 1)[0], set()).add(node)
    return {"source": str(source), "files": files, "commit": run.get("commit"), "complete_run": run.get("complete"),
            "cases": len(cases), "entries": entries, "seconds": run.get("seconds"),
            "deadline_seconds": run.get("deadline_seconds"), "materialization": run.get("materialization"),
            "deadline": deadline_facts(run),
            "executed": {path: len(nodes) for path, nodes in sorted(executed.items())},
            "executed_files": sorted(executed), "reports_read": reports_read,
            "targets": targets, "record": {key: run.get(key) for key in (
                "complete", "returncode", "cases", "failed", "crashed", "seconds")},
            "lanes": [{key: lane.get(key) for key in ("lane", "returncode", "cases", "seconds", "output", *LANE_OPTIONAL)
                       if key in lane or key not in LANE_OPTIONAL} for lane in run.get("lanes") or []],
            "selection": run.get("selection"), "floor": run.get("floor"), "footprint": run.get("footprint"),
            "local_tests": run.get("local_tests"), "node": run.get("node"), "user_site": run.get("user_site"),
            "by_file": local, "problems": problems, "complete": not problems}


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def deadline_facts(run):
    """A run's end read against the deadline its own run.json records (``deadline_seconds``: what remained of the
    checks' one budget when it began; ``seconds``: from that same start to its end). ``state`` names it; ``contradiction``
    is set when the record contradicts its own deadline: a lane cut at the deadline (exit 124) before it, or a run that
    outlived it by more than GRACE. A materialization out of time under its own recorded bound (GIT_BOUND, below the
    deadline) is not that: the run is incomplete, ended at that bound, and its lanes never started (exit 124, output
    none, no cases). No deadline is assumed when none is recorded (``state`` says so)."""
    limit, took = _number(run.get("deadline_seconds")), _number(run.get("seconds"))
    lanes = run.get("lanes") or []
    cut = [lane.get("lane") for lane in lanes if lane.get("returncode") == 124]
    materialization = run.get("materialization") or {}
    materialized = materialization.get("outcome")
    if materialized in ("timeout", "not_started"):
        cut.insert(0, "materialization (%s)" % materialized)
    facts = {"deadline_seconds": limit, "seconds": took, "cut": cut, "contradiction": None}
    if limit is None or took is None:
        facts["state"] = "not judged: run.json records %s" % ", ".join(
            "no %s" % key for key, value in (("deadline_seconds", limit), ("seconds", took)) if value is None)
        return facts
    bound = _number(materialization.get("timeout"))
    if (materialized == "timeout" and bound is not None and abs(bound - min(GIT_BOUND, limit)) <= 0.05
            and bound - TOLERANCE <= took <= bound + GRACE and run.get("complete") is False
            and [lane.get("lane") for lane in lanes] == LANES
            and all(lane.get("returncode") == 124 and lane.get("output") == "none" and _number(lane.get("cases")) == 0
                    for lane in lanes)
            and took < limit - TOLERANCE):
        facts["state"] = ("incomplete: its materialization ran out of the Git owner's own %.1f s bound at %.1f s, "
                          "before its recorded deadline of %.1f s; no lane started (%s exit 124, output none)" % (
                              bound, took, limit, ", ".join(LANES)))
        return facts
    if cut and took < limit - TOLERANCE:
        facts["contradiction"] = facts["state"] = "cut (%s) at %.1f s, before its recorded deadline of %.1f s" % (
            ", ".join(cut), took, limit)
        return facts
    if took > limit + GRACE:
        facts["contradiction"] = "ended at %.1f s, past its recorded deadline of %.1f s by more than %.0f s" % (
            took, limit, GRACE)
    if cut:
        facts["state"] = "cut at its recorded deadline (%s; %.1f of %.1f s)" % (", ".join(cut), took, limit)
    elif run.get("complete") is True:
        facts["state"] = "complete within its recorded deadline (%.1f of %.1f s)" % (took, limit)
    else:
        facts["state"] = "incomplete before its recorded deadline, not cut by it (%.1f of %.1f s; %s)" % (
            took, limit, ", ".join("%s exit %s%s%s" % (lane.get("lane"), lane.get("returncode"), (
                ", its tree not proven gone" if lane.get("containment") else ""), (
                ", none of its cases failed" if lane.get("no_failing_case") else "")) for lane in lanes))
    return facts


def bound_run(role, run, row, tests, expect):
    """Why RUN (``copy_run`` of the folder the checks name for ROLE) does not carry what the record says of it; []
    when it does. Its files and run.json are complete; its commit is the record's for that role (new
    ``composition_commit``, baseline ``layers_base.k2``, target_run ``target.target_sha``); both lanes, in order, ended
    complete (exit 0 or 1 with cases, 5 without); and the checks' summary of it is run.json's. For W' also: the
    selection is the targets every lane read, and the floor is the bound target's own (source, tests, digest)."""
    if not run:
        return ["no %s run folder" % role]
    problems = list(run.get("problems") or [])
    if "record" not in run:   # the folder or its run.json is gone: nothing more to read
        return problems
    record, lanes = run["record"], run.get("lanes") or []
    want = {"new": row.get("composition_commit"), "baseline": (row.get("layers_base") or {}).get("k2"),
            "target_run": (row.get("target") or {}).get("target_sha")}[role]
    if not HEX40.match(str(want or "")) or run.get("commit") != want:
        problems.append("commit %s, not the record's %s" % (run.get("commit"), want))
    if record.get("complete") is not True:
        problems.append("run.json complete is %r" % record.get("complete"))
    if [lane.get("lane") for lane in lanes] != LANES:
        problems.append("lanes %s, not %s" % ([lane.get("lane") for lane in lanes], LANES))
    for lane in lanes:
        code, cases = lane.get("returncode"), lane.get("cases")
        if lane.get("output") != "complete" or not (code in (0, 1) and cases or code == 5 and not cases):
            problems.append("the %s lane ended %s (exit %s, %s cases)" % (lane.get("lane"), lane.get("output"), code,
                                                                       cases))
        if lane.get("containment"):
            problems.append("the %s lane's process tree was not proven gone: %s" % (lane.get("lane"),
                                                                                     lane["containment"]))
        if lane.get("no_failing_case"):   # its session failed outside its tests: no case names that failure
            problems.append("the %s lane exited %s though none of its cases failed" % (lane.get("lane"), code))
    summary = dict(tests.get(role) or {})
    if role == "new" and "failed" in tests:   # W' failures sit beside its summary, not in it
        summary["failed"] = tests["failed"]
    for key in ("complete", "returncode", "cases", "crashed", "seconds", "failed"):
        if key in summary and summary[key] != record.get(key):
            problems.append("the checks record %s %r, run.json %r" % (key, summary[key], record.get(key)))
    if "lanes" in summary and summary["lanes"] != [{key: lane.get(key) for key in ("lane", "returncode", "cases",
                                                                                  "seconds", *LANE_OPTIONAL)
                                                    if key in lane or key not in LANE_OPTIONAL} for lane in lanes]:
        problems.append("the checks record other lanes than run.json")
    if role == "new":
        targets, selection, floor = run.get("targets") or [], run.get("selection") or {}, run.get("floor") or {}
        # The targets are what the lanes were asked to run, not what ran (``copy_run``'s ``executed``). A selected
        # tier lists its modules, so they are its selection's files; a full tier names the one suite directory, and
        # its selection's file count is the selector's, never a count of target lines.
        tier = tests.get("tier")
        if selection.get("tier") != tier or selection.get("file_count") != tests.get("file_count"):
            problems.append("the selection record (%s, %s files) is not the checks' (%s, %s files)" % (
                selection.get("tier"), selection.get("file_count"), tier, tests.get("file_count")))
        if tier == "selected" and (selection.get("files") != targets or len(targets) != tests.get("file_count")):
            problems.append("the selected tier's %d targets are not its selection's %s files" % (
                len(targets), tests.get("file_count")))
        elif tier == "full" and (len(targets) != 1 or targets[0].endswith(".py") or "::" in targets[0]):
            problems.append("the full tier's targets are %s, not the one suite directory" % targets[:3])
        elif tier not in ("selected", "full"):
            problems.append("the tier is %r" % tier)
        if (floor.get("source") != "%s:%s" % (expect["target"], SELECTOR) or floor.get("tests") != expect["floor"]
                or floor.get("sha256") != sha256_bytes(json.dumps(floor.get("tests"), sort_keys=True).encode())
                or tests.get("floor") != floor):
            problems.append("the floor %s (sha256 %s) is not the bound target's" % (floor.get("source"),
                                                                                    floor.get("sha256")))
        counted = (len(run.get("footprint") or []), len(run.get("local_tests") or []))
        if (tests.get("changed"), tests.get("local_tests")) != counted:
            problems.append("the checks count %s changed paths and %s local tests, run.json %d and %d" % (
                tests.get("changed"), tests.get("local_tests"), *counted))
    return problems


def classify(row, refusal, crash, runs=None):
    """The product's outcome: READY, or NONREADY named (and, for a run that did not complete, its cost lower bound;
    for checks a missing prerequisite left unverified, that prerequisite)."""
    if crash:
        return {"outcome": "NONREADY", "named": "crashed: %s: %s" % (crash["type"], crash["text"][:300])}
    if refusal:
        return {"outcome": "NONREADY", "named": "refused: %s: %s" % (refusal["code"], refusal["text"][:300])}
    state, reason = str((row or {}).get("state") or ""), str((row or {}).get("reason") or "")
    tests = ((row or {}).get("checks") or {}).get("tests") or {}
    if state == "ready" and (row.get("checks") or {}).get("ok") is True:
        return {"outcome": "READY", "named": "ready", "already_failing": tests.get("already_failing") or []}
    named = {"outcome": "NONREADY", "named": "%s: %s" % (state or "no record", reason or "-")}
    if tests.get("unverified"):
        new, copied = tests.get("new") or {}, (runs or {}).get("new") or {}
        # Both read from the record, never assumed: budget_seconds is the checks' one budget, which W', W and T share;
        # deadline_seconds is what remained of it when W' began (its run.json), None where no W' run was read.
        named.update(unverified=tests["unverified"], cost_lower_bound={
            "seconds": new.get("seconds"), "cases_reported": new.get("cases"), "lanes": new.get("lanes"),
            "files_selected": tests.get("file_count"), "budget_seconds": tests.get("budget_seconds"),
            "deadline_seconds": copied.get("deadline_seconds"),
            "deadline": (copied.get("deadline") or {}).get("state")})
        if tests.get("prerequisite"):
            named["prerequisite"] = tests["prerequisite"]
    for key in ("blocking", "floor_missing", "already_failing"):
        if tests.get(key):
            named[key] = tests[key]
    return named


def adjudicate(bind, row, outcome, runs, before, after, resolved_budget=None):
    """PASS, FAIL or UNKNOWN for the bound expectation, with every check that decided it. ``resolved_budget`` is the
    checks' budget as the loaded implementation resolved it before preparing (``verify_loaded``), when it has one."""
    expect, kind = bind["expect"], bind["kind"]
    tests = ((row or {}).get("checks") or {}).get("tests") or {}
    fails, unknown, checks = [], [], {}

    def check(name, ok, detail=""):
        checks[name] = {"ok": ok, "detail": detail}
        (fails if ok is False else unknown if ok is None else []).append("%s %s" % (name, detail) if detail else name)

    if row is None:
        check("record", None, outcome["named"])
    else:
        check("target", (row.get("target") or {}).get("target_sha") == expect["target"],
              str((row.get("target") or {}).get("target_sha")))
        if tests and "tier" in tests:
            # What W' was asked to run: run.json targets, which copy_run binds to targets.txt and every lane's @file
            # (a selected tier's modules; a full tier's one suite directory). What it ran: the files its complete JUnit
            # reports hold at least one node of (copy_run's executed, unique node ids). Membership is read from the
            # latter. A file missing there is a FAIL only when every report was read and the run was complete;
            # otherwise it is not established.
            copied = runs.get("new") or {}
            targets, ran = sorted(copied.get("targets") or []), set(copied.get("executed_files") or [])
            settled = bool(copied.get("reports_read")) and (copied.get("record") or {}).get("complete") is True

            def executed(paths):
                missing = sorted(set(paths) - ran)
                return (True if ran and not missing else False if settled else None), missing

            footprint_paths = sorted(tests.get("footprint") or copied.get("footprint") or [])
            check("tier", tests.get("tier") == expect["tier"], str(tests.get("tier")))
            ok, missing = executed(expect["floor"])
            check("floor executed on W'", ok, "%d of %d floor files in W''s reports (%d files, %d unique nodes)%s" % (
                len(expect["floor"]) - len(missing), len(expect["floor"]), len(ran), copied.get("cases") or 0,
                "; not there: " + ", ".join(missing[:8]) if missing else ""))
            if kind == "clean":
                check("selection is exactly the target floor", targets == sorted(expect["floor"]) if targets else None,
                      ", ".join(targets[:12]))
                check("executed within the selection", (ran <= set(targets)) if ran and targets else None,
                      ", ".join(sorted(ran - set(targets))[:8]))
                check("no local footprint", (tests.get("changed") == 0) and not footprint_paths, str(footprint_paths[:5]))
                check("no local tests", tests.get("local_tests") == 0, str(tests.get("local_tests")))
            else:
                evolved = bind["root"]["evolved"]
                check("local cutover in the footprint", evolved["cutover"] in footprint_paths if footprint_paths else None,
                      evolved["cutover"])
                check("local test executed on W'", executed([evolved["local_test"]])[0], evolved["local_test"])
                verdicts = (copied.get("by_file") or {}).get(evolved["local_test"])
                check("local test passed on W'", (bool(verdicts) and set(verdicts) == {"passed"}) if verdicts is not None
                      else None, str(verdicts))
            if tests.get("prerequisite"):   # nothing ran: named, never a test failure
                need_of = tests["prerequisite"]
                check("prerequisite", None, "%s %s: %s" % (need_of.get("prerequisite"), need_of.get("status"),
                                                           tests.get("unverified") or "-"))
        elif outcome["outcome"] == "READY":
            check("checks record", False, "ready without a tests record")
        else:   # a record that names no selection and no run says nothing of what ran: not that none did
            check("checks record", None, "the record names no selection or test run, so what ran is not established "
                                         "(%s)" % outcome["named"])
        # Every run the checks name is read against its role. A READY rests on W' and, when it excused W' failures,
        # on W: those are demanded, and one missing, incomplete or bound elsewhere is not established, never PASS.
        # T's attribution run belongs to blocking failures and is never demanded.
        ready = outcome["outcome"] == "READY"
        new = runs.get("new") or {}
        failed, excused = (new.get("record") or {}).get("failed") or {}, outcome.get("already_failing") or []
        demanded = ("new", "baseline") if ready and (failed or excused) else ("new",) if ready else ()
        for role in ROLES:
            if role not in demanded and not (tests.get(role) or {}).get("evidence"):
                continue
            problems = bound_run(role, runs.get(role), row, tests, expect)
            if role == "baseline" and ready:   # each excused W' failure fails the same way in W's own run
                proof, targets = ((runs.get(role) or {}).get("record") or {}).get("failed") or {}, (
                    runs.get(role) or {}).get("targets") or []
                if "record" in new and sorted(failed) != sorted(excused):
                    problems.append("W' failed %s, the record excuses %s" % (sorted(failed)[:5], sorted(excused)[:5]))
                unproved = [nodeid for nodeid in excused if nodeid not in targets or nodeid not in failed
                            or proof.get(nodeid) != failed[nodeid]]
                if unproved:
                    problems.append("not shown failing the same way on W' and W: %s" % ", ".join(unproved[:5]))
            check("%s run evidence" % role.replace("_run", ""), None if problems else True, "; ".join(problems))
        # Each run read against the deadline its own run.json records, and the runs against the checks' one budget:
        # W' begins with at most the whole budget, and W, then T, with at most what the run before them left of it.
        budget, previous = _number(tests.get("budget_seconds")), None
        read_runs = [(role, runs[role]) for role in ROLES if (runs.get(role) or {}).get("deadline")]
        for role, copied in read_runs:
            facts = copied["deadline"]
            check("%s deadline" % role.replace("_run", ""), False if facts["contradiction"] else
                  True if facts["deadline_seconds"] is not None and facts["seconds"] is not None else None,
                  facts["contradiction"] or facts["state"])
        if read_runs:
            shared, unread = [], []
            if budget is None or budget <= 0:
                unread.append("the checks record no budget_seconds (%r)" % tests.get("budget_seconds"))
            elif resolved_budget is not None and budget != resolved_budget:
                shared.append("the record's budget is %s s, the implementation resolves %s s here" % (
                    budget, resolved_budget))
            for role, copied in read_runs:
                limit, took = copied["deadline"]["deadline_seconds"], copied["deadline"]["seconds"]
                allowed = budget if previous is None else (
                    None if previous[1] is None or previous[2] is None else previous[1] - previous[2])
                if limit is None or allowed is None:
                    unread.append("%s's deadline is not read against %s" % (
                        role, "the budget" if previous is None else previous[0]))
                elif limit > allowed + TOLERANCE:
                    shared.append("%s began with %.1f s, more than the %.1f s %s left" % (
                        role, limit, allowed, "budget" if previous is None else previous[0]))
                previous = (role, limit, took)
            check("one budget", False if shared else None if unread else True, "; ".join(shared + unread) or (
                "%s s shared by %s" % (budget, ", ".join(role for role, _copied in read_runs))))
        # A READY rests only on complete runs: an incomplete W', or an incomplete W it excused failures through, or a
        # record that names its checks unverified, never becomes ready. Excuses through a W that excuses nothing are
        # likewise a contradiction.
        if ready:
            baseline, contradictions = tests.get("baseline") or {}, []
            if tests.get("unverified") or tests.get("prerequisite"):
                contradictions.append("the record calls the checks unverified: %s" % (tests.get("unverified") or
                                                                                    tests.get("prerequisite")))
            if (tests.get("new") or {}).get("complete") is not True or (new.get("record") or {}).get("complete") is False:
                contradictions.append("W' complete is %r in the record, %r in run.json" % (
                    (tests.get("new") or {}).get("complete"), (new.get("record") or {}).get("complete")))
            if "baseline" in demanded and (baseline.get("complete") is not True or (
                    (runs.get("baseline") or {}).get("record") or {}).get("complete") is False):
                contradictions.append("W complete is %r in the record, %r in run.json" % (
                    baseline.get("complete"), ((runs.get("baseline") or {}).get("record") or {}).get("complete")))
            if excused and baseline.get("excuses") is False:
                contradictions.append("%d failures excused through a W that excuses none (%s)" % (
                    len(excused), baseline.get("why") or "-"))
            check("ready on complete runs", False if contradictions else True, "; ".join(contradictions))
    if before is None or "unread" in before or after is None or "unread" in after:
        check("serving checkout kept", None, "a footprint read failed")
    else:
        check("serving checkout kept", before == after, "H/I/W/U witnesses before and after")
    if outcome["outcome"] != "READY":
        (unknown if outcome.get("unverified") or outcome["named"].startswith(("crashed", "refused")) or row is None
         else fails).append("outcome %s" % outcome["named"])
    verdict = "FAIL" if fails else "UNKNOWN" if unknown else "PASS"
    return verdict, checks, fails, unknown


# ------------------------------------------------------------------ main
def run(bind_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir()   # create-only; its parent must exist
    receipt = {"driver": {"path": real(__file__), "sha256": sha256_file(__file__)}, "started": utc(), "steps": [],
               "green": False}
    rc, verdict, detail = 2, "UNKNOWN", ""
    try:
        bind, raw = load_bind(bind_path)
        receipt["bind"] = {"path": str(bind_path), "sha256": sha256_bytes(raw), "content": bind}
        receipt["interpreter"] = check_interpreter(bind)
        receipt["packages"] = check_packages(bind)
        receipt["implementation"] = check_implementation(bind)
        receipt["root"] = check_root(bind)
        before = footprint(bind)
        need("unread" not in before, "root", "the serving checkout's footprint is unreadable: %s" % before.get("unread"))
        receipt["footprint_before"] = before
        receipt["steps"].append({"step": "verified", "at": utc()})
        receipt["configuration"] = bind_configuration(bind)
        modules = import_implementation(bind)
        receipt["loaded_before"] = verify_loaded(bind, modules)
        receipt["boot_binding"] = init_like_boot(bind, modules["supervisor.git_ops"])
        receipt["steps"].append({"step": "configured and imported", "at": utc()})
    except Refused as exc:
        receipt.update(refused={"stage": exc.stage, "text": exc.text}, ended=utc())
        detail = "refused before any effect at %s: %s" % (exc.stage, exc.text)
        write_new(out_dir / "receipt.json", json.dumps(receipt, indent=1, sort_keys=True, default=str) + "\n")
        write_new(out_dir / "verdict.txt", "verdict=UNKNOWN expected=%s detail=%s\n" % (
            receipt.get("bind", {}).get("content", {}).get("kind", "-"), detail))
        print(detail, file=sys.stderr)
        return 2
    prepare, checks_owner = modules["supervisor.update_prepare"], modules["supervisor.update_checks"].workspace_checks
    row = refusal = crash = None
    started = time.monotonic()
    receipt["steps"].append({"step": "prepare requested", "at": utc()})
    try:
        prepare.admit(None)
        row = prepare.prepare(fetch=True, without_local_changes=False, checks=checks_owner, expected=None, admitted=True)
    except prepare.PrepareRefused as exc:
        refusal = {"code": exc.code, "text": exc.text, "facts": exc.facts}
        prepare.note_refusal(exc)
    except Exception as exc:  # the implementation's own failure: kept as it was, never retried
        crash = {"type": type(exc).__name__, "text": str(exc), "traceback": traceback.format_exc()}
    receipt["prepare"] = {"seconds": round(time.monotonic() - started, 1), "ended": utc(), "refusal": refusal,
                          "crash": crash, "record": row}
    try:
        try:
            receipt["projection"] = prepare.projection()
            receipt["stored_record"] = prepare.find(row["candidate_id"]) if row else None
        except Exception as exc:
            receipt["projection_error"] = "%s: %s" % (type(exc).__name__, exc)
        tests = ((row or {}).get("checks") or {}).get("tests") or {}
        # Only the runs the record names are copied. A preparation that failed after running tests but before saving its
        # checks names none: their folders stay in the candidate's scratch, uncopied, and what ran is not established.
        runs = {}
        for role in ROLES:
            folder = (tests.get(role) or {}).get("evidence")
            if folder:
                runs[role] = copy_run(folder, out_dir / "runs" / role)
        receipt["runs"] = runs
        # The record's own words on its budget, prerequisites, packages and comparisons, kept beside the runs read.
        receipt["checks_contract"] = {
            **{key: tests.get(key) for key in ("budget_seconds", "test_evidence", "node", "user_site", "prerequisite",
                                               "unverified", "process_root", "process_root_release")},
            "baseline": {key: (tests.get("baseline") or {}).get(key) for key in ("complete", "excuses", "not_excused",
                                                                                 "why")},
            "target_run": {key: (tests.get("target_run") or {}).get(key) for key in ("complete", "unavailable",
                                                                                     "presence")}}
        inside, foreign = loaded_modules(bind["implementation"]["dir"])
        receipt["loaded_after"] = {"modules": inside, "foreign": foreign}
        after = footprint(bind)
        receipt["footprint_after"] = after
        outcome = classify(row, refusal, crash, runs)
        if foreign:
            outcome = {"outcome": "NONREADY", "named": "refused: product modules loaded from outside: %s" % foreign[:3]}
        verdict, decided, fails, unknown = adjudicate(bind, row, outcome, runs, before, after, (
            receipt.get("loaded_before") or {}).get("readback", {}).get("checks_budget_seconds"))
        failed_cases = any("failed" in verdicts for run_facts in runs.values()
                           for verdicts in (run_facts.get("by_file") or {}).values())
        receipt["green"] = (verdict == "PASS" and not outcome.get("already_failing") and not failed_cases)
    except Exception as exc:  # the receipt is written whatever the reading of it met
        outcome = classify(row, refusal, crash, receipt.get("runs"))
        verdict, decided, fails, unknown = "UNKNOWN", {}, [], ["receipt %s: %s" % (type(exc).__name__, exc)]
        receipt["receipt_error"] = traceback.format_exc()
    receipt.update(outcome=outcome, verdict=verdict, checks=decided, ended=utc())
    detail = "%s; %s" % (outcome["named"], "; ".join(fails or unknown) or "every bound expectation held")
    if outcome.get("already_failing"):   # ready (or not) on failures W shares: never green, whatever the verdict
        detail += "; already failing on W, not green: %s" % ", ".join(outcome["already_failing"][:5])
    if outcome.get("cost_lower_bound"):
        detail += "; cost lower bound %s" % json.dumps(outcome["cost_lower_bound"], sort_keys=True)
    # A process root the checks kept (something beneath it not proven gone) is named; the product's verdict stands.
    release = (receipt.get("checks_contract") or {}).get("process_root_release") or {}
    if release.get("state") == "retained":
        detail += "; process root retained: %s (%s)" % (release.get("path"), str(release.get("reason") or "-")[:300])
    write_new(out_dir / "receipt.json", json.dumps(receipt, indent=1, sort_keys=True, default=str) + "\n")
    write_new(out_dir / "outcome.txt", "outcome=%s detail=%s\n" % (outcome["outcome"], outcome["named"]))
    write_new(out_dir / "verdict.txt", "verdict=%s expected=%s tier=%s green=%s detail=%s\n" % (
        verdict, bind["kind"], bind["expect"]["tier"], "yes" if receipt["green"] is True else "no", detail))
    print("%s %s" % (verdict, detail))
    return {"PASS": 0, "FAIL": 1}.get(verdict, 3)


def main(argv):
    if len(argv) != 3 or argv[0] != "run":
        print(__doc__, file=sys.stderr)
        return 2
    return run(argv[1], argv[2])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
