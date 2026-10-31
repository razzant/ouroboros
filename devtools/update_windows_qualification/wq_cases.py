"""The cases (claims and failure conditions: README). Three private roots, run one after another:

U  released v7.8.0 app, packaged interpreter's user site: entry.U, A, B, C, E.U
D  released v7.8.0 app with ``--runtime-venv "<root>\\app root\\runtime venv"``: entry.D, D, E.D
F  bare Windows CPython (no venv, not python-standalone), private environment: F, E.F

Each mode: provision (mirror of the fixture targets, managed checkout cloned from the release seed and moved to
fixture S0, the Git-ignored checks-only stub), open the mode's Job, run its cases in order, and in ``finally``
export the evidence and close the Job. A case that cannot start because an earlier one of its mode did not
pass is NOT_RUN; nothing is retried, skipped into PASS or repaired by the harness.
"""
import json
import os
import pathlib
import shutil
import subprocess
import time

import wq_fixtures as fx
from wq_custody import Custody, identities, known_roles, launcher_ids, unread
from wq_product import (Install, console_launcher_parts, content_changes, content_inventory, file_digest, git,
                        inventory, layers, path_id, same_inventory, shebang_names, user_site, venv_site)
from wq_verdict import (HEX40, MODES, PASS, UNSETTLED, Case, CaseStop, activation_request, closed_adoption, conjoin,
                        generation_seen, layers_equal, prepared_environment, prepared_ready, same_path, settled, under,
                        witness_for)

DEADLINE = {"first_ready": 2700, "ready": 900, "prepare": 2700, "settle": 1800, "tool": 180, "stop": 90}
PORTS = {"U": 28770, "D": 28771, "F": 28772}
# Each mode's own host-service port beside its main one: the server's boot binds OUROBOROS_HOST_SERVICE_PORT
# (product default 8767) and its boot fails when that port is taken (e3823042's answer 1). Distinct from every
# main port, outside each main port's fallback window (port+1 .. port+9) and the product defaults 8765-8774.
HOST_SERVICE_PORTS = {"U": 28790, "D": 28791, "F": 28792}
SENTINELS = ("qualification/H.txt", "qualification/U.bin")
H_TEXT, I_TEXT, W_TEXT = (b"H: committed by the owner before the update\n", b"I: staged by the owner\n",
                          b"W: an unstaged edit on top of the staged one\n")
U_BYTES = bytes(range(256)) + b"\r\n\x00winqual untracked binary\n"

# What each case must have recorded to PASS: the claims, by name (README "What a PASS proves").
_ENTRY = ("server_ready", "frozen_root_exited_after_handover", "source_launcher_successor", "serving_tree_is_S0",
          "server_imports_base_generation", "served_version")
_PREPARE = ("prepare_accepted", "prepared_ready", "target_bound", "checks_only_stub_reported",
            "prepared_beside_fixed_path", "prepared_holds_target_version", "serving_untouched_by_preparation")
_ADOPTED = ("activation_accepted", "settled_adopted", "attempt_bound_to_candidate_composition", "adopted_composition",
            "ready_after_settlement", "prepared_directory_selected",
            "fresh_launcher", "fresh_server")
_COLD = ("previous_processes_stopped", "previous_endpoint_gone", "no_process_outside_custody", "cold_ready",
         "all_processes_new", "same_composition", "no_dependency_reconstruction",
         "cold_server_imports_selected_generation", "data_markers_kept")
REQUIRED_CHECKS = {
    "entry.U": _ENTRY,
    "A": ("channel_development", "marker_winqual-a-before_written", "owner_layers_seeded", "native_holder_acknowledged",
          *_PREPARE, "marker_winqual-a-after_written", "holder_alive_before_switch", *_ADOPTED,
          "serving_directory_kept_aside", "holder_survived_switch", "new_server_imports_T2_generation",
          "owner_layers_carried", "data_markers_kept", "holder_released"),
    "B": (*_PREPARE, "marker_winqual-b-after_written", "cwd_holder_acknowledged", "settled_returned_retry",
          "refusal_names_system_error", "holder_alive_through_attempt", "coherent_previous_generation",
          "previous_generation_serves", "launcher_not_replaced_without_switch", "no_crash_loop", "holder_released",
          *_ADOPTED, "retry_is_a_new_attempt", "serving_directory_kept_aside", "new_server_imports_T3_generation",
          "owner_layers_carried", "data_markers_kept"),
    "C": (*_PREPARE, "marker_winqual-c-after_written", "activation_accepted", "settled_returned",
          "attempt_bound_to_candidate_composition",
          "ready_after_settlement", "intended_dependency_failure", "coherent_previous_composition",
          "previous_code_imports_previous_package", "fresh_process_imports_restored_package", "data_markers_kept"),
    "E.U": (*_COLD, "released_app_handed_over_again", "owner_layers_carried"),
    "entry.D": (*_ENTRY, "runtime_venv_made_by_product", "server_runs_in_venv_on_app_python"),
    "D": ("channel_development", "marker_winqual-d-before_written", "venv_python_holder_acknowledged",
          "console_image_holder_acknowledged", *_PREPARE, "prepared_console_launcher_names_generation",
          "marker_winqual-d-after_written", *_ADOPTED, "product_launcher_runs_from_venv_redirector",
          "serving_directory_kept_aside",
          "console_launcher_relocated_only_in_its_shebang", "not_reinstalled",
          "relocated_console_script_runs_new_generation", "venv_python_holder_survived",
          "console_image_holder_survived", "new_server_imports_T5_generation", "data_markers_kept",
          "venv_python_holder_released", "console_image_holder_released"),
    "E.D": (*_COLD, "released_app_handed_over_again", "console_script_after_cold_entry"),
    "F": ("owner_installed_base_lock_into_interpreter", "global_packages_measured", "server_ready",
          "server_on_bare_interpreter",
          "channel_development", "marker_winqual-f-before_written", *_PREPARE, "private_environment_made_by_product",
          "global_packages_untouched_by_preparation", "marker_winqual-f-after_written", *_ADOPTED,
          "server_runs_in_private_environment", "server_is_the_base_interpreter_itself", "global_packages_unchanged",
          "data_markers_kept"),
    "E.F": (*_COLD, "source_launcher_stopped", "direct_entry_ready", "direct_command_is_waited_parent",
            "direct_entry_runs_in_private_environment", "module_cli_entry_exits_0",
            "module_entry_redirected_before_imports", "global_packages_unchanged"),
}
# Required only once the mode ended, never while its cases run: ``_sequence`` reads each case's status right after
# its step to decide whether the next case may start, and the evidence is exported after the last step. A case
# that ran gets this requirement, and its check, in ``_close_and_export``, before any terminal status is read.
TERMINAL_CHECKS = ("evidence_exported",)


class Mode:
    def __init__(self, ctx, label):
        self.ctx, self.label = ctx, label
        self.root = ctx.run_root / label  # no space: Git file URLs and the mirror stay plain
        self.install = Install(self.root, PORTS[label])
        self.mirror = self.root / "mirror" / "official.git"
        self.evidence = ctx.evidence / "modes" / label
        self.markers, self.expected_layers, self.prepared_ids = [], None, []
        self.custody = None

    # ------------------------------------------------------------------ environment
    def env(self, **extra):
        """The installation's environment: an allowlist of the host's, this mode's private roots and ports, the
        wheelhouse, the mirror mapping and the two equipment switches. No credential, no PYTHON*/PIP_* of the host."""
        keep = ("SystemRoot", "WINDIR", "COMSPEC", "PATHEXT", "SystemDrive", "NUMBER_OF_PROCESSORS", "OS",
                "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "PROCESSOR_LEVEL", "PROCESSOR_REVISION",
                "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "CommonProgramFiles", "CommonProgramFiles(x86)",
                "CommonProgramW6432", "ProgramData", "ALLUSERSPROFILE", "PUBLIC", "USERNAME", "USERDOMAIN",
                "COMPUTERNAME")
        env = {key: os.environ[key] for key in keep if key in os.environ}
        system = pathlib.Path(os.environ["SystemRoot"])
        home = self.root / "home"
        env.update({
            "PATH": os.pathsep.join([self.ctx.git_dir, str(system / "System32"), str(system),
                                     str(system / "System32" / "WindowsPowerShell" / "v1.0")]),
            "USERPROFILE": str(home), "HOME": str(home), "APPDATA": str(home / "AppData" / "Roaming"),
            "LOCALAPPDATA": str(home / "AppData" / "Local"), "TEMP": str(self.root / "tmp"), "TMP": str(self.root / "tmp"),
            "OUROBOROS_APP_ROOT": str(self.install.app_root), "OUROBOROS_SERVER_PORT": str(PORTS[self.label]),
            "OUROBOROS_HOST_SERVICE_PORT": str(HOST_SERVICE_PORTS[self.label]),
            "PIP_NO_INDEX": "1", "PIP_FIND_LINKS": self.ctx.wheelhouse.as_uri(), "PIP_NO_INPUT": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_CONFIG_FILE": os.devnull,
            "PIP_CACHE_DIR": str(self.root / "cache" / "pip"),
            "GIT_CONFIG_GLOBAL": str(self.root / "gitconfig"), "GIT_TERMINAL_PROMPT": "0",
            fx.WITNESS_ENV: str(self.install.witness_dir), fx.STUB_ENV: "1",
        })
        env.update({key: str(value) for key, value in extra.items()})
        return env

    # ------------------------------------------------------------------ provisioning (recorded, never a verdict)
    def provision(self):
        ctx, emit = self.ctx, self.ctx.raw.emit
        for path in (self.root / "home" / "AppData" / "Roaming", self.root / "home" / "AppData" / "Local",
                     self.root / "tmp", self.root / "cache", self.install.witness_dir, self.install.logs,
                     self.evidence):
            path.mkdir(parents=True, exist_ok=True)
        fixture_env, mirror_url = self.env(), self.mirror.as_uri()
        mapped = "".join(f"\tinsteadOf = {url}\n" for url in ctx.release["official_urls"])
        (self.root / "gitconfig").write_text(  # before any Git command of this mode reads it
            f'[url "{mirror_url}"]\n{mapped}[user]\n\tname = Windows qualification owner\n'
            "\temail = winqual-owner@invalid\n", encoding="utf-8")
        git(self.root, "clone", "--bare", "--quiet", str(ctx.fixture_repo), str(self.mirror), env=fixture_env)
        for branch in ("ouroboros", "main", "ouroboros-stable"):
            git(self.mirror, "update-ref", f"refs/heads/{branch}", ctx.commits["S0"], env=fixture_env)
        repo = self.install.repo
        repo.parent.mkdir(parents=True, exist_ok=True)
        # As the released bootstrap makes a managed checkout (clone of its own seed), then moved to S0.
        git(repo.parent, "clone", "--quiet", str(ctx.release["internal"] / "repo.bundle"), str(repo), env=fixture_env)
        git(repo, "remote", "remove", "origin", env=fixture_env)
        git(repo, "fetch", "--quiet", "--no-tags", mirror_url, "refs/heads/winqual/S0", env=fixture_env)
        git(repo, "checkout", "--quiet", "-B", "ouroboros", "FETCH_HEAD", env=fixture_env)
        git(repo, "branch", "--force", "ouroboros-stable", ctx.release["manifest"]["source_sha"], env=fixture_env)
        shutil.copyfile(ctx.equipment / "stub" / "sitecustomize.py", repo / "sitecustomize.py")
        with open(repo / ".git" / "info" / "exclude", "a", encoding="utf-8") as fh:
            fh.write("\n/sitecustomize.py\n")
        head, status = git(repo, "rev-parse", "HEAD", env=fixture_env), git(repo, "status", "--porcelain", env=fixture_env)
        emit("provisioned", mode=self.label, head=head, status=status, mirror=mirror_url,
             managed_metadata=(repo / ".git" / "ouroboros-managed.json").exists())
        if head != ctx.commits["S0"] or status:
            raise RuntimeError(f"provisioned checkout is {head} with status {status!r}, expected clean S0")

    def mirror_to(self, name):
        git(self.mirror, "update-ref", "refs/heads/ouroboros", self.ctx.commits[name], env=self.env())
        self.ctx.raw.emit("official_target", mode=self.label, name=name, sha=self.ctx.commits[name])

    # ------------------------------------------------------------------ starting processes
    def start(self, label, argv, *, cwd=None, interactive=False, **env):
        return self.custody.spawn(label, argv, env=self.env(**env), cwd=cwd or self.root, logs=self.install.logs,
                                  interactive=interactive)

    def start_released(self, label, extra=()):
        return self.start(label, [self.ctx.release["exe"], "--no-ui", "--launch-intent", "owner", *extra])

    def run_bounded(self, label, argv, deadline, *, cwd=None, **env):
        """A short command in the Job; ``(returncode or None at the deadline, stdout text)``."""
        proc, _identity, _ = self.start(label, argv, cwd=cwd, **env)
        try:
            rc = proc.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            rc = None
        text = (self.install.logs / f"{label}.stdout").read_text(encoding="utf-8", errors="replace")
        self.ctx.raw.emit("bounded", mode=self.label, label=label, returncode=rc)
        return rc, text

    def hold(self, label, argv, **env):
        proc, identity, holder = self.start(label, argv, interactive=True, **env)
        ready = holder.reply(timeout=180)
        return {"proc": proc, "identity": identity, "io": holder, "ready": ready}

    def release_holder(self, holder):
        reply = holder["io"].request("exit", timeout=60)
        try:
            holder["proc"].wait(timeout=60)
            exited = True
        except subprocess.TimeoutExpired:
            exited = False  # a holder is equipment: one still running at the deadline is unmeasured, not refuted
        state = self.custody.alive(holder["identity"])  # tri-state: unobserved stays None
        return reply, None if state is None or not exited else state is False

    # ------------------------------------------------------------------ observations
    def server_witness(self, server, since_ns=0):
        return witness_for(self.install.witnesses(), server, "ouroboros.gateway.router", since_ns)

    def serving(self, fixed, site, server=None):
        """What must not move while a candidate is prepared: layers, the fixed package path, the serving
        distributions, the serving process and the owner's markers."""
        return {"layers": self.layers(), "env_id": path_id(fixed),
                "inventory": inventory(site), "server": server and (server["pid"], server["birth"]),
                "markers": self.install.markers(self.markers)}

    def layers(self):
        """H, branch, I, W/U status and, once seeded, the owner sentinels (each one required to be read)."""
        return layers(self.install.repo, SENTINELS if self.expected_layers else (), self.env())

    def head(self):
        return git(self.install.repo, "rev-parse", "HEAD", env=self.env())

    def snapshot(self, tag, **values):
        path = self.evidence / f"{tag}.json"
        path.write_text(json.dumps(values, indent=1, sort_keys=True, default=str), encoding="utf-8")

    def export(self):
        """Copy the installation's logs and records (never its packages or checkout) into the evidence.
        Never raises. ``ok`` only when the required folders (the product's logs, the witnesses, the harness
        logs) existed and every file found was copied; any other outcome leaves the mode's cases unmeasured."""
        copied, errors, missing = [], [], []
        required = (("data-logs", self.install.data / "logs"), ("witness", self.install.witness_dir),
                    ("harness-logs", self.install.logs))
        sources = [*required, ("state-private_copies", self.install.data / "state" / "private_copies"),
                   ("state-body_adoption", self.install.data / "state" / "body_adoption")]
        files = [self.install.data / "state" / name for name in ("update_activation.json", "server_port.bindings.json",
                                                                  "server_process.json", "server_port")]
        files += [self.install.repo / ".git" / "ouroboros-managed.json", self.root / "gitconfig"]
        try:
            missing = [name for name, folder in required if not folder.is_dir()]
            for name, folder in sources:
                for path in sorted(folder.rglob("*")) if folder.is_dir() else []:
                    if path.is_file():
                        target = self.evidence / name / path.relative_to(folder)
                        try:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(path, target)
                            copied.append(str(target))
                        except OSError as exc:
                            errors.append(f"{path}: {exc}")
            for path in files:
                if path.is_file():
                    try:
                        shutil.copy2(path, self.evidence / path.name)
                        copied.append(path.name)
                    except OSError as exc:
                        errors.append(f"{path}: {exc}")
        except Exception as exc:  # a walk that failed part-way: what was copied stays, the export is not ok
            errors.append(f"export stopped: {exc!r}")
        outcome = {"ok": not errors and not missing, "copied": len(copied), "errors": errors[:50],
                   "missing_required": missing}
        try:
            self.ctx.raw.emit("exported", mode=self.label, **outcome)
        except Exception as exc:
            outcome.update(ok=False, errors=[*outcome["errors"], f"event not written: {exc!r}"])
        return outcome

    # ------------------------------------------------------------------ shared steps
    def entry(self, case, extra, site, prefix):
        """The unchanged released app hands the provisioned checkout to its source launcher."""
        proc, root, _ = self.start_released("released-app", extra)
        server = self.install.wait_ready(self.custody, DEADLINE["first_ready"])
        case.require("server_ready", _ready(self.install, server), server=server)
        try:
            rc = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            rc = None
        case.check("frozen_root_exited_after_handover", None if rc is None else rc == 0, returncode=rc, root=root)
        members = self.custody.members()
        roles = known_roles(members)
        internal, repo = str(self.ctx.release["internal"]).lower(), str(self.install.repo).lower()
        successors = [row for row in (roles or {}).get("launcher", []) if row.get("alive")
                      and repo in str(row.get("cmdline")).lower() and "--seed-bundle" in str(row.get("cmdline"))
                      and internal in str(row.get("cmdline")).lower()]
        case.check("source_launcher_successor", None if roles is None else bool(successors) and not roles["frozen"]
                   and all("--runtime-venv" not in str(row.get("cmdline")) for row in successors), roles=roles,
                   unread=unread(members))
        status, view = self.install.http("GET", "/api/update/status")
        case.check("serving_tree_is_S0", conjoin(
            git(self.install.repo, "rev-parse", "HEAD", env=self.env()) == self.ctx.commits["S0"],
            None if status is None else isinstance(view, dict) and view.get("running_sha") == self.ctx.commits["S0"]),
            status=status, view=view)
        witness = self.server_witness(server)
        case.check("server_imports_base_generation", generation_seen(witness, "1.0", site, prefix), witness=witness)
        case.check("served_version", (server["health"] or {}).get("runtime_version") == self.ctx.candidate_version,
                   health=server["health"])
        return server

    def channel_and_marker(self, case, marker):
        ok, facts = self.install.set_channel("development")
        case.require("channel_development", None if facts.get("post") is None or facts.get("seen") is None else ok,
                     **facts)
        self.marker(case, marker)

    def marker(self, case, name):
        ok, facts = self.install.add_marker(name)
        case.require(f"marker_{name}_written", ok, **facts)
        self.markers.append(name)

    def markers_kept(self, case):
        seen = self.install.markers(self.markers)
        return case.check("data_markers_kept", None if seen is None or not self.markers else all(seen.values()),
                          expected=self.markers, seen=seen)

    def prepare(self, case, target, kind, current, site, server):
        """The in-app Prepare of the actual serving process (only its workspace checks stubbed). ``site`` is
        the site-packages serving now (for a bare interpreter: its own), which must not change."""
        self.mirror_to(target)
        before = self.serving(current, site, server)
        status, body = self.install.http("POST", "/api/update/apply", {"strategy": "prepare"}, timeout=120)
        case.require("prepare_accepted", None if status is None else status == 202 and isinstance(body, dict)
                     and body.get("status") == "preparing", http=status, body=body)
        end, view, gone_since = time.monotonic() + DEADLINE["prepare"], {}, None
        while time.monotonic() < end:
            if self.install.server(self.custody) is None:
                gone_since = gone_since or time.monotonic()
                if time.monotonic() - gone_since > 60:
                    view = {**(view or {}), "server_gone": "no live bound server for 60 s during preparation",
                            "server_ended": self.custody.alive(server) is False}  # measured, or unobserved
                    break
            else:
                gone_since = None
            _status, answer = self.install.http("GET", "/api/update/status", timeout=60)
            view = (answer or {}).get("candidate") if isinstance(answer, dict) else {}
            if isinstance(view, dict) and (view.get("refusal") or view.get("state") not in UNSETTLED
                                           and view.get("candidate_id") not in self.prepared_ids):
                break
            time.sleep(3)
        candidate_id = str((view or {}).get("candidate_id") or "")
        if candidate_id and candidate_id in self.prepared_ids:
            view, candidate_id = {**view, "stale": "an earlier attempt's candidate"}, ""
        self.prepared_ids.append(candidate_id)
        record = self.install.candidate(candidate_id) if candidate_id else None
        self.snapshot(f"{case.name}-prepared-{target}", view=view, record=record)
        case.require("prepared_ready", prepared_ready(view), view=view)
        case.check("target_bound", view.get("target_sha") == self.ctx.commits[target] and view.get("channel")
                   == "development" and str(view.get("target_ref") or "").endswith("/ouroboros"), view=view)
        case.check("checks_only_stub_reported", (view.get("checks") or {}).get("stub") == fx.STUB_LABEL,
                   checks=view.get("checks"))
        ok, facts = prepared_environment(record, kind, current)
        case.require("prepared_beside_fixed_path", ok, **facts)
        generation = pathlib.Path(facts["generation"]["user_base"])
        generation_site = user_site(generation) if kind == "user_site" else venv_site(generation)
        prepared = {"id": path_id(generation), "inventory": inventory(generation_site), "path": generation,
                    "site": generation_site, "candidate_id": candidate_id, "record": record, "view": view}
        version = fx.VERSIONS[target]
        case.check("prepared_holds_target_version",
                   ((prepared["inventory"] or {}).get(fx.DIST) or {}).get("version") == version,
                   inventory=prepared["inventory"])
        still, now = _still_serving(self, server)
        after = self.serving(current, site, now)
        same, changed = layers_equal(before["layers"], after["layers"])
        case.check("serving_untouched_by_preparation", conjoin(
            same, before["env_id"] == after["env_id"], same_inventory(before["inventory"], after["inventory"]), still,
            None if before["markers"] is None or after["markers"] is None else before["markers"] == after["markers"]),
            before=before, after=after, **changed)
        return prepared

    def activate(self, case, candidate_id, expect):
        """POST activate for exactly this candidate; follow THIS adoption id to its settled outcome."""
        old_server = self.install.server(self.custody)
        old_members = self.custody.members()
        old_launchers = launcher_ids(old_members)  # None: an incomplete census, never "no launcher"
        since = time.time_ns()
        status, body = self.install.http("POST", "/api/update/apply",
                                         {"strategy": "activate", "candidate_id": candidate_id}, timeout=180)
        adoption, facts = activation_request(status, body, candidate_id)
        case.require("activation_accepted", None if status is None else bool(adoption), **facts)
        end, trace, closed, record, server = time.monotonic() + DEADLINE["settle"], [], None, None, None
        while time.monotonic() < end:
            handoff = self.install.handoff()
            if isinstance(handoff, dict) and handoff.get("id") == adoption and (not trace or trace[-1] != handoff):
                trace.append(handoff)
            closed = closed_adoption(self.install.supervisor_rows(), adoption)
            record = self.install.candidate(candidate_id)
            if closed and record and record.get("state") != "activating":
                server = self.install.wait_ready(self.custody, DEADLINE["ready"],
                                                 exclude=[(old_server["pid"], old_server["birth"])] if old_server else [])
                break
            time.sleep(1)
        self.snapshot(f"{case.name}-activation-{adoption}", response=body, trace=trace, closed=closed, record=record)
        case.require(f"settled_{expect}", settled(record, closed, expect), closed=closed,
                     activation=(record or {}).get("activation"), state=(record or {}).get("state"))
        composition = str(((record or {}).get("layers") or {}).get("new_commit") or "")
        case.check("attempt_bound_to_candidate_composition", bool(HEX40.match(composition))
                   and (closed or {}).get("candidate") == composition, composition=composition,
                   closed_candidate=(closed or {}).get("candidate"))
        if expect == "adopted":
            head = self.head()
            case.check("adopted_composition", head == composition, head=head, composition=composition)
        case.require("ready_after_settlement", _ready(self.install, server), server=server)
        return {"adoption": adoption, "since": since, "closed": closed, "record": record, "server": server,
                "old_server": old_server, "old_launchers": old_launchers, "old_members": old_members, "trace": trace}

    def fresh_generation(self, case, result):
        """A switched package path makes the launcher hand itself over: launcher and server are new processes,
        judged only between measured identities (a census or server read that is missing leaves it ``None``)."""
        members, before = self.custody.members(), result["old_launchers"]
        launchers = launcher_ids(members)
        case.check("fresh_launcher", None if launchers is None or before is None else bool(launchers)
                   and not (launchers & before), before=None if before is None else sorted(before),
                   after=None if launchers is None else sorted(launchers), unread=unread(members))
        old, new = result["old_server"], result["server"]
        case.check("fresh_server", None if old is None or new is None else (new["pid"], new["birth"])
                   != (old["pid"], old["birth"]), server=new, old=old)

    def selected(self, case, result, prepared, fixed, serving_id):
        """The prepared directory (by identity) is now at the fixed path; the serving one kept aside."""
        previous = pathlib.Path(f"{fixed}.previous-{result['adoption']}")
        case.check("prepared_directory_selected", path_id(fixed) == prepared["id"] is not None,
                   fixed=str(fixed), now=path_id(fixed), prepared=prepared["id"])
        if serving_id is not None:
            case.check("serving_directory_kept_aside", path_id(previous) == serving_id, previous=str(previous),
                       now=path_id(previous), serving=serving_id)

    def layers_carried(self, case, head=None):
        """H, I, W and the binary U the owner left are where they were, around whatever HEAD now is."""
        expected, repo, env = self.expected_layers, self.install.repo, self.env()
        if expected is None:
            return
        current = git(repo, "rev-parse", "HEAD", env=env)
        ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", expected["commit"], current],
                                  env={**env, "GIT_OPTIONAL_LOCKS": "0"}, capture_output=True).returncode == 0
        staged = git(repo, "ls-files", "--stage", "--", SENTINELS[0], env=env)
        untracked = git(repo, "status", "--porcelain=v2", "--untracked-files=all", "--", SENTINELS[1], env=env)
        case.check("owner_layers_carried", ancestor and (head is None or current == head)
                   and expected["index_blob"] in staged and file_digest(repo / SENTINELS[0]) == expected["worktree"]
                   and file_digest(repo / SENTINELS[1]) == expected["untracked"] and untracked.startswith("? "),
                   head=current, staged=staged, untracked=untracked)

    def seed_layers(self, case):
        repo, env = self.install.repo, self.env(**fx.IDENTITY)
        (repo / "qualification").mkdir(exist_ok=True)
        (repo / SENTINELS[0]).write_bytes(H_TEXT)
        git(repo, "add", "--", SENTINELS[0], env=env)
        git(repo, "commit", "--quiet", "--no-verify", "-m", "owner: a local commit before the update", env=env)
        commit = git(repo, "rev-parse", "HEAD", env=env)
        (repo / SENTINELS[0]).write_bytes(I_TEXT)
        git(repo, "add", "--", SENTINELS[0], env=env)
        (repo / SENTINELS[0]).write_bytes(W_TEXT)
        (repo / SENTINELS[1]).write_bytes(U_BYTES)
        blob = subprocess.run(["git", "hash-object", "--stdin"], input=I_TEXT, capture_output=True, check=True,
                              env={**env, "GIT_OPTIONAL_LOCKS": "0"}).stdout.decode().strip()
        self.expected_layers = {"commit": commit, "index_blob": blob, "worktree": file_digest(repo / SENTINELS[0]),
                                "untracked": file_digest(repo / SENTINELS[1])}
        case.check("owner_layers_seeded", True, **self.expected_layers)

    def cold(self, case, starter, server, fixed, expected_version, site, prefix, released=False):
        """Stop every attributed process, prove the endpoint gone, start the same command again."""
        before = {"id": path_id(fixed), "inventory": inventory(site), "head": self.head()}
        seen = identities(self.custody.members())  # None: an incomplete census, never "no process"
        stop = self.custody.stop_all(f"{case.name}: cold entry")  # the harness's own stop: unconfirmed is unmeasured
        case.require("previous_processes_stopped", True if stop["confirmed"] is True else None, **stop)
        state = self.custody.alive(server)
        case.require("previous_endpoint_gone", None if state is None else state is False
                     and self.install.endpoint_gone(server),
                     server=server)
        escaped, scan = self.custody.escaped()
        case.require("no_process_outside_custody", None if escaped is None else escaped == [], escaped=escaped,
                     scan=scan)
        proc, _root, _ = starter()
        cold = self.install.wait_ready(self.custody, DEADLINE["first_ready"], exclude=seen or ())
        case.require("cold_ready", _ready(self.install, cold), server=cold)
        if released:  # the unchanged released app hands over again, to the switched checkout
            try:
                rc = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                rc = None
            members = self.custody.members()
            roles = known_roles(members)
            case.check("released_app_handed_over_again", conjoin(None if rc is None else rc == 0, None if roles is None
                       else any(str(self.install.repo).lower() in str(row.get("cmdline")).lower()
                                for row in roles["launcher"])), returncode=rc,
                       launchers=None if roles is None else roles["launcher"], unread=unread(members))
        now = identities(self.custody.members())
        case.check("all_processes_new", None if now is None or seen is None else bool(now) and not (now & seen),
                   before=None if seen is None else sorted(seen), after=None if now is None else sorted(now))
        case.check("same_composition", self.head() == before["head"], head=before["head"])
        case.check("no_dependency_reconstruction", path_id(fixed) == before["id"] is not None
                   and same_inventory(before["inventory"], inventory(site)), before=before)
        case.check("cold_server_imports_selected_generation",
                   generation_seen(self.server_witness(cold), expected_version, site, prefix),
                   witness=self.server_witness(cold))
        self.markers_kept(case)
        self.layers_carried(case)
        return cold


# ---------------------------------------------------------------------- mode runners
def run_mode(ctx, label, selected):
    mode = Mode(ctx, label)
    cases = {name: Case(name, ctx.raw.emit, REQUIRED_CHECKS[name]) for name in MODES[label]}
    for name, case in cases.items():
        if name not in selected:
            case.not_run("not selected (partial run)")
    teardown = {"status": "confirmed", "note": "custody never opened: no process of this mode was started"}
    try:
        mode.provision()
        mode.custody = Custody(ctx.win, ctx.raw, label, [mode.root, ctx.release["dir"], ctx.bare_root])
        {"U": _user_site, "D": _runtime_venv, "F": _bare_python}[label](mode, cases, selected)
    except Exception as exc:  # equipment or provisioning failure: never a product verdict
        ctx.raw.emit("mode_error", mode=label, error=repr(exc))
        for case in cases.values():
            if not case.checks and case.forced is None:
                case.not_run(f"mode {label} stopped before this case: {exc!r}")
    finally:
        teardown = _close_and_export(mode, cases, teardown)
    return cases, teardown


def _close_and_export(mode, cases, teardown):
    """Close the mode's Job, then export its evidence, on every path; a case that ran gets its export check.
    The export is confirmed only after the teardown's drain of the holders' output completed: a holder can be
    gone while its last bytes are still being transcribed (a teardown without a drain record is undrained)."""
    drain = None  # custody never opened: no holder was started, nothing is being transcribed
    if mode.custody is not None:
        try:
            teardown = mode.custody.close()
        except Exception as exc:  # close() never raises; a broken one still leaves custody unknown
            teardown = {"status": "unknown", "close_error": repr(exc)}
        drain = teardown.get("drain") if isinstance(teardown.get("drain"), dict) else {"complete": None}
    exported = mode.export()
    if drain is not None and drain.get("complete") is not True:
        exported = {**exported, "ok": False,
                    "errors": [*(exported.get("errors") or []), f"holder output not drained: {drain}"]}
    teardown = {**teardown, "export": exported}
    for case in cases.values():
        if case.checks:  # a case that measured anything is auditable only with its evidence
            case.required = (*case.required, *(name for name in TERMINAL_CHECKS if name not in case.required))
            case.check("evidence_exported", True if exported.get("ok") is True else None,
                       **{key: exported.get(key) for key in ("copied", "errors", "missing_required")})
    return teardown


def _sequence(mode, cases, selected, steps):
    """Run ``steps`` (case name -> function) in order; a case after a non-PASS one is NOT_RUN."""
    blocked = ""
    for name, step in steps:
        case = cases[name]
        if name not in selected:
            blocked = blocked or f"{name} not selected"
            continue
        if blocked:
            case.not_run(f"needs {blocked}")
            continue
        try:
            step(case)
        except CaseStop:
            pass
        except Exception as exc:  # an equipment or read failure mid-case: never leaves earlier checks to pass
            case.check("case_completed", None, error=repr(exc))
        if case.status != PASS:
            blocked = f"{name} to pass (it is {case.status})"


def _user_site(mode, cases, selected):
    internal = mode.ctx.release["internal"]
    prefix, fixed = internal / "python-standalone", mode.install.user_base
    site = user_site(fixed)
    state = {}

    def entry(case):
        state["server"] = mode.entry(case, (), site, prefix)

    def case_a(case):
        mode.channel_and_marker(case, "winqual-a-before")
        mode.seed_layers(case)
        serving_id = path_id(fixed)
        holder = mode.hold("A-pyd-holder", [mode.ctx.release["python"], "-m", fx.PACKAGE, "hold", "--native"],
                           PYTHONUSERBASE=fixed)
        ready = holder["ready"] or {}
        native = ready.get("native") or {}
        case.require("native_holder_acknowledged", None if not ready else ready.get("event") == "ready" and (
            ready.get("package") or {}).get("version") == "1.0" and native.get("status") == "loaded"
            and under(native.get("realpath"), site) and bool(native.get("mapped_device")), ready=ready)
        prepared = mode.prepare(case, "T2", "user_site", fixed, site, state["server"])
        mode.marker(case, "winqual-a-after")
        case.require("holder_alive_before_switch", mode.custody.alive(holder["identity"]))
        result = mode.activate(case, prepared["candidate_id"], "adopted")
        alive = mode.custody.alive(holder["identity"])
        report = holder["io"].request("report") if alive else None  # an unanswered holder is unmeasured
        case.check("holder_survived_switch", None if alive is None or (alive and report is None) else alive and (
            (report.get("package") or {}).get("version") == "1.0"), report=report)
        mode.selected(case, result, prepared, fixed, serving_id)
        mode.fresh_generation(case, result)
        case.check("new_server_imports_T2_generation", generation_seen(mode.server_witness(result["server"],
                   result["since"]), "2.0", site, prefix), witness=mode.server_witness(result["server"]))
        mode.layers_carried(case)
        mode.markers_kept(case)
        reply, exited = mode.release_holder(holder)
        case.check("holder_released", exited, reply=reply)
        state["server"] = result["server"]

    def case_b(case):
        prepared = mode.prepare(case, "T3", "user_site", fixed, site, state["server"])
        mode.marker(case, "winqual-b-after")
        package_dir = site / fx.PACKAGE
        holder = mode.hold("B-cwd-holder", [mode.ctx.release["python"], "-m", fx.PACKAGE, "hold", "--cwd", package_dir],
                           PYTHONUSERBASE=fixed)
        ready = holder["ready"] or {}
        case.require("cwd_holder_acknowledged", None if not ready else ready.get("event") == "ready" and same_path(
            ready.get("cwd_final") or ready.get("cwd"), package_dir), ready=ready)
        before = {"layers": mode.layers(), "id": path_id(fixed),
                  "inventory": inventory(site), "generation": path_id(prepared["path"])}
        launchers = launcher_ids(mode.custody.members())  # None: an incomplete census, never "no launcher"
        refused = mode.activate(case, prepared["candidate_id"], "returned_retry")
        decision = (refused["record"].get("activation") or {}).get("decision") or {}
        error = decision.get("error") or {}
        case.check("refusal_names_system_error", error.get("step") == "serving_aside" and error.get("winerror")
                   is not None, error=error, detail=decision.get("detail"))
        case.check("holder_alive_through_attempt", mode.custody.alive(holder["identity"]))
        same, changed = layers_equal(before["layers"], mode.layers())
        case.check("coherent_previous_generation", None if same is None else same and path_id(fixed) == before["id"]
                   and same_inventory(before["inventory"], inventory(site))
                   and path_id(prepared["path"]) == before["generation"] == prepared["id"], **changed)
        case.check("previous_generation_serves", generation_seen(mode.server_witness(refused["server"], refused["since"]),
                   "2.0", site, prefix), witness=mode.server_witness(refused["server"]))
        kept = launcher_ids(mode.custody.members())
        case.check("launcher_not_replaced_without_switch", None if launchers is None or kept is None
                   else bool(launchers) and kept == launchers, before=None if launchers is None else sorted(launchers),
                   after=None if kept is None else sorted(kept))
        time.sleep(20)  # a crash loop would replace the serving process within this window
        stable, now = _still_serving(mode, refused["server"])  # one read, pid AND birth
        case.check("no_crash_loop", stable, server=now, serving=refused["server"])
        mode.markers_kept(case)
        reply, exited = mode.release_holder(holder)
        case.require("holder_released", exited, reply=reply)
        retried = mode.activate(case, prepared["candidate_id"], "adopted")
        case.check("retry_is_a_new_attempt", retried["adoption"] != refused["adoption"],
                   first=refused["adoption"], second=retried["adoption"])
        mode.selected(case, retried, prepared, fixed, before["id"])
        mode.fresh_generation(case, retried)
        case.check("new_server_imports_T3_generation", generation_seen(mode.server_witness(retried["server"],
                   retried["since"]), "3.0", site, prefix), witness=mode.server_witness(retried["server"]))
        mode.layers_carried(case)
        mode.markers_kept(case)
        state["server"] = retried["server"]

    def case_c(case):
        prepared = mode.prepare(case, "T4", "user_site", fixed, site, state["server"])
        mode.marker(case, "winqual-c-after")
        before = {"layers": mode.layers(), "id": path_id(fixed),
                  "inventory": inventory(site)}
        result = mode.activate(case, prepared["candidate_id"], "returned")
        detail = str(((result["record"].get("activation") or {}).get("decision") or {}).get("detail") or "")
        case.check("intended_dependency_failure", "cannot import name 'stable_api'" in detail and fx.PACKAGE in detail,
                   detail=detail)
        same, changed = layers_equal(before["layers"], mode.layers())
        case.check("coherent_previous_composition", None if same is None else same and path_id(fixed) == before["id"]
                   and same_inventory(before["inventory"], inventory(site))
                   and path_id(prepared["path"]) == prepared["id"], **changed)
        case.check("previous_code_imports_previous_package", generation_seen(
            mode.server_witness(result["server"], result["since"]), "3.0", site, prefix),
            witness=mode.server_witness(result["server"]))
        rc, out = mode.run_bounded("C-fresh-observer", [mode.ctx.release["python"], "-c",
                                   f"import json, {fx.PACKAGE} as q; print(json.dumps(q.facts('observer')))"],
                                   DEADLINE["tool"], PYTHONUSERBASE=fixed)
        observed = _last_json(out)
        case.check("fresh_process_imports_restored_package", None if rc is None else rc == 0
                   and generation_seen(observed, "3.0", site), returncode=rc, observed=observed)
        mode.markers_kept(case)
        state["server"] = result["server"]

    def case_e(case):
        mode.cold(case, lambda: mode.start_released("released-app-cold"), state["server"], fixed, "3.0", site, prefix,
                  released=True)

    _sequence(mode, cases, selected, (("entry.U", entry), ("A", case_a), ("B", case_b), ("C", case_c),
                                      ("E.U", case_e)))


def _runtime_venv(mode, cases, selected):
    venv = mode.install.app_root / "runtime venv"
    site, scripts, state = venv_site(venv), venv / "Scripts", {}
    base = mode.ctx.release["internal"] / "python-standalone"

    def entry(case):
        state["server"] = mode.entry(case, ("--runtime-venv", venv), site, venv)
        marker = _read_json(venv / "ouroboros-runtime-venv.json")
        case.check("runtime_venv_made_by_product", (marker or {}).get("state") == "ready", marker=marker)
        start = _start_of(mode, state["server"])  # the equipment's own witness: a missing one is unmeasured
        case.check("server_runs_in_venv_on_app_python", None if start is None else same_path(start.get("prefix"), venv)
                   and same_path(start.get("base_prefix"), base), start=start)

    def case_d(case):
        mode.channel_and_marker(case, "winqual-d-before")
        serving_id = path_id(venv)
        redirector = mode.hold("D-venv-python-holder", [scripts / "python.exe", "-m", fx.PACKAGE, "hold", "--native"],
                               PYTHONNOUSERSITE="1")
        image = mode.hold("D-console-image-holder", [scripts / "ouroboros-qual-tool.exe", "hold", "--native"],
                          PYTHONNOUSERSITE="1")
        members = mode.custody.members()
        images = [str(row.get("image")) for row in members or []]
        unseen = members is None or any(row.get("alive") is None for row in members)  # an unread one may be it
        for name, holder, path in (("venv_python", redirector, scripts / "python.exe"),
                                   ("console_image", image, scripts / "ouroboros-qual-tool.exe")):
            ready = holder["ready"] or {}
            seen = True if any(same_path(found, path) for found in images) else (None if unseen else False)
            case.require(f"{name}_holder_acknowledged", None if not ready else conjoin(
                ready.get("event") == "ready" and same_path(ready.get("prefix"), venv)
                and (ready.get("native") or {}).get("status") == "loaded", seen), ready=ready, images=images)
        prepared = mode.prepare(case, "T5", "venv", venv, site, state["server"])
        tool = prepared["path"] / "Scripts" / "ouroboros-qual-tool.exe"
        before_parts = console_launcher_parts(tool.read_bytes()) if tool.is_file() else None
        case.require("prepared_console_launcher_names_generation", before_parts is not None and shebang_names(
            before_parts[1], prepared["path"] / "Scripts" / "python.exe"), shebang=before_parts and before_parts[1])
        prepared_dist = ((prepared["inventory"] or {}).get(fx.DIST) or {}).get("id")
        mode.marker(case, "winqual-d-after")
        result = mode.activate(case, prepared["candidate_id"], "adopted")
        # The product's own holder: v7.8.0 hands over to the source launcher on ``--runtime-venv``'s interpreter
        # (its launcher.py sets EMBEDDED_PYTHON to the venv's python before ``_exec_source_launcher``), so the
        # launcher alive at this activation runs from the venv's Scripts\python.exe redirector. It waits on the
        # server that performs the switch inside its kill-on-close Job: it holds that image through the rename.
        members = result.get("old_members")  # the census the activation read just before its request
        roles = known_roles(members)
        own = [row for row in (roles or {}).get("launcher", []) if row.get("alive")
               and same_path(row.get("image"), scripts / "python.exe")]
        case.check("product_launcher_runs_from_venv_redirector", None if roles is None else bool(own),
                   launchers=None if roles is None else roles["launcher"], unread=unread(members))
        mode.selected(case, result, prepared, venv, serving_id)
        after_parts = console_launcher_parts((scripts / "ouroboros-qual-tool.exe").read_bytes())
        case.check("console_launcher_relocated_only_in_its_shebang", after_parts is not None
                   and after_parts[0] == before_parts[0] and after_parts[2] == before_parts[2]
                   and shebang_names(after_parts[1], scripts / "python.exe"), shebang=after_parts and after_parts[1])
        case.check("not_reinstalled", ((inventory(site) or {}).get(fx.DIST) or {}).get("id") == prepared_dist is not None,
                   prepared=prepared_dist, now=inventory(site))
        rc, out = mode.run_bounded("D-relocated-tool", [scripts / "ouroboros-qual-tool.exe", "report"], DEADLINE["tool"],
                                   PYTHONNOUSERSITE="1")
        report = _last_json(out)
        case.check("relocated_console_script_runs_new_generation", None if rc is None else rc == 0
                   and generation_seen(report, "5.0", site, venv)
                   and same_path((report or {}).get("executable"), scripts / "python.exe"), returncode=rc, report=report)
        for name, holder in (("venv_python", redirector), ("console_image", image)):
            case.check(f"{name}_holder_survived", mode.custody.alive(holder["identity"]))
        mode.fresh_generation(case, result)
        case.check("new_server_imports_T5_generation", generation_seen(mode.server_witness(result["server"],
                   result["since"]), "5.0", site, venv), witness=mode.server_witness(result["server"]))
        mode.markers_kept(case)
        for name, holder in (("venv_python", redirector), ("console_image", image)):
            reply, exited = mode.release_holder(holder)
            case.check(f"{name}_holder_released", exited, reply=reply)
        state["server"] = result["server"]

    def case_e(case):
        mode.cold(case, lambda: mode.start_released("released-app-cold", ("--runtime-venv", venv)), state["server"],
                  venv, "5.0", site, venv, released=True)
        rc, out = mode.run_bounded("E.D-tool", [scripts / "ouroboros-qual-tool.exe", "report"], DEADLINE["tool"],
                                   PYTHONNOUSERSITE="1")
        case.check("console_script_after_cold_entry", None if rc is None else rc == 0
                   and generation_seen(_last_json(out), "5.0", site, venv), returncode=rc)

    _sequence(mode, cases, selected, (("entry.D", entry), ("D", case_d), ("E.D", case_e)))


def _bare_python(mode, cases, selected):
    python, bare = mode.ctx.bare_python, mode.ctx.bare_root
    global_site, private = bare / "Lib" / "site-packages", mode.install.private
    private_site, state = venv_site(private), {}
    roaming = mode.root / "home" / "AppData" / "Roaming" / "Python"
    launcher = [python, mode.install.repo / "launcher.py", "--no-ui", "--launch-intent", "owner",
                "--seed-bundle", mode.ctx.release["internal"]]

    def case_f(case):
        rc, _ = mode.run_bounded("F-owner-installs-base-lock", [python, "-m", "pip", "install", "-r",
                                 mode.install.repo / fx.LOCK], 2400, cwd=mode.install.repo)
        case.require("owner_installed_base_lock_into_interpreter", None if rc is None else rc == 0, returncode=rc)
        baseline = content_inventory(global_site)
        case.require("global_packages_measured", baseline is not None, files=len(baseline or {}))
        state["baseline"] = baseline
        mode.snapshot("F-global-baseline", content=baseline)
        mode.start("source-launcher", launcher)
        server = mode.install.wait_ready(mode.custody, DEADLINE["first_ready"])
        case.require("server_ready", _ready(mode.install, server), server=server)
        case.check("server_on_bare_interpreter", generation_seen(mode.server_witness(server), "1.0", global_site, bare),
                   witness=mode.server_witness(server))
        mode.channel_and_marker(case, "winqual-f-before")
        prepared = mode.prepare(case, "T5", "private", private, global_site, server)
        record_generation = (prepared["record"].get("environment") or {}).get("generation") or {}
        marker = _read_json(prepared["path"] / "ouroboros-private-environment.json")
        config = (prepared["path"] / "pyvenv.cfg").read_text(encoding="utf-8", errors="replace") \
            if (prepared["path"] / "pyvenv.cfg").is_file() else ""
        case.check("private_environment_made_by_product", same_path((marker or {}).get("repo_dir"), mode.install.repo)
                   and str(bare).lower() in config.lower(), marker=marker, pyvenv=config, pip=record_generation.get("pip"))
        changes = content_changes(baseline, content_inventory(global_site))
        case.check("global_packages_untouched_by_preparation", None if changes is None else changes == [],
                   changes=(changes or [])[:50])
        mode.marker(case, "winqual-f-after")
        result = mode.activate(case, prepared["candidate_id"], "adopted")
        mode.selected(case, result, prepared, private, None)
        mode.fresh_generation(case, result)
        witness = mode.server_witness(result["server"], result["since"])
        case.check("server_runs_in_private_environment", generation_seen(witness, "5.0", private_site, private)
                   and same_path((witness or {}).get("base_prefix"), bare), witness=witness)
        case.check("server_is_the_base_interpreter_itself", same_path(result["server"].get("image"), python),
                   image=result["server"].get("image"))
        changes = content_changes(baseline, content_inventory(global_site))
        case.check("global_packages_unchanged", None if changes is None else changes == [] and not roaming.exists(),
                   changes=(changes or [])[:50], roaming_user_site=roaming.exists())
        mode.markers_kept(case)
        state["server"] = result["server"]

    def case_e(case):
        server = mode.cold(case, lambda: mode.start("source-launcher-cold", launcher), state["server"], private,
                           "5.0", private_site, private)
        stop = mode.custody.stop_all("E.F: before the direct entry")  # the harness's stop: unconfirmed is unmeasured
        case.require("source_launcher_stopped", None if stop["confirmed"] is not True
                     else mode.install.endpoint_gone(server), stop=stop)
        _proc, direct, _ = mode.start("direct-server", [python, "server.py"], cwd=mode.install.repo)
        served = mode.install.wait_ready(mode.custody, DEADLINE["first_ready"])
        case.require("direct_entry_ready", _ready(mode.install, served), server=served)
        members = mode.custody.members()
        child = next((row for row in members or [] if row.get("pid") == served["pid"]), {})
        parent_alive = mode.custody.alive(direct)
        case.check("direct_command_is_waited_parent", None if parent_alive is None or members is None
                   or (child and child.get("alive") is None) else child.get("ppid") == direct["pid"] and parent_alive,
                   child=child, parent=direct)
        case.check("direct_entry_runs_in_private_environment", generation_seen(mode.server_witness(served), "5.0",
                   private_site, private), witness=mode.server_witness(served))
        rc, _ = mode.run_bounded("module-cli-help", [python, "-m", "ouroboros.cli", "--help"], DEADLINE["tool"],
                                 cwd=mode.install.repo)
        case.check("module_cli_entry_exits_0", None if rc is None else rc == 0, returncode=rc)
        rc, out = mode.run_bounded("module-entry-facts", [python, "-c", "import ouroboros, json, " + fx.PACKAGE
                                   + " as q; print(json.dumps(q.facts('module-entry', with_native=False)))"],
                                   DEADLINE["tool"], cwd=mode.install.repo)
        facts = _last_json(out)
        case.check("module_entry_redirected_before_imports", None if rc is None else rc == 0
                   and generation_seen(facts, "5.0", private_site, private), returncode=rc, facts=facts)
        changes = content_changes(state["baseline"], content_inventory(global_site))
        case.check("global_packages_unchanged", None if changes is None else changes == [] and not roaming.exists(),
                   changes=(changes or [])[:50], roaming_user_site=roaming.exists())

    _sequence(mode, cases, selected, (("F", case_f), ("E.F", case_e)))


# ---------------------------------------------------------------------- small readers
def _ready(install, server):
    """What ``Install.wait_ready`` measured: ``True`` for a ready server; ``False`` only for a terminal answer
    (the last health of a new server said its boot failed); else ``None``: the deadline passed unmeasured."""
    if server is not None:
        return True
    return False if getattr(install, "boot_failed", None) is True else None


def _still_serving(mode, serving):
    """``(same, now)`` from ONE read of the bound server: whether ``now`` is still ``serving`` by pid AND birth
    (a new birth on the same pid is another process). With no bound server read, ``same`` is ``False`` only
    when ``serving`` is measured gone, else ``None``."""
    now = mode.install.server(mode.custody)
    if not isinstance(serving, dict):
        return None, now
    if now is not None:
        return (now["pid"], now["birth"]) == (serving["pid"], serving["birth"]), now
    return (False if mode.custody.alive(serving) is False else None), None


def _read_json(path):
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _last_json(text):
    for line in reversed(str(text or "").splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _start_of(mode, server):
    """The interpreter-start witness (sitecustomize, before any hook) of exactly that process."""
    rows = [row for row in mode.install.witnesses() if row.get("schema") == "winqual-start/1"
            and row.get("pid") == server.get("pid") and str(row.get("birth")) == str(server.get("birth"))]
    return rows[-1] if rows else None
