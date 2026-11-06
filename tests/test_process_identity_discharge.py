"""A recorded identity is discharged only on positive evidence (lifecycle identity repair).

Each closed false discharge keeps its positive control. A Linux boot time or a birth zone that
cannot be read spells no birth: never epoch zero, a guessed UTC or the server's own ``TZ``. A
recorded group leader that a partial ``/proc`` table omits is read by name. Unreadable keeps
it owed. Alive with its birth, it stays reachable wherever its group went. Absence, a corpse
or another birth resolves it without signalling a replacement. One birth has one spelling:
``ps`` pads a one-digit day, the engine's index collapses it. A cleanup claim never recreates
a retired record and never claims a rotated one.

Synthetic kernel answers live only in ``process_census``'s own ``open`` and ``os`` names;
signals go to a recorder. The real children are this test's own and end by exact pid.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import re
import subprocess
import sys
import time
import zoneinfo
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import platform_layer as _pl
from ouroboros import process_census as pc
from ouroboros import process_scope as ps
from ouroboros import task_results

pytestmark = [pytest.mark.skipif(os.name == "nt", reason="the environment census is POSIX-only")]

PID, OTHER, MEMBER = 120001, 120002, 120005
BOOT, HZ, TICKS = 1_700_000_000, 100, 98_765
ENGINE_BIRTH, HOST_BIRTH = "Thu Oct 1 12:00:00 2026", "Thu Oct  1 12:00:00 2026"  # one instant, two spellings
OTHER_BIRTH = "Fri Oct  2 12:00:00 2026"  # another process on the same pid


def _stat(pid: int, *, pgid: int = 0, ticks: int = TICKS, state: str = "S") -> bytes:
    pgid = pgid or pid
    return (f"{pid} (worker) {state} 1 {pgid} {pgid} 0 -1 4194560 100 0 0 0 5 3 0 0 20 0 1 0 "
            f"{ticks} 1000000 50").encode()


def _env(**extra: str) -> dict:
    """This process's environment without its own ``TZ`` key, plus ``extra``."""
    return {**{key: value for key, value in os.environ.items() if key != "TZ"}, **extra}


class _Kernel:
    """Linux-shaped kernel answers, seen only through ``process_census``'s own ``open`` and
    ``os`` names: the real ``os`` module, other modules and threads are untouched. A stat
    answer is bytes, an exception, or a list read in turn (the last one repeats)."""

    def __init__(self, monkeypatch, stats, *, btime=b"btime %d\n" % BOOT, localtime=None, environ=None):
        self.stats, self.btime, self.localtime = dict(stats), btime, localtime
        self.environ = _env() if environ is None else environ
        self.path = SimpleNamespace(isdir=lambda path: str(path) == "/proc")
        monkeypatch.setattr(pc, "open", self.open, raising=False)
        monkeypatch.setattr(pc, "os", self)
        monkeypatch.setattr(pc, "_BOOT_EPOCH", [])
        monkeypatch.setattr(pc, "_BIRTH_ZONE", [])

    def __getattr__(self, name):
        return getattr(os, name)

    def listdir(self, path):
        assert str(path) == "/proc", path
        return [str(pid) for pid in self.stats]

    def sysconf(self, name):
        assert name == "SC_CLK_TCK", name
        return HZ

    def open(self, path, *args, **kwargs):
        path = str(path)
        if path == "/proc/stat":
            return self._serve(self.btime, path)
        if path == "/etc/localtime":
            return self._serve(self.localtime, path)
        match = re.fullmatch(r"/proc/(\d+)/(stat|environ)", path)
        assert match, f"unexpected read {path}"
        if int(match.group(1)) not in self.stats:
            raise FileNotFoundError(2, "no such process", path)
        return io.BytesIO(b"") if match.group(2) == "environ" else self._serve(self.stats[int(match.group(1))], path)

    @staticmethod
    def _serve(answer, path):
        if isinstance(answer, list):
            answer = answer.pop(0) if len(answer) > 1 else answer[0]
        if answer is None:
            raise FileNotFoundError(2, "absent", path)
        if isinstance(answer, BaseException):
            raise answer
        return io.BytesIO(answer)


def _signals(monkeypatch, kernel=None, *, result=_pl.SIGNAL_SENT) -> list:
    """Record every signal; an accepted one ends the synthetic process. No host process is signalled."""
    sent = []

    def deliver(pid, *, graceful):
        sent.append(pid)
        if kernel is not None and result == _pl.SIGNAL_SENT:
            kernel.stats.pop(pid, None)
        return result

    monkeypatch.setattr(_pl, "deliver_signal", deliver)
    monkeypatch.setattr(ps, "_RESCAN_SETTLE_SEC", 0)
    monkeypatch.setattr(ps, "_KILL_CONFIRM_SEC", 0.0)
    return sent


def _pending(root: Path, members: list) -> Path:
    path = ps.panic_sweep_pending_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"requested_at": "test", "excluded": [], "known_members": members}), encoding="utf-8")
    return path


def _zone_file(name: str = "Pacific/Honolulu") -> bytes:
    for base in zoneinfo.TZPATH:
        if (Path(base) / name).is_file():
            return (Path(base) / name).read_bytes()
    pytest.skip(f"no {name} zone file on this host")


def _spelled(epoch: int, zone) -> str:
    """The C-locale ``lstart`` spelling of ``epoch`` in ``zone`` (ps pads a one-digit day)."""
    when = dt.datetime.fromtimestamp(epoch, zone)
    return (f"{pc._WEEKDAYS[when.weekday()]} {pc._MONTH_NAMES[when.month - 1]} {when.day:>2} "
            f"{when:%H:%M:%S} {when.year}")


# --------------------------------------------------------------------------- 1. boot time

@pytest.mark.parametrize("btime", [PermissionError(13, "denied"), b"cpu 1 2 3\n", b"btime \n"], ids=["unreadable", "missing", "empty"])
def test_no_boot_time_spells_no_birth_in_either_reader_and_a_later_read_recovers(tmp_path, monkeypatch, btime):
    kernel = _Kernel(monkeypatch, {PID: _stat(PID)}, btime=btime)
    sent = _signals(monkeypatch, kernel)
    birth = pc.canonical_birth(pc.lstart_text(BOOT + TICKS // HZ))  # what Panic saved for the live process
    assert pc._proc_identity(PID).state == pc.IDENTITY_UNREADABLE, "present but unspellable: never a 1970 birth"
    assert pc.read_identities([PID])[PID].state == pc.IDENTITY_UNREADABLE
    assert pc.census() is None and pc.census(["OURO_TASK_SCOPE_" + "a" * 32]) is None, "unknown, never an empty table"
    assert pc._BOOT_EPOCH == [], "a failed read is not cached"
    pending = _pending(tmp_path, [{"pid": PID, "birth": birth}])
    result = ps.complete_pending_panic_sweep(tmp_path)
    assert (result["settled"], result["captured_gone"], result["ended"], sent) == (False, [], [], [])
    assert result["unverified"] == [{"pid": PID, "birth": birth, "reason": ps.UNVERIFIED_IDENTITY}]
    assert result["pending_retired"] is False and pending.is_file()
    assert json.loads(pending.read_text(encoding="utf-8"))["known_members"] == [{"pid": PID, "birth": birth}]
    kernel.btime = b"btime %d\n" % BOOT  # readable again: the same saved identity is judged and ended
    result = ps.complete_pending_panic_sweep(tmp_path)
    assert sent == [PID] and result["ended"] == [PID] and result["settled"] is True
    assert result["pending_retired"] is True and not pending.exists() and pc._BOOT_EPOCH == [float(BOOT)]


def test_a_readable_boot_time_spells_the_recorded_birth_and_absence_needs_none(tmp_path, monkeypatch):
    kernel = _Kernel(monkeypatch, {PID: _stat(PID), OTHER: _stat(OTHER, ticks=TICKS + 500)})
    birth = pc.canonical_birth(pc.lstart_text(BOOT + TICKS // HZ))
    identity = pc._proc_identity(PID)
    assert (identity.state, identity.birth, identity.pgid) == (pc.IDENTITY_ALIVE, birth, PID)
    rows = pc.census()
    assert set(rows) == {PID, OTHER} and rows[PID].birth == birth and rows[PID].epoch == BOOT + TICKS // HZ
    kernel.btime = PermissionError(13, "denied")
    monkeypatch.setattr(pc, "_BOOT_EPOCH", [])
    del kernel.stats[PID]  # the kernel no longer has the pid: genuine absence needs no boot time
    assert pc._proc_identity(PID).state == pc.IDENTITY_ABSENT
    sent = _signals(monkeypatch, kernel)
    pending = _pending(tmp_path, [{"pid": PID, "birth": birth}])
    result = ps.complete_pending_panic_sweep(tmp_path)
    assert result["captured_gone"] == [PID] and result["settled"] is True and sent == []
    assert result["pending_retired"] is True and not pending.exists()


# --------------------------------------------------------------------------- 2. the birth zone

@pytest.mark.parametrize("tz", ["", "UTC", "Asia/Tokyo"])
def test_a_tz_keyed_server_spells_and_parses_births_in_the_os_zone_never_its_own(monkeypatch, tz):
    """The engine's ps env carries no ``TZ``: it prints in the OS zone. A server whose own ``TZ``
    key (an empty one too) differs must spell, parse and compare in that same zone."""
    kernel = _Kernel(monkeypatch, {PID: _stat(PID)}, localtime=_zone_file(), environ=_env(TZ=tz))
    honolulu, epoch = zoneinfo.ZoneInfo("Pacific/Honolulu"), BOOT + TICKS // HZ
    expected = _spelled(epoch, honolulu)
    assert pc.lstart_text(epoch) == expected and pc.birth_epoch(expected) == epoch
    assert pc._proc_identity(PID).birth == pc.canonical_birth(expected) == pc.census()[PID].birth
    assert len(pc._BIRTH_ZONE) == 1, "proven once, then cached"
    assert ps._iso_floor(dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat()) == expected
    assert "TZ" not in pc._ps_env() and pc._ps_env()["LC_ALL"] == "C", "our ps child prints like the engine's"
    engine = pc.canonical_birth(expected)  # what the engine's TZ-less ps recorded for this leader
    own = pc.canonical_birth(_spelled(epoch, zoneinfo.ZoneInfo(tz or "UTC")))  # the server's own TZ spelling
    sent = _signals(monkeypatch, kernel)
    outcome = ps.end_scope(groups=[{"pgid": PID, "leader_pid": PID, "leader_birth": own}], exclude={os.getpid(), 1},
                           grace_sec=0, rescans=0)
    assert sent == [] and outcome["groups_reused"] == [PID], "a spelling in the server's own TZ is another birth"
    outcome = ps.end_scope(groups=[{"pgid": PID, "leader_pid": PID, "leader_birth": engine}], exclude={os.getpid(), 1},
                           grace_sec=0, rescans=0)
    assert sent == [PID] and outcome["ended"] == [PID] and outcome["groups_reused"] == []


@pytest.mark.parametrize("localtime", [None, PermissionError(13, "denied"), b"not a zone file"],
                         ids=["absent", "denied", "invalid"])
def test_an_unreadable_os_zone_spells_and_parses_no_birth_and_a_later_read_recovers(tmp_path, monkeypatch, localtime):
    kernel = _Kernel(monkeypatch, {PID: _stat(PID)}, localtime=localtime, environ=_env(TZ="UTC"))
    sent = _signals(monkeypatch, kernel)
    epoch = BOOT + TICKS // HZ
    saved = pc.canonical_birth(_spelled(epoch, zoneinfo.ZoneInfo("Pacific/Honolulu")))
    assert pc.lstart_text(epoch) == "" and pc.birth_epoch(saved) == 0.0, "never a guessed UTC, local or zero instant"
    assert pc._proc_identity(PID).state == pc.IDENTITY_UNREADABLE and pc.census() is None
    assert pc._BIRTH_ZONE == [], "a failed read is not cached"
    pending = _pending(tmp_path, [{"pid": PID, "birth": saved}])
    result = ps.complete_pending_panic_sweep(tmp_path)
    assert (result["settled"], result["captured_gone"], sent) == (False, [], [])
    assert result["unverified"] == [{"pid": PID, "birth": saved, "reason": ps.UNVERIFIED_IDENTITY}] and pending.exists()
    kernel.localtime = _zone_file()  # the zone reads again: the saved identity is judged and ended
    result = ps.complete_pending_panic_sweep(tmp_path)
    assert sent == [PID] and result["ended"] == [PID] and result["pending_retired"] is True and not pending.exists()


def test_without_the_zone_the_ps_census_is_unknown_and_never_runs(monkeypatch):
    kernel = _Kernel(monkeypatch, {}, environ=_env(TZ=""))
    kernel.path = SimpleNamespace(isdir=lambda path: False)  # the ps platform
    monkeypatch.setattr(pc, "subprocess", SimpleNamespace(run=lambda *a, **k: pytest.fail("ps ran without a zone")))
    assert pc.census(["OURO_TASK_SCOPE_" + "b" * 32]) is None


def test_a_recorded_floor_that_cannot_be_read_owes_the_group_instead_of_a_weaker_floor(monkeypatch):
    birth = "Mon Oct 12 12:00:07 2026"
    rows = {MEMBER: ps._Row(MEMBER, 1, PID, birth, ps.birth_epoch(birth))}
    monkeypatch.setattr(ps, "read_identities", lambda pids: pytest.fail("a leaderless group asks nobody by name"))
    for floor in ("not a time", "2026-10-12T12:00:00Z"):  # malformed, and an ISO instant no zone has spelled
        view = ps.scope_members(rows, groups=[{"pgid": PID, "leader_pid": PID, "leader_birth": "", "floor": floor}],
                                scope_start_epoch=1.0)
        assert view["members"] == {} and view["groups_gone"] == view["groups_reused"] == []
        assert view["captured_unverified"] == [{"pid": PID, "birth": "", "reason": ps.UNVERIFIED_IDENTITY}]
    view = ps.scope_members(rows, groups=[{"pgid": PID, "leader_pid": PID, "leader_birth": "",
                                           "floor": "Mon Oct 12 12:00:00 2026"}])
    assert set(view["members"]) == {MEMBER}, "a readable floor still judges the group"
    _Kernel(monkeypatch, {}, environ=_env(TZ="UTC"))  # no zone: the engine's ISO opened_at stays recorded, unread
    assert ps._iso_floor("2026-10-12T12:00:00.000Z") == "2026-10-12T12:00:00.000Z"
    assert ps._iso_floor("not a time") == ""


@pytest.mark.skipif(os.path.isdir("/proc"), reason="Linux spells /proc births itself; ps is the macOS/BSD reader")
def test_the_hosts_ps_child_spells_a_birth_like_the_engines_whatever_tz_the_server_carries(monkeypatch):
    engine_env = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}  # the published engine's ps env

    def lstart(env):
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(os.getpid())], capture_output=True, text=True,
                             timeout=5, env=env)
        return pc.canonical_birth(out.stdout)

    reference = lstart(engine_env)
    assert reference
    zones = ("UTC", "Pacific/Honolulu", "Asia/Tokyo")
    for tz in zones:
        monkeypatch.setattr(pc, "os", SimpleNamespace(environ=_env(TZ=tz)))
        assert lstart(pc._ps_env()) == reference, tz
    assert any(lstart({**engine_env, "TZ": tz}) != reference for tz in zones), "a TZ'd ps respells one birth"


# --------------------------------------------------------------------------- 3. recorded group leaders

def _leader_group(birth: str) -> dict:
    return {"pgid": PID, "leader_pid": PID, "leader_birth": birth}


@pytest.mark.parametrize("answer, expected", [
    ("unreadable", {"sent": [], "gone": [], "reused": [], "owed": True}),
    ("alive", {"sent": [PID], "gone": [], "reused": [], "owed": False}),
    ("moved", {"sent": [PID], "gone": [PID], "reused": [], "owed": False}),
])
def test_a_recorded_leader_a_partial_table_omits_is_read_by_name(monkeypatch, answer, expected):
    """The census could not read the leader's stat (the row is missing from the table); the
    by-name answer decides. Alive with its birth it is ended even if it left its group (the
    group it joined, OTHER, is never signalled); a replacement on the pid never is."""
    denied = PermissionError(13, "denied")
    leader = {"unreadable": denied, "alive": [denied, _stat(PID)], "moved": [denied, _stat(PID, pgid=OTHER)],
              "zombie": [denied, _stat(PID, state="Z")], "reborn": [denied, _stat(PID, ticks=TICKS + 9000)]}
    stats = {OTHER: _stat(OTHER, ticks=TICKS + 50), **({PID: leader[answer]} if answer in leader else {})}
    kernel = _Kernel(monkeypatch, stats)
    sent = _signals(monkeypatch, kernel)
    birth = pc.canonical_birth(pc.lstart_text(BOOT + TICKS // HZ))
    outcome = ps.end_scope(groups=[_leader_group(birth)], exclude={os.getpid(), 1}, grace_sec=0, rescans=0)
    assert sent == expected["sent"] and OTHER not in sent
    assert outcome["groups_gone"] == expected["gone"] and outcome["groups_reused"] == expected["reused"]
    assert outcome["ended"] == expected["sent"]
    assert outcome["already_gone"] == ([PID] if answer == "zombie" else [])
    owed = [{"pid": PID, "birth": birth, "reason": ps.UNVERIFIED_IDENTITY}] if expected["owed"] else []
    assert outcome["unverified"] == owed and outcome["settled"] is (not expected["owed"])


@pytest.mark.parametrize("answer", ["unreadable", "absent", "alive", "reborn"])
def test_an_unreadable_recorded_leader_never_authorizes_a_guessed_group(monkeypatch, answer):
    """A member of the recorded group IS in the table, its leader is not. Only absence (the
    ``cmd &`` orphan) or the leader alive with its birth lets the members be ended."""
    denied = PermissionError(13, "denied")
    leader = {"unreadable": denied, "alive": [denied, _stat(PID)], "reborn": [denied, _stat(PID, ticks=TICKS + 9000)]}
    stats = {MEMBER: _stat(MEMBER, pgid=PID, ticks=TICKS + 100), **({PID: leader[answer]} if answer in leader else {})}
    kernel = _Kernel(monkeypatch, stats)
    sent = _signals(monkeypatch, kernel)
    birth = pc.canonical_birth(pc.lstart_text(BOOT + TICKS // HZ))
    outcome = ps.end_scope(groups=[_leader_group(birth)], exclude={os.getpid(), 1}, grace_sec=0, rescans=0)
    assert sorted(sent) == {"unreadable": [], "absent": [MEMBER], "alive": [PID, MEMBER], "reborn": []}[answer]
    assert outcome["groups_reused"] == ([PID] if answer == "reborn" else [])
    assert outcome["settled"] is (answer != "unreadable")


def test_a_recorded_leader_that_left_its_group_is_ended_from_a_full_table_too(monkeypatch):
    kernel = _Kernel(monkeypatch, {PID: _stat(PID, pgid=OTHER), OTHER: _stat(OTHER, ticks=TICKS + 50)})
    sent = _signals(monkeypatch, kernel)
    birth = pc.canonical_birth(pc.lstart_text(BOOT + TICKS // HZ))
    outcome = ps.end_scope(groups=[_leader_group(birth)], exclude={os.getpid(), 1}, grace_sec=0, rescans=0)
    assert sent == [PID] and outcome["ended"] == [PID] and outcome["groups_gone"] == [PID] and outcome["settled"] is True


def test_a_leaderless_groups_omitted_leader_keeps_the_disclosed_residual(monkeypatch):
    """No recorded leader birth: nothing to read by name, and the floor alone judges the members."""
    kernel = _Kernel(monkeypatch, {PID: PermissionError(13, "denied"), MEMBER: _stat(MEMBER, pgid=PID, ticks=TICKS + 100)})
    sent = _signals(monkeypatch, kernel)
    asked = []
    real_read = ps.read_identities
    monkeypatch.setattr(ps, "read_identities", lambda pids: asked.append(sorted(pids)) or real_read(pids))
    outcome = ps.end_scope(groups=[_leader_group("")], scope_start_epoch=float(BOOT), exclude={os.getpid(), 1},
                           grace_sec=0, rescans=0)
    assert sent == [MEMBER] and outcome["ended"] == [MEMBER] and all(PID not in call for call in asked)


def test_a_task_record_keeps_a_recorded_leader_a_partial_table_could_not_read(tmp_path, monkeypatch):
    """Astra P2 through the task cleanup: the record stays with the leader owed by pid and
    birth; once the leader is positively absent, the next sweep settles and retires it."""
    kernel = _Kernel(monkeypatch, {PID: PermissionError(13, "denied"), OTHER: _stat(OTHER)})
    sent = _signals(monkeypatch, kernel)
    birth = pc.canonical_birth(pc.lstart_text(BOOT + TICKS // HZ))
    ps.task_scope_marker(tmp_path, "known-leader")
    path = ps.task_scope_path(tmp_path, "known-leader")
    record = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**record, "groups": [_leader_group(birth)]}), encoding="utf-8")
    monkeypatch.setattr(task_results, "load_task_result", lambda *args, **kwargs: {"status": "completed"})
    published = []
    monkeypatch.setattr(ps, "_record_task_scope_outcome", lambda root, tid, facts: published.append(dict(facts)) or True)
    result = ps.settle_task_scope(tmp_path, "known-leader")
    assert result is not None and result["settled"] is False and sent == []
    assert PID not in result["groups_gone"] and any(row["pid"] == PID for row in result["unverified"])
    assert path.is_file() and published[-1]["settled"] is False
    assert json.loads(path.read_text(encoding="utf-8"))["known_members"] == [{"pid": PID, "birth": birth}]
    del kernel.stats[PID]
    result = ps.settle_task_scope(tmp_path, "known-leader")
    assert result["settled"] is True and sent == [] and not path.exists() and published[-1]["settled"] is True


def test_a_real_leader_the_table_omits_is_read_by_name_and_ended(monkeypatch):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        birth = ps.measure_birth(child.pid)
        assert birth
        real_census = ps.census
        monkeypatch.setattr(ps, "census", lambda markers=(): {
            pid: row for pid, row in (real_census(markers) or {}).items() if pid != child.pid})
        group = {"pgid": os.getpgid(child.pid), "leader_pid": child.pid, "leader_birth": birth}
        outcome = ps.end_scope(groups=[group], exclude={os.getpid(), 1}, grace_sec=2.0)
        assert outcome["ended"] == [child.pid] and outcome["unverified"] == [] and outcome["settled"] is True
        assert child.wait(timeout=5) is not None
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


# --------------------------------------------------------------------------- 4. one birth spelling

def test_one_birth_has_one_spelling():
    assert pc.canonical_birth(HOST_BIRTH) == pc.canonical_birth(ENGINE_BIRTH) == ENGINE_BIRTH
    assert pc.canonical_birth(" Thu  Oct\t1 12:00:00 2026\n") == ENGINE_BIRTH
    assert pc.canonical_birth(OTHER_BIRTH) != ENGINE_BIRTH
    assert pc._Identity(pc.IDENTITY_ALIVE, HOST_BIRTH, PID).birth == pc._Row(PID, 1, PID, HOST_BIRTH, 1.0).birth == ENGINE_BIRTH
    assert pc.birth_epoch(HOST_BIRTH) == pc.birth_epoch(ENGINE_BIRTH) > 0


@pytest.mark.parametrize(("after", "recorded", "live"), [
    pytest.param("ends", ENGINE_BIRTH, HOST_BIRTH, id='ends-engine-record'),
    pytest.param("survives", HOST_BIRTH, ENGINE_BIRTH, id='survives-old-host-record'),
    pytest.param("fails", ENGINE_BIRTH, HOST_BIRTH, id='fails-engine-record'),
])
def test_either_spelling_of_a_live_recorded_birth_is_that_process(monkeypatch, recorded, live, after):
    """The engine-shaped and the old padded record of one live process are that process: it
    is signalled, and a reading in the other spelling is never a false 'gone' or 'ended'; a
    failed signal stays unverified."""
    state = {"alive": True}
    monkeypatch.setattr(ps, "read_identities", lambda pids: {
        int(pid): ps._Identity(ps.IDENTITY_ALIVE, live, int(pid)) if state["alive"] else ps._Identity(ps.IDENTITY_ABSENT)
        for pid in pids})
    sent = []

    def deliver(pid, *, graceful):
        sent.append(pid)
        state["alive"] = after != "ends"
        return _pl.SIGNAL_FAILED if after == "fails" else _pl.SIGNAL_SENT

    monkeypatch.setattr(_pl, "deliver_signal", deliver)
    monkeypatch.setattr(ps, "_KILL_CONFIRM_SEC", 0.0)
    outcome = ps.end_scope(captured=[{"pid": PID, "birth": recorded}], grace_sec=0, term_first=False, rescans=0)
    assert sent == [PID] and outcome["captured_gone"] == [] and outcome["already_gone"] == []
    if after == "ends":
        assert outcome["ended"] == [PID] and outcome["survivors"] == [] and outcome["settled"] is True
        return
    assert outcome["ended"] == [], "no false 'ended' for a padded reading of the same live birth"
    owed = {"pid": PID, "birth": ENGINE_BIRTH}
    if after == "survives":
        assert outcome["survivors"] == [owed] and outcome["unverified"] == []
    else:
        assert outcome["survivors"] == [] and outcome["unverified"] == [{**owed, "reason": ps.UNVERIFIED_FAILED}]
    assert outcome["settled"] is False and ps._known_members(outcome) == [owed]


@pytest.mark.parametrize("recorded", [ENGINE_BIRTH, HOST_BIRTH], ids=["engine-record", "old-host-record"])
def test_a_genuinely_different_birth_in_either_spelling_is_gone_without_a_signal(monkeypatch, recorded):
    monkeypatch.setattr(ps, "read_identities", lambda pids: {
        int(pid): ps._Identity(ps.IDENTITY_ALIVE, OTHER_BIRTH, int(pid)) for pid in pids})
    monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: pytest.fail("a replacement was signalled"))
    outcome = ps.end_scope(captured=[{"pid": PID, "birth": recorded}], grace_sec=0, rescans=0)
    assert outcome["captured_gone"] == [PID] and outcome["unverified"] == [] and outcome["settled"] is True


def test_one_process_recorded_in_both_spellings_is_one_identity(monkeypatch):
    """An old padded known member and the engine's capture of the same process: one unverified
    entry, one saved member, one signal, one survivor."""
    captured = [{"pid": PID, "birth": HOST_BIRTH}, {"pid": PID, "birth": ENGINE_BIRTH}]
    monkeypatch.setattr(ps, "read_identities", lambda pids: {int(pid): ps._Identity(ps.IDENTITY_UNREADABLE) for pid in pids})
    outcome = ps.end_scope(captured=captured, grace_sec=0, rescans=0)
    assert outcome["unverified"] == [{"pid": PID, "birth": ENGINE_BIRTH, "reason": ps.UNVERIFIED_IDENTITY}]
    assert ps._known_members(outcome) == [{"pid": PID, "birth": ENGINE_BIRTH}]
    monkeypatch.setattr(ps, "census", lambda markers=(): None)
    outcome = ps.end_scope(marker=ps.mint_marker(ps.TASK_SCOPE_PREFIX), captured=captured, grace_sec=0)
    assert outcome["unverified"] == [{"pid": PID, "birth": ENGINE_BIRTH, "reason": ps.UNVERIFIED_CENSUS}]
    monkeypatch.setattr(ps, "read_identities", lambda pids: {
        int(pid): ps._Identity(ps.IDENTITY_ALIVE, HOST_BIRTH, int(pid)) for pid in pids})
    sent = []
    monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: sent.append(pid) or _pl.SIGNAL_SENT)
    monkeypatch.setattr(ps, "_KILL_CONFIRM_SEC", 0.0)
    outcome = ps.end_scope(captured=captured, grace_sec=0, term_first=False, rescans=0)
    assert sent == [PID] and outcome["survivors"] == [{"pid": PID, "birth": ENGINE_BIRTH}]


def test_the_signal_pass_and_the_exit_wait_compare_canonically(monkeypatch):
    monkeypatch.setattr(ps, "read_identities", lambda pids: {
        int(pid): ps._Identity(ps.IDENTITY_ALIVE, HOST_BIRTH, int(pid)) for pid in pids})
    for target in (ENGINE_BIRTH, HOST_BIRTH):
        assert ps._await_exit({PID: target}, time.monotonic()) == ({PID: target}, {}), "still alive, never gone"
    sent = []
    monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: sent.append(pid) or _pl.SIGNAL_SENT)
    outcome = {"already_gone": [], "unverified": []}
    assert ps._signal_pass({PID: ENGINE_BIRTH}, outcome, set(), graceful=False, gone_bucket="already_gone") == {PID: ENGINE_BIRTH}
    assert sent == [PID] and outcome == {"already_gone": [], "unverified": []}


@pytest.mark.parametrize("spelling", ["engine", "old-host"])
def test_a_group_leader_recorded_in_either_spelling_vouches_for_its_padded_row(monkeypatch, spelling):
    """Births on day 5 or 6 in every zone: the ``/proc`` spelling pads them, the engine's does not."""
    boot = int(dt.datetime(2026, 10, 5, 12, tzinfo=dt.timezone.utc).timestamp()) - TICKS // HZ
    table = {PID: _stat(PID), MEMBER: _stat(MEMBER, pgid=PID, ticks=TICKS + 100)}
    kernel = _Kernel(monkeypatch, table, btime=b"btime %d\n" % boot)
    sent = _signals(monkeypatch, kernel)
    padded = pc.lstart_text(boot + TICKS // HZ)
    assert "  " in padded, padded
    same = pc.canonical_birth(padded) if spelling == "engine" else padded
    outcome = ps.end_scope(groups=[_leader_group(same)], exclude={os.getpid(), 1}, grace_sec=0, rescans=0)
    assert sorted(sent) == [PID, MEMBER] and outcome["groups_reused"] == []
    kernel.stats.update(table)
    sent.clear()
    another = pc.lstart_text(boot + TICKS // HZ + 86_400)
    another = pc.canonical_birth(another) if spelling == "engine" else another
    outcome = ps.end_scope(groups=[_leader_group(another)], exclude={os.getpid(), 1}, grace_sec=0, rescans=0)
    assert sent == [] and outcome["groups_reused"] == [PID], "another leader birth never vouches, in either spelling"


def test_a_rotation_carries_an_old_spelling_as_stored(tmp_path):
    ps.task_scope_marker(tmp_path, "spelled")
    path = ps.task_scope_path(tmp_path, "spelled")
    record = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**record, "cleanup": {"state": "pending"},
                                "known_members": [{"pid": PID, "birth": HOST_BIRTH}]}), encoding="utf-8")
    ps.task_scope_marker(tmp_path, "spelled")  # a new launch rotates the record
    assert ps.task_scope_record(tmp_path, "spelled")["known_members"] == [{"pid": PID, "birth": HOST_BIRTH}], \
        "no record is rewritten merely to canonicalize"


# --------------------------------------------------------------------------- 5. the cleanup claim

@pytest.mark.parametrize("interleave", ["retired", "rotated", "emptied"])
def test_a_cleanup_claim_never_recreates_a_retired_record_nor_claims_a_successor(tmp_path, monkeypatch, interleave):
    from ouroboros.task_results import load_task_result, write_task_result

    write_task_result(tmp_path, "claim-race", "completed", result="done")  # before the record: no cleanup thread
    marker = ps.task_scope_marker(tmp_path, "claim-race")
    path = ps.task_scope_path(tmp_path, "claim-race")
    written = {"coverage": ps.COVERAGE_POSIX, "ended": [4242], "settled": True, "marker_role": "task"}
    assert ps._record_task_scope_outcome(tmp_path, "claim-race", written)  # the first cleanup's outcome
    successor = {**json.loads(path.read_text(encoding="utf-8")), "marker": ps.mint_marker(ps.TASK_SCOPE_PREFIX),
                 "previous_markers": [marker]}
    real_update = ps.update_json_locked
    entered = []

    def interleaved(target, updater, *args, **kwargs):
        if Path(target) == path and not entered:  # between the sweep's read and its locked claim
            entered.append(interleave)
            if interleave == "retired":
                path.unlink()  # the task-end cleanup settled and retired it
            else:
                path.write_text("{}" if interleave == "emptied" else json.dumps(successor), encoding="utf-8")
        return real_update(target, updater, *args, **kwargs)

    monkeypatch.setattr(ps, "update_json_locked", interleaved)
    monkeypatch.setattr(ps, "census", lambda *a, **k: pytest.fail("a refused claim read the table"))
    monkeypatch.setattr(ps, "read_identities", lambda *a, **k: pytest.fail("a refused claim read an identity"))
    monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: pytest.fail("a refused claim signalled"))
    assert ps.settle_task_scope(tmp_path, "claim-race") is None and entered == [interleave]
    assert load_task_result(tmp_path, "claim-race")["process_scope"] == written, "the written outcome stands"
    if interleave == "retired":
        assert not path.exists(), "a retired record is never recreated"
    elif interleave == "rotated":
        assert json.loads(path.read_text(encoding="utf-8")) == successor, "the successor's record is untouched"
    else:
        assert path.read_text(encoding="utf-8") == "{}"


def test_an_unraced_claim_still_runs_the_cleanup_and_retires_the_record(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result, write_task_result

    write_task_result(tmp_path, "claimed", "completed", result="done")
    ps.task_scope_marker(tmp_path, "claimed")
    monkeypatch.setattr(ps, "census", lambda markers=(): {})
    monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: pytest.fail("nothing to signal"))
    facts = ps.settle_task_scope(tmp_path, "claimed")
    assert facts["settled"] is True and ps.task_scope_record(tmp_path, "claimed") is None
    assert load_task_result(tmp_path, "claimed")["process_scope"]["settled"] is True


# --------------------------------------------------------------------------- 6. whole-table failure

def _record_group_before_failed_panic(tmp_path, monkeypatch, origin, *, birth=HOST_BIRTH):
    """A group-only identity from the real host writer or the engine index's wire shape.
    Both broad attempts fail; no discovery, identity read or real signal can succeed here."""
    ps.installation_marker(tmp_path)
    group = {"pgid": PID, "leader_pid": PID, "leader_birth": birth}
    if origin == "task":
        monkeypatch.setattr(ps, "leader_facts", lambda pid: (pid, birth))
        recorded = ps.record_task_group(tmp_path, "unfinished", PID)
        assert {key: recorded[key] for key in group} == group
        record = ps.task_scope_record(tmp_path, "unfinished")
        assert record["groups"] == [recorded] and not record.get("known_members")
    else:
        path = ps.engine_run_scopes_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"schema_version": 1, "entries": [{
            "run_id": "recorded-run", "marker": ps.mint_marker(ps.RUN_SCOPE_PREFIX),
            "state": "open", "scope_start": None, "root_groups": [group], "captured": [],
        }]}), encoding="utf-8")
        assert ps.engine_scope_index(tmp_path)[1]["status"] == "read"
    monkeypatch.setattr(ps, "census", lambda markers=(): None)
    monkeypatch.setattr(ps, "read_identities", lambda pids: pytest.fail("failed table started another read"))
    monkeypatch.setattr(ps, "_RESCAN_SETTLE_SEC", 0)
    monkeypatch.setattr(ps, "_await_exit", lambda targets, deadline: ({}, {}))
    sent = []
    monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: sent.append(pid) or _pl.SIGNAL_SENT)
    return sent


@pytest.mark.parametrize(("later", "origin"), [
    ("unreadable", "task"),
    ("same", "engine"),
    ("reused", "task"),
    ("absent", "engine"),
])
def test_failed_broad_table_preserves_recorded_leader_for_known_only_recovery(tmp_path, monkeypatch, origin, later):
    sent = _record_group_before_failed_panic(tmp_path, monkeypatch, origin)
    first = ps.panic_scope_census(tmp_path, exclude={os.getpid(), 1})
    launcher = ps.complete_pending_panic_sweep(tmp_path, exclude={os.getpid(), 1}, installation=True)
    for outcome in (first, launcher):
        assert outcome["census_failed"] and not outcome["settled"] and not outcome["pending_retired"]
        assert outcome["unverified"] == [{"pid": PID, "birth": ENGINE_BIRTH, "reason": ps.UNVERIFIED_CENSUS}]
    pending = ps.panic_sweep_pending_path(tmp_path)
    expected = [{"pid": PID, "birth": ENGINE_BIRTH}]
    assert json.loads(pending.read_text(encoding="utf-8"))["known_members"] == expected
    monkeypatch.setattr(ps, "census", lambda markers=(): pytest.fail("ordinary startup discovered new work"))
    asked = []

    def identities(pids):
        pids = list(pids)
        asked.append(pids)
        assert pids == [PID]
        if later == "unreadable":
            return {PID: ps._Identity(ps.IDENTITY_UNREADABLE)}
        if later == "absent":
            return {PID: ps._Identity(ps.IDENTITY_ABSENT)}
        return {PID: ps._Identity(ps.IDENTITY_ALIVE, HOST_BIRTH if later == "same" else OTHER_BIRTH, PID)}

    monkeypatch.setattr(ps, "read_identities", identities)
    recovered = ps.complete_pending_panic_sweep(tmp_path, exclude={os.getpid(), 1})
    assert asked and recovered["scope"] == "known_members"
    assert sent == ([PID] if later == "same" else [])
    if later == "unreadable":
        assert not recovered["settled"] and not recovered["pending_retired"]
        assert json.loads(pending.read_text(encoding="utf-8"))["known_members"] == expected
    else:
        assert recovered["settled"] and recovered["pending_retired"] and not pending.exists()
        assert recovered["ended"] == ([PID] if later == "same" else [])
        assert recovered["captured_gone"] == ([PID] if later in ("reused", "absent") else [])


@pytest.mark.parametrize(("case", "origin"), [("excluded", "task"), ("leaderless", "engine")])
def test_failed_broad_table_never_invents_an_excluded_or_unidentified_leader(tmp_path, monkeypatch, origin, case):
    sent = _record_group_before_failed_panic(tmp_path, monkeypatch, origin,
                                           birth="" if case == "leaderless" else HOST_BIRTH)
    excluded = {os.getpid(), 1, *([PID] if case == "excluded" else [])}
    first = ps.panic_scope_census(tmp_path, exclude=excluded)
    assert first["census_failed"] and not first["settled"]
    pending = ps.panic_sweep_pending_path(tmp_path)
    assert json.loads(pending.read_text(encoding="utf-8"))["known_members"] == []
    recovered = ps.complete_pending_panic_sweep(tmp_path, exclude=excluded)
    assert recovered["settled"] and recovered["pending_retired"] and sent == []


def test_failed_rescan_does_not_reowe_a_group_leader_already_ended(monkeypatch):
    rows = [{PID: pc._Row(PID, 1, PID, HOST_BIRTH, pc.birth_epoch(HOST_BIRTH), pc.IDENTITY_ALIVE, set())}, None]
    monkeypatch.setattr(ps, "census", lambda markers=(): rows.pop(0))
    monkeypatch.setattr(ps, "read_identities", lambda pids: {
        pid: pc._Identity(pc.IDENTITY_ALIVE, HOST_BIRTH, PID) for pid in pids})
    sent = []
    monkeypatch.setattr(_pl, "deliver_signal", lambda pid, *, graceful: sent.append(pid) or _pl.SIGNAL_SENT)
    monkeypatch.setattr(ps, "_await_exit", lambda targets, deadline: ({}, {}))
    monkeypatch.setattr(ps, "_RESCAN_SETTLE_SEC", 0)
    outcome = ps.end_scope(groups=[{"pgid": PID, "leader_pid": PID, "leader_birth": HOST_BIRTH}],
                           captured=[{"pid": PID, "birth": ENGINE_BIRTH}], term_first=False, rescans=1)
    assert outcome["census_failed"] and not outcome["settled"]
    assert outcome["ended"] == [PID] and sent == [PID]
    assert outcome["unverified"] == [] and ps._known_members(outcome) == []
