"""Small in-container process observer used by the Docker executor.

The host cannot inspect a container PID namespace.  This module only builds the
POSIX-sh fragment executed by ``docker exec``; the executor owns the transport
and interprets its three-valued ``running/exited/unknown`` result.
"""

import shlex


def proc_observer_function(*, proc_root: str = "/proc") -> str:
    """Build the leader/group observer; ``proc_root`` is a fixture seam only.

    Runtime callers always inspect the container's /proc. A missing process is
    exited; present but unreadable or malformed stat data is unknown. Group
    scope also includes the leader for older records; new launches require setsid.
    """
    return r'''
_ouroboros_proc_observe() {
  _opid="$1"; _oscope="$2"; _oroot=__PROC_ROOT__
  case "$_opid" in ''|0|*[!0-9]*) printf unknown; return 2;; esac
  case "$_oscope" in leader|group) ;; *) printf unknown; return 2;; esac
  if [ ! -d "$_oroot" ] || [ ! -r "$_oroot" ] || [ ! -x "$_oroot" ]; then printf unknown; return 2; fi
  _ostat() {
    _op="$1"; _of="$_oroot/$_op/stat"
    if [ ! -d "$_oroot/$_op" ]; then return 2; fi
    _ol=$(cat "$_of" 2>/dev/null) || { [ ! -d "$_oroot/$_op" ] && return 2; return 3; }
    case "$_ol" in "$_op ("*) ;; *) return 3;; esac
    # comm can contain spaces and ')': the last closing delimiter owns state.
    _or=${_ol##*) }
    [ "$_or" != "$_ol" ] || return 3
    set -f
    set -- $_or
    [ "$#" -ge 50 ] || return 3
    case "$1" in R|S|D|T|t|Z|X|x|K|W|P|I) ;; *) return 3;; esac
    case "$2" in ''|*[!0-9]*) return 3;; esac
    case "$3" in ''|*[!0-9]*) return 3;; esac
    printf '%s %s\n' "$1" "$3"
  }
  _leader=$(_ostat "$_opid"); _lrc=$?
  if [ "$_oscope" = leader ]; then
    if [ "$_lrc" -eq 2 ]; then printf exited; return 0; fi
    if [ "$_lrc" -ne 0 ]; then printf unknown; return 2; fi
    _state=${_leader%% *}
    case "$_state" in Z|X|x) printf exited;; *) printf running;; esac
    return 0
  fi
  # Cleanup signals the process group whose id is the recorded leader PID.
  _target="$_opid"
  if [ "$_lrc" -eq 3 ]; then printf unknown; return 2; fi
  _survivors=""
  for _entry in "$_oroot"/[0-9]*; do
    [ -d "$_entry" ] || continue
    _candidate=${_entry##*/}
    _member=$(_ostat "$_candidate"); _mrc=$?
    if [ "$_mrc" -eq 2 ]; then continue; fi
    if [ "$_mrc" -ne 0 ]; then printf unknown; return 2; fi
    _mstate=${_member%% *}; _mpgrp=${_member#* }
    [ "$_candidate" = "$_opid" ] || [ "$_mpgrp" = "$_target" ] || continue
    case "$_mstate" in
      Z|X|x) ;;
      *)
        if [ -n "$_survivors" ]; then _survivors="$_survivors,$_candidate"; else _survivors="$_candidate"; fi
        ;;
    esac
  done
  if [ -n "$_survivors" ]; then
    printf 'surviving process group members: %s\n' "$_survivors" >&2
    printf running
  else
    printf exited
  fi
}
'''.strip().replace("__PROC_ROOT__", shlex.quote(str(proc_root)))
