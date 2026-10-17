"""Feasibility probe for zoom_meeting, run by a person on the target server.

It answers one question before anyone asks the install to join a meeting: can this
host run the Zoom Meeting SDK with these app credentials? It checks the platform,
imports the binding, finds GLib, looks for the audio prerequisites Zoom documents for
raw audio on Linux (``raw_audio``: a PulseAudio server and ``~/.config/zoomus.conf``
with ``system.audio.type=default``), starts the SDK and authenticates the app. It never
joins a meeting and never prints a secret; the credentials appear only as the Presence
account digest a binding names.

The audio step reads this shell's environment by the companion's rule, naming each
missing piece. The companion runs with the environment Ouroboros starts it with, which
can differ; ``zoom_status`` shows the companion's own result of the same check.

    <skill>/.ouroboros_env/python/bin/python <skill>/scripts/probe.py [--settings PATH] [--json]

Credentials come from ZOOM_CLIENT_ID and ZOOM_CLIENT_SECRET in the environment, or from
the same keys in an Ouroboros settings.json given with --settings. Each step prints
PASS, FAIL or NOT_RUN (an earlier step it depends on did not pass); the exit status is
0 only when every step passed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import glib_loop
import raw_audio
from companion_link import app_account_id
from zoom_session import WEB_DOMAIN, build_sdk_jwt, sdk_name

PASS, FAIL, NOT_RUN = "PASS", "FAIL", "NOT_RUN"
STEPS = ("platform", "import_sdk", "glib", "audio_raw", "credentials", "init_sdk", "authenticate", "cleanup")
SUPPORTED_PLATFORM = "linux-x86_64"
AUTH_TIMEOUT_MS = 30000  # how long Zoom's authentication answer may take before the step fails


def _platform_name() -> str:
    return f"{sys.platform}-{platform.machine().lower()}"


def _import_sdk() -> Tuple[Any, str]:
    try:
        module = importlib.import_module("zoom_meeting_sdk")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    try:
        from importlib.metadata import version

        return module, f"zoom-meeting-sdk {version('zoom-meeting-sdk')}"
    except Exception:
        return module, "zoom_meeting_sdk imported"


def load_credentials(env: Mapping[str, str], settings_path: Optional[str]) -> Tuple[str, str, str]:
    """``(client_id, client_secret, where)``; values never leave this process."""
    if settings_path:
        try:
            data = json.loads(Path(settings_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return "", "", f"settings file unreadable ({type(exc).__name__})"
        data = data if isinstance(data, dict) else {}
        return (str(data.get("ZOOM_CLIENT_ID") or "").strip(), str(data.get("ZOOM_CLIENT_SECRET") or "").strip(),
                "settings file")
    return env.get("ZOOM_CLIENT_ID", "").strip(), env.get("ZOOM_CLIENT_SECRET", "").strip(), "environment"


def run_probe(*, env: Mapping[str, str], settings_path: Optional[str] = None,
              import_sdk: Callable[[], Tuple[Any, str]] = _import_sdk,
              make_loop: Callable[[], Any] = glib_loop.make_loop,
              platform_name: Callable[[], str] = _platform_name,
              clock: Callable[[], float] = time.time,
              auth_timeout_ms: int = AUTH_TIMEOUT_MS, uid: Optional[int] = raw_audio.current_uid(),
              user_runtime_root: Path = raw_audio.USER_RUNTIME_ROOT) -> List[Dict[str, str]]:
    results: Dict[str, Tuple[str, str]] = {}
    plat = platform_name()
    results["platform"] = (PASS if plat == SUPPORTED_PLATFORM else FAIL,
                           plat if plat == SUPPORTED_PLATFORM else f"{plat}; the binding ships {SUPPORTED_PLATFORM} only")
    sdk, detail = import_sdk()
    results["import_sdk"] = (PASS if sdk is not None else FAIL, detail)
    loop = None
    if sdk is not None:
        try:
            loop = make_loop()
            results["glib"] = (PASS, f"{loop.backend} ({loop.library})")
        except glib_loop.GLibUnavailable as exc:
            results["glib"] = (FAIL, str(exc))
    results["audio_raw"] = raw_audio.check(dict(env), home=env.get("HOME", ""), uid=uid,
                                           user_runtime_root=user_runtime_root)
    client_id, client_secret, where = load_credentials(env, settings_path)
    if client_id and client_secret:
        results["credentials"] = (PASS, f"from the {where}; Presence account {app_account_id(client_id)}")
    else:
        results["credentials"] = (FAIL, f"ZOOM_CLIENT_ID and ZOOM_CLIENT_SECRET are not both set ({where})")
    if loop is not None:
        results.update(_sdk_steps(sdk, loop, client_id if results["credentials"][0] == PASS else "",
                                  client_secret, clock, auth_timeout_ms))
    return [{"step": step, "status": results.get(step, (NOT_RUN, "an earlier step did not pass"))[0],
             "detail": results.get(step, (NOT_RUN, "an earlier step did not pass"))[1]} for step in STEPS]


def _sdk_steps(sdk: Any, loop: Any, client_id: str, client_secret: str, clock: Callable[[], float],
               auth_timeout_ms: int) -> Dict[str, Tuple[str, str]]:
    results: Dict[str, Tuple[str, str]] = {}
    anchors: Dict[str, Any] = {}

    def on_auth(result: Any) -> None:
        results.setdefault("authenticate", (PASS if result == sdk.AUTHRET_SUCCESS else FAIL, sdk_name(result)))
        loop.quit()

    def start() -> bool:
        try:
            param = sdk.InitParam()
            param.strWebDomain = WEB_DOMAIN
            param.enableGenerateDump = False
            err = sdk.InitSDK(param)
        except Exception as exc:
            results["init_sdk"] = (FAIL, type(exc).__name__)
            loop.quit()
            return False
        if err != sdk.SDKERR_SUCCESS:
            results["init_sdk"] = (FAIL, sdk_name(err))
            loop.quit()
            return False
        results["init_sdk"] = (PASS, f"{sdk_name(err)} ({WEB_DOMAIN})")
        if not client_id:
            loop.quit()
            return False
        try:
            token = build_sdk_jwt(client_id, client_secret, now=int(clock()))
            auth = sdk.CreateAuthService()
            callbacks = sdk.AuthServiceEventCallbacks(onAuthenticationReturnCallback=on_auth)
            context = sdk.AuthContext()
            context.jwt_token = token
            anchors.update(auth=auth, callbacks=callbacks, context=context, token=token)
            auth.SetEvent(callbacks)
            err = auth.SDKAuth(context)
        except Exception as exc:
            results["authenticate"] = (FAIL, type(exc).__name__)
            loop.quit()
            return False
        if err != sdk.SDKERR_SUCCESS:
            results["authenticate"] = (FAIL, f"SDKAuth {sdk_name(err)}")
            loop.quit()
        return False

    def timeout() -> bool:
        results.setdefault("authenticate", (FAIL, f"no answer from Zoom within {auth_timeout_ms // 1000} s"))
        loop.quit()
        return False

    loop.idle_add(start)
    loop.timeout_add(auth_timeout_ms, timeout)
    loop.run()
    if results.get("init_sdk", (FAIL,))[0] == PASS:
        try:
            sdk.CleanUPSDK()
            results["cleanup"] = (PASS, "CleanUPSDK")
        except Exception as exc:
            results["cleanup"] = (FAIL, type(exc).__name__)
    anchors.clear()
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Can this host run the Zoom Meeting SDK with these credentials?")
    parser.add_argument("--settings", help="an Ouroboros settings.json holding ZOOM_CLIENT_ID/ZOOM_CLIENT_SECRET")
    parser.add_argument("--json", action="store_true", help="print one JSON object instead of lines")
    args = parser.parse_args(argv)
    steps = run_probe(env=os.environ, settings_path=args.settings)
    overall = PASS if all(step["status"] == PASS for step in steps) else FAIL
    if args.json:
        print(json.dumps({"result": overall, "steps": steps}, ensure_ascii=False))
    else:
        for step in steps:
            print(f"{step['status']:<8} {step['step']:<13} {step['detail']}")
        print(f"RESULT   {overall}")
    return 0 if overall == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
