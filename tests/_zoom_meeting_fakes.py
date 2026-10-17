"""Fakes for tests/test_skill_zoom_meeting.py: the Zoom binding, the loop, the Host door.

``FakeZoom`` mirrors the zoom-meeting-sdk 0.0.27 surface Hope's code uses: callback
holders accept only the binding's keyword names and the join parameters only its
fields, so a misspelt SDK call fails in a test instead of on a server. ``ManualLoop``
is the GLib loop's interface on virtual time. ``start_fake_door`` serves the Host
Service door (``/identity``, ``POST /call/open``, ``WS /call/{id}/peer``) as a real
Starlette app on a loopback port; ``serve_app`` serves any app that way, the real Host
Service included. ``join_args``, ``drive_to_audio`` and ``open_call_via_audio`` are the
steps a test takes to reach a joined meeting whose audio opened a call.
"""

from __future__ import annotations

import asyncio
import collections
import ctypes
import enum
import heapq
import itertools
import json
import socket
import threading
import time
import types

CLIENT_ID = "fakeClientIdAbc"
CLIENT_SECRET = "fake-client-secret-0123456789abcdef"
PASSCODE = "pass-word-271828"
OBF = "obf-token-3141592653"
HOST_TOKEN = "host-token-5772156649"
MEETING = "85012345678"
BINDING = "b" * 32
MY_ID = 16778240


# ---- the fake binding ---------------------------------------------------------------

_CALLBACK_KEYWORDS = {
    "AuthServiceEventCallbacks": ("onAuthenticationReturnCallback", "onLoginReturnWithReasonCallback",
                                  "onLogoutCallback", "onZoomIdentityExpiredCallback",
                                  "onZoomAuthIdentityExpiredCallback"),
    "MeetingServiceEventCallbacks": ("onMeetingStatusChangedCallback", "onMeetingStatisticsWarningNotificationCallback",
                                     "onMeetingParameterNotificationCallback", "onSuspendParticipantsActivitiesCallback",
                                     "onAICompanionActiveChangeNoticeCallback", "onMeetingTopicChangedCallback",
                                     "onMeetingFullToWatchLiveStreamCallback", "onUserNetworkStatusChangedCallback"),
    "ZoomSDKVirtualAudioMicEventCallbacks": ("onMicInitializeCallback", "onMicStartSendCallback",
                                             "onMicStopSendCallback", "onMicUninitializedCallback"),
    "MeetingRecordingCtrlEventCallbacks": ("onRecordingStatusCallback", "onCloudRecordingStatusCallback",
                                           "onRecordPrivilegeChangedCallback",
                                           "onLocalRecordingPrivilegeRequestStatusCallback",
                                           "onRequestCloudRecordingResponseCallback",
                                           "onLocalRecordingPrivilegeRequestedCallback",
                                           "onStartCloudRecordingRequestedCallback",
                                           "onCloudRecordingStorageFullCallback",
                                           "onEnableAndStartSmartRecordingRequestedCallback",
                                           "onSmartRecordingEnableActionCallbackFunc",
                                           "onTranscodingStatusChangedCallback"),
    "ZoomSDKAudioRawDataDelegateCallbacks": ("onMixedAudioRawDataReceivedCallback",
                                             "onOneWayAudioRawDataReceivedCallback",
                                             "onShareAudioRawDataReceivedCallback",
                                             "onOneWayInterpreterAudioRawDataReceivedCallback",
                                             "collectPerformanceData"),
    "MeetingChatEventCallbacks": ("onChatMsgNotificationCallback", "onChatStatusChangedNotificationCallback",
                                  "onChatMsgDeleteNotificationCallback", "onChatMessageEditNotificationCallback",
                                  "onShareMeetingChatStatusChangedCallback", "onFileSendStartCallback",
                                  "onFileReceivedCallback", "onFileTransferProgressCallback"),
    "MeetingAudioCtrlEventCallbacks": ("onUserAudioStatusChangeCallback", "onUserActiveAudioChangeCallback",
                                       "onHostRequestStartAudioCallback", "onJoin3rdPartyTelephonyAudioCallback",
                                       "onMuteOnEntryStatusChangeCallback"),
}
_JOIN_FIELDS = ("meetingNumber", "userName", "psw", "vanityID", "customer_key", "join_token", "webinarToken",
                "isVideoOff", "isAudioOff", "userZAK", "isMyVoiceInMix", "app_privilege_token", "onBehalfToken",
                "isAudioRawDataStereo", "eAudioRawdataSamplingRate")


def _libc():
    """C stdio without varargs (ctypes variadic calls are unsafe on some ABIs)."""
    libc = ctypes.CDLL(None)
    libc.puts.argtypes, libc.fflush.argtypes = [ctypes.c_char_p], [ctypes.c_void_p]
    return libc


class FakeZoom:
    """A ``zoom_meeting_sdk`` stand-in plus the levers a test pulls to play Zoom's part."""

    def __init__(self) -> None:
        m = self.m = types.ModuleType("zoom_meeting_sdk")
        fake = self
        self.calls: list = []
        self.codes = collections.defaultdict(lambda: "SDKERR_SUCCESS")
        self.topic, self.muted, self.print_on_join = "Weekly sync", True, False
        self.users = {MY_ID: "Aika", 42: "Anna", 7: "Boris"}
        self.sent: list = []
        self.cb: dict = {}
        self.on_sdkauth = None
        self.join_param: dict = {}
        self.init_param: dict = {}
        self.jwt = ""

        def _enum(name, members, export=True):
            cls = enum.IntEnum(name, [(member, index) for index, member in enumerate(members)])
            setattr(m, name, cls)
            if export:
                for member in cls:
                    setattr(m, member.name, member)

        _enum("SDKError", ("SDKERR_SUCCESS", "SDKERR_NO_IMPL", "SDKERR_WRONG_USAGE", "SDKERR_INVALID_PARAMETER",
                           "SDKERR_UNINITIALIZE", "SDKERR_UNAUTHENTICATION", "SDKERR_NO_PERMISSION"))
        _enum("AuthResult", ("AUTHRET_SUCCESS", "AUTHRET_KEYORSECRETWRONG", "AUTHRET_JWTTOKENWRONG"))
        _enum("MeetingStatus", ("MEETING_STATUS_IDLE", "MEETING_STATUS_CONNECTING", "MEETING_STATUS_WAITINGFORHOST",
                                "MEETING_STATUS_INMEETING", "MEETING_STATUS_DISCONNECTING",
                                "MEETING_STATUS_RECONNECTING", "MEETING_STATUS_FAILED", "MEETING_STATUS_ENDED",
                                "MEETING_STATUS_IN_WAITING_ROOM"))
        _enum("MeetingFailCode", ("MEETING_SUCCESS", "MEETING_FAIL_PASSWORD_ERR",
                                  "MEETING_FAIL_AUTHORIZED_USER_NOT_INMEETING"))
        _enum("MeetingEndReason", ("EndMeetingReason_None", "EndMeetingReason_KickByHost",
                                   "EndMeetingReason_EndByHost"), export=False)
        _enum("LeaveMeetingCmd", ("LEAVE_MEETING", "END_MEETING"))
        _enum("SDKUserType", ("SDK_UT_NORMALUSER", "SDK_UT_WITHOUT_LOGIN"))
        _enum("AudioRawdataSamplingRate", ("AudioRawdataSamplingRate_32K", "AudioRawdataSamplingRate_48K"))
        _enum("ZoomSDKAudioChannel", ("ZoomSDKAudioChannel_Mono", "ZoomSDKAudioChannel_Stereo"))
        _enum("RequestLocalRecordingStatus", ("RequestLocalRecording_Granted", "RequestLocalRecording_Denied",
                                              "RequestLocalRecording_Timeout"), export=False)
        _enum("AudioStatus", ("Audio_None", "Audio_Muted", "Audio_UnMuted", "Audio_Muted_ByHost",
                              "Audio_UnMuted_ByHost"), export=False)

        def code(verb):
            return m.SDKError[fake.codes[verb]]

        for cls_name, keywords in _CALLBACK_KEYWORDS.items():
            def __init__(self, _keywords=keywords, **kwargs):
                unknown = set(kwargs) - set(_keywords)
                if unknown:
                    raise TypeError(f"unexpected keyword(s) {sorted(unknown)}")
                self.kw = kwargs
            setattr(m, cls_name, type(cls_name, (), {"__init__": __init__}))

        def slotted(name, fields):
            def __init__(self):
                for field in fields:
                    setattr(self, field, None)
            return type(name, (), {"__slots__": fields, "__init__": __init__})

        m.InitParam = slotted("InitParam", ("strWebDomain", "strBrandingName", "strSupportUrl", "emLanguageID",
                                            "enableGenerateDump", "enableLogByDefault", "uiLogFileSize"))
        m.AuthContext = slotted("AuthContext", ("jwt_token",))
        params = slotted("JoinParam4WithoutLogin", _JOIN_FIELDS)

        class JoinParam:
            __slots__ = ("userType", "_param")

            def __init__(self):
                self.userType, self._param = None, params()

            @property
            def param(self):
                return self._param

        m.JoinParam = JoinParam

        def record(verb, result=None):
            fake.calls.append(verb)
            return result

        def InitSDK(param):
            fake.init_param = {f: getattr(param, f) for f in ("strWebDomain", "enableGenerateDump")}
            return record("InitSDK", code("InitSDK"))

        def CleanUPSDK():
            record("CleanUPSDK")

        class AuthService:
            def SetEvent(self, cb):
                fake.cb["auth"] = cb
                return record("AuthService.SetEvent", code("SetEvent"))

            def SDKAuth(self, context):
                fake.jwt = context.jwt_token
                result = record("SDKAuth", code("SDKAuth"))
                if fake.on_sdkauth:
                    fake.on_sdkauth()
                return result

        class User:
            def __init__(self, uid):
                self.uid = uid

            def GetUserID(self):
                return self.uid

            def GetUserName(self):
                return fake.users[self.uid]

            def IsAudioMuted(self):
                return fake.muted

        class Controllers:
            def JoinVoip(self):
                return record("JoinVoip", code("JoinVoip"))

            def UnMuteAudio(self, uid):
                return record("UnMuteAudio", code("UnMuteAudio"))

            def GetMySelfUser(self):
                return User(MY_ID)

            def GetUserByUserID(self, uid):
                return User(uid) if uid in fake.users else None

            def CanStartRawRecording(self):
                return record("CanStartRawRecording", code("CanStartRawRecording"))

            def RequestLocalRecordingPrivilege(self):
                return record("RequestLocalRecordingPrivilege", code("RequestLocalRecordingPrivilege"))

            def StartRawRecording(self):
                return record("StartRawRecording", code("StartRawRecording"))

        def controller(kind):
            ctrl = Controllers()

            def SetEvent(cb):
                fake.cb[kind] = cb
                return record(f"{kind}.SetEvent", code("SetEvent"))
            ctrl.SetEvent = SetEvent
            return ctrl

        audio_ctrl, recording_ctrl, chat_ctrl = controller("audio"), controller("recording"), controller("chat")

        class MeetingService:
            def SetEvent(self, cb):
                fake.cb["meeting"] = cb
                return record("MeetingService.SetEvent", code("SetEvent"))

            def Join(self, join):
                fake.join_param = {"userType": join.userType, **{f: getattr(join.param, f) for f in _JOIN_FIELDS}}
                if fake.print_on_join:  # like printJoinParam: C stdio, buffered when stdout is not a tty
                    _libc().puts(b"Password: " + join.param.psw.encode())
                return record("Join", code("Join"))

            def Leave(self, cmd):
                fake.leave_cmd = cmd
                return record("Leave", code("Leave"))

            def GetMeetingInfo(self):
                record("GetMeetingInfo")
                return types.SimpleNamespace(GetMeetingTopic=lambda: fake.topic)

            def GetMeetingAudioController(self):
                return audio_ctrl

            def GetMeetingParticipantsController(self):
                return audio_ctrl

            def GetMeetingRecordingController(self):
                return recording_ctrl

            def GetMeetingChatController(self):
                return chat_ctrl

        class Helper:
            def setExternalAudioSource(self, cb):
                fake.cb["mic"] = cb
                return record("setExternalAudioSource", code("setExternalAudioSource"))

            def subscribe(self, delegate, with_interpreters):
                fake.cb["delegate"] = delegate
                return record("subscribe", code("subscribe"))

            def unSubscribe(self):
                return record("unSubscribe", code("unSubscribe"))

        helper = Helper()
        settings = types.SimpleNamespace(GetAudioSettings=lambda: types.SimpleNamespace(
            EnableAutoJoinAudio=lambda flag: record("EnableAutoJoinAudio", flag)))
        m.InitSDK, m.CleanUPSDK = InitSDK, CleanUPSDK
        m.CreateAuthService = lambda: record("CreateAuthService", AuthService())
        m.CreateMeetingService = lambda: record("CreateMeetingService", MeetingService())
        m.CreateSettingService = lambda: record("CreateSettingService", settings)
        m.GetAudioRawdataHelper = lambda: record("GetAudioRawdataHelper", helper)

        class Sender:
            def send(self, data, rate, channel):
                fake.sent.append((bytes(data), rate, channel))
                return m.SDKERR_SUCCESS

        self.sender = Sender()

    # ---- Zoom's side of the conversation --------------------------------------------

    def fire(self, kind, keyword, *args):
        self.cb[kind].kw[keyword](*args)

    def auth(self, result="AUTHRET_SUCCESS"):
        self.fire("auth", "onAuthenticationReturnCallback", self.m.AuthResult[result])

    def status(self, name, result=0):
        self.fire("meeting", "onMeetingStatusChangedCallback", getattr(self.m, name), result)

    def end(self, reason):
        self.status("MEETING_STATUS_ENDED", self.m.MeetingEndReason[reason])

    def mic(self):
        self.fire("mic", "onMicInitializeCallback", self.sender)
        self.fire("mic", "onMicStartSendCallback")

    def answer(self, name):
        self.fire("recording", "onLocalRecordingPrivilegeRequestStatusCallback",
                  self.m.RequestLocalRecordingStatus[f"RequestLocalRecording_{name}"])

    def mixed(self, pcm, rate=32000):
        data = types.SimpleNamespace(GetBuffer=lambda: pcm, GetSampleRate=lambda: rate, GetChannelNum=lambda: 1)
        self.fire("delegate", "onMixedAudioRawDataReceivedCallback", data)

    def one_way(self, node_id):
        self.fire("delegate", "onOneWayAudioRawDataReceivedCallback", None, node_id)

    def chat(self, sender_id, text, to_all=True):
        info = types.SimpleNamespace(GetSenderUserId=lambda: sender_id, GetSenderDisplayName=lambda: self.users[sender_id],
                                     GetContent=lambda: text, IsChatToAll=lambda: to_all)
        self.fire("chat", "onChatMsgNotificationCallback", info, text)

    def host_mute(self):
        status = types.SimpleNamespace(GetUserId=lambda: MY_ID, GetStatus=lambda: self.m.AudioStatus.Audio_Muted_ByHost)
        self.fire("audio", "onUserAudioStatusChangeCallback", [status], "")


class ManualLoop:
    """The GLib loop's interface on virtual time: callbacks run only when pumped."""

    backend, library = "manual", "test"

    def __init__(self) -> None:
        self.now = 0.0
        self.quit_called = False
        self._lock = threading.Lock()
        self._ready: collections.deque = collections.deque()
        self._timers: list = []
        self._seq = itertools.count()

    def idle_add(self, fn):
        with self._lock:
            self._ready.append(fn)
        return 1

    def timeout_add(self, ms, fn):
        with self._lock:
            heapq.heappush(self._timers, (self.now + ms / 1000.0, next(self._seq), ms, fn))
        return 1

    def quit(self):
        self.quit_called = True

    def run_ready(self):
        ran = 0
        while True:
            with self._lock:
                if not self._ready:
                    return ran
                fn = self._ready.popleft()
            fn()
            ran += 1

    def advance(self, seconds):
        target = self.now + seconds
        while True:
            self.run_ready()
            with self._lock:
                if not self._timers or self._timers[0][0] > target:
                    break
                due, _seq, ms, fn = heapq.heappop(self._timers)
                self.now = max(self.now, due)
            if fn():
                self.timeout_add(ms, fn)
        self.now = target
        self.run_ready()

    def pump(self, until, timeout=10.0):
        deadline = time.monotonic() + timeout
        while not until():
            assert time.monotonic() < deadline, "condition not reached"
            self.advance(0.05)
            time.sleep(0.005)

    def run(self):
        while not self.quit_called:
            if self.run_ready():
                continue
            assert self._timers, "loop idle with nothing scheduled"
            self.advance(max(0.0, self._timers[0][0] - self.now))


class FakePeer:
    sent_frames = received_frames = dropped_uplink = 0

    def __init__(self, hello, callbacks):
        self.hello, self.callbacks = hello, callbacks
        self.audio, self.controls, self.closed = [], [], None

    def send_audio(self, pcm, rate):
        self.audio.append((pcm, rate))

    def send_control(self, message):
        self.controls.append(message)

    def close(self, reason):
        self.closed = reason


class FakePacer:
    def __init__(self, send, rate):
        self.rate, self.fed, self.flushes, self.sent_frames, self.flushed_bytes = rate, [], [], 0, 0

    def feed(self, pcm, epoch):
        self.fed.append((pcm, epoch))

    def flush(self, epoch):
        self.flushes.append(epoch)

    def run(self, stop):
        stop.wait(5)

    def close(self):
        pass


class FakeApi:
    def __init__(self, state_dir, data_dir, settings):
        self.state_dir, self.data_dir, self.settings = state_dir, data_dir, settings
        self.tools, self.companions = {}, []

    def register_companion_process(self, name):
        self.companions.append(name)

    def register_tool(self, name, handler, *, description, schema, timeout_sec=60):
        self.tools[name] = types.SimpleNamespace(handler=handler, description=description, schema=schema)

    def get_state_dir(self):
        return str(self.state_dir)

    def get_settings(self, keys):
        return {key: value for key, value in self.settings.items() if key in keys}

    def get_runtime_info(self):
        return {"data_dir": str(self.data_dir)}


class FakeCompanion:
    def __init__(self, account):
        self.account, self.requests = account, []
        self.state, self.sdk, self.after_join, self.join_outcome = "idle", True, "waiting_for_audio", None

    def __call__(self, op, args):
        self.requests.append((op, args))
        if op == "join":
            self.state = self.after_join
            return {"ok": True, "join_id": "j-9", **self.status()}
        if op == "leave":
            left, self.state = self.state != "idle", "idle"
            return {"ok": True, "left": left, **self.status()}
        return {"ok": True, **self.status()}

    def status(self):
        messages = {"idle": "not in a Zoom meeting", "joining": "joining",
                    "waiting_for_audio": "in the meeting, waiting for the host to allow Aika's audio"}
        return {"state": self.state, "message": messages[self.state], "meeting_number": MEETING if self.state != "idle"
                else "", "account_id": self.account, "credentials": True,
                "sdk": {"available": self.sdk, "detail": "" if self.sdk else "zoom_meeting_sdk is not importable"},
                "last_outcome": self.join_outcome}


# ---- drivers: the steps a test takes to reach a joined meeting with audio ------------

def join_args(zoom, **overrides):
    args = {"meeting_number": MEETING, "passcode": PASSCODE, "display_name": "", "binding_id": BINDING,
            "account_id": zoom.companion_link.app_account_id(CLIENT_ID), "on_behalf_token": OBF,
            "requested_by": {"id": "task:t-1", "display_name": "", "requested_via": "web", "task_id": "t-1"}}
    args.update(overrides)
    return args


def drive_to_audio(fake, loop, *, raw_allowed=True):
    loop.run_ready()
    fake.auth()
    loop.run_ready()
    fake.status("MEETING_STATUS_INMEETING")
    loop.advance(2.0)
    fake.mic()
    loop.advance(3.0)
    if raw_allowed:
        loop.advance(1.5)


def open_call_via_audio(fake, loop, bot):
    fake.mixed(b"\x01\x00" * 320)
    loop.pump(lambda: bot.state == "in_call")


def start_fake_door(token: str = HOST_TOKEN):
    """The Host Service door on a free loopback port; returns ``(state, stop)``."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route, WebSocketRoute

    state = types.SimpleNamespace(bodies=[], headers=[], hello=None, frames=[], controls=[], ws=None,
                                  loop=None, refuse=None, closed=False, peers=0)

    async def identity(request):
        if request.headers.get("x-skill-token") != token:
            return JSONResponse({"ok": False}, status_code=403)
        return JSONResponse({"ok": True, "name": "Aika", "description": ""})

    async def call_open(request):
        state.headers.append(dict(request.headers))
        state.bodies.append(await request.json())
        if state.refuse:
            return JSONResponse({"ok": False, "code": state.refuse[0], "error": state.refuse[1]}, status_code=409)
        return JSONResponse({"call_id": "call-1", "peer_path": "/call/call-1/peer"})

    async def peer(websocket):
        if websocket.headers.get("x-skill-token") != token:
            await websocket.close(code=4403)
            return
        await websocket.accept()
        state.ws, state.loop, state.closed, state.hello = websocket, asyncio.get_running_loop(), False, None
        state.peers += 1
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    state.frames.append(message["bytes"])
                elif message.get("text") is not None:
                    control = json.loads(message["text"])
                    if state.hello is None:
                        state.hello = control
                    else:
                        state.controls.append(control)
        finally:
            state.closed = True

    def send(data):
        coroutine = state.ws.send_bytes(data) if isinstance(data, bytes) else state.ws.send_text(data)
        asyncio.run_coroutine_threadsafe(coroutine, state.loop).result(5)

    def close():
        asyncio.run_coroutine_threadsafe(state.ws.close(), state.loop).result(5)

    app = Starlette(routes=[Route("/identity", identity), Route("/call/open", call_open, methods=["POST"]),
                            WebSocketRoute("/call/{call_id}/peer", peer)])
    state.url, stop = serve_app(app, "zoom-fake-door")
    state.send, state.close = send, close
    return state, stop


def serve_app(app, name: str):
    """Serve an ASGI app on a free loopback port; returns ``(url, stop)``."""
    import uvicorn

    with socket.socket() as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        port = probe_socket.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, name=name, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline, f"{name} did not start"
        time.sleep(0.02)

    def stop():
        server.should_exit = True
        thread.join(10)

    return f"http://127.0.0.1:{port}", stop
