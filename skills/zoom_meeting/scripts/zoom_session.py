"""One Zoom meeting through the Zoom Meeting SDK, ported from Hope's zoom_service.

The sequence is Hope's production one: ``InitSDK`` for ``https://zoom.us``, SDK auth
with an HS256 app JWT, ``Join`` as a user without login (video off, audio on,
auto-join audio), then ``JoinVoip`` and unmute, the virtual microphone as the external
audio source, the recording controller (raw recording at once when allowed, else a
local-recording privilege request the meeting host approves in Zoom), raw recording,
and the raw audio delegate: mixed meeting audio at 32 kHz mono PCM16, plus per-user
one-way audio by node id.

Every SDK call runs on the GLib loop thread. Callbacks may arrive on SDK threads, so
each one records a fact and schedules its next step on the loop. The pauses between
phases are Hope's measured settling times. One ``ZoomSession`` serves one join:
``ended`` fires exactly once, after ``CleanUPSDK``, and a late callback from a
finished session is ignored.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

log = logging.getLogger("zoom_bot.session")

WEB_DOMAIN = "https://zoom.us"
JWT_TTL_SEC = 7200
RAW_AUDIO_RATE = 32000


@dataclass(frozen=True)
class SessionTiming:
    """Hope's settling pauses (milliseconds); tests shorten them."""

    audio_setup_ms: int = 2000
    recording_setup_ms: int = 3000
    raw_recording_ms: int = 500
    subscribe_ms: int = 1000
    leave_cleanup_ms: int = 1500


@dataclass
class JoinRequest:
    meeting_number: int
    display_name: str
    passcode: str = ""
    on_behalf_token: str = ""


def build_sdk_jwt(client_id: str, client_secret: str, *, now: int, ttl: int = JWT_TTL_SEC) -> str:
    """The Meeting SDK app JWT: HS256 over ``appKey``, ``iat``, ``exp`` and ``tokenExp``."""
    import jwt  # PyJWT, installed into the skill's isolated environment

    payload = {"appKey": client_id, "iat": int(now), "exp": int(now) + ttl, "tokenExp": int(now) + ttl}
    token = jwt.encode(payload, client_secret, algorithm="HS256")
    return token if isinstance(token, str) else token.decode("ascii")


def sdk_name(value: Any) -> str:
    """An SDK enum's own name (``MEETING_FAIL_PASSWORD_ERR``), the raw value otherwise."""
    return str(getattr(value, "name", "") or value)


def enum_member_name(enum_cls: Any, value: Any) -> str:
    if enum_cls is not None:
        with contextlib.suppress(Exception):
            return str(enum_cls(int(value)).name)
    return sdk_name(value)


def _flush_c_stdio() -> None:
    with contextlib.suppress(Exception):
        libc = ctypes.CDLL(None)
        libc.fflush.argtypes = [ctypes.c_void_p]
        libc.fflush(None)


@contextlib.contextmanager
def quiet_stdout():
    """Point fd 1 at /dev/null around a native call that prints secrets.

    zoom-meeting-sdk 0.0.27 wraps ``IMeetingService.Join`` in ``printJoinParam``, which
    writes the passcode and the ZAK to stdout through ``std::cout``, and the host keeps a
    companion's stdout. C stdio is flushed before fd 1 is restored, so nothing printed
    inside reaches the real descriptor later from a buffer.
    """
    with contextlib.suppress(Exception):
        sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        _flush_c_stdio()
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


_END_REASONS = {
    "EndMeetingReason_KickByHost": "the host removed {name} from the meeting",
    "EndMeetingReason_EndByHost": "the host ended the meeting",
    "EndMeetingReason_JBHTimeOut": "the meeting ended: the host did not join in time",
    "EndMeetingReason_NoAttendee": "the meeting ended: nobody else was in it",
    "EndMeetingReason_FreeMeetingTimeOut": "the meeting ended: its free-plan time ran out",
    "EndMeetingReason_DueToAuthorizedUserLeave": (
        "the meeting ended for {name}: the user whose OBF token admitted {name} left"),
}


class ZoomSession:
    """Drive one join; report through ``listener`` (see ``zoom_bot.ZoomBot``)."""

    def __init__(self, sdk: Any, loop: Any, listener: Any, *, timing: SessionTiming = SessionTiming(),
                 clock: Callable[[], float] = time.time,
                 register_secret: Callable[[str], None] = lambda _value: None) -> None:
        self._sdk = sdk
        self._loop = loop
        self._listener = listener
        self._timing = timing
        self._clock = clock
        self._register_secret = register_secret
        self.request: Optional[JoinRequest] = None
        self.my_user_id: Optional[int] = None
        self.topic = ""
        self._names: Dict[int, str] = {}
        self._anchors: Dict[str, Any] = {}  # SDK callback objects stay referenced while the SDK holds them
        self._initialized = False
        self._meeting_service: Any = None
        self._join_requested = False
        self._audio_helper: Any = None
        self._recording_ctrl: Any = None
        self._setup_done = False
        self._raw_scheduled = False
        self._recording_started = False
        self._subscribed = False
        self._leaving = False
        self._ended = False
        self._outcome = ""

    # ---- public: called on the loop thread ------------------------------------

    @property
    def ended(self) -> bool:
        return self._ended

    def start(self, request: JoinRequest, client_id: str, client_secret: str) -> bool:
        sdk = self._sdk
        self.request = request
        self._emit("session_state", "joining")
        try:
            param = sdk.InitParam()
            param.strWebDomain = WEB_DOMAIN
            param.enableGenerateDump = False  # a crash dump would hold the passcode and the JWT
            err = sdk.InitSDK(param)
            if err != sdk.SDKERR_SUCCESS:
                return self._finish(f"the Zoom Meeting SDK did not start: {sdk_name(err)}")
            self._initialized = True
            token = build_sdk_jwt(client_id, client_secret, now=int(self._clock()))
            self._register_secret(token)
            auth = sdk.CreateAuthService()
            callbacks = sdk.AuthServiceEventCallbacks(onAuthenticationReturnCallback=self._on_auth)
            context = sdk.AuthContext()
            context.jwt_token = token
            # The binding may keep raw char* pointers into these strings: hold them until cleanup.
            self._anchors.update(auth_service=auth, auth=callbacks, auth_context=context, jwt=token)
            auth.SetEvent(callbacks)
            err = auth.SDKAuth(context)
            if err != sdk.SDKERR_SUCCESS:
                return self._finish(f"Zoom did not take the authentication request: {sdk_name(err)}")
        except Exception as exc:
            log.exception("SDK start failed")
            return self._finish(f"the Zoom Meeting SDK did not start: {type(exc).__name__}")
        return False

    def leave(self, outcome: str = "") -> bool:
        """Leave (or abandon a join in progress), then clean up; the first outcome wins."""
        if self._ended:
            return False
        if outcome and not self._outcome:
            self._outcome = outcome
        if self._leaving:
            return False
        self._leaving = True
        self._emit("session_state", "leaving")
        if not self._join_requested:
            return self.cleanup()
        try:
            self._meeting_service.Leave(self._sdk.LEAVE_MEETING)
        except Exception:
            log.exception("Leave failed")
        self._loop.timeout_add(self._timing.leave_cleanup_ms, self.cleanup)
        return False

    def cleanup(self) -> bool:
        """Unsubscribe and ``CleanUPSDK``; report the outcome once. Idempotent."""
        if self._ended:
            return False
        self._ended = True
        if self._subscribed and self._audio_helper is not None:
            with contextlib.suppress(Exception):
                self._audio_helper.unSubscribe()
        if self._initialized:
            try:
                self._sdk.CleanUPSDK()
            except Exception:
                log.exception("CleanUPSDK failed")
        if self.request is not None:
            self.request.passcode = ""
            self.request.on_behalf_token = ""
        self._anchors.clear()
        self._meeting_service = self._audio_helper = self._recording_ctrl = None
        self._listener.ended(self._outcome or "left the meeting")
        return False

    def participant_name(self, node_id: int) -> str:
        if node_id in self._names:
            return self._names[node_id]
        name = ""
        if self._meeting_service is not None:
            with contextlib.suppress(Exception):
                user = self._meeting_service.GetMeetingParticipantsController().GetUserByUserID(node_id)
                name = str(user.GetUserName() or "") if user is not None else ""
        if name:
            self._names[node_id] = name
        return name or f"participant {node_id}"

    # ---- sequence --------------------------------------------------------------

    def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Report one fact unless this session already ended (late SDK callbacks)."""
        if not self._ended:
            getattr(self._listener, event)(*args, **kwargs)

    def _name(self) -> str:
        return self.request.display_name if self.request is not None else "the bot"

    def _finish(self, outcome: str) -> bool:
        if self._join_requested:
            return self.leave(outcome)
        if outcome and not self._outcome:
            self._outcome = outcome
        return self.cleanup()

    def _later(self, step: Callable[[], Any], delay_ms: int = 0) -> None:
        def run() -> bool:
            if not self._ended and not self._leaving:
                step()
            return False

        if delay_ms:
            self._loop.timeout_add(delay_ms, run)
        else:
            self._loop.idle_add(run)

    def _on_auth(self, result: Any) -> None:
        self._later(lambda: self._after_auth(result))

    def _after_auth(self, result: Any) -> None:
        if result != self._sdk.AUTHRET_SUCCESS:
            self._finish(f"Zoom refused the app credentials: {sdk_name(result)}")
            return
        self._join()

    def _join(self) -> None:
        sdk, request = self._sdk, self.request
        try:
            service = sdk.CreateMeetingService()
            callbacks = sdk.MeetingServiceEventCallbacks(onMeetingStatusChangedCallback=self._on_status)
            self._anchors.update(meeting_service=service, meeting=callbacks)
            service.SetEvent(callbacks)
            self._meeting_service = service
            try:
                settings = sdk.CreateSettingService()
                self._anchors["settings"] = settings
                settings.GetAudioSettings().EnableAutoJoinAudio(True)
            except Exception:
                log.warning("auto-join audio setting unavailable")
            join = sdk.JoinParam()
            self._anchors["join_param"] = join
            join.userType = sdk.SDKUserType.SDK_UT_WITHOUT_LOGIN
            param = join.param
            param.meetingNumber = int(request.meeting_number)
            param.userName = request.display_name
            param.psw = request.passcode
            param.isVideoOff = True
            param.isAudioOff = False
            param.isMyVoiceInMix = False
            param.isAudioRawDataStereo = False
            param.eAudioRawdataSamplingRate = sdk.AudioRawdataSamplingRate.AudioRawdataSamplingRate_32K
            if request.on_behalf_token:
                param.onBehalfToken = request.on_behalf_token
            with quiet_stdout():
                err = service.Join(join)
        except Exception as exc:
            log.exception("Join failed")
            self._finish(f"the meeting refused the join: {type(exc).__name__}")
            return
        if err != sdk.SDKERR_SUCCESS:
            self._finish(f"the meeting refused the join: {sdk_name(err)}")
            return
        self._join_requested = True

    def _on_status(self, status: Any, result: Any = 0) -> None:
        def run() -> bool:
            if not self._ended:
                self._handle_status(status, result)
            return False

        self._loop.idle_add(run)

    def _handle_status(self, status: Any, result: Any) -> None:
        sdk = self._sdk
        if status == sdk.MEETING_STATUS_FAILED:
            self._meeting_service_done(
                f"the meeting refused the join: {enum_member_name(getattr(sdk, 'MeetingFailCode', None), result)}")
            return
        if status == sdk.MEETING_STATUS_ENDED:
            reason = enum_member_name(getattr(sdk, "MeetingEndReason", None), result)
            self._meeting_service_done(
                _END_REASONS.get(reason, "the meeting ended ({reason})").format(name=self._name(), reason=reason))
            return
        if self._leaving:
            return
        if status == sdk.MEETING_STATUS_INMEETING:
            if self._setup_done:
                self._emit("session_state", "reconnected")
                return
            self._setup_done = True
            with contextlib.suppress(Exception):
                self.topic = str(self._meeting_service.GetMeetingInfo().GetMeetingTopic() or "")
            self._emit("session_state", "in_meeting")
            self._later(self._setup_audio, self._timing.audio_setup_ms)
            return
        for attr, state in (("MEETING_STATUS_WAITINGFORHOST", "waiting_for_host"),
                            ("MEETING_STATUS_IN_WAITING_ROOM", "waiting_room"),
                            ("MEETING_STATUS_RECONNECTING", "reconnecting")):
            if status == getattr(sdk, attr, None):
                self._emit("session_state", state)
                return

    def _meeting_service_done(self, outcome: str) -> None:
        if outcome and not self._outcome:
            self._outcome = outcome
        self.cleanup()

    def _setup_audio(self) -> None:
        sdk, service = self._sdk, self._meeting_service
        try:
            log.info("JoinVoip: %s", sdk_name(service.GetMeetingAudioController().JoinVoip()))
        except Exception:
            log.exception("JoinVoip failed")
        try:
            me = service.GetMeetingParticipantsController().GetMySelfUser()
            self.my_user_id = int(me.GetUserID())
            if me.IsAudioMuted():
                log.info("UnMuteAudio: %s",
                         sdk_name(service.GetMeetingAudioController().UnMuteAudio(self.my_user_id)))
        except Exception:
            log.exception("UnMuteAudio failed")
        try:
            self._audio_helper = sdk.GetAudioRawdataHelper()
            mic = sdk.ZoomSDKVirtualAudioMicEventCallbacks(
                onMicInitializeCallback=lambda sender: self._emit("mic_ready", sender),
                onMicStartSendCallback=lambda: self._emit("mic_sending", True),
                onMicStopSendCallback=lambda: self._emit("mic_sending", False),
                onMicUninitializedCallback=lambda: self._emit("mic_ready", None),
            )
            self._anchors.update(audio_helper=self._audio_helper, mic=mic)
            self._audio_helper.setExternalAudioSource(mic)
        except Exception:
            log.exception("virtual microphone setup failed")
        self._watch_chat()
        self._watch_own_audio()
        self._later(self._setup_recording, self._timing.recording_setup_ms)

    def _setup_recording(self) -> None:
        sdk = self._sdk
        try:
            ctrl = self._meeting_service.GetMeetingRecordingController()
            callbacks = sdk.MeetingRecordingCtrlEventCallbacks(
                onRecordPrivilegeChangedCallback=self._on_privilege_changed,
                onLocalRecordingPrivilegeRequestStatusCallback=self._on_privilege_answer,
            )
            self._anchors.update(recording_ctrl=ctrl, recording=callbacks)
            ctrl.SetEvent(callbacks)
            self._recording_ctrl = ctrl
            if ctrl.CanStartRawRecording() == sdk.SDKERR_SUCCESS:
                self._schedule_raw_recording()
                return
            err = ctrl.RequestLocalRecordingPrivilege()
        except Exception as exc:
            log.exception("recording controller failed")
            self._finish(f"the meeting's recording controller failed: {type(exc).__name__}")
            return
        if err != sdk.SDKERR_SUCCESS:
            self._finish(f"the meeting does not let {self._name()} ask for its audio: {sdk_name(err)}")
            return
        self._emit("session_state", "waiting_for_audio")

    def _on_privilege_changed(self, can_record: Any) -> None:
        def step() -> None:
            if can_record:
                self._schedule_raw_recording()
            elif self._recording_started:
                self._finish(f"the host withdrew {self._name()}'s audio permission; {self._name()} left the meeting")

        self._later(step)

    def _on_privilege_answer(self, status: Any) -> None:
        name = sdk_name(status)

        def step() -> None:
            if name.endswith("_Granted"):
                self._schedule_raw_recording()
            elif name.endswith("_Denied"):
                self._finish(f"the host declined {self._name()}'s audio request; {self._name()} left the meeting")
            elif name.endswith("_Timeout"):
                self._finish(f"the host did not answer {self._name()}'s audio request in time; "
                             f"{self._name()} left the meeting")

        self._later(step)

    def _schedule_raw_recording(self) -> None:
        if self._raw_scheduled:
            return
        self._raw_scheduled = True
        self._later(self._start_raw_recording, self._timing.raw_recording_ms)

    def _start_raw_recording(self) -> None:
        try:
            err = self._recording_ctrl.StartRawRecording()
        except Exception as exc:
            log.exception("StartRawRecording failed")
            self._finish(f"Zoom did not start raw audio: {type(exc).__name__}")
            return
        if err != self._sdk.SDKERR_SUCCESS:
            self._finish(f"Zoom did not start raw audio: {sdk_name(err)}")
            return
        self._recording_started = True
        self._later(self._subscribe, self._timing.subscribe_ms)

    def _subscribe(self) -> None:
        sdk = self._sdk
        try:
            helper = self._audio_helper or sdk.GetAudioRawdataHelper()
            delegate = sdk.ZoomSDKAudioRawDataDelegateCallbacks(
                onMixedAudioRawDataReceivedCallback=self._on_mixed_audio,
                onOneWayAudioRawDataReceivedCallback=self._on_one_way_audio,
            )
            self._anchors.update(audio_helper=helper, delegate=delegate)
            self._audio_helper = helper
            err = helper.subscribe(delegate, False)
        except Exception as exc:
            log.exception("audio subscribe failed")
            self._finish(f"Zoom did not deliver meeting audio: {type(exc).__name__}")
            return
        if err != sdk.SDKERR_SUCCESS:
            self._finish(f"Zoom did not deliver meeting audio: {sdk_name(err)}")
            return
        self._subscribed = True
        self._emit("session_state", "audio_ready")

    def _watch_chat(self) -> None:
        try:
            ctrl = self._meeting_service.GetMeetingChatController()
            callbacks = self._sdk.MeetingChatEventCallbacks(onChatMsgNotificationCallback=self._on_chat)
            self._anchors.update(chat_ctrl=ctrl, chat=callbacks)
            ctrl.SetEvent(callbacks)
        except Exception:
            log.info("meeting chat callbacks are unavailable")

    def _watch_own_audio(self) -> None:
        try:
            ctrl = self._meeting_service.GetMeetingAudioController()
            callbacks = self._sdk.MeetingAudioCtrlEventCallbacks(onUserAudioStatusChangeCallback=self._on_audio_status)
            self._anchors.update(audio_ctrl=ctrl, audio_status=callbacks)
            ctrl.SetEvent(callbacks)
        except Exception:
            log.info("audio status callbacks are unavailable")

    # ---- raw data and meeting facts (any thread) --------------------------------

    def _on_mixed_audio(self, data: Any) -> None:
        if self._ended:
            return
        try:
            pcm = bytes(data.GetBuffer() or b"")
            rate = int(data.GetSampleRate() or RAW_AUDIO_RATE)
        except Exception:
            return
        if pcm:
            self._emit("mixed_audio", pcm, rate)

    def _on_one_way_audio(self, _data: Any, node_id: Any) -> None:
        if int(node_id) != self.my_user_id:
            self._emit("participant_audio", int(node_id))

    def _on_chat(self, info: Any, content: Any = None) -> None:
        if self._ended:
            return
        try:
            if self.my_user_id is not None and int(info.GetSenderUserId()) == self.my_user_id:
                return
            text = content if isinstance(content, str) and content else info.GetContent()
            self._emit("chat", str(info.GetSenderDisplayName() or ""), str(text or ""), bool(info.IsChatToAll()))
        except Exception:
            log.exception("chat callback failed")

    def _on_audio_status(self, statuses: Any, _detail: Any = None) -> None:
        if self._ended:
            return
        for status in statuses or ():
            try:
                if int(status.GetUserId()) != self.my_user_id:
                    continue
                name = sdk_name(status.GetStatus())
            except Exception:
                continue
            if name in ("Audio_Muted_ByHost", "Audio_MutedAll_ByHost"):
                self._emit("own_audio", muted_by_host=True)
            elif name in ("Audio_UnMuted_ByHost", "Audio_UnMutedAll_ByHost"):
                self._emit("own_audio", muted_by_host=False)
