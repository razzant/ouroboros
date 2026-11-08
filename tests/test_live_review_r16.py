"""Independent R16 boundary probes; only synthetic data and local ports."""


from ouroboros import call_protocol as cp
from ouroboros.call_peers import LiveCallHub
from tests.test_live_proactive_host import synthetic_jpeg


def test_output_flush_keeps_sources_independent_and_each_latest_frame():
    hub = LiveCallHub(cp.CallSpec('review-sources', 1, 'test', 'test'), rates=(24000, 24000))
    try:
        for peer, role in [('owner', cp.PEER_OWNER), ('guest', cp.PEER_MEETING)]:
            hub.join(cp.PeerInfo(peer, role, True, True, video=True), 24000)
            hub.next_event(0)
        data = synthetic_jpeg()
        for peer in ('owner', 'guest'):
            assert hub.receive_video(peer, cp.VideoFrame(hub.epoch, 0, 'image/jpeg', data))
            assert hub.receive_video(peer, cp.VideoFrame(hub.epoch, 1, 'image/jpeg', data))
        hub.flush()
        events = [hub.next_event(0), hub.next_event(1.1)]  # retain the hub's one-second cadence
        assert {event.peer_id for event in events} == {'owner', 'guest'}
        assert all(isinstance(event, cp.PeerVideo) and event.data == data for event in events)
        assert hub.next_event(0) is None, 'only one latest accepted frame per source'
    finally:
        hub.close('test')


def test_transport_rejects_frame_from_prior_playback_epoch():
    hub = LiveCallHub(cp.CallSpec('review-epoch', 1, 'test', 'test'), rates=(24000, 24000))
    try:
        hub.join(cp.PeerInfo('owner', cp.PEER_OWNER, True, True, video=True), 24000)
        hub.next_event(0)
        epoch = hub.epoch
        hub.flush()
        assert not hub.receive_video('owner', cp.VideoFrame(epoch, 0, 'image/jpeg', synthetic_jpeg()))
    finally:
        hub.close('test')
