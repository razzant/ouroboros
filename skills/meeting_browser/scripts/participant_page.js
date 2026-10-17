/* Ouroboros meeting participant, page side.
 *
 * Runs before the meeting page's own scripts, as an init script of Ouroboros's
 * own browser profile. It does two things and decides nothing:
 *  - the microphone the page asks for is a MediaStreamAudioDestinationNode fed
 *    with the live call's output PCM, so the meeting hears Ouroboros without
 *    any audio driver; video requests get a plain "Ouroboros" card, never the
 *    computer's camera;
 *  - every remote WebRTC audio track is tapped into PCM for the call.
 * PCM crosses to the companion through bindings it exposes:
 * __ouroborosRemotePcm(base64, rate) carries meeting audio out,
 * __ouroborosParticipantEvent(name) reports facts such as "microphone_requested";
 * the companion pushes call audio in with __ouroborosFeed(base64) and drops it
 * on an interruption with __ouroborosFlush(). An AudioWorklet does the audio
 * work; where the page's content policy refuses the worklet module, a
 * ScriptProcessor does the same arithmetic on the main thread.
 */
(function installOuroborosParticipant(scope) {
  'use strict';
  if (!scope || scope.__ouroborosParticipant) return;

  var RATE = 48000;
  var TAP_CHUNK_FRAMES = 1920; // 40 ms per binding call
  // Memory bound of queued call audio. Engines stream speech faster than real time, so this
  // holds minutes, not a jitter window; it never shortens an answer in practice.
  var FEED_LIMIT_FRAMES = RATE * 120;
  var MIC_DEVICE = 'ouroboros-microphone';
  var WORKLET = [
    'var LIMIT = ' + FEED_LIMIT_FRAMES + ';',
    'class OuroborosFeed extends AudioWorkletProcessor {',
    '  constructor() {',
    '    super(); this.chunks = []; this.offset = 0; this.total = 0;',
    '    this.port.onmessage = (event) => {',
    '      if (event.data === "flush") { this.chunks = []; this.offset = 0; this.total = 0; return; }',
    '      this.chunks.push(event.data); this.total += event.data.length;',
    '      while (this.total - this.offset > LIMIT && this.chunks.length > 1) {',
    '        this.total -= this.chunks.shift().length; this.offset = 0;',
    '      }',
    '    };',
    '  }',
    '  process(inputs, outputs) {',
    '    var out = outputs[0][0]; var index = 0;',
    '    while (index < out.length && this.chunks.length) {',
    '      var chunk = this.chunks[0]; var take = Math.min(out.length - index, chunk.length - this.offset);',
    '      out.set(chunk.subarray(this.offset, this.offset + take), index);',
    '      index += take; this.offset += take;',
    '      if (this.offset >= chunk.length) { this.total -= chunk.length; this.chunks.shift(); this.offset = 0; }',
    '    }',
    '    out.fill(0, index);',
    '    return true;',
    '  }',
    '}',
    'class OuroborosTap extends AudioWorkletProcessor {',
    '  process(inputs) {',
    '    var input = inputs[0];',
    '    if (input && input.length) {',
    '      var mono = new Float32Array(input[0].length);',
    '      for (var c = 0; c < input.length; c += 1) {',
    '        for (var i = 0; i < mono.length; i += 1) mono[i] += input[c][i] / input.length;',
    '      }',
    '      this.port.postMessage(mono, [mono.buffer]);',
    '    }',
    '    return true;',
    '  }',
    '}',
    'registerProcessor("ouroboros-feed", OuroborosFeed);',
    'registerProcessor("ouroboros-tap", OuroborosTap);'
  ].join('\n');

  var facts = {
    rate: RATE, mode: '', microphoneRequests: 0, published: false, tappedTracks: 0,
    fedFrames: 0, droppedFrames: 0, remoteChunks: 0, errors: []
  };
  var context = null;
  var feed = null;
  var tapInput = null;
  var micStream = null;
  var camera = null;
  var ready = null;
  var queue = [];
  var queued = 0;
  var pending = [];
  var pendingFrames = 0;
  var ourTracks = new Set();
  var tapped = new WeakSet();
  var connections = [];

  function note(name, detail) {
    try {
      if (typeof scope.__ouroborosParticipantEvent === 'function') {
        var sent = scope.__ouroborosParticipantEvent(name, detail || '');
        if (sent && typeof sent.catch === 'function') sent.catch(function () {});
      }
    } catch (error) { /* the companion may not have exposed its bindings yet */ }
  }

  function remember(error) {
    facts.errors.push(String((error && error.message) || error));
    if (facts.errors.length > 8) facts.errors.shift();
  }

  function base64ToFloats(text) {
    var binary = scope.atob(text);
    var bytes = new Uint8Array(binary.length - (binary.length % 2));
    for (var i = 0; i < bytes.length; i += 1) bytes[i] = binary.charCodeAt(i);
    var ints = new Int16Array(bytes.buffer);
    var floats = new Float32Array(ints.length);
    for (var j = 0; j < ints.length; j += 1) floats[j] = ints[j] / 32768;
    return floats;
  }

  function floatsToBase64(floats) {
    var ints = new Int16Array(floats.length);
    for (var i = 0; i < floats.length; i += 1) {
      ints[i] = Math.max(-32768, Math.min(32767, Math.round(floats[i] * 32767)));
    }
    var bytes = new Uint8Array(ints.buffer);
    var binary = '';
    for (var offset = 0; offset < bytes.length; offset += 0x8000) {
      binary += String.fromCharCode.apply(null, bytes.subarray(offset, offset + 0x8000));
    }
    return scope.btoa(binary);
  }

  function audioContext() {
    if (!context) {
      var Context = scope.AudioContext || scope.webkitAudioContext;
      context = new Context({ sampleRate: RATE, latencyHint: 'interactive' });
    }
    if (context.state === 'suspended' && context.resume) context.resume().catch(function () {});
    return context;
  }

  function collect(mono) {
    if (!facts.tappedTracks) return; // no remote audio exists yet: nothing to carry
    pending.push(mono);
    pendingFrames += mono.length;
    if (pendingFrames < TAP_CHUNK_FRAMES) return;
    var joined = new Float32Array(pendingFrames);
    var offset = 0;
    for (var i = 0; i < pending.length; i += 1) { joined.set(pending[i], offset); offset += pending[i].length; }
    pending = [];
    pendingFrames = 0;
    if (typeof scope.__ouroborosRemotePcm !== 'function') return;
    facts.remoteChunks += 1;
    try {
      var sent = scope.__ouroborosRemotePcm(floatsToBase64(joined), RATE);
      if (sent && typeof sent.catch === 'function') sent.catch(remember);
    } catch (error) { remember(error); }
  }

  function fill(out) {
    var index = 0;
    while (index < out.length && queue.length) {
      var chunk = queue[0];
      var take = Math.min(out.length - index, chunk.data.length - chunk.offset);
      out.set(chunk.data.subarray(chunk.offset, chunk.offset + take), index);
      index += take;
      chunk.offset += take;
      queued -= take;
      if (chunk.offset >= chunk.data.length) queue.shift();
    }
    out.fill(0, index);
  }

  function silentSink(node) {
    var sink = context.createGain();
    sink.gain.value = 0;
    node.connect(sink);
    sink.connect(context.destination);
  }

  function scriptFallback() {
    feed = context.createScriptProcessor(1024, 1, 1);
    feed.onaudioprocess = function (event) { fill(event.outputBuffer.getChannelData(0)); };
    var tap = context.createScriptProcessor(1024, 2, 1);
    tap.onaudioprocess = function (event) {
      var input = event.inputBuffer;
      var mono = new Float32Array(input.length);
      for (var c = 0; c < input.numberOfChannels; c += 1) {
        var data = input.getChannelData(c);
        for (var i = 0; i < mono.length; i += 1) mono[i] += data[i] / input.numberOfChannels;
      }
      collect(mono);
    };
    tapInput = context.createGain();
    tapInput.connect(tap);
    silentSink(tap);
    facts.mode = 'script';
  }

  function setup() {
    if (ready) return ready;
    ready = (async function () {
      audioContext();
      try {
        var url = scope.URL.createObjectURL(new scope.Blob([WORKLET], { type: 'application/javascript' }));
        await context.audioWorklet.addModule(url);
        feed = new scope.AudioWorkletNode(context, 'ouroboros-feed',
          { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1] });
        var tap = new scope.AudioWorkletNode(context, 'ouroboros-tap',
          { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
        tap.port.onmessage = function (event) { collect(event.data); };
        tapInput = context.createGain();
        tapInput.connect(tap);
        silentSink(tap);
        facts.mode = 'worklet';
      } catch (error) {
        remember(error);
        scriptFallback();
      }
    })();
    return ready;
  }

  async function microphone() {
    await setup();
    if (!micStream) {
      var destination = context.createMediaStreamDestination();
      destination.channelCount = 1;
      feed.connect(destination);
      micStream = destination.stream;
    }
    return micStream;
  }

  function cameraStream() {
    if (!camera) {
      var canvas = scope.document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 360;
      var paint = function () {
        var g = canvas.getContext('2d');
        g.fillStyle = '#101418';
        g.fillRect(0, 0, canvas.width, canvas.height);
        g.fillStyle = '#e8eef4';
        g.font = '48px sans-serif';
        g.textAlign = 'center';
        g.fillText('Ouroboros', canvas.width / 2, canvas.height / 2 + 16);
      };
      paint();
      scope.setInterval(paint, 500); // a capture stream emits frames only when the canvas is painted
      camera = canvas.captureStream(5);
    }
    return new scope.MediaStream(camera.getVideoTracks().map(function (track) { return track.clone(); }));
  }

  function tapTrack(track) {
    if (!track || track.kind !== 'audio' || tapped.has(track) || ourTracks.has(track)) return;
    tapped.add(track);
    setup().then(function () {
      var stream = new scope.MediaStream([track]);
      // Chrome renders a remote WebRTC track into Web Audio only while a media element consumes it.
      var keeper = scope.document.createElement('audio');
      keeper.muted = true;
      keeper.srcObject = stream;
      var played = keeper.play && keeper.play();
      if (played && typeof played.catch === 'function') played.catch(function () {});
      var source = context.createMediaStreamSource(stream);
      source.connect(tapInput);
      facts.tappedTracks += 1;
      note('remote_audio');
      track.addEventListener('ended', function () {
        try { source.disconnect(); } catch (error) { /* already disconnected */ }
        keeper.srcObject = null;
        facts.tappedTracks = Math.max(0, facts.tappedTracks - 1);
      });
    }).catch(remember);
  }

  function markPublished(track) {
    if (ourTracks.has(track) && !facts.published) {
      facts.published = true;
      note('microphone_published');
    }
  }

  var media = scope.navigator && scope.navigator.mediaDevices;
  if (media) {
    var nativeGetUserMedia = media.getUserMedia ? media.getUserMedia.bind(media) : null;
    var nativeEnumerate = media.enumerateDevices ? media.enumerateDevices.bind(media) : null;
    var replacedGetUserMedia = async function getUserMedia(constraints) {
      var wanted = constraints || {};
      if (!wanted.audio) {
        if (wanted.video) return cameraStream();
        return nativeGetUserMedia(wanted);
      }
      var mic = await microphone();
      var stream = new scope.MediaStream();
      mic.getAudioTracks().forEach(function (track) {
        var copy = track.clone();
        ourTracks.add(copy);
        stream.addTrack(copy);
      });
      if (wanted.video) cameraStream().getVideoTracks().forEach(function (track) { stream.addTrack(track); });
      facts.microphoneRequests += 1;
      note('microphone_requested');
      return stream;
    };
    // The prototype too: a page calling MediaDevices.prototype.getUserMedia gets the same microphone.
    var MediaDevicesPrototype = scope.MediaDevices && scope.MediaDevices.prototype;
    if (MediaDevicesPrototype && MediaDevicesPrototype.getUserMedia) {
      MediaDevicesPrototype.getUserMedia = replacedGetUserMedia;
    }
    media.getUserMedia = replacedGetUserMedia;
    media.enumerateDevices = async function enumerateDevices() {
      var devices = nativeEnumerate ? await nativeEnumerate() : [];
      var microphoneDevice = {
        deviceId: MIC_DEVICE, groupId: MIC_DEVICE, kind: 'audioinput', label: 'Ouroboros',
        toJSON: function () { return { deviceId: MIC_DEVICE, groupId: MIC_DEVICE, kind: 'audioinput', label: 'Ouroboros' }; }
      };
      return [microphoneDevice].concat(devices.filter(function (device) { return device.kind !== 'audioinput'; }));
    };
  }

  var NativePeerConnection = scope.RTCPeerConnection;
  if (NativePeerConnection) {
    var proto = NativePeerConnection.prototype;
    var nativeAddTrack = proto.addTrack;
    var nativeAddTransceiver = proto.addTransceiver;
    if (nativeAddTrack) {
      proto.addTrack = function (track) { markPublished(track); return nativeAddTrack.apply(this, arguments); };
    }
    if (nativeAddTransceiver) {
      proto.addTransceiver = function (trackOrKind) { markPublished(trackOrKind); return nativeAddTransceiver.apply(this, arguments); };
    }
    if (scope.RTCRtpSender && scope.RTCRtpSender.prototype.replaceTrack) {
      var nativeReplace = scope.RTCRtpSender.prototype.replaceTrack;
      scope.RTCRtpSender.prototype.replaceTrack = function (track) { markPublished(track); return nativeReplace.apply(this, arguments); };
    }
    var Tapped = class RTCPeerConnection extends NativePeerConnection {
      constructor() {
        super(...arguments);
        connections.push(this);
        this.addEventListener('track', function (event) { tapTrack(event.track); });
      }
    };
    scope.RTCPeerConnection = Tapped;
    if (scope.webkitRTCPeerConnection) scope.webkitRTCPeerConnection = Tapped;
  }

  async function outboundAudioBytes() {
    var total = 0;
    for (var i = 0; i < connections.length; i += 1) {
      try {
        var report = await connections[i].getStats();
        report.forEach(function (entry) {
          if (entry.type === 'outbound-rtp' && entry.kind === 'audio') total += entry.bytesSent || 0;
        });
      } catch (error) { /* a closed connection reports nothing */ }
    }
    return total;
  }

  Object.defineProperty(scope, '__ouroborosFeed', {
    value: function (text) {
      if (!feed) return false;
      var floats = base64ToFloats(text);
      facts.fedFrames += floats.length;
      if (facts.mode === 'worklet') {
        feed.port.postMessage(floats, [floats.buffer]);
        return true;
      }
      queue.push({ data: floats, offset: 0 });
      queued += floats.length;
      while (queued > FEED_LIMIT_FRAMES && queue.length > 1) {
        var dropped = queue.shift();
        queued -= dropped.data.length - dropped.offset;
        facts.droppedFrames += dropped.data.length - dropped.offset;
      }
      return true;
    }
  });
  Object.defineProperty(scope, '__ouroborosFlush', {
    value: function () {
      if (facts.mode === 'worklet' && feed) feed.port.postMessage('flush');
      queue = [];
      queued = 0;
      return true;
    }
  });
  Object.defineProperty(scope, '__ouroborosParticipant', {
    value: {
      status: function () {
        var live = false;
        ourTracks.forEach(function (track) { if (track.readyState === 'live' && track.enabled) live = true; });
        return Object.assign({}, facts, {
          micLive: live, connections: connections.length,
          contextState: context ? context.state : 'none', errors: facts.errors.slice()
        });
      },
      outboundAudioBytes: outboundAudioBytes
    }
  });
})(typeof window !== 'undefined' ? window : globalThis);
