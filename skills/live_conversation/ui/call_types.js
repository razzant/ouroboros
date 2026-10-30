/** Live audio uses a separate binary socket; the JSON /ws contract is unchanged. */

/**
 * Calls use Google Gemini Live; model and voice override the Settings defaults.
 * @typedef {Object} CallOpenRequest
 * @property {number} chat_id
 * @property {string=} model
 * @property {string=} voice
 * @property {string=} client_message_id
 */
/**
 * @typedef {Object} CallOpenResponse
 * @property {string} call_id
 * @property {string} ws_path
 * @property {boolean=} joined
 */
/**
 * @typedef {Object} CallRefusal
 * @property {false} ok
 * @property {string} code
 * @property {string} error
 */
/**
 * @typedef {Object} HostCallOpenRequest
 * @property {'meeting'} role
 * @property {string=} call_id
 * @property {string=} binding_id
 * @property {Object=} event
 * @property {string=} label
 * @property {boolean=} mic
 * @property {boolean=} speaker
 * @property {boolean=} video
 */
/**
 * @typedef {Object} HostCallOpenResponse
 * @property {string} call_id
 * @property {string} peer_path
 */
/**
 * @typedef {Object} CallHello
 * @property {'hello'} type
 * @property {1} protocol
 * @property {'owner'|'meeting'} role
 * @property {boolean} mic
 * @property {boolean} speaker
 * @property {boolean=} video
 * @property {number} rate
 * @property {string=} label
 */
/**
 * @typedef {Object} CallMute
 * @property {'mute'} type
 * @property {boolean} muted
 */
/**
 * @typedef {Object} CallPlayed
 * @property {'played'} type
 * @property {string} item_id
 * @property {number} played_ms
 */
/**
 * @typedef {Object} CallContext
 * @property {'context'} type
 * @property {string} text
 * @property {string} source
 */
/**
 * @typedef {Object} CallParticipant
 * @property {'participant'} type
 * @property {string} label
 * @property {boolean} speaking
 */
/**
 * @typedef {Object} CallBye
 * @property {'bye'} type
 * @property {string=} reason
 */
/**
 * A call's state; a closed one carries its reason's words (detail) and one next step (action).
 * @typedef {Object} CallState
 * @property {'state'} type
 * @property {'preparing'|'connecting'|'live'|'reconnecting'|'closing'|'closed'} state
 * @property {string} reason
 * @property {string=} engine
 * @property {string=} phase
 * @property {string=} detail
 * @property {'settings'|'retry'=} action
 */
/**
 * Where the call's turn is: an answer awaited, the engine answering, or neither (a joining surface
 * receives the latest).
 * @typedef {Object} CallTurn
 * @property {'turn'} type
 * @property {'awaiting'|'answering'|'idle'} turn
 */
/**
 * What the call is doing between its words: a tool batch running (its tool names), then idle.
 * @typedef {Object} CallActivity
 * @property {'activity'} type
 * @property {'working'|'idle'} activity
 * @property {string} label
 */
/**
 * @typedef {Object} CallCaption
 * @property {'caption'} type
 * @property {'owner'|'ouroboros'} speaker
 * @property {string} text
 * @property {boolean} final
 * @property {string} item_id
 */
/**
 * @typedef {Object} CallFlush
 * @property {'flush'} type
 * @property {number} epoch
 */
/**
 * PCM is mono int16 little-endian. Sequence wraps at 2**32, epoch at 2**16.
 * Flush stops older playback. Caption item_id identifies playback acknowledgements.
 * @typedef {Object} CallSocketContract
 * @property {string} path
 * @property {string} host_open_path
 * @property {string} host_peer_path
 * @property {string} host_auth
 * @property {number} protocol
 * @property {string} audio_header
 * @property {number} audio_header_bytes
 * @property {number} max_audio_payload_bytes
 * @property {number} max_video_payload_bytes
 * @property {number} max_control_bytes
 * @property {Object<string, string>} peer_messages
 * @property {Object<string, string>} server_messages
 */
export {};
