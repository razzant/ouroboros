/** Dependency-free JSDoc mirror of `ouroboros.gateway.call_contracts`: call admission (`POST /api/call/open`,
 *  the host's meeting open) and the `/ws/call/{call_id}` controls; `api_types.js` indexes the envelopes. */

/**
 * @typedef {Object} CallOpenRequest
 * @property {number} chat_id
 * @property {string} client_message_id
 * @property {string=} feature
 */
/**
 * @typedef {Object} CallOpenResponse
 * @property {string} call_id
 * @property {string} ws_path
 * @property {boolean} vision
 * @property {boolean=} joined
 */
/**
 * @typedef {Object} CallRefusal
 * @property {false} ok
 * @property {string} code
 * @property {string} error
 */
/**
 * @typedef {Object} ConversationFinishedOutbound
 * @property {"conversation_finished"} type
 * @property {number} chat_id
 * @property {string} call_id
 * @property {string} task_id
 * @property {string} reason
 */
/**
 * Separate media socket; control schemas are in GET /api/contract.schemas.
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
 * @property {Object.<string,string>} peer_messages
 * @property {Object.<string,string>} server_messages
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
 * @typedef {Object} CallShare
 * @property {'share'} type
 * @property {boolean} sharing
 * @property {'screen'|'camera'} kind
 */
/**
 * @typedef {Object} CallBye
 * @property {'bye'} type
 * @property {string=} reason
 */
/**
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
 * @typedef {Object} CallActivity
 * @property {'activity'} type
 * @property {'working'|'idle'} activity
 * @property {string} label
 */
/**
 * @typedef {Object} CallTurn
 * @property {'turn'} type
 * @property {'awaiting'|'answering'|'idle'} turn
 */
/**
 * @typedef {Object} CallCaption
 * @property {'caption'} type
 * @property {'owner'|'ouroboros'} speaker
 * @property {string} text
 * @property {boolean} final
 * @property {string} item_id
 * @property {boolean=} replace
 */
/**
 * @typedef {Object} CallFlush
 * @property {'flush'} type
 * @property {number} epoch
 */
