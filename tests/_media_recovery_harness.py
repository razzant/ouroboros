"""Real chat media DOM and decoders, with a task-owned HTTP source that can recover.

Shared by the browser regression and the isolated native WKWebView qualification.
No Ouroboros server, credentials or owner state. The MP4 is an eight-second generated
160x90 H.264 test pattern with a silent AAC track (not a muted media element):
ffmpeg -f lavfi -i testsrc2=size=160x90:rate=10 -f lavfi -i anullsrc=r=44100:cl=stereo
-t 8 -c:v libx264 -pix_fmt yuv420p -preset fast -crf 36 -c:a aac -movflags +faststart.
"""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit
import json
import re

REPO = Path(__file__).resolve().parents[1]
PAGE = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="/static/ui.css"><link rel="stylesheet" href="/static/style.css">
<style>body {display:block; margin:24px;} #media-host {max-width:700px;} pre {white-space:pre-wrap;}</style>
</head><body><h1>Chat video recovery</h1><div id="media-host"></div><pre id="facts">Preparing</pre>
<script type="module">
import {createChatMedia} from '/static/modules/chat_media.js';
window.controller = createChatMedia({chatSessionId:'qa', durableChatMediaUrl:u=>u||'',
  formatMsgTime:()=>null, insertMessageNode:n=>document.querySelector('#media-host').append(n),
  senderLabel:()=> 'Owner', stampNodeTimestamp:()=>{}});
window.bubble = controller.buildMediaBubble({type:'video', role:'assistant',
  caption:'A recoverable local source', mime:'video/mp4', download_url:'/clip.mp4'});
document.querySelector('#media-host').append(bubble);
window.media = bubble.querySelector('video'); window.observations=[]; window.clicks=[];
window.mediaFacts = () => ({error:media.error?.code||null, network:media.networkState,
 ready:media.readyState, time:media.currentTime, rate:media.playbackRate, muted:media.muted,
 paused:media.paused, connected:media.isConnected, source:media.currentSrc,
 userAgent:navigator.userAgent});
function record(event) {const facts={event,...mediaFacts()}; observations.push(facts);
 document.querySelector('#facts').textContent=JSON.stringify(facts,null,2);}
for (const name of ['error','loadedmetadata','play','playing','pause','emptied','ended'])
 media.addEventListener(name,()=>record(name));
bubble.addEventListener('click',e=>{clicks.push({trusted:e.isTrusted,
 active:navigator.userActivation?.isActive, action:e.target.closest('[data-media-action]')?.dataset.mediaAction||'surface'});},true);
window.releasePlayer=()=>{controller.release(bubble);bubble.remove();};
record('mounted');
</script></body></html>"""


@contextmanager
def media_server(*, repo=REPO, port=0):
    state = {"available": False, "requests": [], "reports": []}
    clip = (repo / "tests" / "fixtures" / "media-recovery.mp4").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, code, body, mime, headers=()):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # A decoder may stop its speculative range read.

        def do_POST(self):
            if self.path == "/ready":
                state["available"] = True
                self.send(200, b'{"ready":true}', "application/json")
            elif self.path == "/report":
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                state["reports"].append(json.loads(body))
                self.send(200, b'{}', "application/json")
            else:
                self.send(404, b"", "text/plain")

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self.send(200, PAGE.encode(), "text/html")
            elif path == "/state":
                self.send(200, json.dumps(state).encode(), "application/json")
            elif path == "/clip.mp4":
                state["requests"].append({"range": self.headers.get("Range"), "available": state["available"]})
                if not state["available"]:
                    self.send(503, b"Temporarily unavailable", "text/plain")
                    return
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
                if match:
                    start = int(match[1])
                    end = min(int(match[2]) if match[2] else len(clip)-1, len(clip)-1)
                    if start > end:
                        self.send(416, b"", "video/mp4", [("Content-Range", f"bytes */{len(clip)}")])
                        return
                    self.send(206, clip[start:end+1], "video/mp4", [
                        ("Content-Range", f"bytes {start}-{end}/{len(clip)}"), ("Accept-Ranges", "bytes")])
                else:
                    self.send(200, clip, "video/mp4", [("Accept-Ranges", "bytes")])
            elif path.startswith('/static/'):
                source = (repo / "web" / path.removeprefix('/static/')).resolve()
                if source.is_relative_to((repo / "web").resolve()) and source.is_file():
                    mime = "text/css" if source.suffix == ".css" else "text/javascript"
                    self.send(200, source.read_bytes(), mime)
                else:
                    self.send(404, b"", "text/plain")
            else:
                self.send(404, b"", "text/plain")

        do_HEAD = do_GET

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/", state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        assert not worker.is_alive(), "media fixture server did not stop"
