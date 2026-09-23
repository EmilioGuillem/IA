# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Local visual laboratory for conversations between two model backends."""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from code_agent.model_backend import GenerationConfig, create_backend
from code_agent.prompts import build_chat_prompt


HTML = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Conversation Lab</title><style>
:root{color-scheme:dark;--bg:#101316;--panel:#191e23;--line:#303941;--ink:#edf2f4;--muted:#9daab2;--a:#78c6a3;--b:#f2b880;--danger:#f07d7d}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#26352f 0,transparent 35%),var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}
header{padding:24px clamp(16px,5vw,64px) 12px;display:flex;justify-content:space-between;align-items:end;border-bottom:1px solid var(--line)}h1{margin:0;font-size:clamp(24px,4vw,42px);letter-spacing:.02em}small,.muted{color:var(--muted)}main{max-width:1300px;margin:auto;padding:18px clamp(16px,5vw,64px);display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:18px}.feed{min-height:62vh;max-height:70vh;overflow:auto;padding:4px}.turn{border-left:3px solid var(--line);padding:13px 16px;margin:10px 0;background:rgba(25,30,35,.9);border-radius:4px}.turn.a{border-color:var(--a)}.turn.b{border-color:var(--b)}.turn.human{border-color:#a995dc}.label{font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.1em}.body{white-space:pre-wrap;margin-top:6px}.meta{color:var(--muted);font-size:12px;margin-top:8px}.panel{background:var(--panel);border:1px solid var(--line);padding:16px;border-radius:6px;height:max-content}.panel h2{font-size:16px;margin:0 0 12px}.row{display:flex;gap:8px;margin:9px 0}button,input,textarea{font:inherit;color:var(--ink);background:#11161a;border:1px solid var(--line);border-radius:4px;padding:9px}input,textarea{width:100%}textarea{min-height:80px;resize:vertical}button{cursor:pointer}button:hover{border-color:var(--a)}button.primary{background:#255b49;border-color:var(--a)}button.stop{background:#642f35;border-color:var(--danger)}.status{padding:9px;background:#11161a;border-radius:4px;color:var(--muted);margin-bottom:12px}.status.running{color:var(--a)}.error{color:var(--danger)}@media(max-width:850px){main{grid-template-columns:1fr}.feed{max-height:none}}
</style></head><body><header><div><h1>Conversation Lab</h1><small>Dos modelos locales, una conversación observable</small></div><div id="status" class="status">idle</div></header><main><section><div id="feed" class="feed"></div><div class="panel"><h2>Intervención humana</h2><textarea id="human" placeholder="Escribe una instrucción o pregunta para ambos modelos..."></textarea><div class="row"><button class="primary" onclick="sendHuman()">Enviar al diálogo</button><button onclick="clearHuman()">Limpiar</button></div></div></section><aside class="panel"><h2>Control</h2><div class="row"><button class="primary" onclick="start()">Iniciar</button><button class="stop" onclick="stop()">Parar</button></div><label>Turnos máximos<input id="rounds" type="number" min="1" max="1000" value="20"></label><label>Prompt inicial<textarea id="prompt">Analizad una idea interesante y desarrolladla con argumentos distintos.</textarea></label><p class="muted">Cada turno se guarda inmediatamente. Parar evita nuevos turnos.</p></aside></main><script>
let seen=0, timer=null;
async function api(path, options={}){let r=await fetch(path,{headers:{'Content-Type':'application/json'},...options});return r.json()}
function render(s){document.getElementById('status').textContent=s.running?'running':'idle';document.getElementById('status').className='status '+(s.running?'running':'');let f=document.getElementById('feed');if(s.messages.length!==seen){f.innerHTML=s.messages.map(m=>`<article class="turn ${m.role}"><div class="label">${m.label}</div><div class="body">${escapeHtml(m.content)}</div><div class="meta">turno ${m.turn} · ${m.timestamp}</div></article>`).join('');f.scrollTop=f.scrollHeight;seen=s.messages.length}}
function escapeHtml(x){return x.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')}
async function poll(){let s=await api('/api/state');render(s);if(!s.running&&timer){clearInterval(timer);timer=null}}
async function start(){await api('/api/start',{method:'POST',body:JSON.stringify({rounds:+document.getElementById('rounds').value,prompt:document.getElementById('prompt').value})});if(!timer)timer=setInterval(poll,500);poll()}
async function stop(){await api('/api/stop',{method:'POST'});poll()}
async function sendHuman(){let el=document.getElementById('human');if(el.value.trim())await api('/api/human',{method:'POST',body:JSON.stringify({content:el.value})});el.value='';poll()}
function clearHuman(){document.getElementById('human').value=''}poll();
</script></body></html>"""


@dataclass
class Message:
    role: str
    label: str
    content: str
    turn: int
    timestamp: str


@dataclass
class LabState:
    messages: list[Message] = field(default_factory=list)
    running: bool = False
    round: int = 0
    max_rounds: int = 20
    prompt: str = ""
    pending_human: str = ""
    stop_requested: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    worker: threading.Thread | None = None


class ConversationLab:
    def __init__(self, model_a: str, model_b: str, backend_a: str, backend_b: str, output_dir: Path) -> None:
        self.backend_a = create_backend(backend_a, model_a)
        self.backend_b = create_backend(backend_b, model_b)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state = LabState()

    def snapshot(self) -> dict[str, Any]:
        with self.state.lock:
            return {"running": self.state.running, "round": self.state.round, "messages": [asdict(message) for message in self.state.messages]}

    def start(self, prompt: str, rounds: int) -> None:
        with self.state.lock:
            if self.state.running:
                return
            self.state.running = True
            self.state.round = 0
            self.state.max_rounds = max(1, min(rounds, 1000))
            self.state.prompt = prompt.strip()
            self.state.pending_human = ""
            self.state.stop_requested.clear()
            self.state.messages.clear()
            self._open_transcript()
        self.state.worker = threading.Thread(target=self._run, daemon=True)
        self.state.worker.start()

    def stop(self) -> None:
        self.state.stop_requested.set()

    def human(self, content: str) -> None:
        if content.strip():
            cleaned = content.strip()
            with self.state.lock:
                self.state.pending_human = cleaned
            self._record("human", "Tú", cleaned, self.state.round)

    def _run(self) -> None:
        current = self.state.prompt
        try:
            for turn in range(1, self.state.max_rounds + 1):
                if self.state.stop_requested.is_set():
                    break
                with self.state.lock:
                    if self.state.pending_human:
                        current = self.state.pending_human
                        self.state.pending_human = ""
                response_a = self._generate(self.backend_a, current)
                self._record("a", "Modelo A · GGUF/local", response_a, turn)
                if self.state.stop_requested.is_set():
                    break
                response_b = self._generate(self.backend_b, response_a)
                self._record("b", "Modelo B · local", response_b, turn)
                current = response_b
                with self.state.lock:
                    self.state.round = turn
        finally:
            with self.state.lock:
                self.state.running = False

    def _generate(self, backend: Any, text: str) -> str:
        return backend.generate(build_chat_prompt(text), GenerationConfig(max_new_tokens=512, temperature=0.75, top_p=0.95, device="cpu"))

    def _record(self, role: str, label: str, content: str, turn: int) -> None:
        message = Message(role, label, content, turn, datetime.now(timezone.utc).isoformat(timespec="seconds"))
        with self.state.lock:
            self.state.messages.append(message)
        record = asdict(message)
        with (self.output_dir / "conversation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        with (self.output_dir / "conversation.md").open("a", encoding="utf-8") as handle:
            handle.write(f"\n### {label} · turno {turn}\n\n{content}\n")

    def _open_transcript(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = self.output_dir / stamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "conversation.md").write_text("# Conversation Lab\n", encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    lab: ConversationLab

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._json(self.lab.snapshot())
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        payload = self._payload()
        path = urlparse(self.path).path
        if path == "/api/start":
            self.lab.start(str(payload.get("prompt", "")), int(payload.get("rounds", 20)))
            self._json({"ok": True})
        elif path == "/api/stop":
            self.lab.stop()
            self._json({"ok": True})
        elif path == "/api/human":
            self.lab.human(str(payload.get("content", "")))
            self._json({"ok": True})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def _json(self, value: Any) -> None:
        self._send(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json")

    def _send(self, content: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *_args: Any) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local visual two-model conversation lab")
    parser.add_argument("--model-a", required=True, help="GGUF/local model path")
    parser.add_argument("--model-b", required=True, help="Second local model path or Transformers id")
    parser.add_argument("--backend-a", default="llama-cpp", choices=["llama-cpp", "hf", "echo"])
    parser.add_argument("--backend-b", default="hf", choices=["llama-cpp", "hf", "echo"])
    parser.add_argument("--output-dir", default="conversations")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    lab = ConversationLab(args.model_a, args.model_b, args.backend_a, args.backend_b, Path(args.output_dir))
    handler = type("LabHandler", (Handler,), {"lab": lab})
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Conversation Lab: {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        lab.stop()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())