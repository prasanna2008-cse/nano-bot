"""Tiny web chat server (standard library only) for a trained checkpoint.

Usage:
    python app.py            # http://localhost:8000
    python app.py --port 8080
"""

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from model import GPT
from prompts import STOP_MARKERS, build_prompt, clean_reply
from tokenizer import CharTokenizer

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
HOLDBACK = 8  # characters kept back so half a stop marker is never streamed

STATE = {"model": None, "tokenizer": None, "device": "cpu", "lock": threading.Lock()}


def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}\nRun `python train.py` first."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = GPT.from_checkpoint(checkpoint_path, device)
    STATE["model"] = model
    STATE["tokenizer"] = CharTokenizer(checkpoint["chars"])
    STATE["device"] = device
    return checkpoint


class Handler(BaseHTTPRequestHandler):
    server_version = "NanoGPT/1.0"

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")

    def _send_file(self, filename, content_type):
        path = os.path.join(WEB_DIR, filename)
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_file("index.html", "text/html; charset=utf-8")
        elif self.path == "/style.css":
            self._send_file("style.css", "text/css; charset=utf-8")
        elif self.path == "/app.js":
            self._send_file("app.js", "application/javascript; charset=utf-8")
        elif self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self.send_error(404, "Not found")

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            message = str(payload.get("message", "")).strip()[:1000]
            temperature = float(payload.get("temperature", 0.3))
            top_k = int(payload.get("top_k", 40))
            history = payload.get("history") or []
        except (ValueError, TypeError):
            self._send_json({"error": "invalid request"}, status=400)
            return

        if not message:
            self._send_json({"error": "message is required"}, status=400)
            return

        temperature = min(max(temperature, 0.1), 1.5)
        top_k = min(max(top_k, 1), 200)
        turns = []
        if isinstance(history, list):
            for item in history[-8:]:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    turns.append((str(item[0])[:500], str(item[1])[:500]))

        model = STATE["model"]
        tokenizer = STATE["tokenizer"]
        device = STATE["device"]
        prompt = build_prompt(turns, message, model.config.block_size, tokenizer).to(device)

        # No Content-Length: the reply is streamed and the connection close
        # marks the end of the body (HTTP/1.0 style).
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        text = ""
        sent = 0
        try:
            with STATE["lock"], torch.no_grad():
                for token in model.generate(
                    prompt,
                    max_new_tokens=220,
                    temperature=temperature,
                    top_k=top_k,
                ):
                    text += tokenizer.decode([token.item()])
                    if any(marker in text for marker in STOP_MARKERS):
                        break
                    safe = len(text) - HOLDBACK
                    if safe > sent:
                        self.wfile.write(text[sent:safe].encode("utf-8"))
                        self.wfile.flush()
                        sent = safe

            # The trimmed reply starts `lead` characters into the raw text, so
            # shift the cursor before sending what is left.
            lead = len(text) - len(text.lstrip())
            reply = clean_reply(text)
            rest_from = max(0, sent - lead)
            if len(reply) > rest_from:
                self.wfile.write(reply[rest_from:].encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser tab closed mid-reply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/nano_gpt.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    checkpoint = load_model(args.checkpoint)
    print(f"loaded checkpoint (step {checkpoint.get('step')}, val loss {checkpoint.get('val_loss'):.4f})")
    print(f"model ready on {STATE['device']}. Open http://{args.host}:{args.port}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
