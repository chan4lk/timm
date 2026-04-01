"""
OpenAI-compatible API server for the ASMS OKR micro-model.

Exposes the micro-model as a local HTTP endpoint compatible with:
  - LM Studio (custom OpenAI endpoint)
  - Open WebUI
  - Any OpenAI SDK client
  - curl

Usage:
  python deploy/server.py model/checkpoints/best_q4
  python deploy/server.py model/checkpoints/best --port 8800

Then point any OpenAI-compatible client to http://localhost:8800/v1
"""

import argparse
import json
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from inference import OKRInference
from keyflow_bridge import KeyflowBridge

ENGINE: OKRInference | None = None
BRIDGE: KeyflowBridge | None = None
MODEL_NAME = "okr-micro-asms"
CHECKPOINT_DIR: Path | None = None
CURRENT_CHECKPOINT: str = ""
UI_PATH = Path(__file__).parent / "ui.html"


class OpenAIHandler(BaseHTTPRequestHandler):
    """Handles OpenAI-compatible chat/completions requests."""

    def do_GET(self):
        if self.path == "/" or self.path == "/ui":
            self._serve_ui()
        elif self.path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "asms-local",
                    "meta": {
                        "params": ENGINE.model.num_params if ENGINE else 0,
                        "quantized": True,
                        "framework": "mlx",
                    },
                }],
            })
        elif self.path == "/v1/asms/checkpoints":
            self._list_checkpoints()
        elif self.path == "/health":
            self._send_json(200, {"status": "ok", "model": MODEL_NAME})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        elif self.path == "/v1/completions":
            self._handle_completions()
        elif self.path == "/v1/asms/switch":
            self._switch_checkpoint()
        elif self.path == "/v1/asms/connect":
            self._connect_mcp()
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_chat_completions(self):
        body = self._read_body()
        if body is None:
            return

        messages = body.get("messages", [])
        temperature = body.get("temperature", 0.1)
        max_tokens = body.get("max_tokens", 256)
        stream = body.get("stream", False)

        # Extract the last user message as the query
        query = ""
        session_context = {}
        for msg in messages:
            if msg.get("role") == "user":
                query = msg.get("content", "")
            elif msg.get("role") == "system":
                # Try to extract session context from system message
                content = msg.get("content", "")
                if "session_context" in content:
                    try:
                        ctx_start = content.index("{")
                        ctx_end = content.rindex("}") + 1
                        session_context = json.loads(content[ctx_start:ctx_end])
                    except (ValueError, json.JSONDecodeError):
                        pass

        if not query:
            self._send_json(400, {"error": "no user message found"})
            return

        # Run inference
        result = ENGINE.predict(
            query=query,
            session_context=session_context,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        confidence = ENGINE.confidence_score(result)

        # Format response
        # Include both the structured output and a human-readable summary
        content_parts = []
        if result.get("workflow"):
            content_parts.append(f"**Workflow:** {result['workflow']}")
        if result.get("tool_calls"):
            content_parts.append(f"**Tool calls:**\n```json\n{json.dumps(result['tool_calls'], indent=2)}\n```")
        if result.get("methodology_notes"):
            notes = result["methodology_notes"]
            if notes:
                content_parts.append(f"**Methodology:** {json.dumps(notes)}")
        content_parts.append(f"\n_Confidence: {confidence:.2f} | Latency: {result['_inference']['latency_ms']}ms_")

        assistant_content = "\n\n".join(content_parts) if content_parts else result["_inference"]["raw_output"]

        # Execute tool calls against Keyflow MCP if live mode
        live_mode = body.get("asms_live", False)
        mcp_results = []
        validation_notes = []

        if BRIDGE and result.get("tool_calls"):
            for tc in result["tool_calls"]:
                if not tc.get("parse_error"):
                    valid, msg = BRIDGE.validate_tool_call(tc)
                    if not valid:
                        validation_notes.append(f"Invalid: {msg}")
                    elif live_mode:
                        mcp_call = BRIDGE.format_mcp_call(tc)
                        mcp_result = BRIDGE._execute_mcp(mcp_call)
                        mcp_results.append(mcp_result)

        if validation_notes:
            assistant_content += "\n\n**Validation warnings:**\n" + "\n".join(f"- {n}" for n in validation_notes)

        if mcp_results:
            assistant_content += "\n\n**MCP Results:**\n```json\n" + json.dumps(mcp_results, indent=2) + "\n```"

        # Fallback notice
        if confidence < 0.85:
            assistant_content += "\n\n_Low confidence — consider routing to Claude API for this query._"

        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if stream:
            self._stream_response(response_id, assistant_content)
        else:
            self._send_json(200, {
                "id": response_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": result["_inference"]["input_tokens"],
                    "completion_tokens": result["_inference"]["output_tokens"],
                    "total_tokens": result["_inference"]["input_tokens"] + result["_inference"]["output_tokens"],
                },
                "asms": {
                    "workflow": result.get("workflow"),
                    "tool_calls": result.get("tool_calls", []),
                    "methodology_notes": result.get("methodology_notes", {}),
                    "confidence": confidence,
                    "latency_ms": result["_inference"]["latency_ms"],
                    "live_mode": live_mode,
                    "mcp_results": mcp_results if mcp_results else None,
                },
            })

    def _handle_completions(self):
        body = self._read_body()
        if body is None:
            return

        prompt = body.get("prompt", "")
        temperature = body.get("temperature", 0.1)
        max_tokens = body.get("max_tokens", 256)

        result = ENGINE.predict(
            query=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        self._send_json(200, {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{
                "text": result["_inference"]["raw_output"],
                "index": 0,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": result["_inference"]["input_tokens"],
                "completion_tokens": result["_inference"]["output_tokens"],
                "total_tokens": result["_inference"]["input_tokens"] + result["_inference"]["output_tokens"],
            },
        })

    def _stream_response(self, response_id: str, content: str):
        """SSE streaming compatible with OpenAI stream format."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        # Send content in chunks
        chunk_size = 20
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }],
            }
            self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        # Send final chunk
        final = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _serve_ui(self):
        """Serve the chat UI."""
        if UI_PATH.exists():
            html = UI_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        else:
            self._send_json(404, {"error": "ui.html not found"})

    def _list_checkpoints(self):
        """List available model checkpoints."""
        checkpoints = []
        if CHECKPOINT_DIR and CHECKPOINT_DIR.exists():
            for cp in sorted(CHECKPOINT_DIR.iterdir()):
                config_file = cp / "config.json"
                if config_file.exists():
                    with open(config_file) as f:
                        cfg = json.load(f)
                    checkpoints.append({
                        "name": cp.name,
                        "path": str(cp),
                        "params": cfg.get("hidden_dim", 0),
                        "quantized": "quantization" in cfg,
                        "active": str(cp) == CURRENT_CHECKPOINT,
                    })
        self._send_json(200, {"checkpoints": checkpoints})

    def _switch_checkpoint(self):
        """Hot-swap the model to a different checkpoint."""
        global ENGINE, CURRENT_CHECKPOINT
        body = self._read_body()
        if body is None:
            return
        checkpoint = body.get("checkpoint", "")
        cp_path = Path(checkpoint)
        if not (cp_path / "config.json").exists():
            self._send_json(400, {"error": f"Invalid checkpoint: {checkpoint}"})
            return
        try:
            ENGINE = OKRInference(str(cp_path))
            CURRENT_CHECKPOINT = str(cp_path)
            self._send_json(200, {
                "status": "ok",
                "model": cp_path.name,
                "params": ENGINE.model.num_params,
                "checkpoint": str(cp_path),
            })
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _connect_mcp(self):
        """Connect to Keyflow MCP (may trigger OAuth in browser)."""
        if not BRIDGE:
            self._send_json(400, {"status": "error", "message": "Bridge not loaded"})
            return
        ok = BRIDGE.connect()
        if ok:
            self._send_json(200, {
                "status": "connected",
                "endpoint": BRIDGE.endpoint_url,
            })
        else:
            self._send_json(200, {
                "status": "auth_needed",
                "message": "OAuth authorization required — check the terminal for the auth URL",
                "endpoint": BRIDGE.endpoint_url,
            })

    def _read_body(self) -> dict | None:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(400, {"error": "empty request body"})
            return None
        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return None

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def log_message(self, format, *args):
        latency = ""
        if hasattr(self, "_start_time"):
            latency = f" ({(time.time() - self._start_time)*1000:.0f}ms)"
        print(f"[{self.log_date_time_string()}] {format % args}{latency}")


def main():
    global ENGINE, BRIDGE, CHECKPOINT_DIR, CURRENT_CHECKPOINT

    parser = argparse.ArgumentParser(description="ASMS OpenAI-compatible API server")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8800, help="Port")
    parser.add_argument("--tokenizer", help="Path to tokenizer model")
    args = parser.parse_args()

    print("=" * 50)
    print("  ASMS OKR Micro-Model Server")
    print("=" * 50)

    ENGINE = OKRInference(args.checkpoint, args.tokenizer)
    CURRENT_CHECKPOINT = str(Path(args.checkpoint).resolve())
    CHECKPOINT_DIR = Path(args.checkpoint).resolve().parent

    try:
        BRIDGE = KeyflowBridge()
        print("Keyflow bridge: loaded")
    except Exception:
        BRIDGE = None
        print("Keyflow bridge: unavailable (validation disabled)")

    server = HTTPServer((args.host, args.port), OpenAIHandler)
    print(f"\nServer running at http://localhost:{args.port}")
    print(f"  Chat UI:     http://localhost:{args.port}/")
    print(f"  API:         http://localhost:{args.port}/v1/chat/completions")
    print(f"  Models:      http://localhost:{args.port}/v1/models")
    print(f"  Checkpoints: http://localhost:{args.port}/v1/asms/checkpoints")
    print(f"  Health:      http://localhost:{args.port}/health")
    print(f"\nOpenAI-compatible endpoint:")
    print(f"  Base URL: http://localhost:{args.port}/v1")
    print(f"  Model:    {MODEL_NAME}")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
