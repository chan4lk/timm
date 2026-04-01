"""
ASMS Stage 6c: Keyflow MCP Bridge

Maps micro-model output (structured tool calls) to actual Keyflow MCP API calls.
Includes:
  - Confidence-based fallback routing to Claude API
  - Tool call validation against Keyflow schemas
  - Session context management
"""

import json
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

import yaml

SPEC_PATH = Path(__file__).parent.parent / "spec" / "role_spec.yaml"
FALLBACK_CONFIDENCE_THRESHOLD = 0.85


class KeyflowBridge:
    """Bridge between micro-model output and Keyflow MCP tool calls."""

    def __init__(self, mcp_config_path: str | None = None):
        # Load tool schemas from role spec
        with open(SPEC_PATH) as f:
            spec = yaml.safe_load(f)
        self.tool_schemas = {t["name"]: t for t in spec["tools"]}

        # Load MCP config
        if mcp_config_path is None:
            mcp_config_path = str(Path(__file__).parent.parent / ".mcp.json")
        with open(mcp_config_path) as f:
            self.mcp_config = json.load(f)

        # Extract Keyflow endpoint URL and auth from mcp config
        keyflow_cfg = self.mcp_config.get("mcpServers", {}).get("keyflow", {})
        args = keyflow_cfg.get("args", [])
        self.endpoint_url = None
        self.auth_header = None
        for i, arg in enumerate(args):
            if arg.startswith("https://"):
                self.endpoint_url = arg
            if arg == "--header" and i + 1 < len(args):
                self.auth_header = args[i + 1]

        # MCP subprocess state
        self._process = None
        self._connected = False
        self._msg_id = 0

        if self.endpoint_url:
            print(f"Keyflow MCP endpoint: {self.endpoint_url}")
        else:
            print("Warning: Keyflow MCP endpoint not found in .mcp.json")

    def validate_tool_call(self, tool_call: dict) -> tuple[bool, str]:
        """Validate a tool call against the Keyflow schema."""
        tool_name = tool_call.get("tool")
        action = tool_call.get("action")
        params = tool_call.get("params", {})

        if tool_name not in self.tool_schemas:
            return False, f"Unknown tool: {tool_name}"

        schema = self.tool_schemas[tool_name]
        operations = {op["action"]: op for op in schema["operations"]}

        if action not in operations:
            return False, f"Unknown action: {tool_name}.{action}"

        op_schema = operations[action]
        op_params = op_schema.get("params", {})

        # Check required params
        for param_name, param_spec in op_params.items():
            if isinstance(param_spec, dict) and param_spec.get("required") and param_name not in params:
                return False, f"Missing required param: {param_name} for {tool_name}.{action}"

        return True, "valid"

    def format_mcp_call(self, tool_call: dict) -> dict:
        """Format a validated tool call for Keyflow MCP."""
        tool_name = tool_call["tool"]
        action = tool_call["action"]
        params = tool_call.get("params", {})

        return {
            "method": "tools/call",
            "params": {
                "name": f"mcp__keyflow__{tool_name}",
                "arguments": {
                    "action": action,
                    **params,
                },
            },
        }

    def execute(self, model_output: dict, dry_run: bool = True) -> list[dict]:
        """Execute tool calls from model output.

        Args:
            model_output: Parsed output from OKRInference.predict()
            dry_run: If True, validate and format but don't execute

        Returns:
            List of results (or formatted calls in dry_run mode)
        """
        results = []
        tool_calls = model_output.get("tool_calls", [])

        for tc in tool_calls:
            if tc.get("parse_error"):
                results.append({"status": "error", "message": "Unparseable tool call"})
                continue

            valid, msg = self.validate_tool_call(tc)
            if not valid:
                results.append({"status": "invalid", "message": msg, "tool_call": tc})
                continue

            mcp_call = self.format_mcp_call(tc)

            if dry_run:
                results.append({"status": "dry_run", "mcp_call": mcp_call})
            else:
                result = self._execute_mcp(mcp_call)
                results.append(result)

        return results

    def connect(self) -> bool:
        """Spawn mcp-remote subprocess and initialize MCP session.

        The Keyflow MCP server uses OAuth. On first connection, mcp-remote
        will print an auth URL to stderr. The user must visit that URL in
        a browser to authorize. After auth, the connection completes.
        """
        if self._process and self._process.poll() is None:
            return True  # already connected

        if not self.endpoint_url:
            print("No Keyflow MCP endpoint configured")
            return False

        cmd = ["npx", "mcp-remote", self.endpoint_url]
        if self.auth_header:
            cmd += ["--header", self.auth_header]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # mcp-remote needs time to discover OAuth and start the flow
            import os
            import time
            import webbrowser

            os.set_blocking(self._process.stderr.fileno(), False)
            os.set_blocking(self._process.stdout.fileno(), False)

            # Wait for mcp-remote to be ready (may need OAuth)
            auth_url = None
            start = time.time()
            while time.time() - start < 15:
                try:
                    line = self._process.stderr.readline()
                    if line:
                        line = line.strip()
                        if "authorize" in line.lower() and "http" in line:
                            # Extract URL from the line
                            for part in line.split():
                                if part.startswith("http"):
                                    auth_url = part
                                    break
                            if not auth_url and ":\n" not in line:
                                # URL might be on next line
                                next_line = self._process.stderr.readline().strip()
                                if next_line.startswith("http"):
                                    auth_url = next_line
                            if auth_url:
                                print(f"\n{'='*50}")
                                print("  Keyflow MCP requires OAuth authorization.")
                                print(f"  Opening browser to: {auth_url[:80]}...")
                                print(f"{'='*50}\n")
                                webbrowser.open(auth_url)
                        elif "connected" in line.lower() or "ready" in line.lower():
                            break
                except (IOError, OSError):
                    pass
                time.sleep(0.5)

            # Restore blocking mode for normal operation
            os.set_blocking(self._process.stdout.fileno(), True)

            # Wait a bit for OAuth callback if auth was needed
            if auth_url:
                print("Waiting for OAuth authorization (up to 60s)...")
                time.sleep(5)  # give user time to authorize

            # Send initialize request
            init_msg = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "asms-bridge", "version": "1.0"},
                },
            }
            response = self._send_rpc(init_msg, timeout=30)
            if response and "result" in response:
                # Send initialized notification
                self._write_msg({"jsonrpc": "2.0", "method": "notifications/initialized"})
                server_info = response["result"].get("serverInfo", {})
                print(f"MCP session established: {server_info}")
                self._connected = True
                return True
            else:
                print(f"MCP init failed: {response}")
                if auth_url:
                    print("Did you complete the OAuth authorization in the browser?")
                self.disconnect()
                return False
        except FileNotFoundError:
            print("npx not found — install Node.js to enable live MCP")
            return False
        except Exception as e:
            print(f"MCP connection failed: {e}")
            return False

    def disconnect(self):
        """Terminate MCP subprocess."""
        if self._process:
            self._process.terminate()
            self._process = None
            self._connected = False

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _write_msg(self, msg: dict):
        """Write a JSON-RPC message to the subprocess stdin."""
        if not self._process or not self._process.stdin:
            return
        line = json.dumps(msg)
        self._process.stdin.write(line + "\n")
        self._process.stdin.flush()

    def _read_msg(self) -> dict | None:
        """Read a JSON-RPC response from the subprocess stdout."""
        if not self._process or not self._process.stdout:
            return None
        try:
            line = self._process.stdout.readline()
            if line:
                return json.loads(line.strip())
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _send_rpc(self, msg: dict, timeout: float = 10.0) -> dict | None:
        """Send a JSON-RPC request and wait for matching response."""
        import select
        import time

        self._write_msg(msg)
        msg_id = msg.get("id")
        deadline = time.time() + timeout

        while time.time() < deadline:
            if not self._process or not self._process.stdout:
                return None
            # Check if data available
            ready, _, _ = select.select([self._process.stdout], [], [], 0.5)
            if ready:
                response = self._read_msg()
                if response and response.get("id") == msg_id:
                    return response
                # Skip notifications
        return None

    def _execute_mcp(self, mcp_call: dict) -> dict:
        """Execute a single MCP call via the mcp-remote subprocess."""
        if not self._connected:
            if not self.connect():
                return {"status": "error", "message": "MCP not connected"}

        rpc_msg = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": mcp_call["method"],
            "params": mcp_call["params"],
        }

        try:
            response = self._send_rpc(rpc_msg)
            if response is None:
                # Connection may have died, try reconnecting once
                self.disconnect()
                if not self.connect():
                    return {"status": "error", "message": "MCP reconnection failed"}
                response = self._send_rpc(rpc_msg)

            if response is None:
                return {"status": "error", "message": "MCP timeout", "mcp_call": mcp_call}

            if "error" in response:
                return {
                    "status": "mcp_error",
                    "error": response["error"],
                    "mcp_call": mcp_call,
                }
            return {
                "status": "success",
                "result": response.get("result"),
                "mcp_call": mcp_call,
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "mcp_call": mcp_call}

    def should_fallback(self, model_output: dict, confidence: float) -> bool:
        """Determine if this request should fall back to Claude API."""
        if confidence < FALLBACK_CONFIDENCE_THRESHOLD:
            return True

        # Also fallback if tool calls couldn't be validated
        for tc in model_output.get("tool_calls", []):
            if tc.get("parse_error"):
                return True
            valid, _ = self.validate_tool_call(tc)
            if not valid:
                return True

        return False


class OKRPipeline:
    """End-to-end pipeline: query -> micro-model -> Keyflow MCP."""

    def __init__(self, checkpoint_path: str, mcp_config_path: str | None = None):
        from inference import OKRInference

        self.engine = OKRInference(checkpoint_path)
        self.bridge = KeyflowBridge(mcp_config_path)

    def run(
        self,
        query: str,
        session_context: dict | None = None,
        dry_run: bool = True,
    ) -> dict:
        """Full pipeline: query -> predict -> validate -> execute."""
        # 1. Model inference
        prediction = self.engine.predict(query, session_context)
        confidence = self.engine.confidence_score(prediction)

        # 2. Fallback check
        if self.bridge.should_fallback(prediction, confidence):
            return {
                "status": "fallback",
                "reason": f"Low confidence ({confidence:.2f})",
                "prediction": prediction,
            }

        # 3. Validate and execute
        results = self.bridge.execute(prediction, dry_run=dry_run)

        return {
            "status": "success",
            "workflow": prediction.get("workflow"),
            "tool_results": results,
            "methodology_notes": prediction.get("methodology_notes", {}),
            "inference": prediction.get("_inference", {}),
            "confidence": confidence,
        }


if __name__ == "__main__":
    # Demo: validate bridge against role spec
    bridge = KeyflowBridge()
    print("Keyflow Bridge — Tool Schema Validation")
    print("=" * 50)

    test_calls = [
        {"tool": "objective", "action": "create", "params": {"title": "Delight customers", "cycleId": "cyc_1", "ownerId": "usr_1"}},
        {"tool": "key_result", "action": "update", "params": {"keyResultId": "kr_1", "currentValue": 75, "score": 0.7}},
        {"tool": "report", "action": "health", "params": {"cycleId": "cyc_1"}},
        {"tool": "unknown_tool", "action": "do_thing", "params": {}},  # should fail
        {"tool": "objective", "action": "create", "params": {"description": "missing title"}},  # should fail
    ]

    for tc in test_calls:
        valid, msg = bridge.validate_tool_call(tc)
        status = "PASS" if valid else "FAIL"
        print(f"  [{status}] {tc['tool']}.{tc['action']}: {msg}")
