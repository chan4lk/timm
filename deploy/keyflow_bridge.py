"""
ASMS Stage 6c: Keyflow MCP Bridge

Maps micro-model output (structured tool calls) to actual Keyflow MCP API calls.
Includes:
  - Confidence-based fallback routing to Claude API
  - Tool call validation against Keyflow schemas
  - Session context management
"""

import json
import urllib.error
import urllib.parse
import urllib.request
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

        # MCP session state
        self._connected = False
        self._msg_id = 0
        self._session_id = None
        self._access_token = None
        self._client_id = None
        self._client_secret = None

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
        """Connect to Keyflow MCP via direct HTTP (Streamable HTTP transport).

        Uses OAuth client_credentials grant to get a fresh access token,
        then initializes an MCP session over HTTP+SSE.
        """
        if self._connected and self._session_id:
            return True

        if not self.endpoint_url:
            print("No Keyflow MCP endpoint configured")
            return False

        try:
            # Step 1: Discover OAuth endpoints
            discovery_url = self.endpoint_url.rsplit("/api/mcp", 1)[0]
            req = urllib.request.Request(
                f"{discovery_url}/.well-known/oauth-authorization-server"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                oauth_config = json.loads(resp.read().decode())

            token_endpoint = oauth_config["token_endpoint"]
            registration_endpoint = oauth_config.get("registration_endpoint")

            # Step 2: Register a client if we don't have credentials
            if not self._client_id:
                reg_body = json.dumps({
                    "client_name": "ASMS Bridge",
                    "grant_types": ["client_credentials"],
                    "scope": "mcp:read mcp:write",
                    "token_endpoint_auth_method": "client_secret_post",
                }).encode()
                req = urllib.request.Request(
                    registration_endpoint, data=reg_body,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    client_info = json.loads(resp.read().decode())
                self._client_id = client_info["client_id"]
                self._client_secret = client_info["client_secret"]
                print(f"Registered OAuth client: {self._client_id}")

            # Step 3: Get access token via client_credentials
            token_body = urllib.parse.urlencode({
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "scope": "mcp:read mcp:write",
            }).encode()
            req = urllib.request.Request(
                token_endpoint, data=token_body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                token_data = json.loads(resp.read().decode())
            self._access_token = token_data["access_token"]
            print(f"OAuth token acquired (expires in {token_data.get('expires_in', '?')}s)")

            # Step 4: Initialize MCP session
            init_body = json.dumps({
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "asms-bridge", "version": "1.0"},
                },
            }).encode()
            req = urllib.request.Request(
                self.endpoint_url, data=init_body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._access_token}",
                    "Accept": "application/json, text/event-stream",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                # Extract session ID from headers
                self._session_id = resp.headers.get("mcp-session-id")
                body = resp.read().decode()
                # Parse SSE response
                for line in body.split("\n"):
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        server_info = data.get("result", {}).get("serverInfo", {})
                        print(f"MCP connected: {server_info.get('name')} v{server_info.get('version')}")
                        break

            # Step 5: Send initialized notification
            notif_body = json.dumps({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }).encode()
            try:
                req = urllib.request.Request(
                    self.endpoint_url, data=notif_body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._access_token}",
                        "Mcp-Session-Id": self._session_id,
                        "Accept": "application/json",
                    },
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=5)
            except urllib.error.HTTPError:
                pass  # Some servers don't accept notification POSTs

            self._connected = True
            return True

        except Exception as e:
            print(f"MCP connection failed: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Clear MCP session."""
        self._connected = False
        self._session_id = None
        self._access_token = None

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _execute_mcp(self, mcp_call: dict) -> dict:
        """Execute a single MCP tool call via HTTP."""
        if not self._connected:
            if not self.connect():
                return {"status": "error", "message": "MCP not connected"}

        rpc_body = json.dumps({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": mcp_call["method"],
            "params": mcp_call["params"],
        }).encode()

        try:
            req = urllib.request.Request(
                self.endpoint_url, data=rpc_body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._access_token}",
                    "Mcp-Session-Id": self._session_id,
                    "Accept": "application/json, text/event-stream",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read().decode()
                # Parse SSE or plain JSON response
                for line in body.split("\n"):
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "error" in data:
                            return {"status": "mcp_error", "error": data["error"], "mcp_call": mcp_call}
                        return {"status": "success", "result": data.get("result"), "mcp_call": mcp_call}
                # Fallback: try plain JSON
                try:
                    data = json.loads(body)
                    if "error" in data:
                        return {"status": "mcp_error", "error": data["error"], "mcp_call": mcp_call}
                    return {"status": "success", "result": data.get("result"), "mcp_call": mcp_call}
                except json.JSONDecodeError:
                    return {"status": "error", "message": f"Unparseable response: {body[:200]}", "mcp_call": mcp_call}

        except urllib.error.HTTPError as e:
            if e.code == 401:
                # Token expired, reconnect
                self.disconnect()
                if self.connect():
                    return self._execute_mcp(mcp_call)
                return {"status": "error", "message": "Token refresh failed", "mcp_call": mcp_call}
            body = e.read().decode() if e.fp else ""
            return {"status": "http_error", "code": e.code, "message": body[:500], "mcp_call": mcp_call}
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
