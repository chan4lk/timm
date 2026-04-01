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

    def _execute_mcp(self, mcp_call: dict) -> dict:
        """Execute a single MCP call via the configured transport."""
        # In production, this would use the MCP client SDK.
        # For now, format as JSON-RPC for the Keyflow MCP endpoint.
        keyflow_config = self.mcp_config.get("mcpServers", {}).get("keyflow", {})
        if not keyflow_config:
            return {"status": "error", "message": "Keyflow MCP not configured"}

        return {
            "status": "would_execute",
            "mcp_call": mcp_call,
            "endpoint": keyflow_config.get("args", [None, ""])[1],
        }

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
