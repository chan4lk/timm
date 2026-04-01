---
name: asms
description: Compile any tool-use agent specification into a CPU-deployable MLX micro-model using Agent-Specific Model Synthesis. Use when building task-specific small models from agent role specs.
---

# Agent-Specific Model Synthesis (ASMS) — MLX Pipeline

## When to Use

Use ASMS when ALL of the following are true:

- You have a tool-use agent with a **bounded, enumerable decision space** (typically fewer than 10,000 effective decision paths)
- The agent's decisions follow **deterministic or near-deterministic logic** (workflow routing, parameter extraction, constraint enforcement)
- You need **sub-10ms inference latency** or **zero marginal cost per call**
- You want inference to run **locally on Apple Silicon** with no cloud dependency
- The agent's tool schemas are stable enough to justify a one-time training investment

Do NOT use ASMS for:
- Open-ended reasoning, creative writing, or tasks requiring world knowledge
- Agents whose decision space expands unpredictably with user input
- Prototypes where tool schemas change weekly (re-synthesis cost is real)
- Agents with effective cardinality > 50,000 (use fine-tuning instead)

**Reference example throughout this skill:** The OKR agent (`spec/role_spec.yaml`) — 5 tools, 20 operations, 6 workflows, ~500 effective decision paths, compiled to a ~6.7M parameter micro-transformer.

---

## Overview

ASMS treats **the LLM as a compiler, not a runtime**.

Instead of routing every user query to Claude API at $0.01/call with 500ms latency, ASMS uses a large LLM once (offline) to synthesise a task-specific micro-model that captures only the decision logic required for a particular agent. The result is a model sized to the problem — not a 10^12 parameter general-purpose model handling a 10^3 decision space.

**The pipeline in one sentence:** Role spec → synthetic corpus (Claude API, once) → tokenizer → micro-transformer training (MLX, local) → INT4 quantization → inference + MCP bridge.

**Key insight:** A 6.7M parameter model trained on the exact decision space of an OKR agent will outperform a 135M parameter LoRA fine-tune on the same task, because it has no capacity wasted on irrelevant knowledge. This is the "fine-tuning trap" described in ADR-001 (`docs/adr/001-okr-micro-model.md`).

**Cost model for OKR agent:**

```
Component              One-Time    Monthly
───────────────────────────────────────────
Corpus generation      $50         $0
Training compute       $0          $0  (Apple Silicon, local)
Inference compute      $0          $0
Fallback (Claude API)  $0          ~$1
───────────────────────────────────────────
TOTAL                  $50         ~$1
vs. Claude API-only:   $0          ~$10
Breakeven:             ~6 months
```

---

## Pipeline Stages

```
Stage 1  →  Role Specification        spec/role_spec.yaml
Stage 2  →  Decision Space Analysis   (manual calculation)
Stage 3  →  Synthetic Corpus          corpus/generate_corpus.py
Stage 3.5→  Tokenizer Training        model/tokenizer/train_tokenizer.py
Stage 4  →  Architecture Selection    model/architecture.py
Stage 5  →  Curriculum Training       model/train.py
Stage 6a →  INT4 Quantization         deploy/quantize.py
Stage 6b →  Inference Wrapper         deploy/inference.py
Stage 6c →  MCP Bridge                deploy/keyflow_bridge.py
```

All stages can be run end-to-end via `run.py`.

---

## Stage 1: Role Specification

Create `spec/role_spec.yaml` describing your agent completely. This file drives both corpus generation and bridge validation — it is the single source of truth.

**Required sections:**

```yaml
agent_name: <your_agent_name>
version: "1.0"
description: >
  <What this agent does, what methodology it follows, what MCP it uses.>

tools:
  - name: <tool_name>
    description: <what the tool does>
    operations:
      - action: <create|get|list|update|delete|...>
        params:
          <param_name>: { type: <string|number|enum|date|boolean>, required: true/false, description: "..." }
          # For enums, add: values: [opt1, opt2, opt3]
          # For numbers, add: min: 0.0, max: 1.0

workflows:
  - name: <workflow_name>
    description: <what triggers this workflow>
    trigger_phrases:
      - "<example natural language phrase>"
    tool_sequence: [<tool.action>, <tool.action>]
    methodology_checks:           # optional: domain rules to enforce
      - <check_name>

constraints:
  methodology:
    - rule: <rule_id>
      value: <threshold_or_limit>
      description: "<human-readable explanation>"
  anti_patterns:
    - name: <pattern_name>
      description: "<what this looks like>""
      detection: "<detection logic in pseudocode>"
      action: "<what the agent should do>"

input_schema:
  type: object
  properties:
    query: { type: string, description: "Natural language user input" }
    session_context:
      type: object
      properties:
        userId: { type: string }
        # ... agent-specific context fields

decision_space_analysis:
  tool_operations: <count>
  workflow_routes: <count>
  methodology_decisions: <count>
  effective_cardinality: <product_or_estimate>
  complexity_class: <TRIVIAL|LOW|MEDIUM|HIGH>
```

**OKR agent example** (`spec/role_spec.yaml`):
- 5 tools: `cycle`, `objective`, `key_result`, `user`, `report`
- 6 workflows: `goal_to_okr`, `view_okrs`, `check_in`, `onboard`, `reports`, `align`
- 20 total operations across all tools
- Constraints defined separately in `spec/doerr_constraints.yaml` and referenced inline

**Tips for any agent:**
- Be exhaustive with `trigger_phrases` — they directly seed corpus generation
- List every `anti_pattern` your agent should catch; they become adversarial training examples
- The `input_schema.session_context` block defines what runtime state flows into every prediction

---

## Stage 2: Decision Space Analysis

Calculate **effective cardinality** — the size of your agent's decision space.

**Formula:**

```
|D| = |tool_operations| × |workflow_routes| × |methodology_decisions|
```

Where:
- `tool_operations` = total number of (tool, action) pairs across all tools
- `workflow_routes` = number of distinct execution paths through your workflow graph
- `methodology_decisions` = branching factor from domain constraints (e.g. score zones, type ratios, anti-pattern flags)

**OKR agent example:**
```
tool_operations:     20   (cycle×4 + objective×5 + key_result×5 + user×3 + report×3)
workflow_routes:      6   (goal_to_okr, view_okrs, check_in, onboard, reports, align)
methodology_decisions: 8  (committed/aspirational, score zones, anti-pattern flags)
effective_cardinality: ~500  (not raw product — many combos are invalid)
complexity_class: MEDIUM
```

**Complexity classification and architecture selection:**

| Complexity Class | |D| Range    | Recommended Architecture       | Corpus Size |
|-----------------|-------------|-------------------------------|-------------|
| TRIVIAL         | < 50        | 2L / 128H / 2-head            | 200–500     |
| LOW             | 50–500      | 2L / 256H / 4-head            | 500–2K      |
| MEDIUM          | 500–5K      | 4L / 256H / 4-head (~6.7M p)  | 2K–10K      |
| HIGH            | 5K–50K      | 6L / 512H / 8-head (~50M p)   | 10K–50K     |

If `|D| > 50K`, reconsider ASMS — this is the fine-tuning boundary. Consider LoRA on SmolLM-360M instead.

**Record the analysis** in the `decision_space_analysis` block of your `role_spec.yaml`.

---

## Stage 3: Synthetic Corpus Generation

The large LLM (Claude Sonnet) acts as the compiler: given your role spec, it generates (input, tool_calls, reasoning) training triples. You run this once. The corpus captures the full decision surface of your agent.

### 3a. Adapt `corpus/generate_corpus.py`

Two things to customise for a new agent:

**1. Update `SYSTEM_PROMPT`** — describe your agent's domain, tools, and methodology constraints. The OKR prompt explains Doerr methodology and tool schemas in ~20 lines. Be specific: the LLM needs enough context to generate valid tool call parameters.

**2. Update `SCENARIO_TEMPLATES`** — one entry per workflow, with:
- `weight`: fraction of corpus (must sum to 1.0)
- `prompt`: JSON template with `%VARIATION%` placeholder

Each template must define the exact JSON schema for that workflow's training examples. Mirror your `role_spec.yaml` structure:

```python
SCENARIO_TEMPLATES = {
    "<workflow_name>": {
        "weight": 0.25,
        "prompt": """Generate a realistic user query for <workflow description>.

Respond with this exact JSON structure:
{
  "input": {
    "query": "<natural language user input>",
    "session_context": {<your session context fields>}
  },
  "workflow": "<workflow_name>",
  "tool_calls": [
    {"tool": "<tool_name>", "action": "<action>", "params": {...}}
  ],
  "methodology_notes": {
    <domain-specific quality flags>
  }
}

Variation type: %VARIATION%""",
    },
    # ... one entry per workflow
}
```

**3. Update `VARIATIONS`** — define realistic variation types for each category:

```python
VARIATIONS = {
    "normal":      ["<8 realistic natural language variation descriptions>"],
    "edge":        ["<7 boundary and ambiguous cases>"],
    "adversarial": ["<5 cases that should trigger constraint violations>"],
}
```

The OKR adversarial examples target sandbagging, too-many-OKRs, numeric objectives, tasks-as-KRs, and missing metrics. Define equivalents for your domain.

### 3b. Generate the corpus

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# Fast mode — skip consistency filtering (good for first run)
python corpus/generate_corpus.py \
  -n 100 \
  --model claude-sonnet-4-20250514 \
  --skip-consistency \
  -o corpus.jsonl

# Production mode — with k=3 self-consistency filtering
python corpus/generate_corpus.py \
  -n 2000 \
  --model claude-sonnet-4-20250514 \
  -k 3 \
  -w 8 \
  -o corpus.jsonl
```

**Self-consistency filtering** (`-k 3`): Each example is re-generated k times with temperature=0.3. Only examples where all k runs agree on workflow and tool selection are kept. This filters ~20% of examples but significantly improves training quality. Skip for initial experiments; enable for production corpus.

**Output format** — each line of `corpus/data/corpus.jsonl` is a JSON object:
```json
{
  "input": {"query": "...", "session_context": {...}},
  "workflow": "<workflow_name>",
  "tool_calls": [{"tool": "...", "action": "...", "params": {...}}],
  "methodology_notes": {...},
  "_meta": {"workflow": "...", "category": "normal|edge|adversarial", "variation": "..."}
}
```

### 3c. Train the tokenizer

The tokenizer is trained on your specific corpus vocabulary — not a general-purpose tokenizer.

```bash
python model/tokenizer/train_tokenizer.py \
  --corpus corpus.jsonl \
  --vocab-size 1500      # scales with corpus; 1500 works for MEDIUM complexity
```

This produces `model/tokenizer/okr_tokenizer.model`. The `train.py` script reads vocab size from this file dynamically via `sp.GetPieceSize()`.

**Vocab size guidance:**
- TRIVIAL: 500–800
- LOW: 800–1500
- MEDIUM: 1500–4000
- HIGH: 4000–8000

**Rule of thumb from measured data:**
```
vocab_size = min(8000, unique_tokens_in_corpus * 2)
```
With 299 examples the effective unique token count was ~750, making vocab 1500 the ceiling. At 1099 examples, vocab 4000 works well. If you have no corpus yet, train on a seed vocabulary from your role spec and tool schemas first, then grow vocab as corpus expands.

### Recommended corpus sizes and cost estimates

| Agent Complexity | Min Corpus | Production Corpus | Est. Cost (Sonnet) |
|-----------------|------------|-------------------|-------------------|
| TRIVIAL         | 200        | 500               | ~$1               |
| LOW             | 500        | 2,000             | ~$5               |
| MEDIUM (OKR)    | 1,000      | 5,000             | ~$20–50           |
| HIGH            | 5,000      | 20,000            | ~$100–200         |

Costs assume ~2K output tokens/example at Sonnet pricing. Run with `--skip-consistency` initially to validate format before spending on full corpus.

---

## Stage 4: Architecture Selection

### Modify `model/architecture.py`

The `OKRModelConfig` dataclass defines the architecture. For a new agent, rename it or create a new dataclass — the field names are what matter:

```python
@dataclass
class <YourAgent>ModelConfig:
    vocab_size: int = <from_tokenizer>    # set dynamically in train.py
    max_seq_len: int = 512                # 512 handles most agent I/O
    num_layers: int = <see_table>
    hidden_dim: int = <see_table>
    num_heads: int = <see_table>
    ffn_dim: int = <hidden_dim * 2>       # SwiGLU: 2x hidden is standard
    dropout: float = 0.1
    rope_theta: float = 10000.0
```

**Architecture selection table by complexity class:**

| Class   | num_layers | hidden_dim | num_heads | ffn_dim | ~Params | FP16 Size | INT4 Size |
|---------|-----------|-----------|----------|---------|---------|----------|---------|
| TRIVIAL | 2         | 128       | 2        | 256     | ~1.5M   | ~3 MB    | ~0.8 MB |
| LOW     | 2         | 256       | 4        | 512     | ~4M     | ~8 MB    | ~2 MB   |
| MEDIUM  | 4         | 256       | 4        | 512     | ~6.7M   | ~13 MB   | ~3.4 MB |
| HIGH    | 6         | 512       | 8        | 1024    | ~50M    | ~100 MB  | ~25 MB  |

The OKR agent uses the **MEDIUM** config: 4 layers, 256 hidden, 4 heads, 512 FFN.

**Architecture components** (do not change for new agents — these are well-chosen for structured output):
- **RoPE positional encoding** — efficient for short 512-token contexts; no learned positional embeddings needed
- **SwiGLU FFN** — slightly more expressive than ReLU at the same parameter count; `FeedForward` class uses gate + up + down projections
- **Pre-norm with RMSNorm** — more stable training than post-norm for small models
- **Causal masking** — decoder-only, autoregressive generation of structured JSON tokens
- **KV cache** — enabled in `Attention.__call__` for fast inference

**max_seq_len:** 512 tokens handles inputs like `QUERY: <text> CONTEXT: <json> <workflow>...</workflow> <tool>...</tool> <score>...</score>`. If your agent's tool calls are verbose (deeply nested JSON), increase to 1024. If purely key-lookup style, 256 is sufficient.

---

## Stage 5: Training

### Run training

```bash
# Quick validation run (5 epochs, small batch)
uv run model/train.py \
  --corpus corpus.jsonl \
  --epochs 5 \
  --batch-size 16

# Full production run (confirmed working config)
uv run model/train.py \
  --corpus corpus.jsonl \
  --epochs 30 \
  --batch-size 16 \
  --lr 3e-4 \
  --warmup 100 \
  --eval-every 50 \
  --checkpoint-every 500
```

Checkpoints are saved to `model/checkpoints/`. The best validation loss checkpoint is saved as `model/checkpoints/best/`. Always use the best checkpoint — val loss typically plateaus early then slowly rises, so the final epoch is not the best model.

### Curriculum learning phases

Training uses three phases defined in `train.py` via the `curriculum` list. Phase proportions are fixed; adjust `epochs` to control total training time:

| Phase | Epochs % | Data Categories       | Purpose                                    |
|-------|----------|-----------------------|--------------------------------------------|
| 1     | 50%      | normal only           | Learn primary decision boundaries          |
| 2     | 35%      | normal + edge         | Refine boundary behaviour                  |
| 3     | 15%      | normal + edge + adv.  | Harden constraint adherence                |

This ordering matters: the model first learns the happy path, then generalises to ambiguous cases, then hardens against adversarial inputs. Reversing the order degrades final performance.

**Category distribution in corpus** (controlled in `generate_corpus.py`):
```python
{"normal": 0.80, "edge": 0.15, "adversarial": 0.05}
```

### Hyperparameter guidance

**Confirmed working config (OKR agent, 1099 examples, MEDIUM architecture):**
- `batch_size=16`, `epochs=30`, `lr=3e-4`, `warmup=100` steps
- Curriculum phases: 50% / 35% / 15% (normal / +edge / +adversarial)
- Save best by `val_loss`, not final epoch

| Parameter     | Default  | When to Change                                      |
|--------------|----------|-----------------------------------------------------|
| `lr_max`     | 3e-4     | Lower to 1e-4 if training is unstable early         |
| `lr_min`     | 1e-5     | Leave as-is; cosine decay handles the tail          |
| `warmup`     | 100      | Increase to 200 if batch size > 64                  |
| `batch_size` | 16       | 16 confirmed; increase to 32 with larger corpus     |
| `epochs`     | 30       | Val loss plateaus ~epoch 6-7 of Phase 1, then slowly rises — best model is mid-training, not final |
| `dropout`    | 0.1      | Increase to 0.2 if overfitting (train loss << val loss) |

### Expected loss curves

For a MEDIUM complexity corpus of ~1,100 examples (confirmed behaviour):

```
Phase 1 (Normal, epochs 1–15 of 30):
  Epoch 1:  val ~3.5    ← starting point with ~1K corpus
  Epoch 6:  val ~2.0    ← plateau begins; best checkpoint often here
  Epoch 10: val ~2.1    ← slow creep up (mild overfitting starts)
  Epoch 15: val ~2.3

Phase 2 (+ Edge):
  Brief val loss spike as edge cases introduced, then partial recovery

Phase 3 (+ Adversarial):
  Another brief spike, then stabilises
```

**Key behaviour:** Val loss plateaus around epoch 6-7 of Phase 1, then slowly increases due to mild overfitting on a small corpus. The best checkpoint by `val_loss` is saved mid-training — always use it, not the final epoch.

**Target final validation loss:** < 1.5 is functional; < 1.0 is production quality; < 0.5 for high-stakes applications. With 1099 examples, reaching < 1.5 is achievable; < 1.0 requires ~5K+ examples.

The training sequence format is: `<bos>QUERY: {query} CONTEXT: {context} <workflow>{wf}</workflow> <tool>{tool_calls_json}</tool> <score>{methodology_json}</score><eos>`

---

## Stage 6: Deployment

### 6a. Quantize to INT4

```bash
# Quantize the best checkpoint to INT4
python deploy/quantize.py model/checkpoints/best

# Result saved to: model/checkpoints/best_q4/
# OKR example: 13MB FP16 → 3.4MB INT4

# Optional: INT8 for higher accuracy
python deploy/quantize.py model/checkpoints/best --bits 8
```

The quantizer uses MLX `nn.quantize()` which quantizes all `nn.Linear` layers with `group_size=32`. Configuration is saved in `config.json` alongside the weights so the inference wrapper can automatically re-apply quantization on load.

**Accuracy impact:** Structured decision tasks (tool selection, JSON generation) lose < 1% accuracy with INT4. The decision space is discrete — quantization noise rarely crosses a decision boundary.

### 6b. Inference wrapper

`deploy/inference.py` provides the `OKRInference` class. For a new agent, you only need to update `format_input()` to match your training sequence format:

```python
def format_input(self, query: str, session_context: dict | None = None) -> str:
    """Must match the format used in train.py:format_example()"""
    context = json.dumps(session_context or {})
    return f"QUERY: {query} CONTEXT: {context} "
```

**Run inference directly:**

```bash
python deploy/inference.py model/checkpoints/best_q4 \
  --query "I want to improve customer retention"
```

**Use as a library:**

```python
from deploy.inference import OKRInference

engine = OKRInference("model/checkpoints/best_q4")
result = engine.predict(
    query="Show me our Q2 OKRs",
    session_context={"userId": "usr_1", "activeCycleId": "cyc_q2"},
    temperature=0.0,   # MUST be 0.0 — see Troubleshooting: space-token degeneration
)

print(result["workflow"])       # "view_okrs"
print(result["tool_calls"])     # [{"tool": "objective", "action": "list", ...}]
print(result["_inference"]["latency_ms"])  # <5ms on M-series

# Confidence score for fallback routing
confidence = engine.confidence_score(result)  # 0.0 – 1.0
```

**Output format:**

```json
{
  "workflow": "<workflow_name>",
  "tool_calls": [
    {"tool": "<name>", "action": "<action>", "params": {...}}
  ],
  "methodology_notes": {...},
  "_inference": {
    "latency_ms": 3.2,
    "input_tokens": 28,
    "output_tokens": 64,
    "raw_output": "..."
  }
}
```

### 6c. MCP bridge setup

`deploy/keyflow_bridge.py` maps micro-model output to actual MCP calls and handles fallback routing. For a new agent, update the `format_mcp_call()` method to match your MCP tool naming convention:

```python
def format_mcp_call(self, tool_call: dict) -> dict:
    tool_name = tool_call["tool"]
    action = tool_call["action"]
    params = tool_call.get("params", {})

    return {
        "method": "tools/call",
        "params": {
            "name": f"mcp__<your_mcp_server>__{tool_name}",  # update prefix
            "arguments": {"action": action, **params},
        },
    }
```

**MCP connection pattern (Streamable HTTP transport with OAuth)**

The bridge connects via direct HTTP — no `mcp-remote` proxy or Node.js process needed. The protocol is:

1. **Discover OAuth metadata** — `GET <mcp_base_url>/.well-known/oauth-authorization-server`
2. **Register client** — `POST <registration_endpoint>` with `{"client_name": "...", "grant_types": ["client_credentials"]}`
3. **Get token** — `POST <token_endpoint>` with `client_id`, `client_secret`, `grant_type=client_credentials`
4. **Init MCP session** — `POST <mcp_base_url>/mcp` with `{"jsonrpc":"2.0","method":"initialize",...}` and `Authorization: Bearer <token>`; capture the `mcp-session-id` from the response header
5. **Call tools** — `POST <mcp_base_url>/mcp` with `{"jsonrpc":"2.0","method":"tools/call","params":{"name":"...","arguments":{...}}}` and both `Authorization` and `mcp-session-id` headers

This replaces the `.mcp.json` `npx` config for server-side deployment. The bridge in `deploy/keyflow_bridge.py` implements this flow via `connect_mcp()` and `call_tool()`.

**Configure `.mcp.json`** at the repo root (for local Claude Code integration):

```json
{
  "mcpServers": {
    "<your_mcp_name>": {
      "command": "npx",
      "args": ["-y", "@your-org/mcp-server", "https://your-api-endpoint"]
    }
  }
}
```

**Fallback threshold** — in `keyflow_bridge.py`:
```python
FALLBACK_CONFIDENCE_THRESHOLD = 0.85
```
Requests below 0.85 confidence are routed to Claude API. Tune this based on your accuracy requirements. The OKR agent sees ~15% fallback rate on novel phrasing, keeping API costs at ~$1/month.

**End-to-end pipeline class** (`OKRPipeline`):

```python
from deploy.keyflow_bridge import OKRPipeline

pipeline = OKRPipeline(
    checkpoint_path="model/checkpoints/best_q4",
    mcp_config_path=".mcp.json",
)

result = pipeline.run(
    query="Create OKRs for improving customer retention",
    session_context={"userId": "usr_1", "activeCycleId": "cyc_q2"},
    dry_run=True,   # set False to actually execute MCP calls
)

print(result["status"])       # "success" or "fallback"
print(result["confidence"])   # 0.0 – 1.0
print(result["tool_results"]) # validated + formatted MCP calls
```

### 6d. Chat UI + API server

`deploy/server.py` provides:
- **OpenAI-compatible API** at `/v1/chat/completions` — drop-in for any OpenAI SDK client
- **Chat UI** served at `/` (root) — browser-based interface for testing
- **Model switching** via `POST /v1/asms/switch` with `{"checkpoint": "<path>"}` — switch between checkpoints without restarting
- **MCP connection** via `POST /v1/asms/connect` with `{"mcp_url": "<url>"}` — connects the bridge at runtime

```bash
# Start the server (default port 8000)
uv run deploy/server.py

# Open chat UI
open http://localhost:8000

# Switch model at runtime
curl -X POST http://localhost:8000/v1/asms/switch \
  -H "Content-Type: application/json" \
  -d '{"checkpoint": "model/checkpoints/best_q4"}'

# Connect to MCP server
curl -X POST http://localhost:8000/v1/asms/connect \
  -H "Content-Type: application/json" \
  -d '{"mcp_url": "https://your-mcp-endpoint"}'
```

---

## Quick Start

Complete pipeline from zero to deployed model:

```bash
# Prerequisites — all commands run via uv
export ANTHROPIC_API_KEY=sk-ant-...

# 0. Confirm you're in the timm repo root
cd /path/to/timm

# 1. (If adapting for a new agent) Edit spec/role_spec.yaml and
#    corpus/generate_corpus.py for your agent's domain

# 2. Run the full pipeline (OKR agent, 100-example fast mode)
uv run run.py \
  --corpus-size 100 \
  --model claude-sonnet-4-20250514 \
  --epochs 30 \
  --batch-size 16

# 3. Test inference (temperature=0.0 is mandatory)
uv run deploy/inference.py model/checkpoints/best_q4 \
  --query "Show me my Q2 OKRs"

# 4. Start the chat UI + OpenAI-compatible API server
uv run deploy/server.py

# 5. Test bridge validation
uv run deploy/keyflow_bridge.py
```

**All scripts accept `uv run <script>` directly** — no manual pip install or venv activation needed.

**Selective stage execution** (useful during development):

```bash
# Regenerate corpus only
uv run run.py --skip-tokenizer --skip-training --skip-quantize --skip-benchmark

# Retrain from existing corpus
uv run run.py --skip-corpus --skip-tokenizer

# Requantize existing checkpoint
uv run deploy/quantize.py model/checkpoints/best --bits 4

# Benchmark only
uv run run.py --skip-corpus --skip-tokenizer --skip-training --skip-quantize
```

**Expected runtimes on Apple M-series:**

| Stage            | Time (100 examples) | Time (5K examples) |
|-----------------|--------------------|--------------------|
| Corpus gen       | ~3 min             | ~45 min            |
| Tokenizer train  | <1 min             | <1 min             |
| Model training   | ~5 min             | ~60 min            |
| Quantization     | <1 min             | <1 min             |
| Total            | ~10 min            | ~2 hours           |

---

## Scaling Guide

Corpus size is the primary lever for accuracy improvement. Architecture size is secondary — do not upscale architecture before maxing out corpus quality.

**Corpus size vs. accuracy (MEDIUM complexity, OKR agent) — rows marked * are measured, rest are estimates:**

| Corpus Size | Val Loss    | Accuracy (teacher-forced) | Workflow Routing | Fallback Rate |
|------------|-------------|--------------------------|-----------------|---------------|
| 299        | 3.46 *      | 71.7% *                  | 0/6 *           | ~50%          |
| 1,099      | 2.05 *      | 80.7% *                  | 4/6 *           | ~25%          |
| 5,000      | ~0.8 (est)  | ~90% (est)               | 6/6             | ~10%          |
| 10,000     | ~0.55 (est) | ~95% (est)               | 6/6             | ~5%           |

Key observations from measured data:
- Structured output (valid JSON tool calls) only emerges reliably around 1,000+ examples
- Workflow routing jumps from 0/6 to 4/6 between 300 and 1,100 examples — the model needs enough diversity to learn all routes
- Val loss of 2.05 at 1,099 examples is functional but not production-quality; target < 1.0 for reliable routing

**Scaling cost estimates (Claude Sonnet corpus generation):**

| Corpus Size | Est. API Cost | Recommended For         |
|------------|---------------|-------------------------|
| 100        | ~$0.50        | Development / validation |
| 500        | ~$2.50        | Low-traffic agents       |
| 2,000      | ~$10          | Standard production      |
| 5,000      | ~$25          | High-accuracy production |
| 10,000     | ~$50          | Mission-critical agents  |

**When to scale architecture vs. corpus:**

- If `val_loss > 1.0` after 5K examples → scale architecture up one class
- If `val_loss < 0.5` on 2K examples → corpus quality may be more impactful than size; add more adversarial variations
- If fallback rate is high on specific workflows → add more examples for those workflows (adjust `weight` in `SCENARIO_TEMPLATES`)

---

## Troubleshooting

### High rate of `<unk>` tokens in generated output

**Cause:** Vocabulary too small for your domain's terminology.

**Fix:**
```bash
# Check for UNK tokens in encoded corpus
python -c "
import sentencepiece as spm, json
sp = spm.SentencePieceProcessor()
sp.Load('model/tokenizer/okr_tokenizer.model')
unk_count = 0
with open('corpus/data/corpus.jsonl') as f:
    for line in f:
        ex = json.loads(line)
        ids = sp.Encode(ex['input']['query'])
        unk_count += ids.count(sp.unk_id())
print(f'UNK tokens in queries: {unk_count}')
"

# Retrain tokenizer with larger vocab
python model/tokenizer/train_tokenizer.py \
  --corpus corpus.jsonl \
  --vocab-size 3000      # double your current size
```

Then retrain from scratch (tokenizer change invalidates all saved weights).

### Validation loss does not decrease after Phase 1

**Cause:** Corpus too small, or curriculum categories are empty due to generation failures.

**Check:**
```bash
# Inspect category distribution
python -c "
import json
from collections import Counter
cats = Counter()
with open('corpus/data/corpus.jsonl') as f:
    for line in f:
        cats[json.loads(line)['_meta']['category']] += 1
print(cats)
"
```

**Fix:** If `edge` or `adversarial` counts are near zero, regenerate with a higher `--total` and without `--skip-consistency`. At minimum you need ~50 examples per category for Phase 2/3 to have any effect.

### Validation loss much higher than training loss (overfitting)

**Symptoms:** `train_loss < 0.4`, `val_loss > 1.0` by epoch 10.

**Fixes in order of preference:**
1. Increase corpus size (most effective)
2. Increase `dropout` from 0.1 to 0.2 in `OKRModelConfig`
3. Increase `weight_decay` in AdamW from 0.01 to 0.1
4. Reduce model size by one complexity class

### Generated tool calls fail validation in `keyflow_bridge.py`

**Cause:** Model is generating tool/action names or param structures that don't match `role_spec.yaml`.

**Diagnostic:**
```bash
# Run bridge validation script
python deploy/keyflow_bridge.py  # prints PASS/FAIL for test cases

# Check what the model is actually generating
python deploy/inference.py model/checkpoints/best \
  --query "Show me my OKRs"
# Inspect _inference.raw_output for the literal generated text
```

**Fix:** The model learned a schema that drifted from the spec. Re-check that `SYSTEM_PROMPT` and `SCENARIO_TEMPLATES` in `generate_corpus.py` exactly match `role_spec.yaml` tool names and action names. Regenerate corpus after fixing.

### Inference latency > 20ms

**Cause:** Running unquantized FP16 weights, or not using Metal acceleration.

**Fix:**
```bash
# Verify Metal is available
python -c "import mlx.core as mx; print(mx.metal.is_available())"

# Always use quantized checkpoint for production
python deploy/inference.py model/checkpoints/best_q4 --query "test"

# If Metal unavailable (non-Apple hardware), expect 50-200ms instead of <5ms
```

### Fallback rate unexpectedly high (> 30%) after good training

**Cause:** The `confidence_score()` function in `inference.py` is penalising valid outputs, or the `FALLBACK_CONFIDENCE_THRESHOLD` in `keyflow_bridge.py` is set too high.

**Fix:**
```python
# Temporarily lower threshold to diagnose
FALLBACK_CONFIDENCE_THRESHOLD = 0.70  # was 0.85

# Check what confidence scores look like on your test set
engine = OKRInference("model/checkpoints/best_q4")
for query in test_queries:
    result = engine.predict(query, ...)
    print(f"{engine.confidence_score(result):.2f}  {result['workflow']}  {query[:50]}")
```

If most scores cluster around 0.7–0.8, the model is generating valid output but the scoring function is too conservative. Adjust the penalty weights in `confidence_score()` or lower the threshold.

### Space-token degeneration (output is repeated spaces / UNK)

**Cause:** Using any temperature above 0.0 (even 0.1) causes the model to enter a degenerate loop where it samples a space token, which gets decoded as UNK by SentencePiece, which feeds back as a noisy input token for the next step, resulting in an infinite stream of spaces and UNK tokens.

**This is the single most common failure mode when first running inference.**

**Fix:** Always use `temperature=0.0` (greedy decoding). This is mandatory for structured JSON output — the model's decision space is discrete, so greedy is both correct and optimal.

```python
# WRONG — causes degeneration
result = engine.predict(query, temperature=0.1)

# CORRECT
result = engine.predict(query, temperature=0.0)
```

```bash
# WRONG
uv run deploy/inference.py model/checkpoints/best_q4 --query "..." --temperature 0.1

# CORRECT (or omit --temperature; default should be 0.0)
uv run deploy/inference.py model/checkpoints/best_q4 --query "..." --temperature 0.0
```

If you see this in other inference frameworks, ensure `do_sample=False` / `greedy=True` is set explicitly.

### Re-synthesis trigger: when to retrain

Retrain the model when:
- A new tool or action is added to `role_spec.yaml`
- An existing action's required params change
- A new anti-pattern or methodology rule is added
- Fallback rate creeps above 20% (distribution shift in user queries)

Retrain is NOT needed for: changes to session context values, new users/cycles, UI changes that don't affect query phrasing patterns.

---

## Lessons Learned

Empirical findings from the OKR agent implementation that differ from initial estimates:

### Inference

- **Temperature MUST be 0.0.** Even 0.1 causes space-token degeneration (repeated spaces decoded as UNK). This is not negotiable for structured output. See Troubleshooting section for details.
- The micro-model's decision space is discrete — greedy decoding is both safe and optimal.

### Corpus & Accuracy Scaling

Measured data points (OKR agent, MEDIUM architecture):

| Corpus | Val Loss | Teacher-Forced Accuracy | Workflow Routing |
|--------|----------|------------------------|-----------------|
| 299    | 3.46     | 71.7%                  | 0/6             |
| 1,099  | 2.05     | 80.7%                  | 4/6             |

- **Structured output does not emerge until ~1,000 examples.** Below that, the model may predict correct workflow labels but produce malformed JSON tool calls.
- **Workflow routing coverage is non-linear.** 0/6 at 299 examples → 4/6 at 1,099. The remaining 2/6 routes were underrepresented in the corpus (low `weight` in `SCENARIO_TEMPLATES`). Fix by increasing weight for those workflows.
- The original scaling estimates in this skill were optimistic. Treat estimates as aspirational; measure at 300, 1K, and 3K before committing to a production corpus size.

### Tokenizer Vocab Sizing

- The formula `vocab_size = min(8000, unique_tokens_in_corpus * 2)` was validated in practice.
- With a 299-example corpus, vocab > ~1500 had no unique tokens to fill; setting it higher wastes model capacity on padding.
- With 1099 examples, vocab 4000 is appropriate.
- Mismatch between corpus size and vocab size is a hidden failure mode — the tokenizer trains fine but the embedding table is mostly unused, inflating parameter count without benefit.

### Training Dynamics

- Val loss plateaus around epoch 6-7 of Phase 1, then slowly increases — classic mild overfitting on a small corpus.
- The best checkpoint is saved mid-training, not at the final epoch. Always load from `model/checkpoints/best/`.
- `batch_size=16` (not 32) worked better with the 1,099-example corpus — smaller batches provide more gradient updates per epoch, which helps when data is scarce.
- `epochs=30` gave the scheduler enough time to visit the best loss region, but `epochs=20` with a 300-example corpus was insufficient.

### MCP Connectivity

- The Streamable HTTP transport with OAuth `client_credentials` grant works reliably without any Node.js proxy.
- The session ID from the `mcp-session-id` response header must be preserved and sent with every subsequent tool call — stateless retry without it will fail.
- Token expiry (typically 1 hour) requires the bridge to re-authenticate transparently; implement a token refresh check before each call.

### Deployment

- `uv run` is the most reliable way to invoke all scripts — it handles the virtualenv and dependency resolution automatically.
- The server's model-switching endpoint (`/v1/asms/switch`) is valuable during development: you can compare checkpoint quality without restarting the server or the chat UI session.
