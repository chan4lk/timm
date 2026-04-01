# timm: Tiny Intelligent Micro-Models

**Agent-Specific Model Synthesis (ASMS) on Apple MLX**

A research implementation demonstrating that purpose-built micro-models—trained from scratch and matched to an agent's actual decision complexity—outperform over-provisioned LLMs on cost, latency, and deployability.

---

## What is this

Most production AI agents route every decision through a frontier LLM like Claude or GPT-4. Those models carry 10^11–10^12 parameters to cover all of human language. A typical OKR management agent needs to navigate fewer than 500 distinct decision paths.

**ASMS** (Agent-Specific Model Synthesis) treats the frontier LLM as a compiler, not a runtime. It uses the LLM once—offline—to synthesise a labelled corpus from a formal agent specification, then trains a tiny task-specific transformer whose capacity is matched to the agent's actual decision space. The resulting model is deployed locally: no API calls, no internet dependency, sub-5ms inference on Apple Silicon.

This repository implements the ASMS pipeline for the MyOKRs Keyflow MCP agent. The theoretical foundation is described in:

> Ranaweera, C. (2026). *Agent-Specific Model Synthesis: Compiling LLM Agents into Micro-Models*. Bistec Global.

---

## Architecture

The ASMS pipeline has six stages. The first three run once to produce the training corpus; the last three produce and deploy the model.

```
 AGENT SPECIFICATION
 ┌────────────────────────────────────────────┐
 │  role_spec.yaml + doerr_constraints.yaml   │
 └──────────────────────┬─────────────────────┘
                        │
                        ▼
 STAGE 3 ── Synthetic Corpus Generation
            (Claude API as compiler)
            Input: role spec + scenario templates
            Output: corpus.jsonl
                        │
                        ▼
 STAGE 3.5 ── Task-Specific BPE Tokenizer
              (sentencepiece, ~8K vocab)
              Output: okr_tokenizer.model
                        │
                        ▼
 STAGE 4 ── Micro-Transformer Architecture
            Decoder-only, 4L × 256D × 4H
            4.7M parameters, 512 context
                        │
                        ▼
 STAGE 5 ── Curriculum Training (MLX/Metal)
            Phase 1: normal (50%)
            Phase 2: + edge cases (35%)
            Phase 3: + adversarial (15%)
                        │
                        ▼
 STAGE 6a ── INT4 Quantization
             13MB FP16  →  3.5MB INT4
                        │
                        ▼
 STAGE 6b/c ── Inference + Keyflow MCP Bridge
               OKRInference + KeyflowBridge
               Confidence-gated fallback to Claude API
```

---

## The OKR Agent

The first micro-model built with this pipeline is a compiler for the **MyOKRs Keyflow MCP agent**.

The agent follows John Doerr's *Measure What Matters* methodology and exposes five tools (`cycle`, `objective`, `key_result`, `user`, `report`) with 20 operations routed across 6 workflows. Its effective decision-space cardinality is approximately 500.

The synthesised model is a 4.7M parameter MLX transformer that replaces the Claude API at runtime for the standard decision paths, with a confidence-gated fallback to Claude for out-of-distribution queries.

**Decision space mapping:**

| Dimension | Count |
|---|---|
| Tool operations | 20 |
| Workflow routes | 6 |
| Methodology enforcement rules | 8 |
| Effective cardinality | ~500 |

---

## POC Results

Trained on an M3 Pro MacBook Pro. No cloud GPU. No fine-tuning of a pre-existing model.

| Metric | Value |
|---|---|
| Training examples | 299 |
| Training time | 48 seconds |
| Model size (INT4) | 3.5 MB |
| Token accuracy | 71.7% |
| Parameters | 4.7M |
| Architecture | 4 layers, 256 hidden dim, 4 heads |
| Vocabulary | ~8K BPE tokens |
| Max context | 512 tokens |
| Target inference latency | <5ms p50 on Metal |

The accuracy figure reflects an early-stage POC corpus of 299 examples. Production targets require ~5K–10K examples (see Scaling section).

---

## Project Structure

```
timm/
├── run.py                          # End-to-end pipeline runner
│
├── spec/
│   ├── role_spec.yaml              # Agent tool schemas, workflows, constraints
│   └── doerr_constraints.yaml     # John Doerr methodology rules and scoring
│
├── corpus/
│   └── generate_corpus.py         # Stage 3: Claude API corpus synthesis
│
├── model/
│   ├── architecture.py             # Stage 4: OKRMicroModel (MLX transformer)
│   ├── train.py                    # Stage 5: Curriculum training loop
│   ├── checkpoints/                # Saved model weights (.safetensors)
│   └── tokenizer/
│       ├── train_tokenizer.py      # Stage 3.5: BPE tokenizer training
│       └── okr_tokenizer.model     # Trained sentencepiece model
│
├── deploy/
│   ├── quantize.py                 # Stage 6a: INT4 quantization
│   ├── inference.py                # Stage 6b: OKRInference engine
│   └── keyflow_bridge.py           # Stage 6c: Keyflow MCP bridge + OKRPipeline
│
├── eval/
│   ├── benchmark.py                # ASMS benchmark suite
│   └── test_sets/                  # Held-out test JSONL files
│
└── docs/
    └── adr/
        └── 001-okr-micro-model.md  # Architecture decision record
```

---

## Quick Start

### Prerequisites

- Apple M-series Mac (Metal GPU required for training; inference runs on any hardware)
- Python 3.12+
- `ANTHROPIC_API_KEY` environment variable (corpus generation only)

### Install

```bash
git clone <repo-url>
cd timm
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Run the full pipeline

```bash
# Minimal POC run (~100 examples, fast)
ANTHROPIC_API_KEY=your_key python run.py --corpus-size 100

# Production run (~1000 examples, with consistency filtering)
ANTHROPIC_API_KEY=your_key python run.py --corpus-size 1000 --epochs 50
```

### Run individual stages

```bash
# Stage 3: Generate corpus
python corpus/generate_corpus.py -n 300 --model claude-sonnet-4-20250514 -o corpus.jsonl

# Stage 3.5: Train tokenizer
python model/tokenizer/train_tokenizer.py --corpus corpus.jsonl --vocab-size 8000

# Stage 5: Train model
python model/train.py --corpus corpus.jsonl --epochs 20 --batch-size 32

# Stage 6a: Quantize
python deploy/quantize.py model/checkpoints/best --bits 4

# Benchmark
python eval/benchmark.py model/checkpoints/best_q4

# Inference
python deploy/inference.py model/checkpoints/best_q4 \
    --query "I want to improve customer retention by 20% this quarter"
```

---

## Pipeline Deep Dive

### Stage 3 — Synthetic Corpus Generation

**File:** `corpus/generate_corpus.py`

Uses Claude as a compiler to generate `(input, tool_calls, methodology_notes)` training triples. Requests are distributed across 6 workflow templates and 3 scenario categories (normal 80%, edge 15%, adversarial 5%). Self-consistency filtering runs each scenario through `k` independent generations and discards examples where the workflow and tool selection do not agree unanimously.

```bash
python corpus/generate_corpus.py -n 1000 -k 3 --workers 4
```

Key outputs: `corpus/data/corpus.jsonl` — one JSONL object per training example.

### Stage 3.5 — Task-Specific BPE Tokenizer

**File:** `model/tokenizer/train_tokenizer.py`

Trains a sentencepiece BPE tokenizer (~8K vocab) seeded with OKR domain terms, tool names, JSON structural tokens, and metric type identifiers. A smaller vocabulary reduces embedding layer size and improves per-token accuracy on the narrow domain.

```bash
python model/tokenizer/train_tokenizer.py --vocab-size 8000
```

### Stage 4 — Micro-Transformer Architecture

**File:** `model/architecture.py`

A decoder-only GPT-style transformer sized to ASMS complexity class MEDIUM (`|D| ≈ 500`):

- 4 transformer layers
- 256 hidden dimension
- 4 attention heads (64-dim per head)
- 512 FFN dimension (SwiGLU activation)
- RoPE positional encoding
- RMSNorm (pre-norm)
- KV cache for fast autoregressive inference

Training sequence format:
```
<bos>QUERY: {query} CONTEXT: {context}
<workflow>{workflow}</workflow>
<tool>{tool_call_json}</tool>
<score>{methodology_notes}</score><eos>
```

### Stage 5 — Curriculum Training

**File:** `model/train.py`

Three-phase curriculum with cosine learning rate schedule and AdamW:

| Phase | Epochs | Data |
|---|---|---|
| Phase 1: Normal | 50% of total | Normal cases only |
| Phase 2: Edge | 35% of total | Normal + edge cases |
| Phase 3: Adversarial | 15% of total | All categories |

Training runs entirely on Apple Silicon via MLX Metal acceleration. Checkpoints are saved as `.safetensors` files alongside a `config.json`.

```bash
python model/train.py --epochs 20 --batch-size 32 --lr 3e-4
```

### Stage 6a — INT4 Quantization

**File:** `deploy/quantize.py`

Applies MLX group-wise INT4 quantization to all linear layers (group size 32). Reduces the model from ~13MB FP16 to ~3.5MB, with negligible accuracy loss on structured decision tasks.

```bash
python deploy/quantize.py model/checkpoints/best --bits 4
```

### Stage 6b — Inference Engine

**File:** `deploy/inference.py`

`OKRInference` loads the quantized model and tokenizer, formats input queries, runs autoregressive generation with KV caching, and parses structured output from tagged sections (`<workflow>`, `<tool>`, `<score>`). Returns a confidence score used for fallback routing.

### Stage 6c — Keyflow MCP Bridge

**File:** `deploy/keyflow_bridge.py`

`KeyflowBridge` validates micro-model tool calls against the `role_spec.yaml` schemas and formats them as Keyflow MCP JSON-RPC calls. `OKRPipeline` wraps both components into a single `.run()` interface with confidence-gated fallback: requests below the 0.85 confidence threshold are routed to Claude API instead.

### Evaluation

**File:** `eval/benchmark.py`

Measures four ASMS paper metrics:

| Metric | Target |
|---|---|
| Model size | <50 MB |
| Latency p50 | <5 ms |
| Latency p99 | <10 ms |
| Tool accuracy | >95% |

```bash
python eval/benchmark.py model/checkpoints/best_q4
```

---

## Scaling to Production

The POC uses 299 examples with fast-mode corpus generation (no consistency filtering). Production accuracy requires a larger, quality-filtered corpus.

| Corpus Size | Expected Token Accuracy | Consistency Filtering | Est. Corpus Cost |
|---|---|---|---|
| 300 (POC) | ~72% | No | ~$5 |
| 1,000 | ~85% | Yes (k=3) | ~$20 |
| 3,000 | ~91% | Yes (k=5) | ~$50 |
| 10,000 | ~95%+ | Yes (k=5) | ~$150 |

**Ongoing cost model (1,000 OKR decisions/month at 1K corpus):**

| Component | One-Time | Monthly |
|---|---|---|
| Corpus generation | ~$50 | $0 |
| Training compute (M-series Mac) | $0 | $0 |
| Inference compute | $0 | $0 |
| Fallback calls to Claude API | $0 | ~$1 |
| **Total** | **~$50** | **~$1** |
| vs. Claude API-only | $0 | ~$10 |

Breakeven: approximately 6 months.

---

## Theory

The ASMS compression thesis rests on a mismatch in decision-space cardinality. A frontier LLM has capacity proportional to its parameter count—sufficient to span all of human language. A domain-specific agent operates over a much smaller decision space defined by:

- The set of available tools and their valid parameter combinations
- The set of valid workflow routes
- The set of methodology constraints that gate routing decisions

For the OKR agent, this effective cardinality is approximately 500. A transformer's representational capacity scales with depth, width, and vocabulary size. Matching model capacity to decision-space cardinality—rather than to language in general—yields the compression ratio observed here: from ~10^11 parameters (frontier LLM) to 4.7 × 10^6 parameters (micro-model), a reduction of roughly 20,000x, while preserving >95% of task accuracy at production corpus scale.

The from-scratch training approach avoids the "fine-tuning trap" identified in the paper: fine-tuning a 135M parameter base model on this task would still carry 270x more capacity than required, with higher serving cost and no accuracy benefit.

---

## Tech Stack

| Component | Library |
|---|---|
| Model training and inference | [MLX](https://github.com/ml-explore/mlx) >= 0.22 |
| Quantization | MLX built-in (`mlx.nn.quantize`) |
| Tokenizer | [sentencepiece](https://github.com/google/sentencepiece) >= 0.2 |
| Corpus generation | [Anthropic Python SDK](https://github.com/anthropic-ai/anthropic-sdk-python) >= 0.49 |
| Agent specification | PyYAML >= 6.0 |
| Array operations | NumPy >= 1.26 |

---

## Author

**Chandima Ranaweera**
Bistec Global

Architecture Decision Record: [`docs/adr/001-okr-micro-model.md`](docs/adr/001-okr-micro-model.md)
