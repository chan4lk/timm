# ADR-001: OKR Micro-Model via ASMS on Apple MLX

## Status
Accepted

## Date
2026-04-01

## Context
The MyOKRs agent currently relies on Claude API for every OKR management decision — translating goals to OKRs, check-ins, reports, alignment — routed through Keyflow MCP. Each interaction costs ~$0.01 in API calls with 500ms+ latency. The agent's decision space is bounded: 5 tools, 20 operations, 6 workflows, ~500 effective decision paths. Per the ASMS paper (Ranaweera, 2026), this is the canonical over-provisioning scenario where a 10^12 parameter LLM handles a 10^3 decision space.

## Decision
Apply Agent-Specific Model Synthesis (ASMS) to compile the OKR agent's decision logic into a ~6.7M parameter micro-transformer, trained from scratch on Apple MLX, deployable on any M-series Mac with <5ms inference latency.

## Bistec Alignment
- **Default Stack:** Deviates from .NET/Node.js backend — this is a Python/MLX ML pipeline, justified by the research nature of the work and MLX's Apple Silicon exclusivity.
- **Cloud Platform:** No cloud needed. Training and inference run locally on M-series Mac. Production deployment could target Azure Container Apps with CPU-only containers ($0/GPU).
- **Cost Impact:** One-time $60 corpus generation, then $0/month for inference. Replaces ~$120/year in Claude API calls at 1000 OKR decisions/month.

## Alternatives Considered

### Option A: Fine-tune SmolLM-135M on MLX
- Pros: Pre-trained language understanding, MLX-LM has LoRA tooling
- Cons: 135M params for a 500-decision-space = 270x over-provisioned (the "fine-tuning trap" per ASMS §1.2)
- Monthly Cost: $0 (local)

### Option B: Continue with Claude API
- Pros: Zero implementation effort, handles OOD gracefully
- Cons: $0.01/decision, 500ms+ latency, requires internet, data leaves device
- Monthly Cost: ~$10 at current usage

### Option C: ASMS from-scratch micro-model (selected)
- Pros: Model capacity matched to decision complexity, <5ms latency, $0 inference, data stays local
- Cons: Requires corpus generation effort, needs re-synthesis when tools change
- Monthly Cost: $0

## Consequences
- **Positive:** Sub-5ms inference, zero marginal cost, data privacy, validates ASMS paper methodology
- **Negative:** Must re-generate corpus and retrain when Keyflow tools/schemas change
- **Risks:** Model may underperform on OOD queries → mitigated by confidence-based fallback to Claude API

## Cost Analysis

```
COST ESTIMATE — OKR Micro-Model
═══════════════════════════════════════════
Component              One-Time    Monthly
───────────────────────────────────────────
Corpus generation      $50         $0
Training compute       $0          $0
Inference compute      $0          $0
Fallback (Claude API)  $0          ~$1
───────────────────────────────────────────
TOTAL                  $50         ~$1
vs. Claude API-only:   $0          ~$10
───────────────────────────────────────────
Breakeven: ~6 months
```
