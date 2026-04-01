# ASMS Resume Tasks

Current state: v2 model trained on 1,099 examples, 80.7% accuracy, 4/6 workflow routing.

## Next Steps (in order)

### 1. Generate corpus v3 (~3,000 total examples)
Generate ~2,000 more examples using Sonnet agents to reach ~3,000 total.
Target distribution per batch (6 parallel agents):
- goal_to_okr: 200 (split 2x100 to avoid token limits)
- check_in: 200
- view_okrs + reports: 200 (100 each, combined file)
- onboard + align: 200 (100 each, combined file)

After generation:
```bash
cat corpus/data/corpus_v2.jsonl corpus/data/batch3_*.jsonl > corpus/data/corpus_v3.jsonl
```

### 2. Retrain tokenizer and model on v3
```bash
uv run model/tokenizer/train_tokenizer.py --corpus corpus_v3.jsonl --vocab-size 6000
uv run model/train.py --corpus corpus_v3.jsonl --epochs 30 --batch-size 16
uv run deploy/quantize.py model/checkpoints/best
```
Expected: ~87% accuracy, 5-6/6 workflow routing, parseable JSON.

### 3. Evaluate and iterate
```bash
uv run python -c "
import sys, json; sys.path.insert(0, 'model')
from train import load_checkpoint, format_example
# ... run eval (see eval commands in previous session)
"
```
Test autoregressive generation quality. If JSON still malformed, generate more corpus.

### 4. Generate corpus v4 (~5,000-10,000 total)
Repeat step 1 with larger batches until hitting 95% accuracy target.

### 5. Benchmark and publish
```bash
uv run eval/benchmark.py model/checkpoints/best_q4
```
Run full benchmark suite. Compare against ASMS paper predictions.

### 6. Production deployment
- Start server: `uv run deploy/server.py model/checkpoints/best_q4`
- Connect MCP: click "Connect MCP" in UI at http://localhost:8800
- Toggle "Live MCP" for real Keyflow execution

## Key Commands
```bash
# Start server with chat UI
uv run deploy/server.py model/checkpoints/best_q4

# Train on latest corpus
uv run model/train.py --corpus corpus_vN.jsonl --epochs 30 --batch-size 16

# Quantize best checkpoint
uv run deploy/quantize.py model/checkpoints/best

# Test inference
uv run deploy/inference.py model/checkpoints/best --query "Show me my OKRs"

# Retrain tokenizer
uv run model/tokenizer/train_tokenizer.py --corpus corpus_vN.jsonl --vocab-size 6000
```

## Learnings (don't forget)
- Temperature MUST be 0.0 for generation (not 0.1)
- goal_to_okr agent hits 32K token limit at 200 lines — always split 2x100
- MCP uses direct HTTP + OAuth client_credentials (no mcp-remote)
- Val loss plateaus early — best checkpoint is around epoch 6-7, not final
- Keyflow tool actions: `check_in` (not `update`), `health_check` (not `health`), `generate_okrs` (not `onboard`), `align` (not `update` for parent)
