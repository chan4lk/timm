"""
ASMS Stage 3: Synthetic Corpus Generation

Uses Claude API as the "compiler LLM" to generate (input, tool_calls, reasoning)
training triples from the role specification. Implements:
  - Phase 3a: Scenario generation (normal/edge/adversarial)
  - Phase 3b: Decision labelling (tool selection + params)
  - Phase 3c: Self-consistency filtering (k=5 agreement)
  - Phase 3d: Corpus statistics and export
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from anthropic import Anthropic
from tqdm import tqdm

SPEC_DIR = Path(__file__).parent.parent / "spec"
DATA_DIR = Path(__file__).parent / "data"

SYSTEM_PROMPT = """You are a synthetic data generator for an OKR management agent.
Your job is to produce realistic (input, tool_calls, methodology_notes) training examples.

The agent manages OKRs using the Keyflow MCP API with these tools:
- cycle: create/get/list/update OKR cycles
- objective: create/get/list/update/delete objectives
- key_result: create/get/list/update/delete key results
- user: get/list/onboard users
- report: progress/health/summary reports

The agent follows John Doerr's "Measure What Matters" methodology:
- 3-5 objectives per cycle (focus)
- Key results must be measurable with metric types: NUMERIC, PERCENTAGE, BOOLEAN, MILESTONE
- Aspirational OKRs target 0.7 score (stretch)
- Committed OKRs must hit 1.0
- ~60% committed, ~40% aspirational
- Objectives are qualitative and inspirational, NOT numeric
- Key results measure outcomes, NOT tasks

IMPORTANT: Output valid JSON only. No markdown, no commentary."""

SCENARIO_TEMPLATES = {
    "goal_to_okr": {
        "weight": 0.30,
        "prompt": """Generate a realistic user query where someone wants to create OKRs from a business goal.
Include the session context (userId, activeCycleId).

Respond with this exact JSON structure:
{
  "input": {
    "query": "<natural language user input>",
    "session_context": {"userId": "<id>", "activeCycleId": "<id>", "activeCycleName": "<name>"}
  },
  "workflow": "goal_to_okr",
  "tool_calls": [
    {"tool": "objective", "action": "create", "params": {...}},
    {"tool": "key_result", "action": "create", "params": {...}}
  ],
  "methodology_notes": {
    "objective_is_qualitative": true/false,
    "krs_are_measurable": true/false,
    "okr_type": "committed|aspirational",
    "anti_patterns_detected": []
  }
}

Variation type: %VARIATION%""",
    },
    "check_in": {
        "weight": 0.20,
        "prompt": """Generate a realistic user query where someone updates progress on a key result.
The user reports a metric value and the agent updates the KR score.

Respond with this exact JSON structure:
{
  "input": {
    "query": "<natural language progress update>",
    "session_context": {"userId": "<id>", "activeCycleId": "<id>", "activeCycleName": "<name>"}
  },
  "workflow": "check_in",
  "tool_calls": [
    {"tool": "key_result", "action": "update", "params": {"keyResultId": "<id>", "currentValue": <num>, "score": <0.0-1.0>}}
  ],
  "methodology_notes": {
    "score_zone": "red|yellow|green",
    "sandbagging_risk": true/false,
    "coaching_note": "<brief methodology guidance>"
  }
}

Variation type: %VARIATION%""",
    },
    "view_okrs": {
        "weight": 0.15,
        "prompt": """Generate a realistic user query where someone wants to see their current OKRs.

Respond with this exact JSON structure:
{
  "input": {
    "query": "<natural language view request>",
    "session_context": {"userId": "<id>", "activeCycleId": "<id>", "activeCycleName": "<name>"}
  },
  "workflow": "view_okrs",
  "tool_calls": [
    {"tool": "objective", "action": "list", "params": {"cycleId": "<id>", "ownerId": "<id>"}},
    {"tool": "key_result", "action": "list", "params": {"objectiveId": "<id>"}}
  ],
  "methodology_notes": {}
}

Variation type: %VARIATION%""",
    },
    "reports": {
        "weight": 0.15,
        "prompt": """Generate a realistic user query requesting an OKR report or health check.

Respond with this exact JSON structure:
{
  "input": {
    "query": "<natural language report request>",
    "session_context": {"userId": "<id>", "activeCycleId": "<id>", "activeCycleName": "<name>"}
  },
  "workflow": "reports",
  "tool_calls": [
    {"tool": "report", "action": "progress|health|summary", "params": {"cycleId": "<id>", ...}}
  ],
  "methodology_notes": {
    "report_type": "progress|health|summary"
  }
}

Variation type: %VARIATION%""",
    },
    "onboard": {
        "weight": 0.10,
        "prompt": """Generate a realistic user query about onboarding a new team member with starter OKRs.

Respond with this exact JSON structure:
{
  "input": {
    "query": "<natural language onboarding request>",
    "session_context": {"userId": "<id>", "activeCycleId": "<id>", "activeCycleName": "<name>"}
  },
  "workflow": "onboard",
  "tool_calls": [
    {"tool": "user", "action": "onboard", "params": {"name": "<name>", "role": "<role>", "team": "<team>"}},
    {"tool": "objective", "action": "create", "params": {...}},
    {"tool": "key_result", "action": "create", "params": {...}}
  ],
  "methodology_notes": {
    "committed_ratio": 0.6,
    "ramp_appropriate": true/false
  }
}

Variation type: %VARIATION%""",
    },
    "align": {
        "weight": 0.10,
        "prompt": """Generate a realistic user query about aligning or cascading OKRs.

Respond with this exact JSON structure:
{
  "input": {
    "query": "<natural language alignment request>",
    "session_context": {"userId": "<id>", "activeCycleId": "<id>", "activeCycleName": "<name>"}
  },
  "workflow": "align",
  "tool_calls": [
    {"tool": "objective", "action": "list", "params": {"cycleId": "<id>", "level": "company|team"}},
    {"tool": "objective", "action": "update", "params": {"objectiveId": "<id>", "parentObjectiveId": "<id>"}}
  ],
  "methodology_notes": {
    "alignment_direction": "bottom_up|top_down|cross_team"
  }
}

Variation type: %VARIATION%""",
    },
}

VARIATIONS = {
    "normal": [
        "Standard business scenario with typical language",
        "Professional tone, clear intent, well-formed request",
        "Casual conversational tone, informal phrasing",
        "Brief, terse request with minimal context",
        "Detailed request with lots of business context",
        "Request from a manager about their team",
        "Request from an individual contributor about personal OKRs",
        "Request referencing a specific industry (SaaS, healthcare, fintech, retail, education)",
    ],
    "edge": [
        "Ambiguous request that could map to multiple workflows",
        "Request with missing session context (no active cycle)",
        "Request mixing multiple workflows in one message",
        "Request using non-standard OKR terminology",
        "Request with typos and informal abbreviations",
        "Request in a domain with unusual metrics (e.g. negative targets, ratios)",
        "Request for exactly 5 objectives (at the boundary)",
    ],
    "adversarial": [
        "User tries to set score to 1.0 on all aspirational OKRs (sandbagging)",
        "User tries to create more than 5 objectives in one cycle",
        "User provides a numeric objective instead of qualitative",
        "User writes tasks instead of outcomes as key results",
        "User tries to create OKR without measurable key results",
    ],
}


def load_spec():
    with open(SPEC_DIR / "role_spec.yaml") as f:
        return yaml.safe_load(f)


def pick_variation(category: str) -> str:
    return random.choice(VARIATIONS[category])


def pick_category(distribution: dict[str, float] | None = None) -> str:
    if distribution is None:
        distribution = {"normal": 0.80, "edge": 0.15, "adversarial": 0.05}
    r = random.random()
    cumulative = 0.0
    for cat, weight in distribution.items():
        cumulative += weight
        if r <= cumulative:
            return cat
    return "normal"


def pick_workflow() -> str:
    workflows = list(SCENARIO_TEMPLATES.keys())
    weights = [SCENARIO_TEMPLATES[w]["weight"] for w in workflows]
    return random.choices(workflows, weights=weights, k=1)[0]


def generate_single(client: Anthropic, model: str) -> dict | None:
    workflow = pick_workflow()
    category = pick_category()
    variation = pick_variation(category)

    template = SCENARIO_TEMPLATES[workflow]["prompt"]
    prompt = template.replace("%VARIATION%", f"{category} — {variation}")

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
        )
        text = response.content[0].text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[: text.rfind("```")]
        example = json.loads(text)
        example["_meta"] = {
            "workflow": workflow,
            "category": category,
            "variation": variation,
        }
        return example
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"  Parse error ({type(e).__name__}): skipping", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  API error ({type(e).__name__}): {e}", file=sys.stderr)
        return None


def consistency_check(client: Anthropic, model: str, example: dict, k: int = 5) -> bool:
    """Re-generate the decision k times and check agreement on workflow + tool selection."""
    query = example["input"]["query"]
    context = json.dumps(example["input"].get("session_context", {}))

    check_prompt = f"""Given this OKR agent input, determine the correct workflow and tool calls.

User query: "{query}"
Session context: {context}

Respond with JSON only:
{{"workflow": "<workflow_name>", "tools_used": ["<tool.action>", ...]}}"""

    decisions = []
    for _ in range(k):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": check_prompt}],
                temperature=0.3,
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[: text.rfind("```")]
            d = json.loads(text)
            decisions.append((d.get("workflow"), tuple(sorted(d.get("tools_used", [])))))
        except Exception:
            return False

    if len(decisions) < k:
        return False

    # All k decisions must agree on workflow and tool set
    return len(set(decisions)) == 1


def generate_corpus(
    total: int = 1000,
    model: str = "claude-sonnet-4-20250514",
    consistency_k: int = 3,
    workers: int = 4,
    output_file: str = "corpus.jsonl",
    skip_consistency: bool = False,
):
    client = Anthropic()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / output_file

    print(f"Generating {total} examples using {model}")
    print(f"Consistency check: k={consistency_k}, workers={workers}")
    print(f"Output: {output_path}")

    generated = []
    failed_parse = 0
    failed_consistency = 0

    # Phase 3a+3b: Generate and label
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(generate_single, client, model) for _ in range(total)]

        for future in tqdm(as_completed(futures), total=total, desc="Generating"):
            result = future.result()
            if result is None:
                failed_parse += 1
            else:
                generated.append(result)

    print(f"\nGenerated: {len(generated)} / {total} (parse failures: {failed_parse})")

    # Phase 3c: Self-consistency filtering
    if skip_consistency:
        consistent = generated
        print("Skipping consistency check")
    else:
        consistent = []
        print(f"\nRunning consistency check (k={consistency_k})...")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(consistency_check, client, model, ex, consistency_k): ex
                for ex in generated
            }

            for future in tqdm(as_completed(futures), total=len(generated), desc="Filtering"):
                ex = futures[future]
                if future.result():
                    consistent.append(ex)
                else:
                    failed_consistency += 1

        print(f"Consistent: {len(consistent)} / {len(generated)} (filtered: {failed_consistency})")

    # Phase 3d: Export and statistics
    with open(output_path, "w") as f:
        for ex in consistent:
            f.write(json.dumps(ex) + "\n")

    # Print distribution stats
    workflow_counts = Counter(ex["_meta"]["workflow"] for ex in consistent)
    category_counts = Counter(ex["_meta"]["category"] for ex in consistent)

    print(f"\n--- Corpus Statistics ---")
    print(f"Total examples: {len(consistent)}")
    print(f"\nWorkflow distribution:")
    for wf, count in workflow_counts.most_common():
        print(f"  {wf}: {count} ({count/len(consistent)*100:.1f}%)")
    print(f"\nCategory distribution:")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count} ({count/len(consistent)*100:.1f}%)")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASMS Stage 3: Synthetic Corpus Generation")
    parser.add_argument("-n", "--total", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model for generation")
    parser.add_argument("-k", "--consistency-k", type=int, default=3, help="Consistency check repetitions")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("-o", "--output", default="corpus.jsonl", help="Output filename")
    parser.add_argument("--skip-consistency", action="store_true", help="Skip consistency filtering")
    args = parser.parse_args()

    generate_corpus(
        total=args.total,
        model=args.model,
        consistency_k=args.consistency_k,
        workers=args.workers,
        output_file=args.output,
        skip_consistency=args.skip_consistency,
    )
