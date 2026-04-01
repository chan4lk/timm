"""
ASMS Stage 3.5: Task-Specific BPE Tokenizer

Trains a sentencepiece BPE tokenizer on the generated corpus with a reduced
vocabulary (~8K) covering OKR terminology, tool names, and parameter values.
"""

import argparse
import json
import tempfile
from pathlib import Path

import sentencepiece as spm

CORPUS_DIR = Path(__file__).parent.parent.parent / "corpus" / "data"
TOKENIZER_DIR = Path(__file__).parent

# Seed vocabulary: OKR domain terms, tool names, JSON structural tokens
SEED_VOCAB = [
    # Tool names and actions
    "cycle", "objective", "key_result", "user", "report",
    "create", "get", "list", "update", "delete", "onboard",
    "progress", "health", "summary",
    # Workflows
    "goal_to_okr", "check_in", "view_okrs", "reports", "align",
    # Methodology terms
    "committed", "aspirational", "qualitative", "measurable",
    "sandbagging", "stretch", "moonshot", "focus",
    # Metric types
    "NUMERIC", "PERCENTAGE", "BOOLEAN", "MILESTONE",
    # OKR fields
    "cycleId", "objectiveId", "keyResultId", "ownerId", "userId",
    "title", "description", "metricType", "startValue", "targetValue",
    "currentValue", "score", "parentObjectiveId", "level",
    # Levels
    "company", "department", "team", "individual",
    # JSON structure
    "tool_calls", "methodology_notes", "session_context", "workflow",
    "input", "query", "params", "action", "tool",
    # Common business terms
    "revenue", "customer", "retention", "churn", "NPS", "pipeline",
    "onboarding", "conversion", "engagement", "satisfaction",
    "quarter", "sprint", "roadmap", "milestone", "deadline",
]


def extract_text_from_corpus(corpus_path: Path) -> list[str]:
    """Extract all text content from JSONL corpus for tokenizer training."""
    texts = []
    with open(corpus_path) as f:
        for line in f:
            ex = json.loads(line)
            # Extract the query
            if "input" in ex and "query" in ex["input"]:
                texts.append(ex["input"]["query"])
            # Extract tool call params as text
            for tc in ex.get("tool_calls", []):
                texts.append(json.dumps(tc))
            # Extract methodology notes
            if "methodology_notes" in ex:
                texts.append(json.dumps(ex["methodology_notes"]))
    return texts


def train_tokenizer(
    corpus_file: str = "corpus.jsonl",
    vocab_size: int = 8000,
    model_prefix: str = "okr_tokenizer",
):
    corpus_path = CORPUS_DIR / corpus_file

    if not corpus_path.exists():
        print(f"Corpus not found at {corpus_path}")
        print("Generating seed text from vocabulary for initial tokenizer...")
        texts = generate_seed_texts()
    else:
        texts = extract_text_from_corpus(corpus_path)
        print(f"Extracted {len(texts)} text segments from corpus")

    # Write texts to temp file for sentencepiece
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for text in texts:
            f.write(text + "\n")
        # Add seed vocab as extra training lines
        for word in SEED_VOCAB:
            f.write(word + "\n")
        tmp_path = f.name

    output_prefix = str(TOKENIZER_DIR / model_prefix)

    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        user_defined_symbols=[
            "<tool>", "</tool>", "<params>", "</params>",
            "<workflow>", "</workflow>", "<score>", "</score>",
        ],
        num_threads=4,
        train_extremely_large_corpus=False,
    )

    print(f"Tokenizer saved to {output_prefix}.model ({vocab_size} tokens)")

    # Quick test
    sp = spm.SentencePieceProcessor(model_file=f"{output_prefix}.model")
    test = "Create an aspirational objective to improve customer retention"
    tokens = sp.encode(test, out_type=str)
    ids = sp.encode(test)
    print(f"\nTest: '{test}'")
    print(f"Tokens ({len(tokens)}): {tokens}")
    print(f"IDs: {ids}")
    print(f"Roundtrip: '{sp.decode(ids)}'")

    return output_prefix


def generate_seed_texts() -> list[str]:
    """Generate synthetic training text from domain vocabulary when no corpus exists."""
    templates = [
        "I want to improve {metric} by {pct}% this quarter",
        "Create an {type} objective for {domain}",
        "Update the score on {kr} to {score}",
        "Show me the {level} OKRs for {cycle}",
        "Generate a {report} report for the team",
        "Onboard {name} as a {role} on the {team} team",
        "Align my objectives with the {level} goals",
        "Our goal is to {action} {metric} from {start} to {target}",
        "We hit {pct}% on the {kr} key result",
        "How is the {team} team tracking against their OKRs",
    ]
    fills = {
        "metric": ["customer retention", "revenue", "NPS", "conversion rate", "churn", "engagement"],
        "pct": ["10", "20", "30", "50", "15", "25"],
        "type": ["committed", "aspirational"],
        "domain": ["engineering", "sales", "marketing", "product", "customer success"],
        "kr": ["customer churn reduction", "pipeline growth", "feature adoption", "NPS improvement"],
        "score": ["0.3", "0.5", "0.7", "0.8", "0.9", "1.0"],
        "level": ["company", "department", "team", "individual"],
        "cycle": ["Q1 2026", "Q2 2026", "Q3 2026", "H1 2026"],
        "report": ["progress", "health", "summary"],
        "name": ["Sarah", "James", "Priya", "Alex", "Maria"],
        "role": ["data scientist", "engineer", "designer", "product manager", "analyst"],
        "team": ["engineering", "product", "data", "growth", "platform"],
        "action": ["increase", "reduce", "improve", "grow", "accelerate"],
        "start": ["5%", "100", "3.2", "50%", "1000"],
        "target": ["2%", "500", "4.5", "80%", "5000"],
    }

    import random
    texts = []
    for _ in range(5000):
        template = random.choice(templates)
        text = template
        for key, values in fills.items():
            text = text.replace("{" + key + "}", random.choice(values), 1)
        texts.append(text)

    # Add raw JSON examples
    for _ in range(2000):
        tc = {
            "tool": random.choice(["cycle", "objective", "key_result", "user", "report"]),
            "action": random.choice(["create", "get", "list", "update", "delete"]),
            "params": {"cycleId": f"cyc_{random.randint(1,100)}"},
        }
        texts.append(json.dumps(tc))

    return texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train task-specific BPE tokenizer")
    parser.add_argument("--corpus", default="corpus.jsonl", help="Corpus JSONL file")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Vocabulary size")
    parser.add_argument("--prefix", default="okr_tokenizer", help="Model prefix")
    args = parser.parse_args()

    train_tokenizer(corpus_file=args.corpus, vocab_size=args.vocab_size, model_prefix=args.prefix)
