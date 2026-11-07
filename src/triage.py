"""Privacy-first support ticket triage using the supplied offline mock LLM."""

import json
import random
import re
import time
from pathlib import Path

import pandas as pd

import mock_llm


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
TICKET_FILE = RAW / "tickets.csv"
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

ZERO = """CLASSIFY the support ticket into exactly one allowed category: billing, delivery, technical, or fraud.
Reply with JSON only, in the form {"category": "...", "reason": "..."}."""
FEW = """CLASSIFY the support ticket into exactly one allowed category: billing, delivery, technical, or fraud.
Reply with JSON only, in the form {"category": "...", "reason": "..."}.
EXAMPLES:
Unauthorized transaction on my card. -> {"category": "fraud", "reason": "unauthorized transaction"}
My order arrived late. -> {"category": "delivery", "reason": "late delivery"}
The app crashes when I try to log in. -> {"category": "technical", "reason": "app crash"}"""


def redact(text):
    """Redact PII in the required order and return per-pattern substitution counts."""
    counts = {}
    for name, pattern, replacement in PATTERNS:
        text, counts[name] = pattern.subn(replacement, text)
    return text, counts


def _validate_reply(content):
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass
    else:
        if isinstance(parsed, dict):
            category = parsed.get("category")
            if isinstance(category, str) and category in ALLOWED_CATEGORIES:
                return category, "classified_json"
        return "", "flagged_invalid_output"

    words = re.findall(r"[a-z]+", (content or "").lower())
    categories = set(words) & ALLOWED_CATEGORIES
    if len(categories) == 1:
        return categories.pop(), "classified_keyword"
    return "", "flagged_invalid_output"


def _retry_chat(messages, ticket_id):
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages), retry_number
        except mock_llm.TransientLLMError:
            if retry_number == MAX_RETRIES:
                raise
            wait = (
                BASE_DELAY
                * (2**retry_number)
                * random.uniform(0.9, 1.15)
            )
            print(
                f"retry ticket={ticket_id} retry={retry_number} wait={wait:.4f}s"
            )
            time.sleep(wait)
    raise RuntimeError("Retry loop exited without a response.")


def _segment_map():
    segment_file = OUT / "customer_segments.csv"
    if not segment_file.exists():
        raise FileNotFoundError(
            f"Part B segment file is required for ticket routing: {segment_file}"
        )
    segments = pd.read_csv(segment_file, dtype={"customer_id": str})
    return segments.set_index("customer_id").to_dict("index")


def _route(status, category, segment):
    if status in {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if segment is not None and bool(segment.get("is_premium", False)):
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


def run_pipeline(prompt=FEW, token_budget=None, output_path=None, seed=42):
    """Run one complete, ordered ticket pass and optionally persist the result."""
    mock_llm.reset()
    random.seed(seed)

    tickets = pd.read_csv(TICKET_FILE, keep_default_na=False)
    segments = _segment_map()
    redaction_totals = {name: 0 for name, _, _ in PATTERNS}
    redacted_texts = []
    remaining_matches = 0
    for text in tickets["text"].astype(str):
        redacted, counts = redact(text)
        redacted_texts.append(redacted)
        for name, count in counts.items():
            redaction_totals[name] += count
        remaining_matches += sum(
            len(pattern.findall(redacted)) for _, pattern, _ in PATTERNS
        )
    assert remaining_matches == 0, "A PII pattern remained after redaction."

    results = []
    total_tokens = 0
    retry_count = 0
    for (_, ticket), safe_text in zip(tickets.iterrows(), redacted_texts):
        original_text = str(ticket["text"])
        category = ""
        tokens_used = 0
        if not original_text.strip():
            status = "skipped_empty"
        elif mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            status = "budget_exhausted"
        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": safe_text},
            ]
            assert messages[-1]["content"] == safe_text
            assert messages[-1]["content"] == safe_text
            assert all(
                pattern.search(message["content"]) is None
                for message in messages
                for _, pattern, _ in PATTERNS
            ), "Unredacted PII would be sent to the model."
            response, retries_used = _retry_chat(messages, ticket["ticket_id"])
            retry_count += retries_used
            tokens_used = int(response.usage.total_tokens)
            total_tokens += tokens_used
            category, status = _validate_reply(
                response.choices[0].message.content
            )

        segment = segments.get(str(ticket["customer_id"]))
        route, priority = _route(status, category, segment)
        results.append(
            {
                "ticket_id": ticket["ticket_id"],
                "customer_id": ticket["customer_id"],
                "status": status,
                "category": category,
                "segment_name": (
                    segment.get("segment_name", "") if segment is not None else ""
                ),
                "priority": priority,
                "route": route,
                "tokens_used": tokens_used,
                "true_category": ticket["true_category"],
            }
        )

    result = pd.DataFrame(results)
    status_counts = result["status"].value_counts().to_dict()
    classified = result["status"].isin(
        {"classified_json", "classified_keyword"}
    )
    accuracy = (
        float(
            (
                result.loc[classified, "category"]
                == result.loc[classified, "true_category"]
            ).mean()
        )
        if classified.any()
        else float("nan")
    )
    summary = {
        "tickets": len(result),
        "classified": int(classified.sum()),
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_classified_ticket": (
            total_tokens / int(classified.sum()) if classified.any() else 0.0
        ),
        "status_counts": status_counts,
        "redactions": {
            **redaction_totals,
            "total": sum(redaction_totals.values()),
            "remaining_matches": remaining_matches,
        },
        "retry_count": retry_count,
        "budget_exhausted": int((result["status"] == "budget_exhausted").sum()),
    }

    if output_path is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        result[
            [
                "ticket_id",
                "customer_id",
                "status",
                "category",
                "segment_name",
                "priority",
                "route",
                "tokens_used",
            ]
        ].to_csv(output_path, index=False)
    return result, summary


def compare_prompts():
    """Run and summarize zero-shot/few-shot evaluations and a 6000-token demo."""
    zero_results, zero_summary = run_pipeline(
        prompt=ZERO, token_budget=None, seed=42
    )
    few_results, few_summary = run_pipeline(
        prompt=FEW,
        token_budget=None,
        output_path=OUT / "triage_results.csv",
        seed=42,
    )
    zero_classified = zero_results["status"].isin(
        {"classified_json", "classified_keyword"}
    )
    zero_confusion = pd.crosstab(
        zero_results.loc[zero_classified, "true_category"],
        zero_results.loc[zero_classified, "category"],
        rownames=["true_category"],
        colnames=["predicted_category"],
        dropna=False,
    ).reindex(
        index=["billing", "delivery", "technical", "fraud"],
        columns=["billing", "delivery", "technical", "fraud"],
        fill_value=0,
    )
    budget_results, budget_summary = run_pipeline(
        prompt=ZERO,
        token_budget=6000,
        output_path=OUT / "triage_budget_demo.csv",
        seed=42,
    )
    return {
        "zero_shot": zero_summary,
        "zero_shot_confusion": zero_confusion,
        "few_shot": few_summary,
        "budget_demo": budget_summary,
        "few_shot_results": few_results,
        "budget_results": budget_results,
    }


if __name__ == "__main__":
    comparison = compare_prompts()
    print("Zero-shot:", comparison["zero_shot"])
    print("Zero-shot confusion matrix:\n", comparison["zero_shot_confusion"])
    print("Few-shot:", comparison["few_shot"])
    print("6000-token zero-shot budget demo:", comparison["budget_demo"])
