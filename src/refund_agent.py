"""Database-backed refund tools, ReAct execution, and resumable checkpoints."""

import json
import math
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import mock_llm
from src.policy_rag import PolicyRAG


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "instamart.db"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"
RUN_LOG = OUTPUT_DIR / "agent_runs.jsonl"
LEDGER = OUTPUT_DIR / "refund_ledger.jsonl"
POLICY_RAG = PolicyRAG()


class SimulatedAgentCrash(RuntimeError):
    """Raised by the resume demonstration after a durable tool checkpoint."""


def get_order(order_id):
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return dict(
        zip(
            ["order_id", "status", "order_amount", "delivery_minutes"],
            row,
        )
    )


def search_policy(query):
    return {"results": POLICY_RAG.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes, order_status="completed"):
    if order_status != "completed" or delivery_minutes is None:
        return {"refund_inr": 0.0}
    amount = float(order_amount) if float(delivery_minutes) > 45 else 0.0
    return {"refund_inr": round(amount, 2)}


def _validated_refund_action(order_id, refund_inr, tool_name):
    order = get_order(order_id)
    if "error" in order or order["status"] != "completed":
        raise ValueError(f"{tool_name} requires a completed order.")
    expected_refund = calculate_refund(
        order["order_amount"],
        order["delivery_minutes"],
        order_status=order["status"],
    )["refund_inr"]
    if not math.isclose(
        float(refund_inr), expected_refund, rel_tol=0.0, abs_tol=0.01
    ):
        raise ValueError(f"{tool_name} amount does not match the refund rules.")
    if tool_name == "issue_refund" and not 0 < expected_refund <= 1000:
        raise ValueError("Only positive refunds up to INR 1000 can be issued.")
    if tool_name == "escalate_to_human" and expected_refund <= 1000:
        raise ValueError("Only refunds above INR 1000 can be escalated.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }
    with LEDGER.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(record) + "\n")
    return {"refund_inr": record["refund_inr"]}


def issue_refund(order_id, refund_inr):
    return _validated_refund_action(order_id, refund_inr, "issue_refund")


def escalate_to_human(order_id, refund_inr):
    return _validated_refund_action(order_id, refund_inr, "escalate_to_human")


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up order status, amount, and delivery time by order ID.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Retrieve the three policy passages most relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": (
            "Calculate a full late-delivery refund for a completed order "
            "delivered in more than 45 minutes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": ["number", "null"]},
                "order_status": {
                    "type": "string",
                    "enum": ["completed", "cancelled"],
                    "description": "Verified by get_order before refund calculation.",
                },
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an eligible refund up to INR 1000 in the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record an eligible refund above INR 1000 for human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def _checkpoint_path(run_id):
    if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
        raise ValueError("run_id must be a simple file name.")
    return STATE_DIR / f"{run_id}.json"


def _write_checkpoint(
    run_id,
    scenario_id,
    messages,
    tokens_used,
    steps,
    llm_calls,
    max_iterations,
    token_limit,
    time_limit_seconds,
):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": len(steps),
        "tools_executed": [step["tool"] for step in steps],
        "steps": steps,
        "llm_calls": llm_calls,
        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,
    }
    _checkpoint_path(run_id).write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _last_order_lookup(steps):
    for step in reversed(steps):
        if step["tool"] == "get_order":
            return step["observation"]
    return None


def _fallback(status, steps, llm_calls, tokens_used, detail):
    return {
        "status": status,
        "final_answer": f"{status}: {detail}",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
    }


def _continue_agent(
    run_id,
    scenario_id,
    messages,
    steps,
    llm_calls,
    tokens_used,
    max_iterations,
    token_limit,
    time_limit_seconds,
    crash_after=None,
):
    started = time.monotonic()
    mock_llm.reset()
    while llm_calls < max_iterations:
        if time.monotonic() - started >= time_limit_seconds:
            return _fallback(
                "FALLBACK_TIMEOUT",
                steps,
                llm_calls,
                tokens_used,
                "handoff to a human agent; the time limit was reached.",
            )

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += int(response.usage.total_tokens)
        if tokens_used > token_limit:
            return _fallback(
                "FALLBACK_TOKEN_LIMIT",
                steps,
                llm_calls,
                tokens_used,
                "handoff to a human agent; the token limit was exceeded.",
            )

        assistant_message = response.choices[0].message
        if not assistant_message.tool_calls:
            final_answer = assistant_message.content or ""
            return {
                "status": final_answer.split(":", 1)[0],
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }

        tool_call = assistant_message.tool_calls[0]
        tool_name = tool_call.name
        if tool_name not in TOOL_MAP:
            return _fallback(
                "FALLBACK_UNKNOWN_TOOL",
                steps,
                llm_calls,
                tokens_used,
                f"handoff to a human agent; unsupported tool {tool_name!r}.",
            )
        model_arguments = dict(tool_call.arguments)
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "name": tool_name,
                        "arguments": model_arguments,
                    }
                ],
            }
        )

        call_arguments = dict(model_arguments)
        if tool_name == "calculate_refund":
            order = _last_order_lookup(steps)
            if order is None or "error" in order:
                observation = {"refund_inr": 0.0}
            else:
                call_arguments["order_status"] = order["status"]
                observation = TOOL_MAP[tool_name](**call_arguments)
        else:
            observation = TOOL_MAP[tool_name](**call_arguments)

        steps.append(
            {
                "tool": tool_name,
                "arguments": call_arguments,
                "observation": observation,
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(observation, ensure_ascii=False),
            }
        )
        _write_checkpoint(
            run_id,
            scenario_id,
            messages,
            tokens_used,
            steps,
            llm_calls,
            max_iterations,
            token_limit,
            time_limit_seconds,
        )
        if crash_after is not None and len(steps) == crash_after:
            raise SimulatedAgentCrash(
                f"Simulated crash after checkpoint for tool {tool_name}."
            )

    return _fallback(
        "FALLBACK_MAX_ITERATIONS",
        steps,
        llm_calls,
        tokens_used,
        "handoff to a human agent; the maximum model-call count was reached.",
    )


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after=None,
):
    run_id = str(run_id or uuid.uuid4())
    messages = [
        {
            "role": "system",
            "content": (
                "AGENT: Verify the order, consult refund policy, and use only "
                "the supplied refund tools. Never refund a non-completed order."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    return _continue_agent(
        run_id=run_id,
        scenario_id=scenario_id,
        messages=messages,
        steps=[],
        llm_calls=0,
        tokens_used=0,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after=crash_after,
    )


def resume_agent(run_id):
    checkpoint_file = _checkpoint_path(run_id)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"No saved agent checkpoint for run {run_id!r}.")
    checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    if checkpoint.get("run_id") != run_id:
        raise ValueError("Checkpoint run_id does not match the requested run.")
    return _continue_agent(
        run_id=run_id,
        scenario_id=checkpoint["scenario_id"],
        messages=checkpoint["messages"],
        steps=checkpoint["steps"],
        llm_calls=checkpoint["llm_calls"],
        tokens_used=checkpoint["tokens_used"],
        max_iterations=checkpoint["max_iterations"],
        token_limit=checkpoint["token_limit"],
        time_limit_seconds=checkpoint["time_limit_seconds"],
    )


def _append_run(scenario_id, result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "scenario_id": scenario_id,
        "status": result["status"],
        "final_answer": result["final_answer"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "steps": result["steps"],
    }
    with RUN_LOG.open("a", encoding="utf-8") as run_log:
        run_log.write(json.dumps(record, ensure_ascii=False) + "\n")


def _refund_amount(result):
    for step in reversed(result["steps"]):
        if step["tool"] in {"issue_refund", "escalate_to_human"}:
            return float(step["observation"]["refund_inr"])
    return 0.0


def run_scenarios():
    scenarios = pd.read_csv(
        ROOT / "data" / "raw" / "agent_scenarios.csv",
        keep_default_na=False,
    )
    results = []
    for scenario in scenarios.itertuples(index=False):
        result = run_agent(
            scenario.message,
            scenario_id=scenario.scenario_id,
        )
        result["refund_inr"] = _refund_amount(result)
        result["scenario_id"] = scenario.scenario_id
        result["expected_outcome"] = scenario.expected_outcome
        result["expected_refund_inr"] = float(scenario.expected_refund_inr)
        results.append(result)
        _append_run(scenario.scenario_id, result)

    demo_scenario = scenarios.loc[scenarios["scenario_id"].eq("S1")].iloc[0]
    token_demo = run_agent(
        demo_scenario["message"],
        scenario_id="S1_token_limit_200",
        token_limit=200,
    )
    _append_run("S1_token_limit_200", token_demo)
    timeout_demo = run_agent(
        demo_scenario["message"],
        scenario_id="S1_timeout_0",
        time_limit_seconds=0,
    )
    _append_run("S1_timeout_0", timeout_demo)

    crash_run_id = f"s1-resume-{uuid.uuid4()}"
    try:
        run_agent(
            demo_scenario["message"],
            scenario_id="S1_crash_resume",
            run_id=crash_run_id,
            crash_after=2,
        )
    except SimulatedAgentCrash:
        crash_checkpoint = json.loads(
            _checkpoint_path(crash_run_id).read_text(encoding="utf-8")
        )
        if (
            crash_checkpoint["step_number"] != 2
            or crash_checkpoint["tools_executed"]
            != ["get_order", "search_policy"]
        ):
            raise AssertionError("The crash checkpoint did not capture tool 2.")
        resumed = resume_agent(crash_run_id)
    else:
        raise AssertionError("The simulated crash did not occur.")
    resumed["refund_inr"] = _refund_amount(resumed)
    _append_run("S1_crash_resume", resumed)
    return {
        "scenarios": results,
        "token_demo": token_demo,
        "timeout_demo": timeout_demo,
        "crash_resume": resumed,
        "crash_checkpoint_tools": crash_checkpoint["tools_executed"],
        "crash_resume_tools": [step["tool"] for step in resumed["steps"]],
        "crash_run_id": crash_run_id,
    }


def run_part_d():
    from src.policy_rag import evaluate, evaluation_summary

    rag = PolicyRAG(overlap=10)
    zero_overlap_rag = PolicyRAG(overlap=0)
    return {
        "chunk_count": len(rag.chunks),
        "chunk_counts_by_source": pd.Series(
            [chunk["source"] for chunk in rag.chunks]
        ).value_counts().to_dict(),
        "refund_query_sources": [
            hit["source"] for hit in rag.retrieve("refund policy late delivery")
        ],
        "policy_eval": evaluation_summary(evaluate(rag)),
        "zero_overlap_eval": evaluation_summary(evaluate(zero_overlap_rag)),
        "scenario_runs": run_scenarios(),
    }


if __name__ == "__main__":
    report = run_part_d()
    print("Chunk count by source:", report["chunk_counts_by_source"])
    print("Refund query top three:", report["refund_query_sources"])
    print("Policy evaluation:", report["policy_eval"])
    print("Zero-overlap evaluation:", report["zero_overlap_eval"])
    for result in report["scenario_runs"]["scenarios"]:
        print(
            result["scenario_id"],
            result["status"],
            result["refund_inr"],
            result["llm_calls"],
            result["tokens_used"],
        )
    print("Token-limit demo:", report["scenario_runs"]["token_demo"]["status"])
    print("Timeout demo:", report["scenario_runs"]["timeout_demo"]["status"])
    print(
        "Crash/resume tools:",
        report["scenario_runs"]["crash_checkpoint_tools"],
        "then",
        report["scenario_runs"]["crash_resume_tools"],
        "status:",
        report["scenario_runs"]["crash_resume"]["status"],
        "refund:",
        report["scenario_runs"]["crash_resume"]["refund_inr"],
    )
