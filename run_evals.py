import argparse
import importlib
import json
import re

import judge
from langsmith_setup import setup_tracing

AGENTS = {
    "github": ("github_agent", "evals/github_eval_dataset.json"),
    "reddit": ("reddit_agent", "evals/reddit_eval_dataset.json"),
}

def load_eval_dataset(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def score_run(test_case: dict, agent_result: dict) -> dict:
    messages = agent_result.get("messages", [])
    if not messages:
        return {"test": test_case["name"], "score": 0, "max": 3, "passed": False, "error": "No messages returned"}

    final_content = messages[-1].content or ""

    # Extract tool calls from all messages
    tools_called = []
    all_text = []
    for msg in messages:
        if getattr(msg, "content", None):
            all_text.append(str(msg.content))
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if isinstance(tc, dict):
                    tools_called.append(tc.get("name", "unknown"))
                else:
                    tools_called.append(getattr(tc, "name", "unknown"))

    # Scoring
    expected_tools = test_case.get("expected_tools", [])
    tools_matched = any(t in tools_called for t in expected_tools)

    expected_outputs = test_case.get("expected_in_output", [])
    output_hits = sum(1 for exp in expected_outputs if exp.lower() in final_content.lower())
    output_score = output_hits / max(len(expected_outputs), 1)

    # Hallucination canaries: strings that must NOT appear in the final answer
    forbidden = test_case.get("forbidden_in_output", [])
    if any(bad.lower() in final_content.lower() for bad in forbidden):
        output_score = 0.0

    # Limit adherence: our formatters number items "- #1", "- #2", ...
    max_items = test_case.get("max_list_items")
    if max_items is not None:
        indices = [int(m) for m in re.findall(r"- #(\d+)", final_content)]
        if indices and max(indices) > max_items:
            output_score = 0.0

    # A completed run with a non-empty final answer earns the no-crash point.
    # For expect_error cases, the failure must be visible somewhere in the trace
    # (final answer or tool output) rather than silently swallowed.
    has_error = "error" in "\n".join(all_text).lower()
    no_crash = bool(final_content.strip()) if not test_case.get("expect_error") else has_error

    score = (1.0 if tools_matched else 0.0) + output_score + (1.0 if no_crash else 0.0)

    return {
        "test": test_case["name"],
        "score": round(score, 2),
        "max": 3,
        "tools_called": tools_called,
        "passed": score >= 2.0,
        "snippet": final_content[:200].replace("\n", " ")
    }

def run_suite(agent_name: str, judge_enabled: bool = False) -> list:
    module_name, dataset_path = AGENTS[agent_name]
    module = importlib.import_module(module_name)
    dataset = load_eval_dataset(dataset_path)
    results = []

    if judge_enabled:
        if judge.is_configured():
            print(f"⚖️  Faithfulness judge: {judge.JUDGE_MODEL} (remote)\n")
        else:
            print("⚠️  --judge was set but ANTHROPIC_API_KEY is missing from .env; skipping judge.\n")
            judge_enabled = False

    print(f"🔧 [{agent_name}] Running {len(dataset)} eval cases...\n")

    for case in dataset:
        print(f"▶️  {case['name']}: {case['input'][:60]}...")
        try:
            result = module.run_agent(case["input"])
            score = score_run(case, result)
        except Exception as e:
            score = {"test": case["name"], "score": 0, "max": 3, "tools_called": [],
                     "passed": False, "snippet": f"AGENT CRASHED: {type(e).__name__}: {e}"}
            results.append(score)
            print(f"   ❌ CRASH | {score['snippet'][:120]}")
            print()
            continue

        if judge_enabled:
            tool_outputs = "\n\n".join(
                str(m.content) for m in result.get("messages", []) if getattr(m, "type", "") == "tool"
            )
            final_answer = str(result["messages"][-1].content or "")
            try:
                verdict = judge.judge_faithfulness(tool_outputs, final_answer)
                score["faithful"] = verdict["faithful"]
                score["judge_reason"] = verdict["reason"]
                if not verdict["faithful"]:
                    score["passed"] = False
            except Exception as e:
                score["faithful"] = None
                score["judge_reason"] = f"judge error: {type(e).__name__}: {e}"
        results.append(score)

        status = "✅ PASS" if score["passed"] else "❌ FAIL"
        print(f"   {status} | Score: {score['score']}/3 | Tools: {score['tools_called']}")
        if judge_enabled:
            marker = "?" if score.get("faithful") is None else ("✓ faithful" if score["faithful"] else "✗ UNFAITHFUL")
            print(f"   Judge: {marker} — {score.get('judge_reason', '')[:120]}")
        print(f"   Output: {score['snippet'][:100]}...")
        print()

    return results

def main():
    parser = argparse.ArgumentParser(description="Run agent eval suites")
    parser.add_argument("--agent", choices=["github", "reddit", "all"], default="all",
                        help="Which agent suite to run (default: all)")
    parser.add_argument("--judge", action="store_true",
                        help="Enable the remote faithfulness judge (requires ANTHROPIC_API_KEY in .env)")
    args = parser.parse_args()

    setup_tracing()

    names = list(AGENTS) if args.agent == "all" else [args.agent]
    all_results = {name: run_suite(name, judge_enabled=args.judge) for name in names}

    print("=" * 50)
    for name, results in all_results.items():
        total = sum(r["score"] for r in results)
        max_total = sum(r["max"] for r in results)
        passed = sum(1 for r in results if r["passed"])
        print(f"📊 [{name}] FINAL: {total}/{max_total} ({total/max_total*100:.1f}%) | Passed: {passed}/{len(results)}")

if __name__ == "__main__":
    main()
