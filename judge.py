import json
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()

# The judge is deliberately a separate, remote model: faithfulness checking is
# the judgment small local models are worst at, and it makes only one short
# call per test case, so cost is negligible compared to the agent loop.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-5")

PROMPT = """You are grading the faithfulness of an AI agent's answer.

The agent was given a user question, called tools that returned real data, and produced a final answer. Judge ONLY whether the final answer is faithful to the tool outputs: every factual claim in the answer must be supported by (or directly relayed from) the tool data. An answer that honestly reports a tool error or empty result IS faithful. An answer that invents facts, misattributes data, or contradicts the tool outputs is NOT faithful, even if it sounds plausible or contains expected keywords.

Tool outputs:
<tool_data>
{tool_data}
</tool_data>

Agent's final answer:
<answer>
{answer}
</answer>

Respond with a JSON object only, no prose:
{{"faithful": true or false, "reason": "one sentence"}}"""


def is_configured() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def judge_faithfulness(tool_outputs: str, final_answer: str) -> dict:
    """Ask the judge model whether the answer is grounded in the tool data.

    Returns {"faithful": bool, "reason": str}. Raises requests.HTTPError on API errors.
    """
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": JUDGE_MODEL,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": PROMPT.format(tool_data=tool_outputs[:8000], answer=final_answer[:4000]),
            }],
        },
        timeout=60,
    )
    response.raise_for_status()
    text = "".join(block.get("text", "") for block in response.json().get("content", []))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Judge returned unparseable response: {text[:200]}")
    verdict = json.loads(match.group(0))
    return {"faithful": bool(verdict.get("faithful")), "reason": str(verdict.get("reason", ""))}
