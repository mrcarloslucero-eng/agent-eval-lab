# Agent Eval Lab

A local-first eval lab for tool-calling agents. Two agents (GitHub, Reddit)
run entirely on a local Ollama model; one shared runner scores both against
complementary eval datasets, with an optional remote LLM judge grading
faithfulness. Every run traces to LangSmith.

The full write-up — architecture, scoring rubric, failure modes, and the
engineering log of problems hit and fixed — is in
[AGENT_EVAL_LAB_Study_Guide.md](AGENT_EVAL_LAB_Study_Guide.md).

## Architecture

```
github_agent.py   → agent under test #1: get_commits, get_issues
reddit_agent.py   → agent under test #2: get_hot_posts, search_posts
judge.py          → remote faithfulness judge (Claude, one call per case)
run_evals.py      → runner + scorer for both suites
evals/            → one dataset per agent, each targeting different failure modes
```

Design rule: **agent local, judge remote, no conditional routing.** The agent
grinds through multi-turn tool loops cheaply on your GPU; the judge makes one
short call per test case and faithfulness is the judgment small models are
worst at.

## Setup

```bash
python -m venv venv
venv/Scripts/activate          # Windows
pip install -r requirements.txt

ollama pull llama3.1:8b        # the agent under test
```

Copy `.env.example` to `.env` and fill in:

- `LANGCHAIN_API_KEY` — from https://smith.langchain.com (tracing; free tier)
- `ANTHROPIC_API_KEY` — only needed for `--judge`
- `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` — only needed for the Reddit
  suite's happy-path cases (free app at https://www.reddit.com/prefs/apps).
  Without them the Reddit tools fail gracefully and the suite exercises only
  its failure-mode cases.

## Usage

```bash
python run_evals.py                      # both suites, deterministic scoring
python run_evals.py --agent reddit       # one suite
python run_evals.py --judge              # add remote faithfulness judging
```

Pass = score ≥ 2.0/3 (tool selection + output keywords + no crash), plus a
faithful verdict from the judge when `--judge` is on. An UNFAITHFUL verdict
fails the case regardless of score.

## The two datasets on purpose

- `github_eval_dataset.json` — tool selection, expected content, graceful 404.
- `reddit_eval_dataset.json` — limit adherence (`max_list_items`), tool
  disambiguation, hallucination canaries (`forbidden_in_output`), honest
  error reporting (`expect_error`).

One dataset per blind spot — see the study guide for why.

## Security notes

`.env` is gitignored and never committed. Reddit OAuth tokens, LangSmith and
Anthropic keys all live there.
