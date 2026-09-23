# Agent Eval Lab — Study Guide & Reference (Updated)

Second edition. Covers the original GitHub agent lab plus the Reddit agent,
the LLM-as-judge faithfulness layer, and every problem hit along the way with
its fix. For setup instructions see `README.md`.

---

## Part 1: What the Lab Is Now

Two tool-calling agents, two eval datasets, one shared runner, and an optional
remote judge. All agent inference runs locally on an 8GB GPU; only the
faithfulness judge calls the cloud.

```
AGENT EVAL LAB/
├── github_agent.py            # Agent under test #1 (GitHub tools)
├── reddit_agent.py            # Agent under test #2 (Reddit tools)
├── judge.py                   # Remote faithfulness judge (Claude)
├── run_evals.py               # Runner + scorer for both suites (--agent, --judge)
├── langsmith_setup.py         # Tracing configuration
├── evals/
│   ├── github_eval_dataset.json   # 5 cases: tool selection, keywords, graceful 404
│   └── reddit_eval_dataset.json   # 5 cases: NEW failure modes (below)
├── .env.example               # Template for all required keys
└── requirements.txt           # Pinned dependencies
```

### Run commands

```bash
python run_evals.py                          # both suites, no judge
python run_evals.py --agent reddit           # one suite
python run_evals.py --judge                  # add remote faithfulness judging
```

---

## Part 2: The Two Agents

**GitHub agent** (`github_agent.py`): tools `get_commits`, `get_issues`.
Anonymous GitHub API works fine — no auth needed at low volume.

**Reddit agent** (`reddit_agent.py`): tools `get_hot_posts`, `search_posts`.
Reddit **blocks anonymous `.json` access** (returns HTTP 403, or worse, an
HTML block page with HTTP 200 — see Problems & Fixes). The tools therefore
prefer OAuth client-credentials tokens
(`REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` from `.env`, free app at
reddit.com/prefs/apps) and fall back to anonymous requests so the failure is
visible instead of fatal.

Both agents are deliberately identical in shape: same model
(`llama3.1:8b` via Ollama, temperature 0), same ReAct loop, same
`run_agent(query)` entry point. That symmetry is what lets one runner score
both.

---

## Part 3: The Scoring Rubric (3 Points + Judge Gate)

| Criterion | Points | How it is scored |
|---|---|---|
| Tool selection | 1 | Did it call a tool in `expected_tools`? |
| Output quality | 1 | Fraction of `expected_in_output` keywords present |
| No crash | 1 | Completed with a non-empty final answer; `expect_error` cases need the failure visible in the trace |
| **Faithfulness** | gate | With `--judge`: a verdict of UNFAITHFUL forces `passed=False` regardless of score |

New optional per-case fields (both datasets):

- `expect_error: true` — the run *should* produce an error (dead repo,
  dead subreddit). Failure must be visible somewhere in the message trace.
- `forbidden_in_output: [...]` — hallucination canaries. If any string
  appears in the final answer, the output point is zeroed. Used by
  `empty_search_no_hallucination`: an agent that invents a post when search
  returns nothing loses the point.
- `max_list_items: N` — limit adherence. Tool formatters number items
  `- #1`, `- #2`, ...; listing more than N zeroes the output point. This is
  a failure mode the GitHub suite never tested.

Pass threshold: score ≥ 2.0/3 **and** (if judged) faithful.

---

## Part 4: Failure Modes — Why Two Datasets

The datasets are intentionally complementary. A single dataset teaches you
one blind spot at a time.

**GitHub dataset** tests: did it pick the right tool, does the answer contain
expected content, does it survive a 404.

**Reddit dataset** tests the things GitHub's didn't:

1. **Limit adherence** — "exactly 2 posts" → list no more than 2.
2. **Tool disambiguation** — "search for X" must pick `search_posts`, not `get_hot_posts`.
3. **Hallucination under empty results** — gibberish query → must say "no posts", must not invent one.
4. **Graceful degradation at the API level** — blocked/403/missing subreddit → report honestly.

This mirrors the rule from the first edition: happy path + edge case +
adversarial case, per task.

---

## Part 5: The Faithfulness Judge

**Architecture decision: agent local, judge always remote.** No conditional
routing.

- The agent (llama3.1:8b) grinds through multi-turn tool loops — cheap
  locally, correct placement.
- The judge (`judge.py`, Claude Sonnet by default, `JUDGE_MODEL` env override)
  makes **one short call per test case**. Cost per full suite run: pennies.
- Faithfulness is the judgment small models are worst at: comparing answer to
  source for subtle contradiction. A local judge gives precise-looking scores
  with invisible error bars.
- Conditional routing ("use the small judge when...") would itself be a
  component needing evals — a second silent failure surface. Always-on remote
  has zero routing logic to debug.

The judge receives tool outputs + final answer and returns
`{"faithful": bool, "reason": str}`. Honest error reporting counts as
faithful; invented facts are unfaithful even if keyword-rich. Without
`ANTHROPIC_API_KEY`, `--judge` prints a warning and runs unscored.

**Why keyword matching can't do this job:** an agent can hallucinate a wrong
answer full of the word "commit" and pass every keyword check. String match
is a fast gate; faithfulness needs a reader.

---

## Part 6: Problems & Fixes (the actual engineering log)

1. **Reddit anonymous `.json` 403.** `curl https://www.reddit.com/r/LocalLLaMA.json`
   returns 403 + HTML block page. → OAuth client-credentials support in the
   tools; anonymous kept as fallback so the failure is visible.
2. **Reddit 200-with-HTML.** Worse than a 403: the block page sometimes comes
   back as HTTP 200, so `response.json()` exploded with `JSONDecodeError`,
   which crashed the *whole eval run* (tool exception → agent → runner).
   → `JSONDecodeError` caught in `_fetch_posts`, converted to a graceful tool
   error; runner also wraps each case in try/except so one crash = one failed
   case, not an aborted suite.
3. **Windows cp1252 console crash.** `print("✅ ...")` in
   `langsmith_setup.py` died with `UnicodeEncodeError` before any eval ran.
   → `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at import.
4. **Brittle error detection.** `nonexistent_subreddit` FAILED not because
   the agent did badly — it relayed the error as "subreddit does not exist or
   is private" without the literal word "error". The scorer demanded
   `"error" in final_answer`. → `expect_error` cases now check the whole
   message trace (final answer + tool outputs); `no_crash` for normal cases
   means "completed with a non-empty final answer", since real crashes are
   caught at the runner level.
5. **Tool selection was passing unverifiable limits.** "List exactly 2 posts"
   had no teeth. → numbered formatter output + `max_list_items` scorer check.

**The judge does not train the agent — and shouldn't.** The agent is a frozen
base model; nothing about an eval run updates its weights. The learning loop
here is the *developer*: judge findings → improve the scaffolding (tool error
messages, system prompt, tool design) → re-run the suite → confirm the score
moved. Example: the misattribution finding below points at two cheap fixes —
make the tool error say "AUTH BLOCKED, do not guess why" more explicitly, and
add a system-prompt rule like "if a tool errors, relay the error exactly."
That is how this lab gets better. (Actual model learning — LoRA fine-tuning
on faithful trajectories, or RL-style reward from the judge — is a real but
separate project; don't conflate it with evals.)

**The judge's first real run scored the agent 2/5** — it caught soft
hallucinations on three cases: reporting a 403 auth block as "the subreddit
does not exist", reframing it as "no relevant posts found", and inventing
`praw` code details never in the tool output. All three read as plausible,
honest answers. All three passed every keyword check. None survived the
judge. This is a real, actionable finding about the agent — it misattributes
blocked-API errors — not a scoring quirk.

---

## Part 7: Results Baseline (this environment, llama3.1:8b)

| Suite | Score | Passed | Notes |
|---|---|---|---|
| github | 12.5/15 (83.3%) | 5/5 | unchanged after refactor |
| reddit, keyword gate only (no creds) | 10.0/15 (66.7%) | 5/5 | happy-path points await Reddit OAuth keys |
| reddit, with judge (no creds) | 10.0/15 (66.7%) | **2/5** | judge flagged 3 soft hallucinations the keywords missed |
| reddit (with creds) | — | — | rerun after adding keys; expect limit + no-hallucination cases to gain points |

---

## Part 8: Key Takeaways (updated)

1. Evals and tracing turn a demo into a product — and a *faithfulness judge*
   turns evals into evidence.
2. Agent local, judge remote, no routing. The 8GB card is a feature for the
   loop; the judge is too cheap to cheap out on.
3. APIs lie in three ways: status codes (403), bodies (200-with-HTML), and
   your own assumptions (agent paraphrases errors). Score behavior, not words.
4. Datasets should each target a different failure mode, or you're testing
   the same blind spot ten times.
5. Build, break, fix, repeat — every entry in Part 6 was found by running,
   not reading.
