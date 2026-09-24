# Data Analysis Agent

Ask natural-language questions about any CSV file and get answers grounded in real, safely-executed pandas code — built from scratch, including the sandboxing, retry logic, and evaluation.

## What this is

An agent that:
1. Takes a CSV file and a natural-language question
2. Writes pandas code to answer it, using the model's understanding of the data's actual columns/types
3. Executes that code in a restricted sandbox — not full, unrestricted Python
4. If the code fails, feeds the real error back to the model and lets it self-correct (up to 3 attempts)
5. Explains the result in plain language, explicitly disclosing any assumptions or substitutions it made
6. Optionally produces a chart, when the question calls for one

## Why this project

Letting an LLM write and run arbitrary code is a genuinely different risk category from simple text generation or RAG — the model isn't just retrieving/summarizing anymore, it's producing something that gets *executed*. This project is a deliberate exploration of what that requires: safe execution, error-driven self-correction, and an evaluation methodology that checks actual correctness, not just plausibility.

## Architecture

```
CSV file
    │
    ▼
load_dataframe()          — pd.read_csv()
    │
    ▼
answer_question()         — the main orchestration loop
    │
    ├──► generate_code()  — LLM writes pandas code, given real column/dtype info
    │         │
    │         ▼
    ├──► run_code_safely() — executes in a restricted sandbox (see below)
    │         │
    │    ┌────┴────┐
    │  failed     succeeded
    │    │            │
    │    ▼            ▼
    │  (feed error   explain_result() — LLM writes a natural-language
    │   back, retry   answer, disclosing any assumptions made
    │   up to 3x)         │
    │                     ▼
    └───────────────► final answer + optional chart
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install google-genai pandas matplotlib
export GEMINI_API_KEY=your_key_here
```

Run against any CSV:
```bash
python3 data_agent.py your_data.csv "your question here"
```

Run without a CSV (uses built-in sample data):
```bash
python3 data_agent.py
```

Run the eval suite (against the Titanic dataset, or any CSV with matching questions):
```bash
python3 eval_agent.py datasets/titanic.csv
```

## Sandboxing: the core safety mechanism

The model generates arbitrary Python as text; `run_code_safely()` executes it in a restricted environment built on an **allowlist** principle, not a blocklist:

```python
exec(code, {"__builtins__": {}, **SAFE_BUILTINS, "pd": pd, "df": df, "plt": plt})
```

Stripping `__builtins__` entirely removes *all* built-in functions and the `import` machinery in one move — verified experimentally: `import os` fails with `'__import__ not found'`, `open()` fails with `'open' is not defined'`. Only a small, explicit set of safe builtins (`len`, `round`, `str`, etc.) plus `pandas`, the dataframe, and `matplotlib` are added back.

**Honest limitation:** this is a real, commonly-used technique, but not a bulletproof security boundary — sophisticated attacks can sometimes escape Python-level restrictions via object introspection. Production systems handling genuinely untrusted users typically use OS-level process/container isolation, not just `exec()` restrictions. This level of sandboxing is appropriate for demonstrating security awareness in a portfolio project; it is not a substitute for real process isolation at scale.

## The retry loop: verified with a real failure, not staged

Asked the agent "What is the average of the column customers?" against a dataset with no `customers` column:
- **Attempt 1 failed for real:** `KeyError: 'customers'`
- The actual error was fed back to the model
- **Attempt 2 succeeded**, substituting `units_sold` as a reasonable proxy
- `explain_result()` correctly disclosed the substitution rather than silently presenting a number as if it directly answered the literal question

This loop is capped at 3 attempts — a deliberate cost control, since each retry is a full model generation, not a lightweight API call. An unbounded retry loop is a real cost/reliability risk in agentic systems.

## Evaluation

`eval_agent.py` checks the agent against 5 questions with expected answers computed independently (directly in pandas, not via the agent) against the Titanic dataset, using `math.isclose()` for floating-point-safe comparison. **Result: 5/5 passed.**

This is a small eval set, not a comprehensive one — a production version would include more edge cases (ambiguous questions, multi-step questions, questions with no valid answer in the data) and track pass rate over time as the prompt/logic evolves.

## LangGraph Comparison

To deepen my understanding of agentic control flow, I rebuilt the
generate→execute retry loop using LangGraph — a graph-based framework
built for exactly this kind of stateful, looping logic, unlike LangChain's
linear `|` chaining (LCEL), which can't express "try again if this fails."

```bash
python3 scripts/run_langgraph_demo.py datasets/titanic.csv "What was the average age of passengers?"
```

Enable LangSmith tracing by setting `LANGSMITH_TRACING=true` and
`LANGSMITH_API_KEY` in `.env`.

**What I found comparing the two:**
- The hand-built version's retry loop is a plain Python `for` loop with
  manual state tracking (`previous_code`, `previous_error`). LangGraph
  makes that same control flow explicit as a graph — nodes
  (`generate_code`, `execute_code`, `explain_result`) and a router
  function that decides whether to loop back or move on, based on shared
  state. For a loop this simple, the hand-built for-loop is arguably
  easier to read; LangGraph's value shows up more clearly in graphs with
  multiple branches or parallel paths, not a single linear retry.
- Hit a real, undocumented issue: this version of `langchain-google-genai`
  returns `response.content` as a list of content blocks (not a plain
  string) when the model does internal reasoning — verified via direct
  inspection (`type()` and `repr()`) rather than guessing, then handled
  with a small adapter function.
- Verified the retry/stop routing logic with direct unit tests against
  fabricated states (`tests/test_agent_langgraph.py`), rather than relying
  on the LLM happening to fail on a live run — the model often writes
  defensive code that avoids errors entirely, so testing the router
  function in isolation was the only reliable way to confirm the loop
  and stop conditions are actually correct.

## Known limitations

- **Occasional code-syntax leakage into natural-language answers.** `explain_result()` has, in testing, occasionally echoed Python assignment syntax (e.g., starting an answer with `result = "..."`) instead of pure prose. The underlying information was still correct; this is a cosmetic prompt-following issue, not a correctness bug.
- **Structured output was explored but not merged into the main pipeline.** `structured_output_demo.py` demonstrates using a JSON schema (`response_schema`) to get guaranteed-valid output instead of the regex-based markdown-fence-stripping (`extract_code`) used in `data_agent.py`. Verified working in isolation, but not integrated — the current regex approach is proven reliable (5/5 eval pass rate) and replacing it would require re-verifying the full eval suite. A reasonable next step, not done here due to time/quota constraints during development.
- **No multi-step reasoning.** Complex questions requiring multiple chained operations (e.g., "compare X and Y, then plot the difference") are handled as a single code-generation pass — the model may or may not correctly chain the logic within one block of code. A more robust version would break multi-part questions into explicit sequential steps (the ReAct pattern).
- **No cost cap beyond retry count.** Token usage is tracked per request (`token_log`) but there's no hard dollar/token budget enforcement — only the retry count limit indirectly bounds cost.
- **Table/data-shape assumptions.** The agent works well on tabular CSVs with clear column semantics. It has not been tested against deeply nested or unusually structured data.

## What I'd improve with more time

- Integrate structured output into the main pipeline, replacing regex-based parsing
- Add a hard per-request cost cap using the token tracking already in place
- Expand the eval set with ambiguous and multi-step questions
- Add basic prompt-injection awareness — testing what happens if the CSV data itself contains adversarial text
- Build a small Streamlit UI, matching the RAG project's presentation

## Tech stack

- **LLM:** Google Gemini (`gemini-3.6-flash`)
- **Execution sandbox:** Python `exec()` with a stripped, explicitly-allowlisted `__builtins__`
- **Data:** pandas
- **Charts:** matplotlib (non-interactive `Agg` backend)
- **Evaluation:** custom eval script with `math.isclose()` tolerance-based comparison
