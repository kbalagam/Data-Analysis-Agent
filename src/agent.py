"""
Core data analysis agent logic: question -> generated pandas code ->
safe execution -> retry-with-error-feedback on failure -> honest
natural-language answer.

Chain-of-thought (verified with real evidence, see README): activated
only from retry attempt 2 onward, as a deliberate cost/benefit decision.
Tested on a simple question first and found NO measurable correctness
difference on easy cases (both prompts produced the same working code)
— so it isn't worth paying its extra token cost on every single call.
But verified on a genuine failure case (a nonexistent 'customers'
column): attempt 2 with chain-of-thought produced meaningfully more
defensive code (an explicit existence check with a documented fallback
chain) compared to the earlier non-chain-of-thought retry behavior
(a silent, undocumented substitution). Real evidence, not assumed.
"""

import re
import pandas as pd

from src.config import CHART_PATH, MAX_CODE_RETRIES
from src.llm_client import call_with_retry
from src.sandbox import run_code_safely

SYSTEM_PROMPT = (
    "You are a data analysis assistant. Given a pandas DataFrame called `df` "
    "and a question, write Python pandas code to answer it.\n"
    "Rules:\n"
    "- Assign your final answer to a variable named `result`.\n"
    "- Only use pandas (as `pd`) and the dataframe `df`. Do not import anything.\n"
    f"- If the question asks for or would benefit from a chart/plot/graph, "
    f"use matplotlib (available as `plt`) to create it, then save it with "
    f"exactly this line: plt.savefig('{CHART_PATH}'). Still assign a short "
    f"text summary to `result` even when producing a chart.\n"
    "- Return ONLY the code, no explanation, no markdown formatting, no ```python fences.\n"
    "- Keep the code simple and direct."
)

# Used starting from retry attempt 2 onward — see module docstring for
# the reasoning and verification behind this specific design.
CHAIN_OF_THOUGHT_SYSTEM_PROMPT = (
    "You are a data analysis assistant. Given a pandas DataFrame called `df` "
    "and a question, write Python pandas code to answer it.\n"
    "Rules:\n"
    "- Before writing the code, think through the problem step-by-step as "
    "CODE COMMENTS at the top (e.g. '# Step 1: filter for first class'). "
    "Break down exactly what needs to happen, in order.\n"
    "- Then write the actual code implementing those steps.\n"
    "- Assign your final answer to a variable named `result`.\n"
    "- Only use pandas (as `pd`) and the dataframe `df`. Do not import anything.\n"
    f"- If the question asks for or would benefit from a chart/plot/graph, "
    f"use matplotlib (available as `plt`) to create it, then save it with "
    f"exactly this line: plt.savefig('{CHART_PATH}'). Still assign a short "
    f"text summary to `result` even when producing a chart.\n"
    "- Return ONLY code and code comments — no prose explanation outside "
    "comments, no markdown formatting, no ```python fences."
)


def extract_code(response_text: str) -> str:
    """Models don't always follow formatting instructions perfectly —
    strip ```python fences if present, just in case."""
    text = response_text.strip()
    text = re.sub(r"^```(?:python)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def generate_code(client, question: str, df: pd.DataFrame,
                   previous_error: str = None, previous_code: str = None,
                   attempt_number: int = 1) -> tuple[str, dict]:
    columns_info = f"DataFrame columns: {list(df.columns)}\nDataFrame dtypes:\n{df.dtypes}"

    if previous_error:
        prompt = (
            f"{columns_info}\n\n"
            f"Question: {question}\n\n"
            f"Your previous code:\n{previous_code}\n\n"
            f"That code failed with this error:\n{previous_error}\n\n"
            f"Fix the code and try again."
        )
    else:
        prompt = f"{columns_info}\n\nQuestion: {question}"

    system_prompt = CHAIN_OF_THOUGHT_SYSTEM_PROMPT if attempt_number > 1 else SYSTEM_PROMPT
    response = call_with_retry(client, prompt, system_instruction=system_prompt)

    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
    }
    return extract_code(response.text), usage


def explain_result(client, question: str, code: str, result, chart_created: bool) -> tuple[str, dict]:
    chart_note = "A chart was also saved as part of answering this." if chart_created else ""
    prompt = (
        f"Question: {question}\n\n"
        f"Code that was run:\n{code}\n\n"
        f"Result: {result}\n"
        f"{chart_note}\n\n"
        f"Write a short, clear natural-language answer to the question. "
        f"If the code made any assumption or substitution (e.g. used a "
        f"different column than what was literally asked for, because "
        f"the exact one didn't exist), explicitly say so — don't let the "
        f"answer imply something it didn't actually verify."
    )
    response = call_with_retry(client, prompt)
    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
    }
    return response.text, usage


def answer_question(client, question: str, df: pd.DataFrame,
                     skip_explanation: bool = False) -> dict:
    """Full agent loop: generate code, execute safely, retry with error
    feedback on failure, up to MAX_CODE_RETRIES times.

    skip_explanation: when True, skips explain_result() — saves one full
    model generation per question. Used by eval_agent.py, which only
    checks the raw numeric result, never the natural-language
    explanation."""
    import os
    if os.path.exists(CHART_PATH):
        os.remove(CHART_PATH)

    previous_error = None
    previous_code = None
    token_log = []

    for attempt in range(1, MAX_CODE_RETRIES + 1):
        code, usage = generate_code(client, question, df, previous_error, previous_code, attempt_number=attempt)
        token_log.append({"step": f"generate_code_attempt_{attempt}", **usage})

        result = run_code_safely(code, df)

        previous_code = code
        previous_error = result.get("error")

        if not previous_error:
            chart_created = os.path.exists(CHART_PATH)
            if skip_explanation:
                answer = None
            else:
                answer, usage = explain_result(client, question, code, result.get("result"), chart_created)
                token_log.append({"step": "explain_result", **usage})
            return {
                **result,
                "answer": answer,
                "chart_path": CHART_PATH if chart_created else None,
                "token_log": token_log,
            }

        print(f"Code failed on attempt {attempt}:")
        print(previous_error)

    return {
        "success": False,
        "result": None,
        "error": previous_error,
        "answer": None,
        "chart_path": None,
        "token_log": token_log,
    }


def load_dataframe(csv_path: str) -> pd.DataFrame:
    """Kept separate from answer_question() deliberately — that function
    shouldn't need to know or care whether the data came from a CSV, a
    database, or anywhere else."""
    return pd.read_csv(csv_path)
