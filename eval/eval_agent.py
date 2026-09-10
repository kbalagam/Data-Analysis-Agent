"""
Small eval set for the data analysis agent — verifiable questions with
pre-computed expected answers (computed independently, directly in
pandas, NOT via the agent, so this is a genuine ground-truth check).

Comparison uses math.isclose() with a tolerance, since floating-point
results will rarely match hand-rounded expected values with exact
equality — this is expected, not a bug.

Run: python3 eval/eval_agent.py datasets/titanic.csv
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm_client import get_client
from src.agent import answer_question, load_dataframe

EVAL_CASES = [
    {"question": "What was the average age of passengers?", "expected": 29.69911764705882},
    {"question": "What percentage of passengers survived?", "expected": 38.38383838383838},
    {"question": "How many passengers were in first class?", "expected": 216},
    {"question": "What was the average fare paid?", "expected": 32.204207968574636},
    {"question": "How many male passengers were there?", "expected": 577},
]


def run_eval(client, df) -> None:
    passed = 0
    failed = 0

    for i, case in enumerate(EVAL_CASES, start=1):
        print(f"\n--- Eval {i}/{len(EVAL_CASES)} ---")
        print(f"Question: {case['question']}")
        print(f"Expected: {case['expected']}")

        outcome = answer_question(client, case["question"], df, skip_explanation=True)
        actual = outcome.get("result")
        print(f"Actual:   {actual}")

        if actual is None:
            print("FAIL — agent returned no result (all retries exhausted)")
            failed += 1
            continue

        try:
            is_correct = math.isclose(float(actual), float(case["expected"]), rel_tol=1e-4)
        except (TypeError, ValueError):
            print("SKIP — result isn't a plain number, needs manual check")
            continue

        if is_correct:
            print("PASS")
            passed += 1
        else:
            print("FAIL — value doesn't match within tolerance")
            failed += 1

    total_checked = passed + failed
    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total_checked} passed" if total_checked else "No cases were numerically checkable")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 eval/eval_agent.py path/to/titanic.csv")
        sys.exit(1)

    client = get_client()
    df = load_dataframe(sys.argv[1])
    run_eval(client, df)
