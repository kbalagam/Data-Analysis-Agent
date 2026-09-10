"""
Command-line entry point for the data analysis agent.

Run against a real CSV:
    python3 scripts/cli.py path/to/data.csv "your question"

Run with no CSV (uses built-in sample data, for a quick smoke test):
    python3 scripts/cli.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.llm_client import get_client
from src.agent import answer_question, load_dataframe


if __name__ == "__main__":
    client = get_client()

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        question = sys.argv[2] if len(sys.argv) > 2 else None
        df = load_dataframe(csv_path)
        print(f"Loaded {csv_path}: {len(df)} rows, columns: {list(df.columns)}\n")

        if not question:
            print("No question provided. Usage: python3 scripts/cli.py data.csv \"your question\"")
            sys.exit(0)
    else:
        df = pd.DataFrame({
            "region": ["East", "West", "East", "South", "West"],
            "revenue": [1000, 1500, 800, 1200, 2000],
            "units_sold": [10, 15, 8, 12, 20],
        })
        question = "What is the total revenue across all regions?"
        print("No CSV provided — using built-in test data.")
        print(df)
        print()

    print(f"Question: {question}")
    outcome = answer_question(client, question, df)
    print(f"\nRaw result: {outcome.get('result')}")
    print(f"Answer: {outcome.get('answer')}")
    if outcome.get("chart_path"):
        print(f"Chart saved to: {outcome['chart_path']}")

    print("\n--- Token usage log ---")
    total_input = sum(entry["input_tokens"] for entry in outcome.get("token_log", []))
    total_output = sum(entry["output_tokens"] for entry in outcome.get("token_log", []))
    for entry in outcome.get("token_log", []):
        print(f"  {entry['step']}: {entry['input_tokens']} in / {entry['output_tokens']} out")
    print(f"  TOTAL: {total_input} in / {total_output} out ({total_input + total_output} tokens)")
