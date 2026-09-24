"""
Usage:
    python3 scripts/run_langgraph_demo.py datasets/titanic.csv "What was the average age of passengers?"
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from src.agent_langgraph import answer_question_langgraph

if __name__ == "__main__":
    csv_path = sys.argv[1]
    question = sys.argv[2]

    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path}: {len(df)} rows\n")
    print(f"Question: {question}\n")

    final_state = answer_question_langgraph(question, df)

    print(f"Attempts used: {final_state['attempt']}")
    print(f"Success: {final_state['error'] is None}")
    if final_state["error"]:
        print(f"Final error: {final_state['error']}")
    print(f"\nAnswer: {final_state['answer']}")