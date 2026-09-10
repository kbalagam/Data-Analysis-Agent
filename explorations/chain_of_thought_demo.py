"""
Chain-of-thought prompting demo: asking the model to reason step-by-step
via code comments, BEFORE deciding whether to merge this into the real
agent's generate_code(). Kept isolated so we can tell whether any
change in output quality comes from THIS change specifically, not
mixed in with other changes (e.g. structured output).

Comparison: same question, sent once with the ORIGINAL prompt (direct
code, no reasoning) and once with the CHAIN-OF-THOUGHT prompt (reasoning
as comments), so you can see the actual difference side by side.

Run: GEMINI_API_KEY=your_key python3 chain_of_thought_demo.py
"""

import os
import sys
from google import genai
from google.genai import types

MODEL = "gemini-3.6-flash"


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable first.")
        sys.exit(1)
    return genai.Client(api_key=api_key)


ORIGINAL_PROMPT_INSTRUCTION = (
    "You are a data analysis assistant. Given a pandas DataFrame called `df` "
    "and a question, write Python pandas code to answer it.\n"
    "Rules:\n"
    "- Assign your final answer to a variable named `result`.\n"
    "- Only use pandas (as `pd`) and the dataframe `df`. Do not import anything.\n"
    "- Return ONLY the code, no explanation, no markdown formatting, no ```python fences.\n"
    "- Keep the code simple and direct."
)

CHAIN_OF_THOUGHT_INSTRUCTION = (
    "You are a data analysis assistant. Given a pandas DataFrame called `df` "
    "and a question, write Python pandas code to answer it.\n"
    "Rules:\n"
    "- Before writing the code, think through the problem step-by-step as "
    "CODE COMMENTS at the top (e.g. '# Step 1: filter for first class', "
    "'# Step 2: filter for age under 30', '# Step 3: compute survival rate'). "
    "Break down exactly what needs to happen, in order.\n"
    "- Then write the actual code implementing those steps.\n"
    "- Assign your final answer to a variable named `result`.\n"
    "- Only use pandas (as `pd`) and the dataframe `df`. Do not import anything.\n"
    "- Return ONLY code and code comments — no prose explanation outside of "
    "comments, no markdown formatting, no ```python fences.\n"
)


def ask(client: genai.Client, system_instruction: str, question: str, columns: list) -> str:
    prompt = f"DataFrame columns: {columns}\nQuestion: {question}"
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    return response.text


if __name__ == "__main__":
    client = get_client()

    # A genuinely multi-step question, chosen because you correctly
    # identified it as a case where reasoning first plausibly helps.
    columns = ["PassengerId", "Survived", "Pclass", "Age", "Sex", "Fare"]
    question = "What percentage of first-class passengers under 30 survived?"

    print(f"Question: {question}\n")

    print("=" * 70)
    print("ORIGINAL PROMPT (direct code, no reasoning)")
    print("=" * 70)
    print(ask(client, ORIGINAL_PROMPT_INSTRUCTION, question, columns))

    print("\n" + "=" * 70)
    print("CHAIN-OF-THOUGHT PROMPT (reasoning as comments first)")
    print("=" * 70)
    print(ask(client, CHAIN_OF_THOUGHT_INSTRUCTION, question, columns))
