"""
Structured output demo: comparing the fragile free-text + regex approach
(what generate_code() currently does) against guaranteed-schema JSON.

This is a standalone demo, not yet wired into data_agent.py — the goal
is to see the mechanism work in isolation first, same approach as every
other new concept this week (function calling, sandboxing) before
integrating it into the real agent.

Run: GEMINI_API_KEY=your_key python3 structured_output_demo.py
"""

import os
import sys
import json
from google import genai
from google.genai import types

MODEL = "gemini-3.6-flash"


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable first.")
        sys.exit(1)
    return genai.Client(api_key=api_key)


# The schema, built piece by piece and verified correct:
response_schema = types.Schema(
    type="OBJECT",
    properties={
        "code": types.Schema(type="STRING", description="The pandas code that answers the question"),
        "explanation": types.Schema(type="STRING", description="A one-sentence explanation of what the code does"),
    },
    required=["code"],
)


def generate_code_structured(client: genai.Client, question: str, columns: list) -> dict:
    """Same task as generate_code() in data_agent.py, but using
    guaranteed structured output instead of free-text + regex parsing."""
    prompt = (
        f"DataFrame columns: {columns}\n"
        f"Question: {question}\n\n"
        f"Write pandas code (using `df` and `pd`) to answer this question. "
        f"Assign the final answer to a variable named `result`."
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )

    # No regex, no markdown-stripping needed — response.text is
    # GUARANTEED valid JSON matching the schema.
    parsed = json.loads(response.text)
    return parsed


if __name__ == "__main__":
    client = get_client()

    columns = ["region", "revenue", "units_sold"]
    question = "What is the total revenue?"

    print(f"Question: {question}")
    result = generate_code_structured(client, question, columns)

    print("\n--- Structured response (guaranteed valid JSON, no parsing tricks) ---")
    print(f"Code:\n{result['code']}")
    print(f"\nExplanation: {result.get('explanation', '(none provided)')}")
