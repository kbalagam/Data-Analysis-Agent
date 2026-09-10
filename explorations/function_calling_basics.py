"""
Function calling fundamentals — one fake tool, no framework, just the
raw request/response loop.

Reminder of the actual sequence (the model NEVER executes anything —
it only ever describes what it wants; your code does all real work):
  1. You send a prompt + tool definitions to the model
  2. The model's response may contain a "function_call" part instead of text
  3. YOUR code checks for that, and if present, actually runs the real function
  4. YOUR code sends the real result back to the model
  5. The model reads the result and produces the final natural-language answer

Run: GEMINI_API_KEY=your_key python3 function_calling_basics.py
"""

import os
import sys
import time
from google import genai
from google.genai import errors, types

MODEL = "gemini-3.6-flash"


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable first.")
        sys.exit(1)
    return genai.Client(api_key=api_key)


# --- Step 1: the REAL functions that will actually run ---
def get_weather(city: str) -> str:
    """A fake weather lookup — in reality this would call a real API."""
    fake_data = {
        "Boston": "58°F, cloudy",
        "Tokyo": "72°F, sunny",
        "London": "50°F, rainy",
    }
    return fake_data.get(city, f"No weather data available for {city}")


def get_stock_price(ticker: str) -> str:
    """A fake stock price lookup."""
    fake_prices = {
        "AAPL": "$232.50",
        "TSLA": "$412.80",
        "MSFT": "$478.10",
    }
    return fake_prices.get(ticker, f"No price data available for {ticker}")


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """A fake currency converter (fixed fake exchange rate for simplicity)."""
    fake_rate = 0.92  # pretend USD -> EUR rate
    if from_currency == "USD" and to_currency == "EUR":
        return f"{amount * fake_rate:.2f} EUR"
    return f"Conversion from {from_currency} to {to_currency} not supported in this demo"


# --- Step 2: map tool NAME (string, matches function_declarations below)
# to the REAL function object. This is what lets us go from "the model
# said 'get_stock_price'" to "actually call the real get_stock_price
# function" without a big if/elif chain. ---
AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price,
    "convert_currency": convert_currency,
}


# --- Step 3: describe ALL functions to the model (it can't see your
# code, only these descriptions) ---
all_tools = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Get the current weather for a given city.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "city": types.Schema(type="STRING", description="The city name, e.g. Boston")
                },
                required=["city"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_stock_price",
            description="Get the current stock price for a given ticker symbol.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "ticker": types.Schema(type="STRING", description="Stock ticker, e.g. AAPL")
                },
                required=["ticker"],
            ),
        ),
        types.FunctionDeclaration(
            name="convert_currency",
            description="Convert an amount of money from one currency to another.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "amount": types.Schema(type="NUMBER", description="The amount to convert"),
                    "from_currency": types.Schema(type="STRING", description="Source currency code, e.g. USD"),
                    "to_currency": types.Schema(type="STRING", description="Target currency code, e.g. EUR"),
                },
                required=["amount", "from_currency", "to_currency"],
            ),
        ),
    ]
)


def call_with_retry(client: genai.Client, contents: list, max_retries: int = 3):
    """Retry wrapper around the raw generate_content call. Both call sites
    in ask_with_tool (the initial question, and the follow-up after a
    function result) route through this single function — same lesson
    as the Week 1 restructure: one shared retry implementation, not
    duplicated per call site."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(tools=[all_tools]),
            )
            return response
        except errors.APIError as e:
            if getattr(e, "code", None) in (429, 500, 503) and attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [retry {attempt}] API error {e.code}, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  [FAILED] Non-retryable error: {e}")
            raise
        except Exception as e:
            print(f"  [FAILED] Unexpected error: {e}")
            raise
    raise RuntimeError("Exceeded max retries")


def ask_with_tool(client: genai.Client, question: str) -> str:
    # First call: send the question + tool definitions to the model
    response = call_with_retry(client, question)

    for part in response.candidates[0].content.parts:
        if part.function_call:
            function_name = part.function_call.name
            function_args = part.function_call.args

            function_to_call = AVAILABLE_TOOLS.get(function_name)
            if function_to_call is None:
                print(f"Function not available: {function_name}")
                result = None
            else:
                result = function_to_call(**function_args)

            print(f"Model requested: {function_name}({function_args})")
            print(f"Real function returned: {result}")

            # Send the real result back so the model can produce a final answer
            function_response_part = types.Part.from_function_response(
                name=function_name,
                response={"result": result},
            )

            follow_up = call_with_retry(
                client,
                [
                    question,
                    response.candidates[0].content,
                    types.Content(role="user", parts=[function_response_part]),
                ],
            )
            print(f"\nFinal answer: {follow_up.text}")
        else:
            print(part.text)


if __name__ == "__main__":
    client = get_client()

    print("=== Test 1: weather (should call get_weather) ===")
    ask_with_tool(client, "What's the weather like in Boston right now?")

    print("\n=== Test 2: stock price (should call get_stock_price, NOT get_weather) ===")
    ask_with_tool(client, "What's the current price of AAPL stock?")

    print("\n=== Test 3: currency (should call convert_currency) ===")
    ask_with_tool(client, "Convert 100 USD to EUR.")

    print("\n=== Test 4: no tool needed (should just answer in plain text) ===")
    ask_with_tool(client, "What is 15 times 4?")