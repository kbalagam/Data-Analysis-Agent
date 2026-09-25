"""
Single source of truth for talking to the Gemini API — client setup and
retry-with-backoff logic. Same reasoning as the SEC Filing RAG project's
llm_client.py: one copy of call_with_retry, used everywhere, rather than
duplicated per call site.

Accepts a system_instruction parameter (rather than hardcoding one) so
different callers (direct code generation vs. chain-of-thought retry
generation vs. result explanation) can each supply their own prompt
without needing separate retry implementations.
"""

import os
import sys
import time

from google import genai
from google.genai import errors, types

from src.config import MODEL, MAX_API_RETRIES, RETRYABLE_STATUS_CODES


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            api_key = None
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable first.")
        print("  export GEMINI_API_KEY=your_key_here  (or put it in .env)")
        sys.exit(1)
    return genai.Client(api_key=api_key)


def call_with_retry(client: genai.Client, contents, system_instruction: str = None,
                     max_retries: int = MAX_API_RETRIES):
    """Retry wrapper around generate_content. Returns the FULL response
    object (not just .text) — callers extract what they need (text,
    usage_metadata, etc.)."""
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system_instruction),
            )
        except errors.APIError as e:
            if getattr(e, "code", None) in RETRYABLE_STATUS_CODES and attempt < max_retries:
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
