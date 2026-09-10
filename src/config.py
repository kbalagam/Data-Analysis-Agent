"""
Centralized configuration for the data analysis agent. Every constant
used across the pipeline lives here — model name, retry limits, chart
path, safe-execution allowlist. One source of truth, same reasoning as
the SEC Filing RAG project's src/config.py (prevents the class of bug
where a threshold is defined inconsistently in multiple places).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Gemini model ---
# NOTE: model names/versions move fast — if you see a 404 NOT_FOUND
# error, check the error message itself, it usually names the current
# model to switch to.
MODEL = os.environ.get("AGENT_MODEL", "gemini-3.6-flash")

# --- Agent behavior ---
MAX_CODE_RETRIES = 3  # cost control — each retry is a full model generation,
                       # not a lightweight API call (see README for reasoning)
CHART_PATH = "chart_output.png"  # fixed, code-controlled — never model-decided

# --- Retry behavior for transient API errors ---
MAX_API_RETRIES = 3
RETRYABLE_STATUS_CODES = (429, 500, 503)
