"""
Safe(r) code execution for the data analysis agent.

Core principle: ALLOWLIST, not blocklist. We strip __builtins__ entirely,
then explicitly add back only what's needed: a small set of safe
builtins, pandas, the dataframe, and matplotlib for optional chart
generation.

HONEST LIMITATION: this is a real, commonly-used technique but NOT a
bulletproof security boundary. Sophisticated attacks can sometimes
escape Python-level restrictions via object introspection. True
production-grade isolation uses OS-level process/container separation,
not just exec() restrictions. Appropriate for demonstrating security
awareness in a portfolio project, not a substitute for real process
isolation at scale handling untrusted users.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves files, no GUI window
import matplotlib.pyplot as plt

SAFE_BUILTINS = {
    "len": len,
    "list": list,
    "dict": dict,
    "range": range,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "print": print,
    "str": str,
    "int": int,
    "float": float,
}


def run_code_safely(code: str, df: pd.DataFrame) -> dict:
    """Execute model-generated pandas (+ optional matplotlib) code in a
    restricted environment. Returns {"success", "result", "error"}
    rather than raising, so the calling agent loop can decide what to
    do next (e.g. feed the error back for a retry) without a
    try/except at every call site."""
    safe_globals = {
        "__builtins__": {},
        **SAFE_BUILTINS,
        "pd": pd,
        "df": df,
        "plt": plt,
    }
    local_vars = {}

    try:
        exec(code, safe_globals, local_vars)
        result = local_vars.get("result", None)
        return {"success": True, "result": result, "error": None}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}
    finally:
        plt.close("all")  # always clean up figures, even on failure
