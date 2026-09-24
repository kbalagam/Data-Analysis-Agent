"""
Tests for the LangGraph retry-routing logic — verifies the loop/stop
decision independently of live model behavior (the LLM can write
defensive code that avoids errors, so testing the router directly, with
fabricated states, is the reliable way to confirm the retry mechanism
itself is correct).
"""

from langgraph.graph import END
from src.agent_langgraph import route_after_execution


def test_routes_to_generate_code_when_failed_with_attempts_remaining():
    state = {"error": "some fake error", "attempt": 1, "max_retries": 3}
    assert route_after_execution(state) == "generate_code"


def test_routes_to_end_when_failed_and_out_of_attempts():
    state = {"error": "some fake error", "attempt": 3, "max_retries": 3}
    assert route_after_execution(state) == END


def test_routes_to_explain_result_when_succeeded():
    state = {"error": None, "attempt": 1, "max_retries": 3}
    assert route_after_execution(state) == "explain_result"


def test_does_not_loop_one_attempt_too_many():
    """Regression test for the off-by-one we reasoned through earlier:
    using `<=` instead of `<` would allow one extra generate_code call
    past max_retries."""
    state = {"error": "still failing", "attempt": 2, "max_retries": 3}
    assert route_after_execution(state) == "generate_code"  # 2 < 3, one more try allowed

    state_at_cap = {"error": "still failing", "attempt": 3, "max_retries": 3}
    assert route_after_execution(state_at_cap) == END  # 3 is not < 3, must stop