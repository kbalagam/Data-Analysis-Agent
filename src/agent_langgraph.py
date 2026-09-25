"""
LangGraph version of the retry-loop agent — direct comparison against the
hand-built for-loop retry logic in agent.py's answer_question().

Same SYSTEM_PROMPT / CHAIN_OF_THOUGHT_SYSTEM_PROMPT wording, same
run_code_safely() sandbox (pure Python, no LLM involved, so reused as-is —
LangChain has no opinion about sandboxing).
"""

import os
import re
from typing import TypedDict, Optional, Any

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from src.config import MODEL, CHART_PATH
from src.sandbox import run_code_safely

SYSTEM_PROMPT = (
    "You are a data analysis assistant. Given a pandas DataFrame called `df` "
    "and a question, write Python pandas code to answer it.\n"
    "Rules:\n"
    "- Assign your final answer to a variable named `result`.\n"
    "- Only use pandas (as `pd`) and the dataframe `df`. Do not import anything.\n"
    f"- If the question asks for or would benefit from a chart/plot/graph, "
    f"use matplotlib (available as `plt`) to create it, then save it with "
    f"exactly this line: plt.savefig('{CHART_PATH}'). Still assign a short "
    f"text summary to `result` even when producing a chart.\n"
    "- Return ONLY the code, no explanation, no markdown formatting, no ```python fences.\n"
    "- Keep the code simple and direct."
)

CHAIN_OF_THOUGHT_SYSTEM_PROMPT = (
    "You are a data analysis assistant. Given a pandas DataFrame called `df` "
    "and a question, write Python pandas code to answer it.\n"
    "Rules:\n"
    "- Before writing the code, think through the problem step-by-step as "
    "CODE COMMENTS at the top (e.g. '# Step 1: filter for first class'). "
    "Break down exactly what needs to happen, in order.\n"
    "- Then write the actual code implementing those steps.\n"
    "- Assign your final answer to a variable named `result`.\n"
    "- Only use pandas (as `pd`) and the dataframe `df`. Do not import anything.\n"
    f"- If the question asks for or would benefit from a chart/plot/graph, "
    f"use matplotlib (available as `plt`) to create it, then save it with "
    f"exactly this line: plt.savefig('{CHART_PATH}'). Still assign a short "
    f"text summary to `result` even when producing a chart.\n"
    "- Return ONLY code and code comments — no prose explanation outside "
    "comments, no markdown formatting, no ```python fences."
)


def extract_code(response_text: str) -> str:
    text = response_text.strip()
    text = re.sub(r"^```(?:python)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()

def get_text_from_content(content) -> str:
    """response.content can be a plain string, or a list of content
    blocks (e.g. when the model does internal reasoning). Pull out just
    the text parts either way."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


class AgentState(TypedDict):
    question: str
    df: pd.DataFrame
    code: Optional[str]
    error: Optional[str]
    result: Any
    chart_created: bool
    answer: Optional[str]
    attempt: int
    max_retries: int


def _llm() -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            api_key = None
    return ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=api_key,
    )


def generate_code_node(state: AgentState) -> dict:
    new_attempt = state["attempt"] + 1
    system_prompt = CHAIN_OF_THOUGHT_SYSTEM_PROMPT if new_attempt > 1 else SYSTEM_PROMPT

    if new_attempt == 1:
        human_msg = (
            f"Question: {state['question']}\n"
            f"DataFrame columns: {list(state['df'].columns)}"
        )
    else:
        human_msg = (
            f"Question: {state['question']}\n"
            f"DataFrame columns: {list(state['df'].columns)}\n"
            f"Your previous code:\n{state['code']}\n"
            f"It failed with this error:\n{state['error']}\n"
            f"Fix the code."
        )

    response = _llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg),
    ])


    return {"code": extract_code(get_text_from_content(response.content)), "attempt": new_attempt}


def execute_code_node(state: AgentState) -> dict:
    outcome = run_code_safely(state["code"], state["df"])
    return {
        "error": outcome["error"],
        "result": outcome["result"],
        "chart_created": os.path.exists(CHART_PATH),
    }


def explain_result_node(state: AgentState) -> dict:
    prompt = (
        f"Question: {state['question']}\n"
        f"Code that was run:\n{state['code']}\n"
        f"Result: {state['result']}\n"
        f"Chart created: {state['chart_created']}\n\n"
        f"Explain this result in plain, non-technical language, in 2-3 sentences."
    )
    response = _llm().invoke([HumanMessage(content=prompt)])
    return {"answer": get_text_from_content(response.content)}


def route_after_execution(state: AgentState) -> str:
    if state["error"] is None:
        return "explain_result"
    if state["attempt"] < state["max_retries"]:
        return "generate_code"
    return END  # out of retries — give up, same as hand-built fallback


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("generate_code", generate_code_node)
    graph.add_node("execute_code", execute_code_node)
    graph.add_node("explain_result", explain_result_node)

    graph.set_entry_point("generate_code")
    graph.add_edge("generate_code", "execute_code")
    graph.add_conditional_edges(
        "execute_code",
        route_after_execution,
        {"generate_code": "generate_code", "explain_result": "explain_result", END: END},
    )
    graph.add_edge("explain_result", END)

    return graph.compile()


def answer_question_langgraph(question: str, df: pd.DataFrame, max_retries: int = 3) -> dict:
    if os.path.exists(CHART_PATH):
        os.remove(CHART_PATH)

    compiled = build_graph()
    final_state = compiled.invoke({
        "question": question,
        "df": df,
        "code": None,
        "error": None,
        "result": None,
        "chart_created": False,
        "answer": None,
        "attempt": 0,
        "max_retries": max_retries,
    })
    return final_state