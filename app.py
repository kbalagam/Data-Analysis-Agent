"""
Streamlit UI for the Data Analysis Agent — lets you toggle between the
hand-built retry loop (agent.py) and the LangGraph state-machine version
(agent_langgraph.py), so the comparison documented in the README is
directly visible, not just something you have to take on faith.
"""

import os
import streamlit as st
import pandas as pd

from src.llm_client import get_client
from src.agent import answer_question
from src.agent_langgraph import answer_question_langgraph
from src.config import CHART_PATH

st.set_page_config(page_title="Data Analysis Agent", page_icon="📊")
st.title("📊 Data Analysis Agent")
st.caption("Ask a natural-language question about a CSV — the agent writes and safely executes pandas code to answer it.")


@st.cache_resource
def load_client():
    return get_client()


@st.cache_data
def load_default_df():
    default_path = "datasets/titanic.csv"
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    return None


uploaded_file = st.file_uploader("Upload a CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.caption(f"Using uploaded file: {uploaded_file.name}")
else:
    df = load_default_df()
    if df is not None:
        st.caption("No file uploaded — using the built-in Titanic dataset (datasets/titanic.csv).")

if df is not None:
    with st.expander("Preview data"):
        st.dataframe(df.head())

    implementation = st.radio(
        "Implementation",
        ["Hand-built (Python for-loop retry)", "LangGraph (state-machine retry)"],
        horizontal=True,
    )

    question = st.text_input("Your question", placeholder="e.g. What was the average age of passengers?")

    if st.button("Ask", type="primary") and question:
        with st.spinner("Generating and running code..."):
            if implementation.startswith("Hand-built"):
                client = load_client()
                outcome = answer_question(client, question, df)
                success = outcome["error"] is None
                answer = outcome["answer"]
                attempts_used = sum(
                    1 for entry in outcome["token_log"]
                    if entry["step"].startswith("generate_code_attempt_")
                )
                chart_created = outcome["chart_path"] is not None
                error = outcome["error"]
                code = outcome["code"]
            else:
                final_state = answer_question_langgraph(question, df)
                success = final_state["error"] is None
                answer = final_state["answer"]
                attempts_used = final_state["attempt"]
                chart_created = final_state["chart_created"]
                error = final_state["error"]
                code = final_state["code"]

        st.session_state["result"] = {
            "success": success,
            "answer": answer,
            "attempts_used": attempts_used,
            "chart_created": chart_created,
            "error": error,
            "code": code,
        }

    if "result" in st.session_state:
        r = st.session_state["result"]
        left, right = st.columns([2, 3])

        with left:
            st.subheader("Answer" if r["success"] else "Failed")
            if r["success"]:
                st.write(r["answer"])
            else:
                st.error(f"Could not produce a result after {r['attempts_used']} attempt(s).")
                st.code(r["error"], language="text")
            st.caption(f"Attempts used: {r['attempts_used']}")
            if r["chart_created"] and os.path.exists(CHART_PATH):
                st.image(CHART_PATH)

        with right:
            st.subheader("Generated code")
            st.code(r["code"] or "(no code generated)", language="python", wrap_lines=True)
else:
    st.info("Upload a CSV to get started, or add datasets/titanic.csv to the project to use the default.")