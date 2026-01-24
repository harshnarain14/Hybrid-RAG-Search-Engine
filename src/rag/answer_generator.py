import streamlit as st
from groq import Groq


def get_groq_client() -> Groq:
    """
    Get Groq client.
    Supports both local (.env) and Streamlit Cloud (st.secrets).
    """

    groq_key = None

    # Streamlit Cloud
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]

    # Local fallback (.env)
    if not groq_key:
        try:
            from dotenv import dotenv_values
            env = dotenv_values(".env")
            groq_key = env.get("GROQ_API_KEY")
        except Exception:
            pass

    if not groq_key:
        raise RuntimeError("❌ GROQ_API_KEY not found in Streamlit Secrets or .env")

    return Groq(api_key=groq_key)
