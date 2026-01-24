from typing import List
import streamlit as st
from groq import Groq


def get_groq_client() -> Groq:
    """
    Load Groq API key from Streamlit Secrets (cloud)
    or from .env (local fallback).
    """
    groq_key = None

    # Streamlit Cloud
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]

    # Local fallback
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


def generate_answer(
    question: str,
    context: str,
    sources: List[str],
    model: str = "llama-3.1-70b-versatile",
) -> str:
    client = get_groq_client()

    system_prompt = (
        "You are a precise research assistant.\n"
        "Answer ONLY using the provided context.\n"
        "If the answer is not present in the context, say so clearly.\n"
        "Do not hallucinate."
    )

    user_prompt = f"""
QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )

    content = response.choices[0].message.content
    answer = content.strip() if content else "⚠️ No answer generated."

    if sources:
        answer += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

    return answer
