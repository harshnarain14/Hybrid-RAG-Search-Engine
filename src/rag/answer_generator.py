from typing import List
import streamlit as st
from groq import Groq


def get_groq_client() -> Groq:
    """
    Create Groq client using API key.
    - Streamlit Cloud: st.secrets
    - Local dev: .env fallback
    """

    groq_key = None

    # ✅ Streamlit Cloud (PRIMARY)
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]

    # ✅ Local fallback (.env)
    if not groq_key:
        try:
            from dotenv import dotenv_values
            env = dotenv_values(".env")
            groq_key = env.get("GROQ_API_KEY")
        except Exception:
            pass

    if not groq_key:
        raise RuntimeError(
            "❌ GROQ_API_KEY not found. "
            "Add it to Streamlit Secrets or local .env file."
        )

    return Groq(api_key=groq_key)


def generate_answer(
    question: str,
    context: str,
    sources: List[str],
    model: str = "llama-3.3-70b-versatile",
) -> str:
    """
    Generate a grounded answer using Groq LLM and RAG context.
    """

    # ----------------------------
    # Safety checks
    # ----------------------------
    if not question or not question.strip():
        return "⚠️ Please ask a valid question."

    if not context or not context.strip():
        return (
            "⚠️ I could not find relevant information "
            "in the selected sources to answer this question."
        )

    # Clamp context to avoid Groq 400 errors
    MAX_CONTEXT_CHARS = 12000
    context = context[:MAX_CONTEXT_CHARS]

    client = get_groq_client()

    system_prompt = (
        "You are a precise research assistant.\n"
        "Answer ONLY using the provided context.\n"
        "If the answer is not present in the context, say so clearly.\n"
        "Do not hallucinate.\n"
        "Always be concise and factual."
    )

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "ANSWER:"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
    except Exception as e:
        return (
            "⚠️ The language model could not process this request.\n\n"
            f"Reason: {type(e).__name__}"
        )

    message_content = response.choices[0].message.content

    if not message_content:
        answer_text = "⚠️ No answer could be generated from the provided context."
    else:
        answer_text = message_content.strip()

    if sources:
        citations = "\n".join(f"- {s}" for s in sources)
        answer_text += "\n\nSources:\n" + citations

    return answer_text
