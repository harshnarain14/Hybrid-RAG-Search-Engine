from typing import List
from dotenv import dotenv_values
from groq import Groq


def get_groq_client() -> Groq:
    """
    Create Groq client using API key from local .env file.
    """

    env = dotenv_values(".env")
    groq_key = env.get("GROQ_API_KEY")

    if not groq_key:
        raise RuntimeError("❌ GROQ_API_KEY missing in .env file")

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

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
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
