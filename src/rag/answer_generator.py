def generate_answer(
    question: str,
    context: str,
    sources: List[str],
    model: str = "llama-3.3-70b-versatile",
) -> str:
    # ----------------------------
    # Input validation
    # ----------------------------
    if not question or not question.strip():
        return "⚠️ Please ask a valid question."

    if not context or not context.strip():
        return (
            "⚠️ I could not find relevant information "
            "in the selected sources to answer this question."
        )

    # ----------------------------
    # Context safety (Groq limit)
    # ----------------------------
    MAX_CONTEXT_CHARS = 12000  # safe for Groq
    context = context[:MAX_CONTEXT_CHARS]

    client = get_groq_client()

    system_prompt = (
        "You are a precise research assistant.\n"
        "Answer ONLY using the provided context.\n"
        "If the answer is not present in the context, say so clearly.\n"
        "Do not hallucinate."
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

    content = response.choices[0].message.content
    answer = content.strip() if content else "⚠️ No answer generated."

    if sources:
        answer += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

    return answer
