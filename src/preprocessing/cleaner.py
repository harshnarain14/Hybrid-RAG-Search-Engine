import re


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text for embedding and retrieval.

    Steps:
    - Remove null bytes
    - Normalize whitespace
    - Remove excessive newlines
    - Strip leading/trailing spaces
    """

    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", " ")

    # Normalize newlines
    text = re.sub(r"\n{2,}", "\n", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()
