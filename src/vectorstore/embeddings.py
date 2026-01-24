from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Local embedding model (no API key, no quota).
    """

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
