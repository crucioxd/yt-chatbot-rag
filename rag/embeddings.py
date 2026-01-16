# rag/embeddings.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import EMBEDDING_MODEL, TOP_K


PERSIST_DIR = "./chroma_db"


def load_vectorstore(video_id: str):
    """
    Load an existing Chroma vectorstore if it exists.
    Returns (vectorstore, retriever) or (None, None)
    """

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    collection_name = f"youtube_{video_id}"

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=collection_name
    )

    if vectorstore._collection.count() == 0:
        return None, None

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    return vectorstore, retriever


def create_vectorstore(chunks, video_id: str):
    """
    Create a new vectorstore (used only if video is NOT indexed).
    """

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    collection_name = f"youtube_{video_id}"

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=collection_name
    )

    vectorstore.add_documents(chunks)
    vectorstore.persist()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    return vectorstore, retriever
