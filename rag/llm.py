from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from config import LLM_REPO_ID, TEMPERATURE, MAX_NEW_TOKENS


def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS
    )
    return ChatHuggingFace(llm=llm)
