from langchain_core.prompts import PromptTemplate


# ---------------- PROMPTS ----------------

LOCAL_QA_PROMPT = """
You are a precise assistant answering questions based ONLY on the provided context.

Rules:
- Use ONLY the information from the context.
- If the answer is not present, say:
  "The video does not discuss this topic."
- Adjust the length of your answer based on the question:
  - Short and direct for factual questions
  - More detailed and explanatory for "why", "how", or "explain" questions
- Do NOT add external knowledge.

Context:
{context}

Question:
{question}

Answer:
"""


SUMMARY_PROMPT = """
You are summarizing a long-form video transcript.

Rules:
- Base your answer ONLY on the provided excerpts.
- Identify the MAIN THEMES discussed in the video.
- Write clear, structured key takeaways.
- Do NOT invent topics that are not present.

Excerpts:
{context}

Task:
Write the key takeaways from the video.
"""


# ---------------- HELPERS ----------------

def is_summary_question(question: str) -> bool:
    keywords = [
        "summary",
        "summarize",
        "key takeaways",
        "overview",
        "what is this video about",
        "main points"
    ]
    q = question.lower()
    return any(k in q for k in keywords)


def is_deep_question(question: str) -> bool:
    keywords = [
        "explain",
        "why",
        "how",
        "in detail",
        "elaborate",
        "intuition",
        "significance"
    ]
    q = question.lower()
    return any(k in q for k in keywords)


# ---------------- CORE LOGIC ----------------

def answer_question(model, retriever, question: str):
    """
    Routes question to either:
    - Local semantic RAG
    - Global hierarchical summarization
    """

    if is_summary_question(question):
        return _global_summary(model, retriever)

    return _local_qa(model, retriever, question)


# ---------------- LOCAL QA ----------------

def _local_qa(model, retriever, question: str):
    if is_deep_question(question):
        docs = retriever.vectorstore.similarity_search(question, k=8)
    else:
        docs = retriever.invoke(question)

    if not docs:
        return "The video does not discuss this topic.", [], []

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = LOCAL_QA_PROMPT.format(context=context, question=question)
    response = model.invoke(prompt)

    timestamps = _extract_timestamps(docs)

    return response.content, docs, timestamps


# ---------------- GLOBAL SUMMARY ----------------

def _global_summary(model, retriever):
    """
    Hierarchical summarization:
    - Pull representative chunks across the entire video
    - Summarize them into key takeaways
    """

    # 1️⃣ Pull globally representative chunks (NOT question-based)
    docs = retriever.vectorstore.similarity_search(
        "", k=15
    )

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = SUMMARY_PROMPT.format(context=context)

    response = model.invoke(prompt)

    timestamps = _extract_timestamps(docs)

    return response.content, docs, timestamps


# ---------------- TIMESTAMP UTILS ----------------

def _extract_timestamps(docs):
    timestamps = []

    for doc in docs:
        start_time = doc.metadata.get("start_time")
        if start_time is not None:
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)

            timestamps.append({
                "seconds": int(start_time),
                "formatted": f"{minutes:02d}:{seconds:02d}"
            })

    # Deduplicate by seconds
    unique = {}
    for t in timestamps:
        unique[t["seconds"]] = t

    return sorted(unique.values(), key=lambda x: x["seconds"])
