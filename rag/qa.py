from langchain_core.prompts import PromptTemplate


# ---------------- PROMPTS ----------------

LOCAL_QA_PROMPT = """
You are an expert AI Assistant analyzing a video transcript.
You are provided with several distinct chunks of the transcript (Context).

Context:
{context}

User Question: {question}

Instructions:
1. Answer the question comprehensively using ONLY the provided context.
2. If the context contains the answer, explain it in detail.
3. If the context mentions different viewpoints, summarize all of them.
4. Cite the specific moments (e.g., [10:23]) if timestamps are available in the text.
5. If the answer is NOT in the context, strictly say: "I analyzed the available transcript segments, but they do not contain the answer to this specific question."

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

# Update _local_qa to use the new TOP_K from config implicitly or explicitly
def _local_qa(model, retriever, question: str):
    # Simply use the retriever which now uses the higher TOP_K from config
    docs = retriever.invoke(question)

    if not docs:
        return "The video does not discuss this topic.", [], []

    # Inject timestamps into the text fed to the LLM so it can cite them
    context_list = []
    for doc in docs:
        start = doc.metadata.get('start_time', 0)
        formatted_time = f"[{int(start//60)}:{int(start%60):02d}]"
        context_list.append(f"{formatted_time} {doc.page_content}")

    context = "\n\n".join(context_list)

    prompt = LOCAL_QA_PROMPT.format(context=context, question=question)
    response = model.invoke(prompt)

    timestamps = _extract_timestamps(docs)

    return response.content, docs, timestamps


# ---------------- GLOBAL SUMMARY ----------------

def _global_summary(model, retriever):
    """
    Refined Strategy: 
    Instead of searching for "", we search for key structural terms 
    to get the beginning, middle, and end, or main topics.
    """

    # Strategy: specific queries to get a spread of content
    queries = ["introduction and agenda",
               "main conclusion and takeaways", "key arguments and debate"]

    all_docs = []
    for q in queries:
        # Fetch 5 chunks for each query aspect
        docs = retriever.vectorstore.similarity_search(q, k=5)
        all_docs.extend(docs)

    # Remove duplicates
    seen = set()
    unique_docs = []
    for doc in all_docs:
        # Assuming content is the unique identifier
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    # Sort by time so the LLM reads a chronological story
    unique_docs.sort(key=lambda x: x.metadata.get('start_time', 0))

    context = "\n\n".join(
        f"[Time: {int(d.metadata.get('start_time', 0)//60)}:{int(d.metadata.get('start_time', 0)%60):02d}] {d.page_content}"
        for d in unique_docs
    )

    prompt = f"""
    You are an expert content summarizer. Below are key segments from a video transcript, sorted chronologically.

    Transcript Segments:
    {context}

    Task:
    Provide a detailed, bullet-point summary of the video. 
    1. Start with a 1-sentence "Hook" describing the video.
    2. Provide 3-5 "Key Takeaways" with detailed explanations.
    3. End with a "Conclusion".
    
    Summary:
    """

    response = model.invoke(prompt)
    timestamps = _extract_timestamps(unique_docs)

    return response.content, unique_docs, timestamps

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
