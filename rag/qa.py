from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# ---------------- PROMPTS ----------------

# 1. CONDENSE QUESTION PROMPT
# Used to convert "What about the price?" into "What is the price of the iPhone 15?" based on history.
CONDENSE_QUESTION_PROMPT = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
If the follow up question is already standalone, return it as is.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone Question:
"""

# 2. ANSWER PROMPT (Updated with History)
LOCAL_QA_PROMPT = """
You are a helpful AI Assistant discussing a video.

Conversation History:
{chat_history}

Context (Video Transcripts):
{context}

User Question: {question}

Instructions:
1. Answer the question based ONLY on the provided Context.
2. If the user refers to previous messages (e.g., "it", "that"), use the Conversation History to understand what they mean.
3. If the answer is not in the context, politely say you couldn't find it in the video.
4. Keep the answer conversational but factual.

Answer:
"""

SUMMARY_PROMPT = """
You are summarizing a long-form video transcript.

Rules:
- Base your answer ONLY on the provided excerpts.
- Identify the MAIN THEMES discussed in the video.
- Write clear, structured key takeaways.

Excerpts:
{context}

Task:
Write the key takeaways from the video.
"""


# ---------------- HELPERS ----------------

def is_summary_question(question: str) -> bool:
    keywords = ["summary", "summarize",
                "key takeaways", "overview", "main points"]
    q = question.lower()
    return any(k in q for k in keywords)


def format_chat_history(messages):
    """
    Converts list of dicts/messages into a string for the prompt.
    Limit to last 3 exchanges to save tokens.
    """
    formatted_history = []
    # Slice to get only the last 6 messages (3 turns)
    recent_messages = messages[-6:]

    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_history.append(f"{role}: {msg['content']}")

    return "\n".join(formatted_history)


# ---------------- CORE LOGIC ----------------

def answer_question(model, retriever, question: str, chat_history: list = None):
    """
    Main entry point.
    chat_history: List of {"role": "user/assistant", "content": "..."}
    """

    if chat_history is None:
        chat_history = []

    # 1. Handle Global Summaries (History usually not needed for a pure summary request)
    if is_summary_question(question):
        return _global_summary(model, retriever)

    # 2. Contextualize the Question (The "Memory" Step)
    # If we have history, we must check if the user is asking a follow-up question.
    standalone_question = question

    if chat_history:
        history_str = format_chat_history(chat_history)
        condense_prompt = CONDENSE_QUESTION_PROMPT.format(
            chat_history=history_str,
            question=question
        )
        # Ask LLM to rephrase
        response = model.invoke(condense_prompt)
        standalone_question = response.content.strip()
        print(f"🔄 Rephrased Question: {standalone_question}")  # Debugging log

    # 3. Retrieve & Answer
    return _local_qa(model, retriever, standalone_question, question, chat_history)


# ---------------- LOCAL QA ----------------

def _local_qa(model, retriever, standalone_question, original_question, chat_history_list):

    # Use the REPHRASED question to search the vector DB
    docs = retriever.invoke(standalone_question)

    if not docs:
        return "I couldn't find relevant information in the video for that specific question.", [], []

    # Format Context
    context_list = []
    for doc in docs:
        start = doc.metadata.get('start_time', 0)
        formatted_time = f"[{int(start//60)}:{int(start%60):02d}]"
        context_list.append(f"{formatted_time} {doc.page_content}")

    context_str = "\n\n".join(context_list)
    history_str = format_chat_history(chat_history_list)

    # Generate Answer
    prompt = LOCAL_QA_PROMPT.format(
        chat_history=history_str,
        context=context_str,
        # We pass original Q for tone, but context is from standalone Q
        question=original_question
    )

    response = model.invoke(prompt)
    timestamps = _extract_timestamps(docs)

    return response.content, docs, timestamps


# ---------------- GLOBAL SUMMARY ----------------
# (Kept mostly the same, just ensured consistent return signature)

def _global_summary(model, retriever):
    queries = ["introduction and agenda", "main conclusion", "key arguments"]
    all_docs = []
    for q in queries:
        docs = retriever.vectorstore.similarity_search(q, k=5)
        all_docs.extend(docs)

    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    unique_docs.sort(key=lambda x: x.metadata.get('start_time', 0))

    context = "\n\n".join(
        f"[Time: {int(d.metadata.get('start_time', 0)//60)}:{int(d.metadata.get('start_time', 0)%60):02d}] {d.page_content}"
        for d in unique_docs
    )

    prompt = SUMMARY_PROMPT.format(context=context)
    response = model.invoke(prompt)
    timestamps = _extract_timestamps(unique_docs)

    return response.content, unique_docs, timestamps


# ---------------- TIMESTAMP UTILS ----------------

def _extract_timestamps(docs):
    timestamps = []
    for doc in docs:
        start_time = doc.metadata.get("start_time")
        if start_time is not None:
            timestamps.append({
                "seconds": int(start_time),
                "formatted": f"{int(start_time//60):02d}:{int(start_time%60):02d}"
            })

    unique = {t["seconds"]: t for t in timestamps}
    return sorted(unique.values(), key=lambda x: x["seconds"])
