from langchain_core.prompts import PromptTemplate

PROMPT = PromptTemplate.from_template("""
You are a precise assistant answering questions based ONLY on the provided context.

Rules:
- Use ONLY the information from the context.
- If not present, say: "The video does not discuss this topic."
- Be concise.

Context:
{context}

Question:
{question}

Answer:
""")


def answer_question(model, retriever, question: str):
    docs = retriever.invoke(question)

    if not docs:
        return "The video does not discuss this topic."

    context = "\n\n".join(doc.page_content for doc in docs)
    response = model.invoke(
        PROMPT.format(context=context, question=question)
    )
    return response.content, docs
