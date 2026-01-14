import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

import faiss
from dotenv import load_dotenv
# 1) here we extracted the youtube video id from url or id input and fetched the transcript
load_dotenv()


def extract_video_id(youtube_input: str) -> str:
    """
    Accepts a YouTube URL or video ID and returns the video ID.
    """
    # If it's already an ID (11 chars, no special symbols)
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", youtube_input):
        return youtube_input

    # Match common YouTube URL patterns
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, youtube_input)
        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL or Video ID")


user_input = input("Enter youtube video url or id: ")

try:
    video_id = extract_video_id(user_input)
    yt_api = YouTubeTranscriptApi()
    # fetch function is used to get the transcript
    transcript_list = yt_api.fetch(video_id)
    # print(transcript_list)

    transcript_text = " ".join(
        [snippet.text for snippet in transcript_list])
    # print(transcript_text)

except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")


# 2) INDEXING  - TEXT SPLITTING

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.create_documents([transcript_text])
print(len(chunks))


# 3) EMBEDDING & VECTOR STORE CREATION

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

print(vectorstore.index.ntotal)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
print(retriever)


# step 4) augmentation

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.2,
    max_new_tokens=300
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate.from_template("""
You are a precise assistant answering questions based ONLY on the provided context.

Rules:
- Use ONLY the information from the context.
- If the answer is NOT present in the context, say:
  "The video does not discuss this topic."
- Do NOT add external knowledge.
- Be concise and factual.

Context:
{context}

Question:
{question}

Answer:
""")

while True:
    question = input(
        "\nAsk a question about the video (or type 'exit'): ").strip()

    if question.lower() == "exit":
        break

    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        print("No relevant context found.")
        continue

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = prompt.format(
        context=context_text,
        question=question
    )

    answer = model.invoke(final_prompt)

    print("\nAnswer:")
    print(answer.content)
