import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state
if 'transcript_processed' not in st.session_state:
    st.session_state.transcript_processed = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'video_id' not in st.session_state:
    st.session_state.video_id = None
if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = None


def extract_video_id(youtube_input: str) -> str:
    """
    Accepts a YouTube URL or video ID and returns the video ID.
    """
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", youtube_input):
        return youtube_input

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


# Custom CSS to style the answer
st.markdown("""
<style>
    .answer-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
        color: #262730;
        font-size: 16px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.set_page_config(
    page_title="YouTube Transcript Q&A",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube Q&A Assistant")
st.markdown("---")

# Sidebar for video input and instructions
with st.sidebar:
    st.header("📺 Video Input")

    youtube_input = st.text_input(
        "Enter YouTube URL or Video ID:",
        placeholder="https://www.youtube.com/watch?v=... or Video ID",
        key="youtube_input"
    )

    process_button = st.button(
        "🚀 Process Video Transcript", type="primary", use_container_width=True)

    st.markdown("---")
    st.header("📋 How to use:")
    st.markdown("""
    1. Paste a YouTube URL or Video ID
    2. Click 'Process Video Transcript'
    3. Wait for processing to complete
    4. Ask questions about the video in the main area
    """)

    # # Show processing status in sidebar
    # st.markdown("---")
    # st.header("📊 Processing Status")

    # if st.session_state.transcript_processed:
    #     st.success("✅ Transcript Processed")
    #     if st.session_state.chunks:
    #         st.info(f"📊 Created {len(st.session_state.chunks)} text chunks")
    #     if st.session_state.vectorstore:
    #         st.info(
    #             f"🔍 Vector store has {st.session_state.vectorstore.index.ntotal} vectors")
    # else:
    #     st.warning("⏳ Waiting for video input")

# Main content area - Show different content based on processing state
if not st.session_state.transcript_processed:
    # Initial state - show processing form and instructions
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Process a YouTube Video")
        st.markdown("""
        Enter a YouTube URL or Video ID in the sidebar and click **"Process Video Transcript"** to begin.
        
        Once processed, you can ask questions about the video content.
        """)

        # Show example URL
        with st.expander("📌 Example YouTube URLs", expanded=False):
            st.markdown("""
            - https://www.youtube.com/watch?v=dQw4w9WgXcQ
            """)

    with col2:
        st.header("🎯 Features")
        st.markdown("""
        - 📄 **Transcript Extraction**: Get full video transcripts
        - 🔍 **Semantic Search**: Find relevant content using AI embeddings
        - ❓ **Q&A Assistant**: Ask questions about the video content
        - 💬 **Context-Aware**: Answers are based only on video content
        """)

else:
    # After processing - show Q&A section and video info
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("❓ Ask Questions About the Video")

        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="Ask anything about the video content...",
            key="question_input"
        )

        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            ask_button = st.button(
                "🔍 Get Answer", type="primary", use_container_width=True)

        with col_btn2:
            if st.button("🔄 Process New Video", use_container_width=True):
                # Reset session state for new video
                for key in ['transcript_processed', 'vectorstore', 'retriever', 'model', 'chunks', 'video_id', 'transcript_text']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        if ask_button and question:
            with st.spinner("Searching for answer..."):
                try:
                    # Retrieve relevant documents
                    retrieved_docs = st.session_state.retriever.invoke(
                        question)

                    if not retrieved_docs:
                        st.warning(
                            "⚠️ No relevant context found in the video.")
                    else:
                        # Prepare prompt
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

                        context_text = "\n\n".join(
                            doc.page_content for doc in retrieved_docs)
                        final_prompt = prompt.format(
                            context=context_text,
                            question=question
                        )

                        # Get answer
                        answer = st.session_state.model.invoke(final_prompt)

                        # Display answer with clean styling
                        st.markdown("---")
                        st.subheader("💡 Answer")

                        # Use custom container for the answer
                        st.markdown(
                            f'<div class="answer-container">{answer.content}</div>', unsafe_allow_html=True)

                        # Show retrieved context (optional)
                        with st.expander("🔍 View Retrieved Context", expanded=False):
                            for i, doc in enumerate(retrieved_docs):
                                st.markdown(f"**Chunk {i+1}:**")
                                st.text(
                                    doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                st.markdown("---")

                except Exception as e:
                    st.error(f"❌ Error getting answer: {e}")

    with col2:
        st.header("🎬 Video Information")

        # Show video thumbnail
        if st.session_state.video_id:
            st.image(f"http://img.youtube.com/vi/{st.session_state.video_id}/0.jpg",

                     )


# Process video when button is clicked
if process_button and youtube_input:
    # Reset processing flag
    st.session_state.transcript_processed = False

    with st.spinner("Processing video transcript..."):
        try:
            # Step 1: Extract video ID and get transcript
            video_id = extract_video_id(youtube_input)
            st.session_state.video_id = video_id

            # Get transcript
            yt_api = YouTubeTranscriptApi()
            transcript_list = yt_api.fetch(video_id)
            transcript_text = " ".join(
                [snippet.text for snippet in transcript_list])
            st.session_state.transcript_text = transcript_text

            # Step 2: Text splitting
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )

            chunks = splitter.create_documents([transcript_text])
            st.session_state.chunks = chunks

            # Step 3: Embedding & Vector Store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.vectorstore = vectorstore

            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            st.session_state.retriever = retriever

            # Step 4: Initialize model
            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                temperature=0.2,
                max_new_tokens=300
            )

            model = ChatHuggingFace(llm=llm)
            st.session_state.model = model

            st.session_state.transcript_processed = True

            # Show success message and rerun to update UI
            st.success("✅ Video processing completed successfully!")
            st.rerun()

        except TranscriptsDisabled:
            st.error("❌ Transcripts are disabled for this video.")
            st.session_state.transcript_processed = False
        except ValueError as e:
            st.error(f"❌ {e}")
            st.session_state.transcript_processed = False
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            st.session_state.transcript_processed = False

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built by Puneet</p>
</div>
""", unsafe_allow_html=True)
