import streamlit as st
from dotenv import load_dotenv

from ui.styles import inject_css
from ui.sidebar import sidebar_ui
from utils.youtube import extract_video_id, fetch_transcript
from utils.text_processing import split_text
from rag.embeddings import create_vectorstore, load_vectorstore
from rag.llm import load_llm
from rag.qa import answer_question

# ------------------ CONFIG ------------------
load_dotenv()
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide"
)

inject_css()
st.title("🎥 YouTube Q&A Assistant")
st.markdown("---")

# ------------------ SIDEBAR ------------------
youtube_input, process = sidebar_ui()

# ------------------ PROCESS VIDEO ------------------
# ------------------ PROCESS VIDEO ------------------
if process and youtube_input:
    with st.spinner("Processing video..."):
        try:
            video_id = extract_video_id(youtube_input)

            # 1️⃣ Try loading existing vector DB (NO transcript fetch)
            vectorstore, retriever = load_vectorstore(video_id)

            if vectorstore is None:
                # 2️⃣ First time seeing this video → full pipeline
                st.info("New video detected. Fetching transcript...")

                transcript = fetch_transcript(video_id)
                chunks = split_text(transcript)

                vectorstore, retriever = create_vectorstore(chunks, video_id)

                chunks_count = len(chunks)
                vectors_count = vectorstore._collection.count()

                st.info("Video processed and stored in vector DB")

            else:
                # 3️⃣ Already processed → skip transcript + embedding
                st.info("⚡ Video already indexed. Loaded from vector DB")

                chunks_count = "Cached"
                vectors_count = vectorstore._collection.count()

            model = load_llm()

            st.session_state.update({
                "video_id": video_id,
                "retriever": retriever,
                "model": model,
                "chunks_count": chunks_count,
                "vectors_count": vectors_count,
                "transcript_processed": True
            })

            st.success("✅ Ready to answer questions!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {e}")


# ------------------ MAIN UI ------------------
if st.session_state.get("transcript_processed", False):

    left_col, right_col = st.columns([2, 1])

    # ========== LEFT COLUMN ==========
    with left_col:
        st.header("❓ Ask Questions About the Video")

        question = st.text_input(
            "Enter your question",
            placeholder="Ask anything based on the video content..."
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            ask = st.button("🔍 Get Answer", type="primary")

        with col2:
            if st.button("🔄 Process New Video"):
                st.session_state.clear()
                st.rerun()

        if ask and question:
            with st.spinner("Searching for answer..."):
                answer, docs = answer_question(
                    st.session_state.model,
                    st.session_state.retriever,
                    question
                )

                st.markdown("---")
                st.subheader("💡 Answer")
                st.markdown(
                    f"<div class='answer-container'>{answer}</div>",
                    unsafe_allow_html=True
                )

                with st.expander("🔍 View Retrieved Context"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.text(
                            doc.page_content[:400] + "..."
                            if len(doc.page_content) > 400
                            else doc.page_content
                        )
                        st.markdown("---")

    # ========== RIGHT COLUMN ==========
    with right_col:
        st.header("🎬 Video Information")

        st.image(
            f"https://img.youtube.com/vi/{st.session_state.video_id}/0.jpg",
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("📊 Processing Stats")

        st.success("Transcript Processed")
        st.metric("Chunks Created", st.session_state.chunks_count)
        st.metric("Vectors Stored", st.session_state.vectors_count)

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center'>Built by Puneet</div>",
    unsafe_allow_html=True
)
