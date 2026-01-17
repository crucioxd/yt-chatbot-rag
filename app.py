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
    page_title="YouTube Q&A Chatbot",
    page_icon="🎥",
    layout="wide"
)

inject_css()

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "transcript_processed" not in st.session_state:
    st.session_state.transcript_processed = False

# ------------------ SIDEBAR ------------------
# This contains your URL input and the "Process" button
youtube_input, process = sidebar_ui()

# ------------------ PROCESS VIDEO logic ------------------
if process and youtube_input:
    with st.spinner("Processing video..."):
        try:
            video_id = extract_video_id(youtube_input)

            # 1. Load or Create Vector Store
            vectorstore, retriever = load_vectorstore(video_id)
            if vectorstore is None:
                st.info("New video detected. Fetching transcript...")
                transcript = fetch_transcript(video_id)
                chunks = split_text(
                    transcript, max_window_seconds=120, max_chars=4000)
                vectorstore, retriever = create_vectorstore(chunks, video_id)
                chunks_count = len(chunks)
                vectors_count = vectorstore._collection.count()
            else:
                chunks_count = "Cached"
                vectors_count = vectorstore._collection.count()

            model = load_llm()

            # 2. Update state and CLEAR old chat history for the new video
            st.session_state.update({
                "video_id": video_id,
                "retriever": retriever,
                "model": model,
                "chunks_count": chunks_count,
                "vectors_count": vectors_count,
                "transcript_processed": True,
                "messages": []  # Reset chat history for the new video
            })
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ------------------ MAIN UI ------------------
if not st.session_state.transcript_processed:
    # Landing State
    st.title("🎥 YouTube Q&A Assistant")
    st.info(
        "👈 Please enter a YouTube URL in the sidebar and click 'Process Video' to begin.")
else:
    # Chatting State
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.header("❓ Chat with the Video")

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask me anything about the video..."):
            # 1. Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append(
                {"role": "user", "content": prompt})

            # 2. Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Use our new memory-aware answer function
                    # We pass history EXCLUDING the current prompt to the helper
                    answer, docs, timestamps = answer_question(
                        st.session_state.model,
                        st.session_state.retriever,
                        prompt,
                        st.session_state.messages[:-1]
                    )
                    st.markdown(answer)

                    if timestamps:
                        with st.expander("⏱ Sources from Video"):
                            for t in timestamps[:3]:
                                link = f"https://www.youtube.com/watch?v={st.session_state.video_id}&t={t['seconds']}s"
                                st.markdown(f"- [{t['formatted']}]({link})")

            # 3. Save assistant response
            st.session_state.messages.append(
                {"role": "assistant", "content": answer})

    with right_col:
        st.header("🎬 Video Info")
        st.image(
            f"https://img.youtube.com/vi/{st.session_state.video_id}/0.jpg")

        with st.container(border=True):
            st.subheader("📊 Stats")
            st.metric("Chunks", st.session_state.chunks_count)
            st.metric("Vectors", st.session_state.vectors_count)

        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("<div style='text-align:center'>Built by Puneet</div>",
            unsafe_allow_html=True)
