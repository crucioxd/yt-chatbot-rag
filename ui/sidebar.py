import streamlit as st


def sidebar_ui():
    with st.sidebar:
        st.header("📺 Video Input")

        youtube_input = st.text_input(
            "Enter YouTube URL or Video ID",
            placeholder="https://youtube.com/watch?v=..."
        )

        process = st.button(
            "🚀 Process Video",
            type="primary",
            use_container_width=True
        )

        st.markdown("---")
        st.header("📊 Status")

        if st.session_state.get("transcript_processed"):
            st.success("Transcript Ready")
        else:
            st.warning("Waiting for input")

        st.markdown("---")
        st.markdown("""
        **How to use**
        1. Paste YouTube URL / ID  
        2. Click Process  
        3. Ask questions  
        """)

        return youtube_input, process
