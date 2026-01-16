from langchain_core.documents import Document


def split_text(
    transcript_chunks,
    max_window_seconds=45,
    max_chars=1200
):
    """
    Groups transcript into time-based semantic windows.
    Each chunk represents a coherent idea span.
    """

    documents = []

    buffer_text = []
    window_start = None
    window_end = None
    char_count = 0

    for chunk in transcript_chunks:
        text = chunk["text"]
        start = chunk["start"]
        end = start + chunk["duration"]

        if window_start is None:
            window_start = start

        buffer_text.append(text)
        char_count += len(text)
        window_end = end

        # Flush window if it grows too large
        if (
            (window_end - window_start) >= max_window_seconds
            or char_count >= max_chars
        ):
            documents.append(
                Document(
                    page_content=" ".join(buffer_text),
                    metadata={
                        "start_time": window_start,
                        "end_time": window_end
                    }
                )
            )

            # Reset window
            buffer_text = []
            char_count = 0
            window_start = None
            window_end = None

    # Flush leftover
    if buffer_text:
        documents.append(
            Document(
                page_content=" ".join(buffer_text),
                metadata={
                    "start_time": window_start,
                    "end_time": window_end
                }
            )
        )

    return documents
