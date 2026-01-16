import re
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(youtube_input: str) -> str:
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


def fetch_transcript(video_id: str):
    """
    Returns transcript as a list of dicts:
    [{text, start, duration}]
    """
    transcript = YouTubeTranscriptApi().fetch(video_id)

    return [
        {
            "text": snippet.text,
            "start": snippet.start,
            "duration": snippet.duration
        }
        for snippet in transcript
    ]
