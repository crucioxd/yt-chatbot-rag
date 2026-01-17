# config.py

# CHANGE 1: Use a massive, modern embedding model (Open Source & Free)
EMBEDDING_MODEL = "BAAI/bge-m3"
# OR use "nomic-ai/nomic-embed-text-v1.5" (Great for long context)

LLM_REPO_ID = "meta-llama/Llama-3.1-70B-Instruct"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# CHANGE 2: Feed the beast. Llama 3.1 has a huge context window.
# Don't be shy. Give it 20-30 chunks.
TOP_K = 25

# CHANGE 3: Lower temperature for factual answers
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 1024  # Allow it to write longer answers
