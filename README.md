# Semantic Search Engine

Building semantic search from scratch — embeddings, vector databases, and RAG using Voyage AI and ChromaDB.

## Progress
- **Day 1 (Apr 5):** Embeddings fundamentals — converting text to vectors with Voyage AI, cosine similarity implemented from scratch and with numpy
- **Day 2 (Apr 6):** Vector database with ChromaDB Cloud — stored 200 HackerNews titles, queried by semantic similarity using Voyage AI embeddings, compared semantic vs keyword search

## Stack
- Voyage AI (`voyage-3`) — embeddings
- ChromaDB Cloud — vector database
- Python 3.11

## Setup
```bash
py -3.11 -m pip install voyageai chromadb python-dotenv numpy
```

Copy `.env.example` to `.env` and fill in your keys.
