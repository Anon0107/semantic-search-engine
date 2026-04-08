# Semantic Search Engine

Building semantic search from scratch — embeddings, vector databases, and RAG using Voyage AI, ChromaDB, and Claude.

## Progress
- **Day 1 (Apr 5):** Embeddings fundamentals — converting text to vectors with Voyage AI, cosine similarity implemented from scratch and with numpy
- **Day 2 (Apr 6):** Vector database with ChromaDB Cloud — stored 200 HackerNews titles, queried by semantic similarity using Voyage AI embeddings, compared semantic vs keyword search
- **Day 3 (Apr 7):** RAG pipeline from scratch — chunked and ingested `notes.txt`, queried ChromaDB for top-3 relevant chunks, injected into Claude prompt for answers
- **Day 4 (Apr 8):** Advanced RAG — sentence-boundary chunking, section-aware splitting via `---` (custom) separators, query rewriting with Claude, source citation in responses

## RAG Findings (Day 4)
- **Sentence-boundary chunking preserves meaning at chunk edges** — chunks now always end at a full stop, eliminating mid-sentence splits
- **Section separators prevent context bleed** — content from different topics no longer gets grouped into the same chunk
- **Query rewriting improves retrieval accuracy** — Claude rephrases the user's query before embedding, reducing vocabulary mismatch between question and document
- **Specific queries retrieve accurately** — top-3 chunks consistently contain the right context, Claude answers correctly with chunk citations
- **Broad queries across the full dataset remain a limitation** — questions requiring aggregation across many chunks (e.g. "which characters are from X school") exceed the top-3 retrieval window; query decomposition planned for Day 5

## RAG Findings (Day 3)
- **Retrieval works correctly when data covers the topic** — Query returned the right chunks and Claude answered accurately with appropriate uncertainty flagging
- **Data gaps produce honest "I don't know" responses** — Query returned unrelated chunks because notes didn't cover it, Claude correctly refused to answer
- **Small dataset is the primary failure mode** — Sparse coverage means many queries hit irrelevant chunks regardless of embedding quality
- **Chunk boundaries affect answer completeness** — Word-based splitting cuts mid-sentence, causing some chunks to lose meaning at edges; sentence-boundary chunking would improve this

## Stack
- Voyage AI (`voyage-3`) — embeddings
- ChromaDB Cloud — vector database
- Anthropic (`claude-haiku-4-5-20251001`) — answer generation
- Python 3.11

## Setup
```bash
py -3.11 -m pip install voyageai chromadb anthropic python-dotenv numpy
```

Copy `.env.example` to `.env` and fill in your keys.
