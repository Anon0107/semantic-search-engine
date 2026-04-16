# Semantic Search Engine

Building semantic search from scratch — embeddings, vector databases, and RAG using Voyage AI, ChromaDB, and Claude.

## Progress
- **Day 1 (Apr 5):** Embeddings fundamentals — converting text to vectors with Voyage AI, cosine similarity implemented from scratch and with numpy
- **Day 2 (Apr 6):** Vector database with ChromaDB Cloud — stored 200 HackerNews titles, queried by semantic similarity using Voyage AI embeddings, compared semantic vs keyword search
- **Day 3 (Apr 7):** RAG pipeline from scratch — chunked and ingested `notes.txt`, queried ChromaDB for top-3 relevant chunks, injected into Claude prompt for answers
- **Day 4 (Apr 8):** Advanced RAG — sentence-boundary chunking, section-aware splitting via `---` separators, query rewriting with Claude, source citation in responses
- **Day 5 (Apr 9):** Document Q&A chatbot — combined RAG pipeline with CLI chatbot, sentence-boundary chunking, conversation history with auto-compression at 20k tokens, Q&A pairs stored back into ChromaDB for multi-turn memory

## RAG Findings (Day 5)
- **Multi-turn memory works for simple recall** — conversation history preserves prior answers and Claude references them for direct follow-up questions
- **Multi-hop inference across turns is unreliable** — questions requiring Claude to join information from two separate prior answers fail because prior context sits in history as conversational text, not structured retrieval context
- **Storing Q&A pairs in ChromaDB improves multi-hop reasoning** — embedding prior answers and retrieving them alongside document chunks puts historical context in the privileged `<context>` block where Claude reasons over it more reliably
- **Broad queries remain a limitation** — top-3 retrieval window insufficient for aggregation questions; query decomposition planned

## RAG Findings (Day 4)
- **Sentence-boundary chunking preserves meaning at chunk edges** — chunks now always end at a full stop, eliminating mid-sentence splits
- **Section separators prevent context bleed** — content from different topics no longer gets grouped into the same chunk
- **Query rewriting improves retrieval accuracy** — Claude rephrases the user's query before embedding, reducing vocabulary mismatch between question and document
- **Specific queries retrieve accurately** — top-3 chunks consistently contain the right context, Claude answers correctly with chunk citations
- **Broad queries across the full dataset remain a limitation** — questions requiring aggregation across many chunks exceed the top-3 retrieval window; query decomposition planned

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
py -3.11 -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys.