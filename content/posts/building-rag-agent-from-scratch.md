---
title: "Building a RAG Agent from Scratch: Retrieval-Augmented Generation Tutorial"
date: 2026-03-06
description: "Learn how to build a RAG agent from scratch with Python, vector databases, and Claude API. Step-by-step guide with working code."
tags: ["RAG", "retrieval augmented generation", "AI agent", "vector database", "tutorial"]
categories: ["Tutorials"]
keywords: ["RAG agent tutorial", "retrieval augmented generation", "vector database", "AI agent memory", "embedding"]
draft: false
---

*This post contains affiliate links. I may earn a commission at no extra cost to you.*

Retrieval-Augmented Generation (RAG) is one of the most practical patterns for building AI agents that work with real-world data. Instead of relying solely on what a language model learned during training, RAG lets your agent pull in fresh, relevant information at query time. The result: fewer hallucinations, up-to-date answers, and responses grounded in your own data.

In this tutorial, you'll build a working RAG agent from scratch using Python. We'll use [ChromaDB](https://docs.trychroma.com/) as the vector store and the Anthropic Claude API for generation. By the end, you'll have a functional agent that can answer questions about any document corpus you feed it.

## What Is RAG and Why Does It Matter

A standard LLM has a knowledge cutoff — it only knows what it learned during training. If you ask it about your company's internal documentation, last week's news, or a private codebase, it has nothing to work with.

RAG solves this by adding a retrieval step before generation:

1. **Indexing phase**: Your documents are chunked, converted to vector embeddings, and stored in a vector database.
2. **Query phase**: When a user asks a question, the query is also embedded. Similar document chunks are retrieved from the vector DB. Those chunks are injected into the LLM's context as grounding material.
3. **Generation phase**: The LLM generates an answer based on both the retrieved context and the question.

The key insight is that you're not fine-tuning the model — you're giving it the right information at runtime. This is cheaper, faster to update, and often more accurate than fine-tuning for knowledge retrieval tasks.

## Prerequisites and Setup

You'll need Python 3.10+, an [Anthropic API key](https://console.anthropic.com/), and the following packages:

```bash
pip install anthropic chromadb sentence-transformers pypdf2
```

Set your API key:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

The full project structure we're building:

```
rag_agent/
├── ingest.py      # Load and index documents
├── retriever.py   # Query the vector store
├── agent.py       # RAG agent with Claude
└── data/          # Your source documents
```

## Step 1: Ingest and Embed Documents

The first step is chunking your documents and storing their embeddings. Chunk size is one of the most impactful hyperparameters in RAG — too small and you lose context, too large and you dilute relevance.

A good starting point for general text is 512 tokens with a 50-token overlap between chunks.

```python
# ingest.py
import os
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHUNK_SIZE = 512  # characters, not tokens (rough approximation)
CHUNK_OVERLAP = 64

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def ingest_directory(data_dir: str, collection_name: str = "documents") -> None:
    """Load all .txt files from data_dir and index them in ChromaDB."""
    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete existing collection to avoid duplicates on re-run
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    all_chunks = []
    all_ids = []
    all_metadata = []

    for file_path in Path(data_dir).glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{file_path.stem}_chunk_{i}")
            all_metadata.append({"source": file_path.name, "chunk_index": i})

    if not all_chunks:
        print("No documents found.")
        return

    embeddings = encoder.encode(all_chunks, show_progress_bar=True).tolist()

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metadata
    )

    print(f"Indexed {len(all_chunks)} chunks from {data_dir}")

if __name__ == "__main__":
    ingest_directory("./data")
```

Run it once to build your index:

```bash
python ingest.py
```

## Step 2: Build the Retriever

The retriever takes a user query, embeds it using the same model used during indexing, and returns the top-k most similar chunks.

```python
# retriever.py
import chromadb
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, collection_name: str = "documents", top_k: int = 5):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_collection(collection_name)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        """Return top-k relevant chunks for the query."""
        query_embedding = self.encoder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({
                "text": doc,
                "source": meta["source"],
                "relevance_score": 1 - dist  # cosine similarity
            })

        return chunks
```

One practical note: cosine distance and cosine similarity are complementary (similarity = 1 - distance for normalized vectors), so higher `relevance_score` means more relevant.

## Step 3: Assemble the RAG Agent

Now we connect retrieval to generation. The agent formats the retrieved chunks into a context block and passes it to Claude with the user's question.

```python
# agent.py
import anthropic
from retriever import Retriever

SYSTEM_PROMPT = """You are a helpful assistant. Answer questions using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make up information. Cite the source document when relevant."""

class RAGAgent:
    def __init__(self, top_k: int = 5, min_relevance: float = 0.3):
        self.retriever = Retriever(top_k=top_k)
        self.client = anthropic.Anthropic()
        self.min_relevance = min_relevance

    def _format_context(self, chunks: list[dict]) -> str:
        """Build the context block from retrieved chunks."""
        filtered = [c for c in chunks if c["relevance_score"] >= self.min_relevance]
        if not filtered:
            return "No relevant context found."

        lines = []
        for i, chunk in enumerate(filtered, 1):
            lines.append(f"[Source: {chunk['source']} | Relevance: {chunk['relevance_score']:.2f}]")
            lines.append(chunk["text"])
            lines.append("")

        return "\n".join(lines)

    def ask(self, question: str) -> str:
        """Retrieve context and generate an answer."""
        chunks = self.retriever.retrieve(question)
        context = self._format_context(chunks)

        user_message = f"""Context:
{context}

Question: {question}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )

        return response.content[0].text

if __name__ == "__main__":
    agent = RAGAgent()
    while True:
        question = input("\nYour question (or 'quit'): ").strip()
        if question.lower() == "quit":
            break
        answer = agent.ask(question)
        print(f"\nAnswer: {answer}")
```

A quick test with some documents in `./data/`:

```
Your question: What are the main configuration options?

Answer: Based on the documentation, the main configuration options include...
[Source: config_guide.txt | Relevance: 0.87]
```

## Choosing the Right Vector Database

ChromaDB is great for getting started, but you'll want to evaluate options as your dataset grows. Here's a comparison of common choices:

| Database | Hosting | Scale | Best For |
|----------|---------|-------|----------|
| ChromaDB | Local / self-hosted | Up to ~1M docs | Prototyping, small teams |
| Pinecone | Managed cloud | Billions of vectors | Production, no infra overhead |
| Weaviate | Self-hosted / cloud | Large scale | Multi-modal, complex queries |
| Qdrant | Self-hosted / cloud | Large scale | High performance, filtering |
| pgvector | PostgreSQL extension | Medium scale | Teams already on Postgres |

For most projects under 500K documents, ChromaDB running locally is perfectly adequate and free. Move to a managed solution (Pinecone is the most popular) when you need horizontal scaling or SLA guarantees.

## Common Pitfalls and How to Avoid Them

**1. Using a different embedding model at query time than at index time**

If you index with `all-MiniLM-L6-v2` but query with `text-embedding-3-small`, your results will be garbage. The embedding spaces are different. Always use the same model for both operations. Enforce this by storing the model name as collection metadata.

**2. Chunk size mismatch with your content type**

Code files need larger chunks than prose (context around a function matters). Legal documents may need smaller chunks to isolate specific clauses. Don't assume one chunk size fits all. Run a small manual evaluation: retrieve 10 queries and check if the returned chunks actually contain the answer.

**3. Not filtering by relevance threshold**

Without a minimum relevance score, your agent will inject noise into the context even when no relevant document exists. Set `min_relevance` to 0.3–0.4 for a starting point. If the agent says "I don't know" too often, lower it. If it hallucinates, raise it.

**4. Re-ingesting the entire corpus on every update**

For large corpora, full re-indexing is expensive. Use incremental updates: track a hash of each document, only re-embed documents whose hash has changed.

**5. Forgetting to handle the "no context" case**

When the retriever returns nothing relevant, your system prompt must explicitly instruct the LLM to say so — not to make up an answer. The prompt in our example handles this with: "If the answer is not in the context, say 'I don't have enough information.'"

## Conclusion

You now have a working RAG agent built from scratch. The core pattern is straightforward: embed your documents, embed the query, find similar chunks, and pass them as context to an LLM. The complexity comes from tuning the details — chunk size, embedding model, relevance threshold, and prompt engineering.

Start with this baseline, run it against real questions, and measure where it fails. That feedback loop is what separates a demo RAG system from one that works in production.

**Next steps to explore:**
- Add a conversational memory layer so follow-up questions work correctly
- Implement hybrid search (BM25 + vector) for better recall on exact keyword matches
- Add reranking with a cross-encoder model to improve precision before sending context to the LLM

If you want a deeper dive into vector database options, [Pinecone's learning center](https://www.pinecone.io/learn/) has solid, vendor-neutral explanations of the underlying concepts.
