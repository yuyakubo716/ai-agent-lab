---
title: "AI Agent Memory Systems: Short-Term vs Long-Term Persistent Context"
date: 2026-03-06
description: "Learn how AI agent memory works—from in-context buffers to vector stores. Build persistent memory in Python with practical code examples."
tags: ["AI agents", "memory", "vector store", "RAG", "Python"]
categories: ["Tutorials"]
keywords: ["AI agent memory", "persistent memory", "context window", "vector store", "agent state"]
draft: false
---

*This post contains affiliate links. I may earn a commission at no extra cost to you.*

One of the most common frustrations when building AI agents is watching them forget everything the moment a conversation ends—or worse, running out of context halfway through a long session. Memory is what separates a toy chatbot from a production-grade agent.

In this guide, you will build a two-tier memory system: fast in-memory buffer for recent turns, and a persistent vector store for long-term recall. All code is runnable with Python 3.11+ and the Anthropic SDK.

## Why Memory Matters for AI Agents

A language model has no inherent state. Every API call starts cold. Without explicit memory management, your agent:

- Repeats questions it already asked
- Contradicts advice it gave earlier in the session
- Loses user preferences the moment the process restarts
- Burns tokens re-reading the same background context on every turn

Memory systems solve these problems by storing, indexing, and selectively retrieving relevant information before it is fed to the model.

## The Four Types of Agent Memory

Before writing code, it helps to understand the landscape:

| Type | Storage | Speed | Capacity | Use Case |
|------|---------|-------|----------|----------|
| In-context buffer | RAM (prompt) | Instant | ~200K tokens | Current session turns |
| External key-value | Redis / SQLite | Fast | Unlimited | User profiles, preferences |
| Vector store | Chroma / Pinecone | Medium | Unlimited | Semantic retrieval |
| Episodic log | File / DB | Slow | Unlimited | Audit trail, fine-tuning |

Most production agents combine types 1 and 3: a short sliding window in the prompt plus semantic search over historical data.

## Setting Up the Project

```bash
pip install anthropic chromadb sentence-transformers python-dotenv
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Building the In-Context Buffer (Short-Term Memory)

The simplest memory is a fixed-size deque that holds the last N conversation turns:

```python
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str

class ShortTermMemory:
    """Sliding window buffer for recent conversation turns."""

    def __init__(self, max_turns: int = 20):
        self._buffer: deque[Turn] = deque(maxlen=max_turns)

    def add(self, role: Literal["user", "assistant"], content: str) -> None:
        self._buffer.append(Turn(role=role, content=content))

    def to_messages(self) -> list[dict]:
        return [{"role": t.role, "content": t.content} for t in self._buffer]

    def token_estimate(self) -> int:
        """Rough estimate: 1 token ≈ 4 chars."""
        total = sum(len(t.content) for t in self._buffer)
        return total // 4
```

This is trivial but critical. Without it, you either send the full history on every call (expensive) or nothing at all (amnesiac).

## Building the Vector Store (Long-Term Memory)

For long-term memory, we embed text chunks and store them in Chroma. On each new turn, we retrieve the top-k most semantically relevant past exchanges.

```python
import chromadb
from chromadb.utils import embedding_functions
import uuid
from datetime import datetime

class LongTermMemory:
    """Persistent vector store backed by ChromaDB."""

    def __init__(self, persist_dir: str = ".agent_memory"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collection = self._client.get_or_create_collection(
            name="agent_memory",
            embedding_function=self._ef,
        )

    def store(self, text: str, metadata: dict | None = None) -> str:
        doc_id = str(uuid.uuid4())
        self._collection.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[{**(metadata or {}), "timestamp": datetime.utcnow().isoformat()}],
        )
        return doc_id

    def recall(self, query: str, top_k: int = 5) -> list[str]:
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
        )
        return results["documents"][0]
```

Key design decisions here:
- `all-MiniLM-L6-v2` is fast and runs locally—no API calls for embeddings
- `PersistentClient` writes to disk so memory survives process restarts
- We cap `n_results` to avoid querying more results than stored items

## Wiring Both Systems into an Agent

Now combine both memory tiers into a single agent class:

```python
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

class MemoryAgent:
    def __init__(self):
        self._client = anthropic.Anthropic()
        self._short = ShortTermMemory(max_turns=10)
        self._long = LongTermMemory()
        self._system = (
            "You are a helpful assistant with persistent memory. "
            "Long-term memories retrieved from past sessions are provided "
            "under the [RECALLED MEMORIES] section when relevant."
        )

    def chat(self, user_message: str) -> str:
        # 1. Retrieve relevant long-term memories
        recalled = self._long.recall(user_message, top_k=3)

        # 2. Build system prompt with recalled context
        system = self._system
        if recalled:
            memory_block = "\n".join(f"- {m}" for m in recalled)
            system += f"\n\n[RECALLED MEMORIES]\n{memory_block}"

        # 3. Add user turn to short-term buffer
        self._short.add("user", user_message)

        # 4. Call the model
        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=self._short.to_messages(),
        )
        assistant_message = response.content[0].text

        # 5. Store in both memories
        self._short.add("assistant", assistant_message)
        self._long.store(
            f"User: {user_message}\nAssistant: {assistant_message}",
            metadata={"type": "conversation"},
        )

        return assistant_message
```

## Testing the Memory Agent

```python
agent = MemoryAgent()

# Session 1
print(agent.chat("My name is Alex and I prefer Python over JavaScript."))
print(agent.chat("What programming language do I prefer?"))

# Simulate process restart by creating a new agent instance
# Long-term memory persists on disk; short-term starts fresh
agent2 = MemoryAgent()
print(agent2.chat("Do you remember what programming language I prefer?"))
```

Expected output from the third message:
```
Yes! From our previous conversation, you mentioned that you prefer Python over JavaScript.
```

The second agent instance has no short-term buffer, but the vector store recalls the relevant exchange.

## Memory Consolidation: Summarizing Old Turns

A common production pattern is to periodically summarize the short-term buffer and inject the summary into long-term memory. This prevents the prompt from ballooning:

```python
def consolidate(agent: MemoryAgent) -> None:
    """Summarize recent turns and store the summary in long-term memory."""
    turns = agent._short.to_messages()
    if len(turns) < 6:
        return  # Not enough to summarize yet

    summary_prompt = (
        "Summarize the following conversation in 3-5 bullet points, "
        "focusing on facts about the user and decisions made:\n\n"
        + "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    )

    response = agent._client.messages.create(
        model="claude-haiku-4-5-20251001",  # Cheaper model for summarization
        max_tokens=256,
        messages=[{"role": "user", "content": summary_prompt}],
    )
    summary = response.content[0].text
    agent._long.store(summary, metadata={"type": "summary"})
```

Using `claude-haiku-4-5-20251001` for summarization keeps costs low while reserving the more capable model for actual conversations.

## Common Pitfalls and How to Avoid Them

**Pitfall 1: Retrieving too many memories**
Injecting 20 recalled chunks inflates the prompt and confuses the model. Keep `top_k` at 3-5 and filter by a minimum similarity threshold.

```python
results = self._collection.query(
    query_texts=[query],
    n_results=5,
    where_document={"$contains": "preference"},  # Optional metadata filter
)
```

**Pitfall 2: Stale memories contradicting current facts**
Add a TTL to stored memories by writing a cleanup job:

```python
from datetime import datetime, timedelta

def prune_old_memories(ltm: LongTermMemory, days: int = 30) -> int:
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    all_items = ltm._collection.get(include=["metadatas", "ids"])
    old_ids = [
        id_ for id_, meta in zip(all_items["ids"], all_items["metadatas"])
        if meta.get("timestamp", "") < cutoff
    ]
    if old_ids:
        ltm._collection.delete(ids=old_ids)
    return len(old_ids)
```

**Pitfall 3: Embedding model mismatch after upgrade**
If you change the embedding model, all stored vectors become incompatible. Version your collection names: `agent_memory_v2`.

## Scaling to Production

For production deployments, swap the local components for managed services:

- **ChromaDB local** → [Pinecone](https://www.pinecone.io/) (managed, serverless, auto-scaling)
- **In-process deque** → Redis sorted set (survives across multiple API server instances)
- **File persistence** → PostgreSQL with pgvector extension

The interface layer (ShortTermMemory / LongTermMemory classes) stays the same—only the backend changes. This abstraction is worth maintaining from day one.

## What to Build Next

Now that your agent remembers things, the natural next step is giving it tools to act on what it remembers. Check out our [MCP tutorial](/posts/mcp-model-context-protocol-tutorial/) to learn how to connect the agent to external systems like calendars, databases, and APIs—all while keeping memory state across sessions.

## Conclusion

A two-tier memory architecture (short-term buffer + long-term vector store) covers 90% of production use cases. The code in this guide is production-ready with one caveat: swap the local embedding model and ChromaDB for managed services once you exceed a few thousand stored memories.

The most important lesson: treat memory as a first-class design concern from the start. Retrofitting memory onto an agent that was designed without it is painful. Start with the `MemoryAgent` class above and expand from there.
