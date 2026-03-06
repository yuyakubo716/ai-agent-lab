---
title: "LangGraph Tutorial: Building Stateful AI Workflows with Graphs"
date: 2026-03-06
description: "Learn LangGraph from scratch: build stateful AI workflows using graph-based agent orchestration with Python. Complete tutorial with code."
tags: ["LangGraph", "stateful AI", "LangChain", "AI agent", "workflow graph"]
categories: ["Tutorials"]
keywords: ["LangGraph tutorial", "stateful AI", "workflow graph", "LangChain", "agent state management"]
draft: false
---

*This post contains affiliate links. I may earn a commission at no extra cost to you.*

Most AI agent frameworks treat conversations as stateless — every call starts fresh. That works for simple Q&A, but falls apart the moment you need multi-step workflows, conditional branching, or persistent memory across turns. LangGraph solves this by modeling your agent as a directed graph where each node is a processing step and edges determine what happens next.

In this tutorial, you'll build a working stateful AI workflow using LangGraph and Python. We'll start with a minimal example and work up to a multi-node agent with conditional routing — the kind of thing you'd actually deploy in production.

## Why LangGraph Instead of a Simple Chain

LangChain's original LCEL (LangChain Expression Language) chains work linearly: input goes in, output comes out. That's fine for RAG or simple Q&A. But real-world workflows rarely flow in a straight line. You need:

- **Cycles**: Let the agent loop back and retry if it gets a bad result
- **Conditional branching**: Route to different nodes based on what the model decides
- **Persistent state**: Carry information across multiple steps without manual threading
- **Human-in-the-loop**: Pause and wait for approval before continuing

LangGraph gives you all of this through an explicit graph abstraction. You define nodes (Python functions), edges (how nodes connect), and a state schema (what data flows through the graph). The framework handles execution, streaming, and checkpointing.

## Prerequisites and Setup

You'll need Python 3.11+ and an Anthropic API key. Install the dependencies:

```bash
pip install langgraph langchain-anthropic python-dotenv
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=your_key_here
```

LangGraph 0.2+ works independently of LangChain's chain abstractions — you can use it with any LLM client, though we'll use `langchain-anthropic` here for convenience.

## Core Concepts: Nodes, Edges, and State

Before writing code, let's be precise about what LangGraph actually does.

**State** is a typed dictionary that every node reads from and writes to. Define it once, and LangGraph handles passing it between nodes:

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # append-only message list
    next_step: str
    retry_count: int
```

The `Annotated[list, add_messages]` tells LangGraph to append new messages rather than replace the existing list — a critical distinction when building conversational agents.

**Nodes** are plain Python functions that take state and return a dict of updates:

```python
def my_node(state: AgentState) -> dict:
    # read from state
    messages = state["messages"]
    # do some work
    result = call_llm(messages)
    # return partial state update
    return {"messages": [result], "retry_count": state["retry_count"] + 1}
```

**Edges** connect nodes. A basic edge always goes from A to B. A conditional edge runs a function to decide where to go next.

## Building Your First LangGraph Agent

Let's build a research agent that: (1) generates a search query, (2) simulates a web search, and (3) writes a summary. Nodes can loop back if the search result is poor.

```python
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

class ResearchState(TypedDict):
    query: str
    search_results: str
    summary: str
    messages: Annotated[list, add_messages]
    quality_check: str
    attempts: int
```

Now define each node:

```python
def generate_search_query(state: ResearchState) -> dict:
    """Turn the user's question into a precise search query."""
    system = SystemMessage(content=(
        "You are a research assistant. Given a question, output ONLY a "
        "concise search query — no explanation, no extra text."
    ))
    human = HumanMessage(content=f"Question: {state['query']}")
    response = llm.invoke([system, human])
    search_query = response.content.strip()
    return {
        "messages": [human, response],
        "query": search_query,
        "attempts": state.get("attempts", 0)
    }

def simulate_search(state: ResearchState) -> dict:
    """Simulate a web search. Replace with real search API in production."""
    # In production: use Tavily, SerpAPI, or Brave Search API here
    mock_results = f"""
    Search results for: {state['query']}

    Result 1: LangGraph is a library for building stateful multi-actor applications
    with LLMs, built on top of LangChain. It extends LangChain Expression Language
    with the ability to coordinate multiple chains (or actors) across multiple steps
    of computation in a cyclic fashion.

    Result 2: Key features include persistent checkpointing, streaming support,
    human-in-the-loop workflows, and first-class support for multi-agent coordination.

    Result 3: LangGraph uses a graph-based model where nodes are computation steps
    and edges define the flow between them, including conditional routing.
    """
    return {"search_results": mock_results}

def write_summary(state: ResearchState) -> dict:
    """Synthesize search results into a clear summary."""
    system = SystemMessage(content=(
        "You are a technical writer. Synthesize the search results into a "
        "clear, accurate 2-3 paragraph summary. Be specific and factual."
    ))
    human = HumanMessage(content=(
        f"Original question: {state['query']}\n\n"
        f"Search results:\n{state['search_results']}"
    ))
    response = llm.invoke([system, human])
    return {"messages": [human, response], "summary": response.content}

def quality_check(state: ResearchState) -> dict:
    """Evaluate if the summary is good enough or needs another attempt."""
    system = SystemMessage(content=(
        "You are a quality reviewer. Given a summary, respond with exactly "
        "one word: 'PASS' if the summary is clear and specific, or 'RETRY' "
        "if it is vague, too short, or inaccurate."
    ))
    human = HumanMessage(content=f"Summary to review:\n{state['summary']}")
    response = llm.invoke([system, human])
    verdict = "PASS" if "PASS" in response.content.upper() else "RETRY"
    return {
        "messages": [human, response],
        "quality_check": verdict,
        "attempts": state.get("attempts", 0) + 1
    }
```

Define the conditional routing function:

```python
def route_after_check(state: ResearchState) -> Literal["generate_search_query", "end"]:
    """Retry up to 2 times, then force completion."""
    if state["quality_check"] == "PASS" or state.get("attempts", 0) >= 2:
        return "end"
    return "generate_search_query"
```

Assemble the graph:

```python
def build_research_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("generate_search_query", generate_search_query)
    graph.add_node("simulate_search", simulate_search)
    graph.add_node("write_summary", write_summary)
    graph.add_node("quality_check", quality_check)

    # Set entry point
    graph.set_entry_point("generate_search_query")

    # Add edges
    graph.add_edge("generate_search_query", "simulate_search")
    graph.add_edge("simulate_search", "write_summary")
    graph.add_edge("write_summary", "quality_check")

    # Conditional edge: retry or finish
    graph.add_conditional_edges(
        "quality_check",
        route_after_check,
        {
            "generate_search_query": "generate_search_query",
            "end": END
        }
    )

    return graph.compile()
```

Run it:

```python
if __name__ == "__main__":
    app = build_research_graph()

    initial_state = {
        "query": "How does LangGraph handle state persistence?",
        "search_results": "",
        "summary": "",
        "messages": [],
        "quality_check": "",
        "attempts": 0
    }

    result = app.invoke(initial_state)
    print("Final Summary:")
    print(result["summary"])
    print(f"\nCompleted in {result['attempts']} attempt(s)")
```

## Adding Checkpointing for Persistence

One of LangGraph's killer features is checkpointing — automatically saving state so you can resume workflows across sessions, even across process restarts.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# In-memory (for development/testing)
memory_checkpointer = MemorySaver()

# SQLite (for single-server production use)
db_checkpointer = SqliteSaver.from_conn_string("./agent_state.db")

# Compile with checkpointer
app = build_research_graph().compile(checkpointer=db_checkpointer)

# Every invocation needs a thread_id to identify the session
config = {"configurable": {"thread_id": "user-session-42"}}

# First call
result1 = app.invoke(initial_state, config=config)

# Resume from same thread later (state is automatically restored from DB)
result2 = app.invoke(
    {"query": "Tell me more about the conditional edges"},
    config=config
)
```

The thread model means you can build multi-turn agents where each conversation thread has its own isolated state — without any manual session management on your part.

## Streaming Graph Execution

For production UIs, you usually want to stream partial results rather than wait for the full graph to complete. LangGraph supports this natively:

```python
for event in app.stream(initial_state, config=config):
    for node_name, node_output in event.items():
        print(f"[{node_name}] completed")
        if "summary" in node_output:
            print(f"Summary preview: {node_output['summary'][:100]}...")
```

You can also stream at the token level inside a node by using `llm.astream()` with async LangGraph execution — useful when you want real-time character-by-character output in a chat UI.

## Common Pitfalls

**State mutation**: Never mutate `state` in place inside a node. Always return a new dict. LangGraph merges your returned dict with the existing state — direct mutation bypasses this and causes unpredictable behavior.

**Infinite loops**: Conditional edges can create cycles. Always include a max-iteration guard (like the `attempts >= 2` check above) to prevent runaway loops.

**Large state**: State is serialized at every checkpoint. If you store large binary data or embeddings in state, you'll hit performance issues quickly. Store large data externally (S3, a database) and keep only references in state.

**Async vs sync**: LangGraph supports both sync `invoke()` and async `ainvoke()`. Mix them at your peril — using a sync checkpointer with an async graph will deadlock.

## What to Build Next

This tutorial covered the core LangGraph primitives. From here, good next projects are:

- **Multi-agent graphs**: Create separate sub-graphs for specialized agents and connect them via handoff nodes
- **Human-in-the-loop**: Use `interrupt_before=["node_name"]` when compiling to pause for human review
- **Parallel execution**: Use `Send()` to fan out work to multiple nodes simultaneously, then join the results

LangGraph's graph model takes some getting used to, but once it clicks, it's the cleanest way to express complex agent behavior. The explicit state schema alone — compared to passing kwargs through a chain — makes debugging and testing dramatically easier.

For further reference, the [LangGraph documentation](https://langchain-ai.github.io/langgraph/) covers the full API including multi-agent architectures and the Studio debugging interface.
