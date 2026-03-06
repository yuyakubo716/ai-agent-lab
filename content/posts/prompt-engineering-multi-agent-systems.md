---
title: "Prompt Engineering for Multi-Agent Systems: Patterns That Actually Work"
date: 2026-03-06
description: "Master prompt engineering for multi-agent AI systems. Learn role prompting, handoff patterns, and failure-recovery techniques with working Python examples."
tags: ["prompt engineering", "multi-agent", "AI agents", "system prompt", "Claude"]
categories: ["Tutorials"]
keywords: ["prompt engineering multi-agent", "system prompt", "agent persona", "role prompting", "chain-of-thought"]
draft: false
---

*This post contains affiliate links. I may earn a commission at no extra cost to you.*

Single-agent prompting is well-documented. Multi-agent prompting is not—and the failure modes are completely different. When agents hand off to each other, a vague instruction that would produce mediocre output from one model can cause catastrophic failures across a pipeline of five.

This guide covers the prompt engineering patterns that actually work in production multi-agent systems, with examples drawn from running a real multi-agent orchestration framework.

## Why Multi-Agent Prompting is Different

In a single-agent system, a bad prompt gives you a bad answer. You iterate.

In a multi-agent system, a bad prompt in the orchestrator:
- Sends agents down the wrong track
- Causes agents to misunderstand handoff context
- Creates cascading failures where each agent amplifies the previous mistake
- Makes debugging nearly impossible because the failure point is buried in message three of a five-agent chain

The stakes are higher, and the feedback loop is slower. Front-load your prompt quality.

## Pattern 1: Explicit Role Separation with Persona Prompts

Every agent in a multi-agent system needs a crisp persona prompt that answers four questions:
1. Who am I?
2. What is my single responsibility?
3. What am I forbidden from doing?
4. How do I signal completion?

```python
ORCHESTRATOR_PROMPT = """
You are the Orchestrator in a research pipeline. Your single responsibility is
task decomposition and agent coordination.

YOUR ROLE:
- Receive a research goal from the user
- Break it into subtasks (maximum 5)
- Assign each subtask to the correct specialist agent
- Aggregate results into a final report

YOU ARE FORBIDDEN FROM:
- Performing web searches yourself
- Writing code
- Making final decisions without consulting the Analyst agent for tasks involving numbers

COMPLETION SIGNAL:
When you have a complete report, output exactly: FINAL_REPORT: followed by your report.
"""

RESEARCHER_PROMPT = """
You are the Researcher agent. Your single responsibility is web research.

YOUR ROLE:
- Receive a research question from the Orchestrator
- Use the web_search tool to find relevant sources
- Return a structured summary with sources cited

YOU ARE FORBIDDEN FROM:
- Interpreting data (send to Analyst)
- Writing final recommendations (send to Orchestrator)
- Making more than 5 search calls per subtask

COMPLETION SIGNAL:
Output exactly: RESEARCH_COMPLETE: followed by your findings.
"""
```

The "forbidden from" section is the most important part. Without it, agents will over-reach into each other's responsibilities, and you lose the benefits of specialization.

## Pattern 2: Structured Handoff Messages

Agent-to-agent messages should be structured, not conversational. Use a fixed schema:

```python
from pydantic import BaseModel
from typing import Literal

class AgentHandoff(BaseModel):
    from_agent: str
    to_agent: str
    task_id: str
    task_type: Literal["research", "analysis", "writing", "review"]
    instruction: str
    context: dict  # Relevant prior work
    constraints: list[str]  # Hard constraints the receiving agent must respect
    output_format: str  # Exact format expected

# Example handoff from Orchestrator to Researcher
handoff = AgentHandoff(
    from_agent="orchestrator",
    to_agent="researcher",
    task_id="task_001",
    task_type="research",
    instruction="Find the top 3 vector database providers and their pricing models",
    context={"user_goal": "Evaluate vector DBs for a RAG system with 10M vectors"},
    constraints=[
        "Only use sources published after 2025-01-01",
        "Pinecone, Weaviate, and Chroma are the comparison targets",
    ],
    output_format="JSON array with fields: name, pricing_model, cost_per_1m_vectors, notes",
)
```

When the Researcher receives this, it has everything it needs. There is no ambiguity about what output format to produce, and the constraints prevent it from going off-script.

The mistake most teams make is passing raw text between agents. Structured handoffs pay off when you need to debug: you can inspect exactly what instructions each agent received.

## Pattern 3: Chain-of-Thought Injection for Complex Reasoning

For agents that need to reason through multi-step problems, prefix the user turn with a CoT scaffold:

```python
def inject_cot(task: str, steps: list[str]) -> str:
    """Inject chain-of-thought scaffold into a task prompt."""
    scaffold = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
    return f"""
{task}

Think through this step by step:
{scaffold}

Only provide your final answer after completing all steps.
"""

# Example: Analyst agent receives a CoT-scaffolded prompt
analysis_task = inject_cot(
    task="Determine which vector database is best for a 10M vector RAG system with <100ms p95 latency.",
    steps=[
        "List the latency specifications for each database from the research data",
        "Check which ones support HNSW indexing (required for <100ms at 10M vectors)",
        "Compare pricing for 10M vectors at 1000 queries/day",
        "Rank by: latency compliance first, then cost",
    ],
)
```

This pattern is especially powerful for agents that handle complex decisions. The CoT scaffold forces a reasoning trace that you can inspect in the agent's output—invaluable for debugging wrong conclusions.

## Pattern 4: Output Format Contracts

Agents must produce output that downstream agents can parse reliably. Use format contracts:

```python
FORMAT_CONTRACTS = {
    "researcher": {
        "prefix": "RESEARCH_COMPLETE:",
        "schema": {
            "findings": "list of dicts with keys: claim, source, confidence",
            "sources": "list of URLs",
            "gaps": "list of unanswered questions",
        },
    },
    "analyst": {
        "prefix": "ANALYSIS_COMPLETE:",
        "schema": {
            "recommendation": "string (1 sentence)",
            "reasoning": "list of supporting points",
            "confidence": "float 0.0-1.0",
            "caveats": "list of limitations",
        },
    },
}

def build_output_instruction(agent_name: str) -> str:
    contract = FORMAT_CONTRACTS[agent_name]
    schema_lines = "\n".join(
        f"  - {k}: {v}" for k, v in contract["schema"].items()
    )
    return f"""
Your response MUST start with exactly: {contract["prefix"]}
Followed immediately by a JSON object with these fields:
{schema_lines}

Any deviation from this format will cause a pipeline failure.
"""
```

Include this instruction at the end of every system prompt. Agents are far more likely to comply when the format requirement appears in the closing section rather than buried in the middle.

## Pattern 5: Failure Recovery Prompts

Agents will produce malformed output. Build explicit recovery into the orchestrator:

```python
import json
import anthropic

client = anthropic.Anthropic()

def parse_agent_output(raw: str, agent_name: str) -> dict | None:
    """Attempt to parse structured output; return None on failure."""
    contract = FORMAT_CONTRACTS.get(agent_name)
    if not contract:
        return None
    prefix = contract["prefix"]
    if prefix not in raw:
        return None
    json_start = raw.index(prefix) + len(prefix)
    try:
        return json.loads(raw[json_start:].strip())
    except json.JSONDecodeError:
        return None

def recover_malformed_output(raw: str, agent_name: str) -> dict:
    """Ask Claude to fix malformed agent output."""
    contract = FORMAT_CONTRACTS[agent_name]
    recovery_prompt = f"""
The following agent output is malformed. Extract the data and reformat it
as valid JSON matching this schema:
{json.dumps(contract["schema"], indent=2)}

Malformed output:
{raw}

Return ONLY valid JSON. No explanation.
"""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": recovery_prompt}],
    )
    return json.loads(response.content[0].text)
```

Using `claude-haiku-4-5-20251001` for recovery keeps costs minimal. This function is called only on failure paths, and the task is straightforward reformatting rather than complex reasoning.

## Pattern 6: Context Window Budget Management

In a multi-agent chain, each agent accumulates context. By agent 4, you may be at 150K tokens. Budget proactively:

```python
def estimate_tokens(text: str) -> int:
    return len(text) // 4  # ~4 chars per token

class ContextBudget:
    def __init__(self, model_limit: int = 200_000, reserve: float = 0.2):
        self.limit = model_limit
        self.reserve = int(model_limit * reserve)
        self.available = model_limit - self.reserve
        self.used = 0

    def fits(self, text: str) -> bool:
        return self.used + estimate_tokens(text) <= self.available

    def consume(self, text: str) -> None:
        self.used += estimate_tokens(text)

    def remaining_chars(self) -> int:
        return (self.available - self.used) * 4

# Usage in orchestrator
budget = ContextBudget()
for agent_output in collected_results:
    if not budget.fits(agent_output):
        # Summarize before passing to next agent
        agent_output = summarize(agent_output, budget.remaining_chars() // 2)
    budget.consume(agent_output)
```

This prevents silent truncation, which is one of the hardest bugs to diagnose in multi-agent systems: the agent receives a cut-off input and produces confidently wrong output.

## Pattern 7: The Skeptical Reviewer Prompt

For quality-critical pipelines, add a dedicated reviewer agent whose only job is to find problems:

```python
REVIEWER_PROMPT = """
You are the Reviewer agent. You are congenitally skeptical.

YOUR ROLE:
- Receive completed work from other agents
- Find errors, gaps, contradictions, and unsupported claims
- Do NOT suggest how to fix them (that is not your job)
- Return a structured list of problems

YOUR MINDSET:
- Assume every claim needs a source
- Assume every number could be wrong
- Assume every recommendation has a counterargument that was not considered

COMPLETION SIGNAL:
Output exactly: REVIEW_COMPLETE: followed by JSON with:
  - issues: list of {severity: "critical"|"major"|"minor", description: string}
  - pass: boolean (true only if zero critical issues)
"""
```

This pattern catches errors that are invisible to the agent that produced the work. Agents are optimistic about their own output; the reviewer is structurally pessimistic.

## Debugging Multi-Agent Prompt Failures

When a pipeline produces wrong output, work backwards:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("multi_agent")

class InstrumentedAgent:
    """Wrap any agent with full I/O logging."""

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self._system = system_prompt
        self._client = anthropic.Anthropic()

    def run(self, message: str) -> str:
        logger.debug(f"[{self.name}] INPUT: {message[:200]}...")
        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=self._system,
            messages=[{"role": "user", "content": message}],
        )
        output = response.content[0].text
        logger.debug(f"[{self.name}] OUTPUT: {output[:200]}...")
        return output
```

Always log full inputs and outputs in development. In production, log at least the first and last 200 characters plus any structured fields. The failure is almost always in the handoff, not the model itself.

## The Minimal Viable Multi-Agent Prompt Stack

If you take one thing from this guide, it's this three-layer structure that should appear in every agent's system prompt:

```
1. IDENTITY (2-3 sentences)
   Who you are and your single responsibility.

2. BOUNDARIES (bullet list)
   What you will not do, clearly stated.

3. OUTPUT CONTRACT (exact format)
   The exact prefix and JSON schema you must produce.
```

Everything else—CoT scaffolds, handoff context, task-specific instructions—goes in the user turn, not the system prompt. Keep system prompts stable; let user turns vary.

For a deeper dive into the infrastructure side, check out our [LangGraph tutorial](/posts/langgraph-tutorial-stateful-ai-workflows/) on managing stateful workflows across multiple agents.

## Conclusion

Multi-agent prompt engineering is a discipline, not an afterthought. The patterns here—explicit role separation, structured handoffs, output contracts, and skeptical reviewers—are the result of watching multi-agent pipelines fail in predictable ways.

The investment in clean prompt architecture pays back immediately: fewer debugging sessions, more predictable outputs, and pipelines that degrade gracefully instead of silently producing wrong answers. Start with the minimal viable stack, add patterns as your pipeline grows, and treat every agent boundary as a potential failure point worth hardening.
