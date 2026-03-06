---
title: "MCP (Model Context Protocol) Tutorial: Connect AI Agents to Any Tool"
date: 2026-03-06
description: "Learn Model Context Protocol from scratch: connect Claude and other AI agents to databases, APIs, and local tools with MCP servers. Full Python guide."
tags: ["MCP", "Model Context Protocol", "Claude tools", "AI agent integration", "tool calling"]
categories: ["Tutorials"]
keywords: ["Model Context Protocol tutorial", "MCP", "Claude tools", "AI agent integration", "tool calling"]
draft: false
---

*This post contains affiliate links. I may earn a commission at no extra cost to you.*

Claude can browse the web, run code, and read files — but how exactly does that work? The answer is MCP: the Model Context Protocol. MCP is an open standard that defines how AI models communicate with external tools and data sources. Instead of writing custom integration code every time you want to connect an LLM to a new service, MCP gives you a universal adapter layer.

In this tutorial, you'll understand how MCP works and build a working MCP server in Python. By the end, you'll be able to connect Claude to any tool, API, or database you control.

## What Is MCP and Why It Matters

Before MCP, connecting an LLM to external tools meant either (a) baking integration logic directly into your application code or (b) using framework-specific abstractions like LangChain tools. Both approaches have the same flaw: they're tightly coupled to a specific model or framework.

MCP decouples the model from its tools through a client-server protocol:

- **MCP Server**: Exposes tools, resources, and prompts over a standard protocol
- **MCP Client**: Connects to one or more servers and surfaces capabilities to the model
- **Host**: The application (Claude Desktop, your custom agent) that orchestrates everything

The protocol itself runs over standard I/O (for local tools) or HTTP+SSE (for remote services). This means you can write an MCP server once and use it with Claude, Claude Desktop, Cursor, and any other MCP-compatible client — without changing a line of server code.

## Three Types of MCP Capabilities

MCP servers can expose three kinds of capabilities:

**Tools**: Functions the model can call to take actions or retrieve data. These are the most commonly used capability. When Claude decides to use a tool, MCP handles the call and returns the result.

**Resources**: Read-only data sources (files, database records, live feeds) that can be included in the model's context. Unlike tools, resources don't execute side effects.

**Prompts**: Pre-built prompt templates that users can invoke by name. Useful for standardizing complex multi-step instructions.

In practice, most MCP servers you'll encounter today focus on tools. Let's build one.

## Setup

Install the Python MCP SDK:

```bash
pip install mcp anthropic python-dotenv
```

The `mcp` package provides both the server framework and the client utilities you'll use to test locally.

## Building Your First MCP Server

We'll build a file analysis server that exposes two tools: one that reads a file and returns its contents, and one that counts word frequencies in a document. This is the kind of utility an AI agent would use to analyze documents without loading them into context directly.

```python
# file_analysis_server.py
import asyncio
import json
from pathlib import Path
from collections import Counter
import re

from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Create the server instance
server = Server("file-analysis-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Register the tools this server exposes."""
    return [
        types.Tool(
            name="read_file",
            description=(
                "Read the contents of a file at the given path. "
                "Returns the file content as a string. "
                "Maximum file size: 1MB."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file"
                    }
                },
                "required": ["path"]
            }
        ),
        types.Tool(
            name="word_frequency",
            description=(
                "Analyze a text file and return the top N most frequent words. "
                "Ignores common stop words and punctuation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the text file to analyze"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top words to return (default: 20)",
                        "default": 20
                    }
                },
                "required": ["path"]
            }
        )
    ]

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "it", "its", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they"
}

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict
) -> list[types.TextContent]:
    """Execute a tool call and return the result."""

    if name == "read_file":
        path = Path(arguments["path"])

        if not path.exists():
            return [types.TextContent(
                type="text",
                text=f"Error: File not found: {path}"
            )]

        file_size = path.stat().st_size
        if file_size > 1_000_000:  # 1MB limit
            return [types.TextContent(
                type="text",
                text=f"Error: File too large ({file_size:,} bytes). Maximum is 1MB."
            )]

        try:
            content = path.read_text(encoding="utf-8")
            return [types.TextContent(
                type="text",
                text=content
            )]
        except UnicodeDecodeError:
            return [types.TextContent(
                type="text",
                text="Error: File is not valid UTF-8 text."
            )]

    elif name == "word_frequency":
        path = Path(arguments["path"])
        top_n = arguments.get("top_n", 20)

        if not path.exists():
            return [types.TextContent(
                type="text",
                text=f"Error: File not found: {path}"
            )]

        content = path.read_text(encoding="utf-8").lower()
        # Extract words, remove punctuation
        words = re.findall(r"\b[a-z]{3,}\b", content)
        # Filter stop words
        filtered_words = [w for w in words if w not in STOP_WORDS]
        # Count and return top N
        counter = Counter(filtered_words)
        top_words = counter.most_common(top_n)

        result = {
            "total_words": len(words),
            "unique_words": len(set(filtered_words)),
            "top_words": [{"word": w, "count": c} for w, c in top_words]
        }
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    else:
        return [types.TextContent(
            type="text",
            text=f"Error: Unknown tool: {name}"
        )]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="file-analysis-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing the Server Locally

The MCP SDK includes a test client you can run from the command line:

```bash
# Run the server interactively (for debugging)
python file_analysis_server.py
```

But a better way to test is to write a quick client script:

```python
# test_client.py
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_server():
    server_params = StdioServerParameters(
        command="python",
        args=["file_analysis_server.py"],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description[:60]}...")

            # Test read_file tool
            result = await session.call_tool(
                "read_file",
                {"path": "/etc/hostname"}
            )
            print(f"\nread_file result: {result.content[0].text}")

asyncio.run(test_server())
```

## Connecting to Claude via the API

Once your server works, connect it to Claude using the Anthropic SDK's MCP support:

```python
# claude_with_mcp.py
import asyncio
import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_claude_with_mcp(user_message: str):
    server_params = StdioServerParameters(
        command="python",
        args=["file_analysis_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Get tools from MCP server in Anthropic format
            mcp_tools = await session.list_tools()
            tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
                for tool in mcp_tools.tools
            ]

            client = anthropic.Anthropic()
            messages = [{"role": "user", "content": user_message}]

            # Agentic loop
            while True:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4096,
                    tools=tools,
                    messages=messages
                )

                # Add assistant response to history
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })

                if response.stop_reason == "end_turn":
                    # Extract final text response
                    for block in response.content:
                        if hasattr(block, "text"):
                            print(block.text)
                    break

                # Handle tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"Claude calling tool: {block.name}({block.input})")
                        result = await session.call_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result.content[0].text
                        })

                if tool_results:
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })

asyncio.run(run_claude_with_mcp(
    "Read /etc/hostname and tell me the machine name, "
    "then analyze the word frequency in /etc/hosts"
))
```

## Registering Your Server with Claude Desktop

To use your server in Claude Desktop, add it to the config file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "file-analysis": {
      "command": "python",
      "args": ["/absolute/path/to/file_analysis_server.py"]
    }
  }
}
```

Restart Claude Desktop. Your tools will appear in the tool picker, and Claude can invoke them during conversations.

## Building an HTTP MCP Server for Remote Access

The stdio transport is great for local tools, but for shared team infrastructure you'll want an HTTP server. MCP supports SSE (Server-Sent Events) transport for this:

```python
# http_server.py
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
import uvicorn

# Re-use the same `server` object from above

sse = SseServerTransport("/messages")

async def handle_sse(request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0], streams[1],
            server.create_initialization_options()
        )

app = Starlette(routes=[
    Route("/sse", endpoint=handle_sse),
    Mount("/messages", app=sse.handle_post_message)
])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

Now any MCP client can connect to `http://your-server:8080/sse`.

## Common Mistakes to Avoid

**Missing error handling**: MCP servers run as subprocesses. If your server crashes, Claude gets a confusing error. Always wrap tool implementations in try/except and return descriptive error messages as `TextContent`.

**Blocking the event loop**: MCP servers are async. If you make synchronous HTTP calls or file I/O inside an async handler, you'll block the whole server. Use `asyncio.to_thread()` for blocking operations.

**Overly broad tool descriptions**: Claude uses the tool description to decide when to call it. Vague descriptions ("does stuff with files") lead to wrong tool selection. Be specific about what the tool does, what inputs it expects, and what it returns.

**No input validation**: Validate `arguments` before using them. Malformed inputs from Claude (rare but possible during complex reasoning) can crash your server. Check types, ranges, and required fields explicitly.

The [official MCP documentation](https://modelcontextprotocol.io/docs) covers the full specification including resources, prompts, and the sampling protocol for servers that need to call back into the LLM.
