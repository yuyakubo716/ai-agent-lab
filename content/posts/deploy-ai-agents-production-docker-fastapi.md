---
title: "How to Deploy AI Agents to Production: Docker, FastAPI, and AWS"
date: 2026-03-06
description: "Deploy AI agents to production with Docker, FastAPI, and AWS. Complete guide covering containerization, API design, autoscaling, and cost management."
tags: ["deploy AI agents", "Docker AI agent", "FastAPI", "AWS Lambda", "AI deployment"]
categories: ["Tutorials"]
keywords: ["deploy AI agents production", "Docker AI agent", "FastAPI", "AWS Lambda", "AI deployment"]
draft: false
---

*This post contains affiliate links. I may earn a commission at no extra cost to you.*

Getting an AI agent to work on your laptop is one thing. Getting it to run reliably at scale — handling concurrent users, managing costs, surviving traffic spikes, and recovering from failures — is an entirely different engineering problem. This guide walks through a production-grade deployment stack for AI agents using Docker, FastAPI, and AWS.

We'll deploy a real agent: a code review bot that takes a GitHub diff and returns structured feedback. The patterns here apply to any agent built on top of an LLM API.

## Choosing the Right AWS Architecture

Before writing a line of infrastructure code, the most important decision is where your agent actually runs. Two options dominate:

**AWS Lambda** works well when:
- Agent calls are short-lived (under 15 minutes)
- Traffic is bursty and unpredictable
- You want zero infrastructure management
- Cold start latency (2-5 seconds) is acceptable

**ECS Fargate** (or EKS) works better when:
- Agents maintain in-memory state between calls
- You need WebSocket or streaming connections
- Response times must be consistently fast (no cold starts)
- You need full control over the runtime environment

For most API-based agents that use a hosted LLM (Claude, GPT-4, Gemini), Lambda is the right default — especially early on when traffic is unpredictable. We'll deploy to Lambda using a Docker container image, which avoids the 250MB zip artifact limit.

## Project Structure

```
agent-code-reviewer/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app
│   ├── agent.py         # Agent logic
│   ├── models.py        # Pydantic schemas
│   └── config.py        # Environment-based config
├── Dockerfile
├── docker-compose.yml   # For local development
├── requirements.txt
├── .env.example
└── infra/
    ├── lambda_handler.py
    └── terraform/       # Optional IaC
```

## Building the FastAPI Agent

Start with the data models and configuration:

```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    max_tokens: int = 4096
    model: str = "claude-sonnet-4-6"
    request_timeout: int = 120
    environment: str = "development"

    class Config:
        env_file = ".env"

settings = Settings()
```

```python
# app/models.py
from pydantic import BaseModel, Field
from typing import Optional

class CodeReviewRequest(BaseModel):
    diff: str = Field(..., description="Git diff to review", max_length=50_000)
    language: Optional[str] = Field(None, description="Programming language hint")
    focus: Optional[str] = Field(
        None,
        description="Review focus: 'security', 'performance', 'style', or None for all"
    )

class ReviewComment(BaseModel):
    severity: str  # "critical", "warning", "suggestion"
    line_reference: Optional[str]
    issue: str
    recommendation: str

class CodeReviewResponse(BaseModel):
    summary: str
    comments: list[ReviewComment]
    overall_score: int  # 1-10
    estimated_review_time_saved_minutes: int
```

The agent logic:

```python
# app/agent.py
import anthropic
import json
from .config import settings
from .models import CodeReviewRequest, CodeReviewResponse, ReviewComment

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

REVIEW_TOOL = {
    "name": "submit_code_review",
    "description": "Submit the completed code review with structured feedback",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "High-level summary of the changes and main findings"
            },
            "comments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string", "enum": ["critical", "warning", "suggestion"]},
                        "line_reference": {"type": "string"},
                        "issue": {"type": "string"},
                        "recommendation": {"type": "string"}
                    },
                    "required": ["severity", "issue", "recommendation"]
                }
            },
            "overall_score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Code quality score (10 = excellent)"
            },
            "estimated_review_time_saved_minutes": {
                "type": "integer",
                "description": "Estimated manual review time this replaces"
            }
        },
        "required": ["summary", "comments", "overall_score", "estimated_review_time_saved_minutes"]
    }
}

def build_system_prompt(focus: str | None) -> str:
    focus_instruction = ""
    if focus == "security":
        focus_instruction = "\nFocus primarily on security vulnerabilities, injection risks, and unsafe practices."
    elif focus == "performance":
        focus_instruction = "\nFocus primarily on performance bottlenecks, unnecessary allocations, and algorithmic complexity."
    elif focus == "style":
        focus_instruction = "\nFocus primarily on code style, naming conventions, readability, and maintainability."

    return f"""You are an expert code reviewer. Analyze the provided git diff and identify:
- Bugs and logic errors
- Security vulnerabilities
- Performance issues
- Code style and maintainability problems
- Missing error handling
{focus_instruction}
Be specific, actionable, and constructive. Reference specific line changes when possible.
Use the submit_code_review tool to return your structured review."""

async def review_code(request: CodeReviewRequest) -> CodeReviewResponse:
    """Run the code review agent and return structured feedback."""
    messages = [
        {
            "role": "user",
            "content": (
                f"Please review this code diff"
                f"{f' ({request.language})' if request.language else ''}:\n\n"
                f"```diff\n{request.diff}\n```"
            )
        }
    ]

    response = client.messages.create(
        model=settings.model,
        max_tokens=settings.max_tokens,
        system=build_system_prompt(request.focus),
        tools=[REVIEW_TOOL],
        tool_choice={"type": "any"},  # Force tool use for structured output
        messages=messages
    )

    # Extract the tool call result
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_code_review":
            data = block.input
            return CodeReviewResponse(
                summary=data["summary"],
                comments=[ReviewComment(**c) for c in data["comments"]],
                overall_score=data["overall_score"],
                estimated_review_time_saved_minutes=data["estimated_review_time_saved_minutes"]
            )

    raise ValueError("Agent did not produce a structured review")
```

The FastAPI app:

```python
# app/main.py
import time
import uuid
import logging
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .agent import review_code
from .models import CodeReviewRequest, CodeReviewResponse
from .config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting agent API in {settings.environment} mode")
    yield
    logger.info("Shutting down agent API")

app = FastAPI(
    title="Code Review Agent API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else ["https://yourdomain.com"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"]
)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"[{request_id}] {request.method} {request.url.path} {response.status_code} {duration:.2f}s")
    response.headers["X-Request-ID"] = request_id
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy", "environment": settings.environment}

@app.post("/review", response_model=CodeReviewResponse)
async def code_review_endpoint(request: CodeReviewRequest):
    try:
        result = await review_code(request)
        return result
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Retry in 60 seconds.")
    except anthropic.APITimeoutError:
        raise HTTPException(status_code=504, detail="Agent timed out. Try with a smaller diff.")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Dockerizing the Agent

A production Dockerfile should be lean and use multi-stage builds when possible:

```dockerfile
# Dockerfile
FROM python:3.12-slim AS base

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Lambda adapter for AWS Lambda deployment
# Remove this line if deploying to ECS/EC2 instead
COPY infra/lambda_handler.py .

# Non-root user for security
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
```

For local development:

```yaml
# docker-compose.yml
version: "3.9"
services:
  agent:
    build: .
    ports:
      - "8080:8080"
    env_file: .env
    environment:
      - ENVIRONMENT=development
    volumes:
      - ./app:/app/app  # Hot reload in development
    command: uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## Deploying to AWS Lambda

Lambda now supports container images up to 10GB. Use the Lambda adapter to bridge FastAPI to Lambda's event format:

```python
# infra/lambda_handler.py
from mangum import Mangum
from app.main import app

handler = Mangum(app, lifespan="off")
```

Install `mangum` in requirements.txt — it translates Lambda events to ASGI format that FastAPI understands.

Deploy using the AWS CLI:

```bash
# Build and push to ECR
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/code-review-agent

aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ECR_REPO

docker build -t code-review-agent .
docker tag code-review-agent:latest $ECR_REPO:latest
docker push $ECR_REPO:latest

# Create Lambda function (first time)
aws lambda create-function \
  --function-name code-review-agent \
  --package-type Image \
  --code ImageUri=$ECR_REPO:latest \
  --role arn:aws:iam::$AWS_ACCOUNT_ID:role/lambda-execution-role \
  --memory-size 1024 \
  --timeout 120 \
  --environment Variables="{ANTHROPIC_API_KEY=$(aws secretsmanager get-secret-value --secret-id anthropic-api-key --query SecretString --output text)}"

# Subsequent deployments
aws lambda update-function-code \
  --function-name code-review-agent \
  --image-uri $ECR_REPO:latest
```

**Important**: Store your `ANTHROPIC_API_KEY` in AWS Secrets Manager, not as a plain environment variable. The Lambda IAM role needs `secretsmanager:GetSecretValue` permission.

## Managing Costs at Scale

LLM API costs are the dominant cost driver for agent deployments. Three techniques that actually move the needle:

**Input caching**: Claude's prompt caching reduces costs by up to 90% for repeated system prompts. Add the `cache_control` parameter:

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    system=[
        {
            "type": "text",
            "text": build_system_prompt(request.focus),
            "cache_control": {"type": "ephemeral"}  # Cache this block
        }
    ],
    messages=messages
)
```

**Request coalescing**: If multiple requests come in with the same diff (unlikely but possible with PR webhook retries), deduplicate them with a short-TTL cache keyed on the diff hash.

**Diff truncation**: Most LLM cost overruns come from unexpectedly large inputs. Enforce the `max_length=50_000` on your Pydantic model and return a 400 error before making any API calls for oversized inputs.

**Lambda provisioned concurrency**: Cold starts add 2-5 seconds. For latency-sensitive workflows, provision 1-2 concurrent executions — this costs ~$15/month per instance but eliminates cold starts entirely.

## Monitoring and Observability

Add structured logging that you can query in CloudWatch:

```python
import json

def log_agent_call(request_id: str, model: str, input_tokens: int,
                   output_tokens: int, duration_ms: int, status: str):
    """Emit a structured log line for cost tracking and debugging."""
    print(json.dumps({
        "event": "agent_call",
        "request_id": request_id,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": (input_tokens * 3 + output_tokens * 15) / 1_000_000,
        "duration_ms": duration_ms,
        "status": status
    }))
```

Set up CloudWatch Metric Filters on `estimated_cost_usd` and `duration_ms` to track costs and performance over time. Create an alarm if daily cost exceeds your budget threshold — this has saved me from runaway agent loops more than once.

For deeper observability, services like [Langfuse](https://langfuse.com) provide LLM-aware tracing that shows you exactly which prompts are expensive and where agents are failing.

## Load Testing Before Launch

Never push an agent API to production without a load test. Agent calls are slow and expensive — a traffic spike that would be trivial for a standard API can generate hundreds of dollars in LLM costs in minutes.

```bash
# Install k6
brew install k6

# Run a basic load test
k6 run --vus 10 --duration 30s - <<EOF
import http from 'k6/http';
import { check } from 'k6';

const TEST_DIFF = \`
diff --git a/main.py b/main.py
+++ b/main.py
+def get_user(user_id):
+    query = f"SELECT * FROM users WHERE id = {user_id}"
+    return db.execute(query)
\`;

export default function() {
    const res = http.post('http://localhost:8080/review',
        JSON.stringify({ diff: TEST_DIFF }),
        { headers: { 'Content-Type': 'application/json' } }
    );
    check(res, { 'status is 200': (r) => r.status === 200 });
}
EOF
```

Watch both the API response times and your Anthropic usage dashboard during the test. Set a hard limit on Lambda concurrency (`aws lambda put-function-concurrency`) to cap maximum simultaneous LLM calls and prevent runaway costs.

Production AI agent deployments are substantially more complex than standard web APIs — primarily because of latency, cost, and the non-deterministic nature of LLM outputs. Start simple, instrument everything from day one, and add complexity only when you have data showing you need it.
