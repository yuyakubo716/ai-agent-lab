---
title: "How to Build a Multi-Agent System with Claude: A Step-by-Step Guide"
date: 2026-03-05
draft: false
tags: ["multi-agent", "Claude", "AI agents", "automation", "tutorial"]
categories: ["tutorials"]
description: "Learn how to build a production-ready multi-agent system using Claude. Step-by-step guide covering architecture, agent roles, communication protocols, and real-world implementation."
keywords: ["build multi-agent system Claude", "Claude API multi-agent", "multi-agent orchestration tutorial", "AI agent coordination", "Claude Code agents"]
ShowToc: true
TocOpen: false
---

*This post contains affiliate links. I may earn a commission at no extra cost to you.*

If you have ever watched a single AI assistant struggle with a complex, multi-step project—losing context, making contradictory decisions, or simply hitting token limits—you already understand why multi-agent systems matter. In 2026, the question is no longer "should I use AI?" but "how do I make AI work reliably at scale?"

This guide walks you through building a production-ready multi-agent system using Claude. Not theory. Not toy examples. A real architecture that you can run today, with actual code extracted from a live system.

---

## What Is a Multi-Agent System and Why Does It Matter in 2026

A multi-agent system is a network of autonomous AI instances—each with a defined role, persistent state, and communication channel—that collaborate to complete tasks no single agent could handle efficiently alone.

Think of it like a software development team. You would not ask one person to simultaneously gather requirements, write code, review it, and deploy it. The same logic applies to AI. Splitting responsibilities across specialized agents yields:

- **Parallel execution**: Multiple agents work on independent subtasks simultaneously.
- **Role specialization**: Each agent's instructions are tuned to its function—no context pollution.
- **Fault isolation**: A failure in one agent does not cascade to others.
- **Context window management**: Long-running projects exceed any single agent's context. Agents hand off state via files, not memory.

In the system we will build here, agents communicate through a file-based mailbox, coordinate via a command queue, and are monitored by a supervisor agent. The entire system runs inside a tmux session, making it auditable and easy to debug.

This is not an experimental pattern. In the production system that generated this article, five agents are running in parallel: a strategist (Shogun), an orchestrator (Karo), two workers (Ashigaru), and a quality-control specialist (Gunshi).

---

## Architecture Overview — Roles, Communication, and Coordination

Every multi-agent system needs a clear answer to two questions: *who does what*, and *how do agents talk to each other*?

### Agent Roles

We use a hierarchical model with four distinct roles:

| Role | Name | Responsibility |
|------|------|---------------|
| Strategist | Shogun | Receives goals from the user, breaks them into commands, delegates to orchestrator |
| Orchestrator | Karo | Decomposes commands into subtasks, assigns to workers, reviews results |
| Worker | Ashigaru | Executes concrete subtasks (writing, coding, research) |
| Analyst | Gunshi | Quality control, research, strategic planning |

Workers never communicate with the strategist directly. The chain of command is strict: Shogun → Karo → Ashigaru/Gunshi → Karo → Shogun. This prevents the sprawling, uncoordinated agent graphs that cause most multi-agent systems to fail in practice.

### Communication Model

Agents communicate exclusively through a file-based mailbox system. Each agent has an inbox YAML file at `queue/inbox/{agent_id}.yaml`. To send a message, an agent calls a shell script that appends a new entry to the target's inbox file using an exclusive file lock (`flock`).

A background watcher process monitors each inbox with `inotifywait` and delivers a short wake-up signal to the agent's terminal pane when new mail arrives.

This design has a key advantage: **messages are durable**. Even if an agent crashes or is reset, unread messages persist in the file. A recovering agent reads its inbox on startup and resumes from exactly where it left off.

### State Management

All task state lives in YAML files under `queue/tasks/` and `queue/reports/`. No state is stored in agent memory—it is always reconstructed from files. This makes the system resilient to the `/clear` commands and session resets that are a normal part of working with Claude Code.

```
project/
├── queue/
│   ├── inbox/         # Agent mailboxes (one YAML per agent)
│   ├── tasks/         # Assigned task definitions
│   └── reports/       # Completed task reports
├── context/           # Project-level shared context
├── scripts/           # Infrastructure shell scripts
└── instructions/      # Per-role system prompts
```

---

## Setting Up Your Environment (Claude API, tmux, Project Structure)

### Prerequisites

- **Claude API access** via Anthropic (or Claude Code CLI for a simpler setup)
- **tmux** — for running multiple agent sessions in one terminal
- **Python 3.10+** with `anthropic` and `pyyaml` packages
- **inotify-tools** — for the file watcher (`sudo apt install inotify-tools` on Ubuntu)

### Creating the tmux Session

The system runs in a single tmux session with one pane per agent. This gives you a live view of every agent's activity and lets you intervene manually if needed.

```bash
# Create a new tmux session named "multiagent"
tmux new-session -d -s multiagent -x 220 -y 50

# Create panes for each agent
tmux split-window -h -t multiagent:0
tmux split-window -v -t multiagent:0.0
tmux split-window -v -t multiagent:0.1

# Label each pane with its agent ID (stored as a tmux user option)
tmux set-option -t multiagent:0.0 @agent_id "karo"
tmux set-option -t multiagent:0.1 @agent_id "ashigaru1"
tmux set-option -t multiagent:0.2 @agent_id "ashigaru2"
tmux set-option -t multiagent:0.3 @agent_id "gunshi"
```

Each agent identifies itself at startup by querying this option:

```bash
tmux display-message -t "$TMUX_PANE" -p '#{@agent_id}'
```

### Initializing the Queue

```bash
mkdir -p queue/inbox queue/tasks queue/reports

# Initialize an empty inbox for each agent
for agent in shogun karo ashigaru1 ashigaru2 gunshi; do
  echo "messages: []" > "queue/inbox/${agent}.yaml"
done
```

### Installing Python Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install anthropic pyyaml
```

---

## Implementing Agent Roles — Planner, Worker, and Reviewer

The simplest way to implement agents with Claude is to give each one a dedicated system prompt loaded from a file. When an agent starts, it reads its instructions, reads its current task YAML, and begins working.

### Agent Instructions File Structure

Each agent reads its own instructions file at startup. Here is an abbreviated example for the Karo (orchestrator) role:

```markdown
# Karo — Orchestrator Instructions

You are Karo, the orchestrator of the multi-agent system.

## Responsibilities
- Decompose incoming commands into concrete subtasks
- Assign subtasks to worker agents via task YAML + inbox message
- Review worker reports and decide: OK or redo
- Update the dashboard with current task status

## Forbidden Actions
- Never contact the user (Shogun) directly — update dashboard.md only
- Never execute concrete work yourself — delegate to Ashigaru
- Never skip reviewing a worker's output before marking it complete
```

### Task YAML Format

When Karo assigns work to a worker, it writes a task YAML and sends an inbox notification. Here is a real task YAML from our system:

```yaml
task:
  task_id: subtask_192d
  parent_cmd: cmd_192
  description: |
    Create the first blog article for AI Agent Lab.
    Write a 2500-word English article in Hugo + PaperMod format
    and push to GitHub.
  target_path: "/home/ideapad/ai-agent-lab/content/posts/how-to-build-multi-agent-system-claude.md"
  project: auto-revenue
  status: assigned
  timestamp: "2026-03-05T16:55:00"
```

### Worker Execution Loop

A worker (Ashigaru) follows a straightforward loop:

1. Read task YAML from `queue/tasks/ashigaru{N}.yaml`
2. Read any project context files referenced in the task
3. Execute the work (write code, create files, run commands)
4. Write a report YAML to `queue/reports/`
5. Send an inbox message to Karo: "task complete, please review"
6. Check inbox for any new messages before going idle

The worker never polls or sleeps in a loop. It waits for a wake-up signal from the inbox watcher.

### Reviewer Pattern

The Gunshi (analyst/reviewer) receives a completed report, applies quality criteria, and responds with either `status: ok` or `status: redo` plus specific feedback. This prevents the common failure mode where agents endlessly approve each other's low-quality output.

---

## Inter-Agent Communication via File-Based Mailbox

The mailbox system is the backbone of the architecture. Let us look at the real implementation.

### inbox_write.sh

Every agent sends messages by calling this script. It uses `flock` for exclusive locking and Python's `yaml` library for atomic writes via a temp-file-and-rename pattern:

```bash
#!/usr/bin/env bash
# Usage: bash scripts/inbox_write.sh <target_agent> "<message>" <type> <from>
# Example: bash scripts/inbox_write.sh karo "subtask_192d complete" report_received ashigaru1

INBOX="queue/inbox/${TARGET}.yaml"
LOCKFILE="${INBOX}.lock"

_acquire_lock() {
    if command -v flock &>/dev/null; then
        exec 200>"$LOCKFILE"
        flock -w 5 200 || return 1
    else
        # macOS fallback: mkdir lock
        local i=0
        while ! mkdir "${LOCKFILE}.d" 2>/dev/null; do
            sleep 0.1
            i=$((i + 1))
            [ $i -ge 50 ] && return 1
        done
    fi
}
```

After acquiring the lock, the script loads the YAML, appends a new message with a unique ID and timestamp, trims the inbox to 50 messages (oldest read messages first), and atomically writes back using `os.replace()`.

### inbox_watcher.sh

Each agent has a dedicated watcher process that monitors its inbox file:

```bash
# Watch for file modification events (inotifywait is event-driven, not polling)
inotifywait -m -e modify,close_write,moved_to "$INBOX" 2>/dev/null |
while read -r _dir _event _file; do
    unread_count=$(count_unread_messages "$INBOX")
    if [ "$unread_count" -gt 0 ]; then
        # Send a short wake-up signal to the agent's tmux pane
        tmux send-keys -t "$PANE_TARGET" "inbox${unread_count}" ""
        sleep 0.3
        tmux send-keys -t "$PANE_TARGET" "" Enter
    fi
done
```

The wake-up signal is intentionally minimal: just `inbox3` (meaning "3 unread messages"). The full message content never travels through tmux—it stays in the file. The agent reads its own inbox when it wakes up.

### Escalation Protocol

If an agent does not process its inbox within a set time window, the watcher escalates:

| Elapsed | Action |
|---------|--------|
| 0–2 min | Standard tmux nudge |
| 2–4 min | Send Escape × 2 + nudge (cursor bug workaround) |
| 4 min+ | Send `/clear` to force a session reset (max once per 5 min) |

After a `/clear`, the agent re-reads its task YAML and picks up exactly where it left off. This self-healing loop means the system can run unattended overnight without human intervention.

---

## Running Your First Multi-Agent Workflow End to End

Here is the full flow for a simple task: "Write a blog article."

### Step 1: Submit a Command to Shogun

In the Shogun pane, you type your goal directly:

```
Write an article titled "How to Build a Multi-Agent System with Claude"
```

Shogun turns this into a structured command and writes it to `queue/commands/cmd_192.yaml`, then tells Karo to execute it.

### Step 2: Karo Decomposes the Command

Karo reads the command, plans the subtasks, writes task YAMLs, and assigns them:

```bash
bash scripts/inbox_write.sh ashigaru1 \
  "Read task YAML and start work. subtask_192d: blog article." \
  task_assigned karo
```

### Step 3: Ashigaru Executes

Ashigaru1 receives the inbox wake-up, reads `queue/tasks/ashigaru1.yaml`, writes the article to the target path, runs `git push`, and writes a report:

```yaml
# queue/reports/ashigaru1_report.yaml
report:
  task_id: subtask_192d
  agent: ashigaru1
  status: completed
  word_count: 2587
  file_created: content/posts/how-to-build-multi-agent-system-claude.md
  git_push: success
  timestamp: "2026-03-05T17:15:00"
```

Then Ashigaru1 notifies Karo:

```bash
bash scripts/inbox_write.sh karo \
  "subtask_192d complete. Report written. Please review." \
  report_received ashigaru1
```

### Step 4: Karo Reviews

Karo reads the report, spot-checks the output, and either approves or issues a redo with specific feedback. In our system, Gunshi handles quality control before Karo's final approval—adding an extra layer without slowing throughput.

### Step 5: Shogun Gets the Summary

Karo never sends messages directly to Shogun (to avoid interrupting the user's flow). Instead, Karo updates `dashboard.md` with current status. Shogun checks the dashboard when ready. Items requiring a human decision are flagged in a dedicated section.

---

## Performance Tips and Common Pitfalls

### Tip 1: Keep Instructions Files Short

Every agent reads its instructions file at startup. Long instruction files waste tokens on every session reset. Our Karo instructions file is under 300 lines. Ashigaru instructions are under 150. Cut anything that can be inferred from context.

### Tip 2: Never Store State in Agent Memory

The most common failure mode in multi-agent systems is relying on in-context state. After a session reset or `/clear`, that state is gone. Store everything that matters in files. Read from files. Write to files. Treat agent memory as a cache, not a database.

### Tip 3: Use Strict Role Boundaries

If your orchestrator sometimes writes code and your workers sometimes make architectural decisions, you have a coordination problem waiting to happen. Define explicit forbidden actions for each role and enforce them in the instructions file. In our system, workers are forbidden from contacting Shogun directly—a rule that has prevented dozens of accidental interruptions.

### Tip 4: Build in Idempotency

Tasks should be safe to run twice. Workers should check if their output already exists before creating it. Git operations should be idempotent (`git push` on an already-pushed branch is a no-op). This matters because the escalation protocol may trigger a task re-run after a `/clear`.

```python
import os

target = "content/posts/my-article.md"
if os.path.exists(target):
    # Verify it meets acceptance criteria before re-creating
    with open(target) as f:
        content = f.read()
    word_count = len(content.split())
    if word_count >= 2500:
        print(f"Article already exists ({word_count} words). Skipping creation.")
        # Jump straight to git push and report writing
```

### Tip 5: Plan for Batch Token Costs

In a system with five agents running concurrently, your token costs multiply quickly. We follow a batch-processing protocol: validate the approach on the first item before running the full batch. One bad prompt repeated across 30 tasks wastes 30× the tokens. A two-minute review after batch 1 saves hours of redo work.

### Common Pitfall: The Polling Loop

Do not implement agents that poll their inbox in a sleep loop:

```bash
# BAD — burns compute and creates noisy logs
while true; do
    check_inbox
    sleep 5
done
```

Use `inotifywait` for event-driven wake-ups. Your system will be quieter, cheaper to run, and easier to debug.

### Common Pitfall: Skipping Quality Control

It is tempting to skip the reviewer role when you are moving fast. Resist this. In our production system, roughly 20% of tasks require at least one redo. Without automated review, those errors compound across hundreds of articles. A dedicated reviewer agent catches them before they ship.

---

Building a multi-agent system with Claude is less about prompt engineering and more about system design. Define clear roles, use durable file-based communication, and build in self-healing from the start. The result is a system that can run complex, multi-step workflows autonomously—and keep running even when individual agents fail.

The architecture described here is running in production as you read this, generating content, running quality checks, and pushing to GitHub—all without human intervention. Start with the mailbox system and the two-role structure (planner + worker), then add specialization as your workload grows.

If you want to explore the source code for the system described in this article, the full implementation is available as an open-source project. The inbox scripts, watcher logic, agent instructions templates, and task YAML schemas are all there. Clone it, run `first_setup.sh`, and you will have a working multi-agent environment in under ten minutes.

The key insight that took the longest to internalize: a multi-agent system is a distributed system with all the challenges that entails—partial failure, message ordering, idempotency, and observability. Build those guarantees in from day one, and the AI coordination layer becomes straightforward. Bolt them on later, and you will spend more time debugging the infrastructure than doing the actual work.
