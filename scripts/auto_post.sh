#!/usr/bin/env bash
# =============================================================================
# auto_post.sh - Automated Blog Post Generation Pipeline for AI Agent Lab
# =============================================================================
# Usage:
#   bash scripts/auto_post.sh              # Auto-select next pending topic
#   bash scripts/auto_post.sh --topic T003 # Run specific topic by ID
#
# Cron example (daily at 9 AM):
#   0 9 * * * cd /home/ideapad/ai-agent-lab && bash scripts/auto_post.sh >> logs/auto_post.log 2>&1
#
# Pipeline:
#   1. Select next pending topic from content/topics.yaml
#   2. Generate article using Claude CLI
#   3. Quality check (word count, frontmatter, title uniqueness)
#   4. git add -> commit -> push
#   5. Mark topic as published in topics.yaml
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOPICS_FILE="$REPO_DIR/content/topics.yaml"
POSTS_DIR="$REPO_DIR/content/posts"
LOGS_DIR="$REPO_DIR/logs"
MIN_WORDS=1500
LOG_FILE="$LOGS_DIR/auto_post.log"

mkdir -p "$LOGS_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
FORCED_TOPIC_ID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --topic)
            FORCED_TOPIC_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1: Select topic using Python YAML parser
# ---------------------------------------------------------------------------
log "=== Auto Post Pipeline Start ==="

TOPIC_INFO=$(python3 - <<PYEOF
import yaml, sys

topics_file = "$TOPICS_FILE"
forced_id = "$FORCED_TOPIC_ID"

with open(topics_file) as f:
    data = yaml.safe_load(f)

topics = data.get('topics', [])

selected = None
for t in topics:
    if forced_id:
        if t['id'] == forced_id:
            selected = t
            break
    else:
        if t.get('status') == 'pending':
            selected = t
            break

if not selected:
    print("NO_PENDING_TOPIC")
    sys.exit(0)

lsi = ', '.join(selected.get('lsi_keywords', []))
print(selected['id'])
print(selected['title'])
print(selected['slug'])
print(selected['main_keyword'])
print(lsi)
print(selected['category'])
print(selected['audience'])
PYEOF
)

if [[ "$TOPIC_INFO" == "NO_PENDING_TOPIC" ]]; then
    log "No pending topics found. All topics have been published."
    exit 0
fi

TOPIC_ID=$(echo "$TOPIC_INFO" | sed -n '1p')
TOPIC_TITLE=$(echo "$TOPIC_INFO" | sed -n '2p')
TOPIC_SLUG=$(echo "$TOPIC_INFO" | sed -n '3p')
MAIN_KW=$(echo "$TOPIC_INFO" | sed -n '4p')
LSI_KW=$(echo "$TOPIC_INFO" | sed -n '5p')
CATEGORY=$(echo "$TOPIC_INFO" | sed -n '6p')
AUDIENCE=$(echo "$TOPIC_INFO" | sed -n '7p')

log "Selected topic: [$TOPIC_ID] $TOPIC_TITLE"
log "Category: $CATEGORY | Audience: $AUDIENCE"

TODAY=$(date '+%Y-%m-%d')
OUTPUT_FILE="$POSTS_DIR/${TOPIC_SLUG}.md"

# Check for title uniqueness
if ls "$POSTS_DIR"/*.md 2>/dev/null | xargs grep -l "title:.*${TOPIC_TITLE:0:30}" 2>/dev/null | grep -q .; then
    log "ERROR: Duplicate title detected for '$TOPIC_TITLE'. Skipping."
    exit 1
fi

if [[ -f "$OUTPUT_FILE" ]]; then
    log "ERROR: File already exists: $OUTPUT_FILE. Skipping."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Generate article using Claude CLI
# ---------------------------------------------------------------------------
log "Generating article..."

ARTICLE_PROMPT="Write a complete, high-quality blog post for AI Agent Lab (https://ai-agent-lab.github.io).

## Article Requirements

**Title**: $TOPIC_TITLE
**Category**: $CATEGORY
**Target Audience**: $AUDIENCE
**Main Keyword**: $MAIN_KW
**LSI Keywords**: $LSI_KW

## Format

Output ONLY the raw Markdown file (no explanation, no code block wrapper). Start with YAML frontmatter:

---
title: \"$TOPIC_TITLE\"
date: $TODAY
description: \"<meta description, max 155 chars, include CTA>\"
tags: [\"<tag1>\", \"<tag2>\", \"<tag3>\", \"<tag4>\"]
categories: [\"$CATEGORY\"]
keywords: [\"$MAIN_KW\", \"<lsi1>\", \"<lsi2>\"]
draft: false
---

Then the article body.

## Content Standards

1. Start with: *This post contains affiliate links. I may earn a commission at no extra cost to you.*
2. Minimum 1,500 words (aim for 2,000+)
3. 5-7 H2 headings, each containing a relevant keyword
4. Include at least one complete, working code example (Python preferred)
5. Include at least one H2 section titled \"Common Pitfalls\" or \"Troubleshooting\" with real-world gotchas
6. Include one comparison table or numbered list showing concrete benchmarks or step-by-step instructions
7. External link to one authoritative source (official docs, research paper)
8. End with a \"Conclusion\" H2 summarizing key takeaways and a CTA
9. Write from a practitioner's perspective - include real implementation details, not just theory
10. DO NOT fabricate benchmarks or statistics - use realistic estimates and clearly label them as such"

# When running inside a Claude Code session (CLAUDECODE is set), unset it to
# allow the nested claude -p call. The -p (print/non-interactive) mode is safe
# as it makes a single API call and exits without sharing interactive resources.
if [[ -n "${CLAUDECODE:-}" ]]; then
    env -u CLAUDECODE claude -p "$ARTICLE_PROMPT" --model claude-sonnet-4-6 > "$OUTPUT_FILE"
else
    claude -p "$ARTICLE_PROMPT" --model claude-sonnet-4-6 > "$OUTPUT_FILE"
fi

log "Article generated: $OUTPUT_FILE"

# ---------------------------------------------------------------------------
# Step 3: Quality check
# ---------------------------------------------------------------------------
log "Running quality checks..."

WORD_COUNT=$(wc -w < "$OUTPUT_FILE")
log "Word count: $WORD_COUNT (minimum: $MIN_WORDS)"

if [[ "$WORD_COUNT" -lt "$MIN_WORDS" ]]; then
    log "ERROR: Word count $WORD_COUNT is below minimum $MIN_WORDS. Removing file."
    rm -f "$OUTPUT_FILE"
    exit 1
fi

# Check required frontmatter fields
for field in "title:" "date:" "description:" "tags:" "categories:" "draft:"; do
    if ! grep -q "^${field}" "$OUTPUT_FILE"; then
        log "ERROR: Missing frontmatter field '${field}'. Removing file."
        rm -f "$OUTPUT_FILE"
        exit 1
    fi
done

# Check affiliate disclosure
if ! grep -q "affiliate" "$OUTPUT_FILE"; then
    log "WARNING: Affiliate disclosure may be missing from article."
fi

log "Quality checks PASSED (words: $WORD_COUNT)"

# ---------------------------------------------------------------------------
# Step 4: Mark topic as published in topics.yaml
# ---------------------------------------------------------------------------
python3 - <<PYEOF
import yaml

topics_file = "$TOPICS_FILE"
topic_id = "$TOPIC_ID"
today = "$TODAY"

with open(topics_file) as f:
    data = yaml.safe_load(f)

for t in data['topics']:
    if t['id'] == topic_id:
        t['status'] = 'published'
        t['published_at'] = today
        t['output_file'] = "${TOPIC_SLUG}.md"
        break

with open(topics_file, 'w') as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f"Marked topic {topic_id} as published")
PYEOF

log "Topic marked as published in topics.yaml"

# ---------------------------------------------------------------------------
# Step 5: git add -> commit -> push
# ---------------------------------------------------------------------------
log "Committing and pushing to git..."

cd "$REPO_DIR"
git add "content/posts/${TOPIC_SLUG}.md" "content/topics.yaml"
git commit -m "feat(content): add '${TOPIC_TITLE}'

Auto-generated via auto_post.sh pipeline.
Topic ID: ${TOPIC_ID} | Category: ${CATEGORY} | Words: ~${WORD_COUNT}"

git push

log "=== Pipeline Complete: [$TOPIC_ID] $TOPIC_TITLE ==="
log "Published to: content/posts/${TOPIC_SLUG}.md"
