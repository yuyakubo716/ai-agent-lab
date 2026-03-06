---
title: "Building a Web Scraping Agent with Claude and Playwright"
date: 2026-03-06
description: "Build an AI-powered web scraping agent using Claude and Playwright. Extract structured data from any website with natural language instructions."
tags: ["web scraping", "Playwright", "Claude API", "browser automation", "Python"]
categories: ["Tutorials"]
keywords: ["web scraping AI agent", "Playwright", "Claude API", "browser automation", "data extraction agent"]
draft: false
---

*This post contains affiliate links. I may earn a commission at no extra cost to you.*

Web scraping has always been fragile—one DOM change breaks your carefully crafted CSS selectors. AI-powered scraping agents flip this model: instead of brittle selectors, you describe *what* you want in plain English, and the agent figures out how to get it.

This tutorial builds a web scraping agent that combines Claude's reasoning capabilities with Playwright's browser automation. The result is a scraper that adapts to page structure changes and handles JavaScript-heavy sites that traditional scrapers can't touch.

## What We're Building

A Python agent that:
1. Accepts a scraping task in natural language ("Extract all job titles and salaries from this page")
2. Uses Claude to generate a Playwright scraping strategy
3. Executes the strategy in a real browser
4. Returns clean, structured JSON

No hardcoded selectors. No fragile XPath expressions.

## Prerequisites and Setup

```bash
pip install anthropic playwright python-dotenv
playwright install chromium
```

```
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

## The Core Architecture

The agent runs a simple tool-use loop:

1. Claude receives the page HTML + task description
2. Claude calls a `execute_javascript` tool with a JS snippet to extract data
3. Playwright runs the JS in the browser and returns results
4. Claude either returns final data or requests another extraction

This loop runs until Claude signals it has enough data or hits a retry limit.

## Step 1: Browser Manager

First, a thin wrapper around Playwright:

```python
import asyncio
from playwright.async_api import async_playwright, Page, Browser

class BrowserManager:
    def __init__(self):
        self._playwright = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    async def start(self, headless: bool = True) -> None:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=headless)
        self._page = await self._browser.new_page()
        # Mask automation signals
        await self._page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

    async def navigate(self, url: str) -> str:
        """Navigate to URL and return page HTML."""
        await self._page.goto(url, wait_until="networkidle", timeout=30_000)
        return await self._page.content()

    async def run_js(self, script: str) -> str:
        """Execute JavaScript and return JSON-serializable result."""
        try:
            result = await self._page.evaluate(script)
            return str(result)
        except Exception as e:
            return f"ERROR: {e}"

    async def screenshot(self, path: str) -> None:
        await self._page.screenshot(path=path, full_page=True)

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
```

## Step 2: HTML Preprocessing

Raw HTML is too verbose for the model's context window. We strip noise before sending it:

```python
import re
from html.parser import HTMLParser

class HTMLCleaner(HTMLParser):
    """Strip scripts, styles, and non-visible content."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}

    def __init__(self):
        super().__init__()
        self._skip = False
        self._depth = 0
        self.cleaned: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip = True
            self._depth += 1
        elif not self._skip:
            attr_str = " ".join(
                f'{k}="{v}"' for k, v in attrs
                if k in ("class", "id", "data-testid", "aria-label", "href")
            )
            self.cleaned.append(f"<{tag}{' ' + attr_str if attr_str else ''}>")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self._depth -= 1
            if self._depth == 0:
                self._skip = False
        elif not self._skip:
            self.cleaned.append(f"</{tag}>")

    def handle_data(self, data):
        text = data.strip()
        if text and not self._skip:
            self.cleaned.append(text)

def clean_html(raw_html: str, max_chars: int = 40_000) -> str:
    cleaner = HTMLCleaner()
    cleaner.feed(raw_html)
    result = " ".join(cleaner.cleaned)
    # Collapse whitespace
    result = re.sub(r"\s+", " ", result)
    return result[:max_chars]
```

This typically reduces a 200KB HTML page to under 20KB—a 10x reduction that keeps scraping costs reasonable.

## Step 3: The Scraping Agent

Now the core agent that orchestrates Claude and Playwright:

```python
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

TOOLS = [
    {
        "name": "execute_javascript",
        "description": (
            "Execute JavaScript in the browser to extract data from the current page. "
            "The script should return a JSON-serializable value. "
            "Use document.querySelectorAll, textContent, getAttribute, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "JavaScript code to execute. Must return a value.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why you are running this script.",
                },
            },
            "required": ["script", "reason"],
        },
    }
]

class ScrapingAgent:
    def __init__(self):
        self._client = anthropic.Anthropic()
        self._browser = BrowserManager()

    async def scrape(self, url: str, task: str, max_iterations: int = 5) -> dict:
        await self._browser.start()
        try:
            raw_html = await self._browser.navigate(url)
            html = clean_html(raw_html)

            system = (
                "You are a web scraping agent. You receive a page's HTML and a scraping task. "
                "Use the execute_javascript tool to extract the requested data. "
                "Return ONLY valid JSON when you have the final answer—no explanation."
            )

            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Task: {task}\n\n"
                        f"Page URL: {url}\n\n"
                        f"Page HTML (truncated):\n{html}"
                    ),
                }
            ]

            for _ in range(max_iterations):
                response = self._client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    system=system,
                    tools=TOOLS,
                    messages=messages,
                )

                # Check for tool use
                tool_uses = [b for b in response.content if b.type == "tool_use"]
                if not tool_uses:
                    # Final text response
                    final_text = next(
                        (b.text for b in response.content if b.type == "text"), ""
                    )
                    try:
                        return json.loads(final_text)
                    except json.JSONDecodeError:
                        return {"raw": final_text}

                # Execute all tool calls
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for tool_use in tool_uses:
                    js_result = await self._browser.run_js(tool_use.input["script"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": js_result,
                    })
                messages.append({"role": "user", "content": tool_results})

            return {"error": "Max iterations reached"}
        finally:
            await self._browser.close()
```

## Step 4: Running the Agent

```python
async def main():
    agent = ScrapingAgent()

    # Example: scrape Hacker News front page
    result = await agent.scrape(
        url="https://news.ycombinator.com",
        task="Extract the top 10 stories: title, points, and number of comments. Return as JSON array.",
    )
    print(json.dumps(result, indent=2))

asyncio.run(main())
```

Sample output:
```json
[
  {
    "title": "Show HN: I built a local-first SQLite sync engine",
    "points": 342,
    "comments": 87
  },
  {
    "title": "The Unreasonable Effectiveness of Just Asking",
    "points": 289,
    "comments": 134
  }
]
```

## Handling Dynamic Content and Pagination

Many sites load content lazily. The agent can handle this with scroll instructions:

```python
# In your task description, mention the need to scroll:
result = await agent.scrape(
    url="https://example.com/products",
    task=(
        "Scroll to the bottom of the page to load all products, "
        "then extract name and price for every product card. "
        "Return as JSON array."
    ),
)
```

Claude will generate JavaScript that scrolls the page before extracting:

```javascript
// Claude-generated extraction script
await new Promise(resolve => {
  window.scrollTo(0, document.body.scrollHeight);
  setTimeout(resolve, 2000);
});
return Array.from(document.querySelectorAll('.product-card')).map(el => ({
  name: el.querySelector('.product-name')?.textContent?.trim(),
  price: el.querySelector('.price')?.textContent?.trim(),
}));
```

## Rate Limiting and Politeness

Scraping without rate limiting is rude and will get you blocked. Add delays:

```python
import random

class PoliteBrowserManager(BrowserManager):
    async def navigate(self, url: str, min_delay: float = 1.0, max_delay: float = 3.0) -> str:
        await asyncio.sleep(random.uniform(min_delay, max_delay))
        return await super().navigate(url)
```

Always check `robots.txt` before scraping. For sites that explicitly prohibit scraping, consider using their official API instead.

## Dealing with Anti-Bot Measures

Some sites detect headless browsers. Options in increasing order of complexity:

1. **User agent spoofing** (already included in our `add_init_script` call)
2. **Playwright stealth** via the `playwright-stealth` package
3. **Residential proxies** for geo-distributed requests
4. **Browserless.io** — managed headless browser service that handles stealth automatically

For legitimate scraping at scale, [Browserless.io](https://www.browserless.io) handles the infrastructure complexity so you can focus on extraction logic.

## Lessons From Production

After running this pattern on dozens of sites, the main lessons:

**Claude chooses selectors better than humans for unfamiliar HTML.** Point it at a page structure you've never seen and it correctly identifies the data-bearing elements using semantic HTML attributes.

**The HTML truncation limit matters.** Pages over 40K characters after cleaning need either smarter truncation (keep the relevant section) or a two-pass approach: first ask Claude which CSS selector to scroll to, then extract from that subtree.

**Retries should back off.** If `execute_javascript` returns an error three times in a row, the page probably changed structure. Log the failure and move on rather than hammering the same site.

For the next step in building autonomous agents, see our guide on [building a RAG agent from scratch](/posts/building-rag-agent-from-scratch/) to add document knowledge on top of your scraped data.

## Conclusion

Combining Claude's natural language understanding with Playwright's browser automation creates a scraping agent that is dramatically more resilient than selector-based scrapers. The key insight is that you are no longer writing extraction rules—you are writing extraction *goals*, and the agent figures out the implementation.

The code in this guide handles 80% of real-world scraping tasks. The remaining 20%—CAPTCHAs, login walls, heavily obfuscated sites—require additional tooling, but the architecture scales to cover those cases too.
