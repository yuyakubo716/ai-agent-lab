# AI Agent Lab

Practical guides for building AI agents, multi-agent systems, and automation workflows.

**Blog**: https://ai-agent-lab.pages.dev/

## Topics
- Multi-agent system architecture and implementation
- AI agent framework comparisons and tutorials
- Automation workflow design patterns

## Local Development

Prerequisites: [Hugo extended](https://gohugo.io/installation/) v0.146.0+

```bash
git clone --recurse-submodules https://github.com/yuyakubo716/ai-agent-lab.git
cd ai-agent-lab
hugo server -D
# Open http://localhost:1313/ai-agent-lab/
```

## Adding a New Article

1. Create a new markdown file in `content/posts/`:
   ```bash
   hugo new posts/your-article-slug.md
   ```
2. Edit the file with your content (set `draft: false` when ready)
3. Commit and push to `main` — GitHub Actions deploys to Cloudflare Pages automatically

## License

Content © AI Agent Lab. All rights reserved.
